"""Stage E0 regime diagnosis: default-governor baseline and idle-power ladder.

Step 1 (``XPYD_STEP1_ENERGY_DECOMPOSITION.md``) showed that 96-99% of P-pool
energy in all accepted XPYD runs is the idle power of clock-locked L40S GPUs.
E0 answers the two questions that the locked-clock data cannot:

E0-a  What is each GPU's idle power, with vLLM running and the model
      resident, at every locked clock and under the default driver governor
      (``nvidia-smi -rgc -rmc``)?
E0-b  On the unchanged C1 mixed trace and 600/65 ms contract, how do
      STATIC-HIGH, STATIC-TUNED (P=MID/D=HIGH), P-default/D-HIGH and
      all-default compare in energy and SLO attainment?

No feedback decision, adaptive routing, probing, predictor or oracle is used.
Every fixed point is cold-started at locked HIGH, so the comparison is matched.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import statistics
import time
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

from xpyd.phase3b_energy import _launch_monitor, _read_jsonl, _stop_monitors, _wait_capabilities
from xpyd.phase3c_substrate import Phase3CSubstrateHarness, _write_json
from xpyd.phase3d_control import (
    EndpointClockCapability,
    GPUReadback,
    NvidiaSmiClockBackend,
    Phase3DError,
)
from xpyd.stagec0_capacity import percentile
from xpyd.stagec1_headroom import StageC1HeadroomHarness, _write_csv
from xpyd.stagec1_stationary import POOLS, StageC1Error, _trace_signature, load_configs
from xpyd.workload_harness import build_trace

DEFAULT = "DEFAULT"
LEVELS = ("LOW", "MID", "HIGH", DEFAULT)
# Sentinel target meaning "driver default governor, no clock lock".
DEFAULT_TARGET_MHZ = 0


class StageE0Error(StageC1Error):
    pass


class GovernorClockBackend(NvidiaSmiClockBackend):
    """Adds default-governor reset and memory-clock re-lock to the C1 backend."""

    _memory_re = re.compile(r"^MEMORY=\s*(\d+)\s*$", re.MULTILINE)

    def reset_clocks(self, capability: EndpointClockCapability) -> GPUReadback:
        command = (
            f"{self.sudo} {self.nvidia_smi} -i {capability.gpu_id} -rgc\n"
            f"{self.sudo} {self.nvidia_smi} -i {capability.gpu_id} -rmc"
        )
        self._run(capability.node, "set -euo pipefail\n" + command)
        return self.read(capability.endpoint_id, capability.node, capability.gpu_id)

    def lock_memory_clock(self, capability: EndpointClockCapability, memory_mhz: int) -> int:
        query = (
            f"{self.nvidia_smi} -i {capability.gpu_id} "
            "--query-gpu=clocks.current.memory --format=csv,noheader,nounits"
        )
        script = (
            "set -euo pipefail\n"
            f"{self.sudo} {self.nvidia_smi} -i {capability.gpu_id} -lmc {int(memory_mhz)},{int(memory_mhz)}\n"
            "for attempt in 1 2 3 4 5 6 7 8 9 10; do\n"
            f"  actual=\"$({query} | tr -d '[:space:]')\"\n"
            "  printf 'MEMORY=%s\\n' \"${actual}\"\n"
            f"  if [ \"${{actual}}\" = \"{int(memory_mhz)}\" ]; then break; fi\n"
            "  sleep 0.2\n"
            "done\n"
        )
        matches = self._memory_re.findall(self._run(capability.node, script))
        if not matches or int(matches[-1]) != int(memory_mhz):
            raise Phase3DError(
                "memory clock re-lock readback mismatch on %s: %r" % (capability.endpoint_id, matches)
            )
        return int(matches[-1])


def default_governor_clock_audit(
    records: Sequence[Mapping[str, Any]], start_wall_s: float, end_wall_s: float,
) -> dict[str, Any]:
    """Describe, rather than gate, clocks chosen by the driver governor."""
    selected = [
        item for item in records
        if item.get("status") == "success"
        and isinstance(item.get("actual_start_wall_s"), (int, float))
        and start_wall_s <= float(item["actual_start_wall_s"]) <= end_wall_s
    ]

    def describe(field: str) -> dict[str, Any]:
        values = [float(item[field]) for item in selected if item.get(field) is not None]
        return {
            "target_mhz": None,
            "sample_count": len(values),
            "minimum_mhz": min(values) if values else None,
            "maximum_mhz": max(values) if values else None,
            "mean_mhz": statistics.fmean(values) if values else None,
            "median_mhz": statistics.median(values) if values else None,
            "distinct_values": len(set(values)),
        }

    graphics = describe("graphics_clock_mhz")
    return {
        "valid": bool(selected) and graphics["sample_count"] > 0,
        "mode": DEFAULT,
        "required_match_fraction": None,
        "energy_sample_count": len(selected),
        "graphics": graphics,
        "memory": describe("memory_clock_mhz"),
    }


class E0WindowHarness(Phase3CSubstrateHarness):
    """Phase 3C accounting window whose DEFAULT endpoints are unlocked."""

    def __init__(
        self, config: Mapping[str, Any], run_id: str,
        trace_rows: Sequence[Mapping[str, Any]], default_endpoints: set[str],
    ) -> None:
        super().__init__(config, run_id=run_id, trace_rows=trace_rows)
        self.default_endpoints = set(default_endpoints)

    def _metadata_overrides(self) -> dict[str, Any]:
        return {
            "phase": "Stage_E0_regime_diagnosis",
            "clock_modes": {
                endpoint_id: DEFAULT if endpoint_id in self.default_endpoints else "LOCKED"
                for endpoint_id in self.config["fixed_clocks"]
            },
            "clock_semantics": (
                "DEFAULT endpoints run under the driver governor after -rgc/-rmc; "
                "their fixed_clocks entry is nominal only and is not audited as a target"
            ),
        }

    def _clock_audit_for_endpoint(
        self, endpoint_id: str, records: Sequence[Mapping[str, Any]],
        start_wall_s: float, end_wall_s: float,
    ) -> dict[str, Any]:
        if endpoint_id in self.default_endpoints:
            return default_governor_clock_audit(records, start_wall_s, end_wall_s)
        return super()._clock_audit_for_endpoint(endpoint_id, records, start_wall_s, end_wall_s)

    def _clock_hard_gate_name(self) -> str:
        return "fixed_or_default_governor_clocks"

    def _claim_boundary(self) -> str:
        return "E0 fixed-point regime diagnosis; no controller, routing, or policy claim"


def idle_window_summary(
    records: Sequence[Mapping[str, Any]], start_wall_s: float, end_wall_s: float,
    period_s: float, min_coverage: float,
) -> dict[str, Any]:
    selected = sorted(
        (item for item in records
         if item.get("status") == "success"
         and item.get("total_energy_mj") is not None
         and start_wall_s <= float(item["actual_start_wall_s"]) <= end_wall_s),
        key=lambda item: float(item["actual_start_wall_s"]),
    )
    duration = end_wall_s - start_wall_s
    coverage = len(selected) * period_s / duration if duration > 0 else 0.0
    row: dict[str, Any] = {
        "sample_count": len(selected), "requested_duration_s": duration,
        "coverage_ratio": coverage, "valid": False,
    }
    if len(selected) < 2:
        return row
    first, last = selected[0], selected[-1]
    measured = float(last["actual_start_wall_s"]) - float(first["actual_start_wall_s"])
    energy = (float(last["total_energy_mj"]) - float(first["total_energy_mj"])) / 1000.0
    graphics = [float(item["graphics_clock_mhz"]) for item in selected if item.get("graphics_clock_mhz") is not None]
    memory = [float(item["memory_clock_mhz"]) for item in selected if item.get("memory_clock_mhz") is not None]
    power = [float(item["power_w"]) for item in selected if item.get("power_w") is not None]
    utilization = [float(item["gpu_utilization_pct"]) for item in selected if item.get("gpu_utilization_pct") is not None]
    reasons = sorted({str(reason) for item in selected for reason in item.get("clock_throttle_reasons", [])})
    row.update({
        "measured_duration_s": measured,
        "energy_j": energy,
        "idle_power_w": energy / measured if measured > 0 else None,
        "mean_instant_power_w": statistics.fmean(power) if power else None,
        "graphics_min_mhz": min(graphics) if graphics else None,
        "graphics_median_mhz": statistics.median(graphics) if graphics else None,
        "graphics_max_mhz": max(graphics) if graphics else None,
        "memory_min_mhz": min(memory) if memory else None,
        "memory_median_mhz": statistics.median(memory) if memory else None,
        "memory_max_mhz": max(memory) if memory else None,
        "max_gpu_utilization_pct": max(utilization) if utilization else None,
        "throttle_reasons": ",".join(reasons),
        "valid": bool(measured > 0 and coverage >= min_coverage and (max(utilization) if utilization else 0.0) == 0.0),
    })
    return row


class StageE0RegimeHarness(StageC1HeadroomHarness):
    """Idle ladder plus matched fixed-point/default-governor serving windows."""

    def __init__(
        self, runtime: Mapping[str, Any], experiment: Mapping[str, Any],
        run_id: str | None = None, backend: GovernorClockBackend | None = None,
        sleep: Any = time.sleep, wall_clock: Any = time.time,
    ) -> None:
        super().__init__(runtime, experiment, run_id=run_id, backend=backend or GovernorClockBackend())
        self.e0 = dict(self.experiment["stage_e0"])
        # The inherited headroom reporting reads these two keys.
        self.headroom_settings = {
            "repetitions": int(self.e0["repetitions"]),
            "diagnostic_points": self.e0["points"],
        }
        self.sleep = sleep
        self.wall_clock = wall_clock
        self.default_endpoints: set[str] = set()
        self.mode_rows: list[dict[str, Any]] = []
        self.idle_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ points
    def _points(self) -> list[dict[str, Any]]:
        points = self.headroom_settings.get("diagnostic_points")
        if not isinstance(points, list) or not points:
            raise StageE0Error("stage_e0.points is required")
        seen: set[str] = set()
        result = []
        for item in points:
            point = dict(item)
            point_id = str(point.get("id", ""))
            if not point_id or point_id in seen:
                raise StageE0Error("E0 point IDs must be unique and non-empty")
            if point.get("P_level") not in LEVELS or point.get("D_level") not in LEVELS:
                raise StageE0Error("E0 levels must be one of %s" % (LEVELS,))
            seen.add(point_id)
            result.append(point)
        if result[0]["id"] != "HIGH_HIGH" or result[0]["P_level"] != "HIGH" or result[0]["D_level"] != "HIGH":
            raise StageE0Error("E0 points must start with the HIGH_HIGH baseline")
        return result

    def _level_targets(self, p_level: str, d_level: str) -> dict[str, int]:
        levels = self.settings["frequency_levels_mhz"]
        targets = {}
        for pool, level in (("P", p_level), ("D", d_level)):
            for endpoint_id in POOLS[pool]:
                targets[endpoint_id] = DEFAULT_TARGET_MHZ if level == DEFAULT else int(levels[endpoint_id][level])
        return targets

    @staticmethod
    def run_order(points: Sequence[Mapping[str, Any]], repeat: int) -> list[Mapping[str, Any]]:
        """Rotate point order per repeat to counterbalance order/thermal drift."""
        shift = (repeat - 1) % len(points)
        return list(points[shift:]) + list(points[:shift])

    # --------------------------------------------------------------- actuation
    def _apply(
        self, actuator: Any, capabilities: Mapping[str, EndpointClockCapability],
        targets: Mapping[str, int], reason: str,
    ) -> None:
        memory = {
            endpoint_id: int(self.config["fixed_clocks"][endpoint_id]["memory_mhz"])
            for endpoint_id in targets
        }
        for endpoint_id in sorted(targets):
            target = int(targets[endpoint_id])
            capability = capabilities[endpoint_id]
            started = self.wall_clock()
            if target == DEFAULT_TARGET_MHZ:
                if endpoint_id in self.default_endpoints:
                    continue
                readback = self.backend.reset_clocks(capability)
                # The actuator must not poll for a stale locked target.
                actuator.requested.pop(endpoint_id, None)
                self.default_endpoints.add(endpoint_id)
                self.mode_rows.append({
                    "timestamp_unix_s": started, "endpoint_id": endpoint_id,
                    "action": "RESET_TO_DEFAULT_GOVERNOR", "reason": reason,
                    "observed_graphics_after_mhz": readback.graphics_clock_mhz,
                    "latency_s": self.wall_clock() - started,
                })
                continue
            if endpoint_id in self.default_endpoints:
                observed_memory = self.backend.lock_memory_clock(capability, memory[endpoint_id])
                self.default_endpoints.discard(endpoint_id)
                self.mode_rows.append({
                    "timestamp_unix_s": started, "endpoint_id": endpoint_id,
                    "action": "RELOCK_MEMORY_CLOCK", "reason": reason,
                    "observed_memory_after_mhz": observed_memory,
                    "latency_s": self.wall_clock() - started,
                })
            if actuator.requested.get(endpoint_id) != target:
                row = actuator.actuate(endpoint_id, target, reason)
                row.update({"policy": "E0", "block_id": reason})
                self.action_rows.append(row)
                if row["command_status"] != "success" or not row["readback_valid"]:
                    raise StageE0Error("physical actuation failed for %s" % endpoint_id)

    # -------------------------------------------------------------------- idle
    def _measure_idle(self, label: str, state: str) -> None:
        settings = self.e0["idle_ladder"]
        idle_dir = self.run_dir / "idle" / ("%s-%s" % (label, state))
        processes: dict[str, Any] = {}
        logs = []
        stop_files = {}
        start = end = 0.0
        try:
            for value in self.config["endpoints"]:
                endpoint_id = str(value["endpoint_id"])
                endpoint_dir = idle_dir / endpoint_id
                endpoint_dir.mkdir(parents=True)
                process, log = _launch_monitor(value, endpoint_dir, self.config["sampling"])
                processes[endpoint_id] = process
                logs.append(log)
                stop_files[endpoint_id] = endpoint_dir / "stop.requested"
            _wait_capabilities(
                processes, idle_dir,
                timeout_s=float(self.config["sampling"].get("readiness_timeout_s", 120)),
            )
            self.sleep(float(settings["settle_s"]))
            start = self.wall_clock()
            self.sleep(float(settings["measure_s"]))
            end = self.wall_clock()
        finally:
            stop_errors = _stop_monitors(processes, logs, stop_files)
        if stop_errors:
            raise StageE0Error("idle monitor stop failed: %s" % "; ".join(stop_errors))
        for endpoint_id in processes:
            records = _read_jsonl(idle_dir / endpoint_id / "samples.jsonl")
            row = idle_window_summary(
                records, start, end, float(self.config["sampling"]["period_s"]),
                float(self.config["sampling"]["minimum_coverage_ratio"]),
            )
            row.update({
                "ladder": label, "state": state, "endpoint_id": endpoint_id,
                "pool": endpoint_id[0], "start_wall_s": start, "end_wall_s": end,
            })
            self.idle_rows.append(row)

    def _idle_ladder(
        self, actuator: Any, capabilities: Mapping[str, EndpointClockCapability], label: str,
    ) -> None:
        for state in self.e0["idle_ladder"]["states"]:
            if state not in LEVELS:
                raise StageE0Error("unknown idle-ladder state %s" % state)
            self._apply(actuator, capabilities, self._level_targets(state, state), "idle_%s_%s" % (label, state))
            self._measure_idle(label, state)

    # ----------------------------------------------------------------- windows
    def _run_window(
        self, window_id: str, trace_rows: Sequence[Mapping[str, Any]],
        requested: Mapping[str, int], windows_root: Path,
    ) -> Path:
        high = {key: int(value["graphics_mhz"]) for key, value in self.config["fixed_clocks"].items()}
        config = dict(self.config)
        config["fixed_clocks"] = json.loads(json.dumps(self.config["fixed_clocks"]))
        config["output_root"] = windows_root.as_posix()
        config["client"] = dict(self.config["client"], max_concurrency=int(self.settings["max_concurrency"]))
        config["server_logs"] = {}
        config["coverage_policy"] = {
            "required_endpoint_ids": ["P0", "P1", "D0", "D1"],
            "required_pairs": [["P0", "D0"], ["P0", "D1"], ["P1", "D0"], ["P1", "D1"]],
        }
        defaults = {key for key, value in requested.items() if int(value) == DEFAULT_TARGET_MHZ}
        for endpoint_id, frequency in requested.items():
            # DEFAULT endpoints keep their HIGH entry as a nominal placeholder.
            config["fixed_clocks"][endpoint_id]["graphics_mhz"] = (
                high[endpoint_id] if endpoint_id in defaults else int(frequency)
            )
        if defaults != self.default_endpoints:
            raise StageE0Error("window default-governor set disagrees with actuated state")
        E0WindowHarness(config, run_id=window_id, trace_rows=trace_rows, default_endpoints=defaults).run()
        return windows_root / window_id

    # --------------------------------------------------------------------- run
    def run(self) -> Path:
        if self.run_dir.exists():
            raise StageE0Error("E0 output already exists: %s" % self.run_dir)
        windows_root = self.run_dir / "windows"
        windows_root.mkdir(parents=True)
        capabilities = self._discover()
        discovered = {key: self._levels(value) for key, value in capabilities.items()}
        if discovered != self.settings["frequency_levels_mhz"]:
            raise StageE0Error("discovered clocks differ from validated Stage 3D-A levels")
        _write_json(self.run_dir / "capabilities.json", {
            "endpoints": {key: asdict(value) for key, value in capabilities.items()},
            "frequency_levels_mhz": discovered,
        })
        actuator = self._make_actuator(capabilities)
        high = {key: value.selected_high_mhz for key, value in capabilities.items()}
        points = self._points()
        loads = [dict(item) for item in self.settings["load_points"]]
        repetitions = int(self.e0["repetitions"])
        if repetitions < 2:
            raise StageE0Error("E0 requires at least two repeats")
        error: Optional[str] = None
        valid = False
        restoration_valid = False
        try:
            self._apply(actuator, capabilities, high, "e0_initial_high")
            if self.e0["idle_ladder"].get("before_serving", True):
                self._idle_ladder(actuator, capabilities, "pre")
            for repeat in range(1, repetitions + 1):
                for point in self.run_order(points, repeat):
                    for load_point in loads:
                        self._apply(actuator, capabilities, high, "e0_cold_start_high")
                        targets = self._level_targets(str(point["P_level"]), str(point["D_level"]))
                        self._apply(actuator, capabilities, targets, "e0_point_%s" % point["id"])
                        scenario = str(load_point["scenario"])
                        trace = build_trace(
                            self.experiment, scenario,
                            "e0-r%02d-%s-%s" % (repeat, point["id"], load_point["id"]),
                        )
                        window_id = "r%02d-%s-%s" % (repeat, point["id"], load_point["id"])
                        window_dir = self._run_window(window_id, trace, targets, windows_root)
                        row, requests = self._point_measurement(
                            point, window_dir, load_point=load_point, repeat=repeat,
                            context=SimpleNamespace(targets=targets),
                            signature=_trace_signature(trace), window_id=window_id,
                        )
                        for endpoint_id, target in targets.items():
                            row["%s_clock_mode" % endpoint_id] = DEFAULT if target == DEFAULT_TARGET_MHZ else "LOCKED"
                        self.headroom_rows.append(row)
                        self.request_rows.extend(requests)
            if self.e0["idle_ladder"].get("after_serving", True):
                self._idle_ladder(actuator, capabilities, "post")
            valid = True
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
            raise
        finally:
            try:
                self._apply(actuator, capabilities, high, "e0_final_restore_high")
                restoration_valid = not self.default_endpoints and all(
                    actuator.read(key).graphics_clock_mhz == value for key, value in high.items()
                )
            finally:
                if self.run_dir.exists():
                    self._write_reports(valid, restoration_valid, error)
        print(self.run_dir.as_posix())
        return self.run_dir

    # ----------------------------------------------------------------- reports
    def _write_reports(self, run_valid: bool, restoration_valid: bool, error: str | None) -> None:
        super()._write_reports(run_valid, restoration_valid, error)
        legacy_md = self.run_dir / "headroom_diagnosis.md"
        if legacy_md.is_file():
            legacy_md.rename(self.run_dir / "e0_point_table.md")
        _write_csv(self.run_dir / "idle_ladder.csv", self.idle_rows)
        _write_csv(self.run_dir / "clock_mode_actions.csv", self.mode_rows)
        summary = self._e0_summary(run_valid, restoration_valid, error)
        _write_json(self.run_dir / "e0_summary.json", summary)
        (self.run_dir / "e0_summary.md").write_text(render_markdown(summary), encoding="utf-8")

    def _e0_summary(self, run_valid: bool, restoration_valid: bool, error: str | None) -> dict[str, Any]:
        slo = self.settings["slo"]
        attainment = float(slo["attainment"])
        points = []
        for point in self._points():
            for load_point in self.settings["load_points"]:
                rows = [row for row in self.headroom_rows
                        if row["point_id"] == point["id"] and row["load_point"] == load_point["id"]]
                requests = [row for row in self.request_rows
                            if row["point_id"] == point["id"] and row["load_point"] == load_point["id"]]
                if not rows or not requests:
                    continue
                per_repeat = []
                for repeat in sorted({int(row["repeat"]) for row in rows}):
                    selected = [row for row in requests if int(row["repeat"]) == repeat]
                    energy = sum(float(row["total_gross_energy_j"]) for row in rows if int(row["repeat"]) == repeat)
                    per_repeat.append(_slo_and_energy(selected, energy, slo) | {"repeat": repeat})
                energy = sum(float(row["total_gross_energy_j"]) for row in rows)
                aggregate = _slo_and_energy(requests, energy, slo)
                pool = {
                    pool_id: sum(float(row["%s_energy_j" % endpoint_id]) for row in rows
                                 for endpoint_id in POOLS[pool_id])
                    for pool_id in ("P", "D")
                }
                points.append({
                    "point_id": point["id"], "P_level": point["P_level"], "D_level": point["D_level"],
                    "load_point": load_point["id"], "repeats": per_repeat, **aggregate,
                    "P_energy_j": pool["P"], "D_energy_j": pool["D"],
                    "every_repeat_slo_pass": all(item["slo_pass"] for item in per_repeat),
                    "measurement_valid": all(bool(row["measurement_valid"]) for row in rows),
                    "clock_valid": all(bool(row["%s_clock_valid" % e]) for row in rows for e in ("P0", "P1", "D0", "D1")),
                    "energy_valid": all(bool(row["%s_energy_valid" % e]) for row in rows for e in ("P0", "P1", "D0", "D1")),
                    "throttle_valid": all(bool(row["%s_throttle_valid" % e]) for row in rows for e in ("P0", "P1", "D0", "D1")),
                })
        for item in points:
            base = next((other for other in points if other["point_id"] == "HIGH_HIGH"
                         and other["load_point"] == item["load_point"]), None)
            item["raw_saving_vs_high"] = (
                (base["joules_per_request"] - item["joules_per_request"]) / base["joules_per_request"]
                if base else None
            )
        idle = []
        for ladder in sorted({row["ladder"] for row in self.idle_rows}):
            for state in self.e0["idle_ladder"]["states"]:
                for pool_id in ("P", "D"):
                    rows = [row for row in self.idle_rows
                            if row["ladder"] == ladder and row["state"] == state and row["pool"] == pool_id]
                    if rows:
                        idle.append({
                            "ladder": ladder, "state": state, "pool": pool_id,
                            "mean_idle_power_w_per_gpu": statistics.fmean(
                                float(row["idle_power_w"]) for row in rows if row.get("idle_power_w") is not None
                            ) if any(row.get("idle_power_w") is not None for row in rows) else None,
                            "all_valid": all(bool(row["valid"]) for row in rows),
                        })
        rules = _decision_rules(points, idle, self.e0.get("decision_rules", {}))
        matched = bool(all(
            len({str(row["trace_signature"]) for row in self.headroom_rows
                 if row["load_point"] == load_point["id"] and int(row["repeat"]) == repeat}) <= 1
            for load_point in self.settings["load_points"]
            for repeat in range(1, int(self.e0["repetitions"]) + 1)
        ))
        return {
            "schema_version": 1, "phase": "Stage_E0_regime_diagnosis", "run_id": self.run_id,
            "valid": bool(
                run_valid and restoration_valid and matched and points
                and all(item["measurement_valid"] and item["clock_valid"] and item["energy_valid"] for item in points)
                and all(item["all_valid"] for item in idle)
            ),
            "run_completed": run_valid, "restoration_valid": restoration_valid,
            "matched_trace_audit": matched, "error": error,
            "slo": slo, "points": points, "idle_ladder": idle, "decision_rules": rules,
            "claim_boundary": (
                "fixed-point and default-governor regime diagnosis on the unchanged C1 trace; "
                "not a controller-policy result and not a capacity result"
            ),
        }


def _slo_and_energy(requests: Sequence[Mapping[str, Any]], energy_j: float, slo: Mapping[str, Any]) -> dict[str, Any]:
    ttft = [float(row["ttft_ms"]) for row in requests]
    tpot = [float(row["tpot_ms"]) for row in requests]
    ttft_att = sum(value <= float(slo["ttft_ms"]) for value in ttft) / len(ttft)
    tpot_att = sum(value <= float(slo["tpot_ms"]) for value in tpot) / len(tpot)
    return {
        "request_count": len(requests), "total_gross_energy_j": energy_j,
        "joules_per_request": energy_j / len(requests),
        "ttft_p95_ms": percentile(ttft, 0.95), "ttft_p99_ms": percentile(ttft, 0.99),
        "tpot_p95_ms": percentile(tpot, 0.95), "tpot_p99_ms": percentile(tpot, 0.99),
        "ttft_attainment": ttft_att, "tpot_attainment": tpot_att,
        "slo_pass": ttft_att >= float(slo["attainment"]) and tpot_att >= float(slo["attainment"]),
    }


def _decision_rules(
    points: Sequence[Mapping[str, Any]], idle: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate the rules pre-registered in the experiment config."""
    result = []
    for load_point in sorted({item["load_point"] for item in points}):
        by_id = {item["point_id"]: item for item in points if item["load_point"] == load_point}
        default = by_id.get("DEFAULT_DEFAULT")
        tuned = by_id.get("P_MID_D_HIGH")
        p_default = by_id.get("P_DEFAULT_D_HIGH")
        result.append({
            "load_point": load_point,
            "default_meets_slo_every_repeat": bool(default and default["every_repeat_slo_pass"]),
            "default_energy_le_tuned": bool(default and tuned and default["joules_per_request"] <= tuned["joules_per_request"]),
            "p_default_meets_slo_every_repeat": bool(p_default and p_default["every_repeat_slo_pass"]),
            "p_default_energy_le_tuned": bool(p_default and tuned and p_default["joules_per_request"] <= tuned["joules_per_request"]),
        })
    pre = {row["state"]: row["mean_idle_power_w_per_gpu"] for row in idle if row["ladder"] == "pre" and row["pool"] == "P"}
    threshold = config.get("p_default_idle_break_even_w")
    result.append({
        "load_point": "idle",
        "p_default_idle_w": pre.get(DEFAULT),
        "p_default_idle_break_even_w": threshold,
        "p_default_idle_below_break_even": (
            pre.get(DEFAULT) is not None and threshold is not None and pre[DEFAULT] < float(threshold)
        ),
    })
    return result


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# XPYD Stage E0 regime diagnosis", "",
        "Run `%s`; valid = **%s** (completed=%s, restoration=%s, matched trace=%s)." % (
            summary["run_id"], summary["valid"], summary["run_completed"],
            summary["restoration_valid"], summary["matched_trace_audit"],
        ),
    ]
    if summary.get("error"):
        lines += ["", "Error: `%s`" % summary["error"]]
    lines += [
        "", "## E0-a idle power (W per GPU, vLLM running, model resident, no request)", "",
        "| Ladder | State | Pool | Idle W/GPU | Valid |", "|---|---|---|---:|---|",
    ]
    for row in summary["idle_ladder"]:
        value = row["mean_idle_power_w_per_gpu"]
        lines.append("| %s | %s | %s | %s | %s |" % (
            row["ladder"], row["state"], row["pool"],
            "n/a" if value is None else "%.2f" % value, row["all_valid"],
        ))
    lines += [
        "", "## E0-b matched serving points (SLO %s ms TTFT / %s ms TPOT, >= %.0f%%)" % (
            summary["slo"]["ttft_ms"], summary["slo"]["tpot_ms"], 100 * float(summary["slo"]["attainment"])),
        "",
        "| Point | Load | J/request | Raw saving vs HIGH | P / D energy (kJ) | TTFT p95 / attain | TPOT p95 / attain | SLO every repeat | Valid (meas/clock/energy/throttle) |",
        "|---|---|---:|---:|---|---|---|---|---|",
    ]
    for row in summary["points"]:
        saving = row["raw_saving_vs_high"]
        lines.append("| %s | %s | %.1f | %s | %.1f / %.1f | %.1f / %.1f%% | %.2f / %.1f%% | %s | %s/%s/%s/%s |" % (
            row["point_id"], row["load_point"], row["joules_per_request"],
            "n/a" if saving is None else "%.2f%%" % (100 * saving),
            row["P_energy_j"] / 1e3, row["D_energy_j"] / 1e3,
            row["ttft_p95_ms"], 100 * row["ttft_attainment"],
            row["tpot_p95_ms"], 100 * row["tpot_attainment"],
            row["every_repeat_slo_pass"], row["measurement_valid"], row["clock_valid"],
            row["energy_valid"], row["throttle_valid"],
        ))
    lines += ["", "## Pre-registered decision rules", "", "```json",
              json.dumps(summary["decision_rules"], indent=2, sort_keys=True), "```", "",
              summary["claim_boundary"], ""]
    return "\n".join(lines)


def load_e0_configs(runtime_path: Path, experiment_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """C1 fixed-point contract plus the frozen E0 matrix."""
    runtime, experiment = load_configs(runtime_path, experiment_path)
    settings = experiment["stagec1"]
    slo = settings["slo"]
    if (float(slo["ttft_ms"]), float(slo["tpot_ms"]), float(slo["attainment"])) != (600.0, 65.0, 0.95):
        raise StageE0Error("E0 must use the unchanged TIGHT 600/65 ms >=95% contract")
    if settings.get("frequency_levels_mhz") != {
        "P0": {"LOW": 1260, "MID": 1890, "HIGH": 2520},
        "P1": {"LOW": 1260, "MID": 1890, "HIGH": 2520},
        "D0": {"LOW": 750, "MID": 1125, "HIGH": 1500},
        "D1": {"LOW": 750, "MID": 1125, "HIGH": 1500},
    }:
        raise StageE0Error("E0 locked levels must equal the validated C1 levels")
    scenarios = {str(item["id"]): item for item in experiment.get("scenarios", [])}
    for load_point in settings["load_points"]:
        segment = scenarios[str(load_point["scenario"])]["segments"][0]
        if int(segment["request_count"]) != 20 or segment["class_weights"] != {
            "SS": 0.35, "SL": 0.15, "LS": 0.35, "LL": 0.15,
        }:
            raise StageE0Error("E0 must reuse the C1 20-request 35/15/35/15 window")
    e0 = experiment.get("stage_e0")
    if not isinstance(e0, Mapping):
        raise StageE0Error("stage_e0 settings are required")
    ids = [str(item.get("id")) for item in e0.get("points", [])]
    if ids != ["HIGH_HIGH", "P_MID_D_HIGH", "P_DEFAULT_D_HIGH", "DEFAULT_DEFAULT"]:
        raise StageE0Error("E0 point matrix is frozen to HIGH_HIGH, P_MID_D_HIGH, P_DEFAULT_D_HIGH, DEFAULT_DEFAULT")
    if int(e0.get("repetitions", 0)) < 2:
        raise StageE0Error("E0 requires at least two repeats")
    ladder = e0.get("idle_ladder", {})
    if not set(ladder.get("states", [])) <= set(LEVELS) or DEFAULT not in ladder.get("states", []):
        raise StageE0Error("E0 idle ladder must include DEFAULT and only known levels")
    if float(ladder.get("settle_s", 0)) < 30 or float(ladder.get("measure_s", 0)) < 60:
        raise StageE0Error("E0 idle ladder needs >=30 s settle and >=60 s measurement")
    return runtime, experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    runtime, experiment = load_e0_configs(args.runtime_config, args.experiment_config)
    StageE0RegimeHarness(runtime, experiment, args.run_id).run()


if __name__ == "__main__":
    main()
