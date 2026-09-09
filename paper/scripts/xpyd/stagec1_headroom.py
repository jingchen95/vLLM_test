"""Stage C1 fixed-frequency headroom diagnosis.

This is an analysis aid, not a policy evaluation.  It reuses the Stage C1
mixed trace and Phase 3C measurement path while holding both GPU pools at a
specified fixed operating point.  No feedback decision, adaptive routing,
probing, predictor, model, or oracle is used.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from xpyd.phase3c_substrate import _read_json, _write_json
from xpyd.phase3d_control import NvidiaSmiClockBackend
from xpyd.stagec0_capacity import analyze_point, percentile
from xpyd.stagec1_stationary import (
    POOLS,
    ControllerContext,
    StageC1Error,
    StageC1Harness,
    _read_csv,
    _trace_signature,
    load_configs,
)
from xpyd.workload_harness import build_trace


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _level_frequency(
    levels: Mapping[str, Mapping[str, Any]], pool: str, level: str,
) -> int:
    endpoint = POOLS[pool][0]
    value = levels[endpoint].get(level)
    if value is None:
        raise StageC1Error("missing %s frequency level for %s pool" % (level, pool))
    return int(value)


def _actual_endpoint_fields(window_dir: Path) -> dict[str, Any]:
    rows = _read_csv(window_dir / "endpoint_summary.csv")
    result: dict[str, Any] = {}
    if {row["endpoint_id"] for row in rows} != {"P0", "P1", "D0", "D1"}:
        raise StageC1Error("headroom point did not observe all four endpoints")
    for row in rows:
        endpoint = row["endpoint_id"]
        result.update({
            "%s_requested_mhz" % endpoint: int(float(row["configured_graphics_mhz"])),
            "%s_observed_mean_mhz" % endpoint: float(row["observed_graphics_mean_mhz"]),
            "%s_clock_valid" % endpoint: row["clock_valid"] == "True",
            "%s_energy_valid" % endpoint: row["energy_valid"] == "True",
            "%s_throttle_valid" % endpoint: row["throttle_valid"] == "True",
        })
    return result


class StageC1HeadroomHarness(StageC1Harness):
    """Run fixed HH and one-dimensional lower-frequency points."""

    def __init__(
        self, runtime: Mapping[str, Any], experiment: Mapping[str, Any],
        run_id: str | None = None, backend: NvidiaSmiClockBackend | None = None,
    ) -> None:
        super().__init__(runtime, experiment, run_id=run_id, backend=backend, smoke=False)
        self.headroom_settings = self.experiment.get("stagec1_headroom", {})
        self.run_dir = Path(os.path.expandvars(str(experiment["output_root"]))) / self.run_id
        self.headroom_rows: list[dict[str, Any]] = []
        self.request_rows: list[dict[str, Any]] = []

    def _points(self) -> list[dict[str, Any]]:
        points = self.headroom_settings.get("diagnostic_points")
        if not isinstance(points, list) or not points:
            raise StageC1Error("stagec1_headroom.diagnostic_points is required")
        seen: set[str] = set()
        result = []
        for item in points:
            point = dict(item)
            point_id = str(point.get("id", ""))
            if not point_id or point_id in seen:
                raise StageC1Error("headroom point IDs must be unique and non-empty")
            if point.get("P_level") not in {"LOW", "MID", "HIGH"}:
                raise StageC1Error("headroom P_level must be LOW, MID, or HIGH")
            if point.get("D_level") not in {"LOW", "MID", "HIGH"}:
                raise StageC1Error("headroom D_level must be LOW, MID, or HIGH")
            seen.add(point_id)
            result.append(point)
        if result[0].get("P_level") != "HIGH" or result[0].get("D_level") != "HIGH":
            raise StageC1Error("headroom diagnosis must start with HIGH/HIGH")
        return result

    def _point_targets(
        self, point: Mapping[str, Any], capabilities: Mapping[str, Any],
    ) -> dict[str, int]:
        levels = self.settings["frequency_levels_mhz"]
        p = _level_frequency(levels, "P", str(point["P_level"]))
        d = _level_frequency(levels, "D", str(point["D_level"]))
        return {"P0": p, "P1": p, "D0": d, "D1": d}

    def _point_measurement(
        self, point: Mapping[str, Any], window_dir: Path, *, load_point: Mapping[str, Any],
        repeat: int, context: ControllerContext, signature: str, window_id: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        metrics = analyze_point(
            load_point, window_dir, self.settings["slo"], self.settings["trend_thresholds"]
        )
        requests = _read_csv(window_dir / "requests.csv")
        ttft_limit = float(self.settings["slo"]["ttft_ms"])
        tpot_limit = float(self.settings["slo"]["tpot_ms"])
        ttft = [float(row["ttft_ms"]) for row in requests]
        tpot = [float(row["tpot_ms"]) for row in requests]
        row = {key: value for key, value in metrics.items() if key != "endpoint_load"}
        row.update({
            "window_id": window_id, "point_id": point["id"],
            "P_level": point["P_level"], "D_level": point["D_level"],
            "repeat": repeat, "trace_signature": signature,
            "ttft_attainment": sum(value <= ttft_limit for value in ttft) / len(ttft),
            "tpot_attainment": sum(value <= tpot_limit for value in tpot) / len(tpot),
            "slo_contract_pass": (
                sum(value <= ttft_limit for value in ttft) / len(ttft) >= 0.95
                and sum(value <= tpot_limit for value in tpot) / len(tpot) >= 0.95
            ),
            "P0_freq_mhz": context.targets["P0"], "P1_freq_mhz": context.targets["P1"],
            "D0_freq_mhz": context.targets["D0"], "D1_freq_mhz": context.targets["D1"],
            "endpoint_load_json": json.dumps(metrics["endpoint_load"], sort_keys=True),
            **_actual_endpoint_fields(window_dir),
        })
        for request in requests:
            request.update({
                "point_id": point["id"], "P_level": point["P_level"],
                "D_level": point["D_level"], "repeat": repeat,
                "window_id": window_id, "load_point": load_point["id"],
            })
        return row, requests

    def run(self) -> Path:
        if self.run_dir.exists():
            raise StageC1Error("headroom output already exists: %s" % self.run_dir)
        windows_root = self.run_dir / "windows"
        windows_root.mkdir(parents=True)
        capabilities = self._discover()
        discovered = {key: self._levels(value) for key, value in capabilities.items()}
        if discovered != self.settings["frequency_levels_mhz"]:
            raise StageC1Error("discovered clocks differ from validated Stage 3D-A levels")
        _write_json(self.run_dir / "capabilities.json", {
            "endpoints": {key: asdict(value) for key, value in capabilities.items()},
            "frequency_levels_mhz": discovered,
        })
        actuator = self._make_actuator(capabilities)
        high = {key: value.selected_high_mhz for key, value in capabilities.items()}
        points = self._points()
        loads = [dict(item) for item in self.settings["load_points"]]
        repetitions = int(self.headroom_settings.get("repetitions", 2))
        if repetitions < 2:
            raise StageC1Error("headroom diagnosis requires at least two repeats")
        context: ControllerContext | None = None
        error: str | None = None
        valid = False
        restoration_valid = False
        try:
            for repeat in range(1, repetitions + 1):
                for point in points:
                    for load_point in loads:
                        registry, telemetry, scheduler = self._build_scheduler(capabilities)
                        context = ControllerContext(registry, telemetry, scheduler, dict(high))
                        # Every point is cold-started at HIGH/HIGH, so lower
                        # frequency energy and latency are directly comparable.
                        self._actuate_targets(
                            actuator, context, high, "headroom_cold_start_high", "HEADROOM", "cold"
                        )
                        targets = self._point_targets(point, capabilities)
                        self._actuate_targets(
                            actuator, context, targets, "headroom_fixed_point", "HEADROOM", str(point["id"])
                        )
                        scenario = next(
                            str(item["scenario"])
                            for item in loads if item["id"] == load_point["id"]
                        )
                        trace = build_trace(
                            self.experiment, scenario,
                            "headroom-r%02d-%s-%s" % (repeat, point["id"], load_point["id"]),
                        )
                        signature = _trace_signature(trace)
                        window_id = "r%02d-%s-%s" % (repeat, point["id"], load_point["id"])
                        window_dir = self._run_window(window_id, trace, targets, windows_root)
                        row, requests = self._point_measurement(
                            point, window_dir, load_point=load_point, repeat=repeat,
                            context=context, signature=signature, window_id=window_id,
                        )
                        self.headroom_rows.append(row)
                        self.request_rows.extend(requests)
            valid = True
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
            raise
        finally:
            try:
                if context is not None:
                    self._actuate_targets(
                        actuator, context, high, "headroom_final_restore_high", "RESTORE", "final"
                    )
                    restoration_valid = all(
                        actuator.read(key).graphics_clock_mhz == value
                        for key, value in high.items()
                    )
            finally:
                if self.run_dir.exists():
                    self._write_reports(valid, restoration_valid, error)
        print(self.run_dir.as_posix())
        return self.run_dir

    def _make_actuator(self, capabilities: Mapping[str, Any]) -> Any:
        # Keep the construction in one place so this diagnostic cannot acquire
        # a second actuation implementation.
        from xpyd.phase3d_control import PerEndpointClockActuator
        return PerEndpointClockActuator(
            self.backend, capabilities, float(self.config["phase3d"]["minimum_dwell_s"])
        )

    def _write_reports(self, run_valid: bool, restoration_valid: bool, error: str | None) -> None:
        _write_csv(self.run_dir / "headroom_measurements.csv", self.headroom_rows)
        _write_csv(self.run_dir / "headroom_requests.csv", self.request_rows)
        _write_csv(self.run_dir / "clock_actions.csv", self.action_rows)
        _write_csv(self.run_dir / "telemetry_observations.csv", self.telemetry_rows)
        groups: list[dict[str, Any]] = []
        repeat_groups: list[dict[str, Any]] = []
        for point in self._points():
            for load_point in self.settings["load_points"]:
                selected = [
                    row for row in self.headroom_rows
                    if row["point_id"] == point["id"] and row["load_point"] == load_point["id"]
                ]
                if not selected:
                    continue
                requests = [
                    row for row in self.request_rows
                    if row["point_id"] == point["id"] and row["load_point"] == load_point["id"]
                ]
                for repeat in sorted({int(row["repeat"]) for row in selected}):
                    repeat_rows = [row for row in selected if int(row["repeat"]) == repeat]
                    repeat_requests = [row for row in requests if int(row["repeat"]) == repeat]
                    energy = sum(float(row["total_gross_energy_j"]) for row in repeat_rows)
                    ttft = [float(row["ttft_ms"]) for row in repeat_requests]
                    tpot = [float(row["tpot_ms"]) for row in repeat_requests]
                    repeat_groups.append({
                        "point_id": point["id"], "P_level": point["P_level"], "D_level": point["D_level"],
                        "load_point": load_point["id"], "repeat": repeat,
                        "request_count": len(repeat_requests), "total_gross_energy_j": energy,
                        "joules_per_request": energy / len(repeat_requests),
                        "ttft_p95_ms": percentile(ttft, 0.95), "ttft_p99_ms": percentile(ttft, 0.99),
                        "ttft_attainment": sum(value <= 600.0 for value in ttft) / len(ttft),
                        "tpot_p95_ms": percentile(tpot, 0.95), "tpot_p99_ms": percentile(tpot, 0.99),
                        "tpot_attainment": sum(value <= 65.0 for value in tpot) / len(tpot),
                        "throughput_rps_mean": statistics.fmean(float(row["completed_rate_rps"]) for row in repeat_rows),
                        "admission_delay_p95_ms": percentile([float(row["admission_delay_p95_ms"]) for row in repeat_rows], 0.95),
                        "client_waiting_max": max(float(row["client_waiting_max"]) for row in repeat_rows),
                        "backlog_behavior": ",".join(sorted({str(row["backlog_behavior"]) for row in repeat_rows})),
                        "measurement_valid": all(bool(row["measurement_valid"]) for row in repeat_rows),
                        "clock_valid": all(bool(row["P0_clock_valid"]) and bool(row["P1_clock_valid"])
                                            and bool(row["D0_clock_valid"]) and bool(row["D1_clock_valid"])
                                            for row in repeat_rows),
                        "energy_valid": all(bool(row["P0_energy_valid"]) and bool(row["P1_energy_valid"])
                                             and bool(row["D0_energy_valid"]) and bool(row["D1_energy_valid"])
                                             for row in repeat_rows),
                    })
                ttft = [float(row["ttft_ms"]) for row in requests]
                tpot = [float(row["tpot_ms"]) for row in requests]
                energy = sum(float(row["total_gross_energy_j"]) for row in selected)
                count = len(requests)
                aggregate = {
                    "point_id": point["id"], "P_level": point["P_level"], "D_level": point["D_level"],
                    "load_point": load_point["id"], "repeat_count": len({int(row["repeat"]) for row in selected}),
                    "request_count": count, "total_gross_energy_j": energy,
                    "joules_per_request": energy / count,
                    "joules_per_request_repeat_std": statistics.stdev(
                        float(row["joules_per_request"]) for row in repeat_groups
                        if row["point_id"] == point["id"] and row["load_point"] == load_point["id"]
                    ) if len([row for row in repeat_groups if row["point_id"] == point["id"] and row["load_point"] == load_point["id"]]) >= 2 else None,
                    "ttft_p95_ms": percentile(ttft, 0.95), "ttft_p99_ms": percentile(ttft, 0.99),
                    "ttft_attainment": sum(value <= 600.0 for value in ttft) / count,
                    "tpot_p95_ms": percentile(tpot, 0.95), "tpot_p99_ms": percentile(tpot, 0.99),
                    "tpot_attainment": sum(value <= 65.0 for value in tpot) / count,
                    "throughput_rps_mean": statistics.fmean(float(row["completed_rate_rps"]) for row in selected),
                    "admission_delay_p95_ms": percentile([float(row["admission_delay_p95_ms"]) for row in selected], 0.95),
                    "client_waiting_max": max(float(row["client_waiting_max"]) for row in selected),
                    "backlog_behavior": ",".join(sorted({str(row["backlog_behavior"]) for row in selected})),
                    "measurement_valid": all(bool(row["measurement_valid"]) for row in selected),
                    "clock_valid": all(bool(row["P0_clock_valid"]) and bool(row["P1_clock_valid"])
                                        and bool(row["D0_clock_valid"]) and bool(row["D1_clock_valid"])
                                        for row in selected),
                    "energy_valid": all(bool(row["P0_energy_valid"]) and bool(row["P1_energy_valid"])
                                         and bool(row["D0_energy_valid"]) and bool(row["D1_energy_valid"])
                                         for row in selected),
                }
                aggregate["slo_contract_pass"] = aggregate["ttft_attainment"] >= 0.95 and aggregate["tpot_attainment"] >= 0.95
                groups.append(aggregate)
        for row in groups:
            baseline = next(
                item for item in groups
                if item["load_point"] == row["load_point"] and item["point_id"] == "HIGH_HIGH"
            )
            row["energy_saving_vs_high_fraction"] = (
                (baseline["joules_per_request"] - row["joules_per_request"])
                / baseline["joules_per_request"] if row["slo_contract_pass"] else None
            )
            row["safe_headroom_point"] = bool(
                row["slo_contract_pass"] and row["measurement_valid"]
                and row["clock_valid"] and row["energy_valid"]
            )
        _write_csv(self.run_dir / "headroom_repeat_summary.csv", repeat_groups)
        _write_csv(self.run_dir / "headroom_summary.csv", groups)
        matched_trace_audit = bool(all(
            len({str(row["trace_signature"]) for row in self.headroom_rows
                 if row["load_point"] == load_point["id"]
                 and int(row["repeat"]) == repeat}) == 1
            for load_point in self.settings["load_points"]
            for repeat in range(1, int(self.headroom_settings.get("repetitions", 2)) + 1)
        ))
        summary = {
            "schema_version": 1, "phase": "Stage_C1_DVFS_headroom_diagnosis",
            "valid": bool(
                run_valid and restoration_valid and groups
                and matched_trace_audit
                and all(
                    row["measurement_valid"] and row["clock_valid"] and row["energy_valid"]
                    for row in groups
                )
            ),
            "error": error, "repetitions": int(self.headroom_settings.get("repetitions", 2)),
            "diagnostic_points": self._points(), "loads": self.settings["load_points"],
            "slo": self.settings["slo"], "results": groups,
            "matched_trace_audit": matched_trace_audit,
            "claim_boundary": "fixed-frequency headroom diagnosis only; not a controller-policy result",
        }
        _write_json(self.run_dir / "headroom_summary.json", summary)
        lines = [
            "# XPYD Stage C1 DVFS-headroom diagnosis", "",
            "Fixed-frequency, matched mixed traces; no feedback decision or adaptive routing was active.", "",
            "| Point | P/D | Load | J/request | TTFT p95 / attain | TPOT p95 / attain | Throughput | Admission p95 | SLO-safe | Energy vs HH |",
            "|---|---|---|---:|---:|---:|---:|---:|---|---:|",
        ]
        for row in groups:
            lines.append(
                "| %s | %s/%s | %s | %.2f | %.1f / %.1f%% | %.1f / %.1f%% | %.3f | %.2f ms | %s | %s |" % (
                    row["point_id"], row["P_level"], row["D_level"], row["load_point"],
                    row["joules_per_request"], row["ttft_p95_ms"], 100 * row["ttft_attainment"],
                    row["tpot_p95_ms"], 100 * row["tpot_attainment"], row["throughput_rps_mean"],
                    row["admission_delay_p95_ms"], "YES" if row["safe_headroom_point"] else "NO",
                    "n/a" if row["energy_saving_vs_high_fraction"] is None else "%.2f%%" % (100 * row["energy_saving_vs_high_fraction"]),
                )
            )
        lines += [
            "", "A point is SLO-safe only when aggregate TTFT and TPOT attainment are at least 95% and all measurement/clock/energy audits pass.",
            "", "The controller diagnosis is reported separately from these fixed-point measurements; no threshold was retuned from this run.",
        ]
        (self.run_dir / "headroom_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    runtime, experiment = load_configs(args.runtime_config, args.experiment_config)
    StageC1HeadroomHarness(runtime, experiment, args.run_id).run()


if __name__ == "__main__":
    main()
