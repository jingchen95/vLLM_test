"""Phase 4A small measured empirical offline oracle.

This harness enumerates a deliberately pruned action space on the accepted
real 2P2D substrate. It uses no predictor and makes no online decisions.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time
from typing import Any, Mapping, Optional, Sequence

from xpyd.phase3c_substrate import (
    Phase3CSubstrateHarness,
    _read_json,
    _write_json,
    build_registry_and_compatibility,
    load_config as load_phase3c_config,
)
from xpyd.phase3d_control import (
    EndpointClockCapability,
    NvidiaSmiClockBackend,
    PerEndpointClockActuator,
    _atomic_json,
    _csv,
    _phase3c_window_config,
)


class Phase4AError(RuntimeError):
    """A fail-closed Phase 4A configuration, measurement, or audit error."""


@dataclass(frozen=True)
class OracleAction:
    config_id: str
    prefill_endpoint_id: str
    decode_endpoint_id: str
    profile_id: str
    levels: dict[str, str]
    static_baseline: bool = False


def build_action_space(
    pairs: Sequence[Sequence[str]],
    endpoint_ids: Sequence[str],
    profiles: Sequence[Mapping[str, str]],
    canonical_route: Sequence[str],
    canonical_extra_profiles: Sequence[Mapping[str, str]],
) -> list[OracleAction]:
    """Build the documented pruned route/DVFS design without a Cartesian sweep."""
    actions: list[OracleAction] = []
    seen: set[str] = set()

    def add(pair: Sequence[str], profile: Mapping[str, str], baseline: bool = False) -> None:
        p, d = map(str, pair)
        profile_id = str(profile["id"])
        levels = {endpoint_id: "HIGH" if baseline else "LOW" for endpoint_id in endpoint_ids}
        if not baseline:
            levels[p] = str(profile["prefill_level"]).upper()
            levels[d] = str(profile["decode_level"]).upper()
        config_id = "static_max_p0d0" if baseline else "%s_%s_%s" % (
            p.lower(), d.lower(), profile_id.lower()
        )
        if config_id in seen:
            raise Phase4AError("duplicate Phase 4A configuration: %s" % config_id)
        if any(value not in {"LOW", "MID", "HIGH"} for value in levels.values()):
            raise Phase4AError("unsupported frequency level in %s" % config_id)
        seen.add(config_id)
        actions.append(OracleAction(config_id, p, d, profile_id, levels, baseline))

    normalized_pairs = [tuple(map(str, pair)) for pair in pairs]
    canonical = tuple(map(str, canonical_route))
    if canonical not in normalized_pairs:
        raise Phase4AError("canonical route is not explicitly compatible")
    for pair in normalized_pairs:
        for profile in profiles:
            add(pair, profile)
    existing_canonical = {
        (action.levels[canonical[0]], action.levels[canonical[1]])
        for action in actions
        if (action.prefill_endpoint_id, action.decode_endpoint_id) == canonical
    }
    for profile in canonical_extra_profiles:
        levels = (
            str(profile["prefill_level"]).upper(),
            str(profile["decode_level"]).upper(),
        )
        if levels not in existing_canonical:
            add(canonical, profile)
            existing_canonical.add(levels)
    add(canonical, {"id": "ALL_HIGH", "prefill_level": "HIGH", "decode_level": "HIGH"}, True)
    return actions


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values)


def _std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _ci95(values: Sequence[float]) -> float:
    # Five repeats are configured; use exact small-sample t critical values.
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    return critical * _std(values) / math.sqrt(len(values)) if values else math.nan


def aggregate_measurements(
    rows: Sequence[Mapping[str, Any]], expected_repeats: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["workload"]), str(row["config_id"])), []).append(row)
    result = []
    metrics = (
        "total_gpu_gross_energy_j", "joules_per_request",
        "joules_per_output_token", "mean_ttft_ms", "mean_tpot_ms",
        "mean_itl_ms", "mean_e2e_latency_ms", "throughput_requests_s",
    )
    for (workload, config_id), items in sorted(grouped.items()):
        valid_items = [item for item in items if bool(item.get("measurement_valid"))]
        repeat_ids = {int(item["repeat"]) for item in items}
        eligible = (
            len(items) == expected_repeats
            and repeat_ids == set(range(1, expected_repeats + 1))
            and len(valid_items) == expected_repeats
            and all(bool(item.get("slo_pass")) for item in valid_items)
        )
        first = items[0]
        aggregate: dict[str, Any] = {
            "workload": workload,
            "config_id": config_id,
            "prefill_endpoint_id": first["prefill_endpoint_id"],
            "decode_endpoint_id": first["decode_endpoint_id"],
            "profile_id": first["profile_id"],
            "static_baseline": bool(first["static_baseline"]),
            "repeat_count": len(items),
            "valid_repeat_count": len(valid_items),
            "all_repeats_slo_pass": bool(valid_items) and all(
                bool(item.get("slo_pass")) for item in valid_items
            ),
            "oracle_eligible": eligible,
        }
        for endpoint_id in ("P0", "P1", "D0", "D1"):
            aggregate["%s_freq_mhz" % endpoint_id] = first["%s_requested_freq_mhz" % endpoint_id]
        for metric in metrics:
            values = [float(item[metric]) for item in valid_items if item.get(metric) is not None]
            aggregate[metric + "_mean"] = _mean(values) if values else None
            aggregate[metric + "_std"] = _std(values) if values else None
            aggregate[metric + "_ci95_half_width"] = _ci95(values) if values else None
        result.append(aggregate)
    return result


def select_oracles(
    aggregates: Sequence[Mapping[str, Any]], workloads: Sequence[Mapping[str, Any]],
    ttft_slo_ms: float, tpot_slo_ms: float,
) -> list[dict[str, Any]]:
    result = []
    for workload_spec in workloads:
        workload = str(workload_spec["id"])
        candidates = sorted(
            (item for item in aggregates if item["workload"] == workload and item["oracle_eligible"]),
            key=lambda item: (
                float(item["total_gpu_gross_energy_j_mean"]), str(item["config_id"])
            ),
        )
        baseline = next(
            (item for item in aggregates if item["workload"] == workload and item["static_baseline"]),
            None,
        )
        if not candidates or baseline is None or not baseline["oracle_eligible"]:
            continue
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        best_jpr = float(best["joules_per_request_mean"])
        baseline_jpr = float(baseline["joules_per_request_mean"])
        row = {
            "workload": workload,
            "best_P": best["prefill_endpoint_id"],
            "best_D": best["decode_endpoint_id"],
            "best_config_id": best["config_id"],
            "second_best_config_id": second["config_id"] if second else None,
            "baseline_config_id": baseline["config_id"],
            "P0_freq": best["P0_freq_mhz"], "P1_freq": best["P1_freq_mhz"],
            "D0_freq": best["D0_freq_mhz"], "D1_freq": best["D1_freq_mhz"],
            "energy": best["total_gpu_gross_energy_j_mean"],
            "energy_std": best["total_gpu_gross_energy_j_std"],
            "energy_ci95_half_width": best["total_gpu_gross_energy_j_ci95_half_width"],
            "J_per_request": best_jpr,
            "TTFT": best["mean_ttft_ms_mean"],
            "TPOT": best["mean_tpot_ms_mean"],
            "SLO_pass": True,
            "energy_vs_static": (baseline_jpr - best_jpr) / baseline_jpr,
            "second_best_energy_gap": (
                (float(second["joules_per_request_mean"]) - best_jpr) / best_jpr
                if second else None
            ),
            "TTFT_headroom_ms": ttft_slo_ms - float(best["mean_ttft_ms_mean"]),
            "TPOT_headroom_ms": tpot_slo_ms - float(best["mean_tpot_ms_mean"]),
            "eligible_configuration_count": len(candidates),
            "near_optimal_5pct_count": sum(
                float(item["joules_per_request_mean"]) <= best_jpr * 1.05
                for item in candidates
            ),
        }
        result.append(row)
    return result


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    """Render the report from an audited summary without measuring GPUs again."""
    valid = bool(summary["valid"])
    action_space = summary["action_space"]
    slo = summary["slo"]
    oracles = summary["oracles"]
    answers = summary["answers"]
    hard_gates = summary["hard_gates"]
    unique_best = answers["unique_best_configurations"]
    near_counts = answers["optimum_shape"]["near_optimal_5pct_counts"]
    savings = answers["oracle_savings_vs_static"]
    configuration_count = int(action_space["measured_configurations"])
    workload_count = int(summary.get("workload_count", len(oracles)))
    repeats = int(summary.get("repeats", 0))
    if not repeats and configuration_count and workload_count:
        repeats = int(summary["planned_measurement_count"]) // (
            configuration_count * workload_count
        )
    lines = [
        "# XpYd Phase 4A small empirical offline oracle", "",
        "Verdict: **%s**" % ("PASS" if valid else "FAIL"), "",
        "This is the best measured SLO-feasible configuration in a pruned action space; it is not a theoretical optimum.", "",
        "Measured %d configurations x %d workloads x %d repeats (%d windows)."
        % (configuration_count, workload_count, repeats, int(summary["measurement_count"])), "",
        "SLO eligibility requires every repeat mean TTFT <= %.1f ms and mean TPOT <= %.1f ms."
        % (float(slo["ttft_ms"]), float(slo["tpot_ms"])), "",
        "| Workload | Best measured config | Second | J/request | Savings vs static | TTFT / TPOT (ms) | Near-optimal <=5% |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in oracles:
        lines.append(
            f"| {item['workload']} | {item['best_config_id']} | "
            f"{item['second_best_config_id']} | {float(item['J_per_request']):.3f} | "
            f"{float(item['energy_vs_static']):.2%} | "
            f"{float(item['TTFT']):.2f} / {float(item['TPOT']):.2f} | "
            f"{int(item['near_optimal_5pct_count'])} |"
        )
    lines.extend([
        "", "Largest measured main effect: `%s`." % answers["largest_measured_main_effect"],
        "", "Best configuration differs across workloads: **%s** (%s)."
        % (answers["best_configuration_differs_across_workloads"], ", ".join(unique_best) or "none"),
        "", "Oracle J/request savings versus static: min `%.2f%%`, mean `%.2f%%`, max `%.2f%%`."
        % (
            100.0 * float(savings["minimum_fraction"]),
            100.0 * float(savings["mean_fraction"]),
            100.0 * float(savings["maximum_fraction"]),
        ),
        "", f"Near-optimal (within 5%) configuration counts: `{near_counts}`.",
        "", "Ready for Phase 4B: **%s**." % summary["ready_for_phase4b"],
        "", "Hard gates: `%s`" % json.dumps(hard_gates, sort_keys=True), "",
    ])
    if summary.get("error"):
        lines.extend(["Failure: `%s`" % summary["error"], ""])
    return "\n".join(lines)


def _effect_summary(aggregates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = {
        (str(item["workload"]), str(item["prefill_endpoint_id"]),
         str(item["decode_endpoint_id"]), str(item["profile_id"])): item
        for item in aggregates if item["oracle_eligible"] and not item["static_baseline"]
    }

    def energy(item: Mapping[str, Any]) -> float:
        return float(item["joules_per_request_mean"])

    route_effects = []
    by_workload_profile: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for key, item in eligible.items():
        by_workload_profile.setdefault((key[0], key[3]), []).append(item)
    for items in by_workload_profile.values():
        if len(items) == 4:
            values = [energy(item) for item in items]
            route_effects.append((max(values) - min(values)) / min(values))

    p_effects, d_effects, combination_effects = [], [], []
    groups = {(key[0], key[1], key[2]) for key in eligible}
    for workload, p, d in groups:
        profile = {
            key[3]: item for key, item in eligible.items()
            if key[:3] == (workload, p, d)
        }
        for low, high in (("LL", "HL"), ("LH", "HH")):
            if low in profile and high in profile:
                p_effects.append(abs(energy(profile[high]) - energy(profile[low])) / energy(profile[low]))
        for low, high in (("LL", "LH"), ("HL", "HH")):
            if low in profile and high in profile:
                d_effects.append(abs(energy(profile[high]) - energy(profile[low])) / energy(profile[low]))
        if "LL" in profile and "HH" in profile:
            combination_effects.append(abs(energy(profile["HH"]) - energy(profile["LL"])) / energy(profile["LL"]))

    values = {
        "routing_mean_relative_range": _mean(route_effects) if route_effects else None,
        "P_side_DVFS_mean_absolute_effect": _mean(p_effects) if p_effects else None,
        "D_side_DVFS_mean_absolute_effect": _mean(d_effects) if d_effects else None,
        "combined_LL_to_HH_mean_absolute_effect": _mean(combination_effects) if combination_effects else None,
        "comparison_counts": {
            "routing": len(route_effects), "P_side_DVFS": len(p_effects),
            "D_side_DVFS": len(d_effects), "combined": len(combination_effects),
        },
    }
    comparable = {
        "routing": values["routing_mean_relative_range"],
        "P_side_DVFS": values["P_side_DVFS_mean_absolute_effect"],
        "D_side_DVFS": values["D_side_DVFS_mean_absolute_effect"],
    }
    comparable = {key: value for key, value in comparable.items() if value is not None}
    values["largest_measured_main_effect"] = max(comparable, key=comparable.get) if comparable else None
    return values


class Phase4AOracleHarness:
    def __init__(
        self, config: Mapping[str, Any], run_id: Optional[str] = None,
        backend: Optional[NvidiaSmiClockBackend] = None,
    ) -> None:
        self.config = copy.deepcopy(dict(config))
        self.settings = dict(self.config["phase4a"])
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(str(self.settings.get("output_root", "results/phase4a_oracle"))) / self.run_id
        control_file = str(self.config.get("routing_control_file", "")).strip()
        if not control_file:
            raise Phase4AError("Phase 4A requires routing_control_file")
        self.control_file = Path(control_file)
        self.backend = backend or NvidiaSmiClockBackend()
        self.actions: list[dict[str, Any]] = []

    def _accepted_frequencies(self) -> dict[str, dict[str, int]]:
        for key in ("accepted_phase3d_a_audit", "accepted_phase3d_b_audit"):
            path = Path(str(self.settings.get(key, "")))
            if not path.is_file():
                raise Phase4AError("missing accepted prerequisite audit: %s" % path)
            audit = _read_json(path)
            if not audit.get("valid"):
                raise Phase4AError("prerequisite audit is not valid: %s" % path)
        audit = _read_json(Path(str(self.settings["accepted_phase3d_a_audit"])))
        selected = audit.get("selected_frequencies", {})
        result = {}
        for endpoint_id in ("P0", "P1", "D0", "D1"):
            values = selected.get(endpoint_id, {})
            result[endpoint_id] = {
                level: int(values[level]) for level in ("LOW", "MID", "HIGH")
            }
        return result

    def _discover(
        self, accepted: Mapping[str, Mapping[str, int]],
    ) -> dict[str, EndpointClockCapability]:
        result = {}
        for endpoint in self.config["endpoints"]:
            endpoint_id = str(endpoint["endpoint_id"])
            capability = self.backend.discover(
                endpoint_id, str(endpoint["node"]), int(endpoint["gpu_ids"][0]),
                int(accepted[endpoint_id]["HIGH"]),
            )
            actual = {
                "LOW": capability.selected_low_mhz,
                "MID": capability.selected_mid_mhz,
                "HIGH": capability.selected_high_mhz,
            }
            if actual != dict(accepted[endpoint_id]):
                raise Phase4AError("discovered frequencies differ from accepted Phase 3D-A states")
            result[endpoint_id] = capability
        return result

    def _write_routes(self, pairs: Sequence[Sequence[str]], reason: str) -> None:
        _atomic_json(self.control_file, {
            "schema_version": 1, "updated_unix_s": time.time(),
            "pairs": [list(pair) for pair in pairs], "reason": reason,
        })

    def _set_frequencies(
        self, actuator: PerEndpointClockActuator, requested: Mapping[str, int], reason: str,
    ) -> None:
        for endpoint_id in sorted(requested):
            target = int(requested[endpoint_id])
            if actuator.requested.get(endpoint_id) == target:
                continue
            row = actuator.actuate(endpoint_id, target, reason)
            self.actions.append(row)
            if row["command_status"] != "success" or not row["readback_valid"]:
                raise Phase4AError("physical DVFS actuation failed for %s" % endpoint_id)

    def _measurement_row(
        self, window_id: str, workload: Mapping[str, Any], action: OracleAction,
        repeat: int, requested: Mapping[str, int], error: Optional[str] = None,
    ) -> dict[str, Any]:
        window = self.run_dir / "raw" / "windows" / window_id
        row: dict[str, Any] = {
            "run_id": self.run_id, "window_id": window_id,
            "workload": workload["id"], "input_len": workload["input_len"],
            "output_len": workload["output_len"], "rate_rps": workload["rate_rps"],
            "repeat": repeat, "config_id": action.config_id,
            "profile_id": action.profile_id, "static_baseline": action.static_baseline,
            "prefill_endpoint_id": action.prefill_endpoint_id,
            "decode_endpoint_id": action.decode_endpoint_id,
            "measurement_valid": False, "slo_pass": False, "error": error,
        }
        for endpoint_id in ("P0", "P1", "D0", "D1"):
            row["%s_requested_freq_mhz" % endpoint_id] = requested[endpoint_id]
            row["%s_observed_freq_mhz" % endpoint_id] = None
            row["%s_gross_energy_j" % endpoint_id] = None
        summary_path, audit_path = window / "summary.json", window / "audit.json"
        client_path, endpoints_path = window / "client" / "summary.json", window / "endpoint_summary.csv"
        if not all(path.is_file() for path in (summary_path, audit_path, client_path, endpoints_path)):
            return row
        summary, audit, client = _read_json(summary_path), _read_json(audit_path), _read_json(client_path)
        with endpoints_path.open(newline="", encoding="utf-8") as stream:
            endpoint_rows = {item["endpoint_id"]: item for item in csv.DictReader(stream)}
        for endpoint_id in ("P0", "P1", "D0", "D1"):
            row["%s_observed_freq_mhz" % endpoint_id] = float(
                endpoint_rows[endpoint_id]["observed_graphics_mean_mhz"]
            )
            row["%s_gross_energy_j" % endpoint_id] = float(
                endpoint_rows[endpoint_id]["gross_energy_j"]
            )
        duration = float(client["timing_end_unix_s"]) - float(client["timing_start_unix_s"])
        mean_ttft = float(client["mean_ttft_ms"])
        mean_tpot = float(client["mean_tpot_ms"])
        ttft_slo = float(self.settings["slo"]["ttft_ms"])
        tpot_slo = float(self.settings["slo"]["tpot_ms"])
        row.update({
            "completed_requests": int(client["successful_requests"]),
            "output_tokens": int(client["output_tokens_total"]),
            "mean_ttft_ms": mean_ttft, "p99_ttft_ms": client.get("p99_ttft_ms"),
            "mean_tpot_ms": mean_tpot, "p99_tpot_ms": client.get("p99_tpot_ms"),
            "mean_itl_ms": float(client["mean_itl_ms"]),
            "mean_e2e_latency_ms": float(client["mean_e2e_latency_ms"]),
            "total_gpu_gross_energy_j": float(summary["energy_j"]["total"]),
            "joules_per_request": float(summary["joules_per_request"]),
            "joules_per_output_token": float(summary["joules_per_output_token"]),
            "throughput_requests_s": int(client["successful_requests"]) / duration,
            "slo_pass": mean_ttft <= ttft_slo and mean_tpot <= tpot_slo,
            "measurement_valid": bool(audit.get("valid")),
            "hard_gates_json": json.dumps(audit.get("hard_gates", {}), sort_keys=True),
            "error": error,
        })
        return row

    def run(self) -> Path:
        if self.run_dir.exists():
            raise Phase4AError("run directory already exists: %s" % self.run_dir)
        windows_root = self.run_dir / "raw" / "windows"
        windows_root.mkdir(parents=True)
        accepted = self._accepted_frequencies()
        capabilities = self._discover(accepted)
        _write_json(self.run_dir / "raw" / "capabilities.json", {
            key: asdict(value) for key, value in capabilities.items()
        })
        actuator = PerEndpointClockActuator(
            self.backend, capabilities, float(self.settings.get("minimum_dwell_s", 1.0))
        )
        high = {key: values["HIGH"] for key, values in accepted.items()}
        actuator.requested.update(high)
        _, _, pairs = build_registry_and_compatibility(self.config)
        profiles = self.settings["profiles"]
        actions = build_action_space(
            pairs, sorted(accepted), profiles, self.settings["canonical_route"],
            self.settings.get("canonical_extra_profiles", []),
        )
        workloads = [dict(item) for item in self.config["workloads"]]
        repeats = int(self.settings["repeats"])
        _write_json(self.run_dir / "raw" / "measurement_plan.json", {
            "actions": [asdict(action) for action in actions],
            "workloads": workloads, "repeats": repeats,
            "planned_measurements": len(actions) * len(workloads) * repeats,
            "planned_requests": len(actions) * len(workloads) * repeats
                * int(self.settings["requests_per_repeat"]),
            "pruning": self.settings["action_space_pruning"],
        })

        rows: list[dict[str, Any]] = []
        warmup_valid = False
        restoration_valid = False
        error: Optional[str] = None
        try:
            self._set_frequencies(actuator, high, "phase4a_warmup_high")
            self._write_routes(pairs, "phase4a_deterministic_all_pair_warmup")
            warmup_config = _phase3c_window_config(
                self.config, windows_root,
                [{"id": "oracle_warmup", "input_len": 128, "output_len": 16,
                  "count": 4, "rate_rps": 0.5}], high, required_pairs=pairs,
            )
            warmup_config["server_logs"] = {}
            warmup_config["client"]["max_concurrency"] = 1
            Phase3CSubstrateHarness(warmup_config, run_id="warmup").run()
            warmup_valid = bool(_read_json(windows_root / "warmup" / "audit.json").get("valid"))
            if not warmup_valid:
                raise Phase4AError("deterministic all-route warmup audit failed")

            for repeat in range(1, repeats + 1):
                offset = (repeat - 1) * 7 % len(actions)
                ordered_actions = actions[offset:] + actions[:offset]
                for action_index, action in enumerate(ordered_actions):
                    requested = {
                        endpoint_id: accepted[endpoint_id][level]
                        for endpoint_id, level in action.levels.items()
                    }
                    self._set_frequencies(
                        actuator, requested, "phase4a_%s_repeat_%02d" % (action.config_id, repeat)
                    )
                    self._write_routes(
                        [[action.prefill_endpoint_id, action.decode_endpoint_id]],
                        "phase4a_offline_fixed_configuration",
                    )
                    workload_offset = (repeat + action_index) % len(workloads)
                    ordered_workloads = workloads[workload_offset:] + workloads[:workload_offset]
                    for workload in ordered_workloads:
                        window_id = "r%02d-%s-%s" % (repeat, action.config_id, workload["id"])
                        spec = dict(workload)
                        spec["count"] = int(self.settings["requests_per_repeat"])
                        window_config = _phase3c_window_config(
                            self.config, windows_root, [spec], requested,
                            required_pairs=[[action.prefill_endpoint_id, action.decode_endpoint_id]],
                        )
                        # Per-window vLLM logs grow monotonically. Preserve one
                        # complete copy below rather than duplicating them 460 times.
                        window_config["server_logs"] = {}
                        window_config["client"]["max_concurrency"] = 1
                        window_error = None
                        try:
                            Phase3CSubstrateHarness(window_config, run_id=window_id).run()
                        except Exception as exc:
                            window_error = "%s: %s" % (type(exc).__name__, exc)
                        row = self._measurement_row(
                            window_id, workload, action, repeat, requested, window_error
                        )
                        rows.append(row)
                        if not row["measurement_valid"]:
                            raise Phase4AError("invalid measurement window: %s" % window_id)
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
        finally:
            restoration_valid = True
            for endpoint_id in sorted(high):
                row = actuator.actuate(endpoint_id, high[endpoint_id], "phase4a_final_safe_high")
                self.actions.append(row)
                restoration_valid = restoration_valid and row["command_status"] == "success" and row["readback_valid"]

        measurement_fields = (
            "run_id", "window_id", "workload", "input_len", "output_len", "rate_rps",
            "repeat", "config_id", "profile_id", "static_baseline",
            "prefill_endpoint_id", "decode_endpoint_id",
            "P0_requested_freq_mhz", "P1_requested_freq_mhz",
            "D0_requested_freq_mhz", "D1_requested_freq_mhz",
            "P0_observed_freq_mhz", "P1_observed_freq_mhz",
            "D0_observed_freq_mhz", "D1_observed_freq_mhz",
            "completed_requests", "output_tokens", "mean_ttft_ms", "p99_ttft_ms",
            "mean_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "mean_e2e_latency_ms",
            "slo_pass", "P0_gross_energy_j", "P1_gross_energy_j",
            "D0_gross_energy_j", "D1_gross_energy_j", "total_gpu_gross_energy_j",
            "joules_per_request", "joules_per_output_token", "throughput_requests_s",
            "measurement_valid", "hard_gates_json", "error",
        )
        _csv(self.run_dir / "phase4a_measurements.csv", rows, measurement_fields)
        server_log_root = self.run_dir / "raw" / "server_logs"
        server_log_root.mkdir(parents=True, exist_ok=True)
        for value in self.config.get("server_logs", {}).values():
            path = Path(str(value))
            if path.is_file():
                shutil.copyfile(path, server_log_root / path.name)
        _csv(self.run_dir / "raw" / "dvfs_actions.csv", self.actions, (
            "timestamp_unix_s", "endpoint_id", "node", "gpu_id",
            "previous_requested_freq_mhz", "requested_freq_mhz",
            "observed_freq_before_mhz", "observed_freq_after_mhz",
            "command_status", "readback_valid", "transition_readback_latency_s",
            "reason", "error",
        ))
        aggregates = aggregate_measurements(rows, repeats)
        slo = self.settings["slo"]
        oracles = select_oracles(
            aggregates, workloads, float(slo["ttft_ms"]), float(slo["tpot_ms"])
        )
        oracle_fields = (
            "workload", "best_P", "best_D", "best_config_id", "second_best_config_id",
            "baseline_config_id", "P0_freq", "P1_freq", "D0_freq", "D1_freq",
            "energy", "energy_std", "energy_ci95_half_width", "J_per_request",
            "TTFT", "TPOT", "SLO_pass", "energy_vs_static",
            "second_best_energy_gap", "TTFT_headroom_ms", "TPOT_headroom_ms",
            "eligible_configuration_count", "near_optimal_5pct_count",
        )
        _csv(self.run_dir / "phase4a_oracle.csv", oracles, oracle_fields)
        complete = len(rows) == len(actions) * len(workloads) * repeats
        hard_gates = {
            "accepted_phase3d_prerequisites": True,
            "deterministic_all_route_warmup": warmup_valid,
            "complete_measurement_plan": complete,
            "all_measurement_audits_valid": complete and all(row["measurement_valid"] for row in rows),
            "oracle_for_every_workload": len(oracles) == len(workloads),
            "safe_high_restoration": restoration_valid,
            "no_online_feedback_or_models": True,
            "no_unresolved_error": error is None,
        }
        valid = all(hard_gates.values())
        effect_summary = _effect_summary(aggregates)
        unique_best = sorted({str(item["best_config_id"]) for item in oracles})
        savings = [float(item["energy_vs_static"]) for item in oracles]
        near_counts = [int(item["near_optimal_5pct_count"]) for item in oracles]
        answers = {
            "best_configuration_differs_across_workloads": len(unique_best) > 1,
            "unique_best_configurations": unique_best,
            "oracle_savings_vs_static": {
                "minimum_fraction": min(savings) if savings else None,
                "mean_fraction": _mean(savings) if savings else None,
                "maximum_fraction": max(savings) if savings else None,
            },
            "largest_measured_main_effect": effect_summary.get("largest_measured_main_effect"),
            "optimum_shape": {
                "near_optimal_5pct_counts": near_counts,
                "multiple_close_configurations_for_every_workload": bool(near_counts)
                    and all(value > 1 for value in near_counts),
            },
        }
        summary = {
            "phase": "4A_small_empirical_offline_oracle", "valid": valid,
            "hard_gates": hard_gates, "error": error,
            "topology": "Neptune 2xL40S prefill + IO 2xL4 decode, TP1",
            "frequency_levels_mhz": accepted,
            "slo": {"statistic": "per-repeat mean", **slo},
            "action_space": {
                "measured_configurations": len(actions),
                "full_cartesian_configurations": 4 * (3 ** 4),
                "pruning": self.settings["action_space_pruning"],
            },
            "workload_count": len(workloads), "repeats": repeats,
            "measurement_count": len(rows), "planned_measurement_count": len(actions) * len(workloads) * repeats,
            "planned_request_count": len(actions) * len(workloads) * repeats * int(self.settings["requests_per_repeat"]),
            "configuration_aggregates": aggregates,
            "oracles": oracles, "knob_effects": effect_summary,
            "answers": answers,
            "models_trained_or_used": [],
            "claim_boundary": "best measured SLO-feasible configuration in the pruned action space; not a theoretical optimum",
            "ready_for_phase4b": valid and all(row["eligible_configuration_count"] >= 2 for row in oracles),
        }
        _write_json(self.run_dir / "phase4a_summary.json", summary)
        (self.run_dir / "phase4a_summary.md").write_text(
            render_summary_markdown(summary), encoding="utf-8"
        )
        if not valid:
            raise Phase4AError(error or "Phase 4A hard-gate audit failed")
        print(self.run_dir.as_posix())
        return self.run_dir


def load_config(path: Path) -> dict[str, Any]:
    config = load_phase3c_config(path)
    settings = config.get("phase4a")
    if not isinstance(settings, dict):
        raise Phase4AError("Phase 4A config requires a phase4a object")
    if int(settings.get("repeats", 0)) < 3:
        raise Phase4AError("Phase 4A requires at least three repeats")
    if int(settings.get("requests_per_repeat", 0)) < 1:
        raise Phase4AError("requests_per_repeat must be positive")
    endpoints = {str(item["endpoint_id"]): item for item in config["endpoints"]}
    if set(endpoints) != {"P0", "P1", "D0", "D1"}:
        raise Phase4AError("Phase 4A requires exactly P0/P1/D0/D1")
    if any(endpoints[key]["node"] != "neptune" for key in ("P0", "P1")):
        raise Phase4AError("Phase 4A prefill endpoints must be fixed to Neptune")
    if any(endpoints[key]["node"] != "io" for key in ("D0", "D1")):
        raise Phase4AError("Phase 4A decode endpoints must be fixed to IO")
    if any(int(item["tp_degree"]) != 1 for item in endpoints.values()):
        raise Phase4AError("Phase 4A fixes every endpoint at TP1")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    Phase4AOracleHarness(load_config(Path(args.config)), args.run_id).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
