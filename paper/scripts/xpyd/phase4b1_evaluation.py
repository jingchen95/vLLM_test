"""Phase 4B.1 minimal context-aware safe routing-probe smoke.

PASSIVE_FULL preserves the accepted Phase 4B controller. ACTIVE_FULL replaces
only routing ranking with context-route observations of total four-GPU gross
window energy per logical request. The feedback-only DVFS implementation is
reused unchanged and is merely skipped during an explicit probe window.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Optional, Sequence

from xpyd.phase3c_substrate import _read_json, _write_json, build_registry_and_compatibility
from xpyd.phase3d_control import PerEndpointClockActuator, _csv
from xpyd.phase4b_evaluation import (
    ControllerContext,
    Phase4BError,
    Phase4BEvaluationHarness,
    load_config as load_phase4b_config,
)
from xpyd.route_probing import (
    ContextRouteCostStore,
    RouteProbeDecision,
    SafeRouteProber,
    probe_safe_routes,
)


class Phase4B1Error(Phase4BError):
    """A fail-closed Phase 4B.1 configuration, measurement, or audit error."""


def _route_key(route: Sequence[str]) -> str:
    return "%s->%s" % (str(route[0]), str(route[1]))


def _pressure_guard_dry_run(all_pairs: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Exercise the negative probe gate without changing runtime state."""
    costs = ContextRouteCostStore(alpha=1.0)
    costs.observe(
        "pressure_guard", ["P0", "D0"],
        total_system_gross_energy_j=200.0, logical_requests=2,
        timestamp_s=1.0, sequence=1,
    )
    decision = SafeRouteProber(costs).choose(
        "pressure_guard", compatible_routes=all_pairs,
        eligible_routes=all_pairs, probe_safe_routes=(),
        now_s=2.0, sequence=2,
    )
    return {
        "input": "all unseen alternatives marked probe-unsafe by pressure gate",
        "decision": asdict(decision),
        "valid": not decision.probe and decision.mode == "EXPLOIT_RECENT",
        "physical_pressure_induced": False,
    }


class Phase4B1Harness(Phase4BEvaluationHarness):
    """Physical PASSIVE_FULL versus ACTIVE_FULL safe-probing smoke."""

    def __init__(
        self, config: Mapping[str, Any], run_id: Optional[str] = None,
        backend: Optional[Any] = None,
    ) -> None:
        super().__init__(config, run_id=run_id, backend=backend, smoke=True)
        self.probe_settings = dict(self.config["phase4b1"])
        root = os.path.expandvars(str(self.probe_settings["output_root"]))
        self.run_dir = Path(root) / self.run_id

    def run(self) -> Path:
        if self.run_dir.exists():
            raise Phase4B1Error("run directory already exists: %s" % self.run_dir)
        stationary_path = Path(os.path.expandvars(str(
            self.probe_settings["accepted_stationary_audit"]
        )))
        if not stationary_path.is_file():
            raise Phase4B1Error("missing accepted Phase 4B stationary audit")
        stationary_raw = stationary_path.read_bytes()
        stationary_audit = json.loads(stationary_raw)
        if not stationary_audit.get("valid") or stationary_audit.get("smoke"):
            raise Phase4B1Error("Phase 4B.1 requires a valid non-smoke stationary audit")

        windows_root = self.run_dir / "raw" / "windows"
        windows_root.mkdir(parents=True)
        reference = self._reference()
        accepted = self._accepted_frequencies(reference)
        capabilities = self._discover_exact(accepted)
        _write_json(self.run_dir / "raw" / "capabilities.json", {
            key: asdict(value) for key, value in capabilities.items()
        })
        _write_json(self.run_dir / "raw" / "accepted_stationary_reference.json", {
            "path": stationary_path.as_posix(),
            "sha256": hashlib.sha256(stationary_raw).hexdigest(),
            "valid": True,
        })
        _write_json(self.run_dir / "raw" / "frozen_phase4a_reference.json", {
            "path": reference["path"], "sha256": reference["sha256"],
            "oracles": reference["oracles"],
        })

        actuator = PerEndpointClockActuator(
            self.backend, capabilities,
            float(self.config["phase3d"].get("minimum_dwell_s", 1.0)),
        )
        high = {
            endpoint_id: values["HIGH"] for endpoint_id, values in accepted.items()
        }
        actuator.requested.update(high)
        _, _, all_pairs = build_registry_and_compatibility(self.config)
        all_routes = [list(item) for item in all_pairs]
        policies = list(self.probe_settings["policies"])
        contexts = list(self.probe_settings["smoke_contexts"])
        windows_per_context = int(self.probe_settings["smoke_windows_per_context"])
        request_count = int(self.probe_settings["requests_per_window"])
        workload_by_id = {
            str(item["id"]): dict(item) for item in self.config["workloads"]
        }
        blocks = [(policy, context) for policy in policies for context in contexts]
        random.Random(int(self.probe_settings.get("order_seed", 47))).shuffle(blocks)
        _write_json(self.run_dir / "raw" / "phase4b1_plan.json", {
            "phase": "4B.1", "smoke": True, "blocks": blocks,
            "windows_per_context": windows_per_context,
            "requests_per_window": request_count,
            "cost_semantics": (
                "total gross energy of P0+P1+D0+D1 during the control window "
                "divided by logical requests; window-amortized system cost, "
                "not request-level physical energy attribution"
            ),
            "models": [],
        })

        costs = ContextRouteCostStore(
            alpha=float(self.probe_settings["route_cost_ewma_alpha"]),
            maximum_age_s=float(self.probe_settings["route_cost_maximum_age_s"]),
            maximum_age_windows=int(
                self.probe_settings["route_cost_maximum_age_windows"]
            ),
        )
        prober = SafeRouteProber(
            costs,
            minimum_probe_interval_windows=int(
                self.probe_settings["minimum_probe_interval_windows"]
            ),
        )
        rows: list[dict[str, Any]] = []
        request_rows: list[dict[str, Any]] = []
        cost_rows: list[dict[str, Any]] = []
        warmup_audits: list[dict[str, Any]] = []
        restoration_valid = False
        error: Optional[str] = None
        try:
            global_context = self._new_context(capabilities)
            self._actuate(
                actuator, global_context, high, "phase4b1_global_warmup_high",
                controller_action=False, policy="WARMUP", workload="all",
                iteration_id="phase4b1-global-warmup",
            )
            self._write_routes(all_routes, "phase4b1_global_all_route_warmup")
            global_dir = self._run_window(
                "phase4b1-global-warmup",
                {"id": "phase4b1_global_warmup", "input_len": 128,
                 "output_len": 16, "rate_rps": 0.5},
                4, high, all_routes, windows_root, require_all_pairs=True,
            )
            global_audit = _read_json(global_dir / "audit.json")
            warmup_audits.append({"window_id": "phase4b1-global-warmup", **global_audit})
            if not global_audit.get("valid"):
                raise Phase4B1Error("Phase 4B.1 global warmup failed")

            for block_index, (policy, context_key) in enumerate(blocks, 1):
                workload = workload_by_id[context_key]
                context = self._new_context(capabilities)
                warmup_id = "p41-warm-%02d-%s-%s" % (
                    block_index, policy.lower(), context_key
                )
                self._actuate(
                    actuator, context, high, "phase4b1_feedback_warmup_high",
                    controller_action=False, policy=policy, workload=context_key,
                    iteration_id=warmup_id,
                )
                context.selected_pairs = copy.deepcopy(all_routes)
                self._write_routes(all_routes, "phase4b1_all_route_safety_warmup")
                warmup_dir = self._run_window(
                    warmup_id, workload, 4, high, all_routes,
                    windows_root, require_all_pairs=True,
                )
                warmup_audit = _read_json(warmup_dir / "audit.json")
                warmup_audits.append({"window_id": warmup_id, **warmup_audit})
                if not warmup_audit.get("valid"):
                    raise Phase4B1Error("Phase 4B.1 block warmup failed: %s" % warmup_id)
                self._observe_window(
                    warmup_id, warmup_dir, context.registry, context.telemetry
                )

                for sequence in range(1, windows_per_context + 1):
                    iteration_id = "p41-b%02d-w%02d-%s-%s" % (
                        block_index, sequence, policy.lower(), context_key
                    )
                    route_decision: Optional[RouteProbeDecision] = None
                    probe_safety: dict[str, list[str]] = {}
                    before_frequencies = dict(context.target_frequencies)
                    if policy == "ACTIVE_FULL":
                        now = time.time()
                        evaluations = context.scheduler.evaluate_routes(
                            now_s=now, workload_context_key=context_key
                        )
                        eligible_routes = [
                            [item.prefill_endpoint_id, item.decode_endpoint_id]
                            for item in evaluations if item.eligible
                        ]
                        safe_routes, probe_safety = probe_safe_routes(
                            context, evaluations, now,
                            float(self.probe_settings["probe_headroom_fraction"]),
                        )
                        route_decision = prober.choose(
                            context_key, compatible_routes=all_routes,
                            eligible_routes=eligible_routes,
                            probe_safe_routes=safe_routes,
                            now_s=now, sequence=sequence,
                        )
                        pairs = self._decide(
                            policy, context_key, iteration_id, context,
                            capabilities, actuator, all_routes,
                            measured_iteration=True,
                            feedback_context_key=context_key,
                            route_override=route_decision.route,
                            freeze_dvfs=route_decision.freeze_dvfs,
                            routing_decision_mode=route_decision.mode,
                            routing_score_basis="context_route_total_four_gpu_window_joules_per_request",
                            route_costs_json=json.dumps(
                                costs.as_dict(context_key, all_routes, now, sequence),
                                sort_keys=True,
                            ),
                        )
                    else:
                        pairs = self._decide(
                            policy, context_key, iteration_id, context,
                            capabilities, actuator, all_routes,
                            measured_iteration=True,
                            feedback_context_key=context_key,
                            routing_decision_mode="PASSIVE_ENDPOINT_EWMA",
                            routing_score_basis="endpoint_window_gross_energy_per_request",
                        )
                    control = self.control_rows[-1]
                    control["probe_safety_reasons_json"] = json.dumps(
                        probe_safety, sort_keys=True
                    )
                    control["probe_frequency_state_unchanged"] = (
                        not bool(control["route_probe"])
                        or before_frequencies == context.target_frequencies
                    )
                    window_dir = self._run_window(
                        iteration_id, workload, request_count,
                        context.target_frequencies, pairs, windows_root,
                    )
                    row, observed_requests = self._measurement_row(
                        iteration_id, policy, workload, sequence,
                        context, accepted, reference, window_dir,
                    )
                    row.update({
                        "context_key": context_key,
                        "sequence": sequence,
                        "routing_decision_mode": control["routing_decision_mode"],
                        "routing_score_basis": control["routing_score_basis"],
                        "route_probe": control["route_probe"],
                        "dvfs_frozen_for_probe": control["dvfs_frozen_for_probe"],
                        "probe_frequency_state_unchanged": control[
                            "probe_frequency_state_unchanged"
                        ],
                    })
                    rows.append(row)
                    request_rows.extend(observed_requests)
                    if not row["measurement_valid"]:
                        raise Phase4B1Error("Phase 4B.1 window failed: %s" % iteration_id)
                    if policy == "ACTIVE_FULL":
                        if len(pairs) != 1:
                            raise Phase4B1Error("ACTIVE_FULL requires one exact route")
                        total_energy = float(row["total_gpu_gross_energy_j"])
                        logical_requests = int(row["completed_requests"])
                        snapshot = costs.observe(
                            context_key, pairs[0],
                            total_system_gross_energy_j=total_energy,
                            logical_requests=logical_requests,
                            timestamp_s=float(row["timing_end_unix_s"]),
                            sequence=sequence,
                        )
                        cost_rows.append({
                            "window_id": iteration_id,
                            "context_key": context_key,
                            "route": _route_key(pairs[0]),
                            "decision_mode": route_decision.mode if route_decision else None,
                            "total_four_gpu_gross_energy_j": total_energy,
                            "logical_requests": logical_requests,
                            "observed_system_joules_per_request": (
                                total_energy / logical_requests
                            ),
                            "ewma_system_joules_per_request": (
                                snapshot.ewma_system_joules_per_request
                            ),
                            "sample_count": snapshot.sample_count,
                            "timestamp_s": snapshot.last_observation_s,
                            "sequence": sequence,
                            "window_amortized_not_request_attribution": True,
                        })
                    self._observe_window(
                        iteration_id, window_dir, context.registry, context.telemetry
                    )
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
        finally:
            restore_context = self._new_context(capabilities)
            try:
                self._actuate(
                    actuator, restore_context, high,
                    "phase4b1_final_safe_high_restoration",
                    controller_action=False, policy="RESTORE", workload="all",
                    iteration_id="phase4b1-final-restore",
                )
                restoration_valid = all(
                    actuator.read(endpoint_id).graphics_clock_mhz == target
                    for endpoint_id, target in high.items()
                )
            except Exception:
                restoration_valid = False

        pressure_guard = _pressure_guard_dry_run(all_routes)
        _write_json(self.run_dir / "raw" / "pressure_guard_dry_run.json", pressure_guard)
        for control in self.control_rows:
            control.setdefault("probe_safety_reasons_json", "{}")
            control.setdefault("probe_frequency_state_unchanged", True)

        _csv(self.run_dir / "phase4b1_results.csv", rows, tuple(rows[0]) if rows else ())
        _csv(
            self.run_dir / "phase4b1_route_cost_observations.csv", cost_rows,
            tuple(cost_rows[0]) if cost_rows else (),
        )
        _csv(self.run_dir / "phase4b1_control_actions.csv", self.control_rows, (
            "control_timestamp", "control_iteration_id", "policy", "workload",
            "feedback_context_key", "measured_iteration", "telemetry_fresh",
            "stale_endpoint_ids_json", "stale_endpoint_ages_s_json",
            "fallback_reason", "selected_pairs_json", "requested_frequencies_json",
            "routing_changed", "dvfs_action_count", "routing_decision_mode",
            "routing_score_basis", "route_probe", "dvfs_frozen_for_probe",
            "route_cost_context", "route_costs_json", "probe_safety_reasons_json",
            "probe_frequency_state_unchanged", "candidate_evaluations_json",
            "dvfs_recommendations_json",
        ))
        _csv(self.run_dir / "raw" / "requests.csv", request_rows, (
            "window_id", "policy", "workload", "repeat", "request_id",
            "prefill_endpoint_id", "decode_endpoint_id", "ttft_ms", "tpot_ms",
            "mean_itl_ms", "e2e_latency_ms",
        ))
        _csv(self.run_dir / "raw" / "actuator_actions.csv", self.actuator_rows, (
            "timestamp_unix_s", "policy", "workload", "control_iteration_id",
            "controller_action", "endpoint_id", "node", "gpu_id",
            "previous_requested_freq_mhz", "requested_freq_mhz",
            "observed_freq_before_mhz", "observed_freq_after_mhz",
            "command_status", "readback_valid", "transition_readback_latency_s",
            "reason", "error",
        ))
        _csv(self.run_dir / "raw" / "endpoint_telemetry.csv", self.telemetry_rows, (
            "window_id", "endpoint_id", "timestamp_s", "health", "queue",
            "kv_cache_usage_frac", "ttft_ms", "tpot_ms", "power_w", "energy_j",
            "observed_frequency_mhz", "telemetry_age_s_at_observation",
            "measurement_valid",
        ))

        expected = len(blocks) * windows_per_context
        active_rows = [item for item in rows if item["policy"] == "ACTIVE_FULL"]
        passive_rows = [item for item in rows if item["policy"] == "PASSIVE_FULL"]
        active_controls = [
            item for item in self.control_rows if item["policy"] == "ACTIVE_FULL"
        ]
        passive_controls = [
            item for item in self.control_rows if item["policy"] == "PASSIVE_FULL"
        ]
        observed_routes = {_route_key(json.loads(item["selected_pairs_json"])[0]) for item in active_controls}
        passive_routes = {_route_key(json.loads(item["selected_pairs_json"])[0]) for item in passive_controls}
        probe_controls = [item for item in active_controls if item["route_probe"]]
        exploit_controls = [item for item in active_controls if not item["route_probe"]]
        pressure_or_slo_safe = all(
            not json.loads(item["probe_safety_reasons_json"])[
                _route_key(json.loads(item["selected_pairs_json"])[0])
            ]
            for item in probe_controls
        ) if probe_controls else False
        cost_formula_valid = bool(cost_rows) and all(
            abs(
                float(item["observed_system_joules_per_request"])
                - float(item["total_four_gpu_gross_energy_j"])
                / int(item["logical_requests"])
            ) < 1e-9
            for item in cost_rows
        )
        inherited_window_gates = [
            json.loads(item["hard_gates_json"]) for item in rows
        ]
        active_energy = sum(float(item["total_gpu_gross_energy_j"]) for item in active_rows)
        passive_energy = sum(float(item["total_gpu_gross_energy_j"]) for item in passive_rows)
        overhead = (
            (active_energy - passive_energy) / passive_energy
            if passive_energy > 0 else None
        )
        hard_gates = {
            "accepted_full_stationary_audit": bool(stationary_audit.get("valid")),
            "complete_smoke_plan": len(rows) == expected,
            "all_measurements_valid": len(rows) == expected and all(
                item["measurement_valid"] for item in rows
            ),
            "all_warmups_valid": bool(warmup_audits) and all(
                item.get("valid") for item in warmup_audits
            ),
            "passive_lock_in_reproduced": len(passive_routes) == 1,
            "all_compatible_routes_observed_by_active": observed_routes == {
                _route_key(item) for item in all_routes
            },
            "stale_route_cost_refreshed": any(
                item["decision_mode"] == "PROBE_STALE"
                and int(item["sample_count"]) >= 2
                for item in cost_rows
            ),
            "context_specific_system_cost_formula": cost_formula_valid,
            "context_cost_updates_are_isolated": bool(cost_rows) and all(
                item["context_key"] == next(
                    row["workload"] for row in active_rows
                    if row["window_id"] == item["window_id"]
                )
                for item in cost_rows
            ),
            "exact_request_ids_tokens_and_routes": bool(inherited_window_gates) and all(
                gates.get("logical_request_id")
                and gates.get("requested_output_tokens")
                and gates.get("endpoint_assignment")
                and gates.get("explicit_compatibility")
                for gates in inherited_window_gates
            ),
            "energy_windows_and_frequency_readback_valid": bool(inherited_window_gates) and all(
                gates.get("nvml_energy_windows")
                and gates.get("fixed_clocks")
                and gates.get("no_invalidating_thermal_or_hw_slowdown")
                for gates in inherited_window_gates
            ),
            "all_probes_passed_strict_safety_gate": pressure_or_slo_safe,
            "probing_suppressed_under_pressure_dry_run": pressure_guard["valid"],
            "probe_windows_freeze_dvfs": bool(probe_controls) and all(
                item["dvfs_frozen_for_probe"]
                and item["probe_frequency_state_unchanged"]
                and int(item["dvfs_action_count"]) == 0
                for item in probe_controls
            ),
            "dvfs_resumed_outside_probe_windows": bool(exploit_controls) and any(
                int(item["dvfs_action_count"]) > 0 for item in exploit_controls
            ),
            "all_requests_meet_slo": bool(request_rows) and all(
                float(item["ttft_ms"]) <= float(self.settings["slo"]["ttft_ms"])
                and float(item["tpot_ms"]) <= float(self.settings["slo"]["tpot_ms"])
                for item in request_rows
            ),
            "no_fallback_or_correctness_regression": all(
                not item.get("fallback_reason") for item in self.control_rows
            ),
            "safe_high_restoration": restoration_valid,
            "no_models_or_offline_route_lookup": True,
            "no_unresolved_error": error is None,
        }
        valid = all(hard_gates.values())
        audit = {
            "phase": "4B.1_minimal_context_aware_safe_probing",
            "valid": valid, "smoke": True, "hard_gates": hard_gates,
            "error": error, "warmup_audits": warmup_audits,
            "accepted_stationary_audit": stationary_path.as_posix(),
            "models_trained_or_used": [],
            "claim_boundary": (
                "route cost is window-amortized total four-GPU gross energy per "
                "logical request, not request-level physical energy attribution"
            ),
        }
        _write_json(self.run_dir / "phase4b1_audit.json", audit)
        summary = {
            "phase": audit["phase"], "valid": valid, "smoke": True,
            "hard_gates": hard_gates, "error": error,
            "active_observed_routes": sorted(observed_routes),
            "passive_observed_routes": sorted(passive_routes),
            "active_probe_window_count": len(probe_controls),
            "active_exploit_window_count": len(exploit_controls),
            "active_energy_j": active_energy,
            "passive_energy_j": passive_energy,
            "probing_overhead_vs_passive": overhead,
            "slo_violation_count": sum(
                float(item["ttft_ms"]) > float(self.settings["slo"]["ttft_ms"])
                or float(item["tpot_ms"]) > float(self.settings["slo"]["tpot_ms"])
                for item in request_rows
            ),
            "decision_gate": (
                "READY_FOR_DYNAMIC_EVALUATION" if valid else "BLOCKED"
            ),
            "models_trained_or_used": [],
        }
        _write_json(self.run_dir / "phase4b1_summary.json", summary)
        lines = [
            "# XpYd Phase 4B.1 safe-probing smoke", "",
            "Verdict: **%s**" % ("PASS" if valid else "FAIL"), "",
            "Route cost is window-amortized total four-GPU gross energy per logical request; it is not request-level physical attribution.", "",
            "- PASSIVE_FULL routes: `%s`" % sorted(passive_routes),
            "- ACTIVE_FULL routes: `%s`" % sorted(observed_routes),
            "- ACTIVE probe/exploit windows: `%d/%d`" % (
                len(probe_controls), len(exploit_controls)
            ),
            "- Probing overhead versus passive: `%s`" % overhead,
            "- SLO violations: `%d`" % summary["slo_violation_count"],
            "- Decision: **%s**" % summary["decision_gate"], "",
            "Hard gates: `%s`" % json.dumps(hard_gates, sort_keys=True), "",
        ]
        if error:
            lines.extend(["Failure: `%s`" % error, ""])
        (self.run_dir / "phase4b1_summary.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        if not valid:
            raise Phase4B1Error(error or "Phase 4B.1 smoke hard gates failed")
        print(self.run_dir.as_posix())
        return self.run_dir


def load_config(path: Path) -> dict[str, Any]:
    config = load_phase4b_config(path)
    settings = config.get("phase4b1")
    if not isinstance(settings, dict):
        raise Phase4B1Error("Phase 4B.1 config requires a phase4b1 object")
    if tuple(settings.get("policies", ())) != ("PASSIVE_FULL", "ACTIVE_FULL"):
        raise Phase4B1Error("Phase 4B.1 must compare PASSIVE_FULL and ACTIVE_FULL")
    workload_ids = {str(item["id"]) for item in config["workloads"]}
    contexts = set(map(str, settings.get("contexts", ())))
    smoke_contexts = set(map(str, settings.get("smoke_contexts", ())))
    if contexts != workload_ids or not smoke_contexts or not smoke_contexts <= contexts:
        raise Phase4B1Error("Phase 4B.1 contexts must cover configured workloads")
    if int(settings.get("smoke_windows_per_context", 0)) < 7:
        raise Phase4B1Error("Phase 4B.1 smoke needs route coverage, exploit, and stale refresh")
    fraction = float(settings.get("probe_headroom_fraction", 0.0))
    if not 0.0 < fraction < 1.0:
        raise Phase4B1Error("probe_headroom_fraction must be in (0, 1)")
    route_cost_age = float(settings.get("route_cost_maximum_age_s", 0.0))
    latency_age = float(config["phase3d"]["feedback"]["telemetry_max_age_s"])
    if not 0.0 < route_cost_age < latency_age:
        raise Phase4B1Error(
            "route-cost freshness must expire before endpoint latency safety telemetry"
        )
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    Phase4B1Harness(load_config(Path(args.config)), run_id=args.run_id).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
