"""Fail-closed Stage B physical validation for realistic mixed request streams.

Runs B1 concurrency-2, then B2 concurrency-4, then a B3 dynamic mixed trace.
Every endpoint remains at validated HIGH, routing is four-pair round-robin, and
no controller, probing policy, predictor, or model is used. A failed stage
prevents every later stage from running.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib.request import urlopen

from xpyd.phase3c_substrate import (
    Phase3CSubstrateHarness,
    _read_json,
    _write_json,
    load_config as load_runtime_config,
)
from xpyd.workload_harness import build_trace, load_config as load_workload_config


class StageBError(RuntimeError):
    """A Stage B configuration, physical run, or hard gate failed."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_proxy_state(proxy_uri: str) -> dict[str, Any]:
    with urlopen(
        "%s/xpyd/state" % proxy_uri.rstrip("/"), timeout=5.0
    ) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise StageBError("proxy state is not a JSON object")
    return value


def load_configs(
    runtime_path: Path, workload_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = load_runtime_config(runtime_path)
    workload = load_workload_config(workload_path)
    endpoints = {str(item["endpoint_id"]): item for item in runtime["endpoints"]}
    if set(endpoints) != {"P0", "P1", "D0", "D1"}:
        raise StageBError("Stage B requires exactly P0/P1/D0/D1")
    if any(str(endpoints[key]["node"]) != "neptune" for key in ("P0", "P1")):
        raise StageBError("Stage B fixes P0/P1 to Neptune")
    if any(str(endpoints[key]["node"]) != "io" for key in ("D0", "D1")):
        raise StageBError("Stage B fixes D0/D1 to io")
    if any(int(item["tp_degree"]) != 1 for item in endpoints.values()):
        raise StageBError("Stage B fixes every endpoint at TP1")
    expected_clocks = {"P0": 2520, "P1": 2520, "D0": 1500, "D1": 1500}
    observed_clocks = {
        key: int(runtime["fixed_clocks"][key]["graphics_mhz"])
        for key in expected_clocks
    }
    if observed_clocks != expected_clocks:
        raise StageBError("Stage B requires validated HIGH frequencies")
    pairs = {
        (str(item["prefill_endpoint_id"]), str(item["decode_endpoint_id"]))
        for item in runtime["compatible_pairs"] if item.get("supported")
    }
    if pairs != {
        ("P0", "D0"), ("P0", "D1"), ("P1", "D0"), ("P1", "D1")
    }:
        raise StageBError("Stage B requires all four explicit compatible pairs")
    hardware = [
        dict(item) for item in workload["proposed_validation_matrix"]
        if item.get("hardware")
    ]
    if [item.get("id") for item in hardware] != [
        "B1_concurrency_2_smoke", "B2_concurrency_4_smoke",
        "B3_dynamic_harness_smoke",
    ]:
        raise StageBError("Stage B requires the ordered B1/B2/B3 matrix")
    if [int(item.get("max_concurrency", 0)) for item in hardware] != [2, 4, 4]:
        raise StageBError("Stage B concurrency progression must be 2, 4, 4")
    if any(item.get("control") != "disabled" for item in hardware):
        raise StageBError("Stage B forbids controller decisions")
    return runtime, workload


def audit_stage(
    stage: Mapping[str, Any], trace_rows: Sequence[Mapping[str, Any]],
    stage_dir: Path, proxy_before: Mapping[str, Any],
    proxy_after: Mapping[str, Any],
) -> dict[str, Any]:
    base_audit = _read_json(stage_dir / "audit.json")
    client = _read_json(stage_dir / "client" / "summary.json")
    requests = _read_csv(stage_dir / "requests.csv")
    load_events = _read_jsonl(stage_dir / "client" / "load_events.jsonl")
    expected_by_id = {str(item["request_id"]): item for item in trace_rows}
    observed_by_id = {str(item["request_id"]): item for item in requests}
    context_exact = set(expected_by_id) == set(observed_by_id) and all(
        observed_by_id[request_id].get("workload_class")
        == str(expected["workload_class"])
        and observed_by_id[request_id].get("segment") == str(expected["segment"])
        for request_id, expected in expected_by_id.items()
    )
    segments = sorted({str(item["segment"]) for item in trace_rows})
    mixed_segments = all(
        len({
            str(item["workload_class"]) for item in trace_rows
            if str(item["segment"]) == segment
        }) > 1 for segment in segments
    )
    max_concurrency = int(stage["max_concurrency"])
    maximum_client_in_flight = int(
        client.get("maximum_client_in_flight_requests", -1)
    )
    event_counts = {
        event: sum(item.get("event") == event for item in load_events)
        for event in ("TRACE_ARRIVAL", "CLIENT_ADMIT", "REQUEST_COMPLETE")
    }
    final_event = load_events[-1] if load_events else {}
    accepted_delta = int(proxy_after.get("accepted_requests_total", -1)) - int(
        proxy_before.get("accepted_requests_total", -1)
    )
    completed_delta = int(proxy_after.get("completed_requests_total", -1)) - int(
        proxy_before.get("completed_requests_total", -1)
    )
    failed_delta = int(proxy_after.get("failed_requests_total", -1)) - int(
        proxy_before.get("failed_requests_total", -1)
    )
    stage_id = str(stage["id"])
    dynamic_segments_valid = (
        stage_id != "B3_dynamic_harness_smoke"
        or segments == ["decode_period", "light_mix", "prefill_burst", "recovery_mix"]
    )
    hard_gates = {
        "accepted_phase3c_measurement_audit": bool(base_audit.get("valid")),
        "exact_request_context_labels": context_exact,
        "mixed_request_population": mixed_segments,
        "client_concurrency_reached_and_bounded": maximum_client_in_flight
        == max_concurrency,
        "client_admission_events_complete": all(
            count == len(trace_rows) for count in event_counts.values()
        ),
        "client_load_drained": (
            int(final_event.get("client_waiting_requests", -1)) == 0
            and int(final_event.get("client_in_flight_requests", -1)) == 0
        ),
        "admission_delay_measured": all(
            row.get("client_admission_delay_ms") not in (None, "")
            for row in requests
        ),
        "proxy_request_accounting": (
            accepted_delta == len(trace_rows)
            and completed_delta == len(trace_rows)
            and failed_delta == 0
            and int(proxy_after.get("in_flight_requests", -1)) == 0
        ),
        "proxy_concurrency_observed": int(
            proxy_after.get("maximum_in_flight_requests", -1)
        ) >= max_concurrency,
        "dynamic_segment_contract": dynamic_segments_valid,
        "no_controller_or_gpu_policy_change": (
            stage.get("control") == "disabled"
            and stage.get("frequencies") == "all validated HIGH"
            and stage.get("routing")
            == "round-robin over four validated pairs"
        ),
    }
    return {
        "stage": stage_id,
        "scenario": str(stage["scenario"]),
        "valid": all(hard_gates.values()),
        "request_count": len(trace_rows),
        "max_concurrency": max_concurrency,
        "segments": segments,
        "workload_class_counts": client.get("workload_class_counts", {}),
        "maximum_client_waiting_requests": client.get(
            "maximum_client_waiting_requests"
        ),
        "maximum_client_in_flight_requests": maximum_client_in_flight,
        "mean_client_admission_delay_ms": client.get(
            "mean_client_admission_delay_ms"
        ),
        "p99_client_admission_delay_ms": client.get(
            "p99_client_admission_delay_ms"
        ),
        "proxy_before": dict(proxy_before),
        "proxy_after": dict(proxy_after),
        "proxy_deltas": {
            "accepted": accepted_delta,
            "completed": completed_delta,
            "failed": failed_delta,
        },
        "client_load_event_counts": event_counts,
        "phase3c_hard_gates": base_audit.get("hard_gates", {}),
        "hard_gates": hard_gates,
        "claim_boundary": (
            "concurrent mixed-workload substrate validation only; no routing, "
            "DVFS, energy-optimization, or SLO-policy claim"
        ),
    }


class StageBPhysicalHarness:
    def __init__(
        self, runtime: Mapping[str, Any], workload: Mapping[str, Any],
        run_id: Optional[str] = None,
        substrate_factory: Callable[..., Any] = Phase3CSubstrateHarness,
        proxy_state_reader: Callable[[str], dict[str, Any]] = read_proxy_state,
    ) -> None:
        self.runtime = copy.deepcopy(dict(runtime))
        self.workload = copy.deepcopy(dict(workload))
        self.run_id = run_id or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        self.run_dir = Path(os.path.expandvars(str(workload["output_root"]))) / self.run_id
        self.substrate_factory = substrate_factory
        self.proxy_state_reader = proxy_state_reader

    def run(self) -> Path:
        if self.run_dir.exists():
            raise StageBError("Stage B run directory already exists: %s" % self.run_dir)
        self.run_dir.mkdir(parents=True)
        stages = [
            dict(item) for item in self.workload["proposed_validation_matrix"]
            if item.get("hardware")
        ]
        _write_json(self.run_dir / "stageb_plan.json", {
            "schema_version": 1,
            "run_id": self.run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "stages": stages,
            "topology": {"P0": "neptune", "P1": "neptune", "D0": "io", "D1": "io"},
            "frequencies_mhz": {"P0": 2520, "P1": 2520, "D0": 1500, "D1": 1500},
            "controller": "disabled",
            "models": [],
            "gate_order": "B1 pass -> B2 pass -> B3",
        })
        audits = []
        error = None
        for stage in stages:
            stage_id = str(stage["id"])
            trace_rows = build_trace(
                self.workload, str(stage["scenario"]),
                "%s-%s" % (self.run_id, stage_id),
            )
            stage_config = copy.deepcopy(self.runtime)
            stage_config["output_root"] = self.run_dir.as_posix()
            stage_config["client"]["max_concurrency"] = int(
                stage["max_concurrency"]
            )
            stage_config["coverage_policy"] = {
                "required_endpoint_ids": ["P0", "P1", "D0", "D1"],
                "required_pairs": [
                    ["P0", "D0"], ["P0", "D1"],
                    ["P1", "D0"], ["P1", "D1"],
                ],
            }
            before = self.proxy_state_reader(str(stage_config["proxy_uri"]))
            try:
                self.substrate_factory(
                    stage_config, run_id=stage_id, trace_rows=trace_rows
                ).run()
                after = self.proxy_state_reader(str(stage_config["proxy_uri"]))
                audit = audit_stage(
                    stage, trace_rows, self.run_dir / stage_id, before, after
                )
                audits.append(audit)
                _write_json(self.run_dir / (stage_id + "_audit.json"), audit)
                if not audit["valid"]:
                    raise StageBError("%s hard gates failed" % stage_id)
            except Exception as exc:
                error = "%s: %s" % (type(exc).__name__, exc)
                break

        valid = len(audits) == len(stages) and all(item["valid"] for item in audits)
        summary = {
            "schema_version": 1,
            "phase": "Stage_B_realistic_workload_harness_validation",
            "valid": valid,
            "completed_stages": [item["stage"] for item in audits],
            "blocked_stages": [
                str(item["id"]) for item in stages[len(audits):]
            ],
            "stage_audits": audits,
            "error": error,
            "next_action": (
                "define Stage C cold-start feedback DVFS experiment"
                if valid else "stop; diagnose the first failed Stage B gate"
            ),
            "claim_boundary": (
                "substrate/harness validation; no controller effectiveness claim"
            ),
        }
        _write_json(self.run_dir / "stageb_summary.json", summary)
        lines = [
            "# XPYD Stage B realistic workload harness validation", "",
            "Verdict: **%s**" % ("PASS" if valid else "FAIL"), "",
            "| Stage | Scenario | Concurrency | Requests | Client wait max | Admission p99 (ms) | Valid |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for item in audits:
            lines.append(
                "| %(stage)s | %(scenario)s | %(max_concurrency)s | %(request_count)s | "
                "%(maximum_client_waiting_requests)s | %(p99_client_admission_delay_ms)s | %(valid)s |"
                % item
            )
        if error:
            lines.extend(["", "Failure: `%s`" % error])
        lines.extend([
            "", "No routing/DVFS controller or predictor was active.", "",
            "Next action: **%s**" % summary["next_action"], "",
        ])
        (self.run_dir / "stageb_summary.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        if not valid:
            raise StageBError(error or "Stage B hard gates failed")
        print(self.run_dir.as_posix())
        return self.run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--workload-config", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    runtime, workload = load_configs(
        Path(args.runtime_config), Path(args.workload_config)
    )
    StageBPhysicalHarness(runtime, workload, args.run_id).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
