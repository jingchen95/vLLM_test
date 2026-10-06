"""Read-only opportunity replay; stop before any counterfactual measurement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from xpyd.d1_control import exploration_permissions
from xpyd.stage_d1 import StageD1Harness, d1_evidence_from_window


def _value(value: str):
    if value in ("True", "False"):
        return value == "True"
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def replay(source: Path, config: Path) -> dict:
    names = (
        "stage_d1_requests.csv", "stage_d1_measurements.csv",
        "controller_decisions.csv", "clock_segments.csv",
    )
    tables = {
        name: [{key: _value(value) for key, value in row.items()}
               for row in csv.DictReader((source / name).open())]
        for name in names
    }
    measurements = [row for row in tables[names[1]] if row["policy"] == "PER-ENDPOINT-FAST-RR"]
    blocks = sorted({row["block_id"] for row in measurements})
    assert len(blocks) == 6, "expected both loads and all three original repeats"
    outcomes = []
    for block in blocks:
        harness = StageD1Harness.__new__(StageD1Harness)
        harness.settings = json.loads(config.read_text())["stage_d1"]
        controllers = harness._controllers_endpoint()
        harness.decision_rows = []
        arbiter = harness.exploration_arbiters["P"]
        history = []
        for window in (1, 2, 3):
            metrics = dict(next(row for row in measurements if row["block_id"] == block and row["window_index"] == window))
            assert metrics["window_hard_gate_valid"] is True
            requests = [row for row in tables[names[0]] if row["block_id"] == block and row["window_index"] == window]
            segments = [row for row in tables[names[3]] if row["block_id"] == block and row["window_index"] == window]
            p_segments = [row for row in segments if row["endpoint_id"] in ("P0", "P1")]
            assert all(row["segment_kind"] == "STABLE" for row in p_segments), "P replay prefix contains a physical transition"
            for ep in ("P0", "P1"):
                actual = {row["requested_frequency_mhz"] for row in segments if row["endpoint_id"] == ep}
                assert actual == {controllers[ep].current_frequency_mhz}, "never feed HIGH observations to a counterfactual MID state"
            original = [row for row in tables[names[2]] if row["block_id"] == block and row["evidence_window_index"] == window]
            now = min(row["decision_timestamp_s"] for row in original)
            permissions = exploration_permissions(controllers, "P", arbiter)
            decisions = []
            for ep in ("P0", "P1"):
                controller = controllers[ep]
                selected = [row for row in requests if row["prefill_endpoint_id"] == ep]
                evidence = d1_evidence_from_window(
                    "P", metrics, selected, endpoint_id=ep,
                    observed_frequency_mhz=controller.current_frequency_mhz,
                    settings=harness.settings, captured_at_s=now,
                )
                decision = controller.decide(evidence, now_s=now, allow_exploration=permissions[ep])
                row = dict(decision.as_log(), endpoint_id=ep, exploration_permission=permissions[ep])
                target = decision.requested_frequency_mhz
                if target != controller.current_frequency_mhz:
                    # Simulation bookkeeping only, never physical actuation.
                    controller.record_actuation(target, target, readback_valid=True, timestamp_s=now)
                if permissions[ep]:
                    arbiter.record_opportunity(ep, probe_frequency=target if decision.decision == "EXPLORE_DOWN" else None)
                decisions.append(row)
            if window < 3:
                for row in decisions:
                    prior = next(item for item in original if item["endpoint_id"] == row["endpoint_id"])
                    for key in ("decision", "requested_freq_mhz", "evidence_classification"):
                        assert row[key] == prior[key], (block, window, key)
            history.extend({
                "evidence_window": window, "endpoint": row["endpoint_id"],
                "assigned_valid_requests": row["valid_request_count"],
                "input_token_coverage": row["input_token_coverage"],
                "attainment": row["attainment"], "headroom": row["headroom_fraction"],
                "decision": row["decision"], "target_mhz": row["requested_freq_mhz"],
                "permission": row["exploration_permission"],
            } for row in decisions if row["pool"] == "P")
        probes = [row for row in history if row["decision"] == "EXPLORE_DOWN"]
        mid_opportunity = [(row["endpoint"], row["target_mhz"]) for row in probes] == [("P0", 1890), ("P1", 1890)]
        assert 1890 not in controllers["P1"].state.validated_safe_frequencies
        outcomes.append({
            "block": block, "load": metrics["load_point"], "repeat": metrics["repeat"],
            "P_probe_opportunities": probes, "P_decisions": history,
            "stop": "after w3 recommendation, before counterfactual w4 physical measurement",
            "P1_MID_safety_measured": False,
            "both_P_MID_opportunity_gate": mid_opportunity,
        })
    return {
        "kind": "diagnostic_only_not_physical_results", "original_job": 272628,
        "pass": all(block["both_P_MID_opportunity_gate"] for block in outcomes),
        "blocks": outcomes,
        "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names},
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "semantics": "P-side opportunity replay only, holding observed D behavior exogenous. Old w1-w3 physically matched P observations only. Simulated readback bookkeeping, no GPU access. P1 receives a w4 downward opportunity; its target must be reported as recommended, not forced to MID. No counterfactual safety/energy is inferred.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--config", type=Path, default=Path("paper/configs/xpyd_stage_d1_neptune_europa.json"))
    args = parser.parse_args()
    result = replay(args.source, args.config)
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
