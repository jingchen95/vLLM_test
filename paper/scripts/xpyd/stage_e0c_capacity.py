"""Stage E0-c: connector concurrency ladder and real-capacity rate sweep.

Stage C0 never exceeded client concurrency 4 (the highest concurrency Stage B
validated), and its 0.30 req/s "overload" was client-side queueing with zero
endpoint waiting. E0-c measures the real 2P2D capacity at validated HIGH
clocks:

1. Concurrency ladder.  Client cap 4 -> 8 -> 16 -> 32 -> 64, offered fast
   enough to hold the cap.  Every step must pass the full Phase 3C/Stage B
   correctness audit (SSE, request IDs, tokens, routes, NVML, fixed clocks).
   The first failing step stops the ladder; the last passing step is the
   highest validated connector concurrency.  Saturated throughput per step
   gives the throughput-versus-concurrency curve.
2. Poisson rate sweep at the highest validated concurrency, increasing until
   backlog grows for ``stop_after_overloads`` consecutive points.

No controller, DVFS action, routing intelligence or SLO change is involved.
SLO attainment is reported under the unchanged 600/65 ms contract only as a
descriptive column; it does not gate the ladder or the sweep.
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

from xpyd.phase3c_substrate import (
    Phase3CSubstrateHarness,
    _read_json,
    _write_json,
    load_config as load_runtime_config,
)
from xpyd.stagec0_capacity import analyze_point, percentile
from xpyd.workload_harness import build_trace, load_config as load_workload_config

ENDPOINTS = ("P0", "P1", "D0", "D1")
ALL_PAIRS = [["P0", "D0"], ["P0", "D1"], ["P1", "D0"], ["P1", "D1"]]


class StageE0cError(RuntimeError):
    pass


def load_configs(runtime_path: Path, workload_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = load_runtime_config(runtime_path)
    workload = load_workload_config(workload_path)
    settings = workload.get("stage_e0c")
    if not isinstance(settings, Mapping):
        raise StageE0cError("stage_e0c settings are required")
    expected_nodes = dict(workload.get("expected_topology") or {
        "P0": "neptune", "P1": "neptune", "D0": "europa", "D1": "europa",
    })
    if {str(item["endpoint_id"]): str(item["node"]) for item in runtime["endpoints"]} != expected_nodes:
        raise StageE0cError("runtime topology does not match expected_topology %s" % expected_nodes)
    if any(int(item["tp_degree"]) != 1 for item in runtime["endpoints"]):
        raise StageE0cError("E0-c keeps TP=1")
    ladder = [int(value) for value in settings["concurrency_ladder"]]
    if ladder != sorted(set(ladder)) or ladder[0] != 4:
        raise StageE0cError("concurrency ladder must be increasing and start at the validated 4")
    rates = [float(value) for value in settings["sweep_rates_rps"]]
    if rates != sorted(set(rates)) or rates[0] <= 0:
        raise StageE0cError("sweep rates must be positive and increasing")
    if settings.get("control") != "disabled" or settings.get("frequencies") != "all validated HIGH":
        raise StageE0cError("E0-c forbids control and requires fixed validated HIGH clocks")
    weights = settings["class_weights"]
    if weights != {"SS": 0.35, "SL": 0.15, "LS": 0.35, "LL": 0.15}:
        raise StageE0cError("E0-c keeps the C1 35/15/35/15 mix")
    return runtime, workload


def step_trace(
    workload: Mapping[str, Any], step_id: str, rate_rps: float, count: int,
    process: str, seed: int,
) -> list[dict[str, Any]]:
    """One stationary segment injected as its own scenario."""
    config = copy.deepcopy(dict(workload))
    segment: dict[str, Any] = {
        "id": "steady_mix", "request_count": int(count), "rate_rps": float(rate_rps),
        "class_weights": dict(workload["stage_e0c"]["class_weights"]),
    }
    if process == "poisson":
        segment.update({"arrival_process": "poisson", "seed": int(seed)})
    config["scenarios"] = [{"id": "e0c_step", "segments": [segment]}]
    return build_trace(config, "e0c_step", step_id)


def saturated_throughput(requests: Sequence[Mapping[str, Any]]) -> dict[str, Optional[float]]:
    """Requests and output tokens per second over the inter-quartile completion span."""
    completions = sorted(
        (float(row["client_admitted_unix_s"]) + float(row["e2e_latency_ms"]) / 1000.0,
         int(float(row["observed_output_tokens"])))
        for row in requests
    )
    if len(completions) < 8:
        return {"saturated_rps": None, "saturated_output_tok_s": None}
    low, high = len(completions) // 4, (3 * len(completions)) // 4
    span = completions[high][0] - completions[low][0]
    if span <= 0:
        return {"saturated_rps": None, "saturated_output_tok_s": None}
    tokens = sum(item[1] for item in completions[low + 1:high + 1])
    return {"saturated_rps": (high - low) / span, "saturated_output_tok_s": tokens / span}


def attainment(requests: Sequence[Mapping[str, Any]], slo: Mapping[str, Any]) -> dict[str, float]:
    ttft = [float(row["ttft_ms"]) for row in requests]
    tpot = [float(row["tpot_ms"]) for row in requests]
    return {
        "ttft_p50_ms": percentile(ttft, 0.50), "tpot_p50_ms": percentile(tpot, 0.50),
        "ttft_attainment": sum(value <= float(slo["ttft_ms"]) for value in ttft) / len(ttft),
        "tpot_attainment": sum(value <= float(slo["tpot_ms"]) for value in tpot) / len(tpot),
    }


class StageE0cHarness:
    def __init__(
        self, runtime: Mapping[str, Any], workload: Mapping[str, Any],
        run_id: str | None = None,
        window_runner: Optional[Callable[[Mapping[str, Any], str, Sequence[Mapping[str, Any]]], None]] = None,
    ) -> None:
        self.runtime = copy.deepcopy(dict(runtime))
        self.workload = copy.deepcopy(dict(workload))
        self.settings = self.workload["stage_e0c"]
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(os.path.expandvars(str(workload["output_root"]))) / self.run_id
        self.window_runner = window_runner or (
            lambda config, step_id, trace: Phase3CSubstrateHarness(config, run_id=step_id, trace_rows=trace).run()
        )
        self.ladder_rows: list[dict[str, Any]] = []
        self.sweep_rows: list[dict[str, Any]] = []

    def _config(self, concurrency: int) -> dict[str, Any]:
        config = copy.deepcopy(self.runtime)
        config["output_root"] = self.run_dir.as_posix()
        config["client"]["max_concurrency"] = int(concurrency)
        config["client"]["request_timeout_s"] = float(self.settings["request_timeout_s"])
        config["coverage_policy"] = {"required_endpoint_ids": list(ENDPOINTS), "required_pairs": ALL_PAIRS}
        return config

    def _step(
        self, phase: str, step_id: str, concurrency: int, rate: float, count: int,
        process: str, seed: int,
    ) -> dict[str, Any]:
        trace = step_trace(self.workload, step_id, rate, count, process, seed)
        row: dict[str, Any] = {
            "phase": phase, "step_id": step_id, "max_concurrency": concurrency,
            "offered_rate_rps": rate, "planned_requests": count, "arrival_process": process,
        }
        step_dir = self.run_dir / step_id
        try:
            self.window_runner(self._config(concurrency), step_id, trace)
        except Exception as exc:
            row.update({"step_valid": False, "error": "%s: %s" % (type(exc).__name__, exc)})
            failure = step_dir / "failure.json"
            if failure.is_file():
                row["failure"] = json.dumps(_read_json(failure), sort_keys=True)
            return row
        audit = _read_json(step_dir / "audit.json")
        row["hard_gates_json"] = json.dumps(audit.get("hard_gates", {}), sort_keys=True)
        if not audit.get("valid"):
            row.update({"step_valid": False, "error": "Phase 3C audit invalid"})
            return row
        point = {"id": step_id, "offered_rate_rps": rate}
        metrics = analyze_point(point, step_dir, self.settings["slo"], self.settings["trend_thresholds"])
        requests = list(csv.DictReader((step_dir / "requests.csv").open(encoding="utf-8")))
        load = metrics.pop("endpoint_load")
        row.update(metrics)
        row.update(saturated_throughput(requests))
        row.update(attainment(requests, self.settings["slo"]))
        for endpoint_id in ENDPOINTS:
            stats = load.get(endpoint_id, {})
            row["%s_running_max" % endpoint_id] = stats.get("running_max")
            row["%s_waiting_max" % endpoint_id] = stats.get("waiting_max")
        output_tokens = sum(int(float(item["observed_output_tokens"])) for item in requests)
        energy = metrics.get("total_gross_energy_j")
        row["joules_per_output_token"] = energy / output_tokens if energy and output_tokens else None
        row.update({"step_valid": True, "error": None})
        return row

    def run(self) -> Path:
        if self.run_dir.exists():
            raise StageE0cError("E0-c output already exists: %s" % self.run_dir)
        self.run_dir.mkdir(parents=True)
        _write_json(self.run_dir / "e0c_plan.json", {
            "schema_version": 1, "run_id": self.run_id, "settings": self.settings,
        })
        try:
            validated = None
            hold_s = float(self.settings["ladder_hold_seconds_per_request_in_flight"])
            for concurrency in self.settings["concurrency_ladder"]:
                concurrency = int(concurrency)
                count = max(int(self.settings["ladder_min_requests"]),
                            int(self.settings["ladder_requests_per_slot"]) * concurrency)
                row = self._step(
                    "ladder", "ladder-c%03d" % concurrency, concurrency,
                    concurrency / hold_s, count, "uniform", 0,
                )
                self.ladder_rows.append(row)
                if not row["step_valid"]:
                    break
                validated = concurrency
            if validated is None:
                raise StageE0cError("the validated concurrency 4 failed; no sweep attempted")
            overloads = 0
            steady_s = float(self.settings["sweep_steady_s"])
            for index, rate in enumerate(self.settings["sweep_rates_rps"]):
                rate = float(rate)
                count = max(int(self.settings["sweep_min_requests"]), int(math.ceil(rate * steady_s)))
                row = self._step(
                    "sweep", "sweep-r%05.2f" % rate, validated, rate, count, "poisson",
                    int(self.settings["seed"]) + index,
                )
                self.sweep_rows.append(row)
                if not row["step_valid"]:
                    break
                overloads = overloads + 1 if not row.get("sustainable") else 0
                if overloads >= int(self.settings["stop_after_overloads"]):
                    break
        finally:
            self._write_reports()
        print(self.run_dir.as_posix())
        return self.run_dir

    def _write_reports(self) -> None:
        for name, rows in (("e0c_ladder.csv", self.ladder_rows), ("e0c_sweep.csv", self.sweep_rows)):
            fields: list[str] = []
            for row in rows:
                fields += [key for key in row if key not in fields]
            with (self.run_dir / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields or ["empty"])
                writer.writeheader()
                writer.writerows(rows)
        passing = [row for row in self.ladder_rows if row.get("step_valid")]
        sustainable = [row for row in self.sweep_rows if row.get("step_valid") and row.get("sustainable")]
        summary = {
            "schema_version": 1, "phase": "Stage_E0c_capacity", "run_id": self.run_id,
            "max_validated_concurrency": max((row["max_concurrency"] for row in passing), default=None),
            "first_failed_concurrency": next(
                (row["max_concurrency"] for row in self.ladder_rows if not row.get("step_valid")), None),
            "max_sustainable_rate_rps": max((row["offered_rate_rps"] for row in sustainable), default=None),
            "ladder": self.ladder_rows, "sweep": self.sweep_rows,
            "claim_boundary": (
                "fixed HIGH clocks, four-pair round-robin, TP1, C1 class mix; capacity of this "
                "substrate only; SLO columns are descriptive under the unchanged 600/65 ms contract"
            ),
        }
        _write_json(self.run_dir / "e0c_summary.json", summary)
        (self.run_dir / "e0c_summary.md").write_text(render_markdown(summary), encoding="utf-8")


def _fmt(value: Any, pattern: str = "%.2f") -> str:
    return "n/a" if value is None else pattern % value


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# XPYD Stage E0-c capacity", "",
        "Run `%s`. Highest validated connector concurrency: **%s** (first failure: %s). "
        "Highest sustainable Poisson rate: **%s req/s**." % (
            summary["run_id"], summary["max_validated_concurrency"],
            summary["first_failed_concurrency"], _fmt(summary["max_sustainable_rate_rps"])),
        "", "## Concurrency ladder (saturating offered load)", "",
        "| Cap | Valid | Saturated req/s | Output tok/s | TTFT p50/p95 | TPOT p50/p95 | D running max | J/output token | Error |",
        "|---:|---|---:|---:|---|---|---|---:|---|",
    ]
    for row in summary["ladder"]:
        lines.append("| %s | %s | %s | %s | %s / %s | %s / %s | %s / %s | %s | %s |" % (
            row["max_concurrency"], row["step_valid"], _fmt(row.get("saturated_rps"), "%.3f"),
            _fmt(row.get("saturated_output_tok_s"), "%.0f"),
            _fmt(row.get("ttft_p50_ms"), "%.0f"), _fmt(row.get("ttft_p95_ms"), "%.0f"),
            _fmt(row.get("tpot_p50_ms"), "%.1f"), _fmt(row.get("tpot_p95_ms"), "%.1f"),
            _fmt(row.get("D0_running_max"), "%.0f"), _fmt(row.get("D1_running_max"), "%.0f"),
            _fmt(row.get("joules_per_output_token"), "%.3f"), row.get("error") or "",
        ))
    lines += [
        "", "## Poisson rate sweep", "",
        "| Rate | Valid | Completed req/s | Backlog | TTFT p95 / attain | TPOT p95 / attain | J/request | Error |",
        "|---:|---|---:|---|---|---|---:|---|",
    ]
    for row in summary["sweep"]:
        lines.append("| %.2f | %s | %s | %s | %s / %s | %s / %s | %s | %s |" % (
            row["offered_rate_rps"], row["step_valid"], _fmt(row.get("completed_rate_rps"), "%.3f"),
            row.get("backlog_behavior", "n/a"),
            _fmt(row.get("ttft_p95_ms"), "%.0f"), _fmt(row.get("ttft_attainment") and 100 * row["ttft_attainment"], "%.1f%%"),
            _fmt(row.get("tpot_p95_ms"), "%.1f"), _fmt(row.get("tpot_attainment") and 100 * row["tpot_attainment"], "%.1f%%"),
            _fmt(row.get("joules_per_request"), "%.0f"), row.get("error") or "",
        ))
    lines += ["", summary["claim_boundary"], ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--workload-config", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    runtime, workload = load_configs(args.runtime_config, args.workload_config)
    StageE0cHarness(runtime, workload, args.run_id).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
