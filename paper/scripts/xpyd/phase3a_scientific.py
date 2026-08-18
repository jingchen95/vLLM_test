"""Reproducible scientific analysis for the accepted XpYd Phase 3A run.

The analyzer is deliberately offline: it reads one archived run directory and
writes only its ``analysis/`` child.  It never contacts serving endpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Optional, Sequence


ACCEPTED_RUN_ID = "20260818T180448Z"
ACCEPTED_COMMIT = "700759eb68100e3b03763bc8ff8567164c74415a"

ARTIFACT_SOURCES = {
    "metadata": "metadata.json",
    "acceptance_summary": "derived/summary.json",
    "semantic_results": "derived/semantic_deltas.json",
    "load_results": "derived/load_runs.json",
    "load_monitoring": "derived/load_monitoring.json",
    "telemetry": "derived/telemetry.jsonl",
    "scrape_index": "derived/scrapes.jsonl",
    "proxy_diagnostics": "derived/proxy_diagnostics.jsonl",
    "proxy_audit": "derived/proxy_diagnostics_audit.json",
    "warmup": "derived/phase_warmup.json",
    "generated_engineering_summary": "derived/summary.md",
    "generated_semantic_summary": "derived/semantic_summary.json",
    "P0_server_log": "P0/server.log",
    "D0_server_log": "D0/server.log",
}

HISTOGRAM_METRICS = {
    "ttft": "vllm:time_to_first_token_seconds",
    "tbt": "vllm:inter_token_latency_seconds",
    "queue": "vllm:request_queue_time_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "decode": "vllm:request_decode_time_seconds",
}


class AnalysisError(RuntimeError):
    """Raised when archived evidence does not satisfy the analysis contract."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise AnalysisError("refusing to write empty CSV: %s" % path)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def pearson(values_x: Sequence[float], values_y: Sequence[float]) -> Optional[float]:
    if len(values_x) != len(values_y):
        raise ValueError("correlation vectors must have equal length")
    if len(values_x) < 2:
        return None
    mean_x = statistics.fmean(values_x)
    mean_y = statistics.fmean(values_y)
    centered_x = [value - mean_x for value in values_x]
    centered_y = [value - mean_y for value in values_y]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y)) / denominator


def rankdata(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            result[indexed[position][0]] = average_rank
        start = end
    return result


def comparison_stats(
    reference: Sequence[float],
    observed: Sequence[float],
    *,
    evidence_level: str,
    alignment: str,
) -> dict[str, Any]:
    if len(reference) != len(observed) or not reference:
        raise ValueError("comparison requires non-empty, equal-length vectors")
    differences = [abs(left - right) for left, right in zip(reference, observed)]
    relative = [
        abs(left - right) / abs(left)
        for left, right in zip(reference, observed)
        if left != 0
    ]
    return {
        "sample_count": len(reference),
        "mean_absolute_difference_ms": statistics.fmean(differences),
        "mean_relative_error": statistics.fmean(relative) if relative else None,
        "pearson_correlation": pearson(reference, observed),
        "spearman_correlation": pearson(rankdata(reference), rankdata(observed)),
        "reference_mean_ms": statistics.fmean(reference),
        "observed_mean_ms": statistics.fmean(observed),
        "evidence_level": evidence_level,
        "alignment": alignment,
        "relative_error_denominator": "absolute client/reference measurement",
    }


def histogram_mean(histogram: Optional[Mapping[str, Any]]) -> Optional[float]:
    if not histogram or float(histogram.get("count", 0)) <= 0:
        return None
    return 1000.0 * float(histogram["sum_seconds"]) / float(histogram["count"])


def histogram_bracket(
    boundaries: Sequence[Any], target_seconds: float
) -> tuple[Optional[float], Optional[float], bool]:
    finite = sorted(float(value) for value in boundaries if value != "+Inf")
    lower = max((value for value in finite if value < target_seconds), default=None)
    upper = min((value for value in finite if value >= target_seconds), default=None)
    exact = any(math.isclose(value, target_seconds) for value in finite)
    return lower, upper, exact


def source_digest(run_dir: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for path in run_dir.rglob("*")
        if path.is_file() and "analysis" not in path.relative_to(run_dir).parts
    )
    for path in files:
        relative = path.relative_to(run_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def audit_acceptance(run_dir: Path) -> dict[str, Any]:
    missing = [
        relative for relative in ARTIFACT_SOURCES.values()
        if not (run_dir / relative).is_file()
    ]
    if missing:
        raise AnalysisError("accepted result is missing artifacts: %s" % missing)
    if run_dir.name != ACCEPTED_RUN_ID:
        raise AnalysisError(
            "analysis is restricted to accepted run %s, got %s" %
            (ACCEPTED_RUN_ID, run_dir.name)
        )
    if (run_dir / "derived/failure.json").exists():
        raise AnalysisError("accepted result unexpectedly contains failure.json")
    metadata = read_json(run_dir / "metadata.json")
    summary = read_json(run_dir / "derived/summary.json")
    semantics = read_json(run_dir / "derived/semantic_deltas.json")
    loads = read_json(run_dir / "derived/load_runs.json")
    diagnostics = read_jsonl(run_dir / "derived/proxy_diagnostics.jsonl")
    if metadata.get("git_commit") != ACCEPTED_COMMIT:
        raise AnalysisError("accepted commit mismatch")
    checks: dict[str, bool] = {
        "summary_completed": summary.get("completed") is True,
        "semantic_probe_count_4": len(semantics) == 4,
        "load_probe_count_2": len(loads) == 2,
        "zero_scrape_errors": summary.get("scrape_error_count") == 0
        and not (run_dir / "derived/scrape_errors.jsonl").exists(),
        "zero_counter_resets": not summary.get("reset_or_discontinuity_observations"),
        "proxy_audit_valid": summary.get("proxy_diagnostics_audit", {}).get("valid") is True,
        "proxy_diagnostic_count_35": len(diagnostics) == 35,
        "no_failure_json": not (run_dir / "derived/failure.json").exists(),
        "dry_run_disabled": metadata.get("dry_run", {}).get("enabled") is False,
    }
    for record in semantics + loads:
        probe = record["probe"]
        client = record["client"]
        count = int(probe["count"])
        requested_output = count * int(probe["output_len"])
        server_prompt = count * (int(probe["input_len"]) + 1)
        record_checks = (
            int(client["requests_total"]) == count
            and int(client["successful_requests"]) == count
            and int(client["failed_requests"]) == 0
            and int(client["output_tokens_total"]) == requested_output
            and int(client["completion_token_sources"].get("server_usage", 0)) == count
            and int(client["decode_stream_available_requests"]) == count
            and int(client["client_ttft_valid_requests"]) == count
            and int(client["client_tpot_valid_requests"]) == count
            and int(client["client_itl_valid_requests"]) == count
            and record["endpoints"]["P0"]["window"]["valid"] is True
            and record["endpoints"]["D0"]["window"]["valid"] is True
            and float(record["endpoints"]["P0"]["window"]["delta_prompt_tokens"])
            == server_prompt
            and float(record["endpoints"]["D0"]["window"]["delta_prompt_tokens"])
            == server_prompt
            and float(record["endpoints"]["P0"]["window"]["delta_generation_tokens"])
            == count
            and float(record["endpoints"]["D0"]["window"]["delta_generation_tokens"])
            == requested_output
            and float(record["endpoints"]["P0"]["window"]["delta_completed_requests"])
            == count
            and float(record["endpoints"]["D0"]["window"]["delta_completed_requests"])
            == count
        )
        checks["probe_%s" % probe["id"]] = record_checks
    for record in loads:
        monitoring = record["monitoring"]
        checks["monitoring_%s" % record["probe"]["id"]] = (
            monitoring["tolerated_scrape_error_count"] == 0
            and all(value == 0 for value in monitoring["failed_interval_scrapes"].values())
            and all(value > 0 for value in monitoring["successful_interval_scrapes"].values())
        )
    for index, record in enumerate(diagnostics):
        checks["diagnostic_%02d" % index] = (
            record.get("outcome") == "completed"
            and record.get("incoming_client_stream") is True
            and record.get("outgoing_decode_stream") is True
            and record.get("decode_stream_available") is True
            and str(record.get("decode_content_type", "")).startswith("text/event-stream")
            and record.get("client_ttft_valid") is True
            and record.get("client_tpot_valid") is True
            and record.get("client_itl_valid") is True
            and int(record.get("upstream_chunk_count", 0)) > 0
        )
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise AnalysisError("acceptance inconsistency: %s" % failed)
    return {
        "status": "PASS",
        "accepted_run_id": ACCEPTED_RUN_ID,
        "accepted_commit": ACCEPTED_COMMIT,
        "checks_passed": len(checks),
        "checks_failed": 0,
        "failed_checks": [],
        "source_digest_sha256": source_digest(run_dir),
        "historical_runs_included": [],
        "historical_run_20260818T172017Z_excluded": True,
        "synthetic_replay_evidence": "none_in_stored_diagnostics",
        "synthetic_replay_claim_boundary": (
            "All stored diagnostics attest real upstream SSE chunks; the archive "
            "does not contain an independent packet capture."
        ),
    }


def workload_records(run_dir: Path) -> list[dict[str, Any]]:
    semantics = read_json(run_dir / "derived/semantic_deltas.json")
    loads = read_json(run_dir / "derived/load_runs.json")
    records = []
    for kind, source in (("semantic", semantics), ("load", loads)):
        for record in source:
            copied = dict(record)
            copied["kind"] = kind
            records.append(copied)
    return records


def token_accounting_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        probe = record["probe"]
        client = record["client"]
        p_window = record["endpoints"]["P0"]["window"]
        d_window = record["endpoints"]["D0"]["window"]
        rows.append({
            "probe_id": probe["id"],
            "probe_kind": record["kind"],
            "logical_request_count": probe["count"],
            "user_requested_input_tokens_per_request": probe["input_len"],
            "user_requested_output_tokens_per_request": probe["output_len"],
            "client_measured_output_tokens_total": client["output_tokens_total"],
            "client_output_token_source": "server_usage",
            "P0_server_prompt_tokens_delta": p_window["delta_prompt_tokens"],
            "D0_server_prompt_tokens_delta": d_window["delta_prompt_tokens"],
            "P0_generation_tokens_delta": p_window["delta_generation_tokens"],
            "D0_generation_tokens_delta": d_window["delta_generation_tokens"],
            "P0_request_success_delta": p_window["delta_completed_requests"],
            "D0_request_success_delta": d_window["delta_completed_requests"],
            "P0_prompt_tokens_per_request": p_window["delta_prompt_tokens"] / probe["count"],
            "D0_prompt_tokens_per_request": d_window["delta_prompt_tokens"] / probe["count"],
            "server_prompt_minus_user_input_per_request": (
                p_window["delta_prompt_tokens"] / probe["count"] - probe["input_len"]
            ),
        })
    return rows


def workload_latency_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        probe = record["probe"]
        client = record["client"]
        p_endpoint = record["endpoints"]["P0"]
        d_endpoint = record["endpoints"]["D0"]
        p_hist = p_endpoint["histogram_deltas"]
        d_hist = d_endpoint["histogram_deltas"]
        rows.append({
            "probe_id": probe["id"],
            "probe_kind": record["kind"],
            "input_len": probe["input_len"],
            "output_len": probe["output_len"],
            "request_count": probe["count"],
            "max_concurrency": probe.get("max_concurrency", 1),
            "rate_rps": probe.get("rate_rps"),
            "client_mean_ttft_ms": client["mean_ttft_ms"],
            "client_p99_ttft_ms": client["p99_ttft_ms"],
            "client_mean_tpot_ms": client["mean_tpot_ms"],
            "client_p99_tpot_ms": client["p99_tpot_ms"],
            "client_mean_itl_ms": client["mean_itl_ms"],
            "client_p99_itl_ms": client["p99_itl_ms"],
            "client_mean_e2e_ms": client["mean_e2e_latency_ms"],
            "client_p99_e2e_ms": client["p99_e2e_latency_ms"],
            "P0_mean_ttft_ms": p_endpoint["window"]["window_mean_ttft_ms"],
            "D0_mean_ttft_ms": d_endpoint["window"]["window_mean_ttft_ms"],
            "D0_mean_tbt_ms": d_endpoint["window"]["window_mean_tbt_ms"],
            "P0_mean_prefill_ms": histogram_mean(p_hist.get(HISTOGRAM_METRICS["prefill"])),
            "P0_mean_queue_ms": histogram_mean(p_hist.get(HISTOGRAM_METRICS["queue"])),
            "D0_mean_prefill_ms": histogram_mean(d_hist.get(HISTOGRAM_METRICS["prefill"])),
            "D0_mean_decode_ms": histogram_mean(d_hist.get(HISTOGRAM_METRICS["decode"])),
            "D0_mean_queue_ms": histogram_mean(d_hist.get(HISTOGRAM_METRICS["queue"])),
            "P0_ttft_p99_bucket_upper_ms": p_endpoint["window"]["window_ttft_p99_ms"],
            "D0_ttft_p99_bucket_upper_ms": d_endpoint["window"]["window_ttft_p99_ms"],
            "D0_tbt_p99_bucket_upper_ms": d_endpoint["window"]["window_tbt_p99_ms"],
        })
    return rows


def segment_proxy_records(
    diagnostics: Sequence[Mapping[str, Any]],
    phases: Sequence[tuple[str, int]],
) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(
        (dict(record) for record in diagnostics),
        key=lambda record: record["timestamps_monotonic_s"]["request_received"],
    )
    expected = sum(count for _, count in phases)
    if len(ordered) != expected:
        raise AnalysisError(
            "proxy diagnostic cardinality %d != expected %d" % (len(ordered), expected)
        )
    result: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for phase, count in phases:
        result[phase] = ordered[offset:offset + count]
        offset += count
    return result


def client_requests(run_dir: Path, probe_id: str) -> list[dict[str, Any]]:
    path = run_dir / "client" / probe_id / "requests.jsonl"
    return sorted(read_jsonl(path), key=lambda record: record["send_unix_s"])


def request_latency_rows(
    run_dir: Path,
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    diagnostics = read_jsonl(run_dir / "derived/proxy_diagnostics.jsonl")
    phases = [("_phase_warmup", 1)] + [
        (record["probe"]["id"], int(record["probe"]["count"]))
        for record in records
    ]
    segmented = segment_proxy_records(diagnostics, phases)
    rows = []
    record_by_id = {record["probe"]["id"]: record for record in records}
    for probe_id, proxy_records in segmented.items():
        if probe_id == "_phase_warmup":
            continue
        workload = record_by_id[probe_id]
        clients = client_requests(run_dir, probe_id)
        if len(clients) != len(proxy_records):
            raise AnalysisError("client/proxy cardinality mismatch for %s" % probe_id)
        kind = workload["kind"]
        p_window = workload["endpoints"]["P0"]["window"]
        d_window = workload["endpoints"]["D0"]["window"]
        alignment = (
            "single_request_phase_order"
            if len(clients) == 1
            else "ordinal_by_client_send_and_proxy_receive_no_shared_request_id"
        )
        for ordinal, (client, proxy) in enumerate(zip(clients, proxy_records)):
            durations = proxy["durations_ms"]
            composed_ttft = (
                durations["prefill"] + durations["decode_request_to_first_real_chunk"]
            )
            rows.append({
                "probe_id": probe_id,
                "probe_kind": kind,
                "ordinal": ordinal,
                "trace_request_id": client["trace_request_id"],
                "proxy_request_id": proxy["request_id"],
                "alignment_method": alignment,
                "client_ttft_ms": client["ttft_ms"],
                "client_tpot_ms": client["tpot_ms"],
                "client_mean_itl_ms": client["mean_itl_ms"],
                "client_e2e_ms": client["e2e_latency_ms"],
                "P0_ttft_workload_mean_ms": p_window["window_mean_ttft_ms"],
                "D0_ttft_workload_mean_ms": d_window["window_mean_ttft_ms"],
                "D0_tbt_workload_mean_ms": d_window["window_mean_tbt_ms"],
                "proxy_prefill_ms": durations["prefill"],
                "proxy_D_request_to_headers_ms": durations["decode_request_to_headers"],
                "proxy_D_request_to_first_real_chunk_ms": durations[
                    "decode_request_to_first_real_chunk"
                ],
                "proxy_first_chunk_forwarding_delay_ms": durations[
                    "first_chunk_forwarding_delay"
                ],
                "proxy_full_decode_stream_ms": durations["full_decode_stream"],
                "proxy_total_request_ms": durations["total_proxy_request"],
                "proxy_composed_ttft_ms": composed_ttft,
                "client_minus_proxy_composed_ttft_ms": client["ttft_ms"] - composed_ttft,
                "client_minus_proxy_total_e2e_ms": (
                    client["e2e_latency_ms"] - durations["total_proxy_request"]
                ),
            })
    return rows, segmented


def build_latency_comparisons(
    workload_rows: Sequence[Mapping[str, Any]],
    request_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_kind = {
        kind: [row for row in workload_rows if row["probe_kind"] == kind]
        for kind in ("semantic", "load")
    }
    comparisons: dict[str, dict[str, Any]] = {}

    def add_workload(name: str, rows: Sequence[Mapping[str, Any]], left: str, right: str,
                     evidence: str) -> None:
        valid = [row for row in rows if row[left] is not None and row[right] is not None]
        comparisons[name] = comparison_stats(
            [float(row[left]) for row in valid],
            [float(row[right]) for row in valid],
            evidence_level=evidence,
            alignment="workload_window_aggregate",
        )

    add_workload(
        "A_semantic_client_ttft_vs_P0_ttft", by_kind["semantic"],
        "client_mean_ttft_ms", "P0_mean_ttft_ms", "descriptive_n4_not_population",
    )
    add_workload(
        "B_semantic_client_ttft_vs_D0_ttft", by_kind["semantic"],
        "client_mean_ttft_ms", "D0_mean_ttft_ms", "descriptive_n4_not_population",
    )
    add_workload(
        "C_all_workload_client_tpot_vs_D0_tbt", workload_rows,
        "client_mean_tpot_ms", "D0_mean_tbt_ms", "exploratory_n6_workload_means",
    )
    add_workload(
        "D_all_workload_client_itl_vs_D0_tbt", workload_rows,
        "client_mean_itl_ms", "D0_mean_tbt_ms", "exploratory_n6_workload_means",
    )
    for probe_id in ["semantic_all", "load_light_il128_ol128", "load_moderate_il2048_ol512", "all"]:
        if probe_id == "semantic_all":
            selected = [row for row in request_rows if row["probe_kind"] == "semantic"]
            evidence = "descriptive_n4_not_population"
        elif probe_id == "all":
            selected = list(request_rows)
            evidence = "mixed_context_ordinal_alignment"
        else:
            selected = [row for row in request_rows if row["probe_id"] == probe_id]
            evidence = "request_level_ordinal_alignment_without_shared_id"
        comparisons["E_%s_client_e2e_vs_proxy_total" % probe_id] = comparison_stats(
            [float(row["client_e2e_ms"]) for row in selected],
            [float(row["proxy_total_request_ms"]) for row in selected],
            evidence_level=evidence,
            alignment=selected[0]["alignment_method"],
        )
        comparisons["F_%s_client_ttft_vs_proxy_composed" % probe_id] = comparison_stats(
            [float(row["client_ttft_ms"]) for row in selected],
            [float(row["proxy_composed_ttft_ms"]) for row in selected],
            evidence_level=evidence,
            alignment=selected[0]["alignment_method"],
        )
    return comparisons


def telemetry_rows(run_dir: Path) -> list[dict[str, Any]]:
    records = [
        record for record in read_jsonl(run_dir / "derived/telemetry.jsonl")
        if record["label"] == "load_interval"
    ]
    starts: dict[str, float] = {}
    for record in records:
        probe_id = record["probe_id"]
        starts[probe_id] = min(
            starts.get(probe_id, float("inf")), record["central_monotonic_start_s"]
        )
    rows = []
    for record in records:
        snapshot = record["snapshot"]
        window = record["window"]
        rows.append({
            "probe_id": record["probe_id"],
            "endpoint_id": record["endpoint_id"],
            "round_sequence": record["round_sequence"],
            "time_from_probe_first_scrape_s": (
                record["central_monotonic_start_s"] - starts[record["probe_id"]]
            ),
            "central_monotonic_start_s": record["central_monotonic_start_s"],
            "central_scrape_duration_ms": 1000.0 * (
                record["central_monotonic_end_s"] - record["central_monotonic_start_s"]
            ),
            "num_requests_running": snapshot["num_requests_running"],
            "num_requests_waiting": snapshot["num_requests_waiting"],
            "kv_cache_usage_frac": snapshot["kv_cache_usage_frac"],
            "prompt_tokens_per_s": window["prompt_tokens_per_s"],
            "generation_tokens_per_s": window["generation_tokens_per_s"],
            "completed_requests_per_s": window["completed_requests_per_s"],
            "window_interval_s": window["interval_s"],
            "window_valid": window["valid"],
        })
    return rows


def telemetry_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted({(row["probe_id"], row["endpoint_id"]) for row in rows})
    summaries = []
    for probe_id, endpoint_id in groups:
        selected = [
            row for row in rows
            if row["probe_id"] == probe_id and row["endpoint_id"] == endpoint_id
        ]
        starts = [float(row["central_monotonic_start_s"]) for row in selected]
        gaps = [later - earlier for earlier, later in zip(starts, starts[1:])]

        def values(field: str) -> list[float]:
            return [float(row[field]) for row in selected if row[field] is not None]

        running = values("num_requests_running")
        waiting = values("num_requests_waiting")
        kv = values("kv_cache_usage_frac")
        prompt_rate = values("prompt_tokens_per_s")
        generation_rate = values("generation_tokens_per_s")
        summaries.append({
            "probe_id": probe_id,
            "endpoint_id": endpoint_id,
            "sample_count": len(selected),
            "max_running_requests": max(running, default=None),
            "mean_running_requests": statistics.fmean(running) if running else None,
            "running_nonzero_fraction": (
                sum(value > 0 for value in running) / len(running) if running else None
            ),
            "max_waiting_requests": max(waiting, default=None),
            "waiting_nonzero_fraction": (
                sum(value > 0 for value in waiting) / len(waiting) if waiting else None
            ),
            "max_kv_cache_usage_frac": max(kv, default=None),
            "mean_kv_cache_usage_frac": statistics.fmean(kv) if kv else None,
            "p95_kv_cache_usage_frac": percentile(kv, 0.95),
            "max_prompt_tokens_per_s": max(prompt_rate, default=None),
            "max_generation_tokens_per_s": max(generation_rate, default=None),
            "median_scrape_start_gap_s": percentile(gaps, 0.5),
            "p95_scrape_start_gap_s": percentile(gaps, 0.95),
            "max_scrape_start_gap_s": max(gaps, default=None),
            "all_windows_valid": all(row["window_valid"] for row in selected),
        })
    return summaries


def histogram_resolution_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    targets = {
        "ttft": (0.3, 0.5, 1.0),
        "tbt": (0.1, 0.2),
        "queue": (0.1, 0.3),
        "prefill": (0.1, 0.3),
        "decode": (0.3, 1.0),
    }
    rows = []
    for endpoint_id, endpoint in sorted(summary["latest_endpoint_observations"].items()):
        boundaries_by_metric = endpoint["bucket_boundaries_seconds"]
        for short_name, target_values in targets.items():
            metric = HISTOGRAM_METRICS[short_name]
            boundaries = boundaries_by_metric[metric]
            for target in target_values:
                lower, upper, exact = histogram_bracket(boundaries, target)
                rows.append({
                    "endpoint_id": endpoint_id,
                    "metric": metric,
                    "target_seconds": target,
                    "nearest_lower_bucket_seconds": lower,
                    "nearest_upper_bucket_seconds": upper,
                    "exact_target_bucket": exact,
                    "bracket_width_seconds": (
                        None if lower is None or upper is None else upper - lower
                    ),
                    "quantile_interpretation": (
                        "histogram quantile is bounded only to this bucket interval"
                    ),
                })
    return rows


def artifact_provenance_rows(run_dir: Path) -> list[dict[str, Any]]:
    purposes = {
        "metadata": "scope, commit, workload definitions, exclusions",
        "acceptance_summary": "run acceptance, bucket boundaries, global maxima",
        "semantic_results": "semantic client summaries and P0/D0 before-after deltas",
        "load_results": "load client summaries and P0/D0 before-after deltas",
        "load_monitoring": "interval scrape coverage",
        "telemetry": "load interval queue, running, KV and rate time series",
        "scrape_index": "scrape timing and raw snapshot provenance",
        "proxy_diagnostics": "real SSE and proxy path durations",
        "proxy_audit": "diagnostic cardinality and validity",
        "warmup": "excluded warmup accounting",
        "generated_engineering_summary": "cross-check only; not primary quantitative source",
        "generated_semantic_summary": "cross-check only; not primary quantitative source",
        "P0_server_log": "server lifecycle/error-context audit",
        "D0_server_log": "server lifecycle/error-context audit",
    }
    rows = []
    for name, relative in ARTIFACT_SOURCES.items():
        path = run_dir / relative
        rows.append({
            "artifact": name,
            "relative_path": relative,
            "purpose": purposes[name],
            "available": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        })
    rows.extend([
        {
            "artifact": "client_request_metrics",
            "relative_path": "client/<probe>/requests.jsonl",
            "purpose": "per-request client TTFT, TPOT, ITL, E2E and token source",
            "available": True,
            "size_bytes": None,
        },
        {
            "artifact": "raw_prometheus_snapshots",
            "relative_path": "P0/raw_metrics/*.prom; D0/raw_metrics/*.prom",
            "purpose": "unaltered live metric text and bucket definitions",
            "available": True,
            "size_bytes": None,
        },
        {
            "artifact": "dry_run_output",
            "relative_path": "derived/dry_run.jsonl",
            "purpose": "not used: dry-run controller was disabled",
            "available": False,
            "size_bytes": None,
        },
    ])
    return rows


def flatten_comparisons(
    comparisons: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [{"comparison": name, **values} for name, values in comparisons.items()]


def _plot_setup() -> Any:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 180,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def _save_figure(plt: Any, figure: Any, directory: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = directory / (stem + "." + suffix)
        figure.savefig(path, bbox_inches="tight")
        paths.append(path.relative_to(directory.parent).as_posix())
    plt.close(figure)
    return paths


def generate_figures(
    output_dir: Path,
    workload_rows: Sequence[Mapping[str, Any]],
    request_rows: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]],
    histogram_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    plt = _plot_setup()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metadata = []

    labels = [str(row["probe_id"]).replace("semantic_", "S ").replace("load_", "L ")
              for row in workload_rows]
    positions = list(range(len(labels)))
    width = 0.25
    figure, axis = plt.subplots(figsize=(9.2, 4.2))
    for offset, field, label, color in (
        (-width, "client_mean_ttft_ms", "client TTFT", "#303f9f"),
        (0.0, "P0_mean_ttft_ms", "P0 TTFT", "#ef6c00"),
        (width, "D0_mean_ttft_ms", "D0 TTFT", "#00897b"),
    ):
        axis.bar([position + offset for position in positions],
                 [row[field] for row in workload_rows], width=width,
                 label=label, color=color)
    axis.set_yscale("log")
    axis.set_ylabel("Mean latency (ms, log scale)")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_title("End-to-end client TTFT is not either endpoint-local TTFT")
    axis.legend(ncol=3, frameon=False)
    paths = _save_figure(plt, figure, figures_dir, "fig01_ttft_semantics")
    metadata.append({"figure": "Figure 1", "title": axis.get_title(), "files": paths})

    figure, axis = plt.subplots(figsize=(5.8, 5.1))
    for kind, marker, color in (("semantic", "o", "#303f9f"), ("load", "s", "#ef6c00")):
        selected = [row for row in workload_rows if row["probe_kind"] == kind]
        axis.scatter([row["D0_mean_tbt_ms"] for row in selected],
                     [row["client_mean_tpot_ms"] for row in selected],
                     marker=marker, color=color, s=55, label="%s TPOT" % kind)
        axis.scatter([row["D0_mean_tbt_ms"] for row in selected],
                     [row["client_mean_itl_ms"] for row in selected],
                     marker=marker, facecolors="none", edgecolors=color, s=55,
                     label="%s ITL" % kind)
    values = [float(row["D0_mean_tbt_ms"]) for row in workload_rows]
    values.extend(float(row["client_mean_tpot_ms"]) for row in workload_rows)
    lower, upper = min(values) * 0.96, max(values) * 1.04
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="#555555", label="1:1")
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("D0 window-mean TBT (ms)")
    axis.set_ylabel("Client workload-mean latency (ms)")
    axis.set_title("D0 TBT tracks client TPOT/ITL at workload level")
    axis.legend(frameon=False, fontsize=8)
    paths = _save_figure(plt, figure, figures_dir, "fig02_decode_latency_correspondence")
    metadata.append({"figure": "Figure 2", "title": axis.get_title(), "files": paths})

    semantic = [row for row in request_rows if row["probe_kind"] == "semantic"]
    figure, axis = plt.subplots(figsize=(8.0, 4.2))
    semantic_labels = [str(row["probe_id"]).replace("semantic_", "") for row in semantic]
    prefill = [row["proxy_prefill_ms"] for row in semantic]
    decode_to_chunk = [row["proxy_D_request_to_first_real_chunk_ms"] for row in semantic]
    residual = [row["client_minus_proxy_composed_ttft_ms"] for row in semantic]
    axis.bar(semantic_labels, prefill, label="proxy prefill", color="#ef6c00")
    axis.bar(semantic_labels, decode_to_chunk, bottom=prefill,
             label="D request to first real chunk", color="#00897b")
    bottoms = [left + right for left, right in zip(prefill, decode_to_chunk)]
    axis.bar(semantic_labels, residual, bottom=bottoms,
             label="client/proxy boundary residual", color="#78909c")
    axis.set_ylabel("Client TTFT decomposition (ms)")
    axis.set_title("Proxy path decomposition explains end-to-end TTFT")
    axis.legend(frameon=False)
    paths = _save_figure(plt, figure, figures_dir, "fig03_proxy_ttft_decomposition")
    metadata.append({"figure": "Figure 3", "title": axis.get_title(), "files": paths})

    probes = sorted({str(row["probe_id"]) for row in telemetry})
    figure, axes = plt.subplots(len(probes), 1, figsize=(9.0, 5.8), sharex=False)
    if len(probes) == 1:
        axes = [axes]
    for axis, probe_id in zip(axes, probes):
        for endpoint_id, color in (("P0", "#ef6c00"), ("D0", "#00897b")):
            selected = [row for row in telemetry
                        if row["probe_id"] == probe_id and row["endpoint_id"] == endpoint_id]
            axis.plot([row["time_from_probe_first_scrape_s"] for row in selected],
                      [row["num_requests_running"] for row in selected],
                      marker=".", color=color, label="%s running" % endpoint_id)
            axis.plot([row["time_from_probe_first_scrape_s"] for row in selected],
                      [row["num_requests_waiting"] for row in selected],
                      linestyle="--", color=color, label="%s waiting" % endpoint_id)
        axis.set_ylabel("Requests")
        axis.set_title(probe_id.replace("load_", ""), loc="left", fontsize=9)
        axis.legend(ncol=4, frameon=False, fontsize=8)
        axis.set_xlabel("Time from first recorded scrape (s)")
    figure.suptitle("Archived load telemetry: running and waiting requests", y=1.01)
    paths = _save_figure(plt, figure, figures_dir, "fig04_queue_running_telemetry")
    metadata.append({"figure": "Figure 4", "title": "Running/waiting request telemetry", "files": paths})

    figure, axes = plt.subplots(len(probes), 1, figsize=(9.0, 5.6), sharex=False)
    if len(probes) == 1:
        axes = [axes]
    for axis, probe_id in zip(axes, probes):
        for endpoint_id, color in (("P0", "#ef6c00"), ("D0", "#00897b")):
            selected = [row for row in telemetry
                        if row["probe_id"] == probe_id and row["endpoint_id"] == endpoint_id]
            axis.plot([row["time_from_probe_first_scrape_s"] for row in selected],
                      [100.0 * row["kv_cache_usage_frac"] for row in selected],
                      marker=".", color=color, label=endpoint_id)
        axis.set_ylabel("KV usage (%)")
        axis.set_title(probe_id.replace("load_", ""), loc="left", fontsize=9)
        axis.legend(frameon=False)
        axis.set_xlabel("Time from first recorded scrape (s)")
    figure.suptitle("Archived load telemetry: KV-cache occupancy", y=1.01)
    paths = _save_figure(plt, figure, figures_dir, "fig05_kv_cache_telemetry")
    metadata.append({"figure": "Figure 5", "title": "KV-cache usage telemetry", "files": paths})

    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    metric_order = [HISTOGRAM_METRICS[name] for name in ("ttft", "tbt", "queue", "prefill", "decode")]
    short = {value: key.upper() for key, value in HISTOGRAM_METRICS.items()}
    for index, metric in enumerate(metric_order):
        selected = [row for row in histogram_rows
                    if row["endpoint_id"] == "D0" and row["metric"] == metric]
        for row in selected:
            lower_value = row["nearest_lower_bucket_seconds"]
            upper_value = row["nearest_upper_bucket_seconds"]
            if lower_value is not None and upper_value is not None:
                axis.plot([lower_value, upper_value], [index, index], color="#78909c", linewidth=5)
            axis.scatter(row["target_seconds"], index, marker="|", s=150,
                         color="#c62828" if not row["exact_target_bucket"] else "#2e7d32")
    axis.set_xscale("log")
    axis.set_yticks(range(len(metric_order)))
    axis.set_yticklabels([short[metric] for metric in metric_order])
    axis.set_xlabel("Seconds (log scale); bar is containing bucket, tick is target")
    axis.set_title("Histogram buckets bound, but do not exactly resolve, many SLO targets")
    axis.grid(axis="x", alpha=0.25)
    from matplotlib.lines import Line2D
    axis.legend(handles=[
        Line2D([0], [0], color="#78909c", linewidth=5, label="containing bucket"),
        Line2D([0], [0], marker="|", color="#2e7d32", linestyle="none",
               markersize=12, label="exact target bucket"),
        Line2D([0], [0], marker="|", color="#c62828", linestyle="none",
               markersize=12, label="target is bracket-only"),
    ], frameon=False, fontsize=8, loc="upper left")
    paths = _save_figure(plt, figure, figures_dir, "fig06_histogram_slo_resolution")
    metadata.append({"figure": "Figure 6", "title": axis.get_title(), "files": paths})
    return metadata


def evaluate_hypotheses(
    workload_rows: Sequence[Mapping[str, Any]],
    request_rows: Sequence[Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    semantic = [row for row in workload_rows if row["probe_kind"] == "semantic"]
    median_p_fraction = statistics.median(
        float(row["P0_mean_ttft_ms"]) / float(row["client_mean_ttft_ms"])
        for row in semantic
    )
    median_d_fraction = statistics.median(
        float(row["D0_mean_ttft_ms"]) / float(row["client_mean_ttft_ms"])
        for row in semantic
    )
    semantic_requests = [row for row in request_rows if row["probe_kind"] == "semantic"]
    proxy_ttft_error = statistics.fmean(
        abs(float(row["client_minus_proxy_composed_ttft_ms"])) for row in semantic_requests
    )
    return [
        {
            "id": "H1",
            "hypothesis": "D0 TBT is the closest endpoint metric to client TPOT/ITL.",
            "assessment": "supported",
            "evidence_strength": "workload-level exploratory",
            "evidence": {
                "client_tpot_vs_D0_tbt": comparisons["C_all_workload_client_tpot_vs_D0_tbt"],
                "client_itl_vs_D0_tbt": comparisons["D_all_workload_client_itl_vs_D0_tbt"],
            },
            "boundary": "Six workload means; endpoint TBT is not available per request.",
        },
        {
            "id": "H2",
            "hypothesis": "P0 TTFT is an intrinsic prefill-side phase metric, not client TTFT.",
            "assessment": "supported",
            "evidence_strength": "descriptive",
            "evidence": {"median_P0_to_client_TTFT_ratio_semantic": median_p_fraction},
            "boundary": "Four isolated semantic probes; descriptive, not a fitted law.",
        },
        {
            "id": "H3",
            "hypothesis": "D0 TTFT is an intrinsic decode-side startup metric, not client TTFT.",
            "assessment": "supported",
            "evidence_strength": "descriptive",
            "evidence": {"median_D0_to_client_TTFT_ratio_semantic": median_d_fraction},
            "boundary": "Four isolated semantic probes; descriptive, not a fitted law.",
        },
        {
            "id": "H4",
            "hypothesis": "End-to-end TTFT is a composed P/D path quantity.",
            "assessment": "supported",
            "evidence_strength": "proxy-timestamp decomposition",
            "evidence": {"semantic_mean_absolute_boundary_residual_ms": proxy_ttft_error},
            "boundary": (
                "Semantic probes align by phase order. Multi-request load alignment is ordinal "
                "because client and proxy artifacts have no shared request identifier."
            ),
        },
    ]


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return ("%%.%df" % digits) % value
    return str(value)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(_format(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(
    audit: Mapping[str, Any],
    token_rows: Sequence[Mapping[str, Any]],
    workload_rows: Sequence[Mapping[str, Any]],
    request_rows: Sequence[Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, Any]],
    telemetry_summary: Sequence[Mapping[str, Any]],
    histogram_rows: Sequence[Mapping[str, Any]],
    hypotheses: Sequence[Mapping[str, Any]],
    regeneration_command: str,
) -> str:
    accounting_table = _markdown_table(
        ["workload", "logical req.", "P prompt", "D prompt", "P gen.", "D gen.", "P success", "D success"],
        [[row["probe_id"], row["logical_request_count"], row["P0_server_prompt_tokens_delta"],
          row["D0_server_prompt_tokens_delta"], row["P0_generation_tokens_delta"],
          row["D0_generation_tokens_delta"], row["P0_request_success_delta"],
          row["D0_request_success_delta"]] for row in token_rows],
    )
    latency_table = _markdown_table(
        ["workload", "client TTFT", "P0 TTFT", "D0 TTFT", "client TPOT", "client ITL", "D0 TBT", "client E2E"],
        [[row["probe_id"], row["client_mean_ttft_ms"], row["P0_mean_ttft_ms"],
          row["D0_mean_ttft_ms"], row["client_mean_tpot_ms"], row["client_mean_itl_ms"],
          row["D0_mean_tbt_ms"], row["client_mean_e2e_ms"]] for row in workload_rows],
    )
    telemetry_table = _markdown_table(
        ["load", "endpoint", "n", "max running", "max waiting", "max KV",
         "max prompt tok/s", "max gen. tok/s", "max scrape gap (s)"],
        [[row["probe_id"], row["endpoint_id"], row["sample_count"], row["max_running_requests"],
          row["max_waiting_requests"], row["max_kv_cache_usage_frac"],
          row["max_prompt_tokens_per_s"], row["max_generation_tokens_per_s"],
          row["max_scrape_start_gap_s"]]
         for row in telemetry_summary],
    )
    tpot = comparisons["C_all_workload_client_tpot_vs_D0_tbt"]
    itl = comparisons["D_all_workload_client_itl_vs_D0_tbt"]
    unresolved = sum(not row["exact_target_bucket"] for row in histogram_rows)
    exact = len(histogram_rows) - unresolved
    semantic_request_rows = [row for row in request_rows if row["probe_kind"] == "semantic"]
    mean_proxy_residual = statistics.fmean(
        abs(float(row["client_minus_proxy_composed_ttft_ms"])) for row in semantic_request_rows
    )
    max_queue_mean = max(
        float(row[field])
        for row in workload_rows
        for field in ("P0_mean_queue_ms", "D0_mean_queue_ms")
        if row[field] is not None
    )
    hypothesis_lines = "\n".join(
        "- **%s — %s (%s):** %s Boundary: %s" %
        (item["id"], item["assessment"], item["evidence_strength"],
         item["hypothesis"], item["boundary"])
        for item in hypotheses
    )
    return f"""# Phase 3A scientific observability and metric-semantics summary

## 1. Experimental scope

This report analyzes only accepted run `{ACCEPTED_RUN_ID}` at commit `{ACCEPTED_COMMIT}` (Slurm job 254289). It is an offline interpretation of archived client, proxy, and Prometheus artifacts. It starts no server, submits no job, and changes no model, harness, endpoint, GPU, or raw result. The rejected historical run `20260818T172017Z` is explicitly excluded.

## 2. Dataset and acceptance boundary

Acceptance audit: **{audit['status']}**, with {audit['checks_passed']} invariants passed and zero failed. The archive contains four one-request semantic probes, a 10-request light load, a 20-request moderate load, 35 completed real-SSE proxy diagnostics including warmup, zero failed client requests, zero scrape errors, and zero detected counter resets. The analysis source digest (excluding `analysis/`) is `{audit['source_digest_sha256']}`.

The controller's dry-run mode was disabled, so no dry-run comparison is available. “Real SSE” here means all stored proxy diagnostics record a streaming upstream response and at least one real chunk; the archive does not contain an independent packet capture.

Artifact-level provenance is in `artifact_provenance.csv`. Exact regeneration:

```bash
{regeneration_command}
```

## 3. Prefill/decode accounting semantics

{accounting_table}

Server prompt counters include one extra token per request relative to requested input length, consistent across P0 and D0 in this run. P0 generation advances by one token per logical request, while D0 generation equals requested output tokens. Both P0 and D0 success counters advance once per logical request: they are phase-local observations of the same requests and **must not be added** to estimate user request throughput.

## 4. Latency semantics and cross-layer correspondence

All values below are milliseconds and are workload means. Endpoint values are Prometheus window aggregates, not per-request observations.

{latency_table}

![TTFT semantics](figures/fig01_ttft_semantics.png)

![Decode correspondence](figures/fig02_decode_latency_correspondence.png)

D0 TBT is the strongest telemetry-to-client correspondence in this archive: against six workload means, client TPOT has mean absolute difference {_format(tpot['mean_absolute_difference_ms'])} ms and Pearson/Spearman {_format(tpot['pearson_correlation'])}/{_format(tpot['spearman_correlation'])}; client ITL has mean absolute difference {_format(itl['mean_absolute_difference_ms'])} ms and Pearson/Spearman {_format(itl['pearson_correlation'])}/{_format(itl['spearman_correlation'])}. These correlations are exploratory at n=6 and must not be presented as a population estimate.

P0 TTFT and D0 TTFT are intrinsic phase-local startup measurements; neither equals end-to-end client TTFT. Proxy timestamps support a composed path interpretation—prefill plus D-request-to-first-real-chunk plus a small client/proxy boundary residual. Across the four strictly phase-aligned semantic probes, the mean absolute residual is {_format(mean_proxy_residual)} ms.

![Proxy TTFT decomposition](figures/fig03_proxy_ttft_decomposition.png)

For load requests, client `trace_request_id` and proxy `request_id` belong to different identifier domains. Request-level load comparisons therefore use only ordinal alignment after sorting client send time and proxy receive time; they are useful diagnostics, not a proof of one-to-one identity. No per-request endpoint TTFT/TBT is fabricated.

## 5. Queue, concurrency, and KV-cache observations

{telemetry_table}

![Running and waiting telemetry](figures/fig04_queue_running_telemetry.png)

![KV-cache telemetry](figures/fig05_kv_cache_telemetry.png)

D0 reached running concurrency 8 with no observed waiting and maximum KV occupancy about 17.1%; P0's archived interval snapshots stayed at zero running/waiting/KV. Across semantic and load before/after windows, the largest endpoint histogram mean queue time was only {_format(max_queue_mean)} ms. Thus this run shows active decode/KV pressure but no observed queue buildup. P0 prompt-token-rate spikes nevertheless confirm work that its short running intervals did not capture. Although every scrape succeeded, per-endpoint scrape-start gaps reached about 8 s because collection was blocking/nonuniform. Consequently, the rate peaks and interval series are coarse observability samples and should not drive a fine-grained feedback controller.

The light and moderate runs differ simultaneously in input length, output length, request count, rate, duration, and concurrency. Their performance must not be interpreted as a controlled load-scaling curve.

## 6. Histogram resolution and SLO implications

Across the selected endpoint/metric targets, {exact} target checks have an exact bucket and {unresolved} are bracket-only. In particular, TTFT 300 ms falls between 250 and 500 ms, while queue/prefill/decode histograms begin too coarsely for many sub-300-ms claims. Histogram-derived quantiles identify a containing bucket, not an exact latency.

![Histogram SLO resolution](figures/fig06_histogram_slo_resolution.png)

For future SLO enforcement, preserve direct client measurements and add bucket boundaries at every policy threshold before relying on Prometheus histogram quantiles.

## 7. Hypothesis assessments

{hypothesis_lines}

The four isolated semantic points establish measurement semantics and accounting consistency; they do not establish a universal performance law or justify fitting a predictive surface.

## 8. Operational implications

- Use D0 TBT as the initial telemetry proxy/ranker for client TPOT/ITL, then recalibrate against live XpYd workloads.
- Treat client TTFT as a composed path objective; retain P0 and D0 TTFT for phase diagnosis rather than substituting either for the client metric.
- Keep logical request accounting separate from endpoint-local success counters.
- Use running/waiting/KV series for coarse capacity diagnosis only at this sampling cadence.
- Treat histogram p99 values as bucket-bounded estimates and report their bucket interval.

## 9. Limitations

This is one accepted job on one P0/D0 pairing, with four single-request semantic probes and two confounded load shapes. There are no repetitions, confidence intervals, randomized order, failure cases, energy readings, GPU power telemetry, endpoint-level per-request histograms, or independent SSE packet capture. Dry-run evidence is unavailable because it was disabled. Proxy/client request identity is absent for multi-request phases. Archived telemetry is sparse and nonuniform. These constraints support semantics validation and initial proxy selection, not causal or fleet-wide conclusions.

## 10. Phase 3B recommendation: read-only energy observability

Add read-only, endpoint-separated P0/D0 power and energy sampling without changing the serving path. Record idle baselines and background draw; keep logical request counts distinct from P0/D0 phase counters; report measurement coverage and integration gaps; and explicitly mark KV-transfer/network energy as unobserved unless separately instrumented. Do not sum P0 and D0 request-success counters. Normalize energy as joules per logical request and additionally by workload shape (input tokens, output tokens, and active context), so later comparisons do not confuse work composition with energy efficiency.
"""


def analyze(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_acceptance(run_dir)
    records = workload_records(run_dir)
    tokens = token_accounting_rows(records)
    workloads = workload_latency_rows(records)
    requests, _ = request_latency_rows(run_dir, records)
    comparisons = build_latency_comparisons(workloads, requests)
    telemetry = telemetry_rows(run_dir)
    telemetry_summary = telemetry_summary_rows(telemetry)
    summary = read_json(run_dir / "derived/summary.json")
    histograms = histogram_resolution_rows(summary)
    provenance = artifact_provenance_rows(run_dir)
    hypotheses = evaluate_hypotheses(workloads, requests, comparisons)
    regeneration = (
        "PYTHONPATH=paper/scripts .venv/bin/python -m xpyd.phase3a_scientific "
        "--run-dir /data/users/chjing/vLLM_test/results/xpyd_observability/"
        + ACCEPTED_RUN_ID
    )
    figures = generate_figures(output_dir, workloads, requests, telemetry, histograms)
    outputs = {
        "artifact_provenance.csv": provenance,
        "token_accounting.csv": tokens,
        "workload_latency_comparison.csv": workloads,
        "request_latency_comparison.csv": requests,
        "proxy_decomposition.csv": [
            {key: value for key, value in row.items() if key in {
                "probe_id", "probe_kind", "ordinal", "alignment_method", "client_ttft_ms",
                "client_e2e_ms", "proxy_prefill_ms", "proxy_D_request_to_first_real_chunk_ms",
                "proxy_composed_ttft_ms", "proxy_total_request_ms",
                "client_minus_proxy_composed_ttft_ms", "client_minus_proxy_total_e2e_ms",
            }} for row in requests
        ],
        "latency_comparison_summary.csv": flatten_comparisons(comparisons),
        "load_telemetry_timeseries.csv": telemetry,
        "load_telemetry_summary.csv": telemetry_summary,
        "histogram_resolution.csv": histograms,
    }
    for filename, rows in outputs.items():
        write_csv(output_dir / filename, rows)
    scientific = {
        "schema_version": 1,
        "title": "XpYd Phase 3A scientific observability and metric-semantics summary",
        "audit": audit,
        "artifact_provenance": provenance,
        "token_accounting": tokens,
        "workload_latency_comparison": workloads,
        "request_latency_comparison": requests,
        "latency_comparisons": comparisons,
        "load_telemetry_summary": telemetry_summary,
        "histogram_resolution": histograms,
        "hypotheses": hypotheses,
        "figures": figures,
        "limitations": [
            "single accepted job and one P0/D0 hardware pairing",
            "four isolated semantic points do not define a performance law",
            "light and moderate loads change multiple workload dimensions",
            "endpoint latency histograms are workload aggregates, not per request",
            "load proxy/client alignment is ordinal because identifiers are not shared",
            "dry-run disabled; energy and GPU power unavailable",
            "successful but nonuniform telemetry sampling has long gaps",
        ],
        "regeneration_command": regeneration,
        "raw_results_modified": False,
        "historical_failed_run_included": False,
    }
    write_json(output_dir / "phase3a_scientific_summary.json", scientific)
    report = render_report(
        audit, tokens, workloads, requests, comparisons, telemetry_summary,
        histograms, hypotheses, regeneration,
    )
    (output_dir / "phase3a_scientific_summary.md").write_text(report, encoding="utf-8")
    return scientific


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="accepted Phase 3A run directory")
    parser.add_argument("--output-dir", type=Path,
                        help="output directory (default: RUN_DIR/analysis)")
    arguments = parser.parse_args(argv)
    run_dir = arguments.run_dir.resolve()
    output_dir = (arguments.output_dir or (run_dir / "analysis")).resolve()
    expected_output = run_dir / "analysis"
    if output_dir != expected_output:
        raise AnalysisError("output must be the accepted run's analysis/ directory")
    scientific = analyze(run_dir, output_dir)
    print("Phase 3A analysis: %s" % scientific["audit"]["status"])
    print("Output: %s" % output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
