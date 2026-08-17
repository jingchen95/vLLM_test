#!/usr/bin/env python3
"""
Summarize trace sanity benchmark results.

This script is designed for trace-mode outputs produced by
`run_disagg_benchmark.sh` and `run_trace_sanity_suite.sh`.

It scans `results/trace_sanity_*` directories, reads per-run
`*.trace_summary.json` files when available, falls back to parsing the text
summary, estimates aggregate power / energy from monitor CSVs, and emits:

  1. <output_prefix>_per_component.csv
  2. <output_prefix>_per_trace.csv
  3. <output_prefix>_summary.txt

By default, each trace is expected to contain the components A, B, C, and D.
E can also be summarized if present, but is not required unless explicitly
added to --expected-components.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


TRACE_DIR_RE = re.compile(r"^trace_sanity_(T\d+)(?:_([A-Z]+))?$")
TEXT_PATTERNS = {
    "requests_total": re.compile(r"Requests total:\s+([\d.]+)"),
    "successful_requests": re.compile(r"Successful requests:\s+([\d.]+)"),
    "failed_requests": re.compile(r"Failed requests:\s+([\d.]+)"),
    "mean_ttft_ms": re.compile(r"Mean TTFT \(ms\):\s+([\d.]+)"),
    "p99_ttft_ms": re.compile(r"P99 TTFT \(ms\):\s+([\d.]+)"),
    "mean_tpot_ms": re.compile(r"Mean TPOT \(ms\):\s+([\d.]+)"),
    "p99_tpot_ms": re.compile(r"P99 TPOT \(ms\):\s+([\d.]+)"),
    "trace_duration_s": re.compile(r"Replay trace duration \(s\):\s+([\d.]+)"),
    "trace_csv": re.compile(r"Trace CSV:\s+(.+)$", re.MULTILINE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-prefix", default="results/trace_sanity_summary")
    parser.add_argument("--expected-components", default="A,B,C,D")
    return parser.parse_args()


def discover_trace_dirs(results_root: Path) -> List[Path]:
    return sorted(
        path for path in results_root.iterdir()
        if path.is_dir() and TRACE_DIR_RE.match(path.name)
    )


def infer_trace_name(path: Path) -> Optional[str]:
    for ancestor in (path,) + tuple(path.parents):
        match = TRACE_DIR_RE.match(ancestor.name)
        if match:
            return match.group(1)
    return None


def infer_component_from_path(path: Path) -> Optional[str]:
    parent_name = path.parent.name
    stem = path.name.replace(".trace_summary.json", "")
    if parent_name == "A_monolithic_l40s" and stem.startswith("bench_"):
        return "A"
    if parent_name == "B_monolithic_l4" and stem.startswith("bench_"):
        return "B"
    if parent_name == "CD_prefill_decode_only":
        if stem.startswith("prefill_"):
            return "C"
        if stem.startswith("decode_"):
            return "D"
    if parent_name == "E_disaggregated" and stem.startswith("disagg_"):
        return "E"
    return None


def parse_text_summary(path: Path) -> Optional[Dict[str, object]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if "Synthetic Trace Replay Summary" not in text:
        return None

    out: Dict[str, object] = {}
    for key, pattern in TEXT_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        if key == "trace_csv":
            out[key] = match.group(1).strip()
        else:
            value = float(match.group(1))
            out[key] = int(value) if value.is_integer() else value
    return out


def read_summary(path: Path) -> Optional[Dict[str, object]]:
    if path.suffix == ".json":
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return parse_text_summary(path)


def monitor_paths_for_run(component: str, txt_path: Path) -> List[Path]:
    stem = txt_path.stem
    parent = txt_path.parent
    if component in {"A", "B"}:
        monitor_stem = stem.replace("bench_", "monitor_", 1)
        return sorted(parent.glob(f"{monitor_stem}_gpu*.csv"))
    if component in {"C", "D"}:
        return sorted(parent.glob(f"monitor_{stem}_gpu*.csv"))
    if component == "E":
        return sorted(parent.glob(f"monitor_{stem}_*_gpu*.csv"))
    return []


def compute_power_energy(monitor_paths: Iterable[Path],
                         start_ts: Optional[float],
                         end_ts: Optional[float]) -> Dict[str, Optional[float]]:
    total_power = 0.0
    total_energy = 0.0
    total_samples = 0
    used_files = 0

    for path in monitor_paths:
        try:
            with path.open(encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except OSError:
            continue
        if not rows:
            continue

        filtered = []
        for row in rows:
            try:
                ts = float(row.get("timestamp", "nan"))
            except ValueError:
                continue
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            filtered.append(row)
        if not filtered:
            continue

        first_energy = None
        last_energy = None
        power_sum = 0.0
        n = 0
        for row in filtered:
            try:
                power = float(row.get("power_w", "nan"))
            except ValueError:
                power = math.nan
            try:
                energy = float(row.get("total_energy_mj", "nan"))
            except ValueError:
                energy = math.nan

            if not math.isnan(power):
                power_sum += power
                n += 1
            if not math.isnan(energy):
                if first_energy is None:
                    first_energy = energy
                last_energy = energy

        if n == 0:
            continue

        used_files += 1
        total_power += power_sum / n
        total_samples += n
        if first_energy is not None and last_energy is not None and last_energy >= first_energy:
            total_energy += (last_energy - first_energy) / 1000.0

    return {
        "avg_power_w": round(total_power, 3) if used_files else None,
        "energy_j": round(total_energy, 3) if used_files else None,
        "monitor_samples": total_samples if used_files else None,
        "monitor_files": used_files,
    }


def build_component_rows(results_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for trace_dir in discover_trace_dirs(results_root):
        trace_name = infer_trace_name(trace_dir)
        if trace_name is None:
            continue

        json_paths = sorted(trace_dir.rglob("*.trace_summary.json"))
        txt_paths = sorted(trace_dir.rglob("*.txt"))
        seen_txt_paths = set()

        for summary_path in json_paths:
            summary = read_summary(summary_path)
            if summary is None:
                continue
            txt_path = summary_path.with_suffix("").with_suffix(".txt")
            seen_txt_paths.add(txt_path)
            component = infer_component_from_path(summary_path)
            if component is None:
                continue
            power = compute_power_energy(
                monitor_paths_for_run(component, txt_path),
                summary.get("timing_start_unix_s"),
                summary.get("timing_end_unix_s"),
            )
            rows.append(component_row(trace_name, trace_dir, component, txt_path, summary, power))

        for txt_path in txt_paths:
            if txt_path in seen_txt_paths:
                continue
            component = infer_component_from_path(txt_path)
            if component is None:
                continue
            summary = read_summary(txt_path)
            if summary is None:
                continue
            power = compute_power_energy(monitor_paths_for_run(component, txt_path), None, None)
            rows.append(component_row(trace_name, trace_dir, component, txt_path, summary, power))

    return sorted(rows, key=lambda row: (row["trace"], row["component"], row["run_name"]))


def component_row(trace_name: str,
                  trace_dir: Path,
                  component: str,
                  txt_path: Path,
                  summary: Dict[str, object],
                  power: Dict[str, Optional[float]]) -> Dict[str, object]:
    failed = int(summary.get("failed_requests", -1)) if summary.get("failed_requests") is not None else None
    successful = int(summary.get("successful_requests", -1)) if summary.get("successful_requests") is not None else None
    requests_total = int(summary.get("requests_total", -1)) if summary.get("requests_total") is not None else None

    if failed is None or successful is None or requests_total is None:
        status = "INCOMPLETE"
    elif failed == 0 and successful == requests_total:
        status = "PASS"
    else:
        status = "FAIL"

    return {
        "trace": trace_name,
        "trace_key": trace_dir.name,
        "trace_dir": trace_dir.as_posix(),
        "component": component,
        "run_name": txt_path.stem,
        "summary_path": txt_path.with_suffix(".trace_summary.json").as_posix()
        if txt_path.with_suffix(".trace_summary.json").exists() else "",
        "text_path": txt_path.as_posix(),
        "trace_csv": summary.get("trace_csv", ""),
        "requests_total": requests_total,
        "successful_requests": successful,
        "failed_requests": failed,
        "mean_ttft_ms": summary.get("mean_ttft_ms"),
        "p99_ttft_ms": summary.get("p99_ttft_ms"),
        "mean_tpot_ms": summary.get("mean_tpot_ms"),
        "p99_tpot_ms": summary.get("p99_tpot_ms"),
        "trace_duration_s": summary.get("trace_duration_s"),
        "max_concurrency": summary.get("max_concurrency"),
        "num_warmups": summary.get("num_warmups"),
        "timing_start_unix_s": summary.get("timing_start_unix_s"),
        "timing_end_unix_s": summary.get("timing_end_unix_s"),
        "avg_power_w": power["avg_power_w"],
        "energy_j": power["energy_j"],
        "monitor_samples": power["monitor_samples"],
        "monitor_files": power["monitor_files"],
        "status": status,
    }


def build_trace_rows(component_rows: List[Dict[str, object]],
                     expected_components: List[str]) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    for row in component_rows:
        trace = str(row["trace_key"])
        component = str(row["component"])
        grouped[trace][component] = row

    rows: List[Dict[str, object]] = []
    for trace in sorted(grouped):
        component_map = grouped[trace]
        trace_name = str(next(iter(component_map.values()))["trace"])
        missing = [comp for comp in expected_components if comp not in component_map]
        failed = [comp for comp, row in component_map.items() if row["status"] == "FAIL"]
        incomplete = [comp for comp, row in component_map.items() if row["status"] == "INCOMPLETE"]

        if missing or incomplete:
            overall = "INCOMPLETE"
        elif failed:
            overall = "FAIL"
        else:
            overall = "PASS"

        total_requests = sum(int(row["requests_total"] or 0) for row in component_map.values())
        total_success = sum(int(row["successful_requests"] or 0) for row in component_map.values())
        total_failed = sum(int(row["failed_requests"] or 0) for row in component_map.values())
        total_energy = sum(float(row["energy_j"] or 0.0) for row in component_map.values())

        rows.append({
            "trace": trace_name,
            "trace_key": trace,
            "expected_components": ",".join(expected_components),
            "present_components": ",".join(sorted(component_map)),
            "missing_components": ",".join(missing),
            "failed_components": ",".join(sorted(failed)),
            "incomplete_components": ",".join(sorted(incomplete)),
            "overall_status": overall,
            "component_count": len(component_map),
            "requests_total_sum": total_requests,
            "successful_requests_sum": total_success,
            "failed_requests_sum": total_failed,
            "energy_j_sum": round(total_energy, 3),
        })
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path: Path,
                  component_rows: List[Dict[str, object]],
                  trace_rows: List[Dict[str, object]]) -> None:
    lines = []
    lines.append("Trace Sanity Summary")
    lines.append("")
    for row in trace_rows:
        lines.append(
            f"{row['trace_key']} ({row['trace']}): {row['overall_status']} | "
            f"present={row['present_components'] or '-'} | "
            f"missing={row['missing_components'] or '-'} | "
            f"failed={row['failed_components'] or '-'} | "
            f"incomplete={row['incomplete_components'] or '-'}"
        )
    lines.append("")
    lines.append("Per-component:")
    for row in component_rows:
        lines.append(
            f"{row['trace_key']} {row['component']}: {row['status']} | "
            f"ok={row['successful_requests']}/{row['requests_total']} | "
            f"failed={row['failed_requests']} | "
            f"p99_ttft={fmt_float(row['p99_ttft_ms'])} ms | "
            f"p99_tpot={fmt_float(row['p99_tpot_ms'])} ms | "
            f"energy={fmt_float(row['energy_j'])} J"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_float(value: object) -> str:
    if value in (None, "", "nan"):
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_prefix = Path(args.output_prefix)
    expected_components = [item.strip() for item in args.expected_components.split(",") if item.strip()]

    component_rows = build_component_rows(results_root)
    trace_rows = build_trace_rows(component_rows, expected_components)

    write_csv(output_prefix.with_name(output_prefix.name + "_per_component.csv"), component_rows)
    write_csv(output_prefix.with_name(output_prefix.name + "_per_trace.csv"), trace_rows)
    write_summary(output_prefix.with_name(output_prefix.name + "_summary.txt"), component_rows, trace_rows)

    print(output_prefix.with_name(output_prefix.name + "_per_component.csv"))
    print(output_prefix.with_name(output_prefix.name + "_per_trace.csv"))
    print(output_prefix.with_name(output_prefix.name + "_summary.txt"))


if __name__ == "__main__":
    main()
