"""Analysis-only TTFT SLO calibration from an accepted Stage C0 run."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


SUSTAINABLE_POINTS = (
    ("rate_005", 0.05),
    ("rate_010", 0.10),
    ("rate_015", 0.15),
    ("rate_020", 0.20),
)
THRESHOLDS_MS = (550, 575, 600, 625, 650)
GROUPS = ("SS", "SL", "LS", "LL", "SHORT_INPUT", "LONG_INPUT", "ALL")


class StageC05Error(RuntimeError):
    """The accepted Stage C0 inputs are missing or inconsistent."""


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise StageC05Error("cannot calculate a percentile of no observations")
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def _read_requests(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise StageC05Error("empty request file: %s" % path)
    result = []
    for row in rows:
        request_class = str(row["workload_class"])
        if request_class not in {"SS", "SL", "LS", "LL"}:
            raise StageC05Error("unexpected request class: %s" % request_class)
        result.append({
            "request_id": str(row["request_id"]),
            "workload_class": request_class,
            "ttft_ms": float(row["ttft_ms"]),
        })
    return result


def _group(rows: Sequence[Mapping[str, Any]], group: str) -> list[float]:
    if group == "ALL":
        selected = rows
    elif group == "SHORT_INPUT":
        selected = [row for row in rows if str(row["workload_class"])[0] == "S"]
    elif group == "LONG_INPUT":
        selected = [row for row in rows if str(row["workload_class"])[0] == "L"]
    else:
        selected = [row for row in rows if row["workload_class"] == group]
    return [float(row["ttft_ms"]) for row in selected]


def summarize_group(
    load_label: str, offered_rate: str, group: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values = _group(rows, group)
    return {
        "load": load_label,
        "offered_rate_rps": offered_rate,
        "request_group": group,
        "n": len(values),
        "mean_ttft_ms": statistics.fmean(values),
        "p50_ttft_ms": percentile(values, 0.50),
        "p90_ttft_ms": percentile(values, 0.90),
        "p95_ttft_ms": percentile(values, 0.95),
        "p99_ttft_ms": percentile(values, 0.99),
        "max_ttft_ms": max(values),
    }


def sensitivity_rows(
    observations: Sequence[tuple[str, str, Sequence[Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    output = []
    pooled = [row for _, _, rows in observations for row in rows]
    populations = list(observations) + [("POOLED", "ALL", pooled)]
    for threshold in THRESHOLDS_MS:
        for load_label, offered_rate, rows in populations:
            for group in ("ALL", "SHORT_INPUT", "LONG_INPUT"):
                values = _group(rows, group)
                violations = sum(value > threshold for value in values)
                output.append({
                    "ttft_slo_ms": threshold,
                    "load": load_label,
                    "offered_rate_rps": offered_rate,
                    "request_group": group,
                    "n": len(values),
                    "violation_count": violations,
                    "violation_rate": violations / len(values),
                })
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_row(row: Mapping[str, Any]) -> str:
    return (
        "| {load} | {request_group} | {n} | {mean_ttft_ms:.1f} | "
        "{p50_ttft_ms:.1f} | {p90_ttft_ms:.1f} | {p95_ttft_ms:.1f} | "
        "{p99_ttft_ms:.1f} | {max_ttft_ms:.1f} |"
    ).format(**row)


def analyze(run_dir: Path, output_dir: Path, job_id: str) -> tuple[Path, Path, Path]:
    observations = []
    for point, rate in SUSTAINABLE_POINTS:
        rows = _read_requests(run_dir / point / "requests.csv")
        if len(rows) != 48:
            raise StageC05Error("%s expected 48 requests, found %d" % (point, len(rows)))
        observations.append((point, "%.2f" % rate, rows))
    pooled = [row for _, _, rows in observations for row in rows]

    group_rows = [
        summarize_group(load, rate, group, rows)
        for load, rate, rows in observations
        for group in GROUPS
    ]
    group_rows.extend(
        summarize_group("POOLED", "ALL", group, pooled) for group in GROUPS
    )
    threshold_rows = sensitivity_rows(observations)

    class_csv = output_dir / "phasec05_ttft_by_class.csv"
    sensitivity_csv = output_dir / "phasec05_ttft_slo_sensitivity.csv"
    report = output_dir / "XPYD_STAGEC05_SLO_CONTRACT.md"
    _write_csv(class_csv, group_rows)
    _write_csv(sensitivity_csv, threshold_rows)

    selected_groups = {"SHORT_INPUT", "LONG_INPUT"}
    lines = [
        "# XPYD Stage C0.5 TTFT SLO Contract Calibration", "",
        "This is an analysis-only result from accepted Stage C0 job `%s`; no GPU job was run." % job_id,
        "The population contains the four sustainable offered loads (0.05, 0.10, 0.15, and 0.20 req/s), 48 requests per load. A violation is `TTFT > SLO`.", "",
        "## Short-input and long-input TTFT", "",
        "| Load | Group | n | Mean | p50 | p90 | p95 | p99 | Max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        _format_row(row) for row in group_rows
        if row["request_group"] in selected_groups
    )
    lines.extend([
        "", "`SHORT_INPUT = SS + SL`; `LONG_INPUT = LS + LL`. All values are milliseconds.", "",
        "## Per-class TTFT by load", "",
        "| Load | Group | n | Mean | p50 | p90 | p95 | p99 | Max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    lines.extend(
        _format_row(row) for row in group_rows
        if row["load"] != "POOLED" and row["request_group"] in {"SS", "SL", "LS", "LL"}
    )
    lines.extend([
        "",
        "## Per-class pooled TTFT", "",
        "| Load | Group | n | Mean | p50 | p90 | p95 | p99 | Max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    lines.extend(
        _format_row(row) for row in group_rows
        if row["load"] == "POOLED" and row["request_group"] in {"SS", "SL", "LS", "LL"}
    )
    lines.extend([
        "", "## TTFT threshold sensitivity", "",
        "| TTFT SLO (ms) | 0.05 | 0.10 | 0.15 | 0.20 | Pooled |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for threshold in THRESHOLDS_MS:
        selected = [
            row for row in threshold_rows
            if row["ttft_slo_ms"] == threshold and row["request_group"] == "ALL"
        ]
        rates = {row["load"]: row["violation_rate"] for row in selected}
        lines.append(
            "| %d | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% |" % (
                threshold,
                100 * rates["rate_005"], 100 * rates["rate_010"],
                100 * rates["rate_015"], 100 * rates["rate_020"],
                100 * rates["POOLED"],
            )
        )
    lines.extend([
        "", "### Pooled sensitivity by input-length group", "",
        "| TTFT SLO (ms) | Short-input | Long-input | All |",
        "|---:|---:|---:|---:|",
    ])
    for threshold in THRESHOLDS_MS:
        selected = {
            row["request_group"]: row["violation_rate"]
            for row in threshold_rows
            if row["ttft_slo_ms"] == threshold and row["load"] == "POOLED"
        }
        lines.append(
            "| %d | %.2f%% | %.2f%% | %.2f%% |" % (
                threshold, 100 * selected["SHORT_INPUT"],
                100 * selected["LONG_INPUT"], 100 * selected["ALL"],
            )
        )
    short = next(
        row for row in group_rows
        if row["load"] == "POOLED" and row["request_group"] == "SHORT_INPUT"
    )
    long = next(
        row for row in group_rows
        if row["load"] == "POOLED" and row["request_group"] == "LONG_INPUT"
    )
    pooled_sensitivity = {
        row["ttft_slo_ms"]: row for row in threshold_rows
        if row["load"] == "POOLED" and row["request_group"] == "ALL"
    }
    lines.extend([
        "", "## Interpretation", "",
        "- Pooled short-input p95 is %.1f ms; pooled long-input p95 is %.1f ms." % (
            short["p95_ttft_ms"], long["p95_ttft_ms"]
        ),
        "- At 550 ms, %d of %d requests violate (%.2f%%). At 575 ms, %d violate (%.2f%%)." % (
            pooled_sensitivity[550]["violation_count"], pooled_sensitivity[550]["n"],
            100 * pooled_sensitivity[550]["violation_rate"],
            pooled_sensitivity[575]["violation_count"],
            100 * pooled_sensitivity[575]["violation_rate"],
        ),
        "- The 600/625/650 ms rows quantify how much of the remaining tail is isolated rather than load-dependent.",
        "- This sensitivity analysis does not itself select a new SLO. Any contract change must state whether rare fixed-HIGH tail events and workload-class conditioning are in scope.", "",
        "Source: `results/stagec0_capacity/stagec0_neptune_europa_20260904/rate_*/requests.csv`.", "",
    ])
    report.write_text("\n".join(lines), encoding="utf-8")
    return class_csv, sensitivity_csv, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="paper")
    parser.add_argument("--job-id", default="257127")
    args = parser.parse_args()
    for path in analyze(Path(args.run_dir), Path(args.output_dir), args.job_id):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
