"""Offline TTFT decomposition from proxy route timestamps.

Read-only.  For every request in the given window directories:

    client_to_route      route_timestamp - client_admitted
    prefill_incl_kv      prefill_completion - route_timestamp
                         (P compute + synchronous P2P-NCCL KV push + fixed path)
    handoff              decode_start - prefill_completion
    decode_first_token   TTFT - (decode_start - client_admitted)
                         (D first step + KV load + stream to client)

Usage:
    python3 step1_ttft_decomposition.py <windows_dir> [<windows_dir> ...]
"""

from __future__ import annotations

import csv
import os
import statistics
import sys
from collections import defaultdict

SEGMENTS = ("ttft", "client_to_route", "prefill_incl_kv", "handoff", "decode_first_token")


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def window_label(name: str) -> str:
    """Strip the repeat prefix and window suffix: r01-<label>[-wNN]."""
    label = name.split("-", 1)[1] if name.startswith("r") else name
    head, _, tail = label.rpartition("-w")
    return head if tail.isdigit() else label


def decompose(windows_dir: str, groups: dict) -> None:
    for name in sorted(os.listdir(windows_dir)):
        path = os.path.join(windows_dir, name)
        try:
            with open(os.path.join(path, "requests.csv")) as fh:
                requests = {row["request_id"]: row for row in csv.DictReader(fh)}
            with open(os.path.join(path, "routes.csv")) as fh:
                routes = list(csv.DictReader(fh))
        except FileNotFoundError:
            continue
        for route in routes:
            request = requests.get(route["request_id"])
            if request is None or route.get("outcome") != "completed":
                continue
            admitted = float(request["client_admitted_unix_s"])
            routed = float(route["route_timestamp_wall_s"])
            prefilled = float(route["prefill_completion_wall_s"])
            decode_start = float(route["decode_start_wall_s"])
            ttft = float(request["ttft_ms"])
            input_class = "L-in" if int(request["input_len"]) > 512 else "S-in"
            group = groups[(window_label(name), input_class)]
            group["ttft"].append(ttft)
            group["client_to_route"].append((routed - admitted) * 1e3)
            group["prefill_incl_kv"].append((prefilled - routed) * 1e3)
            group["handoff"].append((decode_start - prefilled) * 1e3)
            group["decode_first_token"].append(ttft - (decode_start - admitted) * 1e3)


def main(paths: list[str]) -> None:
    groups: dict = defaultdict(lambda: defaultdict(list))
    for path in paths:
        decompose(path, groups)
    writer = csv.writer(sys.stdout)
    writer.writerow(["label", "input_class", "n"] + [
        "%s_%s_ms" % (segment, stat) for segment in SEGMENTS for stat in ("median", "p95")
    ])
    for (label, input_class), group in sorted(groups.items()):
        row = [label, input_class, len(group["ttft"])]
        for segment in SEGMENTS:
            row += [round(statistics.median(group[segment]), 2), round(percentile(group[segment], 0.95), 2)]
        writer.writerow(row)


if __name__ == "__main__":
    main(sys.argv[1:])
