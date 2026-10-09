"""Offline idle-floor vs load-dependent energy decomposition of XPYD runs.

Read-only analysis of existing NVML samples (0.2 s cadence) and proxy route
timestamps.  It runs no GPU job and changes no controller state.

Definitions
-----------
busy interval  P endpoint: [route_timestamp, kv_handoff_completion]
               D endpoint: [route_timestamp, decode_completion]
               (route_timestamp is conservative for D: KV receive starts then)
idle interval  consecutive-sample interval with an unchanged graphics clock and
               no busy interval of that endpoint within +/- GUARD_S seconds.
P_idle(ep, f)  sum(counter energy) / sum(duration) over all idle intervals of
               that endpoint at locked graphics clock f, pooled over all runs.
floor energy   sum over accounting-window sample intervals of P_idle(ep, f) * dt.
load energy    gross counter energy - floor energy.

Usage:
    python3 step1_energy_decomposition.py <raw_root> <out_dir> [guard_s]
"""

from __future__ import annotations

import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict

ENDPOINTS = ("P0", "P1", "D0", "D1")
GUARD_S = 1.0

# (stage label, run directory relative to raw_root, window-name parser kind)
RUNS = (
    ("C0", "stagec0_capacity/stagec0_neptune_europa_20260904", "c0"),
    ("C1b", "stagec1_headroom/stagec1_headroom_neptune_europa_20260913", "headroom"),
    ("C1c", "stagec1_stationary/stagec1_full_neptune_europa_20260928", "policy"),
    ("C2a", "stagec2a_fast_safety/stagec2a_full_neptune_europa_20261003", "policy"),
    ("D1", "stage_d1/stage_d1_full_212c925", "policy"),
    ("E0", "stage_e0_regime/stage_e0_regime_neptune_europa_20261008", "headroom"),
)

POLICY_RE = re.compile(r"^r(\d+)-(medium|high-sustainable)-(.+)-w(\d+)$")
HEADROOM_RE = re.compile(r"^r(\d+)-([A-Z_]+)-(MEDIUM|HIGH-SUSTAINABLE)$")
BASELINE_POLICIES = {"static-high", "static-high-rr", "HIGH_HIGH"}
# E0 windows record per-endpoint clock modes; all other runs are LOCKED.
LOCKED = "LOCKED"


def parse_window(kind: str, name: str) -> dict | None:
    if kind == "c0":
        return {"load": name, "policy": "static-high", "repeat": 1, "window": 1}
    if kind == "headroom":
        m = HEADROOM_RE.match(name)
        if not m:
            return None
        return {"load": m.group(3).lower(), "policy": m.group(2),
                "repeat": int(m.group(1)), "window": 1}
    m = POLICY_RE.match(name)
    if not m:
        return None
    return {"load": m.group(2), "policy": m.group(3),
            "repeat": int(m.group(1)), "window": int(m.group(4))}


def window_dirs(run_dir: str, kind: str):
    base = run_dir if kind == "c0" else os.path.join(run_dir, "windows")
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if os.path.isfile(os.path.join(path, "summary.json")):
            yield name, path


def load_samples(path: str) -> list[tuple[float, float, float, float]]:
    """Return sorted (wall_s, energy_j, graphics_mhz, gpu_util_pct)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("status") != "success" or r.get("total_energy_mj") is None:
                continue
            rows.append((float(r["actual_start_wall_s"]),
                         float(r["total_energy_mj"]) / 1000.0,
                         float(r.get("graphics_clock_mhz") or 0.0),
                         float(r.get("gpu_utilization_pct") or 0.0)))
    rows.sort()
    return rows


def _f(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_busy(routes_path: str) -> dict[str, list[tuple[float, float]]]:
    busy: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with open(routes_path) as fh:
        for r in csv.DictReader(fh):
            start = _f(r.get("route_timestamp_wall_s"))
            if start is None:
                continue
            p_end = _f(r.get("kv_handoff_completion_wall_s")) or _f(r.get("prefill_completion_wall_s"))
            d_end = _f(r.get("decode_completion_wall_s"))
            p_ep = r.get("selected_prefill_endpoint_id")
            d_ep = r.get("selected_decode_endpoint_id")
            if p_ep and p_end is not None:
                busy[p_ep].append((start, p_end))
            if d_ep and d_end is not None:
                busy[d_ep].append((start, d_end))
    return {ep: merge(iv) for ep, iv in busy.items()}


def merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def overlaps(busy: list[tuple[float, float]], s: float, e: float) -> bool:
    for bs, be in busy:
        if bs <= e and be >= s:
            return True
    return False


def union_length(busy: list[tuple[float, float]], s: float, e: float) -> float:
    return sum(max(0.0, min(be, e) - max(bs, s)) for bs, be in busy)


def iter_intervals(samples):
    for (t0, e0, f0, u0), (t1, e1, f1, _) in zip(samples, samples[1:]):
        dt = t1 - t0
        if dt <= 0 or dt > 1.0:
            continue
        yield t0, t1, e1 - e0, f0, f1, u0


def main(raw_root: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    windows = []
    for stage, rel, kind in RUNS:
        run_dir = os.path.join(raw_root, rel)
        if not os.path.isdir(run_dir):
            continue
        for name, path in window_dirs(run_dir, kind):
            label = parse_window(kind, name)
            if label is None:
                continue
            with open(os.path.join(path, "summary.json")) as fh:
                summary = json.load(fh)
            modes = {}
            meta_path = os.path.join(path, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path) as fh:
                    modes = json.load(fh).get("clock_modes") or {}
            windows.append({"stage": stage, "name": name, "path": path,
                            "summary": summary, **label, "modes": modes,
                            "busy": load_busy(os.path.join(path, "routes.csv"))})

    # Pass 1: idle power per (endpoint, clock), pooled over every window,
    # including pre/post accounting-window samples.
    idle_e: dict[tuple[str, float, str], float] = defaultdict(float)
    idle_t: dict[tuple[str, float, str], float] = defaultdict(float)
    idle_util: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    idle_win: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    for w in windows:
        for ep in ENDPOINTS:
            sp = os.path.join(w["path"], ep, "samples.jsonl")
            if not os.path.exists(sp):
                continue
            samples = load_samples(sp)
            w.setdefault("samples", {})[ep] = samples
            busy = w["busy"].get(ep, [])
            mode = w["modes"].get(ep, LOCKED)
            we: dict[float, float] = defaultdict(float)
            wt: dict[float, float] = defaultdict(float)
            for t0, t1, de, f0, f1, u0 in iter_intervals(samples):
                if f0 != f1 or overlaps(busy, t0 - GUARD_S, t1 + GUARD_S):
                    continue
                key = (ep, f0, mode)
                idle_e[key] += de
                idle_t[key] += t1 - t0
                idle_util[key].append(u0)
                we[f0] += de
                wt[f0] += t1 - t0
            for f, t in wt.items():
                if t >= 5.0:
                    idle_win[(ep, f, mode)].append(we[f] / t)
    p_idle = {k: idle_e[k] / idle_t[k] for k in idle_t if idle_t[k] > 0}

    with open(os.path.join(out_dir, "idle_power_by_clock.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["endpoint", "graphics_mhz", "clock_mode", "idle_power_w", "idle_seconds",
                     "per_window_n", "per_window_min_w", "per_window_median_w",
                     "per_window_max_w", "mean_gpu_util_pct"])
        for key in sorted(p_idle):
            ep, f, mode = key
            pw = idle_win.get(key, [])
            wr.writerow([ep, int(f), mode, round(p_idle[key], 2),
                         round(idle_t[key], 1), len(pw),
                         round(min(pw), 2) if pw else "",
                         round(statistics.median(pw), 2) if pw else "",
                         round(max(pw), 2) if pw else "",
                         round(statistics.mean(idle_util[key]), 2)])

    def idle_power(ep: str, f: float, mode: str) -> float | None:
        if (ep, f, mode) in p_idle:
            return p_idle[(ep, f, mode)]
        known = [g for (e, g, m) in p_idle if e == ep and m == mode]
        if not known:
            return None
        return p_idle[(ep, min(known, key=lambda g: abs(g - f)), mode)]

    # Pass 2: per-window decomposition inside the accounting boundary.
    rows = []
    for w in windows:
        energy = w["summary"].get("endpoint_energy", {})
        row = {k: w[k] for k in ("stage", "name", "load", "policy", "repeat", "window")}
        for ep in ENDPOINTS:
            info = energy.get(ep) or {}
            bound = info.get("actual_measurement_boundary_wall_s") or {}
            samples = w.get("samples", {}).get(ep)
            if not bound or not samples:
                continue
            s, e = float(bound["start"]), float(bound["end"])
            gross = floor = dur = 0.0
            for t0, t1, de, f0, _, _ in iter_intervals(samples):
                if t0 < s - 1e-6 or t1 > e + 1e-6:
                    continue
                gross += de
                dur += t1 - t0
                floor += (idle_power(ep, f0, w["modes"].get(ep, LOCKED)) or 0.0) * (t1 - t0)
            row[f"{ep}_reported_j"] = float(info.get("gross_gpu_energy_j") or 0.0)
            row[f"{ep}_gross_j"] = gross
            row[f"{ep}_floor_j"] = floor
            row[f"{ep}_dur_s"] = dur
            row[f"{ep}_busy_frac"] = union_length(w["busy"].get(ep, []), s, e) / max(e - s, 1e-9)
        row["requests"] = int(w["summary"].get("request_count") or 0)
        rows.append(row)

    fields = ["stage", "name", "load", "policy", "repeat", "window", "requests"]
    for ep in ENDPOINTS:
        fields += [f"{ep}_reported_j", f"{ep}_gross_j", f"{ep}_floor_j", f"{ep}_dur_s", f"{ep}_busy_frac"]
    with open(os.path.join(out_dir, "window_decomposition.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    # Aggregate per (stage, load, policy).
    agg: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    busy_lists: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["stage"], r["load"], r["policy"])
        a = agg[key]
        a["requests"] += r["requests"]
        for ep in ENDPOINTS:
            if f"{ep}_gross_j" not in r:
                continue
            pool = ep[0]
            a[f"{pool}_reported"] += r[f"{ep}_reported_j"]
            a[f"{pool}_gross"] += r[f"{ep}_gross_j"]
            a[f"{pool}_floor"] += r[f"{ep}_floor_j"]
            a[f"{pool}_dur"] += r[f"{ep}_dur_s"]
            busy_lists[key][pool].append(r[f"{ep}_busy_frac"])

    out = []
    for key in sorted(agg):
        a = agg[key]
        tot_gross = a["P_gross"] + a["D_gross"]
        tot_floor = a["P_floor"] + a["D_floor"]
        out.append({
            "stage": key[0], "load": key[1], "policy": key[2],
            "requests": int(a["requests"]),
            "reported_total_kj": (a["P_reported"] + a["D_reported"]) / 1e3,
            "gross_total_kj": tot_gross / 1e3,
            "P_gross_kj": a["P_gross"] / 1e3, "P_floor_kj": a["P_floor"] / 1e3,
            "D_gross_kj": a["D_gross"] / 1e3, "D_floor_kj": a["D_floor"] / 1e3,
            "floor_share_total": tot_floor / tot_gross if tot_gross else 0.0,
            "floor_share_P": a["P_floor"] / a["P_gross"] if a["P_gross"] else 0.0,
            "floor_share_D": a["D_floor"] / a["D_gross"] if a["D_gross"] else 0.0,
            "P_mean_power_w": a["P_gross"] / (a["P_dur"] / 2) if a["P_dur"] else 0.0,
            "D_mean_power_w": a["D_gross"] / (a["D_dur"] / 2) if a["D_dur"] else 0.0,
            "P_busy_frac": statistics.mean(busy_lists[key]["P"]) if busy_lists[key]["P"] else 0.0,
            "D_busy_frac": statistics.mean(busy_lists[key]["D"]) if busy_lists[key]["D"] else 0.0,
            "J_per_request": tot_gross / a["requests"] if a["requests"] else 0.0,
            "floor_J_per_request": tot_floor / a["requests"] if a["requests"] else 0.0,
        })

    # Savings attribution against the matched all-HIGH baseline of the same stage/load.
    by_key = {(o["stage"], o["load"], o["policy"]): o for o in out}
    for o in out:
        base = next((by_key[(o["stage"], o["load"], b)] for b in BASELINE_POLICIES
                     if (o["stage"], o["load"], b) in by_key), None)
        if base is None or base is o or base["requests"] != o["requests"]:
            continue
        d_total = base["gross_total_kj"] - o["gross_total_kj"]
        d_floor = (base["P_floor_kj"] + base["D_floor_kj"]) - (o["P_floor_kj"] + o["D_floor_kj"])
        d_p = base["P_gross_kj"] - o["P_gross_kj"]
        d_p_floor = base["P_floor_kj"] - o["P_floor_kj"]
        o["saving_vs_high_pct"] = 100.0 * d_total / base["gross_total_kj"]
        o["saving_from_floor_share"] = d_floor / d_total if d_total else ""
        o["P_saving_from_floor_share"] = d_p_floor / d_p if d_p else ""

    cols = list(out[0].keys()) + ["saving_vs_high_pct", "saving_from_floor_share", "P_saving_from_floor_share"]
    with open(os.path.join(out_dir, "policy_decomposition.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for o in out:
            wr.writerow({k: (round(o[k], 4) if isinstance(o.get(k), float) else o.get(k, "")) for k in cols})
    print(f"windows={len(windows)} rows={len(rows)} policies={len(out)} -> {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 3:
        GUARD_S = float(sys.argv[3])
    main(sys.argv[1], sys.argv[2])
