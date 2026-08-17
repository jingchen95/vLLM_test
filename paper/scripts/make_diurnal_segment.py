"""Extract a deterministic diurnal segment from the 1-week Azure conv trace and
thin it (preserving timestamps + length distribution) to cluster-feasible
utilization targets.

Segment rule (deterministic, documented): the 3-hour window starting at the
first daily trough of the 1-week conv trace -- the lowest hourly arrival rate in
the first 24 h -- which is 2024-05-12 04:00:00 (21.0 rps), rising to ~25.5 rps by
07:00:00. This is a slowly-rising diurnal segment, the regime a 5-min two-tier
provisioner is designed for.

Thinning: uniform deterministic stride thinning to a target MEAN rate of
util x PEAK_ANCHOR (PEAK_ANCHOR = Static-Disagg feasible peak = 10 rps for conv,
consistent with the main conv production sweep). Timestamps and (snapped) IL/OL
are preserved; only request density is reduced. This is NOT time-rescaling.

Output: traces/azure_diurnal/conv_diurnal_{50,70,85}pct.csv with columns
arrival_time_s, input_len, output_len (load_trace format -> runner --traces).
"""
import sys, csv, os
sys.path.insert(0, "paper/scripts")
from jsep_traces import IL_VALUES, OL_VALUES, snap_to_nearest

RAW = "data/AzureLLMInferenceTrace_conv_1week.csv"
OUT_DIR = "traces/azure_diurnal"
SEG_HOURS = ("2024-05-12 04", "2024-05-12 05", "2024-05-12 06")  # 04:00-07:00
SEG_START = "2024-05-12 04:00:00+00:00"  # UTC-aware, matches the data's +00:00 stamps
PEAK_ANCHOR = 10.0       # Static-Disagg feasible peak (rps), conv
SEG_DUR_S = 3 * 3600.0
UTILS = [0.50, 0.70, 0.85]

def parse_ts(s):
    import datetime as dt
    iso = s.strip().replace("Z", "+00:00")
    if "." in iso:
        head, rest = iso.split(".", 1)
        d = ""; i = 0
        while i < len(rest) and rest[i].isdigit():
            d += rest[i]; i += 1
        iso = f"{head}.{d[:6].ljust(6,'0')}{rest[i:]}"
    return dt.datetime.fromisoformat(iso).timestamp()

# --- Extract the 3-hour segment (fast hour-prefix filter, then parse) ---
seg = []  # (rel_t, il, ol)
t0 = parse_ts(SEG_START)
with open(RAW) as f:
    r = csv.reader(f); next(r)
    for row in r:
        ts = row[0]
        if not ts.startswith(SEG_HOURS):
            continue
        try:
            ctx = int(float(row[1])); gen = int(float(row[2]))
        except (ValueError, IndexError):
            continue
        if ctx <= 0 or gen <= 0:
            continue
        rel = parse_ts(ts) - t0
        if 0.0 <= rel < SEG_DUR_S:
            seg.append((rel, snap_to_nearest(ctx, IL_VALUES), snap_to_nearest(gen, OL_VALUES)))

seg.sort(key=lambda x: x[0])
n = len(seg); native_mean = n / SEG_DUR_S
print(f"Segment {SEG_START} +3h: {n} reqs, native mean = {native_mean:.1f} rps, dur = {SEG_DUR_S/60:.0f} min ({SEG_DUR_S/300:.0f} DS blocks)")

os.makedirs(OUT_DIR, exist_ok=True)
for util in UTILS:
    target = util * PEAK_ANCHOR
    p = min(target / native_mean, 1.0)
    # deterministic stride thinning: keep req i iff floor((i+1)*p) increments,
    # which retains ~p of requests spread evenly (preserves local density/envelope).
    kept = [seg[i] for i in range(n) if int((i + 1) * p) != int(i * p)]
    path = os.path.join(OUT_DIR, f"conv_diurnal_{int(util*100)}pct.csv")
    with open(path, "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["arrival_time_s", "input_len", "output_len"])
        for rel, il, ol in kept:
            w.writerow([f"{rel:.4f}", il, ol])
    print(f"  util={util}: target_mean={target:.1f} rps, p={p:.3f}, kept={len(kept)} reqs "
          f"({len(kept)/SEG_DUR_S:.1f} rps mean) -> {path}")
