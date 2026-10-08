# XPYD Step 1b — TTFT decomposition

**Status:** complete (offline, read-only; no GPU job)  
**Date:** 2026-10-08  
**Script:** `paper/scripts/xpyd/step1_ttft_decomposition.py`  
**Output:** `paper/step1_energy_decomposition/ttft_decomposition_c1b.csv`

## Question

Why is P=MID TTFT-safe while P=LOW is not, and how much of the 600 ms TTFT
budget is actually controlled by the prefill GPU clock?

## Method

Per request, from client and proxy timestamps:

| Segment | Definition |
|---|---|
| client→route | route timestamp − client admission |
| prefill+KV | prefill completion − route timestamp (P compute + synchronous P2P-NCCL KV push + fixed path) |
| handoff | decode start − prefill completion |
| D first token | TTFT − (decode start − client admission): D first step, KV load, stream to client |

Primary data: C1b fixed points (job of 2026-09-13, 2 repeats × 20 requests per
point/load). Cross-check: STATIC windows of C1c (job 269768) and D1 (job 272628).

## Result (C1b, medians, ms)

| P clock | Input | TTFT | client→route | prefill+KV | handoff | D first token |
|---|---|---:|---:|---:|---:|---:|
| HIGH 2520 | 2048 tok | 503–527 | 2 | **374–378** | 0.1 | ~148 |
| MID 1890 | 2048 tok | 488–548 | 2 | **388–389** | 0.1 | ~148 |
| LOW 1260 | 2048 tok | 548–597 | 2 | **450–451** | 0.1 | ~148 |
| HIGH 2520 | 128 tok | 211–232 | 2 | **126–128** | 0.1 | ~77 |
| LOW 1260 | 128 tok | 215–240 | 2 | **132–135** | 0.1 | ~76 |

The C1c and D1 STATIC windows reproduce the long-input segments (prefill+KV
375–377 ms, D first token 150–152 ms).

## Interpretation

- **The P clock controls only ~75 ms of a ~525 ms long-input TTFT.** HIGH→LOW
  adds 75 ms to the prefill+KV segment; with compute ∝ 1/f this bounds
  long-prompt prefill compute at HIGH to ~75 ms (~14% of TTFT). HIGH→MID adds
  only ~13 ms, which is why MID is safe and LOW is not.
- **About 125 ms of the prefill+KV segment is a fixed path cost.** A 128-token
  prompt, whose prefill compute is a few ms, still takes 126–135 ms, almost
  independent of clock. This is the vLLM 0.15.1 P2P-NCCL synchronous-PUT path
  (with chunked prefill and prefix caching disabled), not GPU work.
- **Long-input KV transfer and the D side cost ~250 ms combined.** The remaining
  ~175 ms of the long-input prefill+KV segment plus ~70 ms extra D first-token
  time are transfer/KV-load cost that no P-side DVFS can affect.
- **The TTFT SLO margin is consumed by the substrate, not by compute.** With
  p95 TTFT at HIGH already ~580 ms, only ~20 ms of the 600 ms budget remains;
  P-side DVFS headroom is therefore set by fixed connector overhead.

## Implications

1. TTFT-based safety signals on this substrate mostly observe transfer and
   connector latency. A controller that lowers P clocks is acting on ~14% of
   the signal it monitors.
2. Any re-grounded SLO (plan stage 3) should be stated relative to the
   substrate's unloaded latency, and the fixed ~125 ms path cost should be
   reported separately so that results do not depend on one connector version.
3. A faster KV path (asynchronous transfer, NIXL, or a newer vLLM) would change
   the TTFT budget more than any P-side DVFS setting.

## Limitations

- Timestamps come from the proxy and client hosts; sub-millisecond segments
  (handoff) are below cross-host alignment accuracy.
- The prefill+KV segment cannot be split into compute and transfer from these
  timestamps; the compute bound uses the clock contrast and assumes ∝ 1/f.
- 20 requests per point/load and class; medians are stable, p95 values are
  sensitive to single outliers (e.g. first-request warm-up).
