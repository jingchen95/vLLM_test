# XPYD Stage E0-c — Connector concurrency ladder and capacity (first run)

**Status:** complete (sweep from run r3, ladder from run r4); closed  
**Slurm job:** `275424` (COMPLETED, 2026-10-13 01:53–02:20, 26 min)  
**Run:** `results/stage_e0c_capacity/stage_e0c_capacity_neptune_europa_20261012_r2`  
**Code:** commit `33936ea` (`paper/scripts/xpyd/stage_e0c_capacity.py`)  
Job `274435` (same config) failed before serving because of the Minerva driver
mismatch (595.71.05 kernel vs 595.91 user space, fixed by reboot on 10-12).

## Questions

1. Up to which client concurrency does the vLLM 0.15.1 P2P-NCCL 2P2D
   substrate stay correct (SSE, request IDs, exact tokens, routes, NVML)?
   Stage B had only validated concurrency 4.
2. What is the real capacity of the substrate at locked HIGH clocks, and
   where does the 600/65 ms contract bind?

## Concurrency ladder (uniform arrivals saturating the cap)

| Cap | Requests | Correctness gates | Fixed-clock gate | Saturated req/s | Output tok/s | TTFT p50/p95 (ms) | TPOT p50/p95 (ms) | TTFT / TPOT attainment |
|---:|---:|---|---|---:|---:|---|---|---|
| 4 | 24 | pass | pass | 0.263 | 67 | 530 / 771 | 60.9 / 63.1 | 83% / 100% |
| 8 | 32 | pass | pass | 0.491 | 122 | 457 / 576 | 62.1 / 65.3 | 100% / 81% |
| 16 | 64 | **pass** (64/64, token audit clean) | **fail**: D1 `sw_power_cap`, 44% of samples below 1500 MHz (min 1380) | 0.914 | 238 | 461 / 584 | 64.6 / 71.4 | 97% / 52% |

The ladder stopped at 16 because the original code treated every Phase 3C
hard gate as a stop condition. The only failed gate was `fixed_clocks`: one L4
decode GPU reached its 72 W power cap with 16 concurrent decodes. The
connector itself was correct at 16; 32 and 64 were not tested.

Energy per output token fell from 5.1 J (cap 4) to 3.1 J (cap 8), consistent
with the idle-floor amortisation in Step 1; at cap 16 the clock audit failed
so no energy number is reported.

## Poisson sweep (cap 8)

| Offered req/s | Completed req/s | Backlog | TTFT p95 / attain | TPOT p95 / attain |
|---:|---:|---|---|---|
| 0.30 | 0.266 | stabilizes | 7541 ms / 81.7% (see stalls) | 132 / 73% |
| 0.60 | 0.433 | grows | 575 ms / 97.2% | 66 / 78% |

The 0.30 step contains three proxy stalls (5.7 s, 6.7 s and 21 s): for the
affected requests the client sent the request but the proxy only received it
5–21 s later; once received, routing took 0 ms, and in the same intervals
prefill HTTP calls appeared to take 1.7–4.2 s and decode headers 6.7 s. NVML
samplers on both nodes were on time, so the nodes were not frozen; the proxy
process was. The proxy appended each diagnostics record synchronously inside
the event loop to `results/` on Minerva's `/data` (ZFS, 96% full) while other
users' build jobs ran at 01:48–02:16. The 0.60 step, two minutes later, has no
stall (TTFT max 634 ms). The 0.30 "stabilizes/not sustainable" verdict is an
artefact of those stalls.

## Interpretation

- **Connector correctness is not the limit at 16.** Stage B's concurrency-4
  envelope was conservative by at least 4×.
- **Raw throughput scales almost linearly with concurrency** (0.26 → 0.49 →
  0.91 req/s; 67 → 238 tok/s), i.e. the two L4 decoders are nowhere near
  saturated at the C0 loads.
- **The 65 ms TPOT contract binds around 6–8 in-flight requests** (TPOT p95
  65.3 ms at cap 8, 71.4 ms at cap 16). SLO-bound capacity is therefore
  ~0.4 req/s with this trace: C0's HIGH-SUSTAINABLE (0.20 req/s, ~2.3 in
  flight) is ~50% of it and MEDIUM ~25%.
- **Locked HIGH is not sustainable on L4 above ~8 concurrent decodes.** The
  STATIC-HIGH baseline cannot hold its own operating point at high load;
  Minerva's "sustainable HIGH = 1500 MHz" was calibrated at low load.

## Fixes applied for the rerun

1. `disagg_proxy.py`: diagnostics are queued and written by a daemon thread;
   the event loop no longer touches the filesystem.
2. `stage_e0c_capacity.py`: the fixed-clock gate is recorded per step
   (`clock_valid`) and the ladder continues; only correctness/measurement
   gates stop it. The summary reports both the highest connector-valid and
   the highest clock-stable concurrency.

## Limitations

- One repeat per step; TTFT p95 at cap 4 (771 ms) includes first-request
  warm-up after server start.
- The sweep used cap 8, chosen by the old stop rule; the rerun sweeps at the
  highest connector-valid concurrency.

## Rerun r3 (job 275483, 2026-10-13 04:24–04:50, commit `ea4f3e1`)

Proxy stalls are gone (TTFT p95 581–600 ms in every sweep step). The ladder
again stopped at 16 with the same single failed gate (D1 `sw_power_cap`,
57.5% of samples at 1500 MHz, minimum 1380 MHz; all 11 correctness gates
passed). The first fix never ran: Phase 3C raises after writing
`audit.json` when any gate fails, so the step took the exception path. The
step now classifies the gates whenever `audit.json` exists (commit after
`ea4f3e1`).

| Offered req/s (Poisson, cap 8) | Completed req/s | Backlog | TTFT p95 / attain | TPOT p95 / attain | J/request |
|---:|---:|---|---|---|---:|
| 0.30 | 0.296 | drains | 581 ms / 100% | 64.3 ms / 98.3% | 1003 |
| 0.60 | 0.432 | grows | 587 ms / 95.8% | 65.9 ms / 77.8% | 698 |
| 1.00 | 0.476 | grows | 600 ms / 95.0% | 65.8 ms / 65.8% | 641 |

- **SLO-meeting capacity at cap 8 is between 0.30 and 0.60 req/s**; the
  throughput ceiling at cap 8 is ~0.48 req/s. The TPOT 65 ms contract fails
  first.
- **C0's HIGH-SUSTAINABLE (0.20 req/s) is ~2/3 of the SLO-meeting rate and
  ~40% of the throughput ceiling**, not "the highest sustainable" load; C0's
  overload at 0.30 was the concurrency-4 client cap.
- **Energy per request falls from ~2600 J (0.10 req/s, C1) to ~1000 J at
  0.30 and ~640 J at saturation**: at the C1 loads most energy is idle floor.

## Rerun r4 (job 275490, 2026-10-13 04:52–05:22, commit `c7e3796`)

| Cap | Correctness gates | Locked HIGH held | Saturated req/s | Output tok/s | TTFT p50/p95 | TPOT p50/p95 | D running max | J/output token |
|---:|---|---|---:|---:|---|---|---|---:|
| 4 | pass | yes | 0.263 | 67 | 530 / 816 | 59.8 / 63.1 | 2 / 2 | 5.06 |
| 8 | pass | yes | 0.490 | 122 | 504 / 591 | 61.8 / 65.3 | 4 / 5 | 3.05 |
| 16 | pass | **no** (D1 power cap) | 0.914 | 238 | 495 / 597 | 63.9 / 71.9 | 9 / 10 | 1.73 |
| 32 | **fail**: D1 engine died | — | — | — | — | — | — | — |

At cap 32 the D1 decode engine (L4, 22 GiB) crashed with
`torch.OutOfMemoryError` in `lm_head` while running 16 requests with the KV
cache at 98.4%; the same instant the P2P-NCCL connector logged
`[PUT]Recv Tensor, Out Of Memory` for an incoming layer. The connector
receives pushed KV tensors into GPU memory outside vLLM's KV-cache pool and
the scheduler applies no backpressure, so `--gpu-memory-utilization 0.82`
leaves too little headroom once ~16 decodes are resident per L4. The r4 sweep
then failed its metrics scrape (D1 dead) and is not used; the r3 sweep stands.

## Final E0-c conclusions

1. **Substrate concurrency ceiling: 16 in flight (≈8 per decode endpoint).**
   Correctness holds through 16; at 32 a decode engine OOMs. Stage B's
   concurrency-4 envelope was 4× conservative, but the ceiling is real and is
   a memory limit of the synchronous-PUT P2P-NCCL path on 22 GiB L4s, not a
   GPU-compute limit.
2. **Locked HIGH holds only up to 8 in flight.** Above that an L4 reaches its
   72 W cap and drops to 1380–1470 MHz, so STATIC-HIGH cannot keep its own
   operating point at high load.
3. **SLO-meeting capacity: 0.30–0.60 req/s**, bound by the 65 ms TPOT;
   throughput ceiling at cap 8 ≈ 0.48 req/s, at cap 16 ≈ 0.91 req/s.
4. **Energy per output token falls 3× from cap 4 to cap 16** (5.1 → 1.7 J),
   and per request from ~2600 J at C1's 0.10 req/s to ~640 J at saturation.
   The C1/C2a/D1 loads sat in the idle-floor-dominated regime.

For the re-grounded experiments (plan stage 3): define loads as a fraction of
the measured SLO-meeting capacity (~0.45 req/s with this mix), keep the client
cap at 8 per decode endpoint or below, and treat "locked HIGH" as valid only
up to 8 in flight on L4.
