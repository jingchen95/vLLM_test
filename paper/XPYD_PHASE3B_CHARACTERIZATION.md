# XpYd Phase 3B fixed-frequency workload characterization

This phase measures whether P0 and D0 GPU-board energy contributions change
with workload shape. It is **fixed-frequency characterization using sustainable
hardware operating points** and remains measurement-only/model-free: no model
is trained, no routing changes, and no scheduler or feedback policy is enabled.

## Controlled topology and clocks

- P0: Neptune GPU 0, NVIDIA L40S, graphics 2520 MHz, memory 9001 MHz. The
  existing 2520 MHz operating point is verified by the active-workload clock
  audit in every smoke/repeat window; Neptune's post-reboot read-only NVML
  capability check supports the required energy source.
- D0: Europa GPU 0, NVIDIA L4, graphics 1500 MHz, memory 6251 MHz. 1500 MHz
  is the sustainable active-decode operating point established by the Europa
  frequency probe; it is a D0 setting, not a P0 setting.
- Serving: Mistral-7B, vLLM 0.15.1, P2pNcclConnector, TP=1.

The characterization branch is explicitly actuated, unlike the read-only
Phase 3B preflight. It locks graphics and memory clocks once before serving,
records before/after readback, and restores both defaults in the unconditional
job EXIT trap. It does not enable persistence mode or change power limits.
There are no clock changes between workloads and no telemetry-driven DVFS.
The NVML measurement processes remain isolated and strictly read-only.

## Matrix

The checked-in matrix contains IL128/OL128, IL2048/OL128, IL128/OL512, and
IL2048/OL512. The formal run uses five repeat-level observations per workload
and ten successful requests per repeat at client concurrency one. Workload
order rotates deterministically between repeat blocks. Each repeat has a
three-second cooldown, ten-second idle window, one excluded same-shape warmup,
and one measured workload window.

Run the four-cell smoke first (one request per workload):

```bash
sbatch --nodelist=neptune,europa \
  --export=ALL,EXP=E,L40S_NODE=neptune,L4_NODE=europa,RESULT_DIR=results/xpyd_phase3b_characterization_smoke_runtime,XPYD_PHASE3B_CHARACTERIZATION_CONFIG=paper/configs/xpyd_phase3b_characterization_l40s_l4.json,XPYD_PHASE3B_CHARACTERIZATION_REPEATS=1,XPYD_PHASE3B_CHARACTERIZATION_REQUESTS=1 \
  run_disagg_benchmark.sh
```

Only after the smoke audit passes, run the formal matrix by omitting the two
override variables. Use a unique `RESULT_DIR` and
`XPYD_PHASE3B_CHARACTERIZATION_RUN_ID` for every submission.

## Validity and outputs

Each repeat retains real-SSE client TTFT/TPOT/ITL, endpoint Prometheus queue/KV
windows, exact logical request IDs, P0/D0 request and token deltas, endpoint
energy and idle windows, coverage/drift/missed/error counts, and workload clock
match fractions. A repeat is valid only when both energy windows, both active-
workload clock audits, the Phase 3A audit, token accounting, and P0+D0
aggregation pass.

For Phase 3B, continuous Phase 3A `/metrics` scraping is auxiliary telemetry:
scrape success, misses, longest gaps, queue, and KV observations are retained
and reported, but temporary scrape timeouts do not invalidate an otherwise
complete offline energy window. The hard gates remain real SSE, exact logical
ID/token accounting, client TTFT/TPOT/E2E validity, P0/D0 energy-window and
sampling validity, fixed-clock validity, and absence of an invalidating
thermal/HW slowdown.

Primary energy is gross GPU-board hardware-counter energy. Idle-adjusted energy
is a separately labelled estimate. `P/(P+D)` and `D/(P+D)` are limited
GPU-board contribution shares, not whole-system attribution. CPU/node, NIC,
network, KV-transfer-specific, cooling, and facility energy remain unobserved.

The inference unit is the independent repeat, not an individual 200 ms NVML
sample. The compact summary reports repeat-level median and a deterministic
bootstrap 95% interval when at least two valid repeats exist. Raw results under
`results/**` remain ignored and must not be committed.

## Phase 3B.1 prefill-energy identifiability

The follow-up configuration
`paper/configs/xpyd_phase3b1_prefill_identifiability_l40s_l4.json` reuses the
same P/D proxy, real-SSE audit, fixed-frequency clocks, and read-only NVML
samplers. It phase-aligns P0 prefill intervals using proxy wall-clock
diagnostics and linearly interpolates the cumulative P0 NVML energy counter
between neighboring samples. Negative incremental values are retained as
measurement residuals and are never clamped. The analysis writes
`summary.json`, `summary.md`, `prefill_identifiability.csv`, and the normal
characterization artifacts in a fresh run directory.

The current launcher sets `--max-model-len 4096`. Therefore IL4096 with OL128,
IL8192, and IL16384 are explicitly skipped rather than forcing a context-length
or serving-semantics change. IL3072/OL128 is the largest safe intermediate
probe in the checked-in configuration.
