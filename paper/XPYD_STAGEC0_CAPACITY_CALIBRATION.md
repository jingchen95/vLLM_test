# XPYD Stage C0 — Capacity and Load Calibration

Stage C0 identifies realistic load regimes before any feedback controller is
enabled. It reuses the accepted Stage-B 2P2D transport, client, proxy,
Prometheus, NVML, request-ID, token, SSE, and fixed-clock audit path.

## Fixed experiment semantics

- P0/P1: Neptune L40S, 2520 MHz (validated HIGH)
- D0/D1: Europa L4, 1500 MHz (validated sustainable HIGH)
- TP1 per endpoint; max client concurrency 4
- round-robin over all four validated P-to-D pairs
- stationary mixture: 35% SS, 15% SL, 35% LS, 15% LL
- 48 requests per point at 0.05, 0.10, 0.15, 0.20, and 0.30 req/s
- evaluation SLO: TTFT 550 ms and TPOT 65 ms
- no controller, DVFS policy, adaptive routing, predictor, or training

The range brackets the approximately 0.18 req/s drained throughput observed in
Stage B while adding a low-load point with 20 seconds between arrivals. Each
point runs for multiple minutes and the analysis measures admission-delay and
client-waiting trends over the offered interval. Post-arrival draining is not
misclassified as steady-state sustainability.

Endpoint `running` and `waiting` values are now preserved in
`derived/endpoint_load.csv`. They are the same existing fixed-period
Prometheus samples; parsing and metric semantics are unchanged. Client backlog
is reconstructed from `TRACE_ARRIVAL`, `CLIENT_ADMIT`, and `REQUEST_COMPLETE`
events. Proxy inflight is reconstructed from exact route and completion times.

## Outputs

The run directory contains the full Phase 3C evidence for every point plus:

```text
stagec0_plan.json
stagec0_measurements.csv
stagec0_summary.json
stagec0_summary.md
```

The summary selects LOW, MEDIUM, and HIGH-SUSTAINABLE from valid capacity
measurements whose drained throughput tracks offered load and whose admission
backlog does not diverge. It reports SLO eligibility as a separate Stage C1
gate: capacity calibration remains useful even when the fixed-HIGH baseline
reveals that the chosen SLO is not currently achievable. An overload point is
expected and is not itself a failed measurement.

## Submission

Activate the accepted runtime before submitting:

```bash
source ~/.bashrc
conda activate cuda-env
cd ~/vLLM_test
git pull --ff-only
mkdir -p logs

sbatch \
  --job-name=xpyd_stagec0 \
  --partition=long \
  --nodes=2 \
  --ntasks=16 \
  --ntasks-per-node=8 \
  --nodelist=neptune,europa \
  --exclusive \
  --mem=0 \
  --time=03:00:00 \
  --output=logs/xpyd_stagec0_%j.log \
  --export=ALL,EXP=E,L40S_NODE=neptune,L4_NODE=europa,XPYD_STAGEC0_RUNTIME_CONFIG=paper/configs/xpyd_stageb_runtime_neptune_europa.json,XPYD_STAGEC0_WORKLOAD_CONFIG=paper/configs/xpyd_stagec0_capacity_neptune_europa.json,XPYD_STAGEC0_RUN_ID=stagec0_neptune_europa_20260904 \
  run_disagg_benchmark.sh
```

Expected wall time is roughly 55–90 minutes including server startup, all five
load points, drain time, and cleanup. The three-hour allocation is a safety
bound, not the expected duration.
