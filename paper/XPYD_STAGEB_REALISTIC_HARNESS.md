# XPYD Stage B — Realistic Workload Harness Validation

## Purpose

Validate concurrent mixed-request semantics before any new feedback-control
experiment. This is a substrate and measurement test, not an optimization
evaluation.

## Fixed system

- P0/P1: Neptune L40S, TP1, 2520 MHz
- D0/D1: io L4, TP1, 1500 MHz
- all four validated P→D pairs
- deterministic request-level round-robin
- no routing controller
- no DVFS controller
- no probing or predictive model

The launcher starts one persistent 2P2D server set. The Stage-B runner then
executes the following sequence with hard gates between stages:

1. `B1_concurrency_2_smoke`: 32 stationary mixed requests, concurrency 2.
2. `B2_concurrency_4_smoke`: the same mixed population, concurrency 4; runs
   only if B1 passes.
3. `B3_dynamic_harness_smoke`: 80 mixed requests across light, prefill-burst,
   decode-period, and recovery segments; runs only if B2 passes.

## Hard gates

Each stage requires:

- all accepted Phase 3C SSE, request-ID, token, route, Prometheus, NVML energy,
  fixed-clock, throttle, and compatibility gates;
- exact SS/SL/LS/LL and segment labels;
- a genuinely mixed request population;
- requested client concurrency reached but not exceeded;
- complete arrival/admission/completion event accounting;
- final client and proxy inflight counts drained to zero;
- proxy accepted/completed deltas equal logical request count, with no failures;
- proxy maximum inflight demonstrates the requested concurrency;
- all endpoints fixed at HIGH with hardware readback;
- no controller or policy mutation.

Failure is fail-closed. Later stages are not attempted.

## Submission

From `~/vLLM_test` on Minerva:

```bash
mkdir -p logs
sbatch \
  --job-name=xpyd_stageb \
  --partition=long \
  --nodes=2 \
  --ntasks=16 \
  --ntasks-per-node=8 \
  --nodelist=neptune,io \
  --exclusive \
  --mem=0 \
  --time=03:00:00 \
  --output=logs/xpyd_stageb_%j.log \
  --export=ALL,EXP=E,L40S_NODE=neptune,L4_NODE=io,XPYD_STAGEB_RUNTIME_CONFIG=paper/configs/xpyd_phase4b_evaluation_neptune_io.json,XPYD_STAGEB_WORKLOAD_CONFIG=paper/configs/xpyd_stageb_workload_harness.json,XPYD_STAGEB_RUN_ID=stageb_b1_b3_20260903 \
  run_disagg_benchmark.sh
```

If either named node is drained/down, Slurm keeps the job pending. It must not
fall back to another GPU type.

## Outputs

```text
results/stageb_realistic_harness/<run_id>/
  stageb_plan.json
  B1_concurrency_2_smoke/
  B1_concurrency_2_smoke_audit.json
  B2_concurrency_4_smoke/
  B2_concurrency_4_smoke_audit.json
  B3_dynamic_harness_smoke/
  B3_dynamic_harness_smoke_audit.json
  stageb_summary.json
  stageb_summary.md
```

Raw per-stage client events, proxy diagnostics, endpoint metrics, and NVML
samples remain under each stage directory.

## Claim boundary

A passing run establishes that the physical 2P2D path and measurement system
support the selected concurrent mixed workloads. It does not establish that
round-robin is efficient, that an SLO is feasible, or that any controller is
effective.
