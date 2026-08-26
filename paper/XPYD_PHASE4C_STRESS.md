# XpYd Phase 4C: passive-feedback robustness under stress

Phase 4C compares only `STATIC` and the unchanged `PASSIVE_FULL` controller.
It adds no predictors, probing, routing heuristic, or DVFS mechanism. The
validated topology remains P0/P1 on Neptune L40S and D0/D1 on io L4 at TP1,
with the Phase 3D LOW/MID/HIGH frequency states and compatibility rules.

## Controlled stress design

The tighter SLO is mean TTFT <= 550 ms and mean TPOT <= 65 ms. These targets
are below Phase 4B's 1000/80 ms limits but remain above all four Phase 4A
oracle winners' measured means. SLO violations are also reported from the
individual request records; violations are outcomes, not measurement failures.

The three traces are:

- moderate small@1.0 -> both-heavy@1.0 -> small@1.0;
- high prefill-heavy@2.0 -> decode-heavy@2.0 -> prefill-heavy@2.0;
- light@0.5 -> both-heavy burst@2.0 -> light recovery@0.5.

Each policy/trace block has three independent repeats and three windows per
state: one blind-transition window, one reaction opportunity, and one settling
check. Blocks are seeded and counterbalanced. PASSIVE gets one unmeasured
first-state all-route telemetry warmup with isolated controller state. STATIC
uses all compatible routes at HIGH. The first decision after a transition sees
only the previously observed workload context.

## Connector and load claim boundary

The physical P2pNcclConnector evidence currently supports one in-flight
request per selected route. Phase 4C therefore keeps `max_concurrency=1`.
Stress comes from higher offered rates, sustained demand/reduced idle gaps,
longer prompt/generation shapes, and stronger transitions. Requests waiting on
the client's concurrency semaphore have not yet been sent, so that wait is not
included in the reported TTFT. Phase 4C does not claim multi-request connector
concurrency robustness.

## Oracle boundary

The frozen Phase 4A oracle is never a controller input. A post-hoc oracle gap
is reported only for a window whose workload shape and 0.5 request/s rate
match Phase 4A and whose measured oracle winner satisfies the tighter Phase 4C
SLO. Higher-rate and burst windows explicitly report that the oracle reference
is unavailable. Phase 4C data never recomputes or extends the oracle.

## Run

After pulling the committed revision on Minerva, submit from the repository:

```bash
sbatch --nodelist=neptune,io \
  --export=ALL,EXP=E,L40S_NODE=neptune,L4_NODE=io,RESULT_DIR=results/phase4c_stress_runtime,XPYD_PHASE4C_CONFIG=paper/configs/xpyd_phase4b_evaluation_neptune_io.json,XPYD_PHASE4C_RUN_ID=phase4c_stress_r1_20260826,XPYD_PHASE4B_ORACLE_SUMMARY=results/phase4a_empirical_oracle/phase4a_oracle_r1_20260824T080026Z/phase4a_summary.json,XPYD_PHASE4C_ACCEPTED_DYNAMIC_AUDIT=results/phase4b_dynamic_evaluation/phase4b_dynamic_formal_r1_20260825/phase4b_audit.json \
  run_disagg_benchmark.sh
```

The required compact outputs are `phase4c_results.csv`,
`phase4c_summary.csv`, `phase4c_audit.json`, `phase4c_summary.json`, and
`phase4c_summary.md`. Raw window logs, NVML samples, requests, controls,
telemetry, and actuator records remain below `raw/`.

The summary applies the predeclared A/B/C/D decision gate. A failed measurement
audit is always `D. INCONCLUSIVE`; SLO or energy failure with valid measurement
evidence is a scientific B/C outcome and does not trigger controller changes.
