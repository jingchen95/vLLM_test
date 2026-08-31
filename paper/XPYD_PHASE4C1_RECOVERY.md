# XpYd Phase 4C.1: aligned-SLO asymmetric recovery

Phase 4C.1 is a controlled correction of the safety-policy mismatch diagnosed
after job `255354`. It preserves the original Phase 4C topology, policies,
three traces, rates, order seed, three repeats, three windows per state,
request count, telemetry, routing, and physical measurement path.

## Exact controller change

Only two behaviors change for the Phase 4C.1 overlay configuration:

1. The controller receives the same TTFT/TPOT SLO used by evaluation:
   550/65 ms. The existing relative thresholds and EWMA alpha remain unchanged.
2. A measured latency EWMA above the existing risk boundary recovers directly
   from LOW or MID to HIGH. A downshift remains exactly one level. The latency
   fast path does not require queue pressure.

With the retained 0.8/0.5 fractions, the P-side risk/downshift boundaries are
440/275 ms under the aligned 550 ms TTFT SLO. The controller still uses alpha
0.5, passive routing, telemetry freshness, compatibility fail-closed behavior,
and hardware frequency readback. No predictor, probing, bandit, RL, offline
lookup, endpoint activation, power-limit change, or concurrency change is
introduced.

The original `xpyd_phase4b_evaluation_neptune_io.json` remains unchanged and
continues to reproduce the legacy Phase 4C controller. The small
`xpyd_phase4c1_recovery_neptune_io.json` overlay selects the correction and a
separate result root.

## CPU validation

Tests establish that:

- legacy control still steps up one level;
- recovery mode jumps directly from LOW to HIGH on latency risk with queue 0;
- recovery mode still steps down one level;
- the measured Phase 4C transition values (285 then 633 ms at alpha 0.5) produce
  EWMA 459 ms and trigger recovery under the aligned 550 ms SLO;
- the overlay changes neither traces nor repeat/window/request counts;
- the original Phase 4C configuration remains in legacy mode;
- Phase 4C.1 reports SLO results relative to its paired STATIC baseline, rather
  than repeating the invalid absolute interpretation of the legacy run.

## Physical rerun

After pulling the committed revision on Minerva, submit the same physical
matrix on Neptune/io with only the overlay config and output/run IDs changed:

```bash
sbatch --nodelist=neptune,io \
  --export=ALL,EXP=E,L40S_NODE=neptune,L4_NODE=io,RESULT_DIR=results/phase4c1_recovery_runtime_r1,XPYD_PHASE4C_CONFIG=paper/configs/xpyd_phase4c1_recovery_neptune_io.json,XPYD_PHASE4C_RUN_ID=phase4c1_recovery_r1_20260831,XPYD_PHASE4B_ORACLE_SUMMARY=results/phase4a_empirical_oracle/phase4a_oracle_r1_20260824T080026Z/phase4a_summary.json,XPYD_PHASE4C_ACCEPTED_DYNAMIC_AUDIT=results/phase4b_dynamic_evaluation/phase4b_dynamic_formal_r1_20260825/phase4b_audit.json \
  run_disagg_benchmark.sh
```

Acceptance is based on paired evidence:

- all physical and correctness hard gates remain valid;
- controller TTFT/TPOT SLOs exactly match evaluation;
- direct latency-risk recovery is enabled and physically read back;
- PASSIVE request-level SLO violation rate is no worse than the simultaneous
  STATIC comparator;
- PASSIVE retains positive mean energy savings in every trace.

This test can support or reject the small recovery fix. It cannot establish a
theoretical optimum or justify a different research mechanism.
