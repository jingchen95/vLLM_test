# Phase 4C.2 minimal matched-routing experiment

Phase 4C/4C.1 analysis found that the official 550 ms TTFT SLO lies inside the
STATIC@HIGH heavy-workload distribution, fixed PASSIVE still concentrates all
prefill requests on P0, and 82.1% of its remaining excess failures occur at
P-side HIGH. Phase 4C.2 therefore isolates routing concentration without
changing the SLO or controller.

## Fixed design

- P0/P1: Neptune L40S, TP1, validated HIGH = 2520 MHz.
- D0/D1: io L4, TP1, validated HIGH = 1500 MHz.
- `BALANCED_HIGH`: expose all four validated P-D routes to the existing
  round-robin proxy.
- `P0D0_HIGH`: expose only P0-D0.
- Workloads: validated prefill-heavy and both-heavy at 2 request/s.
- Three repeats, one three-request measured window per policy/workload/repeat:
  12 windows and 36 measured requests total.
- Each measured block receives an unmeasured three-request route-matched
  prefill priming window. Pair order is adjacent and counterbalanced: each
  policy runs first in three of the six workload/repeat pairs.
- Official SLO remains TTFT/TPOT = 550/65 ms.

There is no feedback decision, DVFS transition, predictor, route probe, model,
oracle decision, or SLO tuning. Phase 4A data are not used to choose an action.

## Outputs and hard gates

The job writes:

```text
phase4c_matched_measurements.csv
phase4c_matched_summary.csv
phase4c_matched_audit.json
phase4c_matched_summary.json
phase4c_matched_summary.md
raw/phase4c_matched_requests.csv
raw/phase4c_matched_endpoint_summary.csv
raw/phase4c_matched_plan.json
```

Acceptance requires the previously accepted Phase 4C.1 audit, all 12 windows
and 36 logical requests, valid SSE/token/ID/route/energy audits, exact P0-D0
concentration, both-P coverage in the balanced arm, HIGH requested/observed
clock agreement, no invalidating thermal/HW slowdown, and safe HIGH
restoration.

The result is a small measured routing comparison, not a theoretical optimum.

## Submission

After pulling the committed revision on Minerva:

```bash
sbatch --nodelist=neptune,io \
  --export=ALL,EXP=E,L40S_NODE=neptune,L4_NODE=io,RESULT_DIR=results/phase4c_matched_runtime_r1,XPYD_PHASE4C_CONFIG=paper/configs/xpyd_phase4b_evaluation_neptune_io.json,XPYD_PHASE4C_MATCHED=1,XPYD_PHASE4C_RUN_ID=phase4c_matched_r1_20260901,XPYD_PHASE4B_ORACLE_SUMMARY=results/phase4a_empirical_oracle/phase4a_oracle_r1_20260824T080026Z/phase4a_summary.json,XPYD_PHASE4C_MATCHED_ACCEPTED_AUDIT=results/phase4c1_recovery/phase4c1_recovery_r3_20260831/phase4c_audit.json \
  run_disagg_benchmark.sh
```

Expected runtime is roughly 45--75 minutes including endpoint startup,
route-matched priming, measurements, and teardown. No continuous monitoring is
required; inspect the compact audit and summary after Slurm completion.
