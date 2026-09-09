# XPYD Stage C1 DVFS-headroom diagnosis

This diagnostic separates two explanations for the C1 smoke result: the
feedback controller may be too conservative, or the mixed workload may have
little SLO-safe frequency headroom. It is deliberately not a new policy
evaluation. Every point is fixed-frequency, uses the same four-pair round-robin
route and deterministic 35/15/35/15 mixed trace as C1, and is repeated twice.

The first small point set is:

| Point | P pool | D pool | Purpose |
|---|---:|---:|---|
| `HIGH_HIGH` | HIGH (2520 MHz) | HIGH (1500 MHz) | matched baseline |
| `P_HIGH_D_MID` | HIGH | MID (1125 MHz) | D-only one-step reduction |
| `P_HIGH_D_LOW` | HIGH | LOW (750 MHz) | D-only minimum tested point |
| `P_MID_D_HIGH` | MID (1890 MHz) | HIGH | P-only one-step reduction |
| `P_LOW_D_HIGH` | LOW (1260 MHz) | HIGH | P-only minimum tested point |

No combined lower point is included initially. It should be added only if both
one-dimensional axes show repeatable SLO-safe headroom.

## C1 controller diagnosis

The existing measured-feedback scheduler uses EWMA latency and requires fresh
telemetry, healthy ACTIVE endpoints, observed queue/KV state, and low queue/KV
for a downshift. With the C1 settings, the exact step-down conditions are:

- P endpoint EWMA TTFT `<= 0.50 × 600 = 300 ms`, queue `<= 0`, KV `<= 0.50`;
- D endpoint EWMA TPOT `<= 0.50 × 65 = 32.5 ms`, queue `<= 0`, KV `<= 0.50`.

The step-up/risk condition is EWMA latency `> 0.80 × SLO` (480 ms TTFT or
52 ms TPOT), or queue above 1 or KV above 0.90. With the C1 recovery setting,
latency risk returns directly to HIGH. Severe queue/KV pressure also falls
back to HIGH. A missing/stale latency observation does not justify lowering
frequency.

Job `259228` measured the following before the controller's HOLD decisions:

| Load | P EWMA TTFT | P downstep gate | D EWMA TPOT | D downstep gate | Result |
|---|---:|---:|---:|---:|---|
| MEDIUM | 342.5–343.1 ms | 300 ms | 58.57–58.57 ms | 32.5 ms | P HOLD; D HOLD at HIGH |
| HIGH-SUSTAINABLE | 365.4–369.6 ms | 300 ms | 59.53–59.53 ms | 32.5 ms | P HOLD; D HOLD at HIGH |

The observed client admission p95 was below 0.13 ms and close client waiting
was zero, so client pressure did not force the HOLD. P-side EWMA was above the
downstep gate, while D-side EWMA was above its risk gate and already at HIGH.
The fixed-point experiment therefore tests lower frequencies directly; it does
not loosen a controller threshold based on one anomalous request.

## Reproduction

```bash
export XPYD_STAGEC1_HEADROOM_RUNTIME_CONFIG=paper/configs/xpyd_stageb_runtime_neptune_europa.json
export XPYD_STAGEC1_HEADROOM_EXPERIMENT_CONFIG=paper/configs/xpyd_stagec1_headroom_neptune_europa.json
export XPYD_STAGEC1_HEADROOM_RUN_ID=stagec1_headroom_neptune_europa_20260909
bash run_disagg_benchmark.sh
```

The run emits per-window and per-repeat CSV files, observed clock/energy
readbacks, a JSON audit, and `headroom_diagnosis.md`. A point is called
SLO-safe only when both aggregate 95% attainment gates pass and measurement,
clock, energy, matched-trace, and final-HIGH-restoration audits pass.
