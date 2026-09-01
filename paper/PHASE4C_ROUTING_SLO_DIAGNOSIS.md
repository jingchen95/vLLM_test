# Phase 4C / 4C.1 routing and SLO diagnosis

This is an analysis-only comparison of accepted jobs `255354` (Phase 4C) and
`255370` (Phase 4C.1). No GPU job was run, no SLO was changed, and no
controller, routing, or DVFS code was modified.

## Decision

**E. MIXED**, with the evidence ordered as follows:

1. **SLO feasibility is the strongest established limitation.** STATIC at
   HIGH does not reliably meet 550 ms for prefill-heavy or both-heavy work.
2. **Routing/load distribution is a real structural confound.** PASSIVE sent
   all 243 prefill requests in each job to P0, while STATIC split them 124/119
   over P0/P1. The existing runs do not isolate the causal cost of that
   concentration.
3. **Run variance is material near 550 ms.** It is large enough to change many
   pass/fail labels, although the randomized interleaving provides no evidence
   of a simple directional policy-order bias.
4. **The old DVFS recovery defect was real and was corrected, but remaining
   failures are mostly not low-frequency failures.** In fixed job `255370`, 32
   of 39 STATIC-pass/PASSIVE-fail pairs occurred with the selected P already at
   HIGH.

This evidence does not support attributing the remaining gap primarily to the
fixed DVFS controller. It also does not yet prove that single-P concentration
is the main causal source; the necessary matched block-level routing control is
missing.

## 1. Is 550 ms feasible for STATIC at HIGH?

No, not reliably for the two prefill-intensive workloads. Each independent job
produced exactly 49/243 STATIC violations (20.16%) despite all four endpoints
being at HIGH. All reported request violations in both jobs were TTFT
violations; no request exceeded the 65 ms TPOT SLO. Pooling the two accepted
STATIC runs gives:

| Workload | N | Mean | P50 | P95 | P99 | Min--max | >550 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| small/light | 216 | 272.0 | 282.4 | 338.7 | 354.7 | 152.3--358.0 | 0.0% |
| decode-heavy | 54 | 247.5 | 258.6 | 312.8 | 327.7 | 145.8--337.2 | 0.0% |
| prefill-heavy | 108 | 513.2 | 519.6 | 590.5 | 602.8 | 421.3--607.2 | 36.1% |
| both-heavy | 108 | 538.5 | 559.6 | 622.5 | 650.8 | 433.2--663.0 | 54.6% |

Sensitivity analysis does not change the official 550 ms SLO, but shows where
the boundary lies:

| Workload | >500 ms | >550 ms | >600 ms | >650 ms |
|---|---:|---:|---:|---:|
| prefill-heavy | 59.3% | 36.1% | 2.8% | 0.0% |
| both-heavy | 70.4% | 54.6% | 14.8% | 1.9% |

Thus 550 ms is inside the natural STATIC@HIGH distribution rather than above
its tail. Phase 4C can still use that SLO as a stress objective, but it cannot
interpret every violation as a controller regression. Per-trace,
segment, and repeat statistics are preserved in
`results/phase4c_routing_slo_diagnosis/static_high_ttft.csv`.

## 2. Routing and load distribution

The routing difference is exact and reproducible across both jobs:

| Job/policy | P0 | P1 | P0-D0 | P0-D1 | P1-D0 | P1-D1 |
|---|---:|---:|---:|---:|---:|---:|
| 255354 STATIC | 124 | 119 | 62 | 62 | 59 | 60 |
| 255354 PASSIVE | 243 | 0 | 81 | 162 | 0 | 0 |
| 255370 STATIC | 124 | 119 | 62 | 62 | 59 | 60 |
| 255370 PASSIVE | 243 | 0 | 177 | 66 | 0 | 0 |

In fixed job `255370`, the high prefill/decode/prefill trace used P0-D0 for all
81 PASSIVE requests. The other two PASSIVE traces also used P0 exclusively;
only the D endpoint changed. STATIC remained nearly even over all four routes.

There is no endpoint-queue evidence of overload: the accepted per-window
endpoint summaries contain 0 nonzero queue observations out of 324
endpoint-windows per policy/job, with maximum mean queue depth 0. KV usage was
also low (maximum 0.0633 for PASSIVE and 0.0426 for STATIC). This is consistent
with `max_concurrency=1`: waiting before the client semaphore is invisible to
the endpoint queue. Request counts therefore provide the load-concentration
evidence; queue depth does not refute or confirm a concentration penalty.

STATIC does not show a stable intrinsic P0 disadvantage. For example, in
`255370`, STATIC prefill-heavy mean TTFT was 522.1 ms on P0 versus 518.6 ms on
P1, and both-heavy was 530.6 versus 535.2 ms. In `255354`, P1 was slower for
both heavy classes. The remaining PASSIVE gap therefore cannot be explained as
simply selecting an intrinsically slower P0.

## 3. Matched-condition evidence and its limit

Within `255370`, request positions were matched by trace, repeat, state,
window, and request ordinal. Restricting further to selected P=HIGH and the
same exact P-D route gives:

| Workload | Pairs | STATIC mean | PASSIVE mean | Mean difference | STATIC/PASSIVE violations |
|---|---:|---:|---:|---:|---:|
| small/light | 18 | 282.0 | 284.1 | +2.1 ms | 0 / 0 |
| decode-heavy | 7 | 238.3 | 265.1 | +26.8 ms | 0 / 0 |
| prefill-heavy | 14 | 522.2 | 549.8 | +27.6 ms | 7 / 7 |
| both-heavy | 9 | 519.6 | 556.7 | +37.0 ms | 4 / 6 |
| all | 48 | 390.2 | 409.9 | +19.7 ms | 11 / 13 |

The light case is nearly identical, while heavy matched requests retain a
roughly 28--37 ms PASSIVE penalty. This rules out current-request route and
frequency as a complete explanation. It does **not** isolate block-level load
distribution: the matched PASSIVE request still occurs inside a P0-only block,
whereas neighboring STATIC requests are spread across P0/P1. The two policies
also run in separate physical blocks.

A defensible causal routing comparison is therefore impossible from the
existing data. The minimal missing evidence is a counterbalanced HIGH-frequency
comparison between P0-D0-only routing and the existing four-route round-robin,
under the same workload and block protocol.

## 4. What did the recovery fix improve?

| Metric | 255354 old PASSIVE | 255370 fixed PASSIVE |
|---|---:|---:|
| Total request violations | 79/243 (32.51%) | 77/243 (31.69%) |
| Total gross energy | 814.34 kJ | 886.86 kJ |
| J/request | 3351.2 | 3649.6 |
| Selected-P residency by request | HIGH 15, MID 42, LOW 186 | HIGH 195, MID 24, LOW 24 |
| DVFS recommendations | 32 STEP_DOWN, 0 STEP_UP | 13 STEP_DOWN, 3 RECOVER_HIGH |

The clearest improvement is the decode-heavy to prefill-heavy return:

| Metric | Old | Fixed | Change |
|---|---:|---:|---:|
| Selected P state | LOW for 27/27 requests | HIGH for 27/27 | unsafe LOW removed |
| Mean TTFT | 612.9 ms | 548.6 ms | -64.3 ms |
| P95 TTFT | 675.4 ms | 626.4 ms | -49.0 ms |
| Violations | 23/27 | 17/27 | -6 |
| Segment energy | 49.63 kJ | 53.59 kJ | +3.96 kJ |

This trace did not need to execute `RECOVER_HIGH`: aligned downshift thresholds
prevented the old unsafe descent, so it remained HIGH. The three actual
`RECOVER_HIGH` actions occurred in other both-heavy transitions. Consequently,
the combined rerun validates the effect of SLO alignment plus asymmetric
recovery as a policy package, but it cannot apportion the six improved requests
between the two changes.

The overall violation count improved by only two because other near-boundary
segments varied in the opposite direction. For example, the first
prefill-heavy segment went from 12 to 19 PASSIVE violations across the two
jobs even though the fixed run was at HIGH. Cross-job differences must not be
treated as literal paired causal effects.

## 5. Run order, clocks, and noise

- Both jobs used the exact same randomized 18-block order. PASSIVE blocks had
  mean position 9 and STATIC blocks mean position 10, so there is no simple
  “PASSIVE always ran later” confound.
- The two physical runs were about nine hours apart and each lasted about four
  hours. Background-load samples were not recorded.
- Frequency readback was stable: the minimum endpoint/window target-match
  fraction was 0.9977 in `255354` and 0.9956 in `255370`; none was below 0.95.
  All accepted throttle/thermal hard gates passed.
- Actual temperature samples were not preserved in the compact or runtime
  logs, so temperature drift cannot be quantified. Passing the throttle gate
  only excludes an invalidating thermal/HW slowdown; it is not a temperature
  distribution.
- Across the 27 matching STATIC trace/segment/repeat cells, the cross-job
  change in mean TTFT had mean absolute magnitude 19.7 ms, standard deviation
  27.5 ms, and range -58.3 to +72.3 ms. Within-job repeat-mean standard
  deviations had medians 20.0 ms (`255354`) and 9.9 ms (`255370`), with maxima
  38.2 and 31.0 ms.

This variance is comparable to the 20--37 ms heavy-workload matched gaps and
can move a meaningful fraction of requests across 550 ms. It does not by
itself explain the persistent structural P0-only routing or prove that the
entire net PASSIVE gap is noise.

## 6. Fixed-run paired failures

The 243 request-position pairs in `255370` divide into:

- both pass: 155;
- STATIC pass / PASSIVE fail: 39;
- STATIC fail / PASSIVE pass: 11;
- both fail: 38.

The net fixed-PASSIVE excess is 28 violations. Among the 39
STATIC-pass/PASSIVE-fail candidates:

| Selected P state | Count | Share |
|---|---:|---:|
| HIGH | 32 | 82.1% |
| MID | 2 | 5.1% |
| LOW | 5 | 12.8% |

All 39 occurred on P0: 29 on P0-D0 and 10 on P0-D1. They comprise 21
prefill-heavy and 18 both-heavy cases. All decision/end-window queue evidence
is zero. For the 32 HIGH cases, frequency recovery had already completed or
was unnecessary; these cannot be described as failures caused by remaining at
LOW/MID. Only seven cases are even candidates for a remaining frequency-state
contribution, and this is an upper bound rather than causal attribution.

## 7. Smallest justified next action

Run **one minimal matched-routing experiment**, and nothing broader:

- keep all four GPUs at the validated HIGH frequencies;
- compare fixed P0-D0-only routing against the existing four-route
  round-robin;
- use only prefill-heavy and both-heavy, the two workloads that cross 550 ms;
- counterbalance A/B execution order;
- reuse the existing three repeats x three requests/window protocol and all
  current audits.

This is 36 measured requests total and isolates the remaining question with
the least GPU time. If concentrated HIGH routing reproduces the penalty, the
next mechanism question is load-balanced routing. If it does not, the present
PASSIVE-vs-STATIC difference should be interpreted primarily under the
documented SLO feasibility and run-variance limits. No further DVFS change is
justified before that result.

## Reproducibility artifacts

- `scripts/analyze_phase4c_routing_slo.py`
- `results/phase4c_routing_slo_diagnosis/static_high_ttft.csv`
- `results/phase4c_routing_slo_diagnosis/routing_load.csv`
- `results/phase4c_routing_slo_diagnosis/matched_conditions.csv`
- `results/phase4c_routing_slo_diagnosis/paired_fixed_breakdown.csv`
- `results/phase4c_routing_slo_diagnosis/recovery_effect.csv`
- `results/phase4c_routing_slo_diagnosis/phase4c_routing_slo_diagnosis.json`

The JSON records SHA-256 hashes of every compact accepted input used by the
analysis. Raw experiment output is not duplicated in the repository.
