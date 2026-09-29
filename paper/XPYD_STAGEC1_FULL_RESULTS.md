# XPYD Stage C1c full stationary results

**Status:** complete  
**Slurm job:** `269768` (`COMPLETED`, exit `0:0`, elapsed `02:49:15`)  
**Nodes:** Neptune (P0/P1, L40S) and Europa (D0/D1, L4)  
**Result directory:** `~/vLLM_test/results/stagec1_stationary/stagec1_full_neptune_europa_20260928/`

## Scope and frozen contract

This run evaluates the frozen Stage C1c question:

> Can lightweight runtime feedback discover and exploit SLO-safe DVFS
> headroom under stationary mixed workloads, without offline latency or power
> models?

The comparison is `STATIC-HIGH` versus `FEEDBACK-SAFE-SEARCH`. Both policies
use the same four-pair round-robin routing, TP=1, concurrency=4, workload mix,
and matched request traces. Every block starts at HIGH. Each load/policy has
three repeats, four 20-request windows per repeat, and therefore 240 requests.
The six load/repeat pairs are counterbalanced so that each policy runs first
three times. No repeat was discarded.

The primary contract is request-level TTFT <=600 ms and TPOT <=65 ms, with
at least 95% attainment for each metric. Exploration windows are included in
both energy and SLO results. C1c remains evidence-window-bounded; this run does
not test event-driven safety or compare adaptive search against one-step
search.

## Validity and audit result

The experiment is valid. All 48 measurement windows passed the existing hard
gates, including real SSE/latency, logical request IDs, token semantics, route
and endpoint coverage, fixed-clock readback, four valid NVML energy windows,
and no invalidating thermal or hardware slowdown. All 960 requests had real
SSE, propagated logical IDs, and exact requested/observed output-token counts.
Each of the four P-to-D routes carried exactly 120 requests per policy.

The run-level matched-trace, pool-frequency-symmetry, requested/readback-clock,
final-HIGH-restoration, and window-hard-gate audits all passed. Requested and
mean observed clocks agreed exactly in the exported window summaries for P0,
P1, D0, and D1. Measurement, energy, clock, and throttle validity were true for
every policy/load aggregate.

## Primary aggregate result

Values after `+/-` are sample standard deviations over the three matched
repeats. Energy is gross four-GPU board energy and includes exploration.

| Load | Policy | Energy for 240 requests | J/request | TTFT p95 / p99 / attainment | TPOT p95 / p99 / attainment | Completed req/s | Contract |
|---|---|---:|---:|---:|---:|---:|---|
| MEDIUM, 0.10 req/s | STATIC-HIGH | 628.06 kJ | 2616.91 +/- 3.39 | 559.6 / 577.1 ms / 99.17% | 63.23 / 63.24 ms / 100% | 0.09957 | PASS |
| MEDIUM, 0.10 req/s | FEEDBACK-SAFE-SEARCH | 569.11 kJ | 2371.28 +/- 3.61 | 570.4 / 631.2 ms / 95.00% | 63.24 / 65.38 ms / 96.25% | 0.09947 | **PASS** |
| HIGH-SUSTAINABLE, 0.20 req/s | STATIC-HIGH | 403.66 kJ | 1681.93 +/- 3.56 | 578.4 / 582.7 ms / 100% | 63.21 / 63.23 ms / 100% | 0.17239 | PASS |
| HIGH-SUSTAINABLE, 0.20 req/s | FEEDBACK-SAFE-SEARCH | 367.80 kJ | 1532.52 +/- 2.50 | 607.0 / 650.1 ms / 92.08% | 65.14 / 65.38 ms / 93.75% | 0.17204 | **FAIL** |

At MEDIUM, feedback saves **9.386% +/- 0.126 percentage points** versus its
matched STATIC repeats and passes the primary contract. At HIGH-SUSTAINABLE it
saves **8.883% +/- 0.275 percentage points in raw energy**, but this is not an
SLO-qualified saving because both attainment gates fail. The result exporter
correctly leaves the SLO-qualified HIGH saving unset.

Across both loads, the raw totals are 1031.72 kJ for STATIC and 936.91 kJ for
feedback, a 9.19% reduction. This combined number is descriptive only: it is
not a successful cross-load policy result because feedback fails the HIGH
contract.

## Matched repeat result

| Load | Repeat | STATIC J/req | FEEDBACK J/req | Raw saving | STATIC TTFT/TPOT attainment | FEEDBACK TTFT/TPOT attainment | Feedback contract |
|---|---:|---:|---:|---:|---:|---:|---|
| MEDIUM | 1 | 2613.12 | 2370.07 | 9.301% | 97.50% / 100% | 95.00% / 96.25% | PASS |
| MEDIUM | 2 | 2617.94 | 2368.43 | 9.531% | 100% / 100% | 95.00% / 96.25% | PASS |
| MEDIUM | 3 | 2619.66 | 2375.34 | 9.326% | 100% / 100% | 95.00% / 96.25% | PASS |
| HIGH-SUSTAINABLE | 1 | 1681.80 | 1529.77 | 9.040% | 100% / 100% | 91.25% / 93.75% | FAIL |
| HIGH-SUSTAINABLE | 2 | 1678.43 | 1534.66 | 8.566% | 100% / 100% | 93.75% / 93.75% | FAIL |
| HIGH-SUSTAINABLE | 3 | 1685.55 | 1533.11 | 9.044% | 100% / 100% | 91.25% / 93.75% | FAIL |

Every STATIC block passes. MEDIUM STATIC repeat 1 retains two first-window
TTFT outliers, including the 1497.7 ms first request; the full block still has
97.5% TTFT attainment. This unfavorable baseline observation was not removed.

## Search trajectory and exploration cost

Every feedback repeat produced the same measured search trajectory:

| Window | Measured P/D state | Evidence and next action |
|---:|---|---|
| 1 | P=2520, D=1500 MHz (HIGH/HIGH) | Both pools SAFE; probe P=MID and D=MID. |
| 2 | P=1890, D=1125 MHz (MID/MID) | P=MID SAFE; D=MID UNSAFE and recovered to HIGH. Probe P=LOW. |
| 3 | P=1260, D=1500 MHz (LOW/HIGH) | P=LOW UNSAFE and recovered to the validated P=MID bound. |
| 4 | P=1890, D=1500 MHz (MID/HIGH) | Final evidence SAFE; retain the measured best safe-energy point. |

Thus each repeat uses three probes, two unsafe probes, and two recoveries. The
D pool resolves after two windows/40 requests; the P pool resolves after three
windows/60 requests. Across each load this is 9 probes, 6 unsafe probes, 6
recoveries, 15 physical frequency transitions, and 5 reported control
oscillations. The final validated-safe and best measured safe-energy
frequencies are P=1890 MHz and D=1500 MHz in all six feedback blocks. This is
a per-pool measured search result, not an offline oracle.

The direct exploration cost is latency risk rather than extra gross energy:

| Load | Window/state | J/request | Matched raw energy change | TTFT attainment | TPOT attainment | Window interpretation |
|---|---|---:|---:|---:|---:|---|
| MEDIUM | W1 HIGH/HIGH | 2621.66 | +0.36% energy | 100% | 100% | Cold-start evidence |
| MEDIUM | W2 MID/MID | 2231.20 | -14.79% energy | 100% | 85.00% | D probe unsafe |
| MEDIUM | W3 LOW/HIGH | 2290.36 | -12.53% energy | 80.00% | 100% | P probe unsafe |
| MEDIUM | W4 MID/HIGH | 2341.89 | -10.57% energy | 100% | 100% | Learned safe state |
| HIGH | W1 HIGH/HIGH | 1678.00 | -0.10% energy | 100% | 100% | Cold-start evidence |
| HIGH | W2 MID/MID | 1436.19 | -14.63% energy | 100% | 75.00% | D probe unsafe |
| HIGH | W3 LOW/HIGH | 1495.17 | -11.16% energy | 68.33% | 100% | P probe unsafe |
| HIGH | W4 MID/HIGH | 1520.71 | -9.62% energy | 100% | 100% | Learned safe state |

The unsafe D probe causes 9 TPOT violations at MEDIUM and 15 at HIGH. The
unsafe P probe causes 12 TTFT violations at MEDIUM and 19 at HIGH. W1 and the
post-convergence W4 have 100% attainment at both loads. Consequently, MEDIUM
lands exactly on the 95% TTFT gate across all 240 feedback requests, whereas
the same fixed exploration budget makes HIGH fail in every repeat. The low-
frequency probe windows save raw energy, but those Joules cannot be counted as
SLO-qualified benefit when their latency contract fails.

Mean cumulative J/request and matched saving after each 20-request window are:

| Load | Requests | STATIC cumulative J/req | FEEDBACK cumulative J/req | Matched cumulative saving |
|---|---:|---:|---:|---:|
| MEDIUM | 20 | 2612.23 | 2621.66 | -0.362% |
| MEDIUM | 40 | 2615.29 | 2426.43 | 7.221% |
| MEDIUM | 60 | 2616.33 | 2381.07 | 8.992% |
| MEDIUM | 80 | 2616.91 | 2371.28 | 9.386% |
| HIGH | 20 | 1679.64 | 1678.00 | 0.098% |
| HIGH | 40 | 1681.02 | 1557.09 | 7.372% |
| HIGH | 60 | 1681.69 | 1536.45 | 8.636% |
| HIGH | 80 | 1681.93 | 1532.52 | 8.883% |

## Where the energy saving comes from

| Load | Pool | STATIC energy | FEEDBACK energy | Raw reduction |
|---|---|---:|---:|---:|
| MEDIUM | P0+P1 | 383.85 kJ | 331.40 kJ | 13.66% |
| MEDIUM | D0+D1 | 244.21 kJ | 237.71 kJ | 2.66% |
| HIGH-SUSTAINABLE | P0+P1 | 224.33 kJ | 193.92 kJ | 13.56% |
| HIGH-SUSTAINABLE | D0+D1 | 179.33 kJ | 173.88 kJ | 3.04% |

The feedback P-pool time residency is approximately 25% HIGH, 50% MID, and
25% LOW. D-pool residency is approximately 75% HIGH and 25% MID. LOW D is never
probed because MID already establishes the adjacent unsafe boundary. Most of
the gross energy reduction therefore comes from P-side DVFS.

Admission behavior does not explain the latency failures. Admission p95/p99 is
below 0.12 ms in every aggregate, client waiting never exceeds one, no window
has growing backlog, and proxy inflight maxima are 3 at MEDIUM and 4 at HIGH.
Feedback and STATIC completed throughput are nearly identical at each load.

## Post-hoc comparison with independent C1b

The online controller receives no C1b result. Nevertheless, C1c independently
recovers the C1b boundary exactly: P=MID/D=HIGH is SLO-safe, P=LOW is TTFT-
unsafe, and D=MID is TPOT-unsafe under the 600/65 contract. C1b measured about
10.28% and 9.58% fixed-point energy savings for P=MID/D=HIGH at MEDIUM and
HIGH, respectively; the C1c W4 matched savings of 10.57% and 9.62% are
consistent with those independent measurements. The full feedback-block
savings are smaller because they include the mandatory cold-start HIGH window
and all exploration.

## Answers to the frozen scientific questions

1. **Does feedback discover useful DVFS headroom?** Yes. In every repeat it
   discovers P=1890 MHz as the best measured safe P frequency and correctly
   retains D=1500 MHz after rejecting D=1125 MHz.
2. **Does it reduce total energy relative to matched STATIC-HIGH?** Yes at
   MEDIUM: 9.386% matched gross-energy saving with the contract satisfied. At
   HIGH it reduces raw energy by 8.883%, but that reduction is not an SLO-safe
   policy win.
3. **Does it satisfy the fixed SLO contract?** Only at MEDIUM. All three MEDIUM
   feedback blocks pass, exactly at 95% TTFT attainment. All three HIGH blocks
   fail both TTFT and TPOT attainment because unsafe candidates remain active
   for complete evidence windows. STATIC passes all six blocks.
4. **What exploration/convergence cost does it pay?** Three probes and 60
   requests to resolve P, two windows and 40 requests to resolve D, with two
   unsafe windows and two recoveries per repeat. The measurable cost is 21
   latency violations at MEDIUM and 34 at HIGH, not positive energy overhead;
   unsafe low-frequency windows consume less energy but violate the contract.
5. **Is it consistent with the independent C1b diagnosis?** Yes. The learned
   safe/unsafe frequency boundary and final safe-point energy saving agree with
   C1b. C1c therefore validates feedback discovery, but only partially
   validates the full policy: evidence-window-bounded exploration is too costly
   to preserve the TIGHT SLO at HIGH-SUSTAINABLE load.

No C2 experiment was started.
