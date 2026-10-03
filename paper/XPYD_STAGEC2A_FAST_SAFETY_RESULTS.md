# XPYD Stage C2a fast-safety results

**Status:** complete and accepted  
**Slurm job:** `271338` (`COMPLETED`, exit `0:0`, elapsed `04:07:45`)  
**Nodes:** Neptune (P0/P1, L40S) and Europa (D0/D1, L4)  
**Result directory:** `~/vLLM_test/results/stagec2a_fast_safety/stagec2a_full_neptune_europa_20261003/`

## Scope and frozen contract

Stage C2a asks whether request-level event-driven recovery reduces the SLO cost
of online DVFS exploration relative to the C1 evidence-window boundary. It
compares:

- `STATIC-HIGH`;
- `WINDOW-BOUNDED-FEEDBACK`, which is the C1 exploration and recovery path;
- `FAST-SAFETY-FEEDBACK`, which uses the identical exploration algorithm,
  evidence gates, SLO, action space, and cold start, but permits immediate
  recovery after a fresh hard phase-latency violation.

The experiment keeps TP=1, shared P-pool and D-pool frequency states, fixed
four-pair round-robin routing, concurrency 4, and the C1 mixed workload. It
uses MEDIUM=0.10 req/s and HIGH-SUSTAINABLE=0.20 req/s. The action spaces are
P={1260, 1890, 2520} MHz and D={750, 1125, 1500} MHz. The request-level
contract is TTFT <=600 ms and TPOT <=65 ms, each with at least 95% attainment.

Each load/policy combination has three matched repeats and four 20-request
windows. The 72 windows and 1,440 requests include cold start, every probe,
every triggering violation, recovery time, and post-recovery operation. No
repeat or unfavorable request was removed. Policy order is cyclic across
repeats.

C2a is a stationary isolation test. It does not test workload-context
transitions, request-aware routing, per-endpoint asymmetric DVFS, TP changes,
predictors, or an oracle.

## Validity

The full run is valid. All 72 windows passed the existing request-ID, SSE,
token, route/pair coverage, endpoint assignment, four-GPU NVML energy,
dynamic-clock-segment, readback, resource-overlap, and thermal/HW-slowdown
hard gates. Matched traces, final HIGH restoration, and the run-level window
and clock-segment audits passed. Frequency residency is reconstructed from
actual clock segments; windows containing recovery are not represented as
single-clock windows.

The earlier preflight also passed both P and D fast-recovery gates. The full
run produced no actuator/readback failure. One P-telemetry freshness event
occurred near the end of HIGH repeat 2. The controller correctly failed closed
from P=MID to P=HIGH. This event remains in the reported frequency residency,
energy, final search state, and SLO result.

## Aggregate result

Values after `+/-` are sample standard deviations over three matched repeats.
Energy is gross four-GPU board energy and includes all exploration and
recovery.

| Load | Policy | Energy for 240 requests | J/request | TTFT p95 / p99 / attainment | TPOT p95 / p99 / attainment | Contract |
|---|---|---:|---:|---:|---:|---|
| MEDIUM | STATIC-HIGH | 629.17 kJ | 2621.56 +/- 3.07 | 552.5 / 573.7 ms / 99.17% | 63.23 / 63.24 ms / 100% | PASS |
| MEDIUM | WINDOW-BOUNDED | 569.45 kJ | 2372.72 +/- 2.64 | 599.0 / 623.9 ms / 95.00% | 63.23 / 65.38 ms / 96.25% | PASS |
| MEDIUM | FAST-SAFETY | 576.30 kJ | 2401.23 +/- 0.49 | 559.8 / 607.0 ms / 98.75% | 63.23 / 64.58 ms / 98.75% | **PASS** |
| HIGH-SUSTAINABLE | STATIC-HIGH | 404.10 kJ | 1683.75 +/- 1.87 | 571.3 / 577.8 ms / 100% | 63.20 / 63.22 ms / 100% | PASS |
| HIGH-SUSTAINABLE | WINDOW-BOUNDED | 368.47 kJ | 1535.31 +/- 0.63 | 602.2 / 643.7 ms / 94.58% | 65.12 / 65.36 ms / 93.75% | **FAIL** |
| HIGH-SUSTAINABLE | FAST-SAFETY | 373.37 kJ | 1555.70 +/- 1.49 | 587.0 / 600.5 ms / 98.75% | 63.20 / 64.59 ms / 98.75% | **PASS** |

FAST-SAFETY saves **8.404% +/- 0.098 percentage points** at MEDIUM and
**7.605% +/- 0.071 percentage points** at HIGH relative to its matched STATIC
repeats. All six FAST blocks satisfy the fixed contract. Across both loads,
FAST consumes 949.66 kJ versus 1033.27 kJ for STATIC, an 8.092% reduction.

WINDOW-BOUNDED has lower raw energy because unsafe candidates remain active
longer: it saves 9.492% at MEDIUM and 8.816% at HIGH. The HIGH value is not an
SLO-qualified saving. FAST uses 11.73 kJ more than WINDOW across both loads,
or 1.25%, in exchange for terminating unsafe exposure and passing the SLO.

## Matched-repeat result

| Load | Repeat | STATIC J/req | WINDOW J/req / contract | FAST J/req / contract | FAST saving vs STATIC |
|---|---:|---:|---:|---:|---:|
| MEDIUM | 1 | 2618.24 | 2375.36 / PASS | 2400.77 / PASS | 8.306% |
| MEDIUM | 2 | 2624.29 | 2372.72 / PASS | 2401.17 / PASS | 8.502% |
| MEDIUM | 3 | 2622.14 | 2370.08 / PASS | 2401.75 / PASS | 8.405% |
| HIGH | 1 | 1682.05 | 1535.28 / FAIL | 1553.98 / PASS | 7.614% |
| HIGH | 2 | 1683.45 | 1534.69 / FAIL | 1556.68 / PASS | 7.531% |
| HIGH | 3 | 1685.75 | 1535.96 / FAIL | 1556.43 / PASS | 7.672% |

STATIC passes all six blocks. WINDOW passes all three MEDIUM blocks at or near
the attainment boundary but fails all three HIGH blocks. FAST passes every
matched block at both loads.

## Safety exposure and adaptation latency

The triggering request remains in every SLO calculation. `Post-trigger
requests` counts requests sent after the first hard violation but before both
pool endpoints completed readback.

| Load | Pool | Policy | Unsafe probe requests | Hard violations | Post-trigger requests | First violation to readback |
|---|---|---|---:|---:|---:|---:|
| MEDIUM | D | WINDOW | 60 | 9 | 42 | 148.99 s mean |
| MEDIUM | D | FAST | 21 | 3 | 3 | 2.25 s mean, 1.93--2.47 s |
| MEDIUM | P | WINDOW | 60 | 12 | 42 | 153.93 s mean |
| MEDIUM | P | FAST | 18 | 3 | 0 | 2.05 s mean, 1.89--2.15 s |
| HIGH | D | WINDOW | 60 | 15 | 39 | 89.24 s mean |
| HIGH | D | FAST | 24 | 3 | 3 | 2.09 s mean, 2.08--2.11 s |
| HIGH | P | WINDOW | 60 | 13 | 30 | 73.98 s mean, 68.99--83.93 s |
| HIGH | P | FAST | 28 | 3 | 0 | 2.77 s mean, 2.08--3.18 s |

Across both loads, WINDOW exposes 240 requests in unsafe candidates and
records 49 hard violations. FAST exposes 91 requests and records 12 hard
violations, reductions of 62.1% and 75.5%, respectively. FAST reduces total
unsafe-candidate residence from 1965.17 s to 584.29 s. More importantly, it
reduces requests sent after the first violation from 153 to 6. P recovery
allows no additional request admission after the triggering TTFT violation;
one already scheduled/in-flight request remains after each D trigger.

FAST cannot remove the first violating request because that request supplies
the safety signal. Its measured benefit is terminating further exposure
inside the current accounting window rather than predicting the unsafe point.

## Search behavior and frequency residency

Every feedback repeat uses three downward probes and two unsafe-probe
recoveries. D=MID is rejected after a hard TPOT violation, and P=LOW is
rejected after a hard TTFT violation. Normal exploration remains
evidence-epoch driven: the P=LOW probe does not begin until the prior completed
window establishes P=MID as safe. FAST also records the one additional
telemetry-staleness fallback described below.

Five of six FAST blocks finish with the same independently established safe
point as C1b/C1c: P=MID (1890 MHz), D=HIGH (1500 MHz). HIGH repeat 2 also
recovered P=LOW to P=MID correctly, then a later required-telemetry freshness
failure caused a conservative P=MID-to-HIGH fallback. That block still passes
the SLO and remains included without adjustment.

| Load | Policy | P residency | D residency |
|---|---|---|---|
| MEDIUM | WINDOW | 25.0% HIGH / 50.0% MID / 25.0% LOW | 74.9% HIGH / 25.1% MID |
| MEDIUM | FAST | 25.0% HIGH / 68.6% MID / 6.3% LOW | 92.6% HIGH / 7.4% MID |
| HIGH | WINDOW | 24.9% HIGH / 50.1% MID / 25.0% LOW | 74.9% HIGH / 25.1% MID |
| HIGH | FAST | 26.3% HIGH / 64.4% MID / 9.2% LOW | 92.6% HIGH / 7.4% MID |

FAST preserves most P-side energy benefit while sharply shortening D=MID and
P=LOW residence. Its P-pool energy is 13.26% below STATIC at MEDIUM and 13.04%
below STATIC at HIGH. D-pool energy is about 0.8% below STATIC because fast
recovery spends little time at D=MID. This is expected safety cost, not a
failure to actuate the D frequency.

## Cumulative energy and load behavior

Mean cumulative J/request after each 20-request window is:

| Load | Requests | STATIC | WINDOW | FAST | FAST saving vs STATIC |
|---|---:|---:|---:|---:|---:|
| MEDIUM | 20 | 2616.66 | 2623.53 | 2622.02 | -0.205% |
| MEDIUM | 40 | 2620.37 | 2427.98 | 2464.64 | 5.943% |
| MEDIUM | 60 | 2621.57 | 2382.73 | 2420.09 | 7.685% |
| MEDIUM | 80 | 2621.56 | 2372.72 | 2401.23 | 8.404% |
| HIGH | 20 | 1682.67 | 1683.57 | 1683.39 | -0.043% |
| HIGH | 40 | 1682.54 | 1561.60 | 1590.16 | 5.490% |
| HIGH | 60 | 1683.33 | 1539.17 | 1563.93 | 7.093% |
| HIGH | 80 | 1683.75 | 1535.31 | 1555.70 | 7.605% |

Cold-start evidence costs about 0.2% at MEDIUM and 0.04% at HIGH, then the
energy benefit accumulates despite early recovery. Completed throughput is
effectively matched across policies: about 0.0995 req/s at MEDIUM and 0.1724
req/s at HIGH. Mean admission p95 stays below 0.24 ms, client waiting never
exceeds one, proxy inflight maxima are three and four, and every window drains
without growing client backlog. Queue divergence does not explain the policy
differences.

## Scientific answers

1. **Does event-driven safety materially reduce the C1 SLO cost?** Yes.
   WINDOW reproduces the HIGH failure, with 94.58% TTFT and 93.75% TPOT
   attainment. FAST reaches 98.75% for both metrics and passes all repeats.
   Across both loads it reduces hard exploration violations by 75.5%.
2. **Can the controller recover when an unsafe candidate is encountered?**
   Yes. All 12 P/D unsafe probes recover to their previous validated-safe
   anchors with valid physical readback. Detection-to-readback latency is
   1.89--3.18 s.
3. **Can efficiency operation continue after recovery?** Yes within this
   stationary isolation. After D recovery, normal evidence-driven exploration
   continues to the P probe, and after P recovery the controller operates at
   the learned safe point. One later stale-telemetry event correctly fails
   closed. C2a does not make a workload-transition or context-return claim.
4. **What adaptation and safety cost remains?** Each repeat pays the triggering
   TTFT and TPOT violation. FAST records two violations per repeat instead of
   seven to ten for WINDOW and uses 1.25% more total energy than WINDOW because
   it terminates unsafe low-frequency residence.
5. **What cumulative energy benefit remains?** Relative to matched STATIC,
   FAST retains 8.404% saving at MEDIUM and 7.605% at HIGH while satisfying the
   fixed SLO in all six blocks.
6. **Is the mechanism ready for the next stage?** The isolated hybrid
   mechanism is validated strongly enough to support planning D1
   request-aware routing plus per-endpoint DVFS. This result does not itself
   validate those mechanisms or dynamic context transitions.

Stage C2a is complete. No D1 experiment was started.
