# XPYD Stage C2 dynamic-control result

## Verdict

**STOPPED AT THE CONTEXT-PAIR CALIBRATION GATE.**

The Stage C2 controller implementation and CPU validation are complete, but
the required dynamic workload pair has not been established. No C2 dynamic
preflight or full comparison was launched.

## Calibration execution

- Slurm job: `269855`
- Hardware: Neptune P0/P1 L40S and Europa D0/D1 L4
- Result: `COMPLETED`, exit code 0, elapsed 19:47
- Candidate Context B: 100% long input with the existing 70/30 short/long
  output mix at 0.20 requests/s
- Fixed points: `HIGH/HIGH` and `P=MID,D=HIGH`
- Repeats: two matched traces, 20 requests per point
- SLO: TTFT <= 600 ms and TPOT <= 65 ms, each with >=95% attainment

All request-ID, SSE, token, route, pair-coverage, NVML-energy, fixed-clock,
thermal/HW-slowdown, and resource-overlap hard gates passed. Each route carried
five requests per measurement window. Requested clocks had 100% active-sample
match in every window.

## Measured result

| Repeat | Configuration | TTFT mean | TTFT p95 | TTFT p99 | TTFT attainment | TPOT attainment | J/request |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | HIGH/HIGH | 594.5 ms | 826.1 ms | 942.5 ms | 75% | 100% | 1725.69 |
| 1 | P=MID,D=HIGH | 536.1 ms | 599.3 ms | 601.3 ms | 95% | 100% | 1573.30 |
| 2 | HIGH/HIGH | 528.1 ms | 584.5 ms | 585.1 ms | 100% | 100% | 1743.54 |
| 2 | P=MID,D=HIGH | 532.8 ms | 590.6 ms | 596.8 ms | 100% | 100% | 1572.10 |

The required gate was:

1. HIGH/HIGH must be reproducibly SLO-safe; and
2. P=MID,D=HIGH must be reproducibly unsafe on the same matched context.

Neither condition held. HIGH/HIGH failed in repeat 1, while P=MID,D=HIGH met
the 95% contract in both repeats. Therefore `context_pair_established=false`.

## Diagnostic interpretation

This is not a frequency-lock, routing, request-correlation, energy, or
telemetry failure. The first HIGH/HIGH window contained five TTFT violations;
four were the first four requests and ranged from approximately 765 to 972 ms.
Every later window was much tighter. HIGH/HIGH was always executed before
P=MID,D=HIGH, so the result is consistent with a first-window/server warm-state
effect, but the calibration does not isolate that effect and it must not be
discarded post hoc.

More importantly, the warm measurements show that this candidate context does
not reliably separate HIGH from P=MID under the fixed 600 ms SLO. P=MID was
safe, not unsafe. It therefore cannot support the intended C2 causal test.

## Answers to the Stage C2 questions

The dynamic experiment was correctly gated before these questions could be
answered:

1. Event-driven safety versus the C1 full-window SLO cost: **not measured**.
2. Recovery after increasing workload pressure: **not measured**.
3. Return to an energy-saving state: **not measured**.
4. Adaptation latency: **not measured**.
5. Cumulative energy benefit after adaptation: **not measured**.
6. Readiness for D1: **not established**.

## Next decision

C2 should not proceed with the present Context B. A future task may run one
new, deliberately small and balanced context-search calibration, testing only
HIGH/HIGH and P=MID,D=HIGH on a few predeclared pressure patterns. Such a task
must address first-window order/warm-state effects and stop unless it finds a
reproducible HIGH-safe/P-MID-unsafe separation. It must not retune the SLO or
controller to manufacture that separation.

No preflight, full C2 experiment, or D1 experiment was submitted.
