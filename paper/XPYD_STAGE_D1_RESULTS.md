# XPYD Stage D1: per-endpoint DVFS and runtime-feedback pair routing

## Status

Implementation and CPU preflight are in progress. No D1 scientific result is
claimed until the fixed-asymmetry GPU preflight passes and the matched full
matrix completes.

Stage D1 asks whether request-level full-pair routing can exploit independently
controlled endpoint frequency/load asymmetry while preserving the fixed TIGHT
SLO, using runtime feedback only.

## Frozen contract

- P0/P1: Neptune L40S, TP=1.
- D0/D1: Europa L4, TP=1.
- Four compatible P-to-D pairs, concurrency 4.
- Workload: 35% SS / 15% SL / 35% LS / 15% LL.
- Loads: MEDIUM 0.10 req/s and HIGH-SUSTAINABLE 0.20 req/s.
- SLO: TTFT <= 600 ms and TPOT <= 65 ms, each with at least 95% attainment.
- P frequencies: 1260 / 1890 / 2520 MHz.
- D frequencies: 750 / 1125 / 1500 MHz.
- Gross NVML energy accounting and C2a event-driven fast safety are retained.

The full policies are:

1. `STATIC-HIGH-RR`
2. `SHARED-FAST-RR`
3. `PER-ENDPOINT-FAST-RR`
4. `JOINT-ROUTING-PER-ENDPOINT`

The policies use matched traces, three repeats, five 20-request evidence
windows per load/policy/repeat, and cyclic execution order. Every full block
starts at HIGH.

## Routing semantics

The client supplies the logical request ID, input-token count, requested output
budget, and SS/SL/LS/LL class in admission-time headers. The proxy validates
this context before selecting a pair. Realized output length is not routing
input.

The router evaluates complete compatible P-to-D pairs in this fixed hierarchy:

1. compatibility and endpoint health;
2. fresh, known SLO safety;
3. live pair pressure/backlog;
4. measured window-amortized endpoint energy among safe choices;
5. controlled exploration/tie-breaking.

There is no weighted score. Unknown or stale candidates are not treated as
known-safe. The energy evidence is gross endpoint board energy divided by the
number of requests assigned to that endpoint in the prior accounting window;
it is not physical per-request energy attribution.

Cold start uses exact four-pair RR to obtain runtime observations. Rising live
pressure has priority over energy preference, causing spillover to another
safe pair rather than unconditional collapse onto one low-energy endpoint.

## Per-endpoint control and safety

P0, P1, D0, and D1 each own an independent accepted C1/C2 safe-search state.
Normal downward exploration remains evidence-window driven. A same-phase guard
permits at most one unvalidated endpoint probe at a time and preserves another
validated-safe fallback endpoint.

A hard TTFT violation recovers only the request's selected P endpoint. A hard
TPOT violation recovers only its selected D endpoint. Recovery targets the
nearest prior validated-safe anchor, falling back to HIGH only if no lower safe
anchor exists. Clock transitions retain C2a decision/action/readback timestamps
and segment-based frequency-residency accounting.

## Fixed-asymmetry preflight

The preflight deliberately sets P0=MID and P1=HIGH, with D0=D1=HIGH. It must
verify request-context delivery, use of all four compatible pairs, endpoint
safety/pressure response, runtime energy preference, pressure spillover,
per-endpoint accounting, readback validity, and all existing hard gates.

Preflight result: **PENDING**.

## Matched full result

Full D1 result: **NOT RUN**. The full matrix is gated on an accepted preflight.

## Claim boundary

D1 is a stationary isolation experiment. It does not add TP diversity,
activation/scaling, Canary profiling, predictors, learned models, RL/bandits,
or dynamic workload transitions. It does not claim exact physical energy
attribution to individual requests.
