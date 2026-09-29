# XPYD Stage C2 implementation preflight

## Scope

Stage C2 isolates one question: whether an event-driven safety path reduces the
SLO cost of the unchanged C1c evidence-window-bounded efficiency search. It
does not change routing, the frequency action space, tensor parallelism, the
SLO, or the downward-search algorithm.

The three matched policies are:

- `STATIC-HIGH`;
- `WINDOW-BOUNDED-FEEDBACK`, which retains C1c boundary-only recovery;
- `HYBRID-FAST-SAFETY`, which adds only in-window fail-closed recovery.

All policies use fixed four-pair round-robin routing, 2P2D TP=1, concurrency
four, shared P-pool and D-pool frequency states, and the TIGHT 600 ms TTFT / 65
ms TPOT / 95% attainment contract.

## Two control paths

Normal downshifts still require completed phase-specific evidence. P decisions
use request/input-token coverage and TTFT; D decisions use completion/output-
token coverage and TPOT. The C1c headroom thresholds and action spaces are
unchanged.

While a candidate below the validated-safe anchor is active, the fast path can
recover its pool immediately on a fresh hard TTFT or TPOT violation, clearly
growing/severe client pressure, stale or invalid required telemetry, or an
actuator/readback failure. Recovery targets the nearest higher measured-safe
anchor and otherwise HIGH. The request that triggers recovery remains in the
request and safety-event logs.

The controller receives neither context labels nor transition timestamps.
Context invalidation is derived only from completed request token fields.

## Mid-window clock semantics

C2 accounting windows may contain multiple physical clock states. Each stable
clock interval and transition interval records separate recommendation,
physical-action, and readback timestamps. Fixed-clock validation is performed
per stable segment; transition intervals are excluded from the target-match
denominator. Gross NVML energy remains a whole-window/whole-trace measurement.
Segment overlap energy is diagnostic only and is not physical request-level
attribution.

## Context-pair gate

Before dynamic execution, a small matched calibration tests only `HIGH/HIGH`
and `P=MID,D=HIGH` on a candidate prefill-heavy context. The accepted C1c mixed
context is Context A, where `P=MID,D=HIGH` was safe. Context B is accepted only
if HIGH is SLO-safe and P=MID is reproducibly unsafe under the same SLO. If that
gate fails, C2 stops rather than manufacturing a transition.

The dynamic preflight is `A,A,A,A,B`. The full trace is
`A,A,A,A,B,B,B,A,A,A`, with three cyclically ordered matched repeats. This
length follows the unchanged evidence requirements: establish the lower point
in A, expose the unannounced B transition, then permit evidence-driven
rediscovery after A returns.

## Artifacts and hard gates

The C2 harness writes per-window measurements, request records, controller
decisions, live safety events, physical clock actions, clock segments,
frequency residency, final search state, per-repeat summaries, and adaptation
summaries. It audits matched traces, all inherited request/route/token/energy
gates, dynamic clock segments, in-window fast recovery, and final HIGH
restoration.

The preflight must demonstrate a physical fast recovery before its accounting
window ends. A full run additionally requires an accepted context calibration
and accepted structural preflight artifact.

## CPU validation

The focused tests cover:

1. immediate P recovery after a hard TTFT violation;
2. immediate D recovery after a hard TPOT violation;
3. recovery to the previous validated-safe anchor;
4. evidence-gated normal downward exploration;
5. durable logging of the triggering violating request;
6. correct mid-window clock segmentation and residency accounting;
7. distinct recommendation, action, and readback timestamps;
8. resumed exploration after recovery and a later relaxed context;
9. no mid-window mutation by the window-bounded comparator;
10. fail-closed recovery on invalid required safety telemetry;
11. immediate live TTFT and client-pressure event emission.

The complete CPU suite passed 279 tests before any C2 GPU submission.

## Claim boundary

C2 is a shared-pool DVFS safety-timing isolation. It does not validate
request-aware routing, endpoint-asymmetric DVFS, TP-diverse endpoints,
predictors, offline oracle input, or exact physical request-energy attribution.
