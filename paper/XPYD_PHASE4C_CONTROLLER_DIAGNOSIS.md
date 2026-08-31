# Phase 4C passive-controller diagnosis

This is an analysis-only follow-up to accepted job `255354`. It does not alter
the controller and does not use new GPU measurements.

## Main finding

The suspected recovery asymmetry is present, but the immediate cause is more
specific than cadence alone: the Phase 4C outcome gate uses TTFT/TPOT SLOs of
550/65 ms, while the intentionally unchanged controller still uses its Phase
3D 1000/80 ms SLOs. Consequently, the P-side controller downshifts at EWMA
TTFT <= 500 ms and does not upshift until EWMA TTFT > 800 ms. The 550--800 ms
range can violate the Phase 4C TTFT objective while the controller returns
`HOLD: moderate measured operating region`.

The controller requires no consecutive-window count: one threshold crossing
is sufficient. Normal `STEP_UP` and `STEP_DOWN` actions both move one frequency
level. Only severe pressure (`queue >= 4` or KV usage >= 0.97) requests HIGH in
one action. The minimum dwell is 1 s, far below the observed 60--142 s physical
decision cadence, and was not the recovery limiter.

All selected-P decision-time queue observations in the paired
STATIC-pass/PASSIVE-fail cases were zero. With connector concurrency fixed at
one, excess offered work waits before submission at the client semaphore and
is not represented by endpoint queue depth. The queue-pressure branch therefore
cannot detect that source of demand in this experiment.

## Decode-heavy to prefill-heavy transition

For all three repeats, decode-heavy feedback took the selected P0 to LOW
(1260 MHz). On return to prefill-heavy, P0 remained LOW for all three windows:

| Repeat | Decision-visible P0 EWMA TTFT (ms) | Resulting mean TTFT (ms) | P0 actions | Request violations |
|---:|---|---|---|---:|
| 1 | 285.0, 459.1, 521.0 | 633.3, 582.9, 643.9 | HOLD, HOLD, HOLD | 8/9 |
| 2 | 321.7, 461.9, 525.5 | 602.1, 589.0, 607.3 | HOLD, HOLD, HOLD | 6/9 |
| 3 | 325.3, 473.1, 550.9 | 621.0, 628.6, 607.6 | HOLD, HOLD, HOLD | 9/9 |

Window 1 is deliberately blind and uses the decode-heavy context. In window 2,
the EWMA is still below the controller's 500 ms downshift boundary, so LOW is
reported as having large slack. In window 3, the EWMA is above 500 ms but below
the 800 ms upshift boundary, so the controller holds LOW. Even after the third
outcome, the reconstructed EWMAs would remain about 566--583 ms, still below
800 ms. No `STEP_UP` occurred anywhere in all 81 PASSIVE Phase 4C windows;
there were 32 `STEP_DOWN` recommendations and 292 `HOLD` recommendations.

Across the nine post-transition prefill windows, PASSIVE failed the mean-window
SLO in all nine; STATIC passed eight and failed one. At request level for these
27 paired positions:

- STATIC pass / PASSIVE fail: 17;
- STATIC fail / PASSIVE pass: 3;
- both fail: 6;
- both pass: 1.

Thus this transition has a net 14 additional PASSIVE failures. All 17
STATIC-pass/PASSIVE-fail candidates ran on selected P0 at LOW with observed
decision queue zero. This is strong controller-associated evidence, but it is
not literal causal attribution because the two policies were measured in
separate physical runs.

## Full paired breakdown

Across all 243 paired request positions:

- both pass: 148;
- STATIC pass / PASSIVE fail: 46;
- STATIC fail / PASSIVE pass: 16;
- both fail: 33.

The net excess is 30 PASSIVE failures, matching 79 minus 49 total violations.
Of the 46 STATIC-pass/PASSIVE-fail candidates, 36 occurred at selected-P LOW,
7 at MID, and 3 at HIGH; all 46 had selected-P queue zero. Therefore 46 is a
candidate count, not a defensible claim that every one was caused solely by
DVFS. The post-decode transition subset above is cleaner evidence.

Using the experiment's mean-window SLO definition, 81 paired windows divide
into 53 both-pass, 21 STATIC-pass/PASSIVE-fail, 1 STATIC-fail/PASSIVE-pass, and
6 both-fail windows. With the stricter definition that every request in a
window must pass, the corresponding counts are 45, 4, 1, and 31.

## Controller parameters and cadence

- P0/P1 LOW/MID/HIGH: 1260/1890/2520 MHz.
- D0/D1 LOW/MID/HIGH: 750/1125/1500 MHz.
- EWMA alpha: 0.5.
- Controller TTFT/TPOT SLO: 1000/80 ms.
- Phase 4C reporting TTFT/TPOT SLO: 550/65 ms.
- P downshift boundary: TTFT EWMA <= 500 ms, queue <= 0, KV <= 0.5.
- P upshift boundary: TTFT EWMA > 800 ms, queue > 1, or KV > 0.9.
- Severe one-shot HIGH: queue >= 4 or KV >= 0.97.
- Telemetry maximum age: 120 s; minimum dwell: 1 s.
- Consecutive windows required: none.
- Each Phase 4C window contains 3 requests at max concurrency 1.
- Mean active measurement duration ranged from 22.1--23.7 s for light/prefill
  shapes and 86.9--91.1 s for decode/both-heavy shapes.
- Measured decision-to-decision cadence averaged 63.5--65.5 s for light/prefill
  and 131.7--133.6 s for decode/both-heavy windows.

## Compact artifacts

- `phase4c_window_summary.csv`: all 162 windows, including timestamps,
  workloads, SLO metrics, route, requested/observed P frequencies, and
  decision-visible P queues where available.
- `phase4c_decode_to_prefill_controls.csv`: the 18 PASSIVE windows around the
  three decode-heavy to prefill-heavy transitions, with reconstructed
  decision-visible EWMA/queue/KV, recommendations, reasons, frequencies, and
  post-decision outcomes.
- `phase4c_paired_violation_breakdown.csv`: trace/repeat/segment cross-tabs.
- `phase4c_paired_violation_candidates.csv`: the 46 individual paired
  STATIC-pass/PASSIVE-fail candidates with timestamps, frequency, route, queue,
  and decision EWMA.
- `phase4c_controller_diagnosis.json`: compact audit totals.

The decision state reconstruction uses only each isolated PASSIVE block's
warmup and completed prior windows. Current-window measurements appear only as
outcomes; no future observation is leaked into a decision row.

## Mechanism implication

The next mechanism should first align the controller's safety objective with
the active experiment SLO and provide an asymmetric emergency recovery path.
The data support fast LOW/MID-to-HIGH recovery once a correctly aligned TTFT
risk signal fires. They do not support relying on the current endpoint queue
signal for client-side offered-load backlog. A new GPU experiment is not needed
to establish these two diagnosis points.

The isolated correction and exact-rerun protocol are documented in
`XPYD_PHASE4C1_RECOVERY.md`; the original Phase 4C configuration remains
available for legacy reproduction.
