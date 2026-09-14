# XPYD C1 adaptive safe exploration

Status: implementation and CPU validation complete; no GPU job was run for
this change.  The implementation is ready for a controlled full-C1 run only
after the implementation-default exploration values are reviewed and frozen.

## Scope and claim boundary

Stage C1 remains an isolation experiment: 2P2D, TP=1, four explicit
prefill/decode pairs, fixed round-robin routing, shared P-pool and D-pool
frequency states, stationary mixed workload, and every policy block starting
at HIGH.  This change adds no request-aware routing, endpoint activation,
predictor, bandit, RL, oracle, or request-level physical-energy attribution.

The routing path and serving semantics are unchanged.  Energy is used only as
measured gross phase/pool evidence after a point has passed its safety gates;
an unsafe point can never justify an energy choice.

## Changed implementation

* `paper/scripts/xpyd/adaptive_exploration.py` adds the model-free
  `AdaptivePoolController`, `PoolEvidence`, `PoolSearchState`, and
  `AdaptiveDecision` types.
* `paper/scripts/xpyd/stagec1_stationary.py` accepts the selected SLO regime,
  builds independent P and D controllers when
  `controller_mode=adaptive_safe_search`, converts each completed C1 window to
  phase-specific evidence, actuates the shared pool target, and appends the
  structured decision log.
* `paper/configs/xpyd_stagec1_stationary_neptune_europa.json` selects the new
  mode and records explicit implementation defaults.
* `paper/scripts/xpyd/__init__.py` exports the new CPU-testable primitives.
* `tests/test_xpyd_adaptive_exploration.py` covers the deterministic state
  machine.  Existing Stage C1 tests remain unchanged and continue to exercise
  the legacy coordinator and headroom harness.

The legacy `FeedbackScheduler` and its fixed-fraction behavior remain
available for comparator phases and old dry-runs.  In adaptive C1, the
`step_down_fraction` value is no longer a decision rule; it is retained only
for the legacy mode and compatibility with existing configuration files.

## SLO regimes and evidence

The C1 configuration chooses one of the already specified regimes:

| Regime | TTFT | TPOT | Required attainment |
|---|---:|---:|---:|
| TIGHT | 600 ms | 65 ms | 95% |
| MODERATE | 625 ms | 66 ms | 95% |
| LOOSE | 650 ms | 70 ms | 95% |

The controller receives these values from configuration.  It does not know
the C1b result that P/MID was safe or that D/MID failed at a particular
threshold.

Each P or D pool has a separate evidence record.  P evidence requires recent
request-level TTFT observations and input-token coverage; D evidence requires
request-level TPOT observations and completed-output-token coverage.  Both
also carry observation duration, valid request count, pass/violation counts,
backlog stability, telemetry freshness, health, request-accounting validity,
frequency readback, context signature, and measured gross pool energy.

The implementation defaults in the C1 JSON (`min_valid_requests=8`, token
coverage of 1024, one-second minimum observation, 120-second maximum evidence
timeout, and the two headroom thresholds) are safe dry-run defaults, not
scientific calibration.  They must be reviewed/frozen before the physical
experiment.  Changing a threshold is a configuration change, not a hidden
controller constant.

## State machine

Each pool maintains `PoolSearchState` independently.  It includes:

* `current_frequency`, `known_safe_frequency`, `candidate_frequency`, and
  `unsafe_bound`;
* `current_index`, `last_validated_safe_index`, and `unsafe_index` in the
  ordered supported frequency list;
* `search_status/search_phase` (`COLD`, `EXPAND`, `REFINE`, `HOLD`, or
  `RECOVER`), `evidence_since_action`, and `last_action_reason/timestamp`;
* `best_measured_safe_energy_j`, `context_signature`, `confidence`, and
  `transition_reason`.

The initial state is HIGH/COLD.  A sufficient, fresh, healthy, request-level
SLO-safe observation can select a lower candidate.  A large headroom value
selects a configurable larger index jump; otherwise one lower level is used.
The candidate is *not* marked safe when selected.  Only the next valid
observation at that readback frequency can produce `ACCEPT_SAFE` and update the
known-safe anchor.

If a lower candidate violates the relevant 95% attainment gate, has a latency
statistic at/above its SLO, or fails an audit, the controller immediately
returns to the previous safe anchor (HIGH when that is the anchor), records an
unsafe bound, and does not use the candidate's energy.  If safe and unsafe
indices bracket an untested level, the next exploration is a midpoint
`REFINE`; the search does not restart or exhaustively sweep every point.

## Fast safety path

Fresh risk is handled before efficiency search.  Any of the following cancels
downward exploration: SLO attainment/latency risk, growing client or server
backlog, queue/pressure evidence, stale telemetry, invalid request/measurement
audit, unhealthy endpoint, or requested/readback mismatch.  The action is
`RECOVER_HIGH` (or HOLD when already HIGH), with trigger and reason recorded.
An evidence timeout with no SAFE proof also recovers a non-HIGH candidate.
Outside a probe/action, a stable point is retained; no `running==0` sample is
used as idle evidence.

Context signatures are carried with evidence.  When context invalidation is
enabled and a previously learned signature changes, the pool conservatively
recovers and clears its candidate rather than reusing stale safety evidence.

## Decision logging

`controller_decisions.csv` keeps the existing C1 fields and adds the structured
adaptive fields from `AdaptiveDecision`: pool/phase/SLO regime, current,
candidate, known-safe and unsafe frequencies, search phase, decision/action,
trigger/reason, evidence duration and valid count, P/D token coverage,
phase-latency statistic, attainment, backlog/pressure, freshness, headroom,
jump size, requested/readback frequency and validity, decision timestamp,
action timestamp, and readback timestamp.  `clock_actions.csv` remains the
physical command/readback audit, so decision time and actuator time can be
compared without conflating them.

## CPU validation

`tests/test_xpyd_adaptive_exploration.py` is deterministic and CPU-only:

| Case | Result |
|---|---|
| A: HIGH, strong headroom, lower candidate, SAFE retain | PASS |
| B: candidate SLO violation, immediate rollback | PASS |
| C: insufficient evidence is not SAFE | PASS |
| D: stable latency but growing backlog is risk | PASS |
| E: stale telemetry is conservative | PASS |
| F: safe/unsafe bracket refines midpoint | PASS |
| G: P and D search state is independent | PASS |
| H: readback mismatch is never safe | PASS |

The existing Stage C1 semantic tests also pass.  No full C1 workload and no
GPU/NVML experiment was launched in this task.

## Readiness

The code is ready for a controlled full-C1 **implementation review**, not an
automatic claim of experimental readiness.  Before submitting that GPU job,
freeze the evidence minimum/timeout, headroom jump thresholds, context
invalidation policy, and any experiment-specific pressure interpretation in
the C1 configuration.  Then run the controlled stationary matrix and retain
the adaptive decision log alongside the existing energy, request, clock, and
SLO audits.
