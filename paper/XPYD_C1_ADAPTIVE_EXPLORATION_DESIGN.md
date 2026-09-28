# XPYD C1 adaptive safe exploration

Status: C1c correctness preflight implementation; no GPU job was run for this
change. C1c is stationary and evidence-window-bounded.

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
  builds independent P and D controllers for both online policies, converts
  every completed C1 window (including the final window) to phase-specific
  evidence, actuates the shared pool target, and appends the structured log.
* `paper/configs/xpyd_stagec1_stationary_neptune_europa.json` freezes the
  three-policy C1c matrix and explicit search parameters.
* `paper/scripts/xpyd/__init__.py` exports the new CPU-testable primitives.
* `tests/test_xpyd_adaptive_exploration.py` covers the deterministic state
  machine.  Existing Stage C1 tests remain unchanged and continue to exercise
  the legacy coordinator and headroom harness.

The legacy `FeedbackScheduler` remains available to older phases and dry-runs,
but it is not either C1c feedback policy. ONE-STEP and ADAPTIVE use the same
new evidence/safety state machine; ONE-STEP only restricts every downward
probe to one frequency level.

## SLO regimes and evidence

The primary C1c configuration uses TIGHT:

| Regime | TTFT | TPOT | Required attainment |
|---|---:|---:|---:|
| TIGHT | 600 ms | 65 ms | 95% |
| MODERATE | 625 ms | 66 ms | 95% |
| LOOSE | 650 ms | 70 ms | 95% |

MODERATE and LOOSE remain analysis-only SLO-sensitivity contracts and are not
used by the primary C1c run. The controller receives TIGHT from configuration. It does not know
the C1b result that P/MID was safe or that D/MID failed at a particular
threshold.

Each P or D pool has a separate evidence record.  P evidence requires recent
request-level TTFT observations and input-token coverage; D evidence requires
request-level TPOT observations and completed-output-token coverage.  Both
also carry observation duration, valid request count, pass/violation counts,
backlog stability, telemetry freshness, health, request-accounting validity,
frequency readback, context signature, and measured gross pool energy.

The frozen C1c settings in the JSON (`min_valid_requests=8`, token
coverage of 1024, one-second minimum observation, 120-second maximum evidence
timeout, and the two headroom thresholds) are unchanged by the correctness
patch. Changing a threshold is an explicit experiment-config revision, not a
hidden controller constant.

## State machine

Each pool maintains `PoolSearchState` independently.  It includes:

* `current_frequency`, `known_safe_frequency`, `candidate_frequency`, and
  `unsafe_bound`;
* `current_index`, `last_validated_safe_index`, and `unsafe_index` in the
  ordered supported frequency list;
* `search_status/search_phase` (`COLD`, `EXPAND`, `REFINE`, `HOLD`, or
  `RECOVER`), `evidence_since_action`, and `last_action_reason/timestamp`;
* per-frequency measured-safe energy means,
  `best_measured_safe_frequency/energy`, `context_signature`, `confidence`,
  convergence counts, and `transition_reason`.

The initial state is HIGH/COLD.  A sufficient, fresh, healthy, request-level
SLO-safe observation can select a lower candidate. Headroom below
`headroom_for_one_step` holds, intermediate headroom permits one level, and
headroom at/above `headroom_for_large_jump` permits the configured coarse jump.
The candidate is *not* marked safe when selected.  Only the next valid
observation at that readback frequency can produce `ACCEPT_SAFE` and update the
known-safe anchor.

If a lower candidate violates the relevant 95% attainment gate, has a latency
statistic at/above its SLO, or fails an audit, the controller immediately
returns to the previous safe anchor (HIGH when that is the anchor), records an
unsafe bound, and does not use the candidate's energy.  If safe and unsafe
indices bracket an untested level, the next exploration is a midpoint
`REFINE`. An adjacent unsafe bound resolves the boundary and is never probed
again unless explicit context invalidation clears it. The final window is
classified without starting a new, unmeasured downward probe.

## Window-bound safety path

At each completed-window evidence epoch, fresh risk is handled before
efficiency search. Any of the following cancels
downward exploration: SLO attainment/latency risk, growing client or server
backlog, queue/pressure evidence, stale telemetry, invalid request/measurement
audit, unhealthy endpoint, or requested/readback mismatch.  The action is
`RECOVER_HIGH` (or HOLD when already HIGH), with trigger and reason recorded.
An evidence timeout with no SAFE proof also recovers a non-HIGH candidate.
Outside a probe/action, a stable point is retained. Missing endpoint pressure
remains UNKNOWN and cannot validate a candidate; no `running==0` sample is
used as idle evidence. C1c does not implement or claim event-driven recovery
between windows; that architecture change belongs to C2.

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
action timestamp, and readback timestamp. Recommendation time never updates
the last actual-action time. `clock_actions.csv` remains the
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

The CPU/static preflight is complete. The evidence minimum/timeout, headroom
thresholds, context invalidation policy, TIGHT SLO, three-level action space,
three repetitions, and four-window block structure are frozen in the C1c
configuration. The next authorized step is the controlled stationary GPU
matrix, retaining the adaptive decision log alongside the existing energy,
request, clock, and SLO audits. This readiness statement does not claim that
the later C2 event-driven safety architecture is implemented.
