# XpYd Phase 4B.1: minimal context-aware safe routing probes

## Claim boundary

Phase 4B.1 removes passive routing information starvation without a predictor,
bandit, RL policy, or offline route lookup. It does not change the accepted
feedback-only DVFS decision thresholds or actuation path.

For route `P_i -> D_j` and workload context `c`, ACTIVE_FULL observes:

```text
cost(P_i -> D_j, c) =
    gross window energy of P0 + P1 + D0 + D1
    / logical requests completed in that window
```

This is a window-amortized system cost. It is **not** physical request-level
energy attribution. The cost store is keyed by both context and exact route;
observations are never shared across workload contexts.

PASSIVE_FULL remains the accepted Phase 4B endpoint-EWMA routing policy and is
used only as the smoke comparator. ACTIVE_FULL ranks recent route-level system
costs. The Phase 4A oracle is used only by inherited post-hoc measurement
reporting and is never a routing input.

## Safe probing rule

ACTIVE_FULL normally exploits the lowest recent measured cost among routes
that pass the existing compatibility, health, supported-frequency, queue/KV,
latency, and freshness gates. It may probe one unseen or stale route only when:

- the route is explicitly compatible and currently eligible;
- both endpoints are healthy and active;
- role latency telemetry is within the existing freshness limit;
- TTFT and TPOT are below the configured probe-headroom fraction of their SLO;
- queue and KV pressure are within the existing low-pressure thresholds;
- the probe interval has elapsed.

If these conditions fail, exploration is suppressed. ACTIVE_FULL exploits a
recent safe route if one exists; otherwise it uses the existing explicit
conservative fallback. A route with stale cost but stale endpoint safety
telemetry is not probed.

Unseen-route ordering balances endpoint observations. In the validated 2P2D
topology this avoids starving P1 or D1 while covering the four compatible
pairs. Staleness is bounded by both elapsed time and completed control windows.

## DVFS isolation

Every probe decision sets `dvfs_frozen_for_probe=true`. The requested
frequency map must be identical before and after that control decision, and no
DVFS action may be recorded. Once the controller returns to an exploitation
window, it calls the unchanged Phase 3D/4B feedback-DVFS path, including the
existing dwell/cooldown and successful-readback semantics.

Freezing DVFS prevents a simultaneous route and frequency action within the
probe window. The measured route cost still reflects the already active
hardware frequency state; it is not a frequency-independent counterfactual.

## Small physical smoke

The only authorized physical run in this stage uses `small_light`, two logical
requests per window, and seven windows for each policy:

- PASSIVE_FULL: reproduce the incumbent passive behavior;
- ACTIVE_FULL windows 1--4: safely sample all four compatible routes;
- later ACTIVE_FULL windows: exploit recent minimum cost, resume unchanged
  DVFS, and refresh a route whose cost crosses the window-freshness bound.

The accepted all-route warmup gives unseen routes a bounded initial sampling
opportunity. After that coverage, recurring stale-route probes are separated by
at least one exploitation window. Route cost expires before the endpoint
latency safety telemetry used to authorize its refresh; no probe bypasses the
freshness gate.

The smoke is deliberately not the formal dynamic Phase 4B evaluation. Context
isolation and negative pressure behavior are additionally covered by CPU-only
tests. The physical run does not induce artificial GPU pressure; its audit
records a no-mutation pressure-gate dry run and labels that evidence precisely.

Hard gates require:

- accepted non-smoke Phase 4B stationary prerequisite;
- all warmups and physical windows valid;
- exact logical IDs, tokens, endpoint assignments, and compatibility;
- valid four-endpoint NVML windows and clock readbacks;
- passive lock-in reproduced and all four routes observed by ACTIVE_FULL;
- an unseen/stale route cost refreshed;
- exact four-GPU system-cost arithmetic and context isolation;
- every physical probe passing strict headroom/pressure/freshness gates;
- DVFS frozen in probes and resumed in exploitation windows;
- zero TTFT/TPOT SLO violations, fallbacks, thermal/HW slowdown, and unresolved
  errors;
- safe HIGH restoration.

Only a fully passing smoke may report `READY_FOR_DYNAMIC_EVALUATION`.

## Launcher

Use the validated Neptune L40S plus io L4 allocation and export:

```bash
XPYD_PHASE4B1_CONFIG=paper/configs/xpyd_phase4b_evaluation_neptune_io.json
XPYD_PHASE4B1_RUN_ID=<unique-smoke-run-id>
XPYD_PHASE4B_ORACLE_SUMMARY=results/phase4a_empirical_oracle/phase4a_oracle_r1_20260824T080026Z/phase4a_summary.json
XPYD_PHASE4B_ACCEPTED_STATIONARY_AUDIT=results/phase4b_evaluation/phase4b_stationary_r1_20260824T234000Z/phase4b_audit.json
```

Do not set `XPYD_PHASE4B_CONFIG` in the same job. The launcher rejects multiple
XpYd phase configs.
