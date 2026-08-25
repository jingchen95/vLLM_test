# XpYd Phase 4B controlled policy evaluation

Phase 4B is a scientific comparison of the unchanged Phase 3D feedback-only
controller against simple baselines and the frozen Phase 4A empirical oracle.
It does not train predictors, load legacy XGBoost models, or feed oracle values
to runtime policies.

## Fixed platform

- P0/P1: Neptune L40S, TP1
- D0/D1: IO L4, TP1
- persistent vLLM 0.15.1 endpoints and the validated P2P connector path
- the same four compatible routes and LOW/MID/HIGH states used by Phase 4A
- TTFT SLO 1000 ms and TPOT SLO 80 ms

Uranus is not used.

## Stationary policies

The physical comparison measures:

1. `STATIC`: round-robin across all compatible pairs, every endpoint HIGH;
2. `FEEDBACK_ROUTING_ONLY`: measured feedback routing, every endpoint HIGH;
3. `FEEDBACK_DVFS_ONLY`: fixed P0->D0 routing with measured per-endpoint DVFS;
4. `FULL_FEEDBACK`: measured feedback routing and measured per-endpoint DVFS.

The `EMPIRICAL_ORACLE` is not rerun. Its per-workload energy and near-optimal
set are read unchanged from the accepted Phase 4A summary. The file SHA-256 is
recorded in every Phase 4B run.

Policy/workload blocks use a fixed seeded random order. Feedback state is
isolated per policy/workload block. Each feedback block receives one unmeasured
observation warmup, followed by five measured repeats with two requests per
repeat. Warmup observations are runtime measurements, not oracle information.

Missing or stale telemetry is never interpreted as zero pressure. Routing
policies fall back to validated routes at safe HIGH frequency, and DVFS-only
falls back to fixed P0->D0 at safe HIGH. Requested and observed frequencies,
fallbacks, successful physical actions, routing changes, and freshness are
recorded separately.

## Dynamic stage

After the full stationary and Phase 4B.1 smoke audits pass, the formal dynamic
stage compares exactly `STATIC`, `PASSIVE_FULL`, and `ACTIVE_FULL` on two small
traces:

- `small_light -> prefill_heavy -> small_light`;
- `decode_heavy -> both_heavy -> decode_heavy`.

Each policy/trace block has three independent repeats, and each state lasts
three measured windows. This provides one blind transition
window, one feedback-reaction opportunity after observing the new state, and
one settling check. At the transition itself, the scheduler receives only the
previously observed workload context. The new context is recorded for decisions
only after its first window completes. The context label is diagnostic—the
unchanged Phase 3D scheduler still uses measured endpoint telemetry only.
`ACTIVE_FULL` uses the validated context-route four-GPU window cost and safe
probe cadence; probe windows freeze DVFS. The optional burst trace is omitted
to keep the formal matrix controlled and practical.

Dynamic reporting distinguishes a measured reaction from a stable regime that
needed no action. It records reaction latency, descriptive settling time,
transition and cumulative energy, SLO violations, actions, route/frequency
state, oscillations/reversals, fallbacks, and partial telemetry staleness. A
piecewise Phase 4A oracle is post-hoc context only and never a controller input.

## Claim and stage boundary

The stationary stage must pass before dynamic traces are run. A smoke run uses
one repeat but still covers all four policies and workloads. A successful
smoke validates plumbing only; it is not scientific evidence. A successful
five-repeat stationary run produces the evidence needed to authorize the
separate dynamic stage.

This deliberate gate prevents a stationary failure from being obscured by a
later dynamic trace. The final A/B/C decision about feedback sufficiency remains
pending until dynamic traces are complete and stationary/dynamic evidence is
combined.

## Outputs

Stationary outputs appear under `results/phase4b_evaluation/<run-id>/`; formal
dynamic outputs use `results/phase4b_dynamic_evaluation/<run-id>/`:

- `phase4b_stationary_results.csv`: one row per measured repeat;
- `phase4b_dynamic_results.csv`: per-window energy, latency, cumulative energy,
  routes, clocks, and explicit ACTIVE decision modes;
- `phase4b_dynamic_adaptation.csv`: one row per measured workload transition
  and repeat;
- `phase4b_dynamic_adaptation_aggregate.csv`: reaction/settling variance across
  repeats;
- `phase4b_route_cost_observations.csv`: context-route four-GPU window costs;
- `phase4b_policy_aggregate.csv`: mean/std/CI energy and latency comparison;
- `phase4b_policy_summary.csv`: mean, sample standard deviation, CI, latency
  tails, SLO violations, route distribution, frequency residency, and actions;
- `phase4b_oracle_gap.csv`: normalized energy comparison and <=5%/5-10%/>10%
  classification;
- `phase4b_control_actions.csv`: routing/DVFS decisions and fallbacks;
- `phase4b_audit.json`, `phase4b_summary.json`, and `phase4b_summary.md`;
- `raw/`: physical windows, request rows, telemetry, actuator actions,
  capabilities, the randomized plan, and frozen-oracle provenance.

The primary oracle gap uses J/request:

```text
(policy J/request - Phase 4A oracle J/request) / Phase 4A oracle J/request
```

Gross window energy and J/output-token remain available beside it.

## Launcher

Set:

```bash
XPYD_PHASE4B_CONFIG=paper/configs/xpyd_phase4b_evaluation_neptune_io.json
XPYD_PHASE4B_ORACLE_SUMMARY=results/phase4a_empirical_oracle/phase4a_oracle_r1_20260824T080026Z/phase4a_summary.json
```

Optionally set `XPYD_PHASE4B_RUN_ID`. Set `XPYD_PHASE4B_SMOKE=1` for the
one-repeat physical smoke. The regular run uses all five repeats.

For the gated dynamic stage, additionally set:

```bash
XPYD_PHASE4B_STAGE=dynamic
XPYD_PHASE4B_ACCEPTED_STATIONARY_AUDIT=results/phase4b_evaluation/<accepted-run>/phase4b_audit.json
XPYD_PHASE4B_ACCEPTED_ACTIVE_SMOKE_AUDIT=results/phase4b1_safe_probing/<accepted-smoke>/phase4b1_audit.json
```

The default stage is `stationary`.

Phase 4B.1 safe probing remains a separate smoke harness. Its accepted audit is
a prerequisite for formal dynamic `ACTIVE_FULL`. Dynamic route costs remain
context-specific four-GPU gross window energy per logical request and are never
treated as physical request-level attribution.
