# XpYd Phase 3D control validation

Phase 3D is a two-stage physical validation. It does not train a model, use
the legacy predictors for decisions, search a large configuration space, or
claim energy optimality.

## Stage A: per-endpoint actuator

The harness discovers real supported graphics clocks on each physical GPU,
then selects three supported states at or below the previously validated
sustainable HIGH point. It validates independent downward and upward
transitions for P0, P1, D0, and D1. Every action checks endpoint UUID/PCI
identity and fresh graphics-clock readback. Commands on one physical GPU are
serialized; fresh actual-clock readback uses a bounded post-command poll so an
immediate idle sample cannot masquerade as actuator failure. Minimum dwell is enforced, and all GPUs are explicitly restored
to their conservative validated HIGH state on success or failure. The outer
Slurm cleanup subsequently restores default graphics and memory clocks.

After isolation, three small Phase 3C-audited serving windows run at HIGH,
P0 MID, and restored HIGH. They reuse real SSE, logical request IDs, token and
latency auditing, Prometheus semantics, and per-endpoint NVML energy windows.

Stage A outputs are under `results/phase3d_actuator_validation/<run-id>/`:

- `actuator_actions.csv`
- `actuator_readback.csv`
- `actuator_audit.json`
- `actuator_summary.md`
- raw Phase 3C window logs, metrics, and NVML samples

## Stage B: feedback-only joint routing and DVFS

Stage B refuses to start unless its configured Stage A audit exists and has
`valid: true`. It uses only recent measured queue/KV, latency, energy/power,
health, frequency, and freshness state. The existing predictor-free feedback
scheduler supplies compatible-route and DVFS recommendations. Successful
physical readback—not recommendation time—starts the DVFS dwell interval.

The proxy reads an atomic, freshness-stamped route-control file for every
request and fails closed on missing, stale, malformed, or incompatible state.
At startup or without sufficient feedback the controller uses only previously
validated pairs at conservative HIGH clocks; missing data is never treated as
zero load or available headroom.

The smoke contains only light -> moderate -> light windows. Its outputs are
under `results/phase3d_closed_loop_smoke/<run-id>/` and include the requested
control iterations, request/routes, endpoint telemetry, DVFS actions, energy,
audit, summary, and raw per-window evidence.

Stage B is functional validation only. A later experiment may compare this
feedback-only controller with model-heavy and offline-oracle baselines.
