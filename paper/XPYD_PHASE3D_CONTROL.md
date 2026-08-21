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

The dynamic 2P2D launch explicitly disables endpoint-local prefix caching.
The current P2pNcclConnector does not carry cache-coherence or cache-affinity
state across a route change: a prefix cached on one P endpoint can suppress a
KV send even though the newly selected D endpoint does not own that prefix.
Disabling this optimization is the conservative correctness policy for the
Phase 3C/3D routing substrate; it does not alter the feedback scheduler.

The same substrate disables vLLM 0.15.1 chunked prefill. With local prefix
caching disabled, the nominal IL2048 synthetic prompt contains 2049 tokenizer
tokens and crosses the default 2048-token chunk boundary. Physical Stage B
evidence showed the producer complete that request while the experimental
P2pNcclConnector emitted no KV layer tensors. Non-chunked prefill is therefore
the conservative connector-compatible policy; this is a functional substrate
boundary, not a claim about chunked-prefill performance.

Logical request IDs are also namespaced by workload window. The four vLLM
servers and their KV connectors persist across the complete Stage B sequence,
so restarting ordinal IDs at `phase3c-0000` in every window would collide with
connector completion state from an earlier window. Window-qualified IDs remain
identical end-to-end for correlation while staying unique over that persistent
server lifetime.

The present physical evidence validates one in-flight request per selected
P2pNcclConnector pair. Stage B therefore audits every workload against
`validated_connector_max_concurrency=1`. The moderate window remains heavier
through its longer 2048-token prompt, 256-token generation, and request count;
it does not claim concurrent-request connector validation. Raising concurrency
requires a separate connector-capability experiment before it can be used as a
control-smoke input.

The smoke contains only light -> moderate -> light windows. Its outputs are
under `results/phase3d_closed_loop_smoke/<run-id>/` and include the requested
control iterations, request/routes, endpoint telemetry, DVFS actions, energy,
audit, summary, and raw per-window evidence.

Stage B is functional validation only. A later experiment may compare this
feedback-only controller with model-heavy and offline-oracle baselines.
