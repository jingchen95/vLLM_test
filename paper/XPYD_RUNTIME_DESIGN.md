# XpYd endpoint-runtime architecture

## Scope

This is an architecture boundary for a future energy-aware `XpYd` serving
runtime. It represents heterogeneous, pre-instantiated prefill and decode
endpoints with mixed tensor-parallel degrees, independent P/D routing, and
per-endpoint frequency choices. The code is deliberately isolated under
`paper/scripts/xpyd/`.

It does **not** launch vLLM, discover endpoints, transfer KV state, change GPU
clocks, implement lifecycle transitions, or alter the current SWEEP scheduler,
`EnergyScheduler`, `PoolBundle`, baselines, simulator, models, or results. The
mock runtime validates explicit routes, while the feedback reference selects
routes and recommends clocks without performing runtime I/O or actuation.
The live-observability layer may issue read-only HTTP GET requests to configured
vLLM `/metrics` endpoints; fixture replay performs no network operation.

## Old and new control boundaries

| Concern | Current research stack | XpYd architecture boundary |
|---|---|---|
| Serving unit | `GPUPool` / `PoolBundle` | Immutable `EndpointSpec` plus mutable `EndpointState` |
| Topology | L40S/L4 pool abstractions | Arbitrary GPU-type and node profiles |
| TP | Uniform choice inside a modeled pool/configuration | Mixed, pre-instantiated TP endpoints in one inventory |
| P/D route | Modeled pool pair such as A->B | Independent concrete P endpoint and D endpoint IDs |
| Frequency | Modeled pool-level frequency | Independent frequency setpoint for each selected endpoint |
| Runtime state | Aggregate simulation state | Health, lifecycle, queue, running requests, KV occupancy, timestamp |
| Execution | Simulation/model evaluation | No execution; registry, mock validation, and CPU-only feedback |

The new package is not an adapter that silently turns an existing `PoolBundle`
into a deployed endpoint. Such an adapter would incorrectly imply that the
simulator's active-instance, queue, and frequency decisions have already been
applied to real vLLM processes.

## New abstractions

`EndpointSpec` is immutable launch/placement identity. It has an explicit role,
arbitrary GPU type, node, exact physical GPU IDs, and its own TP degree. The
constructor enforces `len(gpu_ids) == tp_degree` and unique GPU IDs; no TP value
is hard-coded.

`EndpointState` holds observations only: frequency, `ACTIVE`/`WARM`/`COLD`/
`UNAVAILABLE`, health, queue depth, running requests, KV-cache occupancy, and
last-update time. It intentionally defines no state-transition rules.

`HardwareProfile` contains read-only `GPUTypeProfile` and `NodeProfile`
inventories. Product names are data, not branches in the implementation. The
optional `from_gpu_specs()` adapter accepts the existing `GPU_SPECS` shape;
`from_current_gpu_specs()` is the only explicit bridge that imports the current
simulator inventory.

`EndpointRegistry` is an in-memory inventory with registration, state updates,
lookup, role listing, and healthy/active filtering. It has no network discovery.
By default, two simultaneously `ACTIVE` endpoints cannot claim the same
`(node, gpu_id)`. Tests must opt in explicitly to allow overlap.

`RoutingDecision` contains only the fast-path choice: prefill endpoint, decode
endpoint, prefill frequency, and decode frequency. Because TP belongs to each
endpoint, the decision naturally represents `P TP != D TP` without introducing
a shared TP field.

`EndpointTelemetrySample` and `EndpointTelemetrySnapshot` are deliberately
separate from `EndpointState`. State remains the latest operational fact;
telemetry represents observations over time and their smoothed summaries.

## Compatibility

`CompatibilityTable` is keyed by `(connector, prefill_tp, decode_tp)`. Each
`ConnectorCompatibility` row records support, a reason, and optional measured
bandwidth and latency. A route is compatible only when:

1. the first endpoint is prefill and the second is decode;
2. both name the same non-empty KV connector;
3. the exact connector/P-TP/D-TP tuple is present; and
4. that row is marked supported.

Missing connector evidence, mismatched connectors, reversed roles, and unknown
TP pairs all fail closed. In particular, support for equal TP or one cross-TP
pair is never extrapolated to another pair.

Connector compatibility is not connector performance. A future
`TransferProfile`-like abstraction should be keyed by source endpoint/node,
destination endpoint/node, connector, TP pair, and network topology. It may
eventually report bandwidth, latency, and energy per byte. Those measurements
must not be inserted into `CompatibilityTable`, whose only question is whether
the handoff is supported.

## Example topology

The CPU-only mock uses one arbitrary `example_accelerator` node with eight fake
GPUs:

| Endpoint | Role | GPUs | TP |
|---|---|---:|---:|
| P0 | prefill | 0 | 1 |
| P1 | prefill | 1 | 1 |
| P2 | prefill | 2, 3 | 2 |
| D0 | decode | 4 | 1 |
| D1 | decode | 5 | 1 |
| D2 | decode | 6, 7 | 2 |

The example table explicitly permits `mock-kv/TP2->TP1` and
`mock-kv/TP1->TP2`. Therefore `P2->D0` and `P0->D2` produce decisions with
independent clocks. `P0->D0` is rejected because `TP1->TP1` has no evidence
row. Run it with:

```bash
PYTHONPATH=paper/scripts .venv/bin/python -m xpyd.mock_runtime
```

## Control timescales

Fast knobs can change per request or control window without relaunching a
server: select the P endpoint, select the D endpoint, route the request, and
choose independently validated DVFS setpoints.

Slow knobs change the deployed inventory: physical layout, TP degree, endpoint
activation, process launch/termination, and warm/cold lifecycle management.
They require placement and transition policies outside this package. A future
controller must not pretend that TP is a fast knob when it actually requires a
different pre-instantiated endpoint or relaunch.

## Telemetry and predictor-free feedback control

The feedback controller is measurement-driven, feedback-based, profile-light,
CPU-testable, and independent of the canonical model stack. It reads current
`EndpointState`, recent `EndpointTelemetrySnapshot` values, explicit connector
compatibility, hardware-supported clock values, and configured safety
thresholds. It does not import or query `EnergyScheduler`.

It is **not** a learned predictor, an oracle, an SLO guarantee, online RL, a
bandit policy, or a production controller. It observes vLLM but is not connected
to HTTP request routing, NVML, a wattmeter, NIXL/NCCL, or a GPU-clock actuator.
The configurable safety fraction is only a heuristic margin over recently
observed behavior.

### Live observability path

The implemented read-only path is:

```text
vLLM 0.15.1 /metrics
  -> standard-library Prometheus text parser
  -> VLLMMetricsCollector raw cumulative snapshot
  -> per-endpoint cumulative-to-window deltas
  -> TelemetryAggregator EWMA and recent-window summaries
  -> FeedbackScheduler route/DVFS evaluation
  -> JSONL dry-run decision (actuated=false)
```

`VLLMMetricsCollector` joins each `EndpointSpec.http_uri` with `/metrics`, uses
a configurable timeout, validates HTTP status, and performs no retry or hidden
fallback. The repository's physical script exposes vLLM's OpenAI server on
`0.0.0.0` (prefill port 8100 and decode port 8200 in that two-server script)
and checks `/health`; XpYd accepts arbitrary remote host/port URIs and assumes
`/metrics` is served by that same process. It neither launches nor terminates
the process.

The collector recognizes these exact names:

```text
vllm:num_requests_running
vllm:num_requests_waiting
vllm:kv_cache_usage_perc
vllm:prompt_tokens_total
vllm:generation_tokens_total
vllm:request_success_total
vllm:time_to_first_token_seconds
vllm:inter_token_latency_seconds
vllm:e2e_request_latency_seconds
vllm:request_queue_time_seconds
vllm:request_prefill_time_seconds
vllm:request_decode_time_seconds
```

No individual metric is assumed present: absent names are listed in
`missing_metrics` and represented as `None`. The three gauges are used to
synchronize running requests, waiting requests (the `EndpointState.queue_depth`
mapping), and KV occupancy only when present. A successful scrape marks the
endpoint healthy and advances its last-update timestamp. A connection, HTTP,
parse, or mapping failure marks it unhealthy but retains the registry entry,
last successful timestamp, and last known gauge values.

`EndpointState` separately records whether queue depth and KV occupancy have
ever been observed. Their numeric defaults are storage placeholders, not
evidence of zero pressure. An explicit scraped zero is known zero; a first
missing gauge remains unknown; a later missing gauge preserves a prior known
value. Optimized routing and fallback both fail closed on never-observed queue
or KV pressure. Dry-run JSON includes the two observed flags so this distinction
is auditable.

`model_name` is selected explicitly when configured. Without configuration, a
single model label can be inferred, while multiple model series fail clearly
instead of being combined. `request_success_total` is summed over configured
successful finish reasons; the default includes `stop` and `length` and
deliberately excludes abort/error reasons.

All listed metrics are optional at scrape/mapping time so version- or
configuration-dependent omissions are visible rather than fatal. Eligibility
requirements are policy-dependent: queue/running/KV state is updated only from
present gauges, while the configured EWMA/P95/P99 policy requires its matching
TTFT or TBT evidence. The scheduler default remains EWMA for compatibility; the
included dry-run fixture explicitly selects window P99.

### Cumulative counters and histogram windows

Raw snapshots retain process-lifetime counters and histograms. A stateful
tracker subtracts consecutive scrapes and only then derives token/request rates
and recent latency. A first scrape establishes a baseline. A decreasing
counter, decreasing bucket/count/sum, changed bucket layout, or non-positive
interval invalidates the interval; no negative delta is emitted.

This detects observable counter/histogram discontinuities, not process identity.
Without a process start-time or instance-identity signal, a restart whose new
counters have already exceeded the previous values is indistinguishable from a
continuing process. Such monotonic input is therefore accepted as a delta; the
implementation does not claim unconditional restart detection.

Prometheus histogram buckets are cumulative both across bucket boundaries and
over process lifetime. For example, if TTFT is:

| scrape | <=50 ms | <=100 ms | <=200 ms | total |
|---|---:|---:|---:|---:|
| t0 | 10 | 20 | 20 | 20 |
| t1 | 15 | 28 | 30 | 30 |
| t1-t0 window | 5 | 8 | 10 | 10 |

then the recent window has ten observations. Both P95 and P99 first reach their
targets in the finite 200 ms bucket, so this implementation reports a
conservative bucket upper bound of 200 ms. These are bucket-resolution
approximations, not exact request-level percentiles. If a quantile is reached
only in `+Inf`, it remains unknown. The fraction above an SLO is returned only
when the SLO exactly matches a bucket boundary; an SLO inside a bucket returns
`None` rather than an invented exact percentage.

### Raw samples and EWMA

A raw sample may independently contain latency, power, interval energy, queue,
running-request, KV-cache, token, and completion measurements. Missing metrics
remain unknown. They are never filled from another metric or from an offline
model.

For each available scalar metric, the aggregator applies:

```text
m_t = alpha*x_t + (1-alpha)*m_(t-1)
```

The first observation initializes the value and `None` does not update it.
`alpha` is configuration, not a research conclusion. **EWMA summarizes recent
latency level and smooths noise; it is not equivalent to P95/P99 SLO evidence**
and does not predict unobserved configurations. The snapshot therefore retains
both EWMA TTFT/TBT/TPOT and the latest valid window's TTFT/TBT P95/P99, sample
counts, token/request rates, latency-observation timestamp, and age.

Energy semantics remain explicit:

- `energy_j / completed_requests` is computed only when the sample includes
  interval energy and a positive completed-request count.
- `energy_j / output_tokens` is computed only with a positive output-token
  count.
- service rate is computed only when an explicit positive observation interval
  is present.
- an instantaneous power sample is retained as power; it is never converted to
  energy without elapsed time.

### Routing feedback

`FeedbackScheduler.choose_route()` first considers healthy, `ACTIVE` endpoints.
It rejects instantaneous queue/KV pressure, unsupported observed frequencies,
and P/D pairs without exact connector/TP compatibility evidence. Prefill and
decode safety can independently select EWMA, recent-window P95, or recent-window
P99. Tail policies require `min_tail_samples`. Every policy also requires a
latency observation no older than `telemetry_max_age_s`; never-observed,
insufficient, or stale endpoints are not considered measured-safe. A stale
example is an observation at t=1 evaluated at t=100 with a 30-second maximum:
the pair is rejected as `<endpoint>_telemetry_stale`, while only the separately
validated configured fallback may bypass missing/stale latency evidence.

Safe pairs with energy/request observations on both endpoints are ranked by
`P energy/request + D energy/request`. KV-transfer energy is an explicit future
term and is currently omitted. Energy/output-token is collected but is not
mixed into this score because phase/workload comparability has not been
established. When no candidate has comparable per-request energy, endpoint IDs
provide a deterministic ordering rather than an invented power-to-energy
conversion. The optional `workload_context_key` is carried through candidate
evaluation for future accounting only; current energy history is not filtered
or normalized by it.

If no measured-safe route exists, the scheduler uses the fully configured
fallback P/D endpoints and frequencies. The fallback may bypass missing
telemetry, but never health, lifecycle, queue/KV, hardware-frequency, role, or
connector-compatibility checks. Without a valid fallback, selection fails
closed. There is no online exploration.

### DVFS feedback

`choose_frequency_adjustment()` is separate from routing. It returns an
immutable recommendation containing the observed current frequency and the
requested target frequency; it never executes a clock change.

- Missing telemetry produces `HOLD`.
- Latency above the configurable upshift fraction, or queue/KV pressure,
  produces a one-level `STEP_UP`.
- Severe configured pressure may produce `FALLBACK_MAX`.
- Latency at or below the configurable downshift fraction with low pressure
  produces a one-level `STEP_DOWN`.
- The gap between downshift and upshift thresholds is a hysteresis `HOLD`
  region; validation requires `0 < downshift < upshift <= 1`.
- `dvfs_min_dwell_s` is measured only from a separately recorded successful
  external actuation/readback. A recommendation alone never starts dwell, so
  unchanged dry-run state may produce the same recommendation repeatedly.
  Emergency SLO/queue/KV pressure and `FALLBACK_MAX` may bypass dwell after a
  real actuation.
- Minimum/maximum boundaries produce `HOLD`.

Every target is a real member of the endpoint GPU type's allowed-frequency set.
The observed and requested frequencies remain separate because a future
actuator may be delayed or fail readback verification. Dry-run recommendation
history records only recommendation time; it never writes the hypothetical
target back into observed `EndpointState.freq_mhz`. Recommendation and actual
actuation timestamps are separate. `record_dvfs_actuation()` only records a
future actuator's already-successful transition after the registry frequency
matches hardware readback; it performs no frequency mutation itself.

### Dry-run controller and replay

`DryRunController` iterates an arbitrary multi-node inventory, scrapes each
endpoint, updates only present observed gauges, feeds valid windows to the
existing aggregator, invokes the existing scheduler, and emits compact JSONL.
Every record includes endpoint identity/state/freshness, all P/D eligibility
results and reasons, an optional energy score and context key, selected
hypothetical route, fallback status, and hypothetical DVFS recommendations.
Raw histogram buckets are omitted by default and `actuated` is always `false`.

The included six-endpoint replay can be run without CUDA, network access, or a
vLLM process:

```bash
PYTHONPATH=paper/scripts .venv/bin/python -m xpyd.dry_run_controller \
  --config tests/fixtures/vllm_metrics/dry_run_config.json \
  --fixture-manifest tests/fixtures/vllm_metrics/dry_run_manifest.json \
  --output /tmp/xpyd-dry-run.jsonl
```

Live read-only observation uses the same command without `--fixture-manifest`
and honors `--interval` and `--duration`. Fixture text goes through the exact
same parser and collector mapping as an HTTP response.

### Energy and request-context limits

vLLM `/metrics` provides serving latency, queue, token, completion, and KV
signals; it does **not** provide the GPU energy measurements required for
endpoint energy ranking. It never populates `power_w` or `energy_j`. The
physical benchmark's NVML/hardware energy counter remains a separate source;
real GPU/NVML or external-meter integration is future work.

Current endpoint energy history is also not normalized by request/workload
context. Comparing an endpoint serving mostly short prompts with one serving
long prompts is therefore not fair energy evidence. Energy-based ranking is an
experimental placeholder until context-aware accounting is added; no bucketing,
ML, bandit, or exploration mechanism is introduced here.

### Deterministic CPU example

The example adds explicit compatibility evidence for all four TP1/TP2
directions and injects the following single-window measurements:

| Endpoint | Latency EWMA | Energy/request |
|---|---:|---:|
| P0 | TTFT 300 ms | 2.0 J |
| P2 | TTFT 250 ms | 2.5 J |
| D0 | TBT 55 ms | 5.0 J |
| D1 | TBT 92 ms | 4.0 J |
| D2 | TBT 60 ms | 5.5 J |

At TTFT SLO 500 ms, TBT SLO 100 ms, and safety fraction 0.8, D1 is
filtered because 92 ms exceeds the 80 ms margin. P1 is unmeasured. Remaining
pair energies are P0+D0=7.0 J, P0+D2=7.5 J, P2+D0=7.5 J, and P2+D2=8.0 J,
so P0->D0 wins. P2's 250 ms TTFT plus low pressure yields only a one-level
1600->1200 MHz `STEP_DOWN` recommendation.

```bash
PYTHONPATH=paper/scripts .venv/bin/python -m xpyd.feedback_example
```

### Future interfaces, not implementations

Future work needs interfaces for NVML or external-wattmeter energy collection,
NIXL KV-transfer telemetry,
topology-aware transfer cost, a clock actuator with readback, safe exploration,
contextual bandits, and model-assisted counterfactual prediction. None is part
of this non-actuating reference controller.

## Phase 3A live observability validation

Phase 3A does not extend the scheduler. It wraps the existing physical
Experiment E 1×L40S prefill → 1×L4 decode path to test whether real vLLM 0.15.1
metrics have the semantics assumed by this CPU-only architecture.

Controlled semantic probes scrape P0 and D0 before and after a known small
logical request count and compare independent client results with each
endpoint's request, prompt-token, generation-token, and histogram deltas. Short
load probes use configurable rate/duration/shape values and independently
scrape both endpoints on a fixed-period schedule to observe queues, running
requests, KV use, rates, tail buckets, missing metrics, scrape latency/drift,
and explicit late or missed samples.

Every HTTP response is retained verbatim alongside centrally assigned wall and
monotonic timestamps, sequence, endpoint role, and URI. Client TTFT/TPOT/E2E
and success come from the existing streaming trace client; exact output counts
come from final D OpenAI stream usage forwarded unchanged by the proxy, not from
Prometheus or text re-tokenization. The proxy records monotonic prefill,
decode-header, first/last-real-chunk, forwarding, and completion timestamps, but
does not substitute them for client metrics. A phase-level request warms lazy
NCCL state before the first metric baseline and is excluded from measured
windows. Client and endpoint values are therefore independent evidence sources;
agreement is reported as observed correlation, not proof of internal phase
ownership.

The Phase 3A physical branch reuses the established server, connector, proxy,
health-check, log, and process-cleanup path, but skips energy/NVML collection,
GPU frequency/persistence/reset commands, routing actuation, and all model
training. P/D metric semantics remain the object under validation until the
physical run is performed. The design and exact physical commands are in
`XPYD_PHASE3A_OBSERVABILITY.md` and `XPYD_PHASE3A_RUNBOOK.md`.

## Canonical-model provenance and reuse

All four canonical phase families were trained from the full monolithic,
single-pool Phase-2 master tables. The tables contain balanced and
phase-dominant request shapes, but training did not select a phase-dominant-only
subset. “Prefill” and “decode” identify targets/features, not physical serving
separation.

| Canonical family | Training provenance | Not used for training |
|---|---|---|
| L40S prefill classifiers/latency/power | **Monolithic** | phase-only and physically disaggregated rows |
| L40S decode classifiers/latency/power | **Monolithic** | phase-only and physically disaggregated rows |
| L4 prefill classifiers/latency/power | **Monolithic** | phase-only and physically disaggregated rows |
| L4 decode classifiers/latency/power | **Monolithic** | phase-only and physically disaggregated rows |

No canonical estimator is trained solely from phase-dominant, phase-only, or
physically disaggregated measurements. The later `CD_prefill_decode_only` and
`E_disaggregated` data are diagnostic/feasibility evidence, not canonical model
training inputs.

Consequently the current readiness interpretation is:

- L40S prefill characterization: **PARTIALLY READY** — an intrinsic phase model
  exists; validate it under a real P/D runtime.
- L4 decode characterization: **PARTIALLY READY** — an intrinsic phase model
  exists; validate continuous-batching and transferred-KV effects.
- Power models: **HIGHLY REUSABLE** — validate monolithic-to-P/D transferability.
- TTFT/TPOT models: **REUSE AS PRIORS OR RANKERS** — recalibrate for live XpYd.
- Capacity models: **REUSE AS INITIAL ESTIMATORS** — validate endpoint-level
  capacity.

No model was retrained for this architecture work.

## Pre-change baseline record

These checks were run on CPU/macOS before any files were added on 2026-08-17.
The joblib warning about physical-core detection was non-fatal; it fell back to
logical cores.

```text
$ PYTHONPATH=paper/scripts .venv/bin/python paper/scripts/scheduler.py \
    --mode query --il 1024 --ol 128 --rate 10 --slo 500 \
    --model-dir artifacts/paper/models/models_l40s
Loaded models from artifacts/paper/models/models_l40s
Config space: TP=[1, 2, 4], Freq=[210..2520] (10 levels)
Phase-specific stack: prefill + decode
status=NO_SAFE_CONFIG, num_candidates=30, num_safe=0

$ PYTHONPATH=paper/scripts .venv/bin/python paper/scripts/scheduler.py \
    --mode query --il 128 --ol 512 --rate 5 --slo 500 \
    --model-dir artifacts/paper/models/models_l4
Loaded models from artifacts/paper/models/models_l4
Config space: TP=[1, 2, 4, 8], Freq=[210..2040] (10 levels)
Phase-specific stack: prefill + decode
status=OK, recommended=(TP4, 1200 MHz), num_safe=10

$ PYTHONPATH=paper/scripts .venv/bin/python -c '<one-request static-disagg decision>'
{'total_power_w': 1366.8, 'slo_met': True, 'phase': 'STATIC',
 'routes': {'short_short': 'A->B'}}

$ .venv/bin/python -c '<AST-parse every tracked Python file>'
AST parsed 122 tracked Python files

$ bash -n run_disagg_benchmark.sh
bash -n run_disagg_benchmark.sh: OK
```

The known path-integrity check failed only on the same three absent generated
sources already documented in `docs/CODEBASE_AUDIT.md`:

```text
$ make -C paper check-paths
missing .../obs2_regime_search_primary/r2_ttft500_tpot200_mixedlen_obs2_main.pdf
missing .../section42/synthetic_trace_overview.pdf
missing .../section42/tau_kv_sensitivity.pdf
make: *** [check-paths] Error 1
```

The full strategy smoke is not lightweight in this snapshot. It loaded all
canonical L40S/L4 phase stacks and started one window, but did not finish its
first SWEEP search in about 56 seconds, so it was interrupted rather than left
running:

```text
$ .venv/bin/python smoke_test_strategies.py --num-windows 1
Running 1 windows at SLO_TTFT = 500 ms (TPOT = 200 ms)
... no first-window result after ~56 s ...
KeyboardInterrupt in schedulers/sweep.py model evaluation
```

Before the live `/metrics` extension on 2026-08-18, the existing XpYd suite
passed all 44 tests. The original mock emitted P2->D0 and P0->D2, and the
one-request static-disaggregation smoke remained `1366.8 W`, SLO-met, A->B.
The canonical model and result tree hashes were respectively
`c5231b3d3b12a7f4cf101dbcdb3703bf9c55c8ec` and
`0048e0abe4b91a680ae6f462abb01a60cef11829`.

## Remaining implementation work

- A vLLM launcher and endpoint lifecycle manager for the slow knobs.
- Real GPU power/energy and verified clock-readback telemetry (vLLM runtime
  health/queue/request/KV observation is now implemented read-only).
- Measured NIXL/NCCL compatibility rows for cross-TP correctness, plus a
  separate topology-aware KV-transfer performance profile.
- A privileged, verified DVFS actuator with rollback and clock-readback.
- A model adapter that converts endpoint telemetry and canonical priors into
  per-endpoint TTFT, TPOT, capacity, and energy estimates.
- Hardware replay/integration tests before any live routing or actuation policy
  is trusted.
