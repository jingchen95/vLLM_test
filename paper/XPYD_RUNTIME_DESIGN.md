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
mock runtime only validates an already chosen route.

## Old and new control boundaries

| Concern | Current research stack | XpYd architecture boundary |
|---|---|---|
| Serving unit | `GPUPool` / `PoolBundle` | Immutable `EndpointSpec` plus mutable `EndpointState` |
| Topology | L40S/L4 pool abstractions | Arbitrary GPU-type and node profiles |
| TP | Uniform choice inside a modeled pool/configuration | Mixed, pre-instantiated TP endpoints in one inventory |
| P/D route | Modeled pool pair such as A->B | Independent concrete P endpoint and D endpoint IDs |
| Frequency | Modeled pool-level frequency | Independent frequency setpoint for each selected endpoint |
| Runtime state | Aggregate simulation state | Health, lifecycle, queue, running requests, KV occupancy, timestamp |
| Execution | Simulation/model evaluation | No execution yet; registry and mock validation only |

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
sources already documented in `CODEBASE_AUDIT.md`:

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

## Remaining implementation work

- A vLLM launcher and endpoint lifecycle manager for the slow knobs.
- Live health, queue, request, KV-cache, power, and clock telemetry.
- Measured NIXL/NCCL compatibility rows, including cross-TP correctness,
  bandwidth, latency, and failure behavior.
- A privileged, verified DVFS actuator with rollback and clock-readback.
- A model adapter that converts endpoint telemetry and canonical priors into
  per-endpoint TTFT, TPOT, capacity, and energy estimates.
- Hardware replay/integration tests before any live routing or actuation policy
  is trusted.
