# XPYD implementation gap audit

**Audit date:** 2026-09-13
**Authoritative specification:** [`XPYD_SYSTEM_CONTROL_DESIGN.md`](XPYD_SYSTEM_CONTROL_DESIGN.md)
**Scope:** repository/code/configuration audit only; no GPU job, serving workload, threshold tuning, or controller change was performed.

## Audit purpose

The purpose of this audit is to identify the gap between the intended XPYD
architecture/control design and the implementation that is currently in the
repository. It is a planning artifact: it makes mismatches explicit so that
subsequent work does not silently change experiment semantics. It is not a
claim that the missing design elements have been implemented, and it is not a
request to run a new experiment.

Line references below are to the current working tree. `IMPLEMENTED` means the
behavior is present in the inspected path; `PARTIALLY_IMPLEMENTED` means only
a subset of the design contract is present; `EXPERIMENT_ONLY` means the
behavior exists in a later/limited harness but is not the current runtime
contract; `NOT_IMPLEMENTED` means no corresponding implementation was found;
`TBD_BY_MEASUREMENT` means the design intentionally leaves the value to future
calibration.

## 1. End-to-end request path

| Stage | Current implementation and evidence | Inputs / outputs / persistent state | Granularity and gap |
|---|---|---|---|
| Workload construction | `paper/scripts/xpyd/workload_harness.py:148-223`, `build_trace()` | Scenario segments, class weights, arrival rate and requested `input_len`/`output_len` become a trace with `request_id`, `arrival_time_s`, class and segment. | Trace/request granularity. No runtime system-context labels are sent to the server. |
| Admission and client concurrency | `paper/scripts/replay_synthetic_trace.py:36-90,418-517`, `TraceRequest`, `ClientLoadTracker`, `replay_trace()` | Trace arrival schedules a task; an `asyncio` semaphore (`max_concurrency`) gates admission. Waiting/admission timestamps are recorded; client waiting is excluded from server TTFT. | Per request. The semaphore can hide backlog from endpoint queue metrics. |
| Classification | `workload_harness.py:19-40,148-179` (`REQUEST_CLASSES`, `RequestBuckets`) | SS/SL/LS/LL is assigned from requested input and requested output lengths at trace creation. | Per request, based on requested budget, not realized completion. |
| Ingress ID | `replay_synthetic_trace.py:208-388,418-517`; `disagg_proxy.py:258-355` | Client sends `X-Request-Id` and `X-Xpyd-Logical-Request-Id`; proxy reuses an existing ID or generates one once and derives a stable vLLM transport ID. | One logical request. Diagnostics, P/D headers, SSE echo and result joins use this ID. |
| Proxy/router | `disagg_proxy.py:576-760`, `RoundRobinCompatiblePairs`, `FileControlledCompatiblePairs`, `MultiEndpointDisaggProxyCore` | Explicit compatible P/D pair registry; RR or a fresh control-file choice selects one pair; one `DisaggProxyCore` is pre-instantiated for each pair. | Request-level route selection. Current C1 path is pair-aware/compatibility-aware but not context-, pressure-, or energy-aware. |
| P endpoint | `run_disagg_benchmark.sh:846-953,1515-1601` | P servers run on explicit GPUs/ports, TP startup is fixed, producer connector performs prefill and publishes KV. | Endpoint/server process. No runtime TP reconfiguration. |
| KV handoff | `disagg_proxy.py:402-565` (`prepare()`) | Proxy performs P request, marks handoff, then sends D request with the same logical ID/diagnostic headers; connector envelope carries stable vLLM request ID. | Per request, synchronous handoff in the proxy path. |
| D endpoint and SSE | `disagg_proxy.py:366-400,402-565`; `replay_synthetic_trace.py:208-388` | D response stream is consumed; first token gives TTFT, completion usage or retokenization gives output-token count; TPOT/TBT is computed from inter-token timing. | Per request/stream. SSE validity and echoed IDs are audited. |
| Results/diagnostics | `disagg_proxy.py:147-220,366-400`; `phase3c_substrate.py:450-535` | Proxy diagnostics include route, timestamps, phase durations, SSE flags and IDs. Analysis joins diagnostics and client rows by exact logical ID and validates route/event order. | Per request plus window summaries. |
| Telemetry | `phase3a_observability.py:85-184,412-572,877-1075`; `telemetry.py:206-359` | Fixed-period endpoint scrapes collect power/energy, queue/running/KV, token and latency fields; endpoint-local EWMA/snapshot state is maintained. | Continuous fixed-period sampling (independent endpoint workers), not request-level attribution. |
| Window energy | `phase3b_energy.py:139-457,460-504,618-805` | NVML samples are bounded by wall/monotonic window; energy-counter delta or trapezoidal power integration gives gross per-GPU/window energy. All logical requests normalize the window once. | Window/endpoint amortization. It is explicitly not physical request-level energy attribution. |
| Feedback/controller | `phase3d_control.py:706-914,974-1170`; `feedback_scheduler.py:280-433` | After an observed window, scheduler evaluates routes and recommends DVFS actions from endpoint-local telemetry; action/readback rows are persisted. | Current C1 is window-driven. The design calls for evidence epochs plus event-driven safety actions; those are not yet the runtime trigger model. |
| DVFS actuation/readback | `phase3d_control.py:142-370,776-914`; `stagec1_stationary.py:218-305` | `NvidiaSmiClockBackend` sets/readbacks one physical GPU per endpoint; C1 coordinator converts endpoint recommendations into shared P/D pool actions. | Actuator is endpoint-scoped, C1 policy is pool-scoped. Hardware readback is recorded. |

### Waiting points (kept separate)

1. **Pre-admission:** trace arrival can wait until the scheduled task runs (`replay_synthetic_trace.py:418-517`).
2. **Client semaphore:** `max_concurrency` waiting is measured by `ClientLoadTracker` (`replay_synthetic_trace.py:36-90`) and is not server TTFT.
3. **Proxy/router:** request can spend time before pair dispatch; proxy diagnostics expose ingress/route/P/D timestamps (`disagg_proxy.py:147-220,402-565`).
4. **P endpoint:** vLLM running/waiting and sampled KV/queue are telemetry observations, not a complete history of short prefill work (`phase3a_observability.py:1006-1075`).
5. **D endpoint:** decode running/waiting is similarly sampled. Endpoint queue `0` does not prove that the client semaphore or proxy has no backlog (`phase3a_observability.py:534-572`; `phase3c_substrate.py:536-586`).
6. **Other:** KV-transfer and network/connector delay are represented in proxy phase timestamps, but no independent energy or queue attribution is implemented (`disagg_proxy.py:402-565`; `feedback_scheduler.py:567-581`).

## 2. Request context

| Design requirement | Current code | Assessment |
|---|---|---|
| SS/SL/LS/LL labels | `workload_harness.py:19-40` uses input/output bucket thresholds; `build_trace()` writes `workload_class` (`:148-179`). | `IMPLEMENTED` for trace labels. |
| Current thresholds | `RequestBuckets`: input `<=512`, output `<=256` (with the four validated shapes in `workload_harness.py:50-68`). | `CALIBRATED` workload definition, not a universal model. |
| Requested vs actual output | Replay sends `max_tokens=req.output_len` and records completion usage/retokenized actual output (`replay_synthetic_trace.py:418-617`). | Requested budget drives online class; actual completion is post-hoc. |
| Assignment point | At trace construction, before admission and routing (`workload_harness.py:148-179`). | No late reclassification. |
| Routing visibility | Proxy receives IDs and payload but no class/context header (`disagg_proxy.py:258-355`; `replay_synthetic_trace.py:418-517`). | `NOT_IMPLEMENTED` for context-aware routing in the current C1 path. |
| Logs/results | Class/segment and requested/actual lengths are in client/analysis rows (`replay_synthetic_trace.py:519-617`; `phase3c_substrate.py:450-535`). | `IMPLEMENTED` for auditability. |
| Future information/oracle | Phase 4A/4B route stores and oracle artifacts are separate later experiments (`route_probing.py:75-333`; `phase4b1_evaluation.py:280-375`). | Not available to the C1 controller; must not be treated as online information. |

The design's system-context half (load magnitude, phase balance, BURST/RISK) is
described in `XPYD_SYSTEM_CONTROL_DESIGN.md:280-420`, but the current request
payload and C1 scheduler do not carry or maintain those labels.

## 3. System context

The specification defines two context levels: request SS/SL/LS/LL and system
state `LOW/MEDIUM/HIGH`, phase `PREFILL_HEAVY/DECODE_HEAVY/BALANCED`, and a
`BURST_OR_RISK` flag (`XPYD_SYSTEM_CONTROL_DESIGN.md:280-420`). Current evidence:

| Signal | Evidence in repository | Current status/gap |
|---|---|---|
| Offered-load magnitude | C0 accepted rates are .05, .10, optional .15, .20; .30 was overload (`XPYD_SYSTEM_CONTROL_DESIGN.md:424-479`). | `CALIBRATED` experiment points, not a general load classifier. |
| Phase balance | Trace class mix exists, but no maintained `rho_P/rho_D` or phase-balance state in `telemetry.py:206-359` or `feedback_scheduler.py:195-433`. | `NOT_IMPLEMENTED`; exact normalization/boundaries are intentionally TBD. |
| Burst/risk | Scheduler has pressure/risk predicates from latency, queue and KV (`feedback_scheduler.py:301-433,435-565`), but no named BURST state. | `PARTIALLY_IMPLEMENTED` safety proxies; `NOT_IMPLEMENTED` as a system-context label. |
| Utilization/pressure support | Endpoint queue/running/KV, client waiting and admission are collected (`phase3a_observability.py:534-572`; `stagec1_stationary.py:181-187`). | Useful observations, but cadence and sampling blind spots remain. |
| `rho_P/rho_D` | No source computes the design's normalized phase-work ratios. | `TBD_BY_MEASUREMENT`; no formula or threshold is invented here. |

## 4. Feedback granularity

| Design item | Current behavior | Gap |
|---|---|---|
| Routing timing | Pair selection occurs on each request in proxy (`disagg_proxy.py:666-760`). | Matches request-level routing. |
| Telemetry | Independent fixed-period endpoint workers, with scheduled/start/finish/latency/drift/missed fields (`phase3a_observability.py:85-184,412-572`). | `IMPLEMENTED` as sampling; not an evidence-state machine. |
| Normal feedback | `Phase3DClosedLoopHarness` observes a completed window then calls `_control()` (`phase3d_control.py:776-914,974-1170`); C1 repeats this in `stagec1_stationary.py:353-443`. | Fixed-window/previous-window trigger, not minimum-evidence plus maximum-timeout epoch. |
| Evidence minimum | Scheduler has `min_samples`/`min_tail_samples` config (`feedback_scheduler.py:65-90`), but C1 sets `min_samples=1` and control still runs per window (`stagec1_stationary.py:90-140`). | No phase-specific input/output-token evidence gate. |
| Maximum timeout | No evidence-epoch timeout independent of window/control loop was found. | `NOT_IMPLEMENTED`; value `TBD_BY_MEASUREMENT`. |
| P/D evidence | Prefill uses TTFT; decode uses TBT or TPOT (`feedback_scheduler.py:435-565`); endpoint sample carries both only when available (`telemetry.py:29-100`). | No explicit input-token/request coverage vs output-token/completion coverage requirement. |
| Safety action timing | Severe pressure and latency-risk can recommend `RECOVER_HIGH` (`feedback_scheduler.py:301-433`), but it is evaluated at the next control iteration. | `PARTIALLY_IMPLEMENTED`; design requires event-driven immediate safety path. |
| Energy/reporting windows | `phase3b_energy.py:285-457` intentionally keeps wall-bounded windows for accounting. | Compatible with design; windows must not be mistaken for scheduling boundaries. |

## 5. Current C1 controller

The C1 implementation is `paper/scripts/xpyd/stagec1_stationary.py` and uses
the general scheduler in `paper/scripts/xpyd/feedback_scheduler.py`.

### Frequencies and state

- Validation frequencies are selected as high/safe-high, nearest 75% (mid),
  and nearest 50% (low) from supported clocks
  (`phase3d_control.py:118-139`). The accepted C1 levels are **P L40S:
  1260/1890/2520 MHz** and **D L4: 750/1125/1500 MHz**.
- Each policy block starts every endpoint at HIGH and restores HIGH at the end
  (`stagec1_stationary.py:353-443`).
- `POOLS` explicitly groups P0/P1 and D0/D1 (`stagec1_stationary.py:42-52`).
  The pool coordinator only permits a downshift when all endpoint recommendations
  agree, while any risk/upshift forces the pool to HIGH (`stagec1_stationary.py:143-178,270-305`).
- The physical actuator is per endpoint and readback-aware
  (`phase3d_control.py:272-370`); the C1 policy is intentionally shared-pool,
  not final per-endpoint DVFS.

### Conditions and hard-coded/configured values

| Condition/state | Source and current value | Provenance |
|---|---|---|
| TTFT/TPOT SLO | C1 config override 600/65 ms, 95% attainment (`stagec1_stationary.py:90-140`; design `XPYD_SYSTEM_CONTROL_DESIGN.md:424-479`). | `FROZEN` C1 contract / `CALIBRATED` from C0.5. |
| EWMA | `EWMA alpha=.5` in the C1 runtime config (`stagec1_stationary.py:90-140`); endpoint-local update in `telemetry.py:103-133,206-359`. | `IMPLEMENTATION_DEFAULT`/experiment setting, not final architecture. |
| Up-risk | `up_fraction=.8`; latency risk is enabled in C1; queue/KV pressure and admission risk can force HIGH (`stagec1_stationary.py:90-140,181-187`; `feedback_scheduler.py:301-433`). | `IMPLEMENTATION_DEFAULT`/C1 setting. |
| Downshift | `down_fraction=.5`, low queue `0`, low KV `.5`; one supported-clock step is returned (`feedback_scheduler.py:301-433`; pool agreement in `stagec1_stationary.py:143-178`). | `IMPLEMENTATION_DEFAULT`/C1 setting. |
| Queue/KV severe limits | C1 sets max queue `1`, severe queue `4`, max KV `.9`, severe KV `.97`. | `IMPLEMENTATION_DEFAULT`/C1 setting. |
| Freshness | C1 telemetry max age `300 s`; general scheduler default is `30 s` (`stagec1_stationary.py:90-140`; `feedback_scheduler.py:65-90`). | Configuration, not measured architecture constant. |
| Cooldown/dwell | C1 dwell `1 s`; scheduler default dwell `0` (`stagec1_stationary.py:90-140`; `feedback_scheduler.py:65-90`). | `IMPLEMENTATION_DEFAULT`/C1 setting. |
| Minimum samples | C1 `min_samples=1`; general default also `1`, with `min_tail_samples=20` available but not an evidence epoch (`stagec1_stationary.py:90-140`; `feedback_scheduler.py:65-90`). | `IMPLEMENTATION_DEFAULT`; insufficient for the intended evidence contract. |
| Fallback | Safe compatible pair/high-frequency fallback is used for initial/stale/invalid state (`phase3d_control.py:776-914`; `feedback_scheduler.py:583-619`). | `IMPLEMENTED`, conservative safety behavior. |
| Readback/logging | Actuator validates registry frequency and records requested/readback/action rows (`phase3d_control.py:272-370`; `stagec1_stationary.py:218-305`). | `IMPLEMENTED`. |

The controller therefore has a fast-up *recommendation* and conservative
one-step downshift, but no independent event loop: the recommendation can only
be acted on when the next window/control iteration executes. This is the key
granularity mismatch with the authoritative design.

## 6. Per-endpoint state

`EndpointSpec` is immutable placement/role/TP metadata and `EndpointState` is
mutable lifecycle/health/frequency/load state
(`paper/scripts/xpyd/types.py:20-108`; `registry.py:17-97`).

| State field/category | Present source | Missing for the final design |
|---|---|---|
| Endpoint ID, role, node, GPU IDs, TP degree | `EndpointSpec` (`types.py:20-66`) | TP-diverse inventory is not deployed. |
| Lifecycle/health/active | `EndpointState`, registry filters (`types.py:67-108`; `registry.py:17-97`) | No independent health confidence/history. |
| Requested/current/observed frequency | Registry state plus actuator/action rows (`phase3d_control.py:272-370`) | State is not a complete request-vs-readback history. |
| Queue/running/KV and observation timestamp | `EndpointState` and telemetry snapshots (`types.py:67-108`; `telemetry.py:139-165,296-359`) | Short-work blind spots remain. |
| Latency EWMA/window tails | `TelemetryAggregator` (`telemetry.py:206-359`) | Endpoint-local only; no context/system-state partition. |
| Energy and token rates | Window analysis (`phase3b_energy.py:285-457`; `phase3c_substrate.py:588-737`) | Not part of persistent registry state; request attribution is intentionally absent. |
| Search/probing state | `route_probing.py:75-333` only | Not in C1 endpoint state; no frequency headroom/coarse-to-fine state. |
| Context/load/phase/burst state | None in `types.py`/`registry.py` | `NOT_IMPLEMENTED`; design says future system state is required. |

### TP startup property

`run_disagg_benchmark.sh:846-953` launches each server with explicit GPU IDs and
`--tensor-parallel-size`; `EndpointSpec` enforces `len(gpu_ids)==tp_degree`
(`types.py:20-66`). TP is therefore a **startup/inventory property**, not a
runtime action. The validated C1/C0 setup is TP=1; TP2 is not represented as a
runtime-switchable option and is intentionally not implemented here.

## 7. Topology and TP

The validated topology is P0/P1 L40S prefill and D0/D1 L4 decode, all TP=1,
with four explicit compatible logical pairs. Config loading requires at least
two P and two D endpoints and validates explicit pair evidence
(`phase3c_substrate.py:102-182`). The launcher maps one physical GPU to each
server and uses the first IDs for a TP group (`run_disagg_benchmark.sh:846-953`).

| Design question | Current answer | Status |
|---|---|---|
| Are all four P→D links represented? | Yes, explicit compatibility table and one pre-instantiated core per pair (`compatibility.py:53-127`; `disagg_proxy.py:666-760`). | `IMPLEMENTED` for TP1 substrate. |
| Can an endpoint be activated/deactivated at runtime? | Registry health/lifecycle can fail closed, but no topology mutation is used in C1. | `PARTIALLY_IMPLEMENTED`. |
| Is TP runtime-configurable? | No; process startup fixes TP. | `IMPLEMENTED` constraint; TP-diverse runtime inventory is missing. |
| Is TP2 represented? | Only generic type/config validation; no deployed TP2 C1 path. | `NOT_IMPLEMENTED` by design for this stage. |
| Does energy include idle GPUs? | System energy definition includes all four GPUs; window analysis sums selected/all-GPU energy as configured (`phase3b_energy.py:477-504`; design `XPYD_SYSTEM_CONTROL_DESIGN.md:856-925`). | `IMPLEMENTED` accounting principle; not request attribution. |
| Does one DVFS setting cover a TP group? | Actuator addresses each physical endpoint; final per-TP-group policy is not used in C1. | `PARTIALLY_IMPLEMENTED`. |

## 8. Routing

`disagg_proxy.py:576-760` is pair-aware and fail-closed for explicit
compatibility. Baseline C1 uses four-pair round robin. The general scheduler can
enumerate P×D candidates and rank recent endpoint EWMA energy
(`feedback_scheduler.py:195-299,567-581`), while Phase 4B.1 adds a separate
context route-cost store and safe prober (`route_probing.py:75-333`).

| Routing dimension | Current C1 behavior | Gap |
|---|---|---|
| Per logical request | Yes; proxy chooses a pair for each ingress request. | `IMPLEMENTED`. |
| Joint P/D pair awareness | Explicit pair compatibility and pair cores; scheduler evaluates pair candidates. | `IMPLEMENTED` substrate; current RR does not optimize. |
| Candidate constraints | Health/lifecycle/compatibility/freshness and safe fallback. | `IMPLEMENTED` safety subset. |
| Context/class | Class is logged in trace; not passed to router. | `NOT_IMPLEMENTED` in C1. |
| Frequency state | Readback/action is logged; C1 pool state is not a route context. | `PARTIALLY_IMPLEMENTED`. |
| Load/pressure/latency | Used for DVFS safety, not C1 RR routing rank. | `NOT_IMPLEMENTED` for safety→pressure→energy routing hierarchy. |
| Energy objective | Phase 4B.1 can use total four-GPU window energy/request (`route_probing.py:1-6,217-333`); C1 RR does not. | `EXPERIMENT_ONLY`. |
| Exploration | Safe unseen/stale route probing exists in Phase 4B.1, with DVFS freeze. | `EXPERIMENT_ONLY`; not C1 and not a generic frequency search. |

## 9. Load and backlog observability

| Signal | Source/cadence | Trust and limitation |
|---|---|---|
| Offered arrivals/rate/class mix | `workload_harness.py:182-212`; trace schedule and segment summaries. | Planned load, not completed service. |
| Client waiting/admission delay | `replay_synthetic_trace.py:36-90,519-617`. | Required backlog signal; can be large while endpoint queue is zero. |
| Proxy ingress/inflight/active pair | `disagg_proxy.py:666-760`; diagnostics. | Exact request events, but not a queue-depth time series by itself. |
| Endpoint running/waiting/KV | `phase3a_observability.py:474-572`; `phase3c_substrate.py:536-586`. | Fixed-period samples; short P work can be missed. |
| Endpoint service/latency EWMAs | `telemetry.py:206-359`. | Endpoint-local and sample-dependent; no context partition. |
| Admission risk | `stagec1_stationary.py:181-187`: backlog growth, client waiting at close, or admission p95 >250 ms. | Useful C1 safety proxy; not a complete queue model. |
| Scrape health/drift | `phase3a_observability.py:877-1004,1307-1434`. | Auxiliary if configured so; required metrics still gate validity. |
| Freshness | Scheduler compares observation age (`feedback_scheduler.py:301-433`). | Stale data causes fallback at control epoch; no event-triggered independent path. |

**Interpretation rule:** endpoint `queue==0` must not be reported as “no
backlog”; inspect client waiting/admission and proxy inflight as well. Likewise,
sampled P `running==0` is not proof that prefill is idle. This rule is explicit
in `phase3a_observability.py:534-572`, `stagec1_stationary.py:603-610`, and the
authoritative design (`XPYD_SYSTEM_CONTROL_DESIGN.md:534-574`).

## 10. Energy accounting

| Item | Current implementation | Gap/qualification |
|---|---|---|
| Sampling | `FixedPeriodEnergySampler` performs endpoint-local fixed-period NVML queries and records actual start/finish, latency, drift, errors and missed slots (`phase3b_energy.py:139-274`). | Fixed-period measurement, not a control-boundary requirement. |
| Integration | Counter delta `(last-first)/1000` when valid; otherwise trapezoidal power integral (`phase3b_energy.py:285-457`). | Existing reset/boundary/coverage semantics retained. |
| Bounds | Wall/monotonic window boundaries, boundary gap and coverage checks (`phase3b_energy.py:285-457`). | Invalid windows are rejected, not silently repaired. |
| Scope | Per GPU samples roll up to endpoint, P/D pool and all-GPU totals (`phase3b_energy.py:477-504`; `phase3c_substrate.py:588-737`). | Endpoint/window gross energy is available. |
| Idle/background | Optional idle power subtraction only when comparable valid idle data exists (`phase3b_energy.py:460-474`). | Gross energy remains the primary C1 metric; no implicit subtraction. |
| J/request and J/output-token | Window energy divided by logical request/token totals; request token audit drives expected P/D deltas (`phase3b_energy.py:477-504,618-658`). | Amortized metric, not physical attribution. |
| Request-level attribution | Explicitly out of scope; P/D request energy is not inferred from a single request's NVML samples (`phase3b_energy.py:477-504`; design `XPYD_SYSTEM_CONTROL_DESIGN.md:576-624`). | Must not be claimed as per-request physical energy. |
| KV transfer energy | Not included in scheduler pair energy (`feedback_scheduler.py:567-581`). | Gap/TBD for a future physical accounting extension. |

## 11. SLO contract

| Location | Threshold/meaning | Status |
|---|---|---|
| Current C1 config | TTFT <=600 ms, TPOT <=65 ms, at least 95% request attainment (`stagec1_stationary.py:90-140`). | Current frozen C1 contract. |
| C0.5/design | Same 600/65 contract and 35/15/35/15 SS/SL/LS/LL mix (`XPYD_SYSTEM_CONTROL_DESIGN.md:424-479`). | Calibrated contract. |
| General scheduler defaults | `FeedbackSchedulerConfig` defaults TTFT/TPOT fields plus `safety_fraction=.8`, `down_fraction=.5`, `up_fraction=.8` (`feedback_scheduler.py:65-90`). | Implementation defaults; callers must override SLO. |
| C1 latency action | Scheduler uses EWMA TTFT for prefill and TBT or TPOT for decode; risk is compared to `up_fraction*SLO`, slack to `down_fraction*SLO` (`feedback_scheduler.py:301-565`). | Relative control thresholds; not separate SLOs. |
| Historical/stale values | Earlier Phase 4C material used 550/65 or 1000/80 in older contexts; design marks 550 as stale (`XPYD_SYSTEM_CONTROL_DESIGN.md:424-479`; C1 overrides in `stagec1_stationary.py:90-140`). | Do not mix with current C1 analysis. |
| Absolute vs relative | Evaluation attainment is absolute 600/65; controller risk/headroom is relative fractions of those configured values. | Agreement is partial: same base SLO, different control predicates. |
| Queue/SLO interaction | Queue/admission pressure can force HIGH, but is not itself an SLO pass/fail metric (`stagec1_stationary.py:181-187`; `feedback_scheduler.py:301-433`). | Preserve both metrics separately. |

The audit does not modify any threshold. A future implementation must pass the
experiment's SLO contract into the controller and document the relationship
between absolute attainment and relative risk/headroom.

## 12. Harnesses and outputs (Stages B/C0/C1)

| Harness/stage | Main path and outputs | Current gap |
|---|---|---|
| Stage B | `phase3c_substrate.py:381-447` runs persistent 2P2D substrate, fixed-period telemetry and analysis. Outputs include `summary.json`, `audit.json`, `requests.csv`, `routes.csv`, `endpoint_summary.csv`, `energy_summary.csv`, `derived/metrics_delta.json`, `derived/scrape_schedule.csv`, `derived/endpoint_load.csv`, plus raw proxy/server logs (`phase3c_substrate.py:738-820`). | Validated substrate, not controller experiment. |
| Stage C0 | Same substrate with fixed HIGH, RR, concurrency=4 and load sweep. Accepted experiment configuration/results are documented in `XPYD_SYSTEM_CONTROL_DESIGN.md:1338-1455`. | Load classes are empirical points; no general normalized load model. |
| Stage C0.5 | Offline analysis of C0 class-specific TTFT and threshold sensitivity; documented in design/status sections. | Does not add runtime context state. |
| Stage C1 smoke/stationary | `stagec1_stationary.py:307-443` uses STATIC-HIGH or FEEDBACK-DVFS, shared pools, matched traces. | Full C1 comparison has not been completed in the current status. |
| C1 outputs | `stagec1_stationary.py:444-611`: `stagec1_measurements.csv`, `stagec1_requests.csv`, `controller_decisions.csv`, `clock_actions.csv`, `telemetry_observations.csv`, repeat/policy summaries, `stagec1_summary.json/md`. | Controller log is window-oriented; no evidence-epoch/event log schema. |
| Headroom diagnosis | `stagec1_headroom` path and `XPYD_STAGEC1_HEADROOM_DIAGNOSIS.md` define small fixed-frequency points. | Current corrected job is pending; no result may be inferred from source alone. |

## 13. Parameter provenance

The labels below distinguish a value's origin; they are not recommendations.

| Parameter/semantic | Current value/source | Provenance |
|---|---|---|
| 2P2D TP1 topology and four compatible pairs | `phase3c_substrate.py:102-182`; validated C0/C1 config. | `FROZEN` for current experiment. |
| Request buckets (input <=512, output <=256) | `workload_harness.py:19-40,50-68`. | `CALIBRATED` workload definition. |
| C1 mix (35% SS, 15% SL, 35% LS, 15% LL) | `stagec1_stationary.py:90-140`; design status. | `FROZEN` C1 experiment. |
| C0 load points .05/.10/.15/.20 and overload .30 | Design status `XPYD_SYSTEM_CONTROL_DESIGN.md:424-479`. | `CALIBRATED` empirical points. |
| Current C1 SLO 600/65, >=95% | `stagec1_stationary.py:90-140`. | `FROZEN` C1 contract. |
| Safe HIGH P=2520, D=1500 | Design status and launcher (`XPYD_SYSTEM_CONTROL_DESIGN.md:424-479`; `run_disagg_benchmark.sh:1515-1601`). | `CALIBRATED` hardware operating point. |
| LOW/MID/HIGH P=1260/1890/2520, D=750/1125/1500 | `phase3d_control.py:118-139`; C1 config. | `CALIBRATED` tested levels. |
| EWMA alpha=.5, safety/up/down fractions, queue/KV limits, dwell/freshness | `stagec1_stationary.py:90-140`. | `IMPLEMENTATION_DEFAULT`/C1 experiment settings. |
| Generic scheduler defaults 1000/80, safety .8, min samples 1, tail 20, max age 30, queue 8, KV .90, dwell 0 | `feedback_scheduler.py:65-90`. | `IMPLEMENTATION_DEFAULT`; not the current C1 contract unless not overridden. |
| Control-window duration | Configurable `control_window_s` (`workload_harness.py:50-68`); C1 window loop (`stagec1_stationary.py:307-443`). | `IMPLEMENTATION_DEFAULT`/experiment setting; exact evidence epoch not defined. |
| Minimum evidence per epoch | No phase-specific token/request requirement. | `TBD_BY_MEASUREMENT`. |
| Maximum evidence timeout | No implementation. | `TBD_BY_MEASUREMENT`. |
| Load boundaries beyond measured C0 points | No normalized classifier. | `TBD_BY_MEASUREMENT`. |
| Phase-balance/rho normalization and boundaries | No implementation. | `TBD_BY_MEASUREMENT`. |
| BURST/RISK thresholds and event debounce | No named system-state contract. | `TBD_BY_MEASUREMENT`. |
| Coarse-to-fine safe exploration schedule | Route probing has minimum interval/age but no frequency headroom search in C1 (`route_probing.py:217-333`). | `TBD_BY_MEASUREMENT`. |
| Final per-endpoint/per-TP-group policy | C1 intentionally uses shared pools. | `TBD_BY_MEASUREMENT`. |

## 14. Seven agreed decisions

| # | Decision | Status | Evidence and exact gap |
|---:|---|---|---|
| 1 | Evidence epochs + event safety | `PARTIALLY_IMPLEMENTED` | Fixed-period telemetry and window control exist (`phase3a_observability.py:85-184`; `phase3d_control.py:974-1170`); no minimum evidence + maximum timeout epoch or independent event-driven safety loop. |
| 2 | Two-level context: request SS/SL/LS/LL plus system load/phase/BURST/RISK | `PARTIALLY_IMPLEMENTED` | Request labels and load traces exist (`workload_harness.py:19-40,148-212`); system context state is absent from scheduler/registry. |
| 3 | Adaptive safe exploration, coarse-to-fine, one-step baseline | `EXPERIMENT_ONLY` | Phase 4B.1 safe route probing and freeze-DVFS path exists (`route_probing.py:75-333`); no C1 frequency search/headroom state. |
| 4 | Final per-endpoint/per-TP-group DVFS; shared pool only isolation | `PARTIALLY_IMPLEMENTED` | Physical endpoint actuator/readback exists (`phase3d_control.py:272-370`); C1 coordinates P/D shared pools (`stagec1_stationary.py:143-178`). |
| 5 | Fixed pre-instantiated TP-diverse inventory; TP not runtime; idle all GPUs count | `PARTIALLY_IMPLEMENTED` | Fixed TP1 2P2D inventory and all-GPU accounting are present; TP-diverse inventory is not deployed and TP2 is not represented. |
| 6 | Safety-first load-aware pair routing: safety->pressure->energy->exploration | `PARTIALLY_IMPLEMENTED` | Compatibility/fail-closed and route probing exist; current C1 RR lacks context/pressure/energy hierarchy. |
| 7 | Knowledge-cost spectrum STATIC->Feedback->Adaptive XPYD->Canary->Empirical Oracle | `EXPERIMENT_ONLY` | STATIC/feedback and Phase 4A/4B artifacts exist in separate harnesses; no unified knowledge-cost evaluation or canary contract. |

## 15. Current project/stage status

| Stage | Repository/result evidence | Status |
|---|---|---|
| Stage B realistic workload/substrate harness | `phase3c_substrate.py` and design status (`XPYD_SYSTEM_CONTROL_DESIGN.md:1338-1395`). | `COMPLETE`/validated. |
| Stage C0 capacity/load calibration | Accepted job 257127 and documented rates/backlog classification (`XPYD_SYSTEM_CONTROL_DESIGN.md:1395-1455`). | `COMPLETE` for measured points. |
| Stage C0.5 SLO contract calibration | Offline class/threshold analysis documented in design status (`XPYD_SYSTEM_CONTROL_DESIGN.md:1455-1490`). | `COMPLETE` for current 600/65 contract. |
| Stage C1a smoke, shared-pool stationary feedback-DVFS | Job 259228 results/status (`XPYD_SYSTEM_CONTROL_DESIGN.md:1490-1535`). | `COMPLETE` smoke; no claim of full C1 success. |
| Stage C1b fixed-frequency headroom diagnosis | Design status marks it current/running (`XPYD_SYSTEM_CONTROL_DESIGN.md:1535-1625`). Previous job 259405 failed due to an ordering issue; corrected source was submitted as job 261064. | `CURRENT` / waiting for resources/result; do not infer headroom from it yet. |
| Full stationary C1 comparison | No accepted full result in the current repository status. | `NOT_COMPLETE`. |
| C1c/C2 event-epoch, context-aware, final per-endpoint DVFS | No implementation/result. | `NOT_IMPLEMENTED`/future. |

As of the read-only Minerva check on 2026-09-13, Slurm job **261064** was
`PENDING (Priority)` with no completed result. No GPU workload was started for
this audit.

## 16. Gap priority before the next GPU run

### P0 — resolve before changing C1 semantics

- Confirm the accepted C1b headroom result and preserve the current 600/65,
  >=95% contract (`XPYD_SYSTEM_CONTROL_DESIGN.md:1535-1625`).
- Define the evidence-epoch contract: minimum phase-specific request/input or
  output-token coverage and a maximum timeout. These are currently
  `TBD_BY_MEASUREMENT`, not values to invent in code.
- Separate event-driven safety from normal window/evidence control and log both
  the trigger and action time. The current fallback/up recommendation is only
  evaluated at a control iteration (`phase3d_control.py:776-914`).

### P1 — before C1c/C2 or any claim of context-aware control

- Add explicit system-context state (load magnitude, phase balance and
  BURST/RISK) with measured boundaries; retain client/proxy backlog signals.
- Decide whether the next experiment remains shared-pool or moves to the
  intended per-endpoint/per-TP-group state, and make request-vs-readback state
  explicit in logs.
- Specify the safety-first pair-routing hierarchy and context-partitioned route
  observations; do not reuse Phase 4B.1's later experiment as if it were C1.
- Add evidence/event fields to `controller_decisions.csv` without changing
  telemetry metric semantics.

### P2 — later architecture/research extensions

- Deploy a fixed TP-diverse inventory (TP2 remains a startup property),
  evaluate idle-GPU accounting with matched inventories, and define
  cross-platform transferability.
- Unify STATIC, feedback, adaptive XPYD, canary and empirical-oracle knowledge
  costs; keep oracle information offline.
- Investigate independent KV-transfer/network energy attribution only if a
  future question requires it; it is outside the current window-amortized
  metric contract.

## Conclusion

The validated substrate is real 2P2D with exact request IDs, route/SSE audits,
fixed-period endpoint telemetry, energy-window validity checks, and hardware
clock readback. The main architectural gap is not missing basic observability;
it is control granularity and state: C1 currently reacts once per completed
window using endpoint-local observations and shared P/D pool actions, whereas
the design calls for evidence-driven epochs, explicit system context, and
event-driven safety recovery. The repository therefore supports planning the
next controlled change, but it does not justify silently treating those future
semantics as already implemented.
