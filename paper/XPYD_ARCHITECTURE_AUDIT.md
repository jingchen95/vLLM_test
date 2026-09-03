# XPYD Sections 1–14 Architecture Audit

Date: 2026-09-03

Authority: [`XPYD_SYSTEM_CONTROL_DESIGN.md`](XPYD_SYSTEM_CONTROL_DESIGN.md)

Scope: read-only audit of the current endpoint, proxy, workload-client,
telemetry, and controller semantics, followed by a CPU-only workload-harness
redesign. No GPU job was launched and no routing, probing, DVFS, or predictive
policy was added.

## Executive finding

The physical 2P2D substrate, exact request identity, SSE path, compatibility
checks, endpoint telemetry, energy collection, and DVFS actuation remain sound.
The current experiment/control layer does **not yet implement the design
contract's intended mixed-request, request-context-aware control problem**.

The dominant mismatch is not a missing energy model. It is the workload and
load-observation boundary:

- prior experiments normally set `max_concurrency=1`;
- requests waiting at the client semaphore were excluded from TTFT and invisible
  to the controller;
- old windows were homogeneous workload blocks;
- the proxy selected a route per request, but it did not receive the request
  class as an input to routing;
- endpoint queue alone could therefore remain zero while offered demand waited
  elsewhere.

The new harness foundation makes these limitations observable before any new
controller mechanism is considered.

## Section-by-section audit

| Design section | Status after this change | Evidence and remaining mismatch |
|---|---|---|
| 1. System model | **PASS / transport detail open** | `EndpointSpec`, registry state, immutable TP, four persistent processes, and exact pair compatibility match the contract. All four logical pairs are configured. The internal NCCL connection creation/reuse lifetime is still a vLLM implementation detail and has not been instrumented under concurrent cross-pair traffic. |
| 2. Control granularity | **PARTIAL** | The proxy calls its selector for every request; energy and normal DVFS decisions are window-level. The emergency path is still triggered by a completed control-window observation, not by live arrival/backlog events. |
| 3. Cold start | **PASS for existing controlled runs** | Launchers initialize all four endpoints at validated HIGH and the no-control proxy can round-robin explicit compatible pairs. Future mixed runs must preserve this initialization. |
| 4. Request path | **PASS** | Client ID → proxy selection → P → KV handoff → D → real SSE is implemented and audited with exact `request_id + P_i + D_j`. |
| 5. Request classification | **PARTIAL, harness foundation added** | Explicit SS/SL/LS/LL thresholds and labeled mixed traces now exist in `workload_harness.py`. The current online selector still does not consume that context. |
| 6. Request measurements | **PARTIAL** | IDs, token lengths, selected endpoints, TTFT/TPOT/ITL/E2E, completion, and SLO outcomes exist. New rows retain workload class, segment, scheduled arrival, client admission, and admission delay. Exact per-request observed GPU frequency at dispatch is still inferred from the enclosing control state rather than returned atomically by the endpoint. |
| 7. Window aggregation | **PARTIAL** | Existing runners aggregate latency, endpoint queue/KV, tokens, and energy, but previous workload windows were homogeneous. The new generator emits offered mixed-window summaries. A future Stage-B runner must join client events, proxy state, vLLM metrics, and energy on the same fixed windows. |
| 8. Energy | **PASS for measurement; controller semantics mixed** | Four-GPU gross window energy is valid and remains the evaluation objective. Phase 4B.1's context/route observation uses total four-GPU window energy per logical request. The older generic `FeedbackScheduler._pair_energy_per_request` still ranks using P+D endpoint EWMAs; it must not be represented as the final system objective. |
| 9. Feedback state | **PARTIAL** | Endpoint EWMAs, freshness, latency, queue/KV, energy, and context-specific probe costs exist in separate components. There is no unified state containing workload mix, client admission pressure, proxy inflight, and recent route distribution. |
| 10. Applying feedback | **MISMATCH** | DVFS is endpoint/window-driven as intended. Routing-control files select one or more pairs, but prior homogeneous windows did not require the selector to differentiate simultaneous request classes. No new heuristic was added in this audit. |
| 11. Two timescales | **PARTIAL** | Phase 4C.1 implements slow step-down and direct recovery to HIGH from fresh latency risk. It is still window-cadence recovery and cannot react to offered-arrival/backlog risk before a latency sample completes. |
| 12. Workload transitions | **MISMATCH in old experiments; new traces ready** | Old dynamic traces transition between homogeneous blocks. The new `dynamic_mixed_burst` trace changes both class mixture and rate while retaining mixed populations within segments. It is not yet physically validated. |
| 13. Routing semantics | **MISMATCH** | Selection happens per request, and all four pairs are usable without restarting servers. However `selector.choose()` currently takes no request context or live-load argument; routing is round-robin or controller-file selection. The intended `(context, P, D, f_P, f_D)` decision is therefore not implemented. |
| 14. Unknown-state safety | **PASS for endpoint telemetry; context incomplete** | Compatibility, telemetry freshness, health, queue-observed flags, and conservative fallback fail closed. Missing queue is not silently treated as zero. Unseen mixed contexts are not yet represented in the online controller and therefore cannot yet be audited end to end. |

## Exact transport and request-admission audit

The launcher starts four independent vLLM endpoints:

- P0 producer: GPU 0, HTTP 8100, KV 14579;
- P1 producer: GPU 1, HTTP 8101, KV 14580;
- D0 consumer: GPU 0, HTTP 8200, KV 14579;
- D1 consumer: GPU 1, HTTP 8201, KV 14580.

For every request, the proxy chooses one explicit compatible pair and embeds
that pair's producer and consumer addresses in the vLLM request-ID transport
envelope. Consequently P0→D1 and P1→D0 do not require a server restart. This
establishes logical runtime availability, but not whether vLLM eagerly creates
or lazily creates/reuses every internal NCCL connection.

The old client created an async task at each trace arrival, then waited on an
`asyncio.Semaphore`. TTFT began inside `consume_sse_completion`, after semaphore
admission. Thus:

```text
offered arrival → client semaphore wait → HTTP send → first token
```

only the final `HTTP send → first token` interval was TTFT. That definition is
still retained, but the omitted interval is now explicitly recorded as
`client_admission_delay_ms`.

## Workload-harness redesign

### Explicit contexts and mixed traces

`xpyd.workload_harness` defines fixed experiment buckets:

```text
short input  <= 512 tokens
short output <= 256 tokens

SS = 128 / 128
SL = 128 / 512
LS = 2048 / 128
LL = 2048 / 512
```

The checked-in Stage-B config provides one stationary mixed scenario and one
dynamic mixed burst/recovery scenario. It deterministically interleaves classes
so route order is not trivially confounded with one request shape.

### Load visibility

The replay client now records:

- scheduled trace arrival;
- event-loop dispatch lag;
- semaphore admission time and delay;
- client waiting and admitted-inflight events;
- maximum client waiting and inflight counts;
- request class and segment.

The multi-endpoint proxy now exposes read-only `/xpyd/state` counters:

- accepted/completed/failed totals;
- current and maximum proxy inflight;
- unassigned requests;
- active requests by exact pair and endpoint.

System pressure must be reported as a **vector**, not naively summed, because a
single request may appear at multiple stages:

```text
client admission waiting
proxy admitted/inflight
vLLM P/D waiting and running
```

This prevents both the old invisibility problem and double-counting claims.

### Unchanged semantics

- TTFT remains HTTP-send to first real decode token.
- User/offered-load delay can additionally report admission delay + TTFT.
- Proxy routing, connector behavior, controller decisions, and GPU state are
  unchanged.
- No request-level energy attribution is introduced.

## Next experiment matrix

The matrix is encoded in
`configs/xpyd_stageb_workload_harness.json`. It is intentionally a substrate
validation sequence, not a policy comparison:

1. **B0 CPU contract:** validate buckets, mixes, arrival ordering, load events,
   proxy counters, and window summaries.
2. **B1 concurrency-2 smoke:** all endpoints HIGH, four-pair round-robin, no
   controller. Validate connector correctness, exact IDs/tokens/routes/SSE,
   observable admission/inflight/endpoint pressure, and energy-window validity.
3. **B2 concurrency-4 smoke:** run only if B1 passes the same hard gates.
4. **B3 dynamic mixed harness smoke:** run only if B2 passes; still fixed HIGH,
   round-robin, and no controller.

The prior physical connector acceptance limit was concurrency one. Therefore
concurrency two is a deliberate new substrate-validation boundary, not an
assumed supported setting. No long policy evaluation should begin until B1–B3
establish correct concurrent transport and measurement behavior.

## Decision

Architecture audit complete. The code is **READY FOR CPU VALIDATION** and the
small Stage-B physical validation sequence is defined. It is **not yet ready for
a new controller comparison**: concurrent connector behavior and joined
mixed-window load telemetry must first pass B1–B3.
