# XPYD System & Control Design

## 0. Purpose

This document is the authoritative system/control specification for the XPYD project.

All future implementation and experiment design should follow these semantics unless this document is explicitly revised first.

The project goal is:

> **Energy-efficient heterogeneous prefill/decode (P/D) LLM serving using lightweight runtime feedback, with request-level routing and per-endpoint DVFS, while satisfying TTFT/TPOT SLOs.**

The key research question is:

> **How much offline knowledge is actually necessary for energy-efficient heterogeneous P/D control? Can lightweight runtime feedback capture most of the available energy benefit without exhaustive profiling or predictive latency/power models?**

---

# 1. System Model

## 1.1 Heterogeneous P/D pools

XPYD uses phase-level GPU heterogeneity:

- **Prefill pool:** GPU type A
- **Decode pool:** GPU type B

Current testbed example:

- `P0`, `P1`: L40S prefill endpoints
- `D0`, `D1`: L4 decode endpoints

We do **not** require mixed GPU types inside the same pool.

## 1.2 Persistent endpoints

Each P/D endpoint is persistent once launched and has:

- endpoint ID
- role: Prefill or Decode
- node / physical GPU ID(s)
- GPU type
- startup TP
- requested and observed frequency
- health/lifecycle state
- runtime telemetry

TP is a startup property, not a fast runtime control knob.

## 1.3 P→D compatibility

A logical request is assigned to a compatible pair:

```text
P_i → D_j
```

The runtime has demonstrated that multiple P/D pairings can execute correctly.

Keep two concepts separate:

- **logical pair compatibility:** whether `P_i → D_j` works;
- **transport implementation:** whether the connector/network path is persistent, created at startup, or reused internally.

Future code audits must verify the exact transport behavior rather than infer it from logical routing semantics.

---

# 2. Core Control Granularity

XPYD uses **both request-level and window-level control**.

| Component | Granularity |
|---|---|
| Request classification | Request-level |
| P endpoint selection | Request-level |
| D endpoint selection | Request-level |
| TTFT measurement | Request-level |
| TPOT / ITL measurement | Request-level |
| Token accounting | Request-level |
| Endpoint power sampling | Continuous / sampled |
| Endpoint energy | Window-level |
| Workload summary | Window-level |
| Feedback-state update | Window-level |
| Normal energy-saving DVFS | Endpoint-level, usually window-driven |
| Emergency DVFS upshift | Fast / event-driven |
| Final energy objective | Window / trace-level total system energy |

**Core rule: routing is request-level; energy feedback and normal DVFS optimization are window-level.**

---

# 3. Cold Start

At startup there is no trustworthy historical energy/performance knowledge.

## 3.1 Initial frequency

All active endpoints start at a validated safe HIGH frequency:

```text
P0 = HIGH
P1 = HIGH
D0 = HIGH
D1 = HIGH
```

## 3.2 Initial routing

Cold-start routing uses a safe policy that does not require energy knowledge, e.g.:

- round-robin, or
- least-loaded among healthy compatible endpoints.

Cold-start priorities are:

1. preserve SLO safety;
2. spread work sufficiently to gather observations;
3. avoid pretending that an unseen route is known to be energy-optimal.

---

# 4. Request Path

For each request:

```text
Incoming request
      ↓
Request classification
      ↓
Observe current endpoint state
      ↓
Choose Prefill endpoint P_i
      ↓
Choose compatible Decode endpoint D_j
      ↓
Execute Prefill
      ↓
KV handoff
      ↓
Execute Decode
      ↓
Return streamed output
      ↓
Record request-level outcome
```

The final assignment must be auditable as:

```text
request_id + P_i + D_j
```

The P and D choices may be made together at admission or in two implementation stages; that is an implementation detail.

---

# 5. Request Classification

Do not build a giant exact `(IL, OL)` lookup table.

Use coarse request context, e.g.:

```text
SS = short input, short output
SL = short input, long output
LS = long input, short output
LL = long input, long output
```

A richer context may include:

- input-length bucket
- requested output-length bucket
- arrival-rate/load bucket

The bucket definitions must be explicit and fixed for an experiment.

This is context classification, not prediction.

---

# 6. Request-Level Measurements

Every request should record at least:

```text
request_id
arrival timestamp
input tokens
requested / actual output tokens
workload class
selected P endpoint
selected D endpoint
P/D frequency state at dispatch
TTFT
TPOT / ITL
E2E latency
completion status
SLO pass/fail
```

TTFT and TPOT are fundamentally request-level metrics.

---

# 7. Window-Level Aggregation

A control window aggregates recent request-level observations and may contain a **mixed workload**.

Example:

```text
Window W7:
  20% SS
  10% SL
  50% LS
  20% LL
```

Controlled characterization may intentionally use homogeneous windows, but the architecture must support mixed windows.

## 7.1 Latency summary

Aggregate:

- mean TTFT
- p95/p99 TTFT when sample count permits
- mean TPOT
- p95/p99 TPOT when sample count permits
- SLO violation count/rate

Also retain per-class latency when possible, e.g.:

```text
SS TTFT p95
LS TTFT p95
SL TPOT p95
LL TPOT p95
```

## 7.2 Load summary

A window should include:

- logical request count
- arrival rate
- in-flight request count
- scheduler-side backlog
- endpoint-visible queue
- workload-class fractions
- total input tokens
- total output tokens

**Endpoint queue alone is insufficient.**

Previous low-concurrency experiments showed that demand can wait before endpoint submission while endpoint queue remains zero. Future realistic experiments must expose system-level backlog/arrival pressure.

---

# 8. Energy Measurement and Attribution

## 8.1 Physical measurement

For each endpoint `e`, integrate sampled power over a window:

```text
E_e(window)
```

For 2P2D:

```text
E_total(window)
  = E_P0 + E_P1 + E_D0 + E_D1
```

This total system energy is the primary optimization/evaluation objective.

## 8.2 Per-endpoint feedback

For every endpoint/window, retain:

```text
gross energy
request count handled
input tokens handled
output tokens handled
frequency residency
busy/load indicators
```

Useful feedback statistics include:

```text
Prefill endpoint:
E_P / input tokens handled

Decode endpoint:
E_D / output tokens handled
```

and window-amortized:

```text
E_endpoint / requests handled
```

These are feedback statistics, **not exact physical per-request energy**.

## 8.3 Claim boundary

Do not claim:

```text
"Request R17 consumed exactly X joules"
```

from window-level NVML data because:

- requests can overlap;
- idle/background power is included;
- GPUs are shared across requests;
- sampling can be much coarser than short phases.

Use the phrase:

> **window-amortized endpoint/system energy cost**

## 8.4 Measurement resolution

Existing characterization established that short single-request prefill phases can be substantially shorter than the effective NVML power-update interval.

Therefore:

- exact request-level prefill energy is under-resolved;
- endpoint/window energy remains useful;
- the controller should not depend on exact per-request prefill joules.

---

# 9. Feedback State

Maintain compact rolling feedback rather than exhaustive offline tables.

Potential state:

```text
recent workload mix
arrival rate / backlog
per-class TTFT / TPOT statistics
per-endpoint recent energy efficiency
per-endpoint frequency
endpoint health / freshness
recent routing distribution
```

EWMA or similar smoothing is allowed:

```text
S_t = alpha*x_t + (1-alpha)*S_(t-1)
```

This is still model-free when it only smooths measured history.

---

# 10. Applying Feedback

Suppose a recent window shows:

```text
large TTFT headroom
large TPOT headroom
stable low load
```

The controller may conservatively lower some endpoint frequencies for the next interval.

Example:

```text
W0:
P0 HIGH
P1 HIGH
D0 HIGH
D1 HIGH

Next:
P0 HIGH
P1 MID
D0 MID
D1 LOW
```

Requests in the next interval are still routed individually.

Example:

```text
long-input request
→ choose a P endpoint whose current frequency/load is likely safe from recent evidence

short-input request
→ may use a lower-frequency P endpoint if recent measurements show sufficient headroom
```

This is the intended coupling:

```text
request-level routing
×
endpoint-level DVFS state
```

---

# 11. Two Control Timescales

DVFS **actuation latency** and DVFS **control timescale** are different.

A frequency transition may complete in milliseconds, but energy optimization should not necessarily make a fresh DVFS decision for every request.

## 11.1 Fast safety path

React quickly to risk such as:

- sudden arrival-rate increase
- scheduler-side backlog
- sudden shift toward long-input requests
- fresh TTFT risk/violation
- endpoint health change

Example:

```text
long-input burst detected
P LOW/MID → HIGH immediately
```

Principle:

> **Fast up.**

The fast path must not rely only on heavily smoothed EWMA or endpoint queue.

## 11.2 Slow efficiency path

After enough stable evidence:

```text
large SLO headroom
stable load
valid measurements
→ HIGH → MID
→ observe
→ possibly MID → LOW
```

Principle:

> **Slow down.**

This avoids oscillation and preserves interpretable measurements.

---

# 12. Workload Transitions

A realistic controller must handle:

```text
mostly short-input
      ↓
sudden long-input burst
      ↓
stable prefill-heavy period
      ↓
return to light workload
```

Correct behavior:

1. detect workload/backlog risk;
2. move risky phase(s) to a conservative safe frequency quickly;
3. keep routing requests individually;
4. gather fresh measurements;
5. gradually reduce frequency only after safety is re-established.

Unknown/stale contexts should prefer a safe state.

---

# 13. Routing Semantics

Routing remains important when endpoints differ in runtime state.

Example:

```text
P0 = HIGH
P1 = LOW
```

For a long-input request:

```text
P0 may be needed for TTFT safety.
```

For a short-input request:

```text
P1 may be sufficient and cheaper.
```

Similarly, D endpoints may differ in:

- frequency
- current load
- recent TPOT headroom
- health
- KV pressure

The meaningful decision is therefore:

```text
(request context,
 P_i,
 D_j,
 current f_Pi,
 current f_Dj)
```

not routing in isolation.

A same-frequency, low-concurrency experiment showing little route difference does **not** imply routing is unimportant under different endpoint frequency/load states.

---

# 14. Unknown-State Safety

For unseen or stale conditions:

```text
unknown context
→ safe HIGH frequency
→ simple safe routing
→ gather observations
→ optimize gradually
```

Missing telemetry must not silently become:

```text
queue = 0
KV = 0
large SLO headroom
```

Unknown remains unknown.

---

# 15. Objective

Primary objective:

> **Minimize total GPU energy subject to TTFT/TPOT SLOs.**

```text
E_total = E_P0 + E_P1 + E_D0 + E_D1
```

Useful reporting metrics include:

```text
J/request
J/input-token
J/output-token
J/total-token
```

A policy that saves energy by violating SLO is not successful.

---

# 16. Existing Evidence: What to Keep

## 16.1 Keep as infrastructure evidence

Do not discard:

- real P/D observability semantics
- real SSE/token/request accounting
- persistent 2P2D substrate
- compatibility checks
- per-endpoint NVML energy measurement
- per-GPU DVFS actuation
- frequency readback
- safety restoration
- feedback → decision → actuation → readback loop

These establish that the physical system works.

## 16.2 Keep as characterization evidence

Useful observations include:

- meaningful energy optimization opportunity exists
- best measured configuration changes with workload
- feedback DVFS can save energy
- prefill-heavy behavior is strongly TTFT-sensitive
- decode TPOT is often less sensitive to core-frequency reduction
- endpoint/window energy can be measured reliably
- single-request prefill energy is under-resolved
- passive feedback can save substantial energy in some stationary/easier regimes

## 16.3 Reinterpret carefully

Some later experiments used:

- very low concurrency
- client-side serialization/semaphores
- endpoint queues that did not expose waiting demand
- homogeneous workload windows
- SLOs close to the natural fixed-HIGH latency distribution

These remain useful for mechanism diagnosis but are **not** definitive evidence about production-like request-level routing or realistic queueing.

In particular:

> No P0-only routing penalty under `max_concurrency=1` does not imply route concentration is harmless under concurrent serving.

---

# 17. New Evaluation Principles

## 17.1 Realistic concurrency

Experiments must allow multiple requests in flight so that:

- endpoint/server-side load can build;
- backlog becomes observable;
- routing can affect queueing.

Avoid designs where excess demand waits entirely outside the serving system.

## 17.2 Mixed request populations

At least some experiments should mix classes in the same trace/window.

Example:

```text
30% SS
20% SL
35% LS
15% LL
```

## 17.3 Stationary + dynamic

### Stationary

The request distribution/load remains statistically stable for a period.

Purpose:

- steady-state energy/SLO behavior
- controller convergence

### Dynamic

Distribution/load changes over time.

Purpose:

- detection
- emergency response
- adaptation speed
- transition energy

## 17.4 Matched policy conditions

Comparisons must match:

- hardware
- endpoint layout
- TP
- SLO
- workload trace
- arrival process
- concurrency
- routing semantics
- energy accounting
- frequency action space

Do not compare an oracle and feedback policy collected under different experimental semantics.

---

# 18. Revised Experiment Matrix

## Stage A — Semantics audit

Before new long GPU experiments:

1. verify how current P/D transport paths are created;
2. verify whether all compatible `P_i→D_j` pairs are available at request runtime;
3. verify exact request-admission/routing granularity;
4. audit client-side semaphore/max-concurrency behavior;
5. identify where real scheduler-side backlog can be measured.

## Stage B — Realistic workload harness

Build/validate a harness with:

- concurrent in-flight requests
- visible scheduler/server backlog
- request-level SS/SL/LS/LL labels
- mixed request streams
- per-request route records
- per-window endpoint energy
- per-window workload summary

Infrastructure validation only.

## Stage C — Cold-start feedback DVFS

Start all endpoints HIGH and use simple safe routing.

Test whether feedback can:

- gather valid latency/energy observations
- reduce frequency conservatively under stable slack
- preserve SLO
- recover rapidly on workload change

Keep routing simple to isolate DVFS.

## Stage D — Request-level routing with heterogeneous frequency states

Create states such as:

```text
P0 HIGH
P1 LOW
D0 HIGH
D1 LOW
```

Use mixed request classes.

Test whether routing exploits current endpoint states:

```text
long-input → faster P when needed
short-input → lower-frequency P when safe
long-output → suitable D based on TPOT/load state
```

This directly tests the intended routing × DVFS interaction.

## Stage E — Dynamic mixed-workload evaluation

Use traces such as:

```text
low-load mixed traffic
→ prefill-heavy burst
→ decode-heavy period
→ recovery
```

Compare:

- STATIC safe baseline
- feedback DVFS + simple routing
- request-aware routing + feedback DVFS
- empirical/offline reference where fairly measurable

Measure:

- total energy
- SLO violations
- goodput
- adaptation latency
- frequency residency
- routing distribution
- backlog
- oscillation
- telemetry fallback

## Stage F — Offline-knowledge question

Only after the above is clean, ask:

> How close does lightweight feedback get to the best measured/reference policy?

If feedback is sufficient:

```text
stop adding model complexity
```

If a clear gap remains:

```text
identify missing information
→ add the smallest justified prior/model
```

Potential progression:

```text
feedback only
→ tiny empirical prior
→ lightweight contextual predictor
→ model-heavy only if necessary
```

---

# 19. Empirical Oracle

An empirical oracle is:

> **the lowest-energy SLO-valid configuration among configurations actually measured under the same experimental conditions.**

It is not a theoretical global optimum.

A valid oracle comparison must match:

- workload
- concurrency
- routing semantics
- endpoint layout
- frequency action space
- TP
- SLO
- energy definition

---

# 20. Research Positioning

Do not claim novelty in:

- P/D disaggregation itself
- heterogeneous P/D itself
- multi-P/multi-D routing itself
- DVFS itself
- model-free control itself

The intended contribution is the intersection:

> **energy/SLO control for heterogeneous phase-disaggregated LLM serving, combining request-level P/D routing with per-endpoint DVFS using lightweight runtime feedback while minimizing dependence on offline profiling/predictive models.**

The deeper scientific question is:

> **Which parts of the control problem truly require offline knowledge, and which can be handled sufficiently well through online feedback?**

---

# 21. Design Principles

1. **Request-level routing, window-level energy feedback.**
2. **Per-endpoint energy telemetry, system-level energy objective.**
3. **Fast up, slow down.**
4. **Unknown state is not safe state.**
5. **Do not infer exact request energy from coarse NVML windows.**
6. **Do not let client-side serialization hide server-side queueing.**
7. **Do not confuse DVFS actuation latency with control timescale.**
8. **Do not force every window into a single workload class.**
9. **Do not add predictors until a measured failure mode requires them.**
10. **Every future experiment must map directly to a research question.**

---

# 22. Immediate Next Action

Before launching more long GPU experiments:

1. treat this document as the current design contract;
2. audit current implementation against Sections 1–14;
3. identify mismatches between current code and intended semantics;
4. redesign the workload harness for realistic concurrency and mixed request streams;
5. only then define the next experiment matrix.

No new routing heuristic, probing mechanism, or predictive model should be added before this architecture audit is complete.
