Yes. Here is a detailed, Codex-ready description of the **SWEEP-LLM** idea.

You can paste this directly into Codex as a design spec or implementation prompt.

---

# SWEEP-LLM: Detailed Design Idea

## 1. Goal

SWEEP-LLM is a **joint runtime-and-cluster scheduler** for **energy-efficient disaggregated LLM inference on heterogeneous GPUs**.

The key idea is to adapt the spirit of **SWEEP** from heterogeneous CPU task scheduling to LLM serving:

* do **window-based control**
* use **offline characterization models**
* jointly optimize **multiple interacting knobs**
* use **different policies under different workload states**
* minimize energy while satisfying latency SLOs

In SWEEP-LLM, the scheduler controls a serving system with:

* a **prefill pool** on high-compute GPUs, e.g. **L40S**
* a **decode pool** on low-power / energy-efficient GPUs, e.g. **L4**
* optional **spillover** of decode requests onto L40S when L4 is saturated

The system targets two latency SLOs:

* **TTFT**: time to first token
* **TPOT**: time per output token

The optimization goal is:

[
\min \text{Energy}
\quad \text{s.t.} \quad
TTFT \le SLO_{TTFT},;
TPOT \le SLO_{TPOT}
]

---

## 2. Why SWEEP-LLM is needed

LLM inference has two very different phases:

* **Prefill**: compute-bound, high GPU utilization
* **Decode**: memory-bound, lower utilization, token-by-token generation

These phases behave differently on different GPUs.

Example intuition:

* **L40S** is better for prefill
* **L4** is often better for decode from an energy perspective
* but under heavy decode pressure, L4 alone may not be enough
* therefore, routing, tensor parallelism, and frequency must be adjusted **together**

This is the key insight of SWEEP-LLM:

> The best energy-efficient configuration cannot be found by tuning frequency alone or routing alone.
> It emerges from the interaction between **routing**, **parallelism**, and **frequency**.

---

## 3. System scope

SWEEP-LLM assumes a **disaggregated serving architecture**:

* **prefill runs on one GPU pool**
* **decode runs on another GPU pool**
* a **KV transfer** happens between pools when the request moves from prefill to decode

Typical deployment:

* **L40S pool**: preferred for prefill
* **L4 pool**: preferred for decode

But the mapping is **not rigid**:

* prefill is usually routed to L40S
* decode is usually routed to L4
* under decode-heavy pressure, some decode requests may spill to L40S

This makes SWEEP-LLM a **heterogeneity-aware, phase-aware scheduler**.

---

## 4. Main control knobs

SWEEP-LLM jointly optimizes the following knobs every scheduling window.

### 4.1 Routing knobs

These decide how request load is split across GPU pools.

#### `alpha`

Fraction of **prefill** load assigned to the L40S pool.

* `alpha = 1.0` means all prefill goes to L40S
* `alpha = 0.0` means all prefill goes to L4

In practice, first version can fix:

* `alpha = 1.0`

because L4 is usually poor for prefill.

#### `beta`

Fraction of **decode** load assigned to the L4 pool.

* `beta = 1.0` means all decode goes to L4
* `beta = 0.7` means 70% decode on L4, 30% decode spills to L40S
* `beta = 0.5` means half on each pool

This is the direct analogue of **task distribution percentage** in SWEEP.

---

### 4.2 Parallelism knobs

#### `TP_prefill`

Tensor parallelism degree for the prefill pool.

#### `TP_decode_L4`

Tensor parallelism degree for the decode pool on L4.

#### `TP_decode_L40S`

Optional tensor parallelism for decode spill running on L40S.

For a simpler first version:

* use one TP per pool
* or use one shared decode TP choice for both decode-capable pools

---

### 4.3 Frequency knobs

#### `freq_prefill`

GPU frequency for the prefill pool.

#### `freq_decode_L4`

GPU frequency for the L4 decode pool.

#### `freq_decode_L40S`

Optional GPU frequency for decode spill on L40S.

Different pools have different sweet spots:

* prefill tends to favor higher frequency
* decode often has a lower energy-optimal frequency

---

### 4.4 Pool size / active GPU knobs

#### `N_active_L40S`

How many L40S GPUs are active in the current window.

#### `N_active_L4`

How many L4 GPUs are active in the current window.

Under low load, SWEEP-LLM can:

* consolidate load
* power-gate idle GPUs
* reduce the cluster power floor

---

## 5. Scheduling window

SWEEP-LLM uses **window-based scheduling**, similar in spirit to SWEEP.

Example:

* one scheduling interval every **5 seconds**

At each window:

1. collect runtime statistics
2. classify current system state
3. build candidate configurations
4. predict TTFT, TPOT, power, and energy
5. choose the best safe configuration
6. apply it for the next window

This avoids expensive per-request global optimization.

---

## 6. State classification

Instead of only using QPS, SWEEP-LLM classifies each window based on **token pressure**.

Define:

[
D_{prefill} = \lambda \cdot \bar{L}_{in}
]

[
D_{decode} = \lambda \cdot \bar{L}_{out}
]

where:

* (\lambda) = request arrival rate
* (\bar{L}_{in}) = representative prompt length
* (\bar{L}_{out}) = representative output length

This produces four system states.

### 6.1 `PREFILL_HEAVY`

* prompt-token demand dominates
* prefill queue grows
* TTFT is at risk

Policy:

* prioritize prefill capacity
* prefer L40S for prefill
* increase prefill frequency if needed
* keep decode on L4 unless decode is near saturation

---

### 6.2 `DECODE_HEAVY`

* output-token demand dominates
* decode queue grows
* TPOT is at risk

Policy:

* prioritize decode capacity
* first scale up L4 decode resources
* if L4 saturates, decrease `beta` and spill some decode to L40S

This is a key SWEEP-LLM feature.

---

### 6.3 `BALANCED`

* neither phase dominates
* both TTFT and TPOT have slack

Policy:

* run full energy optimization
* consolidate GPUs aggressively if safe
* operate near pool-specific sweet spots

---

### 6.4 `BURST`

* sudden increase in arrival rate
* queue growth rate spikes
* risk of tail-latency explosion

Policy:

* temporary performance-first mode
* activate more GPUs
* raise frequency
* reduce aggressive consolidation
* possibly use L40S to help decode

---

## 7. Why state-aware control matters

The same QPS can produce very different system behavior.

Example:

* workload A: long prompt, short output -> prefill-heavy
* workload B: short prompt, long output -> decode-heavy

Same QPS, totally different optimal actions.

Therefore, SWEEP-LLM is **token-aware**, not merely request-aware.

---

## 8. Offline characterization and models

SWEEP-LLM is model-driven.

It relies on offline profiling data to build lightweight predictors.

### 8.1 Prefill model

For each relevant config, predict:

* TTFT or prefill latency
* prefill power

Inputs may include:

* GPU type
* frequency
* TP
* input length
* request rate / concurrency

---

### 8.2 Decode model

For each relevant config, predict:

* TPOT or decode throughput
* decode power

Inputs may include:

* GPU type
* frequency
* TP
* output length
* context length proxy
* request rate / concurrency

---

### 8.3 KV transfer model

When prefill and decode are on different pools, add a handoff cost:

[
T_{KV}
]

This may be modeled as:

* a constant placeholder
* or a function of prompt length / KV size

Example:

[
TTFT_{total} = TTFT_{prefill} + T_{KV}
]

The structural logic does not depend on having a perfect KV model; a placeholder is enough for the first prototype.

---

## 9. Optimization objective

At each scheduling window, SWEEP-LLM evaluates candidate configurations.

For each candidate (c), predict:

* (\widehat{TTFT}(c))
* (\widehat{TPOT}(c))
* (\widehat{Power}(c))

Then compute:

[
\widehat{Energy}(c) = \widehat{Power}(c) \cdot \Delta t
]

where (\Delta t) is the window length.

Safe candidates satisfy:

[
\widehat{TTFT}(c) \le SLO_{TTFT}
]

[
\widehat{TPOT}(c) \le SLO_{TPOT}
]

Among safe candidates, choose:

[
c^* = \arg\min \widehat{Energy}(c)
]

If no candidate is fully safe, select the one with the lowest SLO violation penalty.

---

## 10. Candidate search

A full continuous optimization space is too large.
SWEEP-LLM should search over a **small discrete candidate set**.

Example candidate sets:

* `alpha ∈ {1.0}` for first version
* `beta ∈ {1.0, 0.75, 0.5}`
* `TP_prefill ∈ {1, 2, 4}`
* `TP_decode ∈ {1, 2, 4}`
* `freq_prefill ∈ {high, sweet, low}`
* `freq_decode_L4 ∈ {high, sweet}`
* `N_active_L40S ∈ {1, 2, ..., max}`
* `N_active_L4 ∈ {1, 2, ..., max}`

State-aware filtering reduces the search space.

Examples:

* `PREFILL_HEAVY`: prefer larger L40S capacity, keep `beta = 1`
* `DECODE_HEAVY`: search `beta < 1` if L4 is not enough
* `BALANCED`: allow aggressive consolidation
* `BURST`: restrict to conservative high-capacity configs

---

## 11. High-load and low-load behavior

This is the closest connection to SWEEP.

### HP mode

Under high pressure, all or most GPUs are active.
The main source of savings is:

* joint tuning of routing
* joint tuning of TP
* pool-specific frequency choices

So in HP-like states (`PREFILL_HEAVY`, `DECODE_HEAVY`, `BURST`):

* frequency and routing are the main knobs
* active GPU counts may be near maximum
* decode spillover may be enabled

---

### LP mode

Under light load, the main source of savings is consolidation.

In `BALANCED` low-load windows, SWEEP-LLM can:

* reduce `N_active_L40S`
* reduce `N_active_L4`
* power-gate idle GPUs
* run surviving GPUs at lower frequencies

This reduces idle cluster power significantly.

---

## 12. Decode spillover: key SWEEP-style mechanism

One important SWEEP-LLM idea is that **phase specialization is preferred, but not rigid**.

Default mapping:

* prefill -> L40S
* decode -> L4

But in decode-heavy windows, L4 may be saturated even at:

* max active GPUs
* max decode TP
* high frequency

At that point, SWEEP-LLM enables **decode spillover**:

* some decode requests are routed to L40S
* controlled by `beta`

This is directly analogous to SWEEP’s heterogeneous task-distribution percentage.

Important design principle:

* do not migrate a request mid-decode in version 1
* instead, route requests to their decode pool at decode admission time

That keeps implementation manageable.

---

## 13. Relationship to prior work

SWEEP-LLM differs from prior systems in the following way:

### DVFS-only methods

They optimize frequency only.
They do not jointly consider routing or TP.

### Performance-oriented disaggregation systems

They separate prefill and decode for throughput or cost.
They do not optimize energy jointly across routing, TP, and frequency.

### DynamoLLM-like approaches

They optimize multiple knobs, but in a decoupled hierarchy.
Routing, parallelism, and frequency do not interact in one joint optimization step.

### SWEEP-LLM

It explicitly models the interaction between:

* which pool gets the load
* how much parallelism that pool uses
* what frequency that pool should run at

This is the central research contribution.

---

## 14. First prototype vs full system

### First prototype

A practical first version of SWEEP-LLM should do:

* disaggregated serving with L40S prefill and L4 decode
* state-aware window classification
* joint search over `beta`, TP, frequency, and active GPU counts
* placeholder KV transfer model
* no online TP reconfiguration in the real system if it requires server restart
* evaluation via measurement-backed trace replay

### More advanced version

Later versions can add:

* explicit `alpha` search
* richer KV model
* better queueing model
* more accurate decode-context model
* online reconfiguration support
* more than two GPU types

---

## 15. Measurement-backed evaluation methodology

Because changing TP in a real vLLM system may require server restart, a fully online dynamic experiment may be impractical.

Therefore, SWEEP-LLM can be evaluated via:

1. offline profiling on real systems
2. construction of a measurement-backed lookup table
3. trace-driven replay with the scheduler

The lookup table maps:

* GPU type
* TP
* frequency
* input length
* output length
* request rate

to:

* TTFT
* TPOT
* power
* energy

This avoids circular model-only validation while preserving realistic system behavior.

---

## 16. One-sentence summary

**SWEEP-LLM is a state-aware, model-driven scheduler for disaggregated heterogeneous LLM serving that jointly optimizes routing, tensor parallelism, GPU frequency, and active pool size to minimize energy under TTFT and TPOT SLOs.**

---

## 17. Codex-oriented compact spec

If you want a shorter implementation-oriented version for Codex, use this:

```text
Build a scheduler called SWEEP-LLM for heterogeneous disaggregated LLM serving.

System model:
- L40S pool is preferred for prefill.
- L4 pool is preferred for decode.
- Requests go through prefill then decode.
- If prefill and decode are on different pools, add KV transfer cost T_kv.

Control interval:
- Every 5 seconds, collect workload stats and choose the next configuration.

Inputs per window:
- request rate
- representative input length
- representative output length
- prefill queue length
- decode queue length
- current TTFT and TPOT
- active GPU counts

State classifier:
- PREFILL_HEAVY if rate * input_len dominates
- DECODE_HEAVY if rate * output_len dominates
- BALANCED if both are moderate
- BURST if queue growth or request rate spikes

Decision variables:
- alpha: fraction of prefill routed to L40S (first version can fix alpha=1)
- beta: fraction of decode routed to L4; remaining decode spills to L40S
- TP_prefill
- TP_decode_L4
- TP_decode_L40S
- freq_prefill
- freq_decode_L4
- freq_decode_L40S
- N_active_L40S
- N_active_L4

Models:
- prefill latency/power model per GPU type
- decode latency/power model per GPU type
- KV transfer cost model

Objective:
- minimize predicted energy in each window
- subject to TTFT <= SLO_TTFT and TPOT <= SLO_TPOT

State-specific policy:
- PREFILL_HEAVY: prioritize L40S capacity and TTFT safety
- DECODE_HEAVY: prioritize L4 decode capacity; if insufficient, reduce beta so L40S helps decode
- BALANCED: do full joint search and aggressive consolidation/power-gating
- BURST: temporary performance-first safe mode

Evaluation:
- use measurement-backed trace replay if online TP changes require vLLM restart
- lookup table entries should come from real profiling data
```