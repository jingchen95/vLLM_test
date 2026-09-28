# XPYD System & Control Design

**Status:** authoritative design/experiment contract
**Last updated:** 2026-09-28
**Current experimental phase:** Stage C1c — stationary evidence-window-bounded safe-search preflight

This file is the authoritative specification for the XPYD project. It is intended to be read by both humans and coding agents such as Codex. When an implementation, experiment script, or older document conflicts with this file, this file wins unless it is explicitly revised first.

Do not silently reinterpret a design decision. If an experiment reveals that a decision is wrong, record the evidence, revise this file, and only then change the main implementation.

---

# 0. Scientific Goal and Current Position

## 0.1 Project goal

XPYD targets:

> **Energy-efficient heterogeneous prefill/decode (P/D) LLM serving using lightweight runtime feedback, request-level endpoint routing, and per-endpoint DVFS, while satisfying TTFT/TPOT SLOs.**

The central scientific question is:

> **How much offline knowledge is actually necessary for energy-efficient heterogeneous P/D control? Can lightweight runtime feedback capture most of the available energy benefit without exhaustive profiling or predictive latency/power models?**

The project is no longer the old model-driven SWEEP-LLM design. The old design is useful for conceptual insight, especially workload/phase separation, knob coupling, and different control timescales, but its simulator/model-generated results are not evidence for the current real-runtime system.

## 0.2 Current hypothesis

The current hypothesis is:

1. coarse request context plus recent runtime state is sufficient to make useful routing decisions;
2. safe online frequency exploration can discover useful energy-saving operating points without dense offline DVFS profiling;
3. per-endpoint DVFS creates meaningful endpoint asymmetry that request-level routing can exploit;
4. pre-instantiated TP-diverse endpoints can expose additional capability/energy choices without making TP a slow online reconfiguration knob;
5. explicit profiling, e.g. Canary exploration, may improve steady-state decisions but must justify its learning energy, capacity cost, convergence time, and knowledge lifetime.

The final paper should therefore compare **energy benefit versus knowledge cost**, not only energy versus a static baseline.

## 0.3 What is and is not novel

Do **not** claim novelty in:

- P/D disaggregation itself;
- heterogeneous P/D serving itself;
- multi-P/multi-D routing itself;
- DVFS itself;
- tensor parallelism itself;
- model-free control itself;
- online exploration itself.

The intended contribution is the intersection:

> **A real measurement-driven control system for heterogeneous phase-disaggregated LLM serving that combines request-level selection among persistent P/D endpoints with per-endpoint DVFS and pre-instantiated TP diversity, while deliberately minimizing dependence on offline predictive models and quantifying the cost/value of additional knowledge.**

---

# 1. Scope and Non-Goals

## 1.1 In scope

The final XPYD runtime may control:

- request-level Prefill endpoint selection;
- request-level Decode endpoint selection;
- per-endpoint / per-TP-group GPU core frequency;
- safe online exploration of frequency operating points;
- load-aware pair routing;
- selection among persistent endpoints with different startup TP degrees.

## 1.2 Out of scope for the main controller

The following are explicitly **not** fast runtime knobs in this project:

- changing TP of a running endpoint;
- destroying/restarting model instances per workload window;
- online model repartitioning;
- GPU autoscaling / powering GPUs on and off;
- dynamic active-GPU-count optimization;
- heavyweight RL;
- a dense offline latency/power predictor unless a measured failure mode later proves it necessary.

A fixed deployed endpoint inventory is assumed during each matched experiment.

## 1.3 Old SWEEP-LLM artifacts

Preserve old SWEEP-LLM simulator code, models, and results for historical/reference purposes. Do not overwrite or reinterpret them as physical P/D ground truth. In particular:

- old single-pool phase models are not authoritative predictors for the current physical disaggregated runtime;
- old model-generated joint-optimization percentages are not current XPYD results;
- old 5-second fixed-window control is not the current control semantics.

---

# 2. System Model

## 2.1 Phase-level heterogeneous pools

XPYD uses phase-level hardware heterogeneity:

- **Prefill pool:** GPU type A;
- **Decode pool:** GPU type B.

Current deployment:

- Prefill: L40S on Neptune;
- Decode: L4 on Europa.

The current validated 2P2D substrate is:

```text
P0: L40S, TP=1
P1: L40S, TP=1
D0: L4,   TP=1
D1: L4,   TP=1
```

All four logical pairs are available:

```text
P0 -> D0
P0 -> D1
P1 -> D0
P1 -> D1
```

The system does **not** require mixed GPU types inside the same pool.

## 2.2 Persistent endpoints

Each serving endpoint is persistent once launched and has:

```text
endpoint_id
role = PREFILL | DECODE
node
physical_gpu_ids[]
gpu_type
startup_tp
supported_frequency_states[]
requested_frequency
observed_frequency
health_state
telemetry_freshness
runtime_load_state
```

A multi-GPU TP endpoint is one serving endpoint. All GPUs in that TP group must use the same requested frequency state during an experiment unless a future hardware study explicitly changes this rule.

## 2.3 P->D compatibility

A request is assigned to a compatible logical pair:

```text
P_i -> D_j
```

Keep these concepts separate:

- **logical pair compatibility:** whether the pair can execute a request correctly;
- **transport implementation:** how vLLM/CUDA/NCCL/network resources are actually created or reused.

Do not infer transport behavior from logical routing. Audit it from code/runtime when needed.

---

# 3. Deployment Plane vs Runtime Control Plane

XPYD deliberately separates slow deployment choices from runtime decisions.

## 3.1 Deployment plane

Fixed for one matched experiment:

```text
GPU inventory
endpoint inventory
P/D role of each endpoint
startup TP of each endpoint
model / serving stack
supported frequency action space
SLO contract
```

## 3.2 Runtime control plane

May change during execution:

```text
request -> P_i
request -> D_j
endpoint frequency
feedback confidence
safe/unsafe frequency knowledge
routing distribution
```

## 3.3 Core rule

> **TP is an endpoint capability, not a fast runtime action. Frequency and routing are runtime actions.**

---

# 4. Control Granularity — Frozen Decision 1

XPYD does **not** use a fixed-duration scheduling window as the sole trigger for control.

The control structure is:

```text
routing             = request-level
telemetry           = continuous / sampled
energy accounting   = accounting interval / trace-level
normal DVFS         = evidence-epoch driven
safety DVFS         = event-driven
```

## 4.1 Request-level actions

At admission, each request is classified and assigned to a P/D pair using information available **at that time**.

## 4.2 Continuous telemetry

Collect continuously or at the fastest reliable scrape/sampling rate:

- endpoint power;
- observed frequency;
- endpoint health;
- running/waiting/inflight indicators;
- proxy accepted/completed/failed counts;
- scheduler/client waiting state where exposed.

## 4.3 Evidence-based efficiency epoch

Normal energy-saving DVFS decisions occur only when enough fresh evidence has accumulated.

An evidence epoch is **not** defined solely by elapsed seconds. It closes when either:

1. the phase has accumulated the configured minimum useful evidence after the previous decision; or
2. the maximum evidence age/timeout is reached.

Evidence is phase-specific:

- Prefill evidence emphasizes completed prefills, input-token coverage, TTFT, and P-side pressure;
- Decode evidence emphasizes completed generations/output tokens, TPOT/ITL, and D-side pressure.

Exact minimum-evidence and timeout values are configuration parameters. They must be calibrated once on the current workload rate and then frozen before policy comparison. They must not be tuned separately for competing policies.

## 4.4 Event-driven safety path

The controller must not wait for the next evidence epoch when fresh risk appears.

Examples:

- sudden arrival-rate increase;
- scheduler/client backlog growth;
- burst of long-input requests;
- burst of long-output budgets;
- fresh TTFT/TPOT violation or near-violation;
- endpoint health/freshness failure;
- rapid pressure increase.

Safety response is immediate and conservative.

> **Fast up; evidence-driven down.**

## 4.5 Fixed windows remain useful

Fixed windows may still be used for:

- power integration;
- plots;
- trace summaries;
- offline analysis;
- matched reporting.

They are not mandatory scheduling boundaries.

---

# 5. Two-Level Context Representation — Frozen Decision 2

XPYD uses **two different context layers**. They must not be collapsed into one label.

## 5.1 Request-level context

Default coarse classes:

```text
SS = short input, short requested output
SL = short input, long requested output
LS = long input, short requested output
LL = long input, long requested output
```

The current controlled benchmark shapes are:

```text
SS = IL 128,  OL 128
SL = IL 128,  OL 512
LS = IL 2048, OL 128
LL = IL 2048, OL 512
```

Current benchmark bucket boundaries are conceptually:

```text
short input  <= 512 tokens
short output <= 256 tokens
```

For synthetic experiments, the request generator knows the requested output budget at admission.

For a future real external request, the online controller may use `max_tokens` / requested generation budget or another admission-time field. **Do not use realized future output length to make an online admission decision.** Realized output length is allowed only for post-hoc feedback and evaluation.

Do not introduce an output-length predictor unless a later measured failure mode requires it.

## 5.2 System-level context: two orthogonal dimensions

System context is represented by:

```text
(load magnitude, phase balance, burst/risk)
```

### Load magnitude

```text
LOW
MEDIUM
HIGH
```

This answers:

> How much aggregate pressure is the deployment experiencing?

### Phase balance

```text
PREFILL_HEAVY
DECODE_HEAVY
BALANCED
```

This answers:

> Which phase is responsible for most of the normalized pressure?

### Independent flag

```text
BURST_OR_RISK = true | false
```

This is not a fourth mutually exclusive workload state. It overlays the normal context.

Examples:

```text
LOW    + PREFILL_HEAVY + no risk
MEDIUM + BALANCED      + no risk
HIGH   + DECODE_HEAVY  + no risk
HIGH   + PREFILL_HEAVY + BURST/RISK
```

## 5.3 Why this replaces the old four-state label

The old labels:

```text
PREFILL_HEAVY
DECODE_HEAVY
BOTH_LOW
BOTH_HIGH
```

mixed load magnitude and phase imbalance into one categorical variable. The new representation keeps them orthogonal so that cases such as `LOW + PREFILL_HEAVY` and `HIGH + PREFILL_HEAVY` are distinguishable.

## 5.4 Continuous pressure underneath the labels

The implementation should retain continuous P/D pressure values and derive coarse labels for control logic and analysis.

Preferred abstraction:

```text
rho_P = normalized recent Prefill demand/pressure
rho_D = normalized recent Decode demand/pressure
```

The normalizer must be based on empirical safe-HIGH reference capacity for the current deployment, not an old predictive model.

A practical implementation may use:

```text
recent input-token arrival rate / reference Prefill token capacity
recent requested-output-token rate / reference Decode token capacity
```

augmented by live backlog/inflight pressure.

The exact reference-capacity values and LOW/MEDIUM/HIGH boundaries must be calibrated and frozen before dynamic-context experiments. Until that calibration is complete, the current stationary experiments continue to use the empirically selected offered-rate labels from Stage C0.

## 5.5 Context is not a fixed action table

Never implement rules such as:

```text
LS -> always P0
SS -> always P1 LOW
```

Context is an input to the decision. Final selection must also consider:

- endpoint frequency/TP capability;
- endpoint pressure;
- system backlog;
- recent class/phase latency evidence;
- telemetry freshness;
- recent energy efficiency;
- exploration state.

---

# 6. Current Empirical Load and SLO Contract

These settings are frozen for the current 2P2D TP1 evaluation unless this file is explicitly revised.

## 6.1 Current stationary workload mix

```text
35% SS
15% SL
35% LS
15% LL
```

## 6.2 Current offered-load labels

From accepted Stage C0 capacity calibration:

```text
LOW              = 0.05 req/s
MEDIUM           = 0.10 req/s
STABLE-MEDIUM    = 0.15 req/s  # optional characterization point
HIGH-SUSTAINABLE = 0.20 req/s
OVERLOAD          = 0.30 req/s
```

`0.30 req/s` is not a valid normal operating load: it produced sustained admission-delay growth and residual waiting demand.

These rate labels are current-testbed experimental labels. They are **not** the final portable definition of normalized system load.

## 6.3 Current SLO contract

The current unified request-level contract is:

```text
TTFT <= 600 ms
TPOT <= 65 ms
```

A tested condition is SLO-valid only if **at least 95% of requests satisfy each relevant threshold**.

This is **not** equivalent to claiming `p99 <= SLO`.

Report p95 and p99 separately, but use the request-level 95%-attainment rule for the current validity gate.

The 600 ms TTFT threshold was selected after Stage C0.5 showed that the previous 550 ms threshold was violated by the fixed-HIGH baseline itself, mainly on long-input requests. Do not retune the SLO after seeing controller results.

## 6.4 Current safe-HIGH frequencies

For the current 2P2D TP1 experiments:

```text
P0/P1 L40S HIGH = 2520 MHz
D0/D1 L4   HIGH = 1500 MHz
```

The L4 value here is the current validated experiment-safe HIGH state, not necessarily the device's maximum supported hardware clock.

---

# 7. Request Path and Required Request Record

For each logical request:

```text
Incoming request
    -> admission timestamp
    -> request context (SS/SL/LS/LL)
    -> observe fresh system/endpoint state
    -> choose compatible P_i, D_j pair
    -> execute Prefill
    -> KV handoff
    -> execute Decode
    -> stream output
    -> record final outcome
```

Each request record must be auditable and contain at least:

```text
request_id
scheduled_arrival_ts
actual_dispatch_ts
admission_delay_ms
input_tokens
requested_output_tokens
actual_output_tokens
request_class
system_load_label_at_dispatch
phase_balance_label_at_dispatch
burst_risk_flag_at_dispatch
selected_p_endpoint
selected_d_endpoint
p_startup_tp
d_startup_tp
p_requested_freq
p_observed_freq
d_requested_freq
d_observed_freq
TTFT_ms
TPOT_or_ITL_ms
E2E_latency_ms
completion_status
TTFT_pass
TPOT_pass
```

Do not omit dispatch/admission timing. Otherwise client-side waiting can disappear from the visible system state.

---

# 8. Telemetry, Pressure, and Measurement Semantics

## 8.1 Request-level measurements

TTFT and TPOT are fundamentally request-level metrics.

Retain per-class distributions when sample count permits.

## 8.2 System pressure must include more than endpoint queue

Endpoint queue alone is insufficient.

The runtime should expose, when available:

```text
client waiting/admission backlog
proxy inflight/unassigned
per-pair active requests
per-endpoint assigned inflight
vLLM running/waiting
outstanding input-token work
outstanding requested-output-token work
```

Do **not** sum counters from different layers blindly; they may represent overlapping populations. Keep the pressure vector and define any derived pressure score explicitly.

## 8.3 P-side running gauge limitation

Short Prefill intervals can be missed by a 1-second metrics scrape. Therefore:

> **P-side sampled running==0 is not proof that the Prefill endpoint was idle.**

Use request attribution, token deltas, completion events, power, and other telemetry together.

## 8.4 Telemetry freshness

Every endpoint/state input used for control must have a freshness timestamp.

Stale or missing telemetry is not interpreted as zero load or large headroom.

---

# 9. Energy Measurement and Attribution

## 9.1 Primary objective

For the current 2P2D system:

```text
E_total = E_P0 + E_P1 + E_D0 + E_D1
```

For future TP-diverse inventories, sum the physical GPU energy of **every deployed GPU**, including idle GPUs assigned to endpoints that handled no requests during the interval.

Primary objective:

> **Minimize total deployed-GPU energy subject to the TTFT/TPOT SLO contract.**

## 9.2 Per-endpoint feedback statistics

For each endpoint/accounting interval retain:

```text
gross energy
request count
input tokens handled
output tokens handled
frequency residency
busy/load indicators
```

Useful amortized feedback:

```text
Prefill: E_P / input tokens handled
Decode:  E_D / output tokens handled
E_endpoint / requests handled
```

These are **window/interval-amortized statistics**, not exact per-request joules.

## 9.3 Measurement-resolution boundary

NVML effective power resolution can be coarser than one Prefill execution.

Therefore:

- do not infer exact single-request Prefill energy;
- do not build the controller around exact request energy attribution;
- total endpoint/system energy over longer matched intervals remains valid.

---

# 10. Cold Start and Unknown-State Safety

At process start or after invalidation:

```text
all active endpoints -> validated HIGH
routing -> safe RR / least-loaded compatible
knowledge confidence -> unknown
```

Cold-start priorities:

1. SLO safety;
2. gather representative observations;
3. avoid traffic lock-in;
4. do not pretend unseen configurations are known-optimal.

For unseen or stale conditions:

```text
unknown context
    -> conservative frequency
    -> safe compatible routing
    -> collect fresh evidence
    -> explore only after evidence is sufficient
```

Unknown telemetry must never silently become:

```text
queue = 0
load = 0
large SLO headroom
safe low frequency
```

---

# 11. Safe Frequency Discovery — Frozen Decision 3

XPYD does not assume that the lowest useful frequency is known offline.

It must discover useful operating points from production/runtime feedback.

## 11.1 One-step descent is only a conservative baseline

A pure:

```text
HIGH -> next lower -> next lower -> ...
```

search may waste substantial energy opportunity when many frequency levels exist.

Do not make one-step hill climbing the final algorithm.

## 11.2 Main design: headroom-guided coarse-to-fine safe exploration

The final exploration mechanism should exploit the ordered frequency space.

Conceptually:

```text
known-safe HIGH
      |
      | sufficient stable headroom
      v
coarse lower-frequency probe
      |
      +-- safe   -> expand downward
      |
      +-- unsafe -> restore previous safe point
                    and refine the safe/unsafe bracket
```

Use a deterministic bracket/refinement state machine rather than an opaque RL policy.

## 11.3 Required per-endpoint search state

Maintain at least:

```text
current_frequency_index
last_validated_safe_frequency
unsafe_bound_if_known
best_measured_safe_energy_point
search_phase = COLD | EXPAND | REFINE | HOLD | RECOVER
context_signature
context_confidence
last_transition_time
```

Frequency candidates are ordered from highest/safest to lowest.

## 11.4 Search semantics

The algorithm must follow these rules:

1. start from the validated HIGH point;
2. do not explore down while telemetry is stale, backlog is growing, or the context is unstable;
3. when headroom is clearly large, a probe may skip multiple frequency levels;
4. after a safe coarse probe, continue toward the lower part of the remaining range;
5. after an unsafe/risky probe, immediately restore the previous known-safe point;
6. once a safe/unsafe bracket exists, refine inside the bracket rather than restarting from HIGH;
7. after the safety boundary is approximately located, compare a small number of safe candidate points for energy because the lowest SLO-safe frequency is not guaranteed to be the minimum-energy point;
8. retain the best **measured, SLO-safe** operating point for the current context;
9. do not exhaustively sweep every frequency during normal online control.

A practical implementation may use midpoint/binary-like refinement over frequency indices. Exact jump aggressiveness is a configurable policy parameter and must be frozen before the main comparison.

## 11.5 Safety classification of a probe

A frequency probe is not declared safe after one lucky request.

A probe can become `VALIDATED_SAFE` only after:

```text
minimum fresh evidence satisfied
AND context stable during evidence
AND no sustained backlog growth
AND current SLO-attainment gate satisfied
AND no hardware/thermal/health failure
```

The exact online evidence count and conservative headroom threshold are **not yet frozen** because Stage C1b is currently measuring real lower-frequency headroom. They must be selected from that characterization before Stage C1c and then frozen.

This is an explicit experiment gate, not an implementation ambiguity.

## 11.6 Context-conditioned knowledge

A safe frequency learned under one context is not universally safe.

Substantial changes in:

```text
load magnitude
phase balance
burst/risk state
endpoint TP/inventory
SLO contract
```

must reduce confidence or invalidate the cached safe point.

The controller may retain the old point as historical evidence, but it cannot treat it as currently validated without fresh confirmation.

## 11.7 What not to add yet

Do not add the following before the deterministic coarse-to-fine controller is evaluated:

- Bayesian optimization;
- SafeOpt;
- contextual bandits;
- Thompson sampling;
- RL;
- learned latency predictor.

These are possible later baselines/refinements only if the simple safe search has a measured convergence/regret problem.

---

# 12. Per-Endpoint DVFS — Frozen Decision 4

The final XPYD controller uses **per-serving-endpoint / per-TP-group DVFS**.

## 12.1 Final semantics

Possible final state:

```text
P0: TP1, HIGH
P1: TP1, LOW
P2: TP2, MID

D0: TP1, HIGH
D1: TP1, LOW
D2: TP2, MID
```

This endpoint asymmetry creates a meaningful routing decision space.

## 12.2 TP-group synchronization

If endpoint `P2` uses two physical GPUs:

```text
P2 = GPU2 + GPU3, TP=2
```

then both GPUs use the same endpoint frequency state for this project.

## 12.3 Shared-pool DVFS is an experimental isolation tool

Early C1/C2 experiments intentionally use:

```text
P0 = P1 = shared P frequency
D0 = D1 = shared D frequency
```

This isolates feedback-DVFS behavior from routing/frequency asymmetry.

Shared-pool DVFS is **not** the final controller architecture.

## 12.4 Local action, system-aware safety

The action is local:

```text
set_frequency(P_i, f)
```

but the safety decision must consider:

```text
endpoint latency
endpoint load
system/client backlog
request mix
phase pressure
routing distribution
telemetry freshness
```

Do not run independent endpoint hill-climbers that ignore the system-level traffic redistribution they cause.


---

# 13. TP-Diverse Endpoint Inventory — Frozen Decision 5

XPYD will introduce TP diversity by **pre-instantiating endpoints with different startup TP degrees**.

## 13.1 TP is not reconfigured online

Do not do:

```text
request arrives
-> stop TP1 endpoint
-> relaunch as TP2
-> reload/repartition model
```

Instead, deploy persistent capability-distinct endpoints before the experiment.

Example with four physical GPUs in one phase pool:

```text
P0 = TP1 -> GPU0
P1 = TP1 -> GPU1
P2 = TP2 -> GPU2 + GPU3
```

A Decode pool may use the same pattern if enough physical GPUs are available.

The runtime then selects among `P0/P1/P2`; it does not change their TP.

## 13.2 3P3D is a useful candidate layout, not a sacred target

The goal is **capability diversity**, not the number three itself.

A useful layout must contain endpoints that expose meaningfully different capacity/latency/energy behavior, e.g. TP1 versus TP2.

If two TP1 endpoints are redundant for a particular experiment, a smaller inventory may be scientifically sufficient. Conversely, matched load-balancing experiments may require duplicate TP1 endpoints.

## 13.3 Idle energy is part of the objective

Every GPU in the fixed deployed inventory contributes to total energy even if its endpoint receives no request.

This prevents a TP2 option from appearing artificially free when two GPUs remain powered but idle.

## 13.4 Matched inventory requirement

When comparing schedulers, use the same:

```text
physical GPUs
endpoint inventory
TP assignment
frequency action space
```

Do not compare a baseline on 2P2D TP1 against XPYD on a richer 3P3D TP1/TP2 deployment and attribute all difference to scheduling.

## 13.5 Provisioning sensitivity is secondary

Possible later sensitivity:

```text
4 x TP1
vs
2 x TP2
vs
2 x TP1 + 1 x TP2
```

This is a deployment study, not part of the main fast runtime controller.

---

# 14. Load-Aware Pair Routing — Frozen Decision 6

Final routing must be **request-aware, safety-first, load-aware, and pair-aware**.

It must not be a fixed class table and must not route solely to the historically lowest-energy endpoint.

## 14.1 Decision object

The logical routing decision is:

```text
(request context,
 system context,
 P_i state,
 D_j state)
    -> compatible pair (P_i, D_j)
```

The pair state includes:

```text
startup TP
current frequency
health/freshness
recent latency evidence
live pressure
recent energy efficiency
exploration confidence
```

## 14.2 Hierarchical routing order

Use this priority order:

```text
1. Compatibility and health
2. SLO-safety eligibility
3. Live pressure / overload avoidance
4. Energy preference
5. Controlled exploration / tie breaking
```

Do not begin with an arbitrary weighted sum such as:

```text
a*energy + b*queue + c*latency + d*class
```

unless later evidence shows the hierarchical rule is insufficient.

## 14.3 Safety filter

A pair is ineligible if:

- either endpoint is unhealthy;
- telemetry required for safe use is stale;
- the endpoint/frequency/context combination has recent evidence of SLO risk;
- a TP/frequency state is incompatible with the request or experiment constraints.

For unknown low-frequency states, prefer a validated safe endpoint until the exploration controller explicitly probes the unknown state.

## 14.4 Pressure filter

Request count alone is not sufficient load information.

Preferred phase pressure approximation:

```text
P endpoint work ~= outstanding input-token work / recent P service rate
D endpoint work ~= outstanding requested-output work / recent D service rate
```

These quantities approximate remaining service time rather than simple queue length.

Initial implementations may combine:

```text
inflight count
outstanding token work
endpoint queue
proxy/client backlog
```

but the exact definition must be logged and fixed for a matched experiment.

## 14.5 Pair-aware rather than independent P and D optimization

Do not assume that independently choosing the least-loaded P and then the least-loaded D is globally safe.

The routing layer must at least evaluate the resulting pair state because traffic concentration on one side can create a bottleneck on the other.

## 14.6 Energy preference

Among pairs that survive safety and pressure filtering, prefer the pair with better recent **measured** amortized energy efficiency for the relevant phase/request context.

Energy history must have a freshness/confidence tag. Do not use old low-load energy statistics as current truth after a major workload shift.

## 14.7 Exploration traffic

A small amount of traffic may be sent to under-observed but safe candidates to avoid permanent lock-in.

Exploration must obey the same safety filter and be logged explicitly as `exploration=true`.

The exploration fraction/trigger is a later routing parameter. Do not hide exploration inside normal routing.

---

# 15. Knowledge-Cost Evaluation — Frozen Decision 7

The paper must evaluate **how much energy benefit is obtained per unit of knowledge acquisition**, not only the final steady-state energy.

## 15.1 Knowledge spectrum

The main spectrum is:

```text
STATIC-HIGH
    no learned knowledge

Feedback-only
    passive production feedback

Adaptive XPYD
    production feedback + sample-efficient safe exploration

Canary / explicit online profiling
    dedicated/duplicated exploration and cached knowledge

Empirical oracle
    exhaustive or broad matched measurement reference
```

The oracle is a reference, not a deployable online algorithm.

## 15.2 Oracle benefit capture

For matched conditions define:

```text
OracleBenefit = E_static - E_oracle

BenefitCapture(policy)
  = (E_static - E_policy)
    / (E_static - E_oracle)
```

Report this only when all energies are collected under the same experimental semantics.

## 15.3 Cumulative energy/regret

Steady-state J/request is insufficient because learning itself consumes energy.

For horizon `H`:

```text
Regret(H) = E_policy_cumulative(H) - E_oracle_cumulative(H)
```

Plot cumulative energy versus requests/time so that convergence and break-even are visible.

## 15.4 Knowledge-acquisition costs to report

At minimum:

1. **exploration energy** — energy spent at non-final operating points;
2. **duplicate/Canary energy** — if explicit duplicate requests are used;
3. **capacity opportunity cost** — GPUs/endpoints reserved for profiling instead of production;
4. **SLO risk during learning** — violations and emergency recoveries;
5. **convergence speed** — requests/time until stable near-best behavior;
6. **knowledge lifetime** — how long a learned decision remains useful after load/mix changes.

## 15.5 Interpretation rule

If Feedback/Adaptive XPYD captures most oracle benefit with low learning cost, do not add a heavier model merely to improve a small residual gap.

If Canary or another stronger-information policy materially improves cumulative energy and amortizes its exploration cost quickly, then explicit knowledge acquisition is justified.

Either outcome answers the scientific question.

---

# 16. Final Controller Architecture

The intended final control loop is:

```text
                         +----------------------+
Incoming request ------> | Request classifier   |
                         | SS / SL / LS / LL    |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Current system state  |
                         | load magnitude        |
                         | phase balance         |
                         | burst/risk            |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Pair router           |
                         | safety                |
                         | pressure              |
                         | energy preference     |
                         | exploration           |
                         +----------+-----------+
                                    |
                                    v
                             P_i -> D_j
                                    |
                                    v
                         real request execution
                                    |
                +-------------------+-------------------+
                |                                       |
                v                                       v
       request-level latency                    sampled telemetry
       token/work accounting                    power/freq/load
                |                                       |
                +-------------------+-------------------+
                                    v
                         +----------------------+
                         | Rolling feedback      |
                         | evidence confidence   |
                         +----------+-----------+
                                    |
                      +-------------+-------------+
                      |                           |
                      v                           v
             event-driven safety        evidence-epoch efficiency
             immediate upshift          safe coarse-to-fine search
```

---

# 17. Controller State Contract for Implementation

A coding agent should implement explicit state objects rather than implicit global variables.

## 17.1 Endpoint state

Minimum logical fields:

```text
EndpointState:
    endpoint_id
    role
    gpu_ids
    gpu_type
    tp
    health
    telemetry_fresh
    requested_freq
    observed_freq
    inflight_requests
    outstanding_input_tokens
    outstanding_requested_output_tokens
    recent_ttft_stats
    recent_tpot_stats
    recent_energy
    recent_input_tokens
    recent_output_tokens
    recent_energy_per_input_token
    recent_energy_per_output_token
    recent_service_rate
    search_state
```

## 17.2 System feedback state

```text
SystemFeedbackState:
    evidence_epoch_id
    start_ts
    end_ts
    request_count
    class_counts
    class_fractions
    input_tokens
    requested_output_tokens
    actual_output_tokens
    arrival_rate
    admission_waiting
    proxy_inflight
    phase_pressure_p
    phase_pressure_d
    normalized_rho_p
    normalized_rho_d
    load_magnitude_label
    phase_balance_label
    burst_risk_flag
    ttft_stats
    tpot_stats
    per_endpoint_state
```

## 17.3 Frequency-search state

```text
FrequencySearchState:
    candidate_freqs_desc
    current_index
    last_safe_index
    unsafe_index_or_none
    best_safe_index_or_none
    search_phase
    evidence_count_at_current
    context_signature
    confidence
    transition_reason
```

## 17.4 Routing-decision record

Every route decision must log:

```text
request_id
context
candidate_pairs_before_filter
candidate_pairs_after_safety_filter
candidate_pairs_after_pressure_filter
selected_pair
selection_reason
exploration_flag
```

The log need not be huge human-readable text; structured JSON/CSV fields are sufficient.

---

# 18. Fast Safety Path

The safety path has higher priority than energy optimization.

## 18.1 Triggers

A fast safety action may be triggered by:

- fresh SLO violation;
- recent SLO attainment falling below contract;
- strongly rising admission/backlog pressure;
- sudden long-input or long-output-demand burst;
- endpoint overload;
- stale/invalid telemetry on an endpoint currently at reduced frequency;
- observed/requested frequency mismatch;
- thermal/hardware slowdown or endpoint health failure.

## 18.2 Action

For the affected phase/endpoint:

```text
reduced frequency -> validated safe HIGH
```

and mark the previous low-frequency validation confidence as reduced/invalid for the new context.

## 18.3 Recovery

After safety restoration:

1. gather fresh evidence;
2. wait for stable context;
3. resume safe exploration only after the efficiency-epoch gate is satisfied.

Do not immediately bounce down again after one safe request.

---

# 19. Evidence Hygiene and Experimental Claims

## 19.1 Valid evidence hierarchy

Prefer, in order:

1. end-to-end real runtime measurements under the current harness;
2. controlled hardware characterization under matched conditions;
3. older diagnostic experiments for mechanism understanding;
4. old simulator/model results only for hypothesis generation.

## 19.2 Claims that are currently valid

The project has already established that:

- the physical P/D runtime executes real requests;
- request IDs/tokens/SSE timing can be observed;
- all 2P2D logical routes are available in the validated substrate;
- per-endpoint NVML energy can be integrated;
- per-GPU DVFS actuation/readback works;
- realistic concurrent mixed traffic can be generated;
- client/proxy waiting can reveal pressure not visible in endpoint queue;
- a stable/overload boundary can be measured on the current deployment;
- the current fixed-HIGH substrate supports a 600/65 request-level SLO contract at the chosen sustainable loads.

## 19.3 Claims that are **not yet** established

Do not yet claim that:

- the final feedback controller saves meaningful energy under the current SLO;
- adaptive coarse-to-fine exploration converges efficiently;
- per-endpoint routing beats RR under realistic asymmetric endpoint states;
- TP1/TP2 endpoint diversity improves energy;
- the full joint XPYD controller approaches the oracle;
- Canary profiling is or is not worth its cost;
- current results transfer across platforms/model families.

---

# 20. Completed Work and Current Evidence

This section records the project status as of 2026-09-13.

## 20.1 Early substrate and semantics work — COMPLETE

Completed infrastructure includes:

- endpoint abstraction and compatibility logic;
- live P/D serving semantics;
- SSE/request/token accounting;
- 1P1D validation;
- 2P2D runtime;
- all four logical P/D pairings;
- per-endpoint telemetry;
- per-GPU DVFS actuation/readback;
- NVML energy integration;
- safety restoration;
- feedback -> decision -> actuation -> readback plumbing.

Older low-concurrency and model-driven experiments are retained as diagnostics only.

## 20.2 Architecture audit / realistic harness — COMPLETE

The audit identified a major old-methodology problem: excess demand could wait outside the serving endpoint because of client-side concurrency control, while endpoint waiting metrics stayed zero.

The corrected harness now records:

- scheduled arrival;
- actual dispatch;
- admission delay;
- waiting/inflight transitions;
- proxy accepted/completed/failed counts;
- per-pair/per-endpoint activity;
- class and segment labels.

## 20.3 Stage B — COMPLETE

Validated:

- concurrency=2 and concurrency=4 execution;
- fixed-HIGH four-pair RR;
- dynamic mixed harness;
- real SSE;
- TTFT/TPOT/ITL validity;
- request/token accounting;
- endpoint coverage;
- energy windows;
- frequency readback;
- no thermal/HW slowdown;
- arrival/admit/complete event completeness;
- drain completion.

The initial high admission delays demonstrated that the harness could expose overload; this motivated explicit capacity calibration.

## 20.4 Stage C0 — Capacity/load calibration — COMPLETE

Accepted job: `257127`.

Fixed setup:

```text
P0/P1: L40S, TP1, 2520 MHz
D0/D1: L4,   TP1, 1500 MHz
max concurrency = 4
routing = four-pair RR
mix = 35% SS / 15% SL / 35% LS / 15% LL
48 requests per offered-rate point
```

Measured classification:

```text
0.05 req/s -> LOW
0.10 req/s -> MEDIUM
0.15 req/s -> stable medium optional point
0.20 req/s -> HIGH-SUSTAINABLE
0.30 req/s -> OVERLOAD
```

Key overload evidence at 0.30 req/s:

- completed throughput approximately 0.239 req/s;
- admission-delay slope approximately 116.9 ms/s;
- late-quartile mean admission delay approximately 15.5 s;
- five requests still waiting when offered arrivals ended.

Fixed-HIGH TTFT p95 stayed roughly 558--571 ms over the load sweep. TPOT p95 stayed roughly 58--63 ms, with no 65 ms TPOT violations in C0.

Gross J/request across offered rates must **not** be interpreted as controller savings because all points contain the same 48 requests but different elapsed durations and therefore different accumulated baseline/idle energy.

## 20.5 Stage C0.5 — SLO calibration — COMPLETE

Analysis-only; no new GPU run.

Using sustainable C0 loads 0.05/0.10/0.15/0.20 req/s:

- pooled short-input TTFT p95: 296.6 ms;
- pooled long-input TTFT p95: 571.4 ms;
- 550 ms TTFT violation rate: 15.10% pooled;
- 575 ms: 2.08% pooled;
- 600 ms: 1.04% pooled and <=4.17% at each sustainable load.

Decision:

```text
TTFT SLO = 600 ms
TPOT SLO = 65 ms
validity = >=95% request-level attainment for each threshold
```

Do not retune this primary contract based on later policy output.

A later paper-level SLO-sensitivity experiment may evaluate looser/tighter budgets as a separate axis, but the primary controller development contract remains fixed.

## 20.6 Stage C1a — stationary feedback smoke — COMPLETE

Job: `259228`.

Result directory recorded in the project history:

```text
results/stagec1_stationary/stagec1_smoke_neptune_europa_20260908/
```

All 12 windows passed validity checks. Two startup `pkill` auxiliary return codes were non-errors.

Results:

```text
MEDIUM STATIC-HIGH
  2614.47 J/request
  TTFT p95 559.3 ms
  TTFT attainment 96.7%
  TPOT p95 63.2 ms
  TPOT attainment 100%

MEDIUM FEEDBACK
  2619.12 J/request
  TTFT p95 552.3 ms
  TTFT attainment 100%
  TPOT p95 63.2 ms
  TPOT attainment 100%

HIGH STATIC-HIGH
  1682.98 J/request
  TTFT p95 572.7 ms
  TTFT attainment 100%
  TPOT p95 63.2 ms
  TPOT attainment 100%

HIGH FEEDBACK
  1684.85 J/request
  TTFT p95 570.7 ms
  TTFT attainment 100%
  TPOT p95 63.2 ms
  TPOT attainment 100%
```

Observed frequencies remained:

```text
P0/P1 = 2520 MHz
D0/D1 = 1500 MHz
```

There were no meaningful frequency transitions. Feedback energy difference was approximately noise-level and slightly negative versus static.

Interpretation:

> **The C1 smoke validates the implementation/hard gates, but it does not demonstrate feedback-DVFS energy savings.**

The controller stayed HIGH because the slow-down gate was conservative and the observed latency margins, especially D-side TPOT, were small.

## 20.7 Stage C1b — fixed-frequency headroom characterization — CURRENT / RUNNING

Current research question:

> **Does a lower SLO-safe frequency region actually exist under the current concurrent mixed workload, or did C1a correctly remain at HIGH?**

This stage must distinguish:

```text
A. lower safe points exist
   -> controller/search policy is too conservative
   -> calibrate safe exploration and continue

B. little/no lower safe region exists
   -> do not loosen policy merely to force energy savings
   -> current SLO/workload leaves little DVFS opportunity
```

The characterization should use the validated harness and matched traces, with only a small number of fixed-frequency points:

1. lower D while P remains HIGH;
2. lower P while D remains HIGH;
3. only if both axes show headroom, test one combined lower point.

Do **not** convert this into an exhaustive sweep.

For every point record:

```text
requested/readback P frequency
requested/readback D frequency
total energy
J/request
TTFT p95/p99 and attainment
TPOT p95/p99 and attainment
throughput
admission delay/backlog
all hard-gate validity fields
```

This stage is the gate before freezing the online step-down/headroom thresholds for Stage C1c.

---

# 21. Progress Assessment

The previous estimate of roughly 60--65% completion was too high if progress is measured toward a **paper-ready scientific result**, because it weighted infrastructure too heavily.

A better assessment is:

```text
Infrastructure / measurement substrate:     ~75-80% complete
Realistic workload + calibration framework: ~70-80% complete
Final controller design specification:      ~55-65% conceptually complete
Final controller implementation:            substantially less complete
Paper-critical mechanism evidence:          ~20-30% complete
Overall paper-ready project progress:        ~35-40%
```

A milestone view is even clearer:

```text
[done]   real P/D substrate and observability
[done]   realistic concurrent mixed harness
[done]   load calibration
[done]   SLO calibration
[partial] stationary feedback DVFS / headroom diagnosis
[pending] adaptive safe exploration validated
[pending] dynamic feedback adaptation
[pending] per-endpoint DVFS asymmetry + request-aware routing
[pending] TP-diverse endpoint inventory
[pending] full XPYD controller
[pending] knowledge-cost / Canary / empirical-oracle comparison
[pending] broad robustness / optional cross-platform validation
```

Therefore the current project should be treated as **roughly one-third to two-fifths complete scientifically**, even though a large amount of difficult systems infrastructure is already working.


---

# 22. Revised Step-by-Step Experiment Roadmap

Every stage below has one primary research question. Do not add extra experiments unless their outcome would change a design decision.

## Stage C1b — Fixed-frequency headroom characterization

**Status:** current/running.

**Question:** Is there real SLO-safe DVFS headroom under the current stationary mixed workload?

**Frozen setup:**

```text
2P2D
all TP1
concurrency = 4
routing = four-pair RR
mix = 35/15/35/15 SS/SL/LS/LL
loads = MEDIUM 0.10 and HIGH 0.20 req/s
SLO = 600/65 with >=95% attainment
```

**Allowed change:** fixed P/D frequency only.

**Decision after result:**

```text
safe lower points exist
  -> freeze conservative search thresholds
  -> implement Stage C1c

no meaningful safe lower points
  -> do not force C1 savings
  -> record tight-SLO limitation
  -> evaluate SLO-sensitivity later as a scientific dimension
```

## Stage C1c — Adaptive safe-exploration controller, stationary workload

**Question:** Can the deterministic coarse-to-fine controller discover a near-best safe frequency region faster than one-step descent, while preserving the SLO?

**Still isolate routing:**

```text
P0/P1 share one P frequency state
D0/D1 share one D frequency state
routing remains RR
```

This stage tests the exploration algorithm, not request-aware routing.

Stage C1c is deliberately **evidence-window-bounded**. Each completed physical
measurement window produces one controller evidence epoch. Safety recovery can
act at that boundary, but C1c does not claim an event-driven fast-safety path
between windows. Implementing that final hybrid/event-driven architecture
requires different fixed-clock execution semantics and belongs to Stage C2.

**Policies:**

```text
STATIC-HIGH
ONE-STEP-FEEDBACK      # conservative search baseline
ADAPTIVE-SAFE-SEARCH   # main XPYD search mechanism
```

`ONE-STEP-FEEDBACK` and `ADAPTIVE-SAFE-SEARCH` must use identical request
evidence, SLO contract, safety gates, validated frequency action space, and
HIGH cold start. The only exploration difference is maximum downward jump:
one level for ONE-STEP and the frozen headroom-guided coarse jump for ADAPTIVE.
The legacy `0.5*SLO` controller is not the C1c one-step baseline.

The primary C1c SLO remains TTFT <= 600 ms and TPOT <= 65 ms with at least
95% request-level attainment. MODERATE/LOOSE contracts remain sensitivity
results and are not policies in the primary experiment.

Each frequency candidate is classified exactly once from its measured window.
A known unsafe adjacent bound must not be re-probed unless an explicit context
invalidation clears that knowledge. The final window is consumed for
classification and convergence accounting, but cannot start an unmeasured new
downward probe.

**Loads:**

```text
MEDIUM 0.10 req/s
HIGH-SUSTAINABLE 0.20 req/s
```

**Metrics:**

- total energy and J/request;
- request-level SLO attainment;
- p95/p99 TTFT/TPOT;
- number of frequency probes;
- number of requests/time until final stable frequency region;
- cumulative energy versus requests;
- emergency restorations;
- frequency residency;
- backlog/admission delay;
- measured best fixed-frequency reference among the small C1b characterization set.

**Success criterion:**

Adaptive search must preserve the primary SLO and converge materially faster or with lower cumulative energy than one-step search. If it does not, keep the simpler one-step mechanism rather than claiming complexity without benefit.

## Stage C1d — SLO-headroom sensitivity, only if required

Run this only if C1b shows that the primary 600/65 contract leaves almost no DVFS headroom.

**Question:** Is lack of savings caused by the feedback method, or simply by an SLO close to fixed-HIGH performance?

Use the same hardware/workload and compare a small number of predeclared SLO budgets. Do not replace the primary 600/65 result; report sensitivity separately.

The exact additional SLO budgets must be selected before observing policy results and documented in the experiment config.

## Stage C2 — Dynamic shared-pool feedback adaptation

**Prerequisite:** Stage C1c controller semantics frozen.

**Question:** Can feedback/exploration react safely to changing load and phase balance without routing/frequency asymmetry as a confound?

Keep shared pool frequencies:

```text
P0 = P1
D0 = D1
```

Use dynamic traces that separately exercise:

```text
load increase/decrease
prefill-heavy transition
decode-heavy transition
burst/risk
recovery
```

This stage also calibrates the final normalized system-context machinery:

```text
rho_P
rho_D
LOW/MEDIUM/HIGH
PREFILL_HEAVY/DECODE_HEAVY/BALANCED
BURST/RISK
```

Reference capacities and context thresholds must be frozen before the final C2 comparison runs.

**Primary metrics:**

- adaptation delay;
- emergency-upshift latency;
- time spent at unnecessarily high frequency after recovery;
- cumulative energy;
- SLO attainment;
- backlog trajectory;
- state-classification trajectory;
- frequency trajectory;
- oscillation count.

**Gate:** do not move to routing experiments until system context and dynamic DVFS behavior are stable and interpretable.

## Stage D1 — Per-endpoint DVFS asymmetry + request-aware routing

**Question:** Does request-aware, load-aware routing provide value when same-TP endpoints differ in frequency/load state?

Start with 2P2D TP1 to isolate routing x DVFS.

Deliberately create asymmetric endpoint states, for example:

```text
P0 HIGH
P1 LOW
D0 HIGH
D1 LOW
```

or allow the validated per-endpoint controller to create them.

Use mixed request classes simultaneously.

**Policies:**

```text
RR
LEAST-LOADED
CLASS-ONLY ROUTING              # optional diagnostic
XPYD SAFETY+PRESSURE+ENERGY ROUTING
```

All policies use the same endpoint frequency states for the isolation comparison unless the specific experiment is explicitly testing closed-loop interaction.

**Required questions:**

1. Do long-input requests benefit from access to a faster P endpoint?
2. Can short-input requests use a lower-frequency endpoint without harming TTFT?
3. Does pressure-aware routing prevent low-frequency endpoints from becoming hotspots?
4. Is pair-aware routing better than choosing P and D independently?
5. Does routing change total energy under matched traffic, not just latency?

**Gate:** routing must show an interpretable benefit under at least one well-defined asymmetric condition before full joint closed-loop evaluation.

## Stage D2 — TP-diverse endpoint inventory

**Question:** Does pre-instantiated TP diversity create additional useful request-routing choices beyond DVFS/load asymmetry?

Candidate inventory, subject to available physical GPUs:

```text
Prefill:
P0 TP1
P1 TP1
P2 TP2

Decode:
D0 TP1
D1 TP1
D2 TP2
```

The exact physical mapping must be documented before the run.

Do not dynamically reconfigure TP.

First characterize the fixed inventory under a small matched set of request classes and loads. Then compare:

```text
TP-agnostic routing
vs
TP-aware request routing
```

with the same inventory and total-energy accounting.

**Required metrics:**

- per-endpoint utilization;
- route distribution by request class;
- total deployed-GPU energy including idle endpoints;
- TTFT/TPOT attainment;
- throughput/goodput;
- backlog;
- whether TP2 is actually selected and under what conditions.

If TP2 is never useful once idle energy is counted, report that result rather than forcing it into the controller.

## Stage E — Full XPYD controller

**Prerequisites:** C1/C2/D1/D2 mechanisms individually understood.

**Question:** What is the energy/SLO behavior when request-aware routing, per-endpoint DVFS, safe exploration, live pressure, and TP-diverse endpoint selection operate together?

Main runtime loop:

```text
request context
+ normalized system context
+ endpoint TP/frequency/load/energy state
-> safety-first pair selection
-> real execution
-> feedback
-> per-endpoint safe exploration
```

**Baselines:**

```text
STATIC-HIGH
RR + FEEDBACK-DVFS
LOAD-AWARE ROUTING + FIXED FREQUENCY
XPYD FULL
```

Additional baselines should be added only if they isolate a specific contribution.

**Workloads:**

- stationary mixed MEDIUM;
- stationary mixed HIGH-SUSTAINABLE;
- prefill-heavy dynamic trace;
- decode-heavy dynamic trace;
- balanced burst trace;
- at least one longer non-stationary trace if practical.

**Metrics:**

- cumulative and total energy;
- J/request and token-normalized reporting;
- request-level SLO attainment;
- p95/p99 TTFT/TPOT;
- goodput;
- backlog/admission delay;
- frequency residency;
- route distribution by class/context;
- exploration probes/recoveries;
- endpoint utilization;
- controller decision overhead.

## Stage F — Knowledge-cost study

**Question:** How much additional knowledge is worth acquiring?

Compare under matched conditions:

```text
STATIC-HIGH
PASSIVE/LOW-KNOWLEDGE FEEDBACK
ADAPTIVE XPYD SAFE EXPLORATION
CANARY / EXPLICIT ONLINE PROFILING
EMPIRICAL ORACLE
```

The Canary mechanism must be evaluated with its full cost:

- duplicate exploration energy;
- reserved endpoint/GPU energy;
- lost production capacity;
- time to populate the table;
- invalidation/relearning after context shift.

Report:

```text
BenefitCapture
cumulative energy
regret vs oracle
break-even request/time horizon
SLO risk during learning
knowledge lifetime
```

This is where the student's Canary line can become a stronger-information baseline rather than a separate unrelated project.

## Stage G — Robustness / cross-platform validation

This is valuable but not a blocker for proving the main mechanism.

Possible validation axes:

- another L40S/L4 node arrangement;
- Grid'5000 platform if the required P/D communication and DVFS control are available;
- another model size/family;
- a longer trace.

Do not start Stage G before the main mechanism has a stable result on the primary platform.

---

# 23. Required Experiment Hard Gates

Every main GPU experiment must fail closed if required correctness conditions are not met.

At minimum verify:

```text
all expected logical requests accounted for
all expected completions or explicit failures recorded
SSE/token accounting valid
TTFT/TPOT fields valid
request class/segment valid
route pair recorded
requested frequency recorded
observed/readback frequency recorded
frequency readback matches intended state within the experiment rule
energy samples cover the experiment interval
no unhandled endpoint failure
no thermal/HW slowdown flag unless the experiment studies it
concurrency cap obeyed
client/proxy waiting state recorded
drain completed or residual backlog explicitly reported
configuration snapshot saved
```

A run with failed hard gates is infrastructure evidence only; do not use it as an energy-policy result.

---

# 24. Required Output Artifacts for Every Main Run

Codex should make each main run self-contained.

Recommended directory contents:

```text
config.json or config.yaml
run_manifest.json
requests.csv
route_decisions.jsonl
controller_state.jsonl
frequency_transitions.csv
telemetry.csv
energy.csv
summary.json
validation_report.md
stdout/stderr logs
```

The manifest must include:

```text
git commit / branch if available
timestamp
hosts
GPU inventory
endpoint inventory
TP mapping
model/vLLM version
frequency action space
SLO contract
workload generator seed
arrival process
class mix
request count/duration
concurrency
policy name
policy parameters
```

Do not produce a summary that cannot be traced back to request/telemetry rows.

---

# 25. Codex Work Protocol

Every future Codex prompt for a research experiment should begin with:

> **Before implementing this experiment, state the exact research question and explain what different possible outcomes would change in the design. If no outcome would change the design, do not run the experiment.**

A valid experiment task must specify:

1. **research question**;
2. **current hypothesis**;
3. **frozen variables**;
4. **single allowed independent variable or mechanism under test**;
5. **policies/baselines**;
6. **workload and SLO contract**;
7. **hard gates**;
8. **metrics**;
9. **required artifacts**;
10. **decision rule after the result**.

Codex should not autonomously expand the research scope by adding predictors, RL, profiling tables, or extra knobs.

When a result is surprising, diagnose measurement/semantics first. Do not immediately add another heuristic.

---

# 26. Design Principles — Final Checklist

1. **Request-level routing; evidence-epoch efficiency control.**
2. **Continuous telemetry; fixed windows only for accounting/reporting when useful.**
3. **Request context and system context are separate.**
4. **System context = load magnitude x phase balance + burst/risk.**
5. **Fast up; evidence-driven down.**
6. **Unknown/stale state is not a safe low-frequency state.**
7. **Online exploration should be safe and sample-efficient, not exhaustive.**
8. **Per-endpoint / per-TP-group DVFS is the final architecture.**
9. **Shared-pool DVFS is only an isolation stage.**
10. **TP is fixed at endpoint startup; select TP through endpoint routing.**
11. **All deployed GPUs, including idle TP endpoints, count toward total energy.**
12. **Routing priority: compatibility/health -> safety -> pressure -> energy -> exploration.**
13. **Endpoint queue alone is not system load.**
14. **Prefer token-aware pressure over raw request count when practical.**
15. **Do not infer exact per-request energy from coarse NVML samples.**
16. **Do not use realized future output length for online admission decisions.**
17. **Do not let client-side serialization hide demand.**
18. **Do not compare policies under different endpoint/TP/SLO/trace/action semantics.**
19. **Do not add a predictor until a measured failure mode requires one.**
20. **Evaluate cumulative learning cost, not only steady-state energy.**
21. **Every experiment must resolve a design question.**
22. **Negative results are valid; never loosen safety merely to manufacture savings.**

---

# 27. Immediate Next Actions

As of this update, the correct sequence is:

```text
1. Finish Stage C1b fixed-frequency headroom characterization.

2. Classify the result:
   A) safe lower-frequency headroom exists
      -> freeze evidence/headroom thresholds
      -> implement C1c adaptive safe search.
   B) little/no headroom exists
      -> record the limitation
      -> do not force downclocking
      -> decide whether C1d SLO sensitivity is needed.

3. Run C1c stationary adaptive-search comparison.

4. Only after C1c is stable, implement/calibrate the normalized
   system-context state for C2 dynamic adaptation.

5. After dynamic feedback works, move to D1 per-endpoint asymmetry
   and request-aware load-aware routing.

6. Only after D1 is interpretable, deploy a matched TP-diverse inventory
   for D2.

7. Combine mechanisms in Stage E.

8. Finish with Stage F knowledge-cost / Canary / empirical-oracle study.
```

Do **not** jump directly from the current C1b diagnosis to a 3P3D full controller. That would combine frequency search, per-endpoint asymmetry, routing, TP diversity, and context logic before the individual mechanisms are validated.

---

# 28. Open Parameters vs Frozen Semantics

The architecture above is frozen. A few numerical controller parameters remain intentionally pending because the current experiment exists specifically to measure them.

## Frozen now

```text
research question
request-level routing semantics
hybrid evidence/event control structure
SS/SL/LS/LL request context
system context dimensions
600/65 primary SLO contract
>=95% request-level attainment gate
safe-HIGH cold start
coarse-to-fine deterministic exploration design
per-endpoint final DVFS
startup/fixed TP semantics
fixed deployed inventory per matched experiment
load-aware safety-first pair-routing hierarchy
total deployed-GPU energy objective
knowledge-cost evaluation framing
```

## Must be calibrated/frozen before the corresponding stage

```text
minimum evidence count/token coverage per epoch
maximum evidence timeout
headroom threshold for allowing a downward probe
coarse-search jump aggressiveness
context invalidation/confidence decay rule
reference P/D capacities used for rho_P/rho_D
LOW/MEDIUM/HIGH normalized boundaries
PREFILL/DECODE/BALANCED imbalance boundary
burst/risk trigger threshold
routing pressure threshold / equivalence band
controlled exploration fraction or trigger
```

These are not permission for arbitrary tuning. Each must be selected from a designated calibration/characterization stage, recorded in the config, and frozen before the main comparison that uses it.

---

# 29. Final Research Story if the Evidence Supports It

The intended paper narrative is:

1. Real heterogeneous P/D serving exposes multiple runtime operating points, but dense offline modeling/profiling is expensive and brittle.
2. XPYD starts from safe high-frequency persistent endpoints and learns useful operating points from real runtime feedback.
3. It represents demand using coarse request context plus normalized system/phase pressure rather than a large predictor/table.
4. A deterministic safe coarse-to-fine search discovers DVFS opportunity efficiently.
5. Per-endpoint frequency asymmetry and pre-instantiated TP diversity create useful endpoint choices.
6. A safety-first load-aware router assigns each request to a suitable P/D pair without overloading the energy-preferred endpoint.
7. The evaluation then asks how much oracle benefit this low-knowledge system captures and whether stronger explicit Canary profiling earns enough extra benefit to amortize its knowledge-acquisition cost.

If any mechanism fails experimentally, revise the story around the evidence rather than preserving the planned claim.
