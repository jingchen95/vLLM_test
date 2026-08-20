# Related Work Comparison — Energy-Efficient Heterogeneous P/D LLM Serving

**Updated:** 2026-08-17  
**Status:** Research-positioning document; not an exhaustive prior-art claim.  
**Current direction:** real prefill/decode (P/D) disaggregated serving with heterogeneous GPUs, persistent mixed-TP endpoints, independent P/D endpoint routing, per-endpoint DVFS, and explicit KV-transfer-aware energy/SLO control.

---

## 0. Executive conclusion

The previous positioning — **“heterogeneous GPU pools + route + TP + DVFS”** — is no longer sufficiently unique by itself.

The closest recent systems occupy different parts of the same design space:

- **DualScale / BiScale**: P/D disaggregation + per-instance TP/provisioning + routing weights + fine-grained DVFS, but on a **homogeneous 16×H100** cluster and strongly dependent on offline latency/power models.
- **PACER**: **heterogeneous P/D** + independent prefill/decode device selection + DVFS, but with **static P/D role assignment**, no model parallelism/TP, lookup-table profiling, and weak/unclear KV-transfer accounting.
- **VoltanaLLM**: P/D disaggregation + SLO-aware frequency control + decode-side state-space routing, but fundamentally a **homogeneous/fixed-topology** design with TP not treated as a runtime selection dimension.
- **FREESH**: heterogeneous GPUs + routing + optimized parallelism + dynamic GPU frequency, but is not presented as a P/D-disaggregated serving runtime.
- **Multi-vendor P/D serving**: heterogeneous/cross-vendor P/D + parallel-strategy and instance-allocation optimization, but energy/DVFS is not the objective.
- **HMA-Serve**: cross-vendor memory-heterogeneous P/D + sophisticated KV-transfer optimization, but targets goodput/cost rather than DVFS-based energy control.

A reviewer-safe novelty target is therefore the **intersection**:

> **A profile-light, topology-aware runtime for energy-efficient P/D-disaggregated serving that operates over a persistent portfolio of heterogeneous, mixed-TP endpoints; independently selects concrete prefill and decode endpoints; adapts per-endpoint DVFS; and explicitly accounts for KV-transfer latency/energy under TTFT/TPOT SLOs.**

As of 2026-08-17, the reviewed literature does not clearly contain one real system combining all of those elements. This should still be written as a literature-based observation, not an unconditional “first” claim.

---

# 1. Are predictive models actually necessary?

## 1.1 Short answer

**No. A learned predictive model is not fundamentally required.**

What a scheduler needs is a mechanism that maps:

```text
current system state
    ↓
candidate action
    ↓
expected SLO risk + expected energy benefit
```

A learned latency/power model is one way to implement this mapping, but not the only way.

| Family | Examples | Offline profiling | Runtime behavior | Main trade-off |
|---|---|---:|---|---|
| Full predictive models + optimizer | DynamoLLM, DualScale, throttLL’eM, existing SWEEP-LLM prototype | High | Predict candidate performance/energy, then optimize | Powerful counterfactual search but high profiling cost and distribution-shift risk |
| Profile/lookup table + controller | PACER | Medium | Lookup profiled sweet spot, then use safety feedback | Simpler than full regressors but still model/hardware-specific |
| Hybrid / model-light feedback | GreenLLM decode, VoltanaLLM feedback layer | Low–medium | Live latency/load feedback corrects or replaces part of prediction | Better adaptability, but control stability matters |
| Fully online learning / model-free policy | AGFT | None for policy pretraining | Online RL/contextual learning tunes frequency while serving | Removes offline data collection but introduces safe-exploration/convergence issues |

The more interesting question is therefore:

> **How much prior characterization is actually necessary to make safe, energy-efficient decisions in a heterogeneous P/D system?**

---

## 1.2 Why models are so common

### A. The action space is combinatorial

Our intended endpoint-level action can include:

```text
prefill endpoint
× decode endpoint
× P frequency
× D frequency
× TP embedded in endpoint identity
× possibly active/warm/cold state
```

Even 3 P endpoints × 3 D endpoints × 6 P frequencies × 6 D frequencies already gives **324** operating points before queue/batch state is considered.

A model makes counterfactual questions cheap:

> “What would happen if this request went to P2→D0 and D0 dropped to 990 MHz?”

Without a model, the answer must come from prior observations or online exploration.

### B. SLO violations make naive exploration dangerous

A bandit can try a poor action and learn that it is poor. In online LLM serving that experiment may be a P99 TTFT or TPOT/TBT violation. Pure online learning therefore needs a safe action set, fallback policy, confidence bounds, or conservative controller.

### C. Queueing and continuous batching make rewards noisy and delayed

Request latency depends on:

```text
queue state
batch composition
other active sequences
KV occupancy
frequency
TP
endpoint type
transfer time
```

The effect of one routing/frequency decision may only become clear several iterations later.

### D. Offline cost is easy to justify when deployment is stable

Many papers implicitly assume:

```text
profile once
serve for days/weeks/months
```

But profiling cost grows roughly with:

\[
N_{model}N_{GPU}N_{role}N_{TP}N_fN_{shape}N_{load}.
\]

That assumption becomes much weaker for heterogeneous systems with changing models, quantization, TP, P/D topology and network paths.

---

## 1.3 Evidence that “no offline model” is a real direction

### AGFT

AGFT explicitly argues that offline models require costly production data and can become stale under workload/model drift. It instead learns a GPU-frequency policy online and states that its controller requires **no offline data collection or pre-training**.

Scope:

```text
continuous-batching inference
+ GPU frequency tuning
```

It does **not** solve heterogeneous P/D endpoint selection, mixed TP, KV transfer or joint P/D routing. It nevertheless proves that the DVFS subproblem does not inherently require an offline predictor.

### GreenLLM

GreenLLM is hybrid:

```text
prefill:
  short traces + compact latency/power model + queue-aware optimization

decode:
  TPS/TBT feedback + dual-loop frequency controller
```

This is strong precedent for using models only where they clearly help.

### PACER

PACER does not require a full regression stack. It profiles energy-optimal frequency sweet spots into lookup tables and reports profiling within roughly two hours per model–hardware pair. That is still offline characterization, but much less than a weeks-long Cartesian campaign.

---

# 2. What went wrong with our previous L40S/L4 profiling approach?

The expensive campaign does not imply that accurate scheduling inherently requires weeks of training data.

## 2.1 The campaign was too Cartesian

The old Phase-2 space swept combinations of:

```text
GPU type
TP
frequency
IL
OL
rate
```

but later scheduler-coverage analysis showed that a small set of query shapes dominated actual scheduler lookups.

A better loop is:

```text
coarse characterization
    ↓
run scheduler/oracle
    ↓
identify decision boundaries / high-value cells
    ↓
profile only those cells
    ↓
update confidence
```

This is **value-directed characterization / active learning**, not uniform brute-force profiling.

## 2.2 The measurement semantics did not match the final P/D problem

The canonical `prefill_*` and `decode_*` models were trained from **monolithic single-pool vLLM measurements**.

Thus the expensive dataset still did not directly measure:

```text
true P endpoint execution
true D endpoint execution
P→D KV transfer
separate P/D queueing
decode continuous batching after transferred KV
endpoint-specific phase power
```

The lesson is not “collect no data.” It is:

> **Profile less, but profile the correct runtime abstraction.**

---

# 3. Recommended modeling philosophy for the new project

I would not commit now to either “full offline ML models” or “fully model-free RL.”

The strongest direction is **profile-light + online-adaptive control**.

## 3.1 Minimal prior calibration

Measure only enough to establish safe operating envelopes:

```text
GPU frequency range
rough phase capacity
idle/active power
TP feasibility
P-TP→D-TP connector compatibility
KV bandwidth/latency
obviously infeasible regions
```

This is a capability/safety layer, not a dense performance model.

## 3.2 Fast online telemetry

Maintain per endpoint:

```text
queue depth
running requests
batch tokens
KV occupancy
recent TTFT/TBT
recent service rate
recent power/energy
current frequency
```

## 3.3 Feedback-based DVFS

A conservative first implementation can be:

```text
if SLO headroom is large:
    decrease frequency one step
elif queue grows or SLO headroom shrinks:
    increase frequency one step
if danger threshold is reached:
    jump to safe/max frequency
```

No learned latency model is necessary for this first controller.

## 3.4 Empirical endpoint ranking

For endpoint/context bucket `x`, maintain quantities such as:

\[
\hat e_i(x)=EWMA(	ext{energy/request})
\]

and

\[
\hat s_i(x)=EWMA(	ext{SLO slack}).
\]

Route only among endpoints with sufficient safety evidence. This can later become a conservative contextual bandit.

## 3.5 Safe exploration is the real challenge

The research problem becomes:

> How can the controller learn which heterogeneous endpoint/TP/frequency combinations are energy-efficient **without violating SLOs while learning**?

Possible mechanisms:

- known-safe fallback;
- one-frequency-step exploration;
- confidence-bound feasibility filter;
- conservative contextual bandit;
- explore only when queue/SLO slack is high;
- transfer a prior from another GPU/model, then correct online.

---

# 4. Closest related work

## 4.1 DualScale / BiScale — closest on P/D + TP + DVFS

**Paper:** *DualScale: Energy-Efficient Disaggregated LLM Serving via Phase-Aware Placement and DVFS*  
(Current arXiv revisions may use the name **BiScale**.)

### Decisions

Tier 1 periodically selects:

```text
number of P instances
number of D instances
TP degree of each instance
baseline frequency of each instance
routing weights
```

Tier 2:

```text
prefill: MPC-based fine DVFS
decode: per-batch minimum-safe frequency
```

### Modeling

```text
offline per-(model,GPU) profiling
→ iteration latency model
→ iteration power model
→ iteration simulator
→ configuration table
→ ILP placement
```

### Correction to our old comparison

We cannot say “DualScale does not consider TP.” It explicitly treats **per-instance TP** as a placement variable, and the ILP can select multiple instance configurations.

Therefore **mixed TP configurations by themselves are not novel**.

### Defensible difference

DualScale is evaluated on a **homogeneous 16×H100** cluster.

Our intended problem is:

```text
heterogeneous GPU types
+ persistent endpoint portfolio
+ TP embedded in endpoint identity
+ independent concrete P/D endpoint selection
+ topology-aware KV transfer
+ per-endpoint DVFS
```

A potentially important systems distinction is **fast routing across already-running mixed-TP endpoints** rather than primarily periodic re-provisioning.

DualScale also notes that provisioning transitions require old instances to be torn down/new ones spun up; its reported windows are largely evaluated in steady state. This motivates:

> Can persistent warm mixed-TP endpoints trade modest idle energy for much faster adaptation and lower transition overhead?

---

## 4.2 PACER — closest threat on heterogeneous P/D + DVFS

**Paper:** *PACER: A Phase-Aware Control System of Energy Efficient LLM Inference*  
**Status in supplied material:** 2026 working draft / ICS submission material.

PACER already has:

```text
heterogeneous A100 + Intel Max 1550
separate P/D worker pools
prefill device selector
decode device selector
phase-specific DVFS
offline sweet-spot lookup tables
vLLM integration
```

Therefore we cannot claim:

> “Prior work does not independently select P and D devices in a heterogeneous cluster.”

PACER does.

### Where PACER stops

It assumes:

```text
static P/D role assignment
single-model serving per device
no model parallelism / TP / PP
```

Its primary control knob is frequency.

### PACER reviewer feedback is a checklist for us

The supplied reviews raise:

1. profiling per model/hardware pair can bottleneck deployment;
2. static P/D assignment cannot react to changing phase demand;
3. TP/PP is missing;
4. KV-transfer latency/energy is not accounted for clearly;
5. per-request decisions must account for batch coupling;
6. trivial max-frequency baselines are insufficient;
7. TPJ alone can hide longer occupancy; use energy/request under SLO;
8. tight production SLOs matter;
9. distribution shift can stale lookup tables.

These should become requirements of our design.

### Differentiation

```text
PACER:
  hetero device selection + DVFS
  static P/D roles
  no TP
  lookup profiling

ours:
  heterogeneous concrete P/D endpoints
  mixed TP endpoint portfolio
  joint endpoint-pair choice
  per-endpoint DVFS
  topology/KV-aware objective
  profile-light online adaptation
```

---

## 4.3 VoltanaLLM — P/D routing + adaptive DVFS is already occupied

**Paper:** *VoltanaLLM: Feedback-Driven Frequency Control and State-Space Routing for Energy-Efficient LLM Serving*  
**Venue:** ISC High Performance 2026.

Modules:

- **EcoFreq**: feedback-driven per-instance frequency control;
- **EcoPred**: lightweight latency predictor, profiled offline and adapted online;
- **EcoRoute**: state-space router, especially for decode instances.

This invalidates:

> “No prior work jointly optimizes routing and DVFS in P/D serving.”

VoltanaLLM already does.

Its main routing novelty is decode-side state-space navigation, while prefill routing is comparatively simple. TP is not a first-class runtime endpoint-selection dimension.

Defensible contrast:

```text
VoltanaLLM:
  P/D + route + DVFS
  homogeneous/fixed GPU setting
  TP not a first-class choice

ours:
  heterogeneous GPU endpoint portfolio
  mixed TP endpoint identities
  P/D endpoint-pair compatibility
  network/KV-aware objective
```

---

## 4.4 FREESH — heterogeneous routing + parallelism + DVFS exists

**Paper:** *FREESH: Fair, Resource- and Energy-Efficient Scheduling for LLM Serving on Heterogeneous GPUs*  
**arXiv:** 2511.00807.

FREESH jointly considers:

```text
heterogeneous GPU instances
parallelism
query routing
dynamic GPU frequency
energy/carbon
```

So we cannot claim:

> “First to jointly optimize heterogeneous GPU routing, parallelism and DVFS for LLM serving.”

The distinction is that FREESH is not presented as a P/D-disaggregated runtime with P→D KV-transfer coupling.

---

## 4.5 Multi-vendor heterogeneous P/D serving

**Paper:** *Disaggregated Prefill and Decoding Inference System for Large Language Model Serving on Multi-Vendor GPUs*  
**arXiv:** 2509.17542.

It explicitly combines:

```text
heterogeneous / multi-vendor P/D
heterogeneous transfer support
parallel-strategy optimization
instance-number allocation
```

Therefore **heterogeneous P/D itself is not novel**.

The useful contrast is:

```text
them: hetero P/D + parallelism/deployment
us:   hetero P/D + TP portfolio + DVFS + energy + online routing/control
```

---

## 4.6 HMA-Serve — heterogeneous P/D and KV transfer are first-class

**Paper:** *HBM Is Not All You Need: Efficient Disaggregated LLM Serving across Memory-heterogeneous Accelerators*  
**arXiv:** 2606.29986.

HMA-Serve pairs GDDR-based accelerators for prefill with HBM GPUs for decode, including cross-vendor operation, and optimizes:

- vendor/precision differences;
- compute-transfer pipelining;
- KV-transfer bandwidth;
- deferred dequantization.

It is not a DVFS-energy scheduler, but it is evidence that a serious heterogeneous P/D paper cannot model the KV path as a single constant.

---

## 4.7 GreenLLM — model-light feedback is plausible

**Paper:** *GreenLLM: SLO-Aware Dynamic Frequency Scaling for Energy-Efficient LLM Serving*  
**arXiv:** 2508.16449.

Homogeneous A100 P/D setup:

```text
prefill:
  length routing
  compact latency/power model
  queue-aware frequency optimization

decode:
  TPS/TBT feedback
  dual-loop DVFS
```

This is strong precedent for our proposed hybrid philosophy.

---

## 4.8 AGFT — explicit no-offline-model DVFS

**Paper:** *AGFT: An Adaptive GPU Frequency Tuner for Real-Time LLM Inference Optimization*  
**arXiv:** 2508.01744.

AGFT explicitly criticizes large offline profiling/modeling datasets and uses online learning for frequency control without offline policy training.

It does not solve our routing/topology problem, but directly supports adding a research axis:

```text
full model
vs
compact lookup
vs
feedback
vs
online-learning controller
```

---

## 4.9 Revisiting P/D energy — disaggregation is not automatically greener

**Paper:** *Revisiting Disaggregated Large Language Model Serving for Performance and Energy Implications*  
**arXiv:** 2601.08833.

Key negative result:

> P/D performance benefits depend on load and KV-transfer medium, and phase-wise independent frequency scaling does not necessarily reduce total energy because disaggregation itself has overhead.

Therefore our evaluation must include:

```text
colocated serving
vs
static P/D
vs
energy-aware P/D
```

and measure:

```text
P energy
D energy
KV/network energy
idle energy
```

---

## 4.10 DynamoLLM

**Paper:** *DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency*  
**HPCA 2025.**

It already combines:

```text
request-type pools
instance count
TP/model parallelism
GPU frequency
routing
```

through hierarchical controllers and energy-performance profiles.

It is non-P/D, but established that **routing + parallelism + DVFS + capacity** should be considered jointly. We cannot claim novelty from merely combining those knobs.

---

# 5. Additional adjacent work

## 5.1 Offline Energy-Optimal LLM Serving

Workload-dependent energy/runtime models using input/output tokens for heterogeneous-system scheduling. Relevant to heterogeneous energy modeling, but not P/D + TP/DVFS online control.

## 5.2 Hybrid Heterogeneous Clusters Can Lower Energy

ACM e-Energy 2024. Routes LLM tasks across heterogeneous resources based on input/output tokens. Early heterogeneous energy-placement result, but coarser than our intended P/D runtime.

## 5.3 Investigating Energy Efficiency Across Tasks and DVFS

Empirical characterization across models, tasks, input features and frequency settings. Useful motivation, not a full P/D serving scheduler.

## 5.4 Kernel-Level DVFS

*Reducing Compute Waste in LLMs through Kernel-Level DVFS* (arXiv:2601.08539) shows that DVFS granularity can go below request/phase level. Thus “fine-grained DVFS” alone is not a novelty axis.

## 5.5 Analytical Performance/Power Model + Fine-Grained DVFS

ASPLOS 2025 work uses analytical/structured models for fine-grained accelerator DVFS. Relevant as an alternative to black-box data-heavy modeling.

## 5.6 Modality Inflation

Stage-level MLLM energy characterization across vision encoder, prefill and decode; shows stage-wise DVFS opportunity. Adjacent, not a direct competitor.

## 5.7 AFlex — latest operator-level disaggregation

**arXiv:** 2608.01891.

Disaggregates Attention and FFN and jointly optimizes resource provisioning, operator-level DVFS, microbatch depth and batching. This shows that “finer-grained phase/operator DVFS” is increasingly crowded.

## 5.8 Load-aware prefill deflection

**arXiv:** 2607.02043.

Lets decode nodes execute chunked prefill work when P is overloaded, sometimes eliminating inter-node KV transfer. It targets SLO performance, not energy, but shows that static P/D role boundaries are already being challenged.

## 5.9 HeteroPanacea

**arXiv:** 2608.03741.

Simulation framework for heterogeneous P/D/A/F specialization, quantization and parallelization. Not an energy-aware physical runtime, but broad heterogeneous disaggregation is clearly an active area.

---

# 6. Novelty matrix

Legend: `✅` core / explicit; `◐` partial, fixed, or secondary; `—` not addressed; `?` unclear.

| Work | Real P/D | Hetero GPUs in one deployment | Concrete P/D routing | TP/parallelism decision | DVFS | Fine online adaptation | Explicit KV/topology cost | Avoids heavy offline model |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DynamoLLM | — | — | — | ✅ | ✅ | ◐ | — | — |
| GreenLLM | ✅ | — | ◐ | fixed topology | ✅ | ✅ decode | ◐ | ◐ |
| PACER | ✅ | ✅ | ✅ | — | ✅ | ◐ | — / unclear | ◐ lookup |
| VoltanaLLM | ✅ | — | ✅ mainly decode | fixed/not knob | ✅ | ✅ | ◐ | ◐ online-adaptive predictor |
| DualScale | ✅ | — | ◐ routing weights | ✅ per instance | ✅ | ✅ | ◐ KV-memory, transfer-energy unclear | — |
| FREESH | not P/D-framed | ✅ | n/a | ✅ | ✅ | ◐ | — | — |
| Multi-vendor P/D | ✅ | ✅ | ◐ | ✅ deployment | — | — | ✅ compatibility/transfer | ? |
| HMA-Serve | ✅ | ✅ | design-driven | ◐ | — | — | ✅ | n/a |
| AGFT | — | — | — | — | ✅ | ✅ online RL | — | ✅ |
| AFlex | A/F disagg | — | system scheduler | ◐ | ✅ | ✅ | ✅ A/F communication concern | ? |
| **Proposed** | **✅** | **✅** | **✅ P + D** | **✅ endpoint portfolio** | **✅ per endpoint** | **✅** | **✅ latency + energy + compatibility** | **target: profile-light** |

---

# 7. What we should NOT claim

Do not claim, without major qualification:

- ❌ **“First energy-efficient P/D-disaggregated serving system.”**  
  VoltanaLLM and DualScale already occupy this space.

- ❌ **“First system to jointly use routing and DVFS for P/D.”**  
  VoltanaLLM explicitly does this.

- ❌ **“First heterogeneous P/D energy system.”**  
  PACER already combines heterogeneous hardware, P/D workers, device selection and DVFS.

- ❌ **“First heterogeneous P/D system.”**  
  Multi-vendor P/D and HMA-Serve already exist.

- ❌ **“First to combine TP and DVFS for LLM serving.”**  
  DynamoLLM and DualScale already do so.

- ❌ **“First to allow mixed TP instance configurations.”**  
  DualScale's Tier-1 formulation already permits per-instance TP configurations.

- ❌ **“First model-free online energy controller.”**  
  AGFT already explores online RL frequency control without offline policy training.

---

# 8. Where the proposed work can still be genuinely differentiated

## 8.1 Persistent heterogeneous endpoint portfolio

Rather than changing TP on the critical path:

```text
P0: GPU-type X, TP1
P1: GPU-type X, TP1
P2: GPU-type X, TP2

D0: GPU-type Y, TP1
D1: GPU-type Y, TP1
D2: GPU-type Y, TP2
```

The fast path chooses an endpoint.

Research question:

> Can persistent warm mixed-TP endpoints trade a modest idle-power cost for faster adaptation and lower reconfiguration overhead?

## 8.2 Joint endpoint-pair selection

The unit of decision is:

\[
(P_i,D_j,f_i,f_j)
\]

The pair jointly determines:

```text
compute behavior
TP behavior
KV-layout compatibility
network path
transfer cost
queue interaction
```

## 8.3 Unequal P/D TP as a physical constraint

If supported by the connector, evaluate:

```text
P TP1 → D TP1
P TP1 → D TP2
P TP2 → D TP1
P TP2 → D TP2
```

The contribution is not that unequal TP exists, but **how TP asymmetry interacts with heterogeneous GPU performance, KV transfer, DVFS and energy-aware routing**.

## 8.4 Topology-aware KV energy

Use an objective closer to:

\[
E_{request}=E_P+E_{KV}+E_D+E_{idle/amortized}+E_{transition}
\]

subject to TTFT/TPOT constraints.

`E_KV` should vary with:

```text
nodes
NICs
transfer backend
P/D TP layouts
GPU types
```

## 8.5 Profile-light / online-adaptive scheduling

This may be the strongest differentiator:

> Can a small calibration set plus closed-loop telemetry approach model-heavy optimizers at a fraction of profiling cost?

---

# 9. Proposed research questions

### RQ1 — Is dense offline modeling necessary?

Compare:

```text
full offline model
compact lookup
feedback controller
online contextual-bandit controller
```

Measure:

```text
profiling GPU-hours
time to deploy a new GPU/model
steady-state energy/request
SLO violations
adaptation after distribution shift
```

### RQ2 — What does heterogeneous endpoint selection add?

Compare homogeneous P/D, heterogeneous fixed-role P/D, and heterogeneous adaptive routing.

### RQ3 — What does TP diversity add?

Compare:

```text
all TP1
fixed best TP
persistent mixed-TP portfolio
```

including idle energy.

### RQ4 — Is joint P/D pair selection better than independent greedy selection?

Compare pair-aware selection against capacity-only or separate P-then-D choices.

### RQ5 — How much does KV topology change the energy optimum?

Vary same-node/cross-node, NCCL/NIXL, bandwidth and P-TP/D-TP pair.

### RQ6 — When does disaggregation beat colocated serving in total energy?

Must be answered explicitly.

### RQ7 — Persistent endpoints vs periodic re-provisioning

Compare warm portfolio against DualScale-like provisioning, including startup, loading, idle and transition costs.

---

# 10. Recommended controller architecture

```text
                     slow path
                         │
             endpoint portfolio manager
                         │
          ┌──────────────┴──────────────┐
          │                             │
      P endpoints                   D endpoints
      TP1/TP2                       TP1/TP2
          │                             │
          └────────── router ───────────┘
                         │
                    fast path
                         │
       queue / batch / KV / power telemetry
                         │
              safe action candidate set
                         │
       ┌─────────────────┴─────────────────┐
       │                                   │
 endpoint-pair selection               DVFS feedback
 empirical/bandit ranking              per endpoint
       │                                   │
       └─────────────────┬─────────────────┘
                         │
                 fallback safe mode
```

Principle:

> **Do not predict what the runtime can measure cheaply. Use prediction only where counterfactual reasoning is worth its profiling cost.**

---

# 11. Baselines we will likely need

1. Colocated vLLM.
2. Static P/D at max frequency.
3. Static P/D at best offline frequency.
4. Heterogeneous device-aware routing only — PACER-style ablation.
5. Routing + DVFS — VoltanaLLM-like baseline.
6. Placement/TP + DVFS — DualScale-like conceptual baseline.
7. Heterogeneous routing + TP without DVFS.
8. Full proposed system.

For the profiling question:

9. Full offline predictive models.
10. Compact lookup.
11. Feedback-only.
12. Online-learning/profile-light controller.

If every published system cannot be reproduced, distinguish clearly between:

```text
author implementation
reimplementation
mechanism-matched ablation
```

---

# 12. Required ablations

Isolate:

```text
+ heterogeneous endpoint selection
+ TP diversity
+ DVFS
+ joint P/D routing
+ KV topology awareness
+ online adaptation
```

Also test:

```text
without E_KV
without queue telemetry
without online correction
without warm endpoints
```

---

# 13. Reviewer-safe positioning sentence

> Recent systems have independently explored key elements of energy-efficient LLM serving: DynamoLLM jointly manages routing, parallelism, and frequency in homogeneous inference clusters; DualScale combines P/D provisioning, per-instance TP, and fine-grained DVFS in homogeneous disaggregated clusters; PACER combines heterogeneous P/D device selection with phase-aware frequency control but assumes static roles and no model parallelism; and VoltanaLLM co-designs routing and feedback-driven DVFS for P/D serving. Our work targets a different intersection: a real heterogeneous P/D runtime with a persistent mixed-TP endpoint portfolio, independent concrete prefill/decode endpoint selection, topology-aware KV-transfer accounting, and profile-light online energy adaptation.

Do not use “the first” until the search is repeated immediately before submission.

---

# 14. Recommended project thesis

The strongest paper is probably **not**:

> “We add more knobs than prior work.”

A stronger thesis is:

> **Energy-efficient heterogeneous P/D serving need not require exhaustive model-hardware profiling. By representing TP as persistent endpoint configurations and combining topology-aware endpoint routing with safe feedback-driven DVFS, the system can adapt online to changing workloads and hardware while approaching the energy efficiency of model-heavy optimizers at a fraction of the profiling cost.**

This directly attacks weaknesses already identified in the closest literature:

```text
offline profiling scalability
static role assumptions
missing TP
missing KV cost
distribution shift
transition overhead
```

---

# 15. Papers / systems reviewed

## Closest energy-serving systems

1. **DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency.** HPCA 2025.
2. **GreenLLM: SLO-Aware Dynamic Frequency Scaling for Energy-Efficient LLM Serving.** arXiv:2508.16449.
3. **PACER: A Phase-Aware Control System of Energy Efficient LLM Inference.** Supplied 2026 working draft/review material.
4. **VoltanaLLM: Feedback-Driven Frequency Control and State-Space Routing for Energy-Efficient LLM Serving.** arXiv:2509.04827; ISC High Performance 2026.
5. **DualScale / BiScale: Energy-Efficient Disaggregated LLM Serving via Phase-Aware Placement and DVFS.** arXiv:2602.18755 family / supplied PDF.
6. **AGFT: An Adaptive GPU Frequency Tuner for Real-Time LLM Inference Optimization.** arXiv:2508.01744.
7. **FREESH: Fair, Resource- and Energy-Efficient Scheduling for LLM Serving on Heterogeneous GPUs.** arXiv:2511.00807.

## Heterogeneous / P-D serving

8. **Disaggregated Prefill and Decoding Inference System for Large Language Model Serving on Multi-Vendor GPUs.** arXiv:2509.17542.
9. **HBM Is Not All You Need: Efficient Disaggregated LLM Serving across Memory-heterogeneous Accelerators.** arXiv:2606.29986.
10. **Revisiting Disaggregated Large Language Model Serving for Performance and Energy Implications.** arXiv:2601.08833.
11. **Towards Load-Aware Prefill Deflection for Disaggregated LLM Serving.** arXiv:2607.02043.
12. **SLO-Aware Compute Resource Allocation for Prefill-Decode Disaggregated LLM Inference.** arXiv:2603.04716.
13. **When Does Disaggregation Pay? Simulating Prefill–Decode–Attention–FFN Specialization for Agentic LLM Inference.** arXiv:2608.03741.

## Other energy/modeling work

14. **Offline Energy-Optimal LLM Serving: Workload-Based Energy Models for LLM Inference on Heterogeneous Systems.**
15. **Hybrid Heterogeneous Clusters Can Lower the Energy Consumption of LLM Inference Workloads.** ACM e-Energy 2024.
16. **Investigating Energy Efficiency and Performance Trade-offs in LLM Inference Across Tasks and DVFS Settings.** arXiv:2501.08219.
17. **Reducing Compute Waste in LLMs through Kernel-Level DVFS.** arXiv:2601.08539.
18. **Using Analytical Performance/Power Model and Fine-Grained DVFS to Enhance AI Accelerator Energy Efficiency.** ASPLOS 2025.
19. **Modality Inflation: Energy Characterization and Optimization Opportunities for MLLM Inference.** arXiv:2512.22695.
20. **Energy-Efficient LLM Serving via Disaggregated Attention–FFN and Flexible Frequency Scaling (AFlex).** arXiv:2608.01891.
21. **Demystifying Cost-Efficiency in LLM Serving over Heterogeneous GPUs.** arXiv:2502.00722.

## Methodological inspiration

22. **SWEEP: Adaptive Task Scheduling for Exploring Energy Performance Trade-offs.**
23. **Computer Architecture's AlphaZero Moment: Automated Discovery in an Encircled World.** arXiv:2604.03312.

The AlphaZero paper is not a direct LLM-serving competitor, but its argument for automated exploration plus continuous telemetry aligns conceptually with replacing exhaustive static characterization by adaptive measurement/search.

---

# 16. Immediate design recommendation

Before collecting a dense Grid'5000 dataset:

1. Bring up real heterogeneous P/D.
2. Measure topology/NCCL/NIXL/KV paths.
3. Implement telemetry and a safe DVFS fallback.
4. Build a **feedback-only reference controller**.
5. Measure how far it is from a small offline oracle on a deliberately sampled configuration set.
6. Add a learned predictor only if the gap is large enough to justify its profiling cost.

Let the evidence decide whether the final system is:

```text
model-free
model-light
or
model-assisted
```

rather than assuming the answer in advance.
