# Related-work gap check (2026-10-08)

Supplement to `RelatedWorkComparison_updated_2026-08-17.md`. Compiled by a
literature-search agent from arXiv abstracts, venue pages and full texts. Spot-check the
high-threat entries before citing them; this list was not independently
re-verified.

**Name collision:** there are two "GreenLLM" papers: the DVFS one
(2508.16449, already covered) and a carbon-aware heterogeneous P/D one
(2412.20322, missing).

## 1. Online / no-offline-profiling energy optimization

| Paper | Venue | Link | Offline profiling | Threat | Note |
|---|---|---|---|---|---|
| Zeus | NSDI'23 | usenix.org/conference/nsdi23/presentation/you | No (just-in-time) | Med | Origin of "explore online instead of profiling"; training, no SLO |
| Perseus | SOSP'24 | arXiv 2312.06902 | Partial (~13 min in-situ) | Med | Reports knowledge-acquisition cost explicitly; model for our reporting |
| HALO (safety-constrained contextual bandit DVFS) | SuRE @ IJCAI-ECAI'26 workshop | ceur-ws.org/Vol-4275/paper18.pdf | No | **High** | No-profiling DVFS with safety zones, vLLM + Mistral-7B; criticises DynamoLLM/throttLL'eM/AGFT/GreenLLM/DualScale for profiling. Single GPU, monolithic, loose 10 s SLO |
| QUICKTOPIA | CCGrid'26 | DOI 10.1109/CCGrid68966.2026.00014 | No | Med | Per-iteration EDP DVFS, single GPU |
| EnergyUCB | WWW'26 | arXiv 2410.11855 | No | Low–Med | Bandit DVFS with slowdown budget; HPC |
| Phase-Decoupled, Model-Calibrated Power Control for P/D | arXiv 2609.11133 | arXiv 2609.11133 | Partial (auto-calibration) | **High** | SLO-guarded self-calibrating P/D power control, 8×B200, −32.3% vs static clock locks; homogeneous, no routing |
| ECO | arXiv 2610.08373 | arXiv 2610.08373 | Partial (budgeted BO) | Med | Treats measurement as a budgeted cost |
| WattGPU | arXiv 2607.02391 | arXiv 2607.02391 | No hardware needed | Med | Spec-based priors; reviewers may ask why not use it |

## 2. Heterogeneous P/D and heterogeneous-GPU serving

| Paper | Venue | Link | Offline profiling | Threat |
|---|---|---|---|---|
| Splitwise | ISCA'24 | arXiv 2311.18677 | Yes | Med: original heterogeneous P/D under power/cost budgets (provisioning) |
| DistServe | OSDI'24 | arXiv 2401.09670 | Yes | Low |
| Mélange | arXiv 2404.14527 | arXiv 2404.14527 | Yes | Low–Med |
| HexGen-2 | ICLR'25 | arXiv 2502.07903 | Yes | Med: heterogeneous P/D + KV links, no DVFS |
| ThunderServe | MLSys'25 | arXiv 2502.09334 | Yes | Low–Med |
| Helix | ASPLOS'25 | arXiv 2406.01566 | Yes | Low |
| Cronus | arXiv 2509.17357 | arXiv 2509.17357 | Partial | Low–Med: high-end/low-end prefill split |
| Hetis | arXiv 2509.08309 | arXiv 2509.08309 | Partial | Low |
| GreenLLM (carbon) | arXiv 2412.20322 | arXiv 2412.20322 | Yes | **Med–High**: heterogeneous P/D reusing older GPUs, −40.6% carbon |
| EcoServe | arXiv 2502.05043 | arXiv 2502.05043 | Yes | Med: carbon-aware rightsizing |

## 3. Energy proportionality, idle power, consolidation, role switching

| Paper | Venue | Link | Offline profiling | Threat |
|---|---|---|---|---|
| POLCA | ASPLOS'24 | arXiv 2308.12908 | Yes | Med: evidence that inference clusters are underutilised |
| TAPAS | ASPLOS'25 | arXiv 2501.02600 | Yes | Low |
| μ-Serve | ATC'24 | usenix.org/conference/atc24/presentation/qiu | Yes | Low–Med |
| Energy Cost of Execution-Idle in GPU Clusters | arXiv 2604.04745 | arXiv 2604.04745 | n/a | **Med–High**: closest to our idle-floor finding; not P/D |
| Festina | arXiv 2606.30391 | arXiv 2606.30391 | Yes | **High** for consolidation (static power); not P/D, not heterogeneous |
| Analytical Power-Aware Provisioning for P/D | arXiv 2609.24639 | arXiv 2609.24639 | Yes | **High** for "how many P/D instances to keep"; homogeneous H200, model-based |
| RAPID | arXiv 2601.12241 | arXiv 2601.12241 | No (feedback) | **High** for feedback-driven P/D control; homogeneous MI300X, power budget not energy |
| TetriInfer | arXiv 2401.11181 | arXiv 2401.11181 | Partial | Low |
| Arrow | arXiv 2505.11916 | arXiv 2505.11916 | No | Med: role switching from live metrics |
| semi-PD | arXiv 2504.19867 | arXiv 2504.19867 | Partial | Med |
| DynaServe | arXiv 2504.09285 | arXiv 2504.09285 | No | Med: colocated vs disaggregated per request |
| HeteroScale | arXiv 2508.19559 | arXiv 2508.19559 | Not stated | Med |
| BlitzScale | OSDI'25 | arXiv 2412.17246 | No | **Med–High**: makes scale-down baselines cheap |
| ServerlessLLM | OSDI'24 | arXiv 2401.14351 | No | Med: cold-start cost reference |
| NVIDIA Dynamo SLA planner | docs | docs.nvidia.com/dynamo | Yes (2–4 h pre-deployment profiling) | Med: industrial profiling-cost reference |
| llm-d variant autoscaler | docs | llm-d.ai/docs/architecture | Unclear | Low–Med |
| The Illusion of Power Capping in LLM Decode | arXiv 2605.11999 | arXiv 2605.11999 | n/a | Med: clock locking vs power capping in decode |

TokenScale (arXiv 2512.03416): only the title was verified.

## 4. SLO conventions

| Paper | TTFT / TPOT(TBT) | Percentile | Arrivals |
|---|---|---|---|
| DistServe | Absolute per app/model (e.g. OPT-13B chat 0.25 s / 0.1 s) | 90% attainment | Poisson |
| Splitwise | Slowdown vs unloaded: TTFT 2×/3×/6×, TBT 1.25×/1.5×/5× at P50/P90/P99 | P50/P90/P99 | Azure sizes, Poisson |
| DynamoLLM | 5× isolated-request latency per class (10×/20× looser) | P99 | Azure 1 h/1 day/1 week |
| VoltanaLLM | 8B: 600 ms / 60 ms | both met | Poisson |
| GreenLLM (DVFS) | TTFT <400 ms short/medium, <2 s long; P95 TBT ≤100 ms | P95 | Alibaba ServeGen, Azure 2024 |
| DualScale | 70B: 600 ms / 100 ms | P99 | Gamma-bursty + Azure |
| Mooncake | 10× TTFT, 5× TBT of lowest-RPS values | P90 | real traces |
| Sarathi-Serve | P99 TBT 5× / 25× of interference-free iteration | P99 | — |

## Implications

- No verified paper combines runtime-feedback, no-profiling energy control with
  SLO safety for heterogeneous P/D. HALO (monolithic DVFS), RAPID (homogeneous
  P/D feedback), and Phase-Decoupled Power Control (homogeneous self-calibrating
  P/D) cover pieces; "no offline profiling" alone is not defensible.
- No verified paper covers energy proportionality / consolidation for
  heterogeneous P/D at low load. Strong adjacent work: execution-idle
  measurement, Festina, 2609.24639, Arrow/DynaServe/semi-PD/RAPID,
  BlitzScale/ServerlessLLM. Reviewers will ask for autoscaling, cold-start and
  colocated baselines.
- Suggested SLO convention: P99 TTFT/TBT at k× unloaded latency of the fastest
  reference path (k ∈ {2, 5, 10}); VoltanaLLM's 8B 600/60 ms as an absolute
  anchor; report attainment (≥90%) and P99; Azure multi-day traces for diurnal
  load, Poisson only for controlled sweeps.
