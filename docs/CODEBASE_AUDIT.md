# Codebase Audit

Audit date: 2026-08-17. Repository root: `/Users/chjing/Downloads/vLLM_test`.

## Executive summary

This is an established energy-aware LLM-serving research workspace, not a blank project. It contains a complete measurement-to-model-to-simulation pipeline for Mistral-7B on 4xL40S and 8xL4, a class-aware joint scheduler (SWEEP-LLM), synthetic and Azure trace replay, extensive paper/evaluation machinery, and a working but fixed physical vLLM P/D experiment.

The central limitation is the runtime boundary: the scheduler's routes, active instance counts, TP choices, and power-gating choices are simulated/model-level decisions. The physical script launches one L40S prefill producer and one L4 decode consumer, uses one proxy route, and hard-codes both endpoints to TP=1. There is no runtime controller that discovers multiple P/D endpoints, applies `decide_window`, or routes individual requests to independently chosen pre-instantiated instances.

The workspace occupies about **28 GB** and is **not a Git repository**: no `.git` directory, branch, history, tracked/untracked/ignored state, or remote exists. Consequently, every pre-audit file is currently unversioned. A root `.gitignore` and curation manifests have been added, but Git has not been initialized.

## Disk inventory

| Path | Size | Category and purpose | Source-controlled | Reproducible | Needed to understand | Needed to rerun | GitHub suitability |
|---|---:|---|---|---|---|---|---|
| `paper/scripts/` | 1.6 MB | Canonical scheduler, predictors, simulator, analyses, figures | No Git exists | Source | Yes | Yes | MUST INCLUDE |
| root `*.py`, `*.sh` | <0.4 MB | Cluster benchmark, NVML monitor, parsers, calibration, smoke tests | No | Source | Yes | Yes | MUST INCLUDE |
| `paper/` | 8.0 MB | Paper source, docs, scripts, publication figures, 1.2 MB build output | No | Mostly source; build output reproducible | Yes | Mostly | INCLUDE except build output |
| `artifacts/paper/models/` | 19 MB | Canonical predictors, split-validation bundles, B5 models, feasibility tables | No | Refit from compact tables; exact hardware data externally required | Yes | Yes for scheduler | MUST INCLUDE |
| `Phase2_Results_L40S/` | 873 MB | L40S raw benchmark/NVML corpus plus 272 KB merged table | No | Hardware-dependent | Summarized table only | Raw data needed to refit/audit | Include only `master_results.csv` |
| `Phase2_Results_L4/` | 2.0 GB | L4 raw benchmark/NVML corpus plus 304 KB merged table | No | Hardware-dependent | Summarized table only | Raw data needed to refit/audit | Include only `master_results.csv` |
| `results/paper/figures/` | 22 GB | Generated exhaustive candidate-evaluation CSVs and figure outputs | No | Yes from models/scripts/configs | No; selected outputs suffice | No | EXTERNALIZE/REGENERATE |
| other `results/paper/` | about 46 MB | Scheduler analyses, frozen summaries/window outputs, synthetic traces | No | Mostly yes | Selected parts | Selected parts | Include analyses, traces, and compact summaries |
| `results/disagg_20260321_v2/` | 519 MB | Physical monolithic, phase-only, shared-TP, and fixed P/D experiments | No | Hardware/version-dependent | Representative summaries useful | Yes for physical evidence | EXTERNALIZE raw corpus |
| `data/` | 584 MB | Azure public trace downloads; 559 MB is the one-week conversation CSV | No | Yes, public download | No | Only for full preprocessing | EXTERNALIZE/DOWNLOAD |
| `traces/` | 145 MB | Processed compact and full Azure replay traces | No | Yes | Compact traces useful | Yes for production replay | Include 4.5 MB compact subset; externalize full |
| `archive/` | 565 MB | Early experiments, scripts, logs, measurements, and reading material | No | Mixed | No for current path | No | Preserve externally; do not delete |
| `.venv/` | 254 MB | Python 3.9.6 macOS analysis environment | No | Reconstructable | No | Environment equivalent needed | EXCLUDE |
| `models_l40s/`, `models_l4/` | 3.4 MB | Older root-level model snapshots | No | Probably, but provenance less clear | No; canonical path differs | No | EXCLUDE as historical duplicates |
| `Obs2_Validation_L4/` | 18 MB | Targeted benchmark/NVML validation and 4 KB merged summary | No | Hardware-dependent | Summary useful | Raw only for audit | Include summary only |
| `logs/` | 84 KB | SLURM/runtime diagnostics | No | Not important | No | No | EXCLUDE |
| `.claude/`, `.vscode/` | 12 KB | Local agent/editor settings | No | Local | No | No | EXCLUDE `.claude`; editor config optional |

The largest individual files are generated search tables under `results/paper/figures/`: many are approximately 296-312 MB. `data/AzureLLMInferenceTrace_conv_1week.csv` is approximately 559 MB. These exceed or approach unsuitable Git object sizes; the generated search tables exceed GitHub's 100 MB per-file limit.

## Research pipeline reconstructed from code

```text
run_disagg_benchmark.sh + gpu_monitor.py
    -> vLLM benchmark text + per-GPU NVML CSV
    -> Phase2_Results_{L40S,L4}/master_results.csv
    -> paper/scripts/model_fitting_{,l4}.py
    -> paper/scripts/model_fitting_runner.py
    -> artifacts/paper/models/models_{l40s,l4}/
    -> paper/scripts/scheduler.py::EnergyScheduler
    -> paper/scripts/schedulers/sweep.py::SweepLLMStrategy
    -> paper/scripts/jsep_simulator.py or sweep_llm_scheduler.py
    -> results/paper/{analyses,synthetic_traces,*_frozen}/
    -> paper/scripts/figures/* and paper/*.tex
```

The physical P/D path is separate:

```text
run_disagg_benchmark.sh::run_experiment_E
    -> one L40S vLLM producer (TP=1)
    -> generated disagg_proxy.py
    -> P2pNcclConnector KV PUT
    -> one L4 vLLM consumer (TP=1)
    -> benchmark/NVML files in results/disagg_20260321_v2/E_disaggregated/
```

No code path connects `SweepLLMStrategy.decide_window()` to those running endpoints.

## Workload and software assumptions

- Primary model: `mistralai/Mistral-7B-v0.1`, BF16, 32 layers, hidden size 4096, GQA (32 query heads/8 KV heads), recorded as 131,072 KV bytes/token. Evidence: `paper/scripts/model_catalog.py` and model `training_metadata.json`.
- Transfer pilot: `mistralai/Mistral-Nemo-Instruct-2407` (12B) phase-only smoke/v1/v2 collections. This is a diagnostic pilot, not a trained production scheduler model; see `paper/same_family_multisize_mistral_nemo12b_pilot_*` and `paper/docs/same_family_multisize_pilot_v1_checklist.md`.
- Characterization IL grid: 32, 128, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192 tokens. OL grid: 32, 128, 512, 1024. Rates observed: 1, 10, 20, 30, 50 requests/s, with additional targeted B5 rates. Evidence: `paper/COVERAGE_AUDIT.md`, `paper/scripts/jsep_traces.py`.
- Synthetic evaluation: four 600-second controlled traces. T1 ramps 2-34 rps with a prefill-heavy mix; T2 ramps 2-24 rps with a decode-heavy mix; T3 holds 24 rps while shifting T1->T2; T4 has 10-rps baseline and 34-rps 20-second spikes. Class shapes are SS=(64,128), SL=(64,512), LS=(1024,128), LL=(1024,512). Evidence: `paper/scripts/jsep_traces.py` and `paper/evaluation_rewrite.tex`.
- Production evaluation: public Azure conversation and code traces, temporal burstiness preserved, mean rate scaled; compact inputs are in `traces/azure_production/` and `traces/azure_diurnal/`. `load_azure_trace()` snaps raw lengths to the profiling grid.
- SLOs: predictors have L40S thresholds 200/500/1000 ms and L4 thresholds 200/500/1000/2000 ms. Main simulator experiments commonly pair TTFT 200/500/1000 ms with TPOT 100/200/400 ms. The default deployed simulator gate is nevertheless `rho`, not direct latency-SLO admission.
- The local analysis environment is Python 3.9.6 with numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.13.1, joblib 1.5.3, and matplotlib 3.9.4. Cluster logs show vLLM 0.15.1. No requirements/lock file currently exists.

## Hardware and topology assumptions

- Pool A: 4xNVIDIA L40S, frequencies 210-2520 MHz, modeled TDP 350 W/GPU, modeled powered-idle 90 W/GPU, TP={1,2,4}.
- Pool B: 8xNVIDIA L4, frequencies 210-2040 MHz, modeled TDP 72 W/GPU, modeled powered-idle 18 W/GPU, TP={1,2,4,8}.
- Memory clock is measured but not controlled independently: 9001 MHz L40S and 6251 MHz L4 in `run_disagg_benchmark.sh`; `paper/COVERAGE_AUDIT.md` states it maps one-to-one to the core-frequency sweep.
- Testbed node names include `neptune`/`uranus` (L40S) and `europa`/`callisto`/`ganymede`/`io` (L4). Experiment E uses two SLURM nodes and forces NCCL sockets (`NCCL_IB_DISABLE=1`), so the measured commodity-network P/D result is not treated as representative of a production disaggregated fabric.
- Simulator KV latency is analytical: `kv_transfer_ms_per_token * input_len` only for cross-GPU-type routes. Paper sweeps use 5/16/41 microseconds/token. The default field is 0.059 ms/token (59 microseconds/token), despite a misleading inline `us/token` comment in `SweepLLMConfig`.
- KV-transfer energy is not modeled. `total_energy_j` is predicted GPU power times the 5-second window; interconnect energy is explicitly outside the current objective.

## Prediction models

Canonical models are selected by `paper/paths.py::paper_model_dir()` under `artifacts/paper/models/`, not by the older root `models_*` directories.

| Model/artifact | Prediction/role | Main features | Target/data | Where used |
|---|---|---|---|---|
| `capacity_model.pkl` | Maximum token throughput surrogate | log IL, log OL, TP, normalized frequency, decode fraction | Log maximum observed TPS by (IL,OL,TP,freq); plateau-confirmed groups preferred | `EnergyScheduler._compute_features()` to derive capacity and rho |
| `cap_groups.pkl` | Exact profiled-cell plateau map and predicted capacity | IL, OL, TP, freq | Grouped Phase-2 table | Applies ETA=0.8 to non-plateau capacity |
| `slo_classifier_<SLO>.pkl` | Legacy unified probability of P99-TTFT violation | lengths, TP, freq, rate/demand, load/rho interactions | Binary P99-TTFT violation | Legacy `recommend()` and compatibility path |
| `regressor_p99.pkl` | Legacy unified P99 TTFT ranking | lengths, TP, freq, rate/demand/rho | log1p P99 TTFT in unsaturated region | Legacy scheduler path |
| `power_model.pkl` | Legacy per-GPU power | lengths, TP, freq, decode fraction, rho | measured `avg_power_w / TP` | Legacy scheduler path |
| `prefill_slo_classifier_<SLO>.pkl` | P-prefill safety probability | IL/OL, TP, freq, rate, prefill/decode demand, prefill/total rho | P99 TTFT violation | Optional strict admission and diagnostics |
| `prefill_latency_regressor_p99.pkl` | Prefill P99 TTFT ranking | IL/OL, TP, freq, rate, prefill demand/rho | P99 TTFT | `predict_prefill_config()` and joint search |
| `prefill_power_model.pkl` | Prefill per-GPU power | IL/OL, TP, freq, rate, prefill/total rho | measured power/GPU | Joint energy score |
| `decode_slo_classifier_<SLO>.pkl` | D-decode safety probability | IL/OL, TP, freq, rate, decode demand, KV-pressure proxy, decode/total rho | P99 TPOT violation | Optional strict admission and diagnostics |
| `decode_latency_regressor_p99.pkl` | Decode P99 TPOT ranking | IL/OL, TP, freq, rate, decode demand/rho, KV-pressure proxy | P99 TPOT | `predict_decode_config()` and joint search |
| `decode_power_model.pkl` | Decode per-GPU power | IL/OL, TP, freq, rate, decode/total rho | measured power/GPU | Joint energy score |
| `decode_capacity.csv` | Goodput-derived decode feasibility by SLO | GPU, IL, OL, TP, freq, TPOT SLO | Stable/SLO-qualified decode TPS | Optional fail-closed `goodput` gate |
| `rho_envelope.csv` | Prefill utilization ceiling by SLO | GPU, TP, freq, TTFT SLO | calibrated maximum safe prefill rho | Optional fail-closed prefill envelope |

### Canonical model provenance

The canonical L40S and L4 stacks were trained from `Phase2_Results_L40S/master_results.csv` and `Phase2_Results_L4/master_results.csv`, respectively. These are **monolithic single-pool measurements**: one GPU pool executed both prefill and decode for each request. The corpus spans balanced and phase-dominant IL/OL shapes, but the training runner does not isolate a phase-dominant subset. “Phase-specific” below describes the prediction target and feature set, not the serving topology used to collect the training row.

| Canonical artifact family | Provenance class | What was actually measured |
|---|---|---|
| `prefill_slo_classifier_*`, `prefill_latency_regressor_p99`, `prefill_power_model` (both GPUs) | **Monolithic**, with balanced and phase-dominant shapes | P99 TTFT or measured aggregate GPU power normalized per TP rank from the full Phase-2 single-pool rows; no phase-only or cross-pool rows |
| `decode_slo_classifier_*`, `decode_latency_regressor_p99`, `decode_power_model` (both GPUs) | **Monolithic**, with balanced and phase-dominant shapes | P99 TPOT or measured aggregate GPU power normalized per TP rank from the same full Phase-2 single-pool rows; no phase-only or cross-pool rows |
| Shared `capacity_model`/`cap_groups` and legacy unified models | **Monolithic** | Total-token throughput, TTFT, and power from the same Phase-2 rows |
| `decode_capacity.csv` and `rho_envelope.csv` calibration tables | **Monolithic-derived** (not separately trained estimators) | Recomputed from the Phase-2 master tables and canonical capacity/rho predictions |

The later `CD_prefill_decode_only` **phase-only** measurements and `E_disaggregated` **physically disaggregated** measurements are not training inputs to any canonical estimator. They support diagnostics, trace-rate calibration, and physical feasibility evidence only. No canonical model is trained solely from a phase-dominant, phase-only, or physically disaggregated corpus.

The current canonical bundles contain 537 averaged L40S points/280 capacity groups and 594 averaged L4 points/261 capacity groups. Metadata reports phase latency rank correlations of about 0.94-0.97 and phase power MAPE around 6-8%, but absolute latency error is higher. L4 capacity new-shape CV MAPE is 50.22%, a material extrapolation warning.

Integrity note: `phase_model_validation_l40s.csv` and `phase_model_validation_l4.csv` inside the canonical directories still show older 241/202-point validation, while `training_metadata.json` records the newer 537/594-point training and validation rows. Treat the metadata as the current training record and regenerate/rename the stale standalone validation CSVs in a later cleanup task.

All canonical binaries are suitable for Git: current L40S and L4 bundles are about 6.9 and 7.2 MB; the largest individual estimator is about 1.9 MB. Training runtime/cost is not recorded. Refitting appears computationally modest; collecting equivalent hardware data is the expensive step.

## Scheduler reconstruction

The current scheduler is `paper/scripts/schedulers/sweep.py::SweepLLMStrategy`. `paper/scripts/sweep_llm_scheduler.py` is a compatibility re-export.

Inputs per 5-second decision window:

- Requests with arrival time, input length, and output length.
- TTFT/TPOT SLO pair (or legacy scalar SLO).
- Previous state/rate and current/target pool bundles.
- Predictor stacks for L40S and L4.
- Configuration thresholds, KV latency, gate mode, and search limits.

Outputs:

- Per request-class route: A->A, A->B, B->A, or B->B, where A=L40S and B=L4.
- One TP degree and one core frequency per GPU-type pool.
- A modeled allocation `(n_prefill_gpus, n_decode_gpus)` per pool and derived active instance counts.
- Predicted pool power/energy, per-class TTFT/TPOT/KV latency, feasibility, and search diagnostics.

Algorithm:

1. `_summarize_requests()` classifies requests using short/long IL and OL thresholds (default 512), aggregates per-class rate, and snaps representative lengths to the characterization grid.
2. `_classify_window()` labels the window BOTH_LOW, PREFILL_HEAVY, DECODE_HEAVY, or BOTH_HEAVY using normalized prefill/decode token demand, hysteresis, and burst detection.
3. `_construct_candidates()` enumerates/prunes pool bundles and frequency pairs. A `PoolBundle(tp,n_pf,n_dc)` represents a uniform TP for all modeled instances in that GPU-type pool.
4. `_search_candidate_routes()` performs bounded depth-first branch-and-bound over per-class routes for each candidate.
5. `_predict_pool_details()` divides pool phase load evenly over the modeled instance count and queries phase-specific power/rho models.
6. `_predict_class_metrics()` adds analytical cross-pool KV latency and applies the selected feasibility gate. Default `rho` checks utilization and predictor validity; stricter classifier/latency and hardware goodput/envelope gates exist but are optional.
7. `_evaluate_candidate()` minimizes predicted **GPU power**, equivalently `power * window_s` energy for the fixed window duration. There is no separate KV/interconnect energy term.
8. Fast search uses current bundles; ideal search can run every window or on triggers. Bundle stability counters model delayed capacity changes, but no runtime actuation occurs.

This is an online-shaped controller evaluated offline. The unoptimized Python controller can make per-window decisions and reports timing, but it is not wired to live vLLM servers, queues, KV occupancy, or endpoint health. `cluster` and `current_time` are explicitly discarded in the current `decide_window()` implementation; queueing and live KV-cache occupancy are not scheduler inputs.

## Existing policies

- `StaticDisaggStrategy`: fixed class route A->B, TP1, all four L40S prefill GPUs, all eight L4 decode GPUs, maximum clocks.
- `GreenLLMStrategy`: same fixed route/layout, exhaustive frequency-only oracle.
- `DynamoLLMMono`: joint search restricted to monolithic A->A or B->B routes; explicitly an inspired/strong baseline, not a faithful DynamoLLM implementation.
- `DualScaleStrategy`: 5-minute coarse placement based on trailing peak plus per-window frequency sweep; inspired extension, not a faithful system reproduction.
- `HierarchicalDisaggStrategy`: sequential route -> TP -> DVFS -> capacity baseline with all four route types.
- `SweepLLMStrategy`: class-aware joint route/TP/frequency/active-capacity search.
- Ablations: `SweepLLMNoRoutingStrategy`, `SweepLLMNoDVFSStrategy`, `SweepLLMNoTPStrategy`, and `SweepLLMNoCapacityStrategy`.
- Motivation-only restricted policies in `motivation_joint_core.py`: `static`, `route_only`, `route_dvfs`, `route_dvfs_tp`, `full_joint`, and `sequential`. These exhaustively evaluate progressively enlarged model-level search spaces; `sequential` commits route -> DVFS -> TP -> active GPUs.

In all of these, “routing” means selecting a **GPU-type pool per phase for each length class**. It does not select a physical server URI or a particular P0/P1/P2/D0/D1/D2 instance.

## Physical vLLM implementation

`run_disagg_benchmark.sh` is substantial and reusable. It:

- Launches vLLM OpenAI servers with explicit `CUDA_VISIBLE_DEVICES`, `--tensor-parallel-size`, and core clocks.
- Runs monolithic L40S/L4 sweeps across TP and frequency.
- Runs phase-dominant prefill/decode profiling on either GPU type, including configurable TP in the Mistral-Nemo pilot.
- Generates a FastAPI proxy for sequential prefill, synchronous P2P NCCL KV PUT, then decode; embeds producer/consumer addresses in the request ID.
- Monitors per-GPU NVML power/energy and records benchmark windows.
- Contains version-specific diagnostic patches for vLLM 0.15.1 P2P connector behavior.

Experiment E is nevertheless fixed: `local tp=1` is passed to both servers; one prefill and one decode address are used; no endpoint pool/router exists; P TP != D TP is not exercised; and no scheduler decision is applied. The completed raw corpus also shows very large TTFT on several fixed-pair runs, and the paper correctly treats commodity-network cross-pool performance as not hardware-validated for the analytical scheduler results.

## Evaluation and result entry points

- `smoke_test_strategies.py`: compact end-to-end model/simulator smoke test.
- `paper/scripts/jsep_simulator.py`: 5-second window simulation and summary metrics.
- `paper/scripts/sweep_llm_scheduler.py`: backward-compatible scheduler API.
- `paper/scripts/benchmark_sweep_*.py`: variant/SLO/trigger sweeps.
- `paper/scripts/motivation_joint_core.py`: restricted and full joint policy evaluation.
- `paper/scripts/search_obs2_regimes.py`: exhaustive Observation-2 regime search; source of the largest generated CSVs.
- `paper/scripts/calibrate_decode_capacity.py`, `calibrate_rho_envelope.py`: optional hardware-feasibility artifacts.
- `paper/scripts/evaluate_phase_decision_quality.py`, `cross_model_validation.py`, `oracle_comparison.py`: model/scheduler decision validation.
- `paper/scripts/figures/`: publication plot generators.
- `results/paper/section42_frozen_main/summary.csv`: synthetic cross-strategy summary.
- `results/paper/prod_{conv,code,diurnal}_frozen/summary.csv`: Azure production replays.
- `results/paper/section45_ablation/summary.csv`: control-knob ablations.
- `results/paper/section47_overhead/summary.csv`: scheduler overhead.

## Git and security audit

- Current branch/history/remote: none, because this directory is not a Git repository.
- Tracked, modified, untracked, ignored, or historically large files: not defined. No historical large-object analysis is possible without `.git` metadata.
- The audit scanned source/configuration text (excluding raw results, `.venv`, data, PDFs) for common private-key, cloud key, bearer token, GitHub token, and API-secret patterns; no match was found.
- `.claude/settings.json` enables a local bypass-permissions mode and should not be published. Cluster hostnames, SLURM details, and job IDs occur in scripts/logs; hostnames are needed by the experiment template, while logs should remain private.
- No guarantee can be made from a pattern scan alone. Run a dedicated secret scanner on the exact staged set before the first commit.

## Ambiguities and preservation warnings

- The canonical artifact location and older root model snapshots differ; do not delete either before provenance is settled.
- Standalone phase validation CSVs appear stale relative to current training metadata.
- `make -C paper check-paths` currently fails because three expected generated result sources are absent: `results/paper/figures/obs2_regime_search_primary/r2_ttft500_tpot200_mixedlen_obs2_main.pdf`, `results/paper/figures/section42/synthetic_trace_overview.pdf`, and `results/paper/figures/section42/tau_kv_sensitivity.pdf`. Publication copies may exist elsewhere, but the canonical source-path invariant is not currently clean.
- The `kv_transfer_ms_per_token` inline unit comment is inconsistent with its name/use.
- The default simulator feasibility gate is utilization-based, so nominal SLO sensitivity is limited; paper text already identifies this as a modeling boundary.
- `active_gpus=0`, activation/deactivation, and idle power are mathematical controls. No warm/cold lifecycle, load time, draining, or reconfiguration cost is applied to real instances.
- `PoolBundle` imposes one TP value per GPU-type pool. It cannot represent simultaneous TP1 and TP2 instances within the same pool.
- A one-window `smoke_test_strategies.py` run loaded every canonical model successfully but was manually stopped after more than 90 seconds while the DynamoLLM-Mono ideal search was still evaluating. This is consistent with the recorded unoptimized-search overhead; it is not a syntax/model-load failure, but the smoke test is not actually “quick” with its current default strategy set.
- Historical files were not deleted, renamed, or rewritten during this audit.
