# Handoff for ChatGPT and Collaborators

## 1. Research goal

This repository studies energy-efficient LLM serving on a heterogeneous cluster with 4xNVIDIA L40S and 8xNVIDIA L4 GPUs. The next direction is real vLLM prefill/decode (P/D) disaggregation with independent P and D pools, pre-instantiated TP1/TP2 endpoints, independent P and D routing, and DVFS. The target objective is `E_prefill + E_KV_transfer + E_decode` under TTFT and TPOT SLOs.

Do not assume the project is new. A substantial model-based joint scheduler and evaluation pipeline already exists. The recommended path is to preserve and extend it, then build the missing runtime control plane.

## 2. Current implementation status

- Complete L40S/L4 hardware characterization tables over length, rate, TP, and core frequency.
- Phase-specific capacity, P99 TTFT, P99 TPOT, SLO-classifier, and per-GPU power models.
- SWEEP-LLM: per-class joint search over GPU-type phase routes, TP, frequency, and modeled active GPU capacity.
- Static, frequency-only, monolithic, two-tier, sequential, and single-knob ablation policies.
- Synthetic and Azure trace replay, validation, sensitivity, overhead, plotting, and paper pipeline.
- Physical vLLM P/D benchmark using P2pNcclConnector, but only one fixed L40S producer and one fixed L4 consumer, both TP1.
- No live multi-endpoint XpYd router, endpoint registry, queue/KV telemetry, unequal-P/D-TP validation, or scheduler-to-vLLM actuation.

Bottom line: **reuse models/data/search/evaluation; refactor resource representation; build the runtime layer.**

## 3. Hardware, model, and workloads

- L40S pool: 4 GPUs; TP={1,2,4}; modeled frequencies 210-2520 MHz; memory clock fixed at 9001 MHz.
- L4 pool: 8 GPUs; TP={1,2,4,8}; modeled frequencies 210-2040 MHz; memory clock fixed at 6251 MHz.
- Primary model: `mistralai/Mistral-7B-v0.1`, BF16. Mistral-Nemo-12B has a limited phase-only transfer pilot, not a production model stack.
- Synthetic classes: SS=(64,128), SL=(64,512), LS=(1024,128), LL=(1024,512). T1/T2/T3/T4 are generated in `paper/scripts/jsep_traces.py`.
- Production inputs: Azure conversation/code traces prepared by `prepare_azure_trace.py`; compact replays live in `traces/azure_production/` and `traces/azure_diurnal/`.
- Main evaluation usually uses a 5-second control window and TTFT/TPOT pairs (200/100, 500/200, 1000/400 ms), but the frozen primary comparison admits with `rho<=1`, not direct latency prediction.
- Local analysis env: Python 3.9.6, numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.13.1, joblib 1.5.3. Cluster logs show vLLM 0.15.1.

## 4. Important source files

| File | Key symbols / purpose |
|---|---|
| `run_disagg_benchmark.sh` | Launch/clock/monitor vLLM; `run_experiment_A/B/CD/E`; embeds generated FastAPI P/D proxy and vLLM 0.15.1 P2P diagnostics |
| `gpu_monitor.py` | NVML power/utilization/energy sampling |
| `parse_disagg_results.py` | Parse B5 benchmark and monitor files into compact lookup rows |
| `paper/scripts/model_training_common.py` | Phase/common feature definitions, capacity/rho construction, dataset preparation |
| `paper/scripts/model_fitting_runner.py` | `run_training_pipeline()`; fits capacity, latency, power, and SLO guards |
| `paper/scripts/scheduler.py` | `EnergyScheduler`; loads canonical models and predicts/recommends TP/frequency |
| `paper/scripts/schedulers/common.py` | `SweepLLMConfig`, route constants, `PoolBundle`, `SearchCandidate`, result types |
| `paper/scripts/schedulers/sweep.py` | `SweepLLMStrategy.decide_window()` and joint candidate/route search |
| `paper/scripts/schedulers/factory.py` | Creates SWEEP and baseline strategies |
| `paper/scripts/schedulers/ablations.py` | No-routing, no-DVFS, no-TP, no-capacity variants |
| `paper/scripts/schedulers/baselines_*.py` | Static, GreenLLM-inspired, DynamoLLM-Mono, DualScale-Ext, Hierarchical-Disagg |
| `paper/scripts/jsep_cluster.py` | 4xL40S/8xL4 simulator constants and modeled cluster state |
| `paper/scripts/jsep_traces.py` | Synthetic trace generators and Azure loader |
| `paper/scripts/jsep_simulator.py` | 5-second discrete-window simulation and summary |
| `paper/scripts/motivation_joint_core.py` | `static`, `route_only`, `route_dvfs`, `route_dvfs_tp`, `full_joint`, `sequential` restricted searches |
| `paper/scripts/sweep_llm_scheduler.py` | Compatibility re-export; implementation is under `schedulers/` |
| `paper/scripts/replay_synthetic_trace.py` | Timed HTTP hardware trace replay |
| `paper/scripts/calibrate_decode_capacity.py` | Builds SLO-qualified decode goodput table |
| `paper/scripts/calibrate_rho_envelope.py` | Builds SLO-conditioned prefill rho envelope |
| `paper/paths.py` | Canonical artifact/result locations |

## 5. Existing models

Canonical bundles are `artifacts/paper/models/models_l40s/` and `artifacts/paper/models/models_l4/`; older root `models_*` directories are not the defaults.

Each canonical stack contains:

- Gradient-boosted capacity model plus `cap_groups` plateau evidence.
- Legacy unified SLO/TTFT/power stack.
- Phase-specific prefill P99-TTFT classifier/regressor/power model.
- Phase-specific decode P99-TPOT classifier/regressor/power model.
- Feature lists, scalers, polynomial transforms, configuration, guard thresholds, and training metadata.

Model provenance: all canonical prefill and decode classifiers, latency regressors, and power estimators were fit from the monolithic single-pool Phase-2 master tables. Those tables include balanced and phase-dominant request shapes, but the models use the full corpus; “phase-specific” refers to TTFT-versus-TPOT targets and features. The later phase-only `CD_prefill_decode_only` and physical `E_disaggregated` measurements were not used to train the canonical stack. See `CODEBASE_AUDIT.md` for the artifact-by-artifact matrix.

The scheduler computes demand `r * (IL + gamma*OL)` with gamma=0.3, derives prefill/decode rho from predicted capacity, and applies ETA=0.8 to non-plateau cells. Optional fail-closed artifacts are `decode_capacity.csv` and `rho_envelope.csv`.

The binaries are small enough to commit (about 14 MB for the two current canonical stacks; largest estimator about 1.9 MB). L4 new-shape capacity extrapolation is weak (metadata CV MAPE 50.22%); avoid treating off-grid predictions as ground truth.

## 6. Scheduler design and routing semantics

`SweepLLMStrategy` groups a window into IL/OL classes, labels its bottleneck state, enumerates candidate pool bundles/frequencies, and uses bounded DFS branch-and-bound to assign one of four routes per class:

- A->A = L40S prefill and decode.
- A->B = L40S prefill, L4 decode.
- B->A = L4 prefill, L40S decode.
- B->B = L4 prefill and decode.

It minimizes predicted GPU power (equivalent to energy over a fixed 5-second window) subject to its configured gate. Cross-pool TTFT adds `kv_transfer_ms_per_token * input_len`; transfer energy is not charged.

Critical caveat: A/B are abstract GPU-type pools, not server endpoints. `PoolBundle(tp,n_pf,n_dc)` has a single TP for every modeled instance in a pool. `active_instances` is arithmetic (`n_gpus / tp`), not a count of live vLLM processes. Queue state, batch, KV occupancy, endpoint health, and actual reconfiguration are absent.

## 7. Existing results

- Synthetic cross-strategy frozen summary: `results/paper/section42_frozen_main/summary.csv`.
- Azure conversation/code/diurnal: `results/paper/prod_conv_frozen/summary.csv`, `prod_code_frozen/summary.csv`, `prod_diurnal_frozen/summary.csv`.
- Ablations: `results/paper/section45_ablation/summary.csv`.
- Scheduler overhead: `results/paper/section47_overhead/summary.csv`.
- Model/scheduler analyses: `results/paper/analyses/`.
- Publication figures: `paper/figures/`; generators: `paper/scripts/figures/`.
- Physical raw P/D status: `results/disagg_20260321_v2/E_disaggregated/status.txt` says `disagg_complete`; raw corpus should be externally archived, not committed.
- Coverage caveats: `paper/COVERAGE_AUDIT.md` and `paper/SCHEDULER_COVERAGE.md`.

The paper's primary results are simulator/model outcomes. Do not present them as an end-to-end online hardware scheduler evaluation.

## 8. Relationship to the new XpYd + TP x DVFS direction

High overlap:

- Exact target GPU types/counts.
- TP/frequency measurement and learned interaction.
- Phase-specific TTFT/TPOT/power/capacity modeling.
- Heterogeneous prefill/decode pool assignment and joint search concepts.
- Traces, policies, validation, and plotting.
- Basic vLLM P/D/KV launch experience.

Low overlap / missing:

- Simultaneous P0/P1/P2 and D0/D1/D2 processes.
- Independent concrete endpoint choice.
- Mixed TP instances within one pool.
- Physical P TP != D TP support.
- Live queues, batches, KV occupancy, and health.
- DVFS/activation actuation and transition costs.
- KV/interconnect energy and contention.

## 9. Known gaps and correctness cautions

- Main default feasibility is rho-based, so TTFT/TPOT SLO sensitivity is not fully realized.
- No E_KV term; only analytical latency for inter-GPU-type transfer.
- No representative-fabric physical validation; current P2P script uses commodity-network NCCL and version-specific installed-source patches.
- Physical Experiment E hard-codes both endpoints TP1.
- One TP per GPU-type pool prevents the proposed TP1+TP1+TP2 simultaneous topology.
- Active/reserve/power-gating controls are mathematical only.
- Canonical standalone `phase_model_validation_*.csv` files appear older than their `training_metadata.json` records.
- `SweepLLMConfig.kv_transfer_ms_per_token` is used in milliseconds/token; one comment incorrectly labels the 0.059 default as microseconds/token.
- No Git history and no dependency lock currently exist.
- `make -C paper check-paths` currently reports three missing generated result-source PDFs (Observation-2 main, synthetic-trace overview, and tau-KV sensitivity); do not claim a clean publication-path check until they are restored or regenerated deliberately.
- The full `smoke_test_strategies.py` strategy set is slow: a one-window audit run was stopped after more than 90 seconds during DynamoLLM-Mono search. Model loading and a direct predictor query succeeded.

## 10. Recommended next steps

1. Publish the curated snapshot using `github_manifest.txt`; put raw corpora in versioned external storage with checksums.
2. Add a reproducible dependency/cluster software manifest and reconcile model validation provenance.
3. Introduce a concrete endpoint inventory schema rather than extending `PoolBundle` ad hoc.
4. Extract the generated P/D proxy into reviewed source and use a supported vLLM connector path.
5. Bring up P0/P1/P2 and D0/D1/D2 with fixed clocks; validate endpoint routing and KV correctness.
6. Test every intended P-TP/D-TP pair and encode connector compatibility as a hard constraint.
7. Add live telemetry and explicit TTFT/TPOT admission; then add DVFS and later warm/cold reserve control.
8. Extend the energy objective with transfer/network and transition energy.
9. Compare simulator decisions with actual per-request hardware outcomes.

Do not implement the new scheduler until these interfaces and evidence boundaries are agreed.

## 11. Representative reproduction commands

From repository root:

```bash
source .venv/bin/activate

# Check canonical path layout. At this snapshot it is expected to report the
# three missing generated figure sources documented above.
make -C paper check-paths

# Query the L40S predictor stack.
PYTHONPATH=paper/scripts python paper/scripts/scheduler.py \
  --mode query --il 1024 --ol 128 --rate 10 --slo 500 \
  --model-dir artifacts/paper/models/models_l40s

# Query L4 by changing model directory.
PYTHONPATH=paper/scripts python paper/scripts/scheduler.py \
  --mode query --il 128 --ol 512 --rate 5 --slo 500 \
  --model-dir artifacts/paper/models/models_l4

# End-to-end simulator/model smoke test. Even one window may take minutes in
# the unoptimized Python joint-search baselines.
python smoke_test_strategies.py --trace results/paper/synthetic_traces/T1.csv \
  --slo 500 --num-windows 1

# Refit canonical predictor stacks from compact master tables.
PYTHONPATH=paper/scripts python paper/scripts/model_fitting.py
PYTHONPATH=paper/scripts python paper/scripts/model_fitting_l4.py

# Build/publish the paper (requires LaTeX for the PDF step).
make -C paper publish-figures
make -C paper pdf

# Physical cluster benchmark examples; review node/GPU/env settings first.
sbatch run_disagg_benchmark.sh
sbatch --export=ALL,EXP=E,E_SMOKE=1 run_disagg_benchmark.sh
```

Refitting overwrites canonical model artifacts and should only be done deliberately after preserving the current bundle. Physical commands require the original SLURM/GPU environment, vLLM stack, model access, and clock-control permissions.

For full audit details, read `CODEBASE_AUDIT.md`, `RESEARCH_GAP_ANALYSIS.md`, and `GITHUB_UPLOAD_PLAN.md` first.
