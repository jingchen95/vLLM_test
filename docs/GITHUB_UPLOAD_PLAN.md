# GitHub Upload Plan

## Target

Create a compact research repository that explains and runs the modeling/simulation pipeline, preserves small trained predictors and representative results, and points to externally archived raw hardware data. Do not attempt to turn the current 28 GB directory into a single blind commit.

The proposed manifest is `github_manifest.txt`; external material is cataloged in `external_artifacts_manifest.tsv`; `.gitignore` is a safe-add guardrail. A temporary Git-index validation selected **613 files totaling 34.75 MiB**, with no selected file larger than about **1.9 MiB**, versus the current **28 GB** working tree.

## MUST INCLUDE

- Root benchmark/monitor/parser/calibration/replay scripts.
- `paper/scripts/`, including `schedulers/`, predictor training, simulator, traces, evaluation, and figure code.
- Paper LaTeX source, bibliography, research notes, and collaborator-facing docs.
- Canonical model bundles and hardware feasibility tables in `artifacts/paper/models/`.
- Compact characterization tables:
  - `Phase2_Results_L40S/master_results.csv`
  - `Phase2_Results_L4/master_results.csv`
  - `Obs2_Validation_L4/obs2_validation_results.csv`
- `calibration_out/` compact tables.
- Compact Azure replay inputs in `traces/azure_production/` and `traces/azure_diurnal/`.
- `results/paper/analyses/`, `results/paper/synthetic_traces/`, and representative frozen `summary.csv` files.
- Publication-sized figures under `paper/figures/`.
- This audit, gap analysis, handoff, manifests, and `.gitignore`.

## SHOULD INCLUDE

- Split-model validation bundles (`models_l40s_A/B`, `models_l4_A/B`) already under `artifacts/paper/models/`; they are small and support cross-model robustness results.
- Small B5 predictors and February snapshots under the same artifact directory, provided their provenance is explained rather than presented as current defaults.
- Mistral-Nemo-12B pilot scripts, matrices, JSON summaries, and checklists under `paper/`; raw pilot measurements stay external.
- Final paper PDF only as an optional GitHub Release asset, not as a source-controlled build product.
- A future `requirements.txt` or lock file and a cluster software manifest. Neither currently exists; do not infer vLLM/CUDA dependencies solely from the macOS `.venv`.

## EXCLUDE FROM GIT

- `.venv/`, Python caches, editor/agent-local state, `.DS_Store`, and the macOS `Icon` resource.
- `paper/build/` and LaTeX intermediates.
- `logs/`, partial outputs, core dumps, temporary files.
- Hugging Face/vLLM caches and model weights (`*.safetensors`, checkpoints, downloaded weights).
- Raw benchmark/NVML monitor files under Phase-2, Observation-2, physical disaggregation, trace sanity, and model-transfer pilot directories.
- Downloadable Azure source CSVs under `data/` and expanded prompt JSONs.
- `results/paper/figures/*_evaluations.csv` and the entire 22 GB generated search-output tree; retain publication copies under `paper/figures/` and code to regenerate them.
- `archive/` and related-work PDFs. Preserve elsewhere; redistribution rights for papers may not exist.
- Older root `models_l40s/` and `models_l4/` snapshots in the initial curated Git commit. The scheduler defaults to canonical `artifacts/paper/models/models_*`; archive the old roots until provenance is resolved.
- `artifacts/paper/analyses/scheduler_query_log.csv`; derived coverage is already documented in `paper/SCHEDULER_COVERAGE.md`.

## EXTERNALIZE AND DOCUMENT

Use an immutable research-data repository such as Zenodo/OSF for a public release, or institutional S3/object storage while the work is private. Publish one versioned archive per logically coherent corpus, not a single opaque 27 GB tarball:

1. L40S Phase-2 raw hardware corpus.
2. L4 Phase-2 raw hardware corpus.
3. Fixed physical vLLM P/D/monolithic corpus.
4. Trace-sanity and Observation-2 validation corpora.
5. Mistral-Nemo-12B transfer-pilot corpus.
6. Optional generated exhaustive scheduler-search tables.

For each archive add SHA-256 checksums, creation command/job script, software/hardware metadata, schema, and the Git commit that produced or consumed it. Do not use Git LFS as a dumping ground for high-churn generated CSVs; a data release or DVC-style remote is more appropriate. GitHub Releases are acceptable for small immutable supporting bundles.

## Trained model decision

Commit the trained predictors. The canonical L40S and L4 bundles are about 6.9 MB and 7.2 MB, and no estimator is larger than about 1.9 MB. They are required to run scheduler queries and trace replay without retraining. Keep, together:

- Estimator/scaler/polynomial `.pkl` files.
- Feature-list `.pkl` files.
- `config.pkl`.
- `cap_groups.pkl`.
- `training_metadata.json`.
- Validation/guard-band summaries.
- Training code and compact source `master_results.csv` tables.

The expensive asset is the measurement campaign, not the small binary model. Note that pickle/joblib files are executable deserialization formats; collaborators should load only trusted repository artifacts.

## Large-file guardrails

- GitHub rejects individual files over 100 MB. The current tree contains dozens of approximately 300 MB evaluation CSVs and a 559 MB Azure CSV.
- Before every initial commit, inspect both the staged set and object sizes. Suggested checks:

```bash
git status --short
git diff --cached --stat
find . -type f -size +50M -not -path './.git/*' -print
git ls-files -z | xargs -0 stat -f '%z %N' | sort -nr | head -50
```

- After the first commit, use `git count-objects -vH`. If an unsuitable large file is accidentally committed but not pushed, fix the commit immediately. If already shared, coordinate a history rewrite rather than running one unilaterally.

## Secret and privacy guardrails

The audit's source/config scan found no obvious private keys, cloud keys, bearer tokens, GitHub tokens, or API secrets. This is not a complete security proof.

- Exclude `.claude/`, runtime logs, SSH material, credentials, and user-specific paths.
- Review cluster hostnames and SLURM account/partition values before making the repository public. Hostnames are embedded in `run_disagg_benchmark.sh`; they are useful defaults but disclose testbed naming.
- The public Azure dataset contains token counts/timestamps, not prompt content in the compact trace, but verify the license and citation in the final README.
- Run a dedicated scanner on staged files, for example `gitleaks detect --no-git --source .`, before committing.
- Treat `.pkl` files as trusted-code artifacts and document their source commit/checksum in releases.

## Git state and preparation status

At audit time this directory has no `.git` metadata. Therefore:

- Current branch: none.
- Untracked/modified/ignored files: not defined until `git init`.
- History and large historical objects: none available to inspect.
- Remote: none.
- Existing `.gitignore`: none; this audit created the proposed root `.gitignore`.

No repository initialization, commit, remote creation, push, or history rewrite was performed.

## Exact next action

First archive the current directory or ensure the raw artifacts have another verified copy. Then initialize Git and stage only the manifest-selected set:

```bash
cd /Users/chjing/Downloads/vLLM_test
git init
git switch -c main
git add .gitignore README.md docs github_manifest.txt external_artifacts_manifest.tsv
git add calibrate_trace_rates.py gpu_monitor.py parse_disagg_results.py plot_section42.py plot_section45.py prepare_azure_trace.py run_disagg_benchmark.sh run_rhocap_ablation.py run_trace_sanity_suite.sh smoke_test_strategies.py summarize_trace_sanity.py validate_obs2_l4.sh
git add paper artifacts/paper/models calibration_out Phase2_Results_L40S/master_results.csv Phase2_Results_L4/master_results.csv Obs2_Validation_L4/obs2_validation_results.csv traces/azure_production traces/azure_diurnal results/paper/analyses results/paper/synthetic_traces
git add results/paper/section42_frozen_main/summary.csv results/paper/prod_conv_frozen/summary.csv results/paper/prod_code_frozen/summary.csv results/paper/prod_diurnal_frozen/summary.csv results/paper/section45_ablation/summary.csv results/paper/section47_overhead/summary.csv
git status --short
git diff --cached --stat
find . -type f -size +50M -not -path './.git/*' -print
```

Review the staged list and secret scan before committing. Do not run `git add -f`, `git add -A`, or push until that review is clean.
