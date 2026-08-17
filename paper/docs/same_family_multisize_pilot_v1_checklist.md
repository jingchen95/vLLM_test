# Same-Family Multi-Size Pilot v1 Checklist

## Scope
This is the first formal same-family multi-size pilot for `mistral_nemo_12b_pilot`.

It is intended to answer:
- whether zero-shot reuse degrades on the larger same-family model
- whether decode, especially on `L4`, is the first meaningful bottleneck
- whether limited new-size profiling is enough to make `oracle` clearly more useful than `zero_shot`
- whether `threshold_adapt` is worth keeping

This is not a full transfer study.

## Matrix
Generated files:
- `paper/same_family_multisize_mistral_nemo12b_pilot_v1_matrix.csv`
- `paper/same_family_multisize_mistral_nemo12b_pilot_v1_summary.json`

Run counts:
- `prefill_l40s = 8`
- `prefill_l4 = 16`
- `decode_l40s = 16`
- `decode_l4 = 32`
- `total = 72`

Result directory:
- `results/same_family_multisize_mistral_nemo12b_pilot_v1`

## Files To Upload To Cluster
Upload these to `~/vLLM_test/` on the cluster root:

- `paper/same_family_multisize_mistral_nemo12b_pilot_v1_collect.sh`
- `paper/same_family_multisize_mistral_nemo12b_pilot_v1_collect_sbatch.sh`
- `paper/same_family_multisize_mistral_nemo12b_pilot_v1_load_sanity.sh`
- `paper/same_family_multisize_mistral_nemo12b_pilot_v1_load_sanity_sbatch.sh`
- `paper/same_family_multisize_mistral_nemo12b_pilot_v1_pipeline.sh`
- `paper/same_family_multisize_mistral_nemo12b_pilot_v1_pipeline_sbatch.sh`
- `paper/same_family_multisize_mistral_nemo12b_pilot_v1_matrix.csv`
- `paper/same_family_multisize_mistral_nemo12b_pilot_v1_summary.json`
- `paper/summarize_same_family_multisize_pilot.py`
- `run_disagg_benchmark.sh`

Notes:
- The current cluster flow should be treated as `load sanity + raw collection`.
- Do not rely on cluster-side summarization as the only copy of the summary outputs.
- Pull raw results back and summarize locally.

## Pre-Submission Checks
- Ensure the old dirty directory is gone on cluster:
  - `results/same_family_multisize_mistral_nemo12b_pilot_smoke`
- Ensure the new directory does not already contain old data:
  - `results/same_family_multisize_mistral_nemo12b_pilot_v1`
- Ensure the uploaded scripts are in `~/vLLM_test/`, not under a missing `paper/` directory
- Ensure `summarize_same_family_multisize_pilot.py` is present in `~/vLLM_test/`

## Recommended Submission Path
Submit the pipeline job:

```bash
cd ~/vLLM_test
sbatch same_family_multisize_mistral_nemo12b_pilot_v1_pipeline_sbatch.sh
```

The pipeline does:
1. load sanity on `L40S TP=1`
2. load sanity on `L4 TP=2,4`
3. abort on any sanity failure
4. run raw collection only if sanity passes

## What To Check When The Job Finishes
Check the main job log:

```bash
cd ~/vLLM_test
tail -n 120 logs/same_family_multisize_pipeline_<jobid>.log
```

Look for:
- any `FAIL` in load sanity
- `vLLM server failed to start`
- missing result files
- benchmark crashes

Then check raw result counts:

```bash
find results/same_family_multisize_mistral_nemo12b_pilot_v1/CD_prefill_decode_only -name '*.txt' | wc -l
find results/same_family_multisize_mistral_nemo12b_pilot_v1/CD_prefill_decode_only -name 'monitor_*.csv' | wc -l
```

Expected `.txt` count:
- `72`

Spot-check that result files include:
- timing window start/end
- measured average power
- measured energy
- `Successful requests`
- `Failed requests`

## Files To Pull Back
Pull these back to local:

- `results/same_family_multisize_mistral_nemo12b_pilot_v1/`
- `logs/same_family_multisize_pipeline_<jobid>.log`
- `same_family_multisize_mistral_nemo12b_pilot_v1_load_sanity_results.csv`

The load sanity CSV is written in the repo root, not under `results/`.

## Local Post-Processing
Run summarization locally:

```bash
python3 paper/summarize_same_family_multisize_pilot.py \
  --results-dir results/same_family_multisize_mistral_nemo12b_pilot_v1/CD_prefill_decode_only \
  --model-id mistral_nemo_12b_pilot \
  --hf-name mistralai/Mistral-Nemo-Instruct-2407 \
  --family mistral \
  --param-count-b 12 \
  --output-prefix results/same_family_multisize_mistral_nemo12b_pilot_v1/same_family_multisize_mistral_nemo12b_pilot_v1
```

Then run evaluation locally:

```bash
source ~/OneDrive\ -\ Chalmers/EnergyEfficientCUDASTF/samples/figures/myenv/bin/activate
python3 paper/scripts/evaluate_same_family_multisize_pilot.py \
  --pilot-csv results/same_family_multisize_mistral_nemo12b_pilot_v1/same_family_multisize_mistral_nemo12b_pilot_v1_phase_measurements.csv \
  --model-dir-l40s models_l40s \
  --model-dir-l4 models_l4 \
  --output-prefix results/same_family_multisize_mistral_nemo12b_pilot_v1/same_family_multisize_mistral_nemo12b_eval_v1
```

## Primary Readout
Inspect these first:
- `*_mode_feasibility_summary.csv`
- `*_mode_prediction_summary.csv`
- `*_mode_decision_summary.csv`
- `*_mode_comparison.csv`
- `*_summary.txt`

Questions to answer:
- Does `zero_shot` stay safe?
- Is `decode / L4` still the hardest slice?
- Does `oracle` clearly improve decision quality over `zero_shot`?
- Does `threshold_adapt` still over-prune or reject too much?

## Interpretation Rule
Use `pilot_v1` to decide whether same-family multi-size is promising.

Do not claim:
- full same-family transfer support
- robust threshold-only adaptation
- cross-family generalization

The right target claim, if the data support it, is:
- same-family multi-size appears plausible with limited additional profiling, and decode-side calibration is the main remaining bottleneck
