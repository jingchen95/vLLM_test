# XpYd Phase 3B minimal physical-preflight runbook

The checked-in command is deliberately one small Uranus L40S P0 to Europa L4
D0 validation, not a workload sweep. Uranus is the accepted L40S endpoint
because its read-only NVML capability probe supports both cumulative energy and
direct power; Neptune's current driver/device state does not.

## CPU validation

From the repository root:

```bash
PYTHONPATH=paper/scripts python3 -m unittest discover -s tests -p 'test_xpyd*.py'
for repeat in 1 2 3 4 5; do
  PYTHONPATH=paper/scripts python3 -m unittest tests.test_xpyd_phase3b_energy.SamplerTests
done
PYTHONPYCACHEPREFIX=/tmp/xpyd-pyc python3 -m py_compile \
  paper/scripts/xpyd/nvml_readonly.py paper/scripts/xpyd/phase3b_energy.py
bash -n run_disagg_benchmark.sh
git diff --check
```

All NVML tests inject a CPU fake. The fake raises immediately if a device
setter/reset symbol is accessed.

## Submit exactly one preflight

On Minerva, with the repository at the reviewed commit:

```bash
cd /data/users/chjing/vLLM_test
source ~/.bashrc
conda activate cuda-env
git status --short
sbatch --nodelist=uranus,europa \
  --export=ALL,EXP=E,E_SMOKE=1,L40S_NODE=uranus,L4_NODE=europa,XPYD_PHASE3B_CONFIG=paper/configs/xpyd_phase3b_l40s_l4.json \
  run_disagg_benchmark.sh
```

`XPYD_PHASE3B_CONFIG` forces the existing no-GPU-mutation branch. It starts the
existing vLLM 0.15.1 P2pNcclConnector P0/D0/proxy stack, then launches one
read-only monitor per node. The sequence is a 1 s capability interval, about
10 s idle, one excluded IL128/OL128 warm-up, one IL128/OL128 semantic request,
five IL128/OL128 light-load requests at 0.5 request/s, and 3 s cooldown. Existing
Phase 3A Prometheus, real-SSE, request-ID, proxy, and token audits are retained.
Monitor readiness allows up to 120 s for cross-node shared-filesystem visibility;
this does not change the 200 ms fixed sampling period or any measurement window.

Monitor without changing the job:

```bash
squeue -u "$USER"
tail -f logs/disagg_bench_<jobid>.log
```

## Inspect

```bash
run_dir="$(find results/xpyd_energy -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
python3 -m json.tool "$run_dir/coverage_audit.json"
python3 -m json.tool "$run_dir/window_summaries.json"
python3 -m json.tool "$run_dir/aggregate.json"
sed -n '1,220p' "$run_dir/preflight_report.md"
wc -l "$run_dir"/P0/samples.jsonl "$run_dir"/D0/samples.jsonl
git status --short
```

The run passes only when real disaggregated SSE succeeds, exact logical IDs and
request/token deltas audit cleanly, both physical identities match, both raw
streams are non-empty, workload windows are valid, and the tiny smoke has no
measurement errors or missed slots. Explicit energy-counter non-support is not
a hardware failure if the labelled power-integral window passes its coverage
rules. The power source is recorded as the direct NVML getter or, when that
getter is explicitly unsupported, NVIDIA's read-only averaged-power field. Any
other counter error is not eligible for energy fallback.

Raw `results/xpyd_energy/**` and the new raw Phase 3A run remain ignored and
must not be staged. Do not use this preflight as a comparison and do not proceed
to a matrix or scheduler work from this runbook.
