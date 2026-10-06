#!/bin/bash
# Minerva full D1 submission: pin the interpreter instead of inheriting PATH.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_dir}"

export PYTHON_BIN="${PYTHON_BIN:-/data/users/chjing/miniforge3/envs/cuda-env/bin/python3}"
accepted_preflight="results/stage_d1/stage_d1_preflight_7158ff2/stage_d1_summary.json"
run_id="${1:-stage_d1_full_$(git rev-parse --short HEAD)}"
test ! -e "results/stage_d1/${run_id}"

"${PYTHON_BIN}" - "${accepted_preflight}" <<'PY'
import json
import sys
import vllm

print("D1 submission interpreter:", sys.executable)
print("D1 submission vLLM:", vllm.__version__)
if vllm.__version__ != "0.15.1":
    raise SystemExit("D1 requires vLLM 0.15.1; no job submitted")
with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
for gate in (
    "valid", "preflight_pass", "chronological_recovery_routing_audit",
    "routing_recovery_publication_audit", "overall_slo_contract_audit",
    "final_safety_state_consistency_audit", "final_high_restoration_audit",
):
    if summary.get(gate) is not True:
        raise SystemExit("Accepted D1 preflight gate missing: " + gate)
print("Accepted D1 preflight verified")
PY

mkdir -p logs
sbatch --job-name=xpyd_d1_full \
    --export="ALL,PYTHON_BIN=${PYTHON_BIN},EXP=E,XPYD_STAGED1_RUNTIME_CONFIG=paper/configs/xpyd_stageb_runtime_neptune_europa.json,XPYD_STAGED1_EXPERIMENT_CONFIG=paper/configs/xpyd_stage_d1_neptune_europa.json,XPYD_STAGED1_RUN_ID=${run_id},XPYD_STAGED1_STAGE=full,XPYD_STAGED1_ACCEPTED_PREFLIGHT=${accepted_preflight}" \
    run_disagg_benchmark.sh
