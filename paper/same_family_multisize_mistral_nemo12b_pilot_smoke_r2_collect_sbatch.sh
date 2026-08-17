#!/usr/bin/env bash
#SBATCH --job-name=sfms_pilot
#SBATCH --partition=long
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --nodelist=neptune,europa
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=23:59:59
#SBATCH --output=logs/same_family_multisize_pilot_%j.log

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"
bash same_family_multisize_mistral_nemo12b_pilot_smoke_r2_collect.sh
