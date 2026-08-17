#!/usr/bin/env bash
#SBATCH --job-name=sfms_load_sanity
#SBATCH --partition=long
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --nodelist=neptune,europa
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/same_family_multisize_load_sanity_%j.log

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"
bash same_family_multisize_mistral_nemo12b_pilot_v2_load_sanity.sh
