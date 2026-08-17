#!/usr/bin/env bash
set -euo pipefail

sbatch --wait --job-name=sfms_pilot_p1 --export=ALL paper/same_family_multisize_mistral_nemo12b_pilot_v1_collect.sh
