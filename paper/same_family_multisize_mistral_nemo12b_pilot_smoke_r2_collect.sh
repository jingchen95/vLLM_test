#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR='results/same_family_multisize_mistral_nemo12b_pilot_smoke_r2'
if [ -d "${RESULT_DIR}/CD_prefill_decode_only" ] && [ -n "$(ls -A "${RESULT_DIR}/CD_prefill_decode_only" 2>/dev/null)" ]; then
  echo "Refusing to reuse non-empty result directory: ${RESULT_DIR}/CD_prefill_decode_only" >&2
  exit 1
fi
mkdir -p "${RESULT_DIR}"

# Pass 1: prefill on L40S, decode on L4
MODEL='mistralai/Mistral-Nemo-Instruct-2407' RESULT_DIR="${RESULT_DIR}" EXP=CD CD_EXTENDED_NAMES=1 NUM_PROMPTS=200 CD_PREFILL_LABEL='l40s_mistral_nemo_12b_pilot' CD_DECODE_LABEL='l4_mistral_nemo_12b_pilot' CD_PREFILL_FREQS='2520' CD_DECODE_FREQS='1410 2040' CD_PREFILL_TPS='1' CD_DECODE_TPS='2 4' CD_PREFILL_ILS='1024' CD_DECODE_ILS='1024' CD_DECODE_OLS='256 1024' CD_PREFILL_RATES='1' CD_DECODE_RATES='1' bash run_disagg_benchmark.sh

# Pass 2: prefill on L4, decode on L40S
MODEL='mistralai/Mistral-Nemo-Instruct-2407' RESULT_DIR="${RESULT_DIR}" EXP=CD CD_EXTENDED_NAMES=1 NUM_PROMPTS=200 CD_PREFILL_NODE='europa' CD_DECODE_NODE='neptune' CD_PREFILL_LABEL='l4_mistral_nemo_12b_pilot' CD_DECODE_LABEL='l40s_mistral_nemo_12b_pilot' CD_PREFILL_MEM_FREQ=6251 CD_DECODE_MEM_FREQ=9001 CD_PREFILL_FREQS='2040' CD_DECODE_FREQS='1500 2520' CD_PREFILL_TPS='2 4' CD_DECODE_TPS='1' CD_PREFILL_ILS='1024' CD_DECODE_ILS='1024' CD_DECODE_OLS='256 1024' CD_PREFILL_RATES='1' CD_DECODE_RATES='1' bash run_disagg_benchmark.sh

# Summarize phase-only pilot results into a model-ready CSV
python3 summarize_same_family_multisize_pilot.py --results-dir 'results/same_family_multisize_mistral_nemo12b_pilot_smoke_r2/CD_prefill_decode_only' --model-id 'mistral_nemo_12b_pilot' --hf-name 'mistralai/Mistral-Nemo-Instruct-2407' --family 'mistral' --param-count-b 12.0 --output-prefix 'same_family_multisize_mistral_nemo12b_pilot_smoke_r2'
