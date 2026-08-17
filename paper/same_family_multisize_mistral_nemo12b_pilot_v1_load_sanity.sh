#!/usr/bin/env bash
set -euo pipefail

MODEL='mistralai/Mistral-Nemo-Instruct-2407'
L40S_NODE='neptune'
L4_NODE='europa'
L40S_MEM_FREQ=9001
L4_MEM_FREQ=6251
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
CURL_BIN="${CURL_BIN:-/usr/bin/curl}"
SEQ_BIN="${SEQ_BIN:-/usr/bin/seq}"
SUDO_BIN="${SUDO_BIN:-/usr/bin/sudo}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-/usr/bin/nvidia-smi}"
RESULT_DIR='results/same_family_multisize_mistral_nemo12b_pilot_v1'
RESULTS_CSV='same_family_multisize_mistral_nemo12b_pilot_v1_load_sanity_results.csv'
L40S_TP_VALUES='1'
L4_TP_VALUES='2 4'

mkdir -p "${RESULT_DIR}" logs
echo 'gpu_type,tp,status,log_path' > "${RESULTS_CSV}"

set_gpu_freq() {
  local gpu_type=$1
  local node=$2
  local mem_freq=$3
  local tp=$4
  local cuda_devs
  if [ "$tp" -eq 2 ]; then cuda_devs='0,1'; else cuda_devs=$(${SEQ_BIN} -s, 0 $((tp - 1))); fi
  srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
    ${SUDO_BIN} ${NVIDIA_SMI_BIN} -pm 1 >/dev/null 2>&1 || true
  local sm_freq=1410
  if [ "${gpu_type}" = 'l40s' ]; then sm_freq=2520; fi
  srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
    ${SUDO_BIN} ${NVIDIA_SMI_BIN} -i ${cuda_devs} -lgc ${sm_freq},${sm_freq} >/dev/null 2>&1 || true
  srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
    ${SUDO_BIN} ${NVIDIA_SMI_BIN} -i ${cuda_devs} -lmc ${mem_freq},${mem_freq} >/dev/null 2>&1 || true
}

run_one() {
  local gpu_type=$1
  local node=$2
  local mem_freq=$3
  local tp=$4
  local port=$5
  local log_file="${RESULT_DIR}/vllm_load_sanity_${node}_tp${tp}.log"
  local health_host="${node}"
  local cuda_devs
  if [ "$tp" -eq 2 ]; then cuda_devs='0,1'; else cuda_devs=$(${SEQ_BIN} -s, 0 $((tp - 1))); fi
  set_gpu_freq "${gpu_type}" "${node}" "${mem_freq}" "$tp"
  srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" --gpus-per-node="${tp}" \
    --gpu-bind="map_gpu:${cuda_devs}" \
    ${PYTHON_BIN} -m vllm.entrypoints.openai.api_server \
      --model "${MODEL}" \
      --port "${port}" \
      --tensor-parallel-size "${tp}" \
      --max-model-len 4096 \
      --disable-log-requests > "${log_file}" 2>&1 &
  local server_pid=$!
  local status='FAIL'
  for _ in $(seq 1 120); do
    if ${CURL_BIN} -s --connect-timeout 3 "http://${health_host}:${port}/health" >/dev/null 2>&1; then
      status='PASS'
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      break
    fi
    sleep 5
  done
  echo "${gpu_type},${tp},${status},${log_file}" | tee -a "${RESULTS_CSV}"
  if kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" pkill -TERM -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
  sleep 10
  srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" pkill -KILL -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
}

port=8100
for tp in ${L40S_TP_VALUES}; do
  run_one 'l40s' "${L40S_NODE}" "${L40S_MEM_FREQ}" "${tp}" "${port}"
  port=$((port + 1))
done

port=8200
for tp in ${L4_TP_VALUES}; do
  run_one 'l4' "${L4_NODE}" "${L4_MEM_FREQ}" "${tp}" "${port}"
  port=$((port + 1))
done

cat "${RESULTS_CSV}"
if grep -q ',FAIL,' "${RESULTS_CSV}"; then
  echo 'Load sanity failed; refusing to continue.' >&2
  exit 1
fi
