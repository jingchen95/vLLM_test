#!/bin/bash
# Grid'5000 (OAR) port of the Minerva 2P2D XPYD lifecycle.
#
# Not submitted directly: submit_g5k.sh prepends the #OAR directives and the
# XPYD_G5K_* parameters and submits the generated job file.
#
# Lifecycle (mirrors run_phase3c_substrate in run_disagg_benchmark.sh):
#   nodes from $OAR_NODE_FILE -> P node / D node by cluster
#   patch vLLM (P2P-NCCL tensor-key + recv timeout)       patch_vllm_p2p.sh
#   discover supported clocks -> LOW/MID/HIGH per pool    select_validation_frequencies
#   write runtime + experiment configs                    gen_g5k_configs.py
#   lock graphics+memory clocks at HIGH (sudo-g5k)
#   start P0/P1 on P node GPUs 0/1, D0/D1 on D node GPUs 0/1 (vLLM 0.15.1)
#   start the proxy on this node, run the stage module, clean up, reset clocks
#
# Remote execution uses oarsh (XPYD_NODE_EXEC=oarsh) and sudo-g5k (XPYD_SUDO).

set -uo pipefail

MODE="${XPYD_G5K_MODE:?e0 or e0c}"
P_CLUSTER="${XPYD_G5K_P_CLUSTER:-chicoree}"
D_CLUSTER="${XPYD_G5K_D_CLUSTER:-chuc}"
REPO="${XPYD_G5K_REPO:-${HOME}/vLLM_test}"
VENV="${XPYD_G5K_VENV:-${HOME}/g5k/xpyd_pd/venv}"
MODEL_DIR="${XPYD_G5K_MODEL_DIR:-${HOME}/g5k/models/Mistral-7B-v0.1}"
RUN_ID="${XPYD_G5K_RUN_ID:-${MODE}_g5k_${OAR_JOB_ID}}"
P_HIGH_OVERRIDE="${XPYD_G5K_P_HIGH_MHZ:-}"
D_HIGH_OVERRIDE="${XPYD_G5K_D_HIGH_MHZ:-}"
PROXY_PORT=8000
STARTUP_TIMEOUT_S="${XPYD_G5K_STARTUP_TIMEOUT_S:-600}"

PY="${VENV}/bin/python"
RESULT_ROOT="${REPO}/results/g5k/${RUN_ID}"
mkdir -p "${RESULT_ROOT}"
# Never call a bare `wait` below: bash >= 5.1 would also wait for this tee.
exec > >(tee -a "${RESULT_ROOT}/launcher.log") 2>&1
echo "== XPYD G5K launcher job=${OAR_JOB_ID} mode=${MODE} run_id=${RUN_ID} driver=$(hostname -f) start=$(date -u +%FT%TZ)"
cd "${REPO}" || exit 2
git rev-parse HEAD > "${RESULT_ROOT}/git_commit.txt" 2>/dev/null || true

# ------------------------------------------------------------------ nodes
sort -u "${OAR_NODE_FILE}" > "${RESULT_ROOT}/nodes.txt"
P_NODE=$(grep "${P_CLUSTER}" "${RESULT_ROOT}/nodes.txt" | head -1)
if [ "${P_CLUSTER}" = "${D_CLUSTER}" ]; then
  # Homogeneous pair: two distinct nodes of the same cluster.
  D_NODE=$(grep "${D_CLUSTER}" "${RESULT_ROOT}/nodes.txt" | grep -v "^${P_NODE}$" | head -1)
else
  D_NODE=$(grep "${D_CLUSTER}" "${RESULT_ROOT}/nodes.txt" | head -1)
fi
DRIVER=$(hostname -f)
if [ -z "${P_NODE}" ] || [ -z "${D_NODE}" ]; then
  echo "FATAL: need one ${P_CLUSTER} and one ${D_CLUSTER} node, got: $(cat "${RESULT_ROOT}/nodes.txt" | tr '\n' ' ')" >&2
  exit 2
fi
P_IP=$(getent ahostsv4 "${P_NODE}" | awk 'NR==1{print $1}')
D_IP=$(getent ahostsv4 "${D_NODE}" | awk 'NR==1{print $1}')
echo "P_NODE=${P_NODE} (${P_IP})  D_NODE=${D_NODE} (${D_IP})  DRIVER=${DRIVER}"

# rexec NODE CMD... : run on NODE (locally when NODE is this host)
rexec() {
  local node=$1; shift
  local cmd="$*"
  if [ "${node}" = "${DRIVER}" ]; then
    /bin/bash -lc "${cmd}"
  else
    # ssh/oarsh re-joins arguments with spaces and the remote shell re-parses
    # them; %q keeps the whole command (JSON quotes included) as one word.
    oarsh "${node}" /bin/bash -lc "$(printf '%q' "${cmd}")"
  fi
}
for N in "${P_NODE}" "${D_NODE}"; do
  rexec "${N}" "hostname && nvidia-smi -L && sudo-g5k id >/dev/null" > "${RESULT_ROOT}/gate_$(echo "${N}" | cut -d. -f1).txt" 2>&1 \
    || { echo "FATAL: gate failed on ${N}"; cat "${RESULT_ROOT}/gate_$(echo "${N}" | cut -d. -f1).txt"; exit 3; }
done

# ------------------------------------------------------------------ cleanup
SERVER_PIDS=()   # "node:pid"
PROXY_PID=""
cleanup() {
  echo "[trap] cleanup $(date -u +%FT%TZ)"
  [ -n "${PROXY_PID}" ] && kill "${PROXY_PID}" 2>/dev/null
  for item in "${SERVER_PIDS[@]:-}"; do
    [ -z "${item}" ] && continue
    rexec "${item%%:*}" "kill ${item##*:} 2>/dev/null; sleep 5; kill -9 ${item##*:} 2>/dev/null; true"
  done
  for N in "${P_NODE}" "${D_NODE}"; do
    rexec "${N}" "sudo-g5k nvidia-smi -rgc >/dev/null 2>&1; sudo-g5k nvidia-smi -rmc >/dev/null 2>&1; nvidia-smi --query-gpu=index,clocks.current.sm,clocks.current.memory --format=csv,noheader" \
      >> "${RESULT_ROOT}/clock_reset_evidence.txt" 2>&1 || true
  done
  echo "[trap] done"
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------ vLLM patches (idempotent)
"${PY}" -c "import vllm, sys; print('vllm', vllm.__version__, sys.executable)" || exit 4
bash "${REPO}/paper/scripts/g5k/patch_vllm_p2p.sh" "${PY}" || exit 4

# ------------------------------------------------------------------ clock discovery
discover() {   # NODE -> "gpu_name|supported_graphics_csv|max_mem_mhz"
  rexec "$1" "nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1; \
    nvidia-smi -q -d SUPPORTED_CLOCKS -i 0 | grep -oE 'Graphics[[:space:]]*:[[:space:]]*[0-9]+' | grep -oE '[0-9]+\$' | sort -nu | paste -sd,; \
    nvidia-smi -q -d SUPPORTED_CLOCKS -i 0 | grep -oE 'Memory[[:space:]]*:[[:space:]]*[0-9]+' | grep -oE '[0-9]+\$' | sort -nu | tail -1"
}
mapfile -t P_INFO < <(discover "${P_NODE}")
mapfile -t D_INFO < <(discover "${D_NODE}")
P_GPU="${P_INFO[0]}"; P_SUP="${P_INFO[1]}"; P_MEM="${P_INFO[2]}"
D_GPU="${D_INFO[0]}"; D_SUP="${D_INFO[1]}"; D_MEM="${D_INFO[2]}"
echo "P: ${P_GPU} mem=${P_MEM} supported=${P_SUP}" ; echo "D: ${D_GPU} mem=${D_MEM} supported=${D_SUP}"
levels() {  # SUPPORTED_CSV HIGH_OVERRIDE -> "LOW,MID,HIGH" using the Minerva selection rule
  PYTHONPATH="${REPO}/paper/scripts" "${PY}" - "$1" "$2" <<'PYEOF'
import sys
from xpyd.phase3d_control import select_validation_frequencies
sup = [int(x) for x in sys.argv[1].split(",") if x]
high = int(sys.argv[2]) if sys.argv[2] else max(sup)
low, mid, high = select_validation_frequencies(sup, high)
print("%d,%d,%d" % (low, mid, high))
PYEOF
}
# Sustainable HIGH: lock the nominal maximum and read back the clock the GPU
# actually runs at. H200 NVL locked at 1785 MHz runs at 1635 MHz (job
# 2234866), so the nominal maximum cannot be validated by readback. Use the
# observed clock when it is lower and hardware-supported (cf. Minerva's L4
# 1500 MHz sustainable-frequency probe). An explicit override wins.
sustainable_high() {  # NODE SUPPORTED_CSV -> MHz
  local node=$1 sup=$2 max observed
  max=$(echo "${sup}" | tr ',' '\n' | sort -n | tail -1)
  observed=$(rexec "${node}" "sudo-g5k nvidia-smi -i 0 -lgc ${max},${max} >/dev/null 2>&1; sleep 3; \
    nvidia-smi -i 0 --query-gpu=clocks.current.sm --format=csv,noheader,nounits | tr -d ' '; \
    sudo-g5k nvidia-smi -i 0 -rgc >/dev/null 2>&1" | tail -1)
  if [ -n "${observed}" ] && [ "${observed}" -lt "${max}" ] && echo ",${sup}," | grep -q ",${observed},"; then
    echo "${observed}"
  else
    echo "${max}"
  fi
}
[ -n "${P_HIGH_OVERRIDE}" ] || P_HIGH_OVERRIDE=$(sustainable_high "${P_NODE}" "${P_SUP}")
[ -n "${D_HIGH_OVERRIDE}" ] || D_HIGH_OVERRIDE=$(sustainable_high "${D_NODE}" "${D_SUP}")
echo "sustainable HIGH: P=${P_HIGH_OVERRIDE} MHz (nominal max $(echo "${P_SUP}" | tr ',' '\n' | sort -n | tail -1))  D=${D_HIGH_OVERRIDE} MHz (nominal max $(echo "${D_SUP}" | tr ',' '\n' | sort -n | tail -1))"
P_LEVELS=$(levels "${P_SUP}" "${P_HIGH_OVERRIDE}") || exit 5
D_LEVELS=$(levels "${D_SUP}" "${D_HIGH_OVERRIDE}") || exit 5
echo "P levels (LOW,MID,HIGH)=${P_LEVELS}  D levels=${D_LEVELS}"

# ------------------------------------------------------------------ configs
mapfile -t CFG < <("${PY}" "${REPO}/paper/scripts/g5k/gen_g5k_configs.py" --mode "${MODE}" \
  --out-dir "${RESULT_ROOT}/configs" --repo "${REPO}" \
  --p-node "${P_NODE}" --d-node "${D_NODE}" --p-ip "${P_IP}" --d-ip "${D_IP}" --proxy-host "${DRIVER}" --proxy-port "${PROXY_PORT}" \
  --p-gpu-name "${P_GPU}" --d-gpu-name "${D_GPU}" --p-levels "${P_LEVELS}" --d-levels "${D_LEVELS}" \
  --p-mem-mhz "${P_MEM}" --d-mem-mhz "${D_MEM}" --model-dir "${MODEL_DIR}" --result-root "${RESULT_ROOT}")
[ "${#CFG[@]}" -eq 2 ] || { echo "FATAL: config generation failed"; exit 5; }
RUNTIME_CFG="${CFG[0]}"; EXPERIMENT_CFG="${CFG[1]}"
echo "runtime=${RUNTIME_CFG} experiment=${EXPERIMENT_CFG}"

# ------------------------------------------------------------------ lock clocks at HIGH
P_HIGH="${P_LEVELS##*,}"; D_HIGH="${D_LEVELS##*,}"
lock_clocks() {  # NODE GRAPHICS MEM (memory clock is fixed on HBM GPUs; not locked)
  rexec "$1" "for i in 0 1; do sudo-g5k nvidia-smi -i \$i -lgc $2,$2 || exit 1; done; sleep 1; \
    nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,clocks.current.sm,clocks.current.memory --format=csv,noheader -i 0,1"
}
{
  echo "== lock $(date -u +%FT%TZ)"
  lock_clocks "${P_NODE}" "${P_HIGH}" "${P_MEM}"
  lock_clocks "${D_NODE}" "${D_HIGH}" "${D_MEM}"
} | tee -a "${RESULT_ROOT}/fixed_clock_evidence.log" || exit 6

# ------------------------------------------------------------------ vLLM servers
KV_P0='{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_port":14579,"kv_connector_extra_config":{"send_type":"PUT"}}'
KV_P1='{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_port":14580,"kv_connector_extra_config":{"send_type":"PUT"}}'
KV_D0='{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_port":14579}'
KV_D1='{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_port":14580}'
COMMON_ARGS="--host 0.0.0.0 --tensor-parallel-size 1 --max-model-len 4096 --no-enable-prefix-caching --no-enable-chunked-prefill --served-model-name mistralai/Mistral-7B-v0.1"
NCCL_ENV="NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=^lo NCCL_DEBUG=INFO"

start_server() {  # NODE ENDPOINT GPU PORT KV_JSON EXTRA
  local node=$1 ep=$2 gpu=$3 port=$4 kv=$5 extra=$6 log="${RESULT_ROOT}/vllm_${2}.log"
  # Background only the server: `cd X && cmd &` would background the whole
  # AND-list, whose subshell keeps the $(...) pipe open forever (job 2230985
  # hung 3.5 h here). stdin from /dev/null so oarsh/ssh can return.
  local cmd="cd ${REPO}; CUDA_VISIBLE_DEVICES=${gpu} ${NCCL_ENV} nohup ${PY} -m vllm.entrypoints.openai.api_server --model ${MODEL_DIR} --port ${port} ${COMMON_ARGS} --kv-transfer-config '${kv}' ${extra} > ${log} 2>&1 < /dev/null & echo \$!"
  local pid
  pid=$(rexec "${node}" "${cmd}" | tail -1)
  SERVER_PIDS+=("${node}:${pid}")
  echo "[${ep}] started on ${node} gpu=${gpu} port=${port} pid=${pid}"
}
wait_health() {  # NODE_IP PORT EP
  local elapsed=0
  while ! curl -s --connect-timeout 5 "http://$1:$2/health" >/dev/null 2>&1; do
    sleep 10; elapsed=$((elapsed + 10))
    if [ "${elapsed}" -ge "${STARTUP_TIMEOUT_S}" ]; then echo "FATAL: $3 not healthy after ${elapsed}s"; tail -30 "${RESULT_ROOT}/vllm_$3.log"; return 1; fi
    [ $((elapsed % 60)) -eq 0 ] && echo "  [$3] ... ${elapsed}s"
  done
  echo "[$3] healthy after ${elapsed}s"
}
start_server "${P_NODE}" P0 0 8100 "${KV_P0}" "--disable-log-requests"
start_server "${P_NODE}" P1 1 8101 "${KV_P1}" "--disable-log-requests"
start_server "${D_NODE}" D0 0 8200 "${KV_D0}" "--gpu-memory-utilization 0.82"
start_server "${D_NODE}" D1 1 8201 "${KV_D1}" "--gpu-memory-utilization 0.82"
wait_health "${P_IP}" 8100 P0 || exit 7
wait_health "${P_IP}" 8101 P1 || exit 7
wait_health "${D_IP}" 8200 D0 || exit 7
wait_health "${D_IP}" 8201 D1 || exit 7

# ------------------------------------------------------------------ proxy + harness environment
export PYTHONPATH="${REPO}/paper/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export XPYD_NODE_EXEC=oarsh XPYD_SUDO=sudo-g5k XPYD_NVIDIA_SMI=nvidia-smi XPYD_MEMORY_CLOCK_CONTROL=0
export XPYD_PROXY_URI="http://127.0.0.1:${PROXY_PORT}"
export XPYD_PROXY_DIAGNOSTICS_LOG="${RESULT_ROOT}/proxy_diagnostics.jsonl"
export XPYD_ROUTING_POLICY=round_robin XPYD_ROUTING_CONTROL_FILE=""
"${PY}" -m xpyd.disagg_proxy --multi-endpoint-config "${RUNTIME_CFG}" --proxy-port "${PROXY_PORT}" \
  --diagnostics-log "${XPYD_PROXY_DIAGNOSTICS_LOG}" > "${RESULT_ROOT}/proxy.log" 2>&1 &
PROXY_PID=$!
elapsed=0
while ! curl -s --connect-timeout 2 "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1; do
  sleep 2; elapsed=$((elapsed + 2)); [ "${elapsed}" -ge 60 ] && { echo "FATAL: proxy not ready"; tail -20 "${RESULT_ROOT}/proxy.log"; exit 8; }
done
echo "proxy ready pid=${PROXY_PID}"

# ------------------------------------------------------------------ stage module
case "${MODE}" in
  e0)  args=(-m xpyd.stage_e0_regime   --runtime-config "${RUNTIME_CFG}" --experiment-config "${EXPERIMENT_CFG}" --run-id "${RUN_ID}") ;;
  e0c) args=(-m xpyd.stage_e0c_capacity --runtime-config "${RUNTIME_CFG}" --workload-config  "${EXPERIMENT_CFG}" --run-id "${RUN_ID}") ;;
  *)   echo "FATAL: unknown mode ${MODE}"; exit 2 ;;
esac
echo "== running ${args[*]} $(date -u +%FT%TZ)"
status=0
"${PY}" "${args[@]}" || status=$?
echo "== stage exit status=${status} $(date -u +%FT%TZ)"
echo "${status}" > "${RESULT_ROOT}/stage_status.txt"
exit "${status}"
