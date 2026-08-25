#!/bin/bash
# =============================================================================
# B5: End-to-End Disaggregated Serving Benchmark
#
# Compares four serving configurations on identical request traces:
#   A) Monolithic L40S   — all requests on uranus/neptune (4×L40S)
#   B) Monolithic L4     — all requests on callisto/... (8×L4)
#   C) Prefill-only L40S — input_len=X, output_len=1 (TTFT + prefill energy)
#   D) Decode-only L4    — input_len=1, output_len=X (TPOT + decode energy)
#   E) Disaggregated     — vLLM P/D disagg: L40S prefill → L4 decode
#
# Hardware:
#   L40S nodes: neptune, uranus  (4 GPUs each)
#   L4 nodes:   callisto, europa, ganymede, io  (8 GPUs each)
#
# Power is collected via NVML polling (collect_power.py) on both nodes
# simultaneously, giving per-node energy breakdown.
#
# Usage:
#   sbatch run_disagg_benchmark.sh
#   sbatch run_disagg_benchmark.sh --export=EXP=A      # only experiment A
#   sbatch run_disagg_benchmark.sh --export=EXP=CD     # only experiments C and D
#   sbatch run_disagg_benchmark.sh --export=EXP=A,E    # run A then E
#   sbatch run_disagg_benchmark.sh --export=GPU_FREQ=2010,TP=1
# =============================================================================
#SBATCH --job-name=jsep_disagg_b5
#SBATCH --partition=long
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --nodelist=neptune,europa
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=23:59:59
#SBATCH --output=logs/disagg_bench_%j.log

# Sudoers requirement (run once on both neptune and europa before submitting):
#   sudo visudo  →  add line:
#   your_username ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p logs results
SUDO_BIN="${SUDO_BIN:-/usr/bin/sudo}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-/usr/bin/nvidia-smi}"
HOSTNAME_BIN="${HOSTNAME_BIN:-/bin/hostname}"
BASH_BIN="${BASH_BIN:-/bin/bash}"
REALPATH_BIN="${REALPATH_BIN:-/bin/realpath}"
CURL_BIN="${CURL_BIN:-/usr/bin/curl}"
PKILL_BIN="${PKILL_BIN:-/usr/bin/pkill}"
TAIL_BIN="${TAIL_BIN:-/usr/bin/tail}"
AWK_BIN="${AWK_BIN:-/usr/bin/awk}"
WC_BIN="${WC_BIN:-/usr/bin/wc}"
SEQ_BIN="${SEQ_BIN:-/usr/bin/seq}"
SYNC_BIN="${SYNC_BIN:-/bin/sync}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
SCONTROL_BIN="${SCONTROL_BIN:-/usr/bin/scontrol}"
# When set, Experiment E starts the existing P/D/proxy lifecycle and hands
# control to a read-only Phase 3A/3B harness. Those paths perform no GPU clock,
# persistence, or reset operations. Phase 3B starts only isolated query-only
# NVML monitor processes.
XPYD_PHASE3A_CONFIG="${XPYD_PHASE3A_CONFIG:-}"
XPYD_PHASE3A_MODE="${XPYD_PHASE3A_MODE:-all}"
XPYD_PHASE3A_SEMANTIC_PROBE_ID="${XPYD_PHASE3A_SEMANTIC_PROBE_ID:-}"
XPYD_PHASE3B_CONFIG="${XPYD_PHASE3B_CONFIG:-}"
XPYD_PHASE3B_RUN_ID="${XPYD_PHASE3B_RUN_ID:-}"
XPYD_PHASE3B_CHARACTERIZATION_CONFIG="${XPYD_PHASE3B_CHARACTERIZATION_CONFIG:-}"
XPYD_PHASE3B_CHARACTERIZATION_RUN_ID="${XPYD_PHASE3B_CHARACTERIZATION_RUN_ID:-}"
XPYD_PHASE3B_CHARACTERIZATION_REPEATS="${XPYD_PHASE3B_CHARACTERIZATION_REPEATS:-}"
XPYD_PHASE3B_CHARACTERIZATION_REQUESTS="${XPYD_PHASE3B_CHARACTERIZATION_REQUESTS:-}"
XPYD_PHASE3C_CONFIG="${XPYD_PHASE3C_CONFIG:-}"
XPYD_PHASE3C_RUN_ID="${XPYD_PHASE3C_RUN_ID:-}"
XPYD_PHASE3D_CONFIG="${XPYD_PHASE3D_CONFIG:-}"
XPYD_PHASE3D_RUN_ID="${XPYD_PHASE3D_RUN_ID:-}"
XPYD_PHASE3D_STAGE="${XPYD_PHASE3D_STAGE:-}"
XPYD_PHASE3D_ACCEPTED_ACTUATOR_AUDIT="${XPYD_PHASE3D_ACCEPTED_ACTUATOR_AUDIT:-}"
XPYD_PHASE4A_CONFIG="${XPYD_PHASE4A_CONFIG:-}"
XPYD_PHASE4A_RUN_ID="${XPYD_PHASE4A_RUN_ID:-}"
XPYD_PHASE4A_ACCEPTED_ACTUATOR_AUDIT="${XPYD_PHASE4A_ACCEPTED_ACTUATOR_AUDIT:-}"
XPYD_PHASE4A_ACCEPTED_CLOSED_LOOP_AUDIT="${XPYD_PHASE4A_ACCEPTED_CLOSED_LOOP_AUDIT:-}"
XPYD_PHASE4B_CONFIG="${XPYD_PHASE4B_CONFIG:-}"
XPYD_PHASE4B_RUN_ID="${XPYD_PHASE4B_RUN_ID:-}"
XPYD_PHASE4B_ORACLE_SUMMARY="${XPYD_PHASE4B_ORACLE_SUMMARY:-}"
XPYD_PHASE4B_SMOKE="${XPYD_PHASE4B_SMOKE:-0}"
XPYD_PHASE4B_STAGE="${XPYD_PHASE4B_STAGE:-stationary}"
XPYD_PHASE4B_ACCEPTED_STATIONARY_AUDIT="${XPYD_PHASE4B_ACCEPTED_STATIONARY_AUDIT:-}"
XPYD_PHASE4B_ACCEPTED_ACTIVE_SMOKE_AUDIT="${XPYD_PHASE4B_ACCEPTED_ACTIVE_SMOKE_AUDIT:-}"
XPYD_PHASE4B1_CONFIG="${XPYD_PHASE4B1_CONFIG:-}"
XPYD_PHASE4B1_RUN_ID="${XPYD_PHASE4B1_RUN_ID:-}"
XPYD_ROUTING_CONTROL_FILE="${XPYD_ROUTING_CONTROL_FILE:-}"
XPYD_NO_GPU_MUTATION="${XPYD_NO_GPU_MUTATION:-0}"
_xpyd_mode_count=0
[ -n "${XPYD_PHASE3A_CONFIG}" ] && _xpyd_mode_count=$((_xpyd_mode_count + 1))
[ -n "${XPYD_PHASE3B_CONFIG}" ] && _xpyd_mode_count=$((_xpyd_mode_count + 1))
[ -n "${XPYD_PHASE3B_CHARACTERIZATION_CONFIG}" ] && _xpyd_mode_count=$((_xpyd_mode_count + 1))
[ -n "${XPYD_PHASE3C_CONFIG}" ] && _xpyd_mode_count=$((_xpyd_mode_count + 1))
[ -n "${XPYD_PHASE3D_CONFIG}" ] && _xpyd_mode_count=$((_xpyd_mode_count + 1))
[ -n "${XPYD_PHASE4A_CONFIG}" ] && _xpyd_mode_count=$((_xpyd_mode_count + 1))
[ -n "${XPYD_PHASE4B_CONFIG}" ] && _xpyd_mode_count=$((_xpyd_mode_count + 1))
[ -n "${XPYD_PHASE4B1_CONFIG}" ] && _xpyd_mode_count=$((_xpyd_mode_count + 1))
if [ "${_xpyd_mode_count}" -gt 1 ]; then
    echo "ERROR: choose exactly one XpYd Phase 3A/3B/3C/3D/4A/4B/4B.1 config" >&2
    exit 2
fi
[ -n "${XPYD_PHASE3A_CONFIG}${XPYD_PHASE3B_CONFIG}${XPYD_PHASE3B_CHARACTERIZATION_CONFIG}${XPYD_PHASE3C_CONFIG}${XPYD_PHASE3D_CONFIG}${XPYD_PHASE4A_CONFIG}${XPYD_PHASE4B_CONFIG}${XPYD_PHASE4B1_CONFIG}" ] && XPYD_NO_GPU_MUTATION=1
XPYD_CHARACTERIZATION_CLOCKS_REQUESTED=0
[ -n "${XPYD_PHASE3B_CHARACTERIZATION_CONFIG}${XPYD_PHASE3C_CONFIG}${XPYD_PHASE3D_CONFIG}${XPYD_PHASE4A_CONFIG}${XPYD_PHASE4B_CONFIG}${XPYD_PHASE4B1_CONFIG}" ] && XPYD_CHARACTERIZATION_CLOCKS_REQUESTED=1
if [ -n "${XPYD_PHASE3D_CONFIG}" ] && [ "${XPYD_PHASE3D_STAGE}" != "A" ] && [ "${XPYD_PHASE3D_STAGE}" != "B" ]; then
    echo "ERROR: XPYD_PHASE3D_STAGE must be A or B" >&2
    exit 2
fi
XPYD_FIXED_CLOCK_EVIDENCE="${XPYD_FIXED_CLOCK_EVIDENCE:-}"
export XPYD_FIXED_CLOCK_EVIDENCE

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# The auditable proxy is checked in so its streaming behavior can be tested.
# It keeps P non-streaming/max_tokens=1 and forwards real D SSE bytes. Launch
# it as a package module so xpyd/types.py cannot shadow the stdlib types module.
# ---------------------------------------------------------------------------
DISAGG_PROXY_MODULE="xpyd.disagg_proxy"

# ---------------------------------------------------------------------------
# Helper: privileged GPU reset / reset-app-clocks on a node.
# These operations require the same sudoers exception as set_gpu_freq().
# ---------------------------------------------------------------------------
_reset_gpu_state() {
    local node="$1"
    if [ "${XPYD_NO_GPU_MUTATION}" -eq 1 ]; then
        echo "  XpYd observation path: skipping GPU reset/app-clock mutation on ${node}"
        return 0
    fi
    local _reset_cmd="set -e; ${SUDO_BIN} ${NVIDIA_SMI_BIN} --gpu-reset 2>/dev/null || ${SUDO_BIN} ${NVIDIA_SMI_BIN} -rac 2>/dev/null"
    if [ "${node}" = "${LOCAL_NODE:-}" ]; then
        "${BASH_BIN}" -c "${_reset_cmd}" || true
    else
        srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
            "${BASH_BIN}" -c "${_reset_cmd}" || true
    fi
}

# ---------------------------------------------------------------------------
# Trap: run cleanup on any exit (normal, error, or signal).
# Ensures vLLM servers and monitors are always torn down, even when set -e
# causes an early exit mid-experiment.  Without this, a failed health-check
# or bench command leaves the local vLLM process alive with a live NCCL
# context — which is exactly what wedges GPUs at 100% for the next config.
# ---------------------------------------------------------------------------
_trap_cleanup() {
    echo "[trap] Cleaning up servers and monitors..."
    _bg_kill TERM "${MONITOR_PIDS[@]}" 2>/dev/null || true
    _bg_kill TERM "${PHASE3C_SERVER_PIDS[@]}" 2>/dev/null || true
    sleep 1
    _bg_kill KILL "${MONITOR_PIDS[@]}" 2>/dev/null || true
    _bg_kill KILL "${PHASE3C_SERVER_PIDS[@]}" 2>/dev/null || true
    [ -n "${L40S_SERVER_PID:-}" ] && kill -TERM "${L40S_SERVER_PID}" 2>/dev/null || true
    [ -n "${L4_SERVER_PID:-}"   ] && kill -TERM "${L4_SERVER_PID}"   2>/dev/null || true
    sleep 10
    [ -n "${L40S_SERVER_PID:-}" ] && kill -KILL "${L40S_SERVER_PID}" 2>/dev/null || true
    [ -n "${L4_SERVER_PID:-}"   ] && kill -KILL "${L4_SERVER_PID}"   2>/dev/null || true
    "${PKILL_BIN}" -KILL -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
    "${PKILL_BIN}" -KILL -f 'python.*vllm' 2>/dev/null || true
    if [ "${XPYD_CHARACTERIZATION_CLOCKS_REQUESTED:-0}" -eq 1 ]; then
        [ -n "${L40S_NODE:-}" ] && reset_characterization_clocks "${L40S_NODE}" P0 0 || true
        [ -n "${L4_NODE:-}" ] && reset_characterization_clocks "${L4_NODE}" D0 0 || true
        if [ -n "${XPYD_PHASE3C_CONFIG:-}${XPYD_PHASE3D_CONFIG:-}${XPYD_PHASE4A_CONFIG:-}${XPYD_PHASE4B_CONFIG:-}${XPYD_PHASE4B1_CONFIG:-}" ]; then
            reset_characterization_clocks "${L40S_NODE}" P1 1 || true
            reset_characterization_clocks "${L4_NODE}" D1 1 || true
        fi
    fi
    local _node
    for _node in "${ACTIVE_NODES[@]:-}"; do
        [ -n "${_node}" ] || continue
        _reset_gpu_state "${_node}"
    done
}
trap _trap_cleanup EXIT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
WORKLOAD_MODE_RAW="${WORKLOAD_MODE:-random}"
WORKLOAD_MODE="$(printf '%s' "${WORKLOAD_MODE_RAW}" | tr '[:upper:]' '[:lower:]')"
TRACE_CSV="${TRACE_CSV:-}"
TRACE_TOKENIZER_MODEL="${TRACE_TOKENIZER_MODEL:-${MODEL}}"
TRACE_REQUEST_TIMEOUT_S="${TRACE_REQUEST_TIMEOUT_S:-900}"
VLLM_STARTUP_TIMEOUT_S="${VLLM_STARTUP_TIMEOUT_S:-1200}"
L40S_NODE="${L40S_NODE:-neptune}"
L4_NODE="${L4_NODE:-europa}"
L40S_GPU_IDS="${L40S_GPU_IDS:-}"
L4_GPU_IDS="${L4_GPU_IDS:-}"
CD_PREFILL_NODE="${CD_PREFILL_NODE:-${L40S_NODE}}"
CD_DECODE_NODE="${CD_DECODE_NODE:-${L4_NODE}}"
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8000
KV_PORT=14579        # NCCL port for KV cache transfer
# NUM_PROMPTS=1000   # v1: 1000 prompts → avg 10min/bench, 60min worst. Too slow.
NUM_PROMPTS="${NUM_PROMPTS:-200}"      # v2: 200 prompts. Comparable to DynamoLLM (30s×5 repeats = 60-1500 reqs).
NUM_WARMUPS="${NUM_WARMUPS:-1}"        # 1 warmup sufficient for steady-state metrics
# ---------------------------------------------------------------------------
# v2 dense calibration: fresh results directory.
# Old 1000-prompt results preserved in results/disagg_20260312_211051_1000prompts_backup
# ---------------------------------------------------------------------------
RESULT_DIR="${RESULT_DIR:-results/disagg_20260321_v2}"
mkdir -p "${RESULT_DIR}"
if [ "${XPYD_CHARACTERIZATION_CLOCKS_REQUESTED}" -eq 1 ] && [ -z "${XPYD_FIXED_CLOCK_EVIDENCE}" ]; then
    XPYD_FIXED_CLOCK_EVIDENCE="${RESULT_DIR}/fixed_clock_evidence.log"
    export XPYD_FIXED_CLOCK_EVIDENCE
fi
if [ "${WORKLOAD_MODE}" != "random" ] && [ "${WORKLOAD_MODE}" != "trace" ]; then
    echo "ERROR: WORKLOAD_MODE must be 'random' or 'trace' (got '${WORKLOAD_MODE_RAW}')."
    exit 1
fi
TRACE_REPLAY_SCRIPT="${TRACE_REPLAY_SCRIPT:-}"
if [ -z "${TRACE_REPLAY_SCRIPT}" ]; then
    if [ -f "${SCRIPT_DIR}/paper/scripts/replay_synthetic_trace.py" ]; then
        TRACE_REPLAY_SCRIPT="${SCRIPT_DIR}/paper/scripts/replay_synthetic_trace.py"
    elif [ -f "${SCRIPT_DIR}/replay_synthetic_trace.py" ]; then
        TRACE_REPLAY_SCRIPT="${SCRIPT_DIR}/replay_synthetic_trace.py"
    fi
fi
if [ "${WORKLOAD_MODE}" = "trace" ] && [ ! -f "${TRACE_CSV}" ]; then
    echo "ERROR: WORKLOAD_MODE=trace requires TRACE_CSV to point to an existing CSV."
    exit 1
fi
if [ "${WORKLOAD_MODE}" = "trace" ] && [ ! -f "${TRACE_REPLAY_SCRIPT}" ]; then
    echo "ERROR: WORKLOAD_MODE=trace requires replay_synthetic_trace.py."
    echo "Tried: ${TRACE_REPLAY_SCRIPT:-<unset>}"
    echo "Set TRACE_REPLAY_SCRIPT explicitly or place the script under:"
    echo "  ${SCRIPT_DIR}/paper/scripts/replay_synthetic_trace.py"
    echo "or"
    echo "  ${SCRIPT_DIR}/replay_synthetic_trace.py"
    exit 1
fi
EXP_RAW="${EXP:-ALL}"
EXP="$(printf '%s' "${EXP_RAW}" | tr '[:lower:]' '[:upper:]')"
EXP="${EXP// /}"
RUN_A=0
RUN_B=0
RUN_CD=0
RUN_E=0
if [ "${EXP}" = "ALL" ]; then
    RUN_A=1
    RUN_B=1
    RUN_CD=1
else
    IFS=',' read -r -a EXP_PARTS <<< "${EXP}"
    local_exp_seen=0
    for _exp_part in "${EXP_PARTS[@]}"; do
        [ -n "${_exp_part}" ] || continue
        local_exp_seen=1
        case "${_exp_part}" in
            A) RUN_A=1 ;;
            B) RUN_B=1 ;;
            C|D|CD) RUN_CD=1 ;;
            E) RUN_E=1 ;;
            *)
                echo "ERROR: unsupported EXP token '${_exp_part}' in EXP='${EXP_RAW}'."
                echo "Use comma-separated tokens from: A, B, CD, E, or ALL."
                exit 1
                ;;
        esac
    done
    if [ "${local_exp_seen}" -eq 0 ]; then
        echo "ERROR: EXP='${EXP_RAW}' did not contain any experiment tokens."
        exit 1
    fi
fi

NEED_L40S=0
NEED_L4=0
if [ "${RUN_A}" -eq 1 ] || [ "${RUN_CD}" -eq 1 ] || [ "${RUN_E}" -eq 1 ]; then
    NEED_L40S=1
fi
if [ "${RUN_B}" -eq 1 ] || [ "${RUN_CD}" -eq 1 ] || [ "${RUN_E}" -eq 1 ]; then
    NEED_L4=1
fi

ACTIVE_NODES=()
[ "${NEED_L40S}" -eq 1 ] && ACTIVE_NODES+=("${L40S_NODE}")
[ "${NEED_L4}" -eq 1 ] && ACTIVE_NODES+=("${L4_NODE}")
if [ "${RUN_CD}" -eq 1 ]; then
    case " ${ACTIVE_NODES[*]} " in
        *" ${CD_PREFILL_NODE} "*) ;;
        *) ACTIVE_NODES+=("${CD_PREFILL_NODE}") ;;
    esac
    case " ${ACTIVE_NODES[*]} " in
        *" ${CD_DECODE_NODE} "*) ;;
        *) ACTIVE_NODES+=("${CD_DECODE_NODE}") ;;
    esac
fi

# DVFS configurations to sweep
# Format: "gpu_freq_mhz:tp_degree"
#
# Dense calibration grid (v2) — matches DynamoLLM methodology:
#   6 frequencies per GPU type × 3 TPs × expanded workloads+rates
#   Existing results are auto-skipped (safe resume via [ -f "${out}" ] check).
#
# L40S: 6 freqs × TP 1,2,4  (18 configs)
# Original 3:   1500, 2010, 2520
# New:          1245, 1755, 2265
L40S_CONFIGS="${L40S_CONFIGS:-2520:1 2265:1 2010:1 1755:1 1500:1 1245:1 2520:2 2010:2 1500:2 2520:4 2010:4 1500:4}"

# L4: 6 freqs × TP 1,2,4  (18 configs)
# Original 4:   1200, 1410, 1620, 2040
# New:          990, 1830
L4_CONFIGS="${L4_CONFIGS:-2040:1 1830:1 1620:1 1410:1 1200:1 990:1 2040:2 1410:2 990:2 2040:4 1410:4 990:4}"

# Request rate and workload matrix (v2 — denser rate coverage)
#   format: "input_len output_len request_rate max_concurrency"
#   Rates: {5, 10, 20, 30, 50} for short/medium; {5, 10, 20} for long
#   rate=2 kept only for balanced baseline; rate=3 dropped (rate=5 sufficient)
#   Comparable to DynamoLLM: 3 load levels (low/med/high), 30s per config
if [ -n "${WORKLOADS_OVERRIDE:-}" ]; then
    IFS=';' read -r -a WORKLOADS <<< "${WORKLOADS_OVERRIDE}"
else
    WORKLOADS=(
        # balanced — full rate sweep including low-load baseline
        "128 128 2 16"
        "128 128 5 32"
        "128 128 10 64"
        "128 128 20 64"
        "128 128 30 64"
        "128 128 50 64"

        # prefill-heavy — rate sweep
        "512 128 5 32"
        "512 128 10 64"
        "512 128 20 64"
        "512 128 30 64"
        "1024 128 5 32"
        "1024 128 10 64"
        "1024 128 20 64"

        # decode-heavy — rate sweep
        "128 512 5 32"
        "128 512 10 64"
        "128 512 20 64"
        "128 512 30 64"
        "128 1024 5 32"
        "128 1024 10 64"
        "128 1024 20 64"

        # extreme decode-heavy
        "64 2048 5 32"
        "64 2048 10 64"
    )
fi

# Experiment E default knobs.
# Smoke mode is for proving the xPyD path works at all.
# Full matrix is the original large benchmark envelope.
E_SMOKE="${E_SMOKE:-0}"
E_FULL_MATRIX="${E_FULL_MATRIX:-0}"
E_NUM_PROMPTS="${E_NUM_PROMPTS:-200}"
E_NUM_WARMUPS="${E_NUM_WARMUPS:-1}"

# Characterization owns its workload matrix and repeat counts.  The outer E
# lifecycle must therefore start exactly one fixed P0+D0 serving stack.
[ -n "${XPYD_PHASE3B_CHARACTERIZATION_CONFIG}" ] && E_SMOKE=1

if [ "${E_SMOKE}" = "1" ]; then
    E_NUM_PROMPTS=20
    E_NUM_WARMUPS=1
    E_DISAGG_CONFIGS=("2520 2040")
    E_WORKLOADS=("128 128 1 4")
elif [ "${E_FULL_MATRIX}" = "1" ]; then
    E_DISAGG_CONFIGS=(
        "2520 2040"
        "2520 1410"
        "2010 1410"
        "2010 2040"
        "2520 1620"
    )
    E_WORKLOADS=("${WORKLOADS[@]}")
else
    E_DISAGG_CONFIGS=(
        "2520 2040"
        "2010 2040"
    )
    E_WORKLOADS=(
        "128 128 1 4"
        "128 128 2 8"
        "512 128 1 4"
        "128 512 1 4"
    )
fi

# Controlled Phase 3B characterization owns its fixed operating points in the
# characterization JSON.  Keep the outer E smoke envelope, but derive the
# actual P0/D0 lock frequencies from that source of truth so the launcher and
# the workload clock audit cannot silently diverge (for example, D0=1500 MHz).
if [ -n "${XPYD_PHASE3B_CHARACTERIZATION_CONFIG}" ]; then
    read -r _xpyd_characterization_p0_freq _xpyd_characterization_d0_freq < <(
        "${PYTHON_BIN}" - "${XPYD_PHASE3B_CHARACTERIZATION_CONFIG}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
clocks = config["fixed_clocks"]
print(int(clocks["P0"]["graphics_mhz"]), int(clocks["D0"]["graphics_mhz"]))
PY
    )
    E_DISAGG_CONFIGS=("${_xpyd_characterization_p0_freq} ${_xpyd_characterization_d0_freq}")
fi

# Shared filesystem path to gpu_monitor.py (must be accessible from all nodes)
GPU_MONITOR_SCRIPT="$(${REALPATH_BIN} gpu_monitor.py)"
MONITOR_INTERVAL=0.05   # 50ms — matches phase2_characterization.sh
# Memory frequencies (fixed; same as phase2_characterization.sh)
L40S_MEM_FREQ=9001
L4_MEM_FREQ=6251
CD_PREFILL_LABEL="${CD_PREFILL_LABEL:-l40s}"
CD_DECODE_LABEL="${CD_DECODE_LABEL:-l4}"
CD_PREFILL_PREFIX="${CD_PREFILL_PREFIX:-prefill_${CD_PREFILL_LABEL}}"
CD_DECODE_PREFIX="${CD_DECODE_PREFIX:-decode_${CD_DECODE_LABEL}}"
CD_PREFILL_MONITOR_PREFIX="${CD_PREFILL_MONITOR_PREFIX:-monitor_${CD_PREFILL_PREFIX}}"
CD_DECODE_MONITOR_PREFIX="${CD_DECODE_MONITOR_PREFIX:-monitor_${CD_DECODE_PREFIX}}"
CD_PREFILL_PORT="${CD_PREFILL_PORT:-${PREFILL_PORT}}"
CD_DECODE_PORT="${CD_DECODE_PORT:-${DECODE_PORT}}"
CD_PREFILL_MEM_FREQ="${CD_PREFILL_MEM_FREQ:-${L40S_MEM_FREQ}}"
CD_DECODE_MEM_FREQ="${CD_DECODE_MEM_FREQ:-${L4_MEM_FREQ}}"
CD_PREFILL_FREQS="${CD_PREFILL_FREQS:-2520 2265 2010 1755 1500 1245}"
CD_DECODE_FREQS="${CD_DECODE_FREQS:-2040 1830 1620 1410 1200 990}"
CD_PREFILL_TPS="${CD_PREFILL_TPS:-1}"
CD_DECODE_TPS="${CD_DECODE_TPS:-1}"
CD_PREFILL_ILS="${CD_PREFILL_ILS:-128 256 512 1024 2048}"
CD_DECODE_ILS="${CD_DECODE_ILS:-2}"
CD_DECODE_OLS="${CD_DECODE_OLS:-64 128 256 512 1024}"
CD_PREFILL_RATES="${CD_PREFILL_RATES:-5 10 20 30 50}"
CD_DECODE_RATES="${CD_DECODE_RATES:-5 10 20 30 50}"

if [ "${WORKLOAD_MODE}" = "trace" ]; then
    TRACE_SINGLETON_WORKLOAD="${TRACE_SINGLETON_WORKLOAD:-1 1 1 64}"
    WORKLOADS=("${TRACE_SINGLETON_WORKLOAD}")
    E_WORKLOADS=("${TRACE_SINGLETON_WORKLOAD}")
    TRACE_PREFILL_MAX_CONCURRENCY="${TRACE_PREFILL_MAX_CONCURRENCY:-64}"
    TRACE_DECODE_MAX_CONCURRENCY="${TRACE_DECODE_MAX_CONCURRENCY:-64}"
    CD_PREFILL_ILS="${TRACE_PREFILL_ILS:-1}"
    CD_PREFILL_RATES="${TRACE_PREFILL_RATES:-1}"
    CD_DECODE_ILS="${TRACE_DECODE_ILS:-1}"
    CD_DECODE_OLS="${TRACE_DECODE_OLS:-1}"
    CD_DECODE_RATES="${TRACE_DECODE_RATES:-1}"
fi

CD_EXTENDED_NAMES="${CD_EXTENDED_NAMES:-0}"

# Global monitor PID arrays (cleared per experiment)
MONITOR_PIDS=()
PHASE3C_SERVER_PIDS=()

# Per-node server PIDs — set by start_vllm_server, cleared by stop_server.
# Used by the EXIT trap and by stop_server for precise kill (not broad pkill).
L40S_SERVER_PID=""
L4_SERVER_PID=""

# Node where this batch script is executing (first in --nodelist)
# Use the short hostname to match L40S_NODE/L4_NODE (which are short names).
# hostname -s is more reliable than SLURMD_NODENAME, which can be FQDN.
LOCAL_NODE="$(${HOSTNAME_BIN} -s)"

# ---------------------------------------------------------------------------
# Helper: verify that every node needed by the selected experiment set is
# actually present in the current Slurm allocation. This catches mismatches
# such as submitting EXP=A with a two-node hardcoded script but only getting
# europa, or submitting EXP=E with a one-node wrapper.
# ---------------------------------------------------------------------------
assert_active_nodes_allocated() {
    local allocated_nodes=()
    if [ -n "${SLURM_JOB_NODELIST:-}" ] && [ -x "${SCONTROL_BIN}" ]; then
        mapfile -t allocated_nodes < <("${SCONTROL_BIN}" show hostnames "${SLURM_JOB_NODELIST}")
    fi

    if [ "${#allocated_nodes[@]}" -eq 0 ] && [ -n "${LOCAL_NODE:-}" ]; then
        allocated_nodes=("${LOCAL_NODE}")
    fi

    local node found
    for node in "${ACTIVE_NODES[@]}"; do
        found=0
        for _alloc_node in "${allocated_nodes[@]}"; do
            if [ "${_alloc_node}" = "${node}" ]; then
                found=1
                break
            fi
        done
        if [ "${found}" -ne 1 ]; then
            echo "ERROR: selected experiment set '${EXP_RAW}' requires node '${node}',"
            echo "       but current allocation is: ${allocated_nodes[*]:-<unknown>}."
            echo "       Submit with a matching node list or use the dedicated wrapper script."
            exit 1
        fi
    done
}


# ---------------------------------------------------------------------------
# Helper: start a long-lived background process on a node.
#   - Local node: run directly (no SLURM step created; avoids "nodes are busy").
#   - Remote node: srun --overlap (allows this step to coexist with the vLLM
#     server step on the same node without triggering resource conflicts).
#   PID is appended to the caller's array (passed by name).
#
# Usage: _bg_start <node> <array_name> <command...>
# ---------------------------------------------------------------------------
_bg_start() {
    local node="$1"
    local -n _arr="$2"   # nameref to caller's array
    shift 2
    if [ "${node}" = "${LOCAL_NODE:-}" ]; then
        # Local node: run directly — zero SLURM step overhead, no "busy" risk.
        "$@" &
        _arr+=($!)
    else
        # Remote node: use srun --overlap so this step is allowed to share the
        # node's resources with any already-running steps (e.g. the vLLM server
        # step).  Without --overlap SLURM returns "Requested nodes are busy"
        # once the server has consumed all tracked GPU resources.
        # No SSH needed — avoids host-key / password-prompt issues entirely.
        srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" "$@" &
        _arr+=($!)
    fi
}

# ---------------------------------------------------------------------------
# Helper: send a signal to a PID that may be local or "node:pid" remote.
# ---------------------------------------------------------------------------
_bg_kill() {
    local sig="$1"; shift
    for pid in "$@"; do
        kill -"${sig}" "${pid}" 2>/dev/null || true
    done
}

# ---------------------------------------------------------------------------
# Helper: start a GPU monitor background process.
# Identical to _bg_start but adds --gpus-per-node=<n> to the remote srun
# step so SLURM's GPU cgroup gives the monitor process access to the GPU
# device files (without this, pynvml queries return None on remote nodes).
#
# Root cause of empty monitor CSVs on remote nodes:
#   SLURM enforces GPU device cgroup isolation per step.  A step with no
#   GPU GRES request gets an empty device cgroup → /dev/nvidia* not
#   accessible → NVML returns errors for all device queries → None values.
#   --overlap allows the monitor step to share the GPUs already claimed by
#   the vLLM server step without triggering "nodes are busy".
#
# Usage: _bg_start_mon <node> <array_name> <n_gpus> <command...>
# ---------------------------------------------------------------------------
_bg_start_mon() {
    local node="$1"
    local -n _arr_mon="$2"
    local n_gpus="$3"
    shift 3
    if [ "${node}" = "${LOCAL_NODE}" ]; then
        "$@" &
        _arr_mon+=($!)
    else
        srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
            --gpus-per-node="${n_gpus}" "$@" &
        _arr_mon+=($!)
    fi
}

# ---------------------------------------------------------------------------
# Helper: get node IP
# ---------------------------------------------------------------------------
get_node_ip() {
    local node=$1
    srun --nodes=1 --ntasks=1 --nodelist="${node}" "${HOSTNAME_BIN}" -I | "${AWK_BIN}" '{print $1}'
}

# ---------------------------------------------------------------------------
# Helper: set GPU application clocks (requires sudoers: NOPASSWD: /usr/bin/nvidia-smi)
# ---------------------------------------------------------------------------
set_gpu_freq() {
    local node=$1
    local gpu_freq=$2
    local tp=$3
    local mem_freq=$4

    echo "  [${node}] Setting GPU clocks: SM=${gpu_freq} MHz, MEM=${mem_freq} MHz (TP=${tp} GPUs)"
    # GPU indices must match start_vllm_server: tp=2 uses GPUs 0,1 on io.
    local gpu_list
    if [ "$tp" -eq 2 ]; then
        gpu_list="0 1"
    else
        gpu_list="\$(${SEQ_BIN} 0 $((tp - 1)))"
    fi
    local _freq_cmd="
        set -euo pipefail
        ${SUDO_BIN} ${NVIDIA_SMI_BIN} -pm 1
        for g in ${gpu_list}; do
            ${SUDO_BIN} ${NVIDIA_SMI_BIN} -i \$g -ac '${mem_freq},${gpu_freq}'
        done
        ${NVIDIA_SMI_BIN} --query-gpu=index,clocks.sm,clocks.mem --format=csv,noheader
    "
    if [ "${node}" = "${LOCAL_NODE}" ]; then
        "${BASH_BIN}" -c "${_freq_cmd}"
    else
        srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" "${BASH_BIN}" -c "${_freq_cmd}"
    fi
}

# ---------------------------------------------------------------------------
# Phase 3B controlled-characterization clock lock.
#
# This is intentionally separate from set_gpu_freq(): it does not enable
# persistence mode, alter power limits, or participate in a feedback loop.
# The selected graphics and memory clocks are locked once before serving and
# reset unconditionally by the EXIT trap.  The read-only Phase 3B NVML module
# remains free of all mutation operations.
# ---------------------------------------------------------------------------
_run_characterization_clock_command() {
    local node="$1"
    local command="$2"
    if [ "${node}" = "${LOCAL_NODE:-}" ]; then
        "${BASH_BIN}" -c "${command}"
    else
        srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
            "${BASH_BIN}" -c "${command}"
    fi
}

set_characterization_clocks() {
    local node="$1"
    local endpoint="$2"
    local gpu_freq="$3"
    local mem_freq="$4"
    local gpu_id="${5:-0}"
    local evidence="${XPYD_FIXED_CLOCK_EVIDENCE}"
    echo "  [${node}] Controlled characterization clock lock: GPU=${gpu_freq} MHz, MEM=${mem_freq} MHz"
    local command="
        set -euo pipefail
        {
            echo 'endpoint=${endpoint} node=${node} event=before_lock'
            ${NVIDIA_SMI_BIN} -i ${gpu_id} --query-gpu=index,name,uuid,pci.bus_id,clocks.current.graphics,clocks.current.memory --format=csv,noheader,nounits
            ${NVIDIA_SMI_BIN} -i ${gpu_id} -q -d SUPPORTED_CLOCKS || true
        } >> '${evidence}' 2>&1
        ${SUDO_BIN} ${NVIDIA_SMI_BIN} -i ${gpu_id} -lgc '${gpu_freq},${gpu_freq}'
        ${SUDO_BIN} ${NVIDIA_SMI_BIN} -i ${gpu_id} -lmc '${mem_freq},${mem_freq}'
        {
            echo 'endpoint=${endpoint} node=${node} event=after_lock target_graphics_mhz=${gpu_freq} target_memory_mhz=${mem_freq}'
            ${NVIDIA_SMI_BIN} -i ${gpu_id} --query-gpu=index,name,uuid,pci.bus_id,clocks.current.graphics,clocks.current.memory --format=csv,noheader,nounits
        } >> '${evidence}' 2>&1
    "
    _run_characterization_clock_command "${node}" "${command}"
}

reset_characterization_clocks() {
    local node="$1"
    local endpoint="$2"
    local gpu_id="${3:-0}"
    local evidence="${XPYD_FIXED_CLOCK_EVIDENCE}"
    [ -n "${evidence}" ] || return 0
    echo "  [${node}] Restoring default graphics and memory clocks"
    local command="
        set +e
        ${SUDO_BIN} ${NVIDIA_SMI_BIN} -i ${gpu_id} -rgc
        rc_graphics=\$?
        ${SUDO_BIN} ${NVIDIA_SMI_BIN} -i ${gpu_id} -rmc
        rc_memory=\$?
        {
            echo 'endpoint=${endpoint} node=${node} event=after_reset'
            ${NVIDIA_SMI_BIN} -i ${gpu_id} --query-gpu=index,name,uuid,pci.bus_id,clocks.current.graphics,clocks.current.memory --format=csv,noheader,nounits
            echo reset_graphics_rc=\${rc_graphics} reset_memory_rc=\${rc_memory}
        } >> '${evidence}' 2>&1
        [ \${rc_graphics} -eq 0 ] && [ \${rc_memory} -eq 0 ]
    "
    _run_characterization_clock_command "${node}" "${command}"
}

# ---------------------------------------------------------------------------
# Helper: start gpu_monitor.py per GPU on a node (background)
# Mirrors start_monitors() from phase2_characterization.sh
# ---------------------------------------------------------------------------
start_monitors() {
    local node=$1
    local log_prefix=$2
    local tp=$3

    # GPU indices must match start_vllm_server: tp=2 uses GPUs 0,1 on io.
    local gpu_start=0
    # [ "$tp" -eq 2 ] && gpu_start=4  # was needed for europa GPUs 4,5

    MONITOR_PIDS=()
    for ((g=gpu_start; g<gpu_start+tp; g++)); do
        _bg_start_mon "${node}" MONITOR_PIDS "${tp}" \
            "${PYTHON_BIN}" "${GPU_MONITOR_SCRIPT}" \
                --monitor \
                --interval ${MONITOR_INTERVAL} \
                --output "${log_prefix}_gpu${g}.csv" \
                --gpu-id ${g}
    done

    # Wait for monitors to initialize
    sleep 2
}

# ---------------------------------------------------------------------------
# Helper: stop all monitors gracefully (SIGTERM → save CSV)
# Mirrors cleanup_monitors() from phase2_characterization.sh
# ---------------------------------------------------------------------------
stop_monitors() {
    _bg_kill TERM "${MONITOR_PIDS[@]}"
    sleep 2
    _bg_kill KILL "${MONITOR_PIDS[@]}"
    MONITOR_PIDS=()
    "${SYNC_BIN}"; sleep 0.5
}

# ---------------------------------------------------------------------------
# Helper: compute avg_power / energy from gpu_monitor CSV files
# Uses the hardware energy counter (total_energy_mj column), windowed by
# start_ts/end_ts — identical logic to phase2_characterization.sh
# ---------------------------------------------------------------------------
compute_power_energy() {
    local log_prefix=$1
    local tp=$2
    local start_ts=$3
    local end_ts=$4

    local total_power=0
    local total_energy=0
    local total_samples=0

    for ((g=0; g<tp; g++)); do
        local log_file="${log_prefix}_gpu${g}.csv"
        [ -f "${log_file}" ] || continue

        # CSV columns: timestamp,datetime,power_w,total_energy_mj,gpu_freq_mhz,
        #              mem_freq_mhz,temperature_c,gpu_util_pct,mem_util_pct
        local result
        result=$("${AWK_BIN}" -F, -v s="${start_ts}" -v e="${end_ts}" '
            BEGIN { n=0; sp=0; fe=-1; le=-1 }
            NR>1 && $1>=s && $1<=e {
                sp += $3+0;
                if (fe<0) fe=$4+0; le=$4+0;
                n++;
            }
            END {
                avg_p = (n>0) ? sp/n : 0;
                energy_j = (fe>=0 && le>fe) ? (le-fe)/1000.0 : 0;
                printf "%.2f,%.3f,%d", avg_p, energy_j, n;
            }' "${log_file}")

        local g_power; g_power=$(echo "${result}" | cut -d, -f1)
        local g_energy; g_energy=$(echo "${result}" | cut -d, -f2)
        local g_n;     g_n=$(echo "${result}" | cut -d, -f3)
        total_power=$("${AWK_BIN}" "BEGIN {print ${total_power}+${g_power}}")
        total_energy=$("${AWK_BIN}" "BEGIN {print ${total_energy}+${g_energy}}")
        total_samples=$((total_samples + g_n))
    done

    echo "${total_power},${total_energy},${total_samples}"
}

# ---------------------------------------------------------------------------
# Helper: start vLLM server and wait for ready
# ---------------------------------------------------------------------------
pick_gpu_ids() {
    local requested_tp=$1
    local raw_ids="${2:-}"
    local ids=()
    local picked=()
    local i

    if [ -z "${raw_ids}" ]; then
        return 1
    fi

    IFS=',' read -r -a ids <<< "${raw_ids}"
    if [ "${#ids[@]}" -lt "${requested_tp}" ]; then
        echo "ERROR: requested TP=${requested_tp}, but GPU override '${raw_ids}' exposes only ${#ids[@]} GPU(s)." >&2
        return 2
    fi

    for ((i=0; i<requested_tp; i++)); do
        picked+=("${ids[$i]}")
    done

    local joined
    joined=$(IFS=,; echo "${picked[*]}")
    echo "${joined}"
}

start_vllm_server() {
    local node=$1
    local port=$2
    local tp=$3
    local extra_args="${4:-}"
    # 5th arg: "log" = enable request logging (for debug); anything else = suppress
    local req_log_flag="--disable-log-requests"
    [ "${5:-}" = "log" ] && req_log_flag=""
    local explicit_cuda_devs="${6:-}"
    local log_file="${RESULT_DIR}/vllm_${node}_${port}.log"

    # Pin GPU indices explicitly so TP=2/4 always gets the intended devices.
    # Without this, a leftover CUDA context on GPU 0 can cause TP=4 to hang.
    # TP=2: use GPUs 0,1 on io node.
    local cuda_devs
    local gpu_override=""
    if [ -n "${explicit_cuda_devs}" ]; then
        gpu_override="${explicit_cuda_devs}"
    elif [ "${node}" = "${L40S_NODE}" ] && [ -n "${L40S_GPU_IDS}" ]; then
        gpu_override="${L40S_GPU_IDS}"
    elif [ "${node}" = "${L4_NODE}" ] && [ -n "${L4_GPU_IDS}" ]; then
        gpu_override="${L4_GPU_IDS}"
    fi

    if [ -n "${gpu_override}" ]; then
        cuda_devs=$(pick_gpu_ids "${tp}" "${gpu_override}") || return 1
    elif [ "$tp" -eq 2 ]; then
        cuda_devs="0,1"
    else
        cuda_devs=$("${SEQ_BIN}" -s, 0 $((tp - 1)))
    fi

    # Start vLLM server directly on local node (no srun step — avoids "nodes
    # are busy") or via srun --overlap on remote node.
    # Save PID so stop_server can kill the exact process instead of broad pkill.
    local _server_pid
    if [ "${node}" = "${LOCAL_NODE}" ]; then
        CUDA_VISIBLE_DEVICES="${cuda_devs}" \
        "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
            --model "${MODEL}" \
            --host 0.0.0.0 \
            --port "${port}" \
            --tensor-parallel-size "${tp}" \
            ${req_log_flag} \
            --max-model-len 4096 \
            ${extra_args} \
            > "${log_file}" 2>&1 &
        _server_pid=$!
        [ -n "${XPYD_PHASE3C_CONFIG:-}${XPYD_PHASE3D_CONFIG:-}${XPYD_PHASE4A_CONFIG:-}${XPYD_PHASE4B_CONFIG:-}${XPYD_PHASE4B1_CONFIG:-}" ] && PHASE3C_SERVER_PIDS+=("${_server_pid}")
        # Record in the appropriate global for the EXIT trap
        if [ "${port}" = "${PREFILL_PORT}" ]; then L40S_SERVER_PID=${_server_pid}
        else                                        L4_SERVER_PID=${_server_pid}; fi
    else
        local step_gpu_count="${tp}"
        local step_gpu_bind="map_gpu:${cuda_devs}"
        local step_environment=()
        if [ -n "${XPYD_PHASE3C_CONFIG:-}${XPYD_PHASE3D_CONFIG:-}${XPYD_PHASE4A_CONFIG:-}${XPYD_PHASE4B_CONFIG:-}${XPYD_PHASE4B1_CONFIG:-}" ]; then
            # Both Phase 3C servers on this node need cgroup access to the
            # explicitly pinned GPU0/1 set. Each process still sees one GPU.
            step_gpu_count=2
            step_gpu_bind="none"
            step_environment=(env "CUDA_VISIBLE_DEVICES=${cuda_devs}")
        fi
        srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
            --gpus-per-node="${step_gpu_count}" \
            --gpu-bind="${step_gpu_bind}" \
            "${step_environment[@]}" \
            "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
                --model "${MODEL}" \
                --host 0.0.0.0 \
                --port "${port}" \
                --tensor-parallel-size "${tp}" \
                ${req_log_flag} \
                --max-model-len 4096 \
                ${extra_args} \
            > "${log_file}" 2>&1 &
        _server_pid=$!
        [ -n "${XPYD_PHASE3C_CONFIG:-}${XPYD_PHASE3D_CONFIG:-}${XPYD_PHASE4A_CONFIG:-}${XPYD_PHASE4B_CONFIG:-}${XPYD_PHASE4B1_CONFIG:-}" ] && PHASE3C_SERVER_PIDS+=("${_server_pid}")
        if [ "${port}" = "${PREFILL_PORT}" ]; then L40S_SERVER_PID=${_server_pid}
        else                                        L4_SERVER_PID=${_server_pid}; fi
    fi

    # Health-check poll: 127.0.0.1 for local (no srun needed), remote IP for remote.
    local node_ip
    if [ "${node}" = "${LOCAL_NODE}" ]; then
        node_ip="127.0.0.1"
    else
        node_ip=$(srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" "${HOSTNAME_BIN}" -I | "${AWK_BIN}" '{print $1}')
    fi
    echo "[${node}] Waiting for vLLM server on port ${port} (${node_ip}), TP=${tp}, GPUs=${cuda_devs}..."
    local elapsed=0
    while ! "${CURL_BIN}" -s --connect-timeout 5 "http://${node_ip}:${port}/health" > /dev/null 2>&1; do
        # Fast-fail: if the server process died already, no point waiting 1200s
        if ! kill -0 "${_server_pid}" 2>/dev/null; then
            echo "[${node}] ERROR: vLLM server process exited early — check ${log_file}"
            "${TAIL_BIN}" -20 "${log_file}" || true
            return 1
        fi
        sleep 10
        elapsed=$((elapsed + 10))
        if [ ${elapsed} -ge "${VLLM_STARTUP_TIMEOUT_S}" ]; then
            echo "[${node}] ERROR: vLLM server failed to start within ${VLLM_STARTUP_TIMEOUT_S}s"
            return 1
        fi
        echo "  [${node}] ... ${elapsed}s"
    done
    echo "[${node}] vLLM server ready (${elapsed}s)"
}

# ---------------------------------------------------------------------------
# Helper: aggressive GPU cleanup between TP transitions
#
# When switching TP degree (e.g., TP=1 → TP=2), NCCL/CUDA contexts from the
# previous configuration can leave GPUs in a stuck state (100% util, no PID).
# The normal stop_server gpu-reset + 3s sleep is sometimes insufficient.
#
# This function performs:
#   1. Kill ALL lingering vLLM/python/NCCL processes on the node
#   2. Double gpu-reset (first clears contexts, second verifies clean state)
#   3. Verify no GPU processes remain via nvidia-smi
#   4. 15s cooldown for driver state to fully settle
# ---------------------------------------------------------------------------
_tp_transition_cleanup() {
    local node=$1

    if [ "${node}" = "${LOCAL_NODE}" ]; then
        # Kill any lingering processes
        "${PKILL_BIN}" -KILL -f 'vllm.entrypoints' 2>/dev/null || true
        "${PKILL_BIN}" -KILL -f 'python.*vllm' 2>/dev/null || true
        "${PKILL_BIN}" -KILL -f 'python.*torch.distributed' 2>/dev/null || true
        sleep 3

        # First gpu-reset to clear zombie NCCL contexts
        echo "    [${node}] gpu-reset #1..."
        _reset_gpu_state "${node}"
        sleep 5

        # Second gpu-reset for verification
        echo "    [${node}] gpu-reset #2..."
        _reset_gpu_state "${node}"
        sleep 3

        # Verify no GPU processes remain
        echo "    [${node}] verifying GPU state..."
        "${NVIDIA_SMI_BIN}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read -r pid; do
            if [ -n "${pid}" ] && [ "${pid}" != "[N/A]" ]; then
                echo "    [${node}] WARNING: killing leftover GPU process PID=${pid}"
                kill -KILL "${pid}" 2>/dev/null || true
            fi
        done

        # Final cooldown
        sleep 5
        echo "    [${node}] GPU state cleanup complete"
    else
        # Remote node: same logic via srun
        srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
            "${PKILL_BIN}" -KILL -f 'vllm.entrypoints' 2>/dev/null || true
        srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
            "${PKILL_BIN}" -KILL -f 'python.*vllm' 2>/dev/null || true
        srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
            "${PKILL_BIN}" -KILL -f 'python.*torch.distributed' 2>/dev/null || true
        sleep 3

        echo "    [${node}] gpu-reset #1..."
        _reset_gpu_state "${node}"
        sleep 5

        echo "    [${node}] gpu-reset #2..."
        _reset_gpu_state "${node}"
        sleep 3

        echo "    [${node}] verifying GPU state..."
        srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
            "${BASH_BIN}" -c "${NVIDIA_SMI_BIN} --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read pid; do [ -n \"\${pid}\" ] && [ \"\${pid}\" != '[N/A]' ] && kill -KILL \"\${pid}\" 2>/dev/null; done" || true

        sleep 5
        echo "    [${node}] GPU state cleanup complete"
    fi
}

# ---------------------------------------------------------------------------
# Helper: stop vLLM server and reset GPU clocks
# ---------------------------------------------------------------------------
stop_server() {
    local node=$1
    local port=$2
    # Root cause of GPU ERR / zombie-context with TP≥2:
    #   vLLM uses NCCL all-reduce across N GPUs.  SIGKILL mid-collective leaves
    #   peer GPUs' kernels stuck (100% util, no PID).  SIGTERM gives vLLM time
    #   to call torch.distributed.destroy_process_group() → safe NCCL teardown.
    #
    # Use the exact PID saved by start_vllm_server — more reliable than pkill.
    local _pid=""
    [ "${port}" = "${PREFILL_PORT}" ] && _pid="${L40S_SERVER_PID:-}"
    [ "${port}" = "${DECODE_PORT}"  ] && _pid="${L4_SERVER_PID:-}"

    if [ "${node}" = "${LOCAL_NODE}" ]; then
        # SIGTERM the exact server PID; also pkill as fallback for workers
        [ -n "${_pid}" ] && kill -TERM "${_pid}" 2>/dev/null || true
        "${PKILL_BIN}" -TERM -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
        # Poll every 2 s for up to 30 s to allow graceful NCCL/CUDA teardown
        local waited=0
        while [ -n "${_pid}" ] && kill -0 "${_pid}" 2>/dev/null; do
            sleep 2; waited=$((waited + 2))
            if [ ${waited} -ge 30 ]; then
                echo "  [${node}] vLLM did not exit in 30s — SIGKILL"
                [ -n "${_pid}" ] && kill -KILL "${_pid}" 2>/dev/null || true
                "${PKILL_BIN}" -KILL -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
                "${PKILL_BIN}" -KILL -f 'python.*vllm' 2>/dev/null || true
                sleep 5
                break
            fi
        done
        # Clear tracked PID
        [ "${port}" = "${PREFILL_PORT}" ] && L40S_SERVER_PID="" || L4_SERVER_PID=""
        # Reset GPU state — clears zombie NCCL/CUDA contexts from any abrupt kills
        _reset_gpu_state "${node}"
        sleep 3
    else
        # For remote node: SIGTERM via srun, fixed 30 s wait (no cheap remote poll)
        srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
            "${PKILL_BIN}" -TERM -f 'vllm.entrypoints.openai.api_server' || true
        echo "  [${node}] waiting 30s for graceful NCCL teardown..."
        sleep 30
        srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
            "${PKILL_BIN}" -KILL -f 'vllm.entrypoints.openai.api_server' || true
        srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
            "${PKILL_BIN}" -KILL -f 'python.*vllm' || true
        sleep 5
        [ "${port}" = "${PREFILL_PORT}" ] && L40S_SERVER_PID="" || L4_SERVER_PID=""
        _reset_gpu_state "${node}"
        sleep 3
    fi
}

# ---------------------------------------------------------------------------
# Helper: run vllm bench serve
# ---------------------------------------------------------------------------
run_bench() {
    local base_url=$1
    local input_len=$2
    local output_len=$3
    local request_rate=$4
    local max_concurrency=$5
    local output_file=$6

    local extra_args="${7:-}"

    vllm bench serve \
        --backend openai \
        --base-url "${base_url}" \
        --model "${MODEL}" \
        --num-prompts "${NUM_PROMPTS}" \
        --dataset-name random \
        --input-len "${input_len}" \
        --output-len "${output_len}" \
        --request-rate "${request_rate}" \
        --max-concurrency "${max_concurrency}" \
        --num-warmups "${NUM_WARMUPS}" \
        --percentile-metrics ttft,tpot,itl \
        ${extra_args} \
        2>&1 | /usr/bin/tee "${output_file}"
}

trace_summary_path() {
    local output_file=$1
    echo "${output_file%.txt}.trace_summary.json"
}

run_trace_replay() {
    local base_url=$1
    local input_len=$2
    local output_len=$3
    local request_rate=$4
    local max_concurrency=$5
    local output_file=$6
    local _trace_summary
    _trace_summary=$(trace_summary_path "${output_file}")

    "${PYTHON_BIN}" "${TRACE_REPLAY_SCRIPT}" \
        --base-url "${base_url}" \
        --model "${MODEL}" \
        --tokenizer-model "${TRACE_TOKENIZER_MODEL}" \
        --trace-csv "${TRACE_CSV}" \
        --output-file "${output_file}" \
        --summary-json "${_trace_summary}" \
        --max-concurrency "${max_concurrency}" \
        --num-warmups "${NUM_WARMUPS}" \
        --request-timeout-s "${TRACE_REQUEST_TIMEOUT_S}"
}

load_trace_window() {
    local output_file=$1
    local _trace_summary
    _trace_summary=$(trace_summary_path "${output_file}")
    "${PYTHON_BIN}" - "${_trace_summary}" << 'PY_EOF'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

start = data.get("timing_start_unix_s")
end = data.get("timing_end_unix_s")
if start is None or end is None:
    raise SystemExit("trace summary missing timing window")

print(f"{start:.6f} {end:.6f}")
PY_EOF
}

run_bench_or_trace() {
    local base_url=$1
    local input_len=$2
    local output_len=$3
    local request_rate=$4
    local max_concurrency=$5
    local output_file=$6
    local extra_args="${7:-}"

    if [ "${WORKLOAD_MODE}" = "trace" ]; then
        run_trace_replay \
            "${base_url}" "${input_len}" "${output_len}" "${request_rate}" \
            "${max_concurrency}" "${output_file}"
    else
        run_bench \
            "${base_url}" "${input_len}" "${output_len}" "${request_rate}" \
            "${max_concurrency}" "${output_file}" "${extra_args}"
    fi
}

# ===========================================================================
# EXPERIMENT A: Monolithic L40S
# ===========================================================================
run_experiment_A() {
    echo "=============================="
    echo "Experiment A: Monolithic L40S"
    echo "=============================="
    local exp_dir="${RESULT_DIR}/A_monolithic_l40s"
    mkdir -p "${exp_dir}"

    local prev_tp=0   # Track TP transitions for extra cleanup

    for cfg in ${L40S_CONFIGS}; do
        local gpu_freq=$(echo ${cfg} | cut -d: -f1)
        local tp=$(echo ${cfg} | cut -d: -f2)
        local mon_prefix="${exp_dir}/monitor_f${gpu_freq}_tp${tp}"

        # --- Check if ALL workloads for this config are already done ---
        # If so, skip the entire config (avoid unnecessary server start/stop
        # which can leave GPUs in a bad state on resume).
        local _all_done=1
        for wl in "${WORKLOADS[@]}"; do
            read -r _il _ol _rate _conc <<< "${wl}"
            local _chk="${exp_dir}/bench_f${gpu_freq}_tp${tp}_il${_il}_ol${_ol}_r${_rate}.txt"
            if [ ! -f "${_chk}" ]; then _all_done=0; break; fi
            local _s; _s=$(grep "Successful requests:" "${_chk}" 2>/dev/null | tail -1 | awk '{print $NF}')
            if [ "${_s:-0}" -le 0 ]; then _all_done=0; break; fi
        done
        if [ "${_all_done}" -eq 1 ]; then
            echo "  SKIP CONFIG (all ${#WORKLOADS[@]} workloads done): f${gpu_freq}_tp${tp}"
            prev_tp=${tp}
            continue
        fi

        # --- TP transition guard ---
        # When switching TP degree, NCCL/CUDA contexts from the previous TP
        # can leave GPUs in a stuck state.  Perform aggressive cleanup:
        #   1. Kill any lingering vLLM/python processes
        #   2. Double gpu-reset with verification
        #   3. 15s cooldown to let driver state fully settle
        if [ "${tp}" -ne "${prev_tp}" ] && [ "${prev_tp}" -ne 0 ]; then
            echo "  >>> TP transition: ${prev_tp} -> ${tp} — performing GPU state cleanup"
            _tp_transition_cleanup "${L40S_NODE}"
        fi
        prev_tp=${tp}

        echo "  L40S: freq=${gpu_freq} MHz, TP=${tp}"
        set_gpu_freq "${L40S_NODE}" "${gpu_freq}" "${tp}" "${L40S_MEM_FREQ}"
        start_vllm_server "${L40S_NODE}" "${PREFILL_PORT}" "${tp}"

        for wl in "${WORKLOADS[@]}"; do
            read -r il ol rate conc <<< "${wl}"
            local out="${exp_dir}/bench_f${gpu_freq}_tp${tp}_il${il}_ol${ol}_r${rate}.txt"

            # Skip already-completed workloads (safe resume).
            # Also re-run if file exists but has 0 successful requests (failed experiment).
            if [ -f "${out}" ] && grep -q "Successful requests:" "${out}" 2>/dev/null; then
                local _succ; _succ=$(grep "Successful requests:" "${out}" | tail -1 | awk '{print $NF}')
                if [ "${_succ:-0}" -gt 0 ]; then
                    echo "    SKIP (done, ${_succ} ok): f${gpu_freq}_tp${tp}_il${il}_ol${ol}_r${rate}"
                    continue
                fi
                echo "    RE-RUN (0 success): f${gpu_freq}_tp${tp}_il${il}_ol${ol}_r${rate}"
            fi

            echo "    il=${il} ol=${ol} rate=${rate}"

            start_monitors "${L40S_NODE}" "${mon_prefix}_il${il}_ol${ol}_r${rate}" "${tp}"
            local t_start
            local t_end
            if [ "${WORKLOAD_MODE}" = "trace" ]; then
                run_bench_or_trace "http://${L40S_NODE}:${PREFILL_PORT}" "${il}" "${ol}" "${rate}" "${conc}" "${out}"
                read -r t_start t_end < <(load_trace_window "${out}")
            else
                t_start=$("${PYTHON_BIN}" -c "import time; print(f'{time.time():.6f}')")
                run_bench_or_trace "http://${L40S_NODE}:${PREFILL_PORT}" "${il}" "${ol}" "${rate}" "${conc}" "${out}"
                t_end=$("${PYTHON_BIN}" -c "import time; print(f'{time.time():.6f}')")
            fi
            stop_monitors

            local pwr_metrics; pwr_metrics=$(compute_power_energy \
                "${mon_prefix}_il${il}_ol${ol}_r${rate}" "${tp}" "${t_start}" "${t_end}")
            echo "      power/energy: ${pwr_metrics}"
        done

        stop_server "${L40S_NODE}" "${PREFILL_PORT}"
    done
}

# ===========================================================================
# EXPERIMENT B: Monolithic L4
# ===========================================================================
run_experiment_B() {
    echo "==========================="
    echo "Experiment B: Monolithic L4"
    echo "==========================="
    local exp_dir="${RESULT_DIR}/B_monolithic_l4"
    mkdir -p "${exp_dir}"

    local prev_tp=0   # Track TP transitions for extra cleanup

    for cfg in ${L4_CONFIGS}; do
        local gpu_freq=$(echo ${cfg} | cut -d: -f1)
        local tp=$(echo ${cfg} | cut -d: -f2)
        local mon_prefix="${exp_dir}/monitor_f${gpu_freq}_tp${tp}"

        # --- Check if ALL workloads for this config are already done ---
        local _all_done=1
        for wl in "${WORKLOADS[@]}"; do
            read -r _il _ol _rate _conc <<< "${wl}"
            local _chk="${exp_dir}/bench_f${gpu_freq}_tp${tp}_il${_il}_ol${_ol}_r${_rate}.txt"
            if [ ! -f "${_chk}" ]; then _all_done=0; break; fi
            local _s; _s=$(grep "Successful requests:" "${_chk}" 2>/dev/null | tail -1 | awk '{print $NF}')
            if [ "${_s:-0}" -le 0 ]; then _all_done=0; break; fi
        done
        if [ "${_all_done}" -eq 1 ]; then
            echo "  SKIP CONFIG (all ${#WORKLOADS[@]} workloads done): f${gpu_freq}_tp${tp}"
            prev_tp=${tp}
            continue
        fi

        # --- TP transition guard ---
        if [ "${tp}" -ne "${prev_tp}" ] && [ "${prev_tp}" -ne 0 ]; then
            echo "  >>> TP transition: ${prev_tp} -> ${tp} — performing GPU state cleanup"
            _tp_transition_cleanup "${L4_NODE}"
        fi
        prev_tp=${tp}

        echo "  L4: freq=${gpu_freq} MHz, TP=${tp}"
        set_gpu_freq "${L4_NODE}" "${gpu_freq}" "${tp}" "${L4_MEM_FREQ}"
        start_vllm_server "${L4_NODE}" "${DECODE_PORT}" "${tp}"

        for wl in "${WORKLOADS[@]}"; do
            read -r il ol rate conc <<< "${wl}"
            local out="${exp_dir}/bench_f${gpu_freq}_tp${tp}_il${il}_ol${ol}_r${rate}.txt"

            # Skip already-completed workloads (safe resume).
            # Also re-run if file exists but has 0 successful requests (failed experiment).
            if [ -f "${out}" ] && grep -q "Successful requests:" "${out}" 2>/dev/null; then
                local _succ; _succ=$(grep "Successful requests:" "${out}" | tail -1 | awk '{print $NF}')
                if [ "${_succ:-0}" -gt 0 ]; then
                    echo "    SKIP (done, ${_succ} ok): f${gpu_freq}_tp${tp}_il${il}_ol${ol}_r${rate}"
                    continue
                fi
                echo "    RE-RUN (0 success): f${gpu_freq}_tp${tp}_il${il}_ol${ol}_r${rate}"
            fi

            echo "    il=${il} ol=${ol} rate=${rate}"

            start_monitors "${L4_NODE}" "${mon_prefix}_il${il}_ol${ol}_r${rate}" "${tp}"
            local t_start
            local t_end
            if [ "${WORKLOAD_MODE}" = "trace" ]; then
                run_bench_or_trace "http://${L4_NODE}:${DECODE_PORT}" "${il}" "${ol}" "${rate}" "${conc}" "${out}"
                read -r t_start t_end < <(load_trace_window "${out}")
            else
                t_start=$("${PYTHON_BIN}" -c "import time; print(f'{time.time():.6f}')")
                run_bench_or_trace "http://${L4_NODE}:${DECODE_PORT}" "${il}" "${ol}" "${rate}" "${conc}" "${out}"
                t_end=$("${PYTHON_BIN}" -c "import time; print(f'{time.time():.6f}')")
            fi
            stop_monitors

            local pwr_metrics; pwr_metrics=$(compute_power_energy \
                "${mon_prefix}_il${il}_ol${ol}_r${rate}" "${tp}" "${t_start}" "${t_end}")
            echo "      power/energy: ${pwr_metrics}"
        done

        stop_server "${L4_NODE}" "${DECODE_PORT}"
    done
}

# ===========================================================================
# EXPERIMENT C+D: Prefill-only (L40S) and Decode-only (L4)
#   output_len=1 approximates prefill-only workload
#   input_len=2  approximates decode-only (minimal KV cache)
# ===========================================================================
run_experiment_CD() {
    echo "========================================"
    echo "Experiments C+D: Prefill-only / Decode-only"
    echo "========================================"
    local exp_dir="${RESULT_DIR}/CD_prefill_decode_only"
    mkdir -p "${exp_dir}"

    if [ "${CD_EXTENDED_NAMES}" != "1" ]; then
        local _n_prefill_tp _n_decode_tp _n_decode_il
        _n_prefill_tp=$(printf '%s\n' ${CD_PREFILL_TPS} | ${WC_BIN} -l | tr -d ' ')
        _n_decode_tp=$(printf '%s\n' ${CD_DECODE_TPS} | ${WC_BIN} -l | tr -d ' ')
        _n_decode_il=$(printf '%s\n' ${CD_DECODE_ILS} | ${WC_BIN} -l | tr -d ' ')
        if [ "${_n_prefill_tp}" -gt 1 ] || [ "${_n_decode_tp}" -gt 1 ] || [ "${_n_decode_il}" -gt 1 ]; then
            echo "ERROR: CD_EXTENDED_NAMES=1 is required when sweeping multiple TP values or decode input lengths."
            echo "Set CD_EXTENDED_NAMES=1 so output filenames remain unique."
            exit 1
        fi
    fi

    # C-like path: configurable pool, output_len=1 (prefill-dominated).
    echo "  C-like phase-only run: ${CD_PREFILL_LABEL} prefill-only on ${CD_PREFILL_NODE} (output_len=1)"
    local prev_prefill_tp=0
    for tp in ${CD_PREFILL_TPS}; do
        if [ "${tp}" -ne "${prev_prefill_tp}" ] && [ "${prev_prefill_tp}" -ne 0 ]; then
            echo "  >>> Prefill TP transition: ${prev_prefill_tp} -> ${tp} — performing GPU state cleanup"
            _tp_transition_cleanup "${CD_PREFILL_NODE}"
        fi
        prev_prefill_tp=${tp}
        for gpu_freq in ${CD_PREFILL_FREQS}; do
            set_gpu_freq "${CD_PREFILL_NODE}" "${gpu_freq}" "${tp}" "${CD_PREFILL_MEM_FREQ}"
            start_vllm_server "${CD_PREFILL_NODE}" "${CD_PREFILL_PORT}" "${tp}"
            for il in ${CD_PREFILL_ILS}; do
                for rate in ${CD_PREFILL_RATES}; do
                    local suffix
                    if [ "${CD_EXTENDED_NAMES}" = "1" ]; then
                        suffix="_f${gpu_freq}_tp${tp}_il${il}_r${rate}"
                    else
                        suffix="_f${gpu_freq}_il${il}_r${rate}"
                    fi
                    local out="${exp_dir}/${CD_PREFILL_PREFIX}${suffix}.txt"
                    local mon_prefix="${exp_dir}/${CD_PREFILL_MONITOR_PREFIX}${suffix}"

                    # Skip already-completed workloads (safe resume)
                    if [ -f "${out}" ] && grep -q "Successful requests:" "${out}" 2>/dev/null; then
                        local _succ; _succ=$(grep "Successful requests:" "${out}" | tail -1 | awk '{print $NF}')
                        if [ "${_succ:-0}" -gt 0 ]; then
                            echo "    SKIP (done, ${_succ} ok): ${CD_PREFILL_PREFIX}${suffix}"
                            continue
                        fi
                        echo "    RE-RUN (0 success): ${CD_PREFILL_PREFIX}${suffix}"
                    fi

                    start_monitors "${CD_PREFILL_NODE}" "${mon_prefix}" "${tp}"
                    local t_start
                    local t_end
                    if [ "${WORKLOAD_MODE}" = "trace" ]; then
                        run_bench_or_trace "http://${CD_PREFILL_NODE}:${CD_PREFILL_PORT}" "${il}" "1" "${rate}" "${TRACE_PREFILL_MAX_CONCURRENCY}" "${out}"
                        read -r t_start t_end < <(load_trace_window "${out}")
                    else
                        t_start=$("${PYTHON_BIN}" -c "import time; print(f'{time.time():.6f}')")
                        run_bench_or_trace "http://${CD_PREFILL_NODE}:${CD_PREFILL_PORT}" "${il}" "1" "${rate}" "64" "${out}"
                        t_end=$("${PYTHON_BIN}" -c "import time; print(f'{time.time():.6f}')")
                    fi
                    stop_monitors

                    local pwr_metrics; pwr_metrics=$(compute_power_energy \
                        "${mon_prefix}" "${tp}" "${t_start}" "${t_end}")
                    echo "      power/energy: ${pwr_metrics}"
                    {
                        echo ""
                        echo "SWEEP-LLM Pilot Metadata"
                        echo "Timing window start (unix_s): ${t_start}"
                        echo "Timing window end (unix_s): ${t_end}"
                        echo "Measured cluster avg_power_w,energy_j,samples: ${pwr_metrics}"
                    } >> "${out}"
                done
            done
            stop_server "${CD_PREFILL_NODE}" "${CD_PREFILL_PORT}"
        done
    done

    # D-like path: configurable pool, configurable context (input_len) and output_len.
    echo "  D-like phase-only run: ${CD_DECODE_LABEL} decode-only on ${CD_DECODE_NODE}"
    local prev_decode_tp=0
    for tp in ${CD_DECODE_TPS}; do
        if [ "${tp}" -ne "${prev_decode_tp}" ] && [ "${prev_decode_tp}" -ne 0 ]; then
            echo "  >>> Decode TP transition: ${prev_decode_tp} -> ${tp} — performing GPU state cleanup"
            _tp_transition_cleanup "${CD_DECODE_NODE}"
        fi
        prev_decode_tp=${tp}
        for gpu_freq in ${CD_DECODE_FREQS}; do
            set_gpu_freq "${CD_DECODE_NODE}" "${gpu_freq}" "${tp}" "${CD_DECODE_MEM_FREQ}"
            start_vllm_server "${CD_DECODE_NODE}" "${CD_DECODE_PORT}" "${tp}"
            for il in ${CD_DECODE_ILS}; do
                for ol in ${CD_DECODE_OLS}; do
                    for rate in ${CD_DECODE_RATES}; do
                        local suffix
                        if [ "${CD_EXTENDED_NAMES}" = "1" ]; then
                            suffix="_f${gpu_freq}_tp${tp}_il${il}_ol${ol}_r${rate}"
                        else
                            suffix="_f${gpu_freq}_ol${ol}_r${rate}"
                        fi
                        local out="${exp_dir}/${CD_DECODE_PREFIX}${suffix}.txt"
                        local mon_prefix="${exp_dir}/${CD_DECODE_MONITOR_PREFIX}${suffix}"

                        # Skip already-completed workloads (safe resume)
                        if [ -f "${out}" ] && grep -q "Successful requests:" "${out}" 2>/dev/null; then
                            local _succ; _succ=$(grep "Successful requests:" "${out}" | tail -1 | awk '{print $NF}')
                            if [ "${_succ:-0}" -gt 0 ]; then
                                echo "    SKIP (done, ${_succ} ok): ${CD_DECODE_PREFIX}${suffix}"
                                continue
                            fi
                            echo "    RE-RUN (0 success): ${CD_DECODE_PREFIX}${suffix}"
                        fi

                        start_monitors "${CD_DECODE_NODE}" "${mon_prefix}" "${tp}"
                        local t_start
                        local t_end
                        if [ "${WORKLOAD_MODE}" = "trace" ]; then
                            run_bench_or_trace "http://${CD_DECODE_NODE}:${CD_DECODE_PORT}" "${il}" "${ol}" "${rate}" "${TRACE_DECODE_MAX_CONCURRENCY}" "${out}"
                            read -r t_start t_end < <(load_trace_window "${out}")
                        else
                            t_start=$("${PYTHON_BIN}" -c "import time; print(f'{time.time():.6f}')")
                            run_bench_or_trace "http://${CD_DECODE_NODE}:${CD_DECODE_PORT}" "${il}" "${ol}" "${rate}" "64" "${out}"
                            t_end=$("${PYTHON_BIN}" -c "import time; print(f'{time.time():.6f}')")
                        fi
                        stop_monitors

                        local pwr_metrics; pwr_metrics=$(compute_power_energy \
                            "${mon_prefix}" "${tp}" "${t_start}" "${t_end}")
                        echo "      power/energy: ${pwr_metrics}"
                        {
                            echo ""
                            echo "SWEEP-LLM Pilot Metadata"
                            echo "Timing window start (unix_s): ${t_start}"
                            echo "Timing window end (unix_s): ${t_end}"
                            echo "Measured cluster avg_power_w,energy_j,samples: ${pwr_metrics}"
                        } >> "${out}"
                    done
                done
            done
            stop_server "${CD_DECODE_NODE}" "${CD_DECODE_PORT}"
        done
    done
}

# ===========================================================================
# EXPERIMENT E: Disaggregated vLLM (L40S prefill → L4 decode)
#
# Requires vLLM >= 0.6.x with disaggregated prefill support.
# Uses PyNcclConnector for cross-node KV cache transfer over network.
#
# Monitors both L40S and L4 nodes per workload using gpu_monitor.py,
# with t_start/t_end timestamps and hardware energy counter — same
# pattern as A/B/CD, but with two independent PID arrays.
# ---------------------------------------------------------------------------
# Phase 3C: smallest extension of the accepted Experiment E lifecycle.
# Two persistent TP=1 producers and two persistent TP=1 consumers are pinned
# to GPUs 0/1 on the already allocated L40S/L4 nodes. The existing proxy core,
# client, metrics parser, and NVML monitor are reused by the Phase 3C harness.
# ---------------------------------------------------------------------------
run_phase3c_substrate() {
    local multi_config="${XPYD_PHASE3C_CONFIG:-${XPYD_PHASE3D_CONFIG:-${XPYD_PHASE4A_CONFIG:-${XPYD_PHASE4B_CONFIG:-${XPYD_PHASE4B1_CONFIG}}}}}"
    local p_ip d_ip p_http d_http
    p_ip=$(get_node_ip "${L40S_NODE}")
    d_ip=$(get_node_ip "${L4_NODE}")
    p_http="${p_ip}"
    d_http="${d_ip}"
    [ "${LOCAL_NODE}" = "${L40S_NODE}" ] && p_http="127.0.0.1"
    [ "${LOCAL_NODE}" = "${L4_NODE}" ] && d_http="127.0.0.1"

    export XPYD_P0_HTTP_HOST="${p_http}" XPYD_P1_HTTP_HOST="${p_http}"
    export XPYD_D0_HTTP_HOST="${d_http}" XPYD_D1_HTTP_HOST="${d_http}"
    export XPYD_P_ADDR_HOST="${p_ip}" XPYD_D_ADDR_HOST="${d_ip}"
    export XPYD_PROXY_URI="http://127.0.0.1:${PROXY_PORT}"
    export XPYD_PROXY_DIAGNOSTICS_LOG="${RESULT_DIR}/xpyd_phase3c_proxy_${SLURM_JOB_ID:-local}.jsonl"
    export XPYD_P0_SERVER_LOG="${RESULT_DIR}/vllm_${L40S_NODE}_8100.log"
    export XPYD_P1_SERVER_LOG="${RESULT_DIR}/vllm_${L40S_NODE}_8101.log"
    export XPYD_D0_SERVER_LOG="${RESULT_DIR}/vllm_${L4_NODE}_8200.log"
    export XPYD_D1_SERVER_LOG="${RESULT_DIR}/vllm_${L4_NODE}_8201.log"
    export PYTHONPATH="${SCRIPT_DIR}/paper/scripts${PYTHONPATH:+:${PYTHONPATH}}"
    export XPYD_PHASE3D_ACCEPTED_ACTUATOR_AUDIT
    export XPYD_PHASE4A_ACCEPTED_ACTUATOR_AUDIT XPYD_PHASE4A_ACCEPTED_CLOSED_LOOP_AUDIT
    export XPYD_PHASE4B_ORACLE_SUMMARY
    export XPYD_PHASE4B_ACCEPTED_STATIONARY_AUDIT
    export XPYD_PHASE4B_ACCEPTED_ACTIVE_SMOKE_AUDIT
    if [ -n "${XPYD_PHASE3D_CONFIG}" ]; then
        if [ "${XPYD_PHASE3D_STAGE}" = "B" ]; then
            XPYD_ROUTING_CONTROL_FILE="${XPYD_ROUTING_CONTROL_FILE:-${RESULT_DIR}/xpyd_phase3d_route_control_${SLURM_JOB_ID:-local}.json}"
        else
            XPYD_ROUTING_CONTROL_FILE=""
        fi
        export XPYD_ROUTING_CONTROL_FILE
    elif [ -n "${XPYD_PHASE4A_CONFIG}" ]; then
        XPYD_ROUTING_CONTROL_FILE="${XPYD_ROUTING_CONTROL_FILE:-${RESULT_DIR}/xpyd_phase4a_route_control_${SLURM_JOB_ID:-local}.json}"
        export XPYD_ROUTING_CONTROL_FILE
    elif [ -n "${XPYD_PHASE4B_CONFIG}${XPYD_PHASE4B1_CONFIG}" ]; then
        XPYD_ROUTING_CONTROL_FILE="${XPYD_ROUTING_CONTROL_FILE:-${RESULT_DIR}/xpyd_phase4b_route_control_${SLURM_JOB_ID:-local}.json}"
        export XPYD_ROUTING_CONTROL_FILE
    fi

    set_characterization_clocks "${L40S_NODE}" P0 2520 "${L40S_MEM_FREQ}" 0
    set_characterization_clocks "${L40S_NODE}" P1 2520 "${L40S_MEM_FREQ}" 1
    set_characterization_clocks "${L4_NODE}" D0 1500 "${L4_MEM_FREQ}" 0
    set_characterization_clocks "${L4_NODE}" D1 1500 "${L4_MEM_FREQ}" 1

    local p0_kv p1_kv d0_kv d1_kv
    p0_kv='{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_port":14579,"kv_connector_extra_config":{"send_type":"PUT"}}'
    p1_kv='{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_port":14580,"kv_connector_extra_config":{"send_type":"PUT"}}'
    d0_kv='{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_port":14579}'
    d1_kv='{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_port":14580}'

    # P2pNcclConnector does not provide cache-coherence metadata across a
    # dynamically selected P->D pair.  A producer-local prefix-cache hit can
    # therefore suppress the KV send even though the newly selected consumer
    # does not own that prefix.  Its v0.15.1 chunked-prefill path also completed
    # a 2049-token producer request without emitting any KV layer tensors.
    # Keep both optimizations disabled on this dynamic 2P2D path until their
    # connector semantics are validated independently.
    local no_local_prefix_cache="--no-enable-prefix-caching"
    local no_chunked_prefill="--no-enable-chunked-prefill"
    start_vllm_server "${L40S_NODE}" 8100 1 "--kv-transfer-config ${p0_kv} ${no_local_prefix_cache} ${no_chunked_prefill}" "" 0
    start_vllm_server "${L40S_NODE}" 8101 1 "--kv-transfer-config ${p1_kv} ${no_local_prefix_cache} ${no_chunked_prefill}" "" 1
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    start_vllm_server "${L4_NODE}" 8200 1 "--kv-transfer-config ${d0_kv} --gpu-memory-utilization 0.82 ${no_local_prefix_cache} ${no_chunked_prefill}" log 0
    start_vllm_server "${L4_NODE}" 8201 1 "--kv-transfer-config ${d1_kv} --gpu-memory-utilization 0.82 ${no_local_prefix_cache} ${no_chunked_prefill}" log 1
    unset PYTORCH_CUDA_ALLOC_CONF

    "${PYTHON_BIN}" -m xpyd.disagg_proxy \
        --multi-endpoint-config "${multi_config}" \
        --proxy-port "${PROXY_PORT}" \
        --diagnostics-log "${XPYD_PROXY_DIAGNOSTICS_LOG}" &
    local proxy_pid=$!
    PHASE3C_SERVER_PIDS+=("${proxy_pid}")
    local elapsed=0
    while ! "${CURL_BIN}" -s --connect-timeout 2 "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ "${elapsed}" -ge 60 ]; then
            echo "  ERROR: Phase 3C proxy failed readiness"
            return 1
        fi
    done

    local args
    if [ -n "${XPYD_PHASE4B1_CONFIG}" ]; then
        args=(-m xpyd.phase4b1_evaluation --config "${XPYD_PHASE4B1_CONFIG}")
        [ -n "${XPYD_PHASE4B1_RUN_ID}" ] && args+=(--run-id "${XPYD_PHASE4B1_RUN_ID}")
    elif [ -n "${XPYD_PHASE4B_CONFIG}" ]; then
        args=(-m xpyd.phase4b_evaluation --config "${XPYD_PHASE4B_CONFIG}")
        [ -n "${XPYD_PHASE4B_RUN_ID}" ] && args+=(--run-id "${XPYD_PHASE4B_RUN_ID}")
        [ "${XPYD_PHASE4B_SMOKE}" = "1" ] && args+=(--smoke)
        args+=(--stage "${XPYD_PHASE4B_STAGE}")
    elif [ -n "${XPYD_PHASE4A_CONFIG}" ]; then
        args=(-m xpyd.phase4a_oracle --config "${XPYD_PHASE4A_CONFIG}")
        [ -n "${XPYD_PHASE4A_RUN_ID}" ] && args+=(--run-id "${XPYD_PHASE4A_RUN_ID}")
    elif [ -n "${XPYD_PHASE3D_CONFIG}" ]; then
        args=(-m xpyd.phase3d_control --config "${XPYD_PHASE3D_CONFIG}" --stage "${XPYD_PHASE3D_STAGE}")
        [ -n "${XPYD_PHASE3D_RUN_ID}" ] && args+=(--run-id "${XPYD_PHASE3D_RUN_ID}")
    else
        args=(-m xpyd.phase3c_substrate --config "${XPYD_PHASE3C_CONFIG}")
        [ -n "${XPYD_PHASE3C_RUN_ID}" ] && args+=(--run-id "${XPYD_PHASE3C_RUN_ID}")
    fi
    local status=0
    "${PYTHON_BIN}" "${args[@]}" || status=$?
    kill "${proxy_pid}" 2>/dev/null || true
    wait "${proxy_pid}" 2>/dev/null || true
    stop_server "${L40S_NODE}" 8100
    stop_server "${L4_NODE}" 8200
    return "${status}"
}

# ===========================================================================
run_experiment_E() {
    echo "=================================="
    echo "Experiment E: Disaggregated vLLM"
    echo "=================================="
    local exp_dir="${RESULT_DIR}/E_disaggregated"
    mkdir -p "${exp_dir}"

    # Check vLLM version supports disagg
    local vllm_ver
    vllm_ver=$("${PYTHON_BIN}" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
    echo "  vLLM version: ${vllm_ver}"
    if [ -n "${XPYD_PHASE3A_CONFIG}${XPYD_PHASE3B_CONFIG}${XPYD_PHASE3B_CHARACTERIZATION_CONFIG}${XPYD_PHASE3C_CONFIG}${XPYD_PHASE3D_CONFIG}${XPYD_PHASE4A_CONFIG}${XPYD_PHASE4B_CONFIG}${XPYD_PHASE4B1_CONFIG}" ] && [ "${vllm_ver}" != "0.15.1" ]; then
        echo "  ERROR: XpYd observation path requires vLLM version 0.15.1 (got ${vllm_ver})"
        return 1
    fi

    # ---------------------------------------------------------------------------
    # Diagnostic patch: add a 60-second timeout to P2pNcclEngine.recv_tensor so
    # that if KV tensors are not found in recv_store within 60 s, the engine
    # logs the tensor_id it is waiting for, logs what IS in recv_store (key
    # mismatch detection), and returns None (triggering kv_load_failure_policy=
    # recompute) instead of blocking the EngineCore step loop forever.
    #
    # Without this patch, a missing / miskeyed tensor causes total silence:
    # no throughput logs, no responses, 900-second proxy timeout per request.
    # With it, we get a warning every 5 s showing exactly which tensor_id is
    # missing and what keys are present, then a graceful fallback.
    # ---------------------------------------------------------------------------
    local _p2p_engine_file
    _p2p_engine_file=$("${PYTHON_BIN}" -c \
        "import vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine as m; print(m.__file__)" \
        2>/dev/null || echo "")
    if [ -n "${_p2p_engine_file}" ]; then
        echo "  Patching recv_tensor with diagnostic timeout: ${_p2p_engine_file}"
        "${PYTHON_BIN}" - "${_p2p_engine_file}" << 'RECV_PATCH_EOF'
import sys, re

path = sys.argv[1]
with open(path, "r") as f:
    src = f.read()

# Only patch once — skip if already patched
if "recv_tensor TIMEOUT" in src:
    print(f"  [patch] already applied, skipping")
    sys.exit(0)

old = (
    "                while tensor_id not in self.recv_store:\n"
    "                    self.recv_store_cv.wait()\n"
    "                tensor = self.recv_store[tensor_id]"
)
new = (
    "                _waited = 0\n"
    "                while tensor_id not in self.recv_store:\n"
    "                    _ok = self.recv_store_cv.wait(timeout=5.0)\n"
    "                    if not _ok:\n"
    "                        _waited += 5\n"
    "                        logger.warning(\n"
    "                            \"⏰recv_tensor TIMEOUT %ds: waiting for \"\n"
    "                            \"tensor_id=[%s] | recv_store has %d keys: %s \"\n"
    "                            \"| rank=%d\",\n"
    "                            _waited, tensor_id, len(self.recv_store),\n"
    "                            sorted(self.recv_store.keys())[:5], self.rank,\n"
    "                        )\n"
    "                        if _waited >= 60:\n"
    "                            logger.warning(\n"
    "                                \"⏰recv_tensor GIVE_UP after %ds for \"\n"
    "                                \"tensor_id=[%s]; returning None\",\n"
    "                                _waited, tensor_id,\n"
    "                            )\n"
    "                            return None\n"
    "                tensor = self.recv_store[tensor_id]"
)

if old not in src:
    print(f"  [patch] ERROR: expected recv_tensor pattern not found in {path}")
    print("  [patch] Snippet around 'recv_store_cv.wait':")
    for i, line in enumerate(src.splitlines()):
        if "recv_store_cv.wait" in line:
            print(f"    L{i}: {line!r}")
    sys.exit(1)

patched = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(patched)
print(f"  [patch] recv_tensor timeout patch applied to {path}")
RECV_PATCH_EOF
    else
        echo "  WARNING: could not locate p2p_nccl_engine.py — skipping recv_tensor patch"
    fi

    # ---------------------------------------------------------------------------
    # Root-cause fix: canonical tensor_id in P2pNcclConnector
    #
    # vLLM internally wraps the user-supplied request_id as:
    #   generate-tokens-{user_id}-{random8hex}
    # where random8hex is generated *independently* on each server.
    # So neptune's tensor key ends in e.g. -8707aa6f and europa's lookup ends
    # in e.g. -9d3c1320 — they never match → europa waits forever.
    #
    # Fix: add _tensor_key() static method that strips both the
    # "generate-tokens-" prefix and the trailing "-[0-9a-f]{8}" suffix,
    # recovering the stable user-supplied request_id that is identical on both
    # servers.  Use it in all three call sites that build tensor_ids:
    #   • start_load_kv  (consumer/europa) — recv_tensor lookup key
    #   • save_kv_layer  (producer/neptune) — send_tensor key
    #   • get_finished   (both sides)       — recv_store cleanup key
    # ---------------------------------------------------------------------------
    local _p2p_connector_file
    _p2p_connector_file=$("${PYTHON_BIN}" -c \
        "import vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector as m; print(m.__file__)" \
        2>/dev/null || echo "")
    if [ -n "${_p2p_connector_file}" ]; then
        echo "  Patching _tensor_key canonical fix: ${_p2p_connector_file}"
        "${PYTHON_BIN}" - "${_p2p_connector_file}" << 'CONNECTOR_PATCH_EOF'
import sys, re

path = sys.argv[1]
with open(path, "r") as f:
    src = f.read()

# Only patch once
if "_tensor_key" in src:
    print(f"  [patch] _tensor_key already applied, skipping")
    sys.exit(0)

# ---- 1. Add _tensor_key static method before start_load_kv ----
tensor_key_method = (
    "    @staticmethod\n"
    "    def _tensor_key(request_id: str) -> str:\n"
    "        \"\"\"Return a stable tensor key from vLLM's internal request_id.\n"
    "        vLLM wraps user-supplied request_id as:\n"
    "            generate-tokens-{user_id}-{random8hex}\n"
    "        The 8-hex suffix is generated independently per-server, causing\n"
    "        producer/consumer tensor_id mismatches.  Strip both components to\n"
    "        recover the original user_id that is identical on all servers.\n"
    "        \"\"\"\n"
    "        import re as _re\n"
    "        s = _re.sub(r'^generate-tokens-', '', request_id)\n"
    "        s = _re.sub(r'-[0-9a-f]{8}$', '', s)\n"
    "        return s\n"
    "\n"
)

# Insert just before start_load_kv definition
anchor = "    def start_load_kv("
if anchor not in src:
    print(f"  [patch] ERROR: could not find start_load_kv anchor in {path}")
    sys.exit(1)
src = src.replace(anchor, tensor_key_method + anchor, 1)

# ---- 2. Fix start_load_kv (consumer recv_tensor lookup key) ----
old2 = 'request.request_id + "#" + layer_name, remote_address'
new2 = 'self._tensor_key(request.request_id) + "#" + layer_name, remote_address'
if old2 not in src:
    print(f"  [patch] ERROR: could not find start_load_kv tensor_id site in {path}")
    sys.exit(1)
src = src.replace(old2, new2, 1)

# ---- 3. Fix save_kv_layer (producer send_tensor key) ----
old3 = 'request_id + "#" + layer_name, kv_cache, remote_address'
new3 = 'self._tensor_key(request_id) + "#" + layer_name, kv_cache, remote_address'
if old3 not in src:
    print(f"  [patch] ERROR: could not find save_kv_layer tensor_id site in {path}")
    sys.exit(1)
src = src.replace(old3, new3, 1)

# ---- 4. Fix get_finished (recv_store cleanup — pass canonical IDs) ----
old4 = "        return self.p2p_nccl_engine.get_finished(finished_req_ids, no_compile_layers)"
new4 = (
    "        canonical_ids = {self._tensor_key(r) for r in finished_req_ids}\n"
    "        return self.p2p_nccl_engine.get_finished(canonical_ids, no_compile_layers)"
)
if old4 not in src:
    print(f"  [patch] ERROR: could not find get_finished call site in {path}")
    sys.exit(1)
src = src.replace(old4, new4, 1)

with open(path, "w") as f:
    f.write(src)
print(f"  [patch] _tensor_key canonical fix applied to {path}")
CONNECTOR_PATCH_EOF
    else
        echo "  WARNING: could not locate p2p_nccl_connector.py — skipping _tensor_key patch"
    fi

    # Phase 3C/3D/4A/4B/4B.1 reuse all connector patches above, then own the 2P2D lifecycle.
    if [ -n "${XPYD_PHASE3C_CONFIG}${XPYD_PHASE3D_CONFIG}${XPYD_PHASE4A_CONFIG}${XPYD_PHASE4B_CONFIG}${XPYD_PHASE4B1_CONFIG}" ]; then
        export NCCL_IB_DISABLE=1
        export NCCL_SOCKET_IFNAME=^lo
        export NCCL_DEBUG=INFO
        local phase3c_status=0
        run_phase3c_substrate || phase3c_status=$?
        if [ "${phase3c_status}" -eq 0 ]; then
            if [ -n "${XPYD_PHASE4B1_CONFIG}" ]; then
                echo "phase4b1_smoke_complete" > "${exp_dir}/status.txt"
            elif [ -n "${XPYD_PHASE4B_CONFIG}" ]; then
                echo "phase4b_${XPYD_PHASE4B_STAGE}_complete" > "${exp_dir}/status.txt"
            elif [ -n "${XPYD_PHASE4A_CONFIG}" ]; then
                echo "phase4a_complete" > "${exp_dir}/status.txt"
            elif [ -n "${XPYD_PHASE3D_CONFIG}" ]; then
                echo "phase3d_${XPYD_PHASE3D_STAGE}_complete" > "${exp_dir}/status.txt"
            else
                echo "phase3c_complete" > "${exp_dir}/status.txt"
            fi
        fi
        return "${phase3c_status}"
    fi

    # NCCL tuning for cross-node Ethernet (10.1.0.x subnet, no InfiniBand).
    # NCCL_IB_DISABLE=1  — skip IB probe (saves ~30s on clusters without IB).
    # NCCL_SOCKET_IFNAME=^lo — exclude only loopback; let NCCL auto-detect the
    #                          actual NIC (eth0/ens*/enp* vary by node).
    #                          Do NOT hardcode eth0,ens3,ens4,ib0 — those names
    #                          don't exist on europa and cause "no socket interface
    #                          found" → ncclCommInitRank crash on the decode side.
    # These are exported here and inherited by all subsequent srun/python calls
    # in this function, including the vLLM server launches.
    export NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME=^lo
    export NCCL_DEBUG=INFO

    # Get node IPs for disagg proxy request_id embedding
    local L40S_IP L4_IP
    L40S_IP=$(get_node_ip "${L40S_NODE}")
    L4_IP=$(get_node_ip "${L4_NODE}")
    echo "  L40S IP (prefill/producer): ${L40S_IP}"
    echo "  L4   IP (decode/consumer):  ${L4_IP}"
    local proxy_prefill_host="${L40S_IP}"
    local proxy_decode_host="${L4_IP}"
    [ "${LOCAL_NODE}" = "${L40S_NODE}" ] && proxy_prefill_host="127.0.0.1"
    [ "${LOCAL_NODE}" = "${L4_NODE}" ] && proxy_decode_host="127.0.0.1"
    echo "  Proxy HTTP -> prefill host: ${proxy_prefill_host}"
    echo "  Proxy HTTP -> decode host:  ${proxy_decode_host}"
    echo "  Proxy advertised prefill addr: ${L40S_IP}"
    echo "  Proxy advertised decode addr:  ${L4_IP}"

    # P2pNcclConnector (V1 engine): kv_ip/kv_rank/kv_parallel_size not used;
    # routing is done via addresses embedded in request_id by the proxy.
    #
    # send_type=PUT (synchronous): producer (neptune) blocks in send_sync() until
    # the KV data has been received by the consumer (europa) before returning the
    # HTTP 200 OK.  The default PUT_ASYNC queues the send in a daemon thread and
    # returns immediately — but that daemon thread may never fire before the job
    # is cancelled, leaving europa waiting in ncclRecv forever.
    local PREFILL_KV_CFG
    PREFILL_KV_CFG=$(cat <<EOF
{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_port":${KV_PORT},"kv_connector_extra_config":{"send_type":"PUT"}}
EOF
)
    local DECODE_KV_CFG
    DECODE_KV_CFG=$(cat <<EOF
{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_port":${KV_PORT}}
EOF
)

    # E is sensitive to both proxy correctness and decode-side capacity.
    # Use E-specific configs/workloads so validation runs don't immediately
    # overdrive a single L4 decode GPU.
    local old_num_prompts="${NUM_PROMPTS}"
    local old_num_warmups="${NUM_WARMUPS}"
    NUM_PROMPTS="${E_NUM_PROMPTS}"
    NUM_WARMUPS="${E_NUM_WARMUPS}"

    for config in "${E_DISAGG_CONFIGS[@]}"; do
        local l40s_freq; l40s_freq=$(echo "${config}" | "${AWK_BIN}" '{print $1}')
        local l4_freq;   l4_freq=$(echo "${config}"   | "${AWK_BIN}" '{print $2}')
        local tp=1
        local mon_prefix="${exp_dir}/monitor_l40sf${l40s_freq}_l4f${l4_freq}"
        echo ""
        echo "  Config: L40S@${l40s_freq}MHz + L4@${l4_freq}MHz"

        # Phase 3A observes the scheduler/runtime default and must not issue
        # clock commands.  The regular Experiment E path remains unchanged.
        if [ -n "${XPYD_PHASE3B_CHARACTERIZATION_CONFIG}" ]; then
            set_characterization_clocks "${L40S_NODE}" P0 "${l40s_freq}" "${L40S_MEM_FREQ}"
            set_characterization_clocks "${L4_NODE}" D0 "${l4_freq}" "${L4_MEM_FREQ}"
        elif [ -z "${XPYD_PHASE3A_CONFIG}${XPYD_PHASE3B_CONFIG}" ]; then
            set_gpu_freq "${L40S_NODE}" "${l40s_freq}" "${tp}" "${L40S_MEM_FREQ}"
            set_gpu_freq "${L4_NODE}"   "${l4_freq}"   "${tp}" "${L4_MEM_FREQ}"
        else
            echo "  XpYd observation path: leaving both GPU clocks untouched"
        fi

        # Start prefill instance (L40S) — P2pNcclConnector (V1 engine native)
        # Note: no single quotes around the JSON — extra_args is unquoted in
        # start_vllm_server so the shell must NOT see literal quote chars here.
        start_vllm_server "${L40S_NODE}" "${PREFILL_PORT}" "${tp}" \
            "--kv-transfer-config ${PREFILL_KV_CFG}" || {
            echo "  ERROR: Prefill server failed to start. Check ${RESULT_DIR}/vllm_${L40S_NODE}_${PREFILL_PORT}.log"
            echo "disagg_not_supported" > "${exp_dir}/status.txt"
            return 0
        }

        # Start decode instance (L4) — "log" enables request logging so we
        # can see whether decode requests reach europa's HTTP server.
        # expandable_segments avoids fragmentation OOM on L4 (22 GiB VRAM);
        # lower gpu-memory-utilization leaves ~3.9 GiB headroom for long batches.
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
        start_vllm_server "${L4_NODE}" "${DECODE_PORT}" "${tp}" \
            "--kv-transfer-config ${DECODE_KV_CFG} --gpu-memory-utilization 0.82" "log" || {
            unset PYTORCH_CUDA_ALLOC_CONF
            echo "  ERROR: Decode server failed to start. Check ${RESULT_DIR}/vllm_${L4_NODE}_${DECODE_PORT}.log"
            stop_server "${L40S_NODE}" "${PREFILL_PORT}"
            return 1
        }
        unset PYTORCH_CUDA_ALLOC_CONF

        # Start the checked-in disagg proxy. It completes the one-token,
        # non-streaming P request before forwarding the real D SSE stream.
        local PROXY_PID
        local proxy_diagnostics_log="${RESULT_DIR}/xpyd_proxy_${SLURM_JOB_ID:-local}.jsonl"
        export XPYD_PROXY_DIAGNOSTICS_LOG="${proxy_diagnostics_log}"
        PYTHONPATH="${SCRIPT_DIR}/paper/scripts${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON_BIN}" -m "${DISAGG_PROXY_MODULE}" \
            --prefill-http-host "${proxy_prefill_host}" --prefill-port "${PREFILL_PORT}" \
            --decode-http-host  "${proxy_decode_host}"  --decode-port  "${DECODE_PORT}" \
            --prefill-addr-host "${L40S_IP}" \
            --decode-addr-host  "${L4_IP}" \
            --kv-port "${KV_PORT}" \
            --proxy-port "${PROXY_PORT}" \
            --diagnostics-log "${proxy_diagnostics_log}" \
            --model "${MODEL}" &
        PROXY_PID=$!
        echo "  Disagg proxy PID=${PROXY_PID}, waiting for readiness..."
        local elapsed=0
        while ! "${CURL_BIN}" -s --connect-timeout 2 "http://127.0.0.1:${PROXY_PORT}/health" > /dev/null 2>&1; do
            sleep 2; elapsed=$((elapsed+2))
            if [ ${elapsed} -ge 60 ]; then
                echo "  ERROR: Proxy failed to start within 60s"
                kill "${PROXY_PID}" 2>/dev/null || true
                stop_server "${L40S_NODE}" "${PREFILL_PORT}"
                stop_server "${L4_NODE}" "${DECODE_PORT}"
                return 1
            fi
        done
        echo "  Proxy ready (${elapsed}s)"

        if [ -n "${XPYD_PHASE3A_CONFIG}${XPYD_PHASE3B_CONFIG}${XPYD_PHASE3B_CHARACTERIZATION_CONFIG}" ]; then
            export XPYD_P_HTTP_URI="http://${proxy_prefill_host}:${PREFILL_PORT}"
            export XPYD_D_HTTP_URI="http://${proxy_decode_host}:${DECODE_PORT}"
            export XPYD_PROXY_URI="http://127.0.0.1:${PROXY_PORT}"
            export XPYD_P_SERVER_LOG="${RESULT_DIR}/vllm_${L40S_NODE}_${PREFILL_PORT}.log"
            export XPYD_D_SERVER_LOG="${RESULT_DIR}/vllm_${L4_NODE}_${DECODE_PORT}.log"
            export PYTHONPATH="${SCRIPT_DIR}/paper/scripts${PYTHONPATH:+:${PYTHONPATH}}"
        fi

        if [ -n "${XPYD_PHASE3B_CHARACTERIZATION_CONFIG}" ]; then
            echo "  Phase 3B: running controlled fixed-clock workload characterization"
            local characterization_status=0
            local characterization_args=(
                -m xpyd.phase3b_characterization run
                --config "${XPYD_PHASE3B_CHARACTERIZATION_CONFIG}"
            )
            if [ -n "${XPYD_PHASE3B_CHARACTERIZATION_RUN_ID}" ]; then
                characterization_args+=(--run-id "${XPYD_PHASE3B_CHARACTERIZATION_RUN_ID}")
            fi
            if [ -n "${XPYD_PHASE3B_CHARACTERIZATION_REPEATS}" ]; then
                characterization_args+=(--repeats "${XPYD_PHASE3B_CHARACTERIZATION_REPEATS}")
            fi
            if [ -n "${XPYD_PHASE3B_CHARACTERIZATION_REQUESTS}" ]; then
                characterization_args+=(--requests-per-repeat "${XPYD_PHASE3B_CHARACTERIZATION_REQUESTS}")
            fi
            "${PYTHON_BIN}" "${characterization_args[@]}" || characterization_status=$?
            kill "${PROXY_PID}" 2>/dev/null || true
            wait "${PROXY_PID}" 2>/dev/null || true
            stop_server "${L40S_NODE}" "${PREFILL_PORT}"
            stop_server "${L4_NODE}" "${DECODE_PORT}"
            if [ "${characterization_status}" -ne 0 ]; then
                echo "  ERROR: Phase 3B characterization failed with status ${characterization_status}"
                return "${characterization_status}"
            fi
            break
        fi

        if [ -n "${XPYD_PHASE3B_CONFIG}" ]; then
            echo "  Phase 3B: running isolated read-only energy preflight"
            local phase3b_status=0
            local phase3b_args=(
                -m xpyd.phase3b_energy preflight
                --config "${XPYD_PHASE3B_CONFIG}"
            )
            if [ -n "${XPYD_PHASE3B_RUN_ID}" ]; then
                phase3b_args+=(--run-id "${XPYD_PHASE3B_RUN_ID}")
            fi
            "${PYTHON_BIN}" "${phase3b_args[@]}" || phase3b_status=$?
            kill "${PROXY_PID}" 2>/dev/null || true
            wait "${PROXY_PID}" 2>/dev/null || true
            stop_server "${L40S_NODE}" "${PREFILL_PORT}"
            stop_server "${L4_NODE}" "${DECODE_PORT}"
            if [ "${phase3b_status}" -ne 0 ]; then
                echo "  ERROR: Phase 3B preflight failed with status ${phase3b_status}"
                return "${phase3b_status}"
            fi
            break
        fi

        if [ -n "${XPYD_PHASE3A_CONFIG}" ]; then
            echo "  Phase 3A: running read-only observability harness"
            local phase3a_status=0
            local phase3a_args=(
                -m xpyd.phase3a_observability run
                --config "${XPYD_PHASE3A_CONFIG}"
                --mode "${XPYD_PHASE3A_MODE}"
            )
            if [ -n "${XPYD_PHASE3A_SEMANTIC_PROBE_ID}" ]; then
                phase3a_args+=(--semantic-probe-id "${XPYD_PHASE3A_SEMANTIC_PROBE_ID}")
            fi
            if [ -n "${XPYD_PHASE3A_LOAD_PROBE_ID:-}" ]; then
                phase3a_args+=(--load-probe-id "${XPYD_PHASE3A_LOAD_PROBE_ID}")
            fi
            "${PYTHON_BIN}" "${phase3a_args[@]}" || phase3a_status=$?
            kill "${PROXY_PID}" 2>/dev/null || true
            wait "${PROXY_PID}" 2>/dev/null || true
            stop_server "${L40S_NODE}" "${PREFILL_PORT}"
            stop_server "${L4_NODE}" "${DECODE_PORT}"
            if [ "${phase3a_status}" -ne 0 ]; then
                echo "  ERROR: Phase 3A harness failed with status ${phase3a_status}"
                return "${phase3a_status}"
            fi
            break
        fi

        # Per-workload: monitors on BOTH nodes + timestamps + energy
        for wl in "${E_WORKLOADS[@]}"; do
            read -r il ol rate conc <<< "${wl}"
            local out="${exp_dir}/disagg_l40sf${l40s_freq}_l4f${l4_freq}_il${il}_ol${ol}_r${rate}.txt"
            local wl_prefix="${mon_prefix}_il${il}_ol${ol}_r${rate}"
            [ -f "${out}" ] && { echo "    SKIP (exists): ${out##*/}"; continue; }
            echo "    il=${il} ol=${ol} rate=${rate}"

            # Start per-GPU monitors on both nodes.
            # _bg_start_mon adds --gpus-per-node so SLURM cgroup lets pynvml
            # access GPU device files on the remote L4 node.
            local L40S_MPIDS=()
            for ((g=0; g<tp; g++)); do
                _bg_start_mon "${L40S_NODE}" L40S_MPIDS "${tp}" \
                    "${PYTHON_BIN}" "${GPU_MONITOR_SCRIPT}" \
                        --monitor --interval ${MONITOR_INTERVAL} \
                        --output "${wl_prefix}_l40s_gpu${g}.csv" \
                        --gpu-id ${g}
            done

            local L4_MPIDS=()
            for ((g=0; g<tp; g++)); do
                _bg_start_mon "${L4_NODE}" L4_MPIDS "${tp}" \
                    "${PYTHON_BIN}" "${GPU_MONITOR_SCRIPT}" \
                        --monitor --interval ${MONITOR_INTERVAL} \
                        --output "${wl_prefix}_l4_gpu${g}.csv" \
                        --gpu-id ${g}
            done

            sleep 2  # allow monitors to initialize

            local t_start
            local t_end
            if [ "${WORKLOAD_MODE}" = "trace" ]; then
                run_bench_or_trace "http://127.0.0.1:${PROXY_PORT}" \
                    "${il}" "${ol}" "${rate}" "${conc}" "${out}"
                read -r t_start t_end < <(load_trace_window "${out}")
            else
                t_start=$("${PYTHON_BIN}" -c "import time; print(f'{time.time():.6f}')")
                # Route through disagg proxy (embeds P/D addresses in request_id)
                run_bench_or_trace "http://127.0.0.1:${PROXY_PORT}" \
                    "${il}" "${ol}" "${rate}" "${conc}" "${out}" "--no-stream"
                t_end=$("${PYTHON_BIN}" -c "import time; print(f'{time.time():.6f}')")
            fi

            # Stop all monitors gracefully (SIGTERM → save CSV, then SIGKILL)
            _bg_kill TERM "${L40S_MPIDS[@]}" "${L4_MPIDS[@]}"
            sleep 2
            _bg_kill KILL "${L40S_MPIDS[@]}" "${L4_MPIDS[@]}"
            "${SYNC_BIN}"; sleep 0.5

            # Compute per-node power/energy using hardware counter delta
            local l40s_pwr; l40s_pwr=$(compute_power_energy \
                "${wl_prefix}_l40s" "${tp}" "${t_start}" "${t_end}")
            local l4_pwr; l4_pwr=$(compute_power_energy \
                "${wl_prefix}_l4" "${tp}" "${t_start}" "${t_end}")

            # Sum total across both nodes
            local total_power; total_power=$("${AWK_BIN}" "BEGIN {printf \"%.2f\", \
                $(echo "${l40s_pwr}" | cut -d, -f1)+$(echo "${l4_pwr}" | cut -d, -f1)}")
            local total_energy; total_energy=$("${AWK_BIN}" "BEGIN {printf \"%.3f\", \
                $(echo "${l40s_pwr}" | cut -d, -f2)+$(echo "${l4_pwr}" | cut -d, -f2)}")
            echo "      L40S power/energy: ${l40s_pwr}  |  L4 power/energy: ${l4_pwr}"
            echo "      Total: power=${total_power}W, energy=${total_energy}J"
        done

        # Stop proxy before tearing down the servers it routes to
        kill "${PROXY_PID}" 2>/dev/null || true
        wait "${PROXY_PID}" 2>/dev/null || true

        stop_server "${L40S_NODE}" "${PREFILL_PORT}"
        stop_server "${L4_NODE}" "${DECODE_PORT}"
    done

    NUM_PROMPTS="${old_num_prompts}"
    NUM_WARMUPS="${old_num_warmups}"

    echo "disagg_complete" > "${exp_dir}/status.txt"
}

# ===========================================================================
# Main
# ===========================================================================
echo "B5 End-to-End Disaggregated Serving Benchmark"
echo "Results directory: ${RESULT_DIR}"
echo "Selected experiment set: ${EXP_RAW}"
echo "Active nodes: ${ACTIVE_NODES[*]}"
echo ""

assert_active_nodes_allocated

# ---------------------------------------------------------------------------
# Pre-flight: kill any orphaned vLLM processes from previous jobs.
# Processes started directly (not via srun) escape SLURM job cgroups and
# can survive job cancellation, holding GPU memory and blocking new servers.
# ---------------------------------------------------------------------------
echo "Pre-flight: killing any orphaned vLLM processes..."
for _node in "${ACTIVE_NODES[@]}"; do
    if [ "${_node}" = "${LOCAL_NODE}" ]; then
        "${PKILL_BIN}" -KILL -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
        "${PKILL_BIN}" -KILL -f 'python.*vllm' 2>/dev/null || true
    else
        srun --overlap --nodes=1 --ntasks=1 --nodelist="${_node}" \
            "${PKILL_BIN}" -KILL -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
        srun --overlap --nodes=1 --ntasks=1 --nodelist="${_node}" \
            "${PKILL_BIN}" -KILL -f 'python.*vllm' 2>/dev/null || true
    fi
done
sleep 3

# Phase 3A/3B must not issue any nvidia-smi mutation, including persistence mode.
if [ "${XPYD_NO_GPU_MUTATION}" -eq 0 ]; then
    for _node in "${ACTIVE_NODES[@]}"; do
        if [ "${_node}" = "${LOCAL_NODE}" ]; then
            "${SUDO_BIN}" "${NVIDIA_SMI_BIN}" -pm 1
        else
            srun --overlap --nodes=1 --ntasks=1 --nodelist="${_node}" \
                "${SUDO_BIN}" "${NVIDIA_SMI_BIN}" -pm 1
        fi
    done
    echo "GPU persistence mode enabled on active nodes."
else
    if [ "${XPYD_CHARACTERIZATION_CLOCKS_REQUESTED}" -eq 1 ]; then
        echo "XpYd characterization: legacy persistence/reset mutations disabled; one fixed-clock lock will be applied."
    else
        echo "XpYd observation path: GPU persistence/clock/reset mutations disabled."
    fi
fi
echo ""

# Run experiments
# v2 dense calibration: re-run A, B, C/D with expanded freq+rate grid.
# Existing results are auto-skipped (safe resume via [ -f "${out}" ] check).
# Only new configs (new freqs, new rates) will be profiled.
[ "${RUN_A}" -eq 1 ] && run_experiment_A
[ "${RUN_B}" -eq 1 ] && run_experiment_B
[ "${RUN_CD}" -eq 1 ] && run_experiment_CD
[ "${RUN_E}" -eq 1 ] && run_experiment_E

echo ""
echo "========================================"
echo "All experiments complete."
echo "Results in: ${RESULT_DIR}"
echo ""
echo "Next step: run parse_disagg_results.py to aggregate metrics"
echo "========================================"
