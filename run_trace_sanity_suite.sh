#!/bin/bash
#SBATCH --job-name=trace_sanity_suite
#SBATCH --partition=long
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --nodelist=neptune,europa
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=23:59:59
#SBATCH --output=logs/trace_sanity_suite_%j.log

set -u

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"
mkdir -p logs results

# Shared defaults for the trace sanity suite.
TRACE_DIR="${TRACE_DIR:-${SCRIPT_DIR}/synthetic_traces}"
TRACE_NAMES="${TRACE_NAMES:-T1 T2 T3 T4}"
EXP_SET="${EXP_SET:-A,B,CD}"
TRACE_REPLAY_SCRIPT="${TRACE_REPLAY_SCRIPT:-${SCRIPT_DIR}/replay_synthetic_trace.py}"
L40S_CONFIGS="${L40S_CONFIGS:-2520:1}"
L4_CONFIGS="${L4_CONFIGS:-2040:1}"
CD_PREFILL_FREQS="${CD_PREFILL_FREQS:-2520}"
CD_DECODE_FREQS="${CD_DECODE_FREQS:-2040}"
CD_PREFILL_TPS="${CD_PREFILL_TPS:-1}"
CD_DECODE_TPS="${CD_DECODE_TPS:-1}"
TRACE_SINGLETON_WORKLOAD="${TRACE_SINGLETON_WORKLOAD:-1 1 1 32}"
TRACE_PREFILL_MAX_CONCURRENCY="${TRACE_PREFILL_MAX_CONCURRENCY:-32}"
TRACE_DECODE_MAX_CONCURRENCY="${TRACE_DECODE_MAX_CONCURRENCY:-32}"
NUM_WARMUPS="${NUM_WARMUPS:-0}"
VLLM_STARTUP_TIMEOUT_S="${VLLM_STARTUP_TIMEOUT_S:-1800}"

FAILURES=0

run_one_trace() {
    local trace_name="$1"
    local trace_csv="${TRACE_DIR}/${trace_name}.csv"
    local result_dir="results/trace_sanity_${trace_name}"

    if [ ! -f "${trace_csv}" ]; then
        echo "ERROR: missing trace CSV: ${trace_csv}"
        return 1
    fi
    if [ ! -f "${TRACE_REPLAY_SCRIPT}" ]; then
        echo "ERROR: missing replay script: ${TRACE_REPLAY_SCRIPT}"
        return 1
    fi

    echo ""
    echo "============================================================"
    echo "Trace sanity: ${trace_name}"
    echo "Trace CSV: ${trace_csv}"
    echo "Replay script: ${TRACE_REPLAY_SCRIPT}"
    echo "Result dir: ${result_dir}"
    echo "Experiments: ${EXP_SET}"
    echo "============================================================"

    EXP="${EXP_SET}" \
    WORKLOAD_MODE=trace \
    TRACE_CSV="${trace_csv}" \
    TRACE_REPLAY_SCRIPT="${TRACE_REPLAY_SCRIPT}" \
    RESULT_DIR="${result_dir}" \
    L40S_CONFIGS="${L40S_CONFIGS}" \
    L4_CONFIGS="${L4_CONFIGS}" \
    CD_PREFILL_FREQS="${CD_PREFILL_FREQS}" \
    CD_DECODE_FREQS="${CD_DECODE_FREQS}" \
    CD_PREFILL_TPS="${CD_PREFILL_TPS}" \
    CD_DECODE_TPS="${CD_DECODE_TPS}" \
    TRACE_SINGLETON_WORKLOAD="${TRACE_SINGLETON_WORKLOAD}" \
    TRACE_PREFILL_MAX_CONCURRENCY="${TRACE_PREFILL_MAX_CONCURRENCY}" \
    TRACE_DECODE_MAX_CONCURRENCY="${TRACE_DECODE_MAX_CONCURRENCY}" \
    NUM_WARMUPS="${NUM_WARMUPS}" \
    VLLM_STARTUP_TIMEOUT_S="${VLLM_STARTUP_TIMEOUT_S}" \
    bash "${SCRIPT_DIR}/run_disagg_benchmark.sh"
}

for trace_name in ${TRACE_NAMES}; do
    if ! run_one_trace "${trace_name}"; then
        echo "WARNING: trace ${trace_name} failed"
        FAILURES=$((FAILURES + 1))
    fi
done

echo ""
echo "============================================================"
if [ "${FAILURES}" -eq 0 ]; then
    echo "Trace sanity suite completed successfully."
else
    echo "Trace sanity suite completed with ${FAILURES} failed trace(s)."
fi
echo "============================================================"

exit "${FAILURES}"
