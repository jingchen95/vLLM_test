#!/usr/bin/env python3
"""
Plan a minimal same-family multi-size pilot for SWEEP-LLM.

The pilot intentionally reuses the phase-only C/D collection path and keeps the
matrix compact while still covering the main scheduler axes:
  - GPU type: L40S, L4
  - phase: prefill-only, decode-only
  - TP: low / medium / high
  - frequency: low / mid / high representative points
  - lengths: short / medium / long
  - request rate: low / medium / high

Outputs:
  - <prefix>_matrix.csv
  - <prefix>_summary.json
  - <prefix>_collect.sh
  - <prefix>_submit_sbatch.sh
  - <prefix>_load_sanity.sh
  - <prefix>_load_sanity_sbatch.sh
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List

from paths import analysis_prefix


MODEL_GPU_TP_ALLOWLIST = {
    # Same-family larger-size pilot: L4 TP=1 is not a valid deployment point
    # for smoke sanity on this model/hardware combination.
    "mistral_nemo_12b_pilot": {
        "l40s": [1, 2, 4],
        "l4": [2, 4],
    },
}


def parse_csv_ints(raw: str) -> List[int]:
    return [int(tok.strip()) for tok in raw.split(",") if tok.strip()]


def shell_join(values: Iterable[int]) -> str:
    return " ".join(str(v) for v in values)


def filter_allowed(values: list[int], allowed: list[int]) -> list[int]:
    return [v for v in values if v in set(allowed)]


def apply_model_tp_constraints(args: argparse.Namespace) -> None:
    allow = MODEL_GPU_TP_ALLOWLIST.get(args.model_id)
    if not allow:
        return
    for gpu_type, phase_attrs in (
        ("l40s", ("prefill_tp_l40s", "decode_tp_l40s")),
        ("l4", ("prefill_tp_l4", "decode_tp_l4")),
    ):
        allowed = allow[gpu_type]
        for attr in phase_attrs:
            setattr(args, attr, filter_allowed(list(getattr(args, attr)), allowed))


def apply_legacy_overrides(args: argparse.Namespace) -> None:
    # Keep older one-list knobs as a compatibility path for ad hoc pilot edits.
    if args.freq_l40s is not None:
        args.prefill_freq_l40s = list(args.freq_l40s)
        args.decode_freq_l40s = list(args.freq_l40s)
    if args.freq_l4 is not None:
        args.prefill_freq_l4 = list(args.freq_l4)
        args.decode_freq_l4 = list(args.freq_l4)
    if args.tp_l40s is not None:
        args.prefill_tp_l40s = list(args.tp_l40s)
        args.decode_tp_l40s = list(args.tp_l40s)
    if args.tp_l4 is not None:
        args.prefill_tp_l4 = list(args.tp_l4)
        args.decode_tp_l4 = list(args.tp_l4)
    if args.rates is not None:
        args.prefill_rates = list(args.rates)
        args.decode_rates = list(args.rates)


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    # Tiny end-to-end sanity run: just enough to validate collection, parsing,
    # and Mode A/B/C evaluation without paying for a real pilot.
    args.prefill_freq_l40s = [2520]
    args.prefill_freq_l4 = [2040]
    args.prefill_tp_l40s = [1]
    args.prefill_tp_l4 = [2, 4]
    args.prefill_ils = [1024]
    args.prefill_rates = [1]

    args.decode_freq_l40s = [1500, 2520]
    args.decode_freq_l4 = [1410, 2040]
    args.decode_tp_l40s = [1]
    args.decode_tp_l4 = [2, 4]
    args.decode_ils = [1024]
    args.decode_ols = [256, 1024]
    args.decode_rates = [1]


def build_rows(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []

    for gpu_type in ("l40s", "l4"):
        prefill_tp_values = args.prefill_tp_l40s if gpu_type == "l40s" else args.prefill_tp_l4
        prefill_freq_values = args.prefill_freq_l40s if gpu_type == "l40s" else args.prefill_freq_l4
        decode_tp_values = args.decode_tp_l40s if gpu_type == "l40s" else args.decode_tp_l4
        decode_freq_values = args.decode_freq_l40s if gpu_type == "l40s" else args.decode_freq_l4

        for tp in prefill_tp_values:
            for freq in prefill_freq_values:
                for il in args.prefill_ils:
                    for rate in args.prefill_rates:
                        rows.append(
                            {
                                "phase": "prefill",
                                "gpu_type": gpu_type,
                                "tp": tp,
                                "freq_mhz": freq,
                                "input_len": il,
                                "output_len": 1,
                                "request_rate": rate,
                            }
                        )
        for tp in decode_tp_values:
            for freq in decode_freq_values:
                for decode_il in args.decode_ils:
                    for ol in args.decode_ols:
                        for rate in args.decode_rates:
                            rows.append(
                                {
                                    "phase": "decode",
                                    "gpu_type": gpu_type,
                                    "tp": tp,
                                    "freq_mhz": freq,
                                    "input_len": decode_il,
                                    "output_len": ol,
                                    "request_rate": rate,
                                }
                            )
    return rows


def write_matrix(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "phase",
        "gpu_type",
        "tp",
        "freq_mhz",
        "input_len",
        "output_len",
        "request_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_collect_script(args: argparse.Namespace, prefix: Path) -> str:
    model_slug = args.model_id.replace("-", "_")
    result_dir = Path(args.result_dir)
    summary_script = "summarize_same_family_multisize_pilot.py"
    output_prefix_name = prefix.name
    env_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"RESULT_DIR='{result_dir.as_posix()}'",
        'if [ -d "${RESULT_DIR}/CD_prefill_decode_only" ] && [ -n "$(ls -A "${RESULT_DIR}/CD_prefill_decode_only" 2>/dev/null)" ]; then',
        '  echo "Refusing to reuse non-empty result directory: ${RESULT_DIR}/CD_prefill_decode_only" >&2',
        "  exit 1",
        "fi",
        'mkdir -p "${RESULT_DIR}"',
        "",
        "# Pass 1: prefill on L40S, decode on L4",
        " ".join(
            [
                f"MODEL='{args.hf_name}'",
                'RESULT_DIR="${RESULT_DIR}"',
                "EXP=CD",
                "CD_EXTENDED_NAMES=1",
                f"NUM_PROMPTS={args.num_prompts}",
                f"CD_PREFILL_LABEL='l40s_{model_slug}'",
                f"CD_DECODE_LABEL='l4_{model_slug}'",
                f"CD_PREFILL_FREQS='{shell_join(args.prefill_freq_l40s)}'",
                f"CD_DECODE_FREQS='{shell_join(args.decode_freq_l4)}'",
                f"CD_PREFILL_TPS='{shell_join(args.prefill_tp_l40s)}'",
                f"CD_DECODE_TPS='{shell_join(args.decode_tp_l4)}'",
                f"CD_PREFILL_ILS='{shell_join(args.prefill_ils)}'",
                f"CD_DECODE_ILS='{shell_join(args.decode_ils)}'",
                f"CD_DECODE_OLS='{shell_join(args.decode_ols)}'",
                f"CD_PREFILL_RATES='{shell_join(args.prefill_rates)}'",
                f"CD_DECODE_RATES='{shell_join(args.decode_rates)}'",
                "bash run_disagg_benchmark.sh",
            ]
        ),
        "",
        "# Pass 2: prefill on L4, decode on L40S",
        " ".join(
            [
                f"MODEL='{args.hf_name}'",
                'RESULT_DIR="${RESULT_DIR}"',
                "EXP=CD",
                "CD_EXTENDED_NAMES=1",
                f"NUM_PROMPTS={args.num_prompts}",
                f"CD_PREFILL_NODE='{args.l4_node}'",
                f"CD_DECODE_NODE='{args.l40s_node}'",
                f"CD_PREFILL_LABEL='l4_{model_slug}'",
                f"CD_DECODE_LABEL='l40s_{model_slug}'",
                f"CD_PREFILL_MEM_FREQ={args.l4_mem_freq}",
                f"CD_DECODE_MEM_FREQ={args.l40s_mem_freq}",
                f"CD_PREFILL_FREQS='{shell_join(args.prefill_freq_l4)}'",
                f"CD_DECODE_FREQS='{shell_join(args.decode_freq_l40s)}'",
                f"CD_PREFILL_TPS='{shell_join(args.prefill_tp_l4)}'",
                f"CD_DECODE_TPS='{shell_join(args.decode_tp_l40s)}'",
                f"CD_PREFILL_ILS='{shell_join(args.prefill_ils)}'",
                f"CD_DECODE_ILS='{shell_join(args.decode_ils)}'",
                f"CD_DECODE_OLS='{shell_join(args.decode_ols)}'",
                f"CD_PREFILL_RATES='{shell_join(args.prefill_rates)}'",
                f"CD_DECODE_RATES='{shell_join(args.decode_rates)}'",
                "bash run_disagg_benchmark.sh",
            ]
        ),
        "",
        "# Summarize phase-only pilot results into a model-ready CSV",
        " ".join(
            [
                f"python3 {summary_script}",
                f"--results-dir '{(result_dir / 'CD_prefill_decode_only').as_posix()}'",
                f"--model-id '{args.model_id}'",
                f"--hf-name '{args.hf_name}'",
                f"--family '{args.family}'",
                f"--param-count-b {args.size_b}",
                f"--output-prefix '{output_prefix_name}'",
            ]
        ),
    ]
    return "\n".join(env_lines) + "\n"


def build_submit_script(prefix: Path) -> str:
    collect_script = prefix.with_name(prefix.name + "_collect.sh")
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"sbatch --wait --job-name=sfms_pilot_p1 --export=ALL {collect_script.as_posix()}",
            "",
        ]
    )


def build_collect_sbatch_script(prefix: Path) -> str:
    collect_script = prefix.with_name(prefix.name + "_collect.sh")
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "#SBATCH --job-name=sfms_pilot",
            "#SBATCH --partition=long",
            "#SBATCH --nodes=2",
            "#SBATCH --ntasks=16",
            "#SBATCH --ntasks-per-node=8",
            "#SBATCH --nodelist=neptune,europa",
            "#SBATCH --exclusive",
            "#SBATCH --mem=0",
            "#SBATCH --time=23:59:59",
            "#SBATCH --output=logs/same_family_multisize_pilot_%j.log",
            "",
            "set -euo pipefail",
            'SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"',
            'cd "${SCRIPT_DIR}"',
            f"bash {collect_script.name}",
            "",
        ]
    )


def build_load_sanity_script(args: argparse.Namespace, prefix: Path) -> str:
    load_results_path = prefix.with_name(prefix.name + "_load_sanity_results.csv").name
    load_logs_dir = Path(args.result_dir).as_posix()
    l40s_tp_values = sorted(set(args.prefill_tp_l40s + args.decode_tp_l40s))
    l4_tp_values = sorted(set(args.prefill_tp_l4 + args.decode_tp_l4))
    l40s_tp_shell = shell_join(l40s_tp_values)
    l4_tp_shell = shell_join(l4_tp_values)
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"MODEL='{args.hf_name}'",
            f"L40S_NODE='{args.l40s_node}'",
            f"L4_NODE='{args.l4_node}'",
            f"L40S_MEM_FREQ={args.l40s_mem_freq}",
            f"L4_MEM_FREQ={args.l4_mem_freq}",
            "PYTHON_BIN=\"${PYTHON_BIN:-$(command -v python3 || command -v python)}\"",
            "CURL_BIN=\"${CURL_BIN:-/usr/bin/curl}\"",
            "SEQ_BIN=\"${SEQ_BIN:-/usr/bin/seq}\"",
            "SUDO_BIN=\"${SUDO_BIN:-/usr/bin/sudo}\"",
            "NVIDIA_SMI_BIN=\"${NVIDIA_SMI_BIN:-/usr/bin/nvidia-smi}\"",
            f"RESULT_DIR='{load_logs_dir}'",
            f"RESULTS_CSV='{load_results_path}'",
            f"L40S_TP_VALUES='{l40s_tp_shell}'",
            f"L4_TP_VALUES='{l4_tp_shell}'",
            "",
            "mkdir -p \"${RESULT_DIR}\" logs",
            "echo 'gpu_type,tp,status,log_path' > \"${RESULTS_CSV}\"",
            "",
            "set_gpu_freq() {",
            "  local gpu_type=$1",
            "  local node=$2",
            "  local mem_freq=$3",
            "  local tp=$4",
            "  local cuda_devs",
            "  if [ \"$tp\" -eq 2 ]; then cuda_devs='0,1'; else cuda_devs=$(${SEQ_BIN} -s, 0 $((tp - 1))); fi",
            "  srun --overlap --nodes=1 --ntasks=1 --nodelist=\"${node}\" \\",
            "    ${SUDO_BIN} ${NVIDIA_SMI_BIN} -pm 1 >/dev/null 2>&1 || true",
            "  local sm_freq=1410",
            "  if [ \"${gpu_type}\" = 'l40s' ]; then sm_freq=2520; fi",
            "  srun --overlap --nodes=1 --ntasks=1 --nodelist=\"${node}\" \\",
            "    ${SUDO_BIN} ${NVIDIA_SMI_BIN} -i ${cuda_devs} -lgc ${sm_freq},${sm_freq} >/dev/null 2>&1 || true",
            "  srun --overlap --nodes=1 --ntasks=1 --nodelist=\"${node}\" \\",
            "    ${SUDO_BIN} ${NVIDIA_SMI_BIN} -i ${cuda_devs} -lmc ${mem_freq},${mem_freq} >/dev/null 2>&1 || true",
            "}",
            "",
            "run_one() {",
            "  local gpu_type=$1",
            "  local node=$2",
            "  local mem_freq=$3",
            "  local tp=$4",
            "  local port=$5",
            "  local log_file=\"${RESULT_DIR}/vllm_load_sanity_${node}_tp${tp}.log\"",
            "  local health_host=\"${node}\"",
            "  local cuda_devs",
            "  if [ \"$tp\" -eq 2 ]; then cuda_devs='0,1'; else cuda_devs=$(${SEQ_BIN} -s, 0 $((tp - 1))); fi",
            "  set_gpu_freq \"${gpu_type}\" \"${node}\" \"${mem_freq}\" \"$tp\"",
            "  srun --overlap --nodes=1 --ntasks=1 --nodelist=\"${node}\" --gpus-per-node=\"${tp}\" \\",
            "    --gpu-bind=\"map_gpu:${cuda_devs}\" \\",
            "    ${PYTHON_BIN} -m vllm.entrypoints.openai.api_server \\",
            "      --model \"${MODEL}\" \\",
            "      --port \"${port}\" \\",
            "      --tensor-parallel-size \"${tp}\" \\",
            "      --max-model-len 4096 \\",
            "      --disable-log-requests > \"${log_file}\" 2>&1 &",
            "  local server_pid=$!",
            "  local status='FAIL'",
            "  for _ in $(seq 1 120); do",
            "    if ${CURL_BIN} -s --connect-timeout 3 \"http://${health_host}:${port}/health\" >/dev/null 2>&1; then",
            "      status='PASS'",
            "      break",
            "    fi",
            "    if ! kill -0 \"${server_pid}\" 2>/dev/null; then",
            "      break",
            "    fi",
            "    sleep 5",
            "  done",
            "  echo \"${gpu_type},${tp},${status},${log_file}\" | tee -a \"${RESULTS_CSV}\"",
            "  if kill -0 \"${server_pid}\" 2>/dev/null; then",
            "    kill \"${server_pid}\" 2>/dev/null || true",
            "    wait \"${server_pid}\" 2>/dev/null || true",
            "  fi",
            "  srun --overlap --nodes=1 --ntasks=1 --nodelist=\"${node}\" pkill -TERM -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true",
            "  sleep 10",
            "  srun --overlap --nodes=1 --ntasks=1 --nodelist=\"${node}\" pkill -KILL -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true",
            "}",
            "",
            "port=8100",
            "for tp in ${L40S_TP_VALUES}; do",
            "  run_one 'l40s' \"${L40S_NODE}\" \"${L40S_MEM_FREQ}\" \"${tp}\" \"${port}\"",
            "  port=$((port + 1))",
            "done",
            "",
            "port=8200",
            "for tp in ${L4_TP_VALUES}; do",
            "  run_one 'l4' \"${L4_NODE}\" \"${L4_MEM_FREQ}\" \"${tp}\" \"${port}\"",
            "  port=$((port + 1))",
            "done",
            "",
            "cat \"${RESULTS_CSV}\"",
            "if grep -q ',FAIL,' \"${RESULTS_CSV}\"; then",
            "  echo 'Load sanity failed; refusing to continue.' >&2",
            "  exit 1",
            "fi",
            "",
        ]
    )


def build_load_sanity_sbatch_script(prefix: Path) -> str:
    sanity_script = prefix.with_name(prefix.name + "_load_sanity.sh")
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "#SBATCH --job-name=sfms_load_sanity",
            "#SBATCH --partition=long",
            "#SBATCH --nodes=2",
            "#SBATCH --ntasks=16",
            "#SBATCH --ntasks-per-node=8",
            "#SBATCH --nodelist=neptune,europa",
            "#SBATCH --exclusive",
            "#SBATCH --mem=0",
            "#SBATCH --time=02:00:00",
            "#SBATCH --output=logs/same_family_multisize_load_sanity_%j.log",
            "",
            "set -euo pipefail",
            'SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"',
            'cd "${SCRIPT_DIR}"',
            f"bash {sanity_script.name}",
            "",
        ]
    )


def build_pipeline_script(prefix: Path) -> str:
    sanity_script = prefix.with_name(prefix.name + "_load_sanity.sh")
    collect_script = prefix.with_name(prefix.name + "_collect.sh")
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"bash {sanity_script.name}",
            f"bash {collect_script.name}",
            "",
        ]
    )


def build_pipeline_sbatch_script(prefix: Path) -> str:
    pipeline_script = prefix.with_name(prefix.name + "_pipeline.sh")
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "#SBATCH --job-name=sfms_pipeline",
            "#SBATCH --partition=long",
            "#SBATCH --nodes=2",
            "#SBATCH --ntasks=16",
            "#SBATCH --ntasks-per-node=8",
            "#SBATCH --nodelist=neptune,europa",
            "#SBATCH --exclusive",
            "#SBATCH --mem=0",
            "#SBATCH --time=23:59:59",
            "#SBATCH --output=logs/same_family_multisize_pipeline_%j.log",
            "",
            "set -euo pipefail",
            'SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"',
            'cd "${SCRIPT_DIR}"',
            f"bash {pipeline_script.name}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="mistral_nemo_12b_pilot")
    parser.add_argument("--hf-name", default="mistralai/Mistral-Nemo-Instruct-2407")
    parser.add_argument("--family", default="mistral")
    parser.add_argument("--size-b", type=float, default=12.0)
    parser.add_argument("--result-dir", default="results/same_family_multisize_mistral_nemo12b_pilot")
    parser.add_argument("--output-prefix", default=str(analysis_prefix("same_family_multisize_mistral_nemo12b_pilot")))
    parser.add_argument("--prefill-freq-l40s", type=parse_csv_ints, default=parse_csv_ints("735,2520"))
    parser.add_argument("--prefill-freq-l4", type=parse_csv_ints, default=parse_csv_ints("780,2040"))
    parser.add_argument("--decode-freq-l40s", type=parse_csv_ints, default=parse_csv_ints("735,1500,2520"))
    parser.add_argument("--decode-freq-l4", type=parse_csv_ints, default=parse_csv_ints("780,1410,2040"))
    parser.add_argument("--prefill-tp-l40s", type=parse_csv_ints, default=parse_csv_ints("1,4"))
    parser.add_argument("--prefill-tp-l4", type=parse_csv_ints, default=parse_csv_ints("1,4"))
    parser.add_argument("--decode-tp-l40s", type=parse_csv_ints, default=parse_csv_ints("1,2,4"))
    parser.add_argument("--decode-tp-l4", type=parse_csv_ints, default=parse_csv_ints("1,2,4"))
    parser.add_argument("--freq-l40s", type=parse_csv_ints)
    parser.add_argument("--freq-l4", type=parse_csv_ints)
    parser.add_argument("--tp-l40s", type=parse_csv_ints)
    parser.add_argument("--tp-l4", type=parse_csv_ints)
    parser.add_argument("--prefill-ils", type=parse_csv_ints, default=parse_csv_ints("128,1024,4096"))
    parser.add_argument("--decode-ils", type=parse_csv_ints, default=parse_csv_ints("128,1024,4096"))
    parser.add_argument("--decode-ols", type=parse_csv_ints, default=parse_csv_ints("32,256,1024"))
    parser.add_argument("--prefill-rates", type=parse_csv_ints, default=parse_csv_ints("1,10"))
    parser.add_argument("--decode-rates", type=parse_csv_ints, default=parse_csv_ints("1,5,10"))
    parser.add_argument("--rates", type=parse_csv_ints)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--l40s-node", default="neptune")
    parser.add_argument("--l4-node", default="europa")
    parser.add_argument("--prefill-port", type=int, default=8100)
    parser.add_argument("--decode-port", type=int, default=8200)
    parser.add_argument("--l40s-mem-freq", type=int, default=9001)
    parser.add_argument("--l4-mem-freq", type=int, default=6251)
    args = parser.parse_args()

    apply_legacy_overrides(args)
    if args.smoke:
        apply_smoke_overrides(args)
    apply_model_tp_constraints(args)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows(args)
    write_matrix(prefix.with_name(prefix.name + "_matrix.csv"), rows)

    prefill_per_gpu = len(args.prefill_tp_l40s) * len(args.prefill_freq_l40s) * len(args.prefill_ils) * len(args.prefill_rates)
    prefill_per_gpu_l4 = len(args.prefill_tp_l4) * len(args.prefill_freq_l4) * len(args.prefill_ils) * len(args.prefill_rates)
    decode_per_gpu = len(args.decode_tp_l40s) * len(args.decode_freq_l40s) * len(args.decode_ils) * len(args.decode_ols) * len(args.decode_rates)
    decode_per_gpu_l4 = len(args.decode_tp_l4) * len(args.decode_freq_l4) * len(args.decode_ils) * len(args.decode_ols) * len(args.decode_rates)

    summary = {
        "model_id": args.model_id,
        "hf_name": args.hf_name,
        "family": args.family,
        "param_count_b": args.size_b,
        "result_dir": args.result_dir,
        "smoke_mode": bool(args.smoke),
        "compact_defaults": {
            "prefill_freq_l40s_mhz": args.prefill_freq_l40s,
            "prefill_freq_l4_mhz": args.prefill_freq_l4,
            "decode_freq_l40s_mhz": args.decode_freq_l40s,
            "decode_freq_l4_mhz": args.decode_freq_l4,
            "prefill_tp_l40s": args.prefill_tp_l40s,
            "prefill_tp_l4": args.prefill_tp_l4,
            "decode_tp_l40s": args.decode_tp_l40s,
            "decode_tp_l4": args.decode_tp_l4,
            "prefill_input_lens": args.prefill_ils,
            "decode_input_lens": args.decode_ils,
            "decode_output_lens": args.decode_ols,
            "prefill_request_rates": args.prefill_rates,
            "decode_request_rates": args.decode_rates,
        },
        "profiling_count": {
            "prefill_l40s": prefill_per_gpu,
            "prefill_l4": prefill_per_gpu_l4,
            "decode_l40s": decode_per_gpu,
            "decode_l4": decode_per_gpu_l4,
            "total_runs": prefill_per_gpu + prefill_per_gpu_l4 + decode_per_gpu + decode_per_gpu_l4,
        },
        "rationale": [
            "The default pilot is decode-heavy: decode keeps low/mid/high DVFS points and TP={1,2,4}, while prefill is intentionally sampled more sparsely.",
            "Prefill keeps only low/high DVFS points, TP={1,4}, and low/high request rates to verify whether zero-shot failure is primarily a decode-side problem.",
            "Short/medium/long length points cover the dominant latency and KV-pressure regimes without a full sweep.",
            "The pilot reuses phase-only profiling so failure modes can be attributed separately to prefill and decode before any end-to-end transfer claim is made.",
            "Smoke mode exists only to validate collection, summarization, and Mode A/B/C evaluation plumbing before launching the real pilot.",
        ],
        "notes": [
            "If TP=4 is invalid on the new size for a specific GPU, drop it and regenerate the matrix rather than silently substituting another point.",
            "This pilot is intended to diagnose same-family multi-size behavior, not to claim transfer is solved.",
        ],
    }
    prefix.with_name(prefix.name + "_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    collect_path = prefix.with_name(prefix.name + "_collect.sh")
    submit_path = prefix.with_name(prefix.name + "_submit_sbatch.sh")
    collect_sbatch_path = prefix.with_name(prefix.name + "_collect_sbatch.sh")
    load_sanity_path = prefix.with_name(prefix.name + "_load_sanity.sh")
    load_sanity_sbatch_path = prefix.with_name(prefix.name + "_load_sanity_sbatch.sh")
    pipeline_path = prefix.with_name(prefix.name + "_pipeline.sh")
    pipeline_sbatch_path = prefix.with_name(prefix.name + "_pipeline_sbatch.sh")
    collect_path.write_text(build_collect_script(args, prefix), encoding="utf-8")
    submit_path.write_text(build_submit_script(prefix), encoding="utf-8")
    collect_sbatch_path.write_text(build_collect_sbatch_script(prefix), encoding="utf-8")
    load_sanity_path.write_text(build_load_sanity_script(args, prefix), encoding="utf-8")
    load_sanity_sbatch_path.write_text(build_load_sanity_sbatch_script(prefix), encoding="utf-8")
    pipeline_path.write_text(build_pipeline_script(prefix), encoding="utf-8")
    pipeline_sbatch_path.write_text(build_pipeline_sbatch_script(prefix), encoding="utf-8")

    print(prefix.with_name(prefix.name + "_matrix.csv"))
    print(prefix.with_name(prefix.name + "_summary.json"))
    print(collect_path)
    print(submit_path)
    print(collect_sbatch_path)
    print(load_sanity_path)
    print(load_sanity_sbatch_path)
    print(pipeline_path)
    print(pipeline_sbatch_path)


if __name__ == "__main__":
    main()
