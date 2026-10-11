"""Generate XPYD runtime and experiment configs for a Grid'5000 OAR job.

Node names, GPU models and supported clocks are only known once the job
starts, so the launcher discovers them and calls this script. The emitted
files mirror the Minerva configs (same SLO, trace, mix, windows, repeats);
only topology, clocks, model path and output roots differ.

Usage:
    gen_g5k_configs.py --mode e0|e0c --out-dir DIR --repo REPO
        --p-node H --d-node H --p-ip IP --d-ip IP --proxy-host HOST
        --p-gpu-name NAME --d-gpu-name NAME
        --p-levels LOW,MID,HIGH --d-levels LOW,MID,HIGH
        --p-mem-mhz N --d-mem-mhz N --model-dir DIR --result-root DIR
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

SERVED_MODEL = "mistralai/Mistral-7B-v0.1"
KV_PORTS = {"P0": 14579, "P1": 14580, "D0": 14579, "D1": 14580}
HTTP_PORTS = {"P0": 8100, "P1": 8101, "D0": 8200, "D1": 8201}
PAIR_REASON = "Grid'5000 2P2D; vLLM 0.15.1 synchronous-PUT TP1 contract"


def _levels(text: str) -> dict[str, int]:
    low, mid, high = (int(item) for item in text.split(","))
    if not 0 < low < mid < high:
        raise SystemExit("levels must satisfy 0 < LOW < MID < HIGH: %s" % text)
    return {"LOW": low, "MID": mid, "HIGH": high}


def _gpu_token(name: str) -> str:
    """Short, case-insensitive substring the monitors match against."""
    for token in ("H200", "H100", "A100", "L40S", "L4", "A40", "V100"):
        if token.lower() in name.lower():
            return token
    return name


def runtime_config(args: argparse.Namespace, p_levels: dict[str, int], d_levels: dict[str, int]) -> dict[str, Any]:
    result_root = Path(args.result_root)
    endpoints = []
    clocks = {}
    for endpoint_id in ("P0", "P1", "D0", "D1"):
        prefill = endpoint_id.startswith("P")
        node = args.p_node if prefill else args.d_node
        ip = args.p_ip if prefill else args.d_ip
        http_host = "127.0.0.1" if node == args.proxy_host else ip
        gpu_index = int(endpoint_id[1])
        endpoints.append({
            "endpoint_id": endpoint_id, "role": "prefill" if prefill else "decode", "node": node,
            "gpu_type": args.p_gpu_name if prefill else args.d_gpu_name,
            "gpu_ids": [gpu_index], "cuda_visible_device": str(gpu_index),
            "tp_degree": 1,
            "http_uri": "http://%s:%d" % (http_host, HTTP_PORTS[endpoint_id]),
            "http_host": http_host, "http_port": HTTP_PORTS[endpoint_id],
            "addr_host": ip, "kv_port": KV_PORTS[endpoint_id],
            "kv_connector": "P2pNcclConnector",
            "configured_frequency_mhz": (p_levels if prefill else d_levels)["HIGH"],
            "expected_gpu_name": _gpu_token(args.p_gpu_name if prefill else args.d_gpu_name),
        })
        clocks[endpoint_id] = {
            "graphics_mhz": (p_levels if prefill else d_levels)["HIGH"],
            "memory_mhz": args.p_mem_mhz if prefill else args.d_mem_mhz,
            "minimum_workload_match_fraction": 0.95,
        }
    pairs = [
        {"prefill_endpoint_id": p, "decode_endpoint_id": d, "connector": "P2pNcclConnector",
         "prefill_tp": 1, "decode_tp": 1, "supported": True, "reason": PAIR_REASON}
        for p in ("P0", "P1") for d in ("D0", "D1")
    ]
    return {
        "model": SERVED_MODEL,
        "tokenizer_model": args.model_dir,
        "vllm_version": "0.15.1",
        "output_root": (result_root / "runtime_windows").as_posix(),
        "proxy_uri": "http://127.0.0.1:%d" % args.proxy_port,
        "proxy_diagnostics_log": (result_root / "proxy_diagnostics.jsonl").as_posix(),
        "routing_policy": "round_robin",
        "routing_control_file": "",
        "routing_control_maximum_age_s": 300.0,
        "routing_cold_start_requests": 4,
        "routing_candidate_probe_interval_requests": 2,
        "routing_exploration_interval_requests": 4,
        "scrape_interval_s": 1.0,
        "collector": {"timeout_s": 5.0},
        "pre_window_s": 1.0,
        "post_window_s": 0.5,
        "sampling": {
            "period_s": 0.2, "late_tolerance_s": 0.05, "readiness_timeout_s": 120.0,
            "maximum_gap_s": 0.5, "maximum_boundary_gap_s": 0.4, "minimum_coverage_ratio": 0.9,
        },
        "client": {
            "script": "paper/scripts/replay_synthetic_trace.py",
            "max_concurrency": 1, "request_timeout_s": 900,
        },
        "endpoints": endpoints,
        "fixed_clocks": clocks,
        "compatible_pairs": pairs,
        "workloads": [{"id": "placeholder", "input_len": 128, "output_len": 128, "count": 1, "rate_rps": 1.0}],
        "platform": "g5k",
        "server_logs": {
            endpoint_id: (result_root / ("vllm_%s.log" % endpoint_id)).as_posix()
            for endpoint_id in ("P0", "P1", "D0", "D1")
        },
    }


def experiment_config(args: argparse.Namespace, p_levels: dict[str, int], d_levels: dict[str, int]) -> dict[str, Any]:
    repo = Path(args.repo)
    source = {
        "e0": "paper/configs/xpyd_stage_e0_regime_neptune_europa.json",
        "e0c": "paper/configs/xpyd_stage_e0c_capacity_neptune_europa.json",
    }[args.mode]
    config = json.loads((repo / source).read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["platform"] = "g5k"
    config["expected_topology"] = {
        "P0": args.p_node, "P1": args.p_node, "D0": args.d_node, "D1": args.d_node,
    }
    config["output_root"] = (Path(args.result_root) / "stage").as_posix()
    config["purpose"] = "Grid'5000 port of: " + str(config.get("purpose", ""))
    if args.mode == "e0":
        levels = {"P0": p_levels, "P1": p_levels, "D0": d_levels, "D1": d_levels}
        config["stagec1"]["frequency_levels_mhz"] = levels
        config["physical_topology"] = {"prefill_node": args.p_node, "decode_node": args.d_node}
    else:
        config["physical_topology"] = {"prefill_node": args.p_node, "decode_node": args.d_node}
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("e0", "e0c"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--p-node", required=True)
    parser.add_argument("--d-node", required=True)
    parser.add_argument("--p-ip", required=True)
    parser.add_argument("--d-ip", required=True)
    parser.add_argument("--proxy-host", required=True, help="node (FQDN) that runs the proxy and the harness")
    parser.add_argument("--proxy-port", type=int, default=8000)
    parser.add_argument("--p-gpu-name", required=True)
    parser.add_argument("--d-gpu-name", required=True)
    parser.add_argument("--p-levels", required=True, help="LOW,MID,HIGH graphics MHz for the prefill pool")
    parser.add_argument("--d-levels", required=True)
    parser.add_argument("--p-mem-mhz", type=int, required=True)
    parser.add_argument("--d-mem-mhz", type=int, required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    p_levels, d_levels = _levels(args.p_levels), _levels(args.d_levels)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = args.out_dir / "runtime_g5k.json"
    experiment_path = args.out_dir / ("experiment_%s_g5k.json" % args.mode)
    runtime_path.write_text(json.dumps(runtime_config(args, p_levels, d_levels), indent=2), encoding="utf-8")
    experiment_path.write_text(json.dumps(experiment_config(args, p_levels, d_levels), indent=2), encoding="utf-8")
    print(runtime_path.as_posix())
    print(experiment_path.as_posix())


if __name__ == "__main__":
    main()
