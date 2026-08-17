#!/usr/bin/env python3
"""
Transfer-ready corpus builders for factorized modeling experiments.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from dataset_manifest import get_dataset_manifest, get_dataset_manifest_for_gpu
from model_catalog import get_model_feature_dict
from model_training_common import prepare_data


def attach_model_metadata(df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    out = df.copy()
    for key, value in get_model_feature_dict(model_id).items():
        out[key] = value
    return out


def load_transfer_ready_dataset(dataset_id: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    manifest = get_dataset_manifest(dataset_id)
    max_freq = 2520 if manifest.gpu_type == "l40s" else 2040
    from model_training_common import GPU_CONFIGS  # local import to avoid cycle

    cfg = GPU_CONFIGS[manifest.gpu_type]
    df = prepare_data(manifest.csv_path, cfg["anomaly_filter"], max_freq)
    df = attach_model_metadata(df, manifest.model_id)
    df["dataset_id"] = manifest.dataset_id
    return df, manifest.to_metadata_dict()


def load_transfer_ready_dataset_for_gpu(gpu_type: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    manifest = get_dataset_manifest_for_gpu(gpu_type)
    return load_transfer_ready_dataset(manifest.dataset_id)


def build_reference_prior_dataset(df: pd.DataFrame, max_freq: int | None = None) -> pd.DataFrame:
    out = df.copy()
    if max_freq is None:
        max_freq = int(out["gpu_freq_mhz"].max())
    ref_tp = int(out["tp_degree"].min())
    ref = out[(out["step"] == "step2a") &
              (out["gpu_freq_mhz"] == max_freq) &
              (out["tp_degree"] == ref_tp)].copy()
    ref["reference_tp"] = ref_tp
    ref["reference_freq_mhz"] = max_freq
    ref["component_type"] = "reference_prior"
    return ref


def build_tp_residual_dataset(df: pd.DataFrame, max_freq: int | None = None) -> pd.DataFrame:
    out = df.copy()
    if max_freq is None:
        max_freq = int(out["gpu_freq_mhz"].max())
    ref_tp = int(out["tp_degree"].min())

    tp_df = out[(out["step"] == "step2b") & (out["gpu_freq_mhz"] == max_freq)].copy()
    ref = out[(out["tp_degree"] == ref_tp) & (out["gpu_freq_mhz"] == max_freq)].copy()
    merge_cols = ["gpu_type", "input_len", "output_len", "request_rate"]
    ref_cols = merge_cols + [
        "mean_ttft_ms", "mean_tpot_ms", "avg_power_w",
        "power_per_gpu", "total_token_throughput_tps"
    ]
    ref = ref[ref_cols].drop_duplicates()
    merged = tp_df.merge(ref, on=merge_cols, how="left", suffixes=("", "_ref"))
    merged = merged[merged["tp_degree"] != ref_tp].copy()
    merged["delta_ttft_ms"] = merged["mean_ttft_ms"] - merged["mean_ttft_ms_ref"]
    merged["delta_tpot_ms"] = merged["mean_tpot_ms"] - merged["mean_tpot_ms_ref"]
    merged["delta_power_per_gpu_w"] = merged["power_per_gpu"] - merged["power_per_gpu_ref"]
    merged["delta_total_tps"] = (
        merged["total_token_throughput_tps"] - merged["total_token_throughput_tps_ref"]
    )
    merged["component_type"] = "tp_residual"
    merged["reference_tp"] = ref_tp
    merged["reference_freq_mhz"] = max_freq
    return merged


def build_frequency_residual_dataset(df: pd.DataFrame, max_freq: int | None = None) -> pd.DataFrame:
    out = df.copy()
    if max_freq is None:
        max_freq = int(out["gpu_freq_mhz"].max())

    freq_df = out[out["step"] == "step2c"].copy()
    ref = out[out["gpu_freq_mhz"] == max_freq].copy()
    merge_cols = ["gpu_type", "input_len", "output_len", "request_rate", "tp_degree"]
    ref_cols = merge_cols + [
        "mean_ttft_ms", "mean_tpot_ms", "avg_power_w",
        "power_per_gpu", "total_token_throughput_tps"
    ]
    ref = ref[ref_cols].drop_duplicates()
    merged = freq_df.merge(ref, on=merge_cols, how="left", suffixes=("", "_ref"))
    merged = merged[merged["gpu_freq_mhz"] != max_freq].copy()
    merged["delta_ttft_ms"] = merged["mean_ttft_ms"] - merged["mean_ttft_ms_ref"]
    merged["delta_tpot_ms"] = merged["mean_tpot_ms"] - merged["mean_tpot_ms_ref"]
    merged["delta_power_per_gpu_w"] = merged["power_per_gpu"] - merged["power_per_gpu_ref"]
    merged["delta_total_tps"] = (
        merged["total_token_throughput_tps"] - merged["total_token_throughput_tps_ref"]
    )
    merged["freq_ratio"] = merged["gpu_freq_mhz"] / max_freq
    merged["component_type"] = "freq_residual"
    merged["reference_freq_mhz"] = max_freq
    return merged


def summarize_factorized_coverage(df: pd.DataFrame) -> Dict[str, object]:
    return {
        "rows": int(len(df)),
        "steps": {k: int(v) for k, v in df["step"].value_counts().sort_index().items()},
        "freqs": sorted(df["gpu_freq_mhz"].dropna().unique().tolist()),
        "tps": sorted(df["tp_degree"].dropna().unique().tolist()),
        "input_lens": sorted(df["input_len"].dropna().unique().tolist()),
        "output_lens": sorted(df["output_len"].dropna().unique().tolist()),
        "request_rates": sorted(df["request_rate"].dropna().unique().tolist()),
        "model_ids": sorted(df["model_id"].dropna().unique().tolist()),
    }


def build_factorized_views(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    max_freq = int(df["gpu_freq_mhz"].max())
    return {
        "reference_prior": build_reference_prior_dataset(df, max_freq=max_freq),
        "tp_residual": build_tp_residual_dataset(df, max_freq=max_freq),
        "freq_residual": build_frequency_residual_dataset(df, max_freq=max_freq),
    }
