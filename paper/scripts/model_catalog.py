#!/usr/bin/env python3
"""
Model descriptor registry for transfer-ready SWEEP-LLM datasets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    hf_name: str
    family: str
    architecture: str
    param_count_b: float
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    precision: str
    kv_bytes_per_token: int
    attention_variant: str

    def to_feature_dict(self) -> Dict[str, object]:
        return {
            "model_id": self.model_id,
            "hf_name": self.hf_name,
            "model_family": self.family,
            "model_architecture": self.architecture,
            "param_count_b": self.param_count_b,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "precision": self.precision,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "attention_variant": self.attention_variant,
        }


MODEL_SPECS: Dict[str, ModelSpec] = {
    "mistral_7b_v01": ModelSpec(
        model_id="mistral_7b_v01",
        hf_name="mistralai/Mistral-7B-v0.1",
        family="mistral",
        architecture="decoder_only_transformer",
        param_count_b=7.0,
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        precision="bf16",
        kv_bytes_per_token=131072,
        attention_variant="gqa_sliding_window",
    ),
    "llama3_8b": ModelSpec(
        model_id="llama3_8b",
        hf_name="meta-llama/Meta-Llama-3-8B",
        family="llama3",
        architecture="decoder_only_transformer",
        param_count_b=8.0,
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        precision="bf16",
        kv_bytes_per_token=65536,
        attention_variant="gqa",
    ),
    "llama2_13b": ModelSpec(
        model_id="llama2_13b",
        hf_name="meta-llama/Llama-2-13b-hf",
        family="llama2",
        architecture="decoder_only_transformer",
        param_count_b=13.0,
        hidden_size=5120,
        num_layers=40,
        num_heads=40,
        num_kv_heads=40,
        head_dim=128,
        precision="bf16",
        kv_bytes_per_token=819200,
        attention_variant="mha",
    ),
}


MODEL_FEATURE_COLUMNS: List[str] = list(
    MODEL_SPECS["mistral_7b_v01"].to_feature_dict().keys()
)


def get_model_spec(model_id: str) -> ModelSpec:
    if model_id not in MODEL_SPECS:
        raise KeyError(f"Unknown model_id: {model_id}")
    return MODEL_SPECS[model_id]


def get_model_feature_dict(model_id: str) -> Dict[str, object]:
    return asdict(get_model_spec(model_id))
