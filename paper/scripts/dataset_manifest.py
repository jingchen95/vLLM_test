#!/usr/bin/env python3
"""
Dataset manifests linking profiling corpora to model descriptors.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

from model_catalog import get_model_feature_dict
from paths import REPO_ROOT


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    csv_path: str
    gpu_type: str
    model_id: str
    serving_stack: str
    notes: str

    def to_metadata_dict(self) -> Dict[str, object]:
        meta = {
            "dataset_id": self.dataset_id,
            "csv_path": self.csv_path,
            "gpu_type": self.gpu_type,
            "model_id": self.model_id,
            "serving_stack": self.serving_stack,
            "notes": self.notes,
        }
        meta.update(get_model_feature_dict(self.model_id))
        return meta


DATASET_MANIFESTS: Dict[str, DatasetManifest] = {
    "phase2_l40s_mistral7b": DatasetManifest(
        dataset_id="phase2_l40s_mistral7b",
        csv_path=str(REPO_ROOT / "Phase2_Results_L40S" / "master_results.csv"),
        gpu_type="l40s",
        model_id="mistral_7b_v01",
        serving_stack="vllm_single_pool",
        notes="Phase 2 L40S characterization for Mistral-7B-v0.1.",
    ),
    "phase2_l4_mistral7b": DatasetManifest(
        dataset_id="phase2_l4_mistral7b",
        csv_path=str(REPO_ROOT / "Phase2_Results_L4" / "master_results.csv"),
        gpu_type="l4",
        model_id="mistral_7b_v01",
        serving_stack="vllm_single_pool",
        notes="Phase 2 L4 characterization for Mistral-7B-v0.1.",
    ),
}

GPU_TO_DATASET_ID = {
    "l40s": "phase2_l40s_mistral7b",
    "l4": "phase2_l4_mistral7b",
}


def get_dataset_manifest(dataset_id: str) -> DatasetManifest:
    if dataset_id not in DATASET_MANIFESTS:
        raise KeyError(f"Unknown dataset_id: {dataset_id}")
    return DATASET_MANIFESTS[dataset_id]


def get_dataset_manifest_for_gpu(gpu_type: str) -> DatasetManifest:
    if gpu_type not in GPU_TO_DATASET_ID:
        raise KeyError(f"No dataset manifest registered for gpu_type={gpu_type}")
    return get_dataset_manifest(GPU_TO_DATASET_ID[gpu_type])
