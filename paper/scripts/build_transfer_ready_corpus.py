#!/usr/bin/env python3
"""
Materialize transfer-ready corpora and factorized views for Phase 2 modeling.
"""

from __future__ import annotations

import argparse
import json
import os

from dataset_manifest import DATASET_MANIFESTS
from factorized_dataset import (
    build_factorized_views,
    load_transfer_ready_dataset,
    summarize_factorized_coverage,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-id",
        choices=sorted(DATASET_MANIFESTS.keys()),
        required=True,
        help="Dataset manifest to materialize.",
    )
    parser.add_argument(
        "--output-dir",
        default="factorized_corpora",
        help="Directory to store the augmented corpus and derived views.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df, meta = load_transfer_ready_dataset(args.dataset_id)
    views = build_factorized_views(df)

    base_csv = os.path.join(args.output_dir, f"{args.dataset_id}_augmented.csv")
    df.to_csv(base_csv, index=False)

    summary = {
        "dataset": meta,
        "full_corpus": summarize_factorized_coverage(df),
        "views": {},
    }
    for name, view_df in views.items():
        out_csv = os.path.join(args.output_dir, f"{args.dataset_id}_{name}.csv")
        view_df.to_csv(out_csv, index=False)
        summary["views"][name] = summarize_factorized_coverage(view_df)
        summary["views"][name]["csv_path"] = out_csv

    summary["full_corpus"]["csv_path"] = base_csv
    summary_path = os.path.join(args.output_dir, f"{args.dataset_id}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
