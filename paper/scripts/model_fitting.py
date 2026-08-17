#!/usr/bin/env python3
"""
L40S training entrypoint for the SWEEP-LLM single-pool model pipeline.
"""

from model_fitting_runner import run_training_pipeline


if __name__ == "__main__":
    run_training_pipeline("l40s")
