"""Backward-compatibility shim for the SWEEP-LLM scheduler family.

The implementation now lives in the ``schedulers`` package (common / sweep /
baselines_* / ablations / factory).  This module re-exports every public symbol
that callers historically imported from ``sweep_llm_scheduler`` so existing
scripts keep working unchanged.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# External symbols that used to live in this module's namespace; some scripts
# import them from here (e.g. oracle_comparison imports snap_to_nearest).
from jsep_cluster import ClusterState, GPU_SPECS
from jsep_scheduler import SchedulingResult
from jsep_traces import IL_VALUES, OL_VALUES, Request, snap_to_nearest
from paths import paper_model_dir
from scheduler import EnergyScheduler

from schedulers.common import (
    POOL_TO_GPU, GPU_TO_POOL,
    ROUTE_AA, ROUTE_AB, ROUTE_BA, ROUTE_BB, ALL_ROUTES,
    SweepLLMConfig, RequestClassSummary, WindowSummary,
    PoolBundle, SearchCandidate, SearchResult,
)
from schedulers.sweep import SweepLLMStrategy
from schedulers.baselines_static import StaticDisaggStrategy, GreenLLMStrategy
from schedulers.baselines_dynamollm import DynamoLLMMono
from schedulers.baselines_dualscale import (
    DualScaleStrategy,
    DualScaleExt20s,
    DualScaleExt1Min,
    DualScaleExt2Min,
)
from schedulers.baselines_hierarchical import HierarchicalDisaggStrategy
from schedulers.ablations import (
    SweepLLMNoRoutingStrategy,
    SweepLLMNoDVFSStrategy,
    SweepLLMNoTPStrategy,
    SweepLLMNoCapacityStrategy,
)
from schedulers.factory import (
    create_sweep_llm_strategies,
    create_dynamollm_strategy,
    create_dualscale_strategy,
    create_greenllm_strategy,
    create_static_disagg_strategy,
    create_hierarchical_disagg_strategy,
    create_sweep_llm_no_routing,
    create_sweep_llm_no_dvfs,
    create_sweep_llm_no_tp,
    create_sweep_llm_no_capacity,
)
