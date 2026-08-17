"""SWEEP-LLM scheduler family.

Step-1 refactor: the former monolithic ``sweep_llm_scheduler.py`` was split into
focused modules with inheritance and logic unchanged:

  * common                  — shared constants, dataclasses, config
  * sweep                   — SweepLLMStrategy (main algorithm)
  * baselines_static        — StaticDisaggStrategy, GreenLLMStrategy
  * baselines_dynamollm     — DynamoLLMMono
  * baselines_dualscale     — DualScaleStrategy
  * baselines_hierarchical  — HierarchicalDisaggStrategy
  * ablations               — SweepLLM No{Routing,DVFS,TP,Capacity} variants
  * factory                 — create_* constructors
"""
from __future__ import annotations

import os
import sys

# Make paper/scripts (this package's parent) importable so the sibling modules
# (jsep_cluster, scheduler, jsep_traces, paths, jsep_scheduler) resolve no
# matter what the caller's sys.path looks like.
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from .common import (
    POOL_TO_GPU, GPU_TO_POOL,
    ROUTE_AA, ROUTE_AB, ROUTE_BA, ROUTE_BB, ALL_ROUTES,
    SweepLLMConfig, RequestClassSummary, WindowSummary,
    PoolBundle, SearchCandidate, SearchResult,
)
from .sweep import SweepLLMStrategy
from .baselines_static import StaticDisaggStrategy, GreenLLMStrategy
from .baselines_dynamollm import DynamoLLMMono
from .baselines_dualscale import DualScaleStrategy
from .baselines_hierarchical import HierarchicalDisaggStrategy
from .ablations import (
    SweepLLMNoRoutingStrategy,
    SweepLLMNoDVFSStrategy,
    SweepLLMNoTPStrategy,
    SweepLLMNoCapacityStrategy,
)
from .factory import (
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
