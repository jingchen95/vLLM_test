"""Factory functions for constructing scheduler strategies (extracted verbatim)."""
from __future__ import annotations

from typing import Dict, Optional

from scheduler import EnergyScheduler
from paths import paper_model_dir

from .common import SweepLLMConfig
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


def _create_ablation(cls, model_dir_l40s: str | None, model_dir_l4: str | None,
                     window_s: float, config: Optional[SweepLLMConfig]):
    model_dir_l40s = str(paper_model_dir("models_l40s")) if model_dir_l40s is None else model_dir_l40s
    model_dir_l4 = str(paper_model_dir("models_l4")) if model_dir_l4 is None else model_dir_l4
    schedulers = {
        "l40s": EnergyScheduler(model_dir=model_dir_l40s),
        "l4": EnergyScheduler(model_dir=model_dir_l4),
    }
    cfg = config or SweepLLMConfig(
        ideal_every_window=False,
        emergency_bundle_override=True,
        print_search_stats=False,
        ideal_refresh_windows=12,
        ideal_load_change_frac=0.50,
        ideal_on_class_mix_change=False,
        ideal_state_change_hold_windows=2,
    )
    return cls(schedulers, config=cfg, window_s=window_s)


def create_sweep_llm_no_routing(model_dir_l40s=None, model_dir_l4=None,
                                window_s: float = 5.0, config=None):
    return _create_ablation(SweepLLMNoRoutingStrategy, model_dir_l40s, model_dir_l4, window_s, config)


def create_sweep_llm_no_dvfs(model_dir_l40s=None, model_dir_l4=None,
                             window_s: float = 5.0, config=None):
    return _create_ablation(SweepLLMNoDVFSStrategy, model_dir_l40s, model_dir_l4, window_s, config)


def create_sweep_llm_no_tp(model_dir_l40s=None, model_dir_l4=None,
                           window_s: float = 5.0, config=None):
    return _create_ablation(SweepLLMNoTPStrategy, model_dir_l40s, model_dir_l4, window_s, config)


def create_sweep_llm_no_capacity(model_dir_l40s=None, model_dir_l4=None,
                                 window_s: float = 5.0, config=None):
    return _create_ablation(SweepLLMNoCapacityStrategy, model_dir_l40s, model_dir_l4, window_s, config)


def create_hierarchical_disagg_strategy(model_dir_l40s: str | None = None,
                                        model_dir_l4: str | None = None,
                                        window_s: float = 5.0,
                                        config: Optional[SweepLLMConfig] = None) -> HierarchicalDisaggStrategy:
    return _create_ablation(HierarchicalDisaggStrategy, model_dir_l40s, model_dir_l4, window_s, config)


def create_dynamollm_strategy(model_dir_l40s: str | None = None,
                              model_dir_l4: str | None = None,
                              window_s: float = 5.0,
                              config: Optional[SweepLLMConfig] = None) -> DynamoLLMMono:
    model_dir_l40s = str(paper_model_dir("models_l40s")) if model_dir_l40s is None else model_dir_l40s
    model_dir_l4 = str(paper_model_dir("models_l4")) if model_dir_l4 is None else model_dir_l4
    schedulers = {
        "l40s": EnergyScheduler(model_dir=model_dir_l40s),
        "l4": EnergyScheduler(model_dir=model_dir_l4),
    }
    cfg = config or SweepLLMConfig(
        ideal_every_window=False,
        emergency_bundle_override=True,
        print_search_stats=False,
        ideal_refresh_windows=12,
        ideal_load_change_frac=0.50,
        ideal_on_class_mix_change=False,
        ideal_state_change_hold_windows=2,
    )
    return DynamoLLMMono(schedulers, config=cfg, window_s=window_s)


def create_dualscale_strategy(model_dir_l40s: str | None = None,
                              model_dir_l4: str | None = None,
                              window_s: float = 5.0,
                              config: Optional[SweepLLMConfig] = None) -> DualScaleStrategy:
    model_dir_l40s = str(paper_model_dir("models_l40s")) if model_dir_l40s is None else model_dir_l40s
    model_dir_l4 = str(paper_model_dir("models_l4")) if model_dir_l4 is None else model_dir_l4
    schedulers = {
        "l40s": EnergyScheduler(model_dir=model_dir_l40s),
        "l4": EnergyScheduler(model_dir=model_dir_l4),
    }
    cfg = config or SweepLLMConfig(
        ideal_every_window=False,
        emergency_bundle_override=True,
        print_search_stats=False,
        ideal_refresh_windows=12,
        ideal_load_change_frac=0.50,
        ideal_on_class_mix_change=False,
        ideal_state_change_hold_windows=2,
    )
    return DualScaleStrategy(schedulers, config=cfg, window_s=window_s)


def create_greenllm_strategy(model_dir_l40s: str | None = None,
                             model_dir_l4: str | None = None,
                             window_s: float = 5.0,
                             config: Optional[SweepLLMConfig] = None) -> GreenLLMStrategy:
    model_dir_l40s = str(paper_model_dir("models_l40s")) if model_dir_l40s is None else model_dir_l40s
    model_dir_l4 = str(paper_model_dir("models_l4")) if model_dir_l4 is None else model_dir_l4
    schedulers = {
        "l40s": EnergyScheduler(model_dir=model_dir_l40s),
        "l4": EnergyScheduler(model_dir=model_dir_l4),
    }
    cfg = config or SweepLLMConfig(
        ideal_every_window=False,
        emergency_bundle_override=False,
        print_search_stats=False,
    )
    return GreenLLMStrategy(schedulers, config=cfg, window_s=window_s)


def create_static_disagg_strategy(model_dir_l40s: str | None = None,
                                  model_dir_l4: str | None = None,
                                  window_s: float = 5.0,
                                  config: Optional[SweepLLMConfig] = None) -> StaticDisaggStrategy:
    model_dir_l40s = str(paper_model_dir("models_l40s")) if model_dir_l40s is None else model_dir_l40s
    model_dir_l4 = str(paper_model_dir("models_l4")) if model_dir_l4 is None else model_dir_l4
    schedulers = {
        "l40s": EnergyScheduler(model_dir=model_dir_l40s),
        "l4": EnergyScheduler(model_dir=model_dir_l4),
    }
    cfg = config or SweepLLMConfig(
        ideal_every_window=False,
        emergency_bundle_override=False,
        print_search_stats=False,
    )
    return StaticDisaggStrategy(schedulers, config=cfg, window_s=window_s)


def create_sweep_llm_strategies(model_dir_l40s: str | None = None,
                                model_dir_l4: str | None = None,
                                window_s: float = 5.0,
                                config: Optional[SweepLLMConfig] = None) -> Dict[str, SweepLLMStrategy]:
    model_dir_l40s = str(paper_model_dir("models_l40s")) if model_dir_l40s is None else model_dir_l40s
    model_dir_l4 = str(paper_model_dir("models_l4")) if model_dir_l4 is None else model_dir_l4
    schedulers = {
        "l40s": EnergyScheduler(model_dir=model_dir_l40s),
        "l4": EnergyScheduler(model_dir=model_dir_l4),
    }
    if config is not None:
        return {
            "sweep_llm": SweepLLMStrategy(schedulers, config=config, window_s=window_s),
        }
    return {
        "sweep_llm": SweepLLMStrategy(
            schedulers,
            config=SweepLLMConfig(ideal_every_window=True),
            window_s=window_s,
        ),
        "sweep_llm_triggered": SweepLLMStrategy(
            schedulers,
            config=SweepLLMConfig(
                ideal_every_window=False,
                ideal_refresh_windows=12,
                ideal_load_change_frac=0.50,
                ideal_on_class_mix_change=False,
                ideal_state_change_hold_windows=2,
            ),
            window_s=window_s,
        ),
        "sweep_llm_emergency": SweepLLMStrategy(
            schedulers,
            config=SweepLLMConfig(
                ideal_every_window=True,
                emergency_bundle_override=True,
            ),
            window_s=window_s,
        ),
        "sweep_llm_hybrid": SweepLLMStrategy(
            schedulers,
            config=SweepLLMConfig(
                ideal_every_window=False,
                emergency_bundle_override=True,
                ideal_refresh_windows=12,
                ideal_load_change_frac=0.50,
                ideal_on_class_mix_change=False,
                ideal_state_change_hold_windows=2,
            ),
            window_s=window_s,
        ),
    }
