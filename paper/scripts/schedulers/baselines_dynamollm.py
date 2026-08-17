"""DynamoLLM baseline (extracted verbatim)."""
from __future__ import annotations

from typing import Tuple

from .common import ROUTE_AA, ROUTE_BB, RequestClassSummary
from .sweep import SweepLLMStrategy


class DynamoLLMMono(SweepLLMStrategy):
    """DynamoLLM-inspired monolithic-routing baseline (a.k.a. Monolithic-Joint).

    This baseline captures DynamoLLM's (Stojkovic et al., 2024) key architectural
    restriction — *no phase-disaggregated execution*: a request's prefill and
    decode run on the same GPU pool (ROUTE_AA or ROUTE_BB), and cross-pool routes
    (ROUTE_AB / ROUTE_BA) are disabled.  It is **not** a faithful reproduction of
    DynamoLLM: it does not implement the hierarchical multi-timescale control
    (30-min scale, 5-min shard, 5-s frequency), the per-request-type pools with
    merging/fragmentation handling, output-length prediction, or reconfiguration
    overhead accounting.

    Instead it reuses the SWEEP-LLM joint search backend with cross-pool routes
    disabled.  That makes it a *stronger* baseline than a literal DynamoLLM
    reproduction in our setting (it gets SWEEP's online joint TP/freq/capacity
    search), which keeps the comparison conservative for SWEEP-LLM: it isolates
    the marginal contribution of disaggregated routing.  In the paper it should
    be labelled "DynamoLLM-Mono" / "Monolithic-Joint", not "DynamoLLM".
    """

    def _admissible_routes(self, cls: RequestClassSummary, state: str) -> Tuple[Tuple[str, str], ...]:
        return (ROUTE_AA, ROUTE_BB)

    def _try_monolithic_consolidation(self, summary, slo, state, burst):
        # Disabled for baseline fairness.  The previous override applied a
        # ROUTE_BB-only, rho-gated shortcut that (a) was asymmetric — it never
        # tried ROUTE_AA / L40S-only scale-in, which DynamoLLM's homogeneous
        # design has no analogue of — and (b) force-set slo_met=True from a
        # require_safe=False evaluation, masking violations the strict gate
        # would catch.  The main joint search already explores both ROUTE_AA
        # and ROUTE_BB monolithic placements, so no shortcut is needed.
        return None


