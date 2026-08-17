"""Fail-closed feasibility lookups for the SWEEP-LLM hardware gate.

Two offline-calibrated tables replace the distorted single `rho<=1` admission test:

* prefill: an SLO-conditioned utilization envelope `rho_pf_max(g, TP, f, SLO_TTFT)`
  (from `calibrate_rho_envelope.py`, prefill rows only).
* decode:  a goodput-derived safe capacity `C_dc_SLO(g, IL, OL, TP, f, SLO_TPOT)`
  (from `calibrate_decode_capacity.py`), used as `rho_dc = D_dc/(eta*C) <= 1`.

Every lookup miss fails CLOSED (returns None / source="none" => the candidate is
infeasible). No implicit fallback to `rho<=1`, no looser-SLO borrowing, no cross-TP
borrowing, no nearest-neighbor length guessing. See paper/DECODE_GATE_WIRING_DESIGN.md.
"""
from __future__ import annotations

import csv
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from jsep_traces import IL_VALUES, OL_VALUES  # discrete profiling grid


def _as_bool(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


@dataclass(frozen=True)
class CapacityLookup:
    capacity_tps: float
    # "measured": exact row, SLO-confirmed        -> eta_confirmed
    # "self_observed_lower_bound": exact, unconfirmed but C>0 -> eta_fallback
    # "lower_freq_bound": borrowed same-TP lower-freq, certified -> eta_fallback
    # "none": no safe capacity                    -> reject
    source: str
    support_key: Optional[tuple]
    requested_slo_ms: float
    selected_slo_ms: Optional[float]


def _largest_le(sorted_vals: List[float], x: float) -> Optional[float]:
    """Largest value in sorted_vals that is <= x, or None."""
    i = bisect_right(sorted_vals, x)
    return sorted_vals[i - 1] if i > 0 else None


def _brackets(x: float, grid: List[int]) -> List[int]:
    """Grid buckets bracketing x: [x] if on-grid, else {floor, ceil} that exist.

    Used for the conservative neighborhood-min length resolution (never a
    nearest-neighbor guess). Values outside the grid clamp to the nearest end
    (a single bucket), which the caller then treats as its only evidence.
    """
    g = sorted(grid)
    if x <= g[0]:
        return [g[0]]
    if x >= g[-1]:
        return [g[-1]]
    hi = next(v for v in g if v >= x)
    if hi == x:
        return [int(hi)]
    lo = max(v for v in g if v < x)
    return [int(lo), int(hi)]


class FeasibilityTables:
    def __init__(self, decode_csv, rho_envelope_csv, gate_mode: str = "goodput"):
        self.gate_mode = gate_mode
        # decode: (gpu,il,ol,tp,freq,slo) -> row ; plus (gpu,il,ol,tp,slo)->{freq:row}
        self._dc: Dict[Tuple, dict] = {}
        self._dc_by_freq: Dict[Tuple, Dict[int, dict]] = {}
        self._dc_slos: Dict[Tuple, List[float]] = {}   # (gpu,il,ol,tp) -> sorted slos
        self._dc_gpus: set = set()
        # prefill: (gpu,tp,freq) -> sorted [(slo_ms, rho_max)]
        self._pf: Dict[Tuple, List[Tuple[float, float]]] = {}
        self._pf_gpus: set = set()

        self._load_decode(Path(decode_csv))
        self._load_prefill(Path(rho_envelope_csv))

        if gate_mode == "goodput":
            for gpu in ("l40s", "l4"):
                if gpu not in self._dc_gpus:
                    raise RuntimeError(
                        f"Goodput decode gate requested, but decode capacity artifact "
                        f"has no rows for gpu={gpu} ({decode_csv}).")
                if gpu not in self._pf_gpus:
                    raise RuntimeError(
                        f"Prefill envelope requested, but rho-envelope artifact has no "
                        f"prefill rows for gpu={gpu} ({rho_envelope_csv}).")

    # ---- loading -------------------------------------------------------------
    def _load_decode(self, path: Path) -> None:
        if not path.exists():
            raise RuntimeError(f"decode capacity artifact missing: {path}")
        required = {"gpu", "il", "ol", "tp", "freq_mhz", "slo_tpot_ms",
                    "C_dc_SLO_tps", "slo_confirmed", "freq_monotone_certified"}
        with open(path) as f:
            rd = csv.DictReader(f)
            if not required.issubset(rd.fieldnames or []):
                raise RuntimeError(
                    f"decode capacity schema mismatch: missing "
                    f"{required - set(rd.fieldnames or [])} in {path}")
            for r in rd:
                gpu = r["gpu"]
                il, ol, tp = int(r["il"]), int(r["ol"]), int(r["tp"])
                freq = int(r["freq_mhz"])
                slo = float(r["slo_tpot_ms"])
                row = {"C": float(r["C_dc_SLO_tps"]),
                       "confirmed": _as_bool(r["slo_confirmed"]),
                       "cert": _as_bool(r["freq_monotone_certified"])}
                self._dc[(gpu, il, ol, tp, freq, slo)] = row
                self._dc_by_freq.setdefault((gpu, il, ol, tp, slo), {})[freq] = row
                self._dc_slos.setdefault((gpu, il, ol, tp), set()).add(slo)
                self._dc_gpus.add(gpu)
        self._dc_slos = {k: sorted(v) for k, v in self._dc_slos.items()}

    def _load_prefill(self, path: Path) -> None:
        if not path.exists():
            raise RuntimeError(f"rho-envelope artifact missing: {path}")
        tmp: Dict[Tuple, List[Tuple[float, float]]] = {}
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get("phase") != "prefill":
                    continue
                gpu = r["gpu"]
                key = (gpu, int(r["tp"]), int(r["freq_mhz"]))
                tmp.setdefault(key, []).append((float(r["slo_ms"]), float(r["rho_max"])))
                self._pf_gpus.add(gpu)
        for key, lst in tmp.items():
            self._pf[key] = sorted(lst)

    # ---- prefill -------------------------------------------------------------
    def prefill_rho_max(self, gpu: str, tp: int, freq: int,
                        slo_ttft_ms: float) -> Optional[float]:
        """SLO-conditioned prefill utilization bound; None => reject (fail closed)."""
        entry = self._pf.get((gpu, tp, freq))
        if not entry:
            return None
        slos = [s for s, _ in entry]
        sel = _largest_le(slos, slo_ttft_ms)
        if sel is None:
            return None
        # entry is small; linear scan for the matching slo
        for s, rho in entry:
            if s == sel:
                return rho
        return None

    # ---- decode --------------------------------------------------------------
    def _decode_one_bucket(self, gpu: str, il: int, ol: int, tp: int, freq: int,
                           slo_tpot_ms: float) -> CapacityLookup:
        # available SLOs for this (gpu,il,ol,tp) (precomputed index):
        slos_here = self._dc_slos.get((gpu, il, ol, tp))
        sel = _largest_le(slos_here, slo_tpot_ms) if slos_here else None
        if sel is None:
            return CapacityLookup(0.0, "none", None, slo_tpot_ms, None)
        key = (gpu, il, ol, tp, freq, sel)
        row = self._dc.get(key)
        if row is not None and row["C"] > 0:
            src = "measured" if row["confirmed"] else "self_observed_lower_bound"
            return CapacityLookup(row["C"], src, key, slo_tpot_ms, sel)
        # borrow a same-TP lower-frequency confirmed bound (only if freq-monotone certified)
        freqs = self._dc_by_freq.get((gpu, il, ol, tp, sel), {})
        best = 0.0
        best_f = None
        for f_prof, r in freqs.items():
            if f_prof <= freq and r["confirmed"] and r["C"] > 0 and r["cert"]:
                if r["C"] > best:
                    best, best_f = r["C"], f_prof
        if best_f is not None:
            return CapacityLookup(best, "lower_freq_bound",
                                  (gpu, il, ol, tp, best_f, sel), slo_tpot_ms, sel)
        return CapacityLookup(0.0, "none", None, slo_tpot_ms, sel)

    def decode_safe_capacity(self, gpu: str, il: float, ol: float, tp: int,
                             freq: int, slo_tpot_ms: float) -> CapacityLookup:
        """Conservative goodput-capacity lookup. Non-grid (il,ol) -> neighborhood-min
        over bracketing grid buckets (never nearest-neighbor). Any bracket bucket that
        cannot be resolved (source="none") makes the whole lookup "none" (fail closed).
        """
        il_bkts = _brackets(il, IL_VALUES)
        ol_bkts = _brackets(ol, OL_VALUES)
        worst: Optional[CapacityLookup] = None
        for ib in il_bkts:
            for ob in ol_bkts:
                lk = self._decode_one_bucket(gpu, ib, ob, tp, freq, slo_tpot_ms)
                if lk.source == "none":
                    return CapacityLookup(0.0, "none", (gpu, ib, ob, tp, freq),
                                          slo_tpot_ms, lk.selected_slo_ms)
                if worst is None or lk.capacity_tps < worst.capacity_tps:
                    worst = lk
        assert worst is not None
        return worst
