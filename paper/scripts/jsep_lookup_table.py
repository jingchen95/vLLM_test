"""
jsep_lookup_table.py — Lookup table backed by b5_lookup_table.csv.

Provides fast nearest-neighbour queries over the B5 measurement grid
for use in simulation (jsep_experiment_f.py and jsep_simulator.py).

Usage:
    from jsep_lookup_table import LookupTable
    lut = LookupTable("b5_lookup_table.csv")
    p99 = lut.query("l40s", 2520, 1, 512, 128, 10.0, "p99_ttft_ms")
    cfg = lut.best_config_for_slo("l40s", 512, 128, 10.0, 300, 50)
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Metrics that can be queried
QUERYABLE_METRICS = [
    "p99_ttft_ms", "mean_ttft_ms",
    "p99_tpot_ms", "mean_tpot_ms",
    "throughput_tps", "avg_power_w",
    "benchmark_duration_s", "successful_requests",
]

# Continuous dimensions used for nearest-neighbour matching
_CONTINUOUS_DIMS = ["freq_mhz", "input_len", "output_len", "rate_rps"]

# Scales for distance calculation (roughly normalise each dimension)
_DIM_SCALES = {
    "freq_mhz":   1000.0,
    "input_len":  512.0,
    "output_len": 512.0,
    "rate_rps":   10.0,
}


class LookupTable:
    """
    Lookup table backed by a b5_lookup_table.csv produced by
    parse_disagg_results.py.

    Nearest-neighbour query strategy:
      - gpu_type: exact match (string)
      - tp:       exact match (integer)
      - exp_type: exact match if provided, else best available
      - freq_mhz, input_len, output_len, rate_rps: L2 nearest neighbour
        in a normalised feature space (see _DIM_SCALES)

    If scipy is available, uses a KDTree for O(log n) queries;
    otherwise falls back to brute-force O(n).
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._df: pd.DataFrame = pd.DataFrame()
        self._kdtrees: Dict[Tuple, object] = {}   # (gpu_type, tp, exp_type) → KDTree
        self._kdtree_rows: Dict[Tuple, np.ndarray] = {}  # matching row indices

        self._load(csv_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, csv_path: str) -> None:
        if not os.path.isfile(csv_path):
            warnings.warn(
                f"LookupTable: CSV not found: {csv_path}. "
                "Queries will use synthetic fallback values."
            )
            return

        df = pd.read_csv(csv_path)

        # Coerce numeric columns
        numeric_cols = _CONTINUOUS_DIMS + QUERYABLE_METRICS
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows where all queryable metrics are NaN
        df = df.dropna(subset=[m for m in QUERYABLE_METRICS if m in df.columns],
                        how="all")

        # Drop failed experiments: successful_requests == 0 means the benchmark
        # failed and all metrics (TTFT, TPOT, throughput) are 0, not real.
        if "successful_requests" in df.columns:
            df = df[df["successful_requests"] > 0]

        self._df = df.reset_index(drop=True)

        # Pre-build KDTrees per (gpu_type, tp, exp_type) group
        self._build_kdtrees()

    def _build_kdtrees(self) -> None:
        """Pre-build KDTrees for each discrete (gpu_type, tp, exp_type) group."""
        try:
            from scipy.spatial import KDTree as _KDTree
            self._KDTree = _KDTree
            self._use_scipy = True
        except ImportError:
            self._use_scipy = False

        df = self._df
        if df.empty:
            return

        for key, grp in df.groupby(["gpu_type", "tp", "exp_type"]):
            # Build normalised feature matrix
            pts = self._normalise_points(grp)
            if pts is None or len(pts) == 0:
                continue
            row_indices = grp.index.to_numpy()
            if self._use_scipy:
                self._kdtrees[key] = self._KDTree(pts)
            else:
                self._kdtrees[key] = pts          # store raw array for brute-force
            self._kdtree_rows[key] = row_indices

    @staticmethod
    def _normalise_points(grp: pd.DataFrame) -> Optional[np.ndarray]:
        """Return (N, 4) array of normalised continuous dimensions."""
        pts = []
        for dim in _CONTINUOUS_DIMS:
            if dim not in grp.columns:
                return None
            col = pd.to_numeric(grp[dim], errors="coerce").fillna(0.0)
            pts.append(col.to_numpy() / _DIM_SCALES[dim])
        return np.column_stack(pts)

    # ------------------------------------------------------------------
    # Core nearest-neighbour lookup
    # ------------------------------------------------------------------

    def _nn_query(self, gpu_type: str, tp: int, exp_type: str,
                  freq_mhz: float, input_len: float,
                  output_len: float, rate_rps: float) -> Optional[pd.Series]:
        """
        Find the nearest-neighbour row for given discrete + continuous dims.

        Falls back gracefully:
          1. Exact (gpu_type, tp, exp_type) match.
          2. If not found, ignore exp_type and use first matching (gpu_type, tp).
          3. If still not found, ignore tp and use any (gpu_type,) match.
        """
        key = (gpu_type, tp, exp_type)
        row = self._nn_in_group(key, freq_mhz, input_len, output_len, rate_rps)
        if row is not None:
            return row

        # Fallback 1: any exp_type for this (gpu_type, tp)
        for k in self._kdtrees:
            if k[0] == gpu_type and k[1] == tp:
                row = self._nn_in_group(k, freq_mhz, input_len, output_len, rate_rps)
                if row is not None:
                    return row

        # Fallback 2: any tp for this gpu_type
        for k in self._kdtrees:
            if k[0] == gpu_type:
                row = self._nn_in_group(k, freq_mhz, input_len, output_len, rate_rps)
                if row is not None:
                    return row

        return None

    def _nn_in_group(self, key: Tuple, freq_mhz: float, input_len: float,
                     output_len: float, rate_rps: float) -> Optional[pd.Series]:
        """Nearest-neighbour inside a pre-built group."""
        if key not in self._kdtrees:
            return None

        q = np.array([
            freq_mhz  / _DIM_SCALES["freq_mhz"],
            input_len  / _DIM_SCALES["input_len"],
            output_len / _DIM_SCALES["output_len"],
            rate_rps   / _DIM_SCALES["rate_rps"],
        ])

        row_indices = self._kdtree_rows[key]

        if self._use_scipy:
            tree = self._kdtrees[key]
            _, idx = tree.query(q)
        else:
            # Brute-force
            pts = self._kdtrees[key]  # stored as raw array
            dists = np.linalg.norm(pts - q, axis=1)
            idx = int(np.argmin(dists))

        df_idx = row_indices[idx]
        return self._df.loc[df_idx]

    # ------------------------------------------------------------------
    # Synthetic fallback (when no real data is available)
    # ------------------------------------------------------------------

    @staticmethod
    def _synthetic_fallback(gpu_type: str, freq_mhz: int, tp: int,
                             input_len: int, output_len: int,
                             rate_rps: float, metric: str) -> float:
        """
        Rough analytical approximation used when lookup table is empty.

        These are order-of-magnitude estimates based on:
          - L40S: ~25 ms prefill TTFT at 2520 MHz for 512-token prompts
          - L4:   ~150 ms prefill TTFT at 2040 MHz for 512-token prompts
          - TPOT scales roughly with output_len / throughput
          - Power from GPU TDP: L40S ~350 W, L4 ~72 W
        """
        warnings.warn(
            f"LookupTable: using synthetic fallback for "
            f"{gpu_type} f={freq_mhz} tp={tp} il={input_len} ol={output_len} "
            f"rate={rate_rps:.1f} metric={metric}"
        )

        # Frequency scaling factor (linear approximation)
        max_freq = {"l40s": 2520, "l4": 2040}.get(gpu_type, 2000)
        freq_scale = freq_mhz / max_freq

        if metric in ("p99_ttft_ms", "mean_ttft_ms"):
            # Base TTFT at max freq: L40S ~25 ms, L4 ~625 ms for il=512
            base = 25.0 if gpu_type == "l40s" else 625.0
            val = base * (input_len / 512.0) / freq_scale
            return val * (1.2 if "p99" in metric else 1.0)

        if metric in ("p99_tpot_ms", "mean_tpot_ms"):
            # Base TPOT: L40S ~12 ms, L4 ~25 ms
            base = 12.0 if gpu_type == "l40s" else 25.0
            val = base / freq_scale
            return val * (1.5 if "p99" in metric else 1.0)

        if metric == "throughput_tps":
            # Rough peak TPS at max freq, TP=1
            base = 2000.0 if gpu_type == "l40s" else 500.0
            return base * freq_scale * tp

        if metric == "avg_power_w":
            # Fraction of TDP
            tdp = 350.0 if gpu_type == "l40s" else 72.0
            return tdp * (0.6 + 0.4 * freq_scale) * tp

        return float("nan")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, gpu_type: str, freq_mhz: int, tp: int,
              input_len: int, output_len: int, rate_rps: float,
              metric: str,
              exp_type: Optional[str] = None) -> float:
        """
        Return the value of `metric` for the nearest-matching config.

        Args:
            gpu_type:   "l40s" or "l4"
            freq_mhz:   GPU SM frequency (MHz)
            tp:         tensor-parallelism degree
            input_len:  prompt length (tokens)
            output_len: generation length (tokens)
            rate_rps:   request rate (req/s)
            metric:     one of QUERYABLE_METRICS
            exp_type:   if set, restrict to "A", "B", "C", or "D"
        Returns:
            float value, or NaN if no row found and no fallback available
        """
        if metric not in QUERYABLE_METRICS:
            raise ValueError(f"Unknown metric: {metric}. Choose from {QUERYABLE_METRICS}")

        if exp_type is None:
            # Default: use monolithic exp for combined TTFT+TPOT,
            # i.e., A for l40s, B for l4
            exp_type = "A" if gpu_type == "l40s" else "B"

        row = self._nn_query(gpu_type, tp, exp_type,
                              float(freq_mhz), float(input_len),
                              float(output_len), float(rate_rps))

        if row is None or pd.isna(row.get(metric, float("nan"))):
            return self._synthetic_fallback(gpu_type, freq_mhz, tp,
                                            input_len, output_len, rate_rps,
                                            metric)

        return float(row[metric])

    def get_valid_configs(self, gpu_type: str) -> List[Tuple[int, int]]:
        """
        Return list of (freq_mhz, tp) tuples present in the data for gpu_type.

        Includes all experiment types (A–D) for completeness.
        """
        if self._df.empty:
            return []
        sub = self._df[self._df["gpu_type"] == gpu_type]
        if sub.empty:
            return []
        pairs = (sub[["freq_mhz", "tp"]]
                 .drop_duplicates()
                 .dropna()
                 .astype(int)
                 .sort_values(["freq_mhz", "tp"]))
        return list(pairs.itertuples(index=False, name=None))

    def get_valid_freqs(self, gpu_type: str,
                         exp_type: Optional[str] = None) -> List[int]:
        """Return sorted unique frequencies present for gpu_type (and optionally exp_type)."""
        if self._df.empty:
            return []
        mask = self._df["gpu_type"] == gpu_type
        if exp_type is not None:
            mask = mask & (self._df["exp_type"] == exp_type)
        vals = self._df.loc[mask, "freq_mhz"].dropna().astype(int).unique()
        return sorted(vals.tolist())

    def get_valid_tps(self, gpu_type: str,
                       exp_type: Optional[str] = None) -> List[int]:
        """Return sorted unique TP degrees present for gpu_type (and optionally exp_type)."""
        if self._df.empty:
            return []
        mask = self._df["gpu_type"] == gpu_type
        if exp_type is not None:
            mask = mask & (self._df["exp_type"] == exp_type)
        vals = self._df.loc[mask, "tp"].dropna().astype(int).unique()
        return sorted(vals.tolist())

    def get_power(self, gpu_type: str, freq_mhz: int, tp: int) -> float:
        """
        Return average power (W) for a (gpu_type, freq_mhz, tp) config,
        averaged across all workloads in the lookup table for that config.

        Falls back to synthetic estimate if data is unavailable.
        """
        if self._df.empty:
            return self._synthetic_fallback(gpu_type, freq_mhz, tp,
                                            512, 128, 10.0, "avg_power_w")

        mask = (
            (self._df["gpu_type"] == gpu_type) &
            (self._df["freq_mhz"].astype(int) == int(freq_mhz)) &
            (self._df["tp"].astype(int) == int(tp))
        )
        sub = self._df[mask]
        if sub.empty or sub["avg_power_w"].isna().all():
            return self._synthetic_fallback(gpu_type, freq_mhz, tp,
                                            512, 128, 10.0, "avg_power_w")
        return float(sub["avg_power_w"].mean())

    def best_config_for_slo(
        self,
        gpu_type: str,
        il: int,
        ol: int,
        rate: float,
        slo_ttft_ms: float,
        slo_tpot_ms: float,
        exp_type: Optional[str] = None,
    ) -> Optional[Tuple[int, int, float]]:
        """
        Find the (freq_mhz, tp, power_w) config that meets the given SLO
        with minimum power.

        SLO is met when:
          p99_ttft_ms <= slo_ttft_ms  AND  p99_tpot_ms <= slo_tpot_ms

        Returns (freq_mhz, tp, avg_power_w) or None if no config meets SLO.
        """
        configs = self.get_valid_configs(gpu_type)
        if not configs:
            # Synthetic fallback: pick max freq
            max_freq = {"l40s": 2520, "l4": 2040}.get(gpu_type, 2040)
            power = self._synthetic_fallback(gpu_type, max_freq, 1,
                                             il, ol, rate, "avg_power_w")
            return (max_freq, 1, power)

        best = None
        best_power = float("inf")

        for freq_mhz, tp in configs:
            ttft = self.query(gpu_type, freq_mhz, tp, il, ol, rate,
                              "p99_ttft_ms", exp_type=exp_type)
            tpot = self.query(gpu_type, freq_mhz, tp, il, ol, rate,
                              "p99_tpot_ms", exp_type=exp_type)
            if ttft <= slo_ttft_ms and tpot <= slo_tpot_ms:
                power = self.get_power(gpu_type, freq_mhz, tp)
                if power < best_power:
                    best_power = power
                    best = (int(freq_mhz), int(tp), power)

        return best

    def energy_per_token_j(
        self,
        gpu_type: str,
        freq_mhz: int,
        tp: int,
        il: int,
        ol: int,
        rate: float,
        exp_type: Optional[str] = None,
    ) -> float:
        """
        Compute energy per output token (J/tok) = avg_power_w / throughput_tps.

        Returns NaN if throughput is zero or unavailable.
        """
        power = self.query(gpu_type, freq_mhz, tp, il, ol, rate,
                           "avg_power_w", exp_type=exp_type)
        tps = self.query(gpu_type, freq_mhz, tp, il, ol, rate,
                         "throughput_tps", exp_type=exp_type)
        if tps == 0 or np.isnan(tps) or np.isnan(power):
            return float("nan")
        return power / tps

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"LookupTable(csv={self.csv_path!r}, "
                f"rows={len(self._df)}, "
                f"groups={len(self._kdtrees)})")


# ---------------------------------------------------------------------------
# Standalone test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test-drive the LookupTable class."
    )
    parser.add_argument("csv", nargs="?", default="b5_lookup_table.csv",
                        help="Path to b5_lookup_table.csv")
    args = parser.parse_args()

    lut = LookupTable(args.csv)
    print(lut)

    print("\nValid L40S configs:", lut.get_valid_configs("l40s")[:5], "...")
    print("Valid L4 configs:",   lut.get_valid_configs("l4")[:5],   "...")

    # Sample query
    for gpu_type, freq, tp in [("l40s", 2520, 1), ("l4", 2040, 1)]:
        ttft  = lut.query(gpu_type, freq, tp, 512, 128, 10.0, "p99_ttft_ms")
        tpot  = lut.query(gpu_type, freq, tp, 512, 128, 10.0, "p99_tpot_ms")
        power = lut.get_power(gpu_type, freq, tp)
        ept   = lut.energy_per_token_j(gpu_type, freq, tp, 512, 128, 10.0)
        print(f"\n{gpu_type.upper()} f={freq} tp={tp}: "
              f"P99-TTFT={ttft:.1f}ms  P99-TPOT={tpot:.1f}ms  "
              f"Power={power:.1f}W  EnergyPerToken={ept:.4f}J")

    # best_config_for_slo
    for gpu_type in ["l40s", "l4"]:
        cfg = lut.best_config_for_slo(gpu_type, 512, 128, 10.0, 300.0, 50.0)
        print(f"\nbest_config_for_slo({gpu_type}, SLO TTFT=300ms, TPOT=50ms): {cfg}")
