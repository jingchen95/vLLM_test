#!/usr/bin/env python3
"""
Energy-efficient LLM inference scheduler (Phase A).

Given a workload (input_len, output_len, request_rate) and SLO threshold,
recommends the optimal (TP_degree, GPU_frequency) configuration that:
  1. Meets SLO constraint (via classifier + guard band)
  2. Minimizes total power (power_per_gpu × TP)

Usage:
  # Generate full lookup table
  python3 scheduler.py --mode table --output schedule_table_l40s.csv

  # Query a specific workload
  python3 scheduler.py --mode query --il 2048 --ol 32 --rate 10 --slo 500
"""

import joblib
import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from paths import PAPER_RESULTS_ANALYSES_DIR, paper_model_dir


class EnergyScheduler:
    """Energy-efficient scheduler for vLLM inference."""

    # The latency regressor is a monotonic gradient-boosted model on a log1p
    # target; predictions are made in log-space and passed through expm1.  When
    # queried far outside the calibration range (rho >> 1), clamp the raw
    # log-space output before expm1 so the scheduler sees a finite, clearly-
    # infeasible latency (≈3.3 × 10^6 ms) instead of IEEE 754 overflow artefacts
    # that corrupt energy-ranking comparisons.  The value 15.0 was chosen so that
    # expm1(15) ≈ 3.27 × 10^6 ms — four orders of magnitude above any SLO we
    # evaluate, but representable as a normal float64.
    _LOG_LATENCY_CAP: float = 15.0

    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir) if model_dir is not None else paper_model_dir("models_l40s")

        # Load config
        self.config = joblib.load(self.model_dir / "config.pkl")
        self.GAMMA = self.config["GAMMA"]
        self.ETA = self.config["ETA"]
        self.MAX_FREQ = self.config["MAX_FREQ"]
        self.GUARD_SETTINGS = self.config["GUARD_SETTINGS"]
        self.PHASE_GUARD_SETTINGS = self.config.get("PHASE_GUARD_SETTINGS", {})
        self.SLO_THRESHOLDS = self.config["SLO_THRESHOLDS"]
        self.TP_DEGREES = self.config["TP_DEGREES"]
        self.FREQUENCIES = self.config["FREQUENCIES"]

        # Load capacity model
        self.capacity_model = joblib.load(self.model_dir / "capacity_model.pkl")
        self.cap_features = joblib.load(self.model_dir / "capacity_features.pkl")
        self.cap_groups = joblib.load(self.model_dir / "cap_groups.pkl")

        # Build plateau lookup from cap_groups
        self._plateau_map = {}
        for _, row in self.cap_groups.iterrows():
            key = (int(row["il"]), int(row["ol"]), int(row["tp"]), int(row["freq"]))
            self._plateau_map[key] = bool(row["plateau_confirmed"])

        # Load classifiers (one per SLO)
        self.classifiers = {}
        self.clf_scalers = {}
        for slo in self.SLO_THRESHOLDS:
            self.classifiers[slo] = joblib.load(self.model_dir / f"slo_classifier_{slo}.pkl")
            self.clf_scalers[slo] = joblib.load(self.model_dir / f"slo_scaler_{slo}.pkl")
        self.clf_features = joblib.load(self.model_dir / "clf_features.pkl")

        # Load regressor
        self.regressor = joblib.load(self.model_dir / "regressor_p99.pkl")
        self.reg_scaler = joblib.load(self.model_dir / "regressor_scaler.pkl")
        self.reg_poly = joblib.load(self.model_dir / "regressor_poly.pkl")
        self.reg_features = joblib.load(self.model_dir / "reg_features.pkl")

        # Load power model
        self.power_model = joblib.load(self.model_dir / "power_model.pkl")
        self.power_features = joblib.load(self.model_dir / "power_features.pkl")

        self.phase_targets = self.config.get("PHASE_MODEL_TARGETS", {})
        self.phase_models = self._load_phase_models()
        self.has_phase_models = len(self.phase_models) == 2

        print(f"Loaded models from {model_dir}")
        print(f"  Config space: TP={self.TP_DEGREES}, "
              f"Freq=[{self.FREQUENCIES[0]}..{self.FREQUENCIES[-1]}] ({len(self.FREQUENCIES)} levels)")
        print(f"  SLO thresholds: {self.SLO_THRESHOLDS}")
        if self.has_phase_models:
            print("  Phase-specific stack: prefill + decode")
        else:
            print("  Phase-specific stack: unavailable, using legacy unified path")

    def _load_phase_models(self) -> Dict[str, dict]:
        phase_models: Dict[str, dict] = {}
        for phase in ("prefill", "decode"):
            feature_path = self.model_dir / f"{phase}_clf_features.pkl"
            reg_feature_path = self.model_dir / f"{phase}_reg_features.pkl"
            power_feature_path = self.model_dir / f"{phase}_power_features.pkl"
            latency_model_path = self.model_dir / f"{phase}_latency_regressor_p99.pkl"
            latency_scaler_path = self.model_dir / f"{phase}_latency_scaler.pkl"
            latency_poly_path = self.model_dir / f"{phase}_latency_poly.pkl"
            power_model_path = self.model_dir / f"{phase}_power_model.pkl"
            if not all(path.exists() for path in (
                feature_path,
                reg_feature_path,
                power_feature_path,
                latency_model_path,
                latency_scaler_path,
                latency_poly_path,
                power_model_path,
            )):
                continue

            phase_models[phase] = {
                "clf_features": joblib.load(feature_path),
                "reg_features": joblib.load(reg_feature_path),
                "power_features": joblib.load(power_feature_path),
                "latency_model": joblib.load(latency_model_path),
                "latency_scaler": joblib.load(latency_scaler_path),
                "latency_poly": joblib.load(latency_poly_path),
                "power_model": joblib.load(power_model_path),
                "classifiers": {},
                "scalers": {},
            }
            for slo in self.SLO_THRESHOLDS:
                clf_path = self.model_dir / f"{phase}_slo_classifier_{slo}.pkl"
                scaler_path = self.model_dir / f"{phase}_slo_scaler_{slo}.pkl"
                if not clf_path.exists() or not scaler_path.exists():
                    phase_models.pop(phase, None)
                    break
                phase_models[phase]["classifiers"][slo] = joblib.load(clf_path)
                phase_models[phase]["scalers"][slo] = joblib.load(scaler_path)
        return phase_models

    def _phase_guard_setting(self, phase: str, slo: int) -> dict:
        phase_cfg = self.PHASE_GUARD_SETTINGS.get(phase, {})
        return phase_cfg.get(slo, phase_cfg.get(str(slo), self.GUARD_SETTINGS[slo]))

    def _compute_features(self, il, ol, tp, freq, rate):
        """Compute all features for a single configuration."""
        log_il = np.log1p(il)
        log_ol = np.log1p(ol)
        freq_norm = freq / self.MAX_FREQ
        decode_frac = ol / (il + ol)
        log_rate = np.log1p(rate)

        # Predict capacity
        X_cap = pd.DataFrame(
            [[log_il, log_ol, tp, freq_norm, decode_frac]],
            columns=self.cap_features,
        )
        C_hat_raw = np.exp(self.capacity_model.predict(X_cap)[0])

        # Apply uncertainty margin if not plateau-confirmed
        is_plateau = self._plateau_map.get((il, ol, tp, freq), False)
        C_hat = C_hat_raw if is_plateau else self.ETA * C_hat_raw

        # Demand and utilization
        D_prefill = rate * il
        D_decode = rate * ol
        demand = D_prefill + self.GAMMA * D_decode
        rho = demand / C_hat
        rho_prefill = D_prefill / C_hat
        rho_decode = (self.GAMMA * D_decode) / C_hat
        prefill_frac = D_prefill / (D_prefill + self.GAMMA * D_decode)

        return {
            "log_il": log_il, "log_ol": log_ol, "tp": tp,
            "freq_norm": freq_norm, "decode_frac": decode_frac,
            "log_rate": log_rate,
            "log_rho": np.log1p(rho),
            "log_rho_total": np.log1p(rho),
            "rho_sq": rho ** 2,
            "rho_overflow": max(0, rho - 1.0),
            "rho_total_overflow": max(0, rho - 1.0),
            "log_d_prefill": np.log1p(D_prefill),
            "log_d_decode": np.log1p(D_decode),
            "log_kv_pressure": np.log1p(rate * il * ol),
            "prefill_frac": prefill_frac,
            "rho_prefill": rho_prefill,
            "rho_decode": rho_decode,
            "log_rho_prefill": np.log1p(rho_prefill),
            "log_rho_decode": np.log1p(rho_decode),
            "rho_prefill_sq": rho_prefill ** 2,
            "rho_decode_sq": rho_decode ** 2,
            "rho_prefill_overflow": max(0, rho_prefill - 1.0),
            "rho_decode_overflow": max(0, rho_decode - 1.0),
            # Extra (not features, but needed for decisions)
            "C_hat": C_hat, "rho": rho, "is_plateau": is_plateau,
        }

    def _predict_legacy_config(self, il, ol, tp, freq, rate, slo):
        """Legacy unified predictor kept for backward compatibility."""
        if slo not in self.SLO_THRESHOLDS:
            slo = min(self.SLO_THRESHOLDS, key=lambda supported: abs(supported - slo))

        feats = self._compute_features(il, ol, tp, freq, rate)

        # Classifier
        X_clf = np.array([[feats[f] for f in self.clf_features]])
        X_clf_scaled = self.clf_scalers[slo].transform(X_clf)
        p_violate = self.classifiers[slo].predict_proba(X_clf_scaled)[0, 1]

        # Guard band
        gs = self.GUARD_SETTINGS[slo]
        is_safe = (p_violate < gs["p_th"]) and (feats["rho"] < gs["rho_th"])

        # Regression (predict P99 TTFT regardless, for ranking)
        X_reg = np.array([[feats[f] for f in self.reg_features]])
        X_reg_poly = self.reg_poly.transform(self.reg_scaler.transform(X_reg))
        p99_pred = float(np.expm1(min(float(self.regressor.predict(X_reg_poly)[0]), self._LOG_LATENCY_CAP)))

        # Power
        X_pow = pd.DataFrame(
            [[feats[f] for f in self.power_features]],
            columns=self.power_features,
        )
        power_per_gpu = float(self.power_model.predict(X_pow)[0])
        total_power = power_per_gpu * tp

        return {
            "tp": tp, "freq_mhz": freq,
            "is_safe": is_safe,
            "p_violate": round(p_violate, 4),
            "rho": round(feats["rho"], 4),
            "p99_ttft_ms": round(p99_pred, 1),
            "power_per_gpu_w": round(power_per_gpu, 1),
            "total_power_w": round(total_power, 1),
        }

    def _predict_phase_config(self, phase: str, il: int, ol: int,
                              tp: int, freq: int, rate: float, slo: int) -> dict:
        if phase not in self.phase_models:
            legacy = self._predict_legacy_config(il, ol, tp, freq, rate, slo)
            if phase == "decode":
                legacy["p99_tpot_ms"] = legacy.get("p99_ttft_ms")
                legacy["latency_ms"] = legacy.get("p99_tpot_ms")
            else:
                legacy["latency_ms"] = legacy.get("p99_ttft_ms")
            legacy["phase"] = phase
            legacy["rho_phase"] = round(legacy["rho"], 4)
            legacy["rho_total"] = round(legacy["rho"], 4)
            return legacy

        if slo not in self.SLO_THRESHOLDS:
            slo = min(self.SLO_THRESHOLDS, key=lambda supported: abs(supported - slo))

        feats = self._compute_features(il, ol, tp, freq, rate)
        spec = self.phase_models[phase]
        target_info = self.phase_targets.get(phase, {})
        rho_col = target_info.get("rho_col", "rho_prefill" if phase == "prefill" else "rho_decode")
        latency_key = "p99_ttft_ms" if phase == "prefill" else "p99_tpot_ms"

        X_clf = np.array([[feats[f] for f in spec["clf_features"]]])
        X_clf_scaled = spec["scalers"][slo].transform(X_clf)
        clf = spec["classifiers"][slo]
        proba = clf.predict_proba(X_clf_scaled)[0]
        if len(proba) == 1:
            only_class = int(clf.classes_[0])
            p_violate = 1.0 if only_class == 1 else 0.0
        else:
            violate_idx = int(np.where(clf.classes_ == 1)[0][0])
            p_violate = float(proba[violate_idx])

        gs = self._phase_guard_setting(phase, slo)
        rho_phase = feats[rho_col]
        is_safe = (p_violate < gs["p_th"]) and (rho_phase < gs["rho_th"])

        X_reg = np.array([[feats[f] for f in spec["reg_features"]]])
        X_reg_poly = spec["latency_poly"].transform(spec["latency_scaler"].transform(X_reg))
        latency_pred = float(np.expm1(min(float(spec["latency_model"].predict(X_reg_poly)[0]), self._LOG_LATENCY_CAP)))

        X_pow = pd.DataFrame(
            [[feats[f] for f in spec["power_features"]]],
            columns=spec["power_features"],
        )
        power_per_gpu = float(spec["power_model"].predict(X_pow)[0])
        total_power = power_per_gpu * tp

        result = {
            "phase": phase,
            "tp": tp,
            "freq_mhz": freq,
            "is_safe": is_safe,
            "p_violate": round(p_violate, 4),
            "rho": round(feats["rho"], 4),
            "rho_phase": round(rho_phase, 4),
            "rho_total": round(feats["rho"], 4),
            "power_per_gpu_w": round(power_per_gpu, 1),
            "total_power_w": round(total_power, 1),
            "latency_ms": round(latency_pred, 1),
            latency_key: round(latency_pred, 1),
        }
        if phase == "prefill":
            result["p99_tpot_ms"] = None
        else:
            result["p99_ttft_ms"] = None
        return result

    def predict_prefill_config(self, il: int, tp: int, freq: int, rate: float, slo: int,
                               proxy_ol: int = 32) -> dict:
        return self._predict_phase_config("prefill", il, proxy_ol, tp, freq, rate, slo)

    def predict_decode_config(self, ol: int, tp: int, freq: int, rate: float, slo: int,
                              proxy_il: int = 32) -> dict:
        return self._predict_phase_config("decode", proxy_il, ol, tp, freq, rate, slo)

    def predict_config(self, il, ol, tp, freq, rate, slo, phase: Optional[str] = None):
        """Predict safety, latency, and power for one configuration.

        When ``phase`` is omitted, this preserves the legacy unified / TTFT-centric
        behavior for backward compatibility. Phase-aware scheduler paths should call
        ``phase='prefill'`` or ``phase='decode'`` explicitly.
        """
        if phase == "prefill":
            return self.predict_prefill_config(il, tp, freq, rate, slo, proxy_ol=ol)
        if phase == "decode":
            return self.predict_decode_config(ol, tp, freq, rate, slo, proxy_il=il)
        return self._predict_legacy_config(il, ol, tp, freq, rate, slo)

    def recommend(self, il, ol, rate, slo):
        """Recommend optimal config: enumerate all, filter safe, pick lowest power."""
        candidates = []
        for tp in self.TP_DEGREES:
            for freq in self.FREQUENCIES:
                pred = self.predict_config(il, ol, tp, freq, rate, slo)
                candidates.append(pred)

        safe = [c for c in candidates if c["is_safe"]]

        if not safe:
            return {
                "workload": {"il": il, "ol": ol, "rate": rate},
                "slo_ms": slo,
                "status": "NO_SAFE_CONFIG",
                "num_candidates": len(candidates),
                "num_safe": 0,
            }

        # Sort by total power (energy minimization)
        safe.sort(key=lambda c: c["total_power_w"])
        best = safe[0]

        # Baseline: max freq, minimum TP that's safe
        baseline_safe = [c for c in safe if c["freq_mhz"] == self.MAX_FREQ]
        if baseline_safe:
            baseline_safe.sort(key=lambda c: c["tp"])
            baseline = baseline_safe[0]
        else:
            # No safe config at max freq — use the highest-freq safe config
            baseline = max(safe, key=lambda c: c["freq_mhz"])

        saving_pct = ((baseline["total_power_w"] - best["total_power_w"])
                      / baseline["total_power_w"] * 100)

        return {
            "workload": {"il": il, "ol": ol, "rate": rate},
            "slo_ms": slo,
            "status": "OK",
            "recommended": best,
            "baseline": {
                "tp": baseline["tp"], "freq_mhz": baseline["freq_mhz"],
                "total_power_w": baseline["total_power_w"],
            },
            "energy_saving_pct": round(saving_pct, 1),
            "num_safe": len(safe),
            "alternatives": safe[1:4],  # top 3 alternatives
        }

    def generate_lookup_table(self, output_file="schedule_table.csv",
                              input_lens=None, output_lens=None, rates=None):
        """Generate full lookup table for all workloads × SLOs."""
        if input_lens is None:
            input_lens = [32, 128, 512, 1024, 2048]
        if output_lens is None:
            output_lens = [32, 128, 512, 1024]
        if rates is None:
            rates = [1, 10, 50]

        rows = []
        total = len(input_lens) * len(output_lens) * len(rates) * len(self.SLO_THRESHOLDS)
        count = 0

        for il in input_lens:
            for ol in output_lens:
                for rate in rates:
                    for slo in self.SLO_THRESHOLDS:
                        count += 1
                        rec = self.recommend(il, ol, rate, slo)

                        row = {
                            "input_len": il, "output_len": ol,
                            "request_rate": rate, "slo_ms": slo,
                        }

                        if rec["status"] == "OK":
                            b = rec["recommended"]
                            row.update({
                                "rec_tp": b["tp"],
                                "rec_freq_mhz": b["freq_mhz"],
                                "pred_p99_ttft_ms": b["p99_ttft_ms"],
                                "pred_power_per_gpu_w": b["power_per_gpu_w"],
                                "pred_total_power_w": b["total_power_w"],
                                "pred_rho": b["rho"],
                                "baseline_tp": rec["baseline"]["tp"],
                                "baseline_freq_mhz": rec["baseline"]["freq_mhz"],
                                "baseline_power_w": rec["baseline"]["total_power_w"],
                                "energy_saving_pct": rec["energy_saving_pct"],
                                "num_safe": rec["num_safe"],
                                "status": "OK",
                            })
                        else:
                            row["status"] = "NO_SAFE_CONFIG"

                        rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)

        n_ok = (df["status"] == "OK").sum()
        n_fail = (df["status"] != "OK").sum()
        print(f"\nLookup table: {output_file}")
        print(f"  Total entries: {len(df)}")
        print(f"  Safe: {n_ok} ({n_ok/len(df)*100:.0f}%)")
        print(f"  No safe config: {n_fail} ({n_fail/len(df)*100:.0f}%)")

        if n_ok > 0:
            ok = df[df["status"] == "OK"]
            print(f"\n  Energy savings summary:")
            print(f"    Mean: {ok['energy_saving_pct'].mean():.1f}%")
            print(f"    Median: {ok['energy_saving_pct'].median():.1f}%")
            print(f"    Max: {ok['energy_saving_pct'].max():.1f}%")

            print(f"\n  Recommended configs distribution:")
            for tp in sorted(ok["rec_tp"].unique()):
                n = (ok["rec_tp"] == tp).sum()
                print(f"    TP={int(tp)}: {n} ({n/len(ok)*100:.0f}%)")

            print(f"\n  Top 10 energy-saving workloads:")
            top = ok.nlargest(10, "energy_saving_pct")
            print(f"  {'il':>5s} {'ol':>5s} {'rate':>4s} {'SLO':>5s} | "
                  f"{'TP':>2s} {'freq':>5s} {'save%':>5s} | baseline")
            for _, r in top.iterrows():
                print(f"  {int(r['input_len']):>5d} {int(r['output_len']):>5d} "
                      f"{int(r['request_rate']):>4d} {int(r['slo_ms']):>5d} | "
                      f"{int(r['rec_tp']):>2d} {int(r['rec_freq_mhz']):>5d} "
                      f"{r['energy_saving_pct']:>5.1f}% | "
                      f"TP={int(r['baseline_tp'])} f={int(r['baseline_freq_mhz'])}")

        return df


def main():
    parser = argparse.ArgumentParser(description="Energy-efficient LLM scheduler")
    parser.add_argument("--mode", choices=["table", "query"], required=True)
    parser.add_argument("--model-dir", default=str(paper_model_dir("models_l40s")))
    parser.add_argument("--output", default=str(PAPER_RESULTS_ANALYSES_DIR / "schedule_table_l40s.csv"))
    parser.add_argument("--il", type=int)
    parser.add_argument("--ol", type=int)
    parser.add_argument("--rate", type=int)
    parser.add_argument("--slo", type=int)

    args = parser.parse_args()
    scheduler = EnergyScheduler(model_dir=args.model_dir)

    if args.mode == "table":
        scheduler.generate_lookup_table(output_file=args.output)

    else:
        if not all([args.il, args.ol, args.rate, args.slo]):
            parser.error("Query mode requires --il, --ol, --rate, --slo")

        rec = scheduler.recommend(args.il, args.ol, args.rate, args.slo)

        # Convert numpy types for JSON serialization
        def to_native(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_native(v) for v in obj]
            return obj

        print(json.dumps(to_native(rec), indent=2))


if __name__ == "__main__":
    main()
