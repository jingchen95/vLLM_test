"""
JSEP Phase B: Scheduling strategies for heterogeneous GPU cluster.

Five strategies:
1. NaiveBaseline     — all GPUs max freq, TP=1, round-robin
2. DVFSOnlyBaseline  — Phase A DVFS per GPU type, 50/50 split
3. RoutingOnlyBaseline — route to cheapest GPU type, max freq
4. DynamoLLMBaseline — decoupled: pick GPU type first, then DVFS
5. JSEPStrategy      — joint phase-aware HP/LP optimization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from scheduler import EnergyScheduler
from jsep_cluster import ClusterState, GPUPool, GPU_SPECS
from typing import Dict, Optional
import numpy as np

from paths import paper_model_dir


def _find_min_safe_freq(scheduler: EnergyScheduler, il, ol, tp, rate, slo):
    """Find minimum safe frequency for a config. Returns (freq, pred) or None."""
    for freq in scheduler.FREQUENCIES:
        pred = scheduler.predict_config(il, ol, tp, freq, rate, slo)
        if pred["is_safe"]:
            return freq, pred
    return None, None


class SchedulingResult:
    """Standard result from any scheduling strategy."""
    def __init__(self, total_power_w, config, slo_met=True, phase="HP"):
        self.total_power_w = total_power_w
        self.config = config      # dict with per-pool details
        self.slo_met = slo_met
        self.phase = phase

    def __repr__(self):
        return f"Result(power={self.total_power_w:.1f}W, phase={self.phase})"


# ---------------------------------------------------------------------------
# Strategy 1: Naive Baseline
# ---------------------------------------------------------------------------
class NaiveBaseline:
    """All GPUs at max freq, TP=1, round-robin 50/50 split."""

    def __init__(self, schedulers: Dict[str, EnergyScheduler]):
        self.schedulers = schedulers

    def decide(self, il, ol, rate, slo, cluster: ClusterState) -> SchedulingResult:
        total_power = 0.0
        config = {}
        slo_met = True

        for gpu_type, sched in self.schedulers.items():
            spec = GPU_SPECS[gpu_type]
            tp = 1
            freq = spec["max_freq_mhz"]
            n_inst = spec["total_gpus"]  # all active at TP=1
            per_inst_rate = (rate * 0.5) / n_inst  # 50/50 split

            pred = sched.predict_config(il, ol, tp, freq, per_inst_rate, slo)
            power_per_inst = pred["total_power_w"]  # = power_per_gpu * tp
            pool_power = n_inst * power_per_inst
            total_power += pool_power

            if not pred["is_safe"]:
                slo_met = False

            config[gpu_type] = {
                "tp": tp, "freq_mhz": freq,
                "active_instances": n_inst,
                "rate_per_instance": round(per_inst_rate, 2),
                "power_per_instance_w": round(power_per_inst, 1),
                "pool_power_w": round(pool_power, 1),
            }

        return SchedulingResult(total_power, config, slo_met, phase="HP")


# ---------------------------------------------------------------------------
# Strategy 2: DVFS-Only Baseline
# ---------------------------------------------------------------------------
class DVFSOnlyBaseline:
    """Phase A DVFS per GPU type independently. 50/50 rate split."""

    def __init__(self, schedulers: Dict[str, EnergyScheduler]):
        self.schedulers = schedulers

    def decide(self, il, ol, rate, slo, cluster: ClusterState) -> SchedulingResult:
        total_power = 0.0
        config = {}
        slo_met = True

        for gpu_type, sched in self.schedulers.items():
            spec = GPU_SPECS[gpu_type]
            half_rate = rate * 0.5

            # Use Phase A recommend() — it picks best (tp, freq)
            rec = sched.recommend(il, ol, half_rate, slo)

            if rec["status"] == "OK":
                best = rec["recommended"]
                tp = best["tp"]
                freq = best["freq_mhz"]
                n_inst = spec["total_gpus"] // tp
                # Re-compute per-instance rate for the actual TP/instances
                per_inst_rate = half_rate / n_inst if n_inst > 0 else 0
                pred = sched.predict_config(il, ol, tp, freq, per_inst_rate, slo)
                power_per_inst = pred["total_power_w"]
                pool_power = n_inst * power_per_inst
                idle_power = (spec["total_gpus"] - n_inst * tp) * spec["idle_power_w"]
                pool_total = pool_power + idle_power
                if not pred["is_safe"]:
                    slo_met = False
            else:
                # No safe config — run at max freq, TP=1
                tp = 1
                freq = spec["max_freq_mhz"]
                n_inst = spec["total_gpus"]
                per_inst_rate = half_rate / n_inst
                pred = sched.predict_config(il, ol, tp, freq, per_inst_rate, slo)
                power_per_inst = pred["total_power_w"]
                pool_total = n_inst * power_per_inst
                slo_met = False

            total_power += pool_total
            config[gpu_type] = {
                "tp": tp, "freq_mhz": freq,
                "active_instances": n_inst if rec["status"] == "OK" else spec["total_gpus"],
                "rate_per_instance": round(per_inst_rate, 2),
                "power_per_instance_w": round(power_per_inst, 1),
                "pool_power_w": round(pool_total, 1),
            }

        return SchedulingResult(total_power, config, slo_met, phase="HP")


# ---------------------------------------------------------------------------
# Strategy 3: Routing-Only Baseline
# ---------------------------------------------------------------------------
class RoutingOnlyBaseline:
    """Route to cheapest GPU type at max freq. No DVFS."""

    def __init__(self, schedulers: Dict[str, EnergyScheduler]):
        self.schedulers = schedulers

    def decide(self, il, ol, rate, slo, cluster: ClusterState) -> SchedulingResult:
        # For each GPU type at max freq, find cheapest safe TP
        options = {}
        for gpu_type, sched in self.schedulers.items():
            spec = GPU_SPECS[gpu_type]
            freq = spec["max_freq_mhz"]
            best_power = float("inf")
            best_tp = None

            for tp in spec["tp_degrees"]:
                n_inst = spec["total_gpus"] // tp
                per_inst_rate = rate / n_inst
                pred = sched.predict_config(il, ol, tp, freq, per_inst_rate, slo)
                if pred["is_safe"]:
                    pool_power = n_inst * pred["total_power_w"]
                    idle_power = (spec["total_gpus"] - n_inst * tp) * spec["idle_power_w"]
                    total = pool_power + idle_power
                    if total < best_power:
                        best_power = total
                        best_tp = tp
                        best_pred = pred

            if best_tp is not None:
                options[gpu_type] = {
                    "tp": best_tp, "freq_mhz": freq,
                    "power": best_power, "pred": best_pred,
                    "n_inst": spec["total_gpus"] // best_tp,
                }

        if not options:
            # Fallback: max everything
            return self._fallback(il, ol, rate, slo)

        # Route 100% to cheapest GPU type
        chosen = min(options, key=lambda g: options[g]["power"])
        other = "l4" if chosen == "l40s" else "l40s"

        config = {}
        total_power = 0.0

        # Active pool
        opt = options[chosen]
        active_power = opt["power"]
        config[chosen] = {
            "tp": opt["tp"], "freq_mhz": opt["freq_mhz"],
            "active_instances": opt["n_inst"],
            "rate_per_instance": round(rate / opt["n_inst"], 2),
            "power_per_instance_w": round(opt["pred"]["total_power_w"], 1),
            "pool_power_w": round(active_power, 1),
        }
        total_power += active_power

        # Idle pool — all GPUs idle
        other_spec = GPU_SPECS[other]
        idle_total = other_spec["total_gpus"] * other_spec["idle_power_w"]
        config[other] = {
            "tp": 1, "freq_mhz": 0,
            "active_instances": 0,
            "rate_per_instance": 0,
            "power_per_instance_w": 0,
            "pool_power_w": round(idle_total, 1),
        }
        total_power += idle_total

        return SchedulingResult(total_power, config, slo_met=True, phase="HP")

    def _fallback(self, il, ol, rate, slo):
        """No safe config anywhere — max everything."""
        total_power = 0.0
        config = {}
        for gpu_type, sched in self.schedulers.items():
            spec = GPU_SPECS[gpu_type]
            freq = spec["max_freq_mhz"]
            n_inst = spec["total_gpus"]
            per_inst_rate = rate * 0.5 / n_inst
            pred = sched.predict_config(il, ol, 1, freq, per_inst_rate, slo)
            pool_power = n_inst * pred["total_power_w"]
            total_power += pool_power
            config[gpu_type] = {
                "tp": 1, "freq_mhz": freq, "active_instances": n_inst,
                "rate_per_instance": round(per_inst_rate, 2),
                "power_per_instance_w": round(pred["total_power_w"], 1),
                "pool_power_w": round(pool_power, 1),
            }
        return SchedulingResult(total_power, config, slo_met=False, phase="HP")


# ---------------------------------------------------------------------------
# Strategy 4: DynamoLLM-style Baseline (Decoupled)
# ---------------------------------------------------------------------------
class DynamoLLMBaseline:
    """Decoupled: first pick GPU type (at max freq), then apply DVFS."""

    def __init__(self, schedulers: Dict[str, EnergyScheduler]):
        self.schedulers = schedulers

    def decide(self, il, ol, rate, slo, cluster: ClusterState) -> SchedulingResult:
        # Step 1: Pick GPU type at max freq (cheapest safe TP)
        gpu_options = {}
        for gpu_type, sched in self.schedulers.items():
            spec = GPU_SPECS[gpu_type]
            freq = spec["max_freq_mhz"]
            for tp in spec["tp_degrees"]:
                n_inst = spec["total_gpus"] // tp
                per_inst_rate = rate / n_inst
                pred = sched.predict_config(il, ol, tp, freq, per_inst_rate, slo)
                if pred["is_safe"]:
                    pool_power = n_inst * pred["total_power_w"]
                    idle_power = (spec["total_gpus"] - n_inst * tp) * spec["idle_power_w"]
                    gpu_options[gpu_type] = {
                        "tp": tp, "power": pool_power + idle_power,
                    }
                    break  # take first (lowest) safe TP

        if not gpu_options:
            return RoutingOnlyBaseline(self.schedulers)._fallback(il, ol, rate, slo)

        # Route to cheapest GPU type
        chosen = min(gpu_options, key=lambda g: gpu_options[g]["power"])
        other = "l4" if chosen == "l40s" else "l40s"

        # Step 2: Apply DVFS to chosen GPU type only
        sched = self.schedulers[chosen]
        spec = GPU_SPECS[chosen]
        rec = sched.recommend(il, ol, rate, slo)

        config = {}
        total_power = 0.0

        if rec["status"] == "OK":
            best = rec["recommended"]
            tp = best["tp"]
            freq = best["freq_mhz"]
            n_inst = spec["total_gpus"] // tp
            per_inst_rate = rate / n_inst
            pred = sched.predict_config(il, ol, tp, freq, per_inst_rate, slo)
            pool_power = n_inst * pred["total_power_w"]
            idle_power = (spec["total_gpus"] - n_inst * tp) * spec["idle_power_w"]
            pool_total = pool_power + idle_power
            config[chosen] = {
                "tp": tp, "freq_mhz": freq,
                "active_instances": n_inst,
                "rate_per_instance": round(per_inst_rate, 2),
                "power_per_instance_w": round(pred["total_power_w"], 1),
                "pool_power_w": round(pool_total, 1),
            }
            total_power += pool_total
        else:
            # Shouldn't happen since we checked above, but fallback
            tp = gpu_options[chosen]["tp"]
            freq = spec["max_freq_mhz"]
            n_inst = spec["total_gpus"] // tp
            per_inst_rate = rate / n_inst
            pred = sched.predict_config(il, ol, tp, freq, per_inst_rate, slo)
            pool_power = n_inst * pred["total_power_w"]
            config[chosen] = {
                "tp": tp, "freq_mhz": freq, "active_instances": n_inst,
                "rate_per_instance": round(per_inst_rate, 2),
                "power_per_instance_w": round(pred["total_power_w"], 1),
                "pool_power_w": round(pool_power, 1),
            }
            total_power += pool_power

        # Other pool: idle
        other_spec = GPU_SPECS[other]
        idle_total = other_spec["total_gpus"] * other_spec["idle_power_w"]
        config[other] = {
            "tp": 1, "freq_mhz": 0, "active_instances": 0,
            "rate_per_instance": 0, "power_per_instance_w": 0,
            "pool_power_w": round(idle_total, 1),
        }
        total_power += idle_total

        return SchedulingResult(total_power, config, slo_met=True, phase="HP")


# ---------------------------------------------------------------------------
# Strategy 5: JSEP (Joint Phase-Aware)
# ---------------------------------------------------------------------------
class JSEPStrategy:
    """Joint optimization with HP/LP phase-aware algorithms."""

    # Phase classification constants (inspired by SWEEP [Chen et al., IPDPS'24])
    # λ_HP = k × N_total × sqrt(SLO_ms / SLO_ref)
    # k  = per-GPU load factor (analogous to SWEEP's 4-tasks-per-core constant)
    # N_total = total GPUs across all pools
    # SLO_ref = reference SLO at which λ_HP = k × N_total
    PHASE_K = 5.0 / 3.0       # per-GPU load factor
    PHASE_SLO_REF = 1000.0    # reference SLO (ms)
    PHASE_LP_RATIO = 0.5      # λ_LP = λ_HP × this ratio (hysteresis band)

    # TP reconfiguration
    RECONFIG_PENALTY_S = 5.0
    RECONFIG_COOLDOWN_S = 30.0

    def __init__(self, schedulers: Dict[str, EnergyScheduler]):
        self.schedulers = schedulers
        self.current_phase = "LP"
        self.last_reconfig_time = -999.0
        self.prev_tp = {"l40s": 1, "l4": 1}
        self._override_thresholds = None  # (hp, lp) override for sensitivity sweep
        # Compute N_total from cluster hardware
        self._n_total = sum(GPU_SPECS[g]["total_gpus"] for g in self.schedulers)

    def _get_thresholds(self, slo):
        """Compute HP/LP thresholds from system capacity and SLO.

        Formula (cf. SWEEP's threshold = 4 × N_cores):
            λ_HP = k × N_total × sqrt(SLO_ms / SLO_ref)
            λ_LP = λ_HP × PHASE_LP_RATIO
        """
        if self._override_thresholds is not None:
            return self._override_thresholds
        import math
        lam_hp = self.PHASE_K * self._n_total * math.sqrt(slo / self.PHASE_SLO_REF)
        lam_lp = lam_hp * self.PHASE_LP_RATIO
        return (lam_hp, lam_lp)

    def classify_phase(self, rate: float, slo: int) -> str:
        """Classify current phase with hysteresis, SLO-aware."""
        hp_th, lp_th = self._get_thresholds(slo)
        if rate >= hp_th:
            self.current_phase = "HP"
        elif rate <= lp_th:
            self.current_phase = "LP"
        # else: keep current_phase (hysteresis)
        return self.current_phase

    def decide(self, il, ol, rate, slo, cluster: ClusterState,
               current_time: float = 0.0) -> SchedulingResult:
        phase = self.classify_phase(rate, slo)
        if phase == "HP":
            return self._hp_decide(il, ol, rate, slo, cluster, current_time)
        else:
            result = self._lp_decide(il, ol, rate, slo, cluster, current_time)
            # If LP couldn't find a feasible config, escalate to HP
            if not result.slo_met:
                hp_result = self._hp_decide(il, ol, rate, slo, cluster, current_time)
                if hp_result.slo_met or hp_result.total_power_w < result.total_power_w:
                    return hp_result
            return result

    def _hp_decide(self, il, ol, rate, slo, cluster, current_time):
        """HP: All GPUs active. Joint optimize (TP, freq, routing ratio)."""
        best_power = float("inf")
        best_result = None

        for tp_l40s in GPU_SPECS["l40s"]["tp_degrees"]:
            n_inst_l40s = GPU_SPECS["l40s"]["total_gpus"] // tp_l40s
            for tp_l4 in GPU_SPECS["l4"]["tp_degrees"]:
                n_inst_l4 = GPU_SPECS["l4"]["total_gpus"] // tp_l4

                # Try different routing ratios (alpha = fraction to L40S)
                for alpha_i in range(11):  # 0.0, 0.1, ..., 1.0
                    alpha = alpha_i / 10.0

                    # Per-instance rates
                    if alpha > 0 and n_inst_l40s > 0:
                        rate_l40s = (alpha * rate) / n_inst_l40s
                    elif alpha > 0:
                        continue  # can't route to L40S with 0 instances
                    else:
                        rate_l40s = 0

                    if alpha < 1.0 and n_inst_l4 > 0:
                        rate_l4 = ((1 - alpha) * rate) / n_inst_l4
                    elif alpha < 1.0:
                        continue
                    else:
                        rate_l4 = 0

                    # Find minimum safe freq for each pool
                    total_power = 0.0
                    feasible = True
                    config = {}

                    # L40S pool
                    if rate_l40s > 0:
                        freq_l40s, pred_l40s = _find_min_safe_freq(
                            self.schedulers["l40s"], il, ol, tp_l40s,
                            rate_l40s, slo)
                        if freq_l40s is None:
                            feasible = False
                            break
                        pool_power = n_inst_l40s * pred_l40s["total_power_w"]
                        total_power += pool_power
                        config["l40s"] = {
                            "tp": tp_l40s, "freq_mhz": freq_l40s,
                            "active_instances": n_inst_l40s,
                            "rate_per_instance": round(rate_l40s, 2),
                            "power_per_instance_w": round(pred_l40s["total_power_w"], 1),
                            "pool_power_w": round(pool_power, 1),
                        }
                    else:
                        # L40S not used — power-gate the pool
                        config["l40s"] = {
                            "tp": tp_l40s, "freq_mhz": 0,
                            "active_instances": 0,
                            "rate_per_instance": 0,
                            "power_per_instance_w": 0,
                            "pool_power_w": 0,
                        }

                    if not feasible:
                        continue

                    # L4 pool
                    if rate_l4 > 0:
                        freq_l4, pred_l4 = _find_min_safe_freq(
                            self.schedulers["l4"], il, ol, tp_l4,
                            rate_l4, slo)
                        if freq_l4 is None:
                            continue  # infeasible
                        pool_power = n_inst_l4 * pred_l4["total_power_w"]
                        total_power += pool_power
                        config["l4"] = {
                            "tp": tp_l4, "freq_mhz": freq_l4,
                            "active_instances": n_inst_l4,
                            "rate_per_instance": round(rate_l4, 2),
                            "power_per_instance_w": round(pred_l4["total_power_w"], 1),
                            "pool_power_w": round(pool_power, 1),
                        }
                    else:
                        # L4 not used — power-gate the pool
                        config["l4"] = {
                            "tp": tp_l4, "freq_mhz": 0,
                            "active_instances": 0,
                            "rate_per_instance": 0,
                            "power_per_instance_w": 0,
                            "pool_power_w": 0,
                        }

                    # Check TP reconfig penalty
                    tp_changed = (tp_l40s != self.prev_tp.get("l40s", tp_l40s) or
                                  tp_l4 != self.prev_tp.get("l4", tp_l4))
                    if tp_changed and (current_time - self.last_reconfig_time < self.RECONFIG_COOLDOWN_S):
                        continue  # cooldown not elapsed

                    if total_power < best_power:
                        best_power = total_power
                        best_config = config.copy()
                        best_tp = {"l40s": tp_l40s, "l4": tp_l4}
                        best_tp_changed = tp_changed

        if best_power < float("inf"):
            if best_tp_changed:
                self.last_reconfig_time = current_time
            self.prev_tp = best_tp
            return SchedulingResult(best_power, best_config, slo_met=True, phase="HP")
        else:
            # No feasible config — fallback to naive
            return NaiveBaseline(self.schedulers).decide(il, ol, rate, slo, cluster)

    def _lp_decide(self, il, ol, rate, slo, cluster, current_time):
        """LP: Can power-gate GPUs. Joint optimize including instance count."""
        best_power = float("inf")
        best_config = None
        best_tp = None
        best_tp_changed = False

        for tp_l40s in GPU_SPECS["l40s"]["tp_degrees"]:
            max_inst_l40s = GPU_SPECS["l40s"]["total_gpus"] // tp_l40s
            for n_active_l40s in range(0, max_inst_l40s + 1):
                for tp_l4 in GPU_SPECS["l4"]["tp_degrees"]:
                    max_inst_l4 = GPU_SPECS["l4"]["total_gpus"] // tp_l4
                    for n_active_l4 in range(0, max_inst_l4 + 1):
                        total_instances = n_active_l40s + n_active_l4
                        if total_instances == 0:
                            continue  # must have at least 1 instance

                        # Routing: try a few alpha values
                        if n_active_l40s == 0:
                            alphas = [0.0]
                        elif n_active_l4 == 0:
                            alphas = [1.0]
                        else:
                            # Proportional + a few alternatives
                            prop = n_active_l40s / total_instances
                            alphas = sorted(set([
                                round(prop, 1),
                                0.0, 0.3, 0.5, 0.7, 1.0
                            ]))
                            alphas = [a for a in alphas
                                      if (a == 0 or n_active_l40s > 0)
                                      and (a == 1.0 or n_active_l4 > 0)]

                        for alpha in alphas:
                            total_power = 0.0
                            feasible = True
                            config = {}

                            # L40S
                            if n_active_l40s > 0 and alpha > 0:
                                rate_l40s = (alpha * rate) / n_active_l40s
                                freq_l40s, pred_l40s = _find_min_safe_freq(
                                    self.schedulers["l40s"], il, ol, tp_l40s,
                                    rate_l40s, slo)
                                if freq_l40s is None:
                                    feasible = False
                                else:
                                    active_power = n_active_l40s * pred_l40s["total_power_w"]
                                    idle_gpus = GPU_SPECS["l40s"]["total_gpus"] - n_active_l40s * tp_l40s
                                    idle_power = idle_gpus * GPU_SPECS["l40s"]["idle_power_w"]
                                    pool_total = active_power + idle_power
                                    total_power += pool_total
                                    config["l40s"] = {
                                        "tp": tp_l40s, "freq_mhz": freq_l40s,
                                        "active_instances": n_active_l40s,
                                        "rate_per_instance": round(rate_l40s, 2),
                                        "power_per_instance_w": round(pred_l40s["total_power_w"], 1),
                                        "pool_power_w": round(pool_total, 1),
                                    }
                            elif n_active_l40s > 0 and alpha == 0:
                                # Active but no traffic — wasteful, skip
                                continue
                            else:
                                # All L40S idle
                                idle = GPU_SPECS["l40s"]["total_gpus"] * GPU_SPECS["l40s"]["idle_power_w"]
                                total_power += idle
                                config["l40s"] = {
                                    "tp": tp_l40s, "freq_mhz": 0,
                                    "active_instances": 0,
                                    "rate_per_instance": 0,
                                    "power_per_instance_w": 0,
                                    "pool_power_w": round(idle, 1),
                                }

                            if not feasible:
                                continue

                            # L4
                            if n_active_l4 > 0 and alpha < 1.0:
                                rate_l4 = ((1 - alpha) * rate) / n_active_l4
                                freq_l4, pred_l4 = _find_min_safe_freq(
                                    self.schedulers["l4"], il, ol, tp_l4,
                                    rate_l4, slo)
                                if freq_l4 is None:
                                    continue
                                active_power = n_active_l4 * pred_l4["total_power_w"]
                                idle_gpus = GPU_SPECS["l4"]["total_gpus"] - n_active_l4 * tp_l4
                                idle_power = idle_gpus * GPU_SPECS["l4"]["idle_power_w"]
                                pool_total = active_power + idle_power
                                total_power += pool_total
                                config["l4"] = {
                                    "tp": tp_l4, "freq_mhz": freq_l4,
                                    "active_instances": n_active_l4,
                                    "rate_per_instance": round(rate_l4, 2),
                                    "power_per_instance_w": round(pred_l4["total_power_w"], 1),
                                    "pool_power_w": round(pool_total, 1),
                                }
                            elif n_active_l4 > 0 and alpha == 1.0:
                                continue  # active but no traffic
                            else:
                                idle = GPU_SPECS["l4"]["total_gpus"] * GPU_SPECS["l4"]["idle_power_w"]
                                total_power += idle
                                config["l4"] = {
                                    "tp": tp_l4, "freq_mhz": 0,
                                    "active_instances": 0,
                                    "rate_per_instance": 0,
                                    "power_per_instance_w": 0,
                                    "pool_power_w": round(idle, 1),
                                }

                            # TP reconfig check
                            tp_changed = (tp_l40s != self.prev_tp.get("l40s", tp_l40s) or
                                          tp_l4 != self.prev_tp.get("l4", tp_l4))
                            if tp_changed and (current_time - self.last_reconfig_time < self.RECONFIG_COOLDOWN_S):
                                continue

                            if total_power < best_power:
                                best_power = total_power
                                best_config = config.copy()
                                best_tp = {"l40s": tp_l40s, "l4": tp_l4}
                                best_tp_changed = tp_changed

        if best_config is not None:
            if best_tp_changed:
                self.last_reconfig_time = current_time
            self.prev_tp = best_tp
            return SchedulingResult(best_power, best_config, slo_met=True, phase="LP")
        else:
            # Fallback to HP
            return self._hp_decide(il, ol, rate, slo, cluster, current_time)


# ---------------------------------------------------------------------------
# Ablation: JSEP-NoLP (always HP, no instance scaling)
# ---------------------------------------------------------------------------
class JSEPNoLPStrategy(JSEPStrategy):
    """JSEP with LP mode disabled — always uses HP algorithm."""

    def classify_phase(self, rate: float, slo: int) -> str:
        self.current_phase = "HP"
        return "HP"


# ---------------------------------------------------------------------------
# Ablation: JSEP-Decoupled (uses JSEP's models but route-then-DVFS)
# ---------------------------------------------------------------------------
class JSEPDecoupledStrategy:
    """
    Decoupled ablation using JSEP's own machinery.
    Step 1: Route 100% to cheapest pool at max freq.
    Step 2: Apply DVFS to the chosen pool only.
    Uses phase-aware HP/LP like full JSEP but decouples routing and DVFS.
    """

    # Reuse JSEP phase classification constants
    PHASE_K = JSEPStrategy.PHASE_K
    PHASE_SLO_REF = JSEPStrategy.PHASE_SLO_REF
    PHASE_LP_RATIO = JSEPStrategy.PHASE_LP_RATIO
    RECONFIG_PENALTY_S = 5.0
    RECONFIG_COOLDOWN_S = 30.0

    def __init__(self, schedulers: Dict[str, EnergyScheduler]):
        self.schedulers = schedulers
        self.current_phase = "LP"
        self.last_reconfig_time = -999.0
        self.prev_tp = {"l40s": 1, "l4": 1}
        self._n_total = sum(GPU_SPECS[g]["total_gpus"] for g in self.schedulers)

    def _get_thresholds(self, slo):
        import math
        lam_hp = self.PHASE_K * self._n_total * math.sqrt(slo / self.PHASE_SLO_REF)
        lam_lp = lam_hp * self.PHASE_LP_RATIO
        return (lam_hp, lam_lp)

    def classify_phase(self, rate: float, slo: int) -> str:
        hp_th, lp_th = self._get_thresholds(slo)
        if rate >= hp_th:
            self.current_phase = "HP"
        elif rate <= lp_th:
            self.current_phase = "LP"
        return self.current_phase

    def decide(self, il, ol, rate, slo, cluster: ClusterState,
               current_time: float = 0.0) -> SchedulingResult:
        phase = self.classify_phase(rate, slo)
        # Step 1: Find cheapest pool at max freq (full load, best TP)
        best_pool = None
        best_pool_power = float("inf")
        best_tp_pool = {}

        for gpu_type, sched in self.schedulers.items():
            spec = GPU_SPECS[gpu_type]
            max_freq = spec["max_freq_mhz"]
            for tp in spec["tp_degrees"]:
                n_inst = spec["total_gpus"] // tp
                rate_per = rate / n_inst
                pred = sched.predict_config(il, ol, tp, max_freq, rate_per, slo)
                if pred["is_safe"]:
                    pool_power = n_inst * pred["total_power_w"]
                    idle_gpus = spec["total_gpus"] - n_inst * tp
                    idle_power = idle_gpus * spec.get("idle_power_w", 0)
                    total = pool_power + idle_power
                    if total < best_pool_power:
                        best_pool_power = total
                        best_pool = gpu_type
                        best_tp_pool = {gpu_type: tp}

        if best_pool is None:
            return NaiveBaseline(self.schedulers).decide(il, ol, rate, slo, cluster)

        # Step 2: Apply DVFS to chosen pool only
        chosen_tp = best_tp_pool[best_pool]
        spec = GPU_SPECS[best_pool]
        n_inst = spec["total_gpus"] // chosen_tp
        rate_per = rate / n_inst

        freq, pred = _find_min_safe_freq(
            self.schedulers[best_pool], il, ol, chosen_tp, rate_per, slo)
        if freq is None:
            freq = spec["max_freq_mhz"]
            pred = self.schedulers[best_pool].predict_config(
                il, ol, chosen_tp, freq, rate_per, slo)

        active_power = n_inst * pred["total_power_w"]
        idle_gpus_chosen = spec["total_gpus"] - n_inst * chosen_tp
        idle_power_chosen = idle_gpus_chosen * spec.get("idle_power_w", 0)

        # Other pool is idle
        other_pool = "l4" if best_pool == "l40s" else "l40s"
        other_spec = GPU_SPECS[other_pool]
        idle_power_other = other_spec["total_gpus"] * other_spec.get("idle_power_w", 0)

        total_power = active_power + idle_power_chosen + idle_power_other

        config = {}
        config[best_pool] = {
            "tp": chosen_tp, "freq_mhz": freq,
            "active_instances": n_inst,
            "rate_per_instance": round(rate_per, 2),
            "power_per_instance_w": round(pred["total_power_w"], 1),
            "pool_power_w": round(active_power + idle_power_chosen, 1),
        }
        config[other_pool] = {
            "tp": 1, "freq_mhz": 0, "active_instances": 0,
            "rate_per_instance": 0, "power_per_instance_w": 0,
            "pool_power_w": round(idle_power_other, 1),
        }

        return SchedulingResult(total_power, config, slo_met=True, phase=phase)


def create_all_strategies(model_dir_l40s=None,
                          model_dir_l4=None) -> dict:
    """Create all five strategies with shared Phase A models."""
    model_dir_l40s = str(paper_model_dir("models_l40s")) if model_dir_l40s is None else model_dir_l40s
    model_dir_l4 = str(paper_model_dir("models_l4")) if model_dir_l4 is None else model_dir_l4
    schedulers = {
        "l40s": EnergyScheduler(model_dir=model_dir_l40s),
        "l4": EnergyScheduler(model_dir=model_dir_l4),
    }
    return {
        "naive": NaiveBaseline(schedulers),
        "dvfs_only": DVFSOnlyBaseline(schedulers),
        "routing_only": RoutingOnlyBaseline(schedulers),
        "dynamollm": DynamoLLMBaseline(schedulers),
        "jsep": JSEPStrategy(schedulers),
    }


def create_ablation_strategies(model_dir_l40s=None,
                                model_dir_l4=None) -> dict:
    """Create ablation variants for fair internal comparison."""
    model_dir_l40s = str(paper_model_dir("models_l40s")) if model_dir_l40s is None else model_dir_l40s
    model_dir_l4 = str(paper_model_dir("models_l4")) if model_dir_l4 is None else model_dir_l4
    schedulers = {
        "l40s": EnergyScheduler(model_dir=model_dir_l40s),
        "l4": EnergyScheduler(model_dir=model_dir_l4),
    }
    return {
        "jsep": JSEPStrategy(schedulers),
        "jsep_no_lp": JSEPNoLPStrategy(schedulers),
        "jsep_decoupled": JSEPDecoupledStrategy(schedulers),
        "dvfs_only": DVFSOnlyBaseline(schedulers),
        "routing_only": RoutingOnlyBaseline(schedulers),
    }
