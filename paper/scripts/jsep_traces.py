"""
JSEP Phase B: Workload trace generation for heterogeneous GPU simulation.

Generates synthetic request traces with different arrival patterns.
Each request has (arrival_time, input_len, output_len).
"""

import math
import random
from dataclasses import dataclass
from typing import List, Optional  # noqa: F401

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


class _StdlibRandomState:
    """Small numpy.RandomState subset used only for trace generation."""

    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def exponential(self, scale: float, size: int = None):
        if size is None:
            return self._rng.expovariate(1.0 / scale)
        return [self._rng.expovariate(1.0 / scale) for _ in range(size)]

    def choice(self, choices, p=None):
        choices = list(choices)
        if p is None:
            return self._rng.choice(choices)
        return self._rng.choices(choices, weights=list(p), k=1)[0]


def _random_state(seed: int):
    if np is not None:
        return np.random.RandomState(seed)
    return _StdlibRandomState(seed)


def _sin(value: float) -> float:
    return np.sin(value) if np is not None else math.sin(value)


def _cumsum(values):
    if np is not None:
        return np.cumsum(values)
    out = []
    total = 0.0
    for value in values:
        total += value
        out.append(total)
    return out

@dataclass
class Request:
    arrival_time: float  # seconds since trace start
    input_len: int
    output_len: int

# Discrete values matching Phase A characterization
IL_VALUES = [32, 128, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
OL_VALUES = [32, 128, 512, 1024]

# Default workload mix profile
DEFAULT_MIX = {
    "short_short": 0.30,  # il<=128, ol<=128
    "short_long":  0.20,  # il<=128, ol>=512
    "long_short":  0.25,  # il>=512, ol<=128
    "long_long":   0.25,  # il>=512, ol>=512
}

MIX_RANGES = {
    "short_short": ([32, 128], [32, 128]),
    "short_long":  ([32, 128], [512, 1024]),
    "long_short":  ([512, 1024, 2048], [32, 128]),
    "long_long":   ([512, 1024, 2048], [512, 1024]),
}

# Controlled synthetic classes used by the current SWEEP-LLM evaluation.
CONTROLLED_CLASSES = {
    "short_short": (64, 128),
    "short_long": (64, 512),
    "long_short": (1024, 128),
    "long_long": (1024, 512),
}

CONTROLLED_MIX_T1 = {
    "long_short": 0.50,
    "long_long": 0.25,
    "short_short": 0.25,
}

CONTROLLED_MIX_T2 = {
    "short_long": 0.50,
    "long_long": 0.25,
    "short_short": 0.25,
}

CONTROLLED_MIX_T4 = {
    "short_short": 0.25,
    "short_long": 0.25,
    "long_short": 0.25,
    "long_long": 0.25,
}

# Calibrated from results/disagg_20260321_v2 via calibrate_trace_rates.py.
# These are conservative evaluation replay rates, roughly 85% of the moderate
# SLO peak estimates instead of the old infeasible 50/100 rps settings.
CONTROLLED_T1_MIN_RPS = 2.0
CONTROLLED_T1_PEAK_RPS = 34.0
CONTROLLED_T2_MIN_RPS = 2.0
CONTROLLED_T2_PEAK_RPS = 24.0
CONTROLLED_T3_RATE_RPS = 24.0
CONTROLLED_T3_TRANSITION_S = 120.0
CONTROLLED_T4_BASE_RPS = 10.0
CONTROLLED_T4_SPIKE_RPS = 34.0
CONTROLLED_T4_SPIKE_DURATION_S = 20.0
CONTROLLED_T4_SPIKE_PERIOD_S = 60.0


def _sample_workload(rng, mix=None):
    """Sample a single (il, ol) from the workload mix profile."""
    mix = mix or DEFAULT_MIX
    categories = list(mix.keys())
    weights = [mix[c] for c in categories]
    cat = rng.choice(categories, p=weights)
    il_choices, ol_choices = MIX_RANGES[cat]
    il = rng.choice(il_choices)
    ol = rng.choice(ol_choices)
    return il, ol


def _generate_requests(rng, times, mix=None):
    """Given arrival times, generate requests with sampled workloads."""
    requests = []
    for t in times:
        il, ol = _sample_workload(rng, mix)
        requests.append(Request(arrival_time=t, input_len=il, output_len=ol))
    return requests


def _sample_controlled_workload(rng, mix):
    categories = list(mix.keys())
    weights = [float(mix[c]) for c in categories]
    weight_sum = sum(weights)
    weights = [weight / weight_sum for weight in weights]
    class_id = rng.choice(categories, p=weights)
    il, ol = CONTROLLED_CLASSES[class_id]
    return class_id, il, ol


def _generate_controlled_requests(rng, times, mix):
    requests = []
    for t in times:
        _, il, ol = _sample_controlled_workload(rng, mix)
        requests.append(Request(arrival_time=t, input_len=il, output_len=ol))
    return requests


def _generate_time_varying_poisson(duration_s: float, rate_fn, seed: int = 42):
    rng = _random_state(seed)
    times = []
    t = 0.0
    while t < duration_s:
        rate = max(float(rate_fn(t)), 0.1)
        t += rng.exponential(1.0 / rate)
        if t < duration_s:
            times.append(t)
    return rng, times


def snap_to_nearest(value, choices):
    """Snap a value to the nearest element in choices."""
    return min(choices, key=lambda x: abs(x - value))


def generate_steady(duration_s: float, rate_rps: float,
                    mix=None, seed: int = 42) -> List[Request]:
    """Constant arrival rate trace."""
    rng = _random_state(seed)
    n_requests = int(duration_s * rate_rps)
    # Poisson process: exponential inter-arrival times
    inter_arrivals = rng.exponential(1.0 / rate_rps, n_requests)
    times = [t for t in _cumsum(inter_arrivals) if t < duration_s]
    return _generate_requests(rng, times, mix)


def generate_bursty(duration_s: float, base_rate: float = 2.0,
                    burst_rate: float = 30.0, burst_duration_s: float = 15.0,
                    burst_interval_s: float = 60.0,
                    mix=None, seed: int = 42) -> List[Request]:
    """Periodic burst trace: low base rate with periodic spikes."""
    rng = _random_state(seed)
    times = []
    t = 0.0
    while t < duration_s:
        # Determine if we're in a burst
        cycle_pos = t % burst_interval_s
        if cycle_pos < burst_duration_s:
            rate = burst_rate
        else:
            rate = base_rate
        # Generate next arrival
        t += rng.exponential(1.0 / rate)
        if t < duration_s:
            times.append(t)
    return _generate_requests(rng, times, mix)


def generate_diurnal(duration_s: float, min_rate: float = 1.0,
                     max_rate: float = 50.0, period_s: Optional[float] = None,
                     mix=None, seed: int = 42) -> List[Request]:
    """Sinusoidal arrival rate simulating day/night pattern."""
    rng = _random_state(seed)
    if period_s is None:
        period_s = duration_s  # one full cycle over the trace
    times = []
    t = 0.0
    while t < duration_s:
        # Current rate from sinusoidal pattern
        rate = min_rate + (max_rate - min_rate) * (
            0.5 * (1 + _sin(2 * math.pi * t / period_s - math.pi / 2))
        )
        rate = max(rate, 0.1)  # floor to avoid division by zero
        t += rng.exponential(1.0 / rate)
        if t < duration_s:
            times.append(t)
    return _generate_requests(rng, times, mix)


# ---------------------------------------------------------------------------
# Structured traces T1–T4: exercise multi-GPU pool model
# ---------------------------------------------------------------------------

def generate_T1_prefill_heavy(duration_s: float = 600.0,
                               seed: int = 42) -> List[Request]:
    """T1: prefill-heavy ramp with controlled request classes."""
    rng, times = _generate_time_varying_poisson(
        duration_s,
        lambda t: CONTROLLED_T1_MIN_RPS + (CONTROLLED_T1_PEAK_RPS - CONTROLLED_T1_MIN_RPS) * (
            0.5 * (1 + _sin(2 * math.pi * t / duration_s - math.pi / 2))
        ),
        seed=seed,
    )
    return _generate_controlled_requests(rng, times, CONTROLLED_MIX_T1)


def generate_T2_decode_heavy(duration_s: float = 600.0,
                              seed: int = 42) -> List[Request]:
    """T2: decode-heavy ramp with controlled request classes."""
    rng, times = _generate_time_varying_poisson(
        duration_s,
        lambda t: CONTROLLED_T2_MIN_RPS + (CONTROLLED_T2_PEAK_RPS - CONTROLLED_T2_MIN_RPS) * (
            0.5 * (1 + _sin(2 * math.pi * t / duration_s - math.pi / 2))
        ),
        seed=seed,
    )
    return _generate_controlled_requests(rng, times, CONTROLLED_MIX_T2)


def generate_T3_phase_shift(duration_s: float = 600.0,
                             seed: int = 42) -> List[Request]:
    """T3: gradual phase shift from T1-style mix to T2-style mix."""
    rng = _random_state(seed)
    times = []
    t = 0.0
    rate = CONTROLLED_T3_RATE_RPS
    while t < duration_s:
        t += rng.exponential(1.0 / rate)
        if t < duration_s:
            times.append(t)
    mid = duration_s / 2.0
    transition_start = mid - CONTROLLED_T3_TRANSITION_S / 2.0
    transition_end = mid + CONTROLLED_T3_TRANSITION_S / 2.0
    requests = []
    for t in times:
        if t <= transition_start:
            mix = CONTROLLED_MIX_T1
        elif t >= transition_end:
            mix = CONTROLLED_MIX_T2
        else:
            alpha = (t - transition_start) / CONTROLLED_T3_TRANSITION_S
            mix = {
                key: (1.0 - alpha) * CONTROLLED_MIX_T1.get(key, 0.0)
                + alpha * CONTROLLED_MIX_T2.get(key, 0.0)
                for key in CONTROLLED_MIX_T4
            }
        _, il, ol = _sample_controlled_workload(rng, mix)
        requests.append(Request(arrival_time=t, input_len=il, output_len=ol))
    return requests


def generate_T4_overload_burst(duration_s: float = 600.0,
                                seed: int = 42,
                                base_rate: float = CONTROLLED_T4_BASE_RPS,
                                spike_rate: float = CONTROLLED_T4_SPIKE_RPS,
                                spike_duration_s: float = CONTROLLED_T4_SPIKE_DURATION_S,
                                spike_period_s: float = CONTROLLED_T4_SPIKE_PERIOD_S,
                                mix: dict = None) -> List[Request]:
    """T4: overload bursts over a balanced controlled mix."""
    mix = mix or CONTROLLED_MIX_T4
    rng, times = _generate_time_varying_poisson(
        duration_s,
        lambda t: spike_rate if (t % spike_period_s) < spike_duration_s else base_rate,
        seed=seed,
    )
    return _generate_controlled_requests(rng, times, mix)


# Convenience: generate all standard traces
def generate_all_traces(duration_s: float = 600.0, seed: int = 42):
    """Generate the standard traces for JSEP evaluation."""
    return {
        "steady": generate_steady(duration_s, rate_rps=10.0, seed=seed),
        "bursty": generate_bursty(duration_s, seed=seed),
        "diurnal": generate_diurnal(duration_s, seed=seed),
        "T1": generate_T1_prefill_heavy(duration_s, seed=seed),
        "T2": generate_T2_decode_heavy(duration_s, seed=seed),
        "T3": generate_T3_phase_shift(duration_s, seed=seed),
        "T4": generate_T4_overload_burst(duration_s, seed=seed),
    }


def load_azure_trace(csv_path: str,
                     duration_s: Optional[float] = None,
                     scale_rate: Optional[float] = None) -> List[Request]:
    """
    Load an Azure production trace from a preprocessed arrival CSV.

    The CSV must have columns: arrival_time_s, input_len, output_len
    (produced by prepare_azure_trace.py).

    Args:
        csv_path:    Path to arrival CSV file.
        duration_s:  If set, truncate trace to this many seconds.
        scale_rate:  If set, rescale arrival times to hit this mean rate (req/s).
    """
    import csv as _csv

    requests = []
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            t   = float(row["arrival_time_s"])
            il  = int(row["input_len"])
            ol  = int(row["output_len"])
            # Snap to nearest supported IL/OL values for model compatibility
            il = snap_to_nearest(il, IL_VALUES)
            ol = snap_to_nearest(ol, OL_VALUES)
            requests.append(Request(arrival_time=t, input_len=il, output_len=ol))

    requests.sort(key=lambda r: r.arrival_time)

    if duration_s is not None:
        requests = [r for r in requests if r.arrival_time < duration_s]

    if scale_rate is not None and len(requests) > 1:
        current_duration = requests[-1].arrival_time
        current_rate     = len(requests) / current_duration
        scale            = current_rate / scale_rate
        requests = [Request(r.arrival_time * scale, r.input_len, r.output_len)
                    for r in requests]

    return requests


if __name__ == "__main__":
    traces = generate_all_traces()
    for name, trace in traces.items():
        rates = []
        # compute avg rate in 5s windows
        for t in range(0, 600, 5):
            n = sum(1 for r in trace if t <= r.arrival_time < t + 5)
            rates.append(n / 5.0)
        mean_rate = sum(rates) / len(rates) if rates else 0.0
        print(f"{name}: {len(trace)} requests, "
              f"rate range [{min(rates):.1f}, {max(rates):.1f}] req/s, "
              f"mean {mean_rate:.1f} req/s")
