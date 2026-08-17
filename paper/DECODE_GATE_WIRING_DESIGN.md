# Wiring design: prefill envelope + decode goodput-capacity gate

Status: IMPLEMENTED (v2, post safety review). Locked decisions: hybrid fallback,
`eta_confirmed=0.85` / `eta_fallback=0.80`. Closes three false-safe entry points: SLO
out-of-range, TP pseudo-dominance, length snapping. **Every lookup miss fails CLOSED.**
Landed: `schedulers/feasibility_tables.py`, config knobs in `common.py`, three edit points
in `sweep.py` (`__init__` table load + fail-fast, decode pool-rho override, gate). Defaults
are legacy (`decode_gate_mode="model_rho"`, `prefill_envelope=False`) => byte-identical to
prior behavior; smoke-tested: all 3 modes run, legacy default unchanged (851.2 W on the test
window). Flip to `goodput`/`True` once the hardware decode freq x shape x SLO table lands
(only ~14-23% grid coverage on existing data — see blocker below).

## Goal
Replace the single distorted `rho_pf>1 or rho_dc>1` admission test in
`sweep.py::_predict_class_metrics` (currently lines ~1521-1524) with the asymmetric,
SLO-aware, fail-closed pair:

```
prefill:  rho_pf <= rho_pf_max(g, TP, f, SLO_TTFT)   # keep model rho, move the bound; None -> reject
decode:   rho_dc := D_dc / (eta * C_dc_SLO(...)) <= 1 # rebuilt goodput capacity; no capacity -> reject
```

Prefill keeps the (sane, SLO-sensitive) capacity-model `rho_pf`, only swapping the fixed
`1.0` for the empirical envelope. Decode discards the capacity-model `rho_dc` (distorted, up
to 116/634) and recomputes utilization against demonstrated goodput capacity. This asymmetry
is what the data forced (HARDWARE_EVAL_MODEL_PLAN #5).

## Artifacts consumed (both offline, no new hardware)
1. `artifacts/paper/models/decode_capacity.csv` — `calibrate_decode_capacity.py`.
   Key `(gpu, il, ol, tp, freq_mhz, slo_tpot_ms)` -> `C_dc_SLO_tps`, `C_dc_stable_tps`,
   `stable_confirmed`, `slo_confirmed`, `n_qual`, `n_rates`. **Follow-up needed:** the
   script must also emit, per `(gpu, il, ol, tp, slo)`, a `freq_monotone_certified` flag
   (C_dc_SLO non-decreasing in freq across confirmed points) so the runtime can borrow a
   lower-frequency bound WITHOUT inferring monotonicity itself.
2. `artifacts/paper/models/rho_envelope.csv` — `calibrate_rho_envelope.py`, **prefill rows
   only** (decode rows superseded by #1, ignored). Key
   `(gpu, phase=prefill, tp, freq_mhz, shape=all, slo_ms)` -> `rho_max`.

## New module: `schedulers/feasibility_tables.py`
Self-contained loader + lookup (uncluttered `sweep.py`, unit-testable). Final contract:

```python
@dataclass(frozen=True)
class CapacityLookup:
    capacity_tps: float
    source: Literal["measured",              # exact row, slo_confirmed -> eta_confirmed
                    "self_observed_lower_bound",  # exact row, unconfirmed but C>0 -> eta_fallback
                    "lower_freq_bound",       # borrowed, same-TP lower-freq, certified -> eta_fallback
                    "none"]                   # -> reject
    support_key: tuple | None
    requested_slo_ms: float
    selected_slo_ms: float | None

class FeasibilityTables:
    def __init__(self, decode_csv, rho_envelope_csv):
        # FAIL FAST: if a table is missing / schema mismatch / a GPU has no rows,
        # raise at construction. The goodput gate must never silently revert to model-rho.

    def prefill_rho_max(self, gpu, tp, freq, slo_ttft_ms) -> Optional[float]:
        # exact (gpu,tp,freq); SLO = largest tabulated slo_ms <= requested.
        # No tabulated SLO <= requested  -> None (reject; do NOT borrow a looser SLO).
        # Missing (gpu,tp,freq) in envelope mode -> None (fail closed; only explicit
        # legacy mode uses the fixed rho_pf<=1).

    def decode_safe_capacity(self, gpu, il, ol, tp, freq, slo_tpot_ms) -> CapacityLookup:
        # SLO = largest tabulated slo_tpot_ms <= requested; none -> source="none".
        # Length keys must be EXACT grid buckets (see length rule); non-exact resolves
        # to the conservative neighborhood-min, never a nearest-neighbor guess.
```

### SLO-bucket rule (both tables) — BLOCKER #1 fix
Pick the **largest tabulated SLO `<=` the requested SLO**. A tighter request must never
borrow a looser SLO's frontier (a 50 ms frontier admits configs with 30<P99TPOT<=50, so
using it for a 30 ms request is false-safe). **If no tabulated SLO `<=` request exists ->
reject** (`source="none"` / `prefill_rho_max -> None`). No "use smallest tabulated" fallback.

### Length rule (IL/OL) — BLOCKER #3 fix
Nearest-snapping the weighted-mean length is NOT safety-preserving (mean IL 7000 -> 6144
underestimates KV pressure -> overestimates capacity). v1:
- Prefer **per-class exact lookup** on the discrete class lengths (they already come from
  `IL_VALUES`/`OL_VALUES`), rather than collapsing to a weighted mean then snapping.
- If the pool weighted-mean must be used, resolve a non-grid `(il,ol)` to the
  **neighborhood-min** capacity over the bracketing grid buckets
  `{floor_il,ceil_il} x {floor_ol,ceil_ol}` (guaranteed conservative, no monotonicity
  assumption). No certified bracket -> reject.
- Weighted-mean pooling stays a **hardware-validation-pending approximation**; the
  longer-term gate uses the class-demand vector `D_dc,k = lambda_k * OL_k`, not
  `(sum_k lambda_k / n) * mean_OL`. Lookup does NO aggressive nearest extrapolation.

### Fallback order (locked v1) — BLOCKER #2 fix
1. `measured`: exact row, `slo_confirmed` -> `C_dc_SLO`, `eta_confirmed`.
2. `self_observed_lower_bound`: exact row, unconfirmed but `C_dc_SLO>0` (system demonstrably
   sustained that throughput while stable) -> use it as a lower bound, `eta_fallback`.
3. `lower_freq_bound`: same `(gpu,il,ol,tp,slo)`, **same TP**, lower frequency, and
   `freq_monotone_certified` -> `max_{f'<=f, confirmed} C_dc_SLO`, `eta_fallback`.
4. else `none` -> reject.

**No cross-TP fallback in v1.** `TP'<=TP` does NOT imply `C(TP')<=C(TP)` (tensor-parallel
comm, kernel efficiency, KV/comm bottleneck shift). Cross-TP borrowing, if ever added, must
consume an offline-emitted `(target_tp, support_tp, dominance_certified)` relation — the
runtime never infers dominance from numeric TP ordering.

## Config knobs (add to `SweepLLMConfig`)
```
decode_eta_confirmed: float = 0.85
decode_eta_fallback:  float = 0.80          # borrowed/unconfirmed capacity gets lower confidence
decode_gate_mode: str = "goodput"           # "goodput" (new) | "model_rho" (explicit legacy)
prefill_envelope: bool = True               # False -> legacy fixed rho_pf<=1
```
`decode_gate_mode="model_rho"` + `prefill_envelope=False` reproduces the old numbers exactly.
No large commented-out old code: the legacy path stays reachable via these flags and git
history is the record. One-line marker only: `# Legacy model-rho path (frozen-baseline repro).`
Only `sweep.py` (SWEEP strategy) is touched; other strategies keep the shared `rho<=1` gate.

## eta usage
```
lk = feas.decode_safe_capacity(...)
if lk.source == "measured":                              eta = cfg.decode_eta_confirmed
elif lk.source in {"self_observed_lower_bound","lower_freq_bound"}: eta = cfg.decode_eta_fallback
else:  return infeasible
rho_dc = demand_tps / (eta * lk.capacity_tps)

rho_pf_bound = feas.prefill_rho_max(...)                 # None -> infeasible (no implicit 1.0)
```
Sensitivity (hardware): confirmed eta in {0.80,0.85,0.90}; fallback = confirmed - 0.05.

## Exact edit points in `sweep.py`
1. `__init__`: `self.feas = FeasibilityTables(...)` once (paths via `paper_model_dir`);
   construction raises if `decode_gate_mode=="goodput"` and artifacts are unusable.
2. Pool-rho blocks (~1444-1457 and mirror ~1427-1442): in goodput mode set
   `pool_rho_dc[_pl] = (dc_rate/n_inst * demand_ol) / (eta * C)` from
   `feas.decode_safe_capacity(...)`; `source=="none"` -> `inf`. Keep model rho as
   `pool_rho_dc_model` for logging only.
3. Gate (~1521-1524):
   ```
   rho_pf_bound = feas.prefill_rho_max(gpu_pf, tp_pf, f_pf, slo_pf) if cfg.prefill_envelope else 1.0
   if rho_pf_bound is None: reject = True
   else: reject = (rho_pf > rho_pf_bound or rho_dc > 1.0 or raw_ttft < 0 or raw_tpot < 0)
   ```
   `rho_dc` already carries eta + goodput capacity, so its bound stays `1.0`.
4. `raw_tpot < 0` / `raw_ttft < 0` — this is a **prediction-validity check** (invalid /
   out-of-region regressor output), NOT an overload sentinel (TPOT can look healthy under
   overload). Decode-overload admission is decided solely by the goodput capacity gate.
5. Logging per window: `(C_dc_SLO, source, selected_slo, rho_dc_model, rho_dc_goodput,
   rho_pf_bound)` so the hardware run reports false-safe rate and borrow/none frequency.

## Calibration follow-up before wiring (small)
`calibrate_decode_capacity.py`: emit `freq_monotone_certified` per `(gpu,il,ol,tp,slo)`
(needed for `lower_freq_bound`). Also regenerate `rho_envelope.csv` for the prefill rows.

## DATA-COVERAGE BLOCKER found while building the module (read before wiring)
The existing Phase-2 decode profiling swept FREQUENCY for only ~14 (L4) / ~17 (L40S)
shapes, almost all at OL=32 (the old case1/case2 calibration). Most (il,ol,tp) shapes
have a SINGLE profiled frequency. Consequently the fail-closed gate resolves only
**14% (L4) / 23% (L40S)** of the decode DVFS grid (measured + self_observed + lower_freq);
the other ~80% correctly return `none`. Wiring the gate as the live default now would make
SWEEP reject ~80% of decode-DVFS candidates and collapse its search.

The trained decode capacity model cannot safely fill the gap: model-implied vs goodput
capacity correlates only 0.81 (L4) / **0.55 (L40S)** with **2.6x / 4.1x residual spread**,
so no single per-(gpu,tp) scale re-anchors it within safety tolerance.

**Conclusion:** the goodput methodology is correct and validated (median rho -> ~1 where
data exists), but a deployable per-frequency decode gate REQUIRES a hardware decode
freq x shape x SLO sweep. That sweep becomes the #1 hardware-campaign deliverable (it is
literally the gate's input table). Pre-hardware, the gate is code-ready but data-starved;
since simulation is dropped, there is no pre-hardware number that depends on it.

Recommended wiring posture: land the module + flags with `decode_gate_mode` default kept
at `model_rho` (explicitly labeled NOT SLO-aware, mechanics only) until the hardware table
lands; flip the default to `goodput` when coverage is sufficient. This keeps the fail-closed
gate ready without collapsing any interim mechanics run.

## Open validation (hardware campaign, not blocking wiring)
- eta sensitivity + false-safe (admitted-but-overloads) rate.
- confirm `self_observed_lower_bound` / `lower_freq_bound` configs behave conservatively.
- per-shape prefill envelope (`--per-shape`) if the "all" bucket is non-monotone.
- move mixture gate from weighted-mean to the class-demand vector.
