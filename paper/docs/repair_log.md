# SWEEP-LLM Evidence Repair Log

This document tracks each issue identified (from peer review or internal analysis),
how it was solved, and which files were changed. Ordered by execution date.

---

## ⚠️ CURRENT AUTHORITATIVE RESULTS (after 2026-05-26 baseline overhaul)

All §4.2 synthetic numbers come from the frozen baseline overhaul. Use ONLY:

- **Source of truth:** `results/paper/section42_frozen_main/summary.csv`
  (216 runs; runner = `run_section42_sweep.py`, the authoritative windowizer).
- **DO NOT** use `results/paper/section42_sweep/summary.csv` for §4.2 (pre-overhaul).
- **DO NOT** use any `/tmp` point-script numbers (wrong windowize → DualScale invalid).
- **DO NOT** cite pre-overhaul claims: 259 kJ SWEEP, 317 kJ DynamoLLM, "−18% vs Dynamo",
  "−68% vs Static" as the headline, or the hand-rolled 38.8% feasible-frontier figure.

**Headline claim (authoritative):** among 0%-modeled-violation strategies, SWEEP-LLM is
the lowest-energy operating point on all four synthetic traces, saving **9.6–31.9%
(mean 18.6%)** over the best feasible non-SWEEP baseline (Hierarchical-Disagg on
T1/T3/T4, DynamoLLM-Mono on T2). Aggregate (mean/36 runs): SWEEP 252.7 / Hierarchical
316.1 / DynamoLLM-Mono 315.7 (4.2% viol) / DualScale-Ext 254.6 (38.1% viol) /
GreenLLM-OracleDVFS 556.4 / Static 817.9 kJ.

"SLO violation" everywhere means **modeled violation under a common rho gate** (not a
request-level SLO guarantee). See the 2026-05-26 overhaul section below for details.

---

## Week 1 Tasks (2026-05-25)

---

### Task 1 — Same-pool model validation table

**Problem**

Reviewer R1 attacked: *"The simulation/modeling stack is not trustworthy because it was never validated against real hardware."*

The paper had no explicit section showing prediction error vs. measured hardware values.

**Scope clarification (added after review of Task 1)**

This task validates the **model components** (latency regressor, feasibility
classifier, power model) through grouped cross-validation against real Phase 2
hardware measurements. It does **not** constitute end-to-end simulator validation
(which would require replaying a request trace on hardware and comparing
scheduler-selected configs against oracle-measured outcomes). The paper and
`models.tex` now reflect this distinction explicitly.

**What already existed**

Real hardware benchmarks were already collected during Phase 2 profiling:
- `Phase2_Results_L40S/master_results.csv` — 1622 rows of L40S measurements
  (multiple frequencies, TP degrees, IL/OL combinations, 3 runs each)
- `Phase2_Results_L4/master_results.csv` — equivalent L4 data

Cross-validated model accuracy was already computed and stored in:
- `models_l40s/phase_model_validation_l40s.csv`
- `models_l4/phase_model_validation_l4.csv`

**Solution**

Wrote `paper/scripts/generate_validation_table.py` that reads both CSVs and outputs:
1. A human-readable summary printed to stdout
2. CSV files at `results/paper/analyses/validation_regression.csv` and
   `results/paper/analyses/validation_feasibility.csv`
3. A LaTeX table at `results/paper/analyses/validation_tables.tex`

Updated `paper/models.tex`:
- Paragraph heading changed: "Validation results." → "**Same-pool model validation.**"
- Added sentence clarifying scope: validates model components via cross-validation,
  not end-to-end simulator replay
- False-safe claim made precise: 0 false-safe at 200ms SLO; ≤3 (≤2.7%) at looser
  SLOs; `ρ_cap` guard band provides additional conservative layer in practice

**Key numbers to cite in paper**

| GPU  | Metric             | MAPE   | R²    | Spearman |
|------|--------------------|--------|-------|----------|
| L40S | P99 TTFT           | 39.0%  | 0.205 | 0.774    |
| L40S | P99 TPOT           | 22.9%  | 0.389 | 0.845    |
| L40S | Prefill power/GPU  |  5.6%  | 0.965 | 0.982    |
| L40S | Decode power/GPU   |  5.4%  | 0.967 | 0.984    |
| L4   | P99 TTFT           | 52.5%  | 0.521 | 0.832    |
| L4   | P99 TPOT           | 43.2%  | 0.574 | 0.872    |
| L4   | Prefill power/GPU  |  5.7%  | 0.940 | 0.957    |
| L4   | Decode power/GPU   |  5.3%  | 0.946 | 0.962    |

Feasibility classifier (the actual safety gate used for scheduling decisions):
- SLO=200ms: **0 false-safe** on both L40S and L4 (prefill phase)
- SLO=500ms and 1000ms: false-safe ≤ 3 out of ~110 test configs (≤ 2.7%)
- `ρ_cap` guard band provides additional conservative layer at runtime

**Known weakness**: latency regression MAPE is high (22–52%). The paper now
explicitly states the scheduler uses only the feasibility classifier (not the
regression) as the primary safety gate. Do not claim "latency model is accurate".

**Remaining gap**: end-to-end simulator validation (trace replay on hardware) would
more fully address reviewer R1's attack. Currently acknowledged as outside scope.

**Narrative for paper (§Models — already updated)**

> Same-pool model accuracy is evaluated through grouped cross-validation against
> hardware measurements from Phase 2 profiling. Power prediction: MAPE ≈ 5–6%,
> R² > 0.94. Latency regression MAPE higher (22–52%); Spearman rank correlation
> 0.77–0.87 confirms ranking is preserved. Feasibility classifier: 0 false-safe at
> SLO=200ms; ≤3 false-safe at looser SLOs. ρ_cap guard band provides additional
> conservative layer.

**Files changed**
- `paper/scripts/generate_validation_table.py` — NEW: generates tables
- `results/paper/analyses/validation_regression.csv` — NEW: generated output
- `results/paper/analyses/validation_feasibility.csv` — NEW: generated output
- `results/paper/analyses/validation_tables.tex` — NEW: generated LaTeX
- `paper/models.tex` — heading renamed; false-safe claim expanded; scope clarified

---

### Task 2 — Idle power sensitivity (code)

**Problem**

Reviewer R3 attacked: *"The energy savings depend on a 0W idle GPU assumption."*

The simulator already modeled idle power as non-zero:
- L40S: 90 W per idle GPU (4 GPUs → 360 W total when idle)
- L4: 18 W per idle GPU (8 GPUs → 144 W total when idle)

But this was hardcoded and not mentioned in the paper. Reviewers could not vary it.

**Solution**

1. Added `set_idle_power_w(l40s_idle_w, l4_idle_w)` function to
   `paper/scripts/jsep_cluster.py` — patches `GPU_SPECS` before any
   `ClusterState` is constructed.

2. Added `--idle-l40s-w` and `--idle-l4-w` CLI arguments to
   `run_section42_sweep.py` (defaults=90W, 18W). If non-default values are
   given, `set_idle_power_w()` is called before any sweep run.

**Usage for sensitivity sweep**
```bash
# Baseline (current default: 90W L40S, 18W L4)
.venv/bin/python run_section42_sweep.py --output-dir results/paper/sensitivity_idle/baseline ...

# Zero idle (best-case power gating)
.venv/bin/python run_section42_sweep.py --idle-l40s-w 0 --idle-l4-w 0 \
  --output-dir results/paper/sensitivity_idle/zero ...

# Conservative idle (lower than default)
.venv/bin/python run_section42_sweep.py --idle-l40s-w 50 --idle-l4-w 10 \
  --output-dir results/paper/sensitivity_idle/low ...

# Pessimistic idle (higher — e.g., full node power stays up)
.venv/bin/python run_section42_sweep.py --idle-l40s-w 150 --idle-l4-w 30 \
  --output-dir results/paper/sensitivity_idle/high ...
```

**Narrative for paper (§Sensitivity)**

> Our default energy metric accounts for idle GPU power: L40S GPUs not assigned to
> an active instance consume 90 W each; L4 GPUs consume 18 W each.
> To test sensitivity to power-gating assumptions, we repeat the sweep with idle
> power ranging from 0 W (ideal gating) to 150 W/30 W (pessimistic).
> [Table X] shows that scheduler rankings are preserved across all scenarios, and
> SWEEP-LLM's advantage over DynamoLLM grows under pessimistic idle power because
> SWEEP actively power-gates idle GPUs while DynamoLLM maintains larger active pools.

**Files changed**
- `paper/scripts/jsep_cluster.py` — added `set_idle_power_w()` function
- `run_section42_sweep.py` — added `--idle-l40s-w`, `--idle-l4-w` args;
  import of `set_idle_power_w, GPU_SPECS`

---

### Task 3 — Fix overclaims in Abstract and Introduction

**Problem**

Reviewer R2 attacked: *"The paper claims a deployed production system but the
evaluation is purely simulation."*

The original text said:
- Abstract: "reduces serving energy by 82%" (no qualifier)
- Introduction §contributions: "measurement-driven evaluation with real profiling data"
- `evaluation_rewrite.tex`: multiple instances of "$0\%$ SLO violations" with no
  simulation qualifier, and "reduces total energy by 82%" missing "modeled GPU-side"

**Solution (initial + residual-overclaim pass)**

**Abstract** (`paper/abstract.tex`):
- Added explicit framing: "We evaluate SWEEP-LLM via a measurement-calibrated
  simulation framework: same-pool models are validated against hardware data;
  cross-pool KV-transfer cost is swept analytically."
- Changed "reduces serving energy" → "reduces **modeled GPU-side** energy"
- Changed "$0\%$ SLO violations" → "$0\%$ **modeled** SLO violations"
- Added model accuracy numbers ("power MAPE ≈ 5%, feasibility classifier precision
  ≥ 94%") to establish trust immediately

**Introduction** (`paper/Introduction.tex`, line 193 contributions bullet):
- Replaced the single "measurement-driven evaluation" sentence with three sentences:
  1. Explicitly names the evaluation methodology ("measurement-calibrated simulation",
     same-pool validation numbers, analytical KV sweep)
  2. Synthetic trace results with "modeled GPU-side energy" qualifier +
     "$0\%$ **modeled** SLO violations"
  3. Production trace results with same qualifier

**`evaluation_rewrite.tex`** (residual overclaim pass):
- Line 295: "reduces total energy" → "reduces **modeled GPU-side energy**"
- Line 303: "SWEEP-LLM achieves $0\%$ SLO violations" →
  "**In simulation**, SWEEP-LLM achieves $0\%$ **modeled** SLO violations"
- Lines 372, 403, 411: "$0\%$ SLO violations" → "$0\%$ **modeled** SLO violations"
- Line 436: "SWEEP-LLM violates $0\%$..." → "**In simulation**, SWEEP-LLM achieves
  $0\%$ **modeled** SLO violations on T1--T4"
- Line 444: "to guarantee $0\%$ SLO violations" → "to **target** $0\%$
  **modeled** SLO violations"

**Key principle applied**: every SLO violation claim now carries "modeled" qualifier;
every energy percentage now carries "modeled GPU-side energy" qualifier; simulation
framing is explicit at both abstract and body level.

**Files changed**
- `paper/abstract.tex` — rewritten abstract; "$0\%$ modeled SLO violations" added
- `paper/Introduction.tex` — contributions bullet rewritten; "$0\%$ modeled" added
- `paper/models.tex` — "Validation results" → "Same-pool model validation" heading;
  false-safe claim expanded to cover all SLO levels (see Task 1 update below)
- `paper/evaluation_rewrite.tex` — 6 residual "$0\%$ SLO violations" fixed;
  "reduces total energy" → "reduces modeled GPU-side energy" on line 295

---

---

## Week 2 Tasks (2026-05-25)

---

### Task 4 — Route-share logging in per-window logs

**Problem**

No per-window record of which route (AA/AB/BA/BB) was used. Needed for:
(a) showing ROUTE_AB fraction under different τ_kv assumptions, and
(b) the oracle comparison route summary.

**Solution**

Added two helpers to `run_section42_sweep.py`:
- `_extract_route_fracs(config, win_reqs)`: reads `cfg["routes"]` (dict from class_id → "A->B" string emitted by SweepLLMStrategy), weights by request count, returns `{route_aa_frac, route_ab_frac, route_ba_frac, route_bb_frac}`. For strategies that don't emit routes, all fields are empty string.
- `_req_class_id(req)`: replicates SweepLLMStrategy's class ID logic (long_il≥512, long_ol≥256).

These are now appended to each per-window log row.

**Files changed**
- `run_section42_sweep.py` — helpers added; `_extract_route_fracs` call in log dict

---

### Task 5 — Oracle comparison (search quality validation)

**Problem**

Reviewer R4 attacked: *"State-pruned search is a heuristic with no approximation guarantee. The paper never shows how far it is from optimal."*

**What the oracle does**

`paper/scripts/oracle_comparison.py` implements an exhaustive oracle using SWEEP's own
`_evaluate_candidate` scoring function (same feasibility classifier, same power model).
The only difference: no state-guided pruning. Oracle enumerates:
- All valid PoolBundle combinations for L40S × L4 (≈25 × 70 = 1750 bundle pairs)
- 5 representative frequency levels per pool (or all 10 with `--full-freqs`)
- 4 "single-route-for-all-classes" options (AA, AB, BA, BB)
- Total: ~44,000 candidates per window (vs. 9–2400 for SWEEP)

**Key results (SLO=500ms, τ_kv=16µs/tok, full 10 frequencies) — FINAL after DECODE_HEAVY fix**

| Window | State | Rate | SWEEP (J) | Oracle (J) | Gap% |
|---|---|---|---|---|---|
| BOTH_LOW_balanced | BOTH_LOW | 6 rps | 828.5 | 827.5 | **0.1%** |
| PREFILL_HEAVY | PREFILL_HEAVY | 20 rps | 1638.0 | 1534.0 | **6.8%** |
| BOTH_HEAVY | BOTH_HEAVY | 30 rps | 2741.5 | 2612.0 | **5.0%** |
| PREFILL_HEAVY_hi | PREFILL_HEAVY | 25 rps | 2131.5 | 1897.5 | 12.3% |
| BOTH_LOW_pf_bias | BOTH_LOW | 5 rps | 828.5 | 660.5 | 25.4% |
| BOTH_LOW_dc_bias | BOTH_LOW | 5 rps | 868.0 | 674.0 | 28.8% |
| DECODE_HEAVY | DECODE_HEAVY | 30 rps | **2295.0** | 2295.0 | **0.0%** |

Mean gap: **11.2%** (was 15.8% before DECODE_HEAVY fix). Max gap: 28.8% (BOTH_LOW_dc_bias).
All windows: SWEEP meets SLO whenever oracle meets SLO (0 SLO-miss discrepancies).
Decision time: SWEEP 0.3–1.8s vs oracle 9–14s (SWEEP is 5–35× faster).

**Important technical finding**

The state classifier normalizes load by cluster reference capacity:
- ref_cap_prefill = 23,937 tok/s,  ref_cap_decode = 16,960 tok/s
- BOTH_LOW condition: (D_pf/ref_pf + D_dc/ref_dc) < 0.7
- At Azure trace rates (10–20 rps), ALL windows are BOTH_LOW or occasionally BOTH_HEAVY
- PREFILL_HEAVY exits BOTH_LOW at ≥13.9 rps for long_short workloads
- DECODE_HEAVY exits BOTH_LOW at ≥22.2 rps for short_long workloads

**Narrative for paper (§Search quality) — UPDATED after DECODE_HEAVY fix**

> SWEEP-LLM evaluates 9–2400 candidates per window (depending on state) vs. ≈44K
> for an exhaustive oracle using the same scoring function. In PREFILL_HEAVY and
> BOTH_HEAVY states — where SLO constraints are most binding — the energy gap is
> 5–12%. For DECODE_HEAVY and balanced BOTH_LOW windows, SWEEP matches the oracle
> within 0.1%. The remaining gaps (25–29%) occur in light-but-biased BOTH_LOW
> windows, where the oracle exploits partial GPU allocations (1–2 active GPUs)
> outside SWEEP's BOTH_LOW candidate budget; the absolute energy difference in
> these low-load regimes is 168–194J. In all cases, SWEEP never misses a feasible
> configuration that the oracle finds: whenever the oracle meets SLO, SWEEP also
> meets SLO.

**Files changed**
- `paper/scripts/oracle_comparison.py` — NEW: exhaustive oracle script
- `results/paper/analyses/oracle_comparison.csv` — generated output

---

### Task 5 Fix — DECODE_HEAVY candidate pool expansion (32% gap → 0%)

**Problem identified by Task 5 oracle comparison**

The oracle found that DECODE_HEAVY at 30 rps was 32% cheaper than SWEEP's answer.
Root cause analysis (via exhaustive per-route debugging) revealed:

1. **`_admissible_routes` excluded ROUTE_BA and ROUTE_AA for decode_critical classes**
   (`output_len ≥ theta_ol = 512`). The oracle's optimal uses **ROUTE_BA** (L4 pf +
   L40S dc): L4 handles light prefill (IL=32, trivial load), L40S dedicates all GPUs
   to decode. This avoids over-provisioning and minimizes idle power on both pools.
   Oracle optimal: `A=(tp=1, n_pf=0, n_dc=2), B=(tp=1, n_pf=1, n_dc=0)`, E=2295.0J.

2. **`_prune_bundle_pairs` required `bundle_b.n_dc > 0`**, blocking the ROUTE_BA
   optimal `(B=(1,1,0))` from ever reaching the route DFS.

**Fix (all in `sweep_llm_scheduler.py`)**

1. **`_admissible_routes` DECODE_HEAVY** → return `ALL_ROUTES` for all classes
   (was: ROUTE_AB/ROUTE_BB only for decode_critical).

2. **`_decode_heavy_power_gate_anchors`** — added **Family 3 (ROUTE_BA anchors)**:
   - `A=(tp=1, n_pf=0, n_dc=n)` for n=1..4, `B=(tp=tp_b, n_pf=tp_b, n_dc=0)`
   - These bypass the `bundle_b.n_dc > 0` pruning
   - 16 new Family 3 anchors (4 A-variants × 4 B-variants)

3. **`_select_frequency_pairs` DECODE_HEAVY** non-burst secondary: `low_a` → `freqs_a`
   (full L40S range), matching PREFILL_HEAVY's analogous fix.

**Result**: DECODE_HEAVY gap: **32.0% → 0.0%**. Mean gap across all 7 windows:
15.8% → **11.2%**. SWEEP now matches the oracle exactly on DECODE_HEAVY.

**Files changed**
- `paper/scripts/sweep_llm_scheduler.py`:
  - `_admissible_routes`: DECODE_HEAVY now returns `ALL_ROUTES`
  - `_decode_heavy_power_gate_anchors`: added Family 3 (ROUTE_BA: L40S dc-only + L4 pf-only)
  - `_select_frequency_pairs`: DECODE_HEAVY non-burst secondary uses full `freqs_a`
  - `_construct_candidates`: calls `_decode_heavy_power_gate_anchors` (already wired)
- `paper/evaluation_rewrite.tex`: table updated (DECODE_HEAVY 3029.5→2295.0J, 32%→0%); narrative updated
- `results/paper/analyses/oracle_comparison.csv`: regenerated

---

---

## Week 3 Tasks (2026-05-25)

---

### Task 6 — HierarchicalDisaggStrategy (routing-aware sequential baseline)

**Problem**

Reviewer R2 attacked: *"The main win may be just that SWEEP-LLM has access
to cross-pool routing while DynamoLLM does not. A routing-aware sequential
baseline would isolate whether joint search adds value beyond routing."*

**Design (v2 — revised for baseline fairness)**

`HierarchicalDisaggStrategy` gets the same **per-class** routes as SWEEP
(AA/AB/BA/BB per request class) and uses fully energy-greedy decisions at
each sequential stage:

1. **Route** (stage 1): enumerate all **4^n_classes per-class route
   combinations** (≤256 for 4 classes); evaluate each at TP=1, max freq,
   max capacity; pick the combination with minimum energy.
2. **TP** (stage 2): given selected routes, enumerate all (tp_A, tp_B)
   pairs; pick the **energy-minimum feasible** pair at max freq + max capacity.
   *(v1 used "first feasible ascending" — too weak; changed to energy-minimum.)*
3. **DVFS** (stage 3): given routes + TP, enumerate all (freq_A, freq_B);
   pick energy-minimum feasible at max capacity.
4. **Capacity** (stage 4): given routes + TP + freq, enumerate all valid
   bundles consistent with pool roles and selected TP; pick energy-minimum.

Helper `_pool_roles(routes)` determines which phases each pool handles across
all classes; `_bundle_for_roles(pool, tp, needs_pf, needs_dc)` builds the
max-capacity bundle accordingly (handles mixed monolithic+disagg routing).

**Why v1 was unfair (and what was fixed)**

v1 had two problems:
- Uniform routing: tried only 4 routes applied to ALL classes. SWEEP does
  per-class routing (e.g., long_short → AB, short_long → BA simultaneously).
  v1 artificially limited routing granularity vs. SWEEP.
- "First feasible TP": stopped at the first ascending (tp_A, tp_B) pair that
  passed SLO. This is a weak heuristic that doesn't minimize energy at the TP
  stage, making the baseline appear artificially poor.

v2 fixes both: per-class route enumeration (up to 256 combos) and
energy-minimum TP selection.

**Key design properties**
- No bundle stability, no state-guided pruning, no dual ideal/fast search
- Falls back to `_fallback()` if all stages produce no feasible result
- Evaluation cost: ~256 (route) + 12 (TP) + 100 (freq) + ~500 (capacity) ≈
  870 evaluations/window — less than oracle (44K), more than original Hier (16+100+1750)

**Narrative for paper (§Baselines — already updated)**

> Hierarchical-Disagg receives the same per-class routing options as SWEEP-LLM
> (4^n_classes per-class route assignments) and makes energy-greedy decisions
> sequentially: route → TP → DVFS → capacity.
> Comparing SWEEP vs. Hierarchical-Disagg isolates whether the advantage comes
> from routing access or from joint optimization.

**Files changed**
- `paper/scripts/sweep_llm_scheduler.py` — `HierarchicalDisaggStrategy` rewritten
  (v2): per-class route enumeration, energy-minimum TP, `_pool_roles()`,
  `_bundle_for_roles()`, `_eval_max_capacity()` helpers; factory unchanged
- `run_section42_sweep.py` — unchanged (already registered)
- `paper/evaluation_rewrite.tex` — Hierarchical-Disagg baseline description
  updated to describe per-class routing and energy-greedy TP selection

---

---

### Task 7 — section42_sweep feasibility model bug + rerun

**Problem identified 2026-05-25**

Comparing per-window logs of section42_sweep (run 2026-05-23, pre-fix) vs.
hier_disagg_full (run 2026-05-25, post-fix) revealed that the DynamoLLM strategy
in the old sweep accepted a physically impossible configuration at window 51
(T1, arrival rate = 37 rps):

```
section42 (old):  l4_tp=1, l4_active=2, power=126.6W, slo_met=true
hier_disagg_full: l4_tp=4, l4_active=2, power=424.0W, slo_met=true  ← correct
```

TP=1 with 2 physical GPUs cannot serve 37 rps of Mistral-7B. The correct
configuration is TP=4 (using 2 physical GPU pairs = 8 virtual GPUs) at ~424W.

**Root cause** (confirmed by diagnostic, 2026-05-25): rho model OOD extrapolation
in the `_try_monolithic_consolidation` path.

`DynamoLLMStrategy` finds L4-only configurations via `_try_monolithic_consolidation`,
which bypasses the SLO classifier (`require_safe=False`) and instead accepts
configurations where `_pool_max_rho(result) < rho_cap` (rho_cap=0.85). The bug:
- **Old EnergyScheduler**: at TP=1, 18.5 rps/instance (37 rps total, 2 L4 GPUs),
  extrapolated rho to a value below 0.85 (OOD — far outside training range).
  Result: consolidation accepted L4 TP=1 pf=1 dc=1, returned power=126.6W,
  forced slo_met=True. This configuration is physically impossible (rho=5.3×
  in the correct model).
- **New EnergyScheduler**: correctly predicts rho=5.33 for same input →
  `rho_cap` check blocks it → consolidation finds no L4 option feasible →
  main search selects L40S ROUTE_AA at 677W.

Diagnostic sanity check (post-fix code, T1 window 51, rate=37 rps, IL=512):
```
L4 TP=1, 18.5 rps/inst:  rho=5.33, is_safe=False, ttft_p99=25909ms  (old: ~0.2→accepted)
L4 TP=1, 4.625 rps/inst: rho=1.33, is_safe=False, ttft_p99=1254ms
L4 TP=4, 18.5 rps/inst:  rho=3.00, is_safe=False, ttft_p99=5340ms
Selected (post-fix): L40S ROUTE_AA, n_pf=3, n_dc=1, power=677W
Old result (pre-fix): L4 TP=1, n_pf=1, n_dc=1, power=127W (wrong, rho>>1)
```

Note: TP=4 candidates ARE present in the search (not a coverage gap). The bug is
in the rho model's OOD behavior, not candidate generation. Both TP=1 and TP=4 L4
options are infeasible at 37 rps; the EnergyScheduler now correctly rejects all.

**Impact on paper**

All §4.2 DynamoLLM numbers from section42_sweep are wrong:
- Old: 0% violations at SLO=200/500/1000ms for T1–T4 (incorrect)
- Expected from hier_disagg_full: 3.3–9.2% violations (correct)
- Energy values for high-load DynamoLLM windows are underestimated

**Fix**

Rerun `run_section42_sweep.py` with current (fixed) code for all 4 strategies,
4 traces, 3 SLOs, 3 τkv (144 runs). Old summary backed up at
`results/paper/section42_sweep/summary.csv.pre_fix_backup`.

```bash
.venv/bin/python3 run_section42_sweep.py \
  --strategies static_disagg,greenllm,dynamollm,sweep_llm \
  --traces T1,T2,T3,T4 --slos 200,500,1000 --tau-kvs 5,16,41 \
  --output-dir results/paper/section42_sweep --force
```

**Status**: ✅ COMPLETE (2026-05-25 ~20:15 CEST). All 144 log files written; `summary.csv`
rebuilt via `rebuild_summary_from_logs.py`; §4.2 numbers and Introduction.tex updated.

**Final confirmed results (post-fix, mean across all SLO × τkv conditions)**

| Trace | DynamoLLM E | viol% | SWEEP-LLM E | viol% | SWEEP vs DynamoLLM |
|-------|------------|-------|-------------|-------|--------------------|
| T1 | 196 054 J | 9.17% | 214 888 J | 0.00% | +9.6% (SWEEP worse, but 0 violations) |
| T2 | 242 814 J | 0.00% | 204 220 J | 0.00% | −15.9% (SWEEP better) |
| T3 | 346 695 J | 4.17% | 333 555 J | 0.00% | −3.8% (SWEEP better) |
| T4 | 481 061 J | 3.33% | 284 571 J | 0.00% | −40.9% (SWEEP better) |
| **Mean** | **316 656 J** | **4.2%** | **259 309 J** | **0.00%** | **−18.1%** |

Mean energy ladder (across 36 runs per strategy):
- Static-disagg: 818 kJ
- GreenLLM: 556 kJ (−32% vs Static)
- DynamoLLM: 317 kJ (−43% vs GreenLLM)
- SWEEP-LLM: 259 kJ (−18% vs DynamoLLM, **−68% vs Static**)

**⚠️ DualScale NOT rerun — pre-fix logs (May 23), numbers INVALID**

The Task 7 rerun command specified `--strategies static_disagg,greenllm,dynamollm,sweep_llm`
and did not include DualScale. The 36 DualScale log files in `section42_sweep/logs/` are all
from May 23 13:00–13:14 (pre-fix scheduler).

Pre-fix DualScale numbers from summary.csv (DO NOT CITE — affected by OOD ttft=-1.0 bug):
- Mean across all SLO×kv: 159 kJ (misleadingly low)
- T1 SLO=200: 341 kJ, **10.83% violations**; SLO=500: 86 kJ, 0%; SLO=1000: 84 kJ, 0%
- T4 SLO=200: 396 kJ, **2.5% violations**; SLO=500: 76 kJ, 0%; SLO=1000: 68 kJ, 0%

The dramatic SLO-sensitivity (4× difference between SLO=200 and SLO=500 on T1) is the
same OOD ttft=-1.0 artifact that affected SWEEP and DynamoLLM in the pre-fix run. At
SLO=500/1000, cheap L4 configs with TTFT=-1.0 (OOD) pass the SLO check erroneously.

**Action needed**: rerun DualScale with the fixed scheduler before citing any DualScale
numbers in §4.2.

**Violation structural analysis (T1 DynamoLLM, SLO=500ms log)**

The 11 violated windows (9.17%) are at windows 52–71, all PREFILL_HEAVY state,
arrival rate 32–36 rps. All have:
- Consolidation: rho >> 0.85 → blocked correctly by corrected model
- Main search: ROUTE_AA (L40S monolithic) and ROUTE_BB (L4 monolithic) both fail
  SLO classifier at 32–36 rps with IL=512
- Emergency fallback: selects L4 TP=1, n_pf=4, n_dc=1 with slo_met=False

Root cause: DynamoLLM is restricted to ROUTE_AA and ROUTE_BB (monolithic only).
At 32–36 rps, SWEEP handles this by ROUTE_AA with n_pf=3, n_dc=1 (disaggregated
prefill+decode capacity split). DynamoLLM cannot use this split-role allocation.
The violations are structural (routing restriction), not a feasibility model issue.

**Sanity Check 2: mono_consolidation_rho_cap=0.0 ablation (2026-05-25)**

Ran `run_rhocap_ablation.py`: DynamoLLM with rho_cap=0.0 (consolidation disabled)
vs. default rho_cap=0.85, for T1–T4, SLO=500ms, τkv=16 µs/tok.

| Trace | Baseline E | Ablation E | Delta E | Delta Viol |
|-------|-----------|-----------|---------|-----------|
| T1 | 196 054 J | 217 852 J | **+11.1%** | 0.00% |
| T2 | 242 814 J | 259 546 J | **+6.9%** | 0.00% |
| T3 | 346 695 J | 345 972 J | −0.2% | 0.00% |
| T4 | 481 061 J | 481 061 J | 0.0% | 0.00% |

Findings:
1. **SLO-safe**: Disabling consolidation causes 0 additional violations on all traces.
2. **Actively useful for high-load traces**: T1 +11.1%, T2 +6.9% energy increase when
   disabled. The consolidation correctly identifies moderate-load windows where L4 is
   underloaded (rho<0.85) but the TTFT model (used in require_safe=True path) is
   conservative; the rho-only gate allows cheaper L4-only decisions there.
3. **Inactive for lighter traces**: T3 (−0.2%, rounding noise) and T4 (0.0%) are
   unaffected — consolidation is not triggered for these traces' load profiles.

Interpretation: The rho-only consolidation path is not a "false-safe" bypass.
The corrected rho model correctly gates it: overloaded windows (rho>>0.85) are
blocked, and the ones that pass (rho<0.85) are genuinely underloaded. This
resolves the pre-fix bug cleanly: the old model's OOD rho values allowed
overloaded windows to pass (rho~0.2 at 18.5 rps/inst); the new model correctly
gives rho=5.33, blocking them.

**Files changed**
- `results/paper/section42_sweep/summary.csv.pre_fix_backup` — backup of old results
- `results/paper/section42_sweep/logs/` — 144 logs being overwritten with corrected runs
- `/tmp/rhocap_ablation.csv` — ablation results (8 rows)
- `run_rhocap_ablation.py` — ablation driver script

---

---

## Status & Pending

### Assessment (post-review, 2026-05-25)

| Task | Description | Code | Results | Paper text | Reviewer-ready? |
|------|-------------|------|---------|------------|-----------------|
| Task 1 | Same-pool **model** validation table | ✅ | ✅ CSV + LaTeX | ✅ (heading + false-safe fixed) | **Partial** — model CV only |
| Task 2 | Idle power sensitivity | ✅ code | ✅ sweep done (4 settings × 12 runs) | ❌ sensitivity paragraph not written | **Partial** |
| Task 3 | Fix overclaims | ✅ | — | ✅ full pass done | **Yes** |
| Task 4 | Route-share logging | ✅ | ✅ | — | ✅ |
| Task 5 | Oracle comparison + DECODE_HEAVY fix | ✅ | ✅ FINAL | ✅ (DH gap 32%→0%, mean 11.2%) | **Yes** |
| Task 6 | HierarchicalDisaggStrategy | ✅ code v2 | ✅ 108 runs done | ❌ result numbers not yet in paper | **Partial** |
| Task 7 | section42_sweep feasibility bug rerun | ✅ | ✅ 144 runs (4 strategies); ⚠️ DualScale pre-fix (36 stale logs) | ✅ §4.2 + Introduction updated (DualScale excluded) | **Partial** |

### Done ✅ (code + experiments + paper text)
- [x] Task 1: Model validation table (script + LaTeX + CSV); `models.tex` heading +
  false-safe claim revised; scope clarified
- [x] Task 2: Idle power sensitivity code + **experiments complete** (4 settings: zero/low/baseline/high)
  - Key result: SWEEP and Hier-Disagg 100% insensitive (full pool deactivation)
  - DynamoLLM grows linearly with idle power on T3/T4 (unused pool stays on)
  - Output: `results/paper/analyses/idle_sensitivity.csv`; per-setting dirs `results/paper/idle_sensitivity_{zero,low,baseline,high}/`
- [x] Task 3: Full overclaim pass — "modeled GPU-side energy" and "modeled SLO
  violations" in abstract, Introduction, all 6 instances in evaluation_rewrite.tex
- [x] Task 4: Route-share logging in per-window logs
- [x] Task 5: Oracle comparison + DECODE_HEAVY gap fixed (32%→0%, mean 11.2%)
- [x] Task 6: HierarchicalDisaggStrategy code (v2) + **full sweep complete** (108 runs)
  - Key results (T1–T4, SLOs 200/500/1000, τkv 5/16/41):
    - SWEEP vs Hier-Disagg: mean **19.2%** savings (T1=31.9%, T2=20.2%, T3=11.8%, T4=9.6%)
    - Savings **constant** across all SLOs and τkv (SLO constraint non-binding)
    - SWEEP and Hier-Disagg: 0% violations; DynamoLLM: 3.3–9.2% violations
  - Output: `results/paper/hier_disagg_full/summary.csv`
- [x] Task 6: Baselines paragraph in paper updated (description of v2 strategy)

- [x] Task 7 Sanity Check 2: rho_cap=0.0 ablation **complete** (2026-05-25)
  - T1 +11.1%, T2 +6.9% energy when disabled; 0% delta violations across all traces
  - Consolidation is SLO-safe and actively useful for high-load traces
  - Corrected rho model correctly gates overloaded windows (rho=5.33 → blocked)
- [x] Task 7: section42_sweep rerun **complete** (2026-05-25 ~20:15 CEST)
  - All 144 runs done; summary.csv rebuilt; §4.2 + Introduction.tex updated with final numbers
  - SWEEP: 259 kJ mean (−18% vs DynamoLLM, −68% vs Static); 0% violations all traces
  - DynamoLLM: 317 kJ mean; 9.2%/4.2%/3.3% structural violations on T1/T3/T4

### Pending — experiments
- [ ] **DualScale section42_sweep rerun** — add `dualscale` to the rerun command and re-execute
  (pre-fix logs in `section42_sweep/logs/T*_dualscale_*.csv` are stale and affected by OOD bug)
  ```bash
  python paper/scripts/run_section42_sweep.py \
    --strategies dualscale \
    --traces T1,T2,T3,T4 --slos 200,500,1000 --tau-kvs 5,16,41 \
    --output-dir results/paper/section42_sweep --force
  ```
  Then rebuild summary.csv and regenerate figures.
- [ ] Full v5 sweep across all 6 Azure traces × 3 SLOs (hierarchical_disagg included)

### Pending — paper writing
- [x] Task 2: Idle power sensitivity paragraph written (2026-05-25) — `\paragraph{Idle power sensitivity.}` added at end of §SLO Sensitivity in `evaluation_rewrite.tex`
  - Key claims: SWEEP/Hier-Disagg 100% insensitive (full pool deactivation); DynamoLLM T4 grows 29%→41% SWEEP advantage from zero→baseline idle; T3 grows 3.2%→4.2%
- [x] Task 6: Hierarchical-Disagg result numbers added to §Ablation (2026-05-25) — "Joint search vs. sequential control" paragraph expanded with 108-run comparison
  - SWEEP savings vs Hier-Disagg: T1=31.9%, T2=20.2%, T3=11.8%, T4=9.6%, mean=19.2%; constant across all SLO/τkv; both 0% violations
- [x] Regenerate §4.2 figures — done (2026-05-25): `energy_by_strategy_per_trace.pdf` + `tau_kv_sensitivity.pdf` in `paper/scripts/figures/`

### Remaining evidence gap (acknowledged)
- **Task 1 scope**: only model cross-validation, not end-to-end simulator replay
  against a real trace.

### Open decisions
- DualScale `_DS_COARSE_EVERY=4` (20s refresh) vs paper's "few minutes": **RESOLVED**
  in the 2026-05-26 overhaul below (now 5-min faithful DualScale-Ext).
- BOTH_LOW biased windows (25–29% oracle gap): acceptable given tiny absolute energy
  difference (168–194J out of 660–674J oracle baseline)

---

## Baseline Faithfulness Overhaul (2026-05-26)

**Trigger.** While fixing a DECODE_HEAVY search bug, SWEEP failed to beat DualScale.
Audited each baseline against its source paper and found several were implemented
unfairly (some too strong, some with inconsistent SLO accounting). Overhauled all
baselines for fairness, unified the SLO gate, and froze a consistent result set.
**Supersedes the pre-overhaul §4.2 numbers** (Static 818 / GreenLLM 556 /
DynamoLLM 317 / SWEEP 259; "DynamoLLM 8.6% < SWEEP on T1").

### Code restructure (Step-1 refactor only — no logic change)
- Split monolithic `paper/scripts/sweep_llm_scheduler.py` (3016 lines) into a
  `paper/scripts/schedulers/` package: `common`, `sweep`, `baselines_static`,
  `baselines_dynamollm`, `baselines_dualscale`, `baselines_hierarchical`,
  `ablations`, `factory`. `sweep_llm_scheduler.py` is now a thin re-export wrapper.
- Verified bitwise-identical behavior (py_compile, import of all 42 symbols,
  SWEEP+DualScale per-trace diff, multi-strategy smoke). Inheritance unchanged.
- Backup kept at `sweep_llm_scheduler.py.bak`.

### DualScale → **DualScale-Ext** (faithful two-tier) — `baselines_dualscale.py`
- **Problem:** old impl re-provisioned every 20s (`_DS_COARSE_EVERY=4`) + load-change
  retrigger + current-window lookahead + fine-tier fallback into full search → far
  more agile than DualScale's 5-min coarse provisioning. Unfairly strong; beat SWEEP.
- **Fix:** coarse tier every 5 min (`_DS_COARSE_EVERY=60`); provisions for PEAK over
  trailing lookback (peak-from-PAST history, no lookahead, ×1.05 margin); placement
  locked between triggers; fine tier = per-window oracle freq sweep, max-freq fallback
  (no re-provisioning). Coarse-counter off-by-one fixed. `coarse_every`/`lookback_windows`
  are constructor params. Extends to heterogeneous AA/AB/BA/BB routes → call it
  DualScale-Ext, not DualScale.

### DynamoLLM → **DynamoLLM-Mono** (Monolithic-Joint) — `baselines_dynamollm.py`
- **Problem:** not faithful DynamoLLM (no hierarchical multi-timescale control /
  request-type pools / fragmentation handling). `_try_monolithic_consolidation` applied
  a ROUTE_BB-only, rho-gated shortcut that force-set `slo_met=True` from a
  `require_safe=False` eval — masking violations, asymmetrically favoring L4.
- **Fix:** disabled the consolidation shortcut (main joint search already explores
  AA/BB); honest docstring. Class/factory rename deferred to avoid ripple (DualScale
  inherits it). Paper label = DynamoLLM-Mono.

### GreenLLM → **GreenLLM-OracleDVFS** — `baselines_static.py`
- Not the faithful TPS heuristic — it's an oracle exhaustive (freq_a,freq_b) sweep
  (generous baseline, conservative for SWEEP). Docstring relabeled; fallback-path
  `slo_met` switched to the common rho-gate.

### Unified SLO gate (decision: Option A) — `common.py`, `baselines_static.py`, `baselines_dualscale.py`
- **Problem:** the 5 strategies used THREE different SLO gates: rho-gate (SWEEP,
  Dynamo, GreenLLM main), is_safe classifier (DualScale, after a prior change), latency
  regression (StaticDisagg, GreenLLM fallback). Not comparable.
- **Fix:** ALL strategies use the common rho-gate. StaticDisagg + GreenLLM-fallback
  `slo_met` now from rho-gate. DualScale strict is_safe is opt-in
  (`strict_classifier_gate=False` default) → appendix sensitivity only.
- **Paper wording:** "modeled violation under a common rho gate", NOT "request-level
  SLO guaranteed". Latency regressor (cv_mape~8000%) is a stated limitation.

### Frozen results — authoritative source = `run_section42_sweep.py`
- 216-run frozen main sweep → `results/paper/section42_frozen_main/summary.csv`
  (static_disagg, greenllm, dynamollm, dualscale, hierarchical_disagg, sweep_llm ×
  T1–T4 × SLO {200,500,1000} × τkv {5,16,41}; 0 failed).
- **Aggregate (mean/36 runs):** Static 817.9 / GreenLLM 556.4 / DynamoLLM-Mono 315.7
  (4.2% viol) / Hierarchical-Disagg 316.1 (0%) / DualScale-Ext 254.6 (38.1% viol) /
  **SWEEP 252.7 (0%)**.
- **Table A — feasible frontier (SLO=500, τkv=16, TPOT=200):** among 0%-viol strategies
  SWEEP is lowest-energy on all 4 traces; **mean 18.6% (9.6–31.9%)** below best feasible
  non-SWEEP (Hierarchical on T1/T3/T4, DynamoLLM-Mono on T2). (Earlier hand-rolled 38.8%
  was wrong — it omitted Hierarchical-Disagg, the real competitor.)
- DualScale-Ext's only feasible point is T4 (others 39–68% viol); its low aggregate
  energy is an under-provisioning artifact, not a valid saving.

### Windowing gotcha (important)
- Authoritative path = the runner's `windowize()` (absolute-time buckets from t=0,
  includes empty windows, current_time=i·W). Hand-rolled `/tmp` point scripts diverged
  (skip-empty, trace-relative time). SWEEP robust (214.9 vs 215.5, ~0.3%); **DualScale
  chaotically sensitive** (166.6/44.2% via runner vs 205.1/35.8% via `/tmp` on T1@5min,
  because the 60-window locked placement amplifies tiny windowing shifts). Confirmed by
  replicating `windowize()` exactly.
- ⇒ Table A (from runner) is valid; **cadence Table B and all `/tmp` DualScale numbers
  are invalid** — regenerate through the runner.

### Derived tables (scripted — no hand-computing)
- `paper/scripts/make_frozen_synthetic_tables.py` reads the frozen sweep(s) and emits
  to `results/paper/section42_frozen_main/derived/`: `raw_energy_violation.csv`,
  `aggregate.csv`, `table_a_feasible_frontier.csv`, `sweep_route_share.csv`,
  `table_b_cadence.csv`.
- [x] **Cadence Table B regenerated via the runner** (DualScale-Ext subclasses
  `DualScaleExt{20s,1Min,2Min}`, `_DS_COARSE_EVERY` ∈ {4,12,24}; 5-min = `dualscale`
  from frozen main). Data: `results/paper/dualscale_cadence_rhogate/summary.csv`.
  Result: **no DualScale-Ext cadence reaches 0% modeled violation on any trace**
  (min 0.8% at T4@5min); SWEEP is the sole 0%-viol point; every cheaper-than-SWEEP
  DualScale point carries 12.5–68% violations. (`sweep_llm` in the cadence run matches
  the frozen main exactly — same windowizer.)

- [x] **180-run frozen ablation sweep** → `results/paper/section42_frozen_ablation/`
  (sweep_llm + no_routing/no_dvfs/no_tp/no_capacity × T1–4 × 3 SLO × 3 τkv; 0 failed).
  Leave-one-out vs full SWEEP (mean/36, all 0% viol): −Routing +3% (1.03), −DVFS +4%
  (1.04), −Capacity +6% (1.06), −TP +11% (1.11). No knob redundant; TP largest. Per-trace
  in `derived/ablation_per_trace.csv` (−TP hits T4 1.36×, −Routing hits T1 1.08×).

### Evaluation rewrite (2026-05-27, evaluation_rewrite.tex)
- [x] Synthetic results → feasible-frontier framing (Table `tab:section42_main`, mean 18.6%);
  cadence subsection + Table `tab:cadence`; ablation → marginal-loss Table `tab:ablation`
  (TP +11% largest) + substitutability + SLO-insensitivity; SLO-Sensitivity rewritten
  (removed false "Static/GreenLLM violate 100%" claims).
- [x] Consistency pass: common $\rho$-gate defined in Setup; baselines 4→5 + DualScale-Ext
  defined (cite `biscale2025`); T1 "cheaper" typo fixed; τkv/SLO reconciled; Model-Validation
  + Oracle reworded to the common $\rho$-gate (classifier = validated component / opt-in);
  stale Production section disabled via `\iffalse…\fi` + placeholder; **stale 82% claim removed**.
- [x] Scheduler Overhead table refreshed from frozen data (6 strategies). **Finding:** SWEEP
  simulator decision time rose to 871 ms mean / 800 ms p50 / 5.5 s max. Only **3 of 1440
  SWEEP windows (0.21%, all T3 phase-transition) exceed the 5 s period**; no other strategy
  exceeds it. Reframed as unoptimized-Python limitation + async/background deployment, NOT a
  hard-real-time claim. All overhead/derived numbers scripted in `make_frozen_synthetic_tables.py`
  → `derived/` (8 CSVs incl. `overhead.csv`).
- [x] **Abstract / Introduction / conclusion propagated** (2026-05-27): feasible-frontier
  headline ("lowest-energy 0%-modeled-violation scheduler; saves 9.6--31.9%, 18.6% mean over
  best feasible non-SWEEP baseline"); removed stale 82%/68%/34%/18% energy claims and the
  disabled Azure production claims (TODO comments mark restoration after the redo); baseline
  names standardized. Also fixed two active-text risks: idle paragraph -> placeholder (stale kJ
  removed, table deferred); Model-Validation "0% false-safe independently" softened; "In
  deployment"->"A production implementation could"; τkv confirmed exactly identical (0.00%).
- [x] **Idle-sensitivity re-run** (frozen baselines; zero/low/baseline/high =
  `results/paper/idle_frozen_{zero,low,high}` + frozen-main baseline). Result: 5/6 schedulers
  100% idle-insensitive; only DynamoLLM-Mono sensitive (T4 403→481→467 kJ across
  zero→baseline→high, +15.9% zero→high; T3 +1.5%). Quantitative idle paragraph restored in
  evaluation_rewrite.tex; scripted as `derived/idle_sensitivity.csv` (9th derived CSV).
- [x] **Production redo (conv)** (2026-05-27): fixed `prepare_azure_trace.py` datetime-timestamp
  parsing bug; regenerated full ~52-min conv (17,690 reqs) + code (7,918 reqs) traces to
  `traces/azure_production_full/`. Anchor = max Static-Disagg 0%-feasible rate under rho-gate
  (conv 10 rps; code ~2 rps — long prompts saturate L40S). conv sweep (18 runs,
  `results/paper/prod_conv_frozen/`) at 50/70/85% util (5/7/8.5 rps, 7--12 DS blocks):
  SWEEP lowest-energy 0%-viol at every util, saving **35.5/29.6/27.9%** over Hierarchical-Disagg;
  **DualScale-Ext still under-provisions (6.5→26.6% viol) even on the longer trace** (two-tier
  limitation is not a short-trace artifact); DynamoLLM-Mono 0.1→9.3%. Wrote conv production
  section (Table `tab:prod_conv`), re-enabled section (old stale content kept in `\iffalse`),
  restored conv production sentences in abstract/intro/conclusion.
- [x] **Production redo (code)** (`results/paper/prod_code_frozen/`, peak=2 rps, 18 runs).
  Code is prefill-bound; the testbed is undersized so NO strategy is fully feasible (even
  Static 0.1--0.4%). SWEEP is Pareto-best (lowest viol ≤0.1% AND lowest energy among
  near-feasible), saving 6.8--7.7% vs Hierarchical and ~41% vs Static; margins smaller than
  conv (prefill-bound pins work to L40S). DualScale-Ext 3--10.6%, DynamoLLM-Mono 1.3--3.6% viol.
  Wrote code section (Table `tab:prod_code`); broadened abstract/intro/conclusion to
  "conversation and code".
- [x] **Code-prose honesty pass** (2026-05-27): tightened the code paragraph to state that
  at 85% load DualScale-Ext's 2141 kJ dips just below SWEEP's 2159 kJ but at 10.6% vs 0.1%
  modeled violations (not a feasible point); replaced "Pareto-dominant" with "Pareto-best
  among near-feasible strategies". conclusion.tex given the same "among near-feasible" qualifier.
- [x] **Production tables scripted** `paper/scripts/make_production_tables.py`: reads
  `prod_conv_frozen/` + `prod_code_frozen/` (+ `prod_diurnal_frozen/` when present), emits
  paper-ready LaTeX bodies + per-util savings. Feasibility rule is self-consistent (baseline
  viol ≤ SWEEP's own viol). Reproduces every paper cell: conv best-feasible=Hierarchical
  35.5/29.6/27.9%; code no-feasible-baseline, vs-Hierarchical 7.7/6.8/7.0%, auto-flags
  DualScale-85% as cheaper-but-infeasible (2141 kJ @ 10.6%v).
- [~] **1-week-derived diurnal robustness study** IN PROGRESS: deterministic 3h trough→peak
  segment (2024-05-12 04:00 UTC, 234k reqs, 21.7 rps, 36 DS blocks), window-level thinning to
  util×10 rps (NOT time-rescaling) -> `traces/azure_diurnal/conv_diurnal_{50,70,85}pct.csv`
  via `paper/scripts/make_diurnal_segment.py`. 70% smoke running (`prod_diurnal_smoke/`):
  static 14781 kJ/0%, greenllm 10045 kJ/0%, dynamollm 5129 kJ/0.09% all sane; dualscale +
  hierarchical + sweep still running.
- [ ] Remaining: finish diurnal study (smoke→3 utils); fold diurnal into the table script;
  deferred code cleanup.

### Deferred code cleanup — DONE (2026-05-28)
All three items applied; smoke test on T1@SLO=500/τkv=16 confirmed all 6 strategies are
**bitwise-identical** to the frozen reference (energy + violation match to all digits).
- [x] **Class rename** `DynamoLLMStrategy` → `DynamoLLMMono` in `baselines_dynamollm.py`,
  `factory.py`, `sweep_llm_scheduler.py` (shim), `schedulers/__init__.py`,
  `run_section42_sweep.py`, `run_rhocap_ablation.py`. Factory string key `"dynamollm"`
  is NOT renamed (it's the column value in every frozen `summary.csv`); display name
  "DynamoLLM-Mono" already lives in the `DISPLAY` maps of the table scripts.
- [x] **DualScale re-parented**: `class DualScaleStrategy(DynamoLLMStrategy)` →
  `class DualScaleStrategy(SweepLLMStrategy)`. DualScale never used either of
  `DynamoLLMMono`'s two overrides (`_admissible_routes`, `_try_monolithic_consolidation`),
  so the inheritance was misleading. New MRO: `DualScaleStrategy → SweepLLMStrategy → object`.
- [x] **Unused imports removed** from `baselines_dynamollm`, `baselines_dualscale`,
  `baselines_static`, `baselines_hierarchical`, `ablations`, `sweep`, and `factory`;
  also added the missing `from typing import Dict, Optional` to `factory.py` (signatures
  referenced them via `__future__ annotations` without an import).
