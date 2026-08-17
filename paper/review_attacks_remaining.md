# SWEEP-LLM: Remaining Reviewer Attacks and Remedies

This file covers attacks **not yet fixed** in the paper (A1 and A4 are resource-constrained;
B1–B6 and C1 are medium-priority methodological / writing issues).

---

## A. Paper-Level Issues

### A1 — No BiScale/DualScale baseline
**Attack**: The paper calls DualScale "the most closely related work" in the introduction but
does not evaluate against it. Reviewers will ask why the closest prior system is absent from
the comparison table.

**Remedy (needed, requires experiment)**:
Implement a DualScale-style baseline in the simulator: two-tier search that fixes phase-aware
placement at coarse timescale and sweeps DVFS at fine timescale, on a *homogeneous* pool.
Compare on the same six production-trace configurations. If DualScale code is not public, the
paper can describe a faithful re-implementation, as it does for GreenLLM and DynamoLLM.

**Alternative (if experiments not feasible)**: Add a paragraph in §4.5 (Ablation) arguing
that the ablation already covers the joint-vs-hierarchical axis (the Sequential policy from §3.2
matches DualScale's two-tier decomposition), and cite the motivation §3.2 result (26–38% gap)
as the lower bound on DualScale's structural disadvantage. This is weaker but defensible.

---

### A2 — Single model (Mistral-7B) on a single small cluster
**Attack**: All results are for one model on 4×L40S + 8×L4. KV-transfer overhead, TP
tradeoffs, and frequency sensitivity scale differently with model size. The paper cannot claim
generality without at least one additional model or cluster configuration.

**Remedy (needed, requires experiment)**:
- Run the simulation with **Llama-3-8B** (same family, similar size, different architecture)
  using the existing L40S/L4 calibration data (or a quick re-calibration).
- Optionally add one larger model (e.g., Mistral-Nemo 12B using the same-family pilot
  already mentioned in the commented-out §4 text) to show that the relative ranking of
  schedulers is preserved.
- Add a paragraph in §4.1 clarifying scope: "We focus on 7–8B parameter models on this
  specific two-pool cluster; larger models with higher KV-cache sizes may shift the relative
  weight of routing vs. DVFS savings, which we leave to future work."

---

## B. Methodological Concerns

### B1 — 87% MAPE for L4 prefill TTFT
**Attack**: The scheduler makes all L4 prefill routing decisions with a model that has 87%
mean absolute prediction error. The paper attributes this to P99 outliers but does not show
a histogram. Reviewers may question whether 0% SLO violations relies on this model being
accidentally conservative.

**Remedy**:
- Add a MAPE histogram or CDF for L4 prefill TTFT predictions showing that the error is
  dominated by outliers at extreme input lengths (il > 4096), while the bulk of predictions
  are within 20–30%.
- Clarify that the SLO classifier (Stage 2) uses a separate cost-sensitive classifier with
  calibrated guard-band thresholds, not the latency regressor. The 87% MAPE affects ranking,
  not feasibility admission. The 0% false-safe rate is from the classifier, not the regressor.
- Consider recomputing MAPE after removing the top 1% of prediction errors to show the
  "typical" error, alongside the full-dataset number.

---

### B2 — Cross-model savings 5–8% vs. headline 82% — context missing
**Attack**: §4.6 reports "SWEEP-LLM savings over the naïve all-max-frequency baseline are
5.9%, 5.2%, 7.8%." But Static-disagg is all-max-frequency, and the headline result is 82%
vs. Static-disagg. A reviewer who misreads the cross-model section will think SWEEP-LLM
only saves 5–8%.

**Remedy**:
Add one clarifying sentence at the start of §4.6 (cross-model robustness):

> "The cross-model experiment tests model-stack robustness in isolation using a *single-pool
> monolithic* setup to remove confounding from cross-pool routing; the 5–9% savings reported
> here represent the within-pool DVFS/TP gain under a held-out model, not the full
> disaggregated system gain (82% vs. Static-disagg in §4.2)."

---

### B3 — Ablation does not include the sequential policy
**Attack**: The motivation (§3.2) shows joint search beats sequential by 26–38%. But §4.5
ablates one knob at a time and never directly compares SWEEP-LLM to a sequential policy.
The link between the motivation result and the ablation is broken.

**Remedy**:
- Add one data point to Figure `ablation_energy.pdf`: a "Sequential" bar that applies
  controls in the order routing → DVFS → TP → capacity (same as §3.2 definition), and
  compare it to the full SWEEP-LLM.
- If re-running the simulation is feasible, this directly validates the §3.2 motivation claim
  in the full trace evaluation, not only in the controlled single-window setting of §3.2.
- If not feasible, add a cross-reference: "The motivation experiment (§3.2) establishes a
  26–38% gap between joint and sequential search on controlled single-window states; the
  ablation (§4.5) covers the individual-knob contributions but not the joint-vs-sequential
  axis directly."

---

### B4 — DynamoLLM's slow group-provisioning timescale not modeled
**Attack**: The paper says "We do not model DynamoLLM's slow group-provisioning timescale
separately." Real DynamoLLM separates coarse-scale instance scaling from fine-scale
TP/frequency control. Giving DynamoLLM per-window reconfiguration freedom equal to
SWEEP-LLM may overestimate DynamoLLM's practical agility (making it *harder* for SWEEP-LLM
to win, so this actually helps SWEEP-LLM's case — but reviewers may see it as unfair in the
other direction).

**Remedy**:
Tighten the existing caveat in §4.1:
> "We simulate DynamoLLM with per-window TP, frequency, and active-capacity freedom, which
> is more agile than the real system's coarse group-provisioning timescale. This over-grants
> DynamoLLM reconfiguration flexibility relative to SWEEP-LLM, making the comparison
> conservative from SWEEP-LLM's perspective."

---

## C. Writing and Clarity Issues

### C1 — Broken forward reference: production trace overhead
**Attack**: §4.7 (Scheduler Overhead) line states: "Overhead on production traces is higher
and workload-dependent; see §4.3 for production overhead." But §4.3 reports no overhead
numbers.

**Remedy** (quick fix):
Change the sentence to: "Overhead on production traces is higher and workload-dependent due
to longer request sequences and larger class mixes; a full per-trace breakdown is deferred to
a live-system evaluation."

Alternatively, add a one-line overhead note to §4.3 (e.g., mean decision time on production
traces) if the simulator already records it.

---

### C2 — Ablation sub-unity bars on T4 unexplained
**Attack**: §4.5 reports "a few sub-1.00× bars on T4 in the R+D+C column (0.87, 0.92, 0.95
at tight, moderate, loose) reflect a transient TP>1 bundle…" This is described as an
"artefact" but no resolution is given.

**Remedy**:
Either (a) fix the bundle-stability counter logic so bursty arrivals do not create lingering
TP>1 bundles that reduce energy below the full-system baseline, and re-run; or (b) add a
sentence saying this reflects a known conservative behavior of the bundle-stability gating
and does not affect the ordering of contributions across the four knobs.

---

## Summary Table

| ID  | Issue                                      | Effort   | Priority | Status   |
|-----|--------------------------------------------|----------|----------|----------|
| A1  | No DualScale baseline                      | High     | High     | **Open** |
| A2  | Single model / small cluster               | Medium   | High     | **Open** |
| B1  | 87% MAPE — clarify ranking-only scope      | Low      | Medium   | Fixed ✓  |
| B2  | Cross-model 5–8% vs. 82% — add context    | Trivial  | Medium   | Fixed ✓  |
| B3  | Ablation missing sequential comparison     | Medium   | Medium   | Fixed ✓  |
| B4  | DynamoLLM timescale caveat — tighten       | Trivial  | Low      | Fixed ✓  |
| C1  | Broken overhead cross-reference            | Trivial  | Low      | Fixed ✓  |
| C2  | T4 sub-unity ablation bars unexplained     | Low      | Low      | Fixed ✓  |

Only **A1** (DualScale baseline) and **A2** (multi-model evaluation) require new experiments.
