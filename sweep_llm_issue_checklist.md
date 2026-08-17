# SWEEP-LLM Paper Review Issue Checklist

Use this checklist to track the remaining paper-repair tasks. Check items off as they are completed and verified in the compiled PDF.

## Status Legend

- `[x]` Completed and verified in the compiled PDF.
- `[ ]` Not completed or still needs verification.
- `Priority`: P0 is highest priority; P5 is final polish.

---

## P0 — Design vs. implementation consistency

**Goal:** A reviewer should not be able to say that the algorithm described in the paper differs from the implementation used for evaluation.

**Current status:** Major issue closed. The route-mask and state-classifier inconsistencies have been fixed in the design text.

- [x] State-classifier priority matches implementation: `BOTH_LOW -> PREFILL_HEAVY -> DECODE_HEAVY -> BOTH_HEAVY`.
- [x] Hysteresis wording matches implementation: margin-based state acceptance, with persistence used separately for ideal-search triggering.
- [x] The paper states that all four routes are admissible in every state: `A->A`, `A->B`, `B->A`, `B->B`.
- [x] State-specific search is described as candidate-set, anchor, and ordering control rather than route masking.
- [x] PREFILL_HEAVY explains residual `B->A` traffic as anchor-driven behavior rather than a contradiction.
- [x] DECODE_HEAVY explains `B->A` as an admitted route family that closes the decode-heavy optimality gap.
- [x] Final grep: confirm no active old route-mask language remains, such as:
  - `admissible per-class route sets`
  - `state-based masking`
  - `R_c^{(s)}`
  - critical `{A->A, A->B}` route masks
  - critical `{A->B, B->B}` route masks
  - **DONE:** grep over the 9 live `\input` files (non-comment lines only) returns zero hits for `admissible per-class route sets`, `state-based masking`, `\mathcal{R}_c^{(s)}`, and the directional `{A->A,A->B}`/`{A->B,B->B}` masks. The only `\mathcal{R}_c` symbol in `design_rewrite.tex` is the state-independent definition `\mathcal{R}_c=\{A->A,A->B,B->A,B->B\}` (the common route set). Old route-mask drafts live only in the non-compiled `design.tex`/`evaluation.tex`.

---

## P1 — Feasibility semantics

**Goal:** The paper must clearly distinguish request-level SLO feasibility, classifier feasibility, and the common utilization-based `rho`-gate used in the frozen cross-strategy evaluation.

**Current status:** Mostly solved, but not fully closed. The main conceptual fix is in place, and the gate-sensitivity pilot is a strong addition. A few active wording artifacts still need cleanup.

- [x] Main evaluation uses the common utilization-based `rho <= 1` feasibility gate for all strategies.
- [x] Evaluation setup defines modeled violation as rejection by the common `rho`-gate.
- [x] Evaluation setup states that the common `rho`-gate is not a request-level P99 latency guarantee.
- [x] Models section no longer treats the feasibility classifier as the main cross-strategy evaluation gate.
- [x] Models section describes the classifier as a stricter SLO-oriented admission mode and validation component.
- [x] Gate-sensitivity pilot added for `rho`, `classifier+rho`, and `latency+rho`.
- [x] Gate-sensitivity text states the pilot is representative, not a full re-evaluation.
- [x] Gate-sensitivity result supports that SWEEP is not exploiting a loose `rho`-gate.
- [x] Abstract: verify no remaining phrase says `modeled SLO violations`; use `modeled violations under the common rho-gate` or equivalent.
  - **DONE:** `abstract.tex` lines 7--9 already read `modeled violations under that gate` / `at $0\%$ modeled violations`. Grep for `modeled SLO violation` over live files = 0 hits.
- [x] Introduction contribution bullet: verify no remaining phrase implies request-level SLO replay.
  - **DONE:** `Introduction.tex:197` reads `modeled GPU-side energy under a common utilization-based feasibility gate` and `lowest-energy $0\%$-modeled-violation scheduler`; no request-level replay implied. (The only other `SLO` use, line 85, is the generic acronym definition.)
- [x] Metrics paragraph: replace bare `SLO violation rate` with `modeled violation rate under the common rho-gate`.
  - **DONE:** `evaluation_rewrite.tex:200` already says `modeled violation rate`; fixed the bare-`SLO` instance at `:620` ("energy and **modeled violation rates (common $\rho$-gate)**").
  - **FOLLOW-UP FIX (rendering bug):** `:193` had a **bare `\rho`** (no `$...$`) in running prose. LaTeX silently entered math mode at `\rho` and rendered the rest of the sentence as math (spaces collapsed, italic) -- "throughput(requests/s), and the selected control decisions..." Fixed to `$\rho$-gate`; PDF now renders "common $\rho$-gate (%), throughput (requests/s)," correctly. Scanned all live files: every other `\rho` is inside `$...$` or an `equation` environment (none bare).
- [x] Table captions: replace `modeled SLO-violation rate` with `modeled violation rate under the common rho-gate`.
  - **DONE:** `evaluation_rewrite.tex:626` caption now reads `modeled violation rate (\%, common $\rho$-gate)`. The frozen-results caption (line 298) already used `modeled violation rate`.
- [x] Introduction design paragraph: replace `lowest SLO violation` with `lowest modeled violation under the configured feasibility gate`.
  - **DONE:** no active `lowest SLO violation` phrase exists; the Pareto/cadence phrasings already say `lowest violations` in modeled-violation contexts (`evaluation_rewrite.tex:509`, `conclusion.tex:11`). Nothing to change.
- [x] Fix active `Section ??` / `\S??` references connected to the evaluation configuration.
  - **DONE:** grep for `??` over live files = 0; compiled `build/main.pdf` text has 0 occurrences of `??` (all `\ref`/`\S\ref` resolve).
- [x] Optional strengthening: run the same gate-sensitivity pilot on T3/T4, especially T4, to cover phase transitions and bursts.
  - **DONE:** ran T3 and T4 (`gate_sensitivity_pilot.py`, 120s, SLO=500/200, $\tau_{kv}$=16, all 6 strategies, gates rho/classifier+rho/latency+rho). Both hold under all three gates; results in `results/paper/analyses/gate_sensitivity/gate_sensitivity_results.csv` (+ manifest.json). §gate_sensitivity paragraph updated from "representative T1/T2 check" to "reduced pilot on all four synthetic traces (T1--T4)".

---

## P2 — Motivation Figure 2 circularity / model-generated motivation

**Goal:** Avoid the reviewer attack that Figure 2 uses SWEEP's own predictive model to justify SWEEP.

**Current status:** Partially mitigated. Targeted hardware validation exists, but it validates building blocks rather than the full joint-vs-sequential result.

- [x] Label Figure 2 clearly as a model-predicted controlled study.
  - **DONE:** `motivation_rewrite.tex` Fig.~caption now ends: "This is a model-predicted controlled study built from hardware-calibrated per-configuration models, not a direct end-to-end hardware replay; Table~\ref{tab:obs2_hw_validation} spot-checks the underlying per-configuration mechanisms on the real L40S/L4 testbed."
- [x] Add text near Figure 2 explaining that it is built from hardware-calibrated per-configuration models.
  - **DONE:** new `\paragraph{Grounding the model-predicted study.}` after the joint-search conclusion (uses the suggested safe wording; cites \S\ref{sec:phase_a_models} for the grouped-CV grounding).
- [x] Add a small hardware spot-validation table using the existing `compare_obs2_validation.py` results.
  - **DONE:** new Table~\ref{tab:obs2_hw_validation} in `motivation_rewrite.tex` (resolves to Table 1; verified 0 `??` in PDF). Numbers regenerated from `compare_obs2_validation.py` against the current models.
- [x] State clearly that the hardware spot checks validate per-configuration mechanisms, not the full end-to-end joint optimizer.
  - **DONE:** paragraph + table caption both state the checks validate per-configuration primitives and that "the joint-versus-sequential comparison itself is a composition of calibrated per-configuration models rather than a direct hardware replay."
- [x] Include the existing L4 low-frequency decode validation point:
  - L4, TP=4, il=32/ol=1024, 1200 MHz.
  - Measured TPOT about 24 ms and power about 279 W.
  - Model predicts about 28 ms, conservatively.
  - **DONE:** first two rows of Table 1 (1200 MHz: meas TPOT 24.2 ms / model 28.2 ms / 279 W; 1410 MHz: 24.2 / 27.9 / 278 W); called out in prose as the most counter-intuitive primitive.
- [x] Include the L4 short-prefill frequency-threshold validation point, if it is part of the final script/table.
  - **DONE:** last two rows of Table 1 (TP1, il128/ol32, 1410 & 1755 MHz: meas TTFT 192 ms / model 219 ms; flat across frequency, as predicted).
- [ ] Add 2-3 L40S validation points if feasible, so the spot validation is not L4-only.
  - **DEFERRED (needs hardware):** `Obs2_Validation_L4/` contains L4-only runs. Adding L40S points requires new cluster runs (e.g., L40S prefill TP4@990 and decode@2010). `validate_obs2_l4.sh` can be extended for this. Not doable locally.
- [ ] Optional strongest fix: run one end-to-end hardware comparison of Full-joint vs Sequential for one representative state, if it can be done without misleading Ethernet/KV-transfer artifacts.
  - **DEFERRED (needs hardware):** requires a real L40S+L4 serving run; on the testbed's commodity Ethernet, cross-pool KV transfer would be unrepresentative (see P-note in evaluation setup), so this must use co-located/fast-fabric pools. Cluster job.
- [x] Avoid claiming that Figure 2 is fully hardware-measured unless the full result is actually measured.
  - **DONE:** caption and paragraph both explicitly say "not a direct end-to-end hardware replay" / "a composition of calibrated per-configuration models."

Suggested safe wording:

> Figure 2 is a model-predicted controlled study, not a direct end-to-end hardware replay. To avoid using the model as an ungrounded motivation, we validate the key per-configuration mechanisms on the real L40S/L4 testbed. These spot checks validate the hardware primitives used by the model; the joint-vs-sequential comparison is a composition of calibrated per-configuration models rather than a direct hardware replay.

---

## P3 — Oracle wording / search-quality comparison

**Goal:** The oracle comparison should not overclaim exhaustive optimality if it does not enumerate SWEEP's full per-class decision space.

**Current status:** Open.

- [x] Check whether the oracle enumerates all per-class route maps or only four uniform route choices: `AA`, `AB`, `BA`, `BB`.
  - **DONE:** confirmed in `paper/scripts/oracle_comparison.py:94-98` --- comment "for simplicity, use one route for all classes (4 options)"; it builds `route_options` as the 4 uniform maps. It *is* exhaustive over bundle pairs and frequency pairs (line 105 `product(bundles_a, bundles_b, freqs_a, freqs_b)`) but **not** over per-class route maps.
- [x] If it only enumerates uniform-route choices, rename it from `exhaustive oracle` to something like `uniform-route exhaustive oracle`.
  - **DONE:** renamed throughout `sec:eval_oracle`: subsection title -> "Search Quality vs.\ a Uniform-Route Exhaustive Oracle"; body -> "a \emph{uniform-route} exhaustive oracle"; table caption updated.
- [ ] If feasible, run a true oracle for the selected windows (all bundle pairs, all frequency pairs, all per-class route maps up to `4^K = 256`).
  - **DEFERRED (optional):** not run. The restricted oracle is now honestly labeled; a true per-class oracle is ~256x more route maps per candidate and is a separate experiment. Left as future strengthening, not required given the honest reframing.
- [x] Avoid wording such as `global optimum` unless the full SWEEP decision space is actually enumerated.
  - **DONE:** no `global optimum` claim exists; added explicit "not a global optimum over SWEEP-LLM's full decision space" and "a search-quality reference over the bundle/frequency space."
- [x] Reframe the section as a search-quality sanity check if using a restricted oracle.
  - **DONE:** body now frames it as a search-quality reference and notes SWEEP-LLM "can match it despite searching far fewer candidates" because the oracle lacks per-class routing.
- [x] Ensure the table caption explicitly states the oracle's search space.
  - **DONE:** `tab:oracle_gap` caption now states "exhaustive over bundles and frequencies, but a single route per window---it does not search per-class route maps."

---

## P4 — Baseline wording and fairness

**Goal:** Baseline names and descriptions should be honest about what is faithful, what is inspired, and what is an upper-bound/oracle variant.

**Current status:** Mostly solved, but final grep needed.

- [x] Use `DynamoLLM-Mono` for the monolithic-routing baseline.
- [x] Use `GreenLLM-OracleDVFS` for the oracle frequency-sweep DVFS baseline.
- [x] Use `Hierarchical-Disagg` for the routing-aware sequential baseline.
- [x] Use `DualScale-Ext` for the heterogeneous two-tier provisioner.
- [x] Soften `faithful DualScale` language.
  - **DONE:** `evaluation_rewrite.tex:179` -> "a DualScale-inspired two-tier provisioner ... faithful to DualScale's coarse/fine control \emph{structure} but ... not a faithful reproduction; we extend it with heterogeneous routing and oracle frequency selection." `:391` -> "follows DualScale's two-tier control structure". `abstract.tex:9` -> "a DualScale-style $5$-minute two-tier provisioner". (The "not faithful" disclaimers at 177/396 are deliberate, kept.)
- [x] Confirm all tables and captions use the standardized baseline names.
  - **DONE:** grep shows consistent `DynamoLLM-Mono`, `GreenLLM-OracleDVFS`, `Hierarchical-Disagg`, `DualScale-Ext`, `Static-Disagg` across `evaluation_rewrite.tex` tables/captions. (Minor: \S motivation's Fig.\ uses its own illustrative label `Static-disagg`, self-consistent within that figure; noted under P5.)
- [x] Confirm related-work text distinguishes prior systems from your inspired/extended baselines.
  - **DONE:** added a clause to `related.tex`: the evaluation "does not reproduce them verbatim; instead it compares against strong inspired/extended variants ... DynamoLLM-Mono, DualScale-Ext, and GreenLLM-OracleDVFS (\S\ref{sec:eval_setup})."
- [x] Avoid implying DynamoLLM-Mono or GreenLLM-OracleDVFS are full faithful reproductions.
  - **DONE:** DynamoLLM-Mono (`:177`) explicitly "is \emph{not} a faithful reproduction" + lists omissions; GreenLLM-OracleDVFS (`:176`) "Rather than reproduce GreenLLM's TPS-based heuristic, we give it an \emph{oracle} ... a generous single-knob DVFS upper bound."

---

## P5 — Final polish and mechanical cleanup

**Goal:** Remove distracting paper-quality issues before submission.

**Current status:** Open.

- [x] Fix `Seqential` -> `Sequential` in Figure 2 and surrounding text.
  - **DONE:** grep for `Seqential` over all of `paper/` (tex, figure scripts, CSVs) = 0 hits; the figure and text already use `Sequential`.
- [x] Fix `comprises of` -> `comprises` or `consists of`.
  - **DONE:** `motivation_rewrite.tex:171` now reads "the decode-heavy state comprises $30\%$ SL, ...".
- [x] Fix all active `Section ??`, `\S??`, `Fig. ??`, `Table ??`, and undefined references.
  - **DONE:** compiled `build/main.pdf` text has 0 occurrences of `??`; no Undefined-control / undefined-reference messages in `build/main.log`. (Note: latexmk needs one extra pdflatex pass to resolve refs because the EPS auto-conversion makes pdflatex return a non-fatal exit 1; see build note below.)
- [x] Confirm Figure 2 caption states whether the figure is measured or model-predicted.
  - **DONE (P2):** caption says "a model-predicted controlled study ... not a direct end-to-end hardware replay."
- [x] Confirm all production-trace claims say `trace replay`, not production deployment.
  - **DONE:** no `production deployment` phrasing in live files; text uses "Azure production-trace replays" / "trace replay".
- [x] Confirm all energy claims say `modeled GPU-side energy` where appropriate.
  - **DONE:** headline energy claims qualified as `modeled GPU-side energy` (abstract, intro, conclusion, eval; 4 live mentions).
- [x] Confirm cross-pool KV-transfer energy is not claimed as measured.
  - **DONE:** `models.tex:239` -- "the reported energy ... does not separately include interconnect or KV-transfer energy."
- [x] Confirm cross-pool KV-transfer latency is described as analytically swept.
  - **DONE:** `evaluation_rewrite.tex:185,189` -- "model the cross-pool KV-cache transfer cost analytically rather than measure it ... swept across $\{5,16,41\}\,\mu$s/tok."
- [x] Final grep for stale claims:
  - **DONE:** over live files -- `modeled SLO violation`=0, `SLO violation rate`=0 (now "modeled violation rate (common $\rho$-gate)"), `faithful DualScale`=0, `primary safety gate`=0, `Seqential`=0, `comprises of`=0, `??`=0 in PDF. `82%` appears only in the non-compiled `evaluation.tex`; `68%` appears only in legitimate DualScale-Ext modeled-violation contexts ($39$--$68\%$).

**Build note (not a content defect):** `latexmk` returns exit 12 because the on-the-fly EPS->PDF conversion of `figures/overview.eps` makes `pdflatex` return a non-fatal exit 1 (it still writes a correct 20-page PDF). This requires one extra `pdflatex` pass to resolve newly-added cross-references. Optional one-time fix (awaiting go-ahead, touches the `\includegraphics` line): `epstopdf paper/figures/overview.eps` and point the include at extensionless `overview` so `pdflatex` uses the PDF directly -> clean exit 0 and automatic ref resolution.

---

## P6 — Design-section rewrite & evaluation honesty (recent pass)

**Goal:** Keep the rewritten Design section internally consistent, at the right altitude, and aligned with the implementation; close the output-length and same-window honesty gaps in Evaluation.

- [x] Design reread after rewrite: active text consistently uses "estimated next-window workload", distinguishes constrained vs. ideal search, keeps all four routes admissible in every state, and explains the stability ramp / fast-adoption / emergency promotion without contradicting the implementation.
- [x] Overview condensed to a roadmap: components 1--4 are high-level pointers; mechanics (per-class stats + demand equation, two-search formalism, bundle definition + promotion) live in their subsections, not duplicated in the Overview.
- [x] Component headings match their subsection titles (Request Classification, Workload Classification, State-Specific Joint Search, Reconfiguration-Aware Execution); state names uniformly `\textsc{}`; "execution"/"state-specific" consistent across design/intro/conclusion/eval.
- [x] One TP degree per pool justified (homogeneous model-partition layout, no re-sharding on phase reallocation); per-phase TP noted as future work in `conclusion.tex`.
- [x] Reference deployment for `C_pf^ref`/`C_dc^ref` made explicit (TP=1, all GPUs active, max freq), matching `schedulers/sweep.py`.
- [x] `eq:token_demands` + per-class statistics relocated to `sec:window_classifier` (defined before use); the per-class tilde quantities linked to the per-request `il`/predicted `ol-hat` from `sec:request_classes`.
- [x] Bundle-promotion wording corrected: "`H_k` consecutive ideal-search updates" (matches `_update_bundle_targets`), not "consecutive windows".
- [x] No active broken refs: `alg:pipeline`, `eq:pool_bundle`, `eq:token_demands` are referenced only in commented/`\iffalse` blocks; `\iffalse` copies of `eq:ideal_search`/`eq:constrained_search` are skipped (no duplicate labels).
- [x] Evaluation oracle caveat extended to cover BOTH the same-window workload summary AND the admission-time output-length estimate (instantiated from each request's recorded output length); deployment alternative (predictor / requested generation budget) stated (`evaluation_rewrite.tex` Workload-summaries paragraph).
- [x] Verified consistent: power MAPE is 6--8% in all active text (the 4.4--5.5% figure is commented/dead); latency Spearman 0.94--0.97 consistent across `models.tex` and `evaluation_rewrite.tex`.
- [x] Verified present: search-overhead limitation (`sec:eval_overhead`): mean 870.8 / p99 4152 / max 5511 ms, 3/1440 windows over the 5 s period, explicit "not a hard real-time controller" + async production path.
- [ ] Add the emergency-override "disabled by default in the evaluated configuration" one-liner to Evaluation setup / `tab:eval_params` (method described in `sec:reconfig`; evaluated-configuration choice belongs in eval). Still missing from the eval section.

---

## Recommended next-step order (updated)

1. Add the emergency-override "disabled by default in the evaluated configuration" note to Evaluation setup (the last open honesty item; P6).
2. Final compile / PDF pass: confirm 0 `??` and no undefined references after the Design/Eval edits (the EPS exit-12 quirk needs one extra `pdflatex` pass).
3. Optional, only if time allows (all currently deferred): P2 L40S validation points, P2 end-to-end hardware Full-joint vs. Sequential, P3 true per-class oracle.

