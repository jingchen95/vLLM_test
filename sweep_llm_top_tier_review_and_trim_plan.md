# SWEEP-LLM Top-Tier Systems Review and 15-Page Trim Plan

This checklist is written as a critical SC/HPCA/ICS/IPDPS-style review. It focuses on: problem statement, method correctness, logical flow, experimental coverage, remaining reviewer risks, and concrete trimming actions to bring the paper from ~20 pages toward ~15 pages.


---

# Second-round PDF review after 15-page revision

**Reviewed artifact:** `main.pdf`, 15 pages, compiled after the major trimming pass.

**Updated verdict:** The paper is now much closer to a top-tier systems submission.
The core story is visible, the main evaluation is compact, and the feasibility/Pareto framing is much more defensible than in the first reviewed version.
If the remaining consistency and cleanup issues below are fixed, the paper moves from “borderline / weak reject risk” toward **borderline accept / weak accept potential**, depending on reviewer tolerance for measurement-calibrated simulation.

## Round-2 high-level assessment

**What is now strong:**

- The paper is now 15 pages including references and appendix, so the earlier length problem is largely solved.
- The main claim is clear: SWEEP-LLM stays on the feasible/Pareto frontier under a common utilization-based modeled feasibility gate.
- The evaluation flow is now strong: setup, synthetic traces, production traces, ablation, model validation, scheduler overhead.
- The production section is especially credible because it shows both a strong conversation-trace result and a smaller, more honest prefill-bound code-trace result.
- The conclusion is now appropriately short and focused on the contribution rather than future work.
- Related work is concise and keeps the right distinctions.

**What still needs attention before submission:**

- Several stale or inconsistent details remain in the appendix and references.
- Figure 2 currently does not visibly label the motivation experiment as model-predicted, which reopens the circularity concern from the first review.
- A few typos and grammar issues remain in prominent places.
- The bibliography contains author/name corruption and descriptive notes that look like internal annotations.
- The model-validation classifier table may alarm reviewers because some rows show `FS% = 100.0%` without enough context.

## Must-fix before any submission

### R2-1. Figure 2 must be labeled as model-predicted

**Issue:** The current Figure 2 caption in the PDF says only that it shows energy comparisons and selected configurations.
It does not state that the joint-vs-sequential study is model-predicted / measurement-calibrated rather than an end-to-end hardware replay.
This matters because one of the original top reviewer risks was circularity in the motivation experiment.

**Fix:** Add a short caveat directly to the Figure 2 caption or immediately before the figure:

> This is a controlled model-predicted study using hardware-calibrated per-configuration models; it illustrates knob coupling rather than validating end-to-end deployment performance.

- [x] Add the model-predicted caveat to Figure 2 caption or nearby text. **(activated; hardware-validation table+paragraph still commented)**
- [ ] Verify the caveat appears in the compiled PDF.

### R2-2. Fix the stale Appendix D KV-transfer statement

**Issue:** Appendix D currently says that at `tau_kv = 41 us/tok`, the transfer time is “at most ≈5 ms” for the prompt lengths in the synthetic traces.
This is incorrect for a 1024-token prompt: `41 us/token * 1024 tokens ≈ 42 ms`.
The main text was already fixed to say “tens of milliseconds,” but the appendix still contains the stale 5 ms claim.

**Why it matters:** This is a numerical correctness bug.
A reviewer checking the appendix can catch it immediately.

**Fix:** Change the appendix explanation to match the main text:

> even the largest tested transfer penalty adds only tens of milliseconds for the longest synthetic prompts and does not change the utilization-gated candidate choice.

- [x] Fix Appendix D transfer-time statement.
- [x] Search for all remaining “5 ms” or “few milliseconds” KV-transfer claims. **(only appendix:93 was active; fixed)**

### R2-3. Remove stale internal wording in appendix

**Issue:** Appendix D still says “the frozen cross-strategy comparison.”
“Frozen” is an internal term and should not appear in the paper.

**Fix:** Replace with:

> the common cross-strategy comparison

or:

> the main cross-strategy comparison

- [x] Remove “frozen” from appendix prose. **(active appendix:93 fixed; dead-block uses remain harmless)**

### R2-4. Fix visible typos and grammar in Motivation

**Issue:** Section 3.2 has multiple visible instances of “Seqential” in the PDF.
Even if this comes from a macro or spelling in source, it reads as a typo.

**Issue:** This sentence is ungrammatical:

> These numbers represent an ordered control-space expansion, not as independent marginal contributions.

**Fix:** Use:

> These numbers represent an ordered control-space expansion, not independent marginal contributions.

- [x] Fix “Seqential” → “Sequential” everywhere. **(0 hits in source)**
- [x] Fix the “not as independent” grammar issue.
- [ ] Search the compiled PDF for other visible typos.

### R2-5. Fix typo in Models section

**Issue:** Page 7 contains “modelcomponents” without a space in the sentence about validation.

**Fix:** Change to “model components.”

- [x] Fix `modelcomponents` → `model components`. **(N/A in live source; verify in PDF)**

### R2-6. Clean the bibliography

**Issue:** The references contain visible problems that look unprofessional:

- Reference [3] has a corrupted author string: “Raphaël andMusic”.
- Reference [15] has a suspicious author name: “Ínigo Inardova”.
- Several entries include descriptive notes in the rendered bibliography, such as “Two-tier phase-aware placement and DVFS for disaggregated serving,” “Phase-level DVFS, up to 34% energy reduction,” and similar annotation text.

**Why it matters:** These look like internal notes rather than publication metadata.
They will reduce reviewer confidence even if the technical content is strong.

- [~] Fix corrupted author names in `references.bib`. **(andMusic, Inardova→Goiri, chen2024sweep done; DynamoLLM author list pending)**
- [x] Remove internal explanatory `note` fields from references unless the venue explicitly allows them. **(all 7 removed)**
- [ ] Recompile and scan the rendered bibliography manually.

### R2-7. Check Azure trace citation consistency

**Issue:** The PDF production-trace paragraph currently cites only Splitwise `[12]`, while the section later discusses a one-week Azure trace that appears to correspond to the DynamoLLM/Azure 2024 trace.
If the source has already been changed to cite both Splitwise and DynamoLLM, the PDF may simply be stale.

**Fix options:**

- If the main conv/code traces are Splitwise/Azure 2023 and the diurnal trace is DynamoLLM/Azure 2024, cite both in the production-trace setup sentence.
- Or cite Splitwise in the conv/code sentence and cite DynamoLLM explicitly in the one-week diurnal sentence.

- [~] Verify source and compiled PDF agree on the Azure trace citations. **(source now cites patel2024splitwise; PDF verify pending)**
- [ ] Cite DynamoLLM wherever the one-week Azure trace is used. **(decision pending — is the diurnal trace DynamoLLM/Azure-2024?)**

## Important but not necessarily blocking

### R2-8. Model-validation classifier table is potentially confusing

**Issue:** Table 9 reports some rows with `FS% = 100.0%`, even when the absolute false-safe count is only 1.
This can look alarming because readers may interpret it as “the classifier is unsafe,” even though the surrounding prose says the optional saturation gate reduces decision-level false-safe rate from 34.5% to 6.5%.

**Possible fixes:**

- Rename `FS%` to make the denominator explicit.
- Add a short caption phrase explaining the denominator.
- Move the detailed classifier table to the appendix and keep only the saturation-gate summary in the main text.

**Recommendation:** If page pressure or reviewer-risk reduction matters, move Table 9 to the appendix.
The prose already carries the important safety result.

- [ ] Clarify Table 9 denominator or move the table to appendix.
- [ ] Make sure the main text explains that these optional classifiers are not the main cross-strategy gate.

### R2-9. Figure/table density is acceptable but still high

**Assessment:** The paper now has many tables, but most are justified.
The main text tables that should stay are:

- baseline/control-space table;
- scheduler-parameter table, unless page pressure is severe;
- synthetic result table;
- conversation and code replay tables;
- ablation table;
- model accuracy table;
- overhead table.

**Most movable table:** Table 9, the feasibility classifier table.
It is detailed, takes space, and is the most likely to create confusion.

- [ ] Consider moving Table 9 to appendix if the paper feels too table-heavy.

### R2-10. Appendix order and references

**Observation:** The compiled PDF places references before the appendices.
This is not necessarily wrong, but some venue formats expect appendices before references or have specific appendix placement rules.

- [ ] Check the target venue template requirements for appendix placement.
- [ ] If the venue requires references last, move appendices before references or use the venue’s recommended appendix mechanism.

### R2-11. Remove internal script path from Appendix B

**Issue:** Appendix B includes an internal script path:

> `(paper/scripts/make_diurnal_segment.py)`

This is useful for your own reproducibility but looks internal in the submitted PDF unless the artifact includes that script and the path is meaningful.

**Fix:** Either remove it or convert it into an artifact reference if the code will be released.

- [ ] Remove or formalize the internal script-path reference.

## Section-by-section second-round comments

### Abstract and Introduction

**Assessment:** Strong and much improved.
The abstract correctly states measurement-calibrated simulation, common utilization-based feasibility gate, and feasible/Pareto frontier results.
The introduction now frames the four coupled decisions clearly.

**Remaining concern:** The introduction still says the goal is to satisfy TTFT and TPOT SLOs, while the main evaluation uses a utilization gate.
This is acceptable because §6.1 explains the distinction, but avoid adding any stronger “SLO guarantee” language later.

- [ ] Keep “modeled violation” and “common utilization gate” language in abstract/introduction.
- [ ] Do not use “SLO violation” for the main evaluation metric.

### Background

**Assessment:** Good and concise.
It gives only the necessary concepts: inference phases, phase disaggregation, SLOs, and TP.
No major issue.

### Motivation

**Assessment:** The motivation is conceptually strong.
Figure 1 supports phase-dependent placement well.
Figure 2 supports joint control, but needs the model-predicted caveat restored.

**Required fixes:** R2-1 and R2-4.

### Design

**Assessment:** The design section is now coherent and readable.
The distinction between state classification, state-specific search, and reconfiguration-aware actuation is clear.
The four routes remain admissible in all states, which avoids the earlier B→A contradiction.

**Minor presentation issue:** Table 1 has some line wrapping in the Decode_Heavy row.
This is not blocking, but if you need polish, shorten “B decode capacity; A decode fallback” to “B decode + A fallback.”

- [ ] Optional: shorten Table 1 row text to improve layout.

### Models

**Assessment:** The model section is much better scoped than before.
It clearly says the models are deployment-specific and that cross-pool transfer is analytically swept.

**Required fix:** R2-5.

**Minor issue:** The section says the stricter admission mode sharply reduces false-safe admissions in §6.5, which is correct.
Make sure the reader is not left thinking this stricter mode is used in the main comparison.
The current text already says the main comparison uses the common rho-gate.

### Evaluation

**Assessment:** This is now the strongest version of the evaluation so far.
The flow is good and the claims are disciplined.
The synthetic, production, ablation, validation, and overhead subsections each serve a distinct purpose.

**Good choices:**

- Synthetic results compare on the feasible frontier, not raw energy.
- Production results separate conversation and code regimes.
- Code trace is framed as cluster-limited / near-feasible Pareto, which is honest.
- Ablation correctly presents marginal losses, not standalone knob contributions.
- Model validation does not overclaim request-level SLO guarantees.
- Overhead is honest about the Python prototype not being hard real-time.

**Required fixes:** R2-2, R2-3, R2-7.

**Recommended cleanup:** R2-8.

### Related Work

**Assessment:** Concise and appropriate.
It now avoids long citation lists and makes the core distinction clear.
No major issue.

### Conclusion

**Assessment:** The conclusion is now the right length.
It restates the contribution and key evidence without introducing new future work.
No major issue.

### References and Appendix

**Assessment:** This is now the weakest part of the compiled PDF.
The appendix contains some stale text, and the bibliography contains visible metadata problems.
These are easy to fix but important.

**Required fixes:** R2-2, R2-3, R2-6, R2-11.

## Updated final checklist

### Must fix

- [x] Add model-predicted caveat to Figure 2 caption or nearby text. **(caption caveat activated; the hardware spot-validation table + grounding paragraph are still commented out — restore decision pending.)**
- [x] Fix Appendix D `≈5 ms` KV-transfer statement. **(→ "tens of milliseconds for the longest synthetic prompts")**
- [x] Remove “frozen” from appendix prose. **(active appendix:93 fixed; remaining "frozen" only inside dead `\iffalse` blocks)**
- [x] Fix “Seqential” typo. **(0 hits in source; PDF was stale)**
- [x] Fix “not as independent marginal contributions.”
- [x] Fix `modelcomponents` typo. **(N/A — live `models_rewrite` has "model components"; old `models.tex` is dead)**
- [~] Clean bibliography author names and internal notes. **(all 7 `note=` fields removed; `andMusic`→`and Music`; `Inardova`→`Goiri`; `chen2024sweep` authors corrected. DynamoLLM author list still needs verification.)**
- [~] Verify Azure trace citation coverage. **(conv/code cite `patel2024splitwise`; confirm whether the one-week diurnal trace needs `stojkovic2024dynamollm`.)**

### Should fix

- [ ] Clarify or move Table 9 feasibility classifier details.
- [x] Remove or formalize the internal script path in Appendix B.
- [ ] Check appendix/reference ordering for the target venue.
- [x] Optionally shorten Table 1 row text. **(→ "B decode + A fallback")**

### Already strong

- [x] 15-page target achieved.
- [x] Abstract and contribution framing are much clearer.
- [x] Evaluation structure is coherent.
- [x] Main results use feasible/Pareto framing.
- [x] Production conversation/code story is credible.
- [x] Ablation is logically consistent with the motivation.
- [x] Scheduler overhead is honest and appropriately limited.
- [x] Related work and conclusion are concise.

---

---

## Executive verdict

**Current likely review outcome:** borderline / weak reject to borderline accept depending on reviewer tolerance for measurement-calibrated simulation.

**Best current strengths:**

- The problem is timely and important: energy-efficient LLM inference on heterogeneous disaggregated GPU clusters.
- The paper now has a much more defensible claim: **feasible/Pareto frontier under a common modeled feasibility gate**, not raw energy alone.
- The strongest baseline comparison is now meaningful: **SWEEP-LLM vs. Hierarchical-Disagg**, where both get the same per-class routing space but one searches jointly and the other greedily/sequentially.
- Production-trace replays are now nuanced and credible: conversation shows large feasible-frontier gains; code shows smaller gains in a prefill-bound cluster-limited regime.
- The paper now honestly labels Figure 2 as model-predicted and adds hardware spot-validation.

**Top remaining risks:**

1. The manuscript is too long and too detailed for a top-tier systems paper. The central story is buried.
2. Model validation is split across Sections 5 and 6.7 with different reported numbers, which can look inconsistent.
3. The main evaluation uses a rho-gate rather than true request-level P99 SLO replay. The paper now says this, but reviewers may still object unless the limitation is handled cleanly and concisely.
4. Figure 2 circularity is mitigated, but not fully eliminated: hardware validation covers primitives, not end-to-end joint-vs-sequential execution.
5. Some text still over-explains implementation details that should be moved to appendix or compressed.

**Recommended target structure for 15 pages:**

- Abstract + Intro: 1.5 pages
- Background + Motivation: 2 pages
- Design: 3 pages
- Models + validation summary: 2 pages
- Evaluation: 5 pages
- Related + conclusion: 1.5 pages

---

# A. Problem statement and paper positioning

## A1. Problem clarity

**Assessment:** The problem statement is mostly clear: heterogeneous phase-disaggregated serving exposes four coupled decisions: routing, TP, DVFS, and active GPU allocation. The intro explains why these interact and why fixed or decomposed control is insufficient.

**Remaining issue:** The framing still occasionally reads like the goal is true TTFT/TPOT-SLO satisfaction, while the main results are under a modeled utilization-feasibility gate. You already clarify this later, but the intro and background still set a stronger expectation than the evaluation can fully satisfy.

- [x] In the introduction, keep “under TTFT/TPOT SLOs” as the conceptual target, but add one early phrase that the evaluation reports **modeled GPU-side feasibility under a common utilization gate**.
- [x] Avoid implying an end-to-end deployed SLO guarantee. Use “measurement-calibrated trace replay” / “modeled feasibility” consistently in the intro.
- [x] In the final contribution bullet, change “over the strongest baseline” to “over the strongest feasible or near-feasible baseline” because code is Pareto/near-feasible rather than strict 0% feasible.

**Suggested rewrite for last contribution bullet:**

> Across synthetic and Azure trace replays, SWEEP-LLM stays on the feasible/Pareto frontier under a common utilization-based feasibility gate: it is the lowest-energy 0%-modeled-violation scheduler on all synthetic traces, saving 18.6% on average, and saves 27.9--35.5% on Azure conversation traffic and 6.8--7.7% on prefill-bound Azure code traffic over the strongest feasible or near-feasible baseline.

## A2. Novelty statement

**Assessment:** The novelty is real but should be stated more sharply. The contribution is not just “we jointly optimize four knobs.” It is:

- per-class routing across heterogeneous pools;
- joint route/TP/DVFS/capacity search;
- state-guided candidate generation so the search is tractable;
- reconfiguration-aware actuation separating fast and slow knobs.

- [x] Add one sentence in the intro that distinguishes SWEEP-LLM from simply adding more knobs to an existing scheduler:
  > The key challenge is not the availability of these knobs individually, but that their feasible energy-minimizing setting is non-separable across workload states.
- [x] Make “state-guided joint search” the core algorithmic contribution, not just “model-driven scheduling.”

---

# B. Method and design correctness

## B1. Design/implementation consistency

**Assessment:** This is now mostly solved. The paper says all routes are admissible in all states; state changes candidate sets, anchors, and ordering. This matches the design direction and avoids the previous B→A contradiction.

- [x] State classifier order is now BOTH_LOW → PREFILL_HEAVY → DECODE_HEAVY → BOTH_HEAVY.
- [x] Hysteresis wording is now margin-based and separate from ideal-search persistence.
- [x] Route admissibility is common across states.
- [x] PREFILL_HEAVY and DECODE_HEAVY explain B→A as anchor-driven/admitted behavior.

**Remaining suggestion:** Algorithm 2 still says pruning “violates Eq. 4,” while the evaluated implementation uses the configured feasibility gate. This is acceptable if Eq. 4 is conceptual, but a reviewer may still see a mismatch.

- [x] In Algorithm 2 lines referring to Eq. 4, replace “violates Eq. 4” with “violates the configured feasibility gate” or “violates the conceptual feasibility target / configured admission rule.” **(DONE)**
- [x] In Section 4.4, add one short phrase: **(DONE — added the rho-gate/stricter-gate instantiation sentence after "They differ only in the candidate set explored.")**
  > In implementation, the admission rule can be instantiated as the common rho-gate or a stricter classifier/latency gate; the evaluation uses the former for fairness.

## B2. Search policy complexity

**Assessment:** The state-specific search section is correct but long. It reads like a code walkthrough. A top-tier paper should communicate the principle, not every derivation detail.

- [x] Compress the four state-specific subsections by 30--40%. **(DONE — PREFILL_HEAVY/DECODE_HEAVY verbose candidate/allocation prose compressed to one sentence each (knob focus + class order now in the table); BOTH_LOW/BOTH_HEAVY/Burst already compact. P0 route-admissibility + B→A sentences preserved verbatim.)**
- [x] Move detailed anchor-family descriptions to appendix or a small table. **(DONE — "Anchor bias" column in Table~\ref{tab:state_policy} captures the per-state anchors; detailed anchor families remain in code.)**
- [x] Replace long prose for PREFILL_HEAVY/DECODE_HEAVY with a table: **(DONE — added Table~\ref{tab:state_policy}: State / Candidate focus / Class order / Anchor bias, with caption stating all four routes are admissible in every state.)**

| State | Candidate focus | Class order | Anchor bias | Rationale |
|---|---|---|---|---|
| PREFILL_HEAVY | TP_A, n_A,pf, f_A | decreasing input | A-side prefill + B-side spillover | relieve prefill bottleneck |
| DECODE_HEAVY | TP_B, n_B,dc, f_B | decreasing output | B-side decode + B→A escape | relieve decode bottleneck |
| BOTH_LOW | compact bundles | light first / within-pool tie | consolidation | power-gate |
| BOTH_HEAVY | broadest retained set | total tokens | all routes | avoid bottleneck |

This would likely save ~0.5 page.

## B3. Reconfiguration-aware execution

**Assessment:** The fast/slow knob split makes sense and is important. It is a real systems contribution. However, it remains simulated rather than deployed. The paper should not oversell production readiness.

- [ ] Keep the stability-counter explanation, but shorten it.
- [ ] Consider moving the emergency override paragraph to appendix or a footnote because it is disabled in evaluation.
- [ ] In conclusion/limitations, state that actual vLLM integration and asynchronous decision application remain future work.

---

# C. Model and feasibility semantics

## C1. Common rho-gate story

**Assessment:** This has improved a lot. The evaluation now clearly defines modeled violations as rho-gate violations, not request-level P99 replay. The gate-sensitivity pilot is a strong defense.

**Remaining top-tier concern:** Some reviewers will still object that the paper claims SLO-aware scheduling while evaluating utilization feasibility. The best defense is to keep this discussion short, precise, and visible.

- [ ] Keep the “Modeled feasibility gate” paragraph in Evaluation Setup.
- [ ] Keep gate-sensitivity paragraph, but consider moving detailed explanation of classifier+rho degeneracy to appendix if trimming.
- [ ] Use “modeled violation under the common rho-gate” consistently; do not use “SLO violation” for the main metric.
- [ ] In abstract/conclusion, keep “modeled violations under that gate” rather than “SLO violations.”

## C2. Inconsistent model validation numbers

**Major issue.** Section 5 Table 2 reports power MAPE around 6--8% and latency MAPE 16--25%, while Section 6.7 reports power MAPE 4.4--5.5% and latency MAPE 28--87%. These may correspond to different model versions, datasets, or validation protocols, but the paper does not make that distinction clear. A reviewer can interpret this as inconsistency.

- [x] Decide whether Section 5 Table 2 and Section 6.7 are reporting the same validation protocol. If yes, make the numbers identical. **(DONE — same grouped-CV protocol; de-duplicated so §6.7 points to Table 2 instead of restating numbers.)**
- [ ] If they are different, explicitly label them:
  - Table 2: grouped cross-validation on Phase-2 profiling corpus.
  - Section 6.7: scheduler-facing validation on held-out sweep / current model stack.
- [ ] Better: keep **one model validation table in the main paper** and move the other to appendix.
- [x] Remove duplicated validation prose. Right now Section 5 and Section 6.7 both explain the same model accuracy story. **(DONE — removed the duplicated prediction-error prose from §6.7; kept only its unique decision-quality/cross-model content.)**

**Recommended fix for trimming:** Merge Section 6.7 into Section 5 or move it to appendix. In main text, keep a compact “Model validation summary” paragraph with one table.

## C3. Model section length

**Assessment:** The model section is too detailed for the main paper. It includes profiling corpus details, model families, training procedures, and validation methodology. This is useful but can be shortened.

- [ ] Compress Section 5 to: model interface, rho/capacity, power/latency/feasibility outputs, cross-pool composition, one validation table.
- [ ] Move the following to appendix:
  - dense/TP/DVFS/held-out sweep breakdown;
  - exact feature lists;
  - model family choices and monotonicity details;
  - detailed validation methodology bullets;
  - Mistral-Nemo pilot transfer paragraph.

Estimated page savings: **1.0--1.5 pages**.

---

# D. Motivation and Figure 2

## D1. Figure 2 circularity

**Assessment:** Minimum defensible fix is now in place. The caption says Figure 2 is model-predicted, Table 1 spot-checks mechanisms, and the prose admits it is not a direct end-to-end hardware replay.

- [x] Figure 2 labeled as model-predicted.
- [x] Table 1 added with hardware spot-validation.
- [x] Text says primitives are validated, not full joint-vs-sequential.

**Remaining concern:** Table 1 is L4-only, while Figure 2 uses both L40S and L4 and makes a joint-vs-sequential claim. This is now honest but still not ideal.

- [ ] Optional strengthening: add 2--3 L40S spot-validation rows if hardware time is available.
- [ ] Optional strongest fix: one end-to-end hardware comparison of Full-joint vs Sequential for one state, only if it can be done without misleading Ethernet artifacts.
- [ ] If no new hardware is run, keep current limitation wording. Do not claim P2 is fully eliminated.

Estimated page savings: **0.3--0.5 page**, but do not over-trim if Figure 2 is central.

---

# E. Evaluation coverage and claims

## E1. Synthetic evaluation

**Assessment:** Strong and mostly coherent. It tests the four intended trace types and gives a feasible-frontier summary. The route-share mechanism paragraph is useful.

- [x] T1/T2/T3/T4 cover prefill-heavy, decode-heavy, phase shift, balanced burst.
- [x] Table 5 reports all baselines and feasible frontier.
- [x] Gate-sensitivity pilot now covers T1--T4.

**Trimming suggestions:**

- [x] Move Figure 4 trace visualization to appendix or shrink it if page budget is tight. It is useful but not essential once trace definitions are clear. **(DONE — Figure → Appendix~\ref{sec:synth_traces_appendix}; main keeps "the realized rate, class mix, and per-window state match the design intent (Appendix…)".)**
- [x] Compress “Raw energy alone is misleading” into 2--3 sentences. **(DONE — old paragraph commented; replaced with 2 sentences keeping the DualScale/DynamoLLM-Mono violation numbers + the feasible-frontier framing.)**
- [~] Move detailed route-share mechanism paragraph to appendix if necessary, but keep one sentence explaining why DynamoLLM-Mono fails on T1/T4. **(KEPT in main for now — the "Mechanism: per-class joint routing" paragraph is a core explanation; can move later if page budget is tight.)**

Estimated page savings: **0.4--0.7 page**.

## E2. DualScale cadence sensitivity

**Assessment:** Important because DualScale is a likely reviewer focus. However, the table and prose are long.

- [x] Keep Table 6 only if DualScale is central to reviewer defense. **(Decided: production traces now carry the DualScale defense; moved the full sweep to appendix.)**
- [x] Otherwise move Table 6 to appendix and summarize in one sentence: **(DONE — Table~6 + detailed prose → Appendix~\ref{sec:cadence_detail}; main keeps a labeled subsection (`sec:cadence`, so the 4 cross-refs still resolve) with a 1–2 sentence summary.)**
  > Across cadences from 20s to 5min, DualScale-Ext never reaches a 0%-modeled-violation point below SWEEP-LLM; lower-energy points are infeasible.
- [ ] Keep the caveat that 600s synthetic traces are stress tests and production traces are fairer for two-tier provisioning.

Estimated page savings if moved: **0.5 page**.

## E3. Production traces

**Assessment:** This is now one of the strongest parts of the paper. Keep it. The conv/code/diurnal story is nuanced and convincing.

**However:** The production section is too long and includes three production-style studies: conv, code, diurnal. For 15 pages, you probably cannot keep all details.

- [ ] Keep Table 7 conv trace in main paper.
- [ ] Keep Table 8 code trace in main paper if you want to show workload generality and honesty.
- [x] Move Table 9 diurnal robustness to appendix, or compress it to one paragraph without the full table. **(DONE — full subsubsection + Table moved to Appendix~B (`sec:prod_diurnal`); main text keeps a one-paragraph summary + `% MOVED TO APPENDIX` marker.)**
- [ ] If keeping diurnal in main text, shorten the three findings to 1 paragraph.

Estimated page savings from moving/compressing diurnal: **0.7--1.0 page**.

## E4. Code workload wording

**Minor issue:** Table 8 has SWEEP at 0.0% violation for 70% util, but the text says “no strategy is perfectly feasible here.” That is true only across all utilization settings, not every single cell.

- [x] Change “no strategy is perfectly feasible here” to: **(DONE)**
  > No strategy is perfectly feasible across all utilization levels; even Static-Disagg has 0.1--0.4% modeled violations.

## E5. SLO/KV sensitivity

**Assessment:** This section is scientifically honest but not very exciting because the result is “insensitive due to rho-gate.” It is useful as a limitation, but not worth much main-paper space.

- [~] Compress Section 6.5 to one paragraph in Evaluation Setup or after Table 5. **(PARTIAL — idle paragraph + Figure 5 moved out; SLO-insensitivity result + feasibility breakdown kept in §6.5. Can compress further if desired.)**
- [x] Move idle power sensitivity to appendix unless it is needed to defend a specific claim. **(DONE — Appendix~C `sec:eval_idle`; §6.5 keeps a one-sentence summary + marker.)**
- [x] If Figure 5 shows flat lines, consider moving it to appendix or removing it entirely and stating the result in text. **(DONE — Figure moved to Appendix~D `sec:eval_taukv`; the 0.00% flat result stays in §6.5 text.)**

Estimated page savings: **0.6--0.9 page**.

## E6. Ablation

**Assessment:** Good and important. Keep Table 10. The interpretation is mature: marginal knob contributions are small because knobs are substitutable.

- [ ] Keep Table 10 in main paper.
- [ ] Shorten the DVFS explanation to 1 paragraph.
- [ ] Move “maximum frequency is realistic because vLLM/SGLang/TGI...” to a footnote or remove.
- [ ] Keep the Hierarchical-Disagg comparison, but combine it with Table 5 or Table 10 discussion.

Estimated page savings: **0.2--0.4 page**.

## E7. Search-quality oracle

**Assessment:** Wording is now honest: uniform-route exhaustive oracle, not global optimum. This is a good sanity check but not essential to the main story.

- [x] Oracle no longer overclaims global optimality.
- [x] Table caption states it is uniform-route only.

**Trimming:**

- [x] Move Section 6.8 and Table 11 to appendix if page pressure is high. **(DONE — moved to Appendix~A (`sec:eval_oracle`, `tab:oracle_gap`).)**
- [x] Keep one sentence in main text: **(DONE — "A uniform-route exhaustive sanity check (Appendix A) shows SWEEP-LLM is within 0--12\% of the oracle on binding states and never misses a feasible configuration the oracle finds.")**
  > A uniform-route exhaustive sanity check shows SWEEP-LLM is within 0--12% on binding states and never misses feasibility; details in appendix.

Estimated page savings: **0.5 page**.

## E8. Scheduler overhead

**Assessment:** Important limitation. Keep a compact version in main paper.

- [ ] Keep Table 12 or reduce it to SWEEP vs Hierarchical vs DualScale only.
- [ ] Compress prose to: median 0.80s, p99 4.15s, max 5.5s, 3/1440 windows exceed 5s; not hard real-time; async implementation needed.
- [ ] Move full strategy overhead table to appendix if needed.

Estimated page savings: **0.2--0.4 page**.

---

# F. Baselines and related work

## F1. Baseline fairness

**Assessment:** Much improved. Baselines are now honestly named and described. Hierarchical-Disagg is especially important.

- [x] DynamoLLM-Mono clearly described as inspired, not faithful.
- [x] GreenLLM-OracleDVFS clearly described as oracle frequency sweep.
- [x] DualScale-Ext says faithful to coarse/fine structure but not a full reproduction.
- [x] Hierarchical-Disagg isolates joint search vs routing access.

**Remaining concern:** The baseline definitions in Evaluation Setup are long.

- [x] Replace long baseline bullets with a compact table: **(DONE — added Table~\ref{tab:baselines} (Routing/TP/Freq/Active-GPUs/Reconfig per strategy); compressed the five bullets to one line each, keeping only the honesty disclaimers (oracle upper bound, not-faithful-reproduction, etc.). Old bullets commented for trace.)**

| Baseline | Routing | TP | Frequency | Active GPUs | Reconfiguration |
|---|---|---|---|---|---|
| Static-Disagg | fixed AB | fixed | max | all | none |
| GreenLLM-OracleDVFS | fixed AB | fixed | oracle | all | freq only |
| DynamoLLM-Mono | AA/BB only | joint | joint | joint | SWEEP backend |
| Hierarchical-Disagg | all per-class | sequential | sequential | sequential | no stability |
| DualScale-Ext | all routes | coarse | fine | coarse | 5-min |
| SWEEP | all per-class | joint | joint | joint | fast/slow |

Estimated page savings: **0.5--0.7 page**.

## F2. Related work length

**Assessment:** Related work is reasonable but can be shorter.

- [x] Reduce Related Work to 0.5--0.75 page. **(DONE — `related.tex` compressed from 4 paragraphs to 3 (folded power/thermal + offline together) with tighter prose; old version commented for trace. All citations retained and resolve.)**
- [x] Keep only distinctions that matter: DVFS-only, disaggregated/homogeneous, heterogeneity/offline, power/thermal scheduling. **(DONE — the three paragraphs are exactly DVFS-only; heterogeneous/disaggregated (homogeneous-targeted closest systems + our inspired/extended variants); power/thermal/offline.)**
- [ ] Remove detailed citation subtitles in the reference entries if not required by venue style. **(N/A here — this is a `references.bib`/bbl-style matter, not the Related Work prose; left for the venue's reference-format pass.)**

Estimated page savings: **0.3--0.5 page**.

---

# G. Logical flow

## G1. Current flow

Current flow is generally good:

1. Background/motivation shows heterogeneity and joint control matter.
2. Design explains state-aware joint search.
3. Models explain measurement-calibrated simulator.
4. Evaluation shows feasible frontier on synthetic and production traces.

**Main flow issue:** Models and evaluation methodology are too interleaved and repetitive. The reader sees model validation in Section 5, calibration methodology in Section 6.1, and model validation again in Section 6.7.

- [ ] Decide one location for model training/validation details.
- [ ] Keep Evaluation Setup focused on baselines, traces, metrics, rho-gate.
- [ ] Move calibration methodology out of Evaluation Setup unless needed for reproducibility.

## G2. Recommended main-paper flow after trimming

- [x] Section 1: Intro, concise.
- [x] Section 2: Background, half-page.
- [ ] Section 3: Motivation, keep Figure 1 + Figure 2 but concise.
- [ ] Section 4: SWEEP design, table-driven state search.
- [ ] Section 5: Models and validation summary, one table.
- [ ] Section 6: Evaluation:
  - Setup + baselines table;
  - Synthetic feasible frontier;
  - Production traces;
  - Ablation;
  - Overhead/limitations compact.
- [ ] Appendix: gate sensitivity, cadence sweep details, diurnal robustness, oracle, idle sensitivity, full model validation.

---

# H. Top-tier reviewer attack list

Use this as a final defense checklist.

## H1. “This is only simulation.”

Current defense: measurement-calibrated per-pool models, hardware validation, production-trace replay, analytic KV sweep.

- [ ] Keep explicit limitation that cross-pool KV transfer is analytic, not measured.
- [ ] Keep Figure 2 hardware spot-validation table or move to appendix but cite it.
- [ ] Avoid claiming deployed-system energy savings.

## H2. “Your SLO metric is not request-level SLO.”

Current defense: rho-gate clearly defined; gate-sensitivity pilot; request-level latency validation remains limitation.

- [ ] Keep “not a request-level latency guarantee” sentence.
- [ ] Keep gate-sensitivity result concise.
- [ ] Avoid “SLO violation” wording for main metric.

## H3. “You crippled baselines.”

Current defense: strong inspired variants; oracle DVFS; Hierarchical-Disagg; DualScale cadence and production traces.

- [ ] Keep Hierarchical-Disagg as the main non-SWEEP comparison.
- [ ] Avoid overclaiming faithful reproduction of DynamoLLM/DualScale.
- [ ] Use feasible/Pareto frontier rather than raw energy.

## H4. “Figure 2 is circular.”

Current defense: model-predicted label + hardware spot-validation.

- [ ] Keep limitation that joint-vs-sequential is model-composed.
- [ ] Add L40S spot validation only if time permits.

## H5. “Scheduler overhead is too high.”

Current defense: unoptimized simulator, median/p99 within window, 3/1440 over, async implementation.

- [ ] Keep this as limitation, not a claim of production readiness.
- [ ] Avoid saying “online-ready” unless implementation is optimized.

## H6. “Results are too specific to Mistral-7B and L40S/L4.”

Current defense: target deployment-specific scheduler-enabling model; reprofiling is offline cost.

- [ ] State deployment-specific scope clearly.
- [ ] Move Mistral-Nemo pilot to appendix or limitations.

---

# I. 15-page trimming plan

Current paper is about 20 pages including references. To target 15 pages, prioritize cuts that reduce detail without weakening the core claim.

## Must-cut / move to appendix

- [ ] Move detailed model training/profiling methodology to appendix. Save ~1.0--1.5 pages.
- [ ] Move duplicate model validation Section 6.7 or merge into Section 5. Save ~0.7--1.0 page.
- [ ] Move diurnal robustness Table 9 to appendix or compress to one paragraph. Save ~0.7--1.0 page.
- [ ] Move oracle Section 6.8/Table 11 to appendix. Save ~0.5 page.
- [ ] Move idle sensitivity and SLO/KV flat-line details to appendix; remove or shrink Figure 5. Save ~0.6--0.9 page.

## Should-cut / compress

- [ ] Convert baseline descriptions to a compact table. Save ~0.5 page.
- [ ] Compress state-specific design subsections using a state-policy table. Save ~0.5 page.
- [ ] Shorten production-code explanation and cadence discussion. Save ~0.3--0.5 page.
- [ ] Reduce Related Work. Save ~0.3--0.5 page.
- [ ] Shorten conclusion. Save ~0.1--0.2 page.

## Optional cuts

- [ ] Move Figure 4 trace visualization to appendix. Save ~0.4 page.
- [ ] Move Table 1 hardware spot-validation to appendix if Figure 2 caption cites it; keep only if P2 circularity risk feels high. Save ~0.25 page.

## Recommended main-paper tables/figures to keep

- [ ] Figure 1: phase-specific winner regions.
- [ ] Figure 2: joint optimization motivation, labeled model-predicted.
- [ ] Figure 3: system overview, if compact.
- [ ] Table 5: main synthetic results.
- [ ] Table 7: Azure conv results.
- [ ] Table 8: Azure code results.
- [ ] Table 10: ablation.
- [ ] One compact model validation table.
- [ ] Optional compact overhead table or one overhead sentence.

## Recommended appendix material

- [ ] Full model training details.
- [ ] Full model validation diagnostics.
- [ ] Gate-sensitivity pilot table.
- [ ] DualScale cadence sweep.
- [ ] Diurnal robustness.
- [ ] Uniform-route oracle.
- [ ] Idle power sensitivity.
- [ ] Full overhead table.

---

# J. Concrete next steps

## Immediate correctness fixes

- [x] Resolve model validation number inconsistency between Section 5 Table 2 and Section 6.7.
  - **DONE:** §6.7 "Prediction error" restated stale pre-overhaul numbers (power 4.4–5.5%, latency 28–87%); commented it out and replaced with a pointer to Table 2 (§5) + current numbers (power 6–8%, Spearman 0.94–0.97); commented two now-redundant follow-ons. Table 2 (§5) is now the single source of truth.
- [x] Change code workload sentence: “no strategy is perfectly feasible here” → “no strategy is perfectly feasible across all utilization levels.”
  - **DONE:** `evaluation_rewrite.tex` (old commented, new below). PDF: “…perfectly feasible across all utilization levels, not even the fully-provisioned Static-Disagg…”.
- [x] Replace Algorithm 2 “violates Eq. 4” phrasing with “violates the configured feasibility gate/admission rule.”
  - **DONE:** Alg. 2 now “rejected by the configured admission rule (… instantiated as the common $\rho$-gate)” / “admitted by the configured rule”; added the §4.4 sentence on instantiating the admission rule.
- [x] Confirm the final PDF no longer has Metrics rendering issues.
  - **DONE:** fixed bare `\rho`→`$\rho$` at `evaluation_rewrite.tex:193`; renders “common $\rho$-gate (%), throughput (requests/s),” correctly. 0 `??`, no LaTeX errors.

## Immediate trimming actions

- [ ] Merge or move Section 6.7 model validation.
- [ ] Convert baseline descriptions to a table.
- [ ] Move diurnal robustness to appendix.
- [ ] Move oracle section to appendix.
- [ ] Compress SLO/KV/idle sensitivity.

## Optional experiment/code review needs

I do not need more code to assess the paper argument right now. More code would be useful only for:

- verifying actual runtime behavior of `NeedIdealSearch`, bundle stability, and emergency override;
- verifying the exact baseline implementation if you suspect hidden asymmetries;
- auditing the true per-class route search versus the paper algorithm.

At this stage, the paper’s biggest risks are writing/positioning/validation consistency and length, not missing code details.

---

# K. Final reviewer-style score estimate

**If submitted as current 20-page version:** borderline / weak reject due to length, simulation-heavy evaluation, and model-validation redundancy/inconsistency.

**If trimmed to 15 pages with consistency fixes:** borderline accept / weak accept potential. The core idea is strong enough; the paper needs sharper packaging and fewer distractions.


# SWEEP-LLM Current Paper Issue Tracker

This file tracks only the issues that remain after the 15-page revision of `main.pdf`.
Old first-round trim-plan items have been removed because the paper structure, section lengths, related work, conclusion, and main evaluation have already been substantially revised.
Use `[x]` only after the issue is fixed in source and verified in the compiled PDF.

---

## Current verdict

The paper is now much closer to a top-tier systems submission.
The main story is visible, the evaluation is compact, and the feasibility/Pareto framing is much more defensible than in the earlier draft.
The remaining risks are mostly cleanup and consistency issues, not fundamental structure problems.

**Current expected review posture:** borderline accept / weak accept potential if the remaining issues below are fixed.
The largest remaining reviewer risk is still that the evaluation is measurement-calibrated simulation rather than a full deployed system, but the paper now states this more honestly.

---

## Must fix before submission

### M1. Add a model-predicted caveat to Figure 2

**Status:** Caption caveat DONE; the hardware spot-validation table (`tab:obs2_hw_validation`) + grounding paragraph are still commented out (restore decision pending).

**Issue:** Figure 2 supports the motivation for joint optimization, but the compiled PDF does not visibly state that the study is model-predicted / measurement-calibrated rather than an end-to-end hardware replay.
This can reopen the circularity concern.

**Action:** Add a short caveat to the Figure 2 caption or immediately before the figure.

Suggested wording:

> This is a controlled model-predicted study using hardware-calibrated per-configuration models; it illustrates knob coupling rather than validating end-to-end deployment performance.

- [ ] Add the caveat in source.
- [ ] Verify the caveat appears in the compiled PDF.

### M2. Fix stale Appendix D KV-transfer text

**Status:** DONE — "≈5 ms" → "tens of milliseconds for the longest synthetic prompts" (appendix.tex:93).

**Issue:** Appendix D still says the largest tested transfer penalty adds at most about `5 ms`.
This is incorrect for a 1024-token prompt: `41 us/token * 1024 tokens ≈ 42 ms`.
The main text already uses the correct “tens of milliseconds” wording.

**Action:** Replace the stale appendix sentence with wording consistent with the main text.

Suggested wording:

> even the largest tested transfer penalty adds only tens of milliseconds for the longest synthetic prompts and does not change the utilization-gated candidate choice.

- [ ] Fix Appendix D.
- [ ] Search for all remaining `5 ms`, `few milliseconds`, or stale KV-transfer claims.

### M3. Remove internal/stale wording from Appendix D

**Status:** DONE — appendix:93 "frozen" → "main"; remaining "frozen" occurrences are only inside dead `\iffalse` blocks (not rendered).

**Issue:** Appendix D still says “the frozen cross-strategy comparison.”
“Frozen” is an internal term and should not appear in the paper.

**Action:** Replace with one of:

> the common cross-strategy comparison

or

> the main cross-strategy comparison

- [ ] Remove all active uses of “frozen” from the submitted source/PDF.

### M4. Fix visible Motivation typos and grammar

**Status:** DONE — grammar "not as independent" → "not independent"; `Seqential` has 0 hits in source (PDF was stale).

**Issue 1:** Section 3.2 visibly contains `Seqential` in the compiled PDF.
This should be `Sequential`.

**Issue 2:** This sentence is ungrammatical:

> These numbers represent an ordered control-space expansion, not as independent marginal contributions.

**Action:** Fix to:

> These numbers represent an ordered control-space expansion, not independent marginal contributions.

- [ ] Fix `Seqential` → `Sequential` everywhere.
- [ ] Fix the “not as independent” grammar issue.
- [ ] Search the compiled PDF for other visible typos.

### M5. Fix typo in Models section

**Status:** DONE / N/A — no `modelcomponents` string in live source; `models_rewrite` has "model components" (proper space). Old `models.tex` (where the PDF instance came from) is no longer compiled. Verify in next PDF.

**Issue:** Page 7 contains `modelcomponents` without a space.

**Action:** Change to `model components`.

- [x] Fix `modelcomponents` → `model components`. **(N/A in live source; verify in PDF)**

### M6. Clean bibliography metadata

**Status:** Partial — all 7 internal `note=` fields removed; `andMusic`→`and Music`; `Inardova, {\'I}nigo`→`Goiri, {\'I}\~nigo`; `chen2024sweep` authors corrected to Chen/Manivannan/Goel/Peric{\`a}s. STILL PENDING: verify the DynamoLLM (`stojkovic2024dynamollm`) author list (currently 5 authors ending "Zhang, Fan" — looks incomplete).

**Issue:** The rendered bibliography contains visible metadata problems:

- Reference [3] has corrupted author text: `Raphaël andMusic`.
- Reference [15] has a suspicious author name: `Ínigo Inardova`.
- Several references render internal explanatory notes, such as “Two-tier phase-aware placement and DVFS for disaggregated serving,” “Phase-level DVFS, up to 34% energy reduction,” and similar notes.

**Action:** Clean `references.bib`.

- [ ] Fix corrupted author names.
- [ ] Remove internal explanatory `note` fields unless required by venue style.
- [ ] Recompile and manually scan the bibliography.

### M7. Verify Azure trace citations after recompile

**Status:** Partial — undefined `\cite{azure_llm_trace}` replaced with `\cite{patel2024splitwise}` (conv/code traces). PENDING: confirm whether the one-week diurnal trace is DynamoLLM/Azure-2024 and, if so, cite `stojkovic2024dynamollm` there; verify in compiled PDF.

**Issue:** The reviewed PDF cited only Splitwise in the production trace setup sentence, while the text also discusses a one-week Azure trace that appears to correspond to DynamoLLM/Azure 2024.
The source was later updated to cite both Splitwise and DynamoLLM, but this must be verified in the compiled PDF.

**Action:** Ensure the production-trace paragraph cites both relevant sources, or cite DynamoLLM explicitly where the one-week trace is introduced.

- [ ] Verify compiled PDF uses the intended Azure trace citations.
- [ ] If the one-week trace is from DynamoLLM/Azure 2024, cite DynamoLLM at that point.

---

## Should fix / high-value cleanup

### S1. Clarify or move the feasibility-classifier table

**Status:** Open.

**Issue:** Table 9 reports some rows with `FS% = 100.0%`, even when the false-safe count is only 1.
This can look alarming because a reader may interpret it as the classifier being broadly unsafe.
The surrounding prose says the optional throughput-saturation gate reduces decision-level false-safe rate from `34.5%` to `6.5%`, which is the more important safety result.

**Options:**

1. Rename `FS%` so the denominator is explicit.
2. Add a caption note explaining the denominator.
3. Move the detailed classifier table to the appendix and keep the short saturation-gate summary in main text.

**Recommendation:** Move Table 9 to the appendix if page/table density or reviewer confusion is a concern.

- [ ] Clarify the denominator or move the table to appendix.
- [ ] Verify the main text still states that optional classifiers are not used for the main cross-strategy comparison.

### S2. Remove internal script path from Appendix B

**Status:** DONE — removed `(paper/scripts/make_diurnal_segment.py)` from Appendix B.

**Issue:** Appendix B includes an internal script path:

> `(paper/scripts/make_diurnal_segment.py)`

This looks internal unless the artifact release includes the script at that path.

**Action:** Remove the path or convert it into a proper artifact reference.

- [ ] Remove or formalize the script-path reference.

### S3. Check appendix/reference ordering for the target venue

**Status:** Open.

**Issue:** The compiled PDF places references before the appendices.
This may be acceptable, but some venue templates have strict appendix/reference ordering.

- [ ] Check the target venue requirements.
- [ ] Adjust appendix placement only if required.

### S4. Optional layout polish for Table 1

**Status:** DONE — Decode_Heavy row shortened to "B decode + A fallback".

**Issue:** Table 1 has some awkward line wrapping in the `Decode_Heavy` row.

**Possible fix:** Shorten row text, for example:

> `B decode + A fallback`

instead of:

> `B decode capacity; A decode fallback`

- [ ] Optional: shorten Table 1 row text if layout still looks awkward after final compile.

---

## Already fixed / no longer active

These issues were important in earlier drafts but are no longer active blockers.
They are kept here only so we do not re-open old work unnecessarily.

- [x] Paper length reduced to 15 pages including references and appendix.
- [x] Abstract rewritten and shortened.
- [x] Introduction now frames the four coupled controls clearly.
- [x] Background condensed.
- [x] Motivation now focuses on phase-dependent placement and joint optimization.
- [x] Design section reorganized around request classification, workload classification, joint search, and reconfiguration-aware execution.
- [x] Design now states all four routes remain admissible in every state.
- [x] Models section scoped as deployment-specific, measurement-calibrated empirical modeling.
- [x] Evaluation structure reduced to six main subsections: setup, synthetic, production, ablation, model validation, overhead.
- [x] Cadence sweep moved out of the main evaluation flow.
- [x] SLO sensitivity moved out of the main evaluation flow and reframed as a modeling limitation.
- [x] Production traces compressed and split into conversation and code regimes.
- [x] Code trace is framed as cluster-limited / near-feasible Pareto rather than strict 0%-feasible frontier.
- [x] Ablation now reports marginal losses and no longer uses stale Motivation numbers.
- [x] Model Validation compressed and aligned with the common rho-gate evaluation semantics.
- [x] Scheduler Overhead honestly states the Python simulator is not a hard real-time controller.
- [x] Related Work condensed and cleaned.
- [x] Conclusion reduced to one focused paragraph.
- [x] Main evaluation table captions shortened.
- [x] Most bold mini-headings in result subsections removed to reduce template-like style.

---

## Current section-by-section status

### Abstract and Introduction

**Status:** Good.

The abstract and introduction correctly state the measurement-calibrated simulator, common utilization-based feasibility gate, and feasible/Pareto frontier results.
Avoid adding any stronger deployed-SLO guarantee language.

### Background

**Status:** Good.

Concise and sufficient.
No active issue.

### Motivation

**Status:** Strong but needs cleanup.

Active remaining issues:

- M1: Figure 2 needs model-predicted caveat.
- M4: `Seqential` typo and grammar issue.

### Design

**Status:** Good.

The design is coherent and readable.
Only optional layout polish remains for Table 1.

### Models

**Status:** Good but needs typo fix.

Active remaining issue:

- M5: `modelcomponents` typo.

### Evaluation

**Status:** Strong.

The evaluation flow and claims are now disciplined.
Active remaining issues are mostly appendix/evaluation cleanup:

- M2: Appendix D stale KV-transfer value.
- M3: Appendix D “frozen” wording.
- M7: Azure citation verification.
- S1: Table 9 classifier-table clarity.

### Related Work

**Status:** Good.

Concise and appropriate.
No active issue.

### Conclusion

**Status:** Good.

One paragraph and focused on the main contribution.
No active issue.

### References and Appendix

**Status:** Needs cleanup.

This is currently the weakest part of the compiled PDF because it contains stale appendix text and bibliography metadata issues.
Active remaining issues:

- M2, M3, M6, M7, S2, S3.

---

## Final pre-submission checklist

### Required

- [x] M1: Add model-predicted caveat to Figure 2. **(caption caveat done; hardware-validation table+paragraph still commented — restore decision pending)**
- [x] M2: Fix Appendix D `≈5 ms` KV-transfer statement.
- [x] M3: Remove “frozen” from appendix prose.
- [x] M4: Fix `Seqential` and motivation grammar.
- [x] M5: Fix `modelcomponents` typo. **(N/A in live source)**
- [~] M6: Clean bibliography names and internal notes. **(notes+names+SWEEP authors done; DynamoLLM author list pending)**
- [~] M7: Verify Azure trace citations in compiled PDF. **(Splitwise cite done; diurnal/DynamoLLM citation + PDF verify pending)**

### Recommended

- [ ] S1: Clarify or move Table 9.
- [x] S2: Remove or formalize Appendix B script path.
- [ ] S3: Check appendix/reference ordering.
- [x] S4: Optional Table 1 layout polish.

### Final PDF scan

- [ ] Search PDF/source for `frozen`.
- [ ] Search PDF/source for `Seqential`.
- [ ] Search PDF/source for `modelcomponents`.
- [ ] Search PDF/source for `5 ms` and `few milliseconds`.
- [ ] Search PDF/source for `SLO violation` and replace with `modeled violation` where referring to main evaluation.
- [ ] Scan all captions for overly long or stale wording.
- [ ] Scan bibliography for notes, corrupted names, and arXiv/year inconsistencies.
# SWEEP-LLM Current Paper Issue Tracker

This file tracks only the issues that remain after the latest 15-page compiled `main.pdf` review.
Old first-round trim-plan content has been removed so this file can be used as a live checklist.
Use `[x]` only after the issue is fixed in source and verified in the compiled PDF.

---

## Third-round status summary

**Reviewed artifact:** latest compiled `main.pdf`, 15 pages.

**Overall status:** The paper is now structurally strong.
The main story, evaluation flow, related work, and conclusion are in good shape.
The remaining issues are mostly final cleanup, PDF/source consistency, citation cleanup, and a few visible typos.

**Current expected review posture:** borderline accept / weak accept potential if the remaining visible issues are fixed.
The biggest remaining technical reviewer risk is still measurement-calibrated simulation rather than a full deployed system, but the paper now states that limitation clearly.

---

## Must fix before submission

### M1. Figure 2 model-predicted caveat

**Status:** Fixed in compiled PDF.

**Evidence from latest PDF:** Figure 2 caption now states that the study is model-predicted and built from hardware-calibrated per-configuration models.
This addresses the earlier circularity concern for the motivation experiment.

- [x] Add caveat in source.
- [x] Verify caveat appears in compiled PDF.

### M2. Appendix D KV-transfer statement

**Status:** Fixed in compiled PDF.

**Evidence from latest PDF:** Appendix D now says the transfer time is at most “tens of milliseconds for the longest synthetic prompts,” not `≈5 ms`.
This matches the main-text wording.

- [x] Fix Appendix D transfer-time statement.
- [x] Search for stale `5 ms` / `few milliseconds` KV-transfer claims.

### M3. Remove active “frozen” wording

**Status:** Fixed in compiled PDF.

**Evidence from latest PDF:** Appendix D now says “main cross-strategy comparison,” not “frozen cross-strategy comparison.”

- [x] Remove active uses of “frozen” from submitted text.

### M4. Fix visible `Seqential` typo in Motivation

**Status:** Still open in compiled PDF.

**Issue:** The latest PDF still visibly contains `Seqential` in Section 3.2.
It appears in the motivation text around the sequential-vs-joint comparison.
Even if source search previously showed 0 hits, the compiled PDF proves the typo is still active through a macro, generated label, figure text, or stale source path.

**Action:** Search all active source files, figure/table generation scripts, and macros for variants of the misspelling.
Likely searches:

```bash
grep -Rni "Seqential\|seqential\|SEQENTIAL" .
```

Also check generated figure/table text if Figure 2 is produced from a script.

- [ ] Fix all rendered `Seqential` occurrences to `Sequential`.
- [ ] Recompile and verify the compiled PDF no longer shows `Seqential`.

### M5. Fix `modelcomponents` typo in Models section

**Status:** Still open in compiled PDF.

**Issue:** The latest PDF still shows `modelcomponents` without a space near the end of Section 5.
This means the compiled source is still using a file or line where the typo remains, even if another source file has already been fixed.

**Action:** Search all active model files and main inputs.
Likely searches:

```bash
grep -Rni "modelcomponents\|model components" .
```

Then verify which model file is actually included by `main.tex`.

- [ ] Fix the active source to say `model components`.
- [ ] Recompile and verify the typo is gone from the PDF.

### M6. Clean bibliography metadata and verify final reference list

**Status:** Mostly fixed; still needs final manual PDF scan.

**What is fixed in latest PDF:**

- Internal explanatory `note` fields no longer appear in the rendered references.
- The corrupted SWEEP author string is fixed.
- The suspicious `Ínigo Inardova` entry is fixed to `Íñigo Goiri`.

**Remaining checks:**

- Verify the DynamoLLM author list is complete and correct.
- Verify arXiv/year/venue details for very recent papers.
- Manually scan the rendered bibliography once more after final compilation.

- [x] Remove internal explanatory `note` fields.
- [x] Fix visibly corrupted author names caught in previous PDF.
- [ ] Verify DynamoLLM author list.
- [ ] Final manual scan of rendered bibliography.

### M7. Verify Azure trace citation coverage

**Status:** Still open / needs decision.

**Issue:** The latest PDF production-trace setup still cites only Splitwise `[12]`, while Appendix B discusses a one-week Azure conversation trace.
If the one-week trace comes from DynamoLLM/Azure 2024, the paper should cite DynamoLLM either in the production-trace setup sentence or at the Appendix B introduction of the one-week trace.

**Action options:**

1. If all Azure traces used in this paper come from the Splitwise/Azure 2023 trace release, keep `[12]` and mark this resolved.
2. If the one-week diurnal trace is from the DynamoLLM/Azure 2024 trace release, cite DynamoLLM when introducing the one-week trace.

Recommended wording for Appendix B if the one-week trace is DynamoLLM/Azure 2024:

> ... from the one-week Azure conversation trace used by DynamoLLM~\cite{stojkovic2024dynamollm} ...

- [ ] Decide which Azure trace release each production experiment uses.
- [ ] Add DynamoLLM citation if the one-week trace is from Azure 2024 / DynamoLLM.
- [ ] Recompile and verify citation coverage.

---

## Should fix / high-value cleanup

### S1. Clarify or move Table 9 feasibility-classifier details

**Status:** Still open.

**Issue:** Table 9 remains in the main text and still shows several `FS% = 100.0%` rows with very small false-safe counts.
This can alarm reviewers because the denominator is unclear.
The prose already carries the more important result: adding the throughput-saturation gate reduces decision-level false-safe rate from `34.5%` to `6.5%` and saturated false-safe selections from `91` to `2`.

**Recommended fix:** Move Table 9 to the appendix, or rename/clarify the `FS%` denominator in the caption.

Possible caption clarification if keeping it in main text:

> FS% is the fraction of admitted-but-unsafe configurations among the unsafe subset for that GPU/phase/SLO cell; counts should be read together with the absolute false-safe column.

Better option:

> Move Table 9 to appendix and keep the saturation-gate summary in the main text.

- [ ] Clarify `FS%` denominator or move Table 9 to appendix.
- [ ] Verify the main text still says optional classifiers are not used for the main cross-strategy comparison.

### S2. Appendix B internal script path

**Status:** Fixed in compiled PDF.

**Evidence from latest PDF:** The internal script path no longer appears in Appendix B.

- [x] Remove internal script-path reference.

### S3. Appendix/reference ordering

**Status:** Open, venue-dependent.

**Issue:** The compiled PDF places references before appendices.
This may be acceptable, but target venues differ.

- [ ] Check target venue requirements.
- [ ] Adjust appendix placement only if required.

### S4. Table 1 layout polish

**Status:** Mostly fixed.

**Evidence from latest PDF:** The Decode_Heavy row is shorter, but still wraps slightly because the table is narrow.
This is acceptable and not a blocker.

- [x] Shorten Decode_Heavy row text.
- [ ] Optional: further reduce table text only if final layout still bothers you.

### S5. Placeholder conference metadata

**Status:** Open but likely ignored until final template pass.

**Issue:** The PDF still shows `Conference'17, July 2017, Washington, DC, USA` in the header.
This is the ACM template placeholder.

**Action:** Ignore during drafting if intentional, but remove/fill correctly before submission.

- [ ] Replace placeholder conference metadata before final submission.

---

## Current section-by-section status

### Abstract and Introduction

**Status:** Good.

The abstract and introduction correctly frame the measurement-calibrated simulator, common utilization-based modeled feasibility gate, and feasible/Pareto frontier results.
Avoid adding stronger deployed-SLO guarantee language.

### Background

**Status:** Good.

Concise and sufficient.
No active issue.

### Motivation

**Status:** Strong but still has one visible typo.

Active remaining issue:

- M4: `Seqential` still appears in the compiled PDF.

Figure 2’s model-predicted caveat is now visible and acceptable.

### Design

**Status:** Good.

The design is coherent.
All four routes remain admissible in all states.
Table 1 is acceptable after row shortening.

### Models

**Status:** Good but needs typo fix.

Active remaining issue:

- M5: `modelcomponents` still appears in compiled PDF.

### Evaluation

**Status:** Strong.

The evaluation flow remains disciplined.
Synthetic, production, ablation, validation, and overhead each serve a distinct purpose.

Remaining evaluation-related issue:

- S1: Table 9 may confuse reviewers unless clarified or moved.

### Related Work

**Status:** Good.

Concise and appropriate.
No active issue.

### Conclusion

**Status:** Good.

One paragraph and focused on the main contribution.
No active issue.

### References and Appendix

**Status:** Mostly cleaned, but still needs final citation/reference verification.

Active remaining issues:

- M6: final bibliography verification.
- M7: Azure trace citation decision.
- S3: appendix/reference ordering check.
- S5: conference metadata finalization.

---

## Final pre-submission checklist

### Required

- [x] M1: Add model-predicted caveat to Figure 2.
- [x] M2: Fix Appendix D KV-transfer statement.
- [x] M3: Remove active “frozen” wording.
- [ ] M4: Fix rendered `Seqential` typo.
- [ ] M5: Fix rendered `modelcomponents` typo.
- [~] M6: Final bibliography verification.
- [ ] M7: Resolve Azure one-week trace citation.

### Recommended

- [ ] S1: Clarify or move Table 9.
- [x] S2: Remove Appendix B script path.
- [ ] S3: Check appendix/reference ordering.
- [x] S4: Table 1 layout polish.
- [ ] S5: Replace placeholder conference metadata before final submission.

### Final PDF/source search commands

Run these after the next compile:

```bash
grep -Rni "Seqential\|seqential\|SEQENTIAL" .
grep -Rni "modelcomponents" .
grep -Rni "frozen" .
grep -Rni "5 ms\|few milliseconds" .
grep -Rni "SLO violation" .
```

Then manually scan:

- Figure 2 caption.
- Section 5 final paragraph.
- Production trace citations.
- Table 9 caption/placement.
- Bibliography.
- Appendix B and Appendix D.