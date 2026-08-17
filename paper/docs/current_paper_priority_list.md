

# Current Paper Priority List

## Goal
Keep the paper scoped and executable.
The main narrative should remain centered on the current single-family, single-size SWEEP-LLM deployment. Any multi-size evidence is an extension, not the main backbone of the paper.

---

## Must-have
These items are required for the paper to be complete.

### 1. Main system evaluation on Mistral 7B
- [ ] controlled synthetic trace evaluation completed
- [ ] production trace evaluation completed
- [ ] main baselines finalized
- [ ] main result figures/tables generated
- [ ] main energy / SLO / throughput comparisons written into the paper

### 2. Core paper sections stabilized
- [ ] motivation finalized
- [ ] models section finalized for the current deployment-specific stack
- [ ] evaluation setup section filled with concrete experimental details
- [ ] baselines and metrics described consistently across sections
- [ ] contribution and limitation wording aligned with actual evidence

### 3. Model stack validated well enough for paper use
- [ ] phase-specific feasibility / latency / power stack frozen for v1
- [ ] guard-band calibration results saved and interpretable
- [ ] decision-quality validation included in the paper narrative
- [ ] no unresolved blocker in the current single-size scheduler path

---

## Should-have
These items would materially strengthen the paper and should be pursued after the must-have items are under control.

### 4. Same-family larger-variant pilot on Mistral 12B
- [ ] smoke run passes end to end
- [ ] decode-focused intermediate pilot completed
- [ ] zero-shot vs oracle vs threshold-adapt comparison completed
- [ ] decision on whether to include the pilot in the paper is made

### 5. Multi-size claim policy for the paper
- [ ] if pilot evidence is weak, keep same-family multi-size as future work or a brief pilot note
- [ ] if pilot evidence is convincing, include it as an exploratory extension
- [ ] do not let this pilot redefine the main contribution of the paper

---

## Nice-to-have
These items are useful only if time remains after the main paper and the Mistral 12B pilot are on track.

### 6. Qwen2.5 7B -> 14B minimal corroboration pilot
- [ ] only attempt this if Mistral 12B pilot is already informative
- [ ] keep it minimal and targeted
- [ ] use it as corroborating evidence, not as a second full mainline evaluation

### 7. Extra model/generalization analysis
- [ ] deeper decode hard-bin analysis beyond what is needed for the paper
- [ ] broader same-family or cross-family generalization plans
- [ ] more ambitious transfer-oriented model redesign

---

## Current sequencing rule
Work in this order unless a blocker forces a change:

1. finish the Mistral 7B main evaluation
2. keep the paper text synchronized with the actual current system
3. complete the Mistral 12B smoke run and inspect it
4. if justified, run the decode-focused intermediate Mistral 12B pilot
5. only then decide whether a larger Mistral 12B pilot is worth it
6. only after that consider a minimal Qwen corroboration pilot

---

## Stop conditions
Use these to prevent scope creep.

### Stop adding work if:
- [ ] the main Mistral 7B evaluation is still incomplete
- [ ] a new experiment does not directly reduce a likely reviewer criticism
- [ ] a new experiment would delay the main paper backbone
- [ ] the same-family pilot has not yet produced interpretable evidence

### Safe to expand scope only if:
- [ ] main Mistral 7B results are already solid
- [ ] same-family pilot pipeline is stable
- [ ] the added experiment provides clear reviewer-facing value

---

## Current operating rule
For now:
- keep the paper centered on Mistral 7B
- treat Mistral 12B as the primary extension
- treat Qwen as optional corroboration only