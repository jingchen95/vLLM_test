

# Same-Family Multi-Size Pilot Checklist

## Purpose
This file tracks the internal go/no-go criteria for the same-family multi-size pilot.
It is **not** paper text. Its purpose is to keep the pilot scoped, interpretable, and useful for deciding whether same-family multi-size evidence should appear in the paper.

---

## 1. Smoke Run Success Criteria
The smoke run is only meant to verify that the pipeline works end to end.
It is considered successful only if all of the following hold:

### Collection
- [ ] L40S results are produced
- [ ] L4 results are produced
- [ ] prefill-only runs complete
- [ ] decode-only runs complete
- [ ] result files contain timing / energy fields expected by the summarization script

### Summarization
- [ ] summarization script runs without manual patching
- [ ] summarized pilot CSV is produced
- [ ] phase labels, GPU labels, TP, frequency, lengths, and rate fields are populated correctly

### Evaluation
- [ ] zero-shot evaluation runs successfully
- [ ] oracle evaluation runs successfully
- [ ] threshold-adapt evaluation runs successfully (if enabled)
- [ ] evaluation outputs are written in machine-readable form

### Failure handling
If any item above fails, stop and fix the pipeline before launching a larger pilot.
Do **not** proceed to intermediate or full pilot collection.

---

## 2. Intermediate Pilot Go/No-Go Criteria
The intermediate pilot is worth running only if the smoke run passes and the resulting outputs are interpretable.

We proceed to the intermediate pilot if the smoke run suggests that the comparison modes can answer the following questions clearly:
- Does zero-shot reuse degrade on the larger model size?
- Is decode the first phase to fail?
- Does oracle retraining recover a meaningful amount of lost quality?
- Is threshold-only adaptation clearly insufficient or only partially helpful?

### Evidence that justifies an intermediate pilot
- [ ] zero-shot shows noticeable degradation relative to oracle in at least one important metric
- [ ] decode looks at least as hard as prefill, preferably harder
- [ ] outputs are stable enough that the comparison is interpretable
- [ ] no obvious pipeline artifact dominates the results

### Evidence that argues against continuing
- [ ] zero-shot and oracle are nearly identical
- [ ] all three modes fail in the same way, suggesting a pipeline problem rather than a transfer problem
- [ ] smoke data are too noisy or incomplete to support interpretation

---

## 3. What to Inspect First When Results Arrive
Check these in order.

### A. Feasibility / safety
- [ ] false-safe
- [ ] false-unsafe
- [ ] admit rate
- [ ] safe precision
- [ ] unsafe admission rate

### B. Prediction quality
- [ ] TTFT error for prefill
- [ ] TPOT error for decode
- [ ] power error

### C. Scheduler usefulness
- [ ] top-1 match
- [ ] regret / excess energy relative to measured best feasible config
- [ ] pruning ratio or oracle rejection rate

### D. Failure localization
- [ ] which GPU type fails first?
- [ ] which phase fails first?
- [ ] is the hardest region decode-side, especially on L4?
- [ ] are failures concentrated at specific SLO / TP / frequency / rate combinations?

---

## 4. Decision Rules for Paper Claims
Use the pilot results conservatively.

### Case A: only smoke run succeeds
Do **not** claim same-family multi-size support in the paper.
At most, note that the pipeline exists and that a pilot is in progress.

### Case B: intermediate pilot is informative but limited
Possible paper claim:
- a **same-family multi-size pilot** suggests that zero-shot reuse degrades and that limited adaptation may be promising

Do **not** claim robust same-family transfer.

### Case C: pilot clearly shows zero-shot failure and oracle recovery
Possible paper claim:
- same-family multi-size extension appears feasible with limited additional profiling / recalibration

Still keep the wording modest unless the evidence is very strong.

### Case D: threshold-only adaptation closes little of the gap
Interpretation:
- same-family multi-size likely requires more than guard retuning alone
- this is still useful evidence for the paper

---

## 5. Current Strategic Rule
Until the pilot produces convincing results:
- keep the main paper narrative centered on the current single-family / single-size deployment
- treat same-family multi-size as an extension or pilot
- do not rewrite the main contribution around transfer

---

## 6. Immediate Next Step After Smoke Completes
1. Verify the smoke success criteria above.
2. Inspect zero-shot vs oracle first.
3. Decide whether to launch the decode-focused intermediate pilot.
4. Only then consider a larger same-family profiling run.