# XPYD Stage D1 — Closure

**Status:** CLOSED without the planned fairness rerun  
**Date:** 2026-10-08  
**Decision owner:** PI

## Decision

Stage D1 is closed. The planned final rerun with fair same-phase exploration
and endpoint-local mixed-clock evidence (preflight in
`XPYD_D1_FAIRNESS_PREFLIGHT.md`, commit `6243194`) will not be executed. D2
(TP-diverse inventory) is paused until the regime diagnosis (Stage E0) is
complete.

## What D1 established

- Job `272628` is a valid measurement of the original D1 policy
  (`XPYD_STAGE_D1_RESULTS.md`): SHARED 8.6% / 8.0%, PER-ENDPOINT 5.1% / 5.0%
  (one HIGH repeat fails TPOT at 94%), JOINT 2.2% / 3.3% energy saving versus
  STATIC-HIGH-RR at MEDIUM / HIGH-SUSTAINABLE.
- The diagnostic audit (`XPYD_D1_DIAGNOSTIC_AUDIT.md`) showed that the
  PER-ENDPOINT/JOINT gap is an exploration-opportunity defect (P1 never probed
  MID), not a physical result.
- Mechanism-level findings that remain valid: validated-safe state must be
  revocable when routing changes local load; recovery must be routing-atomic;
  routing and online DVFS learning are coupled through evidence allocation.

## Why the rerun is not worth running

1. **The experiment cannot discriminate the hypothesis.** P0/P1 are identical
   L40S GPUs and D0/D1 identical L4 GPUs under balanced round-robin load and a
   stationary trace. The optimal per-endpoint configuration is therefore the
   symmetric one, i.e. SHARED. With corrected fairness, PER-ENDPOINT can at
   best match SHARED while paying more exploration; it cannot show a benefit of
   independent endpoint control.
2. **Routing has no energy lever in this regime.** All four GPUs stay powered
   and clock-locked. Step 1 (`XPYD_STEP1_ENERGY_DECOMPOSITION.md`) shows that
   95–100% of every D1 P-side saving is an idle-floor reduction and that
   JOINT's only measurable routing effect was +~21 J/request D-side
   load-dependent energy. Phase 4B already measured routing-only at
   −1.0% to +0.4%.
3. **The preflight gate already failed for a frozen-rule reason.** The replay
   assigns P1 a coarse HIGH→LOW jump in MEDIUM repeat 2 (11.41% headroom >
   10% threshold); forcing MID would require a new frontier rule, i.e. further
   policy change after seeing data.

## What is preserved

- Job `272628` remains the original-policy result.
- The fairness arbiter and endpoint-local clock evidence (commit `6243194`,
  343 CPU tests passing at that commit) are kept for a future setting where
  endpoint asymmetry is real (heterogeneous or TP-diverse endpoints, skewed
  load).

## Allowed claims

- Per-endpoint control requires revocable safety state and routing-atomic
  recovery.
- Routing and feedback learning are coupled through evidence allocation.
- The original D1 policy saved less than SHARED, and the audit attributes the
  gap to exploration arbitration.

Not allowed: that per-endpoint DVFS or joint routing is better or worse than
shared control. D1 as designed cannot answer that question.

## Lesson carried forward

Before each new stage, check that (i) the design can discriminate the
hypothesis (no symmetric setup for an asymmetry question), (ii) the oracle
upper bound exceeds the best simple baseline by a meaningful margin, and
(iii) the strongest simple baselines are included (default governor,
STATIC-TUNED, colocated serving).
