# XPYD Stage C1c correctness preflight review

Date: 2026-09-28

Verdict: **CPU/static preflight PASS.** The stationary C1c implementation is
logically ready for the controlled full-C1c GPU experiment. No GPU, serving,
SSH, Slurm, or NVML job was launched during this task.

## Authoritative-design synchronization

The expanded `XPYD_SYSTEM_CONTROL_DESIGN.md` working copy was synchronized to
the repository first in commit `84793b1`. Its C1c section was then clarified:
C1c is a stationary, evidence-window-bounded safe-search validation. It does
not claim the final event-driven between-window safety architecture; that
execution-semantics change belongs to C2.

## Correctness fixes

1. **Known-unsafe re-probing:** an unsafe bound now permanently constrains the
   current-context search. An adjacent safe/unsafe pair resolves the boundary;
   a wider bracket refines only untested interior indices. Context
   invalidation is the only operation that clears this knowledge.
2. **Headroom gate:** `headroom_for_one_step` is now active. Below it the
   controller holds; between it and `headroom_for_large_jump` it probes one
   level; at/above the large threshold ADAPTIVE may use its frozen coarse jump.
3. **Final-window evidence:** every measured window is consumed exactly once.
   The last window runs a finalization epoch that can accept SAFE evidence or
   record/recover from UNSAFE/risk, but cannot launch an unmeasured downward
   probe.
4. **Unknown pressure:** missing P/D waiting observations remain UNKNOWN.
   Explicit observed zero can satisfy the pressure gate; UNKNOWN cannot make a
   candidate SAFE. Sampled `running==0` is not used as idle evidence.
5. **Measured safe-energy point:** each pool keeps a running measured-energy
   mean and observation count by SAFE frequency, plus the empirically best
   measured SAFE frequency/energy. Unsafe energy is never used. Once search is
   resolved, the controller retains/selects that measured best SAFE point.
6. **Identifiable C1c policy matrix:** offline replay showed that ONE-STEP and
   ADAPTIVE have identical TIGHT-SLO action sequences with the frozen
   thresholds. The primary matrix therefore compares STATIC-HIGH with
   FEEDBACK-SAFE-SEARCH. The legacy `0.5*SLO` `FeedbackScheduler` is not a C1c
   policy, and C1c makes no adaptive-versus-one-step claim.
7. **Window-bounded claim:** controller and reports now explicitly say that
   C1c safety decisions happen at completed-window evidence epochs. No
   event-driven safety claim is made.
8. **Action/readback timing:** recommendation time no longer updates the last
   actual-action timestamp. Physical action start and successful readback time
   come from the actuator result and remain distinct from decision time.
9. **Full-C1c reporting:** output now includes probe/unsafe-probe/recovery
   counts, windows and requests to convergence, final validated and final
   selected frequencies, best measured SAFE-energy frequency, per-window
   cumulative energy/requests/J-request, frequency residency, and request-level
   TTFT/TPOT attainment. `stagec1_search_convergence.csv` is added.

## Files changed

- `paper/XPYD_SYSTEM_CONTROL_DESIGN.md`
- `paper/XPYD_C1_ADAPTIVE_EXPLORATION_DESIGN.md`
- `paper/XPYD_C1_PREFLIGHT_REVIEW.md`
- `paper/configs/xpyd_stagec1_stationary_neptune_europa.json`
- `paper/scripts/xpyd/adaptive_exploration.py`
- `paper/scripts/xpyd/stagec1_stationary.py`
- `tests/test_xpyd_adaptive_exploration.py`
- `tests/test_xpyd_stagec1_stationary.py`

## Focused regression tests

The focused CPU tests cover:

- HIGH safe -> MID probe -> MID unsafe -> HIGH recovery -> no MID re-probe;
- headroom HOLD, one-level, and coarse-jump regions;
- final candidate classification with no unmeasured follow-on probe;
- final state exported into convergence reporting;
- missing endpoint pressure remains UNKNOWN while explicit zero is usable;
- best measured SAFE energy may differ from the lowest SAFE frequency;
- deferred ONE-STEP and ADAPTIVE configurations share gates/SLO/evidence and
  differ only in jump limit;
- balanced two-policy ordering across all six load/repeat pairs;
- policy-independent trace namespaces for exact matched request streams;
- recommendation, physical action, and readback timestamps remain distinct.

Complete CPU-only XpYd suite:

```text
Ran 250 tests in 27.827s
OK
```

`git diff --check` also passed.

## Frozen full-C1c experiment matrix

```text
Topology:             P0/P1 L40S, D0/D1 L4, TP=1
Routing:              fixed four-pair round robin
Frequency state:      shared P pool and shared D pool
P action space:       1260 / 1890 / 2520 MHz
D action space:       750 / 1125 / 1500 MHz
Cold start:           all endpoints HIGH for every block
Concurrency:          4
Workload mix:         35% SS / 15% SL / 35% LS / 15% LL
Loads:                MEDIUM 0.10; HIGH-SUSTAINABLE 0.20 req/s
Windows/block:        4
Requests/window:      20
Repetitions:          3
Primary SLO:          TTFT <= 600 ms; TPOT <= 65 ms; >=95% each
Policies:             STATIC-HIGH / FEEDBACK-SAFE-SEARCH
Total blocks:         12
Total windows:        48
Planned requests:     960
```

Balanced policy order across the six load/repeat pairs:

```text
r1 MEDIUM: STATIC-HIGH -> FEEDBACK-SAFE-SEARCH
r1 HIGH:   FEEDBACK-SAFE-SEARCH -> STATIC-HIGH
r2 MEDIUM: FEEDBACK-SAFE-SEARCH -> STATIC-HIGH
r2 HIGH:   STATIC-HIGH -> FEEDBACK-SAFE-SEARCH
r3 MEDIUM: STATIC-HIGH -> FEEDBACK-SAFE-SEARCH
r3 HIGH:   FEEDBACK-SAFE-SEARCH -> STATIC-HIGH
```

MODERATE and LOOSE SLOs remain sensitivity-analysis results and are not used
in this primary C1c matrix. Any adaptive-versus-one-step experiment is also
deferred to sensitivity work where the frozen policies produce different
actions.

## Scope confirmation

No routing change, TP change, per-endpoint asymmetric DVFS, system-context
controller, predictor, model, bandit, RL, C2 mechanism, or GPU experiment was
added or run.
