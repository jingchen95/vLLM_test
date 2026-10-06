# XPYD D1 fairness correction preflight

## Status

**STOPPED BEFORE GPU EXECUTION.** CPU tests pass and the repeated P0 opportunity bias is removed, but the exact replay gate requiring a P1 MID recommendation fails in one of the six original blocks. No final D1 rerun has been submitted. Job 272628 and its original-policy results remain unchanged. D1 is not yet closed, and D2 has not started.

The frozen coarse-jump rule explains the remaining conflict: MEDIUM repeat 2 gives P1 11.41% observed TTFT headroom at HIGH, so the unchanged adaptive controller recommends HIGH to LOW when it finally receives permission. Making that recommendation MID would require an explicit breadth-first frequency-frontier constraint, not merely fair endpoint selection. No such additional constraint or threshold change was applied after observing the replay.

## Scientific question and frozen contract

The proposed final matched experiment asks whether fair same-phase opportunities and endpoint-local clock evidence change independently learned DVFS residency and measured energy. It does not aim to make JOINT win. A useful result could show that both independently measured P endpoints reach MID and the original gap shrinks, or that routing/context/safety constraints still limit independent control. Either physical outcome would close D1 without another policy revision.

Only arbitration and endpoint-local mixed-clock evidence have changed. The original config, workload, request classes, routing hierarchy, ENERGY_PREFERENCE calculation, context inference, SLO, evidence thresholds, action spaces and five-window horizon are unchanged. The intended final matrix remains:

| Variable | Frozen value |
| --- | --- |
| Hardware | Neptune P0/P1 L40S, Europa D0/D1 L4, TP=1 |
| Policies | STATIC-HIGH-RR, SHARED-FAST-RR, PER-ENDPOINT-FAST-RR, JOINT-ROUTING-PER-ENDPOINT |
| Loads | MEDIUM 0.10 req/s, HIGH-SUSTAINABLE 0.20 req/s |
| Repeats and horizon | 3 matched repeats, 5 windows/block, 20 requests/window |
| Concurrency and mix | 4; 35/15/35/15 SS/SL/LS/LL |
| SLO | TTFT <=600 ms, TPOT <=65 ms, each >=95% request attainment |
| P action space | 1260, 1890, 2520 MHz |
| D action space | 750, 1125, 1500 MHz |
| Headroom rules | one step >=0.01; configured two-level jump >=0.10 |
| Cold start and traces | Every block HIGH; original trace generation/seeds and policy order |

## Implemented corrections

### Phase-local exploration history

`PhaseExplorationArbiter` maintains explicit consumed opportunities and successfully actuated probe-target history per endpoint, separately for P and D, reset at each cold-start block. Probe history records opportunities, not safety proof. Context changes do not erase the opportunity history or transfer another endpoint's safety evidence.

Priority is shallowest previously attempted frequency depth, then fewest physical probes, then fewest consumed opportunities, then a rotating position in the caller-supplied phase order. Endpoint names and generic `windows_observed`/`requests_observed` counters never break recurring ties. The supplied order can seed P0 or P1 first; reversing it reverses the first opportunity symmetrically. An active unvalidated candidate still denies any new same-phase probe. A peer must remain validated-safe; resolved/converged controllers do not consume a new exploration slot.

For balanced evidence with one-step headroom, P0 receives MID after window 1, its candidate is independently accepted after window 2, and P1 receives the next slot after window 3 rather than P0 taking a deeper LOW probe. P1 is still unvalidated at the moment of that recommendation. Only its own subsequent physical observations can prove safety.

### Endpoint-local clock evidence

The D1 measurement row now records `endpoint_mid_window_transition_json`, derived from actual accounting-window clock segments. Independent efficiency classification and endpoint frequency-qualified energy/latency feedback use the affected endpoint's flag. P0 transition evidence is excluded without excluding stable P1; the same applies with identities reversed and for D0/D1. Shared-pool control remains phase-scoped, and class/pair feedback retains its existing conservative whole-pair mixed-clock exclusion.

Missing endpoint flags remain ineligible; legacy records without the new field retain conservative phase-wide fallback. All existing pressure, request/token, freshness, health, readback and SLO gates remain in effect. NVML gross-energy integration and the clock segment ledger were not changed.

## CPU validation

Complete CPU suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=paper/scripts \
  python3 -m unittest discover -s tests
```

Result: **343 tests passed**, including nine new test methods and symmetric subcases. `git diff --check` passes.

New regression coverage includes fair first/second opportunities, denial of deeper priority before an untested peer, independence from generic evidence counters, rotating name-independent ties, reversed endpoint ordering, both P/D phases, failed-probe handoff, safe-fallback/no-concurrent-candidate invariants, real mocked efficiency-path validation, physical segment-derived flags, clean-peer eligibility during another endpoint's transition, missing clock flags, and preservation of the frozen coarse-jump rule. Existing fast-safety, previous-safe-anchor, revocation, recovery publication and readback-failure tests continue to pass.

## Diagnostic replay and stop condition

Replay source: original job 272628 `stage_d1_full_212c925`, using its requests, window measurements, controller decisions and clock segments. [d1_fairness_272628_replay.json](d1_fairness_272628_replay.json) contains all P decisions and source/config SHA-256 values.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=paper/scripts \
  python3 -m xpyd.d1_fairness_replay \
  /path/to/original/stage_d1_full_212c925
```

This is a P-side opportunity replay, holding original observed D behavior exogenous. It consumes only windows 1–3, whose actual P frequencies match the replay state. Windows 1–2 reproduce the original P recommendations. The changed recommendation occurs after window 3, for a hypothetical window 4. Replay stops there: it never substitutes original P1 HIGH observations for counterfactual P1 MID/LOW observations, never actuates hardware, and never infers counterfactual energy or safety. Its command exits nonzero because the exact all-block MID gate fails.

| Load | Repeat | Initial P0 opportunity | Next P opportunity | P1 HIGH TTFT headroom after w3 | Exact MID gate |
| --- | --- | --- | --- | --- | --- |
| MEDIUM | 1 | MID for w2 | P1 MID for w4 | 4.33% | PASS |
| MEDIUM | 2 | MID for w2 | P1 LOW for w4 | 11.41% | FAIL |
| MEDIUM | 3 | MID for w2 | P1 MID for w4 | 4.23% | PASS |
| HIGH-SUSTAINABLE | 1 | MID for w2 | P1 MID for w4 | 8.95% | PASS |
| HIGH-SUSTAINABLE | 2 | MID for w2 | P1 MID for w4 | 7.01% | PASS |
| HIGH-SUSTAINABLE | 3 | MID for w2 | P1 MID for w4 | 4.32% | PASS |

All six blocks hand the second exploration opportunity to P1; none repeatedly gives P0 the deeper slot. In MEDIUM repeat 2, P1 has 10 valid requests, 10,880 input tokens, 100% TTFT attainment, and p95 TTFT 531.55 ms. `(600 - 531.55)/600 = 11.41%`, which exceeds the frozen `headroom_for_large_jump=0.10`. The controller therefore selects 1260 MHz rather than 1890 MHz. This is not evidence starvation, an unsafe MID measurement, or a replay identity error.

The user required stopping if replay did not demonstrate both MID opportunities. That condition is honored: **NO GPU JOB LAUNCHED**. A strict MID-first contract needs explicit authorization for a phase-local breadth-first frontier cap on a coarse-jump recommendation. Numeric headroom thresholds need not change, but that cap would constrain the current coarse-jump candidate selection. It has not been silently implemented to satisfy the gate.

## Changed files and preserved results

- `paper/scripts/xpyd/d1_control.py`: explicit phase-local opportunity/probe-history arbiter.
- `paper/scripts/xpyd/stage_d1.py`: block-local arbiter lifecycle/logging and endpoint-local mixed-clock eligibility.
- `tests/test_xpyd_d1_control.py` and `tests/test_xpyd_stage_d1.py`: focused regression coverage.
- `paper/scripts/xpyd/d1_fairness_replay.py`: read-only replay with a failing exact-MID launch gate.
- `paper/d1_fairness_272628_replay.json` and this preflight report: diagnostic artifacts only.

`XPYD_STAGE_D1_RESULTS.md`, `XPYD_D1_DIAGNOSTIC_AUDIT.md`, original scientific data, experiment configs, adaptive-controller thresholds and routing code are unchanged. No new physical energy/residency/SLO comparison exists yet. When the replay contract is resolved, any authorized final run must use a new run directory/report and retain 272628 as the original-policy baseline; no intermediate-result tuning or automatic D2 launch is permitted.
