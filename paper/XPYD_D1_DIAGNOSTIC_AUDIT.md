# XPYD D1 diagnostic audit — job 272628

## Scope and verdict

Offline audit of `stage_d1_full_212c925` (Neptune L40S P0/P1, Europa L4 D0/D1, TP=1). Job 272628 completed in 06:48:24 with exit code 0:0. The 24 independent cold-start blocks contain 120 accounting windows and 2,400 measured requests. No runtime, controller, routing, SLO, threshold, configuration, or scientific measurement was changed. No GPU job was submitted. Raw artifacts remain unchanged on Minerva.

**Main finding:** the energy gap is primarily an exploration-opportunity / evidence-allocation issue, not evidence that P1 genuinely requires HIGH. PER-ENDPOINT repeatedly grants P0 the probe slot and never measures P1 at MID. JOINT adds routing-conditioned evidence shortfalls, coarse LOW probes, and local-context resets. The implemented guard does not provide the fair endpoint alternation suggested by its comment. These findings do not establish that changing the guard or router would make JOINT win.

Corrected verdict: **MEASUREMENT_AND_ROUTING_VALID_WITH_POLICY_SLO_FAILURE**. Physical/accounting/routing checks pass after a properly block-scoped audit. The HIGH-SUSTAINABLE PER-ENDPOINT repeat 1 TPOT attainment is 94%, a real failure that remains in every result. A pooled 97.33% TPOT attainment does not erase that repeat.

Machine-readable offline corrections are in [d1_diagnostic_272628_summary.json](d1_diagnostic_272628_summary.json). Its `valid=true` means checked measurement/routing/state gates passed, explicitly not that every policy repeat is SLO-safe. `all_policy_repeats_slo_safe=false` preserves the scientific failure. The original `stage_d1_summary.json` was not overwritten.

## 1. Reconstruction method and provenance

- Executed implementation: `212c9256f81e87b911a8b81c3420e7f13c880059`. Last accepted D1 preflight implementation: `7158ff28acf78735555250f5cc85e366725142f4` (job 272172), accepted/documented in `50c2aa7`. The relevant controller/routing/config implementation is identical between preflight and full run.
- Original source directory: `/data/users/chjing/vLLM_test/results/stage_d1/stage_d1_full_212c925/`. Sources: controller decisions, clock actions/segments, safety/eligibility events, endpoint search state, routing decisions, requests, window measurements, repeat summary, and original summary. CSV SHA-256 values are preserved in the corrected JSON.
- Join routes to their physical accounting interval using the routing decision timestamp, then `(block_id, window_index, request_id)`. Logical request IDs are deliberately reused by matched policies; request ID alone is not a unique cross-policy join key.
- Replay the unmodified P-side `AdaptivePoolController` with the logged evidence time, original selected requests and measurements, original exploration permission, and logged successful recovery/actuation readback. All 154 decision/recovery/final-state comparisons match. Replayed states are reconstruction, not additional hardware measurements.
- Check all 48 independent final endpoint search states, not only the final block. All are internally consistent. Check all 480 endpoint/window clock streams: no segment gaps/overlaps and requested/readback clocks agree.
- A normal epoch after window k selects the state for window k+1. The epoch after window 5 is finalization: it consumes evidence but starts no new measured probe. Final restoration to HIGH is outside measured residency.
- Minimum evidence is 8 valid requests, 1,024 input tokens for P, and 1 s observation duration. It is checked against the current preceding window; two insufficient seven-request windows are not combined into a sufficient fourteen-request observation. The evidence timeout is 120 s.
- P context is `long-input-high` at >=75% long-input requests (>512 input tokens), otherwise `mixed-input`. This is derived from the endpoint's assigned requests; the global stationary mix does not imply a stationary local endpoint context.

## 2. Measured energy gap, retaining every repeat

| Load | Policy | J/request, mean ± sample SD | Matched STATIC saving, mean ± SD | SLO-pass repeats |
| --- | --- | --- | --- | --- |
| MEDIUM | STATIC-HIGH-RR | 2615.53 ± 2.03 | baseline | 3/3 |
| MEDIUM | SHARED-FAST-RR | 2389.45 ± 2.77 | 8.64% ± 0.12 pp | 3/3 |
| MEDIUM | PER-ENDPOINT-FAST-RR | 2482.92 ± 26.37 | 5.07% ± 1.00 pp | 3/3 |
| MEDIUM | JOINT-ROUTING-PER-ENDPOINT | 2557.06 ± 12.80 | 2.24% ± 0.53 pp | 3/3 |
| HIGH-SUSTAINABLE | STATIC-HIGH-RR | 1681.40 ± 0.70 | baseline | 3/3 |
| HIGH-SUSTAINABLE | SHARED-FAST-RR | 1546.88 ± 3.37 | 8.00% ± 0.16 pp | 3/3 |
| HIGH-SUSTAINABLE | PER-ENDPOINT-FAST-RR | 1598.14 ± 21.66 | 4.95% ± 1.25 pp | 2/3 |
| HIGH-SUSTAINABLE | JOINT-ROUTING-PER-ENDPOINT | 1625.90 ± 12.26 | 3.30% ± 0.69 pp | 3/3 |

P/D gross energy decomposition (J/request, all repeats/windows included):

| Load | Policy | P pool | D pool | Total | P increase vs SHARED | D change vs SHARED | Net increase vs SHARED |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MEDIUM | STATIC-HIGH-RR | 1595.89 | 1019.64 | 2615.53 | 220.02 | 6.06 | 226.08 |
| MEDIUM | SHARED-FAST-RR | 1375.86 | 1013.59 | 2389.45 | 0.00 | 0.00 | 0.00 |
| MEDIUM | PER-ENDPOINT-FAST-RR | 1496.26 | 986.66 | 2482.92 | 120.39 | -26.93 | 93.46 |
| MEDIUM | JOINT-ROUTING-PER-ENDPOINT | 1530.05 | 1027.01 | 2557.06 | 154.19 | 13.42 | 167.61 |
| HIGH-SUSTAINABLE | STATIC-HIGH-RR | 934.59 | 746.81 | 1681.40 | 130.25 | 4.26 | 134.51 |
| HIGH-SUSTAINABLE | SHARED-FAST-RR | 804.33 | 742.55 | 1546.88 | 0.00 | 0.00 | 0.00 |
| HIGH-SUSTAINABLE | PER-ENDPOINT-FAST-RR | 875.90 | 722.24 | 1598.14 | 71.57 | -20.31 | 51.26 |
| HIGH-SUSTAINABLE | JOINT-ROUTING-PER-ENDPOINT | 899.90 | 725.99 | 1625.90 | 95.57 | -16.56 | 79.02 |

PER-ENDPOINT has lower D energy than SHARED, but its higher P energy more than cancels that benefit: +120.39 P / −26.93 D / +93.46 total J/request at MEDIUM; +71.57 / −20.31 / +51.26 at HIGH. JOINT's P increase is +154.19 J/request at MEDIUM and +95.57 at HIGH. Thus the requested P-side focus is supported by the measured pool decomposition. This is an energy accounting difference, not exact attribution of joules to a particular control decision.

## 3. Did both P endpoints get a real MID opportunity?

| Load | Repeat | Policy | P0 MID measured windows | P1 MID measured windows | Final learned P0/P1 | Reason HIGH remained |
| --- | --- | --- | --- | --- | --- | --- |
| HIGH-SUSTAINABLE | 1 | JOINT-ROUTING-PER-ENDPOINT | 5 | none | M/H | guard denial + local-context invalidation / evidence shortfall; see block timeline |
| HIGH-SUSTAINABLE | 1 | PER-ENDPOINT-FAST-RR | 2,3,4,5 | none | M/H | P1: repeated guard denial; phase-wide mixed-window skip; block finalization |
| HIGH-SUSTAINABLE | 1 | SHARED-FAST-RR | 2,3,4,5 | 2,3,4,5 | M/M | both tested MID; HIGH only cold start |
| HIGH-SUSTAINABLE | 1 | STATIC-HIGH-RR | none | none | H/H | static policy: no exploration by definition |
| MEDIUM | 1 | JOINT-ROUTING-PER-ENDPOINT | 5 | none | M/H | guard denial + local-context invalidation / evidence shortfall; see block timeline |
| MEDIUM | 1 | PER-ENDPOINT-FAST-RR | 2,3,4,5 | none | M/H | P1: repeated guard denial; phase-wide mixed-window skip; block finalization |
| MEDIUM | 1 | SHARED-FAST-RR | 2,3,4,5 | 2,3,4,5 | M/M | both tested MID; HIGH only cold start |
| MEDIUM | 1 | STATIC-HIGH-RR | none | none | H/H | static policy: no exploration by definition |
| HIGH-SUSTAINABLE | 2 | JOINT-ROUTING-PER-ENDPOINT | 2,3 | none | H/H | guard denial + local-context invalidation / evidence shortfall; see block timeline |
| HIGH-SUSTAINABLE | 2 | PER-ENDPOINT-FAST-RR | 2,3,4,5 | none | M/H | P1: repeated guard denial; phase-wide mixed-window skip; block finalization |
| HIGH-SUSTAINABLE | 2 | SHARED-FAST-RR | 2,3,4,5 | 2,3,4,5 | M/M | both tested MID; HIGH only cold start |
| HIGH-SUSTAINABLE | 2 | STATIC-HIGH-RR | none | none | H/H | static policy: no exploration by definition |
| MEDIUM | 2 | JOINT-ROUTING-PER-ENDPOINT | none | none | H/H | guard denial + local-context invalidation / evidence shortfall; see block timeline |
| MEDIUM | 2 | PER-ENDPOINT-FAST-RR | 2,3,4,5 | none | M/H | P1: repeated guard denial; phase-wide mixed-window skip; block finalization |
| MEDIUM | 2 | SHARED-FAST-RR | 2,3,4,5 | 2,3,4,5 | M/M | both tested MID; HIGH only cold start |
| MEDIUM | 2 | STATIC-HIGH-RR | none | none | H/H | static policy: no exploration by definition |
| HIGH-SUSTAINABLE | 3 | JOINT-ROUTING-PER-ENDPOINT | none | 4,5 | H/M | guard denial + local-context invalidation / evidence shortfall; see block timeline |
| HIGH-SUSTAINABLE | 3 | PER-ENDPOINT-FAST-RR | 2,3,4,5 | none | M/H | P1: repeated guard denial; phase-wide mixed-window skip; block finalization |
| HIGH-SUSTAINABLE | 3 | SHARED-FAST-RR | 2,3,4,5 | 2,3,4,5 | M/M | both tested MID; HIGH only cold start |
| HIGH-SUSTAINABLE | 3 | STATIC-HIGH-RR | none | none | H/H | static policy: no exploration by definition |
| MEDIUM | 3 | JOINT-ROUTING-PER-ENDPOINT | 5 | none | M/H | guard denial + local-context invalidation / evidence shortfall; see block timeline |
| MEDIUM | 3 | PER-ENDPOINT-FAST-RR | 2,3,4,5 | none | M/H | P1: repeated guard denial; phase-wide mixed-window skip; block finalization |
| MEDIUM | 3 | SHARED-FAST-RR | 2,3,4,5 | 2,3,4,5 | M/M | both tested MID; HIGH only cold start |
| MEDIUM | 3 | STATIC-HIGH-RR | none | none | H/H | static policy: no exploration by definition |

SHARED physically measures both P endpoints at MID in every load/repeat. PER-ENDPOINT physically measures P0 MID in all six blocks but P1 MID in zero of six. P1 receives exactly ten requests per window (50/block), has no safety revocation, no measured MID-unsafe result, and no P telemetry failure. Under JOINT, P1 reaches MID only in HIGH repeat 3 (windows 4–5), where it is validated safe. P0 skips MID via a two-level HIGH→LOW jump in five of six JOINT blocks; only HIGH repeat 2 initially probes MID directly.

## 4. Why the same-phase guard blocks the second endpoint

The code in `paper/scripts/xpyd/d1_control.py::exploration_permissions` does the following:

```text
if any same-phase candidate is active:
    deny exploration to both endpoints
else:
    choose min(endpoint, key=(windows_observed, requests_observed, endpoint_id))
```

`windows_observed` and `requests_observed` count calls/evidence consumed by `decide`, not how many probes an endpoint was granted. Under balanced RR, both controllers advance those counts equally. Ties are always broken by endpoint ID, selecting P0. The guard is computed once before the endpoint loop; accepting P0 MID within that loop does not release a slot to P1 at the same epoch.

| Window evidence consumed | P0/P1 guard counters before epoch | Permission | Physical result for next window |
| --- | --- | --- | --- |
| w1 | (0 windows,0 requests)/(0,0) | P0 only | w2: P0=MID candidate, P1=HIGH |
| w2 | (1,10)/(1,10); P0 candidate active | neither | P0 MID accepted; w3 remains MID/HIGH |
| w3 | (2,20)/(2,20), no active candidate | P0 only | w4: P0=LOW probe, P1=HIGH |
| w4 | P0 recovers LOW→MID; P-phase mixed flag set | neither endpoint consumes SAFE evidence | w5 remains MID/HIGH; P1's unchanged HIGH evidence also skipped |
| w5 | (3,30)/(3,30) before finalization | P0 tie winner, but exploration disabled for everyone | P0 boundary resolved; P1 remains HIGH and unconverged |

This pattern occurs in all six PER-ENDPOINT blocks. It is not merely unavoidable serialization latency: the priority rule repeatedly selects the same endpoint. With the same balanced evidence, a converged P0 can still win the next tie without launching a probe, so a longer block alone does not guarantee a fair P1 opportunity. This is a concrete control-opportunity defect/design deficiency, not evidence that independent DVFS has a worse physical safe operating point.

A second coupling is the **phase-wide mixed-frequency skip** in `_endpoint_efficiency_control`: a recovery on P0 sets `P_mid_window_transition`, and SAFE classification is skipped for P1 too, despite P1 staying at HIGH. It removes one useful evidence epoch from each PER-ENDPOINT block. This is separate from telemetry failure.

The logged `FINAL_EVIDENCE` trigger is also overloaded: a guard-denied regular epoch passes `allow_exploration=False`, so it logs 'final window classified' even in windows 1–4. In this audit, only actual window-5 finalization is treated as the block ending. Guard denial is reconstructed separately.

## 5. JOINT: local evidence imbalance and context churn

| Load | Endpoint | Assigned requests /300 | SS/SL/LS/LL | Input tokens | ENERGY_PREFERENCE | OTHER routing reasons |
| --- | --- | --- | --- | --- | --- | --- |
| MEDIUM | P0 | 138 | 70/26/29/13 | 98304 | 4 | COLD_START_RR:6; LOWER_PRESSURE:52; EXPLORATION:76 |
| MEDIUM | P1 | 162 | 35/19/76/32 | 228096 | 38 | COLD_START_RR:6; EXPLORATION:39; LOWER_PRESSURE:49; ONLY_SAFE_PAIR:30 |
| HIGH-SUSTAINABLE | P0 | 145 | 50/29/49/17 | 145280 | 6 | COLD_START_RR:6; LOWER_PRESSURE:62; EXPLORATION:61; ONLY_SAFE_PAIR:10 |
| HIGH-SUSTAINABLE | P1 | 155 | 55/16/56/28 | 181120 | 13 | COLD_START_RR:6; EXPLORATION:40; LOWER_PRESSURE:61; ONLY_SAFE_PAIR:35 |

No endpoint receives zero requests in any accounting window. This is **threshold-relative evidence imbalance**, not complete traffic starvation. The decisive MEDIUM candidate windows give P0 only seven short-input requests, 896 input tokens: both minimum request and token coverage fail. P0 LOW is therefore unclassified, not proven SLO-unsafe, and the ~200 s completed window exceeds the 120 s evidence timeout. All three MEDIUM repeats recover P0 LOW→HIGH at that boundary for `EVIDENCE_TIMEOUT`, with 100% observed TTFT attainment.

P0's short-biased HIGH p95 TTFT is often below 540 ms, exceeding 10% headroom and legitimately selecting the frozen two-level HIGH→LOW jump. Once the candidate is active, routing/safety eligibility sends most long-input work to validated HIGH P1. The endpoint contexts then alternate between mixed-input and long-input-high. The controller invalidates prior frequency/energy knowledge on a context-signature change **before** classifying current evidence. This can restore HIGH even when that just-finished window had 100% TTFT attainment.

JOINT records **9 P context-change decisions at MEDIUM and 10 at HIGH**; all 19 logged windows have 100% TTFT attainment. These are not latency violations. For example, HIGH repeat 2 measures P0=MID in windows 2–3, but its window-3 long-input fraction crosses 75%. The controller discards the MID knowledge and recovers HIGH after window 3 (`CONTEXT_CHANGE`, p95 596.20 ms, attainment 100%).

**Does ENERGY_PREFERENCE reinforce the imbalance?** It favors P1 in 38/42 MEDIUM energy-preference decisions and 13/19 HIGH decisions. That is consistent with reinforcing P1's larger assigned-request denominator and local workload load, while occasional preference for validated LOW P0 brings longer requests back and changes its context. However, the initial seven-request LOW candidate shortfall occurs while P0 decisions are `EXPLORATION` and long-input requests predominantly use `ONLY_SAFE_PAIR`, not because LOW P0 was an ENERGY_PREFERENCE winner. Safe eligibility, pressure, and candidate exploration create the initial skew; energy preference is contributory after safety proof, not a proven sole cause. No alternate-routing counterfactual was run.

The five-window horizon then truncates rediscovery: MEDIUM repeats 1/3 do not probe P0 MID until window 5, and finalization validates it without permitting P1 to start. HIGH repeat 3 is the exception where lower prior request counts give P1 the guard slot, and P1 successfully tests MID in window 4.

## 6. Concrete classifications of HIGH residency

| Policy / endpoint | Classification | Evidence |
| --- | --- | --- |
| STATIC P0/P1 | another concrete reason: fixed baseline | Both HIGH by definition; no opportunity requested. |
| SHARED P0/P1 | another concrete reason: cold start | Window 1 HIGH; both MID from window 2, except brief LOW exploration/recovery. |
| PER-ENDPOINT P1, all six blocks | same-phase probe serialization / deterministic arbitration starvation | Ten requests/window, enough tokens, fresh telemetry and 100% TTFT evidence; guard never permits P1. |
| PER-ENDPOINT P1, after w4 | another concrete reason: phase-wide mixed-frequency evidence exclusion | P0 recovery invalidates the SAFE evidence epoch for both P endpoints. |
| PER-ENDPOINT P0 LOW intervals | SLO-unsafe evidence / safety revocation | Fresh hard TTFT violation revokes LOW and restores previously safe MID; not a requirement to return HIGH. |
| JOINT P0, all MEDIUM w2 candidate windows | insufficient evidence / routing-conditioned threshold starvation | Seven requests /896 tokens; 100% TTFT, but insufficient SAFE proof; EVIDENCE_TIMEOUT restores HIGH. |
| JOINT P1, MEDIUM r1/r3 w1 and HIGH r3 w1 | insufficient evidence | Seven assigned requests, below the eight-request gate. |
| JOINT P0/P1 context-change epochs | another concrete reason: local-context invalidation | Nineteen P context resets; frequency safety/energy knowledge cleared despite 100% current attainment. |
| JOINT P0, HIGH r1/r3 | SLO-unsafe evidence / safety revocation | Two LOW→HIGH fast recoveries; no previously validated MID anchor for those recoveries. |
| JOINT P1, five of six blocks | same-phase arbitration + context churn; block ended before exploration | Never actuated MID. Not a measured MID failure. |
| JOINT late MID candidates | block ended before further exploration | Final evidence is consumed, but no sixth measurement window or new unmeasured downward probe exists. |
| P telemetry / actuator failure | not observed as a cause | No P stale/invalid/readback-failure trigger in the logged decisions/events; all physical hard gates pass. |

## 7. P-pair occupancy: windows, request-equivalents, admissions

For each physical accounting window, intersect P0/P1 stable segments chronologically. Keep transition intervals separate; do not pretend clocks are stable during actuation. `window-equivalent = duration_in_state / full_window_duration`; `request-equivalent = 20 × window-equivalent`. These are duration-normalized accounting units, not physical per-request energy attribution or actual counts. Actual admissions are separately counted from routing decision timestamps. Both-MID means both P clocks are 1890 MHz, one-MID/one-HIGH is order-independent, and any-LOW includes either/both at 1260 MHz. The five categories partition each window, with no gap or overlap.

| Load | Policy | Both HIGH | One MID + HIGH | Both MID | Any LOW | Transition |
| --- | --- | --- | --- | --- | --- | --- |
| MEDIUM | STATIC-HIGH-RR | 15.000 win / 300.00 req-eq / 300 admissions | 0.000 win / 0.00 req-eq / 0 admissions | 0.000 win / 0.00 req-eq / 0 admissions | 0.000 win / 0.00 req-eq / 0 admissions | 0.000 win / 0.00 req-eq / 0 admissions |
| MEDIUM | SHARED-FAST-RR | 3.000 win / 60.00 req-eq / 60 admissions | 0.000 win / 0.00 req-eq / 0 admissions | 11.206 win / 224.12 req-eq / 222 admissions | 0.765 win / 15.31 req-eq / 18 admissions | 0.028 win / 0.57 req-eq / 0 admissions |
| MEDIUM | PER-ENDPOINT-FAST-RR | 3.000 win / 60.00 req-eq / 60 admissions | 11.230 win / 224.59 req-eq / 222 admissions | 0.000 win / 0.00 req-eq / 0 admissions | 0.761 win / 15.22 req-eq / 18 admissions | 0.009 win / 0.18 req-eq / 0 admissions |
| MEDIUM | JOINT-ROUTING-PER-ENDPOINT | 8.000 win / 160.00 req-eq / 160 admissions | 2.000 win / 40.00 req-eq / 40 admissions | 0.000 win / 0.00 req-eq / 0 admissions | 5.000 win / 100.00 req-eq / 100 admissions | 0.000 win / 0.00 req-eq / 0 admissions |
| HIGH-SUSTAINABLE | STATIC-HIGH-RR | 15.000 win / 300.00 req-eq / 300 admissions | 0.000 win / 0.00 req-eq / 0 admissions | 0.000 win / 0.00 req-eq / 0 admissions | 0.000 win / 0.00 req-eq / 0 admissions | 0.000 win / 0.00 req-eq / 0 admissions |
| HIGH-SUSTAINABLE | SHARED-FAST-RR | 3.000 win / 60.00 req-eq / 60 admissions | 0.000 win / 0.00 req-eq / 0 admissions | 11.522 win / 230.44 req-eq / 228 admissions | 0.452 win / 9.05 req-eq / 12 admissions | 0.026 win / 0.51 req-eq / 0 admissions |
| HIGH-SUSTAINABLE | PER-ENDPOINT-FAST-RR | 3.000 win / 60.00 req-eq / 60 admissions | 11.227 win / 224.55 req-eq / 220 admissions | 0.000 win / 0.00 req-eq / 0 admissions | 0.757 win / 15.13 req-eq / 20 admissions | 0.016 win / 0.32 req-eq / 0 admissions |
| HIGH-SUSTAINABLE | JOINT-ROUTING-PER-ENDPOINT | 8.327 win / 166.55 req-eq / 163 admissions | 5.000 win / 100.00 req-eq / 100 admissions | 0.000 win / 0.00 req-eq / 0 admissions | 1.662 win / 33.24 req-eq / 37 admissions | 0.011 win / 0.22 req-eq / 0 admissions |

Each row totals 15 window-equivalents /300 request-equivalents /300 actual admissions. SHARED has roughly 11.2–11.5 both-MID window-equivalents. PER-ENDPOINT replaces these with one-MID/one-HIGH and has zero both-MID. JOINT also has zero both-MID, with about eight both-HIGH windows/load. The per-repeat partition is retained below.

| Load | Repeat | Policy | Both HIGH win-eq | One MID + HIGH win-eq | Both MID win-eq | Any LOW win-eq | Transition win-eq |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HIGH-SUSTAINABLE | 1 | JOINT-ROUTING-PER-ENDPOINT | 2.772 | 1.000 | 0.000 | 1.223 | 0.005 |
| HIGH-SUSTAINABLE | 1 | PER-ENDPOINT-FAST-RR | 1.000 | 3.686 | 0.000 | 0.308 | 0.005 |
| HIGH-SUSTAINABLE | 1 | SHARED-FAST-RR | 1.000 | 0.000 | 3.761 | 0.226 | 0.013 |
| HIGH-SUSTAINABLE | 1 | STATIC-HIGH-RR | 5.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| MEDIUM | 1 | JOINT-ROUTING-PER-ENDPOINT | 3.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| MEDIUM | 1 | PER-ENDPOINT-FAST-RR | 1.000 | 3.743 | 0.000 | 0.254 | 0.003 |
| MEDIUM | 1 | SHARED-FAST-RR | 1.000 | 0.000 | 3.737 | 0.255 | 0.008 |
| MEDIUM | 1 | STATIC-HIGH-RR | 5.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| HIGH-SUSTAINABLE | 2 | JOINT-ROUTING-PER-ENDPOINT | 3.000 | 2.000 | 0.000 | 0.000 | 0.000 |
| HIGH-SUSTAINABLE | 2 | PER-ENDPOINT-FAST-RR | 1.000 | 3.771 | 0.000 | 0.224 | 0.005 |
| HIGH-SUSTAINABLE | 2 | SHARED-FAST-RR | 1.000 | 0.000 | 4.000 | 0.000 | 0.000 |
| HIGH-SUSTAINABLE | 2 | STATIC-HIGH-RR | 5.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| MEDIUM | 2 | JOINT-ROUTING-PER-ENDPOINT | 2.000 | 0.000 | 0.000 | 3.000 | 0.000 |
| MEDIUM | 2 | PER-ENDPOINT-FAST-RR | 1.000 | 3.743 | 0.000 | 0.254 | 0.003 |
| MEDIUM | 2 | SHARED-FAST-RR | 1.000 | 0.000 | 3.734 | 0.255 | 0.010 |
| MEDIUM | 2 | STATIC-HIGH-RR | 5.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| HIGH-SUSTAINABLE | 3 | JOINT-ROUTING-PER-ENDPOINT | 2.555 | 2.000 | 0.000 | 0.439 | 0.006 |
| HIGH-SUSTAINABLE | 3 | PER-ENDPOINT-FAST-RR | 1.000 | 3.770 | 0.000 | 0.224 | 0.005 |
| HIGH-SUSTAINABLE | 3 | SHARED-FAST-RR | 1.000 | 0.000 | 3.761 | 0.226 | 0.013 |
| HIGH-SUSTAINABLE | 3 | STATIC-HIGH-RR | 5.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| MEDIUM | 3 | JOINT-ROUTING-PER-ENDPOINT | 3.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| MEDIUM | 3 | PER-ENDPOINT-FAST-RR | 1.000 | 3.743 | 0.000 | 0.254 | 0.003 |
| MEDIUM | 3 | SHARED-FAST-RR | 1.000 | 0.000 | 3.735 | 0.255 | 0.010 |
| MEDIUM | 3 | STATIC-HIGH-RR | 5.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## 8. SHARED versus independent control on the same timeline

| Accounting window | SHARED actual behavior | PER-ENDPOINT actual behavior | What P1 would need for equivalent both-MID |
| --- | --- | --- | --- |
| w1 | 20 pooled observations at HIGH | 10 P0 +10 P1 HIGH observations | P1 already meets count/token/headroom gates, but P0 owns the one probe slot. |
| w2 | Both P at MID; classified SAFE from pooled evidence | P0 MID candidate; P1 HIGH; active-candidate guard denies both | Complete P0 MID proof, then give P1 a later exclusive slot; do not infer P1 MID safe without measuring it. |
| w3 | Often brief shared LOW probe→MID recovery | P0 MID accepted; P1 still HIGH; tie chooses P0 for next LOW probe | A fair handoff could instead schedule P1 HIGH→MID for w4, with P0 kept at validated MID. This is a logical opportunity, not a measured counterfactual result. |
| w4 | Both MID | P0 LOW→MID recovery; P1 HIGH; both evidence epochs subsequently skipped | Measure P1 MID, then obtain enough SAFE evidence. Existing code never gives it that opportunity. |
| w5 | Both MID safe steady state | P0 MID /P1 HIGH; no further probes at finalization | Retain both MID only if P1's measured MID result is safe; that measurement is absent in this policy. |

Consequently the existing five-window experiment does not identify independently learned P1's MID safety boundary. SHARED proves that both physical endpoints can run MID under the matched workload, but does not transfer that proof into the independent online controllers. No C1b/SHARED safe state was supplied as online controller knowledge.

## 9. Which mechanism primarily explains the energy gap?

1. **SHARED vs PER-ENDPOINT:** deterministic probe arbitration keeps P1 at HIGH despite sufficient evidence, plus a redundant candidate-acceptance epoch and phase-wide mixed-window exclusion. P0's own MID/LOW residency is nearly the same as the SHARED pool at MEDIUM; the missing P1 downshift dominates the P energy difference. This is stronger than merely 'serial exploration takes longer'.
2. **JOINT vs SHARED:** the same opportunity asymmetry plus workload-conditioned coarse LOW probes, per-window insufficient evidence/timeouts, and repeated local-context invalidation reduce sustainable MID residency. Energy-preference selections mostly favor P1, but do not alone explain the initial candidate shortfall.
3. **Safety recovery/revocation:** real LOW risks exist and correctly count in the scientific result. They are not the main reason P1 stays HIGH: independent P1 never revokes a safe point, and most JOINT resets happen without a TTFT violation. MEDIUM JOINT has zero fast safety recoveries but remains substantially worse in P energy.
4. **Genuinely different safe operating points:** not established as the primary explanation. P1 MID was never measured in PER-ENDPOINT or five JOINT blocks. In the one JOINT block where P1 MID is tried, it validates safe. Routing does change local input mixes, so equivalence under all possible endpoint mixes is not claimed.
5. No thresholds were adjusted, no exploratory windows or unfavorable repeats discarded, and no new counterfactual GPU experiment was run. The audit diagnoses the executed policy; it does not assert a revised policy would outperform SHARED.

## 10. Offline corrections and internally consistent acceptance

| Item | Original report | Offline correction |
| --- | --- | --- |
| Recovery counts | Same all-load total repeated in each load row | MEDIUM: STATIC 0, SHARED 6, PER-ENDPOINT 6, JOINT 0. HIGH: 0,5,7,2. Count FAST_RECOVER events, not individual GPU commands. |
| Frequency residency | Segment selection filtered policy only, combining loads | Filter policy AND load; per-repeat tables also filter block. Stable-only residency uses the unchanged original helper; transition seconds are reported separately. |
| Routing chronology | Eight violations /four unique requests, applying prior block revocations to later independent cold starts | Zero block-scoped violations in all 2,400 routing decisions and 15 endpoint recovery intervals. State/knowledge resets at every block. |
| Top-level audit / validity | valid=true alongside failed chronological/revocation booleans | Corrected measurement/routing validity true, both corrected audit booleans true, violation count zero. Status explicitly includes the genuine policy SLO failure. |
| Final endpoint state | Original helper checked latest row per endpoint across the whole run | Check all 48 independent block-end states. All consistent, with final HIGH restoration separately valid. |
| Preflight-only flags | Full output includes preflight_pass=false and asymmetry=false | Not full-run failures: deliberately fixed-asymmetry preflight is not part of the full policy matrix. Excluded from the corrected full verdict. |
| Misleading FINAL_EVIDENCE log text | Regular guard-denied epochs labeled as final | Documentation separates guard-denied ordinary epochs from genuine after-w5 finalization; original controller logs are not edited. |

Why the original eight flags are false: P0 LOW recovery in `r01-medium-per-endpoint-fast-rr` and `r02-medium-per-endpoint-fast-rr` is incorrectly compared with ENERGY_PREFERENCE routes in **another independently cold-started block**, `r02-medium-joint-routing-per-endpoint`, window 5, request suffixes 0001/0007/0011/0015. That JOINT block validates LOW from nine short requests in window 4 before using it in window 5. It later invalidates the point after a local context change at finalization. No prior independent-policy revocation is inherited. The same four requests are flagged twice by the two earlier recovery events.

| Load | Policy | Fast recoveries P/D/total | P0 stable L/M/H % | P1 stable L/M/H % | P transition seconds P0/P1 |
| --- | --- | --- | --- | --- | --- |
| MEDIUM | STATIC-HIGH-RR | 0/0/0 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.000/0.000 |
| MEDIUM | SHARED-FAST-RR | 3/3/6 | 5.08/74.91/20.01 | 5.17/74.80/20.02 | 1.886/3.807 |
| MEDIUM | PER-ENDPOINT-FAST-RR | 3/3/6 | 5.08/74.91/20.01 | 0.00/0.00/100.00 | 1.828/0.000 |
| MEDIUM | JOINT-ROUTING-PER-ENDPOINT | 0/0/0 | 33.43/13.30/53.27 | 0.00/0.00/100.00 | 0.000/0.000 |
| HIGH-SUSTAINABLE | STATIC-HIGH-RR | 0/0/0 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.000/0.000 |
| HIGH-SUSTAINABLE | SHARED-FAST-RR | 2/3/5 | 2.99/77.00/20.02 | 3.10/76.88/20.02 | 1.483/1.477 |
| HIGH-SUSTAINABLE | PER-ENDPOINT-FAST-RR | 3/4/7 | 5.06/74.93/20.01 | 0.00/0.00/100.00 | 1.872/0.000 |
| HIGH-SUSTAINABLE | JOINT-ROUTING-PER-ENDPOINT | 2/0/2 | 11.09/20.00/68.91 | 0.00/13.35/86.65 | 1.262/0.000 |

Full corrected D-side residency is retained in the JSON; no D-side scientific measurements were changed. All 26 fast-recovery events include 15 independent endpoint recoveries and 11 shared-pool recovery events. Shared-pool events are not mistakenly counted as two independent events.

## 11. Exact relevant git diff against accepted preflight

The relevant control/routing diff is **exactly empty (0 bytes)**. This matters: these mechanisms already existed in the accepted preflight implementation; they were not introduced by an unreviewed full-run controller change. Preflight used fixed P0=MID/P1=HIGH and only three measured windows, so it did not exercise cold-start fairness for both independent P endpoints.

```bash
git diff --no-ext-diff 7158ff28acf78735555250f5cc85e366725142f4 \
  212c9256f81e87b911a8b81c3420e7f13c880059 -- \
  paper/scripts/xpyd \
  paper/configs/xpyd_stage_d1_neptune_europa.json
```

Exact stdout: empty. A byte-exact copy is [xpyd_d1_full_vs_accepted_preflight.diff](xpyd_d1_full_vs_accepted_preflight.diff). The same comparison from acceptance-documentation commit `50c2aa7` is also empty. Broader changed paths were only the two D1 documents, the interpreter-pinning submission wrapper, and the launcher's interpreter diagnostic. No exploration, routing, safety, or frequency configuration changed.

For clarity, the existing launcher diagnostic addition is outside control/routing logic. It only prints `Python interpreter: ${PYTHON_BIN}` before checking vLLM. The submission wrapper pins cuda-env/vLLM and verifies accepted-preflight input; neither change explains the policy energy gap.

## Appendix A. Complete P window timelines

L/M/H =1260/1890/2520 MHz. Clock paths list stable segment duration in seconds; `T→freq` is an explicit action/readback transition. Classes are SS/SL/LS/LL counts. Request counts/TTFT statistics are original assigned requests, including violations. Routing reasons are original decisions. State is reconstructed with the unchanged code: `S` is validated-safe frequency set, `C` candidate frequency (`*` active; `~` retained but inactive), `R` revoked set, `K` known-safe anchor. Initial HIGH is a conservative cold-start anchor, not an extra measured calibration. A state after the final epoch can differ from the last measured clock.

### HIGH-SUSTAINABLE, repeat 1, JOINT-ROUTING-PER-ENDPOINT

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:115.991 | 9 | 1/3/4/1 | 10752 | 532.32 /100.0% | COLD_START_RR:2; LOWER_PRESSURE:5; EXPLORATION:2 |
| 1 | P1 | H:115.991 | 11 | 6/0/3/2 | 11008 | 559.44 /100.0% | COLD_START_RR:2; EXPLORATION:4; LOWER_PRESSURE:5 |
| 2 | P0 | L:116.225 | 9 | 5/3/1/0 | 3072 | 471.73 /100.0% | EXPLORATION:9 |
| 2 | P1 | H:116.225 | 11 | 2/0/6/3 | 18688 | 581.66 /100.0% | ONLY_SAFE_PAIR:10; EXPLORATION:1 |
| 3 | P0 | L:25.928; T→H:0.615; H:89.877 | 9 | 0/1/6/2 | 16512 | 589.07 /88.9% | LOWER_PRESSURE:9 |
| 3 | P1 | H:116.420 | 11 | 7/2/1/1 | 5248 | 552.43 /100.0% | EXPLORATION:3; LOWER_PRESSURE:6; ENERGY_PREFERENCE:2 |
| 4 | P0 | H:116.038 | 9 | 3/1/2/3 | 10752 | 578.49 /100.0% | LOWER_PRESSURE:4; EXPLORATION:5 |
| 4 | P1 | H:116.038 | 11 | 4/2/5/0 | 11008 | 586.33 /100.0% | EXPLORATION:1; LOWER_PRESSURE:10 |
| 5 | P0 | M:116.228 | 9 | 5/3/1/0 | 3072 | 438.58 /100.0% | EXPLORATION:9 |
| 5 | P1 | H:116.228 | 11 | 2/0/6/3 | 18688 | 568.91 /100.0% | ONLY_SAFE_PAIR:10; EXPLORATION:1 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | EXPLORE_DOWN:H→L; SEARCH_DOWN | 9 /10752 | K=H, S=H, C=L*, R=none | 2026-10-06T10:23:06.583Z |
| 1 | P1 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 11 /11008 | K=H, S=H, C=none, R=none | 2026-10-06T10:23:06.583Z |
| 2 | P0 | K=H, S=H, C=L*, R=none | P0:deny[1w,9r,candidate]; P1:deny[1w,11r] | ACCEPT_SAFE:L→L; FINAL_EVIDENCE | 9 /3072 | K=L, S=L/H, C=none, R=none | 2026-10-06T10:25:47.477Z |
| 2 | P1 | K=H, S=H, C=none, R=none | P0:deny[1w,9r,candidate]; P1:deny[1w,11r] | HOLD:H→H; CONTEXT_CHANGE | 11 /18688 | K=H, S=H, C=none, R=none | 2026-10-06T10:25:47.477Z |
| 3 | P0 | K=L, S=L/H, C=none, R=none | P0:allow[2w,18r]; P1:deny[2w,22r] | HOLD:—→—; MIXED_FREQUENCY_WINDOW | not consumed /— | K=H, S=H, C=none, R=L | 2026-10-06T10:28:25.974Z |
| 3 | P1 | K=H, S=H, C=none, R=none | P0:allow[2w,18r]; P1:deny[2w,22r] | HOLD:—→—; MIXED_FREQUENCY_WINDOW | not consumed /— | K=H, S=H, C=none, R=none | 2026-10-06T10:28:25.974Z |
| 4 | P0 | K=H, S=H, C=none, R=L | P0:allow[2w,18r]; P1:deny[2w,22r] | EXPLORE_DOWN:H→M; REFINE | 9 /10752 | K=H, S=H, C=M*, R=L | 2026-10-06T10:31:02.450Z |
| 4 | P1 | K=H, S=H, C=none, R=none | P0:allow[2w,18r]; P1:deny[2w,22r] | HOLD:H→H; CONTEXT_CHANGE | 11 /11008 | K=H, S=H, C=none, R=none | 2026-10-06T10:31:02.450Z |
| 5 | P0 | K=H, S=H, C=M*, R=L | P0:deny[3w,27r,candidate]; P1:deny[3w,33r] | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 9 /3072 | K=M, S=M/H, C=none, R=L | 2026-10-06T10:33:43.017Z |
| 5 | P1 | K=H, S=H, C=none, R=none | P0:deny[3w,27r,candidate]; P1:deny[3w,33r] | HOLD:H→H; CONTEXT_CHANGE [final] | 11 /18688 | K=H, S=H, C=none, R=none | 2026-10-06T10:33:43.017Z |

Final measured residency and learned states for this block: P0 L/M/H = 24.50/20.03/55.47%; P1 = 0.00/0.00/100.00%. Final P0 K=M, S=M/H, C=none, R=L; P1 K=H, S=H, C=none, R=none. Final controller consumed counters P0=4 windows/36 requests, P1=4 windows/44 requests.

### HIGH-SUSTAINABLE, repeat 1, PER-ENDPOINT-FAST-RR

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:116.015 | 10 | 3/1/4/2 | 12800 | 575.14 /100.0% | FIXED_RR:10 |
| 1 | P1 | H:116.015 | 10 | 4/2/3/1 | 8960 | 562.86 /100.0% | FIXED_RR:10 |
| 2 | P0 | M:116.023 | 10 | 3/1/4/2 | 12800 | 588.64 /100.0% | FIXED_RR:10 |
| 2 | P1 | H:116.023 | 10 | 4/2/3/1 | 8960 | 543.00 /100.0% | FIXED_RR:10 |
| 3 | P0 | M:116.027 | 10 | 3/1/4/2 | 12800 | 590.89 /100.0% | FIXED_RR:10 |
| 3 | P1 | H:116.027 | 10 | 4/2/3/1 | 8960 | 546.31 /100.0% | FIXED_RR:10 |
| 4 | P0 | L:35.971; T→M:0.638; M:80.001 | 10 | 3/1/3/3 | 12800 | 620.79 /90.0% | FIXED_RR:10 |
| 4 | P1 | H:116.610 | 10 | 4/2/4/0 | 8960 | 552.67 /100.0% | FIXED_RR:10 |
| 5 | P0 | M:116.179 | 10 | 4/1/4/1 | 10880 | 560.81 /100.0% | FIXED_RR:10 |
| 5 | P1 | H:116.179 | 10 | 3/2/3/2 | 10880 | 563.87 /100.0% | FIXED_RR:10 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | EXPLORE_DOWN:H→M; SEARCH_DOWN | 10 /12800 | K=H, S=H, C=M*, R=none | 2026-10-06T11:02:43.709Z |
| 1 | P1 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 10 /8960 | K=H, S=H, C=none, R=none | 2026-10-06T11:02:43.709Z |
| 2 | P0 | K=H, S=H, C=M*, R=none | P0:deny[1w,10r,candidate]; P1:deny[1w,10r] | ACCEPT_SAFE:M→M; FINAL_EVIDENCE | 10 /12800 | K=M, S=M/H, C=none, R=none | 2026-10-06T11:05:24.821Z |
| 2 | P1 | K=H, S=H, C=none, R=none | P0:deny[1w,10r,candidate]; P1:deny[1w,10r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 10 /8960 | K=H, S=H, C=none, R=none | 2026-10-06T11:05:24.821Z |
| 3 | P0 | K=M, S=M/H, C=none, R=none | P0:allow[2w,20r]; P1:deny[2w,20r] | EXPLORE_DOWN:M→L; SEARCH_DOWN | 10 /12800 | K=M, S=M/H, C=L*, R=none | 2026-10-06T11:08:01.388Z |
| 3 | P1 | K=H, S=H, C=none, R=none | P0:allow[2w,20r]; P1:deny[2w,20r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 10 /8960 | K=H, S=H, C=none, R=none | 2026-10-06T11:08:01.388Z |
| 4 | P0 | K=M, S=M/H, C=L*, R=none | P0:allow[3w,30r]; P1:deny[3w,30r] | HOLD:—→—; MIXED_FREQUENCY_WINDOW | not consumed /— | K=M, S=M/H, C=none, R=L | 2026-10-06T11:10:40.637Z |
| 4 | P1 | K=H, S=H, C=none, R=none | P0:allow[3w,30r]; P1:deny[3w,30r] | HOLD:—→—; MIXED_FREQUENCY_WINDOW | not consumed /— | K=H, S=H, C=none, R=none | 2026-10-06T11:10:40.637Z |
| 5 | P0 | K=M, S=M/H, C=none, R=L | P0:allow[3w,30r]; P1:deny[3w,30r] | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 10 /10880 | K=M, S=M/H, C=none, R=L | 2026-10-06T11:13:17.433Z |
| 5 | P1 | K=H, S=H, C=none, R=none | P0:allow[3w,30r]; P1:deny[3w,30r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE [final] | 10 /10880 | K=H, S=H, C=none, R=none | 2026-10-06T11:13:17.433Z |

Final measured residency and learned states for this block: P0 L/M/H = 6.20/73.81/20.00%; P1 = 0.00/0.00/100.00%. Final P0 K=M, S=M/H, C=none, R=L; P1 K=H, S=H, C=none, R=none. Final controller consumed counters P0=4 windows/40 requests, P1=4 windows/40 requests.

### HIGH-SUSTAINABLE, repeat 1, SHARED-FAST-RR

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:116.014 | 10 | 3/1/4/2 | 12800 | 580.44 /100.0% | FIXED_RR:10 |
| 1 | P1 | H:116.014 | 10 | 4/2/3/1 | 8960 | 588.19 /100.0% | FIXED_RR:10 |
| 2 | P0 | M:116.025 | 10 | 3/1/4/2 | 12800 | 582.87 /100.0% | FIXED_RR:10 |
| 2 | P1 | M:116.025 | 10 | 4/2/3/1 | 8960 | 580.37 /100.0% | FIXED_RR:10 |
| 3 | P0 | L:25.958; T→M:0.871; M:89.197 | 10 | 3/1/4/2 | 12800 | 583.42 /100.0% | FIXED_RR:10 |
| 3 | P1 | L:27.096; T→M:0.585; M:88.345 | 10 | 4/2/3/1 | 8960 | 601.74 /90.0% | FIXED_RR:10 |
| 4 | P0 | M:116.023 | 10 | 3/1/4/2 | 12800 | 588.63 /100.0% | FIXED_RR:10 |
| 4 | P1 | M:116.023 | 10 | 4/2/3/1 | 8960 | 597.08 /100.0% | FIXED_RR:10 |
| 5 | P0 | M:116.029 | 10 | 3/1/4/2 | 12800 | 591.82 /100.0% | FIXED_RR:10 |
| 5 | P1 | M:116.029 | 10 | 4/2/3/1 | 8960 | 548.39 /100.0% | FIXED_RR:10 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | shared pool | EXPLORE_DOWN:H→M; SEARCH_DOWN | 20 /21760 | K=H, S=H, C=M*, R=none | 2026-10-06T10:49:28.467Z |
| 1 | P1 | K=H, S=H, C=none, R=none | shared pool | EXPLORE_DOWN:H→M; SEARCH_DOWN | 20 /21760 | K=H, S=H, C=M*, R=none | 2026-10-06T10:49:28.467Z |
| 2 | P0 | K=H, S=H, C=M*, R=none | shared pool | EXPLORE_DOWN:M→L; SEARCH_DOWN | 20 /21760 | K=M, S=M/H, C=L*, R=none | 2026-10-06T10:52:10.060Z |
| 2 | P1 | K=H, S=H, C=M*, R=none | shared pool | EXPLORE_DOWN:M→L; SEARCH_DOWN | 20 /21760 | K=M, S=M/H, C=L*, R=none | 2026-10-06T10:52:10.060Z |
| 3 | P0 | K=M, S=M/H, C=L*, R=none | shared pool | HOLD:—→M; MIXED_CLOCK_EVIDENCE | not consumed /— | K=M, S=M/H, C=none, R=L | 2026-10-06T10:54:49.639Z |
| 3 | P1 | K=M, S=M/H, C=L*, R=none | shared pool | HOLD:—→M; MIXED_CLOCK_EVIDENCE | not consumed /— | K=M, S=M/H, C=none, R=L | 2026-10-06T10:54:49.639Z |
| 4 | P0 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; SEARCH_CONVERGED | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T10:57:26.415Z |
| 4 | P1 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; SEARCH_CONVERGED | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T10:57:26.415Z |
| 5 | P0 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T11:00:03.200Z |
| 5 | P1 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T11:00:03.200Z |

Final measured residency and learned states for this block: P0 L/M/H = 4.48/75.49/20.03%; P1 = 4.68/75.31/20.02%. Final P0 K=M, S=M/H, C=none, R=L; P1 K=M, S=M/H, C=none, R=L. Final controller consumed counters P0=5 windows/100 requests, P1=5 windows/100 requests (same pooled counters, not separate endpoint evidence).

### HIGH-SUSTAINABLE, repeat 1, STATIC-HIGH-RR

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:116.011 | 10 | 3/1/4/2 | 12800 | 576.38 /100.0% | FIXED_RR:10 |
| 1 | P1 | H:116.011 | 10 | 4/2/3/1 | 8960 | 584.04 /100.0% | FIXED_RR:10 |
| 2 | P0 | H:116.022 | 10 | 3/1/4/2 | 12800 | 585.03 /100.0% | FIXED_RR:10 |
| 2 | P1 | H:116.022 | 10 | 4/2/3/1 | 8960 | 569.94 /100.0% | FIXED_RR:10 |
| 3 | P0 | H:116.021 | 10 | 3/1/4/2 | 12800 | 585.46 /100.0% | FIXED_RR:10 |
| 3 | P1 | H:116.021 | 10 | 4/2/3/1 | 8960 | 570.78 /100.0% | FIXED_RR:10 |
| 4 | P0 | H:116.017 | 10 | 3/1/4/2 | 12800 | 576.90 /100.0% | FIXED_RR:10 |
| 4 | P1 | H:116.017 | 10 | 4/2/3/1 | 8960 | 539.02 /100.0% | FIXED_RR:10 |
| 5 | P0 | H:116.019 | 10 | 3/1/4/2 | 12800 | 586.28 /100.0% | FIXED_RR:10 |
| 5 | P1 | H:116.019 | 10 | 4/2/3/1 | 8960 | 568.65 /100.0% | FIXED_RR:10 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 1 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 2 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 2 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 3 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 3 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 4 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 4 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 5 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 5 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |

Final measured residency and learned states for this block: P0 L/M/H = 0.00/0.00/100.00%; P1 = 0.00/0.00/100.00%. Final P0 K=H, S=H, C=none, R=none; P1 K=H, S=H, C=none, R=none. Final controller consumed counters P0=0 windows/0 requests, P1=0 windows/0 requests.

### MEDIUM, repeat 1, JOINT-ROUTING-PER-ENDPOINT

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:200.268 | 13 | 7/0/5/1 | 13184 | 479.19 /100.0% | COLD_START_RR:2; LOWER_PRESSURE:9; EXPLORATION:2 |
| 1 | P1 | H:200.268 | 7 | 0/3/2/2 | 8576 | 475.80 /100.0% | COLD_START_RR:2; EXPLORATION:4; LOWER_PRESSURE:1 |
| 2 | P0 | L:201.850 | 7 | 4/3/0/0 | 896 | 205.66 /100.0% | EXPLORATION:7 |
| 2 | P1 | H:201.850 | 13 | 3/0/7/3 | 20864 | 572.88 /100.0% | ONLY_SAFE_PAIR:10; EXPLORATION:3 |
| 3 | P0 | H:200.554 | 7 | 3/0/2/2 | 8576 | 474.72 /100.0% | LOWER_PRESSURE:7 |
| 3 | P1 | H:200.554 | 13 | 4/3/5/1 | 13184 | 490.17 /100.0% | EXPLORATION:4; ENERGY_PREFERENCE:6; LOWER_PRESSURE:3 |
| 4 | P0 | H:200.875 | 12 | 6/3/3/0 | 7296 | 564.02 /100.0% | EXPLORATION:9; LOWER_PRESSURE:3 |
| 4 | P1 | H:200.875 | 8 | 1/0/4/3 | 14464 | 490.59 /100.0% | ENERGY_PREFERENCE:4; LOWER_PRESSURE:4 |
| 5 | P0 | M:200.252 | 9 | 6/3/0/0 | 1152 | 288.01 /100.0% | EXPLORATION:9 |
| 5 | P1 | H:200.252 | 11 | 1/0/7/3 | 20608 | 483.79 /100.0% | ENERGY_PREFERENCE:4; LOWER_PRESSURE:7 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | EXPLORE_DOWN:H→L; SEARCH_DOWN | 13 /13184 | K=H, S=H, C=L*, R=none | 2026-10-06T07:21:59.188Z |
| 1 | P1 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | HOLD:H→H; INSUFFICIENT_EVIDENCE | 7 /8576 | K=H, S=H, C=none, R=none | 2026-10-06T07:21:59.188Z |
| 2 | P0 | K=H, S=H, C=L*, R=none | P0:deny[1w,13r,candidate]; P1:deny[1w,7r] | RECOVER:L→H; EVIDENCE_TIMEOUT | 7 /896 | K=H, S=H, C=none, R=none | 2026-10-06T07:26:04.112Z |
| 2 | P1 | K=H, S=H, C=none, R=none | P0:deny[1w,13r,candidate]; P1:deny[1w,7r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 13 /20864 | K=H, S=H, C=none, R=none | 2026-10-06T07:26:04.112Z |
| 3 | P0 | K=H, S=H, C=none, R=none | P0:allow[2w,20r]; P1:deny[2w,20r] | HOLD:H→H; INSUFFICIENT_EVIDENCE | 7 /8576 | K=H, S=H, C=none, R=none | 2026-10-06T07:30:08.577Z |
| 3 | P1 | K=H, S=H, C=none, R=none | P0:allow[2w,20r]; P1:deny[2w,20r] | HOLD:H→H; CONTEXT_CHANGE | 13 /13184 | K=H, S=H, C=none, R=none | 2026-10-06T07:30:08.577Z |
| 4 | P0 | K=H, S=H, C=none, R=none | P0:allow[3w,27r]; P1:deny[3w,33r] | EXPLORE_DOWN:H→M; SEARCH_DOWN | 12 /7296 | K=H, S=H, C=M*, R=none | 2026-10-06T07:34:12.271Z |
| 4 | P1 | K=H, S=H, C=none, R=none | P0:allow[3w,27r]; P1:deny[3w,33r] | HOLD:H→H; CONTEXT_CHANGE | 8 /14464 | K=H, S=H, C=none, R=none | 2026-10-06T07:34:12.271Z |
| 5 | P0 | K=H, S=H, C=M*, R=none | P0:deny[4w,39r,candidate]; P1:deny[4w,41r] | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 9 /1152 | K=M, S=M/H, C=none, R=none | 2026-10-06T07:38:16.693Z |
| 5 | P1 | K=H, S=H, C=none, R=none | P0:deny[4w,39r,candidate]; P1:deny[4w,41r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE [final] | 11 /20608 | K=H, S=H, C=none, R=none | 2026-10-06T07:38:16.693Z |

Final measured residency and learned states for this block: P0 L/M/H = 20.11/19.95/59.94%; P1 = 0.00/0.00/100.00%. Final P0 K=M, S=M/H, C=none, R=none; P1 K=H, S=H, C=none, R=none. Final controller consumed counters P0=5 windows/48 requests, P1=5 windows/52 requests.

### MEDIUM, repeat 1, PER-ENDPOINT-FAST-RR

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:200.867 | 10 | 4/2/3/1 | 8960 | 555.27 /100.0% | FIXED_RR:10 |
| 1 | P1 | H:200.867 | 10 | 3/1/4/2 | 12800 | 560.60 /100.0% | FIXED_RR:10 |
| 2 | P0 | M:200.876 | 10 | 4/2/3/1 | 8960 | 557.87 /100.0% | FIXED_RR:10 |
| 2 | P1 | H:200.876 | 10 | 3/1/4/2 | 12800 | 572.73 /100.0% | FIXED_RR:10 |
| 3 | P0 | M:200.878 | 10 | 4/2/3/1 | 8960 | 551.72 /100.0% | FIXED_RR:10 |
| 3 | P1 | H:200.878 | 10 | 3/1/4/2 | 12800 | 574.00 /100.0% | FIXED_RR:10 |
| 4 | P0 | L:50.933; T→M:0.632; M:149.321 | 10 | 4/2/3/1 | 8960 | 587.04 /90.0% | FIXED_RR:10 |
| 4 | P1 | H:200.886 | 10 | 3/1/4/2 | 12800 | 563.71 /100.0% | FIXED_RR:10 |
| 5 | P0 | M:200.886 | 10 | 4/2/3/1 | 8960 | 554.29 /100.0% | FIXED_RR:10 |
| 5 | P1 | H:200.886 | 10 | 3/1/4/2 | 12800 | 572.86 /100.0% | FIXED_RR:10 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | EXPLORE_DOWN:H→M; SEARCH_DOWN | 10 /8960 | K=H, S=H, C=M*, R=none | 2026-10-06T07:01:41.035Z |
| 1 | P1 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 10 /12800 | K=H, S=H, C=none, R=none | 2026-10-06T07:01:41.035Z |
| 2 | P0 | K=H, S=H, C=M*, R=none | P0:deny[1w,10r,candidate]; P1:deny[1w,10r] | ACCEPT_SAFE:M→M; FINAL_EVIDENCE | 10 /8960 | K=M, S=M/H, C=none, R=none | 2026-10-06T07:05:46.590Z |
| 2 | P1 | K=H, S=H, C=none, R=none | P0:deny[1w,10r,candidate]; P1:deny[1w,10r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 10 /12800 | K=H, S=H, C=none, R=none | 2026-10-06T07:05:46.590Z |
| 3 | P0 | K=M, S=M/H, C=none, R=none | P0:allow[2w,20r]; P1:deny[2w,20r] | EXPLORE_DOWN:M→L; SEARCH_DOWN | 10 /8960 | K=M, S=M/H, C=L*, R=none | 2026-10-06T07:09:48.057Z |
| 3 | P1 | K=H, S=H, C=none, R=none | P0:allow[2w,20r]; P1:deny[2w,20r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 10 /12800 | K=H, S=H, C=none, R=none | 2026-10-06T07:09:48.057Z |
| 4 | P0 | K=M, S=M/H, C=L*, R=none | P0:allow[3w,30r]; P1:deny[3w,30r] | HOLD:—→—; MIXED_FREQUENCY_WINDOW | not consumed /— | K=M, S=M/H, C=none, R=L | 2026-10-06T07:13:52.739Z |
| 4 | P1 | K=H, S=H, C=none, R=none | P0:allow[3w,30r]; P1:deny[3w,30r] | HOLD:—→—; MIXED_FREQUENCY_WINDOW | not consumed /— | K=H, S=H, C=none, R=none | 2026-10-06T07:13:52.739Z |
| 5 | P0 | K=M, S=M/H, C=none, R=L | P0:allow[3w,30r]; P1:deny[3w,30r] | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 10 /8960 | K=M, S=M/H, C=none, R=L | 2026-10-06T07:17:54.387Z |
| 5 | P1 | K=H, S=H, C=none, R=none | P0:allow[3w,30r]; P1:deny[3w,30r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE [final] | 10 /12800 | K=H, S=H, C=none, R=none | 2026-10-06T07:17:54.387Z |

Final measured residency and learned states for this block: P0 L/M/H = 5.07/74.91/20.01%; P1 = 0.00/0.00/100.00%. Final P0 K=M, S=M/H, C=none, R=L; P1 K=H, S=H, C=none, R=none. Final controller consumed counters P0=4 windows/40 requests, P1=4 windows/40 requests.

### MEDIUM, repeat 1, SHARED-FAST-RR

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:200.866 | 10 | 4/2/3/1 | 8960 | 555.51 /100.0% | FIXED_RR:10 |
| 1 | P1 | H:200.866 | 10 | 3/1/4/2 | 12800 | 561.35 /100.0% | FIXED_RR:10 |
| 2 | P0 | M:200.876 | 10 | 4/2/3/1 | 8960 | 566.51 /100.0% | FIXED_RR:10 |
| 2 | P1 | M:200.876 | 10 | 3/1/4/2 | 12800 | 570.26 /100.0% | FIXED_RR:10 |
| 3 | P0 | L:50.990; T→M:0.632; M:149.250 | 10 | 4/2/3/1 | 8960 | 603.98 /90.0% | FIXED_RR:10 |
| 3 | P1 | L:51.921; T→M:0.891; M:148.061 | 10 | 3/1/4/2 | 12800 | 572.88 /100.0% | FIXED_RR:10 |
| 4 | P0 | M:200.870 | 10 | 4/2/3/1 | 8960 | 563.82 /100.0% | FIXED_RR:10 |
| 4 | P1 | M:200.870 | 10 | 3/1/4/2 | 12800 | 562.99 /100.0% | FIXED_RR:10 |
| 5 | P0 | M:200.873 | 10 | 4/2/3/1 | 8960 | 566.09 /100.0% | FIXED_RR:10 |
| 5 | P1 | M:200.873 | 10 | 3/1/4/2 | 12800 | 561.50 /100.0% | FIXED_RR:10 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | shared pool | EXPLORE_DOWN:H→M; SEARCH_DOWN | 20 /21760 | K=H, S=H, C=M*, R=none | 2026-10-06T06:41:17.615Z |
| 1 | P1 | K=H, S=H, C=none, R=none | shared pool | EXPLORE_DOWN:H→M; SEARCH_DOWN | 20 /21760 | K=H, S=H, C=M*, R=none | 2026-10-06T06:41:17.615Z |
| 2 | P0 | K=H, S=H, C=M*, R=none | shared pool | EXPLORE_DOWN:M→L; SEARCH_DOWN | 20 /21760 | K=M, S=M/H, C=L*, R=none | 2026-10-06T06:45:26.007Z |
| 2 | P1 | K=H, S=H, C=M*, R=none | shared pool | EXPLORE_DOWN:M→L; SEARCH_DOWN | 20 /21760 | K=M, S=M/H, C=L*, R=none | 2026-10-06T06:45:26.007Z |
| 3 | P0 | K=M, S=M/H, C=L*, R=none | shared pool | HOLD:—→M; MIXED_CLOCK_EVIDENCE | not consumed /— | K=M, S=M/H, C=none, R=L | 2026-10-06T06:49:30.565Z |
| 3 | P1 | K=M, S=M/H, C=L*, R=none | shared pool | HOLD:—→M; MIXED_CLOCK_EVIDENCE | not consumed /— | K=M, S=M/H, C=none, R=L | 2026-10-06T06:49:30.565Z |
| 4 | P0 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; SEARCH_CONVERGED | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T06:53:33.189Z |
| 4 | P1 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; SEARCH_CONVERGED | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T06:53:33.189Z |
| 5 | P0 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T06:57:35.748Z |
| 5 | P1 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T06:57:35.748Z |

Final measured residency and learned states for this block: P0 L/M/H = 5.08/74.91/20.01%; P1 = 5.17/74.81/20.02%. Final P0 K=M, S=M/H, C=none, R=L; P1 K=M, S=M/H, C=none, R=L. Final controller consumed counters P0=5 windows/100 requests, P1=5 windows/100 requests (same pooled counters, not separate endpoint evidence).

### MEDIUM, repeat 1, STATIC-HIGH-RR

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:200.868 | 10 | 4/2/3/1 | 8960 | 749.22 /80.0% | FIXED_RR:10 |
| 1 | P1 | H:200.868 | 10 | 3/1/4/2 | 12800 | 565.27 /100.0% | FIXED_RR:10 |
| 2 | P0 | H:200.864 | 10 | 4/2/3/1 | 8960 | 550.55 /100.0% | FIXED_RR:10 |
| 2 | P1 | H:200.864 | 10 | 3/1/4/2 | 12800 | 565.27 /100.0% | FIXED_RR:10 |
| 3 | P0 | H:200.876 | 10 | 4/2/3/1 | 8960 | 562.95 /100.0% | FIXED_RR:10 |
| 3 | P1 | H:200.876 | 10 | 3/1/4/2 | 12800 | 572.42 /100.0% | FIXED_RR:10 |
| 4 | P0 | H:200.869 | 10 | 4/2/3/1 | 8960 | 557.78 /100.0% | FIXED_RR:10 |
| 4 | P1 | H:200.869 | 10 | 3/1/4/2 | 12800 | 560.54 /100.0% | FIXED_RR:10 |
| 5 | P0 | H:200.865 | 10 | 4/2/3/1 | 8960 | 559.51 /100.0% | FIXED_RR:10 |
| 5 | P1 | H:200.865 | 10 | 3/1/4/2 | 12800 | 543.69 /100.0% | FIXED_RR:10 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 1 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 2 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 2 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 3 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 3 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 4 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 4 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 5 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 5 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |

Final measured residency and learned states for this block: P0 L/M/H = 0.00/0.00/100.00%; P1 = 0.00/0.00/100.00%. Final P0 K=H, S=H, C=none, R=none; P1 K=H, S=H, C=none, R=none. Final controller consumed counters P0=0 windows/0 requests, P1=0 windows/0 requests.

### HIGH-SUSTAINABLE, repeat 2, JOINT-ROUTING-PER-ENDPOINT

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:115.993 | 9 | 1/3/4/1 | 10752 | 569.36 /100.0% | COLD_START_RR:2; LOWER_PRESSURE:5; EXPLORATION:2 |
| 1 | P1 | H:115.993 | 11 | 6/0/3/2 | 11008 | 566.82 /100.0% | COLD_START_RR:2; EXPLORATION:4; LOWER_PRESSURE:5 |
| 2 | P0 | M:116.229 | 9 | 5/3/1/0 | 3072 | 438.89 /100.0% | EXPLORATION:9 |
| 2 | P1 | H:116.229 | 11 | 2/0/6/3 | 18688 | 584.51 /100.0% | ONLY_SAFE_PAIR:10; EXPLORATION:1 |
| 3 | P0 | M:115.756 | 11 | 2/0/6/3 | 18688 | 596.20 /100.0% | LOWER_PRESSURE:5; ENERGY_PREFERENCE:6 |
| 3 | P1 | H:115.756 | 9 | 5/3/1/0 | 3072 | 460.00 /100.0% | EXPLORATION:3; LOWER_PRESSURE:6 |
| 4 | P0 | H:116.463 | 9 | 4/2/2/1 | 6912 | 557.44 /100.0% | EXPLORATION:2; LOWER_PRESSURE:7 |
| 4 | P1 | H:116.463 | 11 | 3/1/5/2 | 14848 | 588.36 /100.0% | LOWER_PRESSURE:6; ENERGY_PREFERENCE:3; EXPLORATION:2 |
| 5 | P0 | H:116.024 | 10 | 6/3/1/0 | 3200 | 432.85 /100.0% | EXPLORATION:10 |
| 5 | P1 | H:116.024 | 10 | 1/0/6/3 | 18560 | 585.18 /100.0% | LOWER_PRESSURE:7; ENERGY_PREFERENCE:3 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | EXPLORE_DOWN:H→M; SEARCH_DOWN | 9 /10752 | K=H, S=H, C=M*, R=none | 2026-10-06T11:55:24.039Z |
| 1 | P1 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 11 /11008 | K=H, S=H, C=none, R=none | 2026-10-06T11:55:24.039Z |
| 2 | P0 | K=H, S=H, C=M*, R=none | P0:deny[1w,9r,candidate]; P1:deny[1w,11r] | ACCEPT_SAFE:M→M; FINAL_EVIDENCE | 9 /3072 | K=M, S=M/H, C=none, R=none | 2026-10-06T11:58:04.571Z |
| 2 | P1 | K=H, S=H, C=none, R=none | P0:deny[1w,9r,candidate]; P1:deny[1w,11r] | HOLD:H→H; CONTEXT_CHANGE | 11 /18688 | K=H, S=H, C=none, R=none | 2026-10-06T11:58:04.571Z |
| 3 | P0 | K=M, S=M/H, C=none, R=none | P0:allow[2w,18r]; P1:deny[2w,22r] | RECOVER:M→H; CONTEXT_CHANGE | 11 /18688 | K=H, S=H, C=none, R=none | 2026-10-06T12:00:41.523Z |
| 3 | P1 | K=H, S=H, C=none, R=none | P0:allow[2w,18r]; P1:deny[2w,22r] | HOLD:H→H; CONTEXT_CHANGE | 9 /3072 | K=H, S=H, C=none, R=none | 2026-10-06T12:00:41.523Z |
| 4 | P0 | K=H, S=H, C=none, R=none | P0:allow[3w,29r]; P1:deny[3w,31r] | HOLD:H→H; CONTEXT_CHANGE | 9 /6912 | K=H, S=H, C=none, R=none | 2026-10-06T12:03:20.399Z |
| 4 | P1 | K=H, S=H, C=none, R=none | P0:allow[3w,29r]; P1:deny[3w,31r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 11 /14848 | K=H, S=H, C=none, R=none | 2026-10-06T12:03:20.399Z |
| 5 | P0 | K=H, S=H, C=none, R=none | P0:allow[4w,38r]; P1:deny[4w,42r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE [final] | 10 /3200 | K=H, S=H, C=none, R=none | 2026-10-06T12:06:00.126Z |
| 5 | P1 | K=H, S=H, C=none, R=none | P0:allow[4w,38r]; P1:deny[4w,42r] | HOLD:H→H; CONTEXT_CHANGE [final] | 10 /18560 | K=H, S=H, C=none, R=none | 2026-10-06T12:06:00.126Z |

Final measured residency and learned states for this block: P0 L/M/H = 0.00/39.97/60.03%; P1 = 0.00/0.00/100.00%. Final P0 K=H, S=H, C=none, R=none; P1 K=H, S=H, C=none, R=none. Final controller consumed counters P0=5 windows/48 requests, P1=5 windows/52 requests.

### HIGH-SUSTAINABLE, repeat 2, PER-ENDPOINT-FAST-RR

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:116.053 | 10 | 4/1/4/1 | 10880 | 563.83 /100.0% | FIXED_RR:10 |
| 1 | P1 | H:116.053 | 10 | 3/2/3/2 | 10880 | 539.72 /100.0% | FIXED_RR:10 |
| 2 | P0 | M:116.035 | 10 | 4/1/4/1 | 10880 | 579.61 /100.0% | FIXED_RR:10 |
| 2 | P1 | H:116.035 | 10 | 3/2/3/2 | 10880 | 562.95 /100.0% | FIXED_RR:10 |
| 3 | P0 | M:116.035 | 10 | 4/1/4/1 | 10880 | 590.98 /100.0% | FIXED_RR:10 |
| 3 | P1 | H:116.035 | 10 | 3/2/3/2 | 10880 | 557.93 /100.0% | FIXED_RR:10 |
| 4 | P0 | L:25.959; T→M:0.624; M:89.448 | 10 | 4/1/4/1 | 10880 | 621.92 /90.0% | FIXED_RR:10 |
| 4 | P1 | H:116.031 | 10 | 3/2/3/2 | 10880 | 554.56 /100.0% | FIXED_RR:10 |
| 5 | P0 | M:116.029 | 10 | 4/1/4/1 | 10880 | 588.86 /100.0% | FIXED_RR:10 |
| 5 | P1 | H:116.029 | 10 | 3/2/3/2 | 10880 | 554.52 /100.0% | FIXED_RR:10 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | EXPLORE_DOWN:H→M; SEARCH_DOWN | 10 /10880 | K=H, S=H, C=M*, R=none | 2026-10-06T11:42:14.621Z |
| 1 | P1 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 10 /10880 | K=H, S=H, C=none, R=none | 2026-10-06T11:42:14.621Z |
| 2 | P0 | K=H, S=H, C=M*, R=none | P0:deny[1w,10r,candidate]; P1:deny[1w,10r] | ACCEPT_SAFE:M→M; FINAL_EVIDENCE | 10 /10880 | K=M, S=M/H, C=none, R=none | 2026-10-06T11:44:53.321Z |
| 2 | P1 | K=H, S=H, C=none, R=none | P0:deny[1w,10r,candidate]; P1:deny[1w,10r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 10 /10880 | K=H, S=H, C=none, R=none | 2026-10-06T11:44:53.321Z |
| 3 | P0 | K=M, S=M/H, C=none, R=none | P0:allow[2w,20r]; P1:deny[2w,20r] | EXPLORE_DOWN:M→L; SEARCH_DOWN | 10 /10880 | K=M, S=M/H, C=L*, R=none | 2026-10-06T11:47:29.672Z |
| 3 | P1 | K=H, S=H, C=none, R=none | P0:allow[2w,20r]; P1:deny[2w,20r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 10 /10880 | K=H, S=H, C=none, R=none | 2026-10-06T11:47:29.672Z |
| 4 | P0 | K=M, S=M/H, C=L*, R=none | P0:allow[3w,30r]; P1:deny[3w,30r] | HOLD:—→—; MIXED_FREQUENCY_WINDOW | not consumed /— | K=M, S=M/H, C=none, R=L | 2026-10-06T11:50:07.569Z |
| 4 | P1 | K=H, S=H, C=none, R=none | P0:allow[3w,30r]; P1:deny[3w,30r] | HOLD:—→—; MIXED_FREQUENCY_WINDOW | not consumed /— | K=H, S=H, C=none, R=none | 2026-10-06T11:50:07.569Z |
| 5 | P0 | K=M, S=M/H, C=none, R=L | P0:allow[3w,30r]; P1:deny[3w,30r] | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 10 /10880 | K=M, S=M/H, C=none, R=L | 2026-10-06T11:52:46.219Z |
| 5 | P1 | K=H, S=H, C=none, R=none | P0:allow[3w,30r]; P1:deny[3w,30r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE [final] | 10 /10880 | K=H, S=H, C=none, R=none | 2026-10-06T11:52:46.219Z |

Final measured residency and learned states for this block: P0 L/M/H = 4.48/75.50/20.02%; P1 = 0.00/0.00/100.00%. Final P0 K=M, S=M/H, C=none, R=L; P1 K=H, S=H, C=none, R=none. Final controller consumed counters P0=4 windows/40 requests, P1=4 windows/40 requests.

### HIGH-SUSTAINABLE, repeat 2, SHARED-FAST-RR

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:116.046 | 10 | 4/1/4/1 | 10880 | 566.53 /100.0% | FIXED_RR:10 |
| 1 | P1 | H:116.046 | 10 | 3/2/3/2 | 10880 | 535.98 /100.0% | FIXED_RR:10 |
| 2 | P0 | M:116.052 | 10 | 4/1/4/1 | 10880 | 572.46 /100.0% | FIXED_RR:10 |
| 2 | P1 | M:116.052 | 10 | 3/2/3/2 | 10880 | 595.98 /100.0% | FIXED_RR:10 |
| 3 | P0 | M:116.049 | 10 | 4/1/4/1 | 10880 | 565.46 /100.0% | FIXED_RR:10 |
| 3 | P1 | M:116.049 | 10 | 3/2/3/2 | 10880 | 569.24 /100.0% | FIXED_RR:10 |
| 4 | P0 | M:116.047 | 10 | 4/1/4/1 | 10880 | 564.54 /100.0% | FIXED_RR:10 |
| 4 | P1 | M:116.047 | 10 | 3/2/3/2 | 10880 | 568.50 /100.0% | FIXED_RR:10 |
| 5 | P0 | M:116.045 | 10 | 4/1/4/1 | 10880 | 563.93 /100.0% | FIXED_RR:10 |
| 5 | P1 | M:116.045 | 10 | 3/2/3/2 | 10880 | 566.58 /100.0% | FIXED_RR:10 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | shared pool | EXPLORE_DOWN:H→M; SEARCH_DOWN | 20 /21760 | K=H, S=H, C=M*, R=none | 2026-10-06T11:29:01.161Z |
| 1 | P1 | K=H, S=H, C=none, R=none | shared pool | EXPLORE_DOWN:H→M; SEARCH_DOWN | 20 /21760 | K=H, S=H, C=M*, R=none | 2026-10-06T11:29:01.161Z |
| 2 | P0 | K=H, S=H, C=M*, R=none | shared pool | ACCEPT_SAFE:M→M; ENOUGH_EVIDENCE | 20 /21760 | K=M, S=M/H, C=none, R=none | 2026-10-06T11:31:43.767Z |
| 2 | P1 | K=H, S=H, C=M*, R=none | shared pool | ACCEPT_SAFE:M→M; ENOUGH_EVIDENCE | 20 /21760 | K=M, S=M/H, C=none, R=none | 2026-10-06T11:31:43.767Z |
| 3 | P0 | K=M, S=M/H, C=none, R=none | shared pool | ACCEPT_SAFE:M→M; SEARCH_CONVERGED | 20 /21760 | K=M, S=M/H, C=none, R=none | 2026-10-06T11:34:21.078Z |
| 3 | P1 | K=M, S=M/H, C=none, R=none | shared pool | ACCEPT_SAFE:M→M; SEARCH_CONVERGED | 20 /21760 | K=M, S=M/H, C=none, R=none | 2026-10-06T11:34:21.078Z |
| 4 | P0 | K=M, S=M/H, C=none, R=none | shared pool | ACCEPT_SAFE:M→M; SEARCH_CONVERGED | 20 /21760 | K=M, S=M/H, C=none, R=none | 2026-10-06T11:36:57.750Z |
| 4 | P1 | K=M, S=M/H, C=none, R=none | shared pool | ACCEPT_SAFE:M→M; SEARCH_CONVERGED | 20 /21760 | K=M, S=M/H, C=none, R=none | 2026-10-06T11:36:57.750Z |
| 5 | P0 | K=M, S=M/H, C=none, R=none | shared pool | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 20 /21760 | K=M, S=M/H, C=none, R=none | 2026-10-06T11:39:34.094Z |
| 5 | P1 | K=M, S=M/H, C=none, R=none | shared pool | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 20 /21760 | K=M, S=M/H, C=none, R=none | 2026-10-06T11:39:34.094Z |

Final measured residency and learned states for this block: P0 L/M/H = 0.00/80.00/20.00%; P1 = 0.00/80.00/20.00%. Final P0 K=M, S=M/H, C=none, R=none; P1 K=M, S=M/H, C=none, R=none. Final controller consumed counters P0=5 windows/100 requests, P1=5 windows/100 requests (same pooled counters, not separate endpoint evidence).

### HIGH-SUSTAINABLE, repeat 2, STATIC-HIGH-RR

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:116.042 | 10 | 4/1/4/1 | 10880 | 558.75 /100.0% | FIXED_RR:10 |
| 1 | P1 | H:116.042 | 10 | 3/2/3/2 | 10880 | 532.07 /100.0% | FIXED_RR:10 |
| 2 | P0 | H:116.035 | 10 | 4/1/4/1 | 10880 | 558.99 /100.0% | FIXED_RR:10 |
| 2 | P1 | H:116.035 | 10 | 3/2/3/2 | 10880 | 555.62 /100.0% | FIXED_RR:10 |
| 3 | P0 | H:116.036 | 10 | 4/1/4/1 | 10880 | 565.44 /100.0% | FIXED_RR:10 |
| 3 | P1 | H:116.036 | 10 | 3/2/3/2 | 10880 | 559.35 /100.0% | FIXED_RR:10 |
| 4 | P0 | H:116.036 | 10 | 4/1/4/1 | 10880 | 563.03 /100.0% | FIXED_RR:10 |
| 4 | P1 | H:116.036 | 10 | 3/2/3/2 | 10880 | 583.12 /100.0% | FIXED_RR:10 |
| 5 | P0 | H:116.044 | 10 | 4/1/4/1 | 10880 | 566.47 /100.0% | FIXED_RR:10 |
| 5 | P1 | H:116.044 | 10 | 3/2/3/2 | 10880 | 534.66 /100.0% | FIXED_RR:10 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 1 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 2 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 2 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 3 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 3 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 4 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 4 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 5 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 5 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |

Final measured residency and learned states for this block: P0 L/M/H = 0.00/0.00/100.00%; P1 = 0.00/0.00/100.00%. Final P0 K=H, S=H, C=none, R=none; P1 K=H, S=H, C=none, R=none. Final controller consumed counters P0=0 windows/0 requests, P1=0 windows/0 requests.

### MEDIUM, repeat 2, JOINT-ROUTING-PER-ENDPOINT

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:200.258 | 8 | 1/2/3/2 | 10624 | 474.84 /100.0% | COLD_START_RR:2; LOWER_PRESSURE:3; EXPLORATION:3 |
| 1 | P1 | H:200.258 | 12 | 6/1/4/1 | 11136 | 471.10 /100.0% | COLD_START_RR:2; EXPLORATION:4; LOWER_PRESSURE:6 |
| 2 | P0 | L:201.853 | 7 | 4/3/0/0 | 896 | 207.04 /100.0% | EXPLORATION:7 |
| 2 | P1 | H:201.853 | 13 | 3/0/7/3 | 20864 | 572.07 /100.0% | ONLY_SAFE_PAIR:10; EXPLORATION:3 |
| 3 | P0 | H:200.553 | 9 | 5/0/2/2 | 8832 | 474.40 /100.0% | EXPLORATION:2; LOWER_PRESSURE:7 |
| 3 | P1 | H:200.553 | 11 | 2/3/5/1 | 12928 | 479.03 /100.0% | ENERGY_PREFERENCE:6; LOWER_PRESSURE:3; EXPLORATION:2 |
| 4 | P0 | L:200.250 | 9 | 6/3/0/0 | 1152 | 282.17 /100.0% | EXPLORATION:9 |
| 4 | P1 | H:200.250 | 11 | 1/0/7/3 | 20608 | 480.55 /100.0% | ENERGY_PREFERENCE:4; LOWER_PRESSURE:7 |
| 5 | P0 | L:200.937 | 9 | 2/0/4/3 | 14592 | 548.35 /100.0% | EXPLORATION:1; ENERGY_PREFERENCE:4; LOWER_PRESSURE:4 |
| 5 | P1 | H:200.937 | 11 | 5/3/3/0 | 7168 | 571.02 /100.0% | EXPLORATION:8; LOWER_PRESSURE:3 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | EXPLORE_DOWN:H→L; SEARCH_DOWN | 8 /10624 | K=H, S=H, C=L*, R=none | 2026-10-06T08:22:55.686Z |
| 1 | P1 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 12 /11136 | K=H, S=H, C=none, R=none | 2026-10-06T08:22:55.686Z |
| 2 | P0 | K=H, S=H, C=L*, R=none | P0:deny[1w,8r,candidate]; P1:deny[1w,12r] | RECOVER:L→H; EVIDENCE_TIMEOUT | 7 /896 | K=H, S=H, C=none, R=none | 2026-10-06T08:27:02.437Z |
| 2 | P1 | K=H, S=H, C=none, R=none | P0:deny[1w,8r,candidate]; P1:deny[1w,12r] | HOLD:H→H; CONTEXT_CHANGE | 13 /20864 | K=H, S=H, C=none, R=none | 2026-10-06T08:27:02.437Z |
| 3 | P0 | K=H, S=H, C=none, R=none | P0:allow[2w,15r]; P1:deny[2w,25r] | EXPLORE_DOWN:H→L; SEARCH_DOWN | 9 /8832 | K=H, S=H, C=L*, R=none | 2026-10-06T08:31:08.172Z |
| 3 | P1 | K=H, S=H, C=none, R=none | P0:allow[2w,15r]; P1:deny[2w,25r] | HOLD:H→H; CONTEXT_CHANGE | 11 /12928 | K=H, S=H, C=none, R=none | 2026-10-06T08:31:08.172Z |
| 4 | P0 | K=H, S=H, C=L*, R=none | P0:deny[3w,24r,candidate]; P1:deny[3w,36r] | ACCEPT_SAFE:L→L; FINAL_EVIDENCE | 9 /1152 | K=L, S=L/H, C=none, R=none | 2026-10-06T08:35:10.827Z |
| 4 | P1 | K=H, S=H, C=none, R=none | P0:deny[3w,24r,candidate]; P1:deny[3w,36r] | HOLD:H→H; CONTEXT_CHANGE | 11 /20608 | K=H, S=H, C=none, R=none | 2026-10-06T08:35:10.827Z |
| 5 | P0 | K=L, S=L/H, C=none, R=none | P0:allow[4w,33r]; P1:deny[4w,47r] | RECOVER:L→H; CONTEXT_CHANGE [final] | 9 /14592 | K=H, S=H, C=none, R=none | 2026-10-06T08:39:12.879Z |
| 5 | P1 | K=H, S=H, C=none, R=none | P0:allow[4w,33r]; P1:deny[4w,47r] | HOLD:H→H; CONTEXT_CHANGE [final] | 11 /7168 | K=H, S=H, C=none, R=none | 2026-10-06T08:39:12.879Z |

Final measured residency and learned states for this block: P0 L/M/H = 60.07/0.00/39.93%; P1 = 0.00/0.00/100.00%. Final P0 K=H, S=H, C=none, R=none; P1 K=H, S=H, C=none, R=none. Final controller consumed counters P0=5 windows/42 requests, P1=5 windows/58 requests.

### MEDIUM, repeat 2, PER-ENDPOINT-FAST-RR

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:200.882 | 10 | 4/1/4/1 | 10880 | 568.51 /100.0% | FIXED_RR:10 |
| 1 | P1 | H:200.882 | 10 | 3/2/3/2 | 10880 | 528.10 /100.0% | FIXED_RR:10 |
| 2 | P0 | M:200.889 | 10 | 4/1/4/1 | 10880 | 571.86 /100.0% | FIXED_RR:10 |
| 2 | P1 | H:200.889 | 10 | 3/2/3/2 | 10880 | 530.06 /100.0% | FIXED_RR:10 |
| 3 | P0 | M:200.893 | 10 | 4/1/4/1 | 10880 | 568.02 /100.0% | FIXED_RR:10 |
| 3 | P1 | H:200.893 | 10 | 3/2/3/2 | 10880 | 531.55 /100.0% | FIXED_RR:10 |
| 4 | P0 | L:50.977; T→M:0.603; M:149.309 | 10 | 4/1/4/1 | 10880 | 593.18 /90.0% | FIXED_RR:10 |
| 4 | P1 | H:200.888 | 10 | 3/2/3/2 | 10880 | 528.74 /100.0% | FIXED_RR:10 |
| 5 | P0 | M:200.889 | 10 | 4/1/4/1 | 10880 | 564.77 /100.0% | FIXED_RR:10 |
| 5 | P1 | H:200.889 | 10 | 3/2/3/2 | 10880 | 528.79 /100.0% | FIXED_RR:10 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | EXPLORE_DOWN:H→M; SEARCH_DOWN | 10 /10880 | K=H, S=H, C=M*, R=none | 2026-10-06T08:02:41.776Z |
| 1 | P1 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 10 /10880 | K=H, S=H, C=none, R=none | 2026-10-06T08:02:41.776Z |
| 2 | P0 | K=H, S=H, C=M*, R=none | P0:deny[1w,10r,candidate]; P1:deny[1w,10r] | ACCEPT_SAFE:M→M; FINAL_EVIDENCE | 10 /10880 | K=M, S=M/H, C=none, R=none | 2026-10-06T08:06:46.694Z |
| 2 | P1 | K=H, S=H, C=none, R=none | P0:deny[1w,10r,candidate]; P1:deny[1w,10r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 10 /10880 | K=H, S=H, C=none, R=none | 2026-10-06T08:06:46.694Z |
| 3 | P0 | K=M, S=M/H, C=none, R=none | P0:allow[2w,20r]; P1:deny[2w,20r] | EXPLORE_DOWN:M→L; SEARCH_DOWN | 10 /10880 | K=M, S=M/H, C=L*, R=none | 2026-10-06T08:10:48.386Z |
| 3 | P1 | K=H, S=H, C=none, R=none | P0:allow[2w,20r]; P1:deny[2w,20r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 10 /10880 | K=H, S=H, C=none, R=none | 2026-10-06T08:10:48.386Z |
| 4 | P0 | K=M, S=M/H, C=L*, R=none | P0:allow[3w,30r]; P1:deny[3w,30r] | HOLD:—→—; MIXED_FREQUENCY_WINDOW | not consumed /— | K=M, S=M/H, C=none, R=L | 2026-10-06T08:14:52.095Z |
| 4 | P1 | K=H, S=H, C=none, R=none | P0:allow[3w,30r]; P1:deny[3w,30r] | HOLD:—→—; MIXED_FREQUENCY_WINDOW | not consumed /— | K=H, S=H, C=none, R=none | 2026-10-06T08:14:52.095Z |
| 5 | P0 | K=M, S=M/H, C=none, R=L | P0:allow[3w,30r]; P1:deny[3w,30r] | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 10 /10880 | K=M, S=M/H, C=none, R=L | 2026-10-06T08:18:53.199Z |
| 5 | P1 | K=H, S=H, C=none, R=none | P0:allow[3w,30r]; P1:deny[3w,30r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE [final] | 10 /10880 | K=H, S=H, C=none, R=none | 2026-10-06T08:18:53.199Z |

Final measured residency and learned states for this block: P0 L/M/H = 5.08/74.91/20.01%; P1 = 0.00/0.00/100.00%. Final P0 K=M, S=M/H, C=none, R=L; P1 K=H, S=H, C=none, R=none. Final controller consumed counters P0=4 windows/40 requests, P1=4 windows/40 requests.

### MEDIUM, repeat 2, SHARED-FAST-RR

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:200.878 | 10 | 4/1/4/1 | 10880 | 567.40 /100.0% | FIXED_RR:10 |
| 1 | P1 | H:200.878 | 10 | 3/2/3/2 | 10880 | 522.39 /100.0% | FIXED_RR:10 |
| 2 | P0 | M:200.892 | 10 | 4/1/4/1 | 10880 | 574.19 /100.0% | FIXED_RR:10 |
| 2 | P1 | M:200.892 | 10 | 3/2/3/2 | 10880 | 537.87 /100.0% | FIXED_RR:10 |
| 3 | P0 | L:50.950; T→M:0.622; M:149.318 | 10 | 4/1/4/1 | 10880 | 613.06 /90.0% | FIXED_RR:10 |
| 3 | P1 | L:51.875; T→M:1.472; M:147.542 | 10 | 3/2/3/2 | 10880 | 562.64 /100.0% | FIXED_RR:10 |
| 4 | P0 | M:200.889 | 10 | 4/1/4/1 | 10880 | 575.32 /100.0% | FIXED_RR:10 |
| 4 | P1 | M:200.889 | 10 | 3/2/3/2 | 10880 | 534.97 /100.0% | FIXED_RR:10 |
| 5 | P0 | M:200.887 | 10 | 4/1/4/1 | 10880 | 579.95 /100.0% | FIXED_RR:10 |
| 5 | P1 | M:200.887 | 10 | 3/2/3/2 | 10880 | 535.51 /100.0% | FIXED_RR:10 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | shared pool | EXPLORE_DOWN:H→M; SEARCH_DOWN | 20 /21760 | K=H, S=H, C=M*, R=none | 2026-10-06T07:42:20.893Z |
| 1 | P1 | K=H, S=H, C=none, R=none | shared pool | EXPLORE_DOWN:H→M; SEARCH_DOWN | 20 /21760 | K=H, S=H, C=M*, R=none | 2026-10-06T07:42:20.893Z |
| 2 | P0 | K=H, S=H, C=M*, R=none | shared pool | EXPLORE_DOWN:M→L; SEARCH_DOWN | 20 /21760 | K=M, S=M/H, C=L*, R=none | 2026-10-06T07:46:29.273Z |
| 2 | P1 | K=H, S=H, C=M*, R=none | shared pool | EXPLORE_DOWN:M→L; SEARCH_DOWN | 20 /21760 | K=M, S=M/H, C=L*, R=none | 2026-10-06T07:46:29.273Z |
| 3 | P0 | K=M, S=M/H, C=L*, R=none | shared pool | HOLD:—→M; MIXED_CLOCK_EVIDENCE | not consumed /— | K=M, S=M/H, C=none, R=L | 2026-10-06T07:50:33.769Z |
| 3 | P1 | K=M, S=M/H, C=L*, R=none | shared pool | HOLD:—→M; MIXED_CLOCK_EVIDENCE | not consumed /— | K=M, S=M/H, C=none, R=L | 2026-10-06T07:50:33.769Z |
| 4 | P0 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; SEARCH_CONVERGED | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T07:54:35.140Z |
| 4 | P1 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; SEARCH_CONVERGED | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T07:54:35.140Z |
| 5 | P0 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T07:58:36.999Z |
| 5 | P1 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T07:58:36.999Z |

Final measured residency and learned states for this block: P0 L/M/H = 5.08/74.91/20.01%; P1 = 5.17/74.80/20.03%. Final P0 K=M, S=M/H, C=none, R=L; P1 K=M, S=M/H, C=none, R=L. Final controller consumed counters P0=5 windows/100 requests, P1=5 windows/100 requests (same pooled counters, not separate endpoint evidence).

### MEDIUM, repeat 2, STATIC-HIGH-RR

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:200.865 | 10 | 4/2/3/1 | 8960 | 557.13 /100.0% | FIXED_RR:10 |
| 1 | P1 | H:200.865 | 10 | 3/1/4/2 | 12800 | 565.17 /100.0% | FIXED_RR:10 |
| 2 | P0 | H:200.864 | 10 | 4/2/3/1 | 8960 | 561.72 /100.0% | FIXED_RR:10 |
| 2 | P1 | H:200.864 | 10 | 3/1/4/2 | 12800 | 563.21 /100.0% | FIXED_RR:10 |
| 3 | P0 | H:200.864 | 10 | 4/2/3/1 | 8960 | 558.15 /100.0% | FIXED_RR:10 |
| 3 | P1 | H:200.864 | 10 | 3/1/4/2 | 12800 | 556.10 /100.0% | FIXED_RR:10 |
| 4 | P0 | H:200.867 | 10 | 4/2/3/1 | 8960 | 557.61 /100.0% | FIXED_RR:10 |
| 4 | P1 | H:200.867 | 10 | 3/1/4/2 | 12800 | 539.51 /100.0% | FIXED_RR:10 |
| 5 | P0 | H:200.865 | 10 | 4/2/3/1 | 8960 | 559.44 /100.0% | FIXED_RR:10 |
| 5 | P1 | H:200.865 | 10 | 3/1/4/2 | 12800 | 565.21 /100.0% | FIXED_RR:10 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 1 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 2 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 2 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 3 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 3 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 4 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 4 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 5 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 5 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |

Final measured residency and learned states for this block: P0 L/M/H = 0.00/0.00/100.00%; P1 = 0.00/0.00/100.00%. Final P0 K=H, S=H, C=none, R=none; P1 K=H, S=H, C=none, R=none. Final controller consumed counters P0=0 windows/0 requests, P1=0 windows/0 requests.

### HIGH-SUSTAINABLE, repeat 3, JOINT-ROUTING-PER-ENDPOINT

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:116.061 | 13 | 6/1/6/0 | 13184 | 537.75 /100.0% | COLD_START_RR:2; LOWER_PRESSURE:11 |
| 1 | P1 | H:116.061 | 7 | 1/2/1/3 | 8576 | 555.71 /100.0% | COLD_START_RR:2; LOWER_PRESSURE:1; EXPLORATION:4 |
| 2 | P0 | L:50.959; T→H:0.647; H:64.447 | 10 | 6/3/1/0 | 3200 | 492.77 /90.0% | EXPLORATION:10 |
| 2 | P1 | H:116.052 | 10 | 1/0/6/3 | 18560 | 570.66 /100.0% | ONLY_SAFE_PAIR:5; LOWER_PRESSURE:3; ENERGY_PREFERENCE:2 |
| 3 | P0 | H:116.477 | 9 | 0/1/6/2 | 16512 | 577.32 /100.0% | LOWER_PRESSURE:9 |
| 3 | P1 | H:116.477 | 11 | 7/2/1/1 | 5248 | 579.93 /100.0% | EXPLORATION:5; LOWER_PRESSURE:6 |
| 4 | P0 | H:116.245 | 11 | 2/0/6/3 | 18688 | 566.92 /100.0% | ONLY_SAFE_PAIR:10; EXPLORATION:1 |
| 4 | P1 | M:116.245 | 9 | 5/3/1/0 | 3072 | 445.02 /100.0% | EXPLORATION:9 |
| 5 | P0 | H:116.468 | 9 | 4/2/2/1 | 6912 | 572.74 /100.0% | EXPLORATION:2; LOWER_PRESSURE:7 |
| 5 | P1 | M:116.468 | 11 | 3/1/5/2 | 14848 | 577.80 /100.0% | LOWER_PRESSURE:6; ENERGY_PREFERENCE:3; EXPLORATION:2 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | EXPLORE_DOWN:H→L; SEARCH_DOWN | 13 /13184 | K=H, S=H, C=L*, R=none | 2026-10-06T12:35:04.822Z |
| 1 | P1 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | HOLD:H→H; INSUFFICIENT_EVIDENCE | 7 /8576 | K=H, S=H, C=none, R=none | 2026-10-06T12:35:04.822Z |
| 2 | P0 | K=H, S=H, C=L*, R=none | P0:deny[1w,13r]; P1:allow[1w,7r] | HOLD:—→—; MIXED_FREQUENCY_WINDOW | not consumed /— | K=H, S=H, C=none, R=L | 2026-10-06T12:37:44.933Z |
| 2 | P1 | K=H, S=H, C=none, R=none | P0:deny[1w,13r]; P1:allow[1w,7r] | HOLD:—→—; MIXED_FREQUENCY_WINDOW | not consumed /— | K=H, S=H, C=none, R=none | 2026-10-06T12:37:44.933Z |
| 3 | P0 | K=H, S=H, C=none, R=L | P0:deny[1w,13r]; P1:allow[1w,7r] | HOLD:H→H; CONTEXT_CHANGE | 9 /16512 | K=H, S=H, C=none, R=none | 2026-10-06T12:40:22.675Z |
| 3 | P1 | K=H, S=H, C=none, R=none | P0:deny[1w,13r]; P1:allow[1w,7r] | EXPLORE_DOWN:H→M; SEARCH_DOWN | 11 /5248 | K=H, S=H, C=M*, R=none | 2026-10-06T12:40:22.675Z |
| 4 | P0 | K=H, S=H, C=none, R=none | P0:deny[2w,22r]; P1:deny[2w,18r,candidate] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 11 /18688 | K=H, S=H, C=none, R=none | 2026-10-06T12:43:01.835Z |
| 4 | P1 | K=H, S=H, C=M*, R=none | P0:deny[2w,22r]; P1:deny[2w,18r,candidate] | ACCEPT_SAFE:M→M; FINAL_EVIDENCE | 9 /3072 | K=M, S=M/H, C=none, R=none | 2026-10-06T12:43:01.835Z |
| 5 | P0 | K=H, S=H, C=none, R=none | P0:deny[3w,33r]; P1:allow[3w,27r] | HOLD:H→H; CONTEXT_CHANGE [final] | 9 /6912 | K=H, S=H, C=none, R=none | 2026-10-06T12:45:40.097Z |
| 5 | P1 | K=M, S=M/H, C=none, R=none | P0:deny[3w,33r]; P1:allow[3w,27r] | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 11 /14848 | K=M, S=M/H, C=none, R=none | 2026-10-06T12:45:40.097Z |

Final measured residency and learned states for this block: P0 L/M/H = 8.78/0.00/91.22%; P1 = 0.00/40.03/59.97%. Final P0 K=H, S=H, C=none, R=none; P1 K=M, S=M/H, C=none, R=none. Final controller consumed counters P0=4 windows/42 requests, P1=4 windows/38 requests.

### HIGH-SUSTAINABLE, repeat 3, PER-ENDPOINT-FAST-RR

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:116.026 | 10 | 4/2/3/1 | 8960 | 541.96 /100.0% | FIXED_RR:10 |
| 1 | P1 | H:116.026 | 10 | 3/1/4/2 | 12800 | 580.43 /100.0% | FIXED_RR:10 |
| 2 | P0 | M:116.028 | 10 | 4/2/3/1 | 8960 | 588.96 /100.0% | FIXED_RR:10 |
| 2 | P1 | H:116.028 | 10 | 3/1/4/2 | 12800 | 578.33 /100.0% | FIXED_RR:10 |
| 3 | P0 | M:116.021 | 10 | 4/2/3/1 | 8960 | 590.96 /100.0% | FIXED_RR:10 |
| 3 | P1 | H:116.021 | 10 | 3/1/4/2 | 12800 | 574.10 /100.0% | FIXED_RR:10 |
| 4 | P0 | L:26.035; T→M:0.610; M:89.384 | 10 | 4/2/3/1 | 8960 | 624.09 /90.0% | FIXED_RR:10 |
| 4 | P1 | H:116.029 | 10 | 3/1/4/2 | 12800 | 584.29 /100.0% | FIXED_RR:10 |
| 5 | P0 | M:116.024 | 10 | 4/2/3/1 | 8960 | 589.79 /100.0% | FIXED_RR:10 |
| 5 | P1 | H:116.024 | 10 | 3/1/4/2 | 12800 | 581.30 /100.0% | FIXED_RR:10 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | EXPLORE_DOWN:H→M; SEARCH_DOWN | 10 /8960 | K=H, S=H, C=M*, R=none | 2026-10-06T12:21:53.063Z |
| 1 | P1 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 10 /12800 | K=H, S=H, C=none, R=none | 2026-10-06T12:21:53.063Z |
| 2 | P0 | K=H, S=H, C=M*, R=none | P0:deny[1w,10r,candidate]; P1:deny[1w,10r] | ACCEPT_SAFE:M→M; FINAL_EVIDENCE | 10 /8960 | K=M, S=M/H, C=none, R=none | 2026-10-06T12:24:31.976Z |
| 2 | P1 | K=H, S=H, C=none, R=none | P0:deny[1w,10r,candidate]; P1:deny[1w,10r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 10 /12800 | K=H, S=H, C=none, R=none | 2026-10-06T12:24:31.976Z |
| 3 | P0 | K=M, S=M/H, C=none, R=none | P0:allow[2w,20r]; P1:deny[2w,20r] | EXPLORE_DOWN:M→L; SEARCH_DOWN | 10 /8960 | K=M, S=M/H, C=L*, R=none | 2026-10-06T12:27:08.238Z |
| 3 | P1 | K=H, S=H, C=none, R=none | P0:allow[2w,20r]; P1:deny[2w,20r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 10 /12800 | K=H, S=H, C=none, R=none | 2026-10-06T12:27:08.238Z |
| 4 | P0 | K=M, S=M/H, C=L*, R=none | P0:allow[3w,30r]; P1:deny[3w,30r] | HOLD:—→—; MIXED_FREQUENCY_WINDOW | not consumed /— | K=M, S=M/H, C=none, R=L | 2026-10-06T12:29:47.345Z |
| 4 | P1 | K=H, S=H, C=none, R=none | P0:allow[3w,30r]; P1:deny[3w,30r] | HOLD:—→—; MIXED_FREQUENCY_WINDOW | not consumed /— | K=H, S=H, C=none, R=none | 2026-10-06T12:29:47.345Z |
| 5 | P0 | K=M, S=M/H, C=none, R=L | P0:allow[3w,30r]; P1:deny[3w,30r] | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 10 /8960 | K=M, S=M/H, C=none, R=L | 2026-10-06T12:32:23.794Z |
| 5 | P1 | K=H, S=H, C=none, R=none | P0:allow[3w,30r]; P1:deny[3w,30r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE [final] | 10 /12800 | K=H, S=H, C=none, R=none | 2026-10-06T12:32:23.794Z |

Final measured residency and learned states for this block: P0 L/M/H = 4.49/75.49/20.02%; P1 = 0.00/0.00/100.00%. Final P0 K=M, S=M/H, C=none, R=L; P1 K=H, S=H, C=none, R=none. Final controller consumed counters P0=4 windows/40 requests, P1=4 windows/40 requests.

### HIGH-SUSTAINABLE, repeat 3, SHARED-FAST-RR

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:116.011 | 10 | 4/2/3/1 | 8960 | 567.48 /100.0% | FIXED_RR:10 |
| 1 | P1 | H:116.011 | 10 | 3/1/4/2 | 12800 | 572.80 /100.0% | FIXED_RR:10 |
| 2 | P0 | M:116.043 | 10 | 4/2/3/1 | 8960 | 562.33 /100.0% | FIXED_RR:10 |
| 2 | P1 | M:116.043 | 10 | 3/1/4/2 | 12800 | 580.69 /100.0% | FIXED_RR:10 |
| 3 | P0 | L:25.983; T→M:0.613; M:89.441 | 10 | 4/2/3/1 | 8960 | 583.56 /90.0% | FIXED_RR:10 |
| 3 | P1 | L:26.890; T→M:0.892; M:88.254 | 10 | 3/1/4/2 | 12800 | 592.85 /90.0% | FIXED_RR:10 |
| 4 | P0 | M:116.040 | 10 | 4/2/3/1 | 8960 | 552.82 /100.0% | FIXED_RR:10 |
| 4 | P1 | M:116.040 | 10 | 3/1/4/2 | 12800 | 597.62 /100.0% | FIXED_RR:10 |
| 5 | P0 | M:116.039 | 10 | 4/2/3/1 | 8960 | 551.99 /100.0% | FIXED_RR:10 |
| 5 | P1 | M:116.039 | 10 | 3/1/4/2 | 12800 | 576.62 /100.0% | FIXED_RR:10 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | shared pool | EXPLORE_DOWN:H→M; SEARCH_DOWN | 20 /21760 | K=H, S=H, C=M*, R=none | 2026-10-06T12:08:38.014Z |
| 1 | P1 | K=H, S=H, C=none, R=none | shared pool | EXPLORE_DOWN:H→M; SEARCH_DOWN | 20 /21760 | K=H, S=H, C=M*, R=none | 2026-10-06T12:08:38.014Z |
| 2 | P0 | K=H, S=H, C=M*, R=none | shared pool | EXPLORE_DOWN:M→L; SEARCH_DOWN | 20 /21760 | K=M, S=M/H, C=L*, R=none | 2026-10-06T12:11:20.204Z |
| 2 | P1 | K=H, S=H, C=M*, R=none | shared pool | EXPLORE_DOWN:M→L; SEARCH_DOWN | 20 /21760 | K=M, S=M/H, C=L*, R=none | 2026-10-06T12:11:20.204Z |
| 3 | P0 | K=M, S=M/H, C=L*, R=none | shared pool | HOLD:—→M; MIXED_CLOCK_EVIDENCE | not consumed /— | K=M, S=M/H, C=none, R=L | 2026-10-06T12:13:59.150Z |
| 3 | P1 | K=M, S=M/H, C=L*, R=none | shared pool | HOLD:—→M; MIXED_CLOCK_EVIDENCE | not consumed /— | K=M, S=M/H, C=none, R=L | 2026-10-06T12:13:59.150Z |
| 4 | P0 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; SEARCH_CONVERGED | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T12:16:35.874Z |
| 4 | P1 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; SEARCH_CONVERGED | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T12:16:35.874Z |
| 5 | P0 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T12:19:13.509Z |
| 5 | P1 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T12:19:13.509Z |

Final measured residency and learned states for this block: P0 L/M/H = 4.48/75.50/20.02%; P1 = 4.64/75.33/20.03%. Final P0 K=M, S=M/H, C=none, R=L; P1 K=M, S=M/H, C=none, R=L. Final controller consumed counters P0=5 windows/100 requests, P1=5 windows/100 requests (same pooled counters, not separate endpoint evidence).

### HIGH-SUSTAINABLE, repeat 3, STATIC-HIGH-RR

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:116.028 | 10 | 4/2/3/1 | 8960 | 543.92 /100.0% | FIXED_RR:10 |
| 1 | P1 | H:116.028 | 10 | 3/1/4/2 | 12800 | 575.59 /100.0% | FIXED_RR:10 |
| 2 | P0 | H:116.031 | 10 | 4/2/3/1 | 8960 | 543.16 /100.0% | FIXED_RR:10 |
| 2 | P1 | H:116.031 | 10 | 3/1/4/2 | 12800 | 572.63 /100.0% | FIXED_RR:10 |
| 3 | P0 | H:116.020 | 10 | 4/2/3/1 | 8960 | 537.32 /100.0% | FIXED_RR:10 |
| 3 | P1 | H:116.020 | 10 | 3/1/4/2 | 12800 | 574.34 /100.0% | FIXED_RR:10 |
| 4 | P0 | H:116.034 | 10 | 4/2/3/1 | 8960 | 545.68 /100.0% | FIXED_RR:10 |
| 4 | P1 | H:116.034 | 10 | 3/1/4/2 | 12800 | 583.97 /100.0% | FIXED_RR:10 |
| 5 | P0 | H:116.030 | 10 | 4/2/3/1 | 8960 | 544.01 /100.0% | FIXED_RR:10 |
| 5 | P1 | H:116.030 | 10 | 3/1/4/2 | 12800 | 587.67 /100.0% | FIXED_RR:10 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 1 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 2 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 2 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 3 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 3 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 4 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 4 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 5 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 5 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |

Final measured residency and learned states for this block: P0 L/M/H = 0.00/0.00/100.00%; P1 = 0.00/0.00/100.00%. Final P0 K=H, S=H, C=none, R=none; P1 K=H, S=H, C=none, R=none. Final controller consumed counters P0=0 windows/0 requests, P1=0 windows/0 requests.

### MEDIUM, repeat 3, JOINT-ROUTING-PER-ENDPOINT

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:200.264 | 13 | 7/0/5/1 | 13184 | 478.48 /100.0% | COLD_START_RR:2; LOWER_PRESSURE:9; EXPLORATION:2 |
| 1 | P1 | H:200.264 | 7 | 0/3/2/2 | 8576 | 469.98 /100.0% | COLD_START_RR:2; EXPLORATION:4; LOWER_PRESSURE:1 |
| 2 | P0 | L:201.841 | 7 | 4/3/0/0 | 896 | 201.29 /100.0% | EXPLORATION:7 |
| 2 | P1 | H:201.841 | 13 | 3/0/7/3 | 20864 | 562.76 /100.0% | ONLY_SAFE_PAIR:10; EXPLORATION:3 |
| 3 | P0 | H:200.551 | 7 | 3/0/2/2 | 8576 | 474.09 /100.0% | LOWER_PRESSURE:7 |
| 3 | P1 | H:200.551 | 13 | 4/3/5/1 | 13184 | 475.91 /100.0% | EXPLORATION:4; ENERGY_PREFERENCE:6; LOWER_PRESSURE:3 |
| 4 | P0 | H:200.866 | 12 | 6/3/3/0 | 7296 | 560.31 /100.0% | EXPLORATION:9; LOWER_PRESSURE:3 |
| 4 | P1 | H:200.866 | 8 | 1/0/4/3 | 14464 | 479.20 /100.0% | ENERGY_PREFERENCE:4; LOWER_PRESSURE:4 |
| 5 | P0 | M:200.251 | 9 | 6/3/0/0 | 1152 | 290.64 /100.0% | EXPLORATION:9 |
| 5 | P1 | H:200.251 | 11 | 1/0/7/3 | 20608 | 477.68 /100.0% | ENERGY_PREFERENCE:4; LOWER_PRESSURE:7 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | EXPLORE_DOWN:H→L; SEARCH_DOWN | 13 /13184 | K=H, S=H, C=L*, R=none | 2026-10-06T09:23:42.400Z |
| 1 | P1 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | HOLD:H→H; INSUFFICIENT_EVIDENCE | 7 /8576 | K=H, S=H, C=none, R=none | 2026-10-06T09:23:42.400Z |
| 2 | P0 | K=H, S=H, C=L*, R=none | P0:deny[1w,13r,candidate]; P1:deny[1w,7r] | RECOVER:L→H; EVIDENCE_TIMEOUT | 7 /896 | K=H, S=H, C=none, R=none | 2026-10-06T09:27:47.550Z |
| 2 | P1 | K=H, S=H, C=none, R=none | P0:deny[1w,13r,candidate]; P1:deny[1w,7r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 13 /20864 | K=H, S=H, C=none, R=none | 2026-10-06T09:27:47.550Z |
| 3 | P0 | K=H, S=H, C=none, R=none | P0:allow[2w,20r]; P1:deny[2w,20r] | HOLD:H→H; INSUFFICIENT_EVIDENCE | 7 /8576 | K=H, S=H, C=none, R=none | 2026-10-06T09:31:51.185Z |
| 3 | P1 | K=H, S=H, C=none, R=none | P0:allow[2w,20r]; P1:deny[2w,20r] | HOLD:H→H; CONTEXT_CHANGE | 13 /13184 | K=H, S=H, C=none, R=none | 2026-10-06T09:31:51.185Z |
| 4 | P0 | K=H, S=H, C=none, R=none | P0:allow[3w,27r]; P1:deny[3w,33r] | EXPLORE_DOWN:H→M; SEARCH_DOWN | 12 /7296 | K=H, S=H, C=M*, R=none | 2026-10-06T09:35:54.963Z |
| 4 | P1 | K=H, S=H, C=none, R=none | P0:allow[3w,27r]; P1:deny[3w,33r] | HOLD:H→H; CONTEXT_CHANGE | 8 /14464 | K=H, S=H, C=none, R=none | 2026-10-06T09:35:54.963Z |
| 5 | P0 | K=H, S=H, C=M*, R=none | P0:deny[4w,39r,candidate]; P1:deny[4w,41r] | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 9 /1152 | K=M, S=M/H, C=none, R=none | 2026-10-06T09:39:56.424Z |
| 5 | P1 | K=H, S=H, C=none, R=none | P0:deny[4w,39r,candidate]; P1:deny[4w,41r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE [final] | 11 /20608 | K=H, S=H, C=none, R=none | 2026-10-06T09:39:56.424Z |

Final measured residency and learned states for this block: P0 L/M/H = 20.11/19.95/59.94%; P1 = 0.00/0.00/100.00%. Final P0 K=M, S=M/H, C=none, R=none; P1 K=H, S=H, C=none, R=none. Final controller consumed counters P0=5 windows/48 requests, P1=5 windows/52 requests.

### MEDIUM, repeat 3, PER-ENDPOINT-FAST-RR

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:200.871 | 10 | 4/2/3/1 | 8960 | 559.14 /100.0% | FIXED_RR:10 |
| 1 | P1 | H:200.871 | 10 | 3/1/4/2 | 12800 | 568.28 /100.0% | FIXED_RR:10 |
| 2 | P0 | M:200.878 | 10 | 4/2/3/1 | 8960 | 562.14 /100.0% | FIXED_RR:10 |
| 2 | P1 | H:200.878 | 10 | 3/1/4/2 | 12800 | 570.02 /100.0% | FIXED_RR:10 |
| 3 | P0 | M:200.882 | 10 | 4/2/3/1 | 8960 | 552.57 /100.0% | FIXED_RR:10 |
| 3 | P1 | H:200.882 | 10 | 3/1/4/2 | 12800 | 574.59 /100.0% | FIXED_RR:10 |
| 4 | P0 | L:50.999; T→M:0.593; M:149.277 | 10 | 4/2/3/1 | 8960 | 590.59 /90.0% | FIXED_RR:10 |
| 4 | P1 | H:200.869 | 10 | 3/1/4/2 | 12800 | 557.06 /100.0% | FIXED_RR:10 |
| 5 | P0 | M:200.888 | 10 | 4/2/3/1 | 8960 | 550.43 /100.0% | FIXED_RR:10 |
| 5 | P1 | H:200.888 | 10 | 3/1/4/2 | 12800 | 575.87 /100.0% | FIXED_RR:10 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | EXPLORE_DOWN:H→M; SEARCH_DOWN | 10 /8960 | K=H, S=H, C=M*, R=none | 2026-10-06T09:03:26.110Z |
| 1 | P1 | K=H, S=H, C=none, R=none | P0:allow[0w,0r]; P1:deny[0w,0r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 10 /12800 | K=H, S=H, C=none, R=none | 2026-10-06T09:03:26.110Z |
| 2 | P0 | K=H, S=H, C=M*, R=none | P0:deny[1w,10r,candidate]; P1:deny[1w,10r] | ACCEPT_SAFE:M→M; FINAL_EVIDENCE | 10 /8960 | K=M, S=M/H, C=none, R=none | 2026-10-06T09:07:31.185Z |
| 2 | P1 | K=H, S=H, C=none, R=none | P0:deny[1w,10r,candidate]; P1:deny[1w,10r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 10 /12800 | K=H, S=H, C=none, R=none | 2026-10-06T09:07:31.185Z |
| 3 | P0 | K=M, S=M/H, C=none, R=none | P0:allow[2w,20r]; P1:deny[2w,20r] | EXPLORE_DOWN:M→L; SEARCH_DOWN | 10 /8960 | K=M, S=M/H, C=L*, R=none | 2026-10-06T09:11:34.703Z |
| 3 | P1 | K=H, S=H, C=none, R=none | P0:allow[2w,20r]; P1:deny[2w,20r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE | 10 /12800 | K=H, S=H, C=none, R=none | 2026-10-06T09:11:34.703Z |
| 4 | P0 | K=M, S=M/H, C=L*, R=none | P0:allow[3w,30r]; P1:deny[3w,30r] | HOLD:—→—; MIXED_FREQUENCY_WINDOW | not consumed /— | K=M, S=M/H, C=none, R=L | 2026-10-06T09:15:39.184Z |
| 4 | P1 | K=H, S=H, C=none, R=none | P0:allow[3w,30r]; P1:deny[3w,30r] | HOLD:—→—; MIXED_FREQUENCY_WINDOW | not consumed /— | K=H, S=H, C=none, R=none | 2026-10-06T09:15:39.184Z |
| 5 | P0 | K=M, S=M/H, C=none, R=L | P0:allow[3w,30r]; P1:deny[3w,30r] | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 10 /8960 | K=M, S=M/H, C=none, R=L | 2026-10-06T09:19:40.820Z |
| 5 | P1 | K=H, S=H, C=none, R=none | P0:allow[3w,30r]; P1:deny[3w,30r] | ACCEPT_SAFE:H→H; FINAL_EVIDENCE [final] | 10 /12800 | K=H, S=H, C=none, R=none | 2026-10-06T09:19:40.820Z |

Final measured residency and learned states for this block: P0 L/M/H = 5.08/74.91/20.01%; P1 = 0.00/0.00/100.00%. Final P0 K=M, S=M/H, C=none, R=L; P1 K=H, S=H, C=none, R=none. Final controller consumed counters P0=4 windows/40 requests, P1=4 windows/40 requests.

### MEDIUM, repeat 3, SHARED-FAST-RR

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:200.874 | 10 | 4/1/4/1 | 10880 | 572.18 /100.0% | FIXED_RR:10 |
| 1 | P1 | H:200.874 | 10 | 3/2/3/2 | 10880 | 522.53 /100.0% | FIXED_RR:10 |
| 2 | P0 | M:200.894 | 10 | 4/1/4/1 | 10880 | 582.43 /100.0% | FIXED_RR:10 |
| 2 | P1 | M:200.894 | 10 | 3/2/3/2 | 10880 | 539.22 /100.0% | FIXED_RR:10 |
| 3 | P0 | L:50.933; T→M:0.632; M:149.332 | 10 | 4/1/4/1 | 10880 | 606.77 /90.0% | FIXED_RR:10 |
| 3 | P1 | L:51.854; T→M:1.444; M:147.599 | 10 | 3/2/3/2 | 10880 | 562.51 /100.0% | FIXED_RR:10 |
| 4 | P0 | M:200.890 | 10 | 4/1/4/1 | 10880 | 575.05 /100.0% | FIXED_RR:10 |
| 4 | P1 | M:200.890 | 10 | 3/2/3/2 | 10880 | 538.63 /100.0% | FIXED_RR:10 |
| 5 | P0 | M:200.885 | 10 | 4/1/4/1 | 10880 | 575.52 /100.0% | FIXED_RR:10 |
| 5 | P1 | M:200.885 | 10 | 3/2/3/2 | 10880 | 535.06 /100.0% | FIXED_RR:10 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | shared pool | EXPLORE_DOWN:H→M; SEARCH_DOWN | 20 /21760 | K=H, S=H, C=M*, R=none | 2026-10-06T10:04:10.053Z |
| 1 | P1 | K=H, S=H, C=none, R=none | shared pool | EXPLORE_DOWN:H→M; SEARCH_DOWN | 20 /21760 | K=H, S=H, C=M*, R=none | 2026-10-06T10:04:10.053Z |
| 2 | P0 | K=H, S=H, C=M*, R=none | shared pool | EXPLORE_DOWN:M→L; SEARCH_DOWN | 20 /21760 | K=M, S=M/H, C=L*, R=none | 2026-10-06T10:08:17.762Z |
| 2 | P1 | K=H, S=H, C=M*, R=none | shared pool | EXPLORE_DOWN:M→L; SEARCH_DOWN | 20 /21760 | K=M, S=M/H, C=L*, R=none | 2026-10-06T10:08:17.762Z |
| 3 | P0 | K=M, S=M/H, C=L*, R=none | shared pool | HOLD:—→M; MIXED_CLOCK_EVIDENCE | not consumed /— | K=M, S=M/H, C=none, R=L | 2026-10-06T10:12:22.054Z |
| 3 | P1 | K=M, S=M/H, C=L*, R=none | shared pool | HOLD:—→M; MIXED_CLOCK_EVIDENCE | not consumed /— | K=M, S=M/H, C=none, R=L | 2026-10-06T10:12:22.054Z |
| 4 | P0 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; SEARCH_CONVERGED | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T10:16:23.901Z |
| 4 | P1 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; SEARCH_CONVERGED | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T10:16:23.901Z |
| 5 | P0 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T10:20:26.314Z |
| 5 | P1 | K=M, S=M/H, C=none, R=L | shared pool | ACCEPT_SAFE:M→M; FINAL_EVIDENCE [final] | 20 /21760 | K=M, S=M/H, C=none, R=L | 2026-10-06T10:20:26.314Z |

Final measured residency and learned states for this block: P0 L/M/H = 5.07/74.91/20.01%; P1 = 5.17/74.80/20.03%. Final P0 K=M, S=M/H, C=none, R=L; P1 K=M, S=M/H, C=none, R=L. Final controller consumed counters P0=5 windows/100 requests, P1=5 windows/100 requests (same pooled counters, not separate endpoint evidence).

### MEDIUM, repeat 3, STATIC-HIGH-RR

| Window | P | Clock path (seconds) | Assigned n | SS/SL/LS/LL | Input tokens | TTFT p95 ms /attainment | Routing reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | H:200.885 | 10 | 4/1/4/1 | 10880 | 568.10 /100.0% | FIXED_RR:10 |
| 1 | P1 | H:200.885 | 10 | 3/2/3/2 | 10880 | 527.40 /100.0% | FIXED_RR:10 |
| 2 | P0 | H:200.875 | 10 | 4/1/4/1 | 10880 | 566.09 /100.0% | FIXED_RR:10 |
| 2 | P1 | H:200.875 | 10 | 3/2/3/2 | 10880 | 522.42 /100.0% | FIXED_RR:10 |
| 3 | P0 | H:200.872 | 10 | 4/1/4/1 | 10880 | 563.53 /100.0% | FIXED_RR:10 |
| 3 | P1 | H:200.872 | 10 | 3/2/3/2 | 10880 | 518.20 /100.0% | FIXED_RR:10 |
| 4 | P0 | H:200.878 | 10 | 4/1/4/1 | 10880 | 568.90 /100.0% | FIXED_RR:10 |
| 4 | P1 | H:200.878 | 10 | 3/2/3/2 | 10880 | 524.28 /100.0% | FIXED_RR:10 |
| 5 | P0 | H:200.877 | 10 | 4/1/4/1 | 10880 | 568.71 /100.0% | FIXED_RR:10 |
| 5 | P1 | H:200.877 | 10 | 3/2/3/2 | 10880 | 525.42 /100.0% | FIXED_RR:10 |

| Window evidence | P | Entry state | Guard before epoch (P0,P1) | Epoch recommendation /trigger | Controller evidence n /input tokens | After epoch state | Evidence time UTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 1 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 2 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 2 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 3 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 3 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 4 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 4 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 5 | P0 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |
| 5 | P1 | K=H, S=H, C=none, R=none | not applicable | STATIC HOLD | not consumed | K=H, S=H, C=none, R=none | — |

Final measured residency and learned states for this block: P0 L/M/H = 0.00/0.00/100.00%; P1 = 0.00/0.00/100.00%. Final P0 K=H, S=H, C=none, R=none; P1 K=H, S=H, C=none, R=none. Final controller consumed counters P0=0 windows/0 requests, P1=0 windows/0 requests.

## Appendix B. Every P safety event and successful physical recovery

| Block | Window | Endpoint/pool | Trigger /action | Request | Prior→target MHz | Decision timestamp | Action timestamp | Readback timestamp | Readback MHz /valid | TTFT ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r01-medium-shared-fast-rr | 3 | P pool | HARD_TTFT_VIOLATION /FAST_RECOVER | d1-medium-r01-w03-0005 | 1260→1890 | 1791269218.562707 | 1791269218.918412 | 1791269220.739671 | 1890 /true | 634.04 |
| r01-medium-per-endpoint-fast-rr | 4 | P0 | HARD_TTFT_VIOLATION /FAST_RECOVER | d1-medium-r01-w04-0005 | 1260→1890 | 1791270680.706902 | 1791270681.010765 | 1791270681.642968 | 1890 /true | 615.15 |
| r02-medium-shared-fast-rr | 3 | P pool | HARD_TTFT_VIOLATION /FAST_RECOVER | d1-medium-r02-w03-0005 | 1260→1890 | 1791272881.731162 | 1791272882.038718 | 1791272884.436419 | 1890 /true | 641.48 |
| r02-medium-per-endpoint-fast-rr | 4 | P0 | HARD_TTFT_VIOLATION /FAST_RECOVER | d1-medium-r02-w04-0005 | 1260→1890 | 1791274340.144951 | 1791274340.449649 | 1791274341.052413 | 1890 /true | 613.51 |
| r03-medium-per-endpoint-fast-rr | 4 | P0 | HARD_TTFT_VIOLATION /FAST_RECOVER | d1-medium-r03-w04-0005 | 1260→1890 | 1791277987.195252 | 1791277987.496531 | 1791277988.089670 | 1890 /true | 622.09 |
| r03-medium-shared-fast-rr | 3 | P pool | HARD_TTFT_VIOLATION /FAST_RECOVER | d1-medium-r03-w03-0005 | 1260→1890 | 1791281390.040576 | 1791281390.341439 | 1791281392.706187 | 1890 /true | 631.41 |
| r01-high-sustainable-joint-routing-per-endpoint | 3 | P0 | HARD_TTFT_VIOLATION /LOG_ONLY_NO_ACTIVE_CANDIDATE | d1-high-sustainable-r01-w03-0005 | 1260→1260 | 1791282413.494582 | — | — | — /— | 602.89 |
| r01-high-sustainable-joint-routing-per-endpoint | 3 | P0 | ENDPOINT_EVIDENCE_SLO_RISK /FAST_RECOVER | d1-high-sustainable-r01-w03-0005 | 1260→2520 | 1791282413.522297 | 1791282413.818632 | 1791282414.434033 | 2520 /true | 602.89 |
| r01-high-sustainable-shared-fast-rr | 3 | P pool | HARD_TTFT_VIOLATION /FAST_RECOVER | d1-high-sustainable-r01-w03-0005 | 1260→1890 | 1791283997.387490 | 1791283997.738592 | 1791283999.461385 | 1890 /true | 605.51 |
| r01-high-sustainable-per-endpoint-fast-rr | 4 | P0 | HARD_TTFT_VIOLATION /FAST_RECOVER | d1-high-sustainable-r01-w04-0007 | 1260→1890 | 1791284958.034629 | 1791284958.346002 | 1791284958.983746 | 1890 /true | 653.56 |
| r02-high-sustainable-per-endpoint-fast-rr | 4 | P0 | HARD_TTFT_VIOLATION /FAST_RECOVER | d1-high-sustainable-r02-w04-0005 | 1260→1890 | 1791287315.505302 | 1791287315.805043 | 1791287316.429202 | 1890 /true | 648.06 |
| r03-high-sustainable-shared-fast-rr | 3 | P pool | HARD_TTFT_VIOLATION /FAST_RECOVER | d1-high-sustainable-r03-w03-0005 | 1260→1890 | 1791288747.040486 | 1791288747.415349 | 1791288749.214891 | 1890 /true | 606.61 |
| r03-high-sustainable-shared-fast-rr | 3 | P pool | HARD_TTFT_VIOLATION /LOG_ONLY_NO_ACTIVE_CANDIDATE | d1-high-sustainable-r03-w03-0015 | 1890→1890 | 1791288797.033169 | — | — | — /— | 600.38 |
| r03-high-sustainable-per-endpoint-fast-rr | 4 | P0 | HARD_TTFT_VIOLATION /FAST_RECOVER | d1-high-sustainable-r03-w04-0005 | 1260→1890 | 1791289695.203183 | 1791289695.509495 | 1791289696.119824 | 1890 /true | 650.23 |
| r03-high-sustainable-joint-routing-per-endpoint | 2 | P0 | HARD_TTFT_VIOLATION /FAST_RECOVER | d1-high-sustainable-r03-w02-0010 | 1260→2520 | 1791290197.800468 | 1791290198.100877 | 1791290198.747401 | 2520 /true | 646.78 |

Window-boundary `EVIDENCE_TIMEOUT` and `CONTEXT_CHANGE` recoveries are efficiency decisions, not FAST_RECOVER events; they remain in Appendix A and in original `clock_actions.csv`. No count silently combines them with fast safety.

## Appendix C. Every endpoint final residency (repeat-scoped)

| Load | Repeat | Policy | P0 L/M/H % | P1 L/M/H % | D0 LOW/MID/HIGH % | D1 LOW/MID/HIGH % |
| --- | --- | --- | --- | --- | --- | --- |
| HIGH-SUSTAINABLE | 1 | JOINT-ROUTING-PER-ENDPOINT | 24.50/20.03/55.47 | 0.00/0.00/100.00 | 0.00/40.02/59.98 | 0.00/0.00/100.00 |
| HIGH-SUSTAINABLE | 1 | PER-ENDPOINT-FAST-RR | 6.20/73.81/20.00 | 0.00/0.00/100.00 | 10.23/64.98/24.78 | 0.00/0.00/100.00 |
| HIGH-SUSTAINABLE | 1 | SHARED-FAST-RR | 4.48/75.49/20.03 | 4.68/75.31/20.02 | 0.00/5.92/94.08 | 0.00/6.11/93.89 |
| HIGH-SUSTAINABLE | 1 | STATIC-HIGH-RR | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 |
| MEDIUM | 1 | JOINT-ROUTING-PER-ENDPOINT | 20.11/19.95/59.94 | 0.00/0.00/100.00 | 0.00/60.07/39.93 | 0.00/0.00/100.00 |
| MEDIUM | 1 | PER-ENDPOINT-FAST-RR | 5.07/74.91/20.01 | 0.00/0.00/100.00 | 10.89/69.10/20.01 | 0.00/0.00/100.00 |
| MEDIUM | 1 | SHARED-FAST-RR | 5.08/74.91/20.01 | 5.17/74.81/20.02 | 0.00/5.90/94.10 | 0.00/6.00/94.00 |
| MEDIUM | 1 | STATIC-HIGH-RR | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 |
| HIGH-SUSTAINABLE | 2 | JOINT-ROUTING-PER-ENDPOINT | 0.00/39.97/60.03 | 0.00/0.00/100.00 | 0.00/40.01/59.99 | 0.00/0.00/100.00 |
| HIGH-SUSTAINABLE | 2 | PER-ENDPOINT-FAST-RR | 4.48/75.50/20.02 | 0.00/0.00/100.00 | 0.00/5.91/94.09 | 0.00/0.00/100.00 |
| HIGH-SUSTAINABLE | 2 | SHARED-FAST-RR | 0.00/80.00/20.00 | 0.00/80.00/20.00 | 0.00/5.91/94.09 | 0.00/6.09/93.91 |
| HIGH-SUSTAINABLE | 2 | STATIC-HIGH-RR | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 |
| MEDIUM | 2 | JOINT-ROUTING-PER-ENDPOINT | 60.07/0.00/39.93 | 0.00/0.00/100.00 | 0.00/40.12/59.88 | 0.00/0.00/100.00 |
| MEDIUM | 2 | PER-ENDPOINT-FAST-RR | 5.08/74.91/20.01 | 0.00/0.00/100.00 | 0.00/5.90/94.10 | 0.00/0.00/100.00 |
| MEDIUM | 2 | SHARED-FAST-RR | 5.08/74.91/20.01 | 5.17/74.80/20.03 | 0.00/5.90/94.10 | 0.00/6.00/94.00 |
| MEDIUM | 2 | STATIC-HIGH-RR | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 |
| HIGH-SUSTAINABLE | 3 | JOINT-ROUTING-PER-ENDPOINT | 8.78/0.00/91.22 | 0.00/40.03/59.97 | 0.00/39.96/60.04 | 0.00/0.00/100.00 |
| HIGH-SUSTAINABLE | 3 | PER-ENDPOINT-FAST-RR | 4.49/75.49/20.02 | 0.00/0.00/100.00 | 10.22/69.75/20.02 | 0.00/0.00/100.00 |
| HIGH-SUSTAINABLE | 3 | SHARED-FAST-RR | 4.48/75.50/20.02 | 4.64/75.33/20.03 | 0.00/5.90/94.10 | 0.00/6.08/93.92 |
| HIGH-SUSTAINABLE | 3 | STATIC-HIGH-RR | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 |
| MEDIUM | 3 | JOINT-ROUTING-PER-ENDPOINT | 20.11/19.95/59.94 | 0.00/0.00/100.00 | 0.00/60.07/39.93 | 0.00/0.00/100.00 |
| MEDIUM | 3 | PER-ENDPOINT-FAST-RR | 5.08/74.91/20.01 | 0.00/0.00/100.00 | 10.90/69.09/20.01 | 0.00/0.00/100.00 |
| MEDIUM | 3 | SHARED-FAST-RR | 5.07/74.91/20.01 | 5.17/74.80/20.03 | 0.00/5.90/94.10 | 0.00/5.99/94.01 |
| MEDIUM | 3 | STATIC-HIGH-RR | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 |

## Appendix D. Per-repeat scientific results, unchanged

| Load | Repeat | Policy | Gross energy J | J/request | TTFT attainment | TPOT attainment | SLO pass | Raw paired saving |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HIGH-SUSTAINABLE | 1 | STATIC-HIGH-RR | 168114.22 | 1681.14 | 100.0% | 100.0% | true | baseline |
| HIGH-SUSTAINABLE | 1 | SHARED-FAST-RR | 154632.36 | 1546.32 | 99.0% | 99.0% | true | 8.02% |
| HIGH-SUSTAINABLE | 1 | PER-ENDPOINT-FAST-RR | 158715.67 | 1587.16 | 99.0% | 94.0% | false | 5.59% |
| HIGH-SUSTAINABLE | 1 | JOINT-ROUTING-PER-ENDPOINT | 162589.13 | 1625.89 | 99.0% | 100.0% | true | 3.29% |
| HIGH-SUSTAINABLE | 2 | STATIC-HIGH-RR | 168219.00 | 1682.19 | 100.0% | 100.0% | true | baseline |
| HIGH-SUSTAINABLE | 2 | SHARED-FAST-RR | 155049.21 | 1550.49 | 100.0% | 99.0% | true | 7.83% |
| HIGH-SUSTAINABLE | 2 | PER-ENDPOINT-FAST-RR | 162309.03 | 1623.09 | 99.0% | 99.0% | true | 3.51% |
| HIGH-SUSTAINABLE | 2 | JOINT-ROUTING-PER-ENDPOINT | 163815.77 | 1638.16 | 100.0% | 100.0% | true | 2.62% |
| HIGH-SUSTAINABLE | 3 | STATIC-HIGH-RR | 168085.63 | 1680.86 | 100.0% | 100.0% | true | baseline |
| HIGH-SUSTAINABLE | 3 | SHARED-FAST-RR | 154382.79 | 1543.83 | 98.0% | 99.0% | true | 8.15% |
| HIGH-SUSTAINABLE | 3 | PER-ENDPOINT-FAST-RR | 158416.60 | 1584.17 | 99.0% | 99.0% | true | 5.75% |
| HIGH-SUSTAINABLE | 3 | JOINT-ROUTING-PER-ENDPOINT | 161364.43 | 1613.64 | 99.0% | 100.0% | true | 4.00% |
| MEDIUM | 1 | STATIC-HIGH-RR | 261326.03 | 2613.26 | 98.0% | 100.0% | true | baseline |
| MEDIUM | 1 | SHARED-FAST-RR | 238818.50 | 2388.18 | 99.0% | 99.0% | true | 8.61% |
| MEDIUM | 1 | PER-ENDPOINT-FAST-RR | 246998.07 | 2469.98 | 99.0% | 99.0% | true | 5.48% |
| MEDIUM | 1 | JOINT-ROUTING-PER-ENDPOINT | 256634.22 | 2566.34 | 100.0% | 100.0% | true | 1.80% |
| MEDIUM | 2 | STATIC-HIGH-RR | 261616.67 | 2616.17 | 100.0% | 100.0% | true | baseline |
| MEDIUM | 2 | SHARED-FAST-RR | 239263.38 | 2392.63 | 99.0% | 99.0% | true | 8.54% |
| MEDIUM | 2 | PER-ENDPOINT-FAST-RR | 251325.54 | 2513.26 | 99.0% | 99.0% | true | 3.93% |
| MEDIUM | 2 | JOINT-ROUTING-PER-ENDPOINT | 254245.60 | 2542.46 | 100.0% | 100.0% | true | 2.82% |
| MEDIUM | 3 | STATIC-HIGH-RR | 261717.11 | 2617.17 | 100.0% | 100.0% | true | baseline |
| MEDIUM | 3 | SHARED-FAST-RR | 238753.64 | 2387.54 | 99.0% | 99.0% | true | 8.77% |
| MEDIUM | 3 | PER-ENDPOINT-FAST-RR | 246551.19 | 2465.51 | 99.0% | 99.0% | true | 5.79% |
| MEDIUM | 3 | JOINT-ROUTING-PER-ENDPOINT | 256239.65 | 2562.40 | 100.0% | 100.0% | true | 2.09% |

## Completion boundary

This is an offline explanation of an accepted physical run with one genuine policy-repeat SLO failure. No JOINT improvement is manufactured or claimed. Only this document, its derived corrected audit JSON, and the exact empty control/routing diff were added. Runtime code and scientific artifacts are unchanged; no GPU job or new experiment was launched.
