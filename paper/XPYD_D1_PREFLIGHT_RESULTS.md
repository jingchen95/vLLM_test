# XPYD D1 preflight results

## Status

Job `271516` is **rejected as a schema-integration failure**. It is not a D1
mechanism result, and the full D1 experiment has not been launched.

The job completed 60 requests and exercised all four compatible routes, real
SSE, request-ID propagation, energy measurement, per-endpoint clocks, and the
endpoint-local fast-safety path. However, D1 controller/report code consumed
field names from an older request schema. The normalized measurement CSV uses
`request_id`, `prefill_endpoint_id`, `decode_endpoint_id`, and
`requested_output_len`; the D1 path expected `trace_request_id`,
`selected_prefill_endpoint_id`, `selected_decode_endpoint_id`, and
`output_len`. Consequently assigned traffic appeared as zero endpoint evidence
and final reporting raised `KeyError: trace_request_id`.

## Corrective patch

- One strict normalization boundary maps the real normalized CSV into the
  canonical internal D1 request record.
- Missing required fields now fail before controller or report evaluation.
- Controller, class/pair feedback, endpoint summaries, and reports consume only
  canonical D1 fields.
- Per-window proxy diagnostics are captured immediately so later windows cannot
  overwrite the audit input.
- Per-endpoint summaries explicitly include request/class/token/latency/energy
  evidence and final safe-search/probe/recovery state.
- The preflight now performs one predeclared all-HIGH warm-up window before the
  measured P0=MID/P1=HIGH asymmetry. Warm-up requests are separately recorded
  with `evidence_eligible=false` and never enter controller evidence.
- Requested output budget remains the online context; realized output length is
  retained only as measurement evidence.

## Validation state

Focused CPU regression tests cover real-schema normalization, hard failure for
missing fields, exact P0-to-D1 evidence locality, nonzero endpoint evidence,
endpoint energy locality, class/pair feedback, requested-versus-realized output
semantics, canonical report generation, independent endpoint search state, and
warm-up exclusion.

## Corrected preflight: job 271610

Job `271610` completed normally in 17 minutes 3 seconds on Neptune and Europa.
The schema-integration correction is validated:

- 20 declared all-HIGH warm-up requests were recorded separately and excluded
  from controller evidence;
- 60 fixed-asymmetry measurement requests reached the canonical D1 path;
- every physical measurement hard gate passed in all three measured windows;
- request context, real SSE, logical IDs, requested token budgets, gross NVML
  energy, and requested/readback clocks were valid;
- all four routes were used: P0-D0 18, P0-D1 14, P1-D0 13, P1-D1 15;
- routing produced 27 exploration, 23 lower-pressure, and 10 energy-preference
  selections;
- all four endpoints received nonzero request, class, token, latency, and
  energy evidence;
- P0=MID/P1=HIGH asymmetry was physically observed, followed by P0 recovery to
  HIGH after the derived P context changed;
- final clocks were restored successfully.

Endpoint evidence was:

| Endpoint | Requests | Input tokens | Requested/actual output tokens | Phase latency mean / p95 (ms) | Gross energy (kJ) | Final current / validated / unsafe (MHz) |
|---|---:|---:|---:|---:|---:|---|
| P0 | 32 | 42,496 | 7,552 / 7,552 | TTFT 420.89 / 576.88 | 24.786 | 2520 / 2520 / none |
| P1 | 28 | 22,784 | 7,040 / 7,040 | TTFT 335.60 / 575.94 | 28.446 | 2520 / 2520 / none |
| D0 | 31 | 28,928 | 7,808 / 7,808 | TPOT 60.31 / 65.26 | 20.989 | 1125 / 1125 / 1125 |
| D1 | 29 | 36,352 | 6,784 / 6,784 | TPOT 59.56 / 63.22 | 21.959 | 1500 / 1500 / none |

### Acceptance failure

The corrected preflight is **NOT ACCEPTED** despite the generated
`stage_d1_summary.json` containing `preflight_pass=true`.

Across the 60 measured requests, TTFT attainment was 100%, but TPOT attainment
was only 93.33%, below the fixed 95% contract. Windows 1 and 2 had 100% TPOT
attainment. In window 3, D0=MID produced four hard TPOT violations among its 11
assigned requests, so the D0 endpoint evidence correctly classified that final
window as `UNSAFE` with attainment 7/11 (63.64%).

The remaining problem is concrete safety-state and acceptance bookkeeping:

1. D0=MID was accepted after safe evidence in window 2.
2. The final-window evidence then classified the same frequency as unsafe.
3. Because the current and last validated-safe indices were both MID, the
   controller selected `HOLD` rather than the previously measured HIGH anchor.
4. Final state therefore reports MID simultaneously as
   `final_validated_safe_frequency`, `unsafe_bound`, and a converged boundary.
5. The preflight acceptance expression checks structural hard gates and routing
   behavior, but not the fixed SLO contract or safe/unsafe state consistency;
   its `true` result is therefore a false positive.

The four violations were logged, but fast safety correctly used
`LOG_ONLY_NO_ACTIVE_CANDIDATE` because D0=MID had already ceased to be an active
candidate after window 2. Thus this run did not physically re-exercise an
endpoint-local fast recovery.

This finding does not justify SLO or threshold tuning. Before another preflight,
the minimum required correction is to invalidate a previously accepted safe
frequency when fresh final evidence makes it unsafe, recover/select a strictly
higher previously measured safe anchor, and make acceptance require both the
fixed SLO and a non-contradictory final search state. No full D1 job was
launched.

Final acceptance token: **WITHHELD**.

## Safety-state correction prepared after job 271610

The follow-up patch keeps the frozen SLO, thresholds, routing hierarchy,
workload, and frequency action space unchanged. It makes validated safety
explicitly revocable per context:

- sufficient endpoint-local evidence below 95% revokes the active lower
  frequency and immediately recovers to a strictly higher validated anchor;
- one isolated hard-latency outlier remains logged but does not permanently
  revoke a validated point while recent attainment remains at least 95%;
- revoked safe-energy samples remain in historical state but are removed from
  current routing eligibility;
- a mixed-frequency recovery window cannot be relabeled as single-frequency
  energy/latency evidence;
- the final routing snapshot drops stale class/energy evidence after recovery;
- acceptance now requires both overall TTFT/TPOT attainment of at least 95%
  and a consistent final validated-safe/revoked/unsafe state.

Complete CPU validation after this correction: **326 tests passed**. A new
corrected D1 preflight is required; full D1 remains prohibited.

## Revocation preflight: job 271682

Job `271682` completed normally in 17 minutes 5 seconds on Neptune and Europa.
Most of the correction behaved as intended:

- all three 20-request measurement windows passed the physical hard gates;
- measured TTFT attainment was 100% and TPOT attainment was 98.33%;
- D0 at 1125 MHz was revoked after fresh evidence fell to 16/17 TPOT passes
  (94.12%);
- D0 recovered to the strictly higher validated-safe 1500 MHz anchor, with
  valid readback;
- final D0 state was internally consistent: validated safe `{1500}`, revoked
  and unsafe bound `{1125}`;
- the revoked 1125 MHz energy observation remained in historical reporting but
  was removed from final eligible safe-energy state;
- the complete NVML accounting window remained valid. D0 clock segments had no
  time gap or overlap and the gross window energy was counted once.

The run is nevertheless **NOT ACCEPTED**. Manual chronological auditing found a
remaining race between safety-state revocation and routing-state publication:

1. Request `d1-high-sustainable-r01-w03-0015` completed at
   `1791149148.2001276` with TPOT 65.369 ms. Its fresh D0 evidence triggered
   revocation of 1125 MHz and recovery toward 1500 MHz.
2. Physical actuation began at `1791149148.6338384`; successful 1500 MHz
   readback completed at `1791149149.3657272`.
3. Request `d1-high-sustainable-r01-w03-0017` was routed at
   `1791149149.309751`, about 56 ms before readback. It still observed the old
   D0=1125 routing snapshot and selected P1-D0 with `ENERGY_PREFERENCE` using
   the stale 1125 MHz energy evidence.
4. Later admissions correctly saw D0=1500 with no eligible old energy evidence.

The current recovery callback publishes the new routing state only after
actuation/readback, while request admission continues concurrently. Therefore a
revoked operating point can remain energy-preference eligible during the
recovery interval. The generated `revoked_energy_routing_ineligible_audit=true`
checks only the final routing snapshot and does not detect this chronological
violation; consequently generated `preflight_pass=true` is another false
positive.

The required next correction is narrow: revoke/disable the unsafe frequency's
energy eligibility atomically when the safety decision is made, before further
admission can consult it (or equivalently gate admission through the recovery),
then publish the successful readback state. The audit must examine every routing
decision after the revocation timestamp, not only the final snapshot.

Final acceptance token: **WITHHELD**. Full D1 was not launched.
