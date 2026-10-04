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

Corrected GPU preflight: **PENDING**.

Final acceptance token remains pending:

```text
D1_PREFLIGHT_PASS
```

No full D1 job may be launched until the corrected preflight is reviewed.
