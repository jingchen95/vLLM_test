# XpYd Phase 3A observability-validation design

## Boundary and question

Phase 3A asks one narrow question: do the metrics exposed by the real vLLM
0.15.1 1×L40S prefill endpoint and 1×L4 decode endpoint have the semantics the
frozen XpYd control plane assumes?  It correlates independent client results
with raw P0 and D0 `/metrics` observations.  It is not an optimization,
capacity, energy, routing, DVFS, predictor-training, or scheduler experiment.

The implementation reuses the existing Experiment E physical path:

- P0 is the vLLM OpenAI server on the L40S node (`neptune`, HTTP 8100), with
  `P2pNcclConnector`, `kv_producer`, and synchronous `PUT`;
- D0 is the vLLM OpenAI server on the L4 node (`europa`, HTTP 8200), with
  `P2pNcclConnector` and `kv_consumer`;
- the checked-in local proxy listens on 8000, sends a non-streaming one-token
  prefill request, then forwards the decode endpoint's real OpenAI SSE bytes;
- the existing server startup, health checks, diagnostic connector patches,
  proxy, server logs, and graceful process cleanup are retained.

When `XPYD_PHASE3A_CONFIG` is set, the physical script skips `set_gpu_freq`,
GPU/NVML monitors, power/energy calculations, persistence-mode changes, and GPU
reset/app-clock cleanup.  The Python coordinator has no GPU or server-lifecycle
API and always records `actuated=false`.

## Two modes

The controlled semantic mode takes a P0 and D0 baseline, runs a small known
trace through the existing streaming client, takes a second pair of scrapes,
and derives independent endpoint deltas.  The checked-in configuration starts
with one logical request for each configurable `(IL, OL)` pair: `(128,128)`,
`(2048,128)`, `(128,512)`, and `(2048,512)`.

The short-load mode generates a deterministic trace from configurable rate,
duration, shape, and concurrency values. Independent P0 and D0 workers scrape
on the same configured fixed-period schedule (1 second in the example) while
the client is alive. A slow endpoint cannot shift the other endpoint's next
scheduled sample. Per-endpoint overlap is forbidden: schedule slots that pass
while the prior scrape remains in flight are recorded as missed. The rates in
the example are bring-up values, not claims of scientifically correct
light/moderate load or endpoint capacity.

## Independent evidence and timestamps

The trace client records request success/failure, exact requested input/output
lengths, actual output tokens, TTFT, TPOT when multiple output tokens are
observed, E2E latency, and send/completion times.  These values are obtained
from the client stream, never from Prometheus. The decode endpoint derives exact
actual output counts and emits them in final OpenAI stream usage; the proxy
forwards that event unchanged. The canonical client rejects runs where any
successful request falls back to lossy text re-tokenization or lacks audited
real decode streaming.

One configured phase-level request initializes lazy cross-node NCCL/runtime
state before the first preflight scrape. Its client artifacts and summary are
retained, but it is explicitly excluded from all semantic and load measurement
windows.

Each metrics GET is centrally annotated with a global sequence, scrape-round
sequence, scheduled monotonic time, actual wall/monotonic start and finish,
scrape latency, scheduling drift, late status, endpoint ID, role, and exact
URI. `scrapes.jsonl` also contains explicit missed records with no fabricated
snapshot. Node-local log clocks remain supplementary evidence. This is a
fixed-period best-effort sampler, not a real-time scheduling claim.

`VLLMMetricsCollector.scrape_raw()` fetches once and returns both the exact text
and the parsed snapshot.  Thus raw retention and derivation cannot accidentally
refer to different HTTP responses.  `VLLMWindowTracker` remains the only
cumulative-to-window implementation.

## Artifacts

Each run is stored under:

```text
results/xpyd_observability/<run_id>/
  metadata.json
  client/<probe_id>/
    trace.csv
    command.json
    client.log
    requests.jsonl
    summary.json
    summary.txt
  client/_phase_warmup/             # when configured; excluded from windows
  P0/
    raw_metrics/*.prom
    server.log
  D0/
    raw_metrics/*.prom
    server.log
  derived/
    scrapes.jsonl
    telemetry.jsonl
    dry_run.jsonl                 # only when explicitly configured
    semantic_deltas.json          # semantic mode
    load_runs.json                # load mode
    phase_warmup.json             # when configured
    proxy_diagnostics.jsonl       # proxy monotonic streaming audit
    semantic_summary.json
    semantic_summary.md
    summary.json
    summary.md
    failure.json                  # only on failure
```

`metadata.json` records the commit, model/tokenizer, vLLM version, endpoint
topology, nodes/GPUs, connector, client protocol, probes, scrape interval,
server-log references, Slurm context, and explicit exclusions.  Secret-shaped
configuration keys are rejected rather than copied.

`summary.json` reports observed missing metrics, gauge maxima, token/request
rates, reconstructed tail upper bounds, scrape latency/drift and missed/late
counts, central scrape gaps, reset-like
discontinuities, all actual histogram boundaries, and whether exact 300/500/
1000 ms TTFT and 100/200 ms inter-token boundaries are visible.  These target
values are inspection points only; the harness does not enforce them as SLOs.

Each `proxy_diagnostics.jsonl` record captures the logical request ID, the
vLLM transport request ID, incoming/outgoing stream flags,
the actual D Content-Type, stream availability, request/prefill/decode/header/
first-chunk/last-chunk/completion monotonic timestamps, first-chunk forwarding
delay, and request/decode durations. These diagnostics audit the path; they are
never substituted for client TTFT, TPOT, or ITL.

## Interpretation and reset boundary

The generated semantic report uses only `approximately_matches`, `differs`, or
`unavailable` for numerical comparisons.  It does not infer that P0 “owns”
prefill or D0 “owns” decode merely because a counter moved.  Client TTFT through
this proxy includes prefill, KV transfer, and time to the first streamed decode
token, so it is not directly equivalent to either endpoint's internal TTFT
histogram.

Counter decreases, histogram decreases, and bucket-layout changes invalidate a
window and are marked as an observable reset/discontinuity.  Monotonic counters
cannot prove process continuity: a restarted process that has already exceeded
the old values is indistinguishable without process identity/start-time
evidence.  Optional restart experiments must use the existing physical
lifecycle externally and preserve the before/after raw scrapes and server logs;
the coordinator is intentionally not a general process manager.

## Failure policy

P0/D0 unreachability, non-200/empty metrics, parse ambiguity, client nonzero
exit, any request failure, absent client summary, missing exact server token
usage or real decode streaming when required, absent configured server/proxy
diagnostic log, or result-write error fails the run. Missing individual vLLM
metric families do not fabricate zeros: they remain `null` and are listed for
analysis.

## Streaming correctness boundary

vLLM 0.15.1 accepts a `stream` field on `/inference/v1/generate`, but its
`ServingTokens` implementation explicitly consumes the full engine generator
and returns JSON. The earlier proxy sent `stream=true` there, buffered that JSON,
then fabricated token timing by replaying decoded tokens with sleeps. Phase 3A
now follows the version-matched vLLM disaggregated proxy pattern instead: both P
and D use `/v1/completions` with the same `X-Request-Id`; P remains
`stream=false,max_tokens=1`, while D uses real `stream=true` and its response
bytes are forwarded incrementally. A non-SSE D response fails closed with
TTFT/TPOT/ITL invalid rather than being replayed.

The trace client's request ID is now the end-to-end logical ID. The proxy reuses
it when present or generates it once at ingress, echoes it to the client,
records it in diagnostics, and forwards it unchanged to P0/D0 in
`X-Xpyd-Logical-Request-Id`. vLLM 0.15.1's P2pNcclConnector additionally
requires peer addresses inside `X-Request-Id`; the proxy preserves that serving
contract and uses the same logical ID as the stable suffix of that transport
envelope. Thus diagnostics `request_id` and client `trace_request_id` join
exactly without altering routing or KV-transfer semantics.

## Model provenance

No model is trained or recalibrated in Phase 3A.  Canonical prefill/decode
classifiers, latency regressors, and power estimators were trained from
monolithic single-pool Phase-2 rows spanning balanced and phase-dominant shapes;
none was trained solely from phase-dominant, phase-only, or physically
disaggregated measurements.  The artifact-level matrix and reuse status remain
in `docs/CODEBASE_AUDIT.md` and `XPYD_RUNTIME_DESIGN.md`.
