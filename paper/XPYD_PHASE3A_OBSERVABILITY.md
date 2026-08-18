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
- the existing local proxy listens on 8000, sends a one-token prefill request,
  then streams the decode request/result to the client;
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
duration, shape, and concurrency values.  It scrapes both endpoints every
configured interval (1 second in the example) while the client is alive.  The
rates in the example are bring-up values, not claims of scientifically correct
light/moderate load or endpoint capacity.

## Independent evidence and timestamps

The trace client records request success/failure, exact requested input/output
lengths, actual output tokens, TTFT, TPOT when multiple output tokens are
observed, E2E latency, and send/completion times.  These values are obtained
from the client stream and tokenizer, never from Prometheus.

Each metrics GET is centrally annotated with a global sequence, scrape-round
sequence, wall-clock start/end, monotonic start/end, endpoint ID, role, and
exact URI.  Node-local log clocks remain supplementary evidence.

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
rates, reconstructed tail upper bounds, central scrape gaps, reset-like
discontinuities, all actual histogram boundaries, and whether exact 300/500/
1000 ms TTFT and 100/200 ms inter-token boundaries are visible.  These target
values are inspection points only; the harness does not enforce them as SLOs.

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
exit, any request failure, absent client summary, absent configured server log,
or result-write error fails the run.  Missing individual vLLM metric families
do not fabricate zeros: they remain `null` and are listed for analysis.

## Model provenance

No model is trained or recalibrated in Phase 3A.  Canonical prefill/decode
classifiers, latency regressors, and power estimators were trained from
monolithic single-pool Phase-2 rows spanning balanced and phase-dominant shapes;
none was trained solely from phase-dominant, phase-only, or physically
disaggregated measurements.  The artifact-level matrix and reuse status remain
in `CODEBASE_AUDIT.md` and `XPYD_RUNTIME_DESIGN.md`.
