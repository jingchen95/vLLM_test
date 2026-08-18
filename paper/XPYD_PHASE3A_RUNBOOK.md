# XpYd Phase 3A physical runbook (Chalmers)

This runbook prepares and invokes the existing `neptune` L40S → `europa` L4
Experiment E path. It is a procedure; each physical result is claimed only by
its archived Slurm and run-directory artifacts.

## 1. Review and preflight

Run from the repository root on the Chalmers login node:

```bash
cd "$(git rev-parse --show-toplevel)"
git status --short
python3 -m json.tool paper/configs/xpyd_phase3a_l40s_l4.json >/dev/null
bash -n run_disagg_benchmark.sh
```

Confirm the serving environment resolves the required packages and exact vLLM
version before submitting:

```bash
python3 -c 'import aiohttp, transformers, vllm; print(vllm.__version__)'
```

Expected vLLM output is `0.15.1`.  Also confirm the model is already available
through the same credentials/cache used by the existing Experiment E setup.
Do not put credentials or tokens in the Phase 3A JSON.

## 2. Optional one-request-first configuration

The checked-in semantic probes already use `count: 1`.  Submit only the first
shape for the lowest-risk bring-up:

```bash
cd "$(git rev-parse --show-toplevel)"
sbatch --export=ALL,EXP=E,E_SMOKE=1,XPYD_PHASE3A_CONFIG=paper/configs/xpyd_phase3a_l40s_l4.json,XPYD_PHASE3A_MODE=semantic,XPYD_PHASE3A_SEMANTIC_PROBE_ID=semantic_il128_ol128 \
  run_disagg_benchmark.sh
```

This makes one unmeasured phase-level warmup request before the first preflight
scrape, then exactly one configured logical request inside the measured
before/after window. The warmup initializes lazy NCCL/runtime state and is saved
under `client/_phase_warmup` plus `derived/phase_warmup.json`. The full command
below runs all four single-request semantic shapes followed by two short loads.
Adjust rate/duration only in a copied config and keep that config with results.

## 3. Submit the physical Phase 3A job

```bash
cd "$(git rev-parse --show-toplevel)"
sbatch --export=ALL,EXP=E,E_SMOKE=1,XPYD_PHASE3A_CONFIG=paper/configs/xpyd_phase3a_l40s_l4.json \
  run_disagg_benchmark.sh
```

This explicitly selects only Experiment E.  `XPYD_PHASE3A_CONFIG` causes the
script to start the existing P0, D0, and proxy, then run the observer.  In this
branch it does not call `set_gpu_freq`, start `gpu_monitor.py`, calculate
energy, enable persistence mode, or reset clocks/GPUs.  It still uses the
existing server/proxy lifecycle and cleans up those processes.

Monitor the Slurm job without changing it:

```bash
squeue -u "$USER"
tail -f logs/disagg_bench_<jobid>.log
```

Replace `<jobid>` with the ID printed by `sbatch`. Preflight, before, after, and
semantic endpoint scrapes remain strict. During a load, isolated interval
scrape failures are preserved in `derived/scrape_errors.jsonl` while the client
is allowed to finish; insufficient per-endpoint interval coverage or too many
consecutive failures still produces `derived/failure.json` and a nonzero exit.

## 4. Inspect evidence

```bash
latest_run="$(find results/xpyd_observability -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
python3 -m json.tool "$latest_run/metadata.json"
python3 -m json.tool "$latest_run/derived/summary.json"
python3 -m json.tool "$latest_run/derived/semantic_summary.json"
sed -n '1,20p' "$latest_run/derived/proxy_diagnostics.jsonl"
sed -n '1,240p' "$latest_run/derived/semantic_summary.md"
wc -l "$latest_run"/P0/raw_metrics/*.prom "$latest_run"/D0/raw_metrics/*.prom
```

For each semantic shape, compare client logical requests/input/output tokens
with P0 and D0 request/prompt/generation deltas.  Record what happened; do not
assume both endpoints increment identically.  Inspect the exact bucket arrays
and `target_boundary_visibility` around TTFT 0.3/0.5/1.0 seconds and
inter-token 0.1/0.2 seconds.  If a target lies inside a bucket, report only the
available bucket-resolution bound.

Confirm each client `summary.json` reports every successful request under
`completion_token_sources.server_usage`. The canonical config fails instead of
silently accepting lossy text re-tokenization.

Also confirm `decode_stream_available_requests` and all three client latency
validity counts equal `successful_requests`. In each proxy diagnostic record,
the incoming/outgoing stream flags must be true, D Content-Type must be
`text/event-stream`, and first-real-chunk timestamps must precede the last chunk
and response completion. A non-SSE D response is an explicit failure; never
interpret it as measured TTFT/TPOT/ITL.

For short loads, inspect `endpoint_window_behavior`, queue/running/KV maxima,
rate maxima, tail bounds, missing metric unions, and `max_central_scrape_gap_s`.
Use `telemetry.jsonl` for evolution rather than interpreting maxima alone.
Also inspect `load_runs.json` for strict before/after endpoint deltas and
`load_monitoring.json` for successful, failed, and consecutive interval scrape
counts. Check `scheduled_interval_scrapes`, `missed_interval_scrapes`,
`late_interval_scrapes`, and per-endpoint maximum/mean scheduling drift. Every
actual attempt in `scrapes.jsonl` has scheduled/start/finish/latency fields;
missed slots have `status: missed` and no fabricated timing or metric payload.
A completed run may contain explicitly tolerated interval gaps, but it must
satisfy the configured coverage policy.

Finally, `proxy_diagnostics_audit.json` must report `valid: true` and its record
count must equal the successful warmup + semantic + load request count. This
prevents a run from completing with a truncated proxy diagnostic artifact. It
must also report `logical_request_ids_exactly_match: true`: client
`logical_request_id`/`trace_request_id` and proxy diagnostic `request_id` are
the same identifier, not ordinally inferred matches.

## 5. Re-run analysis without touching servers

```bash
PYTHONPATH=paper/scripts python3 -m xpyd.phase3a_observability analyze \
  --run-dir "$latest_run"
```

This consumes saved artifacts only and rewrites the semantic JSON/Markdown
summary.  It does not contact P0/D0, run a client, or mutate hardware.

## 6. Optional controlled restart evidence

Restart validation is deliberately not automated by the observer.  If it is
needed, use the existing Experiment E stop/start lifecycle in a separately
reviewed run, mark the before/stop/start/ready/after phases in the Slurm log,
and preserve raw scrapes immediately before and after.  A counter or histogram
decrease/layout change will be marked as a discontinuity.  Absence of a
decrease is not proof that no restart occurred.  Do not introduce a second
general-purpose process manager for this check.

## 7. Archive the run

Keep the entire run directory together.  At minimum do not separate
`metadata.json`, client artifacts, P0/D0 raw metrics, both server logs, and
derived files.  Record the Slurm job log beside the run when copying results
off-cluster.
