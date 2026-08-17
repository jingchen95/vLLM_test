# First hardware profiling batch — L4, IL=1024, OL=512, TP=1, 10 frequencies

Goal: measure the **safe decode capacity** of one hardware shape
(`L4, IL=1024, OL=512, TP=1`) at each of the 10 L4 clocks. These are the top-10
missing scheduler cells (49.5% of decode lookups); confirming them raises
lookup-weighted coverage 6.9% -> 56.4%. Deliverable is a plan + manifest, NOT
execution. Does not touch sweep.py, does not enable the goodput gate.

Machine-readable manifest: `paper/hardware/first_batch_manifest.json`.

---

## 1. Existing infrastructure (reused, not reinvented)

The Phase-2 harness `archive/early_sbatch/phase2_characterization_l4.sh` already
does every atomic step; the batch only needs an **adaptive rate driver** wrapped
around its primitives.

| Need | How the existing harness does it |
|---|---|
| Launch a decode run | `vllm bench serve --backend openai --base-url http://localhost:8000 --model $MODEL --num-prompts N --request-rate R --dataset-name random --random-input-len IL --random-output-len OL --max-concurrency C --num-warmups 10` |
| Set GPU frequency | `set_gpu_freq()` -> `sudo nvidia-smi -i $g -ac "6251,<freq>"` (application clocks; **set live, server stays up**). Reset: `sudo nvidia-smi -rac`. Persistence: `sudo nvidia-smi -pm 1`. |
| TP=1 | `start_server()` -> `python -m vllm.entrypoints.openai.api_server --tensor-parallel-size 1 --max-model-len 9216`; `CUDA_VISIBLE_DEVICES` = 1 GPU. |
| IL=1024 / OL=512 | `--random-input-len 1024 --random-output-len 512` (real lengths; no proxy). |
| Request rate | `--request-rate R` (accepts floats -> adaptive rates work unchanged). |
| Warm-up | `--num-warmups 10`. |
| Steady-state / drain | implicit: bench runs `num_prompts` and waits for all to complete (drain inflates `benchmark_duration_s` under overload — our backlog signal). `sleep 3` between runs. |
| Power / energy / temp | `gpu_monitor.py` (NVML energy counter + temperature), started per run. |
| Raw results | appended to `Phase2_Results_L4/master_results.csv` (30-col schema); per-run bench + monitor files under a step dir. Idempotent skip via `already_done`. |

Fields available in `master_results.csv` and used here:

| Field | Column | Use |
|---|---|---|
| `request_rate` | offered rate R | offered_decode_tps = R x OL |
| `output_token_throughput_tps` | achieved decode tok/s | served_ratio numerator; **capacity** |
| `p99_tpot_ms` | per-token p99 | SLO-bucket filter (post-hoc in calibrate) |
| `request_throughput_rps` | achieved req/s | cross-check served ratio |
| `benchmark_duration_s`, `num_prompts` | window / count | backlog proxy (duration_ratio) |
| `avg_power_w`, `energy_j` | power/energy | recorded (not gating) |
| bench file exit / parse | — | run validity |

**Not available:** an explicit queue-length / unfinished-request time series.
Backlog growth is therefore inferred from `duration_ratio` and `served_ratio`
(see blockers).

---

## 2. Stability decision for ONE run

**Corrected (do not use aggregate throughput as the gate).** The two ratios
```
served_ratio   = output_token_throughput_tps / (request_rate * OL)
duration_ratio = benchmark_duration_s / (num_prompts / request_rate)
```
are **not independent**: since `output_token_throughput_tps ≈ num_prompts*OL /
benchmark_duration_s`, we have `served_ratio ≈ 1 / duration_ratio` — one quantity,
not two. Worse, `benchmark_duration_s = injection_window + tail_drain`, and the
last-arriving request must still generate OL=512 tokens
(`tail_drain ≈ OL * p99_tpot ≈ 10-30 s`). So **even a perfectly stable run has
`served_ratio < 1` and `duration_ratio > 1` purely from finite-batch drain.**
Thresholding `served_ratio >= 0.95` on the whole-run aggregate FALSE-FLAGS stable
runs as overload; it must NOT be the stability gate.

**PRIMARY signal — steady-state backlog (drain-immune).** Scrape vLLM `/metrics`
(Prometheus) at ~1 s during the run: `vllm:num_requests_waiting` (queue),
`vllm:num_requests_running`, `vllm:gpu_cache_usage_perc`. Steady-state window =
after the warm-up ramp and before injection ends (excludes tail drain). A run is
**STABLE** iff, over the steady-state window:
- `num_requests_waiting` does not trend up (linear-fit slope <= eps; queue bounded), AND
- KV cache is not pinned ~100% with a simultaneously growing queue, AND
- `run_ok`: bench completed, no error/timeout, and the vLLM log shows no OOM or
  preemption/recompute.

Overload = a persistently growing queue — the direct, tail-immune signal.

**SECONDARY (coarse sanity only):** a served ratio computed over the INJECTION
window (tokens generated during injection / injection_duration), never over
`benchmark_duration_s`.

**SLO buckets:** `p99_tpot <= SLO` is applied per bucket *post-hoc* by
`calibrate_decode_capacity.py`. Note: that script currently computes stability from
the aggregate `output_token_throughput_tps` and so inherits the same drain bias — it
must be reconciled with this steady-state definition (blocker 2) before the batch
data feeds it, e.g. by writing a drain-corrected/steady-state throughput or a
per-run `stable` flag into the results.

The adaptive search brackets the **C_dc_stable** knee on the PRIMARY (queue-slope)
signal; one rate sweep still yields every `C_dc_SLO` via the post-hoc p99_tpot filter.

---

## 3. Adaptive rate search (per frequency)

Motivation from existing data: nearest measured neighbors (`L4 IL=1024 OL=128`,
`IL=512 OL=128`) are pinned at `C_dc_stable ~= 126 tok/s` because the only stable
legacy rate was `1` and the next tested was `10` — the knee in (1,10) was never
bracketed. So the legacy grid {1,10,20,30,50} is insufficient; we search adaptively.

### Pseudocode
```
def profile_frequency(freq, seed_rate, min_rate, max_rate, max_runs):
    set_gpu_freq(freq); settle(15s)
    runs = []                         # each: (rate, stable, achieved_tps, p99_tpot)
    # ---- A. bracket ----
    r = clamp(seed_rate, min_rate, max_rate)
    res = measure(freq, r); runs.append(res)
    if res.stable:
        hi_stable = r
        while res.stable and r < max_rate and len(runs) < max_runs:
            r = min(max_rate, r * 1.5); res = measure(freq, r); runs.append(res)
            if res.stable: hi_stable = r
        lo_unstable = r if not res.stable else None      # None => max_rate still stable
    else:
        lo_unstable = r
        while (not res.stable) and r > min_rate and len(runs) < max_runs:
            r = max(min_rate, r / 1.5); res = measure(freq, r); runs.append(res)
            if not res.stable: lo_unstable = r
        hi_stable = r if res.stable else None            # None => min_rate still unstable
    # ---- safeguards ----
    if hi_stable is None:      return record(freq, runs, confirmed=False, flag="min_rate_unstable")
    if lo_unstable is None:    return record(freq, runs, confirmed=False, flag="max_rate_stable")  # lower bound
    # ---- B. refine (bisection) ----
    prev_cap = None
    while (lo_unstable - hi_stable)/hi_stable > 0.10 and len(runs) < max_runs:
        r = 0.5*(hi_stable + lo_unstable); res = measure(freq, r); runs.append(res)
        if res.stable: hi_stable = r
        else:          lo_unstable = r
        cap = max(x.achieved_tps for x in runs if x.stable)
        if prev_cap is not None and abs(cap-prev_cap)/prev_cap < 0.03: break   # capacity converged
        prev_cap = cap
    return record(freq, runs, confirmed=True)            # knee bracketed
```
`measure()` runs ONE `vllm bench serve` at that rate (Section 7 template) WHILE
scraping `/metrics`, then applies the Section-2 **steady-state backlog** stability
test (not the aggregate served_ratio).

Safeguards (also in manifest):
- `min_rate = 0.5`, `max_rate = 32`, `max_runs_per_freq = 12`.
- **max_rate still stable** -> capacity is a lower bound; `confirmed_knee=false`.
- **min_rate still unstable** -> record served@min_rate as a lower bound;
  `confirmed_knee=false`, flag `shape_infeasible_below_min` for review.
- **Seed:** first frequency uses the manifest `initial_rate`; every later
  frequency reuses the previous frequency's converged `hi_stable` scaled by the
  clock ratio (`seed = prev_hi_stable * freq/prev_freq`) — capacity is ~monotone
  in clock, so this lands near the knee and cuts runs.

---

## 4. Capacity output (per frequency)

`C_dc_stable = max(output_token_throughput_tps over STABLE runs)`.
For each SLO bucket s in {50,100,200,500} ms:
`C_dc_SLO(s) = max(output_token_throughput_tps over runs that are STABLE and p99_tpot<=s)`
(computed by `calibrate_decode_capacity.py` from the recorded sweep — one hardware
cell -> 4 SLO rows).

Recorded per (gpu,il,ol,tp,freq): `C_dc_stable`, `C_dc_SLO[s]`,
`highest_stable_rate`, `lowest_unstable_rate`, `served_ratio@selected_stable`,
`p99_tpot@selected_stable`, `n_runs`, `confirmed_knee`, `source="hardware_batch1"`,
run IDs / timestamps. Capacity is **single-instance** (one L4, TP=1).

---

## 5. Replication & noise

Existing Phase-2 used 3 repeated runs per fixed config, averaged. For the adaptive
batch we do NOT triple every probe (wasteful). Policy:
- Search probes: 1 run each (fast bracketing/refinement).
- **Confirmation:** re-run the final `highest_stable` point **x3** and the
  `lowest_unstable` point **x2**; if reps disagree on stability, add a **3rd** rep.
- **Aggregation (conservative):** a point counts as STABLE only if **all** reps are
  stable; `C_point = MIN(output_token_throughput_tps over stable reps)`. Never
  average a stable with an unstable outcome into a "stable" capacity. If the
  highest-stable point fails re-confirmation, fall back to the next-lower stable
  probe and re-confirm.

---

## 6. Run ordering (the 10 frequencies)

`2040 -> 1830 -> 1620 -> 1410 -> 1200 -> 990 -> 780 -> 570 -> 360 -> 210` (descending).
- Server stays up the whole campaign; clocks are set live (`nvidia-smi -ac`) with a
  15 s settle after each change — no per-frequency model reload.
- Descending avoids the flagged **strictly-increasing** order and runs the hottest
  (max-clock) cells first, while the GPU is coolest — so any thermal drift makes
  high-clock capacity, if anything, conservative, not inflated.
- Each frequency reuses the previous one's converged boundary (scaled by clock
  ratio) as its seed — capacity is ~monotone in clock, so this minimizes runs.
- Temperature is logged per run (`gpu_monitor.py`) for a post-hoc thermal-bias check;
  if a high-clock cell shows temperature-correlated capacity, re-run it in isolation.

---

## 7. One-frequency command template (pipeline-validation dry run)

Hardened per review: monitor backgrounded (`& MON=$!`), server readiness polled,
`SERVER_PID` saved, `trap cleanup EXIT` stops monitor+server and resets clocks even
on failure, `--max-concurrency 32` for the dry run (see §8 — do NOT trust 62 yet).
**This dry run validates the pipeline only; it draws NO stability conclusion.**

```bash
set -euo pipefail
cd /path/to/vLLM_test
GPU=0; MODEL="mistralai/Mistral-7B-v0.1"; PORT=8000; MEM=6251
IL=1024; OL=512; FREQ=2040; RATE=3.0
OUT="Phase2_Results_L4/batch1_il1024_ol512_tp1"; mkdir -p "$OUT"
LABEL="dryrun_f${FREQ}_r${RATE}"; MON=""; SERVER_PID=""
cleanup() {
  [[ -n "${MON:-}" ]]        && { kill -TERM "$MON" 2>/dev/null || true; wait "$MON" 2>/dev/null || true; }
  [[ -n "${SERVER_PID:-}" ]] && { kill -TERM "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
  sudo nvidia-smi -i "$GPU" -rac >/dev/null 2>&1 || true
}
trap cleanup EXIT
sudo nvidia-smi -pm 1
CUDA_VISIBLE_DEVICES="$GPU" python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --tensor-parallel-size 1 --max-model-len 9216 --port "$PORT" \
  > "$OUT/vllm_server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 180); do curl -fsS "http://localhost:${PORT}/v1/models" >/dev/null && break; sleep 2; done
curl -fsS "http://localhost:${PORT}/v1/models" >/dev/null || { echo "vLLM not ready" >&2; exit 1; }
sudo nvidia-smi -i "$GPU" -ac "${MEM},${FREQ}"; sleep 15
python3 gpu_monitor.py --monitor --interval 0.1 \
  --output "$OUT/monitor_${LABEL}_gpu0.csv" --gpu-id "$GPU" &
MON=$!
NUM_PROMPTS=$(python3 -c "print(min(1000, max(100, round(${RATE}*90))))")
vllm bench serve --backend openai --base-url "http://localhost:${PORT}" --model "$MODEL" \
  --num-prompts "$NUM_PROMPTS" --request-rate "$RATE" --dataset-name random \
  --random-input-len "$IL" --random-output-len "$OL" \
  --max-concurrency 32 --num-warmups 10 | tee "$OUT/bench_${LABEL}.txt"
```

For the real capacity runs (not this dry run) `measure()` additionally scrapes
`/metrics` (`vllm:num_requests_waiting/running`, `vllm:gpu_cache_usage_perc`) on a
~1 s interval into a CSV, so the §2 steady-state backlog test can be applied. The
adaptive driver (Section 3) is a thin loop over this template plus that scraper; it
is the only new code and does NOT modify the scheduler or the gate.

---

## 8. Pre-execution validation

- [x] All 10 frequencies are valid L4 clocks (`ALL_GPU_FREQS` L4 = the manifest list).
- [x] IL=1024/OL=512/TP=1 supported: 1024+512=1536 << `--max-model-len 9216`; TP=1 used in Phase-2 step2a.
- [ ] Memory: the `0.128 MB/tok x 1024 = 131 MB/req -> 62` estimate is NOT sufficient
      to mark safe — KV per sequence grows to IL+OL=1536 tokens during decode, and it
      ignores weights, CUDA graphs, allocator reservation, vLLM block fragmentation,
      and workspace. **Dry run uses `--max-concurrency 32`**; read vLLM's reported KV
      blocks / peak memory / any preemption-recompute, then set the real cap. The
      capacity campaign MUST use one **unified** concurrency policy across all runs
      (else runs are not comparable).
- [x] `output_token_throughput_tps` available and defined (generated tokens / duration).
- [x] No proxy IL/OL in the profiling or calibration path (profiler uses real
      `--random-input-len/--random-output-len`; scheduler proxy already removed).
- [x] Rows compatible with `calibrate_decode_capacity.py` (same master_results schema;
      it groups by (il,ol,tp,freq,rate), reads float rates, ignores `step`).
- [x] One hardware cell (gpu,il,ol,tp,freq) -> 4 SLO rows from the SAME sweep; not
      counted as 4 cells (calibrate emits per-SLO rows from one config).

---

## 9. Unresolved blockers

1. **[P0] Finite-batch drain breaks the aggregate stability signal.**
   `served_ratio ≈ 1/duration_ratio` (one signal, not two) and both are biased low
   by tail drain, so `served_ratio>=0.95` on whole-run throughput false-flags stable
   runs as overload. **Before any capacity search**, add a `/metrics` scraper
   (`vllm:num_requests_waiting/running`, `gpu_cache_usage_perc`) and switch the
   stability gate to the steady-state queue-slope test (§2). Options: (a) recommended
   — scrape `/metrics`; (b) fallback — greatly lengthen injection and measure
   throughput over the injection window only (never the whole benchmark_duration).
2. **[P0] calibrate_decode_capacity inherits the same drain bias.** It computes
   `served_ratio` from aggregate `output_token_throughput_tps`. Reconcile it with the
   steady-state definition (write a drain-corrected/steady-state throughput or a
   per-run `stable` flag into the results) so the batch's stable points are not
   re-rejected downstream. (Does not change the scheduler/gate.)
3. **[P1] max_concurrency policy undetermined.** The `62` estimate is unsafe (§8);
   set it from vLLM's reported KV blocks after the dry run, and use ONE unified value
   for all capacity runs.
4. **Adaptive driver not yet written.** The Section-3 loop + `/metrics` scraper must
   be scripted (reuses `vllm bench serve` + `set_gpu_freq` + `gpu_monitor.py`); ~1
   small script, no measurement-infra change.
5. **Cluster + sudo.** Runs on the SLURM L4 node; `nvidia-smi -ac/-pm` needs the
   sudoers whitelist. Cannot run from this workstation.
6. **Very-low-rate run duration.** If the knee sits below ~1 req/s, `num_prompts`
   floor (100) makes a run take up to ~10 min at OL=512; acceptable but bounds
   campaign throughput.
7. ~~`gpu_monitor.py` CLI flags~~ — RESOLVED: `--monitor --interval 0.1
   --output <file>_gpu0.csv --gpu-id 0`; template updated.
8. **Confirm `vllm bench serve` prints `Output token throughput (tok/s)`** in the
   installed vLLM version (parser keys on it) — verify in the dry run.
