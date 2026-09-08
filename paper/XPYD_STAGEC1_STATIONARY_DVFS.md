# XPYD Stage C1: stationary feedback-only DVFS

Stage C1 compares `STATIC-HIGH` with `FEEDBACK-DVFS` under the calibrated
stationary mixed workload. Both policies use the same deterministic request
sequence, fixed four-pair round-robin routing, TP=1, and maximum client
concurrency 4. The evaluated loads are MEDIUM (0.10 req/s) and
HIGH-SUSTAINABLE (0.20 req/s); every 20-request window contains exactly 7 SS,
3 SL, 7 LS, and 3 LL requests.

## Controller audit and minimal C1 change

The existing Phase 3D controller already has the required primitives:

- downshift is one supported frequency level at a time and is allowed only
  with measured latency headroom plus low queue/KV pressure;
- with `latency_risk_to_max`, fresh TTFT/TPOT risk recovers directly to HIGH;
- missing or stale required telemetry fails closed at HIGH.

It did **not** already implement the complete C1 contract: recommendations and
actuation were endpoint-specific, and client admission/backlog was not a
fail-safe input. Stage C1 adds only a pool coordinator. P0/P1 or D0/D1 descend
only when both endpoints recommend the same one-step downshift. A risk from
either endpoint, or fresh client admission/backlog pressure, sends the whole
pool to HIGH. This fast recovery does not depend on queue telemetry. Sampled
P-side `running==0` is never interpreted as proof of idleness.

The controller SLO is aligned with evaluation: TTFT 600 ms and TPOT 65 ms.
Both require at least 95% request attainment. Energy savings are reportable
only when both gates pass.

Validated frequency levels are reused without adding points:

- P0/P1 L40S: 1260 / 1890 / 2520 MHz (LOW/MID/HIGH)
- D0/D1 L4: 750 / 1125 / 1500 MHz (LOW/MID/HIGH)

## Reproduction

The launcher uses the existing Stage-B persistent 2P2D lifecycle and Phase 3C
measurement/audit path. A smoke run is selected with
`XPYD_STAGEC1_SMOKE=1`; omit it for the configured two-repeat evaluation.

```bash
export XPYD_STAGEC1_RUNTIME_CONFIG=paper/configs/xpyd_stageb_runtime_neptune_europa.json
export XPYD_STAGEC1_EXPERIMENT_CONFIG=paper/configs/xpyd_stagec1_stationary_neptune_europa.json
export XPYD_STAGEC1_RUN_ID=stagec1_smoke_neptune_europa
export XPYD_STAGEC1_SMOKE=1
bash run_disagg_benchmark.sh
```

Outputs include per-window measurements, request rows, controller decisions,
clock actions/readbacks, telemetry observations, aggregate policy results, and
a hard-gate summary. The claim boundary is stationary feedback-only pool DVFS;
there is no request-aware routing, probing, predictor, model, or oracle.
