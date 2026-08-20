# XpYd Phase 3C: real multi-endpoint substrate validation

Phase 3C is infrastructure validation, not a routing study. It extends the
accepted Experiment E lifecycle to two persistent TP=1 prefill endpoints and
two persistent TP=1 decode endpoints while preserving real SSE, logical IDs,
vLLM metrics, and read-only NVML energy accounting.

The checked-in physical layout is:

| Endpoint | Node / GPU | Role | HTTP / KV port | Fixed clock |
|---|---|---|---|---:|
| P0 | Neptune / 0, L40S | prefill | 8100 / 14579 | 2520 MHz |
| P1 | Neptune / 1, L40S | prefill | 8101 / 14580 | 2520 MHz |
| D0 | Io / 0, L4 | decode | 8200 / 14579 | 1500 MHz |
| D1 | Io / 1, L4 | decode | 8201 / 14580 | 1500 MHz |

The proxy performs deterministic round-robin over the four exact pairs listed
in `compatible_pairs`. Exact-pair evidence is fail-closed: once a table has
endpoint-pair entries, a connector/TP match cannot make an unlisted pair
eligible. The current entries define the bounded validation envelope; a pair
is reported physically compatible only if its real request and KV handoff pass
the Phase 3C audit.

The smoke uses four IL128/OL128 and four IL2048/OL128 requests, interleaved so
each exact pair carries both shapes. It is intentionally not a rate, workload,
or routing sweep. Gross board energy covers the shared controlled serving
window; no request-level prefill incremental energy is inferred.

Submit from the repository root on Minerva:

```bash
sbatch --nodelist=neptune,io \
  --export=ALL,EXP=E,L40S_NODE=neptune,L4_NODE=io,XPYD_PHASE3C_CONFIG=paper/configs/xpyd_phase3c_2p2d_l40s_l4.json,RESULT_DIR=results/phase3c_2p2d_smoke \
  run_disagg_benchmark.sh
```

The run writes a dedicated directory below `results/xpyd_phase3c_substrate/`
with `summary.md`, `summary.json`, `requests.csv`, `routes.csv`,
`endpoint_summary.csv`, `energy_summary.csv`, `audit.json`, raw server/proxy
logs, raw Prometheus scrapes, and per-endpoint NVML samples.

Hard acceptance requires all four endpoints and all four pairs to be exercised,
100% real SSE and ID propagation, exact output tokens, Phase 3A token semantics,
valid per-board energy windows, fixed-clock matches, no resource overlap, and
no thermal/hardware slowdown. There is no model training, adaptive routing, or
DVFS actuation in this phase.
