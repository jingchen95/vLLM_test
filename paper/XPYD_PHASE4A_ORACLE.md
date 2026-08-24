# XpYd Phase 4A small empirical oracle

Phase 4A measures a small offline ground truth on the accepted persistent
2P2D substrate. P0/P1 are TP1 L40S endpoints on Neptune and D0/D1 are TP1 L4
endpoints on IO. It trains no model, invokes no feedback decision, changes no
endpoint lifecycle state, and does not run a dynamic workload trace.

## Measured action space

Phase 3D-A supplied the only allowed graphics-clock states:

- P0/P1 L40S: LOW 1260, MID 1890, HIGH 2520 MHz
- D0/D1 L4: LOW 750, MID 1125, HIGH 1500 MHz

The full route x per-endpoint-frequency Cartesian space has `4 * 3^4 = 324`
configurations. Phase 4A prunes this to 23 measured configurations:

- all four accepted P-to-D routes;
- active-pair profiles LL, MM, HH, HL, and LH on every route;
- canonical P0-to-D0 profiles ML and MH to expose MID P-side cross-points;
- one simple static baseline: P0-to-D0 with all four endpoints at HIGH.

Unselected endpoints remain active at LOW for the 22 oracle candidates. They
are never deactivated. The static baseline keeps every endpoint at HIGH. This
design supports matched routing, P-side DVFS, D-side DVFS, and combined
contrasts without claiming coverage of the omitted Cartesian points.

## Workloads and repetition

Four previously validated stationary shapes are measured: IL128/OL128,
IL2048/OL128, IL128/OL512, and IL2048/OL512. Every workload/configuration cell
uses two sequential requests and five independent repeats. Before measurement,
one excluded four-request round-robin warmup establishes all four NCCL
communicators so first-connection cost does not select an apparent winner.

The experiment contains 23 configurations x 4 workloads x 5 repeats = 460
measurement windows and 920 measured requests. Configuration and workload
orders are deterministically rotated between repeats to reduce order bias.

## Validity, SLO, and oracle definition

Every window reuses the Phase 3C request/route/token/SSE/clock/throttle audit
and four-endpoint NVML gross-energy windows. The reported energy is total gross
board energy across P0, P1, D0, and D1. No request-level prefill incremental
energy is inferred.

The selected SLO semantics match the functional controller validation:
per-repeat mean TTFT <= 1000 ms and per-repeat mean TPOT <= 80 ms. A
configuration is oracle-eligible only when all five repeats are valid and all
five pass both SLOs. For each workload, the empirical offline oracle is the
eligible configuration with minimum mean measured total four-GPU gross energy.
The second-best candidate, static baseline, sample standard deviation, 95%
small-sample confidence interval, SLO headroom, and number of candidates within
5% of the best are also reported.

This is the best configuration in the measured, pruned action space. It is not
a theoretical optimum and Phase 4A makes no novelty or online-performance
claim.

## Outputs

Compact outputs appear under `results/phase4a_empirical_oracle/<run-id>/`:

- `phase4a_measurements.csv`
- `phase4a_oracle.csv`
- `phase4a_summary.json`
- `phase4a_summary.md`

Raw per-window logs, samples, audits, metrics, client results, clock actions,
capabilities, and the exact measurement plan are retained below `raw/`.
