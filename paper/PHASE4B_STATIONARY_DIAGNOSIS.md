# Phase 4B stationary passive-feedback diagnosis

## Scope and evidence

This is an analysis-only diagnosis. It changes no scheduler, controller,
routing, DVFS, or measurement code and uses no new GPU experiment. The fixed
offline reference is the accepted Phase 4A run
`phase4a_oracle_r1_20260824T080026Z`; the evaluated stationary run is
`phase4b_stationary_r1_20260824T234000Z` (Slurm job 255011). All 80 Phase 4B
windows and all 160 requests were valid. The later job 255047 is only a
successful six-window dynamic-path smoke and is not evidence in this
stationary diagnosis.

SLOs are TTFT <= 1000 ms and TPOT <= 80 ms. Energy is total gross energy of
P0, P1, D0, and D1. Values below are means over five repeats; `+/-` is sample
standard deviation. A machine-readable version is in
`paper/phase4b_stationary_diagnosis.csv`.

## Primary result

`vs static` is the percentage change in J/request, so negative values are
savings. The oracle is the best valid configuration in the fixed Phase 4A
measured action space, not a theoretical optimum.

| Workload | Policy | J/request +/- std | vs static | Oracle gap | TTFT / TPOT (ms) | TTFT/TPOT violations | Throughput (req/s) | Valid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| small | STATIC | 1799.30 +/- 3.11 | 0.00% | 20.37% | 205.35 / 55.94 | 0 / 0 | 0.13679 | 5/5 |
| small | ROUTING-ONLY | 1781.45 +/- 8.38 | -0.99% | 19.18% | 184.96 / 55.95 | 0 / 0 | 0.13716 | 5/5 |
| small | DVFS-ONLY | 1676.40 +/- 10.79 | -6.83% | 12.15% | 185.63 / 55.95 | 0 / 0 | 0.13715 | 5/5 |
| small | FULL | 1554.92 +/- 16.58 | -13.58% | 4.02% | 189.43 / 55.95 | 0 / 0 | 0.13708 | 5/5 |
| small | ORACLE | 1494.81 +/- 2.03 | -16.92% | 0.00% | 195.88 / 57.18 | 0 / 0 | 0.13408 | 5/5 |
| prefill-heavy | STATIC | 1949.61 +/- 8.61 | 0.00% | 17.32% | 480.43 / 58.08 | 0 / 0 | 0.12729 | 5/5 |
| prefill-heavy | ROUTING-ONLY | 1951.59 +/- 10.56 | +0.10% | 17.44% | 475.01 / 58.09 | 0 / 0 | 0.12735 | 5/5 |
| prefill-heavy | DVFS-ONLY | 1850.55 +/- 4.86 | -5.08% | 11.36% | 538.24 / 58.08 | 0 / 0 | 0.12634 | 5/5 |
| prefill-heavy | FULL | 1718.16 +/- 14.89 | -11.87% | 3.39% | 531.51 / 58.07 | 0 / 0 | 0.12647 | 5/5 |
| prefill-heavy | ORACLE | 1661.75 +/- 5.73 | -14.76% | 0.00% | 481.21 / 59.20 | 0 / 0 | 0.12500 | 5/5 |
| decode-heavy | STATIC | 7145.63 +/- 8.61 | 0.00% | 18.75% | 189.66 / 56.17 | 0 / 0 | 0.03461 | 5/5 |
| decode-heavy | ROUTING-ONLY | 7176.38 +/- 14.61 | +0.43% | 19.26% | 186.11 / 56.17 | 0 / 0 | 0.03461 | 5/5 |
| decode-heavy | DVFS-ONLY | 6776.05 +/- 19.83 | -5.17% | 12.61% | 188.84 / 56.17 | 0 / 0 | 0.03461 | 5/5 |
| decode-heavy | FULL | 6285.94 +/- 53.12 | -12.03% | 4.46% | 194.37 / 56.17 | 0 / 0 | 0.03461 | 5/5 |
| decode-heavy | ORACLE | 6017.33 +/- 7.15 | -15.79% | 0.00% | 195.06 / 57.64 | 0 / 0 | 0.03373 | 5/5 |
| both-heavy | STATIC | 7521.10 +/- 16.43 | 0.00% | 17.36% | 477.57 / 58.29 | 0 / 0 | 0.03304 | 5/5 |
| both-heavy | ROUTING-ONLY | 7546.88 +/- 11.00 | +0.34% | 17.76% | 470.40 / 58.28 | 0 / 0 | 0.03305 | 5/5 |
| both-heavy | DVFS-ONLY | 7192.60 +/- 16.86 | -4.37% | 12.23% | 539.23 / 58.27 | 0 / 0 | 0.03298 | 5/5 |
| both-heavy | FULL | 6638.07 +/- 53.40 | -11.74% | 3.58% | 541.36 / 58.28 | 0 / 0 | 0.03298 | 5/5 |
| both-heavy | ORACLE | 6408.84 +/- 8.18 | -14.79% | 0.00% | 483.02 / 59.47 | 0 / 0 | 0.03239 | 5/5 |

Across workloads, STATIC is 17.32--20.37% above the oracle (mean 18.45%).
The oracle saves 14.76--16.92% relative to the Phase 4B STATIC measurements
(mean 15.57%). All policies preserve both SLOs in this stationary regime.

## What produced FULL's saving

| Contrast against STATIC | small | prefill | decode | both | Mean |
|---|---:|---:|---:|---:|---:|
| ROUTING-ONLY saving | +0.99% | -0.10% | -0.43% | -0.34% | +0.03% |
| DVFS-ONLY saving | +6.83% | +5.08% | +5.17% | +4.37% | +5.36% |
| FULL saving | +13.58% | +11.87% | +12.03% | +11.74% | +12.31% |
| FULL additional saving over DVFS-ONLY | +7.25% | +7.15% | +7.23% | +7.71% | +7.34% |

The direct routing ablation is essentially zero and changes sign across
workloads. It does not support attributing FULL's saving to route choice.

The action logs give the mechanism:

- DVFS-ONLY fixes P0->D0. P0 goes MID in repeat 1 and LOW in repeats 2--5;
  P1 remains HIGH. D0 and D1 remain HIGH.
- FULL first observes all endpoints in its unmeasured warmup. Both P0 and P1
  go MID in repeat 1 and LOW in repeats 2--5, even though only one P endpoint
  receives measured requests. D0 and D1 remain HIGH.
- FULL therefore reduces the gross board energy of the otherwise unused
  second L40S. This explains the large FULL-minus-DVFS-ONLY contrast. It is a
  P-side DVFS coverage/joint-orchestration effect, not evidence that selecting
  P0->D0 or P0->D1 saved that energy.
- There were four successful controller DVFS actions per FULL workload, two
  per DVFS-ONLY workload, and no D-side frequency action.

A strictly identified four-way causal decomposition is not possible with
these four policies because there is no ablation that lowers both P GPUs while
holding routing fixed. The defensible attribution is: direct routing about
0%; observed single-P DVFS about 5.36%; an additional about 7.34% associated
with lowering the second P GPU in FULL; observed D-side DVFS 0%. This is
consistent with Phase 4A's ordering (P-side 7.49%, combined DVFS 8.48%, D-side
1.49%, routing 1.14%), while the precise percentages are not directly
comparable because the policy trajectories differ.

## Routing timeline and lock-in

Each routing policy began with a four-route observation warmup. The one
reported `routing_change` per block is the transition from that all-pair
warmup to a single route; there were no changes between the five measured
windows.

| Policy | Workload | Measured route sequence | Request fraction | Alternative endpoints last served / energy EWMA updated | First latency-stale repeat (age) | Lock-in |
|---|---|---|---|---|---|---|
| ROUTING-ONLY | small | P0->D1 x5 | P0->D1 100% | warmup only: P1, D0 | r4 (165 s) | yes |
| ROUTING-ONLY | prefill | P0->D1 x5 | P0->D1 100% | warmup only: P1, D0 | r3 (127 s) | yes |
| ROUTING-ONLY | decode | P0->D1 x5 | P0->D1 100% | warmup only: P1, D0 | r3 (203 s) | yes |
| ROUTING-ONLY | both | P0->D1 x5 | P0->D1 100% | warmup only: P1, D0 | r3 (209 s) | yes |
| FULL | small | P0->D0 x5 | P0->D0 100% | warmup only: P1, D1 | r4 (171 s) | yes |
| FULL | prefill | P0->D1 x5 | P0->D1 100% | warmup only: P1, D0 | r4 (173 s) | yes |
| FULL | decode | P0->D0 x5 | P0->D0 100% | warmup only: P1, D1 | r3 (203 s) | yes |
| FULL | both | P0->D1 x5 | P0->D1 100% | warmup only: P1, D0 | r3 (215 s) | yes |

Before expiry, all four routes were eligible. At the listed repeat the logs
reject the three alternatives as `P1_telemetry_stale` and/or
`D0_telemetry_stale`/`D1_telemetry_stale`; only the incumbent remains
eligible. This exactly realizes the proposed starvation chain:

1. the warmup score selects one route;
2. only its P and D endpoints receive measured requests;
3. only those endpoints obtain new latency and energy-per-request observations;
4. alternative role-latency observations exceed the 120 s limit;
5. all alternative pairs become ineligible;
6. the incumbent remains selected without a counterfactual measurement.

The word `telemetry` needs precision here. Queue/KV scrapes can continue for
unused endpoints, and endpoint samples continue to carry board energy. The
route rejection is driven specifically by missing/stale TTFT for an unused P
or TPOT for an unused D. Energy EWMA has no independent freshness timestamp;
when endpoint request count is zero, its old energy-per-request value is
retained rather than updated or explicitly expired.

The first-window score was also sensitive to warmup variation. ROUTING-ONLY
selected P0->D1 for all workloads. FULL selected P0->D0 for small/decode and
P0->D1 for prefill/both. Thus two nominally routing-enabled policies selected
different routes for the same workload without any workload-specific prior;
the observed differences came from their separate warmup measurements.

## Actual routing-energy score

For endpoint `e` and observed window `t`, the implementation computes

```text
x_e,t = gross_window_board_energy_e,t / requests_routed_through_e,t
S_e,t = 0.5 * x_e,t + 0.5 * S_e,t-1
Score(P_i -> D_j) = S_P_i,t + S_D_j,t
```

If either endpoint lacks an energy EWMA, the route has no energy score and is
ordered deterministically after scored candidates. Hard compatibility,
frequency, health, latency, queue, and KV gates are applied before energy
ranking. KV-transfer energy is explicitly excluded.

This score has seven limitations relevant to the result:

1. It is endpoint/window-amortized, not request-level attribution.
2. It uses gross board energy, not incremental energy above idle.
3. It is additive by endpoint and therefore not pair-specific.
4. It omits the gross energy of the two unselected but still powered GPUs,
   whereas the evaluation objective includes them.
5. It ranks J/request, not J/output-token or another workload-normalized cost.
6. `workload_context_key` is recorded with candidates but does not index a
   separate energy state; the EWMA itself is not workload-specific.
7. It has no counterfactual estimate for a route not recently exercised.

The stationary harness creates a fresh controller and same-workload warmup
for every policy/workload block. Therefore cross-workload EWMA contamination
did **not** cause the stationary outcomes. That limitation matters for a
persistent dynamic trace, but it is not an explanation for job 255011.

## Alignment with total-system energy

The scientific objective is

```text
E_total = E_P0 + E_P1 + E_D0 + E_D1.
```

The route score contains only the selected P and D endpoint estimates. It is
therefore not mathematically aligned with the four-board objective. The
stationary evidence is consistent with that mismatch:

- ROUTING-ONLY's minimum warmup score selected P0->D1 for all four workloads.
  Relative to STATIC total energy this agreed only for small (-0.99%); it was
  slightly worse for prefill (+0.10%), decode (+0.43%), and both (+0.34%).
- Those contrasts compare a single route with round-robin rather than a clean
  same-window route counterfactual, so they demonstrate lack of consistent
  alignment, not a causal per-request route penalty.
- Phase 4A's all-LOW route spread was only about 0.15--0.62% across the four
  routes, reinforcing that route identity is a weak energy knob here.
- FULL is near the oracle while choosing a non-oracle route in three of four
  workloads. Its low energy follows the two-P DVFS state; the data cannot
  credit the route score.

Accordingly: one weak agreement case, three weak disagreement cases, and
insufficient evidence for route-level causal effects. The implementation and
the total-energy objective are structurally mismatched even where their
rankings happen to agree.

## Why FULL remains above the oracle

| Workload | FULL gap | Stable FULL state | Phase 4A oracle | Evidence-backed diagnosis |
|---|---:|---|---|---|
| small | 4.02% | P0->D0; P0/P1 LOW; D0/D1 HIGH | P0->D0; all LOW | P converged; D never stepped down |
| prefill | 3.39% | P0->D1; P0/P1 LOW; D0/D1 HIGH | P1->D0; selected P/D MID, others LOW | D conservatism plus route/config mismatch; broad basin makes route mismatch small |
| decode | 4.46% | P0->D0; P0/P1 LOW; D0/D1 HIGH | P1->D1; all LOW | P converged; D never stepped down; route mismatch is small in Phase 4A |
| both | 3.58% | P0->D1; P0/P1 LOW; D0/D1 HIGH | P0->D0; selected P/D MID, others LOW | D conservatism plus route/config mismatch; broad basin makes route mismatch small |

The strongest evidenced residual cause is D-side safety conservatism. Observed
TPOT is about 56--58 ms. The controller steps down only below
`0.5 * 80 = 40 ms`, so both D GPUs remain HIGH despite Phase 4A showing valid
LOW/MID D configurations. This is a configured conservative rule, not failed
actuation. P-side convergence worked: both P GPUs reached LOW after the first
measured FULL window.

Route lock-in and energy-score mismatch are real, but the current data do not
show that they account for most of FULL's remaining 3.39--4.46% gap. Phase 4A
showed 13--14 configurations within 5% and first/second gaps of only
0.002--0.36%, so exact route mismatch is usually energetically minor.
No stationary evidence supports workload-context contamination. No SLO,
queue/KV fallback, failed actuation, or thermal/HW slowdown explains the gap.
The repeat standard deviations are smaller than the persistent mean gaps, so
noise contributes uncertainty but does not erase the systematic residual.

## Near-optimal basin

The runtime configurations are not exact members of the pruned Phase 4A
action space: for example, FULL keeps both D GPUs HIGH, while the Phase 4A
single-route profiles generally lowered the unused D. Therefore an exact
configuration-membership claim is invalid. The recorded exact-match fraction
is 0/5 for FULL and DVFS-ONLY and must not be interpreted as 0% energy
near-optimality.

Measured per-window oracle-gap regions are still comparable:

| Policy | Workload | <=5% windows | 5--10% | >10% |
|---|---|---:|---:|---:|
| DVFS-ONLY | each of four workloads | 0/5 | 0/5 | 5/5 |
| FULL | small | 4/5 | 1/5 | 0/5 |
| FULL | prefill | 5/5 | 0/5 | 0/5 |
| FULL | decode | 4/5 | 1/5 | 0/5 |
| FULL | both | 4/5 | 1/5 | 0/5 |

For small, decode, and both, the first MID-frequency FULL window is in the
5--10% region (5.98%, 6.03%, and 5.03%); the later LOW-P windows are <=5%.
Prefill is <=5% from the first window. This shows fast P-side DVFS convergence
into a broad near-oracle energy region, despite route mismatch.

## Classification and hypothesis test

Stationary classification: **B. PASSIVE FEEDBACK SUFFICIENT FOR DVFS BUT NOT
ROUTING.**

The qualification is important: DVFS-ONLY lowers only the exercised P and is
still 11.36--12.61% above oracle. FULL's all-endpoint warmup lets passive DVFS
lower both P GPUs and reaches 3.39--4.46% gaps with zero SLO violations. Thus
the local sequential DVFS signal is effective when the endpoint has recent
observations. Routing itself shows almost no saving and loses counterfactuals.

Hypothesis result: **SUPPORTED.** P frequency actions are evaluated
sequentially on observed endpoints and converge. Routing selects one warmup
winner, provides no further observations for alternatives, and makes those
alternatives ineligible after 127--215 seconds. This is direct evidence for
information starvation, not merely a conceptual concern.

## Decision

Choose **Direction 2**: DVFS feedback is good, but routing suffers information
starvation. Before the main dynamic evaluation, design (do not yet implement)
the smallest mechanism containing:

1. safe periodic probing of currently compatible/SLO-safe alternatives;
2. workload-context-specific observed route cost;
3. separate freshness for latency and energy evidence;
4. freshness-aware exploration followed by exploitation of recently measured
   near-optimal routes;
5. a score aligned with the declared four-GPU total-energy objective, or an
   explicitly narrowed objective if only active-route energy is intended.

This mechanism addresses the observed failure exactly: after the initial
choice, alternatives receive no requests, their TTFT/TPOT observations expire,
and passive routing cannot discover whether another route is better. No bandit,
predictor, or model should be selected until this minimal design and its claim
boundary are specified.

## Direct answers

1. STATIC is 17.32--20.37% above oracle; oracle saves 14.76--16.92% versus the
   measured STATIC baseline.
2. ROUTING-ONLY saves -0.43% to +0.99%, averaging only +0.03%.
3. DVFS-ONLY saves 4.37--6.83%, averaging 5.36%.
4. FULL saves 11.74--13.58%, averaging 12.31%.
5. FULL's oracle gap is 3.39--4.46%, averaging 3.86%.
6. FULL's benefit is mainly P-side DVFS, especially lowering both P boards;
   direct routing benefit is unsupported.
7. Route lock-in occurred for every workload under both routing-enabled
   policies.
8. The endpoint-pair score is not aligned with total four-board energy.
9. The hypothesis that passive feedback works better for DVFS than routing is
   supported.
10. Proceed with Direction 2's minimal safe, context-aware probing design;
    do not run the main dynamic comparison with the current routing claim.
