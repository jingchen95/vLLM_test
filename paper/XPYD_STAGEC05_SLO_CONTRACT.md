# XPYD Stage C0.5 TTFT SLO Contract Calibration

This is an analysis-only result from accepted Stage C0 job `257127`; no GPU job was run.
The population contains the four sustainable offered loads (0.05, 0.10, 0.15, and 0.20 req/s), 48 requests per load. A violation is `TTFT > SLO`.

## Short-input and long-input TTFT

| Load | Group | n | Mean | p50 | p90 | p95 | p99 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rate_005 | SHORT_INPUT | 24 | 231.4 | 197.5 | 226.2 | 468.7 | 669.0 | 716.4 |
| rate_005 | LONG_INPUT | 24 | 490.6 | 468.0 | 484.5 | 569.6 | 821.0 | 891.7 |
| rate_010 | SHORT_INPUT | 24 | 209.0 | 197.2 | 280.2 | 283.6 | 286.4 | 287.2 |
| rate_010 | LONG_INPUT | 24 | 503.8 | 469.9 | 564.2 | 569.5 | 574.9 | 576.4 |
| rate_015 | SHORT_INPUT | 24 | 225.5 | 199.2 | 278.3 | 282.8 | 285.4 | 285.9 |
| rate_015 | LONG_INPUT | 24 | 524.3 | 543.9 | 570.9 | 571.4 | 571.5 | 571.5 |
| rate_020 | SHORT_INPUT | 24 | 250.4 | 277.0 | 298.6 | 302.4 | 304.8 | 305.4 |
| rate_020 | LONG_INPUT | 24 | 527.5 | 526.5 | 567.1 | 568.4 | 570.1 | 570.6 |
| POOLED | SHORT_INPUT | 96 | 229.1 | 197.9 | 288.9 | 296.6 | 520.7 | 716.4 |
| POOLED | LONG_INPUT | 96 | 511.5 | 483.1 | 569.4 | 571.4 | 599.6 | 891.7 |

`SHORT_INPUT = SS + SL`; `LONG_INPUT = LS + LL`. All values are milliseconds.

## Per-class TTFT by load

| Load | Group | n | Mean | p50 | p90 | p95 | p99 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rate_005 | SS | 17 | 226.5 | 192.9 | 218.9 | 329.0 | 638.9 | 716.4 |
| rate_005 | SL | 7 | 243.3 | 200.9 | 331.6 | 421.0 | 492.5 | 510.4 |
| rate_005 | LS | 17 | 493.5 | 468.1 | 482.3 | 567.7 | 826.9 | 891.7 |
| rate_005 | LL | 7 | 483.4 | 467.8 | 519.4 | 551.8 | 577.8 | 584.2 |
| rate_010 | SS | 17 | 214.0 | 195.7 | 283.4 | 284.4 | 286.6 | 287.2 |
| rate_010 | SL | 7 | 196.9 | 197.6 | 200.9 | 202.3 | 203.5 | 203.8 |
| rate_010 | LS | 17 | 519.8 | 548.6 | 568.9 | 571.1 | 575.4 | 576.4 |
| rate_010 | LL | 7 | 464.9 | 463.8 | 469.7 | 469.7 | 469.8 | 469.8 |
| rate_015 | SS | 17 | 239.7 | 253.7 | 280.6 | 284.0 | 285.5 | 285.9 |
| rate_015 | SL | 7 | 191.1 | 195.2 | 196.9 | 197.1 | 197.3 | 197.3 |
| rate_015 | LS | 17 | 549.5 | 564.8 | 571.4 | 571.4 | 571.5 | 571.5 |
| rate_015 | LL | 7 | 463.0 | 464.2 | 465.5 | 466.5 | 467.2 | 467.4 |
| rate_020 | SS | 17 | 273.0 | 281.8 | 301.1 | 303.4 | 305.0 | 305.4 |
| rate_020 | SL | 7 | 195.3 | 195.7 | 198.3 | 199.2 | 200.0 | 200.2 |
| rate_020 | LS | 17 | 541.2 | 557.7 | 568.0 | 569.0 | 570.3 | 570.6 |
| rate_020 | LL | 7 | 494.1 | 467.5 | 539.1 | 539.6 | 540.0 | 540.1 |

## Per-class pooled TTFT

| Load | Group | n | Mean | p50 | p90 | p95 | p99 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| POOLED | SS | 68 | 238.3 | 202.8 | 292.3 | 298.4 | 441.0 | 716.4 |
| POOLED | SL | 28 | 206.7 | 196.6 | 202.7 | 209.4 | 430.0 | 510.4 |
| POOLED | LS | 68 | 526.0 | 529.9 | 569.7 | 571.4 | 680.5 | 891.7 |
| POOLED | LL | 28 | 476.3 | 464.2 | 529.1 | 539.5 | 572.3 | 584.2 |

## TTFT threshold sensitivity

| TTFT SLO (ms) | 0.05 | 0.10 | 0.15 | 0.20 | Pooled |
|---:|---:|---:|---:|---:|---:|
| 550 | 6.25% | 12.50% | 22.92% | 18.75% | 15.10% |
| 575 | 6.25% | 2.08% | 0.00% | 0.00% | 2.08% |
| 600 | 4.17% | 0.00% | 0.00% | 0.00% | 1.04% |
| 625 | 4.17% | 0.00% | 0.00% | 0.00% | 1.04% |
| 650 | 4.17% | 0.00% | 0.00% | 0.00% | 1.04% |

### Pooled sensitivity by input-length group

| TTFT SLO (ms) | Short-input | Long-input | All |
|---:|---:|---:|---:|
| 550 | 1.04% | 29.17% | 15.10% |
| 575 | 1.04% | 3.12% | 2.08% |
| 600 | 1.04% | 1.04% | 1.04% |
| 625 | 1.04% | 1.04% | 1.04% |
| 650 | 1.04% | 1.04% | 1.04% |

## Interpretation

- Pooled short-input p95 is 296.6 ms; pooled long-input p95 is 571.4 ms.
- At 550 ms, 29 of 192 requests violate (15.10%). At 575 ms, 4 violate (2.08%).
- The 600/625/650 ms rows quantify how much of the remaining tail is isolated rather than load-dependent.
- This sensitivity analysis does not itself select a new SLO. Any contract change must state whether rare fixed-HIGH tail events and workload-class conditioning are in scope.

Source: `results/stagec0_capacity/stagec0_neptune_europa_20260904/rate_*/requests.csv`.
