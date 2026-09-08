# XPYD Stage C0 Capacity/Load Calibration Results

Job `257127` completed successfully on Neptune (P0/P1 L40S) and Europa
(D0/D1 L4) in 47 minutes 13 seconds. All five Phase 3C measurement audits
passed at fixed validated HIGH frequencies with four-pair round-robin routing,
max concurrency four, and no controller or adaptive policy.

## Capacity result

| Offered req/s | Completed req/s | Admission p95 / p99 | Mean / max client waiting | Mean / max proxy inflight | Backlog | Capacity class |
|---:|---:|---:|---:|---:|---|---|
| 0.05 | 0.051 | 0.05 / 0.08 ms | 0.00 / 1 | 0.70 / 2 | drains | LOW |
| 0.10 | 0.100 | 0.06 / 0.08 ms | 0.00 / 1 | 1.43 / 3 | drains | MEDIUM |
| 0.15 | 0.145 | 0.10 / 0.13 ms | 0.00 / 1 | 2.12 / 4 | drains | MEDIUM |
| 0.20 | 0.188 | 0.04 / 0.07 ms | 0.00 / 1 | 2.84 / 4 | drains | HIGH-SUSTAINABLE |
| 0.30 | 0.239 | 16920 / 19369 ms | 2.07 / 6 | 3.80 / 4 | grows | OVERLOAD |

The exact selected rates are:

- LOW: **0.05 req/s**
- MEDIUM: **0.10 req/s** (`0.15 req/s` is also a measured stable medium point)
- HIGH-SUSTAINABLE: **0.20 req/s**

At 0.30 req/s, admission-delay slope was 116.9 ms per elapsed second,
late-quartile mean admission delay reached 15.5 seconds, and five requests
were still waiting when offered arrivals ended. Therefore it is overload even
though endpoint waiting metrics remained zero and all requests eventually
drained.

## Latency and SLO gate

| Offered req/s | TTFT mean / p95 / p99 (ms) | TPOT mean / p95 / p99 (ms) | 550/65 violation rate |
|---:|---:|---:|---:|
| 0.05 | 361.0 / 558.4 / 809.3 | 57.1 / 58.3 / 58.3 | 6.2% |
| 0.10 | 356.4 / 563.5 / 573.3 | 58.5 / 63.2 / 63.2 | 12.5% |
| 0.15 | 374.9 / 570.8 / 571.5 | 59.1 / 63.1 / 63.1 | 22.9% |
| 0.20 | 388.9 / 567.0 / 569.6 | 59.8 / 63.2 / 63.2 | 18.8% |
| 0.30 | 404.8 / 565.6 / 570.1 | 60.5 / 63.1 / 63.2 | 16.7% |

No request violated the 65 ms TPOT SLO. Nearly all TTFT violations were
long-input (LS) requests, and TTFT p95 stayed near 558--571 ms across the
entire load sweep. This is a fixed-HIGH baseline/SLO mismatch rather than a
queue-driven latency collapse. At 0.05 req/s there were also isolated tail
outliers up to 891.7 ms.

Capacity classification is therefore valid, but Stage C1 is **NOT READY**
under the present rule requiring at most 5% violations of a 550 ms TTFT SLO:
the fixed-HIGH baseline already fails that gate. Before enabling feedback
DVFS, the experiment must explicitly resolve the SLO contract (for example,
retain 550 ms and accept that the present substrate cannot meet it, or justify
a workload-aware/empirically supportable TTFT boundary). This decision should
be made without rerunning capacity calibration.

## Gross board energy

| Offered req/s | P0 / P1 (kJ) | D0 / D1 (kJ) | Total (kJ) | J/request |
|---:|---:|---:|---:|---:|
| 0.05 | 72.34 / 76.69 | 37.50 / 37.77 | 224.29 | 4672.7 |
| 0.10 | 37.04 / 39.27 | 24.24 / 24.64 | 125.19 | 2608.1 |
| 0.15 | 25.73 / 27.21 | 19.93 / 20.18 | 93.05 | 1938.6 |
| 0.20 | 20.05 / 21.18 | 16.54 / 17.00 | 74.77 | 1557.7 |
| 0.30 | 15.89 / 16.74 | 13.03 / 13.52 | 59.18 | 1232.9 |

These are gross finite-window board energies for the same 48-request
population. Their decline with offered rate primarily reflects shorter elapsed
windows and less accumulated baseline power; it is not a controller-energy
result.

## Telemetry boundary

D0/D1 mean running counts rose from about 0.34 each at 0.05 req/s to 1.68 and
1.78 at 0.30 req/s. Endpoint waiting remained zero. P0/P1 running gauges were
zero in the one-second samples because the short prefill intervals were not
captured; request attribution, prompt-token deltas, latency, fixed-clock,
thermal, and energy audits nevertheless passed. P-side running gauges should
not be used alone as a capacity signal at this scrape period.
