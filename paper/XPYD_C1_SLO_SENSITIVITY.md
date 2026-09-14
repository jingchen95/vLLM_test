# XPYD C1 SLO sensitivity study

**Scope:** analysis only; no GPU job was run and no controller/configuration was changed.
**Input:** completed Stage C1b headroom diagnosis, Slurm job `261064`,
`~/vLLM_test/results/stagec1_headroom/stagec1_headroom_neptune_europa_20260913/`.

## Purpose and method

The fixed-frequency headroom run measured five points on the validated 2P2D
system (P0/P1 L40S, D0/D1 L4, TP=1):

| Point ID | P pool | D pool |
|---|---:|---:|
| `HIGH_HIGH` | HIGH = 2520 MHz | HIGH = 1500 MHz |
| `P_HIGH_D_MID` | HIGH = 2520 MHz | MID = 1125 MHz |
| `P_HIGH_D_LOW` | HIGH = 2520 MHz | LOW = 750 MHz |
| `P_MID_D_HIGH` | MID = 1890 MHz | HIGH = 1500 MHz |
| `P_LOW_D_HIGH` | LOW = 1260 MHz | HIGH = 1500 MHz |

The request-level file contains two matched repeats per point and load, 20
requests per repeat (40 requests per point/load):
`headroom_requests.csv`.  For each candidate threshold, attainment is the
fraction of these requests with the measured value `<=` the threshold.  Since
the target is 95%, at least 38 of 40 pooled requests must pass.  No values were
imputed, clamped, or retuned.

For the boundary tables, the **raw boundary** is the 38th-smallest observed
request latency (the smallest inclusive threshold that gives 38/40).  The
**grid boundary** is the first value on a reporting grid: TTFT in 5 ms steps
and TPOT in 0.1 ms steps.  The coarse tables below show useful round thresholds
for interpretation.

## 1. Measured energy and baseline latency

Energy is the measured gross four-GPU window energy amortized over logical
requests, as reported by `headroom_summary.csv`; it is not request-level
physical attribution.

| Load | Point | J/request (mean ± repeat std) | Energy vs HIGH/HIGH | TTFT p95 (pooled) | TPOT p95 (pooled) |
|---|---|---:|---:|---:|---:|
| MEDIUM | HIGH/HIGH | 2590.64 ± 11.07 | 0.00% | 589.6 ms | 63.25 ms |
| MEDIUM | P_HIGH_D_MID | 2493.49 ± 2.11 | −3.74% | 547.0 ms | 65.41 ms |
| MEDIUM | P_HIGH_D_LOW | 2479.87 ± 1.35 | −4.28% | 576.2 ms | 69.93 ms |
| MEDIUM | P_MID_D_HIGH | 2324.35 ± 1.89 | **−10.28%** | 573.1 ms | 63.25 ms |
| MEDIUM | P_LOW_D_HIGH | 2281.45 ± 1.38 | −11.93% | 637.1 ms | 63.25 ms |
| HIGH-SUSTAINABLE | HIGH/HIGH | 1665.00 ± 0.40 | 0.00% | 576.2 ms | 63.23 ms |
| HIGH-SUSTAINABLE | P_HIGH_D_MID | 1586.03 ± 0.83 | −4.74% | 554.8 ms | 65.37 ms |
| HIGH-SUSTAINABLE | P_HIGH_D_LOW | 1576.50 ± 4.39 | −5.32% | 588.1 ms | 69.90 ms |
| HIGH-SUSTAINABLE | P_MID_D_HIGH | 1505.51 ± 0.005 | **−9.58%** | 586.0 ms | 63.23 ms |
| HIGH-SUSTAINABLE | P_LOW_D_HIGH | 1479.83 ± 0.47 | −11.12% | 642.9 ms | 63.23 ms |

The fixed-HIGH baseline's pooled 95th-order values are approximately **580.0
ms TTFT / 63.25 ms TPOT** at MEDIUM and **576.0 ms / 63.23 ms** at
HIGH-SUSTAINABLE.  The MEDIUM baseline has a long tail (one repeat is only
90% TTFT attainment and includes a 1568.6 ms request), so the 95% boundary is
not the same as a guarantee on p99.

## 2. TTFT threshold versus request attainment

Each cell is pooled request attainment for that load/point (40 requests).  The
columns include every fixed-frequency point; TPOT is not involved in this
table.

### MEDIUM (0.10 req/s)

| TTFT threshold | HIGH/HIGH | P_HIGH_D_MID | P_HIGH_D_LOW | P_MID_D_HIGH | P_LOW_D_HIGH |
|---:|---:|---:|---:|---:|---:|
| 550 ms | 77.5% | 95.0% | 80.0% | 80.0% | 80.0% |
| 575 ms | 90.0% | 95.0% | 87.5% | 95.0% | 80.0% |
| 600 ms | 95.0% | 100.0% | 100.0% | 100.0% | 80.0% |
| 625 ms | 95.0% | 100.0% | 100.0% | 100.0% | 85.0% |
| 650 ms | 95.0% | 100.0% | 100.0% | 100.0% | 100.0% |

### HIGH-SUSTAINABLE (0.20 req/s)

| TTFT threshold | HIGH/HIGH | P_HIGH_D_MID | P_HIGH_D_LOW | P_MID_D_HIGH | P_LOW_D_HIGH |
|---:|---:|---:|---:|---:|---:|
| 550 ms | 85.0% | 92.5% | 67.5% | 75.0% | 65.0% |
| 575 ms | 90.0% | 100.0% | 85.0% | 90.0% | 65.0% |
| 600 ms | 100.0% | 100.0% | 100.0% | 100.0% | 75.0% |
| 625 ms | 100.0% | 100.0% | 100.0% | 100.0% | 87.5% |
| 650 ms | 100.0% | 100.0% | 100.0% | 100.0% | 97.5% |

### TTFT 95% boundaries

| Point | MEDIUM raw / 5-ms grid | HIGH-SUSTAINABLE raw / 5-ms grid | Interpretation |
|---|---:|---:|---|
| HIGH/HIGH | 579.99 / **580 ms** | 576.03 / **580 ms** | Fixed-HIGH reference. |
| P_HIGH_D_MID | 545.01 / **550 ms** | 554.71 / **555 ms** | D frequency does not create a TTFT problem. |
| P_HIGH_D_LOW | 576.14 / **580 ms** | 587.98 / **590 ms** | TTFT still passes, but TPOT does not. |
| P_MID_D_HIGH | 572.39 / **575 ms** | 585.87 / **590 ms** | P=MID is safe at current 600 ms. |
| P_LOW_D_HIGH | 636.71 / **640 ms** | 642.69 / **645 ms** | Needs substantially more TTFT slack. |

The load effect is modest for P=LOW (about +6 ms at the raw boundary), but
P=MID has a higher boundary at HIGH-SUSTAINABLE than at MEDIUM (about +13.5
ms).  Both remain below the current 600 ms contract.

## 3. TPOT threshold versus request attainment

Each cell is pooled request attainment for that load/point.  TTFT is not
involved in this table.

### MEDIUM (0.10 req/s)

| TPOT threshold | HIGH/HIGH | P_HIGH_D_MID | P_HIGH_D_LOW | P_MID_D_HIGH | P_LOW_D_HIGH |
|---:|---:|---:|---:|---:|---:|
| 63.0 ms | 85.0% | 85.0% | 65.0% | 85.0% | 85.0% |
| 64.0 ms | 100.0% | 85.0% | 80.0% | 100.0% | 100.0% |
| 65.0 ms | 100.0% | 85.0% | 80.0% | 100.0% | 100.0% |
| 65.5 ms | 100.0% | 100.0% | 80.0% | 100.0% | 100.0% |
| 66.0 ms | 100.0% | 100.0% | 85.0% | 100.0% | 100.0% |
| 68.0 ms | 100.0% | 100.0% | 85.0% | 100.0% | 100.0% |
| 70.0 ms | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

### HIGH-SUSTAINABLE (0.20 req/s)

| TPOT threshold | HIGH/HIGH | P_HIGH_D_MID | P_HIGH_D_LOW | P_MID_D_HIGH | P_LOW_D_HIGH |
|---:|---:|---:|---:|---:|---:|
| 63.0 ms | 75.0% | 75.0% | 55.0% | 75.0% | 75.0% |
| 64.0 ms | 100.0% | 75.0% | 55.0% | 100.0% | 100.0% |
| 65.0 ms | 100.0% | 75.0% | 60.0% | 100.0% | 100.0% |
| 65.5 ms | 100.0% | 100.0% | 65.0% | 100.0% | 100.0% |
| 66.0 ms | 100.0% | 100.0% | 65.0% | 100.0% | 100.0% |
| 68.0 ms | 100.0% | 100.0% | 75.0% | 100.0% | 100.0% |
| 70.0 ms | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

### TPOT 95% boundaries

| Point | MEDIUM raw / 0.1-ms grid | HIGH-SUSTAINABLE raw / 0.1-ms grid | Interpretation |
|---|---:|---:|---|
| HIGH/HIGH | 63.247 / **63.3 ms** | 63.225 / **63.3 ms** | Fixed-HIGH reference. |
| P_HIGH_D_MID | 65.411 / **65.5 ms** | 65.368 / **65.4 ms** | First safe threshold is just above 65 ms. |
| P_HIGH_D_LOW | 69.929 / **70.0 ms** | 69.903 / **70.0 ms** | Needs about 5 ms extra TPOT slack. |
| P_MID_D_HIGH | 63.255 / **63.3 ms** | 63.231 / **63.3 ms** | P DVFS does not materially change TPOT. |
| P_LOW_D_HIGH | 63.247 / **63.3 ms** | 63.226 / **63.3 ms** | P DVFS does not materially change TPOT. |

The D-side boundary is almost load-invariant in this small sample, but the
attainment at the current 65 ms contract is not: D=MID is 85%/75% and D=LOW
is 80%/60% (MEDIUM/HIGH-SUSTAINABLE).  This is why the lower D points are not
safe under the current contract even though their mean throughput/backlog is
stable.

### Compact boundary plot (pooled attainment)

```text
TPOT threshold       65       65.5       66        68        70 ms
D=MID, MEDIUM       85%       100%      100%      100%      100%
D=MID, HIGH         75%       100%      100%      100%      100%
D=LOW, MEDIUM       80%        80%       85%       85%      100%
D=LOW, HIGH         60%        65%       65%       75%      100%

TTFT threshold      600       625       640       645       650 ms
P=MID, MEDIUM      100%      100%      100%      100%      100%
P=MID, HIGH        100%      100%      100%      100%      100%
P=LOW, MEDIUM       80%       85%      100%      100%      100%
P=LOW, HIGH         75%       88%       88%       98%      98%
```

## 4. Candidate SLO contracts

These are sensitivity-study labels, not a choice made to force a frequency
point to pass.  The reference is the measured fixed-HIGH distribution above;
each looser contract adds explicit latency slack.  An operational contract
must still come from the service requirement, not from the energy result.

| Candidate | TTFT | TPOT | What the measured data supports | Caveat |
|---|---:|---:|---|---|
| **TIGHT (current)** | <=600 ms | <=65 ms | HIGH/HIGH and P=MID,D=HIGH pass both loads; P=LOW and D<MID are not safe. | MEDIUM HIGH/HIGH is exactly 95% pooled and one repeat is 90%; retain repeats. |
| **MODERATE (sensitivity)** | <=625 ms | <=66 ms | P=MID,D=HIGH and D=MID points pass both loads; P=LOW and D=LOW still fail at least one dimension. | This is ~45 ms TTFT and ~2.75 ms TPOT above fixed-HIGH pooled p95; validate independently if adopted. |
| **LOOSE (sensitivity)** | <=650 ms | <=70 ms | P=LOW,D=HIGH and D=LOW reach >=95% pooled attainment in both loads (P=LOW/HIGH is 97.5% TTFT). | Only valid if an external product SLO accepts this slack; it is not a justification to relax the current contract. |

The current C1 contract should remain **600/65 ms, >=95% attainment** unless an
application-level requirement independently selects another row. In
particular, the 650/70 row is descriptive: it exposes the first round-number
region in which the low-frequency axes become feasible, but does not make them
acceptable for the current experiment.

## 5. Decision for full C1

1. **P-side:** the diagnosis supports an energy-saving HIGH -> MID opportunity
   at both measured loads (about 9.6--10.3% lower gross J/request).  P=LOW is
   not safe under 600 ms TTFT.
2. **D-side:** D=MID first reaches 95% only at approximately 65.4--65.5 ms;
   D=LOW only at approximately 70.0 ms.  Neither is safe at TPOT <=65 ms.
3. **Load dependence:** HIGH-SUSTAINABLE is somewhat worse for P=MID TTFT and
   current-contract D=MID/D=LOW attainment, but the 95% threshold boundaries
   are close to the MEDIUM boundaries.
4. **Controller implication:** do not change SLO thresholds or loosen the
   controller to manufacture savings.  For the current 600/65 C1 evaluation,
   the safe action space should be P HIGH/MID with D held HIGH.  This is an
   action-space guard based on measured feasibility, not a new controller
   policy.
5. **Readiness:** the analysis is sufficient to plan the full C1 comparison,
   but full C1 should not be launched with D downshifts enabled.  No GPU job is
   launched by this study.

## Reproducibility

The calculation uses only `headroom_requests.csv` and
`headroom_summary.csv` from the completed `261064` result directory.  The
original fixed-frequency hard-gate audits remain unchanged; this document
recomputes only the request-level threshold predicates.
