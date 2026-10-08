# XPYD Step 1 — Idle-floor vs load-dependent energy decomposition

**Status:** complete (offline, read-only; no GPU job)  
**Date:** 2026-10-08  
**Script:** `paper/scripts/xpyd/step1_energy_decomposition.py`  
**Outputs:** `paper/step1_energy_decomposition/{idle_power_by_clock,policy_decomposition,window_decomposition}.csv`

## Question

Before any further controller work, test the premise that the measured XPYD
energy, and the reported DVFS savings, are dominated by the idle power of
clock-locked GPUs rather than by the energy of serving work.

Pre-registered gate (set before running the analysis):

1. idle floor >= ~70% of total gross energy, and
2. >= ~80% of P-side savings explained by idle-floor differences.

If both hold, proceed to the E0 diagnostic job (default-governor baseline,
idle ladder, uncapped capacity, aggregated baseline). Otherwise revisit the
diagnosis.

## Inputs

Raw 0.2 s NVML samples (`samples.jsonl`: energy counter, power, graphics
clock, utilization) and proxy route timestamps (`routes.csv`) from every
accounting window of:

| Stage | Run | Windows |
|---|---|---:|
| C0 | `stagec0_neptune_europa_20260904` (job 257127) | 5 |
| C1b | `stagec1_headroom_neptune_europa_20260913` | 20 |
| C1c | `stagec1_full_neptune_europa_20260928` (job 269768) | 48 |
| C2a | `stagec2a_full_neptune_europa_20261003` (job 271338) | 72 |
| D1 | `stage_d1_full_212c925` (job 272628) | 120 |

## Method

- **Busy interval.** P endpoint: route timestamp to KV-handoff completion.
  D endpoint: route timestamp to decode completion (conservative: KV receive
  starts at routing).
- **Idle interval.** A consecutive-sample interval with unchanged graphics
  clock and no busy interval of that endpoint within ±1 s. Pre/post-window
  samples are included, so every locked clock has thousands of idle seconds.
- **Idle power** `P_idle(endpoint, clock)` = counter energy / duration over all
  idle intervals, pooled across runs.
- **Floor energy** of a window = Σ `P_idle(endpoint, clock(t)) · dt` over the
  accounting boundary. **Load-dependent energy** = gross − floor.

**Validation.** Counter-integrated gross energy reproduces every reported
aggregate (e.g. C1c STATIC MEDIUM 628.1 kJ, D1 STATIC MEDIUM 2616 J/request).
**Robustness.** Guard 0.5 s / 1 s / 2 s changes every idle-power estimate by
<= 0.1 W. Idle GPU utilization is 0.0% in every idle class.

## 1. Locked-clock idle power (model loaded, vLLM running, no request)

| Endpoint | Clock (MHz) | Idle power (W) | Idle seconds | Per-window range (W) |
|---|---:|---:|---:|---|
| P0 L40S | 2520 | 76.3 | 26,195 | 75.0–77.0 |
| P0 L40S | 1890 | 64.4 | 15,978 | 63.5–64.8 |
| P0 L40S | 1260 | 61.9 | 5,412 | 61.2–62.3 |
| P1 L40S | 2520 | 79.9 | 32,194 | 79.3–80.7 |
| P1 L40S | 1890 | 64.8 | 11,577 | 64.5–65.2 |
| P1 L40S | 1260 | 62.5 | 3,643 | 62.2–62.7 |
| D0 L4 | 1500 | 24.7 | 16,184 | 23.8–25.6 |
| D0 L4 | 1125 | 22.2 | 4,239 | 21.9–22.9 |
| D0 L4 | 750 | 21.3 | 549 | 21.2–21.4 |
| D1 L4 | 1500 | 24.8 | 17,839 | 23.6–25.9 |
| D1 L4 | 1125 | 22.4 | 1,759 | 22.0–22.9 |
| D1 L4 | 750 | 21.6 | 313 | 21.5–21.7 |

HIGH→MID removes 12–15 W per idle L40S; MID→LOW removes only ~2.5 W. No
locked clock brings an idle L40S below ~62 W. Default-governor (unlocked) idle
power cannot be measured from these data because every run locked clocks.

## 2. Decomposition per policy

Floor share = floor / gross. Busy = fraction of the accounting window with at
least one request in flight on that endpoint (mean over endpoints).

| Stage | Load | Policy | J/req | Floor share (total / P / D) | P busy | D busy | P load-dep. J/req | D load-dep. J/req |
|---|---|---|---:|---|---:|---:|---:|---:|
| C0 | 0.05 | STATIC-HIGH | 4673 | 86.9% / 99.4% / 62.4% | 0.7% | 35.5% | — | — |
| C0 | 0.10 | STATIC-HIGH | 2608 | 79.0% / 98.5% / 48.7% | 1.2% | 59.4% | — | — |
| C0 | 0.20 | STATIC-HIGH | 1558 | 70.3% / 96.9% / 37.7% | 2.3% | 92.0% | — | — |
| C0 | 0.30 | STATIC-HIGH | 1233 | 69.7% / 96.0% / 37.4% | 2.9% | 95.4% | — | — |
| C1c | MEDIUM | STATIC-HIGH | 2617 | 78.9% / 98.0% / 48.8% | 1.2% | 59.9% | 31.4 | 520.7 |
| C1c | MEDIUM | FEEDBACK | 2371 | 77.5% / 98.0% / 49.0% | 1.2% | 60.1% | 27.3 | 505.3 |
| C1c | HIGH | STATIC-HIGH | 1682 | 70.8% / 96.8% / 38.4% | 2.0% | 90.3% | 30.0 | 460.5 |
| C1c | HIGH | FEEDBACK | 1533 | 69.3% / 96.8% / 38.7% | 2.1% | 90.4% | 26.3 | 444.2 |
| C2a | MEDIUM | FAST | 2401 | 77.2% / 98.0% / 48.7% | 1.2% | 59.9% | 28.2 | 518.9 |
| C2a | HIGH | FAST | 1556 | 68.8% / 96.6% / 38.4% | 2.1% | 90.3% | 28.0 | 456.6 |
| D1 | MEDIUM | STATIC-HIGH-RR | 2616 | 78.9% / 98.2% / 48.7% | 1.2% | 59.9% | 28.0 | 522.9 |
| D1 | MEDIUM | SHARED | 2389 | 77.1% / 98.0% / 48.7% | 1.2% | 60.0% | 27.8 | 519.6 |
| D1 | MEDIUM | PER-ENDPOINT | 2483 | 78.7% / 98.3% / 48.9% | 1.2% | 60.1% | 25.9 | 504.0 |
| D1 | MEDIUM | JOINT | 2557 | 77.7% / 98.2% / 47.1% | 1.2% | 63.7% | 27.7 | 543.7 |
| D1 | HIGH | STATIC-HIGH-RR | 1681 | 70.9% / 96.9% / 38.4% | 2.1% | 90.3% | 29.4 | 460.1 |
| D1 | HIGH | SHARED | 1547 | 68.8% / 96.8% / 38.4% | 2.2% | 90.4% | 25.5 | 457.5 |
| D1 | HIGH | PER-ENDPOINT | 1598 | 70.6% / 97.0% / 38.6% | 2.1% | 90.5% | 26.6 | 443.3 |
| D1 | HIGH | JOINT | 1626 | 70.9% / 96.8% / 38.8% | 2.1% | 89.6% | 28.5 | 444.6 |

The two L40S prefill GPUs consume 55–61% of all measured energy while being
busy 1–3% of the time. Prefill work itself costs ~25–34 J/request; the
P-side idle floor charged to each request under STATIC-HIGH is ~900 J (HIGH)
to ~1,570 J (MEDIUM).

## 3. Where the reported savings come from

Savings versus the matched all-HIGH baseline, and the fraction explained by
the idle-floor difference:

| Stage | Load | Policy | Saving | From floor (total) | From floor (P side) |
|---|---|---|---:|---:|---:|
| C1b | MEDIUM | P=MID/D=HIGH (fixed) | 10.28% | 102% | 100% |
| C1b | HIGH | P=MID/D=HIGH (fixed) | 9.58% | 98% | 99% |
| C1b | MEDIUM | P=HIGH/D=MID (fixed) | 3.75% | 42% | — |
| C1b | HIGH | P=HIGH/D=MID (fixed) | 4.74% | 25% | — |
| C1c | MEDIUM | FEEDBACK | 9.39% | 92% | 98% |
| C1c | HIGH | FEEDBACK | 8.88% | 87% | 97% |
| C2a | MEDIUM | FAST | 8.40% | 95% | 97% |
| C2a | HIGH | FAST | 7.61% | 94% | 97% |
| D1 | MEDIUM | SHARED | 8.64% | 99% | 100% |
| D1 | MEDIUM | PER-ENDPOINT | 5.07% | 84% | 98% |
| D1 | MEDIUM | JOINT | 2.24% | 135% | 100% |
| D1 | HIGH | SHARED | 8.00% | 95% | 97% |
| D1 | HIGH | PER-ENDPOINT | 4.95% | 76% | 95% |
| D1 | HIGH | JOINT | 3.30% | 71% | 97% |

(>100% means load-dependent energy increased while the floor decreased; for
D1 JOINT MEDIUM, D-side load-dependent energy rose by ~21 J/request.)

## Gate outcome

1. **Floor share >= 70%: PASS at MEDIUM (77–79%) and at the lowest load
   (87%); borderline at HIGH-SUSTAINABLE (69–71%).**
2. **P-side savings from floor >= 80%: PASS (95–100% in every policy).**

The diagnosis is confirmed. Every accepted XPYD P-side DVFS saving (C1b,
C1c, C2a, D1) is a reduction of the idle power of L40S GPUs that are busy
1–3% of the time. Prefill work energy is essentially unchanged (~4 J/request
difference).

## Additional findings

- **Hard ceiling of locked-clock P-side DVFS.** At MEDIUM, mean total power is
  ~261 W. Even locking both L40S at LOW for the whole run removes only
  ~32 W (12.3%); MID removes ~27 W (10.4%). C1/C2a/D1 controllers operate
  inside this ceiling, and the LOW probes risked the SLO for an extra ~5 W
  (~2% of total).
- **D-side DVFS does act on real work.** For fixed D=MID/LOW, only 13–42% of
  the saving is floor; the rest is decode energy. That lever is blocked by the
  65 ms TPOT contract, not by physics.
- **Under-utilization also costs decode efficiency.** D load-dependent energy
  per request falls from ~520 J at MEDIUM to ~460 J at HIGH as more decodes
  overlap, consistent with better batching at higher load.
- **Routing in D1 had no floor lever.** All four GPUs stay powered and locked;
  JOINT's only measurable routing effect was more D-side load-dependent energy.

## Implications for E0

- **Default-governor break-even.** Because P energy is ~98% floor, a
  default-governor (unlocked) baseline beats STATIC-TUNED (P=MID) on the P side
  whenever its mean idle L40S power is below ~64.6 W, and beats STATIC-HIGH
  below ~78 W, provided TTFT holds. E0-a must measure this directly, with the
  model loaded and vLLM running.
- E0-b must check whether clock ramp-up from idle under the default governor
  costs TTFT attainment under the unchanged 600/65 contract.

## Limitations

- Proxy timestamps (Europa) and P-side NVML samples (Neptune) come from
  different hosts; the ±1 s guard and the guard-insensitivity check bound this.
- Idle power is measured with vLLM running and the model resident, which is
  the operationally relevant floor, not bare-board idle.
- Idle power is pooled per endpoint and clock; per-window variation is
  within ±1.5 W.
- No counterfactual energy is inferred for configurations that were not run.
