# XPYD Stage E0 on Grid'5000 — A100 + A100 regime diagnosis

**Status:** complete and valid  
**OAR job:** `2234560` (Lille, chuc ×2, 2026-10-18 night, stage exit 0)  
**Run:** `~/vLLM_test/results/g5k/e0_g5k_a100pair_night3` on Lille  
**Code:** commit `1cadf3e` (G5K port: `paper/scripts/g5k/`, `stage_e0_regime.py`)  
**Copied outputs:** `paper/step1_energy_decomposition/e0_g5k_a100pair_{summary.json,idle_ladder.csv}`

Same matrix, trace, SLO (600/65 ms, >=95%) and decision rules as the Minerva
E0 (`XPYD_STAGE_E0_RESULTS.md`, job 274238). Both pools are A100-SXM4-40GB
(the only H200 node was booked through the week). Levels chosen by the same
rule: LOW 705, MID 1065, HIGH 1410 MHz. A100 memory clock is fixed (HBM,
1215 MHz) and cannot be locked. Earlier attempts (2230985, 2233968) failed on
launcher bugs before measuring anything.

## E0-a: idle power per GPU with vLLM resident (pre / post ladder)

| State | A100 idle W (prefill pair) | A100 idle W (decode pair) | Observed clock |
|---|---|---|---|
| Locked HIGH | 111.6–116.9 | 112.4–119.1 | 1410 |
| Locked MID | 74.7–77.8 | 74.4–75.8 | 1065 |
| Locked LOW | 65.8–69.0 | 65.6–66.0 | 705 |
| **DEFAULT governor** | **76.0–78.8** | **75.9–77.2** | **1095, every sample** |

Without any CUDA context (validation job 2230332) the same A100s idled at
**58.4 W and 210 MHz**. With the vLLM servers resident, the default governor
holds the A100 at its application clock (1095 MHz) instead.

Pre-registered P-side rule: default idle 77.4 W is not below the 64.6 W
break-even (that threshold was derived for the L40S).

## E0-b: matched serving points

| Point | Load | J/request | vs HIGH_HIGH | P / D energy (kJ) | TTFT p95 | TPOT p95 | SLO every repeat |
|---|---|---:|---:|---|---:|---:|---|
| HIGH_HIGH | MEDIUM | 4963 | 0 | 131.9 / 165.9 | 314 ms | 13.8 ms | yes |
| P_MID_D_HIGH | MEDIUM | 4304 | 13.3% | 88.9 / 169.4 | 301 ms | 13.8 ms | yes |
| P_DEFAULT_D_HIGH | MEDIUM | 4372 | 11.9% | 92.8 / 169.5 | 295 ms | 13.8 ms | yes |
| **DEFAULT_DEFAULT** | MEDIUM | **3773** | **24.0%** | 92.6 / 133.8 | 296 ms | 13.8 ms | yes |
| HIGH_HIGH | HIGH-SUST. | 2824 | 0 | 68.3 / 101.2 | 264 ms | 13.8 ms | yes |
| P_MID_D_HIGH | HIGH-SUST. | 2462 | 12.8% | 45.0 / 102.7 | 297 ms | 13.8 ms | yes |
| P_DEFAULT_D_HIGH | HIGH-SUST. | 2518 | 10.8% | 48.1 / 103.0 | 294 ms | 13.8 ms | yes |
| **DEFAULT_DEFAULT** | HIGH-SUST. | **2280** | **19.3%** | 48.1 / 88.6 | 294 ms | 13.8 ms | yes |

Every point meets the contract in every repeat. On A100 the 600/65 ms
contract is very loose: TPOT is 13.8 ms (21% of the bound) and TTFT p95
~300 ms.

## Pre-registered rules

| Rule | Minerva L40S/L4 | G5K A100/A100 |
|---|---|---|
| DEFAULT meets SLO every repeat and J/request <= P_MID_D_HIGH → direction A | not triggered | **triggered** (both loads) |
| DEFAULT or P-default violates SLO → direction B | not triggered | not triggered |
| Default idle P GPU < 64.6 W | false (80.3 W) | false (77.4 W) |

**Caveat on the triggered rule.** P_MID_D_HIGH was the measured optimum on
Minerva and is reused as "STATIC-TUNED" by name. On A100 it is not tuned: it
keeps the decode pair at 1410 MHz (114 W idle) although TPOT is 13.8 ms
against a 65 ms bound. Locked LOW idles at ~66 W, i.e. ~11 W per GPU below
DEFAULT. A static point such as P=MID/D=LOW (not measured) would very likely
beat DEFAULT. The rule therefore shows that DEFAULT beats locked HIGH and a
mis-transferred static point, not that DEFAULT beats a tuned static point.

## Cross-platform conclusions

1. **Common mechanism: a resident serving process prevents deep idle.** On
   both platforms the GPU, with vLLM resident and zero utilisation, stays at
   a high "resident" clock instead of the deep-idle state seen without a CUDA
   context: L40S/L4 at their maximum (2520 / 2040 MHz), A100 at its
   application clock (1095 MHz). The deep-idle floor (A100: 58 W at
   210 MHz) is unreachable while the model is loaded.
2. **What differs is where the resident clock sits.** On L40S/L4 it equals
   the maximum, so DEFAULT ≈ STATIC-HIGH and explicit idle-clock control is
   needed to save anything. On A100 it is 1095/1410, so DEFAULT already saves
   19–24% versus locked HIGH, and explicit control can save a further ~11 W per
   idle GPU (DEFAULT 77 W → locked LOW 66 W).
3. **Locked STATIC-HIGH is a platform-dependent straw man.** On A100 it costs
   +37 W per idle GPU over DEFAULT. Baselines in the re-grounded evaluation
   must be DEFAULT plus a per-platform tuned static point, not HIGH.
4. **The controllable idle gap per GPU** (DEFAULT idle − locked LOW idle):
   L40S ~16 W, L4 ~7 W, A100 ~11 W. The additional gap to deep idle (A100:
   66 → 58 W) requires releasing the model, i.e. consolidation.
5. **Knowledge transfer across hardware fails silently.** The Minerva-tuned
   point, transferred by level name, uses 14% (MEDIUM) and 8% (HIGH-SUST.)
   more energy than DEFAULT on A100, while still passing the SLO. This
   is direct evidence for the cost of stale or transferred knowledge, which
   the "low offline knowledge" thesis needs.

## Limitations

- Homogeneous A100/A100 pair: no heterogeneity effect is tested here.
- No A100-tuned static point was measured; conclusion 2's "further ~11 W"
  is an idle-power difference, not a serving result.
- Loads are the Minerva C1 loads (0.10 / 0.20 req/s); A100 capacity is far
  higher, so both points are deep in the idle-dominated regime.
- Idle power keyed by clock only; no per-window decomposition was run for
  this job yet.
