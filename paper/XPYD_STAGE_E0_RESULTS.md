# XPYD Stage E0 — Regime diagnosis results

**Status:** complete and valid  
**Slurm job:** `274238` (COMPLETED, exit 0:0, 2026-10-09 01:44–03:52, 2 h 07 min)  
**Run:** `results/stage_e0_regime/stage_e0_regime_neptune_europa_20261008`  
**Code:** commit `5ca91a9` (`paper/scripts/xpyd/stage_e0_regime.py`)  
**Copied outputs:** `paper/step1_energy_decomposition/e0_summary.json`, `e0_idle_ladder.csv`

## Questions (registered before data, see config `stage_e0.decision_rules`)

- **E0-a:** what is the idle power of each GPU (vLLM running, model resident)
  at every locked clock and under the default driver governor?
- **E0-b:** on the unchanged C1 trace and 600/65 ms >= 95% contract, how do
  STATIC-HIGH, STATIC-TUNED (P=MID/D=HIGH), P-default/D-HIGH and all-default
  compare?

## Validity

Every hard gate passed: 24 serving windows (4 points × 2 loads × 3 repeats ×
20 requests), measurement/clock/energy/throttle validity for all points,
matched traces, valid idle windows, and final HIGH plus memory-clock
restoration. The `srun ... exit code 1` lines at teardown also appear in D1 job
272628 and are a known shutdown artifact.

## E0-a: idle power per GPU (W)

| State | L40S (pre / post ladder) | L40S observed clock | L4 (pre / post) | L4 observed clock |
|---|---|---|---|---|
| Locked HIGH | 77.9 / 77.7 | 2520 | 23.1 / 23.6 | 1500 |
| Locked MID | 64.6 / 64.2 | 1890 | 21.0 / 21.1 | 1125 |
| Locked LOW | 62.1 / 61.9 | 1260 | 20.4 / 20.4 | 750 |
| **DEFAULT governor** | **80.3 / 77.8** | **2520, every sample** | **27.7 / 27.6** | **2040, every sample** |

Under the default governor, with the vLLM servers resident, no GPU ever left
its maximum clock during 210 s of zero-utilization idle (1049 samples per GPU,
no throttle reason reported, including no `gpu_idle`). The locked values
reproduce Step 1's estimates from older runs within 1 W.

**Pre-registered P-side rule:** default idle L40S power 80.3 W is **not**
below the 64.6 W break-even; the default governor cannot beat STATIC-TUNED on
the P side.

## E0-b: matched serving points

| Point | Load | J/request | vs HIGH_HIGH | vs DEFAULT_DEFAULT | TTFT p95 / attain | TPOT p95 / attain | SLO every repeat |
|---|---|---:|---:|---:|---|---|---|
| HIGH_HIGH | MEDIUM | 2606.6 | 0 | +3.2% saving | 564.6 / 96.7% | 63.24 / 100% | No (cold start, see below) |
| P_MID_D_HIGH | MEDIUM | 2345.2 | **10.0%** | **12.9%** | 569.2 / 100% | 63.24 / 100% | Yes |
| P_DEFAULT_D_HIGH | MEDIUM | 2624.0 | −0.7% | +2.6% | 555.7 / 100% | 63.24 / 100% | Yes |
| DEFAULT_DEFAULT | MEDIUM | 2693.5 | −3.3% | 0 | 563.4 / 100% | 62.65 / 100% | Yes |
| HIGH_HIGH | HIGH-SUST. | 1674.2 | 0 | +2.3% saving | 578.5 / 100% | 63.21 / 100% | Yes |
| P_MID_D_HIGH | HIGH-SUST. | 1516.7 | **9.4%** | **11.5%** | 564.3 / 100% | 63.22 / 100% | Yes |
| P_DEFAULT_D_HIGH | HIGH-SUST. | 1675.6 | −0.1% | +2.2% | 582.1 / 100% | 63.21 / 100% | Yes |
| DEFAULT_DEFAULT | HIGH-SUST. | 1713.2 | −2.3% | 0 | 565.4 / 100% | 62.68 / 100% | Yes |

The only SLO-violating requests in the whole run are the first two requests of
HIGH_HIGH MEDIUM repeat 1 (TTFT 727 and 838 ms). This window was the first
serving window after the 16-minute idle ladder; it is the same first-request
warm-up effect seen in C1 (1497.7 ms) and in the C2 calibration, not a
property of HIGH clocks.

P_MID_D_HIGH reproduces C1b (10.28% / 9.58%) within 0.3 pp.

## Decomposition (Step 1 method, idle power keyed by clock mode)

| Point | Load | P floor share | P mean power | D mean power |
|---|---|---:|---:|---:|
| HIGH_HIGH | MEDIUM | 98.5% | 158.7 W | 101.1 W |
| P_MID_D_HIGH | MEDIUM | 97.8% | 132.1 W | 101.6 W |
| DEFAULT_DEFAULT | MEDIUM | 98.4% | 158.6 W | 110.0 W |

- STATIC-TUNED's saving is 99–104% idle-floor reduction on the P side, as in
  every earlier run.
- DEFAULT is energetically identical to STATIC-HIGH on the P side. On the D
  side it runs the L4s at 2040 MHz, which shortens TPOT by ~0.6 ms but adds
  ~7–9 W per pool, so all-default costs 2.3–3.3% more than STATIC-HIGH.

## Pre-registered rules

| Rule | Outcome |
|---|---|
| DEFAULT meets the contract in every repeat **and** J/request <= STATIC-TUNED → direction A | Not triggered: DEFAULT meets the contract but uses 12.9% / 11.5% more energy |
| DEFAULT or P-default violates the contract → direction B with DEFAULT baseline | Not triggered: both meet the contract |
| Default idle L40S < 64.6 W | False: 80.3 W |

The observed outcome is a case the two branches did not anticipate: the
default governor is SLO-safe but no more efficient than locking at HIGH,
because a resident vLLM process keeps every GPU at maximum clock.

## Conclusions

1. **The production default is effectively STATIC-HIGH.** STATIC-HIGH was
   therefore a fair baseline for C1/C2a/D1, although for a different reason
   than originally argued.
2. **Idle-clock control has real value over the zero-knowledge baseline.**
   STATIC-TUNED saves 11.5–12.9% versus the default governor with no SLO cost.
   The driver cannot capture this; it must come from explicit control.
3. **The lever is idle power, and locked DVFS has a hard ceiling.** No locked
   clock takes an idle L40S below ~62 W, so P-side DVFS can save at most
   ~12–13% at these loads. Taking a P GPU out of service entirely would remove
   62–80 W, i.e. roughly 24–30% of total power at MEDIUM, but only if the freed
   GPU actually leaves the powered, model-resident state (released to other
   work or powered down) and the cold-start cost is paid. Merely removing it
   from routing leaves it at >= 62 W. Consolidation is therefore the larger
   but costlier lever, and DVFS the safer one.
4. **Knowledge needed for the DVFS lever is small.** One offline point (P=MID)
   captures it, and C1c/C2a feedback found the same point. The no-profiling
   argument therefore needs a larger action space or a non-stationary workload
   to show a difference (see the knowledge-cost discussion in the chat record).

## Caveats

- The "no idle downclock" behaviour was measured with vLLM 0.15.1 and the
  P2P-NCCL connector resident. Whether a colocated vLLM process, or a process
  without NCCL communicators, behaves the same is open (E0-d).
- Loads are still the C0 client-capped 0.10 / 0.20 req/s. E0-c (job 274435)
  measures real capacity; conclusions 2–3 must be re-checked at realistic
  utilization.
- Clock-mode-keyed idle power for DEFAULT inside serving windows (P 76–80 W, D
  30–31 W) is slightly different from the idle ladder (D 27.7 W), probably
  temperature; it does not change any conclusion.
