# Research Gap Analysis

## Conclusion

Choose **B. Existing foundation is useful but scheduler/runtime architecture needs significant changes.**

Do not start over. Reuse the measurement pipeline, L40S/L4 characterization, predictor/training stack, workload traces, joint-search concepts, policy baselines, simulator, plotting code, and fixed-pair vLLM/KV-transfer experience. Refactor the scheduler's resource representation and build a new runtime control plane for real XpYd endpoint routing.

The overlap is high for modeling and experiment infrastructure, medium/high for the optimization objective and search logic, and low for physical XpYd serving integration. The physical runtime—not the research foundation—is the major missing layer.

## Old and proposed problems

### Existing implemented problem

Given a 5-second window of request lengths/rates, use learned L40S/L4 performance and power models to choose, in simulation:

- a prefill GPU-type pool and decode GPU-type pool per length class;
- one TP and one frequency per GPU-type pool;
- modeled prefill/decode GPU allocations and active instance counts;
- the lowest predicted GPU power/energy candidate passing a configurable feasibility gate.

The existing controller also compares static, frequency-only, monolithic, two-tier, sequential, joint, and single-knob ablation policies.

### Proposed problem

Operate real independent prefill and decode pools containing multiple simultaneously available, pre-instantiated vLLM endpoints with different TP values; independently choose a concrete P endpoint and D endpoint per request/workload state; support P TP != D TP where the connector permits; include queue/KV/runtime state; control per-instance DVFS; and minimize prefill + transfer + decode energy under explicit TTFT and TPOT SLOs.

## Component-by-component status

| Component | Status | Evidence and implication |
|---|---|---|
| L40S prefill characterization | **PARTIALLY READY — intrinsic phase model available; validate under real P/D runtime** | `Phase2_Results_L40S/master_results.csv`; `model_fitting_runner.py`; canonical `prefill_*` models. These characterize TTFT and power from monolithic single-pool execution, including prefill-dominant shapes; they do not establish behavior after physical P/D separation, KV transfer, or live queueing. |
| L4 decode characterization | **PARTIALLY READY — intrinsic phase model available; validate continuous-batching/KV effects** | `Phase2_Results_L4/master_results.csv`; canonical `decode_*` models; `decode_capacity.csv`. These provide an intrinsic decode prior, but continuous batching, transferred-KV state, decode queueing, and tight-TPOT boundaries require live P/D validation. |
| L40S DVFS model | **READY** | Ten modeled core frequencies 210-2520 MHz in `GPU_SPECS` and predictor config; power/latency/capacity include normalized frequency. |
| L4 DVFS model | **READY** | Ten modeled core frequencies 210-2040 MHz and phase models. Sparse non-max-frequency coverage is a caveat. |
| TP modeling | **PARTIALLY REUSABLE** | TP is a measured/model feature and searched over L40S {1,2,4}, L4 {1,2,4,8}. `PoolBundle` allows different TP across A/B, but one uniform TP per GPU-type pool; it cannot model simultaneous TP1+TP2 instances in one pool. |
| TP x frequency interaction | **PARTIALLY REUSABLE** | Both axes enter the learned models and joint search; Observation-2/motivation code explicitly searches their combinations. Coverage audit shows only 8-12% of `(IL,OL,freq)` cells include every TP and many shapes are single-clock, so targeted profiling is still needed. |
| TTFT predictor | **REUSE AS PRIOR / RANKER; recalibrate for live XpYd** | `prefill_latency_regressor_p99.pkl` and classifiers target P99 TTFT; `EnergyScheduler.predict_prefill_config()`. Rank correlation is useful for initial configuration ordering, but absolute tails must be recalibrated with queueing and KV-transfer delay from the deployed endpoint topology. |
| TPOT predictor | **REUSE AS PRIOR / RANKER; recalibrate for live XpYd** | `decode_latency_regressor_p99.pkl` and classifiers target P99 TPOT; `predict_decode_config()`. Use the existing ordering signal as a prior, then recalibrate for continuous batching, decode queues, transferred KV state, and L4 tight-SLO behavior. |
| Power models | **HIGHLY REUSABLE — validate transferability** | Canonical prefill/decode per-GPU power estimators already cover GPU type, workload shape, TP, frequency, rate, and modeled utilization. Validate their transfer from monolithic measurements to separated endpoints and continuous batching before using them as deployed energy ground truth. |
| Energy objective | **NEEDS MODIFICATION** | Per-phase GPU power and window energy are implemented in `_predict_pool_details()`/`_evaluate_candidate()`. KV/interconnect energy is omitted, so the proposed `E_prefill + E_KV + E_decode` objective is incomplete. |
| Capacity models | **REUSE AS INITIAL ESTIMATOR; validate endpoint-level capacity** | Capacity GBR, ETA margin, rho decomposition, goodput table, and prefill envelope provide an initial load estimator. Live queue delay, batching, KV occupancy, and per-endpoint effective capacity are missing; L4 new-shape capacity CV error is high. |
| KV transfer model | **PARTIALLY REUSABLE** | Analytical latency `tau_kv * IL`, a budget gate, physical P2P NCCL experiments, and a working proxy exist. Transfer energy, contention, topology, queueing, connector compatibility matrix, and representative-fabric validation are missing. |
| Independent P routing | **NEEDS MODIFICATION** | Per-class choice of prefill **pool A/B** exists in `_admissible_routes()` and route DFS. It does not choose P0/P1/P2 physical endpoints. |
| Independent D routing | **NEEDS MODIFICATION** | Per-class decode pool can differ from prefill pool. It does not choose D0/D1/D2 endpoints or use live D queue/KV state. |
| XpYd physical vLLM instances | **MISSING** | `run_experiment_E()` launches one producer and one consumer only. No registry, discovery, health/load polling, or multi-endpoint request router exists. |
| TP1/TP2 pre-instantiated instances | **MISSING at runtime** | Benchmark scripts can relaunch different TP configurations and phase-only pilots sweep TP, but they do not keep mixed-TP endpoint sets alive simultaneously. |
| P TP != D TP support | **UNCLEAR / unvalidated** | Simulation can assign different A/B TP values. Physical experiment declares one `local tp=1` for both servers. Connector behavior for unequal TP has not been tested in this repository. |
| Online scheduler | **PARTIALLY REUSABLE** | `decide_window()` is stateful and timed, but evaluated offline. It discards `cluster` and `current_time`, has no live telemetry adapter, and emits model configs rather than runtime actions. |
| SLO-aware optimization | **NEEDS MODIFICATION** | Phase predictors/classifiers and explicit TTFT/TPOT modes exist. Main frozen evaluation uses a common rho gate, making SLO/KV sensitivity weak; real deployment needs explicit SLO feasibility with calibrated queue/transfer uncertainty. |
| Reserve GPU activation | **PARTIALLY REUSABLE** | Active GPU count and power-gated capacity are searched mathematically. There is no activation API, startup delay, weight loading, draining, or reserve policy. |
| Warm/cold instance management | **MISSING** | Bundle stability is only a simulator counter; no warm/cold lifecycle exists. |
| Real vLLM integration | **PARTIALLY REUSABLE** | Strong benchmark/launch/monitoring/P2P proxy code exists, including vLLM 0.15.1 connector debugging. Scheduler-to-runtime application and robust upstream integration are absent; current script patches installed vLLM source at runtime. |
| Trace replay | **READY for simulation; PARTIAL for hardware** | Synthetic/Azure simulator replay is mature. `replay_synthetic_trace.py` and `WORKLOAD_MODE=trace` provide hardware plumbing, but not multi-endpoint scheduler-driven replay. |
| Evaluation infrastructure | **READY** | Frozen synthetic/Azure summaries, ablations, sensitivity studies, model validation, overhead measurement, figure generation, and paper integration exist. It must be extended to distinguish simulator decisions from actual served outcomes. |

## What “routing” currently means

The route tuple is `(prefill_pool, decode_pool)` with A=L40S and B=L4:

- A->A: both phases on the L40S pool.
- A->B: L40S prefill, L4 decode.
- B->A: L4 prefill, L40S decode.
- B->B: both phases on the L4 pool.

`SweepLLMStrategy` may choose a different tuple for each length class in a window. It then aggregates every class assigned to a pool/phase, divides the load evenly over a modeled number of identical instances, and queries one pool TP/frequency model.

This is **not physical endpoint routing**. There is no P0/P1/P2 or D0/D1/D2 identifier, URI, queue, health state, KV-cache state, or request dispatch. The output's `active_instances` is derived from GPU counts divided by uniform TP; it does not correspond to instantiated server processes.

## Reuse recommendation by layer

| Layer | Recommendation | Rationale |
|---|---|---|
| Raw L40S/L4 measurements | **REUSE and extend selectively** | They already cover the target GPUs, TP, frequency, lengths, rate, power, TTFT, and TPOT. Add only missing high-value TP x frequency x phase cells. |
| TTFT/TPOT models | **REUSE AS PRIOR / RANKER; recalibrate for live XpYd** | Preserve their configuration-ordering signal, then recalibrate absolute latency and feasibility against real endpoint queues, continuous batching, KV transfer, and P/D topology. |
| Capacity models | **REUSE AS INITIAL ESTIMATOR; validate endpoint-level capacity** | Preserve the shared capacity/rho machinery as an initialization prior, then measure effective capacity per concrete endpoint and runtime state. |
| Power models | **HIGHLY REUSABLE; validate transferability** | Preserve phase GPU-power models and validate monolithic-to-P/D transferability. Extend the energy objective with transfer/network, actuation/startup, and duration-aware accounting. |
| Search and policies | **REFACTOR/EXTEND** | Reuse class aggregation, state classification, joint-search/branch-and-bound, and baselines. Replace `PoolBundle` with an inventory of concrete instances, each with role, TP, GPUs, frequency, status, and telemetry. |
| Simulator | **REUSE/EXTEND** | Preserve trace/window harness and evaluation outputs. Add per-instance queues, service/transfer stages, KV occupancy, activation delays, and physical-outcome replay. |
| Physical benchmark/KV code | **REUSE carefully** | Launch/clock/monitor/proxy work saves substantial effort. Extract the embedded proxy, stop runtime patching of installed vLLM where possible, and validate a supported connector. |
| Runtime control plane | **BUILD** | Multi-P/multi-D server launch, registry, independent selection, telemetry, request-to-request KV metadata, and failure handling do not exist. |
| Historical/paper assets | **PRESERVE** | They document prior directions, results, and limitations. Curate rather than delete. |

## Recommended evolution path

1. Freeze and publish the curated research snapshot described in `GITHUB_UPLOAD_PLAN.md`; keep raw artifacts in an external release with checksums.
2. Reconcile canonical model/validation provenance and add a dependency lock/cluster software manifest.
3. Define an explicit `Instance`/`Endpoint` schema: ID, role, GPU type/IDs, TP, frequency, URI, KV connector/rank metadata, queue, KV occupancy, state, warmup and activation cost.
4. Refactor scheduler inputs/outputs around a concrete instance inventory while retaining existing predictors and policy baselines. Do not dynamically change TP; route to pre-instantiated endpoints.
5. Build a minimal physical pool with P0/P1/P2 and D0/D1/D2, initially fixed clocks and always warm. Validate independent selection and KV correctness before adding DVFS.
6. Explicitly test the connector matrix for P-TP/D-TP pairs (1->1, 1->2, 2->1, 2->2). Mark unsupported pairs as hard constraints rather than assuming compatibility.
7. Measure per-endpoint prefill/decode energy and transfer latency/energy under representative concurrency; complete the proposed objective.
8. Add live queue/batch/KV telemetry and explicit TTFT/TPOT admission. Only then add slower reserve activation and warm/cold management.
9. Run paired simulator-versus-hardware trace replays and report model prediction error at the **decision outcome** level.

## Research claim boundary

The existing repository supports the claim that joint heterogeneous pool assignment, TP, DVFS, and modeled active capacity can be studied effectively with the measured L40S/L4 stack. It does not yet support the claim that an online XpYd vLLM deployment independently routes requests among mixed-TP physical P and D instances, nor that unequal P/D TP KV transfer works. Those are the new engineering and measurement contributions—not a reason to discard the existing work.
