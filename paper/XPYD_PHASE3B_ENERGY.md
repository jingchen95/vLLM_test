# XpYd Phase 3B read-only GPU energy baseline

## Scope and architecture

Phase 3B adds measurement only. It does not change scheduling, routing,
placement, request/KV semantics, clocks, persistence mode, or power limits.
`xpyd.nvml_readonly` is isolated from the historical `gpu_monitor.py` and
contains query operations only. One process runs locally on P0's L40S node and
one independently on D0's L4 node; NVML is node-local, and cross-node monotonic
clocks are never compared.

Each process initializes NVML once, maps its single configured CUDA-visible
device, and pins the result by immutable UUID and PCI bus ID. UUID and PCI
tokens are resolved directly; a numeric CUDA visibility token is resolved to
that physical NVML index and then recorded as UUID/PCI. A missing, multi-device,
non-unique, or expected-identity mismatch is fatal.

## Sources and units

The primary source is `nvmlDeviceGetTotalEnergyConsumption`: cumulative
millijoules since the driver was loaded. A valid window reports

```text
gross_energy_j = (end_counter_mj - start_counter_mj) / 1000
```

The counter method is invalidated by a decrease/reset, identity change,
device/query error in the interval, insufficient boundary coverage, excessive
sampling gap, or insufficient coverage.

Only an explicit `NVML_ERROR_NOT_SUPPORTED` permits the energy fallback. The
fallback integrates valid watt samples over local monotonic time with the
trapezoidal rule and is labelled `power_integral_estimate`; it is never labelled
as a hardware counter. Other counter errors do not trigger fallback. Power is
queried first with `nvmlDeviceGetPowerUsage`. If that getter explicitly reports
unsupported, the source may use NVML's read-only
`NVML_FI_DEV_POWER_AVERAGE` field and records
`nvmlFieldValue_power_average` as the power source. A failure of both invalidates
the window.

NVIDIA defines power in milliwatts and documents that on Ampere (except GA100)
and newer GPUs the value is averaged over about one second. A 200 ms polling
period therefore improves boundary/cadence auditing but does not create
independent 200 ms power measurements or exact short-event energy attribution.

Authoritative semantics: [NVIDIA NVML device queries](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html)
and [NVML power fields](https://docs.nvidia.com/deploy/nvml-api/group__nvmlFieldValueEnums.html).

Graphics/memory clocks, utilization, and temperature are read-only metadata.
Dynamic clock equality is not used as proof of unchanged state. The evidence
for non-actuation is that the new runtime exposes no setter and invokes no
state-changing `nvidia-smi` command.

## Fixed-period records and coverage

Deadlines are `base_monotonic + sequence * period`, not previous-finish plus
period. Reads never overlap within an endpoint. If a read crosses deadlines,
each skipped slot is an explicit `missed` JSONL record. P0 and D0 have separate
processes, so one endpoint cannot shift the other's schedule.

Each attempted record includes scheduled/start/finish local monotonic time,
start/finish wall time, query latency, start drift, late status, identity,
power/counter values, capabilities, metadata, and structured error class and
message. A dedicated writer flushes each record without placing shared-filesystem
latency on the sampling deadline path. A node-local monitor observes a shared
stop marker and drains that writer before exit, preserving a valid prefix after
termination. This is audited best-effort cadence, not a Minerva real-time
guarantee.

Window summaries report requested and actual boundaries, duration, source,
gross energy, power statistics, scheduled/successful/missed/late/error counts,
drift, maximum gap, coverage, boundary gaps, and explicit invalidity reasons.
The physical preflight represents capability, idle, excluded warm-up, semantic,
measured workload, and cooldown windows. The preflight workload is the five
request light-load interval; warm-up and the single semantic correctness probe
are not included in its reported workload energy.

## Normalization and claim boundaries

Logical requests and token totals come from validated client artifacts. P0 and
D0 completion counts are never added. `J/request` and `J/output token` appear
only for a valid window with a positive denominator. These are aggregate-window
normalizations, not exact causal energy attribution to individual requests.

Gross GPU-board energy remains primary. When valid comparable idle and workload
windows exist, the optional value

```text
gross_workload_energy_j - idle_mean_power_w * workload_duration_s
```

is retained separately as an unclamped incremental estimate. A P0+D0 sum is
explicitly a two-GPU-board aggregate, not total system energy.

CPU/node, NIC/network, KV-transfer-specific, cooling, and facility energy are
unobserved. The preflight is a functionality/data-integrity check, not a
performance or energy comparison. It introduces no scheduler, routing, or DVFS
claim.
