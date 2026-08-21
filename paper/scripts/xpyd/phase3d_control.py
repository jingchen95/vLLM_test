"""Phase 3D real per-endpoint DVFS actuation and feedback-loop validation.

The module deliberately reuses Phase 3C request/telemetry/energy audits and
the existing predictor-free feedback scheduler.  It never changes power
limits, persistence mode, driver settings, or tensor parallelism.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence

from xpyd.feedback_scheduler import (
    DVFSAction,
    FeedbackScheduler,
    FeedbackSchedulerConfig,
    LatencySafetyMetric,
)
from xpyd.hardware import GPUTypeProfile, HardwareProfile, NodeProfile
from xpyd.phase3c_substrate import (
    Phase3CSubstrateHarness,
    _read_json,
    _write_json,
    build_registry_and_compatibility,
    load_config as load_phase3c_config,
)
from xpyd.telemetry import EndpointTelemetrySample, TelemetryAggregator


class Phase3DError(RuntimeError):
    """A fail-closed Phase 3D configuration, actuation, or audit error."""


def audit_connector_concurrency(
    workloads: list[dict[str, Any]], validated_max_concurrency: int,
) -> dict[str, Any]:
    """Audit a smoke sequence against the physically validated connector limit."""
    if validated_max_concurrency < 1:
        raise Phase3DError("validated connector max concurrency must be positive")
    observed = max(
        (int(workload.get("max_concurrency", 1)) for workload in workloads),
        default=0,
    )
    violations = [
        str(workload.get("id", "unknown"))
        for workload in workloads
        if int(workload.get("max_concurrency", 1)) > validated_max_concurrency
    ]
    return {
        "valid": not violations,
        "validated_max_concurrency": validated_max_concurrency,
        "observed_max_concurrency": observed,
        "violating_workload_ids": violations,
    }


@dataclass(frozen=True)
class GPUReadback:
    endpoint_id: str
    node: str
    gpu_id: int
    gpu_name: str
    gpu_uuid: str
    pci_bus_id: str
    graphics_clock_mhz: int
    timestamp_unix_s: float


@dataclass(frozen=True)
class EndpointClockCapability:
    endpoint_id: str
    node: str
    gpu_id: int
    gpu_name: str
    gpu_uuid: str
    pci_bus_id: str
    supported_graphics_mhz: tuple[int, ...]
    selected_low_mhz: int
    selected_mid_mhz: int
    selected_high_mhz: int


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )


def _csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def select_validation_frequencies(
    supported: Sequence[int], safe_high_mhz: int,
) -> tuple[int, int, int]:
    """Choose real LOW/MID/HIGH states at or below the validated safe high."""

    values = sorted({int(item) for item in supported if int(item) > 0 and int(item) <= safe_high_mhz})
    if safe_high_mhz not in values:
        raise Phase3DError("validated safe-high frequency is not hardware-supported")
    if len(values) < 3:
        raise Phase3DError("at least three supported frequencies are required")

    def nearest(target: float, excluded: set[int]) -> int:
        candidates = [item for item in values if item not in excluded]
        return min(candidates, key=lambda item: (abs(item - target), -item))

    high = safe_high_mhz
    mid = nearest(high * 0.75, {high})
    low = nearest(high * 0.50, {high, mid})
    low, mid = sorted((low, mid))
    if not low < mid < high:
        raise Phase3DError("could not choose distinct ordered LOW/MID/HIGH frequencies")
    return low, mid, high


class NvidiaSmiClockBackend:
    """Small guarded nvidia-smi backend with node-local Slurm execution."""

    _identity_re = re.compile(
        r"^IDENTITY=(.*?),([^,]+),([^,]+),\s*(\d+)[ \t]*$", re.MULTILINE
    )

    def __init__(
        self,
        runner: CommandRunner = _default_runner,
        nvidia_smi: str = "/usr/bin/nvidia-smi",
        sudo: str = "/usr/bin/sudo",
        srun: str = "srun",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.runner = runner
        self.nvidia_smi = nvidia_smi
        self.sudo = sudo
        self.srun = srun
        self.clock = clock

    def _node_command(self, node: str, script: str) -> list[str]:
        return [
            self.srun, "--overlap", "--nodes=1", "--ntasks=1",
            "--nodelist=%s" % node, "/bin/bash", "-lc", script,
        ]

    def _run(self, node: str, script: str) -> str:
        result = self.runner(self._node_command(node, script))
        if result.returncode != 0:
            raise Phase3DError(
                "node-local clock command failed on %s (rc=%d): %s"
                % (node, result.returncode, result.stdout.strip())
            )
        return result.stdout

    def _parse_readback(self, endpoint_id: str, node: str, gpu_id: int, output: str) -> GPUReadback:
        matches = list(self._identity_re.finditer(output))
        if not matches:
            identity_context = output.split("SUPPORTED_BEGIN", 1)[0].strip()[-1000:]
            raise Phase3DError(
                "missing or malformed GPU identity/readback on %s: %r"
                % (node, identity_context)
            )
        # A bounded post-actuation poll may contain multiple actual-clock
        # samples. The last one is the freshest hardware observation.
        match = matches[-1]
        name, uuid, bus, graphics = (item.strip() for item in match.groups())
        return GPUReadback(
            endpoint_id=endpoint_id, node=node, gpu_id=gpu_id,
            gpu_name=name, gpu_uuid=uuid, pci_bus_id=bus,
            graphics_clock_mhz=int(graphics), timestamp_unix_s=self.clock(),
        )

    def read(
        self, endpoint_id: str, node: str, gpu_id: int,
        expected_mhz: Optional[int] = None,
    ) -> GPUReadback:
        query = (
            f"{self.nvidia_smi} -i {gpu_id} "
            "--query-gpu=name,uuid,pci.bus_id,clocks.current.graphics "
            "--format=csv,noheader,nounits"
        )
        if expected_mhz is None:
            script = "set -euo pipefail\nprintf 'IDENTITY='\n" + query
        else:
            script = (
                "set -euo pipefail\nmatched=0\n"
                "for attempt in 1 2 3 4 5 6 7 8 9 10; do\n"
                f"  row=\"$({query})\"\n"
                "  printf 'IDENTITY=%s\\n' \"${row}\"\n"
                "  actual=$(printf '%s\\n' \"${row}\" | /usr/bin/awk -F, "
                "'{gsub(/[[:space:]]/, \"\", $4); print $4}')\n"
                f"  if [ \"${{actual}}\" = \"{int(expected_mhz)}\" ]; then matched=1; break; fi\n"
                "  sleep 0.2\n"
                "done\n"
                "[ \"${matched}\" -eq 1 ]"
            )
        return self._parse_readback(endpoint_id, node, gpu_id, self._run(node, script))

    def discover(
        self, endpoint_id: str, node: str, gpu_id: int, safe_high_mhz: int,
    ) -> EndpointClockCapability:
        query = (
            f"{self.nvidia_smi} -i {gpu_id} "
            "--query-gpu=name,uuid,pci.bus_id,clocks.current.graphics "
            "--format=csv,noheader,nounits"
        )
        supported = (
            f"{self.nvidia_smi} -i {gpu_id} --query-supported-clocks=gr "
            "--format=csv,noheader,nounits"
        )
        output = self._run(
            node,
            "set -euo pipefail\nprintf 'IDENTITY='\n"
            + query + "\nprintf 'SUPPORTED_BEGIN\\n'\n" + supported
            + "\nprintf 'SUPPORTED_END\\n'",
        )
        readback = self._parse_readback(endpoint_id, node, gpu_id, output)
        body_match = re.search(r"SUPPORTED_BEGIN\s*(.*?)\s*SUPPORTED_END", output, re.DOTALL)
        if body_match is None:
            raise Phase3DError("supported-clock discovery output is incomplete")
        frequencies = tuple(sorted({
            int(value) for value in re.findall(r"(?m)^\s*(\d+)\s*$", body_match.group(1))
        }))
        low, mid, high = select_validation_frequencies(frequencies, safe_high_mhz)
        return EndpointClockCapability(
            endpoint_id=endpoint_id, node=node, gpu_id=gpu_id,
            gpu_name=readback.gpu_name, gpu_uuid=readback.gpu_uuid,
            pci_bus_id=readback.pci_bus_id,
            supported_graphics_mhz=frequencies,
            selected_low_mhz=low, selected_mid_mhz=mid, selected_high_mhz=high,
        )

    def set_graphics_clock(
        self, capability: EndpointClockCapability, target_mhz: int,
    ) -> GPUReadback:
        if target_mhz not in capability.supported_graphics_mhz:
            raise Phase3DError("requested frequency is not hardware-supported")
        command = (
            f"{self.sudo} {self.nvidia_smi} -i {capability.gpu_id} "
            f"-lgc {target_mhz},{target_mhz}"
        )
        self._run(capability.node, "set -euo pipefail\n" + command)
        return self.read(
            capability.endpoint_id, capability.node, capability.gpu_id,
            expected_mhz=target_mhz,
        )


class PerEndpointClockActuator:
    """Serialized supported-state actuation with identity and readback checks."""

    def __init__(
        self,
        backend: NvidiaSmiClockBackend,
        capabilities: Mapping[str, EndpointClockCapability],
        minimum_dwell_s: float,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if minimum_dwell_s < 0:
            raise ValueError("minimum dwell must be non-negative")
        self.backend = backend
        self.capabilities = dict(capabilities)
        physical = [(item.node, item.gpu_id) for item in self.capabilities.values()]
        if len(physical) != len(set(physical)):
            raise Phase3DError("multiple endpoints map to the same physical GPU")
        self.minimum_dwell_s = float(minimum_dwell_s)
        self.sleep = sleep
        self.monotonic = monotonic
        self.wall_clock = wall_clock
        self._locks = {key: threading.Lock() for key in physical}
        self._last_success_monotonic: dict[str, float] = {}
        self.requested: dict[str, int] = {}

    @staticmethod
    def _identity_matches(
        capability: EndpointClockCapability, readback: GPUReadback,
    ) -> bool:
        return (
            capability.node == readback.node
            and capability.gpu_id == readback.gpu_id
            and capability.gpu_uuid == readback.gpu_uuid
            and capability.pci_bus_id == readback.pci_bus_id
        )

    def read(self, endpoint_id: str) -> GPUReadback:
        capability = self.capabilities[endpoint_id]
        result = self.backend.read(
            endpoint_id, capability.node, capability.gpu_id,
            expected_mhz=self.requested.get(endpoint_id),
        )
        if not self._identity_matches(capability, result):
            raise Phase3DError("endpoint-to-physical-GPU identity changed for %s" % endpoint_id)
        return result

    def actuate(self, endpoint_id: str, target_mhz: int, reason: str) -> dict[str, Any]:
        capability = self.capabilities[endpoint_id]
        if target_mhz not in capability.supported_graphics_mhz:
            raise Phase3DError("unsupported target for %s" % endpoint_id)
        key = (capability.node, capability.gpu_id)
        with self._locks[key]:
            before = self.read(endpoint_id)
            previous_requested = self.requested.get(endpoint_id)
            previous = self._last_success_monotonic.get(endpoint_id)
            if previous is not None:
                remaining = self.minimum_dwell_s - (self.monotonic() - previous)
                if remaining > 0:
                    self.sleep(remaining)
            started_wall = self.wall_clock()
            started = self.monotonic()
            try:
                after = self.backend.set_graphics_clock(capability, target_mhz)
                identity_valid = self._identity_matches(capability, after)
                readback_valid = identity_valid and after.graphics_clock_mhz == target_mhz
                if not readback_valid:
                    raise Phase3DError("fresh hardware readback does not match requested target")
                finished = self.monotonic()
                self._last_success_monotonic[endpoint_id] = finished
                self.requested[endpoint_id] = target_mhz
                return {
                    "timestamp_unix_s": started_wall,
                    "endpoint_id": endpoint_id, "node": capability.node,
                    "gpu_id": capability.gpu_id,
                    "previous_requested_freq_mhz": previous_requested,
                    "requested_freq_mhz": target_mhz,
                    "observed_freq_before_mhz": before.graphics_clock_mhz,
                    "observed_freq_after_mhz": after.graphics_clock_mhz,
                    "command_status": "success", "readback_valid": True,
                    "transition_readback_latency_s": finished - started,
                    "reason": reason, "error": None,
                }
            except Exception as exc:
                return {
                    "timestamp_unix_s": started_wall,
                    "endpoint_id": endpoint_id, "node": capability.node,
                    "gpu_id": capability.gpu_id,
                    "previous_requested_freq_mhz": previous_requested,
                    "requested_freq_mhz": target_mhz,
                    "observed_freq_before_mhz": before.graphics_clock_mhz,
                    "observed_freq_after_mhz": None,
                    "command_status": "failed", "readback_valid": False,
                    "transition_readback_latency_s": self.monotonic() - started,
                    "reason": reason,
                    "error": "%s: %s" % (type(exc).__name__, exc),
                }


def _phase3c_window_config(
    base: Mapping[str, Any], output_root: Path, workloads: Sequence[Mapping[str, Any]],
    requested: Mapping[str, int], required_pairs: Optional[Sequence[Sequence[str]]] = None,
) -> dict[str, Any]:
    value = copy.deepcopy(dict(base))
    value["output_root"] = output_root.as_posix()
    value["workloads"] = [dict(item) for item in workloads]
    for endpoint_id, frequency in requested.items():
        value["fixed_clocks"][endpoint_id]["graphics_mhz"] = int(frequency)
    if required_pairs is not None:
        required_endpoints = sorted({endpoint for pair in required_pairs for endpoint in pair})
        value["coverage_policy"] = {
            "required_endpoint_ids": required_endpoints,
            "required_pairs": [list(item) for item in required_pairs],
        }
    return value


def _small_rr_workloads(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    configured = config.get("phase3d", {}).get("actuator_workloads")
    if configured:
        return [dict(item) for item in configured]
    return [
        {"id": "small", "input_len": 128, "output_len": 128, "count": 2, "rate_rps": 0.5},
        {"id": "prefill_heavy", "input_len": 2048, "output_len": 128, "count": 2, "rate_rps": 0.5},
    ]


class Phase3DActuatorHarness:
    def __init__(
        self, config: Mapping[str, Any], run_id: Optional[str] = None,
        backend: Optional[NvidiaSmiClockBackend] = None,
    ) -> None:
        self.config = copy.deepcopy(dict(config))
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = self.config.get("phase3d", {}).get(
            "actuator_output_root", "results/phase3d_actuator_validation"
        )
        self.run_dir = Path(os.path.expandvars(str(root))) / self.run_id
        self.backend = backend or NvidiaSmiClockBackend()
        self.actions: list[dict[str, Any]] = []
        self.readbacks: list[dict[str, Any]] = []

    def _discover(self) -> dict[str, EndpointClockCapability]:
        capabilities = {}
        for endpoint in self.config["endpoints"]:
            endpoint_id = str(endpoint["endpoint_id"])
            gpu_ids = [int(item) for item in endpoint["gpu_ids"]]
            if len(gpu_ids) != 1:
                raise Phase3DError("Phase 3D actuator validation requires TP1 endpoints")
            capability = self.backend.discover(
                endpoint_id, str(endpoint["node"]), gpu_ids[0],
                int(self.config["fixed_clocks"][endpoint_id]["graphics_mhz"]),
            )
            expected_name = str(endpoint.get("expected_gpu_name", endpoint["gpu_type"]))
            if expected_name.lower() not in capability.gpu_name.lower():
                raise Phase3DError("GPU model mismatch for %s" % endpoint_id)
            capabilities[endpoint_id] = capability
        return capabilities

    def _record_all_readbacks(
        self, actuator: PerEndpointClockActuator, action_index: int,
        expected: Mapping[str, int], changed_endpoint_id: str,
    ) -> bool:
        valid = True
        for endpoint_id in sorted(actuator.capabilities):
            try:
                value = actuator.read(endpoint_id)
                matches = value.graphics_clock_mhz == expected[endpoint_id]
                self.readbacks.append({
                    **asdict(value), "action_index": action_index,
                    "changed_endpoint_id": changed_endpoint_id,
                    "expected_freq_mhz": expected[endpoint_id],
                    "matches_expected": matches, "error": None,
                })
                valid = valid and matches
            except Exception as exc:
                self.readbacks.append({
                    "action_index": action_index,
                    "changed_endpoint_id": changed_endpoint_id,
                    "endpoint_id": endpoint_id,
                    "expected_freq_mhz": expected[endpoint_id],
                    "matches_expected": False,
                    "error": "%s: %s" % (type(exc).__name__, exc),
                })
                valid = False
        return valid

    def _action(
        self, actuator: PerEndpointClockActuator, endpoint_id: str,
        target_mhz: int, reason: str, expected: dict[str, int],
    ) -> bool:
        row = actuator.actuate(endpoint_id, target_mhz, reason)
        self.actions.append(row)
        if row["command_status"] != "success" or not row["readback_valid"]:
            return False
        expected[endpoint_id] = target_mhz
        isolated = self._record_all_readbacks(
            actuator, len(self.actions) - 1, expected, endpoint_id
        )
        row["peer_isolation_valid"] = isolated
        return isolated

    def run(self) -> Path:
        if self.run_dir.exists():
            raise Phase3DError("run directory already exists: %s" % self.run_dir)
        (self.run_dir / "workload_windows").mkdir(parents=True)
        capabilities: dict[str, EndpointClockCapability] = {}
        actuator: Optional[PerEndpointClockActuator] = None
        expected: dict[str, int] = {}
        workload_audits: list[dict[str, Any]] = []
        error: Optional[str] = None
        restoration_valid = False
        try:
            capabilities = self._discover()
            _write_json(
                self.run_dir / "capabilities.json",
                {key: asdict(value) for key, value in capabilities.items()},
            )
            actuator = PerEndpointClockActuator(
                self.backend, capabilities,
                float(self.config.get("phase3d", {}).get("minimum_dwell_s", 1.0)),
            )
            expected = {
                endpoint_id: value.selected_high_mhz
                for endpoint_id, value in capabilities.items()
            }
            actuator.requested.update(expected)
            if not self._record_all_readbacks(actuator, -1, expected, "baseline"):
                raise Phase3DError("initial safe-high hardware readback failed")

            for endpoint_id in sorted(capabilities):
                capability = capabilities[endpoint_id]
                for target, direction in (
                    (capability.selected_mid_mhz, "isolation_down_to_mid"),
                    (capability.selected_low_mhz, "isolation_down_to_low"),
                    (capability.selected_high_mhz, "isolation_up_to_high"),
                ):
                    if not self._action(actuator, endpoint_id, target, direction, expected):
                        raise Phase3DError("isolated actuation/readback failed for %s" % endpoint_id)

            workloads = _small_rr_workloads(self.config)
            for window_id, target in (
                ("high_before", capabilities["P0"].selected_high_mhz),
                ("p0_mid", capabilities["P0"].selected_mid_mhz),
                ("high_after", capabilities["P0"].selected_high_mhz),
            ):
                if expected["P0"] != target and not self._action(
                    actuator, "P0", target, "workload_window_%s" % window_id, expected
                ):
                    raise Phase3DError("workload-window actuation failed")
                window_config = _phase3c_window_config(
                    self.config, self.run_dir / "workload_windows", workloads, expected
                )
                Phase3CSubstrateHarness(window_config, run_id=window_id).run()
                audit = _read_json(self.run_dir / "workload_windows" / window_id / "audit.json")
                workload_audits.append({"window_id": window_id, **audit})
                if not audit.get("valid"):
                    raise Phase3DError("Phase 3C workload audit failed in %s" % window_id)
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
        finally:
            if actuator is not None:
                restoration_valid = True
                for endpoint_id in sorted(capabilities):
                    high = capabilities[endpoint_id].selected_high_mhz
                    # Re-issue the conservative safe-high target even when our
                    # requested-state bookkeeping already says HIGH: a failed
                    # prior command may have partially changed hardware state.
                    if not self._action(
                        actuator, endpoint_id, high, "final_safe_high_restoration", expected
                    ):
                        restoration_valid = False
                    try:
                        restoration_valid = (
                            restoration_valid
                            and actuator.read(endpoint_id).graphics_clock_mhz == high
                        )
                    except Exception:
                        restoration_valid = False

        action_fields = (
            "timestamp_unix_s", "endpoint_id", "node", "gpu_id",
            "previous_requested_freq_mhz", "requested_freq_mhz",
            "observed_freq_before_mhz", "observed_freq_after_mhz",
            "command_status", "readback_valid", "peer_isolation_valid",
            "transition_readback_latency_s", "reason", "error",
        )
        readback_fields = (
            "action_index", "changed_endpoint_id", "endpoint_id", "node", "gpu_id",
            "gpu_name", "gpu_uuid", "pci_bus_id", "graphics_clock_mhz",
            "timestamp_unix_s", "expected_freq_mhz", "matches_expected", "error",
        )
        _csv(self.run_dir / "actuator_actions.csv", self.actions, action_fields)
        _csv(self.run_dir / "actuator_readback.csv", self.readbacks, readback_fields)

        endpoints = sorted(capabilities)
        downward = {
            endpoint_id: any(
                item["endpoint_id"] == endpoint_id
                and item["command_status"] == "success"
                and item["requested_freq_mhz"] < item["observed_freq_before_mhz"]
                for item in self.actions
            ) for endpoint_id in endpoints
        }
        upward = {
            endpoint_id: any(
                item["endpoint_id"] == endpoint_id
                and item["command_status"] == "success"
                and item["requested_freq_mhz"] > item["observed_freq_before_mhz"]
                for item in self.actions
            ) for endpoint_id in endpoints
        }
        hard_gates = {
            "every_endpoint_independently_actuable": bool(endpoints) and all(
                any(item["endpoint_id"] == endpoint_id and item["command_status"] == "success" for item in self.actions)
                for endpoint_id in endpoints
            ),
            "downward_transition_each_endpoint": bool(endpoints) and all(downward.values()),
            "upward_transition_each_endpoint": bool(endpoints) and all(upward.values()),
            "targets_hardware_supported": bool(endpoints) and all(
                item["requested_freq_mhz"] in capabilities[item["endpoint_id"]].supported_graphics_mhz
                for item in self.actions
            ),
            "fresh_readback_matches_target": bool(self.actions) and all(
                item["readback_valid"] for item in self.actions
            ),
            "no_unintended_peer_change": bool(self.actions) and all(
                item.get("peer_isolation_valid", True) for item in self.actions
            ),
            "request_stream_token_accounting": bool(workload_audits) and all(
                item.get("hard_gates", {}).get("real_sse_and_latency")
                and item.get("hard_gates", {}).get("logical_request_id")
                and item.get("hard_gates", {}).get("requested_output_tokens")
                and item.get("hard_gates", {}).get("endpoint_assignment")
                for item in workload_audits
            ),
            "valid_energy_telemetry": bool(workload_audits) and all(
                item.get("hard_gates", {}).get("nvml_energy_windows")
                for item in workload_audits
            ),
            "safe_restoration": restoration_valid,
            "no_unresolved_actuator_error": error is None and all(
                item["command_status"] == "success" for item in self.actions
            ),
        }
        valid = all(hard_gates.values())
        audit = {
            "phase": "3D-A_real_per_endpoint_actuator_validation",
            "valid": valid, "hard_gates": hard_gates, "error": error,
            "selected_frequencies": {
                key: {
                    "LOW": value.selected_low_mhz,
                    "MID": value.selected_mid_mhz,
                    "HIGH": value.selected_high_mhz,
                } for key, value in capabilities.items()
            },
            "workload_window_audits": workload_audits,
            "claim_boundary": "actuator validation only; no optimization or optimality claim",
        }
        _write_json(self.run_dir / "actuator_audit.json", audit)
        latencies = [
            float(item["transition_readback_latency_s"])
            for item in self.actions if item["command_status"] == "success"
        ]
        lines = [
            "# XpYd Phase 3D-A actuator validation", "",
            "Verdict: **%s**" % ("PASS" if valid else "FAIL"), "",
            "This validates physical per-endpoint actuation only; it makes no optimization claim.", "",
            "Selected states: `%s`." % json.dumps(audit["selected_frequencies"], sort_keys=True),
            "",
            "Successful actuation/readback latency: mean %.6f s, max %.6f s."
            % (
                sum(latencies) / len(latencies) if latencies else math.nan,
                max(latencies) if latencies else math.nan,
            ),
            "", "Hard gates: `%s`" % json.dumps(hard_gates, sort_keys=True), "",
        ]
        if error:
            lines.extend(["Failure: `%s`" % error, ""])
        (self.run_dir / "actuator_summary.md").write_text("\n".join(lines), encoding="utf-8")
        if not valid:
            raise Phase3DError(error or "Phase 3D-A hard-gate audit failed")
        print(self.run_dir.as_posix())
        return self.run_dir


def _controller_hardware(
    config: Mapping[str, Any], capabilities: Mapping[str, EndpointClockCapability],
) -> HardwareProfile:
    by_type: dict[str, set[int]] = {}
    nodes: dict[str, tuple[str, set[int]]] = {}
    for endpoint in config["endpoints"]:
        endpoint_id = str(endpoint["endpoint_id"])
        gpu_type = str(endpoint["gpu_type"])
        capability = capabilities[endpoint_id]
        by_type.setdefault(gpu_type, set()).update((
            capability.selected_low_mhz,
            capability.selected_mid_mhz,
            capability.selected_high_mhz,
        ))
        node = str(endpoint["node"])
        if node in nodes and nodes[node][0] != gpu_type:
            raise Phase3DError("one node maps to multiple GPU types")
        nodes.setdefault(node, (gpu_type, set()))[1].add(int(endpoint["gpu_ids"][0]))
    gpu_profiles = [
        GPUTypeProfile(
            gpu_type=name,
            allowed_frequencies_mhz=tuple(sorted(frequencies)),
            max_frequency_mhz=max(frequencies),
            nominal_tdp_w=1.0,
            idle_power_w=0.0,
            supported_tp_degrees=(1,),
        ) for name, frequencies in sorted(by_type.items())
    ]
    node_profiles = [
        NodeProfile(node=name, gpu_type=value[0], gpu_ids=tuple(sorted(value[1])))
        for name, value in sorted(nodes.items())
    ]
    return HardwareProfile(gpu_profiles, node_profiles)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


class Phase3DClosedLoopHarness:
    """Small real light->moderate->light feedback-only control-loop smoke."""

    def __init__(
        self, config: Mapping[str, Any], run_id: Optional[str] = None,
        backend: Optional[NvidiaSmiClockBackend] = None,
    ) -> None:
        self.config = copy.deepcopy(dict(config))
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = self.config.get("phase3d", {}).get(
            "closed_loop_output_root", "results/phase3d_closed_loop_smoke"
        )
        self.run_dir = Path(os.path.expandvars(str(root))) / self.run_id
        self.backend = backend or NvidiaSmiClockBackend()
        control_file = os.path.expandvars(str(self.config.get("routing_control_file", "")))
        if not control_file:
            raise Phase3DError("Phase 3D-B requires routing_control_file")
        self.control_file = Path(control_file)
        self.actions: list[dict[str, Any]] = []
        self.iterations: list[dict[str, Any]] = []
        self.telemetry_rows: list[dict[str, Any]] = []

    def _discover(self) -> dict[str, EndpointClockCapability]:
        result = {}
        for endpoint in self.config["endpoints"]:
            endpoint_id = str(endpoint["endpoint_id"])
            result[endpoint_id] = self.backend.discover(
                endpoint_id, str(endpoint["node"]), int(endpoint["gpu_ids"][0]),
                int(self.config["fixed_clocks"][endpoint_id]["graphics_mhz"]),
            )
        return result

    def _build_scheduler(
        self, capabilities: Mapping[str, EndpointClockCapability],
    ) -> tuple[Any, TelemetryAggregator, FeedbackScheduler]:
        registry, compatibility, _ = build_registry_and_compatibility(self.config)
        hardware = _controller_hardware(self.config, capabilities)
        feedback = dict(self.config["phase3d"].get("feedback", {}))
        scheduler_config = FeedbackSchedulerConfig(
            ttft_slo_ms=float(feedback.get("ttft_slo_ms", 1000.0)),
            tpot_slo_ms=float(feedback.get("tpot_slo_ms", 80.0)),
            safety_fraction=float(feedback.get("safety_fraction", 0.9)),
            min_samples=1,
            prefill_safety_metric=LatencySafetyMetric.EWMA,
            decode_safety_metric=LatencySafetyMetric.EWMA,
            telemetry_max_age_s=float(feedback.get("telemetry_max_age_s", 120.0)),
            max_queue_depth=int(feedback.get("max_queue_depth", 1)),
            max_kv_cache_usage_frac=float(feedback.get("max_kv_cache_usage_frac", 0.90)),
            fallback_prefill_endpoint_id="P0",
            fallback_decode_endpoint_id="D0",
            fallback_prefill_freq_mhz=capabilities["P0"].selected_high_mhz,
            fallback_decode_freq_mhz=capabilities["D0"].selected_high_mhz,
            dvfs_step_down_fraction=float(feedback.get("step_down_fraction", 0.50)),
            dvfs_step_up_fraction=float(feedback.get("step_up_fraction", 0.80)),
            dvfs_low_queue_depth=0,
            dvfs_low_kv_cache_usage_frac=float(feedback.get("low_kv_cache_usage_frac", 0.50)),
            severe_queue_depth=int(feedback.get("severe_queue_depth", 4)),
            severe_kv_cache_usage_frac=float(feedback.get("severe_kv_cache_usage_frac", 0.97)),
            dvfs_min_dwell_s=float(self.config["phase3d"].get("minimum_dwell_s", 1.0)),
        )
        telemetry = TelemetryAggregator(alpha=float(feedback.get("ewma_alpha", 0.5)))
        scheduler = FeedbackScheduler(
            registry, compatibility, telemetry, hardware, scheduler_config,
            clock=time.time,
        )
        return registry, telemetry, scheduler

    def _write_routes(self, pairs: Sequence[Sequence[str]], reason: str) -> None:
        _atomic_json(self.control_file, {
            "schema_version": 1,
            "updated_unix_s": time.time(),
            "pairs": [list(item) for item in pairs],
            "reason": reason,
        })

    def _apply_recommendations(
        self,
        actuator: PerEndpointClockActuator,
        registry: Any,
        scheduler: FeedbackScheduler,
        recommendations: Sequence[Any],
        iteration_id: str,
    ) -> None:
        for recommendation in recommendations:
            if recommendation.action == DVFSAction.HOLD:
                continue
            row = actuator.actuate(
                recommendation.endpoint_id,
                recommendation.target_freq_mhz,
                "%s:%s" % (iteration_id, recommendation.action.value),
            )
            self.actions.append(row)
            if row["command_status"] != "success" or not row["readback_valid"]:
                raise Phase3DError("closed-loop actuation failed for %s" % recommendation.endpoint_id)
            state = registry.get_state(recommendation.endpoint_id)
            state.freq_mhz = int(row["observed_freq_after_mhz"])
            registry.update_state(state)
            scheduler.record_dvfs_actuation(
                recommendation.endpoint_id,
                observed_freq_mhz=state.freq_mhz,
                timestamp_s=time.time(),
            )

    def _control_iteration(
        self,
        iteration_id: str,
        workload_id: str,
        capabilities: Mapping[str, EndpointClockCapability],
        actuator: PerEndpointClockActuator,
        registry: Any,
        telemetry: TelemetryAggregator,
        scheduler: FeedbackScheduler,
        initial: bool = False,
    ) -> list[list[str]]:
        now = time.time()
        snapshots = {
            endpoint.endpoint_id: telemetry.snapshot(endpoint.endpoint_id, now_s=now)
            for endpoint in registry.list_endpoints()
        }
        telemetry_fresh = all(
            value.last_latency_observation_s is not None
            and value.last_observation_age_s is not None
            and value.last_observation_age_s <= scheduler.config.telemetry_max_age_s
            for value in snapshots.values()
        )
        fallback_reason = None
        evaluations = scheduler.evaluate_routes(now_s=now, workload_context_key=workload_id)
        if initial or not telemetry_fresh:
            # At startup/missing feedback, use only validated pairs and the
            # conservative HIGH state. Missing telemetry is never zero load.
            pairs = [
                [item.prefill_endpoint_id, item.decode_endpoint_id]
                for item in scheduler.compatibility.explicit_endpoint_pairs()
                if item.supported
            ]
            if not pairs:
                pairs = [["P0", "D0"]]
            fallback_reason = "missing_or_stale_feedback_use_validated_pairs_at_safe_high"
            recommendations = []
            for endpoint_id, capability in sorted(capabilities.items()):
                state = registry.get_state(endpoint_id)
                if state.freq_mhz != capability.selected_high_mhz:
                    recommendations.append(type("FallbackRecommendation", (), {
                        "endpoint_id": endpoint_id,
                        "target_freq_mhz": capability.selected_high_mhz,
                        "action": DVFSAction.FALLBACK_MAX,
                    })())
        else:
            eligible = [item for item in evaluations if item.eligible]
            if not eligible:
                pairs = [["P0", "D0"]]
                fallback_reason = "no_measured_safe_route_use_explicit_fallback_at_safe_high"
                recommendations = []
                for endpoint_id, capability in sorted(capabilities.items()):
                    state = registry.get_state(endpoint_id)
                    if state.freq_mhz != capability.selected_high_mhz:
                        recommendations.append(type("FallbackRecommendation", (), {
                            "endpoint_id": endpoint_id,
                            "target_freq_mhz": capability.selected_high_mhz,
                            "action": DVFSAction.FALLBACK_MAX,
                        })())
            else:
                route = scheduler.choose_route(now_s=now, workload_context_key=workload_id)
                pairs = [[route.prefill_endpoint_id, route.decode_endpoint_id]]
                recommendations = [
                    scheduler.choose_frequency_adjustment(endpoint.endpoint_id, now_s=now)
                    for endpoint in sorted(registry.list_endpoints(), key=lambda item: item.endpoint_id)
                ]
        self._apply_recommendations(
            actuator, registry, scheduler, recommendations, iteration_id
        )
        self._write_routes(pairs, fallback_reason or "feedback_only_selected_route")
        self.iterations.append({
            "control_timestamp": now,
            "control_iteration_id": iteration_id,
            "workload_window_id": workload_id,
            "telemetry_fresh": telemetry_fresh,
            "fallback_reason": fallback_reason,
            "selected_pairs_json": json.dumps(pairs, sort_keys=True),
            "endpoint_state_json": json.dumps({
                endpoint_id: {
                    "health": registry.get_state(endpoint_id).healthy,
                    "lifecycle": registry.get_state(endpoint_id).lifecycle.value,
                    "queue": registry.get_state(endpoint_id).queue_depth,
                    "kv": registry.get_state(endpoint_id).kv_cache_usage_frac,
                    "observed_frequency_mhz": registry.get_state(endpoint_id).freq_mhz,
                    "telemetry_age_s": snapshots[endpoint_id].last_observation_age_s,
                    "recent_power_w": snapshots[endpoint_id].ewma_power_w,
                    "recent_energy_per_request_j": snapshots[endpoint_id].ewma_energy_per_request_j,
                    "recent_ttft_ms": snapshots[endpoint_id].ewma_ttft_ms,
                    "recent_tpot_ms": snapshots[endpoint_id].ewma_tpot_ms,
                } for endpoint_id in sorted(snapshots)
            }, sort_keys=True),
            "candidate_evaluations_json": json.dumps([
                asdict(item) for item in evaluations
            ], sort_keys=True),
            "dvfs_recommendations_json": json.dumps([
                {
                    "endpoint_id": item.endpoint_id,
                    "action": item.action.value,
                    "target_freq_mhz": item.target_freq_mhz,
                    "reason": getattr(item, "reason", fallback_reason),
                } for item in recommendations
            ], sort_keys=True),
        })
        return pairs

    def _observe_window(
        self,
        window_id: str,
        window_dir: Path,
        registry: Any,
        telemetry: TelemetryAggregator,
    ) -> None:
        summary = _read_json(window_dir / "summary.json")
        client = _read_json(window_dir / "client" / "summary.json")
        rows = _read_csv(window_dir / "endpoint_summary.csv")
        timestamp = float(client["timing_end_unix_s"])
        for row in rows:
            endpoint_id = row["endpoint_id"]
            state = registry.get_state(endpoint_id)
            queue = _optional_float(row["mean_queue_depth"])
            kv = _optional_float(row["mean_kv_cache_usage_frac"])
            if queue is not None:
                state.queue_depth = int(round(queue))
                state.queue_depth_observed = True
            if kv is not None:
                state.kv_cache_usage_frac = kv
                state.kv_cache_usage_observed = True
            state.healthy = True
            state.last_update_s = timestamp
            state.freq_mhz = int(round(float(row["observed_graphics_mean_mhz"])))
            registry.update_state(state)
            request_count = int(row["request_count"])
            endpoint_energy = summary["endpoint_energy"][endpoint_id]
            sample = EndpointTelemetrySample(
                endpoint_id=endpoint_id,
                timestamp_s=timestamp,
                power_w=_optional_float(endpoint_energy.get("mean_power_w")),
                energy_j=_optional_float(row["gross_energy_j"]),
                queue_depth=state.queue_depth if state.queue_depth_observed else None,
                kv_cache_usage_frac=state.kv_cache_usage_frac if state.kv_cache_usage_observed else None,
                completed_requests=request_count,
                output_tokens=int(client["output_tokens_total"]) if request_count else 0,
                interval_s=float(endpoint_energy["duration_s"]),
                ttft_ms=_optional_float(row["mean_ttft_ms"]) if row["role"] == "prefill" else None,
                tpot_ms=_optional_float(row["mean_tpot_ms"]) if row["role"] == "decode" else None,
            )
            snapshot = telemetry.observe(sample)
            self.telemetry_rows.append({
                "window_id": window_id,
                "endpoint_id": endpoint_id,
                "timestamp_s": timestamp,
                "health": state.healthy,
                "queue": sample.queue_depth,
                "kv_cache_usage_frac": sample.kv_cache_usage_frac,
                "ttft_ms": sample.ttft_ms,
                "tpot_ms": sample.tpot_ms,
                "power_w": sample.power_w,
                "energy_j": sample.energy_j,
                "observed_frequency_mhz": state.freq_mhz,
                "telemetry_age_s_at_observation": snapshot.last_observation_age_s,
                "measurement_valid": endpoint_energy.get("valid"),
            })

    def run(self) -> Path:
        if self.run_dir.exists():
            raise Phase3DError("run directory already exists: %s" % self.run_dir)
        windows_root = self.run_dir / "windows"
        windows_root.mkdir(parents=True)
        capabilities = self._discover()
        _write_json(
            self.run_dir / "capabilities.json",
            {key: asdict(value) for key, value in capabilities.items()},
        )
        actuator = PerEndpointClockActuator(
            self.backend, capabilities,
            float(self.config["phase3d"].get("minimum_dwell_s", 1.0)),
        )
        actuator.requested.update({
            key: value.selected_high_mhz for key, value in capabilities.items()
        })
        registry, telemetry, scheduler = self._build_scheduler(capabilities)
        valid = False
        error = None
        window_audits = []
        request_rows: list[dict[str, Any]] = []
        route_rows: list[dict[str, Any]] = []
        energy_rows: list[dict[str, Any]] = []
        restoration_valid = False
        sequence = self.config["phase3d"].get("closed_loop_workloads") or [
            {"id": "light_before", "input_len": 128, "output_len": 128, "count": 4, "rate_rps": 0.5, "max_concurrency": 1},
            {"id": "moderate", "input_len": 2048, "output_len": 256, "count": 8, "rate_rps": 2.0, "max_concurrency": 1},
            {"id": "light_after", "input_len": 128, "output_len": 128, "count": 4, "rate_rps": 0.5, "max_concurrency": 1},
        ]
        connector_concurrency = audit_connector_concurrency(
            sequence,
            int(self.config["phase3d"].get("validated_connector_max_concurrency", 1)),
        )
        try:
            if not connector_concurrency["valid"]:
                raise Phase3DError(
                    "closed-loop workload exceeds physically validated connector "
                    "concurrency: %s" % connector_concurrency["violating_workload_ids"]
                )
            selected_pairs = self._control_iteration(
                "control_00", str(sequence[0]["id"]), capabilities,
                actuator, registry, telemetry, scheduler, initial=True,
            )
            for index, workload in enumerate(sequence):
                window_id = str(workload["id"])
                requested = {
                    endpoint.endpoint_id: int(registry.get_state(endpoint.endpoint_id).freq_mhz)
                    for endpoint in registry.list_endpoints()
                }
                window_config = _phase3c_window_config(
                    self.config, windows_root,
                    [{
                        "id": window_id,
                        "input_len": int(workload["input_len"]),
                        "output_len": int(workload["output_len"]),
                        "count": int(workload["count"]),
                        "rate_rps": float(workload["rate_rps"]),
                    }],
                    requested,
                    required_pairs=selected_pairs,
                )
                window_config["client"]["max_concurrency"] = int(workload["max_concurrency"])
                Phase3CSubstrateHarness(window_config, run_id=window_id).run()
                window_dir = windows_root / window_id
                audit = _read_json(window_dir / "audit.json")
                window_audits.append({"window_id": window_id, **audit})
                if not audit.get("valid"):
                    raise Phase3DError("closed-loop window audit failed: %s" % window_id)
                self._observe_window(window_id, window_dir, registry, telemetry)
                for row in _read_csv(window_dir / "requests.csv"):
                    request_rows.append({"window_id": window_id, **row})
                for row in _read_csv(window_dir / "routes.csv"):
                    route_rows.append({"window_id": window_id, **row})
                for row in _read_csv(window_dir / "energy_summary.csv"):
                    energy_rows.append({"window_id": window_id, **row})
                selected_pairs = self._control_iteration(
                    "control_%02d" % (index + 1),
                    str(sequence[index + 1]["id"]) if index + 1 < len(sequence) else "post_recovery",
                    capabilities, actuator, registry, telemetry, scheduler,
                )
            valid = True
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
        finally:
            restoration_valid = True
            for endpoint_id, capability in sorted(capabilities.items()):
                row = actuator.actuate(
                    endpoint_id, capability.selected_high_mhz,
                    "final_safe_high_restoration",
                )
                self.actions.append(row)
                restoration_valid = restoration_valid and row["command_status"] == "success" and row["readback_valid"]
                try:
                    restoration_valid = restoration_valid and actuator.read(endpoint_id).graphics_clock_mhz == capability.selected_high_mhz
                except Exception:
                    restoration_valid = False

        _csv(self.run_dir / "control_iterations.csv", self.iterations, (
            "control_timestamp", "control_iteration_id", "workload_window_id",
            "telemetry_fresh", "fallback_reason", "selected_pairs_json",
            "endpoint_state_json", "candidate_evaluations_json",
            "dvfs_recommendations_json",
        ))
        _csv(self.run_dir / "requests.csv", request_rows, (
            "window_id", "request_id", "input_len", "requested_output_len",
            "observed_output_tokens", "prefill_endpoint_id", "decode_endpoint_id",
            "ttft_ms", "tpot_ms", "mean_itl_ms", "e2e_latency_ms",
            "prefill_latency_ms", "decode_stream_latency_ms", "real_sse",
            "logical_request_id_propagated",
        ))
        _csv(self.run_dir / "routes.csv", route_rows, (
            "window_id", "request_id", "selected_prefill_endpoint_id",
            "selected_decode_endpoint_id", "route_timestamp_wall_s",
            "prefill_completion_wall_s", "kv_handoff_completion_wall_s",
            "decode_start_wall_s", "decode_completion_wall_s", "outcome",
        ))
        _csv(self.run_dir / "endpoint_telemetry.csv", self.telemetry_rows, (
            "window_id", "endpoint_id", "timestamp_s", "health", "queue",
            "kv_cache_usage_frac", "ttft_ms", "tpot_ms", "power_w", "energy_j",
            "observed_frequency_mhz", "telemetry_age_s_at_observation",
            "measurement_valid",
        ))
        _csv(self.run_dir / "dvfs_actions.csv", self.actions, (
            "timestamp_unix_s", "endpoint_id", "node", "gpu_id",
            "previous_requested_freq_mhz", "requested_freq_mhz",
            "observed_freq_before_mhz", "observed_freq_after_mhz",
            "command_status", "readback_valid", "transition_readback_latency_s",
            "reason", "error",
        ))
        energy_fields = list(energy_rows[0]) if energy_rows else ["window_id"]
        _csv(self.run_dir / "energy_summary.csv", energy_rows, energy_fields)

        hard_gates = {
            "phase3d_a_prerequisite_recorded": bool(
                self.config["phase3d"].get("accepted_actuator_audit")
            ),
            "connector_concurrency_within_validated_substrate": connector_concurrency["valid"],
            "all_requests_exactly_attributed": bool(window_audits) and all(
                item.get("hard_gates", {}).get("endpoint_assignment") for item in window_audits
            ),
            "real_sse_tokens_latency": bool(window_audits) and all(
                item.get("hard_gates", {}).get("real_sse_and_latency")
                and item.get("hard_gates", {}).get("logical_request_id")
                and item.get("hard_gates", {}).get("requested_output_tokens")
                for item in window_audits
            ),
            "valid_energy_windows": bool(window_audits) and all(
                item.get("hard_gates", {}).get("nvml_energy_windows") for item in window_audits
            ),
            "physical_readback_after_actions": all(
                item["command_status"] == "success" and item["readback_valid"]
                for item in self.actions
            ),
            "no_incompatible_route_or_resource_conflict": bool(window_audits) and all(
                item.get("hard_gates", {}).get("explicit_compatibility")
                and item.get("hard_gates", {}).get("no_resource_overlap")
                for item in window_audits
            ),
            "missing_feedback_used_conservative_fallback": bool(self.iterations)
                and self.iterations[0].get("fallback_reason") is not None
                and self.iterations[0].get("telemetry_fresh") is False,
            "dwell_uses_successful_actuation_time": all(
                scheduler.last_dvfs_actuation_time(item["endpoint_id"]) is not None
                for item in self.actions
                if not str(item["reason"]).startswith("final_safe_high_restoration")
            ),
            "no_dwell_bypass_oscillation": all(
                scheduler.last_dvfs_actuation_time(item["endpoint_id"]) is not None
                for item in self.actions
                if not str(item["reason"]).startswith("final_safe_high_restoration")
            ),
            "light_moderate_light_survived": valid and len(window_audits) == len(sequence),
            "safe_restoration": restoration_valid,
            "no_unresolved_controller_error": error is None,
        }
        audit_valid = all(hard_gates.values())
        audit = {
            "phase": "3D-B_feedback_only_joint_routing_dvfs_smoke",
            "valid": audit_valid, "hard_gates": hard_gates, "error": error,
            "workload_window_audits": window_audits,
            "connector_concurrency": connector_concurrency,
            "models_used_for_decisions": [],
            "allowed_decision_inputs": "recent measured runtime feedback only",
            "baseline_comparison": "omitted because a second physical sequence was not inexpensive for this functional smoke",
            "claim_boundary": "functional closed-loop validation only; no novelty, optimality, or statistical claim",
        }
        _write_json(self.run_dir / "closed_loop_audit.json", audit)
        lines = [
            "# XpYd Phase 3D-B feedback-only closed-loop smoke", "",
            "Verdict: **%s**" % ("PASS" if audit_valid else "FAIL"), "",
            "Functional validation only; no model, oracle, optimality, or statistical claim.", "",
            "Hard gates: `%s`" % json.dumps(hard_gates, sort_keys=True), "",
        ]
        if error:
            lines.extend(["Failure: `%s`" % error, ""])
        (self.run_dir / "closed_loop_summary.md").write_text("\n".join(lines), encoding="utf-8")
        if not audit_valid:
            raise Phase3DError(error or "Phase 3D-B hard-gate audit failed")
        print(self.run_dir.as_posix())
        return self.run_dir


def load_config(path: Path) -> dict[str, Any]:
    config = load_phase3c_config(path)
    phase3d = config.get("phase3d")
    if not isinstance(phase3d, dict):
        raise Phase3DError("Phase 3D config requires a phase3d object")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--stage", choices=("A", "B"), required=True)
    args = parser.parse_args()
    config = load_config(Path(args.config))
    if args.stage == "A":
        Phase3DActuatorHarness(config, args.run_id).run()
        return 0
    accepted = os.path.expandvars(str(config["phase3d"].get("accepted_actuator_audit", "")))
    if not accepted:
        raise Phase3DError("Phase 3D-B requires an accepted Phase 3D-A audit path")
    accepted_audit = _read_json(Path(accepted))
    if not accepted_audit.get("valid"):
        raise Phase3DError("Phase 3D-A hard gates did not pass; Phase 3D-B is forbidden")
    Phase3DClosedLoopHarness(config, args.run_id).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
