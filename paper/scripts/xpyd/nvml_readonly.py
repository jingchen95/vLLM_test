"""Strictly read-only NVML access for XpYd Phase 3B.

This module intentionally exposes no device-state mutation operation.  It is
kept separate from the repository's historical ``gpu_monitor.py`` because that
file also contains clock setters and reset helpers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import socket
from typing import Any, Mapping, Optional


class NVMLReadOnlyError(RuntimeError):
    """Identity, initialization, or query failure in the read-only source."""


def _text(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def structured_error(exc: BaseException) -> dict[str, str]:
    return {"error_type": type(exc).__name__, "error_message": str(exc)}


def _is_not_supported(binding: Any, exc: BaseException) -> bool:
    not_supported = getattr(binding, "NVMLError_NotSupported", None)
    if not_supported is not None and isinstance(exc, not_supported):
        return True
    code = getattr(exc, "value", None)
    return code is not None and code == getattr(binding, "NVML_ERROR_NOT_SUPPORTED", object())


@dataclass(frozen=True)
class GPUIdentity:
    endpoint: str
    role: str
    hostname: str
    configured_cuda_visible_device: str
    nvml_index: int
    gpu_name: str
    uuid: str
    pci_bus_id: str
    driver_version: Optional[str]
    nvml_version: Optional[str]


@dataclass(frozen=True)
class NVMLCapabilities:
    total_energy_supported: bool
    power_supported: bool
    total_energy_probe_mj: Optional[int]
    power_probe_w: Optional[float]
    power_query_source: Optional[str]
    total_energy_error: Optional[Mapping[str, str]]
    power_error: Optional[Mapping[str, str]]
    power_primary_error: Optional[Mapping[str, str]]


class ReadOnlyNVMLSource:
    """One initialized, identity-pinned NVML device source.

    ``binding`` is injectable so CPU tests never load NVML.  Numeric
    ``CUDA_VISIBLE_DEVICES`` tokens are interpreted as physical NVML indices,
    as defined by CUDA.  UUID and PCI tokens are resolved by immutable identity
    and preferred when supplied.  Comma-separated visibility is rejected: one
    monitor process must correspond to exactly one allocated GPU.
    """

    def __init__(
        self,
        endpoint: str,
        role: str,
        cuda_visible_device: str,
        *,
        expected_uuid: Optional[str] = None,
        expected_pci_bus_id: Optional[str] = None,
        expected_gpu_name: Optional[str] = None,
        binding: Any = None,
        hostname: Optional[str] = None,
    ) -> None:
        if not endpoint.strip():
            raise ValueError("endpoint must be non-empty")
        if role not in ("prefill", "decode"):
            raise ValueError("role must be prefill or decode")
        visible = str(cuda_visible_device).strip()
        if not visible or "," in visible:
            raise NVMLReadOnlyError(
                "one unambiguous CUDA-visible device is required (got %r)" % visible
            )
        if binding is None:
            try:
                import pynvml as binding  # type: ignore[no-redef]
            except ImportError as exc:
                raise NVMLReadOnlyError("pynvml is required on GPU monitor nodes") from exc
        self.binding = binding
        self.endpoint = endpoint
        self.role = role
        self.visible = visible
        self._shutdown = False
        self._power_query = None
        self.binding.nvmlInit()
        try:
            self.handle, index = self._resolve_handle(visible)
            identity = self._read_identity(index, hostname or socket.gethostname())
            if expected_uuid and identity.uuid.lower() != expected_uuid.lower():
                raise NVMLReadOnlyError(
                    "GPU UUID mismatch: resolved %s, expected %s" % (identity.uuid, expected_uuid)
                )
            if expected_pci_bus_id and identity.pci_bus_id.lower() != expected_pci_bus_id.lower():
                raise NVMLReadOnlyError(
                    "GPU PCI bus ID mismatch: resolved %s, expected %s" %
                    (identity.pci_bus_id, expected_pci_bus_id)
                )
            if expected_gpu_name and expected_gpu_name.lower() not in identity.gpu_name.lower():
                raise NVMLReadOnlyError(
                    "GPU name mismatch: resolved %s, expected substring %s" %
                    (identity.gpu_name, expected_gpu_name)
                )
            self.identity = identity
            self.capabilities = self._detect_capabilities()
        except Exception:
            self.close()
            raise

    def _enumerated(self) -> list[tuple[Any, int, str, str]]:
        result = []
        for index in range(int(self.binding.nvmlDeviceGetCount())):
            handle = self.binding.nvmlDeviceGetHandleByIndex(index)
            uuid = _text(self.binding.nvmlDeviceGetUUID(handle))
            pci = _text(self.binding.nvmlDeviceGetPciInfo(handle).busId)
            result.append((handle, index, uuid, pci))
        return result

    def _resolve_handle(self, token: str) -> tuple[Any, int]:
        devices = self._enumerated()
        lowered = token.lower()
        matches = []
        if lowered.startswith("gpu-"):
            matches = [item for item in devices if item[2].lower() == lowered]
        elif ":" in lowered:
            matches = [item for item in devices if item[3].lower() == lowered]
        else:
            try:
                index = int(token)
            except ValueError as exc:
                raise NVMLReadOnlyError(
                    "CUDA-visible device must be an NVML index, GPU UUID, or PCI bus ID"
                ) from exc
            matches = [item for item in devices if item[1] == index]
        if len(matches) != 1:
            raise NVMLReadOnlyError(
                "CUDA-visible device %r mapped to %d NVML devices" % (token, len(matches))
            )
        return matches[0][0], matches[0][1]

    def _read_identity(self, index: int, hostname: str) -> GPUIdentity:
        pci = self.binding.nvmlDeviceGetPciInfo(self.handle)
        driver = nvml = None
        try:
            driver = _text(self.binding.nvmlSystemGetDriverVersion())
        except Exception:
            pass
        try:
            nvml = _text(self.binding.nvmlSystemGetNVMLVersion())
        except Exception:
            pass
        return GPUIdentity(
            endpoint=self.endpoint,
            role=self.role,
            hostname=hostname,
            configured_cuda_visible_device=self.visible,
            nvml_index=index,
            gpu_name=_text(self.binding.nvmlDeviceGetName(self.handle)),
            uuid=_text(self.binding.nvmlDeviceGetUUID(self.handle)),
            pci_bus_id=_text(pci.busId),
            driver_version=driver,
            nvml_version=nvml,
        )

    def _probe(self, query: Any, scale: float = 1.0) -> tuple[bool, Optional[float], Optional[dict]]:
        try:
            value = float(query(self.handle)) * scale
            if not math.isfinite(value) or value < 0:
                raise NVMLReadOnlyError("NVML query returned invalid value %r" % value)
            return True, value, None
        except Exception as exc:
            return False, None, {**structured_error(exc), "not_supported": _is_not_supported(self.binding, exc)}

    def _detect_capabilities(self) -> NVMLCapabilities:
        energy_ok, energy, energy_error = self._probe(
            self.binding.nvmlDeviceGetTotalEnergyConsumption
        )
        power_ok, power, power_error = self._probe(
            self.binding.nvmlDeviceGetPowerUsage, 0.001
        )
        power_primary_error = power_error
        power_source = "nvmlDeviceGetPowerUsage" if power_ok else None
        if (
            not power_ok
            and bool((power_error or {}).get("not_supported"))
            and callable(getattr(self.binding, "nvmlDeviceGetFieldValues", None))
            and getattr(self.binding, "NVML_FI_DEV_POWER_AVERAGE", None) is not None
        ):
            power_ok, power, power_error = self._probe(
                self._read_power_average_field_mw, 0.001
            )
            if power_ok:
                power_source = "nvmlFieldValue_power_average"
        if power_source == "nvmlFieldValue_power_average":
            self._power_query = self._read_power_average_field_mw
        elif power_source == "nvmlDeviceGetPowerUsage":
            self._power_query = self.binding.nvmlDeviceGetPowerUsage
        return NVMLCapabilities(
            total_energy_supported=energy_ok,
            power_supported=power_ok,
            total_energy_probe_mj=int(energy) if energy is not None else None,
            power_probe_w=power,
            power_query_source=power_source,
            total_energy_error=energy_error,
            power_error=power_error,
            power_primary_error=power_primary_error,
        )

    def _read_power_average_field_mw(self, handle: Any) -> int:
        values = self.binding.nvmlDeviceGetFieldValues(
            handle, [self.binding.NVML_FI_DEV_POWER_AVERAGE]
        )
        if len(values) != 1:
            raise NVMLReadOnlyError("NVML power field query returned %d values" % len(values))
        value = values[0]
        status = int(value.nvmlReturn)
        if status != int(getattr(self.binding, "NVML_SUCCESS", 0)):
            error_type = getattr(self.binding, "NVMLError", NVMLReadOnlyError)
            raise error_type(status)
        return int(value.value.uiVal)

    def capability_record(self) -> dict[str, Any]:
        return {
            "identity": asdict(self.identity),
            "capabilities": asdict(self.capabilities),
            "measurement_semantics": {
                "total_energy_unit": "millijoules_since_driver_reload",
                "power_unit": "watts_converted_from_milliwatts",
                "power_reading_note": "approximately_one_second_average_on_ampere_or_newer",
            },
            "read_only": True,
        }

    def query(self) -> dict[str, Any]:
        """Read measurement and metadata fields; never change device state."""

        result: dict[str, Any] = {
            "gpu_uuid": None,
            "pci_bus_id": None,
            "power_w": None,
            "total_energy_mj": None,
            "graphics_clock_mhz": None,
            "memory_clock_mhz": None,
            "gpu_utilization_pct": None,
            "memory_utilization_pct": None,
            "temperature_c": None,
            "clock_throttle_reasons_mask": None,
            "clock_throttle_reasons": [],
            "invalidating_thermal_or_hw_slowdown": None,
            "field_errors": {},
        }
        try:
            result["gpu_uuid"] = _text(self.binding.nvmlDeviceGetUUID(self.handle))
            result["pci_bus_id"] = _text(
                self.binding.nvmlDeviceGetPciInfo(self.handle).busId
            )
        except Exception as exc:
            result["field_errors"]["identity"] = structured_error(exc)
        queries = {
            "graphics_clock_mhz": (
                lambda handle: self.binding.nvmlDeviceGetClockInfo(
                    handle, self.binding.NVML_CLOCK_GRAPHICS
                ), 1.0
            ),
            "memory_clock_mhz": (
                lambda handle: self.binding.nvmlDeviceGetClockInfo(
                    handle, self.binding.NVML_CLOCK_MEM
                ), 1.0
            ),
            "temperature_c": (
                lambda handle: self.binding.nvmlDeviceGetTemperature(
                    handle, self.binding.NVML_TEMPERATURE_GPU
                ), 1.0
            ),
        }
        if self._power_query is not None:
            queries["power_w"] = (self._power_query, 0.001)
        if self.capabilities.total_energy_supported:
            queries["total_energy_mj"] = (
                self.binding.nvmlDeviceGetTotalEnergyConsumption, 1.0
            )
        for field, (query, scale) in queries.items():
            try:
                value = float(query(self.handle)) * scale
                if not math.isfinite(value) or value < 0:
                    raise NVMLReadOnlyError("invalid %s value %r" % (field, value))
                result[field] = int(value) if field == "total_energy_mj" else value
            except Exception as exc:
                result["field_errors"][field] = structured_error(exc)
        try:
            utilization = self.binding.nvmlDeviceGetUtilizationRates(self.handle)
            result["gpu_utilization_pct"] = float(utilization.gpu)
            result["memory_utilization_pct"] = float(utilization.memory)
        except Exception as exc:
            result["field_errors"]["utilization"] = structured_error(exc)
        try:
            mask = int(
                self.binding.nvmlDeviceGetCurrentClocksThrottleReasons(
                    self.handle
                )
            )
            reason_constants = {
                "gpu_idle": "nvmlClocksThrottleReasonGpuIdle",
                "applications_clocks_setting": "nvmlClocksThrottleReasonApplicationsClocksSetting",
                "sw_power_cap": "nvmlClocksThrottleReasonSwPowerCap",
                "hw_slowdown": "nvmlClocksThrottleReasonHwSlowdown",
                "sync_boost": "nvmlClocksThrottleReasonSyncBoost",
                "sw_thermal_slowdown": "nvmlClocksThrottleReasonSwThermalSlowdown",
                "hw_thermal_slowdown": "nvmlClocksThrottleReasonHwThermalSlowdown",
                "hw_power_brake_slowdown": "nvmlClocksThrottleReasonHwPowerBrakeSlowdown",
                "display_clock_setting": "nvmlClocksThrottleReasonDisplayClockSetting",
            }
            reasons = []
            invalidating = False
            invalidating_names = {
                "hw_slowdown",
                "sw_thermal_slowdown",
                "hw_thermal_slowdown",
                "hw_power_brake_slowdown",
            }
            for name, attribute in reason_constants.items():
                bit = int(getattr(self.binding, attribute, 0))
                if bit and mask & bit:
                    reasons.append(name)
                    invalidating = invalidating or name in invalidating_names
            result["clock_throttle_reasons_mask"] = mask
            result["clock_throttle_reasons"] = reasons
            result["invalidating_thermal_or_hw_slowdown"] = invalidating
        except Exception as exc:
            result["field_errors"]["clock_throttle_reasons"] = structured_error(exc)
        return result

    def close(self) -> None:
        if not self._shutdown:
            self._shutdown = True
            self.binding.nvmlShutdown()

    def __enter__(self) -> "ReadOnlyNVMLSource":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def configured_cuda_device(explicit: Optional[str] = None) -> str:
    if explicit is not None:
        return explicit
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        raise NVMLReadOnlyError(
            "CUDA_VISIBLE_DEVICES is unset; pass --cuda-visible-device explicitly"
        )
    return visible
