"""CPU-only validation for the Phase 3B read-only energy baseline."""

import io
import json
import math
from pathlib import Path
import tempfile
import threading
import time
import unittest

from xpyd.nvml_readonly import NVMLReadOnlyError, ReadOnlyNVMLSource
from xpyd.phase3b_energy import (
    AsyncFlushingStream,
    FixedPeriodEnergySampler,
    WindowSpec,
    _stop_monitors,
    add_idle_adjustment,
    aggregate_endpoint_workload,
    summarize_window,
    build_parser,
)


class FakeNotSupported(Exception):
    value = 3


class FakeNVML:
    NVMLError_NotSupported = FakeNotSupported
    NVML_ERROR_NOT_SUPPORTED = 3
    NVML_CLOCK_GRAPHICS = 0
    NVML_CLOCK_MEM = 1
    NVML_TEMPERATURE_GPU = 0
    nvmlClocksThrottleReasonGpuIdle = 1
    nvmlClocksThrottleReasonSwPowerCap = 4
    nvmlClocksThrottleReasonHwSlowdown = 8

    class PCI:
        def __init__(self, bus_id):
            self.busId = bus_id

    class Utilization:
        gpu = 12
        memory = 7

    def __init__(self, *, energy_supported=True, energies=None, names=None):
        self.energy_supported = energy_supported
        self.energies = iter(energies or [1000, 1100, 1200, 1300, 1400, 1500])
        self.names = names or ["NVIDIA L40S", "NVIDIA L4"]
        self.initialized = 0
        self.shutdown = 0
        self.setter_accessed = False

    def __getattr__(self, name):
        if name.startswith("nvmlDeviceSet") or name.startswith("nvmlDeviceReset"):
            self.setter_accessed = True
            raise AssertionError("read-only source accessed mutating NVML symbol %s" % name)
        raise AttributeError(name)

    def nvmlInit(self):
        self.initialized += 1

    def nvmlShutdown(self):
        self.shutdown += 1

    def nvmlDeviceGetCount(self):
        return len(self.names)

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetUUID(self, handle):
        return ("GPU-fixture-%d" % handle).encode()

    def nvmlDeviceGetPciInfo(self, handle):
        return self.PCI(("00000000:%02x:00.0" % (handle + 1)).encode())

    def nvmlDeviceGetName(self, handle):
        return self.names[handle].encode()

    def nvmlSystemGetDriverVersion(self):
        return b"fixture-driver"

    def nvmlSystemGetNVMLVersion(self):
        return b"fixture-nvml"

    def nvmlDeviceGetTotalEnergyConsumption(self, handle):
        if not self.energy_supported:
            raise FakeNotSupported("counter unsupported")
        return next(self.energies)

    def nvmlDeviceGetPowerUsage(self, handle):
        return 50000 + handle * 1000

    def nvmlDeviceGetClockInfo(self, handle, clock):
        return 1500 if clock == self.NVML_CLOCK_GRAPHICS else 6000

    def nvmlDeviceGetTemperature(self, handle, sensor):
        return 45

    def nvmlDeviceGetUtilizationRates(self, handle):
        return self.Utilization()

    def nvmlDeviceGetCurrentClocksThrottleReasons(self, handle):
        return self.nvmlClocksThrottleReasonGpuIdle


class FakeFieldPowerNVML(FakeNVML):
    NVML_SUCCESS = 0
    NVML_FI_DEV_POWER_AVERAGE = 185

    class FieldValue:
        nvmlReturn = 0

        def __init__(self):
            self.value = type("FieldUnion", (), {"uiVal": 52500})()

    def nvmlDeviceGetPowerUsage(self, handle):
        raise FakeNotSupported("direct power unsupported")

    def nvmlDeviceGetFieldValues(self, handle, fields):
        self.asserted_fields = fields
        return [self.FieldValue()]


def capability(*, counter=True, unsupported=False, uuid="GPU-fixture-0", pci="00000000:01:00.0"):
    error = None
    if not counter:
        error = {
            "error_type": "FakeNotSupported" if unsupported else "FakeFailure",
            "error_message": "fixture",
            "not_supported": unsupported,
        }
    return {
        "identity": {"uuid": uuid, "pci_bus_id": pci, "gpu_name": "fixture"},
        "capabilities": {
            "total_energy_supported": counter,
            "power_supported": True,
            "total_energy_error": error,
        },
    }


def record(seq, wall, mono, *, energy=1000, power=50.0, status="success", uuid="GPU-fixture-0", pci="00000000:01:00.0", late=False, field_errors=None):
    return {
        "endpoint": "P0",
        "sequence": seq,
        "schedule_index": seq,
        "scheduled_local_monotonic_s": mono,
        "actual_start_local_monotonic_s": mono if status != "missed" else None,
        "actual_finish_local_monotonic_s": mono + 0.001 if status != "missed" else None,
        "actual_start_wall_s": wall if status != "missed" else None,
        "actual_finish_wall_s": wall + 0.001 if status != "missed" else None,
        "query_latency_s": 0.001 if status != "missed" else None,
        "start_drift_s": 0.001 if status != "missed" else None,
        "late": late,
        "missed": status == "missed",
        "status": status,
        "gpu_uuid": uuid,
        "pci_bus_id": pci,
        "power_w": power,
        "total_energy_mj": energy,
        "field_errors": field_errors or {},
    }


def summarize(records, cap=None, window=None, **limits):
    return summarize_window(
        "P0", records,
        window or WindowSpec("workload", 10.0, 12.0, True, 2, 256, 128),
        cap or capability(),
        max_gap_s=limits.get("max_gap_s", 1.1),
        min_coverage_ratio=limits.get("min_coverage_ratio", 0.9),
        max_boundary_gap_s=limits.get("max_boundary_gap_s", 0.2),
    )


class ReadOnlyNVMLTests(unittest.TestCase):
    def test_monitor_cli_accepts_generic_multi_endpoint_ids(self):
        args = build_parser().parse_args([
            "monitor", "--endpoint", "P1", "--role", "prefill",
            "--output", "samples.jsonl",
            "--capability-output", "capability.json",
        ])
        self.assertEqual(args.endpoint, "P1")

    def test_counter_supported_identity_and_read_only_queries(self):
        binding = FakeNVML(energies=[1000, 1100])
        with ReadOnlyNVMLSource(
            "P0", "prefill", "GPU-fixture-0",
            expected_pci_bus_id="00000000:01:00.0",
            expected_gpu_name="L40S", binding=binding, hostname="p-node",
        ) as source:
            self.assertTrue(source.capabilities.total_energy_supported)
            sample = source.query()
            self.assertEqual(sample["total_energy_mj"], 1100)
            self.assertEqual(sample["power_w"], 50.0)
            self.assertEqual(source.identity.nvml_index, 0)
        self.assertEqual(binding.initialized, 1)
        self.assertEqual(binding.shutdown, 1)
        self.assertFalse(binding.setter_accessed)

    def test_counter_unsupported_is_independent_of_power_capability(self):
        binding = FakeNVML(energy_supported=False)
        with ReadOnlyNVMLSource("D0", "decode", "1", binding=binding) as source:
            self.assertFalse(source.capabilities.total_energy_supported)
            self.assertTrue(source.capabilities.power_supported)
            self.assertTrue(source.capabilities.total_energy_error["not_supported"])
            self.assertIsNone(source.query()["total_energy_mj"])

    def test_multi_endpoint_identity_and_throttle_reasons_are_read_only(self):
        binding = FakeNVML()
        with ReadOnlyNVMLSource("P1", "prefill", "0", binding=binding) as source:
            sample = source.query()
        self.assertEqual(sample["clock_throttle_reasons"], ["gpu_idle"])
        self.assertFalse(sample["invalidating_thermal_or_hw_slowdown"])
        self.assertFalse(binding.setter_accessed)

    def test_uuid_pci_mapping_and_mismatch_fail_loudly(self):
        with ReadOnlyNVMLSource("P0", "prefill", "00000000:01:00.0", binding=FakeNVML()) as source:
            self.assertEqual(source.identity.uuid, "GPU-fixture-0")
        with self.assertRaisesRegex(NVMLReadOnlyError, "UUID mismatch"):
            ReadOnlyNVMLSource(
                "P0", "prefill", "0", expected_uuid="GPU-wrong", binding=FakeNVML()
            )
        with self.assertRaisesRegex(NVMLReadOnlyError, "one unambiguous"):
            ReadOnlyNVMLSource("P0", "prefill", "0,1", binding=FakeNVML())

    def test_power_average_field_is_read_only_compatibility_source(self):
        binding = FakeFieldPowerNVML(energy_supported=False)
        with ReadOnlyNVMLSource("P0", "prefill", "0", binding=binding) as source:
            self.assertTrue(source.capabilities.power_supported)
            self.assertEqual(
                source.capabilities.power_query_source,
                "nvmlFieldValue_power_average",
            )
            self.assertEqual(source.query()["power_w"], 52.5)
            self.assertEqual(binding.asserted_fields, [185])
        self.assertFalse(binding.setter_accessed)


class WindowAnalysisTests(unittest.TestCase):
    def test_hardware_counter_monotonic_primary_energy(self):
        result = summarize([
            record(0, 10.05, 20.0, energy=1000, power=40),
            record(1, 11.0, 20.95, energy=31000, power=50),
            record(2, 11.95, 21.9, energy=81000, power=60),
        ])
        self.assertTrue(result["valid"])
        self.assertEqual(result["method"], "hardware_counter")
        self.assertAlmostEqual(result["gross_gpu_energy_j"], 80.0)
        self.assertAlmostEqual(result["energy_j_per_request"], 40.0)

    def test_unsupported_counter_uses_labelled_trapezoidal_fallback(self):
        result = summarize([
            record(0, 10.05, 20.0, energy=None, power=40),
            record(1, 11.0, 20.95, energy=None, power=50),
            record(2, 11.95, 21.9, energy=None, power=60),
        ], capability(counter=False, unsupported=True))
        self.assertTrue(result["valid"])
        self.assertEqual(result["method"], "power_integral_estimate")
        self.assertAlmostEqual(result["gross_gpu_energy_j"], 95.0)

    def test_counter_decrease_invalidates_without_fallback(self):
        result = summarize([
            record(0, 10.05, 20.0, energy=9000),
            record(1, 11.0, 20.95, energy=8000),
            record(2, 11.95, 21.9, energy=10000),
        ])
        self.assertFalse(result["valid"])
        self.assertEqual(result["method"], "hardware_counter")
        self.assertIn("energy_counter_decreased_or_reset", result["invalidity_reasons"])
        self.assertIsNone(result["gross_gpu_energy_j"])

    def test_missing_malformed_nonfinite_and_identity_change_invalidate(self):
        variants = [
            [record(0, 10.05, 20, energy=1000), record(1, 11, 21, energy=None), record(2, 11.95, 21.9, energy=3000)],
            [record(0, 10.05, 20, energy=1000), record(1, 11, 21, energy=float("nan")), record(2, 11.95, 21.9, energy=3000)],
            [record(0, 10.05, 20, energy=1000), record(1, 11, 21, energy=2000, uuid="GPU-other"), record(2, 11.95, 21.9, energy=3000)],
        ]
        for records in variants:
            with self.subTest(records=records):
                self.assertFalse(summarize(records)["valid"])

    def test_irregular_acceptable_spacing_and_excessive_gap(self):
        records = [
            record(0, 10.05, 20.0, energy=None, power=10),
            record(1, 10.55, 20.5, energy=None, power=20),
            record(2, 11.95, 21.9, energy=None, power=30),
        ]
        okay = summarize(records, capability(counter=False, unsupported=True), max_gap_s=1.5)
        self.assertTrue(okay["valid"])
        self.assertAlmostEqual(okay["gross_gpu_energy_j"], 42.5)
        bad = summarize(records, capability(counter=False, unsupported=True), max_gap_s=1.0)
        self.assertFalse(bad["valid"])
        self.assertIn("sampling_gap_exceeds_limit", bad["invalidity_reasons"])

    def test_separate_windows_and_warmup_exclusion(self):
        records = [record(i, 10 + i, 20 + i, energy=1000 + i * 1000) for i in range(7)]
        windows = [
            WindowSpec("idle", 10, 12, False),
            WindowSpec("warmup", 12, 14, False, 1, 128, 128),
            WindowSpec("workload", 14, 16, True, 2, 256, 256),
        ]
        results = [summarize(records, window=item, max_boundary_gap_s=0.01) for item in windows]
        self.assertEqual([item["window"] for item in results], ["idle", "warmup", "workload"])
        self.assertEqual([item["included_in_reported_workload"] for item in results], [False, False, True])

    def test_invalid_denominators_suppress_normalization(self):
        result = summarize(
            [record(0, 10.05, 20, energy=1000), record(1, 11, 20.95, energy=2000), record(2, 11.95, 21.9, energy=3000)],
            window=WindowSpec("workload", 10, 12, True, 0, 0, 0),
        )
        self.assertTrue(result["valid"])
        self.assertIsNone(result["energy_j_per_request"])
        self.assertIsNone(result["energy_j_per_output_token"])

    def test_idle_adjustment_preserves_gross_and_does_not_clamp_negative(self):
        workload = {"endpoint": "P0", "valid": True, "duration_s": 2.0, "gross_gpu_energy_j": 10.0}
        idle = {"endpoint": "P0", "valid": True, "mean_power_w": 8.0}
        add_idle_adjustment(workload, idle)
        self.assertEqual(workload["gross_gpu_energy_j"], 10.0)
        self.assertEqual(workload["idle_adjusted_incremental_estimate_j"], -6.0)

    def test_combined_gpu_sum_counts_logical_requests_once(self):
        windows = [
            {"endpoint": "P0", "window": "workload", "valid": True, "gross_gpu_energy_j": 30.0},
            {"endpoint": "D0", "window": "workload", "valid": True, "gross_gpu_energy_j": 70.0},
        ]
        result = aggregate_endpoint_workload(windows, logical_requests=5, output_tokens=640)
        self.assertEqual(result["gross_gpu_energy_j"], 100.0)
        self.assertEqual(result["logical_request_count"], 5)
        self.assertEqual(result["energy_j_per_request"], 20.0)


class SamplerTests(unittest.TestCase):
    class Source:
        class Identity:
            uuid = "GPU-fixture-0"
            pci_bus_id = "00000000:01:00.0"

        class Capabilities:
            total_energy_supported = True
            power_supported = True

        identity = Identity()
        capabilities = Capabilities()

        def __init__(self, delay=0.0):
            self.delay = delay
            self.value = 0

        def query(self):
            if self.delay:
                time.sleep(self.delay)
            self.value += 1
            return {
                "gpu_uuid": self.identity.uuid, "pci_bus_id": self.identity.pci_bus_id,
                "power_w": 50.0, "total_energy_mj": self.value * 100,
                "field_errors": {},
            }

    def test_late_and_missed_slots_are_explicit(self):
        stream = io.StringIO()
        sampler = FixedPeriodEnergySampler("P0", self.Source(0.025), 0.01, 0.0)
        sampler.run(stream, duration_s=0.055)
        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertTrue(any(item["status"] == "missed" for item in records))
        self.assertTrue(any(item["late"] for item in records if item["status"] == "success"))
        indices = [item["schedule_index"] for item in records]
        self.assertEqual(indices, list(range(len(indices))))

    def test_independent_endpoint_cadence_when_other_is_blocked(self):
        streams = {"P0": io.StringIO(), "D0": io.StringIO()}
        samplers = {
            "P0": FixedPeriodEnergySampler("P0", self.Source(0.03), 0.01, 0.002),
            "D0": FixedPeriodEnergySampler("D0", self.Source(0.001), 0.01, 0.002),
        }
        threads = [threading.Thread(target=samplers[key].run, args=(streams[key],), kwargs={"duration_s": 0.07}) for key in samplers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        p = [json.loads(line) for line in streams["P0"].getvalue().splitlines()]
        d = [json.loads(line) for line in streams["D0"].getvalue().splitlines()]
        self.assertTrue(any(item["missed"] for item in p))
        self.assertFalse(any(item["missed"] for item in d))
        self.assertGreater(sum(item["status"] == "success" for item in d), sum(item["status"] == "success" for item in p))

    def test_graceful_stop_preserves_flushed_partial_jsonl(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "samples.jsonl"
            sampler = FixedPeriodEnergySampler("P0", self.Source(), 0.01, 0.002)
            with path.open("x", encoding="utf-8", buffering=1) as stream:
                thread = threading.Thread(target=sampler.run, args=(stream,))
                thread.start()
                time.sleep(0.035)
                sampler.stop()
                thread.join(timeout=1)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertGreaterEqual(len(records), 2)
            self.assertFalse(thread.is_alive())

    def test_slow_shared_file_flush_does_not_shift_sampling_deadlines(self):
        class SlowStream(io.StringIO):
            def flush(self):
                time.sleep(0.025)
                return super().flush()

        raw = SlowStream()
        stream = AsyncFlushingStream(raw)
        try:
            sampler = FixedPeriodEnergySampler("D0", self.Source(), 0.01, 0.003)
            sampler.run(stream, duration_s=0.055)
        finally:
            stream.close()
        records = [json.loads(line) for line in raw.getvalue().splitlines()]
        self.assertGreaterEqual(len(records), 5)
        self.assertFalse(any(item["missed"] for item in records))

    def test_async_writer_batches_and_drains_large_backlog(self):
        class CountingStream(io.StringIO):
            def __init__(self):
                super().__init__()
                self.flush_count = 0

            def flush(self):
                self.flush_count += 1
                time.sleep(0.001)
                return super().flush()

        raw = CountingStream()
        stream = AsyncFlushingStream(raw)
        records = [json.dumps({"sequence": index}) + "\n" for index in range(2000)]
        for record_text in records:
            stream.write(record_text)
        started = time.monotonic()
        stream.close()
        elapsed = time.monotonic() - started
        self.assertEqual(raw.getvalue(), "".join(records))
        self.assertLessEqual(raw.flush_count, 9)
        self.assertLess(elapsed, 1.0)


class RuntimeGuardTests(unittest.TestCase):
    def test_cross_node_monitor_stop_uses_control_pipe(self):
        class FakeControl:
            def __init__(self, process):
                self.process = process
                self.value = ""

            def write(self, value):
                self.value += value

            def flush(self):
                self.process.returncode = 0

            def close(self):
                return None

        class FakeProcess:
            def __init__(self):
                self.returncode = None
                self.stdin = FakeControl(self)

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                del timeout
                return self.returncode

        with tempfile.TemporaryDirectory() as temp:
            process = FakeProcess()
            log = io.StringIO()
            stop_file = Path(temp) / "stop.requested"
            errors = _stop_monitors(
                {"P0": process}, [log], {"P0": stop_file},
                marker_grace_s=0.0, shutdown_timeout_s=0.1,
            )
            stop_text = stop_file.read_text(encoding="utf-8")
        self.assertEqual(errors, [])
        self.assertEqual(process.stdin.value, "stop\n")
        self.assertEqual(stop_text, "stop\n")

    def test_phase3b_runtime_has_no_mutation_symbols_or_legacy_monitor_import(self):
        root = Path(__file__).resolve().parents[1]
        paths = [
            root / "paper/scripts/xpyd/nvml_readonly.py",
            root / "paper/scripts/xpyd/phase3b_energy.py",
        ]
        source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        forbidden = (
            "nvmlDevice" + "Set",
            "nvmlDevice" + "Reset",
            "from gpu_" + "monitor",
            "import gpu_" + "monitor",
            "nvidia-smi " + "-lgc",
            "nvidia-smi " + "-lmc",
            "nvidia-smi " + "-pl",
            "nvidia-smi " + "-pm",
            "nvidia-smi " + "-rgc",
        )
        for token in forbidden:
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
