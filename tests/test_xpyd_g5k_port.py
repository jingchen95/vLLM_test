"""CPU-only tests for the Grid'5000 port: config generation and node execution."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from xpyd.phase3d_control import NvidiaSmiClockBackend, Phase3DError
from xpyd.stage_e0_regime import load_e0_configs
from xpyd.stage_e0c_capacity import load_configs as load_e0c_configs

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "paper/scripts/g5k/gen_g5k_configs.py"


def generate(mode: str, out_dir: Path) -> tuple[Path, Path]:
    argv = [
        sys.executable, GEN.as_posix(), "--mode", mode, "--out-dir", out_dir.as_posix(), "--repo", ROOT.as_posix(),
        "--p-node", "chicoree-1.lille.grid5000.fr", "--d-node", "chuc-3.lille.grid5000.fr",
        "--p-ip", "172.16.47.1", "--d-ip", "172.16.47.3", "--proxy-host", "chicoree-1.lille.grid5000.fr",
        "--p-gpu-name", "NVIDIA H200 NVL", "--d-gpu-name", "NVIDIA A100-SXM4-40GB",
        "--p-levels", "990,1485,1980", "--d-levels", "705,1065,1410",
        "--p-mem-mhz", "2619", "--d-mem-mhz", "1215",
        "--model-dir", "/home/jchen/g5k/models/Mistral-7B-v0.1",
        "--result-root", (out_dir / "results").as_posix(),
    ]
    output = subprocess.run(argv, check=True, text=True, stdout=subprocess.PIPE).stdout.split()
    return Path(output[0]), Path(output[1])


class G5kPortTests(unittest.TestCase):
    def test_generated_e0_configs_pass_the_frozen_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime_path, experiment_path = generate("e0", Path(tmp))
            runtime, experiment = load_e0_configs(runtime_path, experiment_path)
            levels = experiment["stagec1"]["frequency_levels_mhz"]
            self.assertEqual(levels["P0"], {"LOW": 990, "MID": 1485, "HIGH": 1980})
            self.assertEqual(levels["D1"], {"LOW": 705, "MID": 1065, "HIGH": 1410})
            self.assertEqual(experiment["platform"], "g5k")
            self.assertEqual(experiment["stagec1"]["slo"], {"ttft_ms": 600.0, "tpot_ms": 65.0, "attainment": 0.95})
            endpoints = {item["endpoint_id"]: item for item in runtime["endpoints"]}
            # Proxy runs on the P node: P endpoints are loopback, D endpoints use the D node IP.
            self.assertEqual(endpoints["P0"]["http_uri"], "http://127.0.0.1:8100")
            self.assertEqual(endpoints["D1"]["http_uri"], "http://172.16.47.3:8201")
            self.assertEqual(endpoints["P1"]["expected_gpu_name"], "H200")
            self.assertEqual(endpoints["D0"]["expected_gpu_name"], "A100")
            self.assertEqual(runtime["fixed_clocks"]["P0"]["graphics_mhz"], 1980)
            self.assertEqual(runtime["fixed_clocks"]["D0"]["memory_mhz"], 1215)
            self.assertEqual(runtime["tokenizer_model"], "/home/jchen/g5k/models/Mistral-7B-v0.1")
            self.assertEqual(runtime["model"], "mistralai/Mistral-7B-v0.1")

    def test_generated_e0c_configs_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime_path, workload_path = generate("e0c", Path(tmp))
            runtime, workload = load_e0c_configs(runtime_path, workload_path)
            self.assertEqual(workload["expected_topology"]["D0"], "chuc-3.lille.grid5000.fr")
            self.assertEqual(workload["stage_e0c"]["concurrency_ladder"][0], 4)

    def test_minerva_configs_still_require_minerva_topology(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime_path, experiment_path = generate("e0", Path(tmp))
            old = dict(os.environ)
            os.environ.update({
                "XPYD_PROXY_URI": "x", "XPYD_PROXY_DIAGNOSTICS_LOG": "/tmp/p.jsonl",
                "XPYD_P0_HTTP_HOST": "h", "XPYD_P1_HTTP_HOST": "h", "XPYD_D0_HTTP_HOST": "h",
                "XPYD_D1_HTTP_HOST": "h", "XPYD_P_ADDR_HOST": "h", "XPYD_D_ADDR_HOST": "h",
            })
            try:
                with self.assertRaises(Exception):
                    load_e0_configs(
                        ROOT / "paper/configs/xpyd_stageb_runtime_neptune_europa.json", experiment_path,
                    )
            finally:
                os.environ.clear()
                os.environ.update(old)

    def test_clock_backend_node_exec_switch(self):
        default = NvidiaSmiClockBackend(runner=lambda c: None)
        self.assertEqual(default._node_command("n1", "echo")[0], "srun")
        oar = NvidiaSmiClockBackend(runner=lambda c: None, node_exec="oarsh", sudo="sudo-g5k")
        self.assertEqual(oar._node_command("n1", "echo"), ["oarsh", "n1", "/bin/bash", "-lc", "echo"])
        self.assertEqual(oar.sudo, "sudo-g5k")
        with self.assertRaises(Phase3DError):
            NvidiaSmiClockBackend(runner=lambda c: None, node_exec="rsh")


class MonitorLaunchTests(unittest.TestCase):
    def launch(self, node, host, node_exec):
        from unittest import mock
        import xpyd.phase3b_energy as energy
        endpoint = {"endpoint_id": "P0", "role": "prefill", "node": node, "cuda_visible_device": "0",
                    "expected_gpu_name": "A100"}
        sampling = {"period_s": 0.2, "late_tolerance_s": 0.05}
        captured = {}

        def fake_popen(command, **kwargs):
            captured["command"] = command
            return mock.Mock()

        env = dict(os.environ)
        env.pop("XPYD_NODE_EXEC", None)
        if node_exec:
            env["XPYD_NODE_EXEC"] = node_exec
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(energy.socket, "gethostname", return_value=host), \
                mock.patch.object(energy.subprocess, "Popen", side_effect=fake_popen), \
                mock.patch.dict(os.environ, env, clear=True):
            _, log = energy._launch_monitor(endpoint, Path(tmp), sampling)
            log.close()
        return captured["command"]

    def test_fqdn_of_local_host_runs_locally(self):
        command = self.launch("chuc-2.lille.grid5000.fr", "chuc-2", "oarsh")
        self.assertNotIn(command[0], ("srun", "oarsh"))

    def test_remote_node_uses_oarsh_on_g5k(self):
        command = self.launch("chuc-6.lille.grid5000.fr", "chuc-2", "oarsh")
        self.assertEqual(command[:2], ["oarsh", "chuc-6.lille.grid5000.fr"])

    def test_minerva_remote_node_still_uses_srun(self):
        command = self.launch("europa", "neptune", None)
        self.assertEqual(command[0], "srun")
        self.assertEqual(self.launch("neptune", "neptune", None)[0], sys.executable)


if __name__ == "__main__":
    unittest.main()
