"""CPU-only regression tests for the physical disaggregated proxy path."""

import asyncio
import inspect
import json
from pathlib import Path
import subprocess
import sys
import unittest

from xpyd.disagg_proxy import (
    DisaggProxyCore,
    EndpointTransport,
    MultiEndpointDisaggProxyCore,
    ProxyConfig,
    ProxyUpstreamError,
    ingress_logical_request_id,
)
from xpyd.compatibility import CompatibilityTable, EndpointPairCompatibility
from xpyd.registry import EndpointRegistry
from xpyd.types import EndpointSpec, EndpointState, LifecycleState


class StepClock:
    def __init__(self, step=0.001):
        self.value = 100.0
        self.step = step

    def __call__(self):
        self.value += self.step
        return self.value


class FakeContent:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.completed = False

    async def iter_any(self):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk
        self.completed = True


class FakeResponse:
    def __init__(self, status=200, content_type="application/json", body=b"{}", chunks=()):
        self.status = status
        self.headers = {"Content-Type": content_type}
        self.body = body
        self.content = FakeContent(chunks)
        self.read_calls = 0
        self.released = False

    async def read(self):
        self.read_calls += 1
        return self.body

    def release(self):
        self.released = True


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.posts = []
        self.closed = False

    async def post(self, url, json, headers):
        self.posts.append({"url": url, "json": json, "headers": headers})
        return self.response

    async def close(self):
        self.closed = True


class SessionFactory:
    def __init__(self, responses):
        self.sessions = [FakeSession(response) for response in responses]
        self.index = 0

    def __call__(self):
        session = self.sessions[self.index]
        self.index += 1
        return session


def config():
    return ProxyConfig(
        prefill_http_host="p",
        decode_http_host="d",
        prefill_addr_host="10.0.0.1",
        decode_addr_host="10.0.0.2",
    )


class DisaggProxyStreamingTests(unittest.TestCase):
    def test_multi_endpoint_round_robin_uses_only_exact_compatible_pairs(self):
        registry = EndpointRegistry()
        transports = {}
        for endpoint_id, role, gpu_id, http_port, kv_port in (
            ("P0", "prefill", 0, 8100, 14579),
            ("P1", "prefill", 1, 8101, 14580),
            ("D0", "decode", 2, 8200, 14579),
            ("D1", "decode", 3, 8201, 14580),
        ):
            spec = EndpointSpec(
                endpoint_id, role, "fixture", "node", (gpu_id,), 1,
                kv_connector="test-kv",
            )
            registry.register(spec, EndpointState(
                endpoint_id, 1500, LifecycleState.ACTIVE, True,
            ))
            transports[endpoint_id] = EndpointTransport(
                endpoint_id, "host", http_port, "10.0.0.1", kv_port,
            )
        pairs = (("P0", "D0"), ("P0", "D1"), ("P1", "D0"), ("P1", "D1"))
        table = CompatibilityTable(endpoint_pairs=tuple(
            EndpointPairCompatibility(p, d, "test-kv", 1, 1, True, "fixture")
            for p, d in pairs
        ))
        factory = SessionFactory([
            response
            for _ in pairs
            for response in (
                FakeResponse(),
                FakeResponse(content_type="application/json", body=b"{}"),
            )
        ])
        core = MultiEndpointDisaggProxyCore(
            registry, table, transports, pairs, factory, clock=StepClock(),
            diagnostic_sink=lambda _: None,
        )

        prepared = [
            asyncio.run(core.prepare({"prompt": "x", "stream": False}, "r%d" % index))
            for index in range(4)
        ]
        self.assertEqual(
            [(item.diagnostics.selected_prefill_endpoint_id, item.diagnostics.selected_decode_endpoint_id) for item in prepared],
            list(pairs),
        )
        self.assertIn("___prefill_addr_10.0.0.1:14579", prepared[0].diagnostics.vllm_request_id)
        self.assertIn("___decode_addr_10.0.0.1:14580", prepared[1].diagnostics.vllm_request_id)
        self.assertEqual(prepared[3].headers["X-Xpyd-Prefill-Endpoint"], "P1")
        self.assertEqual(prepared[3].headers["X-Xpyd-Decode-Endpoint"], "D1")

    def test_ingress_reuses_one_id_and_rejects_conflicting_domains(self):
        self.assertEqual(
            ingress_logical_request_id({"X-Request-Id": "client-1"}),
            "client-1",
        )
        self.assertEqual(
            ingress_logical_request_id({
                "X-Request-Id": "client-1",
                "X-Xpyd-Logical-Request-Id": "client-1",
            }),
            "client-1",
        )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            ingress_logical_request_id({
                "X-Request-Id": "client-1",
                "X-Xpyd-Logical-Request-Id": "client-2",
            })

    def test_proxy_module_entrypoint_does_not_shadow_stdlib_types(self):
        repo_root = Path(__file__).resolve().parents[1]
        env = {"PYTHONPATH": str(repo_root / "paper/scripts")}
        result = subprocess.run(
            [sys.executable, "-m", "xpyd.disagg_proxy", "--help"],
            cwd=repo_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--prefill-http-host", result.stdout)

    def test_real_stream_preserves_prefill_contract_and_forwards_incrementally(self):
        chunks = [
            b'data: {"choices":[{"text":"a"}]}\n\n',
            b'data: {"choices":[],"usage":{"completion_tokens":2}}\n\n',
            b"data: [DONE]\n\n",
        ]
        prefill = FakeResponse()
        decode = FakeResponse(content_type="text/event-stream", chunks=chunks)
        factory = SessionFactory([prefill, decode])
        diagnostics = []
        core = DisaggProxyCore(
            config(), factory, clock=StepClock(), diagnostic_sink=diagnostics.append,
        )

        async def scenario():
            prepared = await core.prepare(
                {
                    "model": "m",
                    "prompt": "hello",
                    "max_tokens": 2,
                    "stream": True,
                },
                logical_request_id="trace-request-007",
            )
            self.assertIsNotNone(prepared.stream)
            first = await prepared.stream.__anext__()
            self.assertEqual(first, chunks[0])
            self.assertFalse(decode.content.completed)
            self.assertIsNone(
                prepared.diagnostics.timestamps_monotonic_s["response_completed"]
            )
            remaining = [chunk async for chunk in prepared.stream]
            return prepared, [first] + remaining

        prepared, forwarded = asyncio.run(scenario())
        prefill_post = factory.sessions[0].posts[0]
        decode_post = factory.sessions[1].posts[0]

        self.assertTrue(prefill_post["url"].endswith("/v1/completions"))
        self.assertFalse(prefill_post["json"]["stream"])
        self.assertEqual(prefill_post["json"]["max_tokens"], 1)
        self.assertTrue(decode_post["json"]["stream"])
        self.assertTrue(decode_post["json"]["stream_options"]["include_usage"])
        self.assertEqual(
            prefill_post["headers"]["X-Request-Id"],
            decode_post["headers"]["X-Request-Id"],
        )
        self.assertEqual(
            prefill_post["headers"]["X-Xpyd-Logical-Request-Id"],
            "trace-request-007",
        )
        self.assertEqual(
            decode_post["headers"]["X-Xpyd-Logical-Request-Id"],
            "trace-request-007",
        )
        self.assertTrue(
            prefill_post["headers"]["X-Request-Id"].endswith(
                "_trace-request-007"
            )
        )
        self.assertEqual(prepared.diagnostics.request_id, "trace-request-007")
        self.assertEqual(
            prepared.headers["X-Xpyd-Logical-Request-Id"],
            "trace-request-007",
        )
        self.assertEqual(forwarded, chunks)
        self.assertEqual(decode.read_calls, 0)
        self.assertTrue(decode.content.completed)
        self.assertEqual(prepared.diagnostics.outcome, "completed")
        self.assertTrue(prepared.diagnostics.decode_stream_available)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]["request_id"], "trace-request-007")
        self.assertEqual(
            diagnostics[0]["logical_request_id"], "trace-request-007"
        )
        self.assertIn("prefill_started", diagnostics[0]["timestamps_wall_s"])
        self.assertIn("prefill_completed", diagnostics[0]["timestamps_wall_s"])
        self.assertLess(
            diagnostics[0]["timestamps_wall_s"]["prefill_started"],
            diagnostics[0]["timestamps_wall_s"]["prefill_completed"],
        )
        self.assertEqual(
            diagnostics[0]["vllm_request_id"],
            prefill_post["headers"]["X-Request-Id"],
        )
        self.assertIsNotNone(
            diagnostics[0]["durations_ms"]["first_chunk_forwarding_delay"]
        )
        self.assertNotIn("sleep", inspect.getsource(core._real_decode_stream))

    def test_ingress_generates_one_logical_id_when_client_omits_it(self):
        factory = SessionFactory([
            FakeResponse(),
            FakeResponse(content_type="application/json", body=b"{}"),
        ])
        diagnostics = []
        core = DisaggProxyCore(
            config(), factory, clock=StepClock(), diagnostic_sink=diagnostics.append,
        )

        prepared = asyncio.run(core.prepare({"prompt": "hello", "stream": False}))
        prefill_headers = factory.sessions[0].posts[0]["headers"]
        decode_headers = factory.sessions[1].posts[0]["headers"]

        self.assertTrue(prepared.diagnostics.request_id)
        self.assertEqual(
            prefill_headers["X-Xpyd-Logical-Request-Id"],
            prepared.diagnostics.request_id,
        )
        self.assertEqual(prefill_headers, decode_headers)
        self.assertEqual(diagnostics[0]["request_id"], prepared.diagnostics.request_id)

    def test_nonstreaming_decode_fails_closed_without_replay(self):
        prefill = FakeResponse()
        decode = FakeResponse(
            content_type="application/json",
            body=b'{"choices":[{"token_ids":[1,2]}]}',
        )
        factory = SessionFactory([prefill, decode])
        diagnostics = []
        core = DisaggProxyCore(
            config(), factory, clock=StepClock(), diagnostic_sink=diagnostics.append,
        )

        prepared = asyncio.run(core.prepare({
            "prompt": "hello", "max_tokens": 2, "stream": True,
        }))

        self.assertEqual(prepared.status_code, 502)
        self.assertIsNone(prepared.stream)
        self.assertEqual(decode.read_calls, 1)
        body = json.loads(prepared.body)
        self.assertFalse(body["decode_stream_available"])
        self.assertFalse(body["client_ttft_valid"])
        self.assertFalse(body["client_tpot_valid"])
        self.assertFalse(body["client_itl_valid"])
        self.assertEqual(
            prepared.diagnostics.outcome, "decode_stream_unavailable"
        )
        self.assertIsNone(
            diagnostics[0]["durations_ms"]["full_decode_stream"]
        )

    def test_decode_http_error_fails_clearly(self):
        factory = SessionFactory([
            FakeResponse(),
            FakeResponse(status=500, body=b"decode exploded"),
        ])
        core = DisaggProxyCore(
            config(), factory, clock=StepClock(), diagnostic_sink=lambda _: None,
        )

        with self.assertRaisesRegex(ProxyUpstreamError, "HTTP 500"):
            asyncio.run(core.prepare({"prompt": "x", "stream": True}))

    def test_midstream_error_is_not_converted_to_synthetic_success(self):
        decode = FakeResponse(
            content_type="text/event-stream",
            chunks=[b'data: {"choices":[{"text":"a"}]}\n\n', RuntimeError("boom")],
        )
        diagnostics = []
        core = DisaggProxyCore(
            config(), SessionFactory([FakeResponse(), decode]),
            clock=StepClock(), diagnostic_sink=diagnostics.append,
        )

        async def scenario():
            prepared = await core.prepare({"prompt": "x", "stream": True})
            first = await prepared.stream.__anext__()
            self.assertIn(b'"text":"a"', first)
            with self.assertRaisesRegex(ProxyUpstreamError, "decode SSE stream failed"):
                await prepared.stream.__anext__()

        asyncio.run(scenario())
        self.assertEqual(diagnostics[0]["outcome"], "stream_error")


if __name__ == "__main__":
    unittest.main()
