"""CPU-only tests for exact streaming completion-token accounting."""

import asyncio
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


try:
    import aiohttp  # noqa: F401
except ModuleNotFoundError:
    aiohttp_stub = types.ModuleType("aiohttp")
    aiohttp_stub.ClientSession = object
    aiohttp_stub.ClientTimeout = object
    aiohttp_stub.TCPConnector = object
    sys.modules["aiohttp"] = aiohttp_stub

try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoTokenizer = object
    sys.modules["transformers"] = transformers_stub

import replay_synthetic_trace as replay
from replay_synthetic_trace import (
    ClientLoadTracker, TraceRequest, consume_sse_completion, load_trace,
)


class FakeContent:
    def __init__(self, events):
        self.events = events
        self.consumed = []

    async def __aiter__(self):
        for event in self.events:
            self.consumed.append(event)
            yield event.encode("utf-8")


class FakeResponse:
    status = 200

    def __init__(self, events, headers=None):
        self.content = FakeContent(events)
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def text(self):
        return ""


class FakeSession:
    def __init__(self, events):
        self.response = FakeResponse(events)
        self.posts = []

    def post(self, url, json, headers=None):
        self.posts.append({"url": url, "json": json, "headers": headers})
        return self.response


class FakeTokenizer:
    def __init__(self, retokenized_count):
        self.retokenized_count = retokenized_count
        self.encode_calls = 0

    def encode(self, text, add_special_tokens=False):
        self.encode_calls += 1
        return list(range(self.retokenized_count))


def sse(value):
    return "data: %s\n\n" % json.dumps(value)


class StreamingTokenAccountingTests(unittest.TestCase):
    def test_replay_exposes_waiting_when_offered_load_exceeds_admission(self):
        class FakeClientSession:
            def __init__(self, **_):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        async def fake_consume(*args, **kwargs):
            await asyncio.sleep(0.01)
            now = replay.time.time()
            return {
                "ok": True,
                "send_unix_s": now,
                "complete_unix_s": now,
                "e2e_latency_ms": 10.0,
                "ttft_ms": 2.0,
                "tpot_ms": 1.0,
                "mean_itl_ms": 1.0,
                "completion_tokens": 2,
                "completion_token_source": "server_usage",
                "decode_stream_available": True,
                "client_ttft_valid": True,
                "client_tpot_valid": True,
                "client_itl_valid": True,
                "logical_request_id_propagated": True,
            }

        requests = [
            TraceRequest("r%d" % index, 0.0, 128, 128, "SS", "burst")
            for index in range(4)
        ]
        with mock.patch.object(replay.aiohttp, "ClientSession", FakeClientSession), \
             mock.patch.object(replay.aiohttp, "ClientTimeout", lambda **_: object()), \
             mock.patch.object(replay.aiohttp, "TCPConnector", lambda **_: object()), \
             mock.patch.object(replay, "consume_sse_completion", fake_consume):
            summary = asyncio.run(replay.replay_trace(
                requests, "http://proxy", "model", {128: "prompt"}, object(),
                max_concurrency=2, request_timeout_s=1.0, num_warmups=0,
            ))

        self.assertEqual(summary["maximum_client_in_flight_requests"], 2)
        self.assertGreaterEqual(summary["maximum_client_waiting_requests"], 2)
        delayed = [
            row for row in summary["request_results"]
            if row["client_admission_delay_ms"] > 1.0
        ]
        self.assertGreaterEqual(len(delayed), 2)
        self.assertEqual(summary["workload_class_counts"], {"SS": 4})

    def test_trace_context_columns_and_client_load_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            path.write_text(
                "request_id,arrival_time_s,input_len,output_len,workload_class,segment\n"
                "r0,0.0,128,512,SL,decode_burst\n",
                encoding="utf-8",
            )
            requests = load_trace(path)
        self.assertEqual(requests[0].workload_class, "SL")
        self.assertEqual(requests[0].segment, "decode_burst")

        tracker = ClientLoadTracker()
        tracker.arrive("r0", 1.0, 10.0)
        tracker.arrive("r1", 1.1, 10.1)
        tracker.admit("r0", 1.2, 10.2)
        tracker.complete("r0", 1.3, 10.3)
        tracker.admit("r1", 1.4, 10.4)
        tracker.complete("r1", 1.5, 10.5)
        self.assertEqual(tracker.maximum_waiting, 2)
        self.assertEqual(tracker.maximum_in_flight, 1)
        self.assertEqual(tracker.waiting, 0)
        self.assertEqual(tracker.in_flight, 0)
        self.assertEqual(
            [event["event"] for event in tracker.events],
            [
                "TRACE_ARRIVAL", "TRACE_ARRIVAL", "CLIENT_ADMIT",
                "REQUEST_COMPLETE", "CLIENT_ADMIT", "REQUEST_COMPLETE",
            ],
        )

    def test_server_usage_wins_over_lossy_text_retokenization(self):
        tokenizer = FakeTokenizer(retokenized_count=139)
        events = [
            sse({"choices": [{"text": "first"}]}),
            sse({"choices": [{"text": " second"}]}),
            sse({
                "choices": [],
                "usage": {
                    "prompt_tokens": 128,
                    "completion_tokens": 128,
                    "total_tokens": 256,
                },
            }),
            "data: [DONE]\n\n",
        ]

        result = asyncio.run(consume_sse_completion(
            FakeSession(events), "http://proxy/v1/completions", {}, tokenizer,
        ))

        self.assertEqual(result["completion_tokens"], 128)
        self.assertEqual(result["completion_token_source"], "server_usage")
        self.assertEqual(tokenizer.encode_calls, 0)
        self.assertIsNotNone(result["ttft_ms"])
        self.assertIsNotNone(result["tpot_ms"])

    def test_real_stream_headers_mark_latency_as_valid(self):
        tokenizer = FakeTokenizer(retokenized_count=2)
        events = [
            sse({"choices": [{"text": "a"}]}),
            sse({"choices": [{"text": "b"}]}),
            sse({"choices": [], "usage": {"completion_tokens": 2}}),
            "data: [DONE]\n\n",
        ]
        headers = {
            "X-Xpyd-Incoming-Client-Stream": "true",
            "X-Xpyd-Outgoing-Decode-Stream": "true",
            "X-Xpyd-Decode-Content-Type": "text/event-stream",
            "X-Xpyd-Decode-Stream-Available": "true",
            "X-Xpyd-Logical-Request-Id": "trace-request-007",
        }

        session = FakeSession(events)
        session.response.headers = headers
        result = asyncio.run(consume_sse_completion(
            session,
            "http://proxy/v1/completions",
            {},
            tokenizer,
            logical_request_id="trace-request-007",
        ))
        self.assertTrue(result["decode_stream_available"])
        self.assertTrue(result["client_ttft_valid"])
        self.assertTrue(result["client_tpot_valid"])
        self.assertTrue(result["client_itl_valid"])
        self.assertEqual(result["streamed_text_event_count"], 2)
        self.assertEqual(
            session.posts[0]["headers"]["X-Request-Id"],
            "trace-request-007",
        )
        self.assertEqual(result["logical_request_id"], "trace-request-007")
        self.assertEqual(result["echoed_logical_request_id"], "trace-request-007")
        self.assertTrue(result["logical_request_id_propagated"])

    def test_live_ttft_and_client_load_events_are_emitted_immediately(self):
        tokenizer = FakeTokenizer(retokenized_count=1)
        events = [
            sse({"choices": [{"text": "a"}]}),
            sse({"choices": [], "usage": {"completion_tokens": 1}}),
            "data: [DONE]\n\n",
        ]
        safety_events = []
        result = asyncio.run(consume_sse_completion(
            FakeSession(events), "http://proxy/v1/completions", {}, tokenizer,
            logical_request_id="live-1", safety_event_sink=safety_events.append,
            request_context={"workload_class": "LS", "input_len": 2048},
        ))
        self.assertTrue(result["ok"])
        self.assertEqual(len(safety_events), 1)
        self.assertEqual(safety_events[0]["event"], "TTFT_OBSERVED")
        self.assertEqual(safety_events[0]["logical_request_id"], "live-1")
        self.assertEqual(safety_events[0]["workload_class"], "LS")

        load_events = []
        tracker = ClientLoadTracker(load_events.append)
        tracker.arrive("live-1", 1.0, 10.0)
        tracker.admit("live-1", 1.1, 10.1)
        tracker.complete("live-1", 1.2, 10.2)
        self.assertEqual(
            [row["event"] for row in load_events],
            ["TRACE_ARRIVAL", "CLIENT_ADMIT", "REQUEST_COMPLETE"],
        )

    def test_missing_usage_is_explicitly_marked_as_fallback(self):
        tokenizer = FakeTokenizer(retokenized_count=7)
        events = [
            sse({"choices": [{"text": "legacy stream"}]}),
            "data: [DONE]\n\n",
        ]

        result = asyncio.run(consume_sse_completion(
            FakeSession(events), "http://proxy/v1/completions", {}, tokenizer,
        ))

        self.assertEqual(result["completion_tokens"], 7)
        self.assertEqual(result["completion_token_source"], "retokenized_text_fallback")
        self.assertEqual(tokenizer.encode_calls, 1)

    def test_done_sentinel_is_followed_through_http_eof(self):
        tokenizer = FakeTokenizer(retokenized_count=1)
        events = [
            sse({"choices": [{"text": "a"}]}),
            sse({"choices": [], "usage": {"completion_tokens": 1}}),
            "data: [DONE]\n\n",
            "data: not-json-after-done\n\n",
        ]
        session = FakeSession(events)
        result = asyncio.run(consume_sse_completion(
            session, "http://proxy/v1/completions", {}, tokenizer,
        ))
        self.assertTrue(result["ok"])
        self.assertEqual(session.response.content.consumed, events)


if __name__ == "__main__":
    unittest.main()
