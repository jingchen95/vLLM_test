"""CPU-only tests for exact streaming completion-token accounting."""

import asyncio
import json
import sys
import types
import unittest


try:
    import aiohttp  # noqa: F401
except ModuleNotFoundError:
    aiohttp_stub = types.ModuleType("aiohttp")
    aiohttp_stub.ClientSession = object
    sys.modules["aiohttp"] = aiohttp_stub

try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoTokenizer = object
    sys.modules["transformers"] = transformers_stub

from replay_synthetic_trace import consume_sse_completion


class FakeContent:
    def __init__(self, events):
        self.events = events

    async def __aiter__(self):
        for event in self.events:
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


if __name__ == "__main__":
    unittest.main()
