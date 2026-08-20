"""Auditable disaggregated proxy with fail-closed decode streaming semantics.

The proxy preserves the working prefill contract (non-streaming, one output
token), then forwards the decode server's real OpenAI SSE bytes without
buffering, reconstruction, or timing sleeps.  The core is dependency-light so
its streaming behavior can be covered by CPU-only fake-upstream tests.
"""

import argparse
from dataclasses import dataclass, field
import inspect
import json
import os
from pathlib import Path
import time
import threading
from typing import Any, AsyncIterator, Callable, Mapping, Optional, Sequence
import uuid

from xpyd.compatibility import CompatibilityTable, EndpointPairCompatibility
from xpyd.registry import EndpointRegistry
from xpyd.types import EndpointSpec, EndpointState, LifecycleState


Clock = Callable[[], float]
DiagnosticSink = Callable[[Mapping[str, Any]], None]


class ProxyUpstreamError(RuntimeError):
    """An upstream request or real stream failed."""


@dataclass(frozen=True)
class ProxyConfig:
    prefill_http_host: str
    decode_http_host: str
    prefill_addr_host: str
    decode_addr_host: str
    prefill_port: int = 8100
    decode_port: int = 8200
    kv_port: int = 14579
    prefill_kv_port: Optional[int] = None
    decode_kv_port: Optional[int] = None
    model: str = "mistralai/Mistral-7B-v0.1"
    prefill_endpoint_id: str = "P0"
    decode_endpoint_id: str = "D0"

    @property
    def prefill_url(self) -> str:
        return f"http://{self.prefill_http_host}:{self.prefill_port}/v1/completions"

    @property
    def decode_url(self) -> str:
        return f"http://{self.decode_http_host}:{self.decode_port}/v1/completions"


@dataclass
class ProxyDiagnostics:
    request_id: str
    vllm_request_id: str
    incoming_client_stream: bool
    outgoing_decode_stream: bool
    timestamps_monotonic_s: dict[str, Optional[float]]
    timestamps_wall_s: dict[str, Optional[float]]
    wall_clock: Clock = field(default=time.time, repr=False, compare=False)
    decode_content_type: Optional[str] = None
    decode_stream_available: bool = False
    upstream_chunk_count: int = 0
    upstream_byte_count: int = 0
    outcome: str = "in_progress"
    error: Optional[str] = None
    selected_prefill_endpoint_id: str = "P0"
    selected_decode_endpoint_id: str = "D0"

    def record(self, name: str, value: float) -> None:
        self.timestamps_monotonic_s[name] = value
        self.timestamps_wall_s[name] = self.wall_clock()

    def as_dict(self) -> dict[str, Any]:
        stamps = self.timestamps_monotonic_s

        def duration_ms(start: str, end: str) -> Optional[float]:
            left = stamps.get(start)
            right = stamps.get(end)
            if left is None or right is None:
                return None
            return (right - left) * 1000.0

        return {
            "event": "xpyd_proxy_stream_diagnostics",
            "request_id": self.request_id,
            "logical_request_id": self.request_id,
            "vllm_request_id": self.vllm_request_id,
            "selected_prefill_endpoint_id": self.selected_prefill_endpoint_id,
            "selected_decode_endpoint_id": self.selected_decode_endpoint_id,
            "route": {
                "prefill_endpoint_id": self.selected_prefill_endpoint_id,
                "decode_endpoint_id": self.selected_decode_endpoint_id,
            },
            "incoming_client_stream": self.incoming_client_stream,
            "outgoing_decode_stream": self.outgoing_decode_stream,
            "decode_content_type": self.decode_content_type,
            "decode_stream_available": self.decode_stream_available,
            "client_ttft_valid": self.decode_stream_available,
            "client_tpot_valid": self.decode_stream_available,
            "client_itl_valid": self.decode_stream_available,
            "timestamps_monotonic_s": dict(stamps),
            "timestamps_wall_s": dict(self.timestamps_wall_s),
            "durations_ms": {
                "prefill": duration_ms("prefill_started", "prefill_completed"),
                "decode_request_to_headers": duration_ms(
                    "decode_request_started", "decode_response_headers_received"
                ),
                "decode_request_to_first_real_chunk": duration_ms(
                    "decode_request_started", "decode_first_real_chunk_received"
                ),
                "first_chunk_forwarding_delay": duration_ms(
                    "decode_first_real_chunk_received",
                    "decode_first_real_chunk_forwarded",
                ),
                "full_decode_stream": duration_ms(
                    "decode_first_real_chunk_received", "decode_last_chunk_received"
                ),
                "decode_request_to_last_chunk": duration_ms(
                    "decode_request_started", "decode_last_chunk_received"
                ),
                "total_proxy_request": duration_ms(
                    "request_received", "response_completed"
                ),
            },
            "upstream_chunk_count": self.upstream_chunk_count,
            "upstream_byte_count": self.upstream_byte_count,
            "outcome": self.outcome,
            "error": self.error,
        }


@dataclass
class PreparedProxyResponse:
    status_code: int
    content_type: str
    diagnostics: ProxyDiagnostics
    headers: dict[str, str]
    body: Optional[bytes] = None
    stream: Optional[AsyncIterator[bytes]] = None


def build_prefill_payload(client_body: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(client_body)
    payload["stream"] = False
    payload["max_tokens"] = 1
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = 1
    payload.pop("stream_options", None)
    return payload


def build_decode_payload(client_body: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(client_body)
    requested_stream = bool(client_body.get("stream", False))
    payload["stream"] = requested_stream
    if requested_stream:
        stream_options = dict(payload.get("stream_options") or {})
        stream_options["include_usage"] = True
        payload["stream_options"] = stream_options
    return payload


def upstream_headers(
    logical_request_id: str,
    vllm_request_id: str,
) -> dict[str, str]:
    return {
        "X-Request-Id": vllm_request_id,
        "X-Xpyd-Logical-Request-Id": logical_request_id,
    }


def ingress_logical_request_id(headers: Mapping[str, str]) -> Optional[str]:
    logical_request_id = headers.get("X-Xpyd-Logical-Request-Id")
    request_id = headers.get("X-Request-Id")
    if (
        logical_request_id is not None
        and request_id is not None
        and logical_request_id != request_id
    ):
        raise ValueError("conflicting ingress logical request ID headers")
    return logical_request_id or request_id


async def _close_upstream(response: Any, session: Any) -> None:
    release = getattr(response, "release", None)
    if release is not None:
        released = release()
        if inspect.isawaitable(released):
            await released
    close = getattr(session, "close", None)
    if close is not None:
        closed = close()
        if inspect.isawaitable(closed):
            await closed


class DisaggProxyCore:
    def __init__(
        self,
        config: ProxyConfig,
        session_factory: Callable[[], Any],
        *,
        clock: Clock = time.monotonic,
        wall_clock: Clock = time.time,
        diagnostic_sink: Optional[DiagnosticSink] = None,
    ) -> None:
        self.config = config
        self.session_factory = session_factory
        self.clock = clock
        self.wall_clock = wall_clock
        self.diagnostic_sink = diagnostic_sink or self._print_diagnostics

    @staticmethod
    def _print_diagnostics(value: Mapping[str, Any]) -> None:
        print(json.dumps(dict(value), sort_keys=True), flush=True)

    @staticmethod
    def _logical_request_id(value: Optional[str]) -> str:
        request_id = value if value is not None else uuid.uuid4().hex
        if not request_id or request_id != request_id.strip():
            raise ValueError("logical request ID must be non-empty without edge whitespace")
        if len(request_id) > 256 or any(
            not 33 <= ord(character) <= 126 for character in request_id
        ):
            raise ValueError(
                "logical request ID must be printable ASCII and <=256 characters"
            )
        return request_id

    def _vllm_request_id(self, logical_request_id: str) -> str:
        # vLLM 0.15.1's P2pNcclConnector parses both peer addresses from this
        # transport envelope.  Preserve that contract and use the ingress ID as
        # its stable suffix instead of generating a second identifier domain.
        prefill_kv_port = self.config.prefill_kv_port or self.config.kv_port
        decode_kv_port = self.config.decode_kv_port or self.config.kv_port
        return (
            f"___prefill_addr_{self.config.prefill_addr_host}:{prefill_kv_port}"
            f"___decode_addr_{self.config.decode_addr_host}:{decode_kv_port}_"
            f"{logical_request_id}"
        )

    def _diagnostic_headers(self, diagnostics: ProxyDiagnostics) -> dict[str, str]:
        return {
            "X-Xpyd-Incoming-Client-Stream": str(
                diagnostics.incoming_client_stream
            ).lower(),
            "X-Xpyd-Outgoing-Decode-Stream": str(
                diagnostics.outgoing_decode_stream
            ).lower(),
            "X-Xpyd-Decode-Content-Type": diagnostics.decode_content_type
            or "unavailable",
            "X-Xpyd-Decode-Stream-Available": str(
                diagnostics.decode_stream_available
            ).lower(),
            "X-Xpyd-Logical-Request-Id": diagnostics.request_id,
            "X-Xpyd-Vllm-Request-Id": diagnostics.vllm_request_id,
            "X-Xpyd-Prefill-Endpoint": diagnostics.selected_prefill_endpoint_id,
            "X-Xpyd-Decode-Endpoint": diagnostics.selected_decode_endpoint_id,
        }

    def _emit(self, diagnostics: ProxyDiagnostics) -> None:
        self.diagnostic_sink(diagnostics.as_dict())

    async def _read_and_close(self, response: Any, session: Any) -> bytes:
        try:
            return await response.read()
        finally:
            await _close_upstream(response, session)

    async def _real_decode_stream(
        self,
        response: Any,
        session: Any,
        diagnostics: ProxyDiagnostics,
    ) -> AsyncIterator[bytes]:
        completed = False
        try:
            async for chunk in response.content.iter_any():
                if not chunk:
                    continue
                received = self.clock()
                diagnostics.upstream_chunk_count += 1
                diagnostics.upstream_byte_count += len(chunk)
                if diagnostics.timestamps_monotonic_s.get(
                    "decode_first_real_chunk_received"
                ) is None:
                    diagnostics.record("decode_first_real_chunk_received", received)
                    diagnostics.record(
                        "decode_first_real_chunk_forwarded", self.clock()
                    )
                diagnostics.record("decode_last_chunk_received", received)
                yield chunk
            completed = True
            diagnostics.outcome = "completed"
        except Exception as exc:
            diagnostics.outcome = "stream_error"
            diagnostics.error = repr(exc)
            raise ProxyUpstreamError("decode SSE stream failed") from exc
        finally:
            diagnostics.record("response_completed", self.clock())
            if not completed and diagnostics.outcome == "in_progress":
                diagnostics.outcome = "stream_interrupted"
            await _close_upstream(response, session)
            self._emit(diagnostics)

    async def prepare(
        self,
        client_body: Mapping[str, Any],
        logical_request_id: Optional[str] = None,
    ) -> PreparedProxyResponse:
        incoming_stream = bool(client_body.get("stream", False))
        request_id = self._logical_request_id(logical_request_id)
        vllm_request_id = self._vllm_request_id(request_id)
        diagnostics = ProxyDiagnostics(
            request_id=request_id,
            vllm_request_id=vllm_request_id,
            incoming_client_stream=incoming_stream,
            outgoing_decode_stream=incoming_stream,
            timestamps_monotonic_s={
                "request_received": self.clock(),
                "route_selected": None,
                "prefill_started": None,
                "prefill_completed": None,
                "kv_handoff_completed": None,
                "decode_request_started": None,
                "decode_response_headers_received": None,
                "decode_first_real_chunk_received": None,
                "decode_first_real_chunk_forwarded": None,
                "decode_last_chunk_received": None,
                "response_completed": None,
            },
            timestamps_wall_s={
                "request_received": self.wall_clock(),
                "route_selected": None,
                "prefill_started": None,
                "prefill_completed": None,
                "kv_handoff_completed": None,
                "decode_request_started": None,
                "decode_response_headers_received": None,
                "decode_first_real_chunk_received": None,
                "decode_first_real_chunk_forwarded": None,
                "decode_last_chunk_received": None,
                "response_completed": None,
            },
            wall_clock=self.wall_clock,
            selected_prefill_endpoint_id=self.config.prefill_endpoint_id,
            selected_decode_endpoint_id=self.config.decode_endpoint_id,
        )
        diagnostics.record("route_selected", self.clock())
        headers = upstream_headers(request_id, vllm_request_id)

        prefill_session = self.session_factory()
        diagnostics.record("prefill_started", self.clock())
        try:
            prefill_response = await prefill_session.post(
                self.config.prefill_url,
                json=build_prefill_payload(client_body),
                headers=headers,
            )
            prefill_body = await self._read_and_close(
                prefill_response, prefill_session
            )
        except Exception as exc:
            await _close_upstream(None, prefill_session)
            diagnostics.outcome = "prefill_error"
            diagnostics.error = repr(exc)
            diagnostics.record("response_completed", self.clock())
            self._emit(diagnostics)
            raise ProxyUpstreamError("prefill request failed") from exc
        diagnostics.record("prefill_completed", self.clock())
        # With the accepted synchronous PUT connector, the producer response is
        # returned only after the consumer has received the KV payload.
        diagnostics.record("kv_handoff_completed", self.clock())
        if prefill_response.status != 200:
            diagnostics.outcome = "prefill_http_error"
            diagnostics.error = prefill_body[:300].decode("utf-8", errors="replace")
            diagnostics.record("response_completed", self.clock())
            self._emit(diagnostics)
            raise ProxyUpstreamError(
                f"prefill returned HTTP {prefill_response.status}"
            )

        decode_session = self.session_factory()
        diagnostics.record("decode_request_started", self.clock())
        try:
            decode_response = await decode_session.post(
                self.config.decode_url,
                json=build_decode_payload(client_body),
                headers=headers,
            )
        except Exception as exc:
            await _close_upstream(None, decode_session)
            diagnostics.outcome = "decode_request_error"
            diagnostics.error = repr(exc)
            diagnostics.record("response_completed", self.clock())
            self._emit(diagnostics)
            raise ProxyUpstreamError("decode request failed") from exc

        diagnostics.record("decode_response_headers_received", self.clock())
        diagnostics.decode_content_type = decode_response.headers.get(
            "Content-Type", ""
        )
        is_sse = "text/event-stream" in diagnostics.decode_content_type.lower()
        diagnostics.decode_stream_available = incoming_stream and is_sse

        if decode_response.status != 200:
            error_body = await self._read_and_close(decode_response, decode_session)
            diagnostics.outcome = "decode_http_error"
            diagnostics.error = error_body[:300].decode("utf-8", errors="replace")
            diagnostics.record("response_completed", self.clock())
            self._emit(diagnostics)
            raise ProxyUpstreamError(
                f"decode returned HTTP {decode_response.status}"
            )

        if incoming_stream and not is_sse:
            nonstream_body = await self._read_and_close(
                decode_response, decode_session
            )
            diagnostics.record("decode_last_chunk_received", self.clock())
            diagnostics.record("response_completed", self.clock())
            diagnostics.outcome = "decode_stream_unavailable"
            diagnostics.error = (
                "client requested streaming but D returned "
                f"{diagnostics.decode_content_type or 'unknown Content-Type'}"
            )
            self._emit(diagnostics)
            error = {
                "error": diagnostics.error,
                "decode_stream_available": False,
                "client_ttft_valid": False,
                "client_tpot_valid": False,
                "client_itl_valid": False,
                "upstream_body_preview": nonstream_body[:300].decode(
                    "utf-8", errors="replace"
                ),
            }
            return PreparedProxyResponse(
                status_code=502,
                content_type="application/json",
                diagnostics=diagnostics,
                headers=self._diagnostic_headers(diagnostics),
                body=json.dumps(error).encode("utf-8"),
            )

        if incoming_stream:
            return PreparedProxyResponse(
                status_code=200,
                content_type=diagnostics.decode_content_type,
                diagnostics=diagnostics,
                headers=self._diagnostic_headers(diagnostics),
                stream=self._real_decode_stream(
                    decode_response, decode_session, diagnostics
                ),
            )

        body = await self._read_and_close(decode_response, decode_session)
        diagnostics.record("decode_last_chunk_received", self.clock())
        diagnostics.record("response_completed", self.clock())
        diagnostics.outcome = "completed_nonstreaming_client_request"
        self._emit(diagnostics)
        return PreparedProxyResponse(
            status_code=200,
            content_type=diagnostics.decode_content_type or "application/json",
            diagnostics=diagnostics,
            headers=self._diagnostic_headers(diagnostics),
            body=body,
        )


@dataclass(frozen=True)
class EndpointTransport:
    endpoint_id: str
    http_host: str
    http_port: int
    addr_host: str
    kv_port: int


class RoundRobinCompatiblePairs:
    """Transparent deterministic baseline over explicitly validated pairs."""

    def __init__(
        self,
        registry: EndpointRegistry,
        compatibility: CompatibilityTable,
        pairs: Sequence[tuple[str, str]],
    ) -> None:
        validated = []
        for prefill_id, decode_id in pairs:
            prefill = registry.get_spec(prefill_id)
            decode = registry.get_spec(decode_id)
            if not compatibility.is_compatible(prefill, decode):
                raise ValueError(
                    "unvalidated or incompatible endpoint pair: %s->%s"
                    % (prefill_id, decode_id)
                )
            if prefill not in registry.healthy_active("prefill"):
                raise ValueError("prefill endpoint is not healthy/active: %s" % prefill_id)
            if decode not in registry.healthy_active("decode"):
                raise ValueError("decode endpoint is not healthy/active: %s" % decode_id)
            validated.append((prefill_id, decode_id))
        if not validated:
            raise ValueError("at least one explicitly compatible pair is required")
        if len(set(validated)) != len(validated):
            raise ValueError("compatible endpoint pairs must be unique")
        self.pairs = tuple(validated)
        self._index = 0
        self._lock = threading.Lock()

    def choose(self) -> tuple[str, str]:
        with self._lock:
            pair = self.pairs[self._index % len(self.pairs)]
            self._index += 1
            return pair


class FileControlledCompatiblePairs(RoundRobinCompatiblePairs):
    """Select only compatible pairs named by an atomic controller file.

    The file is read for every request so a separate feedback controller can
    change the next route without restarting persistent serving endpoints.
    Missing, stale, malformed, or incompatible control state fails closed.
    """

    def __init__(
        self,
        registry: EndpointRegistry,
        compatibility: CompatibilityTable,
        pairs: Sequence[tuple[str, str]],
        control_file: str,
        maximum_age_s: float,
        wall_clock: Clock = time.time,
    ) -> None:
        super().__init__(registry, compatibility, pairs)
        self.control_file = Path(control_file)
        self.maximum_age_s = float(maximum_age_s)
        self.wall_clock = wall_clock
        if self.maximum_age_s <= 0:
            raise ValueError("routing control maximum age must be positive")

    def choose(self) -> tuple[str, str]:
        try:
            value = json.loads(self.control_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("routing control file unavailable or invalid") from exc
        updated = value.get("updated_unix_s")
        if not isinstance(updated, (int, float)):
            raise ValueError("routing control file lacks a valid freshness timestamp")
        age_s = self.wall_clock() - float(updated)
        if age_s < -5.0 or age_s > self.maximum_age_s:
            raise ValueError("routing control file is stale")
        raw_pairs = value.get("pairs")
        if not isinstance(raw_pairs, list) or not raw_pairs:
            raise ValueError("routing control file has no selected pairs")
        selected = []
        for raw in raw_pairs:
            if not isinstance(raw, list) or len(raw) != 2:
                raise ValueError("routing control pair must contain P and D endpoint IDs")
            pair = (str(raw[0]), str(raw[1]))
            if pair not in self.pairs:
                raise ValueError("routing control selected an incompatible pair")
            selected.append(pair)
        with self._lock:
            pair = selected[self._index % len(selected)]
            self._index += 1
            return pair


class MultiEndpointDisaggProxyCore:
    """Select one exact compatible pair, then reuse the accepted proxy core."""

    def __init__(
        self,
        registry: EndpointRegistry,
        compatibility: CompatibilityTable,
        transports: Mapping[str, EndpointTransport],
        pairs: Sequence[tuple[str, str]],
        session_factory: Callable[[], Any],
        *,
        model: str = "mistralai/Mistral-7B-v0.1",
        routing_control_file: Optional[str] = None,
        routing_control_maximum_age_s: float = 900.0,
        clock: Clock = time.monotonic,
        wall_clock: Clock = time.time,
        diagnostic_sink: Optional[DiagnosticSink] = None,
    ) -> None:
        self.registry = registry
        self.compatibility = compatibility
        self.transports = dict(transports)
        control_file = os.path.expandvars(str(routing_control_file or ""))
        if control_file:
            self.selector = FileControlledCompatiblePairs(
                registry,
                compatibility,
                pairs,
                control_file,
                float(routing_control_maximum_age_s),
                wall_clock=wall_clock,
            )
        else:
            self.selector = RoundRobinCompatiblePairs(registry, compatibility, pairs)
        self.cores: dict[tuple[str, str], DisaggProxyCore] = {}
        for prefill_id, decode_id in self.selector.pairs:
            prefill = self.transports[prefill_id]
            decode = self.transports[decode_id]
            config = ProxyConfig(
                prefill_http_host=prefill.http_host,
                decode_http_host=decode.http_host,
                prefill_addr_host=prefill.addr_host,
                decode_addr_host=decode.addr_host,
                prefill_port=prefill.http_port,
                decode_port=decode.http_port,
                kv_port=prefill.kv_port,
                prefill_kv_port=prefill.kv_port,
                decode_kv_port=decode.kv_port,
                model=model,
                prefill_endpoint_id=prefill_id,
                decode_endpoint_id=decode_id,
            )
            self.cores[(prefill_id, decode_id)] = DisaggProxyCore(
                config,
                session_factory,
                clock=clock,
                wall_clock=wall_clock,
                diagnostic_sink=diagnostic_sink,
            )

    async def prepare(
        self,
        client_body: Mapping[str, Any],
        logical_request_id: Optional[str] = None,
    ) -> PreparedProxyResponse:
        pair = self.selector.choose()
        return await self.cores[pair].prepare(client_body, logical_request_id)


def _build_multi_core(
    config: Mapping[str, Any],
    session_factory: Callable[[], Any],
    *,
    diagnostic_sink: Optional[DiagnosticSink] = None,
) -> MultiEndpointDisaggProxyCore:
    registry = EndpointRegistry()
    transports = {}
    for value in config["endpoints"]:
        spec = EndpointSpec(
            endpoint_id=str(value["endpoint_id"]),
            role=str(value["role"]),
            gpu_type=str(value["gpu_type"]),
            node=str(value["node"]),
            gpu_ids=tuple(int(item) for item in value["gpu_ids"]),
            tp_degree=int(value["tp_degree"]),
            http_uri="http://%s:%d" % (value["http_host"], int(value["http_port"])),
            kv_connector=str(value["kv_connector"]),
        )
        registry.register(spec, EndpointState(
            endpoint_id=spec.endpoint_id,
            freq_mhz=int(value["configured_frequency_mhz"]),
            lifecycle=LifecycleState.ACTIVE,
            healthy=True,
        ))
        transports[spec.endpoint_id] = EndpointTransport(
            endpoint_id=spec.endpoint_id,
            http_host=str(value["http_host"]),
            http_port=int(value["http_port"]),
            addr_host=str(value["addr_host"]),
            kv_port=int(value["kv_port"]),
        )
    evidence = []
    pairs = []
    for value in config["compatible_pairs"]:
        pair = (str(value["prefill_endpoint_id"]), str(value["decode_endpoint_id"]))
        pairs.append(pair)
        prefill = registry.get_spec(pair[0])
        decode = registry.get_spec(pair[1])
        evidence.append(EndpointPairCompatibility(
            prefill_endpoint_id=pair[0],
            decode_endpoint_id=pair[1],
            connector=str(value["connector"]),
            prefill_tp=int(value.get("prefill_tp", prefill.tp_degree)),
            decode_tp=int(value.get("decode_tp", decode.tp_degree)),
            supported=bool(value["supported"]),
            reason=str(value["reason"]),
        ))
    return MultiEndpointDisaggProxyCore(
        registry,
        CompatibilityTable(endpoint_pairs=evidence),
        transports,
        pairs,
        session_factory,
        model=str(config.get("model", "mistralai/Mistral-7B-v0.1")),
        routing_control_file=config.get("routing_control_file"),
        routing_control_maximum_age_s=float(
            config.get("routing_control_maximum_age_s", 900.0)
        ),
        diagnostic_sink=diagnostic_sink,
    )


def _expand_environment(value: Any) -> Any:
    import os

    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def create_app(
    config: ProxyConfig,
    timeout_s: float = 900.0,
    diagnostic_sink: Optional[DiagnosticSink] = None,
) -> Any:
    import aiohttp
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    timeout = aiohttp.ClientTimeout(total=timeout_s, connect=10, sock_connect=10)
    core = DisaggProxyCore(
        config,
        lambda: aiohttp.ClientSession(timeout=timeout),
        diagnostic_sink=diagnostic_sink,
    )
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions(raw_request: Request) -> Any:
        try:
            logical_request_id = ingress_logical_request_id(raw_request.headers)
            prepared = await core.prepare(
                await raw_request.json(),
                logical_request_id=logical_request_id,
            )
        except (ProxyUpstreamError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=502)
        if prepared.stream is not None:
            return StreamingResponse(
                prepared.stream,
                status_code=prepared.status_code,
                media_type="text/event-stream",
                headers=prepared.headers,
            )
        response_headers = dict(prepared.headers)
        response_headers["Content-Type"] = prepared.content_type
        return Response(
            content=prepared.body or b"",
            status_code=prepared.status_code,
            headers=response_headers,
        )

    return app


def create_multi_app(
    config: Mapping[str, Any],
    timeout_s: float = 900.0,
    diagnostic_sink: Optional[DiagnosticSink] = None,
) -> Any:
    import aiohttp
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    timeout = aiohttp.ClientTimeout(total=timeout_s, connect=10, sock_connect=10)
    core = _build_multi_core(
        config,
        lambda: aiohttp.ClientSession(timeout=timeout),
        diagnostic_sink=diagnostic_sink,
    )
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "routing_policy": "round_robin_explicit_compatible_pairs",
            "compatible_pairs": [list(item) for item in core.selector.pairs],
        }

    @app.post("/v1/completions")
    async def completions(raw_request: Request) -> Any:
        try:
            prepared = await core.prepare(
                await raw_request.json(),
                logical_request_id=ingress_logical_request_id(raw_request.headers),
            )
        except (ProxyUpstreamError, ValueError, KeyError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=502)
        if prepared.stream is not None:
            return StreamingResponse(
                prepared.stream,
                status_code=prepared.status_code,
                media_type="text/event-stream",
                headers=prepared.headers,
            )
        headers = dict(prepared.headers)
        headers["Content-Type"] = prepared.content_type
        return Response(
            content=prepared.body or b"",
            status_code=prepared.status_code,
            headers=headers,
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-endpoint-config")
    parser.add_argument("--prefill-http-host")
    parser.add_argument("--decode-http-host")
    parser.add_argument("--prefill-addr-host")
    parser.add_argument("--decode-addr-host")
    parser.add_argument("--prefill-port", type=int, default=8100)
    parser.add_argument("--decode-port", type=int, default=8200)
    parser.add_argument("--kv-port", type=int, default=14579)
    parser.add_argument("--proxy-port", type=int, default=8000)
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--diagnostics-log")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import uvicorn
    required_single = (
        args.prefill_http_host,
        args.decode_http_host,
        args.prefill_addr_host,
        args.decode_addr_host,
    )
    if not args.multi_endpoint_config and not all(required_single):
        raise SystemExit(
            "single-pair mode requires prefill/decode HTTP and address hosts"
        )
    if args.multi_endpoint_config and any(required_single):
        raise SystemExit(
            "--multi-endpoint-config cannot be combined with single-pair hosts"
        )
    diagnostics_path = args.diagnostics_log

    def diagnostic_sink(value: Mapping[str, Any]) -> None:
        line = json.dumps(dict(value), sort_keys=True)
        print(line, flush=True)
        if diagnostics_path:
            with open(diagnostics_path, "a", encoding="utf-8") as stream:
                stream.write(line + "\n")

    if args.multi_endpoint_config:
        multi_config = _expand_environment(json.loads(
            open(args.multi_endpoint_config, encoding="utf-8").read()
        ))
        print(
            "Multi-endpoint proxy 0.0.0.0:%d config=%s"
            % (args.proxy_port, args.multi_endpoint_config),
            flush=True,
        )
        uvicorn.run(
            create_multi_app(multi_config, diagnostic_sink=diagnostic_sink),
            host="0.0.0.0",
            port=args.proxy_port,
            log_level="warning",
        )
        return 0
    config = ProxyConfig(
        prefill_http_host=args.prefill_http_host,
        decode_http_host=args.decode_http_host,
        prefill_addr_host=args.prefill_addr_host,
        decode_addr_host=args.decode_addr_host,
        prefill_port=args.prefill_port,
        decode_port=args.decode_port,
        kv_port=args.kv_port,
        model=args.model,
    )

    print(
        f"Proxy 0.0.0.0:{args.proxy_port} "
        f"prefill={config.prefill_url} decode={config.decode_url} "
        f"prefill_addr={config.prefill_addr_host}:{config.kv_port} "
        f"decode_addr={config.decode_addr_host}:{config.kv_port}",
        flush=True,
    )
    uvicorn.run(
        create_app(config, diagnostic_sink=diagnostic_sink),
        host="0.0.0.0", port=args.proxy_port,
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
