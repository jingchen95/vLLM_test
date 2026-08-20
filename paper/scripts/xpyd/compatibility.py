"""Evidence-backed P/D KV connector compatibility."""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from xpyd.types import EndpointSpec


@dataclass(frozen=True)
class ConnectorCompatibility:
    connector: str
    prefill_tp: int
    decode_tp: int
    supported: bool
    reason: str
    measured_bandwidth_gbps: Optional[float] = None
    measured_latency_us: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.connector.strip():
            raise ValueError("connector must be non-empty")
        if self.prefill_tp <= 0 or self.decode_tp <= 0:
            raise ValueError("prefill_tp and decode_tp must be positive")
        if self.measured_bandwidth_gbps is not None and self.measured_bandwidth_gbps < 0:
            raise ValueError("measured bandwidth cannot be negative")
        if self.measured_latency_us is not None and self.measured_latency_us < 0:
            raise ValueError("measured latency cannot be negative")


@dataclass(frozen=True)
class EndpointPairCompatibility:
    """Physical evidence for one exact, immutable P->D endpoint pair."""

    prefill_endpoint_id: str
    decode_endpoint_id: str
    connector: str
    prefill_tp: int
    decode_tp: int
    supported: bool
    reason: str

    def __post_init__(self) -> None:
        if not self.prefill_endpoint_id.strip() or not self.decode_endpoint_id.strip():
            raise ValueError("endpoint pair IDs must be non-empty")
        if not self.connector.strip():
            raise ValueError("connector must be non-empty")
        if self.prefill_tp <= 0 or self.decode_tp <= 0:
            raise ValueError("prefill_tp and decode_tp must be positive")
        if not self.reason.strip():
            raise ValueError("endpoint-pair evidence requires a reason")


class CompatibilityTable:
    """Explicit connector/TP evidence table; unknown combinations fail closed."""

    def __init__(
        self,
        entries: Iterable[ConnectorCompatibility] = (),
        endpoint_pairs: Iterable[EndpointPairCompatibility] = (),
    ) -> None:
        self._entries: Dict[Tuple[str, int, int], ConnectorCompatibility] = {}
        self._endpoint_pairs: Dict[
            Tuple[str, str], EndpointPairCompatibility
        ] = {}
        for entry in entries:
            self.add(entry)
        for pair in endpoint_pairs:
            self.add_endpoint_pair(pair)

    def add(self, entry: ConnectorCompatibility) -> None:
        key = (entry.connector, entry.prefill_tp, entry.decode_tp)
        if key in self._entries:
            raise ValueError("duplicate compatibility entry: %r" % (key,))
        self._entries[key] = entry

    def add_endpoint_pair(self, pair: EndpointPairCompatibility) -> None:
        key = (pair.prefill_endpoint_id, pair.decode_endpoint_id)
        if key in self._endpoint_pairs:
            raise ValueError("duplicate endpoint compatibility entry: %r" % (key,))
        self._endpoint_pairs[key] = pair

    def endpoint_pair(
        self, prefill_endpoint_id: str, decode_endpoint_id: str
    ) -> Optional[EndpointPairCompatibility]:
        return self._endpoint_pairs.get(
            (prefill_endpoint_id, decode_endpoint_id)
        )

    def lookup(
        self,
        connector: str,
        prefill_tp: int,
        decode_tp: int,
    ) -> Optional[ConnectorCompatibility]:
        return self._entries.get((connector, prefill_tp, decode_tp))

    def is_compatible(
        self,
        prefill_endpoint: EndpointSpec,
        decode_endpoint: EndpointSpec,
    ) -> bool:
        if prefill_endpoint.role != "prefill" or decode_endpoint.role != "decode":
            return False
        connector = prefill_endpoint.kv_connector
        if not connector or connector != decode_endpoint.kv_connector:
            return False
        pair = self.endpoint_pair(
            prefill_endpoint.endpoint_id, decode_endpoint.endpoint_id
        )
        if pair is not None:
            return bool(
                pair.supported
                and pair.connector == connector
                and pair.prefill_tp == prefill_endpoint.tp_degree
                and pair.decode_tp == decode_endpoint.tp_degree
            )
        # A table configured for exact endpoint evidence is fail-closed for
        # every unlisted pair, even if a generic connector/TP entry exists.
        if self._endpoint_pairs:
            return False
        entry = self.lookup(
            connector, prefill_endpoint.tp_degree, decode_endpoint.tp_degree
        )
        return entry is not None and entry.supported

    def explicit_endpoint_pairs(self) -> Tuple[EndpointPairCompatibility, ...]:
        return tuple(self._endpoint_pairs[key] for key in sorted(self._endpoint_pairs))
