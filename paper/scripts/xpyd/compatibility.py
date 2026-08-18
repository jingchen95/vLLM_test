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


class CompatibilityTable:
    """Explicit connector/TP evidence table; unknown combinations fail closed."""

    def __init__(self, entries: Iterable[ConnectorCompatibility] = ()) -> None:
        self._entries: Dict[Tuple[str, int, int], ConnectorCompatibility] = {}
        for entry in entries:
            self.add(entry)

    def add(self, entry: ConnectorCompatibility) -> None:
        key = (entry.connector, entry.prefill_tp, entry.decode_tp)
        if key in self._entries:
            raise ValueError("duplicate compatibility entry: %r" % (key,))
        self._entries[key] = entry

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
        entry = self.lookup(connector, prefill_endpoint.tp_degree, decode_endpoint.tp_degree)
        return entry is not None and entry.supported
