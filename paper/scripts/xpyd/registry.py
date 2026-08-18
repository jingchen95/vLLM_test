"""In-memory endpoint inventory and observation registry."""

from dataclasses import replace
from typing import Dict, Optional, Tuple

from xpyd.types import EndpointRole, EndpointSpec, EndpointState, LifecycleState


class DuplicateEndpointError(ValueError):
    pass


class GPUAllocationConflictError(ValueError):
    pass


class EndpointRegistry:
    """Stores known endpoint specs/states without network discovery."""

    def __init__(self, allow_gpu_overlap: bool = False) -> None:
        self._specs: Dict[str, EndpointSpec] = {}
        self._states: Dict[str, EndpointState] = {}
        self._allow_gpu_overlap = allow_gpu_overlap

    def register(
        self,
        spec: EndpointSpec,
        state: Optional[EndpointState] = None,
    ) -> None:
        if spec.endpoint_id in self._specs:
            raise DuplicateEndpointError("endpoint already registered: %s" % spec.endpoint_id)
        stored_state = state or EndpointState(
            endpoint_id=spec.endpoint_id,
            freq_mhz=None,
            lifecycle=LifecycleState.COLD,
            healthy=False,
        )
        if stored_state.endpoint_id != spec.endpoint_id:
            raise ValueError("state endpoint_id must match the endpoint spec")
        if stored_state.lifecycle == LifecycleState.ACTIVE:
            self._validate_no_active_gpu_conflict(spec)
        self._specs[spec.endpoint_id] = spec
        self._states[spec.endpoint_id] = replace(stored_state)

    def update_state(self, state: EndpointState) -> None:
        if state.endpoint_id not in self._specs:
            raise KeyError(state.endpoint_id)
        if state.lifecycle == LifecycleState.ACTIVE:
            self._validate_no_active_gpu_conflict(self._specs[state.endpoint_id])
        self._states[state.endpoint_id] = replace(state)

    def get_spec(self, endpoint_id: str) -> EndpointSpec:
        return self._specs[endpoint_id]

    def get_state(self, endpoint_id: str) -> EndpointState:
        return replace(self._states[endpoint_id])

    def list_endpoints(
        self,
        role: Optional[EndpointRole] = None,
        healthy_only: bool = False,
        active_only: bool = False,
    ) -> Tuple[EndpointSpec, ...]:
        endpoints = []
        for endpoint_id, spec in self._specs.items():
            state = self._states[endpoint_id]
            if role is not None and spec.role != role:
                continue
            if healthy_only and not state.healthy:
                continue
            if active_only and state.lifecycle != LifecycleState.ACTIVE:
                continue
            endpoints.append(spec)
        return tuple(endpoints)

    def list_by_role(self, role: EndpointRole) -> Tuple[EndpointSpec, ...]:
        return self.list_endpoints(role=role)

    def healthy_active(self, role: Optional[EndpointRole] = None) -> Tuple[EndpointSpec, ...]:
        return self.list_endpoints(role=role, healthy_only=True, active_only=True)

    def _validate_no_active_gpu_conflict(self, candidate: EndpointSpec) -> None:
        if self._allow_gpu_overlap:
            return
        candidate_gpus = {(candidate.node, gpu_id) for gpu_id in candidate.gpu_ids}
        for endpoint_id, existing in self._specs.items():
            if endpoint_id == candidate.endpoint_id:
                continue
            if self._states[endpoint_id].lifecycle != LifecycleState.ACTIVE:
                continue
            existing_gpus = {(existing.node, gpu_id) for gpu_id in existing.gpu_ids}
            overlap = candidate_gpus.intersection(existing_gpus)
            if overlap:
                raise GPUAllocationConflictError(
                    "active endpoints %s and %s overlap on %s"
                    % (endpoint_id, candidate.endpoint_id, sorted(overlap))
                )
