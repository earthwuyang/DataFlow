from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BatchStats:
    batch_size: int
    latency_ms: float
    oom: bool = False


@dataclass
class DynamicBatchState:
    current_batch_size: int = 8
    min_batch_size: int = 1
    max_batch_size: int = 64
    target_latency_ms: Optional[float] = None
    history: List[BatchStats] = field(default_factory=list)


class LLMBatchAdvisor:
    """LLM-assisted policy hook.

    In this initial implementation, we accept lightweight LLM-style annotations
    from operator physical_profile (e.g., complexity, memory_sensitivity) and
    map them to conservative adjustment recommendations.
    """

    def recommend_step(self, profile: Dict) -> int:
        complexity = str(profile.get("complexity", "medium")).lower()
        mem_sensitive = bool(profile.get("memory_sensitive", False))

        if mem_sensitive or complexity == "high":
            return -1
        if complexity == "low":
            return 1
        return 0


class DynamicBatchController:
    """Runtime dynamic batching controller using feedback and optional LLM hints."""

    def __init__(self, advisor: Optional[LLMBatchAdvisor] = None):
        self.advisor = advisor or LLMBatchAdvisor()

    def update(self, state: DynamicBatchState, observed_latency_ms: float, oom: bool, profile: Optional[Dict] = None) -> int:
        state.history.append(BatchStats(batch_size=state.current_batch_size, latency_ms=observed_latency_ms, oom=oom))

        if oom:
            state.current_batch_size = max(state.min_batch_size, state.current_batch_size // 2)
            return state.current_batch_size

        if state.target_latency_ms is not None:
            if observed_latency_ms < state.target_latency_ms * 0.75:
                state.current_batch_size = min(state.max_batch_size, state.current_batch_size * 2)
            elif observed_latency_ms > state.target_latency_ms * 1.2:
                state.current_batch_size = max(state.min_batch_size, state.current_batch_size // 2)

        if profile:
            step = self.advisor.recommend_step(profile)
            if step > 0:
                state.current_batch_size = min(state.max_batch_size, state.current_batch_size + 1)
            elif step < 0:
                state.current_batch_size = max(state.min_batch_size, state.current_batch_size - 1)

        return state.current_batch_size
