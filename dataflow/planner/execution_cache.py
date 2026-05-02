from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CacheDecision:
    enabled: bool
    cache_key: Optional[str] = None
    reason: str = ""


class StageCacheManager:
    """Simple stage-level cache manager for physical DAG execution.

    This provides deterministic cache keys from operator identity + selected
    execution settings and can be used by runtimes to materialize/reuse stage
    outputs.
    """

    def __init__(self, cache_root: str = ".cache/dataflow_physical"):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def build_cache_key(self, op_name: str, profile: Dict[str, Any], config: Dict[str, Any]) -> str:
        payload = {
            "op_name": op_name,
            "profile": profile,
            "config": config,
        }
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]

    def should_cache(self, estimated_stage_cost_ms: float, selectivity: float) -> CacheDecision:
        # Heuristic: cache expensive or highly selective stages.
        if estimated_stage_cost_ms >= 5_000:
            return CacheDecision(enabled=True, reason="expensive_stage")
        if 0.0 <= selectivity <= 0.2:
            return CacheDecision(enabled=True, reason="high_selectivity_pruning")
        return CacheDecision(enabled=False, reason="low_expected_reuse_benefit")

    def cache_path(self, cache_key: str) -> Path:
        return self.cache_root / f"{cache_key}.jsonl"
