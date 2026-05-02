from __future__ import annotations

from typing import Any, List


def pushdown_filters(op_runtimes: List[Any]) -> List[Any]:
    """Push low-cost selective filter-like ops earlier when explicitly safe."""
    def score(rt: Any):
        prof = getattr(rt.op, "physical_profile", {}) if hasattr(rt, "op") else {}
        if not prof.get("pushdown_safe", False):
            return (1, 0.0)
        sel = float(prof.get("estimated_selectivity", 1.0))
        cost = float(prof.get("estimated_ms_per_sample", 1.0))
        # lower selectivity and lower cost first
        return (0, sel * cost)

    return sorted(op_runtimes, key=score)


def fuse_adjacent_light_ops(op_runtimes: List[Any]) -> List[Any]:
    """Tag fusion groups for adjacent light CPU ops (metadata-only in v1)."""
    for rt in op_runtimes:
        prof = getattr(rt.op, "physical_profile", {}) if hasattr(rt, "op") else {}
        if prof.get("fusible", False) and prof.get("device_hint", "cpu") == "cpu":
            setattr(rt, "fusion_group", "cpu_light")
    return op_runtimes
