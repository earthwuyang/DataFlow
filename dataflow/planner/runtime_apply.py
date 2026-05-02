from __future__ import annotations

from typing import Any, List

from dataflow.planner.telemetry import TelemetryStore
from dataflow.planner.rewrite import pushdown_filters, fuse_adjacent_light_ops
from dataflow.planner.placement import round_robin_gpu_placement


def apply_physical_plan_end_to_end(op_runtimes: List[Any], num_gpus: int = 1) -> dict:
    """Apply rewrite + placement decisions and return execution directives."""
    rewritten = pushdown_filters(op_runtimes)
    rewritten = fuse_adjacent_light_ops(rewritten)
    placement = round_robin_gpu_placement([rt.op_name for rt in rewritten], num_gpus=num_gpus)

    directives = {
        "ordered_ops": [rt.op_name for rt in rewritten],
        "placement": [{"op_name": d.op_name, "gpu_id": d.gpu_id} for d in placement],
    }
    return directives


def update_cost_model_from_telemetry(op_runtimes: List[Any], telemetry: TelemetryStore):
    snap = telemetry.profile_snapshot()
    for rt in op_runtimes:
        if rt.op_name not in snap:
            continue
        prof = getattr(rt.op, "physical_profile", None)
        if not isinstance(prof, dict):
            prof = {}
            setattr(rt.op, "physical_profile", prof)
        prof["estimated_ms_per_sample"] = max(0.001, snap[rt.op_name]["avg_latency_ms"])
