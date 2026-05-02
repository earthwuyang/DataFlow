from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from dataflow.planner.execution_cache import StageCacheManager


@dataclass
class OperatorPhysicalProfile:
    op_name: str
    device_hint: str = "cpu"
    estimated_ms_per_sample: float = 1.0
    estimated_selectivity: float = 1.0
    gpu_memory_mb: float = 0.0
    batch_size_hint: Optional[int] = None
    replica_hint: Optional[int] = None
    latency_table_ms: Optional[Dict[Tuple[int, int], float]] = None
    memory_table_mb: Optional[Dict[Tuple[int, int], float]] = None


@dataclass
class PhysicalStage:
    op_name: str
    op_index: int
    device: str
    estimated_input_rows: float
    estimated_output_rows: float
    estimated_stage_cost_ms: float
    batch_size: Optional[int] = None
    replicas: Optional[int] = None
    cache_enabled: bool = False
    cache_key: Optional[str] = None
    cache_reason: str = ""


@dataclass
class PhysicalPlan:
    stages: List[PhysicalStage] = field(default_factory=list)
    objective: str = "minimize_estimated_total_latency"

    @property
    def estimated_total_cost_ms(self) -> float:
        return sum(stage.estimated_stage_cost_ms for stage in self.stages)


class PhysicalPlanOptimizer:
    """Lightweight physical optimizer for DataFlow pipelines."""

    def __init__(self, enable_reorder: bool = False, cache_manager: Optional[StageCacheManager] = None):
        self.enable_reorder = enable_reorder
        self.cache_manager = cache_manager or StageCacheManager()

    def _profile_from_operator(self, op_runtime: Any) -> OperatorPhysicalProfile:
        op = op_runtime.op
        op_name = op_runtime.op_name

        default_device = "gpu" if "cuda" in str(getattr(op, "device", "")).lower() else "cpu"
        profile = OperatorPhysicalProfile(op_name=op_name, device_hint=default_device)

        raw = getattr(op, "physical_profile", None)
        if isinstance(raw, dict):
            profile.device_hint = str(raw.get("device_hint", profile.device_hint))
            profile.estimated_ms_per_sample = float(raw.get("estimated_ms_per_sample", profile.estimated_ms_per_sample))
            profile.estimated_selectivity = float(raw.get("estimated_selectivity", profile.estimated_selectivity))
            profile.gpu_memory_mb = float(raw.get("gpu_memory_mb", profile.gpu_memory_mb))
            profile.batch_size_hint = raw.get("batch_size_hint")
            profile.replica_hint = raw.get("replica_hint")
            profile.latency_table_ms = raw.get("latency_table_ms")
            profile.memory_table_mb = raw.get("memory_table_mb")
        return profile

    def tune_batch_and_replicas(
        self,
        profile: OperatorPhysicalProfile,
        memory_budget_mb: Optional[float] = None,
        batch_candidates: Optional[List[int]] = None,
        replica_candidates: Optional[List[int]] = None,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Pick (batch, replicas) that maximize throughput under memory limit.

        Uses optional profile tables:
        - latency_table_ms[(batch, replicas)] = latency for one batch
        - memory_table_mb[(batch, replicas)] = peak GPU memory
        """
        if profile.device_hint != "gpu":
            return (profile.batch_size_hint, profile.replica_hint)

        batch_candidates = batch_candidates or [1, 2, 4, 8, 16, 32]
        replica_candidates = replica_candidates or [1, 2, 4]

        best = None
        best_score = -1.0

        for b in batch_candidates:
            for r in replica_candidates:
                lat = None
                if profile.latency_table_ms and (b, r) in profile.latency_table_ms:
                    lat = float(profile.latency_table_ms[(b, r)])
                if lat is None:
                    # fallback rough model
                    lat = max(1e-6, profile.estimated_ms_per_sample * b / max(r, 1))

                mem = None
                if profile.memory_table_mb and (b, r) in profile.memory_table_mb:
                    mem = float(profile.memory_table_mb[(b, r)])
                else:
                    mem = profile.gpu_memory_mb * b

                if memory_budget_mb is not None and mem > memory_budget_mb:
                    continue

                throughput = (b * r) / lat
                if throughput > best_score:
                    best_score = throughput
                    best = (b, r)

        if best is None:
            return (profile.batch_size_hint, profile.replica_hint)
        return best

    def build_plan(self, op_runtimes: List[Any], input_rows: float = 1000.0, memory_budget_mb: Optional[float] = None) -> PhysicalPlan:
        plan = PhysicalPlan()
        rows = max(0.0, float(input_rows))

        for idx, rt in enumerate(op_runtimes):
            prof = self._profile_from_operator(rt)
            safe_sel = min(max(prof.estimated_selectivity, 0.0), 1.0)

            batch, replicas = self.tune_batch_and_replicas(prof, memory_budget_mb=memory_budget_mb)
            eff = max(1, (replicas or 1) * (batch or 1))
            stage_cost = rows * max(prof.estimated_ms_per_sample, 0.0) / eff
            out_rows = rows * safe_sel

            cache_decision = self.cache_manager.should_cache(stage_cost, safe_sel)
            cache_key = None
            if cache_decision.enabled:
                cache_key = self.cache_manager.build_cache_key(
                    op_name=rt.op_name,
                    profile=getattr(rt.op, "physical_profile", {}) if hasattr(rt, "op") else {},
                    config={"batch_size": batch, "replicas": replicas, "device": prof.device_hint},
                )

            plan.stages.append(
                PhysicalStage(
                    op_name=rt.op_name,
                    op_index=idx,
                    device=prof.device_hint,
                    estimated_input_rows=rows,
                    estimated_output_rows=out_rows,
                    estimated_stage_cost_ms=stage_cost,
                    batch_size=batch,
                    replicas=replicas,
                    cache_enabled=cache_decision.enabled,
                    cache_key=cache_key,
                    cache_reason=cache_decision.reason,
                )
            )
            rows = out_rows
        return plan

    def maybe_reorder(self, op_runtimes: List[Any], plan: PhysicalPlan) -> List[Any]:
        if not self.enable_reorder:
            return op_runtimes
        safe = []
        for rt in op_runtimes:
            profile = getattr(rt.op, "physical_profile", None)
            safe.append(isinstance(profile, dict) and bool(profile.get("reorder_safe", False)))
        if not all(safe):
            return op_runtimes

        pairs = list(zip(op_runtimes, plan.stages))
        pairs.sort(key=lambda p: (p[1].estimated_stage_cost_ms, -p[1].estimated_output_rows))
        return [p[0] for p in pairs]
