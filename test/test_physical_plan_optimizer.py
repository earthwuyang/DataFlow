from dataflow.planner import PhysicalPlanOptimizer


class DummyOp:
    def __init__(self, profile=None, device="cpu"):
        self.physical_profile = profile or {}
        self.device = device


class DummyRuntime:
    def __init__(self, name, op):
        self.op_name = name
        self.op = op


def test_profile_cost_selectivity_chain():
    rts = [
        DummyRuntime("cheap_filter", DummyOp({"estimated_ms_per_sample": 0.1, "estimated_selectivity": 0.5})),
        DummyRuntime("expensive_gpu", DummyOp({"device_hint": "gpu", "estimated_ms_per_sample": 2.0, "estimated_selectivity": 1.0})),
    ]
    opt = PhysicalPlanOptimizer(enable_reorder=False)
    plan = opt.build_plan(rts, input_rows=100)

    assert len(plan.stages) == 2
    assert plan.stages[0].estimated_output_rows == 50
    assert plan.stages[1].estimated_input_rows == 50
    assert plan.stages[1].device == "gpu"
    assert plan.estimated_total_cost_ms > 0


def test_selectivity_is_clamped_to_valid_range():
    rts = [DummyRuntime("bad_sel", DummyOp({"estimated_ms_per_sample": 1.0, "estimated_selectivity": 2.5}))]
    opt = PhysicalPlanOptimizer(enable_reorder=False)
    plan = opt.build_plan(rts, input_rows=10)
    assert plan.stages[0].estimated_output_rows == 10


def test_reorder_requires_explicit_safety_on_all_ops():
    a = DummyRuntime("a", DummyOp({"estimated_ms_per_sample": 3.0, "reorder_safe": True}))
    b = DummyRuntime("b", DummyOp({"estimated_ms_per_sample": 1.0}))
    rts = [a, b]

    opt = PhysicalPlanOptimizer(enable_reorder=True)
    plan = opt.build_plan(rts, input_rows=10)
    new_rts = opt.maybe_reorder(rts, plan)

    assert [x.op_name for x in new_rts] == ["a", "b"]


def test_autotune_batch_replica_with_memory_budget():
    prof = {
        "device_hint": "gpu",
        "estimated_ms_per_sample": 2.0,
        "latency_table_ms": {
            (8, 1): 20.0,
            (16, 1): 35.0,
            (16, 2): 22.0,
        },
        "memory_table_mb": {
            (8, 1): 4000,
            (16, 1): 7000,
            (16, 2): 13000,
        },
    }
    rt = [DummyRuntime("gpu_op", DummyOp(prof))]
    opt = PhysicalPlanOptimizer(enable_reorder=False)
    plan = opt.build_plan(rt, input_rows=100, memory_budget_mb=8000)
    assert plan.stages[0].batch_size == 16
    assert plan.stages[0].replicas == 1


def test_plan_marks_cache_for_expensive_stage():
    rt = [DummyRuntime("expensive", DummyOp({"estimated_ms_per_sample": 10.0, "estimated_selectivity": 1.0}))]
    opt = PhysicalPlanOptimizer(enable_reorder=False)
    plan = opt.build_plan(rt, input_rows=1000)
    assert plan.stages[0].cache_enabled is True
    assert plan.stages[0].cache_key is not None
