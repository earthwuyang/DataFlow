from dataflow.planner.runtime_apply import apply_physical_plan_end_to_end, update_cost_model_from_telemetry
from dataflow.planner.telemetry import TelemetryStore


class DummyOp:
    def __init__(self, profile=None):
        self.physical_profile = profile or {}


class DummyRuntime:
    def __init__(self, name, op):
        self.op_name = name
        self.op = op


def test_end_to_end_directives_and_placement():
    rts = [
        DummyRuntime("exp", DummyOp({"pushdown_safe": False})),
        DummyRuntime("flt", DummyOp({"pushdown_safe": True, "estimated_selectivity": 0.1, "estimated_ms_per_sample": 0.1})),
    ]
    d = apply_physical_plan_end_to_end(rts, num_gpus=2)
    assert d["ordered_ops"][0] == "flt"
    assert len(d["placement"]) == 2


def test_telemetry_updates_cost_model():
    rt = DummyRuntime("op1", DummyOp({}))
    tele = TelemetryStore()
    tele.record("op1", 3.0)
    tele.record("op1", 5.0)
    update_cost_model_from_telemetry([rt], tele)
    assert rt.op.physical_profile["estimated_ms_per_sample"] == 4.0
