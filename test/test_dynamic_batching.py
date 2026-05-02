import importlib.util
from pathlib import Path


def _load_module():
    path = Path('dataflow/planner/dynamic_batching.py')
    spec = importlib.util.spec_from_file_location('dynamic_batching', path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_dynamic_batch_grows_when_latency_low():
    m = _load_module()
    ctrl = m.DynamicBatchController()
    state = m.DynamicBatchState(current_batch_size=4, target_latency_ms=100)
    nxt = ctrl.update(state, observed_latency_ms=40, oom=False, profile={"complexity": "low"})
    assert nxt >= 8


def test_dynamic_batch_shrinks_on_oom():
    m = _load_module()
    ctrl = m.DynamicBatchController()
    state = m.DynamicBatchState(current_batch_size=16, target_latency_ms=100)
    nxt = ctrl.update(state, observed_latency_ms=90, oom=True, profile={})
    assert nxt == 8
