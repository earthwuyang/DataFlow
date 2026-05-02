# DataFlow Physical Optimization Update

This document summarizes the physical-optimization work added to the DataFlow framework and provides reproduction guidance.

## 1) What was added

### 1.1 Planner package
A new `dataflow/planner` package has been added with the following modules:

- `physical_optimizer.py`
  - Physical profile and plan dataclasses.
  - Batch/replica autotuning under optional memory budget.
  - Stage-level cache decision annotations.
  - Optional conservative reordering (`reorder_safe` gated).
- `dynamic_batching.py`
  - Runtime feedback controller for batch size adaptation.
  - OOM-triggered downscale.
  - Latency-target adaptation.
  - LLM-style advisory hook (`complexity`, `memory_sensitive`).
- `execution_cache.py`
  - Deterministic stage cache keys.
  - Heuristic cache decision policy for expensive/selective stages.
- `telemetry.py`
  - Per-operator latency/OOM telemetry storage.
  - Snapshot export for cost model updates.
- `rewrite.py`
  - Safe filter pushdown rule (profile-gated).
  - Metadata-level operator fusion grouping for light CPU operators.
- `placement.py`
  - Multi-GPU round-robin placement policy.
- `runtime_apply.py`
  - End-to-end planner application helper (rewrite + fusion + placement directives).
  - Cost-model update utility from telemetry.
- `benchmark/harness.py`
  - Lightweight benchmark + ablation runner.

### 1.2 Pipeline integration
`PipelineABC` now exposes planner-facing state and helper hooks:

- `physical_plan`
- `physical_optimizer`
- `enable_physical_planning`
- `apply_physical_reordering`
- `physical_memory_budget_mb`
- `enable_dynamic_batching`
- `dynamic_batch_controller`
- `dynamic_batch_states`
- `telemetry_store`

New helper methods:

- `suggest_dynamic_batch_size(...)`
- `apply_planner_decisions(num_gpus=...)`
- `profile_and_update_cost_model()`

`compile()` builds a plan summary by default and keeps runtime behavior conservative unless reordering is explicitly enabled.

## 2) Physical profile fields

Operators can optionally expose `op.physical_profile` as a dictionary. Supported keys include:

- `device_hint`: `"cpu" | "gpu"`
- `estimated_ms_per_sample`: float
- `estimated_selectivity`: float in [0,1]
- `gpu_memory_mb`: float
- `batch_size_hint`: int
- `replica_hint`: int
- `latency_table_ms`: `{(batch, replicas): latency_ms}`
- `memory_table_mb`: `{(batch, replicas): memory_mb}`
- `reorder_safe`: bool
- `pushdown_safe`: bool
- `fusible`: bool
- `complexity`: `"low" | "medium" | "high"`
- `memory_sensitive`: bool

## 3) Reproduction guidance

## 3.1 Environment setup

```bash
cd /workspace/DataFlow
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

If missing test deps in your environment, install common extras:

```bash
pip install requests colorlog pytest
```

## 3.2 Static sanity check

```bash
python -m py_compile \
  dataflow/planner/physical_optimizer.py \
  dataflow/planner/dynamic_batching.py \
  dataflow/planner/execution_cache.py \
  dataflow/planner/telemetry.py \
  dataflow/planner/rewrite.py \
  dataflow/planner/placement.py \
  dataflow/planner/runtime_apply.py \
  dataflow/planner/benchmark/harness.py \
  dataflow/pipeline/Pipeline.py
```

## 3.3 Unit tests

Run the new focused tests:

```bash
pytest -q test/test_physical_plan_optimizer.py
pytest -q test/test_dynamic_batching.py
pytest -q test/test_planner_runtime_apply.py
```

## 3.4 Minimal API walkthrough

### Build plan and inspect stages

```python
from dataflow.planner import PhysicalPlanOptimizer

opt = PhysicalPlanOptimizer(enable_reorder=False)
plan = opt.build_plan(op_runtimes, input_rows=10000, memory_budget_mb=12000)
for s in plan.stages:
    print(s.op_name, s.device, s.batch_size, s.replicas, s.cache_enabled, s.cache_reason)
```

### Dynamic batch update

```python
next_bs = pipeline.suggest_dynamic_batch_size(
    op_name="vqa_stage",
    observed_latency_ms=83.2,
    oom=False,
    profile={"complexity": "medium", "memory_sensitive": False},
    target_latency_ms=100,
)
```

### Apply planner directives end-to-end (metadata level)

```python
directives = pipeline.apply_planner_decisions(num_gpus=4)
print(directives["ordered_ops"])
print(directives["placement"])
```

### Update cost model from telemetry

```python
pipeline.telemetry_store.record("vqa_stage", latency_ms=92.0)
pipeline.telemetry_store.record("vqa_stage", latency_ms=88.0)
pipeline.profile_and_update_cost_model()
```

## 4) Current scope and limitations

- Reordering is conservative and explicitly gated.
- Fusion is metadata-level in this version (group tagging only).
- Placement uses round-robin; no topology-aware packing yet.
- Runtime application currently returns directives and updates profiles; full queue-based heterogeneous scheduler integration is a follow-up.

## 5) Recommended next steps

1. Integrate directives with actual stage executor/runtime.
2. Add queue/backpressure and prefetch overlap for CPU/GPU pipelines.
3. Replace heuristic cache policy with cost + reuse model and storage budget.
4. Add topology-aware placement policy (GPU memory/util/affinity aware).
5. Expand benchmark harness with end-to-end workload presets and CSV logging.
