from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class PlacementDecision:
    op_name: str
    gpu_id: int


def round_robin_gpu_placement(op_names: List[str], num_gpus: int) -> List[PlacementDecision]:
    if num_gpus <= 0:
        return [PlacementDecision(op_name=n, gpu_id=-1) for n in op_names]
    out = []
    for i, n in enumerate(op_names):
        out.append(PlacementDecision(op_name=n, gpu_id=i % num_gpus))
    return out
