from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class BenchmarkResult:
    name: str
    wall_time_s: float


def run_case(name: str, fn: Callable[[], None]) -> BenchmarkResult:
    t0 = time.perf_counter()
    fn()
    return BenchmarkResult(name=name, wall_time_s=time.perf_counter() - t0)


def run_ablation(cases: Dict[str, Callable[[], None]]) -> Dict[str, BenchmarkResult]:
    return {name: run_case(name, fn) for name, fn in cases.items()}
