from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List


@dataclass
class OpTelemetry:
    op_name: str
    latencies_ms: List[float] = field(default_factory=list)
    oom_count: int = 0

    def p50(self) -> float:
        vals = sorted(self.latencies_ms)
        if not vals:
            return 0.0
        return vals[len(vals)//2]

    def avg(self) -> float:
        return mean(self.latencies_ms) if self.latencies_ms else 0.0


class TelemetryStore:
    def __init__(self):
        self.ops: Dict[str, OpTelemetry] = {}

    def record(self, op_name: str, latency_ms: float, oom: bool = False):
        t = self.ops.setdefault(op_name, OpTelemetry(op_name=op_name))
        t.latencies_ms.append(float(latency_ms))
        if oom:
            t.oom_count += 1

    def profile_snapshot(self) -> Dict[str, Dict[str, float]]:
        snap = {}
        for name, t in self.ops.items():
            snap[name] = {
                "avg_latency_ms": t.avg(),
                "p50_latency_ms": t.p50(),
                "oom_count": float(t.oom_count),
            }
        return snap
