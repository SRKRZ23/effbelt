"""`cheapest` — the smallest/cheapest config that still passes YOUR bar.

You give candidate configs (each with a relative cost and a way to score
quality on your own task) and a quality bar. cheapest runs the sweep and
returns the lowest-cost config that clears the bar. The quality function is
yours (an eval on your data); the logic here is just the honest sweep +
pick, so the recommendation is only as good as your bar and your eval.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


@dataclass
class Candidate:
    name: str
    cost: float                 # relative $ or GB or latency -- your unit
    quality_fn: Callable[[], float]   # returns a score; higher is better


@dataclass
class SweepRow:
    name: str
    cost: float
    quality: float
    passes: bool


@dataclass
class CheapestResult:
    bar: float
    rows: list[SweepRow]
    pick: SweepRow | None

    def report(self) -> str:
        lines = [f"quality bar = {self.bar:g}"]
        for r in sorted(self.rows, key=lambda x: x.cost):
            mark = "PASS" if r.passes else "fail"
            star = "  <= cheapest passing" if self.pick and r.name == self.pick.name else ""
            lines.append(f"  cost {r.cost:8.3f}  q={r.quality:.4f}  [{mark}]  "
                         f"{r.name}{star}")
        if self.pick:
            lines.append(f"-> use {self.pick.name!r} (cost {self.pick.cost:g}, "
                         f"q={self.pick.quality:.4f})")
        else:
            lines.append("-> NOTHING clears the bar; raise budget or lower the bar.")
        return "\n".join(lines)


def sweep(candidates: list[Candidate], bar: float) -> CheapestResult:
    rows = []
    for c in candidates:
        q = float(c.quality_fn())
        rows.append(SweepRow(c.name, c.cost, q, q >= bar))
    passing = [r for r in rows if r.passes]
    pick = min(passing, key=lambda r: r.cost) if passing else None
    return CheapestResult(bar, rows, pick)
