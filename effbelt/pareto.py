"""`pareto` -- the accuracy-vs-cost frontier and cost-per-correct, so a cost cut
that quietly loses accuracy cannot hide behind a headline percentage.

You give candidate configs (full model, int8, a compressed context, a cheaper
model, ...), each with a relative cost and either per-item correctness on a fixed
task set or an accuracy directly. `pareto` reports:
  - **cost-per-correct** -- the cost to buy one correct answer on average
    (cost / accuracy); the honest efficiency number, since a big token cut with
    a low accuracy may still need so many retries that it is not actually cheaper.
  - the **Pareto frontier** -- the configs that no other config beats on BOTH
    cost and accuracy (the real tradeoff curve), and the dominated ones.
  - **traps** -- a config that advertises a big cost cut but either falls below
    your accuracy floor, or does not even improve cost-per-correct.

"A 90% token cut that loses 15% accuracy is usually a worse deal than it looks"
becomes visible here. The accuracy is yours (an eval on your data); the logic is
only the honest frontier + cost-per-correct, so it is as good as your eval.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math


@dataclass
class Config:
    name: str
    cost: float                              # relative cost/task: tokens, $, latency -- your unit
    correct: Sequence[bool] | None = None    # per-item correctness on a fixed task set
    accuracy: float | None = None            # or give accuracy directly (0..1)
    n: int | None = None                     # task count (needed if accuracy is given directly)

    def acc(self) -> float:
        if self.correct is not None:
            c = list(self.correct)
            return sum(1 for x in c if x) / len(c) if c else 0.0
        if self.accuracy is None:
            raise ValueError(f"{self.name}: give correct[] or accuracy")
        if not 0.0 <= float(self.accuracy) <= 1.0:
            raise ValueError(f"{self.name}: accuracy must be in [0,1]")
        return float(self.accuracy)

    def count(self) -> int:
        if self.correct is not None:
            return len(list(self.correct))
        return int(self.n or 0)


def _wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n. Stdlib only."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / d, (centre + half) / d)


@dataclass
class Row:
    name: str
    cost: float
    accuracy: float
    n: int
    cost_per_correct: float
    on_frontier: bool
    dominated_by: str | None
    distinguishable: bool | None = None   # is the accuracy gap vs baseline real at this N?


@dataclass
class Trap:
    name: str
    cost_cut_pct: float       # headline cost cut vs the most-accurate (baseline) config
    accuracy_drop_pp: float   # percentage points below baseline accuracy
    note: str


@dataclass
class ParetoResult:
    rows: list[Row]
    traps: list[Trap]
    baseline: str
    floor: float | None
    min_n: int

    def report(self) -> str:
        lines: list[str] = []
        if self.min_n and self.min_n < 30:
            lines.append(f"! small task set (N={self.min_n}): accuracy gaps may be noise, not signal.")
        head = f"baseline (most accurate) = {self.baseline!r}"
        if self.floor is not None:
            head += f"; accuracy floor = {self.floor:g}"
        lines.append(head)
        lines.append("   cost      acc     $/correct  frontier  config")
        any_noise = False
        for r in sorted(self.rows, key=lambda x: x.cost):
            flag = "   *    " if r.on_frontier else "  dom   "
            cpc = "     inf" if math.isinf(r.cost_per_correct) else f"{r.cost_per_correct:8.3f}"
            tail = "" if r.on_frontier else f"  (beaten by {r.dominated_by})"
            if r.distinguishable is False:
                tail += "  [~=baseline: gap within noise at this N]"
                any_noise = True
            lines.append(f"  {r.cost:8.3f}  {r.accuracy:.4f}  {cpc}  {flag}  {r.name}{tail}")
        if any_noise:
            lines.append("  (~= means the accuracy gap vs baseline is inside the 95% Wilson interval "
                         "-- treat those two as tied, not better/worse.)")
        if self.traps:
            lines.append("TRAPS (a big cost cut that is not the deal it looks like):")
            for t in self.traps:
                lines.append(f"  ! {t.name}: cost -{t.cost_cut_pct:.0f}% but accuracy -{t.accuracy_drop_pp:.1f}pp"
                             f" -> {t.note}")
        else:
            lines.append("no traps: every cheaper config stays within the floor and improves $/correct.")
        return "\n".join(lines)


def analyze(configs: list[Config], floor: float | None = None,
            min_cut_pct: float = 20.0) -> ParetoResult:
    """Build the frontier, cost-per-correct, and trap flags. Returns a ParetoResult."""
    if not configs:
        raise ValueError("give at least one config")
    pts = [(c.name, float(c.cost), c.acc(), c.count()) for c in configs]
    # baseline = the most accurate config (ties broken by lower cost)
    bname, bcost, bacc, bn = max(pts, key=lambda p: (p[2], -p[1]))
    bcpc = bcost / bacc if bacc > 0 else math.inf

    b_lo, b_hi = _wilson(round(bacc * bn), bn) if bn else (0.0, 1.0)
    rows: list[Row] = []
    for name, cost, acc, n in pts:
        cpc = cost / acc if acc > 0 else math.inf
        dom = None
        for n2, c2, a2, _ in pts:
            if n2 == name:
                continue
            # n2 dominates name if it is no worse on both and strictly better on one
            if c2 <= cost and a2 >= acc and (c2 < cost or a2 > acc):
                dom = n2
                break
        # is this config's accuracy gap from the baseline real, or within noise at this N?
        if name == bname or not n:
            distinguishable = None
        else:
            lo, hi = _wilson(round(acc * n), n)
            distinguishable = (hi < b_lo) or (lo > b_hi)   # intervals do not overlap
        rows.append(Row(name, cost, acc, n, cpc, dom is None, dom, distinguishable))

    traps: list[Trap] = []
    for name, cost, acc, n in pts:
        if name == bname:
            continue
        cut = (bcost - cost) / bcost * 100 if bcost > 0 else 0.0
        cpc = cost / acc if acc > 0 else math.inf
        below_floor = floor is not None and acc < floor
        worse_per_correct = cpc >= bcpc
        if cut >= min_cut_pct and (below_floor or worse_per_correct):
            if below_floor:
                note = f"accuracy {acc:.3f} is below your floor {floor:g} (no retry saves you)"
            else:
                note = (f"$/correct {cpc:.3f} is not better than the baseline {bcpc:.3f} "
                        f"-- the cut is eaten by wrong answers")
            traps.append(Trap(name, cut, (bacc - acc) * 100, note))

    min_n = min((n for _, _, _, n in pts if n), default=0)
    return ParetoResult(rows, traps, bname, floor, min_n)
