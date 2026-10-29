"""`tokenwaste` — where your LLM API bill leaks.

Point it at a JSONL log of requests. Deterministic analysis, no model calls.
Each record is one request; recognised fields (all optional):
  {"model": "gpt-4o", "system": "...", "prompt": "...",
   "input_tokens": 1234, "output_tokens": 56, "retry": false,
   "context_tokens": 1234}
If token counts are missing they are estimated at ~4 chars/token from the
text fields, and the report says so.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from .models import API_PRICES, PRICES_ASOF

CHARS_PER_TOKEN = 4.0


def _est_tokens(text: str | None) -> int:
    return 0 if not text else max(1, round(len(text) / CHARS_PER_TOKEN))


@dataclass
class Record:
    model: str = "generic-large"
    system: str = ""
    prompt: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    context_tokens: int | None = None
    retry: bool = False
    estimated: bool = False

    def in_tok(self) -> int:
        if self.input_tokens is not None:
            return self.input_tokens
        self.estimated = True
        return _est_tokens(self.system) + _est_tokens(self.prompt)

    def out_tok(self) -> int:
        if self.output_tokens is not None:
            return self.output_tokens
        self.estimated = True
        return 0


def _price(model: str) -> tuple[float, float]:
    return API_PRICES.get(model, API_PRICES["generic-large"])


def _cost(model: str, in_tok: int, out_tok: int) -> float:
    pin, pout = _price(model)
    return (in_tok * pin + out_tok * pout) / 1_000_000


@dataclass
class WasteReport:
    n: int
    total_usd: float
    estimated: bool
    findings: list[tuple[str, float, str]] = field(default_factory=list)
    # (category, recoverable_usd, detail)

    @property
    def recoverable_usd(self) -> float:
        """Sum of per-lever savings, CAPPED at the total bill. The levers can
        overlap on the same request (e.g. oversized-model and over-long-context
        both touch one call), so the raw sum is an upper bound, not additive
        truth -- we never report more than 100% of the bill."""
        return min(sum(r for _, r, _ in self.findings), self.total_usd)

    def report(self) -> str:
        head = (f"{self.n} requests, ~${self.total_usd:.4f} total "
                f"(prices as of {PRICES_ASOF}"
                + (", token counts ESTIMATED from text" if self.estimated else "")
                + ")")
        if not self.findings:
            return head + "\n  no obvious waste found."
        pct = 100 * self.recoverable_usd / self.total_usd if self.total_usd else 0
        lines = [head, f"  recoverable up to ~${self.recoverable_usd:.4f} "
                 f"(<= {pct:.0f}% of bill; levers below can overlap):"]
        for cat, usd, detail in sorted(self.findings, key=lambda x: -x[1]):
            lines.append(f"  - {cat:22s} ~${usd:.4f}  {detail}")
        return "\n".join(lines)


def analyze(records: list[Record],
            long_context_tokens: int = 8000) -> WasteReport:
    total = 0.0
    estimated = False
    sys_counts: dict[str, int] = {}
    sys_in: dict[str, int] = {}
    retry_usd = 0.0
    longctx_usd = 0.0
    small_ok_usd = 0.0  # large model on tiny output (candidate to downsize)

    for r in records:
        it, ot = r.in_tok(), r.out_tok()
        estimated = estimated or r.estimated
        c = _cost(r.model, it, ot)
        total += c
        # duplicate system prompts with no prefix cache -> repeated input cost
        if r.system:
            sys_counts[r.system] = sys_counts.get(r.system, 0) + 1
            sys_in[r.system] = _est_tokens(r.system) if r.input_tokens is None \
                else sys_in.get(r.system, _est_tokens(r.system))
        # retries are pure waste
        if r.retry:
            retry_usd += c
        # over-long context
        ctx = r.context_tokens if r.context_tokens is not None else it
        if ctx >= long_context_tokens:
            over = ctx - long_context_tokens
            longctx_usd += _cost(r.model, over, 0)
        # large model whose output is tiny -> maybe a small model would do
        pin, _ = _price(r.model)
        if pin >= 2.0 and ot <= 20:
            small_ok_usd += c - _cost("generic-small", it, ot)

    findings: list[tuple[str, float, str]] = []

    # prefix-cache opportunity: a system prompt sent N>1 times pays its tokens
    # every time without a cache. Recoverable ~= (N-1) * sys_tokens * in-price.
    dup_usd = 0.0
    dup_detail = []
    for sysp, cnt in sys_counts.items():
        if cnt > 1:
            stok = sys_in[sysp]
            # assume the common 'generic-large' input price if unknown mix
            saved = _cost("generic-large", stok * (cnt - 1), 0)
            dup_usd += saved
            dup_detail.append(f"{cnt}x identical system prompt (~{stok} tok)")
    if dup_usd > 0:
        findings.append(("no-prefix-cache", dup_usd,
                         "; ".join(dup_detail[:3])
                         + ("" if len(dup_detail) <= 3 else f"; +{len(dup_detail)-3} more")))
    if retry_usd > 0:
        findings.append(("retries", retry_usd, "requests marked retry=true"))
    if longctx_usd > 0:
        findings.append(("over-long-context", longctx_usd,
                         f"context over {long_context_tokens} tok"))
    if small_ok_usd > 0:
        findings.append(("oversized-model", small_ok_usd,
                         "large model, <=20 output tok -> try a small model"))
    return WasteReport(len(records), total, estimated, findings)


def load_jsonl(path: str) -> list[Record]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(Record(
                model=d.get("model", "generic-large"),
                system=d.get("system", ""),
                prompt=d.get("prompt", ""),
                input_tokens=d.get("input_tokens"),
                output_tokens=d.get("output_tokens"),
                context_tokens=d.get("context_tokens"),
                retry=bool(d.get("retry", False)),
            ))
    return out
