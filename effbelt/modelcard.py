"""`modelcard` — one-page quantization tradeoff card for a model.

Combines the arithmetic `fit` memory model with REAL measured KV-quant
quality (from kvquant, when a live model is supplied). The memory column is
always exact arithmetic; the quality column is only filled with measured
numbers when you pass a live model, and is left blank otherwise (we do not
invent quality numbers).
"""
from __future__ import annotations
from dataclasses import dataclass, field

from .models import ModelShape, MODELS
from .memory import fit, DTYPE_BYTES, GB, kv_cache_bytes


@dataclass
class CardRow:
    kv_dtype: str
    kv_gb: float
    ppl_delta_pct: float | None   # measured, or None if not run

    def q(self) -> str:
        if self.ppl_delta_pct is None:
            return "   (not measured)"
        return f"{self.ppl_delta_pct:+.2f}% ppl"


@dataclass
class ModelCard:
    shape: ModelShape
    seq_len: int
    batch: int
    weight_dtype: str
    weights_gb: float
    rows: list[CardRow] = field(default_factory=list)
    measured_on: str | None = None

    def report(self) -> str:
        base = self.rows[0].kv_gb if self.rows else 0.0
        lines = [
            f"model card: {self.shape.name}  (seq={self.seq_len} batch={self.batch}, "
            f"weights {self.weight_dtype} = {self.weights_gb:.2f} GB)",
            f"  {'kv dtype':9s} {'kv cache':>9s} {'vs fp16':>8s}   quality",
        ]
        for r in self.rows:
            ratio = (base / r.kv_gb) if r.kv_gb else float("inf")
            lines.append(f"  {r.kv_dtype:9s} {r.kv_gb:7.2f}GB {ratio:7.2f}x   {r.q()}")
        if self.measured_on:
            lines.append(f"  quality measured on {self.measured_on}")
        else:
            lines.append("  quality not measured (pass a live model to fill it)")
        return "\n".join(lines)


def build(shape: ModelShape | str, seq_len: int = 4096, batch: int = 1,
          weight_dtype: str = "fp16",
          kv_dtypes: tuple[str, ...] = ("fp16", "int8", "int4"),
          measured_ppl: dict[str, float] | None = None,
          measured_on: str | None = None) -> ModelCard:
    if isinstance(shape, str):
        shape = MODELS[shape.lower()]
    wgb = fit(shape, 1e9, seq_len, batch, weight_dtype, "fp16").weights_gb
    rows = []
    for kv in kv_dtypes:
        kvg = kv_cache_bytes(shape, seq_len, batch, kv) / GB
        delta = (measured_ppl or {}).get(kv)
        rows.append(CardRow(kv, kvg, delta))
    return ModelCard(shape, seq_len, batch, weight_dtype, wgb, rows, measured_on)


def measure_kv_quality(repo: str, text: str | None = None,
                       bits_map: dict[str, int] | None = None,
                       max_tokens: int = 200) -> dict[str, float]:
    """REAL measured KV-quant quality via kvquant on a live model. Returns
    {kv_dtype: ppl_delta_pct_vs_fp16}. Honest, uses the proven engine."""
    from kvquant import kv_ppl_sweep
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    bits_map = bits_map or {"int8": 8, "int4": 4}
    text = text or ("The quick brown fox jumps over the lazy dog. " * 40)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).to(dev).eval()
    bits_list = (None,) + tuple(sorted(set(bits_map.values()), reverse=True))
    # kv_ppl_sweep -> {bits: {"ppl","delta_pct","change"}}
    sweep = kv_ppl_sweep(model, tok, text, bits_list=bits_list, max_tokens=max_tokens)
    out = {"fp16": 0.0}
    for kv, b in bits_map.items():
        row = sweep.get(b)
        if row is not None and row.get("delta_pct") is not None:
            out[kv] = float(row["delta_pct"])
    return out
