"""`fit` — will this model fit on my card?

Pure arithmetic. The KV-cache formula is the one that actually bites people
at long context, and it is cross-checked in the tests against the real
measured cache bytes from kvquant on a live gemma cache.
"""
from __future__ import annotations
from dataclasses import dataclass

from .models import ModelShape, MODELS, GPUS

GB = 1024 ** 3

# bytes per element for a dtype
DTYPE_BYTES: dict[str, float] = {
    "fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5,
}


def weights_bytes(n_params: float, dtype: str = "fp16") -> float:
    return n_params * DTYPE_BYTES[dtype]


def kv_cache_bytes(shape: ModelShape, seq_len: int, batch: int = 1,
                   kv_dtype: str = "fp16") -> float:
    """Standard KV-cache size: batch * seq * layers * 2(K,V) * n_kv_heads *
    head_dim * bytes. Uses n_kv_heads (GQA/MQA shrink this), not n_heads."""
    per_tok = 2 * shape.n_layers * shape.n_kv_heads * shape.head_dim
    return batch * seq_len * per_tok * DTYPE_BYTES[kv_dtype]


@dataclass
class FitResult:
    shape: ModelShape
    gpu_gb: float
    seq_len: int
    batch: int
    weight_dtype: str
    kv_dtype: str
    overhead_gb: float
    weights_gb: float
    kv_gb: float
    total_gb: float

    @property
    def fits(self) -> bool:
        return self.total_gb <= self.gpu_gb

    @property
    def headroom_gb(self) -> float:
        return self.gpu_gb - self.total_gb

    def report(self) -> str:
        v = "FITS" if self.fits else "DOES NOT FIT"
        gpu = f"{self.gpu_gb:g} GB card"
        lines = [
            f"{self.shape.name}  @  seq={self.seq_len} batch={self.batch}  "
            f"weights={self.weight_dtype} kv={self.kv_dtype}",
            f"  weights   {self.weights_gb:6.2f} GB",
            f"  kv cache  {self.kv_gb:6.2f} GB   ({self.shape.n_kv_heads} kv-heads"
            f"{' MQA' if self.shape.n_kv_heads==1 else ' GQA' if self.shape.is_gqa else ''})",
            f"  overhead  {self.overhead_gb:6.2f} GB",
            f"  --------  {self.total_gb:6.2f} GB total",
            f"  {v} on a {gpu}  (headroom {self.headroom_gb:+.2f} GB)",
        ]
        if not self.fits:
            lines.append(_suggest(self))
        return "\n".join(lines)


def _suggest(r: FitResult) -> str:
    # cheapest single lever to make it fit
    tips = []
    if r.weight_dtype in ("fp32", "fp16", "bf16"):
        w8 = weights_bytes(r.shape.n_params, "int8") / GB
        if w8 + r.kv_gb + r.overhead_gb <= r.gpu_gb:
            tips.append("quantize weights to int8")
    if r.kv_dtype in ("fp32", "fp16", "bf16"):
        kv8 = r.kv_gb * (DTYPE_BYTES['int8'] / DTYPE_BYTES[r.kv_dtype])
        if r.weights_gb + kv8 + r.overhead_gb <= r.gpu_gb:
            tips.append("quantize the KV cache to int8 (see kvquant)")
    need = r.total_gb - r.gpu_gb
    bigger = [g for g, v in GPUS.items() if v >= r.total_gb]
    if bigger:
        tips.append(f"or use a bigger card (needs >= {r.total_gb:.1f} GB)")
    head = f"  -> need {need:.1f} GB more. "
    return head + ("; ".join(tips) if tips else "no single-lever fix.")


def fit(shape: ModelShape | str, gpu_gb: float, seq_len: int = 4096,
        batch: int = 1, weight_dtype: str = "fp16", kv_dtype: str = "fp16",
        overhead_gb: float = 1.0) -> FitResult:
    if isinstance(shape, str):
        key = shape.lower()
        if key not in MODELS:
            raise KeyError(f"unknown model {shape!r}; known: {sorted(MODELS)} "
                           f"(or pass dims via --hf)")
        shape = MODELS[key]
    wb = weights_bytes(shape.n_params, weight_dtype) / GB
    kvb = kv_cache_bytes(shape, seq_len, batch, kv_dtype) / GB
    total = wb + kvb + overhead_gb
    return FitResult(shape, gpu_gb, seq_len, batch, weight_dtype, kv_dtype,
                     overhead_gb, wb, kvb, total)


def shape_from_hf(repo: str) -> ModelShape:
    """Authoritative dims from a real HF config (needs transformers + the repo
    cached or downloadable)."""
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(repo)
    c = getattr(c, "text_config", c)
    n_heads = getattr(c, "num_attention_heads")
    n_kv = getattr(c, "num_key_value_heads", n_heads)
    hd = getattr(c, "head_dim", None) or getattr(c, "hidden_size") // n_heads
    return ModelShape(repo, _count_params(repo) or 0.0,
                      getattr(c, "num_hidden_layers"), n_heads, n_kv, hd,
                      getattr(c, "hidden_size"))


def _count_params(repo: str) -> float | None:
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=False)
        st = getattr(info, "safetensors", None)
        if st and getattr(st, "total", None):
            return float(st.total)
    except Exception:
        pass
    return None
