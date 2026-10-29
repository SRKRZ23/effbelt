"""Reference data: model shapes, GPU VRAM, API prices.

Everything here is EDITABLE and APPROXIMATE. Numbers are public-config /
published figures as of 2026-10. For an authoritative model shape, read the
real HF config with `effbelt fit --hf <repo>` instead of this table.
"""
from __future__ import annotations
from dataclasses import dataclass

PRICES_ASOF = "2026-10"


@dataclass(frozen=True)
class ModelShape:
    name: str
    n_params: float          # total parameters
    n_layers: int
    n_heads: int
    n_kv_heads: int          # < n_heads for GQA; 1 for MQA
    head_dim: int
    hidden: int

    @property
    def is_gqa(self) -> bool:
        return self.n_kv_heads < self.n_heads


# Small, honest table. Shapes from each model's public config.
MODELS: dict[str, ModelShape] = {
    "gemma-3-270m":   ModelShape("gemma-3-270m", 268e6, 18, 4, 1, 256, 640),
    "llama-3.1-8b":   ModelShape("llama-3.1-8b", 8.03e9, 32, 32, 8, 128, 4096),
    "llama-3.1-70b":  ModelShape("llama-3.1-70b", 70.6e9, 80, 64, 8, 128, 8192),
    "qwen2.5-7b":     ModelShape("qwen2.5-7b", 7.62e9, 28, 28, 4, 128, 3584),
    "qwen2.5-14b":    ModelShape("qwen2.5-14b", 14.8e9, 48, 40, 8, 128, 5120),
    "qwen2.5-32b":    ModelShape("qwen2.5-32b", 32.8e9, 64, 40, 8, 128, 5120),
    "mistral-7b":     ModelShape("mistral-7b", 7.24e9, 32, 32, 8, 128, 4096),
    "phi-3-mini":     ModelShape("phi-3-mini", 3.82e9, 32, 32, 32, 96, 3072),
}

# GPU usable VRAM in GB (nominal board memory; real free is a bit less).
GPUS: dict[str, float] = {
    "rtx-3060-12g": 12, "rtx-4090": 24, "rtx-5090": 32,
    "a100-40g": 40, "a100-80g": 80, "h100-80g": 80,
    "l4": 24, "l40s": 48, "t4": 16, "mi300x": 192,
    "apple-m-16g": 16, "apple-m-32g": 32, "apple-m-64g": 64,
}

# Rough API prices, USD per 1M tokens (input, output). APPROXIMATE, as of 2026-10.
# Edit to your real contract. Used only to turn token counts into a $ estimate.
API_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o":        (2.50, 10.00),
    "gpt-4o-mini":   (0.15, 0.60),
    "claude-sonnet": (3.00, 15.00),
    "claude-haiku":  (0.80, 4.00),
    "generic-large": (3.00, 15.00),
    "generic-small": (0.20, 0.80),
}
