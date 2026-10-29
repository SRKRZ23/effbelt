"""effbelt — the LLM efficiency toolbelt.

Six small, honest tools that save money and catch silent bugs, all runnable
on a laptop with no GPU and no API key:

  fit         will this model fit on my card? (weights + KV cache + overhead)
  tokenwaste  where your LLM API bill leaks (duplicate prompts, retries, ...)
  evalsanity  is your benchmark lying? (contamination, degenerate data/preds)
  quantcheck  did your quantization actually do anything? (silent no-op guard)
  cheapest    the cheapest config that still clears your quality bar
  modelcard   one-page quant tradeoff card (arithmetic memory + measured ppl)

Sister project: kvquant (the engine behind quantcheck/modelcard).
MIT. If it saved you a GPU, consider sponsoring.
"""
__version__ = "0.1.0"

from . import memory, tokenwaste, evalsanity, quantcheck, cheapest, modelcard

__all__ = ["memory", "tokenwaste", "evalsanity", "quantcheck", "cheapest",
           "modelcard", "__version__"]
