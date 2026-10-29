# effbelt launch copy (honest, measured)

Repo: https://github.com/SRKRZ23/effbelt · Sponsors: https://github.com/sponsors/SRKRZ23
All numbers measured on an Apple-silicon MacBook, offline. Nothing invented.

---

## Hacker News (Show HN)

**Title (72 chars, limit 80):**
```
Show HN: effbelt – 6 tiny tools to stop wasting GPU memory and API money
```

**Body:**
```
effbelt is six small, honest tools for LLM work that run on a laptop — four of
them need no GPU and no API key:

- fit         will this model fit on my card? (weights + KV cache + overhead, GQA/MQA aware)
- tokenwaste  where your API bill leaks (duplicate system prompts, retries, over-long context, oversized model)
- evalsanity  is your benchmark lying? (train/test contamination, degenerate data/preds, imbalance)
- quantcheck  did your quantization actually do anything? (silent no-op + idempotency guard)
- cheapest    the cheapest config that still clears your own quality bar
- modelcard   one-page quant tradeoff card (exact memory + measured ppl)

Two things I tried hard to keep honest:

1) fit's KV-cache formula is cross-checked in the test suite against the REAL
   measured bytes of a live Hugging Face cache — analytic 180.0KB vs measured
   180.0KB, 0.00% off. I didn't want to ship a memory estimator I only eyeballed.

2) tokenwaste's savings levers overlap on the same request, so the headline is
   an upper bound capped at 100% of the bill — never a "save 300%".

quantcheck generalizes a bug I hit building my previous tool (kvquant): most
KV-quant snippets mutate a non-persisting view on transformers 5.x and silently
do nothing — perplexity never moves, no error. quantcheck proves the transform
actually bit.

Measured on gemma-3-270m: int8 KV cache = 2x smaller, -0.33% ppl (free); int4 =
4x smaller but +68% ppl (breaks on a model that small). 18/18 tests pass on a
MacBook, offline.

MIT. pip install effbelt (the four pure tools) or pip install "effbelt[hf]" for
the two that touch a model. Feedback welcome, especially on more model shapes
and real price tables.

https://github.com/SRKRZ23/effbelt
```

---

## LinkedIn (one post)

```
I kept seeing the same two mistakes in LLM projects: teams paying for GPU memory
and API tokens nobody needed, and "quantization" code that silently does nothing.

So I built effbelt — six tiny tools that catch exactly that, and they run on a
laptop. Four of the six need no GPU and no API key.

What's in it:
🧮 fit — will this model fit on my card? (weights + KV cache + overhead, and it
   uses the real KV-head count, so it's correct for GQA/MQA models)
💸 tokenwaste — point it at an API request log and see where the bill leaks:
   duplicate system prompts, retries, over-long context, an oversized model
🔍 evalsanity — catch train/test contamination and degenerate data BEFORE you
   trust a benchmark number
✅ quantcheck — prove your quantization actually changed the tensors, instead of
   silently doing nothing
📉 cheapest — find the smallest model that still clears your own quality bar
📋 modelcard — see the memory-vs-quality tradeoff before you quantize

Two things I cared about: every number is measured on a MacBook, and I refused
to overclaim. The memory math is even cross-checked against a real model's live
cache — down to 0.00% off. And the cost tool reports an honest upper bound, never
a "save 300%".

On a small model, int8 KV cache came out 2× smaller for free (−0.33% perplexity);
int4 was 4× smaller but broke it (+68%). Know before you quantize.

Free and MIT. If it saves you a GPU or a bill, you can sponsor the work.

Built with an AI coding assistant (and a lot of validation runs).

👉 https://github.com/SRKRZ23/effbelt

#LLM #MachineLearning #OpenSource #GPU #Inference #MLOps #AIEngineering #BuildInPublic
```

(Delete the AI-assist line if you prefer — your call. Tag Hugging Face / real ML
contacts in the composer; don't cold-tag strangers.)

---

## X / Twitter (thread, each tweet < 280 weighted)

**1/5**
```
effbelt: six tiny tools to stop wasting GPU memory and LLM API money, and to catch quantization that silently does nothing.

All MIT, run on a laptop, 4 of them need no GPU and no API key.

https://github.com/SRKRZ23/effbelt
```

**2/5**
```
The one I trust most: `fit` tells you if a model fits your card (weights + KV cache + overhead).

Its KV-cache formula is cross-checked against the REAL measured bytes of a live HF cache: analytic 180.0KB vs measured 180.0KB, 0.00% off.
```

**3/5**
```
`tokenwaste` audits an API request log: duplicate system prompts (no prefix cache), retries, over-long context, oversized model.

The levers overlap, so it reports an honest upper bound capped at 100% of your bill, never "save 300%".
```

**4/5**
```
`quantcheck`: did your quantization actually do anything?

Most KV-quant snippets mutate a non-persisting view and silently no-op: ppl never moves, no error. quantcheck proves the transform bit, and flags no-ops and double-quant.
```

**5/5**
```
`modelcard` measures the real tradeoff on a model. On gemma-3-270m:

int8 KV = 2x smaller, -0.33% ppl (free)
int4 KV = 4x smaller, +68% ppl (breaks)

Know before you quantize. Free + MIT; sponsor if it saved you a GPU:
https://github.com/sponsors/SRKRZ23
```

---

## Russian (short, one post — Telegram / local)

```
Собрал effbelt — шесть маленьких инструментов, чтобы не жечь память GPU и деньги
на LLM-API, и ловить квантизацию, которая по-тихому ничего не делает.

Четырём из шести не нужны ни GPU, ни API-ключ. Всё MIT, считается на ноутбуке.

fit — влезет ли модель на карту (веса + KV-кэш + оверхед, с учётом GQA/MQA)
tokenwaste — где утекает счёт за API
evalsanity — врёт ли бенчмарк (контаминация, вырожденные данные)
quantcheck — реально ли сработала квантизация (а не тихий no-op)
cheapest / modelcard — дешевейшая модель под планку / карта tradeoff

Формула памяти сверена с реальным кэшем модели до 0.00%. Честные числа, без
"сэкономь 300%".

https://github.com/SRKRZ23/effbelt
```
