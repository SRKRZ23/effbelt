# effbelt — the LLM efficiency toolbelt

Six small, honest tools that **save money and catch silent bugs** — all runnable
on a laptop with **no GPU and no API key**. Four of them need nothing but Python;
two can optionally measure real numbers on a tiny local model.

```bash
pip install effbelt            # the four pure tools
pip install "effbelt[hf]"      # + quantcheck tensors and live modelcard
```

| tool | one line | needs a model? |
|------|----------|:---:|
| `fit` | will this model fit on my card? (weights + KV cache + overhead) | no |
| `tokenwaste` | where your LLM **API bill leaks** (duplicate prompts, retries, oversized model) | no |
| `evalsanity` | **is your benchmark lying?** (contamination, degenerate data/preds) | no |
| `quantcheck` | **did your quantization actually do anything?** (silent no-op guard) | tensors only |
| `cheapest` | the cheapest config that still clears **your** quality bar | your eval |
| `modelcard` | one-page quant tradeoff card (exact memory + *measured* ppl) | optional |

Every number here was produced on an Apple-silicon MacBook, offline. Nothing is invented.

---

## Why trust the memory math? It matches a real cache to the byte.

`fit`'s KV-cache formula is cross-checked in the test suite against the **real
measured bytes** of a live Hugging Face cache (via the sister project
[kvquant](https://github.com/SRKRZ23/kvquant)):

```
KV formula check: analytic 180.0KB vs measured 180.0KB at seq=10 (0.00% off)
```

## `fit` — stop guessing at OOM

```
$ effbelt fit llama-3.1-70b --gpu rtx-4090 --seq 8192
llama-3.1-70b  @  seq=8192 batch=1  weights=fp16 kv=fp16
  weights   131.50 GB
  kv cache    2.50 GB   (8 kv-heads GQA)
  overhead    1.00 GB
  --------  135.00 GB total
  DOES NOT FIT on a 24 GB card  (headroom -111.00 GB)
```
```
$ effbelt fit llama-3.1-8b --gpu rtx-4090 --seq 32768 --kv int8
  ... FITS on a 24 GB card  (headroom +6.04 GB)
```
Use `--hf <repo>` to pull authoritative dimensions straight from a model's config.

## `tokenwaste` — audit an API request log (JSONL)

```
$ effbelt tokenwaste requests.jsonl
22 requests, ~$0.0780 total (prices as of 2026-10)
  recoverable up to ~$0.0780 (<= 100% of bill; levers below can overlap):
  - oversized-model   ~$0.0718  large model, <=20 output tok -> try a small model
  - over-long-context ~$0.0300  context over 8000 tok
  - no-prefix-cache   ~$0.0183  22x identical system prompt (~290 tok)
  - retries           ~$0.0013  requests marked retry=true
```
The levers overlap on the same request, so the headline is an honest **upper
bound** capped at 100% of the bill — never a made-up "save 300%".

## `evalsanity` — before you believe a leaderboard number

```
$ effbelt evalsanity eval.json
41 test items: 2 flag(s)
  [HIGH] 1/41 test items appear VERBATIM in train -> contamination
  [MED ] labels 93% one class ('A') -> imbalanced
  -> at least one HIGH flag: do NOT trust this number yet.
```

## `quantcheck` — the silent no-op killer

The lesson behind kvquant, generalized to **any** tensor transform you think
quantizes. A no-op (e.g. you mutated a non-persisting view) passes every smoke
test and moves no metric — quantcheck proves it *bit*, estimates effective
bits, and flags the two classic bugs:

```python
from effbelt import quantcheck
r = quantcheck.check(my_quant_fn, some_tensor, claimed_bits=8)
print(r.report())
# quantcheck: SILENT NO-OP   <- if your "quantizer" changed nothing
# quantcheck: SUSPICIOUS     <- not idempotent / coarser than claimed
# quantcheck: OK
```

## `modelcard` — know the tradeoff before you quantize

Exact memory arithmetic, and *measured* quality when you point it at a model:

```
$ effbelt modelcard gemma-3-270m --seq 8192 --live google/gemma-3-270m-it
  kv dtype   kv cache  vs fp16   quality
  fp16         0.14GB    1.00x   +0.00% ppl
  int8         0.07GB    2.00x   -0.33% ppl     <- free: 2x smaller, lossless
  int4         0.04GB    4.00x  +68.41% ppl     <- breaks on this small model
  quality measured on google/gemma-3-270m-it (160 tok, MPS/CPU)
```

## `cheapest` — smallest model that still passes

```python
from effbelt import cheapest
res = cheapest.sweep([
    cheapest.Candidate("7b-int4",  cost=1, quality_fn=lambda: my_eval("7b-int4")),
    cheapest.Candidate("7b-fp16",  cost=2, quality_fn=lambda: my_eval("7b-fp16")),
    cheapest.Candidate("70b-int4", cost=8, quality_fn=lambda: my_eval("70b-int4")),
], bar=0.80)
print(res.report())   # -> use the cheapest one that clears 0.80
```

---

## Honest scope
- Prices and the small model table are **approximate and editable** (see
  `effbelt/models.py`); for authoritative model dims use `fit --hf`.
- `quantcheck`'s effective-bit estimate is advisory at small group sizes — the
  *silent-no-op* and *idempotency* checks are the reliable ones.
- `modelcard` quality numbers are measured only when you pass `--live`; otherwise
  the quality column is left blank. We don't fill it with guesses.

## Tests
All validation runs on a laptop, offline:
```bash
python -m pytest tests/ -v      # 18 tests, incl. the real-cache cross-check
```

## Support
Free and MIT. If it saved you a GPU or a bill, consider sponsoring:
**https://github.com/sponsors/SRKRZ23**

Crypto (receive-only):
- BTC `bc1q9hm4upksmvkzhl8z3zvxgz94da358qfe3tred0`
- ETH / Polygon / USDT-ERC20 `0xa9e3b63799f5E5cc6258042D45869a5a1d39F021`
- USDT-TRC20 `TUyNkMXme1CYAc1wZ8SZ64X9upBc9nwTZ5`
- SOL `j8zZCsWwM8WmUZwMSDHikNuz6E49Vn8nwo9FMQnHSoW`

Sister project: [kvquant](https://github.com/SRKRZ23/kvquant) — the engine behind
`quantcheck` and `modelcard`.
