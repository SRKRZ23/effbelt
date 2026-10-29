"""Validation tests for effbelt — all runnable on a MacBook, no GPU/API.

Deterministic tests for every tool, plus:
  - the KV-cache formula cross-checked against kvquant's REAL measured bytes
    on a live gemma cache (the credibility anchor),
  - one short live gemma modelcard measurement.
Run: python -m pytest tests/ -v   (or: python tests/test_effbelt.py)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from effbelt import memory, tokenwaste, evalsanity, quantcheck, cheapest, modelcard
from effbelt.models import MODELS

GB = memory.GB
REPO = "google/gemma-3-270m-it"


# ---------------------------------------------------------------- memory / fit
def test_weights_math():
    # llama-3.1-8b in fp16: 8.03e9 * 2 bytes = ~14.96 GB
    g = memory.weights_bytes(MODELS["llama-3.1-8b"].n_params, "fp16") / GB
    assert 14.5 < g < 15.5, g
    # int8 is exactly half of fp16
    assert abs(memory.weights_bytes(1e9, "int8") * 2
               - memory.weights_bytes(1e9, "fp16")) < 1


def test_kv_formula_matches_hand_count():
    s = MODELS["gemma-3-270m"]  # 18 layers, 1 kv-head (MQA), head_dim 256
    # fp16: 2(K,V) * 18 * 1 * 256 * 2 bytes = 18432 bytes/token
    per_tok = memory.kv_cache_bytes(s, seq_len=1, batch=1, kv_dtype="fp16")
    assert per_tok == 2 * 18 * 1 * 256 * 2, per_tok


def test_gqa_shrinks_kv():
    # llama-70b has 64 heads but 8 kv-heads -> KV is 8x smaller than if MHA
    s = MODELS["llama-3.1-70b"]
    kv = memory.kv_cache_bytes(s, 4096, 1, "fp16")
    mha = 4096 * 2 * s.n_layers * s.n_heads * s.head_dim * 2
    assert abs(mha / kv - s.n_heads / s.n_kv_heads) < 1e-6


def test_fit_verdict_and_suggestion():
    # 70b fp16 weights (~141 GB) cannot fit a 24 GB card
    r = memory.fit("llama-3.1-70b", gpu_gb=24, seq_len=4096)
    assert not r.fits and "DOES NOT FIT" in r.report()
    # 8b fp16 fits a 24 GB card with room
    r2 = memory.fit("llama-3.1-8b", gpu_gb=24, seq_len=4096)
    assert r2.fits and r2.headroom_gb > 0


def test_kv_formula_vs_real_measured_cache():
    """Cross-check the analytic KV formula against kvquant.cache_bytes on a
    REAL gemma cache. This is the honesty anchor for the whole memory model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kvquant import cache_bytes

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float16).to(dev).eval()
    ids = tok("Measure the key value cache size precisely please.",
              return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        out = model(ids, use_cache=True)
    cache = out.past_key_values
    seq = cache.layers[0].keys.shape[-2]           # real tokens held
    measured = cache_bytes(cache)                  # real tensor bytes (fp16)
    analytic = memory.kv_cache_bytes(MODELS["gemma-3-270m"], seq_len=seq,
                                     batch=1, kv_dtype="fp16")
    # allow a few % for per-token scale / layout overhead kvquant may count
    rel = abs(measured - analytic) / measured
    assert rel < 0.05, f"analytic {analytic} vs measured {measured} ({rel:.1%})"
    print(f"\n  KV formula check: analytic {analytic/1024:.1f}KB vs measured "
          f"{measured/1024:.1f}KB at seq={seq} ({rel:.2%} off)")


# -------------------------------------------------------------- tokenwaste
def test_tokenwaste_finds_planted_waste():
    sysp = "You are a helpful assistant. " * 50  # ~350 chars ~ 88 tok
    recs = [tokenwaste.Record(model="gpt-4o", system=sysp, prompt="hi",
                              input_tokens=100, output_tokens=5)
            for _ in range(10)]
    recs.append(tokenwaste.Record(model="gpt-4o", system=sysp, prompt="redo",
                                  input_tokens=100, output_tokens=5, retry=True))
    rep = tokenwaste.analyze(recs)
    cats = {c for c, _, _ in rep.findings}
    assert "no-prefix-cache" in cats      # 11x identical system prompt
    assert "retries" in cats              # one retry
    assert "oversized-model" in cats      # gpt-4o with <=20 output tokens
    assert rep.recoverable_usd > 0


def test_tokenwaste_clean_log_has_no_findings():
    recs = [tokenwaste.Record(model="gpt-4o-mini", prompt=f"unique {i}",
                              input_tokens=50, output_tokens=200)
            for i in range(5)]
    assert tokenwaste.analyze(recs).findings == []


# -------------------------------------------------------------- evalsanity
def test_evalsanity_catches_contamination():
    train = ["the capital of france is paris", "water boils at 100 c"]
    test = ["the capital of france is paris", "a brand new unseen question here"]
    rep = evalsanity.check(test, train_texts=train)
    assert not rep.clean  # exact overlap -> HIGH
    assert any("verbatim" in m.lower() or "contamination" in m.lower()
               for _, m in rep.flags)


def test_evalsanity_catches_degenerate_predictions():
    test = [f"q{i}" for i in range(50)]
    preds = ["yes"] * 50
    rep = evalsanity.check(test, predictions=preds)
    assert not rep.clean
    assert any("same value" in m or "collapsed" in m for _, m in rep.flags)


def test_evalsanity_catches_label_imbalance():
    test = [f"q{i}" for i in range(100)]
    labels = ["A"] * 97 + ["B"] * 3
    rep = evalsanity.check(test, labels=labels)
    assert any("one class" in m for _, m in rep.flags)


def test_evalsanity_clean_passes():
    test = [f"distinct question number {i}" for i in range(60)]
    labels = (["A", "B"] * 30)
    preds = (["A", "B", "A"] * 20)
    rep = evalsanity.check(test, labels=labels, predictions=preds)
    assert rep.clean


# -------------------------------------------------------------- quantcheck
def _symmetric_int8(x):
    import torch
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    q = torch.clamp(torch.round(x / scale), -127, 127)
    return q * scale


def test_quantcheck_real_int8_bites():
    import torch
    x = torch.randn(64, 256)
    r = quantcheck.check(_symmetric_int8, x, claimed_bits=8)
    assert r.bit and r.idempotent
    assert r.rel_error > 0 and r.rel_error < 0.1
    assert not r.suspicious, r.notes


def test_quantcheck_flags_silent_noop():
    import torch
    x = torch.randn(64, 256)
    r = quantcheck.check(lambda t: t.clone(), x, claimed_bits=8)  # identity
    assert not r.bit and r.suspicious
    assert any("no-op" in n for n in r.notes)


def test_quantcheck_flags_non_idempotent():
    import torch
    # adds fresh noise every call -> not idempotent (double-quant-style bug)
    r = quantcheck.check(lambda t: t + 0.01 * torch.randn_like(t),
                         torch.randn(64, 256))
    assert r.bit and not r.idempotent and r.suspicious


# -------------------------------------------------------------- cheapest
def test_cheapest_picks_smallest_passing():
    # quality rises with cost; bar=0.80 -> cheapest passing is the 3rd
    cands = [cheapest.Candidate(f"m{c}", cost=c, quality_fn=(lambda c=c: 0.5 + 0.1 * c))
             for c in (1, 2, 3, 4)]
    res = cheapest.sweep(cands, bar=0.80)
    assert res.pick is not None and res.pick.name == "m3"  # 0.5+0.3=0.80


def test_cheapest_none_passes():
    cands = [cheapest.Candidate("a", 1, lambda: 0.1)]
    assert cheapest.sweep(cands, bar=0.9).pick is None


# -------------------------------------------------------------- modelcard
def test_modelcard_arithmetic_only():
    card = modelcard.build("gemma-3-270m", seq_len=4096)
    assert len(card.rows) == 3
    fp16, int8, int4 = card.rows
    assert abs(fp16.kv_gb / int8.kv_gb - 2.0) < 1e-6   # int8 ~ 2x smaller
    assert abs(fp16.kv_gb / int4.kv_gb - 4.0) < 1e-6
    assert all(r.ppl_delta_pct is None for r in card.rows)  # not measured


def test_modelcard_live_measures_real_ppl():
    """Short REAL measurement via kvquant on gemma. Honest numbers only."""
    measured = modelcard.measure_kv_quality(REPO, max_tokens=120)
    assert "int8" in measured
    # int8 KV on gemma is near-lossless; int4 noticeably worse. Measured, not asserted tight.
    assert abs(measured["int8"]) < 5.0, measured
    card = modelcard.build("gemma-3-270m", seq_len=2048, measured_ppl=measured,
                           measured_on=REPO)
    assert card.rows[1].ppl_delta_pct is not None
    print(f"\n  measured gemma KV-quant: {measured}")


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
