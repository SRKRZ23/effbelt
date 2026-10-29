"""effbelt command-line interface: `effbelt <tool> ...`."""
from __future__ import annotations
import argparse
import sys


def _cmd_fit(a):
    from .memory import fit, shape_from_hf
    from .models import GPUS
    gpu = a.gpu
    if isinstance(gpu, str):
        if gpu.lower() not in GPUS:
            print(f"unknown gpu {gpu!r}; known: {sorted(GPUS)}", file=sys.stderr)
            return 2
        gpu = GPUS[gpu.lower()]
    shape = shape_from_hf(a.hf) if a.hf else a.model
    r = fit(shape, gpu_gb=float(gpu), seq_len=a.seq, batch=a.batch,
            weight_dtype=a.wdtype, kv_dtype=a.kv, overhead_gb=a.overhead)
    print(r.report())
    return 0


def _cmd_tokenwaste(a):
    from .tokenwaste import analyze, load_jsonl
    recs = load_jsonl(a.log)
    print(analyze(recs, long_context_tokens=a.long_context).report())
    return 0


def _cmd_evalsanity(a):
    from .evalsanity import check
    import json
    with open(a.data, encoding="utf-8") as f:
        d = json.load(f)
    rep = check(d["test"], d.get("train"), d.get("labels"), d.get("predictions"))
    print(rep.report())
    return 0 if rep.clean else 1


def _cmd_modelcard(a):
    from .modelcard import build, measure_kv_quality
    measured, on = None, None
    if a.live:
        measured = measure_kv_quality(a.live, max_tokens=a.max_tokens)
        on = f"{a.live} ({a.max_tokens} tok, MPS/CPU)"
    print(build(a.model, seq_len=a.seq, batch=a.batch,
                weight_dtype=a.wdtype, measured_ppl=measured,
                measured_on=on).report())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="effbelt",
                                description="LLM efficiency toolbelt")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="will this model fit on my card?")
    f.add_argument("model", help="known model name, or use --hf")
    f.add_argument("--hf", help="authoritative dims from a HF repo id")
    f.add_argument("--gpu", default="rtx-4090", help="gpu name or GB number")
    f.add_argument("--seq", type=int, default=4096)
    f.add_argument("--batch", type=int, default=1)
    f.add_argument("--wdtype", default="fp16")
    f.add_argument("--kv", default="fp16")
    f.add_argument("--overhead", type=float, default=1.0)
    f.set_defaults(func=_cmd_fit)

    t = sub.add_parser("tokenwaste", help="audit an API request log (jsonl)")
    t.add_argument("log")
    t.add_argument("--long-context", type=int, default=8000)
    t.set_defaults(func=_cmd_tokenwaste)

    e = sub.add_parser("evalsanity", help="is your benchmark lying?")
    e.add_argument("data", help="json with keys: test[, train, labels, predictions]")
    e.set_defaults(func=_cmd_evalsanity)

    m = sub.add_parser("modelcard", help="quant tradeoff card")
    m.add_argument("model")
    m.add_argument("--seq", type=int, default=4096)
    m.add_argument("--batch", type=int, default=1)
    m.add_argument("--wdtype", default="fp16")
    m.add_argument("--live", help="HF repo id to MEASURE real ppl deltas")
    m.add_argument("--max-tokens", type=int, default=200)
    m.set_defaults(func=_cmd_modelcard)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        gpu = getattr(args, "gpu", None)
        if gpu is not None and gpu.replace(".", "", 1).isdigit():
            args.gpu = float(gpu)
    except Exception:
        pass
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
