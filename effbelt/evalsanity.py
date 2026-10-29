"""`evalsanity` — is your benchmark lying to you?

Cheap, deterministic checks that catch the usual ways an eval number is
inflated or meaningless BEFORE you trust it:
  - train/test contamination (exact + near-dup via shingled Jaccard)
  - degenerate data (duplicate rows, near-constant labels)
  - degenerate predictions (model emits one answer, or copies the input)
  - too-good-to-be-true accuracy on a tiny / imbalanced set
No model required. Works on lists of strings.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re


def _shingles(text: str, k: int = 5) -> set[str]:
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class SanityReport:
    n_test: int
    flags: list[tuple[str, str]] = field(default_factory=list)  # (severity, msg)

    @property
    def clean(self) -> bool:
        return not any(s == "HIGH" for s, _ in self.flags)

    def report(self) -> str:
        if not self.flags:
            return f"{self.n_test} test items: no sanity problems found."
        lines = [f"{self.n_test} test items: {len(self.flags)} flag(s)"]
        for sev, msg in sorted(self.flags, key=lambda x: {"HIGH": 0, "MED": 1,
                                                          "LOW": 2}[x[0]]):
            lines.append(f"  [{sev:4s}] {msg}")
        if not self.clean:
            lines.append("  -> at least one HIGH flag: do NOT trust this number yet.")
        return "\n".join(lines)


def check(test_texts: list[str],
          train_texts: list[str] | None = None,
          labels: list | None = None,
          predictions: list | None = None,
          near_dup_threshold: float = 0.8) -> SanityReport:
    flags: list[tuple[str, str]] = []
    n = len(test_texts)

    # ---- contamination ----
    if train_texts:
        train_set = set(t.strip() for t in train_texts)
        exact = sum(1 for t in test_texts if t.strip() in train_set)
        if exact:
            flags.append(("HIGH", f"{exact}/{n} test items appear VERBATIM in "
                                  f"train -> contamination"))
        # near-dup (sampled to stay cheap)
        tr_sh = [_shingles(t) for t in train_texts[:2000]]
        near = 0
        for t in test_texts[:2000]:
            sh = _shingles(t)
            if any(_jaccard(sh, trs) >= near_dup_threshold for trs in tr_sh):
                near += 1
        if near > exact:
            flags.append(("HIGH", f"{near}/{min(n,2000)} test items are NEAR "
                                  f"-duplicates of train (Jaccard>="
                                  f"{near_dup_threshold})"))

    # ---- degenerate test data ----
    uniq = len(set(test_texts))
    if n and uniq < n:
        flags.append(("MED", f"{n-uniq}/{n} duplicate test items"))
    if labels is not None:
        from collections import Counter
        c = Counter(labels)
        top, topn = c.most_common(1)[0]
        frac = topn / len(labels)
        if frac >= 0.95:
            flags.append(("HIGH", f"labels {frac*100:.0f}% one class ({top!r}) "
                                  f"-> accuracy is meaningless; use balanced metric"))
        elif frac >= 0.8:
            flags.append(("MED", f"labels {frac*100:.0f}% one class ({top!r}) "
                                 f"-> imbalanced"))

    # ---- degenerate predictions ----
    if predictions is not None:
        from collections import Counter
        pc = Counter(predictions)
        ptop, ptopn = pc.most_common(1)[0]
        if ptopn / len(predictions) >= 0.95 and len(pc) > 1:
            flags.append(("HIGH", f"predictions {100*ptopn/len(predictions):.0f}% "
                                  f"a single value ({ptop!r}) -> model collapsed"))
        elif len(pc) == 1:
            flags.append(("HIGH", f"predictions are ALL the same value ({ptop!r})"))
        # copying the input?
        if len(predictions) == len(test_texts):
            copied = sum(1 for p, t in zip(predictions, test_texts)
                         if isinstance(p, str) and p.strip() == t.strip())
            if copied / len(predictions) >= 0.3:
                flags.append(("MED", f"{copied}/{len(predictions)} predictions "
                                     f"equal the input text -> possible copying"))

    # ---- tiny set ----
    if 0 < n < 30:
        flags.append(("LOW", f"only {n} test items -> wide confidence interval; "
                             f"a few flips swing the score"))
    return SanityReport(n, flags)
