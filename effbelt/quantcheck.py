"""`quantcheck` — did your quantization actually do anything?

The kvquant lesson, generalized to ANY tensor transform you believe quantizes:
a quant function that returns the input unchanged (a silent no-op, e.g. it
mutated a non-persisting view) passes every smoke test and moves no metric.
quantcheck proves the transform BIT, estimates its effective bit-width, and
flags the two classic bugs: silent no-op and non-idempotent double-quant.

`fn` maps a float tensor -> a (dequantized) float tensor of the same shape
(fake-quant). Needs torch.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class QuantCheckResult:
    bit: bool                 # did it change the tensor at all?
    mean_abs_change: float
    rel_error: float          # ||out-in|| / ||in||
    effective_bits: float     # log2(distinct output levels), capped
    idempotent: bool          # fn(fn(x)) ~= fn(x)
    claimed_bits: int | None
    notes: list[str]

    @property
    def ok(self) -> bool:
        return self.bit and self.idempotent and not self.suspicious

    @property
    def suspicious(self) -> bool:
        return bool(self.notes)

    def report(self) -> str:
        v = "OK" if self.ok else ("SUSPICIOUS" if self.bit else "SILENT NO-OP")
        lines = [
            f"quantcheck: {v}",
            f"  changed tensor:   {self.bit}  (mean |delta| = {self.mean_abs_change:.3e})",
            f"  relative error:   {self.rel_error*100:.3f} %",
            f"  effective bits:   ~{self.effective_bits:.1f}"
            + (f"  (claimed {self.claimed_bits})" if self.claimed_bits else ""),
            f"  idempotent:       {self.idempotent}",
        ]
        for nmsg in self.notes:
            lines.append(f"  ! {nmsg}")
        return "\n".join(lines)


def check(fn, x, claimed_bits: int | None = None,
          idempotent_tol: float = 1e-6) -> QuantCheckResult:
    import torch
    x = x.detach().float()
    out = fn(x).detach().float()
    if out.shape != x.shape:
        raise ValueError(f"fn changed shape {tuple(x.shape)} -> {tuple(out.shape)}; "
                         "quantcheck expects a same-shape fake-quant")
    delta = (out - x).abs()
    mac = float(delta.mean())
    xn = float(x.norm())
    rel = float((out - x).norm() / xn) if xn > 0 else 0.0
    bit = mac > 0.0

    # Effective bits from distinct LEVELS *per group* (last axis). Per-channel
    # quantizers give each row its own scale, so counting distinct values over
    # the whole tensor wildly overestimates the bit-width; count within a row.
    # Note: with group_size <= 2**bits this is a rough lower bound (you cannot
    # tell 8-bit from fp with only 256 samples) -- treat it as advisory.
    rows = out.reshape(-1, out.shape[-1])
    nrows = rows.shape[0]
    idx = torch.randperm(nrows)[:256] if nrows > 256 else torch.arange(nrows)
    levels = max((int(torch.unique(rows[i]).numel()) for i in idx.tolist()),
                 default=1)
    eff = min(32.0, math.log2(levels)) if levels > 1 else 0.0

    # idempotency: fake-quant of an already-quantized value should be a no-op
    out2 = fn(out).detach().float()
    on = float(out.norm())
    idem_err = float((out2 - out).norm() / on) if on > 0 else 0.0
    idempotent = idem_err <= max(idempotent_tol, 0.0)

    notes: list[str] = []
    if not bit:
        notes.append("transform did NOT change the tensor -> silent no-op "
                     "(did you mutate a non-persisting view / wrong attribute?)")
    if bit and not idempotent:
        notes.append(f"NOT idempotent (fn(fn(x)) differs by {idem_err*100:.2f}%)"
                     " -> re-quantizing keeps shifting values; error will "
                     "accumulate (double-quantization bug?)")
    # Only the "coarser than claimed" direction is reliable at small groups:
    # too FEW levels means real over-quantization. The other direction (more
    # levels than claimed) cannot be distinguished from fp when the group has
    # <= 2**bits elements, so we do not flag it.
    if claimed_bits is not None and levels > 1 and out.shape[-1] > 2 ** claimed_bits:
        if eff < max(1.0, claimed_bits - 2.0):
            notes.append(f"only ~{eff:.1f} effective bits vs claimed "
                         f"{claimed_bits} -> coarser than intended")
    return QuantCheckResult(bit, mac, rel, eff, idempotent, claimed_bits, notes)
