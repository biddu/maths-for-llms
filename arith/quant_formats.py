"""Bits per weight, for every format Chapter 13 names.

    python arith/quant_formats.py            the M-13.1 table
    python arith/quant_formats.py --nf4      NF4's levels, and what they cost

Nobody ships 4.000 bits per weight.  A 4-bit format is named for the width of
its *elements*, and the metadata that makes those elements mean anything -- a
scale per group, sometimes a zero-point, sometimes a second-level scale over
the scales -- is real memory.  The spread across current 4-bit formats is 9%,
which on an 8 B model is the difference between fitting a 24 GB card and not.

The one formula the whole chapter turns on:

    bits per weight = b_q + (scale bits + zero-point bits) / g_q

with g_q the group size.  Everything below is that formula at named settings.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Format:
    name: str
    b_q: int                  # element width
    scale_bits: float         # per group
    zero_bits: float          # per group, 0 if the format is symmetric
    g_q: int                  # group size
    second_bits: float = 0.0  # bits per weight for a scale over the scales
    note: str = ""
    named: bool = False       # a format someone ships, as against a setting

    @property
    def bits(self) -> float:
        return (self.b_q + (self.scale_bits + self.zero_bits) / self.g_q
                + self.second_bits)

    def bytes_for(self, n_params: int) -> float:
        return n_params * self.bits / 8


# Double quantisation: NF4 stores an fp8 scale per block of 64, then one fp32
# scale per 256 of those blocks.  The second level is what "double" means and
# it is the difference between 4.5 and 4.127 bits.
FORMATS = [
    Format("int4, fp16 scale per 64", 4, 16, 0, 64),
    Format("int4, fp16 scale per 128", 4, 16, 0, 128, named=True),
    Format("int4, fp16 scale per 256", 4, 16, 0, 256),
    Format("int4, fp16 scale and zero per 128", 4, 16, 16, 128, named=True,
           note="asymmetric: a zero-point costs as much as the scale"),
    Format("MXFP4 (E8M0 scale per 32)", 4, 8, 0, 32, named=True,
           note="power-of-two scale, no zero-point: block floating point"),
    Format("NVFP4 (FP8 scale per 16)", 4, 8, 0, 16, named=True,
           note="smaller blocks buy accuracy and cost metadata"),
    Format("NF4, single quantisation", 4, 32, 0, 64,
           note="an fp32 scale per block of 64"),
    Format("NF4, double quantisation", 4, 8, 0, 64, 32.0 / (256 * 64), named=True,
           note="fp8 scale per 64, fp32 scale per 256 blocks"),
]


def double_quantisation_saving(g_q: int = 64, blocks: int = 256) -> float:
    """E-13.9.  An fp32 scale per block costs 32/g_q bits per weight; replacing
    it with an fp8 scale plus one fp32 scale per `blocks` blocks costs
    8/g_q + 32/(blocks*g_q).  The difference is the saving."""
    return 32.0 / g_q - (8.0 / g_q + 32.0 / (blocks * g_q))


def nf4_levels():
    """The sixteen NF4 values: quantiles of the standard normal, normalised so
    the outermost level is 1, with an exact zero.

    NF4 and Lloyd-Max are both called optimal and they are optimising different
    things.  Lloyd-Max minimises squared error for a given distribution.  NF4
    puts equal probability mass in every bin, which is optimal for *entropy*
    and, unlike Lloyd-Max, gives a fixed table that no kernel has to fit or
    look up per block.  Chapter 13 measures both.
    """
    from scipy.stats import norm
    import numpy as np
    offset = 0.5 * (1 / 32 + 1 / 30)
    neg = norm.ppf(np.linspace(offset, 0.5, 9))[:-1]
    pos = norm.ppf(np.linspace(0.5, 1 - offset, 8))
    lv = np.concatenate([neg, pos])
    return np.sort(lv / abs(lv).max())


def snr_db(b_q: int, loading: float) -> float:
    """D-13.1 step 8.  `loading` is R/s_x, the full-scale range in signal
    standard deviations.  The 6.02 is the derivative and everything else is
    the loading factor, which is where all the trouble lives."""
    import math
    return 6.02 * b_q + 10 * math.log10(12.0 / loading ** 2)


def report(which: str = "table") -> None:
    try:
        from arith.model_d import MODEL_D, total_params
    except ImportError:
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from arith.model_d import MODEL_D, total_params
    n = total_params(MODEL_D)
    if which == "nf4":
        lv = nf4_levels()
        print("NF4's sixteen levels:")
        print("  " + "  ".join(f"{v:+.4f}" for v in lv))
        print(f"  exact zero present: {any(v == 0.0 for v in lv)}"
              f"   outermost {lv.min():.1f} and {lv.max():.1f}")
        print(f"\ndouble quantisation saves {double_quantisation_saving():.5f} bits/param")
        print(f"  on Model D that is {n * double_quantisation_saving() / 8 / 1e6:.0f} MB")
        return
    print(f"{'format':<38}{'bits/weight':>13}{'Model D':>11}{'':>3}note")
    for f in FORMATS:
        print(f"{f.name:<38}{f.bits:>13.4f}{f.bytes_for(n)/1e9:>10.3f}G   {f.note}")
    b = [f.bits for f in FORMATS if f.named]
    print(f"\n  spread across the shipped 4-bit formats: {100*(max(b)/min(b)-1):.1f}%"
          f"  = {n*(max(b)-min(b))/8/1e9:.3f} GB on Model D")
    print("  which is the difference between fitting a 24 GB card and not")
    print(f"\n  and what a bit buys: SNR at a +/-4 s_x clip")
    for bq in (2, 3, 4, 8):
        print(f"    {bq} bits: {snr_db(bq, 8.0):6.2f} dB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nf4", action="store_true")
    a = ap.parse_args()
    report("nf4" if a.nf4 else "table")


if __name__ == "__main__":
    main()
