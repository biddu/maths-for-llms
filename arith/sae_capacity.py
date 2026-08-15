"""How many directions a residual stream holds, and what a dictionary costs.

    python arith/sae_capacity.py            the A-16.1 table
    python arith/sae_capacity.py --bound    capacity against coherence
    python arith/sae_capacity.py --l0       the sustainable L0 from (16.14)

Three formulas, and the third is the only one that predicts anything.

    m <= exp(d eps^2 / 4)                 almost-orthogonal capacity   (16.9)
    k  < (1/2)(1 + 1/(eps kappa))         worst-case interference     (16.12)
    k <= d / (4 ln m)                     the sustainable L0          (16.14)

The first is an existence result about RANDOM configurations and is quadratically
sensitive to a threshold nobody can justify to three significant figures: at
d = 4096 the permitted count moves by a factor of e for a change of 0.005 in eps.
Quote it with that caveat attached or not at all.  The third is a prediction, it
was derived rather than tuned, and it matches published practice.
"""
from __future__ import annotations

import argparse
import math

GiB = 1 << 30


def _model_d():
    try:
        from arith.model_d import MODEL_D
    except ImportError:
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from arith.model_d import MODEL_D
    return MODEL_D


# ------------------------------------------------------------------ capacity
def capacity(d: int, eps: float) -> float:
    """(16.9).  exp(d eps^2 / 4).  Existence, for a random configuration."""
    return math.exp(d * eps ** 2 / 4)


def eps_for(d: int, m: int) -> float:
    """Invert (16.9): the coherence at which a width-m dictionary is admitted."""
    return math.sqrt(4 * math.log(m) / d)


def worst_case_k(eps: float, kappa: float = 1.0) -> float:
    """(16.12).  Note what is absent: neither d nor m appears.  A threshold
    detector on a 4096-wide stream at eps = 0.1 supports about five live
    features, which is why the random-sign refinement matters so much."""
    return 0.5 * (1.0 + 1.0 / (eps * kappa))


def random_sign_k(eps: float, kappa: float = 1.0) -> float:
    """(16.13).  Independent interference signs buy a SQUARE: k ~ eps^-2."""
    return 1.0 / (eps * kappa) ** 2


def sustainable_l0(d: int, m: int) -> float:
    """(16.14).  k <= d / (4 ln m).

    Capacity is exponential in d/k, the width PER ACTIVE FEATURE, not in d.
    Doubling the live-feature count halves the exponent.  That is the actual
    content of "superposition works because features are sparse"."""
    return d / (4.0 * math.log(m))


# -------------------------------------------------------------- the dictionary
def sae_params(d: int, expansion: int) -> dict[str, int]:
    m = d * expansion
    enc, dec = d * m, m * d
    return {"m": m, "W_enc": enc, "W_dec": dec, "b_enc": m, "b_dec": d,
            "total": enc + dec + m + d}


def layer_params(c=None) -> int:
    """One Model D layer, recomputed rather than quoted: grouped-query attention
    plus a SwiGLU block."""
    c = c or _model_d()
    att = 2 * c.d * c.d + 2 * (c.d * c.n_kv * c.d_h)
    mlp = 3 * c.d * c.d_ff
    return att + mlp


# ------------------------------------------------------------------ shrinkage
def soft_threshold_deficit(lam: float, cbar: float = 1.0) -> float:
    """(16.18).  The reconstruction-norm ratio under L1, which is
    1 - lam/(2 cbar) and is INDEPENDENT of how many atoms are active.  That
    independence is what makes the bias a property of lambda alone and not
    something a sparsity sweep can tune away."""
    return 1.0 - lam / (2.0 * cbar)


def report(which: str = "table") -> None:
    c = _model_d()
    d = c.d
    if which == "bound":
        print(f"Capacity of a {d}-wide residual stream, (16.9)\n")
        print(f"  {'eps':>8}{'exp(d eps^2/4)':>18}{'against d':>14}")
        for eps in (0.05, 0.08, 0.10, 0.107, 0.15, 0.20):
            cap = capacity(d, eps)
            print(f"  {eps:>8}{cap:>18.3e}{cap/d:>13.3g}x")
        print(f"\n  exact orthogonality gives m = d = {d:,}")
        print(f"  the bound is VACUOUS below eps = "
              f"{math.sqrt(4*math.log(d)/d):.4f}, where it permits fewer than d")
        print(f"\n  quadratic sensitivity: ln m moves by 1 for a change of "
              f"{2/(d*0.1):.5f} in eps at eps = 0.1")
        print("  so 'inside the bound' is decided by the third significant")
        print("  figure of a threshold nobody can justify.")
        return
    if which == "l0":
        print(f"(16.14): the sustainable L0 at d = {d}\n")
        print(f"  {'m':>10}{'expansion':>12}{'k <= d/(4 ln m)':>18}")
        for exp_ in (8, 16, 32, 64, 128):
            m = d * exp_
            print(f"  {m:>10,}{exp_:>12}{sustainable_l0(d, m):>18.1f}")
        print(f"\n  worst-case (16.12) at eps = 0.1, kappa = 1: k < "
              f"{worst_case_k(0.1):.1f}")
        print(f"  random signs (16.13) at the same: k ~ {random_sign_k(0.1):.0f}")
        print("  the square is the whole reason superposition is usable.")
        return

    exp_ = 32
    p = sae_params(d, exp_)
    lay = layer_params(c)
    wb = 2 * (p["W_enc"] + p["W_dec"])
    print(f"A-16.1  a Model D sparse autoencoder at expansion {exp_}\n")
    print(f"  d = {d:,}, m = {p['m']:,}")
    print(f"  W_enc {p['W_enc']:>14,}    W_dec {p['W_dec']:>14,}")
    print(f"  b_enc {p['b_enc']:>14,}    b_dec {p['b_dec']:>14,}")
    print(f"  total {p['total']:>14,}  = {p['total']/1e9:.3f} B")
    print(f"\n  bf16 weights   {wb:>14,} B = {wb/GiB:.3f} GiB (exactly two)")
    print(f"  bf16 biases    {2*(p['b_enc']+p['b_dec']):>14,} B")
    print(f"  AdamW state at 16 B/param        {16*p['total']/1e9:>8.1f} GB")
    print(f"\n  one Model D layer                {lay/1e6:>8.1f} M parameters")
    print(f"  the SAE is                       {p['total']/lay:>8.2f}x the layer"
          f" it explains")
    print(f"  one per layer, {c.L} layers       {c.L*p['total']/1e9:>8.1f} B"
          f"  = {c.L*wb/GiB:.0f} GiB of bf16")
    print(f"\n  against the bound at eps = 0.1:   {capacity(d, 0.1):>12,.0f}"
          f"   so m is {p['m']/capacity(d, 0.1):.2f}x OUTSIDE it")
    print(f"  m is admitted at eps =            {eps_for(d, p['m']):>12.6f}")
    print(f"  and (16.14) gives a sustainable L0 of "
          f"{sustainable_l0(d, p['m']):.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bound", action="store_true")
    ap.add_argument("--l0", action="store_true")
    a = ap.parse_args()
    report("bound" if a.bound else "l0" if a.l0 else "table")


if __name__ == "__main__":
    main()
