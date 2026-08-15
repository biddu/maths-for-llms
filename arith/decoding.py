"""Chapter 14's arithmetic: what decoding costs and what speculation buys.

    python arith/decoding.py                  the A-14.1 table
    python arith/decoding.py --gamma          speedup against gamma, F-14.4's data
    python arith/decoding.py --counter        the acceptance-counter trap
    python arith/decoding.py --bandwidth      bytes per emitted token

Three formulas and one warning.

    E[tokens/round]  = Sum_{k=0..g} a^k = (1 - a^(g+1))/(1 - a)          (14.16)
    S(a, g, c)       = E[tokens/round] / (1 + g c)                       (14.17)
    net win          <==>  (1/g) Sum_{k=1..g} a^k  >  c                  (14.18)

The warning is that `a` is not the quantity a naive serving counter reports.
See `acceptance_counter`.

The draft cost ratio c is DERIVED here rather than asserted: at batch one, decode
is bandwidth-bound (Chapter 11), so the cost of a forward pass is the cost of
reading the weights, and c is a ratio of parameter counts.  A-14.1's c = 1/8 is
an 8 B target with a 1 B draft, and this file says so in one line rather than
leaving the reader to guess where 1/8 came from.
"""
from __future__ import annotations

import argparse


# --------------------------------------------------------------- the formulas
def tokens_per_round(a: float, g: int) -> float:
    """(14.16).  The tail-sum, not the pmf: P(N >= k) = a^k for k <= g, so
    E[N] = Sum_{k=1..g} a^k, and one further token is emitted in EITHER branch
    (the residual draw on rejection, the bonus token on all-accept), which is
    what makes the sum run from zero."""
    if a >= 1.0:
        return float(g + 1)
    return (1.0 - a ** (g + 1)) / (1.0 - a)


def speedup(a: float, g: int, c: float) -> float:
    """(14.17).  A round costs one target pass plus g draft passes, measured in
    target-equivalents, so the denominator is 1 + gc.  The numerator saturates
    at 1/(1-a) and the denominator grows without bound: S peaks and falls."""
    return tokens_per_round(a, g) / (1.0 + g * c)


def break_even_c(a: float, g: int) -> float:
    """(14.18).  The largest draft cost ratio that still wins, which is the mean
    of a, a^2, ..., a^g.  At g = 1 it is just a."""
    return sum(a ** k for k in range(1, g + 1)) / g


def best_gamma(a: float, c: float, gmax: int = 64) -> tuple[int, float]:
    """No closed form; the blueprint says so and it is true.  Tabulate."""
    return max(((g, speedup(a, g, c)) for g in range(1, gmax + 1)),
               key=lambda t: t[1])


def draft_cost_ratio(n_draft: int, n_target: int) -> float:
    """c, derived rather than asserted.  At batch one a forward pass costs what
    it costs to stream the weights (Chapter 11's roofline), so the ratio of
    costs is the ratio of parameter counts.  This is the step that makes
    A-14.1's c = 1/8 an inference rather than a folk constant, and it is also
    where the argument would change on a machine that was compute-bound at
    batch one, which none of them are."""
    return n_draft / n_target


# ------------------------------------------------------- the counter warning
def acceptance_counter(a: float, g: int) -> dict[str, float]:
    """The trap in D-14.4's "make alpha_acc a counter in the serving loop".

    A round drafts g tokens but TESTS only the positions up to and including the
    first rejection.  Dividing accepted tokens by *drafted* tokens therefore
    divides by the wrong denominator:

        E[accepted] = Sum_{k=1..g} a^k          E[tested] = Sum_{k=0..g-1} a^k

    so accepted/tested = a exactly, while accepted/drafted = (1/g) Sum a^k --
    which is (14.18)'s left-hand side.  The naive counter does not report the
    acceptance rate at all.  It reports the break-even draft cost ratio, and it
    is smaller than a for every a < 1 and g > 1.  A deployment that reads 0.59
    off the wrong counter and concludes its draft is poor is looking at a = 0.8.
    """
    acc = sum(a ** k for k in range(1, g + 1))
    tested = sum(a ** k for k in range(0, g))
    return {"alpha": a,
            "correct_counter": acc / tested,
            "naive_counter": acc / g,
            "understatement": 1.0 - (acc / g) / a}


def plugin_bias(a_bar: float, var: float, g: int) -> dict[str, float]:
    """(14.16) is evaluated at a single a_bar, and the acceptance rate is not
    constant.  Two corrections of OPPOSITE sign, both second order:

      within a round, heterogeneity across the g positions replaces a product
      of unequal numbers by a power of their mean, and by AM-GM that OVERSTATES
      the true expectation;

      across rounds, f(a) = Sum_{k=0..g} a^k is convex, so by Jensen the true
      expectation is UNDERSTATED, by f''(a_bar) var / 2.

    Which dominates is a property of the workload, which is the honest reason
    alpha has to be measured in the loop rather than quoted from a paper.
    """
    fpp = sum(k * (k - 1) * a_bar ** (k - 2) for k in range(2, g + 1))
    base = tokens_per_round(a_bar, g)
    return {"plug_in": base,
            "jensen_correction": 0.5 * fpp * var,
            "corrected": base + 0.5 * fpp * var}


# ------------------------------------------------------------- the bandwidth
def bytes_per_emitted_token(weight_bytes: float, a: float, g: int) -> float:
    """At batch one the weights are re-read every round no matter how many
    tokens the round emits, so speculation divides a fixed cost over more
    output.  This is the whole mechanism, in one line."""
    return weight_bytes / tokens_per_round(a, g)


def verify_intensity_ratio(g: int) -> float:
    """Verifying g+1 positions in one pass multiplies the arithmetic by g+1 and
    leaves the weight traffic alone, so the arithmetic intensity of the target
    pass rises by exactly g+1 over single-token decode.  That is why the
    measured wall-clock speedup EXCEEDS the token-count speedup: the extra
    positions are free on a bandwidth-bound machine."""
    return float(g + 1)


# ------------------------------------------------------------------ reporting
def _model_d_params() -> int:
    try:
        from arith.model_d import MODEL_D, total_params
    except ImportError:
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from arith.model_d import MODEL_D, total_params
    return total_params(MODEL_D)


def report(which: str = "table") -> None:
    n = _model_d_params()
    c = 1 / 8
    if which == "gamma":
        print(f"{'gamma':>6}" + "".join(f"{a:>10}" for a in
                                        (0.5, 0.6, 0.7, 0.8, 0.9)))
        for g in range(1, 17):
            print(f"{g:>6}" + "".join(f"{speedup(a, g, c):>10.4f}"
                                      for a in (0.5, 0.6, 0.7, 0.8, 0.9)))
        print("\n  argmax gamma and the width of the near-optimal plateau:")
        for a in (0.5, 0.6, 0.7, 0.8, 0.9):
            g, s = best_gamma(a, c)
            band = [k for k in range(1, 33) if speedup(a, k, c) >= 0.99 * s]
            print(f"    a={a}: gamma* = {g:>2} ({s:.3f}x), within 1% for "
                  f"gamma in {min(band)}..{max(band)}")
        return
    if which == "counter":
        print("  What a serving counter reports, against the true alpha:")
        print(f"    {'alpha':>7}{'gamma':>7}{'accepted/tested':>18}"
              f"{'accepted/drafted':>19}{'understated by':>16}")
        for a in (0.6, 0.7, 0.8, 0.9):
            for g in (2, 4, 8):
                d = acceptance_counter(a, g)
                print(f"    {a:>7}{g:>7}{d['correct_counter']:>18.4f}"
                      f"{d['naive_counter']:>19.4f}"
                      f"{100*d['understatement']:>15.1f}%")
        print("\n  and the naive counter's value is exactly (14.18)'s break-even c:")
        for a, g in ((0.8, 4), (0.65, 4)):
            print(f"    a={a}, gamma={g}: naive counter "
                  f"{acceptance_counter(a, g)['naive_counter']:.4f}, "
                  f"break-even c {break_even_c(a, g):.4f}")
        return
    if which == "bandwidth":
        wb = n * 2                      # fp16/bf16 weights, one full pass
        print(f"  Model D: {n:,} parameters, {wb/1e9:.3f} GB of bf16 weights")
        print(f"  {'setting':>28}{'tokens/round':>14}{'GB per token':>14}")
        print(f"  {'plain decode':>28}{1.0:>14.4f}{wb/1e9:>14.3f}")
        for a, g in ((0.8, 4), (0.9, 4), (0.8, 8)):
            t = tokens_per_round(a, g)
            print(f"  {f'speculative a={a}, g={g}':>28}{t:>14.4f}"
                  f"{bytes_per_emitted_token(wb, a, g)/1e9:>14.3f}")
        print(f"\n  arithmetic intensity of a verify pass over a decode step: "
              f"{verify_intensity_ratio(4):.1f}x at gamma = 4")
        print("  the verify positions are free on a bandwidth-bound machine,")
        print("  which is why measured wall-clock beats this token-count model")
        return

    print(f"A-14.1  target Model D ({n/1e9:.3f} B), draft 1 B same family")
    print(f"  c = draft/target cost = {draft_cost_ratio(1_000_000_000, n):.4f}"
          f"  (used as 1/8 = {c})")
    print(f"  gamma = 4, alpha_acc = 0.8")
    print(f"    tokens/round = (1 - 0.8^5)/(1 - 0.8) = "
          f"{tokens_per_round(0.8, 4):.4f}")
    print(f"    cost/round   = 1 + 4/8 = {1 + 4 * c}")
    print(f"    speedup      = {speedup(0.8, 4, c):.4f}  -> "
          f"{speedup(0.8, 4, c):.2f}x")
    print("\n  sensitivity in alpha_acc at gamma = 4:")
    for a in (0.9, 0.8, 0.6):
        print(f"    a={a}: {tokens_per_round(a, 4):.4f} tokens, "
              f"{speedup(a, 4, c):.4f} -> {speedup(a, 4, c):.2f}x")
    print("\n  sensitivity in gamma at alpha_acc = 0.8:")
    for g in (2, 4, 8):
        print(f"    g={g}: {tokens_per_round(0.8, g):.4f}/{1 + g * c:.2f} = "
              f"{speedup(0.8, g, c):.4f} -> {speedup(0.8, g, c):.2f}x")
    g, s = best_gamma(0.8, c)
    print(f"\n  alpha_acc, not gamma, is the lever: 0.8 -> 0.6 costs "
          f"{speedup(0.8, 4, c) - speedup(0.6, 4, c):.2f}x,")
    print(f"  while doubling gamma from 4 to 8 costs "
          f"{speedup(0.8, 4, c) - speedup(0.8, 8, c):.2f}x.  The numerator")
    print(f"  saturates at 1/(1-a) = {1 / (1 - 0.8):.1f}; the optimum is "
          f"gamma = {g} at {s:.2f}x.")
    print(f"\n  an MTP head (DC-14.2) at c = 0.05: "
          f"{speedup(0.8, 4, 0.05):.4f} -> {speedup(0.8, 4, 0.05):.2f}x")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gamma", action="store_true")
    ap.add_argument("--counter", action="store_true")
    ap.add_argument("--bandwidth", action="store_true")
    a = ap.parse_args()
    report("gamma" if a.gamma else "counter" if a.counter else
           "bandwidth" if a.bandwidth else "table")


if __name__ == "__main__":
    main()
