"""F-14.2 -- entropy against temperature, with the derivative as an inset.

The figure has one job beyond illustration: to make dH/dT = Var_p(T)(z)/T^3
visible as a *shape*, so that the reader who slips a power (the blueprint did,
and T^2 agrees with T^3 at exactly one point) has somewhere to look.  The lower
panel therefore plots the closed form over a central difference of the upper
panel's curve, not the closed form alone; the two are drawn as a heavy pale line
under a fine dark one, so agreement is visible rather than asserted.

The blueprint asked for the derivative as an inset.  It cannot be one: the
peaked curve crosses the only rectangle large enough to hold it, and an inset
sitting on top of a plotted line is exactly the kind of collision this book
does not ship.  A second panel on a shared axis is also the better reading,
because a function and its derivative want the same abscissa.

Three logit vectors, all V = 128, chosen to span the regimes a served model
actually visits: a near-deterministic position, an ordinary one, and a position
where the model has no idea.  The asymptote log V is ruled, and every curve
approaches it from below because H is monotone.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mfestyle as S

S.apply()

V = 128
rng = np.random.default_rng(1402)
base = np.sort(rng.standard_normal(V))[::-1]

# The three regimes, as multiples of one common shape, so the only difference
# between the curves is the spread of the logits.  That is the point: entropy at
# T = 1 is a statement about logit spread and nothing else.
CASES = [("peaked", 6.0), ("moderate", 2.0), ("near-flat", 0.5)]

Ts = np.geomspace(0.05, 40.0, 600)


def probs(z, T):
    w = z / T
    w = w - w.max()
    e = np.exp(w)
    return e / e.sum()


def entropy(z, T):
    p = probs(z, T)
    return float(-(p * np.log(np.clip(p, 1e-300, None))).sum())


def dHdT(z, T):
    p = probs(z, T)
    var = float((p * z ** 2).sum() - (p * z).sum() ** 2)
    return var / T ** 3


fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.95, 3.05), sharex=True,
                             gridspec_kw={"height_ratios": [1.55, 1.0]},
                             constrained_layout=True)

h = 1e-5
for (name, scale), col, ls in zip(CASES, (S.ACCENT, S.GREY, S.ACCENT_MID),
                                  S.DASHES[:3]):
    z = base * scale
    H = np.array([entropy(z, T) for T in Ts])
    ax.plot(Ts, H, ls=ls, color=col, lw=1.1, zorder=4)
    ax.text(0.052, H[0] + 0.13, name, fontsize=6.3, color=col, ha="left",
            va="bottom")
    # lower panel: the closed form, over a central difference of the curve above
    bx.plot(Ts, [(entropy(z, T + h) - entropy(z, T - h)) / (2 * h) for T in Ts],
            color=S.GREY_LIGHT, lw=2.4, alpha=0.85, zorder=3,
            solid_capstyle="round")
    bx.plot(Ts, [dHdT(z, T) for T in Ts], ls=ls, color=col, lw=0.9, zorder=4)

ax.axhline(np.log(V), color=S.GREY_LIGHT, lw=0.7, ls=(0, (2, 2)), zorder=2)
ax.text(34, np.log(V) - 0.30, r"$\log V$", fontsize=6.4, color=S.GREY,
        ha="right", va="top")
ax.set_ylabel(r"$H(p(T))$, nats")
ax.set_ylim(-0.25, np.log(V) + 0.55)

bx.set_xscale("log")
bx.set_yscale("log")
bx.set_xlabel(r"temperature $T$")
bx.set_ylabel(r"$\mathrm{d}H/\mathrm{d}T$")
bx.set_xlim(0.05, 40)
bx.set_ylim(1e-5, 3e2)
bx.xaxis.set_major_locator(FixedLocator([0.05, 0.2, 1, 5, 40]))
bx.set_xticklabels(["0.05", "0.2", "1", "5", "40"])
bx.xaxis.set_minor_formatter(NullFormatter())
bx.yaxis.set_minor_formatter(NullFormatter())
bx.text(0.985, 0.90, r"pale: central difference of the panel above;"
                     "\n"
                     r"fine: $\mathrm{Var}_{p(T)}(z)/T^{3}$",
        transform=bx.transAxes, fontsize=5.6, ha="right", va="top",
        color=S.GREY, linespacing=1.35)
for a in (ax, bx):
    a.tick_params(length=2.4, width=0.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

fig.savefig(S.out("fig142"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig142"), "written")

# ------------------------------------------------------- what the figure asserts
for name, scale in CASES:
    z = base * scale
    H = np.array([entropy(z, T) for T in Ts])
    print(f"  {name:>10}: H(0.05) = {H[0]:.4f}   H(1) = {entropy(z, 1.0):.4f}"
          f"   H(40) = {H[-1]:.4f}   monotone: {bool(np.all(np.diff(H) >= -1e-12))}")
print(f"  log V = {np.log(V):.4f}")
z = base * 2.0
err = max(abs((entropy(z, T + h) - entropy(z, T - h)) / (2 * h) - dHdT(z, T))
          / dHdT(z, T) for T in (0.3, 1.0, 3.0))
print(f"  worst relative disagreement, central difference vs Var/T^3: {err:.2e}")
