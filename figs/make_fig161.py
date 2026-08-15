"""F-16.1 -- the capacity of a 4096-wide residual stream.  Chapter-defining.

The picture the chapter argues from, and the one that gets screenshotted, so it
has to carry its own caveats.  Three things are drawn:

  * the bound exp(d eps^2 / 4), on a log axis;
  * the d = 4096 line, which is what fits EXACTLY orthogonally;
  * three marked points, at eps = 0.05 (13 directions, below d, so the bound is
    vacuous), eps = 0.1 (28,001, under seven times d), and eps = 0.107 (131,072,
    an expansion-32 dictionary).

The shaded region below the d-line is where the bound permits fewer directions
than exact orthogonality does.  It reaches to eps = 0.090, which is most of the
range anyone would want to quote, and it is the honest content of "exponentially
many directions": exponential in d at FIXED eps, and the constants decide
everything at real widths.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "repo"))
import mfestyle as S
from arith.sae_capacity import capacity, eps_for              # noqa: E402

S.apply()

D = 4096
eps = np.linspace(0.02, 0.16, 500)
m = np.array([capacity(D, e) for e in eps])
eps_vac = np.sqrt(4 * np.log(D) / D)
eps_32 = eps_for(D, 32 * D)

fig, ax = plt.subplots(figsize=(3.95, 2.35), constrained_layout=True)

ax.fill_between(eps, 1, D, where=eps <= eps_vac, color=S.GREY_LIGHT,
                alpha=0.30, lw=0, zorder=1)
ax.plot(eps, m, color=S.ACCENT, lw=1.2, zorder=5)
ax.axhline(D, color=S.GREY, lw=0.8, ls=(0, (4, 2)), zorder=3)
ax.text(0.0225, D * 0.62, r"$m = d = \mathrm{4096}$, exactly orthogonal",
        fontsize=6.1, color=S.GREY, ha="left", va="top")
ax.plot([eps_vac, eps_vac], [1, D], color=S.GREY, lw=0.6, ls=(0, (2, 2)),
        zorder=3)

MARKS = [(0.05, "13", (0.0285, 1.1e6), "left"),
         (0.10, "28,001", (0.1215, 3.2e2), "left"),
         (eps_32, "131,072", (0.1315, 2.2e4), "left")]
for e, lbl, xy, ha in MARKS:
    v = capacity(D, e)
    ax.plot([e], [v], marker="o", ms=3.2, mfc="white", mec=S.ACCENT, mew=0.9,
            zorder=7)
    ax.annotate(rf"$\varepsilon={e:.3f}$" + "\n" + lbl, xy=(e, v), xytext=xy,
                fontsize=6.0, color=S.ACCENT, ha=ha, va="center",
                linespacing=1.25,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GREY_LIGHT,
                                shrinkA=3, shrinkB=3))

ax.text(eps_vac - 0.0025, 1.28,
        r"vacuous below $\varepsilon = 0.090$",
        fontsize=6.0, color=S.GREY, ha="right", va="bottom")

ax.set_yscale("log")
ax.set_xlabel(r"coherence $\varepsilon$, the largest pairwise $|\cos|$ tolerated")
ax.set_ylabel(r"directions $m \leq \exp(d\varepsilon^{2}/4)$")
ax.set_xlim(0.02, 0.16)
ax.set_ylim(1, 3e11)
ax.set_xticks([0.02, 0.05, 0.08, 0.11, 0.14])
ax.set_yticks([1e0, 1e3, 1e6, 1e9])
ax.tick_params(length=2.2, width=0.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

fig.savefig(S.out("fig161"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig161"), "written")
for e in (0.05, 0.09, 0.1, eps_32, 0.15):
    print(f"  eps = {e:.6f}: m <= {capacity(D, e):>14,.0f}"
          f"   ({capacity(D, e)/D:.3g}x d)")
print(f"  vacuous below eps = {eps_vac:.6f}")
print(f"  an expansion-32 dictionary is admitted at eps = {eps_32:.6f}")
