"""F-14.4 -- speedup against draft length, from (14.17).

Five acceptance rates at Model D's draft cost ratio c = 1/8.  What the figure
has to make unmissable is that the curves are ORDERED BY alpha and that moving
along one of them is a much smaller effect than moving between them: A-14.1's
whole conclusion is that acceptance, not draft length, is the lever.  So the
maxima are marked, the break-even line at S = 1 is ruled, and the vertical
distance between curves at their own maxima is left to speak for itself.

The lowest curve is the one that earns its place: at alpha = 0.5 the maximum is
1.40x and the curve crosses BELOW one before gamma = 9, so a bad draft does not
merely fail to help, it costs throughput.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "repo"))
import mfestyle as S
from arith.decoding import speedup                                # noqa: E402

S.apply()

C = 1 / 8
GAMMAS = np.arange(1, 17)
ALPHAS = (0.9, 0.8, 0.7, 0.6, 0.5)
STYLES = ["-", (0, (4, 2)), (0, (5, 1.6, 1, 1.6)), (0, (1, 1.5)),
          (0, (6, 1.6, 1, 1.6, 1, 1.6))]
COLS = [S.ACCENT, S.ACCENT, S.GREY, S.GREY, S.GREY_LIGHT]

fig, ax = plt.subplots(figsize=(3.95, 2.45), constrained_layout=True)

for a, ls, col in zip(ALPHAS, STYLES, COLS):
    y = np.array([speedup(a, int(g), C) for g in GAMMAS])
    ax.plot(GAMMAS, y, ls=ls, color=col, lw=1.05, zorder=4)
    k = int(np.argmax(y))
    ax.plot(GAMMAS[k], y[k], marker="o", ms=3.0, mfc="white", mec=col,
            mew=0.9, zorder=6)
    # direct labelling at the right-hand end, nudged apart where they crowd
    ax.text(16.35, y[-1], rf"$\alpha_{{\mathrm{{acc}}}}={a}$", fontsize=6.2,
            color=col, ha="left", va="center")

ax.axhline(1.0, color=S.GREY_LIGHT, lw=0.7, ls=(0, (2, 2)), zorder=2)
ax.text(1.25, 1.05, "break-even", fontsize=6.0, color=S.GREY, ha="left",
        va="bottom")
ax.annotate("open circles mark $\\gamma^{\\star}$", xy=(9, 3.065),
            xytext=(6.4, 3.42), fontsize=6.0, color=S.GREY, ha="center",
            va="bottom",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GREY_LIGHT,
                            shrinkA=1.5, shrinkB=2.5))
ax.set_xlabel(r"draft length $\gamma$")
ax.set_ylabel(r"speedup $S(\alpha_{\mathrm{acc}},\gamma,c)$, $c=1/8$")
ax.set_xlim(0.6, 16.4)
ax.set_ylim(0.55, 3.62)
ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16])
ax.tick_params(length=2.4, width=0.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

fig.savefig(S.out("fig144"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig144"), "written")

for a in ALPHAS:
    y = [speedup(a, int(g), C) for g in GAMMAS]
    k = int(np.argmax(y))
    below = [int(g) for g, v in zip(GAMMAS, y) if v < 1.0]
    print(f"  alpha={a}: max {y[k]:.4f}x at gamma={GAMMAS[k]}"
          f"   S(16)={y[-1]:.4f}"
          + (f"   below 1 from gamma={min(below)}" if below else ""))
