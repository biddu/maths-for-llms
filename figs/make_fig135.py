"""F-13.5 -- NF4's sixteen levels against sixteen uniform ones.

The density is the standard normal a weight block is assumed to follow after
absmax normalisation.  NF4 places its levels at equal probability mass, so they
crowd where the weights are; int4 places them at equal spacing, so half of them
sit where almost no weight ever lands.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mfestyle as S

REPO = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "..", "repo")
sys.path.insert(0, os.path.abspath(REPO))
from arith.quant_formats import nf4_levels                     # noqa: E402

S.apply()

nf4 = nf4_levels()
uni = np.linspace(-1, 1, 16)
x = np.linspace(-1.05, 1.05, 1200)
sigma = 1 / 2.8                       # a block normalised so its extreme is 1
pdf = np.exp(-x ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)

fig, ax = plt.subplots(figsize=(3.95, 1.98), constrained_layout=True)
ax.fill_between(x, 0, pdf, color=S.ACCENT_PALE, lw=0, zorder=1)
ax.plot(x, pdf, color=S.GREY, lw=0.8, zorder=2)
top = pdf.max()
for v in nf4:
    ax.plot([v, v], [-0.16 * top, -0.04 * top], color=S.ACCENT, lw=0.9, zorder=3)
for v in uni:
    ax.plot([v, v], [1.04 * top, 1.16 * top], color=S.GREY, lw=0.9, zorder=3)
ax.text(-1.03, -0.10 * top, "NF4", fontsize=6.4, color=S.ACCENT, ha="right",
        va="center")
ax.text(-1.03, 1.10 * top, "int4", fontsize=6.4, color=S.GREY, ha="right",
        va="center")
ax.annotate("equal probability mass\nper level", xy=(0.28, -0.10 * top),
            xytext=(0.70, 0.50 * top), fontsize=6.0, color=S.ACCENT,
            ha="center", va="center", linespacing=1.25,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.ACCENT_MID,
                            shrinkA=2, shrinkB=2))
ax.annotate("equal spacing, so the\nouter levels are wasted",
            xy=(-0.93, 1.02 * top), xytext=(-0.66, 0.66 * top), fontsize=6.0,
            color=S.GREY, ha="center", va="center", linespacing=1.25,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GREY_LIGHT,
                            shrinkA=2, shrinkB=2))
ax.set_xlim(-1.30, 1.10)
ax.set_ylim(-0.22 * top, 1.30 * top)
ax.set_yticks([])
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_xlabel("weight, normalised so the block extreme is $1$")
ax.tick_params(length=2, width=0.5)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
fig.savefig(S.out("fig135"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig135"), "written")
print(f"  NF4 levels {len(nf4)}, exact zero present {np.any(nf4 == 0)}")
print(f"  narrowest NF4 gap {np.diff(nf4).min():.4f}, widest {np.diff(nf4).max():.4f}")
print(f"  int4 gap {np.diff(uni)[0]:.4f} everywhere")
