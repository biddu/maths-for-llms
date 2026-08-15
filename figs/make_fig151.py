"""F-15.1 -- preference probability, and the invariance class of the reward.

Two panels for two facts that the whole chapter rests on.

Left: the Bradley-Terry probability against the reward margin, with the
GRADIENT WEIGHT sigma(-Delta) overlaid.  The two curves crossing at the origin
is D-15.2's self-annealing made visible: a pair the model already has right
contributes almost nothing, and at a margin of 8 it contributes 0.03% of what a
backwards pair does.

Right: four completions scored by r, by r + 1, and by r + c(x).  Only the
DIFFERENCES are observable, which is why the whole column can slide.  That
freedom is the same freedom that reappears in Section 15.5 as beta log Z(x),
and it is why Z cancels.  Drawing them together is the point of the figure.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mfestyle as S

S.apply()

sig = lambda u: 1.0 / (1.0 + np.exp(-u))

fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.95, 1.85),
                             gridspec_kw={"width_ratios": [1.15, 1.0]},
                             constrained_layout=True)

# ------------------------------------------------------------------- left
d = np.linspace(-8, 8, 400)
ax.plot(d, sig(d), color=S.ACCENT, lw=1.15, zorder=4)
ax.plot(d, sig(-d), color=S.GREY, lw=1.0, ls=(0, (4, 2)), zorder=4)
ax.axhline(0.5, color=S.GREY_LIGHT, lw=0.6, ls=(0, (1, 2)), zorder=2)
ax.axvline(0.0, color=S.GREY_LIGHT, lw=0.6, ls=(0, (1, 2)), zorder=2)
ax.text(4.6, 0.90, r"$\sigma(\Delta)$", fontsize=6.6, color=S.ACCENT,
        ha="center", va="center")
ax.text(-4.6, 0.90, r"$\sigma(-\Delta)$", fontsize=6.6, color=S.GREY,
        ha="center", va="center")
ax.annotate("the gradient weight:\nzero once the pair\nis comfortably right",
            xy=(5.0, sig(-5.0)), xytext=(1.15, 0.30), fontsize=5.7,
            color=S.GREY, ha="left", va="center", linespacing=1.3,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GREY_LIGHT,
                            shrinkA=2, shrinkB=2))
ax.set_xlabel(r"reward margin $\Delta = r(x,y_w) - r(x,y_l)$")
ax.set_ylabel("probability")
ax.set_xlim(-8, 8)
ax.set_ylim(-0.03, 1.05)
ax.set_xticks([-8, -4, 0, 4, 8])
ax.set_yticks([0, 0.5, 1])
ax.tick_params(length=2.2, width=0.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# ------------------------------------------------------------------ right
# Three scorings of the same four completions.  Each carries its own zero,
# ruled and labelled, so the eye sees one picture translated three times:
# the bars move, every gap between them does not.
r = np.array([1.2, 0.4, -0.3, -1.0])
offsets = [1.0, 0.0, -0.9]
names = [r"$r+1$", r"$r$", r"$r+c(x)$"]
hatches = ["///", "", "..."]
w = 0.25
for k, (off, nm, ht) in enumerate(zip(offsets, names, hatches)):
    x = np.arange(4) + (k - 1) * w
    bx.plot([x[0] - w * 0.6, x[-1] + w * 0.6], [off, off], color=S.GREY_LIGHT,
            lw=0.6, ls=(0, (2, 1.6)), zorder=2)
    bx.bar(x, r, bottom=off, width=w * 0.88, color="white", edgecolor=S.INK,
           lw=0.5, hatch=ht, zorder=3)
    bx.text(3.62, off, nm, fontsize=5.9, color=S.INK, ha="left", va="center")

# the same gap, bracketed on the top and bottom scorings
for k in (0, 2):
    off = offsets[k]
    x0, x1 = 0 + (k - 1) * w, 1 + (k - 1) * w
    y = off + r[0] + 0.20
    bx.plot([x0, x0, x1, x1], [y - 0.11, y, y, y - 0.11], color=S.ACCENT,
            lw=0.7, zorder=5, solid_joinstyle="miter")
    bx.text((x0 + x1) / 2, y + 0.04, r"$0.8$", fontsize=5.8, color=S.ACCENT,
            ha="center", va="bottom")
bx.text(0.5, 3.32, "only the differences are observable", fontsize=6.0,
        color=S.ACCENT, ha="center", va="bottom")
bx.set_xticks(np.arange(4))
bx.set_xticklabels(["$y_1$", "$y_2$", "$y_3$", "$y_4$"])
bx.set_ylabel("reward")
bx.set_xlim(-0.55, 4.35)
bx.set_ylim(-2.3, 3.85)
bx.set_yticks([-2, -1, 0, 1, 2])
bx.tick_params(length=2.2, width=0.5)
for sp in ("top", "right"):
    bx.spines[sp].set_visible(False)

fig.savefig(S.out("fig151"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig151"), "written")
for D in (0, 1, 4, 8):
    print(f"  margin {D}: sigma(D) = {sig(D):.6f}, weight sigma(-D) = {sig(-D):.6f}"
          f"  ({100*sig(-D)/sig(0.0):.3f}% of a tied pair)")
print(f"  the three reward columns differ by a constant and agree on every"
      f" difference: {np.allclose(np.diff(r), np.diff(r + 1))}")
