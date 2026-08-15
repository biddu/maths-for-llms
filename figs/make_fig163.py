"""F-16.3 -- shrinkage, and what the two repairs cost.

Recovered magnitude against true magnitude for three sparse coders on the same
almost-orthogonal dictionary.  The L1 solution is the identity shifted down by
lambda/2 and truncated at it, which is (16.17): one lambda does selection AND
shrinkage, and no sweep separates them.  TopK sits on the identity because it
has no penalty to bias it, and pays by issuing exactly k atoms to every token.
JumpReLU also sits on the identity above its threshold, and pays with a biased
gradient estimator for a step function.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mfestyle as S

S.apply()

LAM, THETA, K = 0.4, 0.2, 12
c = np.linspace(0, 1.6, 400)
soft = np.maximum(c - LAM / 2, 0.0)
jump = np.where(c > THETA, c, 0.0)

fig, ax = plt.subplots(figsize=(3.95, 2.05), constrained_layout=True)

ax.plot(c, c, color=S.GREY_LIGHT, lw=0.8, ls=(0, (2, 2)), zorder=2)
ax.plot(c, soft, color=S.ACCENT, lw=1.2, zorder=5)
ax.plot(c, jump, color=S.GREY, lw=1.0, ls=(0, (5, 1.6, 1, 1.6)), zorder=4)

rng = np.random.default_rng(1603)
ct = np.sort(rng.uniform(0.05, 1.55, K))[::-1]
ax.plot(ct, ct, marker="s", ls="none", ms=2.6, mfc="white", mec=S.GREY,
        mew=0.7, zorder=6)

ax.annotate(r"L1: $z = c - \lambda/2$", xy=(1.30, 1.30 - LAM / 2),
            xytext=(1.36, 0.72), fontsize=6.2, color=S.ACCENT, ha="right",
            va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GREY_LIGHT,
                            shrinkA=2, shrinkB=2))
ax.annotate("TopK and JumpReLU:\non the identity", xy=(1.05, 1.05),
            xytext=(0.50, 1.42), fontsize=6.2, color=S.GREY, ha="center",
            va="center", linespacing=1.3,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GREY_LIGHT,
                            shrinkA=2, shrinkB=2))
ax.annotate(r"exact zeros below $\lambda/2$", xy=(LAM / 2, 0.0),
            xytext=(0.60, 0.10), fontsize=6.0, color=S.ACCENT, ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GREY_LIGHT,
                            shrinkA=2, shrinkB=2))
ax.annotate(r"JumpReLU gates at $\theta$", xy=(THETA, THETA),
            xytext=(0.16, 0.72), fontsize=6.0, color=S.GREY, ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GREY_LIGHT,
                            shrinkA=2, shrinkB=2))

ax.set_xlabel(r"true magnitude $c_j = \langle x, w_j\rangle$")
ax.set_ylabel(r"recovered $z_j$")
ax.set_xlim(0, 1.6)
ax.set_ylim(-0.04, 1.6)
ax.set_xticks([0, 0.2, 0.4, 0.8, 1.2, 1.6])
ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6])
ax.tick_params(length=2.2, width=0.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

fig.savefig(S.out("fig163"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig163"), "written")
act = c > LAM / 2
print(f"  mean bias on active coordinates: {(c[act] - soft[act]).mean():.6f}"
      f"   (lambda/2 = {LAM/2})")
for cbar in (0.5, 1.0, 2.0):
    print(f"  norm ratio at cbar = {cbar}: {1 - LAM/(2*cbar):.4f}")
