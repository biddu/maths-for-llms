"""F-6.2 -- GeLU is not monotone.

The interesting behaviour is a dip of 0.17 in a function that runs to 2 and
beyond, so the axes are cropped to make it visible rather than technically
present.  The shaded band is the interval on which GeLU decreases; its right
edge is the root of GeLU', which is the same abscissa as the minimum, and the
figure draws the link between the two explicitly.

Line style separates the two curves, so the figure carries in black ink.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm

gelu = lambda x: x * norm.cdf(x)
dgelu = lambda x: norm.cdf(x) + x * norm.pdf(x)
xstar = brentq(dgelu, -3.0, -0.1, xtol=1e-14)
ystar = gelu(xstar)

XLO, XHI, YLO, YHI = -3.2, 1.7, -0.36, 1.30
x = np.linspace(XLO, XHI, 1600)

fig, ax = plt.subplots(figsize=(3.90, 1.95), constrained_layout=True)

ax.axvspan(XLO, xstar, color=S.ACCENT_PALE, lw=0, zorder=0, alpha=0.62)
ax.axhline(0, lw=0.5, color=S.GREY_LIGHT, zorder=1)
ax.plot([xstar, xstar], [YLO, dgelu(XHI) * 0 + 0.0], lw=0.5, ls=(0, (1, 1.6)),
        color=S.GREY, zorder=2)

ax.plot(x, gelu(x), lw=1.15, color=S.ACCENT, ls=S.DASHES[0], zorder=4)
ax.plot(x, dgelu(x), lw=0.95, color=S.GREY, ls=S.DASHES[1], zorder=3)

ax.plot([xstar], [ystar], marker="o", ms=3.2, mfc="white", mec=S.ACCENT,
        mew=1.0, zorder=5)
ax.plot([xstar], [0.0], marker="s", ms=2.8, mfc="white", mec=S.GREY,
        mew=0.9, zorder=5)

ax.annotate(r"minimum $(-0.7518,\,-0.1700)$", xy=(xstar, ystar),
            xytext=(0.05, -0.245), fontsize=6.4, ha="left", va="center",
            color=S.INK,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GREY,
                            shrinkA=2, shrinkB=2))
ax.text(-3.05, 1.12, r"$\mathrm{GeLU}(x) = x\,\Phi(x)$", fontsize=7.0,
        color=S.ACCENT, va="center")
ax.text(-3.05, 0.92, r"$\mathrm{GeLU}'(x) = \Phi(x) + x\varphi(x)$",
        fontsize=7.0, color=S.GREY, va="center")
ax.text(-1.95, 0.42, "GeLU decreasing\n($\\mathrm{GeLU}' < 0$)", fontsize=6.4,
        ha="center", va="center", color=S.INK, linespacing=1.35)

ax.set_xlim(XLO, XHI); ax.set_ylim(YLO, YHI)
ax.set_xlabel("$x$", labelpad=1.0)
ax.set_xticks([-3, -2, -1, 0, 1])
ax.set_yticks([0.0, 0.5, 1.0])
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

fig.savefig(S.out("fig62"))
print(S.out("fig62"), "written;",
      "x* = %.6f  GeLU(x*) = %.6f  GeLU'' (0) = %.6f"
      % (xstar, ystar, 2 * norm.pdf(0)))
