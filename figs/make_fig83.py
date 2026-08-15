"""F-8.3 -- label smoothing: the confidence bound it buys and the price it charges.

Left axis, solid: the optimal logit gap z_y - z_j = log((1-eps+eps/V)/(eps/V)),
which is what makes the smoothed objective have a finite minimiser at all.  It
diverges like -log(eps) as eps -> 0, and at eps = 0 exactly there is no finite
minimiser: the unsmoothed loss asks for p_j = 0 and pushes the gap forever.

Right axis, dashed: the loss floor H(q), the smallest cross-entropy the
smoothed objective can attain even from a perfect model.  It is not a small
correction at LLM vocabularies.  At V = 128256 and eps = 0.1 it is 1.501 nats,
which is most of the distance between a good model and a mediocre one.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt

V = 128256
eps = np.logspace(-4, np.log10(0.5), 500)
py = 1 - eps + eps / V
pj = eps / V
gap = np.log(py / pj)
floor = -(py * np.log(py) + (V - 1) * pj * np.log(pj))

E0 = 0.1
g0 = np.log((1 - E0 + E0 / V) / (E0 / V))
f0 = -( (1 - E0 + E0 / V) * np.log(1 - E0 + E0 / V)
        + (V - 1) * (E0 / V) * np.log(E0 / V) )

fig, ax = plt.subplots(figsize=(3.90, 2.05), constrained_layout=True)
ax2 = ax.twinx()

ax.semilogx(eps, gap, color=S.ACCENT, ls=S.DASHES[0], lw=1.15, zorder=4)
ax2.semilogx(eps, floor, color=S.GREY, ls=S.DASHES[1], lw=0.95, zorder=3)

ax.axvline(E0, lw=0.5, ls=(0, (1, 1.8)), color=S.GREY_LIGHT, zorder=1)
ax.plot([E0], [g0], marker="o", ms=3.0, mfc="white", mec=S.ACCENT, mew=1.0, zorder=5)
ax2.plot([E0], [f0], marker="s", ms=2.8, mfc="white", mec=S.GREY, mew=0.9, zorder=5)

ax.annotate(r"$\varepsilon = 0.1$:  gap $13.96$", xy=(E0, g0),
            xytext=(0.0055, 12.4), fontsize=6.3, color=S.ACCENT, ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.ACCENT, shrinkA=2, shrinkB=3))
ax2.annotate(r"floor $1.50$ nats", xy=(E0, f0), xytext=(0.0055, 1.30),
             fontsize=6.3, color=S.GREY, ha="left",
             arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GREY, shrinkA=2, shrinkB=3))
ax.text(1.4e-4, 24.6, "optimal logit gap (left)", fontsize=6.4, color=S.ACCENT)
ax.text(1.4e-4, 22.4, "loss floor $H(q)$ (right)", fontsize=6.4, color=S.GREY)

ax.set_xlabel(r"label-smoothing $\varepsilon$", labelpad=1.5)
ax.set_ylabel("logit gap (nats)", color=S.ACCENT, labelpad=2)
ax2.set_ylabel("loss floor (nats)", color=S.GREY, labelpad=3)
ax.set_xlim(1e-4, 0.5); ax.set_ylim(8, 27); ax2.set_ylim(0, 2.7)
ax.tick_params(axis="y", colors=S.ACCENT, length=2, width=0.5)
ax2.tick_params(axis="y", colors=S.GREY, length=2, width=0.5)
ax.tick_params(axis="x", length=2, width=0.5)
ax.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
ax.spines["left"].set_color(S.ACCENT); ax2.spines["right"].set_color(S.GREY)
ax2.spines["left"].set_visible(False)

fig.savefig(S.out("fig83"))
print(S.out("fig83"), "written; at eps = 0.1, V = %d: gap %.4f, floor %.4f" % (V, g0, f0))
