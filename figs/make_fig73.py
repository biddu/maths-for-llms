"""F-7.3 -- gradient norm by depth, pre-norm and post-norm, L = 32.

Measured on a 32-block stack of the reference implementation at
initialisation, averaged over five seeds, with the gradient at the top of the
stack normalised to 1.  The data are committed as figs/data/fig73.npz because
regenerating them needs the worked backward pass, which lives on the solutions
branch so that it does not hand E-7.9 and E-7.10 to the reader.

Two panels, because the interesting comparison is not the one usually drawn.
Left: standard fan-in initialisation.  Right: the residual projections W_O and
W_down scaled by 1/sqrt(2L), which every modern implementation does.  The
scaling matters more than where the normalisation sits.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt

D = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "fig73.npz"))
L = int(D["L"]); l = np.arange(L + 1)

fig, axes = plt.subplots(1, 2, figsize=(3.90, 2.00), sharey=True,
                         constrained_layout=True)
panels = [(axes[0], "plain", r"fan-in init"),
          (axes[1], "scaled", r"$W_O, W_{\mathrm{down}}$ scaled by $1/\sqrt{2L}$")]
for ax, suf, title in panels:
    for key, colour, dash, lw in (("pre_" + suf, S.ACCENT, S.DASHES[0], 1.15),
                                  ("post_" + suf, S.GREY, S.DASHES[2], 0.95)):
        y = D[key]
        ax.semilogy(l, y, color=colour, ls=dash, lw=lw)
        ax.plot([0], [y[0]], marker="o", ms=2.6, color=colour, mfc="white",
                mew=0.9)
    ax.axhline(1.0, lw=0.5, ls=(0, (1, 2.0)), color=S.GREY_LIGHT, zorder=0)
    ax.set_title(title, fontsize=6.6, pad=3, color=S.INK)
    ax.set_xlabel("block $l$", labelpad=1.5)
    ax.set_xticks([0, 8, 16, 24, 32])
    ax.tick_params(length=2, width=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

axes[0].set_ylabel(r"$\|\partial L/\partial x^{(l)}\|$", labelpad=2)
axes[0].text(1.0, D["post_plain"][0] * 1.35, "post-norm", fontsize=6.3,
             color=S.GREY)
axes[0].text(4.2, 1.9, "pre-norm", fontsize=6.3, color=S.ACCENT)
axes[1].text(1.0, 2.6, "both stable", fontsize=6.3, color=S.INK)
axes[1].set_ylim(0.6, 90)

fig.savefig(S.out("fig73"))
for k in ("pre_plain", "post_plain", "pre_scaled", "post_scaled"):
    print("  %-12s ratio %7.3f  implied delta %.4f"
          % (k, D[k][0] / D[k][-1], (D[k][0] / D[k][-1]) ** (1 / L) - 1))
print(S.out("fig73"), "written")
