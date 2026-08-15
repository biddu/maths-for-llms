"""F-7.2 -- the cost of the softmax backward, with and without the Jacobian.

Both curves are exact counts, not timings, so the figure does not depend on
whose GPU it was drawn on.  For one s x s attention matrix:

    explicit route   build diag(p) - p p^T for each of the s rows and multiply,
                     ~4 s^3 flops, and a whole s x s Jacobian resident at a time
    equation (7.6)   two elementwise products and one row reduction,
                     ~4 s^2 flops, and one length-s vector

Both ratios are exactly s, which is the point: the saving is not a constant
factor, it grows with the context you are trying to serve.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt

s = np.array([512, 1024, 2048, 4096, 8192, 16384], dtype=float)
flops_j, flops_e = 4 * s ** 3, 4 * s ** 2
bytes_j, bytes_e = 4 * s ** 2, 4 * s                 # fp32 working set

fig, axes = plt.subplots(1, 2, figsize=(3.90, 1.98), constrained_layout=True)
for ax, (yj, ye, lab) in zip(axes, ((flops_j, flops_e, "arithmetic (FLOP)"),
                                    (bytes_j, bytes_e, "working set (bytes)"))):
    ax.loglog(s, yj, color=S.GREY, ls=S.DASHES[1], lw=0.95, marker="s", ms=2.6)
    ax.loglog(s, ye, color=S.ACCENT, ls=S.DASHES[0], lw=1.15, marker="o", ms=2.8)
    ax.axvline(8192, lw=0.5, ls=(0, (1, 1.8)), color=S.GREY_LIGHT, zorder=0)
    ax.set_title(lab, fontsize=6.8, pad=3, color=S.INK)
    ax.set_xlabel("sequence length $s$", labelpad=1.5)
    ax.set_xticks([512, 2048, 8192]); ax.set_xticklabels(["512", "2k", "8k"])
    ax.tick_params(length=2, width=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

axes[0].text(560, 1.1e11, "explicit\nJacobian", fontsize=6.2, color=S.GREY,
             va="center", linespacing=1.3)
axes[0].text(1250, 3.0e5, "equation (7.6)", fontsize=6.2, color=S.ACCENT)
axes[1].text(560, 3.0e8, "explicit\nJacobian", fontsize=6.2, color=S.GREY,
             va="center", linespacing=1.3)
axes[1].text(1250, 1.7e3, "equation (7.6)", fontsize=6.2, color=S.ACCENT)
for ax, y, txt in ((axes[0], 3.0e12, r"$s = 8192$"), (axes[1], 1.4e6, r"$s = 8192$")):
    ax.text(8192, y, txt, fontsize=6.0, color=S.INK, ha="center")

fig.savefig(S.out("fig72"))
i = list(s).index(8192)
print(S.out("fig72"), "written; at s = 8192:",
      "%.3g vs %.3g FLOP (x%d);" % (flops_j[i], flops_e[i], flops_j[i]/flops_e[i]),
      "%.1f MB vs %.1f kB (x%d)" % (bytes_j[i]/1e6, bytes_e[i]/1e3,
                                    bytes_j[i]/bytes_e[i]))
