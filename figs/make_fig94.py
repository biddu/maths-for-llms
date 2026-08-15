"""F-9.4 -- Newton-Schulz flattens the spectrum.

Left: the singular values of a real momentum matrix, before and after five
iterations.  The matrix is AdamW's first moment for the gate projection of a
byte-level transformer trained for 300 steps, committed as figs/data/fig94.npz.
Its spectrum decays steeply: the sum-over-max effective rank is 17.7 out of
128, so the AdamW step is concentrated in a handful of directions and most of
the available ones receive almost nothing.

Right: the polynomial p(s) = as + bs^3 + cs^5 composed with itself k times.  It
is not converging to 1 and is not meant to.  Its two positive fixed points are
0.868 and 1.264, and after five steps every singular value above 0.0016 of the
Frobenius norm has been carried into the band [0.68, 1.21].  The direction is what
the linear-minimisation oracle needs; exact orthogonality is not.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, "data", "fig94.npz"))
before, after = D["s_before"], D["s_after"]
a, b, c = D["coeffs"]
p = lambda s: a * s + b * s ** 3 + c * s ** 5

fig, axes = plt.subplots(1, 2, figsize=(3.90, 2.02), constrained_layout=True)
i = np.arange(1, len(before) + 1)
axes[0].semilogy(i, before, color=S.GREY, ls=S.DASHES[1], lw=0.95,
                 marker="s", ms=1.8, markevery=8)
axes[0].semilogy(i, np.sort(after)[::-1], color=S.ACCENT, ls=S.DASHES[0],
                 lw=1.15, marker="o", ms=2.0, markevery=8)
axes[0].text(12, 4.2e-3, "before", fontsize=6.4, color=S.GREY)
axes[0].text(12, 1.45, "after 5 steps", fontsize=6.4, color=S.ACCENT)
axes[0].set_xlabel("singular value index", labelpad=1.5)
axes[0].set_ylabel(r"$s_i / \|G\|_F$", labelpad=2)
axes[0].set_title("a real momentum matrix", fontsize=6.8, pad=3, color=S.INK)
axes[0].set_ylim(5e-4, 4.0)

s = np.linspace(0, 1.28, 900)
axes[1].plot(s, p(s), color=S.ACCENT, ls=S.DASHES[0], lw=1.15, zorder=4)
axes[1].plot(s, s, color=S.GREY, ls=(0, (1, 2.0)), lw=0.6, zorder=1)
axes[1].axhspan(0.7, 1.2024, color=S.ACCENT_PALE, alpha=0.55, lw=0, zorder=0)

# the orbit of one small singular value, five steps, drawn as a cobweb
x = 0.05
for k in range(5):
    y = p(x)
    axes[1].plot([x, x], [x, y], color=S.GREY, lw=0.5, zorder=3)
    axes[1].plot([x, y], [y, y], color=S.GREY, lw=0.5, zorder=3)
    x = y
axes[1].plot([0.05], [0.05], marker="o", ms=2.4, mfc="white", mec=S.GREY,
             mew=0.8, zorder=5)
for fp in (0.86803, 1.26373):
    axes[1].plot([fp], [fp], marker="s", ms=2.8, mfc="white", mec=S.INK, mew=0.9,
                 zorder=5)

axes[1].text(0.05, 0.12, r"$s_0$", fontsize=6.2, color=S.GREY, ha="center")
axes[1].text(0.30, 1.31, r"$p(s) = as + bs^3 + cs^5$", fontsize=6.2, color=S.ACCENT)
axes[1].text(0.10, 1.03, "the band\nafter 5", fontsize=6.0, color=S.INK,
             linespacing=1.3)
axes[1].text(1.05, 0.38, "fixed points\n0.868, 1.264", fontsize=6.0,
             color=S.INK, ha="center", linespacing=1.3)
axes[1].set_xlabel("$s$", labelpad=1.5)
axes[1].set_title("the polynomial, and one orbit", fontsize=6.8, pad=3, color=S.INK)
axes[1].set_xlim(0, 1.28); axes[1].set_ylim(0, 1.55)

for ax in axes:
    ax.tick_params(length=2, width=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

fig.savefig(S.out("fig94"))
print(S.out("fig94"), "written; before [%.2e, %.3f] -> after [%.3f, %.3f]"
      % (before.min(), before.max(), after.min(), after.max()))
