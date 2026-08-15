"""F-6.3 -- what gating buys, as a picture of the degree-2 interaction surface.

Left:  an ungated unit's second-order term, (phi''(0)/2) <x,k>^2, which for GeLU
       is <x,k>^2 / sqrt(2 pi).  It is a square, so it is non-negative
       everywhere, its zero set is the single line perpendicular to k, and its
       level sets are pairs of parallel lines.  One direction, self-interacting.
Right: a gated unit's, (1/2) <x,w><x,v> with w != v.  The symmetric part of
       w v^T has one positive and one negative eigenvalue, so this is a saddle:
       it changes sign, and its zero set is the pair of lines perpendicular to
       w and to v.

Contours only, no fill, so the figure reads identically in colour and in black
ink.  Sign is carried three ways: line style (solid positive, dashed negative),
an explicit + or - in each region, and the heavy zero contour.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt

LIM = 1.5
k = np.array([1.0, 0.35]); k /= np.linalg.norm(k)
w = np.array([1.0, 0.22]); w /= np.linalg.norm(w)
v = np.array([0.22, 1.0]); v /= np.linalg.norm(v)

g = np.linspace(-LIM, LIM, 501)
X, Y = np.meshgrid(g, g)
Z_un = (X * k[0] + Y * k[1]) ** 2 / np.sqrt(2 * np.pi)
Z_gt = 0.5 * (X * w[0] + Y * w[1]) * (X * v[0] + Y * v[1])
perp = lambda u: np.array([-u[1], u[0]])


def nodal(ax, u):
    e = perp(u)
    p = e * (LIM / max(abs(e[0]), abs(e[1])))     # stop exactly on the frame
    ax.plot([-p[0], p[0]], [-p[1], p[1]], lw=1.4, color=S.ACCENT,
            solid_capstyle="butt", zorder=3, clip_on=True)


def arrow(ax, u, name):
    ax.annotate("", xy=1.02 * u, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=S.ACCENT,
                                mutation_scale=6, shrinkA=0, shrinkB=0))
    ax.text(1.19 * u[0], 1.19 * u[1], f"${name}$", fontsize=7.2,
            color=S.ACCENT, ha="center", va="center", zorder=5)


def sign(ax, pos, s):
    ax.text(pos[0], pos[1], s, fontsize=10.0, color=S.INK, ha="center",
            va="center", zorder=5)


fig, axes = plt.subplots(1, 2, figsize=(3.90, 2.16), constrained_layout=True)
levels = np.array([-0.5, -0.32, -0.18, -0.08, -0.02,
                   0.02, 0.08, 0.18, 0.32, 0.5])

for ax, Z in zip(axes, (Z_un, Z_gt)):
    for lv, ls in ((levels[levels > 0], "-"), (levels[levels < 0], (0, (3, 1.8)))):
        keep = lv[(lv > Z.min()) & (lv < Z.max())]
        if len(keep):
            ax.contour(X, Y, Z, levels=keep, colors=S.GREY, linewidths=0.55,
                       linestyles=[ls] * len(keep))
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color(S.GREY_LIGHT)

nodal(axes[0], k); arrow(axes[0], k, "k")
sign(axes[0], -1.05 * k, "$+$"); sign(axes[0], 1.05 * perp(k) + 0.42 * k, "$+$")

nodal(axes[1], w); nodal(axes[1], v)
arrow(axes[1], w, "w"); arrow(axes[1], v, "v")
b = (w + v) / np.linalg.norm(w + v)
c = (w - v) / np.linalg.norm(w - v)
for p, s in ((1.12 * b, "$+$"), (-1.12 * b, "$+$"),
             (1.12 * c, "$-$"), (-1.12 * c, "$-$")):
    sign(axes[1], p, s)

axes[0].set_title(r"ungated: $\langle x,k\rangle^2/\sqrt{2\pi}$",
                  fontsize=6.9, pad=3.0, color=S.INK)
axes[1].set_title(r"gated: $\langle x,w\rangle\langle x,v\rangle/2$",
                  fontsize=6.9, pad=3.0, color=S.INK)
for ax, msg in ((axes[0], "one nodal line, one sign"),
                (axes[1], "two nodal lines, both signs")):
    ax.text(0, -LIM + 0.10, msg, fontsize=6.3, ha="center", va="bottom",
            color=S.INK, zorder=6,
            bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none"))

fig.savefig(S.out("fig63"))
print(S.out("fig63"), "written;",
      "ungated range [%.3f, %.3f]" % (Z_un.min(), Z_un.max()),
      "| gated range [%.3f, %.3f]" % (Z_gt.min(), Z_gt.max()))
