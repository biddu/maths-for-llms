"""F-12.4 -- active against total parameters for shipped models.

The diagonal is where dense models live, and it is the line every "how big is
it?" comparison silently assumes.  A sparse model sits below it by its sparsity
ratio, and the vertical distance from the diagonal is exactly the factor by
which a single parameter count misleads you.

Configurations are the ones the vendors publish; the two reference models are
this book's own and are computed by arith/, not typed in.  See the source list
at the end of the chapter.
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
from arith.model_s import MODEL_S, totals                       # noqa: E402
from arith.model_d import MODEL_D, total_params                 # noqa: E402

S.apply()

t = totals(MODEL_S)
# name, total (B), active (B), is this book's model, label offset
POINTS = [
    ("Model D",  total_params(MODEL_D) / 1e9, total_params(MODEL_D) / 1e9, True,  (1.16, 0.80)),
    ("Model S",  t["total"] / 1e9,            t["active"] / 1e9,           True,  (0.60, 1.34)),
    ("Mixtral 8x7B", 46.7,  12.9,  False, (1.16, 0.78)),
    ("DBRX",         132.0, 36.0,  False, (0.78, 1.30)),
    ("Qwen3 235B",   235.0, 22.0,  False, (1.16, 0.80)),
    ("Qwen3 30B",     30.0,  3.0,  False, (1.16, 0.82)),
]

fig, ax = plt.subplots(figsize=(3.95, 2.62), constrained_layout=True)

grid = np.array([1.0, 2000.0])
ax.loglog(grid, grid, color=S.GREY, lw=0.9, zorder=2)
for r, lab in ((4, "4$\\times$"), (16, "16$\\times$"), (64, "64$\\times$")):
    ax.loglog(grid, grid / r, color=S.GREY_LIGHT, lw=0.6, ls=(0, (1, 2)), zorder=1)

for name, tot, act, ours, (dx, dy) in POINTS:
    col = S.ACCENT if ours else S.GREY
    ax.plot([tot], [act], marker="o", ms=4.2,
            mfc=col if ours else "white", mec=col, mew=1.0, zorder=4)
    ax.text(tot * dx, act * dy, name, fontsize=6.2, color=col,
            ha="right" if dx < 1 else "left", va="center", zorder=5)

# the vertical drop that M-12.1 is about
ax.annotate("", xy=(t["total"] / 1e9, t["active"] / 1e9),
            xytext=(t["total"] / 1e9, t["total"] / 1e9), zorder=3,
            arrowprops=dict(arrowstyle="<->", lw=0.7, color=S.ACCENT,
                            shrinkA=1, shrinkB=3))
ax.text(t["total"] / 1e9 * 1.14, (t["total"] * t["active"]) ** 0.5 / 1e9,
        f"{t['total']/t['active']:.0f}$\\times$", fontsize=6.4, color=S.ACCENT,
        ha="left", va="center")

ax.set_xlim(4, 2000)
ax.set_ylim(1.5, 900)
ax.set_xlabel("total parameters (billions)")
ax.set_ylabel("active parameters per token (billions)")
ax.set_xticks([10, 30, 100, 300, 1000])
ax.set_yticks([3, 10, 30, 100, 300])
ax.set_xticklabels(["10", "30", "100", "300", "1000"])
ax.set_yticklabels(["3", "10", "30", "100", "300"])
ax.tick_params(length=2, width=0.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# label the diagonals along themselves, at the angle the axes actually give
fig.canvas.draw()
p0 = ax.transData.transform((10.0, 10.0))
p1 = ax.transData.transform((100.0, 100.0))
angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
ax.text(5.2, 6.4, "dense: active $=$ total", fontsize=6.2, color=S.GREY,
        rotation=angle, rotation_mode="anchor", ha="left", va="bottom")
for r, lab in ((4, "4$\\times$"), (16, "16$\\times$"), (64, "64$\\times$")):
    ax.text(1750, 1750 / r * 1.06, lab, fontsize=5.8, color=S.GREY_LIGHT,
            rotation=angle, rotation_mode="anchor", ha="right", va="bottom")
fig.savefig(S.out("fig124"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig124"), "written")
for name, tot, act, ours, _ in POINTS:
    print(f"  {name:<14} total {tot:7.1f} B   active {act:6.1f} B"
          f"   sparsity {tot/act:5.1f}x")
