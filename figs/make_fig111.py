"""F-11.1 -- KV-cache size per sequence against context length.

One of the book's five load-bearing figures, so it is drawn to be lifted: no
legend, every line labelled at its own right-hand end, one grey capacity rule,
and the two solid lines placed so the 671B-below-8B inversion reads without
consulting the axis.

Every byte figure comes from arith/kv_cache.py.  Nothing here is typed in.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mfestyle as S

# <tree>/../repo: a no-op when this file is run from the repository, and the
# hop across when it is run from the LaTeX tree.  Both are supported because
# the figure is built in one and typeset in the other.
REPO = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "..", "repo")
sys.path.insert(0, os.path.abspath(REPO))
from arith.kv_cache import schemes, cache_bytes                  # noqa: E402
from arith.model_d import MODEL_D, total_params                  # noqa: E402
from arith.accelerators import DEFAULT, GiB                      # noqa: E402

S.apply()

s = np.logspace(10, 17, 400, base=2.0)
rows = schemes()

# label, style, colour, and where the label sits.  D-MHA and S-MLA are the two
# solid lines: the inversion is the figure's whole point, so they carry the
# strongest mark and sit adjacent in the label stack.
plan = [
    ("D, MHA",          "D MHA", S.DASHES[0], S.GREY,       1.6),
    ("S, MLA",          "S MLA", S.DASHES[0], S.ACCENT,     1.7),
    ("D, GQA (shipped)", "D GQA", S.DASHES[1], S.ACCENT_MID, 1.4),
    ("D, MQA",          "D MQA", S.DASHES[3], S.GREY_LIGHT, 1.2),
    ("S, MHA",          "S MHA", S.DASHES[2], S.GREY,       1.2),
]

fig, ax = plt.subplots(figsize=(3.95, 2.62), constrained_layout=True)
for label, key, dash, col, lw in plan:
    y = cache_bytes(rows[key]["bytes"], s) / GiB
    ax.loglog(s, y, ls=dash, color=col, lw=lw, base=2 if False else 10, zorder=3)

ax.set_xscale("log", base=2)
ax.set_yscale("log", base=10)

# the capacity rule: what is left on the part after Model D's weights
free = (DEFAULT.hbm_bytes - 2 * total_params(MODEL_D)) / GiB
ax.axhline(free, color=S.GREY_LIGHT, lw=0.8, ls=(0, (3, 2)), zorder=2)
ax.text(2 ** 10.15, free * 1.35, f"{free:.0f} GiB left after the weights",
        fontsize=6.0, color=S.GREY, ha="left", va="bottom")

# markers and labels at the right end, stacked so nothing collides
xmax = 2 ** 17
ymax = {k: cache_bytes(rows[k]["bytes"], xmax) / GiB for _, k, _, _, _ in plan}
for label, key, dash, col, lw in plan:
    y = ymax[key]
    ax.plot([xmax], [y], marker="o", ms=2.6, mfc="white", mec=col, mew=0.9, zorder=4)
    ax.text(xmax * 1.13, y, label, fontsize=6.2, color=col, va="center", ha="left")

# the inversion, marked quantitatively: a 671 B model with MLA carries a smaller
# cache per token than an 8 B model with MHA.  D-GQA sits between the two, which
# is arithmetic and not a layout choice, so the point is made by an annotation
# rather than by putting the two solid lines adjacent in the label stack.
xa = 2 ** 15.4
lo, hi = (cache_bytes(rows["S MLA"]["bytes"], xa) / GiB,
          cache_bytes(rows["D MHA"]["bytes"], xa) / GiB)
ax.annotate("", xy=(xa, lo), xytext=(xa, hi), zorder=5,
            arrowprops=dict(arrowstyle="<->", lw=0.7, color=S.ACCENT,
                            shrinkA=0, shrinkB=0))
ax.text(xa * 0.93, (lo * hi) ** 0.5, f"{rows['D MHA']['bytes']/rows['S MLA']['bytes']:.1f}$\\times$",
        fontsize=6.2, color=S.ACCENT, ha="right", va="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=0.8))

# the one crossing inside the plotted range
xc = free * GiB / rows["S MHA"]["bytes"]
ax.plot([xc], [free], marker="x", ms=4.0, mew=1.0, color=S.GREY, zorder=5)
# the region above the rule and left of the S-MHA line is the only empty part
# of the panel, so the crossing is labelled there and pointed at
ax.annotate("one sequence stops fitting", xy=(xc, free * 1.06),
            xytext=(xc * 0.90, free * 3.4), fontsize=6.0, color=S.GREY,
            ha="right", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GREY_LIGHT,
                            shrinkA=2, shrinkB=1))

ax.set_xlim(2 ** 10, 2 ** 17)
ax.set_ylim(1e-2, 1.2e3)
ax.set_xticks([2 ** k for k in (10, 12, 14, 16, 17)])
ax.set_xticklabels(["1k", "4k", "16k", "64k", "128k"])
ax.set_xlabel("context length $s$ (tokens)")
ax.set_ylabel("KV cache per sequence (GiB)")
ax.tick_params(length=2, width=0.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.savefig(S.out("fig111"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig111"), "written")

for label, key, _, _, _ in plan:
    b = rows[key]["bytes"]
    crosses = free * GiB / b
    print(f"  {label:<17} {b/1024:9.2f} KiB/token   {cache_bytes(b, 2**17)/GiB:8.3f} GiB at 128k"
          f"   crosses the rule at s = {crosses:,.0f}")
