"""F-7.4 -- where one block's backward memory goes.

Model D, b = 1, s = 8192, bf16, straight from
arith/model_d.py::activation_memory_backward.  Log x-axis, because the point is
that one entry is two orders of magnitude above the rest and a linear axis
would render everything else as a line of zero width.

Hatching separates what the memory is for: attention, feed-forward, and the
residual-stream tensors that belong to neither.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt

L, d, h, n_kv, d_h, d_ff, s, w = 32, 4096, 32, 8, 128, 14336, 8192, 2
rows = [
    (r"$P = \mathrm{softmax}(S)$", h * s * s * w, "attention", "///"),
    (r"$G,\ U,\ A$", 3 * s * d_ff * w, "feed-forward", "\\\\\\"),
    (r"$x,\hat{x}_1,Q,O_{\mathrm{cat}},y,\hat{x}_2$", 6 * s * d * w, "residual", ""),
    (r"$K,\ V$", 2 * s * n_kv * d_h * w, "attention", "///"),
    (r"$r_1,\ r_2$ (fp32)", 2 * s * 4, "residual", ""),
]
total = sum(r[1] for r in rows)
assert total == 5_435_883_520, total

y = np.arange(len(rows))[::-1]
fig, ax = plt.subplots(figsize=(3.90, 1.86), constrained_layout=True)
for (lab, b, kind, hatch), yy in zip(rows, y):
    face = S.ACCENT_PALE if kind == "attention" else "white"
    edge = S.ACCENT if kind == "attention" else S.GREY
    ax.barh(yy, b, height=0.62, left=1e5, color=face, edgecolor=edge,
            linewidth=0.6, hatch=hatch, zorder=3)
    pct = 100 * b / total
    txt = ("%.2f MB" if b < 1e6 else "%.1f MB") % (b / 1e6)
    ax.text((1e5 + b) * 1.30, yy, txt + ("  (%.1f%%)" % pct if pct > 1 else ""),
            va="center", fontsize=6.1, color=S.INK, zorder=4)

ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
ax.set_xscale("log"); ax.set_xlim(1e5, 1.2e12)
ax.set_xlabel("bytes", labelpad=1.5)
ax.set_xticks([1e6, 1e7, 1e8, 1e9])
ax.set_xticklabels(["1 MB", "10 MB", "100 MB", "1 GB"])
ax.tick_params(length=2, width=0.5)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.text(1.4e5, -0.86, "total %.3f GB per block  |  %.1f GB over $L = 32$"
        % (total / 1e9, total * L / 1e9), fontsize=6.3, color=S.INK)
ax.set_ylim(-1.25, len(rows) - 0.35)

fig.savefig(S.out("fig74"))
print(S.out("fig74"), "written; total %d bytes, P is %.1f%%"
      % (total, 100 * rows[0][1] / total))
