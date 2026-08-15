"""F-11.4 -- the receptive field of a stacked sliding window.

A window of width w admits w positions per layer, so after L layers a query can
in principle be influenced by L*w positions back.  For Model D with w = 4096
that is exactly 131 072, the full extended context, which sounds like a proof
that windows are free.  The figure is drawn to make the qualification visible:
the far end of the horizon is reached only at the last layer, and only through
L successive hops of mixing, each of which passes information through a softmax
average rather than reading it directly.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mfestyle as S

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "..", "repo"))
sys.path.insert(0, REPO)
from arith.model_d import MODEL_D                                # noqa: E402
from arith.kv_cache import window_horizon                        # noqa: E402

S.apply()

w, L = 4096, MODEL_D.L
horizon = window_horizon(w, L)
assert horizon == MODEL_D.extended_context

fig, ax = plt.subplots(figsize=(3.95, 2.30), constrained_layout=True)

for l in range(1, L + 1):
    ax.add_patch(Rectangle((0, l - 0.42), l * w, 0.84, facecolor=S.ACCENT_PALE,
                           edgecolor="none", zorder=2))
# the staircase edge itself, which is the line the eye should follow
ax.step(np.arange(0, L + 1) * w, np.arange(0, L + 1) + 0.42, where="post",
        color=S.ACCENT, lw=1.1, zorder=4)

ax.axvline(MODEL_D.trained_context, color=S.GREY, lw=0.7, ls=(0, (4, 2)), zorder=3)
ax.text(MODEL_D.trained_context * 1.35, L * 0.46, "trained context\n8192",
        fontsize=6.0, color=S.GREY, ha="left", va="center", linespacing=1.25,
        zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

ax.plot([horizon], [L], marker="o", ms=3.0, mfc="white", mec=S.ACCENT, mew=1.0,
        zorder=5)
ax.annotate(f"horizon $Lw$ = {horizon:,}\nreached only here, after {L} hops",
            xy=(horizon, L), xytext=(horizon * 0.99, L * 0.20),
            fontsize=6.0, color=S.ACCENT, ha="right", va="center",
            linespacing=1.25,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.ACCENT_MID,
                            shrinkA=2, shrinkB=2))

for l in (1, 8, 16, 24, 32):
    ax.text(-horizon * 0.012, l, str(l), fontsize=6.0, color=S.GREY,
            ha="right", va="center")
ax.text(-horizon * 0.012, L * 1.16, "layer", fontsize=6.4, color=S.INK,
        ha="right", va="center")

ax.set_xlim(0, horizon * 1.02)
ax.set_ylim(0.2, L + 1.4)
ax.set_yticks([])
ax.set_xticks([0, 32768, 65536, 98304, 131072])
ax.set_xticklabels(["0", "32k", "64k", "96k", "128k"])
ax.set_xlabel("positions behind the query that can influence it")
ax.tick_params(length=2, width=0.5)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
fig.savefig(S.out("fig114"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig114"), "written")
print(f"  w = {w}, L = {L}, horizon = {horizon:,}"
      f" (= extended context? {horizon == MODEL_D.extended_context})")
print(f"  layers needed to cover the trained context {MODEL_D.trained_context}:"
      f" {MODEL_D.trained_context // w}")
