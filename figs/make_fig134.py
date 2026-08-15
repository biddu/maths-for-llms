"""F-13.4 -- where the memory goes, in three regimes.

Three stacked bars, one per regime, segmented into the three terms that move
independently.  The horizontal rules are real device capacities, because the
only question this table is asked is which machine.
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
from arith.model_d import finetune_memory                      # noqa: E402

S.apply()

rows = finetune_memory(b=8, s=4096)
labels = [("full fine-tune", "full"), ("LoRA $r=16$", "lora"), ("QLoRA $r=16$", "qlora")]
terms = [("weights", "weights", S.GREY_LIGHT, "..."),
         ("gradients and optimiser state", "state", S.ACCENT, "///"),
         ("activations", "act", S.ACCENT_PALE, "xxx")]

fig, ax = plt.subplots(figsize=(3.95, 2.34), constrained_layout=True)
ypos = np.arange(len(labels))[::-1]
for i, (nm, key) in enumerate(labels):
    left = 0.0
    for tname, tkey, col, hatch in terms:
        v = rows[key][tkey] / 1e9
        if v <= 0:
            continue
        ax.barh(ypos[i], v, left=left, height=0.52, color="white",
                edgecolor=S.ACCENT if col is S.ACCENT else S.GREY,
                hatch=hatch, lw=0.6, zorder=3)
        if v > 9:
            ax.text(left + v / 2, ypos[i], f"{v:.0f}", fontsize=6.2,
                    ha="center", va="center", zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
        left += v
    ax.text(left + 3, ypos[i], f"{left:.1f} GB", fontsize=6.6, va="center",
            color=S.ACCENT, fontweight="bold")

for cap, nm in ((24, "24 GB"), (40, "40"), (80, "80"), (141, "141")):
    ax.axvline(cap, color=S.GREY_LIGHT, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.text(cap, 2.52, nm, fontsize=5.8, color=S.GREY, ha="center", va="bottom")
ax.text(167, 2.86, "device capacity", fontsize=6.0, color=S.GREY, ha="right")

ax.set_yticks(ypos)
ax.set_yticklabels([nm for nm, _ in labels], fontsize=7.0)
ax.set_xlim(0, 168)
ax.set_ylim(-0.6, 3.05)
ax.set_xlabel("memory at $b = 8$, $s = 4096$ (GB)")
ax.tick_params(length=2, width=0.5)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", hatch=h,
                         edgecolor=S.ACCENT if c is S.ACCENT else S.GREY, lw=0.6)
           for _, _, c, h in terms]
ax.legend(handles, [t for t, _, _, _ in terms], fontsize=6.0, frameon=False,
          loc="lower right", handlelength=1.6, borderpad=0.2, labelspacing=0.35)
fig.savefig(S.out("fig134"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig134"), "written")
for nm, key in labels:
    v = rows[key]
    print(f"  {nm:<18} weights {v['weights']/1e9:7.2f}  state {v['state']/1e9:7.2f}"
          f"  act {v['act']/1e9:5.1f}  total {v['total']/1e9:6.1f} GB")
