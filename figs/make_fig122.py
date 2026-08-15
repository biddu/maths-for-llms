"""F-12.2 -- per-expert load after 2000 steps under three balancing regimes.

Measured, not drawn.  A 64-expert routed MoE with real SwiGLU experts and a
real router is trained three times from the same initialisation, differing only
in what keeps the load balanced: nothing, the auxiliary loss of D-12.2, or the
bias controller of D-12.3.  The histograms are the per-expert token fractions
averaged over the last hundred steps, pooled across three seeds.

Committed data: figs/data/moe_regimes.npz, produced by work/moetoy.py.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mfestyle as S

S.apply()

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "data", "moe_regimes.npz"))
E = z["none_loads"].shape[1]
uniform = 1.0 / E

panels = [
    ("no balancing",            "none", S.GREY,       "//"),
    ("auxiliary loss",          "aux",  S.ACCENT_MID, "xx"),
    ("bias controller",         "bias", S.ACCENT,     ".."),
]

fig, axes = plt.subplots(3, 1, figsize=(3.95, 3.05), sharex=True,
                         constrained_layout=True)
edges = np.linspace(0, 2.8, 43)
for ax, (label, key, col, hatch) in zip(axes, panels):
    f = z[f"{key}_loads"].ravel() / uniform          # in units of the uniform load
    ax.hist(f, bins=edges, color="white", edgecolor=col, lw=0.6,
            hatch=hatch, zorder=3)
    ax.axvline(1.0, color=S.GREY_LIGHT, lw=0.6, ls=(0, (3, 2)), zorder=4)
    ff = z[f"{key}_loads"]
    ratio = (ff.max(1) / np.maximum(ff.min(1), 1e-12)).mean()
    cv = (ff.std(1) / ff.mean(1)).mean()
    loss = z[f"{key}_loss"].mean()
    ax.text(0.985, 0.88, label, transform=ax.transAxes, ha="right", va="top",
            fontsize=6.8, color=col, fontweight="bold")
    ax.text(0.985, 0.60, f"max/min {ratio:.1f}   CV {cv:.3f}\nfinal loss {loss:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.0,
            color=S.GREY, linespacing=1.3)
    ax.set_yticks([])
    ax.tick_params(length=2, width=0.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

axes[-1].set_xlim(0, 2.8)
axes[-1].set_xlabel("per-expert load, in multiples of the uniform share $1/E$")
axes[-1].annotate("the uniform share", xy=(1.02, axes[-1].get_ylim()[1] * 0.55),
                  xytext=(1.35, axes[-1].get_ylim()[1] * 0.72), fontsize=6.0,
                  color=S.GREY, ha="left", va="center",
                  arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GREY_LIGHT,
                                  shrinkA=1, shrinkB=1))
fig.savefig(S.out("fig122"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig122"), "written")
for label, key, _, _ in panels:
    f = z[f"{key}_loads"]
    print(f"  {label:<18} max/min {(f.max(1)/np.maximum(f.min(1),1e-12)).mean():6.2f}"
          f"   CV {(f.std(1)/f.mean(1)).mean():.4f}"
          f"   loss {z[f'{key}_loss'].mean():.4f}"
          f"   experts under a tenth of uniform: {(f < uniform*0.1).sum(1).mean():.1f}")
