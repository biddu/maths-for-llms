"""F-15.4 -- what dividing by the group standard deviation actually does.

For a binary verifiable reward at pass rate p, std(r) = sqrt(p(1-p)), so GRPO's
advantage normalisation multiplies every gradient from prompt x by

    w(p) = 1 / sqrt(p(1-p)),

which is a change of OBJECTIVE, not a variance reduction: the effective loss is
Sum_x w(x) J_x.  The curve has its minimum at p = 0.5 and diverges at both ends,
so the prompts receiving the largest gradients are the ones the model almost
always gets right and the ones it almost never gets right.  That is the opposite
of a curriculum, and it is a bias rather than a feature.

The rug shows a plausible per-prompt pass-rate distribution: bimodal, which is
what a verifier on a mixed-difficulty set produces, and therefore concentrated
in exactly the two regions where w is largest.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mfestyle as S

S.apply()

w = lambda p: 1.0 / np.sqrt(p * (1.0 - p))
p = np.linspace(0.002, 0.998, 800)

fig, ax = plt.subplots(figsize=(3.95, 2.05), constrained_layout=True)

ax.plot(p, w(p), color=S.ACCENT, lw=1.15, zorder=4)
ax.axhline(w(0.5), color=S.GREY_LIGHT, lw=0.6, ls=(0, (2, 2)), zorder=2)
ax.plot([0.5], [w(0.5)], marker="o", ms=3.0, mfc="white", mec=S.ACCENT,
        mew=0.9, zorder=6)
ax.annotate(r"minimum, $w = 2$ at $p = 0.5$", xy=(0.5, w(0.5)),
            xytext=(0.5, 2.75), fontsize=6.0, color=S.GREY, ha="center",
            va="bottom",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GREY_LIGHT,
                            shrinkA=2, shrinkB=3))

for px, lbl, ha in ((0.03, "almost never\nsolved", "left"),
                    (0.97, "almost always\nsolved", "right")):
    ax.annotate(lbl, xy=(px, w(px)), xytext=(px, w(px) * 2.6), fontsize=6.0,
                color=S.ACCENT, ha=ha, va="bottom", linespacing=1.3,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GREY_LIGHT,
                                shrinkA=2, shrinkB=2))

# a bimodal histogram of per-prompt pass rates along the foot of the panel.
# A rug cannot show density once G quantises the rates onto 15 levels, and
# density is the whole point: the prompts pile up exactly where w is largest.
rng = np.random.default_rng(1504)
rates = np.clip(np.concatenate([rng.beta(9.0, 1.6, 400),
                                rng.beta(1.4, 9.0, 400)]), 0.02, 0.98)
G = 16
rates = np.clip(np.round(rates * G) / G, 1 / G, (G - 1) / G)
cnt, edges = np.histogram(rates, bins=np.arange(0, 1.0001, 1 / G))
base, top = 1.27, 1.86
h = base + (top - base) * cnt / cnt.max()
ax.bar(edges[:-1] + 0.5 / G, h - base, bottom=base, width=0.85 / G,
       color=S.GREY_LIGHT, edgecolor="none", zorder=3)
ax.text(0.5, 1.245, r"per-prompt pass rates, $G=16$", fontsize=5.8,
        color=S.GREY, ha="center", va="bottom")

ax.set_yscale("log")
ax.set_xlabel(r"pass rate $p$ on a prompt")
ax.set_ylabel(r"gradient weight $1/\sqrt{p(1-p)}$")
ax.set_xlim(0, 1)
ax.set_ylim(1.20, 60)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticks([2, 5, 10, 20, 50])
ax.set_yticklabels(["2", "5", "10", "20", "50"])
ax.tick_params(length=2.2, width=0.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

fig.savefig(S.out("fig154"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig154"), "written")
for q in (0.01, 0.05, 0.5, 0.95, 0.99):
    print(f"  p = {q:>5}: w = {w(q):7.4f}   relative to p = 0.5: {w(q)/w(0.5):6.4f}")
print(f"  fraction of drawn prompts with w >= 2x the minimum: "
      f"{(w(rates) >= 2 * w(0.5)).mean():.3f}")
