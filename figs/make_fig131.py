"""F-13.1 -- the affine grid, and the sawtooth error it bounds.

Two panels sharing an axis: above, a weight distribution and its four-bit
reconstruction; below, the error, which is a sawtooth of amplitude s_q/2
wherever the clamp is inactive and grows without bound outside it.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mfestyle as S

S.apply()

b_q, clip = 4, 2.5
x = np.linspace(-3.4, 3.4, 4000)
s_q = (2 * clip) / (2 ** b_q - 1)
z = np.round(clip / s_q)
xhat = s_q * (np.clip(np.round(x / s_q) + z, 0, 2 ** b_q - 1) - z)

fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.95, 2.62), sharex=True,
                             height_ratios=[2.1, 1.0], constrained_layout=True)
ax.plot(x, x, color=S.GREY, lw=0.8, ls=(0, (4, 2)), zorder=2)
ax.step(x, xhat, where="mid", color=S.ACCENT, lw=1.0, zorder=3)
for e in (-clip, clip):
    ax.axvline(e, color=S.GREY_LIGHT, lw=0.6, ls=(0, (1, 2)), zorder=1)
ax.set_ylabel("reconstruction $\\hat x$")
ax.text(-3.3, 2.4, f"$b_q = {b_q}$, sixteen levels", fontsize=6.4, color=S.ACCENT)
ax.text(clip + 0.08, -2.6, "clamp", fontsize=6.0, color=S.GREY, ha="left")
ax.set_ylim(-3.4, 3.4)

err = xhat - x
bx.plot(x, err, color=S.ACCENT, lw=0.8, zorder=3)
for sgn in (1, -1):
    bx.axhline(sgn * s_q / 2, color=S.GREY, lw=0.7, ls=(0, (3, 2)), zorder=2)
bx.text(-3.3, s_q / 2 * 1.55, "$\\pm s_q/2$", fontsize=6.2, color=S.GREY, va="bottom")
bx.set_ylabel("error")
bx.set_xlabel("weight $x$, in signal standard deviations")
bx.set_ylim(-0.75, 0.55)
bx.annotate("the bound holds only\ninside the clamp",
            xy=(3.05, -0.5), xytext=(1.30, -0.60), fontsize=6.0, color=S.GREY,
            ha="right", va="center", linespacing=1.25,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GREY_LIGHT,
                            shrinkA=2, shrinkB=2))
for a in (ax, bx):
    a.tick_params(length=2, width=0.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.savefig(S.out("fig131"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig131"), "written")
inside = np.abs(x) <= clip
print(f"  s_q = {s_q:.5f}, s_q/2 = {s_q/2:.5f}")
print(f"  max |error| inside the clamp {np.abs(err[inside]).max():.5f}"
      f"   bound respected: {np.abs(err[inside]).max() <= s_q/2 + 1e-12}")
print(f"  max |error| overall {np.abs(err).max():.4f}, at the far edge")
