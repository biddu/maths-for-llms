"""F-12.3 -- load error against step for three controller gains.

The plotted quantity is the largest per-expert load error measured on a large
fixed set, not on the micro-batch the controller actually acts on.  That
distinction is the figure's second point: what the controller *sees* is a noisy
estimate whose sampling error here is several times the balance it is chasing,
so a plot of the micro-batch error shows nothing but the noise and the three
gains look identical.  Section 12.4 says why a global-batch statistic is not
optional.

Committed data: figs/data/moe_gains.npz, produced by work/moetoy.py.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mfestyle as S

S.apply()

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "data", "moe_gains.npz"))
err, gains, every = z["err"], z["gains"], int(z["every"])
steps = np.arange(err.shape[1]) * every

fig, ax = plt.subplots(figsize=(3.95, 2.30), constrained_layout=True)
def rolling(x, w=41):
    """A running median.  The raw trace is kept underneath at low weight: the
    step-to-step scatter is real and hiding it would misrepresent what a load
    counter actually shows."""
    pad = np.pad(x, (w // 2, w // 2), mode="edge")
    return np.array([np.median(pad[i:i + w]) for i in range(len(x))])


cols = [S.ACCENT, S.ACCENT_MID, S.GREY]
for i, (u, col) in enumerate(zip(gains, cols)):
    ax.semilogy(steps, err[i], ls="-", color=col, lw=0.35, alpha=0.35, zorder=2)
    ax.semilogy(steps, rolling(err[i]), ls=S.DASHES[i], color=col, lw=1.1, zorder=3)
    ax.text(steps[-1] * 1.02, rolling(err[i])[-1], f"$u = {u:g}$",
            fontsize=6.2, color=col, va="center", ha="left")

# the ripple the largest gain never leaves
tail = err[-1, -200:]
ax.fill_between([steps[-200], steps[-1]], tail.min(), tail.max(),
                color=S.GREY_LIGHT, alpha=0.25, lw=0, zorder=1)
ax.annotate(f"ripple, not convergence:\n{tail.min():.4f} to {tail.max():.4f}",
            xy=(steps[-190], tail.max()), xytext=(steps[-1] * 0.60, 0.050),
            fontsize=6.0, color=S.GREY, ha="center", va="center",
            linespacing=1.3,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GREY_LIGHT,
                            shrinkA=2, shrinkB=2))

ax.set_xlim(0, steps[-1] * 1.0)
ax.set_ylim(5e-4, 8e-2)
ax.set_xlabel("training step")
ax.set_ylabel(r"$\max_i |1/E - c_i|$")
ax.tick_params(length=2, width=0.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.savefig(S.out("fig123"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig123"), "written")
for i, u in enumerate(gains):
    t = err[i, -200:]
    print(f"  u = {u:<7g} settles at {t.mean():.5f}   ripple sd {t.std():.5f}"
          f"   half of the initial error reached at step "
          f"{steps[np.argmax(err[i] < err[i,0]/2)] if (err[i] < err[i,0]/2).any() else -1}")
