"""F-16.2 -- interference destroys separability as the live-feature count grows.

A matched-filter read of a superposed vector returns the true intensity plus a
sum of k-1 overlap terms.  At small k those terms are a nuisance; at large k
they are the signal.  The three panels are the same dictionary and the same
read, at k = 10, 50 and 200, and the only thing that changes is how many
features are live at once.  D-16.3's bound says the crossover sits near
k ~ 1/eps^2 = 100 at eps = 0.1, and the middle panel is where it starts.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mfestyle as S

S.apply()

rng = np.random.default_rng(1602)
D, M, EPS = 4096, 8192, 0.1
U = rng.standard_normal((M, D))
U /= np.linalg.norm(U, axis=1, keepdims=True)

KS = (10, 50, 200)
fig, axes = plt.subplots(1, 3, figsize=(3.95, 1.55), sharey=True,
                         constrained_layout=True)

for ax, k in zip(axes, KS):
    act, ina = [], []
    for _ in range(220):
        S_idx = rng.choice(M, k, replace=False)
        z = np.ones(k)
        x = z @ U[S_idx]
        r = U @ x
        mask = np.zeros(M, bool)
        mask[S_idx] = True
        act.append(r[mask])
        ina.append(r[~mask][rng.choice(M - k, k, replace=False)])
    a = np.concatenate(act)
    i = np.concatenate(ina)
    lo, hi = -1.4, 2.4
    bins = np.linspace(lo, hi, 46)
    ax.hist(i, bins=bins, density=True, histtype="stepfilled", color="white",
            edgecolor=S.GREY, lw=0.5, zorder=3)
    ax.hist(a, bins=bins, density=True, histtype="step",
            edgecolor=S.ACCENT, lw=0.7, hatch="///", zorder=4)
    ax.set_title(rf"$k = {k}$", fontsize=7.0, pad=2.5)
    ax.text(0.03, 0.94, f"false reads {100*np.mean(i > np.quantile(a, 0.05)):.1f}%",
            transform=ax.transAxes, fontsize=5.6, color=S.GREY, ha="left",
            va="top", zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    ax.set_xlim(lo, hi)
    ax.set_xticks([-1, 0, 1, 2])
    ax.tick_params(length=2.0, width=0.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_yticks([])
axes[0].set_ylabel("density")
axes[1].set_xlabel(r"matched-filter read $r_i = \langle x, u_i\rangle$")
axes[0].text(0.03, 0.72, "hatched: active\noutline: inactive",
             transform=axes[0].transAxes, fontsize=5.6, color=S.INK,
             ha="left", va="top", linespacing=1.3)

fig.savefig(S.out("fig162"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig162"), "written")
for k in KS:
    S_idx = rng.choice(M, k, replace=False)
    x = np.ones(k) @ U[S_idx]
    r = U @ x
    mask = np.zeros(M, bool)
    mask[S_idx] = True
    print(f"  k={k:>4}: active mean {r[mask].mean():.3f} sd {r[mask].std():.3f}"
          f"   inactive sd {r[~mask].std():.3f}"
          f"   separation {(r[mask].mean()-r[~mask].mean())/r[~mask].std():.2f} sd")
