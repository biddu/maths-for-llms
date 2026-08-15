"""F-13.2 -- per-channel activation magnitude before and after a randomised
Hadamard rotation.  The chapter's set piece: chapter-defining, drawn to
load-bearing standard.

The activation is synthetic and calibrated, and the author note says so.  The
reason is worth stating rather than hiding: emergent activation outliers are a
*scale* phenomenon, first characterised at around seven billion parameters, and
the book's own byte-level toy does not have them.  Trained for six hundred steps
at d = 256 it reaches a per-channel max/median of 2.1 and has not one channel
above five times the median.  Drawing that would misrepresent the problem this
section exists to solve.

What is calibrated: a dozen channels at roughly twenty times the median, fixed
across tokens rather than resampled per token.  That last property is the one
that matters, and it is why per-channel scaling works where per-token does not.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mfestyle as S

S.apply()

d, n_tok, n_out, spike = 4096, 64, 12, 20.0
rng = np.random.default_rng(1301)

# a fixed set of outlier channels, the same for every token
gain = np.ones(d)
gain[rng.choice(d, n_out, replace=False)] = spike
X = rng.standard_normal((n_tok, d)) * gain


def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


Q = hadamard(d) * rng.choice([-1.0, 1.0], d)[:, None] / np.sqrt(d)
Y = X @ Q

before = np.abs(X).max(0)          # per-channel magnitude over the token block
after = np.abs(Y).max(0)
inc = lambda v: np.sqrt(d) * np.abs(v).max() / np.linalg.norm(v)

fig, axes = plt.subplots(1, 2, figsize=(3.95, 2.05), sharey=True,
                         constrained_layout=True)
for ax, (mag, title, col) in zip(axes, (
        (before, "before", S.GREY), (after, "after a randomised rotation", S.ACCENT))):
    ax.vlines(np.arange(d), 0, mag, color=col, lw=0.25, zorder=3)
    med = np.median(mag)
    ax.axhline(med, color=S.GREY_LIGHT, lw=0.7, ls=(0, (3, 2)), zorder=4)
    ax.set_title(title, fontsize=7.0, color=col, pad=3)
    ax.text(0.97, 0.93, f"max/median {mag.max()/med:.1f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.2, color=col)
    # incoherence is a property of a vector, so it is measured on one token
    # row rather than on the per-channel maxima
    row = X[0] if col is S.GREY else Y[0]
    ax.text(0.97, 0.80, f"incoherence {inc(row):.1f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.2,
            color=S.GREY)
    ax.set_xlim(0, d)
    ax.set_xticks([0, 2048, 4096])
    ax.set_xticklabels(["0", "2048", "4096"])
    ax.tick_params(length=2, width=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("channel magnitude")
axes[0].set_xlabel("channel")
axes[1].set_xlabel("channel, after mixing")
axes[0].annotate("median", xy=(300, np.median(before)),
                 xytext=(700, np.median(before) * 4.2), fontsize=6.0,
                 color=S.GREY, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GREY_LIGHT,
                                 shrinkA=1, shrinkB=1))
fig.savefig(S.out("fig132"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig132"), "written")
print(f"  max/median  {before.max()/np.median(before):8.2f} -> {after.max()/np.median(after):.2f}")
print(f"  incoherence {inc(X[0]):8.2f} -> {inc(Y[0]):.2f}"
      f"   (sqrt(2 ln d) = {np.sqrt(2*np.log(d)):.2f})")
print(f"  bits gained log2 of the ratio of maxima:"
      f" {np.log2(before.max()/after.max()):.2f}")
kurt = float(((Y - Y.mean()) ** 4).mean() / ((Y - Y.mean()) ** 2).mean() ** 2)
print(f"  kurtosis after rotation {kurt:.2f}  (a Gaussian is 3)")
print(f"  invariance ||xW - (xQ)(Q^T W)||_inf, fp64:", end=" ")
W = rng.standard_normal((d, 256)) / np.sqrt(d)
print(f"{np.abs(X @ W - (X @ Q) @ (Q.T @ W)).max():.2e}")
