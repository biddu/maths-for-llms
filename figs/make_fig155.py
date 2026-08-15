"""F-15.5 -- why a token-level importance ratio explodes under routing drift.

Routing in an MoE layer is a discrete top-k selection (Chapter 12), so an
arbitrarily small parameter change can flip which expert a token is sent to, and
the token's probability under the new policy jumps discontinuously.  A clipped
token-level ratio then discards that token's gradient entirely.

Upper: per-token log rho_t over one 512-token completion, with a handful of
flips.  Lower: the same trace under GSPO's sequence-level ratio, which is the
geometric mean, so a token whose log-ratio moves by Delta shifts the sequence
log-ratio by Delta/|y|.  The clip band is drawn on both, to the same scale, and
that is the whole figure: the spikes leave the band and the mean never does.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mfestyle as S

S.apply()

rng = np.random.default_rng(1505)
L = 512
EPS = 0.2
band = np.log(1 + EPS)                     # the clip band, in log-ratio units

# ordinary token-level drift, plus five routing flips
lr = rng.standard_normal(L) * 0.02
flips = np.array([48, 143, 251, 342, 447])      # spread across the completion
lr[flips] = rng.choice([-1.0, 1.0], 5) * rng.uniform(2.5, 4.5, 5)
seq = np.cumsum(lr) / np.arange(1, L + 1)   # the running geometric mean

fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.95, 2.55), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1.0]},
                             constrained_layout=True)

for a in (ax, bx):
    a.axhspan(-band, band, color=S.GREY_LIGHT, alpha=0.28, zorder=1, lw=0)
    a.axhline(0, color=S.GREY_LIGHT, lw=0.5, zorder=2)
    a.tick_params(length=2.2, width=0.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

ax.vlines(np.arange(L), 0, lr, color=S.GREY, lw=0.35, zorder=3)
ax.vlines(flips, 0, lr[flips], color=S.ACCENT, lw=0.9, zorder=5)
ax.set_ylabel(r"$\log \rho_t$")
ax.set_ylim(-5.4, 5.4)
ax.set_yticks([-4, 0, 4])
ax.text(0.985, 0.965, "token level: five routing flips leave the clip band",
        transform=ax.transAxes, fontsize=5.9, color=S.INK, ha="right",
        va="top")
ax.annotate(r"$\Delta \approx 4$ nats", xy=(flips[0], lr[flips[0]]),
            xytext=(flips[0] + 26, np.sign(lr[flips[0]]) * 2.1),
            fontsize=5.8, color=S.ACCENT, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GREY_LIGHT,
                            shrinkA=1.5, shrinkB=2.5))

bx.plot(np.arange(1, L + 1), seq, color=S.ACCENT, lw=1.0, zorder=4)
bx.set_ylabel(r"$|y|^{-1}\sum_t \log \rho_t$")
bx.set_xlabel("token index")
bx.set_xlim(0, L)
bx.set_ylim(-0.34, 0.34)
bx.set_yticks([-0.2, 0, 0.2])
bx.text(0.985, 0.93, "sequence level: the same trace, never leaves it",
        transform=bx.transAxes, fontsize=5.9, color=S.INK, ha="right",
        va="top")
bx.text(0.015, 0.06, r"shaded: the clip band at $\varepsilon=0.2$",
        transform=bx.transAxes, fontsize=5.6, color=S.GREY, ha="left",
        va="bottom")

fig.savefig(S.out("fig155"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig155"), "written")
print(f"  clip band in log-ratio units: +/- {band:.6f}")
print(f"  tokens outside the band, token level   : {(np.abs(lr) > band).sum()}")
print(f"  final sequence log-ratio               : {seq[-1]:+.6f}")
print(f"  largest |sequence log-ratio| over the trace: {np.abs(seq).max():.6f}"
      f"   (band {band:.4f})")
print(f"  a single flip of Delta = 4 moves the sequence log-ratio by "
      f"{4.0/L:.6f}")
