"""F-5.1 — what centring is worth: |mean|/RMS of the residual stream by layer.

DRAFT NOTE: synthetic, with the distributional shape a trained checkpoint shows.
The real measurement comes from ch05_norm_residual.ipynb §1 on the hash-pinned
checkpoint; the print edition commits to the order of magnitude, not decimals.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)
L, n = 32, 4000
layers = np.arange(1, L + 1)
# median drifts gently upward with depth; heavy right tail, as measured stacks show
med = 0.009 * (1 + 0.30 * layers / L)
data = [np.exp(rng.normal(np.log(m), 0.55, n)) for m in med]

fig, ax = plt.subplots(figsize=(3.9, 2.05))
bp = ax.boxplot(data, positions=layers, widths=0.62, whis=(1, 99),
                showfliers=False, patch_artist=False)
for part, col, lw in (("boxes", S.ACCENT, 0.6), ("whiskers", S.GREY_LIGHT, 0.5),
                      ("caps", S.GREY_LIGHT, 0.5), ("medians", S.ACCENT, 1.5)):
    plt.setp(bp[part], color=col, linewidth=lw)
ax.axhline(1e-2, color=S.INK, lw=0.7, ls=(0, (4, 2)))
ax.annotate(r"$10^{-2}$", xy=(30.5, 1.15e-2), fontsize=6.2, va="bottom", ha="right", color=S.INK)
ax.set_yscale("log")
ax.set_xlabel("layer"); ax.set_ylabel(r"$|\mathrm{mean}(x)| \,/\, \mathrm{RMS}(x)$")
ax.set_xticks([1, 8, 16, 24, 32]); ax.set_xticklabels([1, 8, 16, 24, 32])
ax.set_xlim(0, L + 1); ax.set_ylim(6e-4, 2e-1)
ax.tick_params(labelsize=6.2, length=2)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.savefig(S.out("fig51"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig51"), "| median rho at l=1,16,32 = %.4f %.4f %.4f"
      % tuple(np.median(data[k]) for k in (0, 15, 31)))
