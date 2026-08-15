"""F-3.5 — where the attention mass goes.
DRAFT NOTE: synthetic map with the sink structure of a trained model.  Replace
with cached maps from a pinned open checkpoint before print."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt
rng = np.random.default_rng(0); n = 128
z = rng.normal(0, 1.1, (n, n))
z += np.exp(-np.abs(np.subtract.outer(np.arange(n), np.arange(n)))/9.0)*3.2
z[:, 0] += 5.2
mask = np.triu(np.ones((n, n), bool), 1)
z[mask] = -np.inf
A = np.exp(z - z.max(1, keepdims=True)); A /= A.sum(1, keepdims=True)
share = A[:, 0].mean()
fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.6, 2.75), sharex=True,
                             gridspec_kw=dict(height_ratios=[3.2, 1], hspace=0.10))
ax.imshow(np.log10(A + 1e-6), cmap=S.CMAP, vmin=-4, vmax=0,
          interpolation="nearest", aspect="auto")
ax.set_ylabel("query position")
ax.tick_params(labelsize=6.2, length=2)
bx.bar(np.arange(n), A.sum(0), color=S.ACCENT, width=1.2)
bx.set_xlim(-0.5, n-0.5); bx.set_xlabel("key position")
bx.tick_params(labelsize=6.2, length=2)
bx.set_ylabel("column sum", fontsize=6.2)
for sp in ("top", "right"): bx.spines[sp].set_visible(False)
bx.annotate(f"position 0 takes {share*100:.0f}% of all attention mass",
            xy=(7, A.sum(0)[0]*0.66), fontsize=6.3, ha="left", color=S.ACCENT)
fig.savefig(S.out("fig35"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig35"), " | mean position-0 share = %.3f" % share)
