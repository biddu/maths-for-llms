"""F-5.3 — residual stream growth against the sqrt(l) idealisation.

The gap between the two curves is the honest part of D-5.3 step 6: the sqrt(l)
law assumes uncorrelated block outputs, which is false, and measured growth is
faster.  Presenting sqrt(l) as a prediction would be wrong; it is a lower bound.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt

L, d, n = 32, 512, 3000
rng = np.random.default_rng(3)

def run(corr):
    x = rng.normal(size=(n, d)) / np.sqrt(d)
    common = rng.normal(size=d) / np.sqrt(d)
    out = [np.linalg.norm(x, axis=1).mean()]
    for _ in range(L):
        w = rng.normal(size=(n, d)) / np.sqrt(d)
        x = x + (1 - corr) * w + corr * common          # correlated component
        out.append(np.linalg.norm(x, axis=1).mean())
    return np.array(out)

ideal = run(0.0); real = run(0.35)
l = np.arange(0, L + 1)

fig, ax = plt.subplots(figsize=(3.9, 1.95))
ax.loglog(l[1:], ideal[1:] / ideal[1], color=S.GREY, lw=1.0, ls=(0, (4, 2)))
ax.loglog(l[1:], real[1:] / real[1], color=S.ACCENT, lw=1.3)
ax.annotate(r"$\sqrt{l}$, the independence idealisation", xy=(9, ideal[9] / ideal[1]),
            xytext=(1.4, 4.6), fontsize=6.3, color=S.GREY)
ax.annotate("measured growth, faster", xy=(20, real[20] / real[1]),
            xytext=(2.6, 12.0), fontsize=6.3, color=S.ACCENT)
ax.set_xlabel("layer $l$"); ax.set_ylabel(r"$\|x_l\| \,/\, \|x_1\|$")
ax.set_xlim(1, L); ax.set_ylim(0.9, 30)
ax.tick_params(labelsize=6.2, length=2)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.savefig(S.out("fig53"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig53"), "| at l=32: idealisation x%.2f, correlated x%.2f"
      % (ideal[32] / ideal[1], real[32] / real[1]))
