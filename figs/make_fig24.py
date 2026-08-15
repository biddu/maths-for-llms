"""F-2.4 -- what mean-centring removes, and what survives it.

Three treatments of the same 40 embedding rows: raw, mean-centred, and centred
with the top principal direction of the centred matrix also projected out.

DRAFT NOTE.  The cloud is synthetic.  It is not arbitrary: the three constants
below were fitted so that the three mean off-diagonal cosines reproduce the
values measured on a real checkpoint (SmolLM2-135M, 40 rows sampled from the
frequent end of the vocabulary) by measure/checkpoint_stats.py --what embeddings:

    raw                                   +0.245
    mean-centred                          +0.124
    mean and top-1 direction removed      +0.003

Replace with the hash-pinned real matrix before print; the caption's numbers
should not move when that lands, which is the point of fitting them here.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt

d, V, m, HEAD, SEED = 256, 4000, 40, 1000, 7
SPREAD, GROUP, NOISE = 9.0, 0.62, 0.42
C_MEAN, A_TOP, A_SECOND = 3.010021, 8.712863, 2.377363   # fitted; see docstring

rng = np.random.default_rng(SEED)
B = rng.standard_normal((3, d))
mu = B[0] / np.linalg.norm(B[0])
u1 = B[1] - (B[1] @ mu) * mu;               u1 /= np.linalg.norm(u1)
u2 = B[2] - (B[2] @ mu) * mu - (B[2] @ u1) * u1;  u2 /= np.linalg.norm(u2)

a = SPREAD * rng.standard_normal(V); a[:HEAD] += A_TOP
b = 1.0 * rng.standard_normal(V);    b[:HEAD] += A_SECOND
Gc = rng.standard_normal((4, d)); Gc -= Gc.mean(0, keepdims=True); Gc *= GROUP
grp = rng.integers(0, 4, size=V)
W = (C_MEAN * mu + a[:, None] * u1 + b[:, None] * u2
     + Gc[grp] + NOISE * rng.standard_normal((V, d)))
# ten rows per group, kept contiguous, so the four blocks sit on the diagonal
idx = np.concatenate([np.sort(rng.choice(np.where(grp[:HEAD] == k)[0], m // 4,
                                         replace=False)) for k in range(4)])

Wc = W - W.mean(0, keepdims=True)
_, _, Vt = np.linalg.svd(Wc, full_matrices=False)
Wa = Wc - (Wc @ Vt[:1].T) @ Vt[:1]


def cosine(M):
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    return Mn @ Mn.T


panels = [("raw", cosine(W[idx])),
          ("mean removed", cosine(Wc[idx])),
          ("mean and top-1 removed", cosine(Wa[idx]))]
off = ~np.eye(m, dtype=bool)

fig, axes = plt.subplots(1, 3, figsize=(3.90, 1.72), constrained_layout=True)
for ax, (title, C) in zip(axes, panels):
    im = ax.imshow(C, cmap=S.CMAP, vmin=-0.4, vmax=1.0, interpolation="nearest")
    ax.set_title(title, fontsize=6.0, pad=2.5, color=S.ACCENT)
    ax.set_xlabel(r"mean off-diagonal $%+.3f$" % C[off].mean(),
                  fontsize=6.0, labelpad=2.0, color=S.INK)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color(S.GREY)

cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.015)
cb.ax.tick_params(labelsize=6.0, width=0.5, length=2)
cb.outline.set_linewidth(0.5)

fig.savefig(S.out("fig24"))
print(S.out("fig24"), "written;",
      "  ".join("%s %+.4f" % (t, C[off].mean()) for t, C in panels))
