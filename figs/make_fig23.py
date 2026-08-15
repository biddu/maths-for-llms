"""F-2.3 — cos(theta) between two random directions, at d = 3, 64, 512, 4096.
One panel per d, common x axis, each with its own +/- 1/sqrt(d) marked.
Greyscale-legible: no colour, no fill, direct labels, no legend."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt
rng = np.random.default_rng(20260813)
n, chunk, dims = 120_000, 10_000, [3, 64, 512, 4096]
fig, axes = plt.subplots(len(dims), 1, figsize=(3.85, 2.9), sharex=True)
COLS = [S.GREY_LIGHT, S.GREY, S.ACCENT_MID, S.ACCENT]
for ax, d, col in zip(axes, dims, COLS):
    acc = np.zeros(500)
    for _ in range(n // chunk):
        x = rng.standard_normal((chunk, d), dtype=np.float32)
        y = rng.standard_normal((chunk, d), dtype=np.float32)
        c = np.einsum("ij,ij->i", x, y) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
        h, edges = np.histogram(c, bins=500, range=(-1, 1))
        acc += h
    dens = acc / (n * (2 / 500))
    ctr = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(ctr, dens, color=col, lw=1.0)
    s = 1 / np.sqrt(d)
    for xv in (-s, s):
        ax.plot([xv, xv], [0, dens.max() * 1.05], color=col, lw=0.55, ls=(0, (2, 1.6)))
    ax.set_xlim(-0.95, 0.95)
    ax.set_ylim(0, dens.max() * 1.75)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.text(0.005, 0.97, rf"$d={d}$", transform=ax.transAxes,
            fontsize=6.8, ha="left", va="top")
    ax.text(0.995, 0.97, rf"$1/\sqrt{{d}}={s:.4f}$", transform=ax.transAxes,
            fontsize=6.4, ha="right", va="top")
axes[-1].set_xlabel(r"$\cos\theta$ between two independent random directions")
fig.tight_layout(pad=0.2, h_pad=0.35)
fig.savefig(S.out("fig23"))
print(S.out("fig23"), " written")
