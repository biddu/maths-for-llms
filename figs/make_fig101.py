"""F-10.1 -- isoFLOP curves, from the fitted law rather than traced from a paper.

At fixed training compute C the token budget is forced, D = C/6N, so the loss
is a function of N alone and it is U-shaped: too few parameters and the
parameter term dominates, too many and the data term does.  The locus of the
minima is the compute-optimal frontier, and D-10.2 is the closed form for it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt

L_INF, A, ALPHA, B, BETA = 1.82, 482.0, 0.348, 2085.4, 0.366
loss = lambda N, D: L_INF + A * N ** -ALPHA + B * D ** -BETA
k = (BETA * B / (ALPHA * A)) ** (1 / BETA)
Nstar = lambda C: (C / (6 * k)) ** (BETA / (ALPHA + BETA))

Cs = np.array([1e19, 1e20, 1e21, 1e22, 1e23, 1e24])
fig, ax = plt.subplots(figsize=(3.90, 2.10), constrained_layout=True)
dash = [S.DASHES[i % 4] for i in range(len(Cs))]
mins = []
for C, dd in zip(Cs, dash):
    Ns = np.logspace(np.log10(Nstar(C)) - 1.15, np.log10(Nstar(C)) + 1.15, 400)
    ax.semilogx(Ns, loss(Ns, C / (6 * Ns)), color=S.GREY, ls=dd, lw=0.8, zorder=2)
    n = Nstar(C); mins.append((n, loss(n, C / (6 * n))))
mins = np.array(mins)
ax.semilogx(mins[:, 0], mins[:, 1], color=S.ACCENT, ls=S.DASHES[0], lw=1.3, zorder=4)
ax.plot(mins[:, 0], mins[:, 1], ls="none", marker="o", ms=3.2, mfc="white",
        mec=S.ACCENT, mew=1.0, zorder=5)
for (n, l), C in zip(mins, Cs):
    # A white bbox keeps the curve from running through the exponent: the two
    # extreme curves are steep enough at this offset to cross their own label.
    ax.text(n * 1.9, l + 0.055, r"$10^{%d}$" % round(np.log10(C)), fontsize=6.0,
            color=S.GREY, ha="left", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
ax.text(3.2e10, 2.55, "compute-optimal\nfrontier", fontsize=6.4,
        color=S.ACCENT, linespacing=1.3, ha="center")
ax.text(3.5e7, 2.15, "each curve is one\nvalue of $C$ (FLOPs)", fontsize=6.2,
        color=S.GREY, linespacing=1.3, ha="left")
ax.set_xlabel("non-embedding parameters $N$", labelpad=1.5)
ax.set_ylabel("loss (nats/token)", labelpad=2)
ax.set_ylim(1.9, 3.4)
ax.tick_params(length=2, width=0.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.savefig(S.out("fig101"))
print(S.out("fig101"), "written; N* from %.2e to %.2e" % (mins[0, 0], mins[-1, 0]))
