"""F-10.4 -- the optimum moves left as you serve more.

Total lifetime compute, 6ND to train plus 2N per generated token to serve, at
a fixed target loss.  With no serving the minimum is D-10.2's.  As the served
token count rises the curve tilts, the minimum slides to smaller N and larger
D, and it also flattens: past a point the choice stops mattering much, which is
why shipped models cluster over a wide range of sizes rather than at a point.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt

L_INF, A, ALPHA, B, BETA = 1.82, 482.0, 0.348, 2085.4, 0.366
TARGET = 2.03227                                     # Model D's fitted loss

def D_at(N):
    r = TARGET - L_INF - A * N ** -ALPHA
    return np.where(r > 0, (B / np.maximum(r, 1e-12)) ** (1 / BETA), np.inf)

N = np.logspace(9.68, 11.45, 900)
D = D_at(N)
fig, ax = plt.subplots(figsize=(3.90, 2.10), constrained_layout=True)
cases = [(0.0, "no serving"), (1e12, "$10^{12}$"), (1e13, "$10^{13}$"),
         (1e14, "$10^{14}$")]
for (Dinf, lab), dd, col in zip(cases,
                                [S.DASHES[0], S.DASHES[1], S.DASHES[2], S.DASHES[3]],
                                [S.ACCENT, S.GREY, S.ACCENT_MID, S.GREY_LIGHT]):
    tot = 6 * N * D + 2 * N * Dinf
    ok = np.isfinite(tot)
    ax.loglog(N[ok], tot[ok], color=col, ls=dd, lw=1.15)
    i = np.nanargmin(np.where(ok, tot, np.inf))
    ax.plot([N[i]], [tot[i]], marker="o", ms=3.0, mfc="white", mec=col, mew=1.0,
            zorder=5)
    ax.text(N[ok][-1] * 1.06, tot[ok][-1], lab, fontsize=6.1, color=col,
            va="center", ha="left")

ax.text(1.15e10, 3.0e25, "the minimum slides left\nas $D_{\\mathrm{inf}}$ rises",
        fontsize=6.2, color=S.INK, linespacing=1.3, ha="left")
ax.set_xlabel("non-embedding parameters $N$", labelpad=1.5)
ax.set_ylabel("lifetime FLOPs", labelpad=2)
ax.set_xlim(4.6e9, 4.5e11)
ax.tick_params(length=2, width=0.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.savefig(S.out("fig104"))
print(S.out("fig104"), "written")
