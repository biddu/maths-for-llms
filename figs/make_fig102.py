"""F-10.3 -- why twenty tokens per parameter is not a law.

D-10.2 step 8: D*/N* is proportional to C^((alpha-beta)/(alpha+beta)).  The
exponent is zero, and only then is the ratio a constant, when alpha = beta
exactly.  Three exponent pairs, all fitted to the same phenomenon, give three
qualitatively different answers: flat, gently falling, and steeply rising.

The value of the ratio is set by A and B; alpha = beta only buys you the
property that it does not move.  That is the whole of the chapter's punchline.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt

FITS = [("$\\alpha = \\beta$ exactly", 1.82, 482.0, 0.348, 2085.4, 0.348,
         S.GREY, S.DASHES[2]),
        ("2024 refit", 1.82, 482.0, 0.348, 2085.4, 0.366, S.ACCENT, S.DASHES[0]),
        ("Chinchilla, as published", 1.70, 406.4, 0.34, 410.7, 0.28,
         S.GREY, S.DASHES[1])]

C = np.logspace(19, 26, 400)
fig, ax = plt.subplots(figsize=(3.90, 2.12), constrained_layout=True)
for name, Li, A, a, B, b, col, dd in FITS:
    k = (b * B / (a * A)) ** (1 / b)
    N = (C / (6 * k)) ** (b / (a + b))
    D = C / (6 * N)
    r = D / N
    ref = r[np.argmin(np.abs(C - 1e21))]              # normalise at C = 1e21
    ax.loglog(C, r / ref, color=col, ls=dd, lw=1.25 if col == S.ACCENT else 0.95)
    ax.text(C[-1] * 1.35, (r / ref)[-1], name, fontsize=6.2, color=col,
            va="center", ha="left")

ax.axhline(1.0, lw=0.5, ls=(0, (1, 2.2)), color=S.GREY_LIGHT, zorder=0)
ax.set_xlabel("training compute $C$ (FLOPs)", labelpad=1.5)
ax.set_ylabel("$D^\\star/N^\\star$, relative to $C = 10^{21}$", labelpad=2)
ax.set_xlim(1e19, 1e26); ax.set_ylim(0.55, 3.6)
ax.set_yticks([0.6, 1, 2, 3])
ax.set_yticklabels(["0.6", "1", "2", "3"])
ax.minorticks_off()
ax.text(1.3e19, 3.2, "at $C = 10^{21}$ the three sit at\n$67$, $21.6$ and $50$ tokens/param",
        fontsize=6.1, color=S.INK, linespacing=1.35)
ax.tick_params(length=2, width=0.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.savefig(S.out("fig102"))
for name, Li, A, a, B, b, _, _ in FITS:
    print("  %-26s exponent in C %+.4f" % (name, (a - b) / (a + b)))
print(S.out("fig102"), "written")
