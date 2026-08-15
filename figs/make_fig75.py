"""F-7.5 -- checkpoint memory M(m), with and without a fused attention kernel.

M(m) = m*M_b + (L/m)*M_act.  The unconstrained minimum is at
m* = sqrt(L * M_act / M_b), the square-root rule.  It is only reachable when
m* <= L; past that the minimum sits on the boundary and the answer is
"checkpoint every layer".  Model D at s = 8192 is on the wrong side of that
line unless attention is fused, which is the whole point of drawing both.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt

L, d, h, n_kv, d_h, d_ff, s, w = 32, 4096, 32, 8, 128, 14336, 8192, 2
M_b = s * d * w
M_probs = h * s * s * w
M_act = 6 * s * d * w + 2 * s * n_kv * d_h * w + M_probs + 3 * s * d_ff * w + 2 * s * 4
M_fused = M_act - M_probs

m = np.linspace(1, 48, 900)
fig, ax = plt.subplots(figsize=(3.90, 2.02), constrained_layout=True)
for M, colour, dash, lw, lab in ((M_act, S.GREY, S.DASHES[1], 0.95, "stored $P$"),
                                 (M_fused, S.ACCENT, S.DASHES[0], 1.15, "recomputed $P$")):
    y = (m * M_b + (L / m) * M) / 1e9
    ax.semilogy(m, y, color=colour, ls=dash, lw=lw)
    ms = math.sqrt(L * M / M_b)
    mc = min(ms, L)
    ax.plot([mc], [(mc * M_b + (L / mc) * M) / 1e9], marker="o", ms=3.2,
            mfc="white", mec=colour, mew=1.0, zorder=5)

ax.axvspan(L, 48, color=S.GREY_LIGHT, alpha=0.16, lw=0, zorder=0)
ax.axvline(L, lw=0.7, color=S.INK, ls=(0, (4, 2)), zorder=1)
ax.text(L + 0.9, 150, "$m > L$ is\nunreachable", fontsize=6.1, ha="left",
        va="top", color=S.GREY, linespacing=1.35)

ax.annotate(r"$m^\star = 50.9$: clipped to $m = L$,  $7.58$ GB", xy=(32, 7.58),
            xytext=(20.0, 27.0), fontsize=6.2, color=S.GREY, ha="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GREY, shrinkA=2, shrinkB=3))
ax.annotate(r"$m^\star = 23.3$,  $M^\star = 3.13$ GB", xy=(23.3, 3.13),
            xytext=(28.0, 2.35), fontsize=6.2, color=S.ACCENT, ha="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.ACCENT, shrinkA=2, shrinkB=3))
ax.text(3.0, 108, "stored $P$", fontsize=6.6, color=S.GREY)
ax.text(3.4, 17.5, "recomputed $P$", fontsize=6.6, color=S.ACCENT)

ax.set_xlabel("checkpointed segments $m$", labelpad=1.5)
ax.set_ylabel("activation memory (GB)", labelpad=2)
ax.set_xlim(1, 48); ax.set_ylim(2.0, 260)
ax.set_yticks([2, 5, 10, 20, 50, 100, 200])
ax.set_yticklabels(["2", "5", "10", "20", "50", "100", "200"])
ax.set_xticks([1, 8, 16, 24, 32, 40, 48])
ax.tick_params(length=2, width=0.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

fig.savefig(S.out("fig75"))
print(S.out("fig75"), "written; M_act/M_b = %.1f and %.1f; m* = %.1f and %.1f"
      % (M_act / M_b, M_fused / M_b, math.sqrt(L * M_act / M_b),
         math.sqrt(L * M_fused / M_b)))
