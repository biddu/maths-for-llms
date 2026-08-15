"""F-3.3 — attention as Nadaraya-Watson kernel smoothing.  LOAD-BEARING.
Three claims in one picture: attention is a weighted average of the values; the
weights come from a kernel evaluated at the query; the kernel is a choice, and a
different choice gives a different answer.  Same keys, same values, two kernels.
Greyscale only.  Attention weight is encoded by LINE WIDTH, never by greyness."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

K   = np.array([[1.05,2.55],[2.10,3.05],[2.95,1.95],[1.40,1.15],[3.15,2.85],[2.60,0.85]])
ANG = np.array([   47.4,     123.9,      169.7,      19.0,      345.7,        4.7 ])
V   = np.c_[np.cos(np.radians(ANG)), np.sin(np.radians(ANG))]
q   = np.array([2.05, 2.00])
r   = np.linalg.norm(K - q, axis=1)
SIG, TRI, AS, OS = 0.45, 1.50, 0.52, 1.85

w_g = np.exp(-r**2/(2*SIG**2)); w_g /= w_g.sum()
w_t = np.clip(1 - r/TRI, 0, None); w_t /= w_t.sum()
outs = {"g": V.T @ w_g, "t": V.T @ w_t}

fig, axes = plt.subplots(1, 2, figsize=(3.95, 2.45))

def panel(ax, w, out, contours, title, sub):
    for rad, ls, lab in contours:
        ax.add_patch(Circle(q, rad, fill=False, ls=ls, lw=0.75, ec=S.GREY, zorder=0))
        th = np.radians(238)
        ax.annotate(lab, xy=q + rad*np.array([np.cos(th), np.sin(th)]),
                    fontsize=5.9, ha="center", va="center", zorder=6, color=S.GREY,
                    bbox=dict(fc="white", ec="none", pad=0.5))
    for kj, wj in zip(K, w):
        ax.plot([q[0], kj[0]], [q[1], kj[1]], color=S.ACCENT_MID,
                lw=0.2 + 5.4*wj, solid_capstyle="round", zorder=1)
    for kj, vj in zip(K, V):
        ax.plot(*kj, "o", mfc="white", mec=S.INK, ms=3.8, mew=0.75, zorder=3)
        ax.annotate("", xy=kj + vj*AS, xytext=kj,
                    arrowprops=dict(arrowstyle="-|>", lw=0.55, color=S.INK,
                                    shrinkA=2.8, shrinkB=0, mutation_scale=4.6), zorder=3)
    ax.plot(*q, "s", color=S.INK, ms=4.6, zorder=5)
    ax.annotate("", xy=q + out*OS, xytext=q,
                arrowprops=dict(arrowstyle="-|>", lw=2.3, color=S.ACCENT,
                                mutation_scale=9.0), zorder=7)
    ax.set_xlim(0.20, 3.95); ax.set_ylim(0.20, 3.95)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title, fontsize=6.9, pad=2.0, color=S.ACCENT)
    ax.annotate(sub, xy=(0.5,-0.015), xycoords="axes fraction",
                ha="center", va="top", fontsize=6.2, color=S.GREY)

gc = [(SIG*np.sqrt(-2*np.log(l)), ls, lab) for l, ls, lab in
      ((0.5,(0,(4,2)),r"$\kappa=0.5$"), (0.2,(0,(5,1.6,1,1.6)),r"$0.2$"),
       (0.05,(0,(1,1.5)),r"$0.05$"))]
panel(axes[0], w_g, outs["g"], gc,
      r"(a)  $\kappa(q,k)=\exp(\langle q,k\rangle/\sqrt{d_h})$",
      "every key contributes")
panel(axes[1], w_t, outs["t"], [(TRI,(0,(4,2)),r"support")],
      r"(b)  a compact-support kernel",
      "flatter inside its support")

axes[0].annotate(r"$k_j$", xy=K[2], xytext=K[2]+np.array([0.10,-0.34]), fontsize=6.4)
axes[0].annotate(r"$v_j$", xy=K[2]+V[2]*AS, xytext=K[2]+V[2]*AS+np.array([-0.05,0.14]),
                 fontsize=6.4, ha="center")
axes[0].annotate(r"$q_i$", xy=q, xytext=q+np.array([0.22,-0.14]), fontsize=6.8)
for ax, k, dx in zip(axes, ("g","t"), (-0.14, 0.14)):
    o = outs[k]
    ax.annotate(r"$\sum_j a_j v_j$", xy=q+o*OS,
                xytext=q+o*OS+np.array([dx, 0.30]),
                fontsize=6.3, ha="center", zorder=8, color=S.ACCENT,
                bbox=dict(fc="white", ec="none", pad=0.6))

fig.tight_layout(pad=0.20, w_pad=0.4)
fig.savefig(S.out("fig33"))
ang = lambda v: np.degrees(np.arctan2(v[1], v[0]))
print(S.out("fig33"), "| a_j (a):", np.round(w_g,3))
print("          | a_j (b):", np.round(w_t,3))
print("          | output directions differ by %.1f degrees"
      % abs(ang(outs['g']) - ang(outs['t'])))
