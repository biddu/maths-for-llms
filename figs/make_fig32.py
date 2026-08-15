"""F-3.2 — softmax saturation: what the top logit gap does to the Jacobian."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt
rng = np.random.default_rng(3); n = 16
base = np.sort(rng.uniform(0, 1.0, n))[::-1]
def logits(D):
    z = base.copy(); z[0] = base[1] + D; return z
def sm(z): e = np.exp(z - z.max()); return e/e.sum()
def jnorm(z):
    p = sm(z); J = np.diag(p) - np.outer(p, p)
    return np.linalg.norm(J, 2)

fig = plt.figure(figsize=(3.9, 2.85))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.45], hspace=0.42, wspace=0.30)
for c, D in enumerate((1, 5, 15)):
    ax = fig.add_subplot(gs[0, c]); p = sm(logits(D))
    ml, sl, bl = ax.stem(np.arange(n), p, basefmt=" ")
    plt.setp(sl, color=S.ACCENT, lw=0.9); plt.setp(ml, color=S.ACCENT, ms=2.6)
    ax.set_ylim(0, 1.05); ax.set_xlim(-1, n)
    ax.set_title(rf"$\Delta={D}$", fontsize=6.6, pad=2, color=S.ACCENT)
    ax.set_xticks([]); ax.tick_params(length=2)
    if c: ax.set_yticklabels([])
    else: ax.set_ylabel(r"$p$", fontsize=6.8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

ax = fig.add_subplot(gs[1, :])
D = np.linspace(0, 20, 400)
emp = np.array([jnorm(logits(d)) for d in D])
ax.semilogy(D, emp, color=S.ACCENT, lw=1.2)
ax.semilogy(D, 4*(n-1)*np.exp(-D), color=S.GREY, lw=0.9, ls=(0, (4, 2)))
ax.annotate("empirical " + r"$\|J\|_2$", xy=(9.0, emp[180]),
            xytext=(4.6, emp[180]*0.012), fontsize=6.4, ha="left", color=S.ACCENT)
ax.annotate(r"bound $4(n-1)e^{-\Delta}$", xy=(15.5, 4*(n-1)*np.exp(-15.5)),
            xytext=(14.4, 4*(n-1)*np.exp(-15.5)*7), fontsize=6.4, ha="left", color=S.GREY)
ax.set_xlabel(r"top-two logit gap $\Delta$"); ax.set_xlim(0, 20)
ax.set_ylim(1e-9, 2e2); ax.set_yticks([1e-8, 1e-5, 1e-2, 1e1])
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.savefig(S.out("fig32"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig32"), " | ||J||_2 at D=0,5,15:", np.round([jnorm(logits(d)) for d in (0,5,15)], 6))
