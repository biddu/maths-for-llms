"""F-9.3 -- cosine commits to a budget; warmup-stable-decay does not.

Three cosine runs at three token budgets are three different curves from step
zero: the schedule encodes the budget, so a run cannot be extended and an early
checkpoint is not a finished model.  Warmup-stable-decay shares one trunk and
branches, so the same run yields three finished models and can be continued.
That branch property, not the shape, is the argument.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt

PEAK, WARM, TOTAL = 1.0, 0.02, 1.0
budgets = [0.45, 0.7, 1.0]


def cosine(t, T, warm=WARM):
    lr = np.where(t < warm, t / warm, 0.0)
    cos = 0.5 * (1 + np.cos(np.pi * np.clip((t - warm) / (T - warm), 0, 1)))
    return PEAK * np.where(t < warm, lr, 0.1 + 0.9 * cos)


def wsd(t, T, warm=WARM, tail=0.10):
    d0 = T - tail * T
    out = np.where(t < warm, t / warm, 1.0)
    dec = np.clip((t - d0) / (tail * T), 0, 1)
    return PEAK * np.where(t > d0, 1.0 - 0.9 * dec, out)


t = np.linspace(0, 1.0, 2400)
fig, axes = plt.subplots(1, 2, figsize=(3.90, 1.98), sharey=True,
                         constrained_layout=True)
for T, dd in zip(budgets, (S.DASHES[1], S.DASHES[2], S.DASHES[0])):
    m = t <= T
    axes[0].plot(t[m], cosine(t[m], T), color=S.GREY, ls=dd, lw=0.95)
    axes[0].text(T, cosine(np.array([T]), T)[0] - 0.06, "%.0f%%" % (100 * T),
                 fontsize=6.0, color=S.GREY, ha="right", va="top")

trunk = t <= 1.0
axes[1].plot(t[trunk], wsd(t[trunk], 10.0), color=S.ACCENT, ls=S.DASHES[0], lw=1.35)
for T in budgets:
    tb = np.linspace(T - 0.10 * T, T, 200)
    axes[1].plot(tb, wsd(tb, T), color=S.GREY, ls=S.DASHES[1], lw=0.9)
    axes[1].plot([T], [0.1], marker="o", ms=2.8, mfc="white", mec=S.GREY, mew=0.8)
    axes[1].text(T, 0.02, "%.0f%%" % (100 * T), fontsize=6.0, color=S.GREY,
                 ha="center", va="bottom")

axes[0].set_title("warmup and cosine, three budgets", fontsize=6.6, pad=3, color=S.INK)
axes[1].set_title("warmup, stable, decay: one trunk", fontsize=6.6, pad=3, color=S.INK)
axes[0].text(0.06, 0.16, "three curves\nfrom step zero", fontsize=6.2,
             color=S.GREY, linespacing=1.3)
axes[1].text(0.06, 0.16, "one shared trunk,\nthree branches", fontsize=6.2,
             color=S.ACCENT, linespacing=1.3)
for a in axes:
    a.set_xlabel("fraction of the longest budget", labelpad=1.5)
    a.set_xlim(0, 1.04); a.set_ylim(0, 1.12)
    a.set_xticks([0, 0.5, 1.0]); a.set_xticklabels(["0", "50%", "100%"])
    a.tick_params(length=2, width=0.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
axes[0].set_ylabel("learning rate / peak", labelpad=2)

fig.savefig(S.out("fig93"))
print(S.out("fig93"), "written")
