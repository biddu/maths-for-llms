"""F-9.2 -- momentum is a low-pass filter, and beta1 sets the cutoff.

The EMA m_t = beta1 m_{t-1} + (1-beta1) g_t is a one-pole IIR filter with
transfer function H(z) = (1-beta1)/(1 - beta1 z^-1).  Its magnitude response
says what momentum actually does to the gradient signal: pass the slowly
varying part, attenuate the fast part.  The time constant is 1/(1-beta1),
which is the number of steps of gradient the estimate is averaging over.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt

w = np.logspace(-4, np.log10(np.pi), 900)
betas = [0.9, 0.95, 0.99]
dash = [S.DASHES[0], S.DASHES[1], S.DASHES[2]]
col = [S.ACCENT, S.GREY, S.ACCENT_MID]

fig, axes = plt.subplots(1, 2, figsize=(3.90, 2.00), constrained_layout=True)
ax, ax2 = axes
for b, dd, cc in zip(betas, dash, col):
    H = (1 - b) / np.abs(1 - b * np.exp(-1j * w))
    ax.loglog(w, H, color=cc, ls=dd, lw=1.05)
    wc = w[np.argmin(np.abs(H - 1 / np.sqrt(2)))]
    ax.plot([wc], [1 / np.sqrt(2)], marker="o", ms=2.6, mfc="white", mec=cc, mew=0.8)
    t = np.arange(0, 320)
    ax2.plot(t, 1 - b ** (t + 1), color=cc, ls=dd, lw=1.05)
    ax2.axvline(1 / (1 - b), lw=0.4, ls=(0, (1, 2)), color=cc)

ax.axhline(1 / np.sqrt(2), lw=0.5, ls=(0, (1, 2.2)), color=S.GREY_LIGHT, zorder=0)
ax.text(1.3e-4, 0.62, r"$-3$ dB", fontsize=6.0, color=S.GREY)
for b, cc in zip(betas, col):
    H = (1 - b) / np.abs(1 - b * np.exp(-1j * w))
    ax.text(w[-1] * 1.12, H[-1], r"$%g$" % b, fontsize=6.2, color=cc,
            va="center", ha="left")
ax.text(w[-1] * 1.12, 1.15, r"$\beta_1$", fontsize=6.2, color=S.INK,
        va="center", ha="left")
ax.set_xlim(1e-4, 12)
ax.set_xlabel(r"frequency $\omega$ (rad/step)", labelpad=1.5)
ax.set_ylabel(r"$|H(\omega)|$", labelpad=2)
ax.set_ylim(2e-3, 1.6)
ax.set_title("magnitude response", fontsize=6.8, pad=3, color=S.INK)

ax2.set_xlabel("steps", labelpad=1.5)
ax2.set_ylabel("step response", labelpad=2)
ax2.set_xlim(0, 320); ax2.set_ylim(0, 1.08)
ax2.set_title(r"$\tau = 1/(1-\beta_1)$ marked", fontsize=6.8, pad=3, color=S.INK)
for a in axes:
    a.tick_params(length=2, width=0.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

fig.savefig(S.out("fig92"))
print(S.out("fig92"), "written; tau =",
      ", ".join("%g -> %.0f steps" % (b, 1 / (1 - b)) for b in betas))
