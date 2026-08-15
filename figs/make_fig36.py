"""F-3.6 — where the quadratic term overtakes the projections, Model D."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt
d, h, dh, nkv = 4096, 32, 128, 8
s = np.logspace(np.log10(256), np.log10(2**19), 400)
proj = 2*s*d*2*(h+nkv)*dh
attn = 4*s**2*d
sstar = (h+nkv)*dh
fig, ax = plt.subplots(figsize=(3.9, 2.2))
ax.loglog(s, proj/1e12, color=S.GREY, lw=1.1)
ax.loglog(s, attn/1e12, color=S.ACCENT, lw=1.3, ls=(0, (4, 2)))
ax.loglog(s, 4*s**2*d/2/1e12, color=S.ACCENT_MID, lw=0.9, ls=(0, (1, 1.5)))
ax.plot(sstar, 4*sstar**2*d/1e12, "o", mfc="white", mec=S.ACCENT, ms=4.6, mew=1.2, zorder=5)
ax.annotate(rf"$s^\star={sstar}$", xy=(sstar, 4*sstar**2*d/1e12),
            xytext=(sstar*1.25, 4*sstar**2*d/1e12/9), fontsize=6.5, color=S.ACCENT)
ax.annotate(r"projections, $\propto s$", xy=(3.2e5, proj[-1]/1e12),
            xytext=(2.0e4, 1.2), fontsize=6.4, color=S.GREY)
ax.annotate(r"scores $+$ $AV$, $\propto s^2$", xy=(2.0e5, 1e3),
            xytext=(1.1e3, 3e2), fontsize=6.4, color=S.ACCENT)
ax.annotate("causal", xy=(4.5e5, 4*(4.5e5)**2*d/2/1e12),
            xytext=(2.1e5, 8e1), fontsize=6.4, color=S.ACCENT_MID)
for x, lab in ((8192, "trained\ncontext"), (131072, "extended\ncontext")):
    ax.axvline(x, color=S.GREY_LIGHT, lw=0.5, ls=(0, (1, 2.4)))
    ax.annotate(lab, xy=(x, 2.4e-3), xytext=(x*0.92, 2.4e-3), fontsize=5.9,
                ha="right", va="bottom")
ax.set_xlabel(r"sequence length $s$"); ax.set_ylabel("TFLOP per layer")
ax.set_ylim(1e-3, 3e3)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(pad=0.2); fig.savefig(S.out("fig36"))
print(S.out("fig36"), " | s* =", sstar, "| at s=8192 ratio attn/proj = %.2f"
      % (4*8192**2*d/(2*8192*d*2*(h+nkv)*dh)))
