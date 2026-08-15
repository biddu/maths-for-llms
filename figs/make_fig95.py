"""F-9.5 -- the two knobs that bound an oversized Adam step.

Left: after a long quiet stretch, one dominant gradient produces an update of
exactly (1 - beta1)/sqrt(1 - beta2).  The markers are simulated Adam, the line
is that closed form, and they agree to four decimals.  At the default
beta2 = 0.999 one gradient can take a step 3.16 times the nominal one; at
beta2 = 0.95 the largest possible step is 0.45, so no overshoot exists.

Right: what epsilon does.  Below a gradient scale of about epsilon the
denominator is dominated by epsilon rather than by sqrt(v), and the update
stops being sign-like and becomes proportional.  Raising epsilon from 1e-8 to
1e-5 moves that crossover up by three decades, which is why epsilon and not the
learning rate is the knob for a coordinate whose gradients have gone quiet.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, "data", "fig95.npz"))

fig, axes = plt.subplots(1, 2, figsize=(3.90, 2.02), constrained_layout=True)
ax = axes[0]
x = 1 - D["b2"]
ax.loglog(x, D["closed_form"], color=S.ACCENT, ls=S.DASHES[0], lw=1.1, zorder=3)
ax.loglog(x, D["peak_b2"], ls="none", marker="o", ms=2.8, mfc="white",
          mec=S.GREY, mew=0.8, zorder=4)
ax.axhline(1.0, lw=0.5, ls=(0, (1, 2.2)), color=S.GREY_LIGHT, zorder=0)
for b2, lab in ((0.999, r"$\beta_2 = 0.999$"), (0.95, r"$\beta_2 = 0.95$")):
    v = 0.1 / np.sqrt(1 - b2)
    ax.plot([1 - b2], [v], marker="s", ms=3.0, mfc="white", mec=S.ACCENT, mew=1.0,
            zorder=5)
    ax.annotate("%s\n%.2f" % (lab, v), xy=(1 - b2, v),
                xytext=(1 - b2, v * (2.6 if b2 == 0.95 else 0.30)),
                fontsize=6.1, color=S.ACCENT, ha="center", linespacing=1.3)
ax.set_xlabel(r"$1 - \beta_2$", labelpad=1.5)
ax.set_ylabel("largest step, in units of $\\eta$", labelpad=2)
ax.set_title("one dominant gradient", fontsize=6.8, pad=3, color=S.INK)
ax.set_ylim(0.2, 9)

ax2 = axes[1]
for key, eps, cc, dd in (("peak_eps8", 1e-8, S.GREY, S.DASHES[1]),
                         ("peak_eps5", 1e-5, S.ACCENT, S.DASHES[0])):
    ax2.loglog(D["scale"], np.maximum(D[key], 1e-4), color=cc, ls=dd, lw=1.05)
    ax2.axvline(eps, lw=0.4, ls=(0, (1, 2.2)), color=cc, zorder=0)
ax2.text(3e-11, 0.16, r"$\varepsilon = 10^{-8}$", fontsize=6.1, color=S.GREY)
ax2.text(3e-9, 0.0011, r"$\varepsilon = 10^{-5}$", fontsize=6.1, color=S.ACCENT)
ax2.set_xlabel("gradient scale", labelpad=1.5)
ax2.set_title(r"where $\varepsilon$ takes over", fontsize=6.8, pad=3, color=S.INK)
ax2.set_ylim(3e-4, 6)

for a in axes:
    a.tick_params(length=2, width=0.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

fig.savefig(S.out("fig95"))
print(S.out("fig95"), "written; max |simulated - closed form| = %.2e"
      % np.abs(D["peak_b2"][:-1] - D["closed_form"][:-1]).max())
