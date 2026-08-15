"""F-4.2 — RoPE in one picture.  LOAD-BEARING (one of the book's five).

(a) In each 2-D block the query and key are rotated by m*theta_i and n*theta_i;
    only the difference survives the inner product.
(b) Eight of Model D's 64 blocks as clocks, showing the hand's travel over the
    8192-token trained context.
(c) Wavelength against block index, with the trained context and YaRN's band
    boundaries marked.

Panel (c)'s data comes from arith/model_d.py --rope-bands, so the figure and the
arithmetic box cannot drift apart.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["hatch.linewidth"] = 0.35
from matplotlib.patches import Circle, Arc

D_H, B_ROPE, L_TRAIN = 128, 500_000, 8192
ALPHA_Y, BETA_Y = 1.0, 32.0
i = np.arange(D_H // 2)
lam = 2 * np.pi * B_ROPE ** (2 * i / D_H)
r = L_TRAIN / lam
gam = np.clip((r - ALPHA_Y) / (BETA_Y - ALPHA_Y), 0, 1)
i_ramp0 = int(np.where(gam < 1 - 1e-12)[0].min())
i_star = int(np.where(gam <= 1e-12)[0].min())

fig = plt.figure(figsize=(3.95, 4.05))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.0, 1.32],
                      hspace=0.30, wspace=0.16)

# ---------------------------------------------------------------- panel (a)
ax = fig.add_subplot(gs[0, 0]); ax.set_aspect("equal")
phq, phk, mth, nth = 0.15, 1.78, 0.95, 1.12
ax.add_patch(Circle((0, 0), 1, fill=False, lw=0.7, ec=S.GREY_LIGHT))
def vec(a, style, col, lw, head):
    ax.annotate("", xy=(np.cos(a), np.sin(a)), xytext=(0, 0),
                arrowprops=dict(arrowstyle=head, lw=lw, color=col, ls=style,
                                mutation_scale=6.5))
vec(phq, "-", S.GREY, 0.9, "-|>")
vec(phk, (0, (3, 1.6)), S.GREY, 0.9, "-|>")
vec(phq + mth, "-", S.ACCENT, 1.6, "-|>")
vec(phk + nth, (0, (3, 1.6)), S.ACCENT, 1.6, "-|>")
for a, lab, col in ((phq, r"$q$", S.GREY), (phk, r"$k$", S.GREY),
                    (phq + mth, r"$R_m q$", S.ACCENT), (phk + nth, r"$R_n k$", S.ACCENT)):
    ax.annotate(lab, xy=(1.20 * np.cos(a), 1.20 * np.sin(a)), fontsize=6.0,
                ha="center", va="center", color=col)
ax.add_patch(Arc((0, 0), 1.30, 1.30, theta1=np.degrees(phq + mth),
                 theta2=np.degrees(phk + nth), lw=1.3, color=S.ACCENT))
ax.annotate("only this angle survives\nthe inner product:\n"
            r"$(\varphi_q-\varphi_k)+(m-n)\theta_i$",
            xy=(0.0, -1.20), fontsize=5.8, ha="center", va="top", color=S.ACCENT)
ax.set_xlim(-1.55, 1.55); ax.set_ylim(-2.30, 1.62)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title("(a) one 2-D block", fontsize=6.6, pad=3.0, color=S.ACCENT)

# ---------------------------------------------------------------- panel (b)
axb = fig.add_subplot(gs[0, 1]); axb.set_aspect("equal")
shown = [0, 8, 16, 24, 32, 40, 48, 56]
R, DX, DY = 0.38, 1.12, 1.62
for n, k in enumerate(shown):
    cx, cy = (n % 4) * DX, -(n // 4) * DY
    wraps = L_TRAIN / lam[k]
    axb.add_patch(Circle((cx, cy), R, fill=False, lw=0.6, ec=S.GREY_LIGHT))
    axb.plot([cx, cx], [cy, cy + R], lw=0.6, color=S.GREY)         # hand at m = 0
    frac = wraps % 1.0
    end = np.pi / 2 - 2 * np.pi * frac
    full = wraps >= 1.0
    if full:                                                        # completed turns
        axb.add_patch(Arc((cx, cy), 1.5 * R, 1.5 * R, theta1=0, theta2=360,
                          lw=0.5, color=S.GREY_LIGHT, ls=(0, (1, 1.2))))
    axb.add_patch(Arc((cx, cy), 1.5 * R, 1.5 * R, theta1=np.degrees(end),
                      theta2=90, lw=1.5 if not full else 1.0,
                      color=S.ACCENT if not full else S.ACCENT_MID))
    axb.annotate("", xy=(cx + R * np.cos(end), cy + R * np.sin(end)), xytext=(cx, cy),
                 arrowprops=dict(arrowstyle="-|>", lw=1.3, color=S.ACCENT,
                                 mutation_scale=5))
    axb.annotate(rf"$i={k}$", xy=(cx, cy + R + 0.16), fontsize=5.5, ha="center",
                 color=S.GREY)
    txt = f"{wraps:,.1f}" if wraps >= 1 else f"{wraps:.3f}"
    axb.annotate(txt, xy=(cx, cy - R - 0.13), fontsize=5.5, ha="center", va="top",
                 color=S.ACCENT if wraps < 1 else S.INK)
axb.annotate("turns completed over the trained context", xy=(1.68, -2.52),
             fontsize=5.7, ha="center", va="top", color=S.GREY)
axb.set_xlim(-0.60, 3.96); axb.set_ylim(-2.80, 0.80)
axb.set_xticks([]); axb.set_yticks([])
for sp in axb.spines.values(): sp.set_visible(False)
axb.set_title("(b) eight of Model D's 64 blocks", fontsize=6.6, pad=1.5, color=S.ACCENT)

# ---------------------------------------------------------------- panel (c)
axc = fig.add_subplot(gs[1, :])
bands = [(0, i_ramp0, r"extrapolate $(\gamma=1)$", "///"),
         (i_ramp0, i_star, "ramp", "|||"),
         (i_star, 64, r"interpolate $(\gamma=0)$", "\\\\\\")]
for lo, hi, lab, hatch in bands:
    axc.axvspan(lo - 0.5, hi - 0.5, facecolor="none", edgecolor="#D6DCE0",
                hatch=hatch, lw=0.0, zorder=0)
    axc.annotate(lab, xy=((lo + hi) / 2, 5.5e6), fontsize=5.7, ha="center",
                 va="top", color=S.GREY,
                 bbox=dict(fc="white", ec="none", pad=0.5))
axc.semilogy(i, lam, color=S.ACCENT, lw=1.3, zorder=3)
axc.axhline(L_TRAIN, color=S.INK, lw=0.7, ls=(0, (4, 2)), zorder=2)
axc.annotate(f"trained context {L_TRAIN}", xy=(1, L_TRAIN * 1.45), fontsize=5.9,
             ha="left", color=S.INK)
for x in (i_ramp0, i_star):
    axc.axvline(x - 0.5, color=S.GREY, lw=0.55, ls=(0, (1, 2)), zorder=2)
axc.plot([i_star], [lam[i_star]], "o", mfc="white", mec=S.ACCENT, ms=4.2, mew=1.1, zorder=4)
axc.annotate(rf"$i^\star={i_star}$", xy=(i_star, lam[i_star]),
             xytext=(i_star + 2.0, lam[i_star] / 22), fontsize=6.2, color=S.ACCENT)
axc.set_xlabel("block index $i$", fontsize=6.8)
axc.set_ylabel(r"wavelength $\lambda_i$ (tokens)", fontsize=6.8)
axc.set_xlim(-0.5, 63.5); axc.set_ylim(3, 9e6)
axc.tick_params(labelsize=6.2, length=2)
for sp in ("top", "right"): axc.spines[sp].set_visible(False)
axc.set_title(r"(c) the wavelength ladder, $b_{\mathrm{rope}}=500{,}000$",
              fontsize=6.6, pad=2.0, color=S.ACCENT)

fig.savefig(S.out("fig42"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig42"), f"| i* = {i_star} | ramp {i_ramp0}..{i_star-1}"
      f" ({i_star-i_ramp0} pairs) | lam_34={lam[34]:.1f} lam_35={lam[35]:.1f}")
