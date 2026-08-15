"""F-4.3 — ALiBi against RoPE: what each does to the logit as distance grows."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt

D_H, B_ROPE = 128, 500_000
i = np.arange(D_H // 2)
theta = B_ROPE ** (-2 * i / D_H)
rng = np.random.default_rng(4)
q = rng.normal(size=D_H // 2) * 0.5
k = rng.normal(size=D_H // 2) * 0.5
a = rng.normal(size=D_H // 2) * 0.5
b = rng.normal(size=D_H // 2) * 0.5

dist = np.arange(0, 4096)
# RoPE: per-block cos/sin combination, summed over blocks
rope = ((q * k + a * b)[None, :] * np.cos(np.outer(dist, theta))
        + (q * b - a * k)[None, :] * np.sin(np.outer(dist, theta))).sum(1)
alibi = -0.25 * dist                       # head with slope 2^{-8h/n} = 1/4

fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.9, 2.35), sharex=True,
                             gridspec_kw=dict(hspace=0.16))
ax.plot(dist, rope, color=S.ACCENT, lw=0.7)
ax.axhline(0, color=S.GREY_LIGHT, lw=0.5)
ax.set_ylabel("RoPE", fontsize=6.6, color=S.ACCENT)
ax.set_ylim(-9, 15)
ax.annotate("bounded oscillation, no decay: the logit at distance 4000\n"
            "is drawn from the same range as the logit at distance 40",
            xy=(150, 14.0), fontsize=6.0, color=S.ACCENT, va="top")
bx.plot(dist, alibi, color=S.GREY, lw=1.2, ls=(0, (4, 2)))
bx.set_ylabel("ALiBi", fontsize=6.6, color=S.GREY)
bx.set_ylim(-1100, 60)
bx.annotate(r"$-m_h\,|n-m|$ with $m_h=1/4$: monotone, and by distance 4000 it"
            "\nswamps any logit the content could produce",
            xy=(150, -640), fontsize=6.0, color=S.GREY, va="top")
bx.set_xlabel(r"key--query distance $|n-m|$")
bx.set_xlim(0, 4096)
for a in (ax, bx):
    a.tick_params(labelsize=6.2, length=2)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.savefig(S.out("fig43"), bbox_inches="tight", pad_inches=0.02)
print(S.out("fig43"), "| RoPE logit range %.2f to %.2f over 4096 tokens"
      % (rope.min(), rope.max()))
