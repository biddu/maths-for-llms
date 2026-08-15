"""F-8.2 -- the same model, three tokenizers: perplexity moves, bits/byte does not.

The bytes-per-token figures are a real measurement: the three tokenizers were
run over the committed held-out sample (measure/sample.txt, 33,192 bytes) and
the counts are in figs/data/fig82_tokenizers.json with the sample's SHA-256.

Everything else is exact arithmetic.  Fix one model's quality at Model D's
0.7707 bits/byte and ask what perplexity it would report under each tokenizer:
CE_bits/token = bpb x bytes/token, and PPL = 2^CE_bits.  The perplexities span
34% while the bits-per-byte are identical by construction, which is the whole
argument for reporting the second and not the first.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mfestyle as S
S.apply()
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data", "fig82_tokenizers.json")))
rows = D["rows"]
BPB = 2.03 / np.log(2) / 3.8                       # Model D's quality

names = [r["name"] for r in rows]
bpt = np.array([r["bytes_per_token"] for r in rows])
ce_bits = BPB * bpt
ppl = 2.0 ** ce_bits
bpb = ce_bits / bpt                                # identically BPB

x = np.arange(len(rows))
hatches = ["", "///", "\\\\\\"]
fig, axes = plt.subplots(1, 2, figsize=(3.90, 2.10), constrained_layout=True)

for ax, vals, title, fmt, lo, hi in (
        (axes[0], ppl, "perplexity", "%.2f", 0, 10.4),
        (axes[1], bpb, "bits per byte", "%.4f", 0, 1.06)):
    for i, v in enumerate(vals):
        ax.bar(x[i], v, width=0.62, color=S.ACCENT_PALE, edgecolor=S.ACCENT,
               linewidth=0.6, hatch=hatches[i], zorder=3)
        ax.text(x[i], v + (hi - lo) * 0.035, fmt % v, ha="center", fontsize=6.4,
                color=S.INK, zorder=4)
    ax.set_title(title, fontsize=6.9, pad=3, color=S.INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n%.2f B/tok" % (n, b) for n, b in zip(names, bpt)],
                       fontsize=6.0, linespacing=1.3)
    ax.set_ylim(lo, hi)
    ax.tick_params(length=2, width=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

axes[0].text(0.5, 9.6, "spread %.0f%%" % (100 * (ppl.max() / ppl.min() - 1)),
             fontsize=6.4, color=S.ACCENT, ha="center")
axes[1].text(1.0, 0.98, "identical", fontsize=6.4, color=S.ACCENT, ha="center")

fig.savefig(S.out("fig82"))
print(S.out("fig82"), "written;",
      " ".join("%s %.4f B/tok -> PPL %.3f" % (n, b, p)
               for n, b, p in zip(names, bpt, ppl)),
      "| bpb spread %.1e" % (bpb.max() - bpb.min()))
