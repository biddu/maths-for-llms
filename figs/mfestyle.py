"""Shared figure style for Mathematics for Everything, Book 3.

One accent, greys elsewhere.  The accent marks the load-bearing element in a
figure; everything else is grey.  Information is always carried redundantly, by
line style, marker shape, hatching and direct labelling, so the same script
produces a colour figure for the digital editions and a black-ink figure for the
KDP paperback interior without losing anything.

    MFE_MODE=colour  python figs/make_fig33.py     -> fig33.pdf      (default)
    MFE_MODE=print   python figs/make_fig33.py     -> fig33_print.pdf

The print interior stays black ink at $0.012 per page.  A standard-colour
interior is $0.0255 and premium colour $0.065, which at 270 pages is the
difference between a $12.55 and a $8.91 paperback royalty, or a negative one.
That is why this module exists.
"""
import os
import matplotlib
matplotlib.use("Agg")
from matplotlib import rcParams

MODE = os.environ.get("MFE_MODE", "colour").lower()
PRINT = MODE in ("print", "bw", "grey", "gray")

if PRINT:
    ACCENT     = "#333333"
    ACCENT_MID = "#6B6B6B"
    ACCENT_PALE= "#E8E8E8"
    INK        = "#000000"
    GREY       = "#555555"
    GREY_LIGHT = "#9A9A9A"
    CMAP       = "gray_r"
    SUFFIX     = "_print"
else:
    ACCENT     = "#0B3C5D"      # deep blue: the load-bearing element
    ACCENT_MID = "#2E7BA6"      # same hue, lighter: secondary emphasis
    ACCENT_PALE= "#CFE2ED"
    INK        = "#1A1A1A"
    GREY       = "#6E7B84"
    GREY_LIGHT = "#AAB4BB"
    CMAP       = "Blues"
    SUFFIX     = ""

# Series order for figures that genuinely need more than two lines.  The accent
# always comes first, so the element the caption is about is the coloured one.
SERIES = [ACCENT, GREY, ACCENT_MID, GREY_LIGHT]
# Line styles run in parallel with the colours and carry the same distinction on
# their own.  Never rely on colour alone.
DASHES = ["-", (0, (4, 2)), (0, (5, 1.6, 1, 1.6)), (0, (1, 1.5))]
MARKERS = ["o", "s", "^", "D"]


def apply():
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Libertinus Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 7.2,
        "axes.linewidth": 0.6,
        "axes.labelsize": 7.0,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def out(stem):
    """Filename for this mode: fig33.pdf in colour, fig33_print.pdf in print."""
    return f"{stem}{SUFFIX}.pdf"
