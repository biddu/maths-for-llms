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
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
from cycler import cycler
from matplotlib import rcParams

# ---------------------------------------------------------------- figure fonts
#
# READ THIS BEFORE CHANGING IT.
#
# Every figure in this book shipped in the wrong typeface until 15 Aug 2026, and
# nothing said so.  This module asked for
#
#     "font.serif": ["Libertinus Serif", "DejaVu Serif"]
#
# which is a *preference list*.  Libertinus is installed for TeX but was never
# registered with fontconfig, so matplotlib could not see it, silently took the
# second entry, and set all 84 figure PDFs in DejaVu Serif with Computer Modern
# mathematics, facing a Libertinus body text.  It is obvious on the page once you
# know and invisible until then, which is the worst kind of defect.
#
# Two changes follow from that, and the second matters more than the first.
# The fonts are registered here from the TeX tree directly rather than hoped for,
# and a missing font is a hard error rather than a substitution.  A fallback
# nobody sees is worse than a build that stops.
#
# The road not taken, recorded so it is not rediscovered.  matplotlib's `pgf`
# backend routes every label through LuaLaTeX and gives type identical to the
# body text, including TeX's math spacing, which is better than what mathtext
# produces.  It was tried for this book and rejected: hatched artists are
# pathological there, because the backend emits one pattern per hatched path, so
# a figure carrying a dozen of them never finishes typesetting and prints no
# error while it fails.  Three of the first twenty figures hung that way.  A
# build that stops with no message is not one an author can run.  If it is ever
# revisited, the switch is three rcParams and the per-figure fix is to draw one
# hatched path instead of many, as F-12.2 now does for its own sake.

_FONT_DIRS = [
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/libertinus-fonts",
    "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/libertinus-fonts",
    os.path.expanduser("~/Library/TinyTeX/texmf-dist/fonts/opentype/public/libertinus-fonts"),
    os.path.expanduser("~/Library/Fonts"),
    "/Library/Fonts",
    "/usr/share/fonts/opentype/libertinus",       # Debian/Ubuntu fonts-libertinus
    "/usr/share/fonts/truetype/libertinus",
    "/usr/share/fonts/*/libertinus*",
]
_REQUIRED = ("Libertinus Serif", "Libertinus Sans", "Libertinus Mono")


def _register_libertinus():
    for pattern in _FONT_DIRS:
        for d in glob.glob(pattern):
            for path in glob.glob(os.path.join(d, "Libertinus*.otf")):
                try:
                    fm.fontManager.addfont(path)
                except Exception:                      # a face matplotlib cannot read
                    pass
    have = {f.name for f in fm.fontManager.ttflist}
    missing = [w for w in _REQUIRED if w not in have]
    if missing and os.environ.get("MFE_ALLOW_FONT_FALLBACK") != "1":
        raise SystemExit(
            "\nmfestyle: " + ", ".join(missing) + " not available to matplotlib.\n"
            "The figures must be set in the same face as the body text, and\n"
            "matplotlib will substitute DejaVu without saying so if allowed to.\n"
            "Install the libertinus-fonts package, or add the directory holding\n"
            "LibertinusSerif-Regular.otf to _FONT_DIRS in figs/mfestyle.py.\n"
            "MFE_ALLOW_FONT_FALLBACK=1 builds anyway; that output is a draft and\n"
            "must not go to a printer.\n"
        )
    return not missing


LIBERTINUS = _register_libertinus()

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
        # No fallback entry.  If Libertinus is missing the import above has
        # already stopped, so a silent substitution cannot happen here either.
        "font.serif": ["Libertinus Serif"],
        # Mathematics in the book's own face rather than Computer Modern.  Large
        # operators and a handful of relations are not in the text face and come
        # from STIX, which is a serif of the same colour; Computer Modern is not.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Libertinus Serif",
        "mathtext.it": "Libertinus Serif:italic",
        "mathtext.bf": "Libertinus Serif:bold",
        "mathtext.cal": "Libertinus Serif:italic",
        "mathtext.sf": "Libertinus Sans",
        "mathtext.tt": "Libertinus Mono",
        "mathtext.fallback": "stix",
        # Type 42 embeds the outlines.  matplotlib's default of Type 3 is a
        # bitmap-ish format some print vendors reject outright, and every figure
        # in the book carried it.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
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
        # THE DEFAULT CYCLE IS NOT NEUTRAL.  Any ax.plot() that forgets color=
        # takes matplotlib's cycle, which is #1f77b4 blue, #ff7f0e orange,
        # #2ca02c green.  In [print] that is ink the black-and-white interior is
        # not supposed to contain: a scan of book_final.pdf found 56 non-grey
        # colour operators across THIRTEEN print figures, every one of them a
        # forgotten color= on an axvline, a marker or a guide line.  Binding the
        # cycle to the book's own palette means a forgotten colour lands on the
        # palette instead of on matplotlib's.
        #
        # Colour only, deliberately.  Adding linestyle to the cycle would also
        # re-dash every line that sets a colour but not a style, which is most
        # of them, and change figures that are correct today.
        "axes.prop_cycle": cycler(color=SERIES),
        # Raster resolution, which matters for exactly five images in the book.
        # Everything here is vector except imshow, and imshow embeds a bitmap at
        # savefig.dpi.  The default is 100, and 100 ppi is what F-2.4's three
        # attention panels and F-3.6's map were carrying: KDP's previewer flags
        # anything under 300 and a printer interpolates the cell edges into mush.
        # 600 costs a few kilobytes on five images and nothing at all on the
        # other thirty-seven, which are vector and unaffected by this line.
        "savefig.dpi": 600,
    })


# \mfefig resolves figure files without a path, so they must land beside the
# chapter sources, one directory up from here.  This is an absolute path on
# purpose: it used to be a bare filename, which meant a generator wrote wherever
# it happened to be invoked from.  Run from the source root it wrote there; run
# from figs/ it wrote here; and buildall.sh then copied figs/*.pdf over the root,
# so a stale copy in figs/ silently replaced a figure that had just been rebuilt.
# Eight figures survived a full regeneration that way.  Nothing in the build now
# copies figures around, because nothing needs to.
_OUTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def out(stem):
    """Absolute path for this mode: fig33.pdf in colour, fig33_print.pdf in print."""
    return os.path.join(_OUTDIR, f"{stem}{SUFFIX}.pdf")
