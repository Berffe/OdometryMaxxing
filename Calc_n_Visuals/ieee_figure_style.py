"""Matplotlib settings for ECC / IEEE two-column figures.

Import and call `use_ieee_style()` before plotting, then size every figure
with `column_figure()`. The point is that the exported PDF is already at its
final printed size, so `\\includegraphics[width=\\columnwidth]` scales by 1.0
and the fonts land where you set them.

Measure the real column width first: put `\\showthe\\columnwidth` in the
preamble, compile, and divide the printed value by 72.27.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

COLUMN_WIDTH_IN = 3.5   # measure with \showthe\columnwidth, then correct this
BASE_FONT_PT = 8


def use_ieee_style() -> None:
    mpl.rcParams.update({
        # Type 3 fonts are a common PDF eXpress rejection. 42 = TrueType.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # IEEE body text is Times. STIX is the metric-compatible math font,
        # so inline symbols in labels match the surrounding text.
        # STIXGeneral ships with matplotlib and is Times-metric, so the text
        # matches the body font even where Times itself is not installed.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",

        "font.size": BASE_FONT_PT,
        "axes.labelsize": BASE_FONT_PT,
        "axes.titlesize": BASE_FONT_PT,
        "xtick.labelsize": BASE_FONT_PT - 1,
        "ytick.labelsize": BASE_FONT_PT - 1,
        "legend.fontsize": BASE_FONT_PT - 1,

        # Thin lines read better at this size; default 1.5 looks clumsy.
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.linewidth": 0.4,

        "figure.dpi": 150,
        "savefig.dpi": 300,

        # NOT "tight". A tight bounding box trims the canvas below the
        # requested figsize, and \includegraphics[width=\columnwidth] then
        # scales the figure back UP, enlarging every font by the same factor.
        # Keep the canvas fixed and let tight_layout fit the axes inside it.
        "savefig.bbox": "standard",
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,

        # Transparency sometimes trips PDF eXpress. Use solid fills with a
        # light colour rather than alpha where it matters.
        "savefig.transparent": False,
    })


def column_figure(height_ratio: float = 0.68):
    """One figure at exactly one column wide. Do not scale it afterwards."""
    return plt.subplots(figsize=(COLUMN_WIDTH_IN,
                                 COLUMN_WIDTH_IN * height_ratio))


def save(fig, path: str) -> None:
    """Vector PDF. Check the result at 100% zoom, not fitted to the window."""
    fig.savefig(path, format="pdf")


# --- Feasibility region: greyscale-safe element choices ---------------------
#
# ceiling   solid,   dark      K_max(h), slopes to zero at contact
# floor     dashed,  dark      K_min, flat over the probe window
# schedule  dash-dot, mid      the gain trajectory inside the wedge
# wedge     solid light fill   admissible region, no hatching (hatching at
#                              this size turns into mud when printed)
# h_gear    thin vertical, dotted, annotated inline
#
# Label the curves INLINE with ax.annotate rather than using a legend: at 3.5
# inches a legend box costs more area than the curves it explains.
#
# For the abort case, prefer a second panel over an inset. Insets at this
# width are unreadable in print.