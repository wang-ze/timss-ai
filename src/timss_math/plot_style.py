"""The look of the package's graphs: the chart surface, ink, the categorical series colors, a color and marker per
grade, and the matplotlib style (``STYLE``, for ``matplotlib.rc_context``).
"""

# Chart surface, ink, and the categorical series colors in their fixed order (each validated for color-vision
# deficiencies against the surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Grades keep one color and marker in every graph; the marker is a second channel besides the color.
GRADE_STYLES = {4: (SERIES[0], "o"), 8: (SERIES[1], "s"), 12: (SERIES[2], "^")}
STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    # Family names rather than "sans-serif", which matplotlib would resolve when it draws, outside this style; glyphs
    # that Arial lacks, such as subscripts, fall back to DejaVu Sans.
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "legend.frameon": False,
    "legend.fontsize": 9,
}
