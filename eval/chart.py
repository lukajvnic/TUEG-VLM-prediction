"""Bar charts for eval run summaries.

Metric-agnostic on purpose: `write_bar_chart` takes labels and values on a
0-based scale, so accuracy, balanced accuracy or recall all render the same way.
Callers own the wording; this module owns the drawing.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as DrawPath

DPI = 150
SLOT_PX = 40           # horizontal budget per bar, air included
BAR_MAX_PX = 24        # bars never fill their slot
BAR_GAP_PX = 16        # air between neighbouring bars
CORNER_PX = 4          # rounded data-end, square at the baseline
PLOT_HEIGHT_PX = 380
PLOT_MIN_WIDTH_PX = 640
MARGIN_PX = {"left": 90, "right": 48, "top": 76, "bottom": 150}

SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

DEFAULT_TICKS = (0, 0.25, 0.5, 0.75, 1.0)
HEADROOM = 1.06        # space above the tallest bar for its direct label


def make_figure(bar_count):
    plot_width = max(PLOT_MIN_WIDTH_PX, bar_count * SLOT_PX)
    width = MARGIN_PX["left"] + plot_width + MARGIN_PX["right"]
    height = MARGIN_PX["top"] + PLOT_HEIGHT_PX + MARGIN_PX["bottom"]
    figure = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI, facecolor=SURFACE)
    axes = figure.add_axes((
        MARGIN_PX["left"] / width,
        MARGIN_PX["bottom"] / height,
        plot_width / width,
        PLOT_HEIGHT_PX / height,
    ))
    axes.set_facecolor(SURFACE)
    return figure, axes


def data_per_pixel(axes):
    inverse = axes.transData.inverted()
    origin_x, origin_y = inverse.transform((0, 0))
    unit_x, unit_y = inverse.transform((1, 1))
    return unit_x - origin_x, unit_y - origin_y


def rounded_bar(x, height, width, corner_x, corner_y):
    """A bar with rounded data-end and square baseline, as a closed path."""
    corner_x, corner_y = min(corner_x, width / 2), min(corner_y, height)
    left, right = x - width / 2, x + width / 2
    vertices = [
        (left, 0),
        (left, height - corner_y),
        (left, height), (left + corner_x, height),
        (right - corner_x, height),
        (right, height), (right, height - corner_y),
        (right, 0),
        (left, 0),
    ]
    codes = [
        DrawPath.MOVETO,
        DrawPath.LINETO,
        DrawPath.CURVE3, DrawPath.CURVE3,
        DrawPath.LINETO,
        DrawPath.CURVE3, DrawPath.CURVE3,
        DrawPath.LINETO,
        DrawPath.CLOSEPOLY,
    ]
    return PathPatch(DrawPath(vertices, codes), facecolor=SERIES, edgecolor="none", zorder=2)


def draw_bars(axes, values):
    """One hue for every bar — length already carries the magnitude."""
    unit_x, unit_y = data_per_pixel(axes)
    slot_px = axes.get_window_extent().width / max(len(values), 1)
    width_px = min(BAR_MAX_PX, max(1.0, slot_px - BAR_GAP_PX))
    for x, value in enumerate(values):
        axes.add_patch(rounded_bar(x, value, width_px * unit_x, CORNER_PX * unit_x, CORNER_PX * unit_y))


def style_axes(axes, labels, y_label, y_ticks):
    axes.set_xlim(-0.5, len(labels) - 0.5)
    axes.set_ylim(0, max(y_ticks) * HEADROOM)
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels(labels, rotation=45, ha="right")
    axes.set_yticks(y_ticks)
    axes.set_ylabel(y_label, color=INK_SECONDARY, fontsize=10, labelpad=10)
    axes.tick_params(axis="both", length=0, colors=INK_MUTED, labelsize=9)
    axes.set_axisbelow(True)
    axes.yaxis.grid(True, color=GRIDLINE, linewidth=1, linestyle="-")
    axes.xaxis.grid(False)
    for side, spine in axes.spines.items():
        spine.set_visible(side == "bottom")
    axes.spines["bottom"].set(color=BASELINE, linewidth=1)


def add_titles(axes, title, subtitle):
    axes.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left", pad=30)
    annotate(axes, subtitle, (0, 1), ("axes fraction", "axes fraction"), (0, 12), INK_SECONDARY, 9,
             ha="left", va="bottom")


def add_reference(axes, value, label):
    """The threshold rides in the margin — over the bars it would collide."""
    axes.axhline(value, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 4)), zorder=3)
    annotate(axes, label, (1, value), ("axes fraction", "data"), (8, 0), INK_MUTED, 8,
             ha="left", va="center")


def label_peak(axes, values, fmt):
    """Direct-label the extreme only; the axis carries the rest."""
    peak = max(range(len(values)), key=lambda index: values[index])
    annotate(axes, fmt.format(values[peak]), (peak, values[peak]), ("data", "data"), (0, 7), INK, 9,
             ha="center", va="bottom")


def annotate(axes, text, xy, coords, offset, color, size, **align):
    axes.annotate(
        text, xy=xy, xycoords=coords, xytext=offset, textcoords="offset points",
        color=color, fontsize=size, annotation_clip=False, **align,
    )


def write_bar_chart(path, labels, values, title, subtitle, y_label, reference=None,
                    y_ticks=DEFAULT_TICKS, value_format="{:.3f}"):
    if not values:
        return
    figure, axes = make_figure(len(values))
    style_axes(axes, labels, y_label, y_ticks)
    add_titles(axes, title, subtitle)
    draw_bars(axes, values)
    label_peak(axes, values, value_format)
    if reference:
        add_reference(axes, *reference)
    figure.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.16)
    plt.close(figure)
