"""The look of the paper's figures (Figures 3, 4 and 5) for the plots of the tool, so a plot on a page reads as the panel
it comes from: the growth states as blue-grey bands named I, II, III in serif, thresholds named above the plot, a ribbon
with dots, time ticks on round hours counted from the first frame, bold axis titles, the legend under the plot. The
places of labels and the limits come from the data of the loaded recording. Plain matplotlib, no Qt.

The colours are those of the figure scripts (f4style.py, f5common.py); the font sizes are the tool's (seeding_plots.house_rc)."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.lines import Line2D

INK, G1, G2 = "#1a1a1a", "#6e6e6e", "#a6a6a6"
STATE_SHADE = {"I": "#e3e9ef", "II": "#c9d5e0", "III": "#aebfd0"}          # calm blue-grey steps
MICROCOLONY_ORANGE, MATRIX_GREEN, PREDICTED_GREY, OUTSIDE_GREY = "#E08A00", "#1fbf3f", "#7a7a7a", "#8a8a8a"
CURSOR_RED = "#cc3311"                                                      # the cursor of the page's strip
DASH = (0, (3, 2))
# One layout for every plot card of the pages, so any two plots side by side line up: the title line, two note lines
# (what the page adds: the value at the cursor, a rule; or the names of the thresholds), one box for the axes, one band
# under it for the legend (two rows).
CARD_LEFT, CARD_RIGHT = 0.17, 0.955
TITLE_Y, NOTE_Y, NOTE2_Y = 0.955, 0.898, 0.843
AXES_BOTTOM, AXES_TOP = 0.235, 0.79
LEGEND_Y, LEGEND_ROW = 0.085, 0.056                                             # the title; a threshold's name sits ABOVE_Y over the axes
STEPS_H = (0.25, 0.5, 1, 2, 3, 6, 12, 24, 48, 96, 168)


def title(fig, left: float, text: str):
    return fig.text(left, TITLE_Y, text, fontsize=12, fontweight="bold", color=INK, va="center")


def card(fig, title_text: str, note: str = "", note2: str = "", left: float = CARD_LEFT, right: float = CARD_RIGHT):
    """A plot card: the title, up to two note lines, and the axes in the box every card shares. Returns the axes."""
    fig.clear(); title(fig, left, title_text)
    if note:
        fig.text(left, NOTE_Y, note, fontsize=9.5, color="#555555", va="center")
    if note2:
        fig.text(left, NOTE2_Y, note2, fontsize=8.5, color="#777777", va="center")
    ax = fig.add_axes([left, AXES_BOTTOM, right - left, AXES_TOP - AXES_BOTTOM])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def tile_grid(fig, n: int, columns: int, title_in: float = 0.2, caption_in: float = 0.0, margin_in: float = 0.08, gap_in: float = 0.1) -> List[list]:
    """Places for n square pictures in rows of `columns`, as large as the figure allows, each with room for a line above
    it (its hour) and caption_in inches under it. Returns [x, y, width, height] in figure fractions, row by row."""
    width, height = fig.get_figwidth(), fig.get_figheight(); rows = -(-int(n) // int(columns))
    cell_w, cell_h = (width - 2 * margin_in) / columns, (height - 2 * margin_in) / rows
    side = max(min(cell_w - gap_in, cell_h - title_in - caption_in - gap_in), 0.1); out = []
    for k in range(int(n)):
        r, c = divmod(k, int(columns))
        x = margin_in + c * cell_w + (cell_w - side) / 2
        top = height - margin_in - r * cell_h - title_in - (cell_h - title_in - caption_in - side) / 2
        out.append([x / width, (top - side) / height, side / width, side / height])
    return out


def time_ticks(first_h: float, last_h: float, most: int = 8) -> np.ndarray:
    """Hours to tick: a round step (6 h, 12 h, ...) counted from the first frame, at most `most` steps."""
    span = max(float(last_h) - float(first_h), 1e-9)
    step = next((s for s in STEPS_H if span / s <= most), STEPS_H[-1])
    return np.arange(float(first_h), float(last_h) + 1e-9, step)


def time_axis(ax, first_h: float, x_end: float, show: bool = True, most: int = 8):
    """The time axis of a panel: from just before the first frame to just after x_end."""
    span = max(x_end - first_h, 1e-9); ax.set_xlim(first_h - 0.012 * span - 0.0, x_end + 0.04 * span)
    ticks = time_ticks(first_h, x_end, most); ax.set_xticks(ticks); ax._time_axis = True          # a click on it moves the page's cursor
    if show:
        ax.set_xticklabels([f"{t:g}" for t in ticks]); ax.set_xlabel("Time (h)", fontweight="bold")
    else:
        ax.set_xticklabels([])


def count_axis(ax, top: float, label: str, bins: int = 4):
    """A y axis from 0 with round ticks and a comma in the thousands."""
    from matplotlib.ticker import MaxNLocator
    ticks = [t for t in MaxNLocator(nbins=bins, steps=[1, 2, 2.5, 5, 10]).tick_values(0, top) if 0 <= t <= top]
    ax.set_ylim(0, top); ax.set_yticks(ticks); ax.set_yticklabels([f"{t:,.0f}" if float(t).is_integer() else f"{t:g}" for t in ticks])
    ax.set_ylabel(label, fontweight="bold")


def bands(ax, states: Sequence[Tuple[str, float, float]]):
    """The growth states as shaded bands across the plot."""
    for name, a, b in states:
        ax.axvspan(a, b, color=STATE_SHADE[name], lw=0, zorder=0)


def band_names(ax, states: Sequence[Tuple[str, float, float]], y: float = 0.88, size: float = 13):
    """I, II, III in the middle of the part of each band that the plot shows, where it is wide enough."""
    lo, hi = ax.get_xlim(); out = []
    for name, a, b in states:
        a, b = max(a, lo), min(b, hi)
        if b - a > 0.07 * (hi - lo):
            out.append(ax.text((a + b) / 2, y, name, transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=size, fontweight="bold",
                               color=INK, family="serif", zorder=4))
    return out


def threshold(fig, ax, x: Optional[float], name: str, others: Sequence = ()):
    """A dashed line at a threshold on every panel of `others` and `ax`, its name above `ax` (the top panel)."""
    lo, hi = ax.get_xlim()
    if x is None or not np.isfinite(x) or not lo <= x <= hi:
        return None
    for a in (ax, *others):
        a.axvline(x, color=INK, lw=1.0, ls=DASH, zorder=1)
    box = ax.get_position()
    return fig.text(box.x0 + (x - lo) / (hi - lo) * box.width, NOTE2_Y, name, fontsize=9.5, fontweight="bold", color=INK, ha="center", va="center")


def cursor(ax, cursor_h: Optional[float]):
    """The page's cursor: a red line, as on the strip."""
    lo, hi = ax.get_xlim()
    if cursor_h is not None and lo <= cursor_h <= hi:
        ax.axvline(cursor_h, color=CURSOR_RED, lw=1.2, zorder=6)


def ribbon(ax, hours, mean, sd, colour, alpha: float = 0.15):
    """Mean plus and minus the s.d. of the positions, as a band."""
    hours, mean, sd = np.asarray(hours, float), np.asarray(mean, float), np.asarray(sd, float); ok = np.isfinite(mean)
    if ok.any() and np.nanmax(sd[ok]) > 0:
        ax.fill_between(hours[ok], (mean - sd)[ok], (mean + sd)[ok], color=colour, alpha=alpha, lw=0, zorder=2)


def dots(ax, hours, mean, colour, joined: bool = True, size: float = 4.0, width: float = 1.6):
    """The measured values as dots, joined by a line or not."""
    hours, mean = np.asarray(hours, float), np.asarray(mean, float); ok = np.isfinite(mean)
    return ax.plot(hours[ok], mean[ok], marker="o", ms=size, color=colour, lw=width, ls="-" if joined else "none", zorder=4)[0]


def legend_under(fig, left: float, rows: List[List[Tuple[str, str, str]]], y: float = LEGEND_Y, row_height: float = LEGEND_ROW):
    """The legend under the plot, as in the figures: rows of (kind, colour, words). Kinds: "dots" (dots on a line),
    "dot" (dots alone), "line", "dashed", "range" (a thick line with a large dot), "big dot", "ring"."""
    drawn = []
    for k, row in enumerate(rows):
        x, at = left, y - k * row_height
        for kind, colour, words in row:
            a, b = x, x + 0.05
            if kind in ("dots", "line", "dashed", "range"):
                fig.add_artist(Line2D([a, b], [at, at], transform=fig.transFigure, color=colour, lw=3.0 if kind == "range" else 1.6,
                                      ls=DASH if kind == "dashed" else "-", marker="o" if kind == "dots" else None, ms=4.0, solid_capstyle="butt"))
            if kind in ("dot", "range", "big dot", "ring"):
                fig.add_artist(Line2D([(a + b) / 2], [at], transform=fig.transFigure, color=colour, marker="o", ls="none",
                                      ms=7.0 if kind in ("range", "big dot") else 6.0 if kind == "ring" else 4.0,
                                      mfc="white" if kind == "ring" else colour, mew=1.4))
            text = fig.text(b + 0.012, at, words, fontsize=9, color=INK, va="center"); drawn.append(text); fig.canvas.draw()
            x = b + 0.012 + text.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width + 0.035
    return drawn
