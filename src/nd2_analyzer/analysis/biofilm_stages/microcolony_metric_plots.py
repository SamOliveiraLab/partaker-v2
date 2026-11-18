"""The plots of the Figure 3 metrics on the Microcolonies page (microcolony_metrics.MicrocolonyMetrics): every measured
position, mean and s.d. over the positions. Plain Python and matplotlib, no Qt.

These plots count the linked groups of each frame by themselves (close and enough). The plots of microcolony_plots show
the microcolonies of one position followed in time (close, enough and lasting, with their merges); the two kinds do not
give the same count, and every plot here says which kind it is under its title."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from . import seeding_plots as SP
from .microcolony_metrics import MicrocolonyMetrics
from .microcolony_rules import MicrocolonySettings

ORANGE = "#E08A00"             # the microcolony orange of Figure 3
FIGURES = (("m_count", "Number of microcolonies, all positions"), ("m_per", "Cells per microcolony"),
           ("m_in", "Cells in microcolonies, all positions"), ("m_density", "Local cell density"), ("m_spacing", "Microcolony spacing"),
           ("m_alignment", "Cell alignment"), ("m_largest", "Largest microcolony"), ("m_radial", "Density inside microcolonies"),
           ("m_area", "Microcolony area"))
TITLE = dict(FIGURES)
MEASURE_FIRST = "Click Measure (all positions) to see this plot"
RING_MIN_CELLS = 15            # a ring of the density map needs this many cells to get a colour
HEAT = LinearSegmentedColormap.from_list("microcolony", ["#fff3e0", "#fbd9a8", ORANGE, "#7a3d00"])


def rule_note(settings: MicrocolonySettings = MicrocolonySettings()) -> str:
    return f"groups of {settings.min_cells} or more linked cells in each frame, not followed in time"


RULE_NOTE = rule_note()


def _positions(metrics: MicrocolonyMetrics) -> str:
    n = len(metrics.positions); return f"{n} position{'' if n == 1 else 's'}"


def _axes(fig, metrics: MicrocolonyMetrics, key: str, note: str, left: float = 0.16, right: float = 0.95):
    fig.clear(); ax = fig.add_axes([left, 0.15, right - left, 0.56]); SP.open_frame(ax)
    fig.text(left, 0.93, TITLE[key], fontsize=12, fontweight="bold", color=SP.INK, va="center")
    fig.text(left, 0.87, note, fontsize=9.5, color="#555555", va="center")
    fig.text(left, 0.82, rule_note(metrics.settings), fontsize=8.5, color="#777777", va="center")
    return ax


def _key(fig, left: float, items):
    x = left
    for colour, style, words in items:
        xs = [x + 0.0225] if "marker" in style else [x, x + 0.045]
        fig.add_artist(Line2D(xs, [0.765] * len(xs), transform=fig.transFigure, color=colour, lw=2, **style))
        t = fig.text(x + 0.055, 0.765, words, fontsize=9, color="#333333", va="center"); fig.canvas.draw()
        x += 0.055 + t.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width + 0.03


def _nearest(hours: np.ndarray, cursor_h: Optional[float]) -> Optional[int]:
    return int(np.argmin(np.abs(np.asarray(hours) - cursor_h))) if cursor_h is not None and len(hours) else None


def _time_axis(ax, hours: np.ndarray, cursor_h: Optional[float]):
    first, last = float(hours[0]), float(hours[-1]); pad = (last - first) * 0.03 or 0.5
    if cursor_h is not None and first <= cursor_h <= last:
        ax.axvline(cursor_h, color="#9a9a9a", lw=1.2, zorder=1)
    ax.set_xlim(first - pad, last + pad); ax.set_xlabel("time after inoculation (h)")


def _band(ax, hours, mean, sd, colour=SP.BLUE, k: Optional[int] = None, low=None, high=None):
    ok = np.isfinite(mean)
    ax.fill_between(hours[ok], (mean - sd if low is None else low)[ok], (mean + sd if high is None else high)[ok], color=colour, alpha=0.15, lw=0, zorder=2)
    ax.plot(hours[ok], mean[ok], color=colour, lw=2, zorder=3)
    if k is not None and np.isfinite(mean[k]):
        ax.plot([hours[k]], [mean[k]], "o", ms=7, mfc="white", mec=colour, mew=2, zorder=5)


def _simple(fig, metrics: MicrocolonyMetrics, key: str, what: str, cursor_h: Optional[float], words, ylabel: str, top: Optional[float] = None):
    """One curve with its spread. words(value, hour) -> the note at the cursor."""
    hours, mean, sd = metrics.mean_of(what); k = _nearest(hours, cursor_h)
    note = f"mean and s.d. of {_positions(metrics)}" if k is None else words(float(mean[k]), float(hours[k]))
    ax = _axes(fig, metrics, key, note); _band(ax, hours, mean, sd, k=k)
    highest = float(np.nanmax(np.r_[(mean + sd)[np.isfinite(mean)], 1e-9]))
    ax.set_ylim(0, top if top is not None else highest * 1.12); ax.set_ylabel(ylabel); _time_axis(ax, hours, cursor_h)
    return ax


@SP.with_house_style
def draw_count(fig, metrics: MicrocolonyMetrics, cursor_h: Optional[float] = None):
    return _simple(fig, metrics, "m_count", "microcolonies", cursor_h, lambda v, h: f"{v:.1f} at {h:g} h, mean of {_positions(metrics)}", "microcolonies")


@SP.with_house_style
def draw_per(fig, metrics: MicrocolonyMetrics, cursor_h: Optional[float] = None):
    """Cells per microcolony on a logarithmic axis; the mean is the mean of the logarithms, as Figure 3 f draws it."""
    hours, mean, sd_log = metrics.mean_of("cells_per_microcolony", geometric=True); k = _nearest(hours, cursor_h)
    said = lambda v: f"{v:,.0f}" if v >= 100 else f"{v:.1f}"
    note = f"mean and s.d. of {_positions(metrics)}" if k is None or not np.isfinite(mean[k]) else f"{said(float(mean[k]))} cells at {hours[k]:g} h, mean of {_positions(metrics)}"
    ax = _axes(fig, metrics, "m_per", note)
    low, high = mean / 10 ** sd_log, mean * 10 ** sd_log; _band(ax, hours, mean, sd_log, k=k, low=low, high=high)
    seen = np.r_[low[np.isfinite(low)], high[np.isfinite(high)]]
    lo, hi = (float(seen.min()), float(seen.max())) if seen.size else (1.0, 10.0)
    ax.set_yscale("log"); ax.set_ylim(10 ** np.floor(np.log10(max(lo, 1.0))), 10 ** np.ceil(np.log10(max(hi, 10.0)) + 1e-9))
    ticks = [t for t in (1, 10, 100, 1000, 10000) if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    ax.set_yticks(ticks, [f"{t:,}" for t in ticks]); ax.minorticks_off(); ax.set_ylabel("cells per microcolony"); _time_axis(ax, hours, cursor_h)
    return ax


@SP.with_house_style
def draw_in(fig, metrics: MicrocolonyMetrics, cursor_h: Optional[float] = None):
    """Cells in microcolonies, with the same cells placed at random as rings where that was measured."""
    hours, mean, sd = metrics.mean_of("cells_in_pct"); _h, chance, _s = metrics.mean_of("random_pct"); k = _nearest(hours, cursor_h)
    note = f"mean and s.d. of {_positions(metrics)}" if k is None else f"{mean[k]:.1f} % at {hours[k]:g} h, mean of {_positions(metrics)}"
    ax = _axes(fig, metrics, "m_in", note); _band(ax, hours, mean, sd, k=k)
    rings = np.isfinite(chance)
    if rings.any():
        ax.plot(hours[rings], chance[rings], "o", ms=7, mfc="white", mec=SP.GREY, mew=1.6, zorder=4)
        _key(fig, 0.16, [(SP.BLUE, {}, "measured"), (SP.GREY, dict(marker="o", ms=7, mfc="white", mew=1.6, ls=""), "the same cells placed at random")])
    ax.set_ylim(0, 105); ax.set_ylabel("cells in microcolonies (%)"); _time_axis(ax, hours, cursor_h)
    return ax


@SP.with_house_style
def draw_density(fig, metrics: MicrocolonyMetrics, cursor_h: Optional[float] = None):
    hours, inside, sd_in = metrics.mean_of("density_in"); _h, single, sd_single = metrics.mean_of("density_single"); k = _nearest(hours, cursor_h)
    said = lambda v: "none" if not np.isfinite(v) else f"{v:.1f}"
    note = f"mean and s.d. of {_positions(metrics)}" if k is None else f"{said(inside[k])} in microcolonies, {said(single[k])} for single cells, at {hours[k]:g} h"
    ax = _axes(fig, metrics, "m_density", note)
    _band(ax, hours, single, sd_single, colour=SP.GREY, k=k); _band(ax, hours, inside, sd_in, colour=ORANGE, k=k)
    both = np.r_[(inside + sd_in)[np.isfinite(inside)], (single + sd_single)[np.isfinite(single)], 1e-9]
    ax.set_ylim(0, float(both.max()) * 1.12); ax.set_ylabel("cells within 3 µm, per 100 µm²"); _time_axis(ax, hours, cursor_h)
    _key(fig, 0.16, [(ORANGE, {}, "cells in microcolonies"), (SP.GREY, {}, "single cells")])
    return ax


@SP.with_house_style
def draw_spacing(fig, metrics: MicrocolonyMetrics, cursor_h: Optional[float] = None):
    hours, have = metrics.have("spacing_um")
    words = lambda v, h: (f"{v:.1f} µm at {h:g} h, mean of {int(have[int(np.argmin(np.abs(hours - h)))])} of {_positions(metrics)}"
                          if np.isfinite(v) else f"fewer than two microcolonies at {h:g} h")
    return _simple(fig, metrics, "m_spacing", "spacing_um", cursor_h, words, "to the nearest microcolony (µm)")


@SP.with_house_style
def draw_alignment(fig, metrics: MicrocolonyMetrics, cursor_h: Optional[float] = None):
    words = lambda v, h: f"{v:.2f} at {h:g} h (1: parallel to the neighbours, 0: no order)" if np.isfinite(v) else f"no microcolony at {h:g} h"
    return _simple(fig, metrics, "m_alignment", "alignment", cursor_h, words, "alignment with neighbours", top=1.0)


@SP.with_house_style
def draw_largest(fig, metrics: MicrocolonyMetrics, cursor_h: Optional[float] = None):
    words = lambda v, h: f"{v:.1f} % at {h:g} h, mean of {_positions(metrics)}" if np.isfinite(v) else f"no microcolony at {h:g} h"
    return _simple(fig, metrics, "m_largest", "largest_share_pct", cursor_h, words, "cells in the largest microcolony\n(% of the cells in microcolonies)", top=105)


@SP.with_house_style
def draw_area(fig, metrics: MicrocolonyMetrics, cursor_h: Optional[float] = None):
    words = lambda v, h: f"{v:.1f} % of the frame at {h:g} h, mean of {_positions(metrics)}"
    return _simple(fig, metrics, "m_area", "area_pct", cursor_h, words, "surface covered by microcolonies (%)", top=100)


@SP.with_house_style
def draw_radial(fig, metrics: MicrocolonyMetrics, cursor_h: Optional[float] = None, min_cells: int = RING_MIN_CELLS):
    """Local cell density against the distance from the centre of each cell's own microcolony, all positions together."""
    hours, edges, density = metrics.radial(min_cells); filled = np.isfinite(density)
    if not filled.any():
        ax = _axes(fig, metrics, "m_radial", f"cells within 3 µm, per 100 µm², {_positions(metrics)} together"); ax.axis("off")
        ax.text(0.5, 0.5, f"fewer than {min_cells} cells at every distance", ha="center", va="center", transform=ax.transAxes, color=SP.EMPTY_GREY)
        return ax
    ax = _axes(fig, metrics, "m_radial", f"cells within 3 µm, per 100 µm², {_positions(metrics)} together", right=0.80)
    step = float(np.median(np.diff(hours))) if len(hours) > 1 else 1.0; top = int(np.flatnonzero(filled.any(0)).max()) + 1
    lo, hi = float(np.nanmin(density)), float(np.nanmax(density))
    image = ax.imshow(np.ma.masked_invalid(density[:, :top]).T, origin="lower", aspect="auto", cmap=HEAT, vmin=np.floor(lo), vmax=max(np.ceil(hi), np.floor(lo) + 1),
                      extent=(hours[0] - step / 2, hours[-1] + step / 2, float(edges[0]), float(edges[top])), interpolation="nearest")
    if cursor_h is not None and hours[0] <= cursor_h <= hours[-1]:
        ax.axvline(cursor_h, color="#555555", lw=1.2)
    ax.set_xlabel("time after inoculation (h)"); ax.set_ylabel("distance from the microcolony's centre (µm)")
    bar_axes = fig.add_axes([0.83, 0.15, 0.025, 0.56]); bar = fig.colorbar(image, cax=bar_axes); bar.outline.set_linewidth(0.6)
    bar.set_label("cells per 100 µm²")
    return ax


DRAW = {"m_count": draw_count, "m_per": draw_per, "m_in": draw_in, "m_density": draw_density, "m_spacing": draw_spacing,
        "m_alignment": draw_alignment, "m_largest": draw_largest, "m_radial": draw_radial, "m_area": draw_area}
