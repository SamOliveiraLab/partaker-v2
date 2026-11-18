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

from . import figure_style as FS
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
STAIN = ("stain", "Matrix stain over time")    # Figure 4 d, drawn by growth_plots.draw_inside from the Maturation tab's measure
STAIN_FIRST = "Matrix stain: on the Maturation tab click Dye frames..., then Measure. It then shows here."
RING_MIN_CELLS = 15            # a ring of the density map needs this many cells to get a colour
HEAT = LinearSegmentedColormap.from_list("microcolony", ["#fff3e0", "#fbd9a8", ORANGE, "#7a3d00"])


SPACING_MIN = 5                # the spacing between microcolonies is shown while every position has this many


def spacing_until(metrics: MicrocolonyMetrics) -> float:
    """The last hour at which every position has SPACING_MIN or more microcolonies, before the first hour at which one
    has fewer (Figure 3 i stops there, "while several microcolonies exist": 39 h on run 2). Later the nearest
    neighbour is the gap between two or three large microcolonies, not a spacing."""
    hours, rows = metrics._shared(metrics.values["microcolonies"]); few = np.flatnonzero((rows < SPACING_MIN).any(0))
    if not few.size:
        return float(hours[-1])
    return float(hours[max(int(few[0]) - 1, 0)])


def rule_note(settings: MicrocolonySettings = MicrocolonySettings()) -> str:
    return f"groups of {settings.min_cells} or more linked cells in each frame, not followed in time"


RULE_NOTE = rule_note()


def _positions(metrics: MicrocolonyMetrics) -> str:
    n = len(metrics.positions); return f"{n} position{'' if n == 1 else 's'}"


def _axes(fig, metrics: MicrocolonyMetrics, key: str, note: str, left: float = FS.CARD_LEFT, right: float = FS.CARD_RIGHT):
    """The panel as Figure 3 draws its metrics: the title, what the page adds under it (the value at the cursor, which
    microcolonies are counted), the plot, and room under it for the legend."""
    return FS.card(fig, TITLE[key], note, rule_note(metrics.settings), left=left, right=right)


def _nearest(hours: np.ndarray, cursor_h: Optional[float]) -> Optional[int]:
    return int(np.argmin(np.abs(np.asarray(hours) - cursor_h))) if cursor_h is not None and len(hours) else None


def _time_axis(ax, hours: np.ndarray, cursor_h: Optional[float]):
    FS.time_axis(ax, float(hours[0]), float(hours[-1])); FS.cursor(ax, cursor_h)


def _band(ax, hours, mean, sd, colour=FS.INK, k: Optional[int] = None, low=None, high=None):
    """Mean of the positions as dots on a line, their spread as a ribbon (as the bands of Figure 3)."""
    ok = np.isfinite(mean)
    ax.fill_between(hours[ok], (mean - sd if low is None else low)[ok], (mean + sd if high is None else high)[ok], color=colour, alpha=0.15, lw=0, zorder=2)
    ax.plot(hours[ok], mean[ok], color=colour, lw=1.5, marker="o", ms=3.5, zorder=3)
    if k is not None and np.isfinite(mean[k]):
        ax.plot([hours[k]], [mean[k]], "o", ms=7, mfc="white", mec=colour, mew=2, zorder=5)


def _simple(fig, metrics: MicrocolonyMetrics, key: str, what: str, cursor_h: Optional[float], words, ylabel: str, top: Optional[float] = None,
            until: Optional[float] = None, after: str = ""):
    """One curve with its spread. words(value, hour) -> the note at the cursor. until: the curve is shown to that hour
    only; `after` is the note when the cursor is past it."""
    hours, mean, sd = metrics.mean_of(what)
    if until is not None:
        keep = hours <= until + 1e-9; hours, mean, sd = hours[keep], mean[keep], sd[keep]
    k = _nearest(hours, cursor_h)
    note = f"mean and s.d. of {_positions(metrics)}" if k is None else words(float(mean[k]), float(hours[k]))
    if until is not None and cursor_h is not None and cursor_h > until + 1e-9:
        note = after
    ax = _axes(fig, metrics, key, note); _band(ax, hours, mean, sd, k=k)
    highest = float(np.nanmax(np.r_[(mean + sd)[np.isfinite(mean)], 1e-9]))
    ax.set_ylim(0, top if top is not None else highest * 1.12); ax.set_ylabel(ylabel, fontweight="bold"); _time_axis(ax, hours, cursor_h)
    return ax


@SP.with_house_style
def draw_count(fig, metrics: MicrocolonyMetrics, cursor_h: Optional[float] = None):
    return _simple(fig, metrics, "m_count", "microcolonies", cursor_h, lambda v, h: f"{v:.1f} at {h:g} h, mean of {_positions(metrics)}", "Microcolonies")


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
    ax.set_yscale("log"); ax.set_ylim(max(lo, 1.0) / 2.5, max(hi, 10.0) * 2.5)                    # the decades the data needs, as Figure 3 f (4 to 3,000 on run 2)
    ticks = [t for t in (1, 10, 100, 1000, 10000) if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    ax.set_yticks(ticks, [f"{t:,}" for t in ticks]); ax.minorticks_off(); ax.set_ylabel("Cells", fontweight="bold"); _time_axis(ax, hours, cursor_h)
    return ax


@SP.with_house_style
def draw_in(fig, metrics: MicrocolonyMetrics, cursor_h: Optional[float] = None):
    """Cells in microcolonies, with the same cells placed at random as rings where that was measured."""
    hours, mean, sd = metrics.mean_of("cells_in_pct"); _h, chance, _s = metrics.mean_of("random_pct"); k = _nearest(hours, cursor_h)
    note = f"mean and s.d. of {_positions(metrics)}" if k is None else f"{mean[k]:.1f} % at {hours[k]:g} h, mean of {_positions(metrics)}"
    ax = _axes(fig, metrics, "m_in", note); _band(ax, hours, mean, sd, k=k)
    rings = np.isfinite(chance)
    if rings.any():
        ax.plot(hours[rings], chance[rings], "o", ms=7, mfc="white", mec=FS.G1, mew=1.6, zorder=4)
    ax.set_ylim(0, 105); ax.set_ylabel("Cells (%)", fontweight="bold"); _time_axis(ax, hours, cursor_h)
    FS.legend_under(fig, FS.CARD_LEFT, [[("dots", FS.INK, "measured")] + ([("ring", FS.G1, "random placement")] if rings.any() else [])])
    return ax


@SP.with_house_style
def draw_density(fig, metrics: MicrocolonyMetrics, cursor_h: Optional[float] = None):
    hours, inside, sd_in = metrics.mean_of("density_in"); _h, single, sd_single = metrics.mean_of("density_single"); k = _nearest(hours, cursor_h)
    said = lambda v: "none" if not np.isfinite(v) else f"{v:.1f}"
    note = f"mean and s.d. of {_positions(metrics)}" if k is None else f"{said(inside[k])} in microcolonies, {said(single[k])} for single cells, at {hours[k]:g} h"
    ax = _axes(fig, metrics, "m_density", note)
    _band(ax, hours, single, sd_single, colour=FS.G1, k=k); _band(ax, hours, inside, sd_in, colour=ORANGE, k=k)
    both = np.r_[(inside + sd_in)[np.isfinite(inside)], (single + sd_single)[np.isfinite(single)], 1e-9]
    ax.set_ylim(0, float(both.max()) * 1.12); ax.set_ylabel("Cells per 100 µm²", fontweight="bold"); _time_axis(ax, hours, cursor_h)
    FS.legend_under(fig, FS.CARD_LEFT, [[("dots", ORANGE, "in microcolonies"), ("dots", FS.G1, "single cells")]])
    return ax


@SP.with_house_style
def draw_spacing(fig, metrics: MicrocolonyMetrics, cursor_h: Optional[float] = None):
    hours, have = metrics.have("spacing_um")
    words = lambda v, h: (f"{v:.1f} µm at {h:g} h, mean of {int(have[int(np.argmin(np.abs(hours - h)))])} of {_positions(metrics)}"
                          if np.isfinite(v) else f"fewer than two microcolonies at {h:g} h")
    until = spacing_until(metrics)
    return _simple(fig, metrics, "m_spacing", "spacing_um", cursor_h, words, "Nearest neighbour (µm)", until=until,
                   after=f"shown to {until:g} h: after it a position has fewer than {SPACING_MIN} microcolonies")


@SP.with_house_style
def draw_alignment(fig, metrics: MicrocolonyMetrics, cursor_h: Optional[float] = None):
    words = lambda v, h: f"{v:.2f} at {h:g} h (1: parallel to the neighbours, 0: no order)" if np.isfinite(v) else f"no microcolony at {h:g} h"
    return _simple(fig, metrics, "m_alignment", "alignment", cursor_h, words, "Alignment with neighbours", top=1.0)


@SP.with_house_style
def draw_largest(fig, metrics: MicrocolonyMetrics, cursor_h: Optional[float] = None):
    words = lambda v, h: f"{v:.1f} % at {h:g} h, mean of {_positions(metrics)}" if np.isfinite(v) else f"no microcolony at {h:g} h"
    return _simple(fig, metrics, "m_largest", "largest_share_pct", cursor_h, words, "Share of microcolony cells (%)", top=105)


@SP.with_house_style
def draw_area(fig, metrics: MicrocolonyMetrics, cursor_h: Optional[float] = None):
    words = lambda v, h: f"{v:.1f} % of the frame at {h:g} h, mean of {_positions(metrics)}"
    return _simple(fig, metrics, "m_area", "area_pct", cursor_h, words, "Surface covered (%)", top=100)


@SP.with_house_style
def draw_radial(fig, metrics: MicrocolonyMetrics, cursor_h: Optional[float] = None, min_cells: int = RING_MIN_CELLS):
    """Local cell density against the distance from the centre of each cell's own microcolony, all positions together."""
    hours, edges, density = metrics.radial(min_cells); filled = np.isfinite(density)
    if not filled.any():
        ax = _axes(fig, metrics, "m_radial", f"cells within 3 µm, per 100 µm², {_positions(metrics)} together"); ax.axis("off")
        ax.text(0.5, 0.5, f"fewer than {min_cells} cells at every distance", ha="center", va="center", transform=ax.transAxes, color=SP.EMPTY_GREY)
        return ax
    ax = _axes(fig, metrics, "m_radial", f"cells within 3 µm, per 100 µm², {_positions(metrics)} together", right=0.80)
    top = int(np.flatnonzero(filled.any(0)).max()) + 1; lo, hi = float(np.nanmin(density)), float(np.nanmax(density))
    low = 5.0 * np.floor(lo / 5.0); high = max(5.0 * np.ceil(hi / 5.0), low + 5.0)                 # round ends, so the colour scale reads 10, 20, 30
    between = (hours[:-1] + hours[1:]) / 2 if len(hours) > 1 else np.array([])                     # each column at its own hour, also when hours are missing
    half = float(np.median(np.diff(hours))) / 2 if len(hours) > 1 else 0.5
    columns = np.r_[hours[0] - half, between, hours[-1] + half]
    image = ax.pcolormesh(columns, edges[:top + 1], np.ma.masked_invalid(density[:, :top]).T, cmap=HEAT, vmin=low, vmax=high, shading="flat")
    ax.set_xlim(columns[0], columns[-1]); ax.set_ylim(float(edges[0]), float(edges[top]))
    ax.set_xticks(FS.time_ticks(float(hours[0]), float(hours[-1]))); ax.set_xlabel("Time (h)", fontweight="bold"); FS.cursor(ax, cursor_h); ax._time_axis = True
    ax.set_ylabel("Distance from centre (µm)", fontweight="bold")
    bar_axes = fig.add_axes([0.83, FS.AXES_BOTTOM, 0.025, FS.AXES_TOP - FS.AXES_BOTTOM]); bar = fig.colorbar(image, cax=bar_axes); bar.outline.set_linewidth(0.6)
    bar.set_label("Cells per 100 µm²", fontweight="bold")
    tens = [v for v in np.arange(10.0 * np.ceil(low / 10.0), high + 1e-9, 10.0)]; bar.set_ticks(tens if tens else [low, high])
    return ax


DRAW = {"m_count": draw_count, "m_per": draw_per, "m_in": draw_in, "m_density": draw_density, "m_spacing": draw_spacing,
        "m_alignment": draw_alignment, "m_largest": draw_largest, "m_radial": draw_radial, "m_area": draw_area}
