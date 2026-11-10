"""Drawing for the Dispersion page: the hour strip, the gallery panels (PANELS, draw_panel) and the export pages. Pure
matplotlib on a given Figure, so the page, the export and the tests draw the same thing.

House style and the collision check come from the Seeding plots (seeding_plots, SP): Arial, bold titles and axis labels,
open frame, ticks out, no grid, nothing within 0.1 in of anything else. Colours: blue for the cells beyond the biofilm,
green for the cloud, grey for support. Frames that are out of focus sit in a light grey band, their points hollow and
not joined to the curve.

Every panel is one function draw_<name>(fig, results, span, cursor_h, settings, cell=None) -> ax. With cell None it owns
the whole figure (one gallery canvas, about 3.6 x 2.6 in); with cell = (x0, y0, w, h) in figure fractions it draws
inside that cell of an export page."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import polars as pl

from nd2_analyzer.analysis.biofilm_stages import seeding_plots as SP
from nd2_analyzer.analysis.dispersion.measure import DispersionSettings

BLUE, GREEN, GREY, INK, RED = SP.BLUE, SP.GREEN, SP.GREY, SP.INK, SP.RED
SKY, YELLOW = "#56B4E9", "#F0E442"            # the view area overlay: free cells, the outline of the current frame
LIGHT = "#a6a6a6"
FOCUS_SHADE = "#ececec"                       # the band behind frames that are out of focus
BODY_SHADE = "#e4e4e4"                        # the biofilm on the map
HEAT_RECT = [0.20, 0.22, 0.55, 0.63]          # a panel with a colour bar at its right
BAR_RECT = [0.785, 0.22, 0.03, 0.63]
PANEL_SIZE_IN = SP.PANEL_SIZE_IN
RUN_MEASURE = "Run Measure to see this plot"
GROUP_TITLES = (("leaving", "Cells leaving the biofilm"), ("where", "Where they go"), ("motion", "Staying or moving"),
                ("checks", "Checks"))
GROUP_HEADING = dict(GROUP_TITLES)


# ---------------------------------------------------------------------------------------------- small helpers
def settings_of(results, settings) -> DispersionSettings:
    if settings is not None:
        return settings
    return results.settings if results is not None else DispersionSettings()


def length_unit(s: DispersionSettings) -> tuple:
    """(unit, factor from pixels)."""
    return ("µm", float(s.pixel_um)) if s.pixel_um else ("px", 1.0)


def area_unit(s: DispersionSettings) -> tuple:
    return ("µm²", float(s.pixel_um) ** 2) if s.pixel_um else ("px²", 1.0)


def thousands(values: np.ndarray, unit: str) -> tuple:
    """Large values in thousands: (values, unit)."""
    v = np.asarray(values, float)
    if np.nanmax(v, initial=0.0) >= 10000:
        return v / 1000.0, f"10³ {unit}"
    return v, unit


def column(results, name: str) -> np.ndarray:
    """One per_frame column as floats, nulls as nan."""
    return results.per_frame[name].cast(pl.Float64).to_numpy()


def frame_step(hours: Sequence[float]) -> float:
    h = np.asarray(hours, float)
    return float(np.median(np.diff(h))) if len(h) > 1 else 1.0


def focus_runs(hours: Sequence[float], in_focus: Sequence[bool]) -> list:
    """The out-of-focus stretches as (first_hour, last_hour) blocks, half a frame step added at each end."""
    h = np.asarray(hours, float); ok = np.asarray(in_focus, bool); half = frame_step(h) / 2
    out, start = [], None
    for i in range(len(h)):
        if not ok[i] and start is None:
            start = i
        if start is not None and (ok[i] or i == len(h) - 1):
            last = i - 1 if ok[i] else i
            out.append((float(h[start] - half), float(h[last] + half))); start = None
    return out


def time_ticks(span: Sequence[float]) -> np.ndarray:
    """Hour ticks a small panel has room for: at most six."""
    a, b = float(span[0]), float(span[1]); width = max(b - a, 1e-6)
    step = 0.25
    for step in (0.25, 0.5, 1, 2, 3, 6, 12, 24, 48):
        if width / step <= 5:
            break
    return np.arange(np.ceil(a / step - 1e-9) * step, b + 1e-9, step)


def time_axis(ax, span, xlim=None):
    a, b = float(span[0]), float(span[1])
    ax.set_xlim(*(xlim or SP.pad_span((a, b), 0.04))); ax.set_xticks(time_ticks((a, b)))
    ax.set_xlabel("time (h)")


def shade_out_of_focus(ax, results):
    for lo, hi in focus_runs(results.hours, results.in_focus):
        ax.axvspan(lo, hi, color=FOCUS_SHADE, lw=0, zorder=0)


def focus_line(ax, results, y, color, label=None, ls="-", pair=False):
    """One curve through the frames in focus, a marker at every frame; the frames out of focus small, pale and apart.
    pair=True for values that need the next frame in focus too."""
    h = results.hours; y = np.asarray(y, float)
    ok = (column(results, "pair_in_focus") > 0 if pair else results.in_focus) & np.isfinite(y)
    ax.plot(h, np.where(ok, y, np.nan), color=color, ls=ls, label=label, zorder=3, **SP.MK)
    off = ~ok & np.isfinite(y)
    if off.any():
        ax.plot(h[off], y[off], ls="none", marker="o", ms=3.5, mfc="white", mec=color, mew=1.0, alpha=0.45, zorder=2)


def top_of(*series) -> float:
    v = np.concatenate([np.asarray(s, float).ravel() for s in series])
    v = v[np.isfinite(v)]
    return float(v.max()) if len(v) else 1.0


def line_panel(fig, results, span, cursor_h, cell, title, ylabel, series, legend=False, headroom=1.15, rect=None):
    """A time panel of one or more curves: series is a list of (values, colour, label, pair)."""
    ax = SP.panel_axes(fig, cell, rect)
    shade_out_of_focus(ax, results)
    for values, colour, label, pair in series:
        focus_line(ax, results, values, colour, label, pair=pair)
    SP.draw_cursor(ax, cursor_h, span)
    time_axis(ax, span)
    ax.set_ylim(0, max(top_of(*[s[0] for s in series]) * (1.38 if legend else headroom), 1e-9))
    if legend:
        ax.legend(loc="upper left", handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
    ax.set_title(title, loc="left"); ax.set_ylabel(ylabel)
    SP.open_frame(ax)
    return ax


def heat_panel(fig, results, span, cursor_h, cell, title, ylabel, y_edges, z, bar_label):
    """A time panel that is a map: hours across, y_edges up, z (rows, frames) as blues, the colour bar at the right."""
    ax = SP.panel_axes(fig, cell, HEAT_RECT)
    h = results.hours; half = frame_step(h) / 2
    x_edges = np.r_[h - half, h[-1] + half]
    sharp = z[:, results.in_focus] if results.in_focus.any() else z
    vmax = max(float(np.percentile(sharp, 99.5)), 1e-9)
    im = ax.pcolormesh(x_edges, y_edges, z, cmap="Blues", vmin=0, vmax=vmax, shading="flat", rasterized=True)
    for lo, hi in focus_runs(h, results.in_focus):
        ax.axvspan(lo, hi, color="white", alpha=0.65, lw=0)
    if cursor_h is not None and span[0] <= float(cursor_h) <= span[1]:
        ax.axvline(float(cursor_h), color=INK, lw=1.0)
    time_axis(ax, span, xlim=(x_edges[0], x_edges[-1]))
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_title(title, loc="left"); ax.set_ylabel(ylabel)
    colour_bar(fig, im, cell, BAR_RECT, bar_label)
    return ax


def colour_bar(fig, im, cell, rect, label):
    cax = fig.add_axes(rect if cell is None else SP.cell_rect(cell, rect))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label); cb.outline.set_linewidth(0.8)
    cb.ax.tick_params(width=0.8, length=3)
    cb.locator = __import__("matplotlib.ticker", fromlist=["MaxNLocator"]).MaxNLocator(4); cb.update_ticks()
    return cb


# ---------------------------------------------------------------------------------------------- the strip
@SP.with_house_style
def draw_strip(ax, span: Sequence[float], cursor_h: float, hours=None, in_focus=None):
    """The recording's span in hours, a tick for every frame, the frames out of focus in a darker band, and the red cursor
    with its hour."""
    from matplotlib.patches import Rectangle
    a, b = float(span[0]), float(span[1])
    ax.set_xlim(*SP.pad_span((a, b), 0.01)); ax.set_ylim(0, 1); ax.set_yticks([])
    ticks = SP.hour_ticks((a, b))
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g} h" for t in ticks])
    ax.tick_params(axis="x", labelsize=8.5, length=3)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    band_y, band_h = 0.02, 0.56
    c = float(min(max(cursor_h, a), b))
    ax.add_patch(Rectangle((a, band_y), b - a, band_h, color="#f2f2f2", ec="none"))
    if hours is not None and len(hours):
        h = np.asarray(hours, float)
        if in_focus is not None:
            for lo, hi in focus_runs(h, in_focus):
                lo, hi = max(lo, a), min(hi, b)
                ax.add_patch(Rectangle((lo, band_y), hi - lo, band_h, color="#d6d6d6", ec="none"))
                x = SP.label_x_clear_of_cursor(ax, lo, hi, "out of focus", c)
                if x is not None:
                    ax.text(x, band_y + band_h / 2, "out of focus", ha="center", va="center", fontsize=8, color=INK)
        ax.vlines(h, band_y, band_y + 0.14, color=GREY, lw=0.8)
    ax.axvline(c, color=RED, lw=1.8)
    near_right = (c - a) > 0.85 * (b - a)
    side = -1 if near_right else 1
    ax.text(c + side * 0.006 * (b - a), 0.70, f"{c:.2f} h", ha="right" if near_right else "left", va="bottom", fontsize=8.5,
            color=RED, fontweight="bold")
    return ax


# ---------------------------------------------------------------------------------------------- cells leaving
@SP.with_house_style
def draw_planktonic_mass(fig, results, span, cursor_h, settings, cell=None):
    return line_panel(fig, results, span, cursor_h, cell, "Planktonic cell mass", "cells (estimated)",
                      [(column(results, "cells_outside"), BLUE, None, False)])


@SP.with_house_style
def draw_cloud_and_free(fig, results, span, cursor_h, settings, cell=None):
    return line_panel(fig, results, span, cursor_h, cell, "In the cloud and beyond it", "cells (estimated)",
                      [(column(results, "cells_cloud"), GREEN, "in the cloud", False),
                       (column(results, "cells_free"), BLUE, "beyond the cloud", False)], legend=True)


@SP.with_house_style
def draw_cloud_area(fig, results, span, cursor_h, settings, cell=None):
    unit, k = area_unit(settings_of(results, settings))
    v, unit = thousands(column(results, "cloud_area_px") * k, unit)
    return line_panel(fig, results, span, cursor_h, cell, "Area of the dispersal cloud", f"area ({unit})",
                      [(v, GREEN, None, False)])


# ---------------------------------------------------------------------------------------------- where they go
@SP.with_house_style
def draw_distance_map(fig, results, span, cursor_h, settings, cell=None):
    unit, k = length_unit(settings_of(results, settings))
    return heat_panel(fig, results, span, cursor_h, cell, "Distance from the biofilm", f"distance ({unit})",
                      results.ring_edges * k, results.distance_map * 100.0, "area covered (%)")


@SP.with_house_style
def draw_exit_map(fig, results, span, cursor_h, settings, cell=None):
    unit, k = length_unit(settings_of(results, settings))
    arc = np.r_[results.outline_arc_px, results.outline_arc_px[-1]][results.stretch_edges] * k
    per = results.exit_map / results.cell_area_px / np.maximum(results.stretch_px[:, None] * k, 1e-9) * 100.0
    return heat_panel(fig, results, span, cursor_h, cell, "Where cells leave", f"along the outline ({unit})", arc, per,
                      f"cells per 100 {unit}")


def outline_ruler(ax, results, unit: str, k: float):
    """The starting outline as a ruler: a tick and the distance along it at round steps, the numbers on the biofilm
    side. The same distance is the y axis of the exit map."""
    c = results.outline_yx; arc = results.outline_arc_px * k
    ax.plot(c[:, 1], c[:, 0], color=INK, lw=1.2)
    total = float(arc[-1]) if len(arc) else 0.0
    step = next((s for s in (25, 50, 100, 200, 250, 500, 1000, 2000) if total / s <= 5), 5000)
    m = results.start_mask
    for d in np.arange(0, total + 1e-9, step):
        i = int(np.searchsorted(arc, d)); i = min(max(i, 3), len(c) - 4)
        ty, tx = c[i + 3] - c[i - 3]; n = np.hypot(ty, tx) or 1.0
        ny, nx = -tx / n, ty / n                                   # a normal of the outline
        y, x = c[i]
        py, px = int(round(y + 12 * ny)), int(round(x + 12 * nx))
        inside = 0 <= py < m.shape[0] and 0 <= px < m.shape[1] and m[py, px]
        if not inside:
            ny, nx = -ny, -nx
        ax.plot([x - 7 * nx, x + 7 * nx], [y - 7 * ny, y + 7 * ny], color=INK, lw=1.2)
        reach = 26 + 26 * abs(nx)                                   # numbers are wider than tall
        ax.text(x + reach * nx, y + reach * ny, f"{d:g}", ha="center", va="center", fontsize=8, color=INK)


@SP.with_house_style
def draw_occupancy(fig, results, span, cursor_h, settings, cell=None):
    """The map: the biofilm grey, the share of the sharp frames with a dark object at each place in blues, the starting
    outline as a ruler."""
    from matplotlib import colormaps
    s = settings_of(results, settings)
    unit, k = length_unit(s)
    fw, fh = fig.get_size_inches(); x0, y0, w, h = cell or (0.0, 0.0, 1.0, 1.0)
    Y, X = results.shape
    img_w = 0.72 * w; img_h = img_w * fw * Y / X / fh
    top = y0 + 0.85 * h
    if cell is None:
        fig.clear()
    ax = fig.add_axes([x0 + 0.05 * w, top - img_h, img_w, img_h])
    occ = results.occupancy * 100.0
    vmax = max(float(np.percentile(occ[occ > 0], 99)) if (occ > 0).any() else 1.0, 1e-9)
    rgb = colormaps["Blues"](np.clip(occ / vmax, 0, 1))[..., :3]
    e = results.settings.edge_px                                    # the margin is not measured: draw the biofilm through it
    body = np.pad(results.start_mask[e:Y - e, e:X - e], e, mode="edge") if e > 0 else results.start_mask
    rgb[body] = np.array([int(BODY_SHADE[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    ax.imshow(rgb, interpolation="nearest")
    outline_ruler(ax, results, unit, k)
    ax.set_xlim(-0.5, X - 0.5); ax.set_ylim(Y - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.8)
    ax.set_title("Where cells were found", loc="left")
    ax.set_xlabel(f"numbers: distance along the outline ({unit})", fontsize=9, fontweight="normal")
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    cax = fig.add_axes([x0 + 0.80 * w, top - img_h, 0.03 * w, img_h])
    cb = fig.colorbar(cm.ScalarMappable(Normalize(0, vmax), "Blues"), cax=cax)
    cb.set_label("share of frames (%)"); cb.outline.set_linewidth(0.8); cb.ax.tick_params(width=0.8, length=3)
    cb.locator = __import__("matplotlib.ticker", fromlist=["MaxNLocator"]).MaxNLocator(4); cb.update_ticks()
    return ax


# ---------------------------------------------------------------------------------------------- staying or moving
@SP.with_house_style
def draw_stay_share(fig, results, span, cursor_h, settings, cell=None):
    ax = line_panel(fig, results, span, cursor_h, cell, "Share staying in place", "share of the mass (%)",
                    [(column(results, "stay_fraction") * 100.0, BLUE, "seen again in the next frame", True)], legend=True)
    ok = column(results, "pair_in_focus") > 0
    ax.plot(results.hours, np.where(ok, column(results, "chance_fraction") * 100.0, np.nan), color=GREY, ls="--", lw=1.4,
            label="expected by chance", zorder=2.5)
    ax.legend(loc="upper left", handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
    return ax


@SP.with_house_style
def draw_stay_and_once(fig, results, span, cursor_h, settings, cell=None):
    return line_panel(fig, results, span, cursor_h, cell, "Staying and seen once", "cells (estimated)",
                      [(column(results, "cells_staying"), BLUE, "stays in place", True),
                       (column(results, "cells_seen_once"), GREY, "seen once", True)], legend=True)


@SP.with_house_style
def draw_object_size(fig, results, span, cursor_h, settings, cell=None):
    """The areas of the free dark objects in the sharp frames; their median is the area of one cell (estimated)."""
    title = "Size of one object"
    a = np.asarray(results.free_areas_px, float)
    if not len(a):
        return SP.grey_panel(fig, title, "no object beyond the biofilm in the sharp frames", cell)
    unit, k = area_unit(settings_of(results, settings))
    ax = SP.panel_axes(fig, cell)
    hi = max(float(np.percentile(a, 97)), results.settings.min_object_px + 4.0)     # the largest 3 % are off the scale
    width = max(int(round((hi - results.settings.min_object_px) / 24)), 1)
    edges = np.arange(results.settings.min_object_px - 0.5, hi + width, width)
    n, _ = np.histogram(a, edges)
    ax.bar(edges[:-1] * k, n, width=np.diff(edges) * k, align="edge", color=BLUE, ec="white", lw=0.5)
    med = results.cell_area_px * k
    ax.axvline(med, color=INK, ls="--", lw=1.4)
    ax.set_ylim(0, float(n.max()) * 1.2)
    ax.text(med + 0.03 * (edges[-1] - edges[0]) * k, float(n.max()) * 1.14, f"median {med:.3g} {unit}", ha="left", va="top",
            fontsize=9, color=INK)
    ax.set_xlim(edges[0] * k, edges[-1] * k)
    ax.set_title(title, loc="left"); ax.set_xlabel(f"area of one object ({unit})"); ax.set_ylabel("objects")
    SP.open_frame(ax)
    return ax


# ---------------------------------------------------------------------------------------------- checks
@SP.with_house_style
def draw_object_counts(fig, results, span, cursor_h, settings, cell=None):
    return line_panel(fig, results, span, cursor_h, cell, "Objects inside and outside", "dark objects",
                      [(column(results, "objects_in_body"), GREY, "inside the outline", False),
                       (column(results, "objects_beyond_body"), BLUE, "outside it", False)], legend=True)


@SP.with_house_style
def draw_focus(fig, results, span, cursor_h, settings, cell=None):
    sharp = column(results, "sharpness"); v, unit = thousands(sharp, "")
    k = 1000.0 if unit else 1.0
    ax = line_panel(fig, results, span, cursor_h, cell, "Focus of each frame",
                    "sharpness (10³)" if unit else "sharpness", [(v, GREY, None, False)], headroom=1.38)
    limit = results.settings.focus_fraction * float(np.median(sharp)) / k
    ax.axhline(limit, color=INK, ls="--", lw=1.2, label="out of focus below this")
    ax.legend(loc="upper left", handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
    return ax


@SP.with_house_style
def draw_body_area(fig, results, span, cursor_h, settings, cell=None):
    start = float(results.start_mask.sum()) or 1.0
    ax = line_panel(fig, results, span, cursor_h, cell, "Area of the biofilm", "area (% of the start)",
                    [(column(results, "body_area_px") / start * 100.0, GREY, None, False)], headroom=1.12)
    ax.axhline(100, color=LIGHT, ls=":", lw=1.2, zorder=1)
    return ax


# ---------------------------------------------------------------------------------------------- the list of panels
# (group, key, title, function, what the plot shows: the tooltip and the caption)
PANELS = [
    ("leaving", "planktonic_mass", "Planktonic cell mass", draw_planktonic_mass,
     "Dark area beyond the starting outline of the biofilm, divided by the area of one cell (estimated)."),
    ("leaving", "cloud_and_free", "In the cloud and beyond it", draw_cloud_and_free,
     "The same mass in two parts: in the cloud (the dense region that grows out of the biofilm) and beyond the cloud "
     "(scattered cells)."),
    ("leaving", "cloud_area", "Area of the dispersal cloud", draw_cloud_area,
     "Area inside the outline of each frame that lies beyond the starting outline."),
    ("where", "distance_map", "Distance from the biofilm", draw_distance_map,
     "How much of each ring around the starting outline is covered by dark objects, frame by frame."),
    ("where", "exit_map", "Where cells leave", draw_exit_map,
     "Cells (estimated) beyond each stretch of the starting outline, frame by frame. The distance along the outline is "
     "marked on the map beside this plot."),
    ("where", "occupancy", "Where cells were found", draw_occupancy,
     "The share of the sharp frames with a dark object at each place. Grey: the biofilm at the start. The numbers are "
     "the distance along its outline."),
    ("motion", "stay_share", "Share staying in place", draw_stay_share,
     "The share of the mass beyond the biofilm that is dark again at the same place in the next frame, and the share "
     "expected if the two frames had nothing to do with each other."),
    ("motion", "stay_and_once", "Staying and seen once", draw_stay_and_once,
     "The mass beyond the biofilm in two parts: at the same place in the next frame, and not there in the next frame."),
    ("motion", "object_size", "Size of one object", draw_object_size,
     "Areas of the dark objects beyond the cloud in the sharp frames. The median is used as the area of one cell. "
     "The largest 3 % are off the scale."),
    ("checks", "object_counts", "Objects inside and outside", draw_object_counts,
     "Dark objects with their centre inside and outside the outline of each frame."),
    ("checks", "focus", "Focus of each frame", draw_focus,
     "Sharpness of each frame. Frames below the dashed line are out of focus: their points are grey and left out of "
     "the curves."),
    ("checks", "body_area", "Area of the biofilm", draw_body_area,
     "Area inside the outline of each frame, as a share of the first sharp frame."),
]
NOT_TIME_KEYS = ("occupancy", "object_size")
TIME_PANEL_KEYS = tuple(p[1] for p in PANELS if p[1] not in NOT_TIME_KEYS)
MAP_PANEL_KEYS = ("distance_map", "exit_map")       # time panels drawn as maps: their cursor is dark and on top


@SP.with_house_style
def draw_panel(fig, panel, results, span, cursor_h, settings=None, cell=None):
    group, key, title, fn, _ = panel
    if results is None:
        return SP.grey_panel(fig, title, RUN_MEASURE, cell)
    return fn(fig, results, span if span is not None else results.span, cursor_h, settings, cell=cell)


def panels_of_group(group: str) -> list:
    return [p for p in PANELS if p[0] == group]


def panel_by_key(key: str):
    return next(p for p in PANELS if p[1] == key)


# ---------------------------------------------------------------------------------------------- export
def new_figure(size=PANEL_SIZE_IN, dpi: int = 100):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=size, dpi=dpi, facecolor="white")
    FigureCanvasAgg(fig)
    return fig


def save_panels(folder: str, results, settings=None, dpi: int = 300) -> list:
    """Every panel as a PNG and a PDF in folder. Returns the paths written."""
    import os
    os.makedirs(folder, exist_ok=True)
    out = []
    for panel in PANELS:
        fig = new_figure()
        draw_panel(fig, panel, results, results.span, None, settings)
        for ext in ("png", "pdf"):
            path = os.path.join(folder, f"{panel[1]}.{ext}")
            fig.savefig(path, dpi=dpi); out.append(path)
    return out
