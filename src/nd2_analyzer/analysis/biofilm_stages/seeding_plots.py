"""Drawing for the Seeding page: the stage strip, the gallery panels (PANELS, draw_panel), the film strip of one followed
cell, the report pages and the empty state. Pure matplotlib on a given Figure, numpy only, so the page, the image export
and the tests draw the same thing.

House style (copied from headless_analysis film_render, not imported): Arial, bold titles and axis labels, open frame,
ticks out, no grid. Colours: Okabe-Ito blue for the main curve, green and grey for support, black for data points, red
only for "divides". Fate colours: leave grey, stay blue, divides red. Nothing within 0.1 in of anything else; collisions(fig)
checks that and the tests assert it is empty.

Every panel is one function draw_<name>(fig, results, span, cursor_h, settings, cell=None) -> ax. With cell None it owns
the whole figure (one gallery canvas, about 3.6 x 2.6 in); with cell = (x0, y0, w, h) in figure fractions it draws
inside that cell of a report page. A panel whose metric the recording cannot give draws its title and the reason in grey."""
from __future__ import annotations

import textwrap
from typing import Optional, Sequence

import numpy as np

BLUE, GREEN, GREY, INK, RED = "#0072B2", "#009E73", "#7f7f7f", "#1a1a1a", "#cc3311"
FATE = {"leave": "#9a9a9a", "stay": BLUE, "divides": RED,
        # what happened to each landing (rules.fate_class_expr), the colours of the Seeding window and the six figures
        "left": "#b9bec3", "while": "#56B4E9", "end": "#009E73", "divided": "#c0392b"}
STAGE_SHADE = {"seeding": "#f0f0f0", "irreversible": "#e2e2e2", "microcolonies": "#d2d2d2"}
STAGE_LABEL = {"seeding": "reversible attachment", "irreversible": "irreversible attachment", "microcolonies": "microcolonies"}
STAGE_SHORT = {"seeding": "reversible", "irreversible": "irreversible", "microcolonies": "microcolonies"}   # for a narrow band
EMPTY_GREY = "#777777"
ARIAL = ["Arial", "Helvetica", "DejaVu Sans"]
MK = dict(marker="o", ms=4.0, mfc="white", mew=1.4)


# ---------------------------------------------------------------------------------------------- style and the check
def house_rc(scale: float = 1.0) -> dict:
    """Partaker 2's publication figure rules sized for a screen canvas: Arial, bold titles and axis labels, no grid,
    open frame, ticks out. scale > 1 for export."""
    k = float(scale)
    return {"font.family": ARIAL, "font.sans-serif": ARIAL, "axes.grid": False,
            "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.2 * k,
            "lines.linewidth": 2.0 * k, "font.size": 10 * k, "axes.titlesize": 11 * k, "axes.titleweight": "bold",
            "axes.labelsize": 10.5 * k, "axes.labelweight": "bold", "xtick.labelsize": 9.5 * k, "ytick.labelsize": 9.5 * k,
            "legend.fontsize": 9.5 * k, "legend.frameon": False,
            "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 4 * k, "ytick.major.size": 4 * k,
            "xtick.major.width": 1.2 * k, "ytick.major.width": 1.2 * k, "xtick.major.pad": 4 * k, "ytick.major.pad": 4 * k,
            "axes.titlepad": 8 * k, "axes.labelpad": 6 * k, "pdf.fonttype": 42, "ps.fonttype": 42,
            "figure.facecolor": "white", "axes.facecolor": "white"}


def collisions(fig, gap_in: float = 0.1, gap_same: float = 0.03):
    """Pairs of figure elements (texts, tick labels, axis titles, legends, images, plots) closer than gap_in inches: a
    publication figure has none. Labels of one plot may sit gap_same from each other."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); dpi = fig.dpi; items = []

    def add(name, art, kind, owner=None):
        if not art.get_visible() or (hasattr(art, "get_text") and not art.get_text().strip()):
            return
        bb = art.get_window_extent(r); items.append((name, (bb.x0 / dpi, bb.y0 / dpi, bb.x1 / dpi, bb.y1 / dpi), kind, owner))

    for t in fig.texts:
        add(f"text '{t.get_text()[:30]}'", t, "text")
    for ax in fig.axes:
        owner = twin_owner(fig, ax)
        kind = "img" if ax.images or ax.get_facecolor()[:3] == (0.0, 0.0, 0.0) or not ax.axison else "plot"
        add(f"{kind} '{ax.get_title(loc='left') or ax.get_title()}'", ax, kind, owner)
        ax = _Owned(ax, owner)
        if kind == "img":
            add(f"title '{ax.get_title(loc='left')[:30]}'", ax._left_title, "label", ax.owner)
            add(f"title '{ax.get_title(loc='right')[:30]}'", ax._right_title, "label", ax.owner)
            for t in ax.texts:
                add(f"call-out '{t.get_text()[:30]}'", t, "own", ax.owner)
            continue
        for t in ax.texts:
            add(f"call-out '{t.get_text()[:30]}'", t, "own", ax.owner)
        if ax.get_legend():
            add("legend", ax.get_legend(), "own", ax.owner)
        for t in (ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label):
            add(f"title '{t.get_text()[:30]}'", t, "label", ax.owner)
        # tick labels that matplotlib keeps outside the axis range are clipped away, and the labels of a hidden axis
        # (the x axis of a twin) are never drawn: neither counts
        x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim())
        for pos, t in zip(ax.get_xticks(), ax.get_xticklabels()):
            if x0 <= pos <= x1 and ax.xaxis.get_visible(): add(f"tick '{t.get_text()}'", t, "label", ax.owner)
        for pos, t in zip(ax.get_yticks(), ax.get_yticklabels()):
            if y0 <= pos <= y1 and ax.yaxis.get_visible(): add(f"tick '{t.get_text()}'", t, "label", ax.owner)
    out = []
    for i in range(len(items)):
        n1, b1, k1, o1 = items[i]
        for j in range(i + 1, len(items)):
            n2, b2, k2, o2 = items[j]
            same = o1 is not None and o1 is o2
            if same and ("plot" in (k1, k2) or "img" in (k1, k2)):
                continue                                   # a plot's own labels sit inside or beside it
            m = gap_same if same else gap_in
            if b1[0] < b2[2] + m and b2[0] < b1[2] + m and b1[1] < b2[3] + m and b2[1] < b1[3] + m:
                out.append(f"{n1}  x  {n2}")
    return out


class _Owned:
    """An axes seen through its owner for the collision check: a twin axes (a second y axis on the same plot) counts as
    part of the axes it shares its place with."""

    def __init__(self, ax, owner):
        self._ax = ax; self.owner = owner

    def __getattr__(self, name):
        return getattr(self._ax, name)


def twin_owner(fig, ax):
    """The axes that owns this one in the collision check: the first axes of the figure that sits at the same place and
    shares the x axis with it (ax itself when there is none)."""
    siblings = ax.get_shared_x_axes().get_siblings(ax)
    for other in fig.axes:
        if other is ax:
            return ax
        if other in siblings and np.allclose(other.get_position().bounds, ax.get_position().bounds):
            return other
    return ax


# ---------------------------------------------------------------------------------------------- small helpers
def metric_by_key(results, key: str):
    """The Metric with this key from a SeedingResults, or None."""
    if results is None:
        return None
    for m in results.metrics:
        if m.key == key:
            return m
    return None


def series_of(results, key: str) -> dict:
    """The series dict of an available metric, else an empty dict."""
    m = metric_by_key(results, key)
    if m is None or not m.available or not m.series:
        return {}
    return m.series


def hour_ticks(span: Sequence[float]) -> np.ndarray:
    """Round hour ticks for a span: every 0.5, 1, 2, 3, 6, 12 or 24 h, then whole days, so that there are at most 12."""
    a, b = float(span[0]), float(span[1])
    width = max(b - a, 1e-6)
    step = 0.5
    for step in (0.5, 1, 2, 3, 6, 12, 24):
        if width / step <= 12:
            break
    while width / step > 12:
        step *= 2
    first = np.ceil(a / step) * step
    return np.arange(first, b + 1e-9, step)


def recording_span(results, span: Optional[Sequence[float]]) -> tuple:
    """The x range of every plot: the recording's span when given, else the hours of the counts table."""
    if span is not None:
        return float(span[0]), float(span[1])
    if results is not None and results.counts is not None and results.counts.height:
        h = results.counts["hour"].to_numpy()
        return float(h.min()), float(h.max())
    return 0.0, 1.0


def pad_span(span: Sequence[float], frac: float = 0.02) -> tuple:
    a, b = float(span[0]), float(span[1]); pad = (b - a) * frac
    return a - pad, b + pad


def open_frame(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def empty_axes(fig, text: str):
    """One grey sentence on an otherwise blank figure."""
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=10, color=EMPTY_GREY, transform=ax.transAxes)
    return ax


def with_house_style(fn):
    """Run a drawing function inside the house rc context."""
    import functools
    import matplotlib

    @functools.wraps(fn)
    def inner(*args, **kw):
        with matplotlib.rc_context(house_rc()):
            return fn(*args, **kw)
    return inner


# ---------------------------------------------------------------------------------------------- the strip
@with_house_style
def draw_strip(ax, span: Sequence[float], cursor_h: float, stage_hours=None, stages=None):
    """The recording's span in hours after inoculation, the frames shaded seeding / microcolonies (two light greys) once
    results exist, a red cursor line with its hour, and hour ticks. stage_hours and stages are the per-frame hour and
    stage name arrays of results.stage_by_frame, or None."""
    from matplotlib.patches import Rectangle
    a, b = float(span[0]), float(span[1])
    x0, x1 = pad_span((a, b), 0.01)
    ax.set_xlim(x0, x1); ax.set_ylim(0, 1); ax.set_yticks([])
    ticks = hour_ticks((a, b))
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g} h" for t in ticks])
    ax.tick_params(axis="x", labelsize=8.5, length=3)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    band_y, band_h = 0.02, 0.56
    c = float(min(max(cursor_h, a), b))
    hidden = None                                   # the stage under the cursor when its band could not show its name
    if stage_hours is not None and stages is not None and len(stage_hours):
        for name, lo, hi in stage_runs(np.asarray(stage_hours, float), list(stages), b):
            ax.add_patch(Rectangle((lo, band_y), hi - lo, band_h, color=STAGE_SHADE.get(name, "#e4e4e4"), ec="none"))
            placed = False
            for word in stage_words(name):
                if "\n" in word and not label_fits_band(ax, word, band_h):
                    continue
                x = label_x_clear_of_cursor(ax, lo, hi, word, c)
                if x is not None:
                    ax.text(x, band_y + band_h / 2, word, ha="center", va="center", fontsize=8, color=INK, linespacing=1.0)
                    placed = True
                    break
            if not placed and lo <= c <= hi:
                hidden = name
    else:
        ax.add_patch(Rectangle((a, band_y), b - a, band_h, color="#f2f2f2", ec="none"))
    ax.axvline(c, color=RED, lw=1.8)
    hour = f"{c:.1f} h"
    stage = STAGE_LABEL.get(hidden, hidden) if hidden else ""
    near_right = (c - a) > 0.85 * (b - a)
    if stage and not near_right:                    # the hour and the stage name together must fit on the right
        near_right = c + 0.006 * (b - a) + label_width_h(ax, hour, 8.5) + label_width_h(ax, stage, 8.5) > ax.get_xlim()[1]
    side = -1 if near_right else 1
    x_hour = c + side * 0.006 * (b - a)
    ax.text(x_hour, 0.70, hour, ha="right" if near_right else "left", va="bottom", fontsize=8.5, color=RED, fontweight="bold")
    if stage:                                       # the name its band had no room for, beside the hour
        ax.text(x_hour + side * label_width_h(ax, hour, 8.5), 0.70, stage, ha="right" if near_right else "left",
                va="bottom", fontsize=8.5, color=INK)
    return ax


def stage_words(name: str) -> list:
    """The names to try in a stage band, longest first: the stage in full, the same on two lines, the short word."""
    full = STAGE_LABEL.get(name, name)
    out = [full] + ([full.replace(" ", "\n", 1)] if " " in full else [])
    short = STAGE_SHORT.get(name, full)
    return out + ([short] if short not in out else [])


def label_fits_band(ax, word: str, band_h: float, size: float = 8.0) -> bool:
    """True when the word, on as many lines as it has, fits inside a band band_h of the axes high (a small margin kept)."""
    fig = ax.figure
    try:
        from matplotlib.text import Text
        t = Text(0, 0, word, fontsize=size, linespacing=1.0, figure=fig)
        h_in = t.get_window_extent(renderer=fig.canvas.get_renderer()).height / fig.dpi
    except Exception:
        h_in = 1.2 * size / 72.0 * (word.count("\n") + 1)
    return h_in + 0.02 <= band_h * fig.get_size_inches()[1] * ax.get_position().height


def stage_runs(hours: np.ndarray, stages: list, end_h: float) -> list:
    """Consecutive frames with the same stage, as (name, first_hour, next_hour) blocks."""
    out = []
    if not len(hours):
        return out
    start, cur = hours[0], stages[0]
    for h, s in zip(hours[1:], stages[1:]):
        if s != cur:
            out.append((cur, float(start), float(h))); start, cur = h, s
    out.append((cur, float(start), float(end_h)))
    return out


def hours_per_inch(ax) -> float:
    fig = ax.figure
    x_in = fig.get_size_inches()[0] * ax.get_position().width
    xa, xb = ax.get_xlim()
    return (xb - xa) / max(x_in, 1e-9)


def label_width_h(ax, name: str, size: float = 8.0) -> float:
    """The width of a band name at 8 pt in hours (its longest line), measured with the figure's renderer (a small margin
    added)."""
    fig = ax.figure
    try:
        from matplotlib.text import Text
        t = Text(0, 0, name, fontsize=size, linespacing=1.0, figure=fig)
        width_in = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi + 0.12
    except Exception:
        width_in = 0.075 * len(name) + 0.3
    return width_in * hours_per_inch(ax)


def band_fits_label(ax, lo: float, hi: float, name: str) -> bool:
    """True when the band is wide enough for its name."""
    return (hi - lo) > label_width_h(ax, name)


def label_x_clear_of_cursor(ax, lo: float, hi: float, name: str, cursor_h: float):
    """Where to write the band name so the cursor line does not cross it: the band centre, else the centre of the wider
    side of the cursor; None when nothing fits."""
    if not band_fits_label(ax, lo, hi, name):
        return None
    w = label_width_h(ax, name); mid = (lo + hi) / 2
    if abs(cursor_h - mid) > w / 2 + 0.1 * hours_per_inch(ax):
        return mid
    left, right = (lo, cursor_h), (cursor_h, hi)
    side = max((left, right), key=lambda p: p[1] - p[0])
    if side[1] - side[0] > w + 0.2 * hours_per_inch(ax):
        return (side[0] + side[1]) / 2
    return None


# ---------------------------------------------------------------------------------------------- the two bottom plots
@with_house_style
def draw_counts(fig, results, span=None, ax=None):
    """Cells on the surface vs hour, the lag fit dashed, the take-off hour as a labelled vertical dashed line. With ax given
    (the export figure) the plot is drawn into that axes and the figure is left alone."""
    if ax is None:
        fig.clear()
    if results is None:
        return empty_axes(fig, "Run Measure to see this plot")
    s = series_of(results, "cells_on_surface")
    if not s:
        return empty_axes(fig, reason_line(results, "cells_on_surface", "no cells per frame to show"))
    ax = ax if ax is not None else fig.add_axes([0.15, 0.24, 0.80, 0.63])
    a, b = recording_span(results, span)
    hour = np.asarray(s["hour"], float); cells = np.asarray(s["cells"], float)
    ax.plot(hour, cells, color=BLUE, lw=1.6, label="cells per frame")
    draw_lag_fit(ax, results, hour)
    draw_take_off(ax, results, (a, b))
    ax.set_xlim(*pad_span((a, b)))
    ax.set_ylim(0, max(float(np.nanmax(cells)) * 1.12, 1.0))
    ax.set_xticks(hour_ticks((a, b)))
    ax.set_title("Cells on the surface", loc="left")
    ax.set_xlabel("time after inoculation (h)"); ax.set_ylabel("cells")
    open_frame(ax)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="upper left", handlelength=1.6)
    return ax


def draw_lag_fit(ax, results, hour: np.ndarray):
    """N(t) = N0 exp((t - t_lag) / tau) dashed grey, from the lag_time metric's value and series (tau_h, n0)."""
    m = metric_by_key(results, "lag_time")
    if m is None or not m.available or m.value is None or not m.series:
        return
    s = m.series
    tau = s.get("tau_h", s.get("tau")); n0 = s.get("n0", s.get("N0"))
    if tau is None or n0 is None or float(tau) <= 0:
        return
    if s.get("fit") is not None and s.get("hour") is not None and len(s["fit"]) == len(s["hour"]):
        t = np.asarray(s["hour"], float); fit = np.asarray(s["fit"], float)       # the rules module's own fitted curve
    else:
        t = np.linspace(float(hour.min()), float(hour.max()), 200)
        fit = float(n0) * np.exp((t - float(m.value)) / float(tau))
    top = float(np.nanmax(ax.lines[0].get_ydata())) * 1.05
    keep = fit <= top
    doubling_h = float(tau) * np.log(2.0)
    ax.plot(t[keep], fit[keep], color=GREY, lw=1.4, ls="--",
            label=f"lag {float(m.value):.1f} h,\ndoubles every {doubling_h:.1f} h")


def draw_take_off(ax, results, span):
    """The take-off hour as a vertical dashed line with its label beside it: on the left of the line, over the flat
    part of the count, unless the line sits in the first third of the span."""
    m = metric_by_key(results, "take_off_hour")
    if m is None or not m.available or m.value is None:
        return
    h = float(m.value); a, b = span
    ax.axvline(h, color=INK, lw=1.2, ls="--")
    near_left = (h - a) < 0.3 * (b - a)
    ax.text(h + (1 if near_left else -1) * 0.012 * (b - a), 0.55, f"take-off {h:.1f} h", transform=ax.get_xaxis_transform(),
            ha="left" if near_left else "right", va="center", fontsize=9, color=INK)


@with_house_style
def draw_stay(fig, results, span=None, stay_window_s: Optional[float] = None, ax=None):
    """% still there after the stay window by landing hour with an SE band, or the residence histogram when the
    stay-by-hour rule is unavailable. With ax given the plot is drawn into that axes."""
    if ax is None:
        fig.clear()
    if results is None:
        return empty_axes(fig, "Run Measure to see this plot")
    window = stay_window_s if stay_window_s is not None else float(results.settings.stay_window_s)
    s = series_of(results, "stay_by_landing_hour")
    if s:
        return draw_stay_by_hour(fig, s, recording_span(results, span), window, ax,
                                 long=series_of(results, "irreversible_begins"))
    h = series_of(results, "residence_median")
    if h:
        return draw_residence_histogram(fig, h, window, ax)
    return empty_axes(fig, reason_line(results, "stay_by_landing_hour", "no stay or residence data to show"))


def draw_stay_by_hour(fig, s: dict, span, window_s: float, ax=None, long: Optional[dict] = None):
    """Two lines by landing hour: still there after the short window (reversible test) and, when the commitment rule
    ran, still there after the long window (attached for good). The gap between them is the switch."""
    ax = ax if ax is not None else fig.add_axes([0.15, 0.24, 0.80, 0.63])
    hour = bin_centres(np.asarray(s["hour"], float)); pct = np.asarray(s["pct"], float)
    se = np.asarray(s.get("se", np.zeros_like(pct)), float)
    ax.fill_between(hour, pct - se, pct + se, color=BLUE, alpha=0.18, lw=0)
    ax.plot(hour, pct, color=BLUE, lw=1.6, **MK, mec=INK, label=f"after {time_words(window_s)}")
    title = f"Still there after {time_words(window_s)}, by landing hour"
    if long and long.get("hour"):
        lh = bin_centres(np.asarray(long["hour"], float)); lp = np.asarray(long["pct"], float)
        lse = np.asarray(long.get("se", np.zeros_like(lp)), float)
        ax.fill_between(lh, lp - lse, lp + lse, color=GREEN, alpha=0.18, lw=0)
        ax.plot(lh, lp, color=GREEN, lw=1.6, marker="s", ms=3.6, mfc="white", mew=1.3, mec=INK,
                label=f"after {time_words(float(long.get('stay_s', 3600)))} (for good)")
        title = f"Still there after {time_words(window_s)} and after {time_words(float(long.get('stay_s', 3600)))}, by landing hour"
    a, b = span
    ax.set_xlim(*pad_span((a, b))); ax.set_ylim(0, 105)
    ax.set_xticks(hour_ticks((a, b))); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(title, loc="left")
    ax.set_xlabel("landing time after inoculation (h)"); ax.set_ylabel("% of landings")
    open_frame(ax)
    ax.legend(loc="upper right", handlelength=1.6)
    return ax


def bin_centres(hour: np.ndarray) -> np.ndarray:
    """The rules bin landings by the floor of their hour: whole-hour values are bins and are drawn at their centres;
    anything else is already a position and is left alone."""
    if hour.size and np.allclose(hour, np.round(hour)):
        return hour + 0.5
    return hour


def draw_residence_histogram(fig, s: dict, window_s: float, ax=None, note: str = ""):
    """The residence histogram on log-spaced time bins, the stay window dashed; note is the number written inside
    (for example "median 6 s")."""
    ax = ax if ax is not None else fig.add_axes([0.15, 0.24, 0.80, 0.63])
    edges = np.asarray(s["edges_s"], float); counts = np.asarray(s["counts"], float)
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=BLUE, ec="white", lw=0.6)
    ax.set_xscale("log"); ax.set_xlim(float(edges[0]) / 1.02, float(edges[-1]) * 1.2)   # room for the last edge's word
    set_time_ticks(ax, float(edges[0]), float(edges[-1]))
    if edges[0] < window_s < edges[-1]:                 # frames longer than the window: the line has no place here
        ax.axvline(window_s, color=INK, lw=1.2, ls="--")
        ax.text(window_s * 1.15, 0.92, time_words(window_s), transform=ax.get_xaxis_transform(), ha="left", va="top",
                fontsize=9, color=INK)
    if note:
        ax.text(0.98, 0.95, note, transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
    ax.text(0.98, 0.78 if note else 0.95, "longer stays in the last bar", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=GREY)
    ax.set_title("How long landed cells stayed", loc="left")
    ax.set_xlabel("time on the surface"); ax.set_ylabel("landings")
    open_frame(ax)
    return ax


def time_words(seconds: float) -> str:
    """A time as it is read: 3 s, 1 min, 40 min, 2 h, 1.4 h."""
    s = float(seconds)
    if s < 60:
        return f"{_short(s)} s"
    if s < 3600:
        return f"{_short(s / 60)} min"
    return f"{_short(s / 3600)} h"


def _short(v: float) -> str:
    """A number with at most one decimal and no trailing zero: 3, 1.4, 40."""
    return f"{v:.1f}".rstrip("0").rstrip(".") if abs(v) < 100 else f"{v:,.0f}"


TIME_TICKS_S = (3.0, 10.0, 60.0, 600.0, 3600.0, 36000.0)   # 3 s, 10 s, 1 min, 10 min, 1 h, 10 h


def set_time_ticks(ax, lo_s: float, hi_s: float, axis: str = "x", wanted=None):
    """Plain time words on a log time axis instead of powers of ten. By default the round words (10 s, 1 min, 10 min,
    1 h, 10 h) and then the two edges (the histogram); wanted gives another list, for example TIME_TICKS_S for the landing
    dots. A tick closer than one e-fold to one already placed is dropped, earlier entries winning (so 1 h beats a 30 min
    edge)."""
    from matplotlib.ticker import NullFormatter
    wanted = list(TIME_TICKS_S[1:]) + [lo_s, hi_s] if wanted is None else list(wanted)
    ticks = []
    for t in wanted:
        if lo_s <= t <= hi_s and all(abs(np.log(t / u)) > 1.0 for u in ticks):
            ticks.append(t)
    which = ax.xaxis if axis == "x" else ax.yaxis
    which.set_ticks(ticks); which.set_ticklabels([time_words(t) for t in ticks])
    which.set_minor_formatter(NullFormatter())


def reason_line(results, key: str, fallback: str) -> str:
    m = metric_by_key(results, key)
    if m is not None and m.reason:
        return m.reason
    return fallback


# ---------------------------------------------------------------------------------------------- the film strip
@with_house_style
def draw_film(fig, frames: Sequence[np.ndarray], hours: Sequence[float], trail: Sequence, fates: Sequence[str],
              pixel_um: Optional[float] = None, max_crops: int = 6):
    """4 to 6 crops of the followed cell side by side: the trail up to that frame and a dot in the fate colour, the hour in
    the corner of each crop, a 2 um scale bar on the first. trail is one (x, y) per crop in crop pixels."""
    fig.clear()
    if not frames:
        return empty_axes(fig, "Pick a cell and click Follow to see its film")
    n = min(len(frames), max_crops)
    picks = np.linspace(0, len(frames) - 1, n).round().astype(int) if len(frames) > n else np.arange(n)
    W, H = fig.get_size_inches()
    gap_in = 0.12; side = 0.06
    w_in = (W - 2 * side - gap_in * (n - 1)) / n
    h_frac = min(0.84, w_in / H)
    w_frac = w_in / W
    axes = []
    for k, i in enumerate(picks):
        left = (side + k * (w_in + gap_in)) / W
        ax = fig.add_axes([left, (1 - h_frac) / 2, w_frac, h_frac])
        draw_crop(ax, np.asarray(frames[i]), hours[i] if i < len(hours) else None, [trail[j] for j in picks[: k + 1] if j < len(trail)],
                  fates[i] if i < len(fates) else "stay", pixel_um if k == 0 else None)
        axes.append(ax)
    return axes


def draw_crop(ax, crop: np.ndarray, hour, trail: list, fate: str, pixel_um: Optional[float]):
    colour = FATE.get(fate, GREY)
    ax.imshow(crop, cmap="gray", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(colour); sp.set_linewidth(1.6)
    if trail:
        xs = [p[0] for p in trail]; ys = [p[1] for p in trail]
        if len(trail) > 1:
            ax.plot(xs, ys, color=colour, lw=1.2, alpha=0.9)
        ax.plot(xs[-1], ys[-1], marker="o", ms=4, color=colour, mec="white", mew=0.8)
    if hour is not None:
        ax.text(0.06, 0.94, f"{float(hour):.1f} h", transform=ax.transAxes, ha="left", va="top", fontsize=7, color="white")
    if pixel_um:
        draw_scale_bar(ax, crop.shape, pixel_um, 2.0)
    ax.set_xlim(-0.5, crop.shape[1] - 0.5); ax.set_ylim(crop.shape[0] - 0.5, -0.5)


def draw_scale_bar(ax, shape, pixel_um: float, length_um: float):
    px = length_um / float(pixel_um)
    h, w = shape
    x0, y = 0.06 * w, 0.90 * h
    ax.plot([x0, x0 + px], [y, y], color="white", lw=2, solid_capstyle="butt")
    ax.text(x0, y - 0.04 * h, f"{length_um:g} µm", ha="left", va="bottom", fontsize=6.5, color="white")


# ---------------------------------------------------------------------------------------------- the gallery panels
PANEL_SIZE_IN = (3.6, 2.6)                    # one gallery canvas
PANEL_RECT = [0.20, 0.22, 0.75, 0.63]         # the axes inside a panel, as fractions of its figure or of its report cell
PANEL_RECT_TWIN = [0.20, 0.22, 0.60, 0.63]    # when a second y axis needs room at the right
PANEL_RECT_WIDE_LEFT = [0.25, 0.22, 0.70, 0.63]   # when the y tick labels are time words ("10 min")
GROUP_TITLES = (("arrivals", "Reversible attachment: arrivals and leaving"), ("commitment", "Irreversible attachment"),
                ("movement", "Movement on the surface"), ("division", "Division and families"), ("flow", "Flow"))
GROUP_HEADING = dict(GROUP_TITLES)
RUN_MEASURE = "Run Measure to see this plot"
NO_TRACKS = "no tracks for this position"
IN_PER_PT_CHAR = 0.0072                       # about the width of one Arial character per point of font size, for wrapping
LANDING_DOTS_KEY = "landing_dots"
TIME_PANEL_KEYS = ("cells_on_surface", "landings", "stay_by_landing_hour", LANDING_DOTS_KEY, "irreversible_begins",
                   "division_count")           # the gallery panels whose x axis is time; the six big figures carry no cursor           # the panels whose x axis is time after inoculation: they show the cursor


def panel_settings(results, settings):
    """The settings the panel reads its thresholds from: the ones given, else the results', else the defaults."""
    if settings is not None:
        return settings
    if results is not None and getattr(results, "settings", None) is not None:
        return results.settings
    from nd2_analyzer.analysis.biofilm_stages.seeding_types import SeedingSettings
    return SeedingSettings()


def cell_rect(cell, rect) -> list:
    """rect (fractions of a cell) placed inside cell = (x0, y0, w, h), both in figure fractions."""
    x0, y0, w, h = cell
    return [x0 + rect[0] * w, y0 + rect[1] * h, rect[2] * w, rect[3] * h]


def panel_axes(fig, cell=None, rect=None):
    """A fresh axes for one panel: the whole figure when cell is None (a gallery canvas, cleared first), else inside that
    cell of a report page. rect is the axes rectangle as fractions of the panel (PANEL_RECT by default)."""
    rect = PANEL_RECT if rect is None else rect
    if cell is None:
        fig.clear()
        return fig.add_axes(rect)
    return fig.add_axes(cell_rect(cell, rect))


def grey_panel(fig, title: str, reason: str, cell=None):
    """An unavailable panel: the title where a plot's title would be and one grey sentence, wrapped to the panel."""
    ax = panel_axes(fig, cell); ax.axis("off")
    ax.set_title(title, loc="left")
    w_in = fig.get_size_inches()[0] * ax.get_position().width
    lines = textwrap.wrap(reason, max(int(w_in / (9.5 * IN_PER_PT_CHAR)), 12))
    ax.text(0.5, 0.5, "\n".join(lines), ha="center", va="center", fontsize=9.5, color=EMPTY_GREY, transform=ax.transAxes,
            linespacing=1.4)
    return ax


def unavailable_reason(results, key: str) -> str:
    """Why a panel keyed on this metric cannot draw: no results, or the metric is unavailable. Empty when it can."""
    if results is None:
        return RUN_MEASURE
    m = metric_by_key(results, key)
    if m is not None and not m.available:
        return m.reason or "not available"
    return ""


def draw_cursor(ax, cursor_h, span):
    """The cursor hour as a thin grey line when it is inside the span."""
    if cursor_h is None:
        return
    a, b = float(span[0]), float(span[1])
    if a <= float(cursor_h) <= b:
        ax.axvline(float(cursor_h), color=GREY, lw=1.0, zorder=0.5)


def finish_time_axis(ax, span, xlabel: str = "time after inoculation (h)"):
    a, b = float(span[0]), float(span[1])
    ax.set_xlim(*pad_span((a, b))); ax.set_xticks(hour_ticks((a, b)))
    ax.set_xlabel(xlabel)


def floats(pt, name: str) -> np.ndarray:
    """One numeric per_track column as floats, nulls as nan."""
    return np.asarray([np.nan if v is None else float(v) for v in pt[name].to_list()], dtype=float)


def landed_rows(results):
    """The per_track rows of landed cells, or None when there is no table."""
    pt = getattr(results, "per_track", None)
    if pt is None or pt.height == 0 or "origin" not in pt.columns:
        return None
    return pt.filter(pt["origin"] == "landed")


def pixel_um_of(results):
    """The pixel size the rules used, read back from a track with both a pixel and a micron shift; None without one."""
    pt = getattr(results, "per_track", None)
    if pt is None or pt.height == 0 or "net_disp_um" not in pt.columns:
        return None
    px = floats(pt, "net_disp_px"); um = floats(pt, "net_disp_um")
    ok = np.isfinite(px) & np.isfinite(um) & (px > 0)
    return float(np.median(um[ok] / px[ok])) if ok.any() else None


def count_label(ax, x: float, y: float, text: str, size: float = 9, color=INK, va="top", ha="right"):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=size, color=color)


def integer_yticks(ax):
    """Whole numbers on an axis that counts things."""
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))


def counts_on_bars(ax, xs, heights, top: float):
    """The count above each bar; top is the axis top so the labels sit inside."""
    for x, h in zip(xs, heights):
        ax.text(x, h + 0.02 * top, f"{int(h):,}", ha="center", va="bottom", fontsize=9, color=INK)


def hist_bins(lo: float, hi: float, n: int = 24) -> np.ndarray:
    return np.linspace(lo, hi, n + 1)


def clip_top(values: np.ndarray, hi: float) -> np.ndarray:
    """Values above hi fall in the last bar."""
    return np.minimum(values, hi - 1e-9)


# ---- Arrivals and leaving
@with_house_style
def draw_cells_on_surface(fig, results, span, cursor_h, settings, cell=None):
    """Panel 1: counts vs hour, lag fit dashed, take-off dashed and labelled (Lee Fig. 2B)."""
    title = "Cells on the surface"
    why = unavailable_reason(results, "cells_on_surface")
    if not why and not series_of(results, "cells_on_surface"):
        why = "no cells per frame to show"
    if why:
        return grey_panel(fig, title, why, cell)
    ax = panel_axes(fig, cell)
    draw_counts(fig, results, span, ax=ax)
    ax.set_title(title, loc="left")
    draw_cursor(ax, cursor_h, recording_span(results, span))
    return ax


@with_house_style
def draw_landings_per_hour(fig, results, span, cursor_h, settings, cell=None):
    """Panel 2: landings by landing hour as bars (Lecuyer Fig. 2A)."""
    title = "Landings per hour"
    why = unavailable_reason(results, "landings")
    if why:
        return grey_panel(fig, title, why, cell)
    s = series_of(results, "landings")
    if s and s.get("hour") is not None:
        hour = np.asarray(s["hour"], float); n = np.asarray(s["landings"], float)
    else:
        landed = landed_rows(results)
        if landed is None or landed.height == 0:
            return grey_panel(fig, title, "no landing in these frames", cell)
        t = floats(landed, "t_start_h"); edges = np.arange(np.floor(t.min()), np.floor(t.max()) + 2)
        n, _ = np.histogram(t, bins=edges); hour = edges[:-1]
    ax = panel_axes(fig, cell)
    a, b = recording_span(results, span)
    ax.bar(hour, n, width=1.0, align="edge", color=BLUE, ec="white", lw=0.6)
    draw_cursor(ax, cursor_h, (a, b))
    finish_time_axis(ax, (a, b), "landing time after inoculation (h)")
    ax.set_ylim(0, max(float(n.max()) * 1.25, 1.0)); integer_yticks(ax)
    m = metric_by_key(results, "landings")
    count_label(ax, 0.98, 0.95, m.text if m is not None and m.available else f"{int(n.sum()):,} landings")
    ax.set_title(title, loc="left"); ax.set_ylabel("landings")
    open_frame(ax)
    return ax


@with_house_style
def draw_how_long_stayed(fig, results, span, cursor_h, settings, cell=None):
    """Panel 3: the residence histogram on log time bins with the median written inside (Lee Fig. 1B)."""
    title = "How long landed cells stayed"
    why = unavailable_reason(results, "residence_median")
    if why:
        return grey_panel(fig, title, why, cell)
    s = series_of(results, "residence_median")
    if not s:
        return grey_panel(fig, title, "no landing with a known end", cell)
    S = panel_settings(results, settings)
    m = metric_by_key(results, "residence_median")
    note = f"median {time_words(m.value)}" if m is not None and m.value is not None else ""
    ax = panel_axes(fig, cell)
    draw_residence_histogram(fig, s, float(S.stay_window_s), ax, note=note)
    ax.set_title(title, loc="left")
    return ax


@with_house_style
def draw_share_still_attached(fig, results, span, cursor_h, settings, cell=None):
    """Panel 4: the survival curve f(t) on a log time axis, the stretched exponential fit dashed when the series has
    tau_s and alpha (Lecuyer Fig. 2B, Eq. 4)."""
    title = "Share still attached after t"
    why = unavailable_reason(results, "survival_curve")
    if why:
        return grey_panel(fig, title, why, cell)
    s = series_of(results, "survival_curve")
    if not s or not s.get("t_s"):
        return grey_panel(fig, title, "no leavers to follow", cell)
    t = np.asarray(s["t_s"], float); f = np.asarray(s["f"], float)
    keep = t > 0
    if not keep.any():
        return grey_panel(fig, title, "no leavers to follow", cell)
    ax = panel_axes(fig, cell)
    lo, hi = float(t[keep].min()), float(t.max())
    ax.plot(t[keep], f[keep], color=BLUE, lw=1.6, drawstyle="steps-post", label="measured")
    if s.get("tau_s") is not None and s.get("alpha") is not None:
        tt = np.geomspace(lo, hi, 200)
        ax.plot(tt, np.exp(-(tt / float(s["tau_s"])) ** float(s["alpha"])), color=GREY, lw=1.4, ls="--",
                label=f"fit: tau {time_words(float(s['tau_s']))}, alpha {float(s['alpha']):.2f}")
        ax.legend(loc="upper right", handlelength=1.6)
    ax.axhline(0.5, color=GREY, lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xlim(lo, hi * 1.05); ax.set_ylim(0, 1.08)
    set_time_ticks(ax, lo, hi, wanted=TIME_TICKS_S[1:] + (lo,))
    ax.set_yticks([0, 0.5, 1.0])
    m = metric_by_key(results, "survival_curve")
    if m is not None and m.available and m.value is not None:
        count_label(ax, 0.98, 0.95 if not ax.get_legend() else 0.72, f"half have left by {time_words(m.value)}")
    ax.set_title(title, loc="left")
    ax.set_xlabel("time on the surface"); ax.set_ylabel("share still attached")
    open_frame(ax)
    return ax


@with_house_style
def draw_still_there_by_hour(fig, results, span, cursor_h, settings, cell=None):
    """Panel 5: still there after the short window and after the long one, by landing hour (Lee Fig. 1B across hours)."""
    S = panel_settings(results, settings)
    long = series_of(results, "irreversible_begins") if results is not None else {}
    stay_s = float(long.get("stay_s", S.irreversible_stay_s)) if long else float(S.irreversible_stay_s)
    title = f"Still there after {time_words(S.stay_window_s)} and after {time_words(stay_s)}"
    why = unavailable_reason(results, "stay_by_landing_hour")
    if why:
        return grey_panel(fig, title, why, cell)
    s = series_of(results, "stay_by_landing_hour")
    if not s:
        return grey_panel(fig, title, "no landing with a known end", cell)
    ax = panel_axes(fig, cell)
    draw_stay_by_hour(fig, s, recording_span(results, span), float(S.stay_window_s), ax, long=long)
    ax.set_title(title, loc="left")
    draw_cursor(ax, cursor_h, recording_span(results, span))
    return ax


# ---- Where irreversibility begins
def landing_points(results, settings):
    """The landed tracks as arrays for the dots panel and for picking: track ids, landing hour, time stayed in seconds,
    the fate word per track (leave: left within the stay window; stay; divides) and whether the track was still there at
    the last frame. None without a per_track table or without landings."""
    landed = landed_rows(results)
    if landed is None or landed.height == 0:
        return None
    S = panel_settings(results, settings)
    ids = landed["track_id"].to_numpy().astype(int)
    x = floats(landed, "t_start_h"); stayed = floats(landed, "residence_s")
    fate = np.asarray(landed["fate"].to_list(), dtype=object)
    word = np.where(fate == "divided", "divides", np.where(stayed > float(S.stay_window_s), "stay", "leave"))
    censored = fate == "present at end"
    return ids, x, stayed, word, censored


def dot_axis_range(stayed: np.ndarray, dt_s: float) -> tuple:
    """The log time axis of the dots: 3 s (or one frame when frames are shorter) to 10 h, stretched when a stay is longer."""
    lo = min(3.0, float(dt_s)) if dt_s else 3.0
    hi = max(36000.0, float(np.nanmax(stayed)) * 1.5) if stayed.size else 36000.0
    return lo, hi


def results_dt_s(results) -> float:
    counts = getattr(results, "counts", None)
    if counts is not None and counts.height > 1:
        h = counts["hour"].to_numpy()
        return float((h[1] - h[0]) * 3600.0)
    return 3.0


@with_house_style
def draw_landing_dots(fig, results, span, cursor_h, settings, cell=None, selected=None):
    """Panel 6: every landing as a dot, x the landing hour, y the time stayed on a log axis; grey left within the stay
    window, blue stayed, red divided; open markers for tracks still there at the end; the attached-for-good line and
    the first committed landing marked. selected is a track id drawn with a black ring."""
    title = "Every landing as a dot"
    if results is None:
        return grey_panel(fig, title, RUN_MEASURE, cell)
    pts = landing_points(results, settings)
    if pts is None:
        pt = getattr(results, "per_track", None)
        return grey_panel(fig, title, NO_TRACKS if pt is None or pt.height == 0 else "no landing in these frames", cell)
    S = panel_settings(results, settings)
    ids, x, stayed, word, censored = pts
    lo, hi = dot_axis_range(stayed, results_dt_s(results))
    y = np.clip(stayed, lo, hi)
    ax = panel_axes(fig, cell, PANEL_RECT_WIDE_LEFT)
    a, b = recording_span(results, span)
    for fate, label in (("leave", "left"), ("stay", "stayed"), ("divides", "divided")):
        c = FATE[fate]
        full = (word == fate) & ~censored; open_ = (word == fate) & censored
        ax.plot(x[full], y[full], ls="none", marker="o", ms=2.8, mfc=c, mec=c, mew=0.6, alpha=0.65, label=label)
        if open_.any():
            ax.plot(x[open_], y[open_], ls="none", marker="o", ms=3.4, mfc="white", mec=c, mew=0.9)
    if selected is not None and (ids == int(selected)).any():
        k = int(np.flatnonzero(ids == int(selected))[0])
        ax.plot([x[k]], [y[k]], ls="none", marker="o", ms=8, mfc="none", mec=INK, mew=1.6, zorder=5)
    ax.set_yscale("log"); ax.set_ylim(lo, hi)
    set_time_ticks(ax, lo, hi, axis="y", wanted=TIME_TICKS_S)
    stay_s = float(S.irreversible_stay_s)
    if lo < stay_s < hi:
        ax.axhline(stay_s, color=INK, lw=1.0, ls="--")
        right_is_emptier = (x > (a + b) / 2).sum() <= (x <= (a + b) / 2).sum()
        ax.text(0.98 if right_is_emptier else 0.02, stay_s * 1.15, "attached for good", transform=ax.get_yaxis_transform(),
                ha="right" if right_is_emptier else "left", va="bottom", fontsize=8.5, color=INK)
    irr = series_of(results, "irreversible_begins")
    if irr and irr.get("first_hour") is not None and a <= float(irr["first_hour"]) <= b:
        h = float(irr["first_hour"])
        ax.plot([h, h], [0, 0.12], transform=ax.get_xaxis_transform(), color=INK, lw=1.5, solid_capstyle="butt")
        near_right = (h - a) > (b - h)                    # the label goes to the side with more room
        ax.text(h + (-1 if near_right else 1) * 0.012 * (b - a), 0.06, f"irreversible {h:.1f} h",
                transform=ax.get_xaxis_transform(), ha="right" if near_right else "left", va="center", fontsize=8.5,
                color=INK, bbox=dict(fc="white", ec="none", pad=1.5))
    draw_cursor(ax, cursor_h, (a, b))
    finish_time_axis(ax, (a, b), "landing time after inoculation (h)")
    ax.set_ylabel("time stayed")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8.5, handletextpad=0.3, columnspacing=1.0,
              markerscale=1.4, borderaxespad=0.0)
    ax.set_title(title, loc="left")
    open_frame(ax)
    return ax


def pick_landing(ax, results, x_display: float, y_display: float, tol_px: float = 8, settings=None):
    """The track id of the landing dot nearest to a display position (matplotlib event.x, event.y of the panel's
    canvas) within tol_px pixels, or None. Uses the axes transform, so it matches what draw_landing_dots drew."""
    if ax is None or results is None:
        return None
    pts = landing_points(results, settings)
    if pts is None:
        return None
    ids, x, stayed, _, _ = pts
    lo, hi = dot_axis_range(stayed, results_dt_s(results))
    xy = ax.transData.transform(np.column_stack([x, np.clip(stayed, lo, hi)]))
    d = np.hypot(xy[:, 0] - float(x_display), xy[:, 1] - float(y_display))
    k = int(np.argmin(d))
    return int(ids[k]) if d[k] <= float(tol_px) else None


@with_house_style
def draw_attach_for_good_by_hour(fig, results, span, cursor_h, settings, cell=None):
    """Panel 7: % of landings that attached for good per landing hour with its SE band, and the count of committed
    landings per hour as small bars on a second y axis (our commitment rule)."""
    title = "Attached for good, by hour"
    why = unavailable_reason(results, "irreversible_begins")
    if why:
        return grey_panel(fig, title, why, cell)
    s = series_of(results, "irreversible_begins")
    if not s or not s.get("hour"):
        return grey_panel(fig, title, "no landing in these frames", cell)
    hour = np.asarray(s["hour"], float); pct = np.asarray(s["pct"], float)
    se = np.asarray(s.get("se", np.zeros_like(pct)), float); n = np.asarray(s.get("n", np.zeros_like(pct)), float)
    committed = np.round(pct / 100.0 * n)
    centre = bin_centres(hour)
    ax = panel_axes(fig, cell, PANEL_RECT_TWIN)
    a, b = recording_span(results, span)
    ax2 = ax.twinx()
    ax2.bar(centre, committed, width=0.8, color=GREY, alpha=0.45, lw=0, zorder=1)
    ax2.set_ylim(0, max(float(committed.max()) * 1.4, 1.0)); integer_yticks(ax2)
    ax2.set_ylabel("committed landings")
    for sp in ("top", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.fill_between(centre, pct - se, pct + se, color=BLUE, alpha=0.18, lw=0)
    ax.plot(centre, pct, color=BLUE, lw=1.6, **MK, mec=INK)
    top = min(max(float(np.nanmax(pct + se)) * 1.3, 5.0), 105.0)
    ax.set_ylim(0, top)
    draw_cursor(ax, cursor_h, (a, b))
    finish_time_axis(ax, (a, b), "landing time after inoculation (h)")
    ax2.set_xlim(ax.get_xlim())
    ax.set_ylabel("% of landings")
    m = metric_by_key(results, "committed_share")
    if m is not None and m.available:
        count_label(ax, 0.98, 0.95, f"{m.value:.1f} % attach for good")
    ax.set_title(title, loc="left")
    open_frame(ax)
    return ax


# ---- Movement on the surface
@with_house_style
def draw_moved_or_still(fig, results, span, cursor_h, settings, cell=None):
    """Panel 8: net displacement of tracks of 3+ frames, the motile cut dashed, the motile share inside (Lecuyer Fig. 3C)."""
    title = "Moved or stayed still"
    why = unavailable_reason(results, "motile_share")
    if why:
        return grey_panel(fig, title, why, cell)
    pt = results.per_track
    if pt is None or pt.height == 0 or "motile" not in pt.columns:
        return grey_panel(fig, title, NO_TRACKS, cell)
    judged = pt.filter(pt["motile"].is_not_null())
    if judged.height == 0:
        return grey_panel(fig, title, "no track of 3 frames or more", cell)
    px_um = pixel_um_of(results)
    d_px = floats(judged, "net_disp_px")
    d = d_px * px_um if px_um else d_px
    unit = "µm" if px_um else "px"
    s = series_of(results, "motile_share")
    cut = float(s.get("threshold_px", np.nan)) * (px_um if px_um else 1.0) if s else np.nan
    hi = max(float(np.nanpercentile(d, 98)), (cut * 3 if np.isfinite(cut) else 0.0), 1e-6)
    ax = panel_axes(fig, cell)
    n, _, _ = ax.hist(clip_top(d, hi), bins=hist_bins(0.0, hi), color=BLUE, ec="white", lw=0.6)
    ax.set_ylim(0, max(float(n.max()) * 1.3, 1.0)); integer_yticks(ax)
    if np.isfinite(cut):
        ax.axvline(cut, color=INK, lw=1.2, ls="--")
    m = metric_by_key(results, "motile_share")
    if m is not None and m.available:
        count_label(ax, 0.98, 0.95, f"{m.value:.1f} % moved")
    count_label(ax, 0.98, 0.80, "longer shifts in the last bar", size=8, color=GREY)
    ax.set_xlim(0, hi)
    ax.set_title(title, loc="left")
    ax.set_xlabel(f"net shift ({unit})"); ax.set_ylabel("tracks")
    open_frame(ax)
    return ax


@with_house_style
def draw_pose_at_landing(fig, results, span, cursor_h, settings, cell=None):
    """Panel 9: length over width at the first frame of landed tracks, the on-end cut dashed, % on end inside
    (Conrad Fig. 3A)."""
    title = "Pose at landing"
    why = unavailable_reason(results, "pose_at_landing")
    if why:
        return grey_panel(fig, title, why, cell)
    s = series_of(results, "pose_at_landing")
    ratio = np.asarray([v for v in s.get("length_over_width", []) if v is not None], float) if s else np.zeros(0)
    if ratio.size == 0:
        return grey_panel(fig, title, "no cell row matched a landing", cell)
    S = panel_settings(results, settings)
    cut = float(S.on_end_length_over_width)
    hi = max(float(np.nanpercentile(ratio, 99)), cut * 2.5)
    ax = panel_axes(fig, cell)
    n, _, _ = ax.hist(clip_top(ratio, hi), bins=hist_bins(1.0, hi), color=BLUE, ec="white", lw=0.6)
    ax.set_ylim(0, max(float(n.max()) * 1.3, 1.0)); integer_yticks(ax)
    ax.axvline(cut, color=INK, lw=1.2, ls="--")
    ax.text(cut * 1.03, 0.97, f"on end below {cut:g}", transform=ax.get_xaxis_transform(), ha="left", va="top",
            fontsize=8.5, color=INK)
    m = metric_by_key(results, "pose_at_landing")
    if m is not None and m.available:
        count_label(ax, 0.98, 0.80, f"{m.value:.1f} % on end")
    ax.set_xlim(1.0, hi)
    ax.set_title(title, loc="left")
    ax.set_xlabel("length over width"); ax.set_ylabel("landings")
    open_frame(ax)
    return ax


@with_house_style
def draw_crawl_then_stop(fig, results, span, cursor_h, settings, cell=None):
    """Panel 10: for tracks of 20+ frames, first-half step against second-half step; crawl-then-stop tracks blue
    (Lee Fig. 5B idea)."""
    title = "Crawl, then stop"
    why = unavailable_reason(results, "crawl_then_stop")
    if why:
        return grey_panel(fig, title, why, cell)
    pt = results.per_track
    if pt is None or pt.height == 0 or "crawl_then_stop" not in pt.columns:
        return grey_panel(fig, title, NO_TRACKS, cell)
    judged = pt.filter(pt["crawl_then_stop"].is_not_null())
    if judged.height == 0:
        return grey_panel(fig, title, "no track of 20 frames or more", cell)
    px_um = pixel_um_of(results)
    k = px_um if px_um else 1.0
    unit = "µm" if px_um else "px"
    h1 = floats(judged, "first_half_step_px") * k; h2 = floats(judged, "second_half_step_px") * k
    crawl = np.asarray(judged["crawl_then_stop"].to_list(), dtype=bool)
    ax = panel_axes(fig, cell)
    top = float(np.nanmax(np.concatenate([h1, h2]))) * 1.08 if h1.size else 1.0
    ax.plot([0, top], [0, top], color=GREY, lw=1.0, ls="--")
    ax.plot(h1[~crawl], h2[~crawl], ls="none", marker="o", ms=3.0, mfc=FATE["leave"], mec=FATE["leave"], alpha=0.6,
            label="others")
    ax.plot(h1[crawl], h2[crawl], ls="none", marker="o", ms=3.6, mfc=BLUE, mec=BLUE, alpha=0.9, label="crawl, then stop")
    ax.set_xlim(0, top); ax.set_ylim(0, top)
    m = metric_by_key(results, "crawl_then_stop")
    if m is not None and m.available:
        count_label(ax, 0.02, 0.95, f"{m.value:.1f} % crawl, then stop", ha="left")
    ax.legend(loc="lower right", handlelength=1.0, fontsize=8.5)
    ax.set_title(title, loc="left")
    ax.set_xlabel(f"step in the first half ({unit} per frame)"); ax.set_ylabel(f"second half ({unit} per frame)")
    open_frame(ax)
    return ax


# ---- Division and families
@with_house_style
def draw_daughters(fig, results, span, cursor_h, settings, cell=None):
    """Panel 11: three bars, both daughters stay, one stays, none stays, with the counts (Lee Fig. 3B)."""
    title = "What happens to the daughters"
    why = unavailable_reason(results, "division_outcome")
    if why:
        return grey_panel(fig, title, why, cell)
    s = series_of(results, "division_outcome")
    if not s:
        return grey_panel(fig, title, "no division scored", cell)
    vals = [float(s.get("two", 0)), float(s.get("one", 0)), float(s.get("zero", 0))]
    ax = panel_axes(fig, cell)
    xs = np.arange(3)
    ax.bar(xs, vals, width=0.6, color=BLUE)
    top = max(max(vals) * 1.3, 1.0)
    counts_on_bars(ax, xs, vals, top)
    ax.set_ylim(0, top); integer_yticks(ax); ax.set_xticks(xs); ax.set_xticklabels(["both stay", "one stays", "none stays"])
    n = int(sum(vals)); voided = int(s.get("voided", 0))
    count_label(ax, 0.98, 0.95, f"{n} divisions scored" + (f", {voided} not scored" if voided else ""), size=8.5)
    ax.set_title(title, loc="left"); ax.set_ylabel("divisions")
    open_frame(ax)
    return ax


@with_house_style
def draw_generations(fig, results, span, cursor_h, settings, cell=None):
    """Panel 12: families by their number of generations, the share of cells in one-generation families inside
    (Lee Fig. 2C)."""
    title = "Generations per family"
    why = unavailable_reason(results, "generations_per_family")
    if why:
        return grey_panel(fig, title, why, cell)
    s = series_of(results, "generations_per_family")
    if not s or not s.get("generations"):
        return grey_panel(fig, title, NO_TRACKS, cell)
    gens = np.asarray(s["generations"], float); fam = np.asarray(s["families"], float)
    ax = panel_axes(fig, cell)
    ax.bar(gens, fam, width=0.6, color=BLUE)
    top = max(float(fam.max()) * 1.3, 1.0)
    counts_on_bars(ax, gens, fam, top)
    ax.set_ylim(0, top); integer_yticks(ax); ax.set_xticks(gens); ax.set_xticklabels([f"{int(g)}" for g in gens])
    ax.set_xlim(gens.min() - 0.6, max(gens.max(), gens.min() + 2) + 0.6)
    m = metric_by_key(results, "generations_per_family")
    if m is not None and m.available:
        count_label(ax, 0.98, 0.95, f"{m.value:.1f} % of cells in\none-generation families")
    ax.set_title(title, loc="left")
    ax.set_xlabel("generations in the family"); ax.set_ylabel("families")
    open_frame(ax)
    return ax


@with_house_style
def draw_time_between_divisions(fig, results, span, cursor_h, settings, cell=None):
    """Panel 13: hours between a mother's split and her daughter's, the median inside (Lee Fig. S1A, Hubert Fig. 5)."""
    title = "Time between divisions"
    why = unavailable_reason(results, "division_time_h")
    if why:
        return grey_panel(fig, title, why, cell)
    s = series_of(results, "division_time_h")
    gaps = np.asarray(s.get("hours", []), float) if s else np.zeros(0)
    if gaps.size == 0:
        return grey_panel(fig, title, "no family with two divisions in a row", cell)
    S = panel_settings(results, settings)
    hi = float(S.max_division_time_h)
    ax = panel_axes(fig, cell)
    ax.hist(clip_top(gaps, hi), bins=hist_bins(0.0, hi, 20), color=BLUE, ec="white", lw=0.6)
    integer_yticks(ax)
    m = metric_by_key(results, "division_time_h")
    if m is not None and m.available:
        count_label(ax, 0.98, 0.95, f"median {m.text}, {gaps.size} pairs")
    ax.set_xlim(0, hi)
    ax.set_title(title, loc="left")
    ax.set_xlabel("hours between divisions"); ax.set_ylabel("pairs")
    open_frame(ax)
    return ax


def division_bins(span) -> np.ndarray:
    """Bars along the recording: one per hour, finer for a short span."""
    a, b = float(span[0]), float(span[1])
    width = b - a
    step = 1.0 if width >= 12 else 0.5 if width >= 6 else 0.25
    return np.arange(np.floor(a / step) * step, b + step, step)


@with_house_style
def draw_divisions_over_time(fig, results, span, cursor_h, settings, cell=None):
    """Panel 14: divisions along the recording as bars, with the cursor (our own)."""
    title = "Divisions over time"
    why = unavailable_reason(results, "division_count")
    if why:
        return grey_panel(fig, title, why, cell)
    s = series_of(results, "division_count")
    hours = np.asarray(s.get("hour", []), float) if s else np.zeros(0)
    if hours.size == 0:
        return grey_panel(fig, title, "no division in these frames", cell)
    a, b = recording_span(results, span)
    edges = division_bins((a, b)); n, _ = np.histogram(hours, bins=edges)
    ax = panel_axes(fig, cell)
    ax.bar(edges[:-1], n, width=np.diff(edges), align="edge", color=BLUE, ec="white", lw=0.6)
    draw_cursor(ax, cursor_h, (a, b))
    finish_time_axis(ax, (a, b))
    ax.set_ylim(0, max(float(n.max()) * 1.25, 1.0)); integer_yticks(ax)
    step = float(edges[1] - edges[0])
    count_label(ax, 0.98, 0.95, f"{hours.size} divisions")
    ax.set_title(title, loc="left"); ax.set_ylabel(f"divisions per {time_words(step * 3600)}")
    open_frame(ax)
    return ax


# ---- Flow
@with_house_style
def draw_shift_along_flow(fig, results, span, cursor_h, settings, cell=None):
    """Panel 15: start-to-end shift projected on the flow axis, zero line, mean inside, downstream to the right
    (Lecuyer Fig. S1)."""
    title = "Shift along the flow"
    why = unavailable_reason(results, "along_flow_shift")
    if why:
        return grey_panel(fig, title, why, cell)
    s = series_of(results, "along_flow_shift")
    vals = np.asarray(s.get("along_flow_um", []), float) if s else np.zeros(0)
    if vals.size == 0:
        return grey_panel(fig, title, "no track with two frames", cell)
    lim = max(float(np.nanpercentile(np.abs(vals), 98)), 1e-6)
    ax = panel_axes(fig, cell)
    n, _, _ = ax.hist(np.clip(vals, -lim + 1e-9, lim - 1e-9), bins=hist_bins(-lim, lim, 30), color=BLUE, ec="white", lw=0.6)
    ax.axvline(0, ymax=0.72, color=INK, lw=1.0, ls=":")                 # the zero line stops under the words
    ax.set_ylim(0, max(float(n.max()) * 1.45, 1.0)); integer_yticks(ax)   # the words sit above the peak at zero
    m = metric_by_key(results, "along_flow_shift")
    if m is not None and m.available:
        count_label(ax, 0.98, 0.95, f"mean {m.text}")
    count_label(ax, 0.98, 0.80, "downstream to the right", size=8, color=GREY)
    ax.set_xlim(-lim, lim)
    ax.set_title(title, loc="left")
    ax.set_xlabel("shift along the flow (µm)"); ax.set_ylabel("tracks")
    open_frame(ax)
    return ax


@with_house_style
def draw_angle_to_flow(fig, results, span, cursor_h, settings, cell=None):
    """Panel 16: angle between the cell and the flow at its first frame, 0 to 90 degrees, the aligned cut dashed
    (Lecuyer Methods)."""
    title = "Angle to the flow at landing"
    why = unavailable_reason(results, "orientation_vs_flow")
    if why:
        return grey_panel(fig, title, why, cell)
    s = series_of(results, "orientation_vs_flow")
    ang = np.asarray([v for v in s.get("off_axis_deg", []) if v is not None], float) if s else np.zeros(0)
    if ang.size == 0:
        return grey_panel(fig, title, "no cell row matched a track start", cell)
    S = panel_settings(results, settings)
    cut = float(S.align_angle_deg)
    ax = panel_axes(fig, cell)
    ax.hist(np.clip(ang, 0, 90 - 1e-9), bins=np.arange(0, 91, 10), color=BLUE, ec="white", lw=0.6)
    integer_yticks(ax)
    ax.axvline(cut, color=INK, lw=1.2, ls="--")
    m = metric_by_key(results, "orientation_vs_flow")
    if m is not None and m.available:
        count_label(ax, 0.98, 0.95, f"{m.value:.1f} % within {cut:g} degrees")
    ax.set_xlim(0, 90); ax.set_xticks([0, 30, 60, 90])
    ax.set_title(title, loc="left")
    ax.set_xlabel("angle to the flow (degrees)"); ax.set_ylabel("cells")
    open_frame(ax)
    return ax


@with_house_style
def draw_dividers_and_holders(fig, results, span, cursor_h, settings, cell=None):
    """Panel 17: dividers and holders among long stayers (Hubert rule) with the counts, the growth rate written inside
    with the shear when known (Hubert Fig. 3c and 4)."""
    title = "Dividers and holders"
    why = unavailable_reason(results, "dividers_vs_holders")
    if why:
        return grey_panel(fig, title, why, cell)
    s = series_of(results, "dividers_vs_holders")
    if not s:
        return grey_panel(fig, title, "no track lasted long enough", cell)
    vals = [float(s.get("dividers", 0)), float(s.get("holders", 0))]
    ax = panel_axes(fig, cell)
    xs = np.arange(2)
    ax.bar(xs, vals, width=0.5, color=BLUE)
    top = max(max(vals) * 1.3, 1.0)
    counts_on_bars(ax, xs, vals, top)
    ax.set_ylim(0, top); integer_yticks(ax); ax.set_xticks(xs)
    ax.set_xticklabels([f"dividers\n{int(s.get('by_split', 0))} split, {int(s.get('by_movement', 0))} moved", "holders"])
    eta = metric_by_key(results, "growth_rate_eta")
    if eta is not None and eta.available:
        shear = (eta.series or {}).get("shear_mpa")
        words = f"{eta.text} at {shear:g} mPa" if shear is not None else f"{eta.text} (shear unknown)"
        count_label(ax, 0.98, 0.95, f"growth rate {words}", size=8.5)
    elif eta is not None:
        count_label(ax, 0.98, 0.95, f"growth rate: {eta.reason}", size=8, color=EMPTY_GREY)
    ax.set_title(title, loc="left"); ax.set_ylabel("long stayers")
    open_frame(ax)
    return ax


PANELS = [
    ("arrivals", "cells_on_surface", "Cells on the surface", draw_cells_on_surface, False),
    ("arrivals", "landings", "Landings per hour", draw_landings_per_hour, False),
    ("arrivals", "residence_median", "How long landed cells stayed", draw_how_long_stayed, False),
    ("arrivals", "survival_curve", "Share still attached after t", draw_share_still_attached, False),
    ("arrivals", "stay_by_landing_hour", "Still there after 30 s and after 1 h", draw_still_there_by_hour, False),
    ("commitment", LANDING_DOTS_KEY, "Every landing as a dot", draw_landing_dots, False),
    ("commitment", "irreversible_begins", "Attached for good, by hour", draw_attach_for_good_by_hour, False),
    ("movement", "motile_share", "Moved or stayed still", draw_moved_or_still, False),
    ("movement", "pose_at_landing", "Pose at landing", draw_pose_at_landing, False),
    ("movement", "crawl_then_stop", "Crawl, then stop", draw_crawl_then_stop, False),
    ("division", "division_outcome", "What happens to the daughters", draw_daughters, False),
    ("division", "generations_per_family", "Generations per family", draw_generations, False),
    ("division", "division_time_h", "Time between divisions", draw_time_between_divisions, False),
    ("division", "division_count", "Divisions over time", draw_divisions_over_time, False),
    ("flow", "along_flow_shift", "Shift along the flow", draw_shift_along_flow, True),
    ("flow", "orientation_vs_flow", "Angle to the flow at landing", draw_angle_to_flow, True),
    ("flow", "dividers_vs_holders", "Dividers and holders", draw_dividers_and_holders, True),
]


@with_house_style
def draw_panel(fig, panel, results, span, cursor_h, settings=None, cell=None, **kw):
    """Draw one PANELS entry: the grey reason when results is None or the panel's metric is unavailable (the reason
    comes from that metric), else the panel's own function, which checks its data. Extra keywords go to the function
    (selected= for the landing dots)."""
    group, key, title, fn, needs_flow = panel
    if results is None:
        return grey_panel(fig, title, RUN_MEASURE, cell)
    why = unavailable_reason(results, key)
    if why:
        return grey_panel(fig, title, why, cell)
    return fn(fig, results, span, cursor_h, settings, cell=cell, **kw)


def panels_of_group(group: str, has_flow: bool = True) -> list:
    return [p for p in PANELS if p[0] == group and (has_flow or not p[4])]


def group_keys(has_flow: bool) -> list:
    return [g for g, _ in GROUP_TITLES if (has_flow or g != "flow") and panels_of_group(g, has_flow)]


# ---------------------------------------------------------------------------------------------- the report pages
REPORT_PAGE_IN = (11.0, 8.5)   # landscape letter
REPORT_SIDE_IN = 0.25
REPORT_TITLE_IN = 0.32         # the recording name, from the top edge
REPORT_HEADING_IN = 0.62       # the group heading, from the top edge
REPORT_GRID_TOP_IN = 0.95      # the panel grid starts here, from the top edge
REPORT_BOTTOM_IN = 0.25
REPORT_COLS, REPORT_ROWS = 3, 2
TABLE_ROW_IN = 0.21
TABLE_HEAD_IN = 0.32
TABLE_LINE_IN = 0.17           # a second line of a wrapped reason


def report_size(results=None, has_flow: bool = False, cards_per_row: int = 3) -> tuple:
    """(width, height) in inches of one report page. Kept for the controller; every page is landscape letter."""
    return REPORT_PAGE_IN


def new_figure(size=REPORT_PAGE_IN, dpi: int = 100):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=size, dpi=dpi); FigureCanvasAgg(fig)
    return fig


def report_cells(fig, with_title: bool) -> list:
    """The REPORT_ROWS x REPORT_COLS cells of a page as (x0, y0, w, h) in figure fractions, row by row."""
    W, H = fig.get_size_inches()
    top = H - (REPORT_GRID_TOP_IN if with_title else REPORT_GRID_TOP_IN - REPORT_TITLE_IN)
    cw = (W - 2 * REPORT_SIDE_IN) / REPORT_COLS
    ch = (top - REPORT_BOTTOM_IN) / REPORT_ROWS
    return [((REPORT_SIDE_IN + c * cw) / W, (REPORT_BOTTOM_IN + (REPORT_ROWS - 1 - r) * ch) / H, cw / W, ch / H)
            for r in range(REPORT_ROWS) for c in range(REPORT_COLS)]


def page_head(fig, title: str, heading: str):
    W, H = fig.get_size_inches()
    y = H
    if title:
        fig.text(REPORT_SIDE_IN / W, 1 - REPORT_TITLE_IN / H, title, ha="left", va="center", fontsize=12, fontweight="bold",
                 color=INK)
        y_head = REPORT_HEADING_IN
    else:
        y_head = REPORT_TITLE_IN
    fig.text(REPORT_SIDE_IN / W, 1 - y_head / H, heading, ha="left", va="center", fontsize=11, color=INK)


@with_house_style
def draw_group_page(fig, group: str, results, span=None, has_flow: bool = False, title: str = "", settings=None,
                    cursor_h=None):
    """One report page: the recording name, the group heading, the group's panels in a 3 x 2 grid."""
    fig.clear()
    page_head(fig, title, GROUP_HEADING.get(group, group))
    cells = report_cells(fig, bool(title))
    panels = panels_of_group(group, has_flow)
    if results is None:
        return [grey_panel(fig, panels[0][2] if panels else GROUP_HEADING.get(group, group), RUN_MEASURE, cells[0])]
    return [draw_panel(fig, p, results, span, cursor_h, settings, cell=cell) for p, cell in zip(panels, cells)]


def numbers_rows(results, has_flow: bool = False) -> list:
    """The compact numbers table as rows: ("head", heading) and ("metric", name, value words, n words, available,
    source). Unavailable metrics carry their reason as the value. The page's Numbers table and the report share it."""
    rows = []
    for key, heading in GROUP_TITLES:
        if key == "flow" and not has_flow:
            continue
        metrics = [m for m in results.metrics if m.group == key]
        if not metrics:
            continue
        rows.append(("head", heading))
        for m in metrics:
            value = m.text if m.available else (m.reason or "not available")
            n = f"{m.n:,}" if m.n is not None and m.available else ""
            rows.append(("metric", m.name, value, n, bool(m.available), m.source))
    return rows


@with_house_style
def draw_numbers_page(fig, results, has_flow: bool = False, title: str = ""):
    """The last report page: one row per metric with its name, value and n, grouped like the gallery."""
    fig.clear()
    page_head(fig, title, "Numbers")
    W, H = fig.get_size_inches()
    top_in = REPORT_GRID_TOP_IN if title else REPORT_GRID_TOP_IN - REPORT_TITLE_IN
    ax = fig.add_axes([REPORT_SIDE_IN / W, REPORT_BOTTOM_IN / H, 1 - 2 * REPORT_SIDE_IN / W, 1 - (top_in + REPORT_BOTTOM_IN) / H])
    ax.axis("off")
    if results is None:
        ax.text(0.5, 0.5, RUN_MEASURE, ha="center", va="center", fontsize=10, color=EMPTY_GREY, transform=ax.transAxes)
        return ax
    height_in = H - top_in - REPORT_BOTTOM_IN
    width_in = W - 2 * REPORT_SIDE_IN
    x_value, x_n = 0.36, 1.0
    wrap_chars = max(int((width_in * (x_n - x_value) - 0.8) / (9 * IN_PER_PT_CHAR)), 20)
    rows = numbers_rows(results, has_flow)
    y_in = 0.0

    def put(x, y, text, **kw):
        ax.text(x, 1 - y / height_in, text, transform=ax.transAxes, va="center", **kw)

    put(0.0, 0.10, "Metric", ha="left", fontsize=9, fontweight="bold", color=INK)
    put(x_value, 0.10, "Value", ha="left", fontsize=9, fontweight="bold", color=INK)
    put(x_n, 0.10, "n", ha="right", fontsize=9, fontweight="bold", color=INK)
    y_in = TABLE_ROW_IN
    for row in rows:
        if row[0] == "head":
            put(0.0, y_in + 0.20, row[1], ha="left", fontsize=10, fontweight="bold", color=INK)
            y_in += TABLE_HEAD_IN
            continue
        _, name, value, n, available, source = row
        colour = INK if available else EMPTY_GREY
        lines = textwrap.wrap(value, wrap_chars) or [""]
        put(0.0, y_in + 0.10, name, ha="left", fontsize=9, color=colour)
        put(x_value, y_in + 0.10, lines[0], ha="left", fontsize=9, color=colour, fontweight="bold" if available else "normal")
        put(x_n, y_in + 0.10, n, ha="right", fontsize=9, color=colour)
        y_in += TABLE_ROW_IN
        for extra in lines[1:]:
            put(x_value, y_in + 0.06, extra, ha="left", fontsize=9, color=colour)
            y_in += TABLE_LINE_IN
    if y_in > height_in:
        put(0.0, height_in - 0.05, "the table runs past the page", ha="left", fontsize=8, color=EMPTY_GREY)
    return ax


def draw_report_pages(results, span=None, has_flow: bool = False, title: str = "", settings=None, dpi: int = 100) -> list:
    """The report as figures: one page per group with its panels in a 3 x 2 grid, then the numbers table. Each page is
    11 x 8.5 in landscape. Save them one per PDF page and the first one as the PNG."""
    figs = []
    for group in group_keys(has_flow):
        fig = new_figure(REPORT_PAGE_IN, dpi)
        draw_group_page(fig, group, results, span, has_flow, title, settings)
        figs.append(fig)
    fig = new_figure(REPORT_PAGE_IN, dpi)
    draw_numbers_page(fig, results, has_flow, title)
    figs.append(fig)
    return figs


def draw_report(fig, results, span=None, has_flow: bool = False, title: str = "", cards_per_row: int = 3):
    """Page one of the report (the first group) drawn into a given figure, for callers of the old export."""
    groups = group_keys(has_flow)
    return draw_group_page(fig, groups[0], results, span, has_flow, title)
