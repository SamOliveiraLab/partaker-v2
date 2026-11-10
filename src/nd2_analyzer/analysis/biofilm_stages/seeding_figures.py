"""The six big figures of the Seeding page, two at a time beside each other, picked from a list: one message each, in the
lineage film's style (Arial, bold titles and axis labels, open frame, the lineage red for divisions). Same signature as the
gallery panels, draw_<name>(fig, results, span, cursor_h, settings, cell=None) -> ax, so the page's PanelCanvas draws them.
Colours are what happened to each landing (rules.fate_class_expr): left within the stay window, stayed a while, stayed to
the end, divided. Margins are set in inches, so a figure keeps its labels inside however wide its canvas is."""
from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

from nd2_analyzer.analysis.biofilm_stages import seeding_plots as SP
from nd2_analyzer.analysis.biofilm_stages.seeding_rules import FATE_KEYS, fate_class_expr, fate_class_labels
from nd2_analyzer.analysis.biofilm_stages.seeding_types import SeedingSettings

LINEAGE_RED = "#c0392b"            # the lineage film's tree colour (film_render.ACCENT)
NEUTRAL = "#2b2b2b"
FATE_COLOURS = {k: SP.FATE[k] for k in FATE_KEYS}
LONG_STAY_S = 1800.0               # the time-on-surface figure draws one line per landing that stayed this long
DAUGHTER_STAY_S = 300.0            # a daughter "stays" in the division trees when it stays this long
FIGURE_SIZE_IN = (6.0, 4.3)
NARROW_IN = 5.6                    # below this width the fates legend takes two rows
# margins in inches: left, bottom, right, top (title and legend above the axes)
MARGIN = (0.85, 0.62, 0.22, 0.55)
MARGIN_LEGEND_TOP = (0.85, 0.62, 0.22, 0.95)
MARGIN_LEGEND_TOP_NARROW = (0.85, 0.62, 0.22, 1.18)
MARGIN_NO_Y = (0.25, 0.62, 0.22, 0.75)
# the same as fractions, for the cells of a report page
RECT = [0.13, 0.15, 0.82, 0.70]
RECT_LEGEND_TOP = [0.13, 0.15, 0.82, 0.60]


def rect_for(fig, cell, margin_in, fallback) -> list:
    """The axes rectangle: margins in inches of the whole figure when cell is None, the fallback fractions in a cell."""
    if cell is not None:
        return fallback
    W, H = fig.get_size_inches()
    l, b, r, t = margin_in
    return [l / W, b / H, max(0.1, 1 - (l + r) / W), max(0.1, 1 - (b + t) / H)]


def settings_of(results, settings) -> SeedingSettings:
    return settings or getattr(results, "settings", None) or SeedingSettings()


def classed(results, settings=None) -> pl.DataFrame:
    """per_track with its fate_class column (computed when an older results object lacks it)."""
    pt = results.per_track
    if pt is None or not pt.height:
        return pl.DataFrame()
    if "fate_class" not in pt.columns:
        pt = pt.with_columns(fate_class_expr(settings_of(results, settings)).alias("fate_class"))
    return pt


def landed_classed(results, settings=None) -> pl.DataFrame:
    pt = classed(results, settings)
    return pt.filter(pl.col("origin") == "landed") if pt.height else pt


def measured_end(results, span=None) -> float:
    """The last measured hour: tracks still there at the last measured frame are drawn up to here, never past it."""
    counts = getattr(results, "counts", None)
    end = float(counts["hour"].max()) if counts is not None and counts.height else None
    if span is not None:
        end = float(span[1]) if end is None else min(end, float(span[1]))
    return end if end is not None else 0.0


def irreversible_hour(results, settings=None) -> Optional[float]:
    m = SP.metric_by_key(results, "irreversible_begins")
    if m is not None and m.available and m.value is not None:
        return float(m.value)
    L = landed_classed(results, settings)
    if not L.height:
        return None
    first = L.filter(pl.col("fate_class").is_in(["divided", "end"]))
    return float(first["t_start_h"].min()) if first.height else None


def irreversible_line(ax, results, settings, label: bool = False, y: float = 0.85):
    h = irreversible_hour(results, settings)
    if h is None:
        return
    a, b = ax.get_xlim()
    if not a <= h <= b:
        return
    ax.axvline(h, color=LINEAGE_RED, lw=1.3, ls=(0, (4, 3)), zorder=1)
    if label:
        right = (h - a) < (b - h)                      # the label goes to the side of the line with more room
        ax.text(h + (0.012 if right else -0.012) * (b - a), y, f"irreversible attachment\nbegins {h:.1f} h",
                transform=ax.get_xaxis_transform(), ha="left" if right else "right", va="top", fontsize=9.5,
                color=LINEAGE_RED, bbox=dict(fc="white", ec="none", pad=1.0))


def span_of(results, span):
    return SP.recording_span(results, span)


# ---------------------------------------------------------------------------------------------- 1. cells on the surface
@SP.with_house_style
def draw_cells(fig, results, span, cursor_h, settings, cell=None):
    """Cells per frame (6-min means), the lag fit dashed, the take-off as a short dotted mark with its hour."""
    counts = results.counts
    if counts is None or not counts.height:
        return SP.grey_panel(fig, "Cells on the surface", "no cell counts", cell)
    ax = SP.panel_axes(fig, cell, rect_for(fig, cell, MARGIN, RECT))
    c = counts.with_columns((pl.col("hour") * 10).floor().alias("b")).group_by("b").agg(
        pl.col("hour").mean(), pl.col("cells").mean()).sort("b")
    ax.plot(c["hour"].to_numpy(), c["cells"].to_numpy(), color=SP.BLUE, lw=1.8, label="cells on the surface")
    lag = SP.series_of(results, "lag_time")
    top = float(np.nanmax(c["cells"].to_numpy())) if c.height else 1.0
    a, b = span_of(results, span)
    if lag and lag.get("t_lag_h") is not None and lag.get("tau_h"):
        hh = np.linspace(a, b, 300)
        fit = float(lag["n0"]) * np.exp((hh - float(lag["t_lag_h"])) / float(lag["tau_h"]))
        keep = fit <= top * 1.2
        ax.plot(hh[keep], fit[keep], color=SP.INK, lw=1.2, ls=(0, (5, 3)),
                label=f"lag fit: {float(lag['t_lag_h']):.1f} h, doubles every {float(lag['tau_h']) * np.log(2):.1f} h")
    m = SP.metric_by_key(results, "take_off_hour")
    ax.set_ylim(0, top * 1.18)
    if m is not None and m.available and m.value is not None:
        ax.vlines(float(m.value), 0, top * 0.5, color=SP.GREY, lw=1.1, ls=(0, (2, 3)))
        ax.text(float(m.value), top * 0.54, f"take-off {float(m.value):.1f} h", ha="center", va="bottom", fontsize=9.5, color="#4a4a4a")
    SP.finish_time_axis(ax, (a, b))
    ax.set_ylabel("cells"); ax.set_title("Cells on the surface", loc="left")
    ax.legend(loc="upper left", handlelength=2.2)
    SP.open_frame(ax)
    return ax


# ---------------------------------------------------------------------------------------------- 2. still there after the window
@SP.with_house_style
def draw_still_there(fig, results, span, cursor_h, settings, cell=None):
    S = settings_of(results, settings)
    s = SP.series_of(results, "stay_by_landing_hour")
    title = f"Still there after {S.stay_window_s:g} s, by landing hour"
    if not s or not s.get("hour"):
        return SP.grey_panel(fig, title, SP.reason_line(results, "stay_by_landing_hour", "no landings to score"), cell)
    ax = SP.panel_axes(fig, cell, rect_for(fig, cell, MARGIN, RECT))
    h = SP.bin_centres(np.asarray(s["hour"], float))
    p = np.asarray(s["pct"], float); se = np.asarray(s.get("se") or np.zeros_like(p), float)
    ax.fill_between(h, np.clip(p - se, 0, None), p + se, color=SP.BLUE, alpha=0.14, lw=0)
    ax.plot(h, p, color=SP.BLUE, lw=1.8, marker="o", ms=5.5, mfc="white", mew=1.4)
    hi = max(10.0, float(np.nanmax(p + se)) * 1.25) if p.size else 10.0
    ax.set_ylim(0, hi)
    SP.finish_time_axis(ax, span_of(results, span), "landing time after inoculation (h)")
    irreversible_line(ax, results, settings, label=True)
    ax.set_ylabel("% of landings"); ax.set_title(title, loc="left")
    SP.open_frame(ax)
    return ax


# ---------------------------------------------------------------------------------------------- 3. what each landing does
@SP.with_house_style
def draw_fates_by_hour(fig, results, span, cursor_h, settings, cell=None):
    S = settings_of(results, settings)
    L = landed_classed(results, S)
    title = "What each landing does, by landing hour"
    if not L.height:
        return SP.grey_panel(fig, title, "no landings", cell)
    narrow = cell is None and fig.get_size_inches()[0] < NARROW_IN
    ax = SP.panel_axes(fig, cell, rect_for(fig, cell, MARGIN_LEGEND_TOP_NARROW if narrow else MARGIN_LEGEND_TOP, RECT_LEGEND_TOP))
    L = L.with_columns(pl.col("t_start_h").floor().alias("hr"))
    names = fate_class_labels(S)
    for k in FATE_KEYS:
        g = L.filter(pl.col("fate_class") == k).group_by("hr").len().sort("hr")
        if not g.height:
            continue
        ax.plot(g["hr"].to_numpy() + 0.5, g["len"].to_numpy(), color=FATE_COLOURS[k], lw=1.8, marker="o", ms=5.5,
                mfc=FATE_COLOURS[k], mec=SP.INK, mew=0.5, label=names[k])
    ax.set_yscale("log"); ax.set_ylim(0.7, max(10.0, float(L.group_by("hr").len()["len"].max()) * 2.5))
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ticks = [v for v in (1, 10, 100, 1000, 10000) if v <= ax.get_ylim()[1]]
    ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    SP.finish_time_axis(ax, span_of(results, span), "landing time after inoculation (h)")
    irreversible_line(ax, results, settings, label=True, y=0.99)
    ax.set_ylabel("landings per hour")
    ax.set_title(title, loc="left", pad=48 if narrow else 30)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2 if narrow else 4, handlelength=1.4, columnspacing=0.9,
              borderaxespad=0.2, fontsize=8.5)
    SP.open_frame(ax)
    return ax


# ---------------------------------------------------------------------------------------------- 4. late landers stay longer
def kaplan_meier(times, events):
    """Survival steps of time-on-surface: events are leavings, the rest is censored (still there, or divided)."""
    o = np.argsort(times); t = np.asarray(times, float)[o]; e = np.asarray(events, bool)[o]
    n = len(t); xs, ys, s, i = [], [], 1.0, 0
    while i < n:
        j = i
        while j < n and t[j] == t[i]:
            j += 1
        d = int(e[i:j].sum())
        if d:
            s *= 1 - d / (n - i); xs.append(t[i]); ys.append(s)
        i = j
    return np.asarray(xs), np.asarray(ys)


@SP.with_house_style
def draw_late_landers(fig, results, span, cursor_h, settings, cell=None):
    L = landed_classed(results, settings)
    m = SP.metric_by_key(results, "take_off_hour")
    title = "Late landers stay longer"
    if not L.height or m is None or not m.available or m.value is None:
        return SP.grey_panel(fig, title, "needs landings and the take-off hour", cell)
    split = float(m.value)
    ax = SP.panel_axes(fig, cell, rect_for(fig, cell, MARGIN, RECT))
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    dt = SP.results_dt_s(results)
    lo, hi = max(dt, 1e-3), 3 * 3600.0
    for name, col, sel in (("landed before take-off", SP.GREY, pl.col("t_start_h") < split),
                           ("landed after take-off", SP.BLUE, pl.col("t_start_h") >= split)):
        g = L.filter(sel)
        if g.height < 5:
            continue
        x, y = kaplan_meier(g["residence_s"].to_numpy(), g["fate"].is_in(["detached", "left field"]).to_numpy())
        if not x.size:
            continue
        ax.step(x, y, where="post", color=col, lw=2.0, label=f"{name} (n = {g.height:,})")
        for t_mark in (30.0, 3600.0):
            v = float(y[np.searchsorted(x, t_mark, side="right") - 1]) if np.any(x <= t_mark) else 1.0
            if v >= 0.0012:
                ax.plot([t_mark], [v], "o", color=col, ms=6, mec="white", mew=1.0, zorder=4)
                ax.text(t_mark, v * 1.6, f"{100 * v:.0f} %" if v >= 0.01 else f"{100 * v:.1f} %", color=col, fontsize=9.5,
                        fontweight="bold", ha="center", va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(0.0008, 1.3)
    words = {3: "3 s", 30: "30 s", 300: "5 min", 3600: "1 h", 10800: "3 h"}
    ax.xaxis.set_major_locator(FixedLocator([v for v in words if lo <= v <= hi])); ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: words.get(int(round(v)), "")))
    ax.yaxis.set_major_locator(FixedLocator([0.001, 0.01, 0.1, 1])); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: {0.001: "0.1 %", 0.01: "1 %", 0.1: "10 %", 1.0: "100 %"}.get(round(v, 3), "")))
    ax.set_xlabel("time since landing"); ax.set_ylabel("still on the surface"); ax.set_title(title, loc="left")
    ax.legend(loc="lower left", handlelength=1.8)
    SP.open_frame(ax)
    return ax


# ---------------------------------------------------------------------------------------------- 5. the divisions as trees
def daughter_stays(d: dict) -> bool:
    """A daughter stays when it was still there at the end, divided, or stayed DAUGHTER_STAY_S."""
    return d["fate"] in ("present at end", "divided") or float(d["residence_s"]) >= DAUGHTER_STAY_S


@SP.with_house_style
def draw_division_trees(fig, results, span, cursor_h, settings, cell=None):
    pt = classed(results, settings)
    title = "The divisions: one daughter usually stays"
    if not pt.height or "parent" not in pt.columns:
        return SP.grey_panel(fig, title, "no tracks", cell)
    mothers = pt.filter(pl.col("n_children") >= 2).sort("t_end_h")
    if not mothers.height:
        return SP.grey_panel(fig, title, "no division in this recording", cell)
    kids = pt.filter(pl.col("parent").is_in(mothers["track_id"].to_list()))
    a, _b = span_of(results, span)
    b = measured_end(results, span)
    x0 = max(a, float(np.floor(float(mothers["t_start_h"].min()) - 0.3)))
    ax = SP.panel_axes(fig, cell, rect_for(fig, cell, MARGIN_NO_Y, [0.10, 0.12, 0.86, 0.72]))
    title = f"The {mothers.height} divisions: one daughter usually stays"
    y = 0.0
    for m in mothers.iter_rows(named=True):
        tdiv = float(m["t_end_h"])
        ax.plot([max(float(m["t_start_h"]), x0), tdiv], [y, y], color=LINEAGE_RED, lw=3.0, solid_capstyle="butt")
        if m["origin"] == "landed":
            ax.plot([float(m["t_start_h"])], [y], marker="|", ms=8, mew=1.5, color=SP.INK)
        ax.text(max(float(m["t_start_h"]), x0) - 0.012 * (b - x0), y, str(int(m["track_id"])), ha="right", va="center",
                fontsize=8, color="#4a4a4a")
        ds = kids.filter(pl.col("parent") == m["track_id"]).sort("track_id").to_dicts()[:2]
        for k, d in enumerate(ds):
            dy = y + (0.27 if k == 0 else -0.27)
            ax.plot([tdiv, tdiv], [y, dy], color=NEUTRAL, lw=1.0)
            end = b if d["fate"] == "present at end" else min(b, float(d["t_end_h"]))
            if daughter_stays(d):
                ax.plot([tdiv, end], [dy, dy], color=LINEAGE_RED, lw=3.0, solid_capstyle="butt")
            else:
                e = max(end, tdiv + 0.004 * (b - x0))
                ax.plot([tdiv, e], [dy, dy], color=SP.GREY, lw=1.4, solid_capstyle="butt")
                ax.plot([e], [dy], "o", ms=4, mfc="white", mec=SP.GREY, mew=1.2)
        y -= 1.0
    # the key, in the empty lower left corner
    ky = y + 3.6
    kx0, kx1 = x0 - 0.06 * (b - x0), x0 + 0.02 * (b - x0)
    ax.plot([kx0, kx1], [ky, ky], color=LINEAGE_RED, lw=3.0, solid_capstyle="butt")
    ax.text(kx1 + 0.02 * (b - x0), ky, "on the surface", va="center", fontsize=8.5, color="#4a4a4a")
    ax.plot([(kx0 + kx1) / 2], [ky - 0.9], "o", ms=4, mfc="white", mec=SP.GREY, mew=1.2)
    ax.text(kx1 + 0.02 * (b - x0), ky - 0.9, f"daughter left within {DAUGHTER_STAY_S / 60:g} min", va="center", fontsize=8.5, color="#4a4a4a")
    ax.plot([(kx0 + kx1) / 2], [ky - 1.8], marker="|", ms=8, mew=1.5, color=SP.INK)
    ax.text(kx1 + 0.02 * (b - x0), ky - 1.8, "mother lands", va="center", fontsize=8.5, color="#4a4a4a")
    ax.set_xlim(x0 - 0.08 * (b - x0), b + 0.01 * (b - x0)); ax.set_ylim(y + 0.4, 0.7)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("time after inoculation (h)")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_title(title, loc="left", pad=22)
    o = SP.series_of(results, "division_outcome") or {}
    if o:
        ax.text(0.0, 1.01, f"both daughters stay {o.get('two', 0)},  one stays {o.get('one', 0)},  none {o.get('zero', 0)}"
                f"  (staying = {DAUGHTER_STAY_S / 60:g} min or more)",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color="#4a4a4a")
    SP.open_frame(ax)
    return ax


# ---------------------------------------------------------------------------------------------- 6. time on the surface
def staying_rows(results, settings=None, min_s: float = LONG_STAY_S) -> list:
    """(row, depth) in drawing order: landings that stayed min_s or more, or divided, or stayed to the end, earliest
    first, each followed by its daughters (and theirs)."""
    pt = classed(results, settings)
    if not pt.height:
        return []
    keep = pt.filter((pl.col("origin") == "landed") & ((pl.col("residence_s") >= min_s) |
                                                     pl.col("fate_class").is_in(["divided", "end"]))).sort("t_start_h")
    by_parent = {}
    if "parent" in pt.columns:
        for d in pt.filter(pl.col("parent").is_not_null()).to_dicts():
            by_parent.setdefault(int(d["parent"]), []).append(d)
    rows = []

    def add(r, depth):
        rows.append((r, depth))
        for d in sorted(by_parent.get(int(r["track_id"]), []), key=lambda d: d["track_id"]):
            add(d, depth + 1)
    for r in keep.to_dicts():
        add(r, 0)
    return rows


def draw_staying_lines(ax, rows, end_h: float, upto: Optional[float] = None, bold=(), labels=()):
    """One horizontal line per row, coloured by its fate class, daughters joined to their mother; drawn up to upto.
    end_h is the last measured hour: a track still there at the end is drawn up to it."""
    ypos = {int(r["track_id"]): -i for i, (r, _d) in enumerate(rows)}
    for i, (r, _d) in enumerate(rows):
        a = float(r["t_start_h"]); b = end_h if r["fate"] == "present at end" else min(end_h, float(r["t_end_h"]))
        if upto is not None:
            if a > upto:
                continue
            b = min(b, upto)
        f = r.get("fate_class", "while"); col = FATE_COLOURS.get(f, FATE_COLOURS["while"])
        lw = 2.6 if f in ("divided", "end") else 1.5
        tid = int(r["track_id"])
        if tid in bold:
            ax.plot([a, max(b, a + 0.03)], [-i, -i], color=SP.INK, lw=lw + 2.2, solid_capstyle="butt", zorder=3)
        ax.plot([a, max(b, a + 0.03)], [-i, -i], color=col, lw=lw, solid_capstyle="butt", zorder=4)
        if tid in labels:
            ax.text(a - 0.1, -i, str(tid), ha="right", va="center", fontsize=9, fontweight="bold", zorder=5)
        p = r.get("parent")
        if p is not None and int(p) in ypos and (upto is None or a <= upto):
            ax.plot([a, a], [ypos[int(p)], -i], color=NEUTRAL, lw=0.8, zorder=4)
    ax.set_ylim(-len(rows) - 2, 3)


def lines_start_hour(rows, span) -> float:
    a, _b = span
    if not rows:
        return a
    first = min(float(r["t_start_h"]) for r, _d in rows)
    return max(a, float(np.floor((first - 2.0) / 2.0) * 2.0))


@SP.with_house_style
def draw_time_on_surface(fig, results, span, cursor_h, settings, cell=None):
    rows = staying_rows(results, settings)
    title = "Time on the surface, one line per cell"
    if not rows:
        return SP.grey_panel(fig, title, f"no landing stayed {LONG_STAY_S / 60:g} min", cell)
    a, b = span_of(results, span)
    end = measured_end(results, span)
    ax = SP.panel_axes(fig, cell, rect_for(fig, cell, MARGIN_NO_Y, [0.05, 0.14, 0.91, 0.72]))
    draw_staying_lines(ax, rows, end)
    x0 = lines_start_hour(rows, (a, b))
    ax.set_xlim(x0, b)
    irreversible_line(ax, results, settings)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_xlabel("time after inoculation (h)"); ax.set_title(title, loc="left")
    L = landed_classed(results, settings)
    long = L.filter(pl.col("residence_s") >= LONG_STAY_S)
    first = f",\nthe first at {float(long['t_start_h'].min()):.1f} h" if long.height else ""
    import matplotlib.pyplot as plt
    names = fate_class_labels(settings_of(results, settings))
    handles = [plt.Line2D([], [], color=FATE_COLOURS[k], lw=2.6) for k in FATE_KEYS]
    words = [names[k] for k in FATE_KEYS]
    h_irr = irreversible_hour(results, settings)
    if h_irr is not None and x0 <= h_irr <= b:                 # the red dashed line, named in the key
        handles.append(plt.Line2D([], [], color=LINEAGE_RED, lw=1.3, ls=(0, (4, 3))))
        words.append(f"irreversible attachment begins {h_irr:.1f} h")
    leg = ax.legend(handles, words, loc="lower left", ncol=1, handlelength=1.4, fontsize=8.5)
    # the counts sit just above the key, whatever its height
    top = leg.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted()).y1
    gap = 0.12 / max(fig.get_size_inches()[1] * ax.get_position().height, 1e-6)
    ax.text(0.01, top + gap, f"{long.height} of {L.height:,} landings\nstayed {LONG_STAY_S / 60:g} min or more{first}.\nDaughters hang\nunder their mother.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color="#4a4a4a")
    SP.open_frame(ax)
    return ax


FIGURES = [
    ("figures", "fig_cells", "Cells on the surface", draw_cells, False),
    ("figures", "fig_still", "Still there after 30 s, by landing hour", draw_still_there, False),
    ("figures", "fig_fates", "What each landing does, by landing hour", draw_fates_by_hour, False),
    ("figures", "fig_late", "Late landers stay longer", draw_late_landers, False),
    ("figures", "fig_divisions", "The divisions as trees", draw_division_trees, False),
    ("figures", "fig_lines", "Time on the surface, one line per cell", draw_time_on_surface, False),
]
FIGURE_DEFAULTS = ("fig_cells", "fig_still")    # the two the page opens with


def figure_by_key(key: str):
    for f in FIGURES:
        if f[1] == key:
            return f
    return None


# ---------------------------------------------------------------------------------------------- the eight cards
def card_facts(results) -> list:
    """(value, meaning, available, tooltip) for the page's eight cards: a number and one plain line on what it means. A
    metric the recording cannot give shows its name and the reason, greyed."""
    S = settings_of(results, None)
    out = []

    def add(key, value, meaning):
        m = SP.metric_by_key(results, key)
        if m is None or not m.available or m.value is None:
            out.append((m.name if m is not None else key.replace("_", " "), (m.reason if m is not None else "not measured") or "not available",
                        False, (m.reason if m is not None else "") or ""))
            return
        out.append((value(m), meaning(m) if callable(meaning) else meaning, True, m.text or ""))

    add("landings", lambda m: f"{int(m.value):,} landings", "cells that touched the surface")
    add("residence_median", lambda m: f"half leave within {SP.time_words(float(m.value))}", "first contact is brief")
    add("leave_within_30s", lambda m: f"{float(m.value):.0f} % leave within {S.stay_window_s:g} s", "reversible attachment")
    add("lag_time", lambda m: f"lag {float(m.value):.1f} h", "before the surface population grows")
    add("irreversible_begins", lambda m: f"irreversible attachment {float(m.value):.1f} h", "first cell that stays for good")

    def daughters(_m):
        o = SP.series_of(results, "division_outcome") or {}
        one, two, zero = int(o.get("one", 0)), int(o.get("two", 0)), int(o.get("zero", 0))
        if not (one or two or zero):
            return "daughters not scored"
        return ("one daughter stays in most" if one >= max(two, zero) else
                "both daughters stay in most" if two >= zero else "both daughters leave in most")
    add("division_count", lambda m: f"{int(m.value)} division{'s' if int(m.value) != 1 else ''}", daughters)
    end_h = measured_end(results)
    add("cells_on_surface", lambda m: f"{int(m.value)} cells at the end", f"on the surface at {end_h:.1f} h")
    add("motile_share", lambda m: f"{float(m.value):.0f} % moved", "crawled while attached")
    return out


def status_words(results, info=None) -> str:
    """The page's one status line in plain words; the rules' full notes go in its tooltip."""
    counts = results.counts
    parts = []
    if counts is not None and counts.height:
        a, b = float(counts["hour"].min()), float(counts["hour"].max())
        dt = SP.results_dt_s(results)
        parts.append(f"Frames {int(counts['frame'].min()):,} to {int(counts['frame'].max()):,} ({a:.1f} to {b:.1f} h), {dt:g} s apart.")
    L = landed_classed(results)
    pt = results.per_track
    divisions = int((pt["n_children"] >= 2).sum()) if pt is not None and pt.height and "n_children" in pt.columns else 0
    parts.append(f"{L.height:,} landings, {pt.height if pt is not None else 0:,} tracks, {divisions} divisions.")
    parts.append("Every number here is computed by the tool on this recording.")
    return "  ".join(parts)
