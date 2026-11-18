"""What the Maturation page shows, read from a growth-states measure (growth_measure.GrowthStates): the positions to
follow, the cards, the strip with the states, and the plots of the picker. Plain Python and matplotlib, no Qt."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from . import growth_states as GS
from . import seeding_plots as SP
from .growth_measure import GrowthStates

STATE_SHADE = {"I": "#e3e9ef", "II": "#c9d5e0", "III": "#aebfd0"}      # calm blue-grey steps, as Figure 5
MATRIX_GREEN, OUTSIDE_GREEN, MODEL_GREY = "#12802b", "#7fd08f", "#7a7a7a"
FIGURES = (("cells", "Number of cells and Th2"), ("rate", "Growth rate"), ("matrix", "Matrix stain and Th1"),
           ("inside", "Matrix stain inside and outside microcolonies"), ("positions", "Number of cells, every position"))
FIGURE_DEFAULTS = ("cells", "matrix")
FIGURE_SIZE_IN = (5.6, 4.6)
NO_DYE = "give the dye frames to measure the matrix stain"


def _span(gs: GrowthStates) -> Tuple[float, float]:
    return min(float(gs.hours[p][0]) for p in gs.positions), max(float(gs.hours[p][-1]) for p in gs.positions)


def _at(hours: np.ndarray, values: np.ndarray, h: float) -> Tuple[float, float]:
    k = int(np.argmin(np.abs(np.asarray(hours) - h))); return float(hours[k]), float(values[k])


def has_matrix(gs: GrowthStates) -> bool:
    return bool(gs.matrix)


def states(gs: GrowthStates) -> List[Tuple[str, float, float]]:
    """(state, start hour, end hour) of the states the recording shows: I to Th1, II to Th2, III after it."""
    if not has_matrix(gs) or gs.th1 is None or gs.th1["th1_h"] is None:
        return []
    first_h, last_h = _span(gs); th1 = float(gs.th1["th1_h"]); th2 = float(gs.th2["th2_h"]) if gs.th2 else None
    out = [("I", first_h, th1), ("II", th1, min(th2, last_h) if th2 is not None else last_h)]
    if th2 is not None and th2 < last_h:
        out.append(("III", th2, last_h))
    return out


def th2_words(gs: GrowthStates) -> str:
    if not gs.th2:
        return ""
    return f"Th2 {gs.th2['th2_h']:.1f} h" + ("" if gs.th2["reached"] else ", predicted")


def entries(gs: GrowthStates) -> List[Tuple[str, List[Tuple[int, str]]]]:
    rows = []
    for p in gs.positions:
        th1 = gs.th1["per_position_h"].get(p) if gs.th1 else None
        first = (f"Th1 {th1:g} h, " if th1 is not None else "no matrix stain that stays, ") if has_matrix(gs) else ""
        rows.append((p, f"Position {p}: {first}{int(gs.cells[p][-1]):,} cells at {gs.hours[p][-1]:g} h"))
    return [("positions measured", rows)]


def cards(gs: GrowthStates, cursor_h: float) -> List[Tuple[str, str, bool]]:
    """Eight cards: the thresholds and the fastest growth, then four numbers at the cursor's hour."""
    hours, cells, _sd = gs.mean_of("cells"); _h, rate, _s = gs.mean_of("rate"); n = len(gs.positions); k = int(np.argmax(rate))
    h, c = _at(hours, cells, cursor_h); _hh, r = _at(hours, rate, cursor_h); where = f"mean of {n} positions" if n > 1 else f"position {gs.positions[0]}"
    out = []
    if has_matrix(gs) and gs.th1 and gs.th1["th1_h"] is not None:
        out.append((f"Th1 {gs.th1['th1_h']:.1f} ± {gs.th1['sd_h']:.1f} h" if n > 1 else f"Th1 {gs.th1['th1_h']:g} h", "matrix stain appears and stays", True))
    else:
        out.append(("Th1 not measured", NO_DYE, False) if not has_matrix(gs) else ("Th1 not reached", "no matrix stain that stays", False))
    if gs.th2:
        low, high = gs.th2["range95_h"]
        out.append((th2_words(gs), "growth down to 10 % of its peak", True)); out.append((f"{low:.1f} to {high:.1f} h", "95 % range of Th2", bool(np.isfinite(low))))
    else:
        out += [("Th2 not found", "no growth curve fits the count", False), ("", "95 % range of Th2", False)]
    out.append((f"fastest growth {hours[k]:g} h", f"{rate[k]:.0f} cells per hour", True))
    out.append((f"{c:,.0f} cells at {h:g} h", f"whole frame, {where}", True))
    out.append((f"{r:.0f} cells per hour", f"growth rate at {h:g} h", True))
    if has_matrix(gs):
        mh, m, _ = gs.mean_of("matrix"); _a, ins, _b = gs.mean_of("inside"); _c, outs, _d = gs.mean_of("outside")
        hm, mv = _at(mh, m, cursor_h); _x, iv = _at(mh, ins, cursor_h); _y, ov = _at(mh, outs, cursor_h)
        out.append((f"{mv:.1f} % at {hm:g} h", "of the frame is matrix stain", True))
        out.append((f"{iv:.1f} % against {ov:.1f} %", f"inside and outside microcolonies, {hm:g} h", True) if np.isfinite(iv) else
                   (f"no microcolony at {hm:g} h", "matrix stain inside and outside microcolonies", False))
    else:
        out += [("matrix stain not measured", NO_DYE, False), ("inside and outside", NO_DYE, False)]
    return out


# ---------------------------------------------------------------------------------------------- the strip
@SP.with_house_style
def draw_strip(ax, gs: Optional[GrowthStates], span: Tuple[float, float], cursor_h: float):
    """The recording's hours with the growth states, Th1, Th2 and the cursor."""
    lo, hi = float(span[0]), float(span[1]); ax.set_xlim(*SP.pad_span((lo, hi), 0.03)); ax.set_ylim(0, 1); ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ticks = SP.hour_ticks((lo, hi)); ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g} h" for t in ticks], fontsize=8.5)
    shown = states(gs) if gs is not None else []
    if not shown:
        ax.axvspan(lo, hi, ymin=0.30, ymax=0.72, color="#f0f0f0", lw=0)
    for name, a, b in shown:
        ax.axvspan(a, b, ymin=0.30, ymax=0.72, color=STATE_SHADE[name], lw=0)
        ax.text((a + b) / 2, 0.51, name, ha="center", va="center", fontsize=9, fontweight="bold", color=SP.INK, family="serif")
    th1 = shown[0][2] if shown else None
    if th1 is not None:
        ax.axvline(th1, ymin=0.26, ymax=0.76, color=SP.INK, lw=1.1, ls=(0, (4, 2)))
        ax.text(th1, 0.90, f"Th1 {th1:.1f} h", ha="center", va="center", fontsize=8.5, fontweight="bold", color=SP.INK)
    if gs is not None and gs.th2:
        th2 = gs.th2["th2_h"]
        if th2 <= hi:
            ax.axvline(th2, ymin=0.26, ymax=0.76, color=SP.INK, lw=1.1, ls=(0, (4, 2)))
        ax.text(hi, 0.90, th2_words(gs) + ("" if th2 <= hi else "  →"), ha="right", va="center", fontsize=8.5, fontweight="bold", color="#666666")
    ax.axvline(cursor_h, color=SP.RED, lw=2.0)
    left_free = th1 is None or (th1 - lo) > 0.22 * (hi - lo)
    ax.text(lo if left_free else (lo + hi) / 2, 0.90, f"{cursor_h:g} h", ha="left" if left_free else "center", va="center", fontsize=8.5, fontweight="bold", color=SP.RED)


# ---------------------------------------------------------------------------------------------- the plots
def _axes(fig, title: str, note: str = "", left: float = 0.16):
    fig.clear(); ax = fig.add_axes([left, 0.15, 0.95 - left, 0.60]); SP.open_frame(ax)
    fig.text(left, 0.93, title, fontsize=12, fontweight="bold", color=SP.INK, va="center")
    if note:
        fig.text(left, 0.865, note, fontsize=9.5, color="#555555", va="center")
    return ax


def _missing(fig, title: str):
    ax = _axes(fig, title); ax.axis("off"); ax.text(0.5, 0.5, NO_DYE, ha="center", va="center", transform=ax.transAxes, color=SP.EMPTY_GREY)
    return ax


def _shade(ax, gs: GrowthStates, x_end: Optional[float] = None):
    for name, a, b in states(gs):
        ax.axvspan(a, b if name != "II" or x_end is None or not gs.th2 or gs.th2["reached"] else min(gs.th2["th2_h"], x_end), color=STATE_SHADE[name], lw=0, zorder=0)


def _cursor(ax, cursor_h: Optional[float]):
    if cursor_h is not None:
        ax.axvline(cursor_h, color="#9a9a9a", lw=1.2, zorder=1)


def _key(fig, left: float, items):
    x = left
    for colour, style, words in items:
        fig.add_artist(__import__("matplotlib").lines.Line2D([x, x + 0.045], [0.80, 0.80], transform=fig.transFigure, color=colour, lw=2, **style))
        t = fig.text(x + 0.055, 0.80, words, fontsize=9, color="#333333", va="center"); fig.canvas.draw()
        x += 0.055 + t.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width + 0.03


@SP.with_house_style
def draw_cells(fig, gs: GrowthStates, cursor_h: Optional[float] = None, selected: Optional[int] = None):
    hours, mean, sd = gs.mean_of("cells"); note = th2_words(gs) + (f" ({gs.th2['range95_h'][0]:.1f} to {gs.th2['range95_h'][1]:.1f} h)" if gs.th2 else "")
    ax = _axes(fig, "Number of cells and Th2", note); x_end = hours[-1]
    if gs.th2 and np.isfinite(gs.th2["th2_h"]):
        th2 = gs.th2["th2_h"]; x_end = max(hours[-1], min(th2 * 1.12, hours[-1] + 3 * (hours[-1] - hours[0])))
        t = np.linspace(hours[0], x_end, 300); params = (gs.th2["K"], gs.th2["r"], gs.th2["t0"])
        ax.plot(t, GS.gompertz(t, *params), color=MODEL_GREY, lw=1.6, ls=(0, (5, 2.5)), zorder=2)
        if th2 <= x_end:
            y2 = float(GS.gompertz(th2, *params)); low, high = gs.th2["range95_h"]
            if np.isfinite(low):
                ax.plot([max(low, hours[0]), min(high, x_end)], [y2, y2], color=MODEL_GREY, lw=3.2, solid_capstyle="butt", zorder=3)
            ax.plot([th2], [y2], "o", ms=7.5, mfc="white", mec=SP.INK, mew=1.6, zorder=5)
    _shade(ax, gs, x_end)
    ax.errorbar(hours, mean, yerr=sd if len(gs.positions) > 1 else None, fmt="o", ms=4, color=SP.INK, ecolor="#9a9a9a", elinewidth=0.8, zorder=4)
    _cursor(ax, cursor_h); ax.set_xlim(hours[0] - 0.03 * (x_end - hours[0]), x_end + 0.03 * (x_end - hours[0])); ax.set_ylim(0, None)
    ax.set_xlabel("time after inoculation (h)"); ax.set_ylabel("number of cells")
    _key(fig, 0.16, [(SP.INK, dict(marker="o", ms=4, ls="none"), "counted"), (MODEL_GREY, dict(ls=(0, (5, 2.5))), "fitted growth curve")] if gs.th2 else
         [(SP.INK, dict(marker="o", ms=4, ls="none"), "counted")])
    return ax


@SP.with_house_style
def draw_rate(fig, gs: GrowthStates, cursor_h: Optional[float] = None, selected: Optional[int] = None):
    hours, rate, _sd = gs.mean_of("rate"); k = int(np.argmax(rate))
    ax = _axes(fig, "Growth rate", f"fastest at {hours[k]:g} h, {rate[k]:.0f} cells per hour")
    _shade(ax, gs); ax.plot(hours, rate, color=SP.INK, lw=2, zorder=3); ax.plot([hours[k]], [rate[k]], "o", ms=7, mfc="white", mec=SP.INK, mew=1.6, zorder=5)
    _cursor(ax, cursor_h); ax.axhline(0, color="#9a9a9a", lw=0.8); ax.set_xlabel("time after inoculation (h)"); ax.set_ylabel("cells per hour")
    return ax


@SP.with_house_style
def draw_matrix(fig, gs: GrowthStates, cursor_h: Optional[float] = None, selected: Optional[int] = None):
    if not has_matrix(gs):
        return _missing(fig, "Matrix stain and Th1")
    hours, mean, sd = gs.mean_of("matrix"); th1 = gs.th1["th1_h"] if gs.th1 else None
    ax = _axes(fig, "Matrix stain and Th1", f"Th1 {th1:.1f} h: the stain appears and stays" if th1 is not None else "no matrix stain that stays")
    _shade(ax, gs)
    if len(gs.positions) > 1:
        ax.fill_between(hours, mean - sd, mean + sd, color=MATRIX_GREEN, alpha=0.15, lw=0, zorder=2)
    ax.plot(hours, mean, color=MATRIX_GREEN, lw=2, zorder=3)
    if th1 is not None:
        ax.axvline(th1, color=SP.INK, lw=1.1, ls=(0, (4, 2)), zorder=2); ax.plot([th1], [float(np.interp(th1, hours, mean))], "o", ms=7, mfc="white", mec=SP.INK, mew=1.6, zorder=5)
    _cursor(ax, cursor_h); ax.set_ylim(0, max(float((mean + sd).max()), 1.0) * 1.1); ax.set_xlabel("time after inoculation (h)"); ax.set_ylabel("matrix stain (% of frame)")
    return ax


@SP.with_house_style
def draw_inside(fig, gs: GrowthStates, cursor_h: Optional[float] = None, selected: Optional[int] = None):
    if not has_matrix(gs):
        return _missing(fig, "Matrix stain inside and outside microcolonies")
    hours, ins, _a = gs.mean_of("inside"); _h, outs, _b = gs.mean_of("outside")
    ax = _axes(fig, "Matrix stain inside and outside microcolonies")
    _shade(ax, gs); ax.plot(hours, ins, color=MATRIX_GREEN, lw=2, zorder=3); ax.plot(hours, outs, color=OUTSIDE_GREEN, lw=2, zorder=3)
    _cursor(ax, cursor_h); top = np.nanmax(np.r_[ins, outs, 1.0]); ax.set_ylim(0, float(top) * 1.1)
    ax.set_xlabel("time after inoculation (h)"); ax.set_ylabel("matrix stain (% of area)")
    _key(fig, 0.16, [(MATRIX_GREEN, {}, "inside microcolonies"), (OUTSIDE_GREEN, {}, "outside")])
    return ax


@SP.with_house_style
def draw_positions(fig, gs: GrowthStates, cursor_h: Optional[float] = None, selected: Optional[int] = None):
    ax = _axes(fig, "Number of cells, every position", f"position {selected} in blue" if selected in gs.positions else "")
    _shade(ax, gs)
    for p in gs.positions:
        hi = p == selected; ax.plot(gs.hours[p], gs.cells[p], color=SP.BLUE if hi else "#9a9a9a", lw=2.2 if hi else 1.2, zorder=4 if hi else 3)
    _cursor(ax, cursor_h); ax.set_ylim(0, None); ax.set_xlabel("time after inoculation (h)"); ax.set_ylabel("number of cells")
    return ax


DRAW = {"cells": draw_cells, "rate": draw_rate, "matrix": draw_matrix, "inside": draw_inside, "positions": draw_positions}
