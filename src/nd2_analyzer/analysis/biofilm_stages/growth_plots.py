"""What the Maturation page shows, read from a growth-states measure (growth_measure.GrowthStates): the positions to
follow, the cards, the strip with the states, and the plots of the picker. Plain Python and matplotlib, no Qt."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from . import figure_style as FS
from . import growth_states as GS
from . import seeding_plots as SP
from .growth_measure import GrowthStates

STATE_SHADE = FS.STATE_SHADE                 # calm blue-grey steps, as Figure 5
MATRIX_GREEN, MICROCOLONY_ORANGE, OUTSIDE_GREY, MODEL_GREY = FS.MATRIX_GREEN, FS.MICROCOLONY_ORANGE, FS.OUTSIDE_GREY, FS.PREDICTED_GREY
# the plots of the picker, named as the panels they are drawn after: Figure 5 d, Figure 5 b, Figure 4 d
FIGURES = (("cells", "Cell count and growth rate, Th2"), ("matrix", "Microcolonies and matrix stain"),
           ("inside", "Matrix stain over time"), ("positions", "Number of cells, every position"))
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
# Drawn as the panels of the figures are (make_figure5_s2.py d, make_figure5_s1.py b, make_figure4_v3.py d), with the
# tool's own numbers; limits and label places come from the data. The cursor of the page is a red line.
LEFT, RIGHT = FS.CARD_LEFT, FS.CARD_RIGHT


def _missing(fig, key: str):
    ax = FS.card(fig, dict(FIGURES)[key]); ax.axis("off")
    ax.text(0.5, 0.5, NO_DYE, ha="center", va="center", transform=ax.transAxes, color=SP.EMPTY_GREY)
    return ax


def _who(gs: GrowthStates) -> str:
    return f"{len(gs.positions)} positions" if len(gs.positions) > 1 else f"position {gs.positions[0]}"


def _th2(gs: GrowthStates) -> Optional[dict]:
    return gs.th2 if gs.th2 and np.isfinite(gs.th2["th2_h"]) else None


@SP.with_house_style
def draw_cells(fig, gs: GrowthStates, cursor_h: Optional[float] = None, selected: Optional[int] = None, shared: Optional[dict] = None):
    """Figure 5 d: the cells counted per frame above, their growth rate below, one time axis; the fitted curve, the
    fastest growth, 10 % of it, and Th2 with its range."""
    fig.clear(); FS.title(fig, LEFT, "Cell count and growth rate, Th2")
    hours, mean, sd = gs.mean_of("cells"); _h, rate, rate_sd = gs.mean_of("rate"); many = len(gs.positions) > 1
    first, last = float(hours[0]), float(hours[-1]); th2 = _th2(gs); th1 = gs.th1["th1_h"] if has_matrix(gs) and gs.th1 else None
    x_end = last
    if th2 is not None and not th2["reached"]:                        # room for the prediction, but not for a range without end
        high = th2["range95_h"][1]; x_end = min(max(th2["th2_h"], high if np.isfinite(high) else th2["th2_h"]), first + 2.5 * (last - first))
        x_end = max(x_end, last)
    top = fig.add_axes([LEFT, FS.AXES_BOTTOM + 0.275, RIGHT - LEFT, FS.AXES_TOP - FS.AXES_BOTTOM - 0.275]); low = fig.add_axes([LEFT, FS.AXES_BOTTOM, RIGHT - LEFT, 0.225])
    for ax in (top, low):
        SP.open_frame(ax)
    FS.time_axis(top, first, x_end, show=False); FS.time_axis(low, first, x_end); lo, hi = top.get_xlim()
    states = []
    if th1 is not None:
        end2 = min(th2["th2_h"], hi) if th2 is not None else hi
        states = [("I", lo, th1), ("II", th1, end2)] + ([("III", end2, hi)] if end2 < hi else [])
    for ax in (top, low):
        FS.bands(ax, states)
        if x_end > last + 0.08 * (x_end - first):
            ax.axvline(last, color=FS.G2, lw=1.0, zorder=1)                     # where the recording ends and the prediction begins
    # the count: ribbon, dots, and the fitted curve (solid over the recording, dashed after it)
    t = np.linspace(first, hi, 400); model = GS.gompertz(t, th2["K"], th2["r"], th2["t0"]) if th2 is not None else None
    if many:
        FS.ribbon(top, hours, mean, sd, FS.INK)
    FS.dots(top, hours, mean, FS.INK, joined=False)
    if model is not None:
        top.plot(t[t <= last], model[t <= last], "-", color=FS.INK, lw=1.4, zorder=3); top.plot(t[t >= last], model[t >= last], "--", color=FS.PREDICTED_GREY, lw=1.6, zorder=3)
    highest = max(float(np.nanmax(mean + (sd if many else 0))), float(model.max()) if model is not None else 0.0, 1.0)
    FS.count_axis(top, highest * 1.38, "Cells per frame"); FS.band_names(top, states, y=0.88)
    if x_end > last + 0.08 * (x_end - first):
        top.text(last + 0.012 * (hi - lo), 0.06, "recording\nends", transform=top.get_xaxis_transform(), fontsize=8.5, ha="left", va="bottom", color=FS.G1)
    # the growth rate: ribbon, joined dots, the rate of the fitted curve, its fastest point
    k = int(np.argmax(rate)); peak_shown = float(np.nanmax(rate + (rate_sd if many else 0)))
    low.axhline(0, color=FS.G2, lw=1.0, zorder=1)
    if many:
        FS.ribbon(low, hours, rate, rate_sd, FS.INK)
    FS.dots(low, hours, rate, FS.INK, joined=True, width=1.3)
    low.plot([hours[k]], [rate[k]], "D", ms=6.5, mfc="white", mec=FS.INK, mew=1.4, zorder=6)
    ceiling = max(peak_shown, 1e-9) * 1.5; low.set_ylim(-0.08 * ceiling, ceiling)
    from matplotlib.ticker import MaxNLocator
    low.set_yticks([v for v in MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(0, ceiling) if 0 <= v <= ceiling])
    low.set_ylabel("Growth rate\n(cells per h)", fontweight="bold")
    right_of_peak = hours[k] + 0.03 * (hi - lo) < lo + 0.55 * (hi - lo)
    low.text(hours[k] + (0.03 if right_of_peak else -0.03) * (hi - lo), peak_shown * 1.1, f"fastest growth, {hours[k]:g} h", fontsize=9,
             ha="left" if right_of_peak else "right", va="bottom", color=FS.INK)
    if th2 is not None:
        tenth = 0.1 * th2["peak_rate"]; low.plot(t, np.gradient(model, t), "--", color=FS.PREDICTED_GREY, lw=1.6, zorder=3)
        low.axhline(tenth, color=FS.INK, lw=1.0, ls=":", zorder=2)
        a, b = th2["range95_h"]
        if np.isfinite(a) and np.isfinite(b):
            low.plot([max(a, lo), min(b, hi)], [tenth, tenth], "-", color=FS.INK, lw=3.0, solid_capstyle="butt", zorder=5)
        if lo <= th2["th2_h"] <= hi:
            low.plot([th2["th2_h"]], [tenth], "o", ms=7.5, color=FS.INK, zorder=6)
        if hours[k] < lo + 0.6 * (hi - lo):                                     # the right end is free of the measured curve's peak
            low.text(hi - 0.012 * (hi - lo), tenth + 0.07 * ceiling, "10 % of the\nfastest growth", fontsize=8.5, ha="right", va="bottom", color=FS.G1)
    FS.threshold(fig, top, th1, "Th1", others=(low,))
    if th2 is not None:
        FS.threshold(fig, top, th2["th2_h"], "Th2" if th2["reached"] else "Th2 (predicted)", others=(low,))
    for ax in (top, low):
        FS.cursor(ax, cursor_h)
    rows = [[("dots", FS.INK, f"measured, {_who(gs)}")] + ([("dashed", FS.PREDICTED_GREY, "Gompertz curve fitted to the count")] if th2 is not None else [])]
    if th2 is not None:
        a, b = th2["range95_h"]
        rows.append([("range", FS.INK, f"Th2 and its 95 % range ({a:.0f} to {b:.0f} h)")] if np.isfinite(a) and np.isfinite(b) else [("big dot", FS.INK, "Th2")])
    FS.legend_under(fig, LEFT, rows)
    return top


def _percent_axes(fig, key: str, label: str):
    """One panel in the card every plot shares."""
    ax = FS.card(fig, dict(FIGURES)[key])
    ax.set_ylim(0, 100); ax.set_yticks([0, 50, 100]); ax.set_ylabel(label, fontweight="bold")
    return ax


@SP.with_house_style
def draw_matrix(fig, gs: GrowthStates, cursor_h: Optional[float] = None, selected: Optional[int] = None, shared: Optional[dict] = None):
    """Figure 5 b: the matrix stain (% of the frame) with Th1 where it appears and stays, on the states I and II; and the
    cells in microcolonies (% of the cells) when the Microcolonies tab has measured every position (shared)."""
    if not has_matrix(gs):
        return _missing(fig, "matrix")
    hours, mean, sd = gs.mean_of("matrix"); th1 = gs.th1["th1_h"] if gs.th1 else None; many = len(gs.positions) > 1
    ax = _percent_axes(fig, "matrix", "Percentage"); first, last = float(hours[0]), float(hours[-1]); FS.time_axis(ax, first, last); lo, hi = ax.get_xlim()
    states = [("I", lo, th1), ("II", th1, hi)] if th1 is not None and lo < th1 < hi else []
    FS.bands(ax, states); rows = []
    share = (shared or {}).get("cells_in_pct")
    if share is not None:
        sh, sm, ss = (np.asarray(v, float) for v in share); keep = (sh >= first - 1e-9) & (sh <= last + 1e-9)
        if keep.any():
            FS.ribbon(ax, sh[keep], sm[keep], ss[keep], FS.MICROCOLONY_ORANGE, 0.25); FS.dots(ax, sh[keep], sm[keep], FS.MICROCOLONY_ORANGE, width=1.8)
            rows.append(("line", FS.MICROCOLONY_ORANGE, "cells in microcolonies (% of cells)"))
    if many:
        FS.ribbon(ax, hours, mean, sd, FS.MATRIX_GREEN, 0.25)
    FS.dots(ax, hours, mean, FS.MATRIX_GREEN, width=1.8); rows.append(("line", FS.MATRIX_GREEN, "matrix stain (% of frame)"))
    FS.band_names(ax, states, y=0.45)
    if states:
        ax.plot([th1], [float(np.interp(th1, hours, mean))], "o", ms=7.5, mfc="white", mec=FS.INK, mew=1.5, zorder=6, clip_on=False)   # Th1 on the stain line
    FS.threshold(fig, ax, th1, "Th1"); FS.cursor(ax, cursor_h); FS.legend_under(fig, LEFT, [rows])
    return ax


@SP.with_house_style
def draw_inside(fig, gs: GrowthStates, cursor_h: Optional[float] = None, selected: Optional[int] = None, shared: Optional[dict] = None):
    """Figure 4 d: the matrix stain inside the microcolonies and outside them, as a share of each area."""
    if not has_matrix(gs):
        return _missing(fig, "inside")
    hours, ins, ins_sd = gs.mean_of("inside"); _h, outs, outs_sd = gs.mean_of("outside"); many = len(gs.positions) > 1
    ax = _percent_axes(fig, "inside", "Area covered (%)"); FS.time_axis(ax, float(hours[0]), float(hours[-1]))
    for values, spread, colour in ((outs, outs_sd, FS.OUTSIDE_GREY), (ins, ins_sd, FS.MATRIX_GREEN)):
        if many:
            FS.ribbon(ax, hours, values, spread, colour, 0.16)
        FS.dots(ax, hours, values, colour, width=1.7)
    FS.cursor(ax, cursor_h)
    FS.legend_under(fig, LEFT, [[("line", FS.MATRIX_GREEN, "matrix stain inside microcolonies"), ("line", FS.OUTSIDE_GREY, "outside")]])
    return ax


@SP.with_house_style
def draw_positions(fig, gs: GrowthStates, cursor_h: Optional[float] = None, selected: Optional[int] = None, shared: Optional[dict] = None):
    """The count of every position, the selected one in black."""
    ax = FS.card(fig, "Number of cells, every position")
    first = min(float(gs.hours[p][0]) for p in gs.positions); last = max(float(gs.hours[p][-1]) for p in gs.positions); FS.time_axis(ax, first, last)
    for p in gs.positions:
        picked = p == selected; ax.plot(gs.hours[p], gs.cells[p], color=FS.INK if picked else FS.G2, lw=2.0 if picked else 1.2, zorder=4 if picked else 3)
    FS.count_axis(ax, max(float(max(gs.cells[p].max() for p in gs.positions)), 1.0) * 1.12, "Cells per frame"); FS.cursor(ax, cursor_h)
    rows = ([("line", FS.INK, f"position {selected}")] if selected in gs.positions else []) + [("line", FS.G2, "the other positions" if selected in gs.positions else "every position")]
    FS.legend_under(fig, LEFT, [rows])
    return ax


DRAW = {"cells": draw_cells, "matrix": draw_matrix, "inside": draw_inside, "positions": draw_positions}
