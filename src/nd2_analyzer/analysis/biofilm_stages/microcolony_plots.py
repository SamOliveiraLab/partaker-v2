"""What the Microcolonies page shows, read from the followed microcolonies (microcolony_follow.MicrocolonyTracks): the list
of microcolonies to follow, the cards, and the plots of the picker. Plain Python and matplotlib, no Qt: the page, the
exports and the tests draw the same way."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import figure_style as FS
from . import seeding_plots as SP
from .microcolony_follow import MicrocolonyTracks

FOLLOWED = "#8e6bbf"           # the microcolony picked in the list, and those that merge into it
BAR_GREY = "#b4b4b4"
FIGURES = (("count", "Number of microcolonies"), ("history", "Merging history"), ("cells", "Cells in microcolonies"),
           ("events", "Merging events"), ("waits", "Time before merging"))
FIGURE_DEFAULTS = ("count", "history")
FIGURE_SIZE_IN = (5.6, 4.6)


def words(n: int, one: str, many: str) -> str:
    return f"{n:,} {one if n == 1 else many}"


# ---------------------------------------------------------------------------------------------- numbers
def count_series(tracks: MicrocolonyTracks) -> Tuple[np.ndarray, np.ndarray]:
    """(hours, lasting microcolonies found at each frame)."""
    keep = tracks.lasting(); seen: Dict[int, set] = {t: set() for t in tracks.frames}
    for r in tracks.rows:
        if r["identity"] in keep:
            seen[r["time"]].add(r["identity"])
    return np.array([tracks.hours[t] for t in tracks.frames]), np.array([len(seen[t]) for t in tracks.frames])


def cells_series(tracks: MicrocolonyTracks) -> Tuple[np.ndarray, np.ndarray]:
    """(hours, % of the cells of each frame that are in a microcolony)."""
    pct = [100.0 * sum(1 for i in tracks.cell_identity[t].values() if i > 0) / max(len(tracks.cell_identity[t]), 1) for t in tracks.frames]
    return np.array([tracks.hours[t] for t in tracks.frames]), np.array(pct)


def merged_into(tracks: MicrocolonyTracks) -> Dict[int, List[Tuple[int, float]]]:
    """identity -> the lasting microcolonies that merged into it, with the hour of each merge."""
    kids: Dict[int, List[Tuple[int, float]]] = {}
    for e in tracks.merges():
        kids.setdefault(int(e["into"]), []).append((int(e["identity"]), float(e["hours"])))
    return kids


def subtree(tracks: MicrocolonyTracks, root: int) -> List[int]:
    """The microcolony and every one that merged into it, directly or through another."""
    kids = merged_into(tracks); out = [int(root)]
    for child, _hour in kids.get(int(root), []):
        out += subtree(tracks, child)
    return out


def end_hour(tracks: MicrocolonyTracks, identity: int) -> float:
    merged = [e for e in tracks.merges() if e["identity"] == identity]
    return float(merged[0]["hours"]) if merged else tracks.hours[tracks.identities[identity]["end"]]


def stain_words(summary: Optional[dict], identity: int) -> str:
    """The second line of a microcolony in the list, once the matrix stain of the position is measured."""
    s = (summary or {}).get(int(identity))
    if s is None:
        return ""
    if s["first_stain_h"] is None:
        return "\n     no matrix stain"
    return f"\n     matrix stain from {s['first_stain_h']:g} h, up to {s['most_cover_pct']:.0f} % of its area"


def entries(tracks: MicrocolonyTracks, stain: Optional[dict] = None) -> List[Tuple[str, List[Tuple[int, str]]]]:
    """The list of the page: (section title, [(identity, line)]), microcolonies still there first, then those that
    merged. stain (matrix_stain.microcolony_stain_summary): each line then says when the microcolony gets its matrix
    stain and how much of its area the stain covers at most."""
    keep = tracks.lasting(); kids = merged_into(tracks); last_h = tracks.hours[tracks.frames[-1]]
    born = lambda i: tracks.hours[tracks.identities[i]["birth"]]
    into = {int(e["identity"]): (int(e["into"]), float(e["hours"])) for e in tracks.merges()}
    still = sorted((i for i in keep if tracks.identities[i]["ended"] == "still there"), key=lambda i: (born(i), i))
    merged = sorted(into, key=lambda i: (born(i), i))
    there = [(i, f"Microcolony {i}: from {born(i):.1f} h" + (f", {len(kids[i])} merged into it" if kids.get(i) else "") + stain_words(stain, i)) for i in still]
    gone = [(i, f"Microcolony {i}: {born(i):.1f} to {into[i][1]:.1f} h, merges into {into[i][0]}" + stain_words(stain, i)) for i in merged]
    return [(f"still there at {last_h:g} h", there), ("merged into another", gone)]


def cards(tracks: MicrocolonyTracks) -> List[Tuple[str, str, bool]]:
    """The eight cards: (number, one plain line, whether the recording can give it)."""
    hours, count = count_series(tracks); _h, pct = cells_series(tracks); waits = tracks.hours_before_merging(); n_merges = len(tracks.merges())
    first_h, last_h = hours[0], hours[-1]; last = tracks.frames[-1]; keep = tracks.lasting()
    biggest = max((r["cells"] for r in tracks.rows if r["time"] == last and r["identity"] in keep), default=0)
    k = int(np.argmax(count))
    return [(f"{words(int(count[k]), 'microcolony', 'microcolonies')} at {hours[k]:g} h", "the most at one time", True),
            (words(n_merges, "merge", "merges"), "two microcolonies become one", True),
            (f"{np.median(waits):g} h before merging" if waits else "no merge yet", "median, from forming to merging", bool(waits)),
            (f"{len(keep):,} followed", f"microcolonies, {first_h:g} to {last_h:g} h", True),
            (f"{pct[-1]:.0f} % of cells at {last_h:g} h", "are in a microcolony", True),
            (f"{words(biggest, 'cell', 'cells')} at {last_h:g} h", "in the largest microcolony", biggest > 0),
            (f"{pct[0]:.1f} % of cells at {first_h:g} h", "are in a microcolony", True),
            (f"{int(count[-1])} left at {last_h:g} h", "after merging", True)]


# ---------------------------------------------------------------------------------------------- the strip
STAGE_SHADE = ("#e3e9ef", "#c9d5e0")           # microcolony formation, maturation: the blue-grey steps of the Maturation strip


def draw_strip(ax, span: Tuple[float, float], cursor_h: float, th1_h: Optional[float] = None):
    """The recording's hours with the cursor. Once Th1 is known (the Maturation tab measured it) and lies inside the
    recording: microcolony formation up to Th1 and maturation after it, as the Maturation strip shows states I and II."""
    lo, hi = float(span[0]), float(span[1]); x0, x1 = SP.pad_span((lo, hi), 0.03); ax.set_xlim(x0, x1); ax.set_ylim(0, 1); ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ticks = SP.hour_ticks((lo, hi)); ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g} h" for t in ticks], fontsize=8.5)
    split = th1_h is not None and lo < float(th1_h) < hi
    if not split:
        ax.axvspan(lo, hi, ymin=0.30, ymax=0.72, color="#f0f0f0", lw=0)
    else:
        th1 = float(th1_h); width_in = ax.get_position().width * ax.figure.get_figwidth()
        for name, a, b, shade in (("microcolony formation", lo, th1, STAGE_SHADE[0]), ("maturation", th1, hi, STAGE_SHADE[1])):
            ax.axvspan(a, b, ymin=0.30, ymax=0.72, color=shade, lw=0)
            if (b - a) / (x1 - x0) * width_in > len(name) * 9 * SP.IN_PER_PT_CHAR + 0.3:      # the name only where the band has room for it
                ax.text((a + b) / 2, 0.51, name, ha="center", va="center", fontsize=9, color=SP.INK)
        ax.axvline(th1, ymin=0.26, ymax=0.76, color=SP.INK, lw=1.1, ls=(0, (4, 2)))
        ax.text(th1, 0.90, f"Th1 {th1:.1f} h", ha="center", va="center", fontsize=8.5, fontweight="bold", color=SP.INK)
    ax.axvline(cursor_h, color=SP.RED, lw=2.0)
    left_free = not split or (float(th1_h) - lo) > 0.22 * (hi - lo)
    ax.text(lo if left_free else (lo + hi) / 2, 0.90, f"{cursor_h:g} h", ha="left" if left_free else "center", va="center", fontsize=8.5, fontweight="bold", color=SP.RED)


# ---------------------------------------------------------------------------------------------- plots
def _axes(fig, title: str, note: str = "", left: float = 0.15):
    fig.clear(); ax = fig.add_axes([left, 0.15, 0.95 - left, 0.62]); SP.open_frame(ax)
    fig.text(left, 0.93, title, fontsize=12, fontweight="bold", color=SP.INK, va="center")
    if note:
        fig.text(left, 0.865, note, fontsize=9.5, color="#555555", va="center")
    return ax


def _time_axis(ax, tracks: MicrocolonyTracks, cursor_h: Optional[float]):
    """Time (h) with round ticks from the first frame and the page's red cursor, as the panels of Figure 3."""
    FS.time_axis(ax, tracks.hours[tracks.frames[0]], tracks.hours[tracks.frames[-1]]); FS.cursor(ax, cursor_h)


@SP.with_house_style
def draw_count(fig, tracks: MicrocolonyTracks, cursor_h: Optional[float] = None, selected: Optional[int] = None):
    hours, count = count_series(tracks); k = int(np.argmin(np.abs(hours - cursor_h))) if cursor_h is not None else None
    ax = _axes(fig, "Number of microcolonies", f"{int(count[k])} at {hours[k]:g} h" if k is not None else "")
    ax.plot(hours, count, color=FS.INK, lw=1.6)
    if k is not None:
        ax.plot([hours[k]], [count[k]], "o", ms=7, mfc="white", mec=FS.INK, mew=1.8, zorder=5)
    ax.set_ylim(0, max(int(count.max()), 1) * 1.12); ax.set_ylabel("Microcolonies", fontweight="bold"); _time_axis(ax, tracks, cursor_h)
    return ax


@SP.with_house_style
def draw_cells(fig, tracks: MicrocolonyTracks, cursor_h: Optional[float] = None, selected: Optional[int] = None):
    hours, pct = cells_series(tracks); k = int(np.argmin(np.abs(hours - cursor_h))) if cursor_h is not None else None
    ax = _axes(fig, "Cells in microcolonies", f"{pct[k]:.0f} % at {hours[k]:g} h" if k is not None else "")
    ax.plot(hours, pct, color=FS.INK, lw=1.6)
    if k is not None:
        ax.plot([hours[k]], [pct[k]], "o", ms=7, mfc="white", mec=FS.INK, mew=1.8, zorder=5)
    ax.set_ylim(0, 105); ax.set_yticks([0, 50, 100]); ax.set_ylabel("Cells (%)", fontweight="bold"); _time_axis(ax, tracks, cursor_h)
    return ax


def forest(tracks: MicrocolonyTracks) -> List[int]:
    """Every lasting microcolony that merged or was merged into, top to bottom, each next to the one it merges into."""
    kids = merged_into(tracks); gone = {int(e["identity"]) for e in tracks.merges()}

    def lay(r):
        above, below = [], []
        for n, (c, _hour) in enumerate(sorted(kids.get(r, []), key=lambda v: v[1])):
            block = lay(c)
            if n % 2 == 0:
                above = block + above
            else:
                below = below + block
        return above + [r] + below

    roots = sorted((i for i in kids if i not in gone), key=lambda i: (tracks.identities[i]["birth"], i))
    return [i for r in roots for i in lay(r)]


@SP.with_house_style
def draw_history(fig, tracks: MicrocolonyTracks, cursor_h: Optional[float] = None, selected: Optional[int] = None):
    order = forest(tracks); ax = _axes(fig, "Merging history", "one bar per microcolony, ending where it merges", left=0.06)
    if not order:
        ax.axis("off"); ax.text(0.5, 0.5, "no merge in these frames", ha="center", va="center", transform=ax.transAxes, color=SP.EMPTY_GREY); return ax
    row = {i: k for k, i in enumerate(order)}; into = {int(e["identity"]): int(e["into"]) for e in tracks.merges()}
    mine = set(subtree(tracks, selected)) if selected in row else set()
    for i in order:
        hi = i in mine; b0, e0 = tracks.hours[tracks.identities[i]["birth"]], end_hour(tracks, i)
        ax.plot([b0, e0], [row[i], row[i]], color=FOLLOWED if hi else BAR_GREY, lw=2.6 if hi else 1.5, solid_capstyle="butt", zorder=4 if hi else 3)
        if i in into:
            both = hi and into[i] in mine
            ax.plot([e0, e0], [row[i], row[into[i]]], color=FOLLOWED if both else "#8a8a8a", lw=0.9 if both else 0.6, zorder=2)
    if selected in row:
        ax.text(tracks.hours[tracks.identities[selected]["birth"]], row[selected], f"{selected} ", ha="right", va="center", fontsize=9.5, color=FOLLOWED, fontweight="bold")
    ax.set_ylim(len(order) + 0.5, -1.5); ax.set_yticks([]); ax.spines["left"].set_visible(False); _time_axis(ax, tracks, cursor_h)
    ax._microcolony_rows = row                     # the page reads it to pick the bar under a click
    return ax


@SP.with_house_style
def draw_events(fig, tracks: MicrocolonyTracks, cursor_h: Optional[float] = None, selected: Optional[int] = None, bin_h: float = 3.0):
    hours = [e["hours"] for e in tracks.merges()]; first_h, last_h = tracks.hours[tracks.frames[0]], tracks.hours[tracks.frames[-1]]
    edges = np.arange(first_h, last_h + bin_h, bin_h); n, _ = np.histogram(hours, edges)
    ax = _axes(fig, "Merging events", f"{words(len(hours), 'merge', 'merges')}, counted every {bin_h:g} h")
    ax.bar(edges[:-1] + bin_h / 2, n, width=bin_h * 0.86, color=FS.G1, edgecolor="none")
    ax.set_ylim(0, max(int(n.max()) if len(n) else 0, 1) * 1.15); ax.set_ylabel(f"Merges per {bin_h:g} h", fontweight="bold"); _time_axis(ax, tracks, cursor_h)
    return ax


@SP.with_house_style
def draw_waits(fig, tracks: MicrocolonyTracks, cursor_h: Optional[float] = None, selected: Optional[int] = None):
    waits = np.array(tracks.hours_before_merging())
    ax = _axes(fig, "Time before merging", f"median {np.median(waits):g} h, {words(len(waits), 'microcolony', 'microcolonies')}" if len(waits) else "")
    if not len(waits):
        ax.axis("off"); ax.text(0.5, 0.5, "no microcolony formed and merged in these frames", ha="center", va="center", transform=ax.transAxes, color=SP.EMPTY_GREY); return ax
    edges = np.arange(0, waits.max() + 2.0, 2.0); n, _ = np.histogram(waits, edges)
    ax.bar(edges[:-1] + 1.0, n, width=1.72, color=FS.G1, edgecolor="none"); ax.axvline(float(np.median(waits)), color=FS.INK, lw=1.0, ls=FS.DASH)
    ax.set_ylim(0, int(n.max()) * 1.15); ax.set_ylabel("Microcolonies", fontweight="bold"); ax.set_xlabel("Time before merging (h)", fontweight="bold")
    return ax


DRAW = {"count": draw_count, "history": draw_history, "cells": draw_cells, "events": draw_events, "waits": draw_waits}
