"""Manual selection next to automatic detection (Partaker 2, Figure 3 c): outlines a person draws around the microcolonies
they see on a frame, against the rule, cell by cell. A cell is inside an outline when the pixel of the cell nearest its
centre is. Plain Python and matplotlib, no Qt."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.path import Path as Outline
from scipy import ndimage as ndi
from skimage.measure import regionprops

from . import seeding_plots as SP
from .microcolony_rules import MicrocolonySettings, drop_small_cells, linked_groups

FIGURE_IN = (10.0, 4.6)
AUTOMATIC = "#E08A00"          # the microcolony orange of Figure 3
CELL_GREY = "#6a6a6a"
Point = Tuple[float, float]    # (x, y) in pixels of the frame


@dataclass
class Comparison:
    cells: int
    both: int                  # in a microcolony for the person and for the rule
    manual_only: int
    automatic_only: int
    neither: int
    manual_count: int          # outlines drawn
    automatic_count: int       # microcolonies by the rule
    by_hand: Dict[int, int]    # cell label -> outline number (1..), 0 for none
    by_rule: Dict[int, int]    # cell label -> group number of its microcolony, 0 for none

    @property
    def agree_pct(self) -> float:
        return 100.0 * (self.both + self.neither) / self.cells if self.cells else 0.0


def compare(labels: np.ndarray, pixel_um: float, outlines: Sequence[Sequence[Point]], settings: MicrocolonySettings = MicrocolonySettings()) -> Comparison:
    cells = drop_small_cells(labels, pixel_um, settings.min_cell_um2); groups = linked_groups(cells, pixel_um, settings.gap_um)
    paths = [Outline(np.asarray(poly, float)) for poly in outlines if len(poly) >= 3]
    by_hand, by_rule = {}, {}
    for p in regionprops(cells):
        cy, cx = p.centroid; co = p.coords; j = np.argmin((co[:, 0] - cy) ** 2 + (co[:, 1] - cx) ** 2); point = (float(co[j, 1]), float(co[j, 0]))
        by_hand[p.label] = next((k for k, path in enumerate(paths, 1) if path.contains_point(point)), 0)
        g = groups.cell_group[p.label]; by_rule[p.label] = g if groups.sizes[g] >= settings.min_cells else 0
    hand = np.array([by_hand[c] > 0 for c in by_hand], bool); rule = np.array([by_rule[c] > 0 for c in by_hand], bool)
    return Comparison(cells=len(by_hand), both=int((hand & rule).sum()), manual_only=int((hand & ~rule).sum()), automatic_only=int((rule & ~hand).sum()),
                      neither=int((~hand & ~rule).sum()), manual_count=len(paths), automatic_count=len({g for g in by_rule.values() if g}),
                      by_hand=by_hand, by_rule=by_rule)


def summary_words(c: Comparison) -> str:
    if c.manual_count == 0:
        return "No outline yet. Click around a microcolony to draw one."
    return f"Same answer for {c.agree_pct:.0f} % of the {c.cells:,} cells in this frame."


@SP.with_house_style
def draw_compare(fig, labels: np.ndarray, pixel_um: float, outlines: Sequence[Sequence[Point]], c: Comparison, title: str,
                 pending: Optional[List[Point]] = None, settings: MicrocolonySettings = MicrocolonySettings()):
    """The frame with the rule's microcolonies in orange and the hand outlines in white, and where the two agree, cell by
    cell. pending: the points of the outline being drawn. Returns the image axes."""
    fig.clear(); cells = drop_small_cells(labels, pixel_um, settings.min_cell_um2); h, w = cells.shape
    lut = np.zeros((int(cells.max()) + 1, 3)); lut[1:] = to_rgb(CELL_GREY)
    for cell, g in c.by_rule.items():
        if g:
            lut[cell] = to_rgb(AUTOMATIC)
    rgb = lut[cells]; mine = np.isin(cells, [cell for cell, g in c.by_rule.items() if g])
    aspect = fig.get_figwidth() / fig.get_figheight(); ax = fig.add_axes([0.03, 0.12, 0.34, 0.34 * aspect * h / w if h <= w else 0.34 * aspect])
    if mine.any():
        grown = ndi.distance_transform_edt(~mine) <= settings.gap_um / 2 / pixel_um; soft = ndi.gaussian_filter(grown.astype(np.float32), 0.45 / pixel_um)
        shade = np.clip(soft * 1.6, 0, 1)[..., None] * 0.28; rgb = np.where((cells == 0)[..., None], rgb * (1 - shade) + np.array(to_rgb(AUTOMATIC)) * shade, rgb)
        ax.imshow(rgb, interpolation="antialiased"); ax.contour(soft, levels=[0.22], colors=[AUTOMATIC], linewidths=1.1)
    else:
        ax.imshow(rgb, interpolation="antialiased")
    for poly in outlines:
        xy = np.asarray(list(poly) + [poly[0]], float); ax.plot(xy[:, 0], xy[:, 1], color="white", lw=1.8, zorder=6)
    if pending:
        xy = np.asarray(pending, float); ax.plot(xy[:, 0], xy[:, 1], color="white", lw=1.4, ls=(0, (3, 2)), marker="o", ms=3.5, mfc="white", mec="black", mew=0.6, zorder=7)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, w); ax.set_ylim(h, 0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.03, 0.955, title, fontsize=10.5, fontweight="bold", color=SP.INK, va="center")
    fig.text(0.03, 0.045, f"white: manual selection, {c.manual_count}      orange: automatic detection, {c.automatic_count}", fontsize=9.5, color=SP.INK, va="center")
    b = fig.add_axes([0.50, 0.24, 0.45, 0.50]); SP.open_frame(b)
    names = ["in a microcolony\nfor both", "in none\nfor both", "manual\nonly", "automatic\nonly"]; values = [c.both, c.neither, c.manual_only, c.automatic_only]
    for k, (v, colour) in enumerate(zip(values, (AUTOMATIC, "#8a8a8a", "white", "#f4c98a"))):
        b.bar(k, v, width=0.62, color=colour, edgecolor=SP.INK, linewidth=1.1); b.text(k, v, f"{v}\n", ha="center", va="bottom", fontsize=10, color=SP.INK, fontweight="bold", linespacing=0.4)
    b.set_xticks(range(4)); b.set_xticklabels(names); b.set_ylabel("cells"); b.set_ylim(0, max(max(values), 1) * 1.22); b.set_xlim(-0.6, 3.6)
    fig.text(0.50, 0.955, "manual selection and automatic detection", fontsize=10.5, fontweight="bold", color=SP.INK, va="center")
    fig.text(0.50, 0.885, summary_words(c), fontsize=9.5, color="#444444", va="center")
    return ax
