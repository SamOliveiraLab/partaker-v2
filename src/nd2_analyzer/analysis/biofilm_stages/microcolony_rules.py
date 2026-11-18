"""The microcolony rule (Partaker 2, Figure 3): close, enough, lasting. Plain Python on the cell labels of a frame.

close   two cells are linked when their outlines are less than gap_um apart (0.5 um, about one cell width)
enough  a linked group of at least min_cells cells (4) is a microcolony
lasting found again hold_h later (1 h); that part needs the frames in time and is done by the follower

Cells smaller than min_cell_um2 (1 um2) are debris and are not cells for any rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import ndimage as ndi

EIGHT_CONNECTED = np.ones((3, 3), bool)


@dataclass(frozen=True)
class MicrocolonySettings:
    gap_um: float = 0.5
    min_cells: int = 4
    hold_h: float = 1.0
    min_cell_um2: float = 1.0


@dataclass
class LinkedGroups:
    group_map: np.ndarray          # every cell grown by half the gap; touching grown cells share a group number, 0 elsewhere
    cell_group: Dict[int, int]     # cell label -> group number
    sizes: np.ndarray              # sizes[group number] = cells in the group


@dataclass
class FrameMicrocolonies:
    regions: np.ndarray            # the grown groups that are microcolonies, numbered 1..n, 0 elsewhere
    cell_colony: Dict[int, int]    # cell label -> microcolony number, 0 for a cell in none
    n_cells: int
    n_groups: int
    n_microcolonies: int
    cells_in_microcolonies: int


def drop_small_cells(labels: np.ndarray, pixel_um: float, min_cell_um2: float) -> np.ndarray:
    """The labels with every object under min_cell_um2 removed."""
    labels = np.asarray(labels).astype(np.int32)
    areas_um2 = np.bincount(labels.ravel()) * pixel_um ** 2
    small = np.flatnonzero(areas_um2 < min_cell_um2)
    small = small[small > 0]
    if small.size:
        labels[np.isin(labels, small)] = 0
    return labels


def linked_groups(labels: np.ndarray, pixel_um: float, gap_um: float = MicrocolonySettings.gap_um, distance: Optional[np.ndarray] = None) -> LinkedGroups:
    """Groups of cells whose outlines are less than gap_um apart. distance: the distance of every pixel to the nearest
    cell (px), when the caller has it already."""
    labels = np.asarray(labels)
    if distance is None:
        distance = ndi.distance_transform_edt(labels == 0)
    group_map, _ = ndi.label(distance <= gap_um / 2 / pixel_um, structure=EIGHT_CONNECTED)
    group_of = np.zeros(int(labels.max()) + 1, np.int64)
    inside = labels > 0
    group_of[labels[inside]] = group_map[inside]                  # a cell is one piece, so all its pixels are in one group
    cells = np.flatnonzero(group_of)
    cell_group = {int(cell): int(group_of[cell]) for cell in cells}
    sizes = np.bincount(group_of[cells], minlength=int(group_map.max()) + 1)
    return LinkedGroups(group_map, cell_group, sizes)


def frame_microcolonies(labels: np.ndarray, pixel_um: float, settings: MicrocolonySettings = MicrocolonySettings()) -> FrameMicrocolonies:
    """The microcolonies of one frame by close and enough."""
    cells = drop_small_cells(labels, pixel_um, settings.min_cell_um2)
    groups = linked_groups(cells, pixel_um, settings.gap_um)
    big = np.flatnonzero(groups.sizes >= settings.min_cells)
    big = big[big > 0]
    number = np.zeros(groups.sizes.size, np.int32)
    number[big] = np.arange(1, big.size + 1)
    cell_colony = {cell: int(number[group]) for cell, group in groups.cell_group.items()}
    return FrameMicrocolonies(regions=number[groups.group_map], cell_colony=cell_colony, n_cells=len(cell_colony),
                              n_groups=int((groups.sizes[1:] > 0).sum()), n_microcolonies=int(big.size),
                              cells_in_microcolonies=int(groups.sizes[big].sum()))
