"""Manual selection next to automatic detection: outlines drawn by hand around microcolonies against the rule, cell by
cell. A toy frame with a hand-counted answer, one frame of run 2 with the outlines of Figure 3 c (skipped when run 2 or
those outlines are not on this machine), and the window where the outlines are drawn."""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nd2_analyzer.analysis.biofilm_stages import microcolony_compare as MC
from test_microcolony_rules import PX, row_of_cells

RUN2 = Path.home() / "Workspace/Datasets/ecoli-run2-june30-2026/mcherry-tl-t24-t72-07-01-2026.nd2"
FIGURE3 = Path.home() / "Workspace/results/figures_3to6/figure3_new"


def toy():
    labels = row_of_cells([3, 3, 3], y=20, shape=(80, 200))          # four linked cells: one automatic microcolony
    labels[60:70, 150:170] = 5                                       # a cell alone, far away
    return labels


def test_cells_are_compared_one_by_one():
    around_three = [(2, 15), (72, 15), (72, 35), (2, 35)]            # a hand outline around the first three cells only
    c = MC.compare(toy(), PX, [around_three])
    assert (c.cells, c.both, c.automatic_only, c.manual_only, c.neither) == (5, 3, 1, 0, 1)
    assert c.manual_count == 1 and c.automatic_count == 1 and c.agree_pct == 80.0


def test_a_hand_outline_around_a_lone_cell_is_a_disagreement():
    around_three = [(2, 15), (72, 15), (72, 35), (2, 35)]; around_lone = [(145, 55), (175, 55), (175, 75), (145, 75)]
    c = MC.compare(toy(), PX, [around_three, around_lone])
    assert (c.both, c.automatic_only, c.manual_only, c.neither) == (3, 1, 1, 0) and c.agree_pct == 60.0 and c.manual_count == 2


def test_no_outline_yet_means_nothing_selected_by_hand():
    c = MC.compare(toy(), PX, [])
    assert (c.both, c.automatic_only, c.manual_only, c.neither) == (0, 4, 0, 1) and c.manual_count == 0


def test_run2_frame_with_the_outlines_of_figure3c():
    """Position 3 at 27 h, the 45 um crop of Figure 3 c: 8 outlines by hand, 8 microcolonies by the rule, the same answer
    for 90.1 % of 101 cells (figure3_new/manual/compare.json, crop 2)."""
    if not RUN2.exists() or not (FIGURE3 / "scripts/manual_polygons.py").exists():
        pytest.skip("run 2 or the hand outlines of Figure 3 c are not on this machine")
    import json, nd2
    from nd2_analyzer.analysis.segmentation import segmentation_cache as sc
    from nd2_analyzer.analysis.segmentation.segmentation_models import SegmentationModels
    from nd2_analyzer.analysis.segmentation.segmentation_service import SegmentationService
    sys.path.insert(0, str(FIGURE3 / "scripts"))
    from manual_polygons import POLY
    where = json.loads((FIGURE3 / "manual/images.json").read_text())["1"]                # crop 2
    f = nd2.ND2File(str(RUN2)); frames = f.to_dask(); pixel_um = float(f.voxel_size().x)
    cache = sc.SegmentationCache(frames[where["t"]:where["t"] + 1, where["position"]:where["position"] + 1, 0:1], pixel_size_getter=lambda: pixel_um)
    labels = np.asarray(cache.with_model(SegmentationModels.OMNIPOSE_BACT_PHASE)[0, 0, 0]); f.close()
    labels = np.asarray(SegmentationService._remove_labels_with_centroids_near_edge(None, labels, SegmentationService.EDGE_CENTROID_MARGIN_PX))
    from nd2_analyzer.analysis.biofilm_stages.microcolony_rules import MicrocolonySettings, drop_small_cells
    labels = drop_small_cells(labels, pixel_um, 1.0)                 # as the figure: the size floor on the whole frame, then the crop
    crop = labels[where["y0"]:where["y0"] + where["S"], where["x0"]:where["x0"] + where["S"]]
    outlines = [[(x / pixel_um, y / pixel_um) for x, y in poly] for poly in POLY[2]]     # drawn in um, x to the right, y down
    c = MC.compare(crop, pixel_um, outlines, settings=MicrocolonySettings(min_cell_um2=0.0))   # a cell cut by the crop edge stays a cell
    assert (c.cells, c.both, c.manual_only, c.automatic_only, c.neither) == (101, 74, 3, 7, 17)
    assert c.manual_count == 8 and c.automatic_count == 8 and round(c.agree_pct, 1) == 90.1


def test_outlines_are_drawn_by_clicking_in_the_window():
    from nd2_analyzer.ui.biofilms.stages.microcolony_compare_window import MicrocolonyCompareDialog
    got = []
    dlg = MicrocolonyCompareDialog(toy(), PX, "toy frame", on_change=got.append)
    assert dlg.comparison.manual_count == 0 and dlg.summary.text() == "No outline yet. Click around a microcolony to draw one."
    for x, y in ((2, 15), (72, 15), (72, 35), (2, 35)):
        dlg.add_point(x, y)
    assert dlg.comparison.manual_count == 0                          # an outline counts once it is closed
    dlg.close_outline()
    assert dlg.comparison.manual_count == 1 and dlg.comparison.both == 3 and got[-1] == [[(2, 15), (72, 15), (72, 35), (2, 35)]]
    assert dlg.summary.text() == "Same answer for 80 % of the 5 cells in this frame."
    dlg.clear_outlines()
    assert dlg.comparison.manual_count == 0 and got[-1] == []
    dlg.close()


def test_the_compare_picture_draws_with_no_element_on_another():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from nd2_analyzer.analysis.biofilm_stages import seeding_plots as SP
    fig = Figure(figsize=MC.FIGURE_IN, dpi=100); FigureCanvasAgg(fig)
    outlines = [[(2, 15), (72, 15), (72, 35), (2, 35)]]
    MC.draw_compare(fig, toy(), PX, outlines, MC.compare(toy(), PX, outlines), "toy frame", pending=[(145, 55), (175, 55)])
    assert SP.collisions(fig) == []
