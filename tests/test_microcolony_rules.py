"""The microcolony rule of Partaker 2 (Figure 3, Nov 2026): cells whose outlines are less than 0.5 um apart are linked, and
a linked group of 4 or more cells is a microcolony. Toy frames with hand-counted answers, and run 2 position 2 through
the tool's own cell detection against the numbers of the figure (skipped when run 2 is not on this machine)."""
from pathlib import Path

import numpy as np
import pytest

from nd2_analyzer.analysis.biofilm_stages import microcolony_rules as MR

RUN2 = Path.home() / "Workspace/Datasets/ecoli-run2-june30-2026/mcherry-tl-t24-t72-07-01-2026.nd2"
PX = 0.1                                             # um per pixel of the toy frames: a 10 x 20 px cell is 1 x 2 um


def row_of_cells(gaps_px, y=20, shape=(60, 200)):
    """Cells of 10 x 20 px in a row; gaps_px[i] background pixels between cell i and cell i + 1."""
    labels = np.zeros(shape, np.int32)
    x = 5
    for i in range(len(gaps_px) + 1):
        labels[y:y + 10, x:x + 20] = i + 1
        x += 20 + (gaps_px[i] if i < len(gaps_px) else 0)
    return labels


def test_cells_less_than_half_a_micron_apart_are_linked():
    labels = row_of_cells([3, 4, 6, 5])              # gaps of 0.3, 0.4, 0.6 and 0.5 um
    groups = MR.linked_groups(labels, PX)
    assert groups.cell_group[1] == groups.cell_group[2] == groups.cell_group[3]          # 0.3 and 0.4 um: linked
    assert len({groups.cell_group[3], groups.cell_group[4], groups.cell_group[5]}) == 3  # 0.6 and 0.5 um: not linked
    assert sorted(groups.sizes[groups.sizes > 0].tolist()) == [1, 1, 3]


def test_four_linked_cells_make_a_microcolony_and_three_do_not():
    labels = np.zeros((80, 200), np.int32)
    labels[:60] = row_of_cells([3, 3, 3], y=10, shape=(60, 200))                         # four linked cells
    three = row_of_cells([3, 3], y=0, shape=(20, 200))
    labels[60:80][three > 0] = three[three > 0] + 4                                      # three linked cells, far below
    frame = MR.frame_microcolonies(labels, PX)
    assert frame.n_cells == 7 and frame.n_groups == 2 and frame.n_microcolonies == 1
    assert [frame.cell_colony[c] for c in (1, 2, 3, 4)] == [1, 1, 1, 1] and [frame.cell_colony[c] for c in (5, 6, 7)] == [0, 0, 0]
    assert set(np.unique(frame.regions[labels > 0])) == {0, 1} and (frame.regions[labels == 5] == 0).all()
    assert frame.cells_in_microcolonies == 4


def test_cells_under_one_square_micron_are_not_cells():
    labels = row_of_cells([3, 3, 3])
    labels[40:45, 5:15] = 9                                                              # 0.5 um2 speck
    frame = MR.frame_microcolonies(labels, PX)
    assert frame.n_cells == 4 and 9 not in frame.cell_colony


@pytest.fixture(scope="module")
def run2_labels():
    if not RUN2.exists():
        pytest.skip("run 2 is not on this machine")
    import nd2
    from nd2_analyzer.analysis.segmentation import segmentation_cache as sc
    from nd2_analyzer.analysis.segmentation.segmentation_models import SegmentationModels
    from nd2_analyzer.analysis.segmentation.segmentation_service import SegmentationService
    f = nd2.ND2File(str(RUN2)); frames = f.to_dask(); pixel_um = float(f.voxel_size().x)

    def labels(t, p=2):
        cache = sc.SegmentationCache(frames[t:t + 1, p:p + 1, 0:1], pixel_size_getter=lambda: pixel_um)
        lab = np.asarray(cache.with_model(SegmentationModels.OMNIPOSE_BACT_PHASE)[0, 0, 0])
        return np.asarray(SegmentationService._remove_labels_with_centroids_near_edge(None, lab, SegmentationService.EDGE_CENTROID_MARGIN_PX))

    yield labels, pixel_um
    f.close()


# (frame, cells, linked groups, microcolonies, % of cells in them): figure3_new/cache/frames.log, position 2, 16 Nov 2026
@pytest.mark.parametrize("t, cells, groups, colonies, pct", [(0, 124, 57, 5, 40), (6, 204, 69, 14, 62), (48, 1584, 34, 5, 97)])
def test_run2_position2_frames_give_the_microcolonies_of_figure3(run2_labels, t, cells, groups, colonies, pct):
    labels, pixel_um = run2_labels
    frame = MR.frame_microcolonies(labels(t), pixel_um)
    assert (frame.n_cells, frame.n_groups, frame.n_microcolonies) == (cells, groups, colonies)
    assert round(100 * frame.cells_in_microcolonies / frame.n_cells) == pct


# ---------------------------------------------------------------- following microcolonies through time
def blocks(gap_px, shape=(120, 200)):
    """Two blocks of four cells (2 x 2, cells 10 x 20 px, 3 px apart inside a block), gap_px between the two blocks."""
    labels = np.zeros(shape, np.int32)
    x0 = 20
    for b in range(2):
        for i, (dy, dx) in enumerate(((0, 0), (0, 23), (13, 0), (13, 23))):
            labels[40 + dy:50 + dy, x0 + dx:x0 + dx + 20] = 4 * b + i + 1
        x0 += 43 + gap_px
    return labels


def test_two_microcolonies_that_touch_and_stay_together_for_an_hour_merge():
    from nd2_analyzer.analysis.biofilm_stages import microcolony_follow as MF
    frames = [blocks(20), blocks(20), blocks(3), blocks(3), blocks(3), blocks(3)]        # apart for 1 h, then 0.3 um apart
    tracks = MF.follow_microcolonies(lambda t: frames[t], range(6), pixel_um=PX, dt_h=0.5, start_hour=24.0)
    assert [tracks.count_at(t) for t in range(6)] == [2, 2, 1, 1, 1, 1]
    merges = tracks.merges()
    assert len(merges) == 1 and merges[0]["time"] == 2 and merges[0]["hours"] == 25.0
    assert {merges[0]["identity"], merges[0]["into"]} == {1, 2}
    assert tracks.cell_identity[5][1] == tracks.cell_identity[5][8] == merges[0]["into"]


def test_a_touch_that_comes_apart_within_the_hour_is_not_a_merge():
    from nd2_analyzer.analysis.biofilm_stages import microcolony_follow as MF
    frames = [blocks(20), blocks(20), blocks(3), blocks(20), blocks(20), blocks(20)]     # one frame of contact only
    tracks = MF.follow_microcolonies(lambda t: frames[t], range(6), pixel_um=PX, dt_h=0.5, start_hour=24.0)
    assert tracks.merges() == [] and [tracks.count_at(t) for t in (0, 3, 5)] == [2, 2, 2]


def test_a_group_seen_in_one_frame_only_is_not_a_lasting_microcolony():
    from nd2_analyzer.analysis.biofilm_stages import microcolony_follow as MF
    alone = np.zeros((120, 200), np.int32)
    frames = [blocks(20), blocks(20), blocks(20), blocks(20)]
    extra = row_of_cells([3, 3, 3], y=90, shape=(120, 200)); flash = frames[1].copy(); flash[extra > 0] = extra[extra > 0] + 8
    frames[1] = flash                                                                    # a third group, in frame 1 only
    tracks = MF.follow_microcolonies(lambda t: frames[t], range(4), pixel_um=PX, dt_h=0.5, start_hour=24.0)
    assert len(tracks.identities) == 3 and len(tracks.lasting()) == 2
    assert [tracks.count_at(t) for t in range(4)] == [2, 2, 2, 2]


RUN2_LABELS = Path.home() / "Workspace/results/partaker2_fixtures/run2_p2_labels"      # the tool's own masks of position 2, saved from its window
FIGURE_DRIFT = Path.home() / "Workspace/results/fig2_v3/run2_p2_every_frame/summary_by_pair.csv"


@pytest.fixture(scope="module")
def run2_masks():
    if not (RUN2_LABELS / "p2_t66.npz").exists():
        pytest.skip("the run 2 masks of the tool are not on this machine")
    import json
    meta = json.loads((RUN2_LABELS / "meta.json").read_text())
    return (lambda t: np.load(RUN2_LABELS / f"p2_t{t}.npz")["cell_labels"]), meta["pixel_um"]


@pytest.mark.slow
def test_run2_position2_merging_history_from_the_tool_alone(run2_masks):
    """Position 2, 24 to 57 h, every 30 min, everything from the tool's own masks. Figure 3 (16 Nov 2026) has 67 merges and
    77 followed microcolonies: the tool has one merge fewer because it measures the drift of the view on its own masks,
    which hold no object under 1 um2, and that moves the drift by one pixel on 5 of the 66 frame pairs (next test)."""
    from nd2_analyzer.analysis.biofilm_stages import microcolony_follow as MF
    labels_at, pixel_um = run2_masks
    tracks = MF.follow_microcolonies(labels_at, range(0, 67), pixel_um=pixel_um, dt_h=0.5, start_hour=24.0)
    assert len(tracks.lasting()) == 76 and len(tracks.merges()) == 66
    waits = tracks.hours_before_merging()
    assert len(waits) == 40 and float(np.median(waits)) == 3.5
    assert [tracks.count_at(t) for t in (6, 36, 66)] == [14, 18, 3]                      # 27, 42 and 57 h, as in Figure 3


@pytest.mark.slow
def test_run2_position2_merging_history_is_the_one_of_figure3_with_its_drift_table(run2_masks):
    """The same masks with the drift table Figure 3 used: figure3_new/figure3_v4_legend.md and cache/p2_metrics.json."""
    from nd2_analyzer.analysis.biofilm_stages import microcolony_follow as MF
    if not FIGURE_DRIFT.exists():
        pytest.skip("the drift table of the figure is not on this machine")
    import csv
    rows = sorted(csv.DictReader(FIGURE_DRIFT.open()), key=lambda r: int(r["frame"]))
    drift = {int(r["frame"]) + 1: (int(r["drift_dy_px"]), int(r["drift_dx_px"])) for r in rows}
    labels_at, pixel_um = run2_masks
    tracks = MF.follow_microcolonies(labels_at, range(0, 67), pixel_um=pixel_um, dt_h=0.5, start_hour=24.0, drift=drift)
    assert len(tracks.lasting()) == 77 and len(tracks.merges()) == 67
    waits = tracks.hours_before_merging()
    assert len(waits) == 41 and float(np.median(waits)) == 3.5
    assert [tracks.count_at(t) for t in (6, 36, 66)] == [14, 18, 3]
