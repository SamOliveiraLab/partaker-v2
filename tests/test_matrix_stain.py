"""Matrix stain (Partaker 2, Figure 4): the dye image zeroed and flattened with the run's own dye frames, the free dye
level measured in open space, a cut-off above it, and a width rule so single stained cells are not matrix. Made-up
frames with known answers; run 2 on six positions against the numbers of the figure is a slow test (--run-slow)."""
from pathlib import Path

import numpy as np
import pytest

from nd2_analyzer.analysis.biofilm_stages import matrix_stain as MX

PX = 0.1                                                     # um per pixel of the toy frames


def test_the_dye_frames_zero_and_flatten_the_image():
    yy, xx = np.mgrid[:120, :120]
    light = 0.6 + 0.8 * xx / 119.0                           # the field is lit unevenly
    before = np.full((3, 120, 120), 200.0)                   # no dye: the camera offset
    after = 200.0 + 1000.0 * light[None].repeat(3, 0)        # free dye everywhere: the shape of the light
    ref = MX.dye_reference(before, after, sigma_px=3)
    assert ref.flat.mean() == pytest.approx(1.0)
    frame = 200.0 + 3000.0 * ref.flat                        # a frame with 3,000 counts of stain everywhere
    assert MX.correct(frame, ref) == pytest.approx(np.full((120, 120), 3000.0), rel=1e-4)


def test_open_space_is_more_than_one_and_a_half_microns_from_any_cell():
    labels = np.zeros((100, 100), np.int32); labels[40:50, 40:60] = 1
    space = MX.open_space(labels, PX)
    assert not space[45, 50] and not space[45, 70] and space[45, 80] and not space[30, 50] and space[20, 50]


def test_the_free_dye_level_comes_from_frames_with_enough_open_space():
    empty = np.zeros((100, 100), np.int32); crowded = np.ones((100, 100), np.int32); crowded[:20] = 0   # no open space left
    frames = [(np.full((100, 100), 2000.0), empty), (np.full((100, 100), 2100.0), empty), (np.full((100, 100), 9000.0), crowded)]
    level, used = MX.free_dye_level(frames, PX)
    assert level == 2050.0 and used == 2                     # the crowded frame is not used


def test_wide_stain_above_the_cut_off_is_matrix_and_a_stained_cell_is_not():
    stain = np.zeros((200, 200), np.float32)
    stain[20:70, 20:70] = 6000.0                             # 5 um wide, above the cut-off: matrix
    stain[120:130, 20:120] = 9000.0                          # 1 um wide, brighter: a single stained cell
    stain[150:190, 150:190] = 4000.0                         # wide but under the cut-off
    counted, single = MX.matrix(stain, PX)
    assert counted[45, 45] and counted[20:70, 20:70].sum() > 2000 and not counted[125, 70] and not counted[170, 170]
    assert single[125, 70] and not single[45, 45] and not single[170, 170]
    assert MX.share(counted) == pytest.approx(100.0 * counted.sum() / stain.size)


def test_stain_is_smoothed_and_counted_from_the_free_dye_level():
    corrected = np.full((60, 60), 2500.0, np.float32)
    assert MX.stain(corrected, level=2000.0) == pytest.approx(np.full((60, 60), 500.0), abs=1e-2)


def test_matrix_inside_and_outside_microcolonies():
    labels = np.zeros((200, 200), np.int32)
    for i, x in enumerate((20, 43, 66, 89)):                 # four cells 0.3 um apart: one microcolony
        labels[40:50, x:x + 20] = i + 1
    labels[150:160, 150:170] = 5                             # a cell alone: not a microcolony
    counted = np.zeros((200, 200), bool); counted[30:60, 10:65] = True; counted[140:170, 140:180] = True
    inside, outside, share_inside, area = MX.inside_outside(counted, labels, PX)
    colony = MX.microcolony_area(labels, PX)
    assert colony[45, 30] and colony[45, 41] and not colony[155, 160] and not colony[100, 100]      # the cells and the gaps between them
    assert area == pytest.approx(100.0 * colony.mean())
    assert inside == pytest.approx(100.0 * (counted & colony).sum() / colony.sum())
    assert outside == pytest.approx(100.0 * (counted & ~colony).sum() / (~colony).sum())
    assert share_inside == pytest.approx(100.0 * (counted & colony).sum() / counted.sum()) and 0 < share_inside < 100


def test_no_microcolony_means_no_inside():
    labels = np.zeros((100, 100), np.int32); labels[40:50, 40:60] = 1
    inside, outside, share_inside, area = MX.inside_outside(np.ones((100, 100), bool), labels, PX)
    assert np.isnan(inside) and outside == 100.0 and share_inside == 0.0 and area == 0.0


# ---------------------------------------------------------------- run 2, from the raw dye channel and the tool's own masks
RUN2 = Path.home() / "Workspace/Datasets/ecoli-run2-june30-2026"
MAIN, BEFORE, AFTER = RUN2 / "mcherry-tl-t24-t72-07-01-2026.nd2", RUN2 / "extra_3_24h/mcherry-tl-t5-b4-dye06-30-2027.nd2", RUN2 / "extra_3_24h/mcherry-tl-t5-06-30-2028.nd2"
P2_LABELS = Path.home() / "Workspace/results/partaker2_fixtures/run2_p2_labels"            # the tool's own masks of position 2
DYE_CHANNEL = 2


@pytest.fixture(scope="module")
def run2():
    if not (MAIN.exists() and BEFORE.exists() and AFTER.exists() and (P2_LABELS / "p2_t66.npz").exists()):
        pytest.skip("run 2, its dye frames or the tool's masks are not on this machine")
    import nd2
    f = nd2.ND2File(str(MAIN)); frames = f.to_dask(); pixel_um = float(f.voxel_size().x)
    ref = MX.dye_reference(MX.read_dye_frames(BEFORE, DYE_CHANNEL), MX.read_dye_frames(AFTER, DYE_CHANNEL))
    yield frames, pixel_um, ref
    f.close()


@pytest.mark.slow
def test_run2_position2_matrix_stain_is_the_one_of_figure4(run2):
    """figure4_new/cache/dye_levels.json and cutoff_frames.csv, position 2, 17 Nov 2026."""
    frames, pixel_um, ref = run2
    green = lambda t: MX.correct(np.asarray(frames[t, 2, DYE_CHANNEL]), ref)
    labels = lambda t: np.load(P2_LABELS / f"p2_t{t}.npz")["cell_labels"]
    level, used = MX.free_dye_level(((green(t), labels(t)) for t in range(0, 25, 3)), pixel_um)
    assert level == pytest.approx(2050.53, abs=0.5) and used == 9
    for t, matrix_pct in ((0, 0.0), (12, 0.0), (30, 0.932), (36, 2.991), (48, 12.458), (66, 41.979)):       # 24, 30, 39, 42, 48, 57 h
        counted, single = MX.matrix(MX.stain(green(t), level), pixel_um)
        assert MX.share(counted) == pytest.approx(matrix_pct, abs=0.002), t
        if t == 0:
            assert MX.share(single) == pytest.approx(0.972, abs=0.002)                                      # single stained cells at 24 h


@pytest.mark.slow
def test_run2_position2_matrix_inside_and_outside_microcolonies_is_the_one_of_figure4(run2):
    """figure4_new/cache/inside_frames.csv, position 2: (frame, % of the frame in microcolonies, % stained inside, % stained outside, % of the stain that is inside)."""
    frames, pixel_um, ref = run2
    green = lambda t: MX.correct(np.asarray(frames[t, 2, DYE_CHANNEL]), ref)
    labels = lambda t: np.load(P2_LABELS / f"p2_t{t}.npz")["cell_labels"]
    level, _used = MX.free_dye_level(((green(t), labels(t)) for t in range(0, 25, 3)), pixel_um)
    for t, area, inside, outside, share_inside in ((36, 57.886, 3.873, 1.779, 74.952), (48, 65.539, 14.913, 7.790, 78.454), (66, 74.824, 44.793, 33.615, 79.840)):
        counted, _single = MX.matrix(MX.stain(green(t), level), pixel_um)
        got = MX.inside_outside(counted, labels(t), pixel_um)
        assert got == pytest.approx((inside, outside, share_inside, area), abs=0.002), t
