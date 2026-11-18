"""The growth-states measure: for every position and frame, the cells counted and the matrix stain measured, then Th1 and
the predicted Th2. A made-up recording of two positions with known answers."""
import numpy as np
import pytest

from nd2_analyzer.analysis.biofilm_stages import growth_measure as GM
from nd2_analyzer.analysis.biofilm_stages import matrix_stain as MX

PX = 0.1


def recording():
    """Two positions, 8 frames 1.5 h apart from 24 h. Cells: frame t holds 4 + 2 t cells (position 1 holds two more).
    Stain: a 5 um block that appears at frame 4 on position 0 and at frame 5 on position 1, over a free dye level of 2,000."""
    def labels(p, t):
        lab = np.zeros((240, 240), np.int32)
        for k in range(4 + 2 * t + (2 if p == 1 else 0)):
            y, x = 10 + 14 * (k // 9), 10 + 24 * (k % 9); lab[y:y + 10, x:x + 20] = k + 1
        return lab

    def dye(p, t):
        img = np.full((240, 240), 2000.0, np.float32)
        if t >= 4 + p:
            img[150:200, 150:200] = 2000.0 + 8000.0
        return img

    return labels, dye


def test_every_position_and_frame_is_counted_and_its_matrix_measured():
    labels, dye = recording()
    ref = MX.DyeReference(zero=np.zeros((240, 240), np.float32), flat=np.ones((240, 240), np.float32))
    out = GM.measure_growth_states(labels, dye, {0: list(range(8)), 1: list(range(8))}, pixel_um=PX, hour_of=lambda t: 24.0 + 1.5 * t, reference=ref)
    assert out.positions == [0, 1] and out.hours[0].tolist() == [24.0, 25.5, 27.0, 28.5, 30.0, 31.5, 33.0, 34.5]
    assert out.cells[0].tolist() == [4, 6, 8, 10, 12, 14, 16, 18] and out.cells[1].tolist() == [6, 8, 10, 12, 14, 16, 18, 20]
    assert out.free_dye == {0: 2000.0, 1: 2000.0}
    assert (out.matrix[0][:4] == 0).all() and (out.matrix[0][4:] > 3.0).all() and (out.matrix[1][:5] == 0).all() and (out.matrix[1][5:] > 3.0).all()
    assert out.th1["per_position_h"] == {0: 30.0, 1: 31.5} and out.th1["th1_h"] == 30.75
    hours, mean, _sd = out.mean_cells()
    assert mean.tolist() == [5, 7, 9, 11, 13, 15, 17, 19] and out.rate[0] == pytest.approx(np.full(8, 2 / 1.5))


def test_without_dye_frames_the_cells_are_still_counted():
    labels, dye = recording()
    out = GM.measure_growth_states(labels, None, {0: list(range(8))}, pixel_um=PX, hour_of=lambda t: 24.0 + 1.5 * t, reference=None)
    assert out.cells[0].tolist() == [4, 6, 8, 10, 12, 14, 16, 18] and out.matrix == {} and out.th1 is None


def test_progress_is_told_frame_by_frame():
    labels, dye = recording(); seen = []
    GM.measure_growth_states(labels, None, {0: [0, 1, 2]}, pixel_um=PX, hour_of=lambda t: 24.0 + t, reference=None, progress=lambda k, n: seen.append((k, n)))
    assert seen == [(1, 3), (2, 3), (3, 3)]
