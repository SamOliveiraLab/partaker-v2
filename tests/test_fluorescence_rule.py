"""Which fluorescence channel a cell is called: the brighter of mCherry and YFP,
and no channel at all on a frame where fluorescence was not imaged."""
import numpy as np

from nd2_analyzer.analysis.metrics_service import MetricsService
from nd2_analyzer.data.frame import TLFrame


def _frame(mcherry, yfp):
    labels = np.zeros((20, 20), np.int32)
    labels[5:15, 8:12] = 1
    return TLFrame(index=(0, 0), labeled_phc=labels, mcherry=mcherry, yfp=yfp,
                   phase=np.full((20, 20), 100, np.uint16))


def _cell(level):
    img = np.zeros((20, 20), np.uint16)
    img[5:15, 8:12] = level
    return img


def test_frame_without_fluorescence_has_no_channel():
    blank = np.zeros((20, 20), np.uint16)
    row = MetricsService.calculate_cell_metrics(_frame(blank, blank.copy()))[0]
    assert row["fluorescence_channel"] == -1
    assert row["fluo_level"] == 0.0


def test_brighter_channel_is_the_cell_channel():
    row = MetricsService.calculate_cell_metrics(_frame(_cell(900), _cell(300)))[0]
    assert row["fluorescence_channel"] == 1 and row["fluo_level"] == 900
    row = MetricsService.calculate_cell_metrics(_frame(_cell(300), _cell(900)))[0]
    assert row["fluorescence_channel"] == 2 and row["fluo_level"] == 900
