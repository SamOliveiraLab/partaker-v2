"""The Maturation page shows the growth states: its list, cards and plots from a growth-states measure, on the made-up
recording of test_growth_measure (two positions; matrix stain from 30 h and 31.5 h)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nd2_analyzer.analysis.biofilm_stages import growth_measure as GM
from nd2_analyzer.analysis.biofilm_stages import growth_plots as GP
from nd2_analyzer.analysis.biofilm_stages import matrix_stain as MX
from nd2_analyzer.analysis.biofilm_stages import seeding_plots as SP
from test_growth_measure import PX, recording


@pytest.fixture(scope="module")
def measured():
    labels, dye = recording()
    ref = MX.DyeReference(zero=np.zeros((240, 240), np.float32), flat=np.ones((240, 240), np.float32))
    return GM.measure_growth_states(labels, dye, {0: list(range(8)), 1: list(range(8))}, pixel_um=PX, hour_of=lambda t: 24.0 + 1.5 * t, reference=ref)


def test_the_list_gives_each_position_its_th1_and_its_cells(measured):
    assert GP.entries(measured) == [("positions measured", [(0, "Position 0: Th1 30 h, 18 cells at 34.5 h"), (1, "Position 1: Th1 31.5 h, 20 cells at 34.5 h")])]


def test_the_first_cards_are_the_thresholds_and_the_last_follow_the_cursor(measured):
    cards = GP.cards(measured, cursor_h=31.5)
    assert cards[0][:2] == ("Th1 30.8 ± 1.1 h", "matrix stain appears and stays")
    assert cards[4][:2] == ("15 cells at 31.5 h", "whole frame, mean of 2 positions")
    assert cards[6][0].endswith("% at 31.5 h") and cards[6][1] == "of the frame is matrix stain"
    assert GP.cards(measured, cursor_h=24.0)[6][0] == "0.0 % at 24 h"


def test_the_states_are_cut_at_th1(measured):
    states = GP.states(measured)
    assert states[0] == ("I", 24.0, 30.75) and states[1][0] == "II" and states[1][1] == 30.75


def test_cells_only_when_no_dye_frames_were_given():
    labels, _dye = recording()
    cells_only = GM.measure_growth_states(labels, None, {0: list(range(8))}, pixel_um=PX, hour_of=lambda t: 24.0 + 1.5 * t, reference=None)
    assert GP.entries(cells_only) == [("positions measured", [(0, "Position 0: 18 cells at 34.5 h")])]
    cards = GP.cards(cells_only, cursor_h=24.0)
    assert cards[0] == ("Th1 not measured", "give the dye frames to measure the matrix stain", False) and cards[4][0] == "4 cells at 24 h"


@pytest.mark.parametrize("key", [k for k, _title in GP.FIGURES])
def test_every_plot_draws_with_no_element_on_another(measured, key):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=GP.FIGURE_SIZE_IN, dpi=100); FigureCanvasAgg(fig)
    GP.DRAW[key](fig, measured, cursor_h=31.5, selected=0)
    assert SP.collisions(fig) == []


def test_the_strip_draws_the_states_and_the_cursor(measured):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(10, 1.0), dpi=100); FigureCanvasAgg(fig); ax = fig.add_axes([0.03, 0.36, 0.94, 0.46])
    GP.draw_strip(ax, measured, (24.0, 34.5), 31.5)
    words = [t.get_text() for t in ax.texts]
    assert "I" in words and "II" in words and any(w.startswith("Th1 30.8 h") for w in words) and SP.collisions(fig) == []
