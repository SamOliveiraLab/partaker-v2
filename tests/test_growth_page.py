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


# ---------------------------------------------------------------- the film of one position, the page and the window
@pytest.fixture()
def film(measured):
    from nd2_analyzer.analysis.biofilm_stages.growth_film import GrowthFilm
    labels, dye = recording()
    ref = MX.DyeReference(zero=np.zeros((240, 240), np.float32), flat=np.ones((240, 240), np.float32))
    phase = lambda p, t: 1000.0 - 300.0 * (labels(p, t) > 0)             # dark cells on a bright field
    return GrowthFilm(measured, labels, phase, dye, ref, PX)


def test_the_film_shows_one_position_frame_by_frame(film):
    film.set_position(1)
    assert film.position == 1 and film.frames == list(range(8))
    rgb = film.picture(6)
    assert rgb.ndim == 3 and rgb.shape[2] == 3 and 0.0 <= rgb.min() and rgb.max() <= 1.0
    stained = rgb[int(175 * rgb.shape[0] / 240), int(175 * rgb.shape[1] / 240)]; plain = rgb[int(100 * rgb.shape[0] / 240), int(100 * rgb.shape[1] / 240)]
    assert stained[1] > stained[0] + 0.2 and abs(plain[1] - plain[0]) < 0.02          # matrix stain is green, the empty field grey
    before = film.picture(2)[int(175 * rgb.shape[0] / 240), int(175 * rgb.shape[1] / 240)]
    assert abs(before[1] - before[0]) < 0.02                                         # no stain there before 31.5 h


def test_the_film_frame_draws_with_no_element_on_another(film):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from nd2_analyzer.analysis.biofilm_stages.growth_film import FIGURE_IN
    film.set_position(0)
    fig = Figure(figsize=FIGURE_IN, dpi=100); FigureCanvasAgg(fig)
    assert film.draw(fig, 5) is not None and SP.collisions(fig) == []


@pytest.fixture()
def page(measured):
    from nd2_analyzer.ui.biofilms.stages.growth_page import GrowthPage
    p = GrowthPage(); p.set_recording("toy recording, 8 frames, 1.5 h apart", (24.0, 34.5)); p.set_measured(measured)
    return p


def test_the_page_lists_the_positions_and_its_cards_follow_the_cursor(page):
    lines = [page.candidates.item(r).text() for r in range(page.candidates.count())]
    assert lines == ["positions measured", "Position 0: Th1 30 h, 18 cells at 34.5 h", "Position 1: Th1 31.5 h, 20 cells at 34.5 h"]
    assert page.selected() == 0 and page.follow_button.isEnabled() and page.cards[0].value.text() == "Th1 30.8 ± 1.1 h"
    page.set_hour(24.0)
    assert page.cards[4].value.text() == "5 cells at 24 h" and page.cards[6].value.text() == "0.0 % at 24 h"
    page.set_hour(34.5)
    assert page.cards[4].value.text() == "19 cells at 34.5 h"
    assert [c.currentText() for c in page.plot_combos] == ["Number of cells and Th2", "Matrix stain and Th1"]


def test_the_page_asks_for_the_dye_frames_the_measure_and_the_window(page):
    asked = []
    page.dye_requested.connect(lambda: asked.append("dye")); page.measure_requested.connect(lambda: asked.append("measure")); page.window_requested.connect(asked.append)
    page.set_measure_enabled(True); page.dye_button.click(); page.measure_button.click(); page.candidates.setCurrentRow(2); page.follow_button.click()
    assert asked == ["dye", "measure", 1]


def test_a_page_with_nothing_measured_says_what_to_do():
    from nd2_analyzer.ui.biofilms.stages.growth_page import GrowthPage
    p = GrowthPage()
    assert not p.measure_button.isEnabled() and not p.follow_button.isEnabled() and "Dye Frames:" in p.step_labels["dye"].text()
    assert p.name == "Maturation"


def test_the_window_plays_the_film_of_a_position(film):
    from nd2_analyzer.ui.biofilms.stages.growth_window import GrowthDialog
    dlg = GrowthDialog(film, [(0, "Position 0"), (1, "Position 1")], position=1)
    assert dlg.combo.currentData() == 1 and dlg.slider.maximum() == 7
    dlg.slider.setValue(4)
    assert dlg.time_label.text() == "frame 4   30 h"
    dlg.combo.setCurrentIndex(0)
    assert film.position == 0 and dlg.slider.value() == 4                 # the other position, at the same hour
    dlg.close()


def test_measure_takes_every_nth_segmented_frame_and_leaves_out_the_unfocused_ones():
    from nd2_analyzer.ui.biofilms.stages.growth_controller import frames_to_measure
    assert frames_to_measure(list(range(10)), step=3) == [0, 3, 6, 9]
    assert frames_to_measure(list(range(10)), step=1, skip=lambda t: 4 <= t <= 6) == [0, 1, 2, 3, 7, 8, 9]
    assert frames_to_measure([2, 3, 4, 5, 8], step=2) == [2, 4, 8] and frames_to_measure([], step=3) == []
    assert frames_to_measure(list(range(11)), step=3) == [0, 3, 6, 9, 10]            # the last frame of the recording is always measured
    assert frames_to_measure(list(range(11)), step=3, skip=lambda t: t == 10) == [0, 3, 6, 9]


def test_the_dye_channel_is_the_one_the_experiment_calls_matrix_dye():
    from nd2_analyzer.ui.biofilms.stages.growth_controller import channel_with_role
    roles = {"0": "phase", "1": "cell_marker", "2": "matrix"}
    assert channel_with_role(roles, "matrix") == 2 and channel_with_role(roles, "phase") == 0 and channel_with_role({"0": "phase"}, "matrix") is None


def test_the_film_paints_no_matrix_stain_on_frames_where_it_was_not_measured():
    """After a loss of focus the matrix stain is not measured (the dye image is on another scale): the film shows those
    frames in grey, as the plot shows no value there."""
    from nd2_analyzer.analysis.biofilm_stages.growth_film import GrowthFilm
    labels, dye = recording()
    ref = MX.DyeReference(zero=np.zeros((240, 240), np.float32), flat=np.ones((240, 240), np.float32))
    stopped = GM.measure_growth_states(labels, dye, {0: list(range(8))}, pixel_um=PX, hour_of=lambda t: 24.0 + 1.5 * t, reference=ref,
                                       matrix_frames={0: list(range(6))}, resamples=20)            # matrix measured on frames 0 to 5 only
    film = GrowthFilm(stopped, labels, lambda p, t: 1000.0 - 300.0 * (labels(p, t) > 0), dye, ref, PX)
    spot = lambda rgb: rgb[int(175 * rgb.shape[0] / 240), int(175 * rgb.shape[1] / 240)]
    measured_frame, later_frame = spot(film.picture(5)), spot(film.picture(7))
    assert measured_frame[1] > measured_frame[0] + 0.2                      # green where the stain was measured
    assert abs(later_frame[1] - later_frame[0]) < 0.02                      # grey on a frame where it was not
