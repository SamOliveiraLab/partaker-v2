"""The Microcolonies page: its list, cards and plots from followed microcolonies, on a toy recording with a hand-counted
answer (two microcolonies of four cells that merge at 25 h)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nd2_analyzer.analysis.biofilm_stages import microcolony_follow as MF
from nd2_analyzer.analysis.biofilm_stages import microcolony_plots as MP
from nd2_analyzer.analysis.biofilm_stages import seeding_plots as SP
from test_microcolony_rules import PX, blocks


@pytest.fixture(scope="module")
def toy():
    frames = [blocks(20), blocks(20), blocks(3), blocks(3), blocks(3), blocks(3)]
    return MF.follow_microcolonies(lambda t: frames[t], range(6), pixel_um=PX, dt_h=0.5, start_hour=24.0), frames


def test_the_list_says_who_is_still_there_and_who_merged(toy):
    tracks, _ = toy
    assert MP.entries(tracks) == [("still there at 26.5 h", [(1, "Microcolony 1: from 24.0 h, 1 merged into it")]),
                                  ("merged into another", [(2, "Microcolony 2: 24.0 to 25.0 h, merges into 1")])]


def test_the_cards_count_what_happened(toy):
    tracks, _ = toy
    values = [value for value, _meaning, _ok in MP.cards(tracks)]
    assert values == ["2 microcolonies at 24 h", "1 merge", "no merge yet", "2 followed", "100 % of cells at 26.5 h",
                      "8 cells at 26.5 h", "100.0 % of cells at 24 h", "1 left at 26.5 h"]
    assert MP.cards(tracks)[2][2] is False            # both were there at the start, so no "forming to merging" time


def test_numbers_over_time(toy):
    tracks, _ = toy
    hours, count = MP.count_series(tracks)
    assert hours.tolist() == [24.0, 24.5, 25.0, 25.5, 26.0, 26.5] and count.tolist() == [2, 2, 1, 1, 1, 1]
    assert MP.subtree(tracks, 1) == [1, 2] and MP.forest(tracks) == [2, 1]


@pytest.mark.parametrize("key", [k for k, _title in MP.FIGURES])
def test_every_plot_draws_with_no_element_on_another(toy, key):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    tracks, _ = toy
    fig = Figure(figsize=MP.FIGURE_SIZE_IN, dpi=100); FigureCanvasAgg(fig)
    MP.DRAW[key](fig, tracks, cursor_h=25.0, selected=1)
    assert SP.collisions(fig) == []


# ---------------------------------------------------------------- the film that follows one microcolony
@pytest.fixture()
def film(toy):
    from nd2_analyzer.analysis.biofilm_stages.microcolony_film import MicrocolonyFilm
    tracks, frames = toy
    return MicrocolonyFilm(tracks, lambda t: frames[t], PX)


def test_the_film_follows_a_microcolony_and_those_that_merge_into_it(film):
    film.set_root(1)
    assert film.order == [2, 1] and film.frames == [0, 1, 2, 3, 4, 5]
    assert film.owner(2, 1) == 2 and film.owner(2, 2) == 1          # microcolony 2 is itself until it merges at frame 2
    assert film.microcolonies_now(1) == 2 and film.microcolonies_now(3) == 1
    assert film.cells_now(1) == 4 and film.cells_now(3) == 8


def test_a_click_on_a_cell_gives_its_microcolony(film):
    film.set_root(1)
    y0, _y1, x0, _x1 = film.box
    oy, ox = film.tracks.offsets[0]
    assert film.identity_at(0, 30 + ox - x0, 45 + oy - y0) == 1        # a cell of the left block (frame coordinates 30, 45)
    assert film.identity_at(0, 100 + ox - x0, 45 + oy - y0) == 2       # a cell of the right block
    assert film.identity_at(0, 5 + ox - x0, 5 + oy - y0) is None       # background


def test_the_film_frame_draws_with_no_element_on_another(film):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from nd2_analyzer.analysis.biofilm_stages.microcolony_film import FIGURE_IN
    film.set_root(1)
    fig = Figure(figsize=FIGURE_IN, dpi=100); FigureCanvasAgg(fig)
    ax_img, ax_history = film.draw(fig, 3)
    assert SP.collisions(fig) == [] and ax_img is not None and ax_history is not None


# ---------------------------------------------------------------- the page and the window (Qt, offscreen)
@pytest.fixture()
def page(toy, film):
    from nd2_analyzer.ui.biofilms.stages.microcolony_page import MicrocolonyPage
    tracks, _ = toy
    p = MicrocolonyPage(); p.set_recording("toy recording, 6 frames, 30 min apart", (24.0, 26.5)); p.set_tracks(tracks, film)
    return p


def test_the_page_lists_the_microcolonies_and_fills_the_cards(page):
    lines = [page.candidates.item(r).text() for r in range(page.candidates.count())]
    assert lines == ["still there at 26.5 h", "Microcolony 1: from 24.0 h, 1 merged into it",
                     "merged into another", "Microcolony 2: 24.0 to 25.0 h, merges into 1"]
    assert page.selected() == 1 and page.follow_button.isEnabled()
    assert [c.value.text() for c in page.cards][:2] == ["2 microcolonies at 24 h", "1 merge"]
    assert [page.plot_combos[k].currentText() for k in range(2)] == ["Number of microcolonies", "Merging history"]


def test_follow_asks_for_the_window_on_the_selected_microcolony(page):
    asked = []
    page.window_requested.connect(asked.append)
    page.candidates.setCurrentRow(3); page.follow_button.click()
    assert asked == [2]


def test_the_page_without_tracks_says_what_to_do():
    from nd2_analyzer.ui.biofilms.stages.microcolony_page import MicrocolonyPage
    p = MicrocolonyPage()
    assert not p.follow_button.isEnabled() and not p.track_button.isEnabled() and p.candidates.count() == 0
    p.set_step_state("segment", "done", "6, position 0"); p.set_track_enabled(True)
    assert p.track_button.isEnabled() and "Segmented Frames:" in p.step_labels["segment"].text()


def test_the_window_plays_the_film_and_follows_a_clicked_microcolony(toy, film):
    from nd2_analyzer.ui.biofilms.stages.microcolony_window import MicrocolonyDialog
    tracks, _ = toy
    dlg = MicrocolonyDialog(film, MP.entries(tracks), root=1)
    assert dlg.slider.maximum() == 5 and dlg.combo.currentData() == 1
    dlg.slider.setValue(3)
    assert dlg.time_label.text() == "frame 3   25.5 h"
    dlg.slider.setValue(0)
    y0, _y1, x0, _x1 = film.box; oy, ox = tracks.offsets[0]
    dlg.follow_at(100 + ox - x0, 45 + oy - y0)                           # a cell of microcolony 2, before it merges
    assert dlg.combo.currentData() == 2 and film.root == 2 and dlg.hint.text() == "Following microcolony 2."
    dlg.close()


def test_track_follows_the_unbroken_run_of_segmented_frames_up_to_the_chosen_hour():
    from nd2_analyzer.ui.biofilms.stages.microcolony_controller import frames_to_follow
    assert frames_to_follow([0, 1, 2, 4, 5], last=10) == [0, 1, 2]          # frame 3 is missing: the run stops before it
    assert frames_to_follow([3, 4, 5, 6], last=5) == [3, 4, 5]
    assert frames_to_follow([], last=5) == [] and frames_to_follow([7], last=9) == [7]
    out_of_focus = lambda t: 6 <= t <= 7
    assert frames_to_follow(list(range(10)), last=9, skip=out_of_focus) == [0, 1, 2, 3, 4, 5]     # the run stops where focus is lost


def test_the_biofilm_stages_tab_holds_the_live_microcolonies_page():
    from nd2_analyzer.ui.biofilms.stages.tab import BiofilmStagesTab
    tab = BiofilmStagesTab()
    assert [tab.tabs.tabText(i) for i in range(4)] == ["Seeding", "Microcolonies", "Maturation", "Dispersion"]
    assert tab.tabs.widget(1) is tab.microcolony_page
    tab.tabs.setCurrentIndex(1)
    assert tab.study_row.isHidden()                    # a live page reads the recording, not a study file
    tab.tabs.setCurrentIndex(2)
    assert tab.study_row.isHidden()                    # Maturation is live too
    tab.tabs.setCurrentIndex(3)
    assert not tab.study_row.isHidden()                # Dispersion still reads a study file


def test_the_film_of_a_microcolony_that_merges_stops_at_its_last_frame_as_itself(film):
    film.set_root(2)                                   # microcolony 2 merges into 1 at frame 2
    assert film.frames == [0, 1]


def test_the_recording_line_says_the_frame_step_in_plain_words():
    from types import SimpleNamespace
    from nd2_analyzer.ui.biofilms.stages.microcolony_controller import recording_words
    info = SimpleNamespace(name="E. coli run 2", n_frames=89, dt_s=1800.0, pixel_um=0.13 / 3, start_hour=24.0, flow={"present": True, "direction": "left to right"})
    assert recording_words(info, position=2) == "E. coli run 2, 89 frames, 30 min apart, 0.0433 µm per px, 24 to 68 h after inoculation, position 2"
    info.dt_s = 3.0; info.n_frames = 35400; info.start_hour = 0.0
    assert recording_words(info, position=0) == "E. coli run 2, 35,400 frames, 3 s apart, 0.0433 µm per px, 0 to 29.5 h after inoculation, position 0"


def test_manual_selection_is_asked_for_from_the_page():
    from nd2_analyzer.ui.biofilms.stages.microcolony_page import MicrocolonyPage
    p = MicrocolonyPage(); asked = []
    p.compare_requested.connect(lambda: asked.append(True))
    assert not p.compare_button.isEnabled()                    # nothing segmented yet: no frame to draw on
    p.set_step_state("segment", "done", "6, position 0"); p.set_track_enabled(True)
    assert p.compare_button.isEnabled() and p.compare_button.text() == "Manual selection"
    p.compare_button.click()
    assert asked == [True]


# ---------------------------------------------------------------- the matrix stain in the Follow window
def left_block_stain(t):
    """Matrix over the left block (microcolony 1) from frame 3 on, in frame coordinates."""
    m = np.zeros((120, 200), bool)
    if t >= 3:
        m[35:68, 15:68] = True
    return m


def test_the_film_shows_the_matrix_stain_when_asked(film):
    film.set_root(1); y0, _y1, x0, _x1 = film.box; oy, ox = film.tracks.offsets[4]; spot = (51 + oy - y0, 41 + ox - x0)      # background between the four cells of the left block, under the matrix
    plain = film.picture(4)[0][spot]
    film.set_stain(left_block_stain); film.show_stain = True
    stained = film.picture(4)[0][spot]
    assert stained[1] > plain[1] + 0.3 and stained[1] > stained[0] + 0.3                 # green where the matrix is
    hours, cover = film.stain_series()
    assert hours.tolist() == [24.0, 24.5, 25.0, 25.5, 26.0, 26.5] and (cover[:3] == 0).all() and (cover[3:] > 20).all()


def test_the_window_has_a_matrix_stain_box_that_needs_the_stain(toy, film):
    from nd2_analyzer.ui.biofilms.stages.microcolony_window import MicrocolonyDialog
    tracks, _ = toy
    dlg = MicrocolonyDialog(film, MP.entries(tracks), root=1)
    assert not dlg.stain_box.isEnabled() and "Maturation" in dlg.stain_box.toolTip()   # no dye frames measured yet
    dlg.close()
    film.set_stain(left_block_stain)
    dlg = MicrocolonyDialog(film, MP.entries(tracks), root=1)
    assert dlg.stain_box.isEnabled() and not film.show_stain
    dlg.stain_box.setChecked(True)
    assert film.show_stain
    dlg.close()


def test_ticking_matrix_stain_measures_the_film_frame_by_frame_and_says_how_far_it_is(toy, film):
    """The stain of every frame of the film has to be measured once (seconds on a long film): the window says which
    frame it is at and keeps its controls still meanwhile, instead of standing frozen."""
    from nd2_analyzer.ui.biofilms.stages.microcolony_window import HINT, MicrocolonyDialog
    tracks, _ = toy
    film.set_root(1); film.set_stain(left_block_stain)
    assert not film.stain_ready()
    dlg = MicrocolonyDialog(film, MP.entries(tracks), root=1)
    said = []; plain = dlg.hint.setText; enabled = []
    dlg.hint.setText = lambda words: (said.append(words), enabled.append(dlg.slider.isEnabled()), plain(words))[-1]
    dlg.stain_box.click()
    assert said[:2] == ["Measuring the matrix stain in this microcolony: frame 1 of 6", "Measuring the matrix stain in this microcolony: frame 2 of 6"]
    assert said[5].endswith("frame 6 of 6") and said[-1] == HINT and enabled[:6] == [False] * 6      # the slider waits while it measures
    assert film.show_stain and film.stain_ready() and dlg.slider.isEnabled() and dlg.stain_box.isEnabled()
    said.clear(); dlg.stain_box.click(); dlg.stain_box.click()                                       # off and on again: nothing to measure
    assert not [w for w in said if w.startswith("Measuring")]
    dlg.follow(2)                                                                                     # another microcolony, the box still ticked
    assert [w for w in said if w.startswith("Measuring")][-1].endswith("frame 2 of 2") and film.stain_ready()
    dlg.close()


def test_the_stain_series_can_be_measured_step_by_step(film):
    film.set_root(1); film.set_stain(left_block_stain)
    steps = list(film.measure_stain())
    assert steps == [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)] and film.stain_ready()
    hours, cover = film.stain_series()
    assert hours.tolist() == [24.0, 24.5, 25.0, 25.5, 26.0, 26.5] and (cover[:3] == 0).all() and (cover[3:] > 20).all()
    assert list(film.measure_stain()) == []                                                           # already measured


def test_closing_the_window_while_the_stain_is_measured_stops_the_measure_quietly(toy, film):
    from nd2_analyzer.ui.biofilms.stages.microcolony_window import MicrocolonyDialog
    tracks, _ = toy
    film.set_root(1); seen = []

    def stain_then_close(t):
        seen.append(t)
        if len(seen) == 2:
            dlg.close()                                    # the user closes the window during the second frame
        return left_block_stain(t)

    film.set_stain(stain_then_close)
    dlg = MicrocolonyDialog(film, MP.entries(tracks), root=1)
    dlg.stain_box.click()                                  # must not raise
    assert len(seen) == 2 and not film.stain_ready()


# ---------------------------------------------------------------- clicks, and the stain measure, made safe (18 Nov, about 03:30)
def test_a_click_between_the_cells_of_a_microcolony_or_just_beside_it_counts(film):
    film.set_root(1); y0, _y1, x0, _x1 = film.box; oy, ox = film.tracks.offsets[0]
    at = lambda x, y: film.identity_at(0, x + ox - x0, y + oy - y0)          # frame coordinates
    assert at(41, 51) == 1                 # the background between the four cells of the left block
    assert at(14, 45) == 1                 # 4 px (0.4 um) left of its area: within one cell width
    assert at(11, 45) is None              # 7 px away: too far
    assert at(68, 45) == 1 and at(78, 45) == 2                                # in the gap between the two, the nearer one


def test_a_click_in_the_picture_while_the_stain_is_measured_is_ignored(toy, film):
    from types import SimpleNamespace
    from nd2_analyzer.ui.biofilms.stages.microcolony_window import MicrocolonyDialog
    tracks, _ = toy
    film.set_root(1); seen = []; y0, _y1, x0, _x1 = film.box; oy, ox = film.tracks.offsets[0]

    def stain_with_a_click(t):
        seen.append(t)
        if len(seen) == 2:                                                 # the user clicks on microcolony 2 during the second frame
            dlg._on_click(SimpleNamespace(inaxes=dlg.ax_img, xdata=100 + ox - x0, ydata=45 + oy - y0))
        return left_block_stain(t)

    film.set_stain(stain_with_a_click)
    dlg = MicrocolonyDialog(film, MP.entries(tracks), root=1)
    dlg.stain_box.click()
    assert film.root == 1 and film.stain_ready() and len(seen) == 6
    dlg.close()


def test_following_another_microcolony_while_the_stain_is_measured_measures_that_one(toy, film):
    """Whatever changes the followed microcolony during the measure, the curve shown is the curve of the microcolony shown."""
    from nd2_analyzer.analysis.biofilm_stages.microcolony_film import MicrocolonyFilm
    from nd2_analyzer.ui.biofilms.stages.microcolony_window import MicrocolonyDialog
    tracks, frames = toy
    film.set_root(1); seen = []

    def stain_then_follow(t):
        seen.append(t)
        if len(seen) == 2:
            dlg.follow(2)                                                  # microcolony 2 is itself for frames 0 and 1
        return left_block_stain(t)

    film.set_stain(stain_then_follow)
    dlg = MicrocolonyDialog(film, MP.entries(tracks), root=1)
    dlg.stain_box.click()
    clean = MicrocolonyFilm(tracks, lambda t: frames[t], PX); clean.set_root(2); clean.set_stain(left_block_stain)
    assert film.root == 2 and film.stain_ready()
    assert [a.tolist() for a in film.stain_series()] == [a.tolist() for a in clean.stain_series()]
    dlg.close()


# ---------------------------------------------------------------- what the sketch of 17 Nov showed and the page lacked
def test_the_strip_shows_microcolony_formation_and_maturation_once_th1_is_known(page):
    words = lambda: [t.get_text() for t in page.strip.ax.texts]
    assert "microcolony formation" not in words()                       # no Th1 yet: a plain band
    page.set_th1(25.5)
    assert words()[:3] == ["microcolony formation", "maturation", "Th1 25.5 h"] and SP.collisions(page.strip.figure) == []
    page.strip.set_hour(25.5); assert SP.collisions(page.strip.figure) == []        # the cursor on Th1 itself
    page.set_th1(None)
    assert "maturation" not in words()


def test_a_th1_outside_the_recording_or_a_band_too_narrow_for_its_name_is_left_out(page):
    page.set_th1(99.0)
    assert "maturation" not in [t.get_text() for t in page.strip.ax.texts]
    page.set_th1(24.1)                                                   # 0.1 h of 2.5: no room for "microcolony formation"
    words = [t.get_text() for t in page.strip.ax.texts]
    assert "microcolony formation" not in words and "maturation" in words and "Th1 24.1 h" in words
    assert SP.collisions(page.strip.figure) == []


def test_the_film_strip_says_which_microcolony_it_shows(page):
    assert page.film_caption.text() == "microcolony 1 and the 1 that merges into it"
    page.candidates.setCurrentRow(3)                                     # microcolony 2: nothing merges into it
    assert page.film_caption.text() == "microcolony 2"
    page.set_tracks(None)
    assert page.film_caption.text() == ""


def test_the_microcolonies_strip_takes_th1_from_the_maturation_measure():
    """One Th1 in the tool: the mean of the measured positions, the one the Maturation strip shows."""
    from test_growth_measure import recording
    from nd2_analyzer.analysis.biofilm_stages import growth_measure as GM, matrix_stain as MX
    from nd2_analyzer.ui.biofilms.stages.tab import BiofilmStagesTab
    labels, dye = recording()
    ref = MX.DyeReference(zero=np.zeros((240, 240), np.float32), flat=np.ones((240, 240), np.float32))
    growth = GM.measure_growth_states(labels, dye, {0: list(range(8)), 1: list(range(8))}, pixel_um=0.1, hour_of=lambda t: 24.0 + 1.5 * t, reference=ref, resamples=20)
    tab = BiofilmStagesTab(); tab.microcolony_page.set_recording("toy", (24.0, 34.5))
    assert tab.microcolony_page.strip.th1_h is None
    tab.growth_controller.measured = growth; tab.growth_controller.show_measured(growth)
    assert tab.microcolony_page.strip.th1_h == 30.75 and "Th1 30.8 h" in [t.get_text() for t in tab.microcolony_page.strip.ax.texts]
    tab.growth_controller.on_new_recording(None)
    assert tab.microcolony_page.strip.th1_h is None
