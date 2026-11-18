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


def test_the_biofilm_stages_tab_holds_the_live_microcolonies_page():
    from nd2_analyzer.ui.biofilms.stages.tab import BiofilmStagesTab
    tab = BiofilmStagesTab()
    assert [tab.tabs.tabText(i) for i in range(4)] == ["Seeding", "Microcolonies", "Maturation", "Dispersion"]
    assert tab.tabs.widget(1) is tab.microcolony_page
    tab.tabs.setCurrentIndex(1)
    assert tab.study_row.isHidden()                    # a live page reads the recording, not a study file
    tab.tabs.setCurrentIndex(2)
    assert not tab.study_row.isHidden()


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
