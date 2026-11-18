"""What an independent read of the night's commits found (18 Nov 2026, 06:00), each as a test, and the Maturation list
moving with the P slider (the coordinator's point: "which position is the picture on the left?")."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace

import numpy as np
import pytest

from nd2_analyzer.analysis.biofilm_stages import matrix_stain as MX
from nd2_analyzer.analysis.biofilm_stages import microcolony_follow as MF
from nd2_analyzer.analysis.biofilm_stages import microcolony_metric_plots as XP
from nd2_analyzer.analysis.biofilm_stages import microcolony_metrics as MM
from nd2_analyzer.analysis.biofilm_stages import seeding_plots as SP
from nd2_analyzer.analysis.biofilm_stages import stage_tables as ST
from test_growth_plots_figure_style import grown
from test_microcolony_metrics import frame
from test_microcolony_rules import PX, blocks

FRAMES = [blocks(20)] * 4
INFO = SimpleNamespace(name="toy", n_frames=6, dt_s=1800.0, pixel_um=PX, start_hour=24.0, channel_roles={"0": "phase", "2": "matrix"}, channel_labels={})


def micro(position=lambda: 0):
    from nd2_analyzer.ui.biofilms.stages.microcolony_controller import MicrocolonyController
    from nd2_analyzer.ui.biofilms.stages.microcolony_page import MicrocolonyPage
    c = MicrocolonyController(MicrocolonyPage(), lambda: INFO, position)
    c.labels_at = lambda p: (lambda t: FRAMES[t])
    return c


def growth_controller():
    from nd2_analyzer.ui.biofilms.stages.growth_controller import GrowthController
    from nd2_analyzer.ui.biofilms.stages.growth_page import GrowthPage
    return GrowthController(GrowthPage(), lambda: INFO)


# ---------------------------------------------------------------- 2: positions whose segmented frames start at different frames
def test_positions_are_measured_at_the_same_hours_even_when_they_start_at_different_frames():
    from nd2_analyzer.ui.biofilms.stages.microcolony_controller import frames_for_metrics
    out = frames_for_metrics({0: list(range(0, 9)), 1: list(range(1, 10))}, step=3, last=88)
    assert out == {0: [0, 3, 6], 1: [3, 6, 9]}                                   # counted from the first frame any position has


def test_a_measure_whose_positions_share_no_hour_says_so_instead_of_failing():
    m = MM.measure_metrics(lambda p, t: frame(), {0: [0, 3, 6], 1: [1, 4, 7]}, 0.1, hour_of=lambda t: 24.0 + 0.5 * t)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    for key in XP.DRAW:
        fig = Figure(figsize=(5.6, 4.6), dpi=100); FigureCanvasAgg(fig); XP.DRAW[key](fig, m, cursor_h=24.0)
        assert [t.get_text() for ax in fig.axes for t in ax.texts] == [XP.NO_SHARED_HOUR], key


# ---------------------------------------------------------------- 3: "Follow until" after a recording is segmented
def test_follow_until_is_set_when_the_position_gets_its_segmented_frames():
    c = micro(); c.masks = lambda position: (None, [])
    c.refresh()                                                                  # loaded, not segmented yet
    assert not c.page.track_button.isEnabled()
    c.masks = lambda position: (None, list(range(6))); c.refresh()               # segmented now (the same position)
    assert c.page.track_button.isEnabled() and c.page.until_spin.maximum() == 26.5 and c.page.until_spin.value() == 26.5
    c.page.until_spin.setValue(25.0); c.refresh()                                # her own choice stays while nothing changes
    assert c.page.until_spin.value() == 25.0


# ---------------------------------------------------------------- 4: the dye is read through the tool when the recording is cropped or registered
def test_the_dye_is_read_straight_from_the_file_only_when_the_tool_changes_nothing_on_the_way():
    from nd2_analyzer.ui.biofilms.stages.growth_controller import direct_path
    plain = SimpleNamespace(image_filename=["/data/run.nd2"], crop_coordinates=None, registration_offsets=None)
    assert direct_path(plain) == "/data/run.nd2" and direct_path(None) is None
    assert direct_path(SimpleNamespace(image_filename=["/data/run.nd2"], crop_coordinates=None, registration_offsets=np.zeros((3, 2)))) is None   # registered
    assert direct_path(SimpleNamespace(image_filename="/data/run.nd2", crop_coordinates=(0, 0, 10, 10), registration_offsets=None)) is None       # cropped


# ---------------------------------------------------------------- 6: an open Follow window follows the stain of the page
def test_an_open_follow_window_gets_the_new_stain_and_loses_a_stain_that_is_gone():
    from nd2_analyzer.analysis.biofilm_stages import microcolony_plots as MP
    from nd2_analyzer.analysis.biofilm_stages.microcolony_film import MicrocolonyFilm
    from nd2_analyzer.ui.biofilms.stages.microcolony_window import MicrocolonyDialog
    from test_microcolony_stain import counted_at
    tracks = MF.follow_microcolonies(lambda t: FRAMES[t], range(4), pixel_um=PX, dt_h=0.5, start_hour=24.0)
    stain = MX.measure_microcolony_stain(tracks, counted_at, PX)
    c = micro(); c.masks = lambda position: (None, list(range(4))); c.tracked[0] = tracks; c.drawn[0] = tracks
    film = MicrocolonyFilm(tracks, lambda t: FRAMES[t], PX); c.window = MicrocolonyDialog(film, MP.entries(tracks), root=1); c.window_position = 0
    c.stains[0] = stain; c.show_stain(0)
    assert film.show_stain and film.stain_ready() and c.window.stain_box.isEnabled() and c.window.stain_box.isChecked()
    c.stains = {}; c.show_stain(0)                                               # the Maturation tab measures again: the old stain is gone
    assert not film.show_stain and film.counted_at is None and not c.window.stain_box.isEnabled() and not c.window.stain_box.isChecked()
    c.window.close()


# ---------------------------------------------------------------- 7: outlines drawn on one recording do not follow into the next
def test_outlines_are_forgotten_when_another_recording_is_loaded():
    c = micro(); c.outlines[(0, 2)] = [[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]]
    c.on_new_recording(None)
    assert c.outlines == {}


# ---------------------------------------------------------------- 8: the table says what the plot draws
def test_the_mean_table_gives_the_geometric_mean_the_plot_draws_and_how_many_positions_have_a_value():
    m = MM.measure_metrics(lambda p, t: frame(with_last_of_b=(p == 0)), {0: [0, 3], 1: [0, 3]}, 0.1, hour_of=lambda t: 24.0 + 0.5 * t)
    head, rows = ST.metric_tables(m)["microcolony_metrics_mean"]
    assert rows[0][head.index("cells_per_microcolony_geometric_mean")] == pytest.approx(np.sqrt(4.5 * 5.0))      # the plot's 4.74, next to the plain mean 4.75
    assert rows[0][head.index("cells_per_microcolony_mean")] == pytest.approx(4.75)
    assert rows[0][head.index("spacing_um_n")] == 1 and rows[0][head.index("cells_n")] == 2                       # only position 0 has two microcolonies


# ---------------------------------------------------------------- 9: a worker stops when the tool is closed
def test_a_worker_stops_at_the_next_frame_when_the_tool_is_closing():
    from nd2_analyzer.ui.biofilms.stages.microcolony_controller import MetricsWorker
    read, got, why = [], [], []
    worker = MetricsWorker(lambda p, t: read.append(t) or frame(), {0: [0, 3, 6, 9]}, 0.1, lambda t: 24.0 + 0.5 * t, random_at=())
    worker.isInterruptionRequested = lambda: True
    worker.done.connect(got.append); worker.failed.connect(why.append); worker.run()
    assert read == [0] and got == [] and why == []                                # no result, no error: it was asked to stop


# ---------------------------------------------------------------- the Maturation list and the P slider move together
def test_the_maturation_list_and_the_p_slider_move_together():
    where = [2]; moved = []
    g = growth_controller(); g.position = lambda: where[0]; g.set_position = lambda p: (moved.append(p), where.__setitem__(0, p))
    g.measured = grown(); g.show_measured(g.measured)                             # positions 1, 2, 3
    assert g.page.selected() == 2 and moved == []                                 # the list opens on the position in the view area
    g.page.candidates.setCurrentRow(3)                                            # a click on position 3
    assert moved == [3] and where == [3]
    where[0] = 1; g.on_view_index_changed((0, 1, 0))                              # the P slider moved to 1
    assert g.page.selected() == 1 and moved == [3]
    where[0] = 7; g.on_view_index_changed((0, 7, 0))                              # a position that was not measured: the list stays
    assert g.page.selected() == 1 and moved == [3]


def test_the_maturation_worker_measures_and_says_how_far_it_is():
    """The Maturation tab's worker runs its measure to the end and says its progress (it once called a helper it did
    not import, so every Measure on that tab failed), and stops when the tool is closing."""
    from nd2_analyzer.ui.biofilms.stages.growth_controller import GrowthWorker
    from test_growth_measure import PX as GROWTH_PX, recording
    labels, dye = recording(); ref = MX.DyeReference(zero=np.zeros((240, 240), np.float32), flat=np.ones((240, 240), np.float32))
    worker = GrowthWorker(labels, {0: list(range(8)), 1: list(range(8))}, GROWTH_PX, lambda t: 24.0 + 1.5 * t, ref, 2, None, dye)
    got, why, said = [], [], []
    worker.done.connect(got.append); worker.failed.connect(why.append); worker.progress.connect(lambda pct, words: said.append(words)); worker.run()
    assert why == [] and len(got) == 1 and got[0].cells[0].tolist() == [4, 6, 8, 10, 12, 14, 16, 18] and said[-1] == "100 %"
    worker = GrowthWorker(labels, {0: list(range(8)), 1: list(range(8))}, GROWTH_PX, lambda t: 24.0 + 1.5 * t, ref, 2, None, dye)
    worker.isInterruptionRequested = lambda: True; got, why = [], []
    worker.done.connect(got.append); worker.failed.connect(why.append); worker.run()
    assert got == [] and why == []
