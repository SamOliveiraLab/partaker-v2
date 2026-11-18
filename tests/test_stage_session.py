"""What the Microcolonies and Maturation pages worked out is saved with the session and comes back on File > Load: the
followed microcolonies, the all-positions metrics, the outlines drawn by hand, the growth states and the dye file choice.
The files hold numbers and text only (no pickled Python), so a session folder from someone else is safe to open."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace

import numpy as np
import pytest

from nd2_analyzer.analysis.biofilm_stages import growth_measure as GM
from nd2_analyzer.analysis.biofilm_stages import matrix_stain as MX
from nd2_analyzer.analysis.biofilm_stages import microcolony_follow as MF
from nd2_analyzer.analysis.biofilm_stages import microcolony_metrics as MM
from nd2_analyzer.analysis.biofilm_stages import microcolony_plots as MP
from test_growth_measure import recording
from test_microcolony_metrics import frame
from test_microcolony_rules import PX, blocks

FRAMES = [blocks(20), blocks(20), blocks(3), blocks(3), blocks(3), blocks(3)]          # two microcolonies that merge at 25 h


@pytest.fixture(scope="module")
def tracks():
    return MF.follow_microcolonies(lambda t: FRAMES[t], range(6), pixel_um=PX, dt_h=0.5, start_hour=24.0)


@pytest.fixture(scope="module")
def metrics():
    return MM.measure_metrics(lambda p, t: frame(with_last_of_b=(p == 0)), {0: [0, 3, 6], 1: [0, 3, 6]}, 0.1, hour_of=lambda t: 24.0 + 0.5 * t,
                              random_hours=(24.0,), placements=2)


@pytest.fixture(scope="module")
def growth():
    labels, dye = recording()
    ref = MX.DyeReference(zero=np.zeros((240, 240), np.float32), flat=np.ones((240, 240), np.float32))
    return GM.measure_growth_states(labels, dye, {0: list(range(8)), 1: list(range(8))}, pixel_um=0.1, hour_of=lambda t: 24.0 + 1.5 * t, reference=ref, resamples=50)


def same(a, b) -> bool:
    """Equal through dictionaries, lists and arrays; a missing number equals a missing number."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and not isinstance(a, str):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=np.asarray(a).dtype.kind == "f")
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def test_followed_microcolonies_come_back_from_a_file_as_they_were(tracks, tmp_path):
    MF.save_tracks(tracks, tmp_path / "tracks.npz"); back = MF.load_tracks(tmp_path / "tracks.npz")
    for name in ("frames", "hours", "offsets", "canvas_shape", "identities", "events", "rows", "cell_identity", "min_cells"):
        assert same(getattr(back, name), getattr(tracks, name)), name
    assert isinstance(back.canvas_shape, tuple) and isinstance(back.offsets[2], tuple) and all(isinstance(k, int) for k in back.cell_identity[0])
    assert all(np.array_equal(back.identity_map(t), tracks.identity_map(t)) for t in tracks.frames)
    assert all(same(list(back.points[t]), list(tracks.points[t])) for t in tracks.frames)
    assert MP.entries(back) == MP.entries(tracks) and MP.cards(back) == MP.cards(tracks)
    assert back.lasting() == tracks.lasting() and back.merges() == tracks.merges()


def test_the_metrics_come_back_from_a_file_as_they_were(metrics, tmp_path):
    MM.save_metrics(metrics, tmp_path / "metrics.npz"); back = MM.load_metrics(tmp_path / "metrics.npz")
    assert back.positions == metrics.positions and back.frames == metrics.frames and back.settings == metrics.settings
    assert same(back.hours, metrics.hours) and same(back.values, metrics.values)
    assert same(back.ring_sum, metrics.ring_sum) and same(back.ring_n, metrics.ring_n) and same(back.ring_edges, metrics.ring_edges)
    assert same(list(back.mean_of("cells_in_pct")), list(metrics.mean_of("cells_in_pct")))


def test_the_growth_states_come_back_from_a_file_as_they_were(growth, tmp_path):
    GM.save_growth_states(growth, tmp_path / "growth.npz"); back = GM.load_growth_states(tmp_path / "growth.npz")
    assert back.positions == growth.positions and back.frames == growth.frames and back.settings == growth.settings
    for name in ("hours", "cells", "rate", "matrix_hours", "matrix", "single", "inside", "outside", "colony_area", "free_dye", "th1", "th2"):
        assert same(getattr(back, name), getattr(growth, name)), name
    assert back.th1["per_position_h"] == {0: 30.0, 1: 31.5} and isinstance(next(iter(back.free_dye)), int)


def test_a_file_that_is_not_ours_is_refused(tmp_path):
    np.savez(tmp_path / "other.npz", something=np.arange(3))
    for load in (MF.load_tracks, MM.load_metrics, GM.load_growth_states):
        with pytest.raises(ValueError):
            load(tmp_path / "other.npz")


# ---------------------------------------------------------------- the tab (first: it builds the controllers in the order the
# tool does, which is what tells the message bus the shape of "experiment_loaded" and "image_data_loaded")

def test_the_stage_tab_saves_and_loads_its_pages_together(tracks, growth, tmp_path):
    from nd2_analyzer.ui.biofilms.stages.tab import BiofilmStagesTab
    tab = BiofilmStagesTab()
    assert tab.load_results(tmp_path) == {"microcolonies": False, "maturation": False}
    tab.microcolony_controller.tracked[0] = tracks; tab.microcolony_controller.drawn[0] = tracks; tab.growth_controller.measured = growth
    tab.save_results(tmp_path)
    assert sorted(f.name for f in (tmp_path / "biofilm_stages").iterdir()) == [
        "growth_states.json", "growth_states.npz", "microcolonies.json", "microcolonies_p0.npz", "microcolonies_p0_drawn.npz"]
    other = BiofilmStagesTab()
    assert other.load_results(tmp_path) == {"microcolonies": True, "maturation": True}
    assert MP.cards(other.microcolony_controller.tracked[0]) == MP.cards(tracks) and same(other.growth_controller.measured.cells, growth.cells)


# ---------------------------------------------------------------- the pages' controllers
INFO = SimpleNamespace(name="toy", n_frames=6, dt_s=1800.0, pixel_um=PX, start_hour=24.0, channel_roles={}, channel_labels={})


def microcolony_controller():
    from nd2_analyzer.ui.biofilms.stages.microcolony_controller import MicrocolonyController
    from nd2_analyzer.ui.biofilms.stages.microcolony_page import MicrocolonyPage
    c = MicrocolonyController(MicrocolonyPage(), lambda: INFO, lambda: 0)
    c.masks = lambda position: (None, list(range(6)))                     # six segmented frames of position 0
    c.labels_at = lambda position: (lambda t: FRAMES[t])
    return c


def test_the_microcolonies_page_is_saved_with_the_session_and_comes_back(tracks, metrics, tmp_path):
    c = microcolony_controller()
    assert c.save_state(tmp_path) == []                                   # nothing tracked or measured: nothing written
    c.tracked[0] = tracks; c.drawn[0] = tracks; c.metrics = metrics; c.outlines[(0, 2)] = [[(1.0, 2.0), (30.0, 2.0), (30.0, 40.5)]]
    c.page.set_every(0.5, 1.5)
    written = c.save_state(tmp_path)
    assert sorted(os.path.basename(f) for f in written) == ["microcolonies.json", "microcolonies_p0.npz", "microcolonies_p0_drawn.npz", "microcolony_metrics.npz"]
    fresh = microcolony_controller()
    assert fresh.load_state(tmp_path) is True
    assert MP.cards(fresh.tracked[0]) == MP.cards(tracks) and MP.cards(fresh.drawn[0]) == MP.cards(tracks)
    assert fresh.outlines == {(0, 2): [[(1.0, 2.0), (30.0, 2.0), (30.0, 40.5)]]}
    assert same(fresh.metrics.values, metrics.values)
    page = fresh.page                                                      # the page shows it all without a click on Track or Measure
    assert page.tracks is fresh.tracked[0] and page.metrics is fresh.metrics and page.film is not None
    assert page.step_states["track"] == ("done", "2 followed, 1 merge") and page.step_states["measure"][0] == "done"
    assert page.every_spin.value() == 1.5


def test_a_session_without_stage_results_loads_as_before(tmp_path):
    fresh = microcolony_controller()
    assert fresh.load_state(tmp_path) is False and fresh.tracked == {} and fresh.metrics is None


def test_a_damaged_file_does_not_stop_the_load(tracks, tmp_path):
    c = microcolony_controller(); c.tracked[0] = tracks; c.drawn[0] = tracks; c.save_state(tmp_path)
    (tmp_path / "biofilm_stages" / "microcolonies_p0.npz").write_bytes(b"not a file of ours")
    fresh = microcolony_controller()
    assert fresh.load_state(tmp_path) is False and fresh.tracked == {} and fresh.page.tracks is None
    assert "could not be read" in fresh.page.status.text()


def test_the_maturation_page_is_saved_with_the_session_and_comes_back(growth, tmp_path):
    from nd2_analyzer.ui.biofilms.stages.growth_controller import GrowthController
    from nd2_analyzer.ui.biofilms.stages.growth_page import GrowthPage
    info = SimpleNamespace(name="toy", n_frames=8, dt_s=5400.0, pixel_um=0.1, start_hour=24.0, channel_roles={"0": "phase", "1": "matrix"}, channel_labels={})
    c = GrowthController(GrowthPage(), lambda: info)
    assert c.save_state(tmp_path) == []
    c.measured = growth; c.dye_paths = ("/somewhere/before.nd2", "/somewhere/after.nd2"); c.page.set_every(1.5, 3.0)
    written = c.save_state(tmp_path)
    assert sorted(os.path.basename(f) for f in written) == ["growth_states.json", "growth_states.npz"]
    fresh = GrowthController(GrowthPage(), lambda: info)
    assert fresh.load_state(tmp_path) is True
    assert fresh.measured.th1["per_position_h"] == {0: 30.0, 1: 31.5} and same(fresh.measured.cells, growth.cells)
    assert fresh.page.measured is fresh.measured and fresh.page.step_states["measure"][0] == "done" and fresh.page.every_spin.value() == 3.0
    assert fresh.dye_paths == ("/somewhere/before.nd2", "/somewhere/after.nd2")
    # the two dye files are not where they were: the numbers are shown, and the page says the dye frames are missing
    assert fresh.reference is None and fresh.page.step_states["dye"][0] == "failed" and "before.nd2" in fresh.page.step_states["dye"][1]


def test_the_page_says_which_position_is_tracked_when_another_one_is_shown(tracks, tmp_path):
    c = microcolony_controller(); c.tracked[0] = tracks; c.drawn[0] = tracks; c.save_state(tmp_path)
    fresh = microcolony_controller(); fresh.position = lambda: 3                    # the view area shows position 3
    assert fresh.load_state(tmp_path) is True and fresh.page.tracks is None
    assert fresh.page.status.text() == "Position 0 is tracked: move the P slider to 0 to see it, or click Track to follow this position too."
    fresh.masks = lambda position: (None, list(range(8)) if position == 0 else [])   # 8 segmented frames of position 0, none elsewhere
    fresh.position = lambda: 5; fresh.refresh()                                     # a position with no segmented frame
    assert fresh.page.status.text() == "Position 0 is tracked: move the P slider to 0 to see it."
    fresh.position = lambda: 0; fresh.refresh()                                     # the P slider moved to 0
    assert fresh.page.tracks is fresh.tracked[0]
    assert fresh.page.until_spin.maximum() == 27.5 and fresh.page.until_spin.value() == 26.5      # "Follow until" shows where its tracks end


def test_a_save_with_nothing_worked_out_removes_the_results_of_an_earlier_save(tracks, growth, tmp_path):
    """Else a later Load would show the old results on whatever masks are in the folder now."""
    from nd2_analyzer.ui.biofilms.stages.growth_controller import GrowthController
    from nd2_analyzer.ui.biofilms.stages.growth_page import GrowthPage
    info = SimpleNamespace(name="toy", n_frames=8, dt_s=5400.0, pixel_um=0.1, start_hour=24.0, channel_roles={}, channel_labels={})
    c = microcolony_controller(); c.tracked[0] = tracks; c.drawn[0] = tracks; c.save_state(tmp_path)
    g = GrowthController(GrowthPage(), lambda: info); g.measured = growth; g.save_state(tmp_path)
    (tmp_path / "biofilm_stages" / "a_file_of_hers.txt").write_text("kept")
    assert len(list((tmp_path / "biofilm_stages").iterdir())) == 6
    assert microcolony_controller().save_state(tmp_path) == [] and GrowthController(GrowthPage(), lambda: info).save_state(tmp_path) == []
    assert [f.name for f in (tmp_path / "biofilm_stages").iterdir()] == ["a_file_of_hers.txt"]      # only the tool's own files go
    assert microcolony_controller().load_state(tmp_path) is False
