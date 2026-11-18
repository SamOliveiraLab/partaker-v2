"""The Numbers button of the Microcolonies and Maturation pages: the tables behind every plot, written as CSV files."""
import csv
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nd2_analyzer.analysis.biofilm_stages import growth_measure as GM
from nd2_analyzer.analysis.biofilm_stages import matrix_stain as MX
from nd2_analyzer.analysis.biofilm_stages import microcolony_follow as MF
from nd2_analyzer.analysis.biofilm_stages import microcolony_metrics as MM
from nd2_analyzer.analysis.biofilm_stages import stage_tables as ST
from test_growth_measure import recording
from test_microcolony_metrics import frame
from test_microcolony_rules import PX, blocks

FRAMES = [blocks(20), blocks(20), blocks(3), blocks(3), blocks(3), blocks(3)]          # two microcolonies of 4 cells that merge at 25 h


@pytest.fixture(scope="module")
def tracks():
    return MF.follow_microcolonies(lambda t: FRAMES[t], range(6), pixel_um=PX, dt_h=0.5, start_hour=24.0)


@pytest.fixture(scope="module")
def metrics():
    return MM.measure_metrics(lambda p, t: frame(with_last_of_b=(p == 0)), {0: [0, 3], 1: [0, 3]}, 0.1, hour_of=lambda t: 24.0 + 0.5 * t)


@pytest.fixture(scope="module")
def growth():
    labels, dye = recording()
    ref = MX.DyeReference(zero=np.zeros((240, 240), np.float32), flat=np.ones((240, 240), np.float32))
    return GM.measure_growth_states(labels, dye, {0: list(range(8)), 1: list(range(8))}, pixel_um=0.1, hour_of=lambda t: 24.0 + 1.5 * t, reference=ref, resamples=50)


def test_the_tables_of_the_followed_microcolonies(tracks):
    t = ST.microcolony_tables(tracks, position=2)
    assert list(t) == ["microcolonies_over_time_p2", "microcolony_merges_p2", "microcolonies_followed_p2", "microcolony_frames_p2"]
    head, rows = t["microcolonies_over_time_p2"]
    assert head == ["hours", "frame", "microcolonies", "cells_in_microcolonies_pct"]
    assert rows == [[24.0, 0, 2, 100.0], [24.5, 1, 2, 100.0], [25.0, 2, 1, 100.0], [25.5, 3, 1, 100.0], [26.0, 4, 1, 100.0], [26.5, 5, 1, 100.0]]
    assert t["microcolony_merges_p2"] == (["hours", "frame", "microcolony", "merges_into", "cells", "cells_of_the_other"], [[25.0, 2, 2, 1, 4, 4]])
    head, rows = t["microcolonies_followed_p2"]
    assert head == ["microcolony", "first_h", "last_h", "began", "ended", "merges_into", "most_cells"]
    assert rows == [[1, 24.0, 26.5, "at start", "still there", "", 8], [2, 24.0, 25.0, "at start", "merged", 1, 4]]
    head, rows = t["microcolony_frames_p2"]
    assert head == ["microcolony", "frame", "hours", "cells", "area_um2"] and len(rows) == 6 + 2 and rows[0][:4] == [1, 0, 24.0, 4]


def test_the_tables_of_the_metrics_of_every_position(metrics):
    t = ST.metric_tables(metrics)
    assert list(t) == ["microcolony_metrics_every_position", "microcolony_metrics_mean", "microcolony_density_by_distance"]
    head, rows = t["microcolony_metrics_every_position"]
    assert head[:5] == ["position", "frame", "hours", "cells", "microcolonies"] and "random_pct" in head and len(rows) == 4
    assert rows[0][:5] == [0, 0, 24.0, 11, 2] and rows[2][:5] == [1, 0, 24.0, 10, 1]
    assert rows[2][head.index("spacing_um")] == ""                            # no neighbour: an empty cell, not "nan"
    head, rows = t["microcolony_metrics_mean"]
    assert head[:4] == ["hours", "positions", "cells_mean", "cells_sd"] and rows[0][:3] == [24.0, 2, 10.5]
    head, rows = t["microcolony_density_by_distance"]
    assert head == ["hours", "from_um", "to_um", "cells", "cells_within_3um_per_100um2"]
    # ring 0 to 2 um holds the 9 microcolony cells of position 0 and the 5 of position 1; their neighbours add up to 18 + 12 + 18
    assert rows[0][:4] == [24.0, 0.0, 2.0, 14] and rows[0][4] == pytest.approx(100 * (18 + 12 + 18) / 14 / (np.pi * 9))


def test_the_tables_of_the_growth_states(growth):
    t = ST.growth_tables(growth)
    assert list(t) == ["growth_states_every_position", "growth_states_mean", "growth_states_thresholds"]
    head, rows = t["growth_states_every_position"]
    assert head == ["position", "frame", "hours", "cells", "cells_per_h", "matrix_pct", "single_stained_cells_pct", "matrix_inside_microcolonies_pct",
                    "matrix_outside_pct", "microcolony_area_pct"]
    assert len(rows) == 16 and rows[0][:4] == [0, 0, 24.0, 4] and rows[8][:4] == [1, 0, 24.0, 6] and rows[0][5] == 0.0
    head, rows = t["growth_states_mean"]
    assert head[:4] == ["hours", "positions", "cells_mean", "cells_sd"] and rows[0][:3] == [24.0, 2, 5.0]
    assert head[4:8] == ["cells_per_h_mean", "cells_per_h_sd", "matrix_pct_mean", "matrix_pct_sd"] and head[-1] == "microcolony_area_pct_sd"   # the names of the first table
    head, rows = t["growth_states_thresholds"]
    assert head == ["threshold", "hours", "low_h", "high_h", "note"]
    assert rows[0][:2] == ["Th1", 30.75] and rows[1][0] == "Th1 position 0" and rows[1][1] == 30.0 and rows[3][0] == "Th2"


def test_the_tables_are_written_as_csv_files(tracks, metrics, tmp_path):
    written = ST.write_tables(tmp_path / "numbers", {**ST.microcolony_tables(tracks, 2), **ST.metric_tables(metrics)})
    assert [os.path.basename(f) for f in written] == [
        "microcolonies_over_time_p2.csv", "microcolony_merges_p2.csv", "microcolonies_followed_p2.csv", "microcolony_frames_p2.csv",
        "microcolony_metrics_every_position.csv", "microcolony_metrics_mean.csv", "microcolony_density_by_distance.csv"]
    with open(written[1], newline="") as f:
        assert list(csv.reader(f)) == [["hours", "frame", "microcolony", "merges_into", "cells", "cells_of_the_other"], ["25.0", "2", "2", "1", "4", "4"]]


# ---------------------------------------------------------------- the button
def test_the_microcolonies_page_has_a_numbers_button(tracks):
    from nd2_analyzer.ui.biofilms.stages.microcolony_page import MicrocolonyPage
    page = MicrocolonyPage()
    assert page.numbers_button.text() == "Numbers" and not page.numbers_button.isEnabled()
    page.set_tracks(tracks, None)
    asked = []; page.export_requested.connect(asked.append); assert page.numbers_button.isEnabled()
    page.numbers_button.click(); assert asked == ["numbers"]


def test_the_maturation_page_has_a_numbers_button(growth):
    from nd2_analyzer.ui.biofilms.stages.growth_page import GrowthPage
    page = GrowthPage()
    assert page.numbers_button.text() == "Numbers" and not page.numbers_button.isEnabled()
    page.set_measured(growth, None)
    asked = []; page.export_requested.connect(asked.append); assert page.numbers_button.isEnabled()
    page.numbers_button.click(); assert asked == ["numbers"]


def test_numbers_writes_the_tables_of_the_microcolonies_page_into_the_chosen_folder(tracks, metrics, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PySide6.QtWidgets import QFileDialog
    from nd2_analyzer.ui.biofilms.stages.microcolony_controller import MicrocolonyController
    from nd2_analyzer.ui.biofilms.stages.microcolony_page import MicrocolonyPage
    info = SimpleNamespace(name="toy", n_frames=6, dt_s=1800.0, pixel_um=PX, start_hour=24.0, channel_roles={}, channel_labels={})
    c = MicrocolonyController(MicrocolonyPage(), lambda: info, lambda: 0)
    c.tracked[2] = tracks; c.metrics = metrics
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path)))
    c.export("numbers")
    assert sorted(f.name for f in tmp_path.iterdir()) == sorted([
        "microcolonies_over_time_p2.csv", "microcolony_merges_p2.csv", "microcolonies_followed_p2.csv", "microcolony_frames_p2.csv",
        "microcolony_metrics_every_position.csv", "microcolony_metrics_mean.csv", "microcolony_density_by_distance.csv"])
    assert c.page.status.text() == f"Numbers saved: 7 CSV files in {tmp_path}"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: ""))      # the dialog was cancelled
    c.page.status.setText("as it was"); c.export("numbers")
    assert c.page.status.text() == "as it was"


def test_numbers_writes_the_tables_of_the_maturation_page_into_the_chosen_folder(growth, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PySide6.QtWidgets import QFileDialog
    from nd2_analyzer.ui.biofilms.stages.growth_controller import GrowthController
    from nd2_analyzer.ui.biofilms.stages.growth_page import GrowthPage
    info = SimpleNamespace(name="toy", n_frames=8, dt_s=5400.0, pixel_um=0.1, start_hour=24.0, channel_roles={}, channel_labels={})
    c = GrowthController(GrowthPage(), lambda: info); c.measured = growth
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path)))
    c.export("numbers")
    assert sorted(f.name for f in tmp_path.iterdir()) == ["growth_states_every_position.csv", "growth_states_mean.csv", "growth_states_thresholds.csv"]
    assert c.page.status.text() == f"Numbers saved: 3 CSV files in {tmp_path}"
