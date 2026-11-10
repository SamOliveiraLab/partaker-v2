"""The seeding rules on the Lee 2018 FC1 fixtures (early: landings and leavers; late: families and divisions) and on
toy data. The fixture tests skip when the fixtures are not on this machine; build them with
tests/fixtures/build_seeding_fixture.py. Run with -s to see the measured numbers."""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nd2_analyzer.analysis.biofilm_stages import seeding_rules as R   # noqa: E402
from nd2_analyzer.analysis.biofilm_stages.seeding_rules import (   # noqa: E402
    SeedingInputs, SeedingSettings, attach_track_ids, candidates_for_film, frame_fates, measure_seeding)

FIXTURES = Path.home() / "Workspace/results/partaker2_fixtures/seeding_lee_fc1"
KEYS_30S = ("leave_within_30s", "stay_by_landing_hour")
FLOW_KEYS = ("along_flow_shift", "orientation_vs_flow", "dividers_vs_holders", "growth_rate_eta")


def load_window(name: str, **changes) -> SeedingInputs:
    d = FIXTURES / name
    if not (d / "meta.json").exists():
        pytest.skip(f"fixture {name} not on this machine")
    meta = json.loads((d / "meta.json").read_text())
    kw = dict(cells=pl.read_parquet(d / "cells.parquet"), tracks=json.loads((d / "tracks.json").read_text()),
              dt_s=meta["dt_s"], pixel_um=meta["pixel_um"], start_hour=meta["start_hour"],
              n_frames=meta["n_frames_recording"], field_shape=tuple(meta["field_shape"]), flow=meta["flow"],
              frame_range=(meta["frames"]["first"], meta["frames"]["last"]))
    kw.update(changes)
    return SeedingInputs(**kw)


@pytest.fixture(scope="module")
def early():
    return measure_seeding(load_window("early"))


@pytest.fixture(scope="module")
def late():
    return measure_seeding(load_window("late"))


# ---------------------------------------------------------------------------------------------------------------------
# the fixtures against the paper's picture

def test_late_counts_rise(late):
    n = late.counts["cells"].to_numpy()
    assert n[-100:].mean() > n[:100].mean()
    assert late.metric("cells_on_surface").available
    lag = late.metric("lag_time")               # the window starts after the lag: the fit cannot be trusted and says so
    assert not lag.available and "start of these frames (27.5 to 29.5 h)" in lag.reason
    assert lag.series["t_lag_h"] < 27.5 and lag.series["n0"] == 60.0
    assert late.metric("take_off_hour").reason == "no frame was ever empty"


def test_most_early_landings_leave_within_30s(early):
    m = early.metric("leave_within_30s")
    assert m.available and m.value / 100 > 0.8
    assert early.metric("landings").value >= 50


def test_early_residence_median_under_10s(early):
    m = early.metric("residence_median")
    assert m.available and m.value < 10
    assert sum(m.series["counts"]) == m.n


def test_late_has_a_division(late):
    assert late.metric("division_count").value >= 1
    assert (late.per_track["fate"] == "divided").sum() >= 1
    assert late.metric("generations_per_family").series["families"][0] > 0


def test_every_metric_has_a_source_and_unavailable_ones_a_reason(early, late):
    for res in (early, late):
        assert len(res.metrics) == 24  # arrivals, commitment, movement, division, flow
        for m in res.metrics:
            assert m.source.strip(), m.key
            assert m.name == m.name[0].upper() + m.name[1:] and "—" not in m.name + m.text + m.reason
            if not m.available:
                assert m.reason.strip() and m.value is None, m.key
            else:
                assert m.value is not None and m.text and not m.reason, m.key


def test_slow_frames_mark_the_30s_rules_unavailable():
    res = measure_seeding(load_window("early", dt_s=1800.0))
    for key in KEYS_30S:
        m = res.metric(key)
        assert not m.available and m.reason == "frames 1800 s apart, the rule needs under 10 s"
    assert not res.metric("dividers_vs_holders").available


def test_no_flow_marks_the_flow_group_unavailable():
    res = measure_seeding(load_window("early", flow=None))
    for key in FLOW_KEYS:
        m = res.metric(key)
        assert not m.available and m.reason == "no flow in this recording" and m.group == "flow"
    res = measure_seeding(load_window("early", flow={"present": False, "shear_mpa": None, "direction": None}))
    assert all(not res.metric(k).available for k in FLOW_KEYS)


def test_empty_tracks_keep_counts_and_lag(late):
    res = measure_seeding(load_window("late", tracks=[]))
    assert res.metric("cells_on_surface").available
    assert res.metric("cells_on_surface").value == late.metric("cells_on_surface").value
    lag, lag_with_tracks = res.metric("lag_time"), late.metric("lag_time")     # the lag fit never looks at the tracks
    assert (lag.available, lag.reason, lag.series["t_lag_h"]) == (lag_with_tracks.available, lag_with_tracks.reason,
                                                                  lag_with_tracks.series["t_lag_h"])
    assert not res.metric("landings").available and res.metric("landings").reason == "no tracks for this position"
    assert res.per_track.height == 0 and res.candidates.height == 0 and frame_fates(res, 33000) == {}
    assert res.metric("handover_hour").available


def test_per_track_columns_and_fates(late):
    pt = late.per_track
    for col in ("track_id", "frame_start", "frame_end", "t_start_h", "t_end_h", "residence_s", "origin", "fate",
                "path_um", "net_disp_um", "motile", "pose_first", "n_children", "generation", "family_id", "divider",
                "along_flow_um"):
        assert col in pt.columns
    assert set(pt["origin"].unique()) <= set(R.ORIGINS) and set(pt["fate"].unique()) <= set(R.FATES)
    assert (pt.filter(pl.col("origin") == "present at start")["frame_start"] == 33000).all()
    assert (pt.filter(pl.col("fate") == "present at end")["frame_end"] == 35399).all()
    fates = frame_fates(late, 34000)
    assert fates and set(fates.values()) <= {"leave", "stay", "divides"}
    assert late.stage_by_frame.height == late.counts.height == 2400
    assert set(late.stage_by_frame["stage"].unique()) <= {"seeding", "microcolonies"}


def test_candidates_mix_the_three_reasons(late):
    c = candidates_for_film(late, k=8)
    assert c.height == 8 and c["track_id"].n_unique() == 8
    whys = " ".join(c["why"].to_list())
    assert "stays" in whys and "divides at" in whys
    assert (c["frame_end"] >= c["frame_start"]).all()


def test_print_measured_numbers(early, late):
    for name, res in (("early", early), ("late", late)):
        print(f"\n[{name}] " + " | ".join(res.notes[:2]))
        for key in ("landings", "leave_within_30s", "residence_median", "survival_curve", "fraction_present_at_end",
                    "lag_time", "take_off_hour", "motile_share", "pose_at_landing", "crawl_then_stop",
                    "division_count", "division_time_h", "division_outcome", "generations_per_family", "handover_hour",
                    "along_flow_shift", "orientation_vs_flow", "dividers_vs_holders", "growth_rate_eta"):
            m = res.metric(key)
            print(f"[{name}] {key:24s} {m.text if m.available else 'not available: ' + m.reason}  (n={m.n})")


# ---------------------------------------------------------------------------------------------------------------------
# toy data: the join, the classification and the handover rule

def toy_cells(points):
    """Cell rows from (time, x, y, major, minor) tuples, one cell_id per frame."""
    rows = []
    for t, x, y, major, minor in points:
        rows.append({"time": t, "position": 0, "cell_id": len(rows) + 1, "area": major * minor, "perimeter": 2 * (major + minor),
                     "eccentricity": 0.8, "major_axis_length": major, "minor_axis_length": minor, "centroid_y": y,
                     "centroid_x": x, "aspect_ratio": major / minor, "circularity": 0.5, "solidity": 1.0,
                     "equivalent_diameter": float(np.sqrt(4 * major * minor / np.pi)), "orientation": 1.5,
                     "morphology_class": "Rod", "y1": 0, "x1": 0, "y2": 1, "x2": 1, "fluorescence_channel": 0,
                     "fluo_level": 0.0})
    return pl.DataFrame(rows)


def toy_track(tid, t, x, y, parent=None, children=()):
    return {"ID": tid, "t": list(t), "x": list(x), "y": list(y), "parent": parent, "children": list(children)}


def toy_inputs(cells, tracks, **kw):
    base = dict(cells=cells, tracks=tracks, dt_s=3.0, pixel_um=0.1, start_hour=1.0, n_frames=100, field_shape=(200, 200),
                flow={"present": True, "shear_mpa": 20.0, "direction": "left to right"}, frame_range=(0, 99))
    base.update(kw)
    return SeedingInputs(**base)


def test_attach_track_ids_takes_the_nearest_row_within_3px():
    cells = toy_cells([(5, 50.0, 50.0, 20, 8), (5, 54.0, 50.0, 20, 8), (6, 50.5, 50.0, 20, 8)])
    tracks = [toy_track(1, [5, 6], [50.2, 50.6], [50.0, 50.1]), toy_track(2, [5], [53.9], [50.1]),
              toy_track(3, [6], [90.0], [90.0])]
    out = attach_track_ids(cells, tracks)
    assert out["track_id"].to_list() == [1, 2, 1]
    res = measure_seeding(toy_inputs(out, tracks))
    assert res.notes[1] == "3 cell rows and 3 tracks from the tool."
    assert res.per_track.filter(pl.col("track_id") == 1)["pose_first"][0] == "flat"


def test_origin_and_fate_classification():
    tracks = [
        toy_track(1, range(0, 30), [100] * 30, [100] * 30, children=[4, 5]),         # present at start, divides
        toy_track(2, range(10, 20), [100] * 10, [120] * 10),                           # landed, detached
        toy_track(3, range(10, 60), np.linspace(5, 60, 50), [150] * 50),               # entered field, detached
        toy_track(4, range(30, 100), [96] * 70, [100] * 70, parent=1),                 # born, present at end
        toy_track(5, range(30, 40), np.linspace(104, 196, 10), [100] * 10, parent=1),  # born, left field
    ]
    res = measure_seeding(toy_inputs(toy_cells([]), tracks))
    pt = res.per_track.sort("track_id")
    assert pt["origin"].to_list() == ["present at start", "landed", "entered field", "born on surface", "born on surface"]
    assert pt["fate"].to_list() == ["divided", "detached", "detached", "present at end", "left field"]
    assert pt["generation"].to_list() == [0, 0, 0, 1, 1] and pt["family_id"].to_list() == [1, 2, 3, 1, 1]
    assert pt["residence_s"].to_list() == [90.0, 30.0, 150.0, 210.0, 30.0]
    assert pt["t_start_h"][1] == pytest.approx(1.0 + 10 * 3 / 3600)
    assert res.metric("landings").value == 1 and res.metric("fraction_present_at_end").value == pytest.approx(20.0)
    assert res.metric("division_outcome").reason == "only 0 divisions scored"      # the leaving daughter voids it
    assert res.metric("along_flow_shift").available and res.metric("along_flow_shift").value > 0
    assert res.metric("motile_share").reason == "no cell rows to give the cell width"
    assert frame_fates(res, 35) == {3: "leave", 4: "stay", 5: "leave"}


def test_leave_within_window_counts_short_landings():
    tracks = [toy_track(i, range(10, 10 + n), [50 + 10 * i] * n, [50] * n) for i, n in enumerate([1, 2, 5, 9, 11, 40], 1)]
    res = measure_seeding(toy_inputs(toy_cells([]), tracks))
    m = res.metric("leave_within_30s")
    assert m.available and m.n == 6 and m.value == pytest.approx(100 * 4 / 6)
    assert res.metric("residence_median").value == pytest.approx(np.median([3, 6, 15, 27, 33, 120]))
    assert res.metric("survival_curve").reason.startswith("only 6 leavers")


def test_microcolony_handover_needs_four_close_cells_for_ten_frames():
    close = [(t, x, y, 12.0, 6.0) for t in range(50, 70) for x, y in [(50, 50), (50, 60), (60, 50), (60, 60)]]
    far = [(t, 150.0, 150.0, 12.0, 6.0) for t in range(0, 100)]
    res = measure_seeding(toy_inputs(toy_cells(close + far), []))
    stages = res.stage_by_frame["stage"].to_list()
    assert stages[:50] == ["seeding"] * 50 and stages[50:] == ["microcolonies"] * 50
    assert res.metric("handover_hour").value == pytest.approx(1.0 + 50 * 3 / 3600)
    brief = [(t, x, y, 12.0, 6.0) for t in range(50, 55) for x, y in [(50, 50), (50, 60), (60, 50), (60, 60)]]
    res = measure_seeding(toy_inputs(toy_cells(brief + far), []))
    assert set(res.stage_by_frame["stage"]) == {"seeding"} and not res.metric("handover_hour").available
    res = measure_seeding(toy_inputs(toy_cells(close + far), [], pixel_um=None))
    assert res.metric("handover_hour").reason == "no pixel size in this recording"


def test_lag_and_take_off_on_a_rising_count():
    hours_per_frame = 3 / 3600
    cells = [(t, 50.0, 50.0, 12.0, 6.0) for t in range(40, 100) for _ in range(int(round(2 * np.exp((t - 40) * hours_per_frame * 20))))]
    res = measure_seeding(toy_inputs(toy_cells(cells), []))
    take = res.metric("take_off_hour")
    assert take.available and take.value == pytest.approx(1.0 + 40 * hours_per_frame)
    lag = res.metric("lag_time")
    assert lag.available and lag.series["tau_h"] == pytest.approx(1 / 20, rel=0.1)
    assert lag.text.startswith("Lag ") and 1.0 <= lag.value <= 1.0 + 100 * hours_per_frame
    assert any(note.startswith("Lag fit: N0 held at the 2 cells") for note in res.notes)
    eta = res.metric("growth_rate_eta")
    assert eta.available and eta.value == pytest.approx(20, rel=0.1) and eta.name == "Growth rate at 20 mPa" and eta.text.endswith("per h")
    flat = measure_seeding(toy_inputs(toy_cells([(t, 50.0, 50.0, 12.0, 6.0) for t in range(100)]), []))
    assert flat.metric("lag_time").reason == "the count never rises in these frames"
    assert flat.metric("take_off_hour").reason == "no frame was ever empty"


def test_lag_before_the_analysed_frames_is_flagged():
    """Counts that already rise at the first frame: the fit puts the lag before the window and the metric says so."""
    hours_per_frame = 3 / 3600
    cells = [(t, 50.0, 50.0, 12.0, 6.0) for t in range(100) for _ in range(int(round(4 * np.exp(t * hours_per_frame * 20))))]
    res = measure_seeding(toy_inputs(toy_cells(cells), []))
    lag = res.metric("lag_time")
    assert not lag.available and lag.reason.startswith("the fitted lag") and "start of these frames" in lag.reason
    assert lag.series["n0"] == 4.0 and lag.value is None
    S = SeedingSettings()
    assert R._lag_outside_frames(5.0, 0.1, 1.0, 2.0, S).endswith("the rise is not in them")
    assert R._lag_outside_frames(1.5, 0.1, 1.0, 2.0, S) == ""
    assert R._lag_outside_frames(1.02, 0.1, 1.0, 2.0, S).endswith("the recording starts after the lag")


def test_frame_range_and_dt_are_required():
    with pytest.raises(ValueError, match="frame_range is needed"):
        measure_seeding(toy_inputs(toy_cells([]), [], frame_range=None))
    with pytest.raises(ValueError, match="dt_s must be positive"):
        measure_seeding(toy_inputs(toy_cells([]), [], dt_s=0.0))


def test_a_daughter_that_divides_counts_as_stayed():
    """Five divisions whose daughters all split again after 2 frames: every one keeps both daughters."""
    tracks = []
    for i in range(5):
        m, a, b = 10 * i + 1, 10 * i + 2, 10 * i + 3
        y = 20 + 30 * i
        tracks.append(toy_track(m, range(10, 20), [100] * 10, [y] * 10, children=[a, b]))
        tracks.append(toy_track(a, range(20, 22), [96] * 2, [y] * 2, parent=m, children=[a + 100, a + 200]))
        tracks.append(toy_track(b, range(20, 22), [104] * 2, [y] * 2, parent=m, children=[b + 100, b + 200]))
        for k in (a + 100, a + 200, b + 100, b + 200):          # the grandchildren stay to the end: their divisions are voided (no kids)
            tracks.append(toy_track(k, range(22, 100), [100 + (k % 7)] * 78, [y] * 78, parent=a if k < b + 100 else b))
    res = measure_seeding(toy_inputs(toy_cells([]), tracks))
    m = res.metric("division_outcome")
    assert m.available and m.series["two"] == 5 and m.text.startswith("One daughter stays in 0 of 5")
    assert res.metric("division_time_h").text == "6 s"            # 2 frames x 3 s between the splits
    assert R._hours_words(0.2) == "12 min" and R._hours_words(1.25) == "1.2 h"


def test_landings_seen_longer_than_the_window_at_the_end_count_as_stayed():
    """Three landings: one leaves after 2 frames, one is still there at the end after 50 frames (a sure stayer), one is
    still there at the end after 5 frames (unknown, left out and noted)."""
    tracks = [toy_track(1, range(10, 12), [50] * 2, [50] * 2), toy_track(2, range(50, 100), [80] * 50, [50] * 50),
              toy_track(3, range(95, 100), [110] * 5, [50] * 5)]
    res = measure_seeding(toy_inputs(toy_cells([]), tracks))
    m = res.metric("leave_within_30s")
    assert m.n == 2 and m.value == pytest.approx(50.0) and m.text == "50.0 %"
    assert res.metric("stay_by_landing_hour").n == 2
    assert res.metric("residence_median").n == 1              # only the landing with an end goes in the histogram
    assert any(note == "1 landing still under 30 s at the last frame left out of the 30 s rules." for note in res.notes)


def test_settings_are_named_and_the_module_has_no_dashes():
    s = SeedingSettings()
    assert s.stay_window_s == 30.0 and s.edge_margin_px == 10 and s.holder_min_track_s == 2400.0
    text = Path(R.__file__).read_text()
    assert "—" not in text and "–" not in text
