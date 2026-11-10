"""Whether the frames show seeding: three signs that hold for any species and frame rate (the surface crowded in the
first frame, microcolonies in the first frame, more cells born than landed), two of them make the page ask; the warning
opens without blocking, the link brings it back, and "Clear them" asks the controller to drop the numbers."""
import dataclasses
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import polars as pl
import pytest

from nd2_analyzer.analysis.biofilm_stages.seeding_rules import SeedingInputs, measure_seeding, not_seeding_signs
from nd2_analyzer.analysis.biofilm_stages.seeding_types import Metric, SeedingSettings

S = SeedingSettings()
FIXTURE = Path.home() / "Workspace/results/partaker2_fixtures/seeding_lee_fc1/late"


def _counts(cells, dt_h=1 / 12):
    return pl.DataFrame({"frame": list(range(len(cells))), "hour": [i * dt_h for i in range(len(cells))], "cells": cells})


def _tracks(landed, born):
    return pl.DataFrame({"origin": ["landed"] * landed + ["born on surface"] * born})


def _handover(hour, available=True):
    return Metric(key="handover_hour", name="Microcolonies from", value=hour if available else None, unit="h",
                  text="", available=available, reason="" if available else "no microcolony", source="", group="division")


def test_a_seeding_recording_shows_no_sign():
    # Lee WT flow cell 1 as the tool measured it: 3 cells at first, 100 at most, 10,460 landed, 38 born, microcolonies at 26 h
    assert not_seeding_signs(_counts([3] + [50] * 10 + [100], dt_h=3 / 3600), _tracks(10460, 38), _handover(26.0), S) == []


def test_a_growing_colony_shows_all_three_signs():
    # Pair 1-5 file 1 position 0: 247 cells at first, 508 at most, 151 landed, 350 born, microcolonies from the first frame
    signs = not_seeding_signs(_counts([247, 300, 400, 508]), _tracks(151, 350), _handover(0.0), S)
    assert len(signs) == 3
    assert "247 cells" in signs[0] and "49 %" in signs[0]
    assert "first frame (0.0 h)" in signs[1]
    assert "(350)" in signs[2] and "(151)" in signs[2]


def test_one_sign_alone_is_not_enough():
    # crowded at the start but cells keep landing and no microcolony yet: a recording that starts after the first landings
    assert not_seeding_signs(_counts([40, 60, 100]), _tracks(500, 20), _handover(5.0), S) == []


def test_a_few_early_cells_are_not_a_crowd():
    # half of the most cells, but only 5: no sign from the start; microcolonies and births still make two signs
    signs = not_seeding_signs(_counts([5, 8, 10]), _tracks(3, 9), _handover(0.0), S)
    assert len(signs) == 2 and not any("cells are already on the surface" in s for s in signs)


def test_no_microcolony_is_no_sign():
    assert not_seeding_signs(_counts([247, 508]), _tracks(500, 20), _handover(None, available=False), S) == []


@pytest.fixture(scope="module")
def late():
    if not (FIXTURE / "meta.json").exists():
        pytest.skip("late fixture not on this machine")
    meta = json.loads((FIXTURE / "meta.json").read_text())
    inp = SeedingInputs(cells=pl.read_parquet(FIXTURE / "cells.parquet"), tracks=json.loads((FIXTURE / "tracks.json").read_text()),
                        dt_s=meta["dt_s"], pixel_um=meta["pixel_um"], start_hour=meta["start_hour"],
                        n_frames=meta["n_frames_recording"], field_shape=tuple(meta["field_shape"]), flow=meta["flow"],
                        frame_range=(meta["frames"]["first"], meta["frames"]["last"]))
    return measure_seeding(inp)


def test_the_late_window_starts_after_the_surface_filled_up(late):
    # 27.5 to 29.5 h of a real seeding recording: crowded start and microcolonies, though landings still outnumber births
    assert len(late.check) == 2
    assert "60 cells" in late.check[0] and "Microcolonies" in late.check[1]
    assert late.notes[0].startswith("Check: these frames may not show seeding.")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_page_warns_without_blocking_and_can_clear(app, late):
    from nd2_analyzer.ui.biofilms.stages.seeding_page import SeedingPage
    page = SeedingPage()
    cleared = []
    page.clear_requested.connect(lambda: cleared.append(True))
    page.set_results(dataclasses.replace(late, check=[]))
    assert page.check_link.isHidden() and page.ask_about_seeding() is None
    page.set_results(late)
    assert not page.check_link.isHidden()
    box = page.ask_about_seeding()
    assert box is not None and box.isVisible() and box.text() == "Do these frames show seeding?"
    assert all(s in box.informativeText() for s in late.check)
    keep, clear = box.buttons()[0], box.buttons()[1]
    assert {keep.text(), clear.text()} == {"Keep the numbers", "Clear them"}
    [b for b in box.buttons() if b.text() == "Keep the numbers"][0].click()
    assert cleared == []
    box = page.ask_about_seeding()
    [b for b in box.buttons() if b.text() == "Clear them"][0].click()
    assert cleared == [True]
    box = page.ask_about_seeding()
    assert box.close() and cleared == [True]                 # closing is keeping: the box goes, nothing is cleared
    page.set_results(None)
    assert page.check_link.isHidden() and page.check_box is None
