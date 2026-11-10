"""The Seeding page offscreen with the late Lee 2018 FC1 fixture measured by the tool's own rules: the page renders with
results and with None at 1200 x 800 and 1900 x 1000 and saves PNGs; nothing reaches past the right edge; the eight cards fill
in; the two big plots switch with the plot lists; the Numbers table has one row per metric; Follow and Field film ask for
the Seeding window; Play on the strip moves the cursor; the plots are free of overlapping elements.

Until the panel functions land in seeding_plots (PANELS, draw_panel, pick_landing), the tests install small stand-ins
built from the drawing functions that already exist, so the page's layout and signals are tested against the interface.
Run from src: cd src && python -m pytest ../tests/test_seeding_page.py"""
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import polars as pl
import pytest

from nd2_analyzer.analysis.biofilm_stages import seeding_plots as SP
from nd2_analyzer.analysis.biofilm_stages.seeding_rules import SeedingInputs, measure_seeding
from nd2_analyzer.analysis.biofilm_stages.seeding_types import RecordingInfo

FIXTURE = Path.home() / "Workspace/results/partaker2_fixtures/seeding_lee_fc1/late"
OUT = Path.home() / "Workspace/results/partaker2_fixtures/seeding_page"
SIZES = ((1200, 800), (1900, 1000))
FLOW = {"present": True, "shear_mpa": 20.0, "direction": "left to right"}


# ---------------------------------------------------------------------------------------------- the fixture results
def load_late() -> SeedingInputs:
    if not (FIXTURE / "meta.json").exists():
        pytest.skip("late fixture not on this machine")
    meta = json.loads((FIXTURE / "meta.json").read_text())
    return SeedingInputs(cells=pl.read_parquet(FIXTURE / "cells.parquet"), tracks=json.loads((FIXTURE / "tracks.json").read_text()),
                         dt_s=meta["dt_s"], pixel_um=meta["pixel_um"], start_hour=meta["start_hour"], n_frames=meta["n_frames_recording"],
                         field_shape=tuple(meta["field_shape"]), flow=meta["flow"], frame_range=(meta["frames"]["first"], meta["frames"]["last"]))


@pytest.fixture(scope="module")
def late():
    return measure_seeding(load_late())


@pytest.fixture(scope="module")
def meta():
    if not (FIXTURE / "meta.json").exists():
        pytest.skip("late fixture not on this machine")
    return json.loads((FIXTURE / "meta.json").read_text())


def info_for(meta: dict, flow: bool) -> RecordingInfo:
    return RecordingInfo(name="lee2018_WT_FC1", n_frames=meta["n_frames_recording"], dt_s=meta["dt_s"], pixel_um=meta["pixel_um"],
                         start_hour=meta["start_hour"], flow=FLOW if flow else None, channel_roles={"0": "cell_marker"},
                         field_shape=tuple(meta["field_shape"]))


def fake_film(rng, n=5):
    frames = []; trail = []; hours = np.linspace(27.6, 29.5, n).tolist(); fates = ["stay"] * (n - 1) + ["divides"]
    for k in range(n):
        img = rng.normal(0.1, 0.03, (120, 120)); cx, cy = 60 + 4 * k, 58 + 2 * k
        yy, xx = np.mgrid[:120, :120]; img[((xx - cx) / 9) ** 2 + ((yy - cy) / 4) ** 2 < 1] = 0.9
        frames.append(np.clip(img, 0, 1)); trail.append((cx, cy))
    return frames, hours, trail, fates


# ---------------------------------------------------------------------------------------------- stand-ins for the panels
def _cursor(ax, cursor_h):
    ax.axvline(float(cursor_h), color=SP.RED, lw=1.0)


@SP.with_house_style
def _stub_counts(fig, results, span, cursor_h, settings):
    ax = SP.draw_counts(fig, results, span); _cursor(ax, cursor_h); return ax


@SP.with_house_style
def _stub_stay(fig, results, span, cursor_h, settings):
    ax = SP.draw_stay(fig, results, span); _cursor(ax, cursor_h); return ax


@SP.with_house_style
def _stub_dots(fig, results, span, cursor_h, settings, selected=None):
    pt = results.per_track.filter(pl.col("origin") == "landed")
    ax = fig.add_axes([0.16, 0.24, 0.80, 0.63])
    x = pt["t_start_h"].to_numpy(); y = np.maximum(pt["residence_s"].to_numpy(), 3.0)
    ax.plot(x, y, ls="none", marker="o", ms=3, mfc=SP.BLUE, mec="none")
    if selected is not None:
        sel = pt.filter(pl.col("track_id") == int(selected))
        if sel.height:
            ax.plot(sel["t_start_h"].to_numpy(), np.maximum(sel["residence_s"].to_numpy(), 3.0), ls="none", marker="o", ms=8, mfc="none", mec=SP.INK, mew=1.5)
    ax.set_yscale("log"); ax.set_ylim(3, 36000); ax.set_xlim(*SP.pad_span(SP.recording_span(results, span)))
    _cursor(ax, cursor_h)
    ax.set_title("Every landing as a dot", loc="left"); ax.set_xlabel("landing time after inoculation (h)"); ax.set_ylabel("time stayed (s)")
    SP.open_frame(ax)
    return ax


def _stub_reason(title, words):
    @SP.with_house_style
    def draw(fig, results, span, cursor_h, settings):
        ax = SP.empty_axes(fig, words); ax.set_title(title, loc="left"); return ax
    return draw


STUB_PANELS = [
    ("arrivals", "cells_on_surface", "Cells on the surface", _stub_counts, False),
    ("arrivals", "stay_by_hour", "Still there after 30 s, by landing hour", _stub_stay, False),
    ("commitment", "landing_dots", "Every landing as a dot", _stub_dots, False),
    ("movement", "moved_or_still", "Moved or stayed still", _stub_reason("Moved or stayed still", "stand-in panel"), False),
    ("division", "daughters", "What happens to the daughters", _stub_reason("What happens to the daughters", "stand-in panel"), False),
    ("flow", "along_flow", "Shift along the flow", _stub_reason("Shift along the flow", "stand-in panel"), True),
]


def _stub_draw_panel(fig, panel, results, span, cursor_h, settings, selected=None):
    fig.clear()
    if results is None:
        ax = SP.empty_axes(fig, "Run Measure to see this plot"); ax.set_title(panel[2], loc="left"); return ax
    if panel[1] == "landing_dots":
        return panel[3](fig, results, span, cursor_h, settings, selected=selected)
    return panel[3](fig, results, span, cursor_h, settings)


def _stub_pick_landing(ax, results, x_display, y_display, tol_px=8):
    pt = results.per_track.filter(pl.col("origin") == "landed")
    pts = ax.transData.transform(np.column_stack([pt["t_start_h"].to_numpy(), np.maximum(pt["residence_s"].to_numpy(), 3.0)]))
    d = np.hypot(pts[:, 0] - x_display, pts[:, 1] - y_display)
    k = int(np.argmin(d))
    return int(pt["track_id"][k]) if d[k] <= tol_px else None


@pytest.fixture(autouse=True)
def panel_interface(monkeypatch):
    """The real panel functions when seeding_plots has them, the stand-ins otherwise."""
    if not hasattr(SP, "PANELS"):
        monkeypatch.setattr(SP, "PANELS", STUB_PANELS, raising=False)
        monkeypatch.setattr(SP, "draw_panel", _stub_draw_panel, raising=False)
    if not hasattr(SP, "pick_landing"):
        monkeypatch.setattr(SP, "pick_landing", _stub_pick_landing, raising=False)


# ---------------------------------------------------------------------------------------------- Qt helpers
@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _fig(size=(5.6, 2.6)):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=size, dpi=100); FigureCanvasAgg(fig); return fig


def _settle(app, page, n=4):
    for _ in range(n):
        app.processEvents()
    page.flush_redraws()
    app.processEvents()


def _grab(app, page, name: str, size=(1200, 800)):
    OUT.mkdir(parents=True, exist_ok=True)
    page.resize(*size); page.show(); _settle(app, page)
    img = page.grab(); img.save(str(OUT / f"{name}_{size[0]}x{size[1]}.png"))
    return img


def _right_edge(page) -> int:
    """The rightmost pixel any visible child widget reaches, in page coordinates."""
    from PySide6.QtWidgets import QWidget
    edge = 0
    for w in page.findChildren(QWidget):
        if not w.isVisible() or w.window() is not page.window():
            continue
        g = w.geometry(); top_left = w.parentWidget().mapTo(page, g.topLeft()) if w.parentWidget() is not None else g.topLeft()
        edge = max(edge, top_left.x() + g.width())
    return edge


def _headings(page) -> list:
    from PySide6.QtWidgets import QLabel
    return [lab.text() for lab in page.gallery_box.findChildren(QLabel)]


def _landing_canvas(page):
    return next(c for c in page.panels if c.key == "landing_dots")


# ---------------------------------------------------------------------------------------------- tests
def test_late_results_have_the_contract_shapes(late):
    assert set(late.counts.columns) >= {"frame", "hour", "cells", "empty"} and late.counts.height == 2400
    assert all(m.source for m in late.metrics) and all(m.reason for m in late.metrics if not m.available)
    assert {"arrivals", "commitment", "movement", "division", "flow"} <= {m.group for m in late.metrics}
    assert late.per_track.filter(pl.col("origin") == "landed").height > 100


def test_page_renders_with_none_and_with_results_at_two_sizes(app, late, meta):
    from nd2_analyzer.analysis.biofilm_stages import seeding_figures as SF
    from nd2_analyzer.ui.biofilms.stages.seeding_page import SeedingPage
    rng = np.random.default_rng(3)
    page = SeedingPage()
    assert not page.measure_button.isEnabled() and not page.strip_play_button.isEnabled() and not page.numbers_button.isEnabled()
    assert page.status.text().startswith("Load a recording")
    for size in SIZES:
        _grab(app, page, "no_results", size)
        assert _right_edge(page) <= size[0]
    assert len(page.panels) == 2 and all(c.time_panel is None for c in page.panels)        # nothing to tell until results exist
    assert [c.key for c in page.panels] == list(SF.FIGURE_DEFAULTS)
    assert all(combo.count() == len(SF.FIGURES) for combo in page.plot_combos)

    page.set_recording(info_for(meta, flow=False))
    page.set_step_state("load", "done", f"{meta['n_frames_recording']:,} frames"); page.set_step_state("segment", "done", "2,400 frames")
    page.set_step_state("track", "done", f"{meta['n_tracks']:,} tracks"); page.set_step_state("measure", "todo")
    assert page.measure_button.isEnabled() and page.strip_play_button.isEnabled()
    page.set_hour(28.0)
    page.set_results(late)
    page.set_film_frames(*fake_film(rng))
    for size in SIZES:
        _grab(app, page, "results_no_flow", size)
        assert _right_edge(page) <= size[0]
    facts = SF.card_facts(late)
    assert len(facts) == len(page.cards) == 8
    assert [c.value.text() for c in page.cards] == [f[0] for f in facts] and any(c.isEnabled() for c in page.cards)
    assert "no flow" in page.recording_label.text() and page.numbers_button.isEnabled()
    assert not page.time_panels()                         # the six big figures carry no cursor, as in the mock

    page.plot_combos[1].setCurrentIndex(page.plot_combos[1].findData("fig_late")); _settle(app, page)
    assert page.panels[1].key == "fig_late"
    page.plot_combos[0].setCurrentIndex(page.plot_combos[0].findData("fig_divisions")); _settle(app, page)
    assert page.panels[0].key == "fig_divisions"
    page.set_recording(info_for(meta, flow=True)); page.set_step_state("measure", "done", f"{len(late.metrics)} rules")
    page.set_results(late); page.set_film_frames(*fake_film(rng)); page.set_hour(29.0)
    for size in SIZES:
        _grab(app, page, "results_flow", size)
        assert _right_edge(page) <= size[0]


def test_cells_to_follow_has_two_sections(app, late, meta):
    from PySide6.QtCore import Qt
    from nd2_analyzer.analysis.biofilm_stages import seeding_figures as SF
    from nd2_analyzer.ui.biofilms.stages.seeding_page import SeedingPage, cells_to_follow
    sections = cells_to_follow(late)
    assert [s[0] for s in sections] == ["irreversible attachment", "reversible attachment: stayed 30 min or more"]
    for_good, long_left = sections[0][1], sections[1][1]
    assert for_good and all(r["origin"] == "landed" for r in for_good + long_left)
    assert all(r["fate"] == "divided" or (r["fate"] == "present at end" and r["residence_s"] >= late.settings.irreversible_stay_s)
               for r in for_good)
    assert all(r["fate"] in ("detached", "left field") and r["residence_s"] >= SF.LONG_STAY_S for r in long_left)
    for rows in (for_good, long_left):
        assert [r["t_start_h"] for r in rows] == sorted(r["t_start_h"] for r in rows)
    page = SeedingPage(); page.set_recording(info_for(meta, flow=True)); page.set_results(late)
    titles = [page.candidates.item(r) for r in range(page.candidates.count()) if page.candidates.item(r).data(Qt.UserRole) is None]
    assert [t.text() for t in titles if not t.text().startswith("and ")] == [s[0] for s in sections if s[1]]
    assert all(not (t.flags() & Qt.ItemIsSelectable) for t in titles)
    first = page.candidates.item(1)
    assert first.text().startswith(f"Track {for_good[0]['track_id']}: lands {for_good[0]['t_start_h']:.1f} h, ")
    assert page.candidates.currentRow() == 1 and page.selected_track() == int(for_good[0]["track_id"])
    page.candidates.setCurrentRow(0)                       # a section title stands for the first track under it
    assert page.selected_track() == int(for_good[0]["track_id"])


def test_numbers_dialog_has_one_row_per_metric(app, late, meta):
    from PySide6.QtCore import Qt
    from nd2_analyzer.ui.biofilms.stages.seeding_page import SeedingPage
    page = SeedingPage(); page.set_recording(info_for(meta, flow=True))
    assert page.open_numbers() is None
    page.set_results(late)
    page.numbers_button.click()
    dlg = page.numbers_dialog
    assert dlg is not None and dlg.isVisible() and dlg.table.rowCount() == len(late.metrics)
    assert dlg.table.columnCount() == 3 and dlg.table.item(0, 0).text() == late.metrics[0].name
    unavailable = [k for k, m in enumerate(late.metrics) if not m.available]
    if unavailable:
        k = unavailable[0]
        assert dlg.table.item(k, 1).text() == late.metrics[k].reason and not (dlg.table.item(k, 1).flags() & Qt.ItemIsEnabled)
    available = [k for k, m in enumerate(late.metrics) if m.available][0]
    assert dlg.table.item(available, 1).text() == late.metrics[available].text
    assert dlg.table.item(available, 0).toolTip() == f"Rule from {late.metrics[available].source}"
    OUT.mkdir(parents=True, exist_ok=True); dlg.grab().save(str(OUT / "numbers_dialog.png"))
    dlg.close()


def test_follow_and_field_film_ask_for_the_seeding_window(app, late, meta):
    from nd2_analyzer.ui.biofilms.stages.seeding_page import SeedingPage
    page = SeedingPage(); page.set_recording(info_for(meta, flow=True)); page.set_results(late)
    got = {"follow": [], "window": []}
    page.follow_requested.connect(got["follow"].append)
    page.window_requested.connect(lambda view, tid: got["window"].append((view, tid)))
    tid = page.selected_track()
    assert tid is not None
    page.follow_button.click(); page.field_button.click(); app.processEvents()
    assert got["follow"] == [tid] and got["window"] == [("Field", tid)]
    assert page.followed_track == tid


def test_strip_play_moves_the_cursor(app, late, meta):
    from nd2_analyzer.ui.biofilms.stages.seeding_page import HOURS_PER_SECOND, PLAY_TICK_MS, SeedingPage
    page = SeedingPage(); page.set_recording(info_for(meta, flow=False)); page.set_results(late)
    hours = []; page.hour_changed.connect(hours.append)
    page.set_hour(28.0)
    page.strip_play_button.click()
    assert page.strip_timer.isActive() and page.strip_play_button.text() == "Pause"
    page._strip_play_step()
    assert len(hours) == 1 and 28.0 + HOURS_PER_SECOND * PLAY_TICK_MS / 1000.0 <= hours[0] < 28.0 + HOURS_PER_SECOND * 0.5
    assert page.redraw_timer.isActive()                    # the time panels wait for the next tick, at most ten a second
    page.flush_redraws(); assert not page.redraw_timer.isActive()
    page.strip_play_button.click()
    assert not page.strip_timer.isActive() and page.strip_play_button.text() == "Play"
    page.set_hour(page.info.end_hour); page.strip_play_button.click()      # Play at the end starts over
    assert hours[-1] == pytest.approx(page.info.start_hour) and page.strip_timer.isActive()
    page.play_button.click()                                                # the film's Play stops the strip's
    assert not page.strip_timer.isActive() and page.play_button.text() == "Stop film"
    page.play_button.click(); assert page.play_button.text() == "Play film"
    page.set_results(None); assert not page.redraw_timer.isActive()


def test_page_signals(app, late, meta):
    from nd2_analyzer.ui.biofilms.stages.seeding_page import SeedingPage
    page = SeedingPage(); got = {}
    page.hour_changed.connect(lambda h: got.setdefault("hour", h)); page.follow_requested.connect(lambda t: got.setdefault("follow", t))
    page.measure_requested.connect(lambda: got.setdefault("measure", True)); page.export_requested.connect(lambda w: got.setdefault("export", w))
    page.set_recording(info_for(meta, flow=False)); page.set_results(late)
    page.set_step_state("segment", "done"); page.set_step_state("track", "done")
    page.measure_button.click(); page.follow_button.click(); page.image_button.click()
    page.strip.set_hour(28.0, emit=True)
    first = page.selected_track()
    assert got == {"measure": True, "follow": first, "export": "image", "hour": 28.0}
    assert page.track_span(first) is not None
    page.set_hour(29.0); assert page.strip.cursor_h == 29.0 and got["hour"] == 28.0


def test_step_words():
    from nd2_analyzer.ui.biofilms.stages.seeding_page import short_detail, step_words
    assert step_words("done", "35,400") == "35,400" and step_words("todo") == "not yet" and step_words("running", "12 %") == "running, 12 %"
    assert "done" not in step_words("done", "") and step_words("failed", "no frames") == "failed, no frames"
    long = "ColumnNotFoundError: " + ", ".join(f"column_{i}" for i in range(40))
    assert len(short_detail(long)) == 60 and short_detail(long).endswith("...") and short_detail("done") == "done"


def test_track_lines():
    from nd2_analyzer.ui.biofilms.stages.seeding_page import track_line
    assert track_line({"track_id": 3930, "t_start_h": 24.2, "t_end_h": 26.7, "fate": "divided"}) == "Track 3930: lands 24.2 h, divides 26.7 h"
    assert track_line({"track_id": 5, "t_start_h": 27.61, "t_end_h": 29.5, "fate": "present at end"}) == "Track 5: lands 27.6 h, stays to the end"
    assert track_line({"track_id": 7, "t_start_h": 27.0, "t_end_h": 28.1, "fate": "detached"}) == "Track 7: lands 27.0 h, left 28.1 h"


def test_a_long_failure_text_does_not_widen_the_page(app, meta):
    from nd2_analyzer.ui.biofilms.stages.seeding_page import SeedingPage
    page = SeedingPage(); page.set_recording(info_for(meta, flow=True))
    before = page.minimumSizeHint().width()
    long = "ColumnNotFoundError: " + ", ".join(f"column_{i}" for i in range(40))
    page.set_step_state("track", "failed", long)
    assert page.minimumSizeHint().width() <= before + 40
    assert page.step_labels["track"].toolTip() == long and page.status.text() == f"Track failed: {long}"


def test_no_dashes_in_the_page_module():
    src = Path(__file__).resolve().parents[1] / "src/nd2_analyzer/ui/biofilms/stages/seeding_page.py"
    text = src.read_text()
    assert chr(0x2014) not in text and chr(0x2013) not in text


@pytest.mark.parametrize("flow", [True, False])
def test_plots_have_no_overlapping_elements(late, meta, flow):
    span = info_for(meta, flow).span
    OUT.mkdir(parents=True, exist_ok=True)
    fig = _fig(); SP.draw_counts(fig, late, span); assert SP.collisions(fig) == []; fig.savefig(OUT / "plot_counts.png", dpi=100)
    fig = _fig(); SP.draw_stay(fig, late, span); assert SP.collisions(fig) == []; fig.savefig(OUT / "plot_stay.png", dpi=100)
    fig = _fig(); SP.draw_counts(fig, None, span); assert SP.collisions(fig) == []
    for cursor, name in ((27.5, "start"), (28.5, "middle"), (29.4, "end")):
        fig = _fig((11.0, 1.0)); ax = fig.add_axes([0.03, 0.36, 0.94, 0.46])
        SP.draw_strip(ax, span, cursor, late.stage_by_frame["hour"].to_numpy(), late.stage_by_frame["stage"].to_list())
        assert SP.collisions(fig) == [], name
        fig.savefig(OUT / f"strip_{name}.png", dpi=100)
    fig = _fig((11.0, 1.0)); ax = fig.add_axes([0.03, 0.36, 0.94, 0.46]); SP.draw_strip(ax, span, 3.0); assert SP.collisions(fig) == []


def test_strip_label_steps_aside_from_the_cursor():
    hours = np.linspace(0, 29.5, 300); stages = np.where(hours < 24.0, "seeding", "microcolonies").tolist()
    fig = _fig((11.0, 1.0)); ax = fig.add_axes([0.03, 0.36, 0.94, 0.46]); SP.draw_strip(ax, (0, 29.5), 12.0, hours, stages)
    seeding = [t for t in ax.texts if t.get_text() == "reversible attachment"][0]
    assert abs(seeding.get_position()[0] - 12.0) > 1.0 and SP.collisions(fig) == []
    OUT.mkdir(parents=True, exist_ok=True); fig.savefig(OUT / "strip_cursor_on_label.png", dpi=100)


def test_film_strip_has_no_overlapping_elements(meta):
    rng = np.random.default_rng(5)
    OUT.mkdir(parents=True, exist_ok=True)
    for n in (4, 6):
        fig = _fig((3.3, 1.25)); axes = SP.draw_film(fig, *fake_film(rng, n), pixel_um=meta["pixel_um"])
        assert len(axes) == n and SP.collisions(fig) == []
        fig.savefig(OUT / f"film_{n}.png", dpi=100)
    fig = _fig((3.3, 1.25)); SP.draw_film(fig, [], [], [], []); assert SP.collisions(fig) == []


def test_narrow_bands_drop_their_label():
    fig = _fig((11.0, 1.0)); ax = fig.add_axes([0.03, 0.36, 0.94, 0.46])
    hours = np.linspace(0, 29.5, 300); stages = np.where(hours < 29.2, "seeding", "microcolonies").tolist()
    SP.draw_strip(ax, (0, 29.5), 10.0, hours, stages)
    labels = [t.get_text() for t in ax.texts]
    assert "reversible attachment" in labels and "microcolonies" not in labels


def test_the_strip_follows_the_view_area_t_slider(meta):
    from nd2_analyzer.ui.biofilms.stages.tab import BiofilmStagesTab
    tab = BiofilmStagesTab()
    tab.seeding_page.set_recording(info_for(meta, flow=False))
    frame = meta["n_frames_recording"] // 2
    tab.follow_frame(frame)
    want = meta["start_hour"] + frame * meta["dt_s"] / 3600.0
    assert abs(tab.seeding_page.strip.cursor_h - want) < 1e-6 and abs(tab.shared_hour - want) < 1e-6
