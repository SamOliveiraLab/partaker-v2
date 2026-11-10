"""The Seeding window and the six big figures on the late Lee 2018 FC1 fixture, measured by the tool's own rules: every
figure draws with no overlapping element and greys out without results; the fate classes and the irreversible rule (a
landing that divides, or is still there at the end after an hour); the eight cards; the window's drawing in both views, the
click that picks a cell, the dialog stepping, switching view and following a clicked cell, and the saved film."""
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import polars as pl
import pytest

from nd2_analyzer.analysis.biofilm_stages import seeding_figures as SF
from nd2_analyzer.analysis.biofilm_stages import seeding_plots as SP
from nd2_analyzer.analysis.biofilm_stages.mask_tracker import track_rows
from nd2_analyzer.analysis.biofilm_stages.seeding_rules import FATE_KEYS, SeedingInputs, measure_seeding

FIXTURE = Path.home() / "Workspace/results/partaker2_fixtures/seeding_lee_fc1/late"
TIF = Path.home() / "Workspace/Datasets/external/lee2018/tiff/lee2018_WT_FC1_29.5h.ome.tif"
OUT = Path.home() / "Workspace/results/partaker2_fixtures/seeding_window"


@pytest.fixture(scope="module")
def meta():
    if not (FIXTURE / "meta.json").exists():
        pytest.skip("late fixture not on this machine")
    return json.loads((FIXTURE / "meta.json").read_text())


@pytest.fixture(scope="module")
def measured(meta):
    """The late window measured with the mask tracker's tracks (they carry the cell ids the window needs)."""
    cells = pl.read_parquet(FIXTURE / "cells.parquet")
    tracks = track_rows(cells, position=0, dt_s=meta["dt_s"])
    inp = SeedingInputs(cells=cells, tracks=tracks, dt_s=meta["dt_s"], pixel_um=meta["pixel_um"], start_hour=meta["start_hour"],
                        n_frames=meta["n_frames_recording"], field_shape=tuple(meta["field_shape"]), flow=meta["flow"],
                        frame_range=(meta["frames"]["first"], meta["frames"]["last"]))
    return measure_seeding(inp), tracks


@pytest.fixture(scope="module")
def span(meta):
    first, last = meta["frames"]["first"], meta["frames"]["last"]
    return (meta["start_hour"] + first * meta["dt_s"] / 3600, meta["start_hour"] + last * meta["dt_s"] / 3600)


@pytest.fixture(scope="module")
def film(meta, measured):
    if not TIF.exists():
        pytest.skip("Lee OME-TIFF not on this machine")
    import tifffile, zarr
    from scipy import ndimage as ndi
    from nd2_analyzer.analysis.biofilm_stages.seeding_film import SeedingFilm
    z = zarr.open(tifffile.imread(str(TIF), aszarr=True), mode="r")
    results, tracks = measured
    return SeedingFilm(results, tracks, lambda t: ndi.label(np.asarray(z[int(t), 0]) > 0)[0], tuple(meta["field_shape"]),
                       meta["dt_s"], meta["start_hour"], meta["pixel_um"], n_frames=meta["n_frames_recording"])


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _fig(size):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=size, dpi=100); FigureCanvasAgg(fig); return fig


def _no_overlap_but_the_inset(fig) -> list:
    """collisions, leaving out only the corner map against the picture it sits on (as in the lineage film)."""
    return [c for c in SP.collisions(fig) if not (c.startswith("img '") and c.endswith("img ''"))]


def _outside(fig) -> list:
    """Texts that reach past the figure's edges."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); W, H = fig.bbox.width, fig.bbox.height; out = []
    for ax in fig.axes:
        x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim())
        ticks = [t for pos, t in zip(ax.get_xticks(), ax.get_xticklabels()) if x0 <= pos <= x1]
        ticks += [t for pos, t in zip(ax.get_yticks(), ax.get_yticklabels()) if y0 <= pos <= y1]   # drawn ticks only
        for t in [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label] + list(ax.texts) + ticks:
            if not t.get_visible() or not t.get_text().strip():
                continue
            bb = t.get_window_extent(r)
            if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
                out.append(t.get_text()[:30])
        leg = ax.get_legend()
        if leg is not None:
            bb = leg.get_window_extent(r)
            if bb.x0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
                out.append("legend")
    return out


def test_fate_classes_and_the_irreversible_rule(measured):
    results, _ = measured
    pt = results.per_track
    assert "fate_class" in pt.columns and set(pt["fate_class"].unique().to_list()) <= set(FATE_KEYS)
    S = results.settings
    irr = results.metric("irreversible_begins")
    assert irr.available
    first = pt.filter(pl.col("track_id") == irr.series["first_track"]).row(0, named=True)
    assert first["origin"] == "landed"
    assert first["fate"] == "divided" or (first["fate"] == "present at end" and first["residence_s"] >= S.irreversible_stay_s)
    earlier = pt.filter((pl.col("origin") == "landed") & (pl.col("t_start_h") < first["t_start_h"]))
    assert not earlier.filter(pl.col("fate_class").is_in(["divided", "end"])).height
    left = pt.filter(pl.col("fate_class") == "left")
    assert left.height and (left["residence_s"] <= S.stay_window_s).all()


def test_every_figure_draws_clean_and_greys_without_results(measured, span):
    results, _ = measured
    OUT.mkdir(parents=True, exist_ok=True)
    for f in SF.FIGURES:
        fig = _fig(SF.FIGURE_SIZE_IN)
        ax = SP.draw_panel(fig, f, results, span, None, results.settings)
        assert ax is not None and SP.collisions(fig) == [], (f[1], SP.collisions(fig))
        fig.savefig(OUT / f"{f[1]}.png", dpi=100)
        fig = _fig(SF.FIGURE_SIZE_IN)
        SP.draw_panel(fig, f, None, span, None, None)
        assert SP.collisions(fig) == []
        for width in (4.08, 5.33, 7.5):                    # the canvas widths the page gives at 1200, 1450 and 1900 px
            fig = _fig((width, 4.3))
            SP.draw_panel(fig, f, results, span, None, results.settings)
            assert _outside(fig) == [] and SP.collisions(fig) == [], (f[1], width, _outside(fig), SP.collisions(fig))


def test_the_eight_cards(measured):
    results, _ = measured
    facts = SF.card_facts(results)
    assert len(facts) == 8 and all(len(f) == 4 for f in facts)
    assert all(f[0] and f[1] for f in facts)
    irr = [f for f in facts if f[0].startswith("irreversible attachment")]
    assert any(f[1] == "reversible attachment" for f in facts)
    assert irr and irr[0][2]


def test_the_film_draws_both_views_and_a_click_picks_the_cell(film):
    root = film.default_root()
    assert root is not None and film.fate[root] in ("divided", "end")
    film.set_root(root)
    assert film.cell_frames and film.family[0] == root
    OUT.mkdir(parents=True, exist_ok=True)
    for name, draw, frame in (("cell", film.draw_cell, film.cell_frames[len(film.cell_frames) // 2]),
                              ("cell_first", film.draw_cell, film.cell_frames[0]),
                              ("field", film.draw_field, film.frame_at(film.info[root]["t_start_h"] + 0.2)),
                              ("field_last", film.draw_field, film.field_frames[-1])):
        fig = _fig((10, 4.6))
        ax_img, ax_right = draw(fig, frame)
        assert ax_img is not None and ax_right is not None
        assert _no_overlap_but_the_inset(fig) == [], name
        fig.savefig(OUT / f"window_{name}.png", dpi=100)
    f = film.cell_frames[0] + 1
    while film.pos_at(root, f) is None and f < film.cell_frames[-1]:
        f += 1
    x, y = film.pos_at(root, f)
    assert film.track_at(f, x, y) == root


def test_the_dialog_steps_switches_view_and_follows_a_click(app, film, measured):
    from types import SimpleNamespace
    from nd2_analyzer.ui.biofilms.stages.seeding_page import cells_to_follow
    from nd2_analyzer.ui.biofilms.stages.seeding_window import SeedingDialog
    results, _ = measured
    dlg = SeedingDialog(film, cells_to_follow(results), view="Cell")
    dlg.resize(1100, 760); dlg.show(); app.processEvents()
    assert dlg.slider.maximum() == len(film.cell_frames) - 1 and dlg.cell_combo.count() > 0
    dlg._step(); app.processEvents()
    assert dlg.step == 1 and dlg.time_label.text().startswith("frame ")
    dlg.view_combo.setCurrentText("Field"); app.processEvents()
    assert dlg.view == "Field" and dlg.slider.maximum() == len(film.field_frames) - 1
    frame = film.field_frames[dlg.step]
    other = next((t for (fr, _c), t in film.lab2track.items() if fr == frame and t != film.root), None)
    if other is not None:
        x, y = film.pos_at(other, frame)
        dlg._on_click(SimpleNamespace(inaxes=dlg.ax_img, xdata=x, ydata=y))
        app.processEvents()
        assert film.root == other and dlg.view == "Cell" and dlg.slider.value() == 0
    OUT.mkdir(parents=True, exist_ok=True); dlg.grab().save(str(OUT / "dialog.png"))
    dlg.close()


def test_the_film_saves_as_a_gif(film, tmp_path):
    root = film.default_root(); film.set_root(root)
    film.cell_frames = film.cell_frames[:4]
    out = film.save_gif(str(tmp_path / "cell.gif"), "Cell")
    assert out and Path(out).stat().st_size > 0


def test_the_film_stops_at_the_measured_end_and_covers_short_families(film, measured):
    results, _ = measured
    assert film.last_frame == int(results.counts["frame"].max())
    short = [t for t, r in film.info.items() if r["origin"] == "landed" and 10 <= r["residence_s"] < 120]
    assert short
    film.set_root(short[0])
    assert int(film.tracks[short[0]]["t"][0]) in film.cell_frames and int(film.tracks[short[0]]["t"][-1]) in film.cell_frames


def test_tracks_without_cell_ids_still_colour_and_click(meta, measured):
    import tifffile, zarr
    from scipy import ndimage as ndi
    from nd2_analyzer.analysis.biofilm_stages.seeding_film import SeedingFilm
    if not TIF.exists():
        pytest.skip("Lee OME-TIFF not on this machine")
    results, tracks = measured
    bare = [{k: v for k, v in tr.items() if k != "cell_ids"} for tr in tracks]
    z = zarr.open(tifffile.imread(str(TIF), aszarr=True), mode="r")
    film = SeedingFilm(results, bare, lambda t: ndi.label(np.asarray(z[int(t), 0]) > 0)[0], tuple(meta["field_shape"]),
                       meta["dt_s"], meta["start_hour"], meta["pixel_um"], n_frames=meta["n_frames_recording"])
    root = film.default_root(); film.set_root(root)
    f = next(f for f in film.cell_frames if film.pos_at(root, f) is not None)
    film.labels(f)
    assert any(tid == root for (fr, _c), tid in film.lab2track.items() if fr == f)
    x, y = film.pos_at(root, f)
    assert film.track_at(f, x, y) == root


def test_daughters_that_stay_are_drawn_red_in_the_cell_view(film):
    stay_both = [t for t in film.info if len(film.kids.get(t, [])) == 2
                 and all(SF.daughter_stays(film.info[k]) for k in film.kids[t])]
    if not stay_both:
        pytest.skip("no division with both daughters staying in this window")
    film.set_root(stay_both[0])
    fig = _fig((10, 4.6))
    _img, tr = film.draw_cell(fig, film.cell_frames[-1])
    import matplotlib.colors as mc
    red = mc.to_rgba(SF.LINEAGE_RED)
    reds = [l for l in tr.get_lines() if mc.to_rgba(l.get_color()) == red and len(l.get_xdata()) == 2 and l.get_ydata()[0] != 0.0]
    assert len(reds) == 2


def test_clicking_the_followed_cell_keeps_the_place(app, film, measured):
    from types import SimpleNamespace
    from nd2_analyzer.ui.biofilms.stages.seeding_page import cells_to_follow
    from nd2_analyzer.ui.biofilms.stages.seeding_window import SeedingDialog
    results, _ = measured
    root = film.default_root()
    dlg = SeedingDialog(film, cells_to_follow(results), view="Cell", root=root)
    dlg.show(); app.processEvents()
    frames = dlg.frames()
    k = next(i for i, f in enumerate(frames) if i > 2 and film.pos_at(root, f) is not None)
    dlg.slider.setValue(k); app.processEvents()
    x, y = film.pos_at(root, frames[k])
    dlg._on_click(SimpleNamespace(inaxes=dlg.ax_img, xdata=x, ydata=y)); app.processEvents()
    assert dlg.step == k and film.root == root
    dlg.close()
