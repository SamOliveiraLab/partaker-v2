"""Every gallery panel of the Seeding page drawn with the Lee 2018 FC1 fixtures (late: families and divisions; early:
landings and leavers), with results None, and with frames 1800 s apart so most rules are unavailable; the report pages
the same way. Every figure is free of overlapping elements and nothing is clipped at the figure edge. PNGs go under
~/Workspace/results/partaker2_fixtures/seeding_panels/ to be looked at. Skips when the fixtures are not on this machine.
Run from src: cd src && python -m pytest ../tests/test_seeding_panels.py"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import polars as pl
import pytest

from nd2_analyzer.analysis.biofilm_stages import seeding_plots as SP
from nd2_analyzer.analysis.biofilm_stages.seeding_rules import SeedingInputs, measure_seeding
from nd2_analyzer.analysis.biofilm_stages.seeding_types import SeedingSettings

FIXTURES = Path.home() / "Workspace/results/partaker2_fixtures/seeding_lee_fc1"
OUT = Path.home() / "Workspace/results/partaker2_fixtures/seeding_panels"
SLOW_DT_S = 1800.0


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


def window_span(inp: SeedingInputs) -> tuple:
    """The hours the fixture window covers, as the page's span would when the recording is just these frames."""
    first, last = inp.frame_range
    return (inp.start_hour + first * inp.dt_s / 3600.0, inp.start_hour + (last + 1) * inp.dt_s / 3600.0)


@pytest.fixture(scope="module")
def late():
    inp = load_window("late")
    return measure_seeding(inp), window_span(inp)


@pytest.fixture(scope="module")
def early():
    inp = load_window("early")
    return measure_seeding(inp), window_span(inp)


@pytest.fixture(scope="module")
def slow():
    inp = load_window("early", dt_s=SLOW_DT_S)
    return measure_seeding(inp), window_span(inp)


def _fig(size=SP.PANEL_SIZE_IN):
    return SP.new_figure(size, dpi=100)


def _save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=100)


def clipped(fig) -> list:
    """Texts, tick labels and legends that reach past the figure edge."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); W, H = fig.bbox.width, fig.bbox.height; out = []

    def check(name, art):
        if not art.get_visible() or (hasattr(art, "get_text") and not art.get_text().strip()):
            return
        bb = art.get_window_extent(r)
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
            out.append(name)

    for t in fig.texts:
        check(f"text '{t.get_text()[:30]}'", t)
    for ax in fig.axes:
        for t in ax.texts:
            check(f"call-out '{t.get_text()[:30]}'", t)
        for t in (ax.title, ax._left_title, ax.xaxis.label, ax.yaxis.label):
            check(f"label '{t.get_text()[:30]}'", t)
        if ax.get_legend():
            check("legend", ax.get_legend())
        if ax.axison:
            x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim())
            for pos, t in zip(ax.get_xticks(), ax.get_xticklabels()):
                if x0 <= pos <= x1: check(f"tick '{t.get_text()}'", t)
            for pos, t in zip(ax.get_yticks(), ax.get_yticklabels()):
                if y0 <= pos <= y1: check(f"tick '{t.get_text()}'", t)
    return out


def grey_reason(ax) -> str:
    """The grey sentence of an unavailable panel, or '' when the panel drew a plot."""
    if ax.axison:
        return ""
    return " ".join(t.get_text().replace("\n", " ") for t in ax.texts)


# ---------------------------------------------------------------------------------------------- the list itself
def test_panels_list_is_complete_and_in_the_addendum_order():
    assert len(SP.PANELS) == 17
    groups = [p[0] for p in SP.PANELS]
    assert groups == ["arrivals"] * 5 + ["commitment"] * 2 + ["movement"] * 3 + ["division"] * 4 + ["flow"] * 3
    assert all(p[4] == (p[0] == "flow") for p in SP.PANELS)
    keys = [p[1] for p in SP.PANELS]
    assert len(set(keys)) == 17 and SP.LANDING_DOTS_KEY in keys
    for group, key, title, fn, needs_flow in SP.PANELS:
        assert title == title[0].upper() + title[1:] and callable(fn) and fn.__name__.startswith("draw_")
    from nd2_analyzer.analysis.biofilm_stages import seeding_figures as SF
    assert set(SP.TIME_PANEL_KEYS) <= set(keys) | {f[1] for f in SF.FIGURES}
    assert dict(SP.GROUP_TITLES)["commitment"] == "Irreversible attachment"


def test_module_has_no_dashes_and_no_orange():
    text = Path(SP.__file__).read_text()
    assert "—" not in text and "–" not in text
    assert "orange" not in text.lower() and "#e69f00" not in text.lower() and "#d55e00" not in text.lower()


# ---------------------------------------------------------------------------------------------- every panel, late fixture
def test_every_panel_draws_the_late_fixture_without_collisions(late):
    results, span = late
    drawn, grey = [], []
    for panel in SP.PANELS:
        fig = _fig()
        ax = SP.draw_panel(fig, panel, results, span, cursor_h=span[0] + 0.6 * (span[1] - span[0]), settings=results.settings)
        assert ax is not None
        assert SP.collisions(fig) == [], panel[1]
        assert clipped(fig) == [], panel[1]
        _save(fig, f"late_{panel[1]}")
        (grey if grey_reason(ax) else drawn).append(panel[1])
    # the late window has tracks, divisions and flow: almost everything draws; the lag and take-off reasons are honest
    assert "landing_dots" in drawn and "irreversible_begins" in drawn and "division_outcome" in drawn
    assert "along_flow_shift" in drawn and "dividers_vs_holders" in drawn
    assert len(drawn) >= 15, (drawn, grey)
    fig = _fig(); ax = SP.draw_panel(fig, SP.PANELS[0], results, span, None, results.settings)
    assert ax.axison and SP.collisions(fig) == []


def test_every_panel_draws_the_early_fixture_without_collisions(early):
    results, span = early
    for panel in SP.PANELS:
        fig = _fig()
        ax = SP.draw_panel(fig, panel, results, span, cursor_h=span[0] + 0.3 * (span[1] - span[0]), settings=results.settings)
        assert SP.collisions(fig) == [], panel[1]
        assert clipped(fig) == [], panel[1]
        _save(fig, f"early_{panel[1]}")


def test_time_panels_show_the_cursor_only_inside_the_span(late):
    results, span = late
    a, b = span
    for panel in SP.PANELS:
        if panel[1] not in SP.TIME_PANEL_KEYS:
            continue
        fig = _fig(); ax = SP.draw_panel(fig, panel, results, span, (a + b) / 2, results.settings)
        cursor = [l for l in ax.lines if l.get_linewidth() == 1.0 and l.get_color() == SP.GREY
                  and np.allclose(l.get_xdata(), (a + b) / 2)]
        assert len(cursor) == 1, panel[1]
        fig = _fig(); ax = SP.draw_panel(fig, panel, results, span, b + 5.0, results.settings)
        assert not [l for l in ax.lines if l.get_color() == SP.GREY and np.allclose(l.get_xdata(), b + 5.0)], panel[1]


def test_panels_with_the_full_recording_span(late):
    """The page passes the recording's span (0 to 29.5 h) while the data sits in its last two hours."""
    results, _ = late
    span = (0.0, 29.5)
    for panel in SP.PANELS:
        if panel[1] not in SP.TIME_PANEL_KEYS:
            continue
        fig = _fig(); SP.draw_panel(fig, panel, results, span, 12.0, results.settings)
        assert SP.collisions(fig) == [], panel[1]
        assert clipped(fig) == [], panel[1]
        _save(fig, f"fullspan_{panel[1]}")


# ---------------------------------------------------------------------------------------------- the empty states
def test_every_panel_with_results_none_shows_run_measure():
    for panel in SP.PANELS:
        fig = _fig(); ax = SP.draw_panel(fig, panel, None, (0.0, 29.5), 3.0, None)
        assert grey_reason(ax) == SP.RUN_MEASURE and ax.get_title(loc="left") == panel[2]
        assert SP.collisions(fig) == [] and clipped(fig) == [], panel[1]
        _save(fig, f"none_{panel[1]}")
    for panel in SP.PANELS:                                   # the functions themselves say the same without results
        fig = _fig(); ax = panel[3](fig, None, (0.0, 29.5), 3.0, SeedingSettings())
        assert grey_reason(ax) == SP.RUN_MEASURE


def test_slow_frames_make_the_fine_rules_unavailable_with_their_reason(slow):
    results, span = slow
    reasons = {}
    for panel in SP.PANELS:
        fig = _fig(); ax = SP.draw_panel(fig, panel, results, span, span[0], results.settings)
        assert SP.collisions(fig) == [] and clipped(fig) == [], panel[1]
        _save(fig, f"slow_{panel[1]}")
        reasons[panel[1]] = grey_reason(ax)
    assert reasons["stay_by_landing_hour"] == "frames 1800 s apart, the rule needs under 10 s"
    assert reasons["dividers_vs_holders"] == "frames 1800 s apart, the rule needs under 120 s"
    for key, reason in reasons.items():
        m = results.metric(key)
        if m is not None and not m.available:
            assert reason == m.reason, key


def test_no_flow_shows_the_flow_reason():
    inp = load_window("early", flow=None)
    results = measure_seeding(inp); span = window_span(inp)
    for panel in SP.PANELS:
        if not panel[4]:
            continue
        fig = _fig(); ax = SP.draw_panel(fig, panel, results, span, span[0], results.settings)
        assert grey_reason(ax) == "no flow in this recording" and SP.collisions(fig) == []


def test_a_long_reason_wraps_inside_the_panel(late):
    results, span = late
    fig = _fig(); ax = SP.draw_panel(fig, SP.PANELS[0], None, span, None, None)
    reason = results.metric("lag_time").reason
    fig = _fig(); ax = SP.grey_panel(fig, "Cells on the surface", reason)
    assert "\n" in ax.texts[0].get_text() and clipped(fig) == [] and SP.collisions(fig) == []
    _save(fig, "grey_long_reason")


# ---------------------------------------------------------------------------------------------- the landing dots
def test_pick_landing_finds_the_dot_under_the_pointer_and_rings_it(late):
    results, span = late
    panel = [p for p in SP.PANELS if p[1] == SP.LANDING_DOTS_KEY][0]
    fig = _fig(); ax = SP.draw_panel(fig, panel, results, span, None, results.settings)
    ids, x, stayed, word, censored = SP.landing_points(results, results.settings)
    k = int(np.argmax(stayed))                                            # the longest stayer sits alone at the top
    lo, hi = SP.dot_axis_range(stayed, 3.0)
    px, py = ax.transData.transform((x[k], min(stayed[k], hi)))
    assert SP.pick_landing(ax, results, px + 2, py - 2, tol_px=8) == int(ids[k])
    assert SP.pick_landing(ax, results, px + 40, py + 40, tol_px=8) is None
    assert SP.pick_landing(ax, None, px, py) is None
    fig = _fig(); ax = SP.draw_panel(fig, panel, results, span, None, results.settings, selected=int(ids[k]))
    rings = [l for l in ax.lines if l.get_markeredgecolor() == SP.INK and l.get_markerfacecolor() == "none"]
    assert len(rings) == 1 and np.allclose(rings[0].get_xdata(), [x[k]])
    assert SP.collisions(fig) == [] and clipped(fig) == []
    _save(fig, "late_landing_dots_selected")
    legend_words = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_words == ["left", "stayed", "divided"]
    assert set(word.tolist()) == {"leave", "stay", "divides"} and censored.sum() == (results.per_track.filter(
        (pl.col("origin") == "landed") & (pl.col("fate") == "present at end")).height)


# ---------------------------------------------------------------------------------------------- the report
@pytest.mark.parametrize("flow", [True, False])
def test_report_pages_one_per_group_plus_the_numbers(late, flow):
    results, span = late
    figs = SP.draw_report_pages(results, span, has_flow=flow, title="lee2018_WT_FC1", settings=results.settings)
    assert len(figs) == (6 if flow else 5)
    for k, fig in enumerate(figs):
        assert tuple(fig.get_size_inches()) == SP.REPORT_PAGE_IN
        assert SP.collisions(fig) == [], k
        assert clipped(fig) == [], k
        _save(fig, f"report_{'flow' if flow else 'no_flow'}_page{k + 1}")
    headings = [t.get_text() for fig in figs for t in fig.texts]
    assert "Irreversible attachment" in headings and ("Flow" in headings) == flow and "Numbers" in headings
    table = figs[-1].axes[0]
    words = [t.get_text() for t in table.texts]
    assert "Landings" in words and "3,029 landings" in words and ("Shift along the flow" in words) == flow
    assert any(t.startswith("the fitted lag") for t in words)             # a long reason is wrapped, not cut


def test_draw_report_still_draws_page_one(late):
    results, span = late
    fig = _fig(SP.report_size(results, True)); SP.draw_report(fig, results, span, has_flow=True, title="lee2018_WT_FC1")
    assert SP.collisions(fig) == [] and clipped(fig) == []
    assert [t.get_text() for t in fig.texts] == ["lee2018_WT_FC1", "Reversible attachment: arrivals and leaving"]
    fig = _fig(SP.report_size(None)); SP.draw_report(fig, None, span)
    assert SP.collisions(fig) == []
    figs = SP.draw_report_pages(None, span, has_flow=False)
    assert len(figs) == 5 and all(SP.collisions(f) == [] for f in figs)


def test_numbers_rows_follow_the_groups(late):
    results, _ = late
    rows = SP.numbers_rows(results, has_flow=True)
    heads = [r[1] for r in rows if r[0] == "head"]
    assert heads == [h for _, h in SP.GROUP_TITLES]
    metric_rows = [r for r in rows if r[0] == "metric"]
    assert len(metric_rows) == len(results.metrics)
    lag = [r for r in metric_rows if r[1] == "Lag before growth"][0]
    assert not lag[4] and lag[2].startswith("the fitted lag") and lag[3] == ""
    assert len(SP.numbers_rows(results, has_flow=False)) == len(rows) - 5
