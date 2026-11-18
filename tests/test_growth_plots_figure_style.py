"""The Maturation plots are drawn as the panels of Figures 5 and 4 are (make_figure5_s2.py d, make_figure5_s1.py b,
make_figure4_v3.py d), from the tool's own measure: bands I, II, III, Th1 and Th2 named above the plot, a ribbon with
dots, the growth rate under the count on one time axis, the legend under the plot. Limits and label places come from the
data, not from run 2."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from nd2_analyzer.analysis.biofilm_stages import growth_plots as GP
from nd2_analyzer.analysis.biofilm_stages import growth_states as GS
from nd2_analyzer.analysis.biofilm_stages import seeding_plots as SP
from nd2_analyzer.analysis.biofilm_stages.growth_measure import GrowthStates


def grown(positions=(1, 2, 3), last_h=68.0, plateau=2600.0, with_dye=True, th2="fit") -> GrowthStates:
    """A made-up measure with the shape of run 2: an S-shaped count, counted every 1.5 h from 24 h; matrix stain from 33 h
    (31.5 h on the first position), measured to 57 h."""
    hours = np.arange(24.0, last_h + 0.01, 1.5); out = GrowthStates(positions=list(positions), frames={}, hours={}, cells={}, rate={})
    for k, p in enumerate(positions):
        out.frames[p] = list(range(0, 3 * len(hours), 3)); out.hours[p] = hours
        out.cells[p] = np.round(GS.gompertz(hours, plateau * (1 + 0.05 * (k - 1)), 0.05, 35.0))          # Th2 near 100 h, as late as run 2's
        out.rate[p] = GS.growth_rate(hours, out.cells[p])
        if with_dye:
            mh = hours[hours <= 57.0]; start = 31.5 if k == 0 else 33.0; m = np.where(mh >= start, 0.06 * (mh - start + 1.5) ** 2, 0.0)
            out.matrix_hours[p], out.matrix[p], out.single[p] = mh, m, np.full(len(mh), 1.0)
            out.inside[p], out.outside[p], out.colony_area[p] = m * 1.2, m * 0.7, np.linspace(10, 70, len(mh)); out.free_dye[p] = 2000.0
    if with_dye:
        out.th1 = GS.th1_of_positions({p: (out.matrix_hours[p], out.matrix[p]) for p in positions})
    if th2 == "fit":
        out.th2 = GS.predict_th2({p: (out.hours[p], out.cells[p]) for p in positions}, resamples=40)
    elif th2 is not None:
        out.th2 = th2
    return out


def drawn(key, gs, cursor_h=40.0, **kw):
    fig = Figure(figsize=GP.FIGURE_SIZE_IN, dpi=100); FigureCanvasAgg(fig); GP.DRAW[key](fig, gs, cursor_h, None, **kw)
    return fig


def words(fig):
    """What the plot says inside its panels, then what the figure says around them: the title first, the legend last."""
    return [t.get_text() for ax in fig.axes for t in ax.texts] + [t.get_text() for t in fig.texts]


def title_of(fig):
    return fig.texts[0].get_text()


def test_the_picker_names_the_plots_as_the_figures_name_their_panels():
    assert GP.FIGURES == (("cells", "Cell count and growth rate, Th2"), ("matrix", "Microcolonies and matrix stain"),
                          ("inside", "Matrix stain over time"), ("positions", "Number of cells, every position"))
    assert GP.FIGURE_DEFAULTS == ("cells", "matrix") and set(GP.DRAW) == {k for k, _t in GP.FIGURES}


def test_count_and_growth_rate_are_one_plot_with_one_time_axis_as_figure_5_d():
    gs = grown(); gs.th2 = {**gs.th2, "range95_h": (gs.th2["th2_h"] - 9, gs.th2["th2_h"] + 18)}             # a range as wide as run 2's
    fig = drawn("cells", gs); count, rate = fig.axes[:2]; said = words(fig)
    assert count.get_ylabel() == "Cells per frame" and rate.get_ylabel() == "Growth rate\n(cells per h)" and rate.get_xlabel() == "Time (h)" and count.get_xlabel() == ""
    assert count.get_xlim() == rate.get_xlim() and count.get_position().y0 > rate.get_position().y1          # the count above the rate
    assert title_of(fig) == "Cell count and growth rate, Th2" and "Th1" in said and "Th2 (predicted)" in said
    assert [w for w in said if w in ("I", "II", "III")] == ["I", "II", "III"] and "recording\nends" in said
    assert f"fastest growth, {gs.mean_of('rate')[0][int(np.argmax(gs.mean_of('rate')[1]))]:g} h" in said and "10 % of the\nfastest growth" in said
    low, high = gs.th2["range95_h"]
    assert said[-3:] == ["measured, 3 positions", "Gompertz curve fitted to the count", f"Th2 and its 95 % range ({low:.0f} to {high:.0f} h)"]
    ticks = [t for t in count.get_xticks() if count.get_xlim()[0] <= t <= count.get_xlim()[1]]
    assert ticks[:4] == [24, 36, 48, 60] and np.allclose(np.diff(ticks), 12)                                  # every 12 h from the first hour
    assert "1,000" in [t.get_text() for t in count.get_yticklabels()]
    assert SP.collisions(fig) == []


def test_the_count_is_dots_with_the_fitted_curve_and_the_rate_is_joined_dots():
    gs = grown(); fig = drawn("cells", gs); count, rate = fig.axes[:2]; n = len(gs.mean_of("cells")[0])
    dots = [l for l in count.lines if l.get_marker() == "o" and len(l.get_xdata()) == n]
    assert len(dots) == 1 and dots[0].get_linestyle() == "None"                                               # measured: dots, not joined
    styles = sorted(l.get_linestyle() for l in count.lines if len(l.get_xdata()) > n)                         # the fitted curve: solid to the last count, dashed after
    assert styles == ["-", "--"]
    joined = [l for l in rate.lines if l.get_marker() == "o" and len(l.get_xdata()) == n]
    assert len(joined) == 1 and joined[0].get_linestyle() == "-"
    dotted = [l for l in rate.lines if l.get_linestyle() == ":"]
    assert len(dotted) == 1 and dotted[0].get_ydata()[0] == pytest.approx(0.1 * gs.th2["peak_rate"])          # 10 % of the fitted curve's peak
    assert count.collections and rate.collections                                                             # the ribbons (s.d. of the positions)


def test_the_cursor_is_a_red_line_on_both_plots():
    fig = drawn("cells", grown(), cursor_h=40.5)
    for ax in fig.axes[:2]:
        assert [l.get_xdata()[0] for l in ax.lines if l.get_color() == SP.RED] == [40.5]


@pytest.mark.parametrize("case", ["no Th2", "Th2 reached", "range unknown", "range very wide", "no dye", "one position"])
def test_the_plot_draws_whatever_the_recording_gives(case):
    if case == "no Th2":
        gs = grown(th2=None)
    elif case == "Th2 reached":
        gs = grown(last_h=120.0)
    elif case == "range unknown":
        gs = grown(); gs.th2 = {**gs.th2, "range95_h": (float("nan"), float("nan"))}
    elif case == "range very wide":
        gs = grown(); gs.th2 = {**gs.th2, "range95_h": (70.0, 900.0)}
    elif case == "no dye":
        gs = grown(with_dye=False)
    else:
        gs = grown(positions=(2,))
    fig = drawn("cells", gs); said = words(fig); count = fig.axes[0]
    assert SP.collisions(fig) == [] and title_of(fig) == "Cell count and growth rate, Th2"
    if case == "no Th2":
        assert not [w for w in said if w.startswith("Th2")] and "III" not in said and said[-1] == "measured, 3 positions"
    if case == "Th2 reached":
        assert "Th2" in said and "Th2 (predicted)" not in said and "III" in said and "recording\nends" not in said
    if case == "range unknown":
        assert said[-1] == "Th2" and "Th2 (predicted)" in said
    if case == "range very wide":
        assert count.get_xlim()[1] < 24 + 2.6 * (68 - 24)                                                     # the axis does not run to 900 h
    if case == "no dye":
        assert "Th1" not in said and "I" not in said
    if case == "one position":
        assert "measured, position 2" in said and not count.collections


def test_microcolonies_and_matrix_stain_as_figure_5_b():
    gs = grown(); hours = np.arange(24.0, 57.01, 1.5)
    shared = {"cells_in_pct": (hours, np.linspace(64, 98, len(hours)), np.full(len(hours), 5.0))}
    fig = drawn("matrix", gs, shared=shared); ax = fig.axes[0]; said = words(fig)
    assert title_of(fig) == "Microcolonies and matrix stain" and "Th1" in said and [w for w in said if w in ("I", "II")] == ["I", "II"]
    assert ax.get_ylabel() == "Percentage" and ax.get_xlabel() == "Time (h)" and ax.get_ylim() == (0, 100) and list(ax.get_yticks()) == [0, 50, 100]
    assert [t for t in ax.get_xticks() if 24 <= t <= 57][:4] == [24, 30, 36, 42]
    assert said[-2:] == ["cells in microcolonies (% of cells)", "matrix stain (% of frame)"]
    assert sorted(l.get_color() for l in ax.lines if l.get_marker() == "o" and len(l.get_xdata()) > 1) == sorted([GP.MICROCOLONY_ORANGE, GP.MATRIX_GREEN])
    assert SP.collisions(fig) == []
    alone = drawn("matrix", gs); assert words(alone)[-1] == "matrix stain (% of frame)" and "cells in microcolonies (% of cells)" not in words(alone)
    assert SP.collisions(alone) == []


def test_matrix_stain_inside_and_outside_as_figure_4_d():
    fig = drawn("inside", grown()); ax = fig.axes[0]; said = words(fig)
    assert title_of(fig) == "Matrix stain over time" and said[-2:] == ["matrix stain inside microcolonies", "outside"]
    assert ax.get_ylabel() == "Area covered (%)" and ax.get_xlabel() == "Time (h)" and ax.get_ylim() == (0, 100)
    assert len([l for l in ax.lines if l.get_marker() == "o" and len(l.get_xdata()) > 1]) == 2 and SP.collisions(fig) == []


def test_every_position_plot_and_a_measure_without_dye():
    fig = drawn("positions", grown()); assert title_of(fig) == "Number of cells, every position" and fig.axes[0].get_ylabel() == "Cells per frame" and SP.collisions(fig) == []
    for key in ("matrix", "inside"):
        assert GP.NO_DYE in words(drawn(key, grown(with_dye=False)))


# ---------------------------------------------------------------- the page and the tab
def test_the_page_draws_the_cells_in_microcolonies_once_it_is_given_them():
    from nd2_analyzer.ui.biofilms.stages.growth_page import GrowthPage
    page = GrowthPage(); page.set_recording("made-up recording", (24.0, 68.0)); page.set_measured(grown())
    legend = lambda: [t.get_text() for t in page.panels[1].figure.texts]
    assert legend()[0] == "Microcolonies and matrix stain" and "cells in microcolonies (% of cells)" not in legend()
    hours = np.arange(24.0, 57.01, 1.5); page.set_shared({"cells_in_pct": (hours, np.linspace(64, 98, len(hours)), np.full(len(hours), 5.0))})
    assert "cells in microcolonies (% of cells)" in legend()
    page.set_shared(None)
    assert "cells in microcolonies (% of cells)" not in legend()


def test_the_maturation_tab_takes_the_cells_in_microcolonies_from_the_microcolonies_measure():
    from nd2_analyzer.analysis.biofilm_stages import microcolony_metrics as MM
    from nd2_analyzer.ui.biofilms.stages.tab import BiofilmStagesTab
    from test_microcolony_metrics import frame
    metrics = MM.measure_metrics(lambda p, t: frame(with_last_of_b=(p == 0)), {0: [0, 3, 6], 1: [0, 3, 6]}, 0.1, hour_of=lambda t: 24.0 + 0.5 * t)
    tab = BiofilmStagesTab()
    assert tab.growth_page.shared is None
    tab.microcolony_controller.metrics = metrics; tab.microcolony_controller.show_metrics(metrics)
    hours, mean, _sd = tab.growth_page.shared["cells_in_pct"]
    assert hours.tolist() == [24.0, 25.5, 27.0] and mean == pytest.approx(np.full(3, (100 * 9 / 11 + 50) / 2))
    tab.microcolony_controller.on_new_recording(None)
    assert tab.growth_page.shared is None


# ---------------------------------------------------------------- one layout for every plot card
def test_every_plot_of_both_pages_sits_in_the_same_card():
    """Any two plots side by side line up: the same title line, the same box for the axes (so "Time (h)" sits at one
    height), the same band for the legend."""
    from nd2_analyzer.analysis.biofilm_stages import figure_style as FS
    from nd2_analyzer.analysis.biofilm_stages import microcolony_follow as MF, microcolony_metric_plots as XP, microcolony_metrics as MM, microcolony_plots as MP
    from test_microcolony_metrics import frame
    from test_microcolony_rules import PX, blocks
    frames = [blocks(20), blocks(20), blocks(3), blocks(3), blocks(3), blocks(3)]
    tracks = MF.follow_microcolonies(lambda t: frames[t], range(6), pixel_um=PX, dt_h=0.5, start_hour=24.0)
    metrics = MM.measure_metrics(lambda p, t: frame(with_last_of_b=(p == 0)), {0: list(range(0, 21, 3)), 1: list(range(0, 21, 3))}, 0.1,
                                 hour_of=lambda t: 24.0 + 0.5 * t, random_hours=(24.0, 27.0), placements=2)
    figures = [drawn(key, grown()) for key in GP.DRAW]
    for key in MP.DRAW:
        fig = Figure(figsize=GP.FIGURE_SIZE_IN, dpi=100); FigureCanvasAgg(fig); MP.DRAW[key](fig, tracks, 25.0, 1); figures.append(fig)
    for key in XP.DRAW:
        fig = Figure(figsize=GP.FIGURE_SIZE_IN, dpi=100); FigureCanvasAgg(fig); XP.DRAW[key](fig, metrics, 27.0); figures.append(fig)
    for fig in figures:
        name = fig.texts[0].get_text(); boxes = [ax.get_position() for ax in fig.axes if ax.axison and ax.get_position().width > 0.1]
        assert fig.texts[0].get_position()[1] == pytest.approx(FS.TITLE_Y), name
        if not boxes:
            continue                                               # a card that only says why it has no plot
        assert min(b.y0 for b in boxes) == pytest.approx(FS.AXES_BOTTOM) and max(b.y1 for b in boxes) == pytest.approx(FS.AXES_TOP), name
        below = [t.get_position()[1] for t in fig.texts if t.get_position()[1] < FS.AXES_BOTTOM]
        assert all(any(abs(y - (FS.LEGEND_Y - k * FS.LEGEND_ROW)) < 1e-9 for k in range(2)) for y in below), name      # legends only in the legend band
        between = [t.get_position()[1] for t in fig.texts[1:] if t.get_position()[1] > FS.AXES_TOP]
        assert all(any(abs(y - v) < 1e-9 for v in (FS.NOTE_Y, FS.NOTE2_Y)) for y in between), name                      # notes and threshold names on the two note lines
