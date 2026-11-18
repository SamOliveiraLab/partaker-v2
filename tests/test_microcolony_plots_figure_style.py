"""The Microcolonies plots are drawn as the panels of Figure 3 are (make_figure3.py, make_figure3_metrics.py): the
figure's axis names in bold, "Time (h)" with round ticks from the first frame, an ink line with dots on a grey ribbon,
the legend under the plot, and the page's red cursor."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from nd2_analyzer.analysis.biofilm_stages import figure_style as FS
from nd2_analyzer.analysis.biofilm_stages import microcolony_follow as MF
from nd2_analyzer.analysis.biofilm_stages import microcolony_metric_plots as XP
from nd2_analyzer.analysis.biofilm_stages import microcolony_metrics as MM
from nd2_analyzer.analysis.biofilm_stages import microcolony_plots as MP
from nd2_analyzer.analysis.biofilm_stages import seeding_plots as SP
from test_microcolony_metrics import PX as METRIC_PX, frame
from test_microcolony_rules import PX, blocks

FRAMES = [blocks(20), blocks(20), blocks(3), blocks(3), blocks(3), blocks(3)]


@pytest.fixture(scope="module")
def tracks():
    return MF.follow_microcolonies(lambda t: FRAMES[t], range(6), pixel_um=PX, dt_h=0.5, start_hour=24.0)


@pytest.fixture(scope="module")
def measured():
    return MM.measure_metrics(lambda p, t: frame(with_last_of_b=(p == 0)), {0: list(range(0, 21, 3)), 1: list(range(0, 21, 3))}, METRIC_PX,
                              hour_of=lambda t: 24.0 + 0.5 * t, random_hours=(24.0, 27.0), placements=2)


def figure():
    fig = Figure(figsize=MP.FIGURE_SIZE_IN, dpi=100); FigureCanvasAgg(fig)
    return fig


def red_cursor(ax):
    return [float(l.get_xdata()[0]) for l in ax.lines if l.get_color() == FS.CURSOR_RED]


@pytest.mark.parametrize("key, x_name, y_name", [("count", "Time (h)", "Microcolonies"), ("cells", "Time (h)", "Cells (%)"), ("events", "Time (h)", "Merges per 3 h"),
                                                 ("history", "Time (h)", "")])
def test_the_plots_of_the_followed_microcolonies_carry_the_figures_axis_names(tracks, key, x_name, y_name):
    fig = figure(); MP.DRAW[key](fig, tracks, cursor_h=25.0, selected=1); ax = fig.axes[0]
    assert ax.get_xlabel() == x_name and ax.get_ylabel() == y_name and ax.xaxis.label.get_fontweight() == "bold"
    assert red_cursor(ax) == [25.0] and SP.collisions(fig) == []


def test_the_time_before_merging_carries_the_figures_axis_names():
    from types import SimpleNamespace
    waited = SimpleNamespace(hours_before_merging=lambda: [1.0, 3.5, 3.5, 6.0])           # four microcolonies that formed and then merged
    fig = figure(); MP.draw_waits(fig, waited); ax = fig.axes[0]
    assert ax.get_xlabel() == "Time before merging (h)" and ax.get_ylabel() == "Microcolonies" and SP.collisions(fig) == []


NAMES = {"m_count": "Microcolonies", "m_per": "Cells", "m_in": "Cells (%)", "m_density": "Cells per 100 µm²", "m_spacing": "Nearest neighbour (µm)",
         "m_alignment": "Alignment with neighbours", "m_largest": "Share of microcolony cells (%)", "m_area": "Surface covered (%)"}


@pytest.mark.parametrize("key", list(NAMES))
def test_the_metric_plots_carry_the_figures_axis_names(measured, key):
    fig = figure(); XP.DRAW[key](fig, measured, cursor_h=27.0); ax = fig.axes[0]
    assert ax.get_ylabel() == NAMES[key] and ax.get_xlabel() == "Time (h)"
    assert red_cursor(ax) == ([27.0] if key != "m_spacing" else [])      # the spacing stops before 27 h here: two positions with one or two microcolonies
    main = [l for l in ax.lines if l.get_marker() == "o" and len(l.get_xdata()) > 2]
    assert main and (main[-1].get_color() in (FS.INK, FS.MICROCOLONY_ORANGE))                 # ink (orange for cells in microcolonies), not the old blue
    assert SP.collisions(fig) == []


def test_the_legends_sit_under_the_plot(measured):
    for key, wanted in (("m_in", ["measured", "random placement"]), ("m_density", ["in microcolonies", "single cells"])):
        fig = figure(); XP.DRAW[key](fig, measured, cursor_h=27.0); legend = fig.texts[-2:]
        assert [t.get_text() for t in legend] == wanted
        assert all(t.get_position()[1] < fig.axes[0].get_position().y0 for t in legend)


def test_the_density_map_carries_the_figures_names(measured):
    fig = figure(); XP.draw_radial(fig, measured, cursor_h=27.0, min_cells=1)
    assert fig.axes[0].get_ylabel() == "Distance from centre (µm)" and fig.axes[0].get_xlabel() == "Time (h)" and fig.axes[1].get_ylabel() == "Cells per 100 µm²"
    assert SP.collisions(fig) == []


# ---------------------------------------------------------------- after the coordinator's comparison with the figures (18 Nov, 06:00)
def made_up(microcolonies, per_microcolony=None, spacing=None):
    """A measure of two positions at four hours, built by hand."""
    import numpy as np
    hours = np.array([24.0, 25.5, 27.0, 28.5]); n = len(hours)
    values = {name: {p: np.full(n, 10.0) for p in (0, 1)} for name in MM.SERIES}
    values["microcolonies"] = {p: np.array(v, float) for p, v in microcolonies.items()}
    if per_microcolony is not None:
        values["cells_per_microcolony"] = {p: np.array(per_microcolony, float) for p in (0, 1)}
    if spacing is not None:
        values["spacing_um"] = {p: np.array(spacing, float) for p in (0, 1)}
    rings = len(np.arange(0, 42, 2.0)) - 1
    return MM.MicrocolonyMetrics(positions=[0, 1], frames={0: [0, 3, 6, 9], 1: [0, 3, 6, 9]}, hours={0: hours, 1: hours}, values=values,
                                 ring_sum={p: np.full((n, rings), 600.0) for p in (0, 1)}, ring_n={p: np.full((n, rings), 30, int) for p in (0, 1)})


def test_the_spacing_stops_where_a_position_has_too_few_microcolonies_as_figure_3_i():
    """Figure 3 i stops at 39 h, "while several microcolonies exist": on run 2 that is the last hour at which every
    position has 5 or more. A nearest neighbour among two or three large microcolonies is not a spacing."""
    m = made_up({0: [6, 6, 6, 6], 1: [6, 5, 4, 6]})
    assert XP.spacing_until(m) == 25.5                                               # 27 h is the first hour with a position under 5; it does not come back
    fig = figure(); XP.draw_spacing(fig, m, cursor_h=28.5); ax = fig.axes[0]
    line = [l for l in ax.lines if l.get_marker() == "o" and len(l.get_xdata()) > 1][0]
    assert list(line.get_xdata()) == [24.0, 25.5] and ax.get_xlim()[1] < 27.0
    said = [t.get_text() for t in fig.texts]
    assert said[1] == "shown to 25.5 h: after it a position has fewer than 5 microcolonies" and SP.collisions(fig) == []
    assert XP.spacing_until(made_up({0: [6, 6, 6, 6], 1: [6, 6, 6, 6]})) == 28.5


def test_cells_per_microcolony_runs_over_the_decades_the_data_needs_as_figure_3_f():
    m = made_up({0: [6] * 4, 1: [6] * 4}, per_microcolony=[14, 25, 330, 400])
    fig = figure(); XP.draw_per(fig, m, cursor_h=24.0); ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["10", "100", "1,000"] and 3 <= ax.get_ylim()[0] <= 10 and 1000 <= ax.get_ylim()[1] <= 3000


def test_the_density_map_has_round_colour_ticks():
    m = made_up({0: [6] * 4, 1: [6] * 4}); m.ring_sum[0][:, 3] = 240.0; m.ring_sum[1][:, 3] = 240.0; m.ring_sum[0][:, 5] = 870.0; m.ring_sum[1][:, 5] = 870.0   # densities 8, 20, 29
    fig = figure(); XP.draw_radial(fig, m, cursor_h=24.0); bar = fig.axes[1]
    assert list(bar.get_yticks()) == [10.0, 20.0, 30.0] and bar.get_ylim() == (5.0, 30.0)
