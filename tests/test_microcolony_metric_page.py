"""The Figure 3 metrics on the Microcolonies page: the plots of the picker, the "All positions" step with its Measure
button, and the wiring. On the made-up frame of test_microcolony_metrics (two positions; position 1 lacks one cell)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nd2_analyzer.analysis.biofilm_stages import microcolony_metric_plots as XP
from nd2_analyzer.analysis.biofilm_stages import microcolony_metrics as MM
from nd2_analyzer.analysis.biofilm_stages import microcolony_plots as MP
from nd2_analyzer.analysis.biofilm_stages import seeding_plots as SP
from test_microcolony_metrics import PX, frame

LABELS = lambda p, t: frame(with_last_of_b=(p == 0))


@pytest.fixture(scope="module")
def measured():
    """24 to 33 h every 1.5 h; random placement at 24 and 27 h."""
    return MM.measure_metrics(LABELS, {0: list(range(0, 21, 3)), 1: list(range(0, 21, 3))}, PX, hour_of=lambda t: 24.0 + 0.5 * t,
                              random_hours=(24.0, 27.0), placements=2)


def figure():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=MP.FIGURE_SIZE_IN, dpi=100); FigureCanvasAgg(fig)
    return fig


def test_the_picker_names_the_metrics_as_figure_3_does():
    assert [title for _key, title in XP.FIGURES] == [
        "Number of microcolonies, all positions", "Cells per microcolony", "Cells in microcolonies, all positions", "Local cell density",
        "Microcolony spacing", "Cell alignment", "Largest microcolony", "Density inside microcolonies", "Microcolony area"]
    assert set(XP.DRAW) == {key for key, _title in XP.FIGURES} and not set(XP.DRAW) & set(MP.DRAW)


@pytest.mark.parametrize("key", [k for k, _title in XP.FIGURES])
def test_every_metric_plot_draws_with_no_element_on_another(measured, key):
    fig = figure(); XP.DRAW[key](fig, measured, cursor_h=27.0)
    assert SP.collisions(fig) == []


def test_the_density_map_draws_with_no_element_on_another(measured):
    fig = figure(); XP.draw_radial(fig, measured, cursor_h=27.0, min_cells=1)           # the made-up frame has 14 cells in its one ring
    assert SP.collisions(fig) == [] and len(fig.axes) == 2 and fig.axes[0].images


def test_a_metric_plot_says_its_number_at_the_measured_hour_nearest_the_cursor(measured):
    fig = figure(); XP.DRAW["m_in"](fig, measured, cursor_h=27.5)                       # measured at 27 and 28.5 h
    words = [t.get_text() for t in fig.texts]
    assert words[0] == "Cells in microcolonies, all positions"
    assert words[1] == "65.9 % at 27 h, mean of 2 positions"                           # (81.8 + 50.0) / 2
    fig = figure(); XP.DRAW["m_per"](fig, measured, cursor_h=24.0)
    assert [t.get_text() for t in fig.texts][1] == "4.7 cells at 24 h, mean of 2 positions"   # the geometric mean of 4.5 and 5
    fig = figure(); XP.DRAW["m_area"](fig, measured, cursor_h=None)
    assert [t.get_text() for t in fig.texts][1] == "mean and s.d. of 2 positions"


def test_every_metric_plot_says_which_microcolonies_it_counts(measured):
    fig = figure(); XP.DRAW["m_count"](fig, measured, cursor_h=24.0)
    assert XP.RULE_NOTE == "groups of 4 or more linked cells in each frame, not followed in time"
    assert XP.RULE_NOTE in [t.get_text() for t in fig.texts]


# ---------------------------------------------------------------- the page
@pytest.fixture()
def page():
    from nd2_analyzer.ui.biofilms.stages.microcolony_page import MicrocolonyPage
    return MicrocolonyPage()


def test_the_page_has_an_all_positions_step_with_a_measure_button(page):
    assert "All positions:" in page.step_labels["measure"].text()
    assert page.measure_button.text() == "Measure" and not page.measure_button.isEnabled() and not page.every_spin.isEnabled()
    page.set_every(0.5, 1.5); page.set_measure_enabled(True)
    assert page.measure_button.isEnabled() and page.every_spin.value() == 1.5 and page.every_spin.minimum() == 0.5
    asked = []; page.measure_requested.connect(lambda: asked.append(True)); page.measure_button.click()
    assert asked == [True]


def test_the_picker_lists_the_metrics_under_the_plots_of_the_followed_microcolonies(page):
    combo = page.plot_combos[0]
    titles = [combo.itemText(i) for i in range(combo.count())]
    assert titles[:5] == [title for _key, title in MP.FIGURES]
    assert titles[5] == "" and titles[6:] == [title for _key, title in XP.FIGURES]      # a separator line between the two


def test_a_metric_plot_asks_for_the_measure_until_it_is_there(page, measured):
    combo = page.plot_combos[0]; combo.setCurrentIndex(combo.findData("m_area"))
    assert [t.get_text() for ax in page.panels[0].figure.axes for t in ax.texts] == ["Click Measure (all positions) to see this plot"]
    page.set_metrics(measured)
    assert [t.get_text() for t in page.panels[0].figure.texts][0] == "Microcolony area"
    assert [t.get_text() for ax in page.panels[1].figure.axes for t in ax.texts] == ["Track the microcolonies to see this plot"]   # the other plot still needs Track
    page.set_metrics(None)
    assert [t.get_text() for ax in page.panels[0].figure.axes for t in ax.texts] == ["Click Measure (all positions) to see this plot"]


# ---------------------------------------------------------------- the wiring
def test_the_measure_takes_every_position_at_the_chosen_step_up_to_the_chosen_frame():
    from nd2_analyzer.ui.biofilms.stages.microcolony_controller import frames_for_metrics, random_hours
    segmented = {1: list(range(89)), 2: list(range(89)), 5: [0], 7: []}
    blurred = lambda t: 69 <= t <= 76
    out = frames_for_metrics(segmented, step=3, last=66, skip=blurred)
    assert sorted(out) == [1, 2] and out[1] == list(range(0, 67, 3))                    # a position with one frame gives no curve
    assert frames_for_metrics({1: list(range(89))}, step=3, last=88, skip=blurred)[1][-5:] == [66, 78, 81, 84, 87]
    assert frames_for_metrics({1: list(range(4, 20))}, step=5, last=88)[1] == [4, 9, 14, 19]      # counted from the first segmented frame
    assert random_hours(24.0) == (24.0, 27.0, 30.0, 33.0)                               # the first nine hours, every three


def test_the_worker_measures_and_hands_the_metrics_over():
    from nd2_analyzer.ui.biofilms.stages.microcolony_controller import MetricsWorker
    worker = MetricsWorker(LABELS, {0: [0, 3], 1: [0, 3]}, PX, lambda t: 24.0 + 0.5 * t, random_at=())
    got, said = [], []
    worker.done.connect(got.append); worker.progress.connect(lambda pct, words: said.append(words))
    worker.run()
    assert len(got) == 1 and got[0].positions == [0, 1] and got[0].values["microcolonies"][0].tolist() == [2, 2]
    assert said[-1] == "100 %"


def test_a_failed_measure_says_why():
    from nd2_analyzer.ui.biofilms.stages.microcolony_controller import MetricsWorker

    def broken(p, t):
        raise OSError("mask file gone")

    worker = MetricsWorker(broken, {0: [0, 3]}, PX, lambda t: 24.0 + 0.5 * t, random_at=())
    why = []; worker.failed.connect(why.append); worker.run()
    assert why == ["OSError: mask file gone"]
