"""The plots inside the two Follow windows carry the figures' look too: the figures' axis names in bold, "Time (h)" with
round ticks from the first frame, and the red cursor of the page."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from nd2_analyzer.analysis.biofilm_stages import figure_style as FS
from nd2_analyzer.analysis.biofilm_stages import growth_measure as GM
from nd2_analyzer.analysis.biofilm_stages import matrix_stain as MX
from nd2_analyzer.analysis.biofilm_stages import microcolony_follow as MF
from nd2_analyzer.analysis.biofilm_stages import seeding_plots as SP
from test_growth_measure import PX as GROWTH_PX, recording
from test_microcolony_rules import PX, blocks


def red_cursor(ax):
    return [float(l.get_xdata()[0]) for l in ax.lines if l.get_color() == FS.CURSOR_RED]


def test_the_maturation_window_plots_carry_the_figures_axis_names():
    from nd2_analyzer.analysis.biofilm_stages.growth_film import FIGURE_IN, GrowthFilm
    labels, dye = recording(); ref = MX.DyeReference(zero=np.zeros((240, 240), np.float32), flat=np.ones((240, 240), np.float32))
    measured = GM.measure_growth_states(labels, dye, {0: list(range(8)), 1: list(range(8))}, pixel_um=GROWTH_PX, hour_of=lambda t: 24.0 + 1.5 * t, reference=ref, resamples=20)
    film = GrowthFilm(measured, labels, lambda p, t: 1000.0 - 300.0 * (labels(p, t) > 0), dye, ref, GROWTH_PX); film.set_position(0)
    fig = Figure(figsize=FIGURE_IN, dpi=100); FigureCanvasAgg(fig); film.draw(fig, 5); _image, count, matrix = fig.axes[:3]
    assert count.get_ylabel() == "Cells per frame" and matrix.get_ylabel() == "Matrix stain\n(% of frame)" and matrix.get_xlabel() == "Time (h)"
    assert matrix.xaxis.label.get_fontweight() == "bold" and red_cursor(count) == [31.5] and red_cursor(matrix) == [31.5]
    assert list(matrix.get_xticks())[:3] == [24.0, 26.0, 28.0] and SP.collisions(fig) == []


def test_the_microcolonies_window_plots_carry_the_figures_axis_names():
    from nd2_analyzer.analysis.biofilm_stages.microcolony_film import FIGURE_IN, MicrocolonyFilm
    frames = [blocks(20), blocks(20), blocks(3), blocks(3), blocks(3), blocks(3)]
    tracks = MF.follow_microcolonies(lambda t: frames[t], range(6), pixel_um=PX, dt_h=0.5, start_hour=24.0)
    film = MicrocolonyFilm(tracks, lambda t: frames[t], PX); film.set_root(1)
    fig = Figure(figsize=FIGURE_IN, dpi=100); FigureCanvasAgg(fig); film.draw(fig, 3); _image, history, lower = fig.axes[:3]
    assert lower.get_ylabel() == "Cells" and lower.get_xlabel() == "Time (h)" and lower.xaxis.label.get_fontweight() == "bold"
    assert red_cursor(history) == [25.5] and red_cursor(lower) == [25.5] and SP.collisions(fig) == []

    def stain(t):
        m = np.zeros((120, 200), bool)
        if t >= 3:
            m[35:68, 15:68] = True
        return m

    film.set_stain(stain, show=True); film.draw(fig, 3)
    assert fig.axes[2].get_ylabel() == "Area covered (%)" and SP.collisions(fig) == []          # the name Figure 4 d gives the stain cover


# ---------------------------------------------------------------- after the coordinator's comparison with the figures (18 Nov, 06:00)
def toy_film():
    from nd2_analyzer.analysis.biofilm_stages.microcolony_film import MicrocolonyFilm
    frames = [blocks(20), blocks(20), blocks(3), blocks(3), blocks(3), blocks(3)]
    tracks = MF.follow_microcolonies(lambda t: frames[t], range(6), pixel_um=PX, dt_h=0.5, start_hour=24.0)
    film = MicrocolonyFilm(tracks, lambda t: frames[t], PX); film.set_root(1)
    return film


def test_the_heading_counts_the_microcolonies_that_are_drawn():
    from nd2_analyzer.analysis.biofilm_stages import microcolony_film as F
    assert F.heading_words(5, drawn=6, later=2) == "Microcolony 5 and the 6 that merge into it (2 more later)"
    assert F.heading_words(1, drawn=1, later=0) == "Microcolony 1 and the 1 that merges into it"
    assert F.heading_words(84, drawn=0, later=0) == "Microcolony 84" and F.heading_words(84, drawn=0, later=3) == "Microcolony 84 (3 merge into it later)"
    film = toy_film(); fig = Figure(figsize=F.FIGURE_IN, dpi=100); FigureCanvasAgg(fig); film.draw(fig, 3)
    said = [t.get_text() for t in fig.texts]
    assert said[0] == "Microcolony 1 and the 1 that merges into it   frame 3   25.5 h" and "Merging history (2 microcolonies)" in said


def test_nothing_sits_on_the_film_but_a_white_scale_bar():
    from nd2_analyzer.analysis.biofilm_stages import microcolony_film as F
    film = toy_film(); fig = Figure(figsize=F.FIGURE_IN, dpi=100); FigureCanvasAgg(fig); film.draw(fig, 3); image = fig.axes[0]
    assert image.child_axes == [] and not image.patches                                  # no field map and no black box over the cells
    assert [t.get_text() for t in image.texts if t.get_text().endswith("µm")]            # the scale bar's words stay, in white
    field = fig.axes[3]                                                                  # the field of view is a small picture of its own, beside the film
    assert field.get_position().x0 >= image.get_position().x1 and field.get_title() == "field of view" and SP.collisions(fig) == []


def test_the_maturation_window_plots_have_dots_states_and_th1():
    from nd2_analyzer.analysis.biofilm_stages.growth_film import FIGURE_IN, GrowthFilm
    labels, dye = recording(); ref = MX.DyeReference(zero=np.zeros((240, 240), np.float32), flat=np.ones((240, 240), np.float32))
    measured = GM.measure_growth_states(labels, dye, {0: list(range(8)), 1: list(range(8))}, pixel_um=GROWTH_PX, hour_of=lambda t: 24.0 + 1.5 * t, reference=ref, resamples=20)
    film = GrowthFilm(measured, labels, lambda p, t: 1000.0 - 300.0 * (labels(p, t) > 0), dye, ref, GROWTH_PX); film.set_position(0)
    fig = Figure(figsize=FIGURE_IN, dpi=100); FigureCanvasAgg(fig); film.draw(fig, 7); _image, count, matrix = fig.axes[:3]
    said = [t.get_text() for t in fig.texts]
    assert said[0] == "Position 0   frame 7   34.5 h   state II" and "Number of cells" in said and "Th1" in said
    for ax in (count, matrix):
        assert [l for l in ax.lines if l.get_marker() == "o" and len(l.get_xdata()) > 2]                 # dots on the line, as on the page
    assert [t.get_text() for t in count.texts] == ["I", "II"] and SP.collisions(fig) == []
