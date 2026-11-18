"""The film strips of the Maturation and Microcolonies pages are rows of real frames, as Figure 5 a and Figure 3 d are:
tiles laid in a grid that uses the height the page gives, on Maturation with the microcolony outlines, the matrix stain
and the number of microcolonies under each frame, and a line where the stain was not measured."""
import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from nd2_analyzer.analysis.biofilm_stages import figure_style as FS
from nd2_analyzer.analysis.biofilm_stages import growth_film as GF
from nd2_analyzer.analysis.biofilm_stages import growth_measure as GM
from nd2_analyzer.analysis.biofilm_stages import matrix_stain as MX
from nd2_analyzer.analysis.biofilm_stages import microcolony_film as F
from nd2_analyzer.analysis.biofilm_stages import microcolony_follow as MF
from nd2_analyzer.analysis.biofilm_stages import seeding_plots as SP
from test_growth_measure import PX as GROWTH_PX, recording
from test_microcolony_rules import PX, blocks


def figure(width, height):
    fig = Figure(figsize=(width, height), dpi=100); FigureCanvasAgg(fig)
    return fig


def side_in(fig, ax):
    box = ax.get_position(); return box.width * fig.get_figwidth(), box.height * fig.get_figheight()


def growth_film(matrix_frames=None):
    labels, dye = recording(); ref = MX.DyeReference(zero=np.zeros((240, 240), np.float32), flat=np.ones((240, 240), np.float32))
    measured = GM.measure_growth_states(labels, dye, {0: list(range(8))}, pixel_um=GROWTH_PX, hour_of=lambda t: 24.0 + 1.5 * t, reference=ref,
                                        matrix_frames=matrix_frames, resamples=20)
    return GF.GrowthFilm(measured, labels, lambda p, t: 1000.0 - 300.0 * (labels(p, t) > 0), dye, ref, GROWTH_PX)


def test_tiles_are_laid_in_a_grid_that_uses_the_height():
    fig = figure(3.4, 5.0); rects = FS.tile_grid(fig, 4, columns=2, title_in=0.2, caption_in=0.36)
    assert len(rects) == 4
    sides = [(w * 3.4, h * 5.0) for _x, _y, w, h in rects]
    assert all(abs(a - b) < 1e-9 for a, b in sides) and sides[0][0] >= 1.15                      # square tiles, over an inch wide
    assert all(0 <= x and x + w <= 1 and 0 <= y and y + h <= 1 for x, y, w, h in rects)
    tops = sorted({round(y, 6) for _x, y, _w, _h in rects}, reverse=True)
    assert len(tops) == 2 and (tops[0] - tops[1]) * 5.0 >= sides[0][1] + 0.2 + 0.36               # two rows, each with room for its hour and its caption


def test_the_maturation_strip_is_four_large_frames_with_outlines_and_the_number_of_microcolonies():
    film = growth_film(); fig = figure(3.4, 5.0); axes = GF.draw_film_strip(fig, film, 4, 31.5)
    assert len(axes) == 4 and all(side_in(fig, ax)[0] >= 1.4 for ax in axes)                      # twice the old tiles
    assert [ax.get_title() for ax in axes] == ["24 h", "27 h", "31.5 h", "34.5 h"]
    assert all(re.fullmatch(r"(\d+ microcolon(y|ies)|no microcolony)", ax.get_xlabel()) for ax in axes)
    assert any(ax.collections for ax in axes)                                                    # the orange outlines of the microcolonies
    assert SP.collisions(fig) == []


def test_a_frame_whose_stain_was_not_measured_says_so():
    film = growth_film(matrix_frames={0: list(range(6))})                                        # the stain measured on frames 0 to 5 of 8
    fig = figure(3.4, 5.0); axes = GF.draw_film_strip(fig, film, 4, 31.5)
    notes = [ax.get_xlabel() for ax in axes]
    assert notes[-1].endswith("\nstain not measured") and "stain not measured" not in notes[0] and SP.collisions(fig) == []


def test_the_microcolonies_strip_is_six_frames_in_two_rows():
    frames = [blocks(20), blocks(20), blocks(3), blocks(3), blocks(3), blocks(3)]
    tracks = MF.follow_microcolonies(lambda t: frames[t], range(6), pixel_um=PX, dt_h=0.5, start_hour=24.0)
    film = F.MicrocolonyFilm(tracks, lambda t: frames[t], PX); film.set_root(1)
    fig = figure(3.4, 2.5); axes = F.draw_film_strip(fig, film, 6, 2)
    assert len(axes) == 6 and all(side_in(fig, ax)[0] >= 0.85 for ax in axes)
    assert len({round(ax.get_position().y0, 6) for ax in axes}) == 2 and SP.collisions(fig) == []


def test_the_pages_give_their_film_strips_the_room():
    from nd2_analyzer.ui.biofilms.stages.growth_page import GrowthPage
    from nd2_analyzer.ui.biofilms.stages.microcolony_page import MicrocolonyPage
    assert GrowthPage().film_canvas.minimumHeight() >= 330 and MicrocolonyPage().film_canvas.minimumHeight() >= 230


# ---------------------------------------------------------------- a click on a time plot moves the cursor (it broke when the axis was renamed "Time (h)")
def test_a_click_on_a_time_plot_moves_the_cursor_on_both_pages():
    from types import SimpleNamespace
    from nd2_analyzer.ui.biofilms.stages.growth_page import GrowthPage
    from nd2_analyzer.ui.biofilms.stages.microcolony_page import MicrocolonyPage
    from test_growth_plots_figure_style import grown
    page = GrowthPage(); page.set_recording("made-up recording", (24.0, 68.0)); page.set_measured(grown())
    count, rate = page.panels[0].figure.axes[:2]
    page._on_plot_press(SimpleNamespace(inaxes=count, xdata=40.0, ydata=1.0)); assert page.hour == 40.0      # the count panel has no axis title of its own
    page._on_plot_press(SimpleNamespace(inaxes=rate, xdata=51.0, ydata=1.0)); assert page.hour == 51.0
    page._on_plot_press(SimpleNamespace(inaxes=rate, xdata=200.0, ydata=1.0)); assert page.hour == 68.0      # not past the recording
    frames = [blocks(20), blocks(20), blocks(3), blocks(3), blocks(3), blocks(3)]
    tracks = MF.follow_microcolonies(lambda t: frames[t], range(6), pixel_um=PX, dt_h=0.5, start_hour=24.0)
    mpage = MicrocolonyPage(); mpage.set_recording("toy", (24.0, 26.5)); mpage.set_tracks(tracks, F.MicrocolonyFilm(tracks, lambda t: frames[t], PX))
    mpage._on_plot_press(mpage.panels[0], SimpleNamespace(inaxes=mpage.panels[0].figure.axes[0], xdata=25.5, ydata=1.0)); assert mpage.hour == 25.5
    combo = mpage.plot_combos[0]; combo.setCurrentIndex(combo.findData("waits"))                                 # not a time plot: its x axis is a duration
    mpage._on_plot_press(mpage.panels[0], SimpleNamespace(inaxes=mpage.panels[0].figure.axes[0], xdata=24.0, ydata=1.0)); assert mpage.hour == 25.5
