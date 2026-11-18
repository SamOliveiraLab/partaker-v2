"""The matrix stain (EPS) on the Microcolonies page itself: when each followed microcolony gets its stain and how much of
it is covered (in the list), the stain in green on the film strip and in the Follow window at once, the plot of the
stain inside and outside microcolonies, and plain words on what to click while there is no stain yet.

Two microcolonies of four cells, 2 um apart, four frames from 24 h; matrix over the left one from frame 2 (25 h)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace

import numpy as np
import pytest

from nd2_analyzer.analysis.biofilm_stages import matrix_stain as MX
from nd2_analyzer.analysis.biofilm_stages import microcolony_follow as MF
from nd2_analyzer.analysis.biofilm_stages import microcolony_metric_plots as XP
from nd2_analyzer.analysis.biofilm_stages import microcolony_plots as MP
from nd2_analyzer.analysis.biofilm_stages.microcolony_film import MicrocolonyFilm
from test_microcolony_rules import PX, blocks

FRAMES = [blocks(20)] * 4


def counted_at(t):
    m = np.zeros((120, 200), bool)
    if t >= 2:
        m[35:68, 15:68] = True
    return m


def dye_at(p, t):
    """The raw dye image the masks above come from: 8,000 counts of stain over a free dye level of 2,000."""
    return np.where(counted_at(t), 10000.0, 2000.0).astype(np.float32)


@pytest.fixture(scope="module")
def tracks():
    return MF.follow_microcolonies(lambda t: FRAMES[t], range(4), pixel_um=PX, dt_h=0.5, start_hour=24.0)


@pytest.fixture(scope="module")
def stain(tracks):
    return MX.measure_microcolony_stain(tracks, counted_at, PX)


def recipe():
    return {"reference": MX.DyeReference(zero=np.zeros((120, 200), np.float32), flat=np.ones((120, 200), np.float32)), "level": 2000.0,
            "settings": MX.MatrixSettings(), "channel": 2, "path": None, "raw_at": dye_at, "position": 0}


def test_the_stain_of_a_position_keeps_every_mask_and_gives_each_microcolony_its_numbers(tracks):
    asked, said = [], []
    stain = MX.measure_microcolony_stain(tracks, lambda t: asked.append(t) or counted_at(t), PX, progress=lambda k, n: said.append((k, n)))
    assert asked == [0, 1, 2, 3] and said[-1] == (4, 4)                                         # every mask worked out once
    assert stain.counted_at(2).dtype == bool and np.array_equal(stain.counted_at(2), counted_at(2)) and not stain.counted_at(1).any()
    assert stain.summary[1]["first_stain_h"] == 25.0 and stain.summary[1]["most_cover_pct"] == pytest.approx(100.0) and stain.summary[2]["first_stain_h"] is None
    hours, cover = stain.series(1)
    assert hours.tolist() == [24.0, 24.5, 25.0, 25.5] and cover.tolist() == pytest.approx([0.0, 0.0, 100.0, 100.0])


def test_the_list_says_when_each_microcolony_gets_its_matrix_stain(tracks, stain):
    plain = MP.entries(tracks)
    assert plain[0] == ("still there at 25.5 h", [(1, "Microcolony 1: from 24.0 h"), (2, "Microcolony 2: from 24.0 h")])
    with_stain = MP.entries(tracks, stain.summary)
    assert with_stain[0][1] == [(1, "Microcolony 1: from 24.0 h\n     matrix stain from 25 h, up to 100 % of its area"),
                                (2, "Microcolony 2: from 24.0 h\n     no matrix stain")]


def test_the_film_shows_the_stain_of_the_position_at_once(tracks, stain):
    film = MicrocolonyFilm(tracks, lambda t: FRAMES[t], PX); film.set_root(1)
    film.set_stain(stain.counted_at, rows=stain.rows, show=True)
    assert film.show_stain and film.stain_ready() and list(film.measure_stain()) == []          # nothing left to measure
    hours, cover = film.stain_series()
    assert hours.tolist() == [24.0, 24.5, 25.0, 25.5] and cover.tolist() == pytest.approx([0.0, 0.0, 100.0, 100.0])
    film.set_root(2)
    assert film.stain_ready() and film.stain_series()[1].tolist() == [0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------- the page
def test_the_page_has_a_matrix_stain_step_and_fills_the_list(tracks, stain):
    from nd2_analyzer.ui.biofilms.stages.microcolony_page import MicrocolonyPage
    page = MicrocolonyPage()
    assert "Matrix stain:" in page.step_labels["stain"].text()
    film = MicrocolonyFilm(tracks, lambda t: FRAMES[t], PX); page.set_tracks(tracks, film); page.candidates.setCurrentRow(2)
    page.set_stain(stain.summary)
    lines = [page.candidates.item(r).text() for r in range(page.candidates.count())]
    assert lines[1] == "Microcolony 1: from 24.0 h\n     matrix stain from 25 h, up to 100 % of its area" and lines[2].endswith("no matrix stain")
    assert page.selected() == 2                                                                 # the selection is kept
    page.set_stain(None)
    assert page.candidates.item(1).text() == "Microcolony 1: from 24.0 h" and page.selected() == 2


def test_the_matrix_stain_plot_says_what_to_click_until_the_maturation_tab_has_measured():
    from nd2_analyzer.ui.biofilms.stages.microcolony_page import MicrocolonyPage
    from test_growth_plots_figure_style import grown
    page = MicrocolonyPage(); combo = page.plot_combos[0]
    assert combo.itemText(combo.findData("stain")) == "Matrix stain over time"
    combo.setCurrentIndex(combo.findData("stain"))
    assert XP.STAIN_FIRST == "Matrix stain: on the Maturation tab click Dye frames..., then Measure. It then shows here."
    assert [t.get_text() for ax in page.panels[0].figure.axes for t in ax.texts] == [XP.STAIN_FIRST]
    page.set_growth(grown())
    assert page.panels[0].figure.texts[0].get_text() == "Matrix stain over time" and page.panels[0].figure.axes[0].get_ylabel() == "Area covered (%)"
    page.set_growth(grown(with_dye=False))                                                      # counted, but no dye frames were given
    assert [t.get_text() for ax in page.panels[0].figure.axes for t in ax.texts] == [XP.STAIN_FIRST]


# ---------------------------------------------------------------- the wiring
def test_the_worker_measures_the_stain_from_the_dye_channel(tracks):
    from nd2_analyzer.ui.biofilms.stages.microcolony_controller import StainWorker
    worker = StainWorker(tracks, recipe(), PX); got, said = [], []
    worker.done.connect(got.append); worker.progress.connect(lambda pct, words: said.append(words)); worker.run()
    assert len(got) == 1 and got[0].summary[1]["first_stain_h"] == 25.0 and got[0].summary[2]["first_stain_h"] is None and said[-1] == "100 %"
    broken = dict(recipe(), raw_at=lambda p, t: (_ for _ in ()).throw(OSError("dye file gone")))
    worker = StainWorker(tracks, broken, PX); why = []; worker.failed.connect(why.append); worker.run()
    assert why == ["OSError: dye file gone"]


def controller(tracks):
    from nd2_analyzer.ui.biofilms.stages.microcolony_controller import MicrocolonyController
    from nd2_analyzer.ui.biofilms.stages.microcolony_page import MicrocolonyPage
    info = SimpleNamespace(name="toy", n_frames=4, dt_s=1800.0, pixel_um=PX, start_hour=24.0, channel_roles={}, channel_labels={})
    c = MicrocolonyController(MicrocolonyPage(), lambda: info, lambda: 0)
    c.masks = lambda position: (None, list(range(4))); c.labels_at = lambda position: (lambda t: FRAMES[t])
    c.tracked[0] = tracks; c.drawn[0] = tracks
    return c


def test_the_stain_is_measured_by_itself_once_tracks_and_dye_are_both_there(tracks, monkeypatch):
    from nd2_analyzer.ui.biofilms.stages import microcolony_controller as M
    monkeypatch.setattr(M.StainWorker, "start", lambda self: self.run())                        # in the tool this runs off the window's thread
    c = controller(tracks); c.refresh()
    assert c.stains == {} and c.page.step_states["stain"][0] == "todo" and "Maturation tab" in c.page.step_labels["stain"].toolTip()
    c.stain_recipe = lambda position: recipe()                                                  # the Maturation tab has its dye frames and its measure
    c.on_stain_source_changed()
    assert c.stains[0].summary[1]["first_stain_h"] == 25.0 and c.page.step_states["stain"][0] == "done"
    assert "matrix stain from 25 h" in c.page.candidates.item(1).text()
    assert c.page.film.show_stain and c.page.film.stain_ready()                                 # green on the film strip, with no click
    r = recipe(); by_the_rule = MX.matrix(MX.stain(MX.correct(dye_at(0, 3), r["reference"]), r["level"], r["settings"]), PX, r["settings"])[0]
    assert np.array_equal(c.stain_source(0)(3), by_the_rule) and by_the_rule[45:58, 25:58].all() and not by_the_rule[:30].any()   # at once in the Follow window
    c.stain_recipe = lambda position: None; c.on_stain_source_changed()                         # the measure was cleared
    assert c.stains == {} and c.page.step_states["stain"][0] == "todo" and c.page.candidates.item(1).text() == "Microcolony 1: from 24.0 h"


def test_the_maturation_controller_gives_the_recipe_of_a_measured_position():
    from nd2_analyzer.ui.biofilms.stages.growth_controller import GrowthController
    from nd2_analyzer.ui.biofilms.stages.growth_page import GrowthPage
    from test_growth_plots_figure_style import grown
    info = SimpleNamespace(name="toy", n_frames=8, dt_s=5400.0, pixel_um=0.1, start_hour=24.0, channel_roles={"0": "phase", "2": "matrix"}, channel_labels={})
    g = GrowthController(GrowthPage(), lambda: info)
    assert g.stain_recipe(2) is None
    g.measured = grown(); g.reference = recipe()["reference"]
    made = g.stain_recipe(2)
    assert made["level"] == 2000.0 and made["channel"] == 2 and made["position"] == 2 and made["settings"] == g.measured.settings and callable(made["raw_at"])
    assert g.stain_recipe(7) is None                                                            # a position that was not measured
