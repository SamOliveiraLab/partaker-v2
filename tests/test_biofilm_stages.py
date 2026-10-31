"""The biofilm stage pages' data and drawing: the study loader on the run-2 results (skipped when they are not on this machine),
the microcolony rule on a toy frame, and every page figure free of overlapping elements."""
import json
from pathlib import Path

import numpy as np
import pytest

from nd2_analyzer.analysis.biofilm_stages import plots as P
from nd2_analyzer.analysis.biofilm_stages.study import BiofilmStudy, StageHours, microcolony_labels
from nd2_analyzer.analysis.tracking.film_render import collisions

STUDY = Path(__file__).resolve().parents[1] / "src/nd2_analyzer/ui/biofilms/stages/studies/ecoli_run2.json"


def test_microcolony_rule_on_a_toy_frame():
    cells = np.zeros((200, 200), np.int32)
    for i, (y, x) in enumerate([(50, 50), (50, 62), (62, 50), (62, 62)], start=1):   # four cells 2 px apart
        cells[y:y + 10, x:x + 10] = i
    cells[150:160, 150:160] = 5; cells[150:160, 162:172] = 6                           # two cells, not enough
    col = microcolony_labels(cells, pixel_um=0.25, blur_um=1.0)
    assert set(np.unique(col[cells > 0])) - {0} and (col[cells == 5] == 0).all() and (col[cells == 1] > 0).all()
    assert len(set(col[cells == 1]) | set(col[cells == 4])) == 1


def test_stages_from_hours():
    sh = StageHours(th1_h=29.5, th2_h=None, first_h=24, last_h=68)
    assert [s[0].split()[0] for s in sh.stages()] == ["I", "II"]
    sh = StageHours(th1_h=29.5, th2_h=63.8, first_h=24, last_h=68)
    assert [s[0].split()[0] for s in sh.stages()] == ["I", "II", "III"]


@pytest.fixture
def study():
    st = BiofilmStudy.from_json(STUDY)
    if not st.available()["results"]:
        pytest.skip("run-2 results not on this machine")
    return st


def test_study_reads_the_tool_tables(study):
    assert study.kept() == [1, 2, 3, 6, 8, 9]
    sh = study.stage_hours(); assert sh.th1_h == pytest.approx(29.5) and sh.th2_h is None and sh.merge_h == 34.0
    inc = study.cells_in_microcolonies(); assert inc.hours[0] == 24.0 and 50 < inc.mean[0] < 65 and inc.mean[-1] > 98
    assert study.matrix_pct().mean[0] == 0.0 and study.area_rate().mean.max() == pytest.approx(2.38, abs=0.01)


def _fig():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(12.5, 6.2), dpi=100); FigureCanvasAgg(fig); return fig


def test_pages_have_no_overlapping_elements(study):
    fr = study.frame(2, 45.0) if study.available()["masks"] else None
    for fn in (lambda f: P.draw_maturation(f, study, 45.0, fr, 2), lambda f: P.draw_microcolonies(f, study, 45.0, fr, 2),
               lambda f: P.draw_matrix_density(f, study, 45.0), lambda f: P.draw_seeding(f, study, 45.0), lambda f: P.draw_dispersion(f, study, 45.0)):
        fig = _fig(); fn(fig); hits = collisions(fig)
        assert hits == [], (fn, hits)
    fig = _fig(); ax = fig.add_axes([0.05, 0.3, 0.9, 0.3]); P.draw_strip(ax, study.stage_hours(), 45.0); assert collisions(fig) == []
