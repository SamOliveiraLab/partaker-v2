"""The twin's seed, rule measurement, results reading and rule-set naming on toy data."""
import json
import numpy as np
import pandas as pd
import pytest

from nd2_analyzer.analysis.twin import seed as S, results as R, measure as M, runner as RUN


def capsule_mask(cx, cy, length_total, width, angle, shape=(80, 120), px=0.1):
    """A rasterised capsule (um) on a px grid."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    x = (xx + 0.5) * px - cx; y = (yy + 0.5) * px - cy
    u = x * np.cos(angle) + y * np.sin(angle); v = -x * np.sin(angle) + y * np.cos(angle)
    r = width / 2
    du = np.clip(np.abs(u) - (length_total - 2 * r) / 2, 0, None)
    return (du ** 2 + v ** 2) <= r * r


def test_capsule_inversion_recovers_length_and_width():
    tab = S.capsule_table()
    from skimage.measure import regionprops
    for Lt, w in [(3.0, 1.0), (2.2, 0.9), (5.0, 1.2)]:
        m = capsule_mask(6, 4, Lt, w, 0.3, shape=(160, 240), px=0.05)
        p = regionprops(m.astype(int))[0]
        L_est, w_est = tab.invert(p.major_axis_length * 0.05, p.area * 0.05 ** 2)
        assert L_est == pytest.approx(Lt, rel=0.04) and w_est == pytest.approx(w, rel=0.04)


def test_frame_seed_positions_relative_to_roi_box():
    mask = np.zeros((80, 120), np.int32)
    mask[capsule_mask(6, 4, 3.0, 1.0, 0.0)] = 1          # inside the ROI
    mask[capsule_mask(1.0, 7.5, 2.0, 0.8, 0.0)] = 2      # outside
    roi = np.zeros((80, 120), bool); roi[10:70, 30:110] = True
    df = S.frame_seed(mask, roi, 0.1)
    assert list(df.cell_id) == [1]
    assert df.x_um.iloc[0] == pytest.approx(6 - 3.0, abs=0.1) and df.y_um.iloc[0] == pytest.approx(4 - 1.0, abs=0.1)
    assert df.length_total_um.iloc[0] == pytest.approx(3.0, rel=0.05) and df.width_um.iloc[0] == pytest.approx(1.0, rel=0.05)
    ch = S.chamber(roi, 0.1, 300)
    assert ch["width_um"] == pytest.approx(8.0) and ch["height_um"] == pytest.approx(6.0)


def fake_run(n_steps=4, frame_s=300.0):
    res = []
    for i in range(n_steps):
        agents = {f"c{j}": dict(location=(2.0 + j, 3.0), length=1.0 + 0.1 * i, radius=0.5, angle=0.0) for j in range(i + 1)}
        agents["far"] = dict(location=(50.0, 3.0), length=1.0, radius=0.5, angle=0.0)   # outside the ROI
        res.append(dict(time=i * frame_s, agents=agents))
    return dict(results=res, roi=dict(x0=1.0, y0=1.0, w=20.0, h=20.0))


def test_results_counts_and_areas_and_score():
    run = fake_run()
    counts, areas = R.curves([run, run], 4, 300.0)
    assert counts.shape == (2, 4) and list(counts[0]) == [1, 2, 3, 4]
    assert areas[0, 0] == pytest.approx(1.0 * 1.0 + np.pi * 0.25)          # L_total 2.0: straight part 1.0 x width 1.0 + the two ends
    sc = R.score(counts, areas, [1, 2, 3, 4], areas[0], from_frame=0)
    assert sc["count_rel_err"] == 0 and sc["area_rel_err"] == 0 and sc["seeds"] == 2


def test_rule_sets_nest():
    assert RUN.rule_set(False, False, False) == "base"
    assert RUN.rule_set(True, False, False) == "position"
    assert RUN.rule_set(True, True, False) == "lag" and RUN.rule_set(False, True, False) == "lag"
    assert RUN.rule_set(True, True, True) == "crowding"
    assert RUN.rule_set(True, True, False, flow=True) == "lag+flow"
    assert RUN.rule_flags("lag+flow") == dict(position=True, lag=True, crowding=False, flow=True)
    assert RUN.rule_flags("base") == dict(position=False, lag=False, crowding=False, flow=False)


def test_twin_page_has_no_overlapping_elements():
    """The page at the canvas size the tab uses: nothing closer than 0.1 in to anything else (the DARPA figure rule)."""
    from matplotlib.figure import Figure
    from nd2_analyzer.analysis.twin.draw import draw_twin
    from nd2_analyzer.analysis.tracking.film_render import collisions
    rng = np.random.default_rng(0)
    mask = np.zeros((300, 180), np.int32)
    for i in range(1, 60):
        y, x = rng.integers(5, 290), rng.integers(5, 170); mask[y:y + 6, x:x + 14] = i
    cells = [(f"c{i}", rng.uniform(1, 10), rng.uniform(1, 18), 3.0, 0.5, rng.uniform(0, 3.14)) for i in range(60)]
    hours = np.arange(28) * 5 / 60; rc = 250 + 10 * np.arange(28); ra = 580 + 16 * np.arange(28)
    sim_counts = np.vstack([rc * 1.05, rc * 1.1]); sim_areas = np.vstack([ra * 1.05, ra * 1.1])
    fig = Figure(figsize=(10.8, 5.2), dpi=100)
    from matplotlib.backends.backend_agg import FigureCanvasAgg; FigureCanvasAgg(fig)
    draw_twin(fig, 14, hours, mask, cells, 11.6, 19.4, rc, ra, sim_counts, sim_areas)
    hits = collisions(fig)
    assert hits == [], hits


def test_measure_adder_on_a_toy_generation():
    # one full generation: cell 2 (parent 1, children 3 and 4) born 2 um long, divides at 4 um after 4 frames of 5 min
    mk = lambda i, ts, x, y, parent=None, children=None: {"ID": i, "t": ts, "x": [x] * len(ts), "y": [y] * len(ts), "parent": parent, "children": children or []}
    tracks = [mk(1, [0], 30, 20, None, [2, 5]), mk(2, [1, 2, 3, 4], 30, 20, 1, [3, 4]), mk(3, [5], 25, 20, 2), mk(4, [5], 35, 20, 2), mk(5, [1], 60, 40, 1)]
    tracks += [mk(6, [1, 2, 3, 4], 60, 60, 1, [7, 8]), mk(7, [5], 55, 60, 6), mk(8, [5], 65, 60, 6), mk(9, [1, 2, 3], 90, 40, 1, [10, 11]), mk(10, [4], 85, 40, 9), mk(11, [4], 95, 40, 9)]
    lengths = {1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0, 5: 2.0, 6: 2.0, 7: 2.0, 8: 2.0, 9: 2.0, 10: 2.0, 11: 2.0}
    def masks_at(t):
        m = np.zeros((100, 120), np.int32)
        for k, x in enumerate(tracks, start=1):
            if t in x["t"]:
                i = x["t"].index(t); grow = 1 + i * (1.0 / max(len(x["t"]) - 1, 1)) if len(x["t"]) > 1 else 1
                m[capsule_mask(x["x"][0] * 0.1, x["y"][0] * 0.1, 2.0 * grow, 1.0, 0.0, shape=(100, 120), px=0.1)] = k
        return m
    ad = M.measure_adder(tracks, masks_at, 0.1, 5.0, width_um=1.0)
    assert ad["n_full_generations"] == 3
    assert ad["birth_total_um"] == pytest.approx(2.0, rel=0.08) and ad["division_total_um"] == pytest.approx(4.0, rel=0.08)
    assert ad["interdivision_min_mean"] == pytest.approx((20 + 20 + 15) / 3)
    assert 0 < ad["alpha_per_h"] < 5
