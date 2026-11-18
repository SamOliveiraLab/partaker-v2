"""Growth states (Partaker 2, Figure 5): the number of cells in the whole frame, its rate of change, and Th2, the time
growth stops, predicted from a Gompertz curve fitted to the count. Made-up curves with known answers, and the counts of
Figure 5 as a known answer for the fit (skipped when that file is not on this machine)."""
from pathlib import Path

import numpy as np
import pytest

from nd2_analyzer.analysis.biofilm_stages import growth_states as GS

FIGURE5_COUNTS = Path.home() / "Workspace/results/figures_3to6/figure5_new/cells_full_frames_24_68h.csv"


def test_cells_under_one_square_micron_are_not_counted():
    labels = np.zeros((60, 60), np.int32)
    labels[5:15, 5:25] = 1; labels[30:40, 5:25] = 2          # two cells of 2 um2 at 0.1 um per px
    labels[50:55, 5:15] = 3                                  # 0.5 um2 of debris
    assert GS.count_cells(labels, pixel_um=0.1) == 2


def test_growth_rate_of_a_steady_rise_is_its_slope():
    hours = np.arange(24, 40, 1.5)
    assert GS.growth_rate(hours, 100 + 12.0 * (hours - 24)) == pytest.approx(np.full(hours.size, 12.0))


def test_growth_rate_looks_three_hours_each_way():
    hours = np.arange(0, 30, 1.5); counts = np.where(hours < 15, 0.0, 10.0 * (hours - 15))     # flat, then 10 cells per hour
    rate = GS.growth_rate(hours, counts)
    assert rate[hours == 6][0] == pytest.approx(0.0) and rate[hours == 24][0] == pytest.approx(10.0)
    assert 0 < rate[hours == 15][0] < 10                      # around the bend the window sees both parts


def test_a_gompertz_curve_is_recovered_from_its_own_points():
    hours = np.arange(24, 70, 1.5)
    K, r, t0 = GS.fit_gompertz(hours, GS.gompertz(hours, 2500.0, 0.07, 38.0))
    assert (K, r, t0) == pytest.approx((2500.0, 0.07, 38.0), rel=1e-4)


def test_th2_is_where_growth_falls_to_a_tenth_of_its_peak():
    # for a Gompertz curve the rate is a tenth of its peak 3.2646 / r after the fastest growth (u + exp(-u) = 1 + ln 10)
    th2, fastest, peak = GS.th2_from_fit((1000.0, 0.1, 40.0))
    assert th2 == pytest.approx(40.0 + 32.646, abs=0.1) and fastest == pytest.approx(40.0, abs=0.1)
    assert peak == pytest.approx(1000.0 * 0.1 / np.e, rel=1e-3)


@pytest.fixture(scope="module")
def figure5_counts():
    if not FIGURE5_COUNTS.exists():
        pytest.skip("the counts of Figure 5 are not on this machine")
    import csv
    per = {}
    for row in csv.DictReader(FIGURE5_COUNTS.open()):
        if 58 <= float(row["h"]) <= 62.4:                    # out of focus on every position
            continue
        per.setdefault(int(row["p"]), []).append((float(row["h"]), float(row["cells"])))
    return {p: (np.array([h for h, _ in v]), np.array([n for _, n in v])) for p, v in per.items()}


def test_the_mean_count_and_fastest_growth_of_figure5(figure5_counts):
    hours, mean, _sd = GS.mean_curve(figure5_counts)
    at = lambda h: float(mean[hours == h][0])
    assert [at(h) for h in (24.0, 36.0, 48.0, 57.0, 68.0)] == pytest.approx([214, 773, 1509, 1788, 2244], abs=0.5)
    rates = np.mean([GS.growth_rate(h, n) for h, n in figure5_counts.values()], axis=0)
    assert hours[int(np.argmax(rates))] == 37.5 and float(rates.max()) == pytest.approx(75.1, abs=0.05)


def test_th2_predicted_from_the_counts_of_figure5(figure5_counts):
    out = GS.predict_th2(figure5_counts, resamples=2000, seed=0)
    assert out["th2_h"] == pytest.approx(90.3, abs=0.05)
    assert out["range95_h"] == pytest.approx((81.1, 108.5), abs=0.1)
    assert out["fastest_h"] == pytest.approx(38.7, abs=0.05)             # of the fitted curve; the counted one is 37.5 h


# ---------------------------------------------------------------- Th1: the matrix stain appears and stays
def test_th1_is_the_first_hour_from_which_matrix_stain_stays():
    hours = [24.0, 25.5, 27.0, 28.5, 30.0, 31.5]
    assert GS.th1(hours, [0, 0.2, 0, 0, 0.1, 0.4]) == 30.0           # the stain at 25.5 h went away again: not yet
    assert GS.th1(hours, [0.3, 0.2, 0.5, 0.4, 0.1, 0.4]) == 24.0     # there from the first frame
    assert GS.th1(hours, [0, 0, 0, 0, 0, 0]) is None and GS.th1(hours, [0, 0, 0.2, 0.3, 0.1, 0]) is None


def test_th1_of_figure5_from_the_matrix_stain_of_figure4():
    table = Path.home() / "Workspace/results/figures_3to6/figure4_new/cache/cutoff_frames.csv"
    if not table.exists():
        pytest.skip("the matrix stain table of Figure 4 is not on this machine")
    import csv
    per = {}
    for row in csv.DictReader(table.open()):
        per.setdefault(int(row["position"]), []).append((float(row["hours"]), float(row["area_used"])))
    out = GS.th1_of_positions({p: ([h for h, _ in v], [a for _, a in v]) for p, v in per.items()})
    assert out["per_position_h"] == {1: 37.5, 2: 33.0, 3: 31.5, 6: 31.5, 8: 39.0, 9: 33.0}
    assert round(out["th1_h"], 1) == 34.2 and round(out["sd_h"], 1) == 3.2
