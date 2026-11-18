"""The microcolony metrics of Figure 3 (f to k, n, q), from the cell labels of a frame alone, on a made-up frame whose
answers are worked out by hand.

The frame (0.1 um per px, 200 x 300 px), every cell 6 x 20 px (1.2 um2):
  A   five parallel rods stacked 3 px (0.3 um) apart                           -> a microcolony of 5 cells
  B   two rods lying flat, two standing upright, each 3 px from the next       -> a microcolony of 4 cells
  two single rods far from everything
"""
import numpy as np
import pytest

from nd2_analyzer.analysis.biofilm_stages import microcolony_metrics as MM

PX = 0.1


def frame(with_last_of_b: bool = True) -> np.ndarray:
    lab = np.zeros((200, 300), np.int32); n = 0
    for k in range(5):                                          # A
        n += 1; lab[60 + 9 * k:66 + 9 * k, 40:60] = n
    n += 1; lab[60:66, 150:170] = n                             # B: flat, flat, upright, upright
    n += 1; lab[69:75, 150:170] = n
    n += 1; lab[58:78, 173:179] = n
    if with_last_of_b:
        n += 1; lab[58:78, 182:188] = n
    n += 1; lab[150:156, 60:80] = n                             # the two single cells
    n += 1; lab[150:156, 220:240] = n
    return lab


def test_the_counts_of_a_frame():
    m = MM.frame_metrics(frame(), PX)
    assert (m.cells, m.microcolonies, m.cells_in_microcolonies) == (11, 2, 9)
    assert m.cells_in_pct == pytest.approx(100 * 9 / 11)
    assert m.cells_per_microcolony == 4.5                                     # 9 cells in 2 microcolonies
    assert m.largest_share_pct == pytest.approx(100 * 5 / 9)                  # A holds 5 of the 9


def test_local_density_counts_the_other_cells_within_3_um():
    # centres within 30 px: A's rods have 3, 4, 4, 4, 3 neighbours; each cell of B has the other 3; the single cells none
    m = MM.frame_metrics(frame(), PX)
    assert m.density_in == pytest.approx(100 * (18 + 12) / 9 / (np.pi * 3.0 ** 2), abs=1e-6)       # 11.79 cells per 100 um2
    assert m.density_single == 0.0


def test_alignment_is_1_for_parallel_neighbours_and_minus_1_for_crossed_ones():
    # A: every neighbour parallel -> 1. B: one parallel and two crossed neighbours -> (1 - 1 - 1) / 3
    m = MM.frame_metrics(frame(), PX)
    assert m.alignment == pytest.approx((5 * 1.0 + 4 * (-1 / 3)) / 9, abs=1e-6)                    # 0.407


def test_area_is_the_cells_grown_by_half_the_gap():
    # a cell grown by 2.5 px is its box widened by 2 px less the 4 corners: 10 x 24 - 4 = 236 px
    # A: 5 x 236 less 4 shared rows of 22 px = 1,092 px.  B: 450 (the flat pair) + 236 + 236 - 16 - 22 = 884 px
    m = MM.frame_metrics(frame(), PX)
    assert m.area_pct == pytest.approx(100 * (1092 + 884) / (200 * 300), abs=1e-9)                 # 3.29 % of the frame


def test_spacing_is_the_distance_between_the_centres_of_neighbouring_microcolonies():
    # centre of A (80.5, 49.5); centre of B (59453 / 884, 150039 / 884) = (67.2545, 169.7274): 120.955 px apart
    m = MM.frame_metrics(frame(), PX)
    assert m.spacing_um == pytest.approx(12.0955, abs=1e-3)
    assert np.isnan(MM.frame_metrics(frame(with_last_of_b=False), PX).spacing_um)                  # one microcolony: no neighbour


def test_density_against_the_distance_from_the_centre_of_the_cells_own_microcolony():
    # rings 1 um wide. A: its rods sit 1.8, 0.9, 0, 0.9, 1.8 um from its centre. B (centre 67.25, 169.75): 1.13, 1.11, 0.58, 1.48 um
    m = MM.frame_metrics(frame(), PX, ring_um=1.0)
    per_cell = 100 / (np.pi * 3.0 ** 2)
    assert m.ring_n[:3].tolist() == [4, 5, 0] and m.ring_n.sum() == 9
    assert m.ring_sum[:2] == pytest.approx([(4 + 4 + 4 + 3) * per_cell, (3 + 3 + 3 + 3 + 3) * per_cell])


def test_a_frame_with_no_cells_or_no_microcolony():
    empty = MM.frame_metrics(np.zeros((200, 300), np.int32), PX)
    assert (empty.cells, empty.microcolonies, empty.cells_in_pct, empty.area_pct) == (0, 0, 0.0, 0.0)
    assert np.isnan(empty.cells_per_microcolony) and np.isnan(empty.density_in) and np.isnan(empty.largest_share_pct)
    lone = np.zeros((200, 300), np.int32); lone[150:156, 60:80] = 1
    m = MM.frame_metrics(lone, PX)
    assert (m.cells, m.microcolonies, m.cells_in_pct, m.density_single) == (1, 0, 0.0, 0.0) and np.isnan(m.density_in)


def test_random_placement_keeps_the_cells_and_breaks_the_groups():
    lab = frame(); rng = np.random.default_rng(0)
    placed = MM.random_placement(lab, rng)
    assert placed.shape == lab.shape and len(np.unique(placed)) - 1 == 11                         # every cell is there, none on another
    areas = np.bincount(placed.ravel())[1:]
    assert areas.min() >= 114 and areas.max() <= 126                                              # turned, so a few px more or less than 120
    assert not np.array_equal(placed, MM.random_placement(lab, np.random.default_rng(1)))
    assert np.array_equal(placed, MM.random_placement(lab, np.random.default_rng(0)))             # the same seed, the same placement
    assert MM.random_share(lab, PX, placements=5, seed=0) < 100 * 9 / 11 - 40                     # 11 cells over 600 um2 seldom link


def test_the_measure_takes_every_position_and_frame_and_averages_the_positions():
    labels = lambda p, t: frame(with_last_of_b=(p == 0))            # position 1 lacks one cell of B: one microcolony of 5
    seen = []
    out = MM.measure_metrics(labels, {0: [0, 3, 6], 1: [0, 3, 6]}, PX, hour_of=lambda t: 24.0 + 0.5 * t, progress=lambda k, n: seen.append((k, n)))
    assert out.positions == [0, 1] and out.hours[1].tolist() == [24.0, 25.5, 27.0] and seen[-1] == (6, 6)
    assert out.values["microcolonies"][0].tolist() == [2, 2, 2] and out.values["microcolonies"][1].tolist() == [1, 1, 1]
    hours, mean, sd = out.mean_of("cells_in_pct")
    assert hours.tolist() == [24.0, 25.5, 27.0] and mean == pytest.approx(np.full(3, (100 * 9 / 11 + 50) / 2))
    assert sd == pytest.approx(np.full(3, np.std([100 * 9 / 11, 50.0], ddof=1)))
    _h, spacing, spacing_sd = out.mean_of("spacing_um")                 # position 1 has no neighbour: the mean is position 0 alone
    assert spacing == pytest.approx(np.full(3, 12.0955), abs=1e-3) and (spacing_sd == 0).all()
    assert out.have("spacing_um")[1].tolist() == [1, 1, 1] and out.have("cells_in_pct")[1].tolist() == [2, 2, 2]   # positions with a value
    _h, per, _sd = out.mean_of("cells_per_microcolony", geometric=True)
    assert per == pytest.approx(np.full(3, np.sqrt(4.5 * 5.0)))
    ring_hours, edges, density = out.radial(min_cells=1)
    assert ring_hours.tolist() == [24.0, 25.5, 27.0] and edges[:3].tolist() == [0.0, 2.0, 4.0]
    assert density.shape == (3, len(edges) - 1) and np.isnan(density[0, 1])                       # every cell is within 2 um of its centre


def test_random_placement_is_measured_only_at_the_hours_asked_for():
    out = MM.measure_metrics(lambda p, t: frame(), {0: [0, 3, 6]}, PX, hour_of=lambda t: 24.0 + 0.5 * t, random_hours=(24.0, 27.0), placements=2)
    r = out.values["random_pct"][0]
    assert np.isfinite(r[[0, 2]]).all() and np.isnan(r[1]) and (r[[0, 2]] < 40).all()


def test_cell_shapes_are_the_ones_scikit_image_measures():
    """Centre, direction and length over width of every cell in one pass, checked against skimage.measure.regionprops."""
    from skimage.draw import ellipse
    from skimage.measure import regionprops
    rng = np.random.default_rng(5); lab = np.zeros((300, 400), np.int32)
    for k in range(1, 25):
        rr, cc = ellipse(30 + 45 * ((k - 1) // 6), 35 + 60 * ((k - 1) % 6), rng.uniform(3, 6), rng.uniform(8, 18), rotation=rng.uniform(-np.pi / 2, np.pi / 2))
        lab[rr, cc] = k
    labels, centres, direction, aspect = MM.cell_shapes(lab)
    props = regionprops(lab)
    assert labels.tolist() == [q.label for q in props]
    assert centres == pytest.approx(np.array([q.centroid for q in props]), abs=1e-9)
    assert aspect == pytest.approx([q.major_axis_length / q.minor_axis_length for q in props], rel=1e-9)
    assert np.cos(2 * (direction - np.array([q.orientation for q in props]))) == pytest.approx(np.ones(len(props)), abs=1e-9)
