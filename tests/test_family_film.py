"""Families and their drawing: descendants, tree rows, camera, owners, and a film frame on a toy recording."""
import numpy as np
import pytest

from nd2_analyzer.analysis.tracking import family as F
from nd2_analyzer.analysis.tracking.film_render import FamilyFilm, tracked_panel, id_colour


def toy_tracks():
    # 1 (frames 0-2) divides into 2 and 3 (frames 3-5); 3 divides into 4 and 5 (frames 6-7); 9 is another family
    mk = lambda i, ts, xs, ys, parent=None, children=None: {"ID": i, "t": ts, "x": xs, "y": ys, "parent": parent, "children": children or []}
    return [mk(1, [0, 1, 2], [20, 20, 20], [20, 20, 20], None, [2, 3]),
            mk(2, [3, 4, 5], [14, 14, 14], [20, 20, 20], 1),
            mk(3, [3, 4, 5], [26, 26, 26], [20, 20, 20], 1, [4, 5]),
            mk(4, [6, 7], [24, 24], [16, 16], 3),
            mk(5, [6, 7], [28, 28], [24, 24], 3),
            mk(9, [0, 1, 2, 3, 4, 5, 6, 7], [50] * 8, [50] * 8, None)]


def toy_masks():
    masks = np.zeros((8, 64, 64), np.int32)
    by = F.index_tracks(toy_tracks())
    lab = 1
    for t in range(8):
        for i in by:
            p = F.point_at(by, i, t)
            if p is None: continue
            x, y = int(p[0]), int(p[1]); masks[t, y - 2:y + 3, x - 2:x + 3] = lab; lab += 1
    return masks


def test_family_and_sizes():
    by = F.index_tracks(toy_tracks())
    assert F.family(by, 1) == [1, 2, 3, 4, 5]
    assert F.families_by_size(by) == [(1, 5), (9, 1)]
    assert F.root_of(by, 5) == 1 and F.root_of(by, 9) == 9


def test_tree_rows_leaves_apart_parents_between():
    by = F.index_tracks(toy_tracks())
    rows = F.tree_rows(by, 1)
    assert rows[2] == 0 and rows[4] == 1 and rows[5] == 2
    assert rows[3] == pytest.approx(1.5) and rows[1] == pytest.approx(0.75)


def test_span_divisions_generations():
    by = F.index_tracks(toy_tracks()); fam = F.family(by, 1)
    assert F.family_span(by, fam) == (0, 7)
    assert F.divisions(by, fam) == [2, 5]
    assert {g: sorted(v) for g, v in F.generations(by, 1, fam).items()} == {0: [1], 1: [2, 3], 2: [4, 5]}
    assert F.interdivision_frames(by, fam) == [3]           # only cell 3 was born and divided inside the recording


def test_camera_follows_and_stays_inside():
    by = F.index_tracks(toy_tracks()); fam = F.family(by, 1)
    sl = F.camera_crop(by, fam, 4, (64, 64), half=10)
    assert sl[0] == slice(10, 30) and sl[1] == slice(10, 30)
    sl = F.camera_crop(by, fam, 0, (64, 64), half=40)       # bigger than the frame allows: clamped to the frame
    assert sl[0].stop - sl[0].start == 64 and sl[0].start == 0


def test_owner_map_and_picking():
    by = F.index_tracks(toy_tracks()); masks = toy_masks()
    own = F.owner_map(by, masks[4], 4)
    assert set(own.values()) == {2, 3, 9}
    assert F.track_at_point(by, 4, 15, 21) == 2
    assert F.track_at_point(by, 4, 5, 5) is None


def test_panel_colours_family_bright_and_others_dim():
    by = F.index_tracks(toy_tracks()); masks = toy_masks(); t = 4
    own = F.owner_map(by, masks[t], t)
    rgb = tracked_panel(masks[t], own, [2], (slice(0, 64), slice(0, 64)))
    assert rgb.shape == (64, 64, 3)
    assert np.allclose(rgb[20, 14], id_colour(2))                 # followed cell at full colour
    assert np.allclose(rgb[20, 26], id_colour(3) * 0.55)          # its sister, dimmed
    assert (rgb[0, 0] == 0).all()                                   # background black


def test_film_frame_and_album_render():
    masks = toy_masks(); hours = np.arange(8) * 5 / 60
    film = FamilyFilm(toy_tracks(), 1, lambda t: masks[t], (64, 64), hours, pixel_um=0.0645, half=16)
    assert film.t_first == 0 and film.t_last == 7 and film.alive(4) == 2
    img = film.frame_image(4, size=(6, 2.5), dpi=60)
    assert img.shape == (150, 360, 3) and img.max() == 255
    from matplotlib.figure import Figure
    from nd2_analyzer.analysis.tracking.film_render import collisions
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(10, 4.6), dpi=100); FigureCanvasAgg(fig); film.draw(fig, 4)
    assert collisions(fig) == [], collisions(fig)
    fig = Figure(figsize=(8, 4)); film.draw_album(fig)
    assert len(fig.axes) == 1 + 4                                   # the tree and four stills: start, divisions at 2 and 5, end
