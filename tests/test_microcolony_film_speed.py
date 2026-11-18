"""The film of a followed microcolony has to stay quick on a full field (2,044 px, a microcolony made of many): its picture
is worked out around each microcolony, not on the whole box once per microcolony, the film strip and the window ask for a
picture no finer than they show, and picking another microcolony does not read every frame again."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from matplotlib.colors import to_rgb
from scipy import ndimage as ndi

from nd2_analyzer.analysis.biofilm_stages import microcolony_film as F
from nd2_analyzer.analysis.biofilm_stages import microcolony_follow as MF
from test_microcolony_rules import PX, blocks

FRAMES = [blocks(20), blocks(20), blocks(3), blocks(3), blocks(3), blocks(3)]          # two microcolonies that merge at 25 h


@pytest.fixture(scope="module")
def tracks():
    return MF.follow_microcolonies(lambda t: FRAMES[t], range(6), pixel_um=PX, dt_h=0.5, start_hour=24.0)


def whole_box_picture(film, t):
    """The picture as it was first written: every step on the whole box, once per microcolony (slow, and the reference)."""
    owners, lab = film.owners_image(t); rgb = np.zeros((*lab.shape, 3)); rgb[lab > 0] = to_rgb(F.LONE); rgb[owners < 0] = to_rgb(F.OTHER)
    lines = []; r_px = film.settings.gap_um / 2 / film.pixel_um
    for r in [int(v) for v in np.unique(owners) if v > 0]:
        mine = owners == r; rgb[mine] = to_rgb(film.palette.get(r, F.OTHER))
        grown = ndi.distance_transform_edt(~mine) <= r_px
        soft = ndi.gaussian_filter(grown.astype(np.float32), 0.45 / film.pixel_um)
        shade = np.clip(soft * 1.6, 0, 1)[..., None] * 0.30; back = (lab == 0)[..., None]
        rgb = np.where(back, rgb * (1 - shade) + np.array(to_rgb(film.palette.get(r, F.OTHER))) * shade, rgb)
        lines.append((soft, film.palette.get(r, F.OTHER), r, ndi.center_of_mass(mine)))
    return rgb, lines


@pytest.mark.parametrize("t", [0, 1, 3])
def test_the_picture_worked_out_around_each_microcolony_is_the_picture_of_the_whole_box(tracks, t):
    film = F.MicrocolonyFilm(tracks, lambda k: FRAMES[k], PX); film.set_root(1)
    rgb, lines = film.picture(t); want_rgb, want_lines = whole_box_picture(film, t)
    assert rgb.shape == want_rgb.shape and np.allclose(rgb, want_rgb, atol=1e-6)
    assert [(colour, r) for _s, colour, r, _c in lines] == [(colour, r) for _s, colour, r, _c in want_lines]
    for (soft, _c, _r, centre), (want_soft, _c2, _r2, want_centre) in zip(lines, want_lines):
        assert np.allclose(soft, want_soft, atol=1e-6) and centre == pytest.approx(want_centre)


def test_a_coarser_picture_is_the_same_scene_with_fewer_pixels(tracks):
    film = F.MicrocolonyFilm(tracks, lambda k: FRAMES[k], PX); film.set_root(1)
    fine, fine_lines = film.picture(0); coarse, coarse_lines = film.picture(0, step=2)
    assert coarse.shape[:2] == fine[::2, ::2].shape[:2] and [r for _s, _c, r, _m in coarse_lines] == [r for _s, _c, r, _m in fine_lines]
    cells = lambda rgb: (rgb.sum(-1) > 0)
    assert np.array_equal(cells(coarse), cells(fine)[::2, ::2])                          # the same cells, every second pixel
    for (soft, _c, _r, centre), (_s, _c2, _r2, fine_centre) in zip(coarse_lines, fine_lines):
        assert soft.shape == coarse.shape[:2] and centre == pytest.approx(fine_centre, abs=1.5)    # centres stay in box pixels


def test_the_step_of_a_picture_follows_the_size_it_is_shown_at():
    assert F.step_for(2044, 160) == 13 and F.step_for(2044, 607) == 3 and F.step_for(300, 607) == 1 and F.step_for(2044, 2044) == 1
    assert F.step_for(2118, 1213) == 2 and F.step_for(1158, 1213) == 1        # the nearest whole step: at most 1.5 times enlarged on screen


def test_the_film_strip_asks_for_pictures_no_finer_than_its_tiles(tracks):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    film = F.MicrocolonyFilm(tracks, lambda k: FRAMES[k], PX); film.set_root(1); asked = []
    plain = film.picture
    film.picture = lambda t, step=1: asked.append(step) or plain(t, step)
    fig = Figure(figsize=(3.3, 1.25), dpi=100); FigureCanvasAgg(fig)
    F.draw_film_strip(fig, film, 5, 2)
    side = film.box[1] - film.box[0]
    assert len(asked) == 5 and set(asked) == {F.step_for(side, F.STRIP_TILE_PX)}


def test_the_window_frame_asks_for_a_picture_as_fine_as_it_shows_and_an_export_for_the_finest(tracks):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    film = F.MicrocolonyFilm(tracks, lambda k: FRAMES[k], PX); film.set_root(1); asked = []
    plain = film.picture
    film.picture = lambda t, step=1: asked.append(step) or plain(t, step)
    film.box = (0, 400, 0, 400)
    fig = Figure(figsize=(2.0, 0.92), dpi=100); FigureCanvasAgg(fig)          # a small figure: the image is 0.37 x 200 = 74 px wide
    film.draw(fig, 2); film.draw(fig, 2, finest=True)
    assert asked == [F.step_for(400, 74), 1] and asked[0] == 5


def test_picking_another_microcolony_does_not_read_the_frames_again():
    long = [blocks(20)] * 8 + [blocks(3)] * 8                                  # 16 frames: more than the film keeps in memory
    followed = MF.follow_microcolonies(lambda t: long[t], range(16), pixel_um=PX, dt_h=0.5, start_hour=24.0)
    reads = []
    film = F.MicrocolonyFilm(followed, lambda k: reads.append(k) or long[k], PX)
    film.set_root(1); first = len(reads); box_of_1 = film.box
    film.set_root(2); film.set_root(1)
    assert first == 16 and len(reads) == first and film.box == box_of_1


def test_the_extent_of_every_microcolony_is_worked_out_while_tracking_and_kept_with_a_saved_session(tracks, tmp_path):
    from nd2_analyzer.ui.biofilms.stages.microcolony_controller import FollowWorker
    worker = FollowWorker(lambda t: FRAMES[t], range(6), PX, 0.5, 24.0)
    worker.run()                                                    # in the tool this runs off the window's thread
    kept = worker.drawn._extents
    assert set(kept) == {1, 2} and sorted(kept[2]) == [0, 1] and len(kept[1]) == 6          # microcolony 2 is itself for two frames
    reads = []
    film = F.MicrocolonyFilm(worker.drawn, lambda k: reads.append(k) or FRAMES[k], PX); film.set_root(1)
    assert reads == []                                              # the film does not read the frames to find its box
    MF.save_tracks(worker.drawn, tmp_path / "t.npz"); back = MF.load_tracks(tmp_path / "t.npz")
    assert back._extents == kept
    plain = MF.follow_microcolonies(lambda t: FRAMES[t], range(6), pixel_um=PX, dt_h=0.5, start_hour=24.0)
    MF.save_tracks(plain, tmp_path / "p.npz")
    assert getattr(MF.load_tracks(tmp_path / "p.npz"), "_extents", None) is None            # none saved, none made up


def test_a_saved_image_is_drawn_with_every_pixel(tracks, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from nd2_analyzer.analysis.biofilm_stages import microcolony_plots as MP
    from nd2_analyzer.ui.biofilms.stages.microcolony_window import MicrocolonyDialog
    film = F.MicrocolonyFilm(tracks, lambda k: FRAMES[k], PX); film.set_root(1)
    dlg = MicrocolonyDialog(film, MP.entries(tracks), root=1)
    film.box = (0, 2000, 0, 2000); asked = []; plain = film.picture                # a full field: on screen the picture is coarser
    film.picture = lambda t, step=1: asked.append(step) or plain(t, step)
    dlg.redraw(); on_screen = asked[-1]
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "m.png"), "PNG (*.png)")))
    dlg.save_image()
    assert on_screen > 1 and asked[-2:] == [1, on_screen] and (tmp_path / "m.png").exists()   # every pixel for the file, then the screen again
