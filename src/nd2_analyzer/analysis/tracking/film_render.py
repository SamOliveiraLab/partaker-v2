"""Drawing of one family: the tracked image panel and the family tree, in the house style of the paper's Figure 3 and the
Week 88 lineage film. numpy + matplotlib (Agg) only, so the lineage window, the GIF export and the tests draw the same thing.

Colours: every cell in its tag colour (dimmed), the followed family at full colour with a white outline, a yellow trail,
a white dot and a green tag number; tree branches in the house red, elbows near-black, the time cursor grey.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import family as F

ACCENT = "#c0392b"
NEUTRAL = "#2b2b2b"
GREY = "#8a8a8a"
TRAIL = "#ffd54a"
TAG = "#00ff00"
ARIAL = ["Arial", "Helvetica", "DejaVu Sans"]


def id_colour(i: int) -> np.ndarray:
    """One fixed colour per tag (the notebook's rule), RGB in [0.35, 1]."""
    r = np.random.default_rng(int(i) * 7919 + 13).random(3)
    return r * 0.65 + 0.35


def tracked_panel(mask: np.ndarray, owner: Dict[int, int], fam: Sequence[int], sl: Tuple[slice, slice], dim: float = 0.55) -> np.ndarray:
    """RGB float image of the crop sl: every cell in its tag colour dimmed, the family's cells bright with a white outline,
    dark seams between cells, black background. Cells without a tag are grey."""
    from scipy.ndimage import binary_dilation
    from skimage.segmentation import find_boundaries
    m = np.asarray(mask)[sl]
    rgb = np.zeros((*m.shape, 3), np.float32)
    fam_set = {int(i) for i in fam}
    keep = np.zeros(m.shape, bool)
    labs = np.unique(m); labs = labs[labs > 0]
    for L in labs:
        L = int(L); sel = m == L
        tag = owner.get(L)
        if tag is None:
            rgb[sel] = np.array([0.35, 0.35, 0.35]) * dim
        elif tag in fam_set:
            rgb[sel] = id_colour(tag); keep |= sel
        else:
            rgb[sel] = id_colour(tag) * dim
    rgb[find_boundaries(m, mode="inner")] *= 0.35
    if keep.any():
        rgb[binary_dilation(find_boundaries(keep, mode="outer"), iterations=1)] = 1.0
    return np.clip(rgb, 0, 1)


def draw_panel(ax, mask, by_id, fam, t, sl, owner=None, trail_n: int = 8, pixel_um: Optional[float] = None, tag_size: float = 8):
    """The tracked panel into a matplotlib axes: image, trails, dots, tags, scale bar."""
    owner = F.owner_map(by_id, mask, t) if owner is None else owner
    ax.imshow(tracked_panel(mask, owner, fam, sl), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    x0, y0 = sl[1].start, sl[0].start
    for i in fam:
        pts = F.trail(by_id, i, t, trail_n)
        if not pts:
            continue
        xs = [p[0] - x0 for p in pts]; ys = [p[1] - y0 for p in pts]
        if len(pts) > 1:
            ax.plot(xs, ys, color=TRAIL, lw=1.4)
        ax.plot(xs[-1], ys[-1], "o", ms=4, color="white")
        ax.text(xs[-1] + 5, ys[-1] - 5, str(i), color=TAG, fontsize=tag_size, fontweight="bold")
    h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
    if pixel_um:
        from matplotlib.patches import Rectangle
        bar = 5.0 / pixel_um
        ax.add_patch(Rectangle((6, h - 12), bar, 3, color="white"))
        ax.text(6, h - 16, "5 um", color="white", fontsize=7)
    ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(h - 0.5, -0.5)


def draw_tree(ax, by_id, rows: Dict[int, float], hours: np.ndarray, upto: Optional[float] = None, lw: float = 3.5,
              label_size: float = 8, picked: Optional[int] = None):
    """The family tree: a red bar per track from its first to its last hour (cut at upto), elbows from a parent's end to its
    children's starts, tag numbers on bars longer than 0.2 h."""
    for i, y in rows.items():
        ts = [int(v) for v in by_id[i]["t"]]
        h0, h1 = float(hours[ts[0]]), float(hours[ts[-1]])
        if upto is not None and h0 > upto:
            continue
        he = h1 if upto is None else min(h1, upto)
        ax.plot([h0, he], [y, y], color=ACCENT, lw=lw * (1.6 if picked == i else 1.0), solid_capstyle="butt")
        if upto is None or he - h0 > 0.2:
            ax.text((h0 + he) / 2, y + 0.06, str(i), fontsize=label_size, ha="center", va="bottom")
        p = F.parent_of(by_id, i)
        if p is not None and p in rows:
            hp = float(hours[int(by_id[p]["t"][-1])])
            if upto is None or hp <= upto:
                ax.plot([hp, hp, h0], [rows[p], y, y], color=NEUTRAL, lw=1.2, solid_joinstyle="miter")
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("time (h)")
    if rows:
        ax.set_ylim(min(rows.values()) - 0.5, max(rows.values()) + 0.6)
    ax.set_xlim(-0.05 * float(hours[-1] or 1), float(hours[-1]) * 1.12 + 0.05)


def house_rc(scale: float = 1.0) -> dict:
    """Partaker 2's publication figure rules (headless_analysis/figure_style.apply_style, the DARPA Q9 figures), sized for a
    screen canvas: Arial, bold titles and axis labels, large numbers, no grid, open frame, ticks out. scale > 1 for export."""
    k = float(scale)
    # the family is given by name (not "sans-serif"): matplotlib resolves a generic family when it draws, with whatever the
    # application's rcParams say then, and the GUI's say DejaVu
    return {"font.family": ARIAL, "font.sans-serif": ARIAL, "axes.grid": False,
            "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.2 * k,
            "lines.linewidth": 2.0 * k, "font.size": 10 * k, "axes.titlesize": 11 * k, "axes.titleweight": "bold",
            "axes.labelsize": 10.5 * k, "axes.labelweight": "bold", "xtick.labelsize": 9.5 * k, "ytick.labelsize": 9.5 * k,
            "legend.fontsize": 9.5 * k, "legend.frameon": False,
            "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 4 * k, "ytick.major.size": 4 * k,
            "xtick.major.width": 1.2 * k, "ytick.major.width": 1.2 * k, "xtick.major.pad": 4 * k, "ytick.major.pad": 4 * k,
            "axes.titlepad": 8 * k, "axes.labelpad": 6 * k, "pdf.fonttype": 42, "ps.fonttype": 42,
            "figure.facecolor": "white", "axes.facecolor": "white"}


def collisions(fig, gap_in: float = 0.1, gap_same: float = 0.03):
    """Pairs of figure elements (texts, tick labels, axis titles, legends, images, plots) closer than gap_in inches, as in the
    DARPA Q9 figure check: a publication figure has none. Labels of one plot may sit gap_same from each other."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); dpi = fig.dpi; W, H = fig.get_size_inches(); items = []

    def add(name, art, kind, owner=None):
        if not art.get_visible() or (hasattr(art, "get_text") and not art.get_text().strip()):
            return
        bb = art.get_window_extent(r); items.append((name, (bb.x0 / dpi, bb.y0 / dpi, bb.x1 / dpi, bb.y1 / dpi), kind, owner))

    for t in fig.texts:
        add(f"text '{t.get_text()[:30]}'", t, "text")
    for ax in fig.axes:
        kind = "img" if ax.images or ax.get_facecolor()[:3] == (0.0, 0.0, 0.0) or not ax.axison else "plot"
        add(f"{kind} '{ax.get_title(loc='left') or ax.get_title()}'", ax, kind, ax)
        if kind == "img":
            add(f"title '{ax.get_title(loc='left')[:30]}'", ax._left_title, "label", ax)
            for t in ax.texts:                                    # call-outs on a picture or a drawn table still count
                add(f"call-out '{t.get_text()[:30]}'", t, "own", ax)
            continue
        for t in ax.texts:
            add(f"call-out '{t.get_text()[:30]}'", t, "own", ax)
        if ax.get_legend():
            add("legend", ax.get_legend(), "own", ax)
        for t in (ax.title, ax._left_title, ax.xaxis.label, ax.yaxis.label):
            add(f"title '{t.get_text()[:30]}'", t, "label", ax)
        # tick labels that matplotlib keeps outside the axis range are clipped away: they do not count
        x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim())
        for pos, t in zip(ax.get_xticks(), ax.get_xticklabels()):
            if x0 <= pos <= x1: add(f"tick '{t.get_text()}'", t, "label", ax)
        for pos, t in zip(ax.get_yticks(), ax.get_yticklabels()):
            if y0 <= pos <= y1: add(f"tick '{t.get_text()}'", t, "label", ax)
    out = []
    for i in range(len(items)):
        n1, b1, k1, o1 = items[i]
        for j in range(i + 1, len(items)):
            n2, b2, k2, o2 = items[j]
            same = o1 is not None and o1 is o2
            if same and ("plot" in (k1, k2) or "img" in (k1, k2)):
                continue                                   # a plot's own labels sit inside or beside it
            m = gap_same if same else gap_in
            if b1[0] < b2[2] + m and b2[0] < b1[2] + m and b1[1] < b2[3] + m and b2[1] < b1[3] + m:
                out.append(f"{n1}  x  {n2}")
    return out


class FamilyFilm:
    """One family's film: frame t -> (image panel, tree up to t). masks_at(t) gives the label frame of t."""

    def __init__(self, tracks: Sequence[dict], root: int, masks_at, shape: Tuple[int, int], hours: np.ndarray,
                 pixel_um: Optional[float] = None, half: int = 110, trail_n: int = 8):
        self.by_id = F.index_tracks(tracks)
        self.kids = F.children_of(self.by_id)
        self.root = int(root)
        self.fam = F.family(self.by_id, self.root, self.kids)
        self.rows = F.tree_rows(self.by_id, self.root, self.kids)
        self.masks_at = masks_at
        self.shape = shape
        self.hours = np.asarray(hours, float)
        self.pixel_um = pixel_um
        self.half = half
        self.trail_n = trail_n
        self.t_first, self.t_last = F.family_span(self.by_id, self.fam)

    def alive(self, t: int) -> int:
        return len(F.alive_at(self.by_id, self.fam, t))

    def crop(self, t: int):
        return F.camera_crop(self.by_id, self.fam, t, self.shape, self.half)

    def draw(self, fig, t: int, ax_img=None, ax_tree=None, ax_inset=None):
        """Draw frame t into the figure's axes (created when not given). Returns (ax_img, ax_tree)."""
        import matplotlib
        with matplotlib.rc_context(house_rc()):
            if ax_img is None or ax_tree is None:
                fig.clear()
                gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], left=0.03, right=0.98, top=0.90, bottom=0.14, wspace=0.12)
                ax_img = fig.add_subplot(gs[0]); ax_tree = fig.add_subplot(gs[1])
            else:
                ax_img.clear(); ax_tree.clear()
            mask = np.asarray(self.masks_at(t)); sl = self.crop(t)
            draw_panel(ax_img, mask, self.by_id, self.fam, t, sl, trail_n=self.trail_n, pixel_um=self.pixel_um)
            ax_img.set_title(f"cell {self.root} and {len(self.fam) - 1} descendants   frame {t}   {self.hours[t]:.2f} h", fontsize=9, loc="left")
            # where the camera is: the whole frame, small, top left of the image so it is read first (made anew: clearing
            # the axes removes it)
            ax_inset = ax_img.inset_axes([0.02, 0.70, 0.22, 0.28])
            ax_inset.imshow(mask > 0, cmap="gray", interpolation="nearest"); ax_inset.set_xticks([]); ax_inset.set_yticks([])
            from matplotlib.patches import Rectangle
            ax_inset.add_patch(Rectangle((sl[1].start, sl[0].start), sl[1].stop - sl[1].start, sl[0].stop - sl[0].start, fill=False, color=TRAIL, lw=1))
            for sp in ax_inset.spines.values():
                sp.set_color("#888888")
            draw_tree(ax_tree, self.by_id, self.rows, self.hours, upto=float(self.hours[t]))
            ax_tree.axvline(float(self.hours[t]), color=GREY, lw=1.2)
            n = self.alive(t)                                    # on the title row, where the time line never goes
            ax_tree.set_title(f"{n} cell{'s' if n != 1 else ''} alive", fontsize=9, loc="right")
        return ax_img, ax_tree

    def frame_image(self, t: int, size=(10, 4.2), dpi=100) -> np.ndarray:
        """Frame t as an RGB array (for the GIF)."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        fig = Figure(figsize=size, dpi=dpi); FigureCanvasAgg(fig)
        self.draw(fig, t)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    def save_gif(self, path, ms_per_frame: int = 350, progress=None) -> str:
        from PIL import Image
        frames = []
        for k, t in enumerate(range(self.t_first, self.t_last + 1)):
            frames.append(Image.fromarray(self.frame_image(t)))
            if progress is not None and progress(k + 1, self.t_last - self.t_first + 1) is False:
                break
        frames += [frames[-1]] * 6
        frames[0].save(str(path), save_all=True, append_images=frames[1:], duration=ms_per_frame, loop=0, optimize=False)
        return str(path)

    def draw_album(self, fig, max_stills: int = 7, half: int = 70):
        """The full tree with a still of the family at its start, at each division and at its end."""
        import matplotlib
        with matplotlib.rc_context(house_rc()):
            fig.clear()
            divs = F.divisions(self.by_id, self.fam, self.kids)
            picks = list(dict.fromkeys([self.t_first] + divs[: max_stills - 2] + [self.t_last]))
            n = len(picks)
            ax = fig.add_axes([0.05, 0.56, 0.92, 0.36])
            draw_tree(ax, self.by_id, self.rows, self.hours)
            for d in divs:
                ax.axvline(float(self.hours[d]), color="#dddddd", lw=0.8, zorder=0)
            w = 0.92 / n
            for k, t in enumerate(picks):
                a = fig.add_axes([0.05 + k * w, 0.09, w * 0.92, 0.30])
                mask = np.asarray(self.masks_at(t)); sl = F.camera_crop(self.by_id, self.fam, t, self.shape, half)
                draw_panel(a, mask, self.by_id, self.fam, t, sl, trail_n=1, pixel_um=None, tag_size=7)
                what = "start" if t == self.t_first and t not in divs else ("division" if t in divs else "end")
                a.set_title(f"{self.hours[t]:.2f} h  {what}", fontsize=8, loc="left")
            gens = F.generations(self.by_id, self.root, self.fam)
            idv = F.interdivision_frames(self.by_id, self.fam, self.kids)
            step_min = (self.hours[1] - self.hours[0]) * 60 if len(self.hours) > 1 else 0
            line = "   ".join(f"generation {g}: {len(v)} cells" for g, v in sorted(gens.items()))
            if idv and step_min:
                line += f"      interdivision time, median: {np.median(idv) * step_min:.0f} min"
            fig.text(0.05, 0.03, line, fontsize=8.5)
