"""The film of one followed microcolony, as the lineage window shows a family: the microcolony and those that merge into it
in the image on the left (one colour each until it merges), the merging history drawing itself on the right, the cells it
holds under it. Plain matplotlib, shared by the Microcolonies window, the page's film strip, the export and the tests."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
from matplotlib.colors import to_rgb
from scipy import ndimage as ndi

from . import figure_style as FS
from . import seeding_plots as SP
from .microcolony_follow import MicrocolonyTracks
from .microcolony_plots import FOLLOWED, merged_into, subtree
from .microcolony_rules import MicrocolonySettings, drop_small_cells

FIGURE_IN = (10.0, 4.6)       # the window's figure, as the lineage window's
PALETTE = ("#56b4e9", "#e3b23c", "#e07aa4", "#4477AA", "#EE6677", "#35b6b4", "#AA3377", "#7fb069", "#c9a227", "#5c6bc0", "#d1813b", "#6f8fd6")
LONE, OTHER = "#3c3c3c", "#707070"                     # a cell in no microcolony; a cell of a microcolony that is not followed
TAG_GREEN = "#39ff39"
STAIN_GREEN = "#1fbf3f"                                # the matrix stain, as in Figure 4
PAD_UM = 3.0
REACH_UM = 0.5                 # a click this near a microcolony (about one cell width) is a click on it
STRIP_TILE_PX = 240            # a tile of the page's film strip is drawn about this many pixels wide (two per point on a fine screen)
IMAGE_SHARE = 0.37             # the picture takes this share of the width of the window's figure


FULL_BARS = 16                 # up to this many bars the merging history draws them 6 points thick


def bar_width(n_bars: int) -> float:
    """The thickness of a bar of the merging history, thinner when there are many."""
    return 6.0 if n_bars <= FULL_BARS else max(1.2, 6.0 * FULL_BARS / n_bars)


def names_to_draw(rows: Dict[int, int], alive, row_px: float, name_px: float, always: Optional[int] = None) -> List[int]:
    """The bars of the merging history that get their name written, top to bottom. With room for every name (a row at
    least as tall as a name) all are named. In a crowded history only the microcolonies that are still themselves now
    (alive) are, each only if it clears the names already kept; `always` (the followed one) keeps its name."""
    if row_px >= name_px:
        return sorted(rows, key=rows.get)
    need = name_px / row_px; kept = [always] if always is not None and always in rows else []
    for i in sorted((i for i in rows if i in alive and i != always), key=rows.get):
        if all(abs(rows[i] - rows[k]) >= need for k in kept):
            kept.append(i)
    return sorted(kept, key=rows.get)


def heading_words(root: int, drawn: int, later: int) -> str:
    """The heading of the film: the followed microcolony and how many others the picture shows merging into it at this
    moment (the bars drawn in the merging history), with those that come later named apart."""
    if not drawn:
        return f"Microcolony {root}" + (f" ({later} merge{'s' if later == 1 else ''} into it later)" if later else "")
    return f"Microcolony {root} and the {drawn} that merge{'s' if drawn == 1 else ''} into it" + (f" ({later} more later)" if later else "")


def step_for(side_px: int, shown_px: float) -> int:
    """Every how many pixels of a box side_px wide to draw, for a picture shown shown_px wide: about as fine as it is
    shown (the nearest whole step, so a picture is enlarged at most 1.5 times on screen)."""
    return max(1, int(round(side_px / max(float(shown_px), 1.0))))


def extents(tracks: MicrocolonyTracks, labels_at: Callable[[int], np.ndarray]) -> Dict[int, Dict[int, tuple]]:
    """identity -> frame -> (y0, y1, x0, x1): the box on the canvas that holds its cells at that frame. Read once for all
    the frames and kept with the tracks, so picking another microcolony does not read the frames again."""
    kept = getattr(tracks, "_extents", None)
    if kept is None:
        kept = {}
        for t in tracks.frames:
            lab = np.asarray(labels_at(t)); lut = np.zeros(int(lab.max()) + 1, np.int32); oy, ox = tracks.offsets[t]
            for label, identity in tracks.cell_identity.get(t, {}).items():
                if label < lut.size:
                    lut[label] = identity
            for identity, where in enumerate(ndi.find_objects(lut[lab]), 1):
                if where is not None:
                    kept.setdefault(identity, {})[t] = (where[0].start + oy, where[0].stop - 1 + oy, where[1].start + ox, where[1].stop - 1 + ox)
        tracks._extents = kept
    return kept


class MicrocolonyFilm:
    def __init__(self, tracks: MicrocolonyTracks, labels_at: Callable[[int], np.ndarray], pixel_um: float,
                 settings: MicrocolonySettings = MicrocolonySettings()):
        self.tracks, self.labels_at, self.pixel_um, self.settings = tracks, labels_at, float(pixel_um), settings
        self.lasting = tracks.lasting()
        self.into = {int(e["identity"]): (int(e["into"]), int(e["time"])) for e in tracks.merges()}
        self.kids = merged_into(tracks)
        self.root: Optional[int] = None; self.order: List[int] = []; self.frames: List[int] = []; self.box = (0, 1, 0, 1)
        self._cells: Dict[int, np.ndarray] = {}
        self.counted_at: Optional[Callable[[int], np.ndarray]] = None      # frame -> the matrix mask of the frame, when it has been measured
        self.show_stain = False; self._counted: Dict[int, np.ndarray] = {}; self._stain_series = None; self._stain_rows = None
        self._small: Dict[tuple, tuple] = {}                               # the coarse pictures of the film strip, kept

    def set_stain(self, counted_at: Optional[Callable[[int], np.ndarray]], rows: Optional[list] = None, show: Optional[bool] = None):
        """Where the matrix stain comes from (the Maturation tab's measure of this position); None when there is none.
        rows (matrix_stain.per_microcolony of the same tracks): the stain of every microcolony is known already, so
        the curve needs no measuring. show: also switch the stain on or off in the picture."""
        self.counted_at = counted_at; self._counted = {}; self._stain_series = None; self._small = {}
        self._stain_rows = None if rows is None or counted_at is None else {(int(r["identity"]), int(r["time"])): 100.0 * float(r["cover"]) for r in rows}
        if show is not None:
            self.show_stain = bool(show)
        if counted_at is None:
            self.show_stain = False

    def counted(self, t: int) -> np.ndarray:
        if t not in self._counted:
            if len(self._counted) > 120:
                self._counted.pop(next(iter(self._counted)))
            self._counted[t] = np.asarray(self.counted_at(t), bool)
        return self._counted[t]

    def stain_ready(self) -> bool:
        """The matrix stain of every frame of this film is measured (stain_series answers at once)."""
        return self._stain_series is not None or self._stain_rows is not None

    def measure_stain(self):
        """Measures the matrix stain inside the followed microcolony frame by frame, giving (frames done, frames in
        all) after each one, so a window can say how far it is. Nothing to do when it is measured already."""
        if self._stain_series is not None or self.counted_at is None:
            return
        if self._stain_rows is not None:                    # measured for every microcolony of the position already
            self._stain_series = (np.array([self.tracks.hours[t] for t in self.frames]), np.array([self._stain_rows.get((self.root, t), 0.0) for t in self.frames]))
            return
        root, frames, source = self.root, list(self.frames), self.counted_at; cover = []
        changed = lambda: self.root != root or self.frames != frames or self.counted_at is not source or self._stain_series is not None
        for k, t in enumerate(frames, 1):
            if changed():
                return                          # another microcolony is followed now: this measure is not its curve
            c = self.counted(t)
            if changed():
                return
            ids = self.tracks.identity_map(t); mine = np.isin(ids, [i for i in self.order if self.owner(i, t) == root]); oy, ox = self.tracks.offsets[t]
            on = np.zeros(ids.shape, bool); on[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
            cover.append(100.0 * float((on & mine).sum()) / max(int(mine.sum()), 1))
            if k == len(frames):
                self._stain_series = (np.array([self.tracks.hours[f] for f in frames]), np.array(cover))
            yield k, len(frames)

    def stain_series(self):
        """(hours, % of the followed microcolony's area that is matrix stain) over the frames of the film."""
        for _step in self.measure_stain():
            pass
        return self._stain_series

    # ------------------------------------------------------------------ which microcolony
    def default_root(self) -> Optional[int]:
        """The microcolony most others merged into (the richest history), else any lasting one."""
        if not self.lasting:
            return None
        return max(sorted(self.lasting), key=lambda i: len(subtree(self.tracks, i)))

    def set_root(self, root: int):
        self.root = int(root); self.order = self._layout(self.root); self.row = {i: k for k, i in enumerate(self.order)}
        self.palette = {self.root: FOLLOWED}
        for k, i in enumerate(i for i in self.order if i != self.root):
            self.palette[i] = PALETTE[k % len(PALETTE)]
        ident = self.tracks.identities; first = min(ident[i]["birth"] for i in self.order)
        last = self.into[self.root][1] - 1 if self.root in self.into else self.tracks.frames[-1]   # its last frame as itself
        last = max(last, first)
        self.frames = [t for t in self.tracks.frames if first <= t <= last]
        self.box = self._box()
        self._cells_series = None; self._stain_series = None; self._small = {}

    def _layout(self, r: int) -> List[int]:
        above, below = [], []
        for n, (c, _hour) in enumerate(sorted(self.kids.get(r, []), key=lambda v: v[1])):
            block = self._layout(c)
            if n % 2 == 0:
                above = block + above
            else:
                below = below + block
        return above + [r] + below

    def owner(self, identity: int, t: int) -> int:
        """The microcolony this one is part of at frame t (itself until it merges)."""
        i = int(identity)
        while i in self.into and self.into[i][1] <= t:
            i = self.into[i][0]
        return i

    def microcolonies_now(self, t: int) -> int:
        return len({self.owner(i, t) for i in self.order if self.tracks.identities[i]["birth"] <= t})

    def cells_now(self, t: int) -> int:
        """Cells of the followed microcolony at frame t, with those merged into it so far."""
        mine = {i for i in self.order if self.owner(i, t) == self.root}
        return sum(1 for i in self.tracks.cell_identity.get(t, {}).values() if i in mine)

    # ------------------------------------------------------------------ pixels
    def cells(self, t: int) -> np.ndarray:
        if t not in self._cells:
            if len(self._cells) > 12:
                self._cells.pop(next(iter(self._cells)))
            self._cells[t] = drop_small_cells(np.asarray(self.labels_at(t)), self.pixel_um, self.settings.min_cell_um2)
        return self._cells[t]

    def _on_canvas(self, t: int, box) -> np.ndarray:
        """The cell labels of frame t inside a canvas box (0 where the frame does not reach)."""
        y0, y1, x0, x1 = box; oy, ox = self.tracks.offsets[t]; lab = self.cells(t); H, W = lab.shape
        out = np.zeros((y1 - y0, x1 - x0), np.int32)
        fy0, fy1, fx0, fx1 = max(y0 - oy, 0), min(y1 - oy, H), max(x0 - ox, 0), min(x1 - ox, W)
        if fy1 > fy0 and fx1 > fx0:
            out[fy0 + oy - y0:fy1 + oy - y0, fx0 + ox - x0:fx1 + ox - x0] = lab[fy0:fy1, fx0:fx1]
        return out

    def _identity_of_labels(self, t: int, n: int) -> np.ndarray:
        lut = np.zeros(n + 1, np.int64)
        for label, identity in self.tracks.cell_identity.get(t, {}).items():
            if label <= n:
                lut[label] = identity
        return lut

    def _box(self):
        """A square on the canvas that holds the followed microcolonies in every frame of the film, with a margin."""
        fam = set(self.order); y0 = x0 = 10 ** 9; y1 = x1 = 0; where = extents(self.tracks, self.labels_at); film = set(self.frames)
        for i in fam:
            for t, (a0, a1, b0, b1) in where.get(i, {}).items():
                if t in film:
                    y0, y1, x0, x1 = min(y0, a0), max(y1, a1), min(x0, b0), max(x1, b1)
        Hc, Wc = self.tracks.canvas_shape
        if y1 <= y0:
            return (0, Hc, 0, Wc)
        pad = int(PAD_UM / self.pixel_um); side = min(max(y1 - y0, x1 - x0) + 2 * pad, Hc, Wc); cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        ya = int(np.clip(cy - side // 2, 0, Hc - side)); xa = int(np.clip(cx - side // 2, 0, Wc - side))
        return (ya, ya + side, xa, xa + side)

    def owners_image(self, t: int) -> np.ndarray:
        """Per pixel of the box: the followed microcolony the cell is in, -1 for a cell of another microcolony, 0 otherwise.
        Cells in no microcolony are told apart from the background by the labels (cells_in_box)."""
        lab = self._on_canvas(t, self.box); lut_id = self._identity_of_labels(t, int(lab.max())); fam = set(self.order)
        lut = np.zeros(lut_id.size, np.int64)
        for label in np.unique(lab):
            i = int(lut_id[label])
            if label and i:
                lut[label] = self.owner(i, t) if i in fam else (-i if i in self.lasting else 0)
        return lut[lab], lab

    def identity_at(self, t: int, x: float, y: float) -> Optional[int]:
        """The lasting microcolony at a point of the picture (box pixels): the one whose area holds the point (its cells
        and the gaps between them), else the nearest within REACH_UM; None further away. For a microcolony of the
        followed family that has merged by frame t, the one it merged into."""
        ids = self.tracks.identity_map(t); height, width = ids.shape; reach = int(round(REACH_UM / self.pixel_um))
        yc, xc = int(round(y)) + self.box[0], int(round(x)) + self.box[2]
        y0, y1, x0, x1 = max(yc - reach, 0), min(yc + reach + 1, height), max(xc - reach, 0), min(xc + reach + 1, width)
        if y1 <= y0 or x1 <= x0:
            return None
        near = ids[y0:y1, x0:x1]; fam = set(self.order); ys, xs = np.nonzero(np.isin(near, list(fam | set(self.lasting))))
        if not ys.size:
            return None
        far = (ys + y0 - yc) ** 2 + (xs + x0 - xc) ** 2; k = int(np.argmin(far))
        if far[k] > reach ** 2:
            return None
        i = int(near[ys[k], xs[k]])
        return int(self.owner(i, t)) if i in fam else i

    # ------------------------------------------------------------------ drawing
    def picture(self, t: int, step: int = 1):
        """(rgb of the box, [(soft outline, colour, identity, centre in box pixels)]) at frame t. step: every step-th
        pixel of the box, for a picture that is shown small (the film strip, the window on a full field). The outline of
        each microcolony is worked out in its own corner of the box, which gives the same picture as the whole box."""
        step = max(1, int(step)); key = (int(t), step, bool(self.show_stain))
        if key in self._small:
            return self._small[key]
        owners, lab = self.owners_image(t)
        if step > 1:
            owners, lab = owners[::step, ::step], lab[::step, ::step]
        rgb = np.zeros((*lab.shape, 3)); rgb[lab > 0] = to_rgb(LONE); rgb[owners < 0] = to_rgb(OTHER)
        px = self.pixel_um * step; r_px = self.settings.gap_um / 2 / px; sigma = 0.45 / px; lines = []
        margin = int(np.ceil(r_px + 4 * sigma)) + 2                       # beyond it the soft outline is exactly zero
        for r, where in enumerate(ndi.find_objects(np.where(owners > 0, owners, 0).astype(np.int32)), 1):
            if where is None:
                continue
            y0, y1 = max(where[0].start - margin, 0), min(where[0].stop + margin, lab.shape[0])
            x0, x1 = max(where[1].start - margin, 0), min(where[1].stop + margin, lab.shape[1]); part = (slice(y0, y1), slice(x0, x1))
            mine = owners[part] == r; colour = np.array(to_rgb(self.palette.get(r, OTHER))); corner = rgb[part]; corner[mine] = colour
            grown = ndi.distance_transform_edt(~mine) <= r_px
            near = ndi.gaussian_filter(grown.astype(np.float32), sigma)
            shade = np.clip(near * 1.6, 0, 1)[..., None] * 0.30; back = (lab[part] == 0)[..., None]
            rgb[part] = np.where(back, corner * (1 - shade) + colour * shade, corner)
            soft = np.zeros(lab.shape, np.float32); soft[part] = near; cy, cx = ndi.center_of_mass(mine)
            lines.append((soft, self.palette.get(r, OTHER), r, ((cy + y0) * step, (cx + x0) * step)))
        if self.show_stain and self.counted_at is not None:               # the cells dimmed, the matrix stain in green over them
            y0, y1, x0, x1 = self.box; oy, ox = self.tracks.offsets[t]; c = self.counted(t); on = np.zeros((y1 - y0, x1 - x0), np.float32)
            fy0, fy1, fx0, fx1 = max(y0 - oy, 0), min(y1 - oy, c.shape[0]), max(x0 - ox, 0), min(x1 - ox, c.shape[1])
            if fy1 > fy0 and fx1 > fx0:
                on[fy0 + oy - y0:fy1 + oy - y0, fx0 + ox - x0:fx1 + ox - x0] = c[fy0:fy1, fx0:fx1]
            a = ndi.gaussian_filter(on[::step, ::step], 1.5 / step)[..., None] * 0.62; rgb = rgb * 0.6 * (1 - a) + np.array(to_rgb(STAIN_GREEN)) * a
        if step >= 4:                                                     # small: kept, so the strip does not work them out at every cursor move
            if len(self._small) > 60:
                self._small.pop(next(iter(self._small)))
            self._small[key] = (rgb, lines)
        return rgb, lines

    def scale_um(self) -> float:
        width_um = (self.box[3] - self.box[2]) * self.pixel_um
        return max((u for u in (1, 2, 5, 10, 20, 50) if u <= width_um / 4), default=1)

    def cells_series(self):
        if getattr(self, "_cells_series", None) is None:
            self._cells_series = (np.array([self.tracks.hours[t] for t in self.frames]), np.array([self.cells_now(t) for t in self.frames]))
        return self._cells_series

    @SP.with_house_style
    def draw(self, fig, t: int, finest: bool = False):
        """One frame of the film. Returns (image axes, history axes). The picture is drawn as fine as the figure shows
        it; finest: every pixel, for an image that is saved larger than it is shown."""
        import matplotlib.patheffects as pe
        fig.clear(); hours = self.tracks.hours; now = hours[t]; h, w = self.box[1] - self.box[0], self.box[3] - self.box[2]
        step = 1 if finest else step_for(w, IMAGE_SHARE * fig.get_figwidth() * fig.dpi)
        rgb, lines = self.picture(t, step)
        aspect = fig.get_figwidth() / fig.get_figheight(); ax = fig.add_axes([0.03, 0.05, IMAGE_SHARE, IMAGE_SHARE * aspect])
        ax.imshow(rgb, interpolation="antialiased", extent=(-0.5, w - 0.5, h - 0.5, -0.5)); ax.set_xticks([]); ax.set_yticks([])   # in box pixels
        for s in ax.spines.values():
            s.set_visible(False)
        for soft, colour, r, (cy, cx) in lines:
            ax.contour(np.arange(soft.shape[1]) * step + (step - 1) / 2, np.arange(soft.shape[0]) * step + (step - 1) / 2, soft,
                       levels=[0.22], colors=[colour], linewidths=1.2)
            ax.text(cx, cy, str(r), color=TAG_GREEN, fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="black")])
        um = self.scale_um(); L = um / self.pixel_um; edge = [pe.withStroke(linewidth=2.6, foreground="black")]
        # the scale bar as Figure 3 d: a white bar and white words on the picture itself, no box over the cells
        ax.plot([w * 0.05, w * 0.05 + L], [h * 0.955, h * 0.955], color="white", lw=2.5, solid_capstyle="butt", zorder=5, path_effects=edge)
        ax.text(w * 0.05 + L / 2, h * 0.94, f"{um:g} µm", color="white", fontsize=9, ha="center", va="bottom", zorder=5, path_effects=edge)
        ax.set_xlim(0, w); ax.set_ylim(h, 0)
        started = [i for i in self.order if hours[self.tracks.identities[i]["birth"]] <= now]     # the bars the merging history draws now
        drawn = len([i for i in started if i != self.root])
        fig.text(0.03, 0.955, f"{heading_words(self.root, drawn, len(self.order) - 1 - drawn)}   frame {t}   {now:g} h", fontsize=10.5, fontweight="bold",
                 color=SP.INK, va="center")
        # the merging history, drawn up to now
        ax2 = fig.add_axes([0.50, 0.47, 0.46, 0.34]); SP.open_frame(ax2); first_h, last_h = hours[self.frames[0]], hours[self.frames[-1]]
        ident = self.tracks.identities; pad = max((last_h - first_h) * 0.06, 0.5); thick = bar_width(len(self.order))
        merged_by_now = lambda i: i in self.into and self.into[i][0] in self.row and hours[self.into[i][1]] <= now
        alive = {i for i in self.order if hours[ident[i]["birth"]] <= now and not merged_by_now(i)}
        named = set(names_to_draw(self.row, alive, 0.34 * fig.get_figheight() * fig.dpi / max(len(self.order), 1), 9 * fig.dpi / 72 * 1.25, always=self.root))
        for i in self.order:
            b0 = hours[ident[i]["birth"]]; joined = i in self.into and self.into[i][0] in self.row
            end = hours[self.into[i][1]] if joined else last_h + pad
            if b0 <= now:
                ax2.plot([b0, min(end, now)], [self.row[i], self.row[i]], color=self.palette[i], lw=thick, solid_capstyle="butt", zorder=3)
                if i in named:
                    ax2.text(b0 - pad * 0.15, self.row[i], str(i), ha="right", va="center", fontsize=9, color=SP.INK)
                if joined and end <= now:
                    ax2.plot([end, end], [self.row[i], self.row[self.into[i][0]]], color=SP.INK, lw=1.4 if thick == 6.0 else 0.7, zorder=2)
        ax2.axvline(now, color=FS.CURSOR_RED, lw=1.2, zorder=6); ax2.set_ylim(len(self.order) - 0.4, -0.6); ax2.set_yticks([]); ax2.spines["left"].set_visible(False)
        ax2.set_xlim(first_h - pad, last_h + pad); ax2.set_xticks(FS.time_ticks(first_h, last_h)); ax2.tick_params(labelbottom=False)
        n_now = self.microcolonies_now(t)
        fig.text(0.50, 0.955, f"Merging history ({len(started)} microcolon{'y' if len(started) == 1 else 'ies'})", fontsize=10.5, fontweight="bold", color=SP.INK, va="center")
        fig.text(0.50, 0.885, f"{SP_words(n_now)} now; each bar ends where it merges", fontsize=9, color="#444444", va="center")
        # the cells it holds
        ax3 = fig.add_axes([0.50, 0.15, 0.46, 0.21]); SP.open_frame(ax3)
        if self.show_stain and self.counted_at is not None:
            hs, cover = self.stain_series(); k = hs <= now; ax3.plot(hs[k], cover[k], color=STAIN_GREEN, lw=2)
            ax3.set_ylim(0, max(float(cover.max()), 1.0) * 1.12); ax3.set_ylabel("Area covered (%)", fontweight="bold")     # Figure 4 d's name for the stain cover
        else:
            hs, ns = self.cells_series(); k = hs <= now; ax3.plot(hs[k], ns[k], color=FOLLOWED, lw=2)
            ax3.set_ylim(0, max(int(ns.max()), 1) * 1.12); ax3.set_ylabel("Cells", fontweight="bold")
        ax3.axvline(now, color=FS.CURSOR_RED, lw=1.2, zorder=6); ax3.set_xlim(first_h - pad, last_h + pad); ax3.set_xticks(FS.time_ticks(first_h, last_h))
        ax3.set_xlabel("Time (h)", fontweight="bold")
        # the field of view: a small picture of its own beside the film (not over it), the film's box in yellow
        Hc, Wc = self.tracks.canvas_shape; coarse = max(1, int(round(max(Hc, Wc) / 256)))
        mini = (self._on_canvas(t, (0, Hc, 0, Wc))[::coarse, ::coarse] > 0).astype(float); image = ax.get_position()
        mw = 0.065; mh = mw * aspect * mini.shape[0] / mini.shape[1]
        field = fig.add_axes([image.x1 + 0.012, image.y1 - mh - 0.06, mw, mh])
        field.imshow(mini, cmap="gray", vmin=0, vmax=1, interpolation="antialiased"); field.set_xticks([]); field.set_yticks([])
        field.set_title("field of view", fontsize=8, color=FS.G1, pad=3, fontweight="normal")
        for side in field.spines.values():
            side.set_color("#bdbdbd"); side.set_linewidth(0.9)
        y0, y1, x0, x1 = self.box
        field.add_patch(__import__("matplotlib").patches.Rectangle((x0 / coarse, y0 / coarse), (x1 - x0) / coarse - 1, (y1 - y0) / coarse - 1, fc="none", ec="#f2c14e", lw=1.2))
        return ax, ax2

    # ------------------------------------------------------------------ export
    def save_gif(self, path: str, progress: Optional[Callable[[int, int], bool]] = None, ms: int = 250):
        """The film as a GIF, one image per frame; progress(k, n) returning False stops it."""
        from PIL import Image
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        images = []
        for k, t in enumerate(self.frames):
            if progress is not None and progress(k, len(self.frames)) is False:
                return None
            fig = Figure(figsize=FIGURE_IN, dpi=100); FigureCanvasAgg(fig); self.draw(fig, t); fig.canvas.draw()
            im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); images.append(im.quantize(colors=160, method=Image.Quantize.MEDIANCUT))
        if not images:
            return None
        images += [images[-1]] * 6
        images[0].save(path, save_all=True, append_images=images[1:], duration=int(ms), loop=0, optimize=True)
        return path


def SP_words(n: int) -> str:
    return f"{n} microcolony" if n == 1 else f"{n} microcolonies"


@SP.with_house_style
def draw_film_strip(fig, film: Optional[MicrocolonyFilm], n: int = 6, now: Optional[int] = None, columns: int = 3):
    """A few frames of the followed microcolony's film in rows of `columns`, as large as the figure allows, the hour
    over each; the one at the cursor in bold."""
    fig.clear()
    if film is None or film.root is None or not film.frames:
        SP.empty_axes(fig, "Pick a microcolony and click Follow to see its film"); return []
    picks = sorted({film.frames[int(round(k))] for k in np.linspace(0, len(film.frames) - 1, min(n, len(film.frames)))})
    near = min(picks, key=lambda t: abs(t - now)) if now is not None else None
    axes = []; step = step_for(film.box[1] - film.box[0], STRIP_TILE_PX)
    for t, box in zip(picks, FS.tile_grid(fig, len(picks), columns if columns else len(picks), title_in=0.2)):
        rgb, lines = film.picture(t, step)
        ax = fig.add_axes(box); ax.imshow(rgb, interpolation="antialiased")
        for soft, colour, _r, _c in lines:
            ax.contour(soft, levels=[0.22], colors=[colour], linewidths=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{film.tracks.hours[t]:g} h", fontsize=8.5, fontweight="bold" if t == near else "normal", pad=3); axes.append(ax)
    return axes
