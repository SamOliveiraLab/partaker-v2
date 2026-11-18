"""The film of one followed microcolony, as the lineage window shows a family: the microcolony and those that merge into it
in the image on the left (one colour each until it merges), the merging history drawing itself on the right, the cells it
holds under it. Plain matplotlib, shared by the Microcolonies window, the page's film strip, the export and the tests."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
from matplotlib.colors import to_rgb
from scipy import ndimage as ndi

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
        self.show_stain = False; self._counted: Dict[int, np.ndarray] = {}; self._stain_series = None

    def set_stain(self, counted_at: Optional[Callable[[int], np.ndarray]]):
        """Where the matrix stain comes from (the Maturation tab's measure of this position); None when there is none."""
        self.counted_at = counted_at; self._counted = {}; self._stain_series = None
        if counted_at is None:
            self.show_stain = False

    def counted(self, t: int) -> np.ndarray:
        if t not in self._counted:
            if len(self._counted) > 120:
                self._counted.pop(next(iter(self._counted)))
            self._counted[t] = np.asarray(self.counted_at(t), bool)
        return self._counted[t]

    def stain_series(self):
        """(hours, % of the followed microcolony's area that is matrix stain) over the frames of the film."""
        if self._stain_series is None:
            cover = []
            for t in self.frames:
                ids = self.tracks.identity_map(t); mine = np.isin(ids, [i for i in self.order if self.owner(i, t) == self.root]); oy, ox = self.tracks.offsets[t]
                c = self.counted(t); on = np.zeros(ids.shape, bool); on[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
                cover.append(100.0 * float((on & mine).sum()) / max(int(mine.sum()), 1))
            self._stain_series = (np.array([self.tracks.hours[t] for t in self.frames]), np.array(cover))
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
        self._cells_series = None; self._stain_series = None

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
        fam = set(self.order); y0 = x0 = 10 ** 9; y1 = x1 = 0
        for t in self.frames:
            lab = self.cells(t); lut = np.isin(self._identity_of_labels(t, int(lab.max())), list(fam)); ys, xs = np.nonzero(lut[lab])
            if ys.size:
                oy, ox = self.tracks.offsets[t]
                y0, y1, x0, x1 = min(y0, ys.min() + oy), max(y1, ys.max() + oy), min(x0, xs.min() + ox), max(x1, xs.max() + ox)
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
        """The lasting microcolony under a point of the picture (box pixels), None on the background or a lone cell."""
        owners, _lab = self.owners_image(t); xi, yi = int(round(x)), int(round(y))
        if not (0 <= yi < owners.shape[0] and 0 <= xi < owners.shape[1]) or owners[yi, xi] == 0:
            return None
        return int(abs(owners[yi, xi]))

    # ------------------------------------------------------------------ drawing
    def picture(self, t: int):
        """(rgb of the box, [(soft outline, colour, identity)]) at frame t."""
        owners, lab = self.owners_image(t); rgb = np.zeros((*lab.shape, 3)); rgb[lab > 0] = to_rgb(LONE); rgb[owners < 0] = to_rgb(OTHER)
        lines = []; r_px = self.settings.gap_um / 2 / self.pixel_um
        for r in [int(v) for v in np.unique(owners) if v > 0]:
            mine = owners == r; rgb[mine] = to_rgb(self.palette.get(r, OTHER))
            grown = ndi.distance_transform_edt(~mine) <= r_px
            soft = ndi.gaussian_filter(grown.astype(np.float32), 0.45 / self.pixel_um)
            shade = np.clip(soft * 1.6, 0, 1)[..., None] * 0.30; back = (lab == 0)[..., None]
            rgb = np.where(back, rgb * (1 - shade) + np.array(to_rgb(self.palette.get(r, OTHER))) * shade, rgb)
            lines.append((soft, self.palette.get(r, OTHER), r, ndi.center_of_mass(mine)))
        if self.show_stain and self.counted_at is not None:               # the cells dimmed, the matrix stain in green over them
            y0, y1, x0, x1 = self.box; oy, ox = self.tracks.offsets[t]; c = self.counted(t); on = np.zeros((y1 - y0, x1 - x0), np.float32)
            fy0, fy1, fx0, fx1 = max(y0 - oy, 0), min(y1 - oy, c.shape[0]), max(x0 - ox, 0), min(x1 - ox, c.shape[1])
            if fy1 > fy0 and fx1 > fx0:
                on[fy0 + oy - y0:fy1 + oy - y0, fx0 + ox - x0:fx1 + ox - x0] = c[fy0:fy1, fx0:fx1]
            a = ndi.gaussian_filter(on, 1.5)[..., None] * 0.62; rgb = rgb * 0.6 * (1 - a) + np.array(to_rgb(STAIN_GREEN)) * a
        return rgb, lines

    def scale_um(self) -> float:
        width_um = (self.box[3] - self.box[2]) * self.pixel_um
        return max((u for u in (1, 2, 5, 10, 20, 50) if u <= width_um / 4), default=1)

    def cells_series(self):
        if getattr(self, "_cells_series", None) is None:
            self._cells_series = (np.array([self.tracks.hours[t] for t in self.frames]), np.array([self.cells_now(t) for t in self.frames]))
        return self._cells_series

    @SP.with_house_style
    def draw(self, fig, t: int):
        """One frame of the film. Returns (image axes, history axes)."""
        import matplotlib.patheffects as pe
        fig.clear(); hours = self.tracks.hours; now = hours[t]; rgb, lines = self.picture(t); h, w = rgb.shape[:2]
        aspect = fig.get_figwidth() / fig.get_figheight(); ax = fig.add_axes([0.03, 0.05, 0.37, 0.37 * aspect])
        ax.imshow(rgb, interpolation="antialiased"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        for soft, colour, r, (cy, cx) in lines:
            ax.contour(soft, levels=[0.22], colors=[colour], linewidths=1.2)
            ax.text(cx, cy, str(r), color=TAG_GREEN, fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="black")])
        um = self.scale_um(); L = um / self.pixel_um
        ax.add_patch(__import__("matplotlib").patches.Rectangle((w * 0.03, h * 0.86), L + w * 0.06, h * 0.11, fc="black", ec="none", zorder=4))
        ax.plot([w * 0.06, w * 0.06 + L], [h * 0.945, h * 0.945], color="white", lw=2.5, solid_capstyle="butt", zorder=5)
        ax.text(w * 0.06 + L / 2, h * 0.93, f"{um:g} µm", color="white", fontsize=9, ha="center", va="bottom", zorder=5)
        ax.set_xlim(0, w); ax.set_ylim(h, 0)
        # the small whole-field picture, top left: where the box is
        Hc, Wc = self.tracks.canvas_shape; step = max(1, int(round(max(Hc, Wc) / 256)))
        mini = (self._on_canvas(t, (0, Hc, 0, Wc))[::step, ::step] > 0).astype(float); mw = 0.22; mh = mw * mini.shape[0] / mini.shape[1]
        mi = ax.inset_axes([0.015, 1 - 0.015 - mh, mw, mh]); mi.set_zorder(7)
        mi.imshow(mini, cmap="gray", vmin=0, vmax=1, interpolation="antialiased"); mi.set_xticks([]); mi.set_yticks([])
        for s in mi.spines.values():
            s.set_color("#bdbdbd"); s.set_linewidth(0.9)
        y0, y1, x0, x1 = self.box
        mi.add_patch(__import__("matplotlib").patches.Rectangle((x0 / step, y0 / step), (x1 - x0) / step - 1, (y1 - y0) / step - 1, fc="none", ec="#f2c14e", lw=1.2))
        n_merge = len(self.order) - 1
        title = f"microcolony {self.root}" + (f" and the {n_merge} that merge into it" if n_merge else "")
        fig.text(0.03, 0.955, f"{title}   frame {t}   {now:g} h", fontsize=10.5, fontweight="bold", color=SP.INK, va="center")
        # the merging history, drawn up to now
        ax2 = fig.add_axes([0.50, 0.47, 0.46, 0.34]); SP.open_frame(ax2); first_h, last_h = hours[self.frames[0]], hours[self.frames[-1]]
        ident = self.tracks.identities; pad = max((last_h - first_h) * 0.06, 0.5)
        for i in self.order:
            b0 = hours[ident[i]["birth"]]; joined = i in self.into and self.into[i][0] in self.row
            end = hours[self.into[i][1]] if joined else last_h + pad
            if b0 <= now:
                ax2.plot([b0, min(end, now)], [self.row[i], self.row[i]], color=self.palette[i], lw=6, solid_capstyle="butt", zorder=3)
                ax2.text(b0 - pad * 0.15, self.row[i], str(i), ha="right", va="center", fontsize=9, color=SP.INK)
                if joined and end <= now:
                    ax2.plot([end, end], [self.row[i], self.row[self.into[i][0]]], color=SP.INK, lw=1.4, zorder=2)
        ax2.axvline(now, color="#9a9a9a", lw=1.2, zorder=1); ax2.set_ylim(len(self.order) - 0.4, -0.6); ax2.set_yticks([]); ax2.spines["left"].set_visible(False)
        ax2.set_xlim(first_h - pad, last_h + pad); ax2.tick_params(labelbottom=False)
        n_now = self.microcolonies_now(t)
        fig.text(0.50, 0.955, "merging history", fontsize=10.5, fontweight="bold", color=SP.INK, va="center")
        fig.text(0.50, 0.885, f"{SP_words(n_now)} now; each bar ends where it merges", fontsize=9, color="#444444", va="center")
        # the cells it holds
        ax3 = fig.add_axes([0.50, 0.15, 0.46, 0.21]); SP.open_frame(ax3)
        if self.show_stain and self.counted_at is not None:
            hs, cover = self.stain_series(); k = hs <= now; ax3.plot(hs[k], cover[k], color=STAIN_GREEN, lw=2)
            ax3.set_ylim(0, max(float(cover.max()), 1.0) * 1.12); ax3.set_ylabel("matrix stain\ninside it (%)")
        else:
            hs, ns = self.cells_series(); k = hs <= now; ax3.plot(hs[k], ns[k], color=FOLLOWED, lw=2)
            ax3.set_ylim(0, max(int(ns.max()), 1) * 1.12); ax3.set_ylabel("cells in it")
        ax3.axvline(now, color="#9a9a9a", lw=1.2, zorder=1); ax3.set_xlim(first_h - pad, last_h + pad); ax3.set_xlabel("time after inoculation (h)")
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
def draw_film_strip(fig, film: Optional[MicrocolonyFilm], n: int = 5, now: Optional[int] = None):
    """A few frames of the followed microcolony's film side by side, the hour over each; the one at the cursor in bold."""
    fig.clear()
    if film is None or film.root is None or not film.frames:
        SP.empty_axes(fig, "Pick a microcolony and click Follow to see its film"); return []
    picks = sorted({film.frames[int(round(k))] for k in np.linspace(0, len(film.frames) - 1, min(n, len(film.frames)))})
    near = min(picks, key=lambda t: abs(t - now)) if now is not None else None
    axes = []; w = 0.92 / len(picks); aspect = fig.get_figwidth() / fig.get_figheight(); side = min(w - 0.012, 0.66 / aspect)
    for k, t in enumerate(picks):
        rgb, lines = film.picture(t); step = max(1, rgb.shape[0] // 160)
        ax = fig.add_axes([0.04 + k * w + (w - side) / 2, 0.08, side, side * aspect]); ax.imshow(rgb[::step, ::step], interpolation="antialiased")
        for soft, colour, _r, _c in lines:
            ax.contour(soft[::step, ::step], levels=[0.22], colors=[colour], linewidths=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{film.tracks.hours[t]:g} h", fontsize=8.5, fontweight="bold" if t == near else "normal", pad=3); axes.append(ax)
    return axes
