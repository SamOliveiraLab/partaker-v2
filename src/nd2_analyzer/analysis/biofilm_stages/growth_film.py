"""The film of one position through the growth states, for the Maturation window and the page's film strip: the whole
frame on the left (phase contrast in grey, the microcolonies outlined in orange, the matrix stain in green), the number
of cells and the matrix stain of that position drawing themselves on the right. Plain matplotlib, no Qt."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
from matplotlib.colors import to_rgb

from . import growth_plots as GP
from . import matrix_stain as MX
from . import seeding_plots as SP
from .growth_measure import GrowthStates

FIGURE_IN = (10.0, 4.6)
PICTURE_PX = 600               # the side the frame is kept at for the film
OUTLINE, STAIN = "#E08A00", "#1fbf3f"


class GrowthFilm:
    def __init__(self, measured: GrowthStates, labels_at: Callable[[int, int], np.ndarray], phase_at: Callable[[int, int], np.ndarray],
                 dye_at: Optional[Callable[[int, int], np.ndarray]], reference: Optional[MX.DyeReference], pixel_um: float):
        self.gs, self.labels_at, self.phase_at, self.dye_at, self.reference, self.pixel_um = measured, labels_at, phase_at, dye_at, reference, float(pixel_um)
        self.position = measured.positions[0] if measured.positions else None; self.frames: List[int] = []
        self._pictures: Dict[tuple, tuple] = {}
        if self.position is not None:
            self.set_position(self.position)

    def set_position(self, position: int):
        self.position = int(position); self.frames = list(self.gs.frames[self.position])

    def hour(self, t: int) -> float:
        return float(self.gs.hours[self.position][self.frames.index(int(t))])

    def frame_at(self, hour: float) -> int:
        h = self.gs.hours[self.position]; return self.frames[int(np.argmin(np.abs(h - hour)))]

    def picture(self, t: int) -> np.ndarray:
        return self._picture(t)[0]

    def _picture(self, t: int):
        """(rgb, soft outline of the microcolonies or None), kept small."""
        key = (self.position, int(t))
        if key not in self._pictures:
            if len(self._pictures) > 200:
                self._pictures.pop(next(iter(self._pictures)))
            p = self.position; ph = np.asarray(self.phase_at(p, t), np.float32); lo, hi = np.percentile(ph, [0.5, 99.5])
            grey = np.clip((ph - lo) / max(hi - lo, 1e-6), 0, 1); rgb = np.repeat((0.18 + 0.72 * grey)[..., None], 3, -1)
            labels = np.asarray(self.labels_at(p, t))
            if self.dye_at is not None and self.reference is not None and p in self.gs.free_dye:
                counted, _single = MX.matrix(MX.stain(MX.correct(self.dye_at(p, t), self.reference), self.gs.free_dye[p], self.gs.settings), self.pixel_um, self.gs.settings)
                a = cv2.GaussianBlur(counted.astype(np.float32), (0, 0), 1.5)[..., None] * 0.62; rgb = rgb * (1 - a) + np.array(to_rgb(STAIN)) * a
            n = PICTURE_PX; h, w = rgb.shape[:2]; size = (n, max(1, int(round(n * h / w))))
            small = np.clip(cv2.resize(rgb.astype(np.float32), size, interpolation=cv2.INTER_AREA), 0, 1)
            colony = MX.microcolony_area(labels, self.pixel_um)
            lines = cv2.resize(cv2.GaussianBlur(colony.astype(np.float32), (0, 0), 0.45 / self.pixel_um), size, interpolation=cv2.INTER_AREA) if colony.any() else None
            self._pictures[key] = (small, lines)
        return self._pictures[key]

    @SP.with_house_style
    def draw(self, fig, t: int):
        """One frame of the film. Returns the image axes."""
        fig.clear(); p = self.position; gs = self.gs; hours = gs.hours[p]; k = self.frames.index(int(t)); now = float(hours[k])
        rgb, lines = self._picture(t); h, w = rgb.shape[:2]; aspect = fig.get_figwidth() / fig.get_figheight()
        ax = fig.add_axes([0.03, 0.05, 0.37, 0.37 * aspect * h / w]); ax.imshow(rgb, interpolation="antialiased"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if lines is not None:
            ax.contour(lines, levels=[0.5], colors=[OUTLINE], linewidths=0.9)
        um = max((u for u in (5, 10, 20, 50) if u <= w * self.pixel_um * (np.asarray(self.labels_at(p, t)).shape[1] / w) / 4), default=5)
        L = um / (self.pixel_um * np.asarray(self.labels_at(p, t)).shape[1] / w)
        ax.add_patch(__import__("matplotlib").patches.Rectangle((w * 0.03, h * 0.87), L + w * 0.06, h * 0.10, fc="black", ec="none", zorder=4))
        ax.plot([w * 0.06, w * 0.06 + L], [h * 0.945, h * 0.945], color="white", lw=2.5, solid_capstyle="butt", zorder=5)
        ax.text(w * 0.06 + L / 2, h * 0.93, f"{um:g} µm", color="white", fontsize=9, ha="center", va="bottom", zorder=5); ax.set_xlim(0, w); ax.set_ylim(h, 0)
        state = next((name for name, a, b in GP.states(gs) if a <= now <= b), "")
        fig.text(0.03, 0.955, f"position {p}   frame {t}   {now:g} h" + (f"   state {state}" if state else ""), fontsize=10.5, fontweight="bold", color=SP.INK, va="center")
        pad = max((hours[-1] - hours[0]) * 0.04, 0.5); seen = hours <= now + 1e-9
        ax2 = fig.add_axes([0.52, 0.52, 0.44, 0.30]); SP.open_frame(ax2)
        for name, a, b in GP.states(gs):
            ax2.axvspan(a, b, color=GP.STATE_SHADE[name], lw=0, zorder=0)
        ax2.plot(hours[seen], gs.cells[p][seen], color=SP.INK, lw=2); ax2.axvline(now, color="#9a9a9a", lw=1.2, zorder=1)
        ax2.set_xlim(hours[0] - pad, hours[-1] + pad); ax2.set_ylim(0, max(int(gs.cells[p].max()), 1) * 1.12); ax2.set_ylabel("cells"); ax2.tick_params(labelbottom=False)
        fig.text(0.52, 0.955, "number of cells", fontsize=10.5, fontweight="bold", color=SP.INK, va="center")
        fig.text(0.52, 0.885, f"{int(gs.cells[p][k]):,} cells now, {gs.rate[p][k]:.0f} more per hour", fontsize=9, color="#444444", va="center")
        ax3 = fig.add_axes([0.52, 0.15, 0.44, 0.24]); SP.open_frame(ax3)
        for name, a, b in GP.states(gs):
            ax3.axvspan(a, b, color=GP.STATE_SHADE[name], lw=0, zorder=0)
        if p in gs.matrix and len(gs.matrix[p]):
            mh = gs.matrix_hours[p]; shown = mh <= now + 1e-9
            ax3.plot(mh[shown], gs.matrix[p][shown], color=GP.MATRIX_GREEN, lw=2); ax3.set_ylim(0, max(float(gs.matrix[p].max()), 1.0) * 1.12)
        ax3.axvline(now, color="#9a9a9a", lw=1.2, zorder=1); ax3.set_xlim(hours[0] - pad, hours[-1] + pad)
        ax3.set_xlabel("time after inoculation (h)"); ax3.set_ylabel("matrix stain\n(% of frame)")
        return ax

    def save_gif(self, path: str, progress: Optional[Callable[[int, int], bool]] = None, ms: int = 250):
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
        images += [images[-1]] * 6; images[0].save(path, save_all=True, append_images=images[1:], duration=int(ms), loop=0, optimize=True)
        return path


@SP.with_house_style
def draw_film_strip(fig, film: Optional[GrowthFilm], n: int = 4, now_h: Optional[float] = None):
    """A few frames of the selected position side by side, the hour over each; the one nearest the cursor in bold."""
    fig.clear()
    if film is None or film.position is None or not film.frames:
        SP.empty_axes(fig, "Measure, then pick a position to see its frames"); return []
    picks = sorted({film.frames[int(round(k))] for k in np.linspace(0, len(film.frames) - 1, min(n, len(film.frames)))})
    near = min(picks, key=lambda t: abs(film.hour(t) - now_h)) if now_h is not None else None
    axes = []; w = 0.92 / len(picks); aspect = fig.get_figwidth() / fig.get_figheight(); side = min(w - 0.012, 0.66 / aspect)
    for k, t in enumerate(picks):
        ax = fig.add_axes([0.04 + k * w + (w - side) / 2, 0.08, side, side * aspect]); ax.imshow(film.picture(t)[::3, ::3], interpolation="antialiased")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{film.hour(t):g} h", fontsize=8.5, fontweight="bold" if t == near else "normal", pad=3); axes.append(ax)
    return axes
