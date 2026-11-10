"""The Seeding window's drawing, shared with its export and the tests: pure matplotlib on a given Figure, no Qt.

Two views, laid out like the lineage film (film_render.FamilyFilm):
  Cell:  one followed track and its daughters. Left, the camera on the family: every cell coloured by what it does, the
         family outlined in white with green ID tags, the yellow trail, a white dot on each centroid, the whole field small
         in the picture's top-left corner with the camera box. Right, its lineage tree: lineage-red bars with the IDs, a
         daughter that leaves within DAUGHTER_STAY_S in grey with an open circle, and under it how far each cell is from the
         landing spot.
  Field: the whole field, every cell coloured by what it does, green ID tags on the cells that stay for good and on the
         followed family. Right, one line per landing that stayed 30 min or more, drawn up to the cursor, the followed
         family in bold with its ID.
Colours are seeding_figures.FATE_COLOURS by rules.fate_class_expr. The film runs over the measured frames only."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import polars as pl

from nd2_analyzer.analysis.biofilm_stages import seeding_figures as SF
from nd2_analyzer.analysis.biofilm_stages import seeding_plots as SP
from nd2_analyzer.analysis.biofilm_stages.seeding_rules import FATE_KEYS, fate_class_labels

TAG, TRAIL, UNKNOWN = "#00ff00", "#ffd54a", "#5a5a5a"
DIM = 0.55                    # Cell view: cells outside the family are dimmed, as in the lineage film
CELL_HALF_UM = 4.0            # the Cell view's camera: this many um each side of the followed cell
FIELD_STEP_H = 0.1            # one Field view step, hours of recording
CELL_STEP_H = 0.04            # the longest Cell view step
CELL_MIN_STEPS = 12           # a short family still gets at least this many steps
CELL_LEAD_H = 0.05            # the Cell view starts this long before the landing
TRAIL_STEPS = 6               # steps of trail behind the followed cell (about 15 min)
CLICK_RADIUS_PX = 15.0        # a click within this distance of a centroid picks that cell
LABEL_CACHE_BYTES = 160e6     # label frames kept in memory
FIGURE_IN = (10.0, 4.6)       # the window's figure, as the lineage window's


def _rgb(colour) -> np.ndarray:
    import matplotlib.colors as mc
    return np.asarray(mc.to_rgb(colour), float)


def _tag(ax, x: float, y: float, text: str, size: float):
    """A green ID tag with a thin black edge, so it reads over cells and over the trail."""
    import matplotlib.patheffects as pe
    ax.text(x, y, text, color=TAG, fontsize=size, fontweight="bold", zorder=6, clip_on=True,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="black")])


def per_point(fig, frac_w: float, frac_h: float, nx: float, ny: float) -> float:
    """Image pixels per typographic point in an image axes frac_w x frac_h of the figure, at any figure size."""
    W, H = fig.get_size_inches()
    return max(nx / (frac_w * W), ny / (frac_h * H)) / 72.0


def place_tags(points: list, em: float, pad: float = 0.0, boxes=(), bounds=None) -> list:
    """(x, y, text) tag anchors (left end of the baseline, image pixels), each moved by whole lines, down first, to the
    nearest spot where it clears the tags placed before it by pad and the boxes (x0, top, x1, bottom: the scale bar,
    the corner map) by pad / 2, between bounds (top, bottom). em is the tag's font size in image pixels. A tag with no
    clear spot within six lines keeps its own."""
    h = pad / 2.0
    taken = [tuple(b) for b in boxes]
    out = []
    for x, y, text in points:
        w = 0.62 * em * len(text)

        def box(yy):
            return (x - h, yy - 0.93 * em - h, x + w + h, yy + 0.22 * em + h)

        def fits(b):
            if bounds is not None and (b[1] < bounds[0] or b[3] > bounds[1]):
                return False
            return all(b[2] <= t[0] or b[0] >= t[2] or b[3] <= t[1] or b[1] >= t[3] for t in taken)

        line = 1.15 * em + pad
        best = next((y + k * line for k in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6) if fits(box(y + k * line))), y)
        taken.append(box(best)); out.append((x, best, text))
    return out


class SeedingFilm:
    """The state behind the Seeding window: the tracks by id, their fate class and daughters, which label is which track
    in each frame, and the frames each view steps through."""

    def __init__(self, results, tracks: Sequence[dict], masks_at: Callable[[int], np.ndarray], field_shape: tuple,
                 dt_s: float, start_hour: float = 0.0, pixel_um: Optional[float] = None, n_frames: Optional[int] = None):
        self.results = results
        self.masks_at = masks_at
        self.field_shape = tuple(int(v) for v in field_shape)
        self.dt_s = float(dt_s)
        self.start_hour = float(start_hour)
        self.pixel_um = pixel_um
        self.pt = SF.classed(results)
        self.fate = dict(zip(self.pt["track_id"].to_list(), self.pt["fate_class"].to_list())) if self.pt.height else {}
        self.info = {int(r["track_id"]): r for r in self.pt.iter_rows(named=True)} if self.pt.height else {}
        self.kids = {}
        for tid, r in self.info.items():
            if r.get("parent") is not None:
                self.kids.setdefault(int(r["parent"]), []).append(tid)
        self.tracks = {}
        self.lab2track = {}
        self.points = {}              # frame -> [(track id, x, y)], for clicks and for tracks without cell ids
        self.needs_mapping = False
        for tr in tracks:
            tid = int(tr["ID"]); t = np.asarray(tr["t"], dtype=int); order = np.argsort(t, kind="stable")
            rec = {"t": t[order], "x": np.asarray(tr["x"], float)[order], "y": np.asarray(tr["y"], float)[order]}
            self.tracks[tid] = rec
            cids = tr.get("cell_ids")
            if cids is not None and len(cids) == len(t):
                for ti, cid in zip(rec["t"], np.asarray(cids)[order]):
                    self.lab2track[(int(ti), int(cid))] = tid
            else:
                self.needs_mapping = True
            for ti, x, y in zip(rec["t"], rec["x"], rec["y"]):
                self.points.setdefault(int(ti), []).append((tid, float(x), float(y)))
        counts = getattr(results, "counts", None)
        if counts is not None and counts.height:
            first, last = int(counts["frame"].min()), int(counts["frame"].max())
        else:
            first = min((int(r["t"][0]) for r in self.tracks.values() if r["t"].size), default=0)
            last = max((int(r["t"][-1]) for r in self.tracks.values() if r["t"].size), default=0)
        if n_frames:
            last = min(last, int(n_frames) - 1)
        self.first_frame, self.last_frame = first, last
        self.span = (self.hour(first), self.hour(last))
        self.rows = SF.staying_rows(results)
        self.lines_x0 = SF.lines_start_hour(self.rows, self.span)
        step = max(1, int(round(FIELD_STEP_H * 3600 / self.dt_s)))
        self.field_frames = list(range(first, last + 1, step))
        if not self.field_frames or self.field_frames[-1] != last:
            self.field_frames.append(last)
        self.counts_by_frame = dict(zip(counts["frame"].to_list(), counts["cells"].to_list())) if counts is not None and counts.height else {}
        st = getattr(results, "stage_by_frame", None)
        self.stage_of = dict(zip(st["frame"].to_list(), st["stage"].to_list())) if st is not None and st.height else {}
        self.irreversible = {t for t, f in self.fate.items() if f in ("divided", "end")}
        self.root: Optional[int] = None
        self.family: list = []
        self.cell_frames: list = []
        self.last_center = None
        self._labels = {}
        self._label_bytes = 0

    # ------------------------------------------------------------------ time and positions
    def hour(self, frame: int) -> float:
        return self.start_hour + float(frame) * self.dt_s / 3600.0

    def frame_at(self, hour: float) -> int:
        return int(np.clip(round((float(hour) - self.start_hour) * 3600.0 / self.dt_s), self.first_frame, self.last_frame))

    def pos_at(self, tid: int, frame: int):
        tr = self.tracks.get(int(tid))
        if tr is None:
            return None
        i = int(np.searchsorted(tr["t"], frame))
        if i < tr["t"].size and tr["t"][i] == frame:
            return float(tr["x"][i]), float(tr["y"][i])
        return None

    def end_hour_of(self, tid: int) -> float:
        r = self.info.get(int(tid))
        if r is None:
            return self.span[1]
        return self.span[1] if r["fate"] == "present at end" else min(self.span[1], float(r["t_end_h"]))

    def labels(self, frame: int) -> np.ndarray:
        """The label frame, kept in a cache of LABEL_CACHE_BYTES; for tracks without cell ids, the labels under their
        points are mapped to them as the frame is read."""
        frame = int(frame)
        if frame not in self._labels:
            lab = np.asarray(self.masks_at(frame))
            while self._labels and self._label_bytes + lab.nbytes > LABEL_CACHE_BYTES:
                old = self._labels.pop(next(iter(self._labels))); self._label_bytes -= old.nbytes
            self._labels[frame] = lab; self._label_bytes += lab.nbytes
            if self.needs_mapping:
                self._map_points(frame, lab)
        return self._labels[frame]

    def _map_points(self, frame: int, lab: np.ndarray):
        for tid, x, y in self.points.get(frame, []):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < lab.shape[0] and 0 <= xi < lab.shape[1] and lab[yi, xi] > 0:
                self.lab2track.setdefault((frame, int(lab[yi, xi])), tid)

    # ------------------------------------------------------------------ the followed family
    def set_root(self, root: int):
        """Follow this track: its daughters join it, and the Cell view steps from just before its landing to the end of
        the last of them, every frame of a short family included."""
        root = int(root)
        self.root = root
        self.last_center = None
        self.family = [root] + sorted(self.kids.get(root, []))
        r = self.info.get(root)
        land = float(r["t_start_h"]) if r is not None else self.span[0]
        a = max(land - CELL_LEAD_H, self.span[0])
        b = min(max(self.end_hour_of(t) for t in self.family) + 0.1, self.span[1])
        fa, fb = self.frame_at(a), self.frame_at(b)
        step = max(1, min(int(round(CELL_STEP_H * 3600 / self.dt_s)), (fb - fa) // CELL_MIN_STEPS if fb > fa else 1))
        frames = set(range(fa, fb + 1, step))
        for t in self.family:                                   # the landing, the division and the last frame of each
            rec = self.tracks.get(t)
            if rec is not None and rec["t"].size:
                frames.update(int(f) for f in (rec["t"][0], rec["t"][-1]) if fa <= f <= fb)
        self.cell_frames = sorted(frames) or [fa]

    def default_root(self) -> Optional[int]:
        """The first landing that stays for good (divides or stays to the end), else the longest stay."""
        L = self.pt.filter(pl.col("origin") == "landed") if self.pt.height else self.pt
        if not L.height:
            return None
        good = L.filter(pl.col("fate_class").is_in(["divided", "end"])).sort("t_start_h")
        if good.height:
            return int(good["track_id"][0])
        return int(L.sort("residence_s", descending=True)["track_id"][0])

    def track_at(self, frame: int, x: float, y: float) -> Optional[int]:
        """The track under (x, y) in field pixels at this frame: the label there, else the nearest track point within
        CLICK_RADIUS_PX."""
        lab = self.labels(frame)
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < lab.shape[0] and 0 <= xi < lab.shape[1] and lab[yi, xi] > 0:
            tid = self.lab2track.get((int(frame), int(lab[yi, xi])))
            if tid is not None:
                return tid
        best, best_d = None, CLICK_RADIUS_PX
        for tid, px, py in self.points.get(int(frame), []):
            d = float(np.hypot(px - x, py - y))
            if d < best_d:
                best, best_d = tid, d
        return best

    # ------------------------------------------------------------------ the picture
    def coloured(self, frame: int, outline=(), paint=None, dim: float = 1.0) -> tuple:
        """The label frame as RGB: each cell in its track's fate colour (grey when no track), a white ring round outline.
        paint {track: colour} gives those tracks their own colour; every other cell is dimmed by dim."""
        from scipy import ndimage as ndi
        lab = self.labels(frame)
        n = int(lab.max()) if lab.size else 0
        lut = np.tile(_rgb(UNKNOWN), (n + 1, 1)); lut[0] = 0.0
        paint = paint or {}
        tid_of = {}
        for cid in range(1, n + 1):
            tid = self.lab2track.get((int(frame), cid))
            tid_of[cid] = tid
            if tid in paint:
                lut[cid] = _rgb(paint[tid]); continue
            if tid is not None:
                lut[cid] = _rgb(SF.FATE_COLOURS.get(self.fate.get(tid, "while"), SF.FATE_COLOURS["while"]))
            lut[cid] = lut[cid] * dim
        rgb = lut[lab]
        if outline:
            want = [cid for cid, tid in tid_of.items() if tid in set(outline)]
            if want:
                sel = np.isin(lab, want)
                ring = ndi.binary_dilation(sel, iterations=3) & ~sel
                rgb[ring] = 1.0
        return rgb, tid_of

    def scale_bar(self, ax, x0: float, y0: float, um: float, colour="white", per_pt: float = 0.0):
        """The bar and its words; returns the box they fill (x0, top, x1, bottom in image pixels) for the tags to keep
        clear of. per_pt is image pixels per point (per_point)."""
        if not self.pixel_um:
            return None
        L = um / float(self.pixel_um)
        words = f"{um:g} µm"
        ax.plot([x0, x0 + L], [y0, y0], color=colour, lw=3.5, solid_capstyle="butt")
        ax.text(x0, y0 - 0.06 * L, words, color=colour, fontsize=8.5, va="bottom", ha="left")
        return (x0, y0 - 0.06 * L - 1.15 * 8.5 * per_pt, x0 + max(L, 0.62 * 8.5 * len(words) * per_pt), y0 + 2.0 * per_pt)

    # ------------------------------------------------------------------ the Field view
    @SP.with_house_style
    def draw_field(self, fig, frame: int):
        frame = int(frame); h = self.hour(frame)
        fig.clear()
        ax = fig.add_axes([0.03, 0.15, 0.41, 0.73])
        rgb, tid_of = self.coloured(frame, outline=self.family)
        ax.imshow(rgb, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"the field   frame {frame}   {h:.2f} h", loc="left", fontsize=10, fontweight="bold")
        Y, X = self.field_shape
        per_pt = per_point(fig, 0.41, 0.73, X, Y)
        bar = self.scale_bar(ax, 0.04 * X, 0.95 * Y, 10, per_pt=per_pt)
        tagged = self.irreversible | set(self.family)
        pts = []
        for _cid, tid in sorted(tid_of.items(), key=lambda kv: (kv[1] is None, kv[1] or 0)):
            if tid in tagged:
                p = self.pos_at(tid, frame)
                if p is not None:
                    right = p[0] > 0.9 * X                      # near the right edge the tag goes to the left of the cell
                    pts.append((p[0] - 12 - 0.06 * X if right else p[0] + 12, p[1] - 12, str(tid)))
        for x, y, text in place_tags(pts, em=7 * per_pt, pad=7 * per_pt, boxes=[bar] if bar else (), bounds=(0, Y)):
            _tag(ax, x, y, text, 7)
        from matplotlib.patches import Patch
        names = fate_class_labels(SF.settings_of(self.results, None))
        fig.legend([Patch(color=SF.FATE_COLOURS[k]) for k in FATE_KEYS], [names[k] for k in FATE_KEYS], ncol=4, loc="lower left",
                   bbox_to_anchor=(0.025, 0.0), frameon=False, fontsize=8, handlelength=1.0, columnspacing=1.0, handletextpad=0.4)
        rax = fig.add_axes([0.52, 0.14, 0.45, 0.72])
        SF.draw_staying_lines(rax, self.rows, self.span[1], upto=h, bold=set(self.family),
                              labels={self.root} if self.root is not None else set())
        rax.axvline(h, ymin=0.09, color="#9a9a9a", lw=1.1)       # stops above the count line
        rax.set_xlim(self.lines_x0, self.span[1]); rax.set_yticks([]); rax.spines["left"].set_visible(False)
        from matplotlib.ticker import MaxNLocator
        rax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        rax.set_xlabel("time after inoculation (h)"); rax.set_title("time on the surface", loc="left", fontsize=10)
        self.stage_title(rax, frame)
        n = self.counts_by_frame.get(frame)
        if n is not None:
            rax.text(0.0, 0.02, f"{int(n)} cell{'' if int(n) == 1 else 's'} on the surface now", transform=rax.transAxes,
                     ha="left", va="bottom", fontsize=9.5)
        SP.open_frame(rax)
        return ax, rax

    # ------------------------------------------------------------------ the Cell view
    @SP.with_house_style
    def draw_cell(self, fig, frame: int):
        if self.root is None:
            r = self.default_root()
            if r is None:
                fig.clear(); return None, None
            self.set_root(r)
        frame = int(frame); h = self.hour(frame)
        root = self.root
        info = self.info.get(root, {})
        land = float(info.get("t_start_h", self.span[0])); tdiv = float(info.get("t_end_h", self.span[1]))
        divides = int(info.get("n_children") or 0) >= 2
        ds = sorted(self.kids.get(root, []), key=lambda k: -float(self.info.get(k, {}).get("residence_s", 0)))
        first_d = ds[0] if ds else None
        second_d = ds[1] if len(ds) > 1 else None
        main = root if (not divides or h < tdiv or first_d is None) else first_d
        p = self.pos_at(main, frame) or self.pos_at(root, frame)
        if p is None:
            p = self.last_center if self.last_center is not None else self.landing_spot()
        self.last_center = p
        half = int(round(CELL_HALF_UM / float(self.pixel_um))) if self.pixel_um else 60
        Y, X = self.field_shape
        cx = int(np.clip(p[0], half, X - half)); cy = int(np.clip(p[1], half, Y - half))
        fig.clear()
        ax = fig.add_axes([0.03, 0.08, 0.41, 0.80])
        alive = [t for t in self.family if self.pos_at(t, frame) is not None]
        root_col = SF.LINEAGE_RED if divides else SF.FATE_COLOURS.get(self.fate.get(root, "while"), SF.FATE_COLOURS["while"])
        paint = {t: self.tree_colour(t, root_col) for t in self.family}      # each cell as its bar in the tree
        rgb, _ = self.coloured(frame, outline=alive, paint=paint, dim=DIM)
        ax.imshow(rgb[cy - half:cy + half, cx - half:cx + half], interpolation="nearest",
                  extent=(cx - half, cx + half, cy + half, cy - half))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"track {root} and its daughters   frame {frame}   {h:.2f} h" if self.kids.get(root)
                     else f"track {root}   frame {frame}   {h:.2f} h", loc="left", fontsize=10, fontweight="bold")
        per_pt = per_point(fig, 0.41, 0.80, 2 * half, 2 * half)
        bar = self.scale_bar(ax, cx - half + 0.1 * half, cy + half - 0.15 * half, 2, per_pt=per_pt)
        corner = (cx - 0.96 * half, cy - 0.96 * half, cx - 0.52 * half, cy - 0.52 * half)   # the corner map below
        pts = []
        for t in alive:
            q = self.pos_at(t, frame)
            if not (cx - half <= q[0] <= cx + half and cy - half <= q[1] <= cy + half):
                continue                                         # outside the camera: no dot, no tag
            ax.plot([q[0]], [q[1]], "o", ms=4, color="white", zorder=5)
            right = q[0] > cx + 0.45 * half                      # near the right edge the tag goes to the left
            pts.append((q[0] - 0.75 * half if right else q[0] + 0.25 * half, q[1] - 0.15 * half, str(t)))
        for x, y, text in place_tags(pts, em=9 * per_pt, pad=7 * per_pt, boxes=[b for b in (bar, corner) if b],
                                     bounds=(cy - half, cy + half)):
            _tag(ax, x, y, text, 9)
        trail = self.trail(main, frame)
        if len(trail) > 1:
            tx, ty = zip(*trail); ax.plot(tx, ty, color=TRAIL, lw=1.6, zorder=4)
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy + half, cy - half)
        ins = ax.inset_axes([0.02, 0.76, 0.22, 0.22])           # the whole field, top left so it is read first
        small = (self.labels(frame) > 0)[::8, ::8]
        ins.imshow(small, cmap="gray", interpolation="nearest"); ins.set_xticks([]); ins.set_yticks([])
        for s in ins.spines.values():
            s.set_color("#9a9a9a")
        from matplotlib.patches import Rectangle
        ins.add_patch(Rectangle(((cx - half) / 8, (cy - half) / 8), 2 * half / 8, 2 * half / 8, fill=False, ec=TRAIL, lw=1.2))

        tr = fig.add_axes([0.52, 0.42, 0.45, 0.46]); mv = fig.add_axes([0.52, 0.12, 0.45, 0.18], sharex=tr)
        tr.set_title("lineage tree", loc="left", fontsize=10)
        self.stage_title(tr, frame)
        last_h = self.hour(self.cell_frames[-1]) if self.cell_frames else h
        span = (land - 0.12 * max(0.5, last_h - land), max(last_h, land + 0.2) + 0.05)
        yM, yS, yL = 0.0, 0.55, -0.55
        lab = dict(bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"), zorder=5)   # the time line passes behind
        if h > land:
            end_root = min(h, tdiv if divides else self.end_hour_of(root))
            tr.plot([land, end_root], [yM, yM], color=root_col, lw=3.4, solid_capstyle="butt")
            tr.text((land + end_root) / 2, yM + 0.07, str(root), ha="center", va="bottom", fontsize=9, **lab)
        tr.plot([land], [yM], marker="|", ms=10, mew=1.8, color=SP.INK)
        tr.text(land - 0.01 * (span[1] - span[0]), yM - 0.12, f"lands {land:.1f} h", ha="right", va="top", fontsize=8.5, color="#4a4a4a", **lab)
        on = 1 if self.pos_at(root, frame) is not None else 0
        if divides and h >= tdiv and first_d is not None:
            on = 0
            tr.plot([tdiv, tdiv], [yL if second_d is not None else yM, yS], color=SF.NEUTRAL, lw=1.1)
            gap = 0.012 * (span[1] - span[0])                  # daughter labels sit at the start of their branch, clear of the time line
            for d, yy in ((first_d, yS), (second_d, yL)):
                if d is None:
                    continue
                drow = self.info.get(d, {})
                d_end_h = self.end_hour_of(d)
                d_end = min(h, max(d_end_h, tdiv + 0.02))
                if SF.daughter_stays(drow) if drow else True:
                    tr.plot([tdiv, d_end], [yy, yy], color=SF.LINEAGE_RED, lw=3.4, solid_capstyle="butt")
                    tr.text(tdiv - gap, yy, str(d), ha="right", va="center", fontsize=9, **lab)
                else:
                    tr.plot([tdiv, d_end], [yy, yy], color=SP.GREY, lw=1.8, solid_capstyle="butt")
                    if h >= d_end_h:
                        tr.plot([d_end], [yy], "o", ms=6, mfc="white", mec=SP.GREY, mew=1.4)
                        tr.text(tdiv - gap, yy, f"{d} left at {d_end_h:.2f} h", ha="right", va="center", fontsize=8.5, color="#4a4a4a", **lab)
                    else:
                        tr.text(tdiv - gap, yy, str(d), ha="right", va="center", fontsize=9, **lab)
                on += 1 if self.pos_at(d, frame) is not None else 0
        tr.text(0.0, 0.98, f"{on} cell{'s' if on != 1 else ''} of this family on the surface", transform=tr.transAxes,
                ha="left", va="top", fontsize=9.5)
        tr.axvline(h, ymax=0.86, color="#9a9a9a", lw=1.1, zorder=1.5)  # stops below the count line, behind the bars
        mv.axvline(h, color="#9a9a9a", lw=1.1)
        tr.set_ylim(-1.0, 1.0); tr.set_yticks([]); tr.spines["left"].set_visible(False); tr.spines["bottom"].set_visible(False)
        tr.tick_params(axis="x", length=0, labelbottom=False)
        spot = self.landing_spot()
        top = 0.0
        for t in (root, first_d, second_d):
            if t is None or t not in self.tracks or not self.pixel_um:
                continue
            col = self.tree_colour(t, root_col)
            rec = self.tracks[t]; tt = self.start_hour + rec["t"] * self.dt_s / 3600.0; keep = tt <= h
            if keep.sum() > 1:
                d = np.hypot(rec["x"][keep] - spot[0], rec["y"][keep] - spot[1]) * float(self.pixel_um)
                mv.plot(tt[keep], d, color=col, lw=1.5); top = max(top, float(d.max()))
        mv.set_xlim(*span); mv.set_ylim(0, max(2.0, np.ceil(top)))
        from matplotlib.ticker import MaxNLocator
        mv.yaxis.set_major_locator(MaxNLocator(nbins=2, integer=True)); mv.xaxis.set_major_locator(MaxNLocator(nbins=6))
        mv.set_ylabel("from landing\nspot (µm)", fontsize=8.5); mv.set_xlabel("time after inoculation (h)")
        tr.set_xlim(*span)
        SP.open_frame(mv)
        return ax, tr

    def stage_title(self, ax, frame: int):
        """The stage of the biofilm at this frame (reversible attachment, irreversible attachment, microcolonies) at the
        top right of the time panel, so the stage changes as the film plays."""
        word = SP.STAGE_LABEL.get(self.stage_of.get(int(frame), ""), "")
        if word:
            ax.set_title(word, loc="right", fontsize=10, fontweight="bold")

    def tree_colour(self, tid: int, root_col: str) -> str:
        """A family member's colour in the tree: the root's own, red for a daughter that stays, grey for one that left."""
        if tid == self.root:
            return root_col
        d = self.info.get(tid)
        return SF.LINEAGE_RED if (SF.daughter_stays(d) if d else True) else SP.GREY

    def landing_spot(self):
        rec = self.tracks.get(self.root)
        if rec is None or not rec["t"].size:
            Y, X = self.field_shape
            return (X / 2.0, Y / 2.0)
        return float(rec["x"][0]), float(rec["y"][0])

    def trail(self, tid: int, frame: int) -> list:
        rec = self.tracks.get(int(tid))
        if rec is None:
            return []
        step = max(1, int(round(CELL_STEP_H * 3600 / self.dt_s)))
        keep = (rec["t"] <= frame) & (rec["t"] >= frame - TRAIL_STEPS * step)
        return list(zip(rec["x"][keep][::max(1, step // 4)], rec["y"][keep][::max(1, step // 4)]))

    # ------------------------------------------------------------------ export
    def frames_of(self, view: str) -> list:
        if view == "Cell":
            return list(self.cell_frames)
        x0 = self.frame_at(self.lines_x0)
        return [f for f in self.field_frames if f >= x0]

    def save_gif(self, path: str, view: str, progress: Optional[Callable[[int, int], bool]] = None, ms: int = 120):
        """The view as a GIF, one image per step; progress(k, n) returning False stops it."""
        from PIL import Image
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        frames = self.frames_of(view)
        images = []
        for k, f in enumerate(frames):
            if progress is not None and progress(k, len(frames)) is False:
                return None
            fig = Figure(figsize=FIGURE_IN, dpi=100); FigureCanvasAgg(fig)
            (self.draw_cell if view == "Cell" else self.draw_field)(fig, f)
            fig.canvas.draw()
            im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            images.append(im.quantize(colors=160, method=Image.Quantize.MEDIANCUT))
        if not images:
            return None
        images += [images[-1]] * 6
        images[0].save(path, save_all=True, append_images=images[1:], duration=int(ms), loop=0, optimize=True)
        return path
