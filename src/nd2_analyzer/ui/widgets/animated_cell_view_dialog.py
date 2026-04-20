"""
Animated Cell View Dialog — the "cartoon" playback view.

Click a bacterium in the chamber, watch its whole life play back with:
- the real chamber frames playing like a movie (respecting ROI / crop)
- the chosen cell spotlighted, everyone else dim but still moving
- a dwell heatmap behind everyone (COMSOL placeholder — later swapped for
  real hydrodynamic fields)
- a trajectory panel that draws itself as time advances
- a morphology filmstrip that builds up thumbnails of the cell over its life
- live stats ticker (length / glow / speed / frame)
- division handling: spotlight splits to follow daughters

This is Approach 2 (Cell View) from the paper-1 deck, done visually.
"""
from __future__ import annotations

from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Ellipse
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
# --------------------------------------------------------------------------- #
def _build_cell_lookup(lineage_tracks):
    """Return ``dict[t] -> list[(cell_id, x, y, track_index)]``.

    We need to answer "which cells are where, at time t?" many times during
    playback, and the raw track list iterates one-cell-at-a-time. One pass
    up front flips that so lookups during rendering are O(cells_in_frame).
    """
    per_frame: dict[int, list[tuple[int, float, float, int]]] = {}
    for tr in lineage_tracks or []:
        cid = tr.get("ID")
        xs = tr.get("x", [])
        ys = tr.get("y", [])
        ts = tr.get("t", list(range(len(xs))))
        if cid is None or not (len(xs) == len(ys) == len(ts)):
            continue
        for i, (x, y, t) in enumerate(zip(xs, ys, ts)):
            per_frame.setdefault(int(t), []).append(
                (int(cid), float(x), float(y), i)
            )
    return per_frame


def _compute_dwell_heatmap(lineage_tracks, image_shape, bins=40):
    """COMSOL placeholder: 2D histogram of *where* cells spend their time.

    Once real hydrodynamic fields are imported, this layer gets swapped
    for the flow/nutrient map. Same canvas position, same alpha.
    """
    if not lineage_tracks or image_shape is None:
        return None
    xs, ys = [], []
    for tr in lineage_tracks:
        xs.extend(tr.get("x", []))
        ys.extend(tr.get("y", []))
    if not xs:
        return None
    h, w = image_shape
    hist, _, _ = np.histogram2d(
        ys,
        xs,
        bins=[bins, bins],
        range=[[0, h], [0, w]],
    )
    # Log-scale so a couple of dwelled-in spots don't wash everything else out.
    return np.log1p(hist)


def _descendants(cell_id, tracks_by_id):
    """Return ``{cell_id, all descendants}`` — used for division-spotlight."""
    seen = {int(cell_id)}
    stack = [int(cell_id)]
    while stack:
        cur = stack.pop()
        tr = tracks_by_id.get(cur)
        if not tr:
            continue
        for ch in tr.get("children", []) or []:
            if ch not in seen:
                seen.add(int(ch))
                stack.append(int(ch))
    return seen


# --------------------------------------------------------------------------- #
# Main dialog                                                                  #
# --------------------------------------------------------------------------- #
class AnimatedCellViewDialog(QDialog):
    """
    Animated per-cell playback.

    Parameters
    ----------
    lineage_tracks : list of dict
        Track dicts with ``ID``, ``x``, ``y``, ``t`` and optionally
        ``children`` / ``parent``.
    image_data : ImageData
        Provides ``.data`` (the ND2 stack) and ``.segmentation_cache``.
    metrics_service : MetricsService
        Used for per-cell length / glow at a given timepoint.
    position : int
        Which position (p) these tracks are for.
    parent : QWidget
    """

    # Default playback cadence before speed multiplier (ms per frame).
    BASE_TICK_MS = 250

    # Size of the thumbnail we crop around a cell for the filmstrip (px).
    THUMB_HALF = 24

    def __init__(
        self,
        lineage_tracks,
        image_data,
        metrics_service,
        position: int = 0,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Animated Cell View — cartoon playback")
        self.setMinimumSize(1400, 900)

        # --- data ---------------------------------------------------- #
        self.tracks = list(lineage_tracks or [])
        self.image_data = image_data
        self.metrics_service = metrics_service
        self.position = int(position)

        self.tracks_by_id = {int(tr["ID"]): tr for tr in self.tracks if "ID" in tr}
        self.cells_per_frame = _build_cell_lookup(self.tracks)

        all_ts = sorted(self.cells_per_frame.keys())
        self.t_min = all_ts[0] if all_ts else 0
        self.t_max = all_ts[-1] if all_ts else 0
        self.current_t = self.t_min

        # --- image shape / ROI bounds ------------------------------- #
        self._image_shape = self._infer_image_shape()
        self._crop = self._infer_roi_bbox()  # (y0, x0, y1, x1) or None
        self._dwell = _compute_dwell_heatmap(self.tracks, self._image_shape)

        # Per-frame geometry cache: t -> list[dict(cell_id, cx, cy, major,
        # minor, orientation)]. Populated lazily the first time a given
        # frame is rendered so scrubbing stays responsive after the warm-up.
        self._geom_cache: dict[int, list[dict]] = {}

        # --- spotlight state ---------------------------------------- #
        self.selected_root_id: Optional[int] = None
        # Full spotlight = selected cell + its descendants (so when it
        # divides we automatically follow the daughters).
        self.spotlight_ids: set[int] = set()

        # --- playback ----------------------------------------------- #
        self.is_playing = False
        self.speed_multiplier = 1.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        # --- build UI ----------------------------------------------- #
        self._init_ui()
        self._render_all()

    # ------------------------------------------------------------------ #
    # Setup helpers                                                      #
    # ------------------------------------------------------------------ #
    def _infer_image_shape(self):
        """Return ``(h, w)`` of one raw frame, or fall back to track-bbox."""
        try:
            shape = self.image_data.data.shape
            if len(shape) >= 2:
                return int(shape[-2]), int(shape[-1])
        except Exception:
            pass
        # Fallback: loose bbox from the track coordinates themselves.
        xs, ys = [], []
        for tr in self.tracks:
            xs.extend(tr.get("x", []))
            ys.extend(tr.get("y", []))
        if xs:
            return int(max(ys)) + 10, int(max(xs)) + 10
        return None

    def _infer_roi_bbox(self):
        """Honour the user's ROI if set, else show the whole frame."""
        try:
            from nd2_analyzer.analysis.roi_helper import ROIHelper

            mask = ROIHelper.get_roi_mask()
        except Exception:
            mask = None
        if mask is None:
            return None
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return None
        return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- Playback controls ------------------------------------- #
        ctrl_row = QHBoxLayout()

        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(self._toggle_play)
        ctrl_row.addWidget(self.play_button)

        ctrl_row.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        for label, mult in [
            ("0.25×", 0.25),
            ("0.5×", 0.5),
            ("1×", 1.0),
            ("2×", 2.0),
            ("4×", 4.0),
            ("8×", 8.0),
        ]:
            self.speed_combo.addItem(label, mult)
        self.speed_combo.setCurrentIndex(2)  # 1×
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        ctrl_row.addWidget(self.speed_combo)

        ctrl_row.addSpacing(12)
        ctrl_row.addWidget(QLabel("Time:"))
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(self.t_min)
        self.time_slider.setMaximum(self.t_max)
        self.time_slider.setValue(self.t_min)
        self.time_slider.valueChanged.connect(self._on_slider_changed)
        ctrl_row.addWidget(self.time_slider, stretch=1)

        self.time_label = QLabel(f"t={self.t_min}")
        ctrl_row.addWidget(self.time_label)

        ctrl_row.addSpacing(12)
        self.follow_daughters_check = QCheckBox("Follow daughters after division")
        self.follow_daughters_check.setChecked(True)
        self.follow_daughters_check.stateChanged.connect(self._refresh_spotlight)
        ctrl_row.addWidget(self.follow_daughters_check)

        self.show_dwell_check = QCheckBox("Dwell heatmap")
        self.show_dwell_check.setChecked(True)
        self.show_dwell_check.stateChanged.connect(self._render_main)
        ctrl_row.addWidget(self.show_dwell_check)

        self.clear_sel_button = QPushButton("Clear selection")
        self.clear_sel_button.clicked.connect(self._clear_selection)
        ctrl_row.addWidget(self.clear_sel_button)

        layout.addLayout(ctrl_row)

        # --- Three panels in a splitter ---------------------------- #
        splitter = QSplitter(Qt.Horizontal)

        # Left: main chamber canvas
        self.main_fig = plt.figure(figsize=(8, 8))
        self.main_canvas = FigureCanvas(self.main_fig)
        self.main_canvas.mpl_connect("button_press_event", self._on_main_click)
        splitter.addWidget(self.main_canvas)

        # Right: two stacked side panels
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.traj_fig = plt.figure(figsize=(5, 4))
        self.traj_canvas = FigureCanvas(self.traj_fig)
        right_layout.addWidget(self.traj_canvas, stretch=3)

        self.strip_fig = plt.figure(figsize=(5, 2))
        self.strip_canvas = FigureCanvas(self.strip_fig)
        right_layout.addWidget(self.strip_canvas, stretch=2)

        splitter.addWidget(right_widget)
        splitter.setSizes([900, 500])
        layout.addWidget(splitter, stretch=1)

        # --- Stats ticker ------------------------------------------ #
        self.stats_label = QLabel("Click a cell in the chamber to start.")
        self.stats_label.setStyleSheet(
            "background:#111;color:#ddd;padding:8px;border-radius:4px;"
            "font-family:monospace;"
        )
        layout.addWidget(self.stats_label)

        # --- Close --------------------------------------------------- #
        bottom = QHBoxLayout()
        bottom.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

    # ------------------------------------------------------------------ #
    # Frame access                                                       #
    # ------------------------------------------------------------------ #
    def _get_raw_frame(self, t: int) -> Optional[np.ndarray]:
        """Return the raw image for (t, position, c=0), honouring ROI crop."""
        try:
            data = self.image_data.data
        except Exception:
            return None
        try:
            if data.ndim == 5:
                frame = data[t, self.position, 0]
            elif data.ndim == 4:
                frame = data[t, self.position]
            elif data.ndim == 3:
                frame = data[t]
            else:
                return None
            frame = np.asarray(frame)
        except Exception:
            return None

        if self._crop is not None:
            y0, x0, y1, x1 = self._crop
            y1 = min(y1, frame.shape[0])
            x1 = min(x1, frame.shape[1])
            frame = frame[y0:y1, x0:x1]
        return frame

    def _xy_in_view(self, x: float, y: float) -> tuple[float, float]:
        """Map absolute (x, y) into the displayed crop's coord system."""
        if self._crop is None:
            return x, y
        y0, x0, _, _ = self._crop
        return x - x0, y - y0

    def _get_cell_geometries(self, t: int) -> list[dict]:
        """Return per-cell rod geometry at frame ``t`` for this position.

        Each entry: ``{cell_id, cx, cy, major, minor, orientation}``.
        ``orientation`` is the skimage regionprops angle in radians
        (range -π/2..π/2, measured CCW from the rows axis to the major
        axis). Result is cached per frame so playback doesn't re-query
        metrics on every tick. Missing/empty results cache an empty
        list so we fall back gracefully to track centroids.
        """
        cached = self._geom_cache.get(int(t))
        if cached is not None:
            return cached

        out: list[dict] = []
        try:
            df = self.metrics_service.query_optimized(
                time=int(t), position=int(self.position)
            )
        except Exception:
            df = None

        if df is not None and hasattr(df, "is_empty") and not df.is_empty():
            # Polars DataFrame path. Be defensive about missing columns
            # (older projects may not have written all fields).
            cols = set(df.columns)
            need = {
                "cell_id", "centroid_x", "centroid_y",
                "major_axis_length", "minor_axis_length", "orientation",
            }
            if need.issubset(cols):
                for row in df.iter_rows(named=True):
                    try:
                        out.append({
                            "cell_id": int(row["cell_id"]),
                            "cx": float(row["centroid_x"]),
                            "cy": float(row["centroid_y"]),
                            "major": float(row["major_axis_length"]),
                            "minor": float(row["minor_axis_length"]),
                            "orientation": float(row["orientation"]),
                        })
                    except Exception:
                        continue

        self._geom_cache[int(t)] = out
        return out

    # ------------------------------------------------------------------ #
    # Selection                                                          #
    # ------------------------------------------------------------------ #
    def _on_main_click(self, event):
        """Click on the main canvas → select the closest cell at current t."""
        if event.xdata is None or event.ydata is None:
            return
        cx, cy = float(event.xdata), float(event.ydata)

        cells_here = self.cells_per_frame.get(self.current_t, [])
        if not cells_here:
            return

        # Convert each cell's world coord into view-space before distance.
        best = None
        best_d2 = float("inf")
        for cid, x, y, _idx in cells_here:
            vx, vy = self._xy_in_view(x, y)
            d2 = (vx - cx) ** 2 + (vy - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = cid

        if best is None:
            return

        # 50-pixel click tolerance so misclicks on empty space don't latch.
        if np.sqrt(best_d2) > 50:
            return

        self.selected_root_id = int(best)
        self._refresh_spotlight()
        self._render_all()

    def _clear_selection(self):
        self.selected_root_id = None
        self.spotlight_ids = set()
        self._render_all()

    def _refresh_spotlight(self):
        if self.selected_root_id is None:
            self.spotlight_ids = set()
            return
        if self.follow_daughters_check.isChecked():
            self.spotlight_ids = _descendants(self.selected_root_id, self.tracks_by_id)
        else:
            self.spotlight_ids = {self.selected_root_id}
        self._render_all()

    # ------------------------------------------------------------------ #
    # Playback                                                           #
    # ------------------------------------------------------------------ #
    def _toggle_play(self):
        if self.is_playing:
            self.timer.stop()
            self.play_button.setText("▶ Play")
            self.is_playing = False
        else:
            self._apply_speed()
            self.timer.start()
            self.play_button.setText("⏸ Pause")
            self.is_playing = True

    def _on_speed_changed(self, _idx):
        self.speed_multiplier = float(self.speed_combo.currentData())
        if self.is_playing:
            self._apply_speed()

    def _apply_speed(self):
        interval = max(16, int(self.BASE_TICK_MS / max(self.speed_multiplier, 0.01)))
        self.timer.setInterval(interval)

    def _on_tick(self):
        nxt = self.current_t + 1
        if nxt > self.t_max:
            nxt = self.t_min  # loop
        self.time_slider.setValue(nxt)  # triggers _on_slider_changed

    def _on_slider_changed(self, value):
        self.current_t = int(value)
        self.time_label.setText(f"t={self.current_t}")
        self._render_all()

    # ------------------------------------------------------------------ #
    # Rendering                                                          #
    # ------------------------------------------------------------------ #
    def _render_all(self):
        self._render_main()
        self._render_trajectory()
        self._render_filmstrip()
        self._render_stats()

    def _render_main(self):
        self.main_fig.clear()
        ax = self.main_fig.add_subplot(111)
        ax.set_facecolor("#000")

        frame = self._get_raw_frame(self.current_t)
        h, w = (self._image_shape or (500, 500))
        if self._crop is not None:
            y0, x0, y1, x1 = self._crop
            h, w = y1 - y0, x1 - x0

        if frame is not None:
            # Normalize for display only.
            f = frame.astype(np.float32)
            lo, hi = np.percentile(f, [1, 99])
            if hi > lo:
                f = np.clip((f - lo) / (hi - lo), 0, 1)
            else:
                f = f * 0
            ax.imshow(f, cmap="gray", aspect="equal", origin="upper")
        else:
            # No raw frame available — just set axes for the overlay.
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)

        # Dwell heatmap (COMSOL placeholder) -------------------------- #
        if self.show_dwell_check.isChecked() and self._dwell is not None:
            extent = (0, w, h, 0)
            if self._crop is None:
                dwell_to_show = self._dwell
            else:
                # The dwell map was built on full-frame coords; extent
                # trick aligns visually but strictly it's approximate.
                dwell_to_show = self._dwell
            ax.imshow(
                dwell_to_show,
                cmap="magma",
                alpha=0.25,
                aspect="auto",
                extent=extent,
                origin="upper",
            )

        # Cells at current t: draw as rod cartoons (ellipses) using real
        # per-cell geometry from the metrics service. When geometry is
        # missing (no segmentation for this t/p) we fall back to a small
        # dot at the track centroid so the frame isn't empty.
        cells_here = self.cells_per_frame.get(self.current_t, [])
        geoms = self._get_cell_geometries(self.current_t)
        geom_by_id = {g["cell_id"]: g for g in geoms}

        hot_fallback_xy: list[tuple[int, float, float]] = []
        dim_fallback_x: list[float] = []
        dim_fallback_y: list[float] = []

        # Spotlight first pass: draw trails underneath the rods so the
        # rod sits on top of its own gold trail.
        if self.spotlight_ids:
            for cid in self.spotlight_ids:
                tr = self.tracks_by_id.get(cid)
                if not tr:
                    continue
                trail_x, trail_y = [], []
                for x, y, t in zip(
                    tr.get("x", []), tr.get("y", []), tr.get("t", [])
                ):
                    if int(t) <= self.current_t:
                        vx, vy = self._xy_in_view(x, y)
                        trail_x.append(vx)
                        trail_y.append(vy)
                if len(trail_x) >= 2:
                    ax.plot(
                        trail_x, trail_y, "-",
                        color="#ffd54a", linewidth=2, alpha=0.9, zorder=3,
                    )

        # Second pass: draw every cell as a rod. Non-spotlight cells are
        # translucent grey; spotlight cells are gold on top (higher
        # zorder) with a thick outline so they stand out in a crowd.
        for cid, x, y, _idx in cells_here:
            g = geom_by_id.get(cid)
            vx, vy = self._xy_in_view(x, y)
            is_hot = cid in self.spotlight_ids

            if g is None:
                # No segmentation-derived geometry; remember for a
                # fallback scatter pass below.
                if is_hot:
                    hot_fallback_xy.append((cid, vx, vy))
                else:
                    dim_fallback_x.append(vx)
                    dim_fallback_y.append(vy)
                continue

            # Use the centroid from metrics (segmentation-true) when we
            # have it; fall back to the tracker's (x, y) if the metrics
            # row is there but its centroid is weird.
            gcx, gcy = self._xy_in_view(g["cx"], g["cy"])
            major = max(float(g["major"]), 2.0)
            minor = max(float(g["minor"]), 1.0)

            # Matplotlib Ellipse angle is degrees CCW from the x-axis,
            # with ``width`` drawn along x before rotation. skimage
            # orientation is measured from the rows (y) axis to the
            # major axis, CCW. With imshow(origin='upper') the y-axis
            # is flipped, which flips the visual rotation sense. The
            # formula below gives rods that track their true orientation
            # on top of the raw image.
            angle_deg = -np.degrees(float(g["orientation"]))

            if is_hot:
                ellipse = Ellipse(
                    (gcx, gcy),
                    width=major, height=minor,
                    angle=angle_deg,
                    facecolor="#ffd54a", edgecolor="#ff8c00",
                    linewidth=2.0, alpha=0.9, zorder=5,
                )
                ax.add_patch(ellipse)
                ax.annotate(
                    str(cid), (gcx, gcy),
                    xytext=(8, -4), textcoords="offset points",
                    color="#ffd54a", fontsize=9, fontweight="bold",
                    zorder=6,
                )
            else:
                ellipse = Ellipse(
                    (gcx, gcy),
                    width=major, height=minor,
                    angle=angle_deg,
                    facecolor="#4a90e2", edgecolor="#2c5f8d",
                    linewidth=0.4, alpha=0.35, zorder=2,
                )
                ax.add_patch(ellipse)

        # Fallback scatter for any cells we had no geometry for. Kept
        # small so it doesn't dominate the view when most cells DO have
        # geometry.
        if dim_fallback_x:
            ax.scatter(
                dim_fallback_x, dim_fallback_y,
                s=6, c="#4a90e2", alpha=0.3, linewidths=0, zorder=2,
            )
        if hot_fallback_xy:
            xs_h = [p[1] for p in hot_fallback_xy]
            ys_h = [p[2] for p in hot_fallback_xy]
            ax.scatter(
                xs_h, ys_h, s=120, facecolors="none",
                edgecolors="#ffd54a", linewidths=2.2, zorder=5,
            )
            for cid, vx, vy in hot_fallback_xy:
                ax.annotate(
                    str(cid), (vx, vy),
                    xytext=(8, -4), textcoords="offset points",
                    color="#ffd54a", fontsize=9, fontweight="bold",
                    zorder=6,
                )

        # Division popup --------------------------------------------- #
        div_popup = self._division_event_here()
        if div_popup is not None:
            ax.text(
                0.02, 0.98, div_popup,
                transform=ax.transAxes,
                color="#ffd54a",
                fontsize=11,
                fontweight="bold",
                va="top",
                bbox=dict(facecolor="#000", alpha=0.75, edgecolor="#ffd54a"),
            )

        ax.set_title(f"Chamber @ t = {self.current_t}", color="#ddd")
        ax.set_xticks([])
        ax.set_yticks([])
        self.main_fig.tight_layout()
        self.main_canvas.draw_idle()

    def _division_event_here(self) -> Optional[str]:
        """If a spotlighted cell divides at this exact frame, return a label."""
        if not self.spotlight_ids:
            return None
        for cid in self.spotlight_ids:
            tr = self.tracks_by_id.get(cid)
            if not tr:
                continue
            ts = tr.get("t", [])
            children = tr.get("children") or []
            if not ts or not children:
                continue
            if int(ts[-1]) == self.current_t:
                return f"DIVIDED: cell {cid} → {list(children)}"
        return None

    def _render_trajectory(self):
        self.traj_fig.clear()
        ax = self.traj_fig.add_subplot(111)
        ax.set_facecolor("#111")

        if self.selected_root_id is None:
            ax.text(
                0.5, 0.5, "No cell selected\n(click on the chamber)",
                color="#888", ha="center", va="center",
                transform=ax.transAxes,
            )
            self.traj_canvas.draw_idle()
            return

        for cid in self.spotlight_ids:
            tr = self.tracks_by_id.get(cid)
            if not tr:
                continue
            xs, ys, ts = [], [], []
            for x, y, t in zip(
                tr.get("x", []), tr.get("y", []), tr.get("t", [])
            ):
                if int(t) <= self.current_t:
                    xs.append(x)
                    ys.append(y)
                    ts.append(int(t))
            if len(xs) >= 1:
                # Colour by time (viridis) so you can see direction of travel.
                ax.plot(xs, ys, "-", color="#5de1ff", linewidth=1.2, alpha=0.6)
                if len(xs) >= 1:
                    ax.scatter(
                        [xs[0]], [ys[0]], s=40, c="#2ecc71", zorder=3,
                        label=f"birth (cell {cid})" if cid == self.selected_root_id else None,
                    )
                    ax.scatter(
                        [xs[-1]], [ys[-1]], s=60, c="#ffd54a", zorder=4,
                        edgecolors="#000", linewidths=1,
                        label=f"now (cell {cid})" if cid == self.selected_root_id else None,
                    )

        if self._image_shape is not None:
            h, w = self._image_shape
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)

        ax.set_title(
            f"Trajectory — cell {self.selected_root_id}"
            + (f" + {len(self.spotlight_ids) - 1} descendants"
               if len(self.spotlight_ids) > 1 else ""),
            color="#ddd", fontsize=10,
        )
        ax.tick_params(colors="#888", labelsize=8)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_color("#444")
        self.traj_fig.tight_layout()
        self.traj_canvas.draw_idle()

    def _render_filmstrip(self):
        """Strip of cropped thumbnails of the selected cell over time.

        Thumbnails are cropped directly from the raw frame around the
        cell's position at each timepoint. Current frame gets a gold
        border so you can see *where in its life* the cell is right now.
        """
        self.strip_fig.clear()
        self.strip_fig.patch.set_facecolor("#111")

        if self.selected_root_id is None:
            ax = self.strip_fig.add_subplot(111)
            ax.set_facecolor("#111")
            ax.text(
                0.5, 0.5, "Select a cell to see its filmstrip.",
                color="#888", ha="center", va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            self.strip_canvas.draw_idle()
            return

        tr = self.tracks_by_id.get(self.selected_root_id)
        if not tr:
            self.strip_canvas.draw_idle()
            return

        ts = tr.get("t", [])
        xs = tr.get("x", [])
        ys = tr.get("y", [])
        if not ts:
            self.strip_canvas.draw_idle()
            return

        # Show at most ~20 thumbnails — subsample evenly through the life.
        n = len(ts)
        max_thumbs = 20
        if n > max_thumbs:
            idxs = np.linspace(0, n - 1, max_thumbs).astype(int)
        else:
            idxs = np.arange(n)

        for slot, i in enumerate(idxs):
            t = int(ts[i])
            x = float(xs[i])
            y = float(ys[i])
            ax = self.strip_fig.add_subplot(1, len(idxs), slot + 1)
            ax.set_facecolor("#111")

            thumb = self._crop_thumbnail(t, x, y)
            if thumb is not None:
                ax.imshow(thumb, cmap="gray")
            else:
                ax.text(
                    0.5, 0.5, "—", color="#666",
                    ha="center", va="center", transform=ax.transAxes,
                )

            ax.set_xticks([])
            ax.set_yticks([])
            # Highlight the thumbnail at (or just before) the current cursor.
            is_current = t <= self.current_t and (
                slot == len(idxs) - 1 or int(ts[idxs[slot + 1]]) > self.current_t
            )
            for side in ("top", "right", "bottom", "left"):
                ax.spines[side].set_color("#ffd54a" if is_current else "#333")
                ax.spines[side].set_linewidth(2 if is_current else 1)
            ax.set_title(f"t={t}", color="#aaa", fontsize=7)

        self.strip_fig.suptitle(
            f"Life of cell {self.selected_root_id}",
            color="#ddd", fontsize=10,
        )
        self.strip_fig.tight_layout()
        self.strip_canvas.draw_idle()

    def _crop_thumbnail(self, t: int, x: float, y: float) -> Optional[np.ndarray]:
        """Crop a small patch around (x, y) at time t from the raw frame."""
        try:
            data = self.image_data.data
            if data.ndim == 5:
                frame = np.asarray(data[t, self.position, 0])
            elif data.ndim == 4:
                frame = np.asarray(data[t, self.position])
            elif data.ndim == 3:
                frame = np.asarray(data[t])
            else:
                return None
        except Exception:
            return None

        h, w = frame.shape
        xi, yi = int(round(x)), int(round(y))
        y0 = max(0, yi - self.THUMB_HALF)
        y1 = min(h, yi + self.THUMB_HALF)
        x0 = max(0, xi - self.THUMB_HALF)
        x1 = min(w, xi + self.THUMB_HALF)
        if y1 <= y0 or x1 <= x0:
            return None
        patch = frame[y0:y1, x0:x1].astype(np.float32)
        lo, hi = np.percentile(patch, [1, 99]) if patch.size else (0, 1)
        if hi > lo:
            patch = np.clip((patch - lo) / (hi - lo), 0, 1)
        else:
            patch = patch * 0
        return patch

    # ------------------------------------------------------------------ #
    # Stats ticker                                                       #
    # ------------------------------------------------------------------ #
    def _render_stats(self):
        if self.selected_root_id is None:
            self.stats_label.setText(
                f"t = {self.current_t}  |  click a bacterium to spotlight it."
            )
            return

        parts = [f"t = {self.current_t:>4}"]
        parts.append(f"spotlight = {sorted(self.spotlight_ids)}")

        # Per-cell live readout for the selected root.
        tr = self.tracks_by_id.get(self.selected_root_id)
        if tr:
            pos_now = self._position_at(tr, self.current_t)
            if pos_now is not None:
                parts.append(f"pos = ({pos_now[0]:.0f}, {pos_now[1]:.0f})")

            speed = self._speed_at(tr, self.current_t)
            if speed is not None:
                parts.append(f"speed ≈ {speed:.1f} px/frame")

        # Length + glow from metrics_service (best-effort; missing is fine).
        length, glow = self._metrics_at(self.selected_root_id, self.current_t)
        if length is not None and not np.isnan(length):
            parts.append(f"length = {length:.1f}")
        if glow is not None and not np.isnan(glow):
            parts.append(f"glow = {glow:.1f}")

        self.stats_label.setText("   |   ".join(parts))

    @staticmethod
    def _position_at(track, t):
        for x, y, tt in zip(track.get("x", []), track.get("y", []), track.get("t", [])):
            if int(tt) == int(t):
                return float(x), float(y)
        return None

    @staticmethod
    def _speed_at(track, t):
        ts = [int(tt) for tt in track.get("t", [])]
        xs = list(track.get("x", []))
        ys = list(track.get("y", []))
        if t not in ts:
            return None
        i = ts.index(int(t))
        if i == 0:
            return 0.0
        dt = ts[i] - ts[i - 1]
        if dt <= 0:
            return None
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        return float(np.hypot(dx, dy) / dt)

    def _metrics_at(self, cell_id: int, t: int):
        """Best-effort length + glow lookup from the metrics service."""
        try:
            df = self.metrics_service.query_optimized(time=int(t), cell_id=int(cell_id))
            if df is None or df.is_empty():
                return None, None
            row = df.row(0, named=True)
            length = row.get("major_axis_length")
            # "glow" — whatever intensity-like column we have handy.
            glow = (
                row.get("mean_intensity")
                or row.get("mean_fluorescence")
                or row.get("intensity_mean")
            )
            return length, glow
        except Exception:
            return None, None
