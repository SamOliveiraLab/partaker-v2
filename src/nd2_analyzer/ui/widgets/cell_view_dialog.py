"""
Cell View Dialog — follow individual bacteria through time.

Select a founding cell from the list on the left. Three synced panels show:
  1. The segmented chamber view with the selected cell glowing bright
  2. The trajectory drawn stroke-by-stroke as time advances
  3. The lineage tree growing as divisions happen

All panels share one time cursor. Play, pause, scrub, adjust speed.
Position-aware: only shows cells from the selected position (p).
"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nd2_analyzer.analysis.roi_helper import ROIHelper


# -------------------------------------------------------------------- #
# Helpers                                                               #
# -------------------------------------------------------------------- #

def _build_cell_lookup(tracks):
    """dict[t] -> list[(cell_id, x, y)]"""
    out: dict[int, list[tuple[int, float, float]]] = {}
    for tr in tracks or []:
        cid = tr.get("ID")
        xs, ys, ts = tr.get("x", []), tr.get("y", []), tr.get("t", [])
        if cid is None or not (len(xs) == len(ys) == len(ts)):
            continue
        for x, y, t in zip(xs, ys, ts):
            out.setdefault(int(t), []).append((int(cid), float(x), float(y)))
    return out


def _find_founders(tracks):
    """Return list of track dicts that have no parent (generation 0)."""
    return [tr for tr in (tracks or []) if tr.get("parent") is None]


def _descendants(root_id, tracks_by_id):
    """BFS from root_id through children -> set of all descendant IDs."""
    result = {root_id}
    queue = [root_id]
    while queue:
        cid = queue.pop(0)
        tr = tracks_by_id.get(cid)
        if not tr:
            continue
        for ch in tr.get("children") or []:
            ch = int(ch)
            if ch not in result:
                result.add(ch)
                queue.append(ch)
    return result


def _track_to_seg_label(labeled_mask, x, y):
    """Look up which segmentation label sits at (x, y) in the mask."""
    yi, xi = int(round(y)), int(round(x))
    if 0 <= yi < labeled_mask.shape[0] and 0 <= xi < labeled_mask.shape[1]:
        return int(labeled_mask[yi, xi])
    return 0


# -------------------------------------------------------------------- #
# Dialog                                                                #
# -------------------------------------------------------------------- #

class CellViewDialog(QDialog):

    BASE_TICK_MS = 400

    def __init__(
        self,
        lineage_tracks,
        metrics_service,
        image_data=None,
        position: int = 0,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Cell View — follow a bacterium through time")
        self.setMinimumSize(1600, 950)

        # --- data ---
        self.tracks = list(lineage_tracks or [])
        self.metrics_service = metrics_service
        self.image_data = image_data
        self.position = int(position)

        self.tracks_by_id = {int(tr["ID"]): tr for tr in self.tracks if "ID" in tr}
        self.cells_per_frame = _build_cell_lookup(self.tracks)
        self.founders = _find_founders(self.tracks)

        all_ts = sorted(self.cells_per_frame.keys())
        self.t_min = all_ts[0] if all_ts else 0
        self.t_max = all_ts[-1] if all_ts else 0
        self.current_t = self.t_min

        # --- image / ROI ---
        self._image_shape = self._infer_image_shape()
        self._crop = self._infer_roi_bbox()

        # --- geometry cache (for rod rendering) ---
        self._geom_cache: dict[int, list[dict]] = {}

        # --- seg label cache: t -> labeled_mask (full frame) ---
        self._seg_cache: dict[int, np.ndarray] = {}

        # --- selection ---
        self.selected_root_id: Optional[int] = None
        self.spotlight_ids: set[int] = set()

        # --- playback ---
        self.is_playing = False
        self.speed_multiplier = 1.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        # --- build UI and render ---
        self._init_ui()
        self._render_all()

    # ================================================================ #
    # Setup                                                            #
    # ================================================================ #

    def _infer_image_shape(self):
        try:
            s = self.image_data.data.shape
            if len(s) >= 2:
                return int(s[-2]), int(s[-1])
        except Exception:
            pass
        xs, ys = [], []
        for tr in self.tracks:
            xs.extend(tr.get("x", []))
            ys.extend(tr.get("y", []))
        if xs:
            return int(max(ys)) + 10, int(max(xs)) + 10
        return None

    def _infer_roi_bbox(self):
        try:
            mask = ROIHelper.get_roi_mask()
        except Exception:
            mask = None
        if mask is None:
            return None
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return None
        return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1

    # ================================================================ #
    # UI                                                               #
    # ================================================================ #

    def _init_ui(self):
        root = QVBoxLayout(self)

        # --- playback controls ---
        ctrl = QHBoxLayout()

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_play)
        ctrl.addWidget(self.play_btn)

        ctrl.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        for label, mult in [("0.25×", 0.25), ("0.5×", 0.5), ("1×", 1.0),
                            ("2×", 2.0), ("4×", 4.0), ("8×", 8.0)]:
            self.speed_combo.addItem(label, mult)
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        ctrl.addWidget(self.speed_combo)

        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Time:"))
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(self.t_min)
        self.time_slider.setMaximum(self.t_max)
        self.time_slider.setValue(self.t_min)
        self.time_slider.valueChanged.connect(self._on_slider_changed)
        ctrl.addWidget(self.time_slider, stretch=1)

        self.time_label = QLabel(f"t={self.t_min}")
        ctrl.addWidget(self.time_label)

        ctrl.addSpacing(12)
        self.follow_daughters_cb = QCheckBox("Follow daughters")
        self.follow_daughters_cb.setChecked(True)
        self.follow_daughters_cb.stateChanged.connect(self._refresh_spotlight)
        ctrl.addWidget(self.follow_daughters_cb)

        root.addLayout(ctrl)

        # --- main area: cell list | chamber | traj + tree ---
        outer_split = QSplitter(Qt.Horizontal)

        # LEFT: founding cell list
        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.addWidget(QLabel("Founding cells (no parent):"))

        self.cell_table = QTableWidget()
        self.cell_table.setColumnCount(4)
        self.cell_table.setHorizontalHeaderLabels(["ID", "Lifespan", "Fate", "Kids"])
        self.cell_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.cell_table.setSelectionMode(QTableWidget.SingleSelection)
        self.cell_table.cellClicked.connect(self._on_table_click)
        hdr = self.cell_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Stretch)
        left_l.addWidget(self.cell_table)

        # filter
        filt_row = QHBoxLayout()
        filt_row.addWidget(QLabel("Show:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All founders", "Long (20+)", "With divisions"])
        self.filter_combo.currentTextChanged.connect(self._populate_table)
        filt_row.addWidget(self.filter_combo)
        left_l.addLayout(filt_row)

        outer_split.addWidget(left_w)

        # CENTER: chamber canvas
        self.main_fig = plt.figure(figsize=(8, 8))
        self.main_canvas = FigureCanvas(self.main_fig)
        self.main_canvas.mpl_connect("button_press_event", self._on_canvas_click)
        outer_split.addWidget(self.main_canvas)

        # RIGHT: trajectory (top) + lineage tree (bottom)
        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(0, 0, 0, 0)

        right_split = QSplitter(Qt.Vertical)

        self.traj_fig = plt.figure(figsize=(5, 4))
        self.traj_canvas = FigureCanvas(self.traj_fig)
        right_split.addWidget(self.traj_canvas)

        self.tree_fig = plt.figure(figsize=(5, 4))
        self.tree_canvas = FigureCanvas(self.tree_fig)
        right_split.addWidget(self.tree_canvas)

        right_split.setSizes([400, 400])
        right_l.addWidget(right_split)
        outer_split.addWidget(right_w)

        outer_split.setSizes([250, 800, 450])
        root.addWidget(outer_split, stretch=1)

        # --- stats bar ---
        self.stats_label = QLabel("Select a cell from the list or click on the chamber.")
        self.stats_label.setStyleSheet(
            "background:#111; color:#ddd; padding:8px; border-radius:4px;"
            "font-family:monospace;"
        )
        root.addWidget(self.stats_label)

        # --- bottom buttons ---
        bottom = QHBoxLayout()
        self.export_btn = QPushButton("Export GIF")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_gif)
        bottom.addWidget(self.export_btn)
        bottom.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)
        root.addLayout(bottom)

        # --- populate the cell table ---
        self._populate_table()

    def _populate_table(self, _filter_text=None):
        filt = self.filter_combo.currentText() if hasattr(self, "filter_combo") else "All founders"
        self.cell_table.setRowCount(0)

        cells = list(self.founders)
        if filt == "Long (20+)":
            cells = [tr for tr in cells if len(tr.get("t", [])) >= 20]
        elif filt == "With divisions":
            cells = [tr for tr in cells if tr.get("children")]

        cells.sort(key=lambda tr: tr.get("ID", 0))
        self.cell_table.setRowCount(len(cells))

        for row, tr in enumerate(cells):
            cid = tr["ID"]
            lifespan = len(tr.get("t", []))
            children = tr.get("children") or []
            has_kids = len(children) > 0
            fate = "divided" if has_kids else (
                "alive" if lifespan >= (self.t_max - self.t_min + 1) else "lost"
            )

            id_item = QTableWidgetItem(str(cid))
            id_item.setData(Qt.UserRole, cid)
            self.cell_table.setItem(row, 0, id_item)
            self.cell_table.setItem(row, 1, QTableWidgetItem(str(lifespan)))
            self.cell_table.setItem(row, 2, QTableWidgetItem(fate))
            self.cell_table.setItem(row, 3, QTableWidgetItem(str(len(children))))

    # ================================================================ #
    # Selection                                                        #
    # ================================================================ #

    def _on_table_click(self, row, _col):
        item = self.cell_table.item(row, 0)
        if not item:
            return
        cid = item.data(Qt.UserRole)
        self._select_cell(int(cid))

    def _on_canvas_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        cx, cy = float(event.xdata), float(event.ydata)
        cells = self.cells_per_frame.get(self.current_t, [])
        if not cells:
            return
        best, best_d2 = None, float("inf")
        for cid, x, y in cells:
            vx, vy = self._xy_in_view(x, y)
            d2 = (vx - cx) ** 2 + (vy - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = cid
        if best is None or np.sqrt(best_d2) > 50:
            return
        # Find the root ancestor of this cell
        root = self._find_root(best)
        self._select_cell(root)

    def _find_root(self, cid):
        visited = set()
        while cid not in visited:
            visited.add(cid)
            tr = self.tracks_by_id.get(cid)
            if not tr or tr.get("parent") is None:
                return cid
            cid = int(tr["parent"])
        return cid

    def _select_cell(self, root_id):
        self.selected_root_id = root_id
        self._refresh_spotlight()
        self.export_btn.setEnabled(True)
        self._render_all()

    def _refresh_spotlight(self):
        if self.selected_root_id is None:
            self.spotlight_ids = set()
            return
        if self.follow_daughters_cb.isChecked():
            self.spotlight_ids = _descendants(self.selected_root_id, self.tracks_by_id)
        else:
            self.spotlight_ids = {self.selected_root_id}
        self._render_all()

    def _clear_selection(self):
        self.selected_root_id = None
        self.spotlight_ids = set()
        self._render_all()

    # ================================================================ #
    # Playback                                                         #
    # ================================================================ #

    def _toggle_play(self):
        if self.is_playing:
            self.timer.stop()
            self.play_btn.setText("▶ Play")
            self.is_playing = False
        else:
            self._apply_speed()
            self.timer.start()
            self.play_btn.setText("⏸ Pause")
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
            nxt = self.t_min
        self.time_slider.setValue(nxt)

    def _on_slider_changed(self, value):
        self.current_t = int(value)
        self.time_label.setText(f"t={self.current_t}")
        self._render_all()

    # ================================================================ #
    # Frame + geometry access                                          #
    # ================================================================ #

    def _xy_in_view(self, x, y):
        if self._crop is None:
            return x, y
        y0, x0, _, _ = self._crop
        return x - x0, y - y0

    def _get_raw_frame(self, t):
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
        if self._crop is not None:
            y0, x0, y1, x1 = self._crop
            frame = frame[y0:min(y1, frame.shape[0]), x0:min(x1, frame.shape[1])]
        return frame

    def _get_labeled_mask(self, t):
        """Get the segmentation labeled mask at (t, position)."""
        cached = self._seg_cache.get(int(t))
        if cached is not None:
            return cached
        try:
            from nd2_analyzer.data.image_data import ImageData
            img = ImageData.get_instance()
            model = img.segmentation_service.models.available_models[0]
            mask = img.segmentation_cache.with_model(model)[(t, self.position, 0)]
            if mask is not None:
                mask = np.asarray(mask)
                from skimage.measure import label as sk_label
                if mask.max() <= 255 and len(np.unique(mask)) <= 100:
                    mask = sk_label(mask > 0)
            self._seg_cache[int(t)] = mask
            return mask
        except Exception:
            return None

    def _get_cell_geometries(self, t):
        cached = self._geom_cache.get(int(t))
        if cached is not None:
            return cached
        out = []
        try:
            df = self.metrics_service.query_optimized(
                time=int(t), position=int(self.position)
            )
        except Exception:
            df = None
        if df is not None and hasattr(df, "is_empty") and not df.is_empty():
            cols = set(df.columns)
            need = {"cell_id", "centroid_x", "centroid_y",
                    "major_axis_length", "minor_axis_length", "orientation"}
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

    # ================================================================ #
    # Rendering — main chamber                                         #
    # ================================================================ #

    def _render_all(self):
        self._render_chamber()
        self._render_trajectory()
        self._render_tree()
        self._render_stats()

    def _render_chamber(self):
        self.main_fig.clear()
        ax = self.main_fig.add_subplot(111)
        ax.set_facecolor("#000")

        h, w = (self._image_shape or (500, 500))
        if self._crop is not None:
            y0, x0, y1, x1 = self._crop
            h, w = y1 - y0, x1 - x0

        # Show the raw frame as background
        frame = self._get_raw_frame(self.current_t)
        if frame is not None:
            f = frame.astype(np.float32)
            lo, hi = np.percentile(f, [1, 99])
            if hi > lo:
                f = np.clip((f - lo) / (hi - lo), 0, 1)
            else:
                f = f * 0
            ax.imshow(f, cmap="gray", aspect="equal", origin="upper")
        else:
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)

        # Get segmentation mask to identify which seg label matches each track
        labeled_mask = self._get_labeled_mask(self.current_t)

        # If we have a spotlight cell, build a colored overlay from the seg mask
        if self.spotlight_ids and labeled_mask is not None:
            # Find which seg labels correspond to spotlight track IDs
            spotlight_seg_labels = set()
            cells_here = self.cells_per_frame.get(self.current_t, [])
            for cid, x, y in cells_here:
                if cid in self.spotlight_ids:
                    seg_label = _track_to_seg_label(labeled_mask, x, y)
                    if seg_label > 0:
                        spotlight_seg_labels.add(seg_label)

            # Build overlay: spotlight cells = gold, others = blue dim
            if self._crop is not None:
                y0, x0, y1, x1 = self._crop
                view_mask = labeled_mask[y0:min(y1, labeled_mask.shape[0]),
                                         x0:min(x1, labeled_mask.shape[1])]
            else:
                view_mask = labeled_mask

            overlay = np.zeros((*view_mask.shape, 4), dtype=np.float32)

            # Dim cells (blue, low alpha)
            all_cell_pixels = view_mask > 0
            overlay[all_cell_pixels] = [0.29, 0.56, 0.89, 0.25]

            # Spotlight cells (gold, high alpha)
            for sl in spotlight_seg_labels:
                spot_pixels = view_mask == sl
                overlay[spot_pixels] = [1.0, 0.84, 0.0, 0.55]

            ax.imshow(overlay, aspect="equal", origin="upper")

        # Draw trajectory trail for spotlight cells
        if self.spotlight_ids:
            for cid in self.spotlight_ids:
                tr = self.tracks_by_id.get(cid)
                if not tr:
                    continue
                trail_x, trail_y = [], []
                for x, y, t in zip(tr.get("x", []), tr.get("y", []),
                                   tr.get("t", [])):
                    if int(t) <= self.current_t:
                        vx, vy = self._xy_in_view(x, y)
                        trail_x.append(vx)
                        trail_y.append(vy)
                if len(trail_x) >= 2:
                    ax.plot(trail_x, trail_y, "-", color="#ffd54a",
                            linewidth=2, alpha=0.8, zorder=10)
                if trail_x:
                    ax.plot(trail_x[-1], trail_y[-1], "o", color="#ffd54a",
                            markersize=8, zorder=11, markeredgecolor="#ff8c00",
                            markeredgewidth=1.5)

        # Division popup
        div_text = self._division_event_here()
        if div_text:
            ax.text(0.02, 0.98, div_text, transform=ax.transAxes,
                    color="#ffd54a", fontsize=11, fontweight="bold", va="top",
                    bbox=dict(facecolor="#000", alpha=0.75, edgecolor="#ffd54a"))

        ax.set_title(f"Chamber @ t={self.current_t}  |  Position {self.position}",
                     color="#ddd", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        self.main_fig.tight_layout()
        self.main_canvas.draw_idle()

    def _division_event_here(self):
        if not self.spotlight_ids:
            return None
        for cid in self.spotlight_ids:
            tr = self.tracks_by_id.get(cid)
            if not tr:
                continue
            ts = tr.get("t", [])
            children = tr.get("children") or []
            if ts and children and int(ts[-1]) == self.current_t:
                return f"DIVIDED: cell {cid} -> {list(children)}"
        return None

    # ================================================================ #
    # Rendering — trajectory panel                                     #
    # ================================================================ #

    def _render_trajectory(self):
        self.traj_fig.clear()
        ax = self.traj_fig.add_subplot(111)
        ax.set_facecolor("#111")

        if self.selected_root_id is None:
            ax.text(0.5, 0.5, "Select a cell", color="#888",
                    ha="center", va="center", transform=ax.transAxes)
            self.traj_canvas.draw_idle()
            return

        # Draw ALL cells in this frame as dim dots for context
        cells_here = self.cells_per_frame.get(self.current_t, [])
        if cells_here:
            all_x = [x for _, x, _ in cells_here]
            all_y = [y for _, _, y in cells_here]
            ax.scatter(all_x, all_y, s=4, c="#333", alpha=0.4, linewidths=0, zorder=1)

        # Draw spotlight cell trajectories
        for cid in self.spotlight_ids:
            tr = self.tracks_by_id.get(cid)
            if not tr:
                continue
            xs, ys = [], []
            for x, y, t in zip(tr.get("x", []), tr.get("y", []),
                               tr.get("t", [])):
                if int(t) <= self.current_t:
                    xs.append(x)
                    ys.append(y)

            if not xs:
                continue

            color = "#ffd54a" if cid == self.selected_root_id else "#5de1ff"
            ax.plot(xs, ys, "-", color=color, linewidth=1.5, alpha=0.8, zorder=3)
            ax.scatter([xs[0]], [ys[0]], s=40, c="#2ecc71", zorder=4)
            ax.scatter([xs[-1]], [ys[-1]], s=60, c=color, zorder=5,
                       edgecolors="#000", linewidths=1)
            ax.annotate(str(cid), (xs[-1], ys[-1]), xytext=(6, -4),
                        textcoords="offset points", color=color,
                        fontsize=8, fontweight="bold", zorder=6)

        if self._image_shape:
            ih, iw = self._image_shape
            ax.set_xlim(0, iw)
            ax.set_ylim(ih, 0)

        n_desc = len(self.spotlight_ids) - 1
        title = f"Trajectory — cell {self.selected_root_id}"
        if n_desc > 0:
            title += f" + {n_desc} descendants"
        ax.set_title(title, color="#ddd", fontsize=10)
        ax.tick_params(colors="#888", labelsize=7)
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_color("#444")
        self.traj_fig.tight_layout()
        self.traj_canvas.draw_idle()

    # ================================================================ #
    # Rendering — lineage tree                                         #
    # ================================================================ #

    def _render_tree(self):
        self.tree_fig.clear()
        ax = self.tree_fig.add_subplot(111)
        ax.set_facecolor("#111")

        if self.selected_root_id is None:
            ax.text(0.5, 0.5, "Select a cell", color="#888",
                    ha="center", va="center", transform=ax.transAxes)
            self.tree_canvas.draw_idle()
            return

        # Collect all family members that are ALIVE up to current_t
        family_alive = set()
        edges_alive = []
        for cid in self.spotlight_ids:
            tr = self.tracks_by_id.get(cid)
            if not tr:
                continue
            ts = tr.get("t", [])
            if not ts:
                continue
            birth = int(min(ts))
            if birth <= self.current_t:
                family_alive.add(cid)
                parent = tr.get("parent")
                if parent is not None and int(parent) in family_alive:
                    edges_alive.append((int(parent), cid))

        if not family_alive:
            ax.text(0.5, 0.5, "Cell not yet born", color="#888",
                    ha="center", va="center", transform=ax.transAxes)
            self.tree_canvas.draw_idle()
            return

        # Layout: simple top-down tree using BFS layers
        positions = {}
        layers: dict[int, list[int]] = {}

        def get_generation(cid):
            gen = 0
            visited = set()
            cur = cid
            while cur not in visited:
                visited.add(cur)
                tr = self.tracks_by_id.get(cur)
                if not tr or tr.get("parent") is None:
                    break
                cur = int(tr["parent"])
                gen += 1
            return gen

        for cid in family_alive:
            gen = get_generation(cid)
            layers.setdefault(gen, []).append(cid)

        max_gen = max(layers.keys()) if layers else 0
        for gen, members in layers.items():
            members.sort()
            n = len(members)
            for i, cid in enumerate(members):
                x = (i + 0.5) / max(n, 1)
                y = 1.0 - gen / max(max_gen + 1, 1)
                positions[cid] = (x, y)

        # Draw edges
        for parent, child in edges_alive:
            if parent in positions and child in positions:
                px, py = positions[parent]
                cx, cy = positions[child]
                ax.plot([px, cx], [py, cy], "-", color="#555", linewidth=1.5, zorder=1)

        # Draw nodes
        for cid, (x, y) in positions.items():
            tr = self.tracks_by_id.get(cid)
            ts = tr.get("t", []) if tr else []
            is_alive_now = self.current_t in [int(tt) for tt in ts]
            has_kids = bool(tr.get("children")) if tr else False

            if is_alive_now:
                color = "#ffd54a" if cid == self.selected_root_id else "#5de1ff"
                size = 200
            elif has_kids and int(max(ts)) <= self.current_t:
                color = "#e74c3c"
                size = 120
            else:
                color = "#888"
                size = 80

            ax.scatter([x], [y], s=size, c=color, zorder=3, edgecolors="#000",
                       linewidths=1)
            ax.annotate(str(cid), (x, y), ha="center", va="center",
                        color="#000" if is_alive_now else "#ddd",
                        fontsize=7, fontweight="bold", zorder=4)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"Lineage tree — cell {self.selected_root_id}", color="#ddd",
                     fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_color("#333")
        self.tree_fig.tight_layout()
        self.tree_canvas.draw_idle()

    # ================================================================ #
    # Stats ticker                                                     #
    # ================================================================ #

    def _render_stats(self):
        if self.selected_root_id is None:
            self.stats_label.setText(
                f"t={self.current_t}  |  Click a cell or pick from the list."
            )
            return

        parts = [f"t={self.current_t:>4}"]
        parts.append(f"selected: {self.selected_root_id}")
        parts.append(f"family: {sorted(self.spotlight_ids)}")

        tr = self.tracks_by_id.get(self.selected_root_id)
        if tr:
            ts = [int(tt) for tt in tr.get("t", [])]
            if self.current_t in ts:
                i = ts.index(self.current_t)
                x = float(tr["x"][i])
                y = float(tr["y"][i])
                parts.append(f"pos=({x:.0f},{y:.0f})")
                if i > 0:
                    dx = float(tr["x"][i]) - float(tr["x"][i - 1])
                    dy = float(tr["y"][i]) - float(tr["y"][i - 1])
                    speed = np.hypot(dx, dy)
                    parts.append(f"speed={speed:.1f} px/frame")

            # morphology from metrics
            try:
                df = self.metrics_service.query_optimized(
                    time=self.current_t, position=self.position
                )
                if df is not None and not df.is_empty():
                    for row in df.iter_rows(named=True):
                        if int(row.get("cell_id", -1)) == self.selected_root_id:
                            if "major_axis_length" in row:
                                parts.append(f"length={row['major_axis_length']:.1f}")
                            if "area" in row:
                                parts.append(f"area={row['area']:.0f}")
                            break
            except Exception:
                pass

        self.stats_label.setText("   |   ".join(parts))

    # ================================================================ #
    # GIF export                                                       #
    # ================================================================ #

    def _export_gif(self):
        if self.selected_root_id is None:
            QMessageBox.warning(self, "No selection", "Select a cell first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Cell Animation",
            f"cell_{self.selected_root_id}.gif",
            "GIF (*.gif)"
        )
        if not path:
            return

        try:
            import imageio
            frames = []
            for t in range(self.t_min, self.t_max + 1):
                self.current_t = t
                self._render_chamber()
                self.main_canvas.draw()
                buf = self.main_canvas.buffer_rgba()
                img = np.asarray(buf)[:, :, :3].copy()
                frames.append(img)

            imageio.mimsave(path, frames, duration=int(self.BASE_TICK_MS), loop=0)
            QMessageBox.information(self, "Done", f"Exported {len(frames)} frames to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
        finally:
            self.time_slider.setValue(self.t_min)
