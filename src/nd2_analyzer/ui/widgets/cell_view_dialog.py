"""
Cell View Dialog — track a single cell across all timepoints.

Shows the same metrics table as "Classify Cells" (for the first frame).
Click a cell → it gets tracked across time with dim/bright highlighting,
exactly like the main view area. Press Play to auto-advance through frames.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
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

from skimage.measure import label as sk_label


class CellViewDialog(QDialog):

    BASE_TICK_MS = 200

    def __init__(
        self,
        lineage_tracks,
        metrics_service,
        image_data=None,
        position: int = 0,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Cell View")
        self.setMinimumSize(1200, 800)

        self.lineage_tracks = list(lineage_tracks or [])
        self.metrics_service = metrics_service
        self.image_data = image_data
        self.position = int(position)

        # Determine time range from tracks
        all_ts = set()
        for tr in self.lineage_tracks:
            all_ts.update(int(t) for t in tr.get("t", []))
        sorted_ts = sorted(all_ts)
        self.t_min = sorted_ts[0] if sorted_ts else 0
        self.t_max = sorted_ts[-1] if sorted_ts else 0

        # Start at the current slider position, not t_min
        from nd2_analyzer.data.appstate import ApplicationState
        appstate = ApplicationState.get_instance()
        if appstate and appstate.view_index:
            self.initial_t = int(appstate.view_index[0])
        else:
            self.initial_t = self.t_min
        self.current_t = self.initial_t

        # Tracking state
        self.tracked_cell_lineage = {}
        self.selected_track_id = None
        self.cell_mapping = {}
        self.tracks_by_id = {int(tr["ID"]): tr for tr in self.lineage_tracks if "ID" in tr}

        # Segmentation model
        self._seg_model = None
        try:
            self._seg_model = image_data.segmentation_service.models.available_models[0]
        except Exception:
            pass

        # Seg cache
        self._seg_cache: dict[int, np.ndarray] = {}

        # Morphology colors
        self.morphology_colors = {
            "Artifact": (128, 128, 128),
            "Divided": (255, 0, 0),
            "Healthy": (0, 255, 0),
            "Elongated": (0, 255, 255),
            "Deformed": (255, 0, 255),
        }

        # Playback
        self.is_playing = False
        self.speed_multiplier = 1.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        self._init_ui()
        self._populate_metrics_table()

    # ------------------------------------------------------------------ #
    # UI                                                                  #
    # ------------------------------------------------------------------ #

    def _init_ui(self):
        root = QVBoxLayout(self)

        # --- playback controls ---
        ctrl = QHBoxLayout()

        self.play_btn = QPushButton("Play")
        self.play_btn.setFixedWidth(80)
        self.play_btn.clicked.connect(self._toggle_play)
        ctrl.addWidget(self.play_btn)

        ctrl.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        for label, mult in [("0.25x", 0.25), ("0.5x", 0.5), ("1x", 1.0),
                            ("2x", 2.0), ("4x", 4.0), ("8x", 8.0)]:
            self.speed_combo.addItem(label, mult)
        self.speed_combo.setCurrentIndex(3)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        ctrl.addWidget(self.speed_combo)

        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Time:"))
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(self.t_min)
        self.time_slider.setMaximum(self.t_max)
        self.time_slider.setValue(self.initial_t)
        self.time_slider.valueChanged.connect(self._on_slider_changed)
        ctrl.addWidget(self.time_slider, stretch=1)

        self.time_label = QLabel(f"t={self.initial_t}")
        ctrl.addWidget(self.time_label)

        root.addLayout(ctrl)

        # --- main area: table | image ---
        splitter = QSplitter(Qt.Horizontal)

        # LEFT: metrics table
        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.addWidget(QLabel(f"Cells (P{self.position}, T{self.initial_t}):"))

        self.metrics_table = QTableWidget()
        self.metrics_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.metrics_table.setSelectionMode(QTableWidget.SingleSelection)
        self.metrics_table.itemClicked.connect(self._on_table_click)
        left_l.addWidget(self.metrics_table)

        # Export buttons
        export_layout = QHBoxLayout()

        self.export_btn = QPushButton("Export GIF")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_gif)
        export_layout.addWidget(self.export_btn)

        self.export_csv_btn = QPushButton("Export Cell History CSV")
        self.export_csv_btn.clicked.connect(self._export_cell_history)
        export_layout.addWidget(self.export_csv_btn)

        left_l.addLayout(export_layout)

        splitter.addWidget(left_w)

        # RIGHT: image (top) + trajectory plot (bottom)
        right_split = QSplitter(Qt.Vertical)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        right_split.addWidget(self.image_label)

        # Bottom panel: trajectory (left) + lineage tree (right)
        bottom_split = QSplitter(Qt.Horizontal)

        self.traj_fig, self.traj_ax = plt.subplots(figsize=(4, 3))
        self.traj_fig.patch.set_facecolor("#111")
        self.traj_canvas = FigureCanvas(self.traj_fig)
        bottom_split.addWidget(self.traj_canvas)

        self.tree_fig, self.tree_ax = plt.subplots(figsize=(4, 3))
        self.tree_fig.patch.set_facecolor("#111")
        self.tree_canvas = FigureCanvas(self.tree_fig)
        bottom_split.addWidget(self.tree_canvas)

        bottom_split.setSizes([500, 500])
        right_split.addWidget(bottom_split)

        right_split.setSizes([450, 350])
        splitter.addWidget(right_split)

        splitter.setSizes([300, 900])
        root.addWidget(splitter, stretch=1)

        # Pre-compute track colors for consistency
        self._track_colors = {}
        np.random.seed(42)
        for track in self.lineage_tracks:
            tid = track["ID"]
            rng = np.random.RandomState(tid * 7 + 3)
            self._track_colors[tid] = tuple(c / 255.0 for c in rng.randint(80, 255, 3))

        # --- status bar ---
        self.status_label = QLabel("Click a cell in the table to track it.")
        self.status_label.setStyleSheet(
            "background:#111; color:#ddd; padding:6px; font-family:monospace;"
        )
        root.addWidget(self.status_label)

    # ------------------------------------------------------------------ #
    # Metrics table (same as morphology "Classify Cells")                 #
    # ------------------------------------------------------------------ #

    def _populate_metrics_table(self):
        """Populate table with cell metrics for the current slider position."""
        try:
            polars_df = self.metrics_service.query_optimized(
                time=self.initial_t, position=self.position
            )
        except Exception as e:
            print(f"CellViewDialog: Failed to query metrics: {e}")
            return

        if polars_df is None or polars_df.is_empty():
            return

        metrics_df = polars_df.to_pandas()

        # Build cell_mapping for spatial lookup later
        self.cell_mapping = {}
        for _, row in metrics_df.iterrows():
            if "cell_id" not in row:
                continue
            cid = int(row["cell_id"])
            if all(col in row for col in ["y1", "x1", "y2", "x2"]):
                bbox = (int(row["y1"]), int(row["x1"]), int(row["y2"]), int(row["x2"]))
            elif "centroid_y" in row and "centroid_x" in row:
                cy, cx = row["centroid_y"], row["centroid_x"]
                bbox = (int(cy - 5), int(cx - 5), int(cy + 5), int(cx + 5))
            else:
                bbox = (0, 0, 10, 10)
            exclude = {"cell_id", "y1", "x1", "y2", "x2"}
            metrics = {c: row[c] for c in row.index if c not in exclude}
            self.cell_mapping[cid] = {"bbox": bbox, "metrics": metrics}

        # Select columns to show
        columns_to_show = [
            "cell_id", "morphology_class", "area", "perimeter",
            "aspect_ratio", "circularity", "solidity", "equivalent_diameter",
        ]
        available = [c for c in columns_to_show if c in metrics_df.columns]
        display_df = metrics_df[available].copy()

        numeric_cols = [c for c in available if c not in ("cell_id", "morphology_class")]
        if numeric_cols:
            display_df[numeric_cols] = display_df[numeric_cols].round(2)

        self.metrics_table.setRowCount(display_df.shape[0])
        self.metrics_table.setColumnCount(display_df.shape[1])
        self.metrics_table.setHorizontalHeaderLabels(list(display_df.columns))

        for r in range(display_df.shape[0]):
            for c_idx, val in enumerate(display_df.iloc[r]):
                item = QTableWidgetItem(str(val))
                self.metrics_table.setItem(r, c_idx, item)

                col_name = display_df.columns[c_idx]
                if col_name == "morphology_class":
                    morph = str(val)
                    if morph in self.morphology_colors:
                        bgr = self.morphology_colors[morph]
                        rgb = (bgr[2], bgr[1], bgr[0])
                        item.setBackground(QColor(*rgb))
                        item.setForeground(
                            QColor(255, 255, 255) if sum(rgb) < 384 else QColor(0, 0, 0)
                        )

    # ------------------------------------------------------------------ #
    # Cell selection and tracking                                         #
    # ------------------------------------------------------------------ #

    def _on_table_click(self, item):
        row = item.row()
        cell_id_item = self.metrics_table.item(row, 0)
        if not cell_id_item:
            return
        cell_id = int(cell_id_item.text())
        self._select_cell_for_tracking(cell_id)

    def _select_cell_for_tracking(self, cell_id):
        """Find cell_id in tracking data via spatial seg-label lookup, build lineage."""
        self.tracked_cell_lineage = {}
        self.selected_track_id = None

        labeled_mask = self._get_labeled_mask(self.initial_t)
        if labeled_mask is None:
            QMessageBox.warning(self, "Error", "No segmentation mask available.")
            return

        # Spatial lookup: which track sits on seg label == cell_id at initial_t?
        selected_track = None
        for track in self.lineage_tracks:
            if "t" not in track or "x" not in track or "y" not in track:
                continue
            for i, t_val in enumerate(track["t"]):
                if int(t_val) == self.initial_t:
                    tx = int(round(track["x"][i]))
                    ty = int(round(track["y"][i]))
                    if 0 <= ty < labeled_mask.shape[0] and 0 <= tx < labeled_mask.shape[1]:
                        if int(labeled_mask[ty, tx]) == cell_id:
                            selected_track = track
                    break
            if selected_track:
                break

        # Fallback: nearest distance
        if not selected_track:
            cell_cx, cell_cy = None, None
            if cell_id in self.cell_mapping:
                info = self.cell_mapping[cell_id]
                m = info.get("metrics", {})
                if "centroid_x" in m and "centroid_y" in m:
                    cell_cx, cell_cy = float(m["centroid_x"]), float(m["centroid_y"])
                if cell_cx is None:
                    y1, x1, y2, x2 = info["bbox"]
                    cell_cx, cell_cy = (x1 + x2) / 2, (y1 + y2) / 2

            if cell_cx is not None:
                min_dist = float("inf")
                for track in self.lineage_tracks:
                    if "t" not in track or "x" not in track or "y" not in track:
                        continue
                    for i, t_val in enumerate(track["t"]):
                        if int(t_val) == self.initial_t:
                            d = np.sqrt((track["x"][i] - cell_cx)**2 + (track["y"][i] - cell_cy)**2)
                            if d < min_dist:
                                min_dist = d
                                selected_track = track
                            break

                if not selected_track or min_dist > 30:
                    QMessageBox.warning(self, "Not Tracked", f"Cell {cell_id} could not be matched.")
                    return

        # Build lineage: map frame -> [track_ids]
        track_id = selected_track["ID"]
        self.selected_track_id = track_id

        for t_val in selected_track.get("t", []):
            t_int = int(t_val)
            self.tracked_cell_lineage.setdefault(t_int, []).append(track_id)

        # Recursively add children
        self._add_children(selected_track.get("children") or [])

        self.export_btn.setEnabled(True)
        n_frames = len(self.tracked_cell_lineage)
        self.status_label.setText(
            f"Cell {cell_id} -> Track {track_id} | tracked across {n_frames} frames | Press Play"
        )

        # Reset to first frame of this track and render
        first_frame = min(self.tracked_cell_lineage.keys())
        self.time_slider.setValue(first_frame)
        self._render_frame()

    def _add_children(self, child_ids):
        for cid in child_ids:
            child_track = None
            for tr in self.lineage_tracks:
                if tr["ID"] == cid:
                    child_track = tr
                    break
            if child_track and "t" in child_track:
                for t_val in child_track["t"]:
                    t_int = int(t_val)
                    self.tracked_cell_lineage.setdefault(t_int, []).append(cid)
                self._add_children(child_track.get("children") or [])

    # ------------------------------------------------------------------ #
    # Segmentation helpers                                                #
    # ------------------------------------------------------------------ #

    def _get_labeled_mask(self, t):
        cached = self._seg_cache.get(int(t))
        if cached is not None:
            return cached
        try:
            seg = self.image_data.segmentation_cache.with_model(self._seg_model)[t, self.position, 0]
            if seg is not None:
                seg = np.asarray(seg)
                if seg.max() <= 255 and len(np.unique(seg)) <= 100:
                    seg = sk_label(seg > 0)

                from nd2_analyzer.analysis.roi_helper import ROIHelper
                roi_mask = ROIHelper.get_roi_mask()
                if roi_mask is not None and roi_mask.shape == seg.shape:
                    seg = seg * roi_mask.astype(seg.dtype)

            self._seg_cache[int(t)] = seg
            return seg
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Frame rendering (same dim/bright as view_area)                      #
    # ------------------------------------------------------------------ #

    def _render_frame(self):
        t = self.current_t

        # Get the labeled/colored segmentation image (like "Classify Cells" view)
        img = self._get_labeled_colored_frame(t)
        if img is None:
            return

        # Apply tracking highlight (dim others, brighten tracked cell)
        img = self._apply_tracking_highlight(img, t)

        # Display image
        self._display_image(img)

        # Update trajectory plot + lineage tree
        self._render_trajectory(t)
        self._render_lineage(t)

    def _get_labeled_colored_frame(self, t):
        """Get the colored labeled segmentation image for this frame."""
        try:
            seg_service = self.image_data.segmentation_service
            raw = self._get_raw_frame(t)
            seg = self.image_data.segmentation_cache.with_model(self._seg_model)[t, self.position, 0]
            if seg is None or raw is None:
                return None
            seg = np.asarray(seg)

            from nd2_analyzer.analysis.roi_helper import ROIHelper
            roi_mask = ROIHelper.get_roi_mask()
            if roi_mask is not None and roi_mask.shape == seg.shape:
                seg = seg * roi_mask.astype(seg.dtype)

            colored = seg_service._apply_colormap(seg)
            if colored.dtype != np.uint8:
                colored = (colored * 255).astype(np.uint8)
            if len(colored.shape) == 2:
                colored = cv2.cvtColor(colored, cv2.COLOR_GRAY2RGB)
            return colored
        except Exception as e:
            print(f"CellViewDialog: Failed to get labeled frame: {e}")
            return None

    def _get_raw_frame(self, t):
        try:
            data = self.image_data.data
            if data.ndim == 5:
                return np.asarray(data[t, self.position, 0])
            elif data.ndim == 4:
                return np.asarray(data[t, self.position])
            elif data.ndim == 3:
                return np.asarray(data[t])
        except Exception:
            pass
        return None

    def _apply_tracking_highlight(self, img, t):
        """Dim/blur all cells, brighten only the tracked cell. No track lines on image."""
        if not self.tracked_cell_lineage or t not in self.tracked_cell_lineage:
            return img

        tracked_ids = self.tracked_cell_lineage[t]
        labeled = self._get_labeled_mask(t)
        if labeled is None:
            return img

        h, w = img.shape[:2]

        # Find tracked cell pixels via spatial lookup
        tracked_mask = np.zeros((h, w), dtype=bool)
        current_positions = []

        for cell_id in tracked_ids:
            for track in self.lineage_tracks:
                if track["ID"] != cell_id:
                    continue
                for i, t_val in enumerate(track.get("t", [])):
                    if int(t_val) == t and i < len(track["x"]) and i < len(track["y"]):
                        x = int(round(track["x"][i]))
                        y = int(round(track["y"][i]))
                        if 0 <= y < labeled.shape[0] and 0 <= x < labeled.shape[1]:
                            cell_label = int(labeled[y, x])
                            if cell_label > 0:
                                tracked_mask |= (labeled == cell_label)
                                current_positions.append((cell_id, x, y))
                        break
                break

        if not tracked_mask.any():
            return img

        # Blur + dim everything
        dimmed = cv2.GaussianBlur(img, (21, 21), 0)
        dimmed = (dimmed.astype(np.float32) * 0.3).astype(np.uint8)

        # Soft halo around tracked cell
        mask_u8 = tracked_mask.astype(np.uint8) * 255
        halo = cv2.GaussianBlur(mask_u8, (51, 51), 0).astype(np.float32) / 255.0
        halo = np.clip(halo, 0.0, 1.0)[..., None]

        out = dimmed.astype(np.float32) * (1.0 - halo) + img.astype(np.float32) * halo
        out[tracked_mask] = img[tracked_mask]
        out = out.astype(np.uint8)

        # Cell ID label
        for cell_id, x, y in current_positions:
            cv2.putText(out, str(cell_id), (x + 8, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, str(cell_id), (x + 8, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        return out

    def _display_image(self, img):
        h, w = img.shape[:2]
        channels = img.shape[2] if img.ndim == 3 else 1

        if channels == 3:
            qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
        else:
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qimg)

        # Scale to fit label while keeping aspect ratio
        label_size = self.image_label.size()
        scaled = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    # ------------------------------------------------------------------ #
    # Trajectory plot (synced with image)                                 #
    # ------------------------------------------------------------------ #

    def _render_trajectory(self, t):
        """Render the Cell Trajectories plot, synced to current time."""
        ax = self.traj_ax
        ax.clear()
        ax.set_facecolor("#111")

        tracked_ids = set()
        if self.tracked_cell_lineage:
            for ids in self.tracked_cell_lineage.values():
                tracked_ids.update(ids)

        if not tracked_ids:
            ax.text(0.5, 0.5, "Select a cell", color="#888",
                    ha="center", va="center", transform=ax.transAxes)
            self.traj_canvas.draw_idle()
            return

        # Draw selected cell's track with fading trail
        for track in self.lineage_tracks:
            if track["ID"] not in tracked_ids:
                continue
            xs, ys = [], []
            for i, t_val in enumerate(track.get("t", [])):
                if int(t_val) <= t:
                    xs.append(track["x"][i])
                    ys.append(track["y"][i])

            if len(xs) < 1:
                continue

            # Fading trail: draw segments with increasing alpha/width
            if len(xs) >= 2:
                n = len(xs)
                trail_len = min(n, 15)
                # Old part of trail (dim, thin)
                if n > trail_len:
                    ax.plot(xs[:n - trail_len + 1], ys[:n - trail_len + 1],
                            "-", color="#ffd54a", linewidth=1.0, alpha=0.2)
                # Recent trail: segments fade from dim to bright
                start = max(0, n - trail_len)
                for j in range(start, n - 1):
                    progress = (j - start) / max(trail_len - 1, 1)
                    alpha = 0.2 + 0.8 * progress
                    width = 1.5 + 2.0 * progress
                    ax.plot([xs[j], xs[j + 1]], [ys[j], ys[j + 1]],
                            "-", color="#ffd54a", linewidth=width, alpha=alpha)

            # Start marker
            ax.plot(xs[0], ys[0], "o", color="#2ecc71", markersize=6, zorder=10)
            # Current position — glowing dot
            ax.plot(xs[-1], ys[-1], "o", color="#ffd54a", markersize=12,
                    alpha=0.3, zorder=11)
            ax.plot(xs[-1], ys[-1], "o", color="#ffd54a", markersize=8,
                    markeredgecolor="#fff", markeredgewidth=1.5, zorder=12)

        # Auto-zoom to fit the full trajectory (all timepoints, not just up to t)
        # so the view stays stable as the line grows
        all_xs, all_ys = [], []
        for track in self.lineage_tracks:
            if track["ID"] in tracked_ids:
                all_xs.extend(track.get("x", []))
                all_ys.extend(track.get("y", []))
        if all_xs:
            pad_x = max((max(all_xs) - min(all_xs)) * 0.15, 20)
            pad_y = max((max(all_ys) - min(all_ys)) * 0.15, 20)
            ax.set_xlim(min(all_xs) - pad_x, max(all_xs) + pad_x)
            ax.set_ylim(max(all_ys) + pad_y, min(all_ys) - pad_y)

        ax.set_xlabel("X Position", color="#aaa", fontsize=9)
        ax.set_ylabel("Y Position", color="#aaa", fontsize=9)
        ax.set_title(f"Cell Track  |  t={t}", color="#ddd", fontsize=10)
        ax.tick_params(colors="#888", labelsize=7)
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_color("#444")

        self.traj_fig.tight_layout()
        self.traj_canvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Lineage tree                                                        #
    # ------------------------------------------------------------------ #

    def _render_lineage(self, t):
        """Render lineage tree that grows as divisions happen."""
        ax = self.tree_ax
        ax.clear()
        ax.set_facecolor("#111")

        tracked_ids = set()
        if self.tracked_cell_lineage:
            for ids in self.tracked_cell_lineage.values():
                tracked_ids.update(ids)

        if not tracked_ids or self.selected_track_id is None:
            ax.text(0.5, 0.5, "Select a cell", color="#888",
                    ha="center", va="center", transform=ax.transAxes)
            self.tree_canvas.draw_idle()
            return

        # Find the root ancestor of the selected track
        root_id = self._find_root(self.selected_track_id)

        # Collect family members alive up to current time
        family = self._get_descendants(root_id)
        family_alive = set()
        edges = []

        for cid in family:
            tr = self.tracks_by_id.get(cid)
            if not tr:
                continue
            ts = tr.get("t", [])
            if not ts:
                continue
            birth = int(min(ts))
            if birth <= t:
                family_alive.add(cid)
                parent = tr.get("parent")
                if parent is not None and int(parent) in family_alive:
                    edges.append((int(parent), cid))

        if not family_alive:
            ax.text(0.5, 0.5, "Not yet born", color="#888",
                    ha="center", va="center", transform=ax.transAxes)
            self.tree_canvas.draw_idle()
            return

        # Layout: generation-based top-down
        positions = {}
        layers: dict[int, list[int]] = {}

        for cid in family_alive:
            gen = self._get_generation(cid)
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
        for parent, child in edges:
            if parent in positions and child in positions:
                px, py = positions[parent]
                cx, cy = positions[child]
                ax.plot([px, cx], [py, cy], "-", color="#555", linewidth=1.5)

        # Draw nodes
        for cid, (x, y) in positions.items():
            tr = self.tracks_by_id.get(cid)
            ts = [int(tt) for tt in tr.get("t", [])] if tr else []
            is_alive = t in ts
            has_kids = bool(tr.get("children")) if tr else False
            just_divided = has_kids and ts and int(max(ts)) == t

            if just_divided:
                color = "#e74c3c"
                size = 180
            elif is_alive:
                color = "#ffd54a" if cid == root_id else "#5de1ff"
                size = 150
            elif has_kids and ts and int(max(ts)) <= t:
                color = "#e74c3c"
                size = 100
            else:
                color = "#888"
                size = 70

            ax.scatter([x], [y], s=size, c=color, zorder=3,
                       edgecolors="#000", linewidths=0.8)
            ax.annotate(str(cid), (x, y), ha="center", va="center",
                        color="#000" if is_alive else "#ddd",
                        fontsize=6, fontweight="bold", zorder=4)

            # Division marker
            if just_divided:
                ax.annotate("DIV", (x, y), xytext=(0, 12),
                            textcoords="offset points", ha="center",
                            color="#e74c3c", fontsize=6, fontweight="bold")

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"Lineage  |  t={t}", color="#ddd", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_color("#333")

        self.tree_fig.tight_layout()
        self.tree_canvas.draw_idle()

    def _find_root(self, cid):
        visited = set()
        while cid not in visited:
            visited.add(cid)
            tr = self.tracks_by_id.get(cid)
            if not tr or tr.get("parent") is None:
                return cid
            cid = int(tr["parent"])
        return cid

    def _get_descendants(self, root_id):
        result = {root_id}
        queue = [root_id]
        while queue:
            cid = queue.pop(0)
            tr = self.tracks_by_id.get(cid)
            if not tr:
                continue
            for ch in tr.get("children") or []:
                ch = int(ch)
                if ch not in result:
                    result.add(ch)
                    queue.append(ch)
        return result

    def _get_generation(self, cid):
        gen = 0
        visited = set()
        while cid not in visited:
            visited.add(cid)
            tr = self.tracks_by_id.get(cid)
            if not tr or tr.get("parent") is None:
                break
            cid = int(tr["parent"])
            gen += 1
        return gen

    # ------------------------------------------------------------------ #
    # Playback                                                            #
    # ------------------------------------------------------------------ #

    def _toggle_play(self):
        if self.is_playing:
            self.timer.stop()
            self.play_btn.setText("Play")
            self.is_playing = False
        else:
            if not self.tracked_cell_lineage:
                QMessageBox.information(self, "No Cell", "Click a cell first.")
                return
            # If we're at the end, restart from beginning
            last_frame = max(self.tracked_cell_lineage.keys())
            first_frame = min(self.tracked_cell_lineage.keys())
            if self.current_t >= last_frame:
                self.time_slider.setValue(first_frame)
            self._apply_speed()
            self.timer.start()
            self.play_btn.setText("Pause")
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
        # Stop at end of track
        if self.tracked_cell_lineage:
            last_frame = max(self.tracked_cell_lineage.keys())
            if nxt > last_frame:
                self.timer.stop()
                self.play_btn.setText("Play")
                self.is_playing = False
                return
        elif nxt > self.t_max:
            self.timer.stop()
            self.play_btn.setText("Play")
            self.is_playing = False
            return
        self.time_slider.setValue(nxt)

    def _on_slider_changed(self, value):
        self.current_t = int(value)
        self.time_label.setText(f"t={self.current_t}")
        self._render_frame()

    # ------------------------------------------------------------------ #
    # Cell History CSV export                                              #
    # ------------------------------------------------------------------ #

    def _export_cell_history(self):
        from nd2_analyzer.analysis.cell_history import CellHistoryBuilder

        if not self.lineage_tracks:
            QMessageBox.warning(self, "No Data", "No tracking data available.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Cell History CSV",
            f"cell_history_p{self.position}.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return

        try:
            builder = CellHistoryBuilder(self.lineage_tracks, self.metrics_service)
            builder.build(min_track_length=5)
            builder.export_to_csv(path)
            QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(builder.cell_database)} cells to:\n{path}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    # ------------------------------------------------------------------ #
    # GIF export                                                          #
    # ------------------------------------------------------------------ #

    def _export_gif(self):
        if not self.tracked_cell_lineage:
            QMessageBox.warning(self, "No Cell", "Select a cell first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Cell Animation",
            f"cell_track_{self.selected_track_id}.gif",
            "GIF (*.gif)",
        )
        if not path:
            return

        try:
            from PIL import Image as PILImage

            frames = []
            t_start = min(self.tracked_cell_lineage.keys())
            t_end = max(self.tracked_cell_lineage.keys())

            for t in range(t_start, t_end + 1):
                self.current_t = t
                img = self._get_labeled_colored_frame(t)
                if img is None:
                    continue
                img = self._apply_tracking_highlight(img, t)
                frames.append(PILImage.fromarray(img))

            if frames:
                frames[0].save(
                    path, save_all=True, append_images=frames[1:],
                    duration=self.BASE_TICK_MS, loop=0,
                )
                QMessageBox.information(self, "Done", f"Exported {len(frames)} frames to:\n{path}")
            else:
                QMessageBox.warning(self, "No Frames", "No frames could be rendered.")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))
        finally:
            self.time_slider.setValue(self.t_min)
