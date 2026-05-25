# tracking_widget.py
import json
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QProgressDialog,
    QMessageBox,
    QProgressBar,
    QFileDialog,
    QApplication,
    QComboBox,
    QLabel,
    QTabWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pubsub import pub
from skimage.measure import label
import imageio
import matplotlib.cm as cm

from nd2_analyzer.analysis.metrics_service import MetricsService
from nd2_analyzer.ui.widgets.environment_view_tab import EnvironmentViewTab
from nd2_analyzer.ui.widgets.digital_twin_tab import DigitalTwinTab
from nd2_analyzer.ui.widgets.track_validation_tab import TrackValidationTab


class TrackingWidget(QWidget):
    """
    Widget for basic cell tracking functionality.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics_service = MetricsService()

        # Initialize state variables
        self.tracked_cells = None
        self.lineage_tracks = None
        self.has_channels = False
        self.image_data = None

        # Per-position tracking store. Keys are ints (position p), values are
        # dicts: {"tracked_cells", "lineage_tracks", "algorithm",
        # "time_range", "saved_at"}. The flat self.tracked_cells /
        # self.lineage_tracks above are a view into whichever position is
        # currently selected, kept for backward compatibility with the rest
        # of the UI.
        self.tracks_by_position = {}

        # Initialize UI components
        self.init_ui()

        # Subscribe to relevant messages
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")
        pub.subscribe(self.provide_lineage_tracks, "get_lineage_tracks")

    def init_ui(self):
        """Initialize the user interface.

        Layout:
          - Selector row (algorithm + position)
          - Upstream/utility row: [Track Cells] ........ [Export Tracking Video]
          - QTabWidget with two tabs: "Cell View" and "Environment View".
            Each tab owns the analyses specific to that approach (paper-1
            deck, Figure 2): Cell View = per-cell tracking/histories/
            motility/lineage; Environment View = grid-based aggregates.
          - Progress bar
        """
        layout = QVBoxLayout(self)

        # --- Selector row: algorithm + position --- #
        selector_row = QHBoxLayout()

        selector_row.addWidget(QLabel("Algorithm:"))
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["btrack", "trackastra", "delta", "ultrack"])
        selector_row.addWidget(self.algorithm_combo)

        selector_row.addSpacing(20)

        selector_row.addWidget(QLabel("Position:"))
        self.position_combo = QComboBox()
        self.position_combo.currentIndexChanged.connect(self._on_position_changed)
        selector_row.addWidget(self.position_combo)

        selector_row.addStretch()
        layout.addLayout(selector_row)

        # --- Upstream/utility row: run tracking + export --- #
        upstream_row = QHBoxLayout()

        self.track_button = QPushButton("Track Cells")
        self.track_button.clicked.connect(self.track_cells)
        upstream_row.addWidget(self.track_button)

        upstream_row.addStretch()

        self.export_video_button = QPushButton("🎬 Export Tracking Video")
        self.export_video_button.clicked.connect(self.export_tracking_video)
        self.export_video_button.setEnabled(False)
        self.export_video_button.setStyleSheet("""
            QPushButton {
                background-color: #FF6B35;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E55A2B;
            }
            QPushButton:pressed {
                background-color: #CC4B21;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        upstream_row.addWidget(self.export_video_button)

        layout.addLayout(upstream_row)

        # --- Two analysis tabs: Cell View vs Environment View --- #
        self.view_tabs = QTabWidget()

        # Cell View tab ------------------------------------------------ #
        cell_view_tab = QWidget()
        cell_layout = QVBoxLayout(cell_view_tab)

        cell_buttons_row = QHBoxLayout()

        self.cell_view_button = QPushButton("📊 Cell View (Histories)")
        self.cell_view_button.clicked.connect(self.open_cell_view)
        self.cell_view_button.setEnabled(False)
        self.cell_view_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        cell_buttons_row.addWidget(self.cell_view_button)

        self.lineage_button = QPushButton("Show Lineage Trees")
        self.lineage_button.clicked.connect(self.show_lineage_dialog)
        self.lineage_button.setEnabled(False)
        cell_buttons_row.addWidget(self.lineage_button)

        self.motility_button = QPushButton("Analyze Motility")
        self.motility_button.clicked.connect(self.analyze_motility)
        self.motility_button.setEnabled(False)
        cell_buttons_row.addWidget(self.motility_button)

        cell_buttons_row.addStretch()
        cell_layout.addLayout(cell_buttons_row)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        cell_layout.addWidget(self.canvas)

        self.view_tabs.addTab(cell_view_tab, "Cell View")

        # Environment View tab ---------------------------------------- #
        self.env_view_tab = EnvironmentViewTab()
        self.view_tabs.addTab(self.env_view_tab, "Environment View")

        # Digital Twin tab --------------------------------------------- #
        self.dt_tab = DigitalTwinTab()
        self.view_tabs.addTab(self.dt_tab, "Digital Twin")

        # Track Validation tab ----------------------------------------- #
        self.validation_tab = TrackValidationTab()
        self.view_tabs.addTab(self.validation_tab, "Track Validation")

        layout.addWidget(self.view_tabs)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def on_image_data_loaded(self, image_data):
        """Handle new image data loading"""
        self.image_data = image_data

        # Reset tracking data
        self.tracked_cells = None
        self.lineage_tracks = None
        self.tracks_by_position = {}

        # Reset UI
        self.lineage_button.setEnabled(False)
        self.motility_button.setEnabled(False)
        self.cell_view_button.setEnabled(False)
        self.export_video_button.setEnabled(False)

        # Determine if image has channels
        shape = image_data.data.shape
        if len(shape) == 5:  # T, P, C, Y, X format
            self.has_channels = True
        else:
            self.has_channels = False

        # Populate the position dropdown (suppress the signal while filling it
        # so _on_position_changed doesn't fire for every addItem call).
        self.position_combo.blockSignals(True)
        self.position_combo.clear()
        p_count = shape[1] if len(shape) >= 4 else 1
        for p in range(p_count):
            self.position_combo.addItem(f"Position {p}", p)
        self.position_combo.blockSignals(False)

        # Clear visualization
        self.figure.clear()
        self.canvas.draw()

        # Clear the environment-view tab too; no tracks yet for this dataset.
        self._refresh_env_view()

    def _current_image_shape(self):
        """Return ``(height, width)`` of the current frames, or ``None``.

        Used by the Environment View tab to map pixel coords to grid
        buckets. Handles both 4D (T,P,Y,X) and 5D (T,P,C,Y,X) layouts.
        """
        if self.image_data is None:
            return None
        try:
            shape = self.image_data.data.shape
        except Exception:
            return None
        if len(shape) >= 2:
            return (int(shape[-2]), int(shape[-1]))
        return None

    def _refresh_env_view(self):
        """Push the currently-active tracks into the Environment View tab."""
        env = getattr(self, "env_view_tab", None)
        if env is None:
            return
        env.set_tracks(self.lineage_tracks, self._current_image_shape())

    def _refresh_dt_tab(self):
        """Push the currently-active tracks into the Digital Twin tab."""
        dt = getattr(self, "dt_tab", None)
        if dt is None:
            return
        dt.set_tracking_data(self.lineage_tracks, self.metrics_service)

    def _refresh_validation_tab(self):
        """Push tracks and image stacks into the Track Validation tab."""
        vt = getattr(self, "validation_tab", None)
        if vt is None:
            return
        raw_stack = self._get_raw_stack()
        vt.set_data(self.lineage_tracks, raw_images=raw_stack)

    def _get_raw_stack(self):
        """Assemble a (T, H, W) raw intensity stack for the current position."""
        if self.image_data is None:
            return None
        try:
            raw_data = self.image_data.data
            p = self.position_combo.currentData()
            if p is None:
                p = 0
            if raw_data.ndim == 5:
                # (T, P, C, Y, X) -- take channel 0
                return np.asarray(raw_data[:, p, 0])
            elif raw_data.ndim == 4:
                # (T, P, Y, X)
                return np.asarray(raw_data[:, p])
            elif raw_data.ndim == 3:
                return np.asarray(raw_data)
        except Exception as e:
            print(f"_get_raw_stack failed: {e}")
        return None

    def _export_tracking_data(self):
        """Auto-export tracking data to JSON for Digital Twin offline use."""
        if not self.lineage_tracks:
            return
        try:
            from pathlib import Path
            out_dir = Path(__file__).resolve().parents[4] / "digital_twin_output"
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / "tracking_data.json"

            export = []
            for track in self.lineage_tracks:
                t = {}
                for key in ("ID", "t", "x", "y", "parent", "children",
                            "morphology", "area", "major_axis_length",
                            "minor_axis_length", "orientation"):
                    val = track.get(key)
                    if val is not None:
                        if isinstance(val, np.ndarray):
                            t[key] = val.tolist()
                        elif isinstance(val, list):
                            t[key] = [
                                float(v) if isinstance(v, (np.floating, np.integer)) else v
                                for v in val
                            ]
                        elif isinstance(val, (np.floating, np.integer)):
                            t[key] = float(val)
                        else:
                            t[key] = val
                export.append(t)

            with open(out_path, "w") as f:
                json.dump(export, f)
            print(f"[DT Export] Saved {len(export)} tracks to {out_path}")
        except Exception as e:
            print(f"[DT Export] Warning: failed to export tracking data: {e}")

    def _on_position_changed(self, _index=None):
        """Swap the active tracks/UI when the position selector changes.

        Each position has its own independent tracks, so changing the
        selector must load that position's cached tracks (if any) into the
        legacy self.lineage_tracks / self.tracked_cells attributes that
        downstream UI still reads.
        """
        selected_p = self.position_combo.currentData()
        entry = self.tracks_by_position.get(selected_p)

        if entry is not None:
            self.tracked_cells = entry.get("tracked_cells")
            self.lineage_tracks = entry.get("lineage_tracks")
        else:
            self.tracked_cells = None
            self.lineage_tracks = None

        has_tracks = bool(self.lineage_tracks)
        self.lineage_button.setEnabled(has_tracks)
        self.motility_button.setEnabled(has_tracks)
        self.cell_view_button.setEnabled(has_tracks)
        self.export_video_button.setEnabled(has_tracks)
        self._refresh_dt_tab()
        self._refresh_validation_tab()

        # Let the rest of the app know which tracks are active for this p.
        pub.sendMessage(
            "tracking_data_available",
            lineage_tracks=self.lineage_tracks if has_tracks else None,
        )

        # Refresh the plot for the newly-selected position.
        self.figure.clear()
        if has_tracks and self.tracked_cells:
            self.visualize_tracks()
        else:
            self.canvas.draw()

        # Feed the environment-view tab the same tracks so its grid
        # aggregates stay in sync with the selected position.
        self._refresh_env_view()

    def provide_lineage_tracks(self, callback):
        """Provide lineage tracks to other components"""
        if hasattr(self, "lineage_tracks") and self.lineage_tracks:
            callback(self.lineage_tracks)
        else:
            callback(None)

    def track_cells(self):
        """Process cell tracking with lineage detection - using segmentation cache like the old architecture"""
        print("\n======= track_cells method called =======")

        # Check if we already have tracking data FOR THE CURRENTLY SELECTED
        # POSITION. Different positions are independent, so we only short
        # circuit when this specific p has cached tracks AND the cached
        # tracks were produced by the algorithm currently selected. If the
        # user switched the algorithm dropdown they almost certainly want
        # to re-track — ask before discarding work.
        current_p = self.position_combo.currentData()
        selected_algo = self.algorithm_combo.currentText()
        existing_entry = self.tracks_by_position.get(current_p)
        if existing_entry and existing_entry.get("lineage_tracks"):
            cached_algo = existing_entry.get("algorithm")

            if cached_algo and cached_algo != selected_algo:
                reply = QMessageBox.question(
                    self,
                    "Re-track with different algorithm?",
                    f"Position {current_p} already has "
                    f"{len(existing_entry['lineage_tracks'])} tracks from "
                    f"'{cached_algo}'.\n\n"
                    f"You selected '{selected_algo}'. Re-track with "
                    f"'{selected_algo}' (existing '{cached_algo}' tracks "
                    f"for this position will be replaced)?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if reply == QMessageBox.No:
                    # Keep the cached tracks; just surface them as the
                    # active view and bail out of track_cells.
                    self.lineage_tracks = existing_entry["lineage_tracks"]
                    self.tracked_cells = existing_entry.get("tracked_cells")
                    self.visualize_tracks()
                    return
                # Fall through to re-tracking below.
                print(
                    f"Re-tracking p={current_p}: '{cached_algo}' -> '{selected_algo}'"
                )
            else:
                # Same algorithm as cached, or no algorithm recorded (legacy
                # data). Reuse the cached tracks as before.
                lineage_tracks = existing_entry["lineage_tracks"]
                tracked_cells = existing_entry.get("tracked_cells")
                print(
                    f"TRACKING DATA EXISTS for p={current_p}: "
                    f"{len(lineage_tracks)} lineage tracks (algo={cached_algo})"
                )

                # Rebuild tracked_cells if missing
                if not tracked_cells:
                    print("Regenerating tracked_cells from lineage_tracks")
                    MIN_TRACK_LENGTH = 2
                    filtered_tracks = [
                        track
                        for track in lineage_tracks
                        if "x" in track and len(track["x"]) >= MIN_TRACK_LENGTH
                    ]
                    filtered_tracks.sort(key=lambda track: len(track["x"]), reverse=True)
                    MAX_TRACKS_TO_DISPLAY = 100
                    tracked_cells = filtered_tracks[:MAX_TRACKS_TO_DISPLAY]
                    existing_entry["tracked_cells"] = tracked_cells

                # Publish active view
                self.lineage_tracks = lineage_tracks
                self.tracked_cells = tracked_cells

                self.lineage_button.setEnabled(True)
                self.motility_button.setEnabled(True)
                self.cell_view_button.setEnabled(True)
                self.export_video_button.setEnabled(True)

                self.visualize_tracks()
                self._refresh_env_view()
                self._refresh_dt_tab()
                self._refresh_validation_tab()

                QMessageBox.information(
                    self,
                    "Using Existing Tracking Data",
                    f"Using cached tracking data for position {current_p} "
                    f"({len(lineage_tracks)} tracks, algorithm={cached_algo}).",
                )
                print("Returning from track_cells without reprocessing")
                return

        # If we get here, we need to run tracking
        print("Continuing with tracking process...")

        if not self.image_data:
            print("Error: Tracking requires a dataset")
            QMessageBox.warning(self, "Error", "Tracking requires a dataset.")
            return

        # Get segmentation parameters from segmentation widget
        segmentation_params = None

        def receive_params(params):
            nonlocal segmentation_params
            segmentation_params = params

        pub.sendMessage("get_segmentation_params", callback=receive_params)

        if not segmentation_params or not segmentation_params.get("positions"):
            QMessageBox.warning(
                self,
                "Error",
                "No segmentation parameters found. Please run segmentation first and select positions/time range.",
            )
            return

        # Use segmentation parameters, but scope to the single selected position
        available_positions = segmentation_params["positions"]
        selected_p = self.position_combo.currentData()

        if selected_p is None:
            QMessageBox.warning(
                self, "Error", "Please select a position to track."
            )
            return

        if selected_p not in available_positions:
            QMessageBox.warning(
                self,
                "Error",
                f"Position {selected_p} has not been segmented.\n"
                f"Segmented positions: {available_positions}\n\n"
                "Please segment this position first, or pick one that has been segmented.",
            )
            return

        selected_positions = [selected_p]
        time_start = segmentation_params["time_start"]
        time_end = segmentation_params["time_end"]
        selected_channel = segmentation_params["channel"]

        print(f"Using segmentation parameters:")
        print(f"  Position: {selected_p} (single)")
        print(f"  Time range: {time_start} - {time_end}")
        print(f"  Channel: {selected_channel}")

        try:
            # Get focus loss intervals to skip bad frames during tracking
            focus_loss_intervals = []
            time_interval_hours = 0.0833
            try:
                from nd2_analyzer.data.appstate import ApplicationState
                _appstate = ApplicationState.get_instance()
                if _appstate and _appstate.experiment:
                    focus_loss_intervals = _appstate.experiment.focus_loss_intervals or []
                    time_interval_hours = _appstate.experiment.time_interval_hours or 0.0833
            except Exception:
                pass

            # First, find all frames that actually have segmentation data
            available_frames = []
            skipped_focus = 0
            for p in selected_positions:
                for t in range(time_start, time_end + 1):
                    t_hours = t * time_interval_hours
                    if any(start <= t_hours < end for start, end in focus_loss_intervals):
                        skipped_focus += 1
                        continue
                    metrics_df = self.metrics_service.query_optimized(
                        time=t, position=p
                    )
                    if not metrics_df.is_empty():
                        available_frames.append((t, p))
            if skipped_focus > 0:
                print(f"  Skipped {skipped_focus} focus-loss frames during tracking")

            if not available_frames:
                QMessageBox.warning(
                    self,
                    "No Segmentation Data",
                    f"No segmentation data found for the selected parameters.\n"
                    f"Positions: {selected_positions}\n"
                    f"Time range: {time_start}-{time_end}\n"
                    f"Channel: {selected_channel}\n\n"
                    f"Please run segmentation first.",
                )
                return

            # Update progress dialog for actual frames to process
            total_frames = len(available_frames)
            progress = QProgressDialog(
                f"Processing {total_frames} frames with existing segmentation...",
                "Cancel",
                0,
                total_frames,
                self,
            )
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            # Prepare frames for tracking - only process frames with existing data
            labeled_frames = []
            raw_frames = []  # parallel list; only used by Trackastra
            frame_count = 0

            for t, p in available_frames:
                if progress.wasCanceled():
                    return
                progress.setValue(frame_count)
                frame_count += 1

                # Get the actual labeled segmentation from cache to preserve cell_ids
                from nd2_analyzer.data.image_data import ImageData

                image_data_instance = ImageData.get_instance()
                if not hasattr(image_data_instance, "segmentation_cache"):
                    print(f"ERROR: No segmentation cache available for T={t}, P={p}")
                    continue

                # Get segmentation from cache (this preserves the original cell_ids)
                labeled_frame = image_data_instance.segmentation_cache[t, p, selected_channel]

                if labeled_frame is None:
                    print(f"WARNING: No cached segmentation for T={t}, P={p}, C={selected_channel}")
                    continue

                # Check if it's binary or already labeled
                max_value = labeled_frame.max()
                unique_values = len(np.unique(labeled_frame))

                # print(f"Frame T={t}, P={p}: max={max_value}, unique={unique_values}")

                # If binary, label it (but this will create sequential IDs)
                # If already labeled (Cellpose/OmniPose), use as-is to preserve IDs
                if max_value <= 255 and unique_values <= 100:
                    # print(f"  Binary mask detected, calling label()...")
                    labeled_frame = label(labeled_frame)

                num_objects = np.max(labeled_frame)
                # print(f"  Found {num_objects} objects in frame")
                labeled_frames.append(labeled_frame)

                # Grab the matching raw intensity frame for Trackastra. Wrapped
                # in a try/except because data layout (channelled vs. not,
                # index vs. name) varies; on failure we just skip raw frames
                # and the tracker will fall back to a mask-based proxy.
                try:
                    raw_data = self.image_data.data
                    if raw_data.ndim == 5:
                        raw_frame = raw_data[t, p, selected_channel]
                    elif raw_data.ndim == 4:
                        raw_frame = raw_data[t, p]
                    else:
                        raw_frame = None
                    if raw_frame is not None:
                        raw_frames.append(np.asarray(raw_frame))
                    else:
                        raw_frames.append(None)
                except Exception as raw_exc:
                    print(f"Could not fetch raw frame for T={t}, P={p}: {raw_exc}")
                    raw_frames.append(None)

            progress.setValue(total_frames)

            if not labeled_frames:
                QMessageBox.warning(self, "Error", "No data found for tracking.")
                return

            labeled_frames = np.array(labeled_frames)
            print(f"Prepared {len(labeled_frames)} frames for tracking")

            # Print object statistics
            total_objects = sum(np.max(frame) for frame in labeled_frames)
            print(f"Total objects across all frames: {total_objects}")

            # Perform tracking
            progress.setLabelText("Running cell tracking with lineage detection...")
            progress.setValue(0)
            progress.setMaximum(100)

            from nd2_analyzer.analysis.tracking.tracking import run_tracker

            algorithm = self.algorithm_combo.currentText()
            print(f"Running tracker: {algorithm} on position {selected_p}")

            # Assemble raw intensity stack for Trackastra. Must align 1:1 with
            # labeled_frames; if any frame failed to fetch a raw image, skip.
            raw_stack = None
            if algorithm == "trackastra":
                if raw_frames and all(f is not None for f in raw_frames) \
                        and len(raw_frames) == len(labeled_frames):
                    raw_stack = np.array(raw_frames)
                    print(f"Passing raw intensity stack to Trackastra: shape={raw_stack.shape}")
                else:
                    print("Raw frames unavailable; Trackastra will use mask-as-proxy.")

            try:
                all_tracks, _ = run_tracker(
                    labeled_frames, algorithm=algorithm, raw_images=raw_stack
                )
            except NotImplementedError as e:
                progress.close()
                QMessageBox.warning(self, "Tracker not available", str(e))
                return

            # Tracking finished; dismiss the progress dialog before we draw.
            progress.close()

            # The tracker sees frames as 0..N-1 indices into labeled_frames;
            # remap each track's t back to absolute experiment time and tag
            # the position. Without this, downstream consumers (cell history,
            # MetricsService join, exports) cannot align tracks to the rest of
            # the data.
            local_to_abs_t = [t for (t, _p) in available_frames]
            for tr in all_tracks:
                tr["t"] = [int(local_to_abs_t[int(li)]) for li in tr.get("t", [])]
                tr["position"] = int(selected_p)

            # Filter tracks by ROI (if defined)
            from nd2_analyzer.analysis.roi_helper import ROIHelper
            if ROIHelper.has_roi():
                pre_roi = len(all_tracks)
                all_tracks = ROIHelper.filter_tracks_by_roi(all_tracks)
                print(f"  ROI filtering: {pre_roi} -> {len(all_tracks)} tracks")

            # Materialize the track_id ↔ (time, position, seg_label) join
            # into MetricsService.df so cell_history queries work correctly.
            try:
                attached = self.metrics_service.attach_track_ids(
                    all_tracks, position=selected_p
                )
                print(f"  attach_track_ids: tagged {attached} metric rows for P={selected_p}")
            except Exception as e:
                print(f"  WARNING: attach_track_ids failed: {e}")

            self.lineage_tracks = all_tracks

            # Filter tracks by length for display
            MIN_TRACK_LENGTH = 5
            filtered_tracks = [
                track for track in all_tracks if len(track["x"]) >= MIN_TRACK_LENGTH
            ]
            filtered_tracks.sort(key=lambda track: len(track["x"]), reverse=True)

            MAX_TRACKS_TO_DISPLAY = 100
            self.tracked_cells = filtered_tracks[:MAX_TRACKS_TO_DISPLAY]

            # Store per-position with metadata so save/load can round-trip.
            self.tracks_by_position[selected_p] = {
                "tracked_cells": self.tracked_cells,
                "lineage_tracks": self.lineage_tracks,
                "algorithm": algorithm,
                "time_range": [time_start, time_end],
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }

            # Update UI
            self.lineage_button.setEnabled(True)
            self.motility_button.setEnabled(True)
            self.cell_view_button.setEnabled(True)
            self.export_video_button.setEnabled(True)

            # Notify other components about tracking data
            pub.sendMessage(
                "tracking_data_available", lineage_tracks=self.lineage_tracks
            )

            # Auto-export tracking data for Digital Twin pipeline
            self._export_tracking_data()

            # Visualize tracks
            self.visualize_tracks()
            self._refresh_env_view()
            self._refresh_dt_tab()
            self._refresh_validation_tab()

            # Show success message with detailed stats
            total_tracks = len(all_tracks)
            long_tracks = len(filtered_tracks)
            displayed_tracks = len(self.tracked_cells)

            QMessageBox.information(
                self,
                "Tracking Complete",
                f"Cell tracking completed successfully.\n\n"
                f"Total tracks detected: {total_tracks}\n"
                f"Tracks spanning {MIN_TRACK_LENGTH}+ frames: {long_tracks}\n"
                f"Tracks displayed: {displayed_tracks}",
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            try:
                progress.close()
            except Exception:
                pass
            QMessageBox.warning(self, "Error", f"Failed to track cells: {str(e)}")

    def visualize_tracks(self):
        """Visualize tracked cell trajectories with statistics"""
        if not self.tracked_cells:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        import matplotlib.cm as cm

        cmap = cm.get_cmap("tab20", min(20, len(self.tracked_cells)))

        # Calculate displacement statistics
        displacements = []
        for track in self.tracked_cells:
            if len(track["x"]) >= 2:  # Need at least start and end points
                # Calculate displacement (distance from start to end)
                start_x, start_y = track["x"][0], track["y"][0]
                end_x, end_y = track["x"][-1], track["y"][-1]
                displacement = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
                displacements.append(displacement)

        # Calculate statistics
        avg_displacement = np.mean(displacements) if displacements else 0
        max_displacement = np.max(displacements) if displacements else 0

        # Plot each track
        for i, track in enumerate(self.tracked_cells):
            color = cmap(i % 20)
            ax.plot(
                track["x"],
                track["y"],
                "-",
                color=color,
                linewidth=1,
                alpha=0.7,
                label=f"Track {track['ID']}",
            )

            # Mark start and end points
            ax.plot(track["x"][0], track["y"][0], "o", color=color, markersize=5)
            ax.plot(track["x"][-1], track["y"][-1], "s", color=color, markersize=5)

        # Add statistics box
        stats_text = f"Displaying top {len(self.tracked_cells)} tracks\n"
        stats_text += f"Avg displacement: {avg_displacement:.1f}px\n"
        stats_text += f"Max displacement: {max_displacement:.1f}px"

        # Add text box with statistics
        props = dict(boxstyle="round", facecolor="white", alpha=0.8)
        ax.text(
            0.05,
            0.05,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=props,
        )

        # Set title and labels
        ax.set_title("Cell Trajectories")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

        self.figure.tight_layout()
        self.canvas.draw()

    def show_lineage_dialog(self):
        """Open the lineage visualization dialog"""
        if not self.lineage_tracks:
            QMessageBox.warning(self, "Error", "No lineage data available.")
            return

        # Open the LineageDialog
        pub.sendMessage(
            "show_lineage_dialog_request", lineage_tracks=self.lineage_tracks
        )

    def save_tracking_data(self, folder_path):
        """Save tracking data to ``<folder_path>/tracking_data/``.

        Layout (one pickle per position, plus a JSON manifest):

            tracking_data/
                p0_tracks.pkl
                p3_tracks.pkl
                manifest.json

        Each pickle stores ``{"tracked_cells", "lineage_tracks"}``. The
        manifest records per-position metadata (algorithm, time range, save
        date). This makes multi-position projects round-trip cleanly without
        one position's tracks clobbering another's.
        """
        try:
            if not self.tracks_by_position:
                print("No tracking data to save (tracks_by_position empty)")
                return False

            tracking_dir = os.path.join(folder_path, "tracking_data")
            os.makedirs(tracking_dir, exist_ok=True)

            manifest = {
                "version": 2,
                "positions": {},
            }

            for p, entry in self.tracks_by_position.items():
                lineage_tracks = entry.get("lineage_tracks")
                tracked_cells = entry.get("tracked_cells")
                if not lineage_tracks:
                    continue

                pkl_name = f"p{p}_tracks.pkl"
                pkl_path = os.path.join(tracking_dir, pkl_name)
                with open(pkl_path, "wb") as f:
                    pickle.dump(
                        {
                            "tracked_cells": tracked_cells,
                            "lineage_tracks": lineage_tracks,
                        },
                        f,
                    )

                manifest["positions"][str(p)] = {
                    "file": pkl_name,
                    "algorithm": entry.get("algorithm"),
                    "time_range": entry.get("time_range"),
                    "saved_at": entry.get("saved_at")
                    or datetime.now().isoformat(timespec="seconds"),
                    "n_lineage_tracks": len(lineage_tracks),
                    "n_tracked_cells": len(tracked_cells) if tracked_cells else 0,
                }
                print(
                    f"Saved p={p}: {len(lineage_tracks)} lineage tracks -> {pkl_path}"
                )

            if not manifest["positions"]:
                print("No non-empty per-position tracks to save")
                return False

            manifest_path = os.path.join(tracking_dir, "manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"Wrote manifest with {len(manifest['positions'])} position(s)")
            return True

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error saving tracking data: {str(e)}")
            return False

    def load_tracking_data(self, folder_path):
        """Load per-position tracking data.

        Prefers the new layout (``tracking_data/manifest.json`` + one pickle
        per position). Falls back to the legacy flat ``tracking_data.pkl``
        file so older projects still open; legacy data is assigned to the
        currently-selected position as a best effort.
        """
        try:
            self.tracks_by_position = {}
            loaded_any = False

            tracking_dir = os.path.join(folder_path, "tracking_data")
            manifest_path = os.path.join(tracking_dir, "manifest.json")

            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)

                for p_str, meta in manifest.get("positions", {}).items():
                    try:
                        p = int(p_str)
                    except ValueError:
                        print(f"Skipping non-integer position key {p_str!r}")
                        continue

                    pkl_path = os.path.join(tracking_dir, meta.get("file", ""))
                    if not os.path.exists(pkl_path):
                        print(f"Manifest references missing file: {pkl_path}")
                        continue

                    with open(pkl_path, "rb") as f:
                        payload = pickle.load(f)

                    # Backfill 'position' on tracks saved before the per-track
                    # position tag was introduced; pickles older than that fix
                    # have no idea which position they belong to without the
                    # manifest key, and downstream joins need it.
                    for tr in (payload.get("lineage_tracks") or []):
                        tr.setdefault("position", p)
                    for tr in (payload.get("tracked_cells") or []):
                        tr.setdefault("position", p)

                    self.tracks_by_position[p] = {
                        "tracked_cells": payload.get("tracked_cells"),
                        "lineage_tracks": payload.get("lineage_tracks"),
                        "algorithm": meta.get("algorithm"),
                        "time_range": meta.get("time_range"),
                        "saved_at": meta.get("saved_at"),
                    }
                    loaded_any = True
                    print(
                        f"Loaded p={p}: "
                        f"{len(payload.get('lineage_tracks') or [])} lineage tracks"
                        f" (algo={meta.get('algorithm')})"
                    )
            else:
                # Legacy fallback: a single flat pickle with no position tag.
                legacy_path = os.path.join(folder_path, "tracking_data.pkl")
                if os.path.exists(legacy_path):
                    print(f"Loading legacy tracking file {legacy_path}")
                    with open(legacy_path, "rb") as f:
                        tracking_data = pickle.load(f)

                    legacy_p = self.position_combo.currentData()
                    if legacy_p is None:
                        legacy_p = 0
                    self.tracks_by_position[legacy_p] = {
                        "tracked_cells": tracking_data.get("tracked_cells"),
                        "lineage_tracks": tracking_data.get("lineage_tracks"),
                        "algorithm": "btrack",
                        "time_range": None,
                        "saved_at": None,
                    }
                    loaded_any = True
                    print(
                        f"Legacy tracks assigned to p={legacy_p} "
                        f"({len(tracking_data.get('lineage_tracks') or [])} tracks)"
                    )

            if not loaded_any:
                print(f"No tracking data found under {folder_path}")
                return False

            # Populate the active view from whichever position is currently
            # selected in the combo (falls back to clearing if the selected
            # p has no stored tracks).
            self._on_position_changed()
            return True

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error loading tracking data: {str(e)}")
            return False

    def analyze_motility(self):
        """Open the motility analysis dialog"""
        if not self.lineage_tracks:
            QMessageBox.warning(self, "Error", "No tracking data available.")
            return

        # Open the MotilityDialog
        pub.sendMessage(
            "show_motility_dialog_request",
            tracked_cells=self.tracked_cells,
            lineage_tracks=self.lineage_tracks,
            image_data=self.image_data,
        )

    def open_cell_view(self):
        """Open the Cell View dialog to build and validate cell histories"""
        if not self.lineage_tracks:
            QMessageBox.warning(self, "Error", "No tracking data available.")
            return

        # Import the dialog
        from nd2_analyzer.ui.widgets.cell_view_dialog import CellViewDialog
        from nd2_analyzer.data.appstate import ApplicationState

        # Pick the position from the main view area's slider
        # (ApplicationState.view_index = (t, p, c)). Fall back to the
        # tracking tab's position combo only if the app slider is unset.
        selected_p = None
        appstate = ApplicationState.get_instance()
        if appstate and appstate.view_index:
            selected_p = int(appstate.view_index[1])
        if selected_p is None:
            selected_p = self.position_combo.currentData()
        if selected_p is None:
            selected_p = 0

        dialog = CellViewDialog(
            self.lineage_tracks,
            self.metrics_service,
            self.image_data,
            position=int(selected_p),
            parent=self,
        )
        dialog.exec()

    def export_tracking_video(self):
        """Export tracking visualization as video"""
        if not self.tracked_cells or not self.lineage_tracks:
            QMessageBox.warning(self, "Error", "No tracking data available to export.")
            return

        # Get save path from user
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_name = f"tracking_animation_{timestamp}.gif"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Tracking Animation", default_name,
            "Animated GIF (*.gif)")

        if not file_path:
            return

        try:
            # Get all unique time points from tracks
            all_time_points = set()
            for track in self.lineage_tracks:
                if 't' in track:
                    all_time_points.update(track['t'])
                else:
                    all_time_points.update(range(len(track.get('x', []))))

            if not all_time_points:
                QMessageBox.warning(self, "Error", "No time data found in tracks.")
                return

            time_points = sorted(all_time_points)

            # Create progress dialog
            progress = QProgressDialog(
                "Generating GIF frames...", "Cancel", 0, len(time_points), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            # Generate frames
            frames = []
            cmap = cm.get_cmap("tab20", min(20, len(self.tracked_cells)))

            for frame_idx, t in enumerate(time_points):
                if progress.wasCanceled():
                    return

                progress.setValue(frame_idx)
                QApplication.processEvents()

                # Create figure for this frame
                fig = plt.figure(figsize=(10, 8), dpi=100)
                ax = fig.add_subplot(111)

                # Plot tracks up to current time point
                for i, track in enumerate(self.tracked_cells):
                    track_times = track.get('t', list(range(len(track.get('x', [])))))

                    # Get indices up to current time
                    indices = [idx for idx, tt in enumerate(track_times) if tt <= t]

                    if indices:
                        color = cmap(i % 20)
                        x_vals = [track['x'][idx] for idx in indices]
                        y_vals = [track['y'][idx] for idx in indices]

                        # Plot track path
                        ax.plot(x_vals, y_vals, '-', color=color, linewidth=1, alpha=0.7)

                        # Plot current position (last point up to time t)
                        ax.plot(x_vals[-1], y_vals[-1], 'o', color=color,
                               markersize=8, alpha=0.9)

                        # Mark start point
                        if len(indices) == len(track_times):  # Full track shown
                            ax.plot(x_vals[0], y_vals[0], 's', color=color,
                                   markersize=5, alpha=0.5)

                # Add time annotation
                ax.text(0.02, 0.98, f'Time: {t}', transform=ax.transAxes,
                       fontsize=14, fontweight='bold', color='black',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='top')

                ax.set_title(f"Cell Tracking - Frame {t}")
                ax.set_xlabel("X Position (pixels)")
                ax.set_ylabel("Y Position (pixels)")
                ax.grid(True, alpha=0.3)

                # Render to image
                fig.canvas.draw()
                width, height = fig.get_size_inches() * fig.dpi
                width, height = int(width), int(height)

                buf = fig.canvas.buffer_rgba()
                img_array = np.asarray(buf).reshape((height, width, 4))
                img_array = img_array[:, :, :3]  # Convert RGBA to RGB

                frames.append(img_array)
                plt.close(fig)

            progress.setValue(len(time_points))

            # Save GIF
            progress.setLabelText("Saving GIF file...")
            QApplication.processEvents()

            if not file_path.lower().endswith('.gif'):
                file_path += '.gif'

            fps = 5  # 5 frames per second
            imageio.mimsave(file_path, frames, fps=fps)

            progress.close()
            QMessageBox.information(self, "Export Complete",
                                   f"Tracking animation exported to:\n{file_path}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Export Error",
                               f"Failed to export GIF: {str(e)}")
