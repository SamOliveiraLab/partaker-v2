# tracking_widget.py
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
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pubsub import pub
from skimage.measure import label
import imageio
import matplotlib.cm as cm

from nd2_analyzer.analysis.metrics_service import MetricsService


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

        # Initialize UI components
        self.init_ui()

        # Subscribe to relevant messages
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")
        pub.subscribe(self.provide_lineage_tracks, "get_lineage_tracks")

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

        # Selector row: algorithm + position
        selector_row = QHBoxLayout()

        selector_row.addWidget(QLabel("Algorithm:"))
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["btrack", "trackastra", "delta", "ultrack"])
        selector_row.addWidget(self.algorithm_combo)

        selector_row.addSpacing(20)

        selector_row.addWidget(QLabel("Position:"))
        self.position_combo = QComboBox()
        selector_row.addWidget(self.position_combo)

        selector_row.addStretch()
        layout.addLayout(selector_row)

        # Create buttons layout
        buttons_layout = QHBoxLayout()

        # Track cells button
        self.track_button = QPushButton("Track Cells")
        self.track_button.clicked.connect(self.track_cells)
        buttons_layout.addWidget(self.track_button)

        # Show lineage tree button
        self.lineage_button = QPushButton("Show Lineage Trees")
        self.lineage_button.clicked.connect(self.show_lineage_dialog)
        self.lineage_button.setEnabled(False)
        buttons_layout.addWidget(self.lineage_button)

        # Motility analysis button
        self.motility_button = QPushButton("Analyze Motility")
        self.motility_button.clicked.connect(self.analyze_motility)
        self.motility_button.setEnabled(False)
        buttons_layout.addWidget(self.motility_button)

        # Cell View button (NEW!)
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
        buttons_layout.addWidget(self.cell_view_button)

        # Export tracking video button
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
        buttons_layout.addWidget(self.export_video_button)

        layout.addLayout(buttons_layout)

        # Add visualization area
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

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

        # Reset UI
        self.lineage_button.setEnabled(False)
        self.motility_button.setEnabled(False)

        # Determine if image has channels
        shape = image_data.data.shape
        if len(shape) == 5:  # T, P, C, Y, X format
            self.has_channels = True
        else:
            self.has_channels = False

        # Populate the position dropdown
        self.position_combo.clear()
        p_count = shape[1] if len(shape) >= 4 else 1
        for p in range(p_count):
            self.position_combo.addItem(f"Position {p}", p)

        # Clear visualization
        self.figure.clear()
        self.canvas.draw()

    def provide_lineage_tracks(self, callback):
        """Provide lineage tracks to other components"""
        if hasattr(self, "lineage_tracks") and self.lineage_tracks:
            callback(self.lineage_tracks)
        else:
            callback(None)

    def track_cells(self):
        """Process cell tracking with lineage detection - using segmentation cache like the old architecture"""
        print("\n======= track_cells method called =======")

        # Check if we already have tracking data
        if hasattr(self, "lineage_tracks") and self.lineage_tracks:
            print(
                f"TRACKING DATA EXISTS: Found {len(self.lineage_tracks)} existing lineage tracks"
            )

            # If we have lineage_tracks but no tracked_cells, generate them
            if not hasattr(self, "tracked_cells") or not self.tracked_cells:
                print("Regenerating tracked_cells from lineage_tracks")
                # Filter tracks by length
                MIN_TRACK_LENGTH = (
                    2  # Using a smaller value since 5 might be too restrictive
                )
                filtered_tracks = [
                    track
                    for track in self.lineage_tracks
                    if "x" in track and len(track["x"]) >= MIN_TRACK_LENGTH
                ]
                filtered_tracks.sort(key=lambda track: len(track["x"]), reverse=True)

                MAX_TRACKS_TO_DISPLAY = 100
                self.tracked_cells = filtered_tracks[:MAX_TRACKS_TO_DISPLAY]
                print(
                    f"Generated {len(self.tracked_cells)} tracked_cells from lineage data"
                )
            else:
                print(
                    f"TRACKED CELLS EXIST: Found {len(self.tracked_cells)} tracked cells"
                )

            # Enable UI elements
            self.lineage_button.setEnabled(True)
            self.motility_button.setEnabled(True)
            self.export_video_button.setEnabled(True)

            # Visualize existing tracks
            print("Visualizing existing tracked cells")
            self.visualize_tracks()

            # Show information message
            QMessageBox.information(
                self,
                "Using Existing Tracking Data",
                f"Using existing tracking data with {len(self.lineage_tracks)} tracks.",
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
            # First, find all frames that actually have segmentation data
            available_frames = []
            for p in selected_positions:
                for t in range(time_start, time_end + 1):
                    metrics_df = self.metrics_service.query_optimized(
                        time=t, position=p
                    )
                    if not metrics_df.is_empty():
                        available_frames.append((t, p))

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
            try:
                all_tracks, _ = run_tracker(labeled_frames, algorithm=algorithm)
            except NotImplementedError as e:
                QMessageBox.warning(self, "Tracker not available", str(e))
                return
            self.lineage_tracks = all_tracks

            # Filter tracks by length for display
            MIN_TRACK_LENGTH = 5
            filtered_tracks = [
                track for track in all_tracks if len(track["x"]) >= MIN_TRACK_LENGTH
            ]
            filtered_tracks.sort(key=lambda track: len(track["x"]), reverse=True)

            MAX_TRACKS_TO_DISPLAY = 100
            self.tracked_cells = filtered_tracks[:MAX_TRACKS_TO_DISPLAY]

            # Update UI
            self.lineage_button.setEnabled(True)
            self.motility_button.setEnabled(True)
            self.cell_view_button.setEnabled(True)
            self.export_video_button.setEnabled(True)

            # Notify other components about tracking data
            pub.sendMessage(
                "tracking_data_available", lineage_tracks=self.lineage_tracks
            )

            # Visualize tracks
            self.visualize_tracks()

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
        """Save tracking data to a file in the specified folder"""
        try:
            # Ensure the folder exists
            os.makedirs(folder_path, exist_ok=True)

            # Prepare tracking data dictionary
            tracking_data = {}

            if hasattr(self, "tracked_cells") and self.tracked_cells is not None:
                tracking_data["tracked_cells"] = self.tracked_cells
                print(f"Saving {len(self.tracked_cells)} tracked cells")
            else:
                print("No tracked_cells to save")

            if hasattr(self, "lineage_tracks") and self.lineage_tracks is not None:
                tracking_data["lineage_tracks"] = self.lineage_tracks
                print(f"Saving {len(self.lineage_tracks)} lineage tracks")
            else:
                print("No lineage_tracks to save")

            # Save data if we have any
            if tracking_data:
                tracking_path = os.path.join(folder_path, "tracking_data.pkl")
                with open(tracking_path, "wb") as f:
                    pickle.dump(tracking_data, f)
                print(f"Tracking data saved to {tracking_path}")
                return True
            else:
                print("No tracking data to save")
                return False

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error saving tracking data: {str(e)}")
            return False

    def load_tracking_data(self, folder_path):
        """
        Load tracking data from a file in the specified folder.

        Args:
            folder_path: Path to the folder containing the data

        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        try:
            tracking_path = os.path.join(folder_path, "tracking_data.pkl")

            if not os.path.exists(tracking_path):
                print(f"No tracking data found at {tracking_path}")
                return False

            with open(tracking_path, "rb") as f:
                tracking_data = pickle.load(f)

            # Load tracked cells if available
            if "tracked_cells" in tracking_data and tracking_data["tracked_cells"]:
                self.tracked_cells = tracking_data["tracked_cells"]
                print(f"Loaded {len(self.tracked_cells)} tracked cells")

            # Load lineage tracks if available
            if "lineage_tracks" in tracking_data and tracking_data["lineage_tracks"]:
                self.lineage_tracks = tracking_data["lineage_tracks"]
                print(f"Loaded {len(self.lineage_tracks)} lineage tracks")

            # Update UI based on loaded data
            if self.lineage_tracks:
                self.lineage_button.setEnabled(True)
                self.motility_button.setEnabled(True)
                self.cell_view_button.setEnabled(True)

                # Notify other components about tracking data (especially MorphologyWidget)
                print(
                    f"Publishing tracking_data_available message with {len(self.lineage_tracks)} tracks"
                )
                pub.sendMessage(
                    "tracking_data_available", lineage_tracks=self.lineage_tracks
                )

                # Visualize tracks if we have tracked_cells
                if self.tracked_cells:
                    self.visualize_tracks()

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

        # Open the dialog
        dialog = CellViewDialog(
            self.lineage_tracks,
            self.metrics_service,
            self.image_data,
            self
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
