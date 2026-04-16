# tracking_widget.py
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QProgressDialog,
                               QMessageBox, QProgressBar)
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from skimage.measure import label
from pubsub import pub
from nd2_analyzer.analysis.metrics_service import MetricsService
import numpy as np
import os
import pickle


class TrackingWidget(QWidget):
    """
    Widget for basic cell tracking functionality.
    """

    def __init__(self, parent=None, metrics_service=None):
        super().__init__(parent)
        self.metrics_service = metrics_service if metrics_service else MetricsService()

        # Initialize state variables
        self.tracked_cells = None
        self.lineage_tracks = None
        self.has_channels = False
        self.image_data = None

        # Initialize UI components
        self.init_ui()

        # Subscribe to relevant messages
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

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

        # Clear visualization
        self.figure.clear()
        self.canvas.draw()

    def track_cells(self):
        """Process cell tracking with lineage detection - using segmentation cache like the old architecture"""
        print("\n======= track_cells method called =======")

        # Check if we already have tracking data
        if hasattr(self, "lineage_tracks") and self.lineage_tracks:
            print(f"TRACKING DATA EXISTS: Found {len(self.lineage_tracks)} existing lineage tracks")

            # If we have lineage_tracks but no tracked_cells, generate them
            if not hasattr(self, "tracked_cells") or not self.tracked_cells:
                print("Regenerating tracked_cells from lineage_tracks")
                # Filter tracks by length
                MIN_TRACK_LENGTH = 2  # Using a smaller value since 5 might be too restrictive
                filtered_tracks = [track for track in self.lineage_tracks if
                                'x' in track and len(track['x']) >= MIN_TRACK_LENGTH]
                filtered_tracks.sort(
                    key=lambda track: len(track['x']), reverse=True)

                MAX_TRACKS_TO_DISPLAY = 100
                self.tracked_cells = filtered_tracks[:MAX_TRACKS_TO_DISPLAY]
                print(f"Generated {len(self.tracked_cells)} tracked_cells from lineage data")
            else:
                print(f"TRACKED CELLS EXIST: Found {len(self.tracked_cells)} tracked cells")

            # Enable UI elements
            self.lineage_button.setEnabled(True)
            self.motility_button.setEnabled(True)

            # Visualize existing tracks
            print("Visualizing existing tracked cells")
            self.visualize_tracks()

            # Show information message
            QMessageBox.information(
                self, "Using Existing Tracking Data",
                f"Using existing tracking data with {len(self.lineage_tracks)} tracks."
            )
            print("Returning from track_cells without reprocessing")
            return

        # If we get here, we need to run tracking
        print("Continuing with tracking process...")

        if not self.image_data or not self.image_data.is_nd2:
            print("Error: Tracking requires an ND2 dataset")
            QMessageBox.warning(
                self, "Error", "Tracking requires an ND2 dataset.")
            return

        # Get segmentation parameters from segmentation widget
        segmentation_params = None
        def receive_params(params):
            nonlocal segmentation_params
            segmentation_params = params
        
        pub.sendMessage("get_segmentation_params", callback=receive_params)
        
        if not segmentation_params or not segmentation_params['positions']:
            QMessageBox.warning(
                self, "Error", 
                "No segmentation parameters found. Please run segmentation first and select positions/time range.")
            return
        
        # Use segmentation parameters
        selected_positions = segmentation_params['positions']
        time_start = segmentation_params['time_start'] 
        time_end = segmentation_params['time_end']
        selected_channel = segmentation_params['channel']
        
        print(f"Using segmentation parameters:")
        print(f"  Positions: {selected_positions}")
        print(f"  Time range: {time_start} - {time_end}")
        print(f"  Channel: {selected_channel}")

        try:
            # Calculate total frames to process
            total_frames = len(selected_positions) * (time_end - time_start + 1)

            progress = QProgressDialog(
                "Preparing frames for tracking...", "Cancel", 0, total_frames, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            # Set up segmentation cache with current model (if this attribute/method exists)
            segmentation_cache_available = False
            if hasattr(self.image_data, 'segmentation_cache'):
                if hasattr(self.image_data.segmentation_cache, 'with_model'):
                    # Get current model name from the ViewAreaWidget using callback pattern
                    current_model = "bact_phase_cp3"  # Default to correct model
                    def receive_model(model_name):
                        nonlocal current_model
                        if model_name:
                            current_model = model_name
                    
                    pub.sendMessage("get_current_model_callback", callback=receive_model)
                    print(f"Using segmentation model: {current_model}")
                    
                    # Set up the segmentation cache with the current model
                    try:
                        self.image_data.segmentation_cache.with_model(current_model)
                        segmentation_cache_available = True
                    except Exception as e:
                        print(f"Error setting up segmentation cache: {str(e)}")
                else:
                    # Cache exists but no with_model method - assume it's already configured
                    segmentation_cache_available = True
            
            if not segmentation_cache_available:
                print("Segmentation cache not available, will use metrics service for all frames")
            
            # First, find all frames that actually have segmentation data
            available_frames = []
            for p in selected_positions:
                for t in range(time_start, time_end + 1):
                    metrics_df = self.metrics_service.query(time=t, position=p, channel=selected_channel)
                    if not metrics_df.is_empty():
                        available_frames.append((t, p))
            
            if not available_frames:
                QMessageBox.warning(
                    self, "No Segmentation Data", 
                    f"No segmentation data found for the selected parameters.\n"
                    f"Positions: {selected_positions}\n"
                    f"Time range: {time_start}-{time_end}\n"
                    f"Channel: {selected_channel}\n\n"
                    f"Please run segmentation first.")
                return
            
            # print(f"Found {len(available_frames)} frames with existing segmentation data")
            # print(f"Available frames: {available_frames}")

            # Update progress dialog for actual frames to process
            total_frames = len(available_frames)
            progress = QProgressDialog(
                f"Processing {total_frames} frames with existing segmentation...", 
                "Cancel", 0, total_frames, self)
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

                # print(f"Processing frame T:{t}, P:{p}, C:{selected_channel}")

                # Get existing segmentation results from MetricsService
                metrics_df = self.metrics_service.query(time=t, position=p, channel=selected_channel)
                
                # Use existing segmentation results from MetricsService
                # print(f"Frame T:{t}, P:{p}: Using existing segmentation results from MetricsService")
                
                # Get frame dimensions from image data
                shape = self.image_data.data.shape
                frame_shape = (shape[-2], shape[-1])  # Last two dimensions are Y, X
                
                # Create binary mask from bounding boxes
                binary_mask = np.zeros(frame_shape, dtype=bool)
                
                for row in metrics_df.to_pandas().itertuples():
                    y1, x1, y2, x2 = row.y1, row.x1, row.y2, row.x2
                    binary_mask[y1:y2, x1:x2] = True
                
                # Apply connected component labeling
                labeled_frame = label(binary_mask)
                num_objects = np.max(labeled_frame)
                # print(f"Frame T:{t}, P:{p}: Using existing metrics - {num_objects} objects detected")
                labeled_frames.append(labeled_frame)

            progress.setValue(total_frames)

            if not labeled_frames:
                QMessageBox.warning(
                    self, "Error", "No data found for tracking.")
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

            from tracking import track_cells
            all_tracks, _ = track_cells(labeled_frames)
            self.lineage_tracks = all_tracks

            # Filter tracks by length for display
            MIN_TRACK_LENGTH = 5
            filtered_tracks = [track for track in all_tracks if len(
                track['x']) >= MIN_TRACK_LENGTH]
            filtered_tracks.sort(
                key=lambda track: len(track['x']), reverse=True)

            MAX_TRACKS_TO_DISPLAY = 100
            self.tracked_cells = filtered_tracks[:MAX_TRACKS_TO_DISPLAY]

            # Update UI
            self.lineage_button.setEnabled(True)
            self.motility_button.setEnabled(True)

            # Notify other components about tracking data
            pub.sendMessage("tracking_data_available",
                            lineage_tracks=self.lineage_tracks)

            # Visualize tracks
            self.visualize_tracks()

            # Show success message with detailed stats
            total_tracks = len(all_tracks)
            long_tracks = len(filtered_tracks)
            displayed_tracks = len(self.tracked_cells)
            
            QMessageBox.information(
                self, "Tracking Complete",
                f"Cell tracking completed successfully.\n\n"
                f"Total tracks detected: {total_tracks}\n"
                f"Tracks spanning {MIN_TRACK_LENGTH}+ frames: {long_tracks}\n"
                f"Tracks displayed: {displayed_tracks}"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(
                self, "Error", f"Failed to track cells: {str(e)}")
    
    def visualize_tracks(self):
        """Visualize tracked cell trajectories with statistics"""
        if not self.tracked_cells:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20', min(20, len(self.tracked_cells)))

        # Calculate displacement statistics
        displacements = []
        for track in self.tracked_cells:
            if len(track['x']) >= 2:  # Need at least start and end points
                # Calculate displacement (distance from start to end)
                start_x, start_y = track['x'][0], track['y'][0]
                end_x, end_y = track['x'][-1], track['y'][-1]
                displacement = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                displacements.append(displacement)

        # Calculate statistics
        avg_displacement = np.mean(displacements) if displacements else 0
        max_displacement = np.max(displacements) if displacements else 0

        # Plot each track
        for i, track in enumerate(self.tracked_cells):
            color = cmap(i % 20)
            ax.plot(track['x'], track['y'], '-', color=color,
                    linewidth=1, alpha=0.7, label=f"Track {track['ID']}")

            # Mark start and end points
            ax.plot(track['x'][0], track['y'][0],
                    'o', color=color, markersize=5)
            ax.plot(track['x'][-1], track['y'][-1],
                    's', color=color, markersize=5)

        # Add statistics box
        stats_text = f"Displaying top {len(self.tracked_cells)} tracks\n"
        stats_text += f"Avg displacement: {avg_displacement:.1f}px\n"
        stats_text += f"Max displacement: {max_displacement:.1f}px"
        
        # Add text box with statistics
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='left', bbox=props)

        # Set title and labels
        ax.set_title('Cell Trajectories')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        self.figure.tight_layout()
        self.canvas.draw()
    
    def show_lineage_dialog(self):
        """Open the lineage visualization dialog"""
        if not self.lineage_tracks:
            QMessageBox.warning(self, "Error", "No lineage data available.")
            return

        # Open the LineageDialog
        pub.sendMessage("show_lineage_dialog_request",
                        lineage_tracks=self.lineage_tracks)

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
                with open(tracking_path, 'wb') as f:
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

            with open(tracking_path, 'rb') as f:
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
        pub.sendMessage("show_motility_dialog_request",
                tracked_cells=self.tracked_cells,
                lineage_tracks=self.lineage_tracks,
                image_data=self.image_data)
