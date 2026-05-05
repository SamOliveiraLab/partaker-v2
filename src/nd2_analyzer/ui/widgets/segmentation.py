from ..biofilms.colony_separator import ColonySeparator
from PySide6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QLabel,
                               QSlider, QPushButton, QSpinBox, QCheckBox,
                               QFrame, QSplitter, QWidget, QComboBox, QAbstractItemView, QListWidget, QProgressBar,
                               QRadioButton, QDialog, QFileDialog)
from PySide6.QtCore import Qt, QTimer
from pubsub import pub
import numpy as np
from ..biofilms.analysis_mode import AnalysisMode, AnalysisModeConfig
from ..biofilms.config.biofilm_config import BiofilmConfig
from ..biofilms.colony_analysis import ColonyGrouper, ColonyTracker
from ..biofilms.colony_time_series_exporter import ColonyTimeSeriesExporter


class SegmentationWidget(QWidget):
    """
    Widget for batch segmentation of time-lapse microscopy images.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_model = None
        self.time_range = (0, 0)
        self.positions = []
        self.channel = 0
        self.mode = "segmented"
        self.is_segmenting = False
        self.queue = []
        self.processed_frames = set()
        self.segmented_storage = {}
        pub.subscribe(self.receive_external_segmentation, "segmentation_result_available")

        # Add biofilm analysis components
        self.analysis_config = AnalysisModeConfig()
        self.biofilm_config = BiofilmConfig()
        self.colony_grouper = None
        self.colony_tracker = None

        # Add colony separator
        self.colony_separator = ColonySeparator()
        self.current_raw_image = None
        self.current_segmented_image = None
        self.colony_overlay_visible = False

        # Set up the UI
        self.init_ui()

        # Subscribe to relevant topics
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")
        pub.subscribe(self.on_image_ready, "image_ready")
        print("✅ SegmentationWidget: Subscribed to 'image_ready' messages")

        # Subscribe to provide segmentation parameters to other widgets
        pub.subscribe(self.provide_segmentation_params, "get_segmentation_params")

        # Timer for handling segmentation requests with a slight delay
        self.request_timer = QTimer(self)
        self.request_timer.setSingleShot(True)
        self.request_timer.timeout.connect(self.process_next_in_queue)

        pub.subscribe(self.on_polygon_point_added, "polygon_point_added")

    def provide_segmentation_params(self, callback):
        """Provide segmentation parameters to other widgets via callback"""
        # Get selected positions
        selected_positions = []
        for item in self.position_list.selectedItems():
            pos = int(item.text().split(" ")[1])
            selected_positions.append(pos)

        params = {
            'positions': selected_positions,
            'time_start': self.time_start_spin.value(),
            'time_end': self.time_end_spin.value(),
            'channel': self.channel_combo.currentIndex(),
            'model': self.model_combo.currentText() if hasattr(self, 'model_combo') else 'bact_phase_cp3'
        }

        print(f"SegmentationWidget: Providing parameters - {params}")
        callback(params)

    def on_polygon_point_added(self, x, y, total_points):
        """Handle when a polygon point is added"""
        self.progress_label.setText(
            f"Polygon point {total_points} added at ({x}, {y}). Continue clicking or finish polygon.")

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

        # Model selection
        model_group = QGroupBox("Segmentation Model")
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Add biofilm configuration if in biofilm mode
        # REMOVED: Biofilm-specific features moved to dedicated Colony Separation tab
        # if self.analysis_config.is_biofilm_mode():
        #     self.add_biofilm_configuration_panel(layout)

        # Position selection (existing code)
        position_group = QGroupBox("Positions")
        position_layout = QVBoxLayout()

        # Position list with checkboxes
        self.position_list = QListWidget()
        self.position_list.setSelectionMode(QAbstractItemView.MultiSelection)
        position_layout.addWidget(self.position_list)

        # Quick selection buttons
        pos_buttons_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_positions)
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self.select_no_positions)
        pos_buttons_layout.addWidget(select_all_btn)
        pos_buttons_layout.addWidget(select_none_btn)
        position_layout.addLayout(pos_buttons_layout)

        position_group.setLayout(position_layout)
        layout.addWidget(position_group)

        # Time range selection
        time_group = QGroupBox("Time Range")
        time_layout = QHBoxLayout()

        time_layout.addWidget(QLabel("From:"))
        self.time_start_spin = QSpinBox()
        self.time_start_spin.setMinimum(0)
        time_layout.addWidget(self.time_start_spin)

        time_layout.addWidget(QLabel("To:"))
        self.time_end_spin = QSpinBox()
        self.time_end_spin.setMinimum(0)
        time_layout.addWidget(self.time_end_spin)

        time_group.setLayout(time_layout)
        layout.addWidget(time_group)

        # Channel and Mode selection
        options_layout = QHBoxLayout()

        # Channel selection
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        channel_layout.addWidget(self.channel_combo)
        options_layout.addLayout(channel_layout)

        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["segmented", "overlay", "labeled"])
        mode_layout.addWidget(self.mode_combo)
        options_layout.addLayout(mode_layout)

        layout.addLayout(options_layout)

        # Segmentation controls
        controls_layout = QHBoxLayout()
        self.segment_button = QPushButton("Segment Selected")
        self.segment_button.clicked.connect(self.start_segmentation)

        # Add colony analysis button if in biofilm mode
        if self.analysis_config.is_biofilm_mode():
            self.analyze_colonies_button = QPushButton("Analyze Colonies")
            self.analyze_colonies_button.clicked.connect(self.analyze_biofilm_colonies)
            self.analyze_colonies_button.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
            controls_layout.addWidget(self.analyze_colonies_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_segmentation)
        self.cancel_button.setEnabled(False)

        self.convert_tif_button = QPushButton("Convert to TIF")
        self.convert_tif_button.clicked.connect(self.convert_to_tif)

        self.export_raw_gif_button = QPushButton("Export Raw GIF")
        self.export_raw_gif_button.clicked.connect(self.export_raw_gif)

        controls_layout.addWidget(self.segment_button)
        controls_layout.addWidget(self.cancel_button)
        controls_layout.addWidget(self.convert_tif_button)
        controls_layout.addWidget(self.export_raw_gif_button)
        layout.addLayout(controls_layout)

        # Progress display
        self.progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)

        self.progress_label = QLabel("Ready")

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)

        self.progress_group.setLayout(progress_layout)
        layout.addWidget(self.progress_group)

        self.setLayout(layout)

    def on_image_data_loaded(self, image_data):
        """Handle new image data loading"""
        # Get dimensions from image_data
        shape = image_data.data.shape
        t_max = shape[0] - 1
        p_max = shape[1] - 1
        c_max = shape[2] - 1

        # Update time range spinboxes
        self.time_start_spin.setMaximum(t_max)
        self.time_end_spin.setMaximum(t_max)
        self.time_end_spin.setValue(t_max)

        # Populate position list
        self.position_list.clear()
        for p in range(p_max + 1):
            self.position_list.addItem(f"Position {p}")

        # Select all positions by default
        self.select_all_positions()

        # Populate channel combo
        self.channel_combo.clear()
        for c in range(c_max + 1):
            self.channel_combo.addItem(f"Channel {c}")

        # Populate model combo
        self.model_combo.clear()
        from nd2_analyzer.analysis.segmentation.segmentation_models import SegmentationModels
        self.model_combo.addItems([
            SegmentationModels.OMNIPOSE_BACT_PHASE,
            SegmentationModels.OMNIPOSE_BACT_FLUOR,
            SegmentationModels.CELLPOSE_BACT_PHASE,
            SegmentationModels.CELLPOSE_BACT_FLUOR,
            SegmentationModels.CELLPOSE,
            SegmentationModels.UNET
        ])

    def select_all_positions(self):
        """Select all positions in the list"""
        for i in range(self.position_list.count()):
            self.position_list.item(i).setSelected(True)

    def select_no_positions(self):
        """Deselect all positions in the list"""
        for i in range(self.position_list.count()):
            self.position_list.item(i).setSelected(False)

    def start_segmentation(self):
        """Start the segmentation process"""
        if self.is_segmenting:
            return

        # Get selected positions
        self.positions = []
        for item in self.position_list.selectedItems():
            pos = int(item.text().split(" ")[1])
            self.positions.append(pos)

        if not self.positions:
            self.progress_label.setText("No positions selected")
            return

        # Get time range
        t_start = self.time_start_spin.value()
        t_end = self.time_end_spin.value()

        if t_end < t_start:
            self.progress_label.setText("Invalid time range")
            return

        self.time_range = (t_start, t_end)

        # Get channel and mode
        self.channel = self.channel_combo.currentIndex()
        self.mode = self.mode_combo.currentText()
        self.current_model = self.model_combo.currentText()

        # Check if metrics data already exists for these parameters
        from nd2_analyzer.analysis.metrics_service import MetricsService
        metrics_service = MetricsService()

        # Get focus loss intervals to skip bad frames
        _focus_loss_skip = set()
        try:
            from nd2_analyzer.data.appstate import ApplicationState
            _appstate = ApplicationState.get_instance()
            if _appstate and _appstate.experiment and _appstate.experiment.focus_loss_intervals:
                for t in range(t_start, t_end + 1):
                    if _appstate.experiment.is_focus_loss_frame(t):
                        _focus_loss_skip.add(t)
                if _focus_loss_skip:
                    print(f"[Segmentation] Skipping {len(_focus_loss_skip)} focus-loss frames")
        except Exception:
            pass

        # Check if we need to segment anything
        frames_to_segment = []
        for p in self.positions:
            for t in range(t_start, t_end + 1):
                if t in _focus_loss_skip:
                    continue
                # If metrics don't exist for this frame, add it to the queue
                if not metrics_service.has_data_for(position=p, time=t, channel=self.channel):
                    frames_to_segment.append((t, p))

        # If all frames already have metrics, no need to segment
        if not frames_to_segment:
            self.progress_label.setText("All frames already segmented")
            return

        # Build queue of frames to process
        self.queue = frames_to_segment

        # Set up progress tracking
        self.processed_frames.clear()
        total_frames = len(self.queue)
        self.progress_bar.setMaximum(total_frames)
        self.progress_bar.setValue(0)

        # Update UI
        self.segment_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.is_segmenting = True
        self.progress_label.setText(f"Segmenting {total_frames} frames...")

        # Start processing
        self.process_next_in_queue()

    def process_next_in_queue(self):
        """Process the next frame in the queue"""
        if not self.is_segmenting or not self.queue:
            if self.is_segmenting:
                self._segmentation_finished()
            return

        # Get next frame to process
        time, position = self.queue[0]

        # Request segmentation
        pub.sendMessage("segmented_image_request",
                        time=time,
                        position=position,
                        channel=self.channel,
                        mode=self.mode,
                        model=self.current_model)

        # Update status
        self.progress_label.setText(f"Processing T={time}, P={position}")

    def on_image_ready(self, image, time, position, channel, mode):
        """Handle when images are ready"""
        print(
            f"📸 SegmentationWidget.on_image_ready CALLED: mode={mode}, T={time}, P={position}, C={channel}, shape={image.shape if image is not None else 'None'}")

        # Store raw image for colony detection
        if mode == "normal" and hasattr(self, 'analysis_config'):
            print(f"  → Mode is 'normal', will store raw image")
            self.current_raw_image = image
            print(f"✅ Stored raw image (shape={image.shape})")

            # Enable colony detection if we're in biofilm mode
            if self.analysis_config.is_biofilm_mode():
                print("🔵 Biofilm mode active")
                if hasattr(self, 'detect_colonies_btn'):
                    self.detect_colonies_btn.setEnabled(True)
                    print("✅ Enabled detect_colonies_btn")

        # Store segmented image for other analysis
        if mode == "segmented" and hasattr(self, 'analysis_config'):
            self.segmented_storage[(position, time, channel)] = image
            self.current_segmented_image = image
            print(f"✅ Stored segmented image (shape={image.shape})")
            print(f"✅ Stored segmented image (T={time}, P={position}, shape={image.shape})")

        # KEEP ALL EXISTING CODE BELOW
        if not self.is_segmenting:
            return

        # Check if this is a response to one of our requests
        if (channel != self.channel or
                position not in self.positions or
                not (self.time_range[0] <= time <= self.time_range[1])):
            return
        if mode in ["segmented", "labeled"]:
            self.segmented_storage[(position, time, channel)] = image

        # Create a unique key for this frame
        frame_key = (time, position, channel)

        # Skip if we've already processed this frame
        if frame_key in self.processed_frames:
            return

        # Mark as processed
        self.processed_frames.add(frame_key)

        # Remove from queue if it's the current item
        if self.queue and (time, position) == self.queue[0]:
            self.queue.pop(0)

        # Update progress
        self.progress_bar.setValue(len(self.processed_frames))

        # Schedule next processing with a small delay
        self.request_timer.start(50)  # 50ms delay

    def export_segmented(self, output_folder):

        import tifffile
        from pathlib import Path
        # Get positions and times to export
        for (pos, t, c), img in self.segmented_storage.items():
            path = Path(output_folder) / f"pos{pos}_t{t}_{c}.tif"
            # Write TIF images to the output directory
            tifffile.imwrite(path, img)
        print(f"✅ Exported {len(self.segmented_storage)} segmented images")

    def cancel_segmentation(self):
        """Cancel the segmentation process"""
        self.is_segmenting = False
        self.queue.clear()
        self.segment_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_label.setText("Cancelled")
        pub.sendMessage("segmentation_cancelled")

    def _segmentation_finished(self):
        """Handle completion of all segmentation tasks"""
        self.is_segmenting = False
        self.segment_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_label.setText("Completed")

        # Notify that batch segmentation is complete
        pub.sendMessage("batch_segmentation_completed",
                        time_range=self.time_range,
                        positions=self.positions,
                        model=self.current_model)

    def add_biofilm_configuration_panel(self, layout):
        """Add biofilm-specific configuration panel"""
        biofilm_group = QGroupBox("Biofilm Analysis Configuration")
        biofilm_layout = QVBoxLayout()

        # Colony separation section - REMOVED: Now in dedicated Colony Separation tab
        # self.add_colony_separation_section(biofilm_layout)

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        biofilm_layout.addWidget(separator)

        # Colony grouping parameters
        grouping_layout = QHBoxLayout()

        grouping_layout.addWidget(QLabel("Connection Distance:"))
        self.connection_distance_spin = QSpinBox()
        self.connection_distance_spin.setMinimum(1)
        self.connection_distance_spin.setMaximum(100)
        self.connection_distance_spin.setValue(
            self.biofilm_config.get_setting("colony_grouping", "connection_distance", 10))
        self.connection_distance_spin.setSuffix(" pixels")
        grouping_layout.addWidget(self.connection_distance_spin)

        grouping_layout.addWidget(QLabel("Min Colony Size:"))
        self.min_colony_size_spin = QSpinBox()
        self.min_colony_size_spin.setMinimum(1)
        self.min_colony_size_spin.setMaximum(1000)
        self.min_colony_size_spin.setValue(self.biofilm_config.get_setting("colony_grouping", "min_colony_size", 5))
        self.min_colony_size_spin.setSuffix(" cells")
        grouping_layout.addWidget(self.min_colony_size_spin)

        biofilm_layout.addLayout(grouping_layout)

        # Tracking parameters
        tracking_layout = QHBoxLayout()

        tracking_layout.addWidget(QLabel("Max Displacement:"))
        self.max_displacement_spin = QSpinBox()
        self.max_displacement_spin.setMinimum(1)
        self.max_displacement_spin.setMaximum(200)
        self.max_displacement_spin.setValue(self.biofilm_config.get_setting("colony_tracking", "max_displacement", 50))
        self.max_displacement_spin.setSuffix(" pixels")
        tracking_layout.addWidget(self.max_displacement_spin)

        tracking_layout.addWidget(QLabel("Overlap Threshold:"))
        self.overlap_threshold_spin = QSpinBox()
        self.overlap_threshold_spin.setMinimum(10)
        self.overlap_threshold_spin.setMaximum(90)
        self.overlap_threshold_spin.setValue(
            int(self.biofilm_config.get_setting("colony_tracking", "overlap_threshold", 0.3) * 100))
        self.overlap_threshold_spin.setSuffix("%")
        tracking_layout.addWidget(self.overlap_threshold_spin)

        biofilm_layout.addLayout(tracking_layout)

        # Configuration presets
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Presets:"))

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom",
            "Fast Growing Biofilms",
            "Mature Biofilms",
            "Microcolonies",
            "Sparse Colonies"
        ])
        self.preset_combo.currentTextChanged.connect(self.apply_configuration_preset)
        preset_layout.addWidget(self.preset_combo)

        biofilm_layout.addLayout(preset_layout)

        biofilm_group.setLayout(biofilm_layout)
        layout.addWidget(biofilm_group)

    def add_colony_separation_section(self, layout):
        """Add BiofilmQ-style colony separation interface"""

        # Colony Separation Header
        colony_sep_label = QLabel("Colony Separation")
        colony_sep_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #2196F3;")
        layout.addWidget(colony_sep_label)

        # Detection parameters
        detection_layout = QHBoxLayout()

        # Intensity threshold slider
        detection_layout.addWidget(QLabel("Intensity Threshold:"))
        self.intensity_threshold_slider = QSlider(Qt.Horizontal)
        self.intensity_threshold_slider.setMinimum(1)
        self.intensity_threshold_slider.setMaximum(99)
        self.intensity_threshold_slider.setValue(20)
        self.intensity_threshold_slider.valueChanged.connect(self.on_intensity_threshold_changed)
        detection_layout.addWidget(self.intensity_threshold_slider)

        self.intensity_threshold_label = QLabel("0.20")
        self.intensity_threshold_label.setMinimumWidth(40)
        detection_layout.addWidget(self.intensity_threshold_label)

        layout.addLayout(detection_layout)

        # Size filtering
        size_layout = QHBoxLayout()

        size_layout.addWidget(QLabel("Min Colony Size:"))
        self.min_size_slider = QSlider(Qt.Horizontal)
        self.min_size_slider.setMinimum(500)
        self.min_size_slider.setMaximum(10000)
        self.min_size_slider.setValue(1000)
        self.min_size_slider.valueChanged.connect(self.on_min_size_changed)
        size_layout.addWidget(self.min_size_slider)

        self.min_size_label = QLabel("100")
        self.min_size_label.setMinimumWidth(40)
        size_layout.addWidget(self.min_size_label)

        size_layout.addWidget(QLabel("Max Size:"))
        self.max_size_slider = QSlider(Qt.Horizontal)
        self.max_size_slider.setMinimum(10000)
        self.max_size_slider.setMaximum(500000)
        self.max_size_slider.setValue(100000)
        self.max_size_slider.valueChanged.connect(self.on_max_size_changed)
        size_layout.addWidget(self.max_size_slider)

        self.max_size_label = QLabel("10000")
        self.max_size_label.setMinimumWidth(50)
        size_layout.addWidget(self.max_size_label)

        layout.addLayout(size_layout)

        # Control buttons
        control_layout = QHBoxLayout()

        self.detect_colonies_btn = QPushButton("Detect Colonies")
        self.detect_colonies_btn.clicked.connect(self.detect_colonies)
        self.detect_colonies_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        control_layout.addWidget(self.detect_colonies_btn)

        self.show_overlay_btn = QPushButton("Show/Hide Overlay")
        self.show_overlay_btn.clicked.connect(self.toggle_colony_overlay)
        self.show_overlay_btn.setEnabled(False)
        control_layout.addWidget(self.show_overlay_btn)

        self.clear_colonies_btn = QPushButton("Clear All")
        self.clear_colonies_btn.clicked.connect(self.clear_all_colonies)
        control_layout.addWidget(self.clear_colonies_btn)

        layout.addLayout(control_layout)

        # Colony count display
        self.colony_count_label = QLabel("Colonies detected: 0")
        self.colony_count_label.setStyleSheet("font-weight: bold; color: #666;")
        layout.addWidget(self.colony_count_label)

    def apply_configuration_preset(self, preset_name):
        """Apply predefined configuration presets"""
        if preset_name == "Custom":
            return

        presets = {
            "Fast Growing Biofilms": {
                "connection_distance": 15,
                "min_colony_size": 8,
                "max_displacement": 60,
                "overlap_threshold": 0.4
            },
            "Mature Biofilms": {
                "connection_distance": 8,
                "min_colony_size": 15,
                "max_displacement": 30,
                "overlap_threshold": 0.5
            },
            "Microcolonies": {
                "connection_distance": 5,
                "min_colony_size": 3,
                "max_displacement": 25,
                "overlap_threshold": 0.2
            },
            "Sparse Colonies": {
                "connection_distance": 20,
                "min_colony_size": 5,
                "max_displacement": 80,
                "overlap_threshold": 0.3
            }
        }

        if preset_name in presets:
            config = presets[preset_name]
            self.connection_distance_spin.setValue(config["connection_distance"])
            self.min_colony_size_spin.setValue(config["min_colony_size"])
            self.max_displacement_spin.setValue(config["max_displacement"])
            self.overlap_threshold_spin.setValue(int(config["overlap_threshold"] * 100))

    def analyze_biofilm_colonies(self):
        """Analyze colonies from existing segmentation data"""
        from nd2_analyzer.analysis.metrics_service import MetricsService

        # Get selected positions
        selected_positions = self.get_selected_positions()
        if not selected_positions:
            self.progress_label.setText("No positions selected. Please select positions first.")
            return

        # Get current configuration
        connection_distance = self.connection_distance_spin.value()
        min_colony_size = self.min_colony_size_spin.value()
        max_displacement = self.max_displacement_spin.value()
        overlap_threshold = self.overlap_threshold_spin.value() / 100.0

        # Update biofilm config
        self.biofilm_config.set_setting("colony_grouping", "connection_distance", connection_distance)
        self.biofilm_config.set_setting("colony_grouping", "min_colony_size", min_colony_size)
        self.biofilm_config.set_setting("colony_tracking", "max_displacement", max_displacement)
        self.biofilm_config.set_setting("colony_tracking", "overlap_threshold", overlap_threshold)

        # Initialize colony analysis tools
        self.colony_grouper = ColonyGrouper(
            connection_distance=connection_distance,
            min_colony_size=min_colony_size
        )
        self.colony_tracker = ColonyTracker(
            max_displacement=max_displacement,
            overlap_threshold=overlap_threshold
        )

        # Get cell data from MetricsService
        metrics_service = MetricsService()

        # Process each time point and position
        time_start = self.time_start_spin.value()
        time_end = self.time_end_spin.value()

        # Get channel index safely
        channel = 0
        if hasattr(self, 'channel_combo') and self.channel_combo.count() > 0:
            channel = self.channel_combo.currentIndex()

        self.progress_label.setText("Analyzing biofilm colonies...")
        self.progress_bar.setValue(0)

        total_frames = len(selected_positions) * (time_end - time_start + 1)
        processed_frames = 0

        colony_data_by_time = {}

        print(
            f"Starting colony analysis for positions {selected_positions}, times {time_start}-{time_end}, channel {channel}")

        for position in selected_positions:
            for time in range(time_start, time_end + 1):
                # Get cell data for this frame
                cell_data = metrics_service.get_cell_data(position=position, time=time, channel=channel)

                if not cell_data.is_empty():
                    print(f"Processing T:{time} P:{position} - found {len(cell_data)} cells")

                    # Group cells into colonies
                    colonies = self.colony_grouper.group_cells_into_colonies(cell_data)
                    print(f"Found {len(colonies)} colonies")

                    # Store colony data in MetricsService
                    for colony in colonies:
                        metrics_service.add_colony_metrics(colony)

                    # Collect for tracking
                    if time not in colony_data_by_time:
                        colony_data_by_time[time] = []
                    colony_data_by_time[time].extend(colonies)
                else:
                    print(f"No cell data found for T:{time} P:{position} C:{channel}")

                processed_frames += 1
                progress = int((processed_frames / total_frames) * 100)
                self.progress_bar.setValue(progress)

        # Track colonies over time
        if colony_data_by_time:
            print(f"Tracking colonies across {len(colony_data_by_time)} time points")
            colony_tracks = self.colony_tracker.track_colonies_over_time(colony_data_by_time)

            # Update metrics dataframe
            metrics_service._update_dataframe()

            self.progress_label.setText(f"Analysis complete! Found {len(colony_tracks)} colony tracks.")
            print(f"Colony analysis complete: {len(colony_tracks)} tracks found")

            # Print some details about the tracks
            for track in colony_tracks[:5]:  # Show first 5 tracks
                track_data = track["dynamics"]
                print(
                    f"Track {track['track_id']}: {track_data['track_length']} frames, displacement: {track_data['total_displacement']:.1f}px, growth: {track_data['growth_rate']:.2f}")
        else:
            self.progress_label.setText("No colonies found. Check segmentation and parameters.")
            print("No colony data found")

        self.progress_bar.setValue(100)

    def get_selected_positions(self):
        """Get list of selected positions from the position list widget"""
        selected_positions = []
        for i in range(self.position_list.count()):
            item = self.position_list.item(i)
            if item.isSelected():
                # Extract position number from item text (e.g., "Position 0" -> 0)
                position_text = item.text()
                if "Position" in position_text:
                    try:
                        position_num = int(position_text.split()[-1])
                        selected_positions.append(position_num)
                    except (ValueError, IndexError):
                        print(f"Warning: Could not parse position from '{position_text}'")

        return selected_positions

    def on_intensity_threshold_changed(self, value):
        """Handle intensity threshold slider change"""
        threshold = value / 100.0
        self.intensity_threshold_label.setText(f"{threshold:.2f}")
        self.colony_separator.update_parameters(intensity_threshold=threshold)

    def on_min_size_changed(self, value):
        """Handle minimum size slider change"""
        self.min_size_label.setText(str(value))
        self.colony_separator.update_parameters(min_colony_size=value)

    def on_max_size_changed(self, value):
        """Handle maximum size slider change"""
        self.max_size_label.setText(str(value))
        self.colony_separator.update_parameters(max_colony_size=value)

    def detect_colonies(self):
        """Detect colonies from current raw image (BiofilmQ approach)"""
        print("🔵 DEBUG: detect_colonies() button clicked!")

        # Get current raw image instead of segmented image
        if not hasattr(self, 'current_raw_image'):
            print("❌ ERROR: current_raw_image attribute doesn't exist!")
            self.progress_label.setText("ERROR: current_raw_image not initialized. Please restart the application.")
            return

        if self.current_raw_image is None:
            print("⚠️ WARNING: No image loaded yet. Please load/view an image first.")
            self.progress_label.setText(
                "⚠️ No image loaded. Please view an image in the viewport first, then try again.")
            return

        print(f"✅ Image loaded: shape={self.current_raw_image.shape}")

        # Detect colonies using raw image
        colonies = self.colony_separator.detect_colonies_from_raw_image(self.current_raw_image)

        # Update UI
        self.colony_count_label.setText(f"Colonies detected: {len(colonies)}")
        self.show_overlay_btn.setEnabled(len(colonies) > 0)

        if len(colonies) > 0:
            self.progress_label.setText(f"Detected {len(colonies)} major biofilm colonies.")

            # Automatically show overlay
            self.colony_overlay_visible = True
            self.update_colony_overlay()

            print(f"DEBUG: Detected {len(colonies)} colonies, enabling export button")
            self.export_colonies_btn.setEnabled(True)

            # Print colony information
            print(f"\nDetected {len(colonies)} biofilm colonies:")
            for colony in colonies:
                print(f"Colony {colony['colony_id']}: Area={colony['area']:.0f} pixels, "
                      f"Center=({colony['centroid'][0]:.1f}, {colony['centroid'][1]:.1f})")
        else:
            self.progress_label.setText("No major biofilm colonies detected. Adjust intensity threshold.")

    def toggle_colony_overlay(self):
        """Toggle colony overlay visibility"""
        self.colony_overlay_visible = not self.colony_overlay_visible
        self.update_colony_overlay()

    def update_colony_overlay(self):
        """Update colony overlay in the view"""
        colonies = self.colony_separator.get_all_colonies()
        print(f"DEBUG: Updating overlay with {len(colonies)} colonies")

        if self.current_raw_image is not None and self.colony_overlay_visible and len(colonies) > 0:
            # Create overlay
            overlay = self.colony_separator.create_colony_overlay(self.current_raw_image.shape)
            print(f"DEBUG: Created overlay with shape {overlay.shape}")

            # Send overlay to ViewArea
            pub.sendMessage("show_colony_overlay", overlay=overlay)
            print("DEBUG: Sent overlay to ViewArea")
        else:
            print(
                f"DEBUG: Not showing overlay - raw_image: {self.current_raw_image is not None}, overlay_visible: {self.colony_overlay_visible}, colonies: {len(colonies)}")
            # Hide overlay
            pub.sendMessage("hide_colony_overlay")

    def clear_all_colonies(self):
        """Clear all detected colonies"""
        self.colony_separator.detected_colonies = []
        self.colony_separator.manual_additions = []
        self.colony_count_label.setText("Colonies detected: 0")
        self.show_overlay_btn.setEnabled(False)
        self.colony_overlay_visible = False
        self.update_colony_overlay()
        self.progress_label.setText("All colonies cleared.")

    def add_colony_separation_section(self, layout):
        """Add BiofilmQ-style colony separation interface with manual tools"""

        # Colony Separation Header
        colony_sep_label = QLabel("Colony Separation")
        colony_sep_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #2196F3;")
        layout.addWidget(colony_sep_label)

        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Selection Mode:"))

        self.auto_mode_radio = QRadioButton("Automatic")
        self.manual_mode_radio = QRadioButton("Manual")
        self.manual_mode_radio.setChecked(True)  # Default to manual

        mode_layout.addWidget(self.auto_mode_radio)
        mode_layout.addWidget(self.manual_mode_radio)
        layout.addLayout(mode_layout)

        # Automatic detection parameters (only shown in auto mode)
        self.auto_params_widget = QWidget()
        auto_params_layout = QVBoxLayout(self.auto_params_widget)

        # Detection parameters
        detection_layout = QHBoxLayout()
        detection_layout.addWidget(QLabel("Intensity Threshold:"))
        self.intensity_threshold_slider = QSlider(Qt.Horizontal)
        self.intensity_threshold_slider.setMinimum(1)
        self.intensity_threshold_slider.setMaximum(99)
        self.intensity_threshold_slider.setValue(20)
        self.intensity_threshold_slider.valueChanged.connect(self.on_intensity_threshold_changed)
        detection_layout.addWidget(self.intensity_threshold_slider)

        self.intensity_threshold_label = QLabel("0.20")
        self.intensity_threshold_label.setMinimumWidth(40)
        detection_layout.addWidget(self.intensity_threshold_label)
        auto_params_layout.addLayout(detection_layout)

        # Size filtering
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Min Colony Size:"))
        self.min_size_slider = QSlider(Qt.Horizontal)
        self.min_size_slider.setMinimum(500)
        self.min_size_slider.setMaximum(10000)
        self.min_size_slider.setValue(1000)
        self.min_size_slider.valueChanged.connect(self.on_min_size_changed)
        size_layout.addWidget(self.min_size_slider)

        self.min_size_label = QLabel("1000")
        self.min_size_label.setMinimumWidth(40)
        size_layout.addWidget(self.min_size_label)

        size_layout.addWidget(QLabel("Max Size:"))
        self.max_size_slider = QSlider(Qt.Horizontal)
        self.max_size_slider.setMinimum(10000)
        self.max_size_slider.setMaximum(500000)
        self.max_size_slider.setValue(100000)
        self.max_size_slider.valueChanged.connect(self.on_max_size_changed)
        size_layout.addWidget(self.max_size_slider)

        self.max_size_label = QLabel("100000")
        self.max_size_label.setMinimumWidth(50)
        size_layout.addWidget(self.max_size_label)
        auto_params_layout.addLayout(size_layout)

        layout.addWidget(self.auto_params_widget)

        # Manual selection instructions
        self.manual_instructions = QLabel("Manual Mode: Click on image to draw polygons around biofilm colonies")
        self.manual_instructions.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.manual_instructions)

        # Control buttons
        control_layout = QHBoxLayout()

        self.detect_colonies_btn = QPushButton("Auto Detect")
        self.detect_colonies_btn.clicked.connect(self.detect_colonies)
        self.detect_colonies_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        control_layout.addWidget(self.detect_colonies_btn)

        self.start_manual_btn = QPushButton("Start Manual Selection")
        self.start_manual_btn.clicked.connect(self.start_manual_selection)
        self.start_manual_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        control_layout.addWidget(self.start_manual_btn)

        self.finish_polygon_btn = QPushButton("Finish Polygon")
        self.finish_polygon_btn.clicked.connect(self.finish_current_polygon)
        self.finish_polygon_btn.setEnabled(False)
        control_layout.addWidget(self.finish_polygon_btn)

        self.cancel_polygon_btn = QPushButton("Cancel Polygon")
        self.cancel_polygon_btn.clicked.connect(self.cancel_current_polygon)
        self.cancel_polygon_btn.setEnabled(False)
        control_layout.addWidget(self.cancel_polygon_btn)

        layout.addLayout(control_layout)

        # Second row of buttons
        control_layout2 = QHBoxLayout()

        self.show_overlay_btn = QPushButton("Show/Hide Overlay")
        self.show_overlay_btn.clicked.connect(self.toggle_colony_overlay)
        self.show_overlay_btn.setEnabled(False)
        control_layout2.addWidget(self.show_overlay_btn)

        self.clear_colonies_btn = QPushButton("Clear All")
        self.clear_colonies_btn.clicked.connect(self.clear_all_colonies)
        control_layout2.addWidget(self.clear_colonies_btn)

        self.export_colonies_btn = QPushButton("Export Colony Series")
        self.export_colonies_btn.clicked.connect(self.export_colony_time_series)
        self.export_colonies_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.export_colonies_btn.setEnabled(False)  # Enable when colonies are detected
        control_layout2.addWidget(self.export_colonies_btn)

        layout.addLayout(control_layout2)

        # Colony count display
        self.colony_count_label = QLabel("Colonies detected: 0")
        self.colony_count_label.setStyleSheet("font-weight: bold; color: #666;")
        layout.addWidget(self.colony_count_label)

        # Connect mode change signals
        self.auto_mode_radio.toggled.connect(self.on_mode_changed)
        self.manual_mode_radio.toggled.connect(self.on_mode_changed)

        # Set initial mode
        self.on_mode_changed()

    def export_colony_time_series(self):
        print("DEBUG: Export button clicked!")  # Add this line first

        if not hasattr(self, 'colony_separator') or not self.colony_separator.get_all_colonies():
            print("DEBUG: No colonies found")  # Add this too
            self.progress_label.setText("No colonies to export. Detect colonies first.")
            return

        # Get export folder
        export_folder = QFileDialog.getExistingDirectory(
            self, "Select Export Folder", "", QFileDialog.ShowDirsOnly)

        if not export_folder:
            return

        # Import the exporter
        from ..biofilms.colony_time_series_exporter import ColonyTimeSeriesExporter

        # Get current parameters
        time_start = self.time_start_spin.value()
        time_end = self.time_end_spin.value()
        position = self.get_selected_positions()[0] if self.get_selected_positions() else 0
        channel = self.channel_combo.currentIndex()

        # Get image data from main app
        image_data = None

        def receive_image_data(data):
            nonlocal image_data
            image_data = data

        pub.sendMessage("get_image_data", callback=receive_image_data)

        if not image_data:
            self.progress_label.setText("No image data available for export.")
            return

        # Create exporter and start export
        exporter = ColonyTimeSeriesExporter(self.colony_separator, image_data)

        # Progress callback
        def update_progress(percent):
            self.progress_bar.setValue(percent)

        self.progress_label.setText("Exporting colony time series...")
        self.export_colonies_btn.setEnabled(False)

        try:
            # Run export
            result = exporter.export_all_colonies(
                export_folder,
                (time_start, time_end),
                position,
                channel,
                export_format="cropped",  # Clean cropped colonies for cube analysis
                progress_callback=update_progress
            )

            if "error" in result:
                self.progress_label.setText(f"Export failed: {result['error']}")
            else:
                exported_count = len(result.get("colonies_exported", []))
                self.progress_label.setText(f"Successfully exported {exported_count} colonies to {export_folder}")

        except Exception as e:
            self.progress_label.setText(f"Export error: {str(e)}")
        finally:
            self.export_colonies_btn.setEnabled(True)
            self.progress_bar.setValue(100)

    def on_mode_changed(self):
        """Handle mode change between automatic and manual"""
        auto_mode = self.auto_mode_radio.isChecked()

        # Show/hide automatic parameters
        self.auto_params_widget.setVisible(auto_mode)

        # Update button visibility
        self.detect_colonies_btn.setVisible(auto_mode)
        self.start_manual_btn.setVisible(not auto_mode)
        self.finish_polygon_btn.setVisible(not auto_mode)
        self.cancel_polygon_btn.setVisible(not auto_mode)

        # Update instructions
        if auto_mode:
            self.manual_instructions.setText("Automatic Mode: Adjust parameters and click 'Auto Detect'")
        else:
            self.manual_instructions.setText(
                "Manual Mode: Click 'Start Manual Selection', then click on image to draw polygons")

    def start_manual_selection(self):
        """Start manual colony selection using Colony ROI Selector"""
        print("🟢 DEBUG: start_manual_selection() button clicked!")

        # Get current raw image
        if not hasattr(self, 'current_raw_image'):
            print("❌ ERROR: current_raw_image attribute doesn't exist!")
            self.progress_label.setText("ERROR: current_raw_image not initialized. Please restart the application.")
            return

        if self.current_raw_image is None:
            print("⚠️ WARNING: No image loaded yet. Please load/view an image first.")
            self.progress_label.setText(
                "⚠️ No image loaded. Please view an image in the viewport first, then try again.")
            return

        print(f"✅ Image loaded: shape={self.current_raw_image.shape}")

        # DEBUG: Check what colonies exist
        all_colonies = self.colony_separator.get_all_colonies()
        print(f"DEBUG: Found {len(all_colonies)} existing colonies before opening dialog")
        for i, colony in enumerate(all_colonies):
            print(f"DEBUG: Colony {i + 1}: ID={colony.get('colony_id')}, source={colony.get('source')}")
            if 'polygon_points' in colony:
                print(f"DEBUG: Colony {i + 1} has polygon_points with {len(colony['polygon_points'])} points")
            else:
                print(f"DEBUG: Colony {i + 1} missing polygon_points!")

        # Get existing colonies to pass to the dialog
        existing_colonies = []
        for colony in all_colonies:
            if 'polygon_points' in colony:
                existing_colonies.append({
                    'colony_id': colony['colony_id'],
                    'polygon': colony['polygon_points'],
                    'mask': None
                })
            else:
                print(f"DEBUG: Skipping colony {colony.get('colony_id')} - no polygon_points")

        print(f"DEBUG: Passing {len(existing_colonies)} colonies to dialog")

        # Import and open dialog
        from nd2_analyzer.ui.dialogs.colony_roi_selector import ColonyROISelector
        roi_dialog = ColonyROISelector(self.current_raw_image, existing_colonies=existing_colonies, parent=self)
        roi_dialog.colonies_selected.connect(self.handle_selected_colonies)

        # Update UI state
        self.start_manual_btn.setEnabled(False)
        self.progress_label.setText("Colony ROI Selector opened. Select colonies and click Accept.")

        # Show dialog
        result = roi_dialog.exec()

        # Reset UI state
        self.start_manual_btn.setEnabled(True)

        if result == QDialog.Accepted:
            self.progress_label.setText(f"Selected {len(self.colony_separator.get_all_colonies())} colonies manually.")
        else:
            self.progress_label.setText("Colony selection cancelled.")

    def handle_selected_colonies(self, colonies_data):
        """Handle colonies selected from ROI dialog"""
        # Clear existing manual colonies
        self.colony_separator.manual_additions = []

        # Add each selected colony
        for colony_data in colonies_data:
            # Convert to colony separator format
            colony = self.colony_separator.add_manual_colony(
                colony_data['polygon'],
                self.current_raw_image.shape
            )

        # Update UI
        self.update_colony_count()

        # MAKE SURE TO ENABLE OVERLAY AND EXPORT
        self.colony_overlay_visible = True
        self.show_overlay_btn.setEnabled(True)
        self.export_colonies_btn.setEnabled(True)  # ADD THIS LINE IF MISSING

        self.update_colony_overlay()

        print(f"Added {len(colonies_data)} colonies from ROI selection")

    def finish_current_polygon(self):
        """Finish the current polygon"""
        if hasattr(self, 'current_raw_image') and self.current_raw_image is not None:
            colony = self.colony_separator.finish_current_polygon(self.current_raw_image.shape)
            if colony:
                self.update_colony_count()
                self.update_colony_overlay()
                self.progress_label.setText(
                    f"Colony {colony['colony_id']} created. Start new polygon or finish selection.")

    def cancel_current_polygon(self):
        """Cancel the current polygon"""
        self.colony_separator.cancel_current_polygon()
        self.progress_label.setText("Polygon cancelled. Start new polygon or finish selection.")

    def update_colony_count(self):
        """Update the colony count display"""
        colonies = self.colony_separator.get_all_colonies()
        self.colony_count_label.setText(f"Colonies detected: {len(colonies)}")
        self.show_overlay_btn.setEnabled(len(colonies) > 0)

    def convert_to_tif(self):
        """Convert selected ND2 data to TIF files"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder_path:
            return

        if not hasattr(self, "segmented_storage") or not self.segmented_storage:
            self.progress_label.setText("No segmented images to export")
            return

        self.export_segmented(folder_path)

        self.progress_label.setText(
            f"Exported {len(self.segmented_storage)} segmented images"
        )

    def export_raw_gif(self):
        """Export raw frames (with crop, registration, and ROI applied) as a
        single animated GIF. Uses the Channel and Time Range from this tab,
        and the first selected position (or current view position if none).
        Works for both fresh ND2 loads and reloaded saved projects."""
        import imageio
        from pathlib import Path
        from PySide6.QtWidgets import QApplication, QMessageBox
        from nd2_analyzer.data.image_data import ImageData
        from nd2_analyzer.data.appstate import ApplicationState
        from nd2_analyzer.analysis.roi_helper import ROIHelper

        image_data = ImageData.get_instance()
        if image_data is None or getattr(image_data, "data", None) is None:
            QMessageBox.warning(self, "No Data", "No image data is loaded.")
            return

        # Pick a single position: first selected, otherwise position 0
        selected_items = self.position_list.selectedItems()
        if selected_items:
            position = int(selected_items[0].text().split()[-1])
        else:
            position = 0

        channel = self.channel_combo.currentIndex() if self.channel_combo.count() > 0 else 0

        total_frames_in_data = image_data.data.shape[0]
        # Read live values from the From/To spinboxes so segmentation doesn't
        # need to have been run first.
        t_start = self.time_start_spin.value()
        t_end = self.time_end_spin.value()
        if t_end < t_start:
            QMessageBox.warning(self, "Invalid Time Range",
                                "End frame must be ≥ start frame.")
            return
        t_start = max(0, t_start)
        t_end = min(total_frames_in_data - 1, t_end)

        # Pick output file (single GIF, user names it)
        appstate = ApplicationState.get_instance()
        exp_name = appstate.experiment.name if (appstate and appstate.experiment) else "export"
        default_name = f"{exp_name}_pos{position}_c{channel}.gif"

        gif_path_str, _ = QFileDialog.getSaveFileName(
            self, "Save GIF As", default_name, "GIF Files (*.gif)"
        )
        if not gif_path_str:
            return
        gif_path = Path(gif_path_str)

        roi_mask = ROIHelper.get_roi_mask()

        def apply_roi(img):
            if roi_mask is not None and roi_mask.shape == img.shape:
                return img * roi_mask.astype(img.dtype)
            return img

        def to_uint8(img, lo, hi):
            if hi <= lo:
                return np.zeros(img.shape, dtype=np.uint8)
            out = np.clip((img.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255)
            return out.astype(np.uint8)

        # Sample 5 evenly-spaced frames to compute a stable contrast window
        T = t_end - t_start + 1
        sample_ts = sorted({t_start, t_start + T // 4, t_start + T // 2,
                            t_start + (3 * T) // 4, t_end})
        sample_pixels = [apply_roi(image_data.get(st, position, channel)).ravel()
                         for st in sample_ts]
        sample = np.concatenate(sample_pixels)
        lo, hi = np.percentile(sample, [1, 99])

        fps = 10
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(T)
        self.progress_bar.setValue(0)

        frames = []
        for i, t in enumerate(range(t_start, t_end + 1), start=1):
            img = apply_roi(image_data.get(t, position, channel))
            frames.append(to_uint8(img, lo, hi))
            self.progress_bar.setValue(i)
            QApplication.processEvents()

        imageio.mimsave(str(gif_path), frames, fps=fps)
        print(f"✅ Exported GIF: {gif_path}")
        self.progress_label.setText(
            f"Exported GIF: pos {position}, channel {channel}, frames {t_start}–{t_end} → {gif_path.name}"
        )

    # Receive segmentation data from outside of Widget
    def receive_external_segmentation(self, image, time, position, channel, model):
        if not hasattr(self, "segmented_storage"):
            self.segmented_storage = {}
        self.segmented_storage[(position, time, channel)] = image
        print(f"📥 Stored external segmentation T={time}, P={position}, C={channel}")