from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QSlider, QSpinBox, QComboBox,
                               QGroupBox, QListWidget, QProgressBar,
                               QCheckBox, QFrame, QTextEdit, QSplitter,
                               QFileDialog, QAbstractItemView, QMessageBox,
                               QDoubleSpinBox, QRadioButton, QButtonGroup, QDialog)
from PySide6.QtCore import Qt, Signal, QThread, QObject
from PySide6.QtGui import QFont
import os
import traceback
from pubsub import pub
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from nd2_analyzer.analysis.biofilm_metric_service import BiofilmMetricService
import polars as pl  # Import Polars
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import disk, binary_closing, binary_dilation, binary_opening
from PIL import Image, ImageDraw, ImageFont
import csv
import gc
import shutil


class EPSAnalysisWidget(QWidget):
    """Widget for EPS analysis based off of fluorescence frames and colony metrics"""

    # Moreau mode uses the best compromise parameters from the Fig. 4b match.
    MOREAU_EPS_CLOSING_RADIUS_PX = 0
    MOREAU_EPS_DILATION_RADIUS_PX = 5
    MOREAU_SMOOTHING_SIGMA = 1.5

    # OTSU mode smoothing before local OTSU thresholding
    OTSU_SMOOTHING_SIGMA = 1.5

    # Shared run-level EPS intensity mapping.
    # The black/white points are estimated once from representative frames and
    # then reused for every selected timepoint and position. This prevents
    # per-frame contrast stretching while preserving substantially more useful
    # EPS contrast than blind 16-bit -> 8-bit division by 257.
    EPS_WINDOW_BLACK_PERCENTILE = 25.0
    EPS_WINDOW_WHITE_PERCENTILE = 99.5
    EPS_WINDOW_WHITE_ACROSS_FRAMES_PERCENTILE = 90.0
    EPS_WINDOW_MAX_SAMPLE_FRAMES = 12
    EPS_BACKGROUND_SIGMA = 200.0

    # Fixed best-candidate cell-threshold settings
    MOREAU_CELL_THRESHOLD = 165
    MOREAU_MORPHOLOGY_KERNEL_DIAMETER_UM = 0.65
    MOREAU_FALLBACK_PIXEL_SIZE_UM = 0.064971092928405

    # Dominant-bound EPS base mask rule
    EPS_DOMINANT_BOUND_INTENSITY_MIN = 90.0
    EPS_DOMINANT_BOUND_INTENSITY_MAX = 255.0
    EPS_DOMINANT_BOUND_BIN_COUNT = 130
    EPS_DOMINANT_BOUND_PEAK_LOG_FRACTION = 0.80
    EPS_DOMINANT_BOUND_USE_CONTIGUOUS_PEAK_BAND = True
    EPS_DOMINANT_BOUND_PADDING = 0.0
    EPS_INVERSE_DOMINANT_BOUND_LIMIT_TO_FOCUS_RANGE = True
    EXPORT_GIF_DELAY_MS = 1000

    # Export options
    EXPORT_VISUALS_ONLY_FOR_ONE_POSITION = True
    EXPORT_VISUAL_POSITION = 0  # zero-based, so Position 0, Position 1, etc.
    EPS_OVERLAY_COLOR_RGB = {
        "red": (255, 0, 0),
        "blue": (0, 153, 255),
        "yellow": (255, 221, 0),
        "green": (0, 255, 0),
    }

    # Raw-intensity mode DISPLAY choice only.
    # Both whole-frame and cell-integrated graphs are always exported.
    # Leave one line active to choose which metric appears in the in-app view.
    RAW_INTENSITY_SECONDARY_METRIC = "whole_frame_integrated"
    # RAW_INTENSITY_SECONDARY_METRIC = "cell_integrated"

    def __init__(self, parent=None):
        super().__init__(parent)

        # State variables
        self.base_folder = ""

        # Initial background filtered reference image
        # self.reference_backgrounds = {}

        # Segmentation
        self.is_segmenting = False
        self.cancel_requested = False
        self.queue = []
        self.biofilm_metric_service = None
        # Temporary compatibility mirror populated when a run finishes
        # BiofilmMetricService is the authoritative result store
        self.eps_results = []

        # Partaker fixed-EPS calibration state. The setup dialog returns one
        # threshold and one shared raw-to-uint8 mapping that must be reused by
        # the complete run so the accepted preview matches exported masks.
        self.partaker_eps_setup = None
        self.partaker_eps_threshold = 40
        self.partaker_eps_smoothing_sigma = 1.5
        self.partaker_drop_frame_zero = False
        self.eps_overlay_color_name = "green"

        # Captures the frame-zero decision used by the current run, regardless
        # of whether it came from the Partaker setup dialog or the main widget.
        self.drop_frame_zero_for_current_run = False

        # Run-level EPS preprocessing state. These values are recalculated when
        # a new segmented EPS analysis starts.
        self.eps_processing_black_point_raw = None
        self.eps_processing_white_point_raw = None
        self.eps_processing_window_source = "uninitialized"
        self.eps_reference_backgrounds = {}

        # Segmentation Event Handler
        from PySide6.QtCore import QTimer
        self.request_timer = QTimer(self)
        self.request_timer.setSingleShot(True)
        self.request_timer.timeout.connect(self.process_next_in_queue)

        self.init_ui()
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")

    def init_ui(self):
        """Initialize the user interface"""
        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)

        # Left side - Configuration
        left_widget = self.create_configuration_panel()
        main_splitter.addWidget(left_widget)

        # Right side - Results
        right_widget = self.create_results_panel()
        main_splitter.addWidget(right_widget)

        # Set splitter proportions (30% left, 70% right)
        main_splitter.setSizes([300, 700])

        # Main layout
        layout = QVBoxLayout(self)
        layout.addWidget(main_splitter)

    def create_configuration_panel(self):
        """Create the left configuration panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Title
        title_label = QLabel("EPS Configuration")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2196F3; margin-bottom: 10px;")
        layout.addWidget(title_label)

        # Input Selection Group
        input_group = self.create_input_selection_group()
        layout.addWidget(input_group)

        # Select Position Group
        position_group = self.select_stage_position()
        layout.addWidget(position_group)

        # Processing Controls Group
        processing_group = self.create_processing_group()
        layout.addWidget(processing_group)

        # Stretch to push everything to top
        layout.addStretch()

        return widget

    def create_input_selection_group(self):
        """Create input selection group"""
        group = QGroupBox("1. Select Analysis Channels")
        layout = QVBoxLayout(group)

        # Select channels for Segmentation
        selection_layout = QVBoxLayout()

        # Analysis Method Controls
        analysis_mode_group = QGroupBox("Analysis Method")
        analysis_mode_layout = QVBoxLayout()

        self.radio_partaker = QRadioButton("Partaker")
        self.radio_moreau = QRadioButton("Moreau")
        self.radio_otsu = QRadioButton("OTSU")
        self.radio_raw_intensities = QRadioButton("Use Raw Intensities")

        # OTSU is the default method.
        self.radio_otsu.setChecked(True)

        self.analysis_method_group = QButtonGroup(self)
        self.analysis_method_group.addButton(self.radio_partaker)
        self.analysis_method_group.addButton(self.radio_moreau)
        self.analysis_method_group.addButton(self.radio_otsu)
        self.analysis_method_group.addButton(self.radio_raw_intensities)

        analysis_mode_layout.addWidget(self.radio_partaker)
        analysis_mode_layout.addWidget(self.radio_moreau)
        analysis_mode_layout.addWidget(self.radio_otsu)
        analysis_mode_layout.addWidget(self.radio_raw_intensities)

        analysis_mode_group.setLayout(analysis_mode_layout)
        selection_layout.addWidget(analysis_mode_group)

        # self.radio_partaker.toggled.connect(self.on_method_changed)
        # self.radio_moreau.toggled.connect(self.on_method_changed)

        selection_layout.addWidget(QLabel("EPS Channel:"))
        self.eps_channel_combo = QComboBox()
        selection_layout.addWidget(self.eps_channel_combo)

        # Cell channel selection
        selection_layout.addWidget(QLabel("Cell Channel:"))
        self.cell_channel_combo = QComboBox()
        selection_layout.addWidget(self.cell_channel_combo)

        # Cell view type selection
        selection_layout.addWidget(QLabel("Select Cell View Type:"))
        self.cell_view_combo = QComboBox()
        self.cell_view_combo.addItems(["Fluorescence", "Phase Contrast"])
        self.cell_view_combo.setCurrentText("Phase Contrast")
        selection_layout.addWidget(self.cell_view_combo)

        # EPS filtering options used by non-Partaker segmentation modes.
        self.gaus_back_corr = QCheckBox("Gaussian Background Subtraction")
        self.gaus_back_corr.setChecked(False)
        self.close_dialate = QCheckBox(
            "Morphologically Close & Dilate EPS Channel"
        )
        self.close_dialate.setChecked(False)

        self.drop_frame_zero_checkbox = QCheckBox(
            "Drop frame 0 from analysis"
        )
        self.drop_frame_zero_checkbox.setChecked(False)
        self.drop_frame_zero_checkbox.setToolTip(
            "Exclude T=0 from processing, exported metrics, plots, and GIFs. "
            "Partaker provides this option inside its threshold setup dialog."
        )

        # These are true EPS filtering controls. Raw-intensity mode disables
        # them, but the frame-zero option remains available.
        self.eps_filter_checkboxes = [
            self.gaus_back_corr,
            self.close_dialate,
        ]

        # Every item here disappears in Partaker mode because Partaker exposes
        # its frame-zero option in EPSSetupDialog and does not use the other
        # two main-widget filtering controls.
        self.partaker_disappearing_checkboxes = [
            self.gaus_back_corr,
            self.close_dialate,
            self.drop_frame_zero_checkbox,
        ]

        for checkbox in self.partaker_disappearing_checkboxes:
            # Keep disabled text readable instead of allowing the platform theme
            # to fade it almost completely into the background.
            checkbox.setStyleSheet(
                "QCheckBox:disabled { color: #777777; }"
            )
            selection_layout.addWidget(checkbox)

        for radio_button in [
            self.radio_partaker,
            self.radio_moreau,
            self.radio_otsu,
            self.radio_raw_intensities,
        ]:
            radio_button.toggled.connect(self.on_analysis_method_changed)

        self.on_analysis_method_changed()

        layout.addLayout(selection_layout)

        return group

    def on_analysis_method_changed(self):
        """Update EPS controls and the Start button for the selected mode."""
        partaker_mode = self.is_partaker_method()
        raw_mode = self.is_raw_intensity_method()

        for checkbox in getattr(
                self,
                "partaker_disappearing_checkboxes",
                [],
        ):
            checkbox.setVisible(not partaker_mode)

        # Raw-intensity mode bypasses EPS filtering, but dropping T=0 is still
        # a valid run-scope choice and therefore stays enabled.
        for checkbox in getattr(self, "eps_filter_checkboxes", []):
            if partaker_mode or raw_mode:
                checkbox.setChecked(False)
            checkbox.setEnabled(not raw_mode)

        if hasattr(self, "drop_frame_zero_checkbox"):
            self.drop_frame_zero_checkbox.setEnabled(True)

        if hasattr(self, "start_segmentation_btn"):
            self.start_segmentation_btn.setText("Start")

    def get_analysis_method(self):
        """Return selected EPS analysis method."""
        if (
                hasattr(self, "radio_raw_intensities")
                and self.radio_raw_intensities.isChecked()
        ):
            return "Raw Intensities"
        if hasattr(self, "radio_moreau") and self.radio_moreau.isChecked():
            return "Moreau"
        if hasattr(self, "radio_otsu") and self.radio_otsu.isChecked():
            return "OTSU"
        return "Partaker"

    def get_selected_cell_channel(self):
        """Return the selected cell channel for Moreau and Partaker cell-mask analysis."""
        if hasattr(self, "cell_channel_combo") and self.cell_channel_combo.count() > 0:
            return int(self.cell_channel_combo.currentIndex())
        return 0

    def is_partaker_method(self):
        """Return True when the current implementation / segmented-cell method is selected."""
        return self.get_analysis_method() == "Partaker"

    def is_moreau_method(self):
        """Return True when the Fig. 4b best-compromise parameter method is selected."""
        return self.get_analysis_method() == "Moreau"

    def is_otsu_method(self):
        """Return True when local OTSU EPS segmentation is selected."""
        return self.get_analysis_method() == "OTSU"

    def is_raw_intensity_method(self):
        """Return True when the full raw EPS frame is analyzed without an EPS mask."""
        return self.get_analysis_method() == "Raw Intensities"

    def select_stage_position(self):
        """Create cube configuration group"""
        group = QGroupBox("2. Select Positions")
        layout = QVBoxLayout(group)

        # Position selection (existing code)
        size_layout = QVBoxLayout()

        # Position list with checkboxes
        self.position_list = QListWidget()
        self.position_list.setSelectionMode(QAbstractItemView.MultiSelection)
        size_layout.addWidget(self.position_list)

        # Quick selection buttons
        pos_buttons_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_positions)
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self.select_no_positions)
        pos_buttons_layout.addWidget(select_all_btn)
        pos_buttons_layout.addWidget(select_none_btn)
        size_layout.addLayout(pos_buttons_layout)

        # Time range selection
        size_layout.addWidget(QLabel("Time Range"))
        time_layout = QHBoxLayout()

        time_layout.addWidget(QLabel("From:"))
        self.time_start_spin = QSpinBox()
        self.time_start_spin.setMinimum(0)
        time_layout.addWidget(self.time_start_spin)

        time_layout.addWidget(QLabel("To:"))
        self.time_end_spin = QSpinBox()
        self.time_end_spin.setMinimum(0)
        time_layout.addWidget(self.time_end_spin)

        size_layout.addLayout(time_layout)

        layout.addLayout(size_layout)

        return group

    def create_processing_group(self):
        """Create processing controls group"""
        group = QGroupBox("3. Processing")
        layout = QVBoxLayout(group)

        # Control buttons
        button_layout = QHBoxLayout()

        # Segment Channel selector and button
        self.start_segmentation_btn = QPushButton("Start")
        self.start_segmentation_btn.clicked.connect(self.segment_selected_channels)
        self.start_segmentation_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 5px;")
        self.start_segmentation_btn.setEnabled(False)
        self.on_analysis_method_changed()

        # Cancel button
        self.cancel_analysis_btn = QPushButton("Cancel")
        self.cancel_analysis_btn.clicked.connect(self.cancel_analysis)
        self.cancel_analysis_btn.setStyleSheet(
            "background-color: #f44336; color: white; font-weight: bold; padding: 5px;")
        self.cancel_analysis_btn.setEnabled(False)

        button_layout.addWidget(self.start_segmentation_btn)
        button_layout.addWidget(self.cancel_analysis_btn)
        layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Select Start to begin Segmentation")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)

        return group

    def create_results_panel(self):
        """Create the right results panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Results title
        results_title = QLabel("Analysis Results")
        results_font = QFont()
        results_font.setBold(True)
        results_font.setPointSize(14)
        results_title.setFont(results_font)
        results_title.setStyleSheet("color: #2196F3; margin-bottom: 10px;")
        layout.addWidget(results_title)

        # Visualization controls
        viz_controls = self.create_visualization_controls()
        layout.addWidget(viz_controls)

        # Matplotlib figure
        self.population_figure = plt.figure(constrained_layout=True)
        self.population_canvas = FigureCanvas(self.population_figure)
        layout.addWidget(self.population_canvas)

        # Export section
        export_section = self.create_export_section()
        layout.addWidget(export_section)

        # Stretch
        layout.addStretch()

        return widget

    def create_visualization_controls(self):
        """Create visualization controls"""
        group = QGroupBox("Visualization")
        layout = QHBoxLayout(group)

        # Frame capture interval selection
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Capture Interval:"))
        self.frame_interval_value = QDoubleSpinBox()
        self.frame_interval_value.setDecimals(3)
        self.frame_interval_value.setRange(0.001, 1e9)
        self.frame_interval_value.setValue(24)  # default = 1 hr
        self.frame_interval_value.setSingleStep(0.1)
        interval_layout.addWidget(self.frame_interval_value)
        # Unit dropdown
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(["ms", "sec", "min", "hr", "day"])
        self.time_unit_combo.setCurrentText("hr")
        interval_layout.addWidget(self.time_unit_combo)
        layout.addLayout(interval_layout)

        # Graph Type selection
        metric_layout = QHBoxLayout()
        metric_layout.addWidget(QLabel("Analysis Type:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItem("EPS Intensity Metrics")
        self.metric_combo.addItem("EPS-Cell Association Metrics")
        # self.metric_combo.addItem("NA")
        metric_layout.addWidget(self.metric_combo)
        layout.addLayout(metric_layout)

        # Regenerate Graph
        self.generate_graph_btn = QPushButton("Generate Graph")
        self.generate_graph_btn.setEnabled(False)
        self.generate_graph_btn.clicked.connect(self.on_plot_avg_sd)
        layout.addWidget(self.generate_graph_btn)

        layout.addStretch()

        return group

    def create_export_section(self):
        """Create export section"""
        group = QGroupBox("Export Results")
        layout = QHBoxLayout(group)

        self.export_csv_btn = QPushButton("Export to CSV")
        self.export_csv_btn.setEnabled(False)
        self.export_plots_btn = QPushButton("Export Plots")
        self.export_plots_btn.setEnabled(False)

        layout.addWidget(self.export_csv_btn)
        layout.addWidget(self.export_plots_btn)
        layout.addStretch()

        return group

    def validate_partaker_segmentation_available(
            self,
            selected_positions=None,
            t_start=None,
            t_end=None,
            focus_loss_skip=None,
    ):
        """
        Check that segmented cell masks are available whenever the selected
        analysis mode uses Partaker cell masks.
        """
        try:
            from nd2_analyzer.data.image_data import ImageData

            mode_name = self.get_analysis_method()

            image_data = ImageData.get_instance()

            if image_data is None or image_data.segmentation_cache is None:
                QMessageBox.warning(
                    self,
                    "Missing cell segmentation",
                    f"{mode_name} with Phase Contrast requires segmented cell masks. "
                    "Run cell segmentation first, choose Fluorescence cell view, "
                    "or switch analysis mode."
                )
                return False

            segmented_storage = image_data.segmentation_cache
            model_name = image_data.segmentation_cache.model_name

            if not model_name:
                QMessageBox.warning(
                    self,
                    "Missing cell segmentation model",
                    f"{mode_name} with Phase Contrast requires a selected cell segmentation model. "
                    "Run cell segmentation first, choose Fluorescence cell view, "
                    "or switch analysis mode."
                )
                return False

            cache = segmented_storage.with_model(model_name)

            if (
                    not hasattr(cache, "mmap_arrays_idx")
                    or model_name not in cache.mmap_arrays_idx
            ):
                QMessageBox.warning(
                    self,
                    "Missing cell segmentation cache",
                    f"{mode_name} with Phase Contrast requires segmented cell masks. "
                    "Run cell segmentation first, choose Fluorescence cell view, "
                    "or switch analysis mode."
                )
                return False

            _mmap_array, index_set = cache.mmap_arrays_idx[model_name]

            if not index_set:
                QMessageBox.warning(
                    self,
                    "Missing segmented cell masks",
                    f"{mode_name} with Phase Contrast requires segmented cell masks. "
                    "Run cell segmentation first, choose Fluorescence cell view, "
                    "or switch analysis mode."
                )
                return False

            selected_cell_channel = self.get_selected_cell_channel()

            available_channels = set()
            available_frame_keys = set()

            for idx in index_set:
                if len(idx) == 3:
                    t, p, c = idx
                    t = int(t)
                    p = int(p)
                    c = int(c)

                    available_channels.add(c)
                    available_frame_keys.add((t, p, c))

            if not available_channels:
                QMessageBox.warning(
                    self,
                    "Cannot verify segmented channel",
                    "The existing segmented cell masks do not appear to store channel information.\n\n"
                    f"{mode_name} needs channel-aware segmentation masks so the selected Cell Channel can be checked.\n"
                    "Please re-run cell segmentation with the current channel-aware cache."
                )
                return False

            if selected_cell_channel not in available_channels:
                QMessageBox.warning(
                    self,
                    "Cell channel not segmented",
                    f"Selected Cell Channel {selected_cell_channel} was not segmented.\n\n"
                    f"Available segmented channels: {sorted(available_channels)}\n\n"
                    "Select the channel that was actually segmented, or run cell segmentation on the selected Cell Channel first."
                )
                return False

            if selected_positions is not None and t_start is not None and t_end is not None:
                focus_loss_skip = focus_loss_skip or set()

                missing = []

                for p in selected_positions:
                    for t in range(t_start, t_end + 1):
                        if t in focus_loss_skip:
                            continue

                        key = (
                            int(t),
                            int(p),
                            int(selected_cell_channel),
                        )

                        if key not in available_frame_keys:
                            missing.append(key)

                if missing:
                    first_t, first_p, first_c = missing[0]

                    QMessageBox.warning(
                        self,
                        "Missing segmented cell masks",
                        f"{mode_name} cannot start because the selected Cell Channel is missing segmented masks.\n\n"
                        f"First missing frame: T={first_t}, P={first_p}, C={first_c}\n"
                        f"Total missing frames: {len(missing)}\n\n"
                        "Run cell segmentation for the selected Cell Channel over the same positions/time range, "
                        "or change the Cell Channel selection."
                    )
                    return False

            return True

        except Exception as e:
            QMessageBox.warning(
                self,
                "Segmentation check failed",
                f"Could not verify segmented cell masks for {mode_name}:\n{e}"
            )
            return False

    def cancel_analysis(self):
        """Cancel EPS segmentation."""
        if getattr(self, "is_segmenting", False):
            self.cancel_requested = True
            pub.sendMessage("segmentation_cancelled")
            self.status_label.setText("Cancelling segmentation...")
            return

    def export_to_csv(self):
        """Export analysis results to CSV file"""
        if not self.analysis_results:
            self.status_label.setText("No results to export.")
            return

        try:
            # Get save location
            from PySide6.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save CSV File", "biofilm_cube_analysis.csv", "CSV Files (*.csv)"
            )

            if not file_path:
                return

            import csv

            # Create CSV with all data
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                header = ['Colony', 'TimePoint', 'Square_X', 'Square_Y', 'Local_Density',
                          'Distance_To_Edge', 'Distance_To_Center', 'Shape_Area',
                          'Intensity_Mean', 'Local_Thickness']
                writer.writerow(header)

                # Write data
                for colony_name, colony_data in self.analysis_results.items():
                    for time_point, time_data in colony_data.items():
                        for i in range(len(time_data['square_positions'])):
                            x, y = time_data['square_positions'][i]
                            row = [
                                colony_name,
                                time_point,
                                x, y,
                                time_data['local_density'][i],
                                time_data['distance_to_edge'][i],
                                time_data['distance_to_center'][i],
                                time_data['shape_area'][i],
                                time_data['intensity_mean'][i],
                                time_data['local_thickness'][i]
                            ]
                            writer.writerow(row)

            self.status_label.setText(f"Data exported to {file_path}")
            self.console_area.append(f"✓ CSV exported: {file_path}")

        except Exception as e:
            self.status_label.setText(f"Export failed: {str(e)}")
            print(f"CSV export error: {e}")

    def export_plots(self):
        """Export plots to files"""
        if not self.analysis_results:
            self.status_label.setText("No results to export.")
            return

        try:
            # Get save directory
            from PySide6.QtWidgets import QFileDialog
            save_dir = QFileDialog.getExistingDirectory(self, "Select Directory for Plots")

            if not save_dir:
                return

            import matplotlib.pyplot as plt
            import numpy as np

            # Parameters to plot
            param_map = {
                'Local Density': 'local_density',
                'Distance to Edge': 'distance_to_edge',
                'Distance to Center': 'distance_to_center',
                'Local Texture': 'local_thickness'
            }

            plots_created = 0

            for colony_name, colony_data in self.analysis_results.items():
                # Get a few representative time points
                time_points = sorted(colony_data.keys(), key=lambda x: int(x))
                sample_times = time_points[::len(time_points) // 4] if len(time_points) > 4 else time_points

                for param_name, data_key in param_map.items():
                    for time_point in sample_times[:3]:  # Max 3 time points per parameter
                        time_data = colony_data[time_point]

                        positions = time_data['square_positions']
                        values = time_data[data_key]

                        # Filter non-zero values
                        filtered_data = [(pos, val) for pos, val in zip(positions, values) if val > 0]

                        if filtered_data:
                            positions, values = zip(*filtered_data)
                            x_coords = [pos[0] for pos in positions]
                            y_coords = [pos[1] for pos in positions]

                            # Create plot
                            plt.figure(figsize=(10, 8))
                            scatter = plt.scatter(x_coords, y_coords, c=values, cmap='viridis', s=20)
                            plt.colorbar(scatter, label=param_name)
                            plt.title(f'{param_name} - {colony_name} - T{time_point.zfill(3)}')
                            plt.xlabel('X Position (pixels)')
                            plt.ylabel('Y Position (pixels)')
                            plt.gca().invert_yaxis()

                            # Save plot
                            filename = f"{colony_name}_{param_name.replace(' ', '_')}_T{time_point.zfill(3)}.png"
                            filepath = os.path.join(save_dir, filename)
                            plt.savefig(filepath, dpi=300, bbox_inches='tight')
                            plt.close()

                            plots_created += 1

            self.status_label.setText(f"Exported {plots_created} plots to {save_dir}")
            self.console_area.append(f"✓ Exported {plots_created} plots")

        except Exception as e:
            self.status_label.setText(f"Plot export failed: {str(e)}")
            print(f"Plot export error: {e}")

    def reset_eps_run_preprocessing(self):
        """Clear cached intensity-window and background-reference state."""
        self.eps_processing_black_point_raw = None
        self.eps_processing_white_point_raw = None
        self.eps_processing_window_source = "uninitialized"
        self.eps_reference_backgrounds = {}

    def build_eps_reference_backgrounds(
            self,
            frame_keys,
            sigma: float | None = None,
    ):
        """
        Build one Gaussian background reference per selected position.

        The background is estimated from time 0 and the same reference is
        subtracted from every frame at that position. This is intentionally
        different from correcting only T=0 or estimating a different broad
        background independently for every timepoint.
        """
        self.eps_reference_backgrounds = {}

        if not self.gaus_back_corr.isChecked():
            return

        from nd2_analyzer.data.image_data import ImageData

        image_data = ImageData.get_instance()
        sigma = float(self.EPS_BACKGROUND_SIGMA if sigma is None else sigma)

        position_channels = sorted({
            (int(position), int(channel))
            for _time, position, channel in frame_keys
        })

        for position, channel in position_channels:
            reference_frame = np.asarray(
                image_data.get(0, position, channel),
                dtype=np.float32,
            )

            reference_background = np.empty_like(
                reference_frame,
                dtype=np.float32,
            )
            gaussian_filter(
                reference_frame,
                sigma=sigma,
                output=reference_background,
            )

            self.eps_reference_backgrounds[position] = reference_background

            print(
                "Cached Gaussian EPS background | "
                f"P={position} T=0 C={channel} sigma={sigma:g}"
            )

    def apply_eps_reference_background(
            self,
            image: np.ndarray,
            position: int,
    ) -> np.ndarray:
        """Subtract the cached T=0 Gaussian background for one position."""
        image = np.asarray(image, dtype=np.float32)

        if not self.gaus_back_corr.isChecked():
            return image

        reference_background = self.eps_reference_backgrounds.get(
            int(position)
        )

        if reference_background is None:
            raise RuntimeError(
                "Gaussian background subtraction was selected, but no "
                f"reference background was cached for Position {position}."
            )

        if reference_background.shape != image.shape:
            raise ValueError(
                "EPS background shape does not match the current frame: "
                f"{reference_background.shape} != {image.shape}."
            )

        corrected = np.array(image, dtype=np.float32, copy=True)
        corrected -= reference_background
        np.maximum(corrected, 0.0, out=corrected)
        return corrected

    def estimate_eps_processing_window(self, frame_keys):
        """
        Estimate one raw-intensity black/white window for the complete run.

        Each representative frame contributes a background-side percentile and
        a bright-signal percentile. The median black candidate and a high
        percentile of the white candidates are reused for every frame, so real
        changes over time are not normalized away.
        """
        if not frame_keys:
            raise ValueError("No EPS frames were supplied for window estimation.")

        from nd2_analyzer.data.image_data import ImageData

        image_data = ImageData.get_instance()
        sample_count = min(
            int(self.EPS_WINDOW_MAX_SAMPLE_FRAMES),
            len(frame_keys),
        )
        sample_indices = np.unique(
            np.linspace(
                0,
                len(frame_keys) - 1,
                sample_count,
                dtype=int,
            )
        )

        black_candidates = []
        white_candidates = []

        for sample_index in sample_indices:
            time, position, channel = frame_keys[int(sample_index)]
            frame = np.asarray(
                image_data.get(time, position, channel),
                dtype=np.float32,
            )
            processing_frame = self.apply_eps_reference_background(
                frame,
                position,
            )

            finite_values = processing_frame[np.isfinite(processing_frame)]
            if finite_values.size == 0:
                continue

            black_candidates.append(float(np.percentile(
                finite_values,
                self.EPS_WINDOW_BLACK_PERCENTILE,
            )))
            white_candidates.append(float(np.percentile(
                finite_values,
                self.EPS_WINDOW_WHITE_PERCENTILE,
            )))

        if not black_candidates or not white_candidates:
            raise ValueError(
                "Could not estimate a finite EPS processing intensity window."
            )

        black_point = float(np.median(black_candidates))
        white_point = float(np.percentile(
            white_candidates,
            self.EPS_WINDOW_WHITE_ACROSS_FRAMES_PERCENTILE,
        ))

        if not np.isfinite(black_point) or not np.isfinite(white_point):
            raise ValueError("Estimated EPS intensity window is not finite.")

        if white_point <= black_point:
            white_point = black_point + 1.0

        self.eps_processing_black_point_raw = black_point
        self.eps_processing_white_point_raw = white_point
        self.eps_processing_window_source = "automatic_run_level"

        print(
            "EPS run-level processing window | "
            f"black_raw={black_point:.3f} | "
            f"white_raw={white_point:.3f} | "
            f"sampled_frames={len(sample_indices)} | "
            f"black_pct={self.EPS_WINDOW_BLACK_PERCENTILE:g} | "
            f"white_pct={self.EPS_WINDOW_WHITE_PERCENTILE:g}"
        )

    def prepare_eps_run_preprocessing(self, frame_keys):
        """Prepare shared background references and intensity scaling."""
        self.reset_eps_run_preprocessing()

        if self.is_raw_intensity_method():
            self.eps_processing_window_source = "not_used_raw_intensity_mode"
            return

        self.build_eps_reference_backgrounds(frame_keys)

        # Partaker must reuse the exact intensity window accepted in the setup
        # dialog. Re-estimating it here could make the full-run masks differ
        # from the previews the user approved.
        if self.is_partaker_method() and self.partaker_eps_setup is not None:
            self.eps_processing_black_point_raw = float(
                self.partaker_eps_setup.processing_black_point_raw
            )
            self.eps_processing_white_point_raw = float(
                self.partaker_eps_setup.processing_white_point_raw
            )
            self.eps_processing_window_source = str(
                self.partaker_eps_setup.processing_window_source
            )
            print(
                "Using Partaker dialog EPS processing window | "
                f"black_raw={self.eps_processing_black_point_raw:.3f} | "
                f"white_raw={self.eps_processing_white_point_raw:.3f}"
            )
            return

        self.estimate_eps_processing_window(frame_keys)

    def open_partaker_eps_setup_dialog(
            self,
            *,
            selected_positions,
            t_start: int,
            t_end: int,
            eps_channel: int,
    ):
        """Open Partaker fixed-threshold calibration and return its result."""
        from nd2_analyzer.data.image_data import ImageData
        from nd2_analyzer.ui.biofilms.eps_setup_dialog import EPSSetupDialog

        image_data = ImageData.get_instance()
        if image_data is None or image_data.data is None:
            QMessageBox.warning(
                self,
                "No image data",
                "Load image data before configuring fixed EPS thresholding."
            )
            return None

        dialog = EPSSetupDialog(
            image_data=image_data,
            eps_channel=eps_channel,
            selected_positions=selected_positions,
            time_start=t_start,
            time_end=t_end,
            initial_drop_frame_zero=self.partaker_drop_frame_zero,
            initial_threshold=self.partaker_eps_threshold,
            smoothing_sigma=self.partaker_eps_smoothing_sigma,
            initial_overlay_color=self.eps_overlay_color_name,
            initial_capture_interval_value=self.frame_interval_value.value(),
            initial_capture_interval_unit=self.time_unit_combo.currentText(),
            parent=self,
        )

        if dialog.exec() != QDialog.Accepted or dialog.result is None:
            return None

        return dialog.result

    def apply_partaker_eps_setup(self, setup) -> None:
        """Copy accepted setup values into the EPS widget and its controls."""
        self.partaker_eps_setup = setup
        self.partaker_eps_threshold = int(setup.threshold_uint8)
        self.partaker_eps_smoothing_sigma = float(setup.smoothing_sigma)
        self.partaker_drop_frame_zero = bool(setup.drop_frame_zero)
        self.drop_frame_zero_for_current_run = bool(setup.drop_frame_zero)
        self.eps_overlay_color_name = self.normalize_eps_overlay_color_name(
            getattr(setup, "overlay_color_name", "green")
        )

        self.time_start_spin.setValue(int(setup.time_start))
        self.time_end_spin.setValue(int(setup.time_end))
        self.eps_channel_combo.setCurrentIndex(int(setup.eps_channel))
        self.frame_interval_value.setValue(
            float(getattr(setup, "capture_interval_value", self.frame_interval_value.value()))
        )
        capture_interval_unit = str(
            getattr(setup, "capture_interval_unit", self.time_unit_combo.currentText())
        )
        capture_interval_index = self.time_unit_combo.findText(
            capture_interval_unit
        )
        if capture_interval_index >= 0:
            self.time_unit_combo.setCurrentIndex(capture_interval_index)

        selected_positions = set(int(value) for value in setup.positions)
        for index in range(self.position_list.count()):
            self.position_list.item(index).setSelected(
                index in selected_positions
            )

        self.status_label.setText(
            "Partaker EPS setup accepted: "
            f"threshold={self.partaker_eps_threshold}, "
            f"overlay={self.eps_overlay_color_name}, "
            f"interval={self.frame_interval_value.value():g} {self.time_unit_combo.currentText()}, "
            f"T={setup.time_start}..{setup.time_end}, "
            f"drop_T0={bool(setup.drop_frame_zero)}, "
            f"positions={list(setup.positions)}"
        )

    @classmethod
    def normalize_eps_overlay_color_name(cls, value) -> str:
        color_name = str(value or "green").strip().lower()
        if color_name not in cls.EPS_OVERLAY_COLOR_RGB:
            return "green"
        return color_name

    @classmethod
    def eps_overlay_rgb_from_name(cls, color_name) -> tuple[int, int, int]:
        return cls.EPS_OVERLAY_COLOR_RGB[
            cls.normalize_eps_overlay_color_name(color_name)
        ]

    def current_eps_overlay_color_name(self) -> str:
        return self.normalize_eps_overlay_color_name(
            getattr(self, "eps_overlay_color_name", "green")
        )

    def eps_processed_value_to_raw(self, value):
        """Convert a scalar 0-255 EPS processing value back to raw units."""
        if not isinstance(value, (int, float, np.integer, np.floating)):
            return np.nan

        value = float(value)
        black_point = self.eps_processing_black_point_raw
        white_point = self.eps_processing_white_point_raw

        if (
                not np.isfinite(value)
                or black_point is None
                or white_point is None
                or white_point <= black_point
        ):
            return np.nan

        return float(
            black_point
            + (value / 255.0) * (white_point - black_point)
        )

    def segment_selected_channels(self):
        """Queue segmentation for selected channels, positions, and time range."""
        self.eps_results = []
        if self.biofilm_metric_service is None:
            self.status_label.setText("No image data available for EPS analysis")
            return
        self.biofilm_metric_service.reset_eps_results()
        self.reset_output_tracking()

        if self.is_segmenting:
            return

        self.start_segmentation_btn.setEnabled(False)
        self.cancel_analysis_btn.setEnabled(True)
        selected_positions = self.get_selected_positions()

        if not selected_positions:
            self.status_label.setText("No positions selected")
            self.start_segmentation_btn.setEnabled(True)
            self.cancel_analysis_btn.setEnabled(False)
            return

        t_start = self.time_start_spin.value()
        t_end = self.time_end_spin.value()

        if t_end < t_start:
            self.status_label.setText("Invalid time range")
            return

        eps_channel = self.eps_channel_combo.currentIndex()

        # Non-Partaker modes use the main-widget checkbox. Partaker replaces
        # this value with the setting accepted in EPSSetupDialog.
        drop_frame_zero = bool(
            self.drop_frame_zero_checkbox.isChecked()
        )

        if self.is_partaker_method():
            setup = self.open_partaker_eps_setup_dialog(
                selected_positions=selected_positions,
                t_start=t_start,
                t_end=t_end,
                eps_channel=eps_channel,
            )
            if setup is None:
                self.status_label.setText("Partaker EPS setup cancelled")
                self.start_segmentation_btn.setEnabled(True)
                self.cancel_analysis_btn.setEnabled(False)
                return

            self.apply_partaker_eps_setup(setup)
            selected_positions = list(setup.positions)
            t_start = int(setup.time_start)
            t_end = int(setup.time_end)
            eps_channel = int(setup.eps_channel)
            drop_frame_zero = bool(setup.drop_frame_zero)

        self.drop_frame_zero_for_current_run = bool(drop_frame_zero)

        # Skip focus-loss frames and any explicitly excluded frame zero.
        focus_loss_skip = set()

        try:
            from nd2_analyzer.data.appstate import ApplicationState
            appstate = ApplicationState.get_instance()

            if (
                    appstate
                    and appstate.experiment
                    and appstate.experiment.focus_loss_intervals
            ):
                for t in range(t_start, t_end + 1):
                    if appstate.experiment.is_focus_loss_frame(t):
                        focus_loss_skip.add(t)

        except Exception:
            pass

        frames_to_skip = set(focus_loss_skip)
        if drop_frame_zero and t_start <= 0 <= t_end:
            frames_to_skip.add(0)

        otsu_phase_uses_partaker_cells = (
                self.is_otsu_method()
                and self.cell_view_combo.currentText() == "Phase Contrast"
        )
        raw_phase_uses_partaker_cells = (
                self.is_raw_intensity_method()
                and self.cell_view_combo.currentText() == "Phase Contrast"
        )
        requires_partaker_cell_masks = (
                self.is_partaker_method()
                or otsu_phase_uses_partaker_cells
                or raw_phase_uses_partaker_cells
        )
        if requires_partaker_cell_masks and not self.validate_partaker_segmentation_available(
                selected_positions=selected_positions,
                t_start=t_start,
                t_end=t_end,
                focus_loss_skip=frames_to_skip,
        ):
            self.start_segmentation_btn.setEnabled(True)
            self.cancel_analysis_btn.setEnabled(False)
            return

        frames_to_analyze = []

        for p in selected_positions:
            for t in range(t_start, t_end + 1):

                if t in frames_to_skip:
                    continue

                frames_to_analyze.append(
                    (t, p, eps_channel)
                )

        if not frames_to_analyze:
            self.status_label.setText(
                "No valid frames remain to analyze. Check the selected time "
                "range, focus-loss intervals, and frame-zero exclusion."
            )
            self.start_segmentation_btn.setEnabled(True)
            self.cancel_analysis_btn.setEnabled(False)
            return

        try:
            self.prepare_eps_run_preprocessing(frames_to_analyze)
        except Exception as e:
            QMessageBox.warning(
                self,
                "EPS preprocessing setup failed",
                f"Could not prepare the shared EPS intensity window:\n{e}"
            )
            self.start_segmentation_btn.setEnabled(True)
            self.cancel_analysis_btn.setEnabled(False)
            return

        self.queue = frames_to_analyze
        print(
            f"Queue length: {len(frames_to_analyze)} | "
            f"drop_frame_zero={self.drop_frame_zero_for_current_run}"
        )

        self.progress_bar.setMaximum(len(self.queue))
        self.progress_bar.setValue(0)

        self.is_segmenting = True
        self.cancel_requested = False

        self.start_segmentation_btn.setEnabled(False)
        self.cancel_analysis_btn.setEnabled(True)

        frame_zero_note = (
            " Frame 0 is excluded."
            if self.drop_frame_zero_for_current_run
            else ""
        )
        self.status_label.setText(
            f"Processing {len(self.queue)} frames...{frame_zero_note}"
        )
        self.process_next_in_queue()

    def on_image_data_loaded(self, image_data):
        if image_data is None or image_data.data is None:
            return

        self.biofilm_metric_service = image_data.biofilm_metric_service

        # A calibration result belongs to one loaded dataset only.
        self.partaker_eps_setup = None
        self.partaker_drop_frame_zero = False
        self.drop_frame_zero_for_current_run = False
        if hasattr(self, "drop_frame_zero_checkbox"):
            self.drop_frame_zero_checkbox.setChecked(False)
        self.start_segmentation_btn.setEnabled(True)

        # Get dimensions from image_data
        shape = image_data.data.shape
        t_max = shape[0] - 1
        p_max = shape[1] - 1
        c_max = shape[2] - 1

        # Populate EPS and Cell channel combo boxes
        self.eps_channel_combo.clear()
        self.cell_channel_combo.clear()
        for c in range(c_max + 1):
            self.eps_channel_combo.addItem(f"Channel {c}")
            self.cell_channel_combo.addItem(f"Channel {c}")

        # Populate position list
        self.position_list.clear()
        for p in range(p_max + 1):
            self.position_list.addItem(f"Position {p}")

        # Update time range spinboxes
        self.time_start_spin.setMaximum(t_max)
        self.time_end_spin.setMaximum(t_max)
        self.time_end_spin.setValue(t_max)

        # Activate Graph Generation Button
        self.generate_graph_btn.setEnabled(True)

    def process_next_in_queue(self):

        if not self.is_segmenting:
            return

        if not self.queue:
            self._segmentation_finished()
            return

        if self.cancel_requested:
            self._segmentation_finished()
            return

        time, position, channel = self.queue.pop(0)

        from nd2_analyzer.data.image_data import ImageData

        image_data = ImageData.get_instance()

        raw_img = image_data.get(
            time,
            position,
            channel
        )

        try:
            metrics = self.process_eps_frame(
                raw_img,
                time,
                position,
                channel
            )

        except Exception as e:
            print(
                f"Failed frame "
                f"T={time} P={position}: {e}"
            )
            traceback.print_exc()

            del raw_img
            gc.collect()
            self.request_timer.start(1)
            return

        cell_channel = self.get_selected_cell_channel()

        analysis_method = metrics.get(
            "analysis_method",
            self.get_analysis_method()
        )
        result_row = {
            "time": time,
            "time_hours": self.get_time_hours(time),
            "capture_interval_value": float(self.frame_interval_value.value()),
            "capture_interval_unit": str(self.time_unit_combo.currentText()),
            "position": position,

            "eps_channel": channel,
            "cell_channel": cell_channel,

            # Keep old alias for plotting compatibility
            "channel": channel,

            "analysis_method": analysis_method,

            "cell_view_type": self.cell_view_combo.currentText(),

            "background_filtering_used": bool(self.gaus_back_corr.isChecked()),
            "eps_morphology_used": bool(self.close_dialate.isChecked()),
            "frame_zero_dropped": bool(
                self.drop_frame_zero_for_current_run
            ),

            "occupancy_fraction": metrics["occupancy_fraction"],
            "occupancy_percent": metrics["occupancy_percent"],
            "mean_intensity": metrics["mean_intensity"],
            "integrated_intensity": metrics["integrated_intensity"],
            "eps_area_pixels": metrics["eps_area_pixels"],
        }

        for optional_key in [
            "eps_threshold_used",
            "eps_thresholding_type",
            "eps_dominant_lower_bound",
            "eps_dominant_upper_bound",
            "eps_processing_black_point_raw",
            "eps_processing_white_point_raw",
            "eps_processing_window_source",
            "eps_processing_intensity_space",
            "eps_threshold_used_raw",
            "eps_dominant_lower_bound_raw",
            "eps_dominant_upper_bound_raw",

            "partaker_gaussian_sigma",
            "otsu_gaussian_sigma",
            "otsu_local_block_size_px",
            "otsu_local_neighborhood_fraction",

            "fraction_eps_coverage",
            "fraction_cells_encased",
            "areal_fraction_percent",
            "cell_occupancy_fraction",
            "cell_occupancy_percent",

            "raw_mean_intensity_whole_frame",
            "raw_integrated_intensity_whole_frame",
            "raw_mean_intensity_within_cells",
            "raw_integrated_intensity_within_cells",
            "raw_signal_within_cells_fraction",
            "integrated_intensity_metric",
            "eps_mask_used",
            "analysis_area_pixels",

            "total_cells",
            "encased_cells",
            "cell_area_pixels",
            "cell_envelope_pixels",
            "eps_colocalized_pixels",
            "eps_pixels_outside_cell_mask",
            "cell_pixels_without_eps",

            "cell_threshold_used",
            "cell_polarity_used",
            "cell_closing_radius_px",
            "cell_dilation_radius_px",
            "cell_segmentation_source",
        ]:
            if optional_key in metrics:
                result_row[optional_key] = metrics[optional_key]

        self.biofilm_metric_service.upsert_eps_frame(result_row)
        completed = self.biofilm_metric_service.eps_result_count

        self.progress_bar.setValue(
            completed
        )

        self.status_label.setText(
            f"Processing T={time} P={position}"
        )

        # Release the large per-frame arrays before scheduling the next frame.
        del metrics
        del raw_img
        gc.collect()

        self.log_process_memory(
            f"After completed frame T={time} P={position}"
        )

        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        self.request_timer.start(1)

    def select_all_positions(self):
        """Select all positions in the list"""
        for i in range(self.position_list.count()):
            self.position_list.item(i).setSelected(True)

    def select_no_positions(self):
        """Deselect all positions in the list"""
        for i in range(self.position_list.count()):
            self.position_list.item(i).setSelected(False)

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

    def _segmentation_finished(self):
        """Handle completion of all segmentation tasks."""
        self.eps_df = self.biofilm_metric_service.get_eps_results()
        self.eps_results = self.biofilm_metric_service.get_eps_result_rows()

        output_dir = self.get_eps_output_root()
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.eps_results:
            csv_rows = []

            for row in self.eps_results:
                clean_row = {}
                for key, value in row.items():
                    if isinstance(value, np.ndarray):
                        continue
                    clean_row[key] = value
                csv_rows.append(clean_row)
            metrics_df = pl.DataFrame(csv_rows)

            metrics_df.write_csv(output_dir / "eps_processing_metrics.csv")

        self.save_processing_heatmaps(output_dir)
        gif_path = self.export_eps_gif(output_dir)
        overlay_gif_paths = self.export_overlay_folder_gifs(output_dir)
        summary_plot_paths = self.save_all_summary_plots(output_dir)

        # Gaussian references are only needed during segmentation and GIF
        # rendering. Release them after all visual exports are complete.
        self.eps_reference_backgrounds = {}

        # The mask cache is only needed while building the summary GIF.
        shutil.rmtree(
            output_dir / "_gif_mask_cache",
            ignore_errors=True,
        )
        gc.collect()

        self.is_segmenting = False
        self.start_segmentation_btn.setEnabled(True)
        self.cancel_analysis_btn.setEnabled(False)

        frame_zero_note = (
            " Frame 0 was excluded."
            if self.drop_frame_zero_for_current_run
            else ""
        )

        if gif_path is not None or overlay_gif_paths or summary_plot_paths:
            self.status_label.setText(
                f"EPS analysis complete "
                f"({len(self.eps_results)} frames).{frame_zero_note} "
                f"Outputs saved to {output_dir}; "
                f"summary GIF saved: {gif_path is not None}; "
                f"overlay GIFs saved: {len(overlay_gif_paths)}; "
                f"graphs saved: {len(summary_plot_paths)}."
            )
        else:
            self.status_label.setText(
                f"EPS analysis complete "
                f"({len(self.eps_results)} frames).{frame_zero_note} "
                f"Outputs saved to {output_dir}"
            )

    def compute_avg_sd_stats(self, df: pl.DataFrame):
        """
        Compute mean ± SD metrics grouped by timepoint.

        This is intentionally flexible:
        - EPS rows have mean_intensity, integrated_intensity, eps_area_pixels
        - Cell-association rows have cell_occupancy_fraction, fraction_eps_coverage,
          fraction_cells_encased
        """
        return BiofilmMetricService.summarize_eps_dataframe(df)

    def add_time_hours_to_stats(self, stats: pl.DataFrame):
        time_hours = [
            self.get_time_hours(float(t))
            for t in stats["time"].to_list()
        ]

        return stats.with_columns(
            pl.Series("time_hours", time_hours)
        )

    @staticmethod
    def dataframe_uses_raw_intensities(df: pl.DataFrame) -> bool:
        if "analysis_method" not in df.columns or df.is_empty():
            return False

        methods = {
            str(value)
            for value in df["analysis_method"].drop_nulls().unique().to_list()
        }
        return "Raw Intensities" in methods

    def get_raw_intensity_secondary_metric(self) -> str:
        """
        Return the raw-intensity metric selected for the in-app graph only.

        Both raw integrated-intensity variants are always calculated and
        exported regardless of this display setting.
        """
        metric = str(self.RAW_INTENSITY_SECONDARY_METRIC)

        if metric not in {
            "whole_frame_integrated",
            "cell_integrated",
        }:
            raise ValueError(
                "RAW_INTENSITY_SECONDARY_METRIC must be "
                "'whole_frame_integrated' or 'cell_integrated'."
            )

        return metric

    def get_intensity_plot_titles(
            self,
            df: pl.DataFrame,
            secondary_metric: str | None = None,
    ):
        """Return graph titles matching the displayed intensity metric."""
        if not self.dataframe_uses_raw_intensities(df):
            return (
                "Mean EPS Fluorescence",
                "Integrated EPS Fluorescence",
            )

        if secondary_metric is None:
            secondary_metric = self.get_raw_intensity_secondary_metric()

        secondary_title = (
            "Integrated Raw EPS Signal Within Cell Mask"
            if secondary_metric == "cell_integrated"
            else "Whole-Frame Integrated Raw EPS Signal"
        )

        return (
            "Mean Raw EPS Intensity (Whole Frame)",
            secondary_title,
        )

    def create_intensity_sd_figure(
            self,
            *,
            time_hours: np.ndarray,
            mean_intensity: np.ndarray,
            mean_std: np.ndarray,
            integrated_intensity: np.ndarray,
            integrated_std: np.ndarray,
            mean_title: str,
            integrated_title: str,
    ):
        """Create the shared two-panel mean/integrated intensity figure."""
        fig, axs = plt.subplots(
            2,
            1,
            figsize=(8, 7),
            constrained_layout=True
        )

        axs[0].plot(
            time_hours,
            mean_intensity,
            "-o",
            color="green",
            linewidth=2
        )
        axs[0].fill_between(
            time_hours,
            mean_intensity - mean_std,
            mean_intensity + mean_std,
            color="purple",
            alpha=0.25
        )
        axs[0].set_title(mean_title)
        axs[0].set_ylabel("Mean Intensity")
        axs[0].set_xlabel("Time (hours)")

        axs[1].plot(
            time_hours,
            integrated_intensity,
            "-o",
            color="green",
            linewidth=2
        )
        axs[1].fill_between(
            time_hours,
            integrated_intensity - integrated_std,
            integrated_intensity + integrated_std,
            color="blue",
            alpha=0.25
        )
        axs[1].set_title(integrated_title)
        axs[1].set_ylabel("Integrated Intensity")
        axs[1].set_xlabel("Time (hours)")

        return fig

    @staticmethod
    def get_raw_stable_baseline_indices(time_hours: np.ndarray):
        """
        Choose baseline timepoints for raw-intensity normalization.

        Prefer 0.5, 1.0, and 1.5 hours when present. If one or more are
        unavailable, fall back to the first three strictly positive timepoints.
        If fewer than three positive timepoints exist, use however many are
        available; if none exist, use the first timepoint.
        """
        times = np.asarray(time_hours, dtype=np.float64)
        if times.size == 0:
            return np.asarray([], dtype=int)

        target_times = np.asarray([0.5, 1.0, 1.5], dtype=np.float64)
        matched_indices = []

        for target in target_times:
            candidates = np.where(np.isclose(times, target, atol=1e-9))[0]
            if candidates.size:
                matched_indices.append(int(candidates[0]))

        if len(matched_indices) == len(target_times):
            return np.asarray(matched_indices, dtype=int)

        positive_indices = np.where(times > 0)[0]
        if positive_indices.size:
            return positive_indices[:3].astype(int, copy=False)

        return np.asarray([0], dtype=int)

    def compute_relative_to_stable_baseline(
            self,
            *,
            values: np.ndarray,
            std_values: np.ndarray,
            time_hours: np.ndarray,
    ):
        """
        Normalize a time series by the median of the chosen stable baseline
        points and scale the SD by the same constant.
        """
        values = np.asarray(values, dtype=np.float64)
        std_values = np.asarray(std_values, dtype=np.float64)
        time_hours = np.asarray(time_hours, dtype=np.float64)

        baseline_indices = self.get_raw_stable_baseline_indices(time_hours)
        if baseline_indices.size == 0:
            baseline_indices = np.asarray([0], dtype=int)

        baseline_value = float(np.median(values[baseline_indices]))
        if not np.isfinite(baseline_value) or baseline_value == 0.0:
            baseline_value = 1.0

        relative_values = values / baseline_value
        relative_std = std_values / baseline_value
        baseline_times_used = time_hours[baseline_indices]

        return (
            relative_values,
            relative_std,
            baseline_value,
            baseline_times_used,
        )

    def create_relative_intensity_sd_figure(
            self,
            *,
            time_hours: np.ndarray,
            mean_relative: np.ndarray,
            mean_relative_std: np.ndarray,
            integrated_relative: np.ndarray,
            integrated_relative_std: np.ndarray,
            integrated_title: str,
            baseline_label: str,
    ):
        """Create a two-panel figure for raw intensities normalized to a stable baseline."""
        fig, axs = plt.subplots(
            2,
            1,
            figsize=(8, 7),
            constrained_layout=True
        )

        axs[0].plot(
            time_hours,
            mean_relative,
            "-o",
            color="green",
            linewidth=2
        )
        axs[0].fill_between(
            time_hours,
            mean_relative - mean_relative_std,
            mean_relative + mean_relative_std,
            color="purple",
            alpha=0.25
        )
        axs[0].axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        axs[0].set_title("Raw EPS Intensity Relative to Stable Baseline (Mean)")
        axs[0].set_ylabel("Relative Mean Intensity")
        axs[0].set_xlabel("Time (hours)")

        axs[1].plot(
            time_hours,
            integrated_relative,
            "-o",
            color="green",
            linewidth=2
        )
        axs[1].fill_between(
            time_hours,
            integrated_relative - integrated_relative_std,
            integrated_relative + integrated_relative_std,
            color="blue",
            alpha=0.25
        )
        axs[1].axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        axs[1].set_title(
            f"Raw EPS Intensity Relative to Stable Baseline ({integrated_title})"
        )
        axs[1].set_ylabel("Relative Integrated Intensity")
        axs[1].set_xlabel("Time (hours)")

        fig.suptitle(
            f"Baseline = median of {baseline_label}",
            fontsize=10,
        )

        return fig

    def save_eps_intensity_sd_plot(self, output_dir: Path):
        """
        Save intensity summary plots.

        Segmented EPS modes save the original EPS intensity figure.

        Raw-intensity mode always saves both alternatives:
        1. whole-frame mean + whole-frame integrated raw signal
        2. whole-frame mean + integrated raw signal within the cell mask

        RAW_INTENSITY_SECONDARY_METRIC does not control these exports.
        """
        if not hasattr(self, "eps_df") or self.eps_df.is_empty():
            return []

        df = self.eps_df
        eps_channel = int(self.eps_channel_combo.currentIndex())

        if "eps_channel" in df.columns:
            df_eps = df.filter(pl.col("eps_channel") == eps_channel)
        else:
            df_eps = df.filter(pl.col("channel") == eps_channel)

        if df_eps.is_empty():
            return []

        stats = self.compute_avg_sd_stats(df_eps)
        stats = self.add_time_hours_to_stats(stats)

        time_hours = stats["time_hours"].to_numpy()
        mean_int = stats["mean_intensity"].to_numpy()
        mean_std = stats["std_mean_intensity"].to_numpy()

        saved_paths = []

        if self.dataframe_uses_raw_intensities(df_eps):
            raw_plot_specs = [
                {
                    "metric_column": "raw_integrated_intensity_whole_frame",
                    "std_column": "std_raw_integrated_intensity_whole_frame",
                    "secondary_metric": "whole_frame_integrated",
                    "filename": "raw_intensity_whole_frame_sd_plot.png",
                    "csv_filename": "raw_intensity_whole_frame_sd_values.csv",
                },
                {
                    "metric_column": "raw_integrated_intensity_within_cells",
                    "std_column": "std_raw_integrated_intensity_within_cells",
                    "secondary_metric": "cell_integrated",
                    "filename": "raw_intensity_within_cells_sd_plot.png",
                    "csv_filename": "raw_intensity_within_cells_sd_values.csv",
                },
            ]

            for spec in raw_plot_specs:
                metric_column = spec["metric_column"]
                std_column = spec["std_column"]

                if (
                        metric_column not in stats.columns
                        or std_column not in stats.columns
                ):
                    continue

                integrated_int = stats[metric_column].to_numpy()
                integrated_std = stats[std_column].to_numpy()

                mean_title, integrated_title = (
                    self.get_intensity_plot_titles(
                        df_eps,
                        secondary_metric=spec["secondary_metric"],
                    )
                )

                fig = self.create_intensity_sd_figure(
                    time_hours=time_hours,
                    mean_intensity=mean_int,
                    mean_std=mean_std,
                    integrated_intensity=integrated_int,
                    integrated_std=integrated_std,
                    mean_title=mean_title,
                    integrated_title=integrated_title,
                )

                output_path = output_dir / spec["filename"]
                fig.savefig(
                    output_path,
                    dpi=300,
                    bbox_inches="tight"
                )
                plt.close(fig)

                (
                    mean_relative,
                    mean_relative_std,
                    mean_baseline_value,
                    mean_baseline_times,
                ) = self.compute_relative_to_stable_baseline(
                    values=mean_int,
                    std_values=mean_std,
                    time_hours=time_hours,
                )
                (
                    integrated_relative,
                    integrated_relative_std,
                    integrated_baseline_value,
                    integrated_baseline_times,
                ) = self.compute_relative_to_stable_baseline(
                    values=integrated_int,
                    std_values=integrated_std,
                    time_hours=time_hours,
                )

                baseline_times_for_label = mean_baseline_times
                baseline_label = ", ".join(
                    f"{value:g} h" for value in baseline_times_for_label
                )
                if not baseline_label:
                    baseline_label = "available stable timepoints"

                relative_fig = self.create_relative_intensity_sd_figure(
                    time_hours=time_hours,
                    mean_relative=mean_relative,
                    mean_relative_std=mean_relative_std,
                    integrated_relative=integrated_relative,
                    integrated_relative_std=integrated_relative_std,
                    integrated_title=integrated_title,
                    baseline_label=baseline_label,
                )

                relative_output_path = output_dir / spec["filename"].replace(
                    "_sd_plot.png",
                    "_relative_to_stable_baseline_sd_plot.png",
                )
                relative_fig.savefig(
                    relative_output_path,
                    dpi=300,
                    bbox_inches="tight"
                )
                plt.close(relative_fig)

                stats.select([
                    "time",
                    "time_hours",
                    "mean_intensity",
                    "std_mean_intensity",
                    metric_column,
                    std_column,
                ]).write_csv(
                    output_dir / spec["csv_filename"]
                )

                relative_table = pl.DataFrame({
                    "time": stats["time"],
                    "time_hours": stats["time_hours"],
                    "relative_mean_intensity": mean_relative,
                    "std_relative_mean_intensity": mean_relative_std,
                    "relative_integrated_intensity": integrated_relative,
                    "std_relative_integrated_intensity": integrated_relative_std,
                    "mean_baseline_value": np.full_like(mean_relative, mean_baseline_value, dtype=np.float64),
                    "integrated_baseline_value": np.full_like(integrated_relative, integrated_baseline_value, dtype=np.float64),
                })
                relative_table.write_csv(
                    output_dir / spec["csv_filename"].replace(
                        "_sd_values.csv",
                        "_relative_to_stable_baseline_sd_values.csv",
                    )
                )

                saved_paths.append(output_path)
                saved_paths.append(relative_output_path)

            # One combined table keeps both raw alternatives together.
            raw_value_columns = [
                column for column in [
                    "time",
                    "time_hours",
                    "mean_intensity",
                    "std_mean_intensity",
                    "raw_integrated_intensity_whole_frame",
                    "std_raw_integrated_intensity_whole_frame",
                    "raw_integrated_intensity_within_cells",
                    "std_raw_integrated_intensity_within_cells",
                ]
                if column in stats.columns
            ]
            stats.select(raw_value_columns).write_csv(
                output_dir / "raw_intensity_all_sd_values.csv"
            )

            return saved_paths

        integrated_int = stats["integrated_intensity"].to_numpy()
        integrated_std = stats["std_integrated_intensity"].to_numpy()
        mean_title, integrated_title = self.get_intensity_plot_titles(df_eps)

        fig = self.create_intensity_sd_figure(
            time_hours=time_hours,
            mean_intensity=mean_int,
            mean_std=mean_std,
            integrated_intensity=integrated_int,
            integrated_std=integrated_std,
            mean_title=mean_title,
            integrated_title=integrated_title,
        )

        output_path = output_dir / "eps_intensity_sd_plot.png"
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)

        stats.write_csv(
            output_dir / "eps_intensity_sd_values.csv"
        )

        return [output_path]

    def save_eps_cell_sd_plot(self, output_dir: Path):
        if not hasattr(self, "eps_df") or self.eps_df.is_empty():
            return None

        df = self.eps_df
        eps_channel = int(self.eps_channel_combo.currentIndex())

        if "eps_channel" in df.columns:
            df_eps = df.filter(pl.col("eps_channel") == eps_channel)
        else:
            df_eps = df.filter(pl.col("channel") == eps_channel)

        if df_eps.is_empty():
            return None

        raw_mode = self.dataframe_uses_raw_intensities(df_eps)

        if raw_mode:
            required_columns = {
                "raw_signal_within_cells_fraction",
                "cell_occupancy_fraction",
            }
        else:
            required_columns = {
                "fraction_cells_encased",
                "cell_occupancy_fraction",
            }

        if not required_columns.issubset(set(df_eps.columns)):
            return None

        stats = self.compute_avg_sd_stats(df_eps)
        stats = self.add_time_hours_to_stats(stats)
        time_hours = stats["time_hours"].to_numpy()

        if raw_mode:
            association = stats["raw_signal_within_cells_fraction"].to_numpy()
            association_std = stats[
                "std_raw_signal_within_cells_fraction"
            ].to_numpy()
            association_title = "Fraction of Raw EPS Signal Within Cell Mask"
            association_ylabel = "Total Raw Signal Within Cells (%)"
        else:
            association = stats["fraction_cells_encased"].to_numpy()
            association_std = stats["std_fraction_cells_encased"].to_numpy()
            association_title = "Cells Encased by EPS"
            association_ylabel = "Cells Encased (%)"

        cell_occ = stats["cell_occupancy"].to_numpy()
        std_cell_occ = stats["std_cell_occupancy"].to_numpy()

        fig, axs = plt.subplots(
            2,
            1,
            figsize=(8, 7),
            constrained_layout=True
        )

        axs[0].plot(
            time_hours,
            association,
            "-o",
            color="black",
            linewidth=2
        )
        axs[0].fill_between(
            time_hours,
            association - association_std,
            association + association_std,
            color="gray",
            alpha=0.25
        )
        axs[0].set_title(association_title)
        axs[0].set_ylabel(association_ylabel)
        axs[0].set_xlabel("Time (hours)")

        # Cell fractional area is intentionally unchanged in raw mode.
        axs[1].plot(
            time_hours,
            cell_occ,
            "-o",
            color="goldenrod",
            linewidth=2
        )
        axs[1].fill_between(
            time_hours,
            cell_occ - std_cell_occ,
            cell_occ + std_cell_occ,
            color="goldenrod",
            alpha=0.25
        )
        axs[1].set_title("Cell Areal Fraction")
        axs[1].set_ylabel("Area Fraction (ρ)")
        axs[1].set_xlabel("Time (hours)")

        output_path = output_dir / "eps_cell_sd_plot.png"
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)

        stats.write_csv(
            output_dir / "eps_cell_sd_values.csv"
        )

        return output_path

    def save_all_summary_plots(self, output_dir: Path):
        saved_paths = []

        intensity_paths = self.save_eps_intensity_sd_plot(output_dir)
        saved_paths.extend(intensity_paths)

        cell_path = self.save_eps_cell_sd_plot(output_dir)
        if cell_path is not None:
            saved_paths.append(cell_path)

        return saved_paths

    def on_plot_avg_sd(self):
        self.population_figure.clear()

        if not hasattr(self, "eps_df"):
            QMessageBox.warning(self, "No data", "Run EPS analysis first.")
            return

        df = self.eps_df
        eps_channel = int(self.eps_channel_combo.currentIndex())

        if "eps_channel" in df.columns:
            df_eps = df.filter(pl.col("eps_channel") == eps_channel)
        else:
            df_eps = df.filter(pl.col("channel") == eps_channel)

        if df_eps.is_empty():
            QMessageBox.warning(
                self,
                "No data",
                "No results were found for the selected EPS channel."
            )
            return

        analysis_type = self.metric_combo.currentText()
        stats = self.compute_avg_sd_stats(df_eps)

        interval_value = self.frame_interval_value.value()
        interval_unit = self.time_unit_combo.currentText()
        hours_conversion = {
            "ms": 1 / 1000 / 3600,
            "sec": 1 / 3600,
            "min": 1 / 60,
            "hr": 1,
            "day": 24
        }
        hours_per_frame = interval_value * hours_conversion[interval_unit]
        time_values = np.asarray(
            [
                float(t) * hours_per_frame
                for t in stats["time"].to_list()
            ],
            dtype=float,
        )

        fig = self.population_figure
        axs = fig.subplots(2, 1)
        raw_mode = self.dataframe_uses_raw_intensities(df_eps)

        if analysis_type == "EPS Intensity Metrics":
            mean_int = np.asarray(stats["mean_intensity"])
            mean_std = np.asarray(stats["std_mean_intensity"])

            if raw_mode:
                display_metric = self.get_raw_intensity_secondary_metric()

                if display_metric == "cell_integrated":
                    integrated_int = np.asarray(
                        stats["raw_integrated_intensity_within_cells"]
                    )
                    integrated_std = np.asarray(
                        stats["std_raw_integrated_intensity_within_cells"]
                    )
                else:
                    integrated_int = np.asarray(
                        stats["raw_integrated_intensity_whole_frame"]
                    )
                    integrated_std = np.asarray(
                        stats["std_raw_integrated_intensity_whole_frame"]
                    )

                mean_title, integrated_title = (
                    self.get_intensity_plot_titles(
                        df_eps,
                        secondary_metric=display_metric,
                    )
                )
            else:
                integrated_int = np.asarray(stats["integrated_intensity"])
                integrated_std = np.asarray(
                    stats["std_integrated_intensity"]
                )
                mean_title, integrated_title = (
                    self.get_intensity_plot_titles(df_eps)
                )

            axs[0].plot(
                time_values,
                mean_int,
                "-o",
                color="green",
                linewidth=2,
            )
            axs[0].fill_between(
                time_values,
                mean_int - mean_std,
                mean_int + mean_std,
                color="purple",
                alpha=0.25,
            )
            axs[0].set_title(mean_title)
            axs[0].set_ylabel("Mean Intensity")
            axs[0].set_xlabel("Time (hours)")

            # RAW_INTENSITY_SECONDARY_METRIC changes only this in-app view.
            # Both raw integrated-intensity alternatives are exported separately.
            axs[1].plot(
                time_values,
                integrated_int,
                "-o",
                color="green",
                linewidth=2,
            )
            axs[1].fill_between(
                time_values,
                integrated_int - integrated_std,
                integrated_int + integrated_std,
                color="blue",
                alpha=0.25,
            )
            axs[1].set_title(integrated_title)
            axs[1].set_ylabel("Integrated Intensity")
            axs[1].set_xlabel("Time (hours)")

        elif analysis_type == "EPS-Cell Association Metrics":
            if raw_mode:
                required_columns = {
                    "raw_signal_within_cells_fraction",
                    "cell_occupancy_fraction",
                }
            else:
                required_columns = {
                    "fraction_cells_encased",
                    "cell_occupancy_fraction",
                }

            if not required_columns.issubset(set(df_eps.columns)):
                QMessageBox.warning(
                    self,
                    "No cell data",
                    "EPS-cell association metrics were not found. "
                    "Re-run EPS processing."
                )
                return

            if raw_mode:
                association = np.asarray(
                    stats["raw_signal_within_cells_fraction"]
                )
                association_std = np.asarray(
                    stats["std_raw_signal_within_cells_fraction"]
                )
                association_title = "Fraction of Raw EPS Signal Within Cell Mask"
                association_ylabel = "Total Raw Signal Within Cells (%)"
            else:
                association = np.asarray(stats["fraction_cells_encased"])
                association_std = np.asarray(
                    stats["std_fraction_cells_encased"]
                )
                association_title = "Cells Encased by EPS"
                association_ylabel = "Cells Encased (%)"

            cell_occ = np.asarray(stats["cell_occupancy"])
            std_cell_occ = np.asarray(stats["std_cell_occupancy"])

            axs[0].plot(
                time_values,
                association,
                "-o",
                color="black",
                linewidth=2,
            )
            axs[0].fill_between(
                time_values,
                association - association_std,
                association + association_std,
                color="gray",
                alpha=0.25,
            )
            axs[0].set_title(association_title)
            axs[0].set_ylabel(association_ylabel)
            axs[0].set_xlabel("Time (hours)")

            # Cell fractional area is unchanged in raw-intensity mode.
            axs[1].plot(
                time_values,
                cell_occ,
                "-o",
                color="goldenrod",
                linewidth=2,
            )
            axs[1].fill_between(
                time_values,
                cell_occ - std_cell_occ,
                cell_occ + std_cell_occ,
                color="goldenrod",
                alpha=0.25,
            )
            axs[1].set_title("Cell Areal Fraction")
            axs[1].set_ylabel("Area Fraction (ρ)")
            axs[1].set_xlabel("Time (hours)")

            fig.subplots_adjust(
                top=0.92,
                bottom=0.08,
                left=0.12,
                right=0.95,
                hspace=0.4,
            )
        else:
            QMessageBox.warning(
                self,
                "Unknown analysis type",
                f"Unknown analysis type selected: {analysis_type}"
            )
            return

        self.population_canvas.draw()

        output_dir = Path("analysis_eps")
        output_dir.mkdir(exist_ok=True)

        if analysis_type == "EPS Intensity Metrics":
            output_filename = (
                "raw_intensity_selected_view.png"
                if raw_mode
                else "eps_intensity_sd_plot.png"
            )
        else:
            output_filename = "eps_cell_sd_plot.png"
        fig.savefig(
            output_dir / output_filename,
            dpi=300,
            bbox_inches="tight"
        )

    def make_dominant_bound_base_eps_mask(
            self,
            eps_image: np.ndarray,
    ):
        eps_for_thresholding = np.asarray(eps_image, dtype=float)

        finite = np.isfinite(eps_for_thresholding)

        focus_range_mask = (
                finite
                & (eps_for_thresholding >= self.EPS_DOMINANT_BOUND_INTENSITY_MIN)
                & (eps_for_thresholding <= self.EPS_DOMINANT_BOUND_INTENSITY_MAX)
        )

        values = eps_for_thresholding[focus_range_mask]

        if values.size == 0:
            empty_mask = np.zeros_like(eps_for_thresholding, dtype=bool)
            return empty_mask, eps_for_thresholding, np.nan, np.nan

        counts, edges = np.histogram(
            values,
            bins=int(self.EPS_DOMINANT_BOUND_BIN_COUNT),
            range=(
                float(self.EPS_DOMINANT_BOUND_INTENSITY_MIN),
                float(self.EPS_DOMINANT_BOUND_INTENSITY_MAX),
            ),
        )

        log_counts = np.log1p(counts.astype(float))

        peak_bin = int(np.argmax(log_counts))
        peak_log_count = float(log_counts[peak_bin])

        if peak_log_count <= 0:
            empty_mask = np.zeros_like(eps_for_thresholding, dtype=bool)
            return empty_mask, eps_for_thresholding, np.nan, np.nan

        keep_bins = (
                log_counts
                >= self.EPS_DOMINANT_BOUND_PEAK_LOG_FRACTION * peak_log_count
        )

        if self.EPS_DOMINANT_BOUND_USE_CONTIGUOUS_PEAK_BAND:
            left_bin = peak_bin
            while left_bin > 0 and keep_bins[left_bin - 1]:
                left_bin -= 1

            right_bin = peak_bin
            while right_bin < len(keep_bins) - 1 and keep_bins[right_bin + 1]:
                right_bin += 1
        else:
            kept = np.flatnonzero(keep_bins)
            left_bin = int(kept[0])
            right_bin = int(kept[-1])

        dominant_lower_bound = (
                float(edges[left_bin])
                - float(self.EPS_DOMINANT_BOUND_PADDING)
        )

        dominant_upper_bound = (
                float(edges[right_bin + 1])
                + float(self.EPS_DOMINANT_BOUND_PADDING)
        )

        dominant_lower_bound = max(
            dominant_lower_bound,
            float(self.EPS_DOMINANT_BOUND_INTENSITY_MIN),
        )

        dominant_upper_bound = min(
            dominant_upper_bound,
            float(self.EPS_DOMINANT_BOUND_INTENSITY_MAX),
        )

        dominant_band_mask = (
                finite
                & (eps_for_thresholding >= dominant_lower_bound)
                & (eps_for_thresholding <= dominant_upper_bound)
        )

        if self.EPS_INVERSE_DOMINANT_BOUND_LIMIT_TO_FOCUS_RANGE:
            base_eps_mask = focus_range_mask & ~dominant_band_mask
        else:
            base_eps_mask = finite & ~dominant_band_mask

        return (
            base_eps_mask,
            eps_for_thresholding,
            dominant_lower_bound,
            dominant_upper_bound,
        )

    def apply_partaker_fixed_eps_thresholding(
            self,
            eps_image: np.ndarray,
    ):
        eps_mask, _eps_for_thresholding, dominant_lower_bound, dominant_upper_bound = (
            self.make_dominant_bound_base_eps_mask(eps_image)
        )

        return eps_mask, dominant_lower_bound, dominant_upper_bound

    def apply_moreau_fixed_eps_thresholding(
            self,
            eps_image: np.ndarray,
    ):
        eps_mask, _eps_for_thresholding, dominant_lower_bound, dominant_upper_bound = (
            self.make_dominant_bound_base_eps_mask(eps_image)
        )

        return eps_mask, dominant_lower_bound, dominant_upper_bound

    def get_moreau_morphology_radius_px(self):
        """
        Match the fixed best-candidate script:
            radius_px = round((0.65 um / 2) / pixel_size_um)
        """

        try:
            from nd2_analyzer.data.image_data import ImageData

            image_data = ImageData.get_instance()
            pixel_size_um = self.get_pixel_size_um(image_data)

            if pixel_size_um <= 0 or pixel_size_um == 1.0:
                pixel_size_um = self.MOREAU_FALLBACK_PIXEL_SIZE_UM

        except Exception:
            pixel_size_um = self.MOREAU_FALLBACK_PIXEL_SIZE_UM

        radius_um = self.MOREAU_MORPHOLOGY_KERNEL_DIAMETER_UM / 2.0

        return max(
            1,
            int(round(radius_um / pixel_size_um))
        )

    def get_pixel_size_um(self, image_data):
        """Get pixel size in um from ImageData, with a safe fallback."""
        voxel_size = getattr(image_data, "voxel_size", None)

        if hasattr(voxel_size, "x"):
            return float(voxel_size.x)

        if isinstance(voxel_size, (int, float)):
            return float(voxel_size)

        return 1.0

    def get_eps_output_root(self):
        output_dir = Path("analysis_eps")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def get_time_hours(self, time: int) -> float:
        interval_value = self.frame_interval_value.value()
        interval_unit = self.time_unit_combo.currentText()

        hours_conversion = {
            "ms": 1 / 1000 / 3600,
            "sec": 1 / 3600,
            "min": 1 / 60,
            "hr": 1,
            "day": 24,
        }

        return float(time) * interval_value * hours_conversion[interval_unit]

    def normalize_image_for_display(self, image: np.ndarray) -> np.ndarray:
        """Generic per-frame display normalization, retained for cell images."""
        img = np.asarray(image, dtype=np.float32)
        finite_mask = np.isfinite(img)

        if not np.any(finite_mask):
            return np.zeros(img.shape, dtype=np.float32)

        finite_values = img[finite_mask]
        low, high = np.percentile(finite_values, [1, 99])

        if high <= low:
            low = float(np.min(finite_values))
            high = float(np.max(finite_values))

        if high <= low:
            return np.zeros(img.shape, dtype=np.float32)

        normalized = np.array(img, dtype=np.float32, copy=True)
        normalized -= np.float32(low)
        normalized /= np.float32(high - low)
        np.clip(normalized, 0.0, 1.0, out=normalized)
        np.nan_to_num(normalized, copy=False)
        return normalized

    def normalize_eps_image_for_display(self, image: np.ndarray) -> np.ndarray:
        """
        Display an EPS frame with the same shared window used for segmentation.

        Raw-intensity mode has no segmentation window, so it falls back to the
        generic per-frame display normalization.
        """
        black_point = self.eps_processing_black_point_raw
        white_point = self.eps_processing_white_point_raw

        if (
                black_point is None
                or white_point is None
                or not np.isfinite(black_point)
                or not np.isfinite(white_point)
                or white_point <= black_point
        ):
            return self.normalize_image_for_display(image)

        img = np.asarray(image, dtype=np.float32)
        normalized = np.array(img, dtype=np.float32, copy=True)
        normalized -= np.float32(black_point)
        normalized /= np.float32(white_point - black_point)
        np.clip(normalized, 0.0, 1.0, out=normalized)
        np.nan_to_num(normalized, copy=False)
        return normalized

    @staticmethod
    def make_single_mask_rgba(
            mask: np.ndarray,
            rgba: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Create a compact uint8 RGBA fill instead of a contourf polygon set."""
        mask = np.asarray(mask, dtype=bool)
        overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
        overlay[mask] = np.asarray(rgba, dtype=np.uint8)
        return overlay

    def make_overlap_rgba(self, cell_mask: np.ndarray, eps_mask: np.ndarray) -> np.ndarray:
        cell_mask = np.asarray(cell_mask, dtype=bool)
        eps_mask = np.asarray(eps_mask, dtype=bool)

        cell_only = cell_mask & ~eps_mask
        eps_only = eps_mask & ~cell_mask
        overlap = cell_mask & eps_mask

        # uint8 uses about 16 MB here instead of about 134 MB for float64.
        overlay = np.zeros((*cell_mask.shape, 4), dtype=np.uint8)

        # Purple = cell only
        overlay[cell_only] = (140, 0, 217, 89)

        # Green = EPS only
        overlay[eps_only] = (0, 255, 0, 115)

        # Yellow = EPS/cell overlap
        overlay[overlap] = (255, 255, 0, 166)

        return overlay

    def save_eps_on_eps_overlay(
            self,
            raw_eps_frame: np.ndarray,
            eps_mask: np.ndarray,
            output_path: Path,
    ):
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.imshow(
            self.normalize_eps_image_for_display(raw_eps_frame),
            cmap="gray"
        )

        overlay_rgb = self.eps_overlay_rgb_from_name(
            self.current_eps_overlay_color_name()
        )
        ax.imshow(
            self.make_single_mask_rgba(
                eps_mask,
                (*overlay_rgb, 64),
            )
        )

        ax.contour(
            eps_mask,
            levels=[0.5],
            colors=[tuple(channel / 255.0 for channel in overlay_rgb)],
            linewidths=0.7
        )

        ax.set_title("Processed EPS mask on shared EPS processing view", fontsize=9)
        ax.axis("off")

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
            pad_inches=0
        )

        fig.clear()
        plt.close(fig)
        del ax, fig
        gc.collect()

    def save_cells_on_cells_overlay(
            self,
            raw_cell_frame: np.ndarray,
            cell_mask: np.ndarray,
            output_path: Path,
    ):
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.imshow(
            self.normalize_image_for_display(raw_cell_frame),
            cmap="gray"
        )

        ax.imshow(
            self.make_single_mask_rgba(
                cell_mask,
                (255, 105, 180, 71),
            )
        )

        ax.contour(
            cell_mask,
            levels=[0.5],
            colors="red",
            linewidths=0.9
        )

        ax.set_title("Processed cell mask on raw cell channel", fontsize=9)
        ax.axis("off")

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
            pad_inches=0
        )

        fig.clear()
        plt.close(fig)
        del ax, fig
        gc.collect()

    def save_raw_signal_within_cells_overlay(
            self,
            raw_eps_frame: np.ndarray,
            cell_mask: np.ndarray,
            output_path: Path,
    ):
        """Show the cell mask directly on the unprocessed EPS channel."""
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.imshow(
            self.normalize_eps_image_for_display(raw_eps_frame),
            cmap="gray"
        )
        ax.imshow(
            self.make_single_mask_rgba(
                cell_mask,
                (255, 105, 180, 64),
            )
        )
        ax.contour(
            cell_mask,
            levels=[0.5],
            colors="red",
            linewidths=0.9
        )
        ax.set_title(
            "Raw EPS signal within the cell mask",
            fontsize=9,
        )
        ax.axis("off")

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
            pad_inches=0
        )

        fig.clear()
        plt.close(fig)
        del ax, fig
        gc.collect()

    def save_eps_cell_overlap_overlay(
            self,
            raw_cell_frame: np.ndarray,
            cell_mask: np.ndarray,
            eps_mask: np.ndarray,
            output_path: Path,
    ):
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.imshow(
            self.normalize_image_for_display(raw_cell_frame),
            cmap="gray"
        )

        ax.imshow(
            self.make_overlap_rgba(
                cell_mask=cell_mask,
                eps_mask=eps_mask
            )
        )

        ax.contour(
            cell_mask,
            levels=[0.5],
            colors="red",
            linewidths=0.9
        )

        ax.set_title(
            "EPS/cell overlap\n"
            "purple=cell only, green=EPS only, yellow=EPS inside cell mask",
            fontsize=9
        )

        ax.axis("off")

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
            pad_inches=0
        )

        fig.clear()
        plt.close(fig)
        del ax, fig
        gc.collect()

    def make_moreau_cell_masks(self, cell_frame: np.ndarray):
        """
        Moreau cell mask.

        Fluorescence mode:
            Match the fixed best-candidate script:
                raw_cell_frame >= 210
                then binary closing
                then binary dilation

        Phase Contrast mode:
            Normalize cell frame to 0-255,
            invert dark cells to bright,
            then threshold >= 210,
            then binary closing
            then binary dilation.
        """

        cell_frame = np.asarray(cell_frame, dtype=float)
        cell_threshold = float(self.MOREAU_CELL_THRESHOLD)

        finite_values = cell_frame[np.isfinite(cell_frame)]

        if finite_values.size == 0:
            cell_mask = np.zeros_like(cell_frame, dtype=bool)
            return cell_mask, cell_mask.copy(), cell_threshold, "empty", cell_frame

        cell_view_type = self.cell_view_combo.currentText()

        if cell_view_type == "Fluorescence":
            # Fixed-script matching path:
            # threshold the raw cell fluorescence channel directly.
            cell_for_thresholding = cell_frame
            polarity_used = "raw_fluorescence"

        else:
            # Phase contrast path:
            # normalize, then invert so dark cells become bright.
            low = np.percentile(finite_values, 1)
            high = np.percentile(finite_values, 99)

            if high <= low:
                cell_mask = np.zeros_like(cell_frame, dtype=bool)
                return cell_mask, cell_mask.copy(), cell_threshold, "flat", cell_frame

            cell_for_thresholding = (cell_frame - low) / (high - low)
            cell_for_thresholding = np.clip(cell_for_thresholding, 0, 1) * 255.0
            cell_for_thresholding = 255.0 - cell_for_thresholding

            polarity_used = "phase_contrast_inverted"

        cell_mask = cell_for_thresholding >= cell_threshold

        morphology_radius_px = self.get_moreau_morphology_radius_px()
        morphology_kernel = disk(morphology_radius_px)

        cell_mask = binary_closing(
            cell_mask,
            footprint=morphology_kernel
        )

        cell_mask = binary_dilation(
            cell_mask,
            footprint=morphology_kernel
        )

        cell_area_mask = cell_mask
        cell_envelope_mask = cell_mask.copy()

        cell_occupancy = (
            np.count_nonzero(cell_mask) / cell_mask.size
            if cell_mask.size > 0
            else 0.0
        )

        print(
            f"Moreau cell view={cell_view_type} | "
            f"polarity={polarity_used} | "
            f"threshold={cell_threshold} | "
            f"morph_radius_px={morphology_radius_px} | "
            f"cell_occ={cell_occupancy:.4f}"
        )

        return (
            cell_area_mask,
            cell_envelope_mask,
            cell_threshold,
            polarity_used,
            cell_for_thresholding,
        )

    def apply_eps_channel_morphology(
            self,
            eps_mask: np.ndarray,
            closing_radius_px: int = 0,
            dilation_radius_px: int = 0,
    ):
        """
        Optional EPS morphology after binarization.

        Order is opening -> closing -> dilation.
        Radius 0 means that operation is skipped.
        """

        morphed = eps_mask.copy()

        if closing_radius_px > 0:
            morphed = binary_closing(
                morphed,
                footprint=disk(closing_radius_px),
            )

        if dilation_radius_px > 0:
            morphed = binary_dilation(
                morphed,
                footprint=disk(dilation_radius_px),
            )

        return morphed

    def get_otsu_local_block_size_px(self, image_shape):
        """
        Local OTSU neighborhood size.

        Uses 1/8 of the frame width/height.
        For a 2048 x 2044 frame, this gives about 256 px.
        """
        height, width = image_shape[:2]

        block_size = int(round(min(height, width) / 8))

        block_size = max(3, block_size)

        # Odd size is cleaner for centered local neighborhoods.
        if block_size % 2 == 0:
            block_size += 1

        return block_size

    def prepare_image_for_rank_otsu_without_normalization(self, image: np.ndarray):
        """
        Convert an already windowed 0-255 image to uint8 for rank OTSU.

        Do not call the raw-image window conversion again here; the shared EPS
        intensity mapping has already been applied in process_eps_frame().
        """
        work = np.asarray(image, dtype=np.float32)
        work = np.nan_to_num(
            work,
            copy=True,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )
        np.clip(work, 0.0, 255.0, out=work)
        np.rint(work, out=work)
        return work.astype(np.uint8, copy=False)

    def convert_eps_image_to_uint8_for_processing(self, image: np.ndarray):
        """
        Map raw EPS intensities to uint8 using the shared run-level window.

        Values at or below the shared black point become 0. Values at or above
        the shared white point become 255. The same mapping is reused across all
        selected positions and timepoints.
        """
        img = np.asarray(image, dtype=np.float32)
        finite_mask = np.isfinite(img)

        if not np.any(finite_mask):
            return np.zeros(img.shape, dtype=np.uint8)

        black_point = self.eps_processing_black_point_raw
        white_point = self.eps_processing_white_point_raw

        if (
                black_point is None
                or white_point is None
                or not np.isfinite(black_point)
                or not np.isfinite(white_point)
                or white_point <= black_point
        ):
            raise RuntimeError(
                "The EPS run-level intensity window has not been initialized."
            )

        work = np.array(img, dtype=np.float32, copy=True)
        np.nan_to_num(
            work,
            copy=False,
            nan=float(black_point),
            posinf=float(white_point),
            neginf=float(black_point),
        )

        work -= np.float32(black_point)
        work /= np.float32(white_point - black_point)
        np.clip(work, 0.0, 1.0, out=work)
        work *= np.float32(255.0)
        np.rint(work, out=work)
        return work.astype(np.uint8, copy=False)

    def segment_eps_partaker(
            self,
            gaussian_corrected: np.ndarray,
            image: np.ndarray,
            smoothing_sigma: float = 1.5,
    ):
        print(
            f"Starting Partaker Gaussian smoothing and fixed EPS thresholding "
            f"for {self}"
        )
        self.log_process_memory("Before Partaker Gaussian smoothing")

        # Light smoothing before segmentation. The input is uint8, so retain
        # uint8 output instead of allowing a larger floating-point result.
        smoothed = np.empty_like(gaussian_corrected, dtype=np.uint8)
        gaussian_filter(
            gaussian_corrected,
            sigma=smoothing_sigma,
            output=smoothed,
        )

        threshold = int(self.partaker_eps_threshold)
        eps_mask = smoothed >= threshold

        # Morphological Dilation & Closing
        if self.close_dialate.isChecked():
            from nd2_analyzer.data.image_data import ImageData

            image_data = ImageData.get_instance()
            pixel_size_um = self.get_pixel_size_um(image_data)

            kernel_radius_px = max(1, int(round((0.65 / 2) / pixel_size_um)))
            kernel = disk(kernel_radius_px)

            eps_mask = binary_closing(eps_mask, footprint=kernel)
            eps_mask = binary_dilation(eps_mask, footprint=kernel)

        return smoothed, eps_mask, {
            "analysis_method": "Partaker",
            "eps_threshold_used": float(threshold),
            "eps_thresholding_type": "fixed_global_uint8",
            "partaker_gaussian_sigma": float(smoothing_sigma),
        }

    def segment_eps_moreau(
            self,
            gaussian_corrected: np.ndarray,
            image: np.ndarray,
            time: int,
            position: int,
    ):

        # Light smoothing before segmentation
        smoothed = gaussian_filter(
            gaussian_corrected,
            sigma=self.MOREAU_SMOOTHING_SIGMA
        )

        # Moreau dominant-bound EPS thresholding
        print(f"Starting Moreau dominant-bound EPS thresholding for {self}")

        from nd2_analyzer.data.image_data import ImageData
        image_data = ImageData.get_instance()

        # Select cell channel / phase image OR fluorescence
        cell_channel = self.get_selected_cell_channel()
        cell_image = image_data.get(
            time,
            position,
            cell_channel
        )

        (
            cell_area_mask,
            cell_envelope_mask,
            cell_threshold,
            cell_polarity_used,
            cell_for_thresholding,
        ) = self.make_moreau_cell_masks(
            cell_frame=cell_image
        )

        eps_mask, dominant_lower_bound, dominant_upper_bound = (
            self.apply_moreau_fixed_eps_thresholding(
                eps_image=smoothed,
            )
        )
        eps_threshold = float(dominant_upper_bound)

        if self.close_dialate.isChecked():
            eps_mask = self.apply_eps_channel_morphology(
                eps_mask=eps_mask,
                closing_radius_px=self.MOREAU_EPS_CLOSING_RADIUS_PX,
                dilation_radius_px=self.MOREAU_EPS_DILATION_RADIUS_PX,
            )

        # The paper-style numerator is only the EPS-positive region inside the
        # dilated cell-envelope mask, even if EPS morphology expands outside it.
        colocalized_mask = eps_mask & cell_envelope_mask

        cell_morphology_radius_px = self.get_moreau_morphology_radius_px()

        print(
            f"P={position} T={time} | "
            f"mode=moreau | "
            f"thr={eps_threshold:.3f}"
        )

        return smoothed, eps_mask, {
            "analysis_method": "Moreau",
            "eps_threshold_used": float(eps_threshold),

            "cell_channel": int(cell_channel),
            "cell_mask": cell_area_mask,
            "cell_thresholding_image": cell_for_thresholding,

            "cell_threshold_used": float(cell_threshold),
            "cell_polarity_used": cell_polarity_used,
            "cell_closing_radius_px": int(cell_morphology_radius_px),
            "cell_dilation_radius_px": int(cell_morphology_radius_px),
            "eps_dominant_lower_bound": float(dominant_lower_bound),
            "eps_dominant_upper_bound": float(dominant_upper_bound),
        }

    def segment_eps_otsu(
            self,
            gaussian_corrected: np.ndarray,
            image: np.ndarray,
            time: int,
            position: int,
    ):
        """
        OTSU EPS segmentation mode.

        EPS segmentation:
            local OTSU
            neighborhood = 1/8 of frame size
            no normalization

        Cell handling is not done here.
        It is handled in process_eps_frame().
        """
        from skimage.filters.rank import otsu as rank_otsu

        # Light Gaussian blur before local OTSU.
        # This smooths small pixel noise without normalizing intensity.
        smoothed = gaussian_filter(
            gaussian_corrected,
            sigma=self.OTSU_SMOOTHING_SIGMA
        )
        eps_for_otsu = self.prepare_image_for_rank_otsu_without_normalization(
            smoothed
        )

        block_size_px = self.get_otsu_local_block_size_px(
            eps_for_otsu.shape
        )

        footprint = np.ones(
            (block_size_px, block_size_px),
            dtype=np.uint8
        )

        print(
            f"Starting local OTSU EPS segmentation | "
            f"P={position} T={time} | "
            f"block_size_px={block_size_px}"
        )

        local_threshold = rank_otsu(
            eps_for_otsu,
            footprint=footprint
        )

        eps_mask = eps_for_otsu > local_threshold

        if self.close_dialate.isChecked():
            from nd2_analyzer.data.image_data import ImageData

            image_data = ImageData.get_instance()
            pixel_size_um = self.get_pixel_size_um(image_data)

            kernel_radius_px = max(
                1,
                int(round((0.65 / 2) / pixel_size_um))
            )

            kernel = disk(kernel_radius_px)

            eps_mask = binary_closing(
                eps_mask,
                footprint=kernel
            )
            eps_mask = binary_dilation(
                eps_mask,
                footprint=kernel
            )

        return eps_for_otsu, eps_mask, {
            "analysis_method": "OTSU",
            "eps_threshold_used": "local_otsu",
            "eps_thresholding_type": "local_otsu",
            "otsu_gaussian_sigma": float(self.OTSU_SMOOTHING_SIGMA),
            "otsu_local_block_size_px": int(block_size_px),
            "otsu_local_neighborhood_fraction": 1 / 8,
        }

    def get_partaker_cell_mask_for_frame(
            self,
            time: int,
            position: int,
    ):
        """
        Get the saved Partaker cell labels for the selected Cell Channel.

        Edge-centroid filtering is already applied when the segmentation is saved
        by SegmentationService.
        """
        from nd2_analyzer.data.image_data import ImageData

        image_data = ImageData.get_instance()
        segmented_storage = image_data.segmentation_cache
        model_name = segmented_storage.model_name
        cache = segmented_storage.with_model(model_name)

        selected_cell_channel = self.get_selected_cell_channel()

        cell_labels = np.asarray(
            cache[(
                time,
                position,
                selected_cell_channel,
                model_name,
            )]
        )

        cell_mask = cell_labels > 0

        return cell_mask, cell_labels

    def compute_eps_cell_metrics_from_masks(
            self,
            eps_mask: np.ndarray,
            cell_mask: np.ndarray,
            cell_labels: np.ndarray | None = None,
            overlap_threshold: float = 0.10,
    ):
        return BiofilmMetricService.calculate_eps_mask_metrics(
            eps_mask=eps_mask,
            cell_mask=cell_mask,
            cell_labels=cell_labels,
            overlap_threshold=overlap_threshold,
        )

    def compute_raw_intensity_cell_metrics(
            self,
            raw_eps_frame: np.ndarray,
            cell_mask: np.ndarray,
            cell_labels: np.ndarray | None = None,
    ):
        """
        Measure unprocessed EPS-channel signal relative to the cell mask.

        raw_signal_within_cells_fraction is:
            100 * integrated raw signal inside cells / integrated raw signal
            across the entire frame.
        """
        return BiofilmMetricService.calculate_eps_raw_intensity_metrics(
            raw_eps_frame=raw_eps_frame,
            cell_mask=cell_mask,
            cell_labels=cell_labels,
        )

    @staticmethod
    def log_process_memory(prefix: str):
        """Print resident memory without making psutil a required dependency."""
        try:
            import psutil

            rss_mb = (
                    psutil.Process(os.getpid()).memory_info().rss
                    / (1024.0 * 1024.0)
            )
            print(f"{prefix} | RSS={rss_mb:.1f} MB")
            return
        except Exception:
            pass

        try:
            import resource
            import sys

            max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # macOS reports bytes; Linux reports KiB.
            rss_mb = (
                max_rss / (1024.0 * 1024.0)
                if sys.platform == "darwin"
                else max_rss / 1024.0
            )
            print(f"{prefix} | max RSS={rss_mb:.1f} MB")
        except Exception:
            # Memory logging is diagnostic only.
            pass

    def reset_output_tracking(self):
        self.heatmap_rows = []
        self.gif_rows = []
        self.overlay_gif_rows = []

    def append_intensity_heatmap_rows(
            self,
            *,
            source: str,
            stage: str,
            image: np.ndarray,
            time: int,
            position: int,
            channel: int,
            mask: np.ndarray | None = None,
            intensity_min: float = 0.0,
            intensity_max: float = 255.0,
            bin_count: int = 256,
    ):
        """
        Store one compact histogram record per frame/stage.

        The previous implementation stored 256 Python dictionaries for every
        histogram. Four histograms per frame produced 1,024 dictionaries per
        frame and tens of thousands of retained objects.
        """
        values = np.asarray(image)

        if mask is not None:
            values = values[np.asarray(mask, dtype=bool)]

        values = values[np.isfinite(values)]
        values = values[
            (values >= intensity_min)
            & (values <= intensity_max)
            ]

        counts, edges = np.histogram(
            values,
            bins=bin_count,
            range=(intensity_min, intensity_max),
        )

        if not hasattr(self, "heatmap_rows"):
            self.heatmap_rows = []

        self.heatmap_rows.append({
            "source": source,
            "stage": stage,
            "time": int(time),
            "time_hours": float(self.get_time_hours(time)),
            "position": int(position),
            "channel": int(channel),
            "bin_starts": edges[:-1].astype(np.float32, copy=False),
            "bin_ends": edges[1:].astype(np.float32, copy=False),
            "counts": counts.astype(np.int64, copy=False),
        })

    def save_processing_heatmaps(self, output_dir: Path):
        if not hasattr(self, "heatmap_rows") or not self.heatmap_rows:
            return

        heatmap_dir = output_dir / "heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        all_csv = heatmap_dir / "all_processing_heatmap_values.csv"
        fieldnames = [
            "source",
            "stage",
            "time",
            "time_hours",
            "position",
            "channel",
            "bin_start",
            "bin_end",
            "count",
        ]

        with open(all_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in self.heatmap_rows:
                for bin_start, bin_end, count in zip(
                        row["bin_starts"],
                        row["bin_ends"],
                        row["counts"],
                ):
                    writer.writerow({
                        "source": row["source"],
                        "stage": row["stage"],
                        "time": row["time"],
                        "time_hours": row["time_hours"],
                        "position": row["position"],
                        "channel": row["channel"],
                        "bin_start": float(bin_start),
                        "bin_end": float(bin_end),
                        "count": int(count),
                    })

        grouped_rows = {}
        for row in self.heatmap_rows:
            key = (row["source"], row["stage"])
            grouped_rows.setdefault(key, []).append(row)

        for (source, stage), rows in grouped_rows.items():
            times = sorted(set(row["time_hours"] for row in rows))
            time_to_col = {value: i for i, value in enumerate(times)}

            bins = np.asarray(rows[0]["bin_starts"], dtype=np.float32)
            matrix = np.zeros(
                (len(bins), len(times)),
                dtype=np.float64,
            )

            for row in rows:
                matrix[:, time_to_col[row["time_hours"]]] += row["counts"]

            label = f"{source}_{stage}_processing_heatmap"
            heatmap_csv = heatmap_dir / f"{label}.csv"

            with open(heatmap_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["bin_start"] + [f"time_h_{t:g}" for t in times])

                for bin_index, bin_value in enumerate(bins):
                    writer.writerow(
                        [float(bin_value)] + matrix[bin_index, :].tolist()
                    )

            fig, ax = plt.subplots(figsize=(12, 7))

            if len(times) > 1:
                dt = float(np.median(np.diff(times)))
            else:
                dt = 1.0

            if not np.isfinite(dt) or dt <= 0:
                dt = 1.0

            extent = [
                float(times[0] - dt / 2),
                float(times[-1] + dt / 2),
                float(bins[0]),
                float(bins[-1]),
            ]

            heatmap_image = ax.imshow(
                np.log1p(matrix),
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap="viridis",
                interpolation="nearest",
            )

            cbar = fig.colorbar(heatmap_image, ax=ax)
            cbar.set_label("log1p(pixel count)")

            ax.set_title(
                f"{source.upper()} channel "
                f"{stage.replace('_', ' ')} processing intensity heatmap"
            )
            ax.set_xlabel("Time (hours)")
            ax.set_ylabel("Intensity")

            fig.tight_layout()
            fig.savefig(
                heatmap_dir / f"{label}.png",
                dpi=300,
                bbox_inches="tight",
            )
            fig.clear()
            plt.close(fig)

    def build_gif_stats(self):
        if not hasattr(self, "gif_rows") or not self.gif_rows:
            return []

        by_time = {}

        for row in self.gif_rows:
            t = float(row["time_hours"])
            by_time.setdefault(t, []).append(
                float(row.get("association_value", 0.0))
            )

        stats = []

        for t in sorted(by_time.keys()):
            values = np.asarray(by_time[t], dtype=float)

            stats.append({
                "time_hours": float(t),
                "mean_association_value": float(np.mean(values)),
                "std_association_value": float(np.std(values)),
            })

        return stats

    def render_eps_gif_frame(
            self,
            *,
            row: dict,
            stats: list[dict],
            frame_number: int,
            total_frames: int,
    ):
        from nd2_analyzer.data.image_data import ImageData
        image_data = ImageData.get_instance()

        time = int(row["time"])
        position = int(row["position"])
        eps_channel = int(row["eps_channel"])
        cell_channel = int(row["cell_channel"])

        with np.load(row["mask_cache_path"], allow_pickle=False) as mask_data:
            mask_shape = tuple(
                int(value) for value in mask_data["mask_shape"]
            )
            pixel_count = int(np.prod(mask_shape))

            eps_mask = np.unpackbits(
                mask_data["eps_mask_packed"],
                count=pixel_count,
            ).reshape(mask_shape).astype(bool, copy=False)

            cell_mask = np.unpackbits(
                mask_data["cell_mask_packed"],
                count=pixel_count,
            ).reshape(mask_shape).astype(bool, copy=False)

        raw_eps_frame = image_data.get(
            time,
            position,
            eps_channel,
        )
        raw_cell_frame = image_data.get(
            time,
            position,
            cell_channel,
        )

        current_time = float(row["time_hours"])
        graph_times_all = np.asarray(
            [item["time_hours"] for item in stats],
            dtype=np.float32,
        )
        graph_y_all = np.asarray(
            [item["mean_association_value"] for item in stats],
            dtype=np.float32,
        )

        association_kind = row.get(
            "association_kind",
            "fraction_cells_encased",
        )
        if association_kind == "raw_signal_within_cells_fraction":
            association_ylabel = "Total Raw Signal Within Cells (%)"
            association_title = "Raw EPS signal within cell mask (%)"
        else:
            association_ylabel = "Cells encased by EPS (%)"
            association_title = "EPS/cell association over time"

        reveal_mask = graph_times_all <= current_time + 1e-9
        t_reveal = graph_times_all[reveal_mask]
        y_reveal = graph_y_all[reveal_mask]

        fig = plt.figure(figsize=(12.5, 7.0), dpi=130)
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.25, 1.0],
            height_ratios=[1.0, 1.0],
            wspace=0.22,
            hspace=0.30,
        )

        ax_graph = fig.add_subplot(gs[:, 0])
        ax_raw_eps = fig.add_subplot(gs[0, 1])
        ax_overlap = fig.add_subplot(gs[1, 1])

        ax_graph.plot(
            t_reveal,
            y_reveal,
            "-o",
            color="black",
            linewidth=2.6,
            markersize=5.0,
        )

        if t_reveal.size:
            ax_graph.scatter(
                [t_reveal[-1]],
                [y_reveal[-1]],
                s=60,
                color="black",
                zorder=5,
            )

        if graph_times_all.size:
            x_min = float(np.nanmin(graph_times_all))
            x_max = float(np.nanmax(graph_times_all))

            if x_max <= x_min:
                x_min -= 0.5
                x_max += 0.5
            else:
                pad = max((x_max - x_min) * 0.035, 0.25)
                x_min -= pad
                x_max += pad

            ax_graph.set_xlim(x_min, x_max)

        ax_graph.set_ylim(0, 105)
        ax_graph.set_xlabel("Time (hours)")
        ax_graph.set_ylabel(association_ylabel)
        ax_graph.set_title(
            f"{association_title}\n"
            f"method={self.get_analysis_method()}, "
            f"cell view={self.cell_view_combo.currentText()}"
        )
        ax_graph.grid(alpha=0.25)

        eps_display_frame = (
            self.apply_eps_reference_background(raw_eps_frame, position)
            if self.gaus_back_corr.isChecked()
            else raw_eps_frame
        )
        ax_raw_eps.imshow(
            self.normalize_eps_image_for_display(eps_display_frame),
            cmap="gray",
        )
        eps_view_name = (
            "Background-corrected EPS processing view"
            if self.gaus_back_corr.isChecked()
            else "EPS processing view"
        )
        ax_raw_eps.set_title(
            f"{eps_view_name}\n"
            f"P={position}, T={time}, EPS C={eps_channel}",
            fontsize=9,
        )
        ax_raw_eps.axis("off")

        ax_overlap.imshow(
            self.normalize_image_for_display(raw_cell_frame),
            cmap="gray",
        )
        ax_overlap.imshow(
            self.make_overlap_rgba(
                cell_mask=cell_mask,
                eps_mask=eps_mask,
            )
        )
        ax_overlap.contour(
            cell_mask,
            levels=[0.5],
            colors="red",
            linewidths=0.55,
        )
        ax_overlap.set_title(
            "EPS/cell overlap\n"
            "purple=cell only, green=EPS only, yellow=EPS inside cell mask",
            fontsize=9,
        )
        ax_overlap.axis("off")

        fig.suptitle(
            f"Frame {frame_number + 1}/{total_frames}",
            fontsize=12,
        )
        fig.canvas.draw()

        rgba = np.asarray(fig.canvas.buffer_rgba())
        rendered_image = Image.fromarray(
            rgba.copy(),
            mode="RGBA",
        ).convert("RGB")

        fig.clear()
        plt.close(fig)
        del fig, rgba, eps_mask, cell_mask
        gc.collect()

        return rendered_image

    def export_eps_gif(self, output_dir: Path):
        if not hasattr(self, "gif_rows") or not self.gif_rows:
            return None

        stats = self.build_gif_stats()
        if not stats:
            return None

        if self.EXPORT_VISUALS_ONLY_FOR_ONE_POSITION:
            display_position = int(self.EXPORT_VISUAL_POSITION)
        else:
            selected_positions = self.get_selected_positions()
            display_position = (
                int(selected_positions[0])
                if selected_positions
                else int(self.gif_rows[0]["position"])
            )

        display_rows = [
            row for row in self.gif_rows
            if int(row["position"]) == display_position
               and "mask_cache_path" in row
        ]
        display_rows = sorted(
            display_rows,
            key=lambda row: (row["time_hours"], row["time"]),
        )

        if not display_rows:
            return None

        total_frames = len(display_rows)
        print(f"Rendering EPS GIF with {total_frames} frames...")

        first_frame = self.render_eps_gif_frame(
            row=display_rows[0],
            stats=stats,
            frame_number=0,
            total_frames=total_frames,
        )

        def remaining_frames():
            for frame_number, row in enumerate(display_rows[1:], start=1):
                frame = self.render_eps_gif_frame(
                    row=row,
                    stats=stats,
                    frame_number=frame_number,
                    total_frames=total_frames,
                )
                try:
                    yield frame
                finally:
                    frame.close()

        gif_path = output_dir / "eps_cell_overlap_summary.gif"
        try:
            first_frame.save(
                gif_path,
                save_all=True,
                append_images=remaining_frames(),
                duration=1000,
                loop=0,
                optimize=False,
            )
        finally:
            first_frame.close()
            gc.collect()

        print(f"Saved EPS GIF: {gif_path}")
        return gif_path

    def add_time_label_to_gif_frame(
            self,
            image_path: str | Path,
            time_index: int,
    ):
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

        # Add a white title bar at the top so T0/T1/T2 is always readable.
        title_height = 46

        labeled = Image.new(
            "RGB",
            (image.width, image.height + title_height),
            color=(255, 255, 255)
        )

        labeled.paste(image, (0, title_height))
        image.close()

        draw = ImageDraw.Draw(labeled)
        label = f"T{time_index}"

        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except Exception:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (labeled.width - text_width) // 2
        y = (title_height - text_height) // 2

        draw.text(
            (x, y),
            label,
            fill=(0, 0, 0),
            font=font
        )

        return labeled

    def export_overlay_gif_for_folder(
            self,
            *,
            rows: list[dict],
            image_key: str,
            output_path: Path,
            position: int,
    ):
        position_rows = sorted(
            (
                row for row in rows
                if int(row["position"]) == int(position)
            ),
            key=lambda row: int(row["time"]),
        )

        if not position_rows:
            return None

        first_frame = self.add_time_label_to_gif_frame(
            image_path=position_rows[0][image_key],
            time_index=int(position_rows[0]["time"]),
        )

        def remaining_frames():
            for row in position_rows[1:]:
                frame = self.add_time_label_to_gif_frame(
                    image_path=row[image_key],
                    time_index=int(row["time"]),
                )
                try:
                    yield frame
                finally:
                    frame.close()

        try:
            first_frame.save(
                output_path,
                save_all=True,
                append_images=remaining_frames(),
                duration=int(self.EXPORT_GIF_DELAY_MS),
                loop=0,
                optimize=False,
            )
        finally:
            first_frame.close()
            gc.collect()

        print(f"Saved overlay GIF: {output_path}")
        return output_path

    def export_overlay_folder_gifs(self, output_dir: Path):
        if not hasattr(self, "overlay_gif_rows") or not self.overlay_gif_rows:
            return []

        eps_on_eps_dir = output_dir / "eps_on_eps"
        cells_on_cells_dir = output_dir / "cells_on_cells"

        positions = sorted({
            int(row["position"])
            for row in self.overlay_gif_rows
        })
        saved_paths = []

        for position in positions:
            eps_gif_path = self.export_overlay_gif_for_folder(
                rows=self.overlay_gif_rows,
                image_key="eps_on_eps_path",
                output_path=eps_on_eps_dir / f"eps_on_eps_pos{position}.gif",
                position=position,
            )

            if eps_gif_path is not None:
                saved_paths.append(eps_gif_path)
            cells_gif_path = self.export_overlay_gif_for_folder(
                rows=self.overlay_gif_rows,
                image_key="cells_on_cells_path",
                output_path=cells_on_cells_dir / f"cells_on_cells_pos{position}.gif",
                position=position,
            )
            if cells_gif_path is not None:
                saved_paths.append(cells_gif_path)

        return saved_paths

    def process_eps_frame(self,
                          frame: np.ndarray,
                          time: int,
                          position: int,
                          channel: int,
                          background_sigma: float = 200,
                          smoothing_sigma: float = 1.5,
                          ):
        """
        Process one EPS microscopy frame.

        Raw Intensities bypasses every EPS filtering/segmentation operation and
        measures the original frame directly. Other modes retain the existing
        mask-based workflow.
        """
        if frame is None:
            raise ValueError("EPS frame is None.")

        image = np.asarray(frame, dtype=np.float32)
        print(f"Original shape: {image.shape}")

        raw_intensity_mode = self.is_raw_intensity_method()
        eps_processing_image = None
        eps_mask = None
        smoothed = None
        eps_display_frame = image

        if raw_intensity_mode:
            print(
                f"Using raw EPS intensities | "
                f"T={time} P={position} C={channel}"
            )
            extra_metrics = {
                "analysis_method": "Raw Intensities",
                "eps_thresholding_type": "none_raw_frame",
                "eps_mask_used": False,
            }
        else:
            # If selected, subtract the same T=0 Gaussian background reference
            # from every frame at this position. This keeps the processing scale
            # consistent over time instead of correcting only the first frame.
            if self.gaus_back_corr.isChecked():
                print(
                    f"Applying cached Gaussian background subtraction | "
                    f"T={time} P={position} sigma={self.EPS_BACKGROUND_SIGMA:g}"
                )
                gaussian_corrected = self.apply_eps_reference_background(
                    image,
                    position,
                )
            else:
                gaussian_corrected = image

            # Use this same intensity space for EPS visualization so the mask
            # is not shown over an unrelated per-frame display transform.
            eps_display_frame = gaussian_corrected

            eps_processing_image = (
                self.convert_eps_image_to_uint8_for_processing(
                    gaussian_corrected
                )
            )

            print(
                f"EPS uint8 processing image | "
                f"T={time} P={position} C={channel} | "
                f"min={eps_processing_image.min()} "
                f"max={eps_processing_image.max()}"
            )

            if self.is_moreau_method():
                smoothed, eps_mask, extra_metrics = self.segment_eps_moreau(
                    gaussian_corrected=eps_processing_image,
                    image=image,
                    time=time,
                    position=position,
                )
            elif self.is_otsu_method():
                smoothed, eps_mask, extra_metrics = self.segment_eps_otsu(
                    gaussian_corrected=eps_processing_image,
                    image=image,
                    time=time,
                    position=position,
                )
            else:
                smoothed, eps_mask, extra_metrics = self.segment_eps_partaker(
                    gaussian_corrected=eps_processing_image,
                    image=image,
                    smoothing_sigma=self.partaker_eps_smoothing_sigma,
                )

            extra_metrics.update({
                "eps_processing_black_point_raw": float(
                    self.eps_processing_black_point_raw
                ),
                "eps_processing_white_point_raw": float(
                    self.eps_processing_white_point_raw
                ),
                "eps_processing_window_source": str(
                    self.eps_processing_window_source
                ),
                "eps_processing_intensity_space": (
                    "t0_gaussian_background_corrected"
                    if self.gaus_back_corr.isChecked()
                    else "raw"
                ),
            })

            raw_equivalent_keys = {
                "eps_threshold_used": "eps_threshold_used_raw",
                "eps_dominant_lower_bound": "eps_dominant_lower_bound_raw",
                "eps_dominant_upper_bound": "eps_dominant_upper_bound_raw",
            }
            for processed_key, raw_key in raw_equivalent_keys.items():
                if processed_key in extra_metrics:
                    raw_value = self.eps_processed_value_to_raw(
                        extra_metrics[processed_key]
                    )
                    if np.isfinite(raw_value):
                        extra_metrics[raw_key] = float(raw_value)

        # -----------------------------
        # Cell mask for association metrics
        # -----------------------------
        from nd2_analyzer.data.image_data import ImageData

        image_data = ImageData.get_instance()
        cell_channel = self.get_selected_cell_channel()
        cell_image = image_data.get(
            time,
            position,
            cell_channel
        )

        cell_labels = None
        cell_view_type = self.cell_view_combo.currentText()

        if self.is_moreau_method():
            cell_mask = extra_metrics.get("cell_mask")
            cell_thresholding_image = extra_metrics.get(
                "cell_thresholding_image",
                cell_image
            )

            if cell_mask is None:
                (
                    cell_mask,
                    _cell_envelope_mask,
                    cell_threshold,
                    cell_polarity_used,
                    cell_thresholding_image,
                ) = self.make_moreau_cell_masks(
                    cell_frame=cell_image
                )

                cell_morphology_radius_px = (
                    self.get_moreau_morphology_radius_px()
                )

                extra_metrics.update({
                    "cell_channel": int(cell_channel),
                    "cell_mask": cell_mask,
                    "cell_thresholding_image": cell_thresholding_image,
                    "cell_threshold_used": float(cell_threshold),
                    "cell_polarity_used": cell_polarity_used,
                    "cell_closing_radius_px": int(
                        cell_morphology_radius_px
                    ),
                    "cell_dilation_radius_px": int(
                        cell_morphology_radius_px
                    ),
                    "cell_segmentation_source": "moreau_thresholding",
                })

        elif self.is_otsu_method() or raw_intensity_mode:
            if cell_view_type == "Fluorescence":
                (
                    cell_mask,
                    _cell_envelope_mask,
                    cell_threshold,
                    cell_polarity_used,
                    cell_thresholding_image,
                ) = self.make_moreau_cell_masks(
                    cell_frame=cell_image
                )

                source_name = (
                    "raw_intensity_fluorescence_thresholding"
                    if raw_intensity_mode
                    else "moreau_fluorescence_thresholding"
                )
                extra_metrics.update({
                    "cell_threshold_used": float(cell_threshold),
                    "cell_polarity_used": cell_polarity_used,
                    "cell_channel": int(cell_channel),
                    "cell_segmentation_source": source_name,
                })
            else:
                cell_mask, cell_labels = (
                    self.get_partaker_cell_mask_for_frame(
                        time=time,
                        position=position
                    )
                )
                cell_thresholding_image = cell_image

                extra_metrics.update({
                    "cell_channel": int(cell_channel),
                    "cell_segmentation_source": (
                        "partaker_segmented_mask"
                    ),
                })
        else:
            cell_mask, cell_labels = (
                self.get_partaker_cell_mask_for_frame(
                    time=time,
                    position=position
                )
            )
            cell_thresholding_image = cell_image

            extra_metrics.update({
                "cell_segmentation_source": "partaker_segmented_mask",
            })

        cell_mask = np.asarray(cell_mask, dtype=bool)

        if raw_intensity_mode:
            cell_association_metrics = (
                self.compute_raw_intensity_cell_metrics(
                    raw_eps_frame=image,
                    cell_mask=cell_mask,
                    cell_labels=cell_labels,
                )
            )
        else:
            cell_association_metrics = (
                self.compute_eps_cell_metrics_from_masks(
                    eps_mask=eps_mask,
                    cell_mask=cell_mask,
                    cell_labels=cell_labels,
                )
            )

        extra_metrics.update(cell_association_metrics)
        export_visuals = self.should_export_visuals_for_position(position)
        output_dir = self.get_eps_output_root()

        # -----------------------------
        # Per-frame visual outputs
        # -----------------------------
        if raw_intensity_mode:
            eps_mask = np.ones_like(cell_mask, dtype=bool)

        eps_mask_dir = output_dir / "eps_masks"
        eps_on_eps_dir = output_dir / "eps_on_eps"
        cells_on_cells_dir = output_dir / "cells_on_cells"
        eps_cell_overlap_dir = output_dir / "eps_cell_overlap"
        raw_signal_cell_dir = output_dir / "raw_signal_within_cells"

        for folder in [
            eps_mask_dir,
            eps_on_eps_dir,
            cells_on_cells_dir,
            eps_cell_overlap_dir,
            raw_signal_cell_dir,
        ]:
            folder.mkdir(parents=True, exist_ok=True)

        eps_mask_filename = f"pos{position}_t{time}_{channel}.jpg"
        filename_base = (
            f"pos{position}_t{time}_"
            f"epsC{channel}_cellC{cell_channel}"
        )

        if export_visuals:
            plt.imsave(
                eps_mask_dir / eps_mask_filename,
                eps_mask,
                cmap="gray"
            )

            eps_on_eps_path = (
                    eps_on_eps_dir
                    / f"{filename_base}_eps_on_eps.png"
            )
            self.save_eps_on_eps_overlay(
                raw_eps_frame=eps_display_frame,
                eps_mask=eps_mask,
                output_path=eps_on_eps_path
            )

            cells_on_cells_path = (
                    cells_on_cells_dir
                    / f"{filename_base}_cells_on_cells.png"
            )
            self.save_cells_on_cells_overlay(
                raw_cell_frame=cell_image,
                cell_mask=cell_mask,
                output_path=cells_on_cells_path
            )

            if not hasattr(self, "overlay_gif_rows"):
                self.overlay_gif_rows = []

            self.overlay_gif_rows.append({
                "time": int(time),
                "position": int(position),
                "eps_on_eps_path": str(eps_on_eps_path),
                "cells_on_cells_path": str(cells_on_cells_path),
            })

            self.save_eps_cell_overlap_overlay(
                raw_cell_frame=cell_image,
                cell_mask=cell_mask,
                eps_mask=eps_mask,
                output_path=(
                        eps_cell_overlap_dir
                        / f"{filename_base}_eps_cell_overlap.png"
                )
            )

            if raw_intensity_mode:
                self.save_raw_signal_within_cells_overlay(
                    raw_eps_frame=frame,
                    cell_mask=cell_mask,
                    output_path=(
                            raw_signal_cell_dir
                            / f"{filename_base}_raw_signal_in_cells.png"
                    ),
                )

        # -----------------------------
        # Heatmap values
        # -----------------------------
        self.append_intensity_heatmap_rows(
            source="eps",
            stage="raw" if raw_intensity_mode else "before",
            image=frame,
            time=time,
            position=position,
            channel=channel,
            mask=None,
        )

        if not raw_intensity_mode:
            self.append_intensity_heatmap_rows(
                source="eps",
                stage="after",
                image=smoothed,
                time=time,
                position=position,
                channel=channel,
                mask=eps_mask,
            )

        self.append_intensity_heatmap_rows(
            source="cell",
            stage="before",
            image=cell_image,
            time=time,
            position=position,
            channel=cell_channel,
            mask=None,
        )
        self.append_intensity_heatmap_rows(
            source="cell",
            stage="after",
            image=cell_thresholding_image,
            time=time,
            position=position,
            channel=cell_channel,
            mask=cell_mask,
        )

        # Binary-mask summary GIFs are intentionally skipped in raw mode.
        if not hasattr(self, "gif_rows"):
            self.gif_rows = []

        association_kind = (
            "raw_signal_within_cells_fraction"
            if raw_intensity_mode
            else "fraction_cells_encased"
        )
        association_value = float(
            cell_association_metrics.get(
                association_kind,
                0.0,
            )
        )

        gif_row = {
            "time": int(time),
            "time_hours": float(self.get_time_hours(time)),
            "position": int(position),
            "eps_channel": int(channel),
            "cell_channel": int(cell_channel),
            "association_kind": association_kind,
            "association_value": association_value,
        }

        if self.should_export_visuals_for_position(position):
            mask_cache_dir = output_dir / "_gif_mask_cache"
            mask_cache_dir.mkdir(parents=True, exist_ok=True)

            mask_cache_path = (
                    mask_cache_dir
                    / f"pos{position}_t{time}_masks.npz"
            )

            np.savez_compressed(
                mask_cache_path,
                mask_shape=np.asarray(
                    eps_mask.shape,
                    dtype=np.int32,
                ),
                eps_mask_packed=np.packbits(
                    np.asarray(eps_mask, dtype=bool).ravel()
                ),
                cell_mask_packed=np.packbits(
                    np.asarray(cell_mask, dtype=bool).ravel()
                ),
            )
            gif_row["mask_cache_path"] = str(mask_cache_path)

        self.gif_rows.append(gif_row)

        # -----------------------------
        # Scalar metrics
        # -----------------------------
        if raw_intensity_mode:
            mean_intensity = cell_association_metrics[
                "raw_mean_intensity_whole_frame"
            ]
            whole_frame_integrated = cell_association_metrics[
                "raw_integrated_intensity_whole_frame"
            ]
            cell_integrated = cell_association_metrics[
                "raw_integrated_intensity_within_cells"
            ]

            # Keep a stable canonical value for the legacy
            # ``integrated_intensity`` column. The display setting is not
            # consulted during processing; both raw alternatives are stored
            # separately and always exported.
            integrated_intensity = whole_frame_integrated
            integrated_metric_name = "whole_frame_integrated"

            finite_values = image[np.isfinite(image)]
            processing_min = (
                float(np.min(finite_values))
                if finite_values.size
                else 0.0
            )
            processing_max = (
                float(np.max(finite_values))
                if finite_values.size
                else 0.0
            )

            metrics = {
                "eps_processing_dtype": str(image.dtype),
                "eps_processing_min": processing_min,
                "eps_processing_max": processing_max,

                # No EPS mask exists in this mode.
                "occupancy_fraction": np.nan,
                "occupancy_percent": np.nan,
                "eps_area_pixels": 0,

                "mean_intensity": mean_intensity,
                "integrated_intensity": integrated_intensity,
                "integrated_intensity_metric": integrated_metric_name,
                "eps_mask_used": False,
                "analysis_area_pixels": int(image.size),
            }
        else:
            eps_pixels = int(np.count_nonzero(eps_mask))
            total_pixels = int(eps_mask.size)

            occupancy_fraction = (
                eps_pixels / total_pixels
                if total_pixels > 0
                else 0.0
            )
            occupancy_percent = occupancy_fraction * 100.0

            if eps_pixels > 0:
                mean_intensity = float(
                    np.mean(image[eps_mask], dtype=np.float64)
                )
                integrated_intensity = float(
                    np.sum(image[eps_mask], dtype=np.float64)
                )
            else:
                mean_intensity = 0.0
                integrated_intensity = 0.0

            metrics = {
                "eps_processing_dtype": str(
                    eps_processing_image.dtype
                ),
                "eps_processing_min": int(
                    eps_processing_image.min()
                ),
                "eps_processing_max": int(
                    eps_processing_image.max()
                ),

                "occupancy_fraction": occupancy_fraction,
                "occupancy_percent": occupancy_percent,
                "mean_intensity": mean_intensity,
                "integrated_intensity": integrated_intensity,
                "eps_area_pixels": eps_pixels,
                "eps_mask_used": True,
                "analysis_area_pixels": total_pixels,
            }

        metrics.update(extra_metrics)
        self.biofilm_metric_service.calculate_microcolony_eps_metrics(
            time=time,
            position=position,
            eps_channel=channel,
            analysis_method=str(
                metrics.get("analysis_method", self.get_analysis_method())
            ),
            raw_eps_frame=image,
            eps_mask=None if raw_intensity_mode else eps_mask,
        )
        return metrics

    def should_export_visuals_for_position(self, position: int) -> bool:
        if not self.EXPORT_VISUALS_ONLY_FOR_ONE_POSITION:
            return True
        return int(position) == int(self.EXPORT_VISUAL_POSITION)
