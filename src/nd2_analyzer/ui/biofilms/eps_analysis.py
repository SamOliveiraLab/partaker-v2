from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QSlider, QSpinBox, QComboBox,
                               QGroupBox, QListWidget, QProgressBar,
                               QCheckBox, QFrame, QTextEdit, QSplitter,
                               QFileDialog, QAbstractItemView, QMessageBox,
                               QDoubleSpinBox, QRadioButton, QButtonGroup)
from PySide6.QtCore import Qt, Signal, QThread, QObject
from PySide6.QtGui import QFont
import os
import math
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage import measure
import tifffile
import traceback
from pubsub import pub
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from nd2_analyzer.analysis.metrics_service import MetricsService
import polars as pl  # Import Polars
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.restoration import rolling_ball
from skimage.filters import threshold_otsu
from skimage.filters.rank import otsu
from skimage.morphology import disk, binary_closing, binary_dilation, binary_opening
from skimage.util import img_as_ubyte


class EPSAnalysisWidget(QWidget):
    """Widget for cube-based analysis of exported colony time series"""

    # Moreau mode uses the best compromise parameters from the Fig. 4b match.
    MOREAU_LOCAL_AREA_PERCENT = 60
    MOREAU_THRESHOLD_SCOPE = "cell_envelope"
    MOREAU_EPS_OPENING_RADIUS_PX = 0
    MOREAU_EPS_CLOSING_RADIUS_PX = 0
    MOREAU_EPS_DILATION_RADIUS_PX = 2
    MOREAU_SMOOTHING_SIGMA = 1.5

    def __init__(self, parent=None):
        super().__init__(parent)

        # State variables
        self.selected_colonies = []
        self.cube_size = 10
        self.analysis_results = {}
        self.base_folder = ""

        # Initial background filtered reference image
        self.reference_backgrounds = {}

        # Metrics Service
        #self.metrics_service = MetricsService()

        # Segmentation
        self.is_segmenting = False
        self.cancel_requested = False
        self.queue = []
        self.eps_results = []

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
        group = QGroupBox("1. Select Segmentation Channel")
        layout = QVBoxLayout(group)

        # Select channels for Segmentation
        selection_layout = QVBoxLayout()

        # Analysis Method Controls
        analysis_mode_group = QGroupBox("Analysis Method")
        analysis_mode_layout = QVBoxLayout()

        self.radio_partaker = QRadioButton("Partaker")
        self.radio_moreau = QRadioButton("Moreau")

        # Partaker is the default method.
        self.radio_partaker.setChecked(True)

        self.analysis_method_group = QButtonGroup(self)
        self.analysis_method_group.addButton(self.radio_partaker)
        self.analysis_method_group.addButton(self.radio_moreau)

        analysis_mode_layout.addWidget(self.radio_partaker)
        analysis_mode_layout.addWidget(self.radio_moreau)

        analysis_mode_group.setLayout(analysis_mode_layout)
        selection_layout.addWidget(analysis_mode_group)

        self.radio_partaker.toggled.connect(self.on_method_changed)
        self.radio_moreau.toggled.connect(self.on_method_changed)

        selection_layout.addWidget(QLabel("EPS Channel:"))
        self.eps_channel_combo = QComboBox()
        selection_layout.addWidget(self.eps_channel_combo)

        # Model selection
        selection_layout.addWidget(QLabel("Select Reference Value:"))
        self.ref_combo = QComboBox()
        selection_layout.addWidget(self.ref_combo)

        # Add image filtering options
        self.gaus_back_corr = QCheckBox("Gaussian Background Subtraction")
        self.gaus_back_corr.setChecked(False)
        self.close_dialate = QCheckBox("Morphologically Close & Dilate EPS Channel")
        self.close_dialate.setChecked(False)
        #self.center_distance_check = QCheckBox("Distance to Center")
        #self.center_distance_check.setChecked(True)
        #self.texture_check = QCheckBox("Local Texture")
        #self.texture_check.setChecked(True)
        for checkbox in [self.gaus_back_corr, self.close_dialate,]:
            selection_layout.addWidget(checkbox)


        layout.addLayout(selection_layout)

        return group

    def get_analysis_method(self):
        """Return selected EPS analysis method."""
        if hasattr(self, "radio_moreau") and self.radio_moreau.isChecked():
            return "Moreau"

        return "Partaker"

    def is_partaker_method(self):
        """Return True when the current implementation / segmented-cell method is selected."""
        return self.get_analysis_method() == "Partaker"

    def is_moreau_method(self):
        """Return True when the Fig. 4b best-compromise parameter method is selected."""
        return self.get_analysis_method() == "Moreau"

    def on_method_changed(self):
        """Update UI defaults when switching EPS analysis methods."""
        if self.is_moreau_method():
            # Moreau mode uses the best compromise EPS morphology setting:
            # EPS opening = 0 px, EPS closing = 0 px, EPS dilation = 2 px.
            self.close_dialate.setChecked(True)
        else:
            # Partaker is the current implementation and remains the default method.
            self.close_dialate.setChecked(False)

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
        self.start_segmentation_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 5px;")
        self.start_segmentation_btn.setEnabled(False)

        # Cancel button
        self.cancel_analysis_btn = QPushButton("Cancel")
        self.cancel_analysis_btn.clicked.connect(self.cancel_analysis)
        self.cancel_analysis_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 5px;")
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
        self.frame_interval_value.setValue(1.0)  # default = 1 hr
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
        self.metric_combo.addItem("Time-Series Line Plot")
        #self.metric_combo.addItem("NA")
        #self.metric_combo.addItem("NA")
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


    def get_selected_parameters(self):
        """Get list of selected parameters"""
        params = []
        if self.density_check.isChecked():
            params.append("Local Density")
        if self.edge_distance_check.isChecked():
            params.append("Distance to Edge")
        if self.center_distance_check.isChecked():
            params.append("Distance to Center")
        if self.texture_check.isChecked():
            params.append("Local Texture")
        if self.fluorescence_check.isChecked():
            params.append("Fluorescence Intensity")
        if self.roughness_check.isChecked():
            params.append("Surface Roughness")

        return params

    def validate_partaker_segmentation_available(self):
        """
        Check that segmented cell masks are available for Partaker mode.

        Moreau mode does not use the segmented-cell cache, so this check should
        only run when Partaker is selected.
        """
        try:
            from nd2_analyzer.data.image_data import ImageData

            image_data = ImageData.get_instance()
            if image_data is None or image_data.segmentation_cache is None:
                QMessageBox.warning(
                    self,
                    "Missing cell segmentation",
                    "Partaker mode requires segmented cell masks. Run cell segmentation first, or switch to Moreau mode."
                )
                return False

            segmented_storage = image_data.segmentation_cache
            model_name = image_data.segmentation_cache.model_name

            if not model_name:
                QMessageBox.warning(
                    self,
                    "Missing cell segmentation model",
                    "Partaker mode requires a selected cell segmentation model. Run cell segmentation first, or switch to Moreau mode."
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
                    "Partaker mode requires segmented cell masks. Run cell segmentation first, or switch to Moreau mode."
                )
                return False

            _mmap_array, index_set = cache.mmap_arrays_idx[model_name]

            if not index_set:
                QMessageBox.warning(
                    self,
                    "Missing segmented cell masks",
                    "Partaker mode requires segmented cell masks. Run cell segmentation first, or switch to Moreau mode."
                )
                return False

            return True

        except Exception as e:
            QMessageBox.warning(
                self,
                "Segmentation check failed",
                f"Could not verify segmented cell masks for Partaker mode:\n{e}"
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

    def generate_heatmap(self):
        """Generate heatmap visualization"""
        if not self.analysis_results:
            self.status_label.setText("No results to visualize.")
            return

        try:
            # Get selected parameter and time point
            selected_param = self.param_combo.currentText()
            selected_time_text = self.time_combo.currentText()

            if not selected_time_text:
                self.status_label.setText("Please select a time point.")
                return

            # Extract time point number
            time_point = selected_time_text.replace('T', '').lstrip('0') or '0'

            # Get first colony data (extend later for multiple colonies)
            first_colony = list(self.analysis_results.keys())[0]
            colony_data = self.analysis_results[first_colony]

            if time_point not in colony_data:
                self.status_label.setText(f"Time point {time_point} not found.")
                return

            time_data = colony_data[time_point]

            # Map parameter names to data keys
            param_map = {
                'Local Density': 'local_density',
                'Distance to Edge': 'distance_to_edge',
                'Distance to Center': 'distance_to_center',
                'Local Texture': 'local_thickness'
            }

            if selected_param not in param_map:
                self.status_label.setText("Parameter not available.")
                return

            data_key = param_map[selected_param]

            # Create simple heatmap
            import matplotlib.pyplot as plt
            import numpy as np

            positions = time_data['square_positions']
            values = time_data[data_key]

            # Filter out zero values if they're just placeholders
            filtered_data = [(pos, val) for pos, val in zip(positions, values) if val > 0]

            if not filtered_data:
                self.status_label.setText("No data to visualize for this parameter.")
                return

            positions, values = zip(*filtered_data)
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]

            # Create plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(x_coords, y_coords, c=values, cmap='viridis', s=20)
            plt.colorbar(scatter, label=selected_param)
            plt.title(f'{selected_param} - {first_colony} - {selected_time_text}')
            plt.xlabel('X Position (pixels)')
            plt.ylabel('Y Position (pixels)')
            plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates

            plt.tight_layout()
            plt.show()

            self.status_label.setText(f"Generated heatmap for {selected_param}")

        except Exception as e:
            self.status_label.setText(f"Visualization failed: {str(e)}")
            print(f"Heatmap error: {e}")

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

    def on_analysis_error(self, error_msg):
        """Handle analysis errors"""
        self.status_label.setText(f"Analysis failed: {error_msg}")
        self.console_area.append(f"ERROR: {error_msg}")
        self.cleanup_thread()

    def on_console_output(self, message):
        """Handle console output from worker"""
        self.console_area.append(message)
        self.console_area.ensureCursorVisible()

    def display_results(self):
        """Display analysis results in the text area"""
        if not self.analysis_results:
            return

        results_text = "Analysis Results Summary:\n\n"

        total_squares = 0
        total_timepoints = 0

        for colony_name, colony_data in self.analysis_results.items():
            results_text += f"{colony_name}:\n"
            results_text += f"  Time points analyzed: {len(colony_data)}\n"
            total_timepoints += len(colony_data)

            # Calculate average squares per timepoint
            squares_per_timepoint = []
            for timepoint_data in colony_data.values():
                squares_count = len(timepoint_data['square_positions'])
                squares_per_timepoint.append(squares_count)
                total_squares += squares_count

            if squares_per_timepoint:
                avg_squares = np.mean(squares_per_timepoint)
                results_text += f"  Average squares per timepoint: {avg_squares:.1f}\n"

                # Show sample parameter values from first timepoint
                first_timepoint = list(colony_data.keys())[0]
                first_data = colony_data[first_timepoint]

                if first_data['local_density'] and any(d > 0 for d in first_data['local_density']):
                    avg_density = np.mean([d for d in first_data['local_density'] if d > 0])
                    results_text += f"  Sample local density: {avg_density:.3f}\n"

                if first_data['distance_to_edge'] and any(d > 0 for d in first_data['distance_to_edge']):
                    avg_edge_dist = np.mean([d for d in first_data['distance_to_edge'] if d > 0])
                    results_text += f"  Sample distance to edge: {avg_edge_dist:.1f} pixels\n"

            results_text += "\n"

        results_text += f"Total Analysis Summary:\n"
        results_text += f"  Total squares analyzed: {total_squares}\n"
        results_text += f"  Total time points: {total_timepoints}\n"
        if len(self.analysis_results) > 0:
            results_text += f"  Average squares per colony: {total_squares / len(self.analysis_results):.1f}\n"

        self.results_area.setText(results_text)

    def segment_selected_channels(self):
        """Queue segmentation for selected channels, positions, and time range."""
        self.eps_results = []

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

        if self.is_partaker_method() and not self.validate_partaker_segmentation_available():
            self.start_segmentation_btn.setEnabled(True)
            self.cancel_analysis_btn.setEnabled(False)
            return

        eps_channel = self.eps_channel_combo.currentIndex()
        self.current_reference = self.ref_combo.currentText()
        self.current_analysis_method = self.get_analysis_method()

        # Skip focus loss frames
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

        frames_to_analyze = []

        for p in selected_positions:
            for t in range(t_start, t_end + 1):

                if t in focus_loss_skip:
                    continue

                frames_to_analyze.append(
                    (t, p, eps_channel)
                )

        self.queue = frames_to_analyze
        print(
            f"Queue length: {len(frames_to_analyze)}"
        )

        self.processed_frames = set()

        self.progress_bar.setMaximum(len(self.queue))
        self.progress_bar.setValue(0)

        self.is_segmenting = True
        self.cancel_requested = False

        self.start_segmentation_btn.setEnabled(False)
        self.cancel_analysis_btn.setEnabled(True)

        self.status_label.setText(
            f"Processing {len(self.queue)} frames..."
        )
        self.process_next_in_queue()



    def on_image_data_loaded(self, image_data):
        if image_data is None or image_data.data is None:
            return

        self.start_segmentation_btn.setEnabled(True)

         # Get dimensions from image_data
        shape = image_data.data.shape
        t_max = shape[0] - 1
        p_max = shape[1] - 1
        c_max = shape[2] - 1

        self.eps_channel_combo.clear()
        for c in range(c_max + 1):
            self.eps_channel_combo.addItem(f"Channel {c}")

        # Populate reference combo
        self.ref_combo.clear()
        #self.ref_combo.addItems([])

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

            self.request_timer.start(1)
            return

        result_row = {
            "time": time,
            "position": position,
            "channel": channel,
            "analysis_method": metrics.get(
                "analysis_method",
                self.get_analysis_method()
            ),
            "eps_mask":
                metrics["eps_mask"],
            "occupancy_fraction":
                metrics["occupancy_fraction"],
            "occupancy_percent":
                metrics["occupancy_percent"],
            "mean_intensity":
                metrics["mean_intensity"],
            "integrated_intensity":
                metrics["integrated_intensity"],
            "eps_area_pixels":
                metrics["eps_area_pixels"]
        }

        for optional_key in [
            "eps_threshold_used",
            "local_block_side_px",
            "fraction_vps_coverage",
            "areal_fraction_percent",
            "cell_area_pixels",
            "cell_envelope_pixels",
            "vps_colocalized_pixels",
        ]:
            if optional_key in metrics:
                result_row[optional_key] = metrics[optional_key]

        self.eps_results.append(result_row)

        completed = len(self.eps_results)

        self.progress_bar.setValue(
            completed
        )

        self.status_label.setText(
            f"Processing T={time} P={position}"
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
        """Handle completion of all segmentation tasks"""
        self.eps_df = pl.DataFrame(self.eps_results)
        self.is_segmenting = False
        self.start_segmentation_btn.setEnabled(True)
        self.cancel_analysis_btn.setEnabled(False)
        self.status_label.setText(
            f"EPS analysis complete "
            f"({len(self.eps_results)} frames)"
        )

    def compute_avg_sd_stats(self,df: pl.DataFrame):
        """
        Compute mean ± SD EPS metrics
        grouped by timepoint.
        """

        stats = (
            df
            .group_by("time")
            .agg([
                # -------------------------
                # EPS Area Fraction
                # -------------------------

                pl.col("occupancy_percent")
                .mean()
                .alias("mean_occupancy"),

                pl.col("occupancy_percent")
                .std()
                .alias("std_occupancy"),

                # -------------------------
                # Mean EPS Intensity
                # -------------------------

                pl.col("mean_intensity")
                .mean()
                .alias("mean_intensity"),

                pl.col("mean_intensity")
                .std()
                .alias("std_intensity"),

                # -------------------------
                # Integrated EPS Intensity
                # -------------------------

                pl.col("integrated_intensity")
                .mean()
                .alias("mean_integrated"),

                pl.col("integrated_intensity")
                .std()
                .alias("std_integrated"),

                # -------------------------
                # EPS Area (pixels)
                # -------------------------

                pl.col("eps_area_pixels")
                .mean()
                .alias("mean_area_pixels"),

                pl.col("eps_area_pixels")
                .std()
                .alias("std_area_pixels"),
            ])
            .sort("time")
        )

        return stats


    def on_plot_avg_sd(self):
        self.population_figure.clear()

        if not hasattr(self, "eps_df"):
            QMessageBox.warning(self,"No data","Run EPS analysis first.")
            return

        df = self.eps_df

        # Compute stats
        eps_channel = int(self.eps_channel_combo.currentIndex())

        df_eps = df.filter(pl.col("channel") == eps_channel)

        print("ALL TIMES:", sorted(df["time"].unique()))
        print("EPS TIMES:", sorted(df_eps["time"].unique()))

        stats_eps = self.compute_avg_sd_stats(df_eps)
        print("\n=== STATS ===")
        print(stats_eps)

        print("\n=== EPS DF CHECK ===")
        print(
            df_eps.select(
                [
                    "time",
                    "occupancy_percent",
                    "mean_intensity",
                    "integrated_intensity",
                    "eps_area_pixels"
                ]
            ).head(20)
        )

        print("\n=== GROUPED STATS INPUT ===")
        print(
            df_eps.group_by("time").agg(
                pl.col("occupancy_percent").count()
            )
        )


        # Custom time Mapping
        unique_times = sorted(stats_eps["time"].to_list())
        real_times = [0, 24, 48]
        if len(unique_times) != len(real_times):
            QMessageBox.warning(
                self,
                "Time mapping error",
                f"Found {len(unique_times)} timepoints, but real_times has {len(real_times)} values."
            )
            return
        time_map = dict(zip(unique_times, real_times))
        time_eps = np.array([
            time_map[t] for t in stats_eps["time"].to_list()
        ])
        """
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
        time_eps = np.array(stats_eps["time"]) * hours_per_frame
        """

        fig = self.population_figure
        axs = fig.subplots(2, 1)

        # --- 1. EPS ---
        mean_int = np.array(stats_eps["mean_integrated"])
        std_int = np.array(stats_eps["std_integrated"])

        axs[0].plot(
            time_eps,
            mean_int,
            '-o',
            color='green',
            linewidth=2
        )

        axs[0].fill_between(
            time_eps,
            mean_int - std_int,
            mean_int + std_int,
            color='green',
            alpha=0.25
        )
        axs[0].set_title("EPS Fluorescence")
        axs[0].set_ylabel("Integrated Intensity")
        axs[0].set_xlabel("Time (hours)")


        # --- 3. MORPHOLOGY (AREA) ---
        mean_occ = np.array(stats_eps["mean_occupancy"])
        std_occ = np.array(stats_eps["std_occupancy"])
        axs[1].plot(
            time_eps,
            mean_occ,
            '-o',
            color='goldenrod',
            linewidth=2
        )

        axs[1].fill_between(
            time_eps,
            mean_occ - std_occ,
            mean_occ + std_occ,
            color='goldenrod',
            alpha=0.25
        )
        axs[1].set_title("EPS Area Fraction")
        axs[1].set_ylabel("Area Fraction (%)")
        axs[1].set_xlabel("Time (hours)")

        fig.subplots_adjust(
            top=0.92,
            bottom=0.08,
            left=0.12,
            right=0.95,
            hspace=0.4
        )

        self.population_canvas.draw()

        # Add to output folder: analysis_results
        from pathlib import Path

        output_dir = Path("analysis_results")
        output_dir.mkdir(exist_ok=True)

        fig.savefig(
            output_dir / "eps_mean_sd_plot.png",
            dpi=300,
            bbox_inches="tight"
        )

        if self.is_partaker_method():
            self.eps_cell_analysis()
            self.on_plot_fraction_cells()
        else:
            self.moreau_cell_envelope_analysis()
            self.on_plot_fraction_cells()

    def fixed_or_otsu_threshold(
            self,
            values: np.ndarray,
            fixed_threshold=None,
    ) -> float:
        values = values[np.isfinite(values)]

        if values.size == 0:
            return 0.0

        if fixed_threshold is not None:
            return float(fixed_threshold)

        if float(values.max()) == float(values.min()):
            return float(values.max())

        return float(threshold_otsu(values))

    def get_pixel_size_um(self, image_data):
        """Get pixel size in um from ImageData, with a safe fallback."""
        voxel_size = getattr(image_data, "voxel_size", None)

        if hasattr(voxel_size, "x"):
            return float(voxel_size.x)

        if isinstance(voxel_size, (int, float)):
            return float(voxel_size)

        return 1.0

    def make_moreau_cell_masks(
            self,
            cell_frame: np.ndarray,
            cell_closing_diameter_um: float = 0.65,
            envelope_dilation_diameter_um: float = 0.65,
    ):
        """
        Return two masks:

        1. cell_area_mask:
           fixed-threshold cell area, morphologically closed.
           This is used for areal fraction rho.

        2. cell_envelope_mask:
           cell_area_mask dilated with a circular kernel matching the VPS envelope.
           This is used as the denominator for the encased/VPS colocalization metric.
        """

        from nd2_analyzer.data.image_data import ImageData

        image_data = ImageData.get_instance()
        pixel_size_um = self.get_pixel_size_um(image_data)

        cell_frame = cell_frame.astype(float)

        if float(cell_frame.max()) == float(cell_frame.min()):
            cell_threshold = float(cell_frame.max())
        else:
            cell_threshold = float(threshold_otsu(cell_frame))

        cell_area_mask = cell_frame > cell_threshold

        closing_radius_px = max(
            1,
            int(round((cell_closing_diameter_um / 2.0) / pixel_size_um)),
        )
        closing_kernel = disk(closing_radius_px)

        cell_area_mask = binary_closing(
            cell_area_mask,
            footprint=closing_kernel,
        )

        dilation_radius_px = max(
            1,
            int(round((envelope_dilation_diameter_um / 2.0) / pixel_size_um)),
        )
        dilation_kernel = disk(dilation_radius_px)

        cell_envelope_mask = binary_dilation(
            cell_area_mask,
            footprint=dilation_kernel,
        )

        return cell_area_mask, cell_envelope_mask, cell_threshold

    def blockwise_local_otsu_eps_mask(
            self,
            eps_image: np.ndarray,
            threshold_scope_mask: np.ndarray,
            local_area_percent: int,
            fixed_eps_threshold=None,
    ):
        """
        Efficient local/adaptive Otsu approximation.

        The image is split into non-overlapping square blocks. Each block gets its own
        Otsu threshold computed from EPS pixels inside threshold_scope_mask.

        local_area_percent controls the block area relative to the whole frame:
            0   -> smallest block, 3 x 3 pixels
            10  -> each local block covers about 10% of the frame area
            100 -> one block covering the full frame, effectively global Otsu

        This is intentionally much faster than rank.otsu with giant sliding disks,
        which becomes painfully slow for 2048 x 2048 images at 10-100% coverage.
        """

        height, width = eps_image.shape
        frame_area = height * width

        if local_area_percent <= 0:
            block_side = 3
        elif local_area_percent >= 100:
            block_side = max(height, width)
        else:
            block_area = frame_area * (local_area_percent / 100.0)
            block_side = int(round(math.sqrt(block_area)))
            block_side = max(3, block_side)

        # Make block side odd so the label is stable/readable.
        if block_side % 2 == 0:
            block_side += 1

        global_fallback_threshold = self.fixed_or_otsu_threshold(
            eps_image[threshold_scope_mask],
            fixed_threshold=fixed_eps_threshold,
        )

        eps_mask = np.zeros_like(threshold_scope_mask, dtype=bool)
        used_thresholds = []

        for y0 in range(0, height, block_side):
            y1 = min(y0 + block_side, height)

            for x0 in range(0, width, block_side):
                x1 = min(x0 + block_side, width)

                block_image = eps_image[y0:y1, x0:x1]
                block_scope = threshold_scope_mask[y0:y1, x0:x1]
                values = block_image[block_scope]

                if values.size < 16 or float(values.max()) == float(values.min()):
                    threshold = global_fallback_threshold
                else:
                    threshold = self.fixed_or_otsu_threshold(
                        values,
                        fixed_threshold=fixed_eps_threshold,
                    )

                used_thresholds.append(threshold)
                eps_mask[y0:y1, x0:x1] = (block_image > threshold) & block_scope

        mean_threshold = float(np.mean(used_thresholds)) if used_thresholds else 0.0
        return eps_mask, mean_threshold, block_side

    def apply_eps_channel_morphology(
            self,
            eps_mask: np.ndarray,
            opening_radius_px: int = 0,
            closing_radius_px: int = 0,
            dilation_radius_px: int = 0,
    ):
        """
        Optional VPS/EPS morphology after binarization.

        Order is opening -> closing -> dilation.
        Radius 0 means that operation is skipped.
        """

        morphed = eps_mask.copy()

        if opening_radius_px > 0:
            morphed = binary_opening(
                morphed,
                footprint=disk(opening_radius_px),
            )

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

    def segment_eps_partaker(
            self,
            gaussian_corrected: np.ndarray,
            image: np.ndarray,
            smoothing_sigma: float = 1.5,
            rolling_ball_fraction: int = 4,
    ):
        # -----------------------------
        # 3. Light smoothing before segmentation
        # -----------------------------

        smoothed = gaussian_filter(
            gaussian_corrected,
            sigma=smoothing_sigma
        )

        # -----------------------------
        # 4. Normalize for Local Otsu
        # -----------------------------
        print(f"Starting Local OTSU for {self}")

        if smoothed.max() > 0:
            smoothed_8bit = img_as_ubyte(
                smoothed / smoothed.max()
            )
        else:
            smoothed_8bit = np.zeros_like(
                smoothed,
                dtype=np.uint8
            )

        # -----------------------------
        # 5. Local Otsu segmentation
        # -----------------------------
        image_height, image_width = image.shape[:2]
        local_otsu_radius = (
                min(image_height, image_width)
                // rolling_ball_fraction
        )

        local_threshold = otsu(
            smoothed_8bit,
            footprint=disk(local_otsu_radius)
        )

        eps_mask = smoothed_8bit > local_threshold

        """
        if smoothed.max() > 0:
            smoothed_8bit = img_as_ubyte(
                smoothed / smoothed.max()
            )
        else:
            smoothed_8bit = np.zeros_like(
                smoothed,
                dtype=np.uint8
            )

        # -----------------------------
        # 5. Global Otsu segmentation
        # -----------------------------
        print(f"Starting Global OTSU for {self}")
        from skimage.filters import threshold_otsu

        global_threshold = threshold_otsu(
            smoothed_8bit
        )

        eps_mask = smoothed_8bit > global_threshold

        """

        """# -----------------------------
        # 5. Morphological Dilation & Closing
        # -----------------------------

        from nd2_analyzer.data.image_data import ImageData

        image_data = ImageData.get_instance()
        pixel_size_um = image_data.voxel_size.x

        kernel_radius_px = max(1, int(round((0.65 / 2) / pixel_size_um)))
        kernel = disk(kernel_radius_px)

        eps_mask = binary_closing(eps_mask, footprint=kernel)
        eps_mask = binary_dilation(eps_mask, footprint=kernel)"""

        if self.close_dialate.isChecked():
            # -----------------------------
            # 5. Morphological Dilation & Closing
            # -----------------------------

            from nd2_analyzer.data.image_data import ImageData

            image_data = ImageData.get_instance()
            pixel_size_um = self.get_pixel_size_um(image_data)

            kernel_radius_px = max(1, int(round((0.65 / 2) / pixel_size_um)))
            kernel = disk(kernel_radius_px)

            eps_mask = binary_closing(eps_mask, footprint=kernel)
            eps_mask = binary_dilation(eps_mask, footprint=kernel)

        return smoothed, eps_mask, {
            "analysis_method": "Partaker",
            "eps_threshold_used": float(np.mean(local_threshold)),
            "local_block_side_px": None,
        }

    def segment_eps_moreau(
            self,
            gaussian_corrected: np.ndarray,
            image: np.ndarray,
            time: int,
            position: int,
    ):
        """
        Moreau mode:
            Local area = 60%
            Threshold scope = cell envelope
            EPS opening = 0 px
            EPS closing = 0 px
            EPS dilation = 2 px
            Smoothing sigma = 1.5
        """

        # -----------------------------
        # 3. Light smoothing before segmentation
        # -----------------------------

        smoothed = gaussian_filter(
            gaussian_corrected,
            sigma=self.MOREAU_SMOOTHING_SIGMA
        )

        # -----------------------------
        # 4. Normalize for Local Otsu
        # -----------------------------
        # Moreau mode keeps the smoothed image in float units, because the
        # blockwise local Otsu threshold is computed directly from EPS values
        # inside the cell-envelope mask.

        # -----------------------------
        # 5. Local Otsu segmentation
        # -----------------------------
        print(f"Starting Moreau Local OTSU for {self}")

        from nd2_analyzer.data.image_data import ImageData

        image_data = ImageData.get_instance()

        # channel 0 = bacteria / phase image
        cell_image = image_data.get(
            time,
            position,
            0
        )

        cell_area_mask, cell_envelope_mask, cell_threshold = self.make_moreau_cell_masks(
            cell_frame=cell_image
        )

        if self.MOREAU_THRESHOLD_SCOPE == "cell_envelope":
            threshold_scope_mask = cell_envelope_mask
        else:
            threshold_scope_mask = np.ones_like(cell_envelope_mask, dtype=bool)

        eps_mask, eps_threshold, local_block_side_px = self.blockwise_local_otsu_eps_mask(
            eps_image=smoothed,
            threshold_scope_mask=threshold_scope_mask,
            local_area_percent=self.MOREAU_LOCAL_AREA_PERCENT,
        )

        if self.close_dialate.isChecked():
            eps_mask = self.apply_eps_channel_morphology(
                eps_mask=eps_mask,
                opening_radius_px=self.MOREAU_EPS_OPENING_RADIUS_PX,
                closing_radius_px=self.MOREAU_EPS_CLOSING_RADIUS_PX,
                dilation_radius_px=self.MOREAU_EPS_DILATION_RADIUS_PX,
            )

        # The paper-style numerator is only the VPS-positive region inside the
        # dilated cell-envelope mask, even if EPS morphology expands outside it.
        colocalized_mask = eps_mask & cell_envelope_mask
        eps_mask = colocalized_mask

        cell_area_pixels = int(np.count_nonzero(cell_area_mask))
        cell_envelope_pixels = int(np.count_nonzero(cell_envelope_mask))
        vps_colocalized_pixels = int(np.count_nonzero(colocalized_mask))
        total_pixels = int(image.size)

        areal_fraction_percent = (
            100.0 * cell_area_pixels / total_pixels
            if total_pixels > 0
            else 0.0
        )

        fraction_vps_coverage = (
            100.0 * vps_colocalized_pixels / cell_envelope_pixels
            if cell_envelope_pixels > 0
            else 0.0
        )

        print(
            f"P={position} T={time} | "
            f"mode=moreau | "
            f"local_area={self.MOREAU_LOCAL_AREA_PERCENT} | "
            f"scope={self.MOREAU_THRESHOLD_SCOPE} | "
            f"thr={eps_threshold:.3f} | "
            f"rho={areal_fraction_percent:.2f}% | "
            f"encased={fraction_vps_coverage:.2f}% | "
            f"cell_area_px={cell_area_pixels:,} | "
            f"cell_env_px={cell_envelope_pixels:,} | "
            f"vps_overlap_px={vps_colocalized_pixels:,}"
        )

        return smoothed, eps_mask, {
            "analysis_method": "Moreau",
            "eps_threshold_used": float(eps_threshold),
            "local_block_side_px": local_block_side_px,
            "fraction_vps_coverage": fraction_vps_coverage,
            "areal_fraction_percent": areal_fraction_percent,
            "cell_area_pixels": cell_area_pixels,
            "cell_envelope_pixels": cell_envelope_pixels,
            "vps_colocalized_pixels": vps_colocalized_pixels,
            "cell_threshold_used": float(cell_threshold),
        }

    def process_eps_frame(self,
            frame: np.ndarray,
            time: int,
            position: int,
            channel: int,
            background_sigma: float = 200,
            smoothing_sigma: float = 1.5,
            rolling_ball_fraction: int = 4,
    ):
        """
        Process one raw EPS microscopy frame.

        Returns:
            dict with corrected image, mask, and EPS metrics.
        """

        # -----------------------------
        # Validate / prepare image
        # -----------------------------

        if frame is None:
            raise ValueError("EPS frame is None.")

        image = frame.astype(float)
        print(f"Original shape: {image.shape}")


        # -----------------------------
        # 1. Large Gaussian background correction
        # -----------------------------
        if self.gaus_back_corr.isChecked():
            print(f"Starting Gaussian background correction for {self}")

            if time == 0:
                gaussian_background = gaussian_filter(image, sigma=background_sigma)
                gaussian_corrected = image - gaussian_background
            else:
                gaussian_corrected = image.copy()

            gaussian_corrected = np.clip(
                gaussian_corrected,
                0,
                None
            )
        else:
            gaussian_corrected = image.copy()
        gaussian_corrected = np.clip(gaussian_corrected, 0, None)

        """
        # -----------------------------
        # 2. Rolling ball background correction
        # -----------------------------
        print(f"Starting Rolling ball correction for {frame}")
        rolling_background = rolling_ball(
            gaussian_corrected,
            radius=rolling_ball_radius
        )

        rolling_corrected = gaussian_corrected - rolling_background

        rolling_corrected = np.clip(
            rolling_corrected,
            0,
            None
        )"""

        if self.is_moreau_method():
            smoothed, eps_mask, extra_metrics = self.segment_eps_moreau(
                gaussian_corrected=gaussian_corrected,
                image=image,
                time=time,
                position=position,
            )
        else:
            smoothed, eps_mask, extra_metrics = self.segment_eps_partaker(
                gaussian_corrected=gaussian_corrected,
                image=image,
                smoothing_sigma=smoothing_sigma,
                rolling_ball_fraction=rolling_ball_fraction,
            )

        # Generate results
        from pathlib import Path
        import matplotlib.pyplot as plt

        output_dir = Path("analysis_results")
        output_dir.mkdir(exist_ok=True)

        # ----------------------------------
        # Export EPS-on-cells overlay
        # ----------------------------------

        from nd2_analyzer.data.image_data import ImageData

        image_data = ImageData.get_instance()

        # channel 0 = bacteria / phase image
        cell_image = image_data.get(
            time,
            position,
            0
        )

        overlay_dir = output_dir / "eps_on_cells"
        overlay_dir.mkdir(exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.imshow(
            cell_image,
            cmap="gray"
        )

        ax.contour(
            eps_mask,
            levels=[0.5],
            colors="red",
            linewidths=1
        )

        ax.axis("off")

        fig.savefig(
            overlay_dir /
            f"pos{position}_t{time}_eps_on_cells.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close(fig)

        # ----------------------------------
        # Export EPS-on-EPS overlay
        # ----------------------------------

        overlay_dir = output_dir / "eps_on_eps"
        overlay_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        fig, ax = plt.subplots(figsize=(8, 8))

        # Raw EPS channel image
        ax.imshow(
            frame,
            cmap="gray"
        )

        # Fill in
        ax.contourf(
            eps_mask,
            levels=[0.5, 1],
            colors=["lime"],
            alpha=0.10
        )

        # EPS mask outline
        ax.contour(
            eps_mask,
            levels=[0.5],
            colors="green",
            linewidths=1
        )

        ax.axis("off")

        fig.savefig(
            overlay_dir /
            f"pos{position}_t{time}_eps_overlay.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close(fig)

        # ----------------------------------
        # Export Otsu results
        # ----------------------------------

        output_filename = (
            f"pos{position}_t{time}_{channel}.jpg"
        )

        otsu_dir = output_dir / "otsu_thresh"

        otsu_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.imsave(
            otsu_dir / output_filename,
            eps_mask,
            cmap="gray"
        )

        # -----------------------------
        # 6. Metrics
        # -----------------------------

        eps_pixels = np.count_nonzero(eps_mask)
        total_pixels = eps_mask.size

        occupancy_fraction = eps_pixels / total_pixels
        occupancy_percent = occupancy_fraction * 100

        if eps_pixels > 0:
            #mean_intensity = gaussian_corrected[eps_mask].mean()
            mean_intensity = image[eps_mask].mean()
            #integrated_intensity = gaussian_corrected[eps_mask].sum()
            """
            eps_signal = gaussian_corrected[eps_mask].sum()

            frame_signal = gaussian_corrected.sum()
            # relative signal
            integrated_intensity = eps_signal / frame_signal
            """
            #eps_signal = gaussian_corrected[eps_mask].sum()
            eps_signal = image[eps_mask].sum()

            # normalized
            integrated_intensity = (eps_signal / image.size)

        else:
            mean_intensity = 0.0
            integrated_intensity = 0.0

        metrics = {
            "smoothed": smoothed,
            "eps_mask": eps_mask,

            "occupancy_fraction": occupancy_fraction,
            "occupancy_percent": occupancy_percent,
            "mean_intensity": mean_intensity,
            "integrated_intensity": integrated_intensity,
            "eps_area_pixels": eps_pixels,
        }

        metrics.update(extra_metrics)
        return metrics

    def analyze_eps_timeseries(self, frames):
        results = []

        for t, frame in enumerate(frames):
            metrics = self.process_eps_frame(frame)

            results.append({
                "time": t,
                "occupancy_fraction":
                    metrics["occupancy_fraction"],
                "mean_intensity":
                    metrics["mean_intensity"],
                "integrated_intensity":
                    metrics["integrated_intensity"]
            })

        return pl.DataFrame(results)

    def eps_cell_analysis(self, overlap_threshold=0.10):
        """
        Compare segmented cell masks against segmented EPS masks.

        Output:
            pl.DataFrame with one row per frame:
            time, position, channel, total_cells, encased_cells,
            fraction_cells_encased
        """


        from nd2_analyzer.data.image_data import ImageData

        if not self.is_partaker_method():
            return self.moreau_cell_envelope_analysis()

        image_data = ImageData.get_instance()
        segmented_storage = image_data.segmentation_cache
        model_name = image_data.segmentation_cache.model_name
        cache = segmented_storage.with_model(model_name)
        mmap_array, index_set = cache.mmap_arrays_idx[model_name]

        segmented_masks = {}

        for idx in index_set:
            if len(idx) == 3:
                t, p, c = idx
            else:
                t, p = idx
                c = image_data.channel_n

            segmented = cache[(t, p, c, model_name)]
            segmented_masks[(t, p, c)] = segmented

        print(f"Loaded {len(segmented_masks)} segmented cell masks ")

        segmented_masks_by_position_time = {}
        for (t, p, c), segmented in segmented_masks.items():
            key = (
                t,
                p,
            )
            if key not in segmented_masks_by_position_time:
                segmented_masks_by_position_time[key] = segmented

        if not hasattr(self, "eps_results") or len(self.eps_results) == 0:
            QMessageBox.warning(self,"No EPS masks","Run EPS segmentation first.")
            return None

        eps_masks = {}
        for row in self.eps_results:
            key = (
                row["time"],
                row["position"],
            )

            if "eps_mask" not in row:
                QMessageBox.warning(self,"Missing EPS masks","EPS masks were not stored. Re-run EPS segmentation "
                    "after adding eps_mask to eps_results.")
                return None
            eps_masks[key] = row["eps_mask"]

        missing_segmented_frames = [
            key
            for key in eps_masks
            if key not in segmented_masks_by_position_time
        ]

        if missing_segmented_frames:
            first_missing = missing_segmented_frames[0]
            QMessageBox.warning(self,"Frame mismatch",
                f"Missing matching segmented cell mask for T={first_missing[0]}, P={first_missing[1]}.")
            return None

        fraction_results = []

        for lookup_key, eps_mask in eps_masks.items():
            t, p = lookup_key
            cell_labels = segmented_masks_by_position_time[lookup_key]
            c = 0



            from skimage.morphology import (binary_closing, binary_dilation, disk)
            from nd2_analyzer.data.image_data import ImageData

            image_data = ImageData.get_instance()
            voxel_size = image_data.voxel_size
            pixel_size_um = self.get_pixel_size_um(image_data)

            radius_um = 0.65 / 2

            radius_px = max(1, int(round(radius_um / pixel_size_um)))
            kernel = disk(radius_px)

            cell_ids = np.unique(cell_labels)
            cell_ids = cell_ids[cell_ids > 0]
            total_cells = len(cell_ids)
            encased_cells = 0

            for cell_id in cell_ids:
                single_cell_mask = cell_labels == cell_id
                cell_pixels = np.count_nonzero(single_cell_mask)

                if cell_pixels == 0:
                    continue

                eps_overlap_pixels = np.count_nonzero(eps_mask & single_cell_mask)
                overlap_fraction = eps_overlap_pixels / cell_pixels

                if overlap_fraction >= overlap_threshold:
                    encased_cells += 1

            # ===================================
            # PAPER-STYLE VPS COVERAGE METRIC
            # ===================================
            # Morphological Dilation & Closing
            all_cells_mask = cell_labels > 0

            mask_pixels = np.count_nonzero(all_cells_mask)

            if mask_pixels > 0:
                fraction_vps_coverage = (np.count_nonzero(eps_mask & all_cells_mask)
                                                / mask_pixels) * 100
            else:
                fraction_vps_coverage = 0.0

            print(
                f"T={t} P={p} | "
                f"VPS coverage={fraction_vps_coverage:.2f}% | "
                f"Mask pixels={mask_pixels:,} | "
                f"EPS pixels in mask={np.count_nonzero(eps_mask & all_cells_mask):,}"
            )

            if total_cells > 0:
                fraction_cells_encased = (encased_cells / total_cells) * 100
            else:
                fraction_cells_encased = 0.0


            fraction_results.append({
                "time": t,
                "position": p,
                "channel": c,
                "total_cells": total_cells,
                "encased_cells": encased_cells,
                "fraction_cells_encased": fraction_cells_encased,
                "fraction_vps_coverage": fraction_vps_coverage,
            })

        fraction_df = pl.DataFrame(fraction_results)
        print("\n=== EPS-CELL FRACTION RESULTS ===")
        print(fraction_df)

        fraction_df = pl.DataFrame(fraction_results)

        self.fraction_cells_df = fraction_df

        print("\n=== EPS-CELL FRACTION RESULTS ===")
        print(fraction_df)

        return fraction_df

    def moreau_cell_envelope_analysis(self):
        """
        Build the Moreau paper-style fraction table without segmented cell masks.

        Moreau mode uses the thresholded cell-envelope mask produced inside
        process_eps_frame, so it does not need ImageData.segmentation_cache.
        """

        if not hasattr(self, "eps_results") or len(self.eps_results) == 0:
            QMessageBox.warning(self,"No EPS masks","Run EPS segmentation first.")
            return None

        fraction_results = []

        for row in self.eps_results:
            fraction_results.append({
                "time": row["time"],
                "position": row["position"],
                "channel": row["channel"],
                "total_cells": 0,
                "encased_cells": 0,
                "fraction_cells_encased": row.get(
                    "fraction_vps_coverage",
                    row.get("occupancy_percent", 0.0)
                ),
                "fraction_vps_coverage": row.get(
                    "fraction_vps_coverage",
                    row.get("occupancy_percent", 0.0)
                ),
                "areal_fraction_percent": row.get(
                    "areal_fraction_percent",
                    0.0
                ),
            })

        fraction_df = pl.DataFrame(fraction_results)

        self.fraction_cells_df = fraction_df

        print("\n=== MOREAU EPS-CELL FRACTION RESULTS ===")
        print(fraction_df)

        return fraction_df

    def on_plot_fraction_cells(self):
        """"Plot for Moreau fig 4. comparison """
        if not hasattr(self, "fraction_cells_df"):
            QMessageBox.warning(self,"No Data","Run EPS-cell analysis first.")
            return

        df = self.fraction_cells_df

        if self.is_moreau_method():
            fraction_column = "fraction_vps_coverage"
            plot_title = "Fraction of Cell-Envelope Mask Encased by EPS"
            y_label = "Cell-Envelope EPS Coverage (%)"
        else:
            fraction_column = "fraction_cells_encased"
            plot_title = "Fraction of Cells Encased by EPS"
            y_label = "Cells Encased (%)"

        stats = (df.group_by("time").agg([
            pl.col(fraction_column).mean().alias("mean_fraction"),

            pl.col(fraction_column).std().alias("std_fraction")
        ]).fill_null(0).sort("time")
                 )



        interval_value = self.frame_interval_value.value()
        interval_unit = self.time_unit_combo.currentText()

        hours_conversion = {
            "ms": 1 / 1000 / 3600,
            "sec": 1 / 3600,
            "min": 1 / 60,
            "hr": 1,
            "day": 24
        }

        hours_per_frame = (interval_value * hours_conversion[interval_unit])
        time_hours = (np.array(stats["time"]) * hours_per_frame)

        mean_fraction = np.array(stats["mean_fraction"])
        std_fraction = np.array(stats["std_fraction"])

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        ax.plot(
            time_hours,
            mean_fraction,
            "-o",
            linewidth=2,
            color="black"
        )

        ax.fill_between(
            time_hours,
            mean_fraction - std_fraction,
            mean_fraction + std_fraction,
            color="gray",
            alpha=0.25
        )

        ax.set_title(plot_title)
        ax.set_ylabel(y_label)
        ax.set_xlabel("Time (hours)")

        output_dir = Path("analysis_results")

        output_dir.mkdir(exist_ok=True)

        fig.savefig(output_dir /
            "fraction_cells_plot.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)
        print("Saved fraction_cells_plot.png")