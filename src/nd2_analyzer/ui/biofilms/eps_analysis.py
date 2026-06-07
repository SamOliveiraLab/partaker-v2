from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QSlider, QSpinBox, QComboBox,
                               QGroupBox, QListWidget, QProgressBar,
                               QCheckBox, QFrame, QTextEdit, QSplitter,
                               QFileDialog, QAbstractItemView, QMessageBox, QDoubleSpinBox)
from PySide6.QtCore import Qt, Signal, QThread, QObject
from PySide6.QtGui import QFont
import os
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
from skimage.filters.rank import otsu
from skimage.morphology import disk
from skimage.util import img_as_ubyte


class EPSAnalysisWidget(QWidget):
    """Widget for cube-based analysis of exported colony time series"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # State variables
        self.selected_colonies = []
        self.cube_size = 10
        self.analysis_results = {}
        self.base_folder = ""

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
        selection_layout.addWidget(QLabel("Cluster Channel:"))
        self.cluster_channel_combo = QComboBox()
        selection_layout.addWidget(self.cluster_channel_combo)
        selection_layout.addWidget(QLabel("EPS Channel:"))
        self.eps_channel_combo = QComboBox()
        selection_layout.addWidget(self.eps_channel_combo)

        # Model selection
        selection_layout.addWidget(QLabel("Segmentation Model"))
        self.model_combo = QComboBox()
        selection_layout.addWidget(self.model_combo)


        layout.addLayout(selection_layout)

        return group

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
        self.start_segmentation_btn = QPushButton("Segment Images")
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

        """# Console output area
        console_label = QLabel("Console Output:")
        console_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(console_label)

        self.console_area = QTextEdit()
        self.console_area.setMaximumHeight(200)
        self.console_area.setStyleSheet(
            "background-color: #2b2b2b; color: #ffffff; font-family: 'Courier New', monospace;")
        layout.addWidget(self.console_area)

        # Results display area
        results_label = QLabel("Analysis Results:")
        results_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(results_label)

        self.results_area = QTextEdit()
        self.results_area.setPlaceholderText("Analysis results will appear here...")
        self.results_area.setMaximumHeight(200)
        layout.addWidget(self.results_area)"""

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

        #cluster_channel = self.cluster_channel_combo.currentIndex()
        eps_channel = self.eps_channel_combo.currentIndex()

        #selected_channels = sorted(set([cluster_channel, eps_channel]))
        selected_channels = sorted(set([eps_channel]))

        self.current_model = self.model_combo.currentText()

        #from nd2_analyzer.analysis.metrics_service import MetricsService
        #metrics_service = MetricsService()

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
            f"Segmenting {len(self.queue)} frames..."
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

        #self.cluster_channel_combo.clear()
        self.eps_channel_combo.clear()
        for c in range(c_max + 1):
            #self.cluster_channel_combo.addItem(f"Channel {c}")
            self.eps_channel_combo.addItem(f"Channel {c}")

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
            metrics = self.process_eps_frame(raw_img)

        except Exception as e:
            print(
                f"Failed frame "
                f"T={time} P={position}: {e}"
            )

            self.request_timer.start(1)
            return

        self.eps_results.append({
            "time": time,
            "position": position,
            "channel": channel,
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
        })

        completed = len(self.eps_results)

        self.progress_bar.setValue(
            completed
        )

        self.status_label.setText(
            f"Processing T={time} P={position}"
        )

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
        #cluster_channel = int(self.cluster_channel_combo.currentIndex())

        df_eps = df.filter(pl.col("channel") == eps_channel)
        #df_cluster = TODO: Need to add for cluster analysis

        print("ALL TIMES:", sorted(df["time"].unique()))
        print("EPS TIMES:", sorted(df_eps["time"].unique()))
        #print("CLUSTER TIMES:", sorted(df_cluster["time"].unique()))

        stats_eps = self.compute_avg_sd_stats(df_eps)
        print("\n=== RAW DF CHECK ===")
        print(df.head(20))

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
        """# --- manual time mapping ---
        unique_times = sorted(self.metrics_service.df["time"].unique())
        # REAL TIMES HERE
        real_times = [0, 24, 48, 85, 96]
        # Build mapping
        time_map = dict(zip(unique_times, real_times))"""
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

        fig = self.population_figure
        axs = fig.subplots(3, 1)

        # --- TIME ---
        time_eps = np.array(stats_eps["time"]) * hours_per_frame
        #time_cluster = np.array(stats_cluster["time"]) * hours_per_frame

        # --- 1. EPS ---
        axs[0].errorbar(
            time_eps,
            stats_eps["mean_integrated"],
            yerr=stats_eps["std_integrated"],
            fmt='-o',
            capsize=5,
            color='green'
        )
        axs[0].set_title("EPS Fluorescence")
        axs[0].set_ylabel("Integrated Intensity (Total)")

        # --- 2. BIOMASS (Integrated) ---
        #axs[1].errorbar(
        #    time_cluster,
        #    stats_cluster["mean_integrated"],
        #    yerr=stats_cluster["std_integrated"],
        #    fmt='-o',
        #    capsize=5,
        #    color='blue'
        #)
        #axs[1].set_title("Integrated Intensity (Biomass Proxy)")
        #axs[1].set_ylabel("Total Signal")

        # --- 3. MORPHOLOGY (AREA) ---
        axs[2].errorbar(
            time_eps,
            stats_eps["mean_occupancy"],
            yerr=stats_eps["std_occupancy"],
            fmt='-o',
            capsize=5,
            color='goldenrod'
        )
        axs[2].set_title("EPS Area Fraction")
        axs[2].set_ylabel("Area Fraction (%)")
        axs[2].set_xlabel("Time (hours)")

        fig.subplots_adjust(
            top=0.92,
            bottom=0.08,
            left=0.12,
            right=0.95,
            hspace=0.4
        )

        self.population_canvas.draw()

    def process_eps_frame(self,
            frame: np.ndarray,
            background_sigma: float = 200,
            smoothing_sigma: float = 1.5,
            rolling_ball_fraction: int = 8,
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

        image_height, image_width = image.shape[:2]

        rolling_ball_radius = (
                min(image_height, image_width)
                // rolling_ball_fraction
        )

        local_otsu_radius = rolling_ball_radius

        # -----------------------------
        # 1. Large Gaussian background correction
        # -----------------------------

        gaussian_background = gaussian_filter(
            image,
            sigma=background_sigma
        )

        gaussian_corrected = image - gaussian_background

        gaussian_corrected = np.clip(
            gaussian_corrected,
            0,
            None
        )

        # -----------------------------
        # 2. Rolling ball background correction
        # -----------------------------

        rolling_background = rolling_ball(
            gaussian_corrected,
            radius=rolling_ball_radius
        )

        rolling_corrected = gaussian_corrected - rolling_background

        rolling_corrected = np.clip(
            rolling_corrected,
            0,
            None
        )

        # -----------------------------
        # 3. Light smoothing before segmentation
        # -----------------------------

        smoothed = gaussian_filter(
            rolling_corrected,
            sigma=smoothing_sigma
        )

        # -----------------------------
        # 4. Normalize for Local Otsu
        # -----------------------------

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

        local_threshold = otsu(
            smoothed_8bit,
            footprint=disk(local_otsu_radius)
        )

        eps_mask = smoothed_8bit > local_threshold

        # -----------------------------
        # 6. Metrics
        # -----------------------------

        eps_pixels = np.count_nonzero(eps_mask)
        total_pixels = eps_mask.size

        occupancy_fraction = eps_pixels / total_pixels
        occupancy_percent = occupancy_fraction * 100

        if eps_pixels > 0:
            mean_intensity = rolling_corrected[eps_mask].mean()
            integrated_intensity = rolling_corrected[eps_mask].sum()
        else:
            mean_intensity = 0.0
            integrated_intensity = 0.0

        return {
            "gaussian_corrected": gaussian_corrected,
            "rolling_background": rolling_background,
            "rolling_corrected": rolling_corrected,
            "smoothed": smoothed,
            "eps_mask": eps_mask,

            "rolling_ball_radius": rolling_ball_radius,
            "local_otsu_radius": local_otsu_radius,

            "occupancy_fraction": occupancy_fraction,
            "occupancy_percent": occupancy_percent,
            "mean_intensity": mean_intensity,
            "integrated_intensity": integrated_intensity,
            "eps_area_pixels": eps_pixels,
        }

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