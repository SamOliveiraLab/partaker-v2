# motility_widget.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QWidget,
    QMessageBox,
    QFileDialog,
    QCheckBox,
    QDialogButtonBox,
    QTableWidget,
    QTableWidgetItem,
    QComboBox,
    QGridLayout,
    QApplication,
    QRadioButton,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.cm import plasma, viridis
from matplotlib.colors import Normalize
from matplotlib.colors import to_hex
from pubsub import pub
from skimage.measure import label, regionprops

from nd2_analyzer.analysis.tracking.tracking import (
    create_density_based_regions_from_forecast_data,
    enhanced_motility_index,
    visualize_motility_with_chamber_regions,
    visualize_motility_map,
)


# Modify in motility_widget.py


class MotilityDialog(QDialog):
    """
    Dialog for analyzing and visualizing cell motility.
    """

    def __init__(self, tracked_cells, lineage_tracks, image_data=None, parent=None):
        super().__init__(parent)

        self.tracked_cells = tracked_cells
        self.lineage_tracks = lineage_tracks
        self.image_data = image_data
        self.motility_metrics = None

        # ADD THIS LINE - Import and initialize MetricsService
        from nd2_analyzer.analysis.metrics_service import MetricsService
        from nd2_analyzer.data.appstate import ApplicationState

        self.metrics_service = MetricsService()

        self.calibration = 0.07

        # Get time interval from experiment settings
        appstate = ApplicationState.get_instance()
        if appstate and appstate.experiment:
            self.hours_per_frame = appstate.experiment.time_interval_hours
            self.focus_loss_intervals = appstate.experiment.focus_loss_intervals
        else:
            # Fallback to default if no experiment loaded
            self.hours_per_frame = 0.05  # Default: 3 minutes per frame
            self.focus_loss_intervals = []
            print("WARNING: No experiment loaded. Using default hours_per_frame = 0.05")

        print(f"MotilityDialog initialized with hours_per_frame = {self.hours_per_frame:.4f}")

        # Set dialog properties
        self.setWindowTitle("Cell Motility Analysis")
        self.setMinimumWidth(1200)
        self.setMinimumHeight(800)

        # Initialize UI
        self.init_ui()

        # Start analysis
        self.analyze_motility()

    def init_ui(self):
        """Initialize the dialog UI"""
        layout = QVBoxLayout(self)

        # Create tab widget - make it fill the entire dialog
        self.tab_widget = QTabWidget()

        # Create tabs
        self.combined_tab = QWidget()
        self.map_tab = QWidget()
        self.metrics_tab = QWidget()
        self.region_tab = QWidget()

        # Set up tab layouts
        self.combined_layout = QVBoxLayout(self.combined_tab)
        self.map_layout = QVBoxLayout(self.map_tab)
        self.metrics_layout = QVBoxLayout(self.metrics_tab)
        self.region_layout = QVBoxLayout(self.region_tab)

        self.track_selector_widget = QWidget()
        track_selector_layout = QHBoxLayout(self.track_selector_widget)

        track_label = QLabel("Select Track:")
        self.track_combo = QComboBox()
        self.track_combo.setEditable(True)
        self.track_combo.setInsertPolicy(QComboBox.NoInsert)
        self.track_combo.currentIndexChanged.connect(self.highlight_selected_track)

        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_track_highlight)

        track_selector_layout.addWidget(track_label)
        track_selector_layout.addWidget(self.track_combo, 1)
        track_selector_layout.addWidget(clear_button)

        # Add to combined layout at the top (before you add other widgets to combined_layout)
        self.combined_layout.insertWidget(0, self.track_selector_widget)

        # Add tabs
        self.tab_widget.addTab(self.combined_tab, "Motility by Region")
        self.tab_widget.addTab(self.map_tab, "Motility Map")

        # Add tab widget - it will now take up all the space
        layout.addWidget(self.tab_widget)

        # Create a status bar at the bottom for the summary, which will be shown only when needed
        self.status_bar = QWidget()
        status_layout = QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(5, 0, 5, 0)  # Minimal margins

        # Summary label (now more compact)
        self.summary_label = QLabel()
        self.summary_label.setTextFormat(Qt.RichText)

        # Export and Close buttons
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)

        status_layout.addWidget(self.summary_label)
        status_layout.addStretch()
        status_layout.addWidget(self.export_button)
        status_layout.addWidget(self.close_button)

        layout.addWidget(self.status_bar)

        self.velocity_time_tab = QWidget()
        self.velocity_time_layout = QVBoxLayout(self.velocity_time_tab)
        self.tab_widget.addTab(self.velocity_time_tab, "Velocity vs. Time")

        self.comparison_tab = QWidget()
        self.comparison_layout = QHBoxLayout(self.comparison_tab)
        self.tab_widget.addTab(self.comparison_tab, "Motility vs. Velocity")

        # Create left side (Motility over time)
        motility_time_panel = QWidget()
        motility_time_layout = QVBoxLayout(motility_time_panel)
        motility_time_panel.setMinimumWidth(500)

        # Motility plot
        self.motility_fig = plt.figure(figsize=(7, 5))
        self.motility_canvas = FigureCanvas(self.motility_fig)
        self.motility_ax = self.motility_fig.add_subplot(111)
        motility_time_layout.addWidget(QLabel("Motility Index Over Time"))
        motility_time_layout.addWidget(self.motility_canvas)

        # Create right side (Velocity over time) - reusing existing code
        velocity_time_panel = QWidget()
        velocity_time_layout = QVBoxLayout(velocity_time_panel)
        velocity_time_panel.setMinimumWidth(500)

        # Velocity plot - create a new one for this tab
        self.velocity_comp_fig = plt.figure(figsize=(7, 5))
        self.velocity_comp_canvas = FigureCanvas(self.velocity_comp_fig)
        self.velocity_comp_ax = self.velocity_comp_fig.add_subplot(111)
        velocity_time_layout.addWidget(QLabel("Velocity Over Time"))
        velocity_time_layout.addWidget(self.velocity_comp_canvas)

        # Add both panels to the comparison layout
        self.comparison_layout.addWidget(motility_time_panel)
        self.comparison_layout.addWidget(velocity_time_panel)

        # Add controls for selecting tracks
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(200)

        # Create a split layout with controls on left, graph on right
        velocity_split = QHBoxLayout()
        self.velocity_time_layout.addLayout(velocity_split)

        # Left side: Controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(300)

        # Track selector
        control_layout.addWidget(QLabel("Select Tracks:"))
        self.velocity_track_list = QTableWidget()
        self.velocity_track_list.setColumnCount(2)
        self.velocity_track_list.setHorizontalHeaderLabels(["Track ID", "Show"])
        self.velocity_track_list.horizontalHeader().setStretchLastSection(True)
        control_layout.addWidget(self.velocity_track_list)

        # Add control buttons
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_velocity_tracks)
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.clear_all_velocity_tracks)
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(clear_all_btn)
        control_layout.addLayout(button_layout)

        # Options
        self.show_average_checkbox = QCheckBox("Show Population Average")
        self.show_average_checkbox.setChecked(True)
        self.show_average_checkbox.stateChanged.connect(self.update_velocity_plot)
        control_layout.addWidget(self.show_average_checkbox)

        self.show_divisions_checkbox = QCheckBox("Show Division Events")
        self.show_divisions_checkbox.setChecked(True)
        self.show_divisions_checkbox.stateChanged.connect(self.update_velocity_plot)
        control_layout.addWidget(self.show_divisions_checkbox)

        # Right side: Graph
        self.velocity_fig = plt.figure(figsize=(8, 6))
        self.velocity_canvas = FigureCanvas(self.velocity_fig)
        self.velocity_ax = self.velocity_fig.add_subplot(111)

        # Add both sides to the layout
        velocity_split.addWidget(control_panel)
        velocity_split.addWidget(self.velocity_canvas, 1)  # Graph gets more space

        self.comparison_layout.insertWidget(0, control_panel)

        # Connect tab changed signal to update the summary visibility
        self.tab_widget.currentChanged.connect(self.update_summary_visibility)

        # Create environmental analysis tab
        self.create_environmental_analysis()

    def create_environmental_analysis(self):
        """Create environmental correlation analysis interface"""

        # Create the environmental tab
        self.environmental_tab = QWidget()
        self.environmental_layout = QVBoxLayout(self.environmental_tab)
        self.tab_widget.addTab(self.environmental_tab, "Environmental Analysis")

        # Import controls
        import_layout = QHBoxLayout()

        self.import_button = QPushButton("Import Environmental Data")
        self.import_button.clicked.connect(self.import_environmental_data)
        import_layout.addWidget(self.import_button)

        self.env_data_label = QLabel("No environmental data loaded")
        import_layout.addWidget(self.env_data_label)

        self.environmental_layout.addLayout(import_layout)

        # Analysis controls
        analysis_layout = QHBoxLayout()

        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "All Cells",
            "Mature Cells Only (position >= 7)",
            "First Generation Only",
            "Extreme Conditions Only"
        ])
        analysis_layout.addWidget(QLabel("Filter:"))
        analysis_layout.addWidget(self.filter_combo)

        self.analyze_button = QPushButton("Run Analysis")
        self.analyze_button.clicked.connect(self.run_environmental_analysis)
        self.analyze_button.setEnabled(False)
        analysis_layout.addWidget(self.analyze_button)

        self.environmental_layout.addLayout(analysis_layout)

        # Results area
        self.env_figure = plt.figure(figsize=(16, 10))  # Bigger figure
        self.env_canvas = FigureCanvas(self.env_figure)
        self.environmental_layout.addWidget(self.env_canvas)

    def import_environmental_data(self):
        """Import Hamed's merged dataset with pressure and velocity data"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Environmental Data",
            "", "CSV Files (*.csv)")

        if file_path:
            try:
                import pandas as pd
                self.env_data = pd.read_csv(file_path)

                # Verify required columns exist
                required_cols = ['pressure_df', 'velocity_df2', 'cell_id', 'time_point',
                            'position_in_track', 'morphology_class']
                missing_cols = [col for col in required_cols if col not in self.env_data.columns]

                if missing_cols:
                    QMessageBox.warning(self, "Missing Columns",
                        f"Missing required columns: {missing_cols}")
                    return

                self.env_data_label.setText(f"Loaded {len(self.env_data)} records")
                self.analyze_button.setEnabled(True)

                QMessageBox.information(self, "Success",
                    f"Loaded environmental data with {len(self.env_data)} cell records")

            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import data: {str(e)}")

    def filter_environmental_data(self):
        """Apply selected filter to environmental data"""
        filter_type = self.filter_combo.currentText()

        if filter_type == "All Cells":
            return self.env_data

        elif filter_type == "Mature Cells Only (position >= 7)":
            return self.env_data[self.env_data['position_in_track'] >= 7]

        elif filter_type == "First Generation Only":
            return self.env_data[self.env_data['parent_id'].isna()]

        elif filter_type == "Extreme Conditions Only":
            # Top and bottom 10% of pressure and velocity conditions
            pressure_high = self.env_data['pressure_df'].quantile(0.9)
            pressure_low = self.env_data['pressure_df'].quantile(0.1)
            velocity_high = self.env_data['velocity_df2'].quantile(0.9)
            velocity_low = self.env_data['velocity_df2'].quantile(0.1)

            extreme_conditions = (
                (self.env_data['pressure_df'] >= pressure_high) |
                (self.env_data['pressure_df'] <= pressure_low) |
                (self.env_data['velocity_df2'] >= velocity_high) |
                (self.env_data['velocity_df2'] <= velocity_low)
            )

            return self.env_data[extreme_conditions]


    def run_environmental_analysis(self):
        """Run the corrected environmental correlation analysis using morphology classes"""

        if not hasattr(self, 'env_data'):
            QMessageBox.warning(self, "No Data", "Please import environmental data first")
            return

        # Filter data based on selection
        filtered_data = self.filter_environmental_data()

        if len(filtered_data) == 0:
            QMessageBox.warning(self, "No Data", "No data remaining after filtering")
            return

        # Clear previous plots
        self.env_figure.clear()

        # Set figure size larger and create better spaced layout
        self.env_figure.set_size_inches(16, 10)

        # Create 2x2 subplot layout with much more spacing
        gs = self.env_figure.add_gridspec(2, 2,
                                        hspace=0.5,    # More vertical space
                                        wspace=0.3,    # More horizontal space
                                        left=0.06,     # Less left margin
                                        right=0.75,    # More space for legends
                                        top=0.95,      # Less top margin
                                        bottom=0.12)   # More bottom space

        # Plot 1: Pressure vs Morphology Distribution
        ax1 = self.env_figure.add_subplot(gs[0, 0])
        self.plot_morphology_vs_environment(
            filtered_data, 'pressure_df', 'Pressure', ax1)

        # Plot 2: Velocity vs Morphology Distribution
        ax2 = self.env_figure.add_subplot(gs[0, 1])
        self.plot_morphology_vs_environment(
            filtered_data, 'velocity_df2', 'Velocity', ax2)

        # Plot 3: Pressure Binned Analysis
        ax3 = self.env_figure.add_subplot(gs[1, 0])
        self.plot_binned_morphology_analysis(
            filtered_data, 'pressure_df', 'Pressure', ax3)

        # Plot 4: Velocity Binned Analysis
        ax4 = self.env_figure.add_subplot(gs[1, 1])
        self.plot_binned_morphology_analysis(
            filtered_data, 'velocity_df2', 'Velocity', ax4)

        # Don't use tight_layout as it interferes with our custom spacing
        self.env_canvas.draw()

        # Show statistical summary
        self.show_morphology_correlation_summary(filtered_data)

    def plot_morphology_vs_environment(self, data, env_col, env_label, ax):
        """Plot morphology class distribution vs environmental variable with better formatting"""

        # Define colors for morphology classes
        color_map = {
            'Healthy': '#1f77b4',
            'Divided': '#ff7f0e',
            'Artifact': '#2ca02c',
            'Elongated': '#d62728',
            'Deformed': '#9467bd'
        }

        # Order morphology classes consistently
        ordered_classes = ['Healthy', 'Divided', 'Artifact', 'Elongated', 'Deformed']
        available_classes = [cls for cls in ordered_classes if cls in data['morphology_class'].unique()]

        # Create y-positions for each class
        y_positions = {cls: i for i, cls in enumerate(available_classes)}

        for morph_class in available_classes:
            if morph_class in color_map:
                class_data = data[data['morphology_class'] == morph_class]
                y_vals = [y_positions[morph_class]] * len(class_data)

                ax.scatter(class_data[env_col], y_vals,
                        alpha=0.6, s=6, color=color_map[morph_class],
                        label=f'{morph_class} (n={len(class_data)})')

        # Customize plot with smaller fonts
        ax.set_xlabel(env_label, fontsize=10)
        ax.set_ylabel('Morphology Class', fontsize=10)
        ax.set_title(f'{env_label} vs Morphology Class', fontsize=11, pad=15)

        # Set y-axis to show class names
        ax.set_yticks(range(len(available_classes)))
        ax.set_yticklabels(available_classes, fontsize=9)

        # Smaller tick labels
        ax.tick_params(axis='x', labelsize=8)

        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_axisbelow(True)

        # Smaller legend positioned better
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    def plot_binned_morphology_analysis(self, data, env_col, env_label, ax):
        """Plot morphology class proportions across environmental bins with better formatting"""

        import numpy as np
        import pandas as pd

        # Create 5 bins across the environmental variable range
        data_copy = data.copy()
        data_copy[f'{env_col}_bin'] = pd.cut(data_copy[env_col], bins=5)

        # Calculate proportions for each bin
        bin_proportions = data_copy.groupby([f'{env_col}_bin', 'morphology_class']).size().unstack(fill_value=0)

        # Convert to percentages
        bin_percentages = bin_proportions.div(bin_proportions.sum(axis=1), axis=0) * 100

        # Define colors (same order as legend)
        color_map = {
            'Healthy': '#1f77b4',
            'Divided': '#ff7f0e',
            'Artifact': '#2ca02c',
            'Elongated': '#d62728',
            'Deformed': '#9467bd'
        }

        # Ensure consistent ordering of morphology classes
        ordered_classes = ['Healthy', 'Divided', 'Artifact', 'Elongated', 'Deformed']
        available_classes = [cls for cls in ordered_classes if cls in bin_percentages.columns]

        # Plot stacked bar chart
        bottom = np.zeros(len(bin_percentages))
        bar_width = 0.7  # Slightly narrower bars

        for morph_class in available_classes:
            values = bin_percentages[morph_class].values
            ax.bar(range(len(bin_percentages)), values, bottom=bottom,
                label=morph_class, color=color_map[morph_class], width=bar_width)
            bottom += values

        # Customize plot appearance with smaller fonts
        ax.set_xlabel(f'{env_label} Range (Low → High)', fontsize=10)
        ax.set_ylabel('Percentage of Cells', fontsize=10)
        ax.set_title(f'Morphology Distribution Across {env_label} Ranges', fontsize=11, pad=15)

        # Fix x-axis with smaller labels
        ax.set_xticks(range(len(bin_percentages)))
        ax.set_xticklabels([f'Bin {i+1}' for i in range(len(bin_percentages))], fontsize=9)

        # Set y-axis to 0-100% with smaller tick labels
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.tick_params(axis='y', labelsize=8)

        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)

        # Smaller legend positioned outside plot area
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

        # Add bin range annotations at the bottom with smaller text
        bin_ranges = []
        for bin_interval in bin_percentages.index:
            left = bin_interval.left
            right = bin_interval.right
            bin_ranges.append(f'{left:.1f}-{right:.1f}')  # Less decimal places

        # Add secondary x-axis labels for ranges with smaller font
        for i, range_text in enumerate(bin_ranges):
            ax.text(i, -7, range_text, ha='center', va='top', fontsize=7, rotation=0)


    def show_morphology_correlation_summary(self, data):
        """Show statistical summary of morphology-environment relationships"""

        import numpy as np
        import pandas as pd
        from scipy import stats

        summary_results = []

        # Analyze pressure effects
        pressure_analysis = self.analyze_morphology_environment_correlation(
            data, 'pressure_df', 'Pressure')
        summary_results.extend(pressure_analysis)

        # Analyze velocity effects
        velocity_analysis = self.analyze_morphology_environment_correlation(
            data, 'velocity_df2', 'Velocity')
        summary_results.extend(velocity_analysis)

        # Format results
        summary_text = f"Morphology-Environment Analysis ({len(data)} cells):\n\n"
        summary_text += "\n".join(summary_results)

        # Show in dialog
        QMessageBox.information(self, "Morphology Correlation Summary", summary_text)

    def analyze_morphology_environment_correlation(self, data, env_col, env_label):
        """Analyze correlation between environment and morphology classes"""

        import numpy as np
        from scipy import stats

        results = []

        # Create high vs low environmental groups
        median_val = data[env_col].median()
        high_env = data[data[env_col] > median_val]
        low_env = data[data[env_col] <= median_val]

        # Analyze each morphology class
        morphology_classes = data['morphology_class'].unique()

        for morph_class in morphology_classes:
            # Calculate proportions in high vs low groups
            high_prop = (high_env['morphology_class'] == morph_class).mean() * 100
            low_prop = (low_env['morphology_class'] == morph_class).mean() * 100

            # Statistical test (chi-square)
            contingency_table = pd.crosstab(
                data[env_col] > median_val,
                data['morphology_class'] == morph_class
            )

            try:
                chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)

                results.append(
                    f"{env_label} vs {morph_class}: "
                    f"High {env_label}: {high_prop:.1f}%, "
                    f"Low {env_label}: {low_prop:.1f}% "
                    f"(p = {p_value:.3f})"
                )
            except:
                results.append(
                    f"{env_label} vs {morph_class}: "
                    f"High {env_label}: {high_prop:.1f}%, "
                    f"Low {env_label}: {low_prop:.1f}% "
                    f"(statistical test failed)"
                )

        return results

    def analyze_motility(self):
        """Analyze cell motility - UPDATED"""
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Get current position and channel
            p = pub.sendMessage("get_current_p", default=0)
            if p is None:
                p = 0
                print(f"Current position was None, defaulting to position {p}")

            c = pub.sendMessage("get_current_c", default=0)
            if c is None:
                c = 0
                print(f"Current channel was None, defaulting to channel {c}")

            print(f"Using position={p}, channel={c}")

            # Determine chamber dimensions from image data
            chamber_dimensions = (1392, 1040)  # default
            if self.image_data and hasattr(self.image_data, "data"):
                if len(self.image_data.data.shape) >= 4:
                    height = self.image_data.data.shape[-2]
                    width = self.image_data.data.shape[-1]
                    chamber_dimensions = (width, height)
                    print(
                        f"Using chamber dimensions from image data: {chamber_dimensions}"
                    )

            # Collect all cell positions
            all_cell_positions = self.collect_cell_positions(p, c)
            print(
                f"Collected {len(all_cell_positions)} cell positions for visualization"
            )

            # Filter tracks by focus loss intervals
            from nd2_analyzer.data.appstate import ApplicationState
            appstate = ApplicationState.get_instance()

            if appstate and appstate.experiment and self.focus_loss_intervals:
                filtered_tracks = []
                for track in self.tracked_cells:
                    filtered = appstate.experiment.filter_track_frames(track)
                    if filtered and len(filtered["x"]) >= 2:  # Keep tracks with at least 2 points
                        filtered_tracks.append(filtered)
                print(f"Focus loss filtering: {len(self.tracked_cells)} → {len(filtered_tracks)} tracks")
                tracks_for_analysis = filtered_tracks
            else:
                tracks_for_analysis = self.tracked_cells

            # Calculate motility metrics
            self.motility_metrics = enhanced_motility_index(
                tracks_for_analysis, chamber_dimensions
            )

            # Create visualizations
            self.create_visualizations(chamber_dimensions, all_cell_positions)

            # ADD THIS LINE - Create density analysis
            self.create_density_analysis()

            # Update summary
            self.update_summary()

        except Exception as e:
            import traceback

            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Failed to analyze motility: {str(e)}")

        finally:
            QApplication.restoreOverrideCursor()

    def collect_cell_positions(self, p, c):
        """Collect cell positions from metrics data or tracks"""
        all_cell_positions = []

        # Try to use metrics service instead of segmentation cache to avoid triggering re-segmentation
        if self.metrics_service:
            try:
                # Determine number of time frames
                t_max = 20
                if hasattr(self.image_data, "data"):
                    t_max = min(20, self.image_data.data.shape[0])

                print(f"Collecting cell positions from metrics data for {t_max} frames")
                for t in range(t_max):
                    try:
                        # Query metrics for this time/position - avoid triggering segmentation
                        metrics_df = self.metrics_service.query_optimized(time=t, position=p)

                        if not metrics_df.is_empty():
                            # Get centroids from metrics
                            for row in metrics_df.to_pandas().itertuples():
                                x = row.centroid_x
                                y = row.centroid_y
                                all_cell_positions.append((x, y))
                    except Exception as frame_error:
                        print(f"Frame {t}: Error querying metrics - {str(frame_error)}")

            except Exception as e:
                print(f"Error collecting cell positions from metrics: {str(e)}")
                all_cell_positions = []

        # Fall back to using tracks if metrics didn't work
        if not all_cell_positions:
            print(f"Falling back to collecting positions from tracks")
            for track in self.lineage_tracks:
                if "x" in track and "y" in track:
                    all_cell_positions.extend(list(zip(track["x"], track["y"])))
            print(f"Collected {len(all_cell_positions)} cell positions from tracks")

        return all_cell_positions

    def create_visualizations(self, chamber_dimensions, all_cell_positions):
        """Create motility visualizations"""

        # Store chamber dimensions for later use
        self.chamber_dimensions = chamber_dimensions
        self.all_cell_positions = all_cell_positions

        # Combined visualization
        combined_fig, self.combined_ax = visualize_motility_with_chamber_regions(
            self.tracked_cells,
            all_cell_positions,
            chamber_dimensions,
            self.motility_metrics,
        )
        combined_canvas = FigureCanvas(combined_fig)
        self.combined_layout.addWidget(combined_canvas)
        self.combined_fig = combined_fig
        self.combined_canvas = combined_canvas

        # Populate track selector
        self.populate_track_selector()
        self.populate_velocity_track_list()

        # Motility map
        map_fig, _ = visualize_motility_map(
            self.tracked_cells, chamber_dimensions, self.motility_metrics
        )
        map_canvas = FigureCanvas(map_fig)
        self.map_layout.addWidget(map_canvas)
        self.map_fig = map_fig

    # Add these new methods to the MotilityDialog class

    def populate_track_selector(self):
        """Populate the track selector with ALL track IDs present in visualization"""
        self.track_combo.clear()
        self.track_combo.addItem("-- Select a track --", None)

        # Create a set of all track IDs
        track_ids = set()

        # First add all tracks from tracked_cells
        for track in self.tracked_cells:
            if "ID" in track:
                track_ids.add(track["ID"])

        # Create a lookup for metrics (if available)
        metrics_lookup = {}
        if self.motility_metrics and "individual_metrics" in self.motility_metrics:
            metrics_lookup = {
                m.get("track_id"): m
                for m in self.motility_metrics["individual_metrics"]
            }

        # Sort track IDs numerically
        sorted_ids = sorted(track_ids)

        # Add each track to combo box
        for track_id in sorted_ids:
            # Check if we have metrics for this track
            if track_id in metrics_lookup:
                metric = metrics_lookup[track_id]
                motility = metric.get("motility_index", 0)
                self.track_combo.addItem(
                    f"Track {track_id} (MI: {motility:.1f})", track_id
                )
            else:
                # Add track even without motility data
                self.track_combo.addItem(f"Track {track_id}", track_id)

    def highlight_selected_track(self, index):
        """Highlight the selected track on the visualization"""
        if index <= 0:  # No selection or the default item
            self.clear_track_highlight()
            return

        # Get the track ID from the combo box
        track_id = self.track_combo.itemData(index)

        # Find the track
        selected_track = next(
            (t for t in self.tracked_cells if t.get("ID", -1) == track_id), None
        )

        # Clear any existing highlight
        self.clear_track_highlight()

        if selected_track and "x" in selected_track and "y" in selected_track:
            # Highlight the track
            self.highlighted_line = self.combined_ax.plot(
                selected_track["x"],
                selected_track["y"],
                "-",
                linewidth=3,
                color="red",
                zorder=100,
            )[0]

            # Add start/end markers
            self.highlighted_start = self.combined_ax.plot(
                selected_track["x"][0],
                selected_track["y"][0],
                "o",
                markersize=8,
                color="red",
                zorder=100,
            )[0]

            self.highlighted_end = self.combined_ax.plot(
                selected_track["x"][-1],
                selected_track["y"][-1],
                "s",
                markersize=8,
                color="red",
                zorder=100,
            )[0]

            # Prepare info text with available data
            info_text = [f"Track ID: {track_id}"]

            # Add track length
            if "x" in selected_track:
                info_text.append(f"Track Length: {len(selected_track['x'])} frames")

            # Calculate path length (if not available)
            if (
                "x" in selected_track
                and "y" in selected_track
                and len(selected_track["x"]) > 1
            ):
                # Calculate path length
                path_length = 0
                for i in range(len(selected_track["x"]) - 1):
                    dx = selected_track["x"][i + 1] - selected_track["x"][i]
                    dy = selected_track["y"][i + 1] - selected_track["y"][i]
                    path_length += np.sqrt(dx**2 + dy**2)
                info_text.append(f"Path Length: {path_length:.1f} px")

            # Look up motility metrics for this track (if available)
            if self.motility_metrics and "individual_metrics" in self.motility_metrics:
                track_metric = next(
                    (
                        m
                        for m in self.motility_metrics["individual_metrics"]
                        if m.get("track_id") == track_id
                    ),
                    None,
                )

                if track_metric:
                    # Add motility metrics
                    info_text.append(
                        f"Motility Index: {track_metric.get('motility_index', 0):.1f}"
                    )
                    info_text.append(
                        f"Avg Velocity: {track_metric.get('avg_velocity', 0):.2f} px/frame"
                    )
                    info_text.append(
                        f"Confinement: {track_metric.get('confinement_ratio', 0):.2f}"
                    )
                    info_text.append(
                        f"Persistence: {track_metric.get('directional_persistence', 0):.2f}"
                    )

            # Add text near track end
            self.highlighted_text = self.combined_ax.text(
                selected_track["x"][-1] + 10,
                selected_track["y"][-1] + 10,
                "\n".join(info_text),
                bbox=dict(
                    facecolor="white", alpha=0.9, edgecolor="black", boxstyle="round"
                ),
                zorder=100,
            )

            # Update the canvas
            self.combined_canvas.draw()

    def clear_track_highlight(self):
        """Clear the highlighted track"""
        # Remove any existing highlight elements
        if hasattr(self, "highlighted_line") and self.highlighted_line:
            self.highlighted_line.remove()
            self.highlighted_line = None

        if hasattr(self, "highlighted_start") and self.highlighted_start:
            self.highlighted_start.remove()
            self.highlighted_start = None

        if hasattr(self, "highlighted_end") and self.highlighted_end:
            self.highlighted_end.remove()
            self.highlighted_end = None

        if hasattr(self, "highlighted_text") and self.highlighted_text:
            self.highlighted_text.remove()
            self.highlighted_text = None

        # Update the canvas
        if hasattr(self, "combined_canvas"):
            self.combined_canvas.draw()

    def update_summary_visibility(self, index):
        """Show or hide the summary based on the current tab"""
        # Hide summary for velocity analysis tab to maximize space
        tab_name = self.tab_widget.tabText(index)
        if tab_name == "Velocity Analysis":
            self.summary_label.setVisible(False)
        else:
            self.summary_label.setVisible(True)
            # Update the summary display if it exists
            if self.motility_metrics:
                self.update_summary()

    def update_summary(self):
        """Update the summary label - now more compact horizontal format"""
        if not self.motility_metrics:
            return

        summary_text = (
            f"<b>Population Avg MI:</b> {self.motility_metrics['population_avg_motility']:.1f}/100 | "
            f"<b>Heterogeneity:</b> {self.motility_metrics['population_heterogeneity']:.2f} | "
            f"<b>Sample:</b> {self.motility_metrics['sample_size']} cells"
        )
        self.summary_label.setText(summary_text)

    def export_results(self):
        """Export analysis results"""
        export_dialog = QDialog(self)
        export_dialog.setWindowTitle("Export Options")
        export_layout = QVBoxLayout(export_dialog)

        export_label = QLabel("Select export options:")
        export_layout.addWidget(export_label)

        export_map = QCheckBox("Export Motility Map")
        export_map.setChecked(True)
        export_layout.addWidget(export_map)

        export_metrics = QCheckBox("Export Detailed Metrics Plot")
        export_metrics.setChecked(True)
        export_layout.addWidget(export_metrics)

        export_regional = QCheckBox("Export Regional Analysis")
        export_regional.setChecked(self.has_regional)
        export_regional.setEnabled(self.has_regional)
        export_layout.addWidget(export_regional)

        export_csv = QCheckBox("Export Metrics as CSV")
        export_csv.setChecked(True)
        export_layout.addWidget(export_csv)

        export_buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        export_buttons.accepted.connect(export_dialog.accept)
        export_buttons.rejected.connect(export_dialog.reject)
        export_layout.addWidget(export_buttons)

        if export_dialog.exec() == QDialog.Accepted:
            save_path, _ = QFileDialog.getSaveFileName(
                export_dialog, "Save Results", "", "All Files (*)"
            )

            if save_path:
                base_path = save_path.replace(".png", "").replace(".csv", "")

                if export_map.isChecked():
                    self.map_fig.savefig(
                        f"{base_path}_motility_map.png", dpi=300, bbox_inches="tight"
                    )

                if export_metrics.isChecked():
                    self.metrics_fig.savefig(
                        f"{base_path}_detailed_metrics.png",
                        dpi=300,
                        bbox_inches="tight",
                    )

                if export_csv.isChecked():
                    metrics_df = pd.DataFrame(
                        self.motility_metrics["individual_metrics"]
                    )
                    metrics_df.to_csv(f"{base_path}_motility_metrics.csv", index=False)

                QMessageBox.information(
                    export_dialog,
                    "Export Complete",
                    f"Results exported to {base_path}_*.png/csv",
                )

    def populate_velocity_track_list(self):
        """Populate the track list for velocity-time analysis with motility information"""
        self.velocity_track_list.setRowCount(0)

        # Create a lookup for metrics
        metrics_lookup = {}
        if self.motility_metrics and "individual_metrics" in self.motility_metrics:
            metrics_lookup = {
                m.get("track_id"): m
                for m in self.motility_metrics["individual_metrics"]
            }

        # Get tracks with metrics
        valid_tracks = []
        for track in self.tracked_cells:
            if "ID" in track and track["ID"] in metrics_lookup:
                valid_tracks.append((track, metrics_lookup[track["ID"]]))

        # Sort by motility index
        valid_tracks.sort(
            key=lambda item: item[1].get("motility_index", 0), reverse=True
        )

        # Add to table with three columns: ID, Motility, Show
        self.velocity_track_list.setColumnCount(3)
        self.velocity_track_list.setHorizontalHeaderLabels(
            ["Track ID", "Motility", "Show"]
        )

        # Add to table
        self.velocity_track_list.setRowCount(len(valid_tracks))

        for i, (track, metrics) in enumerate(valid_tracks):
            # Track ID
            id_item = QTableWidgetItem(str(track["ID"]))
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
            self.velocity_track_list.setItem(i, 0, id_item)

            # Motility Index
            motility_value = metrics.get("motility_index", 0)
            motility_item = QTableWidgetItem(f"{motility_value:.1f}")
            motility_item.setFlags(
                motility_item.flags() & ~Qt.ItemIsEditable
            )  # Make read-only

            # Set background color based on motility (matching the visualization color)
            color_value = plasma(motility_value / 100)
            # Convert to hex color for QTableWidgetItem
            hex_color = to_hex(color_value)
            motility_item.setBackground(QColor(hex_color))

            # Make text white or black depending on background brightness for readability
            # A simple heuristic: if the sum of RGB values is less than 384 (avg 128 per channel), use white text
            r, g, b = [int(255 * c) for c in color_value[:3]]
            if r + g + b < 384:
                motility_item.setForeground(QColor("white"))

            self.velocity_track_list.setItem(i, 1, motility_item)

            # Checkbox
            checkbox = QTableWidgetItem()
            checkbox.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            checkbox.setCheckState(Qt.Unchecked)
            self.velocity_track_list.setItem(i, 2, checkbox)

        # Adjust column widths
        self.velocity_track_list.setColumnWidth(0, 60)  # ID column
        self.velocity_track_list.setColumnWidth(1, 60)  # Motility column

        # Connect to update plot when selection changes
        self.velocity_track_list.itemChanged.connect(self.update_velocity_plot)
        self.velocity_track_list.itemChanged.connect(self.update_comparison_plots)

    def get_selected_velocity_tracks(self):
        """Get list of track IDs that are selected in the velocity tab"""
        selected_ids = []
        for i in range(self.velocity_track_list.rowCount()):
            if (
                self.velocity_track_list.item(i, 2).checkState() == Qt.Checked
            ):  # Changed from column 1 to 2
                track_id = int(self.velocity_track_list.item(i, 0).text())
                selected_ids.append(track_id)
        return selected_ids

    def select_all_velocity_tracks(self):
        """Select all tracks in the velocity analysis tab"""
        for i in range(self.velocity_track_list.rowCount()):
            self.velocity_track_list.item(i, 1).setCheckState(Qt.Checked)

    def clear_all_velocity_tracks(self):
        """Deselect all tracks in the velocity analysis tab"""
        for i in range(self.velocity_track_list.rowCount()):
            self.velocity_track_list.item(i, 1).setCheckState(Qt.Unchecked)

    def update_velocity_plot(self):
        """Update the velocity-time plot based on selected tracks"""
        # Clear the axis
        self.velocity_ax.clear()

        # Get selected tracks
        selected_ids = self.get_selected_velocity_tracks()

        # Default calibration if not set (µm/pixel)
        calibration = self.calibration

        # Use actual time interval from experiment settings
        hours_per_frame = self.hours_per_frame

        # Add a text box with summary statistics
        if selected_ids:
            self.add_summary_stats_box(selected_ids, hours_per_frame)

        # Calculate instantaneous velocities for each selected track
        for track_id in selected_ids:
            track = next(
                (t for t in self.tracked_cells if t.get("ID") == track_id), None
            )
            if not track or "x" not in track or "y" not in track:
                continue

            # Calculate instantaneous velocities
            times_hours = []
            velocities_um_per_s = []

            x = track["x"]
            y = track["y"]
            t = track["t"] if "t" in track else range(len(x))

            for i in range(len(t) - 1):
                dx = x[i + 1] - x[i]
                dy = y[i + 1] - y[i]
                dt_frames = max(1, t[i + 1] - t[i])  # Prevent division by zero

                # Calculate distance in pixels
                distance_pixels = np.sqrt(dx**2 + dy**2)

                # Convert to µm
                distance_um = distance_pixels * calibration

                # Convert time to hours and seconds
                dt_hours = dt_frames * hours_per_frame
                dt_seconds = dt_hours * 3600  # Convert hours to seconds for velocity

                # Calculate velocity in µm/s
                velocity = distance_um / dt_seconds

                # Store time in hours from start
                time_hours = t[i] * hours_per_frame
                times_hours.append(time_hours)
                velocities_um_per_s.append(velocity)

            # Plot this track's velocity
            color = viridis(selected_ids.index(track_id) / max(1, len(selected_ids)))
            self.velocity_ax.plot(
                times_hours,
                velocities_um_per_s,
                "-o",
                label=f"Track {track_id}",
                color=color,
                markersize=3,
                alpha=0.8,
            )

            # Add division markers if enabled
            if (
                self.show_divisions_checkbox.isChecked()
                and "children" in track
                and track["children"]
            ):
                last_time = t[-1] * hours_per_frame
                if velocities_um_per_s:  # Make sure we have velocity data
                    last_velocity = (
                        velocities_um_per_s[-1]
                        if len(velocities_um_per_s) == len(times_hours)
                        else 0
                    )
                    self.velocity_ax.plot(
                        last_time,
                        last_velocity,
                        "r*",
                        markersize=10,
                        label=f"Division (Track {track_id})",
                    )

        # Add population average if requested
        if self.show_average_checkbox.isChecked() and selected_ids:
            self.add_population_average(hours_per_frame)

        # Set labels and title
        self.velocity_ax.set_xlabel("Time (hours)")
        self.velocity_ax.set_ylabel("Velocity (µm/s)")
        self.velocity_ax.set_title("Cell Velocity vs. Time")

        # Add grid and legend
        self.velocity_ax.grid(True, linestyle="--", alpha=0.7)
        if selected_ids:
            self.velocity_ax.legend(loc="upper right")

        # Redraw
        self.velocity_fig.tight_layout()
        self.velocity_canvas.draw()

    def add_summary_stats_box(self, selected_ids, hours_per_frame):
        calibration = self.calibration

        """Add a box with summary statistics for selected tracks"""
        stats_text = []

        for track_id in selected_ids:
            track = next(
                (t for t in self.tracked_cells if t.get("ID") == track_id), None
            )
            if not track or "x" not in track or "y" not in track:
                continue

            # Calculate average velocity in both units
            x = track["x"]
            y = track["y"]
            t = track["t"] if "t" in track else range(len(x))

            total_distance_px = 0
            for i in range(len(x) - 1):
                dx = x[i + 1] - x[i]
                dy = y[i + 1] - y[i]
                distance = np.sqrt(dx**2 + dy**2)
                total_distance_px += distance

            # Calculate average velocities
            total_time_frames = t[-1] - t[0] if len(t) > 1 else 1
            avg_velocity_px_frame = total_distance_px / total_time_frames

            # Convert to µm/s
            total_distance_um = total_distance_px * calibration
            total_time_hours = total_time_frames * hours_per_frame
            total_time_seconds = total_time_hours * 3600
            avg_velocity_um_s = total_distance_um / total_time_seconds

            # Find track metrics if available
            metrics_str = ""
            if self.motility_metrics and "individual_metrics" in self.motility_metrics:
                track_metric = next(
                    (
                        m
                        for m in self.motility_metrics["individual_metrics"]
                        if m.get("track_id") == track_id
                    ),
                    None,
                )
                if track_metric:
                    metrics_str = f"MI: {track_metric.get('motility_index', 0):.1f}"

            # Add to stats text
            stats_text.append(
                f"Track {track_id} {metrics_str}\n"
                f"Avg: {avg_velocity_um_s:.3f} µm/s\n"
                f"     {avg_velocity_px_frame:.2f} px/frame\n"
                f"Length: {len(x)} frames"
            )

        # If we have stats, add the text box
        if stats_text:
            # Position in upper left corner
            self.velocity_ax.text(
                0.02,
                0.98,
                "\n\n".join(stats_text),
                transform=self.velocity_ax.transAxes,
                va="top",
                ha="left",
                bbox=dict(
                    boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"
                ),
                fontsize=9,
            )

    def add_population_average(self, hours_per_frame):
        """Add population average velocity to the plot"""
        # Use self.calibration instead of default
        calibration = self.calibration
        # Collect all velocities across all time points
        time_to_velocities = {}

        for track in self.tracked_cells:
            if "x" not in track or "y" not in track:
                continue

            x = track["x"]
            y = track["y"]
            t = track["t"] if "t" in track else range(len(x))

            for i in range(len(t) - 1):
                dx = x[i + 1] - x[i]
                dy = y[i + 1] - y[i]
                dt_frames = max(1, t[i + 1] - t[i])

                # Calculate distance in pixels and convert to µm
                distance_pixels = np.sqrt(dx**2 + dy**2)
                distance_um = distance_pixels * calibration

                # Convert time to hours and seconds
                dt_hours = dt_frames * hours_per_frame
                dt_seconds = (
                    dt_hours * 3600
                )  # Convert to seconds for velocity calculation

                # Calculate velocity in µm/s
                velocity = distance_um / dt_seconds

                # Store time in hours from start
                time_hours = t[i] * hours_per_frame

                if time_hours not in time_to_velocities:
                    time_to_velocities[time_hours] = []

                time_to_velocities[time_hours].append(velocity)

        # Calculate average at each time point
        times = sorted(time_to_velocities.keys())
        avg_velocities = [np.mean(time_to_velocities[t]) for t in times]

        # Plot the average
        self.velocity_ax.plot(
            times,
            avg_velocities,
            "k--",
            label="Population Average",
            linewidth=2,
            alpha=0.7,
        )

    def calculate_sliding_window_motility(self, track, window_size=10):
        """Calculate motility index over a sliding window"""
        times = track["t"] if "t" in track else range(len(track["x"]))
        x, y = track["x"], track["y"]
        motilities = []

        # Need at least window_size+1 points to calculate first window
        if len(x) <= window_size:
            # If track is shorter than window, just use whole track
            net_displacement = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)

            # Calculate path length
            dx, dy = np.diff(x), np.diff(y)
            step_distances = np.sqrt(dx**2 + dy**2)
            path_length = np.sum(step_distances)

            # Calculate motility index (simple version)
            motility = (net_displacement / path_length) * 100 if path_length > 0 else 0
            return [motility] * len(times)

        # For each time point, calculate motility using previous window_size points
        for i in range(window_size, len(times)):
            # Get window positions (including current point)
            window_x = x[i - window_size : i + 1]
            window_y = y[i - window_size : i + 1]

            # Calculate net displacement in window
            net_displacement = np.sqrt(
                (window_x[-1] - window_x[0]) ** 2 + (window_y[-1] - window_y[0]) ** 2
            )

            # Calculate path length in window
            window_dx, window_dy = np.diff(window_x), np.diff(window_y)
            window_steps = np.sqrt(window_dx**2 + window_dy**2)
            path_length = np.sum(window_steps)

            # Calculate motility index
            motility = (net_displacement / path_length) * 100 if path_length > 0 else 0
            motilities.append(motility)

        # Pad beginning with first calculated value
        padding = [motilities[0]] * window_size
        return padding + motilities

    def update_comparison_plots(self):
        """Update both motility and velocity plots for direct comparison"""
        # Get selected tracks
        selected_ids = self.get_selected_velocity_tracks()

        # Clear axes
        self.motility_ax.clear()
        self.velocity_comp_ax.clear()

        # Use calibration and actual time interval
        calibration = self.calibration
        hours_per_frame = self.hours_per_frame

        # Define sliding window size for motility calculation
        window_size = 5

        # Extract all motility indices for calculating normalization
        all_motility_indices = []
        if self.motility_metrics and "individual_metrics" in self.motility_metrics:
            all_motility_indices = [
                m.get("motility_index", 0)
                for m in self.motility_metrics["individual_metrics"]
            ]

        # Calculate statistics for normalization
        if all_motility_indices:
            mean_motility = np.mean(all_motility_indices)
            std_motility = np.std(all_motility_indices)

            # Use EXACTLY the same normalization calculation as in visualize_motility_with_chamber_regions
            min_val = max(0, mean_motility - 2.5 * std_motility)
            max_val = min(100, mean_motility + 2.5 * std_motility)

            # Ensure range is at least 20 to show variation
            if max_val - min_val < 20:
                center = (max_val + min_val) / 2
                min_val = max(0, center - 10)
                max_val = min(100, center + 10)

            # Create the normalizer
            norm = Normalize(vmin=min_val, vmax=max_val)
        else:
            # Default normalization if no data available
            norm = Normalize(vmin=0, vmax=100)

        # Plot motility and velocity for each selected track
        for track_id in selected_ids:
            track = next(
                (t for t in self.tracked_cells if t.get("ID") == track_id), None
            )
            if not track or "x" not in track or "y" not in track:
                continue

            # Find the motility index for this track
            track_motility = 50  # Default
            if self.motility_metrics and "individual_metrics" in self.motility_metrics:
                for metric in self.motility_metrics["individual_metrics"]:
                    if metric.get("track_id") == track_id:
                        track_motility = metric.get("motility_index", 50)
                        break

            # CRITICAL: Apply the EXACT SAME color mapping as in visualize_motility_with_chamber_regions
            color = plasma(norm(track_motility))

            # Rest of your code for calculating and plotting velocity/motility...
            x = track["x"]
            y = track["y"]
            t = track["t"] if "t" in track else range(len(x))

            # Calculate instantaneous velocities
            times_hours = []
            velocities_um_per_s = []

            for i in range(len(t) - 1):
                dx = x[i + 1] - x[i]
                dy = y[i + 1] - y[i]
                dt_frames = max(1, t[i + 1] - t[i])

                # Calculate distance and velocity
                distance_pixels = np.sqrt(dx**2 + dy**2)
                distance_um = distance_pixels * calibration
                dt_hours = dt_frames * hours_per_frame
                dt_seconds = dt_hours * 3600
                velocity = distance_um / dt_seconds

                # Store values
                time_hours = t[i] * hours_per_frame
                times_hours.append(time_hours)
                velocities_um_per_s.append(velocity)

            # Calculate sliding window motility values
            sliding_motility_values = self.calculate_sliding_window_motility(
                track, window_size
            )

            # Adjust lengths if necessary
            if len(sliding_motility_values) > len(times_hours):
                sliding_motility_values = sliding_motility_values[: len(times_hours)]

            # Plot motility and velocity with EXACTLY the same color
            self.motility_ax.plot(
                times_hours,
                sliding_motility_values,
                "-o",
                label=f"Track {track_id} (MI: {track_motility:.1f})",
                color=color,
                markersize=3,
                alpha=0.8,
            )

            self.velocity_comp_ax.plot(
                times_hours,
                velocities_um_per_s,
                "-o",
                label=f"Track {track_id} (MI: {track_motility:.1f})",
                color=color,
                markersize=3,
                alpha=0.8,
            )

        # Rest of your code for labels, titles, etc.
        self.motility_ax.set_xlabel("Time (hours)")
        self.motility_ax.set_ylabel("Motility Index (0-100)")
        self.motility_ax.set_title("Motility Index Over Time")
        self.motility_ax.grid(True, linestyle="--", alpha=0.7)

        self.velocity_comp_ax.set_xlabel("Time (hours)")
        self.velocity_comp_ax.set_ylabel("Velocity (µm/s)")
        self.velocity_comp_ax.set_title("Velocity Over Time")
        self.velocity_comp_ax.grid(True, linestyle="--", alpha=0.7)

        # Add legends if tracks are selected
        if selected_ids:
            self.motility_ax.legend(loc="upper right")
            self.velocity_comp_ax.legend(loc="upper right")

        # Redraw
        self.motility_fig.tight_layout()
        self.motility_canvas.draw()
        self.velocity_comp_fig.tight_layout()
        self.velocity_comp_canvas.draw()

    def create_density_analysis(self):
        """Create density-based region analysis with track selection prompt"""

        # Quick dialog to choose tracks
        dialog = QDialog(self)
        dialog.setWindowTitle("Density Analysis Options")
        dialog.setFixedSize(350, 150)

        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Choose data scope for density analysis:"))

        filtered_radio = QRadioButton(f"Filtered tracks ({len(self.tracked_cells)})")
        all_radio = QRadioButton(f"All tracks ({len(self.lineage_tracks)})")
        filtered_radio.setChecked(True)  # Default to current behavior

        layout.addWidget(filtered_radio)
        layout.addWidget(all_radio)

        # Buttons
        buttons = QHBoxLayout()
        ok_btn = QPushButton("Analyze")
        cancel_btn = QPushButton("Cancel")
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)

        if dialog.exec() != QDialog.Accepted:
            return  # User cancelled

        # Use selected tracks
        tracks_to_use = (
            self.lineage_tracks if all_radio.isChecked() else self.tracked_cells
        )

        # Create density analysis with selected tracks
        self.density_data = create_density_based_regions_from_forecast_data(
            self.all_cell_positions,
            self.chamber_dimensions,
            grid_size=50,
            tracks=tracks_to_use,  # Use user's selection
        )

        # Create visualization
        self.create_density_visualization()

    def create_density_visualization(self):
        """Create density-based region visualization matching forecast plot"""
        # Create new tab for density analysis
        self.density_tab = QWidget()
        self.density_layout = QVBoxLayout(self.density_tab)
        self.tab_widget.addTab(self.density_tab, "Density Regions (Forecast Data)")

        # Import and add animation button
        from ui.widgets.density_animation import add_animation_button_to_motility_widget
        add_animation_button_to_motility_widget(self)

        # Create matplotlib figure

        density_fig = plt.figure(figsize=(12, 8))
        density_ax = density_fig.add_subplot(111)

        # Plot density heatmap
        density_grid = self.density_data["density_grid"]
        x_bins = self.density_data["x_bins"]
        y_bins = self.density_data["y_bins"]

        # Create heatmap overlay
        im = density_ax.imshow(
            density_grid,
            extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
            origin="lower",
            cmap="viridis",
            alpha=0.6,
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=density_ax)
        cbar.set_label("Cell Count per Grid Block")

        # Overlay actual cell positions (recreate forecast plot)
        export_data = self.density_data["export_data"]
        all_x = [pos["x_position_pixels"] for pos in export_data]
        all_y = [pos["y_position_pixels"] for pos in export_data]

        # Calculate dynamic counts
        unique_cells = len(
            set(
                pos.get("cell_id")
                for pos in export_data
                if pos.get("cell_id") is not None
            )
        )  # Number of unique cells
        total_positions = len(export_data)  # Total position records

        density_ax.scatter(
            all_x,
            all_y,
            s=1,
            color="blue",
            alpha=0.2,
            label=f"Cell Tracking Data ({unique_cells} cells, {total_positions:,} position records)",
        )

        # Add grid lines to show density blocks
        for x in x_bins[::2]:  # Show every other line to avoid clutter
            density_ax.axvline(x, color="white", alpha=0.4, linewidth=0.5)
        for y in y_bins[::2]:
            density_ax.axhline(y, color="white", alpha=0.4, linewidth=0.5)

        # Labels and title
        density_ax.set_xlabel("X Position (pixels)")
        density_ax.set_ylabel("Y Position (pixels)")
        density_ax.set_title("Cell Density Map - Raw Data for Flow Simulations")
        density_ax.legend()

        # Set same limits as chamber
        chamber_dims = self.density_data["chamber_dimensions"]
        density_ax.set_xlim(0, chamber_dims[0])
        density_ax.set_ylim(0, chamber_dims[1])

        # Add canvas to layout
        density_canvas = FigureCanvas(density_fig)
        self.density_layout.addWidget(density_canvas)

        # CREATE BOTTOM PANEL WITH SUMMARY BUTTON
        bottom_panel = QHBoxLayout()

        # Add stretch to push button to the right
        bottom_panel.addStretch()

        # Summary and Export button
        summary_export_btn = QPushButton("Grid Analysis & Export Data")
        summary_export_btn.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)
        summary_export_btn.clicked.connect(self.show_density_summary_dialog)

        bottom_panel.addWidget(summary_export_btn)
        self.density_layout.addLayout(bottom_panel)

    def export_forecast_data(self):
        """Export with dialog to choose filtered vs all tracks, with optional morphology matching"""

        # Quick dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Options")
        dialog.setFixedSize(400, 220)

        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Choose export scope:"))

        filtered_radio = QRadioButton(f"Filtered tracks ({len(self.tracked_cells)})")
        all_radio = QRadioButton(f"All tracks ({len(self.lineage_tracks)})")
        filtered_radio.setChecked(True)

        layout.addWidget(filtered_radio)
        layout.addWidget(all_radio)

        # Add morphology option
        layout.addWidget(QLabel("\nInclude morphological features:"))
        morphology_checkbox = QCheckBox("Include morphology data (area, shape, health)")
        morphology_checkbox.setChecked(True)
        morphology_checkbox.setToolTip("Uses position-based matching to add biological features")
        layout.addWidget(morphology_checkbox)

        # Buttons
        buttons = QHBoxLayout()
        ok_btn = QPushButton("Export")
        cancel_btn = QPushButton("Cancel")
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)

        if dialog.exec() != QDialog.Accepted:
            return

        # Export based on choice
        tracks = self.lineage_tracks if all_radio.isChecked() else self.tracked_cells
        scope = "all" if all_radio.isChecked() else "filtered"
        include_morphology = morphology_checkbox.isChecked()

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Cell Tracking Data",
            f"enhanced_tracking_data_{scope}_tracks.csv",
            "CSV Files (*.csv)",
        )

        if file_path:
            try:
                # Show progress dialog for morphology matching
                if include_morphology:
                    from PySide6.QtWidgets import QProgressDialog
                    progress = QProgressDialog(
                        "Matching tracking data with morphological features...",
                        "Cancel", 0, 100, self)
                    progress.setWindowModality(Qt.WindowModal)
                    progress.show()

                # Get current position and channel for metrics lookup
                from nd2_analyzer.data.appstate import ApplicationState
                appstate = ApplicationState.get_instance()
                p = None
                c = 0
                if appstate and appstate.view_index:
                    _, p, c = appstate.view_index
                if p is None:
                    p = 0

                # Create enhanced data with morphological features
                export_data = []
                total_positions = sum(len(track['x']) for track in tracks)
                processed_positions = 0

                # Cache for metrics to avoid repeated queries
                metrics_cache = {}

                print(f"Starting export with morphology matching for {len(tracks)} tracks")
                print(f"Total positions to process: {total_positions}")
                print(f"Position: {p}, Channel: {c}")

                for track_idx, track in enumerate(tracks):
                    if include_morphology and progress.wasCanceled():
                        return

                    track_id = track.get('ID', -1)
                    x_coords = track['x']
                    y_coords = track['y']
                    t_coords = track.get('t', list(range(len(x_coords))))

                    # Get lineage information
                    parent_id = track.get('parent')
                    children_ids = track.get('children', [])

                    for i, (x, y, t) in enumerate(zip(x_coords, y_coords, t_coords)):
                        # Basic tracking data
                        row_data = {
                            'cell_id': track_id,
                            'time_point': t,
                            'position_in_track': i,
                            'x_pixels': x,
                            'y_pixels': y,
                            'x_um': x * 0.07,  # Default calibration
                            'y_um': y * 0.07,
                            'parent_id': parent_id,
                            'has_children': len(children_ids) > 0,
                            'children_ids': str(children_ids) if children_ids else '',
                            'track_length': len(x_coords),
                            'source': f'{scope}_tracks_with_morphology' if include_morphology else f'{scope}_tracks'
                        }

                        # Add morphological features if requested
                        if include_morphology:
                            morphology_data = self._find_matching_morphology(
                                x, y, t, p, metrics_cache, tolerance=15)

                            if morphology_data:
                                # Add all available morphological features
                                row_data.update({
                                    'morphology_cell_id': morphology_data.get('cell_id'),  # Original per-frame cell_id
                                    'centroid_x': morphology_data.get('centroid_x'),
                                    'centroid_y': morphology_data.get('centroid_y'),
                                    'area_pixels': morphology_data.get('area'),
                                    'perimeter_pixels': morphology_data.get('perimeter'),
                                    'aspect_ratio': morphology_data.get('aspect_ratio'),
                                    'circularity': morphology_data.get('circularity'),
                                    'solidity': morphology_data.get('solidity'),
                                    'eccentricity': morphology_data.get('eccentricity'),
                                    'major_axis_length': morphology_data.get('major_axis_length'),
                                    'minor_axis_length': morphology_data.get('minor_axis_length'),
                                    'equivalent_diameter': morphology_data.get('equivalent_diameter'),
                                    'orientation_radians': morphology_data.get('orientation'),
                                    'morphology_class': morphology_data.get('morphology_class'),
                                    'match_distance': morphology_data.get('_match_distance'),
                                    'bbox_y1': morphology_data.get('y1'),
                                    'bbox_x1': morphology_data.get('x1'),
                                    'bbox_y2': morphology_data.get('y2'),
                                    'bbox_x2': morphology_data.get('x2'),
                                    'fluorescence_channel': morphology_data.get('fluorescence_channel'),
                                    'fluo_level': morphology_data.get('fluo_level'),
                                })
                            else:
                                # No morphology match found - add None values
                                row_data.update({
                                    'morphology_cell_id': None,
                                    'centroid_x': None,
                                    'centroid_y': None,
                                    'area_pixels': None,
                                    'perimeter_pixels': None,
                                    'aspect_ratio': None,
                                    'circularity': None,
                                    'solidity': None,
                                    'eccentricity': None,
                                    'major_axis_length': None,
                                    'minor_axis_length': None,
                                    'equivalent_diameter': None,
                                    'orientation_radians': None,
                                    'morphology_class': None,
                                    'match_distance': None,
                                    'bbox_y1': None, 'bbox_x1': None, 'bbox_y2': None, 'bbox_x2': None,
                                    'fluorescence_channel': None,
                                    'fluo_level': None,
                                })

                        export_data.append(row_data)
                        processed_positions += 1

                        # Update progress
                        if include_morphology and processed_positions % 50 == 0:
                            progress_percent = int((processed_positions / total_positions) * 100)
                            progress.setValue(progress_percent)
                            QApplication.processEvents()

                if include_morphology:
                    progress.setValue(100)

                # Save to CSV
                df = pd.DataFrame(export_data)
                df.to_csv(file_path, index=False)

                # Calculate statistics
                total_positions = len(export_data)
                if include_morphology:
                    matched_positions = df['morphology_class'].notna().sum()
                    match_rate = (matched_positions / total_positions * 100) if total_positions > 0 else 0
                    avg_match_distance = df['match_distance'].mean() if 'match_distance' in df else 0

                    QMessageBox.information(
                        self,
                        "Export Complete",
                        f"Exported {len(tracks)} tracks with {total_positions} position records\n\n"
                        f"Morphology Matching Statistics:\n"
                        f"  • Matched: {matched_positions}/{total_positions} ({match_rate:.1f}%)\n"
                        f"  • Avg match distance: {avg_match_distance:.2f} pixels\n\n"
                        f"Includes: tracking, lineage, and morphology data!"
                    )
                else:
                    QMessageBox.information(
                        self,
                        "Export Complete",
                        f"Exported {len(tracks)} tracks with {total_positions} position records\n"
                        f"Each record includes cell_id, time_point, and lineage info"
                    )

            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Failed to export data: {str(e)}"
                )

    def _find_matching_morphology(self, track_x, track_y, track_t, position, cache, tolerance=15):
        """
        Find morphology metrics that match a track position using spatial-temporal proximity.

        Args:
            track_x, track_y: Track position coordinates
            track_t: Track time point
            position: Image position (P)
            cache: Dictionary to cache metrics queries
            tolerance: Maximum pixel distance for considering a match (default: 15px)

        Returns:
            dict: Morphology data if match found, None otherwise
        """
        # Create cache key
        cache_key = (track_t, position)

        # Check cache first
        if cache_key not in cache:
            try:
                # Query metrics for this time/position
                metrics_df = self.metrics_service.query_optimized(time=track_t, position=position)

                if not metrics_df.is_empty():
                    cache[cache_key] = metrics_df.to_pandas()
                else:
                    cache[cache_key] = pd.DataFrame()  # Empty dataframe

            except Exception as e:
                print(f"Error querying metrics for t={track_t}, p={position}: {e}")
                cache[cache_key] = pd.DataFrame()

        metrics_df = cache[cache_key]

        if metrics_df.empty:
            return None

        # Find closest cell by centroid distance
        best_match = None
        min_distance = float('inf')

        for _, row in metrics_df.iterrows():
            # Calculate distance between track position and cell centroid
            dx = track_x - row['centroid_x']
            dy = track_y - row['centroid_y']
            distance = np.sqrt(dx*dx + dy*dy)

            if distance < tolerance and distance < min_distance:
                min_distance = distance
                best_match = row.to_dict()
                best_match['_match_distance'] = distance  # Store match quality

        return best_match

    def export_density_grid(self):
        """Export density grid analysis"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Density Grid", "density_grid.csv", "CSV Files (*.csv)"
        )

        if file_path:
            df = pd.DataFrame(self.density_data["grid_export_data"])
            df.to_csv(file_path, index=False)

            QMessageBox.information(
                self,
                "Export Successful",
                f"Density grid data exported to {file_path}\n"
                f"Contains {len(df)} grid blocks with cell counts.",
            )

    def show_density_summary_dialog(self):
        """Show optimized summary dialog with better layout and spacing"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Grid-Based Analysis Summary")
        dialog.setMinimumSize(800, 700)
        dialog.setMaximumSize(1000, 900)

        # Clean, modern styling
        dialog.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
            }
            QLabel {
                color: #212529;
                background-color: transparent;
            }
        """)

        # Main layout with proper margins
        main_layout = QVBoxLayout(dialog)
        main_layout.setSpacing(25)
        main_layout.setContentsMargins(40, 30, 40, 30)

        # HEADER SECTION
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setSpacing(8)

        title = QLabel("Grid-Based Analysis")
        title.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 5px;
            }
        """)

        subtitle = QLabel("Spatial distribution analysis of cell positions")
        subtitle.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #6c757d;
                font-style: italic;
            }
        """)

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        main_layout.addWidget(header_widget)

        # DENSITY DISTRIBUTION SECTION
        density_section = QWidget()
        density_layout = QVBoxLayout(density_section)
        density_layout.setSpacing(15)

        density_title = QLabel("Density Distribution")
        density_title.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 10px;
            }
        """)
        density_layout.addWidget(density_title)

        # Get the actual data
        grid_data = self.density_data["grid_export_data"]
        total_cells = self.density_data["total_cells"]

        # Calculate region counts
        region_counts = {}
        for block in grid_data:
            region_type = block["region_type"]
            region_counts[region_type] = region_counts.get(region_type, 0) + 1

        # Create density cards in a 2x2 grid for better space usage
        cards_container = QWidget()
        cards_grid = QGridLayout(cards_container)
        cards_grid.setSpacing(12)

        density_configs = [
            ("High-Density", "#e74c3c", "#ffffff", 0, 0),  # Red, top-left
            ("Medium-Density", "#f39c12", "#ffffff", 0, 1),  # Orange, top-right
            ("Low-Density", "#3498db", "#ffffff", 1, 0),  # Blue, bottom-left
            ("Empty", "#95a5a6", "#ffffff", 1, 1),  # Gray, bottom-right
        ]

        for density_type, bg_color, text_color, row, col in density_configs:
            if density_type in region_counts:
                count = region_counts[density_type]
                percentage = (count / len(grid_data)) * 100 if grid_data else 0

                # Create compact card
                card = QWidget()
                card.setFixedHeight(80)
                card.setStyleSheet(f"""
                    QWidget {{
                        background-color: {bg_color};
                        border: none;
                        border-radius: 10px;
                    }}
                """)

                card_layout = QHBoxLayout(card)
                card_layout.setContentsMargins(20, 15, 20, 15)

                # Left side: type name with bullet
                name_label = QLabel(f"● {density_type}")
                name_label.setStyleSheet(f"""
                    QLabel {{
                        font-size: 15px;
                        font-weight: bold;
                        color: {text_color};
                    }}
                """)
                card_layout.addWidget(name_label)

                card_layout.addStretch()

                # Right side: stats
                stats_widget = QWidget()
                stats_layout = QVBoxLayout(stats_widget)
                stats_layout.setContentsMargins(0, 0, 0, 0)
                stats_layout.setSpacing(2)

                count_label = QLabel(f"{count} blocks")
                count_label.setStyleSheet(f"""
                    QLabel {{
                        font-size: 15px;
                        font-weight: bold;
                        color: {text_color};
                    }}
                """)
                count_label.setAlignment(Qt.AlignRight)

                percent_label = QLabel(f"({percentage:.1f}%)")
                percent_label.setStyleSheet(f"""
                    QLabel {{
                        font-size: 13px;
                        color: {text_color};
                    }}
                """)
                percent_label.setAlignment(Qt.AlignRight)

                stats_layout.addWidget(count_label)
                stats_layout.addWidget(percent_label)
                card_layout.addWidget(stats_widget)

                cards_grid.addWidget(card, row, col)

        density_layout.addWidget(cards_container)
        main_layout.addWidget(density_section)

        # SUMMARY STATISTICS SECTION
        stats_section = QWidget()
        stats_layout = QVBoxLayout(stats_section)
        stats_layout.setSpacing(15)

        stats_title = QLabel("Summary Statistics")
        stats_title.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 10px;
            }
        """)
        stats_layout.addWidget(stats_title)

        # Create 2x2 grid for stats
        stats_container = QWidget()
        stats_grid = QGridLayout(stats_container)
        stats_grid.setSpacing(15)

        # Define the 4 stat cards
        export_data = self.density_data["export_data"]
        unique_cells = len(
            set(
                pos.get("cell_id")
                for pos in export_data
                if pos.get("cell_id") is not None
            )
        )
        total_positions = len(export_data)

        # Define the 4 stat cards with correct labels
        stat_configs = [
            (f"{unique_cells:,}", "Unique Cells", "#3498db", 0, 0),
            (f"{total_positions:,}", "Position Records", "#2ecc71", 0, 1),
            ("50px", "Block Size", "#9b59b6", 1, 0),
            ("4", "Density Types", "#e67e22", 1, 1),
        ]

        for value, label, color, row, col in stat_configs:
            stat_card = QWidget()
            stat_card.setFixedSize(160, 90)
            stat_card.setStyleSheet(f"""
                QWidget {{
                    background-color: #ffffff;
                    border: 2px solid {color};
                    border-radius: 8px;
                }}
            """)

            stat_layout_inner = QVBoxLayout(stat_card)
            stat_layout_inner.setAlignment(Qt.AlignCenter)
            stat_layout_inner.setSpacing(5)

            # Value
            value_label = QLabel(value)
            value_label.setStyleSheet(f"""
                QLabel {{
                    font-size: 22px;
                    font-weight: bold;
                    color: {color};
                }}
            """)
            value_label.setAlignment(Qt.AlignCenter)

            # Description
            desc_label = QLabel(label)
            desc_label.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    color: #6c757d;
                    font-weight: normal;
                    padding: 8px 15px;
                    border-radius: 12px;
                    background-color: rgba(255, 255, 255, 0.8);
                }
            """)
            desc_label.setAlignment(Qt.AlignCenter)

            stat_layout_inner.addWidget(value_label)
            stat_layout_inner.addWidget(desc_label)

            stats_grid.addWidget(stat_card, row, col)

        stats_layout.addWidget(stats_container)
        main_layout.addWidget(stats_section)

        # Add some spacing before buttons
        main_layout.addSpacing(10)

        # EXPORT AND CLOSE BUTTONS
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        export_raw_btn = QPushButton("📊 Export Cell Positions")
        export_raw_btn.setFixedHeight(45)
        export_raw_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: #ffffff;
                border: none;
                padding: 12px 25px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
        """)
        export_raw_btn.clicked.connect(self.export_forecast_data)

        export_grid_btn = QPushButton("🗂️ Export Density Grid")
        export_grid_btn.setFixedHeight(45)
        export_grid_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: #ffffff;
                border: none;
                padding: 12px 25px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        export_grid_btn.clicked.connect(self.export_density_grid)

        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(45)
        close_btn.setFixedWidth(100)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: #ffffff;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:pressed {
                background-color: #495057;
            }
        """)
        close_btn.clicked.connect(dialog.accept)

        button_layout.addWidget(export_raw_btn)
        button_layout.addWidget(export_grid_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)

        main_layout.addLayout(button_layout)

        # Show dialog
        dialog.exec()
