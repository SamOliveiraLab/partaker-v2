# motility_widget.py
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QTabWidget, QWidget, QMessageBox, QProgressDialog,
                             QFileDialog, QCheckBox, QDialogButtonBox, QTextEdit, QLineEdit,
                             QTableWidget, QTableWidgetItem, QComboBox)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import plasma
from matplotlib.colors import Normalize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from skimage.measure import label, regionprops
import pandas as pd
from pubsub import pub

# Modify in motility_widget.py

class MotilityDialog(QDialog):
    """
    Dialog for analyzing and visualizing cell motility.
    """
    
    def __init__(self, tracked_cells, lineage_tracks, image_data=None, parent=None):
        try:
            print("DEBUG: MotilityDialog.__init__ starting...")
            super().__init__(parent)
            print("DEBUG: Super().__init__ completed")
            
            self.tracked_cells = tracked_cells
            self.lineage_tracks = lineage_tracks
            self.image_data = image_data  # Store image data for accessing segmentation cache
            self.motility_metrics = None
            
            print(f"DEBUG: tracked_cells: {len(tracked_cells) if tracked_cells else 0}")
            print(f"DEBUG: lineage_tracks: {len(lineage_tracks) if lineage_tracks else 0}")
            
            self.calibration = 0.07
            
            # Set dialog properties
            print("DEBUG: Setting dialog properties...")
            self.setWindowTitle("Cell Motility Analysis")
            self.setMinimumWidth(1200)  # Increased width for the velocity tab
            self.setMinimumHeight(800)  # Increased height for the velocity tab
            print("DEBUG: Dialog properties set")
            
            # Initialize UI
            print("DEBUG: Initializing UI...")
            self.init_ui()
            print("DEBUG: UI initialized")
            
            # Start analysis
            print("DEBUG: Starting analysis...")
            try:
                self.analyze_motility()
                print("DEBUG: Analysis completed")
            except Exception as analysis_error:
                print(f"DEBUG: Analysis failed: {analysis_error}")
                import traceback
                traceback.print_exc()
                
                # Show error dialog instead of crashing
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Motility Analysis Error", 
                    f"Motility analysis failed:\n{str(analysis_error)}\n\nSee console for details.")
                
                # Still show the dialog but without analysis results
                print("DEBUG: Showing dialog without analysis results")
            
        except Exception as e:
            print(f"DEBUG: Error in MotilityDialog.__init__: {e}")
            import traceback
            traceback.print_exc()
            raise
    
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
    
    def analyze_motility(self):
        """Analyze cell motility"""
        print("DEBUG: analyze_motility() method called")
        
        # Skip cursor change to avoid Qt threading issues
        print("DEBUG: Skipping cursor change to avoid threading issues")
        
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
                    print(f"Using chamber dimensions from image data: {chamber_dimensions}")
            
            # Collect all cell positions
            all_cell_positions = self.collect_cell_positions(p, c)
            print(f"Collected {len(all_cell_positions)} cell positions for visualization")
            
            # Calculate motility metrics
            print("DEBUG: Starting motility analysis...")
            print(f"DEBUG: tracked_cells type: {type(self.tracked_cells)}")
            print(f"DEBUG: tracked_cells length: {len(self.tracked_cells) if self.tracked_cells else 0}")
            
            try:
                print("DEBUG: Importing enhanced_motility_index...")
                from tracking import enhanced_motility_index
                print("DEBUG: Import successful")
                
                print("DEBUG: Calculating motility metrics...")
                self.motility_metrics = enhanced_motility_index(
                    self.tracked_cells, chamber_dimensions)
                print("DEBUG: Motility metrics calculated successfully")
                
            except Exception as e:
                print(f"DEBUG: Error in motility calculation: {e}")
                raise
            
            try:
                print("DEBUG: Creating visualizations...")
                # Create visualizations
                self.create_visualizations(chamber_dimensions, all_cell_positions)
                print("DEBUG: Visualizations created successfully")
                
            except Exception as e:
                print(f"DEBUG: Error in visualization: {e}")
                raise
            
            try:
                print("DEBUG: Updating summary...")
                # Update summary
                self.update_summary()
                print("DEBUG: Summary updated successfully")
                
            except Exception as e:
                print(f"DEBUG: Error in summary update: {e}")
                raise
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Failed to analyze motility: {str(e)}")
    
    def collect_cell_positions(self, p, c):
        """Collect cell positions from segmentation cache or tracks"""
        all_cell_positions = []
        
        # Try to use segmentation cache if available
        if self.image_data and hasattr(self.image_data, 'segmentation_cache'):
            try:
                # Determine number of time frames
                t_max = 20
                if hasattr(self.image_data, 'data'):
                    t_max = min(20, self.image_data.data.shape[0])
                
                for t in range(t_max):
                    try:
                        binary_image = self.image_data.segmentation_cache[t, p, c]
                        if binary_image is not None:
                            labeled_image = label(binary_image)
                            regions = regionprops(labeled_image)
                            for region in regions:
                                y, x = region.centroid
                                all_cell_positions.append((x, y))
                        else:
                            print(f"Frame {t}: No binary image found")
                    except Exception as frame_error:
                        print(f"Error processing frame {t}: {str(frame_error)}")
                
            except Exception as e:
                print(f"Error collecting cell positions from segmentation: {str(e)}")
                all_cell_positions = []
        
        # Fall back to using tracks if needed
        if not all_cell_positions:
            print(f"Falling back to collecting positions from tracks")
            for track in self.lineage_tracks:
                if 'x' in track and 'y' in track:
                    all_cell_positions.extend(list(zip(track['x'], track['y'])))
            print(f"Collected {len(all_cell_positions)} cell positions from tracks")
        
        return all_cell_positions
    
    def create_visualizations(self, chamber_dimensions, all_cell_positions):
        """Create motility visualizations"""
        from nd2_analyzer.analysis.tracking.tracking import (visualize_motility_with_chamber_regions, visualize_motility_map)
        
        # Store chamber dimensions for later use
        self.chamber_dimensions = chamber_dimensions
        self.all_cell_positions = all_cell_positions
        
        # Combined visualization
        combined_fig, self.combined_ax = visualize_motility_with_chamber_regions(
            self.tracked_cells, all_cell_positions, chamber_dimensions, self.motility_metrics)
        combined_canvas = FigureCanvas(combined_fig)
        self.combined_layout.addWidget(combined_canvas)
        self.combined_fig = combined_fig
        self.combined_canvas = combined_canvas
        
        # Populate track selector
        self.populate_track_selector()
        self.populate_velocity_track_list()
        
        # Motility map
        map_fig, _ = visualize_motility_map(
            self.tracked_cells, chamber_dimensions, self.motility_metrics)
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
            if 'ID' in track:
                track_ids.add(track['ID'])
        
        # Create a lookup for metrics (if available)
        metrics_lookup = {}
        if self.motility_metrics and 'individual_metrics' in self.motility_metrics:
            metrics_lookup = {m.get('track_id'): m for m in self.motility_metrics['individual_metrics']}
        
        # Sort track IDs numerically
        sorted_ids = sorted(track_ids)
        
        # Add each track to combo box
        for track_id in sorted_ids:
            # Check if we have metrics for this track
            if track_id in metrics_lookup:
                metric = metrics_lookup[track_id]
                motility = metric.get('motility_index', 0)
                self.track_combo.addItem(f"Track {track_id} (MI: {motility:.1f})", track_id)
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
        selected_track = next((t for t in self.tracked_cells if t.get('ID', -1) == track_id), None)
        
        # Clear any existing highlight
        self.clear_track_highlight()
        
        if selected_track and 'x' in selected_track and 'y' in selected_track:
            # Highlight the track
            self.highlighted_line = self.combined_ax.plot(
                selected_track['x'], selected_track['y'], '-',
                linewidth=3, color='red', zorder=100)[0]
            
            # Add start/end markers
            self.highlighted_start = self.combined_ax.plot(
                selected_track['x'][0], selected_track['y'][0], 'o',
                markersize=8, color='red', zorder=100)[0]
            
            self.highlighted_end = self.combined_ax.plot(
                selected_track['x'][-1], selected_track['y'][-1], 's',
                markersize=8, color='red', zorder=100)[0]
            
            # Prepare info text with available data
            info_text = [f"Track ID: {track_id}"]
            
            # Add track length
            if 'x' in selected_track:
                info_text.append(f"Track Length: {len(selected_track['x'])} frames")
            
            # Calculate path length (if not available)
            if 'x' in selected_track and 'y' in selected_track and len(selected_track['x']) > 1:
                # Calculate path length
                path_length = 0
                for i in range(len(selected_track['x']) - 1):
                    dx = selected_track['x'][i+1] - selected_track['x'][i]
                    dy = selected_track['y'][i+1] - selected_track['y'][i]
                    path_length += np.sqrt(dx**2 + dy**2)
                info_text.append(f"Path Length: {path_length:.1f} px")
            
            # Look up motility metrics for this track (if available)
            if self.motility_metrics and 'individual_metrics' in self.motility_metrics:
                track_metric = next((m for m in self.motility_metrics['individual_metrics'] 
                                if m.get('track_id') == track_id), None)
                
                if track_metric:
                    # Add motility metrics
                    info_text.append(f"Motility Index: {track_metric.get('motility_index', 0):.1f}")
                    info_text.append(f"Avg Velocity: {track_metric.get('avg_velocity', 0):.2f} px/frame")
                    info_text.append(f"Confinement: {track_metric.get('confinement_ratio', 0):.2f}")
                    info_text.append(f"Persistence: {track_metric.get('directional_persistence', 0):.2f}")
            
            # Add text near track end
            self.highlighted_text = self.combined_ax.text(
                selected_track['x'][-1] + 10, selected_track['y'][-1] + 10,
                "\n".join(info_text), 
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'),
                zorder=100)
            
            # Update the canvas
            self.combined_canvas.draw()
        
    
    def clear_track_highlight(self):
        """Clear the highlighted track"""
        # Remove any existing highlight elements
        if hasattr(self, 'highlighted_line') and self.highlighted_line:
            self.highlighted_line.remove()
            self.highlighted_line = None
        
        if hasattr(self, 'highlighted_start') and self.highlighted_start:
            self.highlighted_start.remove()
            self.highlighted_start = None
        
        if hasattr(self, 'highlighted_end') and self.highlighted_end:
            self.highlighted_end.remove()
            self.highlighted_end = None
        
        if hasattr(self, 'highlighted_text') and self.highlighted_text:
            self.highlighted_text.remove()
            self.highlighted_text = None
        
        # Update the canvas
        if hasattr(self, 'combined_canvas'):
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
        
        export_buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        export_buttons.accepted.connect(export_dialog.accept)
        export_buttons.rejected.connect(export_dialog.reject)
        export_layout.addWidget(export_buttons)
        
        if export_dialog.exec() == QDialog.Accepted:
            save_path, _ = QFileDialog.getSaveFileName(
                export_dialog, "Save Results", "", "All Files (*)")
                
            if save_path:
                base_path = save_path.replace(".png", "").replace(".csv", "")
                
                if export_map.isChecked():
                    self.map_fig.savefig(
                        f"{base_path}_motility_map.png", dpi=300, bbox_inches='tight')
                        
                if export_metrics.isChecked():
                    self.metrics_fig.savefig(
                        f"{base_path}_detailed_metrics.png", dpi=300, bbox_inches='tight')
                        
                if export_csv.isChecked():
                    metrics_df = pd.DataFrame(self.motility_metrics['individual_metrics'])
                    metrics_df.to_csv(f"{base_path}_motility_metrics.csv", index=False)
                    
                QMessageBox.information(
                    export_dialog, "Export Complete",
                    f"Results exported to {base_path}_*.png/csv")
                

    def populate_velocity_track_list(self):
        """Populate the track list for velocity-time analysis with motility information"""
        self.velocity_track_list.setRowCount(0)
        
        # Create a lookup for metrics
        metrics_lookup = {}
        if self.motility_metrics and 'individual_metrics' in self.motility_metrics:
            metrics_lookup = {m.get('track_id'): m for m in self.motility_metrics['individual_metrics']}
        
        # Get tracks with metrics
        valid_tracks = []
        for track in self.tracked_cells:
            if 'ID' in track and track['ID'] in metrics_lookup:
                valid_tracks.append((track, metrics_lookup[track['ID']]))
        
        # Sort by motility index
        valid_tracks.sort(key=lambda item: item[1].get('motility_index', 0), reverse=True)
        
        # Add to table with three columns: ID, Motility, Show
        self.velocity_track_list.setColumnCount(3)
        self.velocity_track_list.setHorizontalHeaderLabels(["Track ID", "Motility", "Show"])
        
        # Add to table
        self.velocity_track_list.setRowCount(len(valid_tracks))
        
        # Use the same colormap for consistent visualization
        from matplotlib.cm import plasma
        from matplotlib.colors import to_hex
        
        for i, (track, metrics) in enumerate(valid_tracks):
            # Track ID
            id_item = QTableWidgetItem(str(track['ID']))
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
            self.velocity_track_list.setItem(i, 0, id_item)
            
            # Motility Index
            motility_value = metrics.get('motility_index', 0)
            motility_item = QTableWidgetItem(f"{motility_value:.1f}")
            motility_item.setFlags(motility_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
            
            # Set background color based on motility (matching the visualization color)
            color_value = plasma(motility_value / 100)
            # Convert to hex color for QTableWidgetItem
            hex_color = to_hex(color_value)
            motility_item.setBackground(QColor(hex_color))
            
            # Make text white or black depending on background brightness for readability
            # A simple heuristic: if the sum of RGB values is less than 384 (avg 128 per channel), use white text
            r, g, b = [int(255 * c) for c in color_value[:3]]
            if r + g + b < 384:
                motility_item.setForeground(QColor('white'))
            
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
            if self.velocity_track_list.item(i, 2).checkState() == Qt.Checked:  # Changed from column 1 to 2
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
        
        # Assume frames are 60 minutes apart (1 hour)
        # You can adjust this value based on your actual time between frames
        hours_per_frame = 1
        
        # Add a text box with summary statistics
        if selected_ids:
            self.add_summary_stats_box(selected_ids, hours_per_frame)
        
        # Calculate instantaneous velocities for each selected track
        for track_id in selected_ids:
            track = next((t for t in self.tracked_cells if t.get('ID') == track_id), None)
            if not track or 'x' not in track or 'y' not in track:
                continue
                
            # Calculate instantaneous velocities
            times_hours = []
            velocities_um_per_s = []
            
            x = track['x']
            y = track['y']
            t = track['t'] if 't' in track else range(len(x))
            
            for i in range(len(t) - 1):
                dx = x[i+1] - x[i]
                dy = y[i+1] - y[i]
                dt_frames = max(1, t[i+1] - t[i])  # Prevent division by zero
                
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
            from matplotlib.cm import viridis
            color = viridis(selected_ids.index(track_id) / max(1, len(selected_ids)))
            self.velocity_ax.plot(times_hours, velocities_um_per_s, '-o', label=f"Track {track_id}", 
                                color=color, markersize=3, alpha=0.8)
            
            # Add division markers if enabled
            if self.show_divisions_checkbox.isChecked() and 'children' in track and track['children']:
                last_time = t[-1] * hours_per_frame
                if velocities_um_per_s:  # Make sure we have velocity data
                    last_velocity = velocities_um_per_s[-1] if len(velocities_um_per_s) == len(times_hours) else 0
                    self.velocity_ax.plot(last_time, last_velocity, 'r*', markersize=10, 
                                        label=f"Division (Track {track_id})")
        
        # Add population average if requested
        if self.show_average_checkbox.isChecked() and selected_ids:
            self.add_population_average(hours_per_frame)
        
        # Set labels and title
        self.velocity_ax.set_xlabel('Time (hours)')
        self.velocity_ax.set_ylabel('Velocity (µm/s)')
        self.velocity_ax.set_title('Cell Velocity vs. Time')
        
        # Add grid and legend
        self.velocity_ax.grid(True, linestyle='--', alpha=0.7)
        if selected_ids:
            self.velocity_ax.legend(loc='upper right')
        
        # Redraw
        self.velocity_fig.tight_layout()
        self.velocity_canvas.draw()

    def add_summary_stats_box(self, selected_ids, hours_per_frame=1):
        
        calibration = self.calibration
        
        """Add a box with summary statistics for selected tracks"""
        stats_text = []
        
        for track_id in selected_ids:
            track = next((t for t in self.tracked_cells if t.get('ID') == track_id), None)
            if not track or 'x' not in track or 'y' not in track:
                continue
                
            # Calculate average velocity in both units
            x = track['x']
            y = track['y']
            t = track['t'] if 't' in track else range(len(x))
            
            total_distance_px = 0
            for i in range(len(x) - 1):
                dx = x[i+1] - x[i]
                dy = y[i+1] - y[i]
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
            if self.motility_metrics and 'individual_metrics' in self.motility_metrics:
                track_metric = next((m for m in self.motility_metrics['individual_metrics'] 
                                if m.get('track_id') == track_id), None)
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
                0.02, 0.98, 
                "\n\n".join(stats_text),
                transform=self.velocity_ax.transAxes,
                va='top', ha='left',
                bbox=dict(
                    boxstyle='round',
                    facecolor='white',
                    alpha=0.8,
                    edgecolor='gray'
                ),
                fontsize=9
            )
            
    def add_population_average(self, hours_per_frame=1):
        """Add population average velocity to the plot"""
        # Use self.calibration instead of default
        calibration = self.calibration
        # Collect all velocities across all time points
        time_to_velocities = {}
        
        for track in self.tracked_cells:
            if 'x' not in track or 'y' not in track:
                continue
                
            x = track['x']
            y = track['y']
            t = track['t'] if 't' in track else range(len(x))
            
            for i in range(len(t) - 1):
                dx = x[i+1] - x[i]
                dy = y[i+1] - y[i]
                dt_frames = max(1, t[i+1] - t[i])
                
                # Calculate distance in pixels and convert to µm
                distance_pixels = np.sqrt(dx**2 + dy**2)
                distance_um = distance_pixels * calibration
                
                # Convert time to hours and seconds
                dt_hours = dt_frames * hours_per_frame
                dt_seconds = dt_hours * 3600  # Convert to seconds for velocity calculation
                
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
        self.velocity_ax.plot(times, avg_velocities, 'k--', 
                            label="Population Average", linewidth=2, alpha=0.7)
        
        
    def calculate_sliding_window_motility(self, track, window_size=10):
        """Calculate motility index over a sliding window"""
        times = track['t'] if 't' in track else range(len(track['x']))
        x, y = track['x'], track['y']
        motilities = []
        
        # Need at least window_size+1 points to calculate first window
        if len(x) <= window_size:
            # If track is shorter than window, just use whole track
            net_displacement = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
            
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
            window_x = x[i-window_size:i+1]
            window_y = y[i-window_size:i+1]
            
            # Calculate net displacement in window
            net_displacement = np.sqrt((window_x[-1] - window_x[0])**2 + 
                                    (window_y[-1] - window_y[0])**2)
            
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
        
        # Use calibration
        calibration = self.calibration
        hours_per_frame = 1  # Assume 1 hour between frames
        
        # Define sliding window size for motility calculation
        window_size = 5  
        
        # Extract all motility indices for calculating normalization
        all_motility_indices = []
        if self.motility_metrics and 'individual_metrics' in self.motility_metrics:
            all_motility_indices = [m.get('motility_index', 0) for m in self.motility_metrics['individual_metrics']]
        
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
            track = next((t for t in self.tracked_cells if t.get('ID') == track_id), None)
            if not track or 'x' not in track or 'y' not in track:
                continue
            
            # Find the motility index for this track
            track_motility = 50  # Default
            if self.motility_metrics and 'individual_metrics' in self.motility_metrics:
                for metric in self.motility_metrics['individual_metrics']:
                    if metric.get('track_id') == track_id:
                        track_motility = metric.get('motility_index', 50)
                        break
            
            # CRITICAL: Apply the EXACT SAME color mapping as in visualize_motility_with_chamber_regions
            color = plasma(norm(track_motility))
            
            # Rest of your code for calculating and plotting velocity/motility...
            x = track['x']
            y = track['y']
            t = track['t'] if 't' in track else range(len(x))
            
            # Calculate instantaneous velocities
            times_hours = []
            velocities_um_per_s = []
            
            for i in range(len(t) - 1):
                dx = x[i+1] - x[i]
                dy = y[i+1] - y[i]
                dt_frames = max(1, t[i+1] - t[i])
                
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
            sliding_motility_values = self.calculate_sliding_window_motility(track, window_size)
            
            # Adjust lengths if necessary
            if len(sliding_motility_values) > len(times_hours):
                sliding_motility_values = sliding_motility_values[:len(times_hours)]
            
            # Plot motility and velocity with EXACTLY the same color
            self.motility_ax.plot(times_hours, sliding_motility_values, '-o', 
                        label=f"Track {track_id} (MI: {track_motility:.1f})", 
                        color=color, markersize=3, alpha=0.8)
            
            self.velocity_comp_ax.plot(times_hours, velocities_um_per_s, '-o', 
                            label=f"Track {track_id} (MI: {track_motility:.1f})", 
                            color=color, markersize=3, alpha=0.8)
        
        # Rest of your code for labels, titles, etc.
        self.motility_ax.set_xlabel('Time (hours)')
        self.motility_ax.set_ylabel('Motility Index (0-100)')
        self.motility_ax.set_title('Motility Index Over Time')
        self.motility_ax.grid(True, linestyle='--', alpha=0.7)
        
        self.velocity_comp_ax.set_xlabel('Time (hours)')
        self.velocity_comp_ax.set_ylabel('Velocity (µm/s)')
        self.velocity_comp_ax.set_title('Velocity Over Time')
        self.velocity_comp_ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legends if tracks are selected
        if selected_ids:
            self.motility_ax.legend(loc='upper right')
            self.velocity_comp_ax.legend(loc='upper right')
        
        # Redraw
        self.motility_fig.tight_layout()
        self.motility_canvas.draw()
        self.velocity_comp_fig.tight_layout()
        self.velocity_comp_canvas.draw()