from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                               QComboBox, QTableWidget, QTableWidgetItem, 
                               QMessageBox, QFileDialog, QTabWidget, QDialog)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pubsub import pub
import cv2

from nd2_analyzer.analysis.morphology.morphology import (classify_morphology, extract_cells_and_metrics,
                       annotate_binary_mask, extract_cell_morphologies)

class MorphologyWidget(QWidget):
    def __init__(self, parent=None, metrics_service=None):
        super().__init__(parent)
        self.cell_mapping = {}
        self.selected_cell_id = None
        self.tracked_cell_lineage = {}
        
        # Store reference to metrics service
        self.metrics_service = metrics_service
        
        # Set up morphology colors (same as original)
        self.morphology_colors = {
            "Artifact": (128, 128, 128),  # Gray
            "Divided": (255, 0, 0),       # Blue
            "Healthy": (0, 255, 0),       # Green
            "Elongated": (0, 255, 255),   # Yellow
            "Deformed": (255, 0, 255),    # Magenta
        }
        
        self.morphology_colors_rgb = {
            key: (color[2] / 255, color[1] / 255, color[0] / 255)
            for key, color in self.morphology_colors.items()
        }
        
        # Initialize UI components
        self.initUI()
        
        # Subscribe to relevant messages
        pub.subscribe(self.on_cell_mapping_updated, "cell_mapping_updated")
        pub.subscribe(self.on_segmentation_ready, "segmentation_ready")
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")
        pub.subscribe(self.on_current_view_changed, "current_view_changed")
        pub.subscribe(self.set_tracking_data, "tracking_data_available")
        pub.subscribe(self.process_morphology_time_series, "process_morphology_time_series_requested")
        pub.subscribe(self.provide_cell_mapping, "get_cell_mapping")

    def provide_cell_mapping(self, time, position, channel, cell_id, callback):
        """
        Provide cell mapping data to other components upon request
        
        Args:
            time: Current time point
            position: Current position
            channel: Current channel
            cell_id: ID of the cell to provide mapping for
            callback: Function to receive the mapping data
        """
        print(f"MorphologyWidget: provide_cell_mapping called, have {len(self.cell_mapping)} cells")
        
        if not self.cell_mapping:
            print("No cell mapping data available")
            callback(None)
            return
        
        # If cell_id is specified, check if it exists
        if cell_id is not None and cell_id not in self.cell_mapping:
            print(f"Cell ID {cell_id} not found in cell mapping")
            callback(None)
            return
        
        # Return the mapping (full or filtered by cell_id)
        if cell_id is not None:
            callback({cell_id: self.cell_mapping[cell_id]})
        else:
            callback(self.cell_mapping)
    
    def initUI(self):
        layout = QVBoxLayout(self)
        
        buttons_layout = QHBoxLayout()
    
        buttons_layout.addSpacing(20)
        
        # Add the classify cells button
        self.fetch_metrics_button = QPushButton("Classify Cells")
        self.fetch_metrics_button.clicked.connect(self.fetch_metrics_from_service)
        self.fetch_metrics_button.setStyleSheet("background-color: black; color: white; font-weight: bold;")
        self.fetch_metrics_button.setFixedHeight(40) 
        buttons_layout.addWidget(self.fetch_metrics_button)
        
        buttons_layout.addSpacing(20)
        
        # Add Process Morphology Over Time button
        self.process_morphology_button = QPushButton("Process Morphology Over Time")
        self.process_morphology_button.clicked.connect(self.process_morphology_time_series)
        self.process_morphology_button.setStyleSheet("background-color: black; color: white; font-weight: bold;")
        self.process_morphology_button.setFixedHeight(40)  
        buttons_layout.addWidget(self.process_morphology_button)
        
        buttons_layout.addSpacing(20)
    
        layout.addLayout(buttons_layout)
    
        
        # Create QTabWidget for inner tabs
        inner_tab_widget = QTabWidget()
        self.scatter_tab = QWidget()
        self.table_tab = QWidget()
        
        # Add tabs to the inner tab widget
        inner_tab_widget.addTab(self.scatter_tab, "PCA Plot")
        inner_tab_widget.addTab(self.table_tab, "Metrics Table")
        
        scatter_layout = QVBoxLayout(self.scatter_tab)

        button_layout = QHBoxLayout()
        button_layout.addStretch(1) 
        self.save_plot_button = QPushButton("Save Plot")
        self.save_plot_button.setStyleSheet("background-color: white; color: black; font-size: 12px;")
        self.save_plot_button.clicked.connect(self.save_pca_plot)
        self.save_plot_button.setFixedWidth(120)
        button_layout.addWidget(self.save_plot_button)
        scatter_layout.addLayout(button_layout)

        # PCA scatter plot display 
        self.figure_annot_scatter = plt.figure()
        self.canvas_annot_scatter = FigureCanvas(self.figure_annot_scatter)
        scatter_layout.addWidget(self.canvas_annot_scatter, 1)
        
        # Metrics table tab
        table_layout = QVBoxLayout(self.table_tab)
        
        self.button_layout = QHBoxLayout()
        
        # Export to CSV button
        self.export_button = QPushButton("Export to CSV")
        self.export_button.setStyleSheet("background-color: white; color: black; font-size: 14px;")
        self.export_button.clicked.connect(self.export_metrics_to_csv)
        self.button_layout.addWidget(self.export_button)
        
        # Save GIF button (initially hidden, will appear after cell selection)
        self.save_gif_button = None
        
        table_layout.addLayout(self.button_layout)
        
        # Create metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.itemClicked.connect(self.on_table_item_click)
        table_layout.addWidget(self.metrics_table)
        
        layout.addWidget(inner_tab_widget)
        
    def save_pca_plot(self):
        """Save the PCA scatter plot to a file"""
        if not hasattr(self, "figure_annot_scatter"):
            QMessageBox.warning(self, "Error", "No plot available to save.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save PCA Plot", "", "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)")
        
        if file_path:
            try:
                self.figure_annot_scatter.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(
                    self, "Success", f"PCA plot saved to {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save plot: {str(e)}")
            
    
    def fetch_metrics_from_service(self):
        """Fetch cell metrics and classification from the metrics service dataframe and annotate image"""
        if not self.metrics_service:
            QMessageBox.warning(self, "Error", "Metrics service not available.")
            return
            
        try:
            # Get current frame information
            t = pub.sendMessage("get_current_t", default=0)
            p = pub.sendMessage("get_current_p", default=0)
            c = pub.sendMessage("get_current_c", default=0)
            # Ensure values are not None
            t = t if t is not None else 0
            p = p if p is not None else 0
            c = c if c is not None else 0
            
            print(f"Fetching metrics for T:{t}, P:{p}, C:{c}")
            
            # Request cell metrics from the service for current frame
            polars_df = self.metrics_service.query(time=t, position=p, channel=c)
            
            if polars_df.is_empty():
                print("No metrics found in polars dataframe")
                QMessageBox.warning(self, "No Data", "No metrics available for the current frame.")
                return
            
            print(f"Retrieved polars dataframe with shape: {polars_df.shape}")
            print(f"Columns: {polars_df.columns}")
            
            # Convert polars to pandas
            metrics_df = polars_df.to_pandas()
            print(f"Converted to pandas dataframe with shape: {metrics_df.shape}")
            print(f"Pandas columns: {metrics_df.columns}")
            
            # Convert the metrics dataframe to the cell_mapping format expected by existing code
            self.cell_mapping = {}
            
            for idx, row in metrics_df.iterrows():
                # Use 'cell_id' instead of 'ID'
                if 'cell_id' in row:
                    cell_id = int(row['cell_id'])
                else:
                    print(f"Warning: Row {idx} missing cell_id column")
                    continue
                
                # Extract bounding box - if not available, use defaults
                if all(col in row for col in ['y1', 'x1', 'y2', 'x2']):
                    bbox = (
                        int(row['y1']), 
                        int(row['x1']), 
                        int(row['y2']), 
                        int(row['x2'])
                    )
                else:
                    # Use centroid to create a small bounding box if available
                    if 'centroid_y' in row and 'centroid_x' in row:
                        cy, cx = row['centroid_y'], row['centroid_x']
                        bbox = (int(cy-5), int(cx-5), int(cy+5), int(cx+5))
                        print(f"Created bbox from centroid for cell {cell_id}: {bbox}")
                    else:
                        bbox = (0, 0, 10, 10)  # Default fallback
                        print(f"Using default bbox for cell {cell_id}")
                
                # Extract metrics dictionary - exclude certain columns
                exclude_cols = ['cell_id', 'y1', 'x1', 'y2', 'x2']
                metrics = {col: row[col] for col in row.index if col not in exclude_cols}
                
                # Create cell mapping entry
                self.cell_mapping[cell_id] = {
                    "bbox": bbox,
                    "metrics": metrics
                }
            
            print(f"Created cell_mapping with {len(self.cell_mapping)} entries")
            
            # Update the UI with the fetched metrics
            self.populate_metrics_table()
            self.update_annotation_scatter()
            
            # NEW: Send message to ViewAreaWidget to draw bounding boxes
            pub.sendMessage("draw_cell_bounding_boxes", 
                        time=t, 
                        position=p, 
                        channel=c, 
                        cell_mapping=self.cell_mapping)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error fetching metrics: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to fetch metrics: {str(e)}")
        
    
    def on_table_item_click(self, item):
        """Handle clicks on the metrics table to select and track cells"""
        try:
            row = item.row()
            cell_id = self.metrics_table.item(row, 0).text()
            cell_id = int(cell_id)
            
            # Always highlight the cell in the current frame first
            print(f"Highlighting cell {cell_id} in current frame")
            self.highlight_cell_in_image(cell_id)
            
            # Only attempt tracking as a secondary action
            try:
                print(f"Attempting to track cell {cell_id}")
                self.select_cell_for_tracking(cell_id)
            except Exception as e:
                print(f"Cell tracking failed: {str(e)}")
                # Don't show error message about tracking to the user
                # since highlighting still worked
        except Exception as e:
            print(f"Error in table item click: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to process cell selection: {str(e)}")
        
    def on_pca_point_click(self, event):
        """
        Handle click events on the PCA scatter plot.
        Highlight the corresponding cell in the image.
        """
        if event.artist not in self.figure_annot_scatter.axes[0].collections:
            return

        # Get the index of the clicked point
        ind = event.ind[0]  # event.ind is a list of indices of the picked points

        # Map the index back to the cell_id (similar to list(self.cell_mapping.keys())[ind] in the old code)
        clicked_cell_id = self.pca_data["cell_ids"][ind]
        print(f"Clicked PCA point for cell_id: {clicked_cell_id}")

        # Store the selected cell_id
        self.selected_cell_id = clicked_cell_id

        # Highlight the cell in the image
        self.highlight_cell_in_image(clicked_cell_id)
        
    
    def select_cell_for_tracking(self, cell_id):
        """Select a specific cell to track across frames."""
        print(f"Selected cell {cell_id} for tracking")
        self.selected_cell_id = cell_id
        
        # Check if tracking is available (don't initialize an empty list)
        if not hasattr(self, "lineage_tracks") or not self.lineage_tracks:
            print("No tracking data available")
            reply = QMessageBox.information(
                self, "No Tracking Data",
                "Cell highlighting is working, but tracking data is not available yet.\n\n"
                "Tracking requires running cell tracking first.",
                QMessageBox.Ok | QMessageBox.Help, 
                QMessageBox.Ok
            )
            if reply == QMessageBox.Help:
                QMessageBox.information(
                    self, "How To Track Cells",
                    "To track cells across frames:\n\n"
                    "1. Go to the Morphology / Time tab\n"
                    "2. Click 'Visualize Lineage Tree'\n"
                    "3. Once tracking is complete, return here\n"
                    "4. Select a cell from the table again"
                )
            return
        
        # If we have tracking data, proceed with finding the cell
        self.find_cell_in_tracking_data(cell_id)
     
    def set_tracking_data(self, lineage_tracks):
        """Set tracking data from external source"""
        print(f"Received tracking data with {len(lineage_tracks)} tracks")
        self.lineage_tracks = lineage_tracks
        
        # If we have a previously selected cell, try to track it now
        if hasattr(self, "selected_cell_id") and self.selected_cell_id:
            print(f"Re-attempting to track previously selected cell {self.selected_cell_id}")
            self.find_cell_in_tracking_data(self.selected_cell_id)
    
    def find_cell_in_tracking_data(self, cell_id):
        """Find a cell in the tracking data and prepare tracking information"""
        print(f"Finding cell {cell_id} in tracking data...")
        
        # Clear previous tracking
        self.tracked_cell_lineage = {}
        
        # Find the track for this cell
        selected_track = None
        for track in self.lineage_tracks:
            if track['ID'] == cell_id:
                selected_track = track
                break
                
        if not selected_track:
            print(f"Cell {cell_id} not found in tracking data")
            QMessageBox.warning(
                self, "Cell Not Found",
                f"Cell {cell_id} not found in tracking data."
            )
            return
            
        # Get all frames where this cell appears
        if 't' in selected_track:
            t_values = selected_track['t']
            print(f"Cell {cell_id} appears in frames: {t_values}")
            
            # Map each frame to this cell ID
            for t in t_values:
                if t not in self.tracked_cell_lineage:
                    self.tracked_cell_lineage[t] = []
                self.tracked_cell_lineage[t].append(cell_id)
                
            # Get children cells if any
            if 'children' in selected_track and selected_track['children']:
                print(f"Cell {cell_id} has children: {selected_track['children']}")
                self.add_children_to_tracking(selected_track['children'])
                
        QMessageBox.information(
            self, "Cell Tracking Prepared",
            f"Cell {cell_id} will be tracked across {len(self.tracked_cell_lineage)} frames.\n"
            f"Use the time slider to navigate frames."
        )
        
        # Notify that tracking data is ready
        pub.sendMessage("cell_tracking_ready", cell_id=cell_id, lineage_data=self.tracked_cell_lineage)
    
    def add_children_to_tracking(self, child_ids):
        """Add children cells to tracking data recursively"""
        for child_id in child_ids:
            # Find the child's track
            child_track = None
            for track in self.lineage_tracks:
                if track['ID'] == child_id:
                    child_track = track
                    break
                    
            if child_track and 't' in child_track:
                print(f"Adding child {child_id} to tracking")
                t_values = child_track['t']
                
                # Map each frame to this child ID
                for t in t_values:
                    if t not in self.tracked_cell_lineage:
                        self.tracked_cell_lineage[t] = []
                    self.tracked_cell_lineage[t].append(child_id)
                    
                # Recursively add this child's children
                if 'children' in child_track and child_track['children']:
                    print(f"Child {child_id} has children: {child_track['children']}")
                    self.add_children_to_tracking(child_track['children'])
    
    def save_tracked_cell_as_gif(self):
        """Save the movement of the tracked cell as an animated GIF."""
        if not hasattr(self, "tracked_cell_lineage") or not self.tracked_cell_lineage:
            QMessageBox.warning(self, "Error", "No cell is being tracked.")
            return
            
        # Ask user for save location
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Cell Movement as GIF", "", "GIF Files (*.gif)"
        )
        if not save_path:
            return
            
        # Request GIF creation via pub/sub
        pub.sendMessage("create_tracking_gif_requested", 
                       tracked_data=self.tracked_cell_lineage,
                       save_path=save_path)
    
    
    def highlight_cell_in_image(self, cell_id):
        """
        Request highlighting of a cell in the main image view
        Called when a cell is clicked in the PCA scatter plot
        """
        print(f"MorphologyWidget: Requesting highlight for cell {cell_id}")
        
        # Ensure cell_id is an integer
        cell_id = int(cell_id)
        
        # Send message to request cell highlighting
        pub.sendMessage("highlight_cell_requested", cell_id=cell_id)
    
    def update_annotation_scatter(self):
        """Update the PCA scatter plot with current cell morphology data"""
        try:
            if not hasattr(self, "cell_mapping") or not self.cell_mapping:
                return
                
            # Prepare DataFrame
            metrics_data = [
                {**{"ID": cell_id}, **data["metrics"], **
                    {"Class": data["metrics"]["morphology_class"]}}
                for cell_id, data in self.cell_mapping.items()
            ]
            morphology_df = pd.DataFrame(metrics_data)
            
            # Select numeric features for PCA
            numeric_features = [
                'area',
                'perimeter',
                'equivalent_diameter',
                'orientation',
                'aspect_ratio',
                'circularity',
                'solidity']
            X = morphology_df[numeric_features].values
            
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform PCA
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(X_scaled)
            
            # Store PCA results
            pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
            pca_df['Class'] = morphology_df['Class']
            pca_df['ID'] = morphology_df['ID']
            
            # Plot PCA scatter
            self.figure_annot_scatter.clear()
            ax = self.figure_annot_scatter.add_subplot(111)
            scatter = ax.scatter(
                pca_df['PC1'],
                pca_df['PC2'],
                c=[
                    self.morphology_colors_rgb[class_] for class_ in pca_df['Class']],
                s=50,
                edgecolor='w',
                picker=True)
                
            ax.set_title("PCA Scatter Plot")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            
            # Enable interactive annotations and highlighting
            self.annotate_scatter_points(ax, scatter, pca_df)
            
            self.canvas_annot_scatter.draw()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {str(e)}")
    
    def annotate_scatter_points(self, ax, scatter, pca_df):
        """
        Adds interactive hover annotations and click event to highlight a selected cell.
        """
        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            ha="center",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        annot.set_visible(False)
        
        def update_annot(ind):
            """Update annotation text and position based on hovered point."""
            index = ind["ind"][0]
            pos = scatter.get_offsets()[index]
            annot.xy = pos
            selected_id = int(pca_df.iloc[index]["ID"])
            cell_class = pca_df.iloc[index]["Class"]
            annot.set_text(f"ID: {selected_id}\nClass: {cell_class}")
            annot.get_bbox_patch().set_alpha(0.8)
            
        def on_hover(event):
            """Handle hover events to show annotations."""
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    self.canvas_annot_scatter.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        self.canvas_annot_scatter.draw_idle()
                        
        def on_click(event):
            """Handle click events to highlight the selected cell in the segmented image."""
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    index = ind["ind"][0]
                    selected_id = int(pca_df.iloc[index]["ID"])
                    cell_class = pca_df.iloc[index]["Class"]
                    self.highlight_cell_in_image(selected_id)
                    
        self.canvas_annot_scatter.mpl_connect("motion_notify_event", on_hover)
        self.canvas_annot_scatter.mpl_connect("button_press_event", on_click)
        
        # Add legend using self.morphology_colors_rgb
        handles = [
            plt.Line2D(
                [0], [0],
                marker='o',
                color=color,
                label=key,
                markersize=8,
                linestyle='',
            )
            for key, color in self.morphology_colors_rgb.items()
        ]
        ax.legend(
            handles=handles,
            title="Morphology Class",
            loc='best',
        )
    
    
    def populate_metrics_table(self):
        """Populate the metrics table with cell data, showing only classification-relevant columns"""
        if not self.cell_mapping:
            return
            
        # Convert cell mapping to a DataFrame
        metrics_data = [
            {**{"ID": cell_id}, **data["metrics"]}
            for cell_id, data in self.cell_mapping.items()
        ]
        metrics_df = pd.DataFrame(metrics_data)
        
        print(f"Created metrics dataframe with columns: {metrics_df.columns}")
        
        # Select only classification-relevant columns
        # Include "ID" first, then morphology class, then relevant metrics
        columns_to_show = ["ID", "morphology_class", "area", "perimeter", 
                        "aspect_ratio", "circularity", "solidity", 
                        "equivalent_diameter"]
        
        # Only keep columns that exist in the dataframe
        available_columns = [col for col in columns_to_show if col in metrics_df.columns]
        
        # Filter dataframe to show only selected columns
        metrics_df = metrics_df[available_columns]
        
        # Round numeric columns to 2 decimal places (excluding ID and morphology_class)
        numeric_columns = [col for col in available_columns 
                        if col not in ["ID", "morphology_class"]]
        
        if numeric_columns:
            metrics_df[numeric_columns] = metrics_df[numeric_columns].round(2)
        
        # Update QTableWidget
        self.metrics_table.setRowCount(metrics_df.shape[0])
        self.metrics_table.setColumnCount(metrics_df.shape[1])
        self.metrics_table.setHorizontalHeaderLabels(metrics_df.columns)
        
        for row in range(metrics_df.shape[0]):
            for col, value in enumerate(metrics_df.iloc[row]):
                self.metrics_table.setItem(
                    row, col, QTableWidgetItem(str(value)))
                
                # Highlight the morphology class column with background color
                if metrics_df.columns[col] == "morphology_class":
                    morph_class = str(value)
                    if morph_class in self.morphology_colors:
                        item = self.metrics_table.item(row, col)
                        bgr_color = self.morphology_colors[morph_class]
                        # Convert BGR to RGB for Qt
                        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
                        item.setBackground(QColor(*rgb_color))
                        # Use contrasting text color
                        item.setForeground(QColor(255, 255, 255) if sum(rgb_color) < 384 else QColor(0, 0, 0))
                        
    
    def export_metrics_to_csv(self):
        """Exports the metrics table data to a CSV file."""
        try:
            if not self.cell_mapping:
                QMessageBox.warning(self, "Error", "No cell data available.")
                return
                
            metrics_data = [
                {**{"ID": cell_id}, **data["metrics"]}
                for cell_id, data in self.cell_mapping.items()
            ]
            metrics_df = pd.DataFrame(metrics_data)
            
            if metrics_df.empty:
                QMessageBox.warning(
                    self, "Error", "No data available to export.")
                return
                
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Metrics Data", "", "CSV Files (*.csv);;All Files (*)")
            if save_path:
                metrics_df.to_csv(save_path, index=False)
                QMessageBox.information(
                    self, "Success", f"Metrics data exported to {save_path}"
                )
            else:
                QMessageBox.warning(self, "Cancelled", "Export cancelled.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
    
    # Handlers for pub/sub messages
    def on_cell_mapping_updated(self, cell_mapping):
        """Handler for updated cell mapping data"""
        self.cell_mapping = cell_mapping
        self.populate_metrics_table()
        self.update_annotation_scatter()
    
    def on_segmentation_ready(self, segmentation_data):
        """Handler for when segmentation is ready"""
        # Process segmentation results if needed
        pass
    
    def on_image_data_loaded(self, image_data):
        """Handler for when new image data is loaded"""
        # Reset state for new image data
        self.selected_cell_id = None
        self.tracked_cell_lineage = {}
        if self.save_gif_button:
            self.save_gif_button.setVisible(False)
    
    def on_current_view_changed(self, t, p, c):
        """Handler for when the current view changes (time/position/channel)"""
        # Update display if needed for new t/p/c values
        pass
    
    
    def process_morphology_time_series(self):
        """Process morphology across time points using metrics service"""
        if not self.metrics_service:
            QMessageBox.warning(self, "Error", "Metrics service not available.")
            return
        
        # Get current position and channel
        p = pub.sendMessage("get_current_p", default=0)
        c = pub.sendMessage("get_current_c", default=0)
        
        # Query metrics for all timepoints for this position and channel
        df = self.metrics_service.query(position=p, channel=c)
        
        if df.is_empty():
            QMessageBox.warning(self, "No Data", "No metrics available.")
            return
        
        # Convert to pandas
        metrics_df = df.to_pandas()
        
        # Get all unique morphology classes
        all_morphologies = metrics_df['morphology_class'].unique()
        
        # Group by time
        by_time = metrics_df.groupby('time')
        
        # Create data structures for plotting
        times = sorted(metrics_df['time'].unique())
        morphology_counts = {morphology: [] for morphology in all_morphologies}
        morphology_fractions = {morphology: [] for morphology in all_morphologies}
        total_cells = []
        
        # Calculate counts and fractions for each timepoint
        for t in times:
            t_data = metrics_df[metrics_df['time'] == t]
            t_counts = t_data['morphology_class'].value_counts()
            t_total = len(t_data)
            total_cells.append(t_total)
            
            for morph in all_morphologies:
                count = t_counts.get(morph, 0)
                morphology_counts[morph].append(count)
                morphology_fractions[morph].append(count/t_total if t_total > 0 else 0)
        
        # Clear the figure and create two subplots
        self.figure_annot_scatter.clear()
        ax1 = self.figure_annot_scatter.add_subplot(2, 1, 1)
        ax2 = self.figure_annot_scatter.add_subplot(2, 1, 2)
        
        # Plot fractions
        for morph, fractions in morphology_fractions.items():
            color = self.morphology_colors_rgb.get(morph, (0.5, 0.5, 0.5))
            ax1.plot(times, fractions, 'o-', label=morph, color=color)
        
        ax1.set_title("Morphology Fractions Over Time")
        ax1.set_ylabel("Fraction (%)")
        ax1.legend()
        
        # Plot counts
        for morph, counts in morphology_counts.items():
            color = self.morphology_colors_rgb.get(morph, (0.5, 0.5, 0.5))
            ax2.plot(times, counts, 'o-', label=morph, color=color)
        
        # Add total cells line on secondary y-axis
        ax3 = ax2.twinx()
        ax3.plot(times, total_cells, 'k--', label='Total Cells')
        
        ax2.set_title("Cell Counts By Morphology Over Time")
        ax2.set_xlabel("Time (mins)")
        ax2.set_ylabel("Count by Class")
        ax3.set_ylabel("Total Cell Count")
        
        # Legend for second plot
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Adjust layout and draw
        self.figure_annot_scatter.tight_layout()
        self.canvas_annot_scatter.draw()