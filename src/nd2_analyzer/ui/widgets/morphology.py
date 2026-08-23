import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QFileDialog,
    QTabWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pubsub import pub
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from nd2_analyzer.analysis.metrics_service import MetricsService

class MorphologyWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cell_mapping = {}
        self.displayed_metrics_df = pd.DataFrame()
        self.selected_cell_id = None
        self.tracked_cell_lineage = {}

        # Store reference to metrics service
        self.metrics_service = MetricsService()

        # Set up morphology colors (same as original)
        self.morphology_colors = {
            "Artifact": (128, 128, 128),  # Gray
            "Divided": (255, 0, 0),  # Blue
            "Healthy": (0, 255, 0),  # Green
            "Elongated": (0, 255, 255),  # Yellow
            "Deformed": (255, 0, 255),  # Magenta
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
        pub.subscribe(self.on_view_index_changed, "view_index_changed")
        pub.subscribe(self.set_tracking_data, "tracking_data_available")
        pub.subscribe(
            self.process_morphology_time_series,
            "process_morphology_time_series_requested",
        )
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
        print(
            f"MorphologyWidget: provide_cell_mapping called for T={time}, P={position}, C={channel}, cell_id={cell_id}"
        )

        # Query metrics service for the requested frame on-demand
        try:
            polars_df = self.metrics_service.query_optimized(time=time, position=position)

            if polars_df.is_empty():
                print(f"   No metrics found for T={time}, P={position}")
                callback(None)
                return

            # Convert to pandas and build cell_mapping for this frame
            metrics_df = polars_df.to_pandas()
            frame_cell_mapping = {}

            for idx, row in metrics_df.iterrows():
                if "cell_id" not in row:
                    continue

                row_cell_id = int(row["cell_id"])

                # If specific cell requested, only process that one
                if cell_id is not None and row_cell_id != cell_id:
                    continue

                # Extract bounding box
                if all(col in row for col in ["y1", "x1", "y2", "x2"]):
                    bbox = (int(row["y1"]), int(row["x1"]), int(row["y2"]), int(row["x2"]))
                else:
                    if "centroid_y" in row and "centroid_x" in row:
                        cy, cx = row["centroid_y"], row["centroid_x"]
                        bbox = (int(cy - 5), int(cx - 5), int(cy + 5), int(cx + 5))
                    else:
                        bbox = (0, 0, 10, 10)

                # Extract metrics
                exclude_cols = ["cell_id", "y1", "x1", "y2", "x2"]
                metrics = {col: row[col] for col in row.index if col not in exclude_cols}

                frame_cell_mapping[row_cell_id] = {"bbox": bbox, "metrics": metrics}

            if cell_id is not None:
                if cell_id in frame_cell_mapping:
                    print(f"   ✅ Found cell {cell_id} in frame {time}")
                    callback({cell_id: frame_cell_mapping[cell_id]})
                else:
                    print(f"   ❌ Cell {cell_id} not found in frame {time}")
                    callback(None)
            else:
                print(f"   Returning {len(frame_cell_mapping)} cells for frame {time}")
                callback(frame_cell_mapping)

        except Exception as e:
            print(f"   Error fetching cell mapping: {e}")
            import traceback
            traceback.print_exc()
            callback(None)

    def initUI(self):
        layout = QVBoxLayout(self)

        buttons_layout = QHBoxLayout()

        buttons_layout.addSpacing(20)

        # Add the classify cells button
        self.fetch_metrics_button = QPushButton("Classify Cells")
        self.fetch_metrics_button.clicked.connect(self.fetch_metrics_from_service)
        self.fetch_metrics_button.setStyleSheet(
            "background-color: black; color: white; font-weight: bold;"
        )
        self.fetch_metrics_button.setFixedHeight(40)
        buttons_layout.addWidget(self.fetch_metrics_button)

        buttons_layout.addSpacing(20)

        # Add Process Morphology Over Time button
        self.process_morphology_button = QPushButton("Process Morphology Over Time")
        self.process_morphology_button.clicked.connect(
            self.process_morphology_time_series
        )
        self.process_morphology_button.setStyleSheet(
            "background-color: black; color: white; font-weight: bold;"
        )
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
        self.save_plot_button.setStyleSheet(
            "background-color: white; color: black; font-size: 12px;"
        )
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
        self.export_button.setStyleSheet(
            "background-color: white; color: black; font-size: 14px;"
        )
        self.export_button.clicked.connect(self.export_metrics_to_csv)
        self.button_layout.addWidget(self.export_button)

        # Export to GIF button
        self.export_gif_button = QPushButton("Export to GIF")
        self.export_gif_button.setStyleSheet(
            "background-color: white; color: black; font-size: 14px;"
        )
        self.export_gif_button.clicked.connect(self.create_gif_animations)
        self.button_layout.addWidget(self.export_gif_button)

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
            self,
            "Save PCA Plot",
            "",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)",
        )

        if file_path:
            try:
                self.figure_annot_scatter.savefig(
                    file_path, dpi=300, bbox_inches="tight"
                )
                QMessageBox.information(
                    self, "Success", f"PCA plot saved to {file_path}"
                )
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save plot: {str(e)}")

    def fetch_metrics_from_service(self):
        """Fetch cell metrics and classification from the metrics service dataframe and annotate image"""
        print("🔵 fetch_metrics_from_service CALLED")

        if not self.metrics_service:
            print("❌ Metrics service not available")
            QMessageBox.warning(self, "Error", "Metrics service not available.")
            return

        try:
            # Get current frame information from stored view index
            t = getattr(self, "_current_t", 0)
            p = getattr(self, "_current_p", 0)
            c = getattr(self, "_current_c", 0)

            print(f"   Current view: t={t}, p={p}, c={c}")

            print(f"🔵 Fetching metrics for T:{t}, P:{p}, C:{c}")

            # Request cell metrics from the service for current frame
            print(f"   Querying metrics_service.query_optimized(time={t}, position={p})")
            polars_df = self.metrics_service.query_optimized(time=t, position=p)

            if polars_df.is_empty():
                print("❌ No metrics found in polars dataframe - EMPTY RESULT")
                QMessageBox.warning(
                    self, "No Data", "No metrics available for the current frame."
                )
                return

            print(f"✅ Retrieved polars dataframe with shape: {polars_df.shape}")
            print(f"   Columns: {polars_df.columns}")

            # Convert polars to pandas
            metrics_df = polars_df.to_pandas()
            print(f"Converted to pandas dataframe with shape: {metrics_df.shape}")
            print(f"Pandas columns: {metrics_df.columns}")

            # Convert the metrics dataframe to the cell_mapping format expected by existing code
            self.cell_mapping = {}

            for idx, row in metrics_df.iterrows():
                # Use 'cell_id' instead of 'ID'
                if "cell_id" in row:
                    cell_id = int(row["cell_id"])
                else:
                    print(f"Warning: Row {idx} missing cell_id column")
                    continue

                # Extract bounding box - if not available, use defaults
                if all(col in row for col in ["y1", "x1", "y2", "x2"]):
                    bbox = (
                        int(row["y1"]),
                        int(row["x1"]),
                        int(row["y2"]),
                        int(row["x2"]),
                    )
                else:
                    # Use centroid to create a small bounding box if available
                    if "centroid_y" in row and "centroid_x" in row:
                        cy, cx = row["centroid_y"], row["centroid_x"]
                        bbox = (int(cy - 5), int(cx - 5), int(cy + 5), int(cx + 5))
                        print(f"Created bbox from centroid for cell {cell_id}: {bbox}")
                    else:
                        bbox = (0, 0, 10, 10)  # Default fallback
                        print(f"Using default bbox for cell {cell_id}")

                # Extract metrics dictionary - exclude certain columns
                exclude_cols = ["cell_id", "y1", "x1", "y2", "x2"]
                metrics = {
                    col: row[col] for col in row.index if col not in exclude_cols
                }

                # Create cell mapping entry
                self.cell_mapping[cell_id] = {"bbox": bbox, "metrics": metrics}

            print(f"Created cell_mapping with {len(self.cell_mapping)} entries")

            # Update the UI with the fetched metrics
            self.populate_metrics_table()
            self.update_annotation_scatter()

            # Send message to App to draw ALL cells with morphology colors
            pub.sendMessage(
                "draw_cell_bounding_boxes",
                time=t,
                position=p,
                channel=c,
                cell_mapping=self.cell_mapping,
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error fetching metrics: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to fetch metrics: {str(e)}")

    def on_table_item_click(self, item):
        """Handle clicks on the metrics table to select and track cells"""
        try:
            if self.metrics_table.horizontalHeaderItem(0).text() != "ID":
                return
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
            QMessageBox.warning(
                self, "Error", f"Failed to process cell selection: {str(e)}"
            )

    def select_cell_for_tracking(self, cell_id):
        """Select a specific cell to track across frames."""
        print(f"Selected cell {cell_id} for tracking")
        self.selected_cell_id = cell_id

        # Check if tracking is available (don't initialize an empty list)
        if not hasattr(self, "lineage_tracks") or not self.lineage_tracks:
            print("No tracking data available")
            reply = QMessageBox.information(
                self,
                "No Tracking Data",
                "Cell highlighting is working, but tracking data is not available yet.\n\n"
                "Tracking requires running cell tracking first.",
                QMessageBox.Ok | QMessageBox.Help,
                QMessageBox.Ok,
            )
            if reply == QMessageBox.Help:
                QMessageBox.information(
                    self,
                    "How To Track Cells",
                    "To track cells across frames:\n\n"
                    "1. Go to the Morphology / Time tab\n"
                    "2. Click 'Visualize Lineage Tree'\n"
                    "3. Once tracking is complete, return here\n"
                    "4. Select a cell from the table again",
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
            print(
                f"Re-attempting to track previously selected cell {self.selected_cell_id}"
            )
            self.find_cell_in_tracking_data(self.selected_cell_id)

    def find_cell_in_tracking_data(self, cell_id):
        """Find a cell in the tracking data using spatial seg-label lookup."""
        print(f"Finding cell {cell_id} in tracking data...")

        self.tracked_cell_lineage = {}

        from nd2_analyzer.data.appstate import ApplicationState
        from nd2_analyzer.data.image_data import ImageData
        from skimage.measure import label as sk_label

        appstate = ApplicationState.get_instance()
        if appstate and appstate.view_index:
            t, p, c = appstate.view_index
        else:
            t, p, c = 0, 0, 0

        print(f"Current frame: T={t}, P={p}")

        # Get the labeled segmentation mask for this frame
        image_data = ImageData.get_instance()
        labeled_mask = None
        if image_data and hasattr(image_data, "segmentation_cache"):
            try:
                model = image_data.segmentation_service.models.available_models[0]
                seg = image_data.segmentation_cache.with_model(model)[t, p, c]
                if seg is not None:
                    seg = np.asarray(seg)
                    if seg.max() <= 255 and len(np.unique(seg)) <= 100:
                        labeled_mask = sk_label(seg > 0)
                    else:
                        labeled_mask = seg
            except Exception as e:
                print(f"Could not get segmentation mask: {e}")

        if labeled_mask is None:
            QMessageBox.warning(
                self, "Error",
                "Could not get segmentation mask for this frame."
            )
            return

        # Spatial lookup: find which track's (x,y) sits on seg label == cell_id
        selected_track = None
        for track in self.lineage_tracks:
            if "t" not in track or "x" not in track or "y" not in track:
                continue
            for i, track_t in enumerate(track["t"]):
                if track_t == t:
                    tx = int(round(track["x"][i]))
                    ty = int(round(track["y"][i]))
                    if 0 <= ty < labeled_mask.shape[0] and 0 <= tx < labeled_mask.shape[1]:
                        if int(labeled_mask[ty, tx]) == cell_id:
                            selected_track = track
                            print(f"Track {track['ID']} at ({tx},{ty}) lands on seg label {cell_id}")
                    break
            if selected_track:
                break

        if not selected_track:
            print(f"No track found on seg label {cell_id}, falling back to nearest distance")
            # Fallback: use centroid from metrics if available
            cell_cx, cell_cy = None, None
            if cell_id in self.cell_mapping:
                info = self.cell_mapping[cell_id]
                if "metrics" in info:
                    m = info["metrics"]
                    if "centroid_x" in m and "centroid_y" in m:
                        cell_cx = float(m["centroid_x"])
                        cell_cy = float(m["centroid_y"])
                if cell_cx is None:
                    y1, x1, y2, x2 = info["bbox"]
                    cell_cx, cell_cy = (x1 + x2) / 2, (y1 + y2) / 2

            if cell_cx is not None:
                min_dist = float("inf")
                for track in self.lineage_tracks:
                    if "t" not in track or "x" not in track or "y" not in track:
                        continue
                    for i, track_t in enumerate(track["t"]):
                        if track_t == t:
                            d = np.sqrt((track["x"][i] - cell_cx)**2 + (track["y"][i] - cell_cy)**2)
                            if d < min_dist:
                                min_dist = d
                                selected_track = track
                            break

            if not selected_track or min_dist > 30:
                QMessageBox.warning(
                    self, "Cell Not Tracked",
                    f"Cell {cell_id} could not be matched to any track."
                )
                return
            print(f"Fallback matched cell {cell_id} to Track {selected_track['ID']} (dist={min_dist:.1f}px)")

        print(f"Matched seg cell {cell_id} -> Track ID {selected_track['ID']}")

        # Get all frames where this track appears
        if "t" in selected_track:
            t_values = selected_track["t"]
            print(f"Track {selected_track['ID']} appears in frames: {t_values}")

            # Map each frame to this TRACK ID (not the original cell_id)
            # The track ID is what we need for highlighting across frames
            track_id = selected_track['ID']
            for t in t_values:
                if t not in self.tracked_cell_lineage:
                    self.tracked_cell_lineage[t] = []
                self.tracked_cell_lineage[t].append(track_id)

            # Get children cells if any
            if "children" in selected_track and selected_track["children"]:
                print(f"Cell {cell_id} has children: {selected_track['children']}")
                self.add_children_to_tracking(selected_track["children"])

        QMessageBox.information(
            self,
            "Cell Tracking Prepared",
            f"Cell {cell_id} (Track ID {selected_track['ID']}) will be tracked across {len(self.tracked_cell_lineage)} frames.\n"
            f"Use the time slider to navigate frames and see the cell highlighted.",
        )

        # Notify that tracking data is ready
        # Note: We're now passing track_id, not the original segmentation cell_id
        pub.sendMessage(
            "cell_tracking_ready",
            cell_id=selected_track['ID'],
            lineage_data=self.tracked_cell_lineage,
        )

    def add_children_to_tracking(self, child_ids):
        """Add children cells to tracking data recursively"""
        for child_id in child_ids:
            # Find the child's track
            child_track = None
            for track in self.lineage_tracks:
                if track["ID"] == child_id:
                    child_track = track
                    break

            if child_track and "t" in child_track:
                print(f"Adding child {child_id} to tracking")
                t_values = child_track["t"]

                # Map each frame to this child ID
                for t in t_values:
                    if t not in self.tracked_cell_lineage:
                        self.tracked_cell_lineage[t] = []
                    self.tracked_cell_lineage[t].append(child_id)

                # Recursively add this child's children
                if "children" in child_track and child_track["children"]:
                    print(f"Child {child_id} has children: {child_track['children']}")
                    self.add_children_to_tracking(child_track["children"])

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
                {
                    **{"ID": cell_id},
                    **data["metrics"],
                    **{"Class": data["metrics"]["morphology_class"]},
                }
                for cell_id, data in self.cell_mapping.items()
            ]
            morphology_df = pd.DataFrame(metrics_data)

            # Select numeric features for PCA
            numeric_features = [
                "area",
                "perimeter",
                "equivalent_diameter",
                "orientation",
                "aspect_ratio",
                "circularity",
                "solidity",
            ]
            X = morphology_df[numeric_features].values

            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Perform PCA
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(X_scaled)

            # Store PCA results
            pca_df = pd.DataFrame(principal_components, columns=["PC1", "PC2"])
            pca_df["Class"] = morphology_df["Class"]
            pca_df["ID"] = morphology_df["ID"]

            # Plot PCA scatter
            self.figure_annot_scatter.clear()
            ax = self.figure_annot_scatter.add_subplot(111)
            scatter = ax.scatter(
                pca_df["PC1"],
                pca_df["PC2"],
                c=[self.morphology_colors_rgb[class_] for class_ in pca_df["Class"]],
                s=50,
                edgecolor="w",
                picker=True,
            )

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
                [0],
                [0],
                marker="o",
                color=color,
                label=key,
                markersize=8,
                linestyle="",
            )
            for key, color in self.morphology_colors_rgb.items()
        ]
        ax.legend(
            handles=handles,
            title="Morphology Class",
            loc="best",
        )

    def populate_metrics_table(self, metrics_df=None):
        """Populate the metrics table with cell data, showing only classification-relevant columns"""
        if metrics_df is None:
            if not self.cell_mapping:
                return

            # Convert cell mapping to a DataFrame
            metrics_data = [
                {**{"ID": cell_id}, **data["metrics"]}
                for cell_id, data in self.cell_mapping.items()
            ]
            metrics_df = pd.DataFrame(metrics_data)
        else:
            metrics_df = metrics_df.copy()
            if "cell_id" in metrics_df.columns and "ID" not in metrics_df.columns:
                metrics_df = metrics_df.rename(columns={"cell_id": "ID"})

        print(f"Created metrics dataframe with columns: {metrics_df.columns}")

        # Select only classification-relevant columns
        # Include "ID" first, then morphology class, then relevant metrics
        columns_to_show = [
            "ID",
            "time",
            "morphology_class",
            "area",
            "perimeter",
            "aspect_ratio",
            "circularity",
            "solidity",
            "equivalent_diameter",
        ]

        # Only keep columns that exist in the dataframe
        available_columns = [
            col for col in columns_to_show if col in metrics_df.columns
        ]

        # Filter dataframe to show only selected columns
        metrics_df = metrics_df[available_columns].copy()

        # Round numeric columns to 2 decimal places (excluding ID and morphology_class)
        numeric_columns = [
            col for col in available_columns if col not in ["ID", "morphology_class"]
        ]

        if numeric_columns:
            metrics_df[numeric_columns] = metrics_df[numeric_columns].round(2)

        # Keep the displayed data so CSV export matches the table exactly.
        self.displayed_metrics_df = metrics_df

        # Update QTableWidget
        self.metrics_table.setRowCount(metrics_df.shape[0])
        self.metrics_table.setColumnCount(metrics_df.shape[1])
        self.metrics_table.setHorizontalHeaderLabels(metrics_df.columns)

        for row in range(metrics_df.shape[0]):
            for col, value in enumerate(metrics_df.iloc[row]):
                self.metrics_table.setItem(row, col, QTableWidgetItem(str(value)))

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
                        item.setForeground(
                            QColor(255, 255, 255)
                            if sum(rgb_color) < 384
                            else QColor(0, 0, 0)
                        )

    def populate_time_series_table(self, metrics_df):
        """Populate the table with the values plotted over time."""
        self.displayed_metrics_df = metrics_df
        self.metrics_table.setRowCount(metrics_df.shape[0])
        self.metrics_table.setColumnCount(metrics_df.shape[1])
        self.metrics_table.setHorizontalHeaderLabels(metrics_df.columns)

        for row in range(metrics_df.shape[0]):
            for col, value in enumerate(metrics_df.iloc[row]):
                self.metrics_table.setItem(row, col, QTableWidgetItem(str(value)))

    def export_metrics_to_csv(self):
        """Exports the metrics table data to a CSV file."""
        try:
            metrics_df = self.displayed_metrics_df

            if metrics_df.empty:
                QMessageBox.warning(self, "Error", "No data available to export.")
                return

            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Metrics Data", "", "CSV Files (*.csv);;All Files (*)"
            )
            if save_path:
                metrics_df.to_csv(save_path, index=False)
                QMessageBox.information(
                    self, "Success", f"Metrics data exported to {save_path}"
                )
            else:
                QMessageBox.warning(self, "Cancelled", "Export cancelled.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def create_gif_animations(self):
        """Create GIF animations for original and segmented images across all time points."""
        try:
            from nd2_analyzer.data.appstate import ApplicationState

            # Get application state
            appstate = ApplicationState.get_instance()
            if not appstate or not appstate.image_data:
                QMessageBox.warning(self, "Error", "No image data available.")
                return

            # Get current position and channel
            p, c = 0, 0
            if appstate.view_index:
                _, p, c = appstate.view_index

            image_data = appstate.image_data

            # Get total number of time points
            total_time_points = image_data.data.shape[0]

            # Check which frames have been segmented
            segmented_frames_set = set()
            if hasattr(image_data.segmentation_cache, 'mmap_arrays_idx') and image_data.segmentation_cache.mmap_arrays_idx:
                model_name = list(image_data.segmentation_cache.mmap_arrays_idx.keys())[0]
                _, segmented_indices = image_data.segmentation_cache.mmap_arrays_idx[model_name]
                # Filter for current position and channel
                segmented_frames_set = {idx[0] for idx in segmented_indices if len(idx) >= 3 and idx[1] == p and idx[2] == c}

            if not segmented_frames_set:
                reply = QMessageBox.question(
                    self,
                    "No Segmentation Found",
                    "No segmented frames found for this position/channel.\n\n"
                    "Do you want to export only the original images?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return

            # Ask user where to save the GIFs
            save_dir = QFileDialog.getExistingDirectory(
                self, "Select Directory to Save GIFs"
            )
            if not save_dir:
                QMessageBox.warning(self, "Cancelled", "Export cancelled.")
                return

            # Create lists to store frames
            original_frames = []
            segmented_frames = []

            # Show progress
            from PySide6.QtWidgets import QProgressDialog
            progress = QProgressDialog(
                "Creating GIF animations...", "Cancel", 0, total_time_points, self
            )
            progress.setWindowTitle("Exporting GIFs")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()

            # Loop through all time points
            for t in range(total_time_points):
                if progress.wasCanceled():
                    QMessageBox.information(self, "Cancelled", "GIF export cancelled.")
                    return

                progress.setValue(t)
                progress.setLabelText(f"Processing frame {t + 1}/{total_time_points}...")

                # Get original image
                original_img = image_data.get(t, p, c)

                # Normalize to 8-bit for GIF
                original_img_normalized = ((original_img - original_img.min()) /
                                          (original_img.max() - original_img.min()) * 255).astype(np.uint8)
                original_frames.append(Image.fromarray(original_img_normalized))

                # Get segmented image (labeled) - only if already segmented
                if t in segmented_frames_set:
                    try:
                        # Get the segmentation model name - use the first available model
                        if hasattr(image_data.segmentation_cache, 'mmap_arrays_idx') and image_data.segmentation_cache.mmap_arrays_idx:
                            model_name = list(image_data.segmentation_cache.mmap_arrays_idx.keys())[0]
                            # Directly access the cached array without triggering processing
                            mmap_array, _ = image_data.segmentation_cache.mmap_arrays_idx[model_name]
                            segmented_img = mmap_array[t, p, c]

                            # Create a colored version of the labeled image
                            # Each label gets a unique color
                            from skimage import color
                            from skimage.measure import label

                            # If it's binary, convert to labels
                            if segmented_img.max() <= 1:
                                segmented_img = label(segmented_img)

                            # Create colored label image
                            colored_labels = color.label2rgb(segmented_img, bg_label=0)
                            colored_labels_uint8 = (colored_labels * 255).astype(np.uint8)
                            segmented_frames.append(Image.fromarray(colored_labels_uint8))
                        else:
                            # No segmentation available, create blank image
                            blank = np.zeros_like(original_img_normalized)
                            segmented_frames.append(Image.fromarray(blank))
                    except Exception as e:
                        print(f"Error getting segmentation for frame {t}: {e}")
                        # Create blank image if segmentation fails
                        blank = np.zeros_like(original_img_normalized)
                        segmented_frames.append(Image.fromarray(blank))
                else:
                    # Frame not segmented, create blank image
                    blank = np.zeros_like(original_img_normalized)
                    segmented_frames.append(Image.fromarray(blank))

            progress.setValue(total_time_points)

            # Save GIFs
            original_gif_path = f"{save_dir}/original_p{p}_c{c}.gif"
            segmented_gif_path = f"{save_dir}/segmented_p{p}_c{c}.gif"

            # Save original GIF
            original_frames[0].save(
                original_gif_path,
                save_all=True,
                append_images=original_frames[1:],
                duration=200,  # 200ms per frame
                loop=0
            )

            # Save segmented GIF
            segmented_frames[0].save(
                segmented_gif_path,
                save_all=True,
                append_images=segmented_frames[1:],
                duration=200,  # 200ms per frame
                loop=0
            )

            progress.close()

            segmented_count = len(segmented_frames_set)
            QMessageBox.information(
                self,
                "Success",
                f"GIFs created successfully!\n\n"
                f"Original: {original_gif_path}\n"
                f"Segmented: {segmented_gif_path}\n\n"
                f"Total frames: {total_time_points}\n"
                f"Segmented frames: {segmented_count}/{total_time_points}"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to create GIFs: {str(e)}")

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

    def on_view_index_changed(self, index):
        """Handler for when the current view changes (time/position/channel)"""
        self._current_t, self._current_p, self._current_c = index

    def process_morphology_time_series(self):
        """Process morphology across time points using metrics service."""
        if not self.metrics_service:
            QMessageBox.warning(self, "Error", "Metrics service not available.")
            return

        from nd2_analyzer.data.appstate import ApplicationState
        appstate = ApplicationState.get_instance()

        # Determine the current position (p)
        p = None
        if appstate and appstate.view_index:
            _, p, _ = appstate.view_index

        # Fetch all time‑points for the current position
        df = self.metrics_service.query_optimized(position=p)
        if df.is_empty():
            QMessageBox.warning(self, "No Data", "No metrics available.")
            return

        metrics_df = df.to_pandas()
        all_morphologies = metrics_df["morphology_class"].unique()
        times = sorted(metrics_df["time"].unique())
        morphology_counts = {m: [] for m in all_morphologies}
        morphology_fractions = {m: [] for m in all_morphologies}
        total_cells = []

        for t in times:
            t_data = metrics_df[metrics_df["time"] == t]
            t_counts = t_data["morphology_class"].value_counts()
            t_total = len(t_data)
            total_cells.append(t_total)
            for m in all_morphologies:
                count = t_counts.get(m, 0)
                morphology_counts[m].append(count)
                morphology_fractions[m].append(count / t_total if t_total > 0 else 0)

        # Convert frame numbers to hours if phc_interval is available and valid
        import matplotlib
        matplotlib.rcParams['font.family'] = 'Arial'
        interval_hours = None
        if appstate and appstate.experiment and hasattr(appstate.experiment, 'phc_interval'):
            try:
                val = appstate.experiment.phc_interval / 3600.0
                if val > 0:
                    interval_hours = val
            except Exception:
                interval_hours = None
        times_plot = [t * interval_hours for t in times] if interval_hours else times

        # Show exactly the statistics plotted below: one row per time point.
        time_column = "Time (h)" if interval_hours else "Time (frame)"
        time_series_data = {time_column: times_plot}
        for morph in all_morphologies:
            time_series_data[f"{morph} Fraction"] = morphology_fractions[morph]
        for morph in all_morphologies:
            time_series_data[f"{morph} Count"] = morphology_counts[morph]
        time_series_data["Total Cells"] = total_cells
        self.populate_time_series_table(pd.DataFrame(time_series_data))

        # Clear and set up the large figure (16×10 inches) with two subplots
        self.figure_annot_scatter.clf()
        self.figure_annot_scatter.set_size_inches(16, 10)
        ax1 = self.figure_annot_scatter.add_subplot(2, 1, 1)
        ax2 = self.figure_annot_scatter.add_subplot(2, 1, 2, sharex=ax1)
        ax1.set_facecolor('white')
        ax2.set_facecolor('white')

        # Plot fractions
        line_handles, line_labels = [], []
        for morph, fractions_list in morphology_fractions.items():
            color = self.morphology_colors_rgb.get(morph, (0.5, 0.5, 0.5))
            line, = ax1.plot(times_plot, fractions_list, '-', color=color, linewidth=1.5)
            ax1.scatter(times_plot, fractions_list, color=color, s=25,
                        edgecolor='black', linewidth=0.3)
            line_handles.append(line)
            line_labels.append(morph)

        ax1.set_ylabel("Fraction", fontsize=20, labelpad=5)
        ax1.grid(False)
        ax1.tick_params(labelsize=16)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('gray')
        ax1.spines['bottom'].set_color('gray')

        # Plot counts
        for morph, counts_list in morphology_counts.items():
            color = self.morphology_colors_rgb.get(morph, (0.5, 0.5, 0.5))
            ax2.plot(times_plot, counts_list, '-', color=color, linewidth=1.5)
            ax2.scatter(times_plot, counts_list, color=color, s=25,
                        edgecolor='black', linewidth=0.3)

        # Secondary axis for total cell count
        ax3 = ax2.twinx()
        total_line, = ax3.plot(times_plot, total_cells, '--',
                            color='black', linewidth=1.5)
        ax3.set_ylabel("Total Cell Count", fontsize=20, labelpad=5)

        # Axis labels for counts plot
        ax2.set_ylabel("Count", fontsize=20, labelpad=5)
        x_label = "Time (h)" if interval_hours else "Time (frame)"
        ax2.set_xlabel(x_label, fontsize=20, labelpad=5)
        ax2.grid(False)
        ax2.tick_params(labelsize=16)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('gray')
        ax2.spines['bottom'].set_color('gray')
        ax3.tick_params(labelsize=16)
        ax3.spines['top'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.spines['right'].set_color('gray')
        ax3.spines['bottom'].set_visible(False)

        # Compose and place the legend at the very top
        legend_handles = line_handles + [total_line]
        legend_labels = line_labels + ["Total Cells"]
        legend = self.figure_annot_scatter.legend(
            legend_handles,
            legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(legend_labels),
            frameon=False,
            fontsize=18,
            handlelength=2.0,
            handletextpad=0.5,
            columnspacing=1.0
        )

        # Make legend lines thicker
        for line in legend.get_lines():
            line.set_linewidth(4.0)

        # Provide extra breathing room around panels
        self.figure_annot_scatter.subplots_adjust(hspace=0.5, top=0.86, bottom=0.1)

        # Draw the figure on the canvas
        self.canvas_annot_scatter.draw()
