"""
Cell View Dialog - Interactive Cell History Viewer

This dialog builds and visualizes cell histories, allowing you to:
1. Build cell-based organization from tracking data (ROI-aware)
2. Click on cells to see their complete time series
3. Select all long tracks (20+ frames) for batch export
4. Export animations with trajectory visualization
5. Validate the reorganization worked correctly

IMPORTANT: All analysis respects ROI boundaries when ROI is defined.
Only tracks with ≥50% of points within ROI are included.
"""

import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QSplitter,
    QWidget,
    QTextEdit,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from nd2_analyzer.analysis.cell_history import CellHistoryBuilder
from nd2_analyzer.analysis.roi_helper import ROIHelper


class CellViewDialog(QDialog):
    """
    Dialog for viewing and validating cell histories.

    Shows:
    - Table of all cells with their stats
    - Click a cell to see its complete time series
    - Validation that data reorganization worked
    """

    def __init__(self, lineage_tracks, metrics_service, image_data=None, parent=None):
        super().__init__(parent)

        self.lineage_tracks = lineage_tracks
        self.metrics_service = metrics_service
        self.image_data = image_data
        self.builder = None
        self.cell_database = None
        self.current_cell_id = None

        # Set dialog properties
        self.setWindowTitle("Cell View - Cell History Viewer")
        self.setMinimumWidth(1400)
        self.setMinimumHeight(900)

        # Initialize UI
        self.init_ui()

        # Build cell histories automatically
        self.build_histories()

    def init_ui(self):
        """Initialize the dialog UI"""
        layout = QVBoxLayout(self)

        # Status bar at top
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Building cell histories...")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        status_layout.addWidget(self.status_label)

        self.rebuild_button = QPushButton("Rebuild")
        self.rebuild_button.clicked.connect(self.build_histories)
        status_layout.addWidget(self.rebuild_button)

        layout.addLayout(status_layout)

        # Create splitter for left (table) and right (plots)
        splitter = QSplitter(Qt.Horizontal)

        # LEFT SIDE: Cell list table
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        left_layout.addWidget(QLabel("All Cells (click to view):"))

        # Cell table - allow multi-selection with Ctrl/Cmd
        self.cell_table = QTableWidget()
        self.cell_table.setColumnCount(6)
        self.cell_table.setHorizontalHeaderLabels([
            "Cell ID", "Lifespan", "Fate", "Displacement", "Avg Velocity", "Coverage %"
        ])
        self.cell_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.cell_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)  # Multi-select enabled
        self.cell_table.cellClicked.connect(self.on_cell_clicked)
        left_layout.addWidget(self.cell_table)

        # Filter controls
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))

        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "All Cells",
            "Long Cells (20+ frames)",
            "Divided Cells",
            "High Coverage (>90%)"
        ])
        self.filter_combo.currentTextChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.filter_combo)

        # Add "Select All Long Tracks" button
        self.select_long_button = QPushButton("Select All Long (20+)")
        self.select_long_button.clicked.connect(self.select_all_long_tracks)
        self.select_long_button.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                padding: 5px 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
        """)
        filter_layout.addWidget(self.select_long_button)

        left_layout.addLayout(filter_layout)

        splitter.addWidget(left_widget)

        # RIGHT SIDE: Visualization
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Cell info display
        self.cell_info_label = QLabel("Select a cell to view its history")
        self.cell_info_label.setStyleSheet("""
            background-color: #2b2b2b;
            color: #ffffff;
            padding: 15px;
            border-radius: 5px;
            font-size: 12px;
        """)
        right_layout.addWidget(self.cell_info_label)

        # Plots
        self.figure = plt.figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        # Export options
        from PySide6.QtWidgets import QCheckBox
        export_options_layout = QVBoxLayout()

        self.show_cells_checkbox = QCheckBox("Show cell segmentation in animation")
        self.show_cells_checkbox.setChecked(True)  # Default: show cells
        self.show_cells_checkbox.setStyleSheet("""
            QCheckBox {
                color: #ffffff;
                font-size: 11px;
                padding: 5px;
            }
        """)
        export_options_layout.addWidget(self.show_cells_checkbox)

        right_layout.addLayout(export_options_layout)

        # Action buttons
        validation_layout = QHBoxLayout()

        self.validate_button = QPushButton("🔍 Validate This Cell")
        self.validate_button.clicked.connect(self.validate_current_cell)
        self.validate_button.setEnabled(False)
        validation_layout.addWidget(self.validate_button)

        self.export_animation_button = QPushButton("🎬 Export Animation")
        self.export_animation_button.clicked.connect(self.export_animation)
        self.export_animation_button.setEnabled(False)
        validation_layout.addWidget(self.export_animation_button)

        self.export_button = QPushButton("💾 Export Cell Histories CSV")
        self.export_button.clicked.connect(self.export_csv)
        self.export_button.setEnabled(False)
        validation_layout.addWidget(self.export_button)

        right_layout.addLayout(validation_layout)

        splitter.addWidget(right_widget)

        # Set splitter sizes (30% table, 70% plots)
        splitter.setSizes([400, 1000])
        layout.addWidget(splitter)

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

    def build_histories(self):
        """Build cell histories using CellHistoryBuilder"""
        print("\n" + "="*80)
        print("🔄 Building cell histories from GUI...")
        print("="*80)

        try:
            # VALIDATE ROI BEFORE PROCESSING
            ROIHelper.validate_analysis_within_roi("Cell History Building")

            # Filter tracks by ROI if ROI is defined
            tracks_to_process = self.lineage_tracks
            if ROIHelper.has_roi():
                print("\n🎯 Filtering tracks by ROI (minimum 50% coverage)...")
                original_count = len(tracks_to_process)
                tracks_to_process = ROIHelper.filter_tracks_by_roi(tracks_to_process, min_roi_coverage=0.5)
                filtered_count = len(tracks_to_process)
                print(f"  Filtered: {original_count} tracks → {filtered_count} tracks (within ROI)")

                if filtered_count == 0:
                    QMessageBox.warning(
                        self,
                        "No Tracks in ROI",
                        "No tracks found within the ROI boundaries.\n\n"
                        "Please check your ROI selection or tracking parameters."
                    )
                    return

            # Create builder
            self.builder = CellHistoryBuilder(tracks_to_process, self.metrics_service)

            # Build with minimum 5 frames
            self.cell_database = self.builder.build(min_track_length=5)

            if not self.cell_database:
                QMessageBox.warning(self, "Error", "No cells were processed!")
                return

            # Build segmentation_id -> track_id mapping using frame 0 as reference
            self._build_seg_to_track_mapping()

            # Update status with ROI info
            num_cells = len(self.cell_database)
            long_cells = len([c for c in self.cell_database.values() if c['lifespan'] >= 20])

            roi_info = ROIHelper.get_roi_info()
            roi_text = " | ROI Active" if roi_info['has_roi'] else ""

            self.status_label.setText(
                f"✓ Built {num_cells} cell histories | {long_cells} long cells (20+ frames){roi_text}"
            )
            self.status_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; padding: 10px; color: green;"
            )

            # Populate table
            self.populate_table()

            # Enable export
            self.export_button.setEnabled(True)

            print(f"✓ Cell histories built successfully: {num_cells} cells")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to build cell histories:\n{str(e)}")

    def _build_seg_to_track_mapping(self):
        """Build mapping from segmentation cell IDs to track IDs using frame 0"""
        print("\n🔗 Building segmentation ID → track ID mapping...")

        # Get segmentation at frame 0
        from skimage.measure import regionprops, label
        model = self.image_data.segmentation_service.models.available_models[0]
        seg_mask = self.image_data.segmentation_cache.with_model(model)[(0, 0, 0)]

        # Check if already labeled or binary
        max_value = seg_mask.max()
        unique_values = len(np.unique(seg_mask))
        if max_value > 255 or unique_values > 100:
            labeled = seg_mask
        else:
            labeled = label(seg_mask)

        # For each track, find which segmentation ID it corresponds to at frame 0
        self.seg_to_track = {}  # segmentation_id -> track_id
        self.track_to_seg = {}  # track_id -> segmentation_id

        for track in self.lineage_tracks:
            track_id = track['ID']
            if 't' in track and 'x' in track and 'y' in track:
                # Find if this track exists at frame 0
                for i, t in enumerate(track['t']):
                    if t == 0:
                        x, y = int(track['x'][i]), int(track['y'][i])
                        if 0 <= y < labeled.shape[0] and 0 <= x < labeled.shape[1]:
                            seg_id = labeled[y, x]
                            if seg_id > 0:
                                self.seg_to_track[seg_id] = track_id
                                self.track_to_seg[track_id] = seg_id
                        break

        print(f"  Mapped {len(self.seg_to_track)} segmentation IDs to track IDs")

        # Build hierarchical lineage naming (e.g., 10.1, 10.1.2)
        self._build_lineage_names()

    def _build_lineage_names(self):
        """Build hierarchical names for cells based on lineage (e.g., 10, 10.1, 10.1.2)"""
        print("\n🌳 Building hierarchical lineage names...")

        self.track_to_lineage_name = {}  # track_id -> hierarchical name (e.g., "10.1.2")

        # First pass: Assign base names to cells at frame 0 (generation 0)
        for track_id, seg_id in self.track_to_seg.items():
            self.track_to_lineage_name[track_id] = str(seg_id)

        # Create a mapping of parent -> children for easier traversal
        parent_to_children = {}
        for track in self.lineage_tracks:
            track_id = track['ID']
            children = track.get('children', [])
            if children:
                parent_to_children[track_id] = children

        # Recursive function to assign names to descendants
        def assign_descendant_names(parent_id, parent_name):
            if parent_id not in parent_to_children:
                return

            children = parent_to_children[parent_id]
            for idx, child_id in enumerate(children, 1):
                # Child name = parent_name + "." + index
                child_name = f"{parent_name}.{idx}"
                self.track_to_lineage_name[child_id] = child_name

                # Recursively assign names to this child's descendants
                assign_descendant_names(child_id, child_name)

        # Start recursive naming from all generation 0 cells
        for track_id in self.track_to_seg.keys():
            base_name = self.track_to_lineage_name[track_id]
            assign_descendant_names(track_id, base_name)

        # Count how many cells have hierarchical names
        total_named = len(self.track_to_lineage_name)
        daughters = sum(1 for name in self.track_to_lineage_name.values() if '.' in name)
        print(f"  Created {total_named} hierarchical names ({daughters} daughter cells)")

        # Print some examples
        examples = list(self.track_to_lineage_name.items())[:5]
        print(f"  Examples:")
        for track_id, name in examples:
            print(f"    Track {track_id} → {name}")

    def populate_table(self, filter_func=None):
        """Populate the cell table with data"""
        self.cell_table.setRowCount(0)

        cells = self.cell_database.values()

        # Apply filter if provided
        if filter_func:
            cells = [c for c in cells if filter_func(c)]

        # Sort by segmentation ID (ascending) if available, otherwise by track ID
        if hasattr(self, 'track_to_seg'):
            cells = sorted(cells, key=lambda c: self.track_to_seg.get(c['cell_id'], c['cell_id']))
        else:
            cells = sorted(cells, key=lambda c: c['cell_id'])

        self.cell_table.setRowCount(len(cells))

        for row, cell in enumerate(cells):
            track_id = cell['cell_id']

            # Cell ID - show hierarchical lineage name (e.g., 10, 10.1, 10.1.2)
            if hasattr(self, 'track_to_lineage_name') and track_id in self.track_to_lineage_name:
                cell_id_text = self.track_to_lineage_name[track_id]
            elif hasattr(self, 'track_to_seg') and track_id in self.track_to_seg:
                seg_id = self.track_to_seg[track_id]
                cell_id_text = str(seg_id)
            else:
                cell_id_text = str(track_id)

            # Store track_id as hidden data for later retrieval
            item = QTableWidgetItem(cell_id_text)
            item.setData(Qt.UserRole, track_id)  # Store track ID as hidden data
            self.cell_table.setItem(row, 0, item)

            # Lifespan
            self.cell_table.setItem(row, 1, QTableWidgetItem(str(cell['lifespan'])))

            # Fate
            self.cell_table.setItem(row, 2, QTableWidgetItem(cell['fate']))

            # Displacement
            self.cell_table.setItem(row, 3, QTableWidgetItem(f"{cell['total_displacement']:.1f}"))

            # Avg Velocity
            self.cell_table.setItem(row, 4, QTableWidgetItem(f"{cell['avg_velocity']:.2f}"))

            # Coverage
            coverage_pct = cell['morphology_coverage'] * 100
            self.cell_table.setItem(row, 5, QTableWidgetItem(f"{coverage_pct:.1f}%"))

        self.cell_table.resizeColumnsToContents()

    def apply_filter(self, filter_name):
        """Apply filter to cell table"""
        if filter_name == "All Cells":
            self.populate_table()
        elif filter_name == "Long Cells (20+ frames)":
            self.populate_table(lambda c: c['lifespan'] >= 20)
        elif filter_name == "Divided Cells":
            self.populate_table(lambda c: c['fate'] == 'divided')
        elif filter_name == "High Coverage (>90%)":
            self.populate_table(lambda c: c['morphology_coverage'] > 0.9)

    def select_all_long_tracks(self):
        """Select all cells with lifespan >= 20 frames for batch export"""
        if not self.cell_database:
            return

        # Clear current selection
        self.cell_table.clearSelection()

        # Count long tracks
        long_track_count = 0

        # Use selection model to select multiple rows
        from PySide6.QtCore import QItemSelectionModel
        selection_model = self.cell_table.selectionModel()

        # Select all rows where lifespan >= 20
        for row in range(self.cell_table.rowCount()):
            # Get the lifespan from column 1
            lifespan_item = self.cell_table.item(row, 1)
            if lifespan_item:
                try:
                    lifespan = int(lifespan_item.text())
                    if lifespan >= 20:
                        # Select the entire row using the selection model
                        index = self.cell_table.model().index(row, 0)
                        selection_model.select(
                            index,
                            QItemSelectionModel.Select | QItemSelectionModel.Rows
                        )
                        long_track_count += 1
                except ValueError:
                    continue

        # Enable export button if tracks were selected
        if long_track_count > 0:
            self.export_animation_button.setEnabled(True)

        # Show message
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "Selection Complete",
            f"Selected {long_track_count} cells with lifespan ≥ 20 frames.\n\n"
            f"Click '🎬 Export Animation' to create a video of all selected tracks."
        )

        print(f"\n✓ Selected {long_track_count} long tracks (≥20 frames)")

    def on_cell_clicked(self, row, column):
        """Handle cell selection from table"""
        cell_id_item = self.cell_table.item(row, 0)
        if not cell_id_item:
            return

        # Get track ID from hidden data
        track_id = cell_id_item.data(Qt.UserRole)
        self.current_cell_id = track_id

        # Get cell data using track ID
        cell = self.builder.get_cell(track_id)
        if not cell:
            return

        # Update info display
        self.update_cell_info(cell)

        # Plot cell history
        self.plot_cell_history(cell)

        # Enable action buttons
        self.validate_button.setEnabled(True)
        self.export_animation_button.setEnabled(True)

    def update_cell_info(self, cell):
        """Update the cell info label"""
        info_text = f"""
<b>Cell ID:</b> {cell['cell_id']}<br>
<b>Lifespan:</b> {cell['lifespan']} frames<br>
<b>Fate:</b> {cell['fate']}<br>
<b>Time Range:</b> {cell['start_time']} to {cell['end_time']}<br>
<br>
<b>Movement:</b><br>
• Total Displacement: {cell['total_displacement']:.1f} pixels<br>
• Path Length: {cell['path_length']:.1f} pixels<br>
• Avg Velocity: {cell['avg_velocity']:.2f} pixels/frame<br>
• Directionality: {cell['directionality']:.3f}<br>
<br>
<b>Lineage:</b><br>
• Parent ID: {cell['parent_id'] if cell['parent_id'] else 'None'}<br>
• Children: {len(cell['children_ids'])} ({', '.join(map(str, cell['children_ids'])) if cell['children_ids'] else 'None'})<br>
<br>
<b>Data Quality:</b><br>
• Morphology Coverage: {cell['morphology_coverage']*100:.1f}%<br>
• Data Points Found: {cell['morphology_found']}/{cell['lifespan']}<br>
        """
        self.cell_info_label.setText(info_text)

    def plot_cell_history(self, cell):
        """Plot complete time series for a cell"""
        self.figure.clear()

        times = cell['times']

        # Create 2x2 subplot grid

        # Plot 1: Trajectory
        ax1 = self.figure.add_subplot(2, 2, 1)
        ax1.plot(cell['x_positions'], cell['y_positions'], 'b-', alpha=0.6, linewidth=2)
        ax1.plot(cell['x_positions'][0], cell['y_positions'][0], 'go', markersize=12, label='Start', zorder=5)
        ax1.plot(cell['x_positions'][-1], cell['y_positions'][-1], 'ro', markersize=12, label='End', zorder=5)
        ax1.set_title(f"Cell {cell['cell_id']} - Trajectory", fontsize=12, fontweight='bold')
        ax1.set_xlabel("X Position (pixels)")
        ax1.set_ylabel("Y Position (pixels)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Area over time
        ax2 = self.figure.add_subplot(2, 2, 2)
        areas = cell['area']
        valid_idx = [i for i, a in enumerate(areas) if not np.isnan(a)]
        if valid_idx:
            valid_times = [times[i] for i in valid_idx]
            valid_areas = [areas[i] for i in valid_idx]
            ax2.plot(valid_times, valid_areas, 'b-o', markersize=4, linewidth=2)
            ax2.set_title("Area Over Time", fontsize=12, fontweight='bold')
            ax2.set_xlabel("Time (frames)")
            ax2.set_ylabel("Area (pixels²)")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No area data', ha='center', va='center', transform=ax2.transAxes)

        # Plot 3: Length over time
        ax3 = self.figure.add_subplot(2, 2, 3)
        lengths = cell['length']
        valid_idx = [i for i, l in enumerate(lengths) if not np.isnan(l)]
        if valid_idx:
            valid_times = [times[i] for i in valid_idx]
            valid_lengths = [lengths[i] for i in valid_idx]
            ax3.plot(valid_times, valid_lengths, 'g-o', markersize=4, linewidth=2)
            ax3.set_title("Length Over Time", fontsize=12, fontweight='bold')
            ax3.set_xlabel("Time (frames)")
            ax3.set_ylabel("Length (pixels)")
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No length data', ha='center', va='center', transform=ax3.transAxes)

        # Plot 4: Aspect Ratio over time
        ax4 = self.figure.add_subplot(2, 2, 4)
        aspect_ratios = cell['aspect_ratio']
        valid_idx = [i for i, ar in enumerate(aspect_ratios) if not np.isnan(ar)]
        if valid_idx:
            valid_times = [times[i] for i in valid_idx]
            valid_ratios = [aspect_ratios[i] for i in valid_idx]
            ax4.plot(valid_times, valid_ratios, 'r-o', markersize=4, linewidth=2)
            ax4.set_title("Aspect Ratio Over Time", fontsize=12, fontweight='bold')
            ax4.set_xlabel("Time (frames)")
            ax4.set_ylabel("Aspect Ratio")
            ax4.grid(True, alpha=0.3)

            # Add state thresholds
            ax4.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Deforming threshold')
            ax4.axhline(y=15, color='red', linestyle='--', alpha=0.5, label='Deformed threshold')
            ax4.legend(fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'No aspect ratio data', ha='center', va='center', transform=ax4.transAxes)

        self.figure.tight_layout()
        self.canvas.draw()

    def validate_current_cell(self):
        """Validate the currently selected cell by comparing to original data"""
        if not self.current_cell_id:
            return

        cell = self.builder.get_cell(self.current_cell_id)

        print("\n" + "="*80)
        print(f"🔍 VALIDATING CELL {self.current_cell_id}")
        print("="*80)

        # Pick 3 random timepoints to validate
        num_checks = min(3, len(cell['times']))
        if num_checks == 0:
            QMessageBox.warning(self, "Validation", "No timepoints to validate!")
            return

        check_indices = np.random.choice(len(cell['times']), num_checks, replace=False)

        validation_results = []
        all_valid = True

        for idx in check_indices:
            t = cell['times'][idx]
            x = cell['x_positions'][idx]
            y = cell['y_positions'][idx]
            area_from_db = cell['area'][idx]
            length_from_db = cell['length'][idx]

            print(f"\nChecking timepoint {t}:")
            print(f"  Cell database says:")
            print(f"    Position: ({x:.1f}, {y:.1f})")
            print(f"    Area: {area_from_db:.1f}")
            print(f"    Length: {length_from_db:.1f}")

            # Query original metrics_service
            metrics_df = self.metrics_service.query_optimized(time=t, cell_id=self.current_cell_id)

            if not metrics_df.is_empty():
                row = metrics_df.row(0, named=True)
                area_from_metrics = row['area']
                length_from_metrics = row['major_axis_length']

                print(f"  Metrics service says:")
                print(f"    Position: ({row['centroid_x']:.1f}, {row['centroid_y']:.1f})")
                print(f"    Area: {area_from_metrics:.1f}")
                print(f"    Length: {length_from_metrics:.1f}")

                # Check if they match
                area_match = abs(area_from_db - area_from_metrics) < 0.1 if not np.isnan(area_from_db) else False
                length_match = abs(length_from_db - length_from_metrics) < 0.1 if not np.isnan(length_from_db) else False

                if area_match and length_match:
                    print(f"  ✅ VERIFIED")
                    validation_results.append(f"✅ t={t}: Data matches")
                else:
                    print(f"  ⚠️ MISMATCH")
                    validation_results.append(f"⚠️ t={t}: Data mismatch!")
                    all_valid = False
            else:
                print(f"  ⚠️ No morphology data in metrics_service")
                if np.isnan(area_from_db):
                    validation_results.append(f"✅ t={t}: Correctly marked as NaN")
                else:
                    validation_results.append(f"⚠️ t={t}: Has data but shouldn't")
                    all_valid = False

        # Show results dialog
        result_text = f"Validation Results for Cell {self.current_cell_id}:\n\n"
        result_text += "\n".join(validation_results)
        result_text += f"\n\nOverall: {'✅ All checks passed!' if all_valid else '⚠️ Some mismatches detected'}"

        QMessageBox.information(self, "Validation Results", result_text)

    def export_csv(self):
        """Export cell database to CSV"""
        from PySide6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Cell Histories",
            "cell_histories.csv",
            "CSV Files (*.csv)"
        )

        if file_path:
            try:
                self.builder.export_to_csv(file_path)
                QMessageBox.information(self, "Export Complete", f"Cell histories exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export:\n{str(e)}")

    def _expand_selection_with_lineage(self, track_ids):
        """
        Expand track selection to include daughter cells when parent is selected.
        This ensures division events are fully visualized.

        Args:
            track_ids: List of initially selected track IDs

        Returns:
            tuple: (expanded_track_ids, division_events)
                - expanded_track_ids: List including daughters
                - division_events: Dict mapping parent_id -> [daughter1_id, daughter2_id, division_frame]
        """
        expanded_ids = set(track_ids)
        division_events = {}

        for track_id in track_ids:
            # Find this track
            for track in self.lineage_tracks:
                if track['ID'] == track_id:
                    # Check if this track has children (divided)
                    children = track.get('children', [])
                    if children and len(children) > 0:
                        # Add daughter cells to selection
                        expanded_ids.update(children)

                        # Find division frame (last frame of parent)
                        if 't' in track and len(track['t']) > 0:
                            division_frame = track['t'][-1]
                            division_events[track_id] = {
                                'daughters': children,
                                'frame': division_frame,
                                'position': (track['x'][-1], track['y'][-1]) if 'x' in track and 'y' in track else None
                            }
                            print(f"  Division detected: Cell {track_id} → {children} at frame {division_frame}")
                    break

        return list(expanded_ids), division_events

    def export_animation(self):
        """Export animation of selected cells to GIF file"""
        from PySide6.QtWidgets import QFileDialog
        import cv2

        # Check if we have image data
        if self.image_data is None:
            QMessageBox.warning(self, "No Image Data", "Image data is not available for animation.")
            return

        # Get selected cell IDs from table
        selected_rows = self.cell_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select one or more cells to animate.")
            return

        # Get track IDs from selected rows
        selected_track_ids = []
        selected_display_ids = []  # For filename
        print(f"\nSelected {len(selected_rows)} rows:")
        for row in selected_rows:
            row_num = row.row()
            cell_id_item = self.cell_table.item(row_num, 0)
            if cell_id_item:
                # Get track ID from hidden data
                track_id = cell_id_item.data(Qt.UserRole)
                seg_id = cell_id_item.text()
                print(f"  Row {row_num}: Cell {seg_id} → Track {track_id}")
                selected_track_ids.append(track_id)
                selected_display_ids.append(seg_id)

        # Expand selection to include daughter cells for division visualization
        expanded_track_ids, division_events = self._expand_selection_with_lineage(selected_track_ids)

        if len(expanded_track_ids) > len(selected_track_ids):
            print(f"  Expanded selection: {len(selected_track_ids)} → {len(expanded_track_ids)} cells (including daughters)")

        print(f"\n🎬 Exporting animation for tracks: {expanded_track_ids}")

        # Get cells and find frame range (using expanded track IDs)
        selected_cells = []
        min_frame = float('inf')
        max_frame = 0

        for track_id in expanded_track_ids:
            cell = self.builder.get_cell(track_id)
            if cell:
                selected_cells.append(cell)
                min_frame = min(min_frame, cell['start_time'])
                max_frame = max(max_frame, cell['end_time'])

        if not selected_cells:
            QMessageBox.warning(self, "Error", "Could not load cell data.")
            return

        print(f"Frame range: {min_frame} to {max_frame}")
        print(f"Total frames: {max_frame - min_frame + 1}")

        # Ask user for save location
        default_name = f"cell_animation_{'_'.join(map(str, selected_display_ids[:3]))}.gif"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Cell Animation",
            default_name,
            "GIF Animation (*.gif)"
        )

        if not file_path:
            return  # User cancelled

        # Export the animation
        try:
            show_cells = self.show_cells_checkbox.isChecked()
            self._export_to_video(selected_cells, expanded_track_ids, min_frame, max_frame, file_path, show_cells, division_events)
            QMessageBox.information(self, "Export Complete", f"Animation exported to:\n{file_path}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Export Error", f"Failed to export animation:\n{str(e)}")

    def _export_to_video(self, cells, track_ids, start_frame, end_frame, output_path, show_cells=True, division_events=None):
        """Export animation frames to GIF (cropped to ROI if available, with division visualization)"""
        import cv2
        import imageio
        from skimage.measure import label

        if division_events is None:
            division_events = {}

        print(f"\n{'='*60}")
        print(f"🎬 EXPORTING ANIMATION TO GIF")
        print(f"{'='*60}")
        print(f"Selected track IDs: {track_ids}")
        print(f"Show cells: {show_cells}")
        print(f"Division events: {len(division_events)}")

        # Get model being used for segmentation
        model = self.image_data.segmentation_service.models.available_models[0]
        print(f"Using segmentation model: {model}")

        # Get first frame to determine video size
        first_seg = self.image_data.segmentation_cache.with_model(model)[(start_frame, 0, 0)]
        full_height, full_width = first_seg.shape

        # Check if ROI exists and get crop bounds
        roi_mask = ROIHelper.get_roi_mask()
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = 0, 0, full_width, full_height

        if roi_mask is not None:
            # Find ROI bounding box
            roi_coords = np.argwhere(roi_mask > 0)
            if len(roi_coords) > 0:
                crop_y_min = int(roi_coords[:, 0].min())
                crop_y_max = int(roi_coords[:, 0].max()) + 1
                crop_x_min = int(roi_coords[:, 1].min())
                crop_x_max = int(roi_coords[:, 1].max()) + 1
                print(f"🎯 ROI detected - Cropping video to ROI bounding box")
                print(f"  ROI bounds: X[{crop_x_min}:{crop_x_max}], Y[{crop_y_min}:{crop_y_max}]")
            else:
                print(f"⚠️ ROI mask is empty, using full frame")
        else:
            print(f"ℹ️ No ROI defined, using full frame")

        # Calculate cropped dimensions
        width = crop_x_max - crop_x_min
        height = crop_y_max - crop_y_min

        # Collect frames for GIF
        frames = []
        duration_per_frame = 1000  # milliseconds (1 FPS - liquid slow motion for clear division viewing)

        print(f"GIF size: {width}x{height} (cropped from {full_width}x{full_height})")
        print(f"Duration per frame: {duration_per_frame}ms (slower playback)")
        print(f"Processing frames {start_frame} to {end_frame}...")

        # Create set of selected track IDs
        selected_track_ids = set(track_ids)

        # Process each frame
        for frame_num in range(start_frame, end_frame + 1):
            # Get segmentation mask for this frame
            seg_mask = self.image_data.segmentation_cache.with_model(model)[(frame_num, 0, 0)]

            # Check if already labeled or binary (same logic as view_area)
            max_value = seg_mask.max()
            unique_values = len(np.unique(seg_mask))
            if max_value > 255 or unique_values > 100:
                labeled = seg_mask  # Already labeled
            else:
                labeled = label(seg_mask)  # Binary mask, label it

            # Crop labeled mask to ROI bounds
            labeled_cropped = labeled[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

            # Create frame based on show_cells option
            if show_cells:
                # Apply colormap to get colored labels
                colored_frame = self._apply_colormap_with_labels(labeled_cropped)
            else:
                # Create black background (just tracks, no cells)
                colored_frame = np.zeros((height, width, 3), dtype=np.uint8)

            # For each selected track ID, draw trajectory and highlight current position
            for track_id in selected_track_ids:
                # Find this track in lineage_tracks
                track_data = None
                for track in self.lineage_tracks:
                    if track['ID'] == track_id and 't' in track and 'x' in track and 'y' in track:
                        track_data = track
                        break

                if not track_data:
                    continue

                # Collect all positions from start_frame up to current frame_num
                # Adjust coordinates by crop offset
                trajectory_points = []
                current_position = None
                start_position = None

                for i, t in enumerate(track_data['t']):
                    if start_frame <= t <= frame_num:
                        # Original coordinates
                        x_orig = int(track_data['x'][i])
                        y_orig = int(track_data['y'][i])
                        # Adjust for crop
                        point = (x_orig - crop_x_min, y_orig - crop_y_min)
                        trajectory_points.append(point)

                        if start_position is None:
                            start_position = point

                        if t == frame_num:
                            current_position = point

                # Adjust visualization based on show_cells mode
                if show_cells:
                    # Full mode: thicker lines, larger markers
                    line_thickness = 3
                    marker_size = 8
                    marker_outline = 2
                else:
                    # Minimalist mode: thinner lines, smaller markers
                    line_thickness = 1
                    marker_size = 3
                    marker_outline = 1

                # Draw trajectory line if we have points
                if len(trajectory_points) > 1:
                    # Draw the trajectory line (thickness varies by mode)
                    for i in range(len(trajectory_points) - 1):
                        cv2.line(colored_frame, trajectory_points[i], trajectory_points[i + 1],
                                (255, 255, 255), line_thickness, cv2.LINE_AA)

                # Draw start marker (green circle)
                if start_position:
                    cv2.circle(colored_frame, start_position, marker_size, (0, 255, 0), -1)
                    if show_cells:
                        cv2.circle(colored_frame, start_position, marker_size, (255, 255, 255), marker_outline)

                # Draw current position marker and cell representation
                if current_position:
                    x, y = current_position

                    # In minimalist mode, draw a small rounded shape to represent the cell
                    if not show_cells:
                        # Draw small filled circle representing cell body (light gray)
                        cell_body_radius = 8
                        cv2.circle(colored_frame, current_position, cell_body_radius, (80, 80, 80), -1)
                        # Draw outline
                        cv2.circle(colored_frame, current_position, cell_body_radius, (150, 150, 150), 1)

                    # Draw current position marker (red circle)
                    cv2.circle(colored_frame, current_position, marker_size, (0, 0, 255), -1)
                    if show_cells:
                        cv2.circle(colored_frame, current_position, marker_size, (255, 255, 255), marker_outline)

                    # Get hierarchical lineage name for this track
                    if hasattr(self, 'track_to_lineage_name') and track_id in self.track_to_lineage_name:
                        label_text = self.track_to_lineage_name[track_id]
                    elif hasattr(self, 'track_to_seg') and track_id in self.track_to_seg:
                        # Fallback to segmentation ID
                        label_text = str(self.track_to_seg[track_id])
                    else:
                        # Last resort: track ID
                        label_text = f"T{track_id}"
                    label_pos = (x + 12, y - 12)  # Offset from marker

                    # Draw text with background for visibility
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # Smaller font in minimalist mode
                    if show_cells:
                        font_scale = 0.5
                        thickness = 1
                    else:
                        font_scale = 0.4
                        thickness = 1

                    # Get text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                    # Draw black background rectangle
                    bg_x1 = label_pos[0] - 2
                    bg_y1 = label_pos[1] - text_height - 2
                    bg_x2 = label_pos[0] + text_width + 2
                    bg_y2 = label_pos[1] + 2
                    cv2.rectangle(colored_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)

                    # Draw white text on top
                    cv2.putText(colored_frame, label_text, label_pos,
                              font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # Draw division markers for cells that divide at this frame
            for parent_id, div_info in division_events.items():
                if div_info['frame'] == frame_num and div_info['position'] is not None:
                    # Get division position with crop offset applied
                    div_x_orig, div_y_orig = div_info['position']
                    div_x = int(div_x_orig) - crop_x_min
                    div_y = int(div_y_orig) - crop_y_min

                    # Draw gold star at division point
                    star_size = 15 if show_cells else 12
                    cv2.drawMarker(colored_frame, (div_x, div_y),
                                  (0, 215, 255), cv2.MARKER_STAR,
                                  markerSize=star_size, thickness=2)

                    # Optional: Add small "DIV" text near division marker
                    if show_cells:
                        cv2.putText(colored_frame, "DIV", (div_x + 10, div_y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 215, 255), 1, cv2.LINE_AA)

            # Add frame number
            cv2.putText(colored_frame, f"Frame: {frame_num}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Convert BGR to RGB for GIF (cv2 uses BGR, imageio expects RGB)
            colored_frame_rgb = cv2.cvtColor(colored_frame, cv2.COLOR_BGR2RGB)
            frames.append(colored_frame_rgb)

            if (frame_num - start_frame) % 10 == 0:
                print(f"  Processed frame {frame_num}/{end_frame}")

        # Save as GIF with slower playback
        print(f"💾 Saving GIF with {len(frames)} frames...")
        imageio.mimsave(output_path, frames, duration=duration_per_frame, loop=0)
        print(f"✅ GIF exported successfully!")
        print(f"{'='*60}\n")

    def _apply_colormap_with_labels(self, segmented):
        """Apply distinct colors and add cell ID labels (same as segmentation_service)"""
        import numpy as np
        import matplotlib.colors as mcolors
        import cv2
        from skimage.measure import regionprops

        labels = segmented
        n_labels = labels.max()

        # Generate random hues with fixed high saturation and value for vivid colors
        np.random.seed(42)  # Same seed as segmentation_service for consistency
        hues = np.random.permutation(n_labels) / n_labels

        # Create color lookup table
        lut = np.zeros((n_labels + 1, 3))
        lut[0] = [0, 0, 0]  # Background is black

        for i in range(1, n_labels + 1):
            h = hues[i - 1]
            s = 0.8 + 0.2 * np.random.rand()
            v = 0.8 + 0.2 * np.random.rand()
            lut[i] = mcolors.hsv_to_rgb([h, s, v])

        # Map labels to colors
        colored = lut[labels]
        colored = (colored * 255).astype(np.uint8)

        # Add cell ID text labels
        regions = regionprops(labels)
        for region in regions:
            cell_id = region.label
            centroid_y, centroid_x = region.centroid
            x, y = int(centroid_x), int(centroid_y)

            # Add white text with black outline
            text = str(cell_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1

            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = x - text_width // 2
            text_y = y + text_height // 2

            # Black outline
            cv2.putText(colored, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            # White text on top
            cv2.putText(colored, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return colored
