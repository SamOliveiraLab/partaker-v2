import json
import os
import sys

import numpy as np
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvas
from pubsub import pub

from nd2_analyzer.analysis.metrics_service import MetricsService
from nd2_analyzer.analysis.morphology.morphology import (
    annotate_binary_mask,
)
from nd2_analyzer.data.experiment import Experiment
from nd2_analyzer.data.image_data import ImageData
from .dialogs import AboutDialog, ExperimentDialog
from nd2_analyzer.ui.dialogs.export_raw import ExportDialog
from nd2_analyzer.ui.dialogs.roisel import PolygonROISelector
from nd2_analyzer.ui.dialogs.crop_selection import CropSelector
from nd2_analyzer.ui.biofilms.analysis_mode import AnalysisMode, AnalysisModeConfig
from nd2_analyzer.ui.dialogs.mode_selection_dialog import ModeSelectionDialog
from nd2_analyzer.ui.biofilms.config.biofilm_config import BiofilmConfig
from nd2_analyzer.ui.biofilms.cube_analysis import CubeAnalysisWidget
from nd2_analyzer.ui.biofilms.eps_analysis import EPSAnalysisWidget
from nd2_analyzer.ui.biofilms.colony_separation_widget import ColonySeparationWidget
from .widgets import (
    ViewAreaWidget,
    PopulationWidget,
    SegmentationWidget,
    MorphologyWidget,
    TrackingManager,
)
from ..data.appstate import ApplicationState


class App(QMainWindow):

    SUPPORTED_EXTENSIONS = (".nd2", ".tif", ".tiff", ".ome.tif", ".ome.tiff",
                            ".ome.btf", ".ome.tf2", ".ome.tf8")

    def __init__(self):
        super().__init__()

        self.appstate = ApplicationState.create_instance()

        # Initialize analysis mode configuration FIRST
        self.analysis_config = AnalysisModeConfig()
        self.biofilm_config = BiofilmConfig()

        # Show mode selection dialog
        selected_mode = ModeSelectionDialog.show_mode_selection(
            self.analysis_config, self)
        self.current_analysis_mode = selected_mode

        self.setWindowTitle("Partaker 2 - GUI")
        self.setGeometry(100, 100, 1000, 800)
        self.setAcceptDrops(True)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        self.viewArea = ViewAreaWidget()
        self.layout.addWidget(self.viewArea)

        # Initialize tabs based on analysis mode
        self.tab_widget = QTabWidget()
        if self.current_analysis_mode == AnalysisMode.SINGLE_CELL:
            self.init_single_cell_tabs()
        elif self.current_analysis_mode == AnalysisMode.BIOFILM_CLOUD:
            self.init_biofilm_cloud_tabs()

        self.layout.addWidget(self.tab_widget)

        self.initMenuBar()

        # TODO: do we need to subscribe here?
        # Subscribe to events
        pub.subscribe(self.on_exp_loaded, "experiment_loaded")
        pub.subscribe(self.on_image_request, "image_request")
        pub.subscribe(self.on_segmentation_request, "segmentation_request")
        pub.subscribe(self.on_draw_cell_bounding_boxes, "draw_cell_bounding_boxes")
        # Note: highlight_cell is handled by ViewAreaWidget, not here
        # pub.subscribe(self.highlight_cell, "highlight_cell_requested")

        pub.subscribe(self.provide_image_data, "get_image_data")

    def init_single_cell_tabs(self):
        """Initialize tabs for single cell analysis mode"""
        self.segmentation_tab = SegmentationWidget()
        self.populationTab = PopulationWidget()
        self.morphologyTab = MorphologyWidget()
        self.trackingTab = TrackingManager()

        self.tab_widget.addTab(self.segmentation_tab, "Segmentation")
        self.tab_widget.addTab(self.populationTab, "Population")
        self.tab_widget.addTab(self.morphologyTab, "Morphology")
        self.tab_widget.addTab(self.trackingTab, "Tracking")

    def init_biofilm_cloud_tabs(self):
        """Initialize tabs for biofilm cloud analysis mode"""
        self.segmentation_tab = SegmentationWidget()  # Same segmentation
        self.colony_separation_tab = ColonySeparationWidget()  # New colony separation tab
        self.cube_analysis_tab = CubeAnalysisWidget()
        self.eps_analysis_tab = EPSAnalysisWidget()

        self.tab_widget.addTab(self.segmentation_tab, "Segmentation")
        self.tab_widget.addTab(self.colony_separation_tab, "Colony Separation")
        self.tab_widget.addTab(self.cube_analysis_tab, "Colony Analysis")
        self.tab_widget.addTab(self.eps_analysis_tab, "EPS Analysis")

    # ── Drag & Drop ──────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        mime = event.mimeData()
        if mime.hasUrls():
            for url in mime.urls():
                path = url.toLocalFile()
                if os.path.isdir(path) or path.lower().endswith(self.SUPPORTED_EXTENSIONS):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        mime = event.mimeData()
        if not mime.hasUrls():
            event.ignore()
            return

        directories = []
        files = []
        for url in mime.urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                directories.append(path)
            elif path.lower().endswith(self.SUPPORTED_EXTENSIONS):
                files.append(path)

        if not directories and not files:
            event.ignore()
            return

        event.acceptProposedAction()

        if directories:
            dialog = ExperimentDialog(parent=self, initial_directory=directories[0])
        else:
            dialog = ExperimentDialog(parent=self, initial_files=files)
        dialog.exec_()

    # ── Data helpers ─────────────────────────────────────────────────

    def provide_image_data(self, callback):
        """Provide the image_data object through the callback"""
        if hasattr(self, "image_data"):
            print("Providing image_data to callback")
            callback(self.application_state.image_data)
        else:
            print("No image_data available")
            callback(None)

    def on_exp_loaded(self, experiment: Experiment):
        """Handle experiment loaded event.
        Image data is already loaded by create_experiment; this handles cleanup only."""

        # Clear the previous temporary file if it exists
        old = self.appstate.image_data
        self.appstate.image_data = None
        import gc
        gc.collect()
        if hasattr(old, "_memmap_file"):
            import os
            os.remove(old._memmap_file)

    @staticmethod
    def reconstruct_file_map_from_paths(paths):
        """
        Given a list of TIFF file paths, reconstruct the file_map
        mapping (position, time, channel) -> full path.
        """
        import os
        import re
        file_map = {}
        # Accept both "..._c2.tif" and "..._2.tif" channel suffixes.
        batch_pattern = re.compile(
            r"^pos(?P<p>\d+)_t(?P<t>\d+)_(?:c)?(?P<c>\d+)\.(tif|tiff)$",
            re.IGNORECASE,
        )
        stacked_pattern = re.compile(
            r"^pos(?P<p>\d+)_(?:c)?(?P<c>\d+)\.(tif|tiff)$",
            re.IGNORECASE,
        )

        # Check if any of the paths match the batch pattern, reconstructing the file_map
        for full_path in paths:
            name = os.path.basename(full_path)
            m = batch_pattern.match(name)
            if m:
                p, t, c = map(int, (m["p"], m["t"], m["c"]))
                file_map[(p, t, c)] = full_path
                continue

            m2 = stacked_pattern.match(name)
            if m2:
                p, c = map(int, (m2["p"], m2["c"]))
                file_map[(p, None, c)] = full_path
        return file_map

    @staticmethod
    def _reconstruct_tiff_file_map(raw_map):
        """Convert serialized tiff_file_map keys back to tuple keys."""
        file_map = {}
        for key_str, path in raw_map.items():
            parts = key_str.split(",")
            p = int(parts[0])
            t = int(parts[1]) if parts[1] != "None" else None
            c = int(parts[2])
            file_map[(p, t, c)] = path
        return file_map

    @staticmethod
    def _remap_tiff_file_map_by_filename(file_map, new_root):
        """
        Remap TIFF paths by matching filenames under a new root directory.
        Raises FileNotFoundError if any entry cannot be matched uniquely.
        """
        filename_index = {}
        for root, _, files in os.walk(new_root):
            for name in files:
                if not name.lower().endswith((".tif", ".tiff")):
                    continue
                key = name.lower()
                full_path = os.path.join(root, name)
                filename_index.setdefault(key, []).append(full_path)

        remapped = {}
        unresolved = []
        ambiguous = []
        for key, old_path in file_map.items():
            matches = filename_index.get(os.path.basename(old_path).lower(), [])
            if len(matches) == 1:
                remapped[key] = matches[0]
            elif len(matches) == 0:
                unresolved.append(old_path)
            else:
                ambiguous.append((old_path, matches))

        if unresolved:
            preview = ", ".join(os.path.basename(p) for p in unresolved[:5])
            raise FileNotFoundError(
                f"Could not find {len(unresolved)} TIFF file(s) in the selected folder "
                f"(examples: {preview})."
            )

        if ambiguous:
            preview = ", ".join(os.path.basename(p[0]) for p in ambiguous[:5])
            raise FileNotFoundError(
                f"Found duplicate TIFF filenames for {len(ambiguous)} file(s) "
                f"(examples: {preview}). Please select a more specific folder."
            )

        return remapped

    def on_image_request(self, time, position, channel):
        """Handle requests for raw image data"""
        if not self.application_state.image_data:
            return

        # Retrieve the image data as 5D array
        try:
            image = self.application_state.image_data.data[time, position, channel]

            # Convert to NumPy array if needed
            image = np.array(image)

            # Publish the image
            pub.sendMessage(
                "image_ready",
                image=image,
                time=time,
                position=position,
                channel=channel,
            )
        except Exception as e:
            print(f"Error retrieving image: {e}")

    def on_segmentation_request(self, time, position, channel, model_name):
        """Handle requests for segmentation data"""
        if not self.application_state.image_data:
            return

        try:
            # Get the appropriate segmentation model
            if model_name:
                self.application_state.image_data.segmentation_cache.with_model(
                    model_name
                )
            else:
                self.application_state.image_data.segmentation_cache.with_model(
                    self.model_dropdown.currentText()
                )

            # Get the segmentation
            segmented_image = self.application_state.image_data.segmentation_cache[
                time, position, channel
            ]

            # Publish the segmentation result
            pub.sendMessage(
                "segmentation_ready",
                segmented_image=segmented_image,
                time=time,
                position=position,
                channel=channel,
            )
        except Exception as e:
            print(f"Error retrieving segmentation: {e}")

    def open_roi_selector(self):
        roi_dialog = PolygonROISelector()
        roi_dialog.exec_()  # Use exec_ to make it modal

    def open_crop_selector(self):
        crop_dialog = CropSelector()
        crop_dialog.exec_()

    def on_draw_cell_bounding_boxes(self, time, position, channel, cell_mapping):
        """Handle request to draw cell bounding boxes"""
        # Get the segmentation using the same model as the segmentation cache
        if not hasattr(self, "image_data") or not self.application_state.image_data:
            print("No image data available")
            return

        # Get the current model from the cache
        current_model = self.application_state.image_data.segmentation_cache.model_name
        if not current_model:
            # Default to a standard model if none set
            current_model = "bact_phase_cp3"  # This is CELLPOSE_BACT_PHASE
            self.application_state.image_data.segmentation_cache.with_model(
                current_model
            )

        # Get the segmentation data
        segmented_image = self.application_state.image_data.segmentation_cache[
            time, position, channel
        ]

        if segmented_image is None:
            print(f"No segmentation available for T={time}, P={position}, C={channel}")
            return

        # Create annotated image
        annotated_image = annotate_binary_mask(segmented_image, cell_mapping)

        # Display on the view area's image label
        height, width = annotated_image.shape[:2]
        qimage = QImage(
            annotated_image.data,
            width,
            height,
            annotated_image.strides[0],
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.viewArea.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.viewArea.image_label.setPixmap(pixmap)

        # Store the annotated image
        if hasattr(self, "annotated_image"):
            self.annotated_image = annotated_image

        self.current_cell_mapping = cell_mapping

    def initMenuBar(self):
        # Create the menu bar
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")

        # New Experiment
        experiment_action = QAction("Experiment", self)
        experiment_action.setShortcut("Ctrl+E")
        experiment_action.triggered.connect(self.show_experiment_dialog)
        file_menu.addAction(experiment_action)

        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_to_folder)
        file_menu.addAction(save_action)

        load_action = QAction("Load", self)
        load_action.setShortcut("Ctrl+L")
        load_action.triggered.connect(self.load_from_folder)
        file_menu.addAction(load_action)

        export_raw_action = QAction("Export Raw to TIF...", self)
        export_raw_action.setShortcut("Ctrl+Shift+E")
        export_raw_action.triggered.connect(self.show_export_raw_dialog)
        file_menu.addAction(export_raw_action)

        file_menu.addSeparator()

        switch_mode_action = QAction("Switch Analysis Mode", self)
        switch_mode_action.setShortcut("Ctrl+M")
        switch_mode_action.triggered.connect(self.switch_analysis_mode)
        file_menu.addAction(switch_mode_action)

        help_menu = menu_bar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        # --- Image menu
        image_menu = menu_bar.addMenu("Image")

        crop_action = QAction("Crop", self)
        crop_action.setShortcut("Ctrl+K")
        crop_action.triggered.connect(self.open_crop_selector)
        image_menu.addAction(crop_action)

        reset_crop_action = QAction("Reset Crop", self)
        reset_crop_action.triggered.connect(self.on_reset_crop)
        image_menu.addAction(reset_crop_action)

        registration_action = QAction("Registration", self)
        registration_action.triggered.connect(self.on_registration_pressed)
        image_menu.addAction(registration_action)

        roi_action = QAction("ROI", self)
        roi_action.setShortcut("Ctrl+R")
        roi_action.triggered.connect(self.open_roi_selector)
        image_menu.addAction(roi_action)

        reset_roi_action = QAction("Reset ROI", self)
        reset_roi_action.triggered.connect(self.on_reset_roi)
        image_menu.addAction(reset_roi_action)

        # --- Test menu
        test_menu = menu_bar.addMenu("Test")
        test_action = QAction("Test", self)
        test_action.setShortcut("Ctrl+T")
        test_action.triggered.connect(self.hhln_test)
        test_menu.addAction(test_action)

        if sys.platform == "darwin":
            about_action_mac = QAction("About", self)
            about_action_mac.triggered.connect(self.show_about_dialog)
            self.menuBar().addAction(about_action_mac)

    def on_registration_pressed(self):
        # Get current P from ViewArea
        ImageData.get_instance().do_registration_p(self.viewArea.current_p)

    def on_reset_crop(self):
        pub.sendMessage("crop_reset")

    def on_reset_roi(self):
        pub.sendMessage("roi_reset")

    def hhln_test(self):
        ImageData.load_nd2(
            "/Users/hiram/Documents/EVERYTHING/20-29 Research/22 OliveiraLab/22.12 ND2 analyzer/nd2-analyzer/final_data/rpu_nd2/3-RPU_mCherry_M9Plain_10h002.nd2"
        )

    def switch_analysis_mode(self):
        """Re-open the mode selection dialog and rebuild the UI tabs"""
        dialog = ModeSelectionDialog(self.analysis_config, self)
        result = dialog.exec()
        if result != QDialog.Accepted:
            return

        new_mode = dialog.get_selected_mode()
        if new_mode == self.current_analysis_mode:
            return

        self.current_analysis_mode = new_mode
        self.setWindowTitle(f"Partaker 3 - {new_mode.display_name}")

        while self.tab_widget.count():
            self.tab_widget.removeTab(0)

        if new_mode == AnalysisMode.SINGLE_CELL:
            self.init_single_cell_tabs()
        elif new_mode == AnalysisMode.BIOFILM_CLOUD:
            self.init_biofilm_cloud_tabs()

    def show_experiment_dialog(self):
        experiment = ExperimentDialog()
        experiment.exec_()

    def show_about_dialog(self):
        about_dialog = AboutDialog()
        about_dialog.exec_()

    def show_export_raw_dialog(self):
        """Open the raw export dialog. Pre-fills current P/C from the view area."""
        if ImageData.get_instance() is None:
            QMessageBox.warning(self, "No Data", "Load image data before exporting.")
            return
        export_dialog = ExportDialog(
            self,
            current_p=self.viewArea.current_p,
            current_c=self.viewArea.current_c,
        )
        export_dialog.exec_()

    def save_to_folder(self):
        """Save the current project to a folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Destination Folder", "", QFileDialog.ShowDirsOnly
        )

        if folder_path:
            # Create directory if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)

            # TODO: think of a better way of seeing this information
            # # Log segmentation cache state before saving
            # if hasattr(self, "image_data") and hasattr(self.application_state.image_data, "segmentation_cache"):
            #     cache = self.application_state.image_data.segmentation_cache
            #     current_model = cache.model_name
            #     print(f"DEBUG: Saving project with segmentation cache")
            #     print(f"DEBUG: Current model: {current_model}")
            #
            #     if current_model and current_model in cache.mmap_arrays_idx:
            #         _, indices = cache.mmap_arrays_idx[current_model]
            #         print(f"DEBUG: Cache contains {len(indices)} segmented frames for model {current_model}")
            #         print(f"DEBUG: Sample indices: {list(indices)[:5] if indices else 'None'}")
            #     else:
            #         print(f"DEBUG: No cached frames for model {current_model}")

            # Save image data
            try:
                print(f"DEBUG: Saving image data to {folder_path}")
                ImageData.get_instance().save_to_disk(folder_path)
                print(f"DEBUG: Image data saved successfully")
            except Exception as e:
                print(f"ERROR: Failed to save image data: {str(e)}")
                import traceback

                traceback.print_exc()

            # Save metrics
            metrics_service = MetricsService()
            try:
                print(f"DEBUG: Saving metrics to {folder_path}")
                metrics_service.save_optimized(folder_path)
            except Exception as e:
                print(f"ERROR: Failed to save metrics: {str(e)}")
                import traceback

                traceback.print_exc()

            # Save the experiment with ROI and registration data
            try:
                if self.appstate.experiment is not None:
                    # Get ROI from appstate
                    roi_mask = self.appstate.roi_mask if hasattr(self.appstate, 'roi_mask') else None

                    # Get registration offsets from image_data
                    registration_offsets = None
                    if self.appstate.image_data and hasattr(self.appstate.image_data, 'registration_offsets'):
                        registration_offsets = self.appstate.image_data.registration_offsets

                    # Get crop coordinates from segmentation service
                    crop_coordinates = None
                    if hasattr(self, 'segmentation_service') and hasattr(self.segmentation_service, 'crop_coordinates'):
                        crop_coordinates = self.segmentation_service.crop_coordinates

                    self.appstate.experiment.save(
                        folder_path,
                        roi_mask=roi_mask,
                        registration_offsets=registration_offsets,
                        crop_coordinates=crop_coordinates
                    )
                    print(f"Saved experiment with ROI={roi_mask is not None}, Registration={registration_offsets is not None}")
            except Exception as e:
                print(f"ERROR: Failed to save experiment: {str(e)}")
                import traceback

                traceback.print_exc()

            # Save tracking data
            try:
                print(f"DEBUG: Saving tracking data to {folder_path}")
                tracking_saved = self.trackingTab.tracking_widget.save_tracking_data(folder_path)
                if tracking_saved:
                    print(f"DEBUG: Tracking data saved successfully")
                else:
                    print(f"DEBUG: No tracking data to save")
            except Exception as e:
                print(f"ERROR: Failed to save tracking data: {str(e)}")
                import traceback

                traceback.print_exc()

            # Show success message
            QMessageBox.information(
                self, "Save Complete", f"Project saved to {folder_path}"
            )

    def load_from_folder(self):
        """Load a project from a folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Project Folder", "", QFileDialog.ShowDirsOnly
        )

        if folder_path:
            print(f"Project loaded from folder: {folder_path}")

            try:
                # Load metrics data
                metrics_service = MetricsService()
                metrics_loaded = metrics_service.load_optimized(folder_path)

                # Load experiment with ROI and registration data
                # Try loading; if image files moved, prompt for their new location
                relink_root = None
                try:
                    experiment, roi_mask, registration_offsets, crop_coordinates = Experiment.load(folder_path)
                except FileNotFoundError:
                    relink_root = QFileDialog.getExistingDirectory(
                        self,
                        "Original image files not found — select folder containing the raw data",
                        "",
                        QFileDialog.ShowDirsOnly,
                    )
                    if not relink_root:
                        raise FileNotFoundError(
                            "Image file paths in this project are missing and no replacement folder was selected."
                        )
                    experiment, roi_mask, registration_offsets, crop_coordinates = Experiment.load(
                        folder_path, relink_root=relink_root
                    )
                self.appstate.experiment = experiment

                # Load image data (with TIFF relink fallback for moved datasets)
                image_data = None
                meta_path = os.path.join(folder_path, "image_data.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        meta_json = json.load(f)
                    tiff_file_map_raw = meta_json.get("tiff_file_map")
                    tiff_mode = meta_json.get("tiff_mode")

                    if tiff_file_map_raw and tiff_mode:
                        file_map = self._reconstruct_tiff_file_map(tiff_file_map_raw)
                        all_paths_exist = all(
                            path is not None and os.path.exists(path)
                            for path in file_map.values()
                        )

                        if not all_paths_exist:
                            new_root = QFileDialog.getExistingDirectory(
                                self,
                                "Select tiff folder",
                                "",
                                QFileDialog.ShowDirsOnly,
                            )
                            if not new_root:
                                raise FileNotFoundError(
                                    "TIFF paths in this project are missing and no replacement folder was selected."
                                )
                            file_map = self._remap_tiff_file_map_by_filename(file_map, new_root)

                        image_data = ImageData.load_tiff_directory(file_map, tiff_mode)

                if image_data is None:
                    try:
                        image_data = ImageData.load_from_disk(folder_path, relink_root=relink_root)
                    except FileNotFoundError:
                        if not relink_root:
                            relink_root = QFileDialog.getExistingDirectory(
                                self,
                                "Original image files not found — select folder containing the raw data",
                                "",
                                QFileDialog.ShowDirsOnly,
                            )
                        if not relink_root:
                            raise
                        image_data = ImageData.load_from_disk(folder_path, relink_root=relink_root)

                # Restore registration offsets if they exist
                if registration_offsets is not None:
                    image_data.registration_offsets = registration_offsets
                    print(f"Restored registration offsets with shape {registration_offsets.shape}")

                pub.sendMessage("image_data_loaded", image_data=image_data)

                # Restore ROI if it exists
                if roi_mask is not None:
                    self.appstate.roi_mask = roi_mask
                    print(f"Restored ROI mask with shape {roi_mask.shape}")
                    # Broadcast ROI to view area and segmentation service
                    pub.sendMessage("roi_selected", mask=roi_mask)

                # Restore crop coordinates if they exist
                if crop_coordinates is not None:
                    print(f"Restored crop coordinates: {crop_coordinates}")
                    pub.sendMessage("crop_selected", coords=crop_coordinates)

                # Load tracking data
                tracking_loaded = False
                try:
                    print(f"DEBUG: Loading tracking data from {folder_path}")
                    tracking_loaded = self.trackingTab.tracking_widget.load_tracking_data(folder_path)
                    if tracking_loaded:
                        print(f"DEBUG: Tracking data loaded successfully")
                    else:
                        print(f"DEBUG: No tracking data found")
                except Exception as e:
                    print(f"ERROR: Failed to load tracking data: {str(e)}")
                    import traceback
                    traceback.print_exc()

                # Show success message
                message = f"Project loaded from {folder_path}"
                if metrics_loaded:
                    message += "\nMetrics data loaded successfully"
                if tracking_loaded:
                    message += "\nTracking data loaded successfully"

                QMessageBox.information(self, "Project Loaded", message)

            except Exception as e:
                import traceback

                traceback.print_exc()
                QMessageBox.warning(self, "Error", f"Failed to load project: {str(e)}")

    def highlight_cell(self, cell_id):
        """Highlight a specific cell when clicked on PCA plot"""
        print(f"App: Highlighting cell {cell_id}")

        # Ensure cell ID is an integer
        cell_id = int(cell_id)

        # Check if we have the cell mapping
        if (
            not hasattr(self, "current_cell_mapping")
            or cell_id not in self.current_cell_mapping
        ):
            print(f"Cell {cell_id} not found in current mapping")
            return

        # Get current frame parameters from ViewAreaWidget
        t = self.viewArea.current_t
        p = self.viewArea.current_p
        c = self.viewArea.current_c

        try:
            # Get the segmentation
            # TODO: why the heck are we doing it this way and not asking SegmentationService?
            segmented_image = ImageData.get_instance().segmentation_cache[t, p, c]

            if segmented_image is None:
                print(f"No segmentation available for highlighting")
                return

            # Create single-cell mapping for the selected cell
            single_cell_mapping = {cell_id: self.current_cell_mapping[cell_id]}

            # Import the morphology function
            from nd2_analyzer.analysis.morphology.morphology import annotate_binary_mask

            # Create highlighted image with just the one cell
            highlighted_image = annotate_binary_mask(
                segmented_image, single_cell_mapping
            )

            # TODO: everything related to creating QPixmaps and so on should become their own utility
            # Display on the view area's image label
            height, width = highlighted_image.shape[:2]
            qimage = QImage(
                highlighted_image.data,
                width,
                height,
                highlighted_image.strides[0],
                QImage.Format_RGB888,
            )
            pixmap = QPixmap.fromImage(qimage).scaled(
                self.viewArea.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.viewArea.image_label.setPixmap(pixmap)

        except Exception as e:
            import traceback

            print(f"Error highlighting cell: {e}")
            traceback.print_exc()
