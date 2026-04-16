from pathlib import Path
import sys
import os
import pickle

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
import nd2
import numpy as np
import pandas as pd

from tqdm import tqdm

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from cellpose import utils
from skimage.measure import label, regionprops

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from nd2_analyzer.data.image_data import ImageData

from nd2_analyzer.analysis.morphology.morphology import (
    annotate_binary_mask,
    annotate_image,
    classify_morphology,
    extract_cell_morphologies,
    extract_cells_and_metrics,
)
from nd2_analyzer.analysis.segmentation.segmentation_models import SegmentationModels
from nd2_analyzer.analysis.tracking.tracking import track_cells, visualize_cell_regions, enhanced_motility_index, visualize_motility_map, visualize_motility_with_chamber_regions

from .roisel import PolygonROISelector
from .dialogs import AboutDialog, ExperimentDialog
from .widgets import ViewAreaWidget, PopulationWidget, SegmentationWidget, MorphologyWidget, TrackingManager

from pubsub import pub

from nd2_analyzer.data.experiment import Experiment
from nd2_analyzer.analysis.metrics_service import MetricsService
from nd2_analyzer.ui.biofilms.analysis_mode import AnalysisMode, AnalysisModeConfig
from nd2_analyzer.ui.dialogs.mode_selection_dialog import ModeSelectionDialog
from nd2_analyzer.ui.biofilms.config.biofilm_config import BiofilmConfig
from nd2_analyzer.ui.biofilms.cube_analysis import CubeAnalysisWidget


class MorphologyWorker(QObject):
    progress = Signal(int)  # Progress updates
    finished = Signal(object)  # Finished with results
    error = Signal(str)  # Emit error message

    def __init__(
            self,
            image_data,
            image_frames,
            num_frames,
            position,
            channel):
        super().__init__()
        self.image_data = image_data
        self.image_frames = image_frames
        self.num_frames = num_frames
        self.position = position
        self.channel = channel
        self.cell_mapping = {}

    def run(self):
        results = {}
        try:
            for t in range(self.num_frames):

                current_frame = self.image_frames[t]
                # Skip empty/invalid frames
                if np.mean(current_frame) == 0 or np.std(
                        current_frame) == 0:
                    print(f"Skipping empty frame T={t}")
                    self.progress.emit(t + 1)
                    continue

                t, p, c = (t, self.position, self.channel)

                binary_image = self.image_data.segmentation_cache[t, p, c]

                # Validate segmentation result
                if binary_image is None or binary_image.sum() == 0:
                    print(f"Frame {t}: No valid segmentation")
                    self.progress.emit(t + 1)
                    continue

                # Extract morphology metrics
                cell_mapping = extract_cells_and_metrics(
                    self.image_frames[t], binary_image)

                # Then convert the cell_mapping to a metrics dataframe
                metrics_list = [data["metrics"]
                                for data in cell_mapping.values()]
                metrics = pd.DataFrame(metrics_list)

                if not metrics.empty:
                    total_cells = len(metrics)

                    # Calculate Morphology Fractions
                    morphology_counts = metrics["morphology_class"].value_counts(
                        normalize=True)
                    fractions = morphology_counts.to_dict()

                    # Save results for this frame, including the raw metrics
                    results[t] = {
                        "fractions": fractions,
                        "metrics": metrics  # Include the full metrics dataframe
                    }
                else:
                    print(
                        f"Frame {t}: Metrics computation returned no valid data.")

                self.progress.emit(t + 1)  # Update progress bar

            if results:
                self.finished.emit(results)  # Emit processed results
            else:
                self.error.emit("No valid results found in any frame.")
        except Exception as e:
            self.error.emit(str(e))
            raise e


class App(QMainWindow):

    def __init__(self):
        super().__init__()

        print("DEBUG: Starting app initialization...")

        # Initialize analysis mode configuration FIRST
        self.analysis_config = AnalysisModeConfig()

        self.biofilm_config = BiofilmConfig()

        # Show mode selection dialog
        selected_mode = ModeSelectionDialog.show_mode_selection(
            self.analysis_config, self)
        self.current_analysis_mode = selected_mode

        # Set window properties
        mode_name = selected_mode.display_name
        self.setWindowTitle(f"Partaker 3 - {mode_name}")
        self.setGeometry(100, 100, 1000, 800)

        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create main layout
        self.layout = QHBoxLayout(self.central_widget)

        # Initialize services
        self.metrics_service = MetricsService()

        # Create and add the MorphologyWidget with metrics service
        self.morphology_widget = MorphologyWidget(
            metrics_service=self.metrics_service)

        # Create ViewArea widget
        self.viewArea = ViewAreaWidget()
        self.layout.addWidget(self.viewArea)

        # Initialize other UI components
        self.tab_widget = QTabWidget()

        # Complete initialization based on analysis mode
        self.init_ui()
        self.layout.addWidget(self.tab_widget)

        # Subscribe to events
        pub.subscribe(self.on_exp_loaded, "experiment_loaded")
        pub.subscribe(self.on_image_request, "image_request")
        pub.subscribe(self.on_segmentation_request, "segmentation_request")
        pub.subscribe(self.on_draw_cell_bounding_boxes,
                      "draw_cell_bounding_boxes")
        pub.subscribe(self.highlight_cell, "highlight_cell_requested")

        pub.subscribe(self.provide_image_data, "get_image_data")
        pub.subscribe(self.convert_nd2_to_tif, "convert_nd2_to_tif")

    def provide_image_data(self, callback):
        """Provide the image_data object through the callback"""
        if hasattr(self, "image_data"):
            print("Providing image_data to callback")
            callback(self.image_data)
        else:
            print("No image_data available")
            callback(None)

    def on_exp_loaded(self, experiment: Experiment):
        self.curr_experiment = experiment

        # Instance ImageData and load the first image file
        first_file = experiment.image_files[0]
        if first_file.lower().endswith('.nd2'):
            self.image_data = ImageData.load_nd2(first_file)
        else:
            # TIFF file
            tiff_type = experiment.tiff_types.get(first_file, 'single_sequence')
            tiff_files = [f for f in experiment.image_files if f.lower().endswith(('.tif', '.tiff'))]
            self.image_data = ImageData.load_tiff(tiff_files, tiff_type)
        pub.sendMessage("image_data_loaded", image_data=self.image_data)

    def load_from_folder(self):
        """Load a project from a folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Project Folder", "", QFileDialog.ShowDirsOnly
        )

        if folder_path:
            print(f"DEBUG: Loading project from folder: {folder_path}")

            try:
                # Check for segmentation cache files
                cache_path = os.path.join(folder_path, "segmentation_cache.h5")
                metadata_path = os.path.join(
                    folder_path, "segmentation_metadata.json")

                if os.path.exists(cache_path):
                    print(f"DEBUG: Found segmentation cache at {cache_path}")
                    cache_size = os.path.getsize(cache_path)
                    print(
                        f"DEBUG: Cache file size: {cache_size/1024/1024:.2f} MB")
                else:
                    print(
                        f"DEBUG: No segmentation cache found at {cache_path}")

                if os.path.exists(metadata_path):
                    print(
                        f"DEBUG: Found segmentation metadata at {metadata_path}")
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        print(f"DEBUG: Metadata: {metadata}")
                else:
                    print(
                        f"DEBUG: No segmentation metadata found at {metadata_path}")

                # Load image data
                print(f"DEBUG: Loading image data")
                self.image_data = ImageData.load(folder_path)
                print(f"DEBUG: Image data loaded successfully")

                # Check segmentation cache status after loading
                if hasattr(self.image_data, "segmentation_cache"):
                    cache = self.image_data.segmentation_cache
                    print("DEBUG: Inspecting segmentation cache after loading:")
                    print(f"DEBUG: Cache model name: {cache.model_name}")

                    # Check all models in cache
                    for model_name in cache.mmap_arrays_idx.keys():
                        array, indices = cache.mmap_arrays_idx[model_name]
                        print(
                            f"DEBUG: Model {model_name} has {len(indices)} entries")

                        # Show some sample indices to see their format
                        sample_indices = list(indices)[:5] if indices else []
                        print(f"DEBUG: Sample indices: {sample_indices}")
                        print(f"DEBUG: Array shape: {array.shape}")
                else:
                    print(f"DEBUG: No segmentation cache loaded")

                # Load metrics data
                print(f"DEBUG: Loading metrics data")
                metrics_service = MetricsService()
                metrics_loaded = metrics_service.load_from_file(folder_path)
                print(f"DEBUG: Metrics loaded: {metrics_loaded}")

                if metrics_loaded:
                    # Print some metrics statistics
                    query_result = metrics_service.query()
                    if not query_result.is_empty():
                        print(
                            f"DEBUG: Loaded {query_result.height} metric rows")
                        print(f"DEBUG: Metric columns: {query_result.columns}")

                        # Check a few specific frames
                        for t, p, c in [(0, 0, 0), (0, 1, 0), (1, 0, 0)]:
                            frame_metrics = metrics_service.query(
                                time=t, position=p, channel=c)
                            if not frame_metrics.is_empty():
                                print(
                                    f"DEBUG: Found {frame_metrics.height} metrics for T={t}, P={p}, C={c}")
                            else:
                                print(
                                    f"DEBUG: No metrics found for T={t}, P={p}, C={c}")
                    else:
                        print(f"DEBUG: No metric rows loaded")

                # Update UI based on loaded image data
                if hasattr(self, "viewArea"):
                    # Update the ViewAreaWidget model dropdown
                    if hasattr(self.image_data, "segmentation_cache") and self.image_data.segmentation_cache.model_name:
                        model_name = self.image_data.segmentation_cache.model_name
                        index = self.viewArea.model_dropdown.findText(
                            model_name)
                        if index >= 0:
                            print(
                                f"DEBUG: Setting viewArea model dropdown to {model_name}")
                            self.viewArea.model_dropdown.blockSignals(True)
                            self.viewArea.model_dropdown.setCurrentIndex(index)
                            self.viewArea.model_dropdown.blockSignals(False)
                            self.viewArea.current_model = model_name

                    print(f"DEBUG: Sending image_data_loaded message")
                    pub.sendMessage("image_data_loaded",
                                    image_data=self.image_data)

                # Load tracking data if available
                tracking_loaded = False
                if hasattr(self, "tracking_manager") and hasattr(self.tracking_manager, "tracking_widget"):
                    tracking_loaded = self.tracking_manager.tracking_widget.load_tracking_data(
                        folder_path)
                    print(f"DEBUG: Tracking data loaded: {tracking_loaded}")

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
                QMessageBox.warning(
                    self, "Error", f"Failed to load project: {str(e)}")

    def load_nd2_file(self, file_path):
        self.image_data = ImageData.load_nd2(file_path)
        self.init_controls_nd2(file_path)
        if hasattr(self.image_data, 'shape') and len(self.image_data.shape) > 2:
            self.has_channels = len(self.image_data.shape) > 3
        else:
            self.has_channels = False

    def init_controls_nd2(self, file_path):
        """ This function updates the controls with the ND2 dimensions.
        Must be called after initializing the ND2 file, either way.
        """
        with nd2.ND2File(file_path) as nd2_file:
            self.dimensions = nd2_file.sizes
            info_text = f"Number of dimensions: {nd2_file.sizes}\n"

            for dim, size in self.dimensions.items():
                info_text += f"{dim}: {size}\n"

            if "C" in self.dimensions.keys():
                self.has_channels = True
                self.channel_number = self.dimensions["C"]
                self.slider_c.setMinimum(0)
                self.slider_c.setMaximum(self.channel_number - 1)
            else:
                self.has_channels = False

            self.info_label.setText(info_text)

            # Set the slider range for position (P) immediately based on
            # dimensions
            max_position = self.dimensions.get("P", 1) - 1
            self.slider_p.setMaximum(max_position)
            self.update_controls()
            self.display_image()

            self.image_data.segmentation_cache.with_model(
                self.model_dropdown.currentText())
            print("Segmentation cache cleared.")

    def display_file_info(self, file_path):
        info_text = f"Number of dimensions: {len(self.dimensions)}\n"
        for dim, size in self.dimensions.items():
            info_text += f"{dim}: {size}\n"
        self.info_label.setText(info_text)

    def update_controls(self):
        """Updates all the applications controls based on the dimensions of the lodaded ImageData"""
        t_max = self.dimensions.get("T", 1) - 1
        p_max = self.dimensions.get("P", 1) - 1

        # Initialize sliders with full ranges
        self.slider_t.setMaximum(t_max)
        self.slider_p.setMaximum(p_max)

        self.slider_t.setMaximum(self.dimensions.get("T", 1) - 1)

        max_position = self.dimensions.get("P", 1) - 1
        self.slider_p.setMaximum(max_position)

        # Population tab
        self.time_max_box.setMaximum(self.dimensions.get("T", 1) - 1)

        max_p = (
            self.dimensions.get("P", 1) - 1
            if hasattr(self, "dimensions") and "P" in self.dimensions
            else 0
        )
        # Populate with all possible P values
        for p in range(max_p + 1):
            self.p_dropdown.addItem(str(p))

    def show_cell_area(self, img):

        # Check if the image type is CV_32FC1 and convert to CV_8UC1
        if img.dtype == np.float32 or img.dtype == np.int32 or img.dtype == np.int64:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)

        # Binarize the image using Otsu's thresholding
        _, bw_image = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Calculate connected components with stats
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bw_image, connectivity=8)

        # Extract pixel counts for each component (ignore background)
        # Skip the first label (background)
        pixel_counts = stats[1:, cv2.CC_STAT_AREA]

    def overlay_labels_on_segmentation(self, segmented_images):
        """
        Overlay cell IDs on segmented images.

        Parameters:
        segmented_images (list of np.ndarray): List of segmented images (labeled masks).

        Returns:
        list of np.ndarray: Images with overlaid cell IDs.
        """
        labeled_images = []

        for mask in segmented_images:
            # Convert to RGB for color text overlay
            labeled_image = cv2.cvtColor(
                mask.astype(
                    np.uint8) * 255,
                cv2.COLOR_GRAY2BGR)

            # Get unique cell IDs (ignore background 0)
            unique_ids = np.unique(mask)
            unique_ids = unique_ids[unique_ids != 0]

            for cell_id in unique_ids:
                # Find coordinates of the current cell
                coords = np.column_stack(np.where(mask == cell_id))

                # Calculate centroid of the cell
                centroid_y, centroid_x = coords.mean(axis=0).astype(int)

                # Overlay cell ID text at the centroid
                cv2.putText(
                    labeled_image,
                    str(cell_id),
                    (centroid_x, centroid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,  # Font size
                    (255, 255, 255),  # White color
                    1,  # Thickness
                    cv2.LINE_AA
                )

            labeled_images.append(labeled_image)

        return labeled_images

    def display_image(self):
        t = self.slider_t.value()
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        # Retrieve the current frame
        image_data = self.image_data.data
        if self.image_data.is_nd2:
            image_data = image_data[t, p,
                                    c] if self.has_channels else image_data[t, p]
        else:
            image_data = image_data[t]

        image_data = np.array(image_data)  # Ensure it's a NumPy array
        self.current_image = image_data

        # Apply thresholding or segmentation if selected
        if self.radio_labeled_segmentation.isChecked():
            # Ensure segmentation model is set correctly in segmentation_cache
            selected_model = self.model_dropdown.currentText()
            self.image_data.segmentation_cache.with_model(selected_model)

            # Retrieve segmentation from segmentation_cache
            segmented = self.image_data.segmentation_cache[t, p, c]

            if segmented is None:
                print(f"[ERROR] Segmentation failed for T={t}, P={p}, C={c}")
                QMessageBox.warning(
                    self, "Segmentation Error", "Segmentation failed.")
                return

            # Relabel the segmented regions
            labeled_cells = label(segmented)

            # Ensure relabeling created valid labels
            max_label = labeled_cells.max()
            if max_label == 0:
                print("[ERROR] No valid labeled regions found.")
                QMessageBox.warning(
                    self,
                    "Labeled Segmentation Error",
                    "No valid labeled regions found.")
                return

            # Convert labels to color image
            labeled_image = plt.cm.nipy_spectral(
                labeled_cells.astype(float) / max_label)
            labeled_image = (labeled_image[:, :, :3] * 255).astype(np.uint8)

            # Overlay Cell IDs
            props = regionprops(labeled_cells)
            for prop in props:
                y, x = prop.centroid  # Get centroid coordinates
                cell_id = prop.label  # Get cell ID
                cv2.putText(labeled_image, str(cell_id), (int(x), int(
                    y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # Display the labeled image
            height, width, _ = labeled_image.shape
            qimage = QImage(
                labeled_image.data,
                width,
                height,
                3 * width,
                QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)

            return

        elif self.radio_segmented.isChecked():
            cache_key = (t, p, c)

            self.image_data.segmentation_cache.with_model(
                self.model_dropdown.currentText())  # Setting the model we want
            image_data = self.image_data.segmentation_cache[t, p, c]

            # TODO: check if this is really needed
            metrics = extract_cell_morphologies(image_data)

        else:  # Normal view or overlay
            if self.radio_overlay_outlines.isChecked():
                # Ensure segmentation model is set correctly in
                # segmentation_cache
                selected_model = self.model_dropdown.currentText()
                self.image_data.segmentation_cache.with_model(selected_model)

                # Retrieve segmentation from segmentation_cache
                segmented_image = self.image_data.segmentation_cache[t, p, c]

                if segmented_image is None:
                    print(
                        f"[ERROR] Segmentation failed for T={t}, P={p}, C={c}")
                    QMessageBox.warning(
                        self, "Segmentation Error", "Segmentation failed.")
                    return

                print(
                    f"[SUCCESS] Retrieved segmentation for overlay - T={t}, P={p}, C={c}")

                # Extract outlines
                outlines = utils.masks_to_outlines(segmented_image)
                overlay = image_data.copy()

                # Verify dimensions match before applying overlay
                if outlines.shape == overlay.shape:
                    overlay[outlines] = overlay.max()
                else:
                    print(
                        f"Dimension mismatch - Outline shape: {outlines.shape}, Image shape: {overlay.shape}")
                    outlines = cv2.resize(
                        outlines.astype(np.uint8),
                        (overlay.shape[1], overlay.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
                    overlay[outlines] = overlay.max()

                image_data = overlay

            # Normalize and apply color for normal/overlay views
            image_data = cv2.normalize(
                image_data,
                None,
                0,
                255,
                cv2.NORM_MINMAX).astype(np.uint8)

            # Apply color based on channel for non-binary images
            if self.has_channels:
                colored_image = np.zeros(
                    (image_data.shape[0], image_data.shape[1], 3), dtype=np.uint8)
                if c == 0:  # Phase contrast - grayscale
                    colored_image = cv2.cvtColor(
                        image_data, cv2.COLOR_GRAY2BGR)
                elif c == 1:  # mCherry - red
                    colored_image[:, :, 0] = image_data  # Red channel
                elif c == 2:  # YFP - yellow/green
                    colored_image[:, :, 1] = image_data  # Green channel
                    # Add red to make it yellow
                    colored_image[:, :, 0] = image_data
                image_data = colored_image
                image_format = QImage.Format_RGB888
            else:
                image_format = QImage.Format_Grayscale8

        # Normalize the image safely for grayscale images only
        if len(image_data.shape) == 2:  # Grayscale
            if image_data.max() > 0:
                image_data = (
                    image_data.astype(
                        np.float32) /
                    image_data.max() *
                    65535).astype(
                    np.uint16)
            else:
                image_data = np.zeros_like(image_data, dtype=np.uint16)

        # Determine format based on image type
        if len(image_data.shape) == 3 and image_data.shape[2] == 3:
            image_format = QImage.Format_RGB888
            height, width, _ = image_data.shape
        else:
            image_format = QImage.Format_Grayscale16
            height, width = image_data.shape[:2]

        # Display image
        image = QImage(
            image_data.data,
            width,
            height,
            image_data.strides[0],
            image_format)
        pixmap = QPixmap.fromImage(image).scaled(
            self.image_label.size(),
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

        # Store this processed image for export
        self.processed_images.append(image_data)
        pub.sendMessage("current_view_changed",
                        time=t,
                        position=p,
                        channel=c)

    def on_image_request(self, time, position, channel):
        """Handle image request from other components"""
        if not self.image_data:
            return

        try:
            # Your data shape is (49, 9, 2044, 2048) = (T, P, Y, X)
            data_array = self.image_data.data
            
            # Get the 2D image at [time, position]
            image = data_array[time, position]  # This should give (2044, 2048)
            
            # Convert to NumPy array
            image = np.array(image)
            print(f"DEBUG: Final image shape: {image.shape}")

            # Send the image
            pub.sendMessage("image_ready",
                            image=image,
                            time=time,
                            position=position,
                            channel=channel,
                            mode="normal")
                            
            print(f"DEBUG: Successfully sent image T={time}, P={position}, C={channel}")
            
        except Exception as e:
            print(f"Error retrieving image: {e}")
    
    def on_segmentation_request(self, time, position, channel, model_name):
        """Handle requests for segmentation data"""
        if not self.image_data:
            return

        try:
            # Get the appropriate segmentation model
            if model_name:
                self.image_data.segmentation_cache.with_model(model_name)
            else:
                self.image_data.segmentation_cache.with_model(
                    self.model_dropdown.currentText())

            # Get the segmentation
            segmented_image = self.image_data.segmentation_cache[time,
                                                                 position, channel]

            # Publish the segmentation result
            pub.sendMessage("segmentation_ready",
                            segmented_image=segmented_image,
                            time=time,
                            position=position,
                            channel=channel)
        except Exception as e:
            print(f"Error retrieving segmentation: {e}")

    def open_roi_selector(self):
        # Get the image to use for ROI selection
        image_data = self.current_image

        # Create and show the ROI selector dialog
        roi_dialog = PolygonROISelector(image_data)
        roi_dialog.roi_selected.connect(self.handle_roi_result)
        roi_dialog.exec_()  # Use exec_ to make it modal

    def handle_roi_result(self, mask):
        # Store and apply mask
        self.roi_mask = mask
        self.image_data.segmentation_cache.set_binary_mask(self.roi_mask)
        # Update UI or perform other actions with the new mask
        print(f"ROI mask created with shape: {self.roi_mask.shape}")

    def initMorphologyTimeTab(self):

        # Create a horizontal layout for the tracking buttons
        tracking_buttons_layout = QHBoxLayout()

        # Button for lineage tracking
        self.lineage_button = QPushButton("Visualize Lineage Tree")
        self.lineage_button.clicked.connect(self.track_cells_with_lineage)
        tracking_buttons_layout.addWidget(self.lineage_button)

        # Button to create tracking video
        self.overlay_video_button = QPushButton("Tracking Video")
        self.overlay_video_button.clicked.connect(
            self.overlay_tracks_on_images)
        tracking_buttons_layout.addWidget(self.overlay_video_button)

        self.motility_button = QPushButton("Analyze Cell Motility")
        self.motility_button.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold;")
        self.motility_button.clicked.connect(self.analyze_cell_motility)
        tracking_buttons_layout.addWidget(self.motility_button)

        # Inner Tabs for Plots
        self.plot_tabs = QTabWidget()
        self.morphology_fractions_tab = QWidget()

        # Plot for Morphology Fractions
        morphology_fractions_layout = QVBoxLayout(
            self.morphology_fractions_tab)
        self.figure_morphology_fractions = plt.figure()
        self.canvas_morphology_fractions = FigureCanvas(
            self.figure_morphology_fractions)
        morphology_fractions_layout.addWidget(self.canvas_morphology_fractions)

        # Progress Bar
        self.progress_bar = QProgressBar()

    def on_draw_cell_bounding_boxes(self, time, position, channel, cell_mapping):
        """Handle request to draw cell bounding boxes"""
        # Get the segmentation using the same model as the segmentation cache
        if not hasattr(self, "image_data") or not self.image_data:
            print("No image data available")
            return

        # Get the current model from the cache
        current_model = self.image_data.segmentation_cache.model_name
        if not current_model:
            # Default to a standard model if none set
            current_model = "bact_phase_cp3"  # This is CELLPOSE_BACT_PHASE
            self.image_data.segmentation_cache.with_model(current_model)

        # Get the segmentation data
        segmented_image = self.image_data.segmentation_cache[time,
                                                             position, channel]

        if segmented_image is None:
            print(
                f"No segmentation available for T={time}, P={position}, C={channel}")
            return

        # Create annotated image
        annotated_image = annotate_binary_mask(segmented_image, cell_mapping)

        # Display on the view area's image label
        height, width = annotated_image.shape[:2]
        qimage = QImage(annotated_image.data, width, height,
                        annotated_image.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.viewArea.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.viewArea.image_label.setPixmap(pixmap)

        # Store the annotated image
        if hasattr(self, "annotated_image"):
            self.annotated_image = annotated_image

        self.current_cell_mapping = cell_mapping

    def visualize_tracking(self, tracks):
        """
        Visualizes the tracked cells as trajectories over time.

        Parameters:
        -----------
        tracks : list
            List of tracked cell dictionaries with x, y, and ID information.
        """
        if not tracks:
            print("No valid tracking data to visualize.")
            return

        self.figure_morphology_fractions.clear()
        ax = self.figure_morphology_fractions.add_subplot(111)

        # Create a colormap for the tracks
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

        # Use a colormap that's easier to distinguish
        cmap = cm.get_cmap('tab20', min(20, len(tracks)))

        # Calculate displacement for each track
        track_displacements = []
        for track in tracks:
            if len(track['x']) >= 2:
                # Calculate total displacement (Euclidean distance from start
                # to end)
                start_x, start_y = track['x'][0], track['y'][0]
                end_x, end_y = track['x'][-1], track['y'][-1]
                displacement = np.sqrt(
                    (end_x - start_x)**2 + (end_y - start_y)**2)
                track_displacements.append(displacement)
            else:
                track_displacements.append(0)

        # Sort tracks by displacement (most movement first)
        sorted_indices = np.argsort(track_displacements)[::-1]
        sorted_tracks = [tracks[i] for i in sorted_indices]

        # Plot each track with its own color
        for i, track in enumerate(sorted_tracks):
            # Get track data
            x_coords = track['x']
            y_coords = track['y']
            track_id = track['ID']

            # Get color from colormap
            color = cmap(i % 20)

            # Plot trajectory
            ax.plot(x_coords, y_coords, marker='.', markersize=3,
                    linewidth=1, color=color, label=f'Track {track_id}')

            # Mark start and end points
            ax.plot(x_coords[0], y_coords[0], 'o',
                    color=color, markersize=6)  # Start
            ax.plot(x_coords[-1], y_coords[-1], 's',
                    color=color, markersize=6)  # End

        # Add labels and title
        ax.set_title('Cell Trajectories Over Time')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')

        # Add legend with a reasonable number of entries
        if len(tracks) > 10:
            # For lots of tracks, show just a subset in the legend
            ax.legend(ncol=2, fontsize='small', loc='upper right')
        else:
            ax.legend(loc='best')

        # Add statistics to the plot
        stats_text = f"Displaying top {len(tracks)} tracks\n"

        # Calculate some statistics
        avg_displacement = np.mean(track_displacements[:len(tracks)])
        max_displacement = np.max(track_displacements[:len(tracks)])

        stats_text += f"Avg displacement: {avg_displacement:.1f}px\n"
        stats_text += f"Max displacement: {max_displacement:.1f}px"

        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, bbox=dict(
            facecolor='white', alpha=0.7), verticalalignment='bottom')

        # Draw the plot
        self.canvas_morphology_fractions.draw()

    def overlay_tracks_on_images(self):
        """
        Overlays tracking trajectories on the segmented images and creates a video.
        """
        if not hasattr(self, "tracked_cells") or not self.tracked_cells:
            QMessageBox.warning(
                self,
                "Error",
                "No tracking data available. Run cell tracking first.")
            return

        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        # Get segmented images
        t = self.dimensions["T"]  # Get total number of frames
        segmented_images = []

        # Show progress dialog
        progress = QProgressDialog(
            "Processing frames for video...", "Cancel", 0, t, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        # Get segmented frames
        for i in range(t):
            progress.setValue(i)
            if progress.wasCanceled():
                return

            segmented = self.image_data.segmentation_cache[i, p, c]
            if segmented is not None:
                segmented_images.append(segmented)

        if not segmented_images:
            QMessageBox.warning(self, "Error", "No segmented images found.")
            return

        segmented_images = np.array(segmented_images)

        # Ask user for save location
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Tracked Video", "", "MP4 Files (*.mp4)")
        if not output_path:
            return

        # Update progress dialog for video creation
        progress.setLabelText("Creating tracking video...")
        progress.setValue(0)
        progress.setMaximum(100)

        # Create a progress callback for the overlay function
        def update_progress(value):
            progress.setValue(value)

        # Import the overlay function
        from nd2_analyzer.analysis.tracking.tracking import overlay_tracks_on_images as create_tracking_video

        # Create the video
        try:
            create_tracking_video(
                segmented_images,
                self.tracked_cells,
                save_video=True,
                output_path=output_path,
                show_frames=False,  # Don't show frames in matplotlib
                max_tracks=100,      # Limit to 30 tracks
                progress_callback=update_progress
            )

            QMessageBox.information(
                self,
                "Video Created",
                f"Tracking visualization saved to {output_path}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(
                self, "Error", f"Failed to create tracking video: {str(e)}")

    def track_cells_with_lineage(self):
        # Check if lineage data is already loaded
        if hasattr(self, "lineage_tracks") and self.lineage_tracks is not None:
            # Skip tracking and go straight to visualization
            print("Using previously loaded tracking data")

            reply = QMessageBox.question(
                self, "Lineage Analysis",
                "Would you like to visualize the cell lineage tree?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.show_lineage_dialog()
            return

        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None
        if not self.image_data.is_nd2:
            QMessageBox.warning(
                self, "Error", "Tracking requires an ND2 dataset.")
            return
        try:
            t = self.dimensions["T"]
            progress = QProgressDialog(
                "Preparing frames for tracking...", "Cancel", 0, t, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            self.image_data.segmentation_cache.with_model(
                self.model_dropdown.currentText())
            labeled_frames = []
            for i in range(t):
                if progress.wasCanceled():
                    return
                progress.setValue(i)
                segmented = self.image_data.segmentation_cache[i, p, c]
                if segmented is not None:
                    labeled = label(segmented)
                    labeled_frames.append(labeled)
            progress.setValue(t)
            if not labeled_frames:
                QMessageBox.warning(
                    self, "Error", "No segmented frames found for tracking.")
                return
            labeled_frames = np.array(labeled_frames)
            print(f"Prepared {len(labeled_frames)} frames for tracking")
            progress.setLabelText(
                "Running cell tracking with lineage detection...")
            progress.setValue(0)
            progress.setMaximum(100)
            all_tracks, _ = track_cells(labeled_frames)
            visualize_cell_regions(all_tracks)
            self.lineage_tracks = all_tracks  # Store all tracks
            MIN_TRACK_LENGTH = 5
            filtered_tracks = [track for track in all_tracks if len(
                track['x']) >= MIN_TRACK_LENGTH]
            filtered_tracks.sort(
                key=lambda track: len(track['x']), reverse=True)
            MAX_TRACKS_TO_DISPLAY = 100
            display_tracks = filtered_tracks[:MAX_TRACKS_TO_DISPLAY]
            total_tracks = len(all_tracks)
            long_tracks = len(filtered_tracks)
            displayed_tracks = len(display_tracks)
            self.tracked_cells = display_tracks
            QMessageBox.information(
                self, "Tracking Complete",
                f"Cell tracking completed successfully.\n\n"
                f"Total tracks detected: {total_tracks}\n"
                f"Tracks spanning {MIN_TRACK_LENGTH}+ frames: {long_tracks}\n"
                f"Tracks displayed: {displayed_tracks}"
            )
            self.visualize_tracking(self.tracked_cells)
            reply = QMessageBox.question(
                self, "Lineage Analysis",
                "Would you like to visualize the cell lineage tree?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.show_lineage_dialog()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(
                self,
                "Tracking Error",
                f"Failed to track cells with lineage: {str(e)}")

    def show_timepoint_lineage_comparison(self):
        """
        Display both time zero and time last lineage trees side by side for comparison,
        with a separate tab for growth and division analysis.
        """
        if not hasattr(self, "lineage_tracks") or not self.lineage_tracks:
            QMessageBox.warning(
                self,
                "Error",
                "No tracking data available. Run tracking with lineage first.")
            return

        # Create a dialog for the comparison
        dialog = QDialog(self)
        dialog.setWindowTitle("Lineage Tree Time Comparison")
        dialog.setMinimumWidth(1200)
        dialog.setMinimumHeight(800)
        layout = QVBoxLayout(dialog)

        # Create tab widget for the two different analyses
        tab_widget = QTabWidget()

        # First tab - Time comparison visualization
        comparison_tab = QWidget()
        comparison_layout = QVBoxLayout(comparison_tab)

        # Cell selection options for time comparison
        selection_layout = QHBoxLayout()
        option_group = QButtonGroup(dialog)

        top_radio = QRadioButton("Top Largest Lineage Tree")
        top_radio.setChecked(True)
        option_group.addButton(top_radio)
        selection_layout.addWidget(top_radio)

        cell_radio = QRadioButton("Specific Cell Lineage:")
        option_group.addButton(cell_radio)
        selection_layout.addWidget(cell_radio)

        cell_combo = QComboBox()
        cell_combo.setEnabled(False)
        selection_layout.addWidget(cell_combo)

        # Find dividing cells for the combo box
        dividing_cells = [track['ID']
                          for track in self.lineage_tracks if track.get('children', [])]
        dividing_cells.sort()
        for cell_id in dividing_cells:
            cell_combo.addItem(f"Cell {cell_id}")

        # Enable/disable combo box based on radio selection
        def update_combo_state():
            cell_combo.setEnabled(cell_radio.isChecked())

        top_radio.toggled.connect(update_combo_state)
        cell_radio.toggled.connect(update_combo_state)

        comparison_layout.addLayout(selection_layout)

        # Create a horizontal layout for the two trees
        trees_layout = QHBoxLayout()

        # Time zero tree
        time_zero_widget = QWidget()
        time_zero_layout = QVBoxLayout(time_zero_widget)
        time_zero_label = QLabel("Time Zero (First Appearance)")
        time_zero_label.setAlignment(Qt.AlignCenter)
        time_zero_layout.addWidget(time_zero_label)

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        time_zero_figure = Figure(figsize=(5, 8), tight_layout=True)
        time_zero_canvas = FigureCanvas(time_zero_figure)
        time_zero_layout.addWidget(time_zero_canvas)

        # Time last tree
        time_last_widget = QWidget()
        time_last_layout = QVBoxLayout(time_last_widget)
        time_last_label = QLabel("Time Last (Before Division)")
        time_last_label.setAlignment(Qt.AlignCenter)
        time_last_layout.addWidget(time_last_label)

        time_last_figure = Figure(figsize=(5, 8), tight_layout=True)
        time_last_canvas = FigureCanvas(time_last_figure)
        time_last_layout.addWidget(time_last_canvas)

        # Add widgets to the trees layout
        trees_layout.addWidget(time_zero_widget)
        trees_layout.addWidget(time_last_widget)

        # Add trees layout to comparison tab layout
        comparison_layout.addLayout(trees_layout)

        # Create navigation area for previous/next tree
        nav_layout = QHBoxLayout()
        prev_button = QPushButton("Previous Tree")
        tree_counter_label = QLabel("Tree 1/1")
        tree_counter_label.setAlignment(Qt.AlignCenter)
        next_button = QPushButton("Next Tree")

        nav_layout.addWidget(prev_button)
        nav_layout.addWidget(tree_counter_label)
        nav_layout.addWidget(next_button)
        comparison_layout.addLayout(nav_layout)

        # Add the comparison tab to the tab widget
        tab_widget.addTab(comparison_tab, "Time Comparison")

        # Second tab - Growth & Division analysis
        growth_tab = QWidget()
        growth_layout = QVBoxLayout(growth_tab)

        try:
            # Calculate growth metrics if not already available
            if not hasattr(self, "growth_metrics"):
                progress = QProgressDialog(
                    "Calculating growth metrics...", "Cancel", 0, 100, dialog)
                progress.setWindowModality(Qt.WindowModal)
                progress.setValue(10)
                progress.show()
                QApplication.processEvents()

                self.growth_metrics = self.lineage_visualizer.calculate_growth_and_division_metrics(
                    self.lineage_tracks)

                progress.setValue(50)
                progress.setLabelText("Creating visualization...")
                QApplication.processEvents()

            # Create the growth figure with more space
            from matplotlib.figure import Figure
            # Increase figure size to fit the available space
            figure = Figure(figsize=(12, 10), dpi=100)

            # Set up the subplots with more space between them
            gs = figure.add_gridspec(2, 2, hspace=0.4, wspace=0.4)

            # Add the subplots
            ax1 = figure.add_subplot(gs[0, 0])  # Division Time Distribution
            ax2 = figure.add_subplot(gs[0, 1])  # Growth Rate Distribution
            # Division Time: Parent vs. Child
            ax3 = figure.add_subplot(gs[1, 0])
            ax4 = figure.add_subplot(gs[1, 1])  # Summary statistics

            # 1. Histogram of division times
            ax1.hist(self.growth_metrics['division_times'],
                     bins=20, color='skyblue', edgecolor='black')
            ax1.set_title('Division Time Distribution',
                          fontsize=14, fontweight='bold')
            ax1.set_xlabel('Time (frames)', fontsize=12)
            ax1.set_ylabel('Cell Count', fontsize=12)
            ax1.axvline(self.growth_metrics['avg_division_time'], color='red',
                        linestyle='--', label=f"Mean: {self.growth_metrics['avg_division_time']:.1f}")
            ax1.axvline(self.growth_metrics['median_division_time'], color='green',
                        linestyle='--', label=f"Median: {self.growth_metrics['median_division_time']:.1f}")
            ax1.legend(fontsize=11)

            # 2. Histogram of growth rates
            ax2.hist(self.growth_metrics['growth_rates'],
                     bins=20, color='lightgreen', edgecolor='black')
            ax2.set_title('Growth Rate Distribution',
                          fontsize=14, fontweight='bold')
            ax2.set_xlabel('Growth Rate (ln(2)/division time)', fontsize=12)
            ax2.set_ylabel('Cell Count', fontsize=12)
            ax2.axvline(self.growth_metrics['avg_growth_rate'], color='red',
                        linestyle='--', label=f"Mean: {self.growth_metrics['avg_growth_rate']:.4f}")
            ax2.legend(fontsize=11)

            # 3. Parent vs child division times
            # Calculate parent-child division time pairs
            division_time_by_id = {}
            for track in self.lineage_tracks:
                if 'children' in track and track['children'] and 't' in track and len(track['t']) > 0:
                    dt = track['t'][-1] - track['t'][0]
                    if dt > 0:
                        division_time_by_id[track['ID']] = dt

            parent_child_division_pairs = []
            for track in self.lineage_tracks:
                if 'children' in track and track['children'] and track['ID'] in division_time_by_id:
                    parent_dt = division_time_by_id[track['ID']]

                    for child_id in track['children']:
                        if child_id in division_time_by_id:
                            child_dt = division_time_by_id[child_id]
                            parent_child_division_pairs.append(
                                (parent_dt, child_dt))

            if parent_child_division_pairs:
                parent_dts, child_dts = zip(*parent_child_division_pairs)
                ax3.scatter(parent_dts, child_dts, alpha=0.7, color='blue')
                ax3.set_title('Division Time: Parent vs. Child',
                              fontsize=14, fontweight='bold')
                ax3.set_xlabel('Parent Division Time', fontsize=12)
                ax3.set_ylabel('Child Division Time', fontsize=12)

                # Add y=x reference line
                min_val = min(min(parent_dts), min(child_dts))
                max_val = max(max(parent_dts), max(child_dts))
                ax3.plot([min_val, max_val], [
                         min_val, max_val], 'k--', alpha=0.5)

                # Calculate correlation
                correlation = np.corrcoef(parent_dts, child_dts)[0, 1]
                ax3.text(0.05, 0.95, f"Correlation: {correlation:.2f}",
                         transform=ax3.transAxes,
                         verticalalignment='top',
                         fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.8))

            ax4.axis('off')

            # Format the summary text with clear spacing
            summary = (
                f"Growth & Division Summary\n\n"
                f"Total dividing cells: {self.growth_metrics['total_dividing_cells']}\n\n"
                f"Division Time:\n"
                f"  Mean:    {self.growth_metrics['avg_division_time']:.1f} frames\n"
                f"  Std Dev: {self.growth_metrics['std_division_time']:.1f} frames\n"
                f"  Median:  {self.growth_metrics['median_division_time']:.1f} frames\n\n"
                f"Growth Rate:\n"
                f"  Mean:    {self.growth_metrics['avg_growth_rate']:.4f}\n"
                f"  Std Dev: {self.growth_metrics['std_growth_rate']:.4f}\n"
            )

            if parent_child_division_pairs:
                summary += f"\nParent-Child Division Time\nCorrelation: {correlation:.2f}"

            # Create a visible background for the summary text
            summary_text = ax4.text(0.05, 0.95, summary,
                                    transform=ax4.transAxes,
                                    verticalalignment='top',
                                    horizontalalignment='left',
                                    fontfamily='monospace',
                                    fontsize=12,
                                    bbox=dict(facecolor='white', alpha=0.9,
                                              boxstyle='round,pad=1.0',
                                              edgecolor='gray'))

            # Create the canvas and add to layout
            canvas = FigureCanvas(figure)
            growth_layout.addWidget(canvas)

            # Adjust plot spacing
            figure.subplots_adjust(
                hspace=0.35, wspace=0.35, bottom=0.1, top=0.95, left=0.1, right=0.95)

            # Store the figure for saving later
            growth_fig = figure

            progress.setValue(100)
            progress.close()

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_label = QLabel(
                f"Error creating growth visualization: {str(e)}")
            error_label.setStyleSheet("color: red")
            growth_layout.addWidget(error_label)

        # Add the growth tab to the tab widget
        tab_widget.addTab(growth_tab, "Growth & Division")

        # Add tab widget to main layout
        layout.addWidget(tab_widget)

        # Add control buttons at the bottom
        button_layout = QHBoxLayout()
        view_button = QPushButton("Generate Visualization")
        save_button = QPushButton("Save Images")
        close_button = QPushButton("Close")

        button_layout.addWidget(view_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(close_button)

        # Add export button to the buttons layout
        export_data_button = QPushButton("Export Classification Data")
        export_data_button.clicked.connect(
            lambda: self.lineage_visualizer.export_morphology_classifications(dialog))
        button_layout.addWidget(export_data_button)

        layout.addLayout(button_layout)

        # Store state for tree navigation
        current_tree_index = [0]
        available_trees = []

        # Function to generate the visualizations
        def generate_visualizations():
            # Get selected cell ID if applicable
            selected_cell = None
            if cell_radio.isChecked() and cell_combo.currentText():
                selected_cell = int(
                    cell_combo.currentText().replace("Cell ", ""))
                current_tree_index[0] = 0  # Reset index for specific cell

            # Show progress while generating
            progress = QProgressDialog(
                "Generating visualizations...", "Cancel", 0, 100, dialog)
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()

            try:
                # First, precompute all morphology classifications
                progress.setLabelText(
                    "Precomputing morphology classifications...")
                progress.setValue(10)
                QApplication.processEvents()

                if hasattr(self, "cell_mapping") and self.cell_mapping:
                    self.lineage_visualizer.cell_mapping = self.cell_mapping
                    print(
                        f"Transferring cell mapping with {len(self.cell_mapping)} entries to lineage visualizer")

                # Precompute morphology for consistent classification
                self.lineage_visualizer.precompute_morphology_classifications(
                    self.lineage_tracks)

                progress.setValue(100)
                QApplication.processEvents()

                # If using top trees mode, find all connected components
                if top_radio.isChecked():
                    # Create a graph to find connected components
                    import networkx as nx
                    G = nx.DiGraph()

                    # Add nodes and edges
                    for track in self.lineage_tracks:
                        track_id = track['ID']
                        G.add_node(track_id)
                        if 'children' in track and track['children']:
                            for child_id in track['children']:
                                G.add_edge(track_id, child_id)

                    # Find all connected components
                    components = list(nx.weakly_connected_components(G))
                    # Sort by size (largest first)
                    available_trees.clear()
                    available_trees.extend(sorted(components, key=len, reverse=True)[
                                           :5])  # Top 5 largest

                    # Make sure current index is valid
                    if not available_trees:
                        QMessageBox.warning(
                            dialog, "Error", "No valid lineage trees found")
                        progress.close()
                        return

                    if current_tree_index[0] >= len(available_trees):
                        current_tree_index[0] = 0

                    # Get root of current tree for visualization
                    tree_nodes = list(available_trees[current_tree_index[0]])

                    # Find the root node of this tree (node with no parent in this tree)
                    root_candidates = []
                    for node in tree_nodes:
                        is_root = True
                        for track in self.lineage_tracks:
                            if 'children' in track and node in track['children']:
                                # Only consider parents in same tree
                                if track['ID'] in tree_nodes:
                                    is_root = False
                                    break
                        if is_root:
                            root_candidates.append(node)

                    # Use first root found or smallest ID if no clear root
                    root_cell_id = root_candidates[0] if root_candidates else min(
                        tree_nodes)

                    # Update tree counter
                    tree_counter_label.setText(
                        f"Tree {current_tree_index[0]+1}/{len(available_trees)}")

                    # Enable navigation buttons if we have multiple trees
                    prev_button.setEnabled(len(available_trees) > 1)
                    next_button.setEnabled(len(available_trees) > 1)
                else:
                    # Specific cell selected - use that as root
                    root_cell_id = selected_cell
                    # Disable navigation for specific cell mode
                    prev_button.setEnabled(False)
                    next_button.setEnabled(False)
                    tree_counter_label.setText("Custom Tree")

                # Generate time zero tree with cartoony style
                progress.setLabelText("Generating Time Zero visualization...")
                progress.setValue(50)
                QApplication.processEvents()

                self.lineage_visualizer.create_cartoony_lineage_comparison(
                    self.lineage_tracks, time_zero_canvas,
                    root_cell_id=root_cell_id, time_point="first")

                # Generate time last tree with cartoony style
                progress.setLabelText("Generating Time Last visualization...")
                progress.setValue(70)
                QApplication.processEvents()

                self.lineage_visualizer.create_cartoony_lineage_comparison(
                    self.lineage_tracks, time_last_canvas,
                    root_cell_id=root_cell_id, time_point="last")

                # Calculate diversity metrics
                progress.setLabelText("Calculating diversity metrics...")
                progress.setValue(90)
                QApplication.processEvents()

                metrics = self.lineage_visualizer.calculate_diversity_metrics(
                    self.lineage_tracks)

                progress.setValue(100)

            except Exception as e:
                QMessageBox.warning(
                    dialog, "Error", f"Error generating visualization: {str(e)}")
                print(f"Visualization error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                progress.close()

        # Navigation functions
        def go_to_next_tree():
            if not available_trees or len(available_trees) <= 1:
                return

            # Move to next tree
            current_tree_index[0] = (
                current_tree_index[0] + 1) % len(available_trees)
            generate_visualizations()

        def go_to_previous_tree():
            if not available_trees or len(available_trees) <= 1:
                return

            # Move to previous tree
            current_tree_index[0] = (
                current_tree_index[0] - 1) % len(available_trees)
            generate_visualizations()

        # Connect button signals
        view_button.clicked.connect(generate_visualizations)
        save_button.clicked.connect(
            lambda: save_images(tab_widget.currentIndex()))
        close_button.clicked.connect(dialog.close)
        prev_button.clicked.connect(go_to_previous_tree)
        next_button.clicked.connect(go_to_next_tree)

        # Function to save images depending on which tab is active
        def save_images(tab_index):
            if tab_index == 0:  # Time Comparison tab
                save_path, _ = QFileDialog.getSaveFileName(
                    dialog, "Save Visualization", "", "PNG Files (*.png)")
                if save_path:
                    # Extract base path without extension
                    base_path = save_path.replace(".png", "")

                    # Get current tree info for filename
                    tree_info = ""
                    if top_radio.isChecked():
                        tree_info = f"tree{current_tree_index[0]+1}"
                    else:
                        tree_info = f"cell{cell_combo.currentText().replace('Cell ', '')}"

                    # Save time zero tree
                    time_zero_path = f"{base_path}_{tree_info}_time_zero.png"
                    time_zero_figure.savefig(
                        time_zero_path, dpi=300, bbox_inches='tight')

                    # Save time last tree
                    time_last_path = f"{base_path}_{tree_info}_time_last.png"
                    time_last_figure.savefig(
                        time_last_path, dpi=300, bbox_inches='tight')

                    QMessageBox.information(
                        dialog, "Save Complete",
                        f"Images saved as:\n{time_zero_path}\n{time_last_path}")

            elif tab_index == 1:  # Growth & Division tab
                save_path, _ = QFileDialog.getSaveFileName(
                    dialog, "Save Growth Analysis", "", "PNG Files (*.png)")
                if save_path:
                    growth_fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    QMessageBox.information(
                        dialog, "Save Complete",
                        f"Growth analysis saved as:\n{save_path}")

        # Initial generation
        generate_visualizations()

        # Show the dialog
        dialog.exec_()

    def show_lineage_dialog(self):
        if not hasattr(self, "lineage_tracks") or not self.lineage_tracks:
            QMessageBox.warning(
                self,
                "Error",
                "No tracking data available. Run tracking with lineage first.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Lineage Tree Visualization")
        dialog.setMinimumWidth(800)
        dialog.setMinimumHeight(700)
        layout = QVBoxLayout(dialog)

        # Add visualization type selection
        viz_layout = QHBoxLayout()
        viz_label = QLabel("Visualization Style:")
        viz_layout.addWidget(viz_label)

        viz_type = QComboBox()
        viz_type.addItems(
            ["Standard Lineage Tree", "Morphology-Enhanced Tree"])
        viz_layout.addWidget(viz_type)
        layout.addLayout(viz_layout)

        # Cell selection options (same as original)
        selection_layout = QHBoxLayout()
        option_group = QButtonGroup(dialog)
        top_radio = QRadioButton("Top 5 Largest Lineage Trees")
        top_radio.setChecked(True)
        option_group.addButton(top_radio)
        selection_layout.addWidget(top_radio)
        cell_radio = QRadioButton("Specific Cell Lineage:")
        option_group.addButton(cell_radio)
        selection_layout.addWidget(cell_radio)
        cell_combo = QComboBox()
        cell_combo.setEnabled(False)
        selection_layout.addWidget(cell_combo)
        dividing_cells = [
            track['ID'] for track in self.lineage_tracks if track.get(
                'children', [])]
        dividing_cells.sort()
        for cell_id in dividing_cells:
            cell_combo.addItem(f"Cell {cell_id}")
        layout.addLayout(selection_layout)

        def update_combo_state():
            cell_combo.setEnabled(cell_radio.isChecked())
        top_radio.toggled.connect(update_combo_state)
        cell_radio.toggled.connect(update_combo_state)

        # Canvas for the visualization (same as original)
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        figure = Figure(figsize=(9, 6), tight_layout=True)
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)

        # NEW: Navigation layout for previous/next buttons
        nav_layout = QHBoxLayout()
        prev_button = QPushButton("Previous Tree")
        next_button = QPushButton("Next Tree")

        # Add a label to show current tree number
        tree_counter_label = QLabel("Tree 1/1")
        tree_counter_label.setAlignment(Qt.AlignCenter)

        nav_layout.addWidget(prev_button)
        nav_layout.addWidget(tree_counter_label)
        nav_layout.addWidget(next_button)
        layout.addLayout(nav_layout)

        # Add variables to track current tree index and available trees
        # Use list to allow modification inside closures
        current_tree_index = [0]
        available_trees = []  # Will store the list of trees

        # Original buttons
        button_layout = QHBoxLayout()
        view_button = QPushButton("Visualize")
        save_button = QPushButton("Save")
        close_button = QPushButton("Close")
        button_layout.addWidget(view_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(close_button)

        # Add this button to your existing lineage dialog
        comparison_button = QPushButton("Compare Time Zero vs Time Last")
        comparison_button.clicked.connect(
            self.show_timepoint_lineage_comparison)
        button_layout.addWidget(comparison_button)

        layout.addLayout(button_layout)

        def create_visualization():
            selected_cell = None
            if cell_radio.isChecked() and cell_combo.currentText():
                selected_cell = int(
                    cell_combo.currentText().replace("Cell ", ""))
                # Reset index when showing specific cell
                current_tree_index[0] = 0

            # Handle tree identification - this is common code for both visualization types
            if top_radio.isChecked():
                # Get all trees by analyzing connected components
                import networkx as nx
                G = nx.DiGraph()

                # Build the graph
                for track in self.lineage_tracks:
                    G.add_node(track['ID'])
                    if 'children' in track and track['children']:
                        for child_id in track['children']:
                            G.add_edge(track['ID'], child_id)

                # Find connected components (these are our trees)
                connected_components = list(nx.weakly_connected_components(G))
                # Sort by size (largest first)
                available_trees.clear()
                available_trees.extend(
                    sorted(connected_components, key=len, reverse=True)[:5])

                # Make sure current index is valid
                if not available_trees:
                    current_tree_index[0] = 0
                elif current_tree_index[0] >= len(available_trees):
                    current_tree_index[0] = 0

                # Get root of current tree
                if available_trees:
                    tree_nodes = list(available_trees[current_tree_index[0]])

                    # Find root nodes (no parents in this tree)
                    root_candidates = []
                    for node in tree_nodes:
                        is_root = True
                        for _, child in G.in_edges(node):
                            if child in tree_nodes:
                                is_root = False
                                break
                        if is_root:
                            root_candidates.append(node)

                    # If no clear root, use the earliest appearing cell (lowest ID typically)
                    if root_candidates:
                        root_cell_id = min(root_candidates)
                    else:
                        root_cell_id = min(tree_nodes)

                    # Update counter display
                    tree_counter_label.setText(
                        f"Tree {current_tree_index[0]+1}/{len(available_trees)}")
                else:
                    root_cell_id = None
                    tree_counter_label.setText("Tree 0/0")

                # Enable/disable navigation buttons
                prev_button.setEnabled(len(available_trees) > 1)
                next_button.setEnabled(len(available_trees) > 1)
            else:
                # Specific cell mode
                root_cell_id = selected_cell
                available_trees.clear()
                tree_counter_label.setText("Cell Lineage")

                # Disable navigation buttons in cell-specific mode
                prev_button.setEnabled(False)
                next_button.setEnabled(False)

            # Choose visualization based on combo box selection
            if viz_type.currentText() == "Morphology-Enhanced Tree":
                # Use the morphology visualization with the selected root
                self.lineage_visualizer.visualize_morphology_lineage_tree(
                    self.lineage_tracks, canvas, root_cell_id)
            else:
                # Use the standard visualization with the selected root
                self.lineage_visualizer.create_lineage_tree(
                    self.lineage_tracks, canvas, root_cell_id=root_cell_id)

        def go_to_next_tree():
            if not available_trees or len(available_trees) <= 1:
                return

            # Move to next tree
            current_tree_index[0] = (
                current_tree_index[0] + 1) % len(available_trees)

            # Show activity indicator
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                create_visualization()
            finally:
                QApplication.restoreOverrideCursor()

        def go_to_previous_tree():
            if not available_trees or len(available_trees) <= 1:
                return

            # Move to previous tree
            current_tree_index[0] = (
                current_tree_index[0] - 1) % len(available_trees)

            # Show activity indicator
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                create_visualization()
            finally:
                QApplication.restoreOverrideCursor()

        def save_visualization():
            output_path, _ = QFileDialog.getSaveFileName(
                dialog, "Save Lineage Tree", "", "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)")
            if output_path:
                # Use a higher DPI for better image quality
                figure.savefig(output_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(
                    dialog, "Success", f"Lineage tree saved to {output_path}")

        def maximize_dialog():
            """Toggle between normal and maximized window state"""
            if dialog.isMaximized():
                dialog.showNormal()
            else:
                dialog.showMaximized()

        # Add a maximize button
        maximize_button = QPushButton("Maximize Window")
        maximize_button.clicked.connect(maximize_dialog)
        button_layout.addWidget(maximize_button)

        # Connect signals
        view_button.clicked.connect(create_visualization)
        save_button.clicked.connect(save_visualization)
        close_button.clicked.connect(dialog.close)
        viz_type.currentIndexChanged.connect(create_visualization)

        # Connect navigation buttons
        next_button.clicked.connect(go_to_next_tree)
        prev_button.clicked.connect(go_to_previous_tree)

        # Initial visualization
        create_visualization()
        dialog.exec_()

    def visualize_lineage(self):
        """
        Visualize the lineage tree from tracking data, focusing on a single cell.
        """
        if not hasattr(self, "tracked_cells") or not self.tracked_cells:
            QMessageBox.warning(
                self,
                "Error",
                "No tracking data available. Run cell tracking first.")
            return

        try:
            import networkx as nx
            import matplotlib.pyplot as plt

            # Ask user if they want to pick a specific cell or have one
            # selected automatically
            reply = QMessageBox.question(
                self,
                "Lineage Visualization",
                "Would you like to select a specific cell to view its lineage?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes)

            # Get list of cell IDs
            cell_ids = [track['ID'] for track in self.tracked_cells]

            selected_cell = None
            if reply == QMessageBox.Yes:
                # Create dialog for selection
                dialog = QDialog(self)
                dialog.setWindowTitle("Select Cell")
                layout = QVBoxLayout(dialog)

                label = QLabel("Select a cell to visualize:")
                layout.addWidget(label)

                combo = QComboBox()
                combo.addItems([str(cell_id) for cell_id in cell_ids])
                layout.addWidget(combo)

                buttons = QDialogButtonBox(
                    QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                buttons.accepted.connect(dialog.accept)
                buttons.rejected.connect(dialog.reject)
                layout.addWidget(buttons)

                if dialog.exec_() == QDialog.Accepted:
                    selected_cell = int(combo.currentText())
                else:
                    return
            else:
                # Pick the cell with the longest track
                longest_track = max(self.tracked_cells,
                                    key=lambda t: len(t['x']))
                selected_cell = longest_track['ID']

            # Create a graph to represent the lineage
            G = nx.DiGraph()

            # Add the selected cell as the root node
            track = next(
                (t for t in self.tracked_cells if t['ID'] == selected_cell), None)
            if not track:
                QMessageBox.warning(
                    self, "Error", f"Cell {selected_cell} not found.")
                return

            # Add metadata to the root node
            G.add_node(selected_cell,
                       first_frame=track['t'][0] if track['t'] else 0,
                       frames=len(track['t']) if track['t'] else len(
                           track['x']),
                       track_data=track)

            # Ask for save location
            output_path, _ = QFileDialog.getSaveFileName(
                self, "Save Lineage Tree", "", "PNG Files (*.png);;All Files (*)")

            # Find temporal relationships (potential divisions)
            # For each track, find tracks that start when it ends
            end_frame_map = {}  # Maps end frame to list of tracks ending at that frame
            start_frame_map = {}  # Maps start frame to list of tracks starting at that frame

            for t in self.tracked_cells:
                if 't' in t and t['t']:
                    start_frame = t['t'][0]
                    end_frame = t['t'][-1]

                    if end_frame not in end_frame_map:
                        end_frame_map[end_frame] = []
                    end_frame_map[end_frame].append(t)

                    if start_frame not in start_frame_map:
                        start_frame_map[start_frame] = []
                    start_frame_map[start_frame].append(t)

            # Recursively build the tree from the selected cell
            def build_tree(cell_id, current_depth=0, max_depth=3):
                if current_depth >= max_depth:
                    return

                # Get the track for this cell
                parent_track = next(
                    (t for t in self.tracked_cells if t['ID'] == cell_id), None)
                if not parent_track or 't' not in parent_track or not parent_track['t']:
                    return

                # Get the end frame for this track
                end_frame = parent_track['t'][-1]

                # Get potential children (tracks that start right after this
                # one ends)
                children_candidates = []

                # Look at the next few frames for potential children
                for frame in range(end_frame, end_frame + 3):
                    if frame in start_frame_map:
                        # Get tracks that start at this frame
                        for child_track in start_frame_map[frame]:
                            # Skip if it's the parent itself
                            if child_track['ID'] == parent_track['ID']:
                                continue

                            # Calculate proximity between end of parent and
                            # start of child
                            if len(
                                    parent_track['x']) > 0 and len(
                                    child_track['x']) > 0:
                                parent_end_x = parent_track['x'][-1]
                                parent_end_y = parent_track['y'][-1]
                                child_start_x = child_track['x'][0]
                                child_start_y = child_track['y'][0]

                                # Calculate distance
                                distance = ((parent_end_x - child_start_x)**2 +
                                            (parent_end_y - child_start_y)**2)**0.5

                                # If close enough, consider it a potential
                                # child
                                if distance < 30:  # Adjust threshold as needed
                                    children_candidates.append(
                                        (child_track['ID'], distance))

                # Sort by distance and take up to 2 closest as children (for
                # division)
                children_candidates.sort(key=lambda x: x[1])
                children = [c[0] for c in children_candidates[:2]]

                # Add edges to the graph
                for child_id in children:
                    if child_id not in G:
                        # Get child track
                        child_track = next(
                            (t for t in self.tracked_cells if t['ID'] == child_id), None)
                        if child_track:
                            # Add child to graph
                            G.add_node(
                                child_id,
                                first_frame=child_track['t'][0] if child_track['t'] else 0,
                                frames=len(
                                    child_track['t']) if child_track['t'] else len(
                                    child_track['x']),
                                track_data=child_track)
                            # Add edge
                            G.add_edge(cell_id, child_id)
                            # Recursively build tree for this child
                            build_tree(child_id, current_depth + 1, max_depth)

            # Start building the tree
            build_tree(selected_cell)

            # Visualization
            plt.figure(figsize=(10, 8))

            # Use hierarchical layout for tree
            pos = None
            try:
                import numpy as np
                # Since we don't have GraphViz, create a custom tree layout
                pos = {}
                for node in G.nodes():
                    # Get node depth (how many steps from root)
                    try:
                        path_length = len(nx.shortest_path(
                            G, selected_cell, node)) - 1
                    except nx.NetworkXNoPath:
                        path_length = 0

                    # Get all nodes at this depth
                    nodes_at_depth = [
                        n for n in G.nodes() if len(
                            nx.shortest_path(
                                G,
                                selected_cell,
                                n)) - 1 == path_length]

                    # Calculate x position based on position among siblings
                    index = nodes_at_depth.index(node)
                    num_nodes_at_depth = len(nodes_at_depth)

                    if num_nodes_at_depth > 1:
                        x = index / (num_nodes_at_depth - 1)
                    else:
                        x = 0.5

                    # Adjust to spread out the tree
                    x = (x - 0.5) * (1 + path_length) + 0.5

                    # Y coordinate is negative depth (to grow downward)
                    y = -path_length

                    pos[node] = (x, y)
            except Exception as e:
                print(f"Error creating layout: {e}")
                # Fallback to spring layout
                pos = nx.spring_layout(G)

            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=700,
                                   node_color='lightblue', alpha=0.8)

            # Draw edges with arrows
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=2,
                                   arrows=True, arrowstyle='-|>', arrowsize=20)

            # Draw labels with track info
            labels = {
                n: f"ID: {n}\nFrames: {G.nodes[n]['frames']}" for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

            # Title and styling
            plt.title(f"Cell Lineage Tree (Root: Cell {selected_cell})")
            plt.grid(False)
            plt.axis('off')

            # Add stats
            node_count = G.number_of_nodes()
            edge_count = G.number_of_edges()
            stats_text = f"Total Cells: {node_count}\nDivision Events: {edge_count}\nRoot Cell: {selected_cell}"
            plt.figtext(0.02, 0.02, stats_text, bbox=dict(
                facecolor='white', alpha=0.8))

            # Save if path provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(
                self,
                "Visualization Error",
                f"Failed to visualize lineage tree: {str(e)}"
            )

    def analyze_cell_motility(self):
        """
        Analyze cell motility using the enhanced model and display visualizations.
        """

        if not hasattr(self, "lineage_tracks") or not self.lineage_tracks:
            QMessageBox.warning(
                self,
                "Error",
                "No tracking data available. Run cell tracking first.")
            return

        # Ask user which set of tracks to use - with custom button text
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Motility Analysis Options")
        msg_box.setText("Which tracks would you like to analyze?")
        msg_box.setInformativeText(
            "• Filtered Tracks: Uses only the longest, most reliable tracks (recommended)\n"
            "• All Tracks: Uses all detected cell tracks for a complete population analysis")

        # Create custom buttons
        filtered_button = msg_box.addButton(
            "Filtered Tracks", QMessageBox.ActionRole)
        all_button = msg_box.addButton("All Tracks", QMessageBox.ActionRole)
        cancel_button = msg_box.addButton(QMessageBox.Cancel)

        msg_box.exec()

        # Handle user choice
        if msg_box.clickedButton() == filtered_button:
            tracks_to_analyze = self.tracked_cells
            track_type = "filtered"
        elif msg_box.clickedButton() == all_button:
            tracks_to_analyze = self.lineage_tracks
            track_type = "all"
        else:
            # User clicked Cancel
            return

        # Check if selected tracks exist
        if not tracks_to_analyze:
            QMessageBox.warning(
                self, "Error", f"No {track_type} tracks available.")
            return

        # Show progress dialog
        progress = QProgressDialog(
            "Analyzing cell motility...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        # Get chamber dimensions if available
        chamber_dimensions = None
        try:
            if hasattr(self, "image_data") and hasattr(self.image_data, "data"):
                if self.image_data.is_nd2:
                    if len(self.image_data.data.shape) >= 4:
                        height = self.image_data.data.shape[-2]
                        width = self.image_data.data.shape[-1]
                        chamber_dimensions = (width, height)
                else:
                    height, width = self.image_data.data.shape[1:3]
                    chamber_dimensions = (width, height)

                if chamber_dimensions and (chamber_dimensions[0] < 10 or chamber_dimensions[1] < 10):
                    print(
                        f"Invalid chamber dimensions: {chamber_dimensions}, using defaults")
                    chamber_dimensions = (1392, 1040)
                else:
                    print(f"Using chamber dimensions: {chamber_dimensions}")
            else:
                chamber_dimensions = (1392, 1040)
                print(
                    f"Using default chamber dimensions: {chamber_dimensions}")
        except Exception as e:
            print(f"Error determining chamber dimensions: {e}")
            chamber_dimensions = (1392, 1040)

        try:
            # Calculate enhanced motility metrics
            progress.setValue(20)
            progress.setLabelText("Calculating motility metrics...")
            QApplication.processEvents()

            motility_metrics = enhanced_motility_index(
                tracks_to_analyze, chamber_dimensions)

            progress.setValue(50)
            progress.setLabelText(
                "Collecting cell positions for visualization...")
            QApplication.processEvents()

            # Collect all cell positions from segmentation data
            all_cell_positions = []
            p = self.slider_p.value()
            c = self.slider_c.value() if self.has_channels else None

            try:
                for t in range(min(20, self.dimensions.get("T", 1))):
                    binary_image = self.image_data.segmentation_cache[t, p, c]
                    if binary_image is not None:
                        labeled_image = label(binary_image)
                        for region in regionprops(labeled_image):
                            y, x = region.centroid
                            all_cell_positions.append((x, y))
            except Exception as e:
                print(f"Error collecting cell positions: {e}")
                all_cell_positions = []
                for track in tracks_to_analyze:
                    all_cell_positions.extend(
                        list(zip(track['x'], track['y'])))

            print(
                f"Collected {len(all_cell_positions)} cell positions for visualization")

            # Create combined visualization tab
            progress.setValue(60)
            progress.setLabelText("Creating combined visualization...")
            QApplication.processEvents()

            combined_tab = QWidget()
            combined_layout = QVBoxLayout(combined_tab)
            combined_fig, _ = visualize_motility_with_chamber_regions(
                tracks_to_analyze, all_cell_positions, chamber_dimensions, motility_metrics)
            combined_canvas = FigureCanvas(combined_fig)
            combined_layout.addWidget(combined_canvas)

            # Create dialog and tab widget
            dialog = QDialog(self)
            dialog.setWindowTitle("Cell Motility Analysis")
            dialog.setMinimumWidth(1200)
            dialog.setMinimumHeight(800)
            layout = QVBoxLayout(dialog)
            tab_widget = QTabWidget()

            # Add combined tab as first tab
            tab_widget.insertTab(0, combined_tab, "Motility by Region")
            tab_widget.setCurrentIndex(0)

            # Motility Map Tab
            progress.setValue(40)
            progress.setLabelText("Creating motility visualizations...")
            QApplication.processEvents()

            map_tab = QWidget()
            map_layout = QVBoxLayout(map_tab)
            map_fig, _ = visualize_motility_map(
                tracks_to_analyze, chamber_dimensions, motility_metrics)
            map_canvas = FigureCanvas(map_fig)
            map_layout.addWidget(map_canvas)

            # Detailed Metrics Tab
            metrics_tab = QWidget()
            metrics_layout = QVBoxLayout(metrics_tab)
            metrics_fig = visualize_motility_metrics(motility_metrics)
            metrics_canvas = FigureCanvas(metrics_fig)
            metrics_layout.addWidget(metrics_canvas)

            # Regional Analysis Tab
            region_tab = QWidget()
            region_layout = QVBoxLayout(region_tab)
            if chamber_dimensions:
                progress.setValue(70)
                progress.setLabelText("Analyzing regional variations...")
                QApplication.processEvents()
                regional_analysis, region_fig = analyze_motility_by_region(
                    tracks_to_analyze, chamber_dimensions, motility_metrics)
                region_canvas = FigureCanvas(region_fig)
                region_layout.addWidget(region_canvas)
            else:
                region_label = QLabel(
                    "Chamber dimensions not available for regional analysis.")
                region_label.setAlignment(Qt.AlignCenter)
                region_layout.addWidget(region_label)

            # Add tabs to tab widget
            tab_widget.addTab(map_tab, "Motility Map")
            tab_widget.addTab(metrics_tab, "Detailed Metrics")
            tab_widget.addTab(region_tab, "Regional Analysis")

            # Summary
            summary_text = (
                f"<h3>Motility Analysis Summary</h3>"
                f"<p><b>Population Average Motility Index:</b> {motility_metrics['population_avg_motility']:.1f}/100</p>"
                f"<p><b>Motility Heterogeneity:</b> {motility_metrics['population_heterogeneity']:.2f}</p>"
                f"<p><b>Sample Size:</b> {motility_metrics['sample_size']} cells</p>"
                f"<p>Analysis based on {track_type} tracks.</p>"
            )
            summary_label = QLabel(summary_text)
            summary_label.setTextFormat(Qt.RichText)
            summary_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(summary_label)
            layout.addWidget(tab_widget)

            # Buttons
            button_layout = QHBoxLayout()
            export_button = QPushButton("Export Results")
            close_button = QPushButton("Close")
            button_layout.addWidget(export_button)
            button_layout.addWidget(close_button)
            layout.addLayout(button_layout)

            # Export function
            def export_results():
                export_dialog = QDialog(dialog)
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
                export_regional.setChecked(chamber_dimensions is not None)
                export_regional.setEnabled(chamber_dimensions is not None)
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
                        base_path = save_path.replace(
                            ".png", "").replace(".csv", "")
                        if export_map.isChecked():
                            map_fig.savefig(
                                f"{base_path}_motility_map.png", dpi=300, bbox_inches='tight')
                        if export_metrics.isChecked():
                            metrics_fig.savefig(
                                f"{base_path}_detailed_metrics.png", dpi=300, bbox_inches='tight')
                        if export_regional.isChecked() and chamber_dimensions:
                            region_fig.savefig(
                                f"{base_path}_regional_analysis.png", dpi=300, bbox_inches='tight')
                        if export_csv.isChecked():
                            metrics_df = pd.DataFrame(
                                motility_metrics['individual_metrics'])
                            metrics_df.to_csv(
                                f"{base_path}_motility_metrics.csv", index=False)
                        QMessageBox.information(export_dialog, "Export Complete",
                                                f"Results exported to {base_path}_*.png/csv")

            export_button.clicked.connect(export_results)
            close_button.clicked.connect(dialog.close)

            progress.setValue(100)
            progress.close()

            # Store results
            self.motility_results = {
                "motility_metrics": motility_metrics, "track_type": track_type}

            # Update main UI plot
            self.figure_morphology_fractions.clear()
            ax = self.figure_morphology_fractions.add_subplot(111)
            for track in tracks_to_analyze:
                track_id = track.get('ID', -1)
                metric = next((m for m in motility_metrics['individual_metrics']
                               if m['track_id'] == track_id), None)
                if metric:
                    mi = metric['motility_index']
                    color = plt.cm.coolwarm(mi/100)
                    ax.plot(track['x'], track['y'], '-',
                            color=color, linewidth=1, alpha=0.7)

            ax.set_title(
                f"Cell Motility Map (Population Avg: {motility_metrics['population_avg_motility']:.1f})")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.coolwarm, norm=plt.Normalize(0, 100))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Motility Index")
            self.canvas_morphology_fractions.draw()

            dialog.exec()

        except Exception as e:
            import traceback
            traceback.print_exc()
            progress.close()
            QMessageBox.warning(self, "Analysis Error",
                                f"Error analyzing motility: {str(e)}")

    def process_morphology_time_series(self):
        p = self.slider_p.value()
        c = self.slider_c.value() if "C" in self.dimensions else None  # Default C to None

        if not self.image_data.is_nd2:
            QMessageBox.warning(
                self, "Error", "This feature only supports ND2 datasets.")
            return

        try:
            # Extract image data for all time points
            t = self.dimensions["T"]  # Get the total number of time points
            self.image_data.segmentation_cache.with_model(
                self.model_dropdown.currentText())
            if "C" in self.dimensions:
                image_data = np.array(
                    self.image_data.data[0:t, p, c, :, :].compute()
                    if hasattr(self.image_data.data[0:t, p, c, :, :], "compute")
                    else self.image_data.data[0:t, p, c, :, :]
                )
            else:
                image_data = np.array(
                    self.image_data.data[0:t, p, :, :].compute()
                    if hasattr(self.image_data.data[0:t, p, :, :], "compute")
                    else self.image_data.data[0:t, p, :, :]
                )

            if image_data.size == 0:
                QMessageBox.warning(
                    self,
                    "Error",
                    "No valid data found for the selected position and channel.",
                )
                return
        except Exception as e:
            QMessageBox.warning(
                self, "Data Error", f"Failed to extract image data: {e}"
            )
            return

        num_frames = image_data.shape[0]
        self.progress_bar.setMaximum(num_frames)
        self.progress_bar.setValue(0)

        # Disable the button while the worker is running
        self.segment_button.setEnabled(False)

        # Create the worker and thread
        self.worker = MorphologyWorker(
            self.image_data, image_data, num_frames, p, c
        )
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        # Connect worker signals
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.handle_results)
        self.worker.error.connect(self.handle_error)
        self.worker.finished.connect(self.thread.quit)

        # Cleanup
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Re-enable button when thread finishes
        self.thread.finished.connect(
            lambda: self.segment_button.setEnabled(True))

        self.thread.start()

    def handle_results(self, results):
        if not results:
            QMessageBox.warning(
                self,
                "Error",
                "No valid results received. Please check the input data.")
            return

        # Get all unique morphology classes across all frames
        all_morphologies = set()
        for frame_data in results.values():
            all_morphologies.update(frame_data["fractions"].keys())

        # Initialize dictionaries for both fractions and counts
        morphology_fractions = {morphology: []
                                for morphology in all_morphologies}
        morphology_counts = {morphology: [] for morphology in all_morphologies}
        total_cells_per_frame = []

        # Get the maximum time point
        max_time = max(results.keys())

        # Fill in the data for each time point
        for t in range(max_time + 1):
            if t in results:
                frame_data = results[t]["fractions"]

                # Get raw counts from the metrics table when available
                if "metrics" in results[t]:
                    metrics_df = results[t]["metrics"]
                    class_counts = metrics_df["morphology_class"].value_counts(
                    ).to_dict()
                    total_cells = len(metrics_df)

                    for morph_class, count in class_counts.items():
                        print(
                            f"  {morph_class}: {count} cells ({count/total_cells*100:.1f}%)")
                else:
                    class_counts = {morph: 0 for morph in all_morphologies}
                    total_cells = 0

                # Store total cell count for this frame
                total_cells_per_frame.append(total_cells)

                for morphology in all_morphologies:
                    # Get the fraction if present, otherwise use 0.0
                    fraction = frame_data.get(morphology, 0.0)
                    morphology_fractions[morphology].append(fraction)

                    # Store the raw count
                    count = class_counts.get(morphology, 0)
                    morphology_counts[morphology].append(count)
            else:
                # For frames with no data, append 0.0 for all morphologies
                total_cells_per_frame.append(0)
                for morphology in all_morphologies:
                    morphology_fractions[morphology].append(0.0)
                    morphology_counts[morphology].append(0)

        # Create a figure with two subplots - fractions and counts
        self.figure_morphology_fractions.clear()
        fig = self.figure_morphology_fractions

        # First subplot - fractions (as before)
        ax1 = fig.add_subplot(2, 1, 1)
        for morphology, fractions in morphology_fractions.items():
            color = self.morphology_colors_rgb.get(
                morphology, (0.5, 0.5, 0.5))  # Default to gray if color not found
            ax1.plot(
                range(len(fractions)),
                fractions,
                marker="o",
                label=morphology,
                color=color)
        ax1.set_title("Morphology Fractions Over Time")
        ax1.set_ylabel("Fraction")
        ax1.legend()

        # Second subplot - raw counts
        ax2 = fig.add_subplot(2, 1, 2)
        for morphology, counts in morphology_counts.items():
            color = self.morphology_colors_rgb.get(
                morphology, (0.5, 0.5, 0.5))  # Default to gray if color not found
            ax2.plot(
                range(len(counts)),
                counts,
                marker="o",
                label=morphology,
                color=color)
        ax2.set_title("Cell Counts By Morphology Over Time")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Count")

        # Plot total cell count on a separate axis
        ax3 = ax2.twinx()
        ax3.plot(range(len(total_cells_per_frame)), total_cells_per_frame,
                 color='black', linestyle='--', label='Total Cells')
        ax3.set_ylabel("Total Cell Count")

        # Add combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        fig.tight_layout()
        self.canvas_morphology_fractions.draw()

    def handle_error(self, error_message):
        print(f"Error: {error_message}")
        QMessageBox.warning(self, "Processing Error", error_message)
        raise Exception(error_message)

    def update_plot(self):
        selected_metric = self.metric_dropdown.currentText()

        if not hasattr(self, "morphologies_over_time"):
            QMessageBox.warning(
                self, "Error", "No data to plot. Please process the frames first.")
            return

        if selected_metric not in self.morphologies_over_time.columns:
            QMessageBox.warning(
                self, "Error", f"Metric {selected_metric} not found in results.")
            return

        metric_data = self.morphologies_over_time[selected_metric]
        if metric_data.empty:
            QMessageBox.warning(
                self, "Error", f"No valid data available for {selected_metric}.")
            return

        self.figure_time_series.clear()
        ax = self.figure_time_series.add_subplot(111)
        ax.plot(metric_data, marker="o")
        ax.set_title(f"{selected_metric.capitalize()} Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel(selected_metric.capitalize())
        self.canvas_time_series.draw()

    def initViewArea(self):
        layout = QVBoxLayout(self.viewArea)

        self.image_label = QLabel()
        # Allow the label to scale the image
        self.image_label.setScaledContents(True)
        self.image_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label)

        self.image_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(
            self.show_context_menu)

        annotate_button = QPushButton("Classify Cells")
        annotate_button.clicked.connect(self.annotate_cells)
        layout.addWidget(annotate_button)

        segmentation_group = QGroupBox("Segmentation")
        segmentation_layout = QHBoxLayout()

        segment_position_button = QPushButton("Position")
        segment_position_button.clicked.connect(self.segment_this_p)
        segmentation_layout.addWidget(segment_position_button)

        segment_all_button = QPushButton("Everything")
        segment_all_button.clicked.connect(self.segment_all)
        segmentation_layout.addWidget(segment_all_button)

        segmentation_group.setLayout(segmentation_layout)
        layout.addWidget(segmentation_group)

        nd2_controls_group = QGroupBox("ND2 controls")
        nd2_controls_layout = QVBoxLayout()

        # T controls
        t_layout = QHBoxLayout()
        t_label = QLabel("T: 0")
        t_layout.addWidget(t_label)
        self.t_left_button = QPushButton("<")
        self.t_left_button.clicked.connect(
            lambda: self.slider_t.setValue(self.slider_t.value() - 1)
        )
        t_layout.addWidget(self.t_left_button)

        self.slider_t = QSlider(Qt.Horizontal)
        self.slider_t.valueChanged.connect(self.display_image)
        self.slider_t.valueChanged.connect(
            lambda value: t_label.setText(f"T: {value}"))

        t_layout.addWidget(self.slider_t)

        self.t_right_button = QPushButton(">")
        self.t_right_button.clicked.connect(
            lambda: self.slider_t.setValue(self.slider_t.value() + 1)
        )
        t_layout.addWidget(self.t_right_button)

        nd2_controls_layout.addLayout(t_layout)

        # P controls
        p_layout = QHBoxLayout()
        p_label = QLabel("P: 0")
        p_layout.addWidget(p_label)
        self.p_left_button = QPushButton("<")
        self.p_left_button.clicked.connect(
            lambda: self.slider_p.setValue(self.slider_p.value() - 1)
        )
        p_layout.addWidget(self.p_left_button)

        self.slider_p = QSlider(Qt.Horizontal)
        self.slider_p.valueChanged.connect(self.display_image)
        self.slider_p.valueChanged.connect(
            lambda value: p_label.setText(f"P: {value}"))
        p_layout.addWidget(self.slider_p)

        self.p_right_button = QPushButton(">")
        self.p_right_button.clicked.connect(
            lambda: self.slider_p.setValue(self.slider_p.value() + 1)
        )
        p_layout.addWidget(self.p_right_button)

        nd2_controls_layout.addLayout(p_layout)

        # C control (channel)
        c_layout = QHBoxLayout()
        c_label = QLabel("C: 0")
        c_layout.addWidget(c_label)
        self.c_left_button = QPushButton("<")
        self.c_left_button.clicked.connect(
            lambda: self.slider_c.setValue(self.slider_c.value() - 1)
        )
        c_layout.addWidget(self.c_left_button)

        self.slider_c = QSlider(Qt.Horizontal)
        self.slider_c.valueChanged.connect(self.display_image)
        self.slider_c.valueChanged.connect(
            lambda value: c_label.setText(f"C: {value}"))
        c_layout.addWidget(self.slider_c)

        self.c_right_button = QPushButton(">")
        self.c_right_button.clicked.connect(
            lambda: self.slider_c.setValue(self.slider_c.value() + 1)
        )
        c_layout.addWidget(self.c_right_button)

        nd2_controls_layout.addLayout(c_layout)
        nd2_controls_group.setLayout(nd2_controls_layout)
        layout.addWidget(nd2_controls_group)

        # Create a radio button for thresholding, normal and segmented
        self.radio_normal = QRadioButton("Normal")
        self.radio_segmented = QRadioButton("Segmented")
        self.radio_overlay_outlines = QRadioButton("Overlay with Outlines")
        # Add new radio button for labeled segmentation
        self.radio_labeled_segmentation = QRadioButton("Labeled Segmentation")

        # Create a button group and add the radio buttons to it
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.radio_normal)
        self.button_group.addButton(self.radio_labeled_segmentation)
        self.button_group.addButton(self.radio_segmented)
        self.button_group.addButton(self.radio_overlay_outlines)
        self.button_group.buttonClicked.connect(self.display_image)

        # Set default selection
        self.radio_normal.setChecked(True)

        # Add radio buttons to the layout
        layout.addWidget(self.radio_normal)
        layout.addWidget(self.radio_segmented)
        layout.addWidget(self.radio_overlay_outlines)
        layout.addWidget(self.radio_labeled_segmentation)

        # Segmentation model selection
        model_label = QLabel("Select Segmentation Model:")
        layout.addWidget(model_label)

        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(
            [
                SegmentationModels.CELLPOSE_BACT_PHASE,
                SegmentationModels.CELLPOSE_BACT_FLUOR,
                SegmentationModels.CELLPOSE,
                SegmentationModels.UNET,
                SegmentationModels.OMNIPOSE_BACT_PHASE_AFFINITY])
        self.model_dropdown.currentIndexChanged.connect(
            lambda: self.image_data.segmentation_cache.with_model(
                self.model_dropdown.currentText()))
        layout.addWidget(self.model_dropdown)

    def annotate_cells(self):
        t = self.slider_t.value()
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        # Ensure segmentation model is set correctly in segmentation_cache
        selected_model = self.model_dropdown.currentText()
        self.image_data.segmentation_cache.with_model(selected_model)

        # Retrieve segmentation from segmentation_cache
        segmented_image = self.image_data.segmentation_cache[t, p, c]

        if segmented_image is None:
            print(f"[ERROR] Segmentation failed for T={t}, P={p}, C={c}")
            QMessageBox.warning(self, "Segmentation Error",
                                "Segmentation failed.")
            return

        # Extract cell metrics and bounding boxes
        self.cell_mapping = extract_cells_and_metrics(
            self.image_data.data[t, p, c], segmented_image)

        if not self.cell_mapping:
            QMessageBox.warning(
                self, "No Cells", "No cells detected in the current frame.")
            return

        # Debugging
        # print(f"✅ Stored Cell Mapping: {list(self.cell_mapping.keys())}")

        # Annotate the binary segmented image
        self.annotated_image = annotate_binary_mask(
            segmented_image, self.cell_mapping)

        # Display the annotated image on the main image display
        # Convert annotated image to QImage
        height, width = self.annotated_image.shape[:2]
        qimage = QImage(
            self.annotated_image.data,
            width,
            height,
            self.annotated_image.strides[0],
            QImage.Format_RGB888,
        )

        # Convert to QPixmap and set to QLabel
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

        self.update_annotation_scatter()

    def show_context_menu(self, position):
        context_menu = QMenu(self)

        save_action = context_menu.addAction("Save Annotated Image")
        save_action.triggered.connect(self.save_annotated_image)

        context_menu.exec_(self.image_label.mapToGlobal(position))

    def save_annotated_image(self):
        if not hasattr(
                self,
                "annotated_image") or self.annotated_image is None:
            QMessageBox.warning(self, "Error", "No annotated image to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotated Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)")

        if file_path:
            cv2.imwrite(file_path, self.annotated_image)
            QMessageBox.information(
                self, "Success", f"Annotated image saved to {file_path}")
        else:
            QMessageBox.warning(self, "Error", "No file selected.")

    def export_images(self):
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save As", "", "TIFF Files (*.tif);;All Files (*)"
        )

        if not save_path:
            QMessageBox.warning(self, "Export", "No file selected.")
            return

        # Extract the directory and base name from the selected path
        folder_path = Path(save_path).parent
        custom_base_name = Path(save_path).stem

        max_t_value = self.slider_t.value()
        max_p_value = self.slider_p.value()

        for t in range(max_t_value + 1):
            for p in range(max_p_value + 1):
                # Retrieve the specific frame for time t and position p
                if self.image_data.is_nd2:
                    export_image = (
                        self.image_data.data[t, p].compute()
                        if hasattr(self.image_data.data, "compute")
                        else self.image_data.data[t, p]
                    )
                else:
                    export_image = self.image_data.data[t]

                img_to_save = np.array(export_image)

                # Construct the export path with the custom name and dimensions
                file_path = folder_path / f"{custom_base_name}_P{p}_T{t}.tif"
                cv2.imwrite(str(file_path), img_to_save)

        QMessageBox.information(
            self, "Export", f"Images exported successfully to {folder_path}")

    # Initialize the Export tab with the export button
    def initExportTab(self):
        layout = QVBoxLayout(self.exportTab)
        export_button = QPushButton("Export Images")
        export_button.clicked.connect(self.export_images)
        layout.addWidget(export_button)
        label = QLabel("This Tab Exports processed images sequentially.")
        layout.addWidget(label)

    def save_video(self, file_path):
        # Assuming self.image_data is a 4D numpy array with shape (frames,
        # height, width, channels)
        if hasattr(self, "image_data"):
            print(self.image_data.data.shape)

            with iio.imopen(file_path, "w", plugin="pyav") as writer:
                writer.init_video_stream(
                    "libx264", fps=30, pixel_format="yuv444p")

                writer._video_stream.options = {
                    "preset": "veryslow",
                    "qp": "0",
                }  # 'crf': '0',

                writer.write(self.image_data.data)

            # iio.imwrite(file_path, self.image_data.data,
            #             # plugin="pyav",
            #             plugin="ffmpeg",
            #             fps=30,
            #             codec='libx264',
            #             output_params=['-crf', '0',
            #                             '-preset', 'veryslow',
            #                             '-qp', '0'],
            #             pixelformat='yuv444p')

    def init_ui(self):
        """Initialize UI based on analysis mode"""
        # Create different tab configurations based on analysis mode
        if self.current_analysis_mode == AnalysisMode.SINGLE_CELL:
            self.init_single_cell_tabs()
            # Single cell specific initialization
            self.initMorphologyTimeTab()
            self.initMorphologyVisualizationTab()
        elif self.current_analysis_mode == AnalysisMode.BIOFILM_CLOUD:
            self.init_biofilm_cloud_tabs()

        # Initialize menu bar
        self.initMenuBar()

    def init_single_cell_tabs(self):
        """Initialize tabs for single cell analysis mode (current functionality)"""
        # Initialize tabs as QWidget
        self.segmentation_tab = SegmentationWidget()
        self.populationTab = PopulationWidget()
        self.morphologyTab = MorphologyWidget()
        self.morphologyVisualizationTab = QWidget()
        self.tracking_manager = TrackingManager(metrics_service=self.metrics_service)

        # Add tabs to the QTabWidget
        self.tab_widget.addTab(self.segmentation_tab, "Segmentation")
        self.tab_widget.addTab(self.populationTab, "Population")
        self.tab_widget.addTab(self.morphology_widget, "Morphology")
        self.tab_widget.addTab(self.tracking_manager,
                               "Tracking - Lineage Tree")

    def init_biofilm_cloud_tabs(self):
        """Initialize tabs for biofilm cloud analysis mode (placeholder for now)"""
        # For now, create placeholder tabs - we'll implement these in later tasks
        self.segmentation_tab = SegmentationWidget()  # Same segmentation

        self.cube_analysis_tab = CubeAnalysisWidget()

        placeholder_tab2 = QWidget()
        placeholder_tab2_layout = QVBoxLayout(placeholder_tab2)
        placeholder_tab2_layout.addWidget(
            QLabel("Colony Tracking - Coming Soon"))

        placeholder_tab3 = QWidget()
        placeholder_tab3_layout = QVBoxLayout(placeholder_tab3)
        placeholder_tab3_layout.addWidget(
            QLabel("Spatial Analysis - Coming Soon"))

        # Add tabs to the QTabWidget
        self.tab_widget.addTab(self.segmentation_tab, "Segmentation")
        self.tab_widget.addTab(self.cube_analysis_tab, "Cube Analysis") 
        self.tab_widget.addTab(placeholder_tab2, "Colony Tracking")
        self.tab_widget.addTab(placeholder_tab3, "Spatial Analysis")

    def initMenuBar(self):
        # Create the menu bar
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")

        # New Experiment
        new_experiment_action = QAction("New Experiment", self)
        new_experiment_action.setShortcut("Ctrl+E")
        new_experiment_action.triggered.connect(
            self.show_new_experiment_dialog)
        file_menu.addAction(new_experiment_action)

        # Save action
        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_to_folder)
        file_menu.addAction(save_action)

        # Load action
        load_action = QAction("Load", self)
        load_action.setShortcut("Ctrl+L")
        load_action.triggered.connect(self.load_from_folder)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        switch_mode_action = QAction("Switch Analysis Mode", self)
        switch_mode_action.setShortcut("Ctrl+M")
        switch_mode_action.triggered.connect(self.switch_analysis_mode)
        file_menu.addAction(switch_mode_action)

        # Help menu (for About dialog)
        help_menu = menu_bar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        # macOS specific: "About" in the application menu
        if sys.platform == "darwin":
            about_action_mac = QAction("About", self)
            about_action_mac.triggered.connect(self.show_about_dialog)
            self.menuBar().addAction(about_action_mac)

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
        mode_name = new_mode.display_name
        self.setWindowTitle(f"Partaker 3 - {mode_name}")

        # Remove all existing tabs
        while self.tab_widget.count():
            self.tab_widget.removeTab(0)

        # Rebuild tabs for the selected mode (without recreating the menu bar)
        if self.current_analysis_mode == AnalysisMode.SINGLE_CELL:
            self.init_single_cell_tabs()
            self.initMorphologyTimeTab()
            self.initMorphologyVisualizationTab()
        elif self.current_analysis_mode == AnalysisMode.BIOFILM_CLOUD:
            self.init_biofilm_cloud_tabs()

    def show_new_experiment_dialog(self):
        experiment = ExperimentDialog()
        experiment.exec_()

    def show_about_dialog(self):
        about_dialog = AboutDialog()
        about_dialog.exec_()
        
    def convert_nd2_to_tif(self, positions, time_start, time_end, channel, output_folder):
        """Convert selected ND2 data to TIF files"""
        if not hasattr(self, 'image_data') or not self.image_data:
            return
            
        import os
        import tifffile
        from PySide6.QtWidgets import QMessageBox
        
        total_files = len(positions) * (time_end - time_start + 1)
        converted = 0
        
        for pos in positions:
            for t in range(time_start, time_end + 1):
                try:
                    image = self.image_data._get_raw_image(t, pos, channel)
                    filename = f"pos_{pos:03d}_t_{t:04d}_ch_{channel}.tif"
                    filepath = os.path.join(output_folder, filename)
                    tifffile.imwrite(filepath, image)
                    converted += 1
                except Exception as e:
                    print(f"Error converting t={t}, pos={pos}: {e}")
        
        QMessageBox.information(self, "Export Complete", 
                               f"Successfully exported {converted}/{total_files} TIF files to:\n{output_folder}")

    # In your main App class (app.py)

    def save_to_folder(self):
        """Save the current project to a folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Destination Folder",
            "",
            QFileDialog.ShowDirsOnly
        )

        if folder_path:
            # Create directory if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)

            # Log segmentation cache state before saving
            if hasattr(self, "image_data") and hasattr(self.image_data, "segmentation_cache"):
                cache = self.image_data.segmentation_cache
                current_model = cache.model_name
                print(f"DEBUG: Saving project with segmentation cache")
                print(f"DEBUG: Current model: {current_model}")

                if current_model and current_model in cache.mmap_arrays_idx:
                    _, indices = cache.mmap_arrays_idx[current_model]
                    print(
                        f"DEBUG: Cache contains {len(indices)} segmented frames for model {current_model}")
                    print(
                        f"DEBUG: Sample indices: {list(indices)[:5] if indices else 'None'}")
                else:
                    print(f"DEBUG: No cached frames for model {current_model}")

            # Save image data
            try:
                print(f"DEBUG: Saving image data to {folder_path}")
                self.image_data.save(folder_path)
                print(f"DEBUG: Image data saved successfully")
            except Exception as e:
                print(f"ERROR: Failed to save image data: {str(e)}")
                import traceback
                traceback.print_exc()

            # Save metrics
            metrics_service = MetricsService()
            try:
                print(f"DEBUG: Saving metrics to {folder_path}")
                metrics_saved = metrics_service.save_to_file(folder_path)
                print(f"DEBUG: Metrics saved: {metrics_saved}")
            except Exception as e:
                print(f"ERROR: Failed to save metrics: {str(e)}")
                import traceback
                traceback.print_exc()

            population_saved = False
            if hasattr(self, "populationTab"):
                population_saved = self.populationTab.save_population_data(
                    folder_path)
                print(f"Population data saved: {population_saved}")

            # Save tracking data
            tracking_saved = self.tracking_manager.tracking_widget.save_tracking_data(
                folder_path)
            print(f"DEBUG: Tracking data saved: {tracking_saved}")

            # Show success message
            QMessageBox.information(
                self, "Save Complete",
                f"Project saved to {folder_path}" +
                ("\nIncludes tracking data" if tracking_saved else "")
            )

    def load_from_folder(self):
        """Load a project from a folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Project Folder", "", QFileDialog.ShowDirsOnly
        )

        if folder_path:
            print(f"Project loaded from folder: {folder_path}")

            try:
                # Load image data
                self.image_data = ImageData.load(folder_path)

                # Load metrics data
                metrics_service = MetricsService()
                metrics_loaded = metrics_service.load_from_file(folder_path)

                # Update UI based on loaded image data
                if hasattr(self, "viewArea"):
                    pub.sendMessage("image_data_loaded",
                                    image_data=self.image_data)

                population_loaded = False
                if hasattr(self, "populationTab"):
                    population_loaded = self.populationTab.load_population_data(
                        folder_path)
                    print(f"Population data loaded: {population_loaded}")

                # Load tracking data if available
                tracking_loaded = False
                if hasattr(self, "tracking_manager") and hasattr(self.tracking_manager, "tracking_widget"):
                    tracking_loaded = self.tracking_manager.tracking_widget.load_tracking_data(
                        folder_path)

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
                QMessageBox.warning(
                    self, "Error", f"Failed to load project: {str(e)}")

    def initMorphologyVisualizationTab(self):
        layout = QVBoxLayout(self.morphologyVisualizationTab)

        # Button layout for actions
        button_layout = QHBoxLayout()

        # Add the ideal examples button
        ideal_button = QPushButton("Select Ideal Cells")
        ideal_button.clicked.connect(self.select_ideal_examples)
        button_layout.addWidget(ideal_button)

        # Add similarity analysis button
        similarity_button = QPushButton("Analyze Similarity to Ideals")
        similarity_button.clicked.connect(self.calculate_similarity_to_ideals)
        button_layout.addWidget(similarity_button)

        # Add optimization button
        optimize_button = QPushButton("Optimize Classification")
        optimize_button.clicked.connect(
            self.optimize_classification_parameters)
        button_layout.addWidget(optimize_button)

        layout.addLayout(button_layout)

        # Matplotlib canvas for plotting
        self.figure_morphology_metrics = plt.figure()
        self.canvas_morphology_metrics = FigureCanvas(
            self.figure_morphology_metrics)
        layout.addWidget(self.canvas_morphology_metrics)

    def select_ideal_examples(self):
        """
        Allow the user to select ideal examples of each morphology class.
        """
        # Initialize a dictionary to store ideal examples
        if not hasattr(self, "ideal_examples"):
            self.ideal_examples = {
                "Artifact": None,
                "Divided": None,
                "Healthy": None,
                "Elongated": None,
                "Deformed": None
            }

        # Create a dialog to select ideal cells
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Ideal Cell Examples")
        dialog.setMinimumWidth(600)

        layout = QVBoxLayout(dialog)

        # Instructions
        instructions = QLabel(
            "Select one ideal example for each morphology class.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Create a split layout
        main_layout = QHBoxLayout()

        # Form layout for selection
        form_layout = QFormLayout()
        selection_widget = QWidget()
        selection_widget.setLayout(form_layout)

        # Preview area
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_label = QLabel("Cell Preview")
        preview_layout.addWidget(preview_label)
        self.preview_image = QLabel()
        self.preview_image.setMinimumSize(200, 200)
        self.preview_image.setAlignment(Qt.AlignCenter)
        self.preview_image.setScaledContents(True)
        preview_layout.addWidget(self.preview_image)

        main_layout.addWidget(selection_widget)
        main_layout.addWidget(preview_widget)
        layout.addLayout(main_layout)

        # Create dropdown selectors for each class
        self.ideal_selectors = {}

        # Get all cell IDs and their classifications
        cell_ids = []
        if hasattr(self, "cell_mapping") and self.cell_mapping:
            for cell_id, data in self.cell_mapping.items():
                if "metrics" in data and "morphology_class" in data["metrics"]:
                    cell_class = data["metrics"]["morphology_class"]
                    cell_ids.append((cell_id, cell_class))

        # Sort by class and ID
        cell_ids.sort(key=lambda x: (x[1], x[0]))

        # Create dropdowns for each class
        for class_name in self.ideal_examples.keys():
            combo = QComboBox()
            # Add all cells of this class
            class_cells = [str(cell_id) for cell_id,
                           cell_class in cell_ids if cell_class == class_name]

            if class_cells:
                combo.addItems(class_cells)

                # Set current selection if already defined
                if self.ideal_examples[class_name] is not None:
                    idx = combo.findText(str(self.ideal_examples[class_name]))
                    if idx >= 0:
                        combo.setCurrentIndex(idx)

                # Connect the currentIndexChanged signal
                combo.currentIndexChanged.connect(
                    lambda idx, cn=class_name: self.update_preview(cn))
            else:
                combo.addItem("No cells of this class")
                combo.setEnabled(False)

            self.ideal_selectors[class_name] = combo
            form_layout.addRow(f"Ideal {class_name}:", combo)

        # Buttons
        button_box = QHBoxLayout()
        save_button = QPushButton("Save Ideal Examples")
        save_button.clicked.connect(lambda: self.save_ideal_examples(dialog))
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)

        button_box.addWidget(save_button)
        button_box.addWidget(cancel_button)
        layout.addLayout(button_box)

        # Initially update the preview for the first class with cells
        for class_name in self.ideal_examples.keys():
            if self.ideal_selectors[class_name].isEnabled():
                self.update_preview(class_name)
                break

        # Show the dialog
        dialog.exec_()

    def update_preview(self, class_name):
        """
        Update the preview image for the selected cell.
        """
        combo = self.ideal_selectors[class_name]
        if combo.isEnabled() and combo.currentText() != "No cells of this class":
            try:
                cell_id = int(combo.currentText())
                # Get the bounding box for this cell
                y1, x1, y2, x2 = self.cell_mapping[cell_id]["bbox"]

                # Get the current frame
                t = self.slider_t.value()
                p = self.slider_p.value()
                c = self.slider_c.value() if self.has_channels else None

                # Extract the cell region from the segmented image
                segmented_image = self.image_data.segmentation_cache[t, p, c]

                # Create a visualization focusing on just this cell
                # Make a local crop of the segmented image around the cell
                padding = 10  # Extra pixels around the bounding box
                y_min = max(0, y1 - padding)
                y_max = min(segmented_image.shape[0], y2 + padding)
                x_min = max(0, x1 - padding)
                x_max = min(segmented_image.shape[1], x2 + padding)

                # Crop the segmented region
                cropped_seg = segmented_image[y_min:y_max, x_min:x_max]

                # Convert to RGB and highlight the cell
                cropped_rgb = cv2.cvtColor((cropped_seg > 0).astype(
                    np.uint8) * 255, cv2.COLOR_GRAY2BGR)

                # Create a mask for the target cell
                cell_mask = np.zeros_like(cropped_seg)
                # Adjust bounding box coordinates for the crop
                local_y1, local_x1 = y1 - y_min, x1 - x_min
                local_y2, local_x2 = y2 - y_min, x2 - x_min

                # Use connected components to find the cell within the bounding
                # box
                roi = cropped_seg[max(0, local_y1):min(cropped_seg.shape[0], local_y2),
                                  max(0, local_x1):min(cropped_seg.shape[1], local_x2)]

                if roi.max() > 0:
                    # Use connected components
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                        roi, connectivity=8)

                    # Find largest component
                    largest_label = 1
                    largest_area = 0
                    for label in range(1, num_labels):
                        area = stats[label, cv2.CC_STAT_AREA]
                        if area > largest_area:
                            largest_area = area
                            largest_label = label

                    # Create mask for largest component
                    component_mask = (labels == largest_label).astype(
                        np.uint8) * 255

                    # Place it in the full mask at the right position
                    cell_mask[max(0, local_y1):min(cropped_seg.shape[0], local_y2), max(
                        0, local_x1):min(cropped_seg.shape[1], local_x2)] = component_mask

                # Highlight the cell in the appropriate morphology color
                # Default to red if color not found
                color = self.morphology_colors.get(class_name, (0, 0, 255))
                cropped_rgb[cell_mask > 0] = color

                # Draw bounding box
                cv2.rectangle(cropped_rgb,
                              (max(0, local_x1), max(0, local_y1)),
                              (min(cropped_seg.shape[1] - 1, local_x2),
                               min(cropped_seg.shape[0] - 1, local_y2)),
                              (255, 0, 0), 1)

                # Add text
                cv2.putText(cropped_rgb, f"ID: {cell_id}", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Convert to QImage and display in preview
                height, width = cropped_rgb.shape[:2]
                bytes_per_line = 3 * width
                qimage = QImage(cropped_rgb.data, width, height,
                                bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                self.preview_image.setPixmap(pixmap)

            except Exception as e:
                print(f"Error updating preview: {e}")
                self.preview_image.clear()
                self.preview_image.setText("Preview not available")

    def save_ideal_examples(self, dialog):
        """
        Save the selected ideal examples.
        """
        # Update the ideal examples dictionary
        for class_name, combo in self.ideal_selectors.items():
            if combo.isEnabled() and combo.currentText() != "No cells of this class":
                self.ideal_examples[class_name] = int(combo.currentText())

        # Store the metrics for each ideal example
        self.ideal_metrics = {}
        for class_name, cell_id in self.ideal_examples.items():
            if cell_id is not None:
                self.ideal_metrics[class_name] = self.cell_mapping[cell_id]["metrics"]

        # Print the ideal metrics for debugging
        print("Ideal Metrics:")
        for class_name, metrics in self.ideal_metrics.items():
            print(f"{class_name}: {metrics}")

        # Close the dialog
        QMessageBox.information(
            dialog, "Success", "Ideal examples saved successfully.")
        dialog.accept()

    def calculate_similarity_to_ideals(self):
        """
        Calculate how similar each cell is to the ideal examples of each class.
        """
        if not hasattr(self, "ideal_metrics") or not self.ideal_metrics:
            QMessageBox.warning(
                self,
                "Error",
                "No ideal metrics defined. Please select ideal examples first.")
            return

        # Get all cells from current frame
        t = self.slider_t.value()
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        # Make sure cell mapping exists
        if not hasattr(self, "cell_mapping") or not self.cell_mapping:
            QMessageBox.warning(
                self,
                "Error",
                "No cell data available. Please classify cells first.")
            return

        # Metrics to compare (exclude morphology_class)
        metrics_to_compare = [
            "area",
            "perimeter",
            "equivalent_diameter",
            "orientation",
            "aspect_ratio",
            "circularity",
            "solidity"]

        # Store results
        similarity_results = []

        # For each cell, calculate similarity to each ideal
        for cell_id, cell_data in self.cell_mapping.items():
            cell_metrics = cell_data["metrics"]
            current_class = cell_metrics.get("morphology_class", "Unknown")

            cell_result = {
                "cell_id": cell_id,
                "current_class": current_class
            }

            # Calculate similarity to each ideal
            best_similarity = 0
            best_class = None

            for class_name, ideal in self.ideal_metrics.items():
                if not ideal:  # Skip if no ideal defined for this class
                    continue

                # Calculate Euclidean distance in feature space
                # First normalize each feature to prevent any single feature
                # from dominating
                squared_diff_sum = 0
                valid_metrics = 0

                for metric in metrics_to_compare:
                    if metric in cell_metrics and metric in ideal:
                        # Retrieve values
                        cell_value = cell_metrics[metric]
                        ideal_value = ideal[metric]

                        # Skip if either value is None
                        if cell_value is None or ideal_value is None:
                            continue

                        # Normalize based on ideal value to get relative
                        # difference
                        if ideal_value != 0:
                            normalized_diff = (
                                cell_value - ideal_value) / ideal_value
                            squared_diff_sum += normalized_diff ** 2
                            valid_metrics += 1

                # Calculate similarity (invert distance to get similarity)
                if valid_metrics > 0:
                    distance = (squared_diff_sum / valid_metrics) ** 0.5
                    # Convert to similarity (0-1 scale)
                    similarity = 1 / (1 + distance)

                    cell_result[f"similarity_{class_name}"] = similarity

                    # Keep track of best match
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_class = class_name

            # Record best match
            cell_result["best_match_class"] = best_class
            cell_result["best_match_similarity"] = best_similarity

            # Calculate if classification matches best similarity
            cell_result["matches_best"] = (current_class == best_class)

            similarity_results.append(cell_result)

        # Convert to DataFrame for easier analysis
        import pandas as pd
        self.similarity_df = pd.DataFrame(similarity_results)

        # Calculate overall statistics
        total_cells = len(similarity_results)
        matching_cells = sum(
            1 for result in similarity_results if result["matches_best"])
        match_percentage = (matching_cells / total_cells *
                            100) if total_cells > 0 else 0

        # Display results
        self.display_similarity_results(match_percentage)

        return similarity_results

    def display_similarity_results(self, match_percentage):
        """
        Display the similarity analysis results with consistent coloring and ordering.

        Parameters:
        -----------
        match_percentage : float
            The overall match percentage between current and best match classifications.
        """
        # Define a consistent color scheme for all morphology classes
        morphology_colors = {
            "Artifact": "#4CAF50",  # Green
            "Divided": "#FF9800",   # Orange
            "Healthy": "#2196F3",   # Blue
            "Elongated": "#9C27B0",  # Purple
            "Deformed": "#F44336"   # Red
        }

        # Ensure we have the same class order for both pie charts
        ordered_classes = ["Healthy", "Divided",
                           "Artifact", "Elongated", "Deformed"]

        # Clear the existing figure
        self.figure_morphology_metrics.clear()

        # Create a figure with subplots
        gridspec = self.figure_morphology_metrics.add_gridspec(2, 2)

        # 1. Pie chart of current classifications - ensure consistent order
        ax1 = self.figure_morphology_metrics.add_subplot(gridspec[0, 0])
        current_class_counts = self.similarity_df["current_class"].value_counts(
        )

        # Reorder the data to match our predefined order
        current_data = []
        current_labels = []
        current_colors = []

        for class_name in ordered_classes:
            if class_name in current_class_counts:
                current_data.append(current_class_counts[class_name])
                current_labels.append(class_name)
                current_colors.append(morphology_colors[class_name])

        # Create pie chart with consistent colors
        wedges, texts, autotexts = ax1.pie(
            current_data,
            labels=current_labels,
            colors=current_colors,
            autopct='%1.1f%%'
        )
        ax1.set_title("Current Classification Distribution")

        # Style the pie chart text
        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_weight('bold')

        # 2. Pie chart of best match classifications - using same order and
        # colors
        ax2 = self.figure_morphology_metrics.add_subplot(gridspec[0, 1])
        best_counts = self.similarity_df["best_match_class"].value_counts()

        # Reorder the data to match our predefined order
        best_data = []
        best_labels = []
        best_colors = []

        for class_name in ordered_classes:
            if class_name in best_counts:
                best_data.append(best_counts[class_name])
                best_labels.append(class_name)
                best_colors.append(morphology_colors[class_name])

        # Create pie chart with consistent colors
        wedges, texts, autotexts = ax2.pie(
            best_data,
            labels=best_labels,
            colors=best_colors,
            autopct='%1.1f%%'
        )
        ax2.set_title("Best Match Classification Distribution")

        # Style the pie chart text
        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_weight('bold')

        # 3. Bar chart of match percentage by class
        ax3 = self.figure_morphology_metrics.add_subplot(gridspec[1, 0])
        match_by_class = self.similarity_df.groupby(
            "current_class")["matches_best"].mean() * 100

        # Reorder the data to match our predefined order
        ordered_match_data = []
        ordered_match_index = []
        ordered_match_colors = []

        for class_name in ordered_classes:
            if class_name in match_by_class:
                ordered_match_data.append(match_by_class[class_name])
                ordered_match_index.append(class_name)
                ordered_match_colors.append(morphology_colors[class_name])

        # Create reordered series
        match_by_class_ordered = pd.Series(
            ordered_match_data, index=ordered_match_index)

        # Plot with consistent colors
        bars = ax3.bar(
            match_by_class_ordered.index,
            match_by_class_ordered.values,
            color=ordered_match_colors
        )

        ax3.set_title("Match Percentage by Class")
        ax3.set_ylabel("Match Percentage (%)")
        ax3.set_ylim(0, 100)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=8)

        ax3.set_xticklabels(match_by_class_ordered.index,
                            rotation=45, ha='right')

        # 4. Text summary
        ax4 = self.figure_morphology_metrics.add_subplot(gridspec[1, 1])
        ax4.axis('off')

        # Get total cells
        total_cells = len(self.similarity_df)

        # Create formatted summary text
        summary_text = f"""Classification Analysis Summary:

    Total Cells: {total_cells}
    Matching Current to Best: {match_percentage:.1f}%

    Ideal Examples Used:"""

        # Add ideal examples if available
        if hasattr(self, "ideal_examples"):
            for class_name, cell_id in self.ideal_examples.items():
                if cell_id is not None:
                    summary_text += f"\n{class_name}: Cell #{cell_id}"

        ax4.text(0, 0.7, summary_text, va='top', fontsize=10)

        # Adjust layout and draw
        self.figure_morphology_metrics.tight_layout()
        self.canvas_morphology_metrics.draw()

    def optimize_classification_parameters(self):
        """
        Optimize classification thresholds to maximize similarity to ideal examples.
        """
        if not hasattr(self, "ideal_metrics") or not self.ideal_metrics:
            QMessageBox.warning(
                self,
                "Error",
                "No ideal metrics defined. Please select ideal examples first.")
            return

        # Debug: Print original parameters
        print("\n=== ORIGINAL DEFAULT PARAMETERS ===")
        default_params = {
            "artifact_max_area": 245.510,
            "artifact_max_perimeter": 65.901,

            "divided_max_area": 685.844,
            "divided_max_perimeter": 269.150,
            "divided_max_aspect_ratio": 3.531,

            "healthy_min_circularity": 0.516,
            "healthy_max_circularity": 0.727,
            "healthy_min_aspect_ratio": 1.463,
            "healthy_max_aspect_ratio": 3.292,
            "healthy_min_solidity": 0.880,

            "elongated_min_area": 2398.996,
            "elongated_min_aspect_ratio": 5.278,
            "elongated_max_circularity": 0.245,

            "deformed_max_circularity": 0.589,
            "deformed_max_solidity": 0.706
        }

        for key, value in default_params.items():
            print(f"{key}: {value}")

        # Get current classification distribution
        self.original_classification = {}
        for cell_id, cell_data in self.cell_mapping.items():
            if "metrics" in cell_data and "morphology_class" in cell_data["metrics"]:
                self.original_classification[cell_id] = cell_data["metrics"]["morphology_class"]

        # Debug: Print current classification distribution
        class_counts = {}
        for cls in self.original_classification.values():
            class_counts[cls] = class_counts.get(cls, 0) + 1
        print("\n=== CURRENT CLASSIFICATION DISTRIBUTION ===")
        for cls, count in class_counts.items():
            print(f"{cls}: {count} cells")

        # Make sure cell mapping exists
        if not hasattr(self, "cell_mapping") or not self.cell_mapping:
            QMessageBox.warning(
                self,
                "Error",
                "No cell data available. Please classify cells first.")
            return

        # Create progress dialog
        progress = QProgressDialog(
            "Optimizing classification parameters...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        # Number of random parameter sets to try
        num_trials = 100

        # Parameter ranges to search
        param_ranges = {
            # Artifact parameters (new)
            "artifact_max_area": (200, 600),
            "artifact_max_perimeter": (40, 120),

            # Divided cell parameters (formerly small)
            "divided_max_area": (500, 2000),
            "divided_max_perimeter": (50, 300),
            "divided_max_aspect_ratio": (2.0, 4.0),

            # Healthy cell parameters (formerly normal)
            "healthy_min_circularity": (0.4, 0.7),
            "healthy_max_circularity": (0.7, 0.9),
            "healthy_min_aspect_ratio": (1.2, 2.0),
            "healthy_max_aspect_ratio": (2.0, 4.0),
            "healthy_min_solidity": (0.8, 0.95),

            # Elongated cell parameters
            "elongated_min_area": (2000, 4000),
            "elongated_min_aspect_ratio": (4.0, 7.0),
            "elongated_max_circularity": (0.2, 0.5),

            # Deformed cell parameters
            "deformed_max_circularity": (0.4, 0.7),
            "deformed_max_solidity": (0.7, 0.9)
        }

        best_match_percentage = 0
        best_parameters = None
        best_parameter_trial = -1

        import random
        import numpy as np

        # Try multiple random parameter sets
        for trial in range(num_trials):
            # Update progress
            progress.setValue(int((trial / num_trials) * 100))
            if progress.wasCanceled():
                break

            # Generate random parameters within ranges
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                params[param_name] = random.uniform(min_val, max_val)

            # Make sure max > min for paired parameters
            if params["healthy_min_circularity"] > params["healthy_max_circularity"]:
                params["healthy_min_circularity"], params["healthy_max_circularity"] = \
                    params["healthy_max_circularity"], params["healthy_min_circularity"]

            if params["healthy_min_aspect_ratio"] > params["healthy_max_aspect_ratio"]:
                params["healthy_min_aspect_ratio"], params["healthy_max_aspect_ratio"] = \
                    params["healthy_max_aspect_ratio"], params["healthy_min_aspect_ratio"]

            # Debug for first few trials
            if trial < 3:
                print(f"\n=== TRIAL {trial+1} PARAMETERS ===")
                for key, value in params.items():
                    print(f"{key}: {value:.2f}")

            # Test parameters on all cells
            matches = 0
            total_cells = 0
            classification_counts = {
                "Artifact": 0,
                "Divided": 0,
                "Healthy": 0,
                "Elongated": 0,
                "Deformed": 0
            }

            for cell_id, cell_data in self.cell_mapping.items():
                total_cells += 1

                # Get cell metrics
                cell_metrics = cell_data["metrics"]

                # Find which ideal example this cell is closest to
                best_match_class = None
                best_match_similarity = 0

                for class_name, ideal_metrics in self.ideal_metrics.items():
                    if not ideal_metrics:  # Skip if no ideal for this class
                        continue

                    # Calculate similarity to this ideal
                    metrics_to_compare = [
                        "area",
                        "perimeter",
                        "equivalent_diameter",
                        "aspect_ratio",
                        "circularity",
                        "solidity"]

                    squared_diff_sum = 0
                    valid_metrics = 0

                    for metric in metrics_to_compare:
                        if metric in cell_metrics and metric in ideal_metrics:
                            cell_value = cell_metrics[metric]
                            ideal_value = ideal_metrics[metric]

                            if cell_value is not None and ideal_value is not None and ideal_value != 0:
                                normalized_diff = (
                                    cell_value - ideal_value) / ideal_value
                                squared_diff_sum += normalized_diff ** 2
                                valid_metrics += 1

                    if valid_metrics > 0:
                        distance = (squared_diff_sum / valid_metrics) ** 0.5
                        similarity = 1 / (1 + distance)

                        if similarity > best_match_similarity:
                            best_match_similarity = similarity
                            best_match_class = class_name

                # Classify using current parameter set
                classification = classify_morphology(cell_metrics, params)
                classification_counts[classification] = classification_counts.get(
                    classification, 0) + 1

                # Check if classification matches best similarity
                if classification == best_match_class:
                    matches += 1

            # Calculate match percentage
            match_percentage = (matches / total_cells *
                                100) if total_cells > 0 else 0

            # Debug for a few trials
            if trial < 3 or match_percentage > best_match_percentage:
                print(f"\nTrial {trial+1} results:")
                print(f"Match percentage: {match_percentage:.2f}%")
                print("Classification distribution:")
                for cls, count in classification_counts.items():
                    print(f"  {cls}: {count} cells")

            # Update best if this is better
            if match_percentage > best_match_percentage:
                best_match_percentage = match_percentage
                best_parameters = params.copy()
                best_parameter_trial = trial + 1

        # Close progress dialog
        progress.setValue(100)

        # Debug: Print the best parameters
        print("\n=== BEST PARAMETERS (Trial #{}) ===".format(best_parameter_trial))
        if best_parameters:
            for key, value in best_parameters.items():
                print(f"{key}: {value:.3f}")

        # Store best parameters
        self.best_classification_parameters = best_parameters

        # Simulate reclassification to see differences
        new_classifications = {}
        for cell_id, cell_data in self.cell_mapping.items():
            cell_metrics = cell_data["metrics"]
            new_class = classify_morphology(cell_metrics, best_parameters)
            new_classifications[cell_id] = new_class

        # Compare original vs new classifications
        changes = 0
        change_matrix = {}  # From -> To counts
        for cell_id in self.original_classification:
            original = self.original_classification[cell_id]
            new = new_classifications[cell_id]

            if original != new:
                changes += 1
                key = f"{original} -> {new}"
                change_matrix[key] = change_matrix.get(key, 0) + 1

        print(f"\n=== CLASSIFICATION CHANGES ===")
        print(
            f"Total cells that would change: {changes} out of {len(self.original_classification)}")
        print("Change details:")
        for change, count in change_matrix.items():
            print(f"  {change}: {count} cells")

        # Display results with detailed information
        result_message = (
            f"Optimization complete!\n\n"
            f"Best match percentage: {best_match_percentage:.1f}%\n\n"
            f"Changes if applied: {changes} of {len(self.original_classification)} cells\n")

        # Add detail on most significant changes
        if changes > 0:
            result_message += "\nSignificant changes:\n"
            for change, count in sorted(
                    change_matrix.items(), key=lambda x: x[1], reverse=True)[
                    :3]:
                result_message += f"• {change}: {count} cells\n"

        result_message += "\nWould you like to apply these optimized parameters?"

        reply = QMessageBox.question(
            self,
            "Optimization Results",
            result_message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            # Apply the optimized parameters and reclassify cells
            self.apply_optimized_parameters(new_classifications)

        # Return best parameters
        return best_parameters, best_match_percentage

    def apply_optimized_parameters(self, predicted_classifications=None):
        """
        Apply the optimized classification parameters to all cells.

        Args:
            predicted_classifications: Pre-calculated new classifications (optional)
        """
        if not hasattr(self, "best_classification_parameters"):
            QMessageBox.warning(
                self, "Error", "No optimized parameters available.")
            return

        # For tracking changes
        change_count = 0
        old_class_count = {
            "Artifact": 0,
            "Divided": 0,
            "Healthy": 0,
            "Elongated": 0,
            "Deformed": 0
        }
        new_class_count = {
            "Artifact": 0,
            "Divided": 0,
            "Healthy": 0,
            "Elongated": 0,
            "Deformed": 0
        }

        # Reclassify all cells using optimized parameters
        for cell_id, cell_data in self.cell_mapping.items():
            try:
                # Get current class
                old_class = cell_data["metrics"].get(
                    "morphology_class", "Unknown")
                old_class_count[old_class] = old_class_count.get(
                    old_class, 0) + 1

                # Get or calculate new class
                if predicted_classifications and cell_id in predicted_classifications:
                    new_class = predicted_classifications[cell_id]
                else:
                    cell_metrics = cell_data["metrics"]
                    new_class = classify_morphology(
                        cell_metrics, self.best_classification_parameters)

                # Count change if different
                if old_class != new_class:
                    change_count += 1
                    print(f"Cell {cell_id}: {old_class} -> {new_class}")

                # Update classification
                cell_data["metrics"]["morphology_class"] = new_class
                new_class_count[new_class] = new_class_count.get(
                    new_class, 0) + 1

            except Exception as e:
                print(f"Error reclassifying cell {cell_id}: {e}")

        # Print summary of changes
        print("\n=== CLASSIFICATION CHANGES APPLIED ===")
        print(
            f"Changed classification for {change_count} cells out of {len(self.cell_mapping)}")
        print("\nBefore counts:")
        for cls, count in old_class_count.items():
            print(f"  {cls}: {count}")
        print("\nAfter counts:")
        for cls, count in new_class_count.items():
            print(f"  {cls}: {count}")

        # Update the metrics table if it's visible
        if hasattr(self, "populate_metrics_table"):
            print("Updating metrics table...")
            self.populate_metrics_table()

        # Update any visualization
        if hasattr(self, "update_annotation_scatter"):
            print("Updating annotation scatter...")
            self.update_annotation_scatter()

        # Display a detailed before/after comparison
        self.display_classification_results(old_class_count, new_class_count)

        QMessageBox.information(
            self,
            "Reclassification Complete",
            f"All cells have been reclassified using optimized parameters.\n\n"
            f"Changed classification for {change_count} cells out of {len(self.cell_mapping)}.")

    def display_classification_results(self, before_counts, after_counts):
        """
        Display a chart comparing classification before and after optimization.

        Args:
            before_counts: Dictionary with counts before optimization
            after_counts: Dictionary with counts after optimization
        """
        self.figure_morphology_metrics.clear()
        ax = self.figure_morphology_metrics.add_subplot(111)

        # Get all class labels
        all_classes = sorted(
            set(list(before_counts.keys()) + list(after_counts.keys())))

        # Set up for bar chart
        x = range(len(all_classes))
        width = 0.35

        # Create the bars
        before_values = [before_counts.get(cls, 0) for cls in all_classes]
        after_values = [after_counts.get(cls, 0) for cls in all_classes]

        # Plot the bars
        before_bars = ax.bar([i - width / 2 for i in x],
                             before_values, width, label='Before Optimization')
        after_bars = ax.bar([i + width / 2 for i in x],
                            after_values, width, label='After Optimization')

        # Add labels and titles
        ax.set_xlabel('Morphology Class')
        ax.set_ylabel('Cell Count')
        ax.set_title('Cell Classification Before vs After Optimization')
        ax.set_xticks(x)
        ax.set_xticklabels(all_classes)
        ax.legend()

        # Add value labels on the bars
        for i, v in enumerate(before_values):
            ax.text(i - width / 2, v + 0.5, str(v), ha='center')

        for i, v in enumerate(after_values):
            ax.text(i + width / 2, v + 0.5, str(v), ha='center')

        # Highlight differences
        for i, (before, after) in enumerate(zip(before_values, after_values)):
            if before != after:
                diff = after - before
                color = 'green' if diff > 0 else 'red'
                ax.text(i,
                        max(before,
                            after) + 5,
                        f"{'+' if diff > 0 else ''}{diff}",
                        ha='center',
                        color=color,
                        fontweight='bold')

        # Draw the plot
        self.canvas_morphology_metrics.draw()

    def export_metrics_to_csv(self):
        """
        Exports the metrics table data to a CSV file.
        """
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

    def update_annotation_scatter(self):
        try:
            # Extract current frame and segmentation
            t = self.slider_t.value()
            p = self.slider_p.value()
            c = self.slider_c.value() if self.has_channels else None
            frame = self.get_current_frame(t, p, c)

            segmented_image = self.image_data.segmentation_cache[t, p, c]

            # Extract cell metrics
            self.cell_mapping = extract_cells_and_metrics(
                frame, segmented_image)
            self.populate_metrics_table()

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

    def segment_all(self):
        """
        Segments all positions and timesteps directly.
        Uses the channel selected in the UI
        """

        c = self.slider_c.value() if self.has_channels else None

        # Create list to store segmented results
        segmented_results = []
        frame_num = self.image_data.data.shape[0]
        position_num = self.dimensions.get("P", 1)

        for t in tqdm(range(frame_num), desc="Time"):
            for p in tqdm(range(position_num), desc="Position"):
                self.image_data.segmentation_cache.with_model(
                    self.model_dropdown.currentText())  # Setting the model we want
                segmented = self.image_data.segmentation_cache[t, p, c]

    def segment_this_p(self):
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        # Create list to store segmented results
        segmented_results = []
        total_frames = self.image_data.data.shape[0]

        for t in tqdm(range(total_frames), desc="Segmenting frames"):
            self.image_data.segmentation_cache.with_model(
                self.model_dropdown.currentText())  # Setting the model we want
            segmented = self.image_data.segmentation_cache[t, p, c]

            # Label segmented objects (Assign unique label to each object)
            labeled_cells = label(segmented)

            # Visualize labeled segmentation
            # plt.figure(figsize=(5, 5))
            # # Color-coded labels
            # plt.imshow(labeled_cells, cmap='nipy_spectral')
            # plt.title(f'Labeled Segmentation - Frame {t}')
            # plt.axis('off')
            # plt.show()

            segmented_results.append(labeled_cells)

        # Convert list to numpy array
        self.segmented_time_series = np.array(segmented_results)
        # Set to false when new segmentation is needed
        self.is_time_series_segmented = True

        QMessageBox.information(
            self, "Segmentation Complete",
            f"Segmentation for all time points is complete. Shape: {self.segmented_time_series.shape}"
        )

    def get_current_frame(self, t, p, c=None):
        """
        Retrieve the current frame based on slider values for time, position, and channel.
        """
        if self.image_data.is_nd2:
            if self.has_channels:
                frame = self.image_data.data[t, p, c]
            else:
                frame = self.image_data.data[t, p]
        else:
            frame = self.image_data.data[t]

        # Convert to NumPy array if needed
        return np.array(frame)

    def annotate_scatter_points(self, ax, scatter, pca_df):
        """
        Adds interactive hover annotations and click event to highlight a selected cell.

        Parameters:
        ax : matplotlib.axes.Axes
            The axes object for the scatter plot.
        scatter : matplotlib.collections.PathCollection
            The scatter plot object.
        pca_df : pd.DataFrame
            DataFrame containing PCA results with cell IDs and classes.
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

    def on_table_item_click(self, item):
        row = item.row()

        cell_id = self.metrics_table.item(row, 0).text()

        # print(f"Row {row} clicked, Cell ID: {cell_id}")

        self.highlight_cell_in_image(cell_id)

    def highlight_cell_in_image(self, cell_id):
        # print(f"🔍 Highlighting cell with ID: {cell_id}")

        t = self.slider_t.value()
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        # Get the binary segmentation
        segmented_image = self.image_data.segmentation_cache[t, p, c]

        if segmented_image is None:
            QMessageBox.warning(self, "Error", "Segmented image not found.")
            return

        # Debug info
        unique_labels = np.unique(segmented_image)
        # print(f"🔍 Unique labels in segmented image: {unique_labels}")

        # Ensure cell ID is an integer
        cell_id = int(cell_id)

        # Ensure stored cell mappings exist
        if not hasattr(self, "cell_mapping") or not self.cell_mapping:
            QMessageBox.warning(
                self,
                "Error",
                "No stored cell mappings found. Did you classify cells first?")
            return

        available_ids = list(map(int, self.cell_mapping.keys()))
        # print(f"📝 Available Segmentation Cell IDs: {available_ids}")

        if cell_id not in available_ids:
            QMessageBox.warning(
                self,
                "Error",
                f"Cell ID {cell_id} not found in segmentation. Available IDs: {available_ids}")
            return

        # Get the bounding box coordinates for the selected cell
        y1, x1, y2, x2 = self.cell_mapping[cell_id]["bbox"]

        # Create a visualization of the segmented image
        # Convert binary segmentation to RGB for visualization
        segmented_rgb = cv2.cvtColor((segmented_image > 0).astype(
            np.uint8) * 255, cv2.COLOR_GRAY2BGR)

        # Create a mask for just this cell based on the bounding box
        cell_mask = np.zeros_like(segmented_image, dtype=np.uint8)

        # Extract the region of interest from the segmentation
        roi = segmented_image[y1:y2, x1:x2]

        # If there are cells in the ROI, isolate the main one
        if roi.max() > 0:
            # Use connected components to find distinct objects in the ROI
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                roi, connectivity=8)

            # Find the largest component (excluding background)
            largest_label = 1  # Default to first label
            largest_area = 0

            for label in range(1, num_labels):  # Skip background (0)
                area = stats[label, cv2.CC_STAT_AREA]
                if area > largest_area:
                    largest_area = area
                    largest_label = label

            # Create mask for the largest component
            roi_mask = (labels == largest_label).astype(np.uint8) * 255

            # Place the ROI mask back in the full image mask
            cell_mask[y1:y2, x1:x2] = roi_mask

        # Highlight the cell in red on the segmented image
        segmented_rgb[cell_mask > 0] = [0, 0, 255]  # BGR format - Red

        # Also draw the bounding box in blue
        cv2.rectangle(segmented_rgb, (x1, y1), (x2, y2),
                      (255, 0, 0), 1)  # Blue rectangle

        # Add cell ID text
        cv2.putText(segmented_rgb, str(cell_id), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Green text

        # Convert to QImage and display
        height, width = segmented_rgb.shape[:2]
        bytes_per_line = 3 * width

        qimage = QImage(segmented_rgb.data, width, height,
                        bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

        # print(
        # f"✅ Successfully highlighted cell {cell_id} at bounding box {(y1, x1,
        # y2, x2)}")

    def highlight_cell(self, cell_id):
        """Highlight a specific cell when clicked on PCA plot"""
        print(f"App: Highlighting cell {cell_id}")

        # Ensure cell ID is an integer
        cell_id = int(cell_id)

        # Check if we have the cell mapping
        if not hasattr(self, "current_cell_mapping") or cell_id not in self.current_cell_mapping:
            print(f"Cell {cell_id} not found in current mapping")
            return

        # Get current frame parameters from ViewAreaWidget
        t = self.viewArea.current_t
        p = self.viewArea.current_p
        c = self.viewArea.current_c

        try:
            # Get the segmentation
            segmented_image = self.image_data.segmentation_cache[t, p, c]

            if segmented_image is None:
                print(f"No segmentation available for highlighting")
                return

            # Create single-cell mapping for the selected cell
            single_cell_mapping = {cell_id: self.current_cell_mapping[cell_id]}

            # Import the morphology function
            from nd2_analyzer.analysis.morphology.morphology import annotate_binary_mask

            # Create highlighted image with just the one cell
            highlighted_image = annotate_binary_mask(
                segmented_image, single_cell_mapping)

            # Display on the view area's image label
            height, width = highlighted_image.shape[:2]
            qimage = QImage(highlighted_image.data, width, height,
                            highlighted_image.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage).scaled(
                self.viewArea.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.viewArea.image_label.setPixmap(pixmap)

        except Exception as e:
            import traceback
            print(f"Error highlighting cell: {e}")
            traceback.print_exc()

    # def highlight_selected_cell(self, cell_id, cache_key):
    #     """
    #     Highlights a selected cell on the segmented image when a point on the scatter plot is clicked.

    #     Parameters:
    #     -----------
    #     cell_id : int
    #         ID of the cell to highlight.
    #     cache_key : tuple
    #         Key to retrieve cached segmentation and cell mapping.
    #     """
    #     # Ensure cell mapping exists
    #     if not hasattr(self, "cell_mapping") or not self.cell_mapping:
    #         QMessageBox.warning(self, "Error", "Cell mapping not initialized.")
    #         return

    #     # Retrieve bounding box for the selected cell
    #     if cell_id not in self.cell_mapping:
    #         QMessageBox.warning(self, "Error", f"Cell ID {cell_id} not found.")
    #         return

    #     bbox = self.cell_mapping[cell_id]["bbox"]
    #     y1, x1, y2, x2 = bbox  # Correct order (y1, x1, y2, x2)

    #     # Create a copy of the annotated image to avoid overwriting
    #     highlighted_image = self.annotated_image.copy()
    #     cv2.rectangle(highlighted_image, (x1, y1), (x2, y2),
    #                   (255, 0, 0), 2)  # Highlight with red box

    #     # Display the highlighted image in the scatter plot tab
    #     height, width = highlighted_image.shape[:2]
    #     qimage = QImage(
    #         highlighted_image.data,
    #         width,
    #         height,
    #         highlighted_image.strides[0],
    #         QImage.Format_RGB888,
    #     )
    #     pixmap = QPixmap.fromImage(qimage).scaled(
    #         self.annotated_image_label.size(),
    #         Qt.KeepAspectRatio,
    #         Qt.SmoothTransformation,
    #     )
    #     self.annotated_image_label.setPixmap(pixmap)

    def generate_morphology_data(self):
        # Generate morphological data for the annotated tab
        t = self.slider_t.value()
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        # Get segmented image
        segmented_image = self.get_segmented_data(t, p, c)

        # Extract morphology
        self.morphology_data = extract_cell_morphologies(segmented_image)

        # Automatically plot default metrics
        self.update_annotation_scatter()

    # TODO: remove
    def generate_annotations_and_scatter(self):
        t = self.slider_t.value()
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        # Extract the current frame
        image_data = self.image_data.data
        if self.image_data.is_nd2:
            if self.has_channels:
                image_data = image_data[t, p, c]
            else:
                image_data = image_data[t, p]
        else:
            image_data = image_data[t]

        # Ensure image_data is a numpy array
        image_data = np.array(image_data)

        # Perform segmentation
        segmented_image = segment_this_image(image_data)

        # Extract cells and their metrics
        self.cell_mapping = extract_cells_and_metrics(
            image_data, segmented_image)

        if not self.cell_mapping:
            QMessageBox.warning(
                self, "No Cells", "No cells detected in the current frame."
            )
            return

        # Populate the metrics table
        self.populate_metrics_table()

        # Annotate the original image
        try:
            annotated_image = annotate_image(image_data, self.cell_mapping)
        except ValueError as e:
            print(f"Annotation Error: {e}")
            QMessageBox.warning(self, "Annotation Error", str(e))
            return

        # Display the annotated image
        height, width = annotated_image.shape[:2]
        qimage = QImage(
            annotated_image.data,
            width,
            height,
            annotated_image.strides[0],
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.annotated_image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.annotated_image_label.setPixmap(pixmap)

        # Generate scatter plot
        self.generate_scatter_plot()

    def populate_metrics_table(self):
        if not self.cell_mapping:
            QMessageBox.warning(
                self, "Error", "No cell data available for metrics table.")
            return

        # Convert cell mapping to a DataFrame
        metrics_data = [
            {**{"ID": cell_id}, **data["metrics"]}
            for cell_id, data in self.cell_mapping.items()
        ]
        metrics_df = pd.DataFrame(metrics_data)

        # Round numerical columns to 2 decimal places
        numeric_columns = [
            'area',
            'perimeter',
            'equivalent_diameter',
            'orientation',
            'aspect_ratio',
            'circularity',
            'solidity']  # Adjust based on actual column names
        metrics_df[numeric_columns] = metrics_df[numeric_columns].round(2)

        # Update QTableWidget
        self.metrics_table.setRowCount(metrics_df.shape[0])
        self.metrics_table.setColumnCount(metrics_df.shape[1])
        self.metrics_table.setHorizontalHeaderLabels(metrics_df.columns)

        for row in range(metrics_df.shape[0]):
            for col, value in enumerate(metrics_df.iloc[row]):
                self.metrics_table.setItem(
                    row, col, QTableWidgetItem(str(value)))

    def generate_scatter_plot(self):
        areas = [data["metrics"]["area"]
                 for data in self.cell_mapping.values()]
        perimeters = [
            data["metrics"]["perimeter"] for data in self.cell_mapping.values()
        ]
        ids = list(self.cell_mapping.keys())

        self.figure_scatter_plot.clear()
        ax = self.figure_scatter_plot.add_subplot(111)

        # Create scatter plot with interactivity
        scatter = ax.scatter(
            areas,
            perimeters,
            c=areas,
            cmap="viridis",
            picker=True)
        ax.set_title("Area vs Perimeter")
        ax.set_xlabel("Area")
        ax.set_ylabel("Perimeter")

        # Annotate scatter points with IDs
        for i, txt in enumerate(ids):
            ax.annotate(txt, (areas[i], perimeters[i]))

        # Add click event handling
        self.figure_scatter_plot.canvas.mpl_connect(
            "pick_event", lambda event: self.on_scatter_click(event)
        )

        self.canvas_scatter_plot.draw()

    def on_scatter_click(self, event):
        # Get the index of the clicked point
        ind = event.ind[0]  # Index of the clicked point
        cell_id = list(self.cell_mapping.keys())[ind]

        print(f"Clicked on scatter point: ID {cell_id}")

        # Extract the specific image frame
        t = self.slider_t.value()
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        if self.image_data.is_nd2:
            if self.has_channels:
                image_data = self.image_data.data[t, p, c]
            else:
                image_data = self.image_data.data[t, p]
        else:
            image_data = self.image_data.data[t]

        image_data = np.array(image_data)  # Ensure it's a NumPy array

        # Highlight the corresponding cell in the annotated image
        annotated_image = annotate_image(
            image_data, {cell_id: self.cell_mapping[cell_id]}
        )

        # Update the annotated image display
        height, width = annotated_image.shape[:2]
        qimage = QImage(
            annotated_image.data,
            width,
            height,
            annotated_image.strides[0],
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.annotated_image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.annotated_image_label.setPixmap(pixmap)
