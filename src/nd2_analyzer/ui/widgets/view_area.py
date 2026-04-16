import cv2
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QSizePolicy,
    QComboBox,
    QCheckBox,
)
from pubsub import pub
from skimage import exposure

from nd2_analyzer.analysis.segmentation.segmentation_models import SegmentationModels
from nd2_analyzer.data.image_data import ImageData


class ViewAreaWidget(QWidget):
    """
    A standalone widget for viewing and controlling ND2 image data with segmentation options.
    Uses PyPubSub for communication with other components.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize state variables
        self.current_t = 0
        self.current_p = 0
        self.current_c = 0
        self.current_mode = "normal"  # normal, segmented, or labeled
        self.current_model = None
        self.valid_time_frames = None  # List of valid frame indices (excluding focus loss)

        # Biofilm mode: Colony overlay support
        self.colony_overlay = None
        self.show_colony_overlay = False

        # Biofilm mode: Manual colony selection support
        self.manual_selection_active = False
        self.current_polygon_points = []
        self.colony_separator = None
        self.temp_polygon_overlay = None

        # Set up the UI
        self.init_ui()

        # Tracking state
        self.tracked_cell_lineage = {}  # Maps frame -> [track_ids]
        self.lineage_tracks = None  # Full tracking data

        # Subscribe to relevant topics
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")
        pub.subscribe(self.on_image_ready, "image_ready")
        pub.subscribe(self.highlight_cell, "highlight_cell_requested")
        pub.subscribe(self.provide_current_param, "get_current_t")
        pub.subscribe(self.provide_current_param, "get_current_p")
        pub.subscribe(self.provide_current_param, "get_current_c")
        pub.subscribe(self.on_segmentation_cache_miss, "segmentation_cache_miss")
        pub.subscribe(self.on_experiment_loaded, "experiment_loaded")
        pub.subscribe(self.on_cell_tracking_ready, "cell_tracking_ready")

        # Biofilm mode subscriptions
        pub.subscribe(self.on_show_colony_overlay, "show_colony_overlay")
        pub.subscribe(self.on_hide_colony_overlay, "hide_colony_overlay")
        pub.subscribe(self.enable_manual_selection, "enable_manual_colony_selection")
        pub.subscribe(self.disable_manual_selection, "disable_manual_colony_selection")

    # FOCUS LOSS HANDLING

    def on_experiment_loaded(self, experiment):
        """Handle experiment loading and update valid frames based on focus loss intervals"""
        self._update_valid_frames(experiment)

    def _update_valid_frames(self, experiment):
        """Calculate which frames are valid (not in focus loss intervals)"""
        from nd2_analyzer.data.appstate import ApplicationState

        if experiment is None or not hasattr(experiment, 'focus_loss_intervals'):
            self.valid_time_frames = None
            return

        # Get total number of frames from image data
        image_data = ImageData.get_instance()
        if not image_data or not hasattr(image_data, 'data'):
            self.valid_time_frames = None
            return

        total_frames = image_data.data.shape[0]

        # If no focus loss intervals, all frames are valid
        if not experiment.focus_loss_intervals:
            self.valid_time_frames = None
            return

        # Convert focus loss intervals (in hours) to frame indices
        time_interval_hours = experiment.time_interval_hours
        focus_loss_frames = []

        print(f"Time interval per frame: {time_interval_hours:.4f} hours")
        print(f"Total frames in dataset: {total_frames}")

        for start_hours, end_hours in experiment.focus_loss_intervals:
            start_frame_calc = start_hours / time_interval_hours
            end_frame_calc = end_hours / time_interval_hours

            start_frame = int(start_frame_calc)
            end_frame = int(end_frame_calc)

            print(f"  Raw calculation: {start_hours:.2f}h/{time_interval_hours:.4f} = {start_frame_calc:.2f} → frame {start_frame}")
            print(f"  Raw calculation: {end_hours:.2f}h/{time_interval_hours:.4f} = {end_frame_calc:.2f} → frame {end_frame}")

            # Clamp to actual data range
            start_frame_clamped = max(0, min(start_frame, total_frames - 1))
            end_frame_clamped = max(0, min(end_frame, total_frames - 1))

            if start_frame_clamped <= end_frame_clamped:
                focus_loss_frames.extend(range(start_frame_clamped, end_frame_clamped + 1))
                print(f"  Focus loss interval: {start_hours:.2f}h - {end_hours:.2f}h → frames {start_frame_clamped}-{end_frame_clamped}")
            else:
                print(f"  WARNING: Invalid interval after clamping: {start_frame_clamped}-{end_frame_clamped}")

        # Create list of valid frames (excluding focus loss)
        focus_loss_set = set(focus_loss_frames)
        self.valid_time_frames = [f for f in range(total_frames) if f not in focus_loss_set]

        excluded_count = len(focus_loss_set)
        valid_count = len(self.valid_time_frames)

        print(f"Focus loss filtering: {excluded_count} frames excluded, {valid_count} valid frames remaining (out of {total_frames} total)")

        if valid_count == 0:
            print("WARNING: All frames are marked as focus loss! Reverting to showing all frames.")
            self.valid_time_frames = None

    def _map_slider_to_frame(self, slider_value):
        """Convert slider position to actual frame index (skipping focus loss frames)"""
        if self.valid_time_frames is None:
            return slider_value

        if slider_value >= len(self.valid_time_frames):
            return self.valid_time_frames[-1] if self.valid_time_frames else 0

        return self.valid_time_frames[slider_value]

    def _map_frame_to_slider(self, frame_index):
        """Convert frame index to slider position (accounting for focus loss frames)"""
        if self.valid_time_frames is None:
            return frame_index

        try:
            return self.valid_time_frames.index(frame_index)
        except ValueError:
            # Frame is in focus loss interval, return nearest valid frame
            for i, valid_frame in enumerate(self.valid_time_frames):
                if valid_frame > frame_index:
                    return max(0, i - 1)
            return len(self.valid_time_frames) - 1

    # CENTRALIZED IMAGE PROCESSING FUNCTIONS

    def _normalize_image(self, image):
        """
        Normalize image to appropriate display range based on dtype.

        Args:
            image: Input image array

        Returns:
            Normalized image with appropriate dtype
        """
        if image.dtype == np.uint16:
            # Normalize 16-bit to full range
            return exposure.rescale_intensity(image, out_range=(0, 65535)).astype(
                np.uint16
            )
        elif image.dtype == np.uint8:
            # Already in proper range for 8-bit
            return image
        else:
            # Float or other types - normalize to 8-bit range
            return exposure.rescale_intensity(image, out_range="uint8").astype(np.uint8)

    def _prepare_image_for_display(self, image, normalize=True):
        """
        Standardize image format for Qt display.

        Args:
            image: Input image array
            normalize: Whether to apply normalization

        Returns:
            tuple: (processed_image, width, height, bytes_per_line, qt_format)
        """
        # Work on copy to avoid modifying original
        img = image.copy()

        # Apply normalization if requested
        if normalize:
            img = self._normalize_image(img)

        # Handle grayscale images
        if len(img.shape) == 2:
            # Ensure grayscale is 16-bit for better display quality
            if img.dtype != np.uint16:
                if normalize:
                    img = self._normalize_image(img).astype(np.uint16)
                else:
                    # Scale without normalization
                    if img.max() <= 255:
                        img = (img.astype(np.float32) * 257).astype(np.uint16)
                    else:
                        img = img.astype(np.uint16)

            height, width = img.shape
            bytes_per_line = width * 2  # 2 bytes per 16-bit pixel
            qt_format = QImage.Format_Grayscale16

        # Handle color images
        else:
            # Ensure color images are 8-bit RGB
            if img.dtype != np.uint8:
                if normalize:
                    img = self._normalize_image(img)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)

            # Convert grayscale to RGB if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA to RGB
                img = img[:, :, :3]

            height, width, channels = img.shape
            bytes_per_line = width * channels
            qt_format = QImage.Format_RGB888

        return img, width, height, bytes_per_line, qt_format

    def _convert_segmentation_to_display(self, image, mode):
        """
        Convert segmentation results to display format.

        Args:
            image: Segmentation result
            mode: Display mode ('segmented', 'overlay', 'labeled')

        Returns:
            RGB image ready for display
        """
        img = image.copy()

        if mode == "segmented":
            # Binary mask - convert to grayscale
            if len(img.shape) > 2:
                img = (
                    cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    if img.shape[2] == 3
                    else img[:, :, 0]
                )

            # Ensure proper range
            if img.dtype != np.uint8:
                img = self._normalize_image(img)

            return img

        elif mode in ["overlay", "labeled"]:
            # Should be RGB color image
            if len(img.shape) == 2:
                # Convert grayscale to RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA to RGB
                img = img[:, :, :3]

            # Ensure uint8
            if img.dtype != np.uint8:
                img = self._normalize_image(img)

            return img

        return img

    def _create_qimage_and_display(self, image):
        """
        Create QImage from processed array and display it.

        Args:
            image: Processed image array ready for display
        """
        normalize = self.normalize_checkbox.isChecked()
        processed_img, width, height, bytes_per_line, qt_format = (
            self._prepare_image_for_display(image, normalize)
        )

        # Create QImage
        q_image = QImage(processed_img.data, width, height, bytes_per_line, qt_format)

        # Scale and display
        pixmap = QPixmap.fromImage(q_image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

        # Store for highlighting
        self.current_image_data = image.copy()

    # SIMPLIFIED EVENT HANDLERS

    def on_image_ready(self, image, time, position, channel, mode):
        """Handle both raw and segmented image responses"""
        # Check if this is the image we're currently expecting
        if (
            time != self.current_t
            or position != self.current_p
            or channel != self.current_c
            or mode != self.current_mode
        ):
            return  # Ignore outdated images

        # Store current image for overlays
        self.current_image = image

        # Send data to segmentation widget for exporting
        if mode in ["segmented", "labeled"]:
            pub.sendMessage(
                "segmentation_result_available",
                image=image,  # IMPORTANT: raw, unmodified
                time=time,
                position=position,
                channel=channel,
                model=self.current_model,
            )

        # Process based on mode
        if mode == "normal":
            display_image = image
        else:
            display_image = self._convert_segmentation_to_display(image, mode)

        # Apply tracked cell highlighting if any (LIKE UPSTREAM!)
        display_image = self._apply_tracked_cell_highlighting(
            display_image, time, position, channel, mode
        )

        # Display the processed image
        self._create_qimage_and_display(display_image)

    def on_cell_tracking_ready(self, cell_id, lineage_data):
        """Receive tracking data from MorphologyWidget"""
        print(
            f"ViewAreaWidget: Received tracking data for cell {cell_id} across {len(lineage_data)} frames"
        )
        self.tracked_cell_lineage = lineage_data

        # Get lineage_tracks from tracking widget via pub/sub
        def receive_tracking_data(lineage_tracks):
            self.lineage_tracks = lineage_tracks
            print(f"ViewAreaWidget: Received {len(lineage_tracks)} lineage tracks")

        pub.sendMessage("get_lineage_tracks", callback=receive_tracking_data)

        # Refresh display to show tracking
        self.on_slider_changed()

    def _apply_tracked_cell_highlighting(self, display_image, time, position, channel, mode):
        """Apply tracked cell highlighting - exact copy of upstream logic"""
        # Only highlight if we have tracking data
        if not hasattr(self, "tracked_cell_lineage") or not self.tracked_cell_lineage:
            return display_image

        # Check if current frame has cells to highlight
        if time not in self.tracked_cell_lineage:
            return display_image

        tracked_ids = self.tracked_cell_lineage[time]

        print(f"\n{'='*60}")
        print(f"TRACKING HIGHLIGHT DEBUG - Frame T={time}, P={position}, C={channel}")
        print(f"{'='*60}")
        print(f"Tracked cell IDs for this frame: {tracked_ids}")
        print(f"Current display mode: {mode}")

        # IMPORTANT: Get segmented image directly from cache (ROI/registration already applied during segmentation)
        # The segmentation cache contains the already-processed masks
        from nd2_analyzer.data.image_data import ImageData

        image_data = ImageData.get_instance()
        if not hasattr(image_data, "segmentation_cache"):
            print(f"❌ No segmentation cache available")
            return display_image

        # Get from cache with the current model
        try:
            segmented = image_data.segmentation_cache.with_model(self.current_model)[time, position, channel]
        except Exception as e:
            print(f"❌ Error getting segmentation from cache: {e}")
            segmented = None

        if segmented is None:
            print(f"❌ No segmentation available for tracking highlight")
            return display_image

        # COMPREHENSIVE LOGGING
        print(f"\n📊 SEGMENTATION ANALYSIS:")
        print(f"  Segmentation shape: {segmented.shape}")
        print(f"  Segmentation dtype: {segmented.dtype}")
        print(f"  Segmentation min/max: {segmented.min()}/{segmented.max()}")
        print(f"  Number of unique cell IDs: {len(np.unique(segmented))}")
        print(f"  Total pixels marked as cells: {np.sum(segmented > 0)}")

        # Create a color version of the segmented image
        import cv2
        from skimage.measure import label, regionprops

        # Convert to RGB for display
        segmented_rgb = cv2.cvtColor(
            (segmented > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR
        )

        # Check if segmentation is already labeled or binary
        print(f"\n🏷️  LABELING CHECK:")
        max_value = segmented.max()
        unique_values = len(np.unique(segmented))
        print(f"  Max value: {max_value}, Unique values: {unique_values}")

        # If already labeled (OmniPose/Cellpose), use as-is
        # If binary (UNET), call label()
        if max_value > 255 or unique_values > 100:
            print(f"  ✓ Already labeled! Using cell IDs directly (no renumbering)")
            labeled = segmented
        else:
            print(f"  ✓ Binary mask, calling label() to number cells...")
            labeled = label(segmented)

        num_regions = labeled.max()
        print(f"  Total labeled regions: {num_regions}")

        # Get positions for tracked cells from tracking data
        if not hasattr(self, "lineage_tracks") or not self.lineage_tracks:
            print("No lineage_tracks available for position lookup")
            return display_image

        print(f"\n🎯 POSITION LOOKUP & HIGHLIGHTING:")
        for cell_id in tracked_ids:
            cell_position = None
            print(f"\n  [Cell {cell_id}] Looking for track in {len(self.lineage_tracks)} tracks...")

            # Find this cell's position in tracking data
            for track in self.lineage_tracks:
                if track["ID"] == cell_id:
                    print(f"  [Cell {cell_id}] ✓ Found track")
                    if "t" in track and "x" in track and "y" in track:
                        print(f"  [Cell {cell_id}] Track time range: {track['t'][0]} to {track['t'][-1]}")
                        print(f"  [Cell {cell_id}] Looking for time {time} in track...")
                        for i, t in enumerate(track["t"]):
                            if t == time:
                                if i < len(track["x"]) and i < len(track["y"]):
                                    cell_position = (int(track["x"][i]), int(track["y"][i]))
                                    print(f"  [Cell {cell_id}] ✅ Found position: ({cell_position[0]}, {cell_position[1]})")
                                    print(f"  [Cell {cell_id}] This is index {i} in the track")
                                    break
                        if not cell_position:
                            print(f"  [Cell {cell_id}] ❌ Time {time} not in track times")
                    else:
                        print(f"  [Cell {cell_id}] ❌ Track missing position data")
                    break

            if cell_position:
                x, y = cell_position
                print(f"  [Cell {cell_id}] 📍 Checking labeled image at position ({x}, {y})")
                print(f"  [Cell {cell_id}] Labeled image bounds: width={labeled.shape[1]}, height={labeled.shape[0]}")

                # Find the region in the labeled image
                if 0 <= y < labeled.shape[0] and 0 <= x < labeled.shape[1]:
                    cell_label = labeled[y, x]
                    print(f"  [Cell {cell_id}] Label at position: {cell_label}")

                    if cell_label > 0:
                        # Get mask for this specific cell
                        cell_mask = labeled == cell_label
                        num_pixels = np.sum(cell_mask)
                        print(f"  [Cell {cell_id}] Cell mask has {num_pixels} pixels")

                        # Get region props to verify
                        region_props = regionprops(cell_mask.astype(np.uint8))
                        if region_props:
                            region = region_props[0]
                            y1, x1, y2, x2 = region.bbox
                            centroid_y, centroid_x = region.centroid
                            print(f"  [Cell {cell_id}] Region bbox: ({x1}, {y1}) to ({x2}, {y2})")
                            print(f"  [Cell {cell_id}] Region centroid: ({int(centroid_x)}, {int(centroid_y)})")
                            print(f"  [Cell {cell_id}] Region area: {region.area} pixels")
                            print(f"  [Cell {cell_id}] Track position vs Centroid: ({x}, {y}) vs ({int(centroid_x)}, {int(centroid_y)})")
                            distance = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
                            print(f"  [Cell {cell_id}] Distance from track pos to centroid: {distance:.2f} pixels")

                        # Color the cell blue (BGR format)
                        segmented_rgb[cell_mask] = [255, 0, 0]

                        # Get bounding box for this cell
                        if region_props:
                            # Draw bounding box in green
                            cv2.rectangle(segmented_rgb, (x1, y1), (x2, y2), (0, 255, 0), 1)
                            print(f"  [Cell {cell_id}] ✅ Highlighted successfully!")

                        # Add cell ID text in green
                        cv2.putText(
                            segmented_rgb,
                            str(cell_id),
                            (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                        )
                    else:
                        print(f"    ⚠️ No cell found at position ({x}, {y}) - label is 0")
                else:
                    print(f"    ❌ Position ({x}, {y}) is out of bounds for image shape {labeled.shape}")

        return segmented_rgb

    def on_segmentation_cache_miss(self, time, position, channel, model):
        """Handle when cached segmentation is not available"""
        if (
            time == self.current_t
            and position == self.current_p
            and channel == self.current_c
        ):
            print(
                f"No cached segmentation found for T={time}, P={position}, C={channel}"
            )
            pub.sendMessage(
                "segmented_image_request",
                time=time,
                position=position,
                channel=channel,
                mode=self.current_mode,
                model=self.current_model,
                use_cached=False,
            )

    def provide_current_param(self, topic=pub.AUTO_TOPIC, default=0):
        param_map = {
            "get_current_t": self.current_t,
            "get_current_p": self.current_p,
            "get_current_c": self.current_c,
        }
        topic_name = topic.getName()
        value = param_map.get(topic_name, default)
        print(f"ViewAreaWidget: Providing {topic_name}={value}")
        return value

    # UI INITIALIZATION (keeping existing code structure)
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

        # Image display area
        self.image_label = QLabel()
        self.image_label.setScaledContents(True)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.image_label)

        # ND2 Controls Group
        nd2_controls_group = QGroupBox("ND2 Controls")
        nd2_controls_layout = QVBoxLayout()

        # T controls (Time)
        t_layout = QHBoxLayout()
        self.t_label = QLabel("T: 0")
        t_layout.addWidget(self.t_label)

        self.t_left_button = QPushButton("<")
        self.t_left_button.clicked.connect(
            lambda: self.slider_t.setValue(self.slider_t.value() - 1)
        )
        t_layout.addWidget(self.t_left_button)

        self.slider_t = QSlider(Qt.Horizontal)
        self.slider_t.setMinimum(0)
        self.slider_t.setMaximum(0)
        self.slider_t.valueChanged.connect(self.on_slider_changed)
        # Note: t_label is updated in on_slider_changed to show actual frame number
        t_layout.addWidget(self.slider_t)

        self.t_right_button = QPushButton(">")
        self.t_right_button.clicked.connect(
            lambda: self.slider_t.setValue(self.slider_t.value() + 1)
        )
        t_layout.addWidget(self.t_right_button)

        nd2_controls_layout.addLayout(t_layout)

        # P controls (Position)
        p_layout = QHBoxLayout()
        self.p_label = QLabel("P: 0")
        p_layout.addWidget(self.p_label)

        self.p_left_button = QPushButton("<")
        self.p_left_button.clicked.connect(
            lambda: self.slider_p.setValue(self.slider_p.value() - 1)
        )
        p_layout.addWidget(self.p_left_button)

        self.slider_p = QSlider(Qt.Horizontal)
        self.slider_p.setMinimum(0)
        self.slider_p.setMaximum(0)
        self.slider_p.valueChanged.connect(self.on_slider_changed)
        self.slider_p.valueChanged.connect(
            lambda value: self.p_label.setText(f"P: {value}")
        )
        p_layout.addWidget(self.slider_p)

        self.p_right_button = QPushButton(">")
        self.p_right_button.clicked.connect(
            lambda: self.slider_p.setValue(self.slider_p.value() + 1)
        )
        p_layout.addWidget(self.p_right_button)

        nd2_controls_layout.addLayout(p_layout)

        # C controls (Channel)
        c_layout = QHBoxLayout()
        self.c_label = QLabel("C: 0")
        c_layout.addWidget(self.c_label)

        self.c_left_button = QPushButton("<")
        self.c_left_button.clicked.connect(
            lambda: self.slider_c.setValue(self.slider_c.value() - 1)
        )
        c_layout.addWidget(self.c_left_button)

        self.slider_c = QSlider(Qt.Horizontal)
        self.slider_c.setMinimum(0)
        self.slider_c.setMaximum(0)
        self.slider_c.valueChanged.connect(self.on_slider_changed)
        self.slider_c.valueChanged.connect(
            lambda value: self.c_label.setText(f"C: {value}")
        )
        c_layout.addWidget(self.slider_c)

        self.c_right_button = QPushButton(">")
        self.c_right_button.clicked.connect(
            lambda: self.slider_c.setValue(self.slider_c.value() + 1)
        )
        c_layout.addWidget(self.c_right_button)
        nd2_controls_layout.addLayout(c_layout)

        # Add normalization checkbox
        self.normalize_checkbox = QCheckBox("Normalize Image")
        self.normalize_checkbox.setChecked(True)
        self.normalize_checkbox.stateChanged.connect(self.on_slider_changed)
        nd2_controls_layout.addWidget(self.normalize_checkbox)

        nd2_controls_group.setLayout(nd2_controls_layout)
        layout.addWidget(nd2_controls_group)

        # Display Mode Controls
        display_mode_group = QGroupBox("Display Mode")
        display_mode_layout = QVBoxLayout()

        self.radio_normal = QRadioButton("Normal")
        self.radio_segmented = QRadioButton("Segmented")
        self.radio_overlay_outlines = QRadioButton("Overlay with Outlines")
        self.radio_labeled_segmentation = QRadioButton("Labeled Segmentation")

        self.button_group = QButtonGroup()
        self.button_group.addButton(self.radio_normal)
        self.button_group.addButton(self.radio_segmented)
        self.button_group.addButton(self.radio_overlay_outlines)
        self.button_group.addButton(self.radio_labeled_segmentation)
        self.button_group.buttonClicked.connect(self.on_display_mode_changed)

        self.radio_normal.setChecked(True)

        display_mode_layout.addWidget(self.radio_normal)
        display_mode_layout.addWidget(self.radio_segmented)
        display_mode_layout.addWidget(self.radio_overlay_outlines)
        display_mode_layout.addWidget(self.radio_labeled_segmentation)

        display_mode_group.setLayout(display_mode_layout)
        layout.addWidget(display_mode_group)

        # Segmentation Model Selection
        model_group = QGroupBox("Segmentation Model")
        model_layout = QVBoxLayout()

        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(
            SegmentationModels.available_models
        )
        self.model_dropdown.currentTextChanged.connect(self.on_model_changed)
        self.current_model = self.model_dropdown.currentText()

        model_layout.addWidget(self.model_dropdown)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        self.setLayout(layout)

    def show_context_menu(self, position):
        """Show context menu for the image label"""
        pass

    def check_cache_status(self, t, p, c, model):
        """Check and print the status of the segmentation cache for a specific frame."""
        image_data = None

        def receive_image_data(data):
            nonlocal image_data
            image_data = data

        pub.sendMessage("get_image_data", callback=receive_image_data)

        cache_exists = False
        if image_data and hasattr(image_data, "segmentation_cache"):
            cache = image_data.segmentation_cache

            print("CACHE STATUS:")
            print(f"  Current model: {cache.model_name}")

            if model in cache.mmap_arrays_idx:
                array, indices = cache.mmap_arrays_idx[model]
                print(f"  Model {model} has {len(indices)} cached entries")

                simple_key = (t, p, c)
                model_key = (t, p, c, model)

                if simple_key in indices:
                    print(
                        f"  ✓ Frame T={t}, P={p}, C={c} exists with simple key format"
                    )
                    cache_exists = True
                elif model_key in indices:
                    print(f"  ✓ Frame T={t}, P={p}, C={c} exists with model key format")
                    cache_exists = True
                else:
                    sample_keys = list(indices)[:3] if indices else []
                    print(f"  ✗ Frame T={t}, P={p}, C={c} NOT found in cache")
                    print(f"  Sample cached keys: {sample_keys}")
            else:
                print(f"  ✗ Model {model} not found in cache")
                print(f"  Available models: {list(cache.mmap_arrays_idx.keys())}")
        else:
            print("  ✗ No image data or segmentation cache available")

        return cache_exists

    @Slot(object)
    def on_display_mode_changed(self, button):
        """Handle display mode changes"""
        old_mode = self.current_mode

        if button == self.radio_normal:
            self.current_mode = "normal"
        elif button == self.radio_segmented:
            self.current_mode = "segmented"
        elif button == self.radio_overlay_outlines:
            self.current_mode = "overlay"
        elif button == self.radio_labeled_segmentation:
            self.current_mode = "labeled"

        print(f"Display mode changed from '{old_mode}' to '{self.current_mode}'")

        if self.current_mode in ["segmented", "overlay", "labeled"]:
            t, p, c = self.current_t, self.current_p, self.current_c
            cache_exists = self.check_cache_status(t, p, c, self.current_model)

            from nd2_analyzer.analysis.metrics_service import MetricsService

            metrics_service = MetricsService()
            has_metrics = not metrics_service.query_optimized(
                time=t, position=p
            ).is_empty()

            print(
                f"Frame T={t}, P={p}, C={c}: Cache exists={cache_exists}, Metrics exist={has_metrics}"
            )

            if cache_exists:
                print(f"Using cached segmentation for T={t}, P={p}, C={c}")
                self.on_slider_changed()
                return
            elif has_metrics:
                print(
                    f"Using existing metrics for T={t}, P={p}, C={c} - but need to regenerate segmentation"
                )
                self.on_slider_changed()
                return

        self.on_slider_changed()

    @Slot()
    def on_slider_changed(self):
        """Handle slider value changes and request appropriate image"""
        # Map slider value to actual frame index (skipping focus loss frames)
        slider_t_value = self.slider_t.value()
        actual_t = self._map_slider_to_frame(slider_t_value)

        self.current_t = actual_t
        self.current_p = self.slider_p.value()
        self.current_c = self.slider_c.value()
        t, p, c = self.current_t, self.current_p, self.current_c

        # Update label to show actual frame number
        self.t_label.setText(f"T: {t}")

        print(f"DEBUG: Sliders changed to T={t}, P={p}, C={c}")
        pub.sendMessage("view_index_changed", index=(t, p, c))

        if self.current_mode == "normal":
            image = ImageData.get_instance().get(t, p, c)
            # Broadcast to other widgets first
            print(f"🔔 ViewArea broadcasting image_ready: mode=normal, T={t}, P={p}, C={c}")
            pub.sendMessage("image_ready", image=image, time=t, position=p, channel=c, mode="normal")
            # Then handle locally
            self.on_image_ready(image, t, p, c, "normal")
        else:
            print(
                f"DEBUG: Requesting segmented image for T={t}, P={p}, C={c} with mode={self.current_mode}, model={self.current_model}"
            )
            pub.sendMessage(
                "segmented_image_request",
                time=t,
                position=p,
                channel=c,
                mode=self.current_mode,
                model=self.current_model,
            )

    @Slot(str)
    def on_model_changed(self, model_name):
        """Handle segmentation model changes"""
        self.current_model = model_name

        if self.current_mode in ["segmented", "overlay", "labeled"]:
            self.on_slider_changed()

    def on_image_data_loaded(self, image_data):
        """Handle new image_data loading. Updates slider ranges based on image_data dimensions."""
        from nd2_analyzer.data.appstate import ApplicationState

        _shape = image_data.data.shape
        t_max = _shape[0] - 1
        p_max = _shape[1] - 1

        print(f"Shape: {_shape}, Length: {len(_shape)}")

        if len(_shape) == 5:  # T, P, C, Y, X format
            c_max = _shape[2] - 1
            self.has_channels = True
        elif len(_shape) == 4:  # T, P, Y, X format (no channels)
            c_max = 0
            self.has_channels = False
        else:
            c_max = 0
            self.has_channels = False

        # Update valid frames based on focus loss intervals
        appstate = ApplicationState.get_instance()
        if appstate and appstate.experiment:
            self._update_valid_frames(appstate.experiment)

        # Set slider maximum based on valid frames (if focus loss filtering is active)
        if self.valid_time_frames is not None:
            slider_t_max = len(self.valid_time_frames) - 1
        else:
            slider_t_max = t_max

        self.slider_t.setMaximum(max(0, slider_t_max))
        self.slider_p.setMaximum(max(0, p_max))
        self.slider_c.setMaximum(max(0, c_max))

        self.slider_t.setValue(0)
        self.slider_p.setValue(0)
        self.slider_c.setValue(0)

        self.on_slider_changed()

    def highlight_cell(self, cell_id):
        """Highlight a specific cell in the current image view"""
        print(f"ViewAreaWidget: Highlighting cell {cell_id}")

        # Automatically switch to labeled segmentation mode for highlighting
        if not self.radio_labeled_segmentation.isChecked():
            self.radio_labeled_segmentation.setChecked(True)
            self.on_display_mode_changed(self.radio_labeled_segmentation)

        t, p, c = self.current_t, self.current_p, self.current_c
        cell_mapping = None

        def receive_cell_mapping(mapping):
            nonlocal cell_mapping
            cell_mapping = mapping

        pub.sendMessage(
            "get_cell_mapping",
            time=t,
            position=p,
            channel=c,
            cell_id=cell_id,
            callback=receive_cell_mapping,
        )

        if not cell_mapping or cell_id not in cell_mapping:
            print(f"Cell {cell_id} mapping not available")
            return

        # Request fresh labeled segmentation image to avoid stacking highlights
        fresh_image = None

        def receive_fresh_image(image, time, position, channel, mode):
            nonlocal fresh_image
            if time == t and position == p and channel == c and mode == "labeled":
                fresh_image = image

        # Temporarily subscribe to get the fresh image
        pub.subscribe(receive_fresh_image, "image_ready")

        # Request the fresh labeled segmentation
        pub.sendMessage(
            "segmented_image_request",
            time=t,
            position=p,
            channel=c,
            mode="labeled",
            model=self.current_model,
        )

        # Unsubscribe after getting the image
        pub.unsubscribe(receive_fresh_image, "image_ready")

        if fresh_image is None:
            print("Could not get fresh labeled segmentation image")
            return

        # Convert to display format and use as base for highlighting
        fresh_image = self._convert_segmentation_to_display(fresh_image, "labeled")

        cell_info = cell_mapping[cell_id]
        highlighted_image = fresh_image.copy()

        if len(highlighted_image.shape) == 2:
            highlighted_image = cv2.cvtColor(highlighted_image, cv2.COLOR_GRAY2BGR)

        y1, x1, y2, x2 = cell_info["bbox"]

        # Draw WHITE thick bounding box to highlight the selected cell
        cv2.rectangle(highlighted_image, (x1, y1), (x2, y2), (255, 255, 255), 3)

        # Add white text label with cell ID
        cv2.putText(
            highlighted_image,
            f"Cell {cell_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if "metrics" in cell_info and "morphology_class" in cell_info["metrics"]:
            morph_class = cell_info["metrics"]["morphology_class"]
            # Add white text for morphology class
            cv2.putText(
                highlighted_image,
                f"Class: {morph_class}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        self._create_qimage_and_display(highlighted_image)
        self.highlighted_image = highlighted_image

    # BIOFILM MODE METHODS

    def on_show_colony_overlay(self, overlay):
        """Show colony overlay on image"""
        self.colony_overlay = overlay
        self.show_colony_overlay = True
        if hasattr(self, 'current_image'):
            self._display_with_temp_overlay()

    def on_hide_colony_overlay(self):
        """Hide colony overlay"""
        self.show_colony_overlay = False
        if hasattr(self, 'current_image'):
            self._display_with_temp_overlay()

    def enable_manual_selection(self, colony_separator=None):
        """Enable manual colony selection mode"""
        self.manual_selection_active = True
        self.colony_separator = colony_separator
        self.current_polygon_points = []

        # Enable mouse click events on the image label
        self.image_label.mousePressEvent = self.on_image_click
        self.image_label.setStyleSheet("border: 2px solid blue;")  # Visual feedback
        print("Manual selection enabled. Click on image to draw polygon.")

    def disable_manual_selection(self):
        """Disable manual colony selection mode"""
        self.manual_selection_active = False
        self.current_polygon_points = []

        # Disable mouse events
        self.image_label.mousePressEvent = None
        self.image_label.setStyleSheet("")  # Remove border
        print("Manual selection disabled.")

    def on_image_click(self, event):
        """Handle mouse clicks on the image for polygon drawing"""
        if not self.manual_selection_active or not hasattr(self, 'current_image'):
            return

        # Get click position
        click_x = event.pos().x()
        click_y = event.pos().y()

        # Get image dimensions
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return

        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()
        label_width = self.image_label.width()
        label_height = self.image_label.height()

        # Calculate the actual image position within the label (centered)
        x_offset = (label_width - pixmap_width) / 2
        y_offset = (label_height - pixmap_height) / 2

        # Convert label coordinates to image coordinates
        image_x = int(click_x - x_offset)
        image_y = int(click_y - y_offset)

        # Check if click is within image bounds
        if image_x < 0 or image_x >= pixmap_width or image_y < 0 or image_y >= pixmap_height:
            return

        # Scale coordinates if image was resized
        original_height, original_width = self.current_image.shape[:2]
        scale_x = original_width / pixmap_width
        scale_y = original_height / pixmap_height

        image_x = int(image_x * scale_x)
        image_y = int(image_y * scale_y)

        # Add point to polygon
        self.current_polygon_points.append((image_x, image_y))
        print(f"Added polygon point: ({image_x}, {image_y}). Total points: {len(self.current_polygon_points)}")

        # Send point to colony separator
        if self.colony_separator:
            self.colony_separator.add_polygon_point(image_x, image_y)

        # Visual feedback - draw temporary overlay
        self.draw_polygon_preview()

        # Notify segmentation widget
        pub.sendMessage("polygon_point_added", x=image_x, y=image_y, total_points=len(self.current_polygon_points))

    def draw_polygon_preview(self):
        """Draw a preview of the current polygon being drawn"""
        if len(self.current_polygon_points) < 2:
            return

        # Create overlay on current image
        overlay_image = self.current_image.copy()

        # Normalize to 8-bit if needed
        if overlay_image.dtype != np.uint8:
            overlay_image = ((overlay_image - overlay_image.min()) /
                           (overlay_image.max() - overlay_image.min()) * 255).astype(np.uint8)

        # Convert to RGB if grayscale
        if len(overlay_image.shape) == 2:
            overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_GRAY2RGB)

        # Draw polygon lines
        points = np.array(self.current_polygon_points, dtype=np.int32)
        cv2.polylines(overlay_image, [points], False, (0, 255, 255), 2)  # Yellow lines

        # Draw points
        for point in self.current_polygon_points:
            cv2.circle(overlay_image, point, 3, (0, 0, 255), -1)  # Red dots

        # Store as temp overlay for display
        self.temp_polygon_overlay = overlay_image.copy()

        # Display
        self._display_with_temp_overlay()

    def _display_with_temp_overlay(self):
        """Display the current image with temporary polygon overlay and/or colony overlay"""
        if not hasattr(self, 'current_image'):
            return

        display_image = self.current_image.copy()

        # Normalize to 8-bit if needed
        if display_image.dtype != np.uint8:
            display_image = ((display_image - display_image.min()) /
                           (display_image.max() - display_image.min()) * 255).astype(np.uint8)

        # Convert to RGB if grayscale
        if len(display_image.shape) == 2:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2RGB)

        # Add polygon overlay if present
        if hasattr(self, 'temp_polygon_overlay') and self.temp_polygon_overlay is not None:
            if self.temp_polygon_overlay.shape == display_image.shape:
                overlay_mask = (self.temp_polygon_overlay != display_image).any(axis=2)
                display_image[overlay_mask] = self.temp_polygon_overlay[overlay_mask]

        # Add colony overlay if enabled
        if self.show_colony_overlay and self.colony_overlay is not None:
            if self.colony_overlay.shape[:2] == display_image.shape[:2]:
                overlay_mask = (self.colony_overlay.sum(axis=2) > 0)
                display_image[overlay_mask] = self.colony_overlay[overlay_mask]

        # Display the image
        self._create_qimage_and_display(display_image)
