import os
import sys
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QSlider, QGroupBox, QRadioButton, QButtonGroup,
                               QSizePolicy, QComboBox, QCheckBox)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap, QImage
import numpy as np
from pubsub import pub
import cv2

from nd2_analyzer.analysis.segmentation.segmentation_models import SegmentationModels


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
        self.current_mode = "normal"
        self.current_model = None
        
        # Add colony overlay support
        self.colony_overlay = None
        self.show_colony_overlay = False
        
        # Add manual colony selection support
        self.manual_selection_active = False
        self.current_polygon_points = []
        self.colony_separator = None

        # Set up the UI
        self.init_ui()

        # Subscribe to relevant topics
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")
        pub.subscribe(self.on_image_ready, "image_ready")
        pub.subscribe(self.highlight_cell, "highlight_cell_requested")
        pub.subscribe(self.provide_current_param, "get_current_t")
        pub.subscribe(self.provide_current_param, "get_current_p")
        pub.subscribe(self.provide_current_param, "get_current_c")
        pub.subscribe(self.provide_current_model, "get_current_model")
        pub.subscribe(self.provide_current_model_callback, "get_current_model_callback")
        pub.subscribe(self.on_segmentation_cache_miss, "segmentation_cache_miss")
        
        # Add colony overlay subscriptions
        pub.subscribe(self.on_show_colony_overlay, "show_colony_overlay")
        pub.subscribe(self.on_hide_colony_overlay, "hide_colony_overlay")
        
        # Add manual selection subscriptions
        pub.subscribe(self.enable_manual_selection, "enable_manual_colony_selection")
        pub.subscribe(self.disable_manual_selection, "disable_manual_colony_selection")
        
        
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
        if not self.manual_selection_active:
            return
        
        # Get click position relative to the image
        click_x = event.pos().x()
        click_y = event.pos().y()
        
        # Convert from display coordinates to image coordinates
        if hasattr(self.image_label, 'pixmap') and self.image_label.pixmap():
            pixmap = self.image_label.pixmap()
            label_size = self.image_label.size()
            pixmap_size = pixmap.size()
            
            # Calculate scaling factors
            scale_x = pixmap_size.width() / label_size.width()
            scale_y = pixmap_size.height() / label_size.height()
            
            # Convert to image coordinates
            image_x = int(click_x * scale_x)
            image_y = int(click_y * scale_y)
            
            # Add point to current polygon
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
        
        # Create temporary overlay showing the polygon in progress
        if hasattr(self, 'current_image') and self.current_image is not None:
            # Create overlay with polygon lines
            overlay = np.zeros((*self.current_image.shape[:2], 3), dtype=np.uint8)
            
            # Draw lines between points
            points = np.array(self.current_polygon_points, dtype=np.int32)
            for i in range(len(points) - 1):
                cv2.line(overlay, tuple(points[i]), tuple(points[i + 1]), (0, 255, 0), 2)
            
            # Draw points
            for point in points:
                cv2.circle(overlay, tuple(point), 3, (255, 0, 0), -1)
            
            # Show the overlay
            self.temp_polygon_overlay = overlay
            self._display_with_temp_overlay()

    def _display_with_temp_overlay(self):
        """Display current image with temporary polygon overlay"""
        if not hasattr(self, 'current_image') or self.current_image is None:
            return
        
        display_image = self.current_image.copy()
        
        # Convert to RGB if needed
        if len(display_image.shape) == 2:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2RGB)
        
        # Add temporary polygon overlay
        if hasattr(self, 'temp_polygon_overlay'):
            overlay_mask = (self.temp_polygon_overlay.sum(axis=2) > 0)
            display_image[overlay_mask] = self.temp_polygon_overlay[overlay_mask]
        
        # Add colony overlay if enabled
        if self.show_colony_overlay and self.colony_overlay is not None:
            overlay_mask = (self.colony_overlay.sum(axis=2) > 0)
            display_image[overlay_mask] = self.colony_overlay[overlay_mask]
        
        # Display the image
        self._display_image_array(display_image)

    def _display_image_array(self, image_array):
        """Helper method to display numpy array as QPixmap"""
        try:
            if len(image_array.shape) == 2:
                # Grayscale
                height, width = image_array.shape
                bytes_per_line = width
                q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                # RGB
                height, width = image_array.shape[:2]
                bytes_per_line = 3 * width
                q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale to fit the label
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"Error displaying image: {e}")
            self.image_label.setText(f"Error displaying image: {str(e)}")

    def clear_current_polygon(self):
        """Clear the current polygon being drawn"""
        self.current_polygon_points = []
        if hasattr(self, 'temp_polygon_overlay'):
            del self.temp_polygon_overlay
        # Redisplay image without polygon preview
        if hasattr(self, 'current_image'):
            self._display_with_temp_overlay()

    def on_segmentation_cache_miss(self, time, position, channel, model):
        """Handle when cached segmentation is not available"""
        if (time == self.current_t and 
            position == self.current_p and 
            channel == self.current_c):
            # Notify the user
            print(f"No cached segmentation found for T={time}, P={position}, C={channel}")
            
            # Request segmentation without use_cached flag
            pub.sendMessage("segmented_image_request",
                            time=time,
                            position=position,
                            channel=channel,
                            mode=self.current_mode,
                            model=self.current_model,
                            use_cached=False)  # Force computing new segmentation

    def provide_current_param(self, topic=pub.AUTO_TOPIC, default=0):
        param_map = {
            "get_current_t": self.current_t,
            "get_current_p": self.current_p,
            "get_current_c": self.current_c
        }
        topic_name = topic.getName()
        value = param_map.get(topic_name, default)
        print(f"ViewAreaWidget: Providing {topic_name}={value}")
        return value
    
    def provide_current_model(self, topic=pub.AUTO_TOPIC, default="unet"):
        """Provide the current segmentation model"""
        model = self.current_model if self.current_model else default
        print(f"ViewAreaWidget: Providing current model: {model}")
        return model
    
    def provide_current_model_callback(self, callback):
        """Provide the current segmentation model via callback"""
        model = self.current_model if self.current_model else "bact_phase_cp3"
        print(f"ViewAreaWidget: Providing current model via callback: {model}")
        callback(model)

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

        # Image display area
        self.image_label = QLabel()
        self.image_label.setScaledContents(True)
        self.image_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(
            self.show_context_menu)
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
        # Will be updated when experiment is loaded
        self.slider_t.setMaximum(0)
        self.slider_t.valueChanged.connect(self.on_slider_changed)
        self.slider_t.valueChanged.connect(
            lambda value: self.t_label.setText(f"T: {value}")
        )
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
        # Will be updated when experiment is loaded
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
        # Will be updated when experiment is loaded
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
        self.normalize_checkbox.setChecked(True)  # Default to checked
        self.normalize_checkbox.stateChanged.connect(
            self.on_slider_changed)  # Refresh image when changed
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

        # Set default selection
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
        self.model_dropdown.addItems([
            SegmentationModels.CELLPOSE_BACT_PHASE,
            SegmentationModels.CELLPOSE_BACT_FLUOR,
            SegmentationModels.CELLPOSE,
            SegmentationModels.UNET,
            SegmentationModels.OMNIPOSE_BACT_PHASE_AFFINITY
        ])
        self.model_dropdown.currentTextChanged.connect(self.on_model_changed)
        self.current_model = self.model_dropdown.currentText()

        model_layout.addWidget(self.model_dropdown)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Set the layout
        self.setLayout(layout)

    def show_context_menu(self, position):
        """Show context menu for the image label"""
        # This is a placeholder - implement as needed
        pass

    
    def on_image_ready(self, image, time, position, channel, mode):
        """Handle both raw and segmented image responses"""
        # Check if this is the image we're currently expecting
        if (time != self.current_t or
            position != self.current_p or
            channel != self.current_c or
                mode != self.current_mode):
            return  # Ignore outdated images

        # STORE THE CURRENT IMAGE
        self.current_image = image
        
        # Convert and display based on mode
        if mode == "normal":
            self._display_raw_image(image)
        else:
            self._display_processed_image(image, mode)

    def _display_raw_image(self, image):
        """Display raw microscopy image with optional overlay"""
        try:
            # Normalize for display
            if image.dtype != np.uint8:
                # Normalize to 0-255 range
                image_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            else:
                image_norm = image

            # Convert to RGB for overlay support
            if len(image_norm.shape) == 2:
                display_image = cv2.cvtColor(image_norm, cv2.COLOR_GRAY2RGB)
            else:
                display_image = image_norm

            # Add colony overlay if enabled
            if self.show_colony_overlay and self.colony_overlay is not None:
                # Ensure overlay matches image dimensions
                if self.colony_overlay.shape[:2] == display_image.shape[:2]:
                    # Blend overlay with image (make overlay semi-transparent)
                    overlay_mask = (self.colony_overlay.sum(axis=2) > 0)
                    alpha = 0.7  # Make overlay semi-transparent
                    display_image[overlay_mask] = (alpha * self.colony_overlay[overlay_mask] + 
                                                (1 - alpha) * display_image[overlay_mask]).astype(np.uint8)
                else:
                    print(f"Overlay shape {self.colony_overlay.shape} doesn't match image shape {display_image.shape}")

            # Convert to QImage and display
            height, width = display_image.shape[:2]
            bytes_per_line = 3 * width
            
            q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"Error displaying raw image: {e}")
            self.image_label.setText(f"Error displaying image: {str(e)}")

    def _display_processed_image(self, image, mode):
        """Display processed/segmented image with optional overlay"""
        try:
            # Convert to displayable format
            if mode == "segmented":
                # Convert binary segmentation to 8-bit for display
                display_image = (image > 0).astype(np.uint8) * 255
                # Convert to RGB for overlay support
                display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2RGB)
            else:
                # Handle other modes (overlay, labeled, etc.)
                if len(image.shape) == 2:
                    display_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                else:
                    display_image = image.astype(np.uint8)

            # Add colony overlay if enabled
            if self.show_colony_overlay and self.colony_overlay is not None:
                # Ensure overlay matches image dimensions
                if self.colony_overlay.shape[:2] == display_image.shape[:2]:
                    # Blend overlay with image
                    overlay_mask = (self.colony_overlay.sum(axis=2) > 0)
                    display_image[overlay_mask] = self.colony_overlay[overlay_mask]
                else:
                    print(f"Overlay shape {self.colony_overlay.shape} doesn't match image shape {display_image.shape}")

            # Convert to QImage and display
            height, width = display_image.shape[:2]
            bytes_per_line = 3 * width
            
            q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"Error displaying processed image: {e}")
            # Show error message on label
            self.image_label.setText(f"Error displaying image: {str(e)}")
        
        
    def _update_display(self, q_image):
        """Common display update logic"""
        pixmap = QPixmap.fromImage(q_image).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

        # Store the current image for highlighting
        if hasattr(self, "current_image_data"):
            self.current_image = self.current_image_data.copy()

    def check_cache_status(self, t, p, c, model):
        """
        Check and print the status of the segmentation cache for a specific frame.
        """
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
                
                # Check for our specific key in different formats
                simple_key = (t, p, c)
                model_key = (t, p, c, model)
                
                if simple_key in indices:
                    print(f"  ✓ Frame T={t}, P={p}, C={c} exists with simple key format")
                    cache_exists = True
                elif model_key in indices:
                    print(f"  ✓ Frame T={t}, P={p}, C={c} exists with model key format")
                    cache_exists = True
                else:
                    # Print some sample keys to debug
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

        # Update current mode based on button
        if button == self.radio_normal:
            self.current_mode = "normal"
        elif button == self.radio_segmented:
            self.current_mode = "segmented"
        elif button == self.radio_overlay_outlines:
            self.current_mode = "overlay"
        elif button == self.radio_labeled_segmentation:
            self.current_mode = "labeled"

        print(f"Display mode changed from '{old_mode}' to '{self.current_mode}'")

        # If changing to a segmentation mode, check if metrics data exists
        if self.current_mode in ["segmented", "overlay", "labeled"]:
            # Get current frame parameters
            t = self.current_t
            p = self.current_p
            c = self.current_c
            
            # Check and print cache status
            cache_exists = self.check_cache_status(t, p, c, self.current_model)
            
            # Check metrics data
            from metrics_service import MetricsService
            metrics_service = MetricsService()
            has_metrics = not metrics_service.query(time=t, position=p, channel=c).is_empty()
            
            print(f"Frame T={t}, P={p}, C={c}: Cache exists={cache_exists}, Metrics exist={has_metrics}")

            if cache_exists:
                print(f"Using cached segmentation for T={t}, P={p}, C={c}")
                # Request image with the current mode
                self.on_slider_changed()
                return
            elif has_metrics:
                print(f"Using existing metrics for T={t}, P={p}, C={c} - but need to regenerate segmentation")
                # Use pre-loaded metrics data but need to regenerate segmentation
                self.on_slider_changed()
                return

        # Request new image with updated mode
        self.on_slider_changed()
    
    @Slot()
    def on_slider_changed(self):
        """Handle slider value changes and request appropriate image"""
        # Update current values
        self.current_t = self.slider_t.value()
        self.current_p = self.slider_p.value()
        self.current_c = self.slider_c.value()

        print(f"DEBUG: Sliders changed to T={self.current_t}, P={self.current_p}, C={self.current_c}")

        # Send different requests based on display mode
        if self.current_mode == "normal":
            print(f"DEBUG: Requesting raw image for T={self.current_t}, P={self.current_p}, C={self.current_c}")
            pub.sendMessage("raw_image_request",
                            time=self.current_t,
                            position=self.current_p,
                            channel=self.current_c)
        else:
            print(f"DEBUG: Requesting segmented image for T={self.current_t}, P={self.current_p}, C={self.current_c} with mode={self.current_mode}, model={self.current_model}")
            pub.sendMessage("segmented_image_request",
                            time=self.current_t,
                            position=self.current_p,
                            channel=self.current_c,
                            mode=self.current_mode,
                            model=self.current_model)
        
    
    @Slot(str)
    def on_model_changed(self, model_name):
        """Handle segmentation model changes"""
        self.current_model = model_name

        # If in a segmentation mode, request new image with updated model
        if self.current_mode in ["segmented", "overlay", "labeled"]:
            self.on_slider_changed()

    def on_image_data_loaded(self, image_data):
        """
        Handle new image_data loading.
        Updates slider ranges based on image_data dimensions.

        Args:
            image_data: The loaded image_data object
        """
        # Get dimensions from image_data
        _shape = image_data.data.shape
        t_max = _shape[0] - 1
        p_max = _shape[1] - 1

        print(f"Shape: {_shape}, Length: {len(_shape)}")

        if len(_shape) == 5:  # T, P, C, Y, X format
            c_max = _shape[2] - 1
            self.has_channels = True
        elif len(_shape) == 4:  # T, P, Y, X format (no channels)
            c_max = 0  # Default to single channel
            self.has_channels = False
        else:
            c_max = 0
            self.has_channels = False

        # Update slider ranges
        self.slider_t.setMaximum(max(0, t_max))
        self.slider_p.setMaximum(max(0, p_max))
        self.slider_c.setMaximum(max(0, c_max))

        # Reset slider positions
        self.slider_t.setValue(0)
        self.slider_p.setValue(0)
        self.slider_c.setValue(0)

        # Request initial image
        self.on_slider_changed()

    def apply_label_colormap(self, labeled_image):
        """
        Apply a colormap to a labeled segmentation image.

        Args:
            labeled_image: Labeled segmentation image

        Returns:
            Colored image suitable for display
        """
        # Normalize to 0-255 range
        if labeled_image.max() > 0:
            normalized = (labeled_image.astype(np.float32) /
                          labeled_image.max() * 255).astype(np.uint8)
        else:
            normalized = labeled_image.astype(np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        # Set background (label 0) to black
        colored[labeled_image == 0] = [0, 0, 0]

        return colored

    # TODO: implement highlight cell
    def highlight_cell(self, cell_id):
        """
        Highlight a specific cell in the current image view

        Args:
            cell_id: ID of the cell to highlight
        """
        print(f"ViewAreaWidget: Highlighting cell {cell_id}")

        # Get current frame information
        t = self.current_t
        p = self.current_p
        c = self.current_c

        # Request cell mapping from MorphologyWidget
        cell_mapping = None

        def receive_cell_mapping(mapping):
            nonlocal cell_mapping
            cell_mapping = mapping

        # Send message to request cell mapping data
        pub.sendMessage("get_cell_mapping",
                        time=t,
                        position=p,
                        channel=c,
                        cell_id=cell_id,
                        callback=receive_cell_mapping)

        # Check if we received the cell mapping
        if not cell_mapping or cell_id not in cell_mapping:
            print(f"Cell {cell_id} mapping not available")
            return

        # Get the cell info from the mapping
        cell_info = cell_mapping[cell_id]

        # Check if we have a current image to work with
        if not hasattr(self, "current_image_data") or self.current_image_data is None:
            print("No current image available to highlight cell")
            return

        # Make a copy of the current image for highlighting
        highlighted_image = self.current_image_data.copy()

        # Convert to RGB if grayscale (for drawing colored highlights)
        if len(highlighted_image.shape) == 2:
            highlighted_image = cv2.cvtColor(
                highlighted_image, cv2.COLOR_GRAY2BGR)

        # Get the bounding box
        y1, x1, y2, x2 = cell_info["bbox"]

        # Draw a highlighted rectangle around the cell
        cv2.rectangle(
            highlighted_image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),  # Red color (BGR format)
            2             # Line thickness
        )

        # Add a text label with the cell ID
        cv2.putText(
            highlighted_image,
            f"Cell {cell_id}",
            (x1, y1 - 5),  # Position above the cell
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,           # Font scale
            (0, 255, 0),   # Green color (BGR)
            1              # Line thickness
        )

        # Add morphology class if available
        if "metrics" in cell_info and "morphology_class" in cell_info["metrics"]:
            morph_class = cell_info["metrics"]["morphology_class"]
            cv2.putText(
                highlighted_image,
                f"Class: {morph_class}",
                (x1, y2 + 15),  # Position below the cell
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,           # Font scale
                (255, 255, 0),  # Yellow color (BGR)
                1              # Line thickness
            )

        # Display the highlighted image
        height, width = highlighted_image.shape[:2]
        bytes_per_line = 3 * width  # RGB image

        # Convert to QImage and display
        q_image = QImage(highlighted_image.data, width, height,
                         bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

        # Store the highlighted image
        self.highlighted_image = highlighted_image
        
        
    def on_show_colony_overlay(self, overlay):
        """Handle request to show colony overlay"""
        print(f"DEBUG: ViewArea received overlay with shape {overlay.shape}")
        self.colony_overlay = overlay
        self.show_colony_overlay = True
        
        # Force image refresh to show overlay
        print("DEBUG: Forcing image refresh to show overlay")
        pub.sendMessage("image_request",
                        time=self.current_t,
                        position=self.current_p,
                        channel=self.current_c)

    def on_hide_colony_overlay(self):
        """Handle request to hide colony overlay"""
        self.show_colony_overlay = False
        self._update_display()

    def _update_display(self):
        """Update the current display with or without overlay"""
        # Re-request the current image to trigger display update
        pub.sendMessage("image_request",
                        time=self.current_t,
                        position=self.current_p,
                        channel=self.current_c)
