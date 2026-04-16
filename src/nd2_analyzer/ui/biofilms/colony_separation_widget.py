"""
Colony Separation Widget - Separate tab for manual and automatic colony detection
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QTabWidget, QGroupBox, QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt
from pubsub import pub
import numpy as np

from .colony_separator import ColonySeparator


class ColonySeparationWidget(QWidget):
    """Widget for colony separation with manual and automatic detection modes"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize colony separator
        self.colony_separator = ColonySeparator()
        self.current_raw_image = None
        self.colony_overlay_visible = False
        self.auto_detect_params = {
            "min_size": 1,
            "threshold": 50,
            "kernel": 1
        }

        # Initialize storage for manual and auto-verified colonies
        self.manual_additions = []
        self.detected_colonies = []

        # Subscribe to image updates
        pub.subscribe(self.on_image_ready, "image_ready")

        self.init_ui()

    def init_ui(self):
        """Initialize the UI with two sub-tabs"""
        main_layout = QVBoxLayout(self)

        # Create tab widget for Manual vs Automatic
        self.tab_widget = QTabWidget()

        # Manual Selection Tab
        manual_tab = self.create_manual_selection_tab()
        self.tab_widget.addTab(manual_tab, "Manual Selection")

        # Automatic Thresholding Tab
        threshold_tab = self.create_threshold_tab()
        self.tab_widget.addTab(threshold_tab, "Automatic Detection")

        main_layout.addWidget(self.tab_widget)

        # Common controls at the bottom
        self.add_common_controls(main_layout)

        main_layout.addStretch()

    def create_manual_selection_tab(self):
        """Create the manual selection tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Header
        header = QLabel("Manual Colony Selection")
        header.setStyleSheet("font-weight: bold; font-size: 14px; color: #2196F3;")
        layout.addWidget(header)

        # Instructions
        instructions = QLabel(
            "Use manual selection to draw polygons around colonies.\n"
            "This is useful for fluorescence channels or complex colony shapes."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; padding: 10px;")
        layout.addWidget(instructions)

        # Manual selection button
        self.start_manual_btn = QPushButton("Start Manual Selection")
        self.start_manual_btn.clicked.connect(self.start_manual_selection)
        self.start_manual_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 10px;"
        )
        layout.addWidget(self.start_manual_btn)

        # Info label
        info_label = QLabel(
            "Click 'Start Manual Selection' to open the ROI selector.\n"
            "Draw polygons around each colony you want to track."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-style: italic; color: #999; padding: 10px;")
        layout.addWidget(info_label)

        layout.addStretch()
        return tab

    def create_threshold_tab(self):
        """Create the automatic thresholding tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Header
        header = QLabel("Automatic Colony Detection")
        header.setStyleSheet("font-weight: bold; font-size: 14px; color: #4CAF50;")
        layout.addWidget(header)

        # Instructions
        instructions = QLabel(
            "Automatically detect colonies using Otsu thresholding.\n"
            "Works best with fluorescence images (GFP, RFP, etc.)."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; padding: 10px;")
        layout.addWidget(instructions)

        # Automatic Detection button
        self.start_auto_btn = QPushButton("Start Automatic Detection")
        self.start_auto_btn.clicked.connect(self.open_auto_detect_widget)
        self.start_auto_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;"
        )
        self.start_auto_btn.setEnabled(True)
        layout.addWidget(self.start_auto_btn)

        # Info label
        info_label = QLabel(
            "Click 'Start Automatic Detection' to open auto detection controller.\n"
            "Move sliders to auto-select colonies you want to track."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-style: italic; color: #999; padding: 10px;")
        layout.addWidget(info_label)

        layout.addStretch()
        return tab

    def add_common_controls(self, layout):
        """Add common controls that apply to both modes"""
        # Separator line
        separator = QLabel()
        separator.setFrameStyle(QLabel.HLine | QLabel.Sunken)
        layout.addWidget(separator)

        # Colony count display
        self.colony_count_label = QLabel("Colonies: 0")
        self.colony_count_label.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #2196F3; padding: 10px;"
        )
        layout.addWidget(self.colony_count_label)

        # Control buttons
        button_layout = QHBoxLayout()

        self.show_overlay_btn = QPushButton("Show/Hide Overlay")
        self.show_overlay_btn.clicked.connect(self.toggle_colony_overlay)
        self.show_overlay_btn.setEnabled(False)
        button_layout.addWidget(self.show_overlay_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all_colonies)
        self.clear_btn.setStyleSheet("background-color: #f44336; color: white;")
        button_layout.addWidget(self.clear_btn)

        layout.addLayout(button_layout)

        # Status label
        self.status_label = QLabel("Load an image to begin colony separation.")
        self.status_label.setStyleSheet("color: #666; font-style: italic; padding: 10px;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    # === Event Handlers ===

    def on_image_ready(self, image, time, position, channel, mode):
        """Handle image ready event"""
        if mode == "normal":
            # Store the raw image for colony detection
            self.current_raw_image = image
            self.status_label.setText(f"Image loaded: T={time}, P={position}, C={channel}")

    #def on_closing_radius_changed(self, value):
    #    """Handle closing radius slider change"""
    #    self.closing_radius_label.setText(str(value))
    #    self.colony_separator.update_parameters(closing_radius=value)

    #def on_min_object_changed(self, value):
    #    """Handle min object size slider change"""
    #    self.min_object_label.setText(str(value))
    #    self.colony_separator.update_parameters(min_object_size=value)


    def open_auto_detect_widget(self):
        """Open colony verification dialog"""

        from nd2_analyzer.ui.dialogs.colony_auto_verifier import VerifyColoniesDialog
        colonies = self.colony_separator.detected_colonies

        verify_dialog = VerifyColoniesDialog(
            self.current_raw_image,
            colonies,
            params=self.auto_detect_params,
            parent=self
        )
        verify_dialog.colonies_verified.connect(self.handle_verified_colonies)
        result = verify_dialog.exec()
        if result:
            self.status_label.setText(f"✅ Automatic detection complete: {len(self.get_all_colonies())} colonies.")
        else:
            self.status_label.setText("Automatic detection cancelled.")

    def handle_verified_colonies(self, verified_colonies, params):
        """Handle colonies verified by autodetection"""
        # Preserve the last autodetection state and clear existing auto additions
        self.auto_detect_params = params
        self.detected_colonies = []
        for colony in verified_colonies:
            # Preserve contour if it exists
            if "contour" in colony:
                self.colony_separator.detected_colonies.append(colony)
                self.detected_colonies.append(colony)
            else:
                # Fallback to bounding box if contour is not available
                x1, y1, x2, y2 = colony["bbox"]
                new_colony = self.colony_separator.create_bounding_box_colony(
                    x1, y1, x2, y2,
                    self.current_raw_image.shape,
                )
                self.detected_colonies.append(new_colony)

        self.colony_count_label.setText(f"Colonies: {len(self.get_all_colonies())}")
        self.colony_overlay_visible = True
        self.update_colony_overlay()

    def handle_selected_colonies(self, colonies_data):
        """Handle colonies selected from the ROI selector"""
        self.manual_additions = []
        # Creates new colonies on overlay from polygon points
        for colony in colonies_data:
            polygon = colony['polygon']
            new_colony = self.colony_separator.add_manual_colony(
                polygon,
                self.current_raw_image.shape
            )
            self.manual_additions.append(new_colony)

        self.colony_count_label.setText(f"Colonies: {len(self.get_all_colonies())}")
        self.colony_overlay_visible = True
        self.update_colony_overlay()

    def get_all_colonies(self):
        return self.manual_additions + self.detected_colonies

    def start_manual_selection(self):
        """Start manual colony selection"""
        if self.current_raw_image is None:
            self.status_label.setText("⚠️ No image loaded. Please view an image first.")
            return

        from nd2_analyzer.ui.dialogs.colony_roi_selector import ColonyROISelector

        # Get existing colonies
        existing_colonies = []
        for colony in self.colony_separator.get_all_colonies():
            if 'polygon_points' in colony:
                existing_colonies.append({
                    'colony_id': colony['colony_id'],
                    'polygon': colony['polygon_points'],
                    'mask': None
                })

        # Open ROI selector dialog
        roi_dialog = ColonyROISelector(
            self.current_raw_image,
            existing_colonies=existing_colonies,
            parent=self
        )
        roi_dialog.colonies_selected.connect(self.handle_selected_colonies)

        # Update UI
        self.start_manual_btn.setEnabled(False)
        self.status_label.setText("ROI Selector opened. Draw polygons around colonies.")

        # Show dialog
        result = roi_dialog.exec()

        # Reset UI
        self.start_manual_btn.setEnabled(True)

        if result:
            self.status_label.setText(f"✅ Manual selection complete: {len(self.get_all_colonies())} colonies.")
        else:
            self.status_label.setText("Manual selection cancelled.")

    def toggle_colony_overlay(self):
        """Toggle colony overlay visibility"""
        self.colony_overlay_visible = not self.colony_overlay_visible
        self.update_colony_overlay()

    def update_colony_overlay(self):
        """Update colony overlay in the view"""
        colonies = self.colony_separator.get_all_colonies()

        if self.current_raw_image is not None and self.colony_overlay_visible and len(colonies) > 0:
            # Create overlay
            overlay = self.colony_separator.create_colony_overlay(self.current_raw_image.shape)
            pub.sendMessage("show_colony_overlay", overlay=overlay)
        else:
            # Hide overlay
            pub.sendMessage("hide_colony_overlay")

    def clear_all_colonies(self):
        """Clear all detected colonies"""
        self.manual_additions = []
        self.detected_colonies = []
        self.colony_separator.manual_additions = []
        self.colony_separator.detected_colonies = []
        self.colony_count_label.setText("Colonies: 0")
        self.show_overlay_btn.setEnabled(False)
        self.colony_overlay_visible = False
        self.update_colony_overlay()
        self.status_label.setText("All colonies cleared.")
