from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
                               QWidget, QScrollArea, QFrame, QGroupBox, QSlider, QComboBox, QRubberBand)
from PySide6.QtCore import Qt, Signal, QTimer, QRect, QPoint, QSize, QEvent
from PySide6.QtGui import QPixmap, QImage, QShortcut, QKeySequence
import cv2
import numpy as np
import copy

from nd2_analyzer.ui.biofilms.colony_separator import ColonySeparator

class VerifyColoniesDialog(QDialog):
    """Dialog for Automatic Colony Detection and Verification."""
    colonies_verified = Signal(list, dict)
    def __init__(self, image, colonies, params=None, parent=None):
        super().__init__(parent)

        self.image = image
        self.colonies = colonies.copy()
        self.colony_separator = ColonySeparator()
        self.colony_separator.detected_colonies = self.colonies.copy()
        # Initilize State History Undo/Redo
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = 5
        # Default auto colony modify mode
        self.modify_mode = "Trim"
        self.origin = QPoint()
        self.rubber_band = None
        # Default slider params
        self.params = params or {
            "min_size": 1,
            "threshold": 50,
            "kernel": 1
        }

        self.setWindowTitle("Verify Colonies")
        self.setMinimumSize(800, 600)
        self.resize(900, 650)

        # build the real UI
        self.init_ui()

        # populate widgets
        self.update_colonies_list()
        self.update_display()
        self.setup_image()
        self.trigger_detect()

    def init_ui(self):
        # Setup for Main Layout (Vertical)
        main_layout = QVBoxLayout(self)

        # Top of the dialog: Image and Colony List
        top_layout = QHBoxLayout()

        # Image Stage
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.image_label = QLabel()
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        self.image_label.setAlignment(Qt.AlignCenter)

        # Set mouse detection event for modifying auto-detection
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.image_label)
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)

        # Set colony scroll to fit colony image in dialog
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        left_layout.addWidget(scroll_area)

        self.status_label = QLabel("Detected colonies shown with bounding boxes")
        self.status_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        left_layout.addWidget(self.status_label)
        top_layout.addWidget(left_widget, 2)

        # Colony List Stage
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_widget.setMaximumWidth(300)

        title = QLabel("Detected Colonies")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2196F3;")
        right_layout.addWidget(title)
        colonies_group = QFrame()
        colonies_group.setFrameStyle(QFrame.Box)
        colonies_layout = QVBoxLayout(colonies_group)

        self.colonies_count_label = QLabel(f"Colonies: {len(self.colonies)}")
        self.colonies_count_label.setStyleSheet("font-weight: bold;")
        colonies_layout.addWidget(self.colonies_count_label)

        self.colonies_list_widget = QWidget()
        self.colonies_list_layout = QVBoxLayout(self.colonies_list_widget)

        colonies_scroll = QScrollArea()
        colonies_scroll.setWidget(self.colonies_list_widget)
        colonies_scroll.setWidgetResizable(True)
        colonies_scroll.setMaximumHeight(450)

        colonies_layout.addWidget(colonies_scroll)

        right_layout.addWidget(colonies_group)

        top_layout.addWidget(right_widget, 1)

        # Set top section of main layout
        main_layout.addLayout(top_layout)

        # Add timer functionality for live slider processing
        self.update_timer = QTimer()
        self.update_timer.setInterval(150)   # ms
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.detect_colonies)

        # Set sliders to the middle of layout
        size_group = QGroupBox("Size Filtering")
        size_layout = QVBoxLayout()
        size_group.setLayout(size_layout)

        # Min size slider
        min_size_layout = QHBoxLayout()
        min_size_layout.addWidget(QLabel("Min Colony Size (px):"))

        self.min_size_slider = QSlider(Qt.Horizontal)
        self.min_size_slider.setMinimum(1)
        self.min_size_slider.setMaximum(100)
        self.min_size_slider.setValue(self.params.get("min_size", 1))
        self.min_size_slider.valueChanged.connect(self.on_min_size_changed)
        self.min_size_slider.sliderMoved.connect(self.trigger_detect)
        min_size_layout.addWidget(self.min_size_slider)

        self.min_size_label = QLabel(f"{self.min_size_slider.value()}%")
        self.min_size_label.setMinimumWidth(50)
        self.min_size_label.setStyleSheet("font-weight: bold;")
        min_size_layout.addWidget(self.min_size_label)
        size_layout.addLayout(min_size_layout)

        # Threshold slider
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(self.params.get("threshold", 50))
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        self.threshold_slider.sliderMoved.connect(self.trigger_detect)
        threshold_layout.addWidget(self.threshold_slider)

        self.threshold_label = QLabel(f"{self.threshold_slider.value()}%")
        self.threshold_label.setMinimumWidth(60)
        self.threshold_label.setStyleSheet("font-weight: bold;")
        threshold_layout.addWidget(self.threshold_label)
        size_layout.addLayout(threshold_layout)

        # Kernel slider
        kernel_layout = QHBoxLayout()
        kernel_layout.addWidget(QLabel("Kernel Size (px):"))

        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setMinimum(0)
        self.kernel_slider.setMaximum(100)
        self.kernel_slider.setValue(self.params.get("kernel", 1))
        self.kernel_slider.valueChanged.connect(self.on_kernel_changed)
        self.kernel_slider.sliderMoved.connect(self.trigger_detect)
        kernel_layout.addWidget(self.kernel_slider)

        self.kernel_label = QLabel(f"{self.kernel_slider.value()}%")
        self.kernel_label.setMinimumWidth(60)
        self.kernel_label.setStyleSheet("font-weight: bold;")
        kernel_layout.addWidget(self.kernel_label)

        size_layout.addLayout(kernel_layout)

        main_layout.addWidget(size_group)

        # Set layout for colony modification mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Modify Mode:"))
        # Trim/Add selector
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Trim", "Add"])
        self.mode_selector.currentTextChanged.connect(self.on_mode_changed)

        mode_layout.addWidget(self.mode_selector)

        # Small spacing
        mode_layout.addSpacing(20)

        # Undo button
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo)
        mode_layout.addWidget(self.undo_btn)

        # Redo button
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.clicked.connect(self.redo)
        mode_layout.addWidget(self.redo_btn)

        # Undo shortcut
        self.undo_shortcut = QShortcut(QKeySequence(QKeySequence.Undo), self)
        self.undo_shortcut.activated.connect(self.undo_btn.click)

        # Redo shortcut
        self.redo_shortcut = QShortcut(QKeySequence(QKeySequence.Redo), self)
        self.redo_shortcut.activated.connect(self.redo_btn.click)

        # Push everything left
        mode_layout.addStretch()

        main_layout.addLayout(mode_layout)

        # Set the button layout to the bottom of the main layout
        button_layout = QHBoxLayout()
        # Accept button
        self.ok_btn = QPushButton("Accept Colonies")
        self.ok_btn.clicked.connect(self.accept_colonies)
        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        button_layout.addStretch()  # centers nicely
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(cancel_btn)
        button_layout.addStretch()

        main_layout.addLayout(button_layout)

    def on_min_size_changed(self, value):
        """Handle minimum colony size slider change"""
        self.min_size_label.setText(f"{value}%")
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        total_pixels = h * w

        # convert percent → actual pixel area
        min_area = (value / 100.0) * total_pixels * 0.01
        self.colony_separator.update_parameters(min_colony_size=min_area)

    def on_threshold_changed(self, value):
        """Handle maximum colony size slider change"""
        self.threshold_label.setText(f"{value}%")

        # Map slider value to 0-255 range
        threshold = int((value / 100.0) * 255)
        self.colony_separator.update_parameters(intensity_threshold=threshold)

    def on_kernel_changed(self, value):
        """Handle kernel blur size slider change"""
        self.kernel_label.setText(f"{value}%")
        if self.image is None:
            return

        # Smaller dimensions used for consistency: maps 0–100 → 1–75 (safe range)
        kernel_size = int(value / 100.0 * 75)
        if kernel_size < 1:
            kernel_size = 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Ensure kernel is odd (OpenCV preference)
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.colony_separator.update_parameters(kernel_size=kernel_size)

    def setup_image(self):
        """Setup the image display"""
        if self.image is None:
            self.image_label.setText("No image data")
            return
        # Convert image to displayable format
        if len(self.image.shape) == 2:
            # Grayscale image
            display_image = self.image.copy()
            # Normalize to 0-255
            if display_image.max() > 255 or display_image.dtype != np.uint8:
                display_image = ((display_image - display_image.min()) /
                                 (display_image.max() - display_image.min()) * 255).astype(np.uint8)
        else:
            display_image = self.image

        self.original_image = display_image
        self.update_display()

    def update_display(self):
        if not hasattr(self, 'original_image'):
            return

        # Start with original image
        display_image = self.original_image.copy()

        # Convert to RGB for overlays
        if len(display_image.shape) == 2:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2RGB)

        # Get overlay from colony separator
        overlay = self.colony_separator.create_colony_overlay(display_image.shape[:2])

        # Check if overlay exists and is same size
        if overlay.shape[:2] == display_image.shape[:2]:
            # Blend overlay over the base image
            alpha = 0.6  # transparency factor (0–1)
            display_image = cv2.addWeighted(display_image, 1.0, overlay, alpha, 0)
            # Draw contours and centroids
            colors = [
                (255, 0, 0),  # Blue (OpenCV = BGR)
                (0, 255, 0),  # Green
                (0, 0, 255),  # Red
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
                (255, 128, 0),
                (128, 0, 255),
            ]
            # Get contour and unique contour color
            for i, colony in enumerate(self.colonies):
                cnt = colony.get("contour")
                if cnt is None:
                    continue
                color = colors[i % len(colors)]
                area = colony.get("area", 0)

                # Draw contour
                cv2.drawContours(display_image, [cnt], -1, color, 2)

                # Get the center of the contour and display area
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(
                        display_image,
                        f"{int(area)} px",
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA
                    )

        # Convert to QPixmap and display
        height, width = display_image.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def trigger_detect(self):
        """Trigger detection on the image"""
        self.update_timer.start()

    def detect_colonies(self):
        """Detect colonies automatically using Otsu thresholding"""
        if self.image is None:
            self.status_label.setText("⚠️ No image loaded. Please view an image first.")
            return
        print(f"Detecting colonies from image shape: {self.image.shape}")

        # Detect colonies using Otsu + Triangle method (openCV)
        raw_colonies = self.colony_separator.detect_colonies_otsu(self.image)
        colonies = []
        for i, colony in enumerate(raw_colonies):
            cnt = colony.get("contour")
            if cnt is None:
                continue  # safety

            # Create mask
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 1, -1)
            converted = {
                "colony_id": i + 1,
                "contour": cnt,
                "polygon": cnt.reshape(-1, 2).tolist(),
                "mask": mask,
                "area": cv2.contourArea(cnt),
                "source": "auto"
            }
            colonies.append(converted)
        if len(colonies) > 0:
            self.status_label.setText(f"✅ Detected {len(colonies)} colonies automatically.")

            # Automatically update colony list and display
            self.colonies = colonies
            self.colony_separator.detected_colonies = colonies
            self.update_colonies_list()
            self.update_display()

            # Print colony info
            for colony in colonies:
                print(f"Colony {colony['colony_id']}: Area={colony['area']:.0f}px²")
                print(f"{colony['colony_id']} ({int(colony.get('area', 0))} px)")
        else:
            self.status_label.setText("No colonies detected. Try adjusting the threshold.")


    def update_colonies_list(self):
        """Update the colony list widget with the current colonies."""
        for i in reversed(range(self.colonies_list_layout.count())):
            child = self.colonies_list_layout.itemAt(i).widget()
            if child:
                child.deleteLater()
        # Set widget for colony list
        for i, colony in enumerate(self.colonies):
            colony_widget = QWidget()
            colony_layout = QHBoxLayout(colony_widget)
            colony_layout.setContentsMargins(5, 2, 5, 2)
            label = QLabel(f"Colony {colony['colony_id']}")

            # Set Modify button widget
            delete_btn = QPushButton("Delete")
            delete_btn.setMaximumWidth(60)
            delete_btn.setStyleSheet(
                "background-color: #f44336; color: white; font-size: 10px;"
            )
            delete_btn.clicked.connect(lambda checked, idx=i: self.delete_colony(idx))

            colony_layout.addWidget(label)
            colony_layout.addWidget(delete_btn)
            self.colonies_list_layout.addWidget(colony_widget)

        self.colonies_count_label.setText(f"Colonies: {len(self.colonies)}")

    def accept_colonies(self):
        """Accept the selected colonies and emit signal"""
        params = {
            "min_size": self.min_size_slider.value(),
            "threshold": self.threshold_slider.value(),
            "kernel": self.kernel_slider.value()
        }
        self.colonies_verified.emit(self.colonies, params)
        self.accept()

    def delete_colony(self, index):
        """Delete colony at given index and update colony IDs"""
        self.save_state()
        if 0 <= index < len(self.colonies):
            del self.colonies[index]
            for i, colony in enumerate(self.colonies):
                colony["colony_id"] = i + 1

            self.update_colonies_list()
            self.update_display()

    def on_mode_changed(self, mode):
        """Set Trim/Add mode for modifying auto-detected colonies"""
        self.modify_mode = mode

    def eventFilter(self, obj, event):
        """Handle mouse events for altering auto-detected colonies"""
        if obj == self.image_label:
            if event.type() == QEvent.MouseButtonPress:
                if self.modify_mode not in ["Add", "Trim"]:
                    return False

                self.origin = event.pos()
                self.rubber_band.setGeometry(QRect(self.origin, QSize()))
                self.rubber_band.show()
                return True

            elif event.type() == QEvent.MouseMove:
                if self.rubber_band.isVisible():
                    rect = QRect(self.origin, event.pos()).normalized()
                    self.rubber_band.setGeometry(rect)
                return True

            elif event.type() == QEvent.MouseButtonRelease:
                if self.rubber_band.isVisible():
                    self.rubber_band.hide()
                    rect = QRect(self.origin, event.pos()).normalized()
                    if self.modify_mode == "Add":
                        self.add_colony_from_rect(rect)
                    elif self.modify_mode == "Trim":
                        self.trim_colonies_from_rect(rect)
                return True
        return super().eventFilter(obj, event)

    def add_colony_from_rect(self, rect: QRect):
        """Add a colony or extend colony area to auto-detected colonies"""
        self.save_state()
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return

        label_size = self.image_label.size()
        img_h, img_w = self.original_image.shape[:2]

        # Scale to image coords
        scale_x = img_w / label_size.width()
        scale_y = img_h / label_size.height()

        x1 = int(rect.left() * scale_x)
        y1 = int(rect.top() * scale_y)
        x2 = int(rect.right() * scale_x)
        y2 = int(rect.bottom() * scale_y)

        # Clamp box inside image frame and order bounds
        x1, x2 = sorted((max(0, x1), min(img_w, x2)))
        y1, y2 = sorted((max(0, y1), min(img_h, y2)))

        # Rectangle mask
        rect_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        rect_mask[y1:y2, x1:x2] = 1


        overlapping = []
        # Find overlapping colonies
        for i, colony in enumerate(self.colonies):
            mask = colony.get("mask")
            if mask is None:
                continue
            overlap = np.logical_and(mask, rect_mask)
            if np.sum(overlap) > 20:
                overlapping.append(i)

        if len(overlapping) == 0:
            # No overlaps -> new colony is created
            if len(overlapping) == 0:
                # Find contours from the drawn rectangle
                contours, _ = cv2.findContours(rect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    return
                cnt = max(contours, key=cv2.contourArea)

                new_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.drawContours(new_mask, [cnt], -1, 1, -1)

                new_colony = {
                    "colony_id": None,
                    "contour": cnt,
                    "polygon": cnt.reshape(-1, 2).tolist(),
                    "mask": new_mask,
                    "area": cv2.contourArea(cnt),
                    "source": "manual"
                }
                self.colonies.append(new_colony)

                # Reassign Colony IDs
                for i, colony in enumerate(self.colonies):
                    colony["colony_id"] = i + 1
                print("Created new colony from selection")

                self.colony_separator.detected_colonies = self.colonies
                self.update_display()
                self.update_colonies_list()
                return

        # Merge/Combine masks of all overlapping colonies + rect mask
        combined_mask = rect_mask.copy()
        for idx in overlapping:
            combined_mask = np.logical_or(combined_mask, self.colonies[idx]["mask"])
        combined_mask = combined_mask.astype(np.uint8)

        # Find merged contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        new_cnt = max(contours, key=cv2.contourArea)

        # Create clean (new) mask from contour
        new_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(new_mask, [new_cnt], -1, 1, -1)

        new_colony = {
            "colony_id": None,  # assigned later
            "contour": new_cnt,
            "polygon": new_cnt.reshape(-1, 2).tolist(),
            "mask": new_mask,
            "area": cv2.contourArea(new_cnt),
            "source": "merged" if len(overlapping) > 1 else "expanded"
        }

        # Remove old colonies (starting at end)
        for idx in sorted(overlapping, reverse=True):
            del self.colonies[idx]
        # Add merged colonies
        self.colonies.append(new_colony)

        # Reassign Colony IDs
        for i, colony in enumerate(self.colonies):
            colony["colony_id"] = i + 1
        print(f"Merged {len(overlapping)} colonies")

        # Refresh UI
        self.colony_separator.detected_colonies = self.colonies
        self.update_display()
        self.update_colonies_list()

    def trim_colonies_from_rect(self, rect: QRect):
        """Trim or remove colonies from an auto-detected colony"""
        self.save_state()
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return
        label_size = self.image_label.size()
        img_h, img_w = self.original_image.shape[:2]

        # Scale to image coords
        scale_x = img_w / label_size.width()
        scale_y = img_h / label_size.height()
        x1 = int(rect.left() * scale_x)
        y1 = int(rect.top() * scale_y)
        x2 = int(rect.right() * scale_x)
        y2 = int(rect.bottom() * scale_y)

        # Clamp box inside image frame and order bounds
        x1, x2 = sorted((max(0, x1), min(img_w, x2)))
        y1, y2 = sorted((max(0, y1), min(img_h, y2)))

        # Rectangle mask
        rect_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        rect_mask[y1:y2, x1:x2] = 1

        new_colonies = []
        removed_count = 0
        split_count = 0
        for colony in self.colonies:
            mask = colony.get("mask")
            if mask is None:
                new_colonies.append(colony)
                continue
            overlap = np.logical_and(mask, rect_mask)

            # No overlap, keep colonies
            if np.sum(overlap) <= 20:
                new_colonies.append(colony)
                continue

            # Clip colony masks with drawn rectangle
            new_mask = np.logical_and(mask, np.logical_not(rect_mask)).astype(np.uint8)

            # Fully remove colony
            if np.sum(new_mask) < 10:
                removed_count += 1
                continue

            # Detect fragment colonies from trim
            contours, _ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                removed_count += 1
                continue
            # Create colonies from fragments
            valid_fragments = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Filter tiny noise
                if area < 20:
                    continue
                frag_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.drawContours(frag_mask, [cnt], -1, 1, -1)

                new_colony = {
                    "colony_id": None,  # assigned later
                    "contour": cnt,
                    "polygon": cnt.reshape(-1, 2).tolist(),
                    "mask": frag_mask,
                    "area": area,
                    "source": "split"
                }
                valid_fragments.append(new_colony)
            if len(valid_fragments) == 0:
                removed_count += 1
                continue
            if len(valid_fragments) == 1:
                # Add new altered colonies (fragments) to colony list
                new_colonies.append(valid_fragments[0])
            else:
                # Add split fragments to colony list
                split_count += 1
                new_colonies.extend(valid_fragments)

        # Replace colonies
        self.colonies = new_colonies

        # Reassign IDs cleanly
        for i, colony in enumerate(self.colonies):
            colony["colony_id"] = i + 1

        print(f"Trim complete: {removed_count} removed, {split_count} split.")
        self.status_label.setText(
            f"✂️ Trimmed: {removed_count} removed, {split_count} split into multiple colonies."
        )

        self.colony_separator.detected_colonies = self.colonies
        self.update_display()
        self.update_colonies_list()

    def save_state(self):
        """Save current colony state for undo."""
        print(f"SAVING STATE: {len(self.colonies)} colonies")
        print(f"UNDO STACK SIZE BEFORE: {len(self.undo_stack)}")
        self.undo_stack.append(copy.deepcopy(self.colonies))
        print(f"UNDO STACK SIZE AFTER: {len(self.undo_stack)}")

        # Keep only last 5 actions
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)

        # New action invalidates redo history
        self.redo_stack.clear()

    def restore_state(self, state):
        """Restore colony state and refresh UI."""
        self.colonies = copy.deepcopy(state)

        # Reassign IDs
        for i, colony in enumerate(self.colonies):
            colony["colony_id"] = i + 1

        self.colony_separator.detected_colonies = self.colonies

        self.update_display()
        self.update_colonies_list()

    def undo(self):
        """Undo last action."""
        print("UNDO CALLED")
        print(f"UNDO STACK SIZE: {len(self.undo_stack)}")
        print(f"REDO STACK SIZE: {len(self.redo_stack)}")
        if not self.undo_stack:
            self.status_label.setText("Nothing to undo.")
            return

        # Save current state for redo
        self.redo_stack.append(copy.deepcopy(self.colonies))

        # Restore previous state
        previous_state = self.undo_stack.pop()
        self.restore_state(previous_state)

        self.status_label.setText("↩ Undo")

    def redo(self):
        """Redo last undone action."""
        print("REDO CALLED")
        print(f"UNDO STACK SIZE: {len(self.undo_stack)}")
        print(f"REDO STACK SIZE: {len(self.redo_stack)}")
        if not self.redo_stack:
            self.status_label.setText("Nothing to redo.")
            return

        # Save current state back into undo
        self.undo_stack.append(copy.deepcopy(self.colonies))

        # Restore redo state
        next_state = self.redo_stack.pop()
        self.restore_state(next_state)

        self.status_label.setText("↪ Redo")
