from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QWidget, QScrollArea, QFrame)
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
import numpy as np
import cv2


class ColonyROISelector(QDialog):
    """Dialog for selecting multiple biofilm colonies using polygon ROI"""

    colonies_selected = Signal(list)  # Emits list of colony masks

    def __init__(self, image_data, existing_colonies=None, parent=None):
        super().__init__(parent)

        print(
            f"DEBUG: ColonyROISelector received {len(existing_colonies) if existing_colonies else 0} existing colonies")

        self.image_data = image_data
        self.existing_colonies = existing_colonies or []
        self.current_colonies = []

        # Convert existing colonies
        if self.existing_colonies:
            print(f"DEBUG: Converting {len(self.existing_colonies)} existing colonies")
            for i, colony in enumerate(self.existing_colonies):
                print(f"DEBUG: Converting colony {i + 1}: {colony.keys()}")
                points = colony['polygon']
                mask = self.create_mask_from_polygon(points)
                # Converts polygon to contour (OpenCV format)
                contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                area = cv2.contourArea(contour)
                converted_colony = {
                    "colony_id": len(self.current_colonies) + 1,
                    "polygon": points,
                    "contour": contour,
                    "mask": mask,
                    "area": area,
                    "source": "manual"
                }
                self.current_colonies.append(converted_colony)
                print(f"DEBUG: Added colony {converted_colony['colony_id']} with {len(colony['polygon'])} points")

        print(f"DEBUG: Dialog now has {len(self.current_colonies)} colonies to display")

        self.current_polygon = []
        self.drawing_mode = False

        self.setWindowTitle("Colony ROI Selector")
        self.setModal(True)
        self.resize(1000, 700)

        self.init_ui()
        self.setup_image()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QHBoxLayout(self)

        # Left side - Image display
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Image label
        self.image_label = QLabel()
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.mousePressEvent = self.on_image_click

        # Scroll area for large images
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        left_layout.addWidget(scroll_area)

        # Instructions
        instructions = QLabel("Click on image to draw polygon around biofilm colony")
        instructions.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        left_layout.addWidget(instructions)

        layout.addWidget(left_widget, 2)  # 2/3 of space

        # Right side - Controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_widget.setMaximumWidth(300)

        # Title
        title = QLabel("Colony Selection")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2196F3;")
        right_layout.addWidget(title)

        # Current polygon controls
        polygon_group = QFrame()
        polygon_group.setFrameStyle(QFrame.Box)
        polygon_layout = QVBoxLayout(polygon_group)

        polygon_layout.addWidget(QLabel("Current Polygon:"))

        self.polygon_points_label = QLabel("Points: 0")
        polygon_layout.addWidget(self.polygon_points_label)

        polygon_buttons = QHBoxLayout()

        self.finish_polygon_btn = QPushButton("Finish Polygon")
        self.finish_polygon_btn.clicked.connect(self.finish_current_polygon)
        self.finish_polygon_btn.setEnabled(False)
        self.finish_polygon_btn.setStyleSheet("background-color: #4CAF50; color: white;")

        self.cancel_polygon_btn = QPushButton("Cancel Polygon")
        self.cancel_polygon_btn.clicked.connect(self.cancel_current_polygon)
        self.cancel_polygon_btn.setEnabled(False)

        polygon_buttons.addWidget(self.finish_polygon_btn)
        polygon_buttons.addWidget(self.cancel_polygon_btn)
        polygon_layout.addLayout(polygon_buttons)

        right_layout.addWidget(polygon_group)

        # Selected colonies list
        colonies_group = QFrame()
        colonies_group.setFrameStyle(QFrame.Box)
        colonies_layout = QVBoxLayout(colonies_group)

        colonies_layout.addWidget(QLabel("Selected Colonies:"))

        self.colonies_count_label = QLabel("Colonies: 0")
        self.colonies_count_label.setStyleSheet("font-weight: bold;")
        colonies_layout.addWidget(self.colonies_count_label)

        # Colony list
        self.colonies_list_widget = QWidget()
        self.colonies_list_layout = QVBoxLayout(self.colonies_list_widget)

        colonies_scroll = QScrollArea()
        colonies_scroll.setWidget(self.colonies_list_widget)
        colonies_scroll.setWidgetResizable(True)
        colonies_scroll.setMaximumHeight(200)
        colonies_layout.addWidget(colonies_scroll)

        # Clear all button
        clear_all_btn = QPushButton("Clear All Colonies")
        clear_all_btn.clicked.connect(self.clear_all_colonies)
        clear_all_btn.setStyleSheet("background-color: #f44336; color: white;")
        colonies_layout.addWidget(clear_all_btn)

        right_layout.addWidget(colonies_group)

        # Main dialog buttons
        right_layout.addStretch()

        main_buttons = QVBoxLayout()

        self.ok_btn = QPushButton("Accept Colonies")
        self.ok_btn.clicked.connect(self.accept_colonies)
        self.ok_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        main_buttons.addWidget(self.ok_btn)
        main_buttons.addWidget(cancel_btn)
        right_layout.addLayout(main_buttons)

        layout.addWidget(right_widget, 1)  # 1/3 of space

    def setup_image(self):
        """Setup the image display"""
        if self.image_data is None:
            self.image_label.setText("No image data")
            return

        # Convert image to displayable format
        if len(self.image_data.shape) == 2:
            # Grayscale image
            display_image = self.image_data.copy()
            # Normalize to 0-255
            if display_image.max() > 255 or display_image.dtype != np.uint8:
                display_image = ((display_image - display_image.min()) /
                                 (display_image.max() - display_image.min()) * 255).astype(np.uint8)
        else:
            display_image = self.image_data

        self.original_image = display_image
        self.update_display()

    def update_display(self):
        """Update the image display with overlays"""
        if not hasattr(self, 'original_image'):
            return

        # Start with original image
        display_image = self.original_image.copy()

        # Convert to RGB for overlays
        if len(display_image.shape) == 2:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2RGB)

        # Draw existing colonies
        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]

        # Draw completed colonies
        for i, colony in enumerate(self.current_colonies):
            color = colors[i % len(colors)]
            points = np.array(colony['polygon'], dtype=np.int32)
            cv2.polylines(display_image, [points], True, color, 3)

            # Draw colony ID
            centroid = np.mean(points, axis=0).astype(int)
            cv2.putText(display_image, f"C{i + 1}", tuple(centroid),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Draw current polygon being drawn
        if len(self.current_polygon) > 1:
            points = np.array(self.current_polygon, dtype=np.int32)
            cv2.polylines(display_image, [points], False, (0, 255, 0), 2)

            # Draw points
            for point in points:
                cv2.circle(display_image, tuple(point), 4, (255, 0, 0), -1)
        elif len(self.current_polygon) == 1:
            point = self.current_polygon[0]
            cv2.circle(display_image, tuple(point), 4, (255, 0, 0), -1)

        # Convert to QPixmap and display
        height, width = display_image.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def on_image_click(self, event):
        """Handle mouse clicks on the image"""
        # Get click position
        click_pos = event.pos()

        # Convert to image coordinates
        if self.image_label.pixmap():
            label_size = self.image_label.size()
            pixmap_size = self.image_label.pixmap().size()

            # Calculate scaling
            scale_x = pixmap_size.width() / label_size.width()
            scale_y = pixmap_size.height() / label_size.height()

            # Convert coordinates
            image_x = int(click_pos.x() * scale_x)
            image_y = int(click_pos.y() * scale_y)

            # Add point to current polygon
            self.current_polygon.append([image_x, image_y])

            # Update UI
            self.update_polygon_controls()
            self.update_display()

    def update_polygon_controls(self):
        """Update polygon control buttons and labels"""
        num_points = len(self.current_polygon)
        self.polygon_points_label.setText(f"Points: {num_points}")

        # Enable buttons if we have enough points
        self.finish_polygon_btn.setEnabled(num_points >= 3)
        self.cancel_polygon_btn.setEnabled(num_points > 0)

    def finish_current_polygon(self):
        """Finish the current polygon and add as colony"""
        if len(self.current_polygon) < 3:
            return

        # Create colony data
        points = self.current_polygon.copy()
        mask = self.create_mask_from_polygon(points)
        contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        area = cv2.contourArea(contour)

        colony_data = {
            "colony_id": len(self.current_colonies) + 1,
            "polygon": points,
            "contour": contour,
            "mask": mask,
            "area": area,
            "source": "manual"
        }

        self.current_colonies.append(colony_data)
        self.current_polygon = []

        self.update_polygon_controls()
        self.update_colonies_list()
        self.update_display()

    def cancel_current_polygon(self):
        """Cancel the current polygon"""
        self.current_polygon = []
        self.update_polygon_controls()
        self.update_display()

    def create_mask_from_polygon(self, polygon_points):
        """Create binary mask from polygon points"""
        if not hasattr(self, 'original_image'):
            return None

        mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        points = np.array(polygon_points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)
        return mask

    def update_colonies_list(self):
        """Update the colonies list display"""
        # Clear existing widgets
        for i in reversed(range(self.colonies_list_layout.count())):
            child = self.colonies_list_layout.itemAt(i).widget()
            if child:
                child.deleteLater()

        # Add colony items
        for i, colony in enumerate(self.current_colonies):
            colony_widget = QWidget()
            colony_layout = QHBoxLayout(colony_widget)
            colony_layout.setContentsMargins(5, 2, 5, 2)

            # Colony info
            info_label = QLabel(f"Colony {colony['colony_id']}")
            info_label.setStyleSheet("font-weight: bold;")

            # Delete button
            delete_btn = QPushButton("Delete")
            delete_btn.setMaximumWidth(60)
            delete_btn.setStyleSheet("background-color: #f44336; color: white; font-size: 10px;")
            delete_btn.clicked.connect(lambda checked, idx=i: self.delete_colony(idx))

            colony_layout.addWidget(info_label)
            colony_layout.addWidget(delete_btn)

            self.colonies_list_layout.addWidget(colony_widget)

        # Update count
        self.colonies_count_label.setText(f"Colonies: {len(self.current_colonies)}")

        # Enable/disable OK button
        self.ok_btn.setEnabled(len(self.current_colonies) > 0)

    def delete_colony(self, index):
        """Delete a specific colony"""
        if 0 <= index < len(self.current_colonies):
            del self.current_colonies[index]

            # Update colony IDs
            for i, colony in enumerate(self.current_colonies):
                colony['colony_id'] = i + 1

            self.update_colonies_list()
            self.update_display()

    def clear_all_colonies(self):
        """Clear all selected colonies"""
        self.current_colonies = []
        self.current_polygon = []
        self.update_colonies_list()
        self.update_polygon_controls()
        self.update_display()

    def accept_colonies(self):
        """Accept the selected colonies and close dialog"""
        if self.current_colonies:
            self.colonies_selected.emit(self.current_colonies)
            self.accept()

    def get_selected_colonies(self):
        """Get the list of selected colonies"""
        return self.current_colonies