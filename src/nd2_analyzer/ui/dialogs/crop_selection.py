import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPixmap, QPen, QImage, QColor, QCursor
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSpinBox,
    QGraphicsRectItem,
)

from nd2_analyzer.data.appstate import ApplicationState
from nd2_analyzer.data.image_data import ImageData
from pubsub import pub


class CropSelector(QDialog):
    crop_selected = Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crop Selector")
        self.dragging = False
        self.resizing = False
        self.resize_mode = None
        self.drag_start = QPointF()
        self.start_rect = QRectF()
        self.handle_size = 8  # Size of resize handles in pixels

        # Get current image
        t, p, c = ApplicationState.get_instance().view_index
        curr_image = ImageData.get_instance().get(t, p, 0)  # Always use 0

        # Convert numpy array to QImage/QPixmap
        if isinstance(curr_image, np.ndarray):
            if curr_image.ndim == 2:  # Grayscale
                height, width = curr_image.shape
                if curr_image.dtype != np.uint8:
                    curr_image = (
                        (curr_image - np.min(curr_image))
                        / (np.max(curr_image) - np.min(curr_image))
                        * 255
                    ).astype(np.uint8)
                qimg = QImage(
                    curr_image.data, width, height, width, QImage.Format_Grayscale8
                )
            else:  # RGB
                height, width, channels = curr_image.shape
                qimg = QImage(
                    curr_image.data,
                    width,
                    height,
                    width * channels,
                    QImage.Format_RGB888,
                )
            self.pixmap = QPixmap.fromImage(qimg)
        else:  # Assume it's a file path
            self.pixmap = QPixmap(curr_image)

        self.image_shape = (
            (curr_image.shape[0], curr_image.shape[1])
            if isinstance(curr_image, np.ndarray)
            else (self.pixmap.height(), self.pixmap.width())
        )

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout(self)

        # Graphics view for the image and crop rectangle
        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.view.mousePressEvent = self.view_mousePressEvent
        self.view.mouseMoveEvent = self.view_mouseMoveEvent
        self.view.mouseReleaseEvent = self.view_mouseReleaseEvent
        main_layout.addWidget(self.view)

        # Disable scrollbars
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setContentsMargins(0, 0, 0, 0)

        # Add the image to the scene
        self.image_item = QGraphicsPixmapItem(self.pixmap)
        self.scene.addItem(self.image_item)
        self.scene.setSceneRect(self.pixmap.rect())
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        # Crop rectangle
        self.rect_item = QGraphicsRectItem(0, 0, 0, 0)
        self.rect_item.setPen(QPen(QColor(255, 0, 0, 200), 2))
        self.rect_item.setBrush(QColor(255, 0, 0, 50))
        self.scene.addItem(self.rect_item)

        # Coordinate controls
        controls_layout = QHBoxLayout()

        # X controls
        x_layout = QVBoxLayout()
        x_layout.addWidget(QLabel("X:"))
        self.x_spin = QSpinBox()
        self.x_spin.setRange(0, self.image_shape[1])
        self.x_spin.valueChanged.connect(self.update_rect_from_spinners)
        x_layout.addWidget(self.x_spin)
        controls_layout.addLayout(x_layout)

        # Y controls
        y_layout = QVBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.y_spin = QSpinBox()
        self.y_spin.setRange(0, self.image_shape[0])
        self.y_spin.valueChanged.connect(self.update_rect_from_spinners)
        y_layout.addWidget(self.y_spin)
        controls_layout.addLayout(y_layout)

        # Width controls
        w_layout = QVBoxLayout()
        w_layout.addWidget(QLabel("Width:"))
        self.w_spin = QSpinBox()
        self.w_spin.setRange(1, self.image_shape[1])
        self.w_spin.valueChanged.connect(self.update_rect_from_spinners)
        w_layout.addWidget(self.w_spin)
        controls_layout.addLayout(w_layout)

        # Height controls
        h_layout = QVBoxLayout()
        h_layout.addWidget(QLabel("Height:"))
        self.h_spin = QSpinBox()
        self.h_spin.setRange(1, self.image_shape[0])
        self.h_spin.valueChanged.connect(self.update_rect_from_spinners)
        h_layout.addWidget(self.h_spin)
        controls_layout.addLayout(h_layout)

        main_layout.addLayout(controls_layout)

        # Dimensions label
        self.dimensions_label = QLabel("Dimensions: 0x0")
        main_layout.addWidget(self.dimensions_label)

        # Accept button
        self.accept_button = QPushButton("Accept Crop")
        self.accept_button.clicked.connect(self.accept_crop)
        main_layout.addWidget(self.accept_button)

        # Initialize with default rectangle (50% of image)
        w, h = self.image_shape[1] // 2, self.image_shape[0] // 2
        x, y = (self.image_shape[1] - w) // 2, (self.image_shape[0] - h) // 2
        self.rect_item.setRect(QRectF(x, y, w, h))
        self.update_spinners_from_rect()

    def update_rect_from_spinners(self):
        x = self.x_spin.value()
        y = self.y_spin.value()
        w = self.w_spin.value()
        h = self.h_spin.value()

        # Ensure the rectangle stays within bounds
        x = max(0, min(x, self.image_shape[1] - 1))
        y = max(0, min(y, self.image_shape[0] - 1))
        w = max(1, min(w, self.image_shape[1] - x))
        h = max(1, min(h, self.image_shape[0] - y))

        self.rect_item.setRect(QRectF(x, y, w, h))
        self.dimensions_label.setText(f"Dimensions: {w}x{h}")

    def update_spinners_from_rect(self):
        rect = self.rect_item.rect()

        # Convert to scene coordinates
        scene_rect = self.rect_item.mapRectToScene(rect)
        x = max(0, min(int(scene_rect.x()), self.image_shape[1] - 1))
        y = max(0, min(int(scene_rect.y()), self.image_shape[0] - 1))
        w = max(1, min(int(scene_rect.width()), self.image_shape[1] - x))
        h = max(1, min(int(scene_rect.height()), self.image_shape[0] - y))

        # Update rect if needed to stay within bounds
        if (
            x != scene_rect.x()
            or y != scene_rect.y()
            or w != scene_rect.width()
            or h != scene_rect.height()
        ):
            self.rect_item.setRect(QRectF(x, y, w, h))

        self.x_spin.blockSignals(True)
        self.y_spin.blockSignals(True)
        self.w_spin.blockSignals(True)
        self.h_spin.blockSignals(True)

        self.x_spin.setValue(x)
        self.y_spin.setValue(y)
        self.w_spin.setValue(w)
        self.h_spin.setValue(h)

        self.x_spin.blockSignals(False)
        self.y_spin.blockSignals(False)
        self.w_spin.blockSignals(False)
        self.h_spin.blockSignals(False)

        self.dimensions_label.setText(f"Dimensions: {w}x{h}")

    def get_cursor_position(self, event):
        view_pos = event.pos()
        scene_pos = self.view.mapToScene(view_pos)
        return scene_pos

    def get_resize_mode(self, pos):
        """Determine if position is on a resize handle of the rectangle"""
        rect = self.rect_item.rect()
        scene_rect = self.rect_item.mapRectToScene(rect)

        # Distance threshold for handle detection
        threshold = self.handle_size

        # Check for corner handles first (they take precedence)
        if (
            abs(pos.x() - scene_rect.left()) < threshold
            and abs(pos.y() - scene_rect.top()) < threshold
        ):
            return "topleft"
        elif (
            abs(pos.x() - scene_rect.right()) < threshold
            and abs(pos.y() - scene_rect.top()) < threshold
        ):
            return "topright"
        elif (
            abs(pos.x() - scene_rect.left()) < threshold
            and abs(pos.y() - scene_rect.bottom()) < threshold
        ):
            return "bottomleft"
        elif (
            abs(pos.x() - scene_rect.right()) < threshold
            and abs(pos.y() - scene_rect.bottom()) < threshold
        ):
            return "bottomright"

        # Then check for edge handles
        elif abs(pos.x() - scene_rect.left()) < threshold:
            return "left"
        elif abs(pos.x() - scene_rect.right()) < threshold:
            return "right"
        elif abs(pos.y() - scene_rect.top()) < threshold:
            return "top"
        elif abs(pos.y() - scene_rect.bottom()) < threshold:
            return "bottom"

        # Finally, check if inside the rectangle for moving
        elif scene_rect.contains(pos):
            return "move"

        return None

    def update_cursor(self, event):
        """Update cursor shape based on position"""
        pos = self.get_cursor_position(event)
        mode = self.get_resize_mode(pos)

        if mode == "topleft" or mode == "bottomright":
            self.view.setCursor(Qt.SizeFDiagCursor)
        elif mode == "topright" or mode == "bottomleft":
            self.view.setCursor(Qt.SizeBDiagCursor)
        elif mode == "left" or mode == "right":
            self.view.setCursor(Qt.SizeHorCursor)
        elif mode == "top" or mode == "bottom":
            self.view.setCursor(Qt.SizeVerCursor)
        elif mode == "move":
            self.view.setCursor(Qt.SizeAllCursor)
        else:
            self.view.setCursor(Qt.ArrowCursor)

    def view_mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.get_cursor_position(event)
            self.resize_mode = self.get_resize_mode(pos)

            if self.resize_mode:
                self.dragging = True
                self.drag_start = pos
                self.start_rect = self.rect_item.rect()

    def view_mouseMoveEvent(self, event):
        if not self.dragging:
            # Just update cursor when not dragging
            self.update_cursor(event)
            return

        # Handle dragging based on resize mode
        pos = self.get_cursor_position(event)
        dx = pos.x() - self.drag_start.x()
        dy = pos.y() - self.drag_start.y()

        new_rect = QRectF(self.start_rect)

        if self.resize_mode == "move":
            # Move the entire rectangle
            new_x = self.start_rect.x() + dx
            new_y = self.start_rect.y() + dy

            # Ensure the rectangle stays within image bounds
            new_x = max(0, min(new_x, self.image_shape[1] - self.start_rect.width()))
            new_y = max(0, min(new_y, self.image_shape[0] - self.start_rect.height()))

            new_rect.moveTo(new_x, new_y)

        else:
            # Resize the rectangle
            if "left" in self.resize_mode:
                new_rect.setLeft(
                    min(self.start_rect.right() - 1, self.start_rect.left() + dx)
                )
            if "right" in self.resize_mode:
                new_rect.setRight(
                    max(self.start_rect.left() + 1, self.start_rect.right() + dx)
                )
            if "top" in self.resize_mode:
                new_rect.setTop(
                    min(self.start_rect.bottom() - 1, self.start_rect.top() + dy)
                )
            if "bottom" in self.resize_mode:
                new_rect.setBottom(
                    max(self.start_rect.top() + 1, self.start_rect.bottom() + dy)
                )

        # Ensure rectangle stays within image bounds
        if new_rect.left() < 0:
            new_rect.setLeft(0)
        if new_rect.top() < 0:
            new_rect.setTop(0)
        if new_rect.right() > self.image_shape[1]:
            new_rect.setRight(self.image_shape[1])
        if new_rect.bottom() > self.image_shape[0]:
            new_rect.setBottom(self.image_shape[0])

        # Set the new rectangle
        self.rect_item.setRect(new_rect)
        self.update_spinners_from_rect()

    def view_mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            self.resize_mode = None

    def accept_crop(self):
        rect = self.rect_item.rect()
        crop_coords = (
            int(rect.x()),
            int(rect.y()),
            int(rect.width()),
            int(rect.height()),
        )

        # Emit signal with crop coordinates
        self.crop_selected.emit(crop_coords)

        # Send via pubsub
        pub.sendMessage("crop_selected", coords=crop_coords)

        self.accept()

    def resizeEvent(self, event):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)
