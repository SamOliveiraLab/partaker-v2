import numpy as np
from PySide6.QtCore import Qt, QPointF, Signal, QEvent
from PySide6.QtGui import QPixmap, QPen, QImage, QPolygonF, QColor
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
)

from nd2_analyzer.data.appstate import ApplicationState
from nd2_analyzer.data.image_data import ImageData

from pubsub import pub

import cv2


class PolygonROISelector(QDialog):
    roi_selected = Signal(object)

    def __init__(self):
        super().__init__()

        t, p, c = ApplicationState.get_instance().view_index
        curr_image = ImageData.get_instance().get(t, p, 0)  # Always use 0

        # Convert numpy array to QImage/QPixmap
        if isinstance(curr_image, np.ndarray):
            if curr_image.ndim == 2:  # Grayscale
                height, width = curr_image.shape
                # Scale to 0-255 if not already
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
        self.setWindowTitle("Polygon ROI Selector")
        # self.resize(800, 800)

        # Main layout
        main_layout = QVBoxLayout(self)

        # Instructions label
        instructions = QLabel(
            "Click to add points to the polygon. Double-click to complete."
        )
        main_layout.addWidget(instructions)

        # Graphics view for the image and ROI drawing
        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        main_layout.addWidget(self.view)

        # Disable scrollbars to avoid coordinate offset
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Ensure no extra margins
        self.view.setContentsMargins(0, 0, 0, 0)

        # Add the image to the scene
        self.image_item = QGraphicsPixmapItem(self.pixmap)
        self.image_item.setPos(0, 0)
        self.scene.addItem(self.image_item)
        self.scene.setSceneRect(self.pixmap.rect())
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        # Button layout
        button_layout = QHBoxLayout()

        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_polygon)
        button_layout.addWidget(self.reset_button)

        # Complete button
        self.complete_button = QPushButton("Complete Polygon")
        self.complete_button.clicked.connect(self.complete_polygon)
        button_layout.addWidget(self.complete_button)

        # Accept button
        self.accept_button = QPushButton("Accept ROI")
        self.accept_button.clicked.connect(self.accept_roi)
        self.accept_button.setEnabled(False)
        button_layout.addWidget(self.accept_button)

        main_layout.addLayout(button_layout)

        # Initialize polygon drawing
        self.points = []
        self.polygon = None
        self.drawing = True
        self.mask = None

        # Connect mouse events
        # Resize window

    def mousePressEvent(self, event):
        if not self.drawing:
            return

        # Convert view coordinates to scene coordinates
        view_pos = self.view.mapFromParent(event.pos())
        scene_pos = self.view.mapToScene(view_pos)
        print(event.pos(), view_pos, scene_pos)
        # scene_pos = self.mapTo(self.view, event.pos()

        # Validate coordinates
        if not (
            0 <= scene_pos.x() < self.pixmap.width()
            and 0 <= scene_pos.y() < self.pixmap.height()
        ):
            return

        # Add point to the polygon
        self.points.append((scene_pos.x(), scene_pos.y()))

        # Draw the polygon
        self.draw_polygon()

        # Check if double-click to complete the polygon
        if event.type() == QEvent.Type.MouseButtonDblClick:
            self.complete_polygon()

    def draw_polygon(self):
        # Remove existing polygon if any
        if self.polygon is not None:
            self.scene.removeItem(self.polygon)

        # Create a new polygon
        if len(self.points) > 1:
            polygon = QPolygonF([QPointF(x, y) for x, y in self.points])
            self.polygon = self.scene.addPolygon(polygon, QPen(Qt.red, 2))

    def complete_polygon(self):
        if len(self.points) < 3:
            return

        # Complete the polygon by connecting the last point to the first
        self.drawing = False

        # Close the polygon
        if self.points[0] != self.points[-1]:
            self.points.append(self.points[0])

        # Draw the final polygon
        self.draw_polygon()

        # Fill the polygon with semi-transparent color
        if self.polygon is not None:
            self.polygon.setBrush(QColor(255, 0, 0, 50))

        # Enable accept button
        self.accept_button.setEnabled(True)

    def reset_polygon(self):
        # Remove existing polygon
        if self.polygon is not None:
            self.scene.removeItem(self.polygon)
            self.polygon = None

        # Reset points and enable drawing
        self.points = []
        self.drawing = True
        self.accept_button.setEnabled(False)

    def create_mask(self):
        # Create a binary mask from the polygon
        height, width = self.image_shape
        self.mask = np.zeros((height, width), dtype=np.uint8)

        # Convert scene coordinates to image coordinates
        image_points = []
        for x, y in self.points:
            # Convert from scene coordinates to image coordinates
            ix = int(x)
            iy = int(y)
            # Ensure coordinates are within image bounds
            ix = max(0, min(ix, width - 1))
            iy = max(0, min(iy, height - 1))
            image_points.append((ix, iy))
        points_array = np.array(image_points, dtype=np.int32)

        # Create polygon mask using OpenCV
        cv2.fillPoly(self.mask, [points_array], 255)
        # Covert mask to bool
        self.mask = self.mask.astype(np.uint8)

    def accept_roi(self):
        # Create binary mask from the polygon
        self.create_mask()
        # Emit the signal with the mask
        self.roi_selected.emit(self.mask)

        # TODO: Directly send via pubsubs
        # eg. self.application_state.image_data.segmentation_cache.set_binary_mask(self.roi_mask)
        ApplicationState.get_instance()
        pub.sendMessage("roi_selected", mask=self.mask)

        self.accept()  # This will close the dialog

    def resizeEvent(self, event):
        # Make sure the view fits the image when resizing
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)
