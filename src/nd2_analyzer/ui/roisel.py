# import sys
# import numpy as np
# from PySide6.QtWidgets import (
#     QApplication,
#     QWidget,
#     QVBoxLayout,
#     QHBoxLayout,
#     QPushButton,
#     QLabel,
# )
# from PySide6.QtGui import QImage, QPixmap, QPainter, QPen
# from PySide6.QtCore import Qt, QRect, QSize, Signal

# import cv2

# class ROIWidget(QWidget):
#     # Signal will emit the binary mask as a NumPy array
#     maskCreated = Signal(object)

#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.image = None  # Will hold the QPixmap built from the NumPy array.
#         self.roi = None    # QRect storing the ROI.
#         self.startPoint = None  # Starting point of the ROI paint.
#         self.setMinimumSize(200, 200)

#     def setImage(self, np_image):
#         max_value = np.iinfo(np_image.dtype).max
#         # np_image = cv2.normalize(0, max_value, np_image)
#         np_image = cv2.normalize(np_image, None, 0, 255, cv2.NORM_MINMAX)
#         np_image = np_image.astype(np.uint8)

#         """Load an image from a NumPy array and update the widget."""
#         if np_image.ndim == 2:  # Grayscale image
#             height, width = np_image.shape
#             qimage = QImage(np_image.data, width, height, width, QImage.Format_Grayscale8)
#         elif np_image.ndim == 3 and np_image.shape[2] == 3:  # Color image
#             height, width, _ = np_image.shape
#             bytesPerLine = 3 * width
#             qimage = QImage(np_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
#             qimage = qimage.rgbSwapped()  # Convert from RGB to Qt's expected format
#         else:
#             raise ValueError("Unsupported numpy image format")
#         self.image = QPixmap.fromImage(qimage)
#         self.setFixedSize(self.image.size())
#         self.update()

#     def paintEvent(self, event):
#         painter = QPainter(self)
#         if self.image:
#             painter.drawPixmap(0, 0, self.image)
#         if self.roi:
#             pen = QPen(Qt.red, 2, Qt.DashLine)
#             painter.setPen(pen)
#             painter.drawRect(self.roi)

#     def mousePressEvent(self, event):
#         # In Qt6, use event.position() which returns a QPointF
#         self.startPoint = event.position().toPoint() if hasattr(event, "position") else event.pos()
#         self.roi = QRect(self.startPoint, QSize())
#         self.update()

#     def mouseMoveEvent(self, event):
#         if self.startPoint:
#             currentPos = event.position().toPoint() if hasattr(event, "position") else event.pos()
#             self.roi = QRect(self.startPoint, currentPos).normalized()
#             self.update()

#     def mouseReleaseEvent(self, event):
#         if self.startPoint:
#             currentPos = event.position().toPoint() if hasattr(event, "position") else event.pos()
#             self.roi = QRect(self.startPoint, currentPos).normalized()
#             self.createBinaryMask()
#             self.startPoint = None
#             self.update()

#     def createBinaryMask(self):
#         """Convert the drawn ROI into a binary mask and emit it as a NumPy array."""
#         if not self.image or not self.roi:
#             return

#         # Create a new image for the mask using an 8-bit indexed format.
#         mask_qimage = QImage(self.image.size(), QImage.Format_Indexed8)
#         mask_qimage.setColorCount(2)
#         mask_qimage.setColor(0, 0xFF000000)
#         mask_qimage.setColor(1, 0xFFFFFFFF)
#         mask_qimage.fill(0)  # Start with a completely black image.

#         painter = QPainter(mask_qimage)
#         painter.fillRect(self.roi, Qt.white)  # Mark the ROI area as white.
#         painter.end()

#         # Convert QImage to a NumPy array.
#         ptr = mask_qimage.bits()
#         ptr.setsize(mask_qimage.byteCount())
#         mask_array = np.array(ptr).reshape(mask_qimage.height(), mask_qimage.width())

#         # Emit the mask.
#         self.maskCreated.emit(mask_array)


import sys
import numpy as np

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QLabel)
from PySide6.QtGui import QPixmap, QPen, QImage, QPolygonF, QColor
from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QEvent
import numpy as np
import cv2


class PolygonROISelector(QDialog):
    roi_selected = Signal(object)

    def __init__(self, image_data):
        super().__init__()

        # Convert numpy array to QImage/QPixmap
        if isinstance(image_data, np.ndarray):
            if image_data.ndim == 2:  # Grayscale
                height, width = image_data.shape
                # Scale to 0-255 if not already
                if image_data.dtype != np.uint8:
                    image_data = ((image_data -
                                   np.min(image_data)) /
                                  (np.max(image_data) -
                                   np.min(image_data)) *
                                  255).astype(np.uint8)
                qimg = QImage(image_data.data, width, height,
                              width, QImage.Format_Grayscale8)
            else:  # RGB
                height, width, channels = image_data.shape
                qimg = QImage(
                    image_data.data,
                    width,
                    height,
                    width * channels,
                    QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(qimg)
        else:  # Assume it's a file path
            self.pixmap = QPixmap(image_data)

        self.image_shape = (
            image_data.shape[0],
            image_data.shape[1]) if isinstance(
            image_data,
            np.ndarray) else (
            self.pixmap.height(),
            self.pixmap.width())

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Polygon ROI Selector')

        # Main layout
        main_layout = QVBoxLayout(self)

        # Instructions label
        instructions = QLabel(
            "Click to add points to the polygon. Double-click to complete.")
        main_layout.addWidget(instructions)

        # Graphics view for the image and ROI drawing
        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        main_layout.addWidget(self.view)

        # Add the image to the scene
        self.image_item = QGraphicsPixmapItem(self.pixmap)
        self.scene.addItem(self.image_item)
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
        self.view.mousePressEvent = self.mouse_press_event

        # Resize window
        self.resize(800, 600)

    def mouse_press_event(self, event):
        if not self.drawing:
            return

        # Convert view coordinates to scene coordinates
        scene_pos = self.view.mapToScene(event.pos())

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

        # Create polygon mask using OpenCV
        import cv2
        points_array = np.array(image_points, dtype=np.int32)
        cv2.fillPoly(self.mask, [points_array], 255)

    def get_mask(self):
        return self.mask

    def accept_roi(self):
        # Create binary mask from the polygon
        self.create_mask()
        # Emit the signal with the mask
        self.roi_selected.emit(self.mask)
        self.accept()  # This will close the dialog

    def resizeEvent(self, event):
        # Make sure the view fits the image when resizing
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)
