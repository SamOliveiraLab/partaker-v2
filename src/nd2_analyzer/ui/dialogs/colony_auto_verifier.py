"""Batch automatic colony calibration and frame-by-frame verification dialog.

* position and time-range selection
* optional frame-zero exclusion
* lazy automatic detection for each selected frame
* a fitted, zoomable image viewer
* time navigation, playback, and representative thumbnails
* per-frame Add / Trim / Delete edits that survive frame navigation
* undo and reset-current-frame controls
* reviewed / edited / automatic frame status
* batch output keyed by ``(position, time, channel)``
* export of one overlay image per accepted frame and one GIF per position

The dialog deliberately preserves manually edited frames when global detection
sliders change. Unedited frames are recalculated with the new parameters.
Use "Reset Current to Automatic" when an edited frame should adopt the latest
slider settings.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from nd2_analyzer.ui.biofilms.colony_separator import ColonySeparator


FrameKey = tuple[int, int, int]  # position, time, channel


def _copy_colonies(colonies: Iterable[dict] | None) -> list[dict]:
    """Copy colony dictionaries without sharing mutable NumPy arrays."""
    copied: list[dict] = []
    for colony in colonies or []:
        item = dict(colony)
        for field in ("contour", "mask"):
            value = item.get(field)
            if isinstance(value, np.ndarray):
                item[field] = value.copy()
        polygon = item.get("polygon")
        if isinstance(polygon, list):
            item["polygon"] = [list(point) for point in polygon]
        copied.append(item)
    return copied


class EditableImageView(QGraphicsView):
    """Zoomable full-frame viewer that emits image-coordinate rectangles."""

    rectangle_selected = Signal(object)  # QRectF in image coordinates
    FIT_MARGIN_PX = 6

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setScene(self._scene)

        self._selection_item: QGraphicsRectItem | None = None
        self._selection_origin: QPointF | None = None
        self._edit_mode = "Navigate"
        self._auto_fit_enabled = True
        self._fit_pending = False

        self.setBackgroundBrush(QBrush(QColor("black")))
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumSize(540, 330)
        self.setMouseTracking(True)
        self.set_edit_mode("Trim")

    def set_edit_mode(self, mode: str) -> None:
        self._edit_mode = str(mode)
        if self._edit_mode == "Navigate":
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)

    def set_image(self, pixmap: QPixmap) -> None:
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self._auto_fit_enabled = True
        self.request_fit()

    def request_fit(self) -> None:
        if self._fit_pending:
            return
        self._fit_pending = True
        QTimer.singleShot(0, self._perform_queued_fit)

    def _perform_queued_fit(self) -> None:
        self._fit_pending = False
        self.fit_image()

    def fit_image(self) -> None:
        pixmap = self._pixmap_item.pixmap()
        if pixmap.isNull():
            return

        image_rect = self._pixmap_item.boundingRect()
        viewport_rect = self.viewport().contentsRect()
        if image_rect.width() <= 0 or image_rect.height() <= 0:
            return

        available_width = max(
            1.0, float(viewport_rect.width() - 2 * self.FIT_MARGIN_PX)
        )
        available_height = max(
            1.0, float(viewport_rect.height() - 2 * self.FIT_MARGIN_PX)
        )
        scale_factor = min(
            available_width / float(image_rect.width()),
            available_height / float(image_rect.height()),
        )

        transform = QTransform()
        transform.scale(scale_factor, scale_factor)
        self.setTransform(transform)
        self._scene.setSceneRect(image_rect)
        self.centerOn(image_rect.center())
        self._auto_fit_enabled = True

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self._pixmap_item.pixmap().isNull():
            return
        self._auto_fit_enabled = False
        factor = 1.20 if event.angleDelta().y() > 0 else 1.0 / 1.20
        self.scale(factor, factor)
        event.accept()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._auto_fit_enabled:
            self.request_fit()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._auto_fit_enabled:
            self.request_fit()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self._auto_fit_enabled = True
        self.fit_image()
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if (
            event.button() == Qt.LeftButton
            and self._edit_mode in {"Add", "Trim"}
            and not self._pixmap_item.pixmap().isNull()
        ):
            point = self.mapToScene(event.position().toPoint())
            if not self._pixmap_item.boundingRect().contains(point):
                event.accept()
                return

            self._selection_origin = point
            pen = QPen(QColor("#00A8FF"), 2, Qt.DashLine)
            self._selection_item = QGraphicsRectItem(QRectF(point, point))
            self._selection_item.setPen(pen)
            self._selection_item.setBrush(QBrush(QColor(0, 168, 255, 35)))
            self._selection_item.setZValue(1000)
            self._scene.addItem(self._selection_item)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._selection_item is not None and self._selection_origin is not None:
            point = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._selection_origin, point).normalized()
            self._selection_item.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if (
            event.button() == Qt.LeftButton
            and self._selection_item is not None
            and self._selection_origin is not None
        ):
            point = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._selection_origin, point).normalized()
            rect = rect.intersected(self._pixmap_item.boundingRect())

            self._scene.removeItem(self._selection_item)
            self._selection_item = None
            self._selection_origin = None

            if rect.width() >= 2 and rect.height() >= 2:
                self.rectangle_selected.emit(rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class VerifyColoniesDialog(QDialog):
    """Calibrate and verify automatic colony detection over many frames."""

    # Legacy output: current frame only. Existing parent code can keep using it.
    colonies_verified = Signal(list, dict)

    # New output: { (position, time, channel): [colony dictionaries] }, params.
    batch_colonies_verified = Signal(object, dict)

    DEFAULT_THUMBNAIL_COUNT = 5
    DISPLAY_COLORS = (
        (255, 64, 64),
        (48, 112, 255),
        (30, 210, 90),
        (255, 215, 0),
        (255, 55, 225),
        (0, 220, 220),
        (255, 140, 30),
        (155, 80, 255),
    )

    def __init__(
        self,
        image=None,
        colonies=None,
        params=None,
        parent=None,
        *,
        image_data=None,
        selected_positions: Sequence[int] | None = None,
        time_start: int | None = None,
        time_end: int | None = None,
        channel: int | None = None,
        drop_frame_zero: bool = False,
    ):
        super().__init__(parent)

        self.initial_image = None if image is None else np.asarray(image)
        self.initial_colonies = _copy_colonies(colonies)
        self.params = dict(params or {})
        self.image_data = image_data or self._find_shared_image_data()

        self.time_count, self.position_count, self.channel_count = (
            self._infer_data_dimensions()
        )

        self.current_time = int(
            np.clip(
                0 if time_start is None else time_start,
                0,
                self.time_count - 1,
            )
        )
        self.current_position = int(
            np.clip(
                0 if not selected_positions else selected_positions[0],
                0,
                self.position_count - 1,
            )
        )
        self.current_channel = int(
            np.clip(0 if channel is None else channel, 0, self.channel_count - 1)
        )

        self.initial_time_start = int(
            np.clip(0 if time_start is None else time_start, 0, self.time_count - 1)
        )
        self.initial_time_end = int(
            np.clip(
                self.time_count - 1 if time_end is None else time_end,
                self.initial_time_start,
                self.time_count - 1,
            )
        )
        self.initial_drop_frame_zero = bool(drop_frame_zero)
        self.initial_positions = sorted(
            {
                int(position)
                for position in (
                    selected_positions
                    if selected_positions is not None
                    else [self.current_position]
                )
                if 0 <= int(position) < self.position_count
            }
        )
        if not self.initial_positions:
            self.initial_positions = [self.current_position]

        self.colony_separator = ColonySeparator()
        self.frame_states: dict[FrameKey, dict] = {}
        self._frame_cache: OrderedDict[FrameKey, np.ndarray] = OrderedDict()
        self._frame_cache_limit = 20
        self._initial_colonies_consumed = False
        self._updating_controls = False
        self._shown_once = False
        self.modify_mode = "Trim"

        self.detect_timer = QTimer(self)
        self.detect_timer.setSingleShot(True)
        self.detect_timer.setInterval(160)
        self.detect_timer.timeout.connect(self.detect_current_frame)

        self.scope_timer = QTimer(self)
        self.scope_timer.setSingleShot(True)
        self.scope_timer.setInterval(150)
        self.scope_timer.timeout.connect(self.refresh_scope)

        self.thumbnail_timer = QTimer(self)
        self.thumbnail_timer.setSingleShot(True)
        self.thumbnail_timer.setInterval(220)
        self.thumbnail_timer.timeout.connect(self.update_thumbnails)

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(500)
        self.play_timer.timeout.connect(self.advance_playback)

        self.setWindowTitle("Auto Colony Verifier")
        self.setMinimumSize(1000, 720)
        self.resize(1320, 900)

        self.init_ui()
        self.populate_initial_values()
        QTimer.singleShot(0, self.refresh_scope)

    # ------------------------------------------------------------------
    # Data source and frame state
    # ------------------------------------------------------------------

    @staticmethod
    def _find_shared_image_data():
        try:
            from nd2_analyzer.data.image_data import ImageData

            candidate = ImageData.get_instance()
            if candidate is not None and getattr(candidate, "data", None) is not None:
                return candidate
        except Exception:
            pass
        return None

    def _infer_data_dimensions(self) -> tuple[int, int, int]:
        if self.image_data is not None and getattr(self.image_data, "data", None) is not None:
            shape = tuple(int(value) for value in self.image_data.data.shape)
            if len(shape) >= 3:
                return max(1, shape[0]), max(1, shape[1]), max(1, shape[2])
        return 1, 1, 1

    def get_frame(self, key: FrameKey) -> np.ndarray:
        cached = self._frame_cache.get(key)
        if cached is not None:
            self._frame_cache.move_to_end(key)
            return cached

        position, time, channel = key
        if self.image_data is not None:
            frame = np.asarray(
                self.image_data.get(int(time), int(position), int(channel))
            )
        elif self.initial_image is not None:
            frame = np.asarray(self.initial_image)
        else:
            raise ValueError("No image source is available for colony verification.")

        frame = np.squeeze(frame)
        if frame.ndim != 2:
            raise ValueError(
                "Expected a two-dimensional colony image for "
                f"P={position}, T={time}, C={channel}; received {frame.shape}."
            )

        self._frame_cache[key] = frame
        self._frame_cache.move_to_end(key)
        while len(self._frame_cache) > self._frame_cache_limit:
            self._frame_cache.popitem(last=False)
        return frame

    def get_state(self, key: FrameKey) -> dict:
        state = self.frame_states.get(key)
        if state is not None:
            return state

        seed_colonies: list[dict] = []
        initial_key = (
            int(self.current_position),
            int(self.current_time),
            int(self.current_channel),
        )
        if (
            self.initial_colonies
            and not self._initial_colonies_consumed
            and key == initial_key
        ):
            seed_colonies = _copy_colonies(self.initial_colonies)
            self._initial_colonies_consumed = True

        state = {
            "auto_colonies": _copy_colonies(seed_colonies),
            "colonies": _copy_colonies(seed_colonies),
            "detected": bool(seed_colonies),
            "dirty": not bool(seed_colonies),
            "edited": False,
            "reviewed": False,
            "parameter_stale": False,
            "history": [],
        }
        self.frame_states[key] = state
        return state

    def current_frame_key(self) -> FrameKey | None:
        position = self.current_preview_position()
        if position is None:
            return None
        return (
            int(position),
            int(self.time_slider.value()),
            int(self.channel_combo.currentData()),
        )

    def considered_timepoints(self) -> list[int]:
        start = int(self.time_start_spin.value())
        end = int(self.time_end_spin.value())
        values = list(range(start, end + 1))
        if self.drop_frame_zero_checkbox.isChecked():
            values = [value for value in values if value != 0]
        return values

    def selected_positions(self) -> list[int]:
        positions: list[int] = []
        for index in range(self.position_list.count()):
            if self.position_list.item(index).isSelected():
                positions.append(index)
        return positions

    def selected_frame_keys(self) -> list[FrameKey]:
        channel = int(self.channel_combo.currentData())
        return [
            (position, time, channel)
            for position in self.selected_positions()
            for time in self.considered_timepoints()
        ]

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        heading = QLabel("Automatic Colony Calibration and Verification")
        heading.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #2196F3;"
        )
        main_layout.addWidget(heading)

        explanation = QLabel(
            "Choose the positions and time range, tune one automatic colony "
            "detection rule, and inspect every selected frame. Add, trim, and "
            "delete edits are stored independently for each frame and restored "
            "when you return to it."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #777; padding-bottom: 4px;")
        main_layout.addWidget(explanation)

        main_layout.addWidget(self.create_top_workspace(), 1)
        main_layout.addWidget(self.create_detection_controls())
        main_layout.addWidget(self.create_time_navigation_group())
        main_layout.addWidget(self.create_validation_group())

        status_row = QHBoxLayout()
        self.status_label = QLabel("Preparing colony previews...")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "color: #777; font-style: italic; padding: 4px;"
        )
        status_row.addWidget(self.status_label, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximumWidth(260)
        self.progress_bar.setVisible(False)
        status_row.addWidget(self.progress_bar)
        main_layout.addLayout(status_row)

        bottom_layout = QHBoxLayout()
        self.reset_all_button = QPushButton("Reset Detection Parameters")
        self.reset_all_button.clicked.connect(self.reset_detection_parameters)
        bottom_layout.addWidget(self.reset_all_button)
        bottom_layout.addStretch()

        self.button_box = QDialogButtonBox()
        self.cancel_button = self.button_box.addButton(
            "Cancel", QDialogButtonBox.RejectRole
        )
        self.accept_button = self.button_box.addButton(
            "Accept All Selected Frames", QDialogButtonBox.AcceptRole
        )
        self.accept_button.setStyleSheet(
            "background-color: #1976D2; color: white; "
            "font-weight: bold; padding: 7px 12px;"
        )
        self.button_box.rejected.connect(self.reject)
        self.button_box.accepted.connect(self.accept_results)
        bottom_layout.addWidget(self.button_box)
        main_layout.addLayout(bottom_layout)

    def create_top_workspace(self) -> QWidget:
        """Create the main preview and a non-overlapping tabbed side panel."""
        widget = QWidget()
        widget.setMinimumHeight(410)

        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        viewer = self.create_viewer_panel()
        right_panel = self.create_right_panel()

        # The viewer remains the dominant area, while the side panel is kept
        # wide enough for long labels and buttons on macOS.
        layout.addWidget(viewer, 7)
        layout.addWidget(right_panel, 4)
        return widget

    def create_viewer_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)
        toolbar.addWidget(QLabel("View:"))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Overlay", "Raw", "Binary Mask"])
        self.view_mode_combo.currentTextChanged.connect(self.refresh_display)
        toolbar.addWidget(self.view_mode_combo)

        self.show_labels_checkbox = QCheckBox("Labels")
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.toggled.connect(self.refresh_display)
        toolbar.addWidget(self.show_labels_checkbox)

        toolbar.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(35)
        self.opacity_slider.setMaximumWidth(150)
        self.opacity_slider.valueChanged.connect(self.refresh_display)
        toolbar.addWidget(self.opacity_slider)

        fit_button = QPushButton("Fit Image")
        fit_button.clicked.connect(self.fit_preview)
        toolbar.addWidget(fit_button)

        self.frame_info_label = QLabel("No frame loaded")
        self.frame_info_label.setStyleSheet("font-weight: bold;")
        toolbar.addWidget(self.frame_info_label, 1)
        layout.addLayout(toolbar)

        self.image_view = EditableImageView()
        self.image_view.rectangle_selected.connect(self.apply_rectangle_edit)
        layout.addWidget(self.image_view, 1)
        return widget

    def create_right_panel(self) -> QWidget:
        """Place scope and frame controls on separate tabs.

        Previously both groups were stacked into the same short column. When
        the top workspace became shorter, Qt compressed both groups until the
        form rows and colony controls overlapped. Tabs give each section the
        full side-panel height and keep every label readable.
        """
        widget = QWidget()
        widget.setMinimumWidth(460)

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.right_tabs = QTabWidget()
        self.right_tabs.setDocumentMode(True)
        self.right_tabs.addTab(self.create_scope_group(), "Analysis Scope")
        self.right_tabs.addTab(
            self.create_current_frame_group(),
            "Current Frame Colonies",
        )
        layout.addWidget(self.right_tabs)
        return widget

    def create_scope_group(self) -> QWidget:
        """Create a vertically scrollable scope page with non-compressing rows.

        macOS can aggressively compress ``QFormLayout`` rows when the tab is
        shorter than the controls it contains.  The scope controls therefore
        use labels above their widgets, fixed minimum control heights, and a
        vertical scroll area.  Nothing is allowed to overlap or collapse into
        a one-line strip when the dialog is resized.
        """
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        title = QLabel("Frames included in colony verification")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title)

        positions_label = QLabel("Positions to verify")
        positions_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(positions_label)

        self.position_list = QListWidget()
        self.position_list.setSelectionMode(QAbstractItemView.MultiSelection)
        # Show a useful number of rows without consuming the whole tab. The
        # surrounding scroll area handles unusually small dialog heights.
        visible_rows = min(max(self.position_count, 3), 5)
        self.position_list.setFixedHeight(24 * visible_rows + 8)
        self.position_list.itemSelectionChanged.connect(
            self.schedule_scope_refresh
        )
        layout.addWidget(self.position_list)

        position_buttons = QHBoxLayout()
        position_buttons.setSpacing(8)
        select_all = QPushButton("Select All")
        select_all.setMinimumHeight(28)
        select_all.clicked.connect(self.select_all_positions)
        select_none = QPushButton("Select None")
        select_none.setMinimumHeight(28)
        select_none.clicked.connect(self.select_no_positions)
        position_buttons.addWidget(select_all)
        position_buttons.addWidget(select_none)
        layout.addLayout(position_buttons)

        # Put the two timepoint controls in their own group with labels above
        # the spin boxes. This avoids the compressed label/value rows visible
        # with QFormLayout on macOS.
        time_group = QGroupBox("Time range")
        time_layout = QHBoxLayout(time_group)
        time_layout.setContentsMargins(10, 8, 10, 10)
        time_layout.setSpacing(12)

        start_column = QVBoxLayout()
        start_column.setSpacing(4)
        start_label = QLabel("Start timepoint")
        start_label.setStyleSheet("font-weight: bold;")
        self.time_start_spin = QSpinBox()
        self.time_start_spin.setRange(0, self.time_count - 1)
        self.time_start_spin.setMinimumHeight(30)
        self.time_start_spin.setMinimumWidth(120)
        self.time_start_spin.valueChanged.connect(self.on_time_bounds_changed)
        start_column.addWidget(start_label)
        start_column.addWidget(self.time_start_spin)
        time_layout.addLayout(start_column, 1)

        end_column = QVBoxLayout()
        end_column.setSpacing(4)
        end_label = QLabel("End timepoint")
        end_label.setStyleSheet("font-weight: bold;")
        self.time_end_spin = QSpinBox()
        self.time_end_spin.setRange(0, self.time_count - 1)
        self.time_end_spin.setMinimumHeight(30)
        self.time_end_spin.setMinimumWidth(120)
        self.time_end_spin.valueChanged.connect(self.on_time_bounds_changed)
        end_column.addWidget(end_label)
        end_column.addWidget(self.time_end_spin)
        time_layout.addLayout(end_column, 1)
        layout.addWidget(time_group)

        preview_group = QGroupBox("Preview settings")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(10, 8, 10, 10)
        preview_layout.setSpacing(8)

        preview_position_label = QLabel("Preview position")
        preview_position_label.setStyleSheet("font-weight: bold;")
        self.preview_position_combo = QComboBox()
        self.preview_position_combo.setMinimumHeight(30)
        self.preview_position_combo.currentIndexChanged.connect(
            self.on_preview_position_changed
        )
        preview_layout.addWidget(preview_position_label)
        preview_layout.addWidget(self.preview_position_combo)

        channel_label = QLabel("Cell channel")
        channel_label.setStyleSheet("font-weight: bold;")
        self.channel_combo = QComboBox()
        self.channel_combo.setMinimumHeight(30)
        for channel in range(self.channel_count):
            self.channel_combo.addItem(f"Channel {channel}", channel)
        self.channel_combo.currentIndexChanged.connect(self.on_channel_changed)
        preview_layout.addWidget(channel_label)
        preview_layout.addWidget(self.channel_combo)
        layout.addWidget(preview_group)

        frame_group = QGroupBox("Frame handling")
        frame_layout = QVBoxLayout(frame_group)
        frame_layout.setContentsMargins(10, 8, 10, 10)
        frame_layout.setSpacing(5)

        self.drop_frame_zero_checkbox = QCheckBox(
            "Exclude frame 0 from preview and output"
        )
        self.drop_frame_zero_checkbox.setMinimumHeight(24)
        self.drop_frame_zero_checkbox.setToolTip(
            "Frame 0 will not appear in previews and will not be included "
            "in the accepted batch output."
        )
        self.drop_frame_zero_checkbox.toggled.connect(
            self.schedule_scope_refresh
        )
        frame_layout.addWidget(self.drop_frame_zero_checkbox)

        frame_zero_note = QLabel(
            "When enabled, T=0 is omitted from preview navigation and from "
            "the accepted colony results."
        )
        frame_zero_note.setWordWrap(True)
        frame_zero_note.setStyleSheet("color: #999999; font-size: 11px;")
        frame_layout.addWidget(frame_zero_note)
        layout.addWidget(frame_group)

        summary_frame = QFrame()
        summary_frame.setFrameShape(QFrame.StyledPanel)
        summary_layout = QVBoxLayout(summary_frame)
        summary_layout.setContentsMargins(8, 7, 8, 7)
        self.scope_summary_label = QLabel("—")
        self.scope_summary_label.setWordWrap(True)
        self.scope_summary_label.setMinimumHeight(34)
        self.scope_summary_label.setStyleSheet("color: #AAAAAA;")
        summary_layout.addWidget(self.scope_summary_label)
        layout.addWidget(summary_frame)
        layout.addStretch(1)

        scroll.setWidget(content)
        page_layout.addWidget(scroll)
        return page

    def create_current_frame_group(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        summary_frame = QFrame()
        summary_frame.setFrameShape(QFrame.StyledPanel)
        summary_layout = QVBoxLayout(summary_frame)
        summary_layout.setContentsMargins(8, 6, 8, 6)

        self.colonies_count_label = QLabel("Colonies: 0")
        self.colonies_count_label.setWordWrap(True)
        self.colonies_count_label.setStyleSheet("font-weight: bold;")
        summary_layout.addWidget(self.colonies_count_label)

        self.frame_status_label = QLabel("Unprocessed")
        self.frame_status_label.setWordWrap(True)
        self.frame_status_label.setStyleSheet("color: #AAAAAA;")
        summary_layout.addWidget(self.frame_status_label)
        layout.addWidget(summary_frame)

        # Keep all edit controls above the scrollable colony list so they can
        # never overlap a colony row when the window is short.
        edit_group = QGroupBox("Edit this frame")
        edit_layout = QVBoxLayout(edit_group)
        edit_layout.setContentsMargins(8, 7, 8, 7)
        edit_layout.setSpacing(7)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Modify mode:"))
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Trim", "Add", "Navigate"])
        self.mode_selector.currentTextChanged.connect(self.on_mode_changed)
        mode_row.addWidget(self.mode_selector, 1)
        edit_layout.addLayout(mode_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo_current_frame)
        action_row.addWidget(self.undo_button)

        self.reset_frame_button = QPushButton("Reset to Automatic")
        self.reset_frame_button.setToolTip(
            "Discard manual edits on this frame and rerun automatic "
            "detection using the current shared slider settings."
        )
        self.reset_frame_button.clicked.connect(self.reset_current_frame)
        action_row.addWidget(self.reset_frame_button, 1)
        edit_layout.addLayout(action_row)

        self.reviewed_checkbox = QCheckBox("Mark current frame reviewed")
        self.reviewed_checkbox.toggled.connect(self.on_reviewed_toggled)
        edit_layout.addWidget(self.reviewed_checkbox)

        edit_hint = QLabel(
            "Trim/Add edits apply only to the current frame and are restored "
            "when you return to it."
        )
        edit_hint.setWordWrap(True)
        edit_hint.setStyleSheet("color: #999999; font-size: 11px;")
        edit_layout.addWidget(edit_hint)
        layout.addWidget(edit_group)

        list_title = QLabel("Detected colonies")
        list_title.setStyleSheet("font-weight: bold;")
        layout.addWidget(list_title)

        self.colonies_list_widget = QWidget()
        self.colonies_list_layout = QVBoxLayout(self.colonies_list_widget)
        self.colonies_list_layout.setContentsMargins(4, 4, 4, 4)
        self.colonies_list_layout.setSpacing(5)
        self.colonies_list_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.colonies_list_widget)
        scroll.setMinimumHeight(220)
        layout.addWidget(scroll, 1)
        return page

    def create_detection_controls(self) -> QGroupBox:
        group = QGroupBox("Automatic detection controls — shared across scope")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(4)

        self.min_size_slider, self.min_size_value_label = self._add_slider_row(
            layout,
            "Minimum colony area:",
            1,
            100,
            int(self.params.get("min_size", 1)),
        )
        self.threshold_slider, self.threshold_value_label = self._add_slider_row(
            layout,
            "Intensity threshold:",
            0,
            100,
            int(self.params.get("threshold", 50)),
        )
        self.kernel_slider, self.kernel_value_label = self._add_slider_row(
            layout,
            "Smoothing kernel:",
            0,
            100,
            int(self.params.get("kernel", 1)),
        )

        for slider in (
            self.min_size_slider,
            self.threshold_slider,
            self.kernel_slider,
        ):
            slider.valueChanged.connect(self.on_detection_parameter_changed)

        self.parameter_note_label = QLabel(
            "Unedited frames update automatically. Manually edited frames are "
            "preserved until Reset Current to Automatic is used."
        )
        self.parameter_note_label.setWordWrap(True)
        self.parameter_note_label.setStyleSheet("color: #888;")
        layout.addWidget(self.parameter_note_label)
        group.setMaximumHeight(170)
        return group

    @staticmethod
    def _add_slider_row(
        parent_layout: QVBoxLayout,
        text: str,
        minimum: int,
        maximum: int,
        value: int,
    ) -> tuple[QSlider, QLabel]:
        row = QHBoxLayout()
        label = QLabel(text)
        label.setMinimumWidth(170)
        row.addWidget(label)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(int(np.clip(value, minimum, maximum)))
        row.addWidget(slider, 1)
        value_label = QLabel("—")
        value_label.setMinimumWidth(150)
        value_label.setStyleSheet("font-weight: bold;")
        row.addWidget(value_label)
        parent_layout.addLayout(row)
        return slider, value_label

    def create_time_navigation_group(self) -> QGroupBox:
        group = QGroupBox("Frame navigation")
        group.setMaximumHeight(70)
        layout = QHBoxLayout(group)
        layout.setContentsMargins(10, 5, 10, 5)

        self.first_button = QPushButton("First")
        self.first_button.clicked.connect(self.go_to_first)
        self.previous_button = QPushButton("◀")
        self.previous_button.clicked.connect(self.go_to_previous)
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self.toggle_playback)
        self.next_button = QPushButton("▶")
        self.next_button.clicked.connect(self.go_to_next)
        self.middle_button = QPushButton("Middle")
        self.middle_button.clicked.connect(self.go_to_middle)
        self.last_button = QPushButton("Last")
        self.last_button.clicked.connect(self.go_to_last)

        for button in (
            self.first_button,
            self.previous_button,
            self.play_button,
            self.next_button,
            self.middle_button,
            self.last_button,
        ):
            layout.addWidget(button)

        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.valueChanged.connect(self.on_time_slider_changed)
        layout.addWidget(self.time_slider, 1)

        self.time_value_label = QLabel("T=0")
        self.time_value_label.setMinimumWidth(55)
        layout.addWidget(self.time_value_label)
        return group

    def create_validation_group(self) -> QGroupBox:
        group = QGroupBox("Representative-frame validation — current position")
        group.setMaximumHeight(95)
        layout = QHBoxLayout(group)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        self.thumbnail_buttons: list[QToolButton] = []
        for index in range(self.DEFAULT_THUMBNAIL_COUNT):
            button = QToolButton()
            button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            button.setIconSize(QPixmap(88, 54).size())
            button.setMinimumSize(128, 64)
            button.setMaximumHeight(68)
            button.clicked.connect(
                lambda _checked=False, i=index: self.open_thumbnail(i)
            )
            button.setVisible(False)
            layout.addWidget(button, 1)
            self.thumbnail_buttons.append(button)
        return group

    # ------------------------------------------------------------------
    # Initialization and scope changes
    # ------------------------------------------------------------------

    def populate_initial_values(self) -> None:
        self._updating_controls = True
        try:
            for position in range(self.position_count):
                self.position_list.addItem(f"Position {position}")
                self.position_list.item(position).setSelected(
                    position in self.initial_positions
                )

            self.time_start_spin.setValue(self.initial_time_start)
            self.time_end_spin.setValue(self.initial_time_end)
            self.drop_frame_zero_checkbox.setChecked(
                self.initial_drop_frame_zero
            )
            channel_index = self.channel_combo.findData(self.current_channel)
            self.channel_combo.setCurrentIndex(max(0, channel_index))
        finally:
            self._updating_controls = False
        self.update_parameter_labels()

    def showEvent(self, event) -> None:  # noqa: N802
        """Honor current T/P/C values assigned by legacy parent code before exec."""
        super().showEvent(event)
        if self._shown_once:
            return
        self._shown_once = True

        external_position = int(
            np.clip(getattr(self, "current_position", 0), 0, self.position_count - 1)
        )
        external_time = int(
            np.clip(getattr(self, "current_time", 0), 0, self.time_count - 1)
        )
        external_channel = int(
            np.clip(getattr(self, "current_channel", 0), 0, self.channel_count - 1)
        )

        self._updating_controls = True
        try:
            if not self.selected_positions():
                self.position_list.item(external_position).setSelected(True)
            channel_index = self.channel_combo.findData(external_channel)
            if channel_index >= 0:
                self.channel_combo.setCurrentIndex(channel_index)
            self.current_position = external_position
            self.current_time = external_time
            self.current_channel = external_channel
        finally:
            self._updating_controls = False
        QTimer.singleShot(0, self.refresh_scope)

    def select_all_positions(self) -> None:
        self._updating_controls = True
        try:
            for index in range(self.position_list.count()):
                self.position_list.item(index).setSelected(True)
        finally:
            self._updating_controls = False
        self.schedule_scope_refresh()

    def select_no_positions(self) -> None:
        self._updating_controls = True
        try:
            for index in range(self.position_list.count()):
                self.position_list.item(index).setSelected(False)
        finally:
            self._updating_controls = False
        self.schedule_scope_refresh()

    def schedule_scope_refresh(self) -> None:
        if not self._updating_controls:
            self.scope_timer.start()

    def on_time_bounds_changed(self) -> None:
        if self._updating_controls:
            return
        sender = self.sender()
        start = self.time_start_spin.value()
        end = self.time_end_spin.value()
        self._updating_controls = True
        try:
            if start > end:
                if sender is self.time_start_spin:
                    self.time_end_spin.setValue(start)
                else:
                    self.time_start_spin.setValue(end)
        finally:
            self._updating_controls = False
        self.schedule_scope_refresh()

    def refresh_scope(self) -> None:
        positions = self.selected_positions()
        timepoints = self.considered_timepoints()

        if not positions:
            self.preview_position_combo.clear()
            self.image_view.set_image(QPixmap())
            self.accept_button.setEnabled(False)
            self.scope_summary_label.setText("Select at least one position.")
            self.status_label.setText("Select at least one position.")
            self.clear_thumbnails()
            return

        if not timepoints:
            self.image_view.set_image(QPixmap())
            self.accept_button.setEnabled(False)
            self.scope_summary_label.setText(
                "No frames remain because frame 0 is excluded."
            )
            self.status_label.setText(
                "Extend the time range beyond T=0 or include frame 0."
            )
            self.clear_thumbnails()
            return

        previous_position = self.current_preview_position()
        desired_position = (
            previous_position
            if previous_position in positions
            else (
                self.current_position
                if self.current_position in positions
                else positions[0]
            )
        )
        desired_time = int(
            np.clip(
                getattr(self, "current_time", timepoints[0]),
                timepoints[0],
                timepoints[-1],
            )
        )
        if desired_time not in timepoints:
            desired_time = timepoints[0]

        self._updating_controls = True
        try:
            self.preview_position_combo.clear()
            for position in positions:
                self.preview_position_combo.addItem(
                    f"Position {position}", position
                )
            position_index = self.preview_position_combo.findData(desired_position)
            self.preview_position_combo.setCurrentIndex(max(0, position_index))

            self.time_slider.setRange(timepoints[0], timepoints[-1])
            self.time_slider.setValue(desired_time)
        finally:
            self._updating_controls = False

        frame_count = len(positions) * len(timepoints)
        reviewed = sum(
            bool(self.frame_states.get(key, {}).get("reviewed"))
            for key in self.selected_frame_keys()
        )
        edited = sum(
            bool(self.frame_states.get(key, {}).get("edited"))
            for key in self.selected_frame_keys()
        )
        self.scope_summary_label.setText(
            f"{len(positions)} position(s) × {len(timepoints)} timepoint(s) "
            f"= {frame_count} frames • Reviewed {reviewed} • Edited {edited}"
        )
        self.accept_button.setEnabled(True)
        self.load_current_frame()
        self.thumbnail_timer.start()

    def current_preview_position(self) -> int | None:
        if self.preview_position_combo.count() == 0:
            return None
        value = self.preview_position_combo.currentData()
        return None if value is None else int(value)

    def on_preview_position_changed(self) -> None:
        if self._updating_controls:
            return
        position = self.current_preview_position()
        if position is not None:
            self.current_position = position
        self.load_current_frame()
        self.thumbnail_timer.start()

    def on_channel_changed(self) -> None:
        if self._updating_controls:
            return
        self.current_channel = int(self.channel_combo.currentData())
        self.mark_unedited_states_dirty()
        self.load_current_frame()
        self.thumbnail_timer.start()

    def on_time_slider_changed(self, value: int) -> None:
        self.time_value_label.setText(f"T={value}")
        self.current_time = int(value)
        if not self._updating_controls:
            self.load_current_frame()

    # ------------------------------------------------------------------
    # Detection and display
    # ------------------------------------------------------------------

    def current_detector_parameters(self, image_shape) -> dict:
        height, width = image_shape[:2]
        total_pixels = max(1, int(height * width))

        # Preserve the original dialog's slider semantics:
        # 1..100 corresponds to 0.01%..1.00% of the frame area.
        min_area = (
            self.min_size_slider.value() / 100.0
        ) * total_pixels * 0.01
        threshold = int(round(self.threshold_slider.value() / 100.0 * 255.0))
        kernel = int(self.kernel_slider.value() / 100.0 * 75)
        kernel = max(1, kernel)
        if kernel % 2 == 0:
            kernel += 1

        return {
            "min_size": int(self.min_size_slider.value()),
            "threshold": int(self.threshold_slider.value()),
            "kernel": int(self.kernel_slider.value()),
            "min_colony_size_px": float(min_area),
            "intensity_threshold_uint8": int(threshold),
            "kernel_size_px": int(kernel),
        }

    def apply_detector_parameters(self, image_shape) -> dict:
        values = self.current_detector_parameters(image_shape)
        self.colony_separator.update_parameters(
            min_colony_size=values["min_colony_size_px"],
            intensity_threshold=values["intensity_threshold_uint8"],
            kernel_size=values["kernel_size_px"],
        )
        return values

    def convert_detected_colonies(
        self,
        raw_colonies: Iterable[dict],
        image_shape,
        key: FrameKey,
    ) -> list[dict]:
        position, time, channel = key
        height, width = image_shape[:2]
        converted: list[dict] = []

        for colony in raw_colonies:
            contour = colony.get("contour")
            if contour is None:
                continue
            contour = np.asarray(contour, dtype=np.int32)
            if contour.ndim == 2:
                contour = contour.reshape(-1, 1, 2)

            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, -1)
            converted.append(
                {
                    "data_type": "colony",
                    "position": int(position),
                    "time": int(time),
                    "channel": int(channel),
                    "colony_id": len(converted) + 1,
                    "contour": contour,
                    "polygon": contour.reshape(-1, 2).tolist(),
                    "mask": mask,
                    "area": float(cv2.contourArea(contour)),
                    "source": "auto",
                }
            )
        return converted

    def detect_frame(self, key: FrameKey, *, force: bool = False) -> list[dict]:
        state = self.get_state(key)
        if state["edited"] and not force:
            state["parameter_stale"] = True
            return state["colonies"]
        if state["detected"] and not state["dirty"] and not force:
            return state["colonies"]

        image = self.get_frame(key)
        self.apply_detector_parameters(image.shape)
        raw_colonies = self.colony_separator.detect_colonies_otsu(image)
        colonies = self.convert_detected_colonies(raw_colonies, image.shape, key)

        state["auto_colonies"] = _copy_colonies(colonies)
        state["colonies"] = _copy_colonies(colonies)
        state["detected"] = True
        state["dirty"] = False
        state["parameter_stale"] = False
        if force:
            state["edited"] = False
            state["history"] = []
        return state["colonies"]

    def detect_current_frame(self) -> None:
        key = self.current_frame_key()
        if key is None:
            return
        state = self.get_state(key)
        if state["edited"]:
            state["parameter_stale"] = True
            self.status_label.setText(
                "This frame contains manual edits, so its verified result was "
                "preserved. Use Reset Current to Automatic to apply the latest "
                "slider settings to it."
            )
            self.refresh_display()
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            colonies = self.detect_frame(key, force=True)
            self.status_label.setText(
                f"Detected {len(colonies)} colonies automatically on "
                f"P={key[0]}, T={key[1]}, C={key[2]}."
            )
        except Exception as error:
            self.status_label.setText(f"Automatic detection failed: {error}")
        finally:
            QApplication.restoreOverrideCursor()
        self.refresh_current_frame_ui()
        self.thumbnail_timer.start()

    def load_current_frame(self) -> None:
        key = self.current_frame_key()
        if key is None:
            return
        try:
            state = self.get_state(key)
            if not state["detected"] or (state["dirty"] and not state["edited"]):
                self.detect_frame(key)
            self.refresh_current_frame_ui()
        except Exception as error:
            self.status_label.setText(f"Could not load frame: {error}")
            self.image_view.set_image(QPixmap())

    def refresh_current_frame_ui(self) -> None:
        key = self.current_frame_key()
        if key is None:
            return
        state = self.get_state(key)
        colonies = state["colonies"]
        self.colony_separator.detected_colonies = _copy_colonies(colonies)

        self._updating_controls = True
        try:
            self.reviewed_checkbox.setChecked(bool(state["reviewed"]))
        finally:
            self._updating_controls = False

        status_parts = []
        if state["reviewed"]:
            status_parts.append("Reviewed")
        if state["edited"]:
            status_parts.append("Edited")
        elif state["detected"]:
            status_parts.append("Automatic")
        if state["parameter_stale"]:
            status_parts.append("using preserved edits")
        self.frame_status_label.setText(" • ".join(status_parts) or "Unprocessed")

        self.frame_info_label.setText(
            f"Position {key[0]} • Timepoint {key[1]} • Channel {key[2]} • "
            f"{len(colonies)} colonies"
        )
        self.time_value_label.setText(f"T={key[1]}")
        self.update_colonies_list()
        self.update_parameter_labels()
        self.refresh_display()
        self.refresh_scope_summary_only()

    @staticmethod
    def normalize_for_display(image: np.ndarray) -> np.ndarray:
        work = np.asarray(image)
        if work.dtype == np.uint8:
            return work.copy()
        finite = work[np.isfinite(work)]
        if finite.size == 0:
            return np.zeros(work.shape, dtype=np.uint8)
        low = float(np.min(finite))
        high = float(np.max(finite))
        if high <= low:
            return np.zeros(work.shape, dtype=np.uint8)
        scaled = (work.astype(np.float32) - low) / (high - low)
        return np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)

    def create_display_rgb(
        self,
        key: FrameKey,
        *,
        forced_mode: str | None = None,
        thumbnail: bool = False,
    ) -> np.ndarray:
        image = self.normalize_for_display(self.get_frame(key))
        state = self.get_state(key)
        colonies = state["colonies"]
        mode = forced_mode or self.view_mode_combo.currentText()

        if mode == "Binary Mask":
            union_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for colony in colonies:
                mask = colony.get("mask")
                if isinstance(mask, np.ndarray) and mask.shape == union_mask.shape:
                    union_mask[mask > 0] = 255
                else:
                    contour = colony.get("contour")
                    if contour is not None:
                        cv2.drawContours(union_mask, [contour], -1, 255, -1)
            return np.repeat(union_mask[..., None], 3, axis=2)

        rgb = np.repeat(image[..., None], 3, axis=2)
        if mode == "Raw":
            return rgb

        output = rgb.astype(np.float32)
        alpha = self.opacity_slider.value() / 100.0
        for index, colony in enumerate(colonies):
            contour = colony.get("contour")
            if contour is None:
                continue
            contour = np.asarray(contour, dtype=np.int32)
            color = self.DISPLAY_COLORS[index % len(self.DISPLAY_COLORS)]
            mask = colony.get("mask")
            if not isinstance(mask, np.ndarray) or mask.shape != image.shape:
                mask = np.zeros(image.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 1, -1)

            color_array = np.asarray(color, dtype=np.float32)
            output[mask > 0] = (
                (1.0 - alpha) * output[mask > 0] + alpha * color_array
            )
            cv2.drawContours(output, [contour], -1, color, 2)

            if self.show_labels_checkbox.isChecked() and not thumbnail:
                moments = cv2.moments(contour)
                if moments["m00"]:
                    x = int(moments["m10"] / moments["m00"])
                    y = int(moments["m01"] / moments["m00"])
                    area = int(round(float(colony.get("area", 0))))
                    cv2.putText(
                        output,
                        f"{index + 1}: {area} px2",
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
        return np.clip(output, 0, 255).astype(np.uint8)

    @staticmethod
    def rgb_to_pixmap(rgb: np.ndarray) -> QPixmap:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        height, width = rgb.shape[:2]
        image = QImage(
            rgb.data,
            width,
            height,
            int(rgb.strides[0]),
            QImage.Format_RGB888,
        ).copy()
        return QPixmap.fromImage(image)

    def refresh_display(self, *_args) -> None:
        key = self.current_frame_key()
        if key is None:
            return
        try:
            rgb = self.create_display_rgb(key)
            self.image_view.set_image(self.rgb_to_pixmap(rgb))
        except Exception as error:
            self.status_label.setText(f"Display failed: {error}")

    def fit_preview(self) -> None:
        self.image_view._auto_fit_enabled = True
        self.image_view.request_fit()

    # ------------------------------------------------------------------
    # Slider callbacks
    # ------------------------------------------------------------------

    def update_parameter_labels(self) -> None:
        key = self.current_frame_key()
        shape = None
        if key is not None:
            try:
                shape = self.get_frame(key).shape
            except Exception:
                pass
        if shape is None:
            shape = self.initial_image.shape if self.initial_image is not None else (1, 1)

        values = self.current_detector_parameters(shape)
        self.min_size_value_label.setText(
            f"{values['min_size']}% control → "
            f"{values['min_colony_size_px']:.0f} px²"
        )
        self.threshold_value_label.setText(
            f"{values['threshold']}% → "
            f"{values['intensity_threshold_uint8']} / 255"
        )
        self.kernel_value_label.setText(
            f"{values['kernel']}% → {values['kernel_size_px']} px"
        )

    def on_detection_parameter_changed(self) -> None:
        if self._updating_controls:
            return
        self.update_parameter_labels()
        self.mark_unedited_states_dirty()
        key = self.current_frame_key()
        if key is not None and self.get_state(key)["edited"]:
            self.get_state(key)["parameter_stale"] = True
            self.refresh_current_frame_ui()
        else:
            self.detect_timer.start()
        self.thumbnail_timer.start()

    def mark_unedited_states_dirty(self) -> None:
        for state in self.frame_states.values():
            if state["edited"]:
                state["parameter_stale"] = True
            else:
                state["dirty"] = True

    def reset_detection_parameters(self) -> None:
        self.min_size_slider.setValue(int(self.params.get("min_size", 1)))
        self.threshold_slider.setValue(int(self.params.get("threshold", 50)))
        self.kernel_slider.setValue(int(self.params.get("kernel", 1)))
        self.opacity_slider.setValue(35)
        self.view_mode_combo.setCurrentText("Overlay")

    # ------------------------------------------------------------------
    # Per-frame edits
    # ------------------------------------------------------------------

    def on_mode_changed(self, mode: str) -> None:
        self.modify_mode = mode
        self.image_view.set_edit_mode(mode)
        if mode == "Navigate":
            self.status_label.setText(
                "Navigate mode: drag to pan, use the mouse wheel to zoom, and "
                "double-click to fit the full frame."
            )
        else:
            self.status_label.setText(
                f"{mode} mode: drag a rectangle over the image."
            )

    def push_history(self, state: dict) -> None:
        state["history"].append(_copy_colonies(state["colonies"]))
        if len(state["history"]) > 25:
            state["history"].pop(0)

    def apply_rectangle_edit(self, rect: QRectF) -> None:
        if self.modify_mode not in {"Add", "Trim"}:
            return
        key = self.current_frame_key()
        if key is None:
            return
        image = self.get_frame(key)
        height, width = image.shape[:2]

        x1 = int(np.clip(np.floor(rect.left()), 0, width))
        y1 = int(np.clip(np.floor(rect.top()), 0, height))
        x2 = int(np.clip(np.ceil(rect.right()), 0, width))
        y2 = int(np.clip(np.ceil(rect.bottom()), 0, height))
        if x2 <= x1 or y2 <= y1:
            return

        state = self.get_state(key)
        self.push_history(state)
        if self.modify_mode == "Add":
            self._add_colony_rectangle(state, key, x1, y1, x2, y2, image.shape)
        else:
            self._trim_colonies_rectangle(state, key, x1, y1, x2, y2, image.shape)

        state["edited"] = True
        state["reviewed"] = False
        state["parameter_stale"] = False
        self.refresh_current_frame_ui()
        self.thumbnail_timer.start()

    def _add_colony_rectangle(
        self,
        state: dict,
        key: FrameKey,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        image_shape,
    ) -> None:
        height, width = image_shape[:2]
        rect_mask = np.zeros((height, width), dtype=np.uint8)
        rect_mask[y1:y2, x1:x2] = 1

        overlapping: list[int] = []
        for index, colony in enumerate(state["colonies"]):
            mask = colony.get("mask")
            if isinstance(mask, np.ndarray) and np.count_nonzero(mask & rect_mask) > 20:
                overlapping.append(index)

        combined = rect_mask.copy()
        for index in overlapping:
            combined = np.logical_or(combined, state["colonies"][index]["mask"])
        combined = combined.astype(np.uint8)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return
        contour = max(contours, key=cv2.contourArea)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, -1)

        for index in sorted(overlapping, reverse=True):
            del state["colonies"][index]

        source = (
            "manual"
            if not overlapping
            else "expanded" if len(overlapping) == 1 else "merged"
        )
        state["colonies"].append(
            self._make_colony(contour, mask, key, source)
        )
        self.renumber_colonies(state["colonies"])
        self.status_label.setText(
            "Added a new colony region."
            if not overlapping
            else f"Expanded or merged {len(overlapping)} colony region(s)."
        )

    def _trim_colonies_rectangle(
        self,
        state: dict,
        key: FrameKey,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        image_shape,
    ) -> None:
        height, width = image_shape[:2]
        rect_mask = np.zeros((height, width), dtype=np.uint8)
        rect_mask[y1:y2, x1:x2] = 1

        new_colonies: list[dict] = []
        removed_count = 0
        split_count = 0

        for colony in state["colonies"]:
            mask = colony.get("mask")
            if not isinstance(mask, np.ndarray):
                new_colonies.append(colony)
                continue
            if np.count_nonzero(mask & rect_mask) <= 20:
                new_colonies.append(colony)
                continue

            remaining = np.logical_and(mask, np.logical_not(rect_mask)).astype(np.uint8)
            if np.count_nonzero(remaining) < 10:
                removed_count += 1
                continue

            contours, _ = cv2.findContours(
                remaining, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            fragments = []
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < 20:
                    continue
                fragment_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(fragment_mask, [contour], -1, 1, -1)
                fragments.append(
                    self._make_colony(contour, fragment_mask, key, "trimmed")
                )

            if not fragments:
                removed_count += 1
            else:
                if len(fragments) > 1:
                    split_count += 1
                    for fragment in fragments:
                        fragment["source"] = "split"
                new_colonies.extend(fragments)

        state["colonies"] = new_colonies
        self.renumber_colonies(state["colonies"])
        self.status_label.setText(
            f"Trim complete: {removed_count} removed; "
            f"{split_count} colony region(s) split."
        )

    @staticmethod
    def _make_colony(
        contour: np.ndarray,
        mask: np.ndarray,
        key: FrameKey,
        source: str,
    ) -> dict:
        position, time, channel = key
        contour = np.asarray(contour, dtype=np.int32)
        if contour.ndim == 2:
            contour = contour.reshape(-1, 1, 2)
        return {
            "data_type": "colony",
            "position": int(position),
            "time": int(time),
            "channel": int(channel),
            "colony_id": None,
            "contour": contour,
            "polygon": contour.reshape(-1, 2).tolist(),
            "mask": mask.astype(np.uint8, copy=True),
            "area": float(cv2.contourArea(contour)),
            "source": source,
        }

    @staticmethod
    def renumber_colonies(colonies: list[dict]) -> None:
        for index, colony in enumerate(colonies, start=1):
            colony["colony_id"] = index

    def delete_colony(self, index: int) -> None:
        key = self.current_frame_key()
        if key is None:
            return
        state = self.get_state(key)
        if not 0 <= index < len(state["colonies"]):
            return
        self.push_history(state)
        del state["colonies"][index]
        self.renumber_colonies(state["colonies"])
        state["edited"] = True
        state["reviewed"] = False
        state["parameter_stale"] = False
        self.status_label.setText("Deleted the selected colony from this frame.")
        self.refresh_current_frame_ui()
        self.thumbnail_timer.start()

    def undo_current_frame(self) -> None:
        key = self.current_frame_key()
        if key is None:
            return
        state = self.get_state(key)
        if not state["history"]:
            self.status_label.setText("There is nothing to undo on this frame.")
            return
        state["colonies"] = state["history"].pop()
        state["edited"] = bool(state["history"]) or (
            self._colony_signature(state["colonies"])
            != self._colony_signature(state["auto_colonies"])
        )
        state["reviewed"] = False
        self.status_label.setText("Undid the last edit on this frame.")
        self.refresh_current_frame_ui()
        self.thumbnail_timer.start()

    @staticmethod
    def _colony_signature(colonies: list[dict]) -> tuple:
        return tuple(
            sorted(
                (
                    int(round(float(colony.get("area", 0)))),
                    tuple(np.asarray(colony.get("contour", [])).reshape(-1).tolist()),
                )
                for colony in colonies
            )
        )

    def reset_current_frame(self) -> None:
        key = self.current_frame_key()
        if key is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.detect_frame(key, force=True)
            state = self.get_state(key)
            state["reviewed"] = False
            self.status_label.setText(
                "Reset this frame to a fresh automatic result using the "
                "current slider settings."
            )
        except Exception as error:
            self.status_label.setText(f"Could not reset frame: {error}")
        finally:
            QApplication.restoreOverrideCursor()
        self.refresh_current_frame_ui()
        self.thumbnail_timer.start()

    def on_reviewed_toggled(self, reviewed: bool) -> None:
        if self._updating_controls:
            return
        key = self.current_frame_key()
        if key is None:
            return
        self.get_state(key)["reviewed"] = bool(reviewed)
        self.refresh_current_frame_ui()
        self.thumbnail_timer.start()

    # ------------------------------------------------------------------
    # Colony list and frame summaries
    # ------------------------------------------------------------------

    def update_colonies_list(self) -> None:
        while self.colonies_list_layout.count() > 1:
            item = self.colonies_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        key = self.current_frame_key()
        colonies = [] if key is None else self.get_state(key)["colonies"]
        for index, colony in enumerate(colonies):
            row_widget = QFrame()
            row_widget.setFrameShape(QFrame.StyledPanel)
            row_widget.setMinimumHeight(54)
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(8, 5, 8, 5)
            row.setSpacing(8)

            source = str(colony.get("source", "auto")).title()
            area = float(colony.get("area", 0))
            label = QLabel(
                f"Colony {index + 1}\n{area:,.0f} px² • {source}"
            )
            label.setWordWrap(True)
            row.addWidget(label, 1)

            delete_button = QPushButton("Delete")
            delete_button.setMaximumWidth(64)
            delete_button.setStyleSheet(
                "background-color: #D32F2F; color: white; font-size: 10px;"
            )
            delete_button.clicked.connect(
                lambda _checked=False, i=index: self.delete_colony(i)
            )
            row.addWidget(delete_button)
            self.colonies_list_layout.insertWidget(
                self.colonies_list_layout.count() - 1, row_widget
            )

        total_area = sum(float(colony.get("area", 0)) for colony in colonies)
        self.colonies_count_label.setText(
            f"Colonies: {len(colonies)} • Total area: {total_area:,.0f} px²"
        )

    def refresh_scope_summary_only(self) -> None:
        positions = self.selected_positions()
        timepoints = self.considered_timepoints()
        keys = self.selected_frame_keys()
        reviewed = sum(bool(self.frame_states.get(key, {}).get("reviewed")) for key in keys)
        edited = sum(bool(self.frame_states.get(key, {}).get("edited")) for key in keys)
        detected = sum(bool(self.frame_states.get(key, {}).get("detected")) for key in keys)
        self.scope_summary_label.setText(
            f"{len(positions)} position(s) × {len(timepoints)} timepoint(s) "
            f"= {len(keys)} frames • Processed {detected} • "
            f"Reviewed {reviewed} • Edited {edited}"
        )

    # ------------------------------------------------------------------
    # Representative thumbnails and navigation
    # ------------------------------------------------------------------

    @staticmethod
    def evenly_spaced_values(values: Sequence[int], count: int) -> list[int]:
        if not values:
            return []
        if len(values) <= count:
            return list(values)
        indices = np.linspace(0, len(values) - 1, count, dtype=int)
        return [int(values[index]) for index in np.unique(indices)]

    def clear_thumbnails(self) -> None:
        for button in self.thumbnail_buttons:
            button.setIcon(QIcon())
            button.setText("—")
            button.setProperty("timepoint", None)
            button.setEnabled(False)
            button.setVisible(False)

    def update_thumbnails(self) -> None:
        self.clear_thumbnails()
        position = self.current_preview_position()
        if position is None:
            return
        channel = int(self.channel_combo.currentData())
        times = self.evenly_spaced_values(
            self.considered_timepoints(), self.DEFAULT_THUMBNAIL_COUNT
        )

        for index, time in enumerate(times):
            if index >= len(self.thumbnail_buttons):
                break
            key = (position, time, channel)
            button = self.thumbnail_buttons[index]
            try:
                state = self.get_state(key)
                if not state["detected"] or (state["dirty"] and not state["edited"]):
                    self.detect_frame(key)
                rgb = self.create_display_rgb(
                    key, forced_mode="Overlay", thumbnail=True
                )
                pixmap = self.rgb_to_pixmap(rgb).scaled(
                    88, 54, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )

                if state["reviewed"]:
                    status = "✓"
                elif state["edited"]:
                    status = "✎"
                else:
                    status = "A"
                button.setIcon(QIcon(pixmap))
                button.setText(f"{status}  T={time}\n{len(state['colonies'])} colonies")
                button.setProperty("timepoint", int(time))
                button.setToolTip(
                    "✓ reviewed • ✎ manually edited • A automatic"
                )
                button.setEnabled(True)
                button.setVisible(True)
            except Exception as error:
                button.setText(f"T={time}\nUnavailable")
                button.setToolTip(str(error))
                button.setVisible(True)

        self.refresh_scope_summary_only()

    def open_thumbnail(self, index: int) -> None:
        if not 0 <= index < len(self.thumbnail_buttons):
            return
        time = self.thumbnail_buttons[index].property("timepoint")
        if time is not None:
            self.time_slider.setValue(int(time))

    def go_to_first(self) -> None:
        values = self.considered_timepoints()
        if values:
            self.time_slider.setValue(values[0])

    def go_to_middle(self) -> None:
        values = self.considered_timepoints()
        if values:
            self.time_slider.setValue(values[len(values) // 2])

    def go_to_last(self) -> None:
        values = self.considered_timepoints()
        if values:
            self.time_slider.setValue(values[-1])

    def go_to_previous(self) -> None:
        values = self.considered_timepoints()
        if not values:
            return
        current = self.time_slider.value()
        earlier = [value for value in values if value < current]
        self.time_slider.setValue(earlier[-1] if earlier else values[0])

    def go_to_next(self) -> None:
        values = self.considered_timepoints()
        if not values:
            return
        current = self.time_slider.value()
        later = [value for value in values if value > current]
        self.time_slider.setValue(later[0] if later else values[-1])

    def toggle_playback(self, playing: bool) -> None:
        if playing:
            self.play_button.setText("Pause")
            self.play_timer.start()
        else:
            self.play_button.setText("Play")
            self.play_timer.stop()

    def advance_playback(self) -> None:
        values = self.considered_timepoints()
        if not values:
            return
        current = self.time_slider.value()
        try:
            index = values.index(current)
        except ValueError:
            index = -1
        self.time_slider.setValue(values[(index + 1) % len(values)])

    # ------------------------------------------------------------------
    # Acceptance and export
    # ------------------------------------------------------------------

    def final_params(self) -> dict:
        key = self.current_frame_key()
        shape = self.get_frame(key).shape if key is not None else (1, 1)
        values = self.current_detector_parameters(shape)
        values.update(
            {
                "positions": tuple(self.selected_positions()),
                "time_start": int(self.time_start_spin.value()),
                "time_end": int(self.time_end_spin.value()),
                "drop_frame_zero": bool(
                    self.drop_frame_zero_checkbox.isChecked()
                ),
                "channel": int(self.channel_combo.currentData()),
                "frame_key_order": "(position, time, channel)",
                "edited_frames_preserved_when_sliders_change": True,
                "reviewed_frame_keys": tuple(
                    key
                    for key in self.selected_frame_keys()
                    if self.frame_states.get(key, {}).get("reviewed")
                ),
                "edited_frame_keys": tuple(
                    key
                    for key in self.selected_frame_keys()
                    if self.frame_states.get(key, {}).get("edited")
                ),
            }
        )
        return values

    def accept_results(self) -> None:
        keys = self.selected_frame_keys()
        if not keys:
            QMessageBox.warning(
                self,
                "No colony frames selected",
                "Select at least one position and one timepoint.",
            )
            return

        self.play_timer.stop()
        self.play_button.setChecked(False)
        self.accept_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        results: dict[FrameKey, list[dict]] = {}
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for index, key in enumerate(keys, start=1):
                self.status_label.setText(
                    f"Preparing colony result {index}/{len(keys)}: "
                    f"P={key[0]}, T={key[1]}, C={key[2]}"
                )
                QApplication.processEvents()

                state = self.get_state(key)
                if not state["detected"] or (state["dirty"] and not state["edited"]):
                    self.detect_frame(key)
                colonies = _copy_colonies(state["colonies"])
                self.renumber_colonies(colonies)
                for colony in colonies:
                    colony["position"] = key[0]
                    colony["time"] = key[1]
                    colony["channel"] = key[2]
                results[key] = colonies
                self.progress_bar.setValue(int(round(index / len(keys) * 90)))

            exported_paths = self.export_colony_overlays(results)
            gif_paths = list(getattr(self, "last_overlay_gif_paths", ()))
            params = self.final_params()
            params["overlay_paths"] = tuple(str(path) for path in exported_paths)
            params["overlay_gif_paths"] = tuple(str(path) for path in gif_paths)
            self.progress_bar.setValue(100)

            current_key = self.current_frame_key()
            current_colonies = (
                _copy_colonies(results.get(current_key, []))
                if current_key is not None
                else []
            )

            self.batch_colonies_verified.emit(results, params)
            self.colonies_verified.emit(current_colonies, params)
            self.status_label.setText(
                f"Accepted {len(results)} frames and exported "
                f"{len(exported_paths)} colony overlays plus "
                f"{len(gif_paths)} position GIF(s)."
            )
            self.accept()
        except Exception as error:
            QMessageBox.critical(
                self,
                "Colony verification failed",
                f"Could not prepare the selected colony frames:\n{error}",
            )
            self.status_label.setText(f"Acceptance failed: {error}")
            self.accept_button.setEnabled(True)
        finally:
            QApplication.restoreOverrideCursor()

    def export_colony_overlays(
        self, results: dict[FrameKey, list[dict]]
    ) -> list[Path]:
        project_root = Path(__file__).resolve().parents[4]
        output_dir = project_root / "analysis_results" / "auto_colony_selector"
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        frames_by_position: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = {}

        for key, colonies in results.items():
            state = self.get_state(key)
            previous = state["colonies"]
            state["colonies"] = colonies
            try:
                rgb = self.create_display_rgb(key, forced_mode="Overlay")
            finally:
                state["colonies"] = previous

            position, time, channel = key
            save_path = output_dir / f"pos{position}_t{time}_c{channel}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            saved.append(save_path)
            frames_by_position.setdefault((position, channel), []).append(
                (time, rgb.copy())
            )

        gif_paths = []
        for (position, channel), timed_frames in sorted(frames_by_position.items()):
            frames = [
                frame
                for _time, frame in sorted(timed_frames, key=lambda item: item[0])
            ]
            if not frames:
                continue
            gif_path = output_dir / f"pos{position}_c{channel}_colony_overlays.gif"
            imageio.mimsave(
                gif_path,
                frames,
                duration=0.5,
                loop=0,
            )
            gif_paths.append(gif_path)

        self.last_overlay_gif_paths = tuple(gif_paths)
        return saved

    # Backward-compatible method name retained for callers/tests.
    def export_colony_overlay(self):
        key = self.current_frame_key()
        if key is None:
            return None
        state = self.get_state(key)
        paths = self.export_colony_overlays({key: _copy_colonies(state["colonies"])})
        return paths[0] if paths else None

    def reject(self) -> None:  # noqa: A003
        self.play_timer.stop()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.play_timer.stop()
        self.detect_timer.stop()
        self.scope_timer.stop()
        self.thumbnail_timer.stop()
        super().closeEvent(event)
