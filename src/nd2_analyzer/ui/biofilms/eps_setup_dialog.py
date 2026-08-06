"""Interactive calibration dialog for Partaker fixed EPS thresholding.

The dialog previews a single, globally consistent fixed threshold without
creating masks for every selected frame.  It estimates one shared raw-to-uint8
mapping from representative frames, then reuses that mapping for the large
preview and the five validation thumbnails.

The accepted :class:`EPSSetupResult` is intended to be passed back to
``EPSAnalysisWidget`` so the full analysis uses the exact same positions, time
range, threshold, smoothing, and intensity mapping that the user approved.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap, QTransform
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class EPSSetupResult:
    """Settings approved in :class:`EPSSetupDialog`."""

    positions: tuple[int, ...]
    time_start: int
    time_end: int
    drop_frame_zero: bool
    eps_channel: int
    threshold_uint8: int
    processing_black_point_raw: float
    processing_white_point_raw: float
    smoothing_sigma: float
    preview_position: int
    representative_timepoints: tuple[int, ...]
    sampled_frame_keys: tuple[tuple[int, int, int], ...]
    overlay_color_name: str = "green"
    capture_interval_value: float = 24.0
    capture_interval_unit: str = "hr"
    processing_window_source: str = (
        "partaker_setup_dialog_full_selected_scope_min_max"
    )
    threshold_rule: str = "foreground = shared_uint8 >= threshold"

    def to_dict(self) -> dict:
        """Return a serialization-friendly copy of the result."""
        return asdict(self)


class ZoomableImageView(QGraphicsView):
    """Image viewer that keeps the complete frame inside the viewport."""

    FIT_MARGIN_PX = 6

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setScene(self._scene)

        self._auto_fit_enabled = True
        self._fit_pending = False

        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(Qt.black)

        # Hidden scrollbars prevent their late appearance from reducing the
        # viewport after the fit scale has already been calculated.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.setMinimumSize(520, 260)

    def set_image(self, pixmap: QPixmap) -> None:
        """Display a new image and fit the complete frame after layout."""
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self._auto_fit_enabled = True
        self.request_fit()

    def request_fit(self) -> None:
        """Queue one fit after Qt has finalized the current viewport size."""
        if self._fit_pending:
            return

        self._fit_pending = True
        QTimer.singleShot(0, self._perform_queued_fit)

    def _perform_queued_fit(self) -> None:
        self._fit_pending = False
        self.fit_image()

    def fit_image(self) -> None:
        """
        Fit the complete image into the available viewport.

        The height ratio is always considered, so the full image y-axis remains
        visible. The width ratio is also considered to prevent horizontal
        cropping. Aspect ratio is preserved.
        """
        pixmap = self._pixmap_item.pixmap()
        if pixmap.isNull():
            return

        image_rect = self._pixmap_item.boundingRect()
        viewport_rect = self.viewport().contentsRect()

        image_width = float(image_rect.width())
        image_height = float(image_rect.height())

        available_width = float(
            max(1, viewport_rect.width() - 2 * self.FIT_MARGIN_PX)
        )
        available_height = float(
            max(1, viewport_rect.height() - 2 * self.FIT_MARGIN_PX)
        )

        if image_width <= 0 or image_height <= 0:
            return

        # Height fitting guarantees the complete y-axis is visible.
        fit_height_scale = available_height / image_height

        # Taking the smaller value also guarantees the full x-axis remains
        # visible when the viewport is unusually narrow.
        fit_width_scale = available_width / image_width
        scale_factor = min(fit_height_scale, fit_width_scale)

        transform = QTransform()
        transform.scale(scale_factor, scale_factor)
        self.setTransform(transform)

        self._scene.setSceneRect(image_rect)
        self.centerOn(image_rect.center())
        self._auto_fit_enabled = True

    def wheelEvent(self, event) -> None:  # noqa: N802
        """Allow manual zoom until Fit Image is requested again."""
        if self._pixmap_item.pixmap().isNull():
            return

        self._auto_fit_enabled = False
        factor = 1.20 if event.angleDelta().y() > 0 else 1 / 1.20
        self.scale(factor, factor)

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Refit automatically whenever the preview area changes size."""
        super().resizeEvent(event)
        if self._auto_fit_enabled:
            self.request_fit()

    def showEvent(self, event) -> None:  # noqa: N802
        """Perform a final fit once the viewer becomes visible."""
        super().showEvent(event)
        if self._auto_fit_enabled:
            self.request_fit()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        """Double-clicking restores the complete-frame view."""
        self._auto_fit_enabled = True
        self.fit_image()
        event.accept()


class EPSSetupDialog(QDialog):
    """Calibrate a fixed Partaker EPS threshold on representative frames."""

    setup_accepted = Signal(object)

    DEFAULT_THRESHOLD = 40
    DEFAULT_OVERLAY_ALPHA_PERCENT = 45
    DEFAULT_PREVIEW_COUNT = 5
    OVERLAY_COLOR_OPTIONS = ("Red", "Blue", "Yellow", "Green")
    OVERLAY_COLOR_RGB = {
        "red": (255, 0, 0),
        "blue": (0, 153, 255),
        "yellow": (255, 221, 0),
        "green": (0, 255, 0),
    }
    TIME_UNIT_OPTIONS = ("ms", "sec", "min", "hr", "day")

    def __init__(
        self,
        *,
        image_data,
        eps_channel: int,
        selected_positions: Sequence[int] | None = None,
        time_start: int = 0,
        time_end: int | None = None,
        initial_drop_frame_zero: bool = False,
        initial_threshold: int = DEFAULT_THRESHOLD,
        smoothing_sigma: float = 1.5,
        initial_overlay_color: str = "Green",
        initial_capture_interval_value: float = 24.0,
        initial_capture_interval_unit: str = "hr",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        if image_data is None or getattr(image_data, "data", None) is None:
            raise ValueError("EPSSetupDialog requires loaded image data.")

        shape = tuple(int(value) for value in image_data.data.shape)
        if len(shape) < 3:
            raise ValueError(
                "Expected image data with at least T, P, and C dimensions; "
                f"received shape {shape}."
            )

        self.image_data = image_data
        self.time_count = shape[0]
        self.position_count = shape[1]
        self.channel_count = shape[2]
        self.eps_channel = int(eps_channel)

        if not 0 <= self.eps_channel < self.channel_count:
            raise ValueError(
                f"EPS channel {self.eps_channel} is outside 0.."
                f"{self.channel_count - 1}."
            )

        self.initial_threshold = int(np.clip(initial_threshold, 0, 255))
        self.initial_drop_frame_zero = bool(initial_drop_frame_zero)
        self.smoothing_sigma = float(max(0.0, smoothing_sigma))
        self.initial_overlay_color = self.normalize_overlay_color_name(
            initial_overlay_color
        ).title()
        self.initial_capture_interval_value = float(
            max(0.001, initial_capture_interval_value)
        )
        self.initial_capture_interval_unit = (
            initial_capture_interval_unit
            if initial_capture_interval_unit in self.TIME_UNIT_OPTIONS
            else "hr"
        )

        final_time = self.time_count - 1 if time_end is None else int(time_end)
        self.initial_time_start = int(np.clip(time_start, 0, self.time_count - 1))
        self.initial_time_end = int(
            np.clip(final_time, self.initial_time_start, self.time_count - 1)
        )

        default_positions = (
            sorted({int(position) for position in selected_positions})
            if selected_positions
            else list(range(self.position_count))
        )
        self.initial_positions = [
            position
            for position in default_positions
            if 0 <= position < self.position_count
        ]
        if not self.initial_positions:
            self.initial_positions = [0]

        self.processing_black_point_raw: float | None = None
        self.processing_white_point_raw: float | None = None
        self.sampled_frame_keys: tuple[tuple[int, int, int], ...] = ()
        self.representative_timepoints: tuple[int, ...] = ()
        self.result: EPSSetupResult | None = None

        # The cache holds only a small number of on-demand raw frames.
        self._raw_frame_cache: OrderedDict[
            tuple[int, int, int], np.ndarray
        ] = OrderedDict()
        self._raw_frame_cache_limit = 16
        self._updating_controls = False

        self.preview_update_timer = QTimer(self)
        self.preview_update_timer.setSingleShot(True)
        self.preview_update_timer.setInterval(100)
        self.preview_update_timer.timeout.connect(self.update_preview)

        self.thumbnail_update_timer = QTimer(self)
        self.thumbnail_update_timer.setSingleShot(True)
        self.thumbnail_update_timer.setInterval(180)
        self.thumbnail_update_timer.timeout.connect(self.update_thumbnails)

        self.scope_update_timer = QTimer(self)
        self.scope_update_timer.setSingleShot(True)
        self.scope_update_timer.setInterval(200)
        self.scope_update_timer.timeout.connect(self.refresh_scope)

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(500)
        self.play_timer.timeout.connect(self.advance_playback)

        print(f"EPS setup dialog layout: vertical_v4_drop_frame_zero | file={__file__}")
        self.setWindowTitle("EPS Fixed Threshold Setup — Vertical v4")
        self.setMinimumSize(980, 700)
        self.resize(1220, 900)

        self.init_ui()
        self.populate_initial_values()
        self.run_button.setEnabled(False)
        QTimer.singleShot(0, self.refresh_scope)

    @classmethod
    def normalize_overlay_color_name(cls, value: str | None) -> str:
        color_name = str(value or "green").strip().lower()
        if color_name not in cls.OVERLAY_COLOR_RGB:
            return "green"
        return color_name

    def current_overlay_color_name(self) -> str:
        if not hasattr(self, "overlay_color_combo"):
            return self.normalize_overlay_color_name(self.initial_overlay_color)
        return self.normalize_overlay_color_name(
            self.overlay_color_combo.currentText()
        )

    def current_overlay_color_rgb(self) -> tuple[int, int, int]:
        return self.OVERLAY_COLOR_RGB[self.current_overlay_color_name()]

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        heading = QLabel("Partaker EPS Fixed Threshold Calibration")
        heading.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #2196F3;"
        )
        main_layout.addWidget(heading)

        explanation = QLabel(
            "Choose the analysis scope, inspect a fixed EPS mask on individual "
            "frames, and validate it on five evenly spaced timepoints. The "
            "selected frames are scanned once to establish one shared min/max "
            "intensity mapping, but masks are generated only for previews until "
            "you confirm and run the analysis."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #666; padding-bottom: 4px;")
        main_layout.addWidget(explanation)

        # Top workspace: large preview on the left and analysis controls on the right.
        main_layout.addWidget(self.create_top_workspace(), 1)
        main_layout.addWidget(self.create_threshold_panel())
        main_layout.addWidget(self.create_time_navigation_group())
        main_layout.addWidget(self.create_validation_group())

        self.status_label = QLabel("Preparing representative EPS previews...")
        self.status_label.setStyleSheet(
            "color: #666; font-style: italic; padding: 4px;"
        )
        main_layout.addWidget(self.status_label)

        button_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset Threshold")
        self.reset_button.clicked.connect(self.reset_threshold)
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()

        self.button_box = QDialogButtonBox()
        self.cancel_button = self.button_box.addButton(
            "Cancel", QDialogButtonBox.RejectRole
        )
        self.run_button = self.button_box.addButton(
            "Confirm and Run Analysis", QDialogButtonBox.AcceptRole
        )
        self.run_button.setStyleSheet(
            "background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 7px 12px;"
        )
        self.button_box.rejected.connect(self.reject)
        self.button_box.accepted.connect(self.accept_setup)
        button_layout.addWidget(self.button_box)
        main_layout.addLayout(button_layout)

    def create_top_workspace(self) -> QWidget:
        """Place the expandable image preview beside the analysis-scope controls."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        viewer_panel = self.create_viewer_panel()
        scope_group = self.create_scope_group()

        # Give the preview most of the horizontal space while keeping the
        # right-side form wide enough for the frame-handling text.
        layout.addWidget(viewer_panel, 3)
        layout.addWidget(scope_group, 2)
        return widget

    def create_scope_group(self) -> QGroupBox:
        group = QGroupBox("Analysis scope")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(7)

        layout.addWidget(QLabel("Positions to analyze:"))
        self.position_list = QListWidget()
        self.position_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.position_list.setMinimumHeight(72)
        self.position_list.setMaximumHeight(105)
        self.position_list.itemSelectionChanged.connect(
            self.schedule_scope_refresh
        )
        layout.addWidget(self.position_list)

        position_buttons = QHBoxLayout()
        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(self.select_all_positions)
        select_none_button = QPushButton("Select None")
        select_none_button.clicked.connect(self.select_no_positions)
        position_buttons.addWidget(select_all_button)
        position_buttons.addWidget(select_none_button)
        layout.addLayout(position_buttons)

        scope_form = QFormLayout()
        scope_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        scope_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        scope_form.setVerticalSpacing(8)

        self.time_start_spin = QSpinBox()
        self.time_start_spin.setRange(0, self.time_count - 1)
        self.time_start_spin.valueChanged.connect(self.on_time_bounds_changed)
        scope_form.addRow("Start timepoint:", self.time_start_spin)

        self.time_end_spin = QSpinBox()
        self.time_end_spin.setRange(0, self.time_count - 1)
        self.time_end_spin.valueChanged.connect(self.on_time_bounds_changed)
        scope_form.addRow("End timepoint:", self.time_end_spin)

        self.drop_frame_zero_checkbox = QCheckBox(
            "Exclude frame 0 from previews and analysis"
        )
        self.drop_frame_zero_checkbox.setToolTip(
            "When enabled, T=0 is not used to estimate the shared EPS "
            "intensity window, is not shown in the preview controls, and is "
            "not sent to the full Partaker EPS processing queue."
        )
        self.drop_frame_zero_checkbox.toggled.connect(
            self.schedule_scope_refresh
        )
        scope_form.addRow("Frame handling:", self.drop_frame_zero_checkbox)

        interval_widget = QWidget()
        interval_layout = QHBoxLayout(interval_widget)
        interval_layout.setContentsMargins(0, 0, 0, 0)
        interval_layout.setSpacing(6)

        self.capture_interval_spin = QDoubleSpinBox()
        self.capture_interval_spin.setDecimals(3)
        self.capture_interval_spin.setRange(0.001, 1e9)
        self.capture_interval_spin.setSingleStep(0.1)
        self.capture_interval_spin.setValue(self.initial_capture_interval_value)
        self.capture_interval_spin.setFixedWidth(96)
        interval_layout.addWidget(self.capture_interval_spin)

        self.capture_interval_unit_combo = QComboBox()
        self.capture_interval_unit_combo.addItems(list(self.TIME_UNIT_OPTIONS))
        self.capture_interval_unit_combo.setCurrentText(
            self.initial_capture_interval_unit
        )
        interval_layout.addWidget(self.capture_interval_unit_combo)
        interval_layout.addStretch(1)
        scope_form.addRow("Capture interval:", interval_widget)

        self.preview_position_combo = QComboBox()
        self.preview_position_combo.currentIndexChanged.connect(
            self.on_preview_position_changed
        )
        scope_form.addRow("Preview position:", self.preview_position_combo)

        eps_channel_label = QLabel(f"Channel {self.eps_channel}")
        scope_form.addRow("EPS preview:", eps_channel_label)

        layout.addLayout(scope_form)
        layout.addStretch(1)
        group.setMinimumWidth(390)
        return group

    def create_viewer_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.addWidget(QLabel("View:"))

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Overlay", "Raw EPS", "Mask"])
        self.view_mode_combo.currentTextChanged.connect(
            self.schedule_preview_update
        )
        toolbar.addWidget(self.view_mode_combo)

        toolbar.addWidget(QLabel("Overlay color:"))
        self.overlay_color_combo = QComboBox()
        self.overlay_color_combo.addItems(list(self.OVERLAY_COLOR_OPTIONS))
        self.overlay_color_combo.setCurrentText(self.initial_overlay_color)
        self.overlay_color_combo.currentTextChanged.connect(
            self.on_overlay_color_changed
        )
        toolbar.addWidget(self.overlay_color_combo)

        fit_button = QPushButton("Fit Image")
        fit_button.setToolTip(
            "Show the entire image. Double-clicking the preview also fits it."
        )
        fit_button.clicked.connect(self.fit_preview)
        toolbar.addWidget(fit_button)

        self.frame_info_label = QLabel("No preview frame loaded")
        self.frame_info_label.setStyleSheet("font-weight: bold;")
        toolbar.addWidget(self.frame_info_label, 1)
        layout.addLayout(toolbar)

        self.image_view = ZoomableImageView()
        self.image_view.setMinimumSize(520, 320)
        layout.addWidget(self.image_view, 1)
        return widget

    def create_threshold_panel(self) -> QGroupBox:
        group = QGroupBox("Fixed threshold controls")
        outer_layout = QVBoxLayout(group)
        outer_layout.setContentsMargins(10, 8, 10, 8)
        outer_layout.setSpacing(5)

        threshold_row = QHBoxLayout()
        threshold_row.setSpacing(7)
        threshold_row.addWidget(QLabel("Threshold:"))

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setFixedWidth(72)
        self.threshold_spin.valueChanged.connect(
            self.on_threshold_spin_changed
        )
        threshold_row.addWidget(self.threshold_spin)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(20)
        self.threshold_slider.valueChanged.connect(
            self.on_threshold_slider_changed
        )
        threshold_row.addWidget(self.threshold_slider, 1)

        threshold_row.addWidget(QLabel("Quick:"))
        for value in (20, 40, 60, 80, 100):
            button = QPushButton(str(value))
            button.setFixedWidth(48)
            button.clicked.connect(
                lambda _checked=False, v=value: self.set_threshold(v)
            )
            threshold_row.addWidget(button)

        outer_layout.addLayout(threshold_row)

        options_row = QHBoxLayout()
        options_row.setSpacing(7)
        options_row.addWidget(QLabel("Gaussian sigma:"))

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.0, 20.0)
        self.sigma_spin.setDecimals(2)
        self.sigma_spin.setSingleStep(0.25)
        self.sigma_spin.setValue(self.smoothing_sigma)
        self.sigma_spin.setFixedWidth(78)
        self.sigma_spin.valueChanged.connect(self.on_sigma_changed)
        options_row.addWidget(self.sigma_spin)

        options_row.addSpacing(16)
        options_row.addWidget(QLabel("Overlay opacity:"))

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(self.DEFAULT_OVERLAY_ALPHA_PERCENT)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        options_row.addWidget(self.opacity_slider, 1)

        self.opacity_value_label = QLabel(
            f"{self.DEFAULT_OVERLAY_ALPHA_PERCENT}%"
        )
        self.opacity_value_label.setMinimumWidth(42)
        options_row.addWidget(self.opacity_value_label)
        outer_layout.addLayout(options_row)

        readout_row = QHBoxLayout()
        readout_row.setSpacing(8)

        self.foreground_label = QLabel("Foreground: —")
        self.foreground_label.setWordWrap(True)
        self.foreground_label.setStyleSheet(
            "padding: 4px; border: 1px solid #555;"
        )
        readout_row.addWidget(self.foreground_label, 2)

        self.raw_threshold_label = QLabel("Raw-equivalent threshold: —")
        self.raw_threshold_label.setWordWrap(True)
        self.raw_threshold_label.setStyleSheet(
            "padding: 4px; border: 1px solid #555;"
        )
        readout_row.addWidget(self.raw_threshold_label, 1)

        self.window_label = QLabel("Shared intensity window: —")
        self.window_label.setWordWrap(True)
        self.window_label.setStyleSheet(
            "color: #888; padding: 4px; border: 1px solid #555;"
        )
        readout_row.addWidget(self.window_label, 2)
        outer_layout.addLayout(readout_row)

        group.setMaximumHeight(150)
        return group

    def create_time_navigation_group(self) -> QGroupBox:
        group = QGroupBox("Timepoint preview")
        group.setMaximumHeight(68)
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
        group = QGroupBox("Representative-frame validation")
        group.setMaximumHeight(92)

        layout = QHBoxLayout(group)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        self.thumbnail_buttons: list[QToolButton] = []
        for index in range(self.DEFAULT_PREVIEW_COUNT):
            button = QToolButton()
            button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            button.setIconSize(QPixmap(78, 52).size())
            button.setMinimumSize(120, 60)
            button.setMaximumHeight(66)
            button.setEnabled(False)
            button.clicked.connect(
                lambda _checked=False, i=index: self.open_thumbnail(i)
            )
            layout.addWidget(button, 1)
            self.thumbnail_buttons.append(button)

        return group

    # ------------------------------------------------------------------
    # Initialization and scope management
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
            self.capture_interval_spin.setValue(
                self.initial_capture_interval_value
            )
            self.capture_interval_unit_combo.setCurrentText(
                self.initial_capture_interval_unit
            )
            self.overlay_color_combo.setCurrentText(self.initial_overlay_color)
            self.threshold_slider.setValue(self.initial_threshold)
            self.threshold_spin.setValue(self.initial_threshold)
        finally:
            self._updating_controls = False

    def selected_positions(self) -> list[int]:
        selected = []
        for index in range(self.position_list.count()):
            if self.position_list.item(index).isSelected():
                selected.append(index)
        return selected

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
        if self._updating_controls:
            return
        self.scope_update_timer.start()

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

    def considered_timepoints(self) -> list[int]:
        """Return selected timepoints after applying frame-zero exclusion."""
        start = int(self.time_start_spin.value())
        end = int(self.time_end_spin.value())

        timepoints = list(range(start, end + 1))
        if self.drop_frame_zero_checkbox.isChecked():
            timepoints = [time for time in timepoints if time != 0]

        return timepoints

    def refresh_scope(self) -> None:
        positions = self.selected_positions()
        if not positions:
            self.processing_black_point_raw = None
            self.processing_white_point_raw = None
            self.sampled_frame_keys = ()
            self.representative_timepoints = ()
            self.preview_position_combo.clear()
            self.image_view.set_image(QPixmap())
            self.status_label.setText("Select at least one position.")
            self.run_button.setEnabled(False)
            self.clear_thumbnails()
            return

        start = int(self.time_start_spin.value())
        end = int(self.time_end_spin.value())
        if end < start:
            self.status_label.setText("The end timepoint must not precede start.")
            self.run_button.setEnabled(False)
            return

        considered_timepoints = self.considered_timepoints()
        if not considered_timepoints:
            self.processing_black_point_raw = None
            self.processing_white_point_raw = None
            self.sampled_frame_keys = ()
            self.representative_timepoints = ()
            self.image_view.set_image(QPixmap())
            self.clear_thumbnails()
            self.run_button.setEnabled(False)
            self.status_label.setText(
                "No frames remain in the selected range because frame 0 is "
                "excluded. Extend the range beyond T=0 or include frame 0."
            )
            return

        preview_start = int(considered_timepoints[0])
        preview_end = int(considered_timepoints[-1])

        previous_preview_position = self.current_preview_position()
        self._updating_controls = True
        try:
            self.preview_position_combo.clear()
            for position in positions:
                self.preview_position_combo.addItem(
                    f"Position {position}", position
                )

            desired_position = (
                previous_preview_position
                if previous_preview_position in positions
                else positions[0]
            )
            desired_index = self.preview_position_combo.findData(
                desired_position
            )
            self.preview_position_combo.setCurrentIndex(max(0, desired_index))

            current_time = (
                self.time_slider.value()
                if self.time_slider.maximum() >= self.time_slider.minimum()
                else preview_start
            )
            self.time_slider.setRange(preview_start, preview_end)
            self.time_slider.setValue(
                int(np.clip(current_time, preview_start, preview_end))
            )
        finally:
            self._updating_controls = False

        excluded_note = (
            " Frame 0 is excluded."
            if self.drop_frame_zero_checkbox.isChecked() and start == 0
            else ""
        )
        self.status_label.setText(
            "Estimating one shared EPS intensity window from considered "
            f"frames...{excluded_note}"
        )
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.estimate_processing_window(
                positions,
                considered_timepoints,
            )
            self.representative_timepoints = tuple(
                self.evenly_spaced_values(
                    preview_start,
                    preview_end,
                    self.DEFAULT_PREVIEW_COUNT,
                )
            )
            self.update_window_labels()
            self.update_thumbnails()
            self.update_preview()
            self.run_button.setEnabled(True)
            self.status_label.setText(
                f"Ready. Previewing {len(self.representative_timepoints)} "
                "representative timepoints without generating the full mask "
                f"set.{excluded_note}"
            )
        except Exception as error:
            self.processing_black_point_raw = None
            self.processing_white_point_raw = None
            self.sampled_frame_keys = ()
            self.run_button.setEnabled(False)
            self.clear_thumbnails()
            self.status_label.setText(f"Could not prepare previews: {error}")
            QMessageBox.warning(
                self,
                "EPS preview setup failed",
                f"Could not prepare the shared EPS preview window:\n{error}",
            )
        finally:
            QApplication.restoreOverrideCursor()

    def current_preview_position(self) -> int | None:
        if self.preview_position_combo.count() == 0:
            return None
        value = self.preview_position_combo.currentData()
        return None if value is None else int(value)

    def on_preview_position_changed(self) -> None:
        if self._updating_controls:
            return
        self.update_thumbnails()
        self.schedule_preview_update()

    @staticmethod
    def evenly_spaced_values(start: int, end: int, count: int) -> list[int]:
        if end <= start:
            return [start]
        values = np.linspace(start, end, min(count, end - start + 1), dtype=int)
        return [int(value) for value in np.unique(values)]

    def estimate_processing_window(
        self,
        positions: Sequence[int],
        timepoints: Sequence[int],
    ) -> None:
        frame_keys = [
            (int(time), int(position), self.eps_channel)
            for position in positions
            for time in timepoints
        ]
        if not frame_keys:
            raise ValueError("No frames are available in the selected scope.")

        global_min = np.inf
        global_max = -np.inf

        for frame_number, key in enumerate(frame_keys, start=1):
            frame = self.get_raw_frame(*key)
            finite_values = frame[np.isfinite(frame)]
            if finite_values.size == 0:
                continue

            global_min = min(global_min, float(np.min(finite_values)))
            global_max = max(global_max, float(np.max(finite_values)))
            self.status_label.setText(
                f"Scanning shared intensity range: {frame_number}/"
                f"{len(frame_keys)} selected frames"
            )
            QApplication.processEvents()

        if not np.isfinite(global_min) or not np.isfinite(global_max):
            raise ValueError("Selected frames contain no finite values.")
        if global_max <= global_min:
            raise ValueError(
                "The selected EPS frames have no usable intensity range: "
                f"min={global_min}, max={global_max}."
            )

        self.processing_black_point_raw = float(global_min)
        self.processing_white_point_raw = float(global_max)
        self.sampled_frame_keys = tuple(frame_keys)

    # ------------------------------------------------------------------
    # Frame processing and display
    # ------------------------------------------------------------------

    def get_raw_frame(
        self,
        time: int,
        position: int,
        channel: int,
    ) -> np.ndarray:
        key = (int(time), int(position), int(channel))
        cached = self._raw_frame_cache.get(key)
        if cached is not None:
            self._raw_frame_cache.move_to_end(key)
            return cached

        frame = np.asarray(
            self.image_data.get(*key),
            dtype=np.float32,
        )
        frame = np.squeeze(frame)
        if frame.ndim != 2:
            raise ValueError(
                f"Expected a 2D EPS frame for T={time}, P={position}, "
                f"C={channel}; received shape {frame.shape}."
            )

        self._raw_frame_cache[key] = frame
        self._raw_frame_cache.move_to_end(key)
        while len(self._raw_frame_cache) > self._raw_frame_cache_limit:
            self._raw_frame_cache.popitem(last=False)
        return frame

    def normalize_to_shared_uint8(self, frame: np.ndarray) -> np.ndarray:
        black = self.processing_black_point_raw
        white = self.processing_white_point_raw
        if black is None or white is None or white <= black:
            raise RuntimeError("The shared EPS intensity window is unavailable.")

        work = np.asarray(frame, dtype=np.float32).copy()
        np.nan_to_num(
            work,
            copy=False,
            nan=float(black),
            posinf=float(white),
            neginf=float(black),
        )
        work -= np.float32(black)
        work /= np.float32(white - black)
        np.clip(work, 0.0, 1.0, out=work)
        work *= np.float32(255.0)
        np.rint(work, out=work)
        return work.astype(np.uint8, copy=False)

    def create_processed_preview(
        self,
        time: int,
        position: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_frame = self.get_raw_frame(time, position, self.eps_channel)
        normalized = self.normalize_to_shared_uint8(raw_frame)

        smoothed = np.empty_like(normalized, dtype=np.uint8)
        gaussian_filter(
            normalized,
            sigma=float(self.smoothing_sigma),
            output=smoothed,
        )
        mask = smoothed >= self.threshold_spin.value()
        return normalized, smoothed, mask

    def make_display_rgb(
        self,
        normalized: np.ndarray,
        mask: np.ndarray,
        view_mode: str | None = None,
    ) -> np.ndarray:
        mode = self.view_mode_combo.currentText() if view_mode is None else view_mode

        if mode == "Raw EPS":
            return np.repeat(normalized[..., None], 3, axis=2)
        if mode == "Mask":
            mask_uint8 = mask.astype(np.uint8) * 255
            return np.repeat(mask_uint8[..., None], 3, axis=2)

        rgb = np.repeat(normalized[..., None], 3, axis=2).astype(np.float32)
        alpha = self.opacity_slider.value() / 100.0
        overlay_color = np.asarray(
            self.current_overlay_color_rgb(),
            dtype=np.float32,
        )
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * overlay_color
        return np.clip(rgb, 0, 255).astype(np.uint8)

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

    def schedule_preview_update(self) -> None:
        if self._updating_controls:
            return
        self.preview_update_timer.start()

    def update_preview(self) -> None:
        position = self.current_preview_position()
        if position is None or self.processing_black_point_raw is None:
            return

        time = self.time_slider.value()
        try:
            normalized, _smoothed, mask = self.create_processed_preview(
                time, position
            )
            rgb = self.make_display_rgb(normalized, mask)
            self.image_view.set_image(self.rgb_to_pixmap(rgb))

            foreground_pixels = int(np.count_nonzero(mask))
            total_pixels = int(mask.size)
            foreground_percent = (
                100.0 * foreground_pixels / total_pixels
                if total_pixels
                else 0.0
            )
            self.foreground_label.setText(
                f"Foreground: {foreground_percent:.2f}% "
                f"({foreground_pixels:,}/{total_pixels:,} pixels)"
            )
            self.frame_info_label.setText(
                f"Position {position} • Timepoint {time} • "
                f"EPS Channel {self.eps_channel} • "
                f"Threshold {self.threshold_spin.value()}"
            )
            self.time_value_label.setText(f"T={time}")
            self.update_raw_threshold_label()
        except Exception as error:
            self.status_label.setText(f"Preview failed: {error}")

    def fit_preview(self) -> None:
        self.image_view._auto_fit_enabled = True
        self.image_view.request_fit()

    def update_thumbnails(self) -> None:
        self.clear_thumbnails()
        position = self.current_preview_position()
        if position is None or self.processing_black_point_raw is None:
            return

        for index, time in enumerate(self.representative_timepoints):
            if index >= len(self.thumbnail_buttons):
                break
            try:
                normalized, _smoothed, mask = self.create_processed_preview(
                    time, position
                )
                rgb = self.make_display_rgb(normalized, mask, "Overlay")
                pixmap = self.rgb_to_pixmap(rgb).scaled(
                    78,
                    52,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                button = self.thumbnail_buttons[index]
                button.setVisible(True)
                button.setIcon(QIcon(pixmap))
                button.setText(f"T={time}")
                button.setProperty("timepoint", int(time))
                button.setEnabled(True)
            except Exception as error:
                button = self.thumbnail_buttons[index]
                button.setText(f"T={time}\nUnavailable")
                button.setToolTip(str(error))

    def clear_thumbnails(self) -> None:
        for button in self.thumbnail_buttons:
            button.setIcon(QIcon())
            button.setText("—")
            button.setProperty("timepoint", None)
            button.setEnabled(False)
            button.setVisible(False)

    def open_thumbnail(self, index: int) -> None:
        if not 0 <= index < len(self.thumbnail_buttons):
            return
        time = self.thumbnail_buttons[index].property("timepoint")
        if time is not None:
            self.time_slider.setValue(int(time))

    # ------------------------------------------------------------------
    # Control callbacks
    # ------------------------------------------------------------------

    def set_threshold(self, value: int) -> None:
        self.threshold_slider.setValue(int(np.clip(value, 0, 255)))

    def on_threshold_slider_changed(self, value: int) -> None:
        if self._updating_controls:
            return
        self._updating_controls = True
        try:
            self.threshold_spin.setValue(value)
        finally:
            self._updating_controls = False
        self.update_raw_threshold_label()
        self.schedule_preview_update()
        self.schedule_thumbnail_update()

    def on_threshold_spin_changed(self, value: int) -> None:
        if self._updating_controls:
            return
        self._updating_controls = True
        try:
            self.threshold_slider.setValue(value)
        finally:
            self._updating_controls = False
        self.update_raw_threshold_label()
        self.schedule_preview_update()
        self.schedule_thumbnail_update()

    def on_sigma_changed(self, value: float) -> None:
        self.smoothing_sigma = float(value)
        self.schedule_preview_update()
        self.schedule_thumbnail_update()

    def on_opacity_changed(self, value: int) -> None:
        if hasattr(self, "opacity_value_label"):
            self.opacity_value_label.setText(f"{int(value)}%")
        self.schedule_preview_update()
        self.schedule_thumbnail_update()

    def on_overlay_color_changed(self, _value: str) -> None:
        self.schedule_preview_update()
        self.schedule_thumbnail_update()

    def schedule_thumbnail_update(self) -> None:
        if self._updating_controls:
            return
        self.thumbnail_update_timer.start()

    def update_window_labels(self) -> None:
        black = self.processing_black_point_raw
        white = self.processing_white_point_raw
        if black is None or white is None:
            self.window_label.setText("Shared intensity window: —")
            return
        exclusion_text = (
            "; T=0 excluded"
            if self.drop_frame_zero_checkbox.isChecked()
            else ""
        )
        self.window_label.setText(
            "Shared raw intensity window: "
            f"{black:.3f} to {white:.3f}\n"
            f"Scanned across {len(self.sampled_frame_keys)} considered frames"
            f"{exclusion_text}"
        )
        self.update_raw_threshold_label()

    def update_raw_threshold_label(self) -> None:
        black = self.processing_black_point_raw
        white = self.processing_white_point_raw
        if black is None or white is None or white <= black:
            self.raw_threshold_label.setText("Raw-equivalent threshold: —")
            return
        threshold = self.threshold_spin.value()
        raw_equivalent = black + (threshold / 255.0) * (white - black)
        self.raw_threshold_label.setText(
            f"Raw-equivalent threshold: {raw_equivalent:.3f}"
        )

    def reset_threshold(self) -> None:
        self.set_threshold(self.initial_threshold)
        self.opacity_slider.setValue(self.DEFAULT_OVERLAY_ALPHA_PERCENT)
        self.view_mode_combo.setCurrentText("Overlay")
        self.overlay_color_combo.setCurrentText(self.initial_overlay_color)
        self.sigma_spin.setValue(1.5)

    def on_time_slider_changed(self, value: int) -> None:
        self.time_value_label.setText(f"T={value}")
        if not self._updating_controls:
            self.schedule_preview_update()

    def go_to_first(self) -> None:
        self.time_slider.setValue(self.time_slider.minimum())

    def go_to_middle(self) -> None:
        self.time_slider.setValue(
            (self.time_slider.minimum() + self.time_slider.maximum()) // 2
        )

    def go_to_last(self) -> None:
        self.time_slider.setValue(self.time_slider.maximum())

    def go_to_previous(self) -> None:
        self.time_slider.setValue(
            max(self.time_slider.minimum(), self.time_slider.value() - 1)
        )

    def go_to_next(self) -> None:
        self.time_slider.setValue(
            min(self.time_slider.maximum(), self.time_slider.value() + 1)
        )

    def toggle_playback(self, playing: bool) -> None:
        if playing:
            self.play_button.setText("Pause")
            self.play_timer.start()
        else:
            self.play_button.setText("Play")
            self.play_timer.stop()

    def advance_playback(self) -> None:
        if self.time_slider.value() >= self.time_slider.maximum():
            self.time_slider.setValue(self.time_slider.minimum())
        else:
            self.time_slider.setValue(self.time_slider.value() + 1)

    # ------------------------------------------------------------------
    # Acceptance
    # ------------------------------------------------------------------

    def accept_setup(self) -> None:
        positions = self.selected_positions()
        if not positions:
            QMessageBox.warning(
                self,
                "No positions selected",
                "Select at least one position before running EPS analysis.",
            )
            return

        if (
            self.processing_black_point_raw is None
            or self.processing_white_point_raw is None
            or self.processing_white_point_raw
            <= self.processing_black_point_raw
        ):
            QMessageBox.warning(
                self,
                "EPS preview not ready",
                "The shared EPS intensity window could not be prepared.",
            )
            return

        preview_position = self.current_preview_position()
        if preview_position is None:
            preview_position = positions[0]

        self.result = EPSSetupResult(
            positions=tuple(positions),
            time_start=int(self.time_start_spin.value()),
            time_end=int(self.time_end_spin.value()),
            drop_frame_zero=bool(
                self.drop_frame_zero_checkbox.isChecked()
            ),
            eps_channel=int(self.eps_channel),
            threshold_uint8=int(self.threshold_spin.value()),
            processing_black_point_raw=float(
                self.processing_black_point_raw
            ),
            processing_white_point_raw=float(
                self.processing_white_point_raw
            ),
            smoothing_sigma=float(self.smoothing_sigma),
            preview_position=int(preview_position),
            representative_timepoints=tuple(self.representative_timepoints),
            sampled_frame_keys=tuple(self.sampled_frame_keys),
            overlay_color_name=self.current_overlay_color_name(),
            capture_interval_value=float(self.capture_interval_spin.value()),
            capture_interval_unit=str(
                self.capture_interval_unit_combo.currentText()
            ),
        )
        self.setup_accepted.emit(self.result)
        self.accept()

    def reject(self) -> None:  # noqa: A003 - Qt API name
        self.play_timer.stop()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.play_timer.stop()
        super().closeEvent(event)