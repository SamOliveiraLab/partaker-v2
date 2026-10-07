"""Interactive calibration dialog for Partaker fixed Intensity thresholding.

The dialog previews a single, globally consistent fixed threshold without
creating masks for every selected frame.  It estimates one shared raw-to-uint8
mapping from representative frames, then reuses that mapping for the large
preview and the five validation thumbnails.

The accepted :class:`LiveDeadResult` is intended to be passed back to
``LiveDeadAnalysisWidget`` so the full analysis uses the exact same positions, time
range, threshold, smoothing, and intensity mapping that the user approved.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Event
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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class LiveDeadResult:
    """Settings approved in :class:`LiveDeadDialog`."""

    positions: tuple[int, ...]
    time_start: int
    time_end: int
    drop_frame_zero: bool
    live_channel: int
    dead_channel: int | None
    threshold_uint8: int
    processing_black_point_raw: float
    processing_white_point_raw: float
    smoothing_sigma: float
    preview_position: int
    representative_timepoints: tuple[int, ...]
    sampled_frame_keys: tuple[tuple[int, int, int], ...]
    analysis_mode: str = "live_dead"
    capture_interval_value: float = 24.0
    capture_interval_unit: str = "hr"
    processing_window_source: str = (
        "partaker_setup_dialog_full_selected_scope_min_max"
    )
    threshold_rule: str = "foreground = shared_uint8 >= threshold"
    background_enabled: bool = False
    background_margin_pixels: int = 15
    background_upper_cutoffs: tuple[tuple[int, float], ...] = ()
    background_include_cells: bool = True
    background_signal_thresholds: tuple[tuple[int, float], ...] = ()
    background_method: str = "cell_exclusion_median"
    background_samples: tuple[tuple[int, int, int, int, int, int, int], ...] = ()

    def to_dict(self) -> dict:
        """Return a serialization-friendly copy of the result."""
        values = asdict(self)
        if self.analysis_mode == "cell_viability":
            values["cell_viability_channel"] = values.pop("live_channel")
            values.pop("dead_channel")
        return values


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


class BackgroundSampleImageView(ZoomableImageView):
    """Click-to-place sample rectangles while retaining normal pan/zoom."""
    sample_clicked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sample_mode = False
        self._sample_press = None

    def set_sample_mode(self, enabled):
        self.sample_mode = bool(enabled)
        self.setDragMode(QGraphicsView.NoDrag if enabled else QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self.sample_mode and event.button() == Qt.LeftButton:
            self._sample_press = event.position()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.sample_mode and event.button() == Qt.LeftButton:
            if self._sample_press is not None and (event.position() - self._sample_press).manhattanLength() < 5:
                point = self.mapToScene(event.position().toPoint())
                if self._pixmap_item.boundingRect().contains(point):
                    self.sample_clicked.emit(point.x(), point.y())
            self._sample_press = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def set_image(self, pixmap):
        if pixmap.isNull() or self._pixmap_item.pixmap().isNull():
            super().set_image(pixmap)
            return
        # Updating sample fills must not discard the user's zoom or pan.
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())


class LiveDeadDialog(QDialog):
    """Calibrate a fixed Partaker live fluorescence intensity threshold on representative frames."""

    setup_accepted = Signal(object)

    DEFAULT_THRESHOLD = 40
    DEFAULT_OVERLAY_ALPHA_PERCENT = 45
    DEFAULT_PREVIEW_COUNT = 5
    LIVE_OVERLAY_RGB = (0, 255, 0)
    DEAD_OVERLAY_RGB = (255, 0, 0)
    TIME_UNIT_OPTIONS = ("ms", "sec", "min", "hr", "day")

    def __init__(
        self,
        *,
        image_data,
        live_channel: int,
        dead_channel: int | None,
        analysis_mode: str = "live_dead",
        selected_positions: Sequence[int] | None = None,
        time_start: int = 0,
        time_end: int | None = None,
        excluded_timepoints: Sequence[int] = (),
        cell_label_provider=None,
        distance_provider=None,
        background_estimator=None,
        sample_estimator=None,
        initial_drop_frame_zero: bool = False,
        initial_threshold: int = DEFAULT_THRESHOLD,
        smoothing_sigma: float = 1.5,
        initial_capture_interval_value: float = 24.0,
        initial_capture_interval_unit: str = "hr",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        if image_data is None or getattr(image_data, "data", None) is None:
            raise ValueError("LiveDeadDialog requires loaded image data.")

        shape = tuple(int(value) for value in image_data.data.shape)
        if len(shape) < 3:
            raise ValueError(
                "Expected image data with at least T, P, and C dimensions; "
                f"received shape {shape}."
            )

        self.excluded_timepoints = set(int(t) for t in excluded_timepoints)
        self._last_preview_time = 0
        self.cell_label_provider = cell_label_provider
        self.distance_provider = distance_provider
        self.background_estimator = background_estimator
        self.sample_estimator = sample_estimator
        self.background_samples = {}
        self._background_errors = []
        self._background_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cell-background")
        self._background_cancel = Event()
        self._background_generation = 0
        self._background_tasks = []
        self._background_records = {}
        self._background_preview_serial = 0
        self._background_preview_data = None
        self._background_preview_cache = OrderedDict()
        self._background_threshold_ranges = {}
        self._background_signal_seeded = False
        self._background_closed = False
        self.background_enabled = False
        self.image_data = image_data
        self.time_count = shape[0]
        self.position_count = shape[1]
        self.channel_count = shape[2]
        if analysis_mode not in ("live_dead", "cell_viability"):
            raise ValueError(f"Unknown analysis mode: {analysis_mode}")
        self.analysis_mode = analysis_mode
        self.is_cell_viability = analysis_mode == "cell_viability"
        if self.is_cell_viability:
            self.LIVE_OVERLAY_RGB = self.DEAD_OVERLAY_RGB
        self.analysis_label = "Cell Viability" if self.is_cell_viability else "Live-Dead"
        self.primary_label = "Cell Viability" if self.is_cell_viability else "Live"
        self.live_channel = int(live_channel)
        self.dead_channel = None if self.is_cell_viability else int(dead_channel)
        self.active_channels = ((self.live_channel,) if self.is_cell_viability
                                else (self.live_channel, self.dead_channel))

        if not 0 <= self.live_channel < self.channel_count:
            raise ValueError(
                f"{self.primary_label} Fluorescence channel {self.live_channel} is outside 0.."
                f"{self.channel_count - 1}."
            )
        if self.dead_channel is not None and not 0 <= self.dead_channel < self.channel_count:
            raise ValueError(
                f"Dead Fluorescence channel {self.dead_channel} is outside 0.."
                f"{self.channel_count - 1}."
            )
        self.initial_threshold = int(np.clip(initial_threshold, 0, 255))
        self.initial_drop_frame_zero = bool(initial_drop_frame_zero)
        self.smoothing_sigma = float(max(0.0, smoothing_sigma))
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
        self.result: LiveDeadResult | None = None

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

        self.background_update_timer = QTimer(self)
        self.background_update_timer.setSingleShot(True)
        self.background_update_timer.setInterval(250)
        self.background_update_timer.timeout.connect(self.refresh_scope)
        self.background_poll_timer = QTimer(self)
        self.background_poll_timer.setInterval(50)
        self.background_poll_timer.timeout.connect(self.poll_background_tasks)

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(500)
        self.play_timer.timeout.connect(self.advance_playback)

        print(f"{self.analysis_label} dialog layout: | file={__file__}")
        self.setWindowTitle(f"{self.analysis_label} Fixed Threshold Setup")
        self.setMinimumSize(980, 700)
        self.resize(1220, 900)

        self.init_ui()
        self.populate_initial_values()
        self.run_button.setEnabled(False)
        QTimer.singleShot(0, self.refresh_scope)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        heading = QLabel(f"{self.analysis_label} Fluorescence Intensity Threshold")
        heading.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #2196F3;"
        )
        main_layout.addWidget(heading)

        explanation = QLabel(
            "Choose the analysis scope, inspect the fixed-threshold Cell Viability mask, "
            "and validate it on five evenly spaced timepoints. The selected channel "
            "establishes one min/max intensity mapping; masks are generated only "
            "for previews until you confirm and run the analysis."
            if self.is_cell_viability else
            "Choose the analysis scope, inspect paired fixed-threshold Live and "
            "Dead masks, and validate them on five evenly spaced timepoints. Both "
            "channels are scanned together to establish one shared min/max "
            "intensity mapping, but masks are generated only for previews until "
            "you confirm and run the analysis."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #666; padding-bottom: 4px;")
        main_layout.addWidget(explanation)

        # Top workspace: large preview on the left and analysis controls on the right.
        self.setup_tabs = QTabWidget()
        threshold_tab = QWidget()
        threshold_layout = QVBoxLayout(threshold_tab)
        threshold_layout.addWidget(self.create_top_workspace(), 1)
        threshold_layout.addWidget(self.create_threshold_panel())
        threshold_layout.addWidget(self.create_validation_group())
        self.setup_tabs.addTab(threshold_tab, "Fluorescence Threshold")
        self.setup_tabs.addTab(self.create_background_panel(), "Background Samples")
        self.setup_tabs.currentChanged.connect(self.on_setup_tab_changed)
        main_layout.addWidget(self.setup_tabs, 1)
        main_layout.addWidget(self.create_time_navigation_group())

        self.status_label = QLabel(f"Preparing representative {self.analysis_label} previews...")
        self.status_label.setStyleSheet(
            "color: #666; font-style: italic; padding: 4px;"
        )
        main_layout.addWidget(self.status_label)

        button_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset Threshold")
        self.reset_button.clicked.connect(self.reset_threshold)
        button_layout.addWidget(self.reset_button)
        self.back_button = QPushButton("Back to Threshold")
        self.back_button.clicked.connect(lambda: self.setup_tabs.setCurrentIndex(0))
        self.back_button.setVisible(False)
        button_layout.addWidget(self.back_button)
        button_layout.addStretch()

        self.button_box = QDialogButtonBox()
        self.cancel_button = self.button_box.addButton(
            "Cancel", QDialogButtonBox.RejectRole
        )
        self.run_button = self.button_box.addButton(
            "Confirm Threshold and Continue", QDialogButtonBox.AcceptRole
        )
        self.run_button.setStyleSheet(
            "background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 7px 12px;"
        )
        self.skip_correction_button = self.button_box.addButton(
            "Skip correction", QDialogButtonBox.ActionRole
        )
        self.skip_correction_button.setVisible(False)
        self.skip_correction_button.clicked.connect(self.skip_correction)
        self.button_box.rejected.connect(self.reject)
        self.button_box.accepted.connect(self.accept_setup)
        button_layout.addWidget(self.button_box)
        main_layout.addLayout(button_layout)

    def create_background_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        description = QLabel("Select three cell-free background rectangles for each position, timepoint and channel. "
                             "The pooled mean of their finite original intensities is subtracted before fluorescence smoothing. "
                             "Blue rectangles are sampled areas; the fluorescence threshold and raw intensity mapping stay fixed.")
        description.setWordWrap(True)
        layout.addWidget(description)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Position:"))
        self.background_position_combo = QComboBox()
        self.background_position_combo.currentIndexChanged.connect(self.on_background_position_changed)
        controls.addWidget(self.background_position_combo)
        controls.addWidget(QLabel("Preview image:"))
        self.background_channel_combo = QComboBox()
        self.background_channel_combo.addItem(self.primary_label, self.live_channel)
        if not self.is_cell_viability:
            self.background_channel_combo.addItem("Dead", self.dead_channel)
        self.background_channel_combo.currentIndexChanged.connect(self.on_background_channel_changed)
        controls.addWidget(self.background_channel_combo)
        self.place_sample_button = QPushButton("Place sample")
        self.place_sample_button.setCheckable(True)
        controls.addWidget(self.place_sample_button)
        self.background_show_cells = QCheckBox("Show saved cell outlines")
        self.background_show_cells.toggled.connect(self.render_background_preview)
        controls.addWidget(self.background_show_cells)
        layout.addLayout(controls)
        editor = QHBoxLayout()
        self.background_sample_combo = QComboBox()
        for i in range(3): self.background_sample_combo.addItem(f"Sample {i + 1}", i)
        self.background_sample_combo.currentIndexChanged.connect(self.sync_sample_editor)
        editor.addWidget(self.background_sample_combo)
        self.sample_spins = {}
        for label, value in (("X", 0), ("Y", 0), ("Width", 100), ("Height", 100)):
            editor.addWidget(QLabel(label + ":"))
            spin = QSpinBox()
            spin.setRange(0 if label in ("X", "Y") else 1, 100000)
            spin.setValue(value)
            spin.valueChanged.connect(self.edit_background_sample)
            self.sample_spins[label] = spin
            editor.addWidget(spin)
        self.delete_sample_button = QPushButton("Remove sample")
        self.delete_sample_button.clicked.connect(self.remove_background_sample)
        editor.addWidget(self.delete_sample_button)
        self.copy_samples_button = QPushButton("Copy samples to other channel")
        self.copy_samples_button.setVisible(not self.is_cell_viability)
        self.copy_samples_button.clicked.connect(self.copy_background_samples)
        editor.addWidget(self.copy_samples_button)
        layout.addLayout(editor)
        hint = QLabel("Choose Place sample and click the image to center the selected rectangle. "
                      "Turn it off to pan; scroll to zoom. Edit X/Y/Width/Height to move or resize a sample. "
                      "Samples must stay inside the image and must not overlap.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.background_image_view = BackgroundSampleImageView()
        self.background_image_view.sample_clicked.connect(self.place_background_sample)
        self.place_sample_button.toggled.connect(self.background_image_view.set_sample_mode)
        views = QHBoxLayout()
        original_column = QVBoxLayout()
        original_column.addWidget(QLabel("Original fluorescence with background samples"))
        original_column.addWidget(self.background_image_view, 1)
        corrected_column = QVBoxLayout()
        self.background_corrected_label = QLabel("Select three samples to preview subtraction")
        corrected_column.addWidget(self.background_corrected_label)
        self.background_corrected_view = BackgroundSampleImageView()
        corrected_column.addWidget(self.background_corrected_view, 1)
        views.addLayout(original_column, 1)
        views.addLayout(corrected_column, 1)
        layout.addLayout(views, 1)
        self.background_info_label = QLabel("Choose three background samples.")
        self.background_info_label.setWordWrap(True)
        layout.addWidget(self.background_info_label)
        self.background_review_label = QLabel("Samples are specific to each frame/channel. Choose Skip correction to run without subtraction.")
        self.background_review_label.setWordWrap(True)
        layout.addWidget(self.background_review_label)
        return panel

    def background_signal_thresholds(self):
        return {}

    def seed_background_signal_thresholds(self):
        pass

    def background_cutoffs(self):
        return {}

    def current_background_key(self):
        position = self.current_preview_position()
        channel = self.background_channel_combo.currentData()
        if position is None or channel is None:
            return None
        return int(self.time_slider.value()), int(position), int(channel)

    def on_background_channel_changed(self, *_args):
        self.sync_sample_editor()
        self.request_background_preview()

    def sync_sample_editor(self, *_args):
        if not hasattr(self, "time_slider") or not hasattr(self, "sample_spins"):
            return
        key = self.current_background_key()
        samples = self.background_samples.get(key, [None] * 3)
        index = self.background_sample_combo.currentIndex()
        box = samples[index] if index >= 0 else None
        self.delete_sample_button.setEnabled(box is not None)
        if box is not None:
            for spin, value in zip(self.sample_spins.values(), box):
                spin.blockSignals(True)
                spin.setValue(int(value))
                spin.blockSignals(False)

    def place_background_sample(self, x, y):
        if self._background_preview_data is None:
            return
        key, raw, _cells, _distance = self._background_preview_data
        if key != self.current_background_key():
            return
        width = min(self.sample_spins["Width"].value(), raw.shape[1])
        height = min(self.sample_spins["Height"].value(), raw.shape[0])
        x = int(np.clip(round(x - width / 2), 0, raw.shape[1] - width))
        y = int(np.clip(round(y - height / 2), 0, raw.shape[0] - height))
        index = self.background_sample_combo.currentIndex()
        self.background_samples.setdefault(key, [None] * 3)[index] = (x, y, width, height)
        self.sync_sample_editor()
        self.invalidate_background()
        if index < 2:
            self.background_sample_combo.setCurrentIndex(index + 1)

    def edit_background_sample(self, *_args):
        if not hasattr(self, "time_slider"):
            return
        key = self.current_background_key()
        samples = self.background_samples.get(key)
        index = self.background_sample_combo.currentIndex()
        if samples is not None and samples[index] is not None:
            samples[index] = tuple(spin.value() for spin in self.sample_spins.values())
            self.invalidate_background()

    def remove_background_sample(self):
        key = self.current_background_key()
        if key in self.background_samples:
            self.background_samples[key][self.background_sample_combo.currentIndex()] = None
            self.sync_sample_editor()
            self.invalidate_background()

    def copy_background_samples(self):
        key = self.current_background_key()
        if key is None or len(self.active_channels) != 2:
            return
        for channel in self.active_channels:
            if channel != key[2]:
                self.background_samples[(key[0], key[1], int(channel))] = list(self.background_samples.get(key, [None] * 3))
        self.invalidate_background()

    def invalidate_background(self, *_args):
        if self._updating_controls or self._background_closed:
            return
        self.seed_background_signal_thresholds()
        self._background_cancel.set()
        self._background_generation += 1
        self._background_records = {}
        self._background_errors = []
        self.processing_black_point_raw = None
        self.processing_white_point_raw = None
        self.run_button.setEnabled(False)
        self.render_background_preview()
        self.background_review_label.setText("Settings changed. Preparing the updated intensity window...")
        self.background_update_timer.start()

    def on_setup_tab_changed(self, index):
        self.back_button.setVisible(index == 1)
        self.skip_correction_button.setVisible(index == 1)
        if index == 1 and not self.background_enabled:
            self.background_enabled = True
            self.invalidate_background()
        self.run_button.setText(
            "Confirm and Run Analysis" if index == 1 else "Confirm Threshold and Continue")
        if index == 1:
            self.seed_background_signal_thresholds()
            self.play_timer.stop()
            self.request_background_preview()

    def on_background_position_changed(self, index):
        if self._updating_controls or index < 0:
            return
        position = self.background_position_combo.itemData(index)
        self.preview_position_combo.setCurrentIndex(self.preview_position_combo.findData(position))
        self.request_background_preview()

    @staticmethod
    def scan_background_scope(image_data, label_provider, distance_provider, estimator,
                              frame_keys, margin, cutoffs, cancelled,
                              signal_thresholds=(), include_cells=True):
        records = {}
        low, high = np.inf, -np.inf
        previous = None
        distance = None
        for key in frame_keys:
            if cancelled.is_set():
                return None
            time, position, channel = key
            if previous != (time, position):
                frames = {c: np.asarray(image_data.get(time, position, c), dtype=np.float32)
                          for c in dict(signal_thresholds)}
                distance = distance_provider(label_provider(time, position), frames,
                                             signal_thresholds, include_cells)
                previous = (time, position)
            raw = np.asarray(image_data.get(*key), dtype=np.float32)
            record = estimator(raw, distance, margin, cutoffs.get(channel))
            records[key] = record
            corrected = raw - np.float32(record["median"])
            finite = corrected[np.isfinite(corrected)]
            low = min(low, float(finite.min()))
            high = max(high, float(finite.max()))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("The corrected fluorescence scope has no usable intensity range.")
        return records, float(low), float(high)

    @staticmethod
    def scan_manual_background_scope(image_data, estimator, frame_keys, samples, cancelled):
        records, errors = {}, []
        low, high = np.inf, -np.inf
        for key in frame_keys:
            if cancelled.is_set():
                return None
            try:
                raw = np.asarray(image_data.get(*key), dtype=np.float32)
                record = estimator(raw, samples[key])
                records[key] = record
                finite = raw[np.isfinite(raw)]
                low, high = min(low, float(finite.min())), max(high, float(finite.max()))
            except Exception as error:
                errors.append(f"T={key[0]}, P={key[1]}, C={key[2]}: {error}")
        if not errors and (not np.isfinite(low) or not np.isfinite(high) or high <= low):
            errors.append("The raw fluorescence scope has no usable intensity range.")
        return records, float(low), float(high), errors

    def start_background_scan(self, positions, timepoints):
        keys = tuple((int(t), int(p), int(c)) for p in positions for t in timepoints for c in self.active_channels)
        self._background_cancel.set()
        self._background_cancel = Event()
        self._background_generation += 1
        self.run_button.setEnabled(False)
        missing = [(key, sum(r is not None for r in self.background_samples.get(key, [])))
                   for key in keys if sum(r is not None for r in self.background_samples.get(key, [])) != 3]
        if missing:
            key, count = missing[0]
            text = (f"{len(keys) - len(missing)}/{len(keys)} frame/channels sampled. "
                    f"Next: P={key[1]}, T={key[0]}, C={key[2]} has {count}/3 samples.")
            self.background_review_label.setText(text)
            self.status_label.setText(text)
            self.request_background_preview()
            return
        if self.sample_estimator is None:
            self.background_review_label.setText("Manual background measurement is unavailable.")
            return
        self.sampled_frame_keys = keys
        self.representative_timepoints = tuple(timepoints[index] for index in self.evenly_spaced_values(
            0, len(timepoints) - 1, self.DEFAULT_PREVIEW_COUNT))
        samples = {key: tuple(self.background_samples[key]) for key in keys}
        future = self._background_executor.submit(self.scan_manual_background_scope, self.image_data,
                                                  self.sample_estimator, keys, samples, self._background_cancel)
        self._background_tasks.append((future, self._background_generation, "scan", None))
        self.background_poll_timer.start()
        self.status_label.setText("Measuring the selected background samples...")
        self.request_background_preview()

    @staticmethod
    def load_background_preview(image_data, label_provider, distance_provider, key, signal_thresholds=(), include_cells=True):
        raw = np.asarray(image_data.get(*key), dtype=np.float32)
        cells = np.zeros(raw.shape, dtype=bool)
        if label_provider is not None:
            try:
                labels = np.asarray(label_provider(key[0], key[1]))
                if labels.shape == raw.shape:
                    cells = labels > 0
            except (ValueError, KeyError, IndexError):
                pass
        return key, raw, cells, None

    def request_background_preview(self, *_args):
        if not hasattr(self, "setup_tabs") or self.setup_tabs.currentIndex() != 1 or self._background_closed:
            return
        position = self.current_preview_position()
        if position is None:
            return
        key = (int(self.time_slider.value()), int(position), int(self.background_channel_combo.currentData()))
        if self._background_preview_data is not None and self._background_preview_data[0] == key:
            self.render_background_preview()
            return
        cached = self._background_preview_cache.get(key)
        if cached is not None:
            self._background_preview_data = cached
            self.render_background_preview()
            return
        self._background_preview_serial += 1
        serial = self._background_preview_serial
        # Queued obsolete preview jobs can be dropped; running jobs are ignored on completion.
        for future, _generation, kind, _serial in self._background_tasks:
            if kind == "preview":
                future.cancel()
        future = self._background_executor.submit(
            self.load_background_preview, self.image_data, self.cell_label_provider,
            self.distance_provider, key, self.background_signal_thresholds(),
            True)
        self._background_tasks.append((future, self._background_generation, "preview", serial))
        self.background_poll_timer.start()
        self.background_info_label.setText("Loading fluorescence image for background sampling...")

    def poll_background_tasks(self):
        pending = []
        tasks, self._background_tasks = self._background_tasks, []
        for future, generation, kind, serial in tasks:
            if not future.done():
                pending.append((future, generation, kind, serial))
                continue
            if future.cancelled() or generation != self._background_generation:
                continue
            if kind == "preview" and serial != self._background_preview_serial:
                continue
            try:
                result = future.result()
                if kind == "scan":
                    if result is None:
                        continue
                    self._background_records, low, high, self._background_errors = result
                    if self._background_errors:
                        self.processing_black_point_raw = self.processing_white_point_raw = None
                        self.background_review_label.setText("Background samples need attention: " + " | ".join(self._background_errors))
                        self.status_label.setText(self.background_review_label.text())
                        self.run_button.setEnabled(False)
                        self.render_background_preview()
                        continue
                    self.processing_black_point_raw, self.processing_white_point_raw = low, high
                    self.update_window_labels()
                    self.update_thumbnails()
                    self.update_preview()
                    self.run_button.setEnabled(True)
                    self.on_setup_tab_changed(self.setup_tabs.currentIndex())
                    self.background_review_label.setText(
                        "Correction is ready. Confirm and run analysis, or use Back to Threshold to inspect the corrected preview.")
                    self.status_label.setText("Background correction ready. Confirm and run analysis when ready.")
                    self.request_background_preview()
                else:
                    self._background_preview_data = result
                    self._background_preview_cache[result[0]] = result
                    self._background_preview_cache.move_to_end(result[0])
                    while len(self._background_preview_cache) > 2:
                        self._background_preview_cache.popitem(last=False)
                    self.render_background_preview()
            except Exception as error:
                self.status_label.setText(f"Background preview failed: {error}")
                self.background_info_label.setText(str(error))
                if kind == "scan":
                    self.processing_black_point_raw = None
                    self.processing_white_point_raw = None
                    self.run_button.setEnabled(False)
                    self.background_review_label.setText("Correction unavailable. Adjust settings or choose Skip correction.")
        self._background_tasks = pending + self._background_tasks
        if not self._background_tasks:
            self.background_poll_timer.stop()

    def render_background_preview(self, *_args):
        if self._background_preview_data is None:
            return
        key, raw, cells, _distance = self._background_preview_data
        finite = raw[np.isfinite(raw)]
        low, high = np.percentile(finite, [1, 99.8]) if finite.size else (0., 1.)
        grey = np.clip(np.nan_to_num((raw - low) / max(float(high - low), 1.), nan=0., posinf=1., neginf=0.), 0., 1.)
        rgb = np.repeat(grey[..., None], 3, axis=2) * 255.
        samples = self.background_samples.get(key, [None] * 3)
        corrected_rgb = None
        if all(box is not None for box in samples) and self.sample_estimator is not None:
            try:
                estimate = self.sample_estimator(raw, samples)
                corrected = raw - np.float32(estimate["mean"])
                corrected_grey = np.clip(np.nan_to_num((corrected - low) / max(float(high - low), 1.), nan=0., posinf=1., neginf=0.), 0., 1.)
                corrected_rgb = np.repeat(corrected_grey[..., None], 3, axis=2) * 255.
                self.background_corrected_label.setText(f"Background subtracted: {estimate['mean']:.3f} (same display scale)")
            except ValueError:
                pass
        if corrected_rgb is None:
            self.background_corrected_label.setText("Select three valid samples to preview subtraction")
            self.background_corrected_view.set_image(QPixmap())
        messages = []
        for index, box in enumerate(samples, 1):
            if box is None:
                messages.append(f"Sample {index}: not selected")
                continue
            x, y, width, height = box
            if min(x, y) < 0 or min(width, height) <= 0 or x + width > raw.shape[1] or y + height > raw.shape[0]:
                messages.append(f"Sample {index}: outside image; edit its coordinates")
                continue
            patch = raw[y:y+height, x:x+width]
            values = patch[np.isfinite(patch)]
            rgb[y:y+height, x:x+width] = .70 * rgb[y:y+height, x:x+width] + .30 * np.asarray([0., 160., 255.])
            if corrected_rgb is not None:
                corrected_rgb[y:y+height, x:x+width] = .70 * corrected_rgb[y:y+height, x:x+width] + .30 * np.asarray([0., 160., 255.])
            if values.size:
                messages.append(f"Sample {index}: mean {np.mean(values, dtype=np.float64):.3f}, "
                                f"SD {np.std(values, dtype=np.float64):.3f}, n={values.size:,}")
            else:
                messages.append(f"Sample {index}: no finite pixels")
        if self.background_show_cells.isChecked():
            from scipy.ndimage import binary_erosion
            rgb[cells & ~binary_erosion(cells)] = 255.
        self.background_image_view.set_image(self.rgb_to_pixmap(rgb.astype(np.uint8)))
        if corrected_rgb is not None:
            self.background_corrected_view.set_image(self.rgb_to_pixmap(corrected_rgb.astype(np.uint8)))
        if all(r is not None for r in samples) and self.sample_estimator is not None:
            try:
                record = self.sample_estimator(raw, samples)
                messages.append(f"Background mean to subtract: {record['mean']:.3f}. "
                                f"Sample-mean range: {record['sample_mean_range']:.3f} "
                                f"({100 * record['sample_mean_relative_range']:.1f}% of pooled mean).")
                if "sample_means_disagree" in record["status"]:
                    messages.append("Samples disagree by more than 20%. Inspect for missed cells or a background gradient.")
                if cells.any():
                    overlap = sum(int(cells[y:y+h, x:x+w].sum()) for x,y,w,h in samples)
                    if overlap:
                        messages.append(f"Warning: samples contain {overlap:,} segmented-cell pixels. Inspect and reposition if needed.")
            except ValueError as error:
                messages.append(str(error))
        self.background_info_label.setText("\n".join(messages))
        self.sync_sample_editor()

    def stop_background_jobs(self):
        if self._background_closed:
            return
        self._background_closed = True
        self._background_cancel.set()
        self.background_update_timer.stop()
        self.background_poll_timer.stop()
        self.preview_update_timer.stop()
        self.thumbnail_update_timer.stop()
        self.scope_update_timer.stop()
        for future, _generation, _kind, _serial in self._background_tasks:
            future.cancel()
        self._background_executor.shutdown(wait=False, cancel_futures=True)

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
            f"When enabled, T=0 is not used to estimate the shared {self.analysis_label} "
            "intensity window, is not shown in the preview controls, and is "
            f"not sent to the full {self.analysis_label} processing queue."
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

        live_channel_label = QLabel(f"Channel {self.live_channel}")
        scope_form.addRow(f"{self.primary_label} preview:", live_channel_label)
        if not self.is_cell_viability:
            dead_channel_label = QLabel(f"Channel {self.dead_channel}")
            scope_form.addRow("Dead preview:", dead_channel_label)

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
        self.view_mode_combo.addItems(["Overlay", "Raw Fluorescence", "Mask"])
        self.view_mode_combo.currentTextChanged.connect(
            self.schedule_preview_update
        )
        toolbar.addWidget(self.view_mode_combo)

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

        paired_view = QWidget()
        paired_layout = QHBoxLayout(paired_view)
        paired_layout.setContentsMargins(0, 0, 0, 0)
        paired_layout.setSpacing(6)

        live_panel = QWidget()
        live_layout = QVBoxLayout(live_panel)
        live_layout.setContentsMargins(0, 0, 0, 0)
        live_label = QLabel(self.primary_label)
        live_label.setStyleSheet("font-weight: bold; color: #00aa00;")
        live_label.setAlignment(Qt.AlignCenter)
        live_layout.addWidget(live_label)
        self.live_image_view = ZoomableImageView()
        self.live_image_view.setMinimumSize(250, 320)
        live_layout.addWidget(self.live_image_view, 1)

        dead_panel = QWidget()
        dead_layout = QVBoxLayout(dead_panel)
        dead_layout.setContentsMargins(0, 0, 0, 0)
        dead_label = QLabel("Dead")
        dead_label.setStyleSheet("font-weight: bold; color: #cc0000;")
        dead_label.setAlignment(Qt.AlignCenter)
        dead_layout.addWidget(dead_label)
        self.dead_image_view = ZoomableImageView()
        self.dead_image_view.setMinimumSize(250, 320)
        dead_layout.addWidget(self.dead_image_view, 1)

        paired_layout.addWidget(live_panel, 1)
        paired_layout.addWidget(dead_panel, 1)
        dead_panel.setVisible(not self.is_cell_viability)
        layout.addWidget(paired_view, 1)
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
        options_row.addWidget(QLabel("Gaussian sigma (0 = off):"))

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

        self.live_foreground_label = QLabel(f"{self.primary_label} foreground: —")
        self.live_foreground_label.setWordWrap(True)
        self.live_foreground_label.setStyleSheet(
            "padding: 4px; border: 1px solid #555;"
        )
        readout_row.addWidget(self.live_foreground_label, 2)

        self.dead_foreground_label = QLabel("Dead foreground: —")
        self.dead_foreground_label.setWordWrap(True)
        self.dead_foreground_label.setStyleSheet(
            "padding: 4px; border: 1px solid #555;"
        )
        readout_row.addWidget(self.dead_foreground_label, 2)
        self.dead_foreground_label.setVisible(not self.is_cell_viability)

        self.raw_threshold_label = QLabel("Raw-equivalent threshold: —")
        self.raw_threshold_label.setWordWrap(True)
        self.raw_threshold_label.setStyleSheet(
            "padding: 4px; border: 1px solid #555;"
        )
        readout_row.addWidget(self.raw_threshold_label, 2)

        self.window_label = QLabel("Shared intensity window: —")
        self.window_label.setWordWrap(True)
        self.window_label.setStyleSheet(
            "color: #888; padding: 4px; border: 1px solid #555;"
        )
        readout_row.addWidget(self.window_label, 3)
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
            button.setIconSize(QPixmap(132, 52).size())
            button.setMinimumSize(168, 60)
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
        self._background_cancel.set()
        self._background_generation += 1
        self._background_records = {}
        self.run_button.setEnabled(False)
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

        timepoints = [time for time in range(start, end + 1) if time not in self.excluded_timepoints]
        if self.drop_frame_zero_checkbox.isChecked():
            timepoints = [time for time in timepoints if time != 0]

        return timepoints

    def refresh_scope(self) -> None:
        if self._background_closed:
            return
        self._background_cancel.set()
        self._background_generation += 1
        self._background_records = {}
        positions = self.selected_positions()
        if not positions:
            self.processing_black_point_raw = None
            self.processing_white_point_raw = None
            self.sampled_frame_keys = ()
            self.representative_timepoints = ()
            self.preview_position_combo.clear()
            self.live_image_view.set_image(QPixmap())
            self.dead_image_view.set_image(QPixmap())
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
            self.live_image_view.set_image(QPixmap())
            self.dead_image_view.set_image(QPixmap())
            self.clear_thumbnails()
            self.run_button.setEnabled(False)
            self.status_label.setText(
                "No frames remain in the selected range after exclusions. "
                "Adjust the time range or frame-zero setting."
            )
            return

        preview_start = int(considered_timepoints[0])
        preview_end = int(considered_timepoints[-1])

        self._background_cancel.set()
        self._background_generation += 1
        self._background_records = {}
        self._background_preview_data = None
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
            self.background_position_combo.clear()
            for position in positions:
                self.background_position_combo.addItem(f"Position {position}", position)
            self.background_position_combo.setCurrentIndex(max(0, desired_index))

            current_time = (
                self.time_slider.value()
                if self.time_slider.maximum() >= self.time_slider.minimum()
                else preview_start
            )
            self.time_slider.setRange(preview_start, preview_end)
            self.time_slider.setValue(min(considered_timepoints, key=lambda t: abs(t - current_time)))
        finally:
            self._updating_controls = False

        excluded_note = (
            " Frame 0 is excluded."
            if self.drop_frame_zero_checkbox.isChecked() and start == 0
            else ""
        )
        self.status_label.setText(
            f"Estimating one shared {self.analysis_label} intensity window from considered "
            f"frames...{excluded_note}"
        )
        if self.background_enabled:
            try:
                self.start_background_scan(positions, considered_timepoints)
            except Exception as error:
                self.run_button.setEnabled(False)
                self.status_label.setText(f"Could not prepare background correction: {error}")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.estimate_processing_window(
                positions,
                considered_timepoints,
            )
            self.representative_timepoints = tuple(
                considered_timepoints[index] for index in self.evenly_spaced_values(
                    0, len(considered_timepoints) - 1, self.DEFAULT_PREVIEW_COUNT))
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
                f"{self.analysis_label} preview setup failed",
                f"Could not prepare the shared {self.analysis_label} preview window:\n{error}",
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
        self.background_position_combo.blockSignals(True)
        self.background_position_combo.setCurrentIndex(self.background_position_combo.findData(self.current_preview_position()))
        self.background_position_combo.blockSignals(False)
        self.request_background_preview()

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
            (int(time), int(position), int(channel))
            for position in positions
            for time in timepoints
            for channel in self.active_channels
        ]
        if not frame_keys:
            raise ValueError("No frames are available in the selected scope.")

        global_min = np.inf
        global_max = -np.inf
        channel_ranges = {}

        for frame_number, key in enumerate(frame_keys, start=1):
            frame = self.get_raw_frame(*key)
            finite_values = frame[np.isfinite(frame)]
            if finite_values.size == 0:
                continue

            channel = key[2]
            previous = channel_ranges.get(channel, (np.inf, -np.inf))
            channel_ranges[channel] = (min(previous[0], float(finite_values.min())),
                                       max(previous[1], float(finite_values.max())))
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
                f"The selected {self.analysis_label} fluorescence frames have no usable intensity range: "
                f"min={global_min}, max={global_max}."
            )

        self._background_threshold_ranges = channel_ranges
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
                f"Expected a 2D frame for T={time}, P={position}, "
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
            raise RuntimeError(f"The shared {self.analysis_label} intensity window is unavailable.")

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
        channel: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_frame = self.get_raw_frame(time, position, channel)
        if self.background_enabled:
            record = self._background_records.get((int(time), int(position), int(channel)))
            if record is None:
                raise RuntimeError("The corrected background scope is not ready.")
            raw_frame = raw_frame - np.float32(record.get("value", record["median"]))
        normalized = self.normalize_to_shared_uint8(raw_frame)

        smoothed = np.empty_like(normalized, dtype=np.uint8)
        gaussian_filter(
            normalized,
            sigma=float(self.smoothing_sigma),
            output=smoothed,
        )
        mask = (smoothed >= self.threshold_spin.value()) & np.isfinite(raw_frame)
        return normalized, smoothed, mask

    def make_display_rgb(
        self,
        normalized: np.ndarray,
        mask: np.ndarray,
        overlay_color: tuple[int, int, int],
        view_mode: str | None = None,
    ) -> np.ndarray:
        mode = self.view_mode_combo.currentText() if view_mode is None else view_mode

        if mode == "Raw Fluorescence":
            return np.repeat(normalized[..., None], 3, axis=2)
        if mode == "Mask":
            mask_uint8 = mask.astype(np.uint8) * 255
            return np.repeat(mask_uint8[..., None], 3, axis=2)

        rgb = np.repeat(normalized[..., None], 3, axis=2).astype(np.float32)
        alpha = self.opacity_slider.value() / 100.0
        overlay_color = np.asarray(overlay_color, dtype=np.float32)
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
            live_normalized, _live_smoothed, live_mask = self.create_processed_preview(
                time, position, self.live_channel
            )
            live_rgb = self.make_display_rgb(live_normalized, live_mask, self.LIVE_OVERLAY_RGB)
            self.live_image_view.set_image(self.rgb_to_pixmap(live_rgb))
            readouts = [(self.live_foreground_label, self.primary_label, live_mask)]
            if not self.is_cell_viability:
                dead_normalized, _dead_smoothed, dead_mask = self.create_processed_preview(
                    time, position, self.dead_channel)
                dead_rgb = self.make_display_rgb(dead_normalized, dead_mask, self.DEAD_OVERLAY_RGB)
                self.dead_image_view.set_image(self.rgb_to_pixmap(dead_rgb))
                readouts.append((self.dead_foreground_label, "Dead", dead_mask))

            for label, name, mask in readouts:
                foreground_pixels = int(np.count_nonzero(mask))
                total_pixels = int(mask.size)
                foreground_percent = (
                    100.0 * foreground_pixels / total_pixels
                    if total_pixels
                    else 0.0
                )
                label.setText(
                    f"{name} foreground: {foreground_percent:.2f}% "
                    f"({foreground_pixels:,}/{total_pixels:,} pixels)"
                )

            self.frame_info_label.setText(
                f"Position {position} • Timepoint {time} • "
                + (f"Cell Viability C{self.live_channel} • " if self.is_cell_viability
                   else f"Live C{self.live_channel} / Dead C{self.dead_channel} • ")
                +
                f"Shared threshold {self.threshold_spin.value()}"
            )
            self.time_value_label.setText(f"T={time}")
            self.update_raw_threshold_label()
            if self.background_enabled:
                self.frame_info_label.setText(self.frame_info_label.text() + " • Background corrected")
            self.request_background_preview()
        except Exception as error:
            self.status_label.setText(f"Preview failed: {error}")

    def fit_preview(self) -> None:
        for image_view in ([self.live_image_view] if self.is_cell_viability
                           else [self.live_image_view, self.dead_image_view]):
            image_view._auto_fit_enabled = True
            image_view.request_fit()

    def update_thumbnails(self) -> None:
        self.clear_thumbnails()
        position = self.current_preview_position()
        if position is None or self.processing_black_point_raw is None:
            return

        for index, time in enumerate(self.representative_timepoints):
            if index >= len(self.thumbnail_buttons):
                break
            try:
                live_normalized, _live_smoothed, live_mask = self.create_processed_preview(
                    time, position, self.live_channel
                )
                live_rgb = self.make_display_rgb(
                    live_normalized, live_mask, self.LIVE_OVERLAY_RGB, "Overlay")
                paired_rgb = live_rgb
                if not self.is_cell_viability:
                    dead_normalized, _dead_smoothed, dead_mask = self.create_processed_preview(
                        time, position, self.dead_channel)
                    dead_rgb = self.make_display_rgb(
                        dead_normalized, dead_mask, self.DEAD_OVERLAY_RGB, "Overlay")
                    spacer = np.full((live_rgb.shape[0], 4, 3), 255, dtype=np.uint8)
                    paired_rgb = np.concatenate((live_rgb, spacer, dead_rgb), axis=1)
                pixmap = self.rgb_to_pixmap(paired_rgb).scaled(
                    132,
                    52,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                button = self.thumbnail_buttons[index]
                button.setVisible(True)
                button.setIcon(QIcon(pixmap))
                button.setText(f"T={time}\n" + ("Cell Viability" if self.is_cell_viability else "L | D"))
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
            ("Fixed raw intensity mapping for corrected signal: " if self.background_enabled
             else "Shared raw intensity window: ")
            + f"{black:.3f} to {white:.3f}\n"
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
            ("Corrected-intensity threshold: " if self.background_enabled
             else "Raw-equivalent threshold: ") + f"{raw_equivalent:.3f}"
        )

    def reset_threshold(self) -> None:
        self.set_threshold(self.initial_threshold)
        self.opacity_slider.setValue(self.DEFAULT_OVERLAY_ALPHA_PERCENT)
        self.view_mode_combo.setCurrentText("Overlay")
        self.sigma_spin.setValue(1.5)

    def on_time_slider_changed(self, value: int) -> None:
        if not self._updating_controls:
            allowed = self.considered_timepoints()
            if allowed and value not in allowed:
                candidates = ([t for t in allowed if t > value] if value >= self._last_preview_time
                              else [t for t in allowed if t < value])
                target = (min(candidates) if value >= self._last_preview_time else max(candidates)) if candidates else min(allowed, key=lambda t: abs(t - value))
                self.time_slider.setValue(target)
                return
        self._last_preview_time = value
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

    def skip_correction(self) -> None:
        """Run with the original intensity window, even if correction failed."""
        self.background_enabled = False
        self.background_update_timer.stop()
        self.refresh_scope()
        if self.run_button.isEnabled():
            self.accept_setup()

    def accept_setup(self) -> None:
        positions = self.selected_positions()
        if not positions:
            QMessageBox.warning(
                self,
                "No positions selected",
                f"Select at least one position before running {self.analysis_label} analysis.",
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
                f"{self.analysis_label} preview not ready",
                f"The shared {self.analysis_label} intensity window could not be prepared.",
            )
            return

        if self.setup_tabs.currentIndex() == 0:
            self.seed_background_signal_thresholds()
            self.setup_tabs.setCurrentIndex(1)
            return
        preview_position = self.current_preview_position()
        if preview_position is None:
            preview_position = positions[0]

        self.result = LiveDeadResult(
            positions=tuple(positions),
            time_start=int(self.time_start_spin.value()),
            time_end=int(self.time_end_spin.value()),
            drop_frame_zero=bool(
                self.drop_frame_zero_checkbox.isChecked()
            ),
            live_channel=int(self.live_channel),
            dead_channel=self.dead_channel,
            analysis_mode=self.analysis_mode,
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
            background_enabled=self.background_enabled,
            background_margin_pixels=0,
            background_upper_cutoffs=tuple(sorted(self.background_cutoffs().items())),
            background_include_cells=False,
            background_signal_thresholds=(),
            background_method="manual_samples_mean",
            background_samples=tuple((t, p, c, *rectangle)
                                     for (t, p, c) in sorted(self.sampled_frame_keys)
                                     for rectangle in self.background_samples.get((t, p, c), ()) if rectangle is not None),
            processing_window_source=("partaker_setup_dialog_manual_samples_raw_scope_min_max"
                                      if self.background_enabled
                                      else "partaker_setup_dialog_full_selected_scope_min_max"),
            capture_interval_value=float(self.capture_interval_spin.value()),
            capture_interval_unit=str(
                self.capture_interval_unit_combo.currentText()
            ),
        )
        self.stop_background_jobs()
        self.play_timer.stop()
        self.setup_accepted.emit(self.result)
        self.accept()

    def reject(self) -> None:  # noqa: A003 - Qt API name
        self.play_timer.stop()
        self.stop_background_jobs()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.play_timer.stop()
        self.stop_background_jobs()
        super().closeEvent(event)
