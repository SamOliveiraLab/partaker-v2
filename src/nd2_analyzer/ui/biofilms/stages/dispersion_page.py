"""Show dispersion outputs alongside the corresponding phase-contrast frames."""

import json
from copy import deepcopy
from pathlib import Path

import cv2

import numpy as np
from scipy import ndimage as ndi
from PySide6.QtCore import Qt, Signal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import patheffects
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QFrame, QHBoxLayout, QLabel, QMenu, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from nd2_analyzer.analysis.dispersion.microcolony_dynamics import BORDER, image_motion, render_overlay, whole_colony_mask
from .widgets import PlotPanel


class DispersionStrip(FigureCanvas):
    """Independent scrubber using only Dispersion's frame timestamps."""

    hour_changed = Signal(float)

    def __init__(self, parent=None):
        self.times = []
        self.unit = "Frame"
        self.figure = Figure(figsize=(10, 0.9), facecolor="white")
        super().__init__(self.figure)
        self.setParent(parent)
        self.setFixedHeight(120)
        self.ax = self.figure.add_axes([0.05, 0.42, 0.92, 0.34])
        self.cursor_h = 0.0
        self._drag = False
        self.mpl_connect("button_press_event", self._press)
        self.mpl_connect("motion_notify_event", self._move)
        self.mpl_connect("button_release_event", self._release)
        self.redraw()

    def set_hour(self, value, emit=False):
        snapped = self._snap(value)
        self.cursor_h = float(value) if snapped is None else snapped
        self.redraw()
        if emit and snapped is not None:
            self.hour_changed.emit(snapped)

    def _press(self, event):
        if event.button == 1 and event.inaxes is self.ax and event.xdata is not None:
            self._drag = True
            self.set_hour(event.xdata, emit=True)

    def _release(self, _event):
        self._drag = False

    def redraw(self):
        self.ax.clear()
        if not self.times:
            self.ax.axis("off")
            self.ax.text(0.5, 0.5, "Load a dispersion recording", ha="center",
                         va="center", transform=self.ax.transAxes, color="#666666")
        else:
            first, last = self.times[0], self.times[-1]
            padding = max((last - first) * 0.03, 0.05)
            self.ax.axvspan(first, last, color="#e5e5e5", ymin=0.2, ymax=0.8)
            self.ax.axvline(self.cursor_h, color="#0077b6", linewidth=2)
            self.ax.text(self.cursor_h, 1.02, f"{self.cursor_h:g}" + (" h" if self.unit == "Time (h)" else ""),
                         color="#0077b6", transform=self.ax.get_xaxis_transform(), weight="bold", fontsize=10)
            self.ax.set(xlim=(first - padding, last + padding), ylim=(0, 1), yticks=[])
            self.ax.tick_params(axis="x", labelsize=9)
            self.ax.set_xlabel(self.unit, fontsize=9, labelpad=4)
            for side in ("left", "right", "top"):
                self.ax.spines[side].set_visible(False)
        self.draw_idle()

    def _snap(self, value):
        return min(self.times, key=lambda time: abs(time - value)) if self.times else None

    def _move(self, event):
        if self._drag and event.inaxes is self.ax and event.xdata is not None:
            value = self._snap(event.xdata)
            if value is not None and value != self.cursor_h:
                self.set_hour(value, emit=True)



def fit_analysis_view(axes, image):
    height, width = image.shape[:2]
    if min(height, width) > 2 * BORDER:
        axes.set_xlim(BORDER - .5, width - BORDER - .5)
        axes.set_ylim(height - BORDER - .5, BORDER - .5)


def draw_tracking_overlay(output, frame, selected, *, territory=True, motion=True, point=None, marker=True):
    """Draw the current territory and saved, image-supported frame displacement."""
    if selected is None:
        return output
    mask = frame["region_ids"] == selected
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return output
    if territory:
        edge = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        output[edge] = (0, 230, 255)
    if motion:
        for vector in json.loads(str(frame.get("vectors_json", np.array("[]")))):
            if vector.get("display_id") != selected:
                continue
            start, end = np.asarray(vector["start"], float), np.asarray(vector["end"], float)
            if start.shape != (2,) or end.shape != (2,) or not np.isfinite([start, end]).all():
                continue
            a, b = tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int))
            if a != b:
                cv2.arrowedLine(output, a, b, (255, 255, 255), 1, cv2.LINE_AA, tipLength=.25)
    if territory and marker:
        center = tuple(np.rint(point if point is not None else [xx.mean(), yy.mean()]).astype(int))
        cv2.drawMarker(output, center, (255, 255, 255), cv2.MARKER_DIAMOND, 7, 2, cv2.LINE_AA)
        cv2.drawMarker(output, center, (0, 230, 255), cv2.MARKER_DIAMOND, 5, 1, cv2.LINE_AA)
    return output


def draw_breakup_overlay(output, frame):
    """Color the connected dark body and remaining dark components per territory."""
    ids = frame["region_ids"]
    pink = frame.get("pink_mask", np.zeros(ids.shape, bool)).astype(bool)
    measured = frame.get("measured_mask", frame["inner_mask"]).astype(bool)
    for ident in np.unique(ids[measured & ~pink]):
        if ident <= 0:
            continue
        dark = measured & (ids == ident)
        labels, _ = ndi.label(dark, np.ones((3, 3)))
        counts = np.bincount(labels[dark])
        if not counts.size:
            continue
        counts[0] = 0
        connected = labels == counts.argmax()
        for mask, color in ((dark & connected, (255, 210, 40)), (dark & ~connected, (30, 140, 255))):
            output[mask] = (.3 * output[mask] + .7 * np.array(color)).astype(np.uint8)
    central_dark = pink & measured
    output[central_dark] = (.3 * output[central_dark] + .7 * np.array([30, 140, 255])).astype(np.uint8)
    return output


def advance_colony_trail(source, target, selected, path):
    """Advect a reference point through a checked local image-motion fit."""
    yy, xx = np.nonzero(target["region_ids"] == selected)
    if not len(xx):
        return []
    restart = [[float(xx.mean()), float(yy.mean())]]
    if source is None or not path:
        return restart
    territory = source["region_ids"] == selected
    yy, xx = np.nonzero(territory)
    if not len(xx):
        return restart
    height, width = territory.shape
    x0, y0 = max(BORDER, int(xx.min()) - 24), max(BORDER, int(yy.min()) - 24)
    section = np.s_[y0:min(height - BORDER, int(yy.max()) + 25),
                    x0:min(width - BORDER, int(xx.max()) + 25)]
    _, affine = image_motion(source["raw_gray"][section], target["raw_gray"][section], territory[section])
    if affine is None:
        return restart
    offset = np.array([x0, y0])
    point = affine[:, :2] @ (np.asarray(path[-1]) - offset) + affine[:, 2] + offset
    if not np.isfinite(point).all():
        return restart
    return (path + [point.tolist()])[-6:]


class DispersionSetupDialog(QDialog):
    """Preview draft boundaries without changing the displayed analysis."""

    def __init__(self, parent, read_frame, positions, count, config, hours=None):
        super().__init__(parent)
        self.setWindowTitle("Set up dispersion analysis")
        self.resize(720, 570)
        self.read_frame = read_frame
        self.draft = deepcopy(config)
        self.hours = hours
        self.preview_key = None
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.threshold = QSlider(Qt.Horizontal)
        self.threshold.setRange(0, 100)
        self.threshold.setValue(config["threshold"])
        self.threshold_value = QLabel()
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(self.threshold, 1)
        threshold_row.addWidget(self.threshold_value)
        form.addRow("Boundary threshold", threshold_row)
        self.first_frame, self.last_frame = QSpinBox(), QSpinBox()
        for field, value in ((self.first_frame, config["first"]), (self.last_frame, config["last"])):
            field.setRange(0, count - 1)
            field.setValue(value)
        scope = QHBoxLayout()
        scope.addWidget(self.first_frame)
        scope.addWidget(QLabel("to"))
        scope.addWidget(self.last_frame)
        form.addRow("Frame range", scope)
        self.position = QComboBox()
        for position in positions:
            self.position.addItem(str(position), position)
        self.capture = QSpinBox()
        self.capture.setRange(0, count - 1)
        self.capture.setValue(config["first"])
        preview_row = QHBoxLayout()
        preview_row.addWidget(self.position)
        preview_row.addWidget(QLabel("Frame"))
        preview_row.addWidget(self.capture)
        self.update_button = QPushButton("Update preview")
        preview_row.addWidget(self.update_button)
        form.addRow("Preview position", preview_row)
        layout.addLayout(form)
        self.plot = PlotPanel(self, size=(6, 3))
        self.plot.setMinimumSize(0, 150)
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.axes = self.plot.figure.subplots()
        layout.addWidget(self.plot, 1)
        self.message = QLabel("Click the biofilm to choose its boundary, then update the preview.")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        layout.addWidget(QLabel("Minimum colony area: 60% control · smoothing: 37 px · inner dark area: 50%"))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Confirm and run")
        layout.addWidget(buttons)
        buttons.accepted.connect(self.confirm)
        buttons.rejected.connect(self.reject)
        self.update_button.clicked.connect(self.update_preview)
        self.threshold.valueChanged.connect(self.mark_pending)
        self.first_frame.valueChanged.connect(self.mark_pending)
        self.last_frame.valueChanged.connect(self.mark_pending)
        self.position.currentIndexChanged.connect(self.mark_pending)
        self.capture.valueChanged.connect(self.mark_pending)
        self.plot.mpl_connect("button_press_event", self.pick_colony)
        self.mark_pending()
        self.update_preview()

    def configuration(self):
        return dict(threshold=self.threshold.value(), first=self.first_frame.value(),
                    last=self.last_frame.value(), seeds=deepcopy(self.draft["seeds"]))

    def mark_pending(self, *_):
        self.preview_key = None
        self.threshold_value.setText(f"{self.threshold.value()}%")
        self.message.setText("Settings changed. Select Update preview to check the boundary.")

    def pick_colony(self, event):
        if event.button == 1 and event.inaxes is self.axes and event.xdata is not None and event.ydata is not None:
            self.draft["seeds"][self.position.currentData()] = [event.xdata, event.ydata]
            self.mark_pending()

    def update_preview(self):
        self.preview_key = None
        try:
            position, capture = self.position.currentData(), self.capture.value()
            raw = self.read_frame(position, capture)
            self.axes.clear()
            self.axes.imshow(raw, cmap="gray")
            fit_analysis_view(self.axes, raw)
            self.axes.set_axis_off()
            value = self.hours[capture] if self.hours is not None else capture
            self.axes.set_title(f"{value:g} h" if self.hours is not None else f"Frame {capture}")
            mask = whole_colony_mask(raw, self.threshold.value(), seed=self.draft["seeds"].get(position))
            if not mask.any():
                raise ValueError("No biofilm boundary found. Adjust the threshold or select another colony.")
        except (ValueError, OSError, IndexError) as error:
            self.message.setText(str(error))
            self.plot.draw_idle()
            return False
        edge = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        outline = np.zeros((*mask.shape, 4), np.uint8)
        outline[edge] = (255, 40, 100, 255)
        self.axes.imshow(outline)
        self.plot.figure.tight_layout(pad=.5)
        self.plot.draw_idle()
        self.preview_key = (position, capture, self.threshold.value())
        self.message.setText("Boundary preview updated. Confirm and run applies these settings to the selected range.")
        return True

    def confirm(self):
        if self.first_frame.value() > self.last_frame.value():
            self.message.setText("The first frame must be no later than the last frame.")
            return
        if not self.first_frame.value() <= self.capture.value() <= self.last_frame.value():
            self.capture.setValue(self.first_frame.value())
        if self.preview_key is None and not self.update_preview():
            return
        self.accept()


class DispersionPage(QWidget):
    name = "Dispersion"
    frame_requested = Signal(int, int)
    overlay_changed = Signal()

    def __init__(self):
        super().__init__()
        self.folder = None
        self.rows = []
        self.captures = []
        self._overlay_key = None
        self._overlay_masks = None
        self.recording_count = 0
        self.recording_hours = None
        self.recording_capture = 0
        self.recording_position = 0
        self.colony_seed_points = {}
        self.run_config = None
        self.recording_preview = None
        self.hours = None
        self.preview = None
        self.detail = None
        self.chart = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)
        layout.addWidget(QLabel("<b>Dispersion</b>   How does each microcolony break up over time?"))
        self.strip = DispersionStrip(self)
        self.strip.hour_changed.connect(self.seek_time)
        layout.addWidget(self.strip)
        controls = QHBoxLayout()
        layout.addLayout(controls)
        self.import_button = QToolButton()
        self.import_button.setText("Import dispersion data…")
        self.import_button.setPopupMode(QToolButton.MenuButtonPopup)
        menu = QMenu(self.import_button)
        self.import_recording = menu.addAction("Load recording…")
        self.open_saved = menu.addAction("Open saved Partaker results…")
        self.import_notebook = menu.addAction("Import notebook results…")
        self.import_button.setDefaultAction(self.import_recording)
        self.import_button.setText("Import dispersion data…")
        self.import_button.setMenu(menu)
        controls.addWidget(self.import_button)
        self.run_button = QPushButton("Run dispersion analysis…")
        controls.addWidget(self.run_button)
        self.cancel_button = QPushButton("Cancel")
        controls.addWidget(self.cancel_button)
        controls.addStretch(1)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.status = QLabel()
        self.status.setWordWrap(True)
        body = QHBoxLayout()
        layout.addLayout(body, 1)
        left = QVBoxLayout()
        body.addLayout(left, 1)

        settings = QWidget()
        settings.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        form = QFormLayout(settings)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.addRow(self.status)
        self.config_summary = QLabel("Set boundary and frame range when starting analysis.")
        self.config_summary.setWordWrap(True)
        form.addRow(self.config_summary)
        self.position = QComboBox()
        self.region, self.frame, self.candidates = QComboBox(self), QComboBox(self), QComboBox(self)
        for model in (self.region, self.frame, self.candidates):
            model.hide()
        position_label = QLabel("Position")
        review = QHBoxLayout()
        review.addWidget(position_label)
        review.addWidget(self.position)
        self.review_widgets = [position_label, self.position]
        self.layers = QToolButton()
        self.layers.setText("Show overlays…")
        self.layers.setPopupMode(QToolButton.InstantPopup)
        layer_menu = QMenu(self.layers)
        self.layer_actions = {}
        for label, layer, visible in (
            ("Biofilm boundary", "outer", True), ("Central residual region · pink", "pink", True),
            ("Microcolony territories", "regions", True), ("Dark pixels", "measured", False),
            ("Selected region", "selection", True), ("Candidate crossing vectors", "crossings", False),
            ("Colony tracking · cyan", "tracking", True),
            ("Image-supported motion · white", "motion", True),
            ("Breakup · connected yellow / disconnected blue", "breakup", False),
        ):
            action = layer_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(visible)
            action.toggled.connect(self.update_overlay)
            self.layer_actions[layer] = action
        self.layer_actions["tracking"].setToolTip("Cyan boundary and reference-point diamond; the white trail ends at that diamond, follows up to six supported positions, and restarts after a gap.")
        self.layer_actions["motion"].setToolTip("Saved checked texture displacement from the preceding frame; no arrow means no valid vector. These are not individual-cell tracks.")
        self.layer_actions["breakup"].setToolTip("Per microcolony: largest 8-connected dark component is yellow; other dark components are blue. Only the detected dark-cell pixels in the central region are blue.")
        self.layers.setMenu(layer_menu)
        review.addWidget(self.layers)
        review.addStretch(1)
        form.addRow(review)

        self.field_plot = PlotPanel(self, size=(7, 4))
        self.detail_plot = PlotPanel(self, size=(6, 4))
        self.plot = PlotPanel(self, size=(6, 4))
        self.image_text_color = self.palette().windowText().color().name()
        for panel in (self.field_plot, self.detail_plot, self.plot):
            panel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            panel.setMinimumSize(0, 0)
            panel.figure.set_facecolor("white")
            panel.figure.set_layout_engine("constrained", w_pad=.04, h_pad=.04)
            panel.mpl_connect("button_press_event", self.seek_plot)
        for panel in (self.field_plot, self.detail_plot):
            panel.figure.set_facecolor(self.palette().window().color().name())
            panel.figure.get_layout_engine().set(w_pad=.01, h_pad=.01)
        self.plot.mpl_connect("motion_notify_event", self.drag_time)
        self.plot.mpl_connect("button_release_event", self.release_time)
        self._scrubbing = False
        left.addWidget(self.field_plot, 3)
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.NoFrame)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        settings_scroll.setMinimumSize(0, 0)
        settings_scroll.setMaximumHeight(300)
        settings_scroll.setWidget(settings)
        left.addWidget(settings_scroll, 2)
        right = QVBoxLayout()
        right.addWidget(self.detail_plot, 1)
        right.addWidget(self.plot, 1)
        body.addLayout(right, 1)
        self.position.currentIndexChanged.connect(self.change_position)
        self.region.currentIndexChanged.connect(self.update_overlay)
        self.frame.currentIndexChanged.connect(self.select_frame)
        self.candidates.activated.connect(self.select_candidate)
        self.clear_results()

    def set_recording(self, count, timing, first, last):
        self.recording_count = count
        self.recording_hours = timing["hours"]
        if self.folder is None:
            self.strip.times = list(self.recording_hours) if self.recording_hours is not None else list(range(count))
            self.strip.unit = "Time (h)" if self.recording_hours is not None else "Frame"
            self.recording_capture = min(self.recording_capture, max(0, count - 1))
            self.strip.set_hour(self.strip.times[self.recording_capture] if self.strip.times else 0)
        if self.run_config is None and count:
            first, last = max(0, min(first, count - 1)), max(0, min(last, count - 1))
            self.run_config = dict(threshold=39, first=first, last=last, seeds={})
        if self.folder is None:
            hours = timing["hours"]
            self.status.setText(
                (f"{count} frames · {hours[0]:g}–{hours[-1]:g} h" if hours else f"{count} frames")
                if count else "Load a phase-contrast recording to begin."
            )

    def set_busy(self, busy):
        self.progress.setVisible(busy)
        self.cancel_button.setEnabled(busy)
        self.import_button.setEnabled(not busy)
        self.run_button.setEnabled(bool(self.recording_count) and not busy)

    def configure_run(self, read_frame, positions):
        config = deepcopy(self.run_config)
        config["seeds"] = deepcopy(self.colony_seed_points)
        dialog = DispersionSetupDialog(self, read_frame, positions, self.recording_count, config,
                                       self.recording_hours)
        if dialog.exec() != QDialog.Accepted:
            return None
        return dialog.configuration()

    def show_progress(self, percent, message):
        self.progress.setValue(percent)
        self.status.setText(message)

    def clear_results(self):
        self._trail_key, self._trail_state = None, None
        self.folder = None
        self.run_button.setText("Run dispersion analysis…")
        self.config_summary.setText("Set boundary and frame range when starting analysis.")
        self._overlay_key = None
        self._overlay_masks = None
        self.layers.hide()
        self.rows = []
        self.captures = []
        self.recording_preview = None
        self.recording_capture = 0
        self.recording_position = 0
        self.strip.times = []
        self.strip.redraw()
        self.plot.show()
        self.render_empty()
        for control in self.review_widgets:
            control.hide()

    def show_results(self, folder, manifest, rows, viewer_linked):
        self._trail_key, self._trail_state = None, None
        self.folder = Path(folder)
        self.rows = rows
        self.viewer_linked = viewer_linked
        threshold = manifest.get("settings", {}).get("colony_threshold_percent", 39)
        frames = manifest["frames"]
        self.colony_seed_points = {int(p): list(point) for p, point in
                                   manifest.get("selector_params", {}).get("colony_seed_points", {}).items()}
        self.run_config = dict(threshold=threshold, first=min(frames), last=max(frames),
                               seeds=deepcopy(self.colony_seed_points))
        self.config_summary.setText(f"Boundary threshold {threshold}% · frames {min(frames)}–{max(frames)}")
        self.run_button.setText("Set up new analysis…")
        method = manifest.get("settings", {}).get("dark_measurement", "inside_roi")
        self.config_summary.setToolTip(manifest.get("dark_area_methods", {}).get(method, ""))
        pink = manifest.get("pink_region", {})
        self.automatic_pink = pink.get("method") in ("automatic_connected_expansion", "motion_classification")
        self.central_residual = pink.get("method") == "central_shared_edges"
        self.pink_candidate_frames = {tuple(pair) for pair in pink.get("candidate_frames", [])}
        self._overlay_key = None
        self._overlay_masks = None
        self.uncertain_frames = {tuple(pair) for pair in manifest.get("pink_region", {}).get("uncertain_frames", [])}
        self.captures = manifest["frames"]
        self.hours = manifest["dispersion_timing"]["hours"]
        self.strip.times = [self.time(capture) for capture in self.captures]
        self.strip.unit = "Time (h)" if self.hours is not None else "Frame"
        self.strip.set_hour(self.strip.times[0] if self.strip.times else 0)
        self.populate(self.position, [(str(p), p) for p in manifest["positions"]])
        for control in self.review_widgets:
            control.show()
        self.layers.show()
        self.plot.show()
        self.select_position()

    @staticmethod
    def populate(combo, entries):
        combo.blockSignals(True)
        combo.clear()
        for label, value in entries:
            combo.addItem(label, value)
        combo.blockSignals(False)

    def change_position(self):
        self.select_position()
        self.select_frame()

    def select_position(self):
        self.position_rows = [r for r in self.rows if int(r["position"]) == self.position.currentData()]
        ids = sorted({int(r["region"]) for r in self.position_rows})
        self.populate(self.region, [("Select a microcolony…", None)] + [(str(i), i) for i in ids])
        self.populate(self.frame, [(f"{self.time(t):g} h" if self.hours is not None else str(t), t)
                                   for t in self.captures])
        self.select_region()

    def update_overlay(self):
        if self.folder is not None:
            self.select_region()
            self.overlay_changed.emit()

    def select_region(self):
        region = self.region.currentData()
        self.region_rows = {int(r["capture"]): r for r in self.position_rows if int(r["region"]) == region}
        self.populate(self.candidates, [("Select…", None)] + [
            (f"{self.time(t):g} ({r['outward_crossing_candidates']})", t)
            for t, r in self.region_rows.items() if int(r["outward_crossing_candidates"] or 0) > 0
        ])
        self.render()

    def select_candidate(self, index):
        capture = self.candidates.itemData(index)
        if capture is not None:
            self.frame.setCurrentIndex(self.frame.findData(capture))

    def time(self, capture):
        return capture if self.hours is None else self.hours[capture]

    def select_frame(self):
        capture = self.frame.currentData()
        if capture is not None and self.folder is not None:
            self.render()
            self.frame_requested.emit(self.position.currentData(), capture)

    def follow_frame(self, position, capture):
        position_index = self.position.findData(position)
        if self.folder is None or capture not in self.captures or position_index < 0:
            return
        changed_position = self.position.currentIndex() != position_index
        if changed_position:
            self.position.blockSignals(True)
            self.position.setCurrentIndex(position_index)
            self.position.blockSignals(False)
            self.select_position()
        frame_index = self.frame.findData(capture)
        if frame_index != self.frame.currentIndex():
            self.frame.blockSignals(True)
            self.frame.setCurrentIndex(frame_index)
            self.frame.blockSignals(False)
            self.render()

    def overlay(self, position, capture, *, marker=True):
        if capture not in self.captures or self.position.findData(position) < 0:
            return None
        key = position, capture
        if key != self._overlay_key:
            path = self.folder / "masks" / f"p{position}_t{capture}.npz"
            if not path.is_file():
                return None
            keys = ("raw_gray", "region_ids", "inner_mask", "outer_mask", "pink_mask", "measured_mask", "vectors_json")
            with np.load(path, allow_pickle=False) as masks:
                self._overlay_masks = {name: masks[name] for name in keys if name in masks}
            self._overlay_key = key
        layers = {name for name, action in self.layer_actions.items() if action.isChecked()}
        base_layers = layers - {"regions", "pink", "selection", "measured"} if "breakup" in layers else layers
        overlay = render_overlay(self._overlay_masks, base_layers, selected=self.region.currentData())
        if "breakup" in layers:
            overlay = draw_breakup_overlay(overlay, self._overlay_masks)
        if layers & {"tracking", "motion"}:
            selected = self.region.currentData()
            trail = self.displacement_path(position, capture, selected) if "tracking" in layers and selected is not None else []
            overlay = draw_tracking_overlay(overlay, self._overlay_masks, self.region.currentData(),
                                            territory="tracking" in layers, motion="motion" in layers,
                                            point=trail[-1] if trail else None, marker=marker)
        if "crossings" in layers:
            row = self.region_rows.get(capture, {})
            for start, end in json.loads(row.get("crossing_vectors") or "[]"):
                cv2.arrowedLine(overlay, tuple(map(round, start)), tuple(map(round, end)),
                                (255, 255, 255), 2, tipLength=0.3)
        return overlay

    def displacement_path(self, position, capture, selected):
        key = self.folder, position, selected
        if getattr(self, "_trail_key", None) != key:
            self._trail_key = key
            self._trail_state = dict(paths={}, previous=None, capture=None, count=0)
        state = self._trail_state
        for index in range(state["count"], self.captures.index(capture) + 1):
            current = self.captures[index]
            row = self.region_rows.get(current, {})
            focused = str(row.get("focused", "True")).lower() == "true"
            usable = focused and bool(row) and str(row.get("association_event", "False")).lower() != "true" and not float(row.get("occluded_area_px") or 0)
            path = self.folder / "masks" / f"p{position}_t{current}.npz"
            if not usable or not path.is_file():
                state["paths"][current] = []
                state["previous"] = None
            else:
                with np.load(path, allow_pickle=False) as data:
                    target = {name: data[name] for name in ("region_ids", "raw_gray")}
                consecutive = state["capture"] is not None and current == state["capture"] + 1
                prior = state["paths"].get(state["capture"], []) if consecutive else []
                state["paths"][current] = advance_colony_trail(state["previous"] if consecutive else None,
                                                               target, selected, prior)
                state["previous"] = target
            state["capture"], state["count"] = current, index + 1
        return state["paths"].get(capture, [])

    def draw_displacement(self, axes, selected):
        action = getattr(self, "layer_actions", {}).get("tracking")
        if selected is None or action is None or not action.isChecked():
            return
        path = self.displacement_path(self.position.currentData(), self.frame.currentData(), selected)
        if len(path) > 1:
            points = np.asarray(path)
            axes.plot(points[:, 0], points[:, 1], color="white", linewidth=1.5,
                      path_effects=[patheffects.Stroke(linewidth=3.5, foreground="#222222"),
                                    patheffects.Normal()], zorder=10)
            if axes is self.detail:
                height, width = self._overlay_masks["region_ids"].shape
                left, right = axes.get_xlim()
                bottom, top = axes.get_ylim()
                axes.set_xlim(max(BORDER - .5, min(left, points[:, 0].min() - 4)),
                              min(width - BORDER - .5, max(right, points[:, 0].max() + 4)))
                axes.set_ylim(min(height - BORDER - .5, max(bottom, points[:, 1].max() + 4)),
                              max(BORDER - .5, min(top, points[:, 1].min() - 4)))
        if path:
            axes.plot(path[-1][0], path[-1][1], marker="D", markersize=4,
                      markerfacecolor="#00e6ff", markeredgecolor="white", markeredgewidth=.7,
                      linestyle="None", zorder=11)

    def seek_time(self, value):
        if self.folder is not None and self.captures:
            capture = min(self.captures, key=lambda t: abs(self.time(t) - value))
            self.frame.setCurrentIndex(self.frame.findData(capture))
        elif self.recording_count:
            times = self.recording_hours if self.recording_hours is not None else range(self.recording_count)
            capture = min(range(self.recording_count), key=lambda t: abs(times[t] - value))
            if capture != self.recording_capture:
                self.frame_requested.emit(self.recording_position, capture)

    def show_recording_frame(self, image, position, capture):
        if self.folder is not None or not 0 <= capture < self.recording_count:
            return
        self.recording_capture = capture
        self.recording_position = position
        self.recording_preview = image
        times = self.recording_hours if self.recording_hours is not None else range(self.recording_count)
        self.strip.set_hour(times[capture])
        self.render_empty()

    def seek_plot(self, event):
        if event.button != 1 or event.xdata is None:
            return
        if self.folder is None:
            return
        if event.inaxes is self.chart:
            self._scrubbing = True
            self.seek_time(event.xdata)
        elif event.inaxes in (self.preview, self.detail) and event.ydata is not None and self._overlay_masks is not None:
            labels = self._overlay_masks["region_ids"]
            x, y = int(np.floor(event.xdata + 0.5)), int(np.floor(event.ydata + 0.5))
            if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
                index = self.region.findData(int(labels[y, x]))
                self.region.setCurrentIndex(max(0, index))

    def drag_time(self, event):
        if self._scrubbing and event.inaxes is self.chart and event.xdata is not None:
            self.seek_time(event.xdata)

    def release_time(self, _event):
        self._scrubbing = False

    @staticmethod
    def breakup_score(row):
        """Fraction of dark area outside the largest connected dark patch."""
        if row.get("breakup_score") not in (None, ""):
            return float(row["breakup_score"])
        area, connected = row.get("dark_area_px"), row.get("connected_area_px")
        if area in (None, "") or connected in (None, "") or float(area) <= 0:
            return float("nan")
        return 1.0 - float(connected) / float(area)

    def render_empty(self):
        if self.chart is None:
            self.preview = self.field_plot.figure.subplots()
            self.detail = self.detail_plot.figure.subplots()
            self.chart = self.plot.figure.subplots()
        else:
            self.chart.clear()
            self.preview.clear()
            self.detail.clear()
        if self.recording_preview is None:
            self.preview.set_facecolor("black")
            self.preview.text(0.5, 0.5, "Load a recording", ha="center", va="center",
                              color="white", transform=self.preview.transAxes, fontsize=9, clip_on=True)
            self.preview.set(xticks=[], yticks=[])
        else:
            raw = self.recording_preview
            self.preview.imshow(raw, cmap="gray")
            fit_analysis_view(self.preview, raw)
            self.preview.set_axis_off()
            times = self.recording_hours if self.recording_hours is not None else range(self.recording_count)
            value = times[self.recording_capture]
            self.preview.set_title(f"{value:g} h" if self.recording_hours is not None
                                   else f"Frame {self.recording_capture}", fontsize=10,
                                   color=self.image_text_color)
        self.chart.set(xlabel="Time", ylabel="Breakup score", ylim=(0, 1))
        self.chart.tick_params(labelsize=9)
        self.chart.xaxis.label.set_size(10)
        self.chart.yaxis.label.set_size(10)
        self.render_detail(None, None)
        self.field_plot.draw_idle()
        self.plot.draw_idle()

    def render_detail(self, overlay, selected):
        self.detail.clear()
        self.detail.set_axis_off()
        if selected is None:
            message = "Select a microcolony to view its structure"
        elif overlay is None or self._overlay_masks is None:
            message = "No image available for this frame"
        else:
            yy, xx = np.nonzero(self._overlay_masks["region_ids"] == selected)
            if len(xx):
                detail = overlay.copy()
                other = self._overlay_masks["region_ids"] != selected
                raw = self._overlay_masks.get("raw_gray")
                gray = raw if raw is not None else cv2.cvtColor(overlay, cv2.COLOR_RGB2GRAY)
                detail[other] = np.repeat(gray[..., None], 3, axis=2)[other]
                self.detail.imshow(detail)
                height, width = overlay.shape[:2]
                padding = max(12, .2 * max(np.ptp(xx) + 1, np.ptp(yy) + 1))
                self.detail.set_xlim(max(BORDER - .5, xx.min() - padding),
                                     min(width - BORDER - .5, xx.max() + padding))
                self.detail.set_ylim(min(height - BORDER - .5, yy.max() + padding),
                                     max(BORDER - .5, yy.min() - padding))
                self.detail.set_title(f"Colony {selected} · {self.time(self.frame.currentData()):g}" +
                                      (" h" if self.hours is not None else " frame"), fontsize=10,
                                      color=getattr(self, "image_text_color", "#dddddd"))
                if getattr(self, "layer_actions", {}).get("tracking") is not None:
                    self.draw_displacement(self.detail, selected)
                self.detail.figure.canvas.draw_idle()
                return
            message = f"Colony {selected} not detected in this frame"
        self.detail.text(.5, .5, message, ha="center", va="center", wrap=True,
                         transform=self.detail.transAxes, fontsize=9,
                         color=getattr(self, "image_text_color", "#dddddd"))
        self.detail.figure.canvas.draw_idle()

    def render(self):
        capture = self.frame.currentData()
        if self.folder is None or capture is None:
            return
        self.chart.clear()
        self.preview.clear()
        position = self.position.currentData()
        row = self.region_rows.get(capture, {})
        self.strip.set_hour(self.time(capture))
        overlay = self.overlay(position, capture, marker=False)
        if overlay is not None:
            self.preview.imshow(overlay)
            fit_analysis_view(self.preview, overlay)
        self.preview.set_axis_off()
        selected = self.region.currentData()
        self.preview.set_title("Full biofilm · " + f"{self.time(capture):g}" +
                               (" h" if self.hours is not None else " frame"), fontsize=10,
                               color=self.image_text_color)
        self.render_detail(overlay, selected)
        self.draw_displacement(self.preview, selected)
        if self.region_rows:
            scores = [self.breakup_score(self.region_rows.get(t, {})) for t in self.captures]
            self.chart.plot([self.time(t) for t in self.captures], scores, ".-", color="#0077b6")
            score = self.breakup_score(row)
            if np.isfinite(score):
                self.chart.plot(self.time(capture), score, "o", color="#d95f02")
        if selected is not None:
            self.chart.axvline(self.time(capture), color="#0077b6", linewidth=1)
        first, last = self.time(self.captures[0]), self.time(self.captures[-1])
        padding = max((last - first) * 0.03, 0.05)
        self.chart.set(xlabel="Frame" if self.hours is None else "Time (h)",
                       ylabel="Breakup score", ylim=(0, 1), xlim=(first - padding, last + padding),
                       title=f"Colony {selected}" if selected is not None else "")
        self.chart.tick_params(labelsize=9)
        self.chart.xaxis.label.set_size(10)
        self.chart.yaxis.label.set_size(10)
        self.chart.title.set_fontsize(10)
        self.plot.draw_idle()
        self.field_plot.draw_idle()
        if self.progress.isVisible():
            return
        motion_class = row.get("motion_class")
        if motion_class:
            self.status.setToolTip(
                f"Results: {self.folder}\n"
                f"Identity: {row.get('identity_status')}\n"
                f"Overlap: {row.get('match_iou') or 'unavailable'}; advantage: {row.get('match_advantage') or 'unavailable'}\n"
                f"Motion decision: {row.get('motion_reason')}\n"
                f"Checked points: {row.get('motion_points')}\n"
                f"Residual displacement: {row.get('median_residual_px')} px; threshold: {row.get('motion_threshold_px')} px\n"
                f"High observations in last 3 comparisons: {row.get('motion_high_observations')}"
            )
        else:
            self.status.setToolTip(str(self.folder))
        if getattr(self, "central_residual", False):
            pink_status = "Central residual region · pink. "
        elif motion_class:
            pink_status = {
                "highly_motile_candidate": "High-motion candidate · pink. ",
                "pending": "High motion observed; awaiting repeat evidence. ",
                "typical": "Motion below the high-motion threshold. ",
                "insufficient_evidence": "Insufficient motion evidence. ",
            }.get(motion_class, "")
        elif (position, capture) in self.uncertain_frames:
            pink_status = "Pink region uncertain. "
        elif (position, capture) in self.pink_candidate_frames:
            pink_status = "Automatic pink candidate. "
        else:
            pink_status = "Pink region not detected. " if self.automatic_pink else ""
        score = self.breakup_score(row)
        score_text = f"{score:.3f}" if np.isfinite(score) else "unavailable"
        selection_text = f"Colony {selected} · Breakup score {score_text}. " if selected is not None else ""
        self.status.setText(
            f"{pink_status}{selection_text}".rstrip() + "\n"
            "Click a microcolony to select it. Drag the time cursor on the graph or the timeline to change frames."
        )
        if selected is not None:
            self.chart.set_title(f"Colony {selected} · {score_text}", fontsize=10)
