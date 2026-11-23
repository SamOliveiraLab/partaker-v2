"""Run dispersion in the Biofilm Stages tab."""

import csv
from copy import deepcopy
import json
import logging
from pathlib import Path

from pubsub import pub
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox

from nd2_analyzer.analysis.dispersion.microcolony_dynamics import (
    CancelledAnalysis, DispersionAnalysis, DispersionSettings, find_reviewed_results, reviewed_inputs,
)
from nd2_analyzer.data.appstate import ApplicationState
from nd2_analyzer.data.image_data import ImageData


logger = logging.getLogger(__name__)


class DispersionWorker(QThread):
    progress = Signal(int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, data, settings, positions, frames, timing, focus, parent, selector_params=None):
        super().__init__(parent)
        self.args = data, settings, positions, frames
        self.timing = timing
        self.focus = focus
        self.selector_params = selector_params or {}

    def report_progress(self, percent, message):
        logger.info("Dispersion %s%%: %s", percent, message)
        self.progress.emit(percent, message)

    def run(self):
        analysis = None
        try:
            analysis = DispersionAnalysis(
                *self.args, Path.home() / "Documents" / "Partaker" / "Dispersion",
                dispersion_timing=self.timing, focus_intervals=self.focus, selector_params=self.selector_params,
                progress=self.report_progress, cancelled=self.isInterruptionRequested,
            )
            if analysis.settings.pink_source:
                analysis.prepare_reviewed_recording()
                analysis.loaded_saved = True
            else:
                analysis.prepare_recording()
            analysis.run(parameter_sensitivity=False, export_figures=False)
            if self.isInterruptionRequested():
                raise CancelledAnalysis()
        except CancelledAnalysis:
            if analysis is not None:
                try:
                    analysis.write_manifest("cancelled")
                except OSError:
                    pass
        except Exception as error:
            logger.exception("Dispersion analysis failed")
            if analysis is not None:
                try:
                    analysis.write_manifest("failed")
                except OSError as manifest_error:
                    self.failed.emit(f"{error} (Could not update result manifest: {manifest_error})")
                    return
            self.failed.emit(str(error))
        else:
            logger.info("Dispersion results saved to %s", analysis.output)
            self.completed.emit(analysis)


class DispersionController(QObject):
    def __init__(self, page, parent=None):
        super().__init__(parent)
        self.page = page
        self.open_experiment = None
        self.set_view = None
        self.refresh_view = None
        self.worker = None
        self.image_data = None
        self.transform = None
        self.timing = {"mode": "Frames", "hours": None}
        self.live_result = None
        page.import_recording.triggered.connect(self.load_recording)
        page.open_saved.triggered.connect(self.open_results)
        page.import_notebook.triggered.connect(self.import_notebook)
        page.run_button.clicked.connect(lambda: self.run())
        page.cancel_button.clicked.connect(self.cancel)
        page.frame_requested.connect(self.show_frame)
        page.overlay_changed.connect(self.refresh)
        pub.subscribe(self.on_recording, "image_data_loaded")
        pub.subscribe(self.on_experiment, "experiment_loaded")
        pub.subscribe(self.on_view_index_changed, "view_index_changed")
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)
        self.on_recording(ImageData.get_instance())

    def on_recording(self, image_data):
        offsets = getattr(image_data, "registration_offsets", None)
        transform = (getattr(image_data, "crop_coordinates", None),
                     None if offsets is None else offsets.tobytes())
        if image_data is self.image_data and transform == self.transform:
            return
        self.cancel()
        if image_data is not self.image_data:
            self.page.colony_seed_points.clear()
            self.page.run_config = None
            self.timing = {"mode": "Frames", "hours": None}
        self.image_data, self.transform = image_data, transform
        self.live_result = None
        self.page.clear_results()
        state = ApplicationState.get_instance()
        experiment = state.experiment if state else None
        if experiment is not None and getattr(state, "image_data", None) is image_data:
            self.on_experiment(experiment)
        else:
            self.update_scope()

    def on_experiment(self, experiment):
        self.timing = getattr(experiment, "dispersion_timing", None) or {"mode": "Frames", "hours": None}
        self.update_scope(experiment)

    def update_scope(self, experiment=None):
        count = self.image_data.data.shape[0] if self.image_data is not None else 0
        first, last = getattr(experiment, "time_range", None) or (0, count - 1)
        self.page.set_recording(count, self.timing, first, last)
        self.page.set_busy(self.worker is not None)
        if self.page.folder is None and count:
            index = getattr(ApplicationState.get_instance(), "view_index", None)
            capture, position, channel = index if index is not None else (0, 0, 0)
            if 0 <= capture < count and 0 <= position < self.image_data.data.shape[1]:
                self.preview_recording(position, capture, channel)

    def load_recording(self):
        if self.open_experiment is not None:
            self.open_experiment()

    def import_notebook(self):
        if self.image_data is None:
            self.page.status.setText("Load the matching phase-contrast recording first.")
            return
        folder = QFileDialog.getExistingDirectory(self.page, "Select notebook results")
        if folder:
            self.run(folder)

    def run(self, notebook_folder=None):
        data = self.image_data
        if data is None or self.worker is not None:
            return
        state = ApplicationState.get_instance()
        exp = state.experiment if state else None
        roles = (getattr(exp, "biofilm", None) or {}).get("channel_roles", {})
        phase = next((int(channel) for channel, role in roles.items() if role == "phase"), 0)
        focus = []
        if exp is not None and exp.focus_loss_intervals and exp.time_interval_hours > 0:
            focus = [(a / exp.time_interval_hours, b / exp.time_interval_hours) for a, b in exp.focus_loss_intervals]
        count = data.data.shape[1]
        positions = [p for p in (getattr(exp, "selected_positions", None) or range(count)) if 0 <= p < count]
        path = ""
        try:
            if not positions:
                raise ValueError("Select at least one position in the recording.")
            if not 0 <= phase < data.data.shape[2]:
                raise ValueError("The phase channel is outside the loaded recording.")
            config = deepcopy(self.page.run_config)
            if not notebook_folder:
                config = self.page.configure_run(lambda p, t: data.get(t, p, phase), positions)
                if config is None or data is not self.image_data:
                    return
            frames = list(range(config["first"], config["last"] + 1))
            if notebook_folder:
                path = str(find_reviewed_results(notebook_folder))
                _, _, group = reviewed_inputs(path)
                phase, positions = int(group["channel"]), [int(group["position"])]
                frames = [int(frame["time"]) for frame in group["frames"]]
            if not frames or not positions or min(frames) < 0 or max(frames) >= data.data.shape[0]:
                raise ValueError("Choose a valid frame range and recording.")
            if not 0 <= phase < data.data.shape[2] or any(not 0 <= p < count for p in positions):
                raise ValueError("The selected position or phase channel is outside the loaded recording.")
            settings = DispersionSettings(cell_channel=phase, dark_measurement="intensity_50", pink_source=path,
                                          colony_threshold_percent=config["threshold"])
            worker = self.worker = DispersionWorker(
                data, settings, positions, frames, deepcopy(self.timing), focus, self,
                selector_params={"colony_seed_points": deepcopy(config["seeds"])},
            )
        except (OSError, ValueError, KeyError) as error:
            self.page.status.setText(str(error))
            return
        worker.progress.connect(self.page.show_progress)
        worker.completed.connect(self.on_complete)
        worker.failed.connect(self.on_failed)
        worker.finished.connect(self.on_finished)
        self.page.set_busy(True)
        self.page.show_progress(0, "Preparing dispersion analysis…")
        worker.start()

    def cancel(self):
        if self.worker is not None:
            self.worker.requestInterruption()
            self.page.status.setText("Cancelling dispersion analysis…")

    def shutdown(self):
        if self.worker is not None:
            self.worker.requestInterruption()
            self.worker.wait()

    def on_complete(self, result):
        if self.worker.isInterruptionRequested() or result.data is not self.image_data:
            return
        self.page.progress.hide()
        try:
            self.read_results(result.output, live_result=result)
        except (OSError, ValueError, KeyError) as error:
            logger.exception("Could not open completed dispersion results")
            self.on_failed(str(error))

    def on_failed(self, message):
        self.page.set_busy(self.worker is not None)
        self.page.progress.hide()
        self.page.status.setText(message)
        QMessageBox.warning(self.page, "Dispersion analysis failed", message)

    def on_finished(self):
        worker, self.worker = self.worker, None
        if worker.isInterruptionRequested():
            self.page.status.setText("Dispersion analysis cancelled.")
        self.page.set_busy(False)
        worker.deleteLater()

    def open_results(self):
        path, _ = QFileDialog.getOpenFileName(
            self.page, "Open saved Partaker results", "", "Analysis manifest (analysis_manifest.json)"
        )
        if path:
            try:
                self.read_results(Path(path).parent)
                self.refresh()
            except (OSError, ValueError, KeyError) as error:
                QMessageBox.warning(self.page, "Dispersion results", str(error))

    def read_results(self, folder, live_result=None):
        manifest = json.loads((folder / "analysis_manifest.json").read_text())
        if manifest["status"] != "complete" or "dispersion_timing" not in manifest:
            raise ValueError("Select the manifest for a completed Partaker Dispersion run.")
        with (folder / "dispersion_dynamics.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not manifest.get("frames") or not manifest.get("positions"):
            raise ValueError("The result has no frames or positions to display.")
        for position in manifest["positions"]:
            for capture in manifest["frames"]:
                if not (folder / "masks" / f"p{position}_t{capture}.npz").is_file():
                    raise ValueError(f"Missing result mask for position {position}, frame {capture}.")
        candidate = live_result if live_result is not None else self.live_result
        linked = candidate is not None and candidate.output == folder
        self.page.show_results(folder, manifest, rows, linked)
        if live_result is not None:
            self.live_result = live_result
        self.page.select_frame()

    def show_frame(self, position, capture):
        if self.set_view is None:
            return
        if self.matches_view():
            self.set_view(position, capture, self.live_result.settings.cell_channel)
        elif self.page.folder is None and self.image_data is not None:
            state = ApplicationState.get_instance()
            experiment = state.experiment if state else None
            roles = (getattr(experiment, "biofilm", None) or {}).get("channel_roles", {})
            phase = next((int(channel) for channel, role in roles.items() if role == "phase"), 0)
            self.set_view(position, capture, phase)

    def on_view_index_changed(self, index):
        if not self.page.isVisible():
            return
        capture, position, channel = index
        if self.page.folder is None:
            self.preview_recording(position, capture, channel)
        elif self.matches_view() and channel == self.live_result.settings.cell_channel:
            self.page.follow_frame(position, capture)

    def preview_recording(self, position, capture, channel):
        data = self.image_data
        if data is None:
            return
        if not (0 <= capture < data.data.shape[0] and 0 <= position < data.data.shape[1]
                and 0 <= channel < data.data.shape[2]):
            return
        image = data.get(capture, position, channel)
        self.page.show_recording_frame(image, position, capture)

    def matches_view(self):
        return self.live_result is not None and self.page.folder == self.live_result.output

    def overlay(self, position, capture, channel):
        if self.page.isVisible() and self.matches_view() and channel == self.live_result.settings.cell_channel:
            return self.page.overlay(position, capture)
        return None

    def refresh(self):
        if self.refresh_view is not None:
            self.refresh_view()
