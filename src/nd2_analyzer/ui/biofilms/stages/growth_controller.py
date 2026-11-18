"""The wiring between the tool and the Maturation page (the growth states), as the Seeding and Microcolonies controllers
do for their pages. It reads what the tool has (the loaded recording, the experiment's biofilm block, the cell masks of
every segmented position), reads the two dye frames the user gives, runs the growth-states measure in a worker thread,
moves the view area's T slider with the cursor, opens the Maturation window on Follow and writes the exports."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from PySide6.QtWidgets import QFileDialog
from pubsub import pub

from nd2_analyzer.analysis.biofilm_stages import growth_plots as GP
from nd2_analyzer.analysis.biofilm_stages import matrix_stain as MX
from nd2_analyzer.analysis.biofilm_stages import stage_tables as ST
from nd2_analyzer.analysis.biofilm_stages.growth_film import FIGURE_IN, GrowthFilm
from nd2_analyzer.analysis.biofilm_stages.growth_measure import load_growth_states, measure_growth_states, save_growth_states
from nd2_analyzer.data.appstate import ApplicationState
from nd2_analyzer.data.image_data import ImageData
from .growth_window import GrowthDialog
from .microcolony_controller import STATE_DIR, recording_words

WORKER_WAIT_MS = 30000


def frames_to_measure(segmented: list, step: int = 1, skip: Optional[Callable[[int], bool]] = None) -> list:
    """Every step-th frame counted from the first segmented one, and the last one (a growth curve needs the end of the
    recording), without the frames `skip` names (out of focus)."""
    frames = sorted(int(t) for t in segmented)
    if not frames:
        return []
    return [t for t in frames if ((t - frames[0]) % max(int(step), 1) == 0 or t == frames[-1]) and not (skip is not None and skip(t))]


def channel_with_role(channel_roles: dict, role: str) -> Optional[int]:
    for key, value in sorted((channel_roles or {}).items(), key=lambda kv: int(kv[0])):
        if value == role:
            return int(key)
    return None


class GrowthWorker(QThread):
    """Runs the growth-states measure off the window's thread. The dye channel is read through a file handle of its own,
    so the window can keep showing frames meanwhile."""
    progress = Signal(int, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, labels_at, frames, pixel_um, hour_of, reference, dye_channel, recording_path, fallback_dye_at, matrix_frames=None, parent=None):
        super().__init__(parent)
        self.args = (labels_at, frames, float(pixel_um), hour_of, reference, dye_channel, recording_path, fallback_dye_at); self.generation = 0
        self.matrix_frames = matrix_frames

    def run(self):
        labels_at, frames, pixel_um, hour_of, reference, channel, path, fallback = self.args; handle = None; dye_at = None
        try:
            if reference is not None and channel is not None:
                if path and str(path).lower().endswith(".nd2"):
                    import nd2
                    handle = nd2.ND2File(str(path)); data = handle.to_dask()
                    dye_at = (lambda p, t: np.asarray(data[t, p, channel])) if data.ndim == 5 else None
                dye_at = dye_at or fallback
            out = measure_growth_states(labels_at, dye_at, frames, pixel_um, hour_of, reference if dye_at is not None else None,
                                        progress=lambda k, n: self.progress.emit(int(100 * k / max(n, 1)), f"{100 * k // max(n, 1)} %"),
                                        matrix_frames=self.matrix_frames)
        except Exception as exc:                                   # the page shows the reason in the Growth States step
            self.failed.emit(f"{type(exc).__name__}: {exc}"); return
        finally:
            if handle is not None:
                handle.close()
        self.done.emit(out)


class GrowthController(QObject):
    measured_changed = Signal()                # a measure arrived, was loaded or was cleared: the Microcolonies strip takes its Th1
    # the tool's messages and the methods that hear them, with the names each message carries (see MicrocolonyController)
    LISTENS = (("experiment_loaded", "on_experiment_loaded"), ("image_data_loaded", "on_new_recording"), ("batch_segmentation_completed", "on_segmented"))

    def __init__(self, page, recording_info: Callable[[], object], parent=None):
        super().__init__(parent)
        self.page, self.recording_info = page, recording_info
        self.set_frame: Optional[Callable[[int], None]] = None
        self.measured = None; self.reference: Optional[MX.DyeReference] = None; self.dye_paths: Optional[tuple] = None
        self.worker: Optional[GrowthWorker] = None; self.window: Optional[GrowthDialog] = None; self.generation = 0; self.every_set_for = None
        page.dye_requested.connect(self.choose_dye_frames); page.measure_requested.connect(self.measure); page.hour_changed.connect(self.on_hour_changed)
        page.window_requested.connect(self.open_window); page.export_requested.connect(self.export)
        for topic, method in self.LISTENS:
            pub.subscribe(getattr(self, method), topic)
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    def shutdown(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption(); self.worker.wait(WORKER_WAIT_MS)

    def connect_main_window(self, set_frame: Optional[Callable[[int], None]] = None):
        self.set_frame = set_frame; self.refresh()

    # ------------------------------------------------------------------ what the tool has
    def on_experiment_loaded(self, experiment):
        self.refresh()

    def on_segmented(self, time_range=None, positions=None, model=None):
        self.refresh()

    def on_new_recording(self, image_data):
        self.generation += 1; self.measured = None; self.reference = None; self.dye_paths = None; self.close_window(); self.page.set_measured(None); self.refresh()
        self.measured_changed.emit()

    def segmented(self) -> Dict[int, List[int]]:
        """position -> its segmented frames, from the mask cache of the model with the most frames."""
        array, best = self.mask_array()
        return best

    def mask_array(self):
        image_data = ImageData.get_instance(); cache = getattr(image_data, "segmentation_cache", None) if image_data is not None else None
        chosen = (None, {})
        for _model, entry in (getattr(cache, "mmap_arrays_idx", {}) or {}).items():
            try:
                array, indices = entry
            except (TypeError, ValueError):
                continue
            per: Dict[int, List[int]] = {}
            for i in indices:
                if len(i) >= 2:
                    per.setdefault(int(i[1]), []).append(int(i[0]))
            if sum(len(v) for v in per.values()) > sum(len(v) for v in chosen[1].values()):
                chosen = (array, {p: sorted(set(v)) for p, v in per.items()})
        return chosen

    def hour_of(self, info) -> Callable[[int], float]:
        dt_h = (info.dt_s or 0.0) / 3600.0
        return lambda t: float(info.start_hour) + int(t) * dt_h

    def raw_at(self, channel: int) -> Callable[[int, int], np.ndarray]:
        image_data = ImageData.get_instance()
        return lambda p, t: np.asarray(image_data.get(int(t), int(p), int(channel)))

    def refresh(self):
        info = self.recording_info(); page = self.page
        if info is None:
            page.set_recording("", None); page.set_measure_enabled(False)
            for step in ("load", "segment", "dye", "measure"):
                page.set_step_state(step, "todo")
            return
        per = {p: f for p, f in self.segmented().items() if len(f) >= 2}; dt_h = (info.dt_s or 0.0) / 3600.0; hour = self.hour_of(info)
        words = recording_words(info, 0).rsplit(", position", 1)[0]
        page.set_recording(words + (f", {len(per)} positions segmented" if len(per) != 1 else f", position {next(iter(per))} segmented"),
                           (hour(0), hour(info.n_frames - 1)) if dt_h else None)
        page.set_step_state("load", "done", f"{info.n_frames:,}")
        total = sum(len(f) for f in per.values())
        page.set_step_state("segment", "done" if per else "todo", f"{total:,}, {len(per)} positions" if len(per) != 1 else f"{total:,}, position {next(iter(per))}")
        page.step_labels["segment"].setToolTip("Positions " + ", ".join(str(p) for p in sorted(per)) if per else "")
        busy = self.worker is not None and self.worker.isRunning()
        page.set_measure_enabled(bool(per and dt_h and info.pixel_um) and not busy)
        if dt_h and self.every_set_for != self.generation:
            self.every_set_for = self.generation; page.set_every(dt_h)
        if self.reference is None:
            has_dye = channel_with_role(info.channel_roles, "matrix") is not None
            page.set_step_state("dye", "todo", "" if has_dye else "no matrix dye channel in this experiment")
        if self.measured is None and not busy:
            page.set_step_state("measure", "todo")
            page.status.setText("Give the dye frames (to measure the matrix stain and Th1), then click Measure." if per else
                                "Segment the positions you keep in the Segmentation tab, then come back here.")

    # ------------------------------------------------------------------ the dye frames
    def choose_dye_frames(self):
        before, _ = QFileDialog.getOpenFileName(self.page, "The frame taken BEFORE the dye was added", "", "Recordings (*.nd2)")
        if not before:
            return
        after, _ = QFileDialog.getOpenFileName(self.page, "The frame taken right AFTER the dye was added", str(before), "Recordings (*.nd2)")
        if after:
            self.set_dye_frames(before, after)

    def set_dye_frames(self, before_path: str, after_path: str) -> bool:
        info = self.recording_info(); channel = channel_with_role(info.channel_roles, "matrix") if info is not None else None
        if channel is None:
            self.page.set_step_state("dye", "failed", "no channel of this experiment is the matrix dye"); return False
        try:
            before, after = MX.read_dye_frames(before_path, channel), MX.read_dye_frames(after_path, channel)
            self.reference = MX.dye_reference(before, after); self.dye_paths = (str(before_path), str(after_path))
        except Exception as exc:
            self.reference = None; self.page.set_step_state("dye", "failed", f"{type(exc).__name__}: {exc}"); return False
        self.page.set_step_state("dye", "done", "before and after")
        self.page.step_labels["dye"].setToolTip(f"Before the dye: {before_path} ({len(before)} positions)\nRight after: {after_path} ({len(after)} positions)")
        return True

    # ------------------------------------------------------------------ Measure
    def measure(self):
        info = self.recording_info()
        if info is None or (self.worker is not None and self.worker.isRunning()):
            return
        array, per = self.mask_array(); dt_h = (info.dt_s or 0.0) / 3600.0
        if array is None or not dt_h or not info.pixel_um:
            return
        state = ApplicationState.get_instance(); exp = getattr(state, "experiment", None) if state is not None else None
        skip = (lambda t: bool(exp.is_focus_loss_frame(t))) if exp is not None and hasattr(exp, "is_focus_loss_frame") else None
        step = max(1, int(round(self.page.every_spin.value() / dt_h)))
        frames = {p: frames_to_measure(f, step, skip) for p, f in per.items() if len(f) >= 2}
        frames = {p: f for p, f in frames.items() if len(f) >= 2}
        # the matrix stain is measured up to the first frame that is out of focus: after it the dye image is on another scale
        lost = [t for f in per.values() for t in f if skip is not None and skip(t)]
        matrix_frames = {p: [t for t in f if t < min(lost)] for p, f in frames.items()} if lost else None
        if not frames:
            self.page.set_step_state("measure", "failed", "no position has two segmented frames to measure"); return
        channel = channel_with_role(info.channel_roles, "matrix"); image_data = ImageData.get_instance()
        files = image_data.image_filename; path = files[0] if isinstance(files, (list, tuple)) and files else files
        labels_at = lambda p, t: np.asarray(array[int(t), int(p), 0])
        self.worker = GrowthWorker(labels_at, frames, info.pixel_um, self.hour_of(info), self.reference, channel, path,
                                   self.raw_at(channel) if channel is not None else None, matrix_frames, self)
        self.worker.generation = self.generation
        self.worker.progress.connect(lambda pct, words: self.page.set_step_state("measure", "running", words))
        self.worker.done.connect(self.on_measured); self.worker.failed.connect(lambda why: (self.page.set_step_state("measure", "failed", why), self.refresh()))
        self.page.set_measure_enabled(False); self.page.set_step_state("measure", "running", "0 %"); self.worker.start()

    def counted_at_for(self, position: int) -> Optional[Callable[[int], np.ndarray]]:
        """frame -> the matrix mask of that frame of the position, once the dye frames are given and the position is
        measured; None before. The Microcolonies window uses it for its matrix stain box."""
        info = self.recording_info(); p = int(position)
        if self.measured is None or self.reference is None or info is None or p not in self.measured.free_dye:
            return None
        channel = channel_with_role(info.channel_roles, "matrix"); raw = self.raw_at(channel); level = self.measured.free_dye[p]; settings = self.measured.settings
        return lambda t: MX.matrix(MX.stain(MX.correct(raw(p, t), self.reference), level, settings), info.pixel_um, settings)[0]

    def stain_recipe(self, position: int) -> Optional[dict]:
        """What is needed to measure the matrix stain of a position as this tab did (so there is one free dye level per
        position in the tool): the reference of the dye frames, the position's free dye level, the settings, the dye
        channel and where to read it. None before the dye frames are given and the position is measured."""
        info = self.recording_info(); p = int(position)
        if self.measured is None or self.reference is None or info is None or p not in self.measured.free_dye:
            return None
        channel = channel_with_role(info.channel_roles, "matrix")
        if channel is None:
            return None
        image_data = ImageData.get_instance(); files = getattr(image_data, "image_filename", None) if image_data is not None else None
        path = files[0] if isinstance(files, (list, tuple)) and files else files
        return {"reference": self.reference, "level": float(self.measured.free_dye[p]), "settings": self.measured.settings, "channel": int(channel),
                "path": path, "raw_at": self.raw_at(channel), "position": p}

    def film(self) -> Optional[GrowthFilm]:
        info = self.recording_info(); array, _per = self.mask_array()
        if self.measured is None or info is None or array is None:
            return None
        dye = channel_with_role(info.channel_roles, "matrix"); phase = channel_with_role(info.channel_roles, "phase")
        return GrowthFilm(self.measured, lambda p, t: np.asarray(array[int(t), int(p), 0]), self.raw_at(phase if phase is not None else 0),
                          self.raw_at(dye) if dye is not None and self.reference is not None else None, self.reference, info.pixel_um)

    def on_measured(self, measured):
        worker = self.sender()
        if worker is None or worker.generation != self.generation:
            return
        self.measured = measured; self.show_measured(measured); self.refresh()

    def show_measured(self, measured):
        self.page.set_measured(measured, self.film())
        th1 = f"Th1 {measured.th1['th1_h']:.1f} h" if measured.th1 and measured.th1["th1_h"] is not None else "Th1 not measured"
        th2 = f"Th2 {measured.th2['th2_h']:.1f} h" if measured.th2 else "Th2 not found"
        self.page.set_step_state("measure", "done", f"{th1}, {th2}")
        n = sum(len(f) for f in measured.frames.values()); first, last = GP._span(measured)
        dye = (f"  Free dye {np.mean(list(measured.free_dye.values())):,.0f} counts, cut-off {measured.settings.cut_off:,.0f} above it, matrix at least "
               f"{measured.settings.min_width_um:g} µm wide." if measured.free_dye else "")
        self.page.status.setText(f"{n:,} frames of {len(measured.positions)} positions, {first:g} to {last:g} h.{dye}  Every number here is computed by the tool on this recording.")
        self.measured_changed.emit()

    # ------------------------------------------------------------------ with the saved session
    def save_state(self, folder) -> list:
        """Writes the growth-states measure and the dye file choice into the session folder. Returns the files written;
        none when nothing was measured."""
        out = Path(folder) / STATE_DIR
        for old in (out / "growth_states.npz", out / "growth_states.json"):       # of an earlier save to this folder
            old.unlink(missing_ok=True)
        if self.measured is None:
            return []
        out.mkdir(parents=True, exist_ok=True)
        written = [save_growth_states(self.measured, out / "growth_states.npz")]
        about = {"dye_frames": list(self.dye_paths) if self.dye_paths else None, "every_h": self.page.every_spin.value()}
        (out / "growth_states.json").write_text(json.dumps(about, indent=1)); written.append(str(out / "growth_states.json"))
        return written

    def load_state(self, folder) -> bool:
        """Brings back what save_state wrote. The two dye files are read again from where they were; when they are no
        longer there the numbers are still shown and the Dye frames step says which files are missing."""
        src = Path(folder) / STATE_DIR
        if not (src / "growth_states.npz").exists():
            return False
        try:
            measured = load_growth_states(src / "growth_states.npz")
            about = json.loads((src / "growth_states.json").read_text()) if (src / "growth_states.json").exists() else {}
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.page.status.setText(f"The growth states saved with this session could not be read ({exc}). Click Measure to measure them again.")
            return False
        self.close_window(); self.measured = measured; self.reference = None; self.dye_paths = None; self.refresh()
        info = self.recording_info(); dt_h = (info.dt_s or 0.0) / 3600.0 if info is not None else 0.0
        if dt_h and about.get("every_h"):
            self.page.set_every(dt_h, float(about["every_h"]))
        paths = about.get("dye_frames")
        if paths and all(os.path.exists(p) for p in paths):
            self.set_dye_frames(*paths)
        elif paths:
            self.dye_paths = tuple(paths); self.page.set_step_state("dye", "failed", "not found where they were: " + ", ".join(os.path.basename(p) for p in paths))
            self.page.step_labels["dye"].setToolTip("\n".join(paths))
        self.show_measured(measured)
        return True

    # ------------------------------------------------------------------ cursor, window, exports
    def on_hour_changed(self, hour: float):
        info = self.recording_info()
        if info is None or not info.dt_s or self.set_frame is None:
            return
        frame = int(round((float(hour) - float(info.start_hour)) * 3600.0 / float(info.dt_s)))
        self.set_frame(max(0, min(frame, info.n_frames - 1)))

    def close_window(self):
        if self.window is not None:
            try:
                self.window.close()
            except RuntimeError:
                pass
            self.window = None

    def open_window(self, position):
        film = self.film()
        if film is None:
            return
        self.close_window()
        rows = [row for _title, rows in GP.entries(self.measured) for row in rows]
        self.window = GrowthDialog(film, rows, position=position, parent=self.page.window())
        self.window.destroyed.connect(lambda *_a: setattr(self, "window", None)); self.window.show()

    def export(self, kind: str):
        if kind == "numbers":
            if self.measured is not None:
                folder = QFileDialog.getExistingDirectory(self.page, "Folder for the numbers (CSV files)")
                if folder:
                    written = ST.write_tables(folder, ST.growth_tables(self.measured))
                    self.page.status.setText(f"Numbers saved: {len(written)} CSV files in {folder}")
            return
        film = self.page.film
        if film is None or not film.frames:
            return
        if kind == "film":
            path, _ = QFileDialog.getSaveFileName(self.page, "Save film", f"maturation_position_{film.position}.gif", "GIF (*.gif)")
            if path and film.save_gif(path):
                self.page.status.setText(f"Film saved: {path}")
        else:
            path, _ = QFileDialog.getSaveFileName(self.page, "Save image", f"maturation_position_{film.position}.png", "PNG (*.png);;PDF (*.pdf)")
            if path:
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                from matplotlib.figure import Figure
                fig = Figure(figsize=FIGURE_IN, dpi=100); FigureCanvasAgg(fig); film.draw(fig, film.frame_at(self.page.hour)); fig.savefig(path, dpi=300, facecolor="white")
                self.page.status.setText(f"Image saved: {path}")
