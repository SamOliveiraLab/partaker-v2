"""The wiring between the tool and the Seeding page.

The controller reads what the tool already has (the loaded recording, the experiment's biofilm block, the cell rows of
MetricsService), tells the page the state of each step, links the cell rows into tracks with the mask tracker when the
user clicks Track, runs the seeding rules when the user clicks Measure (both in worker threads with progress), moves the
view area's T slider when the user moves the cursor, cuts the crops of a followed cell for the film strip, and writes the
exports. The tracks live on the controller, one list per position; tracks the tracking widget already holds for a
position are used when the controller has none of its own. The page never reaches into the main window; the main window
hands the controller two callbacks (set_frame, tracking_widget) through connect_main_window.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import polars as pl
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from PySide6.QtWidgets import QApplication, QFileDialog
from pubsub import pub

from nd2_analyzer.analysis.biofilm_stages import seeding_plots as SP
from nd2_analyzer.analysis.biofilm_stages.seeding_film import SeedingFilm
from nd2_analyzer.analysis.biofilm_stages.mask_tracker import MaskTrackerSettings, track_rows
from nd2_analyzer.analysis.biofilm_stages.seeding_rules import SeedingInputs, measure_seeding, metrics_as_records
from nd2_analyzer.analysis.biofilm_stages.seeding_types import RecordingInfo, SeedingResults, SeedingSettings
from nd2_analyzer.analysis.metrics_service import MetricsService
from nd2_analyzer.data.appstate import ApplicationState
from nd2_analyzer.data.image_data import ImageData

FILM_WINDOW_PX = 120          # crop side around the followed cell
FILM_CROPS = 5                # crops in the film strip
GIF_FPS = 8                   # frames per second of the saved film
GIF_MAX_FRAMES = 400          # a longer life is sampled down to this many frames so the file stays small
EXPORT_DPI = 300
RESULT_FILES = {"metrics": "seeding_metrics.json", "tracks": "seeding_tracks.csv", "counts": "seeding_counts.csv",
                "candidates": "seeding_candidates.csv", "stages": "seeding_stages.csv"}
NO_INTERVAL = "frame interval unknown, set it in the Experiment dialog"
WORKER_WAIT_MS = 30000        # how long a closing window waits for a running step before giving up on it
FILM_PROGRESS_EVERY = 20      # frames between two status lines while the film is written


class TrackWorker(QThread):
    """Links one position's cell rows into tracks off the window's thread and reports progress."""
    progress = Signal(int, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, cells: pl.DataFrame, position: int, settings: MaskTrackerSettings, dt_s: float, parent=None):
        super().__init__(parent)
        self.cells = cells
        self.position = int(position)
        self.settings = settings
        self.dt_s = float(dt_s)
        self.generation = 0                                       # which loaded recording the rows belong to

    def run(self):
        def tell(k, n):
            self.progress.emit(int(100 * k / max(n, 1)), f"{k:,} of {n:,} frames")

        try:
            tracks = track_rows(self.cells, self.position, self.settings, progress=tell, dt_s=self.dt_s)
        except Exception as exc:                                   # the page shows the reason in the Track step
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        finally:
            self.cells = None                                      # the rows are a full copy of the position: let them go
        self.done.emit(tracks)


class MeasureWorker(QThread):
    """Runs measure_seeding off the window's thread and reports progress."""
    progress = Signal(int, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, inputs: SeedingInputs, parent=None):
        super().__init__(parent)
        self.inputs = inputs
        self.generation = 0

    def run(self):
        try:
            results = measure_seeding(self.inputs, progress=lambda pct, text: self.progress.emit(int(pct), str(text)))
        except Exception as exc:                                   # the page shows the reason in the Measure step
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        finally:
            self.inputs = None
        self.done.emit(results)


def hour_to_frame(hour: float, dt_s: float, start_hour: float, n_frames: int) -> int:
    """The frame shown at an hour after inoculation, clamped to the recording."""
    if dt_s <= 0:
        return 0
    frame = int(round((float(hour) - float(start_hour)) * 3600.0 / float(dt_s)))
    return max(0, min(frame, max(int(n_frames) - 1, 0)))


def frame_to_hour(frame: int, dt_s: float, start_hour: float) -> float:
    return float(start_hour) + float(frame) * float(dt_s) / 3600.0


def cells_channel(channel_roles: dict) -> int:
    """The channel the film is cut from: the cell marker, else phase, else channel 0."""
    for wanted in ("cell_marker", "phase"):
        for key, role in sorted((channel_roles or {}).items(), key=lambda kv: int(kv[0])):
            if role == wanted:
                return int(key)
    return 0


def crop_window(xs: np.ndarray, ys: np.ndarray, shape, side: int) -> tuple:
    """(x0, y0, side) of a square window that holds the whole track, at least side px wide, inside the field."""
    span = max(float(np.ptp(xs)), float(np.ptp(ys)), 0.0)
    side = int(max(side, np.ceil(span) + 40))
    ny, nx = int(shape[0]), int(shape[1])
    side = min(side, ny, nx)
    cx, cy = float(np.median(xs)), float(np.median(ys))
    x0 = int(round(cx - side / 2)); y0 = int(round(cy - side / 2))
    x0 = max(0, min(x0, nx - side)); y0 = max(0, min(y0, ny - side))
    return x0, y0, side


def pick_frames(t: np.ndarray, n: int) -> np.ndarray:
    """n evenly spaced indices into a track's frames (all of them when the track is shorter)."""
    if t.size <= n:
        return np.arange(t.size)
    return np.linspace(0, t.size - 1, n).round().astype(int)


def fate_word(results: Optional[SeedingResults], track_id: int) -> str:
    """What happened to the track (rules.fate_class_expr: left | while | end | divided), 'while' when unknown."""
    if results is None or results.per_track is None or not results.per_track.height:
        return "while"
    from nd2_analyzer.analysis.biofilm_stages import seeding_figures as SF
    rows = SF.classed(results).filter(pl.col("track_id") == int(track_id))
    if not rows.height:
        return "while"
    return str(rows["fate_class"][0])


class SeedingController(QObject):
    """Connects the tool to one SeedingPage."""

    def __init__(self, page, parent=None):
        super().__init__(parent)
        self.page = page
        self.results: Optional[SeedingResults] = None
        self.inputs: Optional[SeedingInputs] = None
        self.settings = SeedingSettings()
        self.tracker_settings = MaskTrackerSettings()
        self.worker: Optional[MeasureWorker] = None
        self.track_worker: Optional[TrackWorker] = None
        self.own_tracks: dict = {}                            # position -> the tracks the Track step made
        self.measured_position: Optional[int] = None          # the position the results and the followed track belong to
        self.last_position: Optional[int] = None              # the position the steps row describes
        self.generation = 0                                   # counts loaded recordings, so a late worker result is ignored
        self.set_frame: Optional[Callable[[int], None]] = None
        self.tracking_widget: Optional[Callable[[], object]] = None
        self.pubsub_tracks: Optional[list] = None            # the last tracks published, when no tracking widget is reachable
        self.followed_track: Optional[dict] = None
        self.last_folder: Optional[str] = None
        self.window = None                                    # the Seeding window, when open
        self.seeding_film: Optional[SeedingFilm] = None       # its state, made once per measurement
        self.label_src: Optional[dict] = None                 # where the measured cell rows' labels come from
        page.track_requested.connect(self.track)
        page.measure_requested.connect(self.measure)
        page.hour_changed.connect(self.on_hour_changed)
        page.follow_requested.connect(self.follow)
        if hasattr(page, "window_requested"):
            page.window_requested.connect(self.open_window)
        page.export_requested.connect(self.export)
        if hasattr(page, "clear_requested"):
            page.clear_requested.connect(self.clear_results)
        pub.subscribe(self.on_experiment_loaded, "experiment_loaded")
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")
        pub.subscribe(self.on_tracking_data_available, "tracking_data_available")
        pub.subscribe(self.on_segmentation_completed, "batch_segmentation_completed")
        pub.subscribe(self.on_view_index_changed, "view_index_changed")
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    def shutdown(self):
        """Stop the running step before the window goes: a QThread destroyed while it runs aborts the whole process."""
        for worker in (self.track_worker, self.worker):
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                worker.wait(WORKER_WAIT_MS)

    def connect_main_window(self, set_frame: Optional[Callable[[int], None]] = None,
                            tracking_widget: Optional[Callable[[], object]] = None):
        """The two things the main window gives: a way to show a frame, and a way to reach the tracking widget."""
        self.set_frame = set_frame
        self.tracking_widget = tracking_widget
        self.refresh()

    # ------------------------------------------------------------------------------------------ the tool's state
    def on_experiment_loaded(self, experiment):
        self.refresh()

    def on_image_data_loaded(self, image_data):
        self.generation += 1                                  # results of a worker started on the old recording are dropped
        self.results = None
        self.close_window()
        self.inputs = None
        self.own_tracks = {}
        self.measured_position = None
        self.followed_track = None
        self.page.set_results(None)
        self.refresh()

    def on_view_index_changed(self, index):
        """The user moved to another position: the steps row and the buttons must describe that one."""
        p = int(index[1]) if index is not None and len(index) > 1 else 0
        if p != self.last_position:
            self.refresh()

    def on_tracking_data_available(self, lineage_tracks):
        self.pubsub_tracks = list(lineage_tracks) if lineage_tracks else None
        self.refresh()

    def on_segmentation_completed(self, time_range=None, positions=None, model=None):
        self.refresh()

    def experiment(self):
        state = ApplicationState.get_instance()
        return state.experiment if state is not None else None

    def position(self) -> int:
        """The position the page measures: the one shown in the view area, else 0."""
        state = ApplicationState.get_instance()
        if state is not None and state.view_index is not None:
            return int(state.view_index[1])
        return 0

    def recording_info(self) -> Optional[RecordingInfo]:
        """What the page knows about the loaded recording, from ImageData and the experiment."""
        image_data = ImageData.get_instance()
        if image_data is None or getattr(image_data, "data", None) is None:
            return None
        exp = self.experiment()
        biofilm = (getattr(exp, "biofilm", None) or {}) if exp is not None else {}
        dt_s = None
        if exp is not None and getattr(exp, "phc_interval", None):
            dt_s = float(exp.phc_interval)
        if not dt_s:
            dt_s = float(getattr(image_data, "time_increment_s", None) or 0.0)
        shape = image_data.data.shape
        files = image_data.image_filename
        first = files[0] if isinstance(files, (list, tuple)) and files else files
        name = exp.name if exp is not None and getattr(exp, "name", None) else (Path(str(first)).name if first else "recording")
        return RecordingInfo(name=str(name), n_frames=int(shape[0]), dt_s=dt_s, pixel_um=image_data.pixel_size_um(),
                             start_hour=float(biofilm.get("start_hour", 0.0) or 0.0), flow=biofilm.get("flow"),
                             channel_roles=dict(biofilm.get("channel_roles", {}) or {}), field_shape=tuple(int(v) for v in shape[-2:]))

    def cell_rows(self, position: int) -> pl.DataFrame:
        df = MetricsService().df
        if df is None or df.is_empty() or "position" not in df.columns:
            return pl.DataFrame()
        return df.filter(pl.col("position") == int(position))

    def segmented_frames(self, position: int) -> list:
        """Frames of this position the tool has segmented: the cache's index (empty frames included) and the cell rows."""
        frames = set()
        image_data = ImageData.get_instance()
        cache = getattr(image_data, "segmentation_cache", None) if image_data is not None else None
        for _array, indices in (getattr(cache, "mmap_arrays_idx", {}) or {}).values():
            for idx in indices:
                if len(idx) >= 2 and int(idx[1]) == int(position):
                    frames.add(int(idx[0]))
        rows = self.cell_rows(position)
        if rows.height:
            frames.update(int(t) for t in rows["time"].unique().to_list())
        return sorted(frames)

    def tracks(self, position: int) -> list:
        """The tracks of this position: the Track step's own, else the tracking widget's, else the last ones published."""
        own = self.own_tracks.get(int(position))
        if own:
            return list(own)
        widget = self.tracking_widget() if self.tracking_widget is not None else None
        store = getattr(widget, "tracks_by_position", None) if widget is not None else None
        if store:
            entry = store.get(int(position)) or store.get(str(position))
            if entry and entry.get("lineage_tracks"):
                return list(entry["lineage_tracks"])
        if store is None and self.pubsub_tracks:
            return list(self.pubsub_tracks)
        return []

    def refresh(self):
        """Tell the page what the tool has: the recording and the state of Load, Segment and Track. A step that failed keeps
        its reason until the data it waits for appears or the position changes."""
        info = self.recording_info()
        self.page.set_recording(info)
        if info is None:
            self.last_position = None
            for step in ("load", "segment", "track"):
                self.page.set_step_state(step, "todo")
            return
        p = self.position()
        position_changed = p != self.last_position
        self.last_position = p
        if info.dt_s <= 0:
            self.page.set_step_state("load", "done", f"{info.n_frames:,}, frame interval unknown: set it in the Experiment dialog")
        else:
            self.page.set_step_state("load", "done", f"{info.n_frames:,}")
        frames = self.segmented_frames(p)
        if frames:
            self.page.set_step_state("segment", "done", f"{len(frames):,} with cells, position {p}")
        else:
            self.page.set_step_state("segment", "todo")
        if not self.is_tracking():
            tracks = self.tracks(p)
            if tracks:
                self.page.set_step_state("track", "done", self.track_words(tracks))
            elif position_changed or not self.step_failed("track"):
                self.page.set_step_state("track", "todo")
        if self.results is None and not self.is_measuring() and (position_changed or not self.step_failed("measure")):
            self.page.set_step_state("measure", "todo")
        elif self.results is not None and self.measured_position is not None and p != self.measured_position:
            self.page.set_step_state("measure", "done", f"for position {self.measured_position}, click Measure for position {p}")

    def step_failed(self, step: str) -> bool:
        return self.page.step_states.get(step, ("todo", ""))[0] == "failed"

    @staticmethod
    def track_words(tracks: list) -> str:
        """"1,234 with 19 divisions" for the Cell tracks step."""
        divisions = sum(1 for tr in tracks if len(tr.get("children") or []) >= 2)
        return f"{len(tracks):,} with {divisions:,} division" + ("" if divisions == 1 else "s")

    # ------------------------------------------------------------------------------------------ track
    def track(self):
        """Link this position's cell rows into tracks with the mask tracker, off the window's thread."""
        if self.is_tracking() or self.is_measuring():
            return
        info = self.recording_info()
        p = self.position()
        rows = self.cell_rows(p)
        if info is None or not rows.height:
            self.page.set_step_state("track", "failed", "no segmented frames for this position")
            return
        if info.dt_s <= 0:
            self.page.set_step_state("track", "failed", NO_INTERVAL)
            return
        self.page.set_step_state("track", "running", "0 %")
        worker = TrackWorker(rows, p, self.tracker_settings, info.dt_s, self)
        worker.generation = self.generation
        worker.progress.connect(self.on_track_progress)
        worker.done.connect(lambda tracks, position=p, w=worker: self.on_tracked(tracks, position, w))
        worker.failed.connect(lambda reason, w=worker: self.on_track_failed(reason, w))
        worker.finished.connect(self.on_track_worker_finished)
        worker.finished.connect(worker.deleteLater)
        self.track_worker = worker
        worker.start()

    def on_track_progress(self, pct: int, text: str):
        self.page.set_step_state("track", "running", f"{pct} %, {text}")

    def on_tracked(self, tracks, position: int, worker=None):
        if self.stale(worker):
            return                                            # the recording changed while the tracker ran
        self.own_tracks[int(position)] = list(tracks)
        self.results = None
        self.followed_track = None
        self.close_window()
        self.measured_position = None
        self.page.set_results(None)
        if position == self.position():
            self.page.set_step_state("track", "done", self.track_words(tracks))
            self.page.set_step_state("measure", "todo")
        else:
            self.refresh()

    def on_track_failed(self, reason: str, worker=None):
        if not self.stale(worker):
            self.page.set_step_state("track", "failed", reason)

    def on_track_worker_finished(self):
        self.track_worker = None

    def stale(self, worker) -> bool:
        """True when the worker was started on a recording that has since been replaced."""
        return worker is not None and getattr(worker, "generation", self.generation) != self.generation

    def is_tracking(self) -> bool:
        return self.track_worker is not None and self.track_worker.isRunning()

    # ------------------------------------------------------------------------------------------ measure
    def build_inputs(self) -> Optional[SeedingInputs]:
        info = self.recording_info()
        if info is None:
            return None
        p = self.position()
        frames = self.segmented_frames(p)
        if not frames:
            return None
        return SeedingInputs(cells=self.cell_rows(p), tracks=self.tracks(p), dt_s=info.dt_s, pixel_um=info.pixel_um,
                             start_hour=info.start_hour, n_frames=info.n_frames, field_shape=tuple(info.field_shape),
                             flow=info.flow, settings=self.settings, frame_range=(frames[0], frames[-1]))

    def measure(self):
        if self.is_measuring() or self.is_tracking():
            return
        inputs = self.build_inputs()
        if inputs is None:
            self.page.set_step_state("measure", "failed", "no segmented frames for this position")
            return
        if inputs.dt_s <= 0:
            self.page.set_step_state("measure", "failed", NO_INTERVAL)
            return
        if not inputs.tracks:
            self.page.set_step_state("measure", "failed", "click Track first")
            return
        self.page.set_step_state("measure", "running", "0 %")
        worker = MeasureWorker(inputs, self)
        worker.generation = self.generation
        # what this measurement is about, kept with the worker until its results arrive (a failed run changes nothing)
        worker.request = (inputs, self.position(), self.label_source(self.position(), inputs))
        worker.progress.connect(self.on_progress)
        worker.done.connect(lambda results, w=worker: self.on_done(results, w))
        worker.failed.connect(lambda reason, w=worker: self.on_failed(reason, w))
        worker.finished.connect(self.on_worker_finished)
        worker.finished.connect(worker.deleteLater)
        self.worker = worker
        worker.start()

    def on_progress(self, pct: int, text: str):
        self.page.set_step_state("measure", "running", f"{pct} %, {text}")

    def on_done(self, results, worker=None):
        if self.stale(worker):
            return                                            # the recording changed while the rules ran
        self.close_window()
        if worker is not None and getattr(worker, "request", None) is not None:
            self.inputs, self.measured_position, self.label_src = worker.request
        self.followed_track = None
        self.results = results
        available = sum(1 for m in results.metrics if m.available)
        self.page.set_results(results)
        self.page.set_step_state("measure", "done", f"{available} of {len(results.metrics)}")
        if results.check and hasattr(self.page, "ask_about_seeding"):
            self.page.ask_about_seeding()                     # the frames may not show seeding: say why, keep or clear

    def clear_results(self):
        """The not-seeding warning's "Clear them": the numbers go, the tracks stay, Measure can run again."""
        self.close_window()
        self.results = None
        self.page.set_results(None)
        self.page.set_step_state("measure", "todo", "")

    def on_failed(self, reason: str, worker=None):
        if not self.stale(worker):
            self.page.set_step_state("measure", "failed", reason)

    def on_worker_finished(self):
        self.worker = None

    def is_measuring(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    # ------------------------------------------------------------------------------------------ cursor
    def on_hour_changed(self, hour: float):
        info = self.page.info
        if info is None or self.set_frame is None:
            return
        self.set_frame(hour_to_frame(hour, info.dt_s, info.start_hour, info.n_frames))

    # ------------------------------------------------------------------------------------------ the film
    def film_position(self) -> int:
        """The position the followed track came from: the measured one, else the one shown."""
        return self.measured_position if self.measured_position is not None else self.position()

    def track_by_id(self, track_id: int) -> Optional[dict]:
        for tr in self.inputs.tracks if self.inputs is not None else self.tracks(self.film_position()):
            if int(tr.get("ID", -1)) == int(track_id):
                return tr
        return None

    @staticmethod
    def display_crop(crop: np.ndarray) -> np.ndarray:
        """A crop scaled to 0..1 by its own bright pixels, so masks (0 or 1000) and dim fluorescence both show."""
        crop = np.asarray(crop, dtype=float)
        top = float(np.percentile(crop[crop > 0], 99.5)) if np.any(crop > 0) else 0.0
        return np.clip(crop / top, 0.0, 1.0) if top > 0 else crop

    def film_frames(self, track: dict, n: int = FILM_CROPS):
        """Crops around the followed cell at n evenly spaced frames of its life, with the trail in crop pixels."""
        info = self.page.info
        image_data = ImageData.get_instance()
        if info is None or image_data is None:
            return [], [], [], []
        t = np.asarray(track["t"], dtype=int); x = np.asarray(track["x"], dtype=float); y = np.asarray(track["y"], dtype=float)
        order = np.argsort(t, kind="stable"); t, x, y = t[order], x[order], y[order]
        x0, y0, side = crop_window(x, y, info.field_shape, FILM_WINDOW_PX)
        p, c = self.film_position(), cells_channel(info.channel_roles)
        fate = fate_word(self.results, int(track["ID"]))
        frames, hours, trail, fates = [], [], [], []
        for i in pick_frames(t, n):
            full = np.asarray(image_data.get_registered(int(t[i]), p, c))
            frames.append(self.display_crop(full[y0:y0 + side, x0:x0 + side]))
            hours.append(frame_to_hour(int(t[i]), info.dt_s, info.start_hour))
            trail.append((float(x[i] - x0), float(y[i] - y0)))
            fates.append(fate)
        return frames, hours, trail, fates

    def follow(self, track_id: int):
        track = self.track_by_id(track_id)
        if track is None:
            self.page.status.setText(f"Track {track_id} is not in the tracks of position {self.film_position()}.")
            return
        self.followed_track = track
        frames, hours, trail, fates = self.film_frames(track)
        self.page.set_film_frames(frames, hours, trail, fates)
        if hours and self.set_frame is not None:
            self.page.set_hour(hours[0])
            self.on_hour_changed(hours[0])
        self.open_window("Cell", int(track_id))

    # ------------------------------------------------------------------------------------------ the Seeding window
    def close_window(self):
        """New results or a new recording: the open window's film is stale, so it goes (and frees its memory)."""
        self.seeding_film = None
        w, self.window = self.window, None
        if w is not None:
            try:
                w.close()
            except RuntimeError:
                pass

    def _window_gone(self, *_args):
        self.window = None

    def label_source(self, position: int, inputs=None) -> dict:
        """Where the label frames of the measured cell rows come from: the segmentation cache of the model with the most
        frames of this position (read only for those frames), or, for imported masks or when the cache holds nothing,
        the masks channel labelled one blob per label as the Segment step does."""
        from collections import Counter
        from nd2_analyzer.analysis.segmentation.segmentation_models import SegmentationModels
        image_data = ImageData.get_instance()
        cache = getattr(image_data, "segmentation_cache", None) if image_data is not None else None
        best = None
        for model, entry in (getattr(cache, "mmap_arrays_idx", {}) or {}).items():
            try:
                array, indices = entry
            except (TypeError, ValueError):
                continue
            idx = [tuple(int(v) for v in i[:3]) for i in indices if len(i) >= 3 and int(i[1]) == int(position)]
            if idx and (best is None or len(idx) > len(best[2])):
                best = (model, array, idx)
        if best is not None and best[0] != SegmentationModels.IMPORTED_MASKS:
            model, array, idx = best
            c = Counter(i[2] for i in idx).most_common(1)[0][0]
            return {"kind": "cache", "model": model, "array": array, "channel": int(c),
                    "frames": {i[0] for i in idx if i[2] == c}, "position": int(position)}
        c = Counter(i[2] for i in best[2]).most_common(1)[0][0] if best is not None else self.masks_channel(position)
        return {"kind": "masks", "channel": int(c), "position": int(position)}

    def masks_channel(self, position: int) -> int:
        """The channel that holds masks (two values only), 0 when none does."""
        image_data = ImageData.get_instance()
        try:
            shape = image_data.data.shape
            n_c = int(shape[2]) if len(shape) == 5 else 1
            t = int(shape[0]) // 2
            for c in range(n_c):
                img = np.asarray(image_data.get_registered(t, int(position), c))[::16, ::16]
                if np.unique(img).size <= 2:
                    return c
        except Exception:
            pass
        return 0

    def label_frame(self, frame: int) -> np.ndarray:
        """The label frame the measured cell rows came from. Never segments: a cached model is read only for the frames
        it holds (an empty frame otherwise); imported masks are labelled on the fly."""
        from scipy import ndimage as ndi
        image_data = ImageData.get_instance()
        src = self.label_src or self.label_source(self.film_position())
        p = src.get("position", self.film_position())
        shape = tuple(int(v) for v in image_data.data.shape[-2:])
        if src["kind"] == "cache":
            if int(frame) not in src["frames"]:
                return np.zeros(shape, np.int32)
            return np.asarray(src["array"][int(frame), p, src["channel"]])
        return ndi.label(np.asarray(image_data.get_registered(int(frame), p, src["channel"])) > 0)[0]

    def build_film(self) -> Optional[SeedingFilm]:
        """The window's state, from the measurement itself (its tracks, frame interval, start hour and field), once."""
        if self.results is None or self.inputs is None:
            return None
        if self.seeding_film is not None and self.seeding_film.results is self.results:
            return self.seeding_film
        inp = self.inputs
        last = inp.frame_range[1] + 1 if getattr(inp, "frame_range", None) else inp.n_frames
        self.seeding_film = SeedingFilm(self.results, inp.tracks, self.label_frame, inp.field_shape, inp.dt_s, inp.start_hour,
                                        inp.pixel_um, n_frames=last)
        return self.seeding_film

    def open_window(self, view: str = "Cell", track_id=None):
        """The Seeding window on a cell (Cell) or on the whole field (Field); one window, brought forward when open."""
        from .seeding_page import cells_to_follow
        from .seeding_window import SeedingDialog
        film = self.build_film()
        if film is None:
            self.page.status.setText("Click Track and Measure before opening the Seeding window.")
            return None
        w = self.window
        try:
            alive = w is not None and w.isVisible() and w.film is film
        except RuntimeError:                                  # the window was closed and deleted
            alive, self.window = False, None
        if alive:
            if view == "Cell" and track_id is not None:
                w.follow(int(track_id))
            elif w.view_combo.currentText() != view:
                w.view_combo.setCurrentText(view)
            w.raise_(); w.activateWindow()
            return w
        self.window = SeedingDialog(film, cells_to_follow(self.results), view=view,
                                    root=int(track_id) if track_id is not None else None,
                                    open_numbers=self.page.open_numbers, parent=self.page)
        self.window.destroyed.connect(self._window_gone)
        self.window.show()
        return self.window

    # ------------------------------------------------------------------------------------------ exports
    def default_folder(self) -> str:
        if self.last_folder:
            return self.last_folder
        exp = self.experiment()
        files = getattr(exp, "image_files", None) if exp is not None else None
        if files:
            return str(Path(files[0]).parent)
        return str(Path.home())

    def ask_folder(self, title: str) -> Optional[str]:
        folder = QFileDialog.getExistingDirectory(self.page, title, self.default_folder(), QFileDialog.ShowDirsOnly)
        if folder:
            self.last_folder = folder
        return folder or None

    def export(self, kind: str):
        if kind == "film" and self.followed_track is None:
            self.page.status.setText("Pick a cell and click Follow before saving its film.")
            return
        if kind in ("image", "csv") and self.results is None:
            self.page.status.setText("Click Measure before saving.")
            return
        folder = self.ask_folder({"film": "Save the film", "image": "Save the image", "csv": "Save the tables"}.get(kind, "Save"))
        if folder is None:
            return
        if kind == "film":
            path = self.save_film(folder)
        elif kind == "image":
            path = self.save_image(folder)
        else:
            path = self.save_results(folder)
        self.page.status.setText(f"Saved {path}")

    def save_results(self, folder: str) -> str:
        """The metrics as JSON and the tables as CSV next to the experiment: seeding_metrics.json, seeding_tracks.csv,
        seeding_counts.csv (plus the candidates and the stage per frame)."""
        if self.results is None:
            return ""
        out = Path(folder); out.mkdir(parents=True, exist_ok=True)
        payload = {"recording": self.recording_words(), "settings": self.results.settings.__dict__,
                   "notes": list(self.results.notes), "metrics": metrics_as_records(self.results)}
        (out / RESULT_FILES["metrics"]).write_text(json.dumps(payload, indent=2, default=_json_default))
        self.results.per_track.write_csv(out / RESULT_FILES["tracks"])
        self.results.counts.write_csv(out / RESULT_FILES["counts"])
        self.results.candidates.write_csv(out / RESULT_FILES["candidates"])
        self.results.stage_by_frame.write_csv(out / RESULT_FILES["stages"])
        return str(out)

    def recording_words(self) -> dict:
        info = self.page.info
        if info is None:
            return {}
        return {"name": info.name, "n_frames": info.n_frames, "dt_s": info.dt_s, "pixel_um": info.pixel_um,
                "start_hour": info.start_hour, "flow": info.flow, "channel_roles": info.channel_roles,
                "field_shape": list(info.field_shape or ()), "position": self.film_position()}

    def save_image(self, folder: str) -> str:
        """The six figures as one PDF, a page each, and the two shown on the page side by side as the PNG, at print size."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.figure import Figure
        from nd2_analyzer.analysis.biofilm_stages import seeding_figures as SF
        info = self.page.info
        span = info.span if info is not None else None
        stem = Path(folder) / "seeding_figures"
        with PdfPages(str(stem.with_suffix(".pdf"))) as pdf:
            for f in SF.FIGURES:
                fig = Figure(figsize=(7.4, 5.0), dpi=100); FigureCanvasAgg(fig)
                SP.draw_panel(fig, f, self.results, span, None, self.results.settings)
                pdf.savefig(fig)
        shown = [SF.figure_by_key(c.currentData()) for c in getattr(self.page, "plot_combos", [])] or [SF.figure_by_key(k) for k in SF.FIGURE_DEFAULTS]
        fig = Figure(figsize=(7.4 * len(shown), 5.0), dpi=100); FigureCanvasAgg(fig)
        for i, f in enumerate(shown):
            SP.draw_panel(fig, f, self.results, span, None, self.results.settings, cell=(i / len(shown), 0.0, 1.0 / len(shown), 1.0))
        fig.savefig(str(stem.with_suffix(".png")), dpi=EXPORT_DPI)
        return str(stem.with_suffix(".pdf"))

    def say_film_progress(self, k: int, n: int):
        """The film is written on the window's thread: say how far it is and let the window repaint."""
        self.page.status.setText(f"Writing the film: frame {k + 1:,} of {n:,}")
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def save_film(self, folder: str, fps: int = GIF_FPS) -> str:
        """The followed cell's film as the Seeding window draws it (the Cell view), as a GIF."""
        film = self.build_film()
        track_id = int(self.followed_track["ID"])
        path = Path(folder) / f"seeding_track_{track_id}.gif"
        if film is None:
            return ""
        previous = film.root
        film.set_root(track_id)
        try:
            film.save_gif(str(path), "Cell", progress=lambda k, n: (self.say_film_progress(k, n) if k % FILM_PROGRESS_EVERY == 0 else None) is None,
                          ms=int(1000 / max(1, fps)))
        finally:
            if previous is not None and previous != track_id:
                film.set_root(previous)
        return str(path)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
