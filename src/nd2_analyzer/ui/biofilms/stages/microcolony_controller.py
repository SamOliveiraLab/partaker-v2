"""The wiring between the tool and the Microcolonies page, as the Seeding controller does for its page.

It reads what the tool already has (the loaded recording, the experiment's biofilm block, the cell masks of the
segmentation cache for the position in the view area), follows the microcolonies in a worker thread when the user clicks
Track, moves the view area's T slider when the cursor moves, opens the Microcolonies window on Follow and writes the
exports. Nothing is read from any saved table."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from PySide6.QtWidgets import QFileDialog
from pubsub import pub

from nd2_analyzer.analysis.biofilm_stages import matrix_stain as MX
from nd2_analyzer.analysis.biofilm_stages import microcolony_metric_plots as XP
from nd2_analyzer.analysis.biofilm_stages import microcolony_plots as MP
from nd2_analyzer.analysis.biofilm_stages import stage_tables as ST
from nd2_analyzer.analysis.biofilm_stages.microcolony_film import FIGURE_IN, MicrocolonyFilm, extents
from nd2_analyzer.analysis.biofilm_stages.microcolony_follow import close_gaps, follow_microcolonies, load_tracks, save_tracks
from nd2_analyzer.analysis.biofilm_stages.microcolony_metrics import load_metrics, measure_metrics, save_metrics
from nd2_analyzer.data.image_data import ImageData
from .microcolony_window import MicrocolonyDialog

WORKER_WAIT_MS = 30000
STATE_DIR = "biofilm_stages"   # in a saved session: what the stage pages worked out
MEASURE_EVERY_H = 1.5          # the step of the all-positions metrics when the page opens (the step Figure 3 used)
RANDOM_EVERY_H, RANDOM_FOR_H = 3.0, 9.0      # the cells are also placed at random every 3 h of the first 9 h: later the surface is too full


def frames_to_follow(segmented: list, last: int, skip: Optional[Callable[[int], bool]] = None) -> list:
    """The unbroken run of segmented frames from the first one, not past `last` and not into a frame `skip` names
    (out of focus): a microcolony cannot be followed through frames that are blurred."""
    run = []
    for t in sorted(int(v) for v in segmented):
        if t > int(last) or (run and t != run[-1] + 1) or (skip is not None and skip(t)):
            break
        run.append(t)
    return run


def frames_for_metrics(segmented: Dict[int, list], step: int, last: int, skip: Optional[Callable[[int], bool]] = None) -> Dict[int, list]:
    """position -> every step-th of its segmented frames counted from the first, not past `last` and without the frames
    `skip` names (out of focus). A position with fewer than two such frames gives no curve and is left out."""
    out = {}
    for p, frames in segmented.items():
        frames = sorted(int(t) for t in frames)
        keep = [t for t in frames if (t - frames[0]) % max(int(step), 1) == 0 and t <= int(last) and not (skip is not None and skip(t))]
        if len(keep) >= 2:
            out[int(p)] = keep
    return out


def random_hours(start_hour: float) -> tuple:
    """The hours at which the cells are also placed at random."""
    return tuple(float(start_hour) + k * RANDOM_EVERY_H for k in range(int(RANDOM_FOR_H / RANDOM_EVERY_H) + 1))


def recording_words(info, position: int) -> str:
    """The loaded recording in one line, the frame step in the unit a person would say."""
    dt = float(info.dt_s or 0.0)
    step = f"{dt / 3600:g} h" if dt >= 3600 and dt % 3600 == 0 else f"{dt / 60:g} min" if dt >= 60 and dt % 60 == 0 else f"{dt:g} s"
    last_h = round(float(info.start_hour) + (int(info.n_frames) - 1) * dt / 3600.0, 1)
    return (f"{info.name}, {int(info.n_frames):,} frames, {step} apart, {float(info.pixel_um or 0):.4f} µm per px, "
            f"{float(info.start_hour):g} to {last_h:g} h after inoculation, position {int(position)}")


class FollowWorker(QThread):
    """Follows the microcolonies off the window's thread and reports progress."""
    progress = Signal(int, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, labels_at, frames, pixel_um, dt_h, start_hour, parent=None):
        super().__init__(parent)
        self.args = (labels_at, list(frames), float(pixel_um), float(dt_h), float(start_hour)); self.generation = 0; self.position = 0

    def run(self):
        labels_at, frames, pixel_um, dt_h, start_hour = self.args
        try:
            tracks = follow_microcolonies(labels_at, frames, pixel_um, dt_h, start_hour,
                                          progress=lambda k, n: self.progress.emit(int(100 * k / max(n, 1)), f"{100 * k // max(n, 1)} %"))
            self.drawn = close_gaps(tracks, labels_at, pixel_um)      # for the pictures: short gaps drawn through, the record unchanged
            extents(self.drawn, labels_at)                            # where each microcolony is at each frame, so the film opens at once
        except Exception as exc:                                   # the page shows the reason in the Microcolonies step
            self.failed.emit(f"{type(exc).__name__}: {exc}"); return
        self.done.emit(tracks)


class MetricsWorker(QThread):
    """Measures the metrics of every position off the window's thread and reports progress."""
    progress = Signal(int, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, labels_at, frames, pixel_um, hour_of, random_at=(), parent=None):
        super().__init__(parent)
        self.args = (labels_at, dict(frames), float(pixel_um), hour_of, tuple(random_at)); self.generation = 0

    def run(self):
        labels_at, frames, pixel_um, hour_of, random_at = self.args
        try:
            out = measure_metrics(labels_at, frames, pixel_um, hour_of, random_hours=random_at,
                                  progress=lambda k, n: self.progress.emit(int(100 * k / max(n, 1)), f"{100 * k // max(n, 1)} %"))
        except Exception as exc:                                   # the page shows the reason in the All positions step
            self.failed.emit(f"{type(exc).__name__}: {exc}"); return
        self.done.emit(out)


class StainWorker(QThread):
    """Measures the matrix stain of every followed frame of one position off the window's thread: the matrix mask of
    each frame and the stain inside each followed microcolony. The dye channel is read through a file handle of its
    own, so the window can keep showing frames meanwhile. recipe: what the Maturation tab measured for the position
    (reference of the dye frames, free dye level, settings, dye channel, recording path, raw_at as a fallback)."""
    progress = Signal(int, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, tracks, recipe: dict, pixel_um: float, parent=None):
        super().__init__(parent)
        self.tracks, self.recipe, self.pixel_um = tracks, dict(recipe), float(pixel_um); self.generation = 0; self.position = int(recipe.get("position", 0)); self.token = None

    def run(self):
        r = self.recipe; handle = None; p = self.position
        try:
            raw = r["raw_at"]; path = r.get("path")
            if path and str(path).lower().endswith(".nd2"):
                import nd2
                handle = nd2.ND2File(str(path)); data = handle.to_dask()
                if data.ndim == 5:
                    raw = lambda position, t: np.asarray(data[t, position, r["channel"]])
            counted_at = lambda t: MX.matrix(MX.stain(MX.correct(raw(p, int(t)), r["reference"]), r["level"], r["settings"]), self.pixel_um, r["settings"])[0]
            out = MX.measure_microcolony_stain(self.tracks, counted_at, self.pixel_um,
                                               progress=lambda k, n: self.progress.emit(int(100 * k / max(n, 1)), f"{100 * k // max(n, 1)} %"))
        except Exception as exc:                                   # the page shows the reason in the Matrix stain step
            self.failed.emit(f"{type(exc).__name__}: {exc}"); return
        finally:
            if handle is not None:
                handle.close()
        self.done.emit(out)


class MicrocolonyController(QObject):
    metrics_changed = Signal()                 # the all-positions measure arrived, was loaded or was cleared
    # the tool's messages and the methods that hear them; each asks for the names the message carries (as the Seeding
    # controller does), because the message bus takes the shape of a message from whoever listens first
    LISTENS = (("experiment_loaded", "on_experiment_loaded"), ("image_data_loaded", "on_new_recording"),
               ("batch_segmentation_completed", "on_segmented"), ("view_index_changed", "on_view_index_changed"))

    def __init__(self, page, recording_info: Callable[[], object], position: Callable[[], int], parent=None):
        super().__init__(parent)
        self.page, self.recording_info, self.position = page, recording_info, position
        self.set_frame: Optional[Callable[[int], None]] = None
        self.tracked: Dict[int, object] = {}                       # position -> MicrocolonyTracks
        self.worker: Optional[FollowWorker] = None; self.window: Optional[MicrocolonyDialog] = None; self.generation = 0; self.shown_position = None
        page.track_requested.connect(self.track); page.hour_changed.connect(self.on_hour_changed)
        page.window_requested.connect(self.open_window); page.export_requested.connect(self.export); page.compare_requested.connect(self.open_compare)
        self.outlines: Dict[tuple, list] = {}                      # (position, frame) -> the outlines drawn by hand
        self.stain_fallback: Optional[Callable[[int], Optional[Callable[[int], np.ndarray]]]] = None  # position -> (frame -> matrix mask), worked out on demand (the Maturation tab)
        self.stain_recipe: Optional[Callable[[int], Optional[dict]]] = None   # position -> what the Maturation tab measured for it (see StainWorker), None before
        self.stains: Dict[int, object] = {}                        # position -> matrix_stain.MicrocolonyStain of its followed microcolonies
        self.stain_worker: Optional[StainWorker] = None; self.stain_token = 0
        self.drawn: Dict[int, object] = {}                         # position -> the followed microcolonies with their gaps closed, for the pictures
        self.compare_window = None
        self.metrics = None; self.metrics_worker: Optional[MetricsWorker] = None; self.every_set_for = None
        page.measure_requested.connect(self.measure)
        for topic, method in self.LISTENS:
            pub.subscribe(getattr(self, method), topic)
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    def shutdown(self):
        for worker in (self.worker, self.metrics_worker, self.stain_worker):
            if worker is not None and worker.isRunning():
                worker.requestInterruption(); worker.wait(WORKER_WAIT_MS)

    def connect_main_window(self, set_frame: Optional[Callable[[int], None]] = None):
        self.set_frame = set_frame; self.refresh()

    # ------------------------------------------------------------------ what the tool has
    def on_experiment_loaded(self, experiment):
        self.refresh()

    def on_view_index_changed(self, index):
        self.refresh()

    def on_segmented(self, time_range=None, positions=None, model=None):
        self.refresh()

    def on_new_recording(self, image_data):
        self.generation += 1; self.tracked = {}; self.drawn = {}; self.metrics = None; self.stains = {}; self.stain_token += 1; self.close_window()
        self.page.set_tracks(None); self.page.set_metrics(None); self.refresh(); self.metrics_changed.emit()

    def masks(self, position: int):
        """(array, frames): the mask cache of the model with the most frames of this position, and those frames."""
        image_data = ImageData.get_instance(); cache = getattr(image_data, "segmentation_cache", None) if image_data is not None else None
        best = (None, [])
        for _model, entry in (getattr(cache, "mmap_arrays_idx", {}) or {}).items():
            try:
                array, indices = entry
            except (TypeError, ValueError):
                continue
            frames = sorted({int(i[0]) for i in indices if len(i) >= 2 and int(i[1]) == int(position)})
            if len(frames) > len(best[1]):
                best = (array, frames)
        return best

    def all_masks(self):
        """(array, position -> its segmented frames) of the mask cache of the model with the most frames."""
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

    def out_of_focus(self) -> Optional[Callable[[int], bool]]:
        """The experiment's own word on which frames are out of focus (its focus-loss intervals), None when it has none."""
        from nd2_analyzer.data.appstate import ApplicationState
        state = ApplicationState.get_instance(); exp = getattr(state, "experiment", None) if state is not None else None
        if exp is None or not hasattr(exp, "is_focus_loss_frame") or not getattr(exp, "focus_loss_intervals", None):
            return None
        return lambda t: bool(exp.is_focus_loss_frame(int(t)))

    def labels_at(self, position: int) -> Callable[[int], np.ndarray]:
        array, _frames = self.masks(position)
        return lambda t: np.asarray(array[int(t), int(position), 0])

    def refresh(self):
        info = self.recording_info(); page = self.page
        if info is None:
            page.set_recording("", None); page.set_track_enabled(False); page.set_measure_enabled(False)
            for step in ("load", "segment", "track", "measure"):
                page.set_step_state(step, "todo")
            return
        p = self.position(); _array, frames = self.masks(p); dt_h = (info.dt_s or 0.0) / 3600.0
        hour = lambda t: float(info.start_hour) + t * dt_h
        page.set_recording(recording_words(info, p), (hour(0), hour(info.n_frames - 1)) if dt_h else None)
        page.set_step_state("load", "done", f"{info.n_frames:,}")
        page.set_step_state("segment", "done" if frames else "todo", f"{len(frames):,}, position {p}" if frames else "")
        can = bool(dt_h and info.pixel_um and len(frames) >= 2) and not (self.worker is not None and self.worker.isRunning())
        page.set_track_enabled(can)
        measuring = self.metrics_worker is not None and self.metrics_worker.isRunning()
        several = [f for f in self.all_masks()[1].values() if len(f) >= 2]
        page.set_measure_enabled(bool(dt_h and info.pixel_um and several) and not measuring)
        if dt_h and self.every_set_for != self.generation:
            self.every_set_for = self.generation; page.set_every(dt_h, max(dt_h, round(MEASURE_EVERY_H / dt_h) * dt_h))
        if self.metrics is None and not measuring:
            page.set_step_state("measure", "todo")
        if frames and dt_h and p != self.shown_position:
            good = frames_to_follow(frames, frames[-1], self.out_of_focus()) or frames[:1]      # up to the last frame before focus is lost
            page.set_until(hour(good[0]), hour(good[-1]), dt_h)
        tracks = self.tracked.get(p)
        if p != self.shown_position:
            self.shown_position = p; self.show_tracks(tracks, p)
            if tracks is not None:
                page.until_spin.setValue(tracks.hours[tracks.frames[-1]])       # "Follow until" shows where its tracks end
        if tracks is None and not (self.worker is not None and self.worker.isRunning()):
            page.set_step_state("track", "todo"); others = sorted(self.tracked)
            if others:
                where = (f"Position {others[0]} is tracked: move the P slider to {others[0]} to see it" if len(others) == 1 else
                         f"Positions {', '.join(str(o) for o in others)} are tracked: move the P slider to one of them to see it")
                page.status.setText(where + (", or click Track to follow this position too." if can else "."))
            else:
                page.status.setText("Click Track to find the microcolonies of this position and follow them." if can else
                                    "Segment at least two frames of this position in the Segmentation tab, then click Track.")

    def show_tracks(self, tracks, position: int):
        info = self.recording_info()
        if tracks is None or info is None:
            self.page.set_tracks(None); self.show_stain(position); return
        self.page.set_tracks(tracks, MicrocolonyFilm(self.drawn.get(position, tracks), self.labels_at(position), info.pixel_um))
        self.show_stain(position); self.maybe_measure_stain()
        n, m = len(tracks.lasting()), len(tracks.merges())
        self.page.set_step_state("track", "done", f"{n:,} followed, {MP.words(m, 'merge', 'merges')}")
        first, last = tracks.frames[0], tracks.frames[-1]
        self.page.status.setText(f"Frames {first} to {last} ({tracks.hours[first]:g} to {tracks.hours[last]:g} h), position {position}.  "
                                 f"{n:,} microcolonies followed, {MP.words(m, 'merge', 'merges')}.  Every number here is computed by the tool on this recording.")

    # ------------------------------------------------------------------ Track
    def track(self):
        info = self.recording_info()
        if info is None or (self.worker is not None and self.worker.isRunning()):
            return
        p = self.position(); _array, segmented = self.masks(p); dt_h = (info.dt_s or 0.0) / 3600.0
        if not dt_h or not info.pixel_um or len(segmented) < 2:
            return
        last = int(round((self.page.until_spin.value() - float(info.start_hour)) / dt_h))
        frames = frames_to_follow(segmented, last, self.out_of_focus())
        if len(frames) < 2:
            self.page.set_step_state("track", "failed", "fewer than two frames in a row are segmented"); return
        self.worker = FollowWorker(self.labels_at(p), frames, info.pixel_um, dt_h, float(info.start_hour), self)
        self.worker.generation, self.worker.position = self.generation, p
        self.worker.progress.connect(lambda pct, words: self.page.set_step_state("track", "running", words))
        self.worker.done.connect(self.on_tracked); self.worker.failed.connect(lambda why: (self.page.set_step_state("track", "failed", why), self.refresh()))
        self.page.set_track_enabled(False); self.page.set_step_state("track", "running", "0 %"); self.worker.start()

    def on_tracked(self, tracks):
        worker = self.sender()
        if worker is None or worker.generation != self.generation:
            return                                                  # another recording was loaded meanwhile
        self.tracked[worker.position] = tracks; self.drawn[worker.position] = getattr(worker, "drawn", None) or tracks
        self.stains.pop(worker.position, None)                      # the stain was measured on the tracks before these
        if worker.position == self.position():
            self.show_tracks(tracks, worker.position)
        self.refresh()

    # ------------------------------------------------------------------ the matrix stain of the followed microcolonies
    def stain_source(self, position: int) -> Optional[Callable[[int], np.ndarray]]:
        """frame -> the matrix mask of that frame of the position: at once when the stain of the position is measured,
        else worked out frame by frame from the Maturation tab's measure; None when there is no stain to show."""
        stain = self.stains.get(int(position))
        if stain is not None:
            return stain.counted_at
        return self.stain_fallback(int(position)) if self.stain_fallback is not None else None

    def on_stain_source_changed(self):
        """The Maturation tab measured again, loaded or lost its measure: the stain kept here is of the measure before."""
        self.stains = {}; self.stain_token += 1; self.show_stain(self.position()); self.maybe_measure_stain()

    def show_stain(self, position: int):
        """The stain of the position on the page: in the list, on the film strip, in the Matrix stain step."""
        stain = self.stains.get(int(position)); film = self.page.film
        if film is not None:
            if stain is not None:
                film.set_stain(stain.counted_at, rows=stain.rows, show=True)
            else:
                film.set_stain(None)
        self.page.set_stain(stain.summary if stain is not None else None)
        measuring = self.stain_worker is not None and self.stain_worker.isRunning()
        if stain is not None:
            tracks = self.tracked.get(int(position)); keep = tracks.lasting() if tracks is not None else set(stain.summary)
            stained = sum(1 for i in keep if stain.summary.get(i, {}).get("first_stain_h") is not None)
            self.page.set_step_state("stain", "done", f"{stained} of {len(keep)} microcolonies"); self.page.step_labels["stain"].setToolTip(
                "The matrix stain inside each followed microcolony of this position: see the list, the film strip and the Follow window")
        elif not measuring:
            self.page.set_step_state("stain", "todo"); self.page.step_labels["stain"].setToolTip(XP.STAIN_FIRST)

    def maybe_measure_stain(self):
        """Measures the matrix stain of the shown position by itself, as soon as it is tracked and the Maturation tab has
        the dye frames and its measure. Nothing to click."""
        p = int(self.position()); tracks = self.drawn.get(p) or self.tracked.get(p); info = self.recording_info()
        if tracks is None or info is None or p in self.stains or (self.stain_worker is not None and self.stain_worker.isRunning()):
            return
        recipe = self.stain_recipe(p) if self.stain_recipe is not None else None
        if recipe is None:
            return
        self.stain_worker = StainWorker(tracks, recipe, info.pixel_um, self)
        self.stain_worker.generation, self.stain_worker.token = self.generation, self.stain_token
        self.stain_worker.progress.connect(lambda pct, words: self.page.set_step_state("stain", "running", words))
        self.stain_worker.done.connect(self.on_stain_measured)
        self.stain_worker.failed.connect(lambda why: self.page.set_step_state("stain", "failed", why))
        self.page.set_step_state("stain", "running", "0 %"); self.stain_worker.start()

    def on_stain_measured(self, stain):
        worker = self.sender() or self.stain_worker
        if worker is None or worker.generation != self.generation or worker.token != self.stain_token:
            return self.maybe_measure_stain()                       # another recording, or another measure of the dye, meanwhile
        p = worker.position
        if (self.drawn.get(p) or self.tracked.get(p)) is not worker.tracks:
            return self.maybe_measure_stain()                       # tracked again meanwhile
        self.stains[p] = stain
        if p == self.position():
            self.show_stain(p)
        else:
            self.maybe_measure_stain()

    # ------------------------------------------------------------------ Measure (every position, for the metric plots)
    def measure(self):
        info = self.recording_info()
        if info is None or (self.metrics_worker is not None and self.metrics_worker.isRunning()):
            return
        array, per = self.all_masks(); dt_h = (info.dt_s or 0.0) / 3600.0
        if array is None or not dt_h or not info.pixel_um:
            return
        start = float(info.start_hour); last = int(round((self.page.until_spin.value() - start) / dt_h))
        frames = frames_for_metrics(per, max(1, int(round(self.page.every_spin.value() / dt_h))), last, self.out_of_focus())
        if not frames:
            self.page.set_step_state("measure", "failed", "no position has two segmented frames to measure"); return
        self.metrics_worker = MetricsWorker(lambda p, t: np.asarray(array[int(t), int(p), 0]), frames, info.pixel_um,
                                            lambda t: start + int(t) * dt_h, random_hours(start), self)
        self.metrics_worker.generation = self.generation
        self.metrics_worker.progress.connect(lambda pct, words: self.page.set_step_state("measure", "running", words))
        self.metrics_worker.done.connect(self.on_measured)
        self.metrics_worker.failed.connect(lambda why: (self.page.set_step_state("measure", "failed", why), self.refresh()))
        self.page.set_measure_enabled(False); self.page.set_step_state("measure", "running", "0 %"); self.metrics_worker.start()

    def on_measured(self, metrics):
        worker = self.sender()
        if worker is None or worker.generation != self.generation:
            return                                                  # another recording was loaded meanwhile
        self.metrics = metrics; self.show_metrics(metrics); self.refresh()

    def show_metrics(self, metrics):
        self.page.set_metrics(metrics); self.metrics_changed.emit()
        if metrics is None:
            return
        hours = [h for p in metrics.positions for h in metrics.hours[p]]; n = len(metrics.positions)
        self.page.set_step_state("measure", "done", f"{n} position{'' if n == 1 else 's'}, {min(hours):g} to {max(hours):g} h")
        self.page.step_labels["measure"].setToolTip("Positions " + ", ".join(str(p) for p in metrics.positions))

    def save_numbers(self):
        """The numbers behind the plots as CSV files, into a folder the user chooses: the followed microcolonies of every
        tracked position and the metrics of every measured position."""
        tables = {}
        for p, tracks in sorted(self.tracked.items()):
            tables.update(ST.microcolony_tables(tracks, p))
            if p in self.stains:
                tables.update(ST.stain_tables(self.stains[p], p))
        if self.metrics is not None:
            tables.update(ST.metric_tables(self.metrics))
        if not tables:
            return
        folder = QFileDialog.getExistingDirectory(self.page, "Folder for the numbers (CSV files)")
        if folder:
            written = ST.write_tables(folder, tables); self.page.status.setText(f"Numbers saved: {len(written)} CSV files in {folder}")

    # ------------------------------------------------------------------ with the saved session
    def save_state(self, folder) -> list:
        """Writes what the page worked out into the session folder (the followed microcolonies of every tracked position,
        the all-positions metrics, the outlines drawn by hand). Returns the files written; none when there is nothing."""
        out = Path(folder) / STATE_DIR; written = []
        for old in list(out.glob("microcolonies_p*.npz")) + [out / "microcolony_metrics.npz", out / "microcolonies.json"]:
            old.unlink(missing_ok=True)             # of an earlier save to this folder: a later Load must not show them on other masks
        if not self.tracked and self.metrics is None and not self.outlines:
            return []
        out.mkdir(parents=True, exist_ok=True)
        for p, tracks in sorted(self.tracked.items()):
            written.append(save_tracks(tracks, out / f"microcolonies_p{int(p)}.npz"))
            if self.drawn.get(p) is not None:
                written.append(save_tracks(self.drawn[p], out / f"microcolonies_p{int(p)}_drawn.npz"))
        if self.metrics is not None:
            written.append(save_metrics(self.metrics, out / "microcolony_metrics.npz"))
        about = {"positions": sorted(int(p) for p in self.tracked), "every_h": self.page.every_spin.value(),
                 "outlines": [{"position": int(p), "frame": int(t), "outlines": lines} for (p, t), lines in sorted(self.outlines.items())]}
        (out / "microcolonies.json").write_text(json.dumps(about, indent=1)); written.append(str(out / "microcolonies.json"))
        return written

    def load_state(self, folder) -> bool:
        """Brings back what save_state wrote. False when the session holds none of it or it cannot be read (the page
        then starts empty and says so)."""
        src = Path(folder) / STATE_DIR
        if not (src / "microcolonies.json").exists():
            return False
        try:
            about = json.loads((src / "microcolonies.json").read_text())
            tracked = {int(p): load_tracks(src / f"microcolonies_p{int(p)}.npz") for p in about.get("positions", [])}
            drawn = {p: load_tracks(src / f"microcolonies_p{p}_drawn.npz") if (src / f"microcolonies_p{p}_drawn.npz").exists() else t for p, t in tracked.items()}
            metrics = load_metrics(src / "microcolony_metrics.npz") if (src / "microcolony_metrics.npz").exists() else None
            outlines = {(int(o["position"]), int(o["frame"])): [[tuple(float(v) for v in point) for point in line] for line in o["outlines"]]
                        for o in about.get("outlines", [])}
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.page.status.setText(f"The microcolony results saved with this session could not be read ({exc}). Click Track to follow them again.")
            return False
        self.close_window(); self.tracked, self.drawn, self.metrics, self.outlines = tracked, drawn, metrics, outlines
        self.shown_position = None; self.refresh()                      # shows the tracks of the position in the view area
        info = self.recording_info(); dt_h = (info.dt_s or 0.0) / 3600.0 if info is not None else 0.0
        if dt_h and about.get("every_h"):
            self.page.set_every(dt_h, float(about["every_h"]))
        self.show_metrics(metrics)
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

    def open_window(self, identity):
        p = self.position(); tracks = self.tracked.get(p); info = self.recording_info()
        if tracks is None or info is None:
            return
        self.close_window()
        film = MicrocolonyFilm(self.drawn.get(p, tracks), self.labels_at(p), info.pixel_um)     # its own film: the page's strip keeps its place
        stain = self.stains.get(p)
        if stain is not None:
            film.set_stain(stain.counted_at, rows=stain.rows, show=True)     # measured already: the box is ticked and the stain is there at once
        else:
            film.set_stain(self.stain_source(p))
        self.window = MicrocolonyDialog(film, MP.entries(tracks), root=identity, parent=self.page.window())
        self.window.destroyed.connect(lambda *_a: setattr(self, "window", None)); self.window.show()

    def open_compare(self):
        """Manual selection next to automatic detection, on the segmented frame nearest the cursor."""
        from .microcolony_compare_window import MicrocolonyCompareDialog
        info = self.recording_info(); p = self.position(); array, frames = self.masks(p)
        if info is None or not frames or not info.pixel_um:
            return
        dt_h = (info.dt_s or 0.0) / 3600.0; hour = lambda t: float(info.start_hour) + t * dt_h
        t = min(frames, key=lambda f: abs(hour(f) - self.page.hour))
        if self.compare_window is not None:
            try:
                self.compare_window.close()
            except RuntimeError:
                pass
        key = (int(p), int(t))
        self.compare_window = MicrocolonyCompareDialog(np.asarray(array[t, p, 0]), info.pixel_um, f"position {p}   frame {t}   {hour(t):g} h",
                                                       outlines=self.outlines.get(key), on_change=lambda o, k=key: self.outlines.__setitem__(k, o),
                                                       parent=self.page.window())
        self.compare_window.destroyed.connect(lambda *_a: setattr(self, "compare_window", None)); self.compare_window.show()

    def export(self, kind: str):
        if kind == "numbers":
            return self.save_numbers()
        film = self.page.film
        if film is None or film.root is None or not film.frames:
            return
        if kind == "film":
            path, _ = QFileDialog.getSaveFileName(self.page, "Save film", f"microcolony_{film.root}.gif", "GIF (*.gif)")
            if path and film.save_gif(path):
                self.page.status.setText(f"Film saved: {path}")
        else:
            path, _ = QFileDialog.getSaveFileName(self.page, "Save image", f"microcolony_{film.root}.png", "PNG (*.png);;PDF (*.pdf)")
            if path:
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                from matplotlib.figure import Figure
                tracks = film.tracks; t = min(film.frames, key=lambda f: abs(tracks.hours[f] - self.page.hour))
                fig = Figure(figsize=FIGURE_IN, dpi=100); FigureCanvasAgg(fig); film.draw(fig, t, finest=True); fig.savefig(path, dpi=300, facecolor="white")
                self.page.status.setText(f"Image saved: {path}")
