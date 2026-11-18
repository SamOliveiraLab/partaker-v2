"""The wiring between the tool and the Microcolonies page, as the Seeding controller does for its page.

It reads what the tool already has (the loaded recording, the experiment's biofilm block, the cell masks of the
segmentation cache for the position in the view area), follows the microcolonies in a worker thread when the user clicks
Track, moves the view area's T slider when the cursor moves, opens the Microcolonies window on Follow and writes the
exports. Nothing is read from any saved table."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from PySide6.QtWidgets import QFileDialog
from pubsub import pub

from nd2_analyzer.analysis.biofilm_stages import microcolony_plots as MP
from nd2_analyzer.analysis.biofilm_stages.microcolony_film import FIGURE_IN, MicrocolonyFilm
from nd2_analyzer.analysis.biofilm_stages.microcolony_follow import close_gaps, follow_microcolonies
from nd2_analyzer.data.image_data import ImageData
from .microcolony_window import MicrocolonyDialog

WORKER_WAIT_MS = 30000


def frames_to_follow(segmented: list, last: int, skip: Optional[Callable[[int], bool]] = None) -> list:
    """The unbroken run of segmented frames from the first one, not past `last` and not into a frame `skip` names
    (out of focus): a microcolony cannot be followed through frames that are blurred."""
    run = []
    for t in sorted(int(v) for v in segmented):
        if t > int(last) or (run and t != run[-1] + 1) or (skip is not None and skip(t)):
            break
        run.append(t)
    return run


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
        except Exception as exc:                                   # the page shows the reason in the Microcolonies step
            self.failed.emit(f"{type(exc).__name__}: {exc}"); return
        self.done.emit(tracks)


class MicrocolonyController(QObject):
    def __init__(self, page, recording_info: Callable[[], object], position: Callable[[], int], parent=None):
        super().__init__(parent)
        self.page, self.recording_info, self.position = page, recording_info, position
        self.set_frame: Optional[Callable[[int], None]] = None
        self.tracked: Dict[int, object] = {}                       # position -> MicrocolonyTracks
        self.worker: Optional[FollowWorker] = None; self.window: Optional[MicrocolonyDialog] = None; self.generation = 0; self.shown_position = None
        page.track_requested.connect(self.track); page.hour_changed.connect(self.on_hour_changed)
        page.window_requested.connect(self.open_window); page.export_requested.connect(self.export); page.compare_requested.connect(self.open_compare)
        self.outlines: Dict[tuple, list] = {}                      # (position, frame) -> the outlines drawn by hand
        self.stain_source: Optional[Callable[[int], Optional[Callable[[int], np.ndarray]]]] = None   # position -> (frame -> matrix mask), from the Maturation tab
        self.drawn: Dict[int, object] = {}                         # position -> the followed microcolonies with their gaps closed, for the pictures
        self.compare_window = None
        for topic, fn in (("experiment_loaded", self.on_changed), ("image_data_loaded", self.on_new_recording),
                          ("batch_segmentation_completed", self.on_segmented), ("view_index_changed", self.on_changed)):
            pub.subscribe(fn, topic)
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    def shutdown(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption(); self.worker.wait(WORKER_WAIT_MS)

    def connect_main_window(self, set_frame: Optional[Callable[[int], None]] = None):
        self.set_frame = set_frame; self.refresh()

    # ------------------------------------------------------------------ what the tool has
    def on_changed(self, *_args, **_kw):
        self.refresh()

    def on_segmented(self, time_range=None, positions=None, model=None):
        self.refresh()

    def on_new_recording(self, image_data=None):
        self.generation += 1; self.tracked = {}; self.drawn = {}; self.close_window(); self.page.set_tracks(None); self.refresh()

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
            page.set_recording("", None); page.set_track_enabled(False)
            for step in ("load", "segment", "track"):
                page.set_step_state(step, "todo")
            return
        p = self.position(); _array, frames = self.masks(p); dt_h = (info.dt_s or 0.0) / 3600.0
        hour = lambda t: float(info.start_hour) + t * dt_h
        page.set_recording(recording_words(info, p), (hour(0), hour(info.n_frames - 1)) if dt_h else None)
        page.set_step_state("load", "done", f"{info.n_frames:,}")
        page.set_step_state("segment", "done" if frames else "todo", f"{len(frames):,}, position {p}" if frames else "")
        can = bool(dt_h and info.pixel_um and len(frames) >= 2) and not (self.worker is not None and self.worker.isRunning())
        page.set_track_enabled(can)
        if frames and dt_h and p != self.shown_position:
            good = frames_to_follow(frames, frames[-1], self.out_of_focus()) or frames[:1]      # up to the last frame before focus is lost
            page.set_until(hour(good[0]), hour(good[-1]), dt_h)
        tracks = self.tracked.get(p)
        if p != self.shown_position:
            self.shown_position = p; self.show_tracks(tracks, p)
        if tracks is None and not (self.worker is not None and self.worker.isRunning()):
            page.set_step_state("track", "todo")
            page.status.setText("Click Track to find the microcolonies of this position and follow them." if can else
                                "Segment at least two frames of this position in the Segmentation tab, then click Track.")

    def show_tracks(self, tracks, position: int):
        info = self.recording_info()
        if tracks is None or info is None:
            self.page.set_tracks(None); return
        self.page.set_tracks(tracks, MicrocolonyFilm(self.drawn.get(position, tracks), self.labels_at(position), info.pixel_um))
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
        if worker.position == self.position():
            self.show_tracks(tracks, worker.position)
        self.refresh()

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
        film.set_stain(self.stain_source(p) if self.stain_source is not None else None)
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
                fig = Figure(figsize=FIGURE_IN, dpi=100); FigureCanvasAgg(fig); film.draw(fig, t); fig.savefig(path, dpi=300, facecolor="white")
                self.page.status.setText(f"Image saved: {path}")
