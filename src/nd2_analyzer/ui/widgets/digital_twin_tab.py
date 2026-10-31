"""The Digital Twin tab: do the rules measured on this recording make this colony?

Seeds the twin (Viva-munk) from the real cells of the first frame inside the ROI, measures the adder and the position rule
from the tool's own tracks and masks, runs a few seeds in the Viva-munk environment, and shows the real cells beside the
twin's cells at the same frame with the count and area curves. Data and drawing live in analysis.twin; this file is the Qt.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSlider, QSpinBox, QTextEdit,
                               QVBoxLayout, QWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DT_OUTPUT_DIR = str(PROJECT_ROOT / "digital_twin_output")


class TwinWorker(QThread):
    """Seed, measure, run the seeds, read the runs back. Runs off the window's thread."""
    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, tracks, masks_at, roi_mask, pixel_um, frame_s, n_frames, rule, n_seeds, out_dir):
        super().__init__()
        self.tracks, self.masks_at, self.roi_mask = tracks, masks_at, roi_mask
        self.pixel_um, self.frame_s, self.n_frames = pixel_um, frame_s, n_frames
        self.rule, self.n_seeds, self.out_dir = rule, n_seeds, out_dir
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            from nd2_analyzer.analysis.twin import measure as M, results as R, runner as RUN, seed as S
            missing = RUN.viva_available()
            if missing:
                raise RuntimeError(missing)
            self.progress.emit("Seeding the twin from the first frame...")
            seed_df = S.frame_seed(self.masks_at(0), self.roi_mask, self.pixel_um)
            if len(seed_df) < 5:
                raise RuntimeError(f"only {len(seed_df)} cells inside the ROI at frame 0")
            chamber = S.chamber(self.roi_mask, self.pixel_um, self.frame_s)
            width = float(seed_df.width_um.median())
            self.progress.emit(f"{len(seed_df)} seed cells, median width {width:.2f} um. Measuring the adder from the full generations...")
            adder = M.measure_adder(self.tracks, self.masks_at, self.pixel_um, self.frame_s / 60.0, width)
            pos = M.measure_position_slope(self.tracks, self.masks_at, self.roi_mask, self.pixel_um, self.frame_s / 60.0)
            adder.update(pos_slope=pos["pos_slope"], **M.LAG)
            self.progress.emit(f"adder: birth {adder['birth_total_um']:.2f} um, division {adder['division_total_um']:.2f} um, added {adder['added_um']:.2f} um, "
                               f"alpha {adder['alpha_per_h']:.3f}/h from {adder['n_full_generations']} generations ({adder['rule']}); "
                               f"position slope {pos['pos_slope']:.4f}/um ({'measured' if pos.get('measured') else 'default'})")
            paths = RUN.prepare_run(self.out_dir, seed_df, adder, chamber)
            sim_time = (self.n_frames - 1) * self.frame_s
            pkls = []
            for s in range(1, self.n_seeds + 1):
                if self._cancel:
                    raise RuntimeError("cancelled")
                self.progress.emit(f"run {s} of {self.n_seeds}, rule set '{self.rule}', {sim_time / 60:.0f} simulated minutes...")
                pkls.append(RUN.run_seed(paths, self.rule, s, self.out_dir, sim_time, log=self.progress.emit, cancelled=lambda: self._cancel))
            runs = [R.load_run(p) for p in pkls]
            counts, areas = R.curves(runs, self.n_frames, self.frame_s)
            self.finished.emit(dict(folder=self.out_dir, rule=self.rule, adder=adder, position=pos, chamber=chamber, n_seed_cells=len(seed_df),
                                    runs=runs, counts=counts, areas=areas))
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()[-800:]}")


class DigitalTwinTab(QWidget):
    """Digital Twin analysis tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lineage_tracks = None
        self._metrics_service = None
        self._results = None
        self._worker = None
        self._masks = {}
        self._real = None
        self.t = 0
        self._init_ui()

    # ------------------------------------------------------------------ ui
    def _init_ui(self):
        layout = QVBoxLayout(self)
        rules = QHBoxLayout()
        rules.addWidget(QLabel("Rules:"))
        self.position_check = QCheckBox("position (growth falls away from the opening)"); self.position_check.setChecked(True)
        self.lag_check = QCheckBox("slow start (first 75 min)"); self.lag_check.setChecked(True)
        self.crowding_check = QCheckBox("crowding")
        for w in (self.position_check, self.lag_check, self.crowding_check):
            rules.addWidget(w)
        rules.addSpacing(16)
        repeats_label = QLabel("Repeats:")
        repeats_label.setToolTip("How many times the twin is run, each from a different random start. The curves show their mean and spread.")
        rules.addWidget(repeats_label)
        self.seeds_spin = QSpinBox(); self.seeds_spin.setRange(1, 12); self.seeds_spin.setValue(4)
        self.seeds_spin.setToolTip(repeats_label.toolTip())
        rules.addWidget(self.seeds_spin)
        rules.addSpacing(16)
        self.run_button = QPushButton("Run Twin")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._run_pipeline)
        rules.addWidget(self.run_button)
        self.cancel_button = QPushButton("Cancel"); self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        rules.addWidget(self.cancel_button)
        rules.addStretch()
        layout.addLayout(rules)

        self.status_label = QLabel("Load tracking data and run tracking first.")
        layout.addWidget(self.status_label)

        self.figure = Figure(figsize=(10, 4.6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 1)

        time_row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal); self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider)
        time_row.addWidget(self.slider, 1)
        self.time_label = QLabel(""); self.time_label.setMinimumWidth(140)
        time_row.addWidget(self.time_label)
        layout.addLayout(time_row)

        self.rules_label = QLabel("")
        self.rules_label.setWordWrap(True)
        layout.addWidget(self.rules_label)

        bottom = QHBoxLayout()
        self.open_folder_button = QPushButton("Open Output Folder"); self.open_folder_button.setEnabled(False)
        self.open_folder_button.clicked.connect(self._open_output_folder)
        bottom.addWidget(self.open_folder_button)
        self.save_button = QPushButton("Save Image"); self.save_button.clicked.connect(self._save_image)
        bottom.addWidget(self.save_button)
        bottom.addStretch()
        self.console_button = QPushButton("Console"); self.console_button.setCheckable(True)
        self.console_button.setToolTip("Show what the twin run prints")
        bottom.addWidget(self.console_button)
        layout.addLayout(bottom)

        self.log_text = QTextEdit(); self.log_text.setReadOnly(True); self.log_text.setMaximumHeight(90)
        self.log_text.setVisible(False)
        self.console_button.toggled.connect(self.log_text.setVisible)
        layout.addWidget(self.log_text)
        self.setLayout(layout)

    # ------------------------------------------------------------------ data
    def _reading_data(self):
        from nd2_analyzer.data.appstate import ApplicationState
        from nd2_analyzer.data.image_data import ImageData
        self.image_data = ImageData.get_instance()
        st = ApplicationState.get_instance()
        self.position = int(self._lineage_tracks[0].get("position", 0)) if self._lineage_tracks else 0
        self.roi_mask = getattr(st, "roi_mask", None) if st is not None else None
        shape = self.image_data.data.shape if self.image_data is not None else (1, 1, 1, 1, 1)
        self.n_frames = int(shape[0]); self.shape = (int(shape[-2]), int(shape[-1]))
        exp = getattr(st, "experiment", None) if st is not None else None
        interval_h = None
        if exp is not None:
            interval_h = getattr(exp, "time_interval_hours", None) or (float(getattr(exp, "phc_interval", 0) or 0) / 3600 or None)
        self.frame_s = float((interval_h or 5 / 60) * 3600)
        self.hours = np.arange(self.n_frames) * self.frame_s / 3600
        self.pixel_um = (self.image_data.pixel_size_um() if self.image_data is not None and hasattr(self.image_data, "pixel_size_um") else None) or 0.065
        self._masks = {}; self._real = None

    def masks_at(self, t: int) -> np.ndarray:
        t = int(t)
        if t not in self._masks:
            if self.image_data is None:
                self._masks[t] = np.zeros(self.shape, np.int32)
            else:
                m = np.asarray(self.image_data.segmentation_cache[t, self.position, 0])
                if self.roi_mask is not None and self.roi_mask.shape == m.shape:
                    from nd2_analyzer.analysis.segmentation.segmentation_service import apply_roi_mask
                    m = apply_roi_mask(m, self.roi_mask)
                self._masks[t] = m
        return self._masks[t]

    def _box(self):
        from nd2_analyzer.analysis.twin.seed import roi_box
        if self.roi_mask is None:
            return dict(row0=0, row1=self.shape[0], col0=0, col1=self.shape[1])
        return roi_box(self.roi_mask)

    def _real_curves(self):
        if self._real is None:
            counts, areas = [], []
            for t in range(self.n_frames):
                m = self.masks_at(t)
                counts.append(len(np.unique(m)) - 1); areas.append(float((m > 0).sum() * self.pixel_um ** 2))
            self._real = (np.array(counts, float), np.array(areas, float))
        return self._real

    # ------------------------------------------------------------------ public API (TrackingWidget)
    def set_tracking_data(self, lineage_tracks, metrics_service):
        """Receive tracking data from the parent TrackingWidget."""
        self._lineage_tracks = lineage_tracks
        self._metrics_service = metrics_service
        self._results = None
        has_data = lineage_tracks is not None and len(lineage_tracks) > 0
        self.run_button.setEnabled(has_data)
        if not has_data:
            self.status_label.setText("No tracking data available yet.")
            return
        self._reading_data()
        self.slider.blockSignals(True); self.slider.setRange(0, max(self.n_frames - 1, 0)); self.slider.setValue(0); self.slider.blockSignals(False)
        self.t = 0
        roi = "no ROI: the whole frame" if self.roi_mask is None else "inside the ROI"
        self.status_label.setText(f"Ready: {len(lineage_tracks)} tracks, {self.n_frames} frames, {roi}. Choose the rules and press Run Twin.")
        self.redraw()

    # ------------------------------------------------------------------ run
    def _run_pipeline(self):
        if not self._lineage_tracks:
            QMessageBox.warning(self, "No Data", "Run cell tracking first.")
            return
        from nd2_analyzer.analysis.twin import runner as RUN
        if self.roi_mask is None:
            QMessageBox.warning(self, "No ROI", "Draw the chamber ROI first (Image > ROI): the twin is seeded inside it.")
            return
        rule = RUN.rule_set(self.position_check.isChecked(), self.lag_check.isChecked(), self.crowding_check.isChecked())
        out_dir = str(Path(DT_OUTPUT_DIR) / f"twin_{time.strftime('%Y%m%d_%H%M%S')}_{rule}")
        self.run_button.setEnabled(False); self.cancel_button.setEnabled(True)
        self.log_text.clear(); self._log(f"run folder {out_dir}")
        self._worker = TwinWorker(self._lineage_tracks, self.masks_at, self.roi_mask, self.pixel_um, self.frame_s, self.n_frames, rule,
                                  self.seeds_spin.value(), out_dir)
        self._worker.progress.connect(self._log)
        self._worker.finished.connect(self._on_pipeline_done)
        self._worker.error.connect(self._on_pipeline_error)
        self.status_label.setText("Twin running...")
        self._worker.start()

    def _cancel(self):
        if self._worker is not None:
            self._worker.cancel(); self._log("cancelling after the current run...")

    def _on_pipeline_done(self, results: dict):
        self._results = results
        self.run_button.setEnabled(True); self.cancel_button.setEnabled(False); self.open_folder_button.setEnabled(True)
        counts = results["counts"]; rc, _ = self._real_curves()
        end = counts[:, -1]
        self.status_label.setText(f"Twin done: {len(counts)} runs, rule set '{results['rule']}'. At the end: twin {end.mean():.0f} +/- {end.std(ddof=1) if len(end) > 1 else 0:.0f} cells, real {rc[-1]:.0f}.")
        self._log("done"); self.redraw()

    def _on_pipeline_error(self, msg: str):
        self.run_button.setEnabled(True); self.cancel_button.setEnabled(False)
        self.status_label.setText("Twin failed, see the log.")
        self._log("ERROR: " + msg)

    # ------------------------------------------------------------------ drawing
    def _on_slider(self, v):
        self.t = int(v); self.redraw()

    def _rules_text(self) -> str:
        if self._results is None:
            on = [n for n, w in (("position", self.position_check), ("slow start", self.lag_check), ("crowding", self.crowding_check)) if w.isChecked()]
            return "adder growth" + ("".join(f" + {n}" for n in on)) + ".   Seeded from frame 0 inside the ROI; rules measured from the tracks when you press Run."
        r = self._results; a = r["adder"]; p = r["position"]
        parts = [f"adder: birth {a['birth_total_um']:.2f} um, added {a['added_um']:.2f} um, alpha {a['alpha_per_h']:.2f}/h ({a['n_full_generations']} generations)"]
        if r["rule"] in ("position", "lag", "crowding"):
            parts.append(f"position slope {p['pos_slope']:.4f}/um")
        if r["rule"] in ("lag", "crowding"):
            parts.append("slow start 75 min")
        if r["rule"] == "crowding":
            parts.append("crowding")
        return "   |   ".join(parts) + f"   |   {r['n_seed_cells']} seed cells, {len(r['counts'])} runs"

    def redraw(self):
        if self._lineage_tracks is None or not hasattr(self, "n_frames"):
            return
        from nd2_analyzer.analysis.twin import results as R
        from nd2_analyzer.analysis.twin.draw import draw_twin
        rc, ra = self._real_curves(); b = self._box()
        mask_roi = self.masks_at(self.t)[b["row0"]:b["row1"], b["col0"]:b["col1"]]
        w_um = (b["col1"] - b["col0"]) * self.pixel_um; h_um = (b["row1"] - b["row0"]) * self.pixel_um
        sim_cells = sim_counts = sim_areas = None
        if self._results is not None:
            sim_cells = R.cells_at(self._results["runs"][0], self.t, self.frame_s)
            sim_counts, sim_areas = self._results["counts"], self._results["areas"]
        draw_twin(self.figure, self.t, self.hours, mask_roi, sim_cells, w_um, h_um, rc, ra, sim_counts, sim_areas, message="press Run Twin")
        self.rules_label.setText(self._rules_text())
        self.time_label.setText(f"frame {self.t}   {self.hours[self.t]:.2f} h")
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ misc
    def _open_output_folder(self):
        import subprocess
        folder = self._results["folder"] if self._results else DT_OUTPUT_DIR
        subprocess.Popen(["open", folder])

    def _save_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "twin.png", "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); self._log(f"saved {path}")

    def _log(self, msg: str):
        self.log_text.append(msg)
