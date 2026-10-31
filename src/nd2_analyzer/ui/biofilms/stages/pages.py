"""The four stage pages. Each page: the stage strip on top, the page's picture under it, a numbers line, and an import slot
where the user brings in data the tool does not have yet (literature, their own tables). Every page has set_study(study) and
set_hour(h); the tab keeps one cursor for all four."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSlider, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget, QHeaderView, QAbstractItemView)

from nd2_analyzer.analysis.biofilm_stages import plots as P
from .widgets import PlotPanel, StageStrip, StatusRow


class StagePage(QWidget):
    """The frame every stage page shares."""
    name = "stage"
    question = ""
    hour_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.study = None; self.hour = 24.0; self.position = 2
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 4, 6, 4); lay.setSpacing(4)
        head = QLabel(f"<b>{self.name}</b>   {self.question}"); lay.addWidget(head)
        self.strip = StageStrip(); self.strip.hour_changed.connect(self._on_strip); lay.addWidget(self.strip)
        self.controls = QHBoxLayout(); lay.addLayout(self.controls)
        self.build_controls()
        self.body = QVBoxLayout(); lay.addLayout(self.body, 1)
        self.build_body()
        self.status = StatusRow(); lay.addWidget(self.status)
        self.setLayout(lay)

    # hooks
    def build_controls(self): pass
    def build_body(self):
        self.panel = PlotPanel(); self.body.addWidget(self.panel, 1)
    def render(self): pass

    # state
    def set_study(self, study):
        self.study = study
        self.strip.set_stage_hours(study.stage_hours() if study else None, highlight=self.name.lower())
        if study:
            sh = study.stage_hours(); self.hour = max(min(self.hour, sh.last_h), sh.first_h); self.strip.set_hour(self.hour)
        self.render()

    def set_hour(self, h: float):
        self.hour = float(h); self.strip.set_hour(h); self.render()

    def _on_strip(self, h):
        self.hour = float(h); self.render(); self.hour_changed.emit(self.hour)

    def frame(self):
        if self.study is None or not self.study.available()["masks"]: return None
        try:
            return self.study.frame(self.position, self.hour)
        except Exception as exc:
            self.status.say(f"frame not available: {exc}"); return None

    def position_combo(self):
        self.controls.addWidget(QLabel("Position:")); cb = QComboBox(); cb.setMinimumWidth(110)
        cb.currentIndexChanged.connect(lambda _i: self._set_position(cb.currentData())); self.controls.addWidget(cb); return cb

    def fill_positions(self, cb):
        cb.blockSignals(True); cb.clear()
        for p in (self.study.kept() if self.study else []): cb.addItem(f"Position {p}", p)
        i = cb.findData(self.position); cb.setCurrentIndex(i if i >= 0 else 0); cb.blockSignals(False)
        if cb.currentData() is not None: self.position = int(cb.currentData())

    def _set_position(self, p):
        if p is not None: self.position = int(p); self.render()


class SeedingPage(StagePage):
    name = "Seeding"
    question = "cells land, most leave, some stay and divide: measured on the published recordings until our own attachment run exists."

    def build_body(self):
        self.panel = PlotPanel(size=(12.5, 5.0)); self.body.addWidget(self.panel, 2)
        self.table = QTableWidget(0, 5); self.table.setHorizontalHeaderLabels(["Hallmark", "Shown by", "What Partaker finds", "Status", ""])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch); self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers); self.table.verticalHeader().setVisible(False); self.table.setMinimumHeight(220)
        self.body.addWidget(self.table, 1)

    def build_controls(self):
        self.import_button = QPushButton("Import literature table..."); self.import_button.setToolTip("A JSON like studies/hallmarks.json: the hallmarks with their status, and the published numbers to set against the tool's")
        self.import_button.clicked.connect(self._import); self.controls.addWidget(self.import_button); self.controls.addStretch()

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Literature table", "", "JSON (*.json)")
        if path and self.study is not None:
            self.study.literature_path = path; self.render()

    def render(self):
        if self.study is None: return
        self.panel.render(P.draw_seeding, self.study, self.hour)
        lit = self.study.literature(); rows = lit.get("hallmarks", [])
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, key in enumerate(("hallmark", "shown_by", "partaker", "status")):
                it = QTableWidgetItem(str(row.get(key, ""))); self.table.setItem(r, c, it)
            self.table.setItem(r, 4, QTableWidgetItem(""))
        self.table.resizeRowsToContents()
        src = lit.get("source", "no literature file: import one")
        self.status.say(f"Source of the published numbers: {src}.   Our own attachment run (E. coli and P. aeruginosa from inoculation) is still to be recorded; its measurements will replace the literature here.")


class MicrocoloniesPage(StagePage):
    name = "Microcolonies"
    question = "a microcolony is at least 4 cells within 1 um of each other. How many, how big, when do they merge?"

    def build_controls(self):
        self.pos = self.position_combo(); self.controls.addStretch()

    def set_study(self, study):
        super().set_study(study); self.fill_positions(self.pos)

    def render(self):
        if self.study is None: return
        fr = self.frame(); self.panel.render(P.draw_microcolonies, self.study, self.hour, fr, self.position)
        sh = self.study.stage_hours(); dyn = self.study.dynamics(); npf = self.study.colonies_per_field()
        peak = f"{npf.mean.max():.1f} per field at {npf.hours[npf.mean.argmax()]:.0f} h" if npf is not None else "n/a"
        self.status.say(f"Microcolonies peak at {peak}; merge point {sh.merge_h} h (one microcolony holds half the cells in microcolonies and keeps doing so); "
                        f"mergers peak {dyn.get('merge_peak_per_h', 0):.1f} per h at {dyn.get('merge_peak_h', 0):.0f} h; splits are {dyn.get('splits_pct_of_mergers', 0):.0f} % of mergers.")


class MaturationPage(StagePage):
    name = "Maturation"
    question = "when does the matrix appear (Th1), and when does growth stop (Th2)?"

    def build_controls(self):
        self.pos = self.position_combo(); self.controls.addSpacing(12)
        self.controls.addWidget(QLabel("View:")); self.view = QComboBox(); self.view.addItems(["Growth states", "Matrix: the four premises and the threshold", "Matrix density"])
        self.view.currentIndexChanged.connect(lambda _i: self.render()); self.controls.addWidget(self.view)
        self.controls.addSpacing(12); self.controls.addWidget(QLabel("Layers:"))
        self.layer_checks = {}
        for key, text, on in (("cells", "cells", True), ("colonies", "microcolonies", True), ("matrix", "matrix", True), ("dye", "dye", False)):
            cb = QCheckBox(text); cb.setChecked(on); cb.toggled.connect(lambda _v: self.render()); self.layer_checks[key] = cb; self.controls.addWidget(cb)
        self.controls.addSpacing(12); self.controls.addWidget(QLabel("Threshold:"))
        self.threshold = QSlider(Qt.Horizontal); self.threshold.setRange(10, 90); self.threshold.setValue(40); self.threshold.setMaximumWidth(140)
        self.threshold.setToolTip("What-if on the dye threshold; 40 is the validated value (mean + 5 SD of the 24 h dye)")
        self.threshold.valueChanged.connect(lambda _v: self.render()); self.controls.addWidget(self.threshold)
        self.threshold_label = QLabel("40"); self.controls.addWidget(self.threshold_label); self.controls.addStretch()

    def set_study(self, study):
        super().set_study(study); self.fill_positions(self.pos)

    def render(self):
        if self.study is None: return
        fr = self.frame(); layers = tuple(k for k, cb in self.layer_checks.items() if cb.isChecked())
        v = self.view.currentIndex(); self.threshold_label.setText(str(self.threshold.value()))
        if v == 0:
            self.panel.render(P.draw_maturation, self.study, self.hour, fr, self.position, layers)
        elif v == 1:
            self.panel.render(P.draw_matrix_view, self.study, self.hour, fr, self.position, self.threshold.value())
        else:
            self.panel.render(P.draw_matrix_density, self.study, self.hour)
        sh = self.study.stage_hours(); mat = self.study.matrix_pct(); rate = self.study.area_rate()
        th2 = f"Th2 {sh.th2_h:.1f} h" if sh.th2_h is not None else "Th2 not reached: growth has not stopped by the end of the run"
        self.status.say(f"Th1 matrix appears {sh.th1_h} h (SD {sh.th1_sd_h:.1f} h over {len(sh.th1_positions)} positions)   |   {th2}   |   "
                        f"matrix peak {mat.mean.max():.0f} % at {mat.hours[mat.mean.argmax()]:.0f} h   |   cell-area rate peak {rate.mean.max():.2f} % per h at {rate.hours[rate.mean.argmax()]:.0f} h")


class DispersionPage(StagePage):
    name = "Dispersion"
    question = "not measured in this run: the recording ends before growth stops, and a cell that swims off leaves no trace in phase contrast. The candidate signals are shown."

    def build_controls(self):
        self.import_button = QPushButton("Import dispersion data..."); self.import_button.setToolTip("A table of a later recording, or literature, once there is one")
        self.import_button.clicked.connect(lambda: self.status.say("Import for dispersion is a slot: tell me the table's columns and it will plot here."))
        self.controls.addWidget(self.import_button); self.controls.addStretch()

    def render(self):
        if self.study is None: return
        self.panel.render(P.draw_dispersion, self.study, self.hour)
        cands = self.study.dispersion_candidates()
        header = cands.get("header", "") if isinstance(cands, dict) else ""
        items = cands.get("candidates", []) if isinstance(cands, dict) else (cands if isinstance(cands, list) else [])
        words = [f"{c.get('key', '')} {c.get('title', '')}".strip() for c in items if isinstance(c, dict)]
        self.status.say((header + "   Candidates: " + "; ".join(words)) if words else "Candidates: free cells (D1), microcolonies that vanish (D2); the others need the dye or a later run.")
