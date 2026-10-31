"""The Biofilm Stages tab: one study, four stage pages in the story's order, one cursor shared by all."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QPushButton, QTabWidget, QVBoxLayout, QWidget

from nd2_analyzer.analysis.biofilm_stages.study import BiofilmStudy
from .pages import DispersionPage, MaturationPage, MicrocoloniesPage, SeedingPage

DEFAULT_STUDY = Path(__file__).with_name("studies") / "ecoli_run2.json"


class BiofilmStagesTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.study = None
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Study:")); self.study_label = QLabel("none"); top.addWidget(self.study_label, 1)
        self.open_button = QPushButton("Open study..."); self.open_button.setToolTip("A study JSON: the results folder of the tool's run, the masks, the recording, the literature")
        self.open_button.clicked.connect(self._open); top.addWidget(self.open_button)
        self.compute_button = QPushButton("Compute from this recording"); self.compute_button.setEnabled(False)
        self.compute_button.setToolTip("Runs the tool's biofilm pipeline on the loaded recording (to be wired: the pipeline lives on the paper-gui branch)")
        top.addWidget(self.compute_button)
        lay.addLayout(top)
        self.tabs = QTabWidget()
        self.pages = [SeedingPage(), MicrocoloniesPage(), MaturationPage(), DispersionPage()]
        for p in self.pages:
            self.tabs.addTab(p, p.name); p.hour_changed.connect(self._on_hour)
        lay.addWidget(self.tabs, 1)
        self.setLayout(lay)
        if DEFAULT_STUDY.exists():
            try:
                self.load_study(str(DEFAULT_STUDY))
            except Exception as exc:
                self.study_label.setText(f"default study could not be read: {exc}")
        self.tabs.setCurrentIndex(2)

    def load_study(self, path: str):
        self.study = BiofilmStudy.from_json(path)
        av = self.study.available(); have = [k for k, v in av.items() if v]; miss = [k for k, v in av.items() if not v]
        self.study_label.setText(f"{self.study.name}   has: {', '.join(have)}" + (f"   missing: {', '.join(miss)}" if miss else ""))
        for p in self.pages:
            p.set_study(self.study)

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open study", str(DEFAULT_STUDY.parent), "Study (*.json)")
        if path: self.load_study(path)

    def _on_hour(self, h):
        for p in self.pages:
            if p is not self.sender(): p.set_hour(h)
