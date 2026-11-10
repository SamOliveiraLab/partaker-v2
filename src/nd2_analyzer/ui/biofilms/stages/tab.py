"""The Biofilm Stages tab: four stage pages in the story's order, one cursor shared by all.

The Seeding page is the live one: it measures the loaded recording through its controller (Track, Measure, the film, the
exports). The other three pages still read a study JSON until they are rebuilt the same way."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QPushButton, QTabWidget, QVBoxLayout, QWidget

from nd2_analyzer.analysis.biofilm_stages.study import BiofilmStudy
from .pages import DispersionPage, MaturationPage, MicrocoloniesPage
from .seeding_controller import SeedingController
from .seeding_page import SeedingPage

DEFAULT_STUDY = Path(__file__).with_name("studies") / "ecoli_run2.json"


class BiofilmStagesTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.study = None
        self.shared_hour = None
        lay = QVBoxLayout(self)
        self.study_row = QWidget(); top = QHBoxLayout(self.study_row); top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(QLabel("Study:")); self.study_label = QLabel("none"); top.addWidget(self.study_label, 1)
        self.open_button = QPushButton("Open study..."); self.open_button.setToolTip("A study JSON for the Microcolonies, Maturation and Dispersion pages: the results folder of the tool's run, the masks, the recording")
        self.open_button.clicked.connect(self._open); top.addWidget(self.open_button)
        self.compute_button = QPushButton("Compute from this recording"); self.compute_button.setEnabled(False)
        self.compute_button.setToolTip("Runs the tool's biofilm pipeline on the loaded recording (to be wired: the pipeline lives on the paper-gui branch)")
        top.addWidget(self.compute_button)
        lay.addWidget(self.study_row)
        self.tabs = QTabWidget()
        self.seeding_page = SeedingPage()
        self.controller = SeedingController(self.seeding_page, self)
        self.pages = [self.seeding_page, MicrocoloniesPage(), MaturationPage(), DispersionPage()]
        for p in self.pages:
            self.tabs.addTab(p, getattr(p, "name", "Seeding")); p.hour_changed.connect(self._on_hour)
        self.tabs.currentChanged.connect(self._on_tab_shown)
        lay.addWidget(self.tabs, 1)
        self.setLayout(lay)
        if DEFAULT_STUDY.exists():
            try:
                self.load_study(str(DEFAULT_STUDY))
            except Exception as exc:
                self.study_label.setText(f"default study could not be read: {exc}")
        self.tabs.setCurrentIndex(0); self.study_row.setVisible(False)

    def connect_main_window(self, set_frame=None, tracking_widget=None):
        """The main window gives two callbacks: set_frame(frame) moves the view area's T slider, tracking_widget() returns
        the tracking widget (its tracks are used when the Seeding page has not tracked the position itself)."""
        self.controller.connect_main_window(set_frame=set_frame, tracking_widget=tracking_widget)

    def follow_frame(self, frame: int):
        """The view area's T slider moved: the cursor follows it, on the Seeding strip now and on the other pages when
        they are shown. The strip does not call back, so a move that came from the strip ends here."""
        info = self.seeding_page.info
        if info is None or not info.dt_s:
            return
        h = float(info.start_hour) + int(frame) * float(info.dt_s) / 3600.0
        self.shared_hour = h
        if self.seeding_page.strip.cursor_h is None or abs(self.seeding_page.strip.cursor_h - h) > 1e-9:
            self.seeding_page.set_hour(h)

    def save_results(self, folder: str):
        """Write the Seeding page's results next to the experiment when the user saves; nothing when there are none."""
        if self.controller.results is None:
            return None
        return self.controller.save_results(folder)

    def load_study(self, path: str):
        self.study = BiofilmStudy.from_json(path)
        av = self.study.available(); have = [k for k, v in av.items() if v]; miss = [k for k, v in av.items() if not v]
        self.study_label.setText(f"{self.study.name}   has: {', '.join(have)}" + (f"   missing: {', '.join(miss)}" if miss else ""))
        for p in self.pages:
            if hasattr(p, "set_study"):
                p.set_study(self.study)

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open study", str(DEFAULT_STUDY.parent), "Study (*.json)")
        if path: self.load_study(path)

    def _on_hour(self, h):
        """One cursor for all four pages. Only the page on screen is redrawn now; the others take the hour when they are
        shown, so Play on the Seeding strip does not redraw three hidden study pages eight times a second."""
        self.shared_hour = float(h)
        shown = self.tabs.currentWidget()
        if shown is not self.sender() and shown is not None:
            shown.set_hour(h)

    def _on_tab_shown(self, _index):
        shown = self.tabs.currentWidget()
        self.study_row.setVisible(shown is not self.seeding_page)
        if shown is not None and self.shared_hour is not None and getattr(shown, "hour", None) != self.shared_hour:
            shown.set_hour(self.shared_hour)
