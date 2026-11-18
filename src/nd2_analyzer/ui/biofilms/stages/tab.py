"""The biofilm stages: four stage pages in the story's order, one cursor shared by all. In the main window the pages
are tabs of the main bar (move_pages_to), right after Segmentation: biofilm mode is already the biofilm layer, so no
"Biofilm Stages" tab sits around them. This object keeps their controllers and the shared cursor.

The Seeding, Microcolonies and Maturation pages are live: each measures the loaded recording through its controller
(Track or Measure, the film, the exports). The Dispersion page still reads a study JSON until it is rebuilt the same way."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QPushButton, QTabWidget, QVBoxLayout, QWidget

from nd2_analyzer.analysis.biofilm_stages.study import BiofilmStudy
from .microcolony_controller import MicrocolonyController
from .microcolony_page import MicrocolonyPage
from .growth_controller import GrowthController
from .growth_page import GrowthPage
from .pages import DispersionPage
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
        self.microcolony_page = MicrocolonyPage()
        self.microcolony_controller = MicrocolonyController(self.microcolony_page, self.controller.recording_info, self.controller.position, self)
        self.growth_page = GrowthPage()
        self.growth_controller = GrowthController(self.growth_page, self.controller.recording_info, self)
        self.microcolony_controller.stain_fallback = self.growth_controller.counted_at_for    # the matrix stain box of the Microcolonies window
        self.microcolony_controller.stain_recipe = self.growth_controller.stain_recipe        # the stain of the followed microcolonies, measured by itself
        self.growth_controller.measured_changed.connect(self._share_stain)
        self.growth_controller.measured_changed.connect(self._share_th1)                      # one Th1 in the tool: the Maturation tab's
        self.microcolony_controller.metrics_changed.connect(self._share_microcolonies)        # Figure 5 b: cells in microcolonies, on the Maturation plot
        self.live_pages = (self.seeding_page, self.microcolony_page, self.growth_page)
        self.pages = [self.seeding_page, self.microcolony_page, self.growth_page, DispersionPage()]
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

    def move_pages_to(self, tabs: QTabWidget, index: int = -1):
        """Make the four stage pages tabs of `tabs` (the main window's tab bar), from `index` on (-1: at the end). The
        pages that still read a study file get their own Open study button; this widget itself is no longer shown."""
        index = tabs.count() if index < 0 else index
        self.tabs.currentChanged.disconnect(self._on_tab_shown)
        self.study_labels = []
        for k, page in enumerate(self.pages):
            if hasattr(page, "set_study"):
                row = QWidget(); line = QHBoxLayout(row); line.setContentsMargins(0, 0, 0, 0)
                line.addWidget(QLabel("Study:")); label = QLabel(self.study_label.text()); line.addWidget(label, 1); self.study_labels.append(label)
                button = QPushButton("Open study..."); button.setObjectName("open_study"); button.setToolTip(self.open_button.toolTip())
                button.clicked.connect(self._open); line.addWidget(button)
                page.layout().insertWidget(0, row)
            tabs.insertTab(index + k, page, getattr(page, "name", "Seeding"))
        self.tabs = tabs
        tabs.currentChanged.connect(self._on_tab_shown)
        self.hide()

    def connect_main_window(self, set_frame=None, tracking_widget=None):
        """The main window gives two callbacks: set_frame(frame) moves the view area's T slider, tracking_widget() returns
        the tracking widget (its tracks are used when the Seeding page has not tracked the position itself)."""
        self.controller.connect_main_window(set_frame=set_frame, tracking_widget=tracking_widget)
        self.microcolony_controller.connect_main_window(set_frame=set_frame)
        self.growth_controller.connect_main_window(set_frame=set_frame)

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
        shown = self.tabs.currentWidget()
        if shown in (self.microcolony_page, self.growth_page) and abs(shown.hour - h) > 1e-9:
            shown.set_hour(h)

    def _share_th1(self):
        """The Microcolonies strip splits at the Th1 the Maturation tab measured (the mean of its positions)."""
        measured = self.growth_controller.measured
        self.microcolony_page.set_th1(measured.th1["th1_h"] if measured is not None and measured.th1 else None)

    def _share_stain(self):
        """The Microcolonies page shows the matrix stain as soon as the Maturation tab has measured: its plot of the
        stain inside and outside microcolonies, and (measured by itself) the stain of each followed microcolony."""
        self.microcolony_page.set_growth(self.growth_controller.measured); self.microcolony_controller.on_stain_source_changed()

    def _share_microcolonies(self):
        """The Maturation plot "Microcolonies and matrix stain" also draws the cells in microcolonies, once the
        Microcolonies tab has measured every position."""
        metrics = self.microcolony_controller.metrics
        self.growth_page.set_shared({"cells_in_pct": metrics.mean_of("cells_in_pct")} if metrics is not None else None)

    def save_results(self, folder: str):
        """Write the pages' results next to the experiment when the user saves: what the Microcolonies and Maturation
        pages worked out (File > Load brings it back) and the Seeding page's tables; nothing where there is none."""
        self.microcolony_controller.save_state(folder); self.growth_controller.save_state(folder)
        if self.controller.results is None:
            return None
        return self.controller.save_results(folder)

    def load_results(self, folder: str) -> dict:
        """After a session is loaded: bring back what the Microcolonies and Maturation pages had worked out. The growth
        states come first, so the matrix stain is there for the Microcolonies window."""
        maturation = self.growth_controller.load_state(folder)
        return {"microcolonies": self.microcolony_controller.load_state(folder), "maturation": maturation}

    def load_study(self, path: str):
        self.study = BiofilmStudy.from_json(path)
        av = self.study.available(); have = [k for k, v in av.items() if v]; miss = [k for k, v in av.items() if not v]
        self.study_label.setText(f"{self.study.name}   has: {', '.join(have)}" + (f"   missing: {', '.join(miss)}" if miss else ""))
        for label in getattr(self, "study_labels", []):
            label.setText(self.study_label.text())
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
        if shown is not self.sender() and shown in self.pages:
            shown.set_hour(h)

    def _on_tab_shown(self, _index):
        shown = self.tabs.currentWidget()
        if shown not in self.pages:
            return                                      # another tab of the main bar (Segmentation, Live-Dead Analysis)
        self.study_row.setVisible(shown not in self.live_pages)
        if self.shared_hour is not None and getattr(shown, "hour", None) != self.shared_hour:
            shown.set_hour(self.shared_hour)
