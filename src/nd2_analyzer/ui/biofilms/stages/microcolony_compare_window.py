"""The window where a person draws outlines around the microcolonies they see, next to what the rule found: click around
a microcolony, double-click (or Close outline) to finish it. The comparison, cell by cell, redraws as outlines are added.
Drawing and counting are analysis.biofilm_stages.microcolony_compare."""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from nd2_analyzer.analysis.biofilm_stages import microcolony_compare as MC

HINT = "Click around a microcolony to draw its outline. Double-click to finish it. Orange is what the tool found."


class MicrocolonyCompareDialog(QDialog):
    def __init__(self, labels: np.ndarray, pixel_um: float, title: str, outlines: Optional[list] = None,
                 on_change: Optional[Callable[[list], None]] = None, parent=None):
        super().__init__(parent)
        self.labels, self.pixel_um, self.title, self.on_change = np.asarray(labels), float(pixel_um), title, on_change
        self.outlines: List[list] = [list(map(tuple, o)) for o in (outlines or [])]; self.pending: List[tuple] = []; self.ax_img = None
        self.setWindowTitle("Manual selection and automatic detection"); self.setAttribute(Qt.WA_DeleteOnClose); self.setMinimumWidth(1000); self.setMinimumHeight(700)
        layout = QVBoxLayout(self); self.hint = QLabel(HINT); layout.addWidget(self.hint)
        self.figure = Figure(figsize=MC.FIGURE_IN); self.canvas = FigureCanvas(self.figure); self.canvas.mpl_connect("button_press_event", self._on_click)
        layout.addWidget(self.canvas, 1)
        self.summary = QLabel(""); layout.addWidget(self.summary)
        row = QHBoxLayout()
        self.close_outline_button = QPushButton("Close outline"); self.close_outline_button.clicked.connect(self.close_outline); row.addWidget(self.close_outline_button)
        self.undo_button = QPushButton("Undo"); self.undo_button.setToolTip("Takes back the last point, or the last outline"); self.undo_button.clicked.connect(self.undo); row.addWidget(self.undo_button)
        self.clear_button = QPushButton("Clear outlines"); self.clear_button.clicked.connect(self.clear_outlines); row.addWidget(self.clear_button)
        self.save_button = QPushButton("Save Image"); self.save_button.clicked.connect(self.save_image); row.addWidget(self.save_button)
        row.addStretch(); self.close_button = QPushButton("Close"); self.close_button.clicked.connect(self.close); row.addWidget(self.close_button)
        layout.addLayout(row)
        self.comparison = None; self.refresh(tell=False)

    # ---------------------------------------------------------------- drawing an outline
    def add_point(self, x: float, y: float):
        self.pending.append((float(x), float(y))); self.refresh(tell=False)

    def close_outline(self):
        if len(self.pending) >= 3:
            self.outlines.append(list(self.pending))
        self.pending = []; self.refresh()

    def undo(self):
        if self.pending:
            self.pending.pop(); self.refresh(tell=False)
        elif self.outlines:
            self.outlines.pop(); self.refresh()

    def clear_outlines(self):
        self.outlines, self.pending = [], []; self.refresh()

    def _on_click(self, event):
        if self.ax_img is None or event.inaxes is not self.ax_img or event.xdata is None:
            return
        if event.dblclick:
            self.close_outline()
        else:
            self.add_point(event.xdata, event.ydata)

    def refresh(self, tell: bool = True):
        self.comparison = MC.compare(self.labels, self.pixel_um, self.outlines)
        self.ax_img = MC.draw_compare(self.figure, self.labels, self.pixel_um, self.outlines, self.comparison, self.title, pending=self.pending)
        self.summary.setText(MC.summary_words(self.comparison)); self.canvas.draw_idle()
        self.close_outline_button.setEnabled(len(self.pending) >= 3); self.undo_button.setEnabled(bool(self.pending or self.outlines)); self.clear_button.setEnabled(bool(self.outlines or self.pending))
        if tell and self.on_change is not None:
            self.on_change([list(o) for o in self.outlines])

    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "manual_and_automatic.png", "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); self.hint.setText(f"Image saved: {path}")
