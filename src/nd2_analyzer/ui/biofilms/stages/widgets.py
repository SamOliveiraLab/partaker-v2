"""Widgets the stage pages share: the stage strip with a draggable cursor, a plot panel drawn by a function, an import row."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from nd2_analyzer.analysis.biofilm_stages import plots as P


class StageStrip(FigureCanvas):
    """The recording's hours with the stages, Th1 / Th2 and the cursor. Drag or click to move the cursor (hour_changed)."""
    hour_changed = Signal(float)

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(10, 0.9)); super().__init__(self.figure); self.setParent(parent)
        self.setFixedHeight(104); self.ax = self.figure.add_axes([0.04, 0.34, 0.93, 0.46])
        self.stage_hours = None; self.cursor_h = 24.0; self.highlight = None; self._drag = False
        self.mpl_connect("button_press_event", self._press); self.mpl_connect("motion_notify_event", self._move); self.mpl_connect("button_release_event", self._release)

    def set_stage_hours(self, sh, highlight=None):
        self.stage_hours = sh; self.highlight = highlight; self.redraw()

    def set_hour(self, h: float, emit: bool = False):
        self.cursor_h = float(h); self.redraw()
        if emit: self.hour_changed.emit(self.cursor_h)

    def redraw(self):
        self.ax.clear()
        if self.stage_hours is not None:
            P.draw_strip(self.ax, self.stage_hours, self.cursor_h, self.highlight)
        else:
            self.ax.axis("off"); self.ax.text(0.5, 0.5, "no study loaded", ha="center", va="center", transform=self.ax.transAxes, fontsize=9, color="#666666")
        self.draw_idle()

    def _snap(self, x):
        if self.stage_hours is None: return None
        return float(min(max(x, self.stage_hours.first_h), self.stage_hours.last_h))

    def _press(self, ev):
        if ev.inaxes is self.ax and ev.xdata is not None:
            self._drag = True; h = self._snap(ev.xdata)
            if h is not None: self.set_hour(h, emit=True)

    def _move(self, ev):
        if self._drag and ev.inaxes is self.ax and ev.xdata is not None:
            h = self._snap(ev.xdata)
            if h is not None and abs(h - self.cursor_h) >= 0.5: self.set_hour(h, emit=True)

    def _release(self, ev):
        self._drag = False


class PlotPanel(FigureCanvas):
    """A figure a page draws with one of the plots.* functions."""

    def __init__(self, parent=None, size=(12.5, 6.2)):
        self.figure = Figure(figsize=size); super().__init__(self.figure); self.setParent(parent)
        self.setMinimumHeight(380)

    def render(self, fn, *args, **kw):
        out = fn(self.figure, *args, **kw); self.draw_idle(); return out


class StatusRow(QWidget):
    """One line of plain words under a page: what is shown, where it came from, what is missing."""

    def __init__(self, parent=None):
        super().__init__(parent); lay = QHBoxLayout(self); lay.setContentsMargins(4, 0, 4, 0)
        self.label = QLabel(""); self.label.setWordWrap(True); lay.addWidget(self.label, 1)

    def say(self, text: str):
        self.label.setText(text)
