"""The Seeding page: the first stage of a biofilm, measured from the loaded recording. Steps row, stage strip with the
cursor and its Play button, the cells to follow with their film, the gallery of small plots in three columns, and a status
line. Standard Qt widgets only. The page never reaches into the main window: the wiring calls set_recording /
set_step_state / set_results / set_hour / set_film_frames and listens to hour_changed / follow_requested / track_picked /
track_requested / measure_requested / export_requested.

The gallery reads seeding_plots.PANELS, one (group, key, title, fn, needs_flow) per panel, and draws each through
seeding_plots.draw_panel(fig, panel, results, span, cursor_h, settings). Only the panels whose x axis is time after
inoculation follow the cursor, at most ten times a second: when seeding_plots names them (TIME_PANEL_KEYS) the page draws
the cursor line itself over a cached background, so a cursor move costs a blit and not a redraw. The landing panel (key
"landing_dots") is clickable: the nearest dot picks that track through seeding_plots.pick_landing."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import polars as pl
from PySide6.QtCore import Qt, QSize, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from nd2_analyzer.analysis.biofilm_stages import seeding_figures as SF
from nd2_analyzer.analysis.biofilm_stages import seeding_plots as SP
from nd2_analyzer.analysis.biofilm_stages.seeding_types import Metric, RecordingInfo, SeedingResults, SeedingSettings

STEPS = ("load", "segment", "track", "measure")
STEP_NAMES = {"load": "Raw Frames", "segment": "Segmented Frames", "track": "Cell Tracks", "measure": "Seeding Numbers"}
STEP_ACTIONS = {"load": "Load", "segment": "Segment", "track": "Track", "measure": "Measure"}   # for "Track failed: ..."
STATE_WORDS = {"todo": "not yet", "running": "running", "done": "ready", "failed": "failed"}
GROUPS = (("arrivals", "Reversible attachment: arrivals and leaving"), ("commitment", "Irreversible attachment"),
          ("movement", "Movement on the surface"), ("division", "Division and families"), ("flow", "Flow"))
GROUP_HEADINGS = dict(GROUPS)
GALLERY_COLUMNS = 3           # panels per row when the window is wide enough
PANEL_SIZE_IN = (3.6, 2.6)    # one gallery panel at the screen dpi
PANEL_MIN_HEIGHT_PX = 250
PANEL_COLUMN_MIN_PX = 340     # a column narrower than this clips the panel's labels: the gallery drops to fewer columns
LEFT_COLUMN_PX = 340
STEP_DETAIL_CHARS = 60        # a longer step detail (an exception text) is cut in the label and kept whole in its tooltip
BIG_VALUE_PT = 5              # points above the base font for a card's headline
HOURS_PER_SECOND = 1.0        # Play on the strip: hours of recording per second of playback
PLAY_TICK_MS = 100            # the strip's Play timer: ten cursor steps a second
REDRAW_EVERY_MS = 100         # the time panels redraw at most this often while the cursor moves
LANDING_PANEL_KEY = "landing_dots"
PICK_TOLERANCE_PX = 8
MAX_LIST_ROWS = 60            # rows per section of the cells list; the rest is one line "and n more"
FILM_STEPS = 40               # cursor steps of the followed cell's film


def step_words(state: str, detail: str = "") -> str:
    """The state of one step in plain words: what it made once it ran ("35,400"), else "running, 12 %", "not yet"."""
    if state == "done" and detail:
        return detail
    words = STATE_WORDS.get(state, state)
    return f"{words}, {detail}" if detail else words


def short_detail(detail: str, limit: int = STEP_DETAIL_CHARS) -> str:
    """A step detail cut to one short line; the whole text goes in the tooltip."""
    detail = " ".join(str(detail).split())
    return detail if len(detail) <= limit else detail[:limit - 3].rstrip() + "..."


def candidate_line(row: dict) -> str:
    """One list entry from the candidates table: "Track 412: stays 3.4 h, divides at 22.1 h   (20.1 to 23.5 h)"."""
    line = f"Track {int(row['track_id'])}: {row.get('why', '')}".rstrip(": ")
    t0, t1 = row.get("t_start_h"), row.get("t_end_h")
    if t0 is not None and t1 is not None:
        line += f"   ({float(t0):.1f} to {float(t1):.1f} h)"
    return line


def track_line(row: dict) -> str:
    """One list entry from per_track: "Track 3930: lands 24.2 h, divides 26.7 h", "... still there at 29.5 h",
    "... leaves 28.1 h"."""
    fate = row.get("fate", "")
    end = float(row["t_end_h"])
    if fate == "divided":
        tail = f"divides {end:.1f} h"
    elif fate == "present at end":
        tail = "stays to the end"
    else:
        tail = f"left {end:.1f} h"
    return f"Track {int(row['track_id'])}: lands {float(row['t_start_h']):.1f} h, {tail}"


def cells_to_follow(results: SeedingResults) -> list:
    """The two sections of the cells list as (title, rows): landed cells that stay for good (divided, or still there at
    the end after irreversible_stay_s), earliest landing first; then landed cells that stayed SF.LONG_STAY_S or more and
    left, earliest first. Empty when per_track lacks the columns."""
    pt = results.per_track
    needed = {"track_id", "origin", "fate", "t_start_h", "t_end_h", "residence_s"}
    if pt is None or not pt.height or not needed <= set(pt.columns):
        return []
    landed = SF.landed_classed(results)
    for_good = landed.filter(pl.col("fate_class").is_in(["divided", "end"])).sort("t_start_h")
    long_then_left = landed.filter(pl.col("fate").is_in(["detached", "left field"]) &
                                   (pl.col("residence_s") >= SF.LONG_STAY_S)).sort("t_start_h")
    return [("irreversible attachment", for_good.to_dicts()),
            (f"reversible attachment: stayed {SF.LONG_STAY_S / 60:g} min or more", long_then_left.to_dicts())]


def panel_list() -> list:
    """seeding_plots.PANELS, or an empty list while the panel functions have not landed (the gallery then says so)."""
    return list(getattr(SP, "PANELS", None) or [])


def is_time_panel(panel, ax) -> bool:
    """True when a drawn panel's x axis is time after inoculation, so the cursor line belongs on it. seeding_plots may
    name them in TIME_PANEL_KEYS; otherwise the x label says so."""
    keys = getattr(SP, "TIME_PANEL_KEYS", None)
    if keys is not None:
        return panel[1] in keys
    if ax is None or not hasattr(ax, "get_xlabel"):
        return False
    label = ax.get_xlabel().lower()
    return "inoculation" in label or label.endswith("(h)") or "hour" in label


class StripCanvas(FigureCanvas):
    """The stage strip: the recording's hours, the stages shaded once results exist, a red cursor. Click or drag moves the
    cursor and emits hour_changed."""
    hour_changed = Signal(float)

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(10, 1.0)); super().__init__(self.figure); self.setParent(parent)
        self.setFixedHeight(100); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.ax = self.figure.add_axes([0.03, 0.36, 0.94, 0.46])
        self.span = None; self.cursor_h = 0.0; self.stage_hours = None; self.stages = None; self._drag = False
        self.mpl_connect("button_press_event", self._press); self.mpl_connect("motion_notify_event", self._move)
        self.mpl_connect("button_release_event", self._release)
        self.redraw()

    def set_span(self, span):
        self.span = (float(span[0]), float(span[1])) if span is not None else None
        if self.span is not None:
            self.cursor_h = min(max(self.cursor_h, self.span[0]), self.span[1])
        self.redraw()

    def set_stages(self, hours, stages):
        self.stage_hours = None if hours is None else np.asarray(hours, float)
        self.stages = None if stages is None else list(stages)
        self.redraw()

    def set_hour(self, h: float, emit: bool = False):
        self.cursor_h = float(h) if self.span is None else float(min(max(h, self.span[0]), self.span[1]))
        self.redraw()
        if emit:
            self.hour_changed.emit(self.cursor_h)

    def redraw(self):
        self.ax.clear()
        if self.span is None:
            self.ax.axis("off")
            self.ax.text(0.5, 0.5, "Load a recording to see its hours", ha="center", va="center", transform=self.ax.transAxes,
                         fontsize=9, color=SP.EMPTY_GREY)
        else:
            self.ax.axis("on")
            SP.draw_strip(self.ax, self.span, self.cursor_h, self.stage_hours, self.stages)
        self.draw_idle()

    def _hour_at(self, ev) -> Optional[float]:
        if self.span is None or ev.inaxes is not self.ax or ev.xdata is None:
            return None
        return float(min(max(ev.xdata, self.span[0]), self.span[1]))

    def _press(self, ev):
        h = self._hour_at(ev)
        if h is not None:
            self._drag = True; self.set_hour(h, emit=True)

    def _move(self, ev):
        if not self._drag:
            return
        h = self._hour_at(ev)
        if h is not None and abs(h - self.cursor_h) >= self.step_h():
            self.set_hour(h, emit=True)

    def _release(self, ev):
        self._drag = False

    def step_h(self) -> float:
        """The smallest cursor move worth an emit: one 200th of the span."""
        return (self.span[1] - self.span[0]) / 200.0 if self.span else 0.1


class PlotCanvas(FigureCanvas):
    """A canvas drawn by one seeding_plots function (the film strip)."""

    def __init__(self, parent=None, size=(5.6, 2.6), min_height=230):
        self.figure = Figure(figsize=size); super().__init__(self.figure); self.setParent(parent)
        self.setMinimumHeight(min_height); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def render(self, fn, *args, **kw):
        out = fn(self.figure, *args, **kw); self.draw_idle(); return out


class PanelCanvas(FigureCanvas):
    """One gallery panel: a small figure drawn by seeding_plots.draw_panel. Keeps the axes it drew (for picking) and
    whether its x axis is time (for the cursor). On a time panel the page owns the cursor: draw_panel gets cursor_h None,
    the canvas adds an animated grey line and moves it over the cached background (a blit, not a redraw). A mouse press
    on the landing panel reports the display position."""
    pressed = Signal(float, float)

    def __init__(self, panel, parent=None, size=PANEL_SIZE_IN, min_height=PANEL_MIN_HEIGHT_PX):
        self.figure = Figure(figsize=size); super().__init__(self.figure); self.setParent(parent)
        self.size_in = tuple(size)
        self.panel = panel
        self.key = panel[1]
        self.ax = None
        self.time_panel: Optional[bool] = None
        self.owns_cursor = False
        self.cursor_line = None
        self.background = None
        self.cursor_span: Optional[tuple] = None
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setToolTip(panel[2])
        self.mpl_connect("draw_event", self._on_draw)
        if self.key == LANDING_PANEL_KEY:
            self.mpl_connect("button_press_event", self._press)

    def set_panel(self, panel):
        """Show another figure in this canvas (the plot picker)."""
        self.panel = panel; self.key = panel[1]; self.setToolTip(panel[2])

    def sizeHint(self) -> QSize:
        w, h = self.size_in
        return QSize(int(w * self.figure.dpi / max(self.device_pixel_ratio, 1)), int(h * self.figure.dpi / max(self.device_pixel_ratio, 1)))

    def render(self, results, span, cursor_h: float, settings, selected=None):
        """Draw the whole panel. The cursor line is the page's own when seeding_plots names the time panels."""
        self.cursor_line = None; self.background = None
        keys = getattr(SP, "TIME_PANEL_KEYS", None)
        self.owns_cursor = results is not None and keys is not None and self.key in keys
        kw = {"selected": selected} if self.key == LANDING_PANEL_KEY else {}
        self.ax = SP.draw_panel(self.figure, self.panel, results, span, None if self.owns_cursor else cursor_h, settings, **kw)
        self.time_panel = is_time_panel(self.panel, self.ax) if results is not None else None
        if self.owns_cursor and plot_axes(self.ax):
            self.cursor_span = SP.recording_span(results, span)
            self.cursor_line = self.ax.axvline(float(cursor_h), color=SP.GREY, lw=1.0, zorder=0.5, animated=True)
            self.cursor_line.set_visible(self._inside(cursor_h))
        self.draw_idle()
        return self.ax

    def _inside(self, h: float) -> bool:
        a, b = self.cursor_span
        return a <= float(h) <= b

    def move_cursor(self, h: float) -> bool:
        """Move the page's cursor line to hour h over the cached background; False when this panel has no such line
        (then the caller draws the panel again)."""
        if self.cursor_line is None:
            return False
        self.cursor_line.set_xdata([float(h), float(h)]); self.cursor_line.set_visible(self._inside(h))
        if self.background is None:
            self.draw_idle()                                  # the first full draw caches the background and draws the line
            return True
        self.restore_region(self.background)
        self.ax.draw_artist(self.cursor_line)
        self.blit(self.figure.bbox)
        return True

    def _on_draw(self, _event):
        """After every full draw: keep the picture without the cursor, then draw the cursor on top of it."""
        if self.cursor_line is None:
            self.background = None
            return
        self.background = self.copy_from_bbox(self.figure.bbox)
        self.ax.draw_artist(self.cursor_line)

    def _press(self, ev):
        if ev.button == 1 and ev.x is not None and ev.y is not None:
            self.pressed.emit(float(ev.x), float(ev.y))


def plot_axes(ax) -> bool:
    """True for axes that show a plot with an x axis (not a grey reason panel)."""
    return ax is not None and hasattr(ax, "axison") and bool(ax.axison) and bool(ax.get_xlabel())


class MetricCard(QFrame):
    """One metric: the big value, the name, a small source line. Unavailable cards are disabled and show the reason.
    The page no longer shows cards (the numbers live inside their panels and in the Numbers table); the class stays
    for anyone who still builds one."""

    def __init__(self, metric: Metric, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel); self.setFrameShadow(QFrame.Raised)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay = QVBoxLayout(self); lay.setContentsMargins(10, 8, 10, 8); lay.setSpacing(2)
        self.value = QLabel(); self.value.setWordWrap(True)
        self.name = QLabel(metric.name); self.name.setWordWrap(True)
        self.source = QLabel(""); self.source.setWordWrap(True)
        big = QFont(self.value.font()); big.setPointSize(big.pointSize() + BIG_VALUE_PT); big.setBold(True)
        small = QFont(self.source.font()); small.setPointSize(max(small.pointSize() - 2, 7))
        self.source.setFont(small)
        if metric.available:
            self.value.setFont(big); self.value.setText(metric.text or self.format_value(metric))
        else:
            self.value.setText(metric.reason or "needs more video")
            self.setEnabled(False)
        if metric.n is not None and metric.available:
            self.source.setText(f"n = {metric.n:,}")
        self.value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.value); lay.addWidget(self.name); lay.addWidget(self.source)
        self.setToolTip(metric.reason if not metric.available else f"Rule from {metric.source}")

    @staticmethod
    def format_value(metric: Metric) -> str:
        if metric.value is None:
            return ""
        v = float(metric.value)
        text = f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.1f}" if abs(v) >= 1 else f"{v:.2f}"
        return f"{text} {metric.unit}".strip()


class FactCard(QFrame):
    """One number of the page and one plain line on what it means; grey with the reason when the recording cannot give it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel); self.setFrameShadow(QFrame.Raised)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay = QVBoxLayout(self); lay.setContentsMargins(10, 6, 10, 6); lay.setSpacing(1)
        self.value = QLabel(""); self.meaning = QLabel(""); self.meaning.setWordWrap(True)
        big = QFont(self.value.font()); big.setPointSize(big.pointSize() + 3); big.setBold(True); self.value.setFont(big)
        lay.addWidget(self.value); lay.addWidget(self.meaning)

    def show_fact(self, value: str, meaning: str, available: bool, tooltip: str = ""):
        self.value.setText(value); self.meaning.setText(meaning)
        self.setEnabled(bool(available)); self.setToolTip(tooltip)


class GalleryArea(QScrollArea):
    """The scroll area of the gallery; tells the page when its viewport width changes so the grid can reflow."""
    width_changed = Signal(int)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.width_changed.emit(self.viewport().width())


def columns_for(width_px: int) -> int:
    """How many panel columns a gallery this wide holds: up to GALLERY_COLUMNS, each at least PANEL_COLUMN_MIN_PX."""
    return max(1, min(GALLERY_COLUMNS, int(width_px) // PANEL_COLUMN_MIN_PX))


class NumbersDialog(QDialog):
    """The compact numbers table: one row per metric with its name, value and n; the source is the row's tooltip.
    Unavailable metrics show their reason in a disabled row."""
    COLUMNS = ("Metric", "Value", "n")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Numbers")
        self.resize(760, 520)
        lay = QVBoxLayout(self)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setWordWrap(True)                      # a long reason is read whole, on two lines
        lay.addWidget(self.table, 1)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.buttons.rejected.connect(self.close)
        lay.addWidget(self.buttons)

    def fill(self, metrics: list):
        self.table.setRowCount(0)
        self.table.setRowCount(len(metrics))
        for r, m in enumerate(metrics):
            words = m.text if m.available else (m.reason or "not available")
            n = f"{m.n:,}" if m.n is not None else ""
            tip = f"Rule from {m.source}" if m.source else ""
            if not m.available:
                tip = f"Not available: {words}" + (f"\n{tip}" if tip else "")
            for c, text in enumerate((m.name, words, n)):
                item = QTableWidgetItem(text)
                item.setToolTip(tip)
                if not m.available:
                    item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                if c == 2:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, c, item)
        self.table.resizeColumnToContents(0)
        self.table.resizeColumnToContents(2)
        self.table.resizeRowsToContents()


class SeedingPage(QWidget):
    """The Seeding page of the Biofilm Stages tab."""
    hour_changed = Signal(float)
    follow_requested = Signal(int)
    window_requested = Signal(str, object)     # "Field" or "Cell", the selected track or None
    track_picked = Signal(int)
    track_requested = Signal()
    measure_requested = Signal()
    export_requested = Signal(str)
    clear_requested = Signal()                 # "Clear them" in the not-seeding warning

    def __init__(self, parent=None):
        super().__init__(parent)
        self.info: Optional[RecordingInfo] = None
        self.results: Optional[SeedingResults] = None
        self.step_states = {s: ("todo", "") for s in STEPS}
        self.followed_track: Optional[int] = None
        self.followed_span: Optional[tuple] = None
        self.selected_dot: Optional[int] = None
        self.panels: list = []
        self.gallery_has_flow: Optional[bool] = None
        self.gallery_span: Optional[tuple] = None
        self.gallery_columns = GALLERY_COLUMNS
        self.headings: list = []
        self.numbers_dialog: Optional[NumbersDialog] = None
        self.play_timer = QTimer(self); self.play_timer.setInterval(125); self.play_timer.timeout.connect(self._play_step)
        self.play_hours: list = []; self.play_index = 0
        self.strip_timer = QTimer(self); self.strip_timer.setInterval(PLAY_TICK_MS); self.strip_timer.timeout.connect(self._strip_play_step)
        self.redraw_timer = QTimer(self); self.redraw_timer.setSingleShot(True); self.redraw_timer.setInterval(REDRAW_EVERY_MS)
        self.redraw_timer.timeout.connect(self._redraw_time_panels)
        self.pending_hour: Optional[float] = None
        self.last_tick: Optional[float] = None
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 4, 6, 4); lay.setSpacing(4)
        self.recording_label = QLabel("No recording loaded"); self.recording_label.setWordWrap(True)
        lay.addWidget(self.recording_label)
        lay.addLayout(self._build_steps_row())
        lay.addLayout(self._build_strip_row())
        lay.addLayout(self._build_middle_row(), 1)
        self.status = QLabel("Load a recording and segment it, then click Track and Measure."); self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.setLayout(lay)
        self._build_gallery(); self.film.render(SP.draw_film, [], [], [], [])

    # ------------------------------------------------------------------------------------------ building
    def _build_steps_row(self) -> QHBoxLayout:
        row = QHBoxLayout(); row.setSpacing(16)
        self.step_labels = {}
        for step in STEPS:
            lab = QLabel(); self.step_labels[step] = lab; row.addWidget(lab)
        row.addStretch(1)
        # the reminder that stays after the warning closes; clicking it shows the warning again
        self.check_link = QLabel(f'<a href="why" style="color:{SF.LINEAGE_RED}; font-weight:bold">May not be seeding: why?</a>')
        self.check_link.setToolTip("These frames show signs that they are not a seeding recording. Click to see them.")
        self.check_link.linkActivated.connect(lambda _href: self.ask_about_seeding())
        self.check_link.setVisible(False); row.addWidget(self.check_link)
        self.check_box: Optional[QMessageBox] = None
        self.track_button = QPushButton("Track"); self.track_button.setEnabled(False)
        self.track_button.setToolTip("Links the segmented cells frame to frame into tracks with divisions: most overlap, then least displacement")
        self.track_button.clicked.connect(self.track_requested.emit); row.addWidget(self.track_button)
        self.measure_button = QPushButton("Measure"); self.measure_button.setEnabled(False)
        self.measure_button.setToolTip("Runs the seeding rules on the segmented and tracked frames of this recording")
        self.measure_button.clicked.connect(self.measure_requested.emit); row.addWidget(self.measure_button)
        self._refresh_steps()
        return row

    def _build_strip_row(self) -> QHBoxLayout:
        row = QHBoxLayout(); row.setSpacing(8)
        self.strip = StripCanvas(); self.strip.hour_changed.connect(self._on_strip); row.addWidget(self.strip, 1)
        self.strip_play_button = QPushButton("Play"); self.strip_play_button.setEnabled(False)
        self.strip_play_button.setToolTip(f"Moves the cursor through the recording, {HOURS_PER_SECOND:g} h of recording per second; "
                                          "the view area and the time plots follow")
        self.strip_play_button.clicked.connect(self._toggle_strip_play)
        row.addWidget(self.strip_play_button, 0, Qt.AlignVCenter)
        return row

    def _build_middle_row(self) -> QHBoxLayout:
        row = QHBoxLayout(); row.setSpacing(8)
        left = QVBoxLayout(); left.setSpacing(4)
        left.addWidget(QLabel("Cells to follow"))
        self.candidates = QListWidget(); self.candidates.setMinimumHeight(100); self.candidates.setWordWrap(True)
        self.candidates.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.candidates.itemDoubleClicked.connect(lambda _item: self._follow())
        self.candidates.currentItemChanged.connect(self._on_list_selection)
        left.addWidget(self.candidates, 1)
        buttons = QGridLayout(); buttons.setSpacing(4)
        self.follow_button = QPushButton("Follow"); self.follow_button.clicked.connect(self._follow)
        self.follow_button.setToolTip("Opens the Seeding window on the selected cell: its film and its lineage tree")
        self.field_button = QPushButton("Field film"); self.field_button.clicked.connect(self._field_film)
        self.field_button.setToolTip("Opens the Seeding window on the whole field, every cell coloured by what it does")
        self.play_button = QPushButton("Play film"); self.play_button.clicked.connect(self._toggle_play)
        self.play_button.setToolTip("Moves the cursor through the followed cell's life")
        self.film_button = QPushButton("Save film"); self.film_button.clicked.connect(lambda: self.export_requested.emit("film"))
        self.image_button = QPushButton("Save image"); self.image_button.clicked.connect(lambda: self.export_requested.emit("image"))
        self.numbers_button = QPushButton("Numbers"); self.numbers_button.setEnabled(False)
        self.numbers_button.setToolTip("Every metric as a table: name, value, n")
        self.numbers_button.clicked.connect(self.open_numbers)
        buttons.addWidget(self.follow_button, 0, 0); buttons.addWidget(self.field_button, 0, 1); buttons.addWidget(self.play_button, 0, 2)
        buttons.addWidget(self.film_button, 1, 0); buttons.addWidget(self.image_button, 1, 1); buttons.addWidget(self.numbers_button, 1, 2)
        left.addLayout(buttons)
        self.film = PlotCanvas(size=(3.3, 1.25), min_height=115); self.film.setMaximumHeight(135)
        left.addWidget(self.film)
        left_box = QWidget(); left_box.setLayout(left); left_box.setFixedWidth(LEFT_COLUMN_PX)
        row.addWidget(left_box)
        right = QVBoxLayout(); right.setSpacing(6)
        self.cards_grid = QGridLayout(); self.cards_grid.setSpacing(6)
        self.cards = []
        for i in range(8):
            card = FactCard(); self.cards.append(card); self.cards_grid.addWidget(card, i // 4, i % 4)
        right.addLayout(self.cards_grid)
        picker = QHBoxLayout(); picker.setSpacing(8)
        picker.addWidget(QLabel("Plots:"))
        self.plot_combos = []
        for default in SF.FIGURE_DEFAULTS:
            combo = QComboBox()
            for f in SF.FIGURES:
                combo.addItem(f[2], f[1])
            combo.setCurrentIndex(max(0, combo.findData(default)))
            combo.currentIndexChanged.connect(self._on_plot_picked)
            self.plot_combos.append(combo); picker.addWidget(combo)
        picker.addStretch(1)
        right.addLayout(picker)
        plots = QHBoxLayout(); plots.setSpacing(6)
        self.panels = []
        self.resize_timer = QTimer(self); self.resize_timer.setSingleShot(True); self.resize_timer.setInterval(200)
        self.resize_timer.timeout.connect(self._render_all)
        for default in SF.FIGURE_DEFAULTS:
            canvas = PanelCanvas(SF.figure_by_key(default), size=SF.FIGURE_SIZE_IN, min_height=330)
            canvas.mpl_connect("resize_event", lambda _e: self.resize_timer.start())
            self.panels.append(canvas); plots.addWidget(canvas, 1)
        right.addLayout(plots, 1)
        right_box = QWidget(); right_box.setLayout(right)
        row.addWidget(right_box, 1)
        return row

    def _build_gallery(self):
        """The two big plots: drawn again when the recording span changes (the canvases are made once)."""
        span = self.info.span if self.info is not None else None
        if span != self.gallery_span or not getattr(self, "_drawn_once", False):
            self.gallery_span = span
            self._drawn_once = True
            self._render_all()

    def _on_plot_picked(self, _i=None):
        """A plot picked in one of the two lists: that canvas shows it."""
        for combo, canvas in zip(self.plot_combos, self.panels):
            f = SF.figure_by_key(combo.currentData())
            if f is not None and f[1] != canvas.key:
                canvas.set_panel(f)
        self._render_all()

    def _fill_cards(self, results):
        facts = SF.card_facts(results) if results is not None else []
        for i, card in enumerate(self.cards):
            if i < len(facts):
                card.show_fact(*facts[i]); card.show()
            else:
                card.show_fact("", "", False); card.hide() if results is not None else card.show()

    def _clear_gallery(self):
        while self.gallery_grid.count():
            item = self.gallery_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide(); w.setParent(None); w.deleteLater()

    def _place_gallery(self):
        """Put the headings and canvases in the grid with the current column count."""
        cols = self.gallery_columns
        while self.gallery_grid.count():
            self.gallery_grid.takeAt(0)
        for c in range(GALLERY_COLUMNS):
            self.gallery_grid.setColumnStretch(c, 1 if c < cols else 0)
        r = 0
        for head, canvases in self.headings:
            self.gallery_grid.addWidget(head, r, 0, 1, cols); r += 1
            for i, canvas in enumerate(canvases):
                self.gallery_grid.addWidget(canvas, r + i // cols, i % cols)
            r += (len(canvases) + cols - 1) // cols
        for row in range(self.gallery_grid.rowCount()):
            self.gallery_grid.setRowStretch(row, 0)
        self.gallery_grid.setRowStretch(r, 1)

    def _on_gallery_width(self, width_px: int):
        cols = columns_for(width_px)
        if cols != self.gallery_columns:
            self.gallery_columns = cols
            if self.panels:
                self._place_gallery()

    @staticmethod
    def _groups_in_order(show_flow: bool) -> list:
        """The known groups in the page's order, then any other group seeding_plots names, in its order."""
        known = [g for g, _ in GROUPS if show_flow or g != "flow"]
        extra = []
        for p in panel_list():
            if p[0] not in known and p[0] not in extra and (show_flow or not p[4]):
                extra.append(p[0])
        return [(g, GROUP_HEADINGS.get(g, g)) for g in known + extra]

    # ------------------------------------------------------------------------------------------ the API the wiring calls
    def set_recording(self, info: Optional[RecordingInfo]):
        self.info = info
        if info is None:
            self.recording_label.setText("No recording loaded"); self.strip.set_span(None)
        else:
            self.recording_label.setText(self.recording_words(info))
            self.strip.set_span(info.span)
        self._stop_strip_play()
        self.strip_play_button.setEnabled(info is not None)
        self._build_gallery(); self._refresh_steps()

    def set_step_state(self, step: str, state: str, detail: str = ""):
        if step not in self.step_states:
            raise ValueError(f"unknown step {step!r}, expected one of {STEPS}")
        self.step_states[step] = (state, detail)
        self._refresh_steps()

    def set_results(self, results: Optional[SeedingResults]):
        self.results = results
        self.followed_track = None; self.followed_span = None; self.selected_dot = None
        self.play_timer.stop(); self.play_button.setText("Play film")
        self.candidates.clear()
        if results is not None:
            self._fill_candidates(results)
            sbf = results.stage_by_frame
            if sbf is not None and sbf.height:
                self.strip.set_stages(sbf["hour"].to_numpy(), sbf["stage"].to_list())
            else:
                self.strip.set_stages(None, None)
            self.status.setText(SF.status_words(results, self.info))
            self.status.setToolTip("\n".join(str(n) for n in results.notes) if results.notes else "")
        else:
            self.strip.set_stages(None, None)
            self.status.setText("Load a recording and segment it, then click Track and Measure.")
        if self.check_box is not None:
            self.check_box.close(); self.check_box = None   # a warning about earlier numbers goes with them
        self.check_link.setVisible(bool(results is not None and results.check))
        self.numbers_button.setEnabled(results is not None)
        self._fill_cards(results)
        if self.numbers_dialog is not None and self.numbers_dialog.isVisible():
            self.numbers_dialog.fill(results.metrics if results is not None else [])
        self.redraw_timer.stop(); self.pending_hour = None
        self._render_all()
        self.film.render(SP.draw_film, [], [], [], [])

    def ask_about_seeding(self):
        """The warning when the frames do not look like seeding: the signs, then keep the numbers or clear them. It opens
        without blocking (open, not exec), so the window and the tests carry on."""
        signs = self.results.check if self.results is not None else []
        if not signs:
            return None
        if self.check_box is not None:
            self.check_box.close()
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Seeding check")                   # macOS shows no title on a message box: the question is the text
        box.setText("Do these frames show seeding?")
        box.setInformativeText("They do not look like it:\n\n" + "\n\n".join(f"\u2022 {s}" for s in signs)
                               + "\n\nThe Seeding page reads every new cell as a "
                               "landing and every lost cell as one that left. If the recording shows a colony that is already "
                               "growing, or starts after the surface filled up, read these numbers with care.")
        keep = box.addButton("Keep the numbers", QMessageBox.AcceptRole)
        box.addButton("Clear them", QMessageBox.DestructiveRole)
        box.setDefaultButton(keep)
        box.setEscapeButton(keep)                             # Esc or closing the box keeps them (and lets the app quit)
        box.buttonClicked.connect(lambda b, k=keep: None if b is k else self.clear_requested.emit())
        box.setWindowModality(Qt.WindowModal)
        box.open()
        self.check_box = box
        return box

    def set_hour(self, h: float):
        self.strip.set_hour(h, emit=False)
        self._schedule_redraw()

    def set_film_frames(self, frames, hours, trail, fates):
        pixel_um = self.info.pixel_um if self.info is not None else None
        self.film.render(SP.draw_film, list(frames), list(hours), list(trail), list(fates), pixel_um=pixel_um)
        if hours:
            self.followed_span = (float(min(hours)), float(max(hours)))

    # ------------------------------------------------------------------------------------------ pieces
    @staticmethod
    def recording_words(info: RecordingInfo) -> str:
        parts = [info.name or "recording", f"{info.n_frames:,} frames", f"{info.dt_s:g} s apart"]
        if info.pixel_um:
            parts.append(f"{info.pixel_um:.4f} µm per px")
        parts.append(f"{info.start_hour:g} to {info.end_hour:.1f} h after inoculation")
        if info.has_flow:
            direction = (info.flow or {}).get("direction"); shear = (info.flow or {}).get("shear_mpa")
            flow = "flow"
            if direction: flow += f" {direction}"
            if shear is not None: flow += f", {shear:g} mPa"
            parts.append(flow)
        else:
            parts.append("no flow")
        return ", ".join(parts)

    def _refresh_steps(self):
        for step in STEPS:
            state, detail = self.step_states[step]
            label = self.step_labels[step]
            label.setText(f"<b>{STEP_NAMES[step]}:</b> {step_words(state, short_detail(detail))}")
            label.setToolTip(detail if len(" ".join(str(detail).split())) > STEP_DETAIL_CHARS else "")
            if state == "failed" and detail:
                self.status.setText(f"{STEP_ACTIONS[step]} failed: {detail}")
        segmented = self.step_states["segment"][0] == "done"
        tracking = self.step_states["track"][0] == "running"
        measuring = self.step_states["measure"][0] == "running"
        self.track_button.setEnabled(segmented and not tracking and not measuring)
        self.measure_button.setEnabled(segmented and self.step_states["track"][0] == "done" and not tracking and not measuring)

    def _fill_candidates(self, results: SeedingResults):
        """The cells list: the two sections from per_track; the candidates table when per_track has no landings to list;
        one disabled line when there is nothing."""
        sections = [(title, rows) for title, rows in cells_to_follow(results) if rows]
        if sections:
            for title, rows in sections:
                self._add_section_title(title)
                for row in rows[:MAX_LIST_ROWS]:
                    self._add_track_item(track_line(row), int(row["track_id"]))
                if len(rows) > MAX_LIST_ROWS:
                    self._add_section_title(f"and {len(rows) - MAX_LIST_ROWS:,} more")
        else:
            cand = results.candidates
            if cand is None or not cand.height:
                self.candidates.addItem("No tracks to follow yet")
                self.candidates.item(0).setFlags(Qt.NoItemFlags)
                return
            self._add_section_title("worth following")
            for row in cand.iter_rows(named=True):
                self._add_track_item(candidate_line(row), int(row["track_id"]))
        self.candidates.setCurrentRow(self._first_track_row(0))

    def _add_section_title(self, title: str):
        item = QListWidgetItem(title)
        item.setFlags(Qt.ItemIsEnabled)                  # readable, never selectable
        font = QFont(item.font()); font.setBold(True); item.setFont(font)
        self.candidates.addItem(item)

    def _add_track_item(self, text: str, track_id: int):
        item = QListWidgetItem(text); item.setData(Qt.UserRole, int(track_id))
        self.candidates.addItem(item)

    def _first_track_row(self, start: int) -> int:
        """The first row at or after start that is a track (not a section title), or start."""
        for r in range(start, self.candidates.count()):
            if self.candidates.item(r).data(Qt.UserRole) is not None:
                return r
        return start

    def _render_all(self):
        span = self.info.span if self.info is not None else None
        settings = self.results.settings if self.results is not None else SeedingSettings()
        for canvas in self.panels:
            canvas.render(self.results, span, self.strip.cursor_h, settings, selected=self.selected_dot)

    def _render_landing(self):
        span = self.info.span if self.info is not None else None
        settings = self.results.settings if self.results is not None else SeedingSettings()
        for canvas in self.panels:
            if canvas.key == LANDING_PANEL_KEY:
                canvas.render(self.results, span, self.strip.cursor_h, settings, selected=self.selected_dot)

    def _schedule_redraw(self):
        """The time panels follow the cursor, at most once every REDRAW_EVERY_MS."""
        if self.results is None or not self.panels:
            return
        self.pending_hour = self.strip.cursor_h
        if not self.redraw_timer.isActive():
            self.redraw_timer.start()

    def _redraw_time_panels(self):
        """The time panels take the cursor hour: a blit of the cursor line where the canvas owns it, a redraw otherwise."""
        if self.pending_hour is None or self.results is None:
            return
        self.pending_hour = None
        span = self.info.span if self.info is not None else None
        h = self.strip.cursor_h
        for canvas in self.panels:
            if canvas.time_panel and not canvas.move_cursor(h):
                canvas.render(self.results, span, h, self.results.settings, selected=self.selected_dot)

    def flush_redraws(self):
        """Draw whatever the cursor left pending now (the tests and the image grab use it)."""
        if self.redraw_timer.isActive():
            self.redraw_timer.stop()
        self._redraw_time_panels()

    def time_panels(self) -> list:
        return [c for c in self.panels if c.time_panel]

    # ------------------------------------------------------------------------------------------ user actions
    def _on_strip(self, h: float):
        self._schedule_redraw()
        self.hour_changed.emit(float(h))

    def selected_track(self) -> Optional[int]:
        """The track of the selected row; a section title stands for the first track under it."""
        row = self.candidates.currentRow()
        if row < 0:
            return None
        item = self.candidates.item(self._first_track_row(row))
        tid = item.data(Qt.UserRole) if item is not None else None
        return None if tid is None else int(tid)

    def _on_list_selection(self, current, _previous):
        tid = None if current is None else current.data(Qt.UserRole)
        tid = None if tid is None else int(tid)
        if tid != self.selected_dot and self.results is not None:
            self.selected_dot = tid
            self._render_landing()

    def _follow(self):
        tid = self.selected_track()
        if tid is None:
            return
        self.followed_track = tid
        self.followed_span = self.track_span(tid)
        self.follow_requested.emit(tid)

    def _field_film(self):
        self.window_requested.emit("Field", self.selected_track())

    def select_track(self, track_id: int) -> bool:
        """Select the list row of this track; False when the track is not listed."""
        for r in range(self.candidates.count()):
            if self.candidates.item(r).data(Qt.UserRole) == int(track_id):
                self.candidates.setCurrentRow(r)
                return True
        return False

    def _on_landing_press(self, canvas: PanelCanvas, x: float, y: float):
        """A click on the landing panel: the nearest dot within PICK_TOLERANCE_PX picks that track, which is selected in
        the list, ringed in the panel and followed."""
        if self.results is None or canvas.ax is None:
            return
        tid = SP.pick_landing(canvas.ax, self.results, x, y, tol_px=PICK_TOLERANCE_PX)
        if tid is None:
            return
        tid = int(tid)
        if not self.select_track(tid):
            self.selected_dot = tid
            self._render_landing()
        self.followed_track = tid
        self.followed_span = self.track_span(tid)
        self.track_picked.emit(tid)
        self.follow_requested.emit(tid)

    def track_span(self, track_id: int) -> Optional[tuple]:
        """The followed track's hours from per_track (t_start_h, t_end_h), or None."""
        if self.results is None or self.results.per_track is None or not self.results.per_track.height:
            return None
        pt = self.results.per_track
        if "track_id" not in pt.columns or "t_start_h" not in pt.columns or "t_end_h" not in pt.columns:
            return None
        rows = pt.filter(pt["track_id"] == track_id)
        if not rows.height:
            return None
        return float(rows["t_start_h"][0]), float(rows["t_end_h"][0])

    def open_numbers(self) -> Optional[NumbersDialog]:
        """The Numbers table of every metric in a plain dialog; refilled with the current results."""
        if self.results is None:
            return None
        if self.numbers_dialog is None:
            self.numbers_dialog = NumbersDialog(self)
        self.numbers_dialog.fill(self.results.metrics)
        self.numbers_dialog.show(); self.numbers_dialog.raise_()
        return self.numbers_dialog

    # ------------------------------------------------------------------------------------------ playing
    def _toggle_strip_play(self):
        """Play moves the cursor through the recording at HOURS_PER_SECOND and emits hour_changed at every step, so the
        view area and the time panels follow; a second click pauses it where it is."""
        if self.strip_timer.isActive():
            self._stop_strip_play(); return
        span = self.info.span if self.info is not None else None
        if span is None:
            return
        self.play_timer.stop(); self.play_button.setText("Play film")
        if self.strip.cursor_h >= span[1] - 1e-9:
            self.strip.set_hour(span[0], emit=True)
        self.last_tick = time.monotonic()
        self.strip_play_button.setText("Pause"); self.strip_timer.start()

    def _stop_strip_play(self):
        self.strip_timer.stop(); self.strip_play_button.setText("Play")

    def _strip_play_step(self):
        """One tick: advance by the seconds really elapsed (a slow repaint skips ahead rather than slowing the playback)."""
        span = self.info.span if self.info is not None else None
        if span is None:
            self._stop_strip_play(); return
        now = time.monotonic()
        elapsed = PLAY_TICK_MS / 1000.0 if self.last_tick is None else max(now - self.last_tick, PLAY_TICK_MS / 1000.0)
        self.last_tick = now
        h = self.strip.cursor_h + HOURS_PER_SECOND * elapsed
        if h >= span[1]:
            self.strip.set_hour(span[1], emit=True); self._stop_strip_play(); return
        self.strip.set_hour(h, emit=True)

    def _toggle_play(self):
        """Play film moves the cursor through the followed cell's life (about 5 s of playback) and emits hour_changed at
        every step, so the view area follows. A second click stops it."""
        if self.play_timer.isActive():
            self.play_timer.stop(); self.play_button.setText("Play film"); return
        span = self.followed_span or (self.info.span if self.info is not None else None)
        if span is None:
            return
        self._stop_strip_play()
        self.play_hours = list(np.linspace(span[0], span[1], FILM_STEPS)); self.play_index = 0
        self.play_button.setText("Stop film"); self.play_timer.start()

    def _play_step(self):
        if self.play_index >= len(self.play_hours):
            self.play_timer.stop(); self.play_button.setText("Play film"); return
        self.strip.set_hour(self.play_hours[self.play_index], emit=True); self.play_index += 1
