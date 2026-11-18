"""The Maturation page of the biofilm stages: the growth states, built like the Seeding and Microcolonies pages. The steps
with their buttons (the dye frames, Measure), the strip with the states, Th1, Th2 and the cursor, the positions to follow,
eight cards (the thresholds, then four numbers at the cursor), two plots with a picker. Every number comes from the
growth-states measure of the loaded recording (growth_measure); the controller (growth_controller) feeds the page."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
                               QVBoxLayout, QWidget)

from nd2_analyzer.analysis.biofilm_stages import growth_plots as GP
from nd2_analyzer.analysis.biofilm_stages.growth_film import draw_film_strip
from .seeding_page import LEFT_COLUMN_PX, FactCard, PlotCanvas, StripCanvas, step_words

STEPS = ("load", "segment", "dye", "measure")
STEP_NAMES = {"load": "Raw Frames", "segment": "Segmented Frames", "dye": "Dye Frames", "measure": "Growth States"}
PLAY_TICK_MS = 250
STATE_WORDS = "State I ends at Th1, when the matrix stain appears and stays. State II ends at Th2, when growth stops."


class GrowthStrip(StripCanvas):
    """The strip with the growth states, Th1 and Th2 instead of the seeding stages."""

    def __init__(self, parent=None):
        self.measured = None
        super().__init__(parent)

    def set_measured(self, measured):
        self.measured = measured; self.redraw()

    def redraw(self):
        if self.span is None:
            return super().redraw()
        self.ax.clear(); self.ax.axis("on"); GP.draw_strip(self.ax, self.measured, self.span, self.cursor_h); self.draw_idle()


class GrowthPage(QWidget):
    name = "Maturation"
    hour_changed = Signal(float)
    dye_requested = Signal()
    measure_requested = Signal()
    window_requested = Signal(object)          # the selected position
    selection_changed = Signal(object)
    export_requested = Signal(str)             # "film" or "image"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.shared = None                         # see set_shared
        self.measured = None; self.film = None; self.hour = 0.0
        self.step_states = {s: ("todo", "") for s in STEPS}
        self.play_timer = QTimer(self); self.play_timer.setInterval(PLAY_TICK_MS); self.play_timer.timeout.connect(self._play_step)
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 4, 6, 4); lay.setSpacing(4)
        self.recording_label = QLabel("No recording loaded"); self.recording_label.setWordWrap(True); lay.addWidget(self.recording_label)
        lay.addLayout(self._build_steps_row()); lay.addLayout(self._build_strip_row()); lay.addLayout(self._build_middle_row(), 1)
        self.status = QLabel("Segment the positions you keep, give the dye frames, then click Measure.  " + STATE_WORDS); self.status.setWordWrap(True); lay.addWidget(self.status)
        self.film_canvas.render(draw_film_strip, None); self._render_plots()

    # ------------------------------------------------------------------ building
    def _build_steps_row(self) -> QHBoxLayout:
        row = QHBoxLayout(); row.setSpacing(16); self.step_labels = {}
        for step in STEPS:
            lab = QLabel(); self.step_labels[step] = lab; row.addWidget(lab)
        row.addStretch(1)
        row.addWidget(QLabel("Measure every:"))
        self.every_spin = QDoubleSpinBox(); self.every_spin.setDecimals(2); self.every_spin.setSuffix(" h"); self.every_spin.setRange(0.01, 1000.0); self.every_spin.setEnabled(False)
        self.every_spin.setToolTip("The time between two measured frames; the frame step of the recording measures every frame")
        row.addWidget(self.every_spin)
        self.dye_button = QPushButton("Dye frames..."); self.dye_button.setEnabled(False)
        self.dye_button.setToolTip("Two small files of the same run: the frame taken before the dye was added and the one right after. "
                                   "They zero and flatten the dye image, so the matrix stain can be measured")
        self.dye_button.clicked.connect(self.dye_requested.emit); row.addWidget(self.dye_button)
        self.measure_button = QPushButton("Measure"); self.measure_button.setEnabled(False)
        self.measure_button.setToolTip("Counts the cells and measures the matrix stain of every segmented position, then finds Th1 and Th2.  " + STATE_WORDS)
        self.measure_button.clicked.connect(self.measure_requested.emit); row.addWidget(self.measure_button)
        self._refresh_steps()
        return row

    def _build_strip_row(self) -> QHBoxLayout:
        row = QHBoxLayout(); row.setSpacing(8)
        self.strip = GrowthStrip(); self.strip.hour_changed.connect(self._on_strip); row.addWidget(self.strip, 1)
        self.strip_play_button = QPushButton("Play"); self.strip_play_button.setEnabled(False)
        self.strip_play_button.setToolTip("Moves the cursor through the measured frames; the view area, the cards and the plots follow")
        self.strip_play_button.clicked.connect(self._toggle_play); row.addWidget(self.strip_play_button, 0, Qt.AlignVCenter)
        return row

    def _build_middle_row(self) -> QHBoxLayout:
        row = QHBoxLayout(); row.setSpacing(8)
        left = QVBoxLayout(); left.setSpacing(4); left.addWidget(QLabel("Positions to follow"))
        self.candidates = QListWidget(); self.candidates.setMinimumHeight(100); self.candidates.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.candidates.itemDoubleClicked.connect(lambda _item: self._follow()); self.candidates.currentItemChanged.connect(self._on_selection)
        left.addWidget(self.candidates, 0)                     # as tall as its positions: the rest of the column is the frames
        buttons = QGridLayout(); buttons.setSpacing(4)
        self.follow_button = QPushButton("Follow"); self.follow_button.setToolTip("Opens the Maturation window on the selected position: its film, its cells and its matrix stain")
        self.follow_button.clicked.connect(self._follow)
        self.film_button = QPushButton("Save film"); self.film_button.clicked.connect(lambda: self.export_requested.emit("film"))
        self.image_button = QPushButton("Save image"); self.image_button.clicked.connect(lambda: self.export_requested.emit("image"))
        self.numbers_button = QPushButton("Numbers")
        self.numbers_button.setToolTip("Writes the numbers behind every plot of this page as CSV files into a folder you choose")
        self.numbers_button.clicked.connect(lambda: self.export_requested.emit("numbers"))
        for k, b in enumerate((self.follow_button, self.film_button, self.image_button, self.numbers_button)):
            b.setEnabled(False)
        buttons.addWidget(self.follow_button, 0, 0); buttons.addWidget(self.numbers_button, 0, 1)
        buttons.addWidget(self.film_button, 1, 0); buttons.addWidget(self.image_button, 1, 1)
        left.addLayout(buttons)
        self.film_canvas = PlotCanvas(size=(3.3, 4.8), min_height=340); left.addWidget(self.film_canvas, 1)     # the frames of Figure 5 a, large
        left_box = QWidget(); left_box.setLayout(left); left_box.setFixedWidth(LEFT_COLUMN_PX); row.addWidget(left_box)
        right = QVBoxLayout(); right.setSpacing(6)
        grid = QGridLayout(); grid.setSpacing(6); self.cards = []
        for i in range(8):
            card = FactCard(); card.setEnabled(False); self.cards.append(card); grid.addWidget(card, i // 4, i % 4)
        right.addLayout(grid)
        picker = QHBoxLayout(); picker.setSpacing(8); picker.addWidget(QLabel("Plots:")); self.plot_combos = []
        for default in GP.FIGURE_DEFAULTS:
            combo = QComboBox()
            for key, title in GP.FIGURES:
                combo.addItem(title, key)
            combo.setCurrentIndex(max(0, combo.findData(default))); combo.currentIndexChanged.connect(lambda _i: self._render_plots())
            self.plot_combos.append(combo); picker.addWidget(combo)
        picker.addStretch(1); right.addLayout(picker)
        plots = QHBoxLayout(); plots.setSpacing(6); self.panels = []
        for _default in GP.FIGURE_DEFAULTS:
            canvas = PlotCanvas(size=GP.FIGURE_SIZE_IN, min_height=330); canvas.mpl_connect("button_press_event", self._on_plot_press)
            self.panels.append(canvas); plots.addWidget(canvas, 1)
        right.addLayout(plots, 1)
        right_box = QWidget(); right_box.setLayout(right); row.addWidget(right_box, 1)
        return row

    # ------------------------------------------------------------------ state
    def set_recording(self, words: str, span: Optional[tuple]):
        self.recording_label.setText(words or "No recording loaded"); self.strip.set_span(span)
        if span is not None:
            self.hour = min(max(self.hour, span[0]), span[1]); self.strip.set_hour(self.hour)

    def set_step_state(self, step: str, state: str, detail: str = ""):
        self.step_states[step] = (state, detail); self._refresh_steps()

    def _refresh_steps(self):
        for step, label in self.step_labels.items():
            state, detail = self.step_states[step]; label.setText(f"<b>{STEP_NAMES[step]}:</b> {step_words(state, detail)}")

    def set_measure_enabled(self, on: bool):
        self.measure_button.setEnabled(bool(on)); self.dye_button.setEnabled(bool(on)); self.every_spin.setEnabled(bool(on))

    def set_every(self, step_h: float, value_h: Optional[float] = None):
        """The frame step of the recording is the smallest choice and the size of one click."""
        self.every_spin.blockSignals(True); self.every_spin.setMinimum(step_h); self.every_spin.setSingleStep(step_h)
        self.every_spin.setValue(step_h if value_h is None else value_h); self.every_spin.blockSignals(False)

    def set_shared(self, shared):
        """What another page measured and this page's plots also draw: {"cells_in_pct": (hours, mean, s.d.)} from the
        Microcolonies tab's measure of every position (Figure 5 b), or None."""
        self.shared = shared; self._render_plots()

    def set_measured(self, measured, film=None):
        """The growth-states measure of the recording (None clears the page)."""
        self.measured, self.film = measured, film; self.candidates.clear(); self.play_timer.stop(); self.strip_play_button.setText("Play")
        self.strip.set_measured(measured)
        if measured is not None:
            bold = QFont(self.candidates.font()); bold.setBold(True)
            for title, rows in GP.entries(measured):
                head = QListWidgetItem(title); head.setFont(bold); head.setFlags(Qt.ItemIsEnabled); self.candidates.addItem(head)
                for position, line in rows:
                    item = QListWidgetItem(line); item.setData(Qt.UserRole, int(position)); self.candidates.addItem(item)
            rows = sum(self.candidates.sizeHintForRow(r) for r in range(self.candidates.count())) + 2 * self.candidates.frameWidth() + 6
            self.candidates.setFixedHeight(min(max(rows, 60), 240))      # the list as tall as its positions; the frames get the rest
            first = next((r for r in range(self.candidates.count()) if self.candidates.item(r).data(Qt.UserRole) is not None), -1)
            if first >= 0:
                self.candidates.setCurrentRow(first)
        self.strip_play_button.setEnabled(measured is not None)
        self._enable_buttons(); self._render_cards(); self._render_plots(); self._render_film()

    def selected(self) -> Optional[int]:
        item = self.candidates.currentItem(); position = item.data(Qt.UserRole) if item is not None else None
        return int(position) if position is not None else None

    def set_hour(self, h: float):
        self.hour = float(h); self.strip.set_hour(h); self._render_cards(); self._render_plots(); self._render_film()

    # ------------------------------------------------------------------ what the user does
    def _enable_buttons(self):
        on = self.measured is not None and self.selected() is not None
        for b in (self.follow_button, self.film_button, self.image_button):
            b.setEnabled(on and (b is self.follow_button or self.film is not None))
        self.numbers_button.setEnabled(self.measured is not None)

    def _on_selection(self, _current=None, _previous=None):
        position = self.selected()
        if position is not None and self.film is not None:
            self.film.set_position(position)
        self._enable_buttons(); self._render_plots(); self._render_film(); self.selection_changed.emit(position)

    def _follow(self):
        if self.selected() is not None:
            self.window_requested.emit(self.selected())

    def _on_strip(self, h: float):
        self.hour = float(h); self._render_cards(); self._render_plots(); self._render_film(); self.hour_changed.emit(self.hour)

    def _on_plot_press(self, event):
        """A click on a time plot moves the cursor there."""
        if event.inaxes is None or event.xdata is None or self.strip.span is None:
            return
        if getattr(event.inaxes, "_time_axis", False):
            self.strip.set_hour(min(max(event.xdata, self.strip.span[0]), self.strip.span[1]), emit=True)

    def _toggle_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop(); self.strip_play_button.setText("Play"); return
        if self.measured is not None:
            self._play_hours = sorted({float(h) for p in self.measured.positions for h in self.measured.hours[p]})
            self.strip_play_button.setText("Pause"); self.play_timer.start()

    def _play_step(self):
        later = [h for h in getattr(self, "_play_hours", []) if h > self.hour + 1e-9]
        if not later:
            self.play_timer.stop(); self.strip_play_button.setText("Play"); return
        self.strip.set_hour(later[0], emit=True)

    # ------------------------------------------------------------------ drawing
    def _render_cards(self):
        if self.measured is None:
            for card in self.cards:
                card.show_fact("", "", False)
            return
        for card, (value, meaning, ok) in zip(self.cards, GP.cards(self.measured, self.hour)):
            card.show_fact(value, meaning, ok)

    def _render_plots(self):
        for combo, canvas in zip(self.plot_combos, self.panels):
            if self.measured is None:
                canvas.figure.clear(); ax = canvas.figure.add_axes([0, 0, 1, 1]); ax.axis("off")
                ax.text(0.5, 0.5, "Measure the growth states to see this plot", ha="center", va="center", color="#777777", transform=ax.transAxes); canvas.draw_idle()
            else:
                canvas.render(GP.DRAW[combo.currentData()], self.measured, self.hour, self.selected(), shared=self.shared)

    def _render_film(self):
        self.film_canvas.render(draw_film_strip, self.film if self.measured is not None and self.selected() is not None else None, 4, self.hour)
