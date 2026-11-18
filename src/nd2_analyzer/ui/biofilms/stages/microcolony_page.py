"""The Microcolonies page of the Biofilm Stages tab, built like the Seeding page: the steps with a button, the time strip
with its cursor and Play, the microcolonies to follow with Follow and the film strip, eight cards, two plots with a picker.
Every number comes from the microcolonies the tool followed on the loaded recording (microcolony_follow) or, for the
plots under the line of the picker, from the metrics it measured on every segmented position (microcolony_metrics); the
page only shows them. The controller (microcolony_controller) feeds it."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
                               QVBoxLayout, QWidget)

from nd2_analyzer.analysis.biofilm_stages import growth_plots as GP
from nd2_analyzer.analysis.biofilm_stages import microcolony_metric_plots as XP
from nd2_analyzer.analysis.biofilm_stages import microcolony_plots as MP
from nd2_analyzer.analysis.biofilm_stages import microcolony_film as MFilm
from nd2_analyzer.analysis.biofilm_stages.microcolony_film import draw_film_strip
from .seeding_page import LEFT_COLUMN_PX, FactCard, PlotCanvas, StripCanvas, step_words

class MicrocolonyStrip(StripCanvas):
    """The strip with microcolony formation and maturation split at Th1, once the Maturation tab has measured it."""

    def __init__(self, parent=None):
        self.th1_h = None
        super().__init__(parent)

    def set_th1(self, th1_h):
        self.th1_h = None if th1_h is None else float(th1_h); self.redraw()

    def redraw(self):
        if self.span is None:
            return super().redraw()
        self.ax.clear(); self.ax.axis("on"); MP.draw_strip(self.ax, self.span, self.cursor_h, self.th1_h); self.draw_idle()


STEPS = ("load", "segment", "track", "measure", "stain")
STEP_NAMES = {"load": "Raw Frames", "segment": "Segmented Frames", "track": "Microcolonies", "measure": "All positions", "stain": "Matrix stain"}
PLAY_TICK_MS = 250             # Play on the strip: one frame of the recording every quarter of a second
RULE_WORDS = "A microcolony: cells less than 0.5 µm apart, 4 or more of them, still there 1 h later."


class MicrocolonyPage(QWidget):
    name = "Microcolonies"
    hour_changed = Signal(float)
    track_requested = Signal()
    window_requested = Signal(object)          # the selected microcolony
    selection_changed = Signal(object)
    export_requested = Signal(str)             # "film", "image" or "numbers"
    compare_requested = Signal()               # manual selection next to automatic detection, on the frame at the cursor
    measure_requested = Signal()               # the metrics of every segmented position

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracks = None; self.film = None; self.metrics = None; self.hour = 0.0
        self.stain = None                          # matrix stain of each followed microcolony (summary), once measured
        self.growth = None                         # the Maturation tab's measure, for the plot of the stain inside and outside microcolonies
        self.step_states = {s: ("todo", "") for s in STEPS}
        self.play_timer = QTimer(self); self.play_timer.setInterval(PLAY_TICK_MS); self.play_timer.timeout.connect(self._play_step)
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 4, 6, 4); lay.setSpacing(4)
        self.recording_label = QLabel("No recording loaded"); self.recording_label.setWordWrap(True); lay.addWidget(self.recording_label)
        lay.addLayout(self._build_steps_row()); lay.addLayout(self._build_strip_row()); lay.addLayout(self._build_middle_row(), 1)
        self.status = QLabel("Load a recording and segment it, then click Track.  " + RULE_WORDS); self.status.setWordWrap(True); lay.addWidget(self.status)
        self.film_canvas.render(draw_film_strip, None)
        self._render_plots()

    # ------------------------------------------------------------------ building
    def _build_steps_row(self) -> QHBoxLayout:
        row = QHBoxLayout(); row.setSpacing(16); self.step_labels = {}
        for step in STEPS:
            lab = QLabel(); self.step_labels[step] = lab; row.addWidget(lab)
        row.addStretch(1)
        row.addWidget(QLabel("Follow until:"))
        self.until_spin = QDoubleSpinBox(); self.until_spin.setDecimals(1); self.until_spin.setSingleStep(0.5); self.until_spin.setSuffix(" h"); self.until_spin.setEnabled(False)
        self.until_spin.setToolTip("The last hour that is followed: stop before the frames that are out of focus")
        row.addWidget(self.until_spin)
        self.track_button = QPushButton("Track"); self.track_button.setEnabled(False)
        self.track_button.setToolTip("Finds the microcolonies of every segmented frame and follows them through time, with their merges.  " + RULE_WORDS)
        self.track_button.clicked.connect(self.track_requested.emit); row.addWidget(self.track_button)
        self._refresh_steps()
        return row

    def _build_strip_row(self) -> QHBoxLayout:
        row = QHBoxLayout(); row.setSpacing(8)
        self.strip = MicrocolonyStrip(); self.strip.hour_changed.connect(self._on_strip); row.addWidget(self.strip, 1)
        self.strip_play_button = QPushButton("Play"); self.strip_play_button.setEnabled(False)
        self.strip_play_button.setToolTip("Moves the cursor through the followed frames; the view area and the plots follow")
        self.strip_play_button.clicked.connect(self._toggle_play); row.addWidget(self.strip_play_button, 0, Qt.AlignVCenter)
        return row

    def _build_middle_row(self) -> QHBoxLayout:
        row = QHBoxLayout(); row.setSpacing(8)
        left = QVBoxLayout(); left.setSpacing(4); left.addWidget(QLabel("Microcolonies to follow"))
        self.candidates = QListWidget(); self.candidates.setMinimumHeight(100); self.candidates.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.candidates.itemDoubleClicked.connect(lambda _item: self._follow()); self.candidates.currentItemChanged.connect(self._on_selection)
        left.addWidget(self.candidates, 1)
        buttons = QGridLayout(); buttons.setSpacing(4)
        self.follow_button = QPushButton("Follow"); self.follow_button.setToolTip("Opens the Microcolonies window on the selected microcolony: its film and its merging history")
        self.follow_button.clicked.connect(self._follow)
        self.play_button = QPushButton("Play film"); self.play_button.setToolTip("Moves the cursor through the selected microcolony's frames")
        self.play_button.clicked.connect(self._toggle_film_play)
        self.film_button = QPushButton("Save film"); self.film_button.clicked.connect(lambda: self.export_requested.emit("film"))
        self.image_button = QPushButton("Save image"); self.image_button.clicked.connect(lambda: self.export_requested.emit("image"))
        for k, b in enumerate((self.follow_button, self.play_button, self.film_button, self.image_button)):
            b.setEnabled(False); buttons.addWidget(b, k // 2, k % 2)
        self.compare_button = QPushButton("Manual selection"); self.compare_button.setEnabled(False)
        self.compare_button.setToolTip("Draw outlines around the microcolonies you see on the frame at the cursor, next to what the tool found")
        self.compare_button.clicked.connect(self.compare_requested.emit); buttons.addWidget(self.compare_button, 2, 0)
        self.numbers_button = QPushButton("Numbers"); self.numbers_button.setEnabled(False)
        self.numbers_button.setToolTip("Writes the numbers behind every plot of this page as CSV files into a folder you choose")
        self.numbers_button.clicked.connect(lambda: self.export_requested.emit("numbers")); buttons.addWidget(self.numbers_button, 2, 1)
        left.addLayout(buttons)
        self.film_canvas = PlotCanvas(size=(3.3, 2.45), min_height=235); self.film_canvas.setMaximumHeight(255); left.addWidget(self.film_canvas)   # two rows of three frames
        self.film_caption = QLabel(""); self.film_caption.setAlignment(Qt.AlignCenter); left.addWidget(self.film_caption)   # which microcolony the strip shows
        left_box = QWidget(); left_box.setLayout(left); left_box.setFixedWidth(LEFT_COLUMN_PX); row.addWidget(left_box)
        right = QVBoxLayout(); right.setSpacing(6)
        grid = QGridLayout(); grid.setSpacing(6); self.cards = []
        for i in range(8):
            card = FactCard(); card.setEnabled(False); self.cards.append(card); grid.addWidget(card, i // 4, i % 4)
        right.addLayout(grid)
        picker = QHBoxLayout(); picker.setSpacing(8); picker.addWidget(QLabel("Plots:")); self.plot_combos = []
        for default in MP.FIGURE_DEFAULTS:
            combo = QComboBox()
            for key, title in MP.FIGURES:
                combo.addItem(title, key)
            combo.insertSeparator(combo.count())                   # under the line: every position, each frame by itself
            for key, title in XP.FIGURES + (XP.STAIN,):
                combo.addItem(title, key)
            combo.setCurrentIndex(max(0, combo.findData(default))); combo.currentIndexChanged.connect(lambda _i: self._render_plots())
            self.plot_combos.append(combo); picker.addWidget(combo)
        picker.addStretch(1)
        picker.addWidget(QLabel("All positions, every:"))
        self.every_spin = QDoubleSpinBox(); self.every_spin.setDecimals(2); self.every_spin.setSuffix(" h"); self.every_spin.setRange(0.01, 1000.0); self.every_spin.setEnabled(False)
        self.every_spin.setToolTip("The time between two measured frames, from the first frame to the hour of \"Follow until\"")
        picker.addWidget(self.every_spin)
        self.measure_button = QPushButton("Measure"); self.measure_button.setEnabled(False)
        self.measure_button.setToolTip("Measures every segmented position for the plots under the line of the picker: cells per microcolony, local "
                                       "density, spacing, alignment, largest microcolony, area.  " + XP.RULE_NOTE.capitalize() + ".")
        self.measure_button.clicked.connect(self.measure_requested.emit); picker.addWidget(self.measure_button)
        right.addLayout(picker)
        plots = QHBoxLayout(); plots.setSpacing(6); self.panels = []
        for _default in MP.FIGURE_DEFAULTS:
            canvas = PlotCanvas(size=MP.FIGURE_SIZE_IN, min_height=330); canvas.mpl_connect("button_press_event", lambda ev, c=canvas: self._on_plot_press(c, ev))
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

    def set_track_enabled(self, on: bool):
        self.track_button.setEnabled(bool(on)); self.until_spin.setEnabled(bool(on))
        self.compare_button.setEnabled(self.step_states["segment"][0] == "done")

    def set_until(self, first_h: float, last_h: float, step_h: float, value: Optional[float] = None):
        self.until_spin.blockSignals(True); self.until_spin.setRange(first_h, last_h); self.until_spin.setSingleStep(step_h)
        self.until_spin.setValue(last_h if value is None else value); self.until_spin.blockSignals(False)

    def set_measure_enabled(self, on: bool):
        self.measure_button.setEnabled(bool(on)); self.every_spin.setEnabled(bool(on))

    def set_every(self, step_h: float, value_h: Optional[float] = None):
        """The frame step of the recording is the smallest choice and the size of one click."""
        self.every_spin.blockSignals(True); self.every_spin.setMinimum(step_h); self.every_spin.setSingleStep(step_h)
        self.every_spin.setValue(step_h if value_h is None else value_h); self.every_spin.blockSignals(False)

    def set_th1(self, th1_h):
        """Th1 of the recording (the Maturation tab's measure), or None: the strip then shows the two stages."""
        self.strip.set_th1(th1_h)

    def set_metrics(self, metrics):
        """The metrics of every measured position (None clears their plots)."""
        self.metrics = metrics; self._enable_buttons(); self._render_plots()

    def _fill_list(self):
        """The microcolonies to follow, each with its matrix stain once that is measured."""
        self.candidates.clear(); bold = QFont(self.candidates.font()); bold.setBold(True)
        for title, rows in MP.entries(self.tracks, self.stain):
            head = QListWidgetItem(title); head.setFont(bold); head.setFlags(Qt.ItemIsEnabled); self.candidates.addItem(head)
            for identity, line in rows:
                item = QListWidgetItem(line); item.setData(Qt.UserRole, int(identity)); self.candidates.addItem(item)

    def set_stain(self, summary):
        """When each followed microcolony gets its matrix stain and how much of it is covered (None: not measured). The
        list says it under each microcolony; the selection stays."""
        self.stain = summary
        if self.tracks is None:
            return
        kept = self.selected(); self.candidates.blockSignals(True); self._fill_list()
        row = next((r for r in range(self.candidates.count()) if self.candidates.item(r).data(Qt.UserRole) == kept), -1)
        if row >= 0:
            self.candidates.setCurrentRow(row)
        self.candidates.blockSignals(False); self._render_film()

    def set_growth(self, growth):
        """The Maturation tab's measure (or None): the plot "Matrix stain over time" draws it."""
        self.growth = growth; self._render_plots()

    def refresh_film(self):
        self._render_film()

    def set_tracks(self, tracks, film=None):
        """The followed microcolonies of the recording (None clears the page)."""
        self.tracks, self.film = tracks, film; self.stain = None; self.candidates.clear(); self.play_timer.stop(); self.strip_play_button.setText("Play")
        if tracks is None:
            for card in self.cards:
                card.show_fact("", "", False)
        else:
            self._fill_list()
            for card, (value, meaning, ok) in zip(self.cards, MP.cards(tracks)):
                card.show_fact(value, meaning, ok)
            first = next((r for r in range(self.candidates.count()) if self.candidates.item(r).data(Qt.UserRole) is not None), -1)
            if first >= 0:
                self.candidates.setCurrentRow(first)
        self.strip_play_button.setEnabled(tracks is not None)
        self._enable_buttons(); self._render_plots(); self._render_film()

    def selected(self) -> Optional[int]:
        item = self.candidates.currentItem(); identity = item.data(Qt.UserRole) if item is not None else None
        return int(identity) if identity is not None else None

    def select(self, identity: int) -> bool:
        for r in range(self.candidates.count()):
            if self.candidates.item(r).data(Qt.UserRole) == int(identity):
                self.candidates.setCurrentRow(r); return True
        return False

    def set_hour(self, h: float):
        self.hour = float(h); self.strip.set_hour(h); self._render_plots(); self._render_film()

    # ------------------------------------------------------------------ what the user does
    def _enable_buttons(self):
        on = self.tracks is not None and self.selected() is not None
        for b in (self.follow_button, self.play_button, self.film_button, self.image_button):
            b.setEnabled(on)
        self.numbers_button.setEnabled(self.tracks is not None or self.metrics is not None)

    def _on_selection(self, _current=None, _previous=None):
        identity = self.selected()
        if identity is not None and self.film is not None:
            self.film.set_root(identity)
        self._enable_buttons(); self._render_plots(); self._render_film(); self.selection_changed.emit(identity)

    def _follow(self):
        if self.selected() is not None:
            self.window_requested.emit(self.selected())

    def _on_strip(self, h: float):
        self.hour = float(h); self._render_plots(); self._render_film(); self.hour_changed.emit(self.hour)

    def _on_plot_press(self, canvas, event):
        """A click on a bar of the merging history picks that microcolony; a click on any time plot moves the cursor."""
        if event.inaxes is None or event.xdata is None or self.strip.span is None:
            return
        rows = getattr(event.inaxes, "_microcolony_rows", None)
        if rows and event.ydata is not None and self.tracks is not None:
            identity = min(rows, key=lambda i: abs(rows[i] - event.ydata))
            if abs(rows[identity] - event.ydata) <= 0.6 and self.select(identity):
                return
        if getattr(event.inaxes, "_time_axis", False):
            self.strip.set_hour(min(max(event.xdata, self.strip.span[0]), self.strip.span[1]), emit=True)

    def _frame_hours(self) -> list:
        return [self.tracks.hours[t] for t in self.tracks.frames] if self.tracks is not None else []

    def _toggle_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop(); self.strip_play_button.setText("Play"); return
        self._play_hours = self._frame_hours(); self.strip_play_button.setText("Pause"); self.play_timer.start()

    def _toggle_film_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop(); self.strip_play_button.setText("Play"); return
        if self.film is not None and self.film.frames:
            self._play_hours = [self.tracks.hours[t] for t in self.film.frames]; self.strip.set_hour(self._play_hours[0], emit=True)
            self.strip_play_button.setText("Pause"); self.play_timer.start()

    def _play_step(self):
        later = [h for h in getattr(self, "_play_hours", []) if h > self.hour + 1e-9]
        if not later:
            self.play_timer.stop(); self.strip_play_button.setText("Play"); return
        self.strip.set_hour(later[0], emit=True)

    # ------------------------------------------------------------------ drawing
    def _render_plots(self):
        for combo, canvas in zip(self.plot_combos, self.panels):
            key = combo.currentData(); of_all = key in XP.DRAW; of_stain = key == XP.STAIN[0]
            needed = (self.growth if self.growth is not None and GP.has_matrix(self.growth) else None) if of_stain else self.metrics if of_all else self.tracks
            if needed is None:
                canvas.figure.clear(); ax = canvas.figure.add_axes([0.06, 0, 0.88, 1]); ax.axis("off")
                ax.text(0.5, 0.5, XP.STAIN_FIRST if of_stain else XP.MEASURE_FIRST if of_all else "Track the microcolonies to see this plot", ha="center",
                        va="center", color="#777777", transform=ax.transAxes, wrap=True); canvas.draw_idle()
            elif of_stain:
                canvas.render(GP.draw_inside, self.growth, self.hour, None)
            elif of_all:
                canvas.render(XP.DRAW[key], self.metrics, self.hour)
            else:
                canvas.render(MP.DRAW[key], self.tracks, self.hour, self.selected())

    def _render_film(self):
        now = None
        if self.tracks is not None:
            now = min(self.tracks.frames, key=lambda t: abs(self.tracks.hours[t] - self.hour))
        shown = self.film if self.tracks is not None and self.selected() is not None else None
        self.film_canvas.render(draw_film_strip, shown, 6, now)
        merged = len(shown.order) - 1 if shown is not None and shown.root is not None else 0
        self.film_caption.setText("" if shown is None or shown.root is None else MFilm.heading_words(shown.root, drawn=merged, later=0))   # the words of the Follow window
