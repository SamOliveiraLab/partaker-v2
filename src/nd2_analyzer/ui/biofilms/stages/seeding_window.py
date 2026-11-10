"""The Seeding window, laid out like the lineage window: a Cell list (irreversible cells first) and a View box (Cell or
Field) on top, the film on the left with its plot drawing itself on the right, Play and a slider under it, Save Film, Save
Image and Close. Click a cell in the image to follow it. Drawing is analysis.biofilm_stages.seeding_film, shared with the
export and the tests."""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QProgressDialog, QPushButton,
                               QSlider, QVBoxLayout)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from nd2_analyzer.analysis.biofilm_stages.seeding_film import FIGURE_IN, SeedingFilm


def track_entry(row: dict) -> str:
    """'Track 8462   lands 27.1 h, divides 27.5 h' and the like, for the Cell list."""
    fate = row.get("fate", "")
    end = float(row["t_end_h"])
    tail = (f"divides {end:.1f} h" if fate == "divided" else "stays to the end" if fate == "present at end"
            else f"left {end:.1f} h")
    return f"Track {int(row['track_id'])}   lands {float(row['t_start_h']):.1f} h, {tail}"


class SeedingDialog(QDialog):
    MS_PER_STEP = 150

    def __init__(self, film: SeedingFilm, sections: list, view: str = "Cell", root: Optional[int] = None,
                 open_numbers: Optional[Callable[[], None]] = None, parent=None):
        super().__init__(parent)
        self.film = film
        self.view = view if view in ("Cell", "Field") else "Cell"
        self.open_numbers = open_numbers
        self.step = 0
        self.setWindowTitle("Seeding")
        self.setAttribute(Qt.WA_DeleteOnClose)          # a closed window frees its figure and its film's label frames
        self.setMinimumWidth(1000); self.setMinimumHeight(700)
        self.init_ui(sections)
        start = root if root is not None else film.default_root()
        if start is not None:
            film.set_root(start)
        self.select_in_list(film.root)
        self.view_combo.blockSignals(True); self.view_combo.setCurrentText(self.view); self.view_combo.blockSignals(False)
        self.reset_slider(start_hour=None if self.view == "Cell" else self.film.lines_x0)

    # ---------------------------------------------------------------- ui
    def init_ui(self, sections: list):
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Cell:"))
        self.cell_combo = QComboBox(); self.cell_combo.setMinimumWidth(320)
        for _title, rows in sections:
            for row in rows:
                self.cell_combo.addItem(track_entry(row), int(row["track_id"]))
        self.cell_combo.currentIndexChanged.connect(self._on_cell_combo)
        top.addWidget(self.cell_combo)
        top.addSpacing(16)
        top.addWidget(QLabel("View:"))
        self.view_combo = QComboBox(); self.view_combo.addItems(["Cell", "Field"])
        self.view_combo.currentTextChanged.connect(self._on_view)
        top.addWidget(self.view_combo)
        top.addSpacing(16)
        self.hint = QLabel("Click a cell in the image to follow it.")
        top.addWidget(self.hint)
        top.addStretch()
        layout.addLayout(top)

        self.figure = Figure(figsize=FIGURE_IN)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        layout.addWidget(self.canvas, 1)
        self.ax_img = None

        row = QHBoxLayout()
        self.play_button = QPushButton("Play"); self.play_button.setCheckable(True); self.play_button.toggled.connect(self._on_play)
        row.addWidget(self.play_button)
        self.slider = QSlider(Qt.Horizontal); self.slider.valueChanged.connect(self._on_slider)
        row.addWidget(self.slider, 1)
        self.time_label = QLabel(""); self.time_label.setMinimumWidth(170)
        row.addWidget(self.time_label)
        layout.addLayout(row)

        buttons = QHBoxLayout()
        self.save_film_button = QPushButton("Save Film"); self.save_film_button.clicked.connect(self.save_film)
        buttons.addWidget(self.save_film_button)
        self.save_button = QPushButton("Save Image"); self.save_button.clicked.connect(self.save_image)
        buttons.addWidget(self.save_button)
        if self.open_numbers is not None:
            self.numbers_button = QPushButton("Numbers"); self.numbers_button.clicked.connect(self.open_numbers)
            buttons.addWidget(self.numbers_button)
        buttons.addStretch()
        self.close_button = QPushButton("Close"); self.close_button.clicked.connect(self.close)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

        self.timer = QTimer(self); self.timer.setInterval(self.MS_PER_STEP); self.timer.timeout.connect(self._step)

    # ---------------------------------------------------------------- state
    def frames(self) -> list:
        return self.film.cell_frames if self.view == "Cell" else self.film.field_frames

    def reset_slider(self, start_hour: Optional[float] = None):
        frames = self.frames()
        self.slider.blockSignals(True)
        self.slider.setMinimum(0); self.slider.setMaximum(max(len(frames) - 1, 0))
        if start_hour is not None and frames:
            target = self.film.frame_at(start_hour)
            self.step = min(range(len(frames)), key=lambda i: abs(frames[i] - target))
        else:
            self.step = 0
        self.slider.setValue(self.step)
        self.slider.blockSignals(False)
        self.redraw()

    def select_in_list(self, root: Optional[int]):
        if root is None:
            return
        i = self.cell_combo.findData(int(root))
        self.cell_combo.blockSignals(True)
        if i >= 0:
            self.cell_combo.setCurrentIndex(i)
        else:
            r = self.film.info.get(int(root))
            if r is not None:
                self.cell_combo.insertItem(0, track_entry(r), int(root)); self.cell_combo.setCurrentIndex(0)
        self.cell_combo.blockSignals(False)

    def follow(self, root: int):
        """Follow this track: the Cell view on it, from just before its landing."""
        self.film.set_root(int(root))
        self.select_in_list(root)
        if self.view != "Cell":
            self.view = "Cell"
            self.view_combo.blockSignals(True); self.view_combo.setCurrentText("Cell"); self.view_combo.blockSignals(False)
        self.reset_slider()

    def _on_cell_combo(self, i):
        root = self.cell_combo.itemData(i)
        if root is not None:
            self.film.set_root(int(root))
            self.reset_slider(start_hour=None if self.view == "Cell" else self.film.hour(self.frames()[self.step]) if self.frames() else None)

    def _on_view(self, text):
        hour = self.film.hour(self.frames()[self.step]) if self.frames() else None
        self.view = text
        self.reset_slider(start_hour=hour if text == "Field" else None)

    def _on_slider(self, v):
        self.step = int(v)
        self.redraw()

    def _on_play(self, on):
        self.play_button.setText("Pause" if on else "Play")
        self.timer.start() if on else self.timer.stop()

    def _step(self):
        n = len(self.frames())
        if n:
            self.slider.setValue((self.step + 1) % n)

    def _on_click(self, event):
        """A click on a cell of the picture: follow that cell."""
        if self.ax_img is None or event.inaxes is not self.ax_img or event.xdata is None:
            return
        frames = self.frames()
        if not frames:
            return
        tid = self.film.track_at(frames[self.step], float(event.xdata), float(event.ydata))
        if tid is None:
            self.hint.setText("No tracked cell there. Click a cell in the image to follow it.")
            return
        self.hint.setText(f"Following track {tid}.")
        if tid == self.film.root and self.view == "Cell":
            return                                        # the cell already followed: keep the place in its film
        self.follow(tid)

    # ---------------------------------------------------------------- drawing
    def redraw(self):
        frames = self.frames()
        if not frames:
            self.figure.clear(); self.canvas.draw_idle(); self.time_label.setText("")
            return
        f = frames[self.step]
        self.ax_img, _ = (self.film.draw_cell if self.view == "Cell" else self.film.draw_field)(self.figure, f)
        self.time_label.setText(f"frame {f}   {self.film.hour(f):.2f} h")
        self.canvas.draw_idle()

    # ---------------------------------------------------------------- export
    def save_film(self):
        name = f"seeding_track_{self.film.root}.gif" if self.view == "Cell" else "seeding_field.gif"
        path, _ = QFileDialog.getSaveFileName(self, "Save Film", name, "GIF (*.gif)")
        if not path:
            return
        n = len(self.film.frames_of(self.view))
        progress = QProgressDialog("Drawing the film...", "Cancel", 0, n, self)
        progress.setWindowModality(Qt.WindowModal)

        def step(k, total):
            progress.setValue(k)
            return not progress.wasCanceled()

        try:
            out = self.film.save_gif(path, self.view, progress=step)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not save the film: {e}")
            return
        finally:
            progress.close()
        if out:
            self.hint.setText(f"Film saved: {path}")

    def save_image(self):
        stem = f"seeding_track_{self.film.root}" if self.view == "Cell" else "seeding_field"
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", f"{stem}.png", "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            self.hint.setText(f"Image saved: {path}")

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)
