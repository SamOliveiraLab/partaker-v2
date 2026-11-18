"""The Maturation window, laid out like the lineage, Seeding and Microcolonies windows: the position list on top, the film
of the whole frame on the left with the number of cells and the matrix stain drawing themselves on the right, Play and a
slider under it, Save Film, Save Image and Close. Drawing is analysis.biofilm_stages.growth_film."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QProgressDialog, QPushButton, QSlider,
                               QVBoxLayout)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from nd2_analyzer.analysis.biofilm_stages.growth_film import FIGURE_IN, GrowthFilm

HINT = "Grey: phase contrast. Orange outline: microcolonies. Green: matrix stain."


class GrowthDialog(QDialog):
    MS_PER_FRAME = 350

    def __init__(self, film: GrowthFilm, positions: list, position: Optional[int] = None, parent=None):
        super().__init__(parent)
        self.film = film; self.step = 0
        self.setWindowTitle("Maturation"); self.setAttribute(Qt.WA_DeleteOnClose); self.setMinimumWidth(1000); self.setMinimumHeight(700)
        layout = QVBoxLayout(self); top = QHBoxLayout()
        top.addWidget(QLabel("Position:")); self.combo = QComboBox(); self.combo.setMinimumWidth(340)
        for p, line in positions:
            self.combo.addItem(line, int(p))
        top.addWidget(self.combo); top.addSpacing(16); self.hint = QLabel(HINT); top.addWidget(self.hint); top.addStretch(); layout.addLayout(top)
        self.figure = Figure(figsize=FIGURE_IN); self.canvas = FigureCanvas(self.figure); layout.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        self.play_button = QPushButton("Play"); self.play_button.setCheckable(True); self.play_button.toggled.connect(self._on_play); row.addWidget(self.play_button)
        self.slider = QSlider(Qt.Horizontal); self.slider.valueChanged.connect(self._on_slider); row.addWidget(self.slider, 1)
        self.time_label = QLabel(""); self.time_label.setMinimumWidth(170); row.addWidget(self.time_label); layout.addLayout(row)
        buttons = QHBoxLayout()
        self.save_film_button = QPushButton("Save Film"); self.save_film_button.clicked.connect(self.save_film); buttons.addWidget(self.save_film_button)
        self.save_button = QPushButton("Save Image"); self.save_button.clicked.connect(self.save_image); buttons.addWidget(self.save_button)
        buttons.addStretch(); self.close_button = QPushButton("Close"); self.close_button.clicked.connect(self.close); buttons.addWidget(self.close_button)
        layout.addLayout(buttons)
        self.timer = QTimer(self); self.timer.setInterval(self.MS_PER_FRAME); self.timer.timeout.connect(self._tick)
        i = self.combo.findData(int(position if position is not None else film.position)); self.combo.setCurrentIndex(max(i, 0))
        self.combo.currentIndexChanged.connect(self._on_combo); self.show_position(self.combo.currentData(), hour=None)

    def show_position(self, position: int, hour: Optional[float]):
        """The film of this position, at the same hour as before when there was one."""
        self.film.set_position(int(position)); n = len(self.film.frames)
        self.step = self.film.frames.index(self.film.frame_at(hour)) if hour is not None and n else 0
        self.slider.blockSignals(True); self.slider.setMinimum(0); self.slider.setMaximum(max(n - 1, 0)); self.slider.setValue(self.step); self.slider.blockSignals(False)
        self.redraw()

    def _on_combo(self, i):
        hour = self.film.hour(self.film.frames[self.step]) if self.film.frames else None
        self.show_position(self.combo.itemData(i), hour)

    def _on_slider(self, v):
        self.step = int(v); self.redraw()

    def _on_play(self, on):
        self.play_button.setText("Pause" if on else "Play")
        self.timer.start() if on else self.timer.stop()

    def _tick(self):
        if self.film.frames:
            self.slider.setValue((self.step + 1) % len(self.film.frames))

    def redraw(self):
        if not self.film.frames:
            self.figure.clear(); self.canvas.draw_idle(); self.time_label.setText(""); return
        t = self.film.frames[self.step]; self.film.draw(self.figure, t)
        self.time_label.setText(f"frame {t}   {self.film.hour(t):g} h"); self.canvas.draw_idle()

    def save_film(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Film", f"maturation_position_{self.film.position}.gif", "GIF (*.gif)")
        if not path:
            return
        progress = QProgressDialog("Drawing the film...", "Cancel", 0, len(self.film.frames), self); progress.setWindowModality(Qt.WindowModal)

        def step(k, _n):
            progress.setValue(k); return not progress.wasCanceled()

        try:
            out = self.film.save_gif(path, progress=step)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not save the film: {e}"); return
        finally:
            progress.close()
        if out:
            self.hint.setText(f"Film saved: {path}")

    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", f"maturation_position_{self.film.position}.png", "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); self.hint.setText(f"Image saved: {path}")

    def closeEvent(self, event):
        self.timer.stop(); super().closeEvent(event)
