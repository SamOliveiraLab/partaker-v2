"""The Microcolonies window, laid out like the lineage and Seeding windows: the microcolony list on top, the film on the
left with its merging history drawing itself on the right, Play and a slider under it, Save Film, Save Image and Close.
Click a microcolony in the image to follow it. Drawing is analysis.biofilm_stages.microcolony_film."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QProgressDialog, QPushButton, QSlider,
                               QVBoxLayout)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from nd2_analyzer.analysis.biofilm_stages.microcolony_film import FIGURE_IN, MicrocolonyFilm

HINT = "Click a microcolony in the image to follow it."


class MicrocolonyDialog(QDialog):
    MS_PER_FRAME = 350

    def __init__(self, film: MicrocolonyFilm, sections: list, root: Optional[int] = None, parent=None):
        super().__init__(parent)
        self.film = film; self.step = 0; self.ax_img = None
        self.setWindowTitle("Microcolonies"); self.setAttribute(Qt.WA_DeleteOnClose); self.setMinimumWidth(1000); self.setMinimumHeight(700)
        layout = QVBoxLayout(self); top = QHBoxLayout()
        top.addWidget(QLabel("Microcolony:")); self.combo = QComboBox(); self.combo.setMinimumWidth(340)
        for _title, rows in sections:
            for identity, line in rows:
                self.combo.addItem(line, int(identity))
        self.combo.currentIndexChanged.connect(self._on_combo); top.addWidget(self.combo); top.addSpacing(16)
        self.hint = QLabel(HINT); top.addWidget(self.hint); top.addStretch(); layout.addLayout(top)
        self.figure = Figure(figsize=FIGURE_IN); self.canvas = FigureCanvas(self.figure); self.canvas.mpl_connect("button_press_event", self._on_click)
        layout.addWidget(self.canvas, 1)
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
        start = root if root is not None else (film.root if film.root is not None else film.default_root())
        if start is not None:
            self.follow(int(start), say=False)

    # ---------------------------------------------------------------- state
    def follow(self, root: int, say: bool = True):
        """Follow this microcolony from the first frame of its film."""
        self.film.set_root(int(root))
        i = self.combo.findData(int(root)); self.combo.blockSignals(True)
        if i < 0:
            self.combo.insertItem(0, f"Microcolony {int(root)}", int(root)); i = 0
        self.combo.setCurrentIndex(i); self.combo.blockSignals(False)
        self.slider.blockSignals(True); self.slider.setMinimum(0); self.slider.setMaximum(max(len(self.film.frames) - 1, 0)); self.slider.setValue(0); self.slider.blockSignals(False)
        self.step = 0
        if say:
            self.hint.setText(f"Following microcolony {int(root)}.")
        self.redraw()

    def follow_at(self, x: float, y: float):
        """A point of the picture (its pixels): follow the microcolony there."""
        if not self.film.frames:
            return
        identity = self.film.identity_at(self.film.frames[self.step], x, y)
        if identity is None:
            self.hint.setText("No microcolony there. " + HINT); return
        if identity != self.film.root:
            self.follow(identity)

    def _on_combo(self, i):
        root = self.combo.itemData(i)
        if root is not None:
            self.follow(int(root))

    def _on_slider(self, v):
        self.step = int(v); self.redraw()

    def _on_play(self, on):
        self.play_button.setText("Pause" if on else "Play")
        self.timer.start() if on else self.timer.stop()

    def _tick(self):
        if self.film.frames:
            self.slider.setValue((self.step + 1) % len(self.film.frames))

    def _on_click(self, event):
        if self.ax_img is not None and event.inaxes is self.ax_img and event.xdata is not None:
            self.follow_at(float(event.xdata), float(event.ydata))

    def redraw(self):
        if not self.film.frames:
            self.figure.clear(); self.canvas.draw_idle(); self.time_label.setText(""); return
        t = self.film.frames[self.step]; self.ax_img, _ = self.film.draw(self.figure, t)
        self.time_label.setText(f"frame {t}   {self.film.tracks.hours[t]:g} h"); self.canvas.draw_idle()

    # ---------------------------------------------------------------- export
    def save_film(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Film", f"microcolony_{self.film.root}.gif", "GIF (*.gif)")
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
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", f"microcolony_{self.film.root}.png", "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); self.hint.setText(f"Image saved: {path}")

    def closeEvent(self, event):
        self.timer.stop(); super().closeEvent(event)
