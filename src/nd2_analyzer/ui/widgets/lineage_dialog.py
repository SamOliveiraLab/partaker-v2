"""The lineage window: one family followed in the image on the left, its tree drawing itself on the right, played frame by
frame. Pick a family from the list (largest first) or click a cell in the image; Album shows the stills at each division;
Save film writes the GIF. Drawing is analysis.tracking.film_render, shared with the export and the tests."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QProgressDialog, QPushButton,
                               QSlider, QVBoxLayout)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from nd2_analyzer.analysis.tracking import family as F
from nd2_analyzer.analysis.tracking.film_render import FamilyFilm


class LineageDialog(QDialog):
    MS_PER_FRAME = 350

    def __init__(self, lineage_tracks, parent=None):
        super().__init__(parent)
        self.lineage_tracks = list(lineage_tracks or [])
        self.by_id = F.index_tracks(self.lineage_tracks)
        self.families = F.families_by_size(self.by_id)
        self._masks = {}
        self.film = None
        self.t = 0
        self.ax_img = self.ax_tree = self.ax_inset = None
        self._reading_data()
        self.setWindowTitle("Cell Lineage Visualization")
        self.setMinimumWidth(1000)
        self.setMinimumHeight(700)
        self.init_ui()
        if self.families:
            self.set_family(self.families[0][0])

    # ---------------------------------------------------------------- data
    def _reading_data(self):
        """Where the masks, the frame count, the hours and the pixel size come from."""
        from nd2_analyzer.data.appstate import ApplicationState
        from nd2_analyzer.data.image_data import ImageData
        self.image_data = ImageData.get_instance()
        st = ApplicationState.get_instance()
        self.position = int(self.lineage_tracks[0].get("position", 0)) if self.lineage_tracks else 0
        if self.position == 0 and st is not None and getattr(st, "view_index", None):
            self.position = int(st.view_index[1])
        self.roi_mask = getattr(st, "roi_mask", None) if st is not None else None
        shape = self.image_data.data.shape if self.image_data is not None else (1, 1, 1, 1, 1)
        self.n_frames = int(shape[0])
        self.shape = (int(shape[-2]), int(shape[-1]))
        interval_h = None
        exp = getattr(st, "experiment", None) if st is not None else None
        if exp is not None:
            interval_h = getattr(exp, "time_interval_hours", None) or (float(getattr(exp, "phc_interval", 0) or 0) / 3600 or None)
        self.hours = np.arange(self.n_frames) * (interval_h or 5 / 60)
        self.pixel_um = self.image_data.pixel_size_um() if self.image_data is not None and hasattr(self.image_data, "pixel_size_um") else None

    def masks_at(self, t: int) -> np.ndarray:
        """The label frame of t (ROI applied), cached."""
        t = int(t)
        if t not in self._masks:
            if self.image_data is None:
                self._masks[t] = np.zeros(self.shape, np.int32)
            else:
                m = np.asarray(self.image_data.segmentation_cache[t, self.position, 0])
                if self.roi_mask is not None and self.roi_mask.shape == m.shape:
                    from nd2_analyzer.analysis.segmentation.segmentation_service import apply_roi_mask
                    m = apply_roi_mask(m, self.roi_mask)
                self._masks[t] = m
        return self._masks[t]

    # ---------------------------------------------------------------- ui
    def init_ui(self):
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Family:"))
        self.family_combo = QComboBox()
        for root, size in self.families:
            self.family_combo.addItem(f"Cell {root}  ({size} cells)", root)
        self.family_combo.currentIndexChanged.connect(self._on_family_combo)
        top.addWidget(self.family_combo)
        top.addSpacing(16)
        top.addWidget(QLabel("View:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Film", "Album"])
        self.mode_combo.currentIndexChanged.connect(lambda _i: self.redraw())
        top.addWidget(self.mode_combo)
        top.addSpacing(16)
        self.hint = QLabel("Click a cell in the image to follow its family.")
        top.addWidget(self.hint)
        top.addStretch()
        layout.addLayout(top)

        self.figure = Figure(figsize=(10, 4.6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        layout.addWidget(self.canvas, 1)

        time_row = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._on_play)
        time_row.addWidget(self.play_button)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0); self.slider.setMaximum(max(self.n_frames - 1, 0))
        self.slider.valueChanged.connect(self._on_slider)
        time_row.addWidget(self.slider, 1)
        self.time_label = QLabel("")
        self.time_label.setMinimumWidth(150)
        time_row.addWidget(self.time_label)
        layout.addLayout(time_row)

        buttons = QHBoxLayout()
        self.save_film_button = QPushButton("Save Film")
        self.save_film_button.clicked.connect(self.save_film)
        buttons.addWidget(self.save_film_button)
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        buttons.addWidget(self.save_button)
        buttons.addStretch()
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

        self.timer = QTimer(self)
        self.timer.setInterval(self.MS_PER_FRAME)
        self.timer.timeout.connect(self._step)
        if not self.families:
            self.hint.setText("No tracks to show.")

    # ---------------------------------------------------------------- state
    def set_family(self, root: int):
        root = int(root)
        self.film = FamilyFilm(self.lineage_tracks, root, self.masks_at, self.shape, self.hours, pixel_um=self.pixel_um)
        self.ax_img = self.ax_tree = self.ax_inset = None
        i = self.family_combo.findData(root)
        if i >= 0 and self.family_combo.currentIndex() != i:
            self.family_combo.blockSignals(True); self.family_combo.setCurrentIndex(i); self.family_combo.blockSignals(False)
        self.slider.blockSignals(True)
        self.slider.setMinimum(self.film.t_first); self.slider.setMaximum(self.film.t_last)
        self.t = int(np.clip(self.t, self.film.t_first, self.film.t_last))
        self.slider.setValue(self.t)
        self.slider.blockSignals(False)
        self.redraw()

    def _on_family_combo(self, i):
        root = self.family_combo.itemData(i)
        if root is not None:
            self.t = 0
            self.set_family(root)

    def _on_slider(self, v):
        self.t = int(v)
        self.redraw()

    def _on_play(self, on):
        self.play_button.setText("Pause" if on else "Play")
        if on:
            self.timer.start()
        else:
            self.timer.stop()

    def _step(self):
        if self.film is None:
            return
        nxt = self.t + 1 if self.t < self.film.t_last else self.film.t_first
        self.slider.setValue(nxt)

    def _on_click(self, event):
        """A click on a cell of the image: follow that cell's family."""
        if self.film is None or event.inaxes is None or event.inaxes is not self.ax_img or event.xdata is None:
            return
        sl = self.film.crop(self.t)
        x, y = float(event.xdata) + sl[1].start, float(event.ydata) + sl[0].start
        i = F.track_at_point(self.by_id, self.t, x, y, max_dist=15.0)
        if i is None:
            return
        root = F.root_of(self.by_id, i)
        if root != self.film.root:
            self.set_family(root)

    # ---------------------------------------------------------------- drawing
    def redraw(self):
        if self.film is None:
            return
        if self.mode_combo.currentText() == "Album":
            self.film.draw_album(self.figure)
            self.ax_img = self.ax_tree = self.ax_inset = None
            self.time_label.setText(f"{len(self.film.fam)} cells, {len(F.divisions(self.by_id, self.film.fam, self.film.kids))} divisions")
        else:
            self.ax_img, self.ax_tree = self.film.draw(self.figure, self.t, self.ax_img, self.ax_tree)
            self.time_label.setText(f"frame {self.t}   {self.hours[self.t]:.2f} h")
        self.canvas.draw_idle()

    # ---------------------------------------------------------------- export
    def save_film(self):
        if self.film is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Film", f"family_{self.film.root}.gif", "GIF (*.gif)")
        if not path:
            return
        n = self.film.t_last - self.film.t_first + 1
        progress = QProgressDialog("Drawing the film...", "Cancel", 0, n, self)
        progress.setWindowModality(Qt.WindowModal)

        def step(k, total):
            progress.setValue(k)
            return not progress.wasCanceled()

        try:
            self.film.save_gif(path, progress=step)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not save the film: {e}")
            return
        finally:
            progress.close()
        self.hint.setText(f"Film saved: {path}")

    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", f"family_{self.film.root if self.film else 0}.png", "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            self.hint.setText(f"Image saved: {path}")

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)
