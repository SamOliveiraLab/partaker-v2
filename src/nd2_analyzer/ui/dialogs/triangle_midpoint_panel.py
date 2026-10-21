"""Manually selected, asynchronous preview for shared Triangle midpoint masks."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from nd2_analyzer.analysis.live_dead_triangle_midpoint import analyze_pair, analyze_cell_viability
from nd2_analyzer.analysis.live_dead_triangle_reports import diagnostic_figure


class TriangleMidpointPanel(QWidget):
    ready_changed = Signal(bool)

    def __init__(self, image_data, label_provider, live_channel, dead_channel, parent=None, analysis_mode="live_dead"):
        super().__init__(parent)
        self.image_data, self.label_provider = image_data, label_provider
        self.live_channel, self.dead_channel = live_channel, dead_channel
        self.analysis_mode = analysis_mode
        self.channel_indices = ({'cell_viability': live_channel} if analysis_mode == 'cell_viability'
                                else {'live': live_channel, 'dead': dead_channel})
        self.result = None
        self.images = self.labels = None
        self._generation = 0
        self._pending = []
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='triangle-preview')
        self.timer = QTimer(self); self.timer.setInterval(80); self.timer.timeout.connect(self.poll)
        layout = QVBoxLayout(self)
        description = QLabel('Segmented cells (blue) are automatically excluded from the nine background histograms. '
                             'Only finite cell-free pixels are sampled; no extra margin is applied. '
                             'Cells remain included in the cell-only reference and final measurements. '
                             + ('The channel with the lower mean midpoint supplies its grid to both channels. '
                                if analysis_mode == 'live_dead' else 'The cell viability channel supplies its own midpoint grid. ')
                             + 'No background subtraction, image smoothing, fluorescence morphology or Otsu is applied. '
                             'Change scope on the Fluorescence Threshold tab.')
        description.setWordWrap(True); layout.addWidget(description)
        controls = QHBoxLayout()
        self.position = QComboBox(); self.time = QComboBox()
        for title, combo in [('Position', self.position), ('Timepoint', self.time)]:
            controls.addWidget(QLabel(title)); controls.addWidget(combo)
            combo.currentIndexChanged.connect(self.invalidate)
        self.preview_button = QPushButton('Calculate Preview')
        self.preview_button.clicked.connect(self.request_preview); controls.addWidget(self.preview_button)
        layout.addLayout(controls)
        plots = QHBoxLayout()
        self.role = QComboBox()
        for role in self.channel_indices:
            self.role.addItem(role.replace('_', ' ').title(), role)
        self.kind = QComboBox()
        for label, value in [('Fluorescence / exclusion zones + nine histograms', 'sampling'),
                             ('Nine background histograms', 'blocks'), ('Cell-only histogram / left Triangle', 'cells'),
                             ('Before / selected fluorescence / masks / surface', 'preview'),
                             ('Cell-area intensity distributions', 'intensity_area')]:
            self.kind.addItem(label, value)
        self.role.currentIndexChanged.connect(self.render)
        self.kind.currentIndexChanged.connect(self.render)
        plots.addWidget(self.role); plots.addWidget(self.kind, 1); layout.addLayout(plots)
        self.status = QLabel('Open this tab and calculate a preview to use this method.')
        self.status.setWordWrap(True); layout.addWidget(self.status)
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(16, 11), constrained_layout=True))
        self.canvas.setMinimumSize(200, 200)
        layout.addWidget(self.canvas, 1)

    def sync_scope(self, positions, times):
        old = self.key()
        for combo, values in [(self.position, positions), (self.time, times)]:
            previous = combo.currentData()
            combo.blockSignals(True); combo.clear()
            for value in values:
                combo.addItem(str(value), int(value))
            index = combo.findData(previous)
            combo.setCurrentIndex(max(0, index)); combo.blockSignals(False)
        if self.key() != old:
            self.invalidate()

    def key(self):
        return self.time.currentData(), self.position.currentData()

    def invalidate(self, *_args):
        self._generation += 1
        for future, generation, key in self._pending:
            future.cancel()
        self.result = None
        self.images = self.labels = None
        self.ready_changed.emit(False)
        self.status.setText('Calculate a preview for the selected frame. Full analysis processes all selected positions/timepoints.')
        self.canvas.figure.clear(); self.canvas.draw_idle()

    def request_preview(self):
        self.invalidate()
        time, position = self.key()
        if self._closed or time is None or position is None:
            self.status.setText('Select a nonempty analysis scope first.'); return
        if self.label_provider is None:
            self.status.setText('Saved cell labels are unavailable. Segment cells first.'); return
        generation = self._generation
        def calculate():
            images = {role: np.array(self.image_data.get(time, position, channel), copy=True)
                      for role, channel in self.channel_indices.items()}
            labels = np.array(self.label_provider(time, position), copy=True)
            try:
                result = (analyze_cell_viability(images['cell_viability'], labels, self.live_channel)
                          if self.analysis_mode == 'cell_viability' else
                          analyze_pair(images['live'], images['dead'], labels, self.live_channel, self.dead_channel))
            except ValueError as error:
                raise ValueError(f'P={position}, T={time}: {error}') from error
            return result, images, labels
        future = self._executor.submit(calculate)
        self._pending.append((future, generation, (time, position)))
        self.timer.start(); self.status.setText('Calculating cell-excluded background histograms and the shared threshold surface…')

    def poll(self):
        pending = []
        for future, generation, key in self._pending:
            if not future.done():
                pending.append((future, generation, key)); continue
            if generation != self._generation or key != self.key():
                continue
            try:
                self.result, self.images, self.labels = future.result()
                channels = self.result['channels']
                means = '; '.join(f"{role.replace('_', ' ').title()} mean midpoint: {entry['mean_midpoint']:.3f}"
                                  for role, entry in channels.items())
                self.status.setText(means + f". Grid from C{self.result['selected_channel']}. "
                                    'Ready to confirm and run analysis.')
                self.render(); self.ready_changed.emit(True)
            except Exception as error:
                self.result = None; self.ready_changed.emit(False)
                self.status.setText(f'Preview unavailable: {error}')
        self._pending = pending
        if not pending:
            self.timer.stop()

    def showEvent(self, event):
        super().showEvent(event)
        self.render()

    def render(self, *_args):
        if self.result is None or not self.isVisible():
            return
        time, position = self.key()
        titles = {role: f"{role.replace('_', ' ').title()} C{channel} — P={position}, T={time}"
                  for role, channel in self.channel_indices.items()}
        fig = diagnostic_figure(self.result, self.images, self.labels,
                                self.kind.currentData(), self.role.currentData(), titles)
        # Attach to the existing canvas; release the previous figure.
        self.canvas.figure.clear()
        fig.set_size_inches(max(1, self.canvas.width())/fig.dpi,
                            max(1, self.canvas.height())/fig.dpi, forward=False)
        self.canvas.figure = fig; fig.set_canvas(self.canvas)
        self.canvas.draw_idle()

    def shutdown(self):
        self._closed = True; self._generation += 1; self.timer.stop()
        for future, generation, key in self._pending:
            future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
