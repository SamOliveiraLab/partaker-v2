# environment_view_tab.py
"""
Environment View tab — Approach 1 from the paper-1 deck.

Grid-based analysis of the chamber. Instead of tracking individual cells,
we overlay a regular grid over the image and aggregate bacterial state
per grid cell over time, producing density maps, state maps, and
(eventually) correlations against COMSOL hydrodynamic variables.

This widget lives *inside* the Tracking tab alongside the Cell View tab.
Both tabs share the same upstream tracking data; this tab simply
reshapes per-cell tracks into per-grid per-timepoint aggregates.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# --------------------------------------------------------------------------- #
# tracks_to_grid helper                                                        #
# --------------------------------------------------------------------------- #
def tracks_to_grid(
    lineage_tracks: Sequence[dict],
    image_shape: tuple[int, int],
    grid_shape: tuple[int, int],
) -> dict:
    """Aggregate per-cell tracks into per-grid per-timepoint state.

    Parameters
    ----------
    lineage_tracks
        The ``lineage_tracks`` list produced by ``run_tracker``. Each track
        is a dict with at least ``x``, ``y``, ``t`` iterables of equal
        length.
    image_shape
        ``(height, width)`` of the raw frames (pixels).
    grid_shape
        ``(n_rows, n_cols)`` of the overlay grid.

    Returns
    -------
    dict with keys:
        - ``density``  : ndarray ``(T, n_rows, n_cols)`` — cell count per grid
        - ``mean_speed`` : ndarray ``(T, n_rows, n_cols)`` — mean instantaneous
          speed of cells whose previous position is known (NaN where no cells)
        - ``t_values`` : ndarray ``(T,)`` — the sorted unique timepoints
        - ``grid_shape`` : ``(n_rows, n_cols)``
        - ``image_shape`` : ``(height, width)``
    """
    h, w = image_shape
    n_rows, n_cols = grid_shape

    # Collect unique timepoints so we know how many frames to build.
    t_set: set[int] = set()
    for track in lineage_tracks or []:
        ts = track.get("t")
        if ts is None:
            continue
        for t in ts:
            t_set.add(int(t))

    if not t_set:
        return {
            "density": np.zeros((0, n_rows, n_cols)),
            "mean_speed": np.zeros((0, n_rows, n_cols)),
            "t_values": np.array([], dtype=int),
            "grid_shape": grid_shape,
            "image_shape": image_shape,
        }

    # Remove focus loss frames
    try:
        from nd2_analyzer.data.appstate import ApplicationState
        _appstate = ApplicationState.get_instance()
        if _appstate and _appstate.experiment and _appstate.experiment.focus_loss_intervals:
            t_set = {t for t in t_set if not _appstate.experiment.is_focus_loss_frame(t)}
    except Exception:
        pass

    t_values = np.array(sorted(t_set), dtype=int)
    t_index = {int(tv): i for i, tv in enumerate(t_values)}
    T = len(t_values)

    density = np.zeros((T, n_rows, n_cols), dtype=np.int32)
    speed_sum = np.zeros((T, n_rows, n_cols), dtype=np.float64)
    speed_n = np.zeros((T, n_rows, n_cols), dtype=np.int32)

    # Pixel→grid index mapping. Clamp defensively so off-image coords
    # (tracker drift, subpixel artifacts) still land in a valid bucket.
    def _bucket(x: float, y: float) -> tuple[int, int]:
        col = int(np.clip(x / max(w, 1) * n_cols, 0, n_cols - 1))
        row = int(np.clip(y / max(h, 1) * n_rows, 0, n_rows - 1))
        return row, col

    for track in lineage_tracks or []:
        xs = track.get("x", [])
        ys = track.get("y", [])
        ts = track.get("t")
        if ts is None:
            ts = list(range(len(xs)))

        if not (len(xs) == len(ys) == len(ts)):
            # Skip malformed tracks rather than crash the whole map.
            continue

        prev_x = prev_y = prev_t = None
        for x, y, t in zip(xs, ys, ts):
            ti = t_index.get(int(t))
            if ti is None:
                continue

            r, c = _bucket(float(x), float(y))
            density[ti, r, c] += 1

            if prev_x is not None and prev_t is not None and int(t) - int(prev_t) > 0:
                dt = int(t) - int(prev_t)
                speed = float(np.hypot(x - prev_x, y - prev_y)) / dt
                speed_sum[ti, r, c] += speed
                speed_n[ti, r, c] += 1

            prev_x, prev_y, prev_t = x, y, t

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_speed = np.where(speed_n > 0, speed_sum / speed_n, np.nan)

    return {
        "density": density,
        "mean_speed": mean_speed,
        "t_values": t_values,
        "grid_shape": grid_shape,
        "image_shape": image_shape,
    }


# --------------------------------------------------------------------------- #
# EnvironmentViewTab widget                                                    #
# --------------------------------------------------------------------------- #
class EnvironmentViewTab(QWidget):
    """Grid-based environment-view analysis panel.

    Hosted inside the Tracking tab's QTabWidget. Reads tracks from a
    per-position cache (passed in via ``set_tracks``) and renders grid
    aggregates on demand.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._lineage_tracks: list[dict] | None = None
        self._image_shape: tuple[int, int] | None = None
        self._last_aggregates: dict | None = None

        self._init_ui()

    # ------------------------------------------------------------------ #
    # UI                                                                 #
    # ------------------------------------------------------------------ #
    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Grid-configuration controls --------------------------------- #
        grid_box = QGroupBox("Grid overlay")
        grid_row = QHBoxLayout(grid_box)
        grid_row.addWidget(QLabel("Rows:"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 200)
        self.rows_spin.setValue(10)
        grid_row.addWidget(self.rows_spin)

        grid_row.addSpacing(12)
        grid_row.addWidget(QLabel("Cols:"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 200)
        self.cols_spin.setValue(10)
        grid_row.addWidget(self.cols_spin)

        grid_row.addSpacing(12)
        grid_row.addWidget(QLabel("Time:"))
        self.time_combo = QComboBox()
        self.time_combo.setMinimumContentsLength(8)
        grid_row.addWidget(self.time_combo)

        grid_row.addStretch()
        layout.addWidget(grid_box)

        # Action buttons --------------------------------------------- #
        btn_row = QHBoxLayout()
        self.density_button = QPushButton("Density Map")
        self.density_button.clicked.connect(self._show_density)
        btn_row.addWidget(self.density_button)

        self.speed_button = QPushButton("Mean Speed Map")
        self.speed_button.clicked.connect(self._show_speed)
        btn_row.addWidget(self.speed_button)

        self.comsol_button = QPushButton("COMSOL Variables (coming soon)")
        self.comsol_button.setEnabled(False)
        btn_row.addWidget(self.comsol_button)

        self.corr_button = QPushButton("Neighbour Correlations (coming soon)")
        self.corr_button.setEnabled(False)
        btn_row.addWidget(self.corr_button)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Plot area --------------------------------------------------- #
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Start disabled until tracks arrive.
        self._set_buttons_enabled(False)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def set_tracks(
        self,
        lineage_tracks: list[dict] | None,
        image_shape: tuple[int, int] | None,
    ) -> None:
        """Feed fresh tracking data into the tab.

        Called by the tracking widget when the user switches position or
        re-runs tracking. ``image_shape`` is ``(height, width)`` of the
        raw frames so the grid overlay matches pixel coordinates.
        """
        self._lineage_tracks = lineage_tracks
        self._image_shape = image_shape
        self._last_aggregates = None

        has_tracks = bool(lineage_tracks) and image_shape is not None
        self._set_buttons_enabled(has_tracks)

        self.time_combo.clear()
        if has_tracks:
            # Populate time selector from whatever timepoints appear in tracks.
            t_set: set[int] = set()
            for tr in lineage_tracks:
                for t in tr.get("t", []):
                    t_set.add(int(t))
            for t in sorted(t_set):
                self.time_combo.addItem(str(t), t)

        self.figure.clear()
        self.canvas.draw()

    # ------------------------------------------------------------------ #
    # Handlers                                                           #
    # ------------------------------------------------------------------ #
    def _set_buttons_enabled(self, enabled: bool) -> None:
        self.density_button.setEnabled(enabled)
        self.speed_button.setEnabled(enabled)
        # COMSOL / correlations stay disabled regardless — not wired yet.

    def _aggregate(self) -> dict | None:
        if not self._lineage_tracks or self._image_shape is None:
            QMessageBox.warning(
                self, "No data", "Run tracking first, then open Environment View."
            )
            return None
        grid_shape = (self.rows_spin.value(), self.cols_spin.value())
        aggregates = tracks_to_grid(
            self._lineage_tracks, self._image_shape, grid_shape
        )
        self._last_aggregates = aggregates
        return aggregates

    def _current_t_index(self, t_values: np.ndarray) -> int:
        """Return the index into t_values for the currently-selected time.

        Falls back to the first timepoint if the combo is empty.
        """
        if self.time_combo.count() == 0 or len(t_values) == 0:
            return 0
        selected = self.time_combo.currentData()
        if selected is None:
            return 0
        # Find matching entry; default to 0 if not found.
        matches = np.where(t_values == int(selected))[0]
        return int(matches[0]) if matches.size else 0

    def _show_density(self) -> None:
        agg = self._aggregate()
        if agg is None or agg["density"].shape[0] == 0:
            return
        ti = self._current_t_index(agg["t_values"])
        frame = agg["density"][ti]

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        im = ax.imshow(frame, cmap="viridis", origin="upper", aspect="auto")
        ax.set_title(f"Cell density per grid — t = {int(agg['t_values'][ti])}")
        ax.set_xlabel("Grid column")
        ax.set_ylabel("Grid row")
        self.figure.colorbar(im, ax=ax, label="cells")
        self.figure.tight_layout()
        self.canvas.draw()

    def _show_speed(self) -> None:
        agg = self._aggregate()
        if agg is None or agg["mean_speed"].shape[0] == 0:
            return
        ti = self._current_t_index(agg["t_values"])
        frame = agg["mean_speed"][ti]

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        im = ax.imshow(frame, cmap="magma", origin="upper", aspect="auto")
        ax.set_title(
            f"Mean instantaneous speed per grid — t = {int(agg['t_values'][ti])}"
        )
        ax.set_xlabel("Grid column")
        ax.set_ylabel("Grid row")
        self.figure.colorbar(im, ax=ax, label="px / frame")
        self.figure.tight_layout()
        self.canvas.draw()
