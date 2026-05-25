# track_validation_tab.py
"""
Track Validation tab -- visual validation of cell tracking results.

Shows a 4-panel figure (raw GIF frames, tracked overlay, lineage tree,
drift plot) for a user-selected subset of cells.  Before any analysis
is run the tab renders a schematic mockup of the expected layout so the
user knows what to expect.

Lives inside the Tracking page alongside Cell View, Environment View,
and Digital Twin.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.gridspec import GridSpec
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _get_family(tracks: list[dict], root_id: int) -> list[dict]:
    """Return a root track and all its descendants (children, grandchildren ...)."""
    by_id = {t["ID"]: t for t in tracks}
    family_ids: list[int] = []
    queue = [root_id]
    while queue:
        cid = queue.pop(0)
        if cid in by_id:
            family_ids.append(cid)
            queue.extend(by_id[cid].get("children", []))
    return [by_id[i] for i in family_ids if i in by_id]


def _auto_select_roots(tracks: list[dict], strategy: str = "most_divisions",
                       count: int = 3) -> list[int]:
    """Pick *count* root cell IDs using the chosen heuristic."""
    by_id = {t["ID"]: t for t in tracks}

    def _descendants(tid):
        n = 0
        q = list(by_id.get(tid, {}).get("children", []))
        while q:
            c = q.pop(0)
            n += 1
            q.extend(by_id.get(c, {}).get("children", []))
        return n

    if strategy == "most_divisions":
        scored = [(t["ID"], _descendants(t["ID"])) for t in tracks if t.get("parent") is None]
        scored.sort(key=lambda x: x[1], reverse=True)
    elif strategy == "longest_lived":
        scored = [(t["ID"], len(t.get("t", []))) for t in tracks if t.get("parent") is None]
        scored.sort(key=lambda x: x[1], reverse=True)
    else:
        scored = [(t["ID"], 0) for t in tracks if t.get("parent") is None]

    return [s[0] for s in scored[:count]]


# ------------------------------------------------------------------ #
# Persistent color map for cell IDs                                   #
# ------------------------------------------------------------------ #

_CELL_CMAP = matplotlib.colormaps["tab10"]


def _color_for(cell_id: int):
    return _CELL_CMAP(cell_id % 10)


# ------------------------------------------------------------------ #
# The Tab                                                             #
# ------------------------------------------------------------------ #

class TrackValidationTab(QWidget):
    """Four-panel tracking validation: raw frames, tracked overlay,
    lineage tree, and drift plot."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tracks: list[dict] = []
        self._raw_images: Optional[np.ndarray] = None
        self._mask_images: Optional[np.ndarray] = None
        self._selected_families: list[dict] = []
        self._is_mockup = True

        self._init_ui()

    # ---------------------------------------------------------- #
    # UI setup                                                    #
    # ---------------------------------------------------------- #

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # -- Controls row ---------------------------------------- #
        ctrl = QHBoxLayout()

        ctrl.addWidget(QLabel("Select cells:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Most divisions", "Longest lived"])
        ctrl.addWidget(self.strategy_combo)

        ctrl.addWidget(QLabel("Count:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 10)
        self.count_spin.setValue(3)
        ctrl.addWidget(self.count_spin)

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self._on_generate)
        self.generate_btn.setEnabled(False)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #D85A30;
                color: white; border: none;
                padding: 8px 16px; border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover { background-color: #C04828; }
            QPushButton:disabled { background-color: #CCCCCC; color: #666666; }
        """)
        ctrl.addWidget(self.generate_btn)

        ctrl.addStretch()

        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self._on_export)
        self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #EF9F27;
                color: white; border: none;
                padding: 8px 16px; border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover { background-color: #BA7517; }
            QPushButton:disabled { background-color: #CCCCCC; color: #666666; }
        """)
        ctrl.addWidget(self.export_btn)

        layout.addLayout(ctrl)

        # -- Matplotlib figure ----------------------------------- #
        self.fig = plt.figure(figsize=(10, 7), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Draw the mockup on first show
        self._draw_mockup()

    # ---------------------------------------------------------- #
    # Public API (called by TrackingWidget)                       #
    # ---------------------------------------------------------- #

    def set_data(self, tracks: list[dict] | None,
                 raw_images: np.ndarray | None = None,
                 mask_images: np.ndarray | None = None):
        """Inject tracking results and optional image stacks."""
        self._tracks = tracks or []
        self._raw_images = raw_images
        self._mask_images = mask_images
        has_data = len(self._tracks) > 0
        self.generate_btn.setEnabled(has_data)
        if not has_data:
            self._is_mockup = True
            self._draw_mockup()

    # ---------------------------------------------------------- #
    # Mockup rendering                                            #
    # ---------------------------------------------------------- #

    def _draw_mockup(self):
        """Render a placeholder schematic that previews the 4-panel layout."""
        self.fig.clear()
        self._is_mockup = True

        gs = GridSpec(2, 2, figure=self.fig, hspace=0.35, wspace=0.25)
        mock_color = "#E8E6DF"
        accent_red = "#D85A30"
        accent_amber = "#EF9F27"
        text_kw = dict(ha="center", va="center", fontsize=11,
                       color="#888780", style="italic")

        # Panel B: Raw frames
        ax_raw = self.fig.add_subplot(gs[0, 0])
        ax_raw.set_facecolor(mock_color)
        ax_raw.set_title("B   Raw microscopy (GIF)", loc="left",
                         fontsize=10, fontweight="medium")
        # Placeholder cell ellipses
        for cx, cy, angle in [(0.3, 0.5, 20), (0.55, 0.6, -30),
                               (0.7, 0.35, 50), (0.4, 0.75, 0)]:
            e = mpatches.Ellipse((cx, cy), 0.12, 0.05, angle=angle,
                                  fc="#B4B2A9", ec="#888780", lw=0.8)
            ax_raw.add_patch(e)
        ax_raw.text(0.5, 0.1, "phase contrast frames", **text_kw)
        ax_raw.set_xlim(0, 1)
        ax_raw.set_ylim(0, 1)
        ax_raw.set_xticks([])
        ax_raw.set_yticks([])

        # Panel C: Tracked overlay
        ax_trk = self.fig.add_subplot(gs[0, 1])
        ax_trk.set_facecolor(mock_color)
        ax_trk.set_title("C   Tracked overlay (GIF)", loc="left",
                         fontsize=10, fontweight="medium")
        for i, (cx, cy, angle) in enumerate(
                [(0.3, 0.5, 20), (0.55, 0.6, -30),
                 (0.7, 0.35, 50), (0.4, 0.75, 0)]):
            e = mpatches.Ellipse((cx, cy), 0.12, 0.05, angle=angle,
                                  fc="none", ec=accent_red, lw=1.5)
            ax_trk.add_patch(e)
            # Trajectory tail
            dx, dy = 0.08 * np.cos(np.radians(angle)), 0.08 * np.sin(np.radians(angle))
            ax_trk.plot([cx - dx, cx], [cy - dy, cy],
                        color=accent_amber, lw=1.5, alpha=0.8)
            ax_trk.text(cx + 0.06, cy + 0.06, str(i + 1),
                        fontsize=8, color=accent_red, fontweight="bold")
        ax_trk.text(0.5, 0.1, "IDs and tracks overlaid", **text_kw)
        ax_trk.set_xlim(0, 1)
        ax_trk.set_ylim(0, 1)
        ax_trk.set_xticks([])
        ax_trk.set_yticks([])

        # Panel D: Lineage tree
        ax_lin = self.fig.add_subplot(gs[1, 0])
        ax_lin.set_facecolor(mock_color)
        ax_lin.set_title("D   Lineage tree", loc="left",
                         fontsize=10, fontweight="medium")
        # Simple branching tree
        tree_lines = [
            ([0.05, 0.2], [0.5, 0.5]),
            ([0.2, 0.2], [0.3, 0.7]),
            ([0.2, 0.45], [0.7, 0.7]),
            ([0.2, 0.45], [0.3, 0.3]),
            ([0.45, 0.45], [0.2, 0.4]),
            ([0.45, 0.7], [0.4, 0.4]),
            ([0.45, 0.7], [0.2, 0.2]),
            ([0.45, 0.45], [0.6, 0.8]),
            ([0.45, 0.7], [0.8, 0.8]),
            ([0.45, 0.7], [0.6, 0.6]),
        ]
        for xs, ys in tree_lines:
            ax_lin.plot(xs, ys, color=accent_amber, lw=2)
        ax_lin.set_xlabel("Time", fontsize=9, color="#888780")
        ax_lin.set_xlim(0, 1)
        ax_lin.set_ylim(0, 1)
        ax_lin.set_xticks([])
        ax_lin.set_yticks([])

        # Panel E: Drift plot
        ax_dft = self.fig.add_subplot(gs[1, 1])
        ax_dft.set_facecolor(mock_color)
        ax_dft.set_title("E   Drift / accuracy", loc="left",
                         fontsize=10, fontweight="medium")
        x = np.linspace(0, 1, 30)
        for _ in range(5):
            y = 0.15 + 0.05 * np.random.randn(30).cumsum() * 0.02
            y = np.clip(y, 0.05, 0.9)
            ax_dft.plot(x, y, color=accent_red, lw=1, alpha=0.6)
        ax_dft.axhline(0.6, ls="--", color="#888780", lw=0.8)
        ax_dft.text(0.92, 0.63, "threshold", fontsize=8, color="#888780")
        ax_dft.set_xlabel("Frame index", fontsize=9, color="#888780")
        ax_dft.set_ylabel("Displacement", fontsize=9, color="#888780")
        ax_dft.set_xlim(0, 1)
        ax_dft.set_ylim(0, 1)
        ax_dft.set_xticks([])
        ax_dft.set_yticks([])

        self.canvas.draw()

    # ---------------------------------------------------------- #
    # Real rendering                                              #
    # ---------------------------------------------------------- #

    def _on_generate(self):
        if not self._tracks:
            return

        strategy_map = {
            "Most divisions": "most_divisions",
            "Longest lived": "longest_lived",
        }
        strategy = strategy_map.get(self.strategy_combo.currentText(), "most_divisions")
        count = self.count_spin.value()

        root_ids = _auto_select_roots(self._tracks, strategy, count)
        if not root_ids:
            QMessageBox.warning(self, "No cells",
                                "Could not find root cells with the selected strategy.")
            return

        # Gather families
        self._selected_families = []
        for rid in root_ids:
            self._selected_families.extend(_get_family(self._tracks, rid))

        # Deduplicate
        seen = set()
        unique = []
        for t in self._selected_families:
            if t["ID"] not in seen:
                seen.add(t["ID"])
                unique.append(t)
        self._selected_families = unique

        self._draw_real()
        self.export_btn.setEnabled(True)

    def _draw_real(self):
        """Render the 4-panel figure with actual tracking data."""
        self.fig.clear()
        self._is_mockup = False
        tracks = self._selected_families
        if not tracks:
            self._draw_mockup()
            return

        gs = GridSpec(2, 2, figure=self.fig, hspace=0.35, wspace=0.3)

        # ---- Panel B: Raw frame (middle timepoint) ------------ #
        ax_raw = self.fig.add_subplot(gs[0, 0])
        mid_frame_idx = self._get_mid_frame(tracks)
        if self._raw_images is not None and mid_frame_idx < len(self._raw_images):
            ax_raw.imshow(self._raw_images[mid_frame_idx], cmap="gray")
        else:
            ax_raw.text(0.5, 0.5, "Raw images not available",
                        ha="center", va="center", color="#888780")
            ax_raw.set_xlim(0, 1)
            ax_raw.set_ylim(0, 1)
        ax_raw.set_title(f"B   Raw frame (t={mid_frame_idx})", loc="left",
                         fontsize=10, fontweight="medium")
        ax_raw.set_xticks([])
        ax_raw.set_yticks([])

        # ---- Panel C: Tracked overlay ------------------------- #
        ax_trk = self.fig.add_subplot(gs[0, 1])
        if self._raw_images is not None and mid_frame_idx < len(self._raw_images):
            ax_trk.imshow(self._raw_images[mid_frame_idx], cmap="gray", alpha=0.6)

        selected_ids = {t["ID"] for t in tracks}
        for track in tracks:
            ts = track.get("t", [])
            xs = track.get("x", [])
            ys = track.get("y", [])
            color = _color_for(track["ID"])

            # Find position at mid_frame_idx
            if mid_frame_idx in ts:
                idx = ts.index(mid_frame_idx)
                cx, cy = xs[idx], ys[idx]
                ax_trk.plot(cx, cy, "o", color=color, ms=6, mew=1.5, mfc="none")
                ax_trk.annotate(str(track["ID"]), (cx, cy),
                                textcoords="offset points", xytext=(5, 5),
                                fontsize=7, color=color, fontweight="bold")
                # Trajectory tail (last N frames before mid)
                tail_len = 8
                start = max(0, idx - tail_len)
                ax_trk.plot(xs[start:idx + 1], ys[start:idx + 1],
                            "-", color=color, lw=1.2, alpha=0.7)

        ax_trk.set_title(f"C   Tracked overlay (t={mid_frame_idx})", loc="left",
                         fontsize=10, fontweight="medium")
        ax_trk.set_xticks([])
        ax_trk.set_yticks([])
        if self._raw_images is not None and mid_frame_idx < len(self._raw_images):
            h, w = self._raw_images[mid_frame_idx].shape[:2]
            ax_trk.set_xlim(0, w)
            ax_trk.set_ylim(h, 0)

        # ---- Panel D: Lineage tree ---------------------------- #
        ax_lin = self.fig.add_subplot(gs[1, 0])
        self._draw_lineage(ax_lin, tracks)
        ax_lin.set_title("D   Lineage tree", loc="left",
                         fontsize=10, fontweight="medium")

        # ---- Panel E: Drift plot ------------------------------ #
        ax_dft = self.fig.add_subplot(gs[1, 1])
        self._draw_drift(ax_dft, tracks)
        ax_dft.set_title("E   Drift plot", loc="left",
                         fontsize=10, fontweight="medium")

        self.canvas.draw()

    # ---------------------------------------------------------- #
    # Lineage tree renderer                                       #
    # ---------------------------------------------------------- #

    def _draw_lineage(self, ax, tracks: list[dict]):
        """Horizontal lineage tree: time on x, cells stacked on y."""
        by_id = {t["ID"]: t for t in tracks}
        roots = [t for t in tracks if t.get("parent") is None
                 or t["parent"] not in by_id]

        y_counter = [0]
        y_positions: dict[int, float] = {}

        def _layout(tid: int) -> float:
            track = by_id.get(tid)
            if track is None:
                return y_counter[0]
            children = [c for c in track.get("children", []) if c in by_id]
            if not children:
                y = y_counter[0]
                y_counter[0] += 1
                y_positions[tid] = y
                return y
            child_ys = [_layout(c) for c in children]
            y = sum(child_ys) / len(child_ys)
            y_positions[tid] = y
            return y

        for root in roots:
            _layout(root["ID"])

        # Draw branches
        for track in tracks:
            tid = track["ID"]
            if tid not in y_positions:
                continue
            y = y_positions[tid]
            t_vals = track.get("t", [])
            if not t_vals:
                continue
            t_start, t_end = min(t_vals), max(t_vals)
            color = _color_for(tid)

            # Horizontal bar for this cell's lifespan
            ax.plot([t_start, t_end], [y, y], color=color, lw=2.5, solid_capstyle="round")

            # Vertical connector to children
            children = [c for c in track.get("children", []) if c in y_positions]
            if children:
                child_ys = [y_positions[c] for c in children]
                y_min, y_max = min(child_ys), max(child_ys)
                ax.plot([t_end, t_end], [y_min, y_max],
                        color=color, lw=1.5)

        ax.set_xlabel("Frame", fontsize=9)
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    # ---------------------------------------------------------- #
    # Drift plot renderer                                         #
    # ---------------------------------------------------------- #

    def _draw_drift(self, ax, tracks: list[dict]):
        """Frame-to-frame displacement for each selected cell."""
        all_displacements: list[float] = []

        for track in tracks:
            ts = track.get("t", [])
            xs = track.get("x", [])
            ys = track.get("y", [])
            if len(ts) < 2:
                continue

            frames = []
            disps = []
            for i in range(1, len(ts)):
                dt = ts[i] - ts[i - 1]
                if dt == 0:
                    continue
                dx = xs[i] - xs[i - 1]
                dy = ys[i] - ys[i - 1]
                d = np.sqrt(dx ** 2 + dy ** 2) / dt
                frames.append(ts[i])
                disps.append(d)
                all_displacements.append(d)

            color = _color_for(track["ID"])
            ax.plot(frames, disps, "-", color=color, lw=1, alpha=0.7,
                    label=f"Cell {track['ID']}")

        # Threshold line at 95th percentile
        if all_displacements:
            thresh = np.percentile(all_displacements, 95)
            ax.axhline(thresh, ls="--", color="#888780", lw=0.8)
            ax.text(ax.get_xlim()[1] * 0.75, thresh * 1.05,
                    f"p95 = {thresh:.1f}", fontsize=8, color="#888780")

        ax.set_xlabel("Frame", fontsize=9)
        ax.set_ylabel("Displacement (px/frame)", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ---------------------------------------------------------- #
    # Utilities                                                   #
    # ---------------------------------------------------------- #

    def _get_mid_frame(self, tracks: list[dict]) -> int:
        all_t = []
        for t in tracks:
            all_t.extend(t.get("t", []))
        if not all_t:
            return 0
        return int(np.median(all_t))

    # ---------------------------------------------------------- #
    # Export                                                       #
    # ---------------------------------------------------------- #

    def _on_export(self):
        if not self._selected_families:
            return

        folder = QFileDialog.getExistingDirectory(self, "Export folder")
        if not folder:
            return

        out = Path(folder)

        # 1. CSV
        csv_path = out / "tracking_validation.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cell_id", "frame", "x", "y", "parent_id",
                             "children"])
            for track in self._selected_families:
                parent = track.get("parent")
                children = ";".join(str(c) for c in track.get("children", []))
                for i, t in enumerate(track.get("t", [])):
                    writer.writerow([
                        track["ID"], t,
                        track["x"][i], track["y"][i],
                        parent if parent is not None else "",
                        children,
                    ])

        # 2. Composite figure
        fig_path = out / "tracking_validation.png"
        self.fig.savefig(fig_path, dpi=200, bbox_inches="tight")

        QMessageBox.information(
            self, "Export complete",
            f"Saved to {out}:\n  - tracking_validation.csv\n  - tracking_validation.png"
        )
