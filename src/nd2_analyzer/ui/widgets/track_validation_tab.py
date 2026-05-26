# track_validation_tab.py
"""
Track Validation tab.

Six-panel composite for assessing single-cell tracking quality.

A  Pipeline status strip
B  Dense overlay (selected lineage families highlighted on full field)
C  Lineage tree (FAST-style, colored by root, black for unassigned)
D  Trackability gauge (FAST-style, bits per cell across frames)
E  Traccuracy metrics report
F  Drift plot per cell with p95 threshold
G  Per-cell triplet (prev frame, current + ID, mask overlay)
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.gridspec import GridSpec
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# ============================================================ #
# Color palette                                                 #
# ============================================================ #

ACCENT_RED = "#E24B4A"
ACCENT_AMBER = "#EF9F27"
ACCENT_DEEP = "#A82E18"
ACCENT_WARM = "#D85A30"
NEUTRAL = "#888780"
LIGHT_NEUTRAL = "#B4B2A9"
MOCK_BG = "#E8E6DF"

FAMILY_COLORS = [ACCENT_RED, ACCENT_AMBER, ACCENT_DEEP, ACCENT_WARM,
                 "#4FA8A8", "#A88BD8"]


# ============================================================ #
# Helpers                                                       #
# ============================================================ #

def _get_family(tracks, root_id):
    by_id = {t["ID"]: t for t in tracks}
    family_ids, queue = [], [root_id]
    while queue:
        cid = queue.pop(0)
        if cid in by_id:
            family_ids.append(cid)
            queue.extend(by_id[cid].get("children", []))
    return [by_id[i] for i in family_ids if i in by_id]


def _auto_select_roots(tracks, strategy="most_divisions", count=3):
    by_id = {t["ID"]: t for t in tracks}

    def _desc(tid):
        n = 0
        q = list(by_id.get(tid, {}).get("children", []))
        while q:
            c = q.pop(0)
            n += 1
            q.extend(by_id.get(c, {}).get("children", []))
        return n

    if strategy == "most_divisions":
        scored = [(t["ID"], _desc(t["ID"])) for t in tracks
                  if t.get("parent") is None]
    elif strategy == "longest_lived":
        scored = [(t["ID"], len(t.get("t", []))) for t in tracks
                  if t.get("parent") is None]
    else:
        scored = [(t["ID"], 0) for t in tracks if t.get("parent") is None]

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:count]]


def _root_of(tracks, cell_id):
    by_id = {t["ID"]: t for t in tracks}
    cur = cell_id
    while cur in by_id:
        parent = by_id[cur].get("parent")
        if parent is None or parent not in by_id:
            return cur
        cur = parent
    return cell_id


# ============================================================ #
# Trackability (FAST-style)                                     #
# ============================================================ #

def compute_trackability(tracks):
    """Return list of (frame, bits_per_cell) using FAST's formula on
    position features only.  N=2, no feature normalisation."""
    by_frame = {}
    for tr in tracks:
        ts = tr.get("t", [])
        xs = tr.get("x", [])
        ys = tr.get("y", [])
        for i, t in enumerate(ts):
            d = by_frame.setdefault(t, {"pos": [], "disp": {}})
            d["pos"].append((tr["ID"], xs[i], ys[i]))
            if i + 1 < len(ts) and ts[i + 1] == t + 1:
                d["disp"][tr["ID"]] = (xs[i + 1] - xs[i], ys[i + 1] - ys[i])

    out = []
    N = 2
    const = (N / 2.0) * math.log2(6.0 / (math.pi * math.e))
    for t in sorted(by_frame):
        d = by_frame[t]
        if len(d["pos"]) < 5 or len(d["disp"]) < 5:
            continue
        positions = np.array([(p[1], p[2]) for p in d["pos"]])
        displacements = np.array(list(d["disp"].values()))
        try:
            Sx = np.cov(positions.T) + 1e-6 * np.eye(N)
            Sd = np.cov(displacements.T) + 1e-6 * np.eye(N)
            det_x = np.linalg.det(Sx)
            det_d = np.linalg.det(Sd)
            if det_x <= 0 or det_d <= 0:
                continue
            n_o = len(d["pos"])
            r = 0.5 * math.log2(det_x / det_d) + const - math.log2(n_o)
            out.append((t, r))
        except np.linalg.LinAlgError:
            continue
    return out


# ============================================================ #
# Drift                                                         #
# ============================================================ #

def compute_drift(tracks):
    """Return dict: cell_id -> (frames, displacements_px_per_frame)."""
    out = {}
    for tr in tracks:
        ts, xs, ys = tr.get("t", []), tr.get("x", []), tr.get("y", [])
        if len(ts) < 2:
            continue
        frames, disps = [], []
        for i in range(1, len(ts)):
            dt = ts[i] - ts[i - 1]
            if dt == 0:
                continue
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
            frames.append(ts[i])
            disps.append(math.hypot(dx, dy) / dt)
        if disps:
            out[tr["ID"]] = (frames, disps)
    return out


# ============================================================ #
# Self-consistency stats (stand-in for traccuracy until GT)     #
# ============================================================ #

def consistency_stats(tracks):
    """Return dict of pipeline-self-check stats reportable without GT."""
    if not tracks:
        return {}
    by_id = {t["ID"]: t for t in tracks}
    lengths = [len(t.get("t", [])) for t in tracks]
    divisions = sum(1 for t in tracks if len(t.get("children", [])) >= 2)
    roots = sum(1 for t in tracks if t.get("parent") is None
                or t["parent"] not in by_id)
    leaves = sum(1 for t in tracks if not t.get("children"))
    return {
        "Tracks": len(tracks),
        "Roots": roots,
        "Divisions": divisions,
        "Leaves": leaves,
        "Mean length": np.mean(lengths) if lengths else 0,
        "Median length": int(np.median(lengths)) if lengths else 0,
    }


# ============================================================ #
# The tab                                                       #
# ============================================================ #

class TrackValidationTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tracks = []
        self._raw_images = None
        self._selected_families = []
        self._selected_root_ids = []
        self._is_mockup = True

        self._init_ui()
        self._draw_mockup()

    # ---------------------------------------------------------- #
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Controls row
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)

        ctrl.addWidget(QLabel("Pick:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Most divisions", "Longest lived"])
        ctrl.addWidget(self.strategy_combo)

        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 8)
        self.count_spin.setValue(3)
        self.count_spin.setPrefix("n=")
        ctrl.addWidget(self.count_spin)

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self._on_generate)
        self.generate_btn.setEnabled(False)
        self.generate_btn.setStyleSheet(self._btn_style(ACCENT_RED))
        ctrl.addWidget(self.generate_btn)

        ctrl.addSpacing(20)
        ctrl.addWidget(QLabel("Inspect:"))
        self.inspect_combo = QComboBox()
        self.inspect_combo.setEnabled(False)
        self.inspect_combo.setMinimumWidth(120)
        self.inspect_combo.currentIndexChanged.connect(self._on_inspect_changed)
        ctrl.addWidget(self.inspect_combo)

        ctrl.addStretch()

        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self._on_export)
        self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet(self._btn_style(ACCENT_AMBER))
        ctrl.addWidget(self.export_btn)

        layout.addLayout(ctrl)

        # Scrollable matplotlib canvas
        self.fig = plt.figure(figsize=(10, 13), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumSize(900, 1200)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.canvas)
        layout.addWidget(scroll)

    @staticmethod
    def _btn_style(color):
        return f"""
            QPushButton {{
                background-color: {color}; color: white; border: none;
                padding: 6px 14px; border-radius: 5px; font-weight: bold;
            }}
            QPushButton:hover {{ background-color: #888780; }}
            QPushButton:disabled {{
                background-color: #CCCCCC; color: #666666;
            }}
        """

    # ---------------------------------------------------------- #
    # Public API                                                  #
    # ---------------------------------------------------------- #

    def set_data(self, tracks, raw_images=None, mask_images=None):
        self._tracks = tracks or []
        self._raw_images = raw_images
        has_data = bool(self._tracks)
        self.generate_btn.setEnabled(has_data)
        if not has_data:
            self._draw_mockup()

    # ---------------------------------------------------------- #
    # GridSpec helper                                             #
    # ---------------------------------------------------------- #

    def _make_grid(self):
        """Return a dict of axes laid out for the six panels."""
        self.fig.clear()
        gs = GridSpec(
            6, 6, figure=self.fig,
            height_ratios=[0.4, 2.4, 1.6, 1.6, 1.0, 0.05],
            hspace=0.6, wspace=0.5,
        )
        axes = {
            "A": self.fig.add_subplot(gs[0, :]),
            "B": self.fig.add_subplot(gs[1, :]),
            "C": self.fig.add_subplot(gs[2, 0:3]),
            "D": self.fig.add_subplot(gs[2, 3:6]),
            "E": self.fig.add_subplot(gs[3, 0:3]),
            "F": self.fig.add_subplot(gs[3, 3:6]),
            "G1": self.fig.add_subplot(gs[4, 0:2]),
            "G2": self.fig.add_subplot(gs[4, 2:4]),
            "G3": self.fig.add_subplot(gs[4, 4:6]),
        }
        return axes

    # ---------------------------------------------------------- #
    # Mockup                                                      #
    # ---------------------------------------------------------- #

    def _draw_mockup(self):
        self._is_mockup = True
        ax = self._make_grid()

        self._draw_pipeline_strip(ax["A"], current="awaiting tracks")
        self._mock_dense_overlay(ax["B"])
        self._mock_lineage(ax["C"])
        self._mock_trackability(ax["D"])
        self._mock_metrics(ax["E"])
        self._mock_drift(ax["F"])
        self._mock_triplet(ax["G1"], ax["G2"], ax["G3"])

        self.canvas.draw_idle()

    def _mock_dense_overlay(self, ax):
        ax.set_facecolor(MOCK_BG)
        ax.set_title("B   Tracked overlay", loc="left", fontsize=11,
                     fontweight="medium")
        rng = np.random.default_rng(42)
        for _ in range(40):
            x, y = rng.uniform(0.05, 0.95), rng.uniform(0.1, 0.9)
            ang = rng.uniform(0, 180)
            e = mpatches.Ellipse((x, y), 0.04, 0.018, angle=ang,
                                  fc="none", ec=LIGHT_NEUTRAL, lw=0.6)
            ax.add_patch(e)
        for i, (x, y, c) in enumerate([(0.2, 0.3, ACCENT_RED),
                                        (0.5, 0.5, ACCENT_AMBER),
                                        (0.75, 0.7, ACCENT_DEEP)]):
            e = mpatches.Ellipse((x, y), 0.05, 0.022, angle=20,
                                  fc="none", ec=c, lw=1.6)
            ax.add_patch(e)
            ax.text(x, y - 0.05, str(i + 1), color=c, fontsize=9,
                    fontweight="bold", ha="center")
        self._strip_axes(ax)

    def _mock_lineage(self, ax):
        ax.set_facecolor(MOCK_BG)
        ax.set_title("C   Lineage tree", loc="left", fontsize=11,
                     fontweight="medium")
        for offset, color in [(0, ACCENT_RED), (0.5, ACCENT_AMBER)]:
            for xs, ys in [([0.05, 0.2], [0.2 + offset, 0.2 + offset]),
                            ([0.2, 0.2], [0.1 + offset, 0.3 + offset]),
                            ([0.2, 0.45], [0.1 + offset, 0.1 + offset]),
                            ([0.2, 0.45], [0.3 + offset, 0.3 + offset]),
                            ([0.45, 0.75], [0.1 + offset, 0.1 + offset]),
                            ([0.45, 0.75], [0.3 + offset, 0.3 + offset])]:
                ax.plot(xs, ys, color=color, lw=2.2)
        ax.plot([0.45, 0.75], [0.45, 0.45], color="#222", lw=2.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, -0.05, "Time", color=NEUTRAL, fontsize=9,
                ha="center", transform=ax.transAxes)

    def _mock_trackability(self, ax):
        ax.set_facecolor(MOCK_BG)
        ax.set_title("D   Trackability", loc="left", fontsize=11,
                     fontweight="medium")
        x = np.linspace(0, 1, 30)
        y = 8 - np.cumsum(np.random.default_rng(7).normal(0.05, 0.2, 30))
        y = np.clip(y, 1, 10)
        ax.plot(x, y, color=ACCENT_RED, lw=1.6)
        ax.axhline(6, ls="--", color=NEUTRAL, lw=0.7)
        ax.set_ylim(0, 10)
        ax.set_xticks([])
        ax.set_ylabel("bits/cell", fontsize=8, color=NEUTRAL)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def _mock_metrics(self, ax):
        ax.set_facecolor(MOCK_BG)
        ax.set_title("E   Metrics", loc="left", fontsize=11,
                     fontweight="medium")
        ax.text(0.5, 0.5, "awaiting tracks", ha="center", va="center",
                color=NEUTRAL, style="italic")
        self._strip_axes(ax)

    def _mock_drift(self, ax):
        ax.set_facecolor(MOCK_BG)
        ax.set_title("F   Drift", loc="left", fontsize=11,
                     fontweight="medium")
        x = np.linspace(0, 1, 30)
        for c in [ACCENT_RED, ACCENT_AMBER, ACCENT_DEEP]:
            y = 0.1 + np.random.default_rng().normal(0, 0.02, 30).cumsum() * 0.02
            ax.plot(x, np.clip(y, 0, 0.9), color=c, lw=1, alpha=0.7)
        ax.axhline(0.6, ls="--", color=NEUTRAL, lw=0.7)
        ax.set_xticks([])
        ax.set_ylabel("disp/frame", fontsize=8, color=NEUTRAL)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def _mock_triplet(self, ax1, ax2, ax3):
        for ax, label in [(ax1, "previous"), (ax2, "current + ID"),
                          (ax3, "mask overlay")]:
            ax.set_facecolor(MOCK_BG)
            self._strip_axes(ax)
            ax.text(0.5, 0.5, label, ha="center", va="center",
                    color=NEUTRAL, style="italic", fontsize=9)
        ax1.set_title("G   Validation triplet", loc="left", fontsize=11,
                      fontweight="medium")

    # ---------------------------------------------------------- #
    # Real rendering                                              #
    # ---------------------------------------------------------- #

    def _on_generate(self):
        if not self._tracks:
            return
        strategy = "most_divisions" \
            if self.strategy_combo.currentText() == "Most divisions" \
            else "longest_lived"
        roots = _auto_select_roots(self._tracks, strategy,
                                   self.count_spin.value())
        if not roots:
            QMessageBox.warning(self, "No cells",
                                "No root cells matched the chosen strategy.")
            return
        self._selected_root_ids = roots
        fams = []
        seen = set()
        for r in roots:
            for tr in _get_family(self._tracks, r):
                if tr["ID"] not in seen:
                    seen.add(tr["ID"])
                    fams.append(tr)
        self._selected_families = fams

        # populate inspect dropdown
        self.inspect_combo.blockSignals(True)
        self.inspect_combo.clear()
        for tr in fams:
            self.inspect_combo.addItem(f"Cell {tr['ID']}", tr["ID"])
        self.inspect_combo.blockSignals(False)
        self.inspect_combo.setEnabled(True)

        self._draw_real()
        self.export_btn.setEnabled(True)

    def _draw_real(self):
        self._is_mockup = False
        ax = self._make_grid()

        self._draw_pipeline_strip(ax["A"], current="validation")
        self._draw_dense_overlay(ax["B"])
        self._draw_lineage(ax["C"])
        self._draw_trackability(ax["D"])
        self._draw_metrics(ax["E"])
        self._draw_drift(ax["F"])
        self._draw_triplet(ax["G1"], ax["G2"], ax["G3"])

        self.canvas.draw_idle()

    # -- Panel A ----------------------------------------------- #

    def _draw_pipeline_strip(self, ax, current="validation"):
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 1)
        steps = [("ND2", 0.5), ("Segmentation", 1.5), ("Trackastra", 2.5),
                 ("CSV", 3.5), ("Validation", 4.5)]
        done_up_to = {"awaiting tracks": 2, "validation": 5}.get(current, 5)
        for i, (label, x) in enumerate(steps):
            done = i < done_up_to
            color = ACCENT_AMBER if done else "#CCC"
            ax.scatter([x], [0.5], s=160, color=color, zorder=2,
                       edgecolor="white", linewidth=1)
            ax.text(x, 0.5, "\u2713" if done else "\u00b7",
                    ha="center", va="center", color="white",
                    fontsize=11, fontweight="bold", zorder=3)
            ax.text(x, -0.1, label, ha="center", va="top", fontsize=9,
                    color=NEUTRAL if done else "#AAA")
            if i < len(steps) - 1:
                ax.plot([x + 0.15, x + 0.85], [0.5, 0.5],
                        color=ACCENT_AMBER if i < done_up_to - 1 else "#DDD",
                        lw=1.4, zorder=1)
        self._strip_axes(ax)
        ax.set_title("A   Pipeline", loc="left", fontsize=11,
                     fontweight="medium")

    # -- Panel B ----------------------------------------------- #

    def _draw_dense_overlay(self, ax):
        ax.set_title("B   Tracked overlay", loc="left", fontsize=11,
                     fontweight="medium")
        mid_t = self._mid_frame(self._selected_families)

        if (self._raw_images is not None and 0 <= mid_t < len(self._raw_images)):
            ax.imshow(self._raw_images[mid_t], cmap="gray")
            h, w = self._raw_images[mid_t].shape[:2]
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
        else:
            ax.set_facecolor("#1a1a1a")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        # faint outline for every track active at mid_t
        selected_ids = {t["ID"] for t in self._selected_families}
        for tr in self._tracks:
            if mid_t not in tr.get("t", []):
                continue
            idx = tr["t"].index(mid_t)
            if tr["ID"] in selected_ids:
                continue
            ax.plot(tr["x"][idx], tr["y"][idx], "o", ms=3,
                    mfc="none", mec=LIGHT_NEUTRAL, mew=0.5, alpha=0.7)

        # selected families with trail and ID
        for fam_idx, root_id in enumerate(self._selected_root_ids):
            color = FAMILY_COLORS[fam_idx % len(FAMILY_COLORS)]
            for tr in _get_family(self._tracks, root_id):
                if mid_t not in tr.get("t", []):
                    continue
                idx = tr["t"].index(mid_t)
                cx, cy = tr["x"][idx], tr["y"][idx]
                # trail
                start = max(0, idx - 8)
                ax.plot(tr["x"][start:idx + 1], tr["y"][start:idx + 1],
                        "-", color=color, lw=1.3, alpha=0.85)
                # marker + label
                ax.plot(cx, cy, "o", ms=8, mfc="none", mec=color, mew=1.6)
                ax.annotate(str(tr["ID"]), (cx, cy),
                            textcoords="offset points", xytext=(6, 6),
                            fontsize=8, color=color, fontweight="bold")

        ax.set_xticks([])
        ax.set_yticks([])

    # -- Panel C ----------------------------------------------- #

    def _draw_lineage(self, ax):
        ax.set_title("C   Lineage tree", loc="left", fontsize=11,
                     fontweight="medium")
        if not self._selected_families:
            self._strip_axes(ax)
            return

        by_id = {t["ID"]: t for t in self._selected_families}
        root_set = set(self._selected_root_ids)

        # Layout each family on its own y-band
        y_pos = {}
        y_counter = [0]

        def _layout(tid):
            if tid not in by_id:
                return y_counter[0]
            children = [c for c in by_id[tid].get("children", []) if c in by_id]
            if not children:
                y = y_counter[0]
                y_counter[0] += 1
                y_pos[tid] = y
                return y
            child_ys = [_layout(c) for c in children]
            y = sum(child_ys) / len(child_ys)
            y_pos[tid] = y
            return y

        for r in self._selected_root_ids:
            _layout(r)
            y_counter[0] += 0.6  # gap between families

        # detect "unassigned" (root id not in selected set means an orphan
        # subtree, which here means dangling short tracks)
        median_len = np.median([len(t.get("t", []))
                                for t in self._selected_families])

        for tr in self._selected_families:
            tid = tr["ID"]
            if tid not in y_pos:
                continue
            y = y_pos[tid]
            ts = tr.get("t", [])
            if not ts:
                continue
            t_start, t_end = min(ts), max(ts)
            root = _root_of(self._tracks, tid)
            # short stranded leaves marked as suspicious in black
            short_dangling = (
                len(ts) < max(2, median_len * 0.25)
                and not tr.get("children")
                and tr.get("parent") not in by_id
            )
            if short_dangling:
                color = "#222"
            elif root in root_set:
                idx = self._selected_root_ids.index(root)
                color = FAMILY_COLORS[idx % len(FAMILY_COLORS)]
            else:
                color = "#222"

            ax.plot([t_start, t_end], [y, y], color=color, lw=2.2,
                    solid_capstyle="round")
            children = [c for c in tr.get("children", []) if c in y_pos]
            if children:
                ys = [y_pos[c] for c in children]
                ax.plot([t_end, t_end], [min(ys), max(ys)], color=color,
                        lw=1.4)

        ax.set_xlabel("Frame", fontsize=9, color=NEUTRAL)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)

    # -- Panel D ----------------------------------------------- #

    def _draw_trackability(self, ax):
        ax.set_title("D   Trackability", loc="left", fontsize=11,
                     fontweight="medium")
        data = compute_trackability(self._tracks)
        if not data:
            ax.text(0.5, 0.5, "need 5+ cells/frame",
                    ha="center", va="center", color=NEUTRAL, style="italic",
                    transform=ax.transAxes)
            self._strip_axes(ax)
            return
        frames, bits = zip(*data)
        ax.plot(frames, bits, color=ACCENT_RED, lw=1.6)
        ax.fill_between(frames, bits, [min(bits)] * len(bits),
                        color=ACCENT_RED, alpha=0.08)
        ax.axhline(6, ls="--", color=NEUTRAL, lw=0.7)
        ax.text(frames[-1], 6.2, "trust", fontsize=7, color=NEUTRAL,
                ha="right")
        ax.set_xlabel("Frame", fontsize=9, color=NEUTRAL)
        ax.set_ylabel("bits/cell", fontsize=9, color=NEUTRAL)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # -- Panel E ----------------------------------------------- #

    def _draw_metrics(self, ax):
        ax.set_title("E   Metrics", loc="left", fontsize=11,
                     fontweight="medium")
        stats = consistency_stats(self._tracks)
        labels, values = [], []
        order = ["Tracks", "Roots", "Divisions", "Leaves",
                 "Mean length", "Median length"]
        for k in order:
            if k in stats:
                labels.append(k)
                values.append(stats[k])

        y_pos = np.arange(len(labels))
        max_val = max(values) if values else 1
        ax.barh(y_pos, [max_val] * len(values), color=ACCENT_AMBER,
                alpha=0.12, height=0.6)
        ax.barh(y_pos, values, color=ACCENT_AMBER, height=0.6)
        for i, v in enumerate(values):
            ax.text(max_val * 1.02, i, f"{v:.0f}" if isinstance(v, float) else str(v),
                    va="center", fontsize=9, color=ACCENT_DEEP, fontweight="bold")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_xlim(0, max_val * 1.18)
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.text(1.0, -0.16,
                "self-consistency (no GT). traccuracy fields populate with digital twin.",
                ha="right", va="top", fontsize=7, color=NEUTRAL,
                style="italic", transform=ax.transAxes)

    # -- Panel F ----------------------------------------------- #

    def _draw_drift(self, ax):
        ax.set_title("F   Drift", loc="left", fontsize=11,
                     fontweight="medium")
        drift = compute_drift(self._selected_families)
        if not drift:
            ax.text(0.5, 0.5, "no drift data", ha="center", va="center",
                    color=NEUTRAL, style="italic", transform=ax.transAxes)
            self._strip_axes(ax)
            return
        all_d = []
        for fam_idx, root_id in enumerate(self._selected_root_ids):
            color = FAMILY_COLORS[fam_idx % len(FAMILY_COLORS)]
            for tr in _get_family(self._tracks, root_id):
                if tr["ID"] not in drift:
                    continue
                fr, d = drift[tr["ID"]]
                all_d.extend(d)
                ax.plot(fr, d, "-", color=color, lw=1, alpha=0.75)
        if all_d:
            thresh = float(np.percentile(all_d, 95))
            ax.axhline(thresh, ls="--", color=NEUTRAL, lw=0.7)
            ax.text(ax.get_xlim()[1], thresh * 1.05,
                    f"p95={thresh:.1f}", fontsize=7, color=NEUTRAL,
                    ha="right")
        ax.set_xlabel("Frame", fontsize=9, color=NEUTRAL)
        ax.set_ylabel("disp px/frame", fontsize=9, color=NEUTRAL)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # -- Panel G ----------------------------------------------- #

    def _draw_triplet(self, ax_prev, ax_curr, ax_mask):
        ax_prev.set_title("G   Validation triplet", loc="left", fontsize=11,
                          fontweight="medium")

        cell_id = self.inspect_combo.currentData()
        track = next((t for t in self._selected_families
                      if t["ID"] == cell_id), None)
        if track is None and self._selected_families:
            track = self._selected_families[0]
            cell_id = track["ID"]
        if track is None or self._raw_images is None:
            for ax, label in [(ax_prev, "previous"),
                              (ax_curr, "current + ID"),
                              (ax_mask, "mask overlay")]:
                ax.text(0.5, 0.5, label, ha="center", va="center",
                        color=NEUTRAL, style="italic", fontsize=9,
                        transform=ax.transAxes)
                self._strip_axes(ax)
            return

        ts = track.get("t", [])
        xs = track.get("x", [])
        ys = track.get("y", [])
        if len(ts) < 2:
            for ax in (ax_prev, ax_curr, ax_mask):
                self._strip_axes(ax)
            return

        # pick middle index with a previous neighbour
        mid_i = max(1, len(ts) // 2)
        t_prev, t_curr = ts[mid_i - 1], ts[mid_i]
        cx_prev, cy_prev = xs[mid_i - 1], ys[mid_i - 1]
        cx_curr, cy_curr = xs[mid_i], ys[mid_i]

        if t_prev >= len(self._raw_images) or t_curr >= len(self._raw_images):
            for ax in (ax_prev, ax_curr, ax_mask):
                self._strip_axes(ax)
            return

        H, W = self._raw_images[t_curr].shape[:2]
        crop = 60
        x0, y0 = max(0, int(cx_curr) - crop), max(0, int(cy_curr) - crop)
        x1, y1 = min(W, int(cx_curr) + crop), min(H, int(cy_curr) + crop)

        prev_crop = self._raw_images[t_prev][y0:y1, x0:x1]
        curr_crop = self._raw_images[t_curr][y0:y1, x0:x1]

        ax_prev.imshow(prev_crop, cmap="gray")
        ax_prev.text(0.02, 0.98, f"t={t_prev}", transform=ax_prev.transAxes,
                     fontsize=8, color="white", va="top",
                     bbox=dict(boxstyle="round", fc="black", alpha=0.5,
                               ec="none"))
        ax_prev.plot(cx_prev - x0, cy_prev - y0, "o", ms=8,
                     mfc="none", mec=ACCENT_AMBER, mew=1.4)
        ax_prev.set_xlabel("previous", fontsize=9, color=NEUTRAL)

        ax_curr.imshow(curr_crop, cmap="gray")
        ax_curr.text(0.02, 0.98, f"t={t_curr}", transform=ax_curr.transAxes,
                     fontsize=8, color="white", va="top",
                     bbox=dict(boxstyle="round", fc="black", alpha=0.5,
                               ec="none"))
        ax_curr.plot(cx_curr - x0, cy_curr - y0, "o", ms=8,
                     mfc="none", mec=ACCENT_RED, mew=1.6)
        ax_curr.annotate(str(cell_id), (cx_curr - x0, cy_curr - y0),
                         textcoords="offset points", xytext=(6, 6),
                         fontsize=9, color=ACCENT_RED, fontweight="bold")
        ax_curr.set_xlabel(f"current (ID {cell_id})", fontsize=9, color=NEUTRAL)

        ax_mask.imshow(curr_crop, cmap="gray", alpha=0.45)
        circle = mpatches.Circle((cx_curr - x0, cy_curr - y0), 8,
                                  fill=False, ec=ACCENT_AMBER, lw=1.6,
                                  linestyle="-")
        ax_mask.add_patch(circle)
        gt_circle = mpatches.Circle((cx_curr - x0, cy_curr - y0), 11,
                                     fill=False, ec="white", lw=0.8,
                                     linestyle="--")
        ax_mask.add_patch(gt_circle)
        ax_mask.set_xlabel("mask + GT", fontsize=9, color=NEUTRAL)

        for ax in (ax_prev, ax_curr, ax_mask):
            ax.set_xticks([])
            ax.set_yticks([])

    # ---------------------------------------------------------- #
    # Inspect dropdown                                            #
    # ---------------------------------------------------------- #

    def _on_inspect_changed(self):
        if self._is_mockup or not self._selected_families:
            return
        self._draw_real()

    # ---------------------------------------------------------- #
    # Utilities                                                   #
    # ---------------------------------------------------------- #

    @staticmethod
    def _strip_axes(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)

    def _mid_frame(self, tracks):
        all_t = []
        for t in tracks:
            all_t.extend(t.get("t", []))
        return int(np.median(all_t)) if all_t else 0

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

        csv_path = out / "tracking_validation.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cell_id", "frame", "x", "y", "parent_id", "children"])
            for tr in self._selected_families:
                parent = tr.get("parent")
                children = ";".join(str(c) for c in tr.get("children", []))
                for i, t in enumerate(tr.get("t", [])):
                    w.writerow([tr["ID"], t, tr["x"][i], tr["y"][i],
                                parent if parent is not None else "",
                                children])

        fig_path = out / "tracking_validation.png"
        self.fig.savefig(fig_path, dpi=200, bbox_inches="tight",
                         facecolor="white")

        QMessageBox.information(
            self, "Export complete",
            f"Saved to {out}:\n  tracking_validation.csv\n  tracking_validation.png"
        )
