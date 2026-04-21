"""
Digital Twin tab — validates single-cell bacterial growth and morphology
predictions against observed microscopy data.

Sits as a third tab in the Tracking widget alongside Cell View and
Environment View.  Pulls lineage_tracks + metrics_service from the
tracking widget, extracts real cell parameters via CellHistoryBuilder,
runs Viva-munk simulations, and displays comparison figures.

Without COMSOL flow data, validates Var1 (growth), Var2 (morphology),
Var3 (motility), and lineage variability at the population level.
Once COMSOL is available, adds per-flow-zone comparisons.
"""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DT_OUTPUT_DIR = str(PROJECT_ROOT / "digital_twin_output")


class SimulationWorker(QThread):
    """Runs the digital twin pipeline in a background thread."""
    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, cell_database, time_interval_s, pixel_to_um, experiment_type):
        super().__init__()
        self.cell_database = cell_database
        self.time_interval_s = time_interval_s
        self.pixel_to_um = pixel_to_um
        self.experiment_type = experiment_type

    def run(self):
        try:
            from nd2_analyzer.analysis.digital_twin import (
                ParameterExtractor,
                DigitalTwinRunner,
                ComparisonFigureGenerator,
            )

            self.progress.emit("Extracting real cell parameters...")
            extractor = ParameterExtractor(self.cell_database)
            params = extractor.extract_all(self.time_interval_s)
            self.progress.emit(
                f"Extracted: growth rate={params.get('growth_rate_mean', 0)*3600:.4f}/h, "
                f"doubling={params.get('doubling_time_min', 0):.1f}min, "
                f"{params.get('n_cells', 0)} cells"
            )

            self.progress.emit("Running calibrated simulation...")
            runner = DigitalTwinRunner(params, self.pixel_to_um)
            config = runner.build_calibrated_config()
            sim_results = runner.run_simulation(self.experiment_type)
            self.progress.emit(
                f"Calibrated sim done: {sim_results.get('final_cell_count', 0)} cells"
            )

            self.progress.emit("Running default simulation for comparison...")
            default_results = runner.run_default_simulation()
            self.progress.emit(
                f"Default sim done: {default_results.get('final_cell_count', 0)} cells"
            )

            self.progress.emit("Generating comparison figures...")
            generator = ComparisonFigureGenerator(
                params, sim_results, default_results, self.cell_database,
            )
            figure_paths = generator.generate_all()
            self.progress.emit(f"Generated {len(figure_paths)} figures")

            self.finished.emit({
                'params': params,
                'config': config,
                'sim_results': sim_results,
                'default_sim_results': default_results,
                'figure_paths': figure_paths,
            })

        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class DigitalTwinTab(QWidget):
    """Digital Twin analysis tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lineage_tracks = None
        self._metrics_service = None
        self._cell_database = None
        self._results = None
        self._worker = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- Header --- #
        header = QLabel(
            "Digital Twin: Validate single-cell growth, morphology, "
            "and motility predictions against observed data"
        )
        header.setWordWrap(True)
        header.setStyleSheet("font-size: 13px; color: #555; margin-bottom: 8px;")
        layout.addWidget(header)

        # --- Configuration --- #
        config_group = QGroupBox("Simulation Parameters")
        config_layout = QHBoxLayout(config_group)

        config_layout.addWidget(QLabel("Frame interval (s):"))
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(1.0, 7200.0)
        self.interval_spin.setValue(300.0)
        self.interval_spin.setToolTip("Time between microscopy frames in seconds")
        config_layout.addWidget(self.interval_spin)

        config_layout.addSpacing(15)

        config_layout.addWidget(QLabel("Pixel → µm:"))
        self.px_to_um_spin = QDoubleSpinBox()
        self.px_to_um_spin.setRange(0.001, 10.0)
        self.px_to_um_spin.setValue(0.065)
        self.px_to_um_spin.setDecimals(4)
        self.px_to_um_spin.setToolTip("Pixel to micrometer conversion factor")
        config_layout.addWidget(self.px_to_um_spin)

        config_layout.addSpacing(15)

        config_layout.addWidget(QLabel("Experiment:"))
        self.experiment_combo = QComboBox()
        self.experiment_combo.addItems(["daughter_machine", "mother_machine"])
        self.experiment_combo.setToolTip("Viva-munk experiment type to simulate")
        config_layout.addWidget(self.experiment_combo)

        config_layout.addStretch()
        layout.addWidget(config_group)

        # --- Action buttons --- #
        btn_row = QHBoxLayout()

        self.run_button = QPushButton("Run Digital Twin Pipeline")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._run_pipeline)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #7B2FBE;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #6A28A6; }
            QPushButton:pressed { background-color: #5A2090; }
            QPushButton:disabled { background-color: #CCCCCC; color: #666666; }
        """)
        btn_row.addWidget(self.run_button)

        self.open_folder_button = QPushButton("Open Output Folder")
        self.open_folder_button.setEnabled(False)
        self.open_folder_button.clicked.connect(self._open_output_folder)
        btn_row.addWidget(self.open_folder_button)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # --- Status / log --- #
        self.status_label = QLabel("Load tracking data and run tracking first.")
        self.status_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self.log_text)

        # --- Parameter summary (filled after extraction) --- #
        self.params_group = QGroupBox("Extracted Parameters (Observed)")
        self.params_group.hide()
        params_layout = QVBoxLayout(self.params_group)
        self.params_label = QLabel()
        self.params_label.setWordWrap(True)
        self.params_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        params_layout.addWidget(self.params_label)
        layout.addWidget(self.params_group)

        # --- Figures display --- #
        self.figures_scroll = QScrollArea()
        self.figures_scroll.setWidgetResizable(True)
        self.figures_scroll.hide()

        self.figures_container = QWidget()
        self.figures_layout = QVBoxLayout(self.figures_container)
        self.figures_scroll.setWidget(self.figures_container)
        layout.addWidget(self.figures_scroll, stretch=1)

        self.setLayout(layout)

    # ------------------------------------------------------------------ #
    # Public API — called by TrackingWidget when data becomes available
    # ------------------------------------------------------------------ #
    def set_tracking_data(self, lineage_tracks, metrics_service):
        """Receive tracking data from the parent TrackingWidget."""
        self._lineage_tracks = lineage_tracks
        self._metrics_service = metrics_service
        self._cell_database = None

        has_data = lineage_tracks is not None and len(lineage_tracks) > 0
        self.run_button.setEnabled(has_data)
        if has_data:
            self.status_label.setText(
                f"Ready — {len(lineage_tracks)} tracks available. "
                "Click 'Run Digital Twin Pipeline' to start."
            )
            self.status_label.setStyleSheet("color: #2E7D32; font-weight: bold;")
        else:
            self.status_label.setText("No tracking data available yet.")
            self.status_label.setStyleSheet("color: #888; font-style: italic;")

    # ------------------------------------------------------------------ #
    # Pipeline execution
    # ------------------------------------------------------------------ #
    def _run_pipeline(self):
        if not self._lineage_tracks or not self._metrics_service:
            QMessageBox.warning(self, "No Data", "Run cell tracking first.")
            return

        self.run_button.setEnabled(False)
        self.progress_bar.show()
        self.log_text.clear()
        self._log("Building cell database from tracking data...")

        self._build_cell_database()

        if not self._cell_database or len(self._cell_database) == 0:
            self._log("ERROR: No cells in database. Check tracking data.")
            self.run_button.setEnabled(True)
            self.progress_bar.hide()
            return

        self._log(f"Cell database: {len(self._cell_database)} cells")

        self._worker = SimulationWorker(
            self._cell_database,
            self.interval_spin.value(),
            self.px_to_um_spin.value(),
            self.experiment_combo.currentText(),
        )
        self._worker.progress.connect(self._log)
        self._worker.finished.connect(self._on_pipeline_done)
        self._worker.error.connect(self._on_pipeline_error)
        self._worker.start()

    def _build_cell_database(self):
        """Build cell histories from tracking + metrics data."""
        from nd2_analyzer.analysis.cell_history import CellHistoryBuilder

        builder = CellHistoryBuilder(self._lineage_tracks, self._metrics_service)
        self._cell_database = builder.build(min_track_length=5)

    def _on_pipeline_done(self, results: dict):
        self._results = results
        self.progress_bar.hide()
        self.run_button.setEnabled(True)
        self.open_folder_button.setEnabled(True)

        self._log("Pipeline complete!")
        self.status_label.setText("Digital twin pipeline finished successfully.")
        self.status_label.setStyleSheet("color: #2E7D32; font-weight: bold;")

        self._show_params(results['params'])
        self._show_figures(results['figure_paths'])

    def _on_pipeline_error(self, error_msg: str):
        self.progress_bar.hide()
        self.run_button.setEnabled(True)
        self._log(f"ERROR: {error_msg}")
        self.status_label.setText("Pipeline failed — see log for details.")
        self.status_label.setStyleSheet("color: #C62828; font-weight: bold;")

    # ------------------------------------------------------------------ #
    # Display helpers
    # ------------------------------------------------------------------ #
    def _show_params(self, params: dict):
        lines = []
        lines.append(f"Cells analyzed:        {params.get('n_cells', 0)}")
        lines.append(f"Cells with growth:     {params.get('n_cells_with_growth', 0)}")
        lines.append(f"Dividing cells:        {params.get('n_dividing_cells', 0)}")
        lines.append("")
        lines.append(f"Growth rate:           {params.get('growth_rate_mean', 0)*3600:.4f} /h")
        lines.append(f"Doubling time:         {params.get('doubling_time_min', 0):.1f} min")
        lines.append(f"Interdivision time:    {params.get('interdivision_time_mean_min', 0):.1f} min")
        lines.append("")
        lines.append(f"Cell length (px):      {params.get('cell_length_mean_px', 0):.1f} +/- {params.get('cell_length_std_px', 0):.1f}")
        lines.append(f"Cell width (px):       {params.get('cell_width_mean_px', 0):.1f} +/- {params.get('cell_width_std_px', 0):.1f}")
        lines.append(f"Aspect ratio:          {params.get('aspect_ratio_mean', 0):.2f}")
        lines.append("")
        lines.append(f"Avg velocity (px/f):   {params.get('avg_velocity_px_per_frame', 0):.2f}")
        lines.append(f"Avg displacement (px): {params.get('avg_displacement_px', 0):.1f}")
        lines.append(f"Avg directionality:    {params.get('avg_directionality', 0):.3f}")

        self.params_label.setText("\n".join(lines))
        self.params_group.show()

    def _show_figures(self, figure_paths: list):
        while self.figures_layout.count():
            item = self.figures_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for path in figure_paths:
            if not os.path.exists(path):
                continue

            name = os.path.basename(path).replace('fig_', '').replace('.png', '').replace('_', ' ').title()
            title = QLabel(name)
            title.setStyleSheet("font-weight: bold; font-size: 13px; margin-top: 10px;")
            self.figures_layout.addWidget(title)

            pixmap = QPixmap(path)
            if not pixmap.isNull():
                img_label = QLabel()
                scaled = pixmap.scaledToWidth(
                    min(900, self.width() - 40),
                    Qt.SmoothTransformation,
                )
                img_label.setPixmap(scaled)
                img_label.setAlignment(Qt.AlignCenter)
                self.figures_layout.addWidget(img_label)

        self.figures_layout.addStretch()
        self.figures_scroll.show()

    def _open_output_folder(self):
        import subprocess
        os.makedirs(DT_OUTPUT_DIR, exist_ok=True)
        subprocess.Popen(['open', DT_OUTPUT_DIR])

    def _log(self, msg: str):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
