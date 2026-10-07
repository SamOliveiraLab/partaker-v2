from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QSlider, QSpinBox, QComboBox,
                               QGroupBox, QListWidget, QProgressBar,
                               QCheckBox, QFrame, QTextEdit, QSplitter,
                               QFileDialog, QAbstractItemView, QMessageBox,
                               QDoubleSpinBox, QDialog)
from PySide6.QtCore import Qt, Signal, QThread, QObject
from PySide6.QtGui import QFont
import os
import traceback
from pubsub import pub
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from nd2_analyzer.analysis.metrics_service import MetricsService
import polars as pl  # Import Polars
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.stats import ks_2samp
from skimage.morphology import disk, binary_closing, binary_dilation
from skimage.measure import label as label_connected_components, regionprops
from skimage.segmentation import find_boundaries
from PIL import Image, ImageDraw, ImageFont
import csv
import json
import hashlib
import uuid
from datetime import datetime, timezone
import gc
import shutil


class LiveDeadAnalysisWidget(QWidget):
    """Widget for cube-based analysis of exported colony time series"""

    # Shared run-level Live-Dead intensity mapping.
    # The black/white points are estimated once from representative frames and
    # then reused for every selected timepoint and position. This prevents
    # per-frame contrast stretching while preserving substantially more useful
    # Live contrast than blind 16-bit -> 8-bit division by 257.
    LIVE_WINDOW_BLACK_PERCENTILE = 25.0
    LIVE_WINDOW_WHITE_PERCENTILE = 99.5
    LIVE_WINDOW_WHITE_ACROSS_FRAMES_PERCENTILE = 90.0
    LIVE_WINDOW_MAX_SAMPLE_FRAMES = 12
    LIVE_BACKGROUND_SIGMA = 200.0

    # Fixed fluorescence-cell threshold settings. Phase-contrast cell masks
    # continue to come from the existing Partaker segmentation cache.
    CELL_THRESHOLD = 165
    CELL_MORPHOLOGY_KERNEL_DIAMETER_UM = 0.65
    CELL_FALLBACK_PIXEL_SIZE_UM = 0.064971092928405

    EXPORT_GIF_DELAY_MS = 1000

    # Export options
    EXPORT_VISUALS_ONLY_FOR_ONE_POSITION = True
    EXPORT_VISUAL_POSITION = 0  # zero-based, so Position 0, Position 1, etc.
    LIVE_OVERLAY_RGB = (0, 255, 0)
    DEAD_OVERLAY_RGB = (255, 0, 0)


    # Both whole-frame and cell-integrated graphs are always exported.
    # Leave one line active to choose which metric appears in the in-app view.

    def __init__(self, parent=None):
        super().__init__(parent)

        # State variables
        self.base_folder = ""

        # Initial background filtered reference image
        # self.reference_backgrounds = {}

        # Segmentation
        self.is_segmenting = False
        self.cancel_requested = False
        self.queue = []
        self.live_dead_results = []
        self.run_analysis_mode = "live_dead"
        self.individual_export_context = None
        self.individual_export_error = None

        # Partaker fixed Live-Dead calibration state. Both fluorescence
        # channels share one threshold, smoothing value, and intensity mapping.
        # threshold and one shared raw-to-uint8 mapping that must be reused by
        # the complete run so the accepted preview matches exported masks.
        self.partaker_live_dead_setup = None
        self.partaker_live_dead_threshold = 40
        self.partaker_live_dead_smoothing_sigma = 1.5
        self.partaker_drop_frame_zero = False

        # Captures the frame-zero decision used by the current run, regardless
        # of whether it came from the Partaker setup dialog or the main widget.
        self.drop_frame_zero_for_current_run = False

        # Run-level Live-Dead preprocessing state.
        self.live_dead_processing_black_point_raw = None
        self.live_dead_processing_white_point_raw = None
        self.live_dead_processing_window_source = "uninitialized"
        self.live_dead_reference_backgrounds = {}

        # Segmentation Event Handler
        from PySide6.QtCore import QTimer
        self.request_timer = QTimer(self)
        self.request_timer.setSingleShot(True)
        self.request_timer.timeout.connect(self.process_next_in_queue)

        self.init_ui()
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")

    def init_ui(self):
        """Initialize the user interface"""
        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)

        # Left side - Configuration
        left_widget = self.create_configuration_panel()
        main_splitter.addWidget(left_widget)

        # Right side - Results
        right_widget = self.create_results_panel()
        main_splitter.addWidget(right_widget)

        # Set splitter proportions (30% left, 70% right)
        main_splitter.setSizes([300, 700])

        # Main layout
        layout = QVBoxLayout(self)
        layout.addWidget(main_splitter)

    def create_configuration_panel(self):
        """Create the left configuration panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Title
        self.configuration_title = title_label = QLabel("Live/Dead Configuration")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2196F3; margin-bottom: 10px;")
        layout.addWidget(title_label)

        # Input Selection Group
        input_group = self.create_input_selection_group()
        layout.addWidget(input_group)

        # Select Position Group
        position_group = self.select_stage_position()
        layout.addWidget(position_group)

        # Processing Controls Group
        processing_group = self.create_processing_group()
        layout.addWidget(processing_group)

        # Stretch to push everything to top
        layout.addStretch()

        return widget

    def create_input_selection_group(self):
        """Create input selection group"""
        group = QGroupBox("1. Select Analysis Channels")
        layout = QVBoxLayout(group)

        # Select channels for Segmentation
        selection_layout = QVBoxLayout()

        self.cell_viability_toggle = QCheckBox("Cell Viability")
        self.cell_viability_toggle.setToolTip(
            "Analyze one fluorescence channel using the same fixed-threshold workflow."
        )
        selection_layout.addWidget(self.cell_viability_toggle)
        self.live_dead_channel_panel = QWidget()
        paired_layout = QVBoxLayout(self.live_dead_channel_panel)
        paired_layout.setContentsMargins(0, 0, 0, 0)
        paired_layout.addWidget(QLabel("Live Channel:"))
        self.live_channel_combo = QComboBox()
        paired_layout.addWidget(self.live_channel_combo)
        paired_layout.addWidget(QLabel("Dead Channel:"))
        self.dead_channel_combo = QComboBox()
        self.dead_channel_combo.setToolTip(
            "Dead fluorescence channel analyzed alongside the Live channel."
        )
        paired_layout.addWidget(self.dead_channel_combo)
        selection_layout.addWidget(self.live_dead_channel_panel)

        self.cell_viability_channel_panel = QWidget()
        viability_layout = QVBoxLayout(self.cell_viability_channel_panel)
        viability_layout.setContentsMargins(0, 0, 0, 0)
        viability_layout.addWidget(QLabel("Cell Viability Channel:"))
        self.cell_viability_channel_combo = QComboBox()
        viability_layout.addWidget(self.cell_viability_channel_combo)
        self.cell_viability_channel_panel.hide()
        selection_layout.addWidget(self.cell_viability_channel_panel)
        self.cell_viability_toggle.toggled.connect(self.on_workflow_mode_changed)

        # Cell channel selection
        selection_layout.addWidget(QLabel("Cell Channel:"))
        self.cell_channel_combo = QComboBox()
        selection_layout.addWidget(self.cell_channel_combo)

        # Cell view type selection
        selection_layout.addWidget(QLabel("Select Cell View Type:"))
        self.cell_view_combo = QComboBox()
        self.cell_view_combo.addItems(["Fluorescence", "Phase Contrast"])
        self.cell_view_combo.setCurrentText("Phase Contrast")
        selection_layout.addWidget(self.cell_view_combo)

        # Save Postitive Fluorescence Masks Selection
        self.save_fluorescence_masks_checkbox = QCheckBox("Save fluorescence-positive masks")
        self.save_fluorescence_masks_checkbox.setChecked(True)
        self.save_fluorescence_masks_checkbox.setToolTip(
            "Save compressed cell labels and fluorescence masks for every analyzed frame."
        )
        selection_layout.addWidget(self.save_fluorescence_masks_checkbox)

        # Optional preprocessing for Partaker fixed-threshold segmentation.
        self.gaus_back_corr = QCheckBox("Gaussian Background Subtraction")
        self.gaus_back_corr.setChecked(False)
        self.close_dialate = QCheckBox(
            "Morphologically Close & Dilate Live and Dead Channels"
        )
        self.close_dialate.setChecked(False)

        self.drop_frame_zero_checkbox = QCheckBox(
            "Drop frame 0 from analysis"
        )
        self.drop_frame_zero_checkbox.setChecked(False)
        self.drop_frame_zero_checkbox.setToolTip(
            "Exclude T=0 from processing, exported metrics, plots, and GIFs. "
            "The Partaker threshold setup dialog also provides this option."
        )

        self.optional_preprocessing_checkboxes = [
            self.gaus_back_corr,
            self.close_dialate,
            self.drop_frame_zero_checkbox,
        ]

        for checkbox in self.optional_preprocessing_checkboxes:
            # Keep disabled text readable instead of allowing the platform theme
            # to fade it almost completely into the background.
            checkbox.setStyleSheet(
                "QCheckBox:disabled { color: #777777; }"
            )
            selection_layout.addWidget(checkbox)

        self.on_analysis_method_changed()

        layout.addLayout(selection_layout)

        return group

    def selected_analysis_mode(self):
        return "cell_viability" if self.cell_viability_toggle.isChecked() else "live_dead"

    def is_cell_viability_run(self):
        return self.run_analysis_mode == "cell_viability"

    def analysis_label(self):
        return "Cell Viability" if self.is_cell_viability_run() else "Live-Dead"

    def output_prefix(self):
        return self.run_analysis_mode

    def active_channel_specs(self):
        if self.is_cell_viability_run():
            return (("cell_viability", "Cell Viability", self.LIVE_OVERLAY_RGB),)
        return (("live", "Live", self.LIVE_OVERLAY_RGB),
                ("dead", "Dead", self.DEAD_OVERLAY_RGB))

    def on_workflow_mode_changed(self, checked):
        self.live_dead_channel_panel.setVisible(not checked)
        self.cell_viability_channel_panel.setVisible(checked)
        self.configuration_title.setText(
            "Cell Viability Configuration" if checked else "Live/Dead Configuration"
        )
        self.close_dialate.setText(
            "Morphologically Close & Dilate Cell Viability Channel" if checked
            else "Morphologically Close & Dilate Live and Dead Channels"
        )
        # Existing results keep their run mode, plot choices, and export names.
        if not self.live_dead_results:
            self.run_analysis_mode = self.selected_analysis_mode()
            self.refresh_metric_choices()

    def refresh_metric_choices(self):
        if not hasattr(self, "metric_combo"):
            return
        previous = self.metric_combo.currentText()
        self.metric_combo.clear()
        choices = [
            "Cell-Area Staining Composition", "Biomass Retention",
            "Mean Live-Dead Intensity", "Integrated Live-Dead Intensity",
            "Mean Live-Dead Intensity Error Plot",
            "Integrated Live-Dead Intensity Error Plot",
            "Live-Dead Fractional Area", "Mean Cell-Based Intensity",
        ]
        if self.is_cell_viability_run():
            choices = [name.replace("Live-Dead", "Cell Viability") for name in choices]
        else:
            choices.extend([
                "Mean Live-Dead Ratio Comparison", "Integrated Live-Dead Ratio Comparison",
                "Mean Live-Dead Ratio Comparison Error Plot",
                "Integrated Live-Dead Ratio Comparison Error Plot",
            ])
        self.metric_combo.addItems(choices)
        if previous in choices:
            self.metric_combo.setCurrentText(previous)

    def on_analysis_method_changed(self):
        """Keep optional preprocessing available for the fixed workflow."""
        for checkbox in getattr(
                self,
                "optional_preprocessing_checkboxes",
                [],
        ):
            checkbox.setVisible(True)
            checkbox.setEnabled(True)

        if hasattr(self, "start_segmentation_btn"):
            self.start_segmentation_btn.setText("Start")

    def get_analysis_method(self):
        """Return the single supported Live-Dead analysis method."""
        return "Partaker"

    def get_selected_cell_channel(self):
        """Return the selected channel used to identify cells."""
        if hasattr(self, "cell_channel_combo") and self.cell_channel_combo.count() > 0:
            return int(self.cell_channel_combo.currentIndex())
        return 0

    def is_partaker_method(self):
        """Return True when the current implementation / segmented-cell method is selected."""
        return self.get_analysis_method() == "Partaker"

    def select_stage_position(self):
        """Create cube configuration group"""
        group = QGroupBox("2. Select Positions")
        layout = QVBoxLayout(group)

        # Position selection (existing code)
        size_layout = QVBoxLayout()

        # Position list with checkboxes
        self.position_list = QListWidget()
        self.position_list.setSelectionMode(QAbstractItemView.MultiSelection)
        size_layout.addWidget(self.position_list)

        # Quick selection buttons
        pos_buttons_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_positions)
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self.select_no_positions)
        pos_buttons_layout.addWidget(select_all_btn)
        pos_buttons_layout.addWidget(select_none_btn)
        size_layout.addLayout(pos_buttons_layout)

        # Time range selection
        size_layout.addWidget(QLabel("Time Range"))
        time_layout = QHBoxLayout()

        time_layout.addWidget(QLabel("From:"))
        self.time_start_spin = QSpinBox()
        self.time_start_spin.setMinimum(0)
        time_layout.addWidget(self.time_start_spin)

        time_layout.addWidget(QLabel("To:"))
        self.time_end_spin = QSpinBox()
        self.time_end_spin.setMinimum(0)
        time_layout.addWidget(self.time_end_spin)

        size_layout.addLayout(time_layout)

        layout.addLayout(size_layout)

        return group

    def create_processing_group(self):
        """Create processing controls group"""
        group = QGroupBox("3. Processing")
        layout = QVBoxLayout(group)

        # Control buttons
        button_layout = QHBoxLayout()

        # Segment Channel selector and button
        self.start_segmentation_btn = QPushButton("Start")
        self.start_segmentation_btn.clicked.connect(self.segment_selected_channels)
        self.start_segmentation_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 5px;")
        self.start_segmentation_btn.setEnabled(False)
        self.on_analysis_method_changed()

        # Cancel button
        self.cancel_analysis_btn = QPushButton("Cancel")
        self.cancel_analysis_btn.clicked.connect(self.cancel_analysis)
        self.cancel_analysis_btn.setStyleSheet(
            "background-color: #f44336; color: white; font-weight: bold; padding: 5px;")
        self.cancel_analysis_btn.setEnabled(False)

        button_layout.addWidget(self.start_segmentation_btn)
        button_layout.addWidget(self.cancel_analysis_btn)
        layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Select Start to begin Segmentation")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)

        return group

    def create_results_panel(self):
        """Create the right results panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Results title
        results_title = QLabel("Analysis Results")
        results_font = QFont()
        results_font.setBold(True)
        results_font.setPointSize(14)
        results_title.setFont(results_font)
        results_title.setStyleSheet("color: #2196F3; margin-bottom: 10px;")
        layout.addWidget(results_title)

        # Visualization controls
        viz_controls = self.create_visualization_controls()
        layout.addWidget(viz_controls)

        # Matplotlib figure
        self.population_figure = plt.figure(constrained_layout=True)
        self.population_canvas = FigureCanvas(self.population_figure)
        layout.addWidget(self.population_canvas)

        # Export section
        export_section = self.create_export_section()
        layout.addWidget(export_section)

        # Stretch
        layout.addStretch()

        return widget

    def create_visualization_controls(self):
        """Create visualization controls"""
        group = QGroupBox("Visualization")
        layout = QHBoxLayout(group)

        # Frame capture interval selection
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Capture Interval:"))
        self.frame_interval_value = QDoubleSpinBox()
        self.frame_interval_value.setDecimals(3)
        self.frame_interval_value.setRange(0.001, 1e9)
        self.frame_interval_value.setValue(24)  # default = 1 hr
        self.frame_interval_value.setSingleStep(0.1)
        interval_layout.addWidget(self.frame_interval_value)
        # Unit dropdown
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(["ms", "sec", "min", "hr", "day"])
        self.time_unit_combo.setCurrentText("hr")
        interval_layout.addWidget(self.time_unit_combo)
        layout.addLayout(interval_layout)

        # Graph Type selection
        metric_layout = QHBoxLayout()
        metric_layout.addWidget(QLabel("Analysis Type:"))
        self.metric_combo = QComboBox()
        self.refresh_metric_choices()
        metric_layout.addWidget(self.metric_combo)
        layout.addLayout(metric_layout)

        # Regenerate Graph
        self.generate_graph_btn = QPushButton("Generate Graph")
        self.generate_graph_btn.setEnabled(False)
        self.generate_graph_btn.clicked.connect(self.on_plot_avg_sd)
        layout.addWidget(self.generate_graph_btn)

        layout.addStretch()

        return group

    def create_export_section(self):
        """Create export section"""
        group = QGroupBox("Export Results")
        layout = QHBoxLayout(group)

        self.export_csv_btn = QPushButton("Export to CSV")
        self.export_csv_btn.setEnabled(False)
        self.export_csv_btn.clicked.connect(self.export_to_csv)
        self.export_plots_btn = QPushButton("Export Plots")
        self.export_plots_btn.setEnabled(False)
        self.export_plots_btn.clicked.connect(self.export_plots)

        layout.addWidget(self.export_csv_btn)
        layout.addWidget(self.export_plots_btn)
        layout.addStretch()

        return group

    def validate_partaker_segmentation_available(
            self,
            selected_positions=None,
            t_start=None,
            t_end=None,
            focus_loss_skip=None,
    ):
        """
        Check that segmented cell masks are available whenever the selected
        analysis mode uses Partaker cell masks.
        """
        try:
            from nd2_analyzer.data.image_data import ImageData

            mode_name = self.get_analysis_method()

            image_data = ImageData.get_instance()

            if image_data is None or image_data.segmentation_cache is None:
                QMessageBox.warning(
                    self,
                    "Missing cell segmentation",
                    f"{mode_name} with Phase Contrast requires segmented cell masks. "
                    "Run cell segmentation first, choose Fluorescence cell view, "
                    "or switch analysis mode."
                )
                return False

            segmented_storage = image_data.segmentation_cache
            model_name = image_data.segmentation_cache.model_name

            if not model_name:
                QMessageBox.warning(
                    self,
                    "Missing cell segmentation model",
                    f"{mode_name} with Phase Contrast requires a selected cell segmentation model. "
                    "Run cell segmentation first, choose Fluorescence cell view, "
                    "or switch analysis mode."
                )
                return False

            cache = segmented_storage.with_model(model_name)

            if (
                    not hasattr(cache, "mmap_arrays_idx")
                    or model_name not in cache.mmap_arrays_idx
            ):
                QMessageBox.warning(
                    self,
                    "Missing cell segmentation cache",
                    f"{mode_name} with Phase Contrast requires segmented cell masks. "
                    "Run cell segmentation first, choose Fluorescence cell view, "
                    "or switch analysis mode."
                )
                return False

            _mmap_array, index_set = cache.mmap_arrays_idx[model_name]

            if not index_set:
                QMessageBox.warning(
                    self,
                    "Missing segmented cell masks",
                    f"{mode_name} with Phase Contrast requires segmented cell masks. "
                    "Run cell segmentation first, choose Fluorescence cell view, "
                    "or switch analysis mode."
                )
                return False

            selected_cell_channel = self.get_selected_cell_channel()

            available_channels = set()
            available_frame_keys = set()

            for idx in index_set:
                if len(idx) == 3:
                    t, p, c = idx
                    t = int(t)
                    p = int(p)
                    c = int(c)

                    available_channels.add(c)
                    available_frame_keys.add((t, p, c))

            if not available_channels:
                QMessageBox.warning(
                    self,
                    "Cannot verify segmented channel",
                    "The existing segmented cell masks do not appear to store channel information.\n\n"
                    f"{mode_name} needs channel-aware segmentation masks so the selected Cell Channel can be checked.\n"
                    "Please re-run cell segmentation with the current channel-aware cache."
                )
                return False

            if selected_cell_channel not in available_channels:
                QMessageBox.warning(
                    self,
                    "Cell channel not segmented",
                    f"Selected Cell Channel {selected_cell_channel} was not segmented.\n\n"
                    f"Available segmented channels: {sorted(available_channels)}\n\n"
                    "Select the channel that was actually segmented, or run cell segmentation on the selected Cell Channel first."
                )
                return False

            if selected_positions is not None and t_start is not None and t_end is not None:
                focus_loss_skip = focus_loss_skip or set()

                missing = []

                for p in selected_positions:
                    for t in range(t_start, t_end + 1):
                        if t in focus_loss_skip:
                            continue

                        key = (
                            int(t),
                            int(p),
                            int(selected_cell_channel),
                        )

                        if key not in available_frame_keys:
                            missing.append(key)

                if missing:
                    first_t, first_p, first_c = missing[0]

                    QMessageBox.warning(
                        self,
                        "Missing segmented cell masks",
                        f"{mode_name} cannot start because the selected Cell Channel is missing segmented masks.\n\n"
                        f"First missing frame: T={first_t}, P={first_p}, C={first_c}\n"
                        f"Total missing frames: {len(missing)}\n\n"
                        "Run cell segmentation for the selected Cell Channel over the same positions/time range, "
                        "or change the Cell Channel selection."
                    )
                    return False

            return True

        except Exception as e:
            QMessageBox.warning(
                self,
                "Segmentation check failed",
                f"Could not verify segmented cell masks for {mode_name}:\n{e}"
            )
            return False

    def cancel_analysis(self):
        """Cancel Live segmentation."""
        if getattr(self, "is_segmenting", False):
            self.cancel_requested = True
            pub.sendMessage("segmentation_cancelled")
            self.status_label.setText("Cancelling segmentation...")
            return

    def export_to_csv(self):
        """Rewrite the current paired metrics CSV in the analysis output folder."""
        if not hasattr(self, "live_dead_df") or self.live_dead_df.is_empty():
            QMessageBox.warning(self, "No data", f"Run {self.analysis_label()} analysis first.")
            return
        output_path = (
            self.get_live_dead_output_root()
            / f"{self.output_prefix()}_processing_metrics.csv"
        )
        self.live_dead_df.write_csv(output_path)
        self.status_label.setText(f"Metrics saved to {output_path}")

    def export_plots(self):
        """Regenerate all Live-Dead plots in the analysis output folder."""
        if not hasattr(self, "live_dead_df") or self.live_dead_df.is_empty():
            QMessageBox.warning(self, "No data", f"Run {self.analysis_label()} analysis first.")
            return
        output_dir = self.get_live_dead_output_root()
        paths = self.save_all_summary_plots(output_dir)
        self.status_label.setText(
            f"Saved {len(paths)} plots to {output_dir}"
        )

    @staticmethod
    def cell_exclusion_distance(cell_labels):
        """Full-resolution Euclidean distance outside cells; labels are untouched."""
        labels = np.asarray(cell_labels)
        if labels.ndim != 2:
            raise ValueError("Background exclusion requires two-dimensional cell labels.")
        if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
            raise ValueError("Background exclusion requires nonnegative integer cell labels.")
        cells = labels > 0
        if not cells.any():
            return np.full(labels.shape, np.inf, dtype=np.float32)
        return distance_transform_edt(~cells).astype(np.float32)

    @classmethod
    def background_exclusion_distance(cls, cell_labels, fluorescence_frames,
                                      thresholds=(), include_cells=True):
        """Combine cell footprints and raw-channel signal masks before adding a margin."""
        labels = np.asarray(cell_labels)
        if labels.ndim != 2:
            raise ValueError("Background exclusion requires two-dimensional labels.")
        excluded = labels > 0 if include_cells else np.zeros(labels.shape, dtype=bool)
        for channel, threshold in dict(thresholds).items():
            if not np.isfinite(threshold):
                raise ValueError("Fluorescence exclusion thresholds must be finite.")
            frame = np.asarray(fluorescence_frames[channel])
            if frame.shape != labels.shape:
                raise ValueError("Fluorescence exclusion masks and cell labels must align.")
            excluded |= np.isfinite(frame) & (frame >= threshold)
        return cls.cell_exclusion_distance(excluded.astype(np.uint8))

    @staticmethod
    def estimate_cell_free_background(image, distance, margin_pixels=15, upper_cutoff=None):
        """Estimate a constant background without camera metadata or intensity clipping."""
        image = np.asarray(image, dtype=np.float32)
        distance = np.asarray(distance)
        if image.ndim != 2 or image.shape != distance.shape:
            raise ValueError("Fluorescence and cell-exclusion frames must have matching shapes.")
        if margin_pixels < 0:
            raise ValueError("The exclusion margin cannot be negative.")
        eligible = (distance > margin_pixels) & np.isfinite(image)
        if upper_cutoff is not None:
            if not np.isfinite(upper_cutoff):
                raise ValueError("The manual upper-intensity cutoff must be finite.")
            eligible &= image < upper_cutoff
        values = image[eligible]
        if not values.size:
            raise ValueError("No eligible background pixels remain. Reduce the margin or cutoff.")
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        regional = []
        height, width = image.shape
        for ys in (slice(0, height // 2), slice(height // 2, height)):
            for xs in (slice(0, width // 2), slice(width // 2, width)):
                region = image[ys, xs][eligible[ys, xs]]
                regional.append(float(np.median(region)) if region.size else None)
        finite_regions = [value for value in regional if value is not None]
        fraction = float(values.size / image.size)
        notes = []
        if np.isinf(distance).all():
            notes.append("no_cells")
        if fraction < 0.01 or values.size < min(1000, image.size):
            notes.append("limited_background_coverage")
        spread = max(finite_regions) - min(finite_regions) if finite_regions else 0.0
        if mad > 0 and spread > 3 * 1.4826 * mad:
            notes.append("spatial_variation_review")
        return {
            "median": median, "eligible_pixels": int(values.size),
            "eligible_fraction": fraction, "mad": mad,
            "regional_medians": regional, "regional_median_range": float(spread),
            "status": ";".join(notes) if notes else "ok",
            "margin_pixels": int(margin_pixels), "upper_cutoff": upper_cutoff,
        }

    @staticmethod
    def estimate_manual_background(image, rectangles):
        """Measure exactly three non-overlapping half-open rectangles on raw pixels."""
        raw = np.asarray(image, dtype=np.float32)
        if raw.ndim != 2 or len(rectangles) != 3:
            raise ValueError("Select exactly three background rectangles for this frame/channel.")
        samples, values, boxes = [], [], []
        for index, rectangle in enumerate(rectangles, 1):
            if len(rectangle) != 4 or any(int(v) != v for v in rectangle):
                raise ValueError("Sample coordinates and dimensions must be integer pixels.")
            x, y, width, height = map(int, rectangle)
            if min(x, y) < 0 or min(width, height) <= 0 or x + width > raw.shape[1] or y + height > raw.shape[0]:
                raise ValueError(f"Sample {index} is outside the image; move or resize it.")
            if any(x < bx + bw and bx < x + width and y < by + bh and by < y + height
                   for bx, by, bw, bh in boxes):
                raise ValueError("Background samples must not overlap.")
            boxes.append((x, y, width, height))
            region = raw[y:y + height, x:x + width]
            finite = region[np.isfinite(region)]
            if not finite.size:
                raise ValueError(f"Sample {index} has no finite intensity pixels.")
            values.append(finite)
            samples.append({"sample_id": index, "x": x, "y": y, "width": width, "height": height,
                            "x_max_exclusive": x + width, "y_max_exclusive": y + height,
                            "area_pixels": width * height, "valid_pixels": int(finite.size),
                            "mean": float(np.mean(finite, dtype=np.float64)),
                            "median": float(np.median(finite)), "sd": float(np.std(finite, dtype=np.float64))})
        pooled = np.concatenate(values)
        mean = float(np.mean(pooled, dtype=np.float64))
        median = float(np.median(pooled))
        means = [r["mean"] for r in samples]
        spread = max(means) - min(means)
        relative = spread / max(abs(mean), 1e-12)
        flags = []
        if relative > .20:
            flags.append("sample_means_disagree")
        if any(r["valid_pixels"] != r["area_pixels"] for r in samples):
            flags.append("partial_nonfinite_samples")
        return {"method": "manual_samples_mean", "value": mean, "mean": mean, "median": median,
                "sd": float(np.std(pooled, dtype=np.float64)), "samples": samples,
                "eligible_pixels": int(pooled.size), "eligible_fraction": float(pooled.size / raw.size),
                "mad": float(np.median(np.abs(pooled - median))), "regional_medians": [r["median"] for r in samples],
                "regional_median_range": spread, "sample_mean_range": spread,
                "sample_mean_relative_range": relative, "disagreement_threshold_relative_range": .20,
                "status": ";".join(flags) if flags else "ok", "margin_pixels": 0, "upper_cutoff": None,
                "averaging_method": "pooled finite raw pixels; weighted by valid pixel count",
                "intensity_space": "raw before background subtraction and Gaussian smoothing"}

    def manual_background_rectangles(self, time, position, channel):
        return [tuple(row[3:]) for row in getattr(self.partaker_live_dead_setup, "background_samples", ())
                if tuple(row[:3]) == (int(time), int(position), int(channel))]

    def save_background_sample_overlay(self, name, data, time, position):
        """Save raw fluorescence plus sheer blue sample fills, without rectangle outlines."""
        raw = np.asarray(data["raw"], dtype=np.float32)
        finite = raw[np.isfinite(raw)]
        low, high = np.percentile(finite, [1, 99.8]) if finite.size else (0., 1.)
        grey = np.clip(np.nan_to_num((raw - low) / max(float(high - low), 1.), nan=0., posinf=1., neginf=0.), 0., 1.)
        rgb = np.repeat(grey[..., None], 3, axis=2) * 255.
        for sample in data["background"]["samples"]:
            x, y, w, h = (sample[k] for k in ("x", "y", "width", "height"))
            rgb[y:y+h, x:x+w] = .70 * rgb[y:y+h, x:x+w] + .30 * np.asarray([0., 160., 255.])
        folder = self.get_live_dead_output_root() / "background_samples"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"pos{position}_t{time}_{name}_background_samples.png"
        Image.fromarray(np.rint(rgb).astype(np.uint8)).save(path)
        corrected = np.asarray(data["corrected"], dtype=np.float32)
        corrected_grey = np.clip(np.nan_to_num((corrected - low) / max(float(high - low), 1.), nan=0., posinf=1., neginf=0.), 0., 1.)
        corrected_rgb = np.repeat(corrected_grey[..., None], 3, axis=2) * 255.
        for sample in data["background"]["samples"]:
            x, y, w, h = (sample[k] for k in ("x", "y", "width", "height"))
            corrected_rgb[y:y+h, x:x+w] = .70 * corrected_rgb[y:y+h, x:x+w] + .30 * np.asarray([0., 160., 255.])
        corrected_path = folder / f"pos{position}_t{time}_{name}_background_samples_corrected.png"
        Image.fromarray(np.rint(corrected_rgb).astype(np.uint8)).save(corrected_path)
        return {"corrected_overlay_path": str(corrected_path.relative_to(self.get_live_dead_output_root())),
                "overlay_path": str(path.relative_to(self.get_live_dead_output_root())),
                "overlay_alpha": .30, "overlay_rgb": [0, 160, 255],
                "display_percentiles": [1., 99.8], "display_low": float(low), "display_high": float(high)}

    def write_background_samples_csv(self):
        context = self.individual_export_context
        if context is None:
            return
        rows = []
        for frame in context["processing"]["background_frames"]:
            if frame.get("method") != "manual_samples_mean":
                continue
            channel = frame["channel"]
            role = next((k for k, v in context["channel_assignments"].items() if k != "cell" and v == channel), str(channel))
            source_values = context.get("tiff_source_axis_values") or {}
            positions = source_values.get("positions", [])
            channels = source_values.get("channels", [])
            times = source_values.get("times", [])
            p = positions[frame["position"]] if positions else frame["position"]
            c = channels[channel] if channels else channel
            t = times[frame["time"]] if times and frame["time"] < len(times) else frame["time"]
            mapping = context.get("tiff_file_map") or {}
            source = mapping.get(f"{p},{t},{c}", mapping.get(f"{p},0,{c}", ""))
            if not source and context.get("sources"):
                source = context["sources"][0]["path"] if len(context["sources"]) == 1 else ""
            for sample in frame["samples"]:
                rows.append({"run_id": context["run_id"], "dataset_id": context["dataset_id"],
                             "source_file": source, "source_position_id": p, "source_channel_id": c, "source_time_index": t, "source_stack_page_index": frame["time"] if context.get("tiff_import_mode") == "stacked_tiff" else None,
                             "position": frame["position"], "time": frame["time"], "channel": channel,
                             "channel_role": role, "coordinate_space": "cropped_registered_analysis_frame",
                             **sample, "subtracted_background_mean": frame["mean"],
                             "pooled_sd": frame["sd"], "sample_mean_range": frame["sample_mean_range"],
                             "sample_mean_relative_range": frame["sample_mean_relative_range"],
                             "disagreement_threshold_relative_range": .20, "status": frame["status"],
                             "averaging_method": frame["averaging_method"], "intensity_space": frame["intensity_space"],
                             "gaussian_sigma": context["processing"]["smoothing_sigma"],
                             "crop_xywh_json": json.dumps(context.get("transforms", {}).get("crop_xywh")),
                             "registration_offsets_json": json.dumps(context.get("transforms", {}).get("registration_offsets_xy_by_time")),
                             "pixel_calibration_json": json.dumps(context.get("pixel_calibration", {})),
                             "provenance_file": "analysis_provenance.json", "overlay_path": frame.get("overlay_path", ""),
                             "corrected_overlay_path": frame.get("corrected_overlay_path", "")})
        if rows:
            path = self.get_live_dead_output_root() / "background_samples.csv"
            temporary = path.with_suffix(".csv.tmp")
            with temporary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temporary, path)

    def cell_background_enabled(self):
        return bool(getattr(self.partaker_live_dead_setup, "background_enabled", False))

    def make_background_label_provider(self):
        """Snapshot cell settings so dialog workers never access widget controls."""
        from nd2_analyzer.data.image_data import ImageData
        image_data = ImageData.get_instance()
        channel = self.get_selected_cell_channel()
        if self.cell_view_combo.currentText() == "Phase Contrast":
            storage = getattr(image_data, "segmentation_cache", None)
            model = getattr(storage, "model_name", None)
            cache = storage.with_model(model) if storage is not None and model is not None else None
            def labels(time, position):
                if cache is None:
                    raise ValueError("Segment cells before enabling cell-exclusion background correction.")
                array, indices = cache.mmap_arrays_idx[model]
                if (int(time), int(position), int(channel)) not in indices:
                    raise ValueError(f"Saved cell labels unavailable at T={time}, P={position}, C={channel}.")
                return np.asarray(array[time, position, channel])
        else:
            threshold = float(self.CELL_THRESHOLD)
            kernel = disk(self.get_cell_morphology_radius_px())
            def labels(time, position):
                frame = np.asarray(image_data.get(time, position, channel), dtype=float)
                mask = np.isfinite(frame) & (frame >= threshold)
                mask = binary_closing(mask, footprint=kernel)
                mask = binary_dilation(mask, footprint=kernel)
                return label_connected_components(mask)
        return labels

    def background_corrected_plot_data(self, df):
        """Keep raw CSV columns intact while corrected series drive plots."""
        if not self.cell_background_enabled() or df is None:
            return df
        updates = []
        for column in df.columns:
            if "_corrected_" in column:
                target = column.replace("_corrected_", "_", 1)
                if target in df.columns:
                    updates.append(pl.col(column).alias(target))
        return df.with_columns(updates) if updates else df

    def reset_live_dead_run_preprocessing(self):
        """Clear cached intensity-window and background-reference state."""
        self.live_dead_processing_black_point_raw = None
        self.live_dead_processing_white_point_raw = None
        self.live_dead_processing_window_source = "uninitialized"
        self.live_dead_reference_backgrounds = {}

    def build_live_dead_reference_backgrounds(
            self,
            frame_keys,
            sigma: float | None = None,
    ):
        """
        Build one Gaussian background reference per position and channel.

        The background is estimated from time 0 and the same reference is
        subtracted from every frame at that position. This is intentionally
        different from correcting only T=0 or estimating a different broad
        background independently for every timepoint.
        """
        self.live_dead_reference_backgrounds = {}

        if self.cell_background_enabled() or not self.gaus_back_corr.isChecked():
            return

        from nd2_analyzer.data.image_data import ImageData

        image_data = ImageData.get_instance()
        sigma = float(self.LIVE_BACKGROUND_SIGMA if sigma is None else sigma)

        position_channels = sorted({
            (int(position), int(channel))
            for _time, position, channel in frame_keys
        })

        for position, channel in position_channels:
            reference_frame = np.asarray(
                image_data.get(0, position, channel),
                dtype=np.float32,
            )

            reference_background = np.empty_like(
                reference_frame,
                dtype=np.float32,
            )
            gaussian_filter(
                reference_frame,
                sigma=sigma,
                output=reference_background,
            )

            self.live_dead_reference_backgrounds[(position, channel)] = reference_background

            print(
                f"Cached Gaussian {self.analysis_label()} background | "
                f"P={position} T=0 C={channel} sigma={sigma:g}"
            )

    def apply_live_dead_reference_background(
            self,
            image: np.ndarray,
            position: int,
            channel: int,
    ) -> np.ndarray:
        """Subtract the cached T=0 Gaussian background for one position."""
        image = np.asarray(image, dtype=np.float32)

        if not self.gaus_back_corr.isChecked():
            return image

        reference_background = self.live_dead_reference_backgrounds.get(
            (int(position), int(channel))
        )

        if reference_background is None:
            raise RuntimeError(
                "Gaussian background subtraction was selected, but no "
                "reference background was cached for "
                f"Position {position}, Channel {channel}."
            )

        if reference_background.shape != image.shape:
            raise ValueError(
                "Live background shape does not match the current frame: "
                f"{reference_background.shape} != {image.shape}."
            )

        corrected = np.array(image, dtype=np.float32, copy=True)
        corrected -= reference_background
        np.maximum(corrected, 0.0, out=corrected)
        return corrected

    def estimate_live_dead_processing_window(self, frame_keys):
        """
        Estimate one raw-intensity black/white window for the complete run.

        Each representative frame contributes a background-side percentile and
        a bright-signal percentile. The median black candidate and a high
        percentile of the white candidates are reused for every frame, so real
        changes over time are not normalized away.
        """
        if not frame_keys:
            raise ValueError(f"No {self.analysis_label()} frames were supplied for window estimation.")

        from nd2_analyzer.data.image_data import ImageData

        image_data = ImageData.get_instance()
        sample_count = min(
            int(self.LIVE_WINDOW_MAX_SAMPLE_FRAMES),
            len(frame_keys),
        )
        sample_indices = np.unique(
            np.linspace(
                0,
                len(frame_keys) - 1,
                sample_count,
                dtype=int,
            )
        )

        black_candidates = []
        white_candidates = []

        for sample_index in sample_indices:
            time, position, channel = frame_keys[int(sample_index)]
            frame = np.asarray(
                image_data.get(time, position, channel),
                dtype=np.float32,
            )
            processing_frame = self.apply_live_dead_reference_background(
                frame,
                position,
                channel,
            )

            finite_values = processing_frame[np.isfinite(processing_frame)]
            if finite_values.size == 0:
                continue

            black_candidates.append(float(np.percentile(
                finite_values,
                self.LIVE_WINDOW_BLACK_PERCENTILE,
            )))
            white_candidates.append(float(np.percentile(
                finite_values,
                self.LIVE_WINDOW_WHITE_PERCENTILE,
            )))

        if not black_candidates or not white_candidates:
            raise ValueError(
                f"Could not estimate a finite {self.analysis_label()} processing intensity window."
            )

        black_point = float(np.median(black_candidates))
        white_point = float(np.percentile(
            white_candidates,
            self.LIVE_WINDOW_WHITE_ACROSS_FRAMES_PERCENTILE,
        ))

        if not np.isfinite(black_point) or not np.isfinite(white_point):
            raise ValueError(f"Estimated {self.analysis_label()} intensity window is not finite.")

        if white_point <= black_point:
            white_point = black_point + 1.0

        self.live_dead_processing_black_point_raw = black_point
        self.live_dead_processing_white_point_raw = white_point
        self.live_dead_processing_window_source = "automatic_run_level"

        print(
            f"{self.analysis_label()} run-level processing window | "
            f"black_raw={black_point:.3f} | "
            f"white_raw={white_point:.3f} | "
            f"sampled_frames={len(sample_indices)} | "
            f"black_pct={self.LIVE_WINDOW_BLACK_PERCENTILE:g} | "
            f"white_pct={self.LIVE_WINDOW_WHITE_PERCENTILE:g}"
        )

    def prepare_live_dead_run_preprocessing(self, frame_keys):
        """Prepare shared background references and intensity scaling."""
        self.reset_live_dead_run_preprocessing()

        self.build_live_dead_reference_backgrounds(frame_keys)

        # Reuse the dialog's accepted scope mapping exactly. Manual sampling
        # holds the original raw mapping fixed during subtraction; legacy Gaussian
        # correction still estimates its own processing window.
        if (
                self.partaker_live_dead_setup is not None
                and not self.gaus_back_corr.isChecked()
        ):
            self.live_dead_processing_black_point_raw = float(
                self.partaker_live_dead_setup.processing_black_point_raw
            )
            self.live_dead_processing_white_point_raw = float(
                self.partaker_live_dead_setup.processing_white_point_raw
            )
            self.live_dead_processing_window_source = str(
                self.partaker_live_dead_setup.processing_window_source
            )
            print(
                f"Using Partaker dialog {self.analysis_label()} processing window | "
                f"black_raw={self.live_dead_processing_black_point_raw:.3f} | "
                f"white_raw={self.live_dead_processing_white_point_raw:.3f}"
            )
            return

        self.estimate_live_dead_processing_window(frame_keys)

    def open_partaker_live_dead_setup_dialog(
            self,
            *,
            selected_positions,
            t_start: int,
            t_end: int,
            live_channel: int,
            dead_channel: int | None,
    ):
        """Open Partaker fixed-threshold calibration and return its result."""
        from nd2_analyzer.data.image_data import ImageData
        from nd2_analyzer.ui.dialogs.live_dead_dialog import LiveDeadDialog

        image_data = ImageData.get_instance()
        if image_data is None or image_data.data is None:
            QMessageBox.warning(
                self,
                "No image data",
                f"Load image data before configuring fixed {self.analysis_label()} thresholding."
            )
            return None

        from nd2_analyzer.data.appstate import ApplicationState
        experiment = getattr(ApplicationState.get_instance(), "experiment", None)
        skipped_times = tuple(t for t in range(t_start, t_end + 1)
                              if experiment is not None and experiment.is_focus_loss_frame(t))
        dialog = LiveDeadDialog(
            image_data=image_data,
            excluded_timepoints=skipped_times,
            analysis_mode=self.run_analysis_mode,
            live_channel=live_channel,
            dead_channel=dead_channel,
            selected_positions=selected_positions,
            time_start=t_start,
            time_end=t_end,
            cell_label_provider=self.make_background_label_provider(),
            distance_provider=self.background_exclusion_distance,
            background_estimator=self.estimate_cell_free_background,
            sample_estimator=self.estimate_manual_background,
            initial_drop_frame_zero=self.partaker_drop_frame_zero,
            initial_threshold=self.partaker_live_dead_threshold,
            smoothing_sigma=self.partaker_live_dead_smoothing_sigma,
            initial_capture_interval_value=self.frame_interval_value.value(),
            initial_capture_interval_unit=self.time_unit_combo.currentText(),
            parent=self,
        )

        if dialog.exec() != QDialog.Accepted or dialog.result is None:
            return None

        return dialog.result

    def apply_partaker_live_dead_setup(self, setup) -> None:
        """Copy accepted setup values into the Live-Dead widget controls."""
        self.partaker_live_dead_setup = setup
        if getattr(setup, "background_enabled", False) or getattr(setup, "background_method", "") == "manual_samples_mean":
            # Skip correction in the manual-sampling dialog also disables legacy subtraction.
            self.gaus_back_corr.setChecked(False)
        self.partaker_live_dead_threshold = int(setup.threshold_uint8)
        self.partaker_live_dead_smoothing_sigma = float(setup.smoothing_sigma)
        self.partaker_drop_frame_zero = bool(setup.drop_frame_zero)
        self.drop_frame_zero_for_current_run = bool(setup.drop_frame_zero)
        self.time_start_spin.setValue(int(setup.time_start))
        self.time_end_spin.setValue(int(setup.time_end))
        if self.is_cell_viability_run():
            self.cell_viability_channel_combo.setCurrentIndex(int(setup.live_channel))
        else:
            self.live_channel_combo.setCurrentIndex(int(setup.live_channel))
            self.dead_channel_combo.setCurrentIndex(int(setup.dead_channel))
        self.frame_interval_value.setValue(
            float(getattr(setup, "capture_interval_value", self.frame_interval_value.value()))
        )
        capture_interval_unit = str(
            getattr(setup, "capture_interval_unit", self.time_unit_combo.currentText())
        )
        capture_interval_index = self.time_unit_combo.findText(
            capture_interval_unit
        )
        if capture_interval_index >= 0:
            self.time_unit_combo.setCurrentIndex(capture_interval_index)

        selected_positions = set(int(value) for value in setup.positions)
        for index in range(self.position_list.count()):
            self.position_list.item(index).setSelected(
                index in selected_positions
            )

        self.status_label.setText(
            f"Partaker {self.analysis_label()} setup accepted: "
            f"threshold={self.partaker_live_dead_threshold}, "
            + (f"cell_viability=C{setup.live_channel}, " if self.is_cell_viability_run()
               else f"live=C{setup.live_channel}, dead=C{setup.dead_channel}, ")
            +
            f"interval={self.frame_interval_value.value():g} {self.time_unit_combo.currentText()}, "
            f"T={setup.time_start}..{setup.time_end}, "
            f"drop_T0={bool(setup.drop_frame_zero)}, "
            f"positions={list(setup.positions)}"
        )

    def live_dead_processed_value_to_raw(self, value):
        """Convert a scalar shared 0-255 threshold value back to raw units."""
        if not isinstance(value, (int, float, np.integer, np.floating)):
            return np.nan

        value = float(value)
        black_point = self.live_dead_processing_black_point_raw
        white_point = self.live_dead_processing_white_point_raw

        if (
                not np.isfinite(value)
                or black_point is None
                or white_point is None
                or white_point <= black_point
        ):
            return np.nan

        return float(
            black_point
            + (value / 255.0) * (white_point - black_point)
        )

    def segment_selected_channels(self):
        """Queue segmentation for selected channels, positions, and time range."""
        if self.is_segmenting:
            return
        self.individual_export_context = None
        self.individual_export_error = None
        self.run_analysis_mode = self.selected_analysis_mode()
        self.live_dead_results = []
        self.live_dead_df = pl.DataFrame()
        self.partaker_live_dead_setup = None
        self.refresh_metric_choices()
        self.export_csv_btn.setEnabled(False)
        self.export_plots_btn.setEnabled(False)
        self.reset_output_tracking()

        self.start_segmentation_btn.setEnabled(False)
        self.cancel_analysis_btn.setEnabled(True)
        selected_positions = self.get_selected_positions()

        if not selected_positions:
            self.status_label.setText("No positions selected")
            self.start_segmentation_btn.setEnabled(True)
            self.cancel_analysis_btn.setEnabled(False)
            return

        t_start = self.time_start_spin.value()
        t_end = self.time_end_spin.value()

        if t_end < t_start:
            self.status_label.setText("Invalid time range")
            self.start_segmentation_btn.setEnabled(True)
            self.cancel_analysis_btn.setEnabled(False)
            return

        live_channel = (self.cell_viability_channel_combo.currentIndex()
                        if self.is_cell_viability_run()
                        else self.live_channel_combo.currentIndex())
        dead_channel = (None if self.is_cell_viability_run()
                        else self.dead_channel_combo.currentIndex())
        if live_channel < 0 or (dead_channel is not None and dead_channel < 0):
            self.status_label.setText("Select an available fluorescence channel")
            self.start_segmentation_btn.setEnabled(True)
            self.cancel_analysis_btn.setEnabled(False)
            return

        if dead_channel is not None and live_channel == dead_channel:
            QMessageBox.warning(
                self,
                "Select two fluorescence channels",
                "Live and Dead must use different channels."
            )
            self.start_segmentation_btn.setEnabled(True)
            self.cancel_analysis_btn.setEnabled(False)
            return

        # The setup dialog may refine the main-widget frame-zero choice.
        drop_frame_zero = bool(
            self.drop_frame_zero_checkbox.isChecked()
        )

        if self.is_partaker_method():
            setup = self.open_partaker_live_dead_setup_dialog(
                selected_positions=selected_positions,
                t_start=t_start,
                t_end=t_end,
                live_channel=live_channel,
                dead_channel=dead_channel,
            )
            if setup is None:
                self.status_label.setText(f"Partaker {self.analysis_label()} setup cancelled")
                self.start_segmentation_btn.setEnabled(True)
                self.cancel_analysis_btn.setEnabled(False)
                return

            self.apply_partaker_live_dead_setup(setup)
            selected_positions = list(setup.positions)
            t_start = int(setup.time_start)
            t_end = int(setup.time_end)
            live_channel = int(setup.live_channel)
            dead_channel = None if setup.dead_channel is None else int(setup.dead_channel)
            drop_frame_zero = bool(setup.drop_frame_zero)

        self.drop_frame_zero_for_current_run = bool(drop_frame_zero)

        # Skip focus-loss frames and any explicitly excluded frame zero.
        focus_loss_skip = set()

        try:
            from nd2_analyzer.data.appstate import ApplicationState
            appstate = ApplicationState.get_instance()

            if (
                    appstate
                    and appstate.experiment
                    and appstate.experiment.focus_loss_intervals
            ):
                for t in range(t_start, t_end + 1):
                    if appstate.experiment.is_focus_loss_frame(t):
                        focus_loss_skip.add(t)

        except Exception:
            pass

        frames_to_skip = set(focus_loss_skip)
        if drop_frame_zero and t_start <= 0 <= t_end:
            frames_to_skip.add(0)

        # Phase-contrast cells continue to use the saved Partaker masks.
        # Fluorescence cell views retain their direct channel thresholding path.
        requires_partaker_cell_masks = (
            self.cell_view_combo.currentText() == "Phase Contrast"
        )
        if requires_partaker_cell_masks and not self.validate_partaker_segmentation_available(
                selected_positions=selected_positions,
                t_start=t_start,
                t_end=t_end,
                focus_loss_skip=frames_to_skip,
        ):
            self.start_segmentation_btn.setEnabled(True)
            self.cancel_analysis_btn.setEnabled(False)
            return

        frames_to_analyze = []
        preprocessing_frame_keys = []

        for p in selected_positions:
            for t in range(t_start, t_end + 1):

                if t in frames_to_skip:
                    continue

                frames_to_analyze.append(
                    (t, p, live_channel, dead_channel)
                )
                preprocessing_frame_keys.append((t, p, live_channel))
                if dead_channel is not None:
                    preprocessing_frame_keys.append((t, p, dead_channel))

        if not frames_to_analyze:
            self.status_label.setText(
                "No valid frames remain to analyze. Check the selected time "
                "range, focus-loss intervals, and frame-zero exclusion."
            )
            self.start_segmentation_btn.setEnabled(True)
            self.cancel_analysis_btn.setEnabled(False)
            return

        try:
            self.prepare_live_dead_run_preprocessing(preprocessing_frame_keys)
        except Exception as e:
            QMessageBox.warning(
                self,
                f"{self.analysis_label()} preprocessing setup failed",
                f"Could not prepare the shared {self.analysis_label()} intensity window:\n{e}"
            )
            self.start_segmentation_btn.setEnabled(True)
            self.cancel_analysis_btn.setEnabled(False)
            return

        try:
            self.initialize_individual_exports(
                frames=frames_to_analyze, selected_positions=selected_positions,
                time_start=t_start, time_end=t_end, skipped_frames=frames_to_skip,
            )
        except Exception as error:
            self.queue = []
            self.individual_export_error = str(error)
            self.finish_individual_exports(str(error))
            self.status_label.setText(f"Cell export setup failed: {error}")
            self.start_segmentation_btn.setEnabled(True)
            self.cancel_analysis_btn.setEnabled(False)
            return

        self.queue = frames_to_analyze
        print(
            f"Queue length: {len(frames_to_analyze)} | "
            f"drop_frame_zero={self.drop_frame_zero_for_current_run}"
        )

        self.progress_bar.setMaximum(len(self.queue))
        self.progress_bar.setValue(0)

        self.is_segmenting = True
        self.cell_viability_toggle.setEnabled(False)
        self.set_run_configuration_enabled(False)
        self.cancel_requested = False

        self.start_segmentation_btn.setEnabled(False)
        self.cancel_analysis_btn.setEnabled(True)

        frame_zero_note = (
            " Frame 0 is excluded."
            if self.drop_frame_zero_for_current_run
            else ""
        )
        self.status_label.setText(
            f"Processing {len(self.queue)} frames...{frame_zero_note}"
        )
        self.process_next_in_queue()

    def on_image_data_loaded(self, image_data):
        if image_data is None or image_data.data is None:
            return

        # A calibration result belongs to one loaded dataset only.
        self.live_dead_results = []
        self.live_dead_df = pl.DataFrame()
        self.run_analysis_mode = self.selected_analysis_mode()
        self.refresh_metric_choices()
        self.partaker_live_dead_setup = None
        self.partaker_drop_frame_zero = False
        self.drop_frame_zero_for_current_run = False
        if hasattr(self, "drop_frame_zero_checkbox"):
            self.drop_frame_zero_checkbox.setChecked(False)
        self.start_segmentation_btn.setEnabled(True)
        self.export_csv_btn.setEnabled(False)
        self.export_plots_btn.setEnabled(False)

        # Get dimensions from image_data
        shape = image_data.data.shape
        t_max = shape[0] - 1
        p_max = shape[1] - 1
        c_max = shape[2] - 1

        # Populate Live, Dead, and Cell channel combo boxes.
        self.live_channel_combo.clear()
        self.dead_channel_combo.clear()
        self.cell_channel_combo.clear()
        self.cell_viability_channel_combo.clear()
        for c in range(c_max + 1):
            self.live_channel_combo.addItem(f"Channel {c}")
            self.dead_channel_combo.addItem(f"Channel {c}")
            self.cell_channel_combo.addItem(f"Channel {c}")
            self.cell_viability_channel_combo.addItem(f"Channel {c}")
        if self.dead_channel_combo.count() > 1:
            self.dead_channel_combo.setCurrentIndex(1)

        # Populate position list
        self.position_list.clear()
        for p in range(p_max + 1):
            self.position_list.addItem(f"Position {p}")

        # Update time range spinboxes
        self.time_start_spin.setMaximum(t_max)
        self.time_end_spin.setMaximum(t_max)
        self.time_end_spin.setValue(t_max)

        # Activate Graph Generation Button
        self.generate_graph_btn.setEnabled(True)

    def process_next_in_queue(self):

        if not self.is_segmenting:
            return

        if not self.queue:
            self._segmentation_finished()
            return

        if self.cancel_requested:
            self._segmentation_finished()
            return

        time, position, live_channel, dead_channel = self.queue.pop(0)

        from nd2_analyzer.data.image_data import ImageData

        image_data = ImageData.get_instance()

        raw_live = raw_dead = None
        try:
            raw_live = image_data.get(time, position, live_channel)
            raw_dead = (image_data.get(time, position, dead_channel)
                        if dead_channel is not None else None)
            metrics = self.process_live_dead_frame(
                live_frame=raw_live,
                dead_frame=raw_dead,
                time=time,
                position=position,
                live_channel=live_channel,
                dead_channel=dead_channel,
            )

        except Exception as e:
            print(
                f"Failed frame "
                f"T={time} P={position}: {e}"
            )
            traceback.print_exc()
            if self.individual_export_context is not None:
                self.record_individual_frame_failure(time, position, e)
            if self.individual_export_error is not None:
                self.cancel_requested = True

            del raw_live, raw_dead
            gc.collect()
            self.request_timer.start(1)
            return

        cell_channel = self.get_selected_cell_channel()

        result_row = {
            "time": time,
            "time_hours": self.get_time_hours(time),
            "capture_interval_value": float(self.frame_interval_value.value()),
            "capture_interval_unit": str(self.time_unit_combo.currentText()),
            "position": position,

            **({"cell_viability_channel": live_channel} if self.is_cell_viability_run()
               else {"live_channel": live_channel, "dead_channel": dead_channel}),
            "analysis_mode": self.run_analysis_mode,
            "cell_channel": cell_channel,
            "analysis_method": self.get_analysis_method(),

            "cell_view_type": self.cell_view_combo.currentText(),

            "background_filtering_used": self.cell_background_enabled() or bool(self.gaus_back_corr.isChecked()),
            f"{self.output_prefix()}_morphology_used": bool(self.close_dialate.isChecked()),
            "frame_zero_dropped": bool(
                self.drop_frame_zero_for_current_run
            ),
        }
        result_row.update(metrics)

        self.live_dead_results.append(result_row)
        if self.individual_export_context is not None:
            self.individual_export_context["completed_frames"].append(
                {"time": int(time), "position": int(position)}
            )
            try:
                self.write_analysis_provenance()
            except Exception as error:
                self.individual_export_error = str(error)
                self.cancel_requested = True

        completed = len(self.live_dead_results)

        self.progress_bar.setValue(
            completed
        )

        self.status_label.setText(
            f"Processing T={time} P={position}"
        )

        # Release the large per-frame arrays before scheduling the next frame.
        del metrics
        del raw_live, raw_dead
        gc.collect()

        self.log_process_memory(
            f"After completed frame T={time} P={position}"
        )

        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        self.request_timer.start(1)

    def select_all_positions(self):
        """Select all positions in the list"""
        for i in range(self.position_list.count()):
            self.position_list.item(i).setSelected(True)

    def select_no_positions(self):
        """Deselect all positions in the list"""
        for i in range(self.position_list.count()):
            self.position_list.item(i).setSelected(False)

    def get_selected_positions(self):
        """Get list of selected positions from the position list widget"""
        selected_positions = []
        for i in range(self.position_list.count()):
            item = self.position_list.item(i)
            if item.isSelected():
                # Extract position number from item text (e.g., "Position 0" -> 0)
                position_text = item.text()
                if "Position" in position_text:
                    try:
                        position_num = int(position_text.split()[-1])
                        selected_positions.append(position_num)
                    except (ValueError, IndexError):
                        print(f"Warning: Could not parse position from '{position_text}'")

        return selected_positions

    def _segmentation_finished(self):
        """Write paired metrics and all Live-Dead visual outputs."""
        final_output_error = None
        gif_path = None
        overlay_gif_paths = []
        summary_plot_paths = []
        try:
            self.live_dead_df = pl.DataFrame(self.live_dead_results)
            output_dir = self.get_live_dead_output_root()

            if self.live_dead_results:
                csv_rows = []
                for row in self.live_dead_results:
                    csv_rows.append({
                        key: value
                        for key, value in row.items()
                        if not isinstance(value, np.ndarray)
                    })
                pl.DataFrame(csv_rows).write_csv(
                    output_dir / f"{self.output_prefix()}_processing_metrics.csv"
                )

            self.save_processing_heatmaps(output_dir)
            if not self.is_cell_viability_run():
                self.save_live_dead_overlap_tables(output_dir)
            gif_path = self.export_live_dead_gif(output_dir)
            overlay_gif_paths = self.export_overlay_folder_gifs(output_dir)
            summary_plot_paths = self.save_all_summary_plots(output_dir)

        except Exception as error:
            final_output_error = str(error)
            traceback.print_exc()
        finally:
            output_dir = self.get_live_dead_output_root()
            self.finish_individual_exports(final_output_error)

        self.live_dead_reference_backgrounds = {}
        shutil.rmtree(output_dir / "_gif_mask_cache", ignore_errors=True)
        gc.collect()

        self.is_segmenting = False
        self.cell_viability_toggle.setEnabled(True)
        self.set_run_configuration_enabled(True)
        self.start_segmentation_btn.setEnabled(True)
        self.cancel_analysis_btn.setEnabled(False)
        has_results = not self.live_dead_df.is_empty()
        self.export_csv_btn.setEnabled(has_results)
        self.export_plots_btn.setEnabled(has_results)

        frame_zero_note = (
            " Frame 0 was excluded."
            if self.drop_frame_zero_for_current_run
            else ""
        )
        self.status_label.setText(
            f"{self.analysis_label()} analysis "
            f"{self.individual_export_status()} "
            f"({len(self.live_dead_results)} frames).{frame_zero_note} "
            f"Outputs saved to {output_dir}; "
            f"summary GIF saved: {gif_path is not None}; "
            f"overlay GIFs saved: {len(overlay_gif_paths)}; "
            f"graphs saved: {len(summary_plot_paths)}."
            + (f" Export error: {self.individual_export_error}" if self.individual_export_error else "")
        )

    def compute_live_dead_stats(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute position-averaged means and SDs for paired channel metrics."""
        metric_columns = (
            "live_mean_intensity",
            "dead_mean_intensity",
            "live_integrated_intensity",
            "dead_integrated_intensity",
            "live_fractional_area",
            "dead_fractional_area",
            "live_mean_cell_intensity",
            "dead_mean_cell_intensity",
        )
        if self.is_cell_viability_run():
            metric_columns = tuple(column.replace("live_", "cell_viability_", 1)
                                   for column in metric_columns if column.startswith("live_"))
        missing = [column for column in metric_columns if column not in df.columns]
        if missing:
            raise ValueError(
                f"Missing {self.analysis_label()} metric columns: " + ", ".join(missing)
            )

        metric_columns += tuple(column for column in df.columns
                                if "_corrected_" in column and column.endswith("intensity"))
        aggregations = []
        for column in metric_columns:
            aggregations.extend([
                pl.col(column).mean().alias(column),
                pl.col(column).std().fill_null(0).alias(f"std_{column}"),
            ])

        stats = (
            df.group_by("time")
            .agg(aggregations)
            .sort("time")
        )
        return stats.with_columns(pl.Series(
            "time_hours",
            [self.get_time_hours(value) for value in stats["time"].to_list()],
        ))

    @staticmethod
    def get_live_dead_metric_spec(analysis_type: str) -> dict:
        specs = {
            "Cell-Area Staining Composition": {
                "filename": "cell_area_staining_composition.png",
                "plot_type": "cell_area_composition",
            },
            "Biomass Retention": {
                "filename": "biomass_retention.png",
                "plot_type": "biomass_retention",
            },
            "Mean Live-Dead Intensity": {
                "live": "live_mean_intensity",
                "dead": "dead_mean_intensity",
                "ylabel": "Mean Intensity",
                "filename": "mean_live_dead_intensity.png",
            },
            "Integrated Live-Dead Intensity": {
                "live": "live_integrated_intensity",
                "dead": "dead_integrated_intensity",
                "ylabel": "Integrated Intensity",
                "filename": "integrated_live_dead_intensity.png",
            },
            "Mean Live-Dead Intensity Error Plot": {
                "live": "live_mean_intensity",
                "dead": "dead_mean_intensity",
                "ylabel": "Mean Intensity",
                "filename": "mean_live_dead_intensity_error_plot.png",
                "plot_type": "error_bar",
            },
            "Integrated Live-Dead Intensity Error Plot": {
                "live": "live_integrated_intensity",
                "dead": "dead_integrated_intensity",
                "ylabel": "Integrated Intensity",
                "filename": "integrated_live_dead_intensity_error_plot.png",
                "plot_type": "error_bar",
            },
            "Live-Dead Fractional Area": {
                "live": "live_fractional_area",
                "dead": "dead_fractional_area",
                "ylabel": "Fractional Area",
                "filename": "live_dead_fractional_area.png",
            },
            "Mean Cell-Based Intensity": {
                "live": "live_mean_cell_intensity",
                "dead": "dead_mean_cell_intensity",
                "ylabel": "Mean Intensity per Cell",
                "filename": "mean_cell_based_intensity.png",
            },
            "Mean Live-Dead Ratio Comparison": {
                "filename": "image_overlaps/mean_live_dead_ratio_comparison.png",
                "plot_type": "ratio_comparison",
                "comparison_type": "mean",
            },
            "Integrated Live-Dead Ratio Comparison": {
                "filename": "image_overlaps/integrated_live_dead_ratio_comparison.png",
                "plot_type": "ratio_comparison",
                "comparison_type": "integrated",
            },
            "Mean Live-Dead Ratio Comparison Error Plot": {
                "filename": (
                    "image_overlaps/"
                    "mean_live_dead_ratio_comparison_error_plot.png"
                ),
                "plot_type": "ratio_error_bar",
                "comparison_type": "mean",
            },
            "Integrated Live-Dead Ratio Comparison Error Plot": {
                "filename": (
                    "image_overlaps/"
                    "integrated_live_dead_ratio_comparison_error_plot.png"
                ),
                "plot_type": "ratio_error_bar",
                "comparison_type": "integrated",
            },
        }
        lookup = analysis_type.replace("Cell Viability", "Live-Dead")
        if lookup not in specs:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        spec = dict(specs[lookup])
        if "Cell Viability" in analysis_type:
            spec["live"] = spec["live"].replace("live_", "cell_viability_", 1)
            spec.pop("dead", None)
            spec["filename"] = spec["filename"].replace("live_dead", "cell_viability")
        return spec

    def create_live_dead_metric_figure(
            self,
            stats: pl.DataFrame,
            analysis_type: str,
            figure=None,
            raw_df: pl.DataFrame | None = None,
    ):
        """Create the selected Live-Dead metric figure."""
        stats = self.background_corrected_plot_data(stats)
        raw_df = self.background_corrected_plot_data(self.live_dead_df if raw_df is None else raw_df)
        spec = self.get_live_dead_metric_spec(analysis_type)
        if self.is_cell_viability_run() and "live" in spec:
            spec["live"] = spec["live"].replace("live_", "cell_viability_", 1)
            spec.pop("dead", None)
        if spec.get("plot_type") in ("cell_area_composition", "biomass_retention"):
            return self.create_cell_area_figure(
                self.live_dead_df if raw_df is None else raw_df,
                composition=spec["plot_type"] == "cell_area_composition",
                figure=figure,
            )

        if spec.get("plot_type") == "ratio_error_bar":
            if raw_df is None:
                raw_df = self.live_dead_df
            return self.create_live_dead_ratio_error_figure(
                raw_df,
                comparison_type=spec["comparison_type"],
                figure=figure,
            )

        if spec.get("plot_type") == "ratio_comparison":
            if raw_df is None:
                raw_df = self.live_dead_df
            return self.create_live_dead_ratio_comparison_figure(
                raw_df,
                comparison_type=spec["comparison_type"],
                figure=figure,
            )

        if spec.get("plot_type") == "error_bar":
            if raw_df is None:
                raw_df = self.live_dead_df
            return self.create_live_dead_error_figure(
                raw_df,
                analysis_type,
                figure=figure,
            )

        if figure is None:
            figure = plt.figure(figsize=(8, 7), constrained_layout=True)
        else:
            figure.clear()

        axes = np.atleast_1d(figure.subplots(len(self.active_channel_specs()), 1))
        time_hours = stats["time_hours"].to_numpy()
        series = [(axes[0], "Cell Viability" if self.is_cell_viability_run() else "Live",
                   "green", spec["live"])]
        if not self.is_cell_viability_run():
            series.append((axes[1], "Dead", "red", spec["dead"]))
        for axis, channel_name, color, column in series:
            values = stats[column].to_numpy()
            std_values = stats[f"std_{column}"].to_numpy()
            axis.plot(time_hours, values, "-o", color=color, linewidth=2)
            axis.fill_between(
                time_hours,
                values - std_values,
                values + std_values,
                color=color,
                alpha=0.22,
            )
            axis.set_title(channel_name)
            axis.set_ylabel(spec["ylabel"])
            axis.set_xlabel("Time (hours)")

        figure.suptitle(analysis_type + (" (background corrected)" if self.cell_background_enabled() else ""),
                       fontsize=14, fontweight="bold")
        return figure

    def create_live_dead_error_figure(
            self,
            df: pl.DataFrame,
            analysis_type: str,
            figure=None,
    ):
        """Create a two-bar mean +/- SD plot with a Live-vs-Dead KS p-value."""
        spec = self.get_live_dead_metric_spec(analysis_type)
        if self.is_cell_viability_run() and "live" in spec:
            spec["live"] = spec["live"].replace("live_", "cell_viability_", 1)
            spec.pop("dead", None)
        live_values = np.asarray(df[spec["live"]].to_numpy(), dtype=np.float64)
        if self.is_cell_viability_run():
            values = live_values[np.isfinite(live_values)]
            if not values.size:
                raise ValueError("No finite Cell Viability measurements are available.")
            if figure is None:
                figure = plt.figure(figsize=(7, 6), constrained_layout=True)
            else:
                figure.clear()
            axis = figure.subplots()
            axis.bar([0], [values.mean()],
                     yerr=[values.std(ddof=1) if values.size > 1 else 0.0],
                     capsize=8, color="green", alpha=0.72)
            axis.set_xticks([0], ["Cell Viability"])
            axis.set_ylabel(spec["ylabel"])
            axis.set_title(analysis_type + (" (background corrected)" if self.cell_background_enabled() else ""))
            axis.grid(axis="y", alpha=0.2)
            return figure
        dead_values = np.asarray(df[spec["dead"]].to_numpy(), dtype=np.float64)
        live_values = live_values[np.isfinite(live_values)]
        dead_values = dead_values[np.isfinite(dead_values)]

        if not live_values.size or not dead_values.size:
            raise ValueError(
                "Live and Dead measurements are both required for the KS test."
            )

        means = np.asarray([
            np.mean(live_values, dtype=np.float64),
            np.mean(dead_values, dtype=np.float64),
        ])
        standard_deviations = np.asarray([
            np.std(live_values, ddof=1, dtype=np.float64)
            if live_values.size > 1 else 0.0,
            np.std(dead_values, ddof=1, dtype=np.float64)
            if dead_values.size > 1 else 0.0,
        ])
        ks_result = ks_2samp(live_values, dead_values, method="auto")

        if figure is None:
            figure = plt.figure(figsize=(7, 6), constrained_layout=True)
        else:
            figure.clear()
        axis = figure.subplots(1, 1)

        x_positions = np.arange(2)
        axis.bar(
            x_positions,
            means,
            yerr=standard_deviations,
            capsize=8,
            width=0.62,
            color=["green", "red"],
            edgecolor=["darkgreen", "darkred"],
            alpha=0.72,
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
        )
        axis.set_xticks(x_positions, ["Live", "Dead"])
        axis.set_ylabel(spec["ylabel"])
        axis.set_title(analysis_type + (" (background corrected)" if self.cell_background_enabled() else ""), fontsize=14, fontweight="bold")

        upper_value = float(np.max(means + standard_deviations))
        lower_value = min(0.0, float(np.min(means - standard_deviations)))
        value_span = upper_value - lower_value
        if not np.isfinite(value_span) or value_span <= 0:
            value_span = max(abs(upper_value), 1.0)

        bracket_bottom = upper_value + (0.08 * value_span)
        bracket_top = bracket_bottom + (0.04 * value_span)
        axis.plot(
            [0, 0, 1, 1],
            [bracket_bottom, bracket_top, bracket_top, bracket_bottom],
            color="black",
            linewidth=1.0,
        )
        p_value_text = (
            "< 1e-300"
            if ks_result.pvalue < 1e-300
            else f"{ks_result.pvalue:.3g}"
        )
        axis.text(
            0.5,
            bracket_top + (0.025 * value_span),
            f"KS p = {p_value_text}",
            ha="center",
            va="bottom",
        )
        axis.set_ylim(
            lower_value,
            bracket_top + (0.14 * value_span),
        )
        axis.grid(axis="y", alpha=0.2)
        return figure

    def derive_cell_area_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """Derive cell-area fractions and retention within each position.

        The first selected frame is the baseline. Times follow the current
        capture interval, like the existing summary plots. Empty denominators
        are undefined, not zero staining or zero retention.
        """
        if self.is_cell_viability_run():
            return self.derive_cell_viability_area_metrics(df)
        required_columns = (
            "position",
            "time",
            "cell_area_pixels",
            "live_cell_overlap_pixels",
            "dead_cell_overlap_pixels",
            "cell_live_dead_overlap_pixels",
            "live_total_cells",
        )
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise ValueError(
                "Missing cell-area metric columns: " + ", ".join(missing)
            )

        baselines = {}
        records = []

        def percent(numerator, denominator):
            if (
                    numerator is None
                    or denominator is None
                    or not np.isfinite(numerator)
                    or not np.isfinite(denominator)
                    or denominator <= 0
            ):
                return None
            return 100.0 * float(numerator) / float(denominator)

        for row in df.sort(["position", "time"]).to_dicts():
            base = baselines.setdefault(row["position"], row)
            area = row["cell_area_pixels"]
            live = row["live_cell_overlap_pixels"]
            dead = row["dead_cell_overlap_pixels"]
            both = row["cell_live_dead_overlap_pixels"]
            records.append({
                "position": row["position"],
                "time": row["time"],
                "time_hours": self.get_time_hours(row["time"]),
                "baseline_time": base["time"],
                "baseline_time_hours": self.get_time_hours(base["time"]),
                "cell_count": row["live_total_cells"],
                "cell_area_pixels": area,
                "live_only_percent": percent(live - both, area),
                "dead_only_percent": percent(dead - both, area),
                "double_positive_percent": percent(both, area),
                "neither_percent": percent(area - live - dead + both, area),
                "dead_positive_percent": percent(dead, area),
                "cell_count_retention_percent": percent(
                    row["live_total_cells"], base["live_total_cells"]),
                "cell_area_retention_percent": percent(
                    area, base["cell_area_pixels"]),
            })
        return pl.DataFrame(records)

    def derive_cell_viability_area_metrics(self, df):
        baselines = {}
        records = []
        def percent(value, total):
            return 100.0 * value / total if total > 0 else None
        for row in df.sort(["position", "time"]).to_dicts():
            base = baselines.setdefault(row["position"], row)
            area = row["cell_area_pixels"]
            positive = row["cell_viability_cell_overlap_pixels"]
            count = row["cell_viability_total_cells"]
            records.append({
                "position": row["position"], "time": row["time"],
                "time_hours": self.get_time_hours(row["time"]),
                "baseline_time": base["time"],
                "baseline_time_hours": self.get_time_hours(base["time"]),
                "cell_count": count, "cell_area_pixels": area,
                "cell_viability_positive_percent": percent(positive, area),
                "cell_viability_negative_percent": percent(area - positive, area),
                "cell_count_retention_percent": percent(count, base["cell_viability_total_cells"]),
                "cell_area_retention_percent": percent(area, base["cell_area_pixels"]),
            })
        return pl.DataFrame(records)

    def create_cell_area_figure(self, df, *, composition, figure=None):
        """Plot area composition or paired first/last-frame retention."""
        data = self.derive_cell_area_metrics(df)
        if figure is None:
            figure = plt.figure(figsize=(8, 6) if composition else (12, 5),
                                constrained_layout=True)
        else:
            figure.clear()
        if composition:
            columns = (["cell_viability_positive_percent", "cell_viability_negative_percent"]
                       if self.is_cell_viability_run() else
                       ["live_only_percent", "dead_only_percent", "double_positive_percent", "neither_percent"])
            valid_data = data.drop_nulls(columns)
            if valid_data.is_empty():
                raise ValueError(
                    "Cell-area composition requires at least one frame with "
                    "a non-empty segmented cell mask."
                )
            summary = valid_data.group_by("time_hours").agg(
                [pl.col(c).mean() for c in columns]
            ).sort("time_hours")
            ax = figure.subplots()
            x = np.arange(summary.height)
            bottom = np.zeros(summary.height)
            for column, label, color in zip(
                columns,
                (["Cell Viability-positive", "Cell Viability-negative"] if self.is_cell_viability_run()
                 else ["Live-only", "Dead-only", "Double-positive", "Neither"]),
                (["#27896c", "#d7d9dd"] if self.is_cell_viability_run()
                 else ["#27896c", "#d55c66", "#ab86be", "#d7d9dd"]),
            ):
                values = np.asarray(summary[column].to_list(), dtype=float)
                ax.bar(
                    x,
                    values,
                    bottom=bottom,
                    width=0.55,
                    label=label,
                    color=color,
                )
                for index, value in enumerate(values):
                    if np.isfinite(value) and value >= 5:
                        ax.text(index, bottom[index] + value / 2,
                                f"{value:.1f}%", ha="center", va="center")
                bottom += values
            ax.set(xticks=x, xticklabels=[f"{t:g} h" for t in summary["time_hours"]],
                   ylim=(0, 100), ylabel="Segmented cell area (%)",
                   title="Cell-area staining composition\nMean across valid positions")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
                      ncol=2, frameon=False)
        else:
            axes = figure.subplots(1, 3)
            specifications = [
                ("cell_count_retention_percent", "Cell count retention", "% of position baseline"),
                ("cell_area_retention_percent", "Occupied area retention", "% of position baseline"),
                ("cell_viability_positive_percent" if self.is_cell_viability_run() else "dead_positive_percent",
                 "Cell Viability-positive cell area" if self.is_cell_viability_run() else "Dead-positive cell area",
                 "% of segmented cell area"),
            ]
            for ax, (column, title, ylabel) in zip(axes, specifications):
                for group in data.partition_by("position", maintain_order=True):
                    if group.height < 2:
                        continue
                    pair = [group.row(0, named=True), group.row(-1, named=True)]
                    values = [
                        np.nan if row[column] is None else row[column]
                        for row in pair
                    ]
                    if not np.any(np.isfinite(values)):
                        continue
                    ax.plot([r["time_hours"] for r in pair],
                            values,
                            "o-", label=f"Position {pair[0]['position']}")
                ax.set(title=title, xlabel="Time (h)", ylabel=ylabel)
                ax.set_ylim(bottom=0, top=max(105, ax.get_ylim()[1]))
                ax.grid(axis="y", alpha=0.18)
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                axes[0].legend(handles, labels, fontsize="small", frameon=False)
            else:
                axes[0].text(0.5, 0.5, "Two timepoints per position required",
                             ha="center", transform=axes[0].transAxes, wrap=True)
            figure.suptitle(
                "Biomass retention and Cell Viability-positive staining\nPaired first/last selected frames"
                if self.is_cell_viability_run() else
                "Biomass retention and dead-positive staining\n"
                "Paired first/last selected frames; dead-positive includes double-positive"
            )
        return figure

    def save_all_summary_plots(self, output_dir: Path):
        if not hasattr(self, "live_dead_df") or self.live_dead_df.is_empty():
            return []

        stats = self.compute_live_dead_stats(self.live_dead_df)
        stats.write_csv(output_dir / f"{self.output_prefix()}_summary_values.csv")
        self.derive_cell_area_metrics(self.live_dead_df).write_csv(
            output_dir / "cell_area_retention_metrics.csv"
        )

        saved_paths = []
        for index in range(self.metric_combo.count()):
            analysis_type = self.metric_combo.itemText(index)
            spec = self.get_live_dead_metric_spec(analysis_type)
            figure = self.create_live_dead_metric_figure(
                stats,
                analysis_type,
                raw_df=self.live_dead_df,
            )
            output_path = output_dir / spec["filename"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(figure)
            saved_paths.append(output_path)
        return saved_paths

    def on_plot_avg_sd(self):
        if not hasattr(self, "live_dead_df") or self.live_dead_df.is_empty():
            QMessageBox.warning(self, "No data", f"Run {self.analysis_label()} analysis first.")
            return

        analysis_type = self.metric_combo.currentText()
        try:
            stats = self.compute_live_dead_stats(self.live_dead_df)
            self.create_live_dead_metric_figure(
                stats,
                analysis_type,
                figure=self.population_figure,
                raw_df=self.live_dead_df,
            )
        except Exception as error:
            QMessageBox.warning(self, "Cannot generate graph", str(error))
            return

        self.population_canvas.draw()
        output_dir = self.get_live_dead_output_root()
        spec = self.get_live_dead_metric_spec(analysis_type)
        output_path = output_dir / spec["filename"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.population_figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    def get_cell_morphology_radius_px(self):
        """
        Match the fixed best-candidate script:
            radius_px = round((0.65 um / 2) / pixel_size_um)
        """

        try:
            from nd2_analyzer.data.image_data import ImageData

            image_data = ImageData.get_instance()
            pixel_size_um = self.get_pixel_size_um(image_data)

            if pixel_size_um <= 0 or pixel_size_um == 1.0:
                pixel_size_um = self.CELL_FALLBACK_PIXEL_SIZE_UM

        except Exception:
            pixel_size_um = self.CELL_FALLBACK_PIXEL_SIZE_UM

        radius_um = self.CELL_MORPHOLOGY_KERNEL_DIAMETER_UM / 2.0

        return max(
            1,
            int(round(radius_um / pixel_size_um))
        )

    def get_pixel_size_um(self, image_data):
        """Get pixel size in um from ImageData, with a safe fallback."""
        voxel_size = getattr(image_data, "voxel_size", None)

        if hasattr(voxel_size, "x"):
            return float(voxel_size.x)

        if isinstance(voxel_size, (int, float)):
            return float(voxel_size)

        return 1.0

    def get_live_dead_output_root(self):
        output_dir = Path(f"{self.output_prefix()}_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def get_time_hours(self, time: int) -> float:
        interval_value = self.frame_interval_value.value()
        interval_unit = self.time_unit_combo.currentText()

        hours_conversion = {
            "ms": 1 / 1000 / 3600,
            "sec": 1 / 3600,
            "min": 1 / 60,
            "hr": 1,
            "day": 24,
        }

        return float(time) * interval_value * hours_conversion[interval_unit]

    def normalize_image_for_display(self, image: np.ndarray) -> np.ndarray:
        """Generic per-frame display normalization, retained for cell images."""
        img = np.asarray(image, dtype=np.float32)
        finite_mask = np.isfinite(img)

        if not np.any(finite_mask):
            return np.zeros(img.shape, dtype=np.float32)

        finite_values = img[finite_mask]
        low, high = np.percentile(finite_values, [1, 99])

        if high <= low:
            low = float(np.min(finite_values))
            high = float(np.max(finite_values))

        if high <= low:
            return np.zeros(img.shape, dtype=np.float32)

        normalized = np.array(img, dtype=np.float32, copy=True)
        normalized -= np.float32(low)
        normalized /= np.float32(high - low)
        np.clip(normalized, 0.0, 1.0, out=normalized)
        np.nan_to_num(normalized, copy=False)
        return normalized

    def normalize_channel_image_for_display(self, image: np.ndarray) -> np.ndarray:
        """
        Display either fluorescence channel with the shared segmentation window.
        """
        black_point = self.live_dead_processing_black_point_raw
        white_point = self.live_dead_processing_white_point_raw

        if (
                black_point is None
                or white_point is None
                or not np.isfinite(black_point)
                or not np.isfinite(white_point)
                or white_point <= black_point
        ):
            return self.normalize_image_for_display(image)

        img = np.asarray(image, dtype=np.float32)
        normalized = np.array(img, dtype=np.float32, copy=True)
        normalized -= np.float32(black_point)
        normalized /= np.float32(white_point - black_point)
        np.clip(normalized, 0.0, 1.0, out=normalized)
        np.nan_to_num(normalized, copy=False)
        return normalized

    @staticmethod
    def make_single_mask_rgba(
            mask: np.ndarray,
            rgba: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Create a compact uint8 RGBA fill instead of a contourf polygon set."""
        mask = np.asarray(mask, dtype=bool)
        overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
        overlay[mask] = np.asarray(rgba, dtype=np.uint8)
        return overlay

    def make_channel_cell_overlap_rgba(
            self,
            cell_mask: np.ndarray,
            channel_mask: np.ndarray,
            channel_rgb: tuple[int, int, int],
    ) -> np.ndarray:
        """Show cell-only pixels in purple and channel overlap in its fixed color."""
        cell_mask = np.asarray(cell_mask, dtype=bool)
        channel_mask = np.asarray(channel_mask, dtype=bool)
        overlay = np.zeros((*cell_mask.shape, 4), dtype=np.uint8)
        overlay[cell_mask & ~channel_mask] = (140, 0, 217, 70)
        overlay[channel_mask & ~cell_mask] = (*channel_rgb, 105)
        overlay[channel_mask & cell_mask] = (*channel_rgb, 190)
        return overlay

    def save_channel_overlay(
            self,
            raw_frame: np.ndarray,
            channel_mask: np.ndarray,
            channel_name: str,
            channel_rgb: tuple[int, int, int],
            output_path: Path,
    ):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.normalize_channel_image_for_display(raw_frame), cmap="gray")
        ax.imshow(self.make_single_mask_rgba(channel_mask, (*channel_rgb, 70)))
        ax.contour(
            channel_mask,
            levels=[0.5],
            colors=[tuple(value / 255.0 for value in channel_rgb)],
            linewidths=0.8,
        )
        ax.set_title(f"{channel_name} fixed-threshold mask", fontsize=9)
        ax.axis("off")
        fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def save_cells_on_cells_overlay(
            self,
            raw_cell_frame: np.ndarray,
            cell_mask: np.ndarray,
            output_path: Path,
    ):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.normalize_image_for_display(raw_cell_frame), cmap="gray")
        ax.imshow(self.make_single_mask_rgba(cell_mask, (255, 105, 180, 71)))
        ax.contour(cell_mask, levels=[0.5], colors="purple", linewidths=0.9)
        ax.set_title("Processed cell mask on cell channel", fontsize=9)
        ax.axis("off")
        fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def save_channel_cell_overlap_overlay(
            self,
            raw_cell_frame: np.ndarray,
            cell_mask: np.ndarray,
            channel_mask: np.ndarray,
            channel_name: str,
            channel_rgb: tuple[int, int, int],
            output_path: Path,
    ):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.normalize_image_for_display(raw_cell_frame), cmap="gray")
        ax.imshow(self.make_channel_cell_overlap_rgba(
            cell_mask,
            channel_mask,
            channel_rgb,
        ))
        ax.contour(cell_mask, levels=[0.5], colors="purple", linewidths=0.8)
        ax.set_title(
            f"{channel_name}/cell overlap — purple=cell, "
            f"{channel_name.lower()} color=threshold-positive overlap",
            fontsize=9,
        )
        ax.axis("off")
        fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def make_fluorescence_cell_mask(self, cell_frame: np.ndarray):
        """
        Build a cell mask when the selected Cell View Type is Fluorescence.

        The raw cell fluorescence channel is thresholded directly, then binary
        closing and dilation are applied. Phase Contrast continues to use the
        saved Partaker segmentation masks instead of this helper.
        """

        cell_frame = np.asarray(cell_frame, dtype=float)
        cell_threshold = float(self.CELL_THRESHOLD)

        finite_values = cell_frame[np.isfinite(cell_frame)]

        if finite_values.size == 0:
            cell_mask = np.zeros_like(cell_frame, dtype=bool)
            return cell_mask, cell_threshold, "empty", cell_frame

        cell_for_thresholding = cell_frame
        polarity_used = "raw_fluorescence"

        cell_mask = cell_for_thresholding >= cell_threshold

        morphology_radius_px = self.get_cell_morphology_radius_px()
        morphology_kernel = disk(morphology_radius_px)

        cell_mask = binary_closing(
            cell_mask,
            footprint=morphology_kernel
        )

        cell_mask = binary_dilation(
            cell_mask,
            footprint=morphology_kernel
        )

        cell_occupancy = (
            np.count_nonzero(cell_mask) / cell_mask.size
            if cell_mask.size > 0
            else 0.0
        )

        print(
            "Fluorescence cell view | "
            f"polarity={polarity_used} | "
            f"threshold={cell_threshold} | "
            f"morph_radius_px={morphology_radius_px} | "
            f"cell_occ={cell_occupancy:.4f}"
        )

        return (
            cell_mask,
            cell_threshold,
            polarity_used,
            cell_for_thresholding,
        )

    def apply_channel_morphology(
            self,
            channel_mask: np.ndarray,
            closing_radius_px: int = 0,
            dilation_radius_px: int = 0,
    ):
        """
        Optional channel morphology after binarization.

        Order is closing -> dilation.
        Radius 0 means that operation is skipped.
        """

        morphed = channel_mask.copy()

        if closing_radius_px > 0:
            morphed = binary_closing(
                morphed,
                footprint=disk(closing_radius_px),
            )

        if dilation_radius_px > 0:
            morphed = binary_dilation(
                morphed,
                footprint=disk(dilation_radius_px),
            )

        return morphed

    def convert_channel_image_to_uint8_for_processing(self, image: np.ndarray):
        """
        Map raw fluorescence intensities to the shared uint8 processing window.

        Values at or below the shared black point become 0. Values at or above
        the shared white point become 255. The same mapping is reused across all
        selected positions and timepoints.
        """
        img = np.asarray(image, dtype=np.float32)
        finite_mask = np.isfinite(img)

        if not np.any(finite_mask):
            return np.zeros(img.shape, dtype=np.uint8)

        black_point = self.live_dead_processing_black_point_raw
        white_point = self.live_dead_processing_white_point_raw

        if (
                black_point is None
                or white_point is None
                or not np.isfinite(black_point)
                or not np.isfinite(white_point)
                or white_point <= black_point
        ):
            raise RuntimeError(
                f"The {self.analysis_label()} run-level intensity window has not been initialized."
            )

        work = np.array(img, dtype=np.float32, copy=True)
        np.nan_to_num(
            work,
            copy=False,
            nan=float(black_point),
            posinf=float(white_point),
            neginf=float(black_point),
        )

        work -= np.float32(black_point)
        work /= np.float32(white_point - black_point)
        np.clip(work, 0.0, 1.0, out=work)
        work *= np.float32(255.0)
        np.rint(work, out=work)
        return work.astype(np.uint8, copy=False)

    def segment_fixed_channel(
            self,
            gaussian_corrected: np.ndarray,
            smoothing_sigma: float = 1.5,
    ):
        print(
            f"Starting Partaker Gaussian smoothing and shared fixed thresholding "
            f"for {self}"
        )
        self.log_process_memory("Before Partaker Gaussian smoothing")

        # Light smoothing before segmentation. The input is uint8, so retain
        # uint8 output instead of allowing a larger floating-point result.
        smoothed = np.empty_like(gaussian_corrected, dtype=np.uint8)
        gaussian_filter(
            gaussian_corrected,
            sigma=smoothing_sigma,
            output=smoothed,
        )

        threshold = int(self.partaker_live_dead_threshold)
        channel_mask = smoothed >= threshold

        # Optional morphology retained as preprocessing for the fixed method.
        if self.close_dialate.isChecked():
            kernel_radius_px = self.get_cell_morphology_radius_px()
            channel_mask = self.apply_channel_morphology(
                channel_mask,
                closing_radius_px=kernel_radius_px,
                dilation_radius_px=kernel_radius_px,
            )

        return smoothed, channel_mask, {
            "analysis_method": "Partaker",
            "threshold_used": float(threshold),
            "thresholding_type": "fixed_global_uint8",
            "partaker_gaussian_sigma": float(smoothing_sigma),
        }

    def get_partaker_cell_mask_for_frame(
            self,
            time: int,
            position: int,
    ):
        """
        Get the saved Partaker cell labels for the selected Cell Channel.

        Edge-centroid filtering is already applied when the segmentation is saved
        by SegmentationService.
        """
        from nd2_analyzer.data.image_data import ImageData

        image_data = ImageData.get_instance()
        segmented_storage = image_data.segmentation_cache
        model_name = segmented_storage.model_name
        cache = segmented_storage.with_model(model_name)

        selected_cell_channel = self.get_selected_cell_channel()

        cell_labels = np.asarray(
            cache[(
                time,
                position,
                selected_cell_channel,
                model_name,
            )]
        )

        cell_mask = cell_labels > 0

        return cell_mask, cell_labels

    def compute_channel_metrics(
            self,
            raw_image: np.ndarray,
            channel_mask: np.ndarray,
            cell_mask: np.ndarray,
            cell_labels: np.ndarray | None,
    ) -> dict:
        """Compute whole-frame and per-cell metrics for one fluorescence channel."""
        raw_image = np.asarray(raw_image, dtype=np.float32)
        channel_mask = np.asarray(channel_mask, dtype=bool)
        cell_mask = np.asarray(cell_mask, dtype=bool)

        positive_pixels = int(np.count_nonzero(channel_mask))
        total_pixels = int(channel_mask.size)
        fractional_area = (
            positive_pixels / total_pixels if total_pixels > 0 else 0.0
        )
        if positive_pixels:
            positive_values = raw_image[channel_mask]
            positive_values = positive_values[np.isfinite(positive_values)]
            mean_intensity = (
                float(np.mean(positive_values, dtype=np.float64))
                if positive_values.size
                else 0.0
            )
            integrated_intensity = float(
                np.sum(positive_values, dtype=np.float64)
            )
        else:
            mean_intensity = 0.0
            integrated_intensity = 0.0

        labels = (
            np.asarray(cell_labels)
            if cell_labels is not None
            else label_connected_components(cell_mask)
        )
        cell_ids = np.unique(labels)
        cell_ids = cell_ids[cell_ids > 0]
        per_cell_means = []
        for cell_id in cell_ids:
            overlap = (labels == cell_id) & channel_mask
            overlap_values = raw_image[overlap]
            overlap_values = overlap_values[np.isfinite(overlap_values)]
            per_cell_means.append(
                float(np.mean(overlap_values, dtype=np.float64))
                if overlap_values.size
                else 0.0
            )

        mean_cell_intensity = (
            float(np.mean(per_cell_means, dtype=np.float64))
            if per_cell_means
            else 0.0
        )

        return {
            "mean_intensity": mean_intensity,
            "integrated_intensity": integrated_intensity,
            "fractional_area": fractional_area,
            "fractional_area_percent": fractional_area * 100.0,
            "area_pixels": positive_pixels,
            "mean_cell_intensity": mean_cell_intensity,
            "cell_overlap_pixels": int(np.count_nonzero(channel_mask & cell_mask)),
            "total_cells": int(len(cell_ids)),
        }

    @staticmethod
    def log_process_memory(prefix: str):
        """Print resident memory without making psutil a required dependency."""
        try:
            import psutil

            rss_mb = (
                    psutil.Process(os.getpid()).memory_info().rss
                    / (1024.0 * 1024.0)
            )
            print(f"{prefix} | RSS={rss_mb:.1f} MB")
            return
        except Exception:
            pass

        try:
            import resource
            import sys

            max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # macOS reports bytes; Linux reports KiB.
            rss_mb = (
                max_rss / (1024.0 * 1024.0)
                if sys.platform == "darwin"
                else max_rss / 1024.0
            )
            print(f"{prefix} | max RSS={rss_mb:.1f} MB")
        except Exception:
            # Memory logging is diagnostic only.
            pass

    def reset_output_tracking(self):
        self.heatmap_rows = []
        self.gif_rows = []
        self.overlay_gif_rows = []

    def append_intensity_heatmap_rows(
            self,
            *,
            source: str,
            stage: str,
            image: np.ndarray,
            time: int,
            position: int,
            channel: int,
            mask: np.ndarray | None = None,
            intensity_min: float = 0.0,
            intensity_max: float = 255.0,
            bin_count: int = 256,
    ):
        """
        Store one compact histogram record per frame/stage.

        The previous implementation stored 256 Python dictionaries for every
        histogram. Four histograms per frame produced 1,024 dictionaries per
        frame and tens of thousands of retained objects.
        """
        values = np.asarray(image)

        if mask is not None:
            values = values[np.asarray(mask, dtype=bool)]

        values = values[np.isfinite(values)]
        values = values[
            (values >= intensity_min)
            & (values <= intensity_max)
            ]

        counts, edges = np.histogram(
            values,
            bins=bin_count,
            range=(intensity_min, intensity_max),
        )

        if not hasattr(self, "heatmap_rows"):
            self.heatmap_rows = []

        self.heatmap_rows.append({
            "source": source,
            "stage": stage,
            "time": int(time),
            "time_hours": float(self.get_time_hours(time)),
            "position": int(position),
            "channel": int(channel),
            "bin_starts": edges[:-1].astype(np.float32, copy=False),
            "bin_ends": edges[1:].astype(np.float32, copy=False),
            "counts": counts.astype(np.int64, copy=False),
        })

    def save_processing_heatmaps(self, output_dir: Path):
        if not hasattr(self, "heatmap_rows") or not self.heatmap_rows:
            return

        heatmap_dir = output_dir / "heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        all_csv = heatmap_dir / "all_processing_heatmap_values.csv"
        fieldnames = [
            "source",
            "stage",
            "time",
            "time_hours",
            "position",
            "channel",
            "bin_start",
            "bin_end",
            "count",
        ]

        with open(all_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in self.heatmap_rows:
                for bin_start, bin_end, count in zip(
                        row["bin_starts"],
                        row["bin_ends"],
                        row["counts"],
                ):
                    writer.writerow({
                        "source": row["source"],
                        "stage": row["stage"],
                        "time": row["time"],
                        "time_hours": row["time_hours"],
                        "position": row["position"],
                        "channel": row["channel"],
                        "bin_start": float(bin_start),
                        "bin_end": float(bin_end),
                        "count": int(count),
                    })

        grouped_rows = {}
        for row in self.heatmap_rows:
            key = (row["source"], row["stage"])
            grouped_rows.setdefault(key, []).append(row)

        for (source, stage), rows in grouped_rows.items():
            times = sorted(set(row["time_hours"] for row in rows))
            time_to_col = {value: i for i, value in enumerate(times)}

            bins = np.asarray(rows[0]["bin_starts"], dtype=np.float32)
            matrix = np.zeros(
                (len(bins), len(times)),
                dtype=np.float64,
            )

            for row in rows:
                matrix[:, time_to_col[row["time_hours"]]] += row["counts"]

            label = f"{source}_{stage}_processing_heatmap"
            heatmap_csv = heatmap_dir / f"{label}.csv"

            with open(heatmap_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["bin_start"] + [f"time_h_{t:g}" for t in times])

                for bin_index, bin_value in enumerate(bins):
                    writer.writerow(
                        [float(bin_value)] + matrix[bin_index, :].tolist()
                    )

    def set_run_configuration_enabled(self, enabled):
        """Keep the processing settings fixed while allowing Cancel to remain active."""
        for control in (
            self.live_channel_combo, self.dead_channel_combo, self.cell_viability_channel_combo,
            self.cell_channel_combo, self.cell_view_combo,
            self.save_fluorescence_masks_checkbox, self.position_list,
            self.time_start_spin, self.time_end_spin, self.frame_interval_value,
            self.time_unit_combo, self.gaus_back_corr, self.close_dialate,
            self.drop_frame_zero_checkbox,
        ):
            control.setEnabled(enabled)

    @staticmethod
    def _export_json_value(value):
        """Convert NumPy/path metadata without writing NaN or Infinity into JSON."""
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return LiveDeadAnalysisWidget._export_json_value(value.tolist())
        if isinstance(value, np.generic):
            return LiveDeadAnalysisWidget._export_json_value(value.item())
        if isinstance(value, dict):
            return {str(k): LiveDeadAnalysisWidget._export_json_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [LiveDeadAnalysisWidget._export_json_value(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    def individual_cell_fieldnames(self):
        fields = [
            "run_id", "dataset_id", "sample_id", "treatment", "analysis_mode",
            "position", "time", "time_hours", "cell_id", "cell_channel",
            "cell_area_pixels", "cell_area_um2", "centroid_x", "centroid_y",
            "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max",
        ]
        for name, _display, _color in self.active_channel_specs():
            fields.append(f"{name}_channel")
            fields.extend(f"{name}_{suffix}" for suffix in (
                "whole_cell_mean_intensity", "whole_cell_integrated_intensity",
                "whole_cell_valid_pixels", "whole_cell_intensity_status",
                "positive_area_pixels", "positive_area_fraction", "positive_valid_pixels",
                "positive_mean_intensity", "positive_integrated_intensity",
                "positive_intensity_status",
                "corrected_whole_cell_mean_intensity", "corrected_whole_cell_integrated_intensity",
                "corrected_whole_cell_valid_pixels", "corrected_whole_cell_intensity_status",
                "corrected_positive_mean_intensity", "corrected_positive_integrated_intensity",
                "corrected_positive_valid_pixels", "corrected_positive_intensity_status",
                "background_median", "background_eligible_pixels", "background_eligible_fraction",
                "background_mad", "background_status", "background_mean", "background_sd", "background_method",
            ))
        if not self.is_cell_viability_run():
            fields.extend([
                "live_dead_intersection_pixels", "live_dead_union_pixels",
                "live_dead_intersection_over_union", "live_positive_also_dead_fraction",
                "dead_positive_also_live_fraction", "dead_to_live_whole_cell_mean_ratio",
                "ratio_status", "corrected_dead_to_live_whole_cell_mean_ratio", "corrected_ratio_status",
            ])
        return fields

    def build_analysis_provenance(self, *, frames, selected_positions,
                                  time_start, time_end, skipped_frames):
        from nd2_analyzer.data.image_data import ImageData
        from nd2_analyzer.data.appstate import ApplicationState
        image_data = ImageData.get_instance()
        appstate = ApplicationState.get_instance()
        experiment = getattr(appstate, "experiment", None)
        source_paths = getattr(image_data, "image_filename", [])
        if isinstance(source_paths, (str, Path)):
            source_paths = [source_paths]
        paths = list(dict.fromkeys(str(p) for p in source_paths if p is not None))
        sources = []
        for path in paths:
            record = {"path": str(Path(path).expanduser().absolute())}
            try:
                stat = Path(path).stat()
                record.update(size_bytes=stat.st_size, modified_time_ns=stat.st_mtime_ns)
            except OSError:
                record["available_at_run_start"] = False
            sources.append(record)
        file_map = getattr(image_data, "_tiff_file_map", None)
        if file_map is None:
            file_map = {",".join(str(v) for v in k): str(path)
                        for k, path in getattr(experiment, "file_map", {}).items()}
        tiff_axis_values = None
        if file_map:
            keys = [key.split(",") for key in file_map]
            tiff_axis_values = {
                "positions": sorted({int(key[0]) for key in keys}),
                "channels": sorted({int(key[2]) for key in keys}),
                "times": sorted({int(key[1]) for key in keys if key[1] != "None"}),
                "interpretation": "Loaded position/channel/time indices refer to these sorted source IDs; stacked TIFF time is its zero-based page index.",
            }
        shape = tuple(int(v) for v in image_data.data.shape)
        dataset_descriptor = self._export_json_value({"sources": sources, "shape": shape,
                                                     "file_map": file_map})
        dataset_id = hashlib.sha256(json.dumps(dataset_descriptor, sort_keys=True).encode()).hexdigest()
        voxel = getattr(image_data, "voxel_size", None)
        x = getattr(voxel, "x", None)
        y = getattr(voxel, "y", None)
        if isinstance(voxel, (int, float, np.number)):
            x = y = float(voxel)
        known_size = (x is not None and y is not None and np.isfinite(x)
                      and np.isfinite(y) and x > 0 and y > 0)
        channel_names = getattr(experiment, "channel_names", {})
        frame = frames[0]
        channel_indices = {name: int(frame[2] if i == 0 else frame[3])
                           for i, (name, _label, _color) in enumerate(self.active_channel_specs())}
        cell_channel = self.get_selected_cell_channel()
        cache = getattr(image_data, "segmentation_cache", None)
        setup = self.partaker_live_dead_setup
        source_code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        provenance = {
            "schema_version": 1, "run_id": str(uuid.uuid4()), "dataset_id": dataset_id,
            "sample_id": getattr(experiment, "name", ""),
            "treatment": "", "analysis_mode": self.run_analysis_mode,
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
            "sources": sources, "source_dimensions": shape, "source_axes": "TPCYX",
            "source_dtype": str(image_data.data.dtype), "tiff_file_map": file_map,
            "tiff_source_axis_values": tiff_axis_values,
            "tiff_import_mode": getattr(image_data, "_tiff_mode", getattr(experiment, "import_mode", None)),
            "source_mapping": "For directory TIFFs use the stored file map; for stacks use loader time index; otherwise use T/P/C indices in the source dataset.",
            "channel_assignments": {**channel_indices, "cell": cell_channel},
            "configured_channel_names": channel_names,
            "pixel_calibration": {
                "x_um": float(x) if known_size else None, "y_um": float(y) if known_size else None,
                "status": "available_image_metadata" if known_size else "unavailable",
                "morphology_fallback_um": self.CELL_FALLBACK_PIXEL_SIZE_UM,
                "fluorescence_morphology_radius_pixels": self.get_cell_morphology_radius_px()
                    if self.close_dialate.isChecked() else None,
            },
            "transforms": {
                "crop_xywh": getattr(image_data, "crop_coordinates", None),
                "registration_offsets_xy_by_time": getattr(image_data, "registration_offsets", None),
                "application_order": ["crop", "registration"],
                "registration_implementation": "ShiftedImage_2D_numba",
                "registration_sampling": "Output (y,x) samples cropped input (y-YShift,x-XShift), with nearest-edge padding.",
                "coordinate_space": "cropped_registered_analysis_frame",
                "bounding_box_maximum": "exclusive",
            },
            "segmentation": {
                "cell_channel": cell_channel, "cell_view_type": self.cell_view_combo.currentText(),
                "model_name": getattr(cache, "model_name", None),
                "source": "partaker_segmented_mask" if self.cell_view_combo.currentText() == "Phase Contrast"
                    else "fixed_fluorescence_thresholding",
                "fluorescence_cell_threshold": self.CELL_THRESHOLD,
                "fluorescence_cell_morphology_radius_pixels": self.get_cell_morphology_radius_px(),
                "cell_ids": "frame-specific segmentation labels; not tracked identities",
            },
            "processing": {
                "threshold_uint8": self.partaker_live_dead_threshold,
                "threshold_raw_equivalent": (None if self.cell_background_enabled() else
                                             self.live_dead_processed_value_to_raw(self.partaker_live_dead_threshold)),
                "threshold_corrected_equivalent": (self.live_dead_processed_value_to_raw(self.partaker_live_dead_threshold)
                                                   if self.cell_background_enabled() else None),
                "black_point_raw": self.live_dead_processing_black_point_raw,
                "white_point_raw": self.live_dead_processing_white_point_raw,
                "window_source": self.live_dead_processing_window_source,
                "smoothing_sigma": self.partaker_live_dead_smoothing_sigma,
                "background_subtraction": self.cell_background_enabled() or self.gaus_back_corr.isChecked(),
                "background_method": (getattr(self.partaker_live_dead_setup, "background_method", "cell_exclusion_median") if self.cell_background_enabled()
                                      else "t0_gaussian" if self.gaus_back_corr.isChecked() else "none"),
                "background_margin_pixels": getattr(self.partaker_live_dead_setup, "background_margin_pixels", 15),
                "background_upper_cutoffs": getattr(self.partaker_live_dead_setup, "background_upper_cutoffs", ()),
                "background_include_cells": getattr(self.partaker_live_dead_setup, "background_include_cells", True),
                "background_signal_thresholds": getattr(self.partaker_live_dead_setup, "background_signal_thresholds", ()),
                "background_exclusion_rule": ("Manual rectangles only; finite pixels sampled without cell/threshold exclusion"
                                              if getattr(setup, "background_method", "") == "manual_samples_mean" else
                                              "Union of selected raw-channel threshold masks and optional cell footprints, expanded by the margin"),
                "background_samples": getattr(self.partaker_live_dead_setup, "background_samples", ()),
                "background_sample_definition": "Exactly three non-overlapping integer rectangles per frame/channel; x,y,width,height; upper bounds exclusive; sample finite original intensities before smoothing; subtract pooled mean",
                "background_sample_disagreement_rule": "Warn if (max sample mean - min sample mean) / abs(pooled mean) exceeds 0.20; warning does not block confirmation",
                "background_frames": [],
                "background_reference_time": None if self.cell_background_enabled() else 0,
                "background_sigma": None if self.cell_background_enabled() else self.LIVE_BACKGROUND_SIGMA,
                "morphology_enabled": self.close_dialate.isChecked(),
                "threshold_rule": "foreground = smoothed_uint8 >= threshold; optional morphology follows",
                "accepted_setup": setup.to_dict() if setup is not None else None,
            },
            "scope": {
                "positions": selected_positions, "time_start": time_start, "time_end": time_end,
                "skipped_times": sorted(skipped_frames),
                "frame_zero_dropped": self.drop_frame_zero_for_current_run,
                "capture_interval_value": self.frame_interval_value.value(),
                "capture_interval_unit": self.time_unit_combo.currentText(),
                "requested_frames": [{"time": int(t), "position": int(p)} for t, p, _l, _d in frames],
            },
            "measurements": {
                "raw_intensity": "ImageData.get() after crop/registration, before background correction, smoothing or uint8 conversion",
                "corrected_intensity": "Raw minus per-frame/channel pooled manual-sample mean (or legacy cell-free median); negative values retained",
                "whole_cell": "All finite fluorescence pixels in each cell footprint, independently of fluorescence threshold",
                "positive_region": "Cell label intersected with the final fluorescence-positive mask",
                "nonfinite_values": "Excluded; valid pixel counts and intensity statuses are exported",
                "empty_positive_region": "Mean and sum are zero to match existing overlap-based metrics; status explicitly marks empty regions",
                "ratio": "dead whole-cell mean divided by live whole-cell mean; nonpositive denominator or unavailable intensities yield null",
                "overlap_fraction": "Intersection divided by its named denominator; zero denominators yield null",
            },
            "save_fluorescence_positive_masks": self.save_fluorescence_masks_checkbox.isChecked(),
            "mask_format": {"version": 1, "bitorder": "little", "flatten_order": "C",
                            "cell_labels": "integer label array, zero background; IDs preserved"},
            "csv_columns": self.individual_cell_fieldnames(),
            "completed_frames": [], "failed_frames": [], "individual_cell_count": 0,
            "analysis_source_sha256": source_code_hash,
        }
        return self._export_json_value(provenance)

    def initialize_individual_exports(self, **scope):
        self.individual_export_context = self.build_analysis_provenance(**scope)
        self.individual_export_error = None
        output_dir = self.get_live_dead_output_root()
        self.individual_export_context["mask_directory"] = (
            f"fluorescence_masks/{self.individual_export_context['run_id']}"
            if self.individual_export_context["save_fluorescence_positive_masks"] else None
        )
        # ROI masks are context, not a replacement for the actual saved cell labels.
        from nd2_analyzer.data.appstate import ApplicationState
        from nd2_analyzer.data.image_data import ImageData
        appstate = ApplicationState.get_instance()
        cache = getattr(ImageData.get_instance(), "segmentation_cache", None)
        rois = {}
        for name, roi in (("application_roi", getattr(appstate, "roi_mask", None)),
                          ("segmentation_roi", getattr(cache, "binary_mask", None))):
            if roi is not None:
                roi = np.asarray(roi, dtype=bool)
                rois[f"{name}_shape"] = np.asarray(roi.shape, dtype=np.int64)
                rois[f"{name}_packed"] = np.packbits(roi.ravel(), bitorder="little")
        if rois:
            roi_path = output_dir / f"roi_{self.individual_export_context['run_id']}.npz"
            np.savez_compressed(roi_path, **rois)
            self.individual_export_context["roi_masks_path"] = roi_path.name
        csv_path = output_dir / "individual_cell_metrics.csv"
        temporary = csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.individual_cell_fieldnames()).writeheader()
        os.replace(temporary, csv_path)
        self.write_analysis_provenance()

    def write_analysis_provenance(self):
        if self.individual_export_context is None:
            return
        path = self.get_live_dead_output_root() / "analysis_provenance.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._export_json_value(self.individual_export_context),
                                        indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
        self.write_background_samples_csv()

    @staticmethod
    def _individual_intensity_values(values):
        values = np.asarray(values)
        valid = values[np.isfinite(values)]
        if not values.size:
            return 0.0, 0.0, 0, "empty_positive_region"
        if not valid.size:
            return None, None, 0, "no_finite_values"
        return (float(np.mean(valid, dtype=np.float64)), float(np.sum(valid, dtype=np.float64)),
                int(valid.size), "ok" if valid.size == values.size else "partial_nonfinite_values")

    def compute_individual_cell_metrics(self, *, time, position, cell_labels, channel_data):
        """Yield per-cell records using bounding-box crops, without rebuilding masks."""
        context = self.individual_export_context
        labels = np.asarray(cell_labels)
        if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
            raise ValueError("Individual-cell exports require nonnegative 2D integer labels.")
        for name, data in channel_data.items():
            if np.shape(data["raw"]) != labels.shape or np.shape(data["mask"]) != labels.shape:
                raise ValueError(f"Cell labels and {name} frames must have matching shapes.")
        calibration = context["pixel_calibration"]
        for region in regionprops(labels):
            y0, x0, y1, x1 = region.bbox
            cell = labels[y0:y1, x0:x1] == region.label
            area = int(np.count_nonzero(cell))
            row = {
                "run_id": context["run_id"], "dataset_id": context["dataset_id"],
                "sample_id": context["sample_id"], "treatment": context["treatment"],
                "analysis_mode": context["analysis_mode"], "position": int(position),
                "time": int(time), "time_hours": self.get_time_hours(time),
                "cell_id": int(region.label), "cell_channel": context["channel_assignments"]["cell"],
                "cell_area_pixels": area,
                "cell_area_um2": (area * calibration["x_um"] * calibration["y_um"]
                                  if calibration["x_um"] is not None else None),
                "centroid_x": float(region.centroid[1]), "centroid_y": float(region.centroid[0]),
                "bbox_x_min": int(x0), "bbox_y_min": int(y0),
                "bbox_x_max": int(x1), "bbox_y_max": int(y1),
            }
            positive_masks = {}
            for name, data in channel_data.items():
                raw = np.asarray(data["raw"])[y0:y1, x0:x1]
                positive = cell & np.asarray(data["mask"], dtype=bool)[y0:y1, x0:x1]
                positive_masks[name] = positive
                whole_mean, whole_sum, whole_count, whole_status = self._individual_intensity_values(raw[cell])
                mean, total, count, status = self._individual_intensity_values(raw[positive])
                row.update({
                    f"{name}_channel": int(data["channel"]),
                    f"{name}_whole_cell_mean_intensity": whole_mean,
                    f"{name}_whole_cell_integrated_intensity": whole_sum,
                    f"{name}_whole_cell_valid_pixels": whole_count,
                    f"{name}_whole_cell_intensity_status": whole_status,
                    f"{name}_positive_area_pixels": int(positive.sum()),
                    f"{name}_positive_area_fraction": float(positive.sum() / area),
                    f"{name}_positive_valid_pixels": count,
                    f"{name}_positive_mean_intensity": mean,
                    f"{name}_positive_integrated_intensity": total,
                    f"{name}_positive_intensity_status": status,
                })
                corrected = np.asarray(data.get("corrected", data["raw"]))[y0:y1, x0:x1]
                for scope, pixels in (("whole_cell", cell), ("positive", positive)):
                    cmean, csum, ccount, cstatus = self._individual_intensity_values(corrected[pixels])
                    row.update({f"{name}_corrected_{scope}_mean_intensity": cmean,
                                f"{name}_corrected_{scope}_integrated_intensity": csum,
                                f"{name}_corrected_{scope}_valid_pixels": ccount,
                                f"{name}_corrected_{scope}_intensity_status": cstatus})
                background = data.get("background")
                for key in ("median", "eligible_pixels", "eligible_fraction", "mad", "status", "mean", "sd", "method"):
                    row[f"{name}_background_{key}"] = background.get(key) if background else None
            if not self.is_cell_viability_run():
                live, dead = positive_masks["live"], positive_masks["dead"]
                both, union = int((live & dead).sum()), int((live | dead).sum())
                live_mean = row["live_whole_cell_mean_intensity"]
                dead_mean = row["dead_whole_cell_mean_intensity"]
                ratio, ratio_status = None, "ok"
                if live_mean is None or dead_mean is None:
                    ratio_status = "missing_or_nonfinite_intensity"
                elif live_mean <= 0:
                    ratio_status = "nonpositive_denominator"
                else:
                    ratio = dead_mean / live_mean
                    if not np.isfinite(ratio):
                        ratio, ratio_status = None, "nonfinite_ratio"
                row.update({
                    "live_dead_intersection_pixels": both, "live_dead_union_pixels": union,
                    "live_dead_intersection_over_union": both / union if union else None,
                    "live_positive_also_dead_fraction": both / int(live.sum()) if live.any() else None,
                    "dead_positive_also_live_fraction": both / int(dead.sum()) if dead.any() else None,
                    "dead_to_live_whole_cell_mean_ratio": ratio, "ratio_status": ratio_status,
                })
                numerator = row["dead_corrected_whole_cell_mean_intensity"]
                denominator = row["live_corrected_whole_cell_mean_intensity"]
                corrected_ratio = None
                corrected_status = "missing_or_nonfinite_intensity"
                if numerator is not None and denominator is not None:
                    if denominator <= 0:
                        corrected_status = "nonpositive_denominator"
                    else:
                        corrected_ratio = numerator / denominator
                        corrected_status = "ok" if np.isfinite(corrected_ratio) else "nonfinite_ratio"
                        if corrected_status != "ok":
                            corrected_ratio = None
                row.update(corrected_dead_to_live_whole_cell_mean_ratio=corrected_ratio,
                           corrected_ratio_status=corrected_status)
            yield row

    def append_individual_cell_metrics(self, rows):
        """Stream frame records to disk; roll back a failed append to its original size."""
        path = self.get_live_dead_output_root() / "individual_cell_metrics.csv"
        size_before = path.stat().st_size
        count = 0
        try:
            with path.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.individual_export_context["csv_columns"])
                for row in rows:
                    writer.writerow(row)
                    count += 1
        except Exception:
            with path.open("r+b") as stream:
                stream.truncate(size_before)
            raise
        self.individual_export_context["individual_cell_count"] += count

    def save_fluorescence_positive_masks(self, *, time, position, cell_labels, channel_data):
        context = self.individual_export_context
        if not context["save_fluorescence_positive_masks"]:
            return
        labels = np.asarray(cell_labels)
        largest = int(labels.max()) if labels.size else 0
        dtype = np.uint16 if largest <= np.iinfo(np.uint16).max else np.uint32
        if largest > np.iinfo(np.uint32).max:
            dtype = np.uint64
        payload = {
            "format_version": np.asarray(1, dtype=np.uint8), "run_id": np.asarray(context["run_id"]),
            "dataset_id": np.asarray(context["dataset_id"]), "time": np.asarray(time),
            "position": np.asarray(position), "mask_shape": np.asarray(labels.shape, dtype=np.int64),
            "bitorder": np.asarray("little"), "flatten_order": np.asarray("C"),
            "cell_channel": np.asarray(context["channel_assignments"]["cell"]),
            "cell_labels": labels.astype(dtype, copy=False),
            "channel_names": np.asarray(list(channel_data)),
        }
        for name, data in channel_data.items():
            if data.get("background") and data["background"].get("method") == "manual_samples_mean":
                payload[f"{name}_background_sample_rectangles_xywh"] = np.asarray(
                    [(r["x"], r["y"], r["width"], r["height"]) for r in data["background"]["samples"]], dtype=np.int64)
            payload[f"{name}_channel"] = np.asarray(data["channel"])
            payload[f"{name}_mask_packed"] = np.packbits(
                np.asarray(data["mask"], dtype=bool).ravel(order="C"), bitorder="little")
        first = next(iter(channel_data.values()), None)
        if first is not None and first.get("background_excluded") is not None:
            payload["background_excluded_mask_packed"] = np.packbits(
                np.asarray(first["background_excluded"], dtype=bool).ravel(order="C"), bitorder="little")
        folder = self.get_live_dead_output_root() / context["mask_directory"]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"pos{position}_t{time}.npz"
        temporary = path.with_suffix(".npz.tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **payload)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def export_individual_cell_frame(self, **frame):
        try:
            self.save_fluorescence_positive_masks(**frame)
            rows = self.compute_individual_cell_metrics(**frame)
            self.append_individual_cell_metrics(rows)
        except Exception as error:
            self.individual_export_error = str(error)
            raise RuntimeError(f"Individual-cell export failed: {error}") from error

    def record_individual_frame_failure(self, time, position, error):
        self.individual_export_context["failed_frames"].append(
            {"time": int(time), "position": int(position), "error": str(error)})
        try:
            self.write_analysis_provenance()
        except Exception as write_error:
            self.individual_export_error = str(write_error)

    def individual_export_status(self):
        if self.individual_export_error:
            return "failed"
        if self.cancel_requested:
            return "cancelled"
        if self.individual_export_context and self.individual_export_context["failed_frames"]:
            return "completed_with_errors"
        return "complete"

    def finish_individual_exports(self, output_error=None):
        if self.individual_export_context is None:
            return
        if output_error is not None:
            self.individual_export_error = output_error
        context = self.individual_export_context
        context["status"] = self.individual_export_status()
        context["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        context["export_error"] = self.individual_export_error
        context["unfinished_frames"] = [{"time": int(t), "position": int(p)}
                                        for t, p, _l, _d in self.queue]
        try:
            self.write_analysis_provenance()
        except Exception as error:
            self.individual_export_error = str(error)

    def process_live_dead_frame(
            self,
            *,
            live_frame: np.ndarray,
            dead_frame: np.ndarray | None,
            time: int,
            position: int,
            live_channel: int,
            dead_channel: int | None,
    ) -> dict:
        """Process paired Live and Dead frames with one shared fixed threshold."""
        live_image = np.asarray(live_frame, dtype=np.float32)
        inputs = [("cell_viability" if self.is_cell_viability_run() else "live",
                   live_image, live_channel)]
        if not self.is_cell_viability_run():
            inputs.append(("dead", np.asarray(dead_frame, dtype=np.float32), dead_channel))
        from nd2_analyzer.data.image_data import ImageData
        image_data = ImageData.get_instance()
        cell_channel = self.get_selected_cell_channel()
        cell_image = np.asarray(
            image_data.get(time, position, cell_channel),
            dtype=np.float32,
        )
        cell_labels = None
        cell_metadata = {}

        if self.cell_view_combo.currentText() == "Phase Contrast":
            cell_mask, cell_labels = self.get_partaker_cell_mask_for_frame(
                time=time,
                position=position,
            )
            cell_thresholding_image = cell_image
            cell_metadata["cell_segmentation_source"] = "partaker_segmented_mask"
        else:
            (
                cell_mask,
                cell_threshold,
                cell_polarity,
                cell_thresholding_image,
            ) = self.make_fluorescence_cell_mask(cell_image)
            cell_labels = label_connected_components(cell_mask)
            cell_metadata.update({
                "cell_segmentation_source": "fixed_fluorescence_thresholding",
                "cell_threshold_used": float(cell_threshold),
                "cell_polarity_used": cell_polarity,
            })

        cell_mask = np.asarray(cell_mask, dtype=bool)
        distance = None
        if self.cell_background_enabled() and getattr(self.partaker_live_dead_setup, "background_method", "cell_exclusion_median") != "manual_samples_mean":
            setup = self.partaker_live_dead_setup
            distance = self.background_exclusion_distance(
                cell_labels, {int(channel): image for _name, image, channel in inputs},
                getattr(setup, "background_signal_thresholds", ()),
                getattr(setup, "background_include_cells", True))
        channel_data = {}
        for name, image, channel in inputs:
            background = None
            if self.cell_background_enabled():
                setup = self.partaker_live_dead_setup
                cutoff = dict(setup.background_upper_cutoffs).get(int(channel))
                if getattr(setup, "background_method", "cell_exclusion_median") == "manual_samples_mean":
                    background = self.estimate_manual_background(image, self.manual_background_rectangles(time, position, channel))
                    for sample in background["samples"]:
                        x, y, w, h = (sample[k] for k in ("x", "y", "width", "height"))
                        sample["segmented_cell_pixels"] = int(np.count_nonzero(cell_mask[y:y+h, x:x+w]))
                    if any(sample["segmented_cell_pixels"] for sample in background["samples"]):
                        background["status"] = "samples_overlap_segmented_cells" + (";" + background["status"] if background["status"] != "ok" else "")
                else:
                    background = self.estimate_cell_free_background(image, distance, setup.background_margin_pixels, cutoff)
                corrected = image - np.float32(background.get("value", background["median"]))
            else:
                corrected = self.apply_live_dead_reference_background(image, position, channel)
            processing_image = self.convert_channel_image_to_uint8_for_processing(
                corrected
            )
            smoothed, channel_mask, segmentation = self.segment_fixed_channel(
                gaussian_corrected=processing_image,
                smoothing_sigma=self.partaker_live_dead_smoothing_sigma,
            )
            channel_data[name] = {
                "raw": image,
                "display": corrected,
                "corrected": corrected if self.cell_background_enabled() else image,
                "background": background,
                "background_excluded": (distance <= setup.background_margin_pixels if background and distance is not None else None),
                "processing": processing_image,
                "smoothed": smoothed,
                "mask": channel_mask & np.isfinite(image),
                "segmentation": segmentation,
                "channel": int(channel),
            }

        metrics = {
            "shared_threshold_used": float(self.partaker_live_dead_threshold),
            "shared_threshold_used_raw": (None if self.cell_background_enabled() else
                                          self.live_dead_processed_value_to_raw(self.partaker_live_dead_threshold)),
            "shared_threshold_used_corrected": (self.live_dead_processed_value_to_raw(self.partaker_live_dead_threshold)
                                                if self.cell_background_enabled() else None),
            "shared_thresholding_type": "fixed_global_uint8",
            "shared_gaussian_sigma": float(self.partaker_live_dead_smoothing_sigma),
            f"{self.output_prefix()}_processing_black_point_raw": float(
                self.live_dead_processing_black_point_raw
            ),
            f"{self.output_prefix()}_processing_white_point_raw": float(
                self.live_dead_processing_white_point_raw
            ),
            f"{self.output_prefix()}_processing_window_source": str(
                self.live_dead_processing_window_source
            ),
            f"{self.output_prefix()}_processing_intensity_space": (
                getattr(self.partaker_live_dead_setup, "background_method", "cell_exclusion_median") + "_corrected" if self.cell_background_enabled()
                else "t0_gaussian_background_corrected" if self.gaus_back_corr.isChecked()
                else "raw"
            ),
            "cell_area_pixels": int(np.count_nonzero(cell_mask)),
            **cell_metadata,
        }

        for name in channel_data:
            values = self.compute_channel_metrics(
                raw_image=channel_data[name]["raw"],
                channel_mask=channel_data[name]["mask"],
                cell_mask=cell_mask,
                cell_labels=cell_labels,
            )
            metrics.update({f"{name}_{key}": value for key, value in values.items()})
            corrected_values = self.compute_channel_metrics(
                raw_image=channel_data[name]["corrected"], channel_mask=channel_data[name]["mask"],
                cell_mask=cell_mask, cell_labels=cell_labels)
            for key in ("mean_intensity", "integrated_intensity", "mean_cell_intensity"):
                metrics[f"{name}_corrected_{key}"] = corrected_values[key]
            background = channel_data[name]["background"]
            metrics[f"{name}_background_method"] = (background.get("method", "cell_exclusion_median") if background else
                                                        "t0_gaussian" if self.gaus_back_corr.isChecked() else "none")
            for key in ("median", "eligible_pixels", "eligible_fraction", "mad", "regional_median_range", "status", "margin_pixels", "upper_cutoff"):
                metrics[f"{name}_background_{key}"] = background[key] if background else None
            if background and background.get("method") == "manual_samples_mean":
                background.update(self.save_background_sample_overlay(name, channel_data[name], time, position))
                metrics[f"{name}_background_mean"] = background["mean"]
                metrics[f"{name}_background_sd"] = background["sd"]
                metrics[f"{name}_background_sample_mean_relative_range"] = background["sample_mean_relative_range"]
            if background and self.individual_export_context is not None:
                self.individual_export_context["processing"]["background_frames"].append(
                    {"time": int(time), "position": int(position), "channel": int(channel_data[name]["channel"]), **background})
            metrics[f"{name}_processing_min"] = int(
                channel_data[name]["processing"].min()
            )
            metrics[f"{name}_processing_max"] = int(
                channel_data[name]["processing"].max()
            )

        if not self.is_cell_viability_run():
            metrics.update(self.calculate_live_dead_overlap_metrics(
                live_mask=channel_data["live"]["mask"],
                dead_mask=channel_data["dead"]["mask"],
                cell_mask=cell_mask,
            ))
            metrics.update(self.calculate_whole_frame_comparison_metrics(
                live_mean_intensity=metrics["live_mean_intensity"],
                dead_mean_intensity=metrics["dead_mean_intensity"],
                live_integrated_intensity=metrics["live_integrated_intensity"],
                dead_integrated_intensity=metrics["dead_integrated_intensity"],
            ))

        if not self.is_cell_viability_run():
            comparisons = self.calculate_whole_frame_comparison_metrics(
                live_mean_intensity=metrics["live_corrected_mean_intensity"],
                dead_mean_intensity=metrics["dead_corrected_mean_intensity"],
                live_integrated_intensity=metrics["live_corrected_integrated_intensity"],
                dead_integrated_intensity=metrics["dead_corrected_integrated_intensity"])
            metrics.update({key.replace("_live_", "_corrected_live_", 1)
                            if "_live_" in key else key.replace("_dead_", "_corrected_dead_", 1): value
                            for key, value in comparisons.items()})

        output_dir = self.get_live_dead_output_root()
        if self.individual_export_context is not None:
            self.export_individual_cell_frame(
                time=time, position=position, cell_labels=cell_labels,
                channel_data=channel_data,
            )
        export_visuals = self.should_export_visuals_for_position(position)
        if export_visuals:
            base = f"pos{position}_t{time}"
            overlay_row = {"time": int(time), "position": int(position)}
            for name, display_name, color in self.active_channel_specs():
                data = channel_data[name]
                mask_dir = output_dir / f"{name}_masks"
                overlay_dir = output_dir / f"{name}_overlays"
                cell_dir = output_dir / f"{name}_cell_overlap"
                for directory in (mask_dir, overlay_dir, cell_dir):
                    directory.mkdir(parents=True, exist_ok=True)
                plt.imsave(mask_dir / f"{base}_{name}_C{data['channel']}.png",
                           data["mask"], cmap="gray")
                overlay_path = overlay_dir / f"{base}_{name}.png"
                self.save_channel_overlay(data["display"], data["mask"],
                                          display_name, color, overlay_path)
                self.save_channel_cell_overlap_overlay(
                    cell_image, cell_mask, data["mask"], display_name, color,
                    cell_dir / f"{base}_{name}_cell.png")
                overlay_row[f"{name}_overlay_path"] = str(overlay_path)
            if self.is_cell_viability_run():
                self.save_cell_viability_images(output_dir, base, cell_image,
                                                cell_labels, channel_data["cell_viability"])
            else:
                self.save_live_dead_overlap_images(
                    output_dir=output_dir,
                    time=time,
                    position=position,
                    phase_contrast_image=cell_image,
                    cell_labels=cell_labels,
                    live_raw=channel_data["live"]["corrected"],
                    dead_raw=channel_data["dead"]["corrected"],
                    live_processed=channel_data["live"]["processing"],
                    dead_processed=channel_data["dead"]["processing"],
                    live_mask=channel_data["live"]["mask"],
                    dead_mask=channel_data["dead"]["mask"],
                    overlap_percent=metrics["live_dead_overlap_percent"],
                    cell_overlap_percent=(
                        metrics["cell_live_dead_overlap_percent"]
                    ),
                )

            self.overlay_gif_rows.append(overlay_row)

        for name in channel_data:
            data = channel_data[name]
            self.append_intensity_heatmap_rows(
                source=name,
                stage="before",
                image=data["raw"],
                time=time,
                position=position,
                channel=data["channel"],
            )
            if self.cell_background_enabled():
                self.append_intensity_heatmap_rows(
                    source=name, stage="corrected", image=data["corrected"],
                    time=time, position=position, channel=data["channel"],
                    intensity_min=self.live_dead_processing_black_point_raw,
                    intensity_max=self.live_dead_processing_white_point_raw)
            self.append_intensity_heatmap_rows(
                source=name,
                stage="after",
                image=data["smoothed"],
                time=time,
                position=position,
                channel=data["channel"],
                mask=data["mask"],
            )

        self.append_intensity_heatmap_rows(
            source="cell",
            stage="before",
            image=cell_image,
            time=time,
            position=position,
            channel=cell_channel,
        )
        self.append_intensity_heatmap_rows(
            source="cell",
            stage="after",
            image=cell_thresholding_image,
            time=time,
            position=position,
            channel=cell_channel,
            mask=cell_mask,
        )
        return metrics

    def save_cell_viability_images(self, output_dir, base, cell_image, cell_labels, data):
        """Single-channel counterparts of the combined and cell-based images."""
        strength = np.clip(data["processing"].astype(np.float32) / 255.0, 0, 1)
        cell_strength = np.zeros_like(strength)
        for cell_id in np.unique(cell_labels):
            if cell_id == 0:
                continue
            pixels = cell_labels == cell_id
            values = data.get("corrected", data["raw"])[pixels & data["mask"]]
            values = values[np.isfinite(values)]
            if np.sum(values, dtype=np.float64) > 0:
                cell_strength[pixels] = strength[pixels] * data["mask"][pixels]
        for name, signal in (("cell_viability", strength),
                             ("cell_based_cell_viability", cell_strength)):
            color = np.zeros((*signal.shape, 3), dtype=np.float32)
            color[..., 1] = signal
            folder = output_dir / "cell_viability_images" / (
                "cell_based" if name.startswith("cell_based") else "combined")
            folder.mkdir(parents=True, exist_ok=True)
            rendered = np.rint(color * 255).astype(np.uint8)
            blended = self.blend_intensity_color_on_phase_contrast(
                phase_contrast_image=cell_image, color_image=color, signal_strength=signal)
            if name.startswith("cell_based"):
                rendered = self.add_cell_outlines(rendered, cell_labels)
                blended = self.add_cell_outlines(blended, cell_labels)
            Image.fromarray(rendered).save(folder / f"{base}_{name}.png")
            Image.fromarray(blended).save(folder / f"{base}_{name}_on_phase_contrast.png")

    @staticmethod
    def _save_gif_frames(frames: list[Image.Image], output_path: Path, duration: int):
        if not frames:
            return None
        first, *remaining = frames
        try:
            first.save(
                output_path,
                save_all=True,
                append_images=remaining,
                duration=duration,
                loop=0,
                optimize=False,
            )
        finally:
            for frame in frames:
                frame.close()
        return output_path

    def export_overlay_gif_for_folder(
            self,
            *,
            image_key: str,
            output_path: Path,
            position: int,
    ):
        rows = sorted(
            (
                row for row in self.overlay_gif_rows
                if int(row["position"]) == int(position)
                and row.get(image_key)
            ),
            key=lambda row: int(row["time"]),
        )
        frames = []
        for row in rows:
            with Image.open(row[image_key]) as source:
                frame = source.convert("RGB")
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 90, 30), fill="white")
            draw.text((8, 7), f"T{row['time']}", fill="black")
            frames.append(frame)
        return self._save_gif_frames(
            frames,
            output_path,
            int(self.EXPORT_GIF_DELAY_MS),
        )

    def export_overlay_folder_gifs(self, output_dir: Path):
        if not self.overlay_gif_rows:
            return []
        positions = sorted({int(row["position"]) for row in self.overlay_gif_rows})
        specs = tuple((f"{name}_overlay_path", f"{name}_overlays", f"{name}_overlay")
                      for name, _label, _color in self.active_channel_specs())
        saved = []
        for position in positions:
            for key, folder, stem in specs:
                path = self.export_overlay_gif_for_folder(
                    image_key=key,
                    output_path=output_dir / folder / f"{stem}_pos{position}.gif",
                    position=position,
                )
                if path is not None:
                    saved.append(path)
        return saved

    def export_live_dead_gif(self, output_dir: Path):
        """Export paired Live/Dead overlays for the configured visual position."""
        rows = sorted(
            (
                row for row in self.overlay_gif_rows
                if int(row["position"]) == int(self.EXPORT_VISUAL_POSITION)
            ),
            key=lambda row: int(row["time"]),
        )
        frames = []
        for row in rows:
            if self.is_cell_viability_run():
                with Image.open(row["cell_viability_overlay_path"]) as source:
                    image = source.convert("RGB")
                canvas = Image.new("RGB", (image.width, image.height + 38), "white")
                canvas.paste(image, (0, 38))
                ImageDraw.Draw(canvas).text((8, 10), f"Cell Viability — T{row['time']}", fill="green")
                image.close()
                frames.append(canvas)
                continue
            with Image.open(row["live_overlay_path"]) as source:
                live = source.convert("RGB")
            with Image.open(row["dead_overlay_path"]) as source:
                dead = source.convert("RGB")
            height = min(live.height, dead.height)
            live.thumbnail((live.width, height))
            dead.thumbnail((dead.width, height))
            header = 38
            canvas = Image.new(
                "RGB",
                (live.width + dead.width + 8, height + header),
                "white",
            )
            canvas.paste(live, (0, header))
            canvas.paste(dead, (live.width + 8, header))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 10), f"Live — T{row['time']}", fill="green")
            draw.text((live.width + 16, 10), "Dead", fill="red")
            live.close()
            dead.close()
            frames.append(canvas)
        return self._save_gif_frames(
            frames,
            output_dir / f"{self.output_prefix()}_overlay_summary.gif",
            int(self.EXPORT_GIF_DELAY_MS),
        )

    # ------------------------------------------------------------------
    # Live-Dead whole-frame and cell-based ratio comparison analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_live_dead_ratio(numerator: float, denominator: float):
        """Return an explicit undefined value when no Dead denominator exists."""
        if not np.isfinite(denominator) or float(denominator) <= 0:
            return None
        return float(numerator) / float(denominator)

    @staticmethod
    def calculate_live_dead_overlap_metrics(
            *,
            live_mask: np.ndarray,
            dead_mask: np.ndarray,
            cell_mask: np.ndarray,
    ) -> dict:
        """Calculate symmetric Live/Dead mask overlap for a frame."""
        live_mask = np.asarray(live_mask, dtype=bool)
        dead_mask = np.asarray(dead_mask, dtype=bool)
        cell_mask = np.asarray(cell_mask, dtype=bool)
        if (
                live_mask.shape != dead_mask.shape
                or live_mask.shape != cell_mask.shape
        ):
            raise ValueError("Live, Dead, and cell masks must have matching shapes.")

        shared = live_mask & dead_mask
        union = live_mask | dead_mask
        overlap_pixels = int(np.count_nonzero(shared))
        union_pixels = int(np.count_nonzero(union))
        cell_overlap_pixels = int(np.count_nonzero(shared & cell_mask))
        cell_union_pixels = int(np.count_nonzero(union & cell_mask))

        return {
            "live_dead_overlap_pixels": overlap_pixels,
            "live_dead_union_pixels": union_pixels,
            "live_dead_overlap_percent": (
                100.0 * overlap_pixels / union_pixels
                if union_pixels else None
            ),
            "cell_live_dead_overlap_pixels": cell_overlap_pixels,
            "cell_live_dead_union_pixels": cell_union_pixels,
            "cell_live_dead_overlap_percent": (
                100.0 * cell_overlap_pixels / cell_union_pixels
                if cell_union_pixels else None
            ),
        }

    @classmethod
    def calculate_whole_frame_comparison_metrics(
            cls,
            *,
            live_mean_intensity: float,
            dead_mean_intensity: float,
            live_integrated_intensity: float,
            dead_integrated_intensity: float,
    ) -> dict:
        """Derive ratio and signed-difference outputs from existing metrics."""
        live_mean = float(live_mean_intensity)
        dead_mean = float(dead_mean_intensity)
        live_integrated = float(live_integrated_intensity)
        dead_integrated = float(dead_integrated_intensity)
        return {
            "mean_live_dead_intensity_ratio": cls._safe_live_dead_ratio(
                live_mean,
                dead_mean,
            ),
            "mean_dead_minus_live_intensity": dead_mean - live_mean,
            "whole_frame_live_dead_intensity_ratio": cls._safe_live_dead_ratio(
                live_integrated,
                dead_integrated,
            ),
            "whole_frame_dead_minus_live_intensity": (
                dead_integrated - live_integrated
            ),
        }

    @staticmethod
    def add_cell_outlines(image: np.ndarray, cell_labels: np.ndarray) -> np.ndarray:
        """Draw one-pixel white inner boundaries on a copy of a rendered image."""
        labels = np.asarray(cell_labels)
        outlined = np.array(image, dtype=np.uint8, copy=True)
        if labels.ndim != 2 or outlined.shape != (*labels.shape, 3):
            raise ValueError("Cell labels and rendered image must have matching shapes.")
        if not labels.size:
            return outlined
        boundaries = find_boundaries(labels, connectivity=1, mode="inner", background=0)
        # Include cell edges at the frame border, where no background pixel exists.
        boundaries[0, :] |= labels[0, :] > 0
        boundaries[-1, :] |= labels[-1, :] > 0
        boundaries[:, 0] |= labels[:, 0] > 0
        boundaries[:, -1] |= labels[:, -1] > 0
        outlined[boundaries] = (255, 255, 255)
        return outlined

    @staticmethod
    def normalize_phase_contrast_for_reference(image: np.ndarray) -> np.ndarray:
        """Match the main viewer's full-frame min/max display normalization."""
        image = np.asarray(image, dtype=np.float32)
        finite = image[np.isfinite(image)]
        if not finite.size:
            return np.zeros(image.shape, dtype=np.float32)

        low = float(np.min(finite))
        high = float(np.max(finite))
        if high <= low:
            return np.zeros(image.shape, dtype=np.float32)

        normalized = np.array(image, dtype=np.float32, copy=True)
        normalized -= np.float32(low)
        normalized /= np.float32(high - low)
        np.clip(normalized, 0.0, 1.0, out=normalized)
        np.nan_to_num(normalized, copy=False)
        return normalized

    @classmethod
    def blend_intensity_color_on_phase_contrast(
            cls,
            *,
            phase_contrast_image: np.ndarray,
            color_image: np.ndarray,
            signal_strength: np.ndarray,
    ) -> np.ndarray:
        """Overlay an intensity-weighted color image on normalized phase contrast."""
        phase = cls.normalize_phase_contrast_for_reference(phase_contrast_image)
        color_image = np.asarray(color_image, dtype=np.float32)
        strength = np.clip(
            np.asarray(signal_strength, dtype=np.float32),
            0.0,
            1.0,
        )
        if color_image.shape != (*phase.shape, 3):
            raise ValueError("Color and phase-contrast images must have matching shapes.")
        if strength.shape != phase.shape:
            raise ValueError("Signal strength must match the phase-contrast image.")

        hue = np.zeros_like(color_image, dtype=np.float32)
        np.divide(
            color_image,
            strength[..., np.newaxis],
            out=hue,
            where=strength[..., np.newaxis] > 0,
        )
        alpha = np.float32(0.40) * strength
        phase_rgb = np.repeat(phase[..., np.newaxis], 3, axis=2)
        blended = (
            phase_rgb * (1.0 - alpha[..., np.newaxis])
            + hue * alpha[..., np.newaxis]
        )
        np.clip(blended, 0.0, 1.0, out=blended)
        return np.rint(blended * 255.0).astype(np.uint8)

    @classmethod
    def make_live_dead_overlap_images(
            cls,
            *,
            live_raw: np.ndarray,
            dead_raw: np.ndarray,
            live_processed: np.ndarray,
            dead_processed: np.ndarray,
            live_mask: np.ndarray,
            dead_mask: np.ndarray,
            cell_labels: np.ndarray,
            phase_contrast_image: np.ndarray,
    ) -> dict:
        """Create whole-frame and cell-ID-based Live/Dead image products."""
        live_raw = np.asarray(live_raw, dtype=np.float32)
        dead_raw = np.asarray(dead_raw, dtype=np.float32)
        live = np.clip(
            np.asarray(live_processed, dtype=np.float32) / 255.0,
            0.0,
            1.0,
        )
        dead = np.clip(
            np.asarray(dead_processed, dtype=np.float32) / 255.0,
            0.0,
            1.0,
        )
        live_mask = np.asarray(live_mask, dtype=bool)
        dead_mask = np.asarray(dead_mask, dtype=bool)
        cell_labels = np.asarray(cell_labels)
        phase_contrast_image = np.asarray(phase_contrast_image)
        shapes = {
            live_raw.shape,
            dead_raw.shape,
            live.shape,
            dead.shape,
            live_mask.shape,
            dead_mask.shape,
            cell_labels.shape,
            phase_contrast_image.shape,
        }
        if len(shapes) != 1:
            raise ValueError(
                "Live, Dead, cell-label, and reference frames must match."
            )

        combined = np.zeros((*live.shape, 3), dtype=np.float32)
        combined[..., 0] = dead
        combined[..., 1] = live
        combined_strength = np.maximum(live, dead)

        cell_based = np.zeros_like(combined)
        cell_based_strength = np.zeros(live.shape, dtype=np.float32)
        cell_ids = np.unique(cell_labels)
        cell_ids = cell_ids[cell_ids > 0]
        for cell_id in cell_ids:
            cell_pixels = cell_labels == cell_id
            live_positive = cell_pixels & live_mask
            dead_positive = cell_pixels & dead_mask

            live_values = live_raw[live_positive]
            live_values = live_values[np.isfinite(live_values)]
            dead_values = dead_raw[dead_positive]
            dead_values = dead_values[np.isfinite(dead_values)]
            live_signal = max(
                float(np.sum(live_values, dtype=np.float64)),
                0.0,
            )
            dead_signal = max(
                float(np.sum(dead_values, dtype=np.float64)),
                0.0,
            )
            dominant_signal = max(live_signal, dead_signal)
            if dominant_signal <= 0:
                continue

            ratio_color = np.asarray([
                dead_signal / dominant_signal,
                live_signal / dominant_signal,
                0.0,
            ], dtype=np.float32)
            cell_strength = np.maximum(
                live[cell_pixels] * live_mask[cell_pixels],
                dead[cell_pixels] * dead_mask[cell_pixels],
            )
            cell_based_strength[cell_pixels] = cell_strength
            cell_based[cell_pixels] = (
                cell_strength[..., np.newaxis] * ratio_color
            )

        return {
            "combined": np.rint(combined * 255.0).astype(np.uint8),
            "combined_on_phase": cls.blend_intensity_color_on_phase_contrast(
                phase_contrast_image=phase_contrast_image,
                color_image=combined,
                signal_strength=combined_strength,
            ),
            "cell_based": cls.add_cell_outlines(
                np.rint(cell_based * 255.0).astype(np.uint8), cell_labels,
            ),
            "cell_based_on_phase": cls.add_cell_outlines(
                cls.blend_intensity_color_on_phase_contrast(
                    phase_contrast_image=phase_contrast_image,
                    color_image=cell_based,
                    signal_strength=cell_based_strength,
                ),
                cell_labels,
            ),
        }

    @staticmethod
    def add_overlap_percent_label(
            image: np.ndarray,
            overlap_percent: float | None,
    ) -> Image.Image:
        """Add a readable overlap percentage in the bottom-right corner."""
        labeled = Image.fromarray(
            np.asarray(image, dtype=np.uint8)
        ).convert("RGB")
        width, height = labeled.size
        value_text = (
            f"{float(overlap_percent):.2f}%"
            if overlap_percent is not None and np.isfinite(overlap_percent)
            else "undefined"
        )
        label = f"Live-Dead overlap: {value_text}"
        draw = ImageDraw.Draw(labeled, "RGBA")

        label_width = max(1, int(round(width / 6.0)))
        padding = max(4, int(round(label_width * 0.04)))
        available_text_width = max(1, label_width - (2 * padding))
        font_paths = (
            "DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        )

        def load_font(size: int):
            for font_path in font_paths:
                try:
                    return ImageFont.truetype(font_path, size)
                except OSError:
                    continue
            try:
                return ImageFont.load_default(size=size)
            except TypeError:
                return ImageFont.load_default()

        reference_font = load_font(100)
        reference_box = draw.textbbox(
            (0, 0),
            label,
            font=reference_font,
            stroke_width=1,
        )
        reference_width = max(1, reference_box[2] - reference_box[0])
        font_size = max(1, int(100 * available_text_width / reference_width))
        font = load_font(font_size)
        text_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        margin = max(6, int(round(width * 0.005)))
        box_right = width - margin
        box_left = max(0, box_right - label_width)
        box_bottom = height - margin
        box_top = max(0, box_bottom - text_height - (2 * padding))
        text_x = box_left + ((label_width - text_width) // 2) - text_box[0]
        text_y = box_top + padding - text_box[1]
        draw.rectangle(
            (box_left, box_top, box_right, box_bottom),
            fill=(0, 0, 0, 175),
        )
        draw.text(
            (text_x, text_y),
            label,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 255),
        )
        return labeled

    def save_live_dead_overlap_images(
            self,
            *,
            output_dir: Path,
            time: int,
            position: int,
            phase_contrast_image: np.ndarray,
            cell_labels: np.ndarray,
            live_raw: np.ndarray,
            dead_raw: np.ndarray,
            live_processed: np.ndarray,
            dead_processed: np.ndarray,
            live_mask: np.ndarray,
            dead_mask: np.ndarray,
            overlap_percent: float | None,
            cell_overlap_percent: float | None,
    ) -> list[Path]:
        """Save original overlap images plus labeled percentage variants."""
        overlap_root = output_dir / "image_overlaps"
        directories = {
            "combined": overlap_root / "overlap_images",
            "combined_on_phase": overlap_root / "overlap_phase_contrast",
            "cell_based": overlap_root / "cell_based_overlap_images",
            "cell_based_on_phase": overlap_root / "cell_based_phase_contrast",
        }
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        base = f"pos{int(position)}_t{int(time)}"
        paths = {
            "combined": directories["combined"] / f"{base}_live_dead_combined.png",
            "combined_on_phase": (
                directories["combined_on_phase"]
                / f"{base}_live_dead_on_phase_contrast.png"
            ),
            "combined_percent": (
                directories["combined"]
                / f"{base}_live_dead_combined_percent.png"
            ),
            "combined_on_phase_percent": (
                directories["combined_on_phase"]
                / f"{base}_live_dead_on_phase_contrast_percent.png"
            ),
            "cell_based": (
                directories["cell_based"] / f"{base}_cell_based_live_dead.png"
            ),
            "cell_based_percent": (
                directories["cell_based"]
                / f"{base}_cell_based_live_dead_percent.png"
            ),
            "cell_based_on_phase": (
                directories["cell_based_on_phase"]
                / f"{base}_cell_based_live_dead_on_phase_contrast.png"
            ),
            "cell_based_on_phase_percent": (
                directories["cell_based_on_phase"]
                / f"{base}_cell_based_live_dead_on_phase_contrast_percent.png"
            ),
        }
        images = self.make_live_dead_overlap_images(
            live_raw=live_raw,
            dead_raw=dead_raw,
            live_processed=live_processed,
            dead_processed=dead_processed,
            live_mask=live_mask,
            dead_mask=dead_mask,
            cell_labels=cell_labels,
            phase_contrast_image=phase_contrast_image,
        )
        for name in (
                "combined",
                "combined_on_phase",
                "cell_based",
                "cell_based_on_phase",
        ):
            Image.fromarray(images[name]).save(paths[name])
        self.add_overlap_percent_label(
            images["combined"],
            overlap_percent,
        ).save(paths["combined_percent"])
        self.add_overlap_percent_label(
            images["combined_on_phase"],
            overlap_percent,
        ).save(paths["combined_on_phase_percent"])
        self.add_overlap_percent_label(
            images["cell_based"],
            cell_overlap_percent,
        ).save(paths["cell_based_percent"])
        self.add_overlap_percent_label(
            images["cell_based_on_phase"],
            cell_overlap_percent,
        ).save(paths["cell_based_on_phase_percent"])
        return list(paths.values())

    @staticmethod
    def summarize_whole_frame_comparison(df: pl.DataFrame) -> pl.DataFrame:
        """Summarize ratio and signed difference by time across positions."""
        metric_columns = (
            "mean_live_dead_intensity_ratio",
            "mean_dead_minus_live_intensity",
            "whole_frame_live_dead_intensity_ratio",
            "whole_frame_dead_minus_live_intensity",
        )
        optional_metric_columns = (
            "live_dead_overlap_percent",
            "cell_live_dead_overlap_percent",
        )
        missing = [column for column in metric_columns if column not in df.columns]
        if missing:
            raise ValueError(
                "Missing whole-frame comparison columns: " + ", ".join(missing)
            )

        aggregations = [pl.col("time_hours").first().alias("time_hours")]
        summary_columns = metric_columns + tuple(
            column for column in optional_metric_columns
            if column in df.columns
        ) + tuple(column for column in df.columns
                  if "_corrected_" in column and (column.endswith("ratio") or column.endswith("intensity")))
        for column in summary_columns:
            aggregations.extend([
                pl.col(column).mean().alias(column),
                pl.col(column).std().fill_null(0).alias(f"std_{column}"),
            ])
        return df.group_by("time").agg(aggregations).sort("time")

    def save_live_dead_overlap_tables(self, output_dir: Path) -> list[Path]:
        """Write frame-level and time-summary whole-frame comparison tables."""
        if not hasattr(self, "live_dead_df") or self.live_dead_df.is_empty():
            return []

        overlap_root = output_dir / "image_overlaps"
        overlap_root.mkdir(parents=True, exist_ok=True)
        frame_columns = [
            "time",
            "time_hours",
            "position",
            "live_channel",
            "dead_channel",
            "live_mean_intensity",
            "dead_mean_intensity",
            "live_integrated_intensity",
            "dead_integrated_intensity",
            "mean_live_dead_intensity_ratio",
            "mean_dead_minus_live_intensity",
            "whole_frame_live_dead_intensity_ratio",
            "whole_frame_dead_minus_live_intensity",
            "live_dead_overlap_pixels",
            "live_dead_union_pixels",
            "live_dead_overlap_percent",
            "cell_live_dead_overlap_pixels",
            "cell_live_dead_union_pixels",
            "cell_live_dead_overlap_percent",
        ]
        frame_columns.extend(column for column in self.live_dead_df.columns
                             if "_corrected_" in column or "_background_" in column)
        available_columns = [
            column for column in frame_columns
            if column in self.live_dead_df.columns
        ]

        frame_path = overlap_root / "live_dead_overlap_metrics.csv"
        self.live_dead_df.select(available_columns).write_csv(frame_path)
        summary = self.summarize_whole_frame_comparison(self.live_dead_df)
        summary_path = overlap_root / "live_dead_ratio_summary.csv"
        summary.write_csv(summary_path)
        return [frame_path, summary_path]

    @staticmethod
    def get_live_dead_ratio_plot_spec(comparison_type: str) -> dict:
        specs = {
            "mean": {
                "ratio": "mean_live_dead_intensity_ratio",
                "title": "Mean Live-Dead Ratio Comparison",
                "error_title": "Mean Live-Dead Ratio Comparison Error Plot",
                "color": "forestgreen",
            },
            "integrated": {
                "ratio": "whole_frame_live_dead_intensity_ratio",
                "title": "Integrated Live-Dead Ratio Comparison",
                "error_title": (
                    "Integrated Live-Dead Ratio Comparison Error Plot"
                ),
                "color": "goldenrod",
            },
        }
        if comparison_type not in specs:
            raise ValueError(f"Unknown comparison type: {comparison_type}")
        return specs[comparison_type]

    def create_live_dead_ratio_comparison_figure(
            self,
            df: pl.DataFrame,
            comparison_type: str,
            figure=None,
    ):
        """Plot a shaded mean or integrated Live/Dead ratio reference."""
        df = self.background_corrected_plot_data(df)
        summary = self.summarize_whole_frame_comparison(df)
        if summary.is_empty():
            raise ValueError("No Live-Dead comparison measurements are available.")
        comparison_spec = self.get_live_dead_ratio_plot_spec(comparison_type)

        if figure is None:
            figure = plt.figure(figsize=(8, 5), constrained_layout=True)
        else:
            figure.clear()
        axis = figure.subplots(1, 1)
        time_hours = np.asarray(summary["time_hours"].to_numpy(), dtype=float)
        values = np.asarray(
            summary[comparison_spec["ratio"]].to_numpy(),
            dtype=float,
        )
        standard_deviations = np.asarray(
            summary[f"std_{comparison_spec['ratio']}"].to_numpy(),
            dtype=float,
        )
        axis.plot(
            time_hours,
            values,
            "-o",
            color=comparison_spec["color"],
            linewidth=2,
        )
        axis.fill_between(
            time_hours,
            values - standard_deviations,
            values + standard_deviations,
            color=comparison_spec["color"],
            alpha=0.18,
        )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1.1)
        axis.set_ylabel("Live / Dead Intensity Ratio")
        axis.set_xlabel("Time (hours)")
        axis.set_title(comparison_spec["title"] + (" (background corrected)" if self.cell_background_enabled() else ""), fontsize=14, fontweight="bold")
        axis.grid(alpha=0.2)
        return figure

    def create_live_dead_ratio_error_figure(
            self,
            df: pl.DataFrame,
            comparison_type: str,
            figure=None,
    ):
        """Plot per-timepoint mean +/- position SD with ratio labels."""
        df = self.background_corrected_plot_data(df)
        summary = self.summarize_whole_frame_comparison(df)
        if summary.is_empty():
            raise ValueError("No Live-Dead comparison measurements are available.")
        comparison_spec = self.get_live_dead_ratio_plot_spec(comparison_type)

        if figure is None:
            figure = plt.figure(figsize=(8, 5), constrained_layout=True)
        else:
            figure.clear()
        axis = figure.subplots(1, 1)
        time_hours = np.asarray(summary["time_hours"].to_numpy(), dtype=float)
        values = np.asarray(
            summary[comparison_spec["ratio"]].to_numpy(),
            dtype=float,
        )
        standard_deviations = np.asarray(
            summary[f"std_{comparison_spec['ratio']}"].to_numpy(),
            dtype=float,
        )
        standard_deviations = np.nan_to_num(
            standard_deviations,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        finite = np.isfinite(values)
        axis.errorbar(
            time_hours[finite],
            values[finite],
            yerr=standard_deviations[finite],
            fmt="-o",
            color=comparison_spec["color"],
            ecolor="black",
            linewidth=2,
            elinewidth=1.5,
            capsize=5,
            capthick=1.5,
        )

        upper_value = (
            float(np.max(values[finite] + standard_deviations[finite]))
            if np.any(finite)
            else 1.0
        )
        value_span = max(upper_value, 1.0)
        label_offset = 0.035 * value_span
        for time_value, ratio, deviation in zip(
                time_hours,
                values,
                standard_deviations,
        ):
            if np.isfinite(ratio):
                label = f"Live:Dead = {ratio:.2f}:1"
                label_height = ratio + deviation + label_offset
            else:
                label = "Live:Dead = undefined"
                label_height = label_offset
            axis.text(
                time_value,
                label_height,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

        axis.axhline(1.0, color="black", linestyle="--", linewidth=1.1)
        axis.set_ylim(0.0, upper_value + (0.18 * value_span))
        axis.set_ylabel("Live / Dead Intensity Ratio")
        axis.set_xlabel("Time (hours)")
        axis.set_title(
            comparison_spec["error_title"] + (" (background corrected)" if self.cell_background_enabled() else ""),
            fontsize=14,
            fontweight="bold",
        )
        axis.grid(alpha=0.2)
        return figure

    def should_export_visuals_for_position(self, position: int) -> bool:
        if not self.EXPORT_VISUALS_ONLY_FOR_ONE_POSITION:
            return True
        return int(position) == int(self.EXPORT_VISUAL_POSITION)
