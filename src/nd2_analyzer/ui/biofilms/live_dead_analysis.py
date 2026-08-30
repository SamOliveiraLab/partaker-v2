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
from scipy.ndimage import gaussian_filter
from scipy.stats import ks_2samp
from skimage.morphology import disk, binary_closing, binary_dilation
from skimage.measure import label as label_connected_components
from PIL import Image, ImageDraw, ImageFont
import csv
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
        title_label = QLabel("Live/Dead Configuration")
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

        # Live-Dead Channel selection
        selection_layout.addWidget(QLabel("Live Channel:"))
        self.live_channel_combo = QComboBox()
        selection_layout.addWidget(self.live_channel_combo)

        selection_layout.addWidget(QLabel("Dead Channel:"))
        self.dead_channel_combo = QComboBox()
        self.dead_channel_combo.setToolTip(
            "Dead fluorescence channel analyzed alongside the Live channel."
        )
        selection_layout.addWidget(self.dead_channel_combo)

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
        self.metric_combo.addItems([
            "Mean Live-Dead Intensity",
            "Integrated Live-Dead Intensity",
            "Mean Live-Dead Intensity Error Plot",
            "Integrated Live-Dead Intensity Error Plot",
            "Live-Dead Fractional Area",
            "Mean Cell-Based Intensity",
        ])
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
            QMessageBox.warning(self, "No data", "Run Live-Dead analysis first.")
            return
        output_path = (
            self.get_live_dead_output_root()
            / "live_dead_processing_metrics.csv"
        )
        self.live_dead_df.write_csv(output_path)
        self.status_label.setText(f"Metrics saved to {output_path}")

    def export_plots(self):
        """Regenerate all six paired plots in the analysis output folder."""
        if not hasattr(self, "live_dead_df") or self.live_dead_df.is_empty():
            QMessageBox.warning(self, "No data", "Run Live-Dead analysis first.")
            return
        output_dir = self.get_live_dead_output_root()
        paths = self.save_all_summary_plots(output_dir)
        self.status_label.setText(
            f"Saved {len(paths)} plots to {output_dir}"
        )

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

        if not self.gaus_back_corr.isChecked():
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
                "Cached Gaussian Live-Dead background | "
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
            raise ValueError("No Live-Dead frames were supplied for window estimation.")

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
                "Could not estimate a finite Live-Dead processing intensity window."
            )

        black_point = float(np.median(black_candidates))
        white_point = float(np.percentile(
            white_candidates,
            self.LIVE_WINDOW_WHITE_ACROSS_FRAMES_PERCENTILE,
        ))

        if not np.isfinite(black_point) or not np.isfinite(white_point):
            raise ValueError("Estimated Live-Dead intensity window is not finite.")

        if white_point <= black_point:
            white_point = black_point + 1.0

        self.live_dead_processing_black_point_raw = black_point
        self.live_dead_processing_white_point_raw = white_point
        self.live_dead_processing_window_source = "automatic_run_level"

        print(
            "Live-Dead run-level processing window | "
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

        # Reuse the accepted raw-image window when no background correction is
        # selected. Background subtraction changes the intensity space, so its
        # shared processing window must be estimated after correction.
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
                "Using Partaker dialog Live-Dead processing window | "
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
            dead_channel: int,
    ):
        """Open Partaker fixed-threshold calibration and return its result."""
        from nd2_analyzer.data.image_data import ImageData
        from nd2_analyzer.ui.dialogs.live_dead_dialog import LiveDeadDialog

        image_data = ImageData.get_instance()
        if image_data is None or image_data.data is None:
            QMessageBox.warning(
                self,
                "No image data",
                "Load image data before configuring fixed Live-Dead thresholding."
            )
            return None

        dialog = LiveDeadDialog(
            image_data=image_data,
            live_channel=live_channel,
            dead_channel=dead_channel,
            selected_positions=selected_positions,
            time_start=t_start,
            time_end=t_end,
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
        self.partaker_live_dead_threshold = int(setup.threshold_uint8)
        self.partaker_live_dead_smoothing_sigma = float(setup.smoothing_sigma)
        self.partaker_drop_frame_zero = bool(setup.drop_frame_zero)
        self.drop_frame_zero_for_current_run = bool(setup.drop_frame_zero)
        self.time_start_spin.setValue(int(setup.time_start))
        self.time_end_spin.setValue(int(setup.time_end))
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
            "Partaker Live-Dead setup accepted: "
            f"threshold={self.partaker_live_dead_threshold}, "
            f"live=C{setup.live_channel}, dead=C{setup.dead_channel}, "
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
        self.live_dead_results = []
        self.reset_output_tracking()

        if self.is_segmenting:
            return

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
            return

        live_channel = self.live_channel_combo.currentIndex()
        dead_channel = self.dead_channel_combo.currentIndex()

        if live_channel == dead_channel:
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
                self.status_label.setText("Partaker Live-Dead setup cancelled")
                self.start_segmentation_btn.setEnabled(True)
                self.cancel_analysis_btn.setEnabled(False)
                return

            self.apply_partaker_live_dead_setup(setup)
            selected_positions = list(setup.positions)
            t_start = int(setup.time_start)
            t_end = int(setup.time_end)
            live_channel = int(setup.live_channel)
            dead_channel = int(setup.dead_channel)
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
                preprocessing_frame_keys.extend([
                    (t, p, live_channel),
                    (t, p, dead_channel),
                ])

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
                "Live-Dead preprocessing setup failed",
                f"Could not prepare the shared Live-Dead intensity window:\n{e}"
            )
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
        for c in range(c_max + 1):
            self.live_channel_combo.addItem(f"Channel {c}")
            self.dead_channel_combo.addItem(f"Channel {c}")
            self.cell_channel_combo.addItem(f"Channel {c}")
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

        raw_live = image_data.get(time, position, live_channel)
        raw_dead = image_data.get(time, position, dead_channel)

        try:
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

            "live_channel": live_channel,
            "dead_channel": dead_channel,
            "cell_channel": cell_channel,
            "analysis_method": self.get_analysis_method(),

            "cell_view_type": self.cell_view_combo.currentText(),

            "background_filtering_used": bool(self.gaus_back_corr.isChecked()),
            "live_dead_morphology_used": bool(self.close_dialate.isChecked()),
            "frame_zero_dropped": bool(
                self.drop_frame_zero_for_current_run
            ),
        }
        result_row.update(metrics)

        self.live_dead_results.append(result_row)

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
                output_dir / "live_dead_processing_metrics.csv"
            )

        self.save_processing_heatmaps(output_dir)
        gif_path = self.export_live_dead_gif(output_dir)
        overlay_gif_paths = self.export_overlay_folder_gifs(output_dir)
        summary_plot_paths = self.save_all_summary_plots(output_dir)

        self.live_dead_reference_backgrounds = {}
        shutil.rmtree(output_dir / "_gif_mask_cache", ignore_errors=True)
        gc.collect()

        self.is_segmenting = False
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
            f"Live-Dead analysis complete "
            f"({len(self.live_dead_results)} frames).{frame_zero_note} "
            f"Outputs saved to {output_dir}; "
            f"summary GIF saved: {gif_path is not None}; "
            f"overlay GIFs saved: {len(overlay_gif_paths)}; "
            f"graphs saved: {len(summary_plot_paths)}."
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
        missing = [column for column in metric_columns if column not in df.columns]
        if missing:
            raise ValueError(
                "Missing Live-Dead metric columns: " + ", ".join(missing)
            )

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
        }
        if analysis_type not in specs:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        return specs[analysis_type]

    def create_live_dead_metric_figure(
            self,
            stats: pl.DataFrame,
            analysis_type: str,
            figure=None,
            raw_df: pl.DataFrame | None = None,
    ):
        """Create one two-panel figure with Live above Dead."""
        spec = self.get_live_dead_metric_spec(analysis_type)
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

        axes = figure.subplots(2, 1)
        time_hours = stats["time_hours"].to_numpy()

        for axis, channel_name, color, column in (
            (axes[0], "Live", "green", spec["live"]),
            (axes[1], "Dead", "red", spec["dead"]),
        ):
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

        figure.suptitle(analysis_type, fontsize=14, fontweight="bold")
        return figure

    def create_live_dead_error_figure(
            self,
            df: pl.DataFrame,
            analysis_type: str,
            figure=None,
    ):
        """Create a two-bar mean +/- SD plot with a Live-vs-Dead KS p-value."""
        spec = self.get_live_dead_metric_spec(analysis_type)
        live_values = np.asarray(df[spec["live"]].to_numpy(), dtype=np.float64)
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
        axis.set_title(analysis_type, fontsize=14, fontweight="bold")

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

    def save_all_summary_plots(self, output_dir: Path):
        if not hasattr(self, "live_dead_df") or self.live_dead_df.is_empty():
            return []

        stats = self.compute_live_dead_stats(self.live_dead_df)
        stats.write_csv(output_dir / "live_dead_summary_values.csv")

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
            figure.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(figure)
            saved_paths.append(output_path)
        return saved_paths

    def on_plot_avg_sd(self):
        if not hasattr(self, "live_dead_df") or self.live_dead_df.is_empty():
            QMessageBox.warning(self, "No data", "Run Live-Dead analysis first.")
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
        self.population_figure.savefig(
            output_dir / spec["filename"],
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
        output_dir = Path("live_dead_analysis")
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
                "The Live-Dead run-level intensity window has not been initialized."
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

    def process_live_dead_frame(
            self,
            *,
            live_frame: np.ndarray,
            dead_frame: np.ndarray,
            time: int,
            position: int,
            live_channel: int,
            dead_channel: int,
    ) -> dict:
        """Process paired Live and Dead frames with one shared fixed threshold."""
        live_image = np.asarray(live_frame, dtype=np.float32)
        dead_image = np.asarray(dead_frame, dtype=np.float32)

        channel_data = {}
        for name, image, channel in (
            ("live", live_image, live_channel),
            ("dead", dead_image, dead_channel),
        ):
            corrected = self.apply_live_dead_reference_background(
                image,
                position,
                channel,
            )
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
                "processing": processing_image,
                "smoothed": smoothed,
                "mask": channel_mask,
                "segmentation": segmentation,
                "channel": int(channel),
            }

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
        metrics = {
            "shared_threshold_used": float(self.partaker_live_dead_threshold),
            "shared_threshold_used_raw": self.live_dead_processed_value_to_raw(
                self.partaker_live_dead_threshold
            ),
            "shared_thresholding_type": "fixed_global_uint8",
            "shared_gaussian_sigma": float(self.partaker_live_dead_smoothing_sigma),
            "live_dead_processing_black_point_raw": float(
                self.live_dead_processing_black_point_raw
            ),
            "live_dead_processing_white_point_raw": float(
                self.live_dead_processing_white_point_raw
            ),
            "live_dead_processing_window_source": str(
                self.live_dead_processing_window_source
            ),
            "live_dead_processing_intensity_space": (
                "t0_gaussian_background_corrected"
                if self.gaus_back_corr.isChecked()
                else "raw"
            ),
            "cell_area_pixels": int(np.count_nonzero(cell_mask)),
            **cell_metadata,
        }

        for name in ("live", "dead"):
            values = self.compute_channel_metrics(
                raw_image=channel_data[name]["raw"],
                channel_mask=channel_data[name]["mask"],
                cell_mask=cell_mask,
                cell_labels=cell_labels,
            )
            metrics.update({f"{name}_{key}": value for key, value in values.items()})
            metrics[f"{name}_processing_min"] = int(
                channel_data[name]["processing"].min()
            )
            metrics[f"{name}_processing_max"] = int(
                channel_data[name]["processing"].max()
            )

        output_dir = self.get_live_dead_output_root()
        export_visuals = self.should_export_visuals_for_position(position)
        live_overlay_path = None
        dead_overlay_path = None

        if export_visuals:
            directories = {
                "live_masks": output_dir / "live_masks",
                "dead_masks": output_dir / "dead_masks",
                "live_overlays": output_dir / "live_overlays",
                "dead_overlays": output_dir / "dead_overlays",
                "live_cell": output_dir / "live_cell_overlap",
                "dead_cell": output_dir / "dead_cell_overlap",
            }
            for directory in directories.values():
                directory.mkdir(parents=True, exist_ok=True)

            base = f"pos{position}_t{time}"
            plt.imsave(
                directories["live_masks"] / f"{base}_live_C{live_channel}.png",
                channel_data["live"]["mask"],
                cmap="gray",
            )
            plt.imsave(
                directories["dead_masks"] / f"{base}_dead_C{dead_channel}.png",
                channel_data["dead"]["mask"],
                cmap="gray",
            )

            live_overlay_path = directories["live_overlays"] / f"{base}_live.png"
            dead_overlay_path = directories["dead_overlays"] / f"{base}_dead.png"
            self.save_channel_overlay(
                channel_data["live"]["display"],
                channel_data["live"]["mask"],
                "Live",
                self.LIVE_OVERLAY_RGB,
                live_overlay_path,
            )
            self.save_channel_overlay(
                channel_data["dead"]["display"],
                channel_data["dead"]["mask"],
                "Dead",
                self.DEAD_OVERLAY_RGB,
                dead_overlay_path,
            )
            self.save_channel_cell_overlap_overlay(
                cell_image,
                cell_mask,
                channel_data["live"]["mask"],
                "Live",
                self.LIVE_OVERLAY_RGB,
                directories["live_cell"] / f"{base}_live_cell.png",
            )
            self.save_channel_cell_overlap_overlay(
                cell_image,
                cell_mask,
                channel_data["dead"]["mask"],
                "Dead",
                self.DEAD_OVERLAY_RGB,
                directories["dead_cell"] / f"{base}_dead_cell.png",
            )

            self.overlay_gif_rows.append({
                "time": int(time),
                "position": int(position),
                "live_overlay_path": str(live_overlay_path),
                "dead_overlay_path": str(dead_overlay_path),
            })

        for name in ("live", "dead"):
            data = channel_data[name]
            self.append_intensity_heatmap_rows(
                source=name,
                stage="before",
                image=data["raw"],
                time=time,
                position=position,
                channel=data["channel"],
            )
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
        specs = (
            ("live_overlay_path", "live_overlays", "live_overlay"),
            ("dead_overlay_path", "dead_overlays", "dead_overlay"),
        )
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
            output_dir / "live_dead_overlay_summary.gif",
            int(self.EXPORT_GIF_DELAY_MS),
        )

    def should_export_visuals_for_position(self, position: int) -> bool:
        if not self.EXPORT_VISUALS_ONLY_FOR_ONE_POSITION:
            return True
        return int(position) == int(self.EXPORT_VISUAL_POSITION)
