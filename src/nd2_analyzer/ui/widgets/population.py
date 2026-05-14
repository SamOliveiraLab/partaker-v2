import csv
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import polars as pl  # Import Polars
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QMessageBox,
)
import seaborn as sns

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pubsub import pub

# Assume MetricsService is a singleton with a .df attribute (Polars DataFrame)
from nd2_analyzer.analysis.metrics_service import MetricsService
from nd2_analyzer.data.experiment import Experiment

from nd2_analyzer.analysis.population import FluoAnalysisConfig, filter_data, create_sample_data, calculate_population_statistics, generate_component_step_functions, component_intervals

class PopulationWidget(QWidget):
    """
    Widget for plotting population-level fluorescence over time.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics_service = MetricsService()  # Singleton instance
        self.init_ui()
        self.experiment: Experiment = None

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Matplotlib figure
        self.population_figure = plt.figure(constrained_layout=True)
        self.population_canvas = FigureCanvas(self.population_figure)
        layout.addWidget(self.population_canvas)

        # Position selection
        pos_layout = QHBoxLayout()
        pos_layout.addWidget(QLabel("Select Positions:"))
        self.position_list = QListWidget()
        self.position_list.setSelectionMode(QListWidget.MultiSelection)
        pos_layout.addWidget(self.position_list)
        layout.addLayout(pos_layout)

        # Channel selection
        mcherry_ch_layout = QHBoxLayout()
        mcherry_ch_layout.addWidget(QLabel("mCherry Channel:"))
        self.mcherry_channel_combo = QComboBox()
        mcherry_ch_layout.addWidget(self.mcherry_channel_combo)
        layout.addLayout(mcherry_ch_layout)

        yfp_ch_layout = QHBoxLayout()
        yfp_ch_layout.addWidget(QLabel("YFP Channel:"))
        self.yfp_channel_combo = QComboBox()
        yfp_ch_layout.addWidget(self.yfp_channel_combo)
        layout.addLayout(yfp_ch_layout)

        # Metric selection
        metric_layout = QHBoxLayout()
        metric_layout.addWidget(QLabel("Metric:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItem("Mean Intensity")
        self.metric_combo.addItem("Integrated Intensity")
        self.metric_combo.addItem("Normalized Intensity")
        metric_layout.addWidget(self.metric_combo)
        layout.addLayout(metric_layout)

        # Time range selection
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Time Range:"))
        self.time_min_box = QSpinBox()
        self.time_max_box = QSpinBox()
        time_layout.addWidget(self.time_min_box)
        time_layout.addWidget(QLabel("to"))
        time_layout.addWidget(self.time_max_box)
        layout.addLayout(time_layout)

        # bottom buttons
        bottom_btn_layout = QHBoxLayout()

        # Plot button
        plot_btn = QPushButton("Plot Fluorescence")
        plot_btn.clicked.connect(self.on_plot_population_signal)
        bottom_btn_layout.addWidget(plot_btn)

        # Export DataFrame button
        export_btn = QPushButton("Export DataFrame to CSV")
        export_btn.clicked.connect(self.export_dataframe)
        bottom_btn_layout.addWidget(export_btn)

        # Plot fluor. across position
        avg_plot_btn = QPushButton("Plot Avg ± SD (All Positions)")
        avg_plot_btn.clicked.connect(self.on_plot_avg_sd)
        bottom_btn_layout.addWidget(avg_plot_btn)

        # Save current graph/plot button
        self.save_plot_button = QPushButton("Save Plot")
        self.save_plot_button.clicked.connect(self.save_population_plot)
        bottom_btn_layout.addWidget(self.save_plot_button)

        # Calculate RPU button (new)
        rpu_btn = QPushButton("Calculate RPU Reference Values")
        # rpu_btn.clicked.connect(self.calculate_rpu_values)
        bottom_btn_layout.addWidget(rpu_btn)

        layout.addLayout(bottom_btn_layout)

        self.setLayout(layout)

        # Listen for data loading to populate UI
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")
        pub.subscribe(self.on_experiment_loaded, "experiment_loaded")

    def export_dataframe(self):
        # Export the DataFrame to a CSV file
        df = self.metrics_service.df
        if not df.is_empty():
            df.write_csv("cell_metrics.csv")
            print("DataFrame exported to cell_metrics.csv")
        else:
            print("No data to export")

    def get_selected_positions(self):
        selected_positions = [
            int(item.text()) for item in self.position_list.selectedItems()
        ]
        return selected_positions

    def get_selected_time(self):
        return (self.time_min_box.value(), self.time_max_box.value())

    @staticmethod
    def _set_xticks_from_hours(ax, hours_series, step=10):
        if hours_series is None or len(hours_series) == 0:
            return
        arr = np.asarray(hours_series, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        tmax = int(np.nanmax(arr))
        if tmax >= 0:
            ax.set_xticks(range(0, tmax + 1, step))

    def _apply_experiment_to_config(self, analysis_cgf: FluoAnalysisConfig):
        if self.experiment is not None:
            analysis_cgf.time_interval = self.experiment.phc_interval
            analysis_cgf.fluorescence_factor = self.experiment.fluorescence_factor

    def on_image_data_loaded(self, image_data):
        # Populate positions and channels based on image_data shape
        shape = image_data.data.shape
        t_max, p_max, c_max = shape[0] - 1, shape[1] - 1, shape[2] - 1

        self.position_list.clear()
        for p in range(p_max + 1):
            item = QListWidgetItem(f"{p}")
            self.position_list.addItem(item)
        for i in range(self.position_list.count()):
            self.position_list.item(i).setSelected(True)

        # Include channel 0 for phase / single-channel TIFF mean intensity in segmented regions
        self.mcherry_channel_combo.clear()
        for c in range(0, c_max + 1):
            self.mcherry_channel_combo.addItem(str(c))

        self.yfp_channel_combo.clear()
        for c in range(0, c_max + 1):
            self.yfp_channel_combo.addItem(str(c))

        self.time_min_box.setRange(0, t_max)
        self.time_max_box.setRange(0, t_max)
        self.time_max_box.setValue(t_max)

        # Dummy plot for preview
        self.create_dummy_plot()

    def on_experiment_loaded(self, experiment):
        self.experiment = experiment

    def create_dummy_plot(self):
        self.population_figure.clear()

        # Configure based on selected values
        analysis_cgf = FluoAnalysisConfig()
        self._apply_experiment_to_config(analysis_cgf)
        analysis_cgf.selected_positions = self.get_selected_positions()
        analysis_cgf.time_range = self.get_selected_time()

        df = create_sample_data(analysis_cgf)

        # Perform data processing
        df = filter_data(df, analysis_cgf)
        df = calculate_population_statistics(df, analysis_cgf)

        component_intervals = {}
        try:
            if self.experiment and self.experiment.component_intervals:
                component_intervals = dict(self.experiment.component_intervals)
        except Exception:
            pass

        # Normalize with RPU
        text = self.mcherry_channel_combo.currentText()
        if not text.strip():
            print("No channel selected yet; skipping plot init")
            return
        mcherry_channel = int(text)
        text = self.yfp_channel_combo.currentText()
        if not text.strip():
            print("No channel selected yet; skipping plot init")
            return
        yfp_channel = int(text)

        mcherry_subdf = df.filter(pl.col('fluorescence_channel') == mcherry_channel).to_pandas()
        yfp_subdf = df.filter(pl.col('fluorescence_channel') == yfp_channel).to_pandas()

        # mcherry_subdf['mean_intensity'] = mcherry_subdf['mean_intensity'] / mcherry_rpu
        # mcherry_subdf['std_intensity'] = mcherry_subdf['std_intensity'] / mcherry_rpu
        # yfp_subdf['mean_intensity'] = yfp_subdf['mean_intensity'] / yfp_no_roi_rpu
        # yfp_subdf['std_intensity'] = yfp_subdf['std_intensity'] / yfp_no_roi_rpu

        # Plot graph
        fig = self.population_figure
        axs = fig.subplots(5, 1, gridspec_kw={'height_ratios': [3.5, 3.5, 1, 1, 1]})

        # For channel 0 (mCherry)
        ch0 = mcherry_subdf
        if not ch0.empty:
            sns.lineplot(data=ch0, x='time_hours', y='mean_intensity', ax=axs[0], color='red', label='mCherry')
            axs[0].fill_between(ch0['time_hours'],
                                ch0['mean_intensity'] - ch0['std_intensity'],
                                ch0['mean_intensity'] + ch0['std_intensity'],
                                color='red', alpha=0.3)
        axs[0].set_ylabel('mCherry')
        axs[0].set_xlabel('')
        self._set_xticks_from_hours(axs[0], ch0['time_hours'] if not ch0.empty else None)
        axs[0].legend().set_visible(False)  # Hide legend

        # For channel 1 (YFP)
        ch1 = yfp_subdf
        if not ch1.empty:
            sns.lineplot(data=ch1, x='time_hours', y='mean_intensity', ax=axs[1], color='goldenrod', label='YFP')
            axs[1].fill_between(ch1['time_hours'],
                                ch1['mean_intensity'] - ch1['std_intensity'],
                                ch1['mean_intensity'] + ch1['std_intensity'],
                                color='goldenrod', alpha=0.3)
        axs[1].set_ylabel('YFP')
        axs[1].set_xlabel('')
        self._set_xticks_from_hours(axs[1], ch1['time_hours'] if not ch1.empty else None)
        axs[1].legend().set_visible(False)  # Hide legend

        plt.tight_layout()

        # Medium change steps
        t_hours = np.sort(df['time_hours'].unique().to_numpy())
        comp_steps = generate_component_step_functions(component_intervals, t_hours)

        start_index = 2
        for idx, (comp, step) in enumerate(comp_steps.items()):
            axs[start_index + idx].step(t_hours, step, label=f'{comp}', where='post', color='black')
            axs[start_index + idx].set_ylabel(comp)
            self._set_xticks_from_hours(axs[start_index + idx], t_hours)
            axs[start_index + idx].legend().set_visible(False)  # Hide legend

        axs[-1].set_xlabel('Time (h)')

        # plt.savefig('1_9_iptg_on_p_all.pdf')
        # plt.show()

    def on_plot_population_signal(self):
        self.population_figure.clear()

        # Configure based on selected values
        analysis_cgf = FluoAnalysisConfig()
        self._apply_experiment_to_config(analysis_cgf)
        analysis_cgf.selected_positions = self.get_selected_positions()
        analysis_cgf.time_range = self.get_selected_time()

        # Current dataframe
        df = self.metrics_service.df
        print("---BEFORE FILTER---")
        print("mcherry", df.filter(pl.col("fluorescence_channel") == 1)
              .group_by("time")
              .agg([
            pl.mean("fluo_level").alias("mean"),
            pl.std("fluo_level").alias("std"),
            pl.min("fluo_level").alias("min"),
            pl.max("fluo_level").alias("max"),
        ]))
        print("yfp channel", df.filter(pl.col("fluorescence_channel") == 2)
              .group_by("time")
              .agg([
            pl.mean("fluo_level").alias("mean"),
            pl.std("fluo_level").alias("std"),
            pl.min("fluo_level").alias("min"),
            pl.max("fluo_level").alias("max"),
        ]))
        if df.is_empty():
            QMessageBox.warning(
                self,
                "No data",
                "No cell metrics are loaded. Run segmentation first.",
            )
            self.population_canvas.draw()
            return

        # Perform data processing
        df = filter_data(df, analysis_cgf)
        df = calculate_population_statistics(df, analysis_cgf)
        print("---AFTER FILTER---")
        print("mcherry", df.filter(pl.col("fluorescence_channel") == 1)
              .group_by("time")
              .agg([
            pl.mean("mean_intensity").alias("mean1"),
            pl.std("mean_intensity").alias("std1"),
            pl.min("mean_intensity").alias("min1"),
            pl.max("mean_intensity").alias("max1"),
        ]))
        print("yfp channel", df.filter(pl.col("fluorescence_channel") == 2)
        .group_by("time")
        .agg([
            pl.mean("mean_intensity").alias("mean2"),
            pl.std("mean_intensity").alias("std2"),
            pl.min("mean_intensity").alias("min2"),
            pl.max("mean_intensity").alias("max2"),
        ]))

        if df.is_empty():
            QMessageBox.warning(
                self,
                "No data to plot",
                "No rows in the selected time range with valid channel data "
                "(channel ≥ 0). For single-channel TIFFs, choose channel 0 and "
                "re-run segmentation so phase intensity is stored per cell.",
            )
            self.population_canvas.draw()
            return

        component_intervals = {}
        try:
            if self.experiment and self.experiment.component_intervals:
                component_intervals = dict(self.experiment.component_intervals)
        except Exception:
            pass

        # Normalize with RPU
        text = self.mcherry_channel_combo.currentText()
        if not text.strip():
            mcherry_channel = 0
        else:
            mcherry_channel = int(text)
        text = self.yfp_channel_combo.currentText()
        if not text.strip():
            yfp_channel = 0
        else:
            yfp_channel = int(text)

        mcherry_subdf = df.filter(pl.col('fluorescence_channel') == mcherry_channel).to_pandas()
        yfp_subdf = df.filter(pl.col('fluorescence_channel') == yfp_channel).to_pandas()

        if mcherry_subdf.empty and yfp_subdf.empty:
            QMessageBox.warning(
                self,
                "No data to plot",
                "No population statistics for the selected mCherry / YFP channels. "
                "Try channel 0 for single-channel (phase) TIFFs.",
            )
            self.population_canvas.draw()
            return

        # mcherry_subdf['mean_intensity'] = mcherry_subdf['mean_intensity'] / mcherry_rpu
        # mcherry_subdf['std_intensity'] = mcherry_subdf['std_intensity'] / mcherry_rpu
        # yfp_subdf['mean_intensity'] = yfp_subdf['mean_intensity'] / yfp_no_roi_rpu
        # yfp_subdf['std_intensity'] = yfp_subdf['std_intensity'] / yfp_no_roi_rpu

        # Plot graph
        fig = self.population_figure
        axs = fig.subplots(5, 1, gridspec_kw={'height_ratios': [3.5, 3.5, 1, 1, 1]})

        # For channel 0 (mCherry)
        ch0 = mcherry_subdf
        if not ch0.empty:
            sns.lineplot(data=ch0, x='time_hours', y='mean_intensity', ax=axs[0], color='red', label='mCherry')
            axs[0].fill_between(ch0['time_hours'],
                                ch0['mean_intensity'] - ch0['std_intensity'],
                                ch0['mean_intensity'] + ch0['std_intensity'],
                                color='red', alpha=0.3)
        axs[0].set_ylabel('mCherry')
        axs[0].set_xlabel('')
        axs[0].set_ylim(bottom=0)
        self._set_xticks_from_hours(axs[0], ch0['time_hours'] if not ch0.empty else None)
        axs[0].legend().set_visible(False)  # Hide legend

        # For channel 1 (YFP)
        ch1 = yfp_subdf
        if not ch1.empty:
            sns.lineplot(data=ch1, x='time_hours', y='mean_intensity', ax=axs[1], color='goldenrod', label='YFP')
            axs[1].fill_between(ch1['time_hours'],
                                ch1['mean_intensity'] - ch1['std_intensity'],
                                ch1['mean_intensity'] + ch1['std_intensity'],
                                color='goldenrod', alpha=0.3)
        axs[1].set_ylabel('YFP')
        axs[1].set_xlabel('')
        axs[1].set_ylim(bottom=0)
        self._set_xticks_from_hours(axs[1], ch1['time_hours'] if not ch1.empty else None)
        axs[1].legend().set_visible(False)  # Hide legend

        plt.tight_layout()

        # Medium change steps
        t_hours = np.sort(df['time_hours'].unique().to_numpy())
        comp_steps = generate_component_step_functions(component_intervals, t_hours)

        start_index = 2
        for idx, (comp, step) in enumerate(comp_steps.items()):
            axs[start_index + idx].step(t_hours, step, label=f'{comp}', where='post', color='black')
            axs[start_index + idx].set_ylabel(comp)
            self._set_xticks_from_hours(axs[start_index + idx], t_hours)
            axs[start_index + idx].legend().set_visible(False)  # Hide legend

        axs[-1].set_xlabel('Time (h)')

        self.population_canvas.draw()

        # plt.savefig('1_9_iptg_on_p_all.pdf')
        # plt.show()

    def compute_avg_sd_stats(self, df: pl.DataFrame):
        stats = (
            df.group_by("time")
            .agg([
                # --- GFP / fluorescence ---
                pl.col("fluo_level").mean().alias("mean_fluo"),
                pl.col("fluo_level").std().alias("std_fluo"),

                # --- Integrated intensity  ---
                (pl.col("fluo_level") * pl.col("area")).mean().alias("mean_integrated"),
                (pl.col("fluo_level") * pl.col("area")).std().alias("std_integrated"),

                pl.col("area").mean().alias("mean_area"),
                pl.col("area").std().alias("std_area"),

                pl.col("circularity").mean().alias("mean_circ"),
                pl.col("circularity").std().alias("std_circ"),
            ])
            .sort("time")
        )

        return stats

    def on_plot_avg_sd(self):
        self.population_figure.clear()

        df = self.metrics_service.df

        if df.is_empty():
            QMessageBox.warning(self, "No data", "Run segmentation first.")
            return

        # Compute stats
        gfp_channel = int(self.yfp_channel_combo.currentText())
        phc_channel = int(self.mcherry_channel_combo.currentText())

        df_gfp = df.filter(pl.col("fluorescence_channel") == gfp_channel)
        df_phc = df

        print("ALL TIMES:", sorted(df["time"].unique()))
        print("GFP TIMES:", sorted(df_gfp["time"].unique()))
        print("PHC TIMES:", sorted(df_phc["time"].unique()))

        stats_gfp = self.compute_avg_sd_stats(df_gfp)
        print("\n=== RAW DF CHECK ===")
        print(df.select(["time", "fluorescence_channel", "fluo_level"]).head(20))

        print("\n=== GFP DF CHECK ===")
        print(df_gfp.select(["time", "fluo_level"]).head(20))

        print("\n=== GROUPED STATS INPUT ===")
        print(df_gfp.group_by("time").agg(pl.col("fluo_level").count()))
        stats_phc = self.compute_avg_sd_stats(df_phc)


        # --- manual time mapping ---
        unique_times = sorted(self.metrics_service.df["time"].unique())
        # REAL TIMES HERE
        real_times = [0, 24, 48, 85, 96]
        # Build mapping
        time_map = dict(zip(unique_times, real_times))

        fig = self.population_figure
        axs = fig.subplots(3, 1)

        # --- TIME ---
        time_gfp = np.array([time_map[t] for t in stats_gfp["time"]])
        time_phc = np.array([time_map[t] for t in stats_phc["time"]])

        # --- 1. GFP ---
        axs[0].errorbar(
            time_gfp,
            stats_gfp["mean_fluo"],
            yerr=stats_gfp["std_fluo"],
            fmt='-o',
            capsize=5,
            color='green'
        )
        axs[0].set_title("GFP Fluorescence")
        axs[0].set_ylabel("Mean Intensity")

        # --- 2. BIOMASS (Integrated) ---
        axs[1].errorbar(
            time_phc,
            stats_phc["mean_integrated"],
            yerr=stats_phc["std_integrated"],
            fmt='-o',
            capsize=5,
            color='blue'
        )
        axs[1].set_title("Integrated Intensity (Biomass Proxy)")
        axs[1].set_ylabel("Total Signal")

        # --- 3. MORPHOLOGY (AREA) ---
        axs[2].errorbar(
            time_phc,
            stats_phc["mean_area"],
            yerr=stats_phc["std_area"],
            fmt='-o',
            capsize=5,
            color='goldenrod'
        )
        axs[2].set_title("Cell Area (Morphology)")
        axs[2].set_ylabel("Area")
        axs[2].set_xlabel("Time (hours)")

        fig.subplots_adjust(
            top=0.92,
            bottom=0.08,
            left=0.12,
            right=0.95,
            hspace=0.4
        )

        self.population_canvas.draw()

    def save_population_plot(self):
        """Save the current population plot"""
        if not hasattr(self, "population_figure"):
            QMessageBox.warning(self, "Error", "No plot available to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Population Plot",
            "",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)",
        )

        if file_path:
            try:
                self.population_figure.savefig(
                    file_path, dpi=300, bbox_inches="tight"
                )
                QMessageBox.information(
                    self, "Success", f"Plot saved to {file_path}"
                )
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save plot: {str(e)}")
    # def calculate_rpu_values(self):
    #     """
    #     Calculate RPU reference values from all segmented cells across all frames.
    #     Displays results in a dialog and offers to export to CSV.
    #     """
    #     from PySide6.QtWidgets import (
    #         QMessageBox,
    #         QDialog,
    #         QVBoxLayout,
    #         QLabel,
    #         QDialogButtonBox,
    #     )
    #
    #     # Get the metrics DataFrame from the singleton
    #     df = self.metrics_service.df
    #
    #     if df.is_empty():
    #         QMessageBox.warning(
    #             self,
    #             "No Data",
    #             "No metrics data available. Please run segmentation first.",
    #         )
    #         return
    #
    #     # Get available fluorescence channels
    #     fluo_columns = [col for col in df.columns if col.startswith("fluo_")]
    #
    #     if not fluo_columns:
    #         QMessageBox.warning(
    #             self,
    #             "No Data",
    #             "No fluorescence data found in metrics. Please run segmentation with fluorescence channels.",
    #         )
    #         return
    #
    #     # Calculate average values for each channel (ignoring zeros)
    #     rpu_values = {}
    #     for channel_col in fluo_columns:
    #         channel_num = int(channel_col.split("_")[1])  # Extract channel number
    #
    #         # Filter out zeros and calculate the average
    #         channel_data = df.filter(pl.col(channel_col) > 0.1)
    #
    #         if channel_data.height > 0:
    #             avg_value = channel_data[channel_col].mean()
    #             std_value = channel_data[channel_col].std()
    #             cell_count = channel_data.height
    #
    #             channel_name = f"Channel {channel_num}"
    #             if channel_num == 1:
    #                 channel_name = "mCherry"
    #             elif channel_num == 2:
    #                 channel_name = "YFP"
    #
    #             rpu_values[channel_num] = {
    #                 "name": channel_name,
    #                 "avg_value": avg_value,
    #                 "std_value": std_value,
    #                 "cell_count": cell_count,
    #             }
    #
    #     if not rpu_values:
    #         QMessageBox.warning(
    #             self,
    #             "No Data",
    #             "No valid fluorescence data found (all values are zero or missing).",
    #         )
    #         return
    #
    #     # Create a dialog to display the results
    #     dialog = QDialog(self)
    #     dialog.setWindowTitle("RPU Reference Values")
    #     dialog.setMinimumWidth(400)
    #
    #     layout = QVBoxLayout(dialog)
    #
    #     # Add title
    #     title_label = QLabel("<h3>RPU Reference Values</h3>")
    #     title_label.setAlignment(Qt.AlignCenter)
    #     layout.addWidget(title_label)
    #
    #     # Add description
    #     desc_label = QLabel(
    #         "The following reference values were calculated from single-cell analysis "
    #         "across all frames using segmentation model: UNet"
    #     )
    #     desc_label.setWordWrap(True)
    #     layout.addWidget(desc_label)
    #
    #     # Add the calculated values
    #     for channel_num, values in rpu_values.items():
    #         value_label = QLabel(
    #             f"<b>{values['name']} (Channel {channel_num}):</b> {values['avg_value']:.2f} ± {values['std_value']:.2f} "
    #             f"<i>(from {values['cell_count']} cells)</i>"
    #         )
    #         value_label.setTextFormat(Qt.RichText)
    #         layout.addWidget(value_label)
    #
    #     # Add note
    #     note_label = QLabel(
    #         "<i>Note: These values can be used as RPU reference values for normalizing "
    #         "fluorescence measurements in future experiments.</i>"
    #     )
    #     note_label.setWordWrap(True)
    #     layout.addWidget(note_label)
    #
    #     # Add buttons
    #     button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
    #     layout.addWidget(button_box)
    #
    #     # Connect button signals
    #     button_box.accepted.connect(lambda: self.export_rpu_values(rpu_values, dialog))
    #     button_box.rejected.connect(dialog.reject)
    #
    #     # Show the dialog
    #     dialog.exec_()
    #
    # def export_rpu_values(self, rpu_values, dialog):
    #     """Export the calculated RPU values to a CSV file"""
    #
    #     # Ask for save location
    #     file_path, _ = QFileDialog.getSaveFileName(
    #         self,
    #         "Save RPU Reference Values",
    #         "rpu_reference_values.csv",
    #         "CSV Files (*.csv)",
    #     )
    #
    #     if not file_path:
    #         return
    #
    #     try:
    #         with open(file_path, "w", newline="") as csvfile:
    #             writer = csv.writer(csvfile)
    #
    #             # Write header row
    #             writer.writerow(
    #                 [
    #                     "Channel",
    #                     "Channel Name",
    #                     "RPU Reference Value",
    #                     "Standard Deviation",
    #                     "Cell Count",
    #                 ]
    #             )
    #
    #             # Write data rows
    #             for channel_num, values in rpu_values.items():
    #                 writer.writerow(
    #                     [
    #                         channel_num,
    #                         values["name"],
    #                         f"{values['avg_value']:.6f}",
    #                         f"{values['std_value']:.6f}",
    #                         values["cell_count"],
    #                     ]
    #                 )
    #
    #             # Write metadata
    #             writer.writerow([])
    #             writer.writerow(["Segmentation Model", "UNet"])
    #
    #             # If experiment info is available, add it
    #             if self.experiment:
    #                 writer.writerow(
    #                     ["Experiment Name", getattr(self.experiment, "name", "Unknown")]
    #                 )
    #                 writer.writerow(
    #                     [
    #                         "ND2 Files",
    #                         ", ".join(
    #                             getattr(self.experiment, "image_files", ["Unknown"])
    #                         ),
    #                     ]
    #                 )
    #
    #         dialog.accept()
    #
    #         from PySide6.QtWidgets import QMessageBox
    #
    #         QMessageBox.information(
    #             self, "Export Complete", f"RPU reference values saved to:\n{file_path}"
    #         )
    #
    #     except Exception as e:
    #         from PySide6.QtWidgets import QMessageBox
    #
    #         QMessageBox.warning(
    #             self, "Export Error", f"Failed to export RPU values: {str(e)}"
    #         )
