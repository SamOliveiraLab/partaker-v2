import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QSpinBox, QListWidget, QListWidgetItem, QFileDialog
)
import csv
import datetime
from PySide6.QtCore import Qt
from pubsub import pub
import polars as pl  # Import Polars
import os
import pickle

# Assume MetricsService is a singleton with a .df attribute (Polars DataFrame)
from nd2_analyzer.analysis.metrics_service import MetricsService
from nd2_analyzer.data.experiment import Experiment

class PopulationWidget(QWidget):
    """
    Widget for plotting population-level fluorescence over time.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics_service = MetricsService()  # Singleton instance
        self.init_ui()
        self.experiment : Experiment = None

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Matplotlib figure
        self.population_figure = plt.figure()
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
        ch_layout = QHBoxLayout()
        ch_layout.addWidget(QLabel("Fluorescence Channel:"))
        self.channel_combo = QComboBox()
        ch_layout.addWidget(self.channel_combo)
        layout.addLayout(ch_layout)

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

        # Plot button
        plot_btn = QPushButton("Plot Fluorescence")
        plot_btn.clicked.connect(self.plot_fluorescence_signal)
        layout.addWidget(plot_btn)

        # Export buttons (side by side)
        export_layout = QHBoxLayout()
        
        # Export DataFrame button
        export_btn = QPushButton("Export DataFrame to CSV")
        export_btn.clicked.connect(self.export_dataframe)
        export_layout.addWidget(export_btn)
        
        # Calculate RPU button (new)
        rpu_btn = QPushButton("Calculate RPU Reference Values")
        rpu_btn.clicked.connect(self.calculate_rpu_values)
        export_layout.addWidget(rpu_btn)
        
        layout.addLayout(export_layout)

        self.setLayout(layout)

        # Listen for data loading to populate UI
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")
        pub.subscribe(self.on_experiment_loaded, "experiment_loaded")
    
    def on_image_data_loaded(self, image_data):
        # Populate positions and channels based on image_data shape
        shape = image_data.data.shape
        t_max, p_max, c_max = shape[0] - 1, shape[1] - 1, shape[2] - 1

        self.position_list.clear()
        for p in range(p_max + 1):
            item = QListWidgetItem(f"{p}")
            item.setSelected(True)
            self.position_list.addItem(item)

        self.channel_combo.clear()
        for c in range(1, c_max + 1):  # Fluorescence channels start from 1
            self.channel_combo.addItem(str(c))

        self.time_min_box.setRange(0, t_max)
        self.time_max_box.setRange(0, t_max)
        self.time_max_box.setValue(t_max)

    def on_experiment_loaded(self, experiment):
        self.experiment = experiment

    def plot_fluorescence_signal(self):
        # Get user selections
        selected_positions = [
            int(item.text()) for item in self.position_list.selectedItems()
        ]
        if not selected_positions:
            return

        channel = int(self.channel_combo.currentText())
        t_start = self.time_min_box.value()
        t_end = self.time_max_box.value()

        # Get the metrics DataFrame from the singleton
        df = self.metrics_service.df

        if df.is_empty():
            return
        
        # Test each filter condition separately
        pos_filter = df["position"].is_in(selected_positions)
        time_start_filter = df["time"] >= t_start
        time_end_filter = df["time"] <= t_end
        
        print(f"Position filter matches: {df.filter(pos_filter).shape[0]} rows")
        print(f"Time start filter matches: {df.filter(time_start_filter).shape[0]} rows")
        print(f"Time end filter matches: {df.filter(time_end_filter).shape[0]} rows")

        # Filter for selected positions, channel, and time range
        mask = (
            df["position"].is_in(selected_positions)
            & (df["time"] >= t_start)
            & (df["time"] <= t_end)
        )
        subdf = df.filter(mask)

        if subdf.is_empty():
            return

        # The metric we want is based on the selected channel
        metric_col = f"fluo_{channel}"
        metric_name = f"Fluorescence for channel {channel}"

        # Group by time, aggregate mean and std of the selected metric across cells and positions
        grouped = (
            subdf.group_by("time")
            .agg([
                pl.col(metric_col).mean().alias("mean_metric"),
                pl.col(metric_col).std().alias("std_metric"),
            ])
            .sort("time")
        )

        # Filter values smaller than epsilon, and multiply time by the experiment interval constant
        epsilon = 0.1
        valid = grouped.filter(pl.col("mean_metric") > epsilon)

        # Get the interval from experiment or use a default value
        if hasattr(self, "experiment") and self.experiment is not None and hasattr(self.experiment, "interval"):
            factor = self.experiment.interval
        else:
            # Use a default interval value of 600 seconds (10 minutes)
            factor = 600
            print("Warning: Using default interval of 600 seconds (10 minutes)")

        # Interval is in seconds, divide by 3600 to get hours
        result = valid.with_columns(
            (pl.col("time") * factor / 3600).alias("scaled_time")
        )

        times = result["scaled_time"].to_numpy()
        mean_metric = result["mean_metric"].to_numpy()
        std_metric = result["std_metric"].to_numpy()

        # Store the plotted data for saving
        self.last_plotted_times = times
        self.last_plotted_mean = mean_metric
        self.last_plotted_std = std_metric
        self.last_plotted_metric_name = metric_name

        # Plot
        self.population_figure.clear()
        ax = self.population_figure.add_subplot(111)
        ax.plot(times, mean_metric, color="red", label=f"Mean {metric_name}")
        ax.fill_between(times, mean_metric - std_metric, mean_metric + std_metric, color="red", alpha=0.2, label="Std Dev")
        ax.set_xlabel("Time [h]")
        ax.set_ylabel(f"Mean Cell {metric_name}")
        ax.set_title(f"{metric_name} over Time (Positions: {selected_positions}, Channel: {channel})")
        ax.legend()
        self.population_canvas.draw()
    
    def export_dataframe(self):
        # Export the DataFrame to a CSV file
        df = self.metrics_service.df
        if not df.is_empty():
            df.write_csv("cell_metrics.csv")
            print("DataFrame exported to cell_metrics.csv")
        else:
            print("No data to export")


    def save_population_data(self, folder_path):
        """Save population analysis data to the specified folder"""
        try:
            # Create a dictionary to store all relevant population data
            population_data = {
                # Store the last plotted data
                "times": getattr(self, "last_plotted_times", None),
                "mean_metric": getattr(self, "last_plotted_mean", None),
                "std_metric": getattr(self, "last_plotted_std", None),
                "metric_name": getattr(self, "last_plotted_metric_name", None),
                
                # Store the user selections
                "selected_positions": [item.text() for item in self.position_list.selectedItems()],
                "selected_channel": self.channel_combo.currentText() if self.channel_combo.count() > 0 else None,
                "selected_metric": self.metric_combo.currentText(),
                "time_min": self.time_min_box.value(),
                "time_max": self.time_max_box.value()
            }

            # Save to file
            population_file = os.path.join(folder_path, "population_data.pkl")
            with open(population_file, 'wb') as f:
                pickle.dump(population_data, f)

            print(f"Population data saved to {population_file}")
            return True
        except Exception as e:
            print(f"Error saving population data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_population_data(self, folder_path):
        """Load population analysis data from the specified folder"""
        try:
            import os
            import pickle
            
            # Check if population data file exists
            population_file = os.path.join(folder_path, "population_data.pkl")
            if not os.path.exists(population_file):
                print(f"No population data found at {population_file}")
                return False

            # Load the data
            with open(population_file, 'rb') as f:
                population_data = pickle.load(f)
                
            # Store the loaded plot data
            self.last_plotted_times = population_data.get("times")
            self.last_plotted_mean = population_data.get("mean_metric")
            self.last_plotted_std = population_data.get("std_metric")
            self.last_plotted_metric_name = population_data.get("metric_name")
            
            # Restore user selections if possible
            try:
                # Restore selected positions
                selected_positions = population_data.get("selected_positions", [])
                for i in range(self.position_list.count()):
                    item = self.position_list.item(i)
                    item.setSelected(item.text() in selected_positions)
                    
                # Restore selected channel
                selected_channel = population_data.get("selected_channel")
                if selected_channel:
                    index = self.channel_combo.findText(selected_channel)
                    if index >= 0:
                        self.channel_combo.setCurrentIndex(index)
                        
                # Restore selected metric
                selected_metric = population_data.get("selected_metric")
                if selected_metric:
                    index = self.metric_combo.findText(selected_metric)
                    if index >= 0:
                        self.metric_combo.setCurrentIndex(index)
                        
                # Restore time range
                time_min = population_data.get("time_min")
                time_max = population_data.get("time_max")
                if time_min is not None:
                    self.time_min_box.setValue(time_min)
                if time_max is not None:
                    self.time_max_box.setValue(time_max)
            except Exception as e:
                print(f"Error restoring UI state: {e}")
            
            # Update the plot with the loaded data
            self.update_plot_from_loaded_data()
            
            print(f"Population data loaded from {population_file}")
            return True
        except Exception as e:
            print(f"Error loading population data: {e}")
            import traceback
            traceback.print_exc()
            return False


    def update_plot_from_loaded_data(self):
        """Update the plot with loaded data"""
        if (not hasattr(self, "last_plotted_times") or 
            self.last_plotted_times is None or 
            not hasattr(self, "last_plotted_mean") or 
            self.last_plotted_mean is None):
            print("No population data to plot")
            return
            
        try:
            # Get the loaded data
            times = self.last_plotted_times
            mean_metric = self.last_plotted_mean
            std_metric = self.last_plotted_std
            metric_name = self.last_plotted_metric_name or "Fluorescence"
            
            # Plot
            self.population_figure.clear()
            ax = self.population_figure.add_subplot(111)
            ax.plot(times, mean_metric, color="red", label=f"Mean {metric_name}")
            
            if std_metric is not None:
                ax.fill_between(times, mean_metric - std_metric, mean_metric + std_metric, 
                            color="red", alpha=0.2, label="Std Dev")
                            
            ax.set_xlabel("Time [h]")
            ax.set_ylabel(f"Mean Cell {metric_name}")
            
            # Get the channel and positions for the title
            selected_positions = [item.text() for item in self.position_list.selectedItems()]
            channel = self.channel_combo.currentText() if self.channel_combo.count() > 0 else "Unknown"
            
            ax.set_title(f"{metric_name} over Time (Positions: {selected_positions}, Channel: {channel})")
            ax.legend()
            self.population_canvas.draw()
            
            print("Population plot updated with loaded data")
        except Exception as e:
            print(f"Error updating population plot: {e}")
            import traceback
            traceback.print_exc()
            
    def calculate_rpu_values(self):
        """
        Calculate RPU reference values from all segmented cells across all frames.
        Displays results in a dialog and offers to export to CSV.
        """
        from PySide6.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QLabel, QDialogButtonBox, QFileDialog
        import csv
        
        # Get the metrics DataFrame from the singleton
        df = self.metrics_service.df
        
        if df.is_empty():
            QMessageBox.warning(self, "No Data", "No metrics data available. Please run segmentation first.")
            return
        
        # Get available fluorescence channels
        fluo_columns = [col for col in df.columns if col.startswith("fluo_")]
        
        if not fluo_columns:
            QMessageBox.warning(self, "No Data", "No fluorescence data found in metrics. Please run segmentation with fluorescence channels.")
            return
        
        # Calculate average values for each channel (ignoring zeros)
        rpu_values = {}
        for channel_col in fluo_columns:
            channel_num = int(channel_col.split("_")[1])  # Extract channel number
            
            # Filter out zeros and calculate the average
            channel_data = df.filter(pl.col(channel_col) > 0.1)
            
            if channel_data.height > 0:
                avg_value = channel_data[channel_col].mean()
                std_value = channel_data[channel_col].std()
                cell_count = channel_data.height
                
                channel_name = f"Channel {channel_num}"
                if channel_num == 1:
                    channel_name = "mCherry"
                elif channel_num == 2:
                    channel_name = "YFP"
                    
                rpu_values[channel_num] = {
                    "name": channel_name,
                    "avg_value": avg_value,
                    "std_value": std_value,
                    "cell_count": cell_count
                }
        
        if not rpu_values:
            QMessageBox.warning(self, "No Data", "No valid fluorescence data found (all values are zero or missing).")
            return
        
        # Create a dialog to display the results
        dialog = QDialog(self)
        dialog.setWindowTitle("RPU Reference Values")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        
        # Add title
        title_label = QLabel("<h3>RPU Reference Values</h3>")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Add description
        desc_label = QLabel(
            "The following reference values were calculated from single-cell analysis "
            "across all frames using segmentation model: UNet"
        )
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Add the calculated values
        for channel_num, values in rpu_values.items():
            value_label = QLabel(
                f"<b>{values['name']} (Channel {channel_num}):</b> {values['avg_value']:.2f} ± {values['std_value']:.2f} "
                f"<i>(from {values['cell_count']} cells)</i>"
            )
            value_label.setTextFormat(Qt.RichText)
            layout.addWidget(value_label)
        
        # Add note
        note_label = QLabel(
            "<i>Note: These values can be used as RPU reference values for normalizing "
            "fluorescence measurements in future experiments.</i>"
        )
        note_label.setWordWrap(True)
        layout.addWidget(note_label)
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)
        
        # Connect button signals
        button_box.accepted.connect(lambda: self.export_rpu_values(rpu_values, dialog))
        button_box.rejected.connect(dialog.reject)
        
        # Show the dialog
        dialog.exec_()

    def export_rpu_values(self, rpu_values, dialog):
        """Export the calculated RPU values to a CSV file"""
        
        # Ask for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save RPU Reference Values", "rpu_reference_values.csv", "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header row
                writer.writerow(["Channel", "Channel Name", "RPU Reference Value", "Standard Deviation", "Cell Count"])
                
                # Write data rows
                for channel_num, values in rpu_values.items():
                    writer.writerow([
                        channel_num,
                        values['name'],
                        f"{values['avg_value']:.6f}",
                        f"{values['std_value']:.6f}",
                        values['cell_count']
                    ])
                
                # Write metadata
                writer.writerow([])
                writer.writerow(["Segmentation Model", "UNet"])
                
                # If experiment info is available, add it
                if self.experiment:
                    writer.writerow(["Experiment Name", getattr(self.experiment, "name", "Unknown")])
                    writer.writerow(["ND2 Files", ", ".join(getattr(self.experiment, "nd2_files", ["Unknown"]))])
                
            dialog.accept()
            
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Export Complete", f"RPU reference values saved to:\n{file_path}")
            
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Export Error", f"Failed to export RPU values: {str(e)}")