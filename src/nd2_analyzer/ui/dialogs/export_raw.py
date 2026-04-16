import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QDoubleSpinBox,
    QGroupBox,
    QFormLayout,
    QSpinBox,
    QTabWidget,
    QWidget,
    QLabel,
    QProgressBar,
    QMessageBox,
    QApplication
)

from pubsub import pub

from nd2_analyzer.data.image_data import ImageData
from nd2_analyzer.data.appstate import ApplicationState

class ExportDialog(QDialog):
    """Dialog for exporting to TIF files"""


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Raw to TIF")
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)
        self.total_images = 0

        self.file_paths = []
        self.output_directory = None
        self.export_mode = None

        #  Set a name for export directories/files
        appstate = ApplicationState.get_instance()
        if appstate and appstate.experiment:
            self.exp_name = appstate.experiment.name
        else:
            self.exp_name = "export"   # Default name if no experiment is selected

        self.create_ui()



    def create_ui(self):
        """Create the user interface"""
        main_layout = QVBoxLayout()

        # Create a tabbed interface for visualization options
        tab_widget = QTabWidget()

        # Basic Settings
        basic_tab = self.create_basic_tab()
        tab_widget.addTab(basic_tab, "Basic Settings")


        main_layout.addWidget(tab_widget)

        # Main dialog buttons
        button_layout = QHBoxLayout()
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.convert_to_tif)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)

        # Progress display
        self.progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)

        self.progress_label = QLabel("Ready")

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)

        self.progress_group.setLayout(progress_layout)
        main_layout.addWidget(self.progress_group)

        self.setLayout(main_layout)

    def create_basic_tab(self) -> QWidget:
        """Create the basic settings tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Experiment details group
        details_group = QGroupBox("Export Details")
        details_layout = QFormLayout()


        # Output directory widgets
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        self.output_dir_edit.setReadOnly(True)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.select_output_directory)

        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.output_dir_edit)
        dir_layout.addWidget(self.browse_button)
        details_layout.addRow("Output Directory:", dir_layout)

        # Export mode dropdown widget
        self.export_mode_combo = QComboBox()

        self.export_mode_combo.addItems([
            "Batch Directory (TIFF files)",
            "OME-TIFF",
            "Stacked TIFF"
        ])
        # Set default values for export mode
        self.export_mode_combo.setPlaceholderText("Select export mode...")
        self.export_mode_combo.setCurrentIndex(-1)
        # Set values for export mode
        details_layout.addRow("Export Mode:", self.export_mode_combo)
        self.export_mode_combo.currentTextChanged.connect(self.update_export_mode)

        details_group.setLayout(details_layout)
        layout.addWidget(details_group)


        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def update_export_mode(self, mode):
        self.export_mode = mode
        print(f"Export mode: {mode}")

    def select_output_directory(self):
        """Open a file dialog to select the output directory for exported images"""
        folder = QFileDialog.getExistingDirectory(self,"Select Output Folder")
        if folder:
            self.output_directory = folder
            self.output_dir_edit.setText(folder)

    def convert_to_tif(self):
        """Dispatch export based on selected mode"""
        if not self.output_directory:
            QMessageBox.warning(self,"No Output Directory","Please select an output directory first.")
            return

        output_folder = self.output_directory

        mode = self.export_mode_combo.currentText()

        if mode == "Batch Directory (TIFF files)":
            self.batch_tiff(output_folder)
        elif mode == "OME-TIFF":
            self.ome_tiff(output_folder)
        elif mode == "Stacked TIFF":
            self.stacked_tiff(output_folder)
        else:
            QMessageBox.warning(self, "Export Mode", "Please select an export mode.")

    def batch_tiff(self, output_folder):
        """Export as Batch Directory (TIFF files)"""
        import tifffile
        from pathlib import Path
        image_data = ImageData.get_instance()

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        shape = image_data.data.shape
        T = shape[0]
        P = shape[1]
        C = shape[2] if len(shape) == 5 else 1

        # Initialize progress bar
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(T * P * C)
        self.progress_bar.setValue(0)

        total = 0
        for p in range(P):
            # Create folder(s) for this position
            exp_folder = output_folder / f"batch_{self.exp_name}"
            exp_folder.mkdir(parents=True, exist_ok=True)

            pos_folder = exp_folder / f"position_{p}"
            pos_folder.mkdir(parents=True, exist_ok=True)

            for t in range(T):
                for c in range(C):
                    img = image_data.get(t, p, c)

                    path = pos_folder / f"pos{p}_t{t}_{c}.tif"
                    tifffile.imwrite(path, img)

                    total += 1
                    self.progress_bar.setValue(total)
                    QApplication.processEvents()

        self.total_images = total
        print(f"✅ Exported {total} images")
        self.progress_label.setText(f"Exported {total} images: Batch TIFF export complete")

    def ome_tiff(self, output_folder):
        """Export as OME-TIFF"""
        import tifffile as tf
        from pathlib import Path

        image_data = ImageData.get_instance()


        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        ome_path = output_folder / f"{self.exp_name}.ome.tif"

        data = image_data.data

        T, P, C, Y, X = data.shape

        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(T * P)
        self.progress_bar.setValue(0)

        # Create memmap process for OME-TIFF
        target = tf.memmap(
            ome_path,
            shape=(T, P, C, Y, X),
            dtype=data.dtype,
            metadata={"axes": "TPCYX"},
            bigtiff=True
        )

        total = 0
        for p in range(P):
            for t in range(T):
                for c in range(C):
                    img = image_data.get(t, p, c)

                    target[t, p, c] = img

                total += 1
                self.progress_bar.setValue(total)
                QApplication.processEvents()

        self.total_images = total
        print(f"✅ Exported {total} images")

        target.flush()

        self.progress_label.setText(f"Exported {total} images: OME-TIFF export complete")

    def stacked_tiff(self, output_folder):
        """Export stacked, multipage TIFF"""
        import tifffile as tf
        from pathlib import Path

        image_data = ImageData.get_instance()

        output_root = Path(output_folder)
        output_root.mkdir(parents=True, exist_ok=True)

        # Create export subfolder
        exp_folder = output_root / f"stacked_{self.exp_name}"
        exp_folder.mkdir(parents=True, exist_ok=True)

        data = image_data.data
        T = data.shape[0]
        P = data.shape[1]
        C = data.shape[2] if len(data.shape) == 5 else 1

        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(T * P * C)  # Total pages = T * P * C
        self.progress_bar.setValue(0)

        total = 0
        # Loop per position
        for p in range(P):
            for c in range(C):
                stack_path = exp_folder / f"pos{p}_{c}.tif"
                with tf.TiffWriter(stack_path, bigtiff=True) as tif:
                    for t in range(T):
                        # Grab an image frame
                        img = image_data.get(t, p, c)
                        # Write each frame as a new page
                        tif.write(img)

                        total += 1
                        self.progress_bar.setValue(total)
                        QApplication.processEvents()

            print(f"✅ Exported {total} images")
        self.total_images = total
        self.progress_label.setText(f"Exported {total} images: Stacked TIFF export complete")
