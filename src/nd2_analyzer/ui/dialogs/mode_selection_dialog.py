from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QRadioButton, QButtonGroup, 
                               QCheckBox, QFrame, QApplication)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap, QIcon
import sys
import os

from ..biofilms.analysis_mode import AnalysisMode, AnalysisModeConfig

class ModeSelectionDialog(QDialog):
    """Dialog for selecting analysis mode at startup"""
    
    def __init__(self, config: AnalysisModeConfig, parent=None):
        super().__init__(parent)
        
        self.config = config
        self.selected_mode = config.get_mode()
        
        self.setWindowTitle("Partaker - Select Analysis Mode")
        self.setModal(True)
        self.setFixedSize(500, 400)
        
        # Center the dialog on screen
        self.center_on_screen()
        
        self.init_ui()
    
    def center_on_screen(self):
        """Center the dialog on the screen"""
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Title
        title_label = QLabel("Choose Analysis Mode")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Select the type of analysis you want to perform:")
        subtitle_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle_label)
        
        # Mode selection area
        modes_frame = QFrame()
        modes_frame.setFrameStyle(QFrame.Box)
        modes_layout = QVBoxLayout(modes_frame)
        
        # Create button group for radio buttons
        self.mode_group = QButtonGroup()
        
        # Single Cell Mode option
        self.single_cell_radio = QRadioButton(AnalysisMode.SINGLE_CELL.display_name)
        self.single_cell_radio.setChecked(self.selected_mode == AnalysisMode.SINGLE_CELL)
        
        single_cell_desc = QLabel(AnalysisMode.SINGLE_CELL.description)
        single_cell_desc.setWordWrap(True)
        single_cell_desc.setStyleSheet("color: gray; margin-left: 20px; margin-bottom: 10px;")
        
        # Biofilm Cloud Mode option
        self.biofilm_radio = QRadioButton(AnalysisMode.BIOFILM_CLOUD.display_name)
        self.biofilm_radio.setChecked(self.selected_mode == AnalysisMode.BIOFILM_CLOUD)
        
        biofilm_desc = QLabel(AnalysisMode.BIOFILM_CLOUD.description)
        biofilm_desc.setWordWrap(True)
        biofilm_desc.setStyleSheet("color: gray; margin-left: 20px; margin-bottom: 10px;")
        
        # Add radio buttons to group
        self.mode_group.addButton(self.single_cell_radio)
        self.mode_group.addButton(self.biofilm_radio)
        
        # Add to layout
        modes_layout.addWidget(self.single_cell_radio)
        modes_layout.addWidget(single_cell_desc)
        modes_layout.addWidget(self.biofilm_radio)
        modes_layout.addWidget(biofilm_desc)
        
        layout.addWidget(modes_frame)
        
        # Remember choice checkbox
        self.remember_checkbox = QCheckBox("Remember my choice and don't ask again")
        self.remember_checkbox.setChecked(False)
        layout.addWidget(self.remember_checkbox)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        self.ok_button = QPushButton("Continue")
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.accept_selection)
        
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)
        
        layout.addLayout(button_layout)
        
        # Connect radio button signals
        self.single_cell_radio.toggled.connect(self.on_mode_changed)
        self.biofilm_radio.toggled.connect(self.on_mode_changed)
    
    def on_mode_changed(self):
        """Handle mode selection change"""
        if self.single_cell_radio.isChecked():
            self.selected_mode = AnalysisMode.SINGLE_CELL
        elif self.biofilm_radio.isChecked():
            self.selected_mode = AnalysisMode.BIOFILM_CLOUD
    
    def accept_selection(self):
        """Accept the current selection and save config"""
        # Update config
        self.config.set_mode(self.selected_mode, save=False)
        self.config.remember_choice = self.remember_checkbox.isChecked()
        self.config.save_config()
        
        self.accept()
    
    def get_selected_mode(self) -> AnalysisMode:
        """Get the selected mode"""
        return self.selected_mode
    
    @staticmethod
    def show_mode_selection(config: AnalysisModeConfig, parent=None) -> AnalysisMode:
        """Static method to show mode selection dialog and return selected mode"""
        
        # If no config file exists, always show dialog (first time use)
        if not os.path.exists(config.CONFIG_FILE):
            dialog = ModeSelectionDialog(config, parent)
            result = dialog.exec()
            if result == QDialog.Accepted:
                return dialog.get_selected_mode()
            else:
                sys.exit(0)
        
        # If user chose to remember choice and we have a valid mode, use it
        if config.remember_choice and config.get_mode():
            return config.get_mode()
        
        dialog = ModeSelectionDialog(config, parent)
        result = dialog.exec()
        if result == QDialog.Accepted:
            return dialog.get_selected_mode()
        else:
            sys.exit(0)