import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenuBar,
    QMenu,
    QFileDialog,
    QDialog,
    QVBoxLayout,
    QLabel,
    QPushButton)
from PySide6.QtCore import Qt


class AboutDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("About")
        self.setFixedSize(300, 150)

        layout = QVBoxLayout()
        about_label = QLabel(
            'Partaker microfluidic image analyzer.\nVersion 1.0\nCreated by Henrique Núñez and Bukola Akindipe')
        layout.addWidget(about_label)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        self.setLayout(layout)
