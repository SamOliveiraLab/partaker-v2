from PySide6.QtWidgets import QApplication
import sys

from nd2_analyzer.ui import App


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName('Partaker')
    tabWidgetApp = App()
    tabWidgetApp.showMaximized()
    sys.exit(app.exec())