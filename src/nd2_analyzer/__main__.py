import sys

from PySide6.QtWidgets import QApplication, QSplashScreen
from PySide6.QtGui import QPixmap

from nd2_analyzer.ui import App


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Partaker")
    mainWin = App()
    mainWin.show()
    sys.exit(app.exec())
