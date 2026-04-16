import sys

from PySide6.QtWidgets import QApplication, QSplashScreen
from PySide6.QtGui import QPixmap

from nd2_analyzer.ui import App

# base_path = getattr(
#     sys, '_MEIPASS', os.path.dirname(
#         os.path.abspath(__file__)))
# ui_file_path = os.path.join(base_path, "ui.py")

# print("Looking for ui.py:", os.path.exists(ui_file_path))

# if __name__ == "__main__":
#
#     app = QApplication([])
#     pixmap = QPixmap(":/splash.png")
#     splash = QSplashScreen(pixmap)
#     splash.show()
#     app.processEvents()            ...
#
# window = QMainWindow()
# window.show()
# splash.finish(window)
# sys.exit(app.exec())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Partaker")

    # # TODO: Splash screen setup
    # pixmap = QPixmap(":/splashscreen.png")
    # splash = QSplashScreen(pixmap)
    # splash.show()
    # app.processEvents()

    mainWin = App()
    mainWin.show()
    # splash.finish(mainWin)
    sys.exit(app.exec())
