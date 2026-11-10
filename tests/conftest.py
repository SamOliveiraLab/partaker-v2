"""One Qt application for the whole test run, made before any test: a test that starts a bare QCoreApplication (the PIV
motion tests) would otherwise leave no QApplication for the page and window tests, and making a widget then aborts."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_configure(config):
    from PySide6.QtWidgets import QApplication
    config._partaker_qt_app = QApplication.instance() or QApplication([])
