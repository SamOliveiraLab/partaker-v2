"""One Qt application for the whole test run, made before any test: a test that starts a bare QCoreApplication (the PIV
motion tests) would otherwise leave no QApplication for the page and window tests, and making a widget then aborts."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


import pytest


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="also run the checks that take minutes (a whole recording followed through time)")


def pytest_configure(config):
    from PySide6.QtWidgets import QApplication
    config._partaker_qt_app = QApplication.instance() or QApplication([])
    config.addinivalue_line("markers", "slow: takes minutes; runs only with --run-slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="takes minutes: run with --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
