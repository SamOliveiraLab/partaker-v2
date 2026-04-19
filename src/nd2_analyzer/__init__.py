import sys

from PySide6.QtWidgets import QApplication, QSplashScreen
from PySide6.QtGui import QPixmap


def _patch_torch_load_for_cpu():
    """Force CPU mapping and disable weights_only so CUDA-saved checkpoints
    (e.g. omnipose models) load on CPU-only machines. Must run before any
    cellpose/omnipose import. Overrides caller kwargs — setdefault is not
    enough because cellpose passes weights_only=True explicitly."""
    try:
        import torch
    except ImportError:
        return

    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        if not torch.cuda.is_available():
            kwargs["map_location"] = torch.device("cpu")
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = _patched_load


_patch_torch_load_for_cpu()

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from nd2_analyzer.ui import App  # noqa: E402  (must come after patches)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Partaker")
    mainWin = App()
    mainWin.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
