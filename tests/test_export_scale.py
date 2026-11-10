"""The Density Regions export writes x_um/y_um with the file's pixel size, not a fixed 0.07."""
from nd2_analyzer.ui.widgets.motility_widget import export_um_per_px


class _Image:
    def __init__(self, um):
        self.um = um

    def pixel_size_um(self):
        return self.um


def test_file_pixel_size_is_used():
    assert export_um_per_px(_Image(0.064497), 0.07) == 0.064497


def test_default_when_the_file_gives_none():
    assert export_um_per_px(_Image(None), 0.07) == 0.07
    assert export_um_per_px(_Image(1.0), 0.07) == 0.07   # uncalibrated ND2
    assert export_um_per_px(None, 0.07) == 0.07
