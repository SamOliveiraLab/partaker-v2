import numpy as np
from nd2_analyzer.ui.widgets.view_roi import roi_window, window_to_frame


def test_window_is_the_roi_box_with_the_outside_black():
    img = np.arange(100, dtype=np.uint16).reshape(10, 10) + 1
    roi = np.zeros((10, 10), bool); roi[2:6, 3:8] = True; roi[2, 3] = False      # a notch in the corner
    win, box = roi_window(img, roi)
    assert box == dict(row0=2, row1=6, col0=3, col1=8) and win.shape == (4, 5)
    assert win[0, 0] == 0 and win[0, 1] == img[2, 4] and win[3, 4] == img[5, 7]
    assert img[2, 3] == 24                                                       # the frame itself is untouched
    assert window_to_frame(1, 0, box) == (4, 2)


def test_without_roi_or_with_a_wrong_size_nothing_changes():
    img = np.ones((5, 5), np.uint8)
    assert roi_window(img, None)[1] is None
    assert roi_window(img, np.ones((6, 6), bool))[1] is None
    assert window_to_frame(3, 4, None) == (3, 4)


def test_rgb_window():
    img = np.ones((6, 6, 3), np.uint8) * 200
    roi = np.zeros((6, 6), bool); roi[1:4, 1:5] = True
    win, box = roi_window(img, roi)
    assert win.shape == (3, 4, 3) and (win == 200).all()
