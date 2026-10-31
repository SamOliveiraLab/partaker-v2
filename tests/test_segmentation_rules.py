"""The three rules that make the window's masks match the digital-twin rerun:
ROI on the masks, the registered frame into the model, and the pixel size into Omnipose."""
import numpy as np
import pytest

from nd2_analyzer.analysis.segmentation.segmentation_service import apply_roi_mask
from nd2_analyzer.analysis.segmentation.segmentation_models import omnipose_scale_params
from nd2_analyzer.analysis.segmentation import segmentation_cache as sc


def _roi():
    roi = np.zeros((20, 20), bool)
    roi[4:16, 4:16] = True
    return roi


def test_roi_keeps_inside_drops_outside_and_edge():
    seg = np.zeros((20, 20), np.uint16)
    seg[6:9, 6:9] = 7        # inside
    seg[0:2, 0:2] = 8        # outside
    seg[2:6, 10:12] = 9      # crosses the ROI edge
    seg[12:15, 12:15] = 300  # inside, id above 255
    out = apply_roi_mask(seg, _roi())
    assert set(np.unique(out)) == {0, 7, 300}
    assert (out[6:9, 6:9] == 7).all() and (out[12:15, 12:15] == 300).all()


def test_roi_on_binary_mask_stays_binary():
    seg = np.zeros((20, 20), np.uint8)
    seg[6:9, 6:9] = 255
    seg[0:2, 0:2] = 255
    out = apply_roi_mask(seg, _roi())
    assert set(np.unique(out)) == {0, 255} and out[7, 7] == 255 and out[0, 0] == 0


def test_roi_none_is_a_no_op_and_shape_mismatch_raises():
    seg = np.ones((5, 5), np.uint16)
    assert apply_roi_mask(seg, None) is seg
    with pytest.raises(ValueError):
        apply_roi_mask(seg, np.ones((6, 6), bool))


def test_omnipose_scale_for_the_twin_recording():
    params = omnipose_scale_params(0.0645)
    assert params["rescale"] == pytest.approx(0.0645 / 0.065)
    assert params["min_size"] == 240            # 1 um2 at 0.0645 um/px
    assert omnipose_scale_params(None) == {}


def test_cache_segments_the_registered_frame_and_passes_pixel_size(monkeypatch):
    seen = {}

    def fake_segment(self, images, mode, progress=None, preprocess=True, pixel_size_um=None):
        seen["frame"] = np.array(images[0]); seen["pixel"] = pixel_size_um
        return [(images[0] > 0).astype(np.uint16)]

    monkeypatch.setattr(sc.SegmentationModels, "segment_images", fake_segment)
    raw = np.zeros((2, 1, 1, 8, 8), np.uint16)
    raw[0, 0, 0, 1, 1] = 1000

    def registered(t, p, c):
        return np.roll(raw[t, p, c], 3, axis=1)   # a 3 px shift stands in for registration

    cache = sc.SegmentationCache(raw, frame_getter=registered, pixel_size_getter=lambda: 0.0645)
    mask = cache.with_model("omnipose_bact_phase")[0, 0, 0]
    assert seen["pixel"] == 0.0645
    assert seen["frame"][1, 4] == 1000 and seen["frame"][1, 1] == 0
    assert mask[1, 4] == 1 and mask.sum() == 1


def test_invalidate_position_forgets_only_that_position(monkeypatch):
    monkeypatch.setattr(sc.SegmentationModels, "segment_images",
                        lambda self, images, mode, **kw: [np.ones_like(images[0], dtype=np.uint16)])
    raw = np.zeros((1, 2, 1, 4, 4), np.uint16)
    cache = sc.SegmentationCache(raw).with_model("omnipose_bact_phase")
    cache[0, 0, 0]; cache[0, 1, 0]
    _, indices = cache.mmap_arrays_idx["omnipose_bact_phase"]
    assert len(indices) == 2
    assert cache.invalidate_position(1) == 1
    assert indices == {(0, 0, 0)}
