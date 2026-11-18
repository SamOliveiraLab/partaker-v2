"""Cell detection: the frame the model sees, and the cells the tool finds on run 2.

The model has to see the contrast-equalised frame, so cells in a dim part of the field are found like cells in a bright
part. On run 2 (E. coli, 30 June 2026) the tool's own path, the mask cache the window uses and the edge rule of the
segmentation service, has to find the cells the Partaker 2 figures were made from (skipped when run 2 is not on this
machine)."""
from pathlib import Path

import numpy as np
import pytest

from nd2_analyzer.analysis.segmentation import segmentation_cache as sc
from nd2_analyzer.analysis.segmentation.segmentation_models import MIN_CELL_AREA_UM2, SegmentationModels, preprocess_image
from nd2_analyzer.analysis.segmentation.segmentation_service import SegmentationService

RUN2 = Path.home() / "Workspace/Datasets/ecoli-run2-june30-2026/mcherry-tl-t24-t72-07-01-2026.nd2"


def rods_under_uneven_light():
    """Dark rods on a background lit five times brighter on the right than on the left."""
    _y, x = np.mgrid[:256, :256]
    light = 0.2 + 0.8 * x / 255.0
    rods = np.ones((256, 256), np.float32)
    for cy in range(24, 256, 40):
        for cx in range(24, 256, 40):
            rods[cy - 3:cy + 4, cx - 10:cx + 11] = 0.7
    return (1000 * light * rods).astype(np.float32)


def rod_contrast(image, cx):
    """Background minus rod, averaged over the rods of one column of the grid."""
    rods = np.mean([image[cy - 2:cy + 3, cx - 8:cx + 9].mean() for cy in range(24, 256, 40)])
    background = np.mean([image[cy + 12:cy + 18, cx - 8:cx + 9].mean() for cy in range(24, 216, 40)])
    return background - rods


def test_rods_in_dim_light_stand_out_like_rods_in_bright_light():
    seen = preprocess_image(rods_under_uneven_light())
    assert seen.dtype == np.float32 and seen.min() >= 0.0 and seen.max() <= 1.0
    dim, bright = rod_contrast(seen, 24), rod_contrast(seen, 224)
    assert dim > 0.7 * bright, f"rods on the dim side have {dim / bright:.2f} of the contrast of rods on the bright side"


@pytest.fixture(scope="module")
def run2():
    if not RUN2.exists():
        pytest.skip("run 2 is not on this machine")
    import nd2
    f = nd2.ND2File(str(RUN2))
    yield f.to_dask(), float(f.voxel_size().x)
    f.close()


# (frame, position, cells of at least 1 um2 in the whole frame): figure5_new/cells_full_frames_24_68h.csv, 17 Nov 2026
@pytest.mark.parametrize("t, p, cells", [(0, 1, 135), (48, 2, 1584)])
def test_run2_cells_are_the_cells_of_the_figures(run2, t, p, cells):
    frames, pixel_um = run2
    cache = sc.SegmentationCache(frames[t:t + 1, p:p + 1, 0:1], pixel_size_getter=lambda: pixel_um)
    labels = np.asarray(cache.with_model(SegmentationModels.OMNIPOSE_BACT_PHASE)[0, 0, 0])
    labels = np.asarray(SegmentationService._remove_labels_with_centroids_near_edge(None, labels, SegmentationService.EDGE_CENTROID_MARGIN_PX))
    areas_um2 = np.bincount(labels.ravel())[1:] * pixel_um ** 2
    assert int((areas_um2 >= MIN_CELL_AREA_UM2).sum()) == cells
