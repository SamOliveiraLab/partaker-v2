"""Run 2, six positions every 1.5 h from 24 to 57 h, from the tool alone: the masks the tool segmented and saved with the
session give the microcolony metrics of Figure 3 (panels e to k, n, q; the numbers of its cache tables, 16 Nov 2026).
Takes about two minutes: runs only with --run-slow, and is skipped when the saved session is not on this machine."""
import json
from pathlib import Path

import numpy as np
import pytest

from nd2_analyzer.analysis.biofilm_stages.microcolony_metrics import measure_metrics

SESSION_MASKS = Path.home() / "Workspace/results/partaker2_sessions/run2_ecoli/segmentation_cache.h5"     # saved from the tool's window
KEPT = (1, 2, 3, 6, 8, 9)
PIXEL_UM = 0.13 / 3


@pytest.mark.slow
def test_run2_microcolony_metrics_from_the_tool_alone():
    if not SESSION_MASKS.exists():
        pytest.skip("the saved run 2 session is not on this machine")
    import h5py
    with h5py.File(SESSION_MASKS, "r") as f:
        group = f["mmap_arrays/omnipose_bact_phase"]; stored = group["frames"]
        row = {(int(t), int(p)): k for k, (t, p, _c) in enumerate(json.loads(group.attrs["index_set"]))}
        out = measure_metrics(lambda p, t: stored[row[(t, p)]], {p: list(range(0, 67, 3)) for p in KEPT}, PIXEL_UM,
                              hour_of=lambda t: 24.0 + 0.5 * t, random_hours=(24.0, 27.0, 30.0, 33.0))

    def at(what, wanted, **how):
        hours, mean, _sd = out.mean_of(what, **how)
        return [float(mean[np.argmin(np.abs(hours - h))]) for h in wanted]

    assert at("microcolonies", (24, 33, 57)) == pytest.approx([9.0, 19.67, 5.17], abs=0.01)                               # e
    hours, count, _sd = out.mean_of("microcolonies"); assert hours[int(np.argmax(count))] == 33.0
    assert at("cells_per_microcolony", (24, 33, 45, 57)) == pytest.approx([15.93, 27.43, 442.24, 512.54], abs=0.01)       # f, mean of the positions
    assert at("cells_per_microcolony", (24, 33, 45, 57), geometric=True) == pytest.approx([14.3, 24.9, 334.7, 403.5], abs=0.05)   # f, as drawn
    assert at("cells_in_pct", (24, 33, 45, 57)) == pytest.approx([64.25, 87.67, 97.37, 97.96], abs=0.01)                  # g
    # g, random placement. The figure's 3.6, 9.2, 18.2, 32.7 % came from other random draws (about 0.6 points either way)
    # and from a script that shaved 0.7 % off every cell when it turned it; with whole cells 1 to 2 points more link up
    assert at("random_pct", (24, 27, 30, 33)) == pytest.approx([3.0, 9.3, 19.3, 35.4], abs=0.05)                         # the tool's own draws (fixed seed)
    assert at("random_pct", (24, 27, 30, 33)) == pytest.approx([3.6, 9.2, 18.2, 32.7], abs=3.0)                          # the figure
    assert at("density_in", (24, 33, 45, 57)) == pytest.approx([10.97, 13.11, 20.59, 24.30], abs=0.01)                    # h
    assert at("density_single", (24, 33, 45, 57)) == pytest.approx([3.115, 4.12, 8.61, 11.755], abs=0.006)
    assert at("spacing_um", (24, 33, 39)) == pytest.approx([19.93, 12.90, 15.98], abs=0.01)                               # i
    assert at("alignment", (24, 33, 45, 57)) == pytest.approx([0.496, 0.392, 0.327, 0.276], abs=0.001)                    # j
    assert at("largest_share_pct", (24, 33, 39, 45, 57)) == pytest.approx([35.60, 32.40, 79.68, 94.41, 97.19], abs=0.01)  # k
    assert at("area_pct", (24, 33, 45, 57)) == pytest.approx([10.216, 30.408, 61.904, 69.580], abs=0.001)                 # q
    ring_hours, edges, density = out.radial()                                                                             # n
    first, last = int(np.argmin(np.abs(ring_hours - 24.0))), int(np.argmin(np.abs(ring_hours - 57.0)))
    assert edges[:3].tolist() == [0.0, 2.0, 4.0] and density[first, :4] == pytest.approx([13.22, 11.98, 12.02, 10.97], abs=0.01)
    assert np.isnan(density[first, 8]) and density[last, [0, 5, 11]] == pytest.approx([20.61, 23.2, 25.17], abs=0.01)
