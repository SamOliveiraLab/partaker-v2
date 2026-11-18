"""Run 2, six positions, from the tool alone: the masks the tool segmented and saved with the session, the raw dye channel
and the two dye frames give the numbers of Figures 4 and 5 (the handoff table of 17 Nov 2026). Takes about two minutes:
runs only with --run-slow, and is skipped when run 2 or its saved session is not on this machine."""
import json
from pathlib import Path

import numpy as np
import pytest

from nd2_analyzer.analysis.biofilm_stages import growth_states as GS
from nd2_analyzer.analysis.biofilm_stages import matrix_stain as MX
from nd2_analyzer.analysis.biofilm_stages.growth_measure import measure_growth_states

RUN2 = Path.home() / "Workspace/Datasets/ecoli-run2-june30-2026"
MAIN, BEFORE, AFTER = RUN2 / "mcherry-tl-t24-t72-07-01-2026.nd2", RUN2 / "extra_3_24h/mcherry-tl-t5-b4-dye06-30-2027.nd2", RUN2 / "extra_3_24h/mcherry-tl-t5-06-30-2028.nd2"
SESSION_MASKS = Path.home() / "Workspace/results/partaker2_sessions/run2_ecoli/segmentation_cache.h5"     # saved from the tool's window
KEPT = (1, 2, 3, 6, 8, 9)
OUT_OF_FOCUS = range(69, 77)                               # 58.5 to 62 h after inoculation


@pytest.mark.slow
def test_run2_growth_states_from_the_tool_alone():
    if not (MAIN.exists() and BEFORE.exists() and AFTER.exists() and SESSION_MASKS.exists()):
        pytest.skip("run 2, its dye frames or its saved session are not on this machine")
    import h5py, nd2
    from nd2_analyzer.ui.biofilms.stages.growth_controller import frames_to_measure
    with h5py.File(SESSION_MASKS, "r") as f, nd2.ND2File(str(MAIN)) as rec:
        group = f["mmap_arrays/omnipose_bact_phase"]; stored = group["frames"]
        row = {(int(t), int(p)): k for k, (t, p, _c) in enumerate(json.loads(group.attrs["index_set"]))}
        data = rec.to_dask(); pixel_um = float(rec.voxel_size().x)
        frames = {p: frames_to_measure([t for (t, q) in row if q == p], step=3, skip=lambda t: t in OUT_OF_FOCUS) for p in KEPT}
        out = measure_growth_states(lambda p, t: stored[row[(t, p)]], lambda p, t: np.asarray(data[t, p, 2]), frames, pixel_um,
                                    hour_of=lambda t: 24.0 + 0.5 * t, reference=MX.dye_reference(MX.read_dye_frames(BEFORE, 2), MX.read_dye_frames(AFTER, 2)),
                                    matrix_frames={p: [t for t in frames[p] if t < OUT_OF_FOCUS[0]] for p in KEPT})
    at = lambda hours, values, h: float(values[np.argmin(np.abs(hours - h))])
    hours, cells, _sd = out.mean_of("cells"); _h, rate, _s = out.mean_of("rate")
    assert [at(hours, cells, h) for h in (24.0, 36.0, 48.0, 57.0, 68.0)] == pytest.approx([214, 773, 1509, 1788, 2244], abs=0.5)
    assert hours[int(np.argmax(rate))] == 37.5                                                    # fastest growth
    assert out.th2["th2_h"] == pytest.approx(90.3, abs=0.05) and out.th2["range95_h"] == pytest.approx((81.1, 108.5), abs=0.1) and not out.th2["reached"]
    mh, matrix, _m = out.mean_of("matrix")
    assert mh[-1] == 57.0 and [at(mh, matrix, h) for h in (24.0, 30.0, 39.0, 42.0, 48.0, 57.0)] == pytest.approx([0.0, 0.01, 0.43, 1.37, 8.11, 34.53], abs=0.01)
    assert out.th1["per_position_h"] == {1: 37.5, 2: 33.0, 3: 31.5, 6: 31.5, 8: 39.0, 9: 33.0}
    assert round(out.th1["th1_h"], 1) == 34.2 and round(out.th1["sd_h"], 1) == 3.2
    _a, inside, _b = out.mean_of("inside"); _c, outside, _d = out.mean_of("outside")
    assert (at(mh, inside, 57.0), at(mh, outside, 57.0)) == pytest.approx((38.1, 26.0), abs=0.05)
    assert np.mean(list(out.free_dye.values())) == pytest.approx(2030, abs=1)
