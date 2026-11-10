"""Build the Seeding page test fixtures from the Lee 2018 FC1 mask film.

Two windows of the film are turned into the tool's own tables: the per cell rows that MetricsService makes, and the
lineage tracks that run_tracker makes with btrack. Each window is saved as cells.parquet, tracks.json and meta.json under
~/Workspace/results/partaker2_fixtures/seeding_lee_fc1/{early,late}/.

Run with the partaker-v2 venv, offscreen:
    QT_QPA_PLATFORM=offscreen ~/Workspace/partaker-v2/.venv/bin/python tests/fixtures/build_seeding_fixture.py --window both
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import polars as pl
import tifffile
import zarr

from nd2_analyzer.analysis.metrics_service import MetricsService
from nd2_analyzer.analysis.segmentation.segmentation_models import SegmentationModels
from nd2_analyzer.analysis.tracking.tracking import run_tracker
from nd2_analyzer.data.frame import TLFrame

FILM = Path("/Users/smdoliveira/Workspace/Datasets/external/lee2018/tiff/lee2018_WT_FC1_29.5h.ome.tif")
OUT = Path.home() / "Workspace/results/partaker2_fixtures/seeding_lee_fc1"
WINDOWS = {"early": 0, "late": 33000}
DT_S = 3.0
PIXEL_UM = 0.0654
START_HOUR = 0.0
MASK_VALUE = 1000
FLOW = {"present": True, "shear_mpa": None, "direction": "left to right"}


def read_mask_frames(first: int, count: int, chunk: int = 100) -> np.ndarray:
    """Channel 0 of the film for frames first .. first + count - 1, as one uint16 stack (0 or 1000)."""
    with tifffile.TiffFile(FILM) as tf:
        arr = zarr.open(tf.series[0].aszarr(), mode="r")
        n_t, _, ny, nx = arr.shape
        stop = min(first + count, n_t)
        stack = np.empty((stop - first, ny, nx), np.uint16)
        for a in range(first, stop, chunk):
            b = min(a + chunk, stop)
            stack[a - first:b - first] = np.asarray(arr[a:b, 0])
    return stack


def label_stack_in_place(stack: np.ndarray) -> None:
    """Replace each binary mask frame by its connected component labels (the tool's imported mask rule)."""
    for i in range(stack.shape[0]):
        stack[i] = SegmentationModels.label_imported_masks(stack[i])


def cell_rows(stack: np.ndarray, first: int) -> pl.DataFrame:
    """One MetricsService row per cell per frame, time as the absolute frame index."""
    rows = []
    for i in range(stack.shape[0]):
        lab = stack[i]
        raw = (lab > 0).astype(np.uint16) * MASK_VALUE
        frame = TLFrame(index=(first + i, 0), labeled_phc=lab, mcherry=None, yfp=None, phase=raw)
        rows.extend(MetricsService.calculate_cell_metrics(frame))
    return pl.DataFrame(rows)


def tracks_absolute(stack: np.ndarray, first: int) -> list:
    """btrack lineage tracks over the stack, with t shifted to absolute frame indices and the position added."""
    tracks, _graph = run_tracker(stack, algorithm="btrack")
    out = []
    for tr in tracks:
        out.append({
            "ID": int(tr["ID"]),
            "x": [float(v) for v in tr["x"]],
            "y": [float(v) for v in tr["y"]],
            "t": [int(v) + first for v in tr["t"]],
            "parent": None if tr.get("parent") is None else int(tr["parent"]),
            "children": [int(c) for c in tr.get("children", [])],
            "position": 0,
        })
    return out


def build_window(name: str, count: int, out_dir: Path) -> dict:
    first = WINDOWS[name]
    t0 = time.time()
    stack = read_mask_frames(first, count)
    n_frames_recording = tifffile.TiffFile(FILM).series[0].shape[0]
    print(f"[{name}] read {stack.shape[0]} frames in {time.time() - t0:.0f} s", flush=True)
    label_stack_in_place(stack)
    print(f"[{name}] labelled in {time.time() - t0:.0f} s", flush=True)
    cells = cell_rows(stack, first)
    print(f"[{name}] {cells.height} cell rows in {time.time() - t0:.0f} s", flush=True)
    t1 = time.time()
    tracks = tracks_absolute(stack, first)
    print(f"[{name}] {len(tracks)} tracks in {time.time() - t1:.0f} s", flush=True)
    field_shape = [int(stack.shape[1]), int(stack.shape[2])]
    del stack
    gc.collect()

    out_dir.mkdir(parents=True, exist_ok=True)
    cells.write_parquet(out_dir / "cells.parquet")
    (out_dir / "tracks.json").write_text(json.dumps(tracks))
    meta = {
        "source": str(FILM),
        "window": name,
        "frames": {"first": first, "last": first + count - 1, "count": count},
        "n_frames_recording": int(n_frames_recording),
        "dt_s": DT_S,
        "pixel_um": PIXEL_UM,
        "start_hour": START_HOUR,
        "field_shape": field_shape,
        "flow": FLOW,
        "tracker": "btrack",
        "n_cell_rows": int(cells.height),
        "n_tracks": len(tracks),
        "build_seconds": round(time.time() - t0),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{name}] saved to {out_dir} in {meta['build_seconds']} s", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(description="Build the Lee FC1 seeding fixtures.")
    ap.add_argument("--window", choices=["early", "late", "both"], default="both")
    ap.add_argument("--frames", type=int, default=2400, help="frames per window")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    names = ["early", "late"] if args.window == "both" else [args.window]
    for name in names:
        build_window(name, args.frames, args.out / name)


if __name__ == "__main__":
    main()
