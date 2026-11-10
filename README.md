# Partaker 2

Analyze your time lapse data straight from the microscope!

## Quick start

```sh
git clone https://github.com/SamOliveiraLab/partaker-v2.git
cd partaker-v2
uv run gui
```

Python 3.10 or 3.11. The first run installs everything. At start, pick the mode: Single Cell Analysis or Biofilm Cloud Analysis.

## Digital twin (Vivarium)

- Viva-munk, the twin simulator: https://github.com/vivarium-collective/Viva-munk (lab copy: https://github.com/SamOliveiraLab/Viva-munk)
- Viva-munk documentation: https://vivarium-collective.github.io/Viva-munk/
- process-bigraph, the Vivarium engine it runs on: https://github.com/vivarium-collective/process-bigraph

## Data used for the digital twin

- Raw recording: `/Volumes/SAM1/Oliveira_lab_workspace/Oliveira_lab_workspace/Bukola/Bukola_Partaker_New/partaker_dataset_DT/1-SR_1_5_6hPre-C_PlainM9_TS_MC1.nd2`
- Part used: file 1, position 0, frames 0 to 27 (5 min per frame, 135 min), 0.0645 µm per pixel, registered, chamber ROI.
- Tracking export: `/Users/smdoliveira/Workspace/results/for_tolulope_comsol/enhanced_tracking_data_all_tracks_registered_p0_f0-27.csv`

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone this repo and `cd` into it.
3. `uv run gui`

uv's cache is configured in `pyproject.toml` to sit inside the repo (`.uv-cache/`), so installs stay fast on any filesystem — including external drives — without drive-specific config.

Optional: `python -m nd2_analyzer` or `python src/main.py` after the first run (uses the same env).

For Cellpose + MPS notes, see [MouseLand/cellpose#1063](https://github.com/MouseLand/cellpose/issues/1063).

## Motion fields

PIV runs separately from the application's NumPy 1.x/TensorFlow environment.
Create its environment once from the project directory (Python 3.11 or newer):

```sh
uv venv .venv-piv --python 3.12
uv pip install --python .venv-piv/bin/python -r requirements-piv.txt
```

On Windows use `.venv-piv/Scripts/python.exe` for the Python path. Do not install
`requirements-piv.txt` into the main `.venv`. The Motion fields controls allow
selecting another compatible PIV Python. OpenPIV is pinned to 0.26.1.

Accept the microcolony time series and its **Inside ROI** masks, choose the T start
and T end, and click **Export Motion Fields**. This action does not require
single-cell segmentation. Uncheck inner-region outputs to export only the two
whole-colony workflows. Accepted positions/channels in the time range are retained.
The existing colony/cell export remains available separately.

Each colony receives four sets: grayscale/all births, bright segmentation/all
births, grayscale/inner births, and bright segmentation/inner births. An inner
mask controls where a path can start at each capture; a continuing path may leave
that mask but must remain supported inside the colony. Multiple inner components
form one birth zone. Empty inner masks never fall back to whole-colony seeding.
The selector's dark-region inner mask is used exactly as accepted; it is not
inverted to match the separate bright-area tracking mask.

Each set contains `motion_accumulating.gif`, `motion_current_frame.gif`,
`motion_field.png`, `vectors.csv`, `path_segments.csv`, `path_births.csv`, and
metadata. Four sets give **20 motion/source GIFs and 4 field graphs per colony**, plus the density outputs described below. Current-frame movies
show only the current pair's displacement, including continuing and new paths;
previous intervals are cleared. Graphs show time-colored accumulating paths.
Solid segments have full reverse-confirmed local support; dotted segments use
partial or reverse-unconfirmed support. This does not validate accumulated paths
or establish individual-cell identity or polymer concentration.
`vectors.csv` contains the supporting colony field; `birth_zone_eligible` marks
the selected birth zone, and actual new paths are listed in `path_births.csv`.
Plot time is elapsed from the first selected capture.

Outputs are saved in `analysis_results/microcolony_piv/<run-id>/`, with separate
`position_<p>_channel_<c>_colony_<track-id>/` folders. Input arrays, colony/inner/
bright TIFF masks, the launch manifest, worker logs, timing, intensity conversion,
crop, registration offsets, failures, and version metadata are retained. Cancelled
or failed runs are marked in `status.json`; partial results are not marked complete.

Grayscale tracking has no blur. Bright masks use sigma=1 then threshold160 by
default, without removing isolated bright regions. For non-8-bit images one fixed
min/max conversion is used across all selected captures of a position/channel.
Window presets are 128→64→32, 256→128→64→32, and 384→256→128→64→32. Unsupported
source dimensions are rejected rather than silently padded. Coarse correlation
uses real full-frame context; final32px windows must fit the colony. Source images
and selection masks are never zeroed before correlation. The analysis border is15px.

Forward checks retain SNR≥1.2, texture SD≥8, finite estimates and OpenPIV outlier
validation. Measurable reverse disagreements>2px are rejected; unavailable reverse
support remains exploratory. Path interpolation needs≥75% accepted bilinear
weight, with no nearest fallback or reconnection. Missing capture indices break
paths. Failed deformation predictors produce logged gaps, not invented vectors.
Capture metadata supplies timing when enabled; otherwise the stated fallback
interval applies. GIF playback duration is not measurement time. Physical rates
are exported only when spatial calibration exists; time units remain explicit.

Coordinates are in the registered analysis image before any rendering crop.
To map back to acquired pixels: analysis coordinate + crop origin − registration
shift, using the per-frame transform in the manifest. Images and masks must use
the same crop/registration; changing either during preparation cancels the run.

Run the focused tests in the PIV environment:

```sh
.venv-piv/bin/python -m unittest discover -s tests -p test_piv_motion.py
```

### Standalone movies, channel zoom, and density

The paired image/field movies remain available. Each workflow also exports
`source_images.gif`, `motion_accumulating_only.gif`, and
`motion_current_frame_only.gif` as standalone views. Source movies match the
motion interval endpoints; density movies include every capture, including the first.
Four workflows now produce 20 motion/source GIFs and four field graphs.

**Use notebook channel zoom** is on by default. It uses x=220, y=180, width=360,
height=280 in registered analysis-image pixels for the static field graph's zoom
panel. The box is clipped to image dimensions. If it misses the colony entirely,
the graph explicitly reports an automatic fallback; metadata records the actual
bounds and selection reason. Disable the toggle to use automatic zoom.

**Also export density heatmaps and time plots** is on by default, with a 32-pixel
square neighborhood (adjustable). Results go into each colony's
`density/whole_colony/` and, when inner outputs are enabled, `density/inside_roi/`:

- `raw_density_heatmap.gif`: local mean darkness, 1 − grayscale/255, with no segmentation.
- `segmented_density_heatmap.gif`: local fraction of bright segmented pixels.
- `density_over_time.gif` and `density_over_time.png`: raw darkness, bright-area
  fraction, connected bright-patch count, and mean nearest-patch centroid spacing.
- `density_over_time.csv`: per-capture summaries, ROI area, mean gray intensity,
  mean bright-patch area, and spacing; `metadata.json` records definitions.

Both heatmaps use fixed 0–1 scales across time and normalize each neighborhood
by its selected ROI pixel count, excluding outside pixels rather than counting
those as dark. Gray indicates outside the ROI. Empty selections are undefined,
not zero density; centroid spacing is undefined for fewer than two patches.
Bright patches use 8-connectivity with no small-region removal. Centroid spacing
is not the distance between patch edges. Inner density is evaluated inside the
accepted inner mask at each capture, independently of PIV path births or support.
These are image-pattern proxies: raw darkness can change with illumination, and
bright space is not a confirmed cell or γ-PGA measurement. Changing ROI area can
also change the summaries. Local averaging for heatmaps does not blur PIV inputs.
