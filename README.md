# Partaker 2

Analyze your time lapse data straight from the microscope!

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

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
