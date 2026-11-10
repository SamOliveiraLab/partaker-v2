"""The data the Seeding page and the seeding rules share. Plain dataclasses, no Qt, no numbers of their own: every value
is filled by the rules module from the loaded recording. Field names follow the seeding page contract and must stay as they are,
the rules module and the page both import from here."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import polars as pl


@dataclass
class SeedingSettings:
    """Every threshold the papers left undefined, with its default and a one-line meaning."""
    edge_margin_px: int = 10                      # a track that starts or ends within this margin entered or left the field
    stay_window_s: float = 30.0                   # "leaves within 30 s" (Lee Fig. 1B)
    daughter_stay_s: float = 300.0                # how long a daughter must stay to count as stayed (Lee leaves it undefined)
    irreversible_stay_s: float = 3600.0           # a landed cell still there at the end after this long (or one that divides) attached for good
    motile_fraction_of_minor_axis: float = 0.5    # net shift above this fraction of the minor axis = motile (Lecuyer Fig. 3C)
    on_end_length_over_width: float = 1.5         # projected length / width below this = standing on one pole (Conrad Fig. 3A)
    divider_msd_um2: float = 0.25                 # final centroid MSD above this = divider (Hubert Methods)
    holder_min_track_s: float = 2400.0            # a non-divider must be tracked at least this long (Hubert: 40 min)
    microcolony_cells: int = 4                    # handover rule: a group of at least this many cells
    microcolony_gap_um: float = 1.0               # within this gap
    # thresholds the rule texts name beyond the contract's nine, each with its default and meaning
    microcolony_persist_frames: int = 10          # the group must be there this many frames in a row before the handover
    crawl_step_px: float = 0.5                    # mean step per frame above this = moving (rule "crawl, then stop")
    crawl_min_frames: int = 20                    # a track must be this long for the crawl rule
    motile_min_frames: int = 3                    # a track must be this long for the motile rule
    max_division_time_h: float = 5.0              # a longer gap between divisions is a missed division (Lee Fig. S1A)
    align_angle_deg: float = 30.0                 # a cell within this angle of the flow axis is aligned (Lecuyer Methods)
    divider_max_dt_s: float = 120.0               # the divider rule needs frames at least this close (Hubert: 1 min)
    min_fit_frames: int = 20                      # the lag and growth fits need this many frames
    min_survival_leavers: int = 50                # the survival curve needs this many leavers (Lecuyer Eq. 3)
    min_divisions_scored: int = 5                 # the division outcome needs this many scored divisions
    join_radius_px: float = 3.0                   # a track point claims the cell row within this distance
    rising_fraction: float = 0.1                  # the count must rise from the first to the last tenth of the frames
    min_flat_phase_taus: float = 0.5              # the flat phase before the rise must last this many e-fold times inside the frames for the lag to count
    # whether the frames show seeding at all: signs that hold for any species and frame rate
    not_seeding_start_fraction: float = 0.25      # the first frame holds this share of the most cells seen: the surface was not clean
    not_seeding_start_cells: int = 20             # and at least this many cells (a few early cells are normal)
    not_seeding_signs: int = 2                    # this many signs make the page ask whether the recording shows seeding


@dataclass
class Metric:
    """One number on the page, with where it came from and why it may be missing."""
    key: str                          # stable id, snake_case
    name: str                         # plain words, sentence case, short ("Cells on the surface")
    value: Optional[float]            # the headline number, None when unavailable
    unit: str                         # "", "%", "h", "s", "cells", "1/h", "um"
    text: str                         # the headline as it is shown, e.g. "1,346 landings" or "96.8 % leave within 30 s"
    available: bool
    reason: str                       # why not available, one short line ("frames 1800 s apart, the rule needs under 10 s")
    source: str                       # paper and figure, e.g. "Lee 2018 Fig. 1B"
    group: str                        # "arrivals" | "movement" | "division" | "flow"
    series: Optional[dict] = None     # arrays for the plots, keys documented per rule in the contract
    n: Optional[int] = None           # sample size behind the number


@dataclass
class SeedingResults:
    """What measure_seeding returns: the metrics and the tables behind them."""
    metrics: list                     # list[Metric]
    per_track: pl.DataFrame           # one row per track: track_id, frame_start, frame_end, t_start_h, t_end_h, residence_s,
                                      # origin, fate, path_um, net_disp_um, motile, pose_first, n_children, generation,
                                      # family_id, divider, along_flow_um
    counts: pl.DataFrame              # one row per frame: frame, hour, cells, mask_px (or area_px), empty (bool)
    candidates: pl.DataFrame          # tracks worth following in the film: track_id, why, score (and the frame span when known)
    stage_by_frame: pl.DataFrame      # frame, hour, stage in {"seeding", "microcolonies"} by the handover rule
    settings: SeedingSettings
    notes: list = field(default_factory=list)   # plain-word provenance lines
    check: list = field(default_factory=list)   # plain-word signs that the frames do not show seeding; empty when they do

    def metric(self, key: str) -> Optional[Metric]:
        """The metric with this key, or None."""
        for m in self.metrics:
            if m.key == key:
                return m
        return None


@dataclass
class RecordingInfo:
    """What the page knows about the loaded recording before any measurement."""
    name: str
    n_frames: int
    dt_s: float                       # seconds per frame
    pixel_um: Optional[float]
    start_hour: float                 # hours after inoculation at frame 0
    flow: Optional[dict]              # the experiment's flow block or None
    channel_roles: dict = field(default_factory=dict)      # {"0": "phase" | "cell_marker" | "matrix" | "other", ...}
    field_shape: Optional[tuple] = None                    # (Y, X) pixels

    @property
    def end_hour(self) -> float:
        return float(self.start_hour) + float(self.n_frames) * float(self.dt_s) / 3600.0

    @property
    def span(self) -> tuple:
        return (float(self.start_hour), self.end_hour)

    @property
    def has_flow(self) -> bool:
        return bool(self.flow) and bool(self.flow.get("present", False))
