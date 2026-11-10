"""Measure dispersion from a low magnification phase contrast recording: cells leaving a biofilm.

Single cells cannot be resolved at this magnification, so everything is measured on dark objects (a dark object can hold
one cell or several). Three stand-ins for single-cell numbers:
  cells (estimated)   dark area divided by the area of one free object (the median free object in the sharp frames)
  stays in place      dark area that is still dark at the same place in the next frame
  seen once           dark area that is not there in the next frame

The steps: the biofilm outline in every frame (local texture, Otsu, closing), the dark objects in every frame (pixels
more than dark_z local standard deviations darker than their surroundings, groups of at least min_object_px), and three
zones for the dark area: inside the starting outline, in the cloud (inside the current outline, beyond the starting one)
and free (beyond the current outline). The starting outline is the outline of the first sharp frame and stays fixed.

Pure numpy, scipy and scikit-image on a (T, Y, X) stack, so the page, the tests and a script measure the same thing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import polars as pl
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import filters, measure, morphology

INSIDE, CLOUD, FREE = 1, 2, 3                 # zone codes of the dark area


@dataclass
class DispersionSettings:
    edge_px: int = 15                         # frame margin left out (filter edge effects)
    outline_sigma: float = 4.0                # the texture scale of the biofilm body
    outline_close_px: int = 7
    outline_open_px: int = 5
    outline_min_area_px: int = 3000
    dark_z: float = 0.8                       # local standard deviations darker than the surroundings
    min_object_px: int = 4
    focus_fraction: float = 0.5               # a frame is out of focus below this share of the median sharpness
    stay_overlap: float = 0.5                 # an object stays when this share of it is dark again in the next frame
    ring_px: float = 10.0                     # width of the distance rings
    max_distance_px: float = 250.0
    exit_reach_px: float = 120.0              # dark area this close to the outline counts for the stretch it left from
    stretches: int = 60
    pixel_um: Optional[float] = None          # None keeps every length in pixels


@dataclass
class DispersionOutline:
    """The first step on its own: sharpness of every frame and the starting outline."""
    sharpness: np.ndarray
    in_focus: np.ndarray
    ref_frame: int
    start_mask: np.ndarray                    # (Y, X) bool, full frame


@dataclass
class DispersionResults:
    settings: DispersionSettings
    hours: np.ndarray
    shape: tuple                              # (Y, X) of the full frame
    ref_frame: int
    start_mask: np.ndarray                    # (Y, X) bool
    start_edge: np.ndarray                    # (Y, X) bool, the outline one pixel wide
    body_edges: np.ndarray                    # (T, Y, X) bool, the outline of every frame
    zone: np.ndarray                          # (T, Y, X) uint8: 0, INSIDE, CLOUD, FREE
    per_frame: pl.DataFrame
    objects: pl.DataFrame                     # every dark object beyond the starting outline
    cell_area_px: float
    ring_edges: np.ndarray
    distance_map: np.ndarray                  # (rings, T) share of each ring that is dark
    outline_yx: np.ndarray                    # (N, 2) the starting outline, full frame coordinates
    outline_arc_px: np.ndarray                # (N,) distance along it
    stretch_edges: np.ndarray                 # (stretches + 1,) vertex index where each stretch starts
    exit_map: np.ndarray                      # (stretches, T) dark area that left from each stretch
    occupancy: np.ndarray                     # (Y, X) share of the sharp frames with a dark object at this pixel
    stretch_px: np.ndarray = field(default_factory=lambda: np.zeros(0))   # (stretches,) length of each stretch
    free_areas_px: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def span(self) -> tuple:
        return float(self.hours[0]), float(self.hours[-1])

    @property
    def in_focus(self) -> np.ndarray:
        return self.per_frame["in_focus"].to_numpy()


# ---------------------------------------------------------------------------------------------- one frame
def to_gray(frame) -> np.ndarray:
    a = np.asarray(frame)
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.float32)


def crop(a: np.ndarray, edge: int) -> np.ndarray:
    return a[edge:a.shape[0] - edge, edge:a.shape[1] - edge] if edge > 0 else a


def pad_back(a: np.ndarray, edge: int, shape: tuple) -> np.ndarray:
    """A cropped map back in the full frame, the margin empty."""
    if edge <= 0:
        return a
    out = np.zeros(shape, a.dtype)
    out[edge:shape[0] - edge, edge:shape[1] - edge] = a
    return out


def sharpness(g: np.ndarray) -> float:
    """Variance of the Laplacian: high for a sharp frame, low for a blurred one."""
    return float(ndi.laplace(g.astype(np.float32)).var())


def focus_flags(sharp: Sequence[float], fraction: float) -> np.ndarray:
    s = np.asarray(sharp, float)
    return s >= fraction * np.median(s)


def z_map(g: np.ndarray) -> np.ndarray:
    """How many local standard deviations each pixel is darker than its surroundings."""
    a = ndi.gaussian_filter(g.astype(float), 0.6); mu = ndi.gaussian_filter(a, 7)
    sd = np.sqrt(np.maximum(0, ndi.gaussian_filter(a * a, 7) - mu * mu))
    return (mu - a) / np.maximum(sd, 5)


def dark_objects(g: np.ndarray, s: DispersionSettings) -> np.ndarray:
    """Labels of the dark objects of one frame."""
    eight = np.ones((3, 3))
    lab, _ = ndi.label(z_map(g) > s.dark_z, structure=eight)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    lab, _ = ndi.label(sizes[lab] >= s.min_object_px, structure=eight)
    return lab


def fill_holes(m: np.ndarray, gap: int = 41) -> np.ndarray:
    """Holes filled. A biofilm that runs off the frame has holes that open to the frame border: those are filled too
    when the opening is narrower than gap pixels."""
    p = np.pad(m, 1)
    line = np.ones(gap, bool)
    for index, side in (((0, slice(1, -1)), m[0]), ((-1, slice(1, -1)), m[-1]), ((slice(1, -1), 0), m[:, 0]),
                        ((slice(1, -1), -1), m[:, -1])):
        p[index] = ndi.binary_erosion(ndi.binary_dilation(side, line), line, border_value=1)
    return ndi.binary_fill_holes(p)[1:-1, 1:-1]


def body_mask(g: np.ndarray, s: DispersionSettings) -> np.ndarray:
    """The biofilm body: where the image is textured, closed into one piece."""
    a = g.astype(float)
    mu = ndi.gaussian_filter(a, s.outline_sigma)
    sd = np.sqrt(np.maximum(0, ndi.gaussian_filter(a * a, s.outline_sigma) - mu * mu))
    m = sd > filters.threshold_otsu(sd)
    m = morphology.binary_closing(m, morphology.disk(s.outline_close_px))
    m = fill_holes(m)
    m = morphology.binary_opening(m, morphology.disk(s.outline_open_px))
    return morphology.remove_small_objects(m, s.outline_min_area_px)


def edge_of(mask: np.ndarray, width: int = 1) -> np.ndarray:
    """The outline of a mask, width pixels wide. The frame border is not an outline."""
    e = mask & ~ndi.binary_erosion(mask, border_value=1)
    return ndi.binary_dilation(e, iterations=width - 1) if width > 1 else e


# ---------------------------------------------------------------------------------------------- the two steps
def find_outline(frames, settings: Optional[DispersionSettings] = None, progress: Optional[Callable] = None,
                 cancelled: Optional[Callable] = None) -> Optional[DispersionOutline]:
    """Step one: how sharp each frame is, and the biofilm outline of the first sharp frame."""
    s = settings or DispersionSettings()
    n = len(frames); sharp = np.zeros(n)
    for i in range(n):
        if cancelled and cancelled():
            return None
        sharp[i] = sharpness(crop(to_gray(frames[i]), s.edge_px))
        if progress:
            progress(int(90 * (i + 1) / n), f"frame {i + 1} of {n}")
    ok = focus_flags(sharp, s.focus_fraction)
    ref = int(np.argmax(ok))
    g = to_gray(frames[ref])
    return DispersionOutline(sharp, ok, ref, pad_back(body_mask(crop(g, s.edge_px), s), s.edge_px, g.shape))


def measure_dispersion(frames, hours: Sequence[float], settings: Optional[DispersionSettings] = None,
                       progress: Optional[Callable] = None, cancelled: Optional[Callable] = None,
                       outline: Optional[DispersionOutline] = None) -> Optional[DispersionResults]:
    """Step two: every number of the page. frames is (T, Y, X) (or anything indexable frame by frame), hours the hour
    of each frame. Returns None when cancelled."""
    s = settings or DispersionSettings()
    hours = np.asarray(hours, float); n = len(frames); e = s.edge_px
    outline = outline or find_outline(frames, s, cancelled=cancelled)
    if outline is None:
        return None
    full = to_gray(frames[0]).shape
    m0 = crop(outline.start_mask, e); dom = ~m0
    dist, nearest = ndi.distance_transform_edt(dom, return_indices=True)
    labs, bodies = [], []
    for i in range(n):
        if cancelled and cancelled():
            return None
        g = crop(to_gray(frames[i]), e)
        labs.append(dark_objects(g, s)); bodies.append(body_mask(g, s))
        if progress:
            progress(int(80 * (i + 1) / n), f"frame {i + 1} of {n}")
    darks = [l > 0 for l in labs]
    ok = outline.in_focus

    rows, obj_rows, zones, edges = [], [], [], []
    for i in range(n):
        d, b, lab = darks[i], bodies[i], labs[i]
        cloud = b & dom; free = dom & ~b
        zone = np.zeros(d.shape, np.uint8)
        zone[d & m0] = INSIDE; zone[d & cloud] = CLOUD; zone[d & free] = FREE
        zones.append(pad_back(zone, e, full)); edges.append(pad_back(edge_of(b), e, full))
        ids = np.arange(1, lab.max() + 1)
        area = np.bincount(lab.ravel(), minlength=len(ids) + 1)[1:]
        cy, cx = (np.array(v) for v in zip(*ndi.center_of_mass(d, lab, ids))) if len(ids) else (np.zeros(0), np.zeros(0))
        iy, ix = cy.astype(int), cx.astype(int)
        in_start = m0[iy, ix]; in_body = b[iy, ix]
        ozone = np.where(in_start, INSIDE, np.where(in_body, CLOUD, FREE))
        out = d & dom; nxt = darks[i + 1] & dom if i + 1 < n else None
        if nxt is not None:
            kept = ndi.sum(nxt, lab, ids) / np.maximum(area, 1)
            stays = kept >= s.stay_overlap
            stay_px = int((out & nxt).sum()); chance = float(nxt.sum() / max(dom.sum(), 1))
        else:
            stays = np.zeros(len(ids), bool); stay_px = None; chance = None
        outside = ~in_start
        out_px = int(out.sum()); dd = dist[out]
        rows.append({
            "frame": i, "hour": float(hours[i]), "sharpness": float(outline.sharpness[i]), "in_focus": bool(ok[i]),
            "pair_in_focus": bool(ok[i] and i + 1 < n and ok[i + 1]),
            "body_area_px": int(b.sum()), "cloud_area_px": int(cloud.sum()),
            "inside_dark_px": int((d & m0).sum()), "outside_dark_px": out_px,
            "cloud_dark_px": int((d & cloud).sum()), "free_dark_px": int((d & free).sum()),
            "objects_in_body": int(in_body.sum()), "objects_beyond_body": int((~in_body).sum()),
            "objects_outside": int(outside.sum()), "objects_cloud": int((ozone == CLOUD).sum()),
            "objects_free": int((ozone == FREE).sum()),
            "distance_median_px": float(np.median(dd)) if out_px else None,
            "distance_p90_px": float(np.percentile(dd, 90)) if out_px else None,
            "stay_px": stay_px, "moved_px": None if stay_px is None else out_px - stay_px,
            "stay_fraction": None if stay_px is None or not out_px else stay_px / out_px,
            "chance_fraction": chance,
            "objects_staying_fraction": float(stays[outside].mean()) if nxt is not None and outside.any() else None,
        })
        for k in np.flatnonzero(outside):
            obj_rows.append((i, int(ids[k]), float(cy[k] + e), float(cx[k] + e), int(area[k]), int(ozone[k]),
                             float(dist[iy[k], ix[k]]), bool(stays[k]) if nxt is not None else None))
        if progress:
            progress(80 + int(15 * (i + 1) / n), f"zones {i + 1} of {n}")

    per_frame = pl.DataFrame(rows)
    objects = pl.DataFrame(obj_rows, orient="row",
                           schema={"frame": pl.Int64, "id": pl.Int64, "y": pl.Float64, "x": pl.Float64,
                                   "area_px": pl.Int64, "zone": pl.Int64, "distance_px": pl.Float64, "stays": pl.Boolean})
    sharp_frames = np.flatnonzero(ok).tolist()
    free = objects.filter((pl.col("zone") == FREE) & pl.col("frame").is_in(sharp_frames))["area_px"].to_numpy()
    cell = float(np.median(free)) if len(free) else float(s.min_object_px)
    per_frame = per_frame.with_columns(
        (pl.col("outside_dark_px") / cell).alias("cells_outside"), (pl.col("cloud_dark_px") / cell).alias("cells_cloud"),
        (pl.col("free_dark_px") / cell).alias("cells_free"), (pl.col("stay_px") / cell).alias("cells_staying"),
        (pl.col("moved_px") / cell).alias("cells_seen_once"))

    ring_edges = np.arange(0.0, s.max_distance_px + s.ring_px, s.ring_px)
    ring_area = np.histogram(dist[dom], ring_edges)[0]
    dmap = np.array([np.histogram(dist[d & dom], ring_edges)[0] / np.maximum(ring_area, 1) for d in darks]).T

    contours = measure.find_contours(m0.astype(float), 0.5)
    c = max(contours, key=len) if contours else np.zeros((2, 2))
    arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
    seg = np.linspace(0, len(c), s.stretches + 1).astype(int)
    tree = cKDTree(c); exits = []
    for d in darks:
        yy, xx = np.nonzero(d & dom & (dist < s.exit_reach_px))
        if len(yy):
            _, k = tree.query(np.c_[nearest[0][yy, xx], nearest[1][yy, xx]])
            exits.append(np.histogram(k, seg)[0])
        else:
            exits.append(np.zeros(s.stretches, int))
    occ = np.mean([darks[i] for i in sharp_frames], axis=0) * dom if sharp_frames else np.zeros(m0.shape)
    if progress:
        progress(100, "done")
    return DispersionResults(
        settings=s, hours=hours, shape=full, ref_frame=outline.ref_frame, start_mask=outline.start_mask,
        start_edge=pad_back(edge_of(m0), e, full), body_edges=np.array(edges), zone=np.array(zones),
        per_frame=per_frame, objects=objects, cell_area_px=cell, ring_edges=ring_edges, distance_map=dmap,
        outline_yx=c + e, outline_arc_px=arc, stretch_edges=seg, exit_map=np.array(exits).T,
        occupancy=pad_back(occ.astype(np.float32), e, full), stretch_px=np.diff(np.r_[arc, arc[-1]][seg]),
        free_areas_px=free)
