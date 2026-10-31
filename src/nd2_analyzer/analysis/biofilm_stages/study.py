"""A biofilm study: one recording with the results the tool computed on it, read from a results folder (the run-2 pipeline's
tables), its masks and the raw recording, plus the literature the user imports. Plain Python, no Qt: the stage pages, the
figure export and the tests all read a study the same way.

A study is described by a small JSON file:
    {"name": "E. coli run 2", "pipeline_dir": ".../runs/gui_20261025_145305", "masks_dir": ".../story_v2",
     "nd2": ".../mcherry-tl-t24-t72-07-01-2026.nd2", "pixel_um": 0.0433, "hour0": 24.0, "dt_h": 0.5,
     "literature": ".../hallmarks.json", "frame_3h": ".../frame_3h.npz"}
Every number on a page comes from the tables in pipeline_dir; nothing is typed here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Curve:
    hours: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    label: str = ""
    unit: str = ""


@dataclass
class StageHours:
    th1_h: Optional[float] = None
    th1_sd_h: Optional[float] = None
    th1_positions: List[float] = field(default_factory=list)
    th2_h: Optional[float] = None
    th2_rule: str = ""
    last_h: float = 68.0
    first_h: float = 24.0
    merge_h: Optional[float] = None

    def stages(self) -> List[Tuple[str, float, float]]:
        """(name, start, end) of the stages this recording reached."""
        out = []
        a = self.first_h
        if self.th1_h is not None:
            out.append(("I  microcolony formation", a, self.th1_h)); a = self.th1_h
        if self.th2_h is not None:
            out.append(("II  biofilm maturation", a, self.th2_h)); out.append(("III  pre-dispersion", self.th2_h, self.last_h))
        else:
            out.append(("II  biofilm maturation", a, self.last_h))
        return out


class BiofilmStudy:
    def __init__(self, spec: dict, spec_path: Optional[str] = None):
        self.spec = dict(spec); self.spec_path = spec_path
        self.name = spec.get("name", "study")
        self.pipeline_dir = Path(spec["pipeline_dir"]) if spec.get("pipeline_dir") else None
        self.masks_dir = Path(spec["masks_dir"]) if spec.get("masks_dir") else None
        self.nd2 = spec.get("nd2")
        self.pixel_um = float(spec.get("pixel_um", 0.0433))
        self.hour0 = float(spec.get("hour0", 24.0)); self.dt_h = float(spec.get("dt_h", 0.5))
        self.literature_path = spec.get("literature")
        self.frame_3h_path = spec.get("frame_3h")
        self._cache: Dict = {}

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_json(cls, path) -> "BiofilmStudy":
        p = Path(path); spec = json.loads(p.read_text())
        base = p.parent
        for k in ("pipeline_dir", "masks_dir", "nd2", "literature", "frame_3h"):
            v = spec.get(k)
            if v and not str(v).startswith("/") and not str(v).startswith("~"):
                spec[k] = str((base / v).resolve())
            elif v:
                spec[k] = str(Path(v).expanduser())
        return cls(spec, str(p))

    def available(self) -> Dict[str, bool]:
        """What this study can show: results tables, masks, the recording, literature, the 3 h frame."""
        return {"results": bool(self.pipeline_dir and (self.pipeline_dir / "step7_frames.csv").exists()),
                "masks": bool(self.masks_dir and (self.masks_dir / "masks_cells").exists()),
                "recording": bool(self.nd2 and Path(self.nd2).exists()),
                "literature": bool(self.literature_path and Path(self.literature_path).exists()),
                "frame_3h": bool(self.frame_3h_path and Path(self.frame_3h_path).exists())}

    def _csv(self, name):
        import pandas as pd
        key = ("csv", name)
        if key not in self._cache:
            p = self.pipeline_dir / name if self.pipeline_dir else None
            self._cache[key] = pd.read_csv(p) if p and p.exists() else None
        return self._cache[key]

    def _json(self, name):
        key = ("json", name)
        if key not in self._cache:
            p = self.pipeline_dir / name if self.pipeline_dir else None
            self._cache[key] = json.loads(p.read_text()) if p and p.exists() else None
        return self._cache[key]

    # ------------------------------------------------------------------ positions and hours
    def kept(self) -> List[int]:
        j = self._json("positions_kept.json"); return list(j["kept"]) if j else []

    def gate(self) -> Dict:
        return self._json("positions_kept.json") or {}

    def hours_to_t(self, h: float) -> int:
        return int(round((h - self.hour0) / self.dt_h))

    def t_to_hours(self, t: int) -> float:
        return self.hour0 + t * self.dt_h

    def grid_hours(self) -> np.ndarray:
        df = self._csv("step7_frames.csv")
        return np.sort(df.hours.unique()) if df is not None else np.array([])

    def stage_hours(self) -> StageHours:
        st = self._json("step8_growth_states/step8_stages_v11.json") or {}
        mg = self._json("step7_merge.json") or {}
        hrs = self.grid_hours()
        return StageHours(th1_h=st.get("Th1_matrix_appears_h"), th1_sd_h=st.get("Th1_sd_h"), th1_positions=st.get("Th1_per_position_h", []),
                          th2_h=st.get("Th2_rate_first_zero_h"), th2_rule=st.get("Th2_rule", ""),
                          first_h=float(hrs[0]) if len(hrs) else self.hour0, last_h=float(hrs[-1]) if len(hrs) else 68.0,
                          merge_h=mg.get("merge_h"))

    # ------------------------------------------------------------------ curves (mean with SD over the kept positions)
    def _mean_sd(self, df, col) -> Curve:
        kept = self.kept()
        if kept: df = df[df.position.isin(kept)]
        g = df.groupby("hours")[col]
        return Curve(g.mean().index.values.astype(float), g.mean().values.astype(float), g.std(ddof=1).fillna(0).values.astype(float))

    def cells_in_microcolonies(self) -> Optional[Curve]:
        df = self._csv("step7_frames.csv")
        if df is None: return None
        c = self._mean_sd(df, "cells_in_clusters_pct"); c.label, c.unit = "cells in microcolonies", "%"; return c

    def cells_in_microcolonies_random(self) -> Optional[Curve]:
        df = self._csv("step7_random_baseline.csv")
        if df is None: return None
        c = self._mean_sd(df, "cells_in_clusters_pct_random"); c.label, c.unit = "the same cells placed at random", "%"; return c

    def colonies_per_field(self) -> Optional[Curve]:
        df = self._csv("step7_frames.csv")
        if df is None: return None
        c = self._mean_sd(df, "n_clusters"); c.label, c.unit = "microcolonies per field", "count"; return c

    def cells_per_colony(self) -> Optional[Curve]:
        df = self._csv("step7_frames.csv")
        if df is None: return None
        c = self._mean_sd(df, "mean_cells_per_cluster"); c.label, c.unit = "cells per microcolony", "count"; return c

    def matrix_pct(self) -> Optional[Curve]:
        df = self._csv("step5_rows.csv")
        if df is None: return None
        c = self._mean_sd(df, "matrix_pct"); c.label, c.unit = "matrix", "% of frame"; return c

    def cells_covered_pct(self) -> Optional[Curve]:
        df = self._csv("step6_cells_covered.csv")
        if df is None: return None
        c = self._mean_sd(df, "covered_pct"); c.label, c.unit = "cells covered by matrix", "%"; return c

    def area_rate(self) -> Optional[Curve]:
        df = self._csv("step8_growth_states/step8_growth_chain_v11.csv")
        if df is None: return None
        return Curve(df.hours.values.astype(float), df.area_rate_pct_per_h_mean.values.astype(float), df.area_rate_sd.values.astype(float), "cell area rate", "% of frame per h")

    def cell_area(self) -> Optional[Curve]:
        df = self._csv("step8_growth_states/step8_growth_chain_v11.csv")
        if df is None: return None
        return Curve(df.hours.values.astype(float), df.area_pct_mean.values.astype(float), df.area_pct_sd.values.astype(float), "cell area", "% of frame")

    def tile_correlation(self) -> Optional[Curve]:
        df = self._csv("step9_tile_correlation.csv")
        if df is None: return None
        c = self._mean_sd(df, "spearman_dye_vs_cells"); c.label, c.unit = "dye against cells, 4 um tiles", "Spearman r"; return c

    def matrix_per_cell(self) -> Optional[Curve]:
        df = self._csv("step9_matrix_per_cell.csv")
        if df is None: return None
        return Curve(df.hours.values.astype(float), df.m.values.astype(float), df.s.values.astype(float), "matrix per cell", "um2")

    def dynamics(self) -> Dict:
        return self._json("step7d_colony_dynamics.json") or {}

    def events(self):
        return self._csv("step7d_events.csv")

    def dispersion_candidates(self) -> Dict:
        j = self._json("dispersion/candidates.json"); return j or {}

    def dispersion_table(self, name: str):
        return self._csv(f"dispersion/{name}.csv")

    # ------------------------------------------------------------------ frames
    def frame(self, p: int, hour: float) -> Dict[str, np.ndarray]:
        """phase, cells (labels), colonies (labels by the microcolony rule), matrix (bool), dye (8 bit, the tool's processing)
        for position p at the grid hour nearest to hour. Cached per (p, t)."""
        hrs = self.grid_hours()
        h = float(hrs[np.argmin(np.abs(hrs - hour))]) if len(hrs) else hour
        t = self.hours_to_t(h); key = ("frame", p, t)
        if key in self._cache: return self._cache[key]
        out = {"hour": h, "t": t}
        if self.nd2 and Path(self.nd2).exists():
            import nd2
            f = nd2.ND2File(self.nd2); arr = f.to_dask()
            out["phase"] = np.asarray(arr[t, p, 0]); g = np.asarray(arr[t, p, 2]).astype(np.float32); g0 = np.asarray(arr[0, p, 2]).astype(np.float32); f.close()
            win = self._json("run_window.json") or {}
            lo, hi = float(win.get("black_raw", 0)), float(win.get("white_raw", 1))
            from scipy.ndimage import gaussian_filter
            out["dye"] = np.clip((g - gaussian_filter(g0, 200)) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        if self.masks_dir:
            pc = self.masks_dir / "masks_cells" / f"p{p}_t{t}.npz"; pe = self.masks_dir / "masks_eps" / f"p{p}_t{t}.npz"
            if pc.exists():
                out["cells"] = np.load(pc)["cell_labels"]
                out["colonies"] = microcolony_labels(out["cells"], self.pixel_um)
            if pe.exists():
                e = np.load(pe); shape = tuple(e["shape"]); n = shape[0] * shape[1]
                out["matrix"] = np.unpackbits(e["matrix"])[:n].reshape(shape) > 0
                out["eps_mask"] = np.unpackbits(e["eps"])[:n].reshape(shape) > 0
        self._cache[key] = out
        if len(self._cache) > 40:
            for k in [k for k in self._cache if k[0] == "frame"][:10]: self._cache.pop(k, None)
        return out

    def frame_3h(self) -> Optional[Dict[str, np.ndarray]]:
        if not (self.frame_3h_path and Path(self.frame_3h_path).exists()): return None
        z = np.load(self.frame_3h_path); return {k: z[k] for k in z.files}

    # ------------------------------------------------------------------ literature
    def literature(self) -> Dict:
        if self.literature_path and Path(self.literature_path).exists():
            return json.loads(Path(self.literature_path).read_text())
        return {}


def microcolony_labels(cells: np.ndarray, pixel_um: float, min_cells: int = 4, blur_um: float = 1.0, fraction: float = 0.40) -> np.ndarray:
    """The microcolony rule: at least min_cells cells within blur_um of each other (local cell fraction >= fraction after a
    blur_um Gaussian blur of the cell mask). Returns a label image, one label per microcolony, 0 elsewhere (cells outside any
    microcolony are 0)."""
    from scipy.ndimage import gaussian_filter
    from skimage.measure import label, regionprops
    cells = np.asarray(cells)
    dens = gaussian_filter((cells > 0).astype(np.float32), blur_um / pixel_um)
    comp = label(dens >= fraction)
    counts: Dict[int, int] = {}
    owner: Dict[int, int] = {}
    for rp in regionprops(cells):
        cy, cx = map(int, rp.centroid); k = int(comp[cy, cx])
        if k:
            counts[k] = counts.get(k, 0) + 1; owner[int(rp.label)] = k
    keep = {k for k, n in counts.items() if n >= min_cells}
    lut = np.zeros(int(cells.max()) + 1, np.int32)
    for lab, k in owner.items():
        if k in keep: lut[lab] = k
    return lut[cells]
