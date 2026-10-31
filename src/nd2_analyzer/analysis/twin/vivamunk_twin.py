"""The twin itself: Viva-munk (pymunk + process-bigraph) cells seeded from one real frame, growing by the measured adder
with the position rule and the measured lag, run for one seed and one rule set. This is DigitalTwin_rerun/run_twin3.py
(Oct 2026, the validated v3 twin) with its inputs passed on the command line instead of read from the rerun folder.

Runs in the Viva-munk environment, not in the tool's:
  python vivamunk_twin.py --viva <Viva-munk dir> --seed_csv seed.csv --adder_json adder.json --chamber_json chamber.json
                          --rule <base|position|lag|crowding> --seed 1 --out_dir <dir> [--sim_time 8100]
Writes <out_dir>/montage_<rule>.pkl, read back by results.py.
"""
import argparse, json, math, os, pickle, random, sys, time, uuid

import numpy as np
import pandas as pd

RULES = {"base": dict(position=False, lag=False, crowding=False),
         "position": dict(position=True, lag=False, crowding=False),
         "lag": dict(position=True, lag=True, crowding=False),
         "crowding": dict(position=True, lag=True, crowding=True)}
DENSITY, INTERVAL = 0.02, 30.0
FREE_DAMPING = 0.1
ALPHA_CV, DELTA_CV, BIRTH_CV = 0.25, 0.20, 0.15

ap = argparse.ArgumentParser()
ap.add_argument("--viva", required=True); ap.add_argument("--seed_csv", required=True); ap.add_argument("--adder_json", required=True)
ap.add_argument("--chamber_json", required=True); ap.add_argument("--rule", choices=list(RULES), required=True)
ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--sim_time", type=float, default=8100); ap.add_argument("--damping", type=float, default=None)
a = ap.parse_args()

AD = json.load(open(a.adder_json)); CH = json.load(open(a.chamber_json))
ALPHA_MEAN_H, DELTA_UM, BIRTH_UM = float(AD["alpha_per_h"]), float(AD["added_um"]), float(AD["birth_total_um"])
POS_SLOPE = float(AD.get("pos_slope", 0.0129))
LAG_END_MIN, LAG_RAMP_MIN, LAG_FACTOR = float(AD.get("lag_end_min", 75.0)), float(AD.get("lag_ramp_min", 20.0)), float(AD.get("lag_factor", 0.55))
n_frames = int(round(a.sim_time / float(CH["frame_interval_s"]))) or 27


def ramp_raw(t_min):
    return LAG_FACTOR + (1 - LAG_FACTOR) * min(1.0, max(0.0, (t_min - LAG_END_MIN) / LAG_RAMP_MIN))


RAMP_NORM = float(np.mean([ramp_raw(CH["frame_interval_s"] / 60 * (k + 0.5)) for k in range(n_frames)]))


def ramp(t_s):
    return ramp_raw(t_s / 60.0) / RAMP_NORM


random.seed(a.seed); np.random.seed(a.seed); _id_rng = random.Random(a.seed + 1_000_003)
uuid.uuid4 = lambda: uuid.UUID(int=_id_rng.getrandbits(128), version=4)
sys.path.insert(0, a.viva)
from process_bigraph import Composite, gather_emitter_results
from process_bigraph.emitter import emitter_from_wires
from multi_cell.experiments.runner import PYMUNK_CORE
from multi_cell.processes.grow_divide import AdderGrowDivide, _build_segment_daughters
from multi_cell.processes.remove_crossing import make_remove_crossing_process
from multi_cell.processes.pressure import make_pressure_process


def capsule_mass(L, r):
    return DENSITY * (L * 2 * r + math.pi * r * r)


class Adder3(AdderGrowDivide):
    """Adder on TOTAL capsule length, area-conserving division, seed phase from own length, position gradient, measured lag."""
    config_schema = dict(AdderGrowDivide.config_schema, seed_cell={"_type": "boolean", "_default": False},
                         open_x={"_type": "float", "_default": 0.0}, pos_slope={"_type": "float", "_default": 0.0},
                         birth_mean={"_type": "float", "_default": 2.0}, birth_cv={"_type": "float", "_default": 0.15},
                         lag_on={"_type": "boolean", "_default": False}, t0={"_type": "float", "_default": 0.0})

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core); self._t = float(self.config["t0"]); self._birth_total = 0.0

    def _init3(self, agent, rng):
        r = float(agent["radius"]); Lt0 = float(agent["length"]) + 2 * r
        self._alpha_per_s = self._sample_lognormal(float(self.config["alpha_mean_per_h"]), float(self.config["alpha_cv"]), rng) / 3600.0
        self._delta = self._sample_normal_pos(float(self.config["delta_mean"]), float(self.config["delta_cv"]), rng)
        if self.config["seed_cell"]:
            Lb = self._sample_lognormal(float(self.config["birth_mean"]), float(self.config["birth_cv"]), rng)
            added = min(max(Lt0 - Lb, 0.0), 0.95 * self._delta); self._birth_total = Lt0 - added
        else:
            self._birth_total = Lt0
        self._initialized = True

    def _daughter_config(self):
        c = super()._daughter_config()
        c.update(seed_cell=False, open_x=float(self.config["open_x"]), pos_slope=float(self.config["pos_slope"]), birth_mean=float(self.config["birth_mean"]),
                 birth_cv=float(self.config["birth_cv"]), lag_on=bool(self.config["lag_on"]), t0=float(self._t)); return c

    def update(self, state, interval):
        rng = random; aid = state["agent_id"]; agent = state["agents"].get(aid, {})
        if not agent:
            return {"agents": {}}
        if not self._initialized:
            self._init3(agent, rng)
        r = float(agent["radius"]); L_old = float(agent["length"]); m_old = float(agent["mass"]); Lt_old = L_old + 2 * r
        alpha = self._alpha_per_s
        slope = float(self.config["pos_slope"])
        if slope > 0:
            alpha *= max(0.2, 1.0 - slope * max(0.0, float(self.config["open_x"]) - float(agent["location"][0])))
        if self.config["lag_on"]:
            alpha *= ramp(self._t)
        pressure = float(agent.get("pressure", 0.0) or 0.0); pk = float(self.config.get("pressure_k", 5.0))
        if pressure > 0 and pk > 0:
            alpha *= math.exp(-pressure / pk)
        self._t += float(interval)
        Lt_new = Lt_old * math.exp(alpha * float(interval)); L_new = Lt_new - 2 * r
        m_new = capsule_mass(L_new, r)
        upd = {aid: {"type": "segment", "length": L_new - L_old, "mass": m_new - m_old}}
        if Lt_new < self._birth_total + self._delta:
            return {"agents": upd}
        f1 = min(0.65, max(0.35, 0.5 * self._sample_lognormal(1.0, float(self.config["division_noise_cv"]), rng)))
        Lt1, Lt2 = Lt_new * f1, Lt_new * (1 - f1); L1, L2 = max(Lt1 - 2 * r, 0.05), max(Lt2 - 2 * r, 0.05)
        slim = self._daughter_config(); gds = []
        for _ in range(2):
            g = dict(agent.get("grow_divide", {})); g.pop("instance", None); g.setdefault("_type", "process"); g["config"] = dict(slim); gds.append(g)
        d1, d2, i1, i2 = _build_segment_daughters(agent, aid, float(agent["angle"]), r, L1, L2, capsule_mass(L1, r), capsule_mass(L2, r), gds[0], gds[1])
        upd["_add"] = {i1: d1, i2: d2}; upd["_remove"] = [aid]
        return {"agents": upd}


PYMUNK_CORE.register_link("Adder3", Adder3)

seed_df = pd.read_csv(a.seed_csv)
W_um, H_um = float(CH["width_um"]), float(CH["height_um"])
OX, OY = 1.0, 5.0                                             # the ROI sits inside a slightly larger world, opening on the right
ENV_W, ENV_H = float(math.ceil(W_um + 11.0)), float(math.ceil(H_um + 10.0))
R = RULES[a.rule]; damping = a.damping if a.damping is not None else FREE_DAMPING
gd_cfg = dict(agents_key="cells", alpha_mean_per_h=ALPHA_MEAN_H / (1 - POS_SLOPE * 17 if R["position"] else 1), alpha_cv=ALPHA_CV,
              delta_mean=DELTA_UM, delta_cv=DELTA_CV, division_noise_cv=0.07, pressure_k=5.0, seed_cell=True, birth_mean=BIRTH_UM, birth_cv=BIRTH_CV,
              open_x=ENV_W, pos_slope=POS_SLOPE if R["position"] else 0.0, lag_on=bool(R["lag"]), t0=0.0)
agents = {}
for c in seed_df.itertuples():
    r = max(float(c.width_um) / 2, 0.3); L = max(float(c.length_seg_um), 0.05)
    x = min(max(float(c.x_um) + OX, r + 0.5), ENV_W - r - 0.5); y = min(max(float(c.y_um) + OY, r + 0.5), ENV_H - r - 0.5)
    agents[f"p{int(c.cell_id)}"] = dict(id=f"p{int(c.cell_id)}", type="segment", mass=float(capsule_mass(L, r)), length=float(L), radius=float(r),
                                        angle=float(c.angle_x_rad), location=(float(x), float(y)), velocity=(0.0, 0.0), elasticity=0.1, adhesins=0.0,
                                        grow_divide={"_type": "process", "address": "local:Adder3", "config": dict(gd_cfg), "interval": INTERVAL,
                                                     "inputs": {"agent_id": ["id"], "agents": ["..", "..", "cells"]}, "outputs": {"agents": ["..", "..", "cells"]}})
mb = {"env_size": ENV_W, "env_height": ENV_H, "elasticity": 0.1, "damping_per_second": float(damping)}
doc = {"cells": agents, "particles": {},
       "multibody": {"_type": "process", "address": "local:PymunkProcess", "config": mb, "interval": INTERVAL,
                     "inputs": {"segment_cells": ["cells"], "circle_particles": ["particles"]}, "outputs": {"segment_cells": ["cells"], "circle_particles": ["particles"]}},
       "remove_crossing": make_remove_crossing_process(x_max=ENV_W - 1.0, agents_key="cells"),
       "emitter": emitter_from_wires({"agents": ["cells"], "particles": ["particles"], "time": ["global_time"]})}
if R["crowding"]:
    doc["pressure"] = make_pressure_process(agents_key="cells", env_size=ENV_W)
print(f"{a.rule} seed {a.seed}: {len(agents)} cells, alpha {gd_cfg['alpha_mean_per_h']:.3f}/h, delta {DELTA_UM:.2f}, birth {BIRTH_UM:.2f}, rules {R}", flush=True)
t0 = time.time(); sim = Composite({"state": doc}, core=PYMUNK_CORE); sim.run(a.sim_time)
results = gather_emitter_results(sim)[("emitter",)]; os.makedirs(a.out_dir, exist_ok=True)
roi = dict(x0=OX, y0=OY, w=W_um, h=H_um)
with open(os.path.join(a.out_dir, f"montage_{a.rule}.pkl"), "wb") as f:
    pickle.dump({"results": results, "env_size": ENV_W, "env_height": ENV_H, "roi": roi, "seed": a.seed, "rule": a.rule, "sim_time": a.sim_time,
                 "gd": gd_cfg, "damping": damping, "version": 3, "adder": AD, "chamber": CH}, f)
print(f"{a.rule} seed {a.seed}: {len(results)} steps in {time.time() - t0:.0f} s", flush=True)
