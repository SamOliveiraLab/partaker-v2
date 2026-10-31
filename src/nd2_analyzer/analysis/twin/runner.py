"""Running the twin from the tool: a run folder with the seed, the measured rules and the chamber, then one Viva-munk
process per seed in the Viva-munk environment (a subprocess, since Viva-munk needs its own Python)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

VIVA_DIR_DEFAULT = "/Volumes/SAM1/Oliveira_lab_workspace/Oliveira_lab_workspace/Bukola/Viva-munk"
VIVA_PYTHON_DEFAULT = str(Path.home() / "miniconda3" / "envs" / "vivamunk" / "bin" / "python")
SCRIPT = str(Path(__file__).with_name("vivamunk_twin.py"))
RULE_SETS = ["base", "position", "lag", "crowding"]


def rule_set(position: bool, lag: bool, crowding: bool) -> str:
    """The rule set name for the chosen rules (the sets nest: lag needs position, crowding needs both)."""
    if crowding:
        return "crowding"
    if lag:
        return "lag"
    if position:
        return "position"
    return "base"


def viva_available(viva_dir: str = VIVA_DIR_DEFAULT, python: str = VIVA_PYTHON_DEFAULT) -> Optional[str]:
    """None when the twin can run, else what is missing."""
    if not Path(python).exists():
        return f"Viva-munk Python not found: {python}"
    if not Path(viva_dir, "multi_cell").exists():
        return f"Viva-munk not found: {viva_dir}"
    return None


def prepare_run(out_dir, seed_df, adder: Dict, chamber: Dict) -> Dict[str, str]:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = {"seed_csv": str(out / "seed.csv"), "adder_json": str(out / "adder.json"), "chamber_json": str(out / "chamber.json")}
    seed_df.to_csv(paths["seed_csv"], index=False, float_format="%.5f")
    json.dump(adder, open(paths["adder_json"], "w"), indent=1)
    json.dump(chamber, open(paths["chamber_json"], "w"), indent=1)
    return paths


def run_seed(paths: Dict[str, str], rule: str, seed: int, out_dir, sim_time: float, viva_dir=VIVA_DIR_DEFAULT, python=VIVA_PYTHON_DEFAULT,
             log: Optional[Callable[[str], None]] = None, cancelled: Optional[Callable[[], bool]] = None) -> str:
    """One Viva-munk run; returns the pickle path. Raises RuntimeError with the process output on failure."""
    seed_dir = Path(out_dir) / f"seed{seed}"; seed_dir.mkdir(parents=True, exist_ok=True)
    cmd = [python, SCRIPT, "--viva", viva_dir, "--seed_csv", paths["seed_csv"], "--adder_json", paths["adder_json"],
           "--chamber_json", paths["chamber_json"], "--rule", rule, "--seed", str(seed), "--out_dir", str(seed_dir), "--sim_time", str(sim_time)]
    env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    lines = []
    for line in proc.stdout:
        lines.append(line.rstrip())
        if log is not None and line.strip():
            log(line.rstrip())
        if cancelled is not None and cancelled():
            proc.kill(); raise RuntimeError("cancelled")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("Viva-munk failed:\n" + "\n".join(lines[-30:]))
    return str(seed_dir / f"montage_{rule}.pkl")
