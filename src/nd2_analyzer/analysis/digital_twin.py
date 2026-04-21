"""
Digital Twin Bridge — connects PARTAKER microscopy analysis to Viva-munk simulation.

Extracts real cell parameters from CellHistoryBuilder data, calibrates Viva-munk
experiments with those parameters, runs simulations, and generates comparison figures.

Uses a separate Python environment (digital_twin_env) since Viva-munk requires Python 3.11+.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DT_HOME = Path.home() / ".partaker"
DT_ENV_PYTHON = str(DT_HOME / "digital_twin_env" / "bin" / "python")
DT_VIVAMUNK = str(DT_HOME / "digital_twin_vivamunk")
DT_OUTPUT_DIR = str(PROJECT_ROOT / "digital_twin_output")


def _noop_log(msg):
    print(msg)


class ParameterExtractor:
    """Extracts simulation-ready parameters from PARTAKER cell history data."""

    def __init__(self, cell_database: Dict, log=None):
        self.cell_database = cell_database
        self.params = {}
        self.log = log or _noop_log

    def extract_all(self, time_interval_seconds: float = 300.0) -> Dict:
        """Extract all parameters needed for Viva-munk calibration.

        Parameters
        ----------
        time_interval_seconds : float
            Real time between microscopy frames in seconds.
            Default 300s (5 min) — typical for time-lapse bacterial imaging.
        """
        self.dt = time_interval_seconds
        cells = list(self.cell_database.values())

        self.log(f"[Extract] {len(cells)} cells in database, dt={time_interval_seconds}s")

        self.params = {
            'n_cells': len(cells),
            'time_interval_s': self.dt,
        }

        self.log("[Extract] Fitting growth rates (ln(area) vs time)...")
        self._extract_growth_rates(cells)

        self.log("[Extract] Extracting division parameters...")
        self._extract_division_params(cells)

        self.log("[Extract] Extracting morphology statistics...")
        self._extract_morphology(cells)

        self.log("[Extract] Extracting motility statistics...")
        self._extract_movement(cells)

        params_path = os.path.join(DT_OUTPUT_DIR, "extracted_params.json")
        os.makedirs(DT_OUTPUT_DIR, exist_ok=True)
        serializable = {}
        for k, v in self.params.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                serializable[k] = float(v)
            else:
                serializable[k] = v
        with open(params_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        self.log(f"[Extract] Params saved to {params_path}")

        return self.params

    def _extract_growth_rates(self, cells: List[Dict]):
        """Estimate exponential growth rate from area time series.

        For each cell, fit ln(area) vs time. The slope is the growth rate (1/s).
        """
        growth_rates = []

        for cell in cells:
            areas = np.array(cell['area'], dtype=float)
            valid = ~np.isnan(areas) & (areas > 0)
            if valid.sum() < 5:
                continue

            areas_valid = areas[valid]
            times_valid = np.where(valid)[0] * self.dt

            ln_areas = np.log(areas_valid)
            if len(ln_areas) < 2:
                continue

            coeffs = np.polyfit(times_valid, ln_areas, 1)
            rate = coeffs[0]

            if 0 < rate < 0.01:
                growth_rates.append(rate)

        if growth_rates:
            self.params['growth_rate_mean'] = float(np.mean(growth_rates))
            self.params['growth_rate_std'] = float(np.std(growth_rates))
            self.params['growth_rate_cv'] = float(np.std(growth_rates) / np.mean(growth_rates)) if np.mean(growth_rates) > 0 else 0.0
            self.params['doubling_time_s'] = float(np.log(2) / np.mean(growth_rates))
            self.params['doubling_time_min'] = self.params['doubling_time_s'] / 60.0
            self.params['n_cells_with_growth'] = len(growth_rates)
            self.params['growth_rates_all'] = growth_rates
            self.log(f"  -> {len(growth_rates)} cells with valid growth, "
                     f"rate={np.mean(growth_rates)*3600:.4f}/h, "
                     f"Td={self.params['doubling_time_min']:.1f}min")
        else:
            self.log("  -> WARNING: No cells with valid growth rates found")

    def _extract_division_params(self, cells: List[Dict]):
        """Extract division size threshold and interdivision time."""
        dividing_cells = [c for c in cells if c['fate'] == 'divided']

        if dividing_cells:
            pre_division_areas = []
            pre_division_lengths = []
            interdivision_times = []

            for cell in dividing_cells:
                areas = np.array(cell['area'], dtype=float)
                lengths = np.array(cell['length'], dtype=float)
                valid_areas = areas[~np.isnan(areas)]
                valid_lengths = lengths[~np.isnan(lengths)]

                if len(valid_areas) > 0:
                    pre_division_areas.append(float(valid_areas[-1]))
                if len(valid_lengths) > 0:
                    pre_division_lengths.append(float(valid_lengths[-1]))
                interdivision_times.append(cell['lifespan'] * self.dt)

            if pre_division_areas:
                self.params['division_area_mean'] = float(np.mean(pre_division_areas))
                self.params['division_area_std'] = float(np.std(pre_division_areas))
            if pre_division_lengths:
                self.params['division_length_mean'] = float(np.mean(pre_division_lengths))
                self.params['division_length_std'] = float(np.std(pre_division_lengths))
            if interdivision_times:
                self.params['interdivision_time_mean_s'] = float(np.mean(interdivision_times))
                self.params['interdivision_time_std_s'] = float(np.std(interdivision_times))
                self.params['interdivision_time_mean_min'] = self.params['interdivision_time_mean_s'] / 60.0

            self.params['n_dividing_cells'] = len(dividing_cells)
            self.log(f"  -> {len(dividing_cells)} dividing cells, "
                     f"IDT={self.params.get('interdivision_time_mean_min', 0):.1f}min, "
                     f"div length={self.params.get('division_length_mean', 0):.1f}px")
        else:
            self.log("  -> No dividing cells found")

    def _extract_morphology(self, cells: List[Dict]):
        """Extract average cell dimensions."""
        all_lengths = []
        all_widths = []
        all_areas = []
        all_aspect_ratios = []

        for cell in cells:
            lengths = np.array(cell['length'], dtype=float)
            widths = np.array(cell['width'], dtype=float)
            areas = np.array(cell['area'], dtype=float)
            ars = np.array(cell['aspect_ratio'], dtype=float)

            valid_l = lengths[~np.isnan(lengths)]
            valid_w = widths[~np.isnan(widths)]
            valid_a = areas[~np.isnan(areas)]
            valid_ar = ars[~np.isnan(ars)]

            if len(valid_l) > 0:
                all_lengths.append(float(np.mean(valid_l)))
            if len(valid_w) > 0:
                all_widths.append(float(np.mean(valid_w)))
            if len(valid_a) > 0:
                all_areas.append(float(np.mean(valid_a)))
            if len(valid_ar) > 0:
                all_aspect_ratios.append(float(np.mean(valid_ar)))

        if all_lengths:
            self.params['cell_length_mean_px'] = float(np.mean(all_lengths))
            self.params['cell_length_std_px'] = float(np.std(all_lengths))
        if all_widths:
            self.params['cell_width_mean_px'] = float(np.mean(all_widths))
            self.params['cell_width_std_px'] = float(np.std(all_widths))
        if all_areas:
            self.params['cell_area_mean_px'] = float(np.mean(all_areas))
        if all_aspect_ratios:
            self.params['aspect_ratio_mean'] = float(np.mean(all_aspect_ratios))
        self.log(f"  -> length={np.mean(all_lengths):.1f}px, width={np.mean(all_widths):.1f}px, "
                 f"AR={np.mean(all_aspect_ratios):.2f}" if all_lengths else "  -> No morphology data")

    def _extract_movement(self, cells: List[Dict]):
        """Extract movement/motility statistics."""
        velocities = [c['avg_velocity'] for c in cells]
        displacements = [c['total_displacement'] for c in cells]
        directionalities = [c['directionality'] for c in cells]

        if velocities:
            self.params['avg_velocity_px_per_frame'] = float(np.mean(velocities))
            self.params['avg_displacement_px'] = float(np.mean(displacements))
            self.params['avg_directionality'] = float(np.mean(directionalities))
            self.log(f"  -> velocity={np.mean(velocities):.2f}px/f, "
                     f"displacement={np.mean(displacements):.1f}px, "
                     f"directionality={np.mean(directionalities):.3f}")


class DigitalTwinRunner:
    """Runs Viva-munk simulations calibrated with real parameters."""

    def __init__(self, extracted_params: Dict, pixel_to_um: float = 0.065, log=None):
        self.params = extracted_params
        self.px_to_um = pixel_to_um
        self.log = log or _noop_log
        self.output_dir = DT_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def build_calibrated_config(self) -> Dict:
        """Convert extracted pixel-space parameters to Viva-munk config."""
        p = self.params

        growth_rate = p.get('growth_rate_mean', 0.000289)

        cell_length_um = p.get('cell_length_mean_px', 30.0) * self.px_to_um
        cell_width_um = p.get('cell_width_mean_px', 10.0) * self.px_to_um
        cell_radius = cell_width_um / 2.0

        if cell_length_um < 0.5:
            cell_length_um = 2.0
        if cell_radius < 0.1:
            cell_radius = 0.5

        division_length_um = p.get('division_length_mean_px', 60.0) * self.px_to_um
        if division_length_um < cell_length_um:
            division_length_um = cell_length_um * 2.0

        total_sim_time = p.get('interdivision_time_mean_s', 7200.0) * 4.0
        env_size = max(40.0, cell_length_um * 20)

        density = 0.02

        config = {
            'env_size': env_size,
            'growth_rate': growth_rate,
            'cell_length': cell_length_um,
            'cell_radius': cell_radius,
            'density': density,
            'division_threshold': density * (2 * cell_radius) * (division_length_um),
            'total_time': total_sim_time,
            'interval': 30.0,
        }

        self.calibrated_config = config
        self.log(f"[Config] env_size={env_size:.1f}µm, "
                 f"growth_rate={growth_rate:.6f}/s ({growth_rate*3600:.4f}/h), "
                 f"cell_length={cell_length_um:.2f}µm, radius={cell_radius:.2f}µm")
        self.log(f"[Config] division_threshold={config['division_threshold']:.4f}, "
                 f"sim_time={total_sim_time/3600:.1f}h")
        return config

    def run_simulation(self, experiment_type: str = 'daughter_machine') -> Dict:
        """Run a Viva-munk simulation using the calibrated config."""
        config = self.build_calibrated_config()

        script = self._build_simulation_script(experiment_type, config)

        script_path = os.path.join(self.output_dir, '_run_simulation.py')
        results_path = os.path.join(self.output_dir, 'simulation_results.json')

        with open(script_path, 'w') as f:
            f.write(script)

        self.log(f"[Sim] Launching {experiment_type} in digital_twin_env...")
        self.log(f"[Sim] Python: {DT_ENV_PYTHON}")

        result = subprocess.run(
            [DT_ENV_PYTHON, script_path],
            capture_output=True, text=True, timeout=600,
        )

        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                self.log(f"[Sim] {line}")

        if result.returncode != 0:
            self.log(f"[Sim] STDERR: {result.stderr[:500]}")
            raise RuntimeError(f"Simulation failed:\n{result.stderr}")

        with open(results_path, 'r') as f:
            sim_results = json.load(f)

        self.log(f"[Sim] Done: {sim_results.get('final_cell_count', 0)} final cells, "
                 f"{sim_results.get('n_steps', 0)} steps")
        return sim_results

    def _build_simulation_script(self, experiment_type: str, config: Dict) -> str:
        """Generate a Python script that runs in digital_twin_env."""
        results_path = os.path.join(self.output_dir, 'simulation_results.json')
        gif_path = os.path.join(self.output_dir, f'calibrated_{experiment_type}.gif')

        return f'''
import json, time, os, sys, math
sys.path.insert(0, {repr(DT_VIVAMUNK)})

from multi_cell import core_import
from process_bigraph import Composite, gather_emitter_results
from multi_cell.experiments.documents.daughter_machine import daughter_machine_document
from multi_cell.experiments.documents.mother_machine import mother_machine_document
from multi_cell.plots.multibody_plots import simulation_to_gif

core = core_import()

config = {json.dumps(config)}
experiment_type = {repr(experiment_type)}

if experiment_type == "mother_machine":
    doc = mother_machine_document(config=config)
else:
    doc = daughter_machine_document(config=config)

sim = Composite({{"state": doc}}, core=core)

total_time = config["total_time"]
print(f"Running {{experiment_type}} for {{total_time/3600:.1f}} hours...")
start = time.time()
sim.run(total_time)
elapsed = time.time() - start

results = gather_emitter_results(sim)[("emitter",)]
print(f"Done in {{elapsed:.1f}}s - {{len(results)}} steps")

# Extract time series data from simulation
sim_data = {{
    "elapsed_s": elapsed,
    "n_steps": len(results),
    "experiment_type": experiment_type,
    "config": config,
    "timesteps": [],
}}

for step in results:
    agents = step.get("agents", {{}})
    t = step.get("time", 0.0)
    step_data = {{
        "time": t,
        "n_cells": len(agents),
        "cells": [],
    }}
    for aid, agent in agents.items():
        if isinstance(agent, dict):
            step_data["cells"].append({{
                "id": aid,
                "mass": float(agent.get("mass", 0)),
                "length": float(agent.get("length", 0)),
                "radius": float(agent.get("radius", 0)),
                "location": list(agent.get("location", (0, 0))),
                "angle": float(agent.get("angle", 0)),
            }})
    sim_data["timesteps"].append(step_data)

last = results[-1]
sim_data["final_cell_count"] = len(last.get("agents", {{}}))

with open({repr(results_path)}, "w") as f:
    json.dump(sim_data, f, indent=2)
print(f"Results saved to {repr(results_path)}")

# Generate GIF
out_dir = {repr(self.output_dir)}
gif_path = simulation_to_gif(
    results,
    filename=f"calibrated_{{experiment_type}}",
    config={{"env_size": config["env_size"]}},
    out_dir=out_dir,
    color_by_phylogeny=True,
    skip_frames=max(1, len(results) // 150),
    frame_duration_ms=50,
    show_time_title=True,
    world_pad=2.0,
    dpi=150,
)
print(f"GIF saved: {{gif_path}}")
'''

    def run_default_simulation(self) -> Dict:
        """Run a default (uncalibrated) simulation for comparison."""
        script = f'''
import json, time, sys
sys.path.insert(0, {repr(DT_VIVAMUNK)})

from multi_cell import core_import
from process_bigraph import Composite, gather_emitter_results
from multi_cell.experiments.documents.daughter_machine import daughter_machine_document
from multi_cell.plots.multibody_plots import simulation_to_gif

core = core_import()

doc = daughter_machine_document(config={{"env_size": 40}})
sim = Composite({{"state": doc}}, core=core)

total_time = {self.calibrated_config.get("total_time", 28800.0)}
print(f"Running default daughter_machine for {{total_time/3600:.1f}} hours...")
start = time.time()
sim.run(total_time)
elapsed = time.time() - start

results = gather_emitter_results(sim)[("emitter",)]
print(f"Done in {{elapsed:.1f}}s - {{len(results)}} steps")

sim_data = {{
    "elapsed_s": elapsed,
    "n_steps": len(results),
    "experiment_type": "daughter_machine_default",
    "timesteps": [],
}}

for step in results:
    agents = step.get("agents", {{}})
    t = step.get("time", 0.0)
    step_data = {{"time": t, "n_cells": len(agents), "cells": []}}
    for aid, agent in agents.items():
        if isinstance(agent, dict):
            step_data["cells"].append({{
                "id": aid,
                "mass": float(agent.get("mass", 0)),
                "length": float(agent.get("length", 0)),
                "radius": float(agent.get("radius", 0)),
                "location": list(agent.get("location", (0, 0))),
            }})
    sim_data["timesteps"].append(step_data)

sim_data["final_cell_count"] = len(results[-1].get("agents", {{}}))

out_path = {repr(os.path.join(self.output_dir, "default_simulation_results.json"))}
with open(out_path, "w") as f:
    json.dump(sim_data, f, indent=2)
print(f"Default results saved")

gif_path = simulation_to_gif(
    results,
    filename="default_daughter_machine",
    config={{"env_size": 40}},
    out_dir={repr(self.output_dir)},
    color_by_phylogeny=True,
    skip_frames=max(1, len(results) // 150),
    frame_duration_ms=50,
    show_time_title=True,
    world_pad=2.0,
    dpi=150,
)
print(f"Default GIF saved: {{gif_path}}")
'''
        script_path = os.path.join(self.output_dir, '_run_default.py')
        results_path = os.path.join(self.output_dir, 'default_simulation_results.json')

        with open(script_path, 'w') as f:
            f.write(script)

        self.log("[Default] Launching default E. coli simulation...")
        result = subprocess.run(
            [DT_ENV_PYTHON, script_path],
            capture_output=True, text=True, timeout=600,
        )

        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                self.log(f"[Default] {line}")

        if result.returncode != 0:
            self.log(f"[Default] STDERR: {result.stderr[:500]}")
            raise RuntimeError(f"Default simulation failed:\n{result.stderr}")

        with open(results_path, 'r') as f:
            data = json.load(f)

        self.log(f"[Default] Done: {data.get('final_cell_count', 0)} cells")
        return data


class ComparisonFigureGenerator:
    """Generates publication-quality comparison figures: simulated vs observed."""

    # --- Publication palette ---
    C_OBS = '#2E86AB'
    C_SIM = '#A23B72'
    C_DEF = '#F18F01'
    C_THEORY = '#C73E1D'
    C_BG = '#FAFAFA'
    C_GRID = '#E0E0E0'

    def __init__(self, real_params: Dict, sim_results: Dict,
                 default_sim_results: Optional[Dict] = None,
                 cell_database: Optional[Dict] = None,
                 log=None):
        self.real = real_params
        self.sim = sim_results
        self.default_sim = default_sim_results
        self.cell_database = cell_database
        self.log = log or _noop_log
        self.output_dir = DT_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def _pub_style(self):
        """Apply publication-quality matplotlib rcParams."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.titleweight': 'bold',
            'axes.labelsize': 12,
            'axes.labelweight': 'medium',
            'axes.linewidth': 1.2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.facecolor': self.C_BG,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'legend.fontsize': 10,
            'legend.framealpha': 0.9,
            'legend.edgecolor': '#CCCCCC',
            'figure.facecolor': 'white',
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.15,
        })
        return plt

    def generate_all(self) -> List[str]:
        """Generate all comparison figures. Returns list of saved paths."""
        self._pub_style()

        paths = []
        self.log("[Figures] Generating side-by-side validation figure...")
        paths.append(self._plot_side_by_side())
        self.log("[Figures] Generating population growth...")
        paths.append(self._plot_population_growth())
        self.log("[Figures] Generating size distributions...")
        paths.append(self._plot_cell_size_distributions())
        self.log("[Figures] Generating growth rate comparison...")
        paths.append(self._plot_growth_rate_comparison())
        self.log("[Figures] Generating division timing...")
        paths.append(self._plot_division_timing())
        if self.cell_database:
            self.log("[Figures] Generating single cell growth curves...")
            paths.append(self._plot_single_cell_growth_curves())
        self.log("[Figures] Generating validation dashboard...")
        paths.append(self._plot_summary_dashboard())
        result = [p for p in paths if p]
        self.log(f"[Figures] Done — {len(result)} figures saved to {self.output_dir}")
        return result

    def _save(self, fig, name):
        path = os.path.join(self.output_dir, name)
        fig.savefig(path)
        import matplotlib.pyplot as plt
        plt.close(fig)
        return path

    def _plot_side_by_side(self) -> Optional[str]:
        """Generate the main DARPA figure: Observed vs Predicted side-by-side."""
        plt = self._pub_style()

        tracking_path = os.path.join(self.output_dir, "tracking_data.json")
        has_tracking = os.path.exists(tracking_path)
        tracks = []
        if has_tracking:
            with open(tracking_path) as f:
                tracks = json.load(f)
            self.log(f"[Side-by-side] Loaded {len(tracks)} observed tracks")

        fig = plt.figure(figsize=(18, 22), facecolor='white')
        fig.suptitle('DIGITAL TWIN VALIDATION',
                     fontsize=20, fontweight='bold', y=0.97, color='#1a1a2e')
        fig.text(0.5, 0.955,
                 'Observed (PARTAKER)  vs  Predicted (Viva-munk Calibrated Simulation)',
                 ha='center', fontsize=12, color='#666', style='italic')

        fig.text(0.28, 0.935, 'OBSERVED  (PARTAKER)', fontsize=14, fontweight='bold',
                 ha='center', color=self.C_OBS,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#EBF5FB',
                           edgecolor=self.C_OBS, linewidth=2))
        fig.text(0.72, 0.935, 'PREDICTED  (Digital Twin)', fontsize=14, fontweight='bold',
                 ha='center', color=self.C_SIM,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5EBF5',
                           edgecolor=self.C_SIM, linewidth=2))

        gs = fig.add_gridspec(4, 2, hspace=0.32, wspace=0.25,
                              top=0.91, bottom=0.03, left=0.07, right=0.95)

        # --- Row 1: Spatial density ---
        ax1L = fig.add_subplot(gs[0, 0])
        ax1R = fig.add_subplot(gs[0, 1])
        ax1L.set_facecolor('#EBF5FB')
        ax1R.set_facecolor('#F5EBF5')

        sim_grid = self._sim_density_grid(20, 20)
        if has_tracking and tracks:
            obs_grid = self._obs_density_grid(tracks, 20, 20)
            ax1L.imshow(obs_grid, cmap='YlOrRd', aspect='auto', origin='lower')
            ax1L.set_title('Spatial Density — Observed', fontsize=12,
                           fontweight='bold', color=self.C_OBS)
        else:
            ax1L.text(0.5, 0.5, 'Run tracking to populate',
                      transform=ax1L.transAxes, ha='center', va='center',
                      fontsize=12, color='#999', style='italic')
            ax1L.set_title('Spatial Density — Observed (pending)',
                           fontsize=12, fontweight='bold', color='#999')

        ax1R.imshow(sim_grid, cmap='YlOrRd', aspect='auto', origin='lower')
        ax1R.set_title('Spatial Density — Predicted', fontsize=12,
                       fontweight='bold', color=self.C_SIM)
        for ax in (ax1L, ax1R):
            ax.set_xlabel('x'); ax.set_ylabel('y')

        # --- Row 2: Trajectories ---
        ax2L = fig.add_subplot(gs[1, 0])
        ax2R = fig.add_subplot(gs[1, 1])
        ax2L.set_facecolor('#EBF5FB')
        ax2R.set_facecolor('#F5EBF5')

        if has_tracking and tracks:
            self._draw_obs_trajectories(ax2L, tracks, max_tracks=20)
            ax2L.set_title('Cell Trajectories — Observed', fontsize=12,
                           fontweight='bold', color=self.C_OBS)
        else:
            ax2L.text(0.5, 0.5, 'Run tracking to populate',
                      transform=ax2L.transAxes, ha='center', va='center',
                      fontsize=12, color='#999', style='italic')
            ax2L.set_title('Cell Trajectories — Observed (pending)',
                           fontsize=12, fontweight='bold', color='#999')

        self._draw_sim_trajectories(ax2R, max_tracks=20)
        ax2R.set_title('Particle Trajectories — Predicted', fontsize=12,
                       fontweight='bold', color=self.C_SIM)

        # --- Row 3: Single cell growth ---
        ax3L = fig.add_subplot(gs[2, 0])
        ax3R = fig.add_subplot(gs[2, 1])
        ax3L.set_facecolor('#EBF5FB')
        ax3R.set_facecolor('#F5EBF5')

        if self.cell_database:
            cells = list(self.cell_database.values())
            longest = sorted(cells, key=lambda c: c['lifespan'], reverse=True)[:6]
            cmap = plt.cm.get_cmap('Set2', 8)
            dt_s = self.real.get('time_interval_s', 300)
            for i, cell in enumerate(longest):
                areas = np.array(cell['area'], dtype=float)
                valid = ~np.isnan(areas)
                t_min = np.arange(len(areas))[valid] * dt_s / 60.0
                ax3L.plot(t_min, areas[valid], '-o', color=cmap(i), linewidth=1.5,
                          markersize=2, alpha=0.85, label=f"Cell {cell['cell_id']}")
            ax3L.legend(fontsize=7, ncol=2, frameon=True)
            ax3L.set_title('Single Cell Growth — Observed', fontsize=12,
                           fontweight='bold', color=self.C_OBS)
        elif has_tracking and tracks:
            longest_t = sorted(tracks, key=lambda t: len(t.get('area', [])), reverse=True)[:6]
            cmap = plt.cm.get_cmap('Set2', 8)
            for i, tr in enumerate(longest_t):
                areas = tr.get('area', [])
                if not areas:
                    continue
                t_min = np.arange(len(areas)) * self.real.get('time_interval_s', 300) / 60.0
                ax3L.plot(t_min, areas, '-o', color=cmap(i), linewidth=1.5,
                          markersize=2, alpha=0.85, label=f"Cell {tr.get('ID', i)}")
            ax3L.legend(fontsize=7, ncol=2, frameon=True)
            ax3L.set_title('Single Cell Growth — Observed', fontsize=12,
                           fontweight='bold', color=self.C_OBS)
        else:
            ax3L.text(0.5, 0.5, 'Run tracking to populate',
                      transform=ax3L.transAxes, ha='center', va='center',
                      fontsize=12, color='#999', style='italic')
            ax3L.set_title('Single Cell Growth — Observed (pending)',
                           fontsize=12, fontweight='bold', color='#999')
        ax3L.set_xlabel('Time (min)'); ax3L.set_ylabel('Cell area (px²)')
        td_obs = self.real.get('doubling_time_min', 0)
        if td_obs > 0:
            ax3L.text(0.95, 0.05, f'$T_d$ = {td_obs:.1f} min',
                      transform=ax3L.transAxes, ha='right', fontsize=11,
                      fontweight='bold', color=self.C_OBS,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        cell_traces = {}
        for step in self.sim['timesteps']:
            t = step['time'] / 60.0
            for cell in step['cells']:
                cid = cell['id']
                if cid not in cell_traces:
                    cell_traces[cid] = {'times': [], 'lengths': []}
                cell_traces[cid]['times'].append(t)
                cell_traces[cid]['lengths'].append(cell['length'])
        longest_sim = sorted(cell_traces.items(),
                             key=lambda x: len(x[1]['times']), reverse=True)[:6]
        cmap_s = plt.cm.get_cmap('Dark2', 8)
        for i, (cid, trace) in enumerate(longest_sim):
            short_id = cid.split('_')[1][:6] if '_' in cid else cid[:6]
            ax3R.plot(trace['times'], trace['lengths'], '-', color=cmap_s(i),
                      linewidth=1.5, alpha=0.85, label=short_id)
        ax3R.legend(fontsize=7, ncol=2, frameon=True)
        ax3R.set_xlabel('Time (min)'); ax3R.set_ylabel('Cell length (µm)')
        ax3R.set_title('Simulated Cell Growth — Predicted', fontsize=12,
                       fontweight='bold', color=self.C_SIM)
        td_sim = self._estimate_sim_doubling_time()
        if td_sim > 0:
            ax3R.text(0.95, 0.05, f'$T_d$ = {td_sim:.1f} min',
                      transform=ax3R.transAxes, ha='right', fontsize=11,
                      fontweight='bold', color=self.C_SIM,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        for ax in (ax3L, ax3R):
            ax.grid(True, alpha=0.3, color=self.C_GRID)

        # --- Row 4: Population growth + quantitative comparison ---
        ax4L = fig.add_subplot(gs[3, 0])
        ax4R = fig.add_subplot(gs[3, 1])
        ax4L.set_facecolor('#EBF5FB')
        ax4R.set_facecolor('#F5EBF5')

        sim_times = [s['time'] / 3600.0 for s in self.sim['timesteps']]
        sim_counts = [s['n_cells'] for s in self.sim['timesteps']]
        ax4L.plot(sim_times, sim_counts, color=self.C_SIM, linewidth=2.2,
                  label='Calibrated DT')
        ax4L.fill_between(sim_times, 0, sim_counts, color=self.C_SIM, alpha=0.1)
        if self.default_sim:
            def_times = [s['time'] / 3600.0 for s in self.default_sim['timesteps']]
            def_counts = [s['n_cells'] for s in self.default_sim['timesteps']]
            ax4L.plot(def_times, def_counts, '--', color=self.C_DEF, linewidth=1.8,
                      label='Default E. coli')
        doubling_h = self.real.get('doubling_time_s', 2400) / 3600.0
        if doubling_h > 0:
            th_t = np.linspace(0, max(sim_times), 200)
            th_c = np.exp(np.log(2) / doubling_h * th_t)
            ax4L.plot(th_t, th_c, ':', color=self.C_OBS, linewidth=1.5,
                      label=f'Observed $T_d$={doubling_h*60:.0f}min')
        ax4L.set_xlabel('Time (h)'); ax4L.set_ylabel('Cell count')
        ax4L.set_title('Population Growth Comparison', fontsize=12, fontweight='bold')
        ax4L.legend(fontsize=9, frameon=True)
        ax4L.grid(True, alpha=0.3, color=self.C_GRID)
        ax4L.set_xlim(left=0); ax4L.set_ylim(bottom=0)

        # Quantitative table
        ax4R.axis('off')
        ax4R.set_title('Quantitative Summary', fontsize=12, fontweight='bold')
        real_rate = self.real.get('growth_rate_mean', 0) * 3600
        sim_rates = self._compute_sim_growth_rates()
        sim_rate = np.mean(sim_rates) * 3600 if sim_rates else 0
        real_len = self.real.get('cell_length_mean_px', 0) * 0.065
        sim_lengths = [c['length'] for s in self.sim['timesteps']
                       for c in s['cells'] if c.get('length', 0) > 0]
        table_data = [
            ['Growth rate (h⁻¹)', f'{real_rate:.3f}', f'{sim_rate:.3f}'],
            ['Doubling time (min)', f'{td_obs:.1f}', f'{td_sim:.1f}'],
            ['Cell length (µm)', f'{real_len:.2f}',
             f'{np.mean(sim_lengths):.2f}' if sim_lengths else '—'],
            ['Cells analyzed', f'{self.real.get("n_cells", 0)}',
             f'{self.sim.get("final_cell_count", 0)}'],
            ['Dividing cells', f'{self.real.get("n_dividing_cells", 0)}', '—'],
        ]
        table = ax4R.table(cellText=table_data,
                           colLabels=['Parameter', 'Observed', 'Predicted'],
                           loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 1.8)
        for i in range(3):
            table[0, i].set_facecolor('#2C3E50')
            table[0, i].set_text_props(color='white', fontweight='bold')
        for row in range(1, len(table_data) + 1):
            table[row, 0].set_facecolor('#F8F9FA')
            table[row, 1].set_facecolor('#EBF5FB')
            table[row, 2].set_facecolor('#F5EBF5')
            for col in range(3):
                table[row, col].set_edgecolor('#DEE2E6')

        return self._save(fig, 'fig_side_by_side_validation.png')

    def _sim_density_grid(self, nx: int, ny: int) -> np.ndarray:
        grid = np.zeros((ny, nx))
        for step in self.sim['timesteps']:
            for cell in step['cells']:
                loc = cell.get('location', [0, 0])
                xi = min(int(loc[0] / (self.sim['config']['env_size'] / nx)), nx - 1)
                yi = min(int(loc[1] / (self.sim['config']['env_size'] / ny)), ny - 1)
                xi = max(0, xi); yi = max(0, yi)
                grid[yi, xi] += 1
        return grid

    def _obs_density_grid(self, tracks: List[Dict], nx: int, ny: int) -> np.ndarray:
        all_x, all_y = [], []
        for tr in tracks:
            all_x.extend(tr.get('x', []))
            all_y.extend(tr.get('y', []))
        if not all_x:
            return np.zeros((ny, nx))
        all_x, all_y = np.array(all_x), np.array(all_y)
        grid, _, _ = np.histogram2d(all_x, all_y, bins=[nx, ny])
        return grid.T

    def _draw_obs_trajectories(self, ax, tracks: List[Dict], max_tracks: int = 20):
        longest = sorted(tracks, key=lambda t: len(t.get('x', [])), reverse=True)[:max_tracks]
        cmap = __import__('matplotlib').cm.plasma
        for tr in longest:
            xs, ys = tr.get('x', []), tr.get('y', [])
            if len(xs) < 3:
                continue
            xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
            dx = np.diff(xs); dy = np.diff(ys)
            speed = np.sqrt(dx**2 + dy**2)
            norm = speed / (speed.max() + 1e-9)
            for j in range(len(xs) - 1):
                ax.plot([xs[j], xs[j+1]], [ys[j], ys[j+1]],
                        color=cmap(norm[j]), linewidth=1.2, alpha=0.8)
        if longest:
            xs0 = [tr['x'][0] for tr in longest if tr.get('x')]
            ys0 = [tr['y'][0] for tr in longest if tr.get('y')]
            ax.set_xlim(min(xs0) - 50, max(xs0) + 200)
            ax.set_ylim(min(ys0) - 50, max(ys0) + 200)
        ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)')

    def _draw_sim_trajectories(self, ax, max_tracks: int = 20):
        cell_paths = {}
        for step in self.sim['timesteps']:
            for cell in step['cells']:
                cid = cell['id']
                loc = cell.get('location', [0, 0])
                if cid not in cell_paths:
                    cell_paths[cid] = {'x': [], 'y': []}
                cell_paths[cid]['x'].append(loc[0])
                cell_paths[cid]['y'].append(loc[1])
        longest = sorted(cell_paths.items(),
                         key=lambda x: len(x[1]['x']), reverse=True)[:max_tracks]
        cmap = __import__('matplotlib').cm.plasma
        for cid, path in longest:
            xs, ys = np.array(path['x']), np.array(path['y'])
            if len(xs) < 3:
                continue
            dx = np.diff(xs); dy = np.diff(ys)
            speed = np.sqrt(dx**2 + dy**2)
            norm = speed / (speed.max() + 1e-9)
            for j in range(len(xs) - 1):
                ax.plot([xs[j], xs[j+1]], [ys[j], ys[j+1]],
                        color=cmap(norm[j]), linewidth=1.2, alpha=0.8)
        ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)')

    def _plot_population_growth(self) -> str:
        plt = self._pub_style()

        fig, ax = plt.subplots(figsize=(8, 5))

        sim_times = [s['time'] / 3600.0 for s in self.sim['timesteps']]
        sim_counts = [s['n_cells'] for s in self.sim['timesteps']]

        ax.plot(sim_times, sim_counts, color=self.C_SIM, linewidth=2.5,
                label='Calibrated Digital Twin', zorder=3)

        if self.default_sim:
            def_times = [s['time'] / 3600.0 for s in self.default_sim['timesteps']]
            def_counts = [s['n_cells'] for s in self.default_sim['timesteps']]
            ax.plot(def_times, def_counts, color=self.C_DEF, linewidth=2,
                    linestyle='--', label='Default E. coli', zorder=2)

        doubling_time_h = self.real.get('doubling_time_s', 2400) / 3600.0
        if doubling_time_h > 0:
            theory_times = np.linspace(0, max(sim_times), 200)
            theory_counts = np.exp(np.log(2) / doubling_time_h * theory_times)
            ax.plot(theory_times, theory_counts, color=self.C_OBS, linewidth=1.8,
                    linestyle=':', label=f'Theoretical ($T_d$={doubling_time_h*60:.0f} min)',
                    zorder=1)

        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Cell count')
        ax.set_title('Population Growth Dynamics')
        ax.legend(loc='upper left', frameon=True)
        ax.grid(True, alpha=0.4, color=self.C_GRID, linestyle='-', linewidth=0.5)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        return self._save(fig, 'fig_population_growth.png')

    def _plot_cell_size_distributions(self) -> str:
        plt = self._pub_style()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        sim_lengths = [c['length'] for s in self.sim['timesteps']
                       for c in s['cells'] if c.get('length', 0) > 0]

        ax = axes[0]
        if sim_lengths:
            ax.hist(sim_lengths, bins=50, color=self.C_SIM, alpha=0.75,
                    density=True, edgecolor='white', linewidth=0.5,
                    label=f'Simulated ($n$={len(sim_lengths):,})')
        real_length_um = self.real.get('cell_length_mean_px', 0) * 0.065
        if real_length_um > 0:
            ax.axvline(real_length_um, color=self.C_OBS, linewidth=2.5,
                       linestyle='--', label=f'Observed mean: {real_length_um:.2f} µm',
                       zorder=5)
        ax.set_xlabel('Cell length (µm)')
        ax.set_ylabel('Probability density')
        ax.set_title('Cell Length Distribution')
        ax.legend(frameon=True)
        ax.grid(axis='y', alpha=0.3, color=self.C_GRID)

        sim_masses = [c['mass'] for s in self.sim['timesteps']
                      for c in s['cells'] if c.get('mass', 0) > 0]

        ax = axes[1]
        if sim_masses:
            ax.hist(sim_masses, bins=50, color=self.C_SIM, alpha=0.75,
                    density=True, edgecolor='white', linewidth=0.5,
                    label=f'Simulated ($n$={len(sim_masses):,})')
        ax.set_xlabel('Cell mass (a.u.)')
        ax.set_ylabel('Probability density')
        ax.set_title('Cell Mass Distribution')
        ax.legend(frameon=True)
        ax.grid(axis='y', alpha=0.3, color=self.C_GRID)

        fig.tight_layout(w_pad=3)
        return self._save(fig, 'fig_size_distributions.png')

    def _plot_growth_rate_comparison(self) -> str:
        plt = self._pub_style()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        sim_growth_rates = self._compute_sim_growth_rates()

        ax = axes[0]
        real_rates = self.real.get('growth_rates_all', [])
        if real_rates:
            real_h = [r * 3600 for r in real_rates]
            ax.hist(real_h, bins=25, color=self.C_OBS, alpha=0.7, density=True,
                    edgecolor='white', linewidth=0.5,
                    label=f'Observed ($n$={len(real_rates)})')
        if sim_growth_rates:
            sim_h = [r * 3600 for r in sim_growth_rates]
            ax.hist(sim_h, bins=25, color=self.C_SIM, alpha=0.6, density=True,
                    edgecolor='white', linewidth=0.5,
                    label=f'Simulated ($n$={len(sim_growth_rates):,})')
        ax.set_xlabel('Growth rate ($h^{-1}$)')
        ax.set_ylabel('Probability density')
        ax.set_title('Growth Rate Distribution')
        ax.legend(frameon=True)
        ax.grid(axis='y', alpha=0.3, color=self.C_GRID)

        ax = axes[1]
        real_mean = self.real.get('growth_rate_mean', 0) * 3600
        real_std = self.real.get('growth_rate_std', 0) * 3600
        sim_mean = np.mean(sim_growth_rates) * 3600 if sim_growth_rates else 0
        sim_std = np.std(sim_growth_rates) * 3600 if sim_growth_rates else 0
        default_rate = 0.000289 * 3600

        x = np.arange(3)
        means = [real_mean, sim_mean, default_rate]
        stds = [real_std, sim_std, 0]
        colors = [self.C_OBS, self.C_SIM, self.C_DEF]
        labels = ['Observed', 'Calibrated\nSimulation', 'Default\nE. coli']

        bars = ax.bar(x, means, yerr=stds, width=0.6, color=colors,
                      edgecolor='white', linewidth=1.5, capsize=6,
                      error_kw={'linewidth': 1.5, 'capthick': 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Growth rate ($h^{-1}$)')
        ax.set_title('Mean Growth Rate Comparison')
        ax.grid(axis='y', alpha=0.3, color=self.C_GRID)

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds)*0.1,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        fig.tight_layout(w_pad=3)
        return self._save(fig, 'fig_growth_rates.png')

    def _plot_division_timing(self) -> str:
        plt = self._pub_style()

        fig, ax = plt.subplots(figsize=(6, 5))

        real_idt = self.real.get('interdivision_time_mean_min', 0)
        real_std = self.real.get('interdivision_time_std_s', 0) / 60.0
        sim_doubling = self._estimate_sim_doubling_time()
        default_doubling = 40.0

        x = np.arange(3)
        values = [real_idt, sim_doubling, default_doubling]
        stds = [real_std, 0, 0]
        colors = [self.C_OBS, self.C_SIM, self.C_DEF]
        labels = ['Observed', 'Calibrated\nSimulation', 'Default\nE. coli']

        bars = ax.bar(x, values, yerr=stds, width=0.55, color=colors,
                      edgecolor='white', linewidth=1.5, capsize=6,
                      error_kw={'linewidth': 1.5, 'capthick': 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Interdivision time (min)')
        ax.set_title('Division Timing Comparison')
        ax.grid(axis='y', alpha=0.3, color=self.C_GRID)
        ax.set_ylim(bottom=0)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        fig.tight_layout()
        return self._save(fig, 'fig_division_timing.png')

    def _plot_single_cell_growth_curves(self) -> str:
        plt = self._pub_style()
        from matplotlib.lines import Line2D

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        cmap_obs = plt.cm.get_cmap('Set2', 8)
        cmap_sim = plt.cm.get_cmap('Dark2', 8)

        ax = axes[0]
        cells = list(self.cell_database.values())
        longest = sorted(cells, key=lambda c: c['lifespan'], reverse=True)[:6]
        for i, cell in enumerate(longest):
            areas = np.array(cell['area'], dtype=float)
            valid = ~np.isnan(areas)
            times = np.arange(len(areas))[valid] * self.real.get('time_interval_s', 300) / 60.0
            ax.plot(times, areas[valid], '-o', color=cmap_obs(i), linewidth=1.8,
                    markersize=3, alpha=0.85, label=f"Cell {cell['cell_id']}")
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Area (px²)')
        ax.set_title('Observed — Single Cell Growth')
        ax.legend(frameon=True, fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, color=self.C_GRID)

        ax = axes[1]
        cell_traces = {}
        for step in self.sim['timesteps']:
            t = step['time'] / 60.0
            for cell in step['cells']:
                cid = cell['id']
                if cid not in cell_traces:
                    cell_traces[cid] = {'times': [], 'lengths': []}
                cell_traces[cid]['times'].append(t)
                cell_traces[cid]['lengths'].append(cell['length'])

        longest_sim = sorted(cell_traces.items(),
                             key=lambda x: len(x[1]['times']), reverse=True)[:6]
        for i, (cid, trace) in enumerate(longest_sim):
            short_id = cid.split('_')[1][:6] if '_' in cid else cid[:6]
            ax.plot(trace['times'], trace['lengths'], '-', color=cmap_sim(i),
                    linewidth=1.8, alpha=0.85, label=short_id)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Cell length (µm)')
        ax.set_title('Simulated — Single Cell Growth')
        ax.legend(frameon=True, fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, color=self.C_GRID)

        fig.tight_layout(w_pad=3)
        return self._save(fig, 'fig_single_cell_growth.png')

    def _plot_summary_dashboard(self) -> str:
        plt = self._pub_style()

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Digital Twin Validation Dashboard',
                     fontsize=18, fontweight='bold', y=0.98)
        fig.text(0.5, 0.945,
                 'Observed microscopy parameters vs. Viva-munk calibrated simulation',
                 ha='center', fontsize=11, color='#666666', style='italic')

        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35,
                              top=0.9, bottom=0.08, left=0.06, right=0.97)

        # 1. Population growth
        ax1 = fig.add_subplot(gs[0, 0])
        sim_times = [s['time'] / 3600.0 for s in self.sim['timesteps']]
        sim_counts = [s['n_cells'] for s in self.sim['timesteps']]
        ax1.plot(sim_times, sim_counts, color=self.C_SIM, linewidth=2.2)
        ax1.fill_between(sim_times, 0, sim_counts, color=self.C_SIM, alpha=0.1)
        doubling_time_h = self.real.get('doubling_time_s', 2400) / 3600.0
        if doubling_time_h > 0:
            th_t = np.linspace(0, max(sim_times), 200)
            th_c = np.exp(np.log(2) / doubling_time_h * th_t)
            ax1.plot(th_t, th_c, ':', color=self.C_OBS, linewidth=1.5)
        ax1.set_title('Population Growth')
        ax1.set_xlabel('Time (h)')
        ax1.set_ylabel('Cell count')
        ax1.grid(True, alpha=0.3, color=self.C_GRID)
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)

        # 2. Growth rate
        ax2 = fig.add_subplot(gs[0, 1])
        real_rate = self.real.get('growth_rate_mean', 0) * 3600
        sim_rates = self._compute_sim_growth_rates()
        sim_rate = np.mean(sim_rates) * 3600 if sim_rates else 0
        default_rate = 0.000289 * 3600
        x = np.arange(3)
        bars = ax2.bar(x, [real_rate, sim_rate, default_rate], width=0.55,
                       color=[self.C_OBS, self.C_SIM, self.C_DEF],
                       edgecolor='white', linewidth=1.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Observed', 'Simulated', 'Default'], fontsize=9)
        ax2.set_title('Growth Rate ($h^{-1}$)')
        ax2.grid(axis='y', alpha=0.3, color=self.C_GRID)
        for bar, val in zip(bars, [real_rate, sim_rate, default_rate]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 3. Division timing
        ax3 = fig.add_subplot(gs[0, 2])
        real_idt = self.real.get('interdivision_time_mean_min', 0)
        sim_dt = self._estimate_sim_doubling_time()
        x = np.arange(3)
        bars = ax3.bar(x, [real_idt, sim_dt, 40.0], width=0.55,
                       color=[self.C_OBS, self.C_SIM, self.C_DEF],
                       edgecolor='white', linewidth=1.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Observed', 'Simulated', 'Default'], fontsize=9)
        ax3.set_title('Interdivision Time (min)')
        ax3.grid(axis='y', alpha=0.3, color=self.C_GRID)
        for bar, val in zip(bars, [real_idt, sim_dt, 40.0]):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 4. Cell length distribution
        ax4 = fig.add_subplot(gs[1, 0])
        sim_lengths = [c['length'] for s in self.sim['timesteps']
                       for c in s['cells'] if c.get('length', 0) > 0]
        if sim_lengths:
            ax4.hist(sim_lengths, bins=40, color=self.C_SIM, alpha=0.75,
                     density=True, edgecolor='white', linewidth=0.5)
        real_len = self.real.get('cell_length_mean_px', 0) * 0.065
        if real_len > 0:
            ax4.axvline(real_len, color=self.C_OBS, linewidth=2.5, linestyle='--')
            ax4.text(real_len * 1.05, ax4.get_ylim()[1] * 0.85,
                     f'Obs: {real_len:.1f} µm', color=self.C_OBS, fontsize=9, fontweight='bold')
        ax4.set_title('Cell Length Distribution')
        ax4.set_xlabel('Length (µm)')
        ax4.set_ylabel('Density')
        ax4.grid(axis='y', alpha=0.3, color=self.C_GRID)

        # 5. Parameter comparison table
        ax5 = fig.add_subplot(gs[1, 1:])
        ax5.axis('off')
        ax5.set_title('Quantitative Comparison', pad=15)

        table_data = [
            ['Growth rate ($h^{-1}$)',
             f"{real_rate:.3f}",
             f"{sim_rate:.3f}" if sim_rate else "—",
             f"{default_rate:.3f}"],
            ['Doubling time (min)',
             f"{self.real.get('doubling_time_min', 0):.1f}",
             f"{sim_dt:.1f}" if sim_dt else "—",
             "40.0"],
            ['Cell length (µm)',
             f"{real_len:.2f}",
             f"{np.mean(sim_lengths):.2f}" if sim_lengths else "—",
             "2.0"],
            ['Aspect ratio',
             f"{self.real.get('aspect_ratio_mean', 0):.2f}",
             "—", "—"],
            ['Dividing cells',
             f"{self.real.get('n_dividing_cells', 0)}",
             f"{self.sim.get('final_cell_count', 0)}",
             "—"],
            ['Total cells analyzed',
             f"{self.real.get('n_cells', 0)}",
             f"{self.sim.get('final_cell_count', 0)} final",
             "—"],
        ]

        col_labels = ['Parameter', 'Observed', 'Calibrated DT', 'Default E. coli']
        table = ax5.table(cellText=table_data, colLabels=col_labels,
                          loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.6)

        for i in range(len(col_labels)):
            table[0, i].set_facecolor('#2C3E50')
            table[0, i].set_text_props(color='white', fontweight='bold', fontsize=10)
        for row in range(1, len(table_data) + 1):
            bg = '#F8F9FA' if row % 2 == 0 else 'white'
            for col in range(len(col_labels)):
                table[row, col].set_facecolor(bg)
                table[row, col].set_edgecolor('#DEE2E6')

        return self._save(fig, 'fig_validation_dashboard.png')

    def _compute_sim_growth_rates(self) -> List[float]:
        """Estimate per-cell growth rates from simulation data."""
        cell_traces = {}
        for step in self.sim['timesteps']:
            t = step['time']
            for cell in step['cells']:
                cid = cell['id']
                if cid not in cell_traces:
                    cell_traces[cid] = {'times': [], 'lengths': []}
                cell_traces[cid]['times'].append(t)
                cell_traces[cid]['lengths'].append(cell['length'])

        rates = []
        for cid, trace in cell_traces.items():
            if len(trace['times']) < 3:
                continue
            times = np.array(trace['times'])
            lengths = np.array(trace['lengths'])
            valid = lengths > 0
            if valid.sum() < 3:
                continue
            ln_L = np.log(lengths[valid])
            t = times[valid]
            coeffs = np.polyfit(t, ln_L, 1)
            if 0 < coeffs[0] < 0.01:
                rates.append(coeffs[0])
        return rates

    def _estimate_sim_doubling_time(self) -> float:
        """Estimate population doubling time from simulation."""
        timesteps = self.sim['timesteps']
        if len(timesteps) < 2:
            return 0.0

        first_count = max(1, timesteps[0]['n_cells'])
        last_count = timesteps[-1]['n_cells']
        total_time = timesteps[-1]['time'] - timesteps[0]['time']

        if last_count <= first_count or total_time <= 0:
            return 0.0

        n_doublings = math.log2(last_count / first_count)
        if n_doublings <= 0:
            return 0.0

        return (total_time / n_doublings) / 60.0


def run_digital_twin_pipeline(cell_database: Dict,
                               time_interval_seconds: float = 300.0,
                               pixel_to_um: float = 0.065,
                               experiment_type: str = 'daughter_machine') -> Dict:
    """Run the complete digital twin pipeline.

    Parameters
    ----------
    cell_database : dict
        Output from CellHistoryBuilder.build()
    time_interval_seconds : float
        Time between microscopy frames in seconds
    pixel_to_um : float
        Pixel to micrometer conversion factor
    experiment_type : str
        Which Viva-munk experiment to run

    Returns
    -------
    dict with keys: params, config, sim_results, figure_paths
    """
    print("=" * 60)
    print("DIGITAL TWIN PIPELINE")
    print("=" * 60)

    # Step 1: Extract parameters
    print("\n[1/4] Extracting real cell parameters...")
    extractor = ParameterExtractor(cell_database)
    params = extractor.extract_all(time_interval_seconds)
    print(f"  Growth rate: {params.get('growth_rate_mean', 0)*3600:.4f} /h")
    print(f"  Doubling time: {params.get('doubling_time_min', 0):.1f} min")
    print(f"  Cell length: {params.get('cell_length_mean_px', 0):.1f} px = {params.get('cell_length_mean_px', 0)*pixel_to_um:.2f} µm")
    print(f"  Dividing cells: {params.get('n_dividing_cells', 0)}")

    # Step 2: Build calibrated config and run simulation
    print("\n[2/4] Running calibrated simulation...")
    runner = DigitalTwinRunner(params, pixel_to_um)
    config = runner.build_calibrated_config()
    print(f"  Config: growth_rate={config['growth_rate']:.6f}, "
          f"cell_length={config['cell_length']:.2f}µm, "
          f"sim_time={config['total_time']/3600:.1f}h")

    sim_results = runner.run_simulation(experiment_type)
    print(f"  Final cells: {sim_results.get('final_cell_count', 0)}")

    # Step 3: Run default simulation for comparison
    print("\n[3/4] Running default simulation for comparison...")
    default_results = runner.run_default_simulation()
    print(f"  Default final cells: {default_results.get('final_cell_count', 0)}")

    # Step 4: Generate comparison figures
    print("\n[4/4] Generating comparison figures...")
    generator = ComparisonFigureGenerator(
        params, sim_results, default_results, cell_database
    )
    figure_paths = generator.generate_all()
    print(f"  Generated {len(figure_paths)} figures:")
    for p in figure_paths:
        print(f"    - {os.path.basename(p)}")

    print("\n" + "=" * 60)
    print("DIGITAL TWIN PIPELINE COMPLETE")
    print(f"Output directory: {DT_OUTPUT_DIR}")
    print("=" * 60)

    return {
        'params': params,
        'config': config,
        'sim_results': sim_results,
        'default_sim_results': default_results,
        'figure_paths': figure_paths,
    }
