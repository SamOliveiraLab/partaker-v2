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
    """Generates comparison figures: simulated vs observed."""

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

    def generate_all(self) -> List[str]:
        """Generate all comparison figures. Returns list of saved paths."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        paths = []
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

    def _plot_population_growth(self) -> str:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        sim_times = [s['time'] / 3600.0 for s in self.sim['timesteps']]
        sim_counts = [s['n_cells'] for s in self.sim['timesteps']]
        ax.plot(sim_times, sim_counts, 'b-', linewidth=2, label='Calibrated Simulation')

        if self.default_sim:
            def_times = [s['time'] / 3600.0 for s in self.default_sim['timesteps']]
            def_counts = [s['n_cells'] for s in self.default_sim['timesteps']]
            ax.plot(def_times, def_counts, 'r--', linewidth=1.5, label='Default (E. coli)')

        doubling_time_h = self.real.get('doubling_time_s', 2400) / 3600.0
        if doubling_time_h > 0:
            theory_times = np.linspace(0, max(sim_times), 100)
            theory_counts = np.exp(np.log(2) / doubling_time_h * theory_times)
            ax.plot(theory_times, theory_counts, 'g:', linewidth=1.5,
                    label=f'Theoretical (Td={doubling_time_h*60:.0f}min)')

        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Cell Count', fontsize=12)
        ax.set_title('Population Growth: Simulated vs Theoretical', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.output_dir, 'fig_population_growth.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return path

    def _plot_cell_size_distributions(self) -> str:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sim_lengths = []
        for step in self.sim['timesteps']:
            for cell in step['cells']:
                L = cell.get('length', 0)
                if L > 0:
                    sim_lengths.append(L)

        ax = axes[0]
        if sim_lengths:
            ax.hist(sim_lengths, bins=40, alpha=0.7, color='blue', density=True,
                    label=f'Simulated (n={len(sim_lengths)})')
        real_length_um = self.real.get('cell_length_mean_px', 0) * 0.065
        if real_length_um > 0:
            ax.axvline(real_length_um, color='red', linewidth=2, linestyle='--',
                       label=f'Observed mean: {real_length_um:.2f} µm')
        ax.set_xlabel('Cell Length (µm)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Cell Length Distribution', fontsize=12)
        ax.legend(fontsize=9)

        sim_masses = []
        for step in self.sim['timesteps']:
            for cell in step['cells']:
                m = cell.get('mass', 0)
                if m > 0:
                    sim_masses.append(m)

        ax = axes[1]
        if sim_masses:
            ax.hist(sim_masses, bins=40, alpha=0.7, color='blue', density=True,
                    label=f'Simulated (n={len(sim_masses)})')
        ax.set_xlabel('Cell Mass', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Cell Mass Distribution', fontsize=12)
        ax.legend(fontsize=9)

        fig.suptitle('Cell Size Distributions — Calibrated Simulation', fontsize=14, y=1.02)
        path = os.path.join(self.output_dir, 'fig_size_distributions.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return path

    def _plot_growth_rate_comparison(self) -> str:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sim_growth_rates = self._compute_sim_growth_rates()

        ax = axes[0]
        real_rates = self.real.get('growth_rates_all', [])
        if real_rates:
            real_rates_per_h = [r * 3600 for r in real_rates]
            ax.hist(real_rates_per_h, bins=20, alpha=0.6, color='green', density=True,
                    label=f'Observed (n={len(real_rates)})')
        if sim_growth_rates:
            sim_rates_per_h = [r * 3600 for r in sim_growth_rates]
            ax.hist(sim_rates_per_h, bins=20, alpha=0.6, color='blue', density=True,
                    label=f'Simulated (n={len(sim_growth_rates)})')
        ax.set_xlabel('Growth Rate (1/h)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Growth Rate Distribution', fontsize=12)
        ax.legend(fontsize=9)

        ax = axes[1]
        real_mean = self.real.get('growth_rate_mean', 0) * 3600
        real_std = self.real.get('growth_rate_std', 0) * 3600
        sim_mean = np.mean(sim_growth_rates) * 3600 if sim_growth_rates else 0
        sim_std = np.std(sim_growth_rates) * 3600 if sim_growth_rates else 0
        default_rate = 0.000289 * 3600

        categories = ['Observed', 'Calibrated\nSimulation', 'Default\n(E. coli)']
        means = [real_mean, sim_mean, default_rate]
        stds = [real_std, sim_std, 0]
        colors = ['green', 'blue', 'red']

        bars = ax.bar(categories, means, yerr=stds, color=colors, alpha=0.7,
                      capsize=5, edgecolor='black')
        ax.set_ylabel('Growth Rate (1/h)', fontsize=11)
        ax.set_title('Mean Growth Rate Comparison', fontsize=12)

        path = os.path.join(self.output_dir, 'fig_growth_rates.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return path

    def _plot_division_timing(self) -> str:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        real_idt = self.real.get('interdivision_time_mean_min', 0)
        real_std = self.real.get('interdivision_time_std_s', 0) / 60.0

        sim_doubling = self._estimate_sim_doubling_time()
        default_doubling = 40.0

        categories = ['Observed', 'Calibrated\nSimulation', 'Default\n(E. coli)']
        values = [real_idt, sim_doubling, default_doubling]
        stds = [real_std, 0, 0]
        colors = ['green', 'blue', 'red']

        ax.bar(categories, values, yerr=stds, color=colors, alpha=0.7,
               capsize=5, edgecolor='black')
        ax.set_ylabel('Doubling / Interdivision Time (min)', fontsize=11)
        ax.set_title('Division Timing: Observed vs Simulated', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        path = os.path.join(self.output_dir, 'fig_division_timing.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return path

    def _plot_single_cell_growth_curves(self) -> str:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        cells = list(self.cell_database.values())
        longest = sorted(cells, key=lambda c: c['lifespan'], reverse=True)[:5]
        for cell in longest:
            areas = np.array(cell['area'], dtype=float)
            valid = ~np.isnan(areas)
            times = np.arange(len(areas))[valid] * self.real.get('time_interval_s', 300) / 60.0
            ax.plot(times, areas[valid], '-', alpha=0.7,
                    label=f"Cell {cell['cell_id']}")
        ax.set_xlabel('Time (min)', fontsize=11)
        ax.set_ylabel('Area (px²)', fontsize=11)
        ax.set_title('Observed: Single Cell Growth', fontsize=12)
        ax.legend(fontsize=8)

        ax = axes[1]
        for step in self.sim['timesteps'][:1]:
            cell_ids = [c['id'] for c in step['cells']]

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
                             key=lambda x: len(x[1]['times']), reverse=True)[:5]
        for cid, trace in longest_sim:
            ax.plot(trace['times'], trace['lengths'], '-', alpha=0.7,
                    label=f'{cid[:12]}...' if len(cid) > 12 else cid)
        ax.set_xlabel('Time (min)', fontsize=11)
        ax.set_ylabel('Cell Length (µm)', fontsize=11)
        ax.set_title('Simulated: Single Cell Growth', fontsize=12)
        ax.legend(fontsize=8)

        fig.suptitle('Single Cell Growth Curves — Observed vs Simulated', fontsize=14, y=1.02)
        path = os.path.join(self.output_dir, 'fig_single_cell_growth.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return path

    def _plot_summary_dashboard(self) -> str:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Digital Twin Validation Dashboard', fontsize=16, fontweight='bold')

        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        # 1. Population growth
        ax1 = fig.add_subplot(gs[0, 0])
        sim_times = [s['time'] / 3600.0 for s in self.sim['timesteps']]
        sim_counts = [s['n_cells'] for s in self.sim['timesteps']]
        ax1.plot(sim_times, sim_counts, 'b-', linewidth=2)
        ax1.set_title('Population Growth', fontsize=11)
        ax1.set_xlabel('Time (h)')
        ax1.set_ylabel('Cells')

        # 2. Growth rate comparison
        ax2 = fig.add_subplot(gs[0, 1])
        real_rate = self.real.get('growth_rate_mean', 0) * 3600
        sim_rates = self._compute_sim_growth_rates()
        sim_rate = np.mean(sim_rates) * 3600 if sim_rates else 0
        ax2.bar(['Observed', 'Simulated'], [real_rate, sim_rate],
                color=['green', 'blue'], alpha=0.7, edgecolor='black')
        ax2.set_title('Growth Rate (1/h)', fontsize=11)

        # 3. Division timing
        ax3 = fig.add_subplot(gs[0, 2])
        real_idt = self.real.get('interdivision_time_mean_min', 0)
        sim_dt = self._estimate_sim_doubling_time()
        ax3.bar(['Observed', 'Simulated'], [real_idt, sim_dt],
                color=['green', 'blue'], alpha=0.7, edgecolor='black')
        ax3.set_title('Interdivision Time (min)', fontsize=11)

        # 4. Cell size
        ax4 = fig.add_subplot(gs[1, 0])
        sim_lengths = [c['length'] for s in self.sim['timesteps'] for c in s['cells'] if c['length'] > 0]
        if sim_lengths:
            ax4.hist(sim_lengths, bins=30, alpha=0.7, color='blue', density=True)
        ax4.set_title('Sim Cell Lengths', fontsize=11)
        ax4.set_xlabel('Length (µm)')

        # 5. Parameter table
        ax5 = fig.add_subplot(gs[1, 1:])
        ax5.axis('off')
        table_data = [
            ['Parameter', 'Observed', 'Simulated', 'Default E. coli'],
            ['Growth rate (1/h)',
             f"{self.real.get('growth_rate_mean', 0)*3600:.4f}",
             f"{sim_rate:.4f}" if sim_rate else "N/A",
             f"{0.000289*3600:.4f}"],
            ['Doubling time (min)',
             f"{self.real.get('doubling_time_min', 0):.1f}",
             f"{sim_dt:.1f}" if sim_dt else "N/A",
             "40.0"],
            ['Cell length (µm)',
             f"{self.real.get('cell_length_mean_px', 0)*0.065:.2f}",
             f"{np.mean(sim_lengths):.2f}" if sim_lengths else "N/A",
             "2.0"],
            ['Dividing cells',
             f"{self.real.get('n_dividing_cells', 0)}",
             f"{self.sim.get('final_cell_count', 0)}",
             "N/A"],
            ['Cells analyzed',
             f"{self.real.get('n_cells', 0)}",
             f"{self.sim.get('final_cell_count', 0)} final",
             "N/A"],
        ]

        table = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                          loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)

        for i in range(len(table_data[0])):
            table[0, i].set_facecolor('#4472C4')
            table[0, i].set_text_props(color='white', fontweight='bold')

        path = os.path.join(self.output_dir, 'fig_validation_dashboard.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return path

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
