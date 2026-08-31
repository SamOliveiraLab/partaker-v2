import json
import os
from pathlib import Path
import re
from typing import List, Dict, Tuple, Optional

from nd2_analyzer.data.experiment import Experiment
from nd2_analyzer.data.image_data import ImageData
from nd2_analyzer.ui.biofilms.colony_separator import ColonySeparator

from nd2_analyzer.ui.widgets import SegmentationWidget
import numpy as np
import pandas as pd
import cv2

import polars as pl
import json

class ColonyMetricExporter:
    """Creates metrics for single-cells in microcolonies for biofilm analysis"""
    def __init__(self, image_data, colonies, segmentation_storage, model_name, voxel_size):
        self.image_data = image_data
        self.colonies = colonies
        self.segmented_storage = segmentation_storage
        self.model_name = model_name
        self.voxel_size = voxel_size
        self.biofilm_metric_service = image_data.biofilm_metric_service

    def build_table(self):
        """Build a table of colony and cell metrics to export to CSV"""
        if isinstance(self.colonies, dict):
            stored_frames = self.biofilm_metric_service.get_colony_frame_results()
            stored_keys = {
                (row["position"], row["time"], row["channel"])
                for row in stored_frames.to_dicts()
            }
            requested_keys = {
                tuple(int(value) for value in key) for key in self.colonies
            }
            expected_radius = float(
                getattr(
                    self.biofilm_metric_service,
                    "DEFAULT_NEIGHBOR_RADIUS_PX",
                    150.0,
                )
            )
            stored_radii = {
                float(row["neighbor_radius_px"])
                for row in stored_frames.to_dicts()
                if row.get("neighbor_radius_px") is not None
            }
            if (
                stored_keys == requested_keys
                and stored_radii == {expected_radius}
            ):
                return self.biofilm_metric_service.get_colony_results()
        return self.biofilm_metric_service.build_colony_table(
            self.colonies,
            model_name=self.model_name,
            voxel_size=self.voxel_size,
        )

    def export_csv(self, path: str):
        """Export colony metrics and their time-resolved companion tables."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.build_table()
        df.write_csv(output_path)

        companion_tables = {
            "frames": self.biofilm_metric_service.get_colony_frame_results(),
            "cells": self.biofilm_metric_service.get_colony_cell_results(),
            "eps": self.biofilm_metric_service.get_colony_eps_results(),
        }
        for suffix, table in companion_tables.items():
            if table.is_empty():
                continue
            companion_path = output_path.with_name(
                f"{output_path.stem}_{suffix}{output_path.suffix}"
            )
            table.write_csv(companion_path)

    @staticmethod
    def _safe_filename(value) -> str:
        text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
        return text.strip("_") or "unknown"

    @staticmethod
    def _plot_time_distribution(
            ax,
            frame: pd.DataFrame,
            value_column: str,
            *,
            color: str,
            label: str | None = None,
            max_points_per_time: int = 2000,
    ) -> pd.DataFrame:
        """Draw individual observations with a median line and IQR band."""
        import seaborn as sns

        data = frame[["time", value_column]].dropna().copy()
        if data.empty:
            return pd.DataFrame()

        sampled_parts = []
        for _time, time_rows in data.groupby("time", sort=True):
            if len(time_rows) > max_points_per_time:
                time_rows = time_rows.sample(
                    n=max_points_per_time,
                    random_state=0,
                )
            sampled_parts.append(time_rows)
        sampled = pd.concat(sampled_parts, ignore_index=True)

        sns.scatterplot(
            data=sampled,
            x="time",
            y=value_column,
            color=color,
            alpha=0.22,
            s=18,
            linewidth=0,
            legend=False,
            ax=ax,
        )

        grouped = data.groupby("time", sort=True)[value_column]
        summary = grouped.agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            count="size",
        ).reset_index()
        sns.lineplot(
            data=summary,
            x="time",
            y="median",
            color=color,
            marker="o",
            linewidth=2.2,
            label=label,
            ax=ax,
        )
        ax.fill_between(
            summary["time"].to_numpy(dtype=float),
            summary["q25"].to_numpy(dtype=float),
            summary["q75"].to_numpy(dtype=float),
            color=color,
            alpha=0.18,
        )
        ax.set_xlabel("Timepoint (frame)")
        return summary

    @staticmethod
    def _add_count_note(
            ax,
            summary: pd.DataFrame,
            item_name: str,
            counts_by_time: pd.Series | None = None,
    ) -> None:
        if counts_by_time is not None:
            annotation_rows = counts_by_time.rename("count").reset_index()
        else:
            annotation_rows = summary
        if annotation_rows.empty or "count" not in annotation_rows:
            return
        for row in annotation_rows.itertuples(index=False):
            ax.text(
                float(row.time),
                0.98,
                f"n={int(row.count)}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7,
                color="#444444",
            )

    def _uses_physical_units(self) -> bool:
        return bool(
            self.voxel_size is not None
            and getattr(self.voxel_size, "x", None)
            and getattr(self.voxel_size, "y", None)
        )

    def _plot_local_biomass(
            self,
            colonies: pd.DataFrame,
            eps_rows: pd.DataFrame,
            output_path: Path,
            *,
            position: int,
            colony_channel: int,
            eps_channel: int,
            analysis_method: str,
    ) -> bool:
        import matplotlib.pyplot as plt
        import seaborn as sns

        joined = colonies.merge(
            eps_rows,
            left_on=["position", "time", "channel", "colony_id"],
            right_on=["position", "time", "colony_channel", "colony_id"],
            how="inner",
        )
        if joined.empty:
            return False

        use_physical = self._uses_physical_units()
        cell_area = "cell_biomass_area_um2" if use_physical else "cell_biomass_area_px"
        eps_area = "eps_biomass_area_um2" if use_physical else "eps_biomass_area_px"
        if cell_area not in joined or eps_area not in joined:
            return False
        joined = joined.dropna(subset=[cell_area, eps_area]).copy()
        if joined.empty:
            return False
        joined["cell_coverage_percent"] = joined["cell_biomass_fraction"] * 100.0
        joined["eps_coverage_percent"] = joined["eps_fraction_of_colony"] * 100.0

        with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
            fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
            self._plot_time_distribution(
                axes[0], joined, cell_area, color="#4C78A8", label="Cells"
            )
            area_summary = self._plot_time_distribution(
                axes[0], joined, eps_area, color="#F58518", label="EPS"
            )
            axes[0].set_title("Cell and EPS Biomass Area")
            axes[0].set_ylabel(f"Biomass Area ({'µm²' if use_physical else 'px²'})")
            axes[0].legend(title="Biomass source")
            self._add_count_note(axes[0], area_summary, "Colonies")

            self._plot_time_distribution(
                axes[1], joined, "cell_coverage_percent",
                color="#72B7B2", label="Cells",
            )
            coverage_summary = self._plot_time_distribution(
                axes[1], joined, "eps_coverage_percent",
                color="#ECA82C", label="EPS",
            )
            axes[1].set_title("Fraction of Microcolony Occupied by Biomass")
            axes[1].set_ylabel("Microcolony Coverage (%)")
            axes[1].set_ylim(bottom=0)
            axes[1].legend(title="Biomass source")
            self._add_count_note(
                axes[1], coverage_summary, "Colonies"
            )

            fig.suptitle(
                "Local Biomass per Microcolony Over Time — "
                f"P{position}, Colony C{colony_channel}, "
                f"EPS C{eps_channel} ({analysis_method})",
                fontsize=14,
                fontweight="bold",
            )
            fig.text(
                0.5,
                0.01,
                "n is the number of microcolonies represented in each frame.",
                ha="center",
                fontsize=8,
                color="#555555",
            )
            fig.tight_layout(rect=(0, 0.03, 1, 0.96))
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return True

    def _plot_spatial_context(
            self,
            colonies: pd.DataFrame,
            output_path: Path,
            *,
            position: int,
            colony_channel: int,
    ) -> bool:
        import matplotlib.pyplot as plt
        import seaborn as sns

        required = {"nearest_colony_distance_px", "neighbor_count", "time"}
        if colonies.empty or not required.issubset(colonies.columns):
            return False
        neighbor_radius_px = 150.0
        if "neighbor_radius_px" in colonies:
            stored_radii = colonies["neighbor_radius_px"].dropna()
            if not stored_radii.empty:
                neighbor_radius_px = float(stored_radii.iloc[0])
        radius_label = f"{neighbor_radius_px:g}"
        colony_counts = colonies.groupby("time", sort=True).size()

        with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
            fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
            distance_summary = self._plot_time_distribution(
                axes[0],
                colonies,
                "nearest_colony_distance_px",
                color="#2A9D8F",
            )
            axes[0].axhline(
                neighbor_radius_px,
                color="#E45756",
                linestyle="--",
                linewidth=1.6,
                label=f"{radius_label} px neighbor radius",
            )
            axes[0].set_title("Nearest Microcolony Centroid Distance")
            axes[0].set_ylabel("Nearest Centroid Distance (px)")
            axes[0].legend()
            self._add_count_note(
                axes[0],
                distance_summary,
                "Colonies",
                counts_by_time=colony_counts,
            )

            neighbor_summary = self._plot_time_distribution(
                axes[1],
                colonies,
                "neighbor_count",
                color="#287271",
            )
            axes[1].set_title(
                f"Neighboring Microcolonies Within {radius_label} Pixels"
            )
            axes[1].set_ylabel("Neighbor Count")
            axes[1].set_ylim(bottom=0)
            self._add_count_note(
                axes[1],
                neighbor_summary,
                "Colonies",
                counts_by_time=colony_counts,
            )

            fig.suptitle(
                "Microcolony Spatial Context Over Time — "
                f"P{position}, Colony C{colony_channel}",
                fontsize=14,
                fontweight="bold",
            )
            fig.text(
                0.5,
                0.01,
                "Nearest distance uses centroid-to-centroid distance; "
                f"neighbor count uses the {radius_label}-pixel radius. "
                "n is the number of microcolonies in each frame.",
                ha="center",
                fontsize=8,
                color="#555555",
            )
            fig.tight_layout(rect=(0, 0.03, 1, 0.96))
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return True

    def _plot_single_cell_measurements(
            self,
            cells: pd.DataFrame,
            output_dir: Path,
            *,
            position: int,
            colony_channel: int,
    ) -> list[Path]:
        import matplotlib.pyplot as plt
        import seaborn as sns

        if cells.empty:
            return []
        use_physical = self._uses_physical_units()
        measurement_pairs = [
            (
                "area_perimeter",
                "Single-Cell Area and Perimeter Within Microcolonies Over Time",
                [
                    (
                        "area_um2" if use_physical else "area_px",
                        "Cell Area",
                        f"Area ({'µm²' if use_physical else 'px²'})",
                        "#4C78A8",
                    ),
                    (
                        "perimeter_um" if use_physical else "perimeter_px",
                        "Cell Perimeter",
                        f"Perimeter ({'µm' if use_physical else 'px'})",
                        "#B279A2",
                    ),
                ],
            ),
            (
                "roundness_solidity",
                "Single-Cell Roundness and Solidity Within Microcolonies Over Time",
                [
                    ("roundness", "Cell Roundness", "Roundness", "#59A14F"),
                    ("solidity", "Cell Solidity", "Solidity", "#ECA82C"),
                ],
            ),
            (
                "eccentricity_density",
                "Single-Cell Eccentricity and Density Within Microcolonies Over Time",
                [
                    (
                        "eccentricity",
                        "Cell Eccentricity",
                        "Eccentricity",
                        "#E45756",
                    ),
                    (
                        "local_density",
                        "Cell Local Density",
                        "Local Density",
                        "#72B7B2",
                    ),
                ],
            ),
        ]
        all_measurements = [
            measurement
            for _slug, _title, pair in measurement_pairs
            for measurement in pair
        ]
        if not any(column in cells for column, *_rest in all_measurements):
            return []

        counts = cells.groupby("time", sort=True).agg(
            cell_count=("cell_id", "size"),
            colony_count=("colony_id", "nunique"),
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        exported_paths = []
        for slug, figure_title, pair in measurement_pairs:
            if not any(column in cells for column, *_rest in pair):
                continue
            with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
                fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
                for ax, (column, title, ylabel, color) in zip(axes, pair):
                    if column not in cells:
                        ax.set_visible(False)
                        continue
                    self._plot_time_distribution(
                        ax,
                        cells,
                        column,
                        color=color,
                        max_points_per_time=2000,
                    )
                    ax.set_title(title)
                    ax.set_ylabel(ylabel)
                    self._add_count_note(
                        ax,
                        pd.DataFrame(),
                        "Colonies",
                        counts_by_time=counts["colony_count"],
                    )

                fig.suptitle(
                    f"{figure_title} — P{position}, Colony C{colony_channel}",
                    fontsize=14,
                    fontweight="bold",
                )
                fig.text(
                    0.5,
                    0.01,
                    "Cell membership uses centroid-inside-colony assignment. "
                    "The 150-pixel radius applies only to microcolony neighbor count; "
                    "n is the number of microcolonies represented in each frame.",
                    ha="center",
                    fontsize=8,
                    color="#555555",
                )
                fig.tight_layout(rect=(0, 0.03, 1, 0.96))
                output_path = output_dir / f"single_cell_{slug}_over_time.png"
                fig.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                exported_paths.append(output_path)
        return exported_paths

    def _plot_per_colony_eps(
            self,
            eps_rows: pd.DataFrame,
            output_path: Path,
            *,
            position: int,
            eps_channel: int,
            analysis_method: str,
    ) -> bool:
        import matplotlib.pyplot as plt
        import seaborn as sns

        if eps_rows.empty:
            return False
        use_physical = self._uses_physical_units()
        area_column = "eps_biomass_area_um2" if use_physical else "eps_biomass_area_px"
        if area_column not in eps_rows or "eps_fraction_of_colony" not in eps_rows:
            return False
        plot_rows = eps_rows.dropna(
            subset=[area_column, "eps_fraction_of_colony"]
        ).copy()
        if plot_rows.empty:
            return False
        plot_rows["eps_coverage_percent"] = (
            plot_rows["eps_fraction_of_colony"] * 100.0
        )

        with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
            area_summary = self._plot_time_distribution(
                axes[0], plot_rows, area_column, color="#F58518"
            )
            axes[0].set_title("EPS Area Within Each Microcolony")
            axes[0].set_ylabel(f"EPS Area ({'µm²' if use_physical else 'px²'})")
            self._add_count_note(axes[0], area_summary, "Colonies")

            coverage_summary = self._plot_time_distribution(
                axes[1], plot_rows, "eps_coverage_percent", color="#ECA82C"
            )
            axes[1].set_title("Microcolony Area Covered by EPS")
            axes[1].set_ylabel("EPS Coverage (%)")
            axes[1].set_ylim(bottom=0)
            self._add_count_note(
                axes[1], coverage_summary, "Colonies"
            )

            fig.suptitle(
                "Per-Microcolony EPS Measurements Over Time — "
                f"P{position}, EPS C{eps_channel} ({analysis_method})",
                fontsize=14,
                fontweight="bold",
            )
            fig.text(
                0.5,
                0.01,
                "n is the number of microcolonies represented in each frame.",
                ha="center",
                fontsize=8,
                color="#555555",
            )
            fig.tight_layout(rect=(0, 0.03, 1, 0.93))
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return True

    def export_time_resolved_graphs(self, output_dir: str | Path) -> list[Path]:
        """Export time-resolved microcolony summaries from centralized tables."""
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        colony_table = self.biofilm_metric_service.get_colony_results()
        if colony_table.is_empty():
            return []
        colonies = colony_table.to_pandas()

        cell_table = self.biofilm_metric_service.get_colony_cell_results()
        cells = cell_table.to_pandas() if not cell_table.is_empty() else pd.DataFrame()
        eps_table = self.biofilm_metric_service.get_colony_eps_results()
        eps_rows = eps_table.to_pandas() if not eps_table.is_empty() else pd.DataFrame()

        exported_paths = []
        group_columns = ["position", "channel"]
        for (position, colony_channel), colony_rows in colonies.groupby(
                group_columns, sort=True
        ):
            group_dir = (
                output_root
                / f"position_{int(position):03d}"
                / f"colony_channel_{int(colony_channel)}"
            )
            group_dir.mkdir(parents=True, exist_ok=True)

            spatial_path = group_dir / "microcolony_spatial_context_over_time.png"
            if self._plot_spatial_context(
                colony_rows,
                spatial_path,
                position=int(position),
                colony_channel=int(colony_channel),
            ):
                exported_paths.append(spatial_path)

            if not cells.empty:
                cell_rows = cells[
                    (cells["position"] == position)
                    & (cells["colony_channel"] == colony_channel)
                ]
                legacy_cell_plot = group_dir / "single_cell_measurements_over_time.png"
                if legacy_cell_plot.exists():
                    legacy_cell_plot.unlink()
                exported_paths.extend(
                    self._plot_single_cell_measurements(
                        cell_rows,
                        group_dir,
                        position=int(position),
                        colony_channel=int(colony_channel),
                    )
                )

            if eps_rows.empty:
                continue
            mask_based_rows = eps_rows["eps_mask_used"].fillna(False).astype(bool)
            matching_eps = eps_rows[
                (eps_rows["position"] == position)
                & (eps_rows["colony_channel"] == colony_channel)
                & mask_based_rows
            ]
            for (eps_channel, method), method_rows in matching_eps.groupby(
                    ["eps_channel", "analysis_method"], sort=True
            ):
                method_slug = self._safe_filename(method)
                biomass_path = group_dir / (
                    f"local_biomass_over_time_epsC{int(eps_channel)}_"
                    f"{method_slug}.png"
                )
                if self._plot_local_biomass(
                    colony_rows,
                    method_rows,
                    biomass_path,
                    position=int(position),
                    colony_channel=int(colony_channel),
                    eps_channel=int(eps_channel),
                    analysis_method=str(method),
                ):
                    exported_paths.append(biomass_path)

                eps_path = group_dir / (
                    f"per_colony_eps_metrics_over_time_epsC{int(eps_channel)}_"
                    f"{method_slug}.png"
                )
                if self._plot_per_colony_eps(
                    method_rows,
                    eps_path,
                    position=int(position),
                    eps_channel=int(eps_channel),
                    analysis_method=str(method),
                ):
                    exported_paths.append(eps_path)

        return exported_paths

    def print_metrics(self):
        """Debugging function: prints metrics to console"""
        #df = self.build_table()
        cache = self.segmented_storage.with_model(self.model_name)  # or your model
        mmap_array, index_set = cache.mmap_arrays_idx[cache.model_name]

        voxel = self.voxel_size
        voxel_list = [voxel.x, voxel.y, voxel.z]

        print("Voxel sizes:", voxel_list)
        print("Colonies:", len(self.colonies))
        print("Segmented frames:", len(index_set))

        # Example: print all (t, p)
        for idx in index_set:
            print("Segmented index:", idx)
        #print(df)

    def flatten_colony(self, colony: Dict, colony_cell_map: Dict, voxel_size) -> Dict:
        """Summarizes microcolony cluster contour and the cells assigned
        to it, into numerical measurements in dictionary"""
        colony_id = int(colony["colony_id"])
        return self.biofilm_metric_service.calculate_colony_metrics(
            colony,
            colony_cell_map.get(colony_id, []),
            voxel_size,
        )

    def return_cell_metrics(self):
        """Return metrics for every cell in (later) specified region"""
        return self.biofilm_metric_service.get_cell_metrics_from_cache(
            model_name=self.model_name,
            voxel_size=self.voxel_size,
            include_patches=True,
        )

    def compute_local_density(self, cells, radius_fraction=0.10):
        image_shape = (
            np.asarray(cells[0]["raw_img"]).shape[:2]
            if cells
            else None
        )
        return self.biofilm_metric_service.calculate_local_density(
            cells,
            image_shape=image_shape,
            voxel_size=self.voxel_size,
            radius_fraction=radius_fraction,
        )

    def assign_cells_to_colonies(self, cell_data, colonies):
        return self.biofilm_metric_service.assign_cells_to_colonies(
            cell_data,
            colonies,
        )

    def density_to_grid(self, cells, shape):

        grid = np.zeros(shape)

        for c in cells:
            x = int(c["centroid_x"])
            y = int(c["centroid_y"])

            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                grid[y, x] += c["local_density"]

        from scipy.ndimage import gaussian_filter
        grid = gaussian_filter(grid, sigma=5)

        return grid

    def plot_density_seaborn(self, grid, path):
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np

        if self.voxel_size is not None:
            vx = self.voxel_size.x
            vy = self.voxel_size.y
            unit = "µm"
        else:
            # fallback to pixel units
            vx = 1
            vy = 1
            unit = "px"

        h, w = grid.shape

        # Convert pixel indices → microns (if possible)
        x_ax = np.arange(0, w * vx, vx)
        y_ax = np.arange(0, h * vy, vy)

        plt.figure(figsize=(8, 6))

        # Set max intensity at 95th percentile
        vmax = np.percentile(grid[grid > 0], 95)

        ax = sns.heatmap(
            grid,
            cmap="magma",
            vmin=0,
            vmax=vmax,
            xticklabels=False,
            yticklabels=False
        )

        # --- Add ticks every 100 pixels (converted to µm) ---
        step = 100

        xticks = np.arange(0, w, step)
        yticks = np.arange(0, h, step)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.set_xticklabels((xticks * vx).astype(int))
        ax.set_yticklabels((yticks * vy).astype(int))

        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        plt.title("Local Biomass Density Field (Normalized)")

        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()



    def plot_smoothed_density(self, grid, path):
        from scipy.ndimage import gaussian_filter
        import matplotlib.pyplot as plt

        if self.voxel_size is not None:
            vx = self.voxel_size.x
            vy = self.voxel_size.y
            unit = "µm"
        else:
            # fallback to pixel units
            vx = 1
            vy = 1
            unit = "px"

        h, w = grid.shape

        extent = [
            0, w * vx,  # X axis in µm
            0, h * vy  # Y axis in µm
        ]

        smooth = gaussian_filter(grid, sigma=5)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        im0 = ax[0].imshow(grid, cmap='inferno', origin='lower', extent=extent)
        ax[0].set_title("Raw Density")
        ax[0].set_xlabel("X (µm)")
        ax[0].set_ylabel("Y (µm)")

        im1 = ax[1].imshow(smooth, cmap='inferno', origin='lower', extent=extent)
        ax[1].set_title("Smoothed Density")
        ax[1].set_xlabel("X (µm)")
        ax[1].set_ylabel("Y (µm)")

        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_density_contours(self, grid, path):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))

        plt.imshow(grid, cmap='viridis', origin='lower')
        plt.contour(grid, colors='white', linewidths=0.5)

        plt.colorbar(label="Biomass Density (µm³ / µm³)")
        plt.title("Density with Contours")

        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    def density_field_from_neighbors(self, cells, shape, radius_px=20):

        grid = np.zeros(shape)

        from collections import defaultdict
        frames = defaultdict(list)

        for c in cells:
            frames[c["t"]].append(c)

        for t, frame_cells in frames.items():

            cells_at_t = frames[t]
            coords = np.array([[c["centroid_x"], c["centroid_y"]] for c in frame_cells])
            densities = np.array([c["local_density"] for c in frame_cells])

            for i, (x, y) in enumerate(coords):

                x = int(x)
                y = int(y)

                if not (0 <= x < shape[1] and 0 <= y < shape[0]):
                    continue

                dists = np.linalg.norm(coords - [x, y], axis=1)
                neighbors = dists <= radius_px

                grid[y, x] += densities[neighbors].mean()

        from scipy.ndimage import gaussian_filter
        grid = gaussian_filter(grid, sigma=5)

        return grid

    def plot_density_field_from_neighbors(self, grid, path):
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np

        if self.voxel_size is not None:
            vx = self.voxel_size.x
            vy = self.voxel_size.y
            unit = "µm"
        else:
            # fallback to pixel units
            vx = 1
            vy = 1
            unit = "px"

        h, w = grid.shape

        # Convert pixel indices → microns
        x_um = np.arange(0, w * vx, vx)
        y_um = np.arange(0, h * vy, vy)

        plt.figure(figsize=(8, 6))

        ax = sns.heatmap(
            grid,
            cmap="magma",
            xticklabels=False,
            yticklabels=False
        )

        # --- Add ticks every 100 pixels (converted to µm) ---
        step = 100

        xticks = np.arange(0, w, step)
        yticks = np.arange(0, h, step)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.set_xticklabels((xticks * vx).astype(int))
        ax.set_yticklabels((yticks * vy).astype(int))

        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        plt.title("Biofilm Density Field")

        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    def export_grids(self, output_dir, shape):

        os.makedirs(output_dir, exist_ok=True)

        # 1. compute cells
        cell_data = self.return_cell_metrics()
        print("Returned Cell Metrics for Grid!")

        # 2. build grid
        grid = self.density_to_grid(cell_data, shape)

        # 3. build field grid
        grid_f = self.density_field_from_neighbors(cell_data, shape)

        # 3. export all versions

        self.plot_density_seaborn(grid, os.path.join(output_dir, "density_seaborn.png"))

        self.plot_smoothed_density(grid, os.path.join(output_dir, "density_smoothed.png"))

        self.plot_density_contours(grid, os.path.join(output_dir, "density_contours.png"))

        self.plot_density_field_from_neighbors(grid_f, os.path.join(output_dir, "density_field_from_neighbors.png"))

        # cell_data = self.return_cell_metrics()
        # implement cell tracking. TODO: change tracking to trackster
        #cell_data = self.track_cells_hungarian(cell_data)
        #self.plot_velocity_field(cell_data, os.path.join(output_dir, "velocity_field.png"))
        #self.plot_division_heatmap(cell_data, shape, os.path.join(output_dir, "division_heatmap.png"))
        print(f"Saved density grids to {output_dir}")



    def build_tracking_dataset(self, cells):
        from collections import defaultdict

        dataset = []

        # group by track_id
        tracks = defaultdict(list)
        for c in cells:
            tracks[c["track_id"]].append(c)

        for track_id, track_cells in tracks.items():
            track_cells = sorted(track_cells, key=lambda x: x["t"])

            for i in range(len(track_cells) - 1):
                c1 = track_cells[i]
                c2 = track_cells[i + 1]

                dataset.append({
                    "prev_img": c1["raw_img"],
                    "curr_img": c2["raw_img"],
                    "seed_mask": c1["mask"],
                    "target_mask": c2["mask"]
                })

        return dataset

    def build_tracking_input(self, prev_img, curr_img, seed_mask):
        # normalize cell images
        prev_img = prev_img.astype(np.float32) / 255.0
        curr_img = curr_img.astype(np.float32) / 255.0
        seed_mask = seed_mask.astype(np.float32)

        # resize all images
        target_size = (64, 64)
        prev_img = cv2.resize(prev_img, target_size)
        curr_img = cv2.resize(curr_img, target_size)
        seed_mask = cv2.resize(
            seed_mask,
            target_size,
            interpolation=cv2.INTER_NEAREST
        )

        # stack channels like DeLTA
        input_tensor = np.stack([prev_img, curr_img, seed_mask], axis=-1)

        # add batch dimension
        return np.expand_dims(input_tensor, axis=0)



    def plot_velocity_field(self, cells, path):
        import matplotlib.pyplot as plt
        import numpy as np
        from collections import defaultdict
        tracks = defaultdict(list)
        for c in cells:
            tracks[c["track_id"]].append(c)

        plt.figure(figsize=(6, 6))

        cmap = plt.cm.hsv  # circular colormap (perfect for direction)

        for track_id, track_cells in tracks.items():
            track_cells = sorted(track_cells, key=lambda x: x["t"])

            for i in range(len(track_cells) - 1):
                c1 = track_cells[i]
                c2 = track_cells[i + 1]

                x1, y1 = c1["centroid_x"], c1["centroid_y"]
                x2, y2 = c2["centroid_x"], c2["centroid_y"]

                dx = x2 - x1
                dy = y2 - y1

                # compute angle of arrow
                angle = np.arctan2(dy, dx)  # -pi → pi
                # normalize to 0 → 1
                norm_angle = (angle + np.pi) / (2 * np.pi)
                color = cmap(norm_angle)

                plt.arrow(
                    x1, y1,
                    dx, dy,
                    color=color,
                    head_width=2,
                    length_includes_head=True,
                    alpha=0.8
                )
        plt.gca().invert_yaxis()
        plt.title("Cell Velocity Field (Direction Colored)")
        plt.xlabel("X (px)")
        plt.ylabel("Y (px)")

        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_division_heatmap(self, cells, shape, path):
        import matplotlib.pyplot as plt
        import seaborn as sns

        grid = np.zeros(shape)

        for c in cells:
            if "parent_id" in c:  # division detected

                x = int(c["centroid_x"])
                y = int(c["centroid_y"])

                if 0 <= x < shape[1] and 0 <= y < shape[0]:
                    grid[y, x] += 1

        # smooth
        from scipy.ndimage import gaussian_filter
        grid = gaussian_filter(grid, sigma=5)

        plt.figure(figsize=(6, 6))

        sns.heatmap(
            grid,
            cmap="hot",
            xticklabels=False,
            yticklabels=False
        )

        plt.title("Cell Division Heatmap")

        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    def pad_to_size(self, img, target_size):
        """Fit Images to Consistent Size"""
        import numpy as np

        h, w = img.shape
        pad_y = max(0, target_size - h)
        pad_x = max(0, target_size - w)

        top = pad_y // 2
        bottom = pad_y - top
        left = pad_x // 2
        right = pad_x - left

        return np.pad(img,((top, bottom), (left, right)), mode='constant')
