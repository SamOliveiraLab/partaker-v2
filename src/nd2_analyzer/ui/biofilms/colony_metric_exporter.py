import json
import os
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
        return self.biofilm_metric_service.build_colony_table(
            self.colonies,
            model_name=self.model_name,
            voxel_size=self.voxel_size,
        )

    def export_csv(self, path: str):
        """Exports metrics to CSV"""
        df = self.build_table()
        df.write_csv(path)

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
