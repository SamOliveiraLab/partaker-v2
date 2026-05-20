import json
import os
from typing import List, Dict, Tuple, Optional

from nd2_analyzer.data.experiment import Experiment
from nd2_analyzer.data.image_data import ImageData
from nd2_analyzer.analysis.metrics_service import MetricsService
from nd2_analyzer.ui.biofilms.colony_separator import ColonySeparator

from nd2_analyzer.ui.widgets import SegmentationWidget
import numpy as np
import pandas as pd
import cv2

import polars as pl
import json

class ColonyMetricExporter:
    def __init__(self, image_data, colonies, segmentation_storage, model_name, voxel_size):
        self.image_data = image_data
        self.colonies = colonies
        self.segmented_storage = segmentation_storage
        self.model_name = model_name
        self.voxel_size = voxel_size

        """import torch
        import torch.nn as nn

        class SimpleUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, 3, padding=1),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Conv2d(32, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 1, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.decoder(self.encoder(x))"""

        #self.tracking_model = SimpleUNet()
        #self.tracking_model.load_state_dict(torch.load("tracking_unet.pth"))
        #self.tracking_model.eval()

    def build_table(self):

        cell_data = self.return_cell_metrics()
        # cell_data = self.track_cells_hungarian(cell_data)
        colony_cell_map = self.assign_cells_to_colonies(cell_data, self.colonies)
        rows = []
        for colony in self.colonies:
            rows.append(
                self.flatten_colony(colony, colony_cell_map, self.voxel_size)
            )
            print(f"Processed {len(rows)} colonies in table")
        return pl.DataFrame(rows)

    def export_json(self, path: str):
        data = [
            self.flatten_colony(c, self.voxel_size)
            for c in self.colonies
        ]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def export_csv(self, path: str):
        df = self.build_table()
        df.write_csv(path)

    def print_metrics(self):
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

        contour = np.array(colony["contour"], dtype=np.int32)

        area_px = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))

        x, y, w, h = cv2.boundingRect(contour)

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = x + w / 2, y + h / 2

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area_px / hull_area if hull_area > 0 else 0

        # ---------------------------
        # CELL STATS (KEY PART)
        # ---------------------------

        cid = colony["colony_id"]
        cells = colony_cell_map.get(cid, [])

        if cells:
            mean_density = np.mean([c["local_density"] for c in cells])
            total_biomass = np.sum([c["volume_um3"] for c in cells])
        else:
            mean_density = 0
            total_biomass = 0

        cell_count = len(cells)

        if voxel_size is not None:
            area_um2 = area_px * voxel_size.x * voxel_size.y
        else:
            area_um2 = area_px
        density = cell_count / area_um2 if area_um2 > 0 else 0

        # ---------------------------
        # OUTPUT
        # ---------------------------

        flat = {
            "colony_id": cid,

            "area_px": area_px,
            "perimeter_px": perimeter,
            "solidity": solidity,

            "centroid_x_px": cx,
            "centroid_y_px": cy,

            "bbox_x1": x,
            "bbox_y1": y,
            "bbox_x2": x + w,
            "bbox_y2": y + h,

            "cells_per_colony": cell_count,
            "cell_density": density,
        }

        if voxel_size:
            flat.update({
                "area_um2": area_um2,
                "centroid_x_um": cx * voxel_size.x,
                "centroid_y_um": cy * voxel_size.y,
            })
        flat.update({
            "biomass_um3": total_biomass,
            "mean_biofilm_density": mean_density
        })

        if cells:
            times = [c["t"] for c in cells]
            t_span = max(times) - min(times) if len(times) > 1 else 1
            growth_rate = total_biomass / t_span
        else:
            growth_rate = 0

        flat["growth_rate"] = growth_rate

        return flat

    def return_cell_metrics(self):

        cache = self.segmented_storage.with_model(self.model_name)
        mmap_array, index_set = cache.mmap_arrays_idx[cache.model_name]

        from skimage.measure import label, regionprops

        cells = []

        print("Index set:", len(index_set))

        # process frame-by-frame (not all at once mentally)
        for idx in index_set:
            print("Processing idx:", idx)

            if len(idx) == 3:
                t, p, c = idx
            else:
                t, p = idx
                c = self.image_data.channel_n

            segmented = cache[(t, p, c, self.model_name)]
            raw_img = self.image_data.get(t, p, c)

            labeled = label(segmented)
            regions = regionprops(labeled)

            # TEMP frame list (small)
            frame_cells = []

            for region in regions:
                cx = region.centroid[1]
                cy = region.centroid[0]

                area_px = region.area
                if self.voxel_size is not None and self.voxel_size.x:
                    area_um2 = area_px * self.voxel_size.x * self.voxel_size.y

                    volume_um3 = (
                        area_um2 * self.voxel_size.z
                        if self.voxel_size.z else area_um2
                    )
                else:
                    # fallback to pixel units
                    area_um2 = area_px
                    volume_um3 = area_px

                # crop mask (fixes memory issue)
                #y1, x1, y2, x2 = region.bbox
                #cropped_mask = (labeled[y1:y2, x1:x2] == region.label).astype(np.uint8)
                #cropped_img = raw_img[y1:y2, x1:x2]"""


                # FIXED PATCH EXTRACTION FOR MASK SIZE

                PATCH_SIZE = 64
                half = PATCH_SIZE // 2
                cx_int = int(cx)
                cy_int = int(cy)

                # centered crop coordinates
                y1 = max(0, cy_int - half)
                y2 = min(raw_img.shape[0], cy_int + half)

                x1 = max(0, cx_int - half)
                x2 = min(raw_img.shape[1], cx_int + half)

                # full-cell mask first
                cell_mask_full = (labeled == region.label).astype(np.uint8)

                # crop local patch
                cropped_img = raw_img[y1:y2, x1:x2]
                cropped_mask = cell_mask_full[y1:y2, x1:x2]


                # PAD TO 64x64
                cropped_img = self.pad_to_size(cropped_img, PATCH_SIZE)
                cropped_mask = self.pad_to_size(cropped_mask, PATCH_SIZE)

                cropped_img = cropped_img[:PATCH_SIZE, :PATCH_SIZE]
                cropped_mask = cropped_mask[:PATCH_SIZE, :PATCH_SIZE]

                frame_cells.append({
                    "t": t,
                    "p": p,
                    "c": c,
                    "cell_id": region.label,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "area_px": area_px,
                    "area_um2": area_um2,
                    "volume_um3": volume_um3,
                    "mask": cropped_mask,
                    "raw_img": cropped_img,
                })

            # compute density per frame (NOT global)
            densities = self.compute_local_density(frame_cells)

            for i in range(len(frame_cells)):
                frame_cells[i]["local_density"] = densities[i]

            # append AFTER processing frame
            cells.extend(frame_cells)

        print("Successfully returned Cells!")
        return cells

    def compute_local_density(self, cells, radius_um=10):

        densities = [0] * len(cells)

        # Apply voxel size
        if self.voxel_size is not None and self.voxel_size.x:
            r_px = radius_um / self.voxel_size.x
        else:
            # fallback: interpret radius_um as pixels
            r_px = radius_um

        # group by time
        from collections import defaultdict
        frames = defaultdict(list)

        for i, c in enumerate(cells):
            frames[c["t"]].append((i, c))

        for t, frame_cells in frames.items():

            indices, frame_cells = zip(*frame_cells)

            cells_at_t = frames[t]

            coords = np.array([[c["centroid_x"], c["centroid_y"]] for c in frame_cells])
            volumes = np.array([c["volume_um3"] for c in frame_cells])

            for i, center in enumerate(coords):

                dists = np.linalg.norm(coords - center, axis=1)
                neighbors = dists <= r_px

                total_biomass = volumes[neighbors].sum()

                if self.voxel_size is not None and self.voxel_size.z:
                        sphere_vol = (4 / 3) * np.pi * (radius_um ** 3)
                else:
                    sphere_vol = np.pi * (radius_um ** 2)

                density = total_biomass / sphere_vol if sphere_vol > 0 else 0

                densities[indices[i]] = density

        return densities

    def assign_cells_to_colonies(self, cell_data, colonies):

        colony_cell_map = {c["colony_id"]: [] for c in colonies}

        for cell in cell_data:

            x = cell["centroid_x"]
            y = cell["centroid_y"]

            for colony in colonies:

                contour = np.array(colony["contour"], dtype=np.int32)

                # inside = +1, outside = -1, boundary = 0
                inside = cv2.pointPolygonTest(contour, (x, y), False)

                if inside >= 0:
                    colony_cell_map[colony["colony_id"]].append(cell)
                    break  # assume cell belongs to only one colony

        return colony_cell_map

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
        x_um = np.arange(0, w * vx, vx)
        y_um = np.arange(0, h * vy, vy)

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
        print("Started Hungarian Tracking for Grid!")
        # implement cell tracking. TODO: change tracking to UNET
        cell_data = self.track_cells_hungarian(cell_data)
        self.plot_velocity_field(cell_data, os.path.join(output_dir, "velocity_field.png"))
        self.plot_division_heatmap(cell_data, shape, os.path.join(output_dir, "division_heatmap.png"))

        print(f"Saved density grids to {output_dir}")

    def track_cells_hungarian(self, cells):
        from collections import defaultdict
        from scipy.optimize import linear_sum_assignment
        import numpy as np

        tracks = []
        next_id = 0

        # group cells by time
        frames = defaultdict(list)
        for c in cells:
            frames[c["t"]].append(c)

        prev_cells = []

        for t in sorted(frames.keys()):
            curr_cells = frames[t]

            # initialize track_id
            for c in curr_cells:
                c["track_id"] = None

            # FIRST FRAME → assign new IDs
            if not prev_cells:
                for c in curr_cells:
                    c["track_id"] = next_id
                    next_id += 1

            else:
                # BUILD COST MATRIX
                cost_matrix = np.zeros((len(prev_cells), len(curr_cells)))

                for i, p in enumerate(prev_cells):
                    for j, c in enumerate(curr_cells):
                        # Euclidean distance
                        dist = np.linalg.norm([
                            c["centroid_x"] - p["centroid_x"],
                            c["centroid_y"] - p["centroid_y"]
                        ])

                        if dist > 50:
                            cost_matrix[i, j] = 1e6
                            continue  # SKIP IoU

                        # IoU (shape similarity)
                        iou = self.compute_iou(p["mask"], c["mask"])

                        # FINAL COST (tunable)
                        cost = dist - (iou * 20)

                        cost_matrix[i, j] = cost

                # HUNGARIAN MATCHING
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                MAX_DIST = 30
                assigned_curr = set()

                # ASSIGN MATCHES
                for i, j in zip(row_ind, col_ind):

                    if cost_matrix[i, j] < MAX_DIST:
                        curr_cells[j]["track_id"] = prev_cells[i]["track_id"]
                        assigned_curr.add(j)

                # NEW CELLS (not matched)
                for j, c in enumerate(curr_cells):
                    if j not in assigned_curr:
                        c["track_id"] = next_id
                        next_id += 1

                # DIVISION DETECTION
                for i, p in enumerate(prev_cells):

                    matches = []

                    for j, c in enumerate(curr_cells):
                        dist = np.linalg.norm([
                            c["centroid_x"] - p["centroid_x"],
                            c["centroid_y"] - p["centroid_y"]
                        ])

                        if dist < MAX_DIST:
                            matches.append(c)

                    # if one parent → multiple children
                    if len(matches) >= 2:
                        for m in matches:
                            m["parent_id"] = p["track_id"]

            prev_cells = curr_cells
            tracks.extend(curr_cells)

        return tracks

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

    def track_cells_unet(self, cells):
        from collections import defaultdict

        tracks = []
        next_id = 0

        # group cells by time
        frames = defaultdict(list)
        for c in cells:
            frames[c["t"]].append(c)

        prev_cells = []

        for t in sorted(frames.keys()):
            curr_cells = frames[t]

            # assign empty ids
            for c in curr_cells:
                c["track_id"] = None

            # first frame
            if not prev_cells:
                for c in curr_cells:
                    c["track_id"] = next_id
                    next_id += 1

            else:
                for p in prev_cells:
                    input_tensor = self.build_tracking_input(
                        p["raw_img"],  # ← correct per-cell input
                        curr_cells[0]["raw_img"],  # OK (same frame)
                        p["mask"]
                    )

                assigned_curr = set()

                assigned_curr = set()

                for p in prev_cells:

                    prev_img = p["raw_img"]
                    curr_img = curr_cells[0]["raw_img"]

                    input_tensor = self.build_tracking_input(
                        prev_img,
                        curr_img,
                        p["mask"]
                    )

                    import torch
                    with torch.no_grad():
                        x = torch.tensor(input_tensor, dtype=torch.float32)
                        x = x.permute(0, 3, 1, 2)

                        pred = self.tracking_model(x)
                        pred = pred.squeeze().numpy()

                    prediction = (pred > 0.5).astype(np.uint8)

                    overlaps = []
                    for j, c in enumerate(curr_cells):
                        iou = self.compute_iou(prediction, c["mask"])
                        if iou > 0.3:
                            overlaps.append((j, iou))

                    if len(overlaps) == 1:
                        j, _ = overlaps[0]
                        curr_cells[j]["track_id"] = p["track_id"]
                        assigned_curr.add(j)

                    elif len(overlaps) >= 2:
                        for j, _ in overlaps:
                            curr_cells[j]["track_id"] = next_id
                            curr_cells[j]["parent_id"] = p["track_id"]
                            assigned_curr.add(j)
                            next_id += 1

                    # CASE 3: DISAPPEAR → do nothing

                # NEW CELLS (not matched)
                for j, c in enumerate(curr_cells):
                    if j not in assigned_curr:
                        c["track_id"] = next_id
                        next_id += 1

            prev_cells = curr_cells
            tracks.extend(curr_cells)

        return tracks

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

    def build_table_hungarian(self):

        cell_data = self.return_cell_metrics()
        print("Building Hungarian table...")

        # STEP 1: generate ground truth
        cell_data = self.track_cells_hungarian(cell_data)

        # STEP 2: build dataset
        dataset = self.build_tracking_dataset(cell_data)

        print("Training samples:", len(dataset))

        return dataset  # temporarily return dataset for training

    def compute_iou(self, mask1, mask2):
        import cv2

        # resize mask2 to match mask1
        mask2_resized = cv2.resize(
            mask2.astype(np.uint8),
            (mask1.shape[1], mask1.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        intersection = np.logical_and(mask1, mask2_resized).sum()
        union = np.logical_or(mask1, mask2_resized).sum()

        return intersection / union if union > 0 else 0



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
