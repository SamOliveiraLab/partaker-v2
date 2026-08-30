"""Biofilm Metric Service

Biofilm Metric Service is responsible for EPS, colony/cell,
and cube scalar metrics plus their in-session persistence.
Excludes GUI concerns: threshold dialogs, progress reporting, plots, overlays, and GIF generation
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np
import polars as pl
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops


class BiofilmMetricService:
    """Dataset-scoped metric service for the biofilm analysis workflows."""

    EPS_FILENAME = "eps.parquet"
    CELL_FILENAME = "cells.parquet"
    COLONY_FILENAME = "colonies.parquet"
    CUBE_FILENAME = "cube.parquet"

    def __init__(
            self,
            segmentation_cache=None,
            image_getter: Optional[Callable[[int, int, int], np.ndarray]] = None,
            voxel_size_getter: Optional[Callable[[], object]] = None,
    ):
        self.segmentation_cache = segmentation_cache
        self.get_image = image_getter
        self.get_voxel_size = voxel_size_getter or (lambda: None)

        self._eps_rows: dict[tuple, dict] = {}
        self._cell_df = pl.DataFrame()
        self._colony_df = pl.DataFrame()
        self._cube_df = pl.DataFrame()

    def rebind(
            self,
            *,
            segmentation_cache=None,
            image_getter=None,
            voxel_size_getter=None,
    ) -> None:
        """Update dataset dependencies without discarding stored metrics."""
        if segmentation_cache is not None:
            self.segmentation_cache = segmentation_cache
        if image_getter is not None:
            self.get_image = image_getter
        if voxel_size_getter is not None:
            self.get_voxel_size = voxel_size_getter

    # ------------------------------------------------------------------
    # EPS storage and pure calculations
    # ------------------------------------------------------------------

    def reset_eps_results(self) -> None:
        self._eps_rows.clear()

    @staticmethod
    def _eps_row_key(row: dict) -> tuple:
        return (
            int(row["time"]),
            int(row["position"]),
            int(row.get("eps_channel", row.get("channel", -1))),
            int(row.get("cell_channel", -1)),
            str(row.get("analysis_method", "")),
        )

    def upsert_eps_frame(self, row: dict) -> None:
        """Insert or replace one EPS frame result."""
        self._eps_rows[self._eps_row_key(row)] = dict(row)

    def get_eps_result_rows(self) -> list[dict]:
        return [dict(self._eps_rows[key]) for key in sorted(self._eps_rows)]

    @property
    def eps_result_count(self) -> int:
        return len(self._eps_rows)

    def get_eps_results(self) -> pl.DataFrame:
        return self._dataframe_from_rows(self.get_eps_result_rows())

    @staticmethod
    def calculate_eps_mask_metrics(
            eps_mask: np.ndarray,
            cell_mask: np.ndarray,
            cell_labels: np.ndarray | None = None,
            overlap_threshold: float = 0.10,
    ) -> dict:
        """Calculate EPS/cell overlap metrics from already-created masks."""
        eps_mask = np.asarray(eps_mask, dtype=bool)
        cell_mask = np.asarray(cell_mask, dtype=bool)

        if eps_mask.shape != cell_mask.shape:
            raise ValueError(
                "EPS and cell masks must have identical shapes: "
                f"{eps_mask.shape} != {cell_mask.shape}"
            )
        if cell_labels is not None and np.asarray(cell_labels).shape != eps_mask.shape:
            raise ValueError("Cell labels must have the same shape as the masks.")

        total_image_pixels = int(eps_mask.size)
        cell_area_pixels = int(np.count_nonzero(cell_mask))
        eps_area_pixels = int(np.count_nonzero(eps_mask))
        eps_colocalized_pixels = int(np.count_nonzero(eps_mask & cell_mask))
        eps_pixels_outside_cell_mask = int(np.count_nonzero(eps_mask & ~cell_mask))
        cell_pixels_without_eps = int(np.count_nonzero(cell_mask & ~eps_mask))

        cell_occupancy_fraction = (
            cell_area_pixels / total_image_pixels if total_image_pixels else 0.0
        )
        fraction_eps_coverage = (
            100.0 * eps_colocalized_pixels / cell_area_pixels
            if cell_area_pixels
            else 0.0
        )

        total_cells = 0
        encased_cells = 0
        if cell_labels is not None:
            labels_array = np.asarray(cell_labels)
            cell_ids = np.unique(labels_array)
            cell_ids = cell_ids[cell_ids > 0]
            total_cells = int(len(cell_ids))

            for cell_id in cell_ids:
                single_cell_mask = labels_array == cell_id
                single_cell_pixels = int(np.count_nonzero(single_cell_mask))
                if not single_cell_pixels:
                    continue
                overlap_fraction = (
                    np.count_nonzero(eps_mask & single_cell_mask)
                    / single_cell_pixels
                )
                if overlap_fraction >= overlap_threshold:
                    encased_cells += 1

            fraction_cells_encased = (
                100.0 * encased_cells / total_cells if total_cells else 0.0
            )
        else:
            # Binary-only cell masks cannot provide a per-cell denominator.
            fraction_cells_encased = fraction_eps_coverage

        return {
            "total_cells": total_cells,
            "encased_cells": encased_cells,
            "fraction_cells_encased": fraction_cells_encased,
            "fraction_eps_coverage": fraction_eps_coverage,
            "cell_occupancy_fraction": cell_occupancy_fraction,
            "cell_occupancy_percent": cell_occupancy_fraction * 100.0,
            "cell_area_pixels": cell_area_pixels,
            "eps_area_pixels": eps_area_pixels,
            "eps_colocalized_pixels": eps_colocalized_pixels,
            "eps_pixels_outside_cell_mask": eps_pixels_outside_cell_mask,
            "cell_pixels_without_eps": cell_pixels_without_eps,
        }

    @staticmethod
    def calculate_eps_raw_intensity_metrics(
            raw_eps_frame: np.ndarray,
            cell_mask: np.ndarray,
            cell_labels: np.ndarray | None = None,
    ) -> dict:
        """Measure raw EPS-channel signal relative to a cell mask."""
        raw_eps = np.asarray(raw_eps_frame, dtype=np.float32)
        cell_mask = np.asarray(cell_mask, dtype=bool)

        if raw_eps.shape != cell_mask.shape:
            raise ValueError(
                "Raw EPS frame and cell mask must have identical shapes: "
                f"{raw_eps.shape} != {cell_mask.shape}"
            )
        if cell_labels is not None and np.asarray(cell_labels).shape != raw_eps.shape:
            raise ValueError("Cell labels must have the same shape as the EPS frame.")

        finite_mask = np.isfinite(raw_eps)
        valid_cell_mask = cell_mask & finite_mask

        total_image_pixels = int(raw_eps.size)
        cell_area_pixels = int(np.count_nonzero(cell_mask))
        whole_frame_values = raw_eps[finite_mask]
        cell_values = raw_eps[valid_cell_mask]

        raw_mean_whole_frame = (
            float(np.mean(whole_frame_values, dtype=np.float64))
            if whole_frame_values.size
            else 0.0
        )
        raw_integrated_whole_frame = (
            float(np.sum(whole_frame_values, dtype=np.float64))
            if whole_frame_values.size
            else 0.0
        )
        raw_mean_within_cells = (
            float(np.mean(cell_values, dtype=np.float64))
            if cell_values.size
            else 0.0
        )
        raw_integrated_within_cells = (
            float(np.sum(cell_values, dtype=np.float64))
            if cell_values.size
            else 0.0
        )
        raw_signal_within_cells_fraction = (
            100.0
            * raw_integrated_within_cells
            / raw_integrated_whole_frame
            if raw_integrated_whole_frame > 0
            else 0.0
        )
        cell_occupancy_fraction = (
            cell_area_pixels / total_image_pixels if total_image_pixels else 0.0
        )

        total_cells = 0
        if cell_labels is not None:
            cell_ids = np.unique(np.asarray(cell_labels))
            total_cells = int(np.count_nonzero(cell_ids > 0))

        return {
            "raw_mean_intensity_whole_frame": raw_mean_whole_frame,
            "raw_integrated_intensity_whole_frame": raw_integrated_whole_frame,
            "raw_mean_intensity_within_cells": raw_mean_within_cells,
            "raw_integrated_intensity_within_cells": raw_integrated_within_cells,
            "raw_signal_within_cells_fraction": raw_signal_within_cells_fraction,
            "cell_occupancy_fraction": cell_occupancy_fraction,
            "cell_occupancy_percent": cell_occupancy_fraction * 100.0,
            "cell_area_pixels": cell_area_pixels,
            "total_cells": total_cells,
            "analysis_area_pixels": total_image_pixels,
        }

    @staticmethod
    def summarize_eps_dataframe(df: pl.DataFrame) -> pl.DataFrame:
        """Compute mean and standard deviation by time for available EPS metrics."""
        metric_specs = (
            ("cell_occupancy_fraction", "cell_occupancy"),
            ("fraction_eps_coverage", "fraction_eps_coverage"),
            ("fraction_cells_encased", "fraction_cells_encased"),
            ("raw_signal_within_cells_fraction", "raw_signal_within_cells_fraction"),
            ("raw_integrated_intensity_whole_frame", "raw_integrated_intensity_whole_frame"),
            ("raw_integrated_intensity_within_cells", "raw_integrated_intensity_within_cells"),
            ("mean_intensity", "mean_intensity"),
            ("integrated_intensity", "integrated_intensity"),
            ("eps_area_pixels", "mean_area_pixels"),
        )
        aggregations = []
        for source, mean_alias in metric_specs:
            if source not in df.columns:
                continue
            std_alias = (
                "std_area_pixels"
                if source == "eps_area_pixels"
                else f"std_{mean_alias}"
            )
            aggregations.extend([
                pl.col(source).mean().alias(mean_alias),
                pl.col(source).std().fill_null(0).alias(std_alias),
            ])

        if not aggregations:
            raise ValueError("No plottable metric columns found in dataframe.")
        return df.group_by("time").agg(aggregations).sort("time")

    def summarize_eps(self) -> pl.DataFrame:
        return self.summarize_eps_dataframe(self.get_eps_results())

    # ------------------------------------------------------------------
    # Cell and colony calculations
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_to_size(image: np.ndarray, target_size: int) -> np.ndarray:
        height, width = image.shape
        pad_y = max(0, target_size - height)
        pad_x = max(0, target_size - width)
        return np.pad(
            image,
            (
                (pad_y // 2, pad_y - pad_y // 2),
                (pad_x // 2, pad_x - pad_x // 2),
            ),
            mode="constant",
        )

    @staticmethod
    def calculate_cell_geometry(
            labeled_image: np.ndarray,
            *,
            time: int,
            position: int,
            channel: int,
            voxel_size=None,
    ) -> list[dict]:
        """Calculate scalar geometry for every labeled cell in one frame."""
        labeled_image = label(np.asarray(labeled_image))
        has_voxel = (
            voxel_size is not None
            and getattr(voxel_size, "x", None)
            and getattr(voxel_size, "y", None)
        )
        rows = []
        for region in regionprops(labeled_image):
            area_px = float(region.area)
            area_um2 = (
                area_px * float(voxel_size.x) * float(voxel_size.y)
                if has_voxel
                else area_px
            )
            rows.append({
                "t": int(time),
                "p": int(position),
                "c": int(channel),
                "cell_id": int(region.label),
                "centroid_x": float(region.centroid[1]),
                "centroid_y": float(region.centroid[0]),
                "area_px": area_px,
                "area_um2": area_um2,
            })
        return rows

    @staticmethod
    def calculate_local_density(
            cells: list[dict],
            *,
            image_shape: tuple[int, int] | None = None,
            voxel_size=None,
            radius_fraction: float = 0.10,
    ) -> list[float]:
        """Calculate neighborhood area density without retaining image arrays."""
        if not cells:
            return []
        if image_shape is None:
            sample = cells[0].get("raw_img")
            if sample is None:
                raise ValueError("image_shape is required when cells have no raw_img.")
            image_shape = np.asarray(sample).shape[:2]

        neighborhood_radius = min(image_shape) * radius_fraction
        neighborhood_area = np.pi * neighborhood_radius ** 2
        densities = [0.0] * len(cells)

        frames = defaultdict(list)
        for index, cell in enumerate(cells):
            frame_key = (
                int(cell.get("t", 0)),
                int(cell.get("p", 0)),
                int(cell.get("c", 0)),
            )
            frames[frame_key].append((index, cell))

        for frame_cells in frames.values():
            indices, rows = zip(*frame_cells)
            coordinates = np.asarray([
                [row["centroid_x"], row["centroid_y"]] for row in rows
            ])
            area_key = "area_um2" if voxel_size is not None else "area_px"
            areas = np.asarray([float(row[area_key]) for row in rows])

            for local_index, center in enumerate(coordinates):
                distances = np.linalg.norm(coordinates - center, axis=1)
                total_neighbor_area = float(areas[distances <= neighborhood_radius].sum())
                densities[indices[local_index]] = (
                    total_neighbor_area / neighborhood_area
                    if neighborhood_area > 0
                    else 0.0
                )
        return densities

    def get_cell_metrics_from_cache(
            self,
            *,
            model_name: str,
            voxel_size=None,
            include_patches: bool = True,
            patch_size: int = 64,
    ) -> list[dict]:
        """Calculate cell geometry for every cached segmentation frame."""
        if self.segmentation_cache is None or self.get_image is None:
            raise RuntimeError("BiofilmMetricService is not bound to image data.")

        voxel_size = voxel_size if voxel_size is not None else self.get_voxel_size()
        cache = self.segmentation_cache.with_model(model_name)
        _mmap_array, index_set = cache.mmap_arrays_idx[cache.model_name]
        all_cells = []

        for index in sorted(index_set):
            if len(index) == 3:
                time, position, channel = index
            else:
                time, position = index
                channel = 0

            segmented = np.asarray(cache[(time, position, channel, model_name)])
            labeled = label(segmented)
            raw_image = np.asarray(self.get_image(time, position, channel))
            frame_cells = self.calculate_cell_geometry(
                labeled,
                time=time,
                position=position,
                channel=channel,
                voxel_size=voxel_size,
            )

            if include_patches:
                half = patch_size // 2
                for cell in frame_cells:
                    center_x = int(cell["centroid_x"])
                    center_y = int(cell["centroid_y"])
                    y1 = max(0, center_y - half)
                    y2 = min(raw_image.shape[0], center_y + half)
                    x1 = max(0, center_x - half)
                    x2 = min(raw_image.shape[1], center_x + half)
                    full_mask = (labeled == cell["cell_id"]).astype(np.uint8)
                    cell["raw_img"] = self._pad_to_size(
                        raw_image[y1:y2, x1:x2], patch_size
                    )[:patch_size, :patch_size]
                    cell["mask"] = self._pad_to_size(
                        full_mask[y1:y2, x1:x2], patch_size
                    )[:patch_size, :patch_size]

            # Preserve the exporter's existing 64-pixel neighborhood convention.
            density_shape = (patch_size, patch_size) if include_patches else raw_image.shape[:2]
            densities = self.calculate_local_density(
                frame_cells,
                image_shape=density_shape,
                voxel_size=voxel_size,
            )
            for cell, density in zip(frame_cells, densities):
                cell["local_density"] = density
            all_cells.extend(frame_cells)

        scalar_rows = [
            {key: value for key, value in row.items() if not isinstance(value, np.ndarray)}
            for row in all_cells
        ]
        self._cell_df = self._dataframe_from_rows(scalar_rows)
        return all_cells

    @staticmethod
    def _colony_contour(colony: dict) -> np.ndarray:
        points = colony.get("contour")
        if points is None:
            points = colony.get("polygon_points")
        if points is None:
            polygon = colony.get("polygon")
            if polygon is not None:
                points = polygon
        if points is None:
            bbox = colony.get("bbox")
            if bbox is None:
                raise ValueError("Colony requires a contour, polygon, or bounding box.")
            x1, y1, x2, y2 = bbox
            points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        contour = np.asarray(points, dtype=np.int32).reshape(-1, 2)
        if len(contour) < 3:
            raise ValueError("Colony contour must contain at least three points.")
        return contour

    @classmethod
    def assign_cells_to_colonies(
            cls,
            cells: Iterable[dict],
            colonies: Iterable[dict],
    ) -> dict[int, list[dict]]:
        colonies = list(colonies)
        colony_cell_map = {int(colony["colony_id"]): [] for colony in colonies}
        contours = [cls._colony_contour(colony) for colony in colonies]
        for cell in cells:
            point = (float(cell["centroid_x"]), float(cell["centroid_y"]))
            for colony, contour in zip(colonies, contours):
                if cv2.pointPolygonTest(contour, point, False) >= 0:
                    colony_cell_map[int(colony["colony_id"])].append(cell)
                    break
        return colony_cell_map

    @classmethod
    def calculate_colony_metrics(
            cls,
            colony: dict,
            assigned_cells: list[dict],
            voxel_size=None,
    ) -> dict:
        contour = cls._colony_contour(colony)
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        x, y, width, height = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"]:
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
        else:
            center_x, center_y = x + width / 2, y + height / 2

        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / hull_area if hull_area else 0.0
        cell_count = len(assigned_cells)
        mean_density = (
            float(np.mean([cell.get("local_density", 0.0) for cell in assigned_cells]))
            if assigned_cells
            else 0.0
        )
        has_voxel = (
            voxel_size is not None
            and getattr(voxel_size, "x", None)
            and getattr(voxel_size, "y", None)
        )

        area_um2 = (
            area * float(voxel_size.x) * float(voxel_size.y)
            if has_voxel
            else 0.0
        )
        perimeter_um = (
            perimeter * (float(voxel_size.x) + float(voxel_size.y)) / 2
            if has_voxel
            else 0.0
        )
        return {
            "colony_id": int(colony["colony_id"]),
            "area": area,
            "perimeter": perimeter,
            "solidity": solidity,
            "centroid_x": float(center_x),
            "centroid_y": float(center_y),
            "bbox_x1": int(x),
            "bbox_y1": int(y),
            "bbox_x2": int(x + width),
            "bbox_y2": int(y + height),
            "cells_per_colony": cell_count,
            "cell_density": cell_count / area if area else 0.0,
            "has_vox": bool(has_voxel),
            "area_um2": area_um2,
            "perimeter_um": perimeter_um,
            "density_um": cell_count / area_um2 if area_um2 else 0.0,
            "centroid_x_um": center_x * float(voxel_size.x) if has_voxel else 0.0,
            "centroid_y_um": center_y * float(voxel_size.y) if has_voxel else 0.0,
            "mean_biofilm_density": mean_density,
        }

    def build_colony_table(
            self,
            colonies: Iterable[dict],
            *,
            model_name: str,
            voxel_size=None,
    ) -> pl.DataFrame:
        colonies = list(colonies)
        voxel_size = voxel_size if voxel_size is not None else self.get_voxel_size()
        cells = self.get_cell_metrics_from_cache(
            model_name=model_name,
            voxel_size=voxel_size,
            include_patches=True,
        )
        colony_cell_map = self.assign_cells_to_colonies(cells, colonies)
        rows = [
            self.calculate_colony_metrics(
                colony,
                colony_cell_map[int(colony["colony_id"])],
                voxel_size,
            )
            for colony in colonies
        ]
        self._colony_df = self._dataframe_from_rows(rows)
        return self._colony_df.clone()

    def get_cell_results(self) -> pl.DataFrame:
        return self._cell_df.clone()

    def get_colony_results(self) -> pl.DataFrame:
        return self._colony_df.clone()

    # ------------------------------------------------------------------
    # Cube calculations and storage
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_cube_frame_metrics(
            image: np.ndarray,
            colony_mask_and_boxes,
            *,
            square_size: int,
            selected_parameters: Iterable[str],
            should_stop: Optional[Callable[[], bool]] = None,
            log: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Calculate the existing cube metrics without accessing GUI state."""
        if square_size <= 0:
            raise ValueError("square_size must be positive.")
        selected = set(selected_parameters)
        should_stop = should_stop or (lambda: False)
        log = log or (lambda _message: None)

        if np.asarray(image).ndim == 3:
            gray_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        else:
            gray_image = np.asarray(image)

        if isinstance(colony_mask_and_boxes, tuple):
            colony_mask, detected_boxes = colony_mask_and_boxes
        else:
            colony_mask, detected_boxes = colony_mask_and_boxes, []
        colony_mask = np.asarray(colony_mask)
        if colony_mask.shape != gray_image.shape:
            raise ValueError("Colony mask and image must have identical shapes.")

        height, width = gray_image.shape
        results = {
            "square_positions": [],
            "local_density": [],
            "distance_to_edge": [],
            "distance_to_center": [],
            "shape_area": [],
            "intensity_mean": [],
            "local_thickness": [],
        }
        y_coords, x_coords = np.where(colony_mask > 0)
        colony_center = (
            (float(np.mean(x_coords)), float(np.mean(y_coords)))
            if len(x_coords)
            else (0.0, 0.0)
        )
        distances_to_edge = distance_transform_edt(colony_mask)
        step_size = max(square_size // 2, 5)

        regions = []
        if detected_boxes:
            for box_index, (box_x, box_y, box_width, box_height) in enumerate(detected_boxes):
                log(f"      Analyzing box {box_index + 1}/{len(detected_boxes)}")
                if box_width < square_size or box_height < square_size:
                    log(
                        f"      Skipping box {box_index + 1}: "
                        f"too small for {square_size}px squares"
                    )
                    continue
                end_y = min(
                    box_y + box_height - square_size + 1,
                    height - square_size + 1,
                )
                end_x = min(
                    box_x + box_width - square_size + 1,
                    width - square_size + 1,
                )
                if end_y <= box_y or end_x <= box_x:
                    log(
                        f"      Skipping box {box_index + 1}: "
                        "insufficient space for analysis"
                    )
                    continue
                regions.append((box_x, end_x, box_y, end_y))
        else:
            regions.append((0, width - square_size + 1, 0, height - square_size + 1))

        for start_x, end_x, start_y, end_y in regions:
            for y in range(start_y, end_y, step_size):
                for x in range(start_x, end_x, step_size):
                    if should_stop():
                        return results
                    center_x = x + square_size // 2
                    center_y = y + square_size // 2
                    if not (0 <= center_x < width and 0 <= center_y < height):
                        continue
                    if colony_mask[center_y, center_x] <= 0:
                        continue

                    image_square = gray_image[y:y + square_size, x:x + square_size]
                    mask_square = colony_mask[y:y + square_size, x:x + square_size]
                    shape_area = float(np.sum(mask_square))
                    if shape_area <= square_size * square_size * 0.1:
                        continue

                    edge_distance = (
                        float(distances_to_edge[center_y, center_x])
                        if "Distance to Edge" in selected
                        else 0.0
                    )
                    center_distance = (
                        float(np.hypot(center_x - colony_center[0], center_y - colony_center[1]))
                        if "Distance to Center" in selected
                        else 0.0
                    )
                    mask_values = image_square[mask_square > 0]
                    intensity_mean = (
                        float(np.mean(mask_values))
                        if "Fluorescence Intensity" in selected and mask_values.size
                        else 0.0
                    )
                    results["square_positions"].append((center_x, center_y))
                    results["local_density"].append(
                        shape_area / (square_size * square_size)
                        if "Local Density" in selected
                        else 0.0
                    )
                    results["distance_to_edge"].append(edge_distance)
                    results["distance_to_center"].append(center_distance)
                    results["shape_area"].append(shape_area)
                    results["intensity_mean"].append(intensity_mean)
                    results["local_thickness"].append(
                        edge_distance if "Local Texture" in selected else 0.0
                    )
        return results

    @staticmethod
    def flatten_cube_results(results: dict) -> pl.DataFrame:
        rows = []
        for colony_name, colony_data in results.items():
            for time_point, time_data in colony_data.items():
                positions = time_data.get("square_positions", [])
                for index, (x, y) in enumerate(positions):
                    rows.append({
                        "colony": str(colony_name),
                        "time": int(time_point),
                        "square_x": float(x),
                        "square_y": float(y),
                        "local_density": float(time_data["local_density"][index]),
                        "distance_to_edge": float(time_data["distance_to_edge"][index]),
                        "distance_to_center": float(time_data["distance_to_center"][index]),
                        "shape_area": float(time_data["shape_area"][index]),
                        "intensity_mean": float(time_data["intensity_mean"][index]),
                        "local_thickness": float(time_data["local_thickness"][index]),
                    })
        return BiofilmMetricService._dataframe_from_rows(rows)

    def replace_cube_results(self, results: dict) -> None:
        self._cube_df = self.flatten_cube_results(results)

    def get_cube_results(self) -> pl.DataFrame:
        return self._cube_df.clone()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        tables = {
            self.EPS_FILENAME: self.get_eps_results(),
            self.CELL_FILENAME: self._cell_df,
            self.COLONY_FILENAME: self._colony_df,
            self.CUBE_FILENAME: self._cube_df,
        }
        nonempty = {name: table for name, table in tables.items() if not table.is_empty()}
        if not nonempty:
            return
        folder.mkdir(parents=True, exist_ok=True)
        for name, table in nonempty.items():
            table.write_parquet(folder / name)

    def load(self, folder: str | Path) -> None:
        folder = Path(folder)
        eps_path = folder / self.EPS_FILENAME
        if eps_path.exists():
            self._eps_rows = {
                self._eps_row_key(row): row
                for row in pl.read_parquet(eps_path).to_dicts()
            }
        for filename, attribute in (
            (self.CELL_FILENAME, "_cell_df"),
            (self.COLONY_FILENAME, "_colony_df"),
            (self.CUBE_FILENAME, "_cube_df"),
        ):
            path = folder / filename
            if path.exists():
                setattr(self, attribute, pl.read_parquet(path))

    @staticmethod
    def _dataframe_from_rows(rows: Iterable[dict]) -> pl.DataFrame:
        rows = list(rows)
        if not rows:
            return pl.DataFrame()
        return pl.from_dicts(rows, infer_schema_length=None)
