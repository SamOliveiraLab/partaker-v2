import logging
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import polars as pl
from pubsub import pub
from skimage.measure import regionprops, label

from nd2_analyzer.analysis.morphology.morphology import classify_morphology
from nd2_analyzer.data.frame import TLFrame
from nd2_analyzer.data.image_data import ImageData
from nd2_analyzer.utils import timing_decorator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MetricsService")


class MetricsService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Creating OptimizedMetricsService singleton instance")
            cls._instance = super(MetricsService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not getattr(self, "_initialized", False):
            logger.info("Initializing OptimizedMetricsService")

            # Use Polars DataFrame as primary storage
            self.df = pl.DataFrame()

            # Fast lookup cache for pending fluorescence updates
            self._pending_metrics = defaultdict(
                dict
            )  # {(t,p): {cell_id: metrics_dict}}

            # Segmentation cache for fluorescence processing
            self._segmentation_cache = {}  # {(t,p): labeled_image}

            # Batch processing configuration
            self._batch_size = 1000
            self._pending_count = 0

            pub.subscribe(self.compute_metrics_at_frame, "image_ready")
            self._initialized = True

    @timing_decorator("compute_metrics_at_frame")
    def compute_metrics_at_frame(
            self, image: np.ndarray, time, position, channel, mode
    ):
        if mode != "segmented":
            return

        # labeled_frame = segmentation_cache[time, position, 0] # TODO: ensure segmentationservice always returns labeled
        labeled_frame = image

        chan_n = ImageData.get_instance().channel_n
        mcherry_frame = yfp_frame = None
        if chan_n == 3:
            mcherry_frame = ImageData.get_instance().get(time, position, 1)
            yfp_frame = ImageData.get_instance().get(time, position, 2)
        elif chan_n == 2:
            mcherry_frame = ImageData.get_instance().get(time, position, 1)

        phase_frame = ImageData.get_instance().get(time, position, 0)

        curr_analysis_frame = TLFrame(
            index=(time, position),
            labeled_phc=labeled_frame,
            mcherry=mcherry_frame,
            yfp=yfp_frame,
            phase=phase_frame,
        )

        batch_metrics = MetricsService.calculate_cell_metrics(curr_analysis_frame)
        self.update_frame_metrics(batch_metrics)

    @timing_decorator("replace_frame_metrics")
    def update_frame_metrics(self, batch_data: list):
        """
        Updates metrics for a given frame (time/position).

        Deletes this time/position from the dataset if they exist, and overwrites them.

        Args:
            batch_data: Complete list of metrics from calculate_cell_metrics()
        """
        if not batch_data:
            return

        if self.df.is_empty():
            self.df = pl.DataFrame(batch_data)
            return

        # Extract the frame coordinates from the first row
        # (all rows in batch_data have the same time/position)
        first_row = batch_data[0]
        time_point = first_row["time"]
        position = first_row["position"]

        # Delete existing rows for this time/position
        self.df = self.df.filter(
            ~((pl.col("time") == time_point) & (pl.col("position") == position))
        )

        # Create DataFrame from new data and concatenate
        new_df = pl.DataFrame(batch_data)
        self.df = pl.concat([self.df, new_df], how="vertical")

    """
    Computes the metrics for each labeled cell from the segmentation
        # Uses regionprops to get geometrical features

        # For fluorescence analysis:
        # Takes each segmented cell
        # 1. calculate physical metrics
        # 2. identify which fluorescence channel it is from
        # 3. write the actual fluorescence value
    """

    def calculate_cell_metrics(_frame: TLFrame):
        cells = regionprops(_frame.labeled_phc)
        batch_data = []

        use_phase_intensity = (
            _frame.phase is not None
            and _frame.mcherry is None
            and _frame.yfp is None
        )

        # Check for the fluorescence in any channel, if not, -1 and 0 fluorescence
        # mcherry can be None, same for yfp
        back_fluo_mcherry = (
            _frame.mcherry[_frame.labeled_phc == 0].mean()
            if _frame.mcherry is not None
            else -1
        )
        back_fluo_yfp = (
            _frame.yfp[_frame.labeled_phc == 0].mean() if _frame.yfp is not None else -1
        )
        max_back_fluo = max(back_fluo_mcherry, back_fluo_yfp)
        has_fluorescence = True
        if use_phase_intensity:
            has_fluorescence = False
        elif max_back_fluo < 0.01:
            fluorescence_channel = -1
            fluorescence_level = 0.0
            has_fluorescence = False

        for cell in cells:
            cell_id = cell.label

            # Calculate derived metrics
            circularity = (
                (4 * np.pi * cell.area) / (cell.perimeter ** 2)
                if cell.perimeter > 0
                else 0
            )
            aspect_ratio = (
                cell.major_axis_length / cell.minor_axis_length
                if cell.minor_axis_length > 0
                else 1.0
            )
            y1, x1, y2, x2 = cell.bbox

            metrics_dict = {
                "area": cell.area,
                "perimeter": cell.perimeter,
                "aspect_ratio": aspect_ratio,
                "circularity": circularity,
                "solidity": cell.solidity,
                "equivalent_diameter": cell.equivalent_diameter,
                "orientation": cell.orientation,
            }

            # TODO: confirm that we need all these metrics

            # TODO: confirm this function is working
            morphology_class = classify_morphology(metrics_dict)

            # TODO: very important here, we need the computed rpu to know where each of them came from
            # Let's first compute the RPU
            # Based on the background fluorescence, tell which population this cell is from
            if use_phase_intensity:
                mask = _frame.labeled_phc == cell_id
                pix = _frame.phase[mask]
                fluorescence_channel = 0
                fluorescence_level = float(np.mean(pix)) if pix.size else 0.0
            elif has_fluorescence:
                # If only mcherry has fluo
                if back_fluo_mcherry != -1 and back_fluo_yfp == -1:
                    mcherry_fluo = _frame.mcherry[_frame.labeled_phc == cell_id].mean()
                    fluorescence_channel = 1
                    fluorescence_level = mcherry_fluo

                # If both have fluorescence, compare them
                elif back_fluo_mcherry != -1 and back_fluo_yfp != -1:
                    # Select the region corresponding to the cell in the frames
                    mcherry_fluo = _frame.mcherry[_frame.labeled_phc == cell_id].mean()
                    yfp_fluo = _frame.yfp[_frame.labeled_phc == cell_id].mean()
                    if (
                        back_fluo_mcherry != -1
                        and back_fluo_yfp != -1
                        and back_fluo_mcherry > 0
                        and back_fluo_yfp > 0
                    ):
                        if (mcherry_fluo / back_fluo_mcherry) > (yfp_fluo / back_fluo_yfp):
                            fluorescence_channel = 1
                            fluorescence_level = mcherry_fluo
                        else:
                            fluorescence_channel = 2
                            fluorescence_level = yfp_fluo
                    else:
                        fluorescence_channel = -1
                        fluorescence_level = 0.0
                else:
                    fluorescence_channel = -1
                    fluorescence_level = 0.0
            else:
                fluorescence_channel = -1
                fluorescence_level = 0.0

            row_data = {
                "time": _frame.index[0],
                "position": _frame.index[1],
                "cell_id": cell_id,
                "area": cell.area,
                "perimeter": cell.perimeter,
                "eccentricity": cell.eccentricity,
                "major_axis_length": cell.major_axis_length,
                "minor_axis_length": cell.minor_axis_length,
                "centroid_y": cell.centroid[0],
                "centroid_x": cell.centroid[1],
                "aspect_ratio": aspect_ratio,
                "circularity": circularity,
                "solidity": cell.solidity,
                "equivalent_diameter": cell.equivalent_diameter,
                "orientation": cell.orientation,
                "morphology_class": morphology_class,
                "y1": y1,
                "x1": x1,
                "y2": y2,
                "x2": x2,
                "fluorescence_channel": fluorescence_channel,
                "fluo_level": fluorescence_level,
            }

            batch_data.append(row_data)

        return batch_data

    # Checks if selected frame has metrics data
    def has_data_for(self, position: int, time: int, channel=None) -> bool:
        # Check if metrics exist for a frame
        if self.df.is_empty():
            return False
        # Return metrics results for a given frame
        result = self.df.filter(
            (pl.col("time") == time) &
            (pl.col("position") == position)
        )

        return result.height > 0

    @timing_decorator("query_optimized")
    def query_optimized(
            self,
            time: Optional[int] = None,
            position: Optional[int] = None,
            cell_id: Optional[int] = None,
            exclude_focus_loss: bool = True,
    ) -> pl.DataFrame:
        """
        Queries metrics for a given time/position/cell_id.

        Parameters:
        -----------
        time : int, optional
            Time point to filter
        position : int, optional
            Position to filter
        cell_id : int, optional
            Cell ID to filter
        exclude_focus_loss : bool, default True
            If True, excludes frames marked as focus loss in experiment settings

        Returns:
        --------
        pl.DataFrame
            Filtered metrics data
        """
        if self.df.is_empty():
            return pl.DataFrame()

        # Build filter conditions efficiently
        conditions = []
        if time is not None:
            conditions.append(pl.col("time") == time)
        if position is not None:
            conditions.append(pl.col("position") == position)
        if cell_id is not None:
            conditions.append(pl.col("cell_id") == cell_id)

        # Add focus loss filtering
        if exclude_focus_loss:
            from nd2_analyzer.data.appstate import ApplicationState
            appstate = ApplicationState.get_instance()

            if appstate and appstate.experiment and appstate.experiment.focus_loss_intervals:
                # Get time values from dataframe
                time_col = self.df.select("time").to_series()
                time_interval_hours = appstate.experiment.time_interval_hours

                # Build condition to exclude focus loss frames
                focus_loss_mask = pl.lit(False)  # Start with all False
                for start_h, end_h in appstate.experiment.focus_loss_intervals:
                    # Check if time is in this focus loss interval
                    time_hours = time_col * time_interval_hours
                    in_interval = (time_hours >= start_h) & (time_hours < end_h)
                    focus_loss_mask = focus_loss_mask | in_interval

                # Exclude frames that are in focus loss
                conditions.append(~focus_loss_mask)

        if conditions:
            # Combine conditions with logical AND
            combined_condition = conditions[0]
            for condition in conditions[1:]:
                combined_condition = combined_condition & condition
            return self.df.filter(combined_condition)

        return self.df

    @timing_decorator("batch_update_fluorescence")
    def _batch_update_fluorescence(self, updates, fluo_column):
        """Update fluorescence values using Polars joins."""
        if not updates:
            return

        update_df = pl.DataFrame(updates)

        self.df = (
            self.df.join(
                update_df,
                on=["time", "position", "cell_id"],
                how="left",
                suffix="_update",
            )
            .with_columns(
                [
                    pl.when(pl.col(f"{fluo_column}_update").is_not_null())
                    .then(pl.col(f"{fluo_column}_update"))
                    .otherwise(pl.col(fluo_column))
                    .alias(fluo_column)
                ]
            )
            .drop(f"{fluo_column}_update")
        )

    def get_cell_timeseries(self, position: int, cell_id: int) -> pl.DataFrame:
        """Get time series data for a specific cell efficiently."""
        return self.df.filter(
            (pl.col("position") == position) & (pl.col("cell_id") == cell_id)
        ).sort("time")

    def get_position_summary(self, position: int) -> pl.DataFrame:
        """Get summary statistics for a position."""
        return (
            self.df.filter(pl.col("position") == position)
            .group_by("time")
            .agg(
                [
                    pl.count("cell_id").alias("cell_count"),
                    pl.mean("area").alias("mean_area"),
                    pl.mean("fluo_level").alias("mean_fluo_level"),
                ]
            )
            .sort("time")
        )

    def save_optimized(self, folder_path: str):
        """Save data in efficient Parquet format."""
        if not self.df.is_empty():
            parquet_path = os.path.join(folder_path, "metrics_data.parquet")
            self.df.write_parquet(parquet_path)
            logger.info(f"Saved {self.df.height} rows to {parquet_path}")

    def load_optimized(self, folder_path: str):
        """Load data from Parquet format."""
        parquet_path = os.path.join(folder_path, "metrics_data.parquet")
        if os.path.exists(parquet_path):
            self.df = pl.read_parquet(parquet_path)
            logger.info(f"Loaded {self.df.height} rows from {parquet_path}")

    #
    # @timing_decorator("calculate_metrics_optimized")
    # def _calculate_metrics_optimized(self, time_lapse_frame : TLFrame):
    #     """Calculate metrics and store in optimized structure."""
    #     props = regionprops(labeled_image)
    #
    #     # Prepare batch data for efficient DataFrame operations
    #     batch_data = []
    #
    #     for cell in props:
    #         cell_id = cell.label
    #
    #         # Calculate derived metrics
    #         circularity = (4 * np.pi * cell.area) / (cell.perimeter**2) if cell.perimeter > 0 else 0
    #         aspect_ratio = cell.major_axis_length / cell.minor_axis_length if cell.minor_axis_length > 0 else 1.0
    #         y1, x1, y2, x2 = cell.bbox
    #
    #         metrics_dict = {
    #             "area": cell.area,
    #             "perimeter": cell.perimeter,
    #             "aspect_ratio": aspect_ratio,
    #             "circularity": circularity,
    #             "solidity": cell.solidity,
    #             "equivalent_diameter": cell.equivalent_diameter,
    #             "orientation": cell.orientation
    #         }
    #
    #         morphology_class = classify_morphology(metrics_dict)
    #
    #         # Create row data
    #         row_data = {
    #             "time": time,
    #             "position": position,
    #             "cell_id": cell_id,
    #             "channel": channel,
    #             "area": cell.area,
    #             "perimeter": cell.perimeter,
    #             "eccentricity": cell.eccentricity,
    #             "major_axis_length": cell.major_axis_length,
    #             "minor_axis_length": cell.minor_axis_length,
    #             "centroid_y": cell.centroid[0],
    #             "centroid_x": cell.centroid[1],
    #             "aspect_ratio": aspect_ratio,
    #             "circularity": circularity,
    #             "solidity": cell.solidity,
    #             "equivalent_diameter": cell.equivalent_diameter,
    #             "orientation": cell.orientation,
    #             "morphology_class": morphology_class,
    #             "y1": y1, "x1": x1, "y2": y2, "x2": x2,
    #             "fluo_mcherry": None,
    #             "fluo_yfp": None
    #         }
    #
    #         batch_data.append(row_data)
    #
    #         # Store in pending cache for fast fluorescence updates
    #         self._pending_metrics[(time, position)][cell_id] = row_data
    #
    #     # Batch insert into DataFrame
    #     if batch_data:
    #         new_df = pl.DataFrame(batch_data)
    #         self.df = pl.concat([self.df, new_df], how="vertical") if not self.df.is_empty() else new_df
    #         self._pending_count += len(batch_data)
    #
    # @timing_decorator("process_fluorescence")
    # def _process_fluorescence_optimized(self, image, time, position, channel):
    #     """Optimized fluorescence processing using fast lookups."""
    #     cache_key = (time, position)
    #
    #     if cache_key not in self._segmentation_cache:
    #         logger.warning(f"No segmentation for T={time}, P={position}, C={channel}")
    #         return
    #
    #     if cache_key not in self._pending_metrics:
    #         logger.warning(f"No pending metrics for T={time}, P={position}, C={channel}")
    #         return
    #
    #     labeled_image = self._segmentation_cache[cache_key]
    #     pending_cells = self._pending_metrics[cache_key]
    #
    #     if not isinstance(image, np.ndarray):
    #         image = np.array(image)
    #
    #     # Calculate background
    #     background_mask = labeled_image == 0
    #     background_intensity = np.mean(image[background_mask]) if np.any(background_mask) else 0
    #
    #     fluo_column = "fluo_mcherry" if channel == 1 else "fluo_yfp"
    #
    #     # Update fluorescence values efficiently
    #     updates = []
    #     for cell_id, metrics in pending_cells.items():
    #         cell_mask = labeled_image == cell_id
    #         if np.any(cell_mask):
    #             cell_fluorescence = np.mean(image[cell_mask])
    #             fluorescence_value = cell_fluorescence if cell_fluorescence > background_intensity else 0
    #
    #             # Store update for batch processing
    #             updates.append({
    #                 "time": time,
    #                 "position": position,
    #                 "cell_id": cell_id,
    #                 fluo_column: fluorescence_value
    #             })
    #
    #     # Batch update DataFrame using efficient Polars operations
    #     if updates:
    #         self._batch_update_fluorescence(updates, fluo_column)
