import logging
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# warnings.filterwarnings('ignore')
#
# # Set up plotting style for publication-quality figures
# plt.style.use('seaborn-v0_8-whitegrid')
# sns.set_palette("husl")
#
# # Configure matplotlib for high-quality outputs
# plt.rcParams.update({
#     'figure.figsize': (12, 10),
#     'font.size': 12,
#     'axes.titlesize': 14,
#     'axes.labelsize': 12,
#     'xtick.labelsize': 10,
#     'ytick.labelsize': 10,
#     'legend.fontsize': 10,
#     'figure.dpi': 300,
#     'savefig.dpi': 300,
#     'savefig.bbox': 'tight'
# })

logger = logging.getLogger(__name__)


@dataclass
class FluoAnalysisConfig:
    """Configuration class for experiment parameters"""

    # Analysis parameters
    epsilon: float = 0.1  # Minimum fluorescence threshold

    time_interval: float = 300  # Time interval in seconds (10 minutes)
    fluorescence_factor: float = 3  # Factor for fluorescence imaging frequency

    # Selection parameters (placeholders - customize as needed)
    selected_positions: List[int] = None
    time_range: Tuple[int, int] = None

    # Channel mappings
    channel_colors: Dict[str, str] = None
    channel_names: Dict[str, str] = None
    #
    # def __post_init__(self):
    #     if self.selected_positions is None:
    #         self.selected_positions = [0, 1, 2, 3]  # Default positions
    #
    #     if self.time_range is None:
    #         self.time_range = (0, 5000)  # Default time range
    #
    #     if self.channel_colors is None:
    #         self.channel_colors = {
    #             "mcherry": "#FF4444",  # Red
    #             "yfp": "#FFB347",  # Orange/Yellow
    #             "1": "#FF4444",  # Red
    #             "2": "#FFB347"  # Orange/Yellow
    #         }
    #
    #     if self.channel_names is None:
    #         self.channel_names = {
    #             "mcherry": "mCherry",
    #             "yfp": "YFP",
    #             "1": "mCherry",
    #             "2": "YFP"
    #         }

def filter_data(df: pl.DataFrame, config: FluoAnalysisConfig) -> pl.DataFrame:
    """
    Load and preprocess fluorescence data with filtering and scaling
    """
    try:
        logger.debug("Population filter_data: before shape %s", df.shape)

        filtered_df = df.filter(
            # (pl.col("position").is_in(config.selected_positions)) &
            (pl.col("time") >= config.time_range[0]) &
            (pl.col("time") <= config.time_range[1]) &
            (pl.col("fluorescence_channel") >= 0)
        )

        # # # Remove low fluorescence values based on epsilon
        # fluorescence_cols = [col for col in df.columns if col.startswith("fluo_")]

        # for col in fluorescence_cols:
        #     filtered_df = filtered_df.filter(pl.col(col) > config.epsilon)

        # Scale time to hours
        time_scaling_factor = config.time_interval * config.fluorescence_factor / 3600
        filtered_df = filtered_df.with_columns(
            (pl.col("time") * time_scaling_factor).alias("time_hours")
        )

        logger.debug("Population filter_data: after shape %s", filtered_df.shape)
        return filtered_df

    except Exception as e:
        logger.warning("Population filter_data error: %s", e)
        return pl.DataFrame()

def create_sample_data(config: FluoAnalysisConfig) -> pl.DataFrame:
    """Create sample data for demonstration purposes"""
    np.random.seed(42)

    n_timepoints = config.time_range[1] - config.time_range[0] + 1
    n_positions = len(config.selected_positions)
    n_cells_per_position = 50

    data = []

    for t in range(config.time_range[0], config.time_range[1] + 1):
        for pos in config.selected_positions:
            for cell_id in range(n_cells_per_position):
                # Simulate fluorescence with some biological variation
                base_mcherry = 100 + 50 * np.sin(t * 0.1) + np.random.normal(0, 20)
                base_yfp = 80 + 30 * np.cos(t * 0.1) + np.random.normal(0, 15)
                dummy_mcherry = max(config.epsilon + 0.1, base_mcherry)
                dummy_yfp = max(config.epsilon + 0.1, base_yfp)

                data.append({
                    "time": t,
                    "position": pos,
                    "cell_id": f"{pos}_{cell_id}",
                    "fluo_level": dummy_mcherry,
                    "fluorescence_channel": 1
                })

                data.append({
                    "time": t,
                    "position": pos,
                    "cell_id": f"{pos}_{cell_id}",
                    "fluo_level": dummy_yfp,
                    "fluorescence_channel": 2
                })

                data.append({
                    "time": t,
                    "position": pos,
                    "cell_id": f"{pos}_{cell_id}",
                    "fluo_level": 0.5 * (dummy_mcherry + dummy_yfp),
                    "fluorescence_channel": 0
                })

    return pl.DataFrame(data)

def calculate_population_statistics(df: pl.DataFrame, config: FluoAnalysisConfig) -> pl.DataFrame:
    """
    Calculate mean and standard deviation for each timepoint
    """
    stats = (
        df.group_by(["time", "fluorescence_channel"])
        .agg([
            pl.col('fluo_level').mean().alias("mean_intensity"),
            pl.col('fluo_level').std().alias("std_intensity"),
            pl.col('fluo_level').count().alias("cell_count")
        ])
        .sort("time")
    )

    # Add scaled time
    time_scaling_factor = config.time_interval * config.fluorescence_factor / 3600
    stats = stats.with_columns(
        (pl.col("time") * time_scaling_factor).alias("time_hours")
    )

    return stats

import numpy as np
from typing import Dict, List, Tuple

# Define component activation intervals (customize these time ranges)
component_intervals = {
    'aTc': [(0, 19.25)],
    'IPTG': [(19.25, 30.42)],
    'M9': [(30.42, 100)]
}

def generate_component_step_functions(component_intervals: Dict[str, List[Tuple[float, float]]], t: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate step functions for each component's activation over time.

    Args:
        component_intervals: Dict mapping component name to a list of (start,end) tuples (in hours).
        t: np.ndarray of time points (in hours) for the full experiment duration.
    Returns:
        Dict mapping component name to an array with the same length as t, being 1 if active, 0 if not.
    """
    comp_steps = {}
    for comp, intervals in component_intervals.items():
        step = np.zeros_like(t, dtype=float)
        for (start, end) in intervals:
            step[(t >= start) & (t < end)] = 1.0
        comp_steps[comp] = step
    return comp_steps
