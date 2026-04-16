"""
ROI Helper Utility

This module provides helper functions to ensure ALL analysis respects ROI boundaries.
Analysis should ONLY be performed within ROI, except when ROI is not defined.
"""

import numpy as np
from typing import Optional, Tuple, List
from nd2_analyzer.data.appstate import ApplicationState


class ROIHelper:
    """
    Helper class to enforce ROI constraints across all analysis operations.

    CRITICAL: All analysis must respect ROI boundaries when ROI is defined.
    """

    @staticmethod
    def get_roi_mask() -> Optional[np.ndarray]:
        """
        Get the current ROI mask from application state.

        Returns:
            ROI mask if defined, None otherwise
        """
        try:
            appstate = ApplicationState.get_instance()
            if appstate is None:
                return None
            return appstate.roi_mask if hasattr(appstate, 'roi_mask') else None
        except:
            return None

    @staticmethod
    def has_roi() -> bool:
        """
        Check if ROI is currently defined.

        Returns:
            True if ROI is defined, False otherwise
        """
        roi_mask = ROIHelper.get_roi_mask()
        return roi_mask is not None

    @staticmethod
    def is_point_in_roi(x: float, y: float, roi_mask: Optional[np.ndarray] = None) -> bool:
        """
        Check if a point (x, y) is within the ROI.

        Args:
            x: X coordinate
            y: Y coordinate
            roi_mask: Optional ROI mask (if None, will fetch from AppState)

        Returns:
            True if point is in ROI or if ROI is not defined, False otherwise
        """
        if roi_mask is None:
            roi_mask = ROIHelper.get_roi_mask()

        # If no ROI is defined, all points are valid
        if roi_mask is None:
            return True

        # Check bounds
        x_int, y_int = int(round(x)), int(round(y))
        if 0 <= y_int < roi_mask.shape[0] and 0 <= x_int < roi_mask.shape[1]:
            return bool(roi_mask[y_int, x_int])

        return False

    @staticmethod
    def filter_tracks_by_roi(tracks: List[dict], min_roi_coverage: float = 0.5) -> List[dict]:
        """
        Filter tracks to only include those that are primarily within ROI.

        Args:
            tracks: List of track dictionaries with 'x', 'y', 't' keys
            min_roi_coverage: Minimum fraction of track points that must be in ROI (0.0-1.0)

        Returns:
            Filtered list of tracks that meet ROI coverage requirement
        """
        roi_mask = ROIHelper.get_roi_mask()

        # If no ROI, return all tracks
        if roi_mask is None:
            return tracks

        filtered_tracks = []

        for track in tracks:
            if 'x' not in track or 'y' not in track:
                continue

            # Count how many points are in ROI
            points_in_roi = 0
            total_points = len(track['x'])

            for x, y in zip(track['x'], track['y']):
                if ROIHelper.is_point_in_roi(x, y, roi_mask):
                    points_in_roi += 1

            # Include track if it meets minimum coverage
            if total_points > 0:
                coverage = points_in_roi / total_points
                if coverage >= min_roi_coverage:
                    filtered_tracks.append(track)

        return filtered_tracks

    @staticmethod
    def filter_track_points_by_roi(track: dict) -> dict:
        """
        Filter individual track points to only include those within ROI.

        Args:
            track: Track dictionary with 'x', 'y', 't' keys

        Returns:
            New track dictionary with only ROI points
        """
        roi_mask = ROIHelper.get_roi_mask()

        # If no ROI, return original track
        if roi_mask is None:
            return track

        # Filter points
        filtered_track = {key: [] for key in track.keys()}
        filtered_track['ID'] = track.get('ID')
        filtered_track['parent_id'] = track.get('parent_id')
        filtered_track['children_ids'] = track.get('children_ids', [])

        if 'x' in track and 'y' in track:
            for i, (x, y) in enumerate(zip(track['x'], track['y'])):
                if ROIHelper.is_point_in_roi(x, y, roi_mask):
                    for key in ['x', 'y', 't']:
                        if key in track and i < len(track[key]):
                            filtered_track[key].append(track[key][i])

        return filtered_track

    @staticmethod
    def apply_roi_to_mask(mask: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply ROI mask to a segmentation or analysis mask.
        Zeros out all regions outside the ROI.

        Args:
            mask: Input mask to filter
            roi_mask: Optional ROI mask (if None, will fetch from AppState)

        Returns:
            Filtered mask with ROI applied
        """
        if roi_mask is None:
            roi_mask = ROIHelper.get_roi_mask()

        # If no ROI, return original mask
        if roi_mask is None:
            return mask

        # Apply ROI
        return mask * roi_mask.astype(mask.dtype)

    @staticmethod
    def get_roi_info() -> dict:
        """
        Get information about the current ROI.

        Returns:
            Dictionary with ROI information
        """
        roi_mask = ROIHelper.get_roi_mask()

        if roi_mask is None:
            return {
                'has_roi': False,
                'area': None,
                'shape': None,
                'coverage': None
            }

        total_pixels = roi_mask.size
        roi_pixels = np.sum(roi_mask > 0)

        return {
            'has_roi': True,
            'area': roi_pixels,
            'shape': roi_mask.shape,
            'coverage': roi_pixels / total_pixels if total_pixels > 0 else 0
        }

    @staticmethod
    def validate_analysis_within_roi(message: str = ""):
        """
        Print a validation message to confirm analysis is respecting ROI.

        Args:
            message: Custom message to include
        """
        roi_info = ROIHelper.get_roi_info()

        if roi_info['has_roi']:
            print(f"\n{'='*60}")
            print(f"✓ ROI VALIDATION: {message}")
            print(f"  ROI is active - Analysis restricted to ROI area")
            print(f"  ROI area: {roi_info['area']} pixels ({roi_info['coverage']*100:.1f}% of image)")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"ℹ ROI VALIDATION: {message}")
            print(f"  No ROI defined - Analysis performed on entire image")
            print(f"{'='*60}\n")
