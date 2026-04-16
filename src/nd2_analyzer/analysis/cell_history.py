"""
Cell History Builder

This module reorganizes frame-based microscopy data into cell-based format.
Instead of asking "what cells exist at time t?", we ask "what is the complete
history of cell X?"

Author: Generated for Cell View Analysis
Date: 2025-01-14
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CellHistory")


class CellHistoryBuilder:
    """
    Reorganizes tracking and morphology data from frame-based to cell-based format.

    Frame-based: Data organized by timepoint (t=0: [cell1, cell2, ...], t=1: [...])
    Cell-based: Data organized by cell (cell1: [t=0, t=1, ...], cell2: [...])
    """

    def __init__(self, lineage_tracks, metrics_service):
        """
        Initialize the builder.

        Parameters:
        -----------
        lineage_tracks : list
            List of track dictionaries from tracking analysis
        metrics_service : MetricsService
            Service containing morphology metrics (frame-organized)
        """
        print("\n" + "="*80)
        print("🔄 CELL HISTORY BUILDER - Initializing")
        print("="*80)

        self.lineage_tracks = lineage_tracks
        self.metrics_service = metrics_service
        self.cell_database = {}

        print(f"✓ Received {len(lineage_tracks)} cell tracks")
        print(f"✓ Metrics service connected")

        # Calculate some statistics
        total_timepoints = sum(len(track.get('t', track.get('x', []))) for track in lineage_tracks)
        print(f"✓ Total track-timepoints to process: {total_timepoints}")
        print()

    def build(self, min_track_length: int = 5) -> Dict:
        """
        Main reorganization function - converts frame-based to cell-based.

        Parameters:
        -----------
        min_track_length : int
            Minimum number of timepoints a cell must appear in to be included

        Returns:
        --------
        dict : Cell database with cell IDs as keys
        """
        print("="*80)
        print("📊 STEP 1: REORGANIZING DATA FROM FRAME-BASED TO CELL-BASED")
        print("="*80)
        print()

        processed_cells = 0
        skipped_cells = 0
        failed_cells = 0

        for track_idx, track in enumerate(self.lineage_tracks):
            cell_id = track.get('ID', -1)

            # Progress logging every 50 cells
            if track_idx > 0 and track_idx % 50 == 0:
                print(f"  Progress: {track_idx}/{len(self.lineage_tracks)} cells processed...")

            # Extract trajectory data
            x_coords = track.get('x', [])
            y_coords = track.get('y', [])
            times = track.get('t', list(range(len(x_coords))))

            # Filter by minimum track length
            if len(times) < min_track_length:
                skipped_cells += 1
                if track_idx < 5:  # Show first few skips
                    print(f"  ⊗ Cell {cell_id}: Skipped (only {len(times)} timepoints, need {min_track_length})")
                continue

            # Show details for first few cells
            if track_idx < 3:
                print(f"\n  → Processing Cell {cell_id}:")
                print(f"     Track length: {len(times)} timepoints")
                print(f"     Time range: {min(times)} to {max(times)}")
                print(f"     Position range: X=[{min(x_coords):.1f}, {max(x_coords):.1f}], Y=[{min(y_coords):.1f}, {max(y_coords):.1f}]")

            try:
                # Retrieve morphology data for this cell across all its timepoints
                cell_record = self._build_cell_record(cell_id, times, x_coords, y_coords, track)

                if cell_record:
                    self.cell_database[cell_id] = cell_record
                    processed_cells += 1
                else:
                    failed_cells += 1

            except Exception as e:
                logger.error(f"Failed to process cell {cell_id}: {e}")
                failed_cells += 1

        print()
        print("="*80)
        print("📊 REORGANIZATION SUMMARY")
        print("="*80)
        print(f"✓ Successfully processed: {processed_cells} cells")
        print(f"⊗ Skipped (too short): {skipped_cells} cells")
        print(f"✗ Failed (errors): {failed_cells} cells")
        print(f"📦 Cell database size: {len(self.cell_database)} cells")
        print()

        # Calculate statistics
        if self.cell_database:
            self._print_database_statistics()

        return self.cell_database

    def _build_cell_record(self, cell_id: int, times: List[int],
                          x_coords: List[float], y_coords: List[float],
                          track: Dict) -> Optional[Dict]:
        """
        Build a complete record for a single cell by combining tracking and morphology.

        This is where the magic happens - we query the frame-based metrics_service
        for each timepoint and assemble everything into one cell record.
        """
        # Initialize arrays for morphology time series
        areas = []
        lengths = []
        widths = []
        aspect_ratios = []
        circularities = []
        morphology_classes = []
        orientations = []
        eccentricities = []

        # Track which timepoints had morphology data
        morphology_found = 0
        morphology_missing = 0

        # For each timepoint this cell appears in, get its morphology
        for idx, t in enumerate(times):
            # Query metrics_service for this specific cell at this specific time
            # This is the frame-based → cell-based conversion happening here!
            metrics_df = self.metrics_service.query_optimized(
                time=t,
                cell_id=cell_id,
                exclude_focus_loss=True
            )

            if not metrics_df.is_empty() and len(metrics_df) > 0:
                # Found morphology data for this cell at this timepoint
                morphology_found += 1

                # Extract metrics from the query result
                row = metrics_df.row(0, named=True)

                areas.append(row.get('area', 0))
                lengths.append(row.get('major_axis_length', 0))
                widths.append(row.get('minor_axis_length', 0))
                aspect_ratios.append(row.get('aspect_ratio', 1.0))
                circularities.append(row.get('circularity', 0))
                morphology_classes.append(row.get('morphology_class', 'unknown'))
                orientations.append(row.get('orientation', 0))
                eccentricities.append(row.get('eccentricity', 0))
            else:
                # No morphology data found for this timepoint
                morphology_missing += 1

                # Fill with NaN/default values
                areas.append(np.nan)
                lengths.append(np.nan)
                widths.append(np.nan)
                aspect_ratios.append(np.nan)
                circularities.append(np.nan)
                morphology_classes.append('unknown')
                orientations.append(np.nan)
                eccentricities.append(np.nan)

        # Check if we got enough morphology data
        morphology_coverage = morphology_found / len(times) if times else 0

        if morphology_coverage < 0.5:
            # Less than 50% of timepoints have morphology data - skip this cell
            logger.warning(f"Cell {cell_id}: Poor morphology coverage ({morphology_coverage*100:.1f}%), skipping")
            return None

        # Calculate derived metrics
        lifespan = len(times)

        # Calculate displacement (start to end)
        total_displacement = np.sqrt(
            (x_coords[-1] - x_coords[0])**2 +
            (y_coords[-1] - y_coords[0])**2
        )

        # Calculate path length (sum of all steps)
        path_length = 0
        for i in range(len(x_coords) - 1):
            dx = x_coords[i+1] - x_coords[i]
            dy = y_coords[i+1] - y_coords[i]
            path_length += np.sqrt(dx**2 + dy**2)

        # Calculate average velocity
        avg_velocity = path_length / lifespan if lifespan > 0 else 0

        # Determine cell fate
        fate = self._determine_fate(track)

        # Build the complete cell record
        cell_record = {
            # Identity
            'cell_id': cell_id,
            'lifespan': lifespan,
            'fate': fate,

            # Trajectory (position over time)
            'times': times,
            'x_positions': x_coords,
            'y_positions': y_coords,

            # Morphology time series (shape over time)
            'area': areas,
            'length': lengths,  # major_axis_length
            'width': widths,    # minor_axis_length
            'aspect_ratio': aspect_ratios,
            'circularity': circularities,
            'morphology_class': morphology_classes,
            'orientation': orientations,
            'eccentricity': eccentricities,

            # Movement metrics
            'total_displacement': total_displacement,
            'path_length': path_length,
            'avg_velocity': avg_velocity,
            'directionality': total_displacement / path_length if path_length > 0 else 0,

            # Lineage information
            'parent_id': track.get('parent', None),
            'children_ids': track.get('children', []),

            # Data quality
            'morphology_coverage': morphology_coverage,
            'morphology_found': morphology_found,
            'morphology_missing': morphology_missing,

            # Initial and final positions
            'start_position': (x_coords[0], y_coords[0]),
            'end_position': (x_coords[-1], y_coords[-1]),
            'start_time': times[0],
            'end_time': times[-1],
        }

        return cell_record

    def _determine_fate(self, track: Dict) -> str:
        """
        Determine what happened to this cell.

        Returns:
        --------
        str : 'divided', 'died', 'left_fov', or 'alive_at_end'
        """
        children = track.get('children', [])

        if children and len(children) > 0:
            return 'divided'

        # For now, we can't distinguish between died and left FOV
        # This could be enhanced later with additional analysis
        return 'alive_at_end'

    def _print_database_statistics(self):
        """
        Print detailed statistics about the reorganized database.
        """
        print("="*80)
        print("📈 CELL DATABASE STATISTICS")
        print("="*80)

        # Lifespan statistics
        lifespans = [cell['lifespan'] for cell in self.cell_database.values()]
        print(f"\n🔢 LIFESPAN DISTRIBUTION:")
        print(f"   Average: {np.mean(lifespans):.1f} timepoints")
        print(f"   Median: {np.median(lifespans):.1f} timepoints")
        print(f"   Min: {np.min(lifespans)} timepoints")
        print(f"   Max: {np.max(lifespans)} timepoints")
        print(f"   Std Dev: {np.std(lifespans):.1f} timepoints")

        # Count cells by lifespan ranges
        short_cells = sum(1 for l in lifespans if l < 10)
        medium_cells = sum(1 for l in lifespans if 10 <= l < 20)
        long_cells = sum(1 for l in lifespans if l >= 20)

        print(f"\n   📊 By duration:")
        print(f"      Short (<10 frames): {short_cells} cells ({short_cells/len(lifespans)*100:.1f}%)")
        print(f"      Medium (10-19 frames): {medium_cells} cells ({medium_cells/len(lifespans)*100:.1f}%)")
        print(f"      Long (20+ frames): {long_cells} cells ({long_cells/len(lifespans)*100:.1f}%)")

        # Fate statistics
        fates = [cell['fate'] for cell in self.cell_database.values()]
        fate_counts = {}
        for fate in fates:
            fate_counts[fate] = fate_counts.get(fate, 0) + 1

        print(f"\n🎯 CELL FATES:")
        for fate, count in fate_counts.items():
            print(f"   {fate}: {count} cells ({count/len(fates)*100:.1f}%)")

        # Movement statistics
        displacements = [cell['total_displacement'] for cell in self.cell_database.values()]
        velocities = [cell['avg_velocity'] for cell in self.cell_database.values()]

        print(f"\n🚀 MOVEMENT STATISTICS:")
        print(f"   Average displacement: {np.mean(displacements):.1f} pixels")
        print(f"   Average velocity: {np.mean(velocities):.2f} pixels/frame")

        # Morphology coverage
        coverages = [cell['morphology_coverage'] for cell in self.cell_database.values()]
        print(f"\n✓ DATA QUALITY:")
        print(f"   Average morphology coverage: {np.mean(coverages)*100:.1f}%")
        print(f"   Cells with 100% coverage: {sum(1 for c in coverages if c == 1.0)}")
        print(f"   Cells with <80% coverage: {sum(1 for c in coverages if c < 0.8)}")

        print("\n" + "="*80)
        print()

    def export_to_csv(self, output_path: str):
        """
        Export the cell database to CSV format for external analysis.

        This creates a 'wide' format CSV where each row is a cell and columns
        contain arrays (converted to strings) of time series data.
        """
        print(f"\n💾 Exporting cell database to CSV: {output_path}")

        # Prepare data for CSV (flatten arrays to strings)
        export_data = []

        for cell_id, cell_data in self.cell_database.items():
            row = {
                'cell_id': cell_id,
                'lifespan': cell_data['lifespan'],
                'fate': cell_data['fate'],
                'start_time': cell_data['start_time'],
                'end_time': cell_data['end_time'],
                'start_x': cell_data['start_position'][0],
                'start_y': cell_data['start_position'][1],
                'end_x': cell_data['end_position'][0],
                'end_y': cell_data['end_position'][1],
                'total_displacement': cell_data['total_displacement'],
                'path_length': cell_data['path_length'],
                'avg_velocity': cell_data['avg_velocity'],
                'directionality': cell_data['directionality'],
                'parent_id': cell_data['parent_id'],
                'num_children': len(cell_data['children_ids']),
                'morphology_coverage': cell_data['morphology_coverage'],

                # Time series as comma-separated strings
                'times': ','.join(map(str, cell_data['times'])),
                'x_positions': ','.join(map(str, cell_data['x_positions'])),
                'y_positions': ','.join(map(str, cell_data['y_positions'])),
                'areas': ','.join(map(str, cell_data['area'])),
                'lengths': ','.join(map(str, cell_data['length'])),
                'widths': ','.join(map(str, cell_data['width'])),
            }
            export_data.append(row)

        # Create DataFrame and export
        df = pl.DataFrame(export_data)
        df.write_csv(output_path)

        print(f"✓ Exported {len(export_data)} cells to {output_path}")
        print()

    def get_cell(self, cell_id: int) -> Optional[Dict]:
        """
        Retrieve complete data for a single cell.

        Parameters:
        -----------
        cell_id : int
            The ID of the cell to retrieve

        Returns:
        --------
        dict or None : Complete cell record, or None if not found
        """
        return self.cell_database.get(cell_id, None)

    def query_cells(self, min_lifespan: int = None, max_lifespan: int = None,
                   fate: str = None, has_children: bool = None) -> List[Dict]:
        """
        Query cells based on criteria.

        Parameters:
        -----------
        min_lifespan : int, optional
            Minimum lifespan in frames
        max_lifespan : int, optional
            Maximum lifespan in frames
        fate : str, optional
            Cell fate ('divided', 'died', etc.)
        has_children : bool, optional
            Whether cell has children

        Returns:
        --------
        list : List of cell records matching criteria
        """
        results = []

        for cell in self.cell_database.values():
            # Apply filters
            if min_lifespan and cell['lifespan'] < min_lifespan:
                continue
            if max_lifespan and cell['lifespan'] > max_lifespan:
                continue
            if fate and cell['fate'] != fate:
                continue
            if has_children is not None:
                if has_children and len(cell['children_ids']) == 0:
                    continue
                if not has_children and len(cell['children_ids']) > 0:
                    continue

            results.append(cell)

        return results


# Convenience function for quick usage
def build_cell_histories(lineage_tracks, metrics_service,
                        min_track_length: int = 5) -> Dict:
    """
    Quick function to reorganize data to cell-based format.

    Parameters:
    -----------
    lineage_tracks : list
        List of tracking data
    metrics_service : MetricsService
        Morphology metrics service
    min_track_length : int
        Minimum track length to include

    Returns:
    --------
    dict : Cell database
    """
    builder = CellHistoryBuilder(lineage_tracks, metrics_service)
    cell_database = builder.build(min_track_length=min_track_length)
    return cell_database
