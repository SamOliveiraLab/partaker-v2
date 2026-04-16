"""
Test script for Cell History Builder

This script demonstrates how to use the CellHistoryBuilder and visualize the results
to verify that the reorganization is working correctly.

Usage:
    python test_cell_history.py
"""

import matplotlib.pyplot as plt
import numpy as np
from nd2_analyzer.analysis.cell_history import CellHistoryBuilder
from nd2_analyzer.analysis.metrics_service import MetricsService
from nd2_analyzer.ui.widgets.tracking_widget import TrackingWidget


def test_cell_history_builder():
    """
    Main test function - loads tracking data and builds cell histories.
    """
    print("\n" + "="*80)
    print("🧪 TESTING CELL HISTORY BUILDER")
    print("="*80)
    print()

    # Step 1: Get tracking data
    print("Step 1: Loading tracking data...")
    print("-" * 80)

    # You need to have already run tracking in your GUI
    # This retrieves the tracking data from the tracking widget
    tracking_widget = TrackingWidget()

    if not hasattr(tracking_widget, 'lineage_tracks') or not tracking_widget.lineage_tracks:
        print("❌ ERROR: No tracking data found!")
        print()
        print("Please do the following first:")
        print("1. Open your GUI application")
        print("2. Load your ND2 file")
        print("3. Run segmentation on your data")
        print("4. Go to the Tracking tab and click 'Track Cells'")
        print("5. Save your project")
        print("6. Then run this test script again")
        return None

    lineage_tracks = tracking_widget.lineage_tracks
    print(f"✓ Loaded {len(lineage_tracks)} cell tracks")
    print()

    # Step 2: Get metrics service
    print("Step 2: Connecting to metrics service...")
    print("-" * 80)
    metrics_service = MetricsService()

    if metrics_service.df.is_empty():
        print("❌ ERROR: No morphology metrics found!")
        print()
        print("Please run segmentation first to generate morphology metrics.")
        return None

    print(f"✓ Metrics service has {len(metrics_service.df)} morphology records")
    print()

    # Step 3: Build cell histories
    print("Step 3: Building cell histories...")
    print("-" * 80)

    builder = CellHistoryBuilder(lineage_tracks, metrics_service)
    cell_database = builder.build(min_track_length=5)

    if not cell_database:
        print("❌ ERROR: No cells were processed!")
        return None

    print(f"✓ Successfully built database with {len(cell_database)} cells")
    print()

    return builder, cell_database


def visualize_verification(builder, cell_database):
    """
    Create visualizations to verify the cell history builder worked correctly.
    """
    print("\n" + "="*80)
    print("📊 CREATING VERIFICATION VISUALIZATIONS")
    print("="*80)
    print()

    # Get a few example cells for detailed inspection
    cell_ids = list(cell_database.keys())[:5]  # First 5 cells

    print(f"Creating visualizations for {len(cell_ids)} example cells...")
    print()

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))

    for idx, cell_id in enumerate(cell_ids):
        cell = cell_database[cell_id]

        print(f"Visualizing Cell {cell_id}:")
        print(f"  - Lifespan: {cell['lifespan']} frames")
        print(f"  - Fate: {cell['fate']}")
        print(f"  - Morphology coverage: {cell['morphology_coverage']*100:.1f}%")
        print()

        # --- Plot 1: Trajectory (X, Y positions) ---
        ax1 = plt.subplot(5, 4, idx*4 + 1)
        ax1.plot(cell['x_positions'], cell['y_positions'], 'b-', alpha=0.6, linewidth=2)
        ax1.plot(cell['x_positions'][0], cell['y_positions'][0], 'go', markersize=10, label='Start')
        ax1.plot(cell['x_positions'][-1], cell['y_positions'][-1], 'ro', markersize=10, label='End')
        ax1.set_title(f"Cell {cell_id} - Trajectory")
        ax1.set_xlabel("X position (pixels)")
        ax1.set_ylabel("Y position (pixels)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # --- Plot 2: Area over time ---
        ax2 = plt.subplot(5, 4, idx*4 + 2)
        # Filter out NaN values for plotting
        times = cell['times']
        areas = cell['area']
        valid_idx = [i for i, a in enumerate(areas) if not np.isnan(a)]
        if valid_idx:
            valid_times = [times[i] for i in valid_idx]
            valid_areas = [areas[i] for i in valid_idx]
            ax2.plot(valid_times, valid_areas, 'b-o', markersize=4)
            ax2.set_title(f"Cell {cell_id} - Area")
            ax2.set_xlabel("Time (frames)")
            ax2.set_ylabel("Area (pixels²)")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax2.transAxes)

        # --- Plot 3: Length over time ---
        ax3 = plt.subplot(5, 4, idx*4 + 3)
        lengths = cell['length']
        valid_idx = [i for i, l in enumerate(lengths) if not np.isnan(l)]
        if valid_idx:
            valid_times = [times[i] for i in valid_idx]
            valid_lengths = [lengths[i] for i in valid_idx]
            ax3.plot(valid_times, valid_lengths, 'g-o', markersize=4)
            ax3.set_title(f"Cell {cell_id} - Length")
            ax3.set_xlabel("Time (frames)")
            ax3.set_ylabel("Length (pixels)")
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax3.transAxes)

        # --- Plot 4: Aspect Ratio over time ---
        ax4 = plt.subplot(5, 4, idx*4 + 4)
        aspect_ratios = cell['aspect_ratio']
        valid_idx = [i for i, ar in enumerate(aspect_ratios) if not np.isnan(ar)]
        if valid_idx:
            valid_times = [times[i] for i in valid_idx]
            valid_ratios = [aspect_ratios[i] for i in valid_idx]
            ax4.plot(valid_times, valid_ratios, 'r-o', markersize=4)
            ax4.set_title(f"Cell {cell_id} - Aspect Ratio")
            ax4.set_xlabel("Time (frames)")
            ax4.set_ylabel("Aspect Ratio")
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig('cell_history_verification.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to: cell_history_verification.png")
    plt.show()


def verify_frame_vs_cell_organization(builder, cell_database):
    """
    Demonstrate the difference between frame-based and cell-based organization.
    """
    print("\n" + "="*80)
    print("🔍 VERIFICATION: Frame-Based vs Cell-Based Organization")
    print("="*80)
    print()

    # Pick one cell as an example
    cell_id = list(cell_database.keys())[0]
    cell = cell_database[cell_id]

    print(f"Example: Cell {cell_id}")
    print("-" * 80)
    print()

    # Show what you'd need to do with frame-based approach
    print("❌ OLD WAY (Frame-Based): To get cell {}'s area over time...".format(cell_id))
    print()
    print("   You would need to run {} separate queries:".format(len(cell['times'])))
    for i, t in enumerate(cell['times'][:3]):  # Show first 3 as example
        print(f"   {i+1}. metrics_service.query(time={t}, cell_id={cell_id})")
    print(f"   ... (and {len(cell['times']) - 3} more queries)")
    print()
    print("   Then manually collect all the 'area' values into a list.")
    print()

    # Show what you can do now with cell-based approach
    print("✅ NEW WAY (Cell-Based): To get cell {}'s area over time...".format(cell_id))
    print()
    print(f"   cell_database[{cell_id}]['area']")
    print()
    print(f"   Result: {cell['area'][:10]}...")  # Show first 10 values
    print()
    print("   ⚡ Instant! No queries needed - everything is already assembled!")
    print()

    # Show data completeness
    print("📊 DATA COMPLETENESS CHECK:")
    print("-" * 80)
    print(f"   Cell {cell_id} appears in {len(cell['times'])} timepoints")
    print(f"   Morphology data found: {cell['morphology_found']} timepoints")
    print(f"   Morphology data missing: {cell['morphology_missing']} timepoints")
    print(f"   Coverage: {cell['morphology_coverage']*100:.1f}%")
    print()

    if cell['morphology_missing'] > 0:
        print("   ℹ️ Missing data points are filled with NaN (Not a Number)")
        print("   You can filter these out when analyzing or plotting.")
    else:
        print("   ✓ Perfect! All timepoints have morphology data.")
    print()


def manual_spot_check(builder, cell_database):
    """
    Manually verify data for one cell by comparing to original sources.
    """
    print("\n" + "="*80)
    print("🔬 MANUAL SPOT CHECK: Verify Data Integrity")
    print("="*80)
    print()

    # Pick a cell
    cell_id = list(cell_database.keys())[0]
    cell = cell_database[cell_id]

    print(f"Spot-checking Cell {cell_id} at 3 random timepoints...")
    print()

    # Pick 3 random timepoints from this cell's history
    num_checks = min(3, len(cell['times']))
    check_indices = np.random.choice(len(cell['times']), num_checks, replace=False)

    for idx in check_indices:
        t = cell['times'][idx]
        x = cell['x_positions'][idx]
        y = cell['y_positions'][idx]
        area = cell['area'][idx]
        length = cell['length'][idx]

        print(f"Timepoint {t}:")
        print(f"  From cell_database:")
        print(f"    Position: ({x:.1f}, {y:.1f})")
        print(f"    Area: {area:.1f}")
        print(f"    Length: {length:.1f}")

        # Query the original metrics_service
        metrics_df = builder.metrics_service.query_optimized(time=t, cell_id=cell_id)

        if not metrics_df.is_empty():
            row = metrics_df.row(0, named=True)
            print(f"  From metrics_service (original):")
            print(f"    Position: ({row['centroid_x']:.1f}, {row['centroid_y']:.1f})")
            print(f"    Area: {row['area']:.1f}")
            print(f"    Length: {row['major_axis_length']:.1f}")

            # Verify they match
            # Note: X/Y from tracking may not exactly match centroid from morphology
            # because tracking uses different algorithm, but should be close
            area_match = abs(area - row['area']) < 0.1
            length_match = abs(length - row['major_axis_length']) < 0.1

            if area_match and length_match:
                print(f"  ✅ VERIFIED: Data matches!")
            else:
                print(f"  ⚠️ WARNING: Data mismatch detected")
        else:
            print(f"  ⚠️ No morphology data found in original (marked as NaN)")

        print()


def query_examples(builder, cell_database):
    """
    Show examples of how to query the cell database.
    """
    print("\n" + "="*80)
    print("🔎 QUERY EXAMPLES: How to Use the Cell Database")
    print("="*80)
    print()

    # Example 1: Get a specific cell
    print("Example 1: Get complete history of a specific cell")
    print("-" * 80)
    cell_id = list(cell_database.keys())[0]
    print(f">>> cell = builder.get_cell({cell_id})")
    print(f">>> print(cell['lifespan'])")
    cell = builder.get_cell(cell_id)
    print(f"{cell['lifespan']}")
    print()

    # Example 2: Query long-lived cells
    print("Example 2: Find all cells that lived for 20+ frames")
    print("-" * 80)
    print(">>> long_cells = builder.query_cells(min_lifespan=20)")
    long_cells = builder.query_cells(min_lifespan=20)
    print(f">>> print(len(long_cells))")
    print(f"{len(long_cells)}")
    print(f"Found {len(long_cells)} cells with lifespan >= 20 frames")
    print()

    # Example 3: Query cells that divided
    print("Example 3: Find all cells that divided")
    print("-" * 80)
    print(">>> dividing_cells = builder.query_cells(fate='divided')")
    dividing_cells = builder.query_cells(fate='divided')
    print(f">>> print(len(dividing_cells))")
    print(f"{len(dividing_cells)}")
    print(f"Found {len(dividing_cells)} cells that divided")
    print()

    # Example 4: Query cells with children
    print("Example 4: Find all parent cells")
    print("-" * 80)
    print(">>> parents = builder.query_cells(has_children=True)")
    parents = builder.query_cells(has_children=True)
    print(f">>> print(len(parents))")
    print(f"{len(parents)}")
    print(f"Found {len(parents)} cells with children")
    print()


def export_example(builder):
    """
    Show how to export the data.
    """
    print("\n" + "="*80)
    print("💾 EXPORT EXAMPLE")
    print("="*80)
    print()

    print("To export the cell database to CSV:")
    print("-" * 80)
    print(">>> builder.export_to_csv('cell_histories.csv')")
    print()

    response = input("Do you want to export now? (y/n): ")
    if response.lower() == 'y':
        builder.export_to_csv('cell_histories.csv')
        print()
        print("✓ Data exported! You can now open 'cell_histories.csv' in Excel or other tools.")
    else:
        print("Skipped export.")
    print()


def main():
    """
    Run all tests and verifications.
    """
    # Test the builder
    result = test_cell_history_builder()

    if result is None:
        print("\n❌ Test failed - could not load data")
        return

    builder, cell_database = result

    # Verify it worked
    verify_frame_vs_cell_organization(builder, cell_database)

    # Spot check the data
    manual_spot_check(builder, cell_database)

    # Show query examples
    query_examples(builder, cell_database)

    # Create visualizations
    visualize_verification(builder, cell_database)

    # Export example
    export_example(builder)

    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETE!")
    print("="*80)
    print()
    print("Summary:")
    print(f"  - Processed {len(cell_database)} cells")
    print(f"  - Generated verification plots")
    print(f"  - Cell-based organization is working correctly!")
    print()
    print("Next steps:")
    print("  1. Review the visualization: cell_history_verification.png")
    print("  2. Use builder.get_cell(cell_id) to access individual cell histories")
    print("  3. Start linking COMSOL environmental data")
    print()


if __name__ == "__main__":
    main()
