import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Signal

from nd2_analyzer.analysis.morphology.morphology import extract_cells_and_metrics


class MorphologyWorker(QObject):
    progress = Signal(int)  # Progress updates
    finished = Signal(object)  # Finished with results
    error = Signal(str)  # Emit error message

    def __init__(self, image_data, image_frames, num_frames, position, channel):
        super().__init__()
        self.image_data = image_data
        self.image_frames = image_frames
        self.num_frames = num_frames
        self.position = position
        self.channel = channel
        self.cell_mapping = {}

    def run(self):
        results = {}
        try:
            for t in range(self.num_frames):
                current_frame = self.image_frames[t]
                # Skip empty/invalid frames
                if np.mean(current_frame) == 0 or np.std(current_frame) == 0:
                    print(f"Skipping empty frame T={t}")
                    self.progress.emit(t + 1)
                    continue

                t, p, c = (t, self.position, self.channel)

                binary_image = self.image_data.segmentation_cache[t, p, c]

                # Validate segmentation result
                if binary_image is None or binary_image.sum() == 0:
                    print(f"Frame {t}: No valid segmentation")
                    self.progress.emit(t + 1)
                    continue

                # Extract morphology metrics
                cell_mapping = extract_cells_and_metrics(
                    self.image_frames[t], binary_image
                )

                # Then convert the cell_mapping to a metrics dataframe
                metrics_list = [data["metrics"] for data in cell_mapping.values()]
                metrics = pd.DataFrame(metrics_list)

                if not metrics.empty:
                    total_cells = len(metrics)

                    # Calculate Morphology Fractions
                    morphology_counts = metrics["morphology_class"].value_counts(
                        normalize=True
                    )
                    fractions = morphology_counts.to_dict()

                    # Save results for this frame, including the raw metrics
                    results[t] = {
                        "fractions": fractions,
                        "metrics": metrics,  # Include the full metrics dataframe
                    }
                else:
                    print(f"Frame {t}: Metrics computation returned no valid data.")

                self.progress.emit(t + 1)  # Update progress bar

            if results:
                self.finished.emit(results)  # Emit processed results
            else:
                self.error.emit("No valid results found in any frame.")
        except Exception as e:
            self.error.emit(str(e))
            raise e
