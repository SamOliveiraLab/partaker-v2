"""
Singleton that stores the current application state.

Right now it consists of:
- Experiment details, files etc
- The segmentation state
- The collected metrics
"""

from typing import Optional, Tuple

import numpy as np
from pubsub import pub
import threading

from nd2_analyzer.data.experiment import Experiment
from nd2_analyzer.data.image_data import ImageData


class ApplicationState:
    _instance: Optional["ImageData"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.view_index: Optional[Tuple[int, int, int]] = None
        self.roi_mask: Optional[np.ndarray] = None
        self.experiment: Optional[Experiment] = None
        self.image_data: Optional[ImageData] = None

        pub.subscribe(self.on_index_changed, "view_index_changed")
        pub.subscribe(self.on_roi_selected, "roi_selected")
        pub.subscribe(self.on_experiment_loaded, "experiment_loaded")
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")

    def on_roi_selected(self, mask: np.ndarray) -> None:
        self.roi_mask = mask

    def on_index_changed(self, index: Tuple[int, int, int]) -> None:
        self.view_index = index

    def on_experiment_loaded(self, experiment: Experiment) -> None:
        self.experiment = experiment

    def on_image_data_loaded(self, image_data: ImageData) -> None:
        self.image_data = image_data

    @classmethod
    def get_instance(cls) -> Optional["ApplicationState"]:
        """Get the current singleton instance"""
        return cls._instance

    @classmethod
    def create_instance(cls, **kwargs) -> "ApplicationState":
        """Create/replace the singleton instance"""
        with cls._lock:
            # Create new instance
            instance = cls(**kwargs)
            cls._instance = instance

            return instance
