# In __init__.py in the widgets folder
from .morphology import MorphologyWidget
from .population import PopulationWidget
from .segmentation import SegmentationWidget
from .tracking_manager import TrackingManager
from .view_area import ViewAreaWidget

__all__ = [
    "ViewAreaWidget",
    "PopulationWidget",
    "SegmentationWidget",
    "MorphologyWidget",
    "TrackingManager",
]
