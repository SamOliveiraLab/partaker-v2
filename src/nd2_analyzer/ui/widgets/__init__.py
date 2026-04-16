# In __init__.py in the widgets folder
from .view_area import ViewAreaWidget
from .population import PopulationWidget
from .segmentation import SegmentationWidget
from .morphology import MorphologyWidget
from .tracking_manager import TrackingManager

__all__ = [
    'ViewAreaWidget',
    'PopulationWidget',
    'SegmentationWidget',
    'MorphologyWidget',
    'TrackingManager'
]