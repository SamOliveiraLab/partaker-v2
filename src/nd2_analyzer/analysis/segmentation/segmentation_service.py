from pubsub import pub
import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries


class SegmentationService:
    """Service handling image segmentation and cache management"""

    def __init__(self, cache, models, data_getter):
        """
        Args:
            cache: SegmentationCache instance
            models: SegmentationModels instance
            data_getter: Function to retrieve raw images (t, p, c) -> np.ndarray
        """
        self.cache = cache
        self.models = models
        self.get_raw_image = data_getter

        pub.subscribe(self.handle_image_request, "segmented_image_request")

        # Initialize default parameters
        self.overlay_color = (0, 255, 0)  # Green for outlines
        self.label_colormap = 'viridis'

    def handle_image_request(self, time, position, channel, mode, model=None):
        """Handle image requests requiring segmentation"""
        if mode == "normal":
            return  # Let ImageData handle normal requests

        if not model:
            raise ValueError(
                "Segmentation model must be specified for non-normal modes")

        # Check cache first
        cache_key = (time, position, channel, model)

        # Don't calculate segmentation if it's already in the cache
        # Just retrieve it and return
        if model in self.cache.mmap_arrays_idx:
            _, indices = self.cache.mmap_arrays_idx[model]
            if cache_key in indices:
                segmented = self.cache.with_model(model)[cache_key]
                processed_image = self._post_process(
                    raw_image=self.get_raw_image(time, position, channel),
                    segmented=segmented,
                    mode=mode
                )

                pub.sendMessage("image_ready",
                                image=processed_image,
                                time=time,
                                position=position,
                                channel=channel,
                                mode=mode)
                return

        # If not in cache, process as normal
        segmented = self.cache.with_model(model)[cache_key]

        if segmented is None:
            # Cache miss - process and store
            raw_image = self.get_raw_image(time, position, channel)
            segmented = self._process_image(raw_image, model)
            self.cache.with_model(model)[cache_key] = segmented

        # Post-process based on mode
        processed_image = self._post_process(
            raw_image=self.get_raw_image(time, position, channel),
            segmented=segmented,
            mode=mode
        )

        pub.sendMessage("image_ready",
                        image=processed_image,
                        time=time,
                        position=position,
                        channel=channel,
                        mode=mode)

    def _process_image(self, image, model):
        """Process raw image through segmentation pipeline"""
        # Preprocessing
        preprocessed = self.models.preprocess(image, model)
        # Segmentation
        segmented = self.models.segment([preprocessed], model)[0]
        # Postprocessing
        return self.models.postprocess(segmented, model)

    def _post_process(self, raw_image, segmented, mode):
        """Apply final transformations based on display mode"""
        if mode == "segmented":
            return segmented

        if mode == "overlay":
            return self._create_overlay(raw_image, segmented)

        if mode == "labeled":
            return self._apply_colormap(segmented)

        raise ValueError(f"Unknown display mode: {mode}")

    def _create_overlay(self, raw_image, segmented):
        """Create overlay of segmentation outlines on raw image"""
        # Convert raw image to RGB if needed
        if len(raw_image.shape) == 2:
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)

        # Find segmentation boundaries
        boundaries = find_boundaries(segmented, mode='inner')

        # Apply overlay
        overlay = raw_image.copy()
        overlay[boundaries] = self.overlay_color
        return overlay

    def _apply_colormap(self, segmented):
        """Apply colormap to labeled segmentation"""
        from matplotlib.cm import get_cmap
        cmap = get_cmap(self.label_colormap)

        # Normalize labels
        labels = label(segmented)
        normalized = labels.astype(float)/labels.max()

        # Apply colormap and convert to 8-bit RGB
        colored = (cmap(normalized)[..., :3] * 255).astype(np.uint8)
        return colored

    def update_parameters(self, overlay_color=None, colormap=None):
        """Update visualization parameters"""
        if overlay_color:
            self.overlay_color = overlay_color
        if colormap:
            self.label_colormap = colormap
