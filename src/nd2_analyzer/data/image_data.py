from pathlib import Path
import json
import os

import nd2

from nd2_analyzer.analysis.segmentation.segmentation_cache import SegmentationCache
from nd2_analyzer.analysis.segmentation.segmentation_service import SegmentationService
from nd2_analyzer.analysis.segmentation.segmentation_models import SegmentationModels

from pubsub import pub

"""
Can hold either an ND2 file or a series of images
"""


class ImageData:
    def __init__(self, data, path, is_nd2=True):
        self.data = data
        self.nd2_filename = path
        self.processed_images = []
        self.is_nd2 = is_nd2

        # Initialize segmentation components
        self.segmentation_cache = SegmentationCache(data)
        self.segmentation_service = SegmentationService(
            cache=self.segmentation_cache,
            models=SegmentationModels(),
            data_getter=self._get_raw_image
        )

        pub.subscribe(self._access, "raw_image_request")
        pub.sendMessage("image_data_loaded", image_data=self)

    def _get_raw_image(self, t, p, c):
        """Helper method to retrieve raw images"""
        import numpy as np

        # Check bounds and return blank image if out of range
        if len(self.data.shape) == 5:  # - has channel
            if (t >= self.data.shape[0] or p >= self.data.shape[1] or 
                c >= self.data.shape[2] or t < 0 or p < 0 or c < 0):
                # Return blank image with same spatial dimensions
                return np.zeros((self.data.shape[3], self.data.shape[4]), dtype=self.data.dtype)
            raw_image = self.data[t, p, c]
        elif len(self.data.shape) == 4:  # - no channel
            if (t >= self.data.shape[0] or p >= self.data.shape[1] or 
                t < 0 or p < 0):
                # Return blank image with same spatial dimensions
                return np.zeros((self.data.shape[2], self.data.shape[3]), dtype=self.data.dtype)
            raw_image = self.data[t, p]
        else:
            print(f"Unusual data format: {len(self.data.shape)} dimensions")
            if len(self.data.shape) >= 3:
                if t >= self.data.shape[0] or t < 0:
                    return np.zeros((self.data.shape[1], self.data.shape[2]), dtype=self.data.dtype)
                raw_image = self.data[t, p]
            else:
                if t >= self.data.shape[0] or t < 0:
                    return np.zeros(self.data.shape[1:], dtype=self.data.dtype)
                raw_image = self.data[t]

        # Compute if it's a dask array
        if hasattr(raw_image, 'compute'):
            raw_image = raw_image.compute()

        return raw_image

    def _access(self, time, position, channel):

        image = self._get_raw_image(time, position, channel)
        pub.sendMessage("image_ready",
                        image=image,
                        time=time,
                        position=position,
                        channel=channel,
                        mode='normal')

    @classmethod
    def load_nd2(cls, file_path: str):
        with nd2.ND2File(file_path) as nd2_file:
            image_data = ImageData(data=nd2.imread(
                file_path, dask=True), path=file_path, is_nd2=True)

        return image_data
    
    @classmethod
    def load_tiff(cls, file_paths, tiff_type, tiff_types_dict=None):
        """Load TIFF files based on type
        
        Args:
            file_paths: List of TIFF file paths or single path
            tiff_type: 'single_sequence', 'multiframe', or 'series'
            tiff_types_dict: Dictionary mapping file paths to types (for mixed loading)
        """
        import tifffile
        import numpy as np
        
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        if tiff_type == "multiframe":
            return cls._load_multiframe_tiff(file_paths[0])
        elif tiff_type == "single_sequence":
            return cls._load_single_sequence_tiff(file_paths)
        elif tiff_type == "series":
            return cls._load_series_tiff(file_paths)
        else:
            raise ValueError(f"Unknown TIFF type: {tiff_type}")
    
    @classmethod
    def _load_multiframe_tiff(cls, file_path):
        """Load multi-frame TIFF file"""
        import tifffile
        import numpy as np
        
        with tifffile.TiffFile(file_path) as tif:
            # Load all pages as a stack
            data = tif.asarray()
            
            # Ensure we have at least 3 dimensions (T, Y, X)
            if data.ndim == 2:
                data = data[np.newaxis, ...]  # Add time dimension
            elif data.ndim == 3:
                # Could be (T, Y, X) or (Y, X, C) - assume (T, Y, X)
                pass
            elif data.ndim == 4:
                # Could be (T, Y, X, C) - this is what we want
                pass
            
            # Reshape to match ND2 format: (T, P, C, Y, X) or (T, P, Y, X)
            if data.ndim == 3:  # (T, Y, X)
                data = data[:, np.newaxis, ...]  # Add position dimension -> (T, P, Y, X)
            elif data.ndim == 4:  # (T, Y, X, C)
                data = data[:, np.newaxis, :, :, :]  # Add position dimension -> (T, P, Y, X, C)
                # Reorder to (T, P, C, Y, X)
                data = np.transpose(data, (0, 1, 4, 2, 3))
            
        return cls(data=data, path=file_path, is_nd2=False)
    
    @classmethod
    def _load_single_sequence_tiff(cls, file_paths):
        """Load sequence of single TIFF files"""
        import tifffile
        import numpy as np
        
        # Sort files naturally (image_001.tif, image_002.tif, etc.)
        file_paths.sort()
        
        # Load first image to get dimensions
        first_img = tifffile.imread(file_paths[0])
        
        # Create array to hold all images
        if first_img.ndim == 2:  # Grayscale
            data = np.zeros((len(file_paths), 1, first_img.shape[0], first_img.shape[1]), dtype=first_img.dtype)
            for i, fp in enumerate(file_paths):
                data[i, 0] = tifffile.imread(fp)
        elif first_img.ndim == 3:  # Color or multi-channel
            data = np.zeros((len(file_paths), 1, first_img.shape[2], first_img.shape[0], first_img.shape[1]), dtype=first_img.dtype)
            for i, fp in enumerate(file_paths):
                img = tifffile.imread(fp)
                data[i, 0] = np.transpose(img, (2, 0, 1))  # (Y, X, C) -> (C, Y, X)
        
        return cls(data=data, path=file_paths[0], is_nd2=False)
    
    @classmethod 
    def _load_series_tiff(cls, file_paths):
        """Load TIFF series with pos_XXX_t_XXX_ch_X.tif naming"""
        import tifffile
        import numpy as np
        import re
        
        # Parse filenames to extract dimensions
        pattern = r'pos_(\d+)_t_(\d+)_ch_(\d+)'
        file_info = []
        
        for fp in file_paths:
            match = re.search(pattern, fp)
            if match:
                pos, time, ch = map(int, match.groups())
                file_info.append((fp, time, pos, ch))
        
        if not file_info:
            raise ValueError("No files match the expected series naming pattern: pos_XXX_t_XXX_ch_X.tif")
        
        # Get dimensions
        max_t = max(info[1] for info in file_info) + 1
        max_p = max(info[2] for info in file_info) + 1  
        max_c = max(info[3] for info in file_info) + 1
        
        # Load first image to get spatial dimensions
        first_img = tifffile.imread(file_info[0][0])
        h, w = first_img.shape[:2]
        
        # Create data array (T, P, C, Y, X)
        data = np.zeros((max_t, max_p, max_c, h, w), dtype=first_img.dtype)
        
        # Fill the array
        for fp, t, p, c in file_info:
            img = tifffile.imread(fp)
            data[t, p, c] = img
        
        return cls(data=data, path=file_paths[0], is_nd2=False)

    def save(self, filename: str):
        """Saves state to file
        Doesn't save nd2 since it is already stored in a file
        """
        base_dir = Path(filename)
        os.makedirs(base_dir, exist_ok=True)

        # Save segmentation cache if it exists
        if self.segmentation_cache is not None:
            cache_path = base_dir / "segmentation_cache.h5"
            self.segmentation_cache.save(str(cache_path))

        # Save other container data
        container_data = {
            'nd2_filename': self.nd2_filename,
            'is_nd2': self.is_nd2
        }

        # Save container metadata
        with open(base_dir / "image_data.json", 'w') as f:
            json.dump(container_data, f)

    @classmethod
    def load(cls, filename):
        """Load imagedata from path"""
        base_dir = Path(filename)

        # Load imagedata metadata
        meta_path = base_dir / "image_data.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta_json = json.load(f)
                nd2_filename = meta_json.get('nd2_filename')
                is_nd2 = meta_json.get('is_nd2', True)

                # Use load_nd2 to create the instance properly
                if nd2_filename and os.path.exists(nd2_filename):
                    image_data = cls.load_nd2(nd2_filename)

                    # Load segmentation cache if file exists
                    cache_path = base_dir / "segmentation_cache.h5"
                    if cache_path.exists():
                        image_data.segmentation_cache = SegmentationCache.load(
                            str(cache_path), image_data.data)

                    return image_data
                else:
                    raise FileNotFoundError(
                        f"ND2 file not found: {nd2_filename}")
        else:
            raise FileNotFoundError(f"Metadata file not found in {filename}")
