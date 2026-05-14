import json
import os
import threading
from datetime import time
from pathlib import Path
from typing import Union, Sequence, Optional
import time

import dask.array as da
import nd2
import numpy as np
from pubsub import pub

from nd2_analyzer.analysis.segmentation.segmentation_cache import SegmentationCache
from nd2_analyzer.analysis.segmentation.segmentation_models import SegmentationModels
from nd2_analyzer.analysis.segmentation.segmentation_service import SegmentationService
from nd2_analyzer.utils.registration import register_images, ShiftedImage_2D_numba

"""
Singleton, implements data access for the time lapse experiments
"""


class ImageData:
    _instance: Optional["ImageData"] = None
    _lock = threading.Lock()

    def __init__(self, data, path, is_image=True, channel_n=0):
        self.data = data
        self.image_filename = path
        self.processed_images = []
        self.is_image = is_image
        self.registration_offsets: Optional[np.ndarray] = (
            None  # If we are doing registration
        )
        self.crop_coordinates = None
        self.channel_n = channel_n

        # Initialize segmentation components
        self.segmentation_cache = SegmentationCache(data)
        self.segmentation_service = SegmentationService(
            cache=self.segmentation_cache,
            models=SegmentationModels(),
            data_getter=self.get,
        )

        pub.subscribe(self.on_crop_selected, "crop_selected")
        pub.subscribe(self.on_crop_reset, "crop_reset")

    @classmethod
    def get_instance(cls) -> Optional["ImageData"]:
        """Get the current singleton instance"""
        with cls._lock:
            return cls._instance

    @classmethod
    def create_instance(cls, data, path, is_image=True, channel_n=0) -> "ImageData":
        """Create/replace the singleton instance"""
        with cls._lock:
            # Clean up old instance if it exists
            if cls._instance is not None:
                cls._instance._cleanup()

            # Normalize first path
            first_path = path[0] if isinstance(path, (list, tuple)) else path
            first_path = str(first_path)
            lower_path = first_path.lower()
            voxel = None

            # ND2 branch
            if lower_path.endswith(".nd2"):
                with nd2.ND2File(first_path) as f:
                    voxel = f.voxel_size()

            # TIFF / OME-TIFF branch
            elif lower_path.endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff", ".ome.tf2", ".ome.tf8", ".ome.btf")):
                import tifffile

                with tifffile.TiffFile(first_path) as tif:
                    # Try OME metadata first
                    try:
                        if tif.is_ome and tif.ome_metadata:
                            # Keep raw metadata around if useful later
                            ome_xml = tif.ome_metadata
                            # Pixel sizes here later if needed
                            voxel = None
                    except Exception:
                        pass
            else:
                raise ValueError(f"Unsupported image format: {first_path}")

            instance = cls(data, path, is_image, channel_n)
            instance.voxel_size = voxel
            cls._instance = instance

        pub.sendMessage("image_data_loaded", image_data=cls._instance)
        return cls._instance

    def on_crop_selected(self, coords: list):
        self.crop_coordinates = coords
        pub.sendMessage("image_data_loaded", image_data=self)

    def on_crop_reset(self):
        self.crop_coordinates = None
        pub.sendMessage("image_data_loaded", image_data=self)

    def _cleanup(self):
        pass

    def get_channel_n(self):
        return self.channel_n

    def get(self, t, p, c):
        """Helper method to retrieve raw images"""

        if len(self.data.shape) == 5:  # - has channel
            raw_image = self.data[t, p, c]
        elif len(self.data.shape) == 4:  # - no channel
            raw_image = self.data[t, p]
        else:
            print(f"Unusual data format: {len(self.data.shape)} dimensions")
            if len(self.data.shape) >= 3:
                raw_image = self.data[t, p]
            else:
                raw_image = self.data[t]

        # Compute if it's a dask array
        if hasattr(raw_image, "compute"):
            raw_image = raw_image.compute()

        # Crop/Scale if we have it
        if self.crop_coordinates is not None:
            x, y, width, height = self.crop_coordinates
            raw_image = raw_image[y : y + height, x : x + width]

        # Apply registration if we have it
        if self.registration_offsets is not None:
            _x, _y = self.registration_offsets[t]

            raw_image = ShiftedImage_2D_numba(raw_image, _x, _y)

        return raw_image

    def do_registration_p(self, p: int, c: int = 0):
        """
        Runs the image registration at a specific position. Afterward, all images will receive the transformation
        """
        # NOTE: probably heavy because of compute
        image_series = self.data[
            :, p, c
        ].compute()  # Selects position and channel, PHC by default.
        image_series = ((image_series / 65535) * 255).astype(
            np.uint8
        )  # Converting here cause it expects uint8

        # TODO: remove, but just validating the time here

        s = time.time()
        res = register_images(image_series)
        e = time.time()
        print("Image registration took {:.2f} seconds".format(e - s))

        # Store offsets here
        self.registration_offsets = res.offsets

    @classmethod
    def load_nd2(cls, file_paths: Union[str, Sequence[str]], import_mode: str | None = None):
        """
        Load one or more ND2 or TIFF files, verify that channel count, image height and width match,
        crop the P-dimension (second axis) to the smallest found, concatenate along time axis,
        and print the final shape.
        """

        if isinstance(file_paths, str):
            file_paths = [file_paths]

        arrays = []
        p_dims = []
        channels = height = width = None

        for path in file_paths:
            if path.endswith(".nd2"):
                arr = nd2.imread(path, dask=True)  # Lazy load dask
            else:
                import tifffile
                store = tifffile.imread(path, aszarr=True)
                arr = da.from_zarr(store) # Lazy load dask

            shape = arr.shape

            # Handle different file formats
            if path.endswith(".nd2"):
                with nd2.ND2File(path) as f:
                    axes = "".join(f.sizes)
                    arr, axes = cls.normalize_axes(arr, axes)
            elif path.endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff", ".ome.tf2", ".ome.tf8", ".ome.btf")):
                import tifffile
                with tifffile.TiffFile(path) as tif:
                    series = tif.series[0]
                    axes = series.axes
                    arr, axes = cls.normalize_axes(arr, axes)
            else:
                raise ValueError(
                    f"File {path} has shape {shape}; expected (T, P, Y, X) or (T, P, C, Y, X)"
                )

            T, P, C, Y, X = arr.shape

            if channels is None:
                # Set C, Y, X on the first file
                channels, height, width = C, Y, X
            else:
                # Check if files are "castable"
                if C != channels:
                    raise ValueError(f"{path}: channels {C} != {channels}")
                if Y != height:
                    raise ValueError(f"{path}: height {Y} != {height}")
                if X != width:
                    raise ValueError(f"{path}: width {X} != {width}")

            p_dims.append(P)
            arrays.append(arr)

        if import_mode in ["Directory", "batch_tiff", "stacked_tiff"]:
            full_data = arrays[0]
        else:
            if len(arrays) > 1:
                min_p = min(p_dims)
                cropped = [arr[:, :min_p, :, :, :] for arr in arrays]
                full_data = da.concatenate(cropped, axis=0)
                print(f"Cropped P to {min_p}. Final array shape: {full_data.shape}")
            else:
                full_data = arrays[0]
        print(f"Loaded {len(file_paths)} file(s). ")

        inst = cls.create_instance(data=full_data, path=file_paths, is_image=True)
        inst.channel_n = channels

        return inst

    @classmethod
    def load_tiff_directory(cls, file_map, mode, progress_callback=None):
        import tifffile
        import numpy as np
        import dask.array as da

        # Ensure a valid path exists for shape detection
        first_path = None
        for v in file_map.values():
            if v is not None:
                first_path = v
                break
        if first_path is None:
            raise ValueError("No valid TIFF paths found in file_map")

        # Sorted unique axes
        positions = sorted({k[0] for k in file_map})
        channels = sorted({k[2] for k in file_map})

        if mode == "batch_tiff":
            # Use only observed timepoints from filenames instead of forcing 0..max_t.
            # This supports sparse time series (e.g. 0, 24, 48, 96).
            times = sorted({k[1] for k in file_map if k[1] is not None})
        else:
            # Get the number of frames from the first file
            first_path = next(v for v in file_map.values() if v is not None)
            with tifffile.TiffFile(first_path) as tif:
                times = list(range(len(tif.pages)))

        T, P, C = len(times), len(positions), len(channels)

        def _to_2d(frame):
            """Ensure a frame is 2D (Y, X). Handles RGB, RGBA, and multi-page."""
            if frame.ndim == 2:
                return frame
            if frame.ndim == 3 and frame.shape[2] in (3, 4):
                return frame[:, :, 0].copy()
            if frame.ndim == 3:
                return frame[0]
            if frame.ndim == 4 and frame.shape[-1] in (3, 4):
                return frame[0, :, :, 0].copy()
            return frame.reshape(frame.shape[-2], frame.shape[-1])

        # Determine frame size and the widest dtype across all files
        sample = _to_2d(tifffile.imread(first_path))
        Y, X = sample.shape
        widest_dtype = sample.dtype
        for path in file_map.values():
            if path is not None and path != first_path:
                probe = _to_2d(tifffile.imread(path))
                if probe.dtype.itemsize > widest_dtype.itemsize:
                    widest_dtype = probe.dtype
                break  # checking one more file is enough

        print(f"TIFF directory: detected frame size {Y}x{X}, dtype={widest_dtype}")

        total = len(positions) * len(channels) * len(times)
        current = 0
        missing_frames = []
        missing_stacks = []

        import uuid
        filename = f"temp_{uuid.uuid4().hex}.dat"
        data = np.memmap(
            filename,
            dtype=widest_dtype,
            mode="w+",
            shape=(T, P, C, Y, X)
        )

        # Loop over positions and channels to load data
        for pi, p in enumerate(positions):
            for ci, c in enumerate(channels):
                if mode == "batch_tiff":
                    for ti, t in enumerate(times):
                        path = file_map.get((p, t, c))
                        if path is None:
                            missing_frames.append((p, t, c))
                        else:
                            frame = _to_2d(tifffile.imread(path))
                            data[ti, pi, ci] = frame
                        current += 1
                        if progress_callback:
                            progress = int((current / total) * 100)
                            progress_callback(progress)

                else:  # stacked_tiff
                    path = file_map.get((p, None, c))
                    if path is None:
                        missing_stacks.append((p, c))
                        current += T
                        if progress_callback:
                            progress_callback(min(int((current / total) * 100), 100))
                        continue

                    with tifffile.TiffFile(path) as tif:
                        for ti in range(T):
                            frame = _to_2d(tif.pages[ti].asarray())
                            data[ti, pi, ci] = frame
                            current += 1
                            if progress_callback:
                                progress = int((current / total) * 100)
                                progress_callback(progress)

        if missing_frames:
            sample = ", ".join(
                [f"(p={p}, t={t}, c={c})" for p, t, c in missing_frames[:8]]
            )
            suffix = " ..." if len(missing_frames) > 8 else ""
            print(
                f"Warning: {len(missing_frames)} missing frame(s) were filled with zeros. "
                f"Examples: {sample}{suffix}"
            )

        if missing_stacks:
            sample = ", ".join([f"(p={p}, c={c})" for p, c in missing_stacks[:8]])
            suffix = " ..." if len(missing_stacks) > 8 else ""
            print(
                f"Warning: {len(missing_stacks)} missing stack(s) were filled with zeros. "
                f"Examples: {sample}{suffix}"
            )

        # Convert to a dask array for lazy loading
        dask_arr = da.from_array(data, chunks=(1, 1, 1, Y, X))
        inst = cls.create_instance(data=dask_arr, path=list(file_map.values()), is_image=True, channel_n=C)
        inst._memmap_file = filename
        inst._tiff_file_map = {f"{k[0]},{k[1]},{k[2]}": v for k, v in file_map.items()}
        inst._tiff_mode = mode

        return inst

    @staticmethod
    def normalize_axes(arr, axes):
        """Adjusts axes to 5D shape"""
        axes = axes.upper()
        # Ensure required axes exist
        if "T" not in axes:
            arr = np.expand_dims(arr, axis=0)
            axes = "T" + axes
        if "P" not in axes:
            arr = np.expand_dims(arr, axis=1)
            axes = axes.replace("T", "TP")
        if "C" not in axes:
            arr = np.expand_dims(arr, axis=2)
            axes = axes.replace("TP", "TPC")
        return arr, axes

    def save_to_disk(self, filename: str):
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
        container_data = {"image_filename": self.image_filename, "is_image": self.is_image}

        # Include TIFF directory info if present
        if hasattr(self, "_tiff_file_map") and self._tiff_file_map:
            container_data["tiff_file_map"] = self._tiff_file_map
            container_data["tiff_mode"] = self._tiff_mode

        # Save container metadata
        with open(base_dir / "image_data.json", "w") as f:
            json.dump(container_data, f)

    @staticmethod
    def _relink_paths(paths, search_dirs):
        """Try to resolve missing file paths by looking in search_dirs."""
        if isinstance(paths, str):
            paths_list = [paths]
            was_str = True
        else:
            paths_list = list(paths)
            was_str = False

        resolved = []
        for p in paths_list:
            if os.path.exists(p):
                resolved.append(p)
                continue
            basename = os.path.basename(p)
            found = False
            for d in search_dirs:
                candidate = os.path.join(d, basename)
                if os.path.exists(candidate):
                    resolved.append(candidate)
                    found = True
                    break
            if not found:
                return None
        return resolved[0] if was_str else resolved

    @classmethod
    def load_from_disk(cls, filename, relink_root=None):
        """Load imagedata from path"""
        base_dir = Path(filename)

        # Load imagedata metadata
        meta_path = base_dir / "image_data.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta_json = json.load(f)
                image_filename = meta_json.get("image_filename")
                is_image = meta_json.get("is_image", True)

                files_ok = (
                    isinstance(image_filename, str) and os.path.exists(image_filename)
                ) or (
                    isinstance(image_filename, list)
                    and all(os.path.exists(_fname) for _fname in image_filename)
                )

                if not files_ok:
                    search_dirs = [str(base_dir)]
                    if relink_root:
                        search_dirs.append(relink_root)
                    relinked = cls._relink_paths(image_filename, search_dirs)
                    if relinked is not None:
                        image_filename = relinked
                        files_ok = True

                if files_ok:
                    # Check if this was a TIFF directory import
                    tiff_file_map_raw = meta_json.get("tiff_file_map")
                    tiff_mode = meta_json.get("tiff_mode")

                    if tiff_file_map_raw and tiff_mode:
                        # Reconstruct file_map with tuple keys
                        file_map = {}
                        for key_str, path in tiff_file_map_raw.items():
                            parts = key_str.split(",")
                            p = int(parts[0])
                            t = int(parts[1]) if parts[1] != "None" else None
                            c = int(parts[2])
                            file_map[(p, t, c)] = path
                        image_data = cls.load_tiff_directory(file_map, tiff_mode)
                    else:
                        image_data = cls.load_nd2(image_filename)

                    # Load segmentation cache if file exists
                    cache_path = base_dir / "segmentation_cache.h5"
                    if cache_path.exists():
                        image_data.segmentation_cache = SegmentationCache.load(
                            str(cache_path), image_data.data
                        )
                        image_data.segmentation_service = SegmentationService(
                            cache=image_data.segmentation_cache,
                            models=SegmentationModels(),
                            data_getter=image_data.get,
                        )

                    return image_data
                else:
                    raise FileNotFoundError(f"Image file not found: {image_filename}")
        else:
            raise FileNotFoundError(f"Metadata file not found in {filename}")
