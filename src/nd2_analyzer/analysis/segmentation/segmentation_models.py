from asyncio import tasks
from pathlib import Path
import matplotlib.pyplot as plt
import math
import numpy as np
import imageio.v2 as imageio
import os
import cv2

from cachier import cachier
import datetime

from cellpose import models, io, utils

from .unet import unet_segmentation
from scipy.ndimage import gaussian_filter
from skimage import exposure
from skimage.restoration import richardson_lucy
from skimage.measure import label

class SegmentationModels:
    CELLPOSE = 'cellpose'
    UNET = 'unet'
    CELLPOSE_FT_0 = 'cellpose_finetuned'
    CELLPOSE_BACT_PHASE = 'bact_phase_cp3'
    CELLPOSE_BACT_FLUOR = 'bact_fluor_cp3'
    CELLPOSE_BACT_HHLN_MAR_14 = 'CP_20250314_100004_bact_phase_hhln'
    OMNIPOSE_BACT_PHASE_AFFINITY = 'bact_phase_affinity'

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(
                SegmentationModels, cls).__new__(
                cls, *args, **kwargs)
            cls._instance.models = {}
        return cls._instance

    """
    Segments using U-Net
    - Patches image
    - Removes artifacts
    - Indentifies singles components
    """

    def segment_unet(self, images):
        model = self.models[SegmentationModels.UNET]
        # patches = []
        # patch_indices = []

        # # Divide each image into 512x512 patches
        # for img_idx, img in enumerate(images):
        #     img = np.array(img)
        #     height, width = img.shape[:2]

        #     for i in range(0, height, 512):
        #         for j in range(0, width, 512):
        #             patch = img[i:i+512, j:j+512]

        #             # If the patch is smaller than 512x512, pad it with zeros
        #             if patch.shape[0] < 512 or patch.shape[1] < 512:
        #                 padded_patch = np.zeros((512, 512), dtype=patch.dtype)
        #                 padded_patch[:patch.shape[0], :patch.shape[1]] = patch
        #                 patch = padded_patch

        #             patches.append(np.expand_dims(patch, axis=-1))
        #             patch_indices.append((img_idx, i, j))

        # # Segment all patches at once
        # patches = np.array(patches)
        # segmented_patches = model.predict(patches)

        # # Create an empty array to store the segmented results
        # segmented_images = np.zeros((len(images), images[0].shape[0], images[0].shape[1]), dtype=np.float32)

        # Combine the segmented patches back into the original images
        # for idx, (img_idx, i, j) in enumerate(patch_indices):
        #     segmented_patch = segmented_patches[idx, :, :, 0]

        #     # Remove padding if necessary
        #     if i + 512 > segmented_images[img_idx].shape[0]:
        #         segmented_patch = segmented_patch[:segmented_images[img_idx].shape[0] - i, :]
        #     if j + 512 > segmented_images[img_idx].shape[1]:
        #         segmented_patch = segmented_patch[:, :segmented_images[img_idx].shape[1] - j]

        #     segmented_images[img_idx, i:i+512, j:j+512] = segmented_patch

        # Remove artifacts with morphological operations
        # for idx, img in enumerate(segmented_images):
        #     kernel = np.ones((5, 5), np.uint8)
        #     img = cv2.morphologyEx(np.array(img), cv2.MORPH_CLOSE, kernel)
        #     img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        #     segmented_images[idx] = img

        # # Transform each binary cell in an ID using connected components
        # for idx, img in enumerate(segmented_images):
        #     num_labels, labels = cv2.connectedComponents(img)
        #     segmented_images[idx] = labels
        _images = np.array([cv2.resize(np.array(img), (512, 512))
                           for img in images])
        _images = np.expand_dims(_images, axis=-1)
        segmented_images = model.predict(_images)
        segmented_images = np.array([cv2.resize(np.array(
            img), (images[0].shape[1], images[0].shape[0])) for img in segmented_images])

        # for img in segmented_images:
        #     cv2.imshow('Segmented Image', img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        try:
            bw_images = np.zeros_like(segmented_images, dtype=np.uint8)
            # Convert labeled masks to binary
            bw_images[segmented_images > 0.5] = 255
        except Exception as e:
            print(f"Error converting masks to binary: {e}")
            return None

        return bw_images

    def segment_cellpose(self, images, progress, cellpose_inst):
        """
        Segment cells using Cellpose and return binary masks with borders.

        Parameters:
        -----------
        images : list of numpy.ndarray
            The input images to segment.
        progress : callable or Signal
            A callback or signal to update progress.

        Returns:
        --------
        binary_mask_display : numpy.ndarray
            The binary masks with borders for each segmented cell.
        """

        # Ensure images are in the correct format
        images = [img.squeeze() if img.ndim > 2 else img for img in images]

        try:
            # Run segmentation with Cellpose
            masks, _, _ = cellpose_inst.eval(
                images, diameter=None, channels=[0, 0])
            masks = np.array(masks)  # Ensure masks are a NumPy array

            # Label the segmented regions uniquely
            labeled_masks = np.zeros_like(masks, dtype=np.int32)
            for i in range(len(masks)):
                # Proper labeling of segmented regions
                labeled_masks[i] = label(masks[i])

            # Create binary masks for visualization (convert labeled regions to
            # 255)
            bw_images = np.where(labeled_masks > 0, 255, 0).astype(np.uint8)

            # Add outlines to the binary masks
            for i in range(len(masks)):
                outlines = utils.masks_to_outlines(
                    masks[i])  # Corrected from tasks[i] to masks[i]
                bw_images[i][outlines] = 0  # Set outline pixels to black (0)

            # Optionally, pad the binary masks for visualization
            binary_mask_display = np.pad(bw_images, pad_width=(
                (0, 0), (5, 5), (5, 5)), mode='constant', constant_values=0)

        except Exception as e:
            print(f"Error during segmentation or mask processing: {e}")
            return None

        # Update progress if a callback is provided
        if progress:
            if callable(progress):  # If it's a function
                progress(len(images))
            else:  # Assume it's a PyQt signal
                progress.emit(len(images))

        return binary_mask_display

    def segment_omnipose(self, images, progress, omnipose_inst):
        """
        Segment cells using Omnipose and return binary masks.

        Parameters:
        -----------
        images : list of numpy.ndarray
            The input images to segment.
        progress : callable or Signal
            A callback or signal to update progress.
        omnipose_inst : cellpose_omni.models.CellposeModel
            The Omnipose model instance.

        Returns:
        --------
        binary_mask_display : numpy.ndarray
            The binary masks for each segmented cell.
        """
        
        # Ensure images are in the correct format
        images = [img.squeeze() if img.ndim > 2 else img for img in images]
        
        # Apply Omnipose-specific normalization
        try:
            from cellpose_omni import transforms
            from omnipose.utils import normalize99
            
            processed_images = []
            for img in images:
                # Move minimum dimension and convert to single channel if needed
                img_proc = transforms.move_min_dim(img)
                if len(img_proc.shape) > 2:
                    img_proc = np.mean(img_proc, axis=-1)
                # Apply normalize99 for optimal Omnipose performance
                img_proc = normalize99(img_proc)
                processed_images.append(img_proc)
            
            images = processed_images
            
        except ImportError:
            print("Warning: cellpose_omni or omnipose not available, using standard preprocessing")
            # Fallback to standard normalization
            processed_images = []
            for img in images:
                if len(img.shape) > 2:
                    img = np.mean(img, axis=-1)
                img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
                processed_images.append(img_normalized)
            images = processed_images

        try:
            # Define Omnipose parameters optimized for bacterial cells
            params = {
                'channels': None,  # Auto-detect channels
                'rescale': None,   # No rescaling
                'mask_threshold': -2,  # Optimized for bacterial cells
                'flow_threshold': 0,   # Default flow threshold
                'transparency': True,
                'omni': True,         # Enable Omnipose reconstruction
                'cluster': True,      # Use DBSCAN clustering
                'resample': True,     # Run dynamics on rescaled grid
                'verbose': False,
                'tile': False,
                'niter': None,        # Auto-calculate iterations
                'augment': False,
                'affinity_seg': True  # Enable affinity segmentation
            }
            
            # Run segmentation with Omnipose
            masks, _, _ = omnipose_inst.eval(images, **params)
            masks = np.array(masks)

            # Convert to binary masks for compatibility with existing pipeline
            bw_images = np.where(masks > 0, 255, 0).astype(np.uint8)

            # Add outlines (similar to cellpose processing)
            from cellpose import utils
            for i in range(len(masks)):
                outlines = utils.masks_to_outlines(masks[i])
                bw_images[i][outlines] = 0  # Set outline pixels to black

            # Pad binary masks for visualization
            binary_mask_display = np.pad(bw_images, pad_width=(
                (0, 0), (5, 5), (5, 5)), mode='constant', constant_values=0)

        except Exception as e:
            print(f"Error during Omnipose segmentation: {e}")
            return None

        # Update progress if a callback is provided
        if progress:
            if callable(progress):
                progress(len(images))
            else:
                progress.emit(len(images))

        return binary_mask_display

    def segment_images(
            self,
            images,
            mode,
            progress=None,
            preprocess=True):
        print(f"Segmenting images using {mode} model")

        original_shape = images[0].shape

        # Check if this is a Cellpose-based model
        is_cellpose_model = mode in [
            SegmentationModels.CELLPOSE,
            SegmentationModels.CELLPOSE_BACT_PHASE,
            SegmentationModels.CELLPOSE_BACT_FLUOR,
            SegmentationModels.CELLPOSE_BACT_HHLN_MAR_14
        ]
        
        # Check if this is an Omnipose-based model
        is_omnipose_model = mode in [
            SegmentationModels.OMNIPOSE_BACT_PHASE_AFFINITY
        ]

        # Preprocess images if the flag is enabled (skip for Omnipose as it has custom preprocessing)
        if preprocess and not is_omnipose_model:
            images = [preprocess_image(img) for img in images]

        if mode == SegmentationModels.CELLPOSE:
            if SegmentationModels.CELLPOSE not in self.models:
                self.models[self.CELLPOSE] = models.CellposeModel(
                    gpu="PARTAKER_GPU" in os.environ and os.environ["PARTAKER_GPU"] == "1", model_type='deepbacs_cp3')

            segmented_images = self.segment_cellpose(
                images, progress, self.models[mode])

        elif mode == SegmentationModels.CELLPOSE_BACT_PHASE:
            if SegmentationModels.CELLPOSE_BACT_PHASE not in self.models:
                self.models[self.CELLPOSE_BACT_PHASE] = models.CellposeModel(
                    gpu="PARTAKER_GPU" in os.environ and os.environ["PARTAKER_GPU"] == "1", model_type='bact_phase_cp3')

            segmented_images = self.segment_cellpose(
                images, progress, self.models[mode])

        elif mode == SegmentationModels.CELLPOSE_BACT_FLUOR:
            if SegmentationModels.CELLPOSE_BACT_FLUOR not in self.models:
                self.models[self.CELLPOSE_BACT_FLUOR] = models.CellposeModel(
                    gpu="PARTAKER_GPU" in os.environ and os.environ["PARTAKER_GPU"] == "1", model_type='bact_fluor_cp3')

            segmented_images = self.segment_cellpose(
                images, progress, self.models[mode])

        elif mode == SegmentationModels.CELLPOSE_BACT_FLUOR:
            if SegmentationModels.CELLPOSE_BACT_FLUOR not in self.models:
                self.models[self.CELLPOSE_BACT_FLUOR] = models.CellposeModel(
                    gpu="PARTAKER_GPU" in os.environ and os.environ["PARTAKER_GPU"] == "1", model_type='bact_fluor_cp3')

            segmented_images = self.segment_cellpose(
                images, progress, self.models[mode])

        elif mode == SegmentationModels.CELLPOSE_BACT_HHLN_MAR_14:
            if SegmentationModels.CELLPOSE_BACT_HHLN_MAR_14 not in self.models:
                self.models[self.CELLPOSE_BACT_HHLN_MAR_14] = models.CellposeModel(
                    gpu="PARTAKER_GPU" in os.environ and os.environ["PARTAKER_GPU"] == "1", model_type='bact_fluor_cp3')

            segmented_images = self.segment_cellpose(
                images, progress, self.models[mode])

        elif mode == SegmentationModels.UNET:
            if SegmentationModels.UNET not in self.models:
                if "UNET_WEIGHTS" not in os.environ:
                    raise ValueError(
                        "UNET_WEIGHTS environment variable not set")

                target_size_seg = (512, 512)
                self.models[SegmentationModels.UNET] = unet_segmentation(
                    input_size=target_size_seg + (1,), pretrained_weights=os.environ["UNET_WEIGHTS"])

            segmented_images = self.segment_unet(images)

        elif mode == SegmentationModels.OMNIPOSE_BACT_PHASE_AFFINITY:
            if SegmentationModels.OMNIPOSE_BACT_PHASE_AFFINITY not in self.models:
                try:
                    from cellpose_omni import models as omnipose_models
                    from omnipose.gpu import use_gpu
                    
                    # Check GPU availability for Omnipose
                    use_omnipose_gpu = use_gpu() if "PARTAKER_GPU" in os.environ and os.environ["PARTAKER_GPU"] == "1" else False
                    
                    self.models[self.OMNIPOSE_BACT_PHASE_AFFINITY] = omnipose_models.CellposeModel(
                        gpu=use_omnipose_gpu, 
                        model_type='bact_phase_affinity'
                    )
                except ImportError as e:
                    raise ImportError(f"Omnipose not available. Please install with: pip install cellpose-omni omnipose. Error: {e}")

            segmented_images = self.segment_omnipose(
                images, progress, self.models[mode])

        else:
            raise ValueError(f"Invalid segmentation mode: {mode}")

        resized_images = [
            cv2.resize(
                segmented_image,
                (original_shape[1],
                 original_shape[0]),
                interpolation=cv2.INTER_NEAREST) for segmented_image in segmented_images]

        # Apply erosion specifically for Cellpose and Omnipose models
        if is_cellpose_model or is_omnipose_model:
            resized_images = self.apply_morphological_erosion(resized_images)

        # Remove artifacts (optional step that can be enabled with a parameter)
        cleaned_images = [self.remove_artifacts_from_mask(
            img) for img in resized_images]

        return cleaned_images

    def remove_artifacts_from_mask(self, mask, min_area_ratio=0.2):
        """
        Remove artifacts from a segmentation mask based on cell area and morphological opening.

        Parameters:
            mask (np.ndarray): Binary or labeled segmentation mask
            min_area_ratio (float): Minimum area ratio compared to average area

        Returns:
            np.ndarray: Cleaned mask with artifacts removed
        """
        from skimage.measure import label, regionprops
        import numpy as np
        import cv2

        # First apply morphological opening to remove small artifacts
        # Create a structuring element appropriate for E. coli (rod-shaped bacteria)
        # Elliptical/oblong structuring element works well for rod-shaped
        # bacteria
        kernel_size = 3  # Start with a small kernel
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        if np.max(mask) <= 1:
            # Binary mask
            opened_mask = cv2.morphologyEx(
                mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        else:
            # Labeled mask - convert to binary, open, then re-label
            binary_mask = (mask > 0).astype(np.uint8)
            opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            if np.max(opened_mask) == 0:  # If opening removed everything
                return mask  # Return original mask

        # Convert to labeled image
        if np.max(opened_mask) <= 1:
            labeled_mask = label(opened_mask)
        else:
            labeled_mask = opened_mask.copy()

        # Calculate areas
        regions = regionprops(labeled_mask)
        if not regions:
            return mask  # If no regions found, return original

        areas = [region.area for region in regions]

        # Find average area
        mean_area = np.mean(areas)

        # Set threshold
        area_threshold = mean_area * min_area_ratio

        # Create clean mask
        clean_mask = np.zeros_like(mask)
        for region in regions:
            if region.area >= area_threshold:
                if np.max(mask) <= 255:
                    clean_mask[labeled_mask == region.label] = 255
                else:
                    clean_mask[labeled_mask == region.label] = region.label

        return clean_mask

    def apply_morphological_erosion(self, masks, kernel_size=3):
        """
        Apply morphological erosion to segmentation masks.

        Parameters:
            masks (list): List of segmentation masks
            kernel_size (int): Size of the erosion kernel

        Returns:
            list: Eroded segmentation masks
        """
        import cv2
        import numpy as np

        # Create a circular/elliptical structuring element (good for bacterial
        # cells)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        eroded_masks = []
        for mask in masks:
            # For binary masks
            if np.max(mask) <= 1 or np.max(mask) == 255:
                binary_mask = mask.astype(np.uint8)
                if np.max(binary_mask) == 1:
                    binary_mask = binary_mask * 255
                eroded = cv2.erode(binary_mask, kernel, iterations=1)
                eroded_masks.append(eroded)
            # For labeled masks
            else:
                # Create a binary version, erode it, then relabel
                binary = (mask > 0).astype(np.uint8) * 255
                eroded_binary = cv2.erode(binary, kernel, iterations=1)

                # Relabel the eroded binary mask
                from skimage.measure import label
                eroded_labeled = label(eroded_binary)
                eroded_masks.append(eroded_labeled)

        return eroded_masks


def preprocess_image(image):
    """
    Preprocess an image by applying Gaussian blur, CLAHE, and Richardson-Lucy deblurring.

    Parameters:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Preprocessed image.
    """
    normalized_frame = (image - np.min(image)) / \
        (np.max(image) - np.min(image))

    denoised_frame = gaussian_filter(normalized_frame, sigma=1)

    # Apply CLAHE to improve contrast
    clahe = exposure.equalize_adapthist(denoised_frame, clip_limit=0.03)

    # Step 3: Deblur the image
    psf = np.ones((5, 5)) / 25  # Example PSF
    deblurred_frame = richardson_lucy(denoised_frame, psf, num_iter=30)

    return deblurred_frame
