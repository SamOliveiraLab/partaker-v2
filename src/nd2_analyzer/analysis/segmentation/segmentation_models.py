import os

import cv2
import numpy as np
from cellpose_omni import models, utils
from scipy.ndimage import gaussian_filter
from skimage import exposure
from skimage.measure import label
from skimage.restoration import richardson_lucy

from .unet import unet_segmentation

use_gpu = True

class SegmentationModels:

    CELLPOSE_BACT_PHASE = "bact_phase_cp3"
    CELLPOSE_BACT_FLUOR = "bact_fluor_cp3"
    OMNIPOSE_BACT_PHASE = "omnipose_bact_phase"
    OMNIPOSE_BACT_FLUOR = "omnipose_bact_fluo"
    UNET = "unet"
    CELLPOSE = "cellpose_deepbacs"

    available_models = [
        OMNIPOSE_BACT_PHASE,
        OMNIPOSE_BACT_FLUOR,
        CELLPOSE_BACT_PHASE,
        CELLPOSE_BACT_FLUOR,
        UNET,
        CELLPOSE,
    ]

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SegmentationModels, cls).__new__(cls, *args, **kwargs)
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
        _images = np.array([cv2.resize(np.array(img), (512, 512)) for img in images])
        _images = np.expand_dims(_images, axis=-1)
        segmented_images = model.predict(_images)
        segmented_images = np.array(
            [
                cv2.resize(np.array(img), (images[0].shape[1], images[0].shape[0]))
                for img in segmented_images
            ]
        )

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
            masks, _, _ = cellpose_inst.eval(images, diameter=None, channels=[0, 0])
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
                    masks[i]
                )  # Corrected from tasks[i] to masks[i]
                bw_images[i][outlines] = 0  # Set outline pixels to black (0)

            # Optionally, pad the binary masks for visualization
            binary_mask_display = np.pad(
                bw_images,
                pad_width=((0, 0), (5, 5), (5, 5)),
                mode="constant",
                constant_values=0,
            )

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

    def segment_omnipose(self, images, progress, model):
        chans = [0, 0]

        # define parameters
        params = {
            "channels": chans,  # always define this with the model
            "rescale": None,  # upscale or downscale your images, None = no rescaling
            "mask_threshold": -2,  # erode or dilate masks with higher or lower values between -5 and 5
            "flow_threshold": 0,
            # default is .4, but only needed if there are spurious masks to clean up; slows down output
            "transparency": True,  # transparency in flow output
            "omni": True,  # we can turn off Omnipose mask reconstruction, not advised
            "cluster": True,  # use DBSCAN clustering
            "resample": True,  # whether or not to run dynamics on rescaled grid or original grid
            "verbose": False,  # turn on if you want to see more output
            "tile": True,  # average the outputs from flipped (augmented) images; slower, usually not needed
            "niter": None,
            # default None lets Omnipose calculate # of Euler iterations (usually <20) but you can tune it for over/under segmentation
            "augment": False,  # Can optionally rotate the image and average network outputs, usually not needed
            # 'affinity_seg': True, # new feature, stay tuned...
            "batch_size": 4  # default is 8, halved to prevent (out of memory) OOM errors
        }

        print("type:", type(images))
        print("len:", len(images))
        print("shape:", getattr(images, "shape", None))
        print("first image shape:", images[0].shape if isinstance(images, list) else None)
        masks, flows, styles = model.eval(images, **params)
        # masks = np.array(masks)  # Ensure masks are a NumPy array

        return masks

    def segment_images(self, images, mode, progress=None, preprocess=True):
        print(f"Segmenting images using {mode} model")

        original_shape = images[0].shape

        # Preprocess images if the flag is enabled
        if preprocess:
            images = [preprocess_image(img) for img in images]

        if mode == SegmentationModels.CELLPOSE:
            if SegmentationModels.CELLPOSE not in self.models:
                self.models[self.CELLPOSE] = models.CellposeModel(
                    gpu="PARTAKER_GPU" in os.environ
                    and os.environ["PARTAKER_GPU"] == "1",
                    model_type="deepbacs_cp3",
                )

            segmented_images = self.segment_cellpose(
                images, progress, self.models[mode]
            )

        elif mode == SegmentationModels.CELLPOSE_BACT_PHASE:
            if SegmentationModels.CELLPOSE_BACT_PHASE not in self.models:
                self.models[self.CELLPOSE_BACT_PHASE] = models.CellposeModel(
                    gpu="PARTAKER_GPU" in os.environ
                    and os.environ["PARTAKER_GPU"] == "1",
                    model_type="bact_phase_cp3",
                )

            segmented_images = self.segment_cellpose(
                images, progress, self.models[mode]
            )

        elif mode == SegmentationModels.CELLPOSE_BACT_FLUOR:
            if SegmentationModels.CELLPOSE_BACT_FLUOR not in self.models:
                self.models[self.CELLPOSE_BACT_FLUOR] = models.CellposeModel(
                    gpu="PARTAKER_GPU" in os.environ
                    and os.environ["PARTAKER_GPU"] == "1",
                    model_type="bact_fluor_cp3",
                )

            segmented_images = self.segment_cellpose(
                images, progress, self.models[mode]
            )

        elif mode == SegmentationModels.UNET:
            if mode not in self.models:
                if "UNET_WEIGHTS" not in os.environ:
                    raise ValueError("UNET_WEIGHTS environment variable not set")

                target_size_seg = (512, 512)
                self.models[mode] = unet_segmentation(
                    input_size=target_size_seg + (1,),
                    pretrained_weights=os.environ["UNET_WEIGHTS"],
                )

            segmented_images = self.segment_unet(images)

        elif mode == SegmentationModels.OMNIPOSE_BACT_PHASE:
            if mode not in self.models:
                self.models[mode] = models.CellposeModel(
                    gpu=use_gpu, model_type="bact_phase_omni"
                )

            segmented_images = self.segment_omnipose(
                images, progress, self.models[mode]
            )

        elif mode == SegmentationModels.OMNIPOSE_BACT_FLUOR:
            if mode not in self.models:
                self.models[mode] = models.CellposeModel(
                    gpu=use_gpu, model_type="bact_fluor_omni"
                )

            segmented_images = self.segment_omnipose(
                images, progress, self.models[mode]
            )

        else:
            raise ValueError(f"Invalid segmentation mode: {mode}")

        resized_images = [
            self.safe_resize(
                segmented_image,
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            for segmented_image in segmented_images
        ]

        lbl_n = np.unique(resized_images[0]).shape[0]
        print(
            f"[segmentation_models.py:segment_images] Segmentation has {lbl_n} labels"
        )

        return resized_images

    def safe_resize(self, img, size, interpolation=cv2.INTER_NEAREST):
        if img is None:
            raise ValueError("Image is None")

        # Convert bool → uint8
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255

        # Convert float → uint8 safely
        elif img.dtype in [np.float32, np.float64]:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Catch ANY weird dtype
        elif img.dtype not in [np.uint8, np.uint16]:
            img = img.astype(np.uint8)

        return cv2.resize(img, size, interpolation=interpolation)

        # # Apply erosion specifically for Cellpose models
        # if is_cellpose_model:
        #     resized_images = self.apply_morphological_erosion(resized_images)

        # # Remove artifacts (optional step that can be enabled with a parameter)
        # cleaned_images = [self.remove_artifacts_from_mask(
        #     img) for img in resized_images]

        # return cleaned_images

    def remove_artifacts_from_mask(self, mask, min_area_ratio=0.2):
        """
        Remove artifacts from a segmentation mask based on cell area and morphological opening.

        Parameters:
            mask (np.ndarray): Binary or labeled segmentation mask
            min_area_ratio (float): Minimum area ratio compared to average area

        Returns:
            np.ndarray: Cleaned mask with artifacts removed
        """
        import cv2
        import numpy as np
        from skimage.measure import label, regionprops

        # First apply morphological opening to remove small artifacts
        # Create a structuring element appropriate for E. coli (rod-shaped bacteria)
        # Elliptical/oblong structuring element works well for rod-shaped
        # bacteria
        kernel_size = 3  # Start with a small kernel
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        if np.max(mask) <= 1:
            # Binary mask
            opened_mask = cv2.morphologyEx(
                mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
            )
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
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

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
    image = np.asarray(image, dtype=np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    denom = max_val - min_val
    if denom <= 0 or not np.isfinite(denom):
        # Constant/invalid frames (common for missing-data placeholders) have no signal.
        return np.zeros_like(image, dtype=np.float32)

    normalized_frame = (image - min_val) / denom
    normalized_frame = np.nan_to_num(
        normalized_frame, nan=0.0, posinf=0.0, neginf=0.0
    )

    denoised_frame = gaussian_filter(normalized_frame, sigma=1)

    # Apply CLAHE to improve contrast
    clahe = exposure.equalize_adapthist(denoised_frame, clip_limit=0.03)

    # Step 3: Deblur the image
    psf = np.ones((5, 5)) / 25  # Example PSF
    deblurred_frame = richardson_lucy(denoised_frame, psf, num_iter=30)

    return deblurred_frame
