# Model loading. TODO: move to another file

from pathlib import Path
from cellpose import models, io, utils
import datetime
from cachier import cachier
import os
import imageio.v2 as imageio
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.python.ops import array_ops, math_ops
# Adam optimizer instead of SGD...
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers.legacy import Adam  # Adam optimizer
# instead of SGD...
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    UpSampling2D,
    Concatenate,
)

# Segmentation imports
from typing import Union, List, Tuple, Callable, Dict  # Python types
from scipy.ndimage import gaussian_filter
from skimage import exposure
from skimage.restoration import richardson_lucy

#### Entropy and Loss functions ####


def pixelwise_weighted_binary_crossentropy_seg(
    y_true: tf.Tensor, y_pred: tf.Tensor
) -> tf.Tensor:
    """
    Pixel-wise weighted binary cross-entropy loss.
    The code is adapted from the Keras TF backend.
    (see their github)

    Parameters
    ----------
    y_true : Tensor
        Stack of groundtruth segmentation masks + weight maps.
    y_pred : Tensor
        Predicted segmentation masks.

    Returns
    -------
    Tensor
        Pixel-wise weight binary cross-entropy between inputs.

    """
    try:
        # The weights are passed as part of the y_true tensor:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except BaseException:
        print("Gone through an exception!")
        pass

    # Make background weights be equal to the model's prediction
    bool_bkgd = weight == 0 / 255
    weight = tf.where(bool_bkgd, y_pred, weight)

    epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
    cond = y_pred >= zeros
    relu_logits = math_ops.select(cond, y_pred, zeros)
    neg_abs_logits = math_ops.select(cond, -y_pred, y_pred)
    entropy = math_ops.add(
        relu_logits - y_pred * seg,
        math_ops.log1p(math_ops.exp(neg_abs_logits)),
        name=None,
    )

    loss = K.mean(math_ops.multiply(weight, entropy), axis=-1)

    loss = tf.scalar_mul(
        10 ** 6,
        tf.scalar_mul(
            1 /
            tf.math.sqrt(
                tf.math.reduce_sum(weight)),
            loss))

    return loss


############## U-NETS MODEL ##############
"""
A block of layers for 1 contracting level of the U-Net

Parameters
----------
input_layer : tf.Tensor
    The convolutional layer that is the output of the upper level's
    contracting block.
filters : int
    filters input for the Conv2D layers of the block.
conv2d_parameters : dict()
    kwargs for the Conv2D layers of the block.
dropout : float, optional
    Dropout layer rate in the block. Valid range is [0,1). If 0, no dropout
    layer is added.
    The default is 0
name : str, optional
    Name prefix for the layers in this block. The default is "Contracting".

Returns
-------
conv2 : tf.Tensor
    Output of this level's contracting block.

"""


# Contracting Block for the U-Net
def contracting_block(
    input_layer: tf.Tensor,
    filters: int,
    conv2d_parameters: Dict,
    dropout: float = 0,
    name: str = "Contracting",
) -> tf.Tensor:

    # Pooling layer: (sample 'images' down by factor 2)
    pool = MaxPooling2D(pool_size=(2, 2), name=name +
                        "_MaxPooling2D")(input_layer)

    # First Convolution layer
    conv1 = Conv2D(
        filters,
        3,
        **conv2d_parameters,
        name=name +
        "_Conv2D_1")(pool)

    # Second Convolution layer
    conv2 = Conv2D(
        filters,
        3,
        **conv2d_parameters,
        name=name +
        "_Conv2D_2")(conv1)

    # If a dropout is necessary, otherwise just return
    if (dropout == 0):
        return conv2
    else:
        drop = Dropout(dropout, name=name + "_Dropout")(conv2)
        return drop


"""
A block of layers for 1 expanding level of the U-Net

Parameters
----------
input_layer : tf.Tensor
    The convolutional layer that is the output of the lower level's
    expanding block
skip_layer : tf.Tensor
    The convolutional layer that is the output of this level's
    contracting block
filters : int
    filters input for the Conv2D layers of the block.
conv2d_parameters : dict()
    kwargs for the Conv2D layers of the block.
dropout : float, optional
    Dropout layer rate in the block. Valid range is [0,1). If 0, no dropout
    layer is added.
    The default is 0
name : str, optional
    Name prefix for the layers in this block. The default is "Expanding".

Returns
-------
conv3 : tf.Tensor
    Output of this level's expanding block.

"""

# Expanding Block for the U-Net


def expanding_block(
    input_layer: tf.Tensor,
    skip_layer: tf.Tensor,
    filters: int,
    conv2d_parameters: Dict,
    dropout: float = 0,
    name: str = "Expanding",
) -> tf.Tensor:

    # Up-Sampling
    up = UpSampling2D(size=(2, 2), name=name + "_UpSampling2D")(input_layer)
    conv1 = Conv2D(
        filters,
        2,
        **conv2d_parameters,
        name=name +
        "_Conv2D_1")(up)

    # Merge with skip connection layer
    merge = Concatenate(axis=3, name=name +
                        "_Concatenate")([skip_layer, conv1])

    # Convolution Layers
    conv2 = Conv2D(
        filters,
        3,
        **conv2d_parameters,
        name=name +
        "_Conv2D_2")(merge)
    conv3 = Conv2D(
        filters,
        3,
        **conv2d_parameters,
        name=name +
        "_Conv2D_3")(conv2)

    # If there needs dropout, otherwise, lets return
    if (dropout == 0):
        return conv3
    else:
        drop = Dropout(dropout, name=name + "_Dropout")(conv3)
        return drop


"""
Unstacks the mask from the weights in the output tensor for
segmentation and computes binary accuracy

Parameters
----------
y_true : Tensor
Stack of groundtruth segmentation masks + weight maps.
y_pred : Tensor
Predicted segmentation masks.

Returns
-------
Tensor
Binary prediction accuracy.

"""


def unstack_acc(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    try:
        print(y_true)
        print("y_true:", y_true.shape)
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except BaseException:
        pass

    return keras.metrics.binary_accuracy(seg, y_pred)


# Actual U-net
"""
Generic U-Net declaration.

Parameters
----------
input_size : tuple of 3 ints, optional
    Dimensions of the input tensor, excluding batch size.
    The default is (256,32,1).
final_activation : string or function, optional
    Activation function for the final 2D convolutional layer. see
    keras.activations
    The default is 'sigmoid'.
output_classes : int, optional
    Number of output classes, ie dimensionality of the output space of the
    last 2D convolutional layer.
    The default is 1.
dropout : float, optional
    Dropout layer rate in the contracting & expanding blocks. Valid range
    is [0,1). If 0, no dropout layer is added.
    The default is 0.
levels : int, optional
    Number of levels of the U-Net, ie number of successive contraction then
    expansion blocks are combined together.
    The default is 5.

Returns
-------
model : Model
    Defined U-Net model (not compiled yet).

"""


def unet(
    input_size: Tuple[int, int, int] = (256, 32, 1),
    final_activation="sigmoid",
    output_classes=1,
    dropout: float = 0,
    levels: int = 5
) -> Model:

    # Default parameters for convolution
    conv2d_params = {
        "activation": "relu",
        "padding": "same",
        "kernel_initializer": "he_normal",
    }

    # Inputs Layer
    inputs = Input(input_size, name="true_input")

    # First level input convolutional layers:
    # We pass through 2 3x3 Convolution layers...
    filters = 64
    conv = Conv2D(filters, 3, **conv2d_params, name="Level0_Conv2D_1")(inputs)
    conv = Conv2D(filters, 3, **conv2d_params, name="Level0_Conv2D_2")(conv)

    # Generating Contracting Path (that is moving down the encoder block)
    level = 0
    contracting_outputs = [conv]
    for level in range(1, levels):
        filters *= 2
        contracting_outputs.append(
            contracting_block(
                contracting_outputs[-1],
                filters,
                conv2d_params,
                dropout=dropout,
                name=f"Level{level}_Contracting",
            )
        )

    # Generating Expanding Path (that is moving up the decoder block)
    expanding_output = contracting_outputs.pop()
    while level > 0:
        level -= 1
        filters = int(filters / 2)
        expanding_output = expanding_block(
            expanding_output,
            contracting_outputs.pop(),
            filters,
            conv2d_params,
            dropout=dropout,
            name=f"Level{level}_Expanding",
        )

    # Next we have the final output layer
    output = Conv2D(
        output_classes,
        1,
        activation=final_activation,
        name="true_output")(expanding_output)
    model = Model(inputs=inputs, outputs=output)

    return model

# Unets Physical Model for Segmentation, think of it as a wrapper function...


def unet_segmentation(
    pretrained_weights=None,
    input_size: Tuple[int, int, int] = (256, 32, 1),
    levels: int = 5,
) -> Model:  # Force a Model Class to come

    # Run the following inputs into the unet algorithm defined above...
    model = unet(
        input_size=input_size,
        final_activation="sigmoid",
        output_classes=1,
        levels=levels,
    )

    # Learning rate 1e-4
    # loss = pixelwise_weighted_binary_crossentropy_seg,
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=pixelwise_weighted_binary_crossentropy_seg,
        metrics=[unstack_acc]
    )

    # If we have any pre-trained weights...
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

# target_size_seg = (512, 512)

# model = unet_segmentation(input_size = target_size_seg + (1,))
# # model.load_weights('./checkpoints/delta_2_29_01_24_5eps')
# # model.load_weights('./checkpoints/delta_2_19_02_24_200eps')
# model.load_weights('./checkpoints/delta_2_20_02_24_600eps')
# # model.summary()


# @cachier(stale_after=datetime.timedelta(days=3))

def segment_images(
        directory='/Users/hiram/Workspace/OliveiraLab/PartakerV3/src/aligned_data/XY8_Long_PHC',
        weights='/Users/hiram/Workspace/OliveiraLab/PartakerV3/src/checkpoints/delta_2_20_02_24_600eps.index'):

    target_size_seg = (512, 512)
    model = unet_segmentation(input_size=target_size_seg + (1,))
    test_images = Path(directory)
    imgs = list(map(lambda x: cv2.imread(str(x), cv2.IMREAD_GRAYSCALE), sorted(
        [img for img in test_images.iterdir()], key=lambda x: int(x.stem))))

    def my_resize(img):
        a = img[55:960, 150:810]
        a = cv2.resize(a, (512, 512))
        a = np.expand_dims(a, axis=-1)
        return a

    pred_imgs = model.predict(np.array(list(map(my_resize, imgs))))
    return pred_imgs

# Attempt to use cellpose


# Initialize the Cellpose model
# model = models.Cellpose(model_type='cyto')
# cellposemodel = models.Cellpose(gpu=True, model_type='cyto')
# cellposemodel = models.Cellpose(gpu=True, model_type='cyto3')


# class CellposeModelSingleton:
#     _instance = None

#     def __new__(cls, *args, **kwargs):
#         if not cls._instance:
#             cls._instance = super(CellposeModelSingleton, cls).__new__(cls, *args, **kwargs)

#             # Check environment "PARTAKER_GPU": "1" or "0"
#             if "PARTAKER_GPU" in os.environ and os.environ["PARTAKER_GPU"] == "1":
#                 cls._instance.model = models.CellposeModel(gpu=True, model_type='deepbacs_cp3')
#             else:
#                 cls._instance.model = models.CellposeModel(gpu=False, model_type='deepbacs_cp3')

#         return cls._instance


class SegmentationModels:
    CELLPOSE = 'cellpose'
    UNET = 'unet'

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(
                SegmentationModels, cls).__new__(
                cls, *args, **kwargs)
            cls._instance.models = {}
        return cls._instance

    # def get_model(self, mode):
    #     if mode == SegmentationModels.CELLPOSE:
    #         if SegmentationModels.CELLPOSE not in self.models:
    #             if "PARTAKER_GPU" in os.environ and os.environ["PARTAKER_GPU"] == "1":
    #                 self.models[self.CELLPOSE] = models.CellposeModel(gpu=True, model_type='deepbacs_cp3')
    #             else:
    #                 self.models[self.CELLPOSE] = models.CellposeModel(gpu=False, model_type='deepbacs_cp3')
    #         return self.models[SegmentationModels.CELLPOSE]
    #     elif mode == SegmentationModels.UNET:
    #         if SegmentationModels.UNET not in self.models:
    #             target_size_seg = (512, 512)
    #             self.models[SegmentationModels.UNET] = unet_segmentation(input_size=target_size_seg + (1,))
    #         return self.models[SegmentationModels.UNET]
    #     else:
    #         raise ValueError(f"Invalid segmentation mode: {mode}")

    """
    Segments an array of images using unet
    """

    def segment_unet(self, images):
        model = self.models[SegmentationModels.UNET]
        pred_imgs = model.predict(np.array(
            [np.expand_dims(cv2.resize(img, (512, 512)), axis=-1) for img in images]))

        print(pred_imgs.shape)
        return pred_imgs[:, :, :, 0]

    """
    Segment an array of images using cellpose
    """

    def segment_cellpose(self, images, progress):
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
        cellpose_inst = self.models[SegmentationModels.CELLPOSE]

        # Ensure images are in the correct format
        images = [img.squeeze() if img.ndim > 2 else img for img in images]

        try:
            # Run segmentation with Cellpose
            masks, _, _ = cellpose_inst.eval(
                images, diameter=None, channels=[0, 0])
            masks = np.array(masks)  # Ensure masks are a NumPy array

            # Create binary masks with borders
            bw_images = np.zeros_like(
                masks, dtype=np.uint8)  # Initialize binary mask
            bw_images[masks > 0] = 255  # Convert labeled masks to binary

            # Add borders to the binary masks
            for i in range(len(masks)):
                # Get outlines for the current mask
                outlines = utils.masks_to_outlines(masks[i])
                # Set border pixels to 0 (black) on the binary mask
                bw_images[i][outlines] = 0

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

    def segment_images(
            self,
            images,
            mode,
            model_type=None,
            progress=None,
            preprocess=True):

        # Preprocess images if the flag is enabled
        if preprocess:
            images = [preprocess_image(img) for img in images]

        if mode == SegmentationModels.CELLPOSE:
            if SegmentationModels.CELLPOSE not in self.models:
                if "PARTAKER_GPU" in os.environ and os.environ["PARTAKER_GPU"] == "1":
                    self.models[self.CELLPOSE] = models.CellposeModel(
                        gpu=True, model_type=model_type)
                else:
                    self.models[self.CELLPOSE] = models.CellposeModel(
                        gpu=False, model_type=model_type)

            # Ensure the selected model type is applied dynamically
            self.models[self.CELLPOSE].model_type = model_type

            return self.segment_cellpose(images, progress)

        elif mode == SegmentationModels.UNET:
            if SegmentationModels.UNET not in self.models:
                target_size_seg = (512, 512)
                self.models[SegmentationModels.UNET] = unet_segmentation(
                    input_size=target_size_seg + (1,))

            return self.segment_unet(images)

        else:
            raise ValueError(f"Invalid segmentation mode: {mode}")


# # Example usage
# segmentation_models = SegmentationModels()

# # Accessing the CELLPOSE model
# cellpose_model = segmentation_models.get_model(SegmentationModels.CELLPOSE)

# # Accessing the UNET model
# unet_model = segmentation_models.get_model(SegmentationModels.UNET)

# # Segmenting images using the CELLPOSE model
# segmented_images = segmentation_models.segment_images(images, mode=SegmentationModels.CELLPOSE, progress=progress)

# # Segmenting images using the UNET model
# segmented_images = segmentation_models.segment_images(images, mode=SegmentationModels.UNET, progress=progress)


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


# def segment_this_image(image):
#     # Preprocess image before segmentation
#     preprocessed_image = preprocess_image(image)

#     # Use Cellpose for segmentation
#     cellpose_inst = CellposeModelSingleton().model
#     masks, flows, styles = cellpose_inst.eval(preprocessed_image, diameter=None, channels=[0, 0])

#     # Create binary mask
#     bw_image = np.zeros_like(masks, dtype=np.uint8)
#     bw_image[masks > 0] = 255

#     # Debug: Show binary mask
#     # cv2.imshow("Binary Mask", bw_image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#     return bw_image

# def segment_all_images(images, progress=None):
#     # Get the Cellpose model instance
#     cellpose_inst = CellposeModelSingleton().model

#     # Ensure images are in the correct format
#     images = [img.squeeze() if img.ndim > 2 else img for img in images]

#     # Run segmentation
#     try:
#         masks, _, _ = cellpose_inst.eval(images, diameter=None, channels=[0, 0])
#         masks = np.array(masks)  # Ensure masks are a NumPy array
#     except Exception as e:
#         print(f"Error during segmentation: {e}")
#         return None

#     # Create binary black-and-white masks
#     try:
#         bw_images = np.zeros_like(masks, dtype=np.uint8)
#         bw_images[masks > 0] = 255  # Convert labeled masks to binary
#     except Exception as e:
#         print(f"Error converting masks to binary: {e}")
#         return None

#     # Update progress if a callback is provided
#     if progress:
#         if callable(progress):  # If it's a function
#             progress(len(images))
#         else:  # Assume it's a PyQt signal
#             progress.emit(len(images))

#     return bw_images

# """
# Segments one image and returns it, in a single channel
# """
# def _segment_this_image(image):
#     image = np.array(image)

#     target_size_seg = (512, 512)
#     model = unet_segmentation(input_size = target_size_seg + (1,))

#     def my_resize(img):
#         print(img.shape)
#         # a = img[55:960][150:810]
#         a = img
#         a = cv2.resize(a, (512, 512))
#         a = np.expand_dims(a, axis=-1)
#         return a

#     pred_imgs = model.predict(np.array([my_resize(image)]))

    print(pred_imgs.shape)
    return pred_imgs[0, :, :, 0]


def extract_individual_cells(image, segmented_image):
    """
    Extracts individual cells from the original image based on the segmented mask.

    Parameters:
    -----------
    image : np.ndarray
        The original grayscale or raw image.
    segmented_image : np.ndarray
        The binary segmented image where each cell is labeled uniquely.

    Returns:
    --------
    List of tuples where each tuple contains:
        - Cropped cell image (np.ndarray)
        - Bounding box (x, y, w, h)
    """
    # Ensure the images are the same size
    assert image.shape == segmented_image.shape, "Image and segmented image must have the same dimensions."

    # Find connected components in the segmented mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        segmented_image, connectivity=8)

    # Extract individual cells
    extracted_cells = []
    for label in range(1, num_labels):  # Skip the background (label=0)
        # Extract bounding box for the label
        x, y, w, h, area = stats[label]

        # Skip small regions (noise)
        if area < 50:
            continue

        # Crop the corresponding region from the original image
        cropped_cell = image[y:y + h, x:x + w]
        extracted_cells.append((cropped_cell, (x, y, w, h)))

    return extracted_cells


def classify_morphology(metrics):
    """
    Classify cell morphology based on its metrics.

    Parameters:
    - metrics: dict, a dictionary containing cell metrics (area, aspect_ratio, etc.).

    Returns:
    - str, the morphology class (e.g., 'Small', 'Round', 'Normal', 'Elongated', 'Deformed').
    """
    area = metrics.get("area", 0)
    aspect_ratio = metrics.get("aspect_ratio", 0)
    circularity = metrics.get("circularity", 0)

    if area < 300:
        return "Small"
    elif circularity > 0.9 and aspect_ratio < 1.2:
        return "Round"
    elif 1.2 <= aspect_ratio < 3 and 0.7 < circularity <= 0.9:
        return "Normal"
    elif aspect_ratio >= 3 and circularity < 0.7:
        return "Elongated"
    else:
        return "Deformed"


def extract_cells_and_metrics(image, segmented_image):
    """
    Extract individual cells, their bounding boxes, and metrics from a segmented image.

    Parameters:
    - image: np.ndarray, the original grayscale image.
    - segmented_image: np.ndarray, the binary segmented image.

    Returns:
    - cell_mapping: dict, a dictionary with cell IDs as keys and a dictionary of metrics and bounding boxes as values.
    """
    from skimage.measure import regionprops, label
    from skimage.color import rgb2gray
    from skimage.transform import resize

    # Debugging: print input shapes
    print(f"Original image shape: {image.shape}")
    print(f"Segmented image shape: {segmented_image.shape}")

    # Ensure the intensity image is single-channel (convert if multi-channel)
    if image.ndim == 3 and image.shape[-1] in [3, 4]:  # RGB or RGBA
        print("Converting multi-channel image to grayscale.")
        image = rgb2gray(image)

    # Check and handle shape mismatches between the intensity image and the
    # segmented image
    if image.shape != segmented_image.shape:
        print(
            f"Resizing intensity image from {image.shape} to {segmented_image.shape}")
        image = resize(
            image,
            segmented_image.shape,
            preserve_range=True,
            anti_aliasing=True)

    # Label connected regions in the segmented image
    labeled_image = label(segmented_image)

    # Debugging: print labeled image shape
    print(f"Labeled image shape: {labeled_image.shape}")

    # Extract properties for each labeled region
    cell_mapping = {}
    for region in regionprops(labeled_image, intensity_image=image):
        if region.area < 50:  # Filter out small regions (noise)
            continue

        # Calculate bounding box and metrics
        x1, y1, x2, y2 = region.bbox  # Bounding box coordinates
        metrics = {
            "area": region.area,
            "perimeter": region.perimeter,
            "equivalent_diameter": region.equivalent_diameter,
            "orientation": region.orientation,
            "aspect_ratio": region.major_axis_length / region.minor_axis_length
            if region.minor_axis_length > 0
            else 0,
            "circularity": (4 * np.pi * region.area) / (region.perimeter**2)
            if region.perimeter > 0
            else 0,
            "solidity": region.solidity,
        }

        # Classify the cell's morphology
        metrics["morphology_class"] = classify_morphology(metrics)

        # Add cell information to the mapping
        cell_id = len(cell_mapping) + 1
        cell_mapping[cell_id] = {
            "bbox": (x1, y1, x2, y2),
            "metrics": metrics,
        }

    return cell_mapping


def annotate_image(image, cell_mapping):
    """
    Annotate the original image with bounding boxes and IDs for detected cells.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("The input image is not a valid numpy array.")
    print(f"Annotating image of shape: {image.shape}")  # Debugging

    # Ensure it's in RGB format
    annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for cell_id, data in cell_mapping.items():
        x1, y1, x2, y2 = data["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, str(cell_id), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return annotated


def annotate_binary_mask(segmented_image, cell_mapping):
    """
    Annotate the binary segmented mask with bounding boxes and morphology class color codes.

    Parameters:
    -----------
    segmented_image : np.ndarray
        The binary segmented mask (black and white).
    cell_mapping : dict
        Cell ID mapping with metrics and bounding boxes.

    Returns:
    --------
    annotated : np.ndarray
        Annotated binary mask with bounding boxes and labels.
    """
    # Ensure input is grayscale
    if len(segmented_image.shape) == 3:
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Convert grayscale to RGB for annotations
    annotated = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2RGB)

    # Define color mapping for morphology classes
    morphology_colors = {
        "Small": (0, 0, 255),  # Blue
        "Round": (255, 0, 0),  # Red
        "Normal": (0, 255, 0),  # Green
        "Elongated": (255, 255, 0),  # Yellow
        "Deformed": (255, 0, 255),  # Magenta
    }

    for cell_id, data in cell_mapping.items():
        y1, x1, y2, x2 = data["bbox"]

        # Get the morphology class and corresponding color
        morphology_class = data["metrics"].get("morphology_class", "Normal")
        color = morphology_colors.get(
            morphology_class, (255, 255, 255))  # Default to white

        # Draw bounding box with morphology-specific color
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Add text label for cell ID and class
        label = f"{cell_id}: {morphology_class}"
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated
