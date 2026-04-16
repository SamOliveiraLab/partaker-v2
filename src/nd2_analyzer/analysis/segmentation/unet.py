import tensorflow as tf
# import tensorflow.keras as keras

# import keras.api._v2.keras as keras
from keras import backend as K
from tensorflow.python.ops import array_ops, math_ops
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    UpSampling2D,
    Concatenate,
)
# from tensorflow.keras import backend as K
# from tensorflow.python.ops import array_ops, math_ops
# from tensorflow.keras.optimizers import Adam  # Adam optimizer instead of SGD...
# # from tensorflow.keras.optimizers.legacy import Adam  # Adam optimizer instead of SGD...
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Input,
#     Conv2D,
#     MaxPooling2D,
#     Dropout,
#     UpSampling2D,
#     Concatenate,
# )

# Model loading. TODO: move to another file

# target_size_seg = (512, 512)

# model = unet_segmentation(input_size = target_size_seg + (1,))
# # model.load_weights('./checkpoints/delta_2_29_01_24_5eps')
# # model.load_weights('./checkpoints/delta_2_19_02_24_200eps')
# model.load_weights('./checkpoints/delta_2_20_02_24_600eps')
# # model.summary()

# Segmentation imports
from typing import Union, List, Tuple, Callable, Dict  # Python types

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
