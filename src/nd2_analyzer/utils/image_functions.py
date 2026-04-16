import numpy as np
import cv2
import os
import glob
from skimage import exposure
from numpy.lib.function_base import diff

import matplotlib.pyplot as plt

import os
import warnings

from matplotlib import pyplot as plt

# Shifting the image by a margin of pixels
import skimage.transform as trans
from scipy import signal
from PIL import Image
from scipy import stats as stat
from itertools import product

# Image Analysis
import numpy as np
from scipy.fft import fft, ifft

import glob

from tifffile import imread
from matplotlib import pyplot as plt
from skimage import io, exposure, data
import numpy as np
from PIL import Image
from scipy import stats as st
# from aicsimageio import AICSImage

from scipy.optimize import minimize
from numpy import diff
from scipy.signal import find_peaks

from skimage import exposure
from skimage.filters import unsharp_mask
from numpy.polynomial import polynomial as P

from pathlib import Path

from cachier import cachier
import datetime

# Dirty Implementation of Shifting Images


def ShiftedImage_2D(Image, XShift, YShift):
    # Quick guard
    if (XShift == 0 and YShift == 0):
        return Image

    M = np.float32([
        [1, 0, XShift],
        [0, 1, YShift]
    ])

    shifted = cv2.warpAffine(Image, M, (Image.shape[1], Image.shape[0]))
    shifted_image = shifted

    # Shift Down
    if (YShift > 0):
        shifted_image = shifted_image[YShift:]
        shifted_image = np.pad(
            shifted_image, ((YShift, 0), (0, 0)), 'edge')  # Pad Up

    # Shift Up
    if (YShift < 0):
        shifted_image = shifted_image[:shifted.shape[0] - abs(YShift)]
        shifted_image = np.pad(
            shifted_image, ((0, abs(YShift)), (0, 0)), 'edge')  # Pad Down

    # Shift Left
    if (XShift > 0):
        shifted_image = np.delete(shifted_image, slice(0, XShift), 1)
        shifted_image = np.pad(
            shifted_image, ((0, 0), (XShift, 0)), 'edge')  # Pad Left

    if (XShift < 0):
        shifted_image = np.delete(
            shifted_image,
            slice(
                shifted.shape[1] -
                abs(XShift),
                shifted.shape[1]),
            1)
        shifted_image = np.pad(
            shifted_image, ((0, 0), (0, abs(XShift))), 'edge')  # Pad Right

    return shifted_image


def ShiftedImage_3D(Image, XShift, YShift):
    # Quick guard
    if (XShift == 0 and YShift == 0):
        return Image

    M = np.float32([
        [1, 0, XShift],
        [0, 1, YShift]
    ])

    shifted = cv2.warpAffine(Image, M, (Image.shape[1], Image.shape[0]))
    shifted_image = shifted

    # Shift Down
    if (YShift > 0):
        shifted_image = shifted_image[YShift:]
        shifted_image = np.pad(
            shifted_image, ((YShift, 0), (0, 0), (0, 0)), 'constant', constant_values=(
                0,))  # Pad Up

    # Shift Up
    if (YShift < 0):
        shifted_image = shifted_image[:shifted.shape[0] - abs(YShift)]
        shifted_image = np.pad(
            shifted_image, ((0, abs(YShift)), (0, 0), (0, 0)), 'constant', constant_values=(
                0,))  # Pad Down

    # Shift Left
    if (XShift > 0):
        shifted_image = np.delete(shifted_image, slice(0, XShift), 1)
        shifted_image = np.pad(
            shifted_image, ((0, 0), (XShift, 0), (0, 0)), 'constant', constant_values=(
                0,))  # Pad Left

    # Shift Up
    if (XShift < 0):
        shifted_image = np.delete(
            shifted_image,
            slice(
                shifted.shape[1] -
                abs(XShift),
                shifted.shape[1]),
            1)
        shifted_image = np.pad(
            shifted_image, ((0, 0), (0, abs(XShift)), (0, 0)), 'constant', constant_values=(
                0,))  # Pad Right

    plt.imshow(shifted_image)
    plt.show()

    return shifted_image


def SAD(A, B):
    cutA = A.ravel()
    cutB = B.ravel()
    MAE = np.sum(
        np.abs(
            np.subtract(
                cutA,
                cutB,
                dtype=np.float64))) / cutA.shape[0]
    return MAE

# sum of absolute differences (SAD) metric alignment, quick n dirty
# We use a Tree Search Algorithm to find possible alignment
# Let Image_1 be the orginal
# Let Image_2 be the aligned
# Displacement object is our nodes, [x,y]
# Assumption, there is always a better alignment up, down, left, and right
# if its not the same image


def alignment_MAE(Image_1, Image_2, depth_cap):
    iterative_cap = 0
    Best_SAD = SAD(Image_1, Image_2)
    Best_Displacement = [0, 0]
    q = []
    visited_states = [[0, 0]]  # Add (0,0) displacement
    q.append(Best_Displacement)  # Append (0,0) displacement

    while (iterative_cap != depth_cap and q):
        curr_state = q.pop(0)
        x = curr_state[0]
        y = curr_state[1]

        iterative_cap += 1

        movement_arr = [
            [x, y - 1],  # Up
            [x, y + 1],  # Down
            [x + 1, y],  # Left
            [x - 1, y],  # Right
            [x - 1, y - 1],  # Diagonal
            [x + 1, y + 1],  # Diagonal
            [x + 1, y - 1],  # Diagonal
            [x - 1, y + 1],  # Diagonal
        ]

        for move in movement_arr:
            if (move not in visited_states):
                visited_states.append(move)  # Marked as Visited

                # Perform shift and calculate
                new_image = ShiftedImage_2D(Image_2, move[0], move[1])
                cand_SAD = SAD(Image_1, new_image)

                if (cand_SAD < Best_SAD):
                    Best_SAD = cand_SAD
                    Best_Displacement = move

                    q.append(move)

                # This means we cannot find a better move.

    return Best_Displacement, Best_SAD

# Vec4f is (x1, y1, x2, y2)


def y_shift_emphasis(image, block_threshold, MAE_shift):
    output_img = image

    # Need to turn into uint8 for Straight Line Detection
    img = np.uint8(image)
    lsd = cv2.createLineSegmentDetector(0)
    lines_contour = lsd.detect(img)[0]

    drawn_img = lsd.drawSegments(img, lines_contour)
    #     plt.imshow(drawn_img)
    #     plt.show()

    horizontal_lines = {}
    for x in lines_contour:
        for y in x:
            cand_gradient = abs(y[1] - y[3])
            if (cand_gradient < 10):
                horizontal_lines[cand_gradient] = y

    horz = list(horizontal_lines.values())

    top_y = np.min(horz)
    bottom_y = np.max(horz)

    return top_y, bottom_y

# Takes in image and returns the edges for top and bottom parametrically, (x,y)
# Takes in RGB image


def edge_cropping_estimation_vertical(img, m):
    main_bright = img

    local_vertical = []

    # Vertical Cutting
    for row in range(0, main_bright.shape[0]):
        temp_arr = []
        for col in range(0, main_bright.shape[1]):
            temp_arr.append(np.mean(main_bright[row][col]))
        local_vertical.append(np.mean(temp_arr))

    # ================ Vertical axis squish ================
    x_vertical = list(range(1, main_bright.shape[0] + 1))
    y_vertical = local_vertical

    dydx_vertical = diff(y_vertical) / diff(x_vertical)
    y_verticle_dydx = list(range(1, main_bright.shape[0]))

    for i in range(0, len(dydx_vertical)):
        # Below Crazy 150 values
        if ((dydx_vertical[i] >= 150 and i <= 100) or (
                dydx_vertical[i] <= -150 and i <= 100)):
            dydx_vertical[i] = 0

        # Above Crazy 150 values
        if ((dydx_vertical[i] >= 150 and i >= (main_bright.shape[0] - 100))
                or (dydx_vertical[i] <= -150 and (main_bright.shape[0] - 100))):
            dydx_vertical[i] = 0

    top_m_derivatives_ind = np.argpartition(dydx_vertical, m)[m:]
    sorted_ind_m = sorted(top_m_derivatives_ind)
    clustered_sorted_ind_m = []

    cluster_iter_m = 0
    prev_m = sorted_ind_m[0]
    cluster_sum_m = 0

    for i in range(0, len(sorted_ind_m)):
        if (i == len(sorted_ind_m) - 1):
            clustered_sorted_ind_m.append(int(cluster_sum_m / cluster_iter_m))
        # If the previous value is outside the range of the current value, i
        elif (prev_m >= (sorted_ind_m[i] + 100) or prev_m <= (sorted_ind_m[i] - 100)):
            clustered_sorted_ind_m.append(int(cluster_sum_m / cluster_iter_m))
            cluster_sum_m = sorted_ind_m[i]
            cluster_iter_m = 1
            prev_m = sorted_ind_m[i]
        else:
            cluster_sum_m += sorted_ind_m[i]
            cluster_iter_m += 1
            prev_m = sorted_ind_m[i]

    # print("The VERTICAL DERIVATIVE:")
    plt.plot(y_verticle_dydx, dydx_vertical)
    for i in clustered_sorted_ind_m:
        plt.axvline(x=i, color='r')
    plt.show()

    top = clustered_sorted_ind_m[0]
    bottom = clustered_sorted_ind_m[len(clustered_sorted_ind_m) - 1]

    return top, bottom

# Takes in image and returns the edges for top and bottom parametrically, (x,y)
# Assumes Bottom is always min and top is always max


def edge_cropping_estimation_vertical_high_low_distr(img):
    main_bright = img

    local_vertical = []

    # Vertical Cutting
    for row in range(0, main_bright.shape[0]):
        temp_arr = []
        for col in range(0, main_bright.shape[1]):
            temp_arr.append(np.mean(main_bright[row][col]))
        local_vertical.append(np.mean(temp_arr))

    # ================ Vertical axis squish ================
    x_vertical = list(range(1, main_bright.shape[0] + 1))
    y_vertical = local_vertical

    dydx_vertical = diff(y_vertical) / diff(x_vertical)
    y_verticle_dydx = list(range(1, main_bright.shape[0]))

    for i in range(0, len(dydx_vertical)):
        # Below Crazy 150 values
        if ((dydx_vertical[i] >= 150 and i <= 100) or (
                dydx_vertical[i] <= -150 and i <= 100)):
            dydx_vertical[i] = 0

        # Above Crazy 150 values
        if ((dydx_vertical[i] >= 150 and i >= (main_bright.shape[0] - 100))
                or (dydx_vertical[i] <= -150 and (main_bright.shape[0] - 100))):
            dydx_vertical[i] = 0

    max_val = np.max(dydx_vertical)
    max_index = np.where(dydx_vertical == max_val)[0][0]
    while (max_index > (img.shape[1] / 2)):
        # print("Cycling max_index:", max_index)
        # Reset the value as it is not needed anymore
        dydx_vertical[max_index] = 0
        max_val = np.max(dydx_vertical)
        max_index = np.where(dydx_vertical == max_val)[0][0]

    min_val = np.min(dydx_vertical)
    min_index = np.where(dydx_vertical == min_val)[0][0]
    while (min_index < (img.shape[1] / 2)):
        # print("Cycling min_index:", min_index)
        # Reset the value as it is not needed anymore
        dydx_vertical[min_index] = 0
        min_val = np.min(dydx_vertical)
        min_index = np.where(dydx_vertical == min_val)[0][0]

    # print("The VERTICAL DERIVATIVE (Pattern Distribution):")
    plt.plot(y_verticle_dydx, dydx_vertical)
    plt.axvline(x=max_index, color='r')
    plt.axvline(x=min_index, color='r')
    plt.show()

    top = max_index
    bottom = min_index

    return top, bottom


def remove_stage_jitter_MAE(
        output_path,
        source_path,
        YFP_path,
        YFP_output_path,
        Cherry_path,
        Cherry_output_path,
        iteration_depth,
        m,
        verbose,
        MCM):
    # Add Scores path just for curiosity
    scores = []
    X_shifts = []
    Y_shifts = []

    YFP = False
    Cherry = False

    # Create Output folder
    if (not os.path.exists(output_path)):
        os.makedirs(output_path)

    # Check for YFP and Cherry
    if (YFP_path is not None and YFP_path != ''):
        YFP = True
        if (not os.path.exists(YFP_output_path)):
            os.makedirs(YFP_output_path)

    if (Cherry_path is not None and Cherry_path != ''):
        if (not os.path.exists(Cherry_output_path)):
            os.makedirs(Cherry_output_path)
        Cherry = True

    # Get training image files list:
    image_name_arr = glob.glob(os.path.join(
        source_path, "*.png")) + glob.glob(os.path.join(source_path, "*.tif"))
    image_name_arr_sorted = sorted(
        image_name_arr,
        key=lambda x: int(
            Path(x).stem))
#     image_name_arr_sorted = sorted(image_name_arr, key = lambda x:x[64:68]);
#     image_name_arr_sorted = sorted(image_name_arr, key = lambda x:x[26:]);
    if (verbose):
        print(image_name_arr_sorted)

    base_image = os.path.basename(image_name_arr_sorted[0])
    base = cv2.imread(os.path.join(source_path, base_image))
    base = exposure.rescale_intensity(base)  # Get rid of low exposure

    base_top, base_bottom = edge_cropping_estimation_vertical_high_low_distr(
        base)
#     base_top, base_bottom = edge_cropping_estimation_vertical(base, m);

    # Write the base image in target folder
    cv2.imwrite(
        os.path.join(
            output_path,
            os.path.basename(
                image_name_arr_sorted[0])),
        base)

    if base.ndim == 3:
        base = base[:, :, 0]  # Reduce to the 2D

    iteration = 0

    for x in range(1, len(image_name_arr_sorted)):
        iteration += 1

        # Open Current file
        filename = os.path.basename(image_name_arr_sorted[x])
        template_image = cv2.imread(image_name_arr_sorted[x])
        template_image = exposure.rescale_intensity(
            template_image)  # Get rid of low exposure

        print("Looking at file:", filename, "(" + str(iteration) +
              "/" + str(len(image_name_arr_sorted)) + ")")

        template_top, template_bottom = edge_cropping_estimation_vertical_high_low_distr(
            template_image)
#         template_top, template_bottom = edge_cropping_estimation_vertical(template_image, m);

        if template_image.ndim == 3:
            template_image = template_image[:, :, 0]  # Reduce to the 2D

        displacement, score = alignment_MAE(
            base, template_image, iteration_depth)
        scores.append(score)
        print("SCORE:", score)

        if (MCM):
            displacement[0] = 0

        X_shifts.append(displacement[0])
        Y_shifts.append(
            int(np.mean([(base_top - template_top), (base_bottom - template_bottom)])))
        shifted_image = ShiftedImage_2D(template_image, displacement[0], int(
            np.mean([(base_top - template_top), (base_bottom - template_bottom)])))  # X,Y

        # Shift the YFP and Cherry as well
        if (YFP):
            try:
                infile_YFP = image_name_arr_sorted[x].replace(
                    "PHC", "YFP").replace(".png", ".tif")
                filename_YFP = os.path.basename(infile_YFP)
                template_image_YFP = imread(infile_YFP)
                shifted_image_YFP = ShiftedImage_2D(template_image_YFP, displacement[0], int(
                    np.mean([(base_top - template_top), (base_bottom - template_bottom)])))  # X,Y

                # print("YFP:")
                # plt.imshow(shifted_image_YFP)
                # plt.show()

                cv2.imwrite(
                    os.path.join(
                        YFP_output_path,
                        filename_YFP),
                    shifted_image_YFP)
            except Exception as e:
                print(
                    f"YFP Error in image {image_name_arr_sorted[x]}. Exception: {e}")

        if (Cherry):
            try:
                infile_Cherry = image_name_arr_sorted[x].replace(
                    "PHC", "Cherry").replace(".png", ".tif")
                filename_Cherry = os.path.basename(infile_Cherry)
                template_image_Cherry = imread(infile_Cherry)
                shifted_image_Cherry = ShiftedImage_2D(template_image_Cherry, displacement[0], int(
                    np.mean([(base_top - template_top), (base_bottom - template_bottom)])))  # X,Y

                # print("Cherry:")
                # plt.imshow(shifted_image_Cherry)
                # plt.show()

                cv2.imwrite(
                    os.path.join(
                        Cherry_output_path,
                        filename_Cherry),
                    shifted_image_Cherry)
            except Exception as e:
                print(
                    f"mCherry Error in image {image_name_arr_sorted[x]}. Exception: {e}")

        # For my purposes
        background = Image.fromarray(base)
        overlay = Image.fromarray(shifted_image)

        new_img = Image.blend(background, overlay, 0.5)

        print(
            "Overlay for image to compare against jitter (PHC)",
            iteration,
            ":",
            filename)
        # plt.imshow(new_img)
        # plt.show()

        # Write the new image in target folder
        cv2.imwrite(os.path.join(output_path, filename), shifted_image)

    print("Scores:", scores)
    print("The X_Shifts:", X_shifts)
    print("The Y_Shifts:", Y_shifts)

    return scores
