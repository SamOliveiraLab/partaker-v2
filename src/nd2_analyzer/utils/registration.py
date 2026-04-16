from typing import Optional

from numba import jit, prange
import numpy as np
from skimage import exposure
from tqdm import tqdm


@jit(nopython=True)
def ShiftedImage_2D_numba(Image, XShift, YShift):
    """Ultra-fast numba-compiled version."""
    if XShift == 0 and YShift == 0:
        return Image.copy()

    h, w = Image.shape[:2]

    if Image.ndim == 3:
        result = np.zeros((h, w, Image.shape[2]), dtype=Image.dtype)
        channels = Image.shape[2]
    else:
        result = np.zeros((h, w), dtype=Image.dtype)
        channels = 1

    # Parallel pixel-wise shifting with bounds checking
    for i in range(h):
        for j in range(w):
            src_i = i - YShift
            src_j = j - XShift

            if 0 <= src_i < h and 0 <= src_j < w:
                if Image.ndim == 3:
                    for c in range(channels):
                        result[i, j, c] = Image[src_i, src_j, c]
                else:
                    result[i, j] = Image[src_i, src_j]
            else:
                # Edge padding: find nearest valid pixel
                nearest_i = max(0, min(h - 1, src_i))
                nearest_j = max(0, min(w - 1, src_j))

                if Image.ndim == 3:
                    for c in range(channels):
                        result[i, j, c] = Image[nearest_i, nearest_j, c]
                else:
                    result[i, j] = Image[nearest_i, nearest_j]

    return result


# # sum of absolute differences (SAD) metric alignment
# # Optimized version
# @jit(nopython=True, parallel=True)
# def SAD(a, b):
#     # Vectorized SAD; extremely fast
#     flat_a = a.ravel()
#     flat_b = b.ravel()
#     total = 0.0
#     for i in range(len(flat_a)):
#         total += abs(float(flat_a[i]) - float(flat_b[i]))
#     return total / len(flat_a)


# NOTE: I hate 'AI' so much. It gave me f****** unsafe code
# Alternative: Use numba's parallel reduction
@jit(nopython=True)
def SAD(a, b):
    """Parallel-safe SAD using local accumulation."""
    flat_a = a.ravel()
    flat_b = b.ravel()
    n = len(flat_a)

    # Create thread-local storage
    num_threads = min(8, n // 1000)  # Limit threads for small arrays
    local_sums = np.zeros(num_threads, dtype=np.float64)

    for i in range(n):
        thread_id = i % num_threads
        local_sums[thread_id] += abs(float(flat_a[i]) - float(flat_b[i]))

    return np.sum(local_sums) / n


# We use a Tree Search Algorithm to find possible alignment
# Let Image_1 be the orginal
# Let Image_2 be the aligned
# Displacement object is our nodes, [x,y]
# Assumption, there is always a better alignment up, down, left, and right if its not the same image
def alignment_MAE(Image_1, Image_2, depth_cap):
    iterative_cap = 0
    Best_SAD = SAD(Image_1, Image_2)
    Best_Displacement = [0, 0]
    q = []
    visited_states = [[0, 0]]  # Add (0,0) displacement
    q.append(Best_Displacement)  # Append (0,0) displacement

    while iterative_cap != depth_cap and q:
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
            if move not in visited_states:
                visited_states.append(move)  # Marked as Visited

                # Perform shift and calculate
                new_image = ShiftedImage_2D_numba(Image_2, move[0], move[1])
                cand_SAD = SAD(Image_1, new_image)
                if cand_SAD < Best_SAD:
                    Best_SAD = cand_SAD
                    Best_Displacement = move
                    q.append(move)

    # This means we cannot find a better move.
    return Best_Displacement, Best_SAD


# This was a good fix for edge detection
@jit(nopython=True)
def compute_row_means_2d(img):
    """Custom row mean computation for 2D arrays."""
    rows, cols = img.shape
    row_means = np.empty(rows, dtype=np.float64)
    for i in range(rows):
        total = 0.0
        for j in range(cols):
            total += float(img[i, j])
        row_means[i] = total / cols
    return row_means


@jit(nopython=True)
def compute_row_means_3d(img):
    """Custom row mean computation for 3D arrays."""
    rows, cols, channels = img.shape
    row_means = np.empty(rows, dtype=np.float64)
    for i in range(rows):
        total = 0.0
        for j in range(cols):
            for k in range(channels):
                total += float(img[i, j, k])
        row_means[i] = total / (cols * channels)
    return row_means


@jit(nopython=True)
def compute_bottom_top_gradient(row_brightness):
    # Calculate gradient manually
    gradient = np.empty(len(row_brightness) - 1, dtype=np.float64)
    for i in range(len(gradient)):
        gradient[i] = row_brightness[i + 1] - row_brightness[i]

    # Suppress extreme values near edges
    height = row_brightness.shape[0]
    for i in range(len(gradient)):
        if (i <= 100 or i >= height - 100) and abs(gradient[i]) >= 150:
            gradient[i] = 0.0

    # Find top edge (maximum gradient in upper half)
    half_height = height // 2
    max_val = -np.inf
    top_edge = 0
    for i in range(min(half_height, len(gradient))):
        if gradient[i] > max_val:
            max_val = gradient[i]
            top_edge = i

    # Find bottom edge (minimum gradient in lower half)
    min_val = np.inf
    bottom_edge = half_height

    for i in range(half_height, len(gradient)):
        if gradient[i] < min_val:
            min_val = gradient[i]
            bottom_edge = i

    return top_edge, bottom_edge


def edge_detection_numba_fixed(img):
    """
    Numba-compatible vertical edge detection function.

    Args:
        img: Input image (2D or 3D numpy array)

    Returns:
        tuple: (top_edge_row, bottom_edge_row)
    """
    # Compute row-wise brightness averages
    if img.ndim == 3:
        row_brightness = np.mean(img, axis=(1, 2))
    else:
        row_brightness = np.mean(img, axis=1)

    return compute_bottom_top_gradient(row_brightness)


from dataclasses import dataclass


@dataclass
class ImageRegistrationResult:
    registered_images: Optional[np.ndarray] = None
    offsets: Optional[np.ndarray] = None


def register_images(
    img_array,
    iteration_depth: int = 1000,
    m=False,
    verbose: bool = False,
    mcm: bool = False,
    return_images: bool = False,
):
    """
    Previously called remove_stage_jitter_MAE_opt

    Computers offsets for each image, assuming that the first one is the base.
    """

    # Add Scores path just for curiosity
    scores = []
    image_offsets = [(0, 0)]  # Base is the reference, no transformation
    transformed_images = []

    if img_array.ndim != 3:
        print("Give a series of grayscale images. Shape: {}".format(img_array.shape))

    base = exposure.rescale_intensity(img_array[0])
    base_top, base_bottom = edge_detection_numba_fixed(base)

    # TODO: verify
    if base.ndim == 3:
        base = base[:, :, 0]  # Reduce to the 2D

    iteration = 0  # TODO: implement in more pythonic way

    for _frame in tqdm(img_array[1:]):
        iteration += 1

        template_image = exposure.rescale_intensity(_frame)  # Get rid of low exposure

        template_top, template_bottom = edge_detection_numba_fixed(template_image)

        if template_image.ndim == 3:
            template_image = template_image[:, :, 0]  # Reduce to the 2D

        displacement, score = alignment_MAE(base, template_image, iteration_depth)
        scores.append(score)
        # print("SCORE:", score)

        if mcm:
            displacement[0] = 0

        y_shift = int(
            np.mean([(base_top - template_top), (base_bottom - template_bottom)])
        )
        image_offsets.append((displacement[0], y_shift))

        if return_images:
            shifted_image = ShiftedImage_2D_numba(
                template_image, displacement[0], y_shift
            )  # X,Y
            transformed_images.append(shifted_image)

        # For my purposes
        # background = Image.fromarray(base)
        # overlay = Image.fromarray(shifted_image)
        #
        # new_img = Image.blend(background, overlay, 0.5)

        # print("Overlay for image to compare against jitter (PHC)", iteration, ":", filename)
        # plt.imshow(new_img)
        # plt.show()
        #
        # # Write the new image in target folder
        # cv2.imwrite(os.path.join(output_path, filename), shifted_image);

    # print ("Scores:", scores)
    # print("The X_Shifts:", X_shifts)
    # print("The Y_Shifts:", Y_shifts)

    image_offsets = np.array(image_offsets)

    return ImageRegistrationResult(
        registered_images=transformed_images if return_images else None,
        offsets=image_offsets,
    )
