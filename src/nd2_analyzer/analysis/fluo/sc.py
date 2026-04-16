# import numpy as np
# import cv2
# from scipy.ndimage import label
# from .rpu import RPUParams
# from concurrent.futures import ThreadPoolExecutor, as_completed
#
# import matplotlib.pyplot as plt
#
# FLUO_EPSILON = 0.01
#
#
# def process_component(
#         component,
#         labeled_array,
#         fluorescence_image,
#         background_fluo,
#         rpu):
#     mask = labeled_array == component
#     fluorescence_avg = np.array(fluorescence_image[mask].flatten()).mean()
#     # Only adds if the fluorescence is some signal, and if this signal
#     # is larger than 110% of the background
#     if fluorescence_avg <= FLUO_EPSILON or fluorescence_avg < (
#             background_fluo * 1.1):
#         return None
#     return rpu.compute(fluorescence_avg) if rpu else fluorescence_avg
#
#
# def process_image(i, binary_images, fluorescence_images, rpu):
#     result = []
#     labeled_array, num_features = label(binary_images[i])
#
#     # Extract baseline fluorescence from the background
#     background_mask = labeled_array == 0
#     background_fluo = np.array(
#         fluorescence_images[i][background_mask].flatten()).mean()
#
#     print(num_features)
#     with ThreadPoolExecutor() as executor:
#         futures = [
#             executor.submit(
#                 process_component,
#                 component,
#                 labeled_array,
#                 fluorescence_images[i],
#                 background_fluo,
#                 rpu) for component in range(
#                 1,
#                 num_features + 1)]
#         for future in as_completed(futures):
#             res = future.result()
#             if res is not None:
#                 result.append(res)
#
#     return result if result else None
#
#
# def analyze_fluorescence_singlecell(binary_images,
#                                     fluorescence_images,
#                                     rpu: RPUParams = None, parallel=False):
#
#     if not parallel:
#         return analyze_fluorescence_singlecell_sequential(
#             binary_images, fluorescence_images, rpu)
#
#     else:
#         results = []
#         timestamps = []
#
#         for i in range(binary_images.shape[0]):
#             result = process_image(i, binary_images, fluorescence_images, rpu)
#
#             if not result:
#                 continue
#
#             timestamps.append(i)
#             results.append(result)
#
#         print(timestamps)
#         return results, timestamps
#
#
# """
# Isolates the fluorescence against each connected component (cell) in the binary image
# It gets the mean fluorescence value for each connected component
#
# binary_images: array of binary images
# fluorescence_images: array of fluorescence images
#
# Returns:
# sc_fluo: list of lists of fluorescence values for each connected component
# timestamps: timestamp of each valid fluorescence image
# """
#
#
# def analyze_fluorescence_singlecell_sequential(
#         binary_images,
#         fluorescence_images,
#         rpu: RPUParams = None):
#     results = []
#     timestamps = []
#
#     for i in range(binary_images.shape[0]):
#         result = []
#         labeled_array, num_features = label(binary_images[i])
#
#         # Extract baselline fluorescence from the background
#         background_mask = labeled_array == 0
#         background_fluo = np.array(
#             fluorescence_images[i][background_mask].flatten()).mean()
#
#         print(num_features)
#         for component in range(1, num_features + 1):
#             mask = labeled_array == component
#             fluorescence_avg = np.array(
#                 fluorescence_images[i][mask].flatten()).mean()
#             # Only adds if the fluorescence is some signal, and if this signal
#             # is larger than 110% of the background
#             if fluorescence_avg <= FLUO_EPSILON or fluorescence_avg < (
#                     background_fluo * 1.1):
#                 continue
#
#             result.append(rpu.compute(fluorescence_avg)
#                           if rpu else fluorescence_avg)
#
#         if len(result) == 0:
#             continue
#
#         # plt.figure(figsize=(12, 10))
#         # plt.hist(result, bins=100)
#         # plt.show()
#
#         timestamps.append(i)
#         results.append(result)
#
#     print(timestamps)
#
#     return results, timestamps
#
#
# def analyze_fluorescence_total(fluorescence_images, rpu: RPUParams = None):
#     results = []
#     timestamps = []
#
#     for i, fluorescence_image in enumerate(fluorescence_images):
#         fluorescence_avg = fluorescence_image.flatten().mean()
#
#         if fluorescence_avg <= FLUO_EPSILON:
#             continue
#
#         results.append(rpu.compute(fluorescence_avg)
#                        if rpu else fluorescence_avg)
#         timestamps.append(i)
#
#     return results, timestamps
# from nd2_analyzer.data.frame import TLFrame


# def analyze_fluorescence(_frame: TLFrame):
