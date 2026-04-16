import matplotlib
import numpy as np
import imageio.v2 as iio
import cv2
import random


def save_tracks(segmentation, tracks):

    cmap = matplotlib.colormaps['gist_rainbow']

    tracking_views = []

    large_tracks = random.sample([t for t in tracks if len(t) > 35], 100)
    # large_tracks = [t for t in tracks if len(t) > 35]

    for track_n, _track in enumerate(large_tracks):
        track_color = cmap(track_n / len(large_tracks))[:3]

        # print('--------\nTrack: ', track_n)
        for i, _t in enumerate(_track.t):
            # print('timestep:', _t)

            # Adds new frames as needed
            while len(tracking_views) < _t + 1:
                tracking_views.append(np.zeros(segmentation.shape[1:] + (3, )))

            view = tracking_views[_t]

            x = round(max(0, min(segmentation.shape[2] - 1, _track.x[i])))
            y = round(max(0, min(segmentation.shape[1] - 1, _track.y[i])))

            # X and Y flippled, images...
            # view[y, x] = track_color
            cv2.circle(view, (x, y), 4, track_color, -1)

    temp = np.zeros(segmentation.shape[1:] + (3, ))
    for i in range(len(tracking_views)):
        tracking_views[i] += temp
        temp = tracking_views[i].copy()

    tracking_views = [(np.clip(tv, 0, 1) * 255).astype('uint8')
                      for tv in tracking_views]

    tracking_views = np.array(tracking_views)

    # Now, aligning with the segmentation

    iio.mimsave('tracks.gif', tracking_views, fps=15)
