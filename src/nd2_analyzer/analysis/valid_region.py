"""Shared non-destructive valid region for microcolony measurements."""
import numpy as np
from skimage.measure import label

BORDER_PX = 15


def valid_mask(shape):
    h, w = shape[:2]
    if min(h, w) <= 2 * BORDER_PX:
        raise ValueError('Image is too small for the 15-pixel analysis border.')
    mask = np.zeros((h, w), dtype=bool)
    mask[BORDER_PX:-BORDER_PX, BORDER_PX:-BORDER_PX] = True
    return mask


def valid_labels(image):
    """Preserve integer IDs; exclude whole cells crossing the valid boundary."""
    source = np.asarray(image)
    if source.ndim != 2:
        raise ValueError('Tracking requires integer label masks, not RGB exports.')
    values = np.unique(source)
    labels = label(source > 0) if set(values).issubset({0, 1, 255}) else source.astype(np.int32, copy=True)
    valid = valid_mask(labels.shape)
    excluded = np.unique(labels[~valid])
    labels[np.isin(labels, excluded[excluded > 0])] = 0
    labels[~valid] = 0
    return labels
