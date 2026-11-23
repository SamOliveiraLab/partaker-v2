"""Colony geometry in source coordinates, restricted to the analysis interior."""
import cv2
import numpy as np

from .valid_region import BORDER_PX, valid_mask


def analysis_image(image):
    """Return the interior view without modifying the source image."""
    valid_mask(image.shape[:2])  # Validate dimensions using the shared border.
    return image[BORDER_PX:-BORDER_PX, BORDER_PX:-BORDER_PX]



def detect_colony_contours(image, intensity_threshold, min_colony_size, kernel_size):
    """Use the verifier's Otsu-adjusted threshold and filled outer contours."""
    gray = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = float(np.clip(otsu + intensity_threshold - 50, 0, 255))
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if cv2.contourArea(contour) > min_colony_size]


def detect_dark_regions(raw_image, roi_mask, threshold=128, min_area=1, kernel_size=1):
    """Return dark components confined to the supplied ROI, preserving holes."""
    roi = np.asarray(roi_mask, dtype=bool)
    if roi.shape != raw_image.shape:
        raise ValueError("ROI mask must match the image shape")
    if not np.any(roi):
        return np.zeros(roi.shape, dtype=np.uint8)
    image = cv2.normalize(raw_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    mask = ((image <= threshold) & roi).astype(np.uint8)
    kernel_size = max(1, int(kernel_size)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) & roi
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel) & roi
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    keep = np.zeros(count, dtype=np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def interior_colonies(colonies, shape):
    """Drop excluded centroids and give each remaining component its own ROI.

    Geometry and masks remain in source coordinates. Return whether clipping,
    removal, or splitting changed the observations and invalidated tracking.
    """
    from .biofilm_metric_service import BiofilmMetricService

    valid = valid_mask(shape)
    result, changed = [], False
    for colony in colonies:
        original = BiofilmMetricService._colony_mask(colony, shape)
        moments = cv2.moments(original.astype(np.uint8), binaryImage=True)
        if not moments['m00']:
            changed = True
            continue
        cx, cy = moments['m10']/moments['m00'], moments['m01']/moments['m00']
        h, w = shape
        if not (BORDER_PX <= cx < w-BORDER_PX and BORDER_PX <= cy < h-BORDER_PX):
            changed = True
            continue
        mask = original & valid
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if len(c) >= 3 and cv2.contourArea(c) > 0]
        clipped = bool(np.any(original & ~valid))
        modified = clipped or len(contours) != 1
        changed |= modified
        for contour in contours:
            component = np.zeros(shape, np.uint8)
            cv2.drawContours(component, [contour], -1, 1, -1)
            x, y, width, height = cv2.boundingRect(contour)
            m = cv2.moments(component, binaryImage=True)
            item = dict(colony, contour=contour, polygon=contour.reshape(-1, 2).tolist(),
                        polygon_points=contour.reshape(-1, 2).tolist(), mask=component.astype(bool),
                        area=float(cv2.contourArea(contour)), bbox=(x,y,x+width,y+height),
                        bbox_width=width, bbox_height=height,
                        centroid=(m['m10']/m['m00'],m['m01']/m['m00']),
                        border_clipped=clipped or bool(colony.get('border_clipped', False)))
            if modified:
                for field in list(item):
                    if field == 'track_id' or field.startswith('tracking_'):
                        item.pop(field)
                if len(contours) > 1:
                    item['source'] = 'split'
            item['colony_id'] = len(result)+1
            result.append(item)
    return result, changed
