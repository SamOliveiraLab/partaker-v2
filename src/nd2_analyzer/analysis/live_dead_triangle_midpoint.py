"""Cell-excluded background Triangle midpoint masks; no Qt or preprocessing."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

METHOD = 'triangle_shared_midpoint'
SETTINGS = dict(hist_bins=2048, peak_sigma_bins=2.0,
                peak_prominence_fraction=0.03, peak_distance_bins=12,
                block_triangle='rightmost significant peak, basin-restricted longer tail',
                cell_triangle='tallest significant peak, forced left tail',
                selection='lower arithmetic mean of nine midpoints; tie lower channel index',
                interpolation='bilinear at actual block centers, constant edge extension',
                threshold_rule='finite raw intensity > shared surface',
                background_sampling='finite pixels where saved cell labels equal zero',
                background_margin_pixels=0,
                background_bin_range='eligible background pixels per channel/frame',
                cell_bin_range='all finite original pixels per channel/frame',
                otsu_used=False, background_subtraction=False, smoothing_sigma=0)


def grid_slices(shape):
    if len(shape) != 2 or min(shape) < 3:
        raise ValueError('Triangle midpoint requires a 2D image at least 3x3 pixels.')
    ys = np.linspace(0, shape[0], 4, dtype=int)
    xs = np.linspace(0, shape[1], 4, dtype=int)
    return [(r, c, slice(ys[r], ys[r+1]), slice(xs[c], xs[c+1]))
            for r in range(3) for c in range(3)]


def histogram_edges(image):
    values = image[np.isfinite(image)]
    if not values.size:
        raise ValueError('No finite fluorescence pixels.')
    low, high = float(values.min()), float(values.max())
    upper = float(np.nextafter(np.float32(high), np.float32(np.inf)))
    if low == high:
        upper = low + max(1, abs(low)*1e-6)
    if not np.isfinite(upper):
        raise ValueError('Intensity range exceeds the OpenCV histogram range.')
    return np.linspace(low, upper, SETTINGS['hist_bins']+1)


def histogram(image, edges, mask=None):
    valid = np.isfinite(image)
    if mask is not None:
        valid &= mask
    values = np.where(valid, image, 0).astype(np.float32)
    counts = cv2.calcHist([values], [0], valid.astype(np.uint8),
                          [len(edges)-1], [float(edges[0]), float(edges[-1])]).ravel().astype(np.float64)
    if counts.sum() != valid.sum():
        raise ValueError('Histogram omitted pixels; inspect intensity range/precision.')
    return counts


def grade_hist(h):
    smooth = gaussian_filter1d(h, SETTINGS['peak_sigma_bins'])
    if not h.sum():
        return 'empty', np.array([], dtype=int), smooth
    peaks, _ = find_peaks(np.r_[0., smooth, 0.],
                         prominence=smooth.max()*SETTINGS['peak_prominence_fraction'],
                         distance=SETTINGS['peak_distance_bins'])
    peaks = peaks-1
    peaks = peaks[(peaks >= 0) & (peaks < len(h))]
    if not len(peaks):
        peaks = np.array([int(smooth.argmax())])
    return {1: 'uni-modal', 2: 'bi-modal'}.get(len(peaks), 'multi-modal'), peaks, smooth


def triangle_reference(h, edges, *, cell_left=False):
    grade, peaks, smooth = grade_hist(h)
    if not h.sum():
        raise ValueError('Empty histogram: no Triangle reference.')
    occupied = np.flatnonzero(h)
    left, right = int(occupied[0]), int(occupied[-1])
    peak = int(peaks[np.argmax(smooth[peaks])] if cell_left else peaks[-1])
    if cell_left:
        tail = left
    else:
        if len(peaks) > 1:
            previous = int(peaks[-2])
            left = previous + int(np.argmin(smooth[previous:peak+1]))
        tail = left if peak-left > right-peak else right
    if abs(tail-peak) < 2:
        raise ValueError('Peak has no resolved Triangle tail; inspect histogram/binning.')
    indices = np.arange(min(peak, tail), max(peak, tail)+1)
    y = smooth/smooth[peak]
    x = (indices-peak)/abs(tail-peak)
    end_x, end_y = (tail-peak)/abs(tail-peak), y[tail]-1
    distance = np.abs(end_x*(y[indices]-1)-end_y*x)/np.hypot(end_x, end_y)
    interior = (indices != peak) & (indices != tail)
    indices, distance = indices[interior], distance[interior]
    if cell_left and distance.max() <= np.finfo(float).eps:
        raise ValueError('Cell left tail has no resolved bend; no reference is substituted.')
    index = int(indices[np.argmax(distance)])
    centers = (edges[:-1]+edges[1:])/2
    return dict(threshold=float(centers[index]), peak=float(centers[peak]),
                tail=float(centers[tail]), index=index, peak_index=peak,
                tail_index=tail, grade=grade)


def threshold_surface(grid, shape):
    slices = grid_slices(shape)
    xc = [(xs.start+xs.stop-1)/2 for r, c, ys, xs in slices if r == 0]
    yc = [(ys.start+ys.stop-1)/2 for r, c, ys, xs in slices if c == 0]
    along_x = np.vstack([np.interp(np.arange(shape[1]), xc, row) for row in grid])
    surface = np.empty(shape, dtype=np.float32)
    for x in range(shape[1]):
        surface[:, x] = np.interp(np.arange(shape[0]), yc, along_x[:, x])
    return surface


def analyze_channel(image, labels, role):
    """Exclude cell footprints before partitioning; keep cell-only reference separate."""
    image = np.asarray(image)
    if image.shape != labels.shape or image.dtype.kind not in 'uif':
        raise ValueError(f'{role}: numeric grayscale image and cell labels must have matching shapes.')
    finite = np.isfinite(image)
    eligible = (labels == 0) & finite
    if not np.any(eligible):
        raise ValueError(f'{role}: no finite cell-free background pixels in this frame.')
    background_edges = histogram_edges(image[eligible])
    # Preserve the existing cell-only reference/binning and whole-cell intensity plots.
    edges = histogram_edges(image)
    blocks, details, sampling = [], [], []
    for i, (r, c, ys, xs) in enumerate(grid_slices(labels.shape)):
        included = eligible[ys, xs]
        excluded_cells = labels[ys, xs] > 0
        total = int(included.size)
        count = int(included.sum())
        sampling.append(dict(block=i+1, row=r, column=c,
                             bounds_xyxy=[int(v) for v in (xs.start, ys.start, xs.stop, ys.stop)],
                             total_pixels=total, eligible_pixels=count,
                             eligible_fraction=count/total,
                             excluded_cell_pixels=int(excluded_cells.sum()),
                             nonfinite_background_pixels=int((~finite[ys, xs] & ~excluded_cells).sum())))
        if not count:
            raise ValueError(f'{role}, block {i+1}: no finite cell-free background pixels '
                             f'({int(excluded_cells.sum())}/{total} pixels covered by segmented cells).')
        h = histogram(image[ys, xs], background_edges, included)
        blocks.append(h)
        try:
            details.append(triangle_reference(h, background_edges))
        except ValueError as error:
            raise ValueError(f'{role}, block {i+1}: {error}; {count}/{total} eligible background pixels.') from error
    cell_h = histogram(image, edges, labels > 0)
    try:
        cell_detail = triangle_reference(cell_h, edges, cell_left=True)
    except ValueError as error:
        raise ValueError(f'{role}, cell histogram: {error}') from error
    refs = np.array([d['threshold'] for d in details]).reshape(3, 3)
    midpoints = (refs+cell_detail['threshold'])/2
    return dict(edges=edges, background_edges=background_edges, blocks=np.asarray(blocks),
                background_eligible=eligible, sampling=sampling, block_details=details,
                cell_histogram=cell_h, cell_detail=cell_detail, block_references=refs,
                midpoints=midpoints, mean_midpoint=float(midpoints.mean()))


def analyze_channels(images, labels, channel_indices):
    labels = np.asarray(labels)
    if (labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0) or not np.any(labels > 0)):
        raise ValueError('Saved nonnegative integer cell labels with nonempty cell area are required.')
    grid_slices(labels.shape)
    images = {role: np.asarray(image) for role, image in images.items()}
    results = {role: analyze_channel(image, labels, role) for role, image in images.items()}
    channels = {role: int(channel_indices[role]) for role in images}
    selected = min(results, key=lambda role: (results[role]['mean_midpoint'], channels[role]))
    grid = results[selected]['midpoints'].copy()
    surface = threshold_surface(grid, labels.shape)
    for role, image in images.items():
        results[role]['mask'] = np.isfinite(image) & (image > surface)
    valid = np.logical_and.reduce([np.isfinite(image) for image in images.values()])
    compositions = {}
    paired = len(images) == 2
    if paired:
        l, d = results['live']['mask'], results['dead']['mask']
        regions = (l & ~d, d & ~l, l & d, ~l & ~d)
        categories = ['live_only', 'dead_only', 'double_positive', 'neither']
    else:
        positive = results['cell_viability']['mask']
        regions = (positive, ~positive)
        categories = ['cell_viability_positive', 'cell_viability_negative']
    for scope, area in [('cell', valid & (labels > 0)), ('whole_frame', valid)]:
        counts = [int(np.count_nonzero(area & region)) for region in regions]
        total = int(area.sum())
        compositions[scope] = dict(total=total, counts=counts, categories=categories,
                                   percent=[100*c/total for c in counts] if total else None)
    return dict(channels=results, channel_indices=channels, selected_role=selected,
                selected_channel=channels[selected], shared_grid=grid, surface=surface,
                shape=labels.shape, analysis_mode='live_dead' if paired else 'cell_viability',
                composition=compositions)


def analyze_pair(live, dead, labels, live_channel, dead_channel):
    if live_channel == dead_channel:
        raise ValueError('Live and Dead channels must differ.')
    return analyze_channels({'live': live, 'dead': dead}, labels,
                            {'live': live_channel, 'dead': dead_channel})


def analyze_cell_viability(image, labels, channel):
    return analyze_channels({'cell_viability': image}, labels, {'cell_viability': channel})


def audit_record(result):
    return dict(settings=SETTINGS, analysis_mode=result['analysis_mode'], shape=list(result['shape']),
                selected_role=result['selected_role'], selected_channel=result['selected_channel'],
                channel_indices=result['channel_indices'], shared_grid=result['shared_grid'].tolist(),
                channels={role: dict(mean_midpoint=r['mean_midpoint'],
                                     block_references=r['block_references'].tolist(),
                                     midpoints=r['midpoints'].tolist(), cell_triangle=r['cell_detail'],
                                     background_sampling=r['sampling'],
                                     block_triangles=r['block_details'])
                          for role, r in result['channels'].items()},
                composition=result['composition'])
