"""Microcolony PIV export and isolated worker. Importing this module needs NumPy only.

Run this file directly in the PIV environment; never import the application package
in that process (its __init__ imports Qt/TensorFlow). Coordinates precede display crop.
"""
from pathlib import Path
import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import sys
import traceback
import uuid
from datetime import datetime

import numpy as np

PRESETS = {
    128: ((128, 64, 32), (96, 48, 24), 63.5),
    256: ((256, 128, 64, 32), (192, 96, 48, 24), 127.5),
    384: ((384, 256, 128, 64, 32), (352, 224, 96, 48, 24), 191.5),
}
VARIANTS = (('01_grayscale_all', 'gray', False),
            ('02_segmented_all', 'bright', False),
            ('03_grayscale_inner_seed', 'gray', True),
            ('04_segmented_inner_seed', 'bright', True))
SEGMENT_COLUMNS = ['path_id', 'time_start', 'time_end', 'capture_start', 'capture_end',
                   'x0_px', 'y0_px', 'x1_px', 'y1_px', 'dx_px', 'dy_px', 'speed_px_per_time',
                   'vx_um_per_time', 'vy_um_per_time', 'speed_um_per_time',
                   'accepted_weight', 'fully_reverse_confirmed', 'uses_partial_support']
VECTOR_COLUMNS = ['capture_start', 'capture_end', 'x_px', 'y_px', 'dx_px', 'dy_px',
                  'snr', 'texture_sd', 'forward_qualified', 'reverse_confirmed',
                  'accepted', 'birth_zone_eligible', 'reverse_error_px', 'known_reverse_disagreement']
PREFIX = 'PARTAKER_PIV '


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows, columns):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def progress(message, percent=0):
    print(PREFIX + json.dumps(dict(message=message, percent=int(percent))), flush=True)


def settings(config, shape):
    first = int(config.get('window', 384))
    if first not in PRESETS:
        raise ValueError('Choose a 128, 256, or 384 pixel starting window.')
    windows, overlaps, bound = PRESETS[first]
    step = windows[0] - overlaps[0]
    if any((size-windows[0])//step+1 < 4 for size in shape):
        raise ValueError(f'{first}px needs at least four coarse grid nodes in each direction. '
                         'Choose a smaller starting window; the image will not be padded.')
    interval = float(config.get('interval', 1))
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError('Capture interval must be positive.')
    threshold = float(config.get('threshold', 160))
    if not math.isfinite(threshold) or not 0 <= threshold <= 255:
        raise ValueError('Bright threshold must be between 0 and 255.')
    return windows, overlaps, bound


def normalize_frames(frames):
    """One deterministic conversion per position/channel; 8-bit inputs stay exact."""
    if any(a.ndim != 2 or a.shape != frames[0].shape for a in frames):
        raise ValueError('Grayscale frames must be two-dimensional and have identical shapes.')
    if any(not np.isfinite(a).all() for a in frames):
        raise ValueError('Images contain non-finite intensities.')
    if all(a.dtype == np.uint8 for a in frames):
        return [a.copy() for a in frames], dict(method='identity_uint8', low=0., high=255.)
    low = float(min(a.min() for a in frames)); high = float(max(a.max() for a in frames))
    converted = [np.clip((a.astype(float)-low)/max(high-low, 1e-12)*255, 0, 255).astype(np.uint8)
                 for a in frames]
    return converted, dict(method='fixed_run_min_max', low=low, high=high)


def window_support(mask, xs, ys, width=32):
    import cv2
    x, y = np.meshgrid(xs.astype(int), ys.astype(int)); half = width//2
    integral = cv2.integral(mask.astype(np.float64))
    return (integral[y+half, x+half] - integral[y-half, x+half]
            - integral[y+half, x-half] + integral[y-half, x-half]) >= width*width-.01


def sample_grid(field, positions, key='accepted', minimum=.75):
    """Renormalize accepted bilinear weights. No nearest fallback or extrapolation."""
    pos = np.asarray(positions, float).reshape(-1, 2)
    xs, ys = field['xs'], field['ys']
    gx = (pos[:, 0]-xs[0])/(xs[1]-xs[0]); gy = (pos[:, 1]-ys[0])/(ys[1]-ys[0])
    inside = np.isfinite(pos).all(axis=1) & (gx >= 0) & (gy >= 0) & (gx <= len(xs)-1) & (gy <= len(ys)-1)
    ix = np.clip(np.floor(np.nan_to_num(gx)).astype(int), 0, len(xs)-2)
    iy = np.clip(np.floor(np.nan_to_num(gy)).astype(int), 0, len(ys)-2)
    fx, fy = gx-ix, gy-iy
    mass = np.zeros(len(pos)); total = np.zeros((len(pos), 2)); full = inside.copy(); confirmed = inside.copy()
    for dx, dy, weight in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                           (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
        use = weight > 1e-9
        good = field[key][iy+dy, ix+dx]
        full &= ~use | good
        confirmed &= ~use | field.get('confirmed', field[key])[iy+dy, ix+dx]
        ww = np.where(inside & use & good, weight, 0)
        mass += ww
        values = np.column_stack((field['u'][iy+dy, ix+dx], field['v'][iy+dy, ix+dx]))
        total += np.nan_to_num(values)*ww[:, None]
    ok = inside & (mass >= minimum-1e-10)
    if minimum >= 1:
        ok &= full
    move = np.full(total.shape, np.nan)
    move[ok] = total[ok]/mass[ok, None]
    return move, ok, mass, full & confirmed


def estimate_pair(a, b, config):
    """Serial worker only: SNR capture wraps the unmodified OpenPIV correlation."""
    from openpiv import windef, validation, filters
    from scipy.ndimage import distance_transform_edt
    from unittest.mock import patch
    import cv2
    windows, overlaps, bound = settings(config, a.shape)
    s = windef.PIVSettings()
    s.windowsizes = windows; s.overlap = overlaps; s.num_iterations = len(windows)
    s.correlation_method = 'linear'; s.normalized_correlation = True; s.use_vectorized = True
    s.deformation_method = 'symmetric'; s.interpolation_order = 1
    s.sig2noise_method = 'peak2peak'; s.sig2noise_mask = 2
    s.sig2noise_validate = True; s.sig2noise_threshold = 1.2
    s.min_max_u_disp = (-bound, bound); s.min_max_v_disp = (-bound, bound)
    s.std_threshold = 10; s.median_threshold = 5; s.median_size = 1; s.smoothn = False

    def fill(u, v, flags):
        u, v = filters.replace_outliers(np.ma.array(u, mask=np.zeros_like(flags)),
            np.ma.array(v, mask=np.zeros_like(flags)), flags, method=s.filter_method,
            max_iter=s.max_filter_iteration, kernel_size=s.filter_kernel_size)
        uu, vv = np.ma.filled(u, np.nan), np.ma.filled(v, np.nan)
        missing = ~np.isfinite(uu) | ~np.isfinite(vv)
        if missing.all():
            raise ValueError('No validated deformation predictor')
        if missing.any():
            nearest = distance_transform_edt(missing, return_distances=False, return_indices=True)
            uu[missing] = uu[tuple(nearest[:, missing])]; vv[missing] = vv[tuple(nearest[:, missing])]
        return np.ma.array(uu), np.ma.array(vv)

    def raw(first, second):
        captured = []; original = windef.extended_search_area_piv
        def correlation(*args, **kwargs):
            result = original(*args, **kwargs); captured.append(np.array(result[2])); return result
        with patch.object(windef, 'extended_search_area_piv', side_effect=correlation):
            x, y, u, v, snr = windef.first_pass(first.astype(np.float32), second.astype(np.float32), s)
            flags = validation.typical_validation(np.ma.array(u), np.ma.array(v), snr, s)
            u, v = fill(u, v, flags)
            for iteration in range(1, len(windows)):
                x, y, u, v, _, flags = windef.multipass_img_deform(
                    first.astype(np.float32), second.astype(np.float32), iteration, x, y, u, v, s)
                if iteration < len(windows)-1:
                    u, v = fill(u, v, flags)
        u, v = np.ma.filled(u, np.nan), np.ma.filled(v, np.nan)
        xx, yy = x.astype(int), y.astype(int); half = 16
        def sums(image):
            integral = cv2.integral(image.astype(float))
            return (integral[yy+half, xx+half]-integral[yy-half, xx+half]
                    -integral[yy+half, xx-half]+integral[yy-half, xx-half])
        std = np.sqrt(np.maximum(0, sums(first.astype(float)**2)/1024-(sums(first)/1024)**2))
        valid_region = np.zeros(first.shape, bool); valid_region[15:-15, 15:-15] = True
        base = (~np.asarray(flags, bool) & np.isfinite(u) & np.isfinite(v) & np.isfinite(captured[-1])
                & (np.abs(u) < bound) & (np.abs(v) < bound) & (std >= 8) & (captured[-1] >= 1.2)
                & window_support(valid_region, x[0], y[:, 0]))
        return dict(xs=x[0], ys=y[:, 0], u=u, v=v, snr=captured[-1], texture_sd=std, base=base)
    with contextlib.redirect_stdout(io.StringIO()):
        forward = raw(a, b); reverse = raw(b, a)
    xx, yy = np.meshgrid(forward['xs'], forward['ys'])
    delta = np.column_stack((forward['u'].ravel(), forward['v'].ravel()))
    positions = np.column_stack((xx.ravel(), yy.ravel()))+delta
    back, ok, _, _ = sample_grid(reverse, positions, key='base', minimum=1)
    error = np.linalg.norm(delta+back, axis=1).reshape(xx.shape)
    forward['reverse_error'] = error
    forward['confirmed'] = forward['base'] & ok.reshape(xx.shape) & (error <= 2)
    forward['disagreement'] = forward['base'] & np.isfinite(error) & (error > 2)
    forward['accepted'] = forward['base'] & ~forward['disagreement']
    return forward


def inside(mask, point):
    x, y = point
    return bool(np.isfinite(point).all() and 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]
                and mask[min(int(round(y)), mask.shape[0]-1), min(int(round(x)), mask.shape[1]-1)])


def segment_inside(mask, a, b):
    count = max(2, int(np.ceil(np.linalg.norm(b-a)*2))+1)
    return all(inside(mask, point) for point in np.linspace(a, b, count))


def integrate_paths(fields, colonies, seeds, times, config):
    """Seeds govern births only. Existing paths may leave seeds, never the colony."""
    from scipy.spatial import cKDTree
    paths = {}; active = []; segments = []; births = []; next_id = 1
    interval = float(config['interval']); sx, sy = config.get('pixel_um', (None, None))
    for k, f in enumerate(fields):
        if f is None or times[k+1]-times[k] != 1:
            active = []; continue
        if active:
            pos = np.array([paths[i] for i in active])
            _, ok, _, _ = sample_grid(f, pos)
            active = [i for i, p, good in zip(active, pos, ok) if good and inside(colonies[k], p)]
        xx, yy = np.meshgrid(f['xs'], f['ys'])
        candidates = np.column_stack((xx[f['accepted']], yy[f['accepted']]))
        candidates = np.array([p for p in candidates if inside(seeds[k], p)]).reshape(-1, 2)
        if active and len(candidates):
            distance, _ = cKDTree([paths[i] for i in active]).query(candidates)
            candidates = candidates[distance >= 6.4]
        for point in candidates:
            paths[next_id] = point.copy(); active.append(next_id)
            births.append(dict(path_id=next_id, capture=times[k], time=(times[k]-times[0])*interval,
                               x_px=float(point[0]), y_px=float(point[1])))
            next_id += 1
        if not active:
            continue
        positions = np.array([paths[i] for i in active]); delta, ok, weights, confirmed = sample_grid(f, positions)
        surviving = []
        for i, p, move, good, weight, trust in zip(active, positions, delta, ok, weights, confirmed):
            end = p+move
            if not good or not inside(colonies[k], p) or not inside(colonies[k+1], end):
                continue
            if not segment_inside(colonies[k] | colonies[k+1], p, end):
                continue
            row = dict(path_id=i, time_start=(times[k]-times[0])*interval,
                time_end=(times[k+1]-times[0])*interval, capture_start=times[k], capture_end=times[k+1],
                x0_px=float(p[0]), y0_px=float(p[1]), x1_px=float(end[0]), y1_px=float(end[1]),
                dx_px=float(move[0]), dy_px=float(move[1]), speed_px_per_time=float(np.linalg.norm(move)/interval),
                vx_um_per_time='', vy_um_per_time='', speed_um_per_time='', accepted_weight=float(weight),
                fully_reverse_confirmed=bool(trust), uses_partial_support=bool(weight < 1-1e-9))
            if sx and sy:
                row.update(vx_um_per_time=float(move[0]*sx/interval), vy_um_per_time=float(move[1]*sy/interval),
                           speed_um_per_time=float(np.hypot(move[0]*sx, move[1]*sy)/interval))
            segments.append(row); paths[i] = end; surviving.append(i)
        active = surviving
    return segments, births


def current_segments(segments, capture_end):
    return [s for s in segments if s['capture_end'] == capture_end]


def bbox(masks, padding=5):
    yy, xx = np.where(np.logical_or.reduce(masks))
    if not len(xx):
        return (0, 0, masks[0].shape[1], masks[0].shape[0])
    return (max(0, int(xx.min())-padding), max(0, int(yy.min())-padding),
            min(masks[0].shape[1], int(xx.max())+padding+1),
            min(masks[0].shape[0], int(yy.max())+padding+1))


NOTEBOOK_ZOOM_XYWH = (220, 180, 360, 280)


def resolve_zoom(colonies, seeds, config, restricted=False):
    """Notebook coordinates refer to the analysis image, not its display crop."""
    h, w = colonies[0].shape
    if config.get('notebook_zoom', True):
        x, y, width, height = NOTEBOOK_ZOOM_XYWH
        bounds = (max(0, x), max(0, y), min(w, x+width), min(h, y+height))
        x0, y0, x1, y1 = bounds
        if x1 > x0 and y1 > y0 and any(m[y0:y1, x0:x1].any() for m in colonies):
            return bounds, 'notebook preset (clipped to image)'
        reason = 'Notebook box misses this colony; automatic zoom used'
    else:
        reason = 'Automatic zoom (notebook preset disabled)'
    bounds = bbox(seeds if restricted and any(m.any() for m in seeds) else colonies, padding=0)
    if not restricted:
        x0, y0, x1, y1 = bounds
        bounds = (x0+(x1-x0)*.2, y0+(y1-y0)*.2, x1-(x1-x0)*.2, y1-(y1-y0)*.2)
    return bounds, reason


def local_density(values, mask, width=32):
    """ROI-normalized local average; unselected pixels never count as zeros."""
    import cv2
    width = int(width)
    if not 1 <= width <= 256:raise ValueError('Density neighborhood must be 1–256 pixels.')
    weights = cv2.boxFilter(mask.astype(float), -1, (width, width), normalize=False, borderType=cv2.BORDER_CONSTANT)
    sums = cv2.boxFilter(np.where(mask, values, 0.).astype(float), -1, (width, width), normalize=False, borderType=cv2.BORDER_CONSTANT)
    result = np.full(mask.shape, np.nan)
    np.divide(sums, weights, out=result, where=mask & (weights > 0))
    return np.clip(result, 0., 1.)


def density_summary(gray, bright, mask):
    from scipy.ndimage import label
    from scipy.spatial import cKDTree
    area = int(mask.sum())
    row = dict(roi_area_px=area, mean_gray_0_255=None, darkness_proxy=None, bright_area_fraction=None,
               bright_patch_count=0, mean_bright_patch_area_px=None, nearest_patch_centroid_spacing_px=None)
    if not area:return row
    row.update(mean_gray_0_255=float(gray[mask].mean()), darkness_proxy=float((1.-gray[mask]/255.).mean()),
               bright_area_fraction=float((bright[mask] > 0).mean()))
    labels, count = label((bright > 0) & mask, structure=np.ones((3, 3)))
    row['bright_patch_count'] = int(count)
    if count:
        yy, xx = np.nonzero(labels); ids = labels[yy, xx]
        sizes = np.bincount(ids, minlength=count+1)[1:]
        row['mean_bright_patch_area_px'] = float(sizes.mean())
        if count > 1:
            centers = np.column_stack((np.bincount(ids, weights=xx, minlength=count+1)[1:]/sizes,
                                       np.bincount(ids, weights=yy, minlength=count+1)[1:]/sizes))
            row['nearest_patch_centroid_spacing_px'] = float(cKDTree(centers).query(centers, k=2)[0][:, 1].mean())
    return row


def export_density(folder, frames, bright, masks, times, config):
    """Descriptive intensity / bright-space summaries, independent of PIV acceptance."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image
    folder.mkdir(parents=True, exist_ok=True)
    width = int(config.get('density_width', 32)); crop = bbox(masks)
    elapsed = [(t-times[0])*config['interval'] for t in times]
    rows = [dict(capture=t, elapsed_time=et, **density_summary(gray, binary, mask))
            for t, et, gray, binary, mask in zip(times, elapsed, frames, bright, masks)]
    write_csv(folder/'density_over_time.csv', rows, list(rows[0]))
    def save_movie(images, name):
        images[0].save(folder/name, save_all=True, append_images=images[1:], duration=[250]*(len(images)-1)+[1000], loop=0, disposal=2)
    for name, title, values in (
        ('raw', 'Raw image: local darkness proxy', [1.-a/255. for a in frames]),
        ('segmented', 'Segmented image: local bright-area fraction', [(a > 0).astype(float) for a in bright])):
        movie = []
        for k, (value, mask) in enumerate(zip(values, masks)):
            fig = Figure(figsize=(7, 5), layout='constrained'); canvas = FigureCanvasAgg(fig); ax = fig.subplots()
            ax.set_facecolor('#e5e5e5')
            im = ax.imshow(local_density(value, mask, width), cmap='magma', vmin=0, vmax=1, interpolation='nearest')
            x0,y0,x1,y1=crop; ax.set(xlim=(x0,x1),ylim=(y1,y0),xlabel='x (px)',ylabel='y (px)')
            fig.colorbar(im, ax=ax).set_label('Local fraction / proxy (0–1; fixed across captures)')
            ax.set_title(f'{title}\nCapture {times[k]} | t={elapsed[k]:g} {config["time_unit"]}')
            fig.supxlabel(f'{width} × {width} px neighborhood; gray = outside ROI. Not a calibrated cell density.',fontsize=8)
            canvas.draw(); movie.append(Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3].copy())); fig.clear()
        save_movie(movie, name+'_density_heatmap.gif')
    movie = []
    keys = [('darkness_proxy','Mean darkness /\nbright-area fraction'),
            ('bright_patch_count','Bright patch count'),
            ('nearest_patch_centroid_spacing_px','Nearest patch centroid\nspacing (px)')]
    for k in range(len(times)):
        fig = Figure(figsize=(8, 7), layout='constrained'); canvas = FigureCanvasAgg(fig); axes = fig.subplots(3, 1, sharex=True)
        for ax, (key, title) in zip(axes, keys):
            vals = np.array([r[key] if r[key] is not None else np.nan for r in rows], float)
            ax.plot(elapsed[:k+1], vals[:k+1], '-o', markersize=3, label='Raw darkness' if key=='darkness_proxy' else None)
            if key == 'darkness_proxy':
                vals2 = [r['bright_area_fraction'] if r['bright_area_fraction'] is not None else np.nan for r in rows]
                ax.plot(elapsed[:k+1], vals2[:k+1], '-o', markersize=3, label='Segmented bright area'); ax.set_ylim(0, 1); ax.legend(fontsize=8)
            else:
                finite = vals[np.isfinite(vals)]; maximum = float(finite.max()) if len(finite) else 1.
                ax.set_ylim(0, max(1., maximum*1.12))
            ax.set_ylabel(title, fontsize=8); ax.grid(alpha=.2); ax.axvline(elapsed[k], color='gray', lw=.8)
            ax.set_xlim(elapsed[0], elapsed[-1] if elapsed[-1] > elapsed[0] else elapsed[0]+1)
            if not any(m.any() for m in masks):ax.text(.5,.5,'No selected pixels',transform=ax.transAxes,ha='center')
        axes[-1].set_xlabel('Elapsed capture time ('+config['time_unit']+')')
        fig.suptitle(f'Density / pattern summaries | capture {times[k]}')
        fig.supxlabel('Bright patches use 8-connectivity; no small-patch removal. Spacing is centroid-to-centroid, not cell gaps.',fontsize=8)
        canvas.draw(); movie.append(Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3].copy()))
        if k == len(times)-1:fig.savefig(folder/'density_over_time.png', dpi=150)
        fig.clear()
    save_movie(movie, 'density_over_time.gif')
    write_json(folder/'metadata.json', dict(neighborhood_px=width, captures=times, elapsed_time=elapsed,
        raw_definition='Local mean of 1-gray/255 using the fixed run intensity conversion; no segmentation or tracking blur.',
        segmented_definition='Local fraction of bright mask pixels; sigma1 threshold mask, no component removal.',
        roi_normalization='Neighborhood numerator divided by selected pixel count; values outside ROI masked.',
        spacing_definition='Mean nearest 8-connected bright patch centroid distance in pixels. Undefined for fewer than two patches.',
        interpretation='Descriptive image proxies, not calibrated cell density, cell counts, polymer concentration, or material transport. Illumination and changing ROI can change values.',
        fixed_heatmap_limits=[0,1]))


def render(folder, frames, colonies, seeds, segments, births, times, config, variant, restricted):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle
    from PIL import Image
    import cv2
    crop = bbox(colonies); zoom, zoom_reason = resolve_zoom(colonies, seeds, config, restricted)
    # Keep the entire highlighted zoom box visible in the overview.
    h, w = colonies[0].shape
    crop = (max(0,min(crop[0],zoom[0]-5)), max(0,min(crop[1],zoom[1]-5)),
            min(w,max(crop[2],zoom[2]+5)), min(h,max(crop[3],zoom[3]+5)))
    duration = (times[-1]-times[0])*float(config['interval'])
    norm = Normalize(0, max(duration, 1e-6))
    cmap = LinearSegmentedColormap.from_list('capture', ['#0620ff', '#00ce8e'])
    def axes_setup(ax, bounds):
        x0, y0, x1, y1 = bounds; ax.set(xlim=(x0, x1), ylim=(y1, y0), aspect='equal')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
    def outlines(ax, k):
        for mask, color in ((colonies[k], '#59bd2e'), (seeds[k], '#e39b22')):
            if mask.any():
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    q = c[:, 0]; ax.plot(q[:, 0], q[:, 1], color=color, linewidth=.55, alpha=.7)
    def draw(ax, chosen, arrows=False):
        collection = LineCollection([[(s['x0_px'], s['y0_px']), (s['x1_px'], s['y1_px'])] for s in chosen],
            cmap=cmap, norm=norm, linewidths=.65,
            linestyles=['solid' if s['fully_reverse_confirmed'] else (0, (1, 1)) for s in chosen])
        values = np.array([(s['time_start']+s['time_end'])/2 for s in chosen])
        collection.set_array(values); ax.add_collection(collection)
        if arrows and chosen:
            # Arrowheads on the last part of each segment retain dotted confidence shafts.
            ax.quiver([s['x1_px']-.15*s['dx_px'] for s in chosen], [s['y1_px']-.15*s['dy_px'] for s in chosen],
                [.15*s['dx_px'] for s in chosen], [.15*s['dy_px'] for s in chosen], values,
                cmap=cmap, norm=norm, angles='xy', scale_units='xy', scale=1, width=.0018, minlength=0)
        return collection
    def scale_bar(ax, bounds):
        x0,y0,x1,y1=bounds
        length=max(1,int(min(100 if x1-x0>600 else 50,(x1-x0)/4)))
        right=x1-(x1-x0)*.04;bottom=y1-(y1-y0)*.08
        ax.plot([right-length,right],[bottom,bottom],color='black',linewidth=2)
        ax.text(right-length/2,bottom+(y1-y0)*.045,f'{length} px',ha='center',fontsize=7)
    title = variant.replace('_', ' ').title()
    fig = Figure(figsize=(12, 6), layout='constrained'); FigureCanvasAgg(fig); axes = fig.subplots(1, 2)
    artist = draw(axes[0], segments); draw(axes[1], segments)
    for ax, bounds, name in zip(axes, (crop, zoom), ('Microcolony', 'Zoomed field')):
        axes_setup(ax, bounds); ax.set_title(name)
        if births:
            ax.scatter([b['x_px'] for b in births], [b['y_px'] for b in births],
                       c=[b['time'] for b in births], cmap=cmap, norm=norm, s=.6, linewidths=0)
        if not segments:ax.text(.5, .5, 'No supported motion segments', transform=ax.transAxes, ha='center')
        scale_bar(ax,bounds)
    axes[0].add_patch(Rectangle((zoom[0], zoom[1]), zoom[2]-zoom[0], zoom[3]-zoom[1], fill=False, ec='#59bd2e', lw=1))
    fig.colorbar(artist, ax=axes, orientation='horizontal', fraction=.05).set_label('Elapsed capture time ('+config['time_unit']+')')
    fig.suptitle(title+' | '+str(config['window'])+'px starting window')
    fig.supxlabel('Solid: confirmed. Dotted: partial / unconfirmed. '+zoom_reason+'. Coordinates in pixels.', fontsize=8)
    fig.savefig(folder/'motion_field.png', dpi=150)
    fig.clear()
    for accumulating in (True, False):
        movie = []
        for k in range(1, len(times)):
            fig = Figure(figsize=(10, 4.8), layout='constrained'); canvas = FigureCanvasAgg(fig); axes = fig.subplots(1, 2)
            axes[0].imshow(frames[k], cmap='gray', vmin=0, vmax=255); outlines(axes[0], k)
            axes[0].set_title('Current image / selected regions'); axes_setup(axes[0], crop)
            chosen = ([s for s in segments if s['capture_end'] <= times[k]] if accumulating else current_segments(segments, times[k]))
            artist = draw(axes[1], chosen, arrows=not accumulating); axes_setup(axes[1], crop)
            scale_bar(axes[1],crop)
            axes[1].set_title('Accumulating paths' if accumulating else 'Current interval only')
            if not chosen:axes[1].text(.5, .5, 'No supported motion', transform=axes[1].transAxes, ha='center')
            fig.colorbar(artist, ax=axes, orientation='horizontal', fraction=.05).set_label('Elapsed capture time ('+config['time_unit']+')')
            fig.suptitle(f'{title} | capture {times[k-1]} → {times[k]} | t={(times[k]-times[0])*config["interval"]:g} {config["time_unit"]}')
            fig.supxlabel('Solid: confirmed local support | dotted: exploratory | no displacement magnification', fontsize=8)
            canvas.draw(); movie.append(Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3].copy())); fig.clear()
        name = 'motion_accumulating.gif' if accumulating else 'motion_current_frame.gif'
        movie[0].save(folder/name, save_all=True, append_images=movie[1:], duration=[250]*(len(movie)-1)+[1000], loop=0, disposal=2)
    for mode in ('source', 'accumulating', 'current_frame'):
        movie = []
        for k in range(1, len(times)):
            fig = Figure(figsize=(7, 5), layout='constrained'); canvas = FigureCanvasAgg(fig); ax = fig.subplots()
            if mode == 'source':
                ax.imshow(frames[k], cmap='gray', vmin=0, vmax=255); outlines(ax, k)
            else:
                chosen = [s for s in segments if s['capture_end'] <= times[k]] if mode == 'accumulating' else current_segments(segments, times[k])
                artist = draw(ax, chosen, arrows=mode == 'current_frame')
                fig.colorbar(artist, ax=ax, orientation='horizontal', fraction=.05).set_label('Elapsed capture time ('+config['time_unit']+')')
                if not chosen:ax.text(.5,.5,'No supported motion',transform=ax.transAxes,ha='center')
            axes_setup(ax, crop); scale_bar(ax,crop)
            ax.set_title(f'{title} | {mode.replace("_", " ")}\nCapture {times[k-1]} → {times[k]} | t={(times[k]-times[0])*config["interval"]:g} {config["time_unit"]}')
            canvas.draw(); movie.append(Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3].copy())); fig.clear()
        name = 'source_images.gif' if mode == 'source' else 'motion_'+mode+'_only.gif'
        movie[0].save(folder/name,save_all=True,append_images=movie[1:],duration=[250]*(len(movie)-1)+[1000],loop=0,disposal=2)
    return crop


def run_manifest(manifest_path):
    import cv2
    import PIL
    from PIL import Image
    import scipy
    import importlib.metadata
    path = Path(manifest_path); manifest = json.loads(path.read_text()); root = path.parent
    config = manifest['config']; groups = manifest['groups']; total = max(1, sum(2*(len(g['frames'])-1)+4*len({c['track_id'] for f in g['frames'] for c in f['colonies']}) for g in groups)); completed = 0
    provenance = dict(config=config, coordinate_system='registered analysis-image pixels before display crop',
        original_coordinate_transform='original_xy = analysis_xy + crop_origin_xy - registration_shift_xy',
        versions=dict(numpy=np.__version__, scipy=scipy.__version__, openpiv=importlib.metadata.version('openpiv'), pillow=PIL.__version__),
        groups=[], status='running')
    write_json(root/'metadata.json', provenance)
    for group in groups:
        times = [int(f['time']) for f in group['frames']]
        if len(times) < 2 or times != sorted(set(times)):raise ValueError('Choose at least two distinct chronological captures.')
        arrays = [np.load(root/f['file'], allow_pickle=False) for f in group['frames']]
        frames, scale = normalize_frames(arrays); settings(config, frames[0].shape)
        masks = []
        for frame in group['frames']:
            with np.load(root/frame['masks'], allow_pickle=False) as saved:
                masks.append({name:saved[name] for name in saved.files})
        bright = [(cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.) >= config['threshold']).astype(np.uint8)*255 for a in frames]
        fields = {}; failures = []
        for kind, images in (('gray', frames), ('bright', bright)):
            fields[kind] = []
            for k in range(len(times)-1):
                f = None
                if times[k+1]-times[k] != 1:
                    failures.append(dict(input=kind, capture_start=times[k], capture_end=times[k+1], reason='Missing intermediate capture; paths not bridged'))
                else:
                    try:f = estimate_pair(images[k], images[k+1], config)
                    except ValueError as error:
                        if str(error) not in ('No validated deformation predictor', 'Something happened in the validation'):raise
                        failures.append(dict(input=kind, capture_start=times[k], capture_end=times[k+1], reason=str(error)))
                fields[kind].append(f); completed += 1; progress(f'P={group["position"]} C={group["channel"]}: {kind}, pair {k+1}/{len(times)-1}', 20+int(75*completed/total))
        ids = sorted({int(c['track_id']) for f in group['frames'] for c in f['colonies']})
        for track in ids:
            folder = root/f'position_{group["position"]:03d}_channel_{group["channel"]}_colony_{track}'
            folder.mkdir(); mask_dir = folder/'masks'; mask_dir.mkdir()
            colonies = []; seeds = []
            for fm in masks:
                colony = fm.get(f'colony_{track}', np.zeros(frames[0].shape, bool)).astype(bool)
                seed = fm.get(f'inner_{track}', np.zeros(frames[0].shape, bool)).astype(bool) & colony
                colonies.append(colony); seeds.append(seed)
            for name, values in (('colony_masks.tif', colonies), ('inner_masks.tif', seeds), ('bright_masks.tif', bright)):
                ims = [Image.fromarray(a.astype(np.uint8)*(255 if a.dtype == bool else 1)) for a in values]
                ims[0].save(mask_dir/name, save_all=True, append_images=ims[1:], compression='tiff_deflate')
            if config.get('include_density', True):
                export_density(folder/'density'/'whole_colony', frames, bright, colonies, times, config)
                if config.get('include_inner', True):
                    export_density(folder/'density'/'inside_roi', frames, bright, seeds, times, config)
            for variant, kind, restricted in VARIANTS:
                if restricted and not config.get('include_inner', True):continue
                work = folder/variant; work.mkdir()
                ff = []
                for k, original in enumerate(fields[kind]):
                    if original is None:ff.append(None); continue
                    f = dict(original); eligible = window_support(colonies[k], f['xs'], f['ys'])
                    f['accepted'] = original['accepted'] & eligible; f['confirmed'] = original['confirmed'] & eligible
                    ff.append(f)
                segments, births = integrate_paths(ff, colonies, seeds if restricted else colonies, times, config)
                vector_rows = []
                for k, f in enumerate(ff):
                    if f is None:continue
                    xx, yy = np.meshgrid(f['xs'], f['ys'])
                    for i, j in zip(*np.where(window_support(colonies[k], f['xs'], f['ys']))):
                        vector_rows.append(dict(capture_start=times[k], capture_end=times[k+1], x_px=float(xx[i,j]), y_px=float(yy[i,j]),
                            dx_px=float(f['u'][i,j]), dy_px=float(f['v'][i,j]), snr=float(f['snr'][i,j]), texture_sd=float(f['texture_sd'][i,j]),
                            forward_qualified=bool(f['base'][i,j]), reverse_confirmed=bool(f['confirmed'][i,j]), accepted=bool(f['accepted'][i,j]),
                            birth_zone_eligible=inside(seeds[k] if restricted else colonies[k],(xx[i,j],yy[i,j])),
                            reverse_error_px=float(f['reverse_error'][i,j]), known_reverse_disagreement=bool(f['disagreement'][i,j])))
                write_csv(work/'vectors.csv', vector_rows, VECTOR_COLUMNS); write_csv(work/'path_segments.csv', segments, SEGMENT_COLUMNS)
                write_csv(work/'path_births.csv', births, ['path_id','capture','time','x_px','y_px'])
                crop = render(work, frames if kind == 'gray' else bright, colonies, seeds, segments, births, times, config, variant, restricted)
                write_json(work/'metadata.json', dict(variant=variant, birth_zone='inner' if restricted else 'colony',
                    status='complete' if segments else 'no_supported_segments', crop_xyxy=list(crop), zoom_xyxy=list(resolve_zoom(colonies,seeds,config,restricted)[0]),
                    zoom_selection=resolve_zoom(colonies,seeds,config,restricted)[1], segments=len(segments), births=len(births),
                    confirmed_segments=sum(s['fully_reverse_confirmed'] for s in segments),
                    note='Inner masks govern births only. Current-frame GIF includes only that interval, not previous trails.'))
                completed += 1; progress(f'Saved {variant} for colony {track}', 20+int(75*completed/total))
            write_json(folder/'metadata.json', dict(position=group['position'], channel=group['channel'], colony_track_id=track,
                captures=times, intensity_conversion=scale, failures=failures, spatial_context='Full-frame PIV; final32px window must fit colony; path endpoints and connecting segment constrained to colony.'))
        provenance['groups'].append(dict(position=group['position'], channel=group['channel'], captures=times,
            intensity_conversion=scale, input_hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays], failures=failures))
    provenance['status'] = 'complete'; write_json(root/'metadata.json', provenance); progress('Motion export complete', 100)


def create_export_job(parent, image_data, colonies_by_frame, root, config, python, on_progress, on_done):
    """Qt coordinator: prepare one frame per event-loop tick, then use QProcess."""
    from PySide6.QtCore import QObject, QProcess, QTimer
    import copy
    import cv2
    colonies_by_frame = copy.deepcopy(colonies_by_frame)
    class Job(QObject):
        def __init__(self):
            super().__init__(parent)
            self.finished = False; self.cancelled = False; self.process = None; self.buffer = ''; self.errors = ''
            self.kill_timer = QTimer(self); self.kill_timer.setSingleShot(True); self.kill_timer.timeout.connect(self.kill_active)
            self.keys = sorted(colonies_by_frame); self.index = 0; self.groups = {}
            self.crop = copy.deepcopy(image_data.crop_coordinates)
            self.offsets = copy.deepcopy(image_data.registration_offsets)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
            self.folder = Path(root)/stamp
            self.preflight = QProcess(self); self.preflight.setProgram(str(python)); self.preflight.setArguments([str(Path(__file__).resolve()), '--check'])
            from PySide6.QtCore import QProcessEnvironment
            env = QProcessEnvironment.systemEnvironment(); env.remove('PYTHONPATH'); env.remove('PYTHONHOME')
            self.preflight.setProcessEnvironment(env)
            self.preflight.finished.connect(self.checked)
            self.preflight.errorOccurred.connect(lambda _: self.finish(False, 'Cannot start the PIV Python. Select the isolated runtime executable.'))
            self.preflight.start()
        def checked(self, code, status):
            if self.finished:return
            if self.cancelled:return self.finish(False, 'Cancelled')
            if code or status != QProcess.NormalExit:
                return self.finish(False, 'PIV runtime is missing dependencies. Install requirements-piv.txt there.\n'+bytes(self.preflight.readAllStandardError()).decode(errors='replace')[-1500:])
            self.folder.mkdir(parents=True); (self.folder/'inputs').mkdir()
            write_json(self.folder/'status.json', dict(status='preparing'))
            QTimer.singleShot(0, self.prepare_next)
        def prepare_next(self):
            if self.finished:return
            try:
                if self.cancelled:return self.finish(False, 'Cancelled')
                if not np.array_equal(self.crop, image_data.crop_coordinates) or not np.array_equal(self.offsets, image_data.registration_offsets):
                    raise ValueError('Image crop or registration changed during preparation. Export again after finishing those changes.')
                if self.index == len(self.keys):return self.launch()
                p, t, c = self.keys[self.index]
                raw = np.array(image_data.get(t, p, c), copy=True)
                if raw.ndim != 2:raise ValueError('Choose a grayscale channel for motion export.')
                settings(config, raw.shape)
                data = {}; entries = []; seen = set()
                for colony in colonies_by_frame[(p,t,c)]:
                    track = colony.get('track_id')
                    if track is None:raise ValueError('Accept the verified colony time series before exporting motion.')
                    track = int(track)
                    if track in seen:raise ValueError('Colony identities are duplicated within a frame.')
                    seen.add(track)
                    mask = colony.get('mask')
                    if mask is None:
                        mask = np.zeros(raw.shape, np.uint8)
                        contour = colony.get('contour', colony.get('polygon'))
                        if contour is None:raise ValueError('A selected colony has no polygon or mask.')
                        cv2.fillPoly(mask, [np.asarray(contour, np.int32).reshape(-1,1,2)], 1)
                    mask = np.asarray(mask, bool).copy()
                    if mask.shape != raw.shape:raise ValueError('Colony masks and current image differ in shape. Re-select after changing the crop.')
                    mask[:15] = False; mask[-15:] = False; mask[:,:15] = False; mask[:,-15:] = False
                    inner = np.asarray(colony.get('inner_mask', np.zeros(raw.shape)), bool)
                    if inner.shape != raw.shape:raise ValueError('Inner-region mask does not match the image.')
                    data[f'colony_{track}'] = mask; data[f'inner_{track}'] = inner & mask
                    entries.append(dict(track_id=track))
                name=f'inputs/p{p}_t{t}_c{c}'
                np.save(self.folder/(name+'.npy'), raw, allow_pickle=False)
                np.savez_compressed(self.folder/(name+'_masks.npz'), **data)
                self.groups.setdefault((p,c), []).append(dict(time=int(t), file=name+'.npy', masks=name+'_masks.npz', colonies=entries))
                self.index += 1; on_progress(f'Preparing capture {self.index}/{len(self.keys)}', int(20*self.index/len(self.keys)))
                QTimer.singleShot(0, self.prepare_next)
            except Exception as error:self.finish(False, str(error))
        def launch(self):
            try:
                if config.get('include_inner', True):
                    found = False
                    for file in (self.folder/'inputs').glob('*_masks.npz'):
                        with np.load(file, allow_pickle=False) as masks:
                            found |= any(masks[k].any() for k in masks.files if k.startswith('inner_'))
                    if not found:raise ValueError('No inner regions selected. Accept Inside ROI regions, or uncheck inner-region outputs.')
                groups = [dict(position=int(p), channel=int(c), frames=frames) for (p,c),frames in sorted(self.groups.items())]
                if any(len(g['frames'])<2 for g in groups):raise ValueError('Every selected position/channel needs at least two captures.')
                write_json(self.folder/'manifest.json', dict(config=config, groups=groups,
                    source_files=str(image_data.image_filename), crop_xywh=self.crop,
                    registration_offsets=None if self.offsets is None else np.asarray(self.offsets).tolist()))
                self.process = QProcess(self); self.process.setProgram(str(python))
                self.process.setArguments(['-u', str(Path(__file__).resolve()), '--manifest', str(self.folder/'manifest.json')])
                from PySide6.QtCore import QProcessEnvironment
                env = QProcessEnvironment.systemEnvironment(); env.remove('PYTHONPATH'); env.remove('PYTHONHOME')
                env.insert('MPLCONFIGDIR', str(self.folder/'.plot_cache')); self.process.setProcessEnvironment(env)
                self.process.readyReadStandardOutput.connect(self.read_output)
                self.process.readyReadStandardError.connect(self.read_errors)
                self.process.errorOccurred.connect(lambda _: self.finish(False, 'Could not run the PIV worker.'))
                self.process.finished.connect(self.worker_done)
                write_json(self.folder/'status.json', dict(status='running')); self.process.start()
            except Exception as error:self.finish(False, str(error))
        def read_output(self):
            text = bytes(self.process.readAllStandardOutput()).decode(errors='replace')
            with (self.folder/'worker_stdout.log').open('a', encoding='utf-8') as stream:stream.write(text)
            self.buffer += text
            while '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n',1)
                if line.startswith(PREFIX):
                    value = json.loads(line[len(PREFIX):]); on_progress(value['message'], value['percent'])
        def read_errors(self):
            text = bytes(self.process.readAllStandardError()).decode(errors='replace'); self.errors = (self.errors+text)[-4000:]
            with (self.folder/'worker_stderr.log').open('a', encoding='utf-8') as stream:stream.write(text)
        def worker_done(self, code, status):
            self.read_output(); self.read_errors()
            if self.cancelled:return self.finish(False, 'Cancelled; partial outputs retained.')
            self.finish(code == 0 and status == QProcess.NormalExit,
                        'Motion export complete' if code == 0 and status == QProcess.NormalExit else 'Motion export failed.\n'+self.errors)
        def cancel(self):
            if self.finished:return
            self.cancelled = True
            process = self.process if self.process is not None else self.preflight
            if process.state() != QProcess.NotRunning:
                process.terminate()
                self.kill_timer.start(1500)
            else:self.finish(False, 'Cancelled')
        def kill_active(self):
            process = self.process if self.process is not None else self.preflight
            if process.state() != QProcess.NotRunning:process.kill()
        def finish(self, ok, message):
            nonlocal image_data
            if self.finished:return
            self.finished = True; self.kill_timer.stop()
            if self.cancelled:
                ok = False; message = 'Cancelled; partial outputs retained.' if self.folder.exists() else 'Cancelled'
            if self.folder.exists():write_json(self.folder/'status.json',dict(status='complete' if ok else 'cancelled' if self.cancelled else 'failed', message=message))
            on_done(ok, message, self.folder)
            colonies_by_frame.clear(); image_data = None
            self.deleteLater()
    if not colonies_by_frame:raise ValueError('Select and accept microcolonies first.')
    return Job()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--manifest'); parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        import scipy, matplotlib, cv2, PIL
        from openpiv import windef
        import importlib.metadata
        if importlib.metadata.version('openpiv') != '0.26.1':raise ValueError('This workflow requires openpiv==0.26.1.')
        if int(np.__version__.split('.')[0]) != 2:raise ValueError('The isolated PIV runtime requires NumPy 2.x.')
        print(json.dumps(dict(openpiv='0.26.1', numpy=np.__version__, scipy=scipy.__version__))); return
    if not args.manifest:parser.error('--manifest is required')
    run_manifest(args.manifest)


if __name__ == '__main__':
    try:main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
