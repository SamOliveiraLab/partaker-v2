"""Temporal evidence for colony territories and visible fragments."""
from __future__ import annotations

import collections
import json

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.special import expit, logit
from scipy.optimize import linear_sum_assignment
from skimage import measure, morphology, segmentation

STRUCTURE = np.ones((3, 3), bool)


def overlap(a, b):
    return float(np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b)))


def warp(mask, affine):
    return cv2.warpAffine(mask.astype(np.uint8), np.asarray(affine, np.float64),
                          mask.shape[::-1], flags=cv2.INTER_NEAREST).astype(bool)


def normalized_images(raw):
    fine, coarse = [], []
    for image in raw:
        image = image.astype(np.float32)
        mean = ndi.gaussian_filter(image, 15)
        variance = ndi.gaussian_filter(image * image, 15) - mean * mean
        normalized = (image - mean) / np.maximum(5, np.sqrt(np.maximum(0, variance)))
        fine.append(ndi.gaussian_filter(normalized, .7))
        coarse.append(ndi.gaussian_filter(normalized, 2.5))
    return np.asarray(fine), np.asarray(coarse)


def correlation(source, target, mask, affine):
    y, x = np.nonzero(mask)
    if len(x) < 30:
        return -1.
    x0, x1, y0, y1 = x.min(), x.max() + 1, y.min(), y.max() + 1
    yy, xx = np.indices((y1 - y0, x1 - x0), dtype=np.float32)
    xx += x0; yy += y0
    mx = (affine[0, 0] * xx + affine[0, 1] * yy + affine[0, 2]).astype(np.float32)
    my = (affine[1, 0] * xx + affine[1, 1] * yy + affine[1, 2]).astype(np.float32)
    valid = mask[y0:y1, x0:x1] & (mx >= 0) & (my >= 0) & (mx < mask.shape[1] - 1) & (my < mask.shape[0] - 1)
    a = source[y0:y1, x0:x1][valid].astype(float)
    b = cv2.remap(target, mx, my, cv2.INTER_LINEAR)[valid].astype(float)
    if len(a) < 30:
        return -1.
    a -= a.mean()
    b -= b.mean()
    return float(np.dot(a, b) / max(1e-9, np.linalg.norm(a) * np.linalg.norm(b)))


def deformation_fit(source, target, mask, tracking, fine, coarse):
    def geometry(affine):
        singular = np.linalg.svd(affine[:, :2], compute_uv=False)
        y, x = ndi.center_of_mass(mask)
        return singular.min() >= .55 and singular.max() <= 1.9 and .4 <= np.linalg.det(affine[:, :2]) <= 2.5 and np.linalg.norm(affine @ [x, y, 1] - [x, y]) <= 65
    options = []
    affine = None
    points = cv2.goodFeaturesToTrack(tracking[source], maxCorners=180, qualityLevel=.01,
        minDistance=4, mask=ndi.binary_erosion(mask, iterations=2).astype(np.uint8) * 255, blockSize=5)
    if points is not None and len(points) >= 6:
        target_points, forward, _ = cv2.calcOpticalFlowPyrLK(tracking[source], tracking[target], points, None, winSize=(31, 31), maxLevel=4, criteria=(3, 40, .01))
        if target_points is not None:
            back, reverse, _ = cv2.calcOpticalFlowPyrLK(tracking[target], tracking[source], target_points, None, winSize=(31, 31), maxLevel=4, criteria=(3, 40, .01))
            if back is not None:
                good = (forward.ravel() > 0) & (reverse.ravel() > 0) & (np.linalg.norm(back[:, 0] - points[:, 0], axis=1) < 1.5)
                start, end = points[:, 0][good], target_points[:, 0][good]
                if len(start) >= 6:
                    affine, inliers = cv2.estimateAffine2D(start, end, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=1500, confidence=.99, refineIters=10)
                    if affine is not None:
                        accepted = inliers.ravel().astype(bool)
                        hull = cv2.contourArea(cv2.convexHull(start[accepted])) if accepted.sum() >= 3 else 0
                        if accepted.sum() >= 6 and accepted.mean() >= .45 and hull / max(1, mask.sum()) >= .08 and geometry(affine):
                            options.append(affine)
    y, x = np.nonzero(mask)
    if len(x):
        x0, y0 = max(0, x.min() - 16), max(0, y.min() - 16)
        section = np.s_[y0:min(mask.shape[0], y.max() + 17), x0:min(mask.shape[1], x.max() + 17)]
        initial = np.eye(2, 3, dtype=np.float32)
        if affine is not None and geometry(affine):
            initial = affine.astype(np.float32)
            initial[:, 2] += initial[:, :2] @ [x0, y0] - [x0, y0]
        try:
            _, fit = cv2.findTransformECC(coarse[source][section], coarse[target][section], initial, cv2.MOTION_AFFINE, (3, 100, 1e-5), None, 5)
            _, reverse = cv2.findTransformECC(coarse[target][section], coarse[source][section], cv2.invertAffineTransform(fit), cv2.MOTION_AFFINE, (3, 100, 1e-5), None, 5)
            fit, reverse = fit.astype(float), reverse.astype(float)
            for matrix in (fit, reverse):
                matrix[:, 2] += [x0, y0] - matrix[:, :2] @ [x0, y0]
            controls = np.array([[x.min(), y.min()], [x.max(), y.min()], [x.max(), y.max()], [x.min(), y.max()], [x.mean(), y.mean()]])
            moved = controls @ fit[:, :2].T + fit[:, 2]
            cycle = np.max(np.linalg.norm(moved @ reverse[:, :2].T + reverse[:, 2] - controls, axis=1))
            if geometry(fit) and cycle <= 4:
                options.append(fit)
        except cv2.error:
            pass
    scored = [(correlation(fine[source], fine[target], mask, fit), correlation(coarse[source], coarse[target], mask, fit), fit) for fit in options]
    scored = [item for item in scored if item[0] >= .25 and item[1] >= .45]
    if not scored:
        return None
    _, _, fit = max(scored, key=lambda item: .4 * item[0] + .6 * item[1])
    return fit, warp(mask, fit)


def masked_search(source, target, mask, search=65):
    y, x = np.nonzero(mask)
    if len(x) < 120:
        return []
    height, width = mask.shape
    x0, x1 = max(0, x.min() - 4), min(width, x.max() + 5)
    y0, y1 = max(0, y.min() - 4), min(height, y.max() + 5)
    weights = mask[y0:y1, x0:x1].astype(np.float32)
    template = source[y0:y1, x0:x1]
    count = weights.sum()
    centered = (template - (template * weights).sum() / count) * weights
    variance = float(np.sum(centered * centered))
    if variance < 1e-6:
        return []
    sx, sy = max(0, x0 - search), max(0, y0 - search)
    area = np.ascontiguousarray(target[sy:min(height, y1 + search), sx:min(width, x1 + search)], dtype=np.float32)
    numerator = cv2.matchTemplate(area, centered, cv2.TM_CCORR)
    total = cv2.matchTemplate(area, weights, cv2.TM_CCORR)
    squared = cv2.matchTemplate(area * area, weights, cv2.TM_CCORR)
    scores = np.clip(numerator / np.sqrt(np.maximum(1e-12, variance * (squared - total * total / count))), -1, 1)
    peaks = []
    for _ in range(4):
        _, score, _, (px, py) = cv2.minMaxLoc(scores)
        if score < -1:
            break
        peaks.append(dict(dx=int(px + sx - x0), dy=int(py + sy - y0), ncc=float(score)))
        scores[max(0, py - 12):py + 13, max(0, px - 12):px + 13] = -2
    return peaks


def fragment_fit(source, target, mask, fine, coarse):
    depth = ndi.distance_transform_edt(mask)
    ys, xs = np.where(depth >= 7)
    points = []
    for index in np.argsort(depth[ys, xs])[::-1]:
        x, y = int(xs[index]), int(ys[index])
        if all((x - px) ** 2 + (y - py) ** 2 >= 64 for px, py in points):
            points.append((x, y))
        if len(points) >= 40:
            break
    pairs = []
    height, width = mask.shape
    for x, y in points:
        if min(x, y) < 11 or x + 11 >= width or y + 11 >= height:
            continue
        section = np.s_[y - 11:y + 12, x - 11:x + 12]
        x0, y0 = max(0, x - 76), max(0, y - 76)
        area = np.s_[y0:min(height, y + 77), x0:min(width, x + 77)]
        scores = cv2.matchTemplate(fine[target][area], fine[source][section], cv2.TM_CCOEFF_NORMED)
        _, score, _, (px, py) = cv2.minMaxLoc(scores)
        others = scores.copy()
        others[max(0, py - 4):py + 5, max(0, px - 4):px + 5] = -1
        if score < .55 or score - float(others.max()) < .035:
            continue
        tx, ty = x0 + px + 11, y0 + py + 11
        destination = np.s_[ty - 11:ty + 12, tx - 11:tx + 12]
        reverse = cv2.matchTemplate(fine[source][area], fine[target][destination], cv2.TM_CCOEFF_NORMED)
        _, back_score, _, (bx, by) = cv2.minMaxLoc(reverse)
        if back_score < .55 or np.hypot(x0 + bx + 11 - x, y0 + by + 11 - y) > 2:
            continue
        a, b = coarse[source][section].ravel(), coarse[target][destination].ravel()
        if np.std(a) < 1e-6 or np.std(b) < 1e-6 or np.corrcoef(a, b)[0, 1] < .6:
            continue
        pairs.append(((x, y), (tx, ty)))
    if len(pairs) < 4:
        return None
    start, end = [np.ascontiguousarray(points) for points in np.asarray(pairs, np.float32).transpose(1, 0, 2)]
    affine, inliers = cv2.estimateAffinePartial2D(start, end, method=cv2.RANSAC, ransacReprojThreshold=3.45)
    if affine is None:
        return None
    inliers = inliers.ravel().astype(bool)
    if inliers.sum() < 3 or inliers.mean() < .6:
        return None
    singular = np.linalg.svd(affine[:, :2], compute_uv=False)
    hull = cv2.contourArea(cv2.convexHull(start[inliers]))
    if hull < 52.9 or singular.min() < .55 or singular.max() > 1.9:
        return None
    visible = np.zeros(mask.shape, np.uint8)
    for x, y in end[inliers]:
        cv2.circle(visible, (round(float(x)), round(float(y))), 12, 1, -1)
    return affine, warp(mask, affine) & visible.astype(bool)


def recover_density(source, target, mask, density, fine, coarse, outer):
    valid = []
    for images in density:
        peaks = masked_search(images[source], images[target], mask)
        checked = []
        for peak in peaks:
            if peak['ncc'] < .65:
                continue
            affine = np.array([[1., 0, peak['dx']], [0, 1., peak['dy']]])
            if correlation(fine[source], fine[target], mask, affine) < .45 or correlation(coarse[source], coarse[target], mask, affine) < .65:
                continue
            moved = warp(mask, affine) & outer[target]
            reverse = masked_search(images[target], images[source], moved)
            if reverse and np.hypot(peak['dx'] + reverse[0]['dx'], peak['dy'] + reverse[0]['dy']) <= 3:
                checked.append(peak)
        if checked and (len(checked) == 1 or checked[0]['ncc'] - checked[1]['ncc'] >= .035):
            valid.append(checked[0])
    groups = [[b for b in valid if np.hypot(a['dx'] - b['dx'], a['dy'] - b['dy']) <= 3] for a in valid]
    groups = [group for group in groups if len(group) >= 2]
    if not groups:
        return None
    group = max(groups, key=lambda rows: sum(row['ncc'] for row in rows))
    dx, dy = np.median([[row['dx'], row['dy']] for row in group], axis=0)
    affine = np.array([[1., 0, round(dx)], [0, 1., round(dy)]])
    def score(matrix):
        return .7 * correlation(fine[source], fine[target], mask, matrix) + .3 * correlation(coarse[source], coarse[target], mask, matrix)
    decoys = []
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        wrong = affine.copy()
        wrong[:, 2] += max(20, 1.5 * np.sqrt(mask.sum() / np.pi)) * np.array([np.cos(angle), np.sin(angle)])
        decoys.append(score(wrong))
    if score(affine) - max(decoys) < .10:
        return None
    return affine, warp(mask, affine) & outer[target]


def stationary_fields(raw, outer, captures, sigma):
    def smooth(width):
        return np.array([ndi.gaussian_filter(g.astype(np.float32) * m, width) /
                         np.maximum(1e-6, ndi.gaussian_filter(m.astype(float), width)) for g, m in zip(raw, outer)])
    contrast = smooth(sigma) - smooth(max(22, 4 * sigma))
    median, mad, agreement = [], [], []
    for t, capture in enumerate(captures):
        indices = [k for k in range(max(0, t - 2), min(len(raw), t + 3)) if abs(captures[k] - capture) <= 2]
        window = contrast[indices]
        center = np.median(window, axis=0)
        median.append(center)
        mad.append(1.4826 * np.median(np.abs(window - center), axis=0))
        agreement.append(np.mean(np.sign(window) == np.sign(center), axis=0))
    return np.asarray(median), np.asarray(mad), np.asarray(agreement)


def stationary_proposals(fields, coverage, pink, floor=5, ratio=2.):
    median, mad, sign = fields
    output = []
    for t in range(len(pink)):
        stable = (np.abs(median[t]) >= floor) & (np.abs(median[t]) >= ratio * (mad[t] + 2)) & (sign[t] >= .8) & (coverage[t] >= .15)
        stable &= pink[t]
        stable = ndi.binary_opening(stable, structure=morphology.disk(2))
        stable = ndi.binary_closing(stable, structure=morphology.disk(3)) & pink[t] & (coverage[t] >= .15)
        items = []
        for polarity in (-1, 1):
            allowed = stable & (median[t] * polarity > 0)
            peaks = morphology.h_maxima(polarity * median[t], 3) & allowed
            markers, count = ndi.label(peaks, STRUCTURE)
            if not count:
                markers, _ = ndi.label(allowed, STRUCTURE)
            labels = segmentation.watershed(-polarity * median[t], markers, mask=allowed, connectivity=2)
            for prop in measure.regionprops(labels):
                mask = labels == prop.label
                if 180 <= prop.area <= 5000 and 2 * ndi.distance_transform_edt(mask).max() >= 12:
                    items.append(dict(mask=mask, area=int(prop.area), center=np.array(prop.centroid), polarity=polarity, t=t))
        output.append(items)
    return output


def stationary_tracks(fields, coverage, pink, captures, floor=5, ratio=2.):
    scales = [stationary_proposals(field, coverage, pink, floor, ratio) for field in fields]
    observations = []
    for t in range(len(pink)):
        items = []
        for item in scales[1][t]:
            peers = []
            for scale in (0, 2):
                options = [(overlap(item['mask'], other['mask']), other) for other in scales[scale][t] if item['polarity'] == other['polarity']]
                if options:
                    score, peer = max(options, key=lambda pair: pair[0])
                    if score >= .35:
                        peers.append(peer['mask'])
            if not peers:
                continue
            votes = np.sum([item['mask']] + peers, axis=0)
            mask = (votes >= (2 if len(peers) == 2 else 1)) & pink[t]
            components, count = ndi.label(mask, STRUCTURE)
            if not count:
                continue
            sizes = np.bincount(components.ravel()); sizes[0] = 0
            mask = components == sizes.argmax()
            if 180 <= mask.sum() <= 5000 and 2 * ndi.distance_transform_edt(mask).max() >= 12:
                items.append(dict(item, mask=mask, area=int(mask.sum()), center=np.array(ndi.center_of_mass(mask))))
        observations.append(items)
    tracks = []
    for t, items in enumerate(observations):
        active = [k for k, track in enumerate(tracks) if 1 <= captures[t] - captures[track[-1]['t']] <= 2]
        costs = np.full((len(active), len(items)), 1e6)
        for a, k in enumerate(active):
            old = tracks[k][-1]
            for b, item in enumerate(items):
                elapsed = captures[t] - captures[old['t']]
                limit = min(5, .3 * np.sqrt(old['area'] / np.pi)) * elapsed
                score = overlap(old['mask'], item['mask'])
                if old['polarity'] == item['polarity'] and np.linalg.norm(old['center'] - item['center']) <= limit and 1 / 1.5 <= item['area'] / old['area'] <= 1.5 and score >= .4:
                    costs[a, b] = 1 - score
        used = set()
        if costs.size:
            aa, bb = linear_sum_assignment(costs)
            for a, b in zip(aa, bb):
                if costs[a, b] < 1e5:
                    tracks[active[a]].append(items[b]); used.add(b)
        tracks.extend([item] for b, item in enumerate(items) if b not in used)
    return [track for track in tracks if len(track) >= 5 and np.quantile([item['area'] for item in track], .9) / np.quantile([item['area'] for item in track], .1) <= 1.8]


def merge_decision(geometry, intervals, candidate_is_new, parent_is_partial):
    contact = [row for row in geometry if row['contact'] >= 5 and row['distance'] <= 2]
    frames = {row['frame'] for row in contact}
    tested = [row for row in intervals if row['source'] in frames and row['target'] in frames]
    good = [row for row in tested if min(row['parent_ncc'], row['candidate_ncc']) >= .65 and row['penalty'] <= .10 and row['relative_step'] <= 4]
    support = float(np.median([row['parent_support'] for row in contact])) if contact else 0.
    accepted = candidate_is_new and parent_is_partial and len(contact) >= 4 and len(contact) / max(1, len(geometry)) >= .6 and support >= .5 and len(good) >= 3 and len(good) / max(1, len(tested)) >= .75
    return dict(accepted=bool(accepted), first_supported_frame=min(frames) if frames else None,
                contact_frames=len(contact), parent_support=support, good_intervals=len(good), tested_intervals=len(tested))


def refine_regions(raw, labels, pink, outer, inner, captures, *, check=lambda: None, progress=lambda *_: None):
    labels = np.asarray(labels, np.int32).copy()
    outer, inner, pink = np.asarray(outer, bool), np.asarray(inner, bool), np.asarray(pink, bool)
    captures = list(map(int, captures))
    if labels.shape != raw.shape or labels.shape != outer.shape or labels.shape != pink.shape or labels.shape != inner.shape:
        raise ValueError('Region evidence must use the same image coordinates')
    if len(captures) != len(raw) or any(b <= a for a, b in zip(captures, captures[1:])):
        raise ValueError('Region evidence needs increasing capture indices')
    pink_values = np.unique(labels[pink & (labels > 0)])
    pink_id = int(pink_values[0]) if len(pink_values) == 1 and not np.any((labels == pink_values[0]) & ~pink) else int(labels.max()) + 1
    labels[pink | (outer & (labels == 0))] = pink_id
    labels[~outer] = 0
    next_id = int(labels.max()) + 1
    events, partial, projections = [], {}, {}
    fine, coarse = normalized_images(raw)
    clahe = cv2.createCLAHE(2., (8, 8))
    tracking = np.array([clahe.apply(cv2.GaussianBlur(g, (0, 0), 1.)) for g in raw])
    density = [np.array([ndi.uniform_filter(1 - g.astype(np.float32) / 255, width) - ndi.gaussian_filter(1 - g.astype(np.float32) / 255, 48) for g in raw]) for width in (7, 11, 17)]
    history = {}
    for t in range(len(raw)):
        check(); progress(t, 'Checking territory continuity')
        visible_fragments = []
        if not outer[t].any():
            history.clear()
            continue
        for ident, state in list(history.items()):
            check()
            source, mask, last_full = state
            current = labels[t] == ident
            ratio = current.sum() / max(1, mask.sum())
            recovered = None
            if .55 <= ratio <= 1.8:
                if ident in partial:
                    partial[ident].add(t)
                    visible_fragments.append((ident, current.copy()))
                    events.append(dict(capture=captures[t], display_id=ident, event='partially_observed', observed_area_px=int(current.sum())))
                    anchor_t, full = projections[ident]
                    peaks = masked_search(coarse[anchor_t], coarse[t], labels[anchor_t] == ident)
                    if peaks:
                        shift = np.array([[1., 0, peaks[0]['dx']], [0, 1., peaks[0]['dy']]])
                        projections[ident, t] = warp(full, shift)
                continue
            status = 'observed_continuity'
            if ident not in partial:
                fit = fragment_fit(source, t, mask, fine, coarse)
                if fit is not None and fit[1].sum() >= 120:
                    recovered = fit; status = 'partially_observed'
                else:
                    recovered = deformation_fit(source, t, mask, tracking, fine, coarse)
            if recovered is not None:
                affine, predicted = recovered
                predicted &= outer[t]
                radius = max(40, 1.5 * np.sqrt(mask.sum() / np.pi))
                score = .7 * correlation(fine[source], fine[t], mask, affine) + .3 * correlation(coarse[source], coarse[t], mask, affine)
                decoys = []
                for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    wrong = affine.copy()
                    wrong[:, 2] += radius * np.array([np.cos(angle), np.sin(angle)])
                    decoys.append(.7 * correlation(fine[source], fine[t], mask, wrong) + .3 * correlation(coarse[source], coarse[t], mask, wrong))
                if score - max(decoys) < .10 or predicted.sum() < 120:
                    recovered = None
                elif .77 <= current.sum() / max(1, predicted.sum()) <= 1.3 and overlap(current, predicted) >= .5:
                    continue
            if recovered is None:
                recovered = recover_density(source, t, mask, density, fine, coarse, outer)
            if recovered is None:
                fit = fragment_fit(source, t, mask, fine, coarse)
                if fit is not None and fit[1].sum() >= 120:
                    recovered = fit; status = 'partially_observed'
            if recovered is None:
                if current.any():
                    labels[t][current] = pink_id
                events.append(dict(capture=captures[t], display_id=ident, event='unresolved_correspondence'))
                continue
            affine, observed = recovered
            observed = observed & outer[t]
            if status != 'partially_observed' and ident not in partial:
                observed &= (labels[t] == pink_id) | (labels[t] == ident)
            if observed.sum() < 120:
                continue
            labels[t][current] = pink_id
            labels[t][observed] = ident
            if status == 'partially_observed' or ident in partial:
                visible_fragments.append((ident, observed.copy()))
                partial.setdefault(ident, set()).add(t)
                if ident not in projections:
                    projections[ident] = (t, warp(last_full, affine))
                anchor_t, full = projections[ident]
                if t != anchor_t:
                    anchor_mask = labels[anchor_t] == ident
                    peaks = masked_search(coarse[anchor_t], coarse[t], anchor_mask)
                    if peaks:
                        shift = np.array([[1., 0, peaks[0]['dx']], [0, 1., peaks[0]['dy']]])
                        projections[ident, t] = warp(full, shift)
                else:
                    projections[ident, t] = full
            events.append(dict(capture=captures[t], display_id=ident, event=status, observed_area_px=int(observed.sum())))
        ownership = np.full(labels.shape[1:], -1., np.float32)
        for ident, observed in visible_fragments:
            depth = ndi.distance_transform_edt(observed)
            score = depth / max(1., depth.max())
            supported = observed & (score > ownership)
            labels[t][supported] = ident
            ownership[supported] = score[supported]
        for ident in np.unique(labels[t]):
            if ident in (0, pink_id):
                continue
            mask = labels[t] == ident
            old_full = history[ident][2] if ident in history and ident in partial else mask
            history[int(ident)] = (t, mask.copy(), old_full.copy())
    pink = labels == pink_id
    coverage = np.array([ndi.uniform_filter(m.astype(float), 25) / np.maximum(1e-6, ndi.uniform_filter(o.astype(float), 25)) for m, o in zip(inner, outer)])
    fields = []
    for scale in (5, 8, 12):
        check(); progress(scale, 'Measuring persistent structure')
        fields.append(stationary_fields(raw, outer, captures, scale))
    trials = []
    for floor, ratio in ((5, 2.), (4, 1.5), (7, 2.5)):
        check()
        trials.append(stationary_tracks(fields, coverage, pink, captures, floor, ratio))
    def volume(track):
        result = np.zeros(labels.shape, bool)
        for item in track:
            result[item['t']] = item['mask']
        return result
    alternatives = [[volume(track) for track in trial] for trial in trials[1:]]
    new_ids = []
    for track in trials[0]:
        check()
        mask = volume(track)
        agreement = [max((overlap(mask, candidate) for candidate in trial), default=0.) for trial in alternatives]
        if min(agreement) < .5:
            continue
        ident = next_id; next_id += 1
        labels[mask & (labels == pink_id)] = ident
        previous = None
        median, mad, sign = fields[1]
        for t in range(track[0]['t'], len(labels)):
            observed = labels[t] == ident
            if not observed.any() and previous is not None:
                stable = (np.abs(median[t]) >= 5) & (np.abs(median[t]) >= 2 * (mad[t] + 2)) & (sign[t] >= .8)
                stable &= (median[t] * track[0]['polarity'] > 0) & (coverage[t] >= .15)
                support = stable & (labels[t] == pink_id) & ndi.binary_dilation(previous, iterations=4)
                support = ndi.binary_closing(support, iterations=2) & (labels[t] == pink_id)
                factor = support.sum() / max(1, previous.sum())
                if support.sum() >= 120 and overlap(support, previous) >= .5 and .6 <= factor <= 1.6:
                    labels[t][support] = ident
                    observed = support
            if observed.any():
                previous = observed
        new_ids.append(ident)
        events.append(dict(display_id=ident, event='persistent_structure', capture=captures[track[0]['t']], confirmation_capture=captures[track[4]['t']], observed_captures=[captures[t] for t in range(len(labels)) if (labels[t] == ident).any()], parameter_agreement=agreement))
    pre = labels.copy()
    proposals = []
    for parent, times in partial.items():
        for candidate in new_ids:
            check()
            geometry, intervals = [], []
            visible = [t for t in sorted(times) if (pre[t] == parent).any() and (pre[t] == candidate).any()]
            for t in visible:
                a, b = pre[t] == parent, pre[t] == candidate
                prior = projections.get((parent, t))
                support = float((ndi.binary_dilation(prior, iterations=5) & b).sum() / b.sum()) if prior is not None else 0.
                geometry.append(dict(frame=t, contact=int((ndi.binary_dilation(a, structure=STRUCTURE) & b).sum()), distance=float(ndi.distance_transform_edt(~a)[b].min()), parent_support=support))
            for t, u in zip(visible, visible[1:]):
                if captures[u] - captures[t] != 1:
                    continue
                a, b = pre[t] == parent, pre[t] == candidate
                fits = [masked_search(coarse[t], coarse[u], mask, 12) for mask in (a, b, a | b)]
                if not all(fits):
                    continue
                joint = fits[2][0]
                affine = np.array([[1., 0, joint['dx']], [0, 1., joint['dy']]])
                scores = [correlation(coarse[t], coarse[u], mask, affine) for mask in (a, b)]
                delta = (np.array(ndi.center_of_mass(pre[u] == candidate)) - ndi.center_of_mass(pre[u] == parent)) - (np.array(ndi.center_of_mass(b)) - ndi.center_of_mass(a))
                intervals.append(dict(source=t, target=u, parent_ncc=scores[0], candidate_ncc=scores[1], penalty=max(fits[0][0]['ncc'] - scores[0], fits[1][0]['ncc'] - scores[1]), relative_step=float(np.linalg.norm(delta))))
            decision = merge_decision(geometry, intervals, True, True)
            if decision['accepted']:
                proposals.append((candidate, parent, decision))
    aliases = []
    for candidate in new_ids:
        options = [(parent, decision) for ident, parent, decision in proposals if ident == candidate]
        options.sort(key=lambda pair: pair[1]['parent_support'], reverse=True)
        if not options or len(options) > 1 and options[0][1]['parent_support'] - options[1][1]['parent_support'] < .1:
            continue
        parent, decision = options[0]
        first = decision['first_supported_frame']
        aliases.append((candidate, parent, first))
        events.append(dict(event='fragment_reconciliation', display_id=parent, alias_id=candidate, capture=captures[first], **decision))
    # Close only narrow interfaces; existing territory pixels are protected.
    yy, xx = np.indices(labels.shape[1:])
    for t in range(len(labels)):
        check()
        established = (pre[t] > 0) & (pre[t] != pink_id) & ~np.isin(pre[t], new_ids)
        if not established.any():
            continue
        old_distance, old_nearest = ndi.distance_transform_edt(~established, return_indices=True)
        for ident in new_ids:
            mask = labels[t] == ident
            if not mask.any():
                continue
            distance, nearest = ndi.distance_transform_edt(~mask, return_indices=True)
            cosine = ((yy - nearest[0]) * (yy - old_nearest[0]) + (xx - nearest[1]) * (xx - old_nearest[1])) / np.maximum(1e-9, distance * old_distance)
            seam = (labels[t] == pink_id) & (distance <= 3) & (old_distance <= 3) & (distance + old_distance <= 5) & (cosine < -.25)
            labels[t][seam] = ident
    for candidate, parent, first in aliases:
        for t in range(first, len(labels)):
            labels[t][labels[t] == candidate] = parent
    if not np.array_equal(labels > 0, outer):
        raise ValueError('Territory partition does not cover the biofilm')
    return labels, labels == pink_id, events


FLOW_LIMITS = dict(forward_backward_error=2., max_flow=18., max_gray_residual=30.)



def components(mask):
    return ndi.label(mask,structure=np.ones((3,3)))


def flow_pair(prev,cur,prev_mask,cur_mask):
    args=(None,.5,4,31,5,7,1.5,0)
    forward=cv2.calcOpticalFlowFarneback(prev,cur,*args)
    backward=cv2.calcOpticalFlowFarneback(cur,prev,*args)
    yy,xx=np.indices(cur.shape,dtype=np.float32)
    mx=xx+backward[...,0]; my=yy+backward[...,1]
    def remap(a,nearest=False):
        return cv2.remap(a,mx,my,cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    back_forward=remap(forward)
    error=np.linalg.norm(backward+back_forward,axis=-1)
    residual=np.abs(cur.astype(float)-remap(prev.astype(np.float32)))
    valid=(cur_mask & (remap(prev_mask.astype(np.uint8),True)>0)
           & (mx>=0)&(my>=0)&(mx<cur.shape[1]-1)&(my<cur.shape[0]-1)
           & (error<=FLOW_LIMITS['forward_backward_error'])
           & (np.linalg.norm(backward,axis=-1)<=FLOW_LIMITS['max_flow'])
           & (residual<=FLOW_LIMITS['max_gray_residual']))
    return dict(forward=forward,backward=backward,valid=valid,remap=remap)


def darkness_map(image, mask, width=32):
    """Exact notebook box filter with an ROI-normalized denominator."""
    values = 1.0-image.astype(float)/255.0
    support = cv2.boxFilter(mask.astype(float),-1,(width,width),normalize=False,borderType=cv2.BORDER_CONSTANT)
    sums = cv2.boxFilter(np.where(mask,values,0),-1,(width,width),normalize=False,borderType=cv2.BORDER_CONSTANT)
    result = np.full(image.shape,np.nan)
    np.divide(sums,support,out=result,where=mask & (support>0))
    return np.clip(result,0,1),support/(width*width)


def split_regions(image, mask, params):
    """Prominent density peaks, spacing, watershed and reflooding."""
    d,support = darkness_map(image,mask,int(params['width']))
    sigma = float(params['smooth_sigma'])
    numerator = ndi.gaussian_filter(np.where(mask,d,0),sigma,mode='constant')
    denominator = ndi.gaussian_filter(mask.astype(float),sigma,mode='constant')
    smooth = np.divide(numerator,denominator,out=np.zeros_like(d),where=denominator>0)
    surface = np.where(mask,smooth,-1.0)
    maxima = morphology.h_maxima(surface,float(params['prominence'])).astype(bool)
    maxima &= mask & (support>=params['min_support_fraction'])
    components,n = ndi.label(maxima,structure=np.ones((3,3)))
    candidates = []
    for c in range(1,n+1):
        yy,xx = np.where(components==c)
        j = np.argmax(smooth[yy,xx])
        candidates.append((float(smooth[yy[j],xx[j]]),int(yy[j]),int(xx[j])))
    markers = np.zeros(mask.shape,np.int32)
    chosen = []
    for height,y,x in sorted(candidates,reverse=True):
        if all((y-v)**2+(x-u)**2>=params['min_peak_distance']**2 for v,u in chosen):
            chosen.append((y,x));markers[y,x]=len(chosen)
    def flood(m):
        if not m.any():
            return np.zeros(mask.shape,np.int32)
        return segmentation.watershed(-smooth,m,mask=mask,connectivity=2).astype(np.int32)
    labels = flood(markers)
    while labels.any():
        sizes = np.bincount(labels.ravel())
        tiny = np.flatnonzero((sizes<params['min_region_area']) & (np.arange(len(sizes))>0))
        if not len(tiny):
            break
        markers[np.isin(markers,tiny)]=0
        labels=flood(markers)
    labels,_,_ = segmentation.relabel_sequential(labels)
    return dict(darkness=d,smooth=np.where(mask,smooth,np.nan),labels=labels.astype(np.int32),markers=markers,support=support)


def image_motion(previous, current, domain, *, max_corners=300, min_distance=10):
    """Forward/backward-checked raw texture displacement and robust affine fit.

    Require FB <=1px, 9x9 patch NCC >=.60 in locally normalized and
    enhanced images, and displacement <=120px.
    No rejected points are replaced by centroid shifts or synthetic zero vectors.
    """
    if not domain.any():
        return [],None
    def normalize(a):
        a=a.astype(np.float32);mu=ndi.gaussian_filter(a,15)
        sd=np.sqrt(np.maximum(0,ndi.gaussian_filter(a*a,15)-mu*mu))
        return (a-mu)/np.maximum(sd,5)
    clahe=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8))
    a=clahe.apply(previous);b=clahe.apply(current)
    points=cv2.goodFeaturesToTrack(a,maxCorners=max_corners,qualityLevel=.01,minDistance=min_distance,mask=domain.astype(np.uint8)*255,blockSize=5)
    if points is None:
        return [],None
    params=dict(winSize=(25,25),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    target,ok,_=cv2.calcOpticalFlowPyrLK(a,b,points,None,**params)
    if target is None:
        return [],None
    finite=np.isfinite(target).all(axis=(1,2))
    safe=np.where(finite[:,None,None],target,points).astype(np.float32)
    back,reverse,_=cv2.calcOpticalFlowPyrLK(b,a,safe,None,**params)
    if back is None:
        return [],None
    na,nb=normalize(previous),normalize(current)
    h,w=previous.shape;vectors=[]
    def ncc(x,y):
        x=x.astype(float).ravel();y=y.astype(float).ravel();x-=x.mean();y-=y.mean()
        den=np.linalg.norm(x)*np.linalg.norm(y)
        return float(x@y/den) if den>1e-9 else -1.
    for i,(src,dst,rev) in enumerate(zip(points[:,0],safe[:,0],back[:,0])):
        if not finite[i] or not ok[i,0] or not reverse[i,0]:
            continue
        if not np.isfinite(rev).all():
            continue
        error=float(np.linalg.norm(rev-src));distance=float(np.linalg.norm(dst-src))
        if error>1 or distance>120:
            continue
        x,y=np.rint(src).astype(int);tx,ty=np.rint(dst).astype(int)
        if min(x,y,tx,ty)<4 or x>=w-4 or tx>=w-4 or y>=h-4 or ty>=h-4:
            continue
        enhanced=ncc(a[y-4:y+5,x-4:x+5],b[ty-4:ty+5,tx-4:tx+5])
        normalized=ncc(na[y-4:y+5,x-4:x+5],nb[ty-4:ty+5,tx-4:tx+5])
        if min(enhanced,normalized)<.60:
            continue
        vectors.append(dict(start=src.tolist(),end=dst.tolist(),dx_px=float(dst[0]-src[0]),dy_px=float(dst[1]-src[1]),fb_error_px=error,enhanced_ncc=enhanced,normalized_ncc=normalized))
    affine=None
    if len(vectors)>=6:
        src=np.array([v['start'] for v in vectors],np.float32)
        dst=np.array([v['end'] for v in vectors],np.float32)
        matrix,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=1.5)
        if matrix is not None and inliers is not None:
            fraction=float(inliers.mean());singular=np.linalg.svd(matrix[:,:2],compute_uv=False)
            residual=np.linalg.norm(cv2.transform(src[None],matrix)[0]-dst,axis=1)
            good=inliers.ravel()>0
            if fraction>=.60 and good.sum()>=6 and np.median(residual[good])<=1.5 and singular.min()>=.6 and singular.max()<=1.6:
                affine=matrix
    return vectors,affine


def temporal_partition(images,domains,*,check=lambda: None):
    gray=np.asarray(images); masks=np.asarray(domains); captures=np.arange(len(gray))
    P=dict(width=32,smooth_sigma=2.,prominence=.025,min_peak_distance=18,min_region_area=120,min_support_fraction=.5)
    S=dict(temporal_weight=.35,post_temporal_sigma=1.2,relief_sigma=18.,relief_gain=.65,birth_prominence=.04,continue_prominence=.012,match_distance=18.,seed_exclusion=32.,birth_observations=2,retire_observations=2,boundary_prior=.012,max_flow=18.,forward_backward_error=2.,max_gray_residual=30.)
    def normalized_gaussian(values,mask,sigma):
        num=ndi.gaussian_filter(np.where(mask,values,0).astype(float),sigma,mode='constant')
        den=ndi.gaussian_filter(mask.astype(float),sigma,mode='constant')
        return np.divide(num,den,out=np.zeros_like(num),where=den>1e-8)

    def flow_pair(prev,cur,prev_mask,cur_mask):
        args=(None,.5,4,31,5,7,1.5,0)
        forward=cv2.calcOpticalFlowFarneback(prev,cur,*args)
        backward=cv2.calcOpticalFlowFarneback(cur,prev,*args)
        yy,xx=np.indices(cur.shape,dtype=np.float32)
        mx=xx+backward[...,0]; my=yy+backward[...,1]
        def remap(a,nearest=False):
            return cv2.remap(a,mx,my,cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        back_forward=remap(forward)
        error=np.linalg.norm(backward+back_forward,axis=-1)
        residual=np.abs(cur.astype(float)-remap(prev.astype(np.float32)))
        valid=(cur_mask & (remap(prev_mask.astype(np.uint8),True)>0)
               & (mx>=0)&(my>=0)&(mx<cur.shape[1]-1)&(my<cur.shape[0]-1)
               & (error<=S['forward_backward_error'])
               & (np.linalg.norm(backward,axis=-1)<=S['max_flow'])
               & (residual<=S['max_gray_residual']))
        return dict(forward=forward,backward=backward,valid=valid,remap=remap)

    def detect_peaks(surface,mask,support,prominence):
        peaks=morphology.h_maxima(np.where(mask,surface,-2),prominence).astype(bool)
        peaks &= mask & (support>=P['min_support_fraction'])
        cc,n=ndi.label(peaks,structure=np.ones((3,3)))
        candidates=[]
        for i in range(1,n+1):
            ys,xs=np.where(cc==i); j=np.argmax(surface[ys,xs])
            candidates.append((float(surface[ys[j],xs[j]]),float(ys[j]),float(xs[j])))
        keep=[]
        for value,y,x in sorted(candidates,reverse=True):
            if all((y-v[0])**2+(x-v[1])**2>=P['min_peak_distance']**2 for v in keep):
                keep.append((y,x,value))
        return keep

    def advect(y,x,motion):
        iy,ix=int(round(y)),int(round(x))
        if motion is None: return y,x,False
        dx,dy=motion['forward'][np.clip(iy,0,gray.shape[1]-1),np.clip(ix,0,gray.shape[2]-1)]
        py,px=float(y+dy),float(x+dx)
        jy,jx=int(round(py)),int(round(px))
        good=(0<=jy<gray.shape[1] and 0<=jx<gray.shape[2] and motion['valid'][jy,jx])
        return (py,px,True) if good else (y,x,False)

    def assignment(predictions,peaks,maximum):
        if not predictions or not peaks: return {},{}
        a=np.array([[p['y'],p['x']] for p in predictions]); b=np.array([p[:2] for p in peaks])
        distances=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)
        # Dummy columns allow unmatched tracks; inadmissible assignments are penalized.
        costs=np.where(distances<=maximum,distances,1e6)
        costs=np.column_stack((costs,np.full((len(a),len(a)),maximum+.01)))
        ii,jj=linear_sum_assignment(costs)
        matches={int(i):int(j) for i,j in zip(ii,jj) if j<len(b) and distances[i,j]<=maximum}
        gaps={i:(float(np.sort(distances[i])[1]-np.sort(distances[i])[0]) if len(b)>1 else None) for i in matches}
        return matches,gaps


    signals=[]; reliefs=[]; enhanced=[]; motion_maps=[]; flow_rows=[]
    for k in range(len(gray)):
        check()
        d,support=darkness_map(gray[k],masks[k],P['width'])
        current=normalized_gaussian(d,masks[k],P['smooth_sigma'])
        motion=None
        if k and captures[k]-captures[k-1]==1:
            motion=flow_pair(gray[k-1],gray[k],masks[k-1],masks[k])
            old=motion['remap'](signals[-1].astype(np.float32))
            current=np.where(motion['valid'],(1-S['temporal_weight'])*current+S['temporal_weight']*old,current)
        current=normalized_gaussian(current,masks[k],S['post_temporal_sigma'])
        local=current-normalized_gaussian(current,masks[k],S['relief_sigma'])
        sharp=current+S['relief_gain']*local
        signals.append(current); reliefs.append(local); enhanced.append(sharp); motion_maps.append(motion)
        flow_rows.append(dict(capture=int(captures[k]),supported_fraction=float(motion['valid'][masks[k]].mean())
                             if motion and masks[k].any() else None))

    stable=[]; seed_rows=[]; events=[]; active={}; pending=[]; next_id=1
    for k in range(len(gray)):
        check()
        mask=masks[k]; surface=enhanced[k]; motion=motion_maps[k]
        _,support=darkness_map(gray[k],mask,P['width'])
        low=detect_peaks(signals[k],mask,support,S['continue_prominence'])
        high=detect_peaks(signals[k],mask,support,S['birth_prominence'])
        if k==0 or captures[k]-captures[k-1]!=1:
            if k: events.append(dict(capture=int(captures[k]),event='reset_at_capture_gap',display_id=None))
            active={}; pending=[]
            initial=split_regions(gray[k],mask,P)
            for r in measure.regionprops(initial['labels']):
                yy,xx=np.where(initial['labels']==r.label)
                # Select the actual retained marker in this baseline basin.
                marked=initial['markers'][yy,xx]>0
                if not marked.any(): continue
                y,x=yy[marked][0],xx[marked][0]
                ident=next_id; next_id+=1
                active[ident]=dict(y=float(y),x=float(x),misses=0,status='initial',flow_supported=False,ambiguity_gap=None)
                events.append(dict(capture=int(captures[k]),event='initial_seed',display_id=ident))
        else:
            ids=sorted(active); predicted=[]
            for ident in ids:
                a=active[ident]; y,x,ok=advect(a['y'],a['x'],motion)
                predicted.append(dict(y=y,x=x,flow_supported=ok))
            matches,gaps=assignment(predicted,low,S['match_distance'])
            updated={}
            for i,ident in enumerate(ids):
                pred=predicted[i]
                if i in matches:
                    y,x,_=low[matches[i]]
                    gap=gaps[i]
                    status='ambiguous_peak_match' if gap is not None and gap<2 else 'peak_supported'
                    updated[ident]=dict(y=y,x=x,misses=0,status=status,flow_supported=pred['flow_supported'],ambiguity_gap=gap)
                else:
                    misses=active[ident]['misses']+1
                    y,x=pred['y'],pred['x']; iy,ix=int(round(y)),int(round(x))
                    inside=0<=iy<mask.shape[0] and 0<=ix<mask.shape[1] and mask[iy,ix]
                    if misses<S['retire_observations'] and inside:
                        updated[ident]=dict(y=y,x=x,misses=misses,status='held_without_peak',flow_supported=pred['flow_supported'],ambiguity_gap=None)
                    else: events.append(dict(capture=int(captures[k]),event='retired_no_peak_or_left_roi',display_id=ident))
            active=updated
            # Pending birth hypotheses must match a high-prominence peak again.
            fresh=[p for p in high if all(np.hypot(p[0]-a['y'],p[1]-a['x'])>=S['seed_exclusion'] for a in active.values())]
            predictions=[]
            for a in pending:
                y,x,_=advect(a['y'],a['x'],motion); predictions.append(dict(y=y,x=x))
            pm,_=assignment(predictions,fresh,S['match_distance'])
            used=set(); next_pending=[]
            for i,j in pm.items():
                y,x,_=fresh[j]; hits=pending[i]['hits']+1; used.add(j)
                if hits>=S['birth_observations']:
                    if all(np.hypot(y-a['y'],x-a['x'])>=S['seed_exclusion'] for a in active.values()):
                        ident=next_id;next_id+=1
                        active[ident]=dict(y=y,x=x,misses=0,status='confirmed_birth',flow_supported=False,ambiguity_gap=None)
                        events.append(dict(capture=int(captures[k]),event='confirmed_birth',display_id=ident))
                else: next_pending.append(dict(y=y,x=x,hits=hits))
            for j,(y,x,_) in enumerate(fresh):
                if j not in used: next_pending.append(dict(y=y,x=x,hits=1))
            pending=next_pending
        markers=np.zeros(mask.shape,np.int32)
        for ident,a in list(active.items()):
            y,x=int(round(a['y'])),int(round(a['x']))
            if not mask[y,x] or markers[y,x]>0:
                active.pop(ident); events.append(dict(capture=int(captures[k]),event='retired_seed_collision',display_id=ident));continue
            markers[y,x]=ident
        terrain=-surface.copy()
        if k and motion is not None:
            old=motion['remap'](stable[-1].astype(np.float32),True).astype(np.int32)
            old_boundary=segmentation.find_boundaries(old,mode='inner')
            # Only internal prior boundaries; never force a prior ROI outline.
            old_boundary &= ndi.binary_erosion(old>0,iterations=3)
            ridge=ndi.gaussian_filter(ndi.binary_dilation(old_boundary).astype(float),1.)
            terrain+=S['boundary_prior']*ridge*motion['valid']
        lab=segmentation.watershed(terrain,markers,mask=mask,connectivity=2) if markers.any() else np.zeros(mask.shape,np.int32)
        lab=lab.astype(np.int32); assert not lab[~mask].any()
        stable.append(lab)
        for ident,a in active.items():
            seed_rows.append(dict(capture=int(captures[k]),display_id=ident,seed_x_px=a['x'],seed_y_px=a['y'],
                evidence_status=a['status'],uncertain=a['status'] in ['held_without_peak','ambiguous_peak_match'],
                flow_supported=a['flow_supported'],misses=a['misses'],ambiguity_gap_px=a['ambiguity_gap']))
    stable=np.stack(stable)
    return dict(labels=stable,signals=np.asarray(signals),enhanced=np.asarray(enhanced),seeds=seed_rows,motion=motion_maps,events=events)


def pocket_candidates(image, mask, signal, enhanced, labels, markers, motion):
    _, support = darkness_map(image, mask, 32)
    relief = signal - ndi.gaussian_filter(signal, 18)
    centers = np.argwhere(markers > 0)
    yy, xx = np.indices(mask.shape)
    vectors, affine = motion
    if vectors and affine is not None:
        start = np.array([v['start'] for v in vectors])
        end = np.array([v['end'] for v in vectors])
        residual = end - (start @ affine[:, :2].T + affine[:, 2])
        xy = np.rint(start).astype(int)
    else:
        xy = np.empty((0, 2), int); residual = np.empty((0, 2))
    output, audit = [], []
    for polarity, field in [('dark', enhanced), ('light', -relief)]:
        extrema = morphology.h_maxima(np.where(mask, field, -2), .006) & mask & (support >= .5)
        components, count = ndi.label(extrema, np.ones((3, 3)))
        points = []
        for ident, section in enumerate(ndi.find_objects(components), 1):
            values = np.where(components[section] == ident, field[section], -np.inf)
            y, x = np.unravel_index(values.argmax(), values.shape)
            points.append((y + section[0].start, x + section[1].start))
        seeds = np.zeros(mask.shape, np.int32)
        for ident, (y, x) in enumerate(points, 1):
            seeds[y, x] = ident
        basins = segmentation.watershed(-field, seeds, mask=mask)
        for index, (y, x) in enumerate(points):
            distance = float(np.min(np.linalg.norm(centers - [y, x], axis=1)))
            if distance < 32:
                continue
            radius = 42 if polarity == 'dark' else 48
            nearby = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
            if polarity == 'dark':
                test = markers.copy(); test[y, x] = int(markers.max()) + 1
                region = (segmentation.watershed(-enhanced, test, mask=mask) == test[y, x]) & nearby
                if not region.any():
                    continue
                background = np.quantile(enhanced[region], .15)
                region &= enhanced >= background + .15 * (enhanced[y, x] - background)
            else:
                group = [i + 1 for i, (v, u) in enumerate(points) if np.hypot(y - v, x - u) <= 24]
                region = np.isin(basins, group) & (relief < 0) & nearby
            cc, _ = ndi.label(region, np.ones((3, 3)))
            if not cc[y, x]:
                continue
            region = cc == cc[y, x]
            area = int(region.sum())
            if area < 120:
                continue
            rim = ndi.binary_dilation(region, iterations=3) & ~region
            neighbors = len(np.unique(labels[rim & (labels > 0)]))
            v = residual[region[xy[:, 1], xy[:, 0]]]
            coherence = float(np.linalg.norm(v.mean(0)) / max(1e-6, np.linalg.norm(v, axis=1).mean())) if len(v) else 0.
            dispersion = float(np.median(np.linalg.norm(v - np.median(v, axis=0), axis=1))) if len(v) else None
            mean = float(signal[region].mean())
            if polarity == 'dark':
                accepted = 500 <= area <= 1000 and mean >= .54 and relief[region].mean() < 0 and relief[region].max() >= .02
            else:
                accepted = area >= 1200 and mean <= .5101 and neighbors >= 3 and len(v) >= 4 and coherence >= .9 and dispersion <= .5
            row = dict(polarity=polarity, area=area, marker_distance=distance, mean_darkness=mean,
                       neighbors=neighbors, points=len(v), coherence=coherence,
                       dispersion=dispersion, accepted=bool(accepted), center=[int(y), int(x)])
            audit.append(row)
            if accepted and not any(r['polarity'] == polarity and (region & r['mask']).sum() / area > .5 for r in output):
                output.append(dict(row, mask=region))
    return output, audit


class RawMotion:
    def __init__(self, images):
        self.raw = np.asarray(images)
        self.height, self.width = images.shape[1:]
        clahe = cv2.createCLAHE(2., (8, 8))
        self.tracking = np.array([clahe.apply(cv2.GaussianBlur(g, (0, 0), 1.)) for g in images])
        normalized = []
        for g in images:
            a = g.astype(np.float32); mean = ndi.gaussian_filter(a, 15)
            sd = np.sqrt(np.maximum(0, ndi.gaussian_filter(a * a, 15) - mean * mean))
            normalized.append((a - mean) / np.maximum(sd, 5))
        self.fine = np.array([ndi.gaussian_filter(a, .7) for a in normalized])
        self.coarse = np.array([ndi.gaussian_filter(a, 2.5) for a in normalized])
        self.density = {w: np.array([ndi.uniform_filter(1 - g.astype(np.float32) / 255, w) -
                                   ndi.gaussian_filter(1 - g.astype(np.float32) / 255, 48)
                                   for g in images]) for w in (7, 11, 17)}

    @staticmethod
    def warp(mask, affine):
        return cv2.warpAffine(mask.astype(np.uint8), np.asarray(affine, float),
                             mask.shape[::-1], flags=cv2.INTER_NEAREST).astype(bool)

    @staticmethod
    def geometry(affine, mask):
        singular = np.linalg.svd(affine[:, :2], compute_uv=False)
        y, x = ndi.center_of_mass(mask)
        determinant = float(np.linalg.det(affine[:, :2]))
        valid = singular.min() >= .55 and singular.max() <= 1.9 and .4 <= determinant <= 2.5
        valid &= np.linalg.norm(affine @ [x, y, 1] - [x, y]) <= 65
        return bool(valid), determinant

    def ncc(self, source, target, mask, affine, images=None):
        images = self.fine if images is None else images
        y, x = np.nonzero(mask)
        if len(x) < 30:
            return 0.
        y0, y1, x0, x1 = y.min(), y.max() + 1, x.min(), x.max() + 1
        yy, xx = np.indices((y1-y0, x1-x0), dtype=np.float32); yy += y0; xx += x0
        mx = (affine[0, 0] * xx + affine[0, 1] * yy + affine[0, 2]).astype(np.float32)
        my = (affine[1, 0] * xx + affine[1, 1] * yy + affine[1, 2]).astype(np.float32)
        valid = mask[y0:y1,x0:x1] & (mx>=0) & (my>=0) & (mx<self.width-1) & (my<self.height-1)
        a = images[source,y0:y1,x0:x1][valid].astype(float)
        b = cv2.remap(images[target],mx,my,cv2.INTER_LINEAR)[valid].astype(float)
        if len(a)<30 or a.std()<1e-5 or b.std()<1e-5:
            return 0.
        a -= a.mean(); b -= b.mean()
        return float(a @ b / max(1e-9, np.linalg.norm(a)*np.linalg.norm(b)))

    def fit(self, source, target, mask):
        options = []; affine = None
        points = cv2.goodFeaturesToTrack(self.tracking[source], maxCorners=180, qualityLevel=.01,
            minDistance=4, mask=ndi.binary_erosion(mask,iterations=2).astype(np.uint8)*255, blockSize=5)
        if points is not None and len(points)>=6:
            params = dict(winSize=(31,31), maxLevel=4, criteria=(3,40,.01))
            end, forward, _ = cv2.calcOpticalFlowPyrLK(self.tracking[source],self.tracking[target],points,None,**params)
            if end is not None:
                back, reverse, _ = cv2.calcOpticalFlowPyrLK(self.tracking[target],self.tracking[source],end,None,**params)
                if back is not None:
                    good = (forward.ravel()>0)&(reverse.ravel()>0)&(np.linalg.norm(back[:,0]-points[:,0],axis=1)<1.5)
                    a,b = points[:,0][good],end[:,0][good]
                    if len(a)>=6:
                        affine, inside = cv2.estimateAffine2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=1500,confidence=.99,refineIters=10)
                        if affine is not None:
                            inside=inside.ravel().astype(bool)
                            hull=cv2.contourArea(cv2.convexHull(a[inside])) if inside.sum()>=3 else 0
                            if inside.sum()>=6 and inside.mean()>=.45 and hull/max(1,mask.sum())>=.08 and self.geometry(affine,mask)[0]:
                                options.append(affine)
        y,x=np.nonzero(mask)
        if len(x):
            x0,y0=max(0,x.min()-16),max(0,y.min()-16)
            section=np.s_[y0:min(self.height,y.max()+17),x0:min(self.width,x.max()+17)]
            initial=np.eye(2,3,dtype=np.float32)
            if affine is not None and self.geometry(affine,mask)[0]:
                initial=affine.astype(np.float32);initial[:,2]+=initial[:,:2]@[x0,y0]-[x0,y0]
            try:
                _,fit=cv2.findTransformECC(self.coarse[source][section],self.coarse[target][section],initial,cv2.MOTION_AFFINE,(3,100,1e-5),None,5)
                _,reverse=cv2.findTransformECC(self.coarse[target][section],self.coarse[source][section],cv2.invertAffineTransform(fit),cv2.MOTION_AFFINE,(3,100,1e-5),None,5)
                fit,reverse=fit.astype(float),reverse.astype(float)
                for matrix in (fit,reverse):matrix[:,2]+=[x0,y0]-matrix[:,:2]@[x0,y0]
                controls=np.array([[x.min(),y.min()],[x.max(),y.min()],[x.max(),y.max()],[x.min(),y.max()],[x.mean(),y.mean()]])
                moved=controls@fit[:,:2].T+fit[:,2]
                cycle=np.max(np.linalg.norm(moved@reverse[:,:2].T+reverse[:,2]-controls,axis=1))
                if self.geometry(fit,mask)[0] and cycle<=4:options.append(fit)
            except cv2.error:
                pass
        candidates=[]
        for fit in options:
            fine=self.ncc(source,target,mask,fit);coarse=self.ncc(source,target,mask,fit,self.coarse)
            if fine>=.25 and coarse>=.45:
                candidates.append(dict(affine=fit,fine=fine,coarse=coarse,determinant=self.geometry(fit,mask)[1]))
        return max(candidates,key=lambda row:.6*row['coarse']+.4*row['fine']) if candidates else None

    def decoys(self, source, target, mask, affine):
        radius=max(40,1.5*np.sqrt(mask.sum()/np.pi));values=[]
        for scale in (1.,1.5):
            for theta in np.linspace(0,2*np.pi,8,endpoint=False):
                wrong=affine.copy();wrong[:,2]+=radius*scale*np.array([np.cos(theta),np.sin(theta)])
                y,x=ndi.center_of_mass(mask);tx,ty=wrong@[x,y,1]
                if 0<=tx<self.width and 0<=ty<self.height:
                    values.append(.7*self.ncc(source,target,mask,wrong)+.3*self.ncc(source,target,mask,wrong,self.coarse))
        return values

    @staticmethod
    def search(source,target,mask,radius=65):
        y,x=np.nonzero(mask)
        if len(x)<120:return []
        h,w=mask.shape;x0,x1=max(0,x.min()-4),min(w,x.max()+5);y0,y1=max(0,y.min()-4),min(h,y.max()+5)
        weights=mask[y0:y1,x0:x1].astype(np.float32);template=source[y0:y1,x0:x1]
        count=weights.sum();center=(template-(template*weights).sum()/count)*weights;var=float((center*center).sum())
        if var<1e-6:return []
        sx,sy=max(0,x0-radius),max(0,y0-radius)
        area=np.ascontiguousarray(target[sy:min(h,y1+radius),sx:min(w,x1+radius)],np.float32)
        total=cv2.matchTemplate(area,weights,cv2.TM_CCORR);squared=cv2.matchTemplate(area*area,weights,cv2.TM_CCORR)
        scores=np.clip(cv2.matchTemplate(area,center,cv2.TM_CCORR)/np.sqrt(np.maximum(1e-12,var*(squared-total*total/count))),-1,1)
        peaks=[]
        for _ in range(4):
            _,score,_,(px,py)=cv2.minMaxLoc(scores)
            if score< -1:break
            peaks.append(dict(dx=int(px+sx-x0),dy=int(py+sy-y0),ncc=float(score)))
            scores[max(0,py-12):py+13,max(0,px-12):px+13]=-2
        return peaks

    def density_fit(self,source,target,mask,domain):
        accepted=[]
        for width,images in self.density.items():
            checked=[]
            for peak in self.search(images[source],images[target],mask):
                affine=np.array([[1.,0,peak['dx']],[0,1.,peak['dy']]])
                if peak['ncc']<.65 or self.ncc(source,target,mask,affine)<.45 or self.ncc(source,target,mask,affine,self.coarse)<.65:continue
                moved=self.warp(mask,affine)&domain
                reverse=self.search(images[target],images[source],moved)
                if reverse and np.hypot(peak['dx']+reverse[0]['dx'],peak['dy']+reverse[0]['dy'])<=3:checked.append(peak)
            if checked and (len(checked)==1 or checked[0]['ncc']-checked[1]['ncc']>=.035):accepted.append(checked[0])
        groups=[[b for b in accepted if np.hypot(a['dx']-b['dx'],a['dy']-b['dy'])<=3] for a in accepted]
        groups=[g for g in groups if len(g)>=2]
        if not groups:return None
        group=max(groups,key=lambda rows:sum(r['ncc'] for r in rows))
        dx,dy=np.median([[r['dx'],r['dy']] for r in group],axis=0)
        affine=np.array([[1.,0,round(dx)],[0,1.,round(dy)]])
        score=.7*self.ncc(source,target,mask,affine)+.3*self.ncc(source,target,mask,affine,self.coarse)
        negatives=self.decoys(source,target,mask,affine)
        if len(negatives)<4 or score-max(negatives)<.1:return None
        return dict(affine=affine,mask=self.warp(mask,affine)&domain,fine=self.ncc(source,target,mask,affine),coarse=self.ncc(source,target,mask,affine,self.coarse),score=score)

    def partial_fit(self,source,target,mask,domain):
        depth=ndi.distance_transform_edt(mask);ys,xs=np.where(depth>=7);points=[]
        for i in np.argsort(depth[ys,xs])[::-1]:
            x,y=int(xs[i]),int(ys[i])
            if all((x-u)**2+(y-v)**2>=64 for u,v in points):points.append((x,y))
            if len(points)>=40:break
        pairs=[]
        for x,y in points:
            if min(x,y)<11 or x+11>=self.width or y+11>=self.height:continue
            section=np.s_[y-11:y+12,x-11:x+12];sx,sy=max(0,x-76),max(0,y-76)
            area=np.s_[sy:min(self.height,y+77),sx:min(self.width,x+77)]
            scores=cv2.matchTemplate(self.fine[target][area],self.fine[source][section],cv2.TM_CCOEFF_NORMED)
            _,score,_,(px,py)=cv2.minMaxLoc(scores);other=scores.copy();other[max(0,py-4):py+5,max(0,px-4):px+5]=-1
            if score<.55 or score-float(other.max())<.035:continue
            tx,ty=sx+px+11,sy+py+11;dest=np.s_[ty-11:ty+12,tx-11:tx+12]
            reverse=cv2.matchTemplate(self.fine[source][area],self.fine[target][dest],cv2.TM_CCOEFF_NORMED)
            _,back,_,(bx,by)=cv2.minMaxLoc(reverse)
            if back<.55 or np.hypot(sx+bx+11-x,sy+by+11-y)>2:continue
            a,b=self.coarse[source][section].ravel(),self.coarse[target][dest].ravel()
            if a.std()<1e-5 or b.std()<1e-5 or np.corrcoef(a,b)[0,1]<.6:continue
            pairs.append(((x,y),(tx,ty)))
        if len(pairs)<4:return None
        a,b=[np.ascontiguousarray(v) for v in np.asarray(pairs,np.float32).transpose(1,0,2)]
        affine,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=3.45)
        if affine is None:return None
        inside=inliers.ravel().astype(bool)
        if inside.sum()<3 or inside.mean()<.6 or cv2.contourArea(cv2.convexHull(a[inside]))<52.9 or not self.geometry(affine,mask)[0]:return None
        visible=np.zeros(mask.shape,np.uint8)
        for x,y in b[inside]:cv2.circle(visible,(round(float(x)),round(float(y))),12,1,-1)
        observed=self.warp(mask,affine)&visible.astype(bool)&domain
        if observed.sum()<120:return None
        return dict(affine=affine,mask=observed,fine=self.ncc(source,target,mask,affine),coarse=self.ncc(source,target,mask,affine,self.coarse),score=float(inside.mean()))


def largest(mask):
    cc,count=ndi.label(mask,np.ones((3,3)))
    if not count:return mask
    sizes=np.bincount(cc.ravel());sizes[0]=0
    return cc==sizes.argmax()


def iou(a,b):
    return float((a&b).sum()/max(1,(a|b).sum()))


def raw_dark_z(image):
    image = image.astype(float)
    mean = ndi.gaussian_filter(image, 7)
    sd = np.sqrt(np.maximum(0, ndi.gaussian_filter(image * image, 7) - mean * mean))
    return (mean - ndi.gaussian_filter(image, .6)) / np.maximum(5, sd)


def raw_material(images,domains):
    output=[]
    for image,domain in zip(images,domains):
        z=raw_dark_z(image)
        cc,_=ndi.label((z>.8)&domain,np.ones((3,3)))
        sizes=np.bincount(cc.ravel());valid=np.flatnonzero(sizes>=4);valid=valid[valid>0]
        output.append(np.isin(cc,valid))
    return np.asarray(output)


def track_territories(images,domains,initial,proposals,*,area_factor=1.3,check=lambda:None):
    cv2.setRNGSeed(19)
    motion=RawMotion(images);n,h,w=images.shape;material=raw_material(images,domains)
    contrast=np.array([g[m].std() for g,m in zip(images,domains)])
    sharpness=np.array([cv2.Laplacian(g,cv2.CV_32F)[m].var() for g,m in zip(images,domains)])
    quality=(contrast>=.5*contrast[0])&(sharpness>=.5*sharpness[0])
    tracks={int(i):dict(mask=initial==i,last=0,belief=.95,partial=False,full=initial==i)
            for i in np.unique(initial[initial>1])}
    frames=[initial.copy()];rows=[];observations=[{i:s['mask'].copy() for i,s in tracks.items()}]
    full_predictions={};partial_frames={};raw_scores=[]
    for t in range(1,n):
        check();options={};controls={}
        for ident,state in tracks.items():
            fit=None if state['partial'] else motion.fit(state['last'],t,state['mask'])
            options[ident]=fit
            if fit is not None and fit['fine']>.4:
                controls.setdefault(state['last'],[]).append((ident,fit['fine']))
        owner=np.zeros((h,w),np.int32);scores=np.full((h,w),-np.inf);current={};partials=[]
        for ident,state in tracks.items():
            check();source=state['last'];old=state['mask'];fit=options[ident]
            proposal=np.zeros((h,w),bool);status='unresolved_correspondence';fine=coarse=belief=None
            guard=False;support_change=False;affine=None
            if fit is not None:
                affine=fit['affine'];fine,coarse=fit['fine'],fit['coarse'];score=.7*fine+.3*coarse
                negatives=motion.decoys(source,t,old,affine)
                positive=np.array([v for other,v in controls.get(source,[]) if other!=ident])
                positive=positive[positive>=np.quantile(positive,.5)] if len(positive) else positive
                if len(negatives)>=4 and len(positive)>=3:
                    control=float(np.median(positive));null_sd=max(.08,float(np.std(negatives)));coherent_sd=max(.18,.2*abs(control))
                    good=-.5*((score-control)/coherent_sd)**2-np.log(coherent_sd)
                    bad=-.5*((score-np.mean(negatives))/null_sd)**2-np.log(null_sd)
                    belief=float(expit(.5*logit(np.clip(state['belief'],.001,.995))+np.clip(good-bad,-12,12)))
                    state['belief']=belief
                    if belief>=.65:
                        predicted=largest(motion.warp(old,affine)&domains[t])
                        if predicted.sum()>=120 and predicted.sum()/max(1,old.sum()*fit['determinant'])>=.6:
                            proposal=predicted;status='observed'
                            candidate=proposals[t]==ident
                            if candidate.sum()>=120:
                                agreement=iou(candidate,predicted)
                                guard=abs(np.log(candidate.sum()/predicted.sum()))>np.log(area_factor)
                                if guard and quality[source] and quality[t]:
                                    old_material=motion.warp(material[source],affine)
                                    removed=predicted&~candidate;added=candidate&~predicted
                                    loss=(old_material&removed).sum()>=30 and (material[t]&removed).sum()<=.5*(old_material&removed).sum()
                                    gain=(material[t]&added).sum()>=30 and (material[t]&added).sum()-(old_material&added).sum()>=.5*(material[t]&added).sum()
                                    common=motion.warp(candidate,cv2.invertAffineTransform(affine))&old
                                    common=ndi.binary_erosion(common,iterations=2)
                                    support_change=bool((loss or gain) and common.sum()>=120 and motion.ncc(source,t,common,affine)>=.6 and agreement>=.3)
                                if support_change:
                                    proposal=candidate;status='observed_support_change'
                                elif not guard and agreement>=.5:
                                    def signed(mask):return ndi.distance_transform_edt(mask)-ndi.distance_transform_edt(~mask)
                                    proposal=largest((.8*signed(predicted)+.2*signed(candidate)>0)&domains[t])
                                elif guard:
                                    status='area_proposal_rejected'
            if not proposal.any():
                recovery=motion.density_fit(source,t,old,domains[t]) if state['partial'] else None
                if recovery is None:
                    recovery=motion.partial_fit(source,t,old,domains[t])
                if recovery is not None:
                    proposal=recovery['mask'];affine=recovery['affine'];fine=recovery['fine'];coarse=recovery['coarse']
                    status='partially_observed';state['partial']=True
                    partial_frames.setdefault(ident,set()).add(t)
                    if (ident,'anchor') not in full_predictions:
                        full_predictions[ident,'anchor']=(t,motion.warp(state['full'],affine),proposal.copy())
                    anchor,full,anchor_mask=full_predictions[ident,'anchor']
                    if t==anchor:
                        full_predictions[ident,t]=full
                    else:
                        peaks=motion.search(motion.coarse[anchor],motion.coarse[t],anchor_mask)
                        if peaks:
                            a=np.array([[1.,0,peaks[0]['dx']],[0,1.,peaks[0]['dy']]])
                            full_predictions[ident,t]=motion.warp(full,a)
            if proposal.any():
                current[ident]=proposal.copy();state.update(mask=proposal.copy(),last=t)
                if not state['partial']:state['full']=proposal.copy()
                depth=ndi.distance_transform_edt(proposal)
                value=depth*(.6+.4*max(0.,fine if fine is not None else 0.))
                take=proposal&(value>scores);owner[take]=ident;scores[take]=value[take]
                if state['partial']:partials.append((ident,proposal))
            rows.append(dict(frame=t,region=ident,source_frame=source,status=status,
                observed_area_px=int(proposal.sum()),fine_ncc=fine,coarse_ncc=coarse,belief=belief,
                area_proposal_rejected=bool(guard),support_change=bool(support_change),partial=state['partial']))
        partial_score=np.full((h,w),-np.inf)
        for ident,mask in partials:
            depth=ndi.distance_transform_edt(mask);value=depth/max(1.,depth.max())
            take=mask&(value>partial_score);owner[take]=ident;partial_score[take]=value[take]
        owner[domains[t]&(owner==0)]=1;owner[~domains[t]]=0
        frames.append(owner);observations.append(current)
    return dict(labels=np.asarray(frames),evidence=rows,observations=observations,
                partial_frames=partial_frames,full_predictions=full_predictions,material=material,motion=motion)


def nearest_distinct(labels):
    h,w=labels.shape;yy,xx=np.indices(labels.shape)
    first=np.full(labels.shape,np.inf);second=first.copy()
    y1=yy.copy();x1=xx.copy();y2=yy.copy();x2=xx.copy()
    for ident in np.unique(labels[labels>1]):
        distance,nearest=ndi.distance_transform_edt(labels!=ident,return_indices=True)
        win=distance<first;runner=(~win)&(distance<second)
        second[win]=first[win];y2[win]=y1[win];x2[win]=x1[win]
        first[win]=distance[win];y1[win]=nearest[0][win];x1[win]=nearest[1][win]
        second[runner]=distance[runner];y2[runner]=nearest[0][runner];x2[runner]=nearest[1][runner]
    with np.errstate(invalid='ignore',divide='ignore'):
        cosine=((yy-y1)*(yy-y2)+(xx-x1)*(xx-x2))/np.maximum(1e-9,first*second)
    return first,second,cosine


def resolve_seams(labels,domain,terrain,*,width=6,gap=10,material=None):
    if not (labels==1).any() or not (labels>1).any():return labels.copy(),dict(changed_px=0)
    d1,d2,cosine=nearest_distinct(labels)
    eligible=(labels==1)&domain&(d1<=width)&(d1+d2<=gap)&(cosine<=-.25)
    if material is not None:eligible&=material>=.15
    markers=np.where(labels>1,labels,0)
    filled=segmentation.watershed(terrain,markers,mask=(labels>1)|eligible,connectivity=2,compactness=.05)
    result=labels.copy();result[eligible&(filled>1)]=filled[eligible&(filled>1)]
    orphan=eligible&(filled<=1)
    if orphan.any():
        _,nearest=ndi.distance_transform_edt(labels<=1,return_indices=True)
        result[orphan]=labels[nearest[0][orphan],nearest[1][orphan]]
    assert np.array_equal(result[labels>1],labels[labels>1])
    assert np.array_equal(result>0,domain)
    remaining=(result==1)&(d1<=width)&(d1+d2<=gap)&(cosine<=-.25)
    return result,dict(changed_px=int((result!=labels).sum()),remaining_geometric_seam_px=int(remaining.sum()),orphan_seam_px=int(orphan.sum()))


def joint_partition(labels,domain,signal,*,pink_radius=5):
    if not (labels==1).any():return labels.copy()
    markers=np.zeros(labels.shape,np.int32)
    for ident in np.unique(labels[labels>1]):
        mask=labels==ident;depth=ndi.distance_transform_edt(mask)
        core=mask&(depth>=max(2,min(6,.5*depth.max())))
        if not core.any():core=mask&(depth==depth.max())
        markers[core]=ident
    pink=labels==1;core=ndi.distance_transform_edt(pink)>=pink_radius
    markers[core]=1
    if not (markers>1).any():return labels.copy()
    relief=ndi.gaussian_filter(signal,18)-signal
    result=segmentation.watershed(relief,markers,mask=domain,connectivity=2,compactness=.002).astype(np.int32)
    allowed=ndi.binary_propagation(core,mask=pink) if core.any() else np.zeros_like(pink)
    spill=(result==1)&~ndi.binary_dilation(allowed,iterations=2)
    if spill.any():
        clean=np.where((result>1),result,0)
        filled=segmentation.watershed(relief,clean,mask=(result>1)|spill,connectivity=2)
        result[spill]=filled[spill]
        orphan=spill&(result==0)
        if orphan.any():
            _,nearest=ndi.distance_transform_edt(clean==0,return_indices=True)
            result[orphan]=clean[nearest[0][orphan],nearest[1][orphan]]
    assert np.array_equal(result>0,domain)
    return result


def structure_fields(raw,outer,scale,*,check=lambda: None):
    def smooth(width):
        return np.array([ndi.gaussian_filter(g.astype(np.float32)*m,width)/
                         np.maximum(1e-6,ndi.gaussian_filter(m.astype(np.float32),width)) for g,m in zip(raw,outer)])
    contrast=smooth(scale)-smooth(max(22,4*scale));median=[];mad=[];agreement=[]
    for t in range(len(raw)):
        check()
        window=contrast[max(0,t-2):min(len(raw),t+3)]
        center=np.median(window,axis=0)
        median.append(center);mad.append(1.4826*np.median(np.abs(window-center),axis=0))
        agreement.append(np.mean(np.sign(window)==np.sign(center),axis=0))
    return np.array(median),np.array(mad),np.array(agreement)


def proposals(fields,coverage,pink,floor=5,ratio=2.):
    median,mad,agreement=fields;observations=[]
    for t in range(len(pink)):
        stable=(np.abs(median[t])>=floor)&(np.abs(median[t])>=ratio*(mad[t]+2))&(agreement[t]>=.8)&(coverage[t]>=.15)&pink[t]
        stable=ndi.binary_opening(stable,structure=morphology.disk(2))
        stable=ndi.binary_closing(stable,structure=morphology.disk(3))&pink[t]&(coverage[t]>=.15)
        items=[]
        for sign in (-1,1):
            allowed=stable&(median[t]*sign>0)
            markers,count=ndi.label(morphology.h_maxima(sign*median[t],3)&allowed,np.ones((3,3)))
            if not count:markers,_=ndi.label(allowed,np.ones((3,3)))
            labels=segmentation.watershed(-sign*median[t],markers,mask=allowed,connectivity=2)
            for region in measure.regionprops(labels):
                mask=labels==region.label
                if 180<=region.area<=5000 and 2*ndi.distance_transform_edt(mask).max()>=12:
                    items.append(dict(t=t,mask=mask,area=int(region.area),center=np.array(region.centroid),polarity=sign))
        observations.append(items)
    return observations


def consensus_tracks(fields,coverage,pink,floor=5,ratio=2.):
    scales=[proposals(field,coverage,pink,floor,ratio) for field in fields];observations=[]
    for t in range(len(pink)):
        selected=[]
        for item in scales[1][t]:
            peers=[]
            for scale in (0,2):
                options=[(iou(item['mask'],other['mask']),other) for other in scales[scale][t] if item['polarity']==other['polarity']]
                if options:
                    score,peer=max(options,key=lambda x:x[0])
                    if score>=.35:peers.append(peer['mask'])
            if not peers:continue
            votes=np.sum([item['mask']]+peers,axis=0)
            mask=largest((votes>=(2 if len(peers)==2 else 1))&pink[t])
            if 180<=mask.sum()<=5000 and 2*ndi.distance_transform_edt(mask).max()>=12:
                selected.append(dict(item,mask=mask,area=int(mask.sum()),center=np.array(ndi.center_of_mass(mask))))
        observations.append(selected)
    tracks=[]
    for t,items in enumerate(observations):
        active=[i for i,tr in enumerate(tracks) if 1<=t-tr[-1]['t']<=2]
        costs=np.full((len(active),len(items)),1e6)
        for a,k in enumerate(active):
            old=tracks[k][-1]
            for b,item in enumerate(items):
                elapsed=t-old['t'];limit=min(5,.3*np.sqrt(old['area']/np.pi))*elapsed;score=iou(old['mask'],item['mask'])
                if old['polarity']==item['polarity'] and np.linalg.norm(old['center']-item['center'])<=limit and 1/1.5<=item['area']/old['area']<=1.5 and score>=.4:costs[a,b]=1-score
        used=set()
        if costs.size:
            aa,bb=linear_sum_assignment(costs)
            for a,b in zip(aa,bb):
                if costs[a,b]<1e5:tracks[active[a]].append(items[b]);used.add(b)
        tracks.extend([item] for i,item in enumerate(items) if i not in used)
    return [tr for tr in tracks if len(tr)>=5 and np.quantile([r['area'] for r in tr],.9)/np.quantile([r['area'] for r in tr],.1)<=1.8]


def add_structures(raw,outer,material,baseline,*,check=lambda: None):
    pink=baseline==1
    coverage=np.array([ndi.uniform_filter(m.astype(float),25)/np.maximum(1e-6,ndi.uniform_filter(o.astype(float),25)) for m,o in zip(material,outer)])
    fields=[structure_fields(raw,outer,scale,check=check) for scale in (5,8,12)]
    trials=[consensus_tracks(fields,coverage,pink,floor,ratio) for floor,ratio in ((5,2.),(4,1.5),(7,2.5))]
    def volume(track):
        mask=np.zeros_like(pink)
        for row in track:mask[row['t']]=row['mask']
        return mask
    alternatives=[[volume(track) for track in group] for group in trials[1:]]
    labels=baseline.copy();next_id=int(labels.max())+1;events=[]
    for track in trials[0]:
        mask=volume(track)
        agreement=[max((iou(mask,other) for other in group),default=0.) for group in alternatives]
        if min(agreement)<.5:continue
        ident=next_id;next_id+=1
        labels[mask&(labels==1)]=ident
        previous=None
        median,mad,sign=fields[1]
        for t in range(track[0]['t'],len(labels)):
            observed=labels[t]==ident
            if not observed.any() and previous is not None:
                stable=(np.abs(median[t])>=5)&(np.abs(median[t])>=2*(mad[t]+2))&(sign[t]>=.8)
                stable&=(median[t]*track[0]['polarity']>0)&(coverage[t]>=.15)
                support=stable&(labels[t]==1)&ndi.binary_dilation(previous,iterations=4)
                support=ndi.binary_closing(support,iterations=2)&(labels[t]==1)
                factor=support.sum()/max(1,previous.sum())
                if support.sum()>=120 and iou(support,previous)>=.5 and .6<=factor<=1.6:
                    labels[t][support]=ident;observed=support
            if observed.any():previous=observed
        events.append(dict(region=ident,kind='persistent_structure',first_frame=track[0]['t'],confirmation_frame=track[4]['t'],
            observed_frames=[t for t in range(len(labels)) if (labels[t]==ident).any()],parameter_agreement=agreement,
            polarity=track[0]['polarity']))
    return labels,events


def merge_fragments(labels,new_ids,partials,full_predictions,motion,maximum_gap=2,*,check=lambda: None):
    reports=[];accepted=[]
    for parent,times in partials.items():
        check()
        for candidate in new_ids:
            geometry=[];intervals=[]
            visible=[t for t in times if (labels[t]==parent).any() and (labels[t]==candidate).any()]
            for t in sorted(visible):
                a,b=labels[t]==parent,labels[t]==candidate;prior=full_predictions.get((parent,t))
                support=float((ndi.binary_dilation(prior,iterations=5)&b).sum()/b.sum()) if prior is not None else 0.
                distance=ndi.distance_transform_edt(~a)
                geometry.append(dict(frame=t,contact=int((ndi.binary_dilation(a,structure=np.ones((3,3)))&b).sum()),
                    proximity_pixels=int(((distance<=maximum_gap)&b).sum()),distance=float(distance[b].min()),parent_support=support))
            contact=[r for r in geometry if r['proximity_pixels']>=5 and r['distance']<=maximum_gap];frames={r['frame'] for r in contact}
            for t,u in zip(sorted(visible),sorted(visible)[1:]):
                if u-t!=1 or t not in frames or u not in frames:continue
                a,b=labels[t]==parent,labels[t]==candidate
                fits=[motion.search(motion.coarse[t],motion.coarse[u],mask,radius=12) for mask in (a,b,a|b)]
                if not all(fits):continue
                fit=fits[2][0];matrix=np.array([[1.,0,fit['dx']],[0,1.,fit['dy']]])
                scores=[motion.ncc(t,u,mask,matrix,motion.coarse) for mask in (a,b)]
                delta=(np.array(ndi.center_of_mass(labels[u]==candidate))-ndi.center_of_mass(labels[u]==parent))-(np.array(ndi.center_of_mass(b))-ndi.center_of_mass(a))
                intervals.append(dict(source=t,target=u,parent_ncc=scores[0],candidate_ncc=scores[1],penalty=max(fits[0][0]['ncc']-scores[0],fits[1][0]['ncc']-scores[1]),relative_step=float(np.linalg.norm(delta))))
            good=[r for r in intervals if min(r['parent_ncc'],r['candidate_ncc'])>=.65 and r['penalty']<=.1 and r['relative_step']<=4]
            prior=float(np.median([r['parent_support'] for r in contact])) if contact else 0.
            take=len(contact)>=4 and len(contact)/max(1,len(geometry))>=.6 and prior>=.5 and len(good)>=3 and len(good)/max(1,len(intervals))>=.75
            report=dict(parent=int(parent),candidate=int(candidate),accepted=bool(take),maximum_gap=maximum_gap,geometry=geometry,intervals=intervals,good_intervals=len(good),parent_support=prior,first_supported_frame=min(frames) if frames else None)
            reports.append(report)
            if take:accepted.append(report)
    result=labels.copy();events=[]
    for candidate in new_ids:
        options=sorted([r for r in accepted if r['candidate']==candidate],key=lambda r:r['parent_support'],reverse=True)
        if not options or len(options)>1 and options[0]['parent_support']-options[1]['parent_support']<.1:continue
        row=options[0]
        for t in range(row['first_supported_frame'],len(labels)):result[t][result[t]==candidate]=row['parent']
        events.append(dict(parent=row['parent'],alias=candidate,first_frame=row['first_supported_frame'],good_intervals=row['good_intervals']))
    return result,events,reports


def remove_narrow_pink(labels,domain,terrain,radius):
    pink=labels==1
    if not pink.any() or not (labels>1).any():return labels.copy(),0
    broad=ndi.binary_opening(pink,structure=morphology.disk(radius))
    narrow=pink&~broad
    markers=np.where(labels>1,labels,0)
    flooded=segmentation.watershed(terrain,markers,mask=(labels>1)|narrow,connectivity=2)
    result=labels.copy();take=narrow&(flooded>1);result[take]=flooded[take]
    orphan=narrow&~take
    if orphan.any():
        _,nearest=ndi.distance_transform_edt(labels<=1,return_indices=True)
        result[orphan]=labels[nearest[0][orphan],nearest[1][orphan]]
    assert np.array_equal(result[labels>1],labels[labels>1])
    assert np.array_equal(result>0,domain)
    return result,int(narrow.sum())


def absorb_enclosed_pockets(labels,domain,terrain,protected=None):
    components,count=ndi.label(labels==1,np.ones((3,3)));result=labels.copy()
    if not count or not (labels>1).any():return result,0
    distance=ndi.distance_transform_edt(labels<=1);selected=np.zeros_like(domain)
    protected=np.zeros_like(domain) if protected is None else protected
    boundary=domain&~ndi.binary_erosion(domain)
    for ident in range(1,count+1):
        mask=components==ident
        if mask.sum()<=750 and distance[mask].max()<=12 and not (mask&boundary).any() and not (mask&protected).any():
            selected|=mask
    if selected.any():
        markers=np.where(labels>1,labels,0)
        filled=segmentation.watershed(terrain,markers,mask=(labels>1)|selected,connectivity=2)
        result[selected]=filled[selected]
        orphan=selected&(result==0)
        if orphan.any():
            _,nearest=ndi.distance_transform_edt(labels<=1,return_indices=True)
            result[orphan]=labels[nearest[0][orphan],nearest[1][orphan]]
    assert np.array_equal(result>0,domain)
    assert np.array_equal(result[labels>1],labels[labels>1])
    return result,int(selected.sum())


def local_ncc(a,b,mask,width=23):
    weight=ndi.uniform_filter(mask.astype(float),width)
    def mean(values):return ndi.uniform_filter(np.where(mask,values,0).astype(float),width)/np.maximum(1e-6,weight)
    ma,mb=mean(a),mean(b)
    covariance=mean(a*b)-ma*mb
    variance=np.sqrt(np.maximum(0,mean(a*a)-ma*ma)*np.maximum(0,mean(b*b)-mb*mb))
    return np.divide(covariance,variance,out=np.zeros_like(covariance),where=variance>1e-6)


def evidence(images,outer,labels,*,check=lambda: None):
    model=RawMotion(images);n=len(images);positive=[np.zeros(images.shape[1:],bool)]
    sustained=[positive[0].copy()];maps=[None];rows=[]
    for t in range(1,n):
        check()
        motion=flow_pair(images[t-1],images[t],outer[t-1],outer[t]);maps.append(motion)
        correlation=local_ncc(motion['remap'](model.coarse[t-1]),model.coarse[t],outer[t])
        backward=motion['backward']
        residual=np.hypot(backward[...,0]-ndi.gaussian_filter(backward[...,0],15),backward[...,1]-ndi.gaussian_filter(backward[...,1],15))
        controls=motion['valid']&(labels[t]>1)&(correlation>=.65)
        threshold=max(.5,float(np.quantile(residual[controls],.95))) if controls.sum()>=120 else np.inf
        image=images[t].astype(float)
        texture=np.sqrt(np.maximum(0,ndi.uniform_filter(image*image,23)-ndi.uniform_filter(image,23)**2))
        fraction=ndi.uniform_filter((motion['valid']&(residual>threshold)).astype(float),31)
        current=motion['valid']&(residual>threshold)&(correlation<.45)&(fraction>.25)&(texture>=5)&outer[t]
        positive.append(current)
        votes=current.astype(int)+(motion['remap'](positive[t-1].astype(np.float32))>.5)
        if t>=2:
            previous=maps[t-1]['remap'](positive[t-2].astype(np.float32))
            votes+=motion['remap'](previous)>.5
        mask=ndi.binary_opening((votes>=2)&outer[t],structure=morphology.disk(3))
        mask=morphology.remove_small_objects(mask,min_size=180)
        sustained.append(mask)
        pink=labels[t]==1
        rows.append(dict(frame=t,pink_area_px=int(pink.sum()),motion_supported_pink_px=int((pink&mask).sum()),
            checked_flow_fraction_in_pink=float(motion['valid'][pink].mean()) if pink.any() else None,
            motion_supported_pink_fraction=float((pink&mask).sum()/max(1,pink.sum())),
            residual_threshold_px=threshold if np.isfinite(threshold) else None,control_pixels=int(controls.sum())))
    return np.array(sustained),rows


def localize_residual(labels,domain,terrain,*,radius=5,interface=12):
    labels=np.asarray(labels,np.int32);pink=labels==1
    if not pink.any() or not (labels>1).any():
        return labels.copy(),dict(reassigned_px=0,core_area_px=0)
    broad=ndi.binary_opening(pink,structure=morphology.disk(radius))
    cc,count=ndi.label(broad,np.ones((3,3)))
    if not count:
        return labels.copy(),dict(reassigned_px=0,core_area_px=0,status='no_broad_compartment')
    sizes=np.bincount(cc.ravel());sizes[0]=0;core=cc==sizes.argmax()
    # Select the dominant broad compartment; proximity permits parts separated by retained regions.
    distance=ndi.distance_transform_edt(~core)
    retained=pink&(distance<=interface)
    removed=pink&~retained
    markers=np.where(labels>1,labels,0)
    filled=segmentation.watershed(terrain,markers,mask=(labels>1)|removed,connectivity=2,compactness=.05)
    result=labels.copy();take=removed&(filled>1);result[take]=filled[take]
    orphan=removed&~take
    if orphan.any():
        _,nearest=ndi.distance_transform_edt(labels<=1,return_indices=True)
        result[orphan]=labels[nearest[0][orphan],nearest[1][orphan]]
    assert np.array_equal(result[labels>1],labels[labels>1])
    assert np.array_equal(result>0,domain)
    return result,dict(reassigned_px=int(removed.sum()),core_area_px=int(core.sum()),orphan_px=int(orphan.sum()),status='localized')


def finish_shared_edges(labels,domain,terrain,*,max_passes=64):
    result=labels.copy();pink=labels==1
    protected=ndi.distance_transform_edt(pink)>=6
    changes=[]
    for _ in range(max_passes):
        proposed,audit=resolve_seams(result,domain,terrain)
        proposed[protected]=1
        changed=int((proposed!=result).sum());changes.append(changed);result=proposed
        if changed==0:break
    if (result==1).any() and (result>1).any():
        d1,d2,cosine=nearest_distinct(result)
        residual=(result==1)&domain&(d1<=6)&(d1+d2<=10)&(cosine<=-.25)
    else:residual=np.zeros_like(domain)
    assert np.array_equal(result[labels>1],labels[labels>1])
    assert np.array_equal(result>0,domain)
    return result,dict(passes=len(changes),changed_px=sum(changes),changes=changes,
        remaining_unprotected_seam_px=int((residual&~protected).sum()),protected_broad_pink_interface_px=int((residual&protected).sum()),
        converged=changes[-1]==0)


def smooth_joint(lab,roi,sigma=2.5,band=6,tolerance=.08):
    """All IDs compete over one interface; protect cores and local topology."""
    ids=np.unique(lab[lab>0]);h,w=lab.shape;fixed=np.zeros(lab.shape,bool);scores=[]
    for ident in ids:
        mask=lab==ident
        d=ndi.distance_transform_edt(mask)-ndi.distance_transform_edt(~mask)
        field=ndi.gaussian_filter(np.clip(d,-20,20),sigma)
        # Compensate systematic area shrinkage before joint competition.
        field-=float(np.quantile(field[roi],1-mask.sum()/roi.sum()))
        field[d < -band]=-1e6
        fixed|=d>=band
        cc,n=ndi.label(mask,STRUCTURE)
        for i,sl in enumerate(ndi.find_objects(cc),1):
            part=np.where(cc[sl]==i,d[sl],-np.inf);y,x=np.unravel_index(part.argmax(),part.shape);fixed[y+sl[0].start,x+sl[1].start]=True
        scores.append(field.astype(np.float32))
    scores=np.stack(scores);index=np.zeros(int(ids.max())+1,int);index[ids]=np.arange(len(ids))
    winner=ids[scores.argmax(axis=0)]
    gain=scores.max(axis=0)-np.take_along_axis(scores,index[lab][None],axis=0)[0]
    eligible=roi&~fixed&(winner!=lab)&(gain>0)
    coords=np.argwhere(eligible);coords=coords[np.argsort(gain[eligible])[::-1]]
    out=lab.copy();areas=np.bincount(out.ravel(),minlength=int(ids.max())+1);initial=areas.copy();changes=0
    for sweep in range(3):
        changed=0
        for y,x in coords:
            old=int(out[y,x]);new=int(winner[y,x])
            if old==new or min(y,x)==0 or y==h-1 or x==w-1:continue
            if areas[old]-1<initial[old]*(1-tolerance) or areas[new]+1>initial[new]*(1+tolerance):continue
            patch=out[y-1:y+2,x-1:x+2]
            remove=patch==old;remove[1,1]=False;add=patch==new;add[1,1]=False
            if ndi.label(remove,STRUCTURE)[1]!=1 or ndi.label(add,STRUCTURE)[1]!=1:continue
            out[y,x]=new;areas[old]-=1;areas[new]+=1;changed+=1
        changes+=changed
        if not changed:break
    assert np.array_equal(out>0,roi)
    assert np.array_equal(out[fixed],lab[fixed])
    assert np.array_equal(np.unique(out),np.unique(lab))
    for ident in ids:
        assert ndi.label(out==ident,STRUCTURE)[1]==ndi.label(lab==ident,STRUCTURE)[1]
        assert abs((out==ident).sum()/(lab==ident).sum()-1)<=tolerance+1e-8
    return out,changes


def resolve_edge_collars(labels,domain,supported,*,width_fraction=.3,protected=None,edge_candidates=None):
    result=labels.copy();pink=labels==1
    if not pink.any() or not supported:return result,dict(changed_px=0,by_region={})
    exterior_distance,exterior_index=ndi.distance_transform_edt(domain,return_indices=True)
    colony_distance,colony_index=ndi.distance_transform_edt(labels<=1,return_indices=True)
    neighbor=labels[colony_index[0],colony_index[1]]
    areas=np.bincount(labels.ravel());radii=np.sqrt(areas/np.pi)
    edge=domain&ndi.binary_dilation(~domain,structure=np.ones((3,3)))
    edge_counts=np.bincount(labels[edge],minlength=len(areas))
    edge_candidates=set(edge_candidates) if edge_candidates is not None else {int(i) for i in supported if i<len(edge_counts) and edge_counts[i]>=max(5,.1*radii[i])}
    supported=set(supported)&edge_candidates
    width=width_fraction*radii[neighbor]
    yy,xx=np.indices(labels.shape)
    dot=(yy-colony_index[0])*(yy-exterior_index[0])+(xx-colony_index[1])*(xx-exterior_index[1])
    cosine=dot/np.maximum(1e-9,colony_distance*exterior_distance)
    eligible=pink&domain&np.isin(neighbor,list(supported))&(colony_distance<=width)&(colony_distance+exterior_distance<=2*width)&(cosine<=-.25)
    if protected is not None:eligible&=~protected
    backside=np.zeros_like(domain);cores={}
    for ident in supported:
        radius=max(2,round(.4*radii[ident]))
        broad=ndi.binary_opening(pink,structure=morphology.disk(radius))
        cc,count=ndi.label(broad,np.ones((3,3)))
        if not count:continue
        sizes=np.bincount(cc.ravel());sizes[0]=0;core=cc==sizes.argmax()
        cores[ident]=broad
        _,nearest=ndi.distance_transform_edt(~core,return_indices=True)
        cy,cx=ndi.center_of_mass(labels==ident);iy,ix=int(round(cy)),int(round(cx))
        vy,vx=nearest[0,iy,ix]-cy,nearest[1,iy,ix]-cx
        backside|=(neighbor==ident)&(((yy-cy)*vy+(xx-cx)*vx)<=0)&~core
    eligible&=backside
    # Only proposals attached to the supporting territory may extend its visible boundary.
    take=np.zeros_like(domain)
    for ident in np.unique(neighbor[eligible]):
        extension=eligible&(neighbor==ident)
        attached=ndi.binary_propagation(labels==ident,mask=(labels==ident)|extension,structure=np.ones((3,3)))
        take|=extension&attached
    result[take]=neighbor[take]
    # A severed collar tip remains residual ownership, not a new independent flow site.
    before_cc,_=ndi.label(pink,np.ones((3,3)))
    after_cc,count=ndi.label(result==1,np.ones((3,3)))
    anchors=np.zeros_like(domain)
    for ident in np.unique(neighbor[take]):anchors|=cores[int(ident)]
    anchored_parents=set(np.unique(before_cc[anchors&pink]))-{0}
    severed=np.zeros_like(domain)
    for k in range(1,count+1):
        part=after_cc==k
        if (part&anchors).any():continue
        if protected is not None and (part&protected).any():continue
        parents=set(np.unique(before_cc[part]))-{0}
        if parents&anchored_parents:severed|=part
    if severed.any():
        from skimage.segmentation import watershed
        markers=np.where(result>1,result,0)
        filled=watershed(np.zeros_like(result,float),markers,mask=(result>1)|severed,connectivity=2)
        assigned=severed&(filled>1);result[assigned]=filled[assigned]
        orphan=severed&~assigned
        if orphan.any():
            _,near=ndi.distance_transform_edt(markers==0,return_indices=True)
            result[orphan]=markers[near[0][orphan],near[1][orphan]]
    assert np.array_equal(result[labels>1],labels[labels>1])
    assert np.array_equal(result>0,domain)
    changed=result!=labels
    return result,dict(changed_px=int(changed.sum()),collar_px=int(take.sum()),severed_tip_px=int(severed.sum()),
        by_region={str(int(i)):int((changed&(result==i)).sum()) for i in np.unique(result[changed])},supported_edge_regions=sorted(supported))


def texture_continuity(raw, labels, dark, *, check=lambda: None):
    at_raw = raw
    at_lab = labels
    at_dark = dark
    at_n, at_h, at_w = labels.shape
    at_ids = [int(i) for i in np.unique(labels[0]) if i > 1]
    at_clahe = cv2.createCLAHE(2.0, (8, 8))
    at_images = np.stack([at_clahe.apply(cv2.GaussianBlur(g, (0, 0), 0.8)) for g in at_raw])
    at_norm = []
    for g in at_raw:
        a = g.astype(np.float32)
        mu = ndi.gaussian_filter(a, 12)
        sd = np.sqrt(np.maximum(0, ndi.gaussian_filter(a * a, 12) - mu * mu))
        at_norm.append((a - mu) / np.maximum(sd, 5))
    at_norm = np.stack(at_norm)
    at_params = dict(max_points_per_colony=12, seed_tile_px=20, seed_min_distance_px=7, forward_backward_px=1.0, patch_ncc=0.6, max_step_px=120.0, lk_window_px=25, lk_pyramid_max_level=4, trail_positions=6)

    def at_corr(a, b):
        a = a.ravel().astype(float)
        b = b.ravel().astype(float)
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-06 and b.std() > 1e-06 else -1.0

    def at_match(src, dst, srcnorm, dstnorm, p, level=4):
        q, st, err = cv2.calcOpticalFlowPyrLK(src, dst, p.reshape(-1, 1, 2).astype(np.float32), None, winSize=(25, 25), maxLevel=level, criteria=(3, 50, 0.01), minEigThreshold=0.0001)
        if q is None:
            return (p.copy(), [dict(accepted=False, reason='optical_flow_failed') for _ in p])
        finite = np.isfinite(q).all(axis=(1, 2))
        q = np.where(finite[:, None, None], q, p.reshape(-1, 1, 2)).astype(np.float32)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(dst, src, q, None, winSize=(25, 25), maxLevel=level, criteria=(3, 50, 0.01), minEigThreshold=0.0001)
        if back is None:
            return (p.copy(), [dict(accepted=False, reason='optical_flow_failed') for _ in p])
        q = q[:, 0]
        back = back[:, 0]
        fb = np.linalg.norm(back - p, axis=1)
        rows = []
        for j, (x, y) in enumerate(p):
            tx, ty = q[j]
            reason = 'accepted'
            fine = raw = -1.0
            if not finite[j] or not np.isfinite(back[j]).all() or not st[j, 0] or not st2[j, 0]:
                reason = 'optical_flow_failed'
            elif not (4 <= tx < src.shape[1] - 4 and 4 <= ty < src.shape[0] - 4):
                reason = 'left_image'
            elif fb[j] > 1:
                reason = 'forward_backward_inconsistent'
            elif np.linalg.norm(q[j] - p[j]) > 120:
                reason = 'excessive_step'
            else:
                fine = at_corr(cv2.getRectSubPix(src, (9, 9), (float(x), float(y))), cv2.getRectSubPix(dst, (9, 9), (float(tx), float(ty))))
                raw = at_corr(cv2.getRectSubPix(srcnorm, (9, 9), (float(x), float(y))), cv2.getRectSubPix(dstnorm, (9, 9), (float(tx), float(ty))))
                if min(fine, raw) < 0.6:
                    reason = 'patch_appearance_changed'
            rows.append(dict(accepted=reason == 'accepted', reason=reason, forward_backward_error_px=float(fb[j]), enhanced_patch_ncc=fine, normalized_raw_patch_ncc=raw))
        if level > 0:
            bad = np.array([not r['accepted'] for r in rows])
            if bad.any():
                retry, checks = at_match(src, dst, srcnorm, dstnorm, p[bad], level=max(0, level - 2))
                for j, pos, check in zip(np.flatnonzero(bad), retry, checks):
                    if check['accepted']:
                        q[j] = pos
                        rows[j] = check
        return (q, rows)
    at_seeds = []
    for ident in at_ids:
        m = at_lab[0] == ident
        seedmask = m & at_dark[0]
        seedmask[:5] = False
        seedmask[-5:] = False
        seedmask[:, :5] = False
        seedmask[:, -5:] = False
        if seedmask.sum() < 4:
            seedmask = m.copy()
            seedmask[:5] = False
            seedmask[-5:] = False
            seedmask[:, :5] = False
            seedmask[:, -5:] = False
        corners = cv2.goodFeaturesToTrack(at_images[0], maxCorners=150, qualityLevel=0.005, minDistance=4, mask=seedmask.astype(np.uint8) * 255, blockSize=5)
        if corners is not None:
            pts = corners[:, 0]
        else:
            pts = np.argwhere(seedmask)[:, ::-1].astype(np.float32)
        if not len(pts):
            continue
        center = np.array(ndi.center_of_mass(m)[::-1])
        central = int(np.argmin(np.linalg.norm(pts - center, axis=1)))
        chosen = [pts[central]]
        tiles = {tuple((pts[central] // 20).astype(int))}
        for p in pts:
            tile = tuple((p // 20).astype(int))
            if tile not in tiles and min((np.linalg.norm(p - c) for c in chosen)) >= 7:
                chosen.append(p)
                tiles.add(tile)
            if len(chosen) >= 12:
                break
        if len(chosen) < min(4, len(pts)):
            for p in pts:
                if min((np.linalg.norm(p - c) for c in chosen)) >= 7:
                    chosen.append(p)
                if len(chosen) >= 4:
                    break
        for j, p in enumerate(chosen):
            at_seeds.append(dict(track_id=len(at_seeds), origin_colony_id=ident, central_seed=j == 0, x=float(p[0]), y=float(p[1])))
    at_paths = {r['track_id']: [(0, np.array([r['x'], r['y']], np.float32))] for r in at_seeds}
    at_active = list(at_paths)
    at_seedmap = {r['track_id']: r for r in at_seeds}
    at_rows = []
    at_failures = []
    at_counts = [len(at_active)]
    for r in at_seeds:
        at_rows.append(dict(capture=0, track_id=r['track_id'], origin_colony_id=r['origin_colony_id'], central_seed=r['central_seed'], source_x_px=np.nan, source_y_px=np.nan, x_px=r['x'], y_px=r['y'], dx_px=np.nan, dy_px=np.nan, displacement_px=np.nan, current_region_id=r['origin_colony_id'], in_dark_mask=bool(at_dark[0, int(round(r['y'])), int(round(r['x']))]), forward_backward_error_px=np.nan, enhanced_patch_ncc=np.nan, normalized_raw_patch_ncc=np.nan))
    for k in range(at_n - 1):
        check()
        if not at_active:
            at_counts.append(0)
            continue
        p = np.stack([at_paths[t][-1][1] for t in at_active])
        q, quality = at_match(at_images[k], at_images[k + 1], at_norm[k], at_norm[k + 1], p)
        nextactive = []
        for t, pt, qt, qual in zip(at_active, p, q, quality):
            seed = at_seedmap[t]
            if not qual['accepted']:
                at_failures.append(dict(track_id=t, origin_colony_id=seed['origin_colony_id'], target_capture=k + 1, last_observed_capture=k, last_x_px=float(pt[0]), last_y_px=float(pt[1]), **qual))
                continue
            assert at_paths[t][-1][0] == k
            at_paths[t].append((k + 1, qt.copy()))
            nextactive.append(t)
            x, y = np.rint(qt).astype(int)
            at_rows.append(dict(capture=k + 1, track_id=t, origin_colony_id=seed['origin_colony_id'], central_seed=seed['central_seed'], source_x_px=float(pt[0]), source_y_px=float(pt[1]), x_px=float(qt[0]), y_px=float(qt[1]), dx_px=float(qt[0] - pt[0]), dy_px=float(qt[1] - pt[1]), displacement_px=float(np.linalg.norm(qt - pt)), current_region_id=int(at_lab[k + 1, y, x]), in_dark_mask=bool(at_dark[k + 1, y, x]), forward_backward_error_px=qual['forward_backward_error_px'], enhanced_patch_ncc=qual['enhanced_patch_ncc'], normalized_raw_patch_ncc=qual['normalized_raw_patch_ncc']))
        at_active = nextactive
        at_counts.append(len(at_active))
    ct_old = labels
    ct_new = labels.copy()
    ct_n, ct_h, ct_w = labels.shape
    ct_initial = at_ids
    ct_all = [int(i) for i in np.unique(labels) if i > 1]
    ct_points = at_rows
    ct_pts = {}
    ct_origin = {}
    ct_p0 = {}
    for row in ct_points:
        k = int(row['capture'])
        t = int(row['track_id'])
        position = np.array([float(row['x_px']), float(row['y_px'])])
        ct_pts[k, t] = position
        ct_origin[t] = int(row['origin_colony_id'])
        if k == 0:
            ct_p0[t] = position
    ct_first = {i: next((k for k in range(ct_n) if (ct_old[k] == i).any())) for i in ct_all}
    ct_warp = lambda m, A: cv2.warpAffine(m.astype(np.uint8), A, (ct_w, ct_h), flags=cv2.INTER_NEAREST) > 0
    ct_predcache = {}

    def ct_predict(origin, k):
        key = (origin, k)
        if key in ct_predcache:
            return ct_predcache[key]
        ids = [t for t in ct_origin if ct_origin[t] == origin and (k, t) in ct_pts and (t in ct_p0)]
        result = None
        if len(ids) >= 3:
            p = np.stack([ct_p0[t] for t in ids]).astype(np.float32)
            q = np.stack([ct_pts[k, t] for t in ids]).astype(np.float32)
            A, ins = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=2000, confidence=0.99)
            if A is not None:
                inside = ins.ravel().astype(bool)
                sv = np.linalg.svd(A[:, :2], compute_uv=False)
                err = np.linalg.norm(p @ A[:, :2].T + A[:, 2] - q, axis=1)
                valid = inside.sum() >= 3 and inside.mean() >= 0.6 and (np.median(err[inside]) <= 1.5) and (sv.min() >= 0.6) and (sv.max() <= 1.6)
                if valid:
                    result = dict(mask=ct_warp(ct_old[0] == origin, A), A=A, inliers=int(inside.sum()), inlier_fraction=float(inside.mean()), median_residual_px=float(np.median(err[inside])), point_ids=[ids[j] for j in np.flatnonzero(inside)])
        ct_predcache[key] = result
        return result
    ct_audit = []
    ct_alias = {}
    for ident in sorted(set(ct_all) - set(ct_initial)):
        first = ct_first[ident]
        frames = [k for k in range(first, ct_n) if (ct_old[k] == ident).any()][:3]
        votes = collections.defaultdict(set)
        for r in ct_points:
            if int(r['capture']) in frames and int(r['current_region_id']) == ident:
                votes[int(r['origin_colony_id'])].add(int(r['track_id']))
        ranking = sorted([(len(v), i) for i, v in votes.items()], reverse=True)
        top, parent = ranking[0] if ranking else (0, 0)
        runner = ranking[1][0] if len(ranking) > 1 else 0
        dominance = top / max(1, sum((n for n, _ in ranking)))
        m = ct_old[first] == ident
        bestoverlap = 0.0
        last = -1
        if parent:
            prior = [k for k in range(first) if (ct_old[k] == parent).any()]
            if prior:
                last = prior[-1]
                oldm = ct_old[last] == parent
                bestoverlap = float((oldm & m).sum() / max(1, min(oldm.sum(), m.sum())))
            pred = ct_predict(parent, first)
            if pred is not None:
                bestoverlap = max(bestoverlap, float((pred['mask'] & m).sum() / max(1, min(pred['mask'].sum(), m.sum()))))
        accept = bool(parent in ct_initial and top >= 3 and (dominance >= 0.75) and (top - runner >= 2) and (bestoverlap >= 0.5))
        if accept:
            ct_alias[ident] = parent
            ct_new[ct_old == ident] = parent
        ct_audit.append(dict(source_label=ident, first_capture=first, candidate_parent=parent if parent else '', accepted=accept, unique_supporting_tracks=top, runner_up_tracks=runner, origin_vote_fraction=dominance, max_spatial_overlap_fraction=bestoverlap, last_prior_parent_capture=last, decision='inherit_parent_id_and_frame0_baseline' if accept else 'unresolved_parent', vote_counts=json.dumps({str(i): len(v) for i, v in votes.items()})))
    ct_recovered = []
    for k in range(1, ct_n):
        check()
        for origin in ct_initial:
            if (ct_new[k] == origin).any():
                continue
            pred = ct_predict(origin, k)
            if pred is None:
                continue
            candidate = pred['mask'] & (ct_new[k] > 1)
            fraction = candidate.sum() / max(1, pred['mask'].sum())
            if fraction < 0.35:
                continue
            support = 0
            for t in pred['point_ids']:
                x, y = np.rint(ct_pts[k, t]).astype(int)
                if 0 <= x < ct_w and 0 <= y < ct_h and candidate[y, x]:
                    support += 1
            if support < 2:
                continue
            donors = [int(j) for j in np.unique(ct_new[k][candidate]) if j > 1]
            safe = True
            for j in donors:
                donor = ct_new[k] == j
                if (donor & ~candidate).sum() < 0.25 * donor.sum():
                    safe = False
            if not safe:
                continue
            ct_new[k][candidate] = origin
            ct_recovered.append(dict(capture=k, recovered_id=origin, donor_ids=json.dumps(donors), recovered_pixels=int(candidate.sum()), predicted_support_fraction=float(fraction), supporting_destination_points=support, affine_inliers=pred['inliers'], inlier_fraction=pred['inlier_fraction'], median_residual_px=pred['median_residual_px'], matrix=json.dumps(pred['A'].tolist()), decision='missing_initial_identity_recovered_within_nonpink_support'))
    return (ct_new, dict(aliases=ct_alias, recoveries=ct_recovered, points=at_rows))


def accepted_partition(raw, outer, *, geometry_raw=None, check=lambda: None, progress=lambda *_: None):
    """Track image-supported territories and reserve a central residual compartment."""
    check()
    raw=np.asarray(raw);outer=np.asarray(outer,bool)
    if raw.ndim!=3 or raw.shape!=outer.shape or not len(raw):
        raise ValueError('Expected matching nonempty image and boundary stacks')
    if not outer.any():
        return dict(labels=np.zeros(raw.shape,np.int32), material=np.zeros_like(outer), events=[])
    if not raw[0][outer[0]].size or np.std(raw[0][outer[0]]) <= 1e-6:
        return dict(labels=np.where(outer,1,0).astype(np.int32),material=raw_material(raw,outer),events=[])
    full=np.pad(raw,((0,0),(15,15),(15,15)),mode='reflect')
    domains=np.pad(outer,((0,0),(15,15),(15,15)))
    progress(0,'Locating persistent density peaks')
    temporal=temporal_partition(full,domains,check=check)
    params=dict(width=32,smooth_sigma=2.,prominence=.025,min_peak_distance=18,min_region_area=120,min_support_fraction=.5)
    initial=split_regions(full[0],domains[0],params)
    if not initial['labels'].any():
        return dict(labels=np.where(outer,1,0).astype(np.int32),material=raw_material(raw,outer),events=[])
    motion=image_motion(full[0],full[1],domains[0],max_corners=1200,min_distance=6) if len(full)>1 else ([],None)
    pockets,_=pocket_candidates(full[0],domains[0],temporal['signals'][0],temporal['enhanced'][0],temporal['labels'][0],initial['markers'],motion)
    first=np.where(temporal['labels'][0]>0,temporal['labels'][0]+1,0).astype(np.int32)
    next_id=int(first.max())+1
    for polarity in ('light','dark'):
        for pocket in pockets:
            if pocket['polarity']==polarity:
                first[pocket['mask']]=next_id;next_id+=1
    ordinary=temporal['labels'][:,15:-15,15:-15]
    number=int(ordinary[0].max());extra=len(np.unique(first[first>1]))-number
    proposals=np.where(ordinary>0,ordinary+1,0).astype(np.int32)
    proposals[ordinary>number]+=extra
    check();progress(20,'Tracking colony texture and visible fragments')
    tracks=track_territories(raw,outer,first[15:-15,15:-15],proposals,check=check)
    progress(45,'Confirming persistent new colonies')
    values=raw.astype(np.float32)
    texture=np.array([np.sqrt(np.maximum(0,ndi.gaussian_filter(g*g,7)-ndi.gaussian_filter(g,7)**2))>=5 for g in values])&outer
    check();added,births=add_structures(raw,outer,texture,tracks['labels'],check=check)
    merged,aliases,pairs=merge_fragments(added,[b['region'] for b in births],tracks['partial_frames'],
        {key:value for key,value in tracks['full_predictions'].items() if isinstance(key[1],int)},tracks['motion'],maximum_gap=8,check=check)
    protected,motion_rows=evidence(raw,outer,merged,check=check)
    frames=[];events=[];central=[]
    signals=temporal['signals'][:,15:-15,15:-15];enhanced=temporal['enhanced'][:,15:-15,15:-15]
    geometry=temporal if geometry_raw is None else temporal_partition(geometry_raw,domains,check=check)
    geometry_signals=geometry['signals'][:,15:-15,15:-15]
    edge=outer[0]&ndi.binary_dilation(~outer[0],structure=STRUCTURE)
    edge_ids=set(int(i) for i in np.unique(merged[0][edge]) if i>1)
    supported={t:set() for t in range(len(raw))}
    for row in tracks['evidence']:
        if row['status'] in ('observed','area_proposal_rejected','observed_support_change') and not row['partial'] and row['fine_ncc']>=.65 and row['coarse_ncc']>=.75:
            supported[row['frame']].add(row['region'])
        events.append(dict(capture=row['frame'],display_id=row['region'],event=row['status'],**{k:v for k,v in row.items() if k not in ('frame','region','status')}))
    for t,lab in enumerate(merged):
        check();progress(65+30*t//len(raw),f'Resolving shared boundaries: frame {t+1}/{len(raw)}')
        terrain=ndi.gaussian_gradient_magnitude(signals[t],1.5)
        current=joint_partition(lab,outer[t],enhanced[t])
        current,_=resolve_seams(current,outer[t],terrain)
        current,_=remove_narrow_pink(current,outer[t],terrain,5)
        current,_=absorb_enclosed_pockets(current,outer[t],terrain,protected=protected[t])
        terrain=ndi.gaussian_gradient_magnitude(geometry_signals[t],1.5)
        current,_=localize_residual(current,outer[t],terrain,interface=12)
        current,_=resolve_seams(current,outer[t],terrain)
        if t:current,_=smooth_joint(current,outer[t],sigma=1.,band=3,tolerance=.03)
        central.append(current.copy())
        current,_=finish_shared_edges(current,outer[t],terrain)
        frames.append(current)
    continued,point_audit=texture_continuity(raw,np.asarray(frames),tracks['material'],check=check)
    for alias,parent in point_audit['aliases'].items():
        for t in range(len(raw)):
            if (frames[t]==alias).any():events.append(dict(capture=t,display_id=parent,event='fragment_reconciliation',alias=alias))
    for recovery in point_audit['recoveries']:
        events.append(dict(capture=recovery['capture'],display_id=recovery['recovered_id'],event='observed_continuity'))
    frames=[]
    for t,current in enumerate(continued):
        check()
        terrain=ndi.gaussian_gradient_magnitude(geometry_signals[t],1.5)
        original_pink=central[t]==1
        current,collar=resolve_edge_collars(current,outer[t],supported[t],width_fraction=.3,edge_candidates=edge_ids)
        protection=(ndi.distance_transform_edt(original_pink)>=6)&(current==1)
        for _ in range(64):
            revised,_=resolve_seams(current,outer[t],terrain)
            revised[protection]=1
            if np.array_equal(revised,current):break
            current=revised
        else:raise RuntimeError('Shared boundaries did not converge')
        assert np.array_equal(current>0,outer[t])
        frames.append(current)
    for birth in births:
        for t in range(len(raw)):
            if (frames[t]==birth['region']).any():events.append(dict(capture=t,display_id=birth['region'],event='persistent_structure'))
    return dict(labels=np.asarray(frames), material=tracks['material'], events=events, aliases=aliases)
