"""Track verified colony regions; identities are scoped to position/channel."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def track_colonies_by_frame(colonies_by_frame, *, max_search_radius=50, max_lost=5):
    """Return copied detections with track IDs and conservative review flags.

    Frame keys are (position, time, channel). Frame-local colony IDs and
    geometry remain unchanged. IDs are only stable within this tracking run.
    Strong mask overlap repairs fragmented btrack identities. Merge/split events
    are geometric candidates for review, not biological lineage determinations.
    """
    import btrack

    results = {key: [dict(c) for c in colonies] for key, colonies in colonies_by_frame.items()}
    groups = defaultdict(list)
    for key in sorted(results):
        groups[(key[0], key[2])].append(key)

    for keys in groups.values():
        objects, detections, regions = [], [], defaultdict(list)
        origin = min(key[1] for key in keys)
        for key in keys:
            for colony in results[key]:
                mask = colony.get("mask")
                if isinstance(mask, np.ndarray):
                    moments = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
                    contour = colony.get("contour")
                    if contour is None:
                        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        contour = max(contours, key=cv2.contourArea) if contours else None
                else:
                    contour = colony.get("contour", colony.get("polygon"))
                    contour = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
                    moments = cv2.moments(contour)
                if not moments["m00"] or contour is None:
                    raise ValueError(f"Empty colony geometry at {key}, colony {colony.get('colony_id')}")
                x, y = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
                colony.update(track_id=None, tracking_status="tracked", tracking_review=False,
                              tracking_centroid_x=float(x), tracking_centroid_y=float(y))
                ref = len(objects)
                objects.append(btrack.btypes.PyTrackObject.from_dict(
                    {"ID": ref, "t": int(key[1] - origin), "x": x, "y": y, "z": 0.0}))
                detections.append(colony)
                regions[key[1]].append((ref, np.asarray(contour, dtype=np.int32), mask))
        if not objects:
            continue

        ambiguous = set()

        if len({obj.t for obj in objects}) == 1:
            for i, colony in enumerate(detections, 1):
                colony["track_id"] = i
                if i - 1 in ambiguous:
                    colony.update(tracking_status="ambiguous_merge_split", tracking_review=True)
            _repair_continuity(detections, regions, max_lost=max_lost)
            continue

        with btrack.BayesianTracker() as tracker:
            config = btrack.config.load_config(Path(__file__).parent / "config" / "btrack_config.json")
            # Configure a fresh in-memory model, leaving the cell config untouched.
            config.hypothesis_model.hypotheses = ["P_init", "P_term", "P_link"]
            config.motion_model.max_lost = int(max_lost)
            tracker.configure(config)
            tracker.max_search_radius = float(max_search_radius)
            tracker.append(objects)
            # append assigns IDs; retain its exact references, never spatially rematch.
            by_ref = {obj.ID: (i, detections[i]) for i, obj in enumerate(objects)}
            tracker.track(tracking_updates=["MOTION"])
            # Link gaps without discarding detections or inferring cell divisions.
            tracker.optimize(options={"tm_lim": 30000, "mip_gap": 0.01})
            tracks = [list(track.refs) for track in tracker.tracks]

        next_id, seen = 0, set()
        for refs in tracks:
            active_id = None
            for ref in refs:
                if ref < 0:  # predicted gap, not a measured colony
                    continue
                if ref not in by_ref or ref in seen:
                    raise RuntimeError("Unexpected or duplicated btrack object reference")
                seen.add(ref)
                index, colony = by_ref[ref]
                if active_id is None or index in ambiguous:
                    next_id += 1
                    active_id = next_id
                colony["track_id"] = active_id
                if index in ambiguous:
                    colony.update(tracking_status="ambiguous_merge_split", tracking_review=True)
                    active_id = None
        # Some btrack versions omit isolated observations. Preserve them explicitly.
        for ref in sorted(set(by_ref) - seen):
            index, colony = by_ref[ref]
            next_id += 1
            colony.update(track_id=next_id, tracking_review=True,
                          tracking_status="ambiguous_merge_split" if index in ambiguous else "unlinked")
        _repair_continuity(detections, regions, max_lost=max_lost)
    return results


def _repair_continuity(detections, regions, *, max_lost=5):
    """Assign chronological IDs, with explicit parent IDs for geometric events.

    Parent IDs are JSON strings for scalar CSV/Parquet compatibility. Strong
    overlap means >=50% of the smaller region; event contributions must also
    occupy >=10% of the larger region. Thresholds are intentionally centralized.
    """
    import json

    raw_ids = [c["track_id"] for c in detections]
    areas = {}
    for entries in regions.values():
        for ref, contour, mask in entries:
            if isinstance(mask, np.ndarray):
                areas[ref] = int(np.count_nonzero(mask))
            else:
                x, y, w, h = cv2.boundingRect(contour)
                patch = np.zeros((h, w), np.uint8)
                cv2.drawContours(patch, [contour.reshape(-1, 1, 2) - (x, y)], -1, 1, -1)
                areas[ref] = int(np.count_nonzero(patch))
    next_id = 0
    raw_last, retired = {}, set()
    for time in sorted(regions):
        previous, current = regions.get(time - 1, []), regions[time]
        incoming, outgoing = defaultdict(list), defaultdict(list)
        for a, ca, ma in previous:
            for b, cb, mb in current:
                xa, ya, wa, ha = cv2.boundingRect(ca)
                xb, yb, wb, hb = cv2.boundingRect(cb)
                x0, y0 = max(xa, xb), max(ya, yb)
                x1, y1 = min(xa + wa, xb + wb), min(ya + ha, yb + hb)
                if x1 <= x0 or y1 <= y0:
                    continue
                patches = []
                for contour, mask in ((ca, ma), (cb, mb)):
                    if isinstance(mask, np.ndarray):
                        patch = mask[y0:y1, x0:x1] > 0
                    else:
                        patch = np.zeros((y1-y0, x1-x0), np.uint8)
                        cv2.drawContours(patch, [contour.reshape(-1, 1, 2) - (x0, y0)], -1, 1, -1)
                    patches.append(patch)
                overlap = int(np.count_nonzero(patches[0] & patches[1]))
                if overlap >= 0.5 * min(areas[a], areas[b]) and overlap >= 0.1 * max(areas[a], areas[b]):
                    incoming[b].append((a, overlap))
                    outgoing[a].append(b)
        assignments = {}
        for b, _, _ in current:
            colony = detections[b]
            colony.update(tracking_event="", tracking_parent_ids="[]")
            links = sorted(incoming[b], key=lambda pair: (-pair[1], pair[0]))
            parents = sorted({detections[a]["track_id"] for a, _ in links})
            split = any(len(outgoing[a]) > 1 for a, _ in links)
            merge = len(links) > 1
            chosen = None
            if merge or split:
                event = "merge_split" if merge and split else "merge" if merge else "split"
                colony.update(tracking_event=event, tracking_parent_ids=json.dumps(parents),
                              tracking_status=event, tracking_review=True)
                # A clear dominant merge parent may continue. Splits start new IDs
                # to prevent duplicate identities in a frame.
                if merge and not split and links[0][1] >= 2 * links[1][1]:
                    chosen = detections[links[0][0]]["track_id"]
                retired.update(parent for parent in parents if parent != chosen)
            elif links:
                a = links[0][0]
                before = detections[a]
                distance = np.hypot(colony["tracking_centroid_x"] - before["tracking_centroid_x"],
                                    colony["tracking_centroid_y"] - before["tracking_centroid_y"])
                ratio = areas[b] / max(areas[a], 1)
                if 0.25 <= ratio <= 4 and distance <= max(100, 0.5 * np.sqrt(max(areas[a], areas[b]))):
                    chosen = before["track_id"]
                    colony.update(tracking_status="overlap_linked", tracking_review=False)
            assignments[b] = chosen
        used = {v for v in assignments.values() if v is not None}
        for b, _, _ in current:
            colony = detections[b]
            chosen = assignments[b]
            if chosen is None and not colony["tracking_event"]:
                last = raw_last.get(raw_ids[b])
                if last and time - last[0] <= max_lost + 1 and last[1] not in retired | used:
                    chosen = last[1]
            if chosen is None:
                next_id += 1
                chosen = next_id
            colony["track_id"] = chosen
            used.add(chosen)
        # Update only after the entire frame has been assigned.
        for b, _, _ in current:
            raw_last[raw_ids[b]] = (time, detections[b]["track_id"])
