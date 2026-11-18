"""Following microcolonies through time (Partaker 2, Figure 3): every microcolony carries an identity, and identities merge.

What a microcolony is at each frame comes from the rule only (microcolony_rules: close and enough). This module adds the
bookkeeping over time, on one canvas where the drift of the view between frames is taken out:

same place   the region now lies on the region before (at least overlap_frac of the earlier region and overlap_min_px)
merge        two identities are found in one region now, at least sqrt(n) cells of the smaller one (and at least 2) are
             linked to the other, and their earlier regions are still in one region in each of the next frames for hold_h
             (1 h). The identity with more cells goes on, the other ends there. A touch that has not held yet is divided
             between the two (every pixel to the nearest earlier territory)
split        one identity is found in two regions now and they stay apart for hold_h: the piece with more cells keeps
             the identity, the other starts a new one
seen again   an identity that is not found for up to seen_again_h (it fell under 4 cells) and is found again in the same
             place goes on
lasting      a microcolony that is found in at least two frames (and reached 4 cells)
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops

from .microcolony_rules import MicrocolonySettings, drop_small_cells, linked_groups


@dataclass(frozen=True)
class FollowSettings:
    rule: MicrocolonySettings = MicrocolonySettings()
    seen_again_h: float = 1.0      # how long an identity may be missing and still go on
    overlap_frac: float = 0.10     # of the earlier region that has to lie on the region now
    overlap_min_px: int = 20
    edge_um: float = 1.0           # a region within this of the edge of the view "touches the edge"
    contact: float = 1.0           # cells of the smaller microcolony that must be linked to the other: contact * sqrt(n), at least 2
    max_shift_px: int = 400        # largest drift between two frames that is looked for


@dataclass
class MicrocolonyTracks:
    frames: List[int]
    hours: Dict[int, float]
    offsets: Dict[int, tuple]                       # frame -> (oy, ox): canvas[y + oy, x + ox] = frame[y, x]
    canvas_shape: tuple
    identities: Dict[int, dict]                     # identity -> birth, born, parent, end, ended, into
    events: List[dict]                              # time, hours, event, identity, into, cells_before, cells_other_before
    rows: List[dict]                                # one row per identity and frame: cells, area_um2, cy_canvas, cx_canvas
    cell_identity: Dict[int, Dict[int, int]]        # frame -> cell label -> identity (0 for a cell in no microcolony)
    min_cells: int = 4
    maps: Dict[int, bytes] = field(default_factory=dict, repr=False)      # frame -> the identity of every pixel of the canvas, packed
    points: Dict[int, tuple] = field(default_factory=dict, repr=False)    # frame -> (cell labels, y, x of a pixel inside each cell, frame coordinates)
    _lasting: Optional[set] = field(default=None, repr=False)

    def identity_map(self, frame: int) -> np.ndarray:
        """The microcolony each pixel of the canvas belongs to at this frame (0 for none): the area of every microcolony."""
        return np.frombuffer(zlib.decompress(self.maps[int(frame)]), np.uint16).reshape(self.canvas_shape).astype(np.int32)

    def lasting(self) -> set:
        """Identities found in at least two frames that reached min_cells."""
        if self._lasting is None:
            n, most = {}, {}
            for r in self.rows:
                n[r["identity"]] = n.get(r["identity"], 0) + 1; most[r["identity"]] = max(most.get(r["identity"], 0), r["cells"])
            self._lasting = {i for i in n if n[i] >= 2 and most[i] >= self.min_cells}
        return self._lasting

    def merges(self) -> List[dict]:
        """The merges between lasting microcolonies, in time order."""
        keep = self.lasting()
        return [e for e in self.events if e["event"] == "join" and e["identity"] in keep and e["into"] in keep]

    def count_at(self, frame: int) -> int:
        """Lasting microcolonies found in this frame."""
        keep = self.lasting()
        return len({r["identity"] for r in self.rows if r["time"] == frame and r["identity"] in keep})

    def hours_before_merging(self) -> List[float]:
        """For every lasting microcolony that formed (or split off) during the recording and merged: hours from forming to merging."""
        return [e["hours"] - self.hours[self.identities[e["identity"]]["birth"]] for e in self.merges()
                if self.identities[e["identity"]]["born"] in ("formed", "split")]


def cell_shift(mask_a: np.ndarray, mask_b: np.ndarray, max_shift: int = 400):
    """(dy, dx) to add to frame-b coordinates to land on frame a, from the cross-correlation of the two blurred cell masks."""
    A, B = ndi.gaussian_filter(mask_a.astype(np.float32), 2), ndi.gaussian_filter(mask_b.astype(np.float32), 2)
    xc = np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(A) * np.conj(np.fft.rfft2(B)), s=A.shape))
    cy, cx = np.array(A.shape) // 2
    m = int(min(max_shift, cy - 1, cx - 1))
    w = xc[cy - m:cy + m + 1, cx - m:cx + m + 1]
    dy, dx = np.unravel_index(np.argmax(w), w.shape)
    return int(dy - m), int(dx - m)


@dataclass
class _Frame:
    regions: np.ndarray            # microcolonies of the frame, 1..n (frame coordinates)
    labels: np.ndarray             # cell label of every cell
    y: np.ndarray                  # a pixel inside each cell (the one nearest its centre)
    x: np.ndarray
    colony: np.ndarray             # microcolony of each cell, 0 for none
    links: np.ndarray              # pairs of cell labels that are linked


def _frame(labels: np.ndarray, pixel_um: float, S: MicrocolonySettings) -> _Frame:
    cells = drop_small_cells(labels, pixel_um, S.min_cell_um2)
    dist, (iy, ix) = ndi.distance_transform_edt(cells == 0, return_indices=True)       # one distance map for the groups and for the links
    groups = linked_groups(cells, pixel_um, S.gap_um, distance=dist)
    keep = np.flatnonzero(groups.sizes >= S.min_cells); keep = keep[keep > 0]
    number = np.zeros(groups.sizes.size, np.int32); number[keep] = np.arange(1, keep.size + 1)
    lab, ys, xs, col = [], [], [], []
    for p in regionprops(cells):
        cy, cx = p.centroid; co = p.coords; j = np.argmin((co[:, 0] - cy) ** 2 + (co[:, 1] - cx) ** 2)
        lab.append(p.label); ys.append(int(co[j, 0])); xs.append(int(co[j, 1])); col.append(int(number[groups.cell_group[p.label]]))
    grown = np.where(dist <= S.gap_um / 2 / pixel_um, cells[iy, ix], 0); pairs = set()   # every pixel within half the gap of a cell takes that cell
    for a, b in ((grown[:, :-1], grown[:, 1:]), (grown[:-1, :], grown[1:, :])):
        m = (a != b) & (a > 0) & (b > 0)
        if m.any():
            pr = np.unique(np.stack([np.minimum(a[m], b[m]), np.maximum(a[m], b[m])], 1), axis=0); pairs |= {(int(u), int(v)) for u, v in pr}
    return _Frame(number[groups.group_map].astype(np.int32), np.array(lab, np.int64), np.array(ys, np.int64), np.array(xs, np.int64),
                  np.array(col, np.int64), np.array(sorted(pairs), np.int64).reshape(-1, 2))


def follow_microcolonies(labels_at: Callable[[int], np.ndarray], frames: Sequence[int], pixel_um: float, dt_h: float, start_hour: float = 0.0,
                         settings: FollowSettings = FollowSettings(), progress: Optional[Callable[[int, int], None]] = None,
                         drift: Optional[Dict[int, tuple]] = None) -> MicrocolonyTracks:
    """Follow the microcolonies of consecutive frames. labels_at(t) gives the cell labels of frame t.
    drift[t] = (dy, dx) of frame t against the frame before it, when it is known; else it is measured on the cell masks."""
    S = settings; T = [int(t) for t in frames]; t_end = T[-1]; first = T[0]
    hold = max(1, int(round(S.rule.hold_h / dt_h))); gap = max(1, int(round(S.seen_again_h / dt_h)))
    hours = {t: float(start_hour + t * dt_h) for t in T}       # start_hour is the hour of frame 0 of the recording
    edge_px = int(round(S.edge_um / pixel_um))

    # ---- pass 1: the cells and microcolonies of every frame, and the drift of the view from frame to frame
    FR: Dict[int, _Frame] = {}; cum = {first: (0, 0)}; prev_mask = None; shape = None
    for k, t in enumerate(T):
        lab = np.asarray(labels_at(t)); shape = lab.shape; FR[t] = _frame(lab, pixel_um, S.rule); mask = lab > 0
        if prev_mask is not None:
            dy, dx = drift[t] if drift is not None else cell_shift(prev_mask, mask, S.max_shift_px); cum[t] = (cum[T[k - 1]][0] + dy, cum[T[k - 1]][1] + dx)
        prev_mask = mask
        if progress:
            progress(k + 1, 2 * len(T))
    min_y, min_x = min(v[0] for v in cum.values()), min(v[1] for v in cum.values())
    oy = {t: cum[t][0] - min_y for t in T}; ox = {t: cum[t][1] - min_x for t in T}
    Hf, Wf = shape; Hc, Wc = Hf + max(oy.values()), Wf + max(ox.values())

    def canvas(a, t):
        out = np.zeros((Hc, Wc), np.int32); out[oy[t]:oy[t] + Hf, ox[t]:ox[t] + Wf] = a; return out

    R: Dict[int, np.ndarray] = {}

    def regions(t):
        if t not in R:
            R[t] = canvas(FR[t].regions, t)
        return R[t]

    def home(terr, u):
        """The region of frame u that a territory (bounding box, mask inside it) lies on (0 if none passes the overlap test)."""
        sl, m = terr; n = int(m.sum())
        if n == 0 or u not in FR:
            return 0
        cnt = np.bincount(regions(u)[sl][m]); cnt[0] = 0; k = int(cnt.argmax())
        return k if cnt[k] >= max(S.overlap_min_px, S.overlap_frac * n) else 0

    def touches_edge(terr, t):
        sl, m = terr; ys, xs = np.nonzero(m); y0, y1, x0, x1 = ys.min() + sl[0].start, ys.max() + sl[0].start, xs.min() + sl[1].start, xs.max() + sl[1].start
        return bool(y0 - oy[t] < edge_px or x0 - ox[t] < edge_px or oy[t] + Hf - 1 - y1 < edge_px or ox[t] + Wf - 1 - x1 < edge_px)

    def boxes(image):
        """identity or region number -> its bounding box."""
        return {k + 1: sl for k, sl in enumerate(ndi.find_objects(image)) if sl is not None}

    # ---- pass 2: the bookkeeping
    ident: Dict[int, dict] = {}; rows: List[dict] = []; events: List[dict] = []; cell_identity: Dict[int, Dict[int, int]] = {}; maps: Dict[int, bytes] = {}
    I_prev = np.zeros((Hc, Wc), np.int32); last_seen: Dict[int, tuple] = {}; next_id = 1; cells_of_prev: Dict[int, int] = {}
    for k_t, t in enumerate(T):
        Rt = regions(t); fr = FR[t]; I_now = np.zeros((Hc, Wc), np.int32)
        ccy, ccx = fr.y + oy[t], fr.x + ox[t]
        box_prev = boxes(I_prev); ids_prev = sorted(box_prev)
        terr = {}; area_prev = {}
        for i in ids_prev:
            sl = box_prev[i]; terr[i] = (sl, I_prev[sl] == i); area_prev[i] = int(terr[i][1].sum())
        cells_prev = {i: cells_of_prev.get(i, 0) for i in ids_prev}
        found: Dict[int, list] = {}                    # region now -> identities found in it
        for i in ids_prev:
            sl, m = terr[i]; cnt = np.bincount(Rt[sl][m]); cnt[0] = 0
            for k in np.flatnonzero(cnt >= max(S.overlap_min_px, S.overlap_frac * area_prev[i])):
                found.setdefault(int(k), []).append(i)
        # contact between identities that share a region now: every cell of the region goes to the nearest earlier territory
        contact = {}
        if S.contact > 0:
            craw = Rt[ccy, ccx] if len(fr.labels) else np.zeros(0, np.int32)
            index_of = np.full(int(fr.labels.max()) + 1 if len(fr.labels) else 1, -1, np.int64); index_of[fr.labels] = np.arange(len(fr.labels))
            nu, nv = (index_of[fr.links[:, 0]], index_of[fr.links[:, 1]]) if len(fr.links) else (np.zeros(0, np.int64), np.zeros(0, np.int64))
            for k, lst in found.items():
                if len(lst) < 2:
                    continue
                sel = np.flatnonzero(craw == k); ys_, xs_ = ccy[sel], ccx[sel]
                y0, y1, x0, x1 = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
                for i in lst:
                    s_i = terr[i][0]; y0, y1, x0, x1 = min(y0, s_i[0].start), max(y1, s_i[0].stop), min(x0, s_i[1].start), max(x1, s_i[1].stop)
                seed = np.zeros((y1 - y0, x1 - x0), np.int32); pv = I_prev[y0:y1, x0:x1]
                for i in lst:
                    seed[pv == i] = i
                _, (iy, ix) = ndi.distance_transform_edt(seed == 0, return_indices=True)
                owner = np.zeros(len(fr.labels), np.int32); owner[sel] = seed[iy, ix][ys_ - y0, xs_ - x0]
                ou, ov = owner[nu], owner[nv]
                for a in range(len(lst)):
                    for b in range(a + 1, len(lst)):
                        i, j = lst[a], lst[b]; ij, ji = (ou == i) & (ov == j), (ou == j) & (ov == i)
                        ci = set(nu[ij].tolist()) | set(nv[ji].tolist()); cj = set(nv[ij].tolist()) | set(nu[ji].tolist())
                        ni, nj = int((owner == i).sum()), int((owner == j).sum())
                        contact[(i, j)] = (len(ci), ni) if ni <= nj else (len(cj), nj)
        # merges that hold: union-find over the identities found together now
        alias = {i: i for i in ids_prev}

        def root(i):
            while alias[i] != i:
                i = alias[i]
            return i

        later = [u for u in T[k_t + 1:k_t + 1 + hold]]
        for k, lst in found.items():
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    i, j = lst[a], lst[b]; Ti, Tj = terr[i], terr[j]; ok, seen = True, 0
                    for u in later:
                        hi, hj = home(Ti, u), home(Tj, u)
                        if hi and hj:
                            seen += 1; ok &= hi == hj
                    if S.contact > 0:
                        linked, n_small = contact.get((i, j), contact.get((j, i), (0, 0)))
                        ok &= n_small > 0 and linked >= max(2, S.contact * np.sqrt(n_small))
                    if ok and (seen > 0 or t == t_end):
                        ri, rj = root(i), root(j)
                        if ri != rj:
                            keep, gone = (ri, rj) if (cells_prev.get(ri, 0), -ri) >= (cells_prev.get(rj, 0), -rj) else (rj, ri)
                            alias[gone] = keep
        for i in ids_prev:
            r = root(i)
            if r != i and ident[i]["end"] is None:
                ident[i].update(end=t, ended="joined", into=r)
                events.append({"time": t, "hours": hours[t], "event": "join", "identity": i, "into": r,
                               "cells_before": cells_prev.get(i, 0), "cells_other_before": cells_prev.get(r, 0)})
        # paint every region of the frame
        box_now = boxes(Rt)
        for k in sorted(box_now):
            sl = box_now[k]
            sl = (slice(max(sl[0].start - 2, 0), sl[0].stop + 2), slice(max(sl[1].start - 2, 0), sl[1].stop + 2))
            reg = Rt[sl] == k; here = sorted({root(i) for i in found.get(k, [])})
            if not here:                               # seen again, or a new identity
                best, most = None, 0
                for i, (tl, sl0, m0) in list(last_seen.items()):
                    if T.index(t) - T.index(tl) <= gap + 1:
                        o = int((m0 & (Rt[sl0] == k)).sum())
                        if o >= max(S.overlap_min_px, S.overlap_frac * m0.sum()) and o > most:
                            best, most = i, o
                if best is not None:
                    here = [best]; ident[best].update(end=None, ended=None); last_seen.pop(best)
                    events.append({"time": t, "hours": hours[t], "event": "seen again", "identity": best})
                else:
                    born = "at start" if k_t == 0 else ("entered" if touches_edge((sl, reg), t) else "formed")
                    ident[next_id] = {"birth": t, "born": born, "parent": None, "end": None, "ended": None, "into": None}
                    here = [next_id]; next_id += 1
            if len(here) == 1:
                I_now[sl][reg] = here[0]
            else:                                      # a touch that has not held yet: every pixel to the nearest earlier territory
                seed = np.zeros(reg.shape, np.int32); prev_here = I_prev[sl]
                for i in ids_prev:
                    if root(i) in here:
                        seed[(prev_here == i) & reg] = root(i)
                _, (iy, ix) = ndi.distance_transform_edt(seed == 0, return_indices=True)
                I_now[sl][reg] = seed[iy, ix][reg]
        # splits: one identity in two or more separate regions now
        cid = I_now[ccy, ccx] if len(fr.labels) else np.zeros(0, np.int32); craw_now = Rt[ccy, ccx] if len(fr.labels) else np.zeros(0, np.int32)
        box_id = boxes(I_now)
        for i in sorted(box_id):
            sl_i = box_id[i]; m_i = I_now[sl_i] == i; R_i = Rt[sl_i]; ks = [int(v) for v in np.unique(R_i[m_i]) if v > 0]
            if len(ks) < 2:
                continue
            pieces = sorted(((int(((cid == i) & (craw_now == k) & (fr.colony > 0)).sum()), int((m_i & (R_i == k)).sum()), k) for k in ks), reverse=True)
            main = pieces[0][2]
            for ncell, _npx, k in pieces[1:]:
                Pm, Pk = (sl_i, m_i & (R_i == main)), (sl_i, m_i & (R_i == k)); apart, seen = True, 0
                for u in later:
                    h1, h2 = home(Pm, u), home(Pk, u)
                    if h1 and h2:
                        seen += 1; apart &= h1 != h2
                    elif h1 or h2:
                        seen += 1                      # one piece is gone: they are apart
                if apart and seen > 0:
                    ident[next_id] = {"birth": t, "born": "split", "parent": i, "end": None, "ended": None, "into": None}
                    I_now[sl_i][Pk[1]] = next_id
                    events.append({"time": t, "hours": hours[t], "event": "split", "identity": next_id, "into": i, "cells_before": ncell})
                    next_id += 1
        # the table of this frame
        cid = np.where(fr.colony > 0, I_now[ccy, ccx], 0) if len(fr.labels) else np.zeros(0, np.int32)
        cell_identity[t] = {int(l): int(i) for l, i in zip(fr.labels, cid)}
        box_id = boxes(I_now); now_ids = sorted(box_id); cells_of_prev = {}
        for i in now_ids:
            sl = box_id[i]; reg = I_now[sl] == i; ys, xs = np.nonzero(reg); n = int((cid == i).sum()); cells_of_prev[i] = n
            rows.append({"identity": i, "time": t, "hours": hours[t], "cells": n, "area_um2": float(reg.sum()) * pixel_um ** 2,
                         "cy_canvas": float(ys.mean() + sl[0].start), "cx_canvas": float(xs.mean() + sl[1].start), "at_view_edge": touches_edge((sl, reg), t)})
        # identities not found now: remember where they were, close them when they have been missing too long
        for i in ids_prev:
            if root(i) == i and i not in now_ids and i not in last_seen:
                last_seen[i] = (t, terr[i][0], terr[i][1]); ident[i].update(end=t, ended="pending")
        for i, (tl, sl0, m0) in list(last_seen.items()):
            if T.index(t) - T.index(tl) > gap:
                out_of_view = touches_edge((sl0, m0), t)
                ident[i].update(end=tl, ended="left the view" if out_of_view else "fell under 4 cells"); last_seen.pop(i)
                events.append({"time": tl, "hours": hours[tl], "event": ident[i]["ended"], "identity": i})
        I_prev = I_now; maps[t] = pack_map(I_now)
        for u in [u for u in R if u < t]:
            R.pop(u)                                    # earlier frames are not looked at again
        if progress:
            progress(len(T) + k_t + 1, 2 * len(T))
    for d in ident.values():
        if d["ended"] == "pending":
            d.update(ended="fell under 4 cells")
        if d["end"] is None:
            d.update(end=t_end, ended="still there")
    events.sort(key=lambda e: (e["time"], e["event"]))
    return MicrocolonyTracks(frames=T, hours=hours, offsets={t: (int(oy[t]), int(ox[t])) for t in T}, canvas_shape=(int(Hc), int(Wc)),
                             identities=ident, events=events, rows=rows, cell_identity=cell_identity, min_cells=S.rule.min_cells,
                             maps=maps, points={t: (FR[t].labels, FR[t].y, FR[t].x) for t in T})


def pack_map(identity_map: np.ndarray) -> bytes:
    return zlib.compress(np.ascontiguousarray(identity_map, np.uint16).tobytes(), 1)


def close_gaps(tracks: MicrocolonyTracks, labels_at: Callable[[int], np.ndarray], pixel_um: float, max_gap_frames: int = 3,
               settings: MicrocolonySettings = MicrocolonySettings()) -> MicrocolonyTracks:
    """The followed microcolonies with their short gaps drawn through, for pictures and for what is measured inside a
    microcolony. A microcolony that is not found for up to max_gap_frames (it fell under 4 cells) and is found again in
    the same place keeps its cells in between: the cells on its area before or after the gap that belong to no other
    microcolony get its identity, and its area is painted around them. Merges, births and ends are not changed; the
    filled rows are marked gap_closed."""
    seen: Dict[int, List[int]] = {}
    for r in tracks.rows:
        seen.setdefault(r["identity"], []).append(r["time"])
    order = {t: k for k, t in enumerate(tracks.frames)}; todo: Dict[int, list] = {}
    for i, ts in seen.items():
        ts = sorted(set(ts))
        for tp, tn in zip(ts[:-1], ts[1:]):
            if 1 <= order[tn] - order[tp] - 1 <= max_gap_frames:
                for k in range(order[tp] + 1, order[tn]):
                    todo.setdefault(tracks.frames[k], []).append((int(i), int(tp), int(tn)))
    r_px = settings.gap_um / 2 / pixel_um; pad = int(round(settings.gap_um / pixel_um))
    maps = dict(tracks.maps); cell_identity = {t: dict(v) for t, v in tracks.cell_identity.items()}
    rows = [dict(r, gap_closed=False) for r in tracks.rows]
    for t in sorted(todo):
        I = tracks.identity_map(t); oy, ox = tracks.offsets[t]; lab = drop_small_cells(np.asarray(labels_at(t)), pixel_um, settings.min_cell_um2)
        can = np.zeros(I.shape, np.int32); can[oy:oy + lab.shape[0], ox:ox + lab.shape[1]] = lab
        labels, ys, xs = tracks.points[t]; cy, cx = ys + oy, xs + ox; cid = np.array([cell_identity[t].get(int(l), 0) for l in labels], np.int64)
        for i, tp, tn in todo[t]:
            terr = (tracks.identity_map(tp) == i) | (tracks.identity_map(tn) == i); py, px_ = np.nonzero(terr)
            sl = (slice(max(py.min() - pad, 0), py.max() + pad + 1), slice(max(px_.min() - pad, 0), px_.max() + pad + 1))
            near = np.zeros(I.shape, bool); near[sl] = ndi.binary_dilation(terr[sl], iterations=max(pad // 2, 1))
            take = np.flatnonzero(near[cy, cx] & (cid == 0)) if len(labels) else np.zeros(0, int)
            if take.size == 0:
                continue
            cid[take] = i
            grown = (ndi.distance_transform_edt(~np.isin(can[sl], labels[take])) <= r_px) & (I[sl] == 0); I[sl][grown] = i
            gy, gx = np.nonzero(grown)
            rows.append({"identity": i, "time": t, "hours": tracks.hours[t], "cells": int(take.size), "area_um2": float(grown.sum()) * pixel_um ** 2,
                         "cy_canvas": float(gy.mean() + sl[0].start), "cx_canvas": float(gx.mean() + sl[1].start), "at_view_edge": False, "gap_closed": True})
        cell_identity[t] = {int(l): int(v) for l, v in zip(labels, cid)}; maps[t] = pack_map(I)
    rows.sort(key=lambda r: (r["time"], r["identity"]))
    return replace(tracks, rows=rows, cell_identity=cell_identity, maps=maps, _lasting=None)


# ---------------------------------------------------------------- with the saved session
def save_tracks(tracks: MicrocolonyTracks, path) -> str:
    """The followed microcolonies as a file of numbers and text (stage_files)."""
    from . import stage_files as SF
    header = {"frames": list(tracks.frames), "hours": tracks.hours, "offsets": tracks.offsets, "canvas_shape": list(tracks.canvas_shape),
              "identities": tracks.identities, "events": tracks.events, "rows": tracks.rows, "cell_identity": tracks.cell_identity,
              "min_cells": int(tracks.min_cells)}
    arrays = {f"map_{t}": np.frombuffer(packed, np.uint8) for t, packed in tracks.maps.items()}
    arrays.update({f"points_{t}": np.stack([np.asarray(a, np.int64) for a in triple]) for t, triple in tracks.points.items()})
    return SF.write(path, "followed microcolonies", header, arrays)


def load_tracks(path) -> MicrocolonyTracks:
    from . import stage_files as SF
    h, arrays = SF.read(path, "followed microcolonies")
    return MicrocolonyTracks(
        frames=[int(t) for t in h["frames"]], hours={int(t): float(v) for t, v in h["hours"].items()},
        offsets={int(t): tuple(int(v) for v in o) for t, o in h["offsets"].items()}, canvas_shape=tuple(int(v) for v in h["canvas_shape"]),
        identities=SF.int_keys(h["identities"]), events=h["events"], rows=h["rows"],
        cell_identity={int(t): {int(cell): int(i) for cell, i in cells.items()} for t, cells in h["cell_identity"].items()},
        min_cells=int(h["min_cells"]),
        maps={int(k[4:]): v.tobytes() for k, v in arrays.items() if k.startswith("map_")},
        points={int(k[7:]): tuple(v) for k, v in arrays.items() if k.startswith("points_")})
