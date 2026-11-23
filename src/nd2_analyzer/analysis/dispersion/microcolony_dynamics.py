"""Dispersion analysis of verified outer and Inside ROI masks.

This backend also reads reviewed notebook result files using their saved
mask and manifest format. It shares no runtime code with the notebook backend.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation

from .measure import crop, pad_back, z_map
from ..colony_regions import detect_colony_contours, detect_dark_regions

BACKEND = 'dispersion_regions_motion_v5'
BORDER = 15


@dataclass(frozen=True)
class DispersionSettings:
    cell_channel: int = 0  # the shared image channel; not an individual-cell model
    sample_step: int = 1
    mask_kind: str = 'colony'
    width: int = 32
    smooth_sigma: float = 2.0
    prominence: float = 0.025
    min_peak_distance: int = 18
    min_region_area: int = 120
    min_support_fraction: float = 0.50
    match_iou: float = 0.30
    match_margin: float = 0.10
    temporal_weight: float = 0.35
    pink_speed_ratio: float = 1.5
    pink_min_points: int = 6
    pink_persistence: int = 2
    motion: bool = True
    interval: float = 1.0
    time_unit: str = 'capture'
    dark_measurement: str = 'intensity_50'
    pink_source: str = ""  # optional reviewed region_ids.tif; blank = validated notebook discovery
    colony_threshold_percent: int = 39
    pixel_um: float = 0.0  # zero = unavailable; never borrow another dataset's scale

    def validate(self):
        if not 0 <= self.match_margin <= 1 or not 0 <= self.temporal_weight <= 1:
            raise ValueError("Matching margin and temporal weight must be between 0 and 1")
        if self.pink_speed_ratio <= 1 or self.pink_min_points < 6 or not 1 <= self.pink_persistence <= 3:
            raise ValueError("Invalid motion classification parameters")
        if not 0 <= self.colony_threshold_percent <= 100:
            raise ValueError("Colony intensity threshold must be between 0 and 100 percent")
        if self.dark_measurement not in ('inside_roi','hybrid','raw_z','intensity_50'):
            raise ValueError('Unknown dark-area measurement')
        if self.mask_kind not in ('colony', 'inner'):
            raise ValueError('Analysis domain must be the outer biofilm or Inside ROI')
        if not 1 <= self.width <= 256 or self.sample_step < 1:
            raise ValueError('Invalid neighborhood or sampling interval')
        if self.min_region_area < 1 or self.min_peak_distance < 0:
            raise ValueError('Invalid region area or peak spacing')
        if not 0 < self.prominence <= 1 or not 0 <= self.min_support_fraction <= 1 or not 0 < self.match_iou <= 1:
            raise ValueError('Invalid peak prominence, support or matching threshold')
        if not np.isfinite(self.smooth_sigma) or self.smooth_sigma <= 0 or not np.isfinite(self.interval) or self.interval <= 0:
            raise ValueError('Smoothing and interval must be positive')
        if not np.isfinite(self.pixel_um) or self.pixel_um < 0:
            raise ValueError('Pixel calibration must be nonnegative')



def whole_colony_mask(raw, threshold_percent, *, previous=None, seed=None):
    """Detect the verifier-style outer colony, following the same body across frames."""
    interior = crop(raw, BORDER)
    # The verifier's 60% area control means 0.60% of the analysis image.
    minimum = interior.size * 0.006
    contours = detect_colony_contours(interior, round(threshold_percent * 255 / 100), minimum, 37)
    candidates = []
    for contour in contours:
        mask = np.zeros(interior.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, -1)
        candidates.append(pad_back(mask.astype(bool), BORDER, raw.shape))
    if seed is not None and previous is None:
        x, y = map(int, seed)
        candidates = [mask for mask in candidates if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]]
    if not candidates:
        raise ValueError("No whole colony detected. Adjust the intensity threshold to enclose the colony.")
    if previous is None:
        return max(candidates, key=np.count_nonzero)
    overlaps = [int((mask & previous).sum()) for mask in candidates]
    if not max(overlaps):
        raise ValueError("The selected whole colony has no overlap with the next frame. Review the intensity threshold and frame range.")
    return candidates[int(np.argmax(overlaps))]


def inner_dark_mask(raw, outer):
    """Detect normalized dark intensity within the 15-pixel analysis crop."""
    mask = detect_dark_regions(crop(raw, BORDER), crop(outer, BORDER), 128, 1, 1)
    return pad_back(mask.astype(bool), BORDER, raw.shape)


def components(mask):
    return ndi.label(mask,structure=np.ones((3,3)))

disk = morphology.disk

def grow_from_enclosed_roi(initial,rois,distal_gate,attachment=8.,minimum_area=60):
    selected=initial&rois[0]
    history=[selected.copy()];additions=[np.zeros_like(selected)]
    component_rows=[]
    for k in range(1,len(rois)):
        gain=rois[k]&~rois[k-1]
        candidates=gain&distal_gate
        cc,n=ndi.label(candidates,structure=np.ones((3,3)))
        sizes=np.bincount(cc.ravel())
        supported_front=selected&distal_gate
        near=ndi.distance_transform_edt(~supported_front)<=attachment if supported_front.any() else np.zeros_like(selected)
        touching=set(int(i) for i in np.unique(cc[near&candidates]) if i>0)
        keep=[i for i in touching if sizes[i]>=minimum_area]
        added=np.isin(cc,keep)
        for i in range(1,n+1):
            if sizes[i]<minimum_area:continue
            ys,xs=np.where(cc==i)
            component_rows.append(dict(frame_index=k,component=i,area_px=int(sizes[i]),
                centroid_x_px=float(xs.mean()),centroid_y_px=float(ys.mean()),
                connected_to_selected_front=i in touching,selected=i in keep))
        selected=(selected|added)&rois[k]
        history.append(selected.copy());additions.append(added)
    return np.stack(history),np.stack(additions),component_rows

def make_connected_region(original,anchor,roi,radius=16,interface_factor=2,debris_limit=4):
    original=original&roi
    original_cc,original_n=components(original)
    anchor_part=anchor&original
    expansion=original&~anchor
    bridge=np.zeros_like(original)
    result=original.copy()
    if original_n>1 and anchor_part.any() and expansion.any():
        d_anchor=ndi.distance_transform_edt(~anchor_part)
        d_growth=ndi.distance_transform_edt(~expansion)
        interface=(d_anchor<=interface_factor*radius)&(d_growth<=interface_factor*radius)
        # Closing = dilation then erosion. OR with original retains source pixels.
        closed=ndi.binary_closing(original,structure=disk(radius),border_value=0)
        bridge=closed&interface&roi&~original
        result=original|bridge
    cc,n=components(result)
    removed=np.zeros_like(original)
    if n>1:
        sizes=np.bincount(cc.ravel())
        tiny=[i for i in range(1,len(sizes)) if sizes[i]<=debris_limit and not np.any((cc==i)&anchor)]
        removed=np.isin(cc,tiny)
        result &= ~removed
    final_n=components(result)[1]
    # Do not silently enlarge the radius if a substantive component remains separate.
    return result,bridge,removed,dict(before_components=original_n,after_components=final_n)


S = dict(forward_backward_error=2.,max_flow=18.,max_gray_residual=30.)

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

def notebook_support(image,outer,inner,previous=None):
    """Notebook strict/hybrid support and independent normalized-dark detector.

    Operates on the same 15px-cropped analysis image as the later notebooks.
    Hybrid additions are reported separately and never relabeled as saved pixels.
    """
    crop=np.s_[BORDER:-BORDER,BORDER:-BORDER]
    g=image[crop];roi=outer[crop];saved=inner[crop]
    den=ndi.uniform_filter(roi.astype(float),size=32,mode='constant')
    num=ndi.uniform_filter(saved.astype(float),size=32,mode='constant')
    fraction=np.divide(num,den,out=np.zeros_like(num),where=den>1e-9);fraction[~roi]=0
    motion=None
    if previous is not None:
        old_image,old_outer,old_fraction,old_hybrid=previous
        motion=flow_pair(old_image[crop],g,old_outer[crop],roi)
        prior=motion['remap'](old_fraction[crop].astype(np.float32))
        fraction=np.where(motion['valid'],.65*fraction+.35*prior,fraction)
    fraction[~roi]=0
    near=ndi.distance_transform_edt(~saved)<=3 if saved.any() else np.zeros(saved.shape,bool)
    fill=(fraction>=.65)&near&roi
    if motion is not None:
        prior=motion['remap'](old_hybrid[crop].astype(np.uint8),True)>0
        fill |= prior&motion['valid']&(fraction>=.55)&near&roi
    hybrid=(saved|fill)&roi
    return tuple(pad_back(values, BORDER, image.shape) for values in (fraction, hybrid, z_map(g)))


def threshold_dark(z,domain,threshold=.8,minimum=4):
    labels,n=ndi.label((z>threshold)&domain,structure=np.ones((3,3)))
    sizes=np.bincount(labels.ravel());keep=np.flatnonzero(sizes>=minimum);keep=keep[keep>0]
    return np.where(np.isin(labels,keep),labels,0).astype(np.int32)


def shared_border_partition(labels,roi,local_fraction):
    """Notebook narrow-seam reassignment; all original non-pink pixels stay fixed."""
    if components(labels==1)[1]!=1 or not (labels>1).any():
        return labels.copy(),dict(status='no_connected_pink_or_neighboring_regions')
    from skimage.segmentation import watershed
    from skimage.graph import route_through_array
    base=[labels];masks=[roi];fraction=[local_fraction];captures=[0]
    H,W=labels.shape;yy,xx=np.indices((H,W));result=[];change_rows=[];topology_rows=[]
    seam_masks=[]
    for k,lab in enumerate(base):
        d1=np.full(lab.shape,np.inf);d2=d1.copy();y1=yy.copy();y2=yy.copy();x1=xx.copy();x2=xx.copy()
        for ident in np.unique(lab[lab>1]):
            d,idx=ndi.distance_transform_edt(lab!=ident,return_indices=True)
            win=d<d1;sec=(~win)&(d<d2)
            d2[win]=d1[win];y2[win]=y1[win];x2[win]=x1[win]
            d1[win]=d[win];y1[win]=idx[0][win];x1[win]=idx[1][win]
            d2[sec]=d[sec];y2[sec]=idx[0][sec];x2[sec]=idx[1][sec]
        with np.errstate(invalid="ignore",divide="ignore"):
            cosine=((yy-y1)*(yy-y2)+(xx-x1)*(xx-x2))/np.maximum(1e-9,d1*d2)
        seams=(lab<=1)&masks[k]&(d1<=6)&(d1+d2<=10)&(cosine<=-.25)&(fraction[k]>=.35)
        terrain=ndi.gaussian_gradient_magnitude(fraction[k],1.5)
        terrain/=max(1e-6,float(np.quantile(terrain[masks[k]],.95)))
        markers=np.where(lab>1,lab,0)
        filled=watershed(terrain,markers=markers,mask=(lab>1)|seams,compactness=.05)
        a=lab.copy();selected=seams&(filled>1);a[selected]=filled[selected]
        cc,count=ndi.label(a==1,np.ones((3,3)));sizes=np.bincount(cc.ravel());sizes[0]=0;main=int(sizes.argmax())
        absorb=np.zeros(lab.shape,bool)
        for component in range(1,count+1):
            island=cc==component
            if component!=main and sizes[component]<=750 and d1[island].max()<=12 and np.median(fraction[k][island])>=.35:
                absorb|=island
                topology_rows.append(dict(capture=int(captures[k]),operation='absorb_small_enclosed_pink_pocket',pixels=int(island.sum())))
        if absorb.any():
            markers=np.where(a>1,a,0)
            filled=watershed(terrain,markers=markers,mask=(a>1)|absorb,compactness=.05)
            a[absorb]=filled[absorb]
        # Preserve the main pink composite by restoring a minimal path through original pink only.
        restored=0
        for attempt in range(10):
            cc,count=ndi.label(a==1,np.ones((3,3)))
            if count==1:break
            sizes=np.bincount(cc.ravel());sizes[0]=0;main=int(sizes.argmax());other=int(np.flatnonzero((sizes>0)&(np.arange(len(sizes))!=main))[0])
            distance,indices=ndi.distance_transform_edt(cc!=main,return_indices=True)
            candidates=np.argwhere(cc==other);ey,ex=candidates[np.argmin(distance[cc==other])]
            sy,sx=int(indices[0,ey,ex]),int(indices[1,ey,ex])
            cost=np.where(lab==1,np.where(a==1,.01,1.),np.inf)
            path,_=route_through_array(cost,(sy,sx),(int(ey),int(ex)),fully_connected=True,geometric=True)
            path=np.array(path);restored+=int((a[path[:,0],path[:,1]]!=1).sum());a[path[:,0],path[:,1]]=1
        if restored:topology_rows.append(dict(capture=int(captures[k]),operation='restore_original_pink_connector',pixels=restored))
        assert ndi.label(a==1,np.ones((3,3)))[1]==1
        changed=a!=lab
        for ident in np.unique(a[changed]):
            p=changed&(a==ident)
            change_rows.append(dict(capture=int(captures[k]),new_candidate_id=int(ident),pixels=int(p.sum()),
                from_pink_pixels=int((p&(lab==1)).sum()),from_unassigned_pixels=int((p&(lab==0)).sum())))
        result.append(a);seam_masks.append(seams)
    return result[0],dict(status='applied',changes=change_rows,topology=topology_rows)


def reviewed_inputs(path):
    """Resolve a saved notebook or Partaker result's paired, ordered inputs."""
    path=Path(path)
    root=path.parent.parent
    manifest_path=root/'input_run'/'manifest.json'
    if manifest_path.is_file():
        payload=json.loads(manifest_path.read_text());base=manifest_path.parent
        groups=payload['groups']
    else:
        manifest_path=root/'analysis_manifest.json'
        payload=json.loads(manifest_path.read_text());base=root
        position=int(path.parent.name.removeprefix('position_'))
        entries=[dict(time=r['time'],file=r['file'],masks=r['masks'],colonies=[dict(track_id=i) for i in r['parents']]) for r in payload['inputs'] if r['position']==position]
        groups=[dict(position=position,channel=payload['settings']['cell_channel'],frames=entries)]
    if len(groups)!=1:raise ValueError('Select saved results for a single position')
    group=groups[0];group=dict(group,frames=sorted(group['frames'],key=lambda r:int(r['time'])))
    return manifest_path,base,group


def find_reviewed_results(folder):
    folder=Path(folder).expanduser()
    preferred=[folder/'region_ids.tif',folder/'results'/'region_ids.tif',folder/'current_crop15'/'results'/'region_ids.tif']
    candidates=preferred+sorted(folder.glob('position_*/region_ids.tif'))+sorted(folder.glob('**/results/region_ids.tif'))
    errors=[]
    for path in dict.fromkeys(candidates):
        if not path.is_file():continue
        try:reviewed_inputs(path);return path
        except (OSError,ValueError,KeyError) as error:errors.append(str(error))
    raise ValueError('No saved region_ids.tif with matching input images and masks was found. Select the notebook results folder or a Partaker microcolony results folder, rather than a masks-only PIV export.')


def sampled_frames(times, step=1):
    times = sorted(set(int(t) for t in times))
    return sorted(set(times[::step]+times[-1:]))


def darkness_map(image, mask, width=32):
    """Exact notebook box filter with an ROI-normalized denominator."""
    values = 1.0-image.astype(float)/255.0
    support = cv2.boxFilter(mask.astype(float),-1,(width,width),normalize=False,borderType=cv2.BORDER_CONSTANT)
    sums = cv2.boxFilter(np.where(mask,values,0),-1,(width,width),normalize=False,borderType=cv2.BORDER_CONSTANT)
    result = np.full(image.shape,np.nan)
    np.divide(sums,support,out=result,where=mask & (support>0))
    return np.clip(result,0,1),support/(width*width)


def split_regions(image, mask, params, *, check=None):
    """Exact notebook prominent peaks, spacing, watershed and reflooding."""
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
        if check is not None:
            check()
        sizes = np.bincount(labels.ravel())
        tiny = np.flatnonzero((sizes > 0) & (sizes < params['min_region_area']))
        tiny = tiny[tiny > 0]
        if not len(tiny):
            break
        remove = np.isin(markers, tiny) & (markers > 0)
        if not remove.any():
            raise ValueError("Microcolony partition could not remove a small region's marker. Analysis stopped to avoid an endless watershed loop.")
        markers[remove] = 0
        labels=flood(markers)
    labels,_,_ = segmentation.relabel_sequential(labels)
    return dict(darkness=d,smooth=np.where(mask,smooth,np.nan),labels=labels.astype(np.int32),markers=markers,support=support)


def match_ids(previous, labels, next_id, allow_match, match_iou=.30, match_margin=.10):
    """Retain an ID only for a reciprocal best overlap with a clear advantage."""
    curr_ids = np.unique(labels[labels > 0])
    pairs = []
    if allow_match:
        together = (previous > 0) & (labels > 0)
        values, counts = np.unique(np.column_stack((previous[together], labels[together])), axis=0, return_counts=True)
        old_sizes = dict(zip(*np.unique(previous[previous > 0], return_counts=True)))
        new_sizes = dict(zip(*np.unique(labels[labels > 0], return_counts=True)))
        for (old, new), intersection in zip(values, counts):
            iou = float(intersection / (old_sizes[old] + new_sizes[new] - intersection))
            pairs.append((int(old), int(new), iou))
    output = np.zeros(labels.shape, np.int32)
    info = {}
    for new in curr_ids:
        options = sorted((pair for pair in pairs if pair[1] == new), key=lambda pair: pair[2], reverse=True)
        iou = None
        advantage = None
        accepted = False
        if options:
            old, _, score = options[0]
            old_alternatives = [pair[2] for pair in pairs if pair[0] == old and pair[1] != new]
            new_alternative = options[1][2] if len(options) > 1 else 0.0
            advantage = score - max(new_alternative, max(old_alternatives, default=0.0))
            accepted = score >= match_iou and advantage >= match_margin
        if accepted:
            ident, iou, status = old, score, "reciprocal_overlap"
        else:
            ident = next_id
            next_id += 1
            status = "ambiguous_overlap" if options else "new_or_unmatched"
        output[labels == new] = ident
        info[int(new)] = dict(display_id=int(ident), match_iou=iou, match_advantage=advantage, identity_status=status)
    return output, info, next_id


def recover_region_continuity(ids, info, labels, history, image, capture, minimum_iou, check):
    """Recover missing IDs only when checked texture motion supports a unique target."""
    used = set(int(i) for i in np.unique(ids[ids > 0]))
    proposals = []
    height, width = image.shape
    for ident, (source_capture, source_mask, source_image) in history.items():
        check()
        if ident in used or not 1 <= capture - source_capture <= 2:
            continue
        vectors, affine = image_motion(source_image, image, source_mask)
        if affine is None:
            continue
        projected = cv2.warpAffine(source_mask.astype(np.uint8), affine, (width, height),
                                   flags=cv2.INTER_NEAREST).astype(bool)
        choices = []
        for local, metadata in info.items():
            if metadata["identity_status"] in ("unique_overlap", "reciprocal_overlap"):
                continue
            target = labels == local
            intersection = int((projected & target).sum())
            union = int((projected | target).sum())
            iou = intersection / max(1, union)
            if iou >= minimum_iou:
                choices.append((iou, local))
        choices.sort(reverse=True)
        if choices and (len(choices) == 1 or choices[0][0] - choices[1][0] >= 0.1):
            iou, local = choices[0]
            proposals.append((local, ident, iou, source_capture, len(vectors)))
    for local, ident, iou, source_capture, points in proposals:
        if sum(candidate[0] == local for candidate in proposals) != 1:
            continue
        ids[labels == local] = ident
        info[local] = dict(display_id=ident, match_iou=iou, identity_status="motion_supported_continuity",
                           continuity_source_capture=source_capture, continuity_support_points=points)
    return ids, info


def image_motion(previous, current, domain, *, max_corners=300):
    """Forward/backward-checked raw texture displacement and robust affine fit.

    Numerical checks from the motion-memory notebook: FB <=1px, 9x9 patch NCC
    >=.60 in locally normalized and enhanced images, displacement <=120px.
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
    points=cv2.goodFeaturesToTrack(a,maxCorners=max_corners,qualityLevel=.01,minDistance=10,mask=domain.astype(np.uint8)*255,blockSize=5)
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


class CancelledAnalysis(Exception):
    pass


class DispersionAnalysis:
    backend=BACKEND

    def __init__(self,image_data,settings,positions,times,output_root,*,progress=None,cancelled=None,edits=None,selector_params=None,
                 dispersion_timing=None, focus_intervals=(), identity_overrides=None):
        settings.validate();self.data=image_data;self.settings=settings
        self.positions=sorted(set(int(p) for p in positions));self.sampled=sampled_frames(times,settings.sample_step)
        if not self.positions or not self.sampled:
            raise ValueError('Select positions and frames first')
        self.output=Path(output_root)/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
        self.dispersion_timing=dispersion_timing
        self.focus_intervals=tuple(focus_intervals)
        self.identity_overrides=dict(identity_overrides or {})
        self.frames=list(self.sampled);self.kept=list(self.positions);self.preview_key=None
        self.progress=progress or (lambda *_:None);self.cancelled=cancelled or (lambda:False)
        self.selector_params=selector_params or {};self.edits=edits or {};self.pink_masks={};self.reviewed_regions={};self.pink_provenance={'method':'unavailable'}
        self.colony_rows=[];self.frame_rows=[];self.track_rows=[];self.motion_rows=[];self.motion_class_rows=[];self.dark_sensitivity_rows=[];self.inputs={};self.prepared_colonies={}
        self.output.mkdir(parents=True);(self.output/'inputs').mkdir();(self.output/'masks').mkdir()
        self._transform=(json.dumps(getattr(image_data,'crop_coordinates',None)),None if getattr(image_data,'registration_offsets',None) is None else np.array(image_data.registration_offsets,copy=True))
        self.write_manifest('preparing')

    def check(self):
        if self.cancelled():
            raise CancelledAnalysis('Analysis cancelled; partial output is marked incomplete')
        if json.dumps(getattr(self.data,'crop_coordinates',None))!=self._transform[0]:
            raise ValueError('Image crop changed while processing; prepare masks again')
        current=getattr(self.data,'registration_offsets',None);prior=self._transform[1]
        if (current is None)!=(prior is None) or current is not None and not np.array_equal(current,prior):
            raise ValueError('Image registration changed while processing')

    def add_input(self,key,raw,colonies):
        """Snapshot the exact verifier geometry/dark masks before background work."""
        self.check();p,t,c=map(int,key);raw=np.asarray(raw)
        if raw.ndim!=2 or min(raw.shape)<=2*BORDER or not np.isfinite(raw).all():
            raise ValueError(f'Expected a finite grayscale frame at {key}')
        masks={};parents=[];snapshot=[]
        for index,colony in enumerate(colonies,1):
            outer=np.asarray(colony['mask'],bool).copy()
            inner=np.asarray(colony['inner_mask'],bool).copy()
            if outer.shape!=raw.shape or inner.shape!=raw.shape:
                raise ValueError(f'Selector masks and raw image have different coordinates at {key}')
            valid=np.zeros(raw.shape,bool);valid[BORDER:-BORDER,BORDER:-BORDER]=True
            outer &= valid;inner &= outer
            parent=int(colony.get('track_id') or colony.get('colony_id') or index)
            if parent in parents:
                raise ValueError(f'Duplicate parent ROI ID at {key}')
            parents.append(parent);masks[f'colony_{parent}']=outer;masks[f'inner_{parent}']=inner
            snapshot.append(dict(colony,mask=outer.copy(),inner_mask=inner.copy()))
        if not parents:
            raise ValueError(f'No outer ROI at position {p}, frame {t}. Set the biofilm boundary before running analysis.')
        name=f'p{p}_t{t}_c{c}'
        np.save(self.output/'inputs'/f'{name}.npy',raw,allow_pickle=False)
        np.savez_compressed(self.output/'inputs'/f'{name}_masks.npz',**masks)
        self.inputs[(p,t)]=dict(file=f'inputs/{name}.npy',masks=f'inputs/{name}_masks.npz',parents=parents,
            raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),masks_sha256={k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in masks.items()})
        self.prepared_colonies[(p,t,c)]=snapshot

    def prepare_recording(self):
        """Detect a single outer biofilm on each selected phase-contrast frame."""
        frames = [(p, t) for p in self.positions for t in self.sampled]
        previous = {}
        self.selector_params.update(outer_mask_method="verifier_otsu_adjusted",
                                    min_colony_area_percent=60, smoothing_kernel_percent=46,
                                    smoothing_kernel_px=37,
                                    intensity_threshold_percent=self.settings.colony_threshold_percent)
        if self.settings.dark_measurement == 'intensity_50':
            self.selector_params.update(inner_threshold=50, inner_min_area=1, inner_kernel=1)
        for index, (position, capture) in enumerate(frames):
            self.check()
            self.progress(15 * index // len(frames), f"Detecting biofilm · P={position}, T={capture}")
            channel = self.settings.cell_channel
            raw = np.asarray(self.data.get(capture, position, channel))
            if raw.ndim != 2 or min(raw.shape) <= 2 * BORDER or not np.isfinite(raw).all():
                raise ValueError(f"Expected a finite phase-contrast image at P={position}, T={capture}.")
            seed = self.selector_params.get("colony_seed_points", {}).get(position)
            outer = whole_colony_mask(raw, self.settings.colony_threshold_percent,
                                      previous=previous.get(position), seed=seed)
            previous[position] = outer
            self.add_input((position, capture, channel), raw, [
                dict(track_id=1, mask=outer, inner_mask=inner_dark_mask(raw, outer)
                     if self.settings.dark_measurement == 'intensity_50' else np.zeros_like(outer)),
            ])

    def prepare_reviewed_recording(self):
        """Check and import the paired outer masks for an explicitly selected notebook run."""
        _, base, group = reviewed_inputs(self.settings.pink_source)
        self.selector_params["outer_mask_method"] = "paired_reviewed_masks"
        position, channel = int(group["position"]), self.settings.cell_channel
        for entry in group["frames"]:
            self.check()
            capture = int(entry["time"])
            raw = np.asarray(self.data.get(capture, position, channel))
            reference = np.load(base / entry["file"], allow_pickle=False)
            cropped = reference.shape == crop(raw, BORDER).shape
            current = crop(raw, BORDER) if cropped else raw
            if reference.dtype != current.dtype or not np.array_equal(reference, current):
                raise ValueError(f"Saved image differs at P={position}, T={capture}. Load the matching recording.")
            colonies = []
            with np.load(base / entry["masks"], allow_pickle=False) as masks:
                for region in entry["colonies"]:
                    ident = int(region["track_id"])
                    outer = pad_back(masks[f"colony_{ident}"].astype(bool), BORDER if cropped else 0, raw.shape)
                    inner = pad_back(masks[f"inner_{ident}"].astype(bool), BORDER if cropped else 0, raw.shape)
                    colonies.append(dict(track_id=ident, mask=outer, inner_mask=inner))
            self.add_input((position, capture, channel), raw, colonies)

    def detect_pink(self):
        """Find a persistent local outward expansion and follow its connected growth."""
        provenance = self.pink_provenance = dict(method="automatic_connected_expansion", candidate_frames=[], uncertain_frames=[])
        for position in self.positions:
            rois = []
            for capture in self.sampled:
                self.check()
                masks = self.source_masks(position, capture)
                rois.append(np.logical_or.reduce([
                    masks[f"colony_{parent}"] for parent in self.inputs[position, capture]["parents"]
                ]))
            if len(rois) < 3:
                provenance["uncertain_frames"].extend([position, t] for t in self.sampled)
                continue
            # Use a majority baseline and require the expansion to persist through the final two frames.
            baseline_count = min(3, len(rois) - 2)
            baseline = np.count_nonzero(rois[:baseline_count], axis=0) > baseline_count / 2
            late = rois[-2] & rois[-1]
            labels, _ = components(late & ~baseline)
            sizes = np.bincount(labels.ravel())[1:]
            minimum = max(60, int(0.002 * baseline.sum()))
            if not sizes.size or sizes.max() < minimum:
                provenance["uncertain_frames"].extend([position, t] for t in self.sampled)
                continue
            largest = int(sizes.argmax())
            candidate = labels == largest + 1
            competing = sizes.size > 1 and np.partition(sizes, -2)[-2] >= 0.65 * sizes[largest]
            for index in range(baseline_count, len(rois)):
                seen = candidate & rois[index]
                if seen.sum() < minimum // 2:
                    continue
                # Preserve the first connected expansion as the anchor and use the existing growth rules thereafter.
                seeds, _ = components(seen)
                seed = seeds == np.bincount(seeds[seen])[1:].argmax() + 1
                gate = (ndi.distance_transform_edt(~candidate) <= 16) & ~baseline
                history, _, _ = grow_from_enclosed_roi(seed, rois[index:], gate)
                for capture, mask, roi in zip(self.sampled[index:], history, rois[index:]):
                    self.check()
                    connected, _, _, _ = make_connected_region(mask, seed, roi)
                    self.pink_masks[position, capture] = connected
                    if connected.any():
                        provenance["candidate_frames"].append([position, capture])
                    if competing or not connected.any():
                        provenance["uncertain_frames"].append([position, capture])
                break

    def raw(self,p,t):
        return np.load(self.output/self.inputs[p,t]['file'],allow_pickle=False)

    def source_masks(self,p,t):
        with np.load(self.output/self.inputs[p,t]['masks'],allow_pickle=False) as z:
            return {k:z[k] for k in z.files}

    def mask_path(self,p,t):
        return self.output/'masks'/f'p{p}_t{t}.npz'

    def load_frame(self,p,t):
        with np.load(self.mask_path(p,t),allow_pickle=False) as z:
            return {k:z[k] for k in z.files}

    def write_csv(self,name,rows,fields=None):
        fields=fields or list(dict.fromkeys(k for row in rows for k in row)) or ['capture']
        with (self.output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

    def write_manifest(self,status):
        payload=dict(backend=BACKEND,status=status,settings=asdict(self.settings),selector_params=self.selector_params,
                     continuity=dict(method="reciprocal_best_overlap",minimum_iou=self.settings.match_iou,
                                     minimum_advantage=self.settings.match_margin,
                                     temporal_boundary_weight=self.settings.temporal_weight,
                                     maximum_missing_frames=1),
                     positions=self.positions,frames=self.frames,preview_key=self.preview_key,
                     intensity_conversion=getattr(self,'intensity_conversion',None),pink_region=self.pink_provenance,
                     source_image=str(getattr(self.data,'image_filename','')),
                     coordinate_system='source analysis-image pixels; mask detection excludes 15px image border',border_px=BORDER,
                     inputs=[dict(position=p,time=t,**value) for (p,t),value in sorted(self.inputs.items())],
                     dark_area_methods=dict(intensity_50='Per-frame crop min-max normalization to uint8; Gaussian 5x5; intensity <=128 (50%); morphology 1px; eight-connected area >=1px; current colony territory',inside_roi='exact saved Inside ROI',hybrid='32px fraction, 0.65/0.35 validated temporal blend; within 3px; 0.65 new / 0.55 temporal fill',raw_z='Gaussian 0.6/7px; SD floor 5; z>0.8; eight-connected area >=4px'),
                     note='Dark-area method is explicit. Saved Inside ROI is retained when available; inferred support and normalized raw-image detection have separate fields. Region territory includes valley pixels. Counts are image regions/dark objects, not individual cells. Motion uses raw-image evidence; unsupported motion stays unavailable.')
        if self.pink_provenance.get('method') == 'central_shared_edges':
            payload['continuity'] = dict(method='checked_affine_and_partial_texture_correspondence',
                retention_belief_threshold=.65, maximum_proposal_area_factor=1.3,
                persistent_structure_minimum_observations=5,
                fragment_maximum_gap_px=8, reset_on_capture_or_focus_gap=True)
            payload['dark_area_methods']['raw_z'] = (
                'Mean and SD of raw intensity at sigma 7px; foreground sigma 0.6px; '
                'SD floor 5; z>0.8; eight-connected area >=4px; current colony territory')
        if self.dispersion_timing is not None:
            payload['dispersion_timing']=self.dispersion_timing
            payload['focus_intervals']=self.focus_intervals
            payload['region_id_edits']=[
                dict(position=p, capture=t, mask_sha256=signature, region_id=ident)
                for (p,t,signature),ident in self.identity_overrides.items()
            ]
        if self.settings.dark_measurement == 'intensity_50':
            payload['inner_dark_detector'] = dict(threshold_percent=50, threshold_uint8=128,
                comparison='<=', minimum_area_px=1, morphology_kernel_px=1,
                gaussian_kernel_px=[5, 5], gaussian_sigma=0, connectivity=8,
                normalization='per_frame_analysis_crop_min_max_to_uint8',
                domain='current_colony_territory',
                breakup_formula='1-largest_connected_dark_area/total_dark_area')
        (self.output/'analysis_manifest.json').write_text(json.dumps(payload,indent=2,allow_nan=False))

    def track_outer_inputs(self):
        """Keep parent ROI identities compatible with the existing exporter."""
        if all(c.get('track_id') is not None for values in self.prepared_colonies.values() for c in values):
            return
        from nd2_analyzer.analysis.tracking.colony_tracking import track_colonies_by_frame
        self.check()
        self.prepared_colonies=track_colonies_by_frame(self.prepared_colonies)
        for (p,t,c),colonies in self.prepared_colonies.items():
            self.check();masks={};parents=[]
            for colony in colonies:
                parent=int(colony['track_id']);parents.append(parent)
                masks[f'colony_{parent}']=np.asarray(colony['mask'],bool)
                masks[f'inner_{parent}']=np.asarray(colony['inner_mask'],bool)&masks[f'colony_{parent}']
            np.savez_compressed(self.output/self.inputs[p,t]['masks'],**masks)
            self.inputs[p,t]['parents']=parents
            self.inputs[p,t]['masks_sha256']={k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in masks.items()}

    def prepare_pink(self):
        """Reuse reviewed pink ID 1 only when paired raw frames match exactly.

        A drawn seed instead follows the notebook's enclosed-ROI growth rule and
        bounded connection repair. It is a reviewed compartment, not a flow class.
        """
        import tifffile
        explicit=self.settings.pink_source.strip()
        candidates=[Path(explicit).expanduser()] if explicit else []
        if not explicit and self.dispersion_timing is None:
            candidates=sorted(
                (Path.home()/'Documents'/'Codex').glob('*/t/outputs/microcolony_motion_memory/current_crop15/results/region_ids.tif'),reverse=True)
        failures=[]
        for path in candidates:
            self.check()
            if self.edits:break  # an explicit drawn seed takes priority
            try:
                manifest_path,input_base,group=reviewed_inputs(path)
                stack=tifffile.imread(path)
                pink_path=path.parent/'pink_mask.tif'
                pink_stack=tifffile.imread(pink_path).astype(bool) if pink_path.is_file() else stack==1
                if pink_stack.shape!=stack.shape:raise ValueError('Saved pink mask shape differs from region IDs')
                entries=group['frames']
                if stack.ndim!=3 or len(stack)!=len(entries):raise ValueError('Reviewed mask/frame count mismatch')
                mapping={int(f['time']):(i,f) for i,f in enumerate(entries)}
                if int(group['channel'])!=self.settings.cell_channel:raise ValueError('Reviewed phase channel differs')
                position=int(group['position'])
                if position not in self.positions:raise ValueError('Reviewed position is not selected')
                prepared={};reviewed={}
                for t in self.sampled:
                    self.check();i,entry=mapping[t];raw=self.raw(position,t)
                    reference=np.load(input_base/entry['file'],allow_pickle=False)
                    cropped=reference.shape==raw[BORDER:-BORDER,BORDER:-BORDER].shape
                    current=raw[BORDER:-BORDER,BORDER:-BORDER] if cropped else raw
                    if reference.dtype!=current.dtype or not np.array_equal(reference,current):
                        raise ValueError(f'Reviewed raw image does not match P={position}, T={t}')
                    mask=pink_stack[i]
                    if mask.shape!=reference.shape:raise ValueError('Reviewed mask coordinates differ from paired raw image')
                    full=np.zeros(raw.shape,bool)
                    if cropped:full[BORDER:-BORDER,BORDER:-BORDER]=mask
                    else:full[:]=mask
                    prepared[position,t]=full
                    owner=np.zeros(raw.shape,np.int32)
                    if cropped:owner[BORDER:-BORDER,BORDER:-BORDER]=stack[i]
                    else:owner[:]=stack[i]
                    if not np.issubdtype(stack.dtype,np.integer) or np.any(owner<0):raise ValueError('Reviewed region IDs must be nonnegative integers')
                    reviewed[position,t]=owner
                self.pink_masks=prepared;self.reviewed_regions=reviewed
                self.pink_provenance=dict(method='reviewed_notebook_id_1',source=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),paired_manifest=str(manifest_path),
                    accepted_partition='all canonical IDs, including pink, recovered colonies, shared borders and continuity aliases',
                    matched_raw_frames=len(prepared),position=position,coordinates='paired input; 15px crop restored when applicable')
                return
            except (OSError,ValueError,KeyError,IndexError) as error:
                failures.append(str(error))
        if explicit and not self.edits:
            raise ValueError('Cannot use reviewed pink mask: '+('; '.join(failures) or 'missing source'))
        if not self.edits and self.dispersion_timing is not None:
            self.pink_provenance = dict(method="motion_classification", status="pending",
                                        identity_policy="motion class preserves region IDs")
            return
        if not self.edits:
            self.pink_provenance=dict(method='unavailable',reason='No matching reviewed notebook masks; draw a pink seed for this sequence.')
            return
        if self.settings.sample_step!=1 or any(b-a!=1 for a,b in zip(self.sampled,self.sampled[1:])):
            raise ValueError('Drawn pink growth requires every consecutive frame (Analyze every N frames = 1)')
        details=[]
        for p,edit in self.edits.items():
            p=int(p)
            if p not in self.positions:continue
            if int(edit['capture'])!=self.sampled[0]:
                raise ValueError('Draw the pink seed on the first selected frame, then analyze the same frame range')
            polygon=np.asarray(edit['polygon'],np.float64)
            if polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<3 or not np.isfinite(polygon).all():
                raise ValueError('Pink seed requires a finite polygon with at least three vertices')
            shape=self.raw(p,self.sampled[0]).shape
            seed=np.zeros(shape,np.uint8);cv2.fillPoly(seed,[np.rint(polygon).astype(np.int32)],1);seed=seed.astype(bool)
            rois=[]
            for t in self.sampled:
                masks=self.source_masks(p,t);roi=np.zeros(shape,bool)
                for parent in self.inputs[p,t]['parents']:roi |= masks[f'colony_{parent}']
                rois.append(roi)
            if not (seed & rois[0]).any():raise ValueError('Pink seed is outside the selected biofilm')
            tip=polygon[-1];axis=tip-polygon.mean(axis=0);length=np.linalg.norm(axis)
            if length<1:raise ValueError('Finish the pink polygon at its growing tip')
            axis/=length;yy,xx=np.indices(shape)
            gate=(xx-tip[0])*axis[0]+(yy-tip[1])*axis[1]>=-8
            history,additions,component_rows=grow_from_enclosed_roi(seed,rois,gate,attachment=8.,minimum_area=60)
            for t,mask,roi in zip(self.sampled,history,rois):
                self.check();connected,bridge,removed,counts=make_connected_region(mask,seed,roi,radius=16,interface_factor=2,debris_limit=4)
                self.pink_masks[p,t]=connected
                details.append(dict(position=p,capture=t,bridge_area_px=int(bridge.sum()),removed_debris_px=int(removed.sum()),**counts))
        self.pink_provenance=dict(method='drawn_seed_notebook_growth',seeds=self.edits,attachment_px=8,
            minimum_gain_area_px=60,distal_margin_px=8,connection_radius_px=16,connection_interface_factor=2,
            debris_limit_px=4,connectivity=details)

    def apply_region_id_edits(self, position, capture, labels, ids, info):
        edits = {signature: ident for (p, t, signature), ident in self.identity_overrides.items()
                 if (p, t) == (position, capture)}
        applied = set()
        for local, metadata in info.items():
            mask = labels == local
            signature = hashlib.sha256(mask.tobytes()).hexdigest()
            if signature in edits:
                ident = edits[signature]
                if ident != metadata['display_id'] and (ids == ident).any():
                    raise ValueError(f'Region ID {ident} is already assigned at P={position}, T={capture}')
                ids[mask]=ident
                info[local]=dict(display_id=ident,match_iou=None,identity_status='manual_region_id')
                applied.add(signature)
        return applied

    def elapsed(self, start, end):
        hours = self.dispersion_timing.get('hours') if self.dispersion_timing is not None else None
        return (end - start) * self.settings.interval if hours is None else hours[end] - hours[start]

    def run(self,preview_key=None, *, parameter_sensitivity=True, export_figures=True):
        self.preview_key=preview_key;s=self.settings;params=asdict(s)
        time_unit = 'h' if self.dispersion_timing and self.dispersion_timing.get('hours') is not None else s.time_unit
        required=[(p,t) for p in self.positions for t in self.sampled]
        if set(required)-set(self.inputs):
            raise ValueError('Missing in-window input snapshots; prepare the selected masks first')
        self.write_manifest('running')
        try:
            self.track_outer_inputs()
            automatic = self.dispersion_timing is not None and preview_key is None and not s.pink_source and not self.edits
            if not automatic:
                self.prepare_pink()
            # Fixed run-level conversion, exactly as the notebook (identity for uint8).
            low=min(float(self.raw(*key).min()) for key in required)
            high=max(float(self.raw(*key).max()) for key in required)
            if all(self.raw(*key).dtype==np.uint8 for key in required):
                self.intensity_conversion=dict(method='identity_uint8',low=0,high=255)
            else:
                if high<=low:raise ValueError('The selected raw image series has no intensity range')
                self.intensity_conversion=dict(method='fixed_run_min_max',low=low,high=high)
            def gray(p,t):
                a=self.raw(p,t)
                if self.intensity_conversion['method']=='identity_uint8':return a
                return np.clip((a.astype(float)-low)/(high-low)*255,0,255).astype(np.uint8)
            applied_edits=set()
            previous={};previous_gray={};previous_outer={};previous_support={};previous_surfaces={};continuity_history={};pink_ids={};next_id=max([int(a.max()) for a in self.reviewed_regions.values()]+[0])+1;self._baseline={}
            if automatic:
                self.initialize_automatic_frames(gray)
            for index,(p,t) in enumerate([] if automatic else required):
                self.check();self.progress(15 + int(65*index/len(required)),f'Measuring microcolonies: P={p}, T={t}')
                image=gray(p,t);masks=self.source_masks(p,t);shape=image.shape
                regions=np.zeros(shape,np.int32);frame_labels=np.zeros(shape,np.int32)
                outer_union=np.zeros(shape,bool);inner_union=np.zeros(shape,bool);darkness=np.full(shape,np.nan);markers=np.zeros(shape,np.int32)
                records=[];frame_motion=[];label_offset=0
                pink_union=np.zeros(shape,bool);measured_union=np.zeros(shape,bool);hybrid_union=np.zeros(shape,bool);raw_union=np.zeros(shape,bool)
                fraction_full=np.zeros(shape,float);z_full=np.zeros(shape,float)
                prior_t=previous_gray.get(p,(None,None))[0]
                consecutive=prior_t is not None and t-prior_t==1
                for parent in self.inputs[p,t]['parents']:
                    outer=masks[f'colony_{parent}'];inner=masks[f'inner_{parent}']
                    if np.any(outer_union & outer):
                        raise ValueError(f'Outer ROIs overlap at P={p}, T={t}; trim their overlap before analysis')
                    outer_union |= outer;inner_union |= inner
                    support_key=(p,parent)
                    prior=previous_support.get(support_key)
                    fraction,hybrid,dark_z=notebook_support(image,outer,inner,prior[1] if prior is not None and prior[0]==prior_t and consecutive else None)
                    previous_support[support_key]=(t,(image.copy(),outer.copy(),fraction.copy(),hybrid.copy()))
                    raw_mask=threshold_dark(dark_z,outer) > 0
                    for threshold in (.5,.8,1.1):
                        for minimum in (1,4,9):
                            objects=threshold_dark(dark_z,outer,threshold,minimum)
                            self.dark_sensitivity_rows.append(dict(position=p,capture=t,parent_roi_id=parent,detector='normalized_raw_dark_z',threshold=threshold,minimum_area_px=minimum,object_count=int(len(np.unique(objects[objects>0]))),dark_area_px=int((objects>0).sum())))
                    measured={'inside_roi':inner,'hybrid':hybrid,'raw_z':raw_mask,
                              'intensity_50':inner_dark_mask(self.raw(p,t), outer)}[s.dark_measurement]
                    measured_union |= measured;hybrid_union |= hybrid;raw_union |= raw_mask
                    fraction_full[outer]=fraction[outer];z_full[outer]=dark_z[outer]
                    domain=outer if s.mask_kind=='colony' else inner
                    if (p,t) in self.reviewed_regions:
                        d,_=darkness_map(image,domain,s.width)
                        res=dict(darkness=d,markers=np.zeros(shape,np.int32))
                        labels=np.where(domain,self.reviewed_regions[p,t],0).astype(np.int32)
                    else:
                        res = split_regions(image, domain, params, check=self.check)
                        surface = np.nan_to_num(res["smooth"])
                        prior_surface = previous_surfaces.get((p, parent))
                        if consecutive and prior_surface is not None and prior_surface[0] == prior_t:
                            _, previous_image, previous_outer_mask, old_surface = prior_surface
                            flow = flow_pair(previous_image, image, previous_outer_mask, outer)
                            transported = flow["remap"](old_surface.astype(np.float32))
                            weight = s.temporal_weight
                            surface = np.where(flow["valid"], (1 - weight) * surface + weight * transported, surface)
                            markers = np.zeros(shape, np.int32)
                            for local in np.unique(res["labels"][res["labels"] > 0]):
                                mask = res["labels"] == local
                                y, x = np.unravel_index(np.argmax(np.where(mask, surface, -1)), shape)
                                markers[y, x] = local
                            if markers.any():
                                res["labels"] = segmentation.watershed(-surface, markers, mask=domain, connectivity=2).astype(np.int32)
                                res["markers"] = markers
                        previous_surfaces[p, parent] = (t, image, outer, surface)
                        labels = res["labels"].copy()
                    pink=self.pink_masks.get((p,t),np.zeros(shape,bool)) & domain
                    if pink.any() and (p,t) not in self.reviewed_regions:
                        temporary=np.where(pink,1,np.where(labels>0,labels+1,0)).astype(np.int32)
                        temporary,audit=shared_border_partition(temporary,domain,fraction)
                        self.pink_provenance.setdefault('shared_border_updates',[]).append(dict(position=p,capture=t,parent_roi_id=parent,**audit))
                        pink=temporary==1
                        self.pink_masks[p,t][domain]=pink[domain]
                        labels=np.where(temporary>1,temporary-1,0).astype(np.int32)
                    labels[pink]=0
                    labels,_,_=segmentation.relabel_sequential(labels)
                    reviewed=self.reviewed_regions.get((p,t))
                    if reviewed is not None:
                        # The final notebook partition already contains every reviewed
                        # pink release, seam assignment, recovered ROI and accepted alias.
                        labels=np.where(domain,reviewed,0).astype(np.int32);ids=labels.copy()
                        info={int(i):dict(display_id=int(i),match_iou=None,identity_status='reviewed_notebook_partition') for i in np.unique(labels[labels>0])}
                        pink_local=1 if (labels==1).any() else None
                    else:
                        old=previous.get((p,parent),(None,np.zeros(shape,np.int32)))
                        # Without an existing parent track ID, a radically changed outer
                        # ROI must not silently inherit another parent's display IDs.
                        prior_outer=previous_outer.get((p,parent))
                        outer_iou=float((outer & prior_outer).sum()/max(1,(outer|prior_outer).sum())) if prior_outer is not None else 0.
                        allow=consecutive and old[0]==prior_t and outer_iou>=s.match_iou
                        ids,info,next_id=match_ids(old[1],labels,next_id,allow,s.match_iou,s.match_margin)
                        history = continuity_history.setdefault((p, parent), {})
                        if allow:
                            ids, info = recover_region_continuity(
                                ids, info, labels, history, image, t, s.match_iou, self.check
                            )
                        if self.identity_overrides:
                            applied = self.apply_region_id_edits(p,t,labels,ids,info)
                            applied_edits.update((p,t,signature) for signature in applied)
                            next_id=max(next_id,int(ids.max())+1)
                        for ident in list(history):
                            if t - history[ident][0] > 1:
                                del history[ident]
                        for local, metadata in info.items():
                            history[metadata["display_id"]] = (t, labels == local, image)
                        previous[(p,parent)]=(t,ids.copy());previous_outer[(p,parent)]=outer
                        pink_local=None
                        if pink.any():
                            if (p,parent) not in pink_ids:
                                pink_ids[p,parent]=next_id;next_id+=1
                            pink_local=int(labels.max())+1;labels[pink]=pink_local;ids[pink]=pink_ids[p,parent]
                            info[pink_local]=dict(display_id=pink_ids[p,parent],match_iou=None,identity_status='automatic_pink_compartment' if not self.edits else 'reviewed_pink_compartment')
                    pink_union |= pink & domain
                    if self.identity_overrides and np.intersect1d(regions[regions>0],ids[ids>0]).size:
                        raise ValueError(f'Region ID assigned to multiple colonies at P={p}, T={t}')
                    regions[domain]=ids[domain];darkness[domain]=res['darkness'][domain]
                    marked=res['markers']>0;markers[marked]=res['markers'][marked]+label_offset
                    frame_labels[labels>0]=labels[labels>0]+label_offset
                    label_offset+=int(labels.max())
                    boundary=domain & ~ndi.binary_erosion(domain)
                    for prop in measure.regionprops(labels,intensity_image=image):
                        local=int(prop.label);ident=info[local]['display_id'];territory=labels==local;cell_area=territory & measured
                        count_labels,n=ndi.label(cell_area,structure=np.ones((3,3)))
                        sizes=np.bincount(count_labels.ravel());object_count=int(np.count_nonzero(sizes[1:]>=4))
                        dark_area=int(cell_area.sum());key=(p,ident)
                        self._baseline.setdefault(key,dark_area)
                        base=self._baseline[key]
                        row=dict(position=p,channel=s.cell_channel,parent_roi_id=parent,capture=t,elapsed_time=self.elapsed(self.sampled[0],t),time_unit=time_unit,
                            frame_label=local,dark_measurement=s.dark_measurement,strict_inner_area_px=int((territory&inner).sum()),
                            hybrid_supported_area_px=int((territory&hybrid).sum()),inferred_gap_area_px=int((territory&hybrid&~inner).sum()),
                            normalized_dark_area_px=int((territory&raw_mask).sum()),region_type='central_pink' if (reviewed is not None and local == 1) or (reviewed is None and local == pink_local) else 'microcolony',**info[local],territory_area_px=int(prop.area),dark_area_px=dark_area,
                            dark_area_fraction=float(dark_area/max(1,prop.area)),dark_area_ratio_to_first=dark_area/base if base else None,
                            dark_object_count_min4px=object_count,centroid_x_px=float(prop.centroid[1]),centroid_y_px=float(prop.centroid[0]),
                            mean_gray_0_255=float(prop.intensity_mean),mean_local_darkness=float(np.nanmean(res['darkness'][territory])),
                            perimeter_px=float(prop.perimeter),eccentricity=float(prop.eccentricity),solidity=float(prop.solidity),
                            touches_roi_boundary=bool(np.any(territory & boundary)))
                        if s.pixel_um>0:
                            row.update(territory_area_um2=float(prop.area)*s.pixel_um**2,dark_area_um2=dark_area*s.pixel_um**2)
                        records.append(row);self.colony_rows.append(row)
                if consecutive and s.motion:
                    prev_image=previous_gray[p][1]
                    vectors,affine=image_motion(prev_image,image,previous_gray[p][2])
                    for vector in vectors:
                        x,y=np.rint(vector['start']).astype(int)
                        ident=int(previous_gray[p][3][y,x])
                        vector.update(position=p,capture=t,source_capture=prior_t,display_id=ident,
                                      elapsed_interval=self.elapsed(prior_t,t),time_unit=time_unit)
                        frame_motion.append(vector);self.motion_rows.append(vector)
                previous_gray[p]=(t,image,measured_union.copy(),regions.copy())
                self.frame_rows.append(dict(position=p,capture=t,elapsed_time=self.elapsed(self.sampled[0],t),time_unit=time_unit,
                    region_count=len(records),outer_area_px=int(outer_union.sum()),dark_measurement=s.dark_measurement,dark_area_px=int(measured_union.sum()),strict_inner_area_px=int(inner_union.sum()),
                    measured_dark_area_px=int(measured_union.sum()),hybrid_supported_area_px=int(hybrid_union.sum()),normalized_dark_area_px=int(raw_union.sum()),
                    assigned_territory_px=int((regions>0).sum()),assigned_dark_area_px=int(((regions>0)&measured_union).sum()),
                    unassigned_dark_area_px=int(((regions==0)&measured_union).sum()),supported_motion_vectors=len(frame_motion)))
                np.savez_compressed(self.mask_path(p,t),raw_gray=image,outer_mask=outer_union,inner_mask=inner_union,
                    region_ids=regions,frame_labels=frame_labels,pink_mask=pink_union,measured_mask=measured_union,hybrid_mask=hybrid_union,raw_dark_mask=raw_union,inner_fraction=fraction_full,dark_z=z_full,cell_region_ids=np.where(measured_union,regions,0).astype(np.int32),strict_inner_roi_ids=np.where(inner_union,regions,0).astype(np.int32),hybrid_inner_roi_ids=np.where(hybrid_union,regions,0).astype(np.int32),darkness=darkness,markers=markers,
                    records_json=np.array(json.dumps(records)),vectors_json=np.array(json.dumps(frame_motion)))
            if not automatic and set(self.identity_overrides)-applied_edits:
                raise ValueError('A region ID edit no longer matches the selected masks or scope. Clear ID edits and rerun.')
            self.track_rows=self.colony_rows
            if self.dispersion_timing is not None and preview_key is None:
                if not automatic and not self.reviewed_regions and not self.edits:
                    self.progress(80, 'Classifying region motion')
                    self.classify_region_motion()
                    self.write_csv('region_motion.csv', self.motion_class_rows)
                self.progress(15 if automatic else 84, 'Refining colony territories')
                self.refine_territories(automatic=automatic)
            self.write_csv('region_properties.csv',self.colony_rows)
            self.write_csv('region_continuity.csv', [
                {key: row.get(key) for key in ("position", "capture", "display_id", "identity_status",
                                              "match_iou", "match_advantage", "continuity_source_capture", "continuity_support_points")}
                for row in self.colony_rows
            ])
            self.write_csv('pink_region_properties.csv',[r for r in self.colony_rows if r['region_type']=='central_pink'])
            self.write_csv('frame_summary.csv',self.frame_rows)
            self.write_csv('dark_area_sensitivity.csv',self.dark_sensitivity_rows)
            self.write_csv('image_motion_vectors.csv',[{k:v for k,v in r.items() if k not in ('start','end')}|dict(x_px=r['start'][0],y_px=r['start'][1],target_x_px=r['end'][0],target_y_px=r['end'][1]) for r in self.motion_rows])
            if parameter_sensitivity and not getattr(self,'loaded_saved',False):
                self.progress(83,'Checking parameter sensitivity')
                self.sensitivity(gray)
            if self.dispersion_timing is not None and preview_key is None:
                self.progress(94, 'Measuring microcolony fragmentation and motion')
                self.write_csv('dispersion_dynamics.csv', measure_dynamics(self))
            if export_figures:
                self.progress(99, 'Exporting region maps and movies')
                self.export()
            self.write_manifest('complete' if preview_key is None else 'preview_complete')
            self.progress(100,f'Complete: {len(self.colony_rows)} region observations')
            return self
        except CancelledAnalysis:
            self.write_manifest('cancelled');raise
        except Exception:
            self.write_manifest('failed');raise
        finally:
            self.progress=lambda *_:None;self.cancelled=lambda:False

    def initialize_automatic_frames(self, gray):
        from .region_rules import raw_material, raw_dark_z

        for position in self.positions:
            for capture in self.frames:
                self.check()
                image = gray(position, capture)
                masks = self.source_masks(position, capture)
                outer = np.zeros(image.shape, bool)
                inner = np.zeros(image.shape, bool)
                for parent in self.inputs[position, capture]['parents']:
                    boundary = masks[f'colony_{parent}']
                    if (outer & boundary).any():
                        raise ValueError('Outer biofilm masks overlap')
                    outer |= boundary
                    inner |= masks[f'inner_{parent}']
                material = raw_material(crop(image, BORDER)[None], crop(outer, BORDER)[None])[0]
                material = pad_back(material, BORDER, image.shape)
                fraction, hybrid, dark_z = notebook_support(image, outer, inner)
                dark_z = pad_back(raw_dark_z(crop(image, BORDER)), BORDER, image.shape)
                measured = {'raw_z': material, 'inside_roi': inner, 'hybrid': hybrid,
                            'intensity_50': inner_dark_mask(self.raw(position, capture), outer)}[self.settings.dark_measurement]
                darkness, _ = darkness_map(image, outer, self.settings.width)
                ids = np.zeros(image.shape, np.int32)
                np.savez_compressed(self.mask_path(position, capture), raw_gray=image,
                    outer_mask=outer, inner_mask=inner, region_ids=ids, frame_labels=ids,
                    pink_mask=np.zeros_like(outer), measured_mask=measured,
                    hybrid_mask=hybrid, raw_dark_mask=material, inner_fraction=fraction,
                    dark_z=dark_z, cell_region_ids=ids, strict_inner_roi_ids=ids,
                    hybrid_inner_roi_ids=ids, darkness=darkness, markers=ids,
                    records_json=np.array('[]'), vectors_json=np.array('[]'))
                self.frame_rows.append(dict(position=position, capture=capture,
                    elapsed_time=self.elapsed(self.frames[0], capture),
                    time_unit='h' if self.dispersion_timing.get('hours') is not None else self.settings.time_unit,
                    region_count=0, outer_area_px=int(outer.sum()),
                    dark_measurement=self.settings.dark_measurement,
                    strict_inner_area_px=int(inner.sum()), hybrid_supported_area_px=int(hybrid.sum())))
        self.pink_provenance = dict(method='central_shared_edges',
            interpretation='Central residual compartment; not a motility classification',
            initialization_padding='reflect_15px_from_analysis_crop',
            peak_parameters=dict(width=32, smooth_sigma=2., prominence=.025,
                min_peak_distance=18, min_region_area=120, min_support_fraction=.5),
            initial_motion_sampling=dict(max_corners=1200, min_distance_px=6),
            central_interface_px=12, narrow_pink_radius_px=5,
            edge_collar_radius_fraction=.3, shared_edge_sigma_px=1.,
            shared_edge_band_px=3, maximum_smoothing_area_change=.03,
            breakup=dict(domain='current_colony_territory', threshold=.8,
                background_sigma_px=7., foreground_sigma_px=.6, sd_floor=5.,
                background_statistics='unsmoothed_raw', minimum_object_area_px=4,
                connectivity=8, formula='1-largest_connected_dark_area/total_dark_area'))
        if self.settings.dark_measurement == 'intensity_50':
            self.pink_provenance['breakup'] = dict(domain='current_colony_territory',
                detector='normalized_intensity', threshold_percent=50, threshold_uint8=128,
                normalization='per_frame_analysis_crop_min_max_to_uint8', gaussian_kernel_px=[5, 5],
                gaussian_sigma=0, morphology_kernel_px=1, minimum_object_area_px=1,
                connectivity=8, formula='1-largest_connected_dark_area/total_dark_area')


    def automatic_territories(self, position, frames, valid):
        from .region_rules import accepted_partition, image_motion as checked_motion

        ids = np.zeros((len(frames), *frames[0]['raw_gray'].shape), np.int32)
        events, blocks, block = [], [], []
        for index, (capture, focused) in enumerate(zip(self.frames, valid)):
            if not focused or not frames[index]['outer_mask'].any():
                if block:
                    blocks.append(block); block = []
                continue
            if block and capture != self.frames[block[-1]] + 1:
                blocks.append(block); block = []
            block.append(index)
        if block:
            blocks.append(block)
        maximum = 1
        applied = set()
        for block in blocks:
            result = accepted_partition(
                np.array([crop(frames[i]['raw_gray'], BORDER) for i in block]),
                np.array([crop(frames[i]['outer_mask'], BORDER) for i in block]),
                geometry_raw=np.array([frames[i]['raw_gray'] for i in block]),
                check=self.check, progress=lambda percent, message: self.progress(15 + percent * 78 // 100, message))
            labels = result['labels'].copy()
            offset = maximum - 1
            labels[labels > 1] += offset
            maximum = max(maximum, int(labels.max()))
            for local, index in enumerate(block):
                capture = self.frames[index]
                ids[index] = pad_back(labels[local], BORDER, ids.shape[1:])
                if local and self.settings.motion:
                    source = block[local - 1]
                    vectors, _ = checked_motion(crop(frames[source]['raw_gray'], BORDER),
                        crop(frames[index]['raw_gray'], BORDER), crop(frames[source]['outer_mask'], BORDER))
                    for vector in vectors:
                        vector['start'] = (np.asarray(vector['start']) + BORDER).tolist()
                        vector['end'] = (np.asarray(vector['end']) + BORDER).tolist()
                        vector.update(position=position, capture=capture,
                            source_capture=self.frames[source], elapsed_interval=self.elapsed(self.frames[source], capture),
                            time_unit='h' if self.dispersion_timing.get('hours') is not None else self.settings.time_unit)
                    frames[index]['vectors_json'] = np.array(json.dumps(vectors))
            events.extend(dict(event, capture=self.frames[block[event['capture']]],
                               display_id=event['display_id'] + offset) for event in result['events'])
            for alias in result.get('aliases', []):
                events.append(dict(capture=self.frames[block[alias['first_frame']]],
                    display_id=alias['parent'] + offset, alias=alias['alias'] + offset,
                    event='fragment_reconciliation', good_intervals=alias['good_intervals']))
        for index, capture in enumerate(self.frames):
            for signature, ident in [(signature, ident) for (p, t, signature), ident in self.identity_overrides.items()
                                     if (p, t) == (position, capture)]:
                for local in np.unique(ids[index][ids[index] > 1]):
                    mask = ids[index] == local
                    if hashlib.sha256(mask.tobytes()).hexdigest() != signature:
                        continue
                    if ident != local and (ids[index] == ident).any():
                        raise ValueError(f'Region ID {ident} is already assigned at P={position}, T={capture}')
                    ids[index][mask] = ident
                    applied.add((position, capture, signature))
                    events.append(dict(capture=capture, display_id=ident, event='manual_region_id'))
                    break
        missing = {key for key in self.identity_overrides if key[0] == position} - applied
        if missing:
            raise ValueError('A region ID edit no longer matches the final masks. Clear ID edits and rerun.')
        return ids, events

    def refine_territories(self, *, automatic=False):
        from .region_rules import refine_regions

        old = {(row['position'], row['capture'], row['display_id']): row for row in self.colony_rows}
        rows, events, motion_rows = [], [], []
        summaries = {(row['position'], row['capture']): row for row in self.frame_rows}
        for position in self.positions:
            frames = [self.load_frame(position, capture) for capture in self.frames]
            valid = [not any(a <= capture < b for a, b in self.focus_intervals) for capture in self.frames]
            outer = np.array([frame['outer_mask'] for frame in frames])
            evidence_outer = outer.copy()
            evidence_outer[~np.asarray(valid)] = False
            if automatic:
                ids, decisions = self.automatic_territories(position, frames, valid)
                pink = ids == 1
            else:
                ids, pink, decisions = refine_regions(
                    np.array([frame['raw_gray'] for frame in frames]),
                    np.array([frame['region_ids'] for frame in frames]),
                    np.array([frame['pink_mask'] for frame in frames]),
                    evidence_outer,
                    np.array([frame['inner_mask'] if frame['inner_mask'].any() else frame['raw_dark_mask'] for frame in frames]), self.frames,
                    check=self.check, progress=lambda _, message: self.progress(84, message),
                )
            events.extend(dict(position=position, **decision) for decision in decisions)
            associations = {(event['capture'], event['display_id']) for event in decisions if event['event'] == 'fragment_reconciliation'}
            states = {(event['capture'], event['display_id']): event['event'] for event in decisions if event['event'] in ('partially_observed', 'observed_continuity', 'persistent_structure', 'observed', 'area_proposal_rejected', 'observed_support_change', 'manual_region_id')}
            baselines = {}
            indices = {capture: index for index, capture in enumerate(self.frames)}
            for index, (capture, frame) in enumerate(zip(self.frames, frames)):
                self.check()
                if not automatic and not valid[index]:
                    ids[index] = frame['region_ids']
                    pink[index] = frame['pink_mask']
                labels = ids[index]
                records = []
                boundary = frame['outer_mask'] & ~ndi.binary_erosion(frame['outer_mask'])
                for prop in measure.regionprops(labels, intensity_image=frame['raw_gray']):
                    ident = int(prop.label)
                    mask = labels == ident
                    prior = old.get((position, capture, ident))
                    if prior is None:
                        neighbors = [row for (p, t, _), row in old.items() if p == position and t == capture]
                        history = [row for (p, t, i), row in old.items() if p == position and i == ident and t <= capture]
                        prior = max(history, key=lambda row: row['capture']) if history else (
                            max(neighbors, key=lambda row: np.count_nonzero(mask & (frame['region_ids'] == row['display_id']))) if neighbors else {})
                        prior = {key: value for key, value in prior.items() if not key.startswith('motion_') and key not in ('match_iou', 'match_advantage', 'continuity_source_capture', 'continuity_support_points')}
                        prior['identity_status'] = 'automatic_pink_compartment' if np.any(mask & pink[index]) else 'persistent_structure'
                    record = dict(prior)
                    measured = mask & frame['measured_mask']
                    area = int(measured.sum())
                    baselines.setdefault(ident, area)
                    status = states.get((capture, ident), record.get('identity_status', 'observed_continuity'))
                    record.update(position=position, channel=self.settings.cell_channel, capture=capture,
                        parent_roi_id=record.get('parent_roi_id', 1), display_id=ident, frame_label=ident,
                        elapsed_time=self.elapsed(self.sampled[0], capture),
                        time_unit='h' if self.dispersion_timing.get('hours') is not None else self.settings.time_unit,
                        identity_status=status, association_event=(capture, ident) in associations,
                        occluded_area_px=int(((frame['region_ids'] == ident) & (labels != ident) & (labels > 0)).sum()),
                        region_type='central_pink' if np.any(mask & pink[index]) else 'microcolony',
                        territory_area_px=int(prop.area), dark_area_px=area,
                        dark_measurement=self.settings.dark_measurement,
                        strict_inner_area_px=int((mask & frame['inner_mask']).sum()),
                        hybrid_supported_area_px=int((mask & frame['hybrid_mask']).sum()),
                        inferred_gap_area_px=int((mask & frame['hybrid_mask'] & ~frame['inner_mask']).sum()),
                        normalized_dark_area_px=int((mask & frame['raw_dark_mask']).sum()),
                        dark_area_fraction=area / max(1, int(prop.area)),
                        dark_area_ratio_to_first=area / baselines[ident] if baselines[ident] else None,
                        dark_object_count_min1px=int(ndi.label(measured, np.ones((3, 3)))[1]),
                        dark_object_count_min4px=int(sum(prop.area >= 4 for prop in measure.regionprops(ndi.label(measured, np.ones((3, 3)))[0]))),
                        centroid_x_px=float(prop.centroid[1]), centroid_y_px=float(prop.centroid[0]),
                        mean_gray_0_255=float(prop.intensity_mean), mean_local_darkness=float(np.mean(frame['darkness'][mask])),
                        perimeter_px=float(prop.perimeter), eccentricity=float(prop.eccentricity), solidity=float(prop.solidity),
                        touches_roi_boundary=bool(np.any(mask & boundary)))
                    if self.settings.pixel_um > 0:
                        record.update(territory_area_um2=float(prop.area) * self.settings.pixel_um ** 2,
                                      dark_area_um2=area * self.settings.pixel_um ** 2)
                    records.append(record); rows.append(record)
                frame.update(underlying_region_ids=frame['region_ids'].copy(),
                    region_ids=labels, frame_labels=labels, pink_mask=pink[index],
                    cell_region_ids=np.where(frame['measured_mask'], labels, 0).astype(np.int32),
                    strict_inner_roi_ids=np.where(frame['inner_mask'], labels, 0).astype(np.int32),
                    hybrid_inner_roi_ids=np.where(frame['hybrid_mask'], labels, 0).astype(np.int32),
                    records_json=np.array(json.dumps(records)))
                vectors = json.loads(str(frame['vectors_json']))
                for vector in vectors:
                    x, y = np.rint(vector['start']).astype(int)
                    source = ids[indices[vector['source_capture']]]
                    vector['display_id'] = int(source[y, x]) if 0 <= y < source.shape[0] and 0 <= x < source.shape[1] else 0
                motion_rows.extend(vectors)
                frame['vectors_json'] = np.array(json.dumps(vectors))
                np.savez_compressed(self.mask_path(position, capture), **frame)
                summary = summaries[position, capture]
                summary.update(region_count=len(records), assigned_territory_px=int((labels > 0).sum()),
                    supported_motion_vectors=len(vectors),
                    dark_area_px=int(frame['measured_mask'].sum()),
                    measured_dark_area_px=int(frame['measured_mask'].sum()),
                    normalized_dark_area_px=int(frame['raw_dark_mask'].sum()),
                    assigned_dark_area_px=int(((labels > 0) & frame['measured_mask']).sum()),
                    unassigned_dark_area_px=int(((labels == 0) & frame['measured_mask']).sum()))
        self.colony_rows = rows
        self.track_rows = rows
        self.motion_rows = motion_rows
        self.write_csv('region_events.csv', events)
        self.pink_provenance.update(territory_rules='central_shared_edges_v2' if automatic else 'temporal_structure_and_fragment_evidence_v1',
            interpretation='Central residual material; persistent image-supported structure retains colony ownership',
            retrospective_confirmation=True, rule_events=len(events),
            structure_evidence=dict(scales_px=[5, 8, 12], contrast_floor=5, stability_ratio=2,
                polarity_agreement=.8, minimum_observations=5, area_quantile_ratio=1.8,
                parameter_overlap=.5),
            fragment_evidence=dict(minimum_contact_px=5, minimum_contact_frames=4,
                contact_fraction=.6, parent_support=.5, common_motion_ncc=.65,
                maximum_shared_fit_penalty=.1, maximum_relative_step_px=4,
                minimum_motion_intervals=3, motion_interval_fraction=.75))

    def classify_region_motion(self):
        """Classify checked motion while keeping every territory and display ID intact."""
        settings = self.settings
        hours = self.dispersion_timing.get("hours")
        records = {(r["position"], r["capture"], r["display_id"]): r for r in self.colony_rows}
        uncertain_frames = []
        candidate_frames = []
        for position in self.positions:
            previous = None
            history = {}
            for frame_index, capture in enumerate(self.frames):
                self.check()
                self.progress(80 + int(3 * frame_index / len(self.frames)),
                              f"Classifying motion · P={position}, T={capture}")
                frame = self.load_frame(position, capture)
                labels = frame["region_ids"]
                pink = np.zeros(labels.shape, bool)
                observed = [int(i) for i in np.unique(labels[labels > 0])]
                source_ids = np.array([], dtype=int)
                residual = np.array([], dtype=float)
                vectors = []
                background = threshold = interval = None
                scene_reason = "first_selected_frame" if previous is None else "nonconsecutive_frame"
                if not settings.motion:
                    scene_reason = "motion_disabled"
                focused = not any(a <= capture < b for a, b in self.focus_intervals)
                if settings.motion and previous is not None and focused and previous[2] and capture == previous[0] + 1:
                    source_capture, source, _ = previous
                    vectors, affine = image_motion(source["raw_gray"], frame["raw_gray"],
                                                   source["outer_mask"], max_corners=2400)
                    scene_reason = "bulk_motion_unresolved"
                    if affine is not None:
                        start = np.array([v["start"] for v in vectors])
                        end = np.array([v["end"] for v in vectors])
                        xy = np.rint(start).astype(int)
                        source_ids = source["region_ids"][xy[:, 1], xy[:, 0]]
                        predicted = start @ affine[:, :2].T + affine[:, 2]
                        residual = np.linalg.norm(end - predicted, axis=1)
                        background = float(np.median(residual))
                        tracking_error = float(np.median([v["fb_error_px"] for v in vectors]))
                        threshold = max(0.5, 2 * tracking_error, settings.pink_speed_ratio * background)
                        interval = hours[capture] - hours[source_capture] if hours is not None else (capture - source_capture) * settings.interval
                        if interval <= 0:
                            raise ValueError("Motion classification needs increasing timestamps")
                        scene_reason = "measured"
                if not focused or previous is not None and not previous[2]:
                    scene_reason = "focus_excluded"
                frame_uncertain = False
                for ident in observed:
                    selected = source_ids == ident
                    support = int(selected.sum())
                    median = float(np.median(residual[selected])) if support else None
                    classification = "insufficient_evidence"
                    reason = scene_reason
                    high = False
                    flags = history.setdefault(ident, [])
                    if scene_reason == "measured" and support >= settings.pink_min_points:
                        high = median > threshold
                        flags.append((capture, high))
                        flags[:] = [(t, flag) for t, flag in flags if t >= capture - 2]
                        hits = sum(flag for _, flag in flags)
                        if high and hits >= settings.pink_persistence:
                            classification, reason = "highly_motile_candidate", "repeated_motion_above_threshold"
                            pink |= labels == ident
                        elif high:
                            classification, reason = "pending", "awaiting_repeat_observation"
                        else:
                            classification, reason = "typical", "motion_below_threshold"
                    else:
                        flags.clear()
                        hits = 0
                        frame_uncertain = True
                        if scene_reason == "measured":
                            reason = "too_few_checked_points"
                    evidence = dict(
                        position=position, capture=capture, region=ident, motion_class=classification,
                        motion_reason=reason, motion_points=support,
                        median_residual_px=median, background_residual_px=background,
                        motion_threshold_px=threshold,
                        motion_speed=median / interval if median is not None and interval is not None else None,
                        motion_time_unit="h" if hours is not None else settings.time_unit,
                        motion_high_observations=hits,
                    )
                    self.motion_class_rows.append(evidence)
                    record = records.get((position, capture, ident))
                    if record is not None:
                        record.update({key: value for key, value in evidence.items()
                                       if key not in ("position", "capture", "region")})
                for ident in list(history):
                    if ident not in observed:
                        del history[ident]
                if pink.any():
                    candidate_frames.append([position, capture])
                if frame_uncertain:
                    uncertain_frames.append([position, capture])
                # Only the overlay class changes. Territory geometry and identity arrays are preserved.
                frame["pink_mask"] = pink
                frame["records_json"] = np.array(json.dumps([
                    records[position, capture, ident] for ident in observed
                    if (position, capture, ident) in records
                ]))
                np.savez_compressed(self.mask_path(position, capture), **frame)
                previous = capture, frame, focused
        self.pink_provenance = dict(
            method="motion_classification", status="complete", candidate_frames=candidate_frames,
            uncertain_frames=uncertain_frames, minimum_checked_points=settings.pink_min_points,
            speed_ratio=settings.pink_speed_ratio, minimum_residual_px=0.5,
            uncertainty_floor="twice median forward/backward error",
            persistence=f"{settings.pink_persistence} high observations in the last 3 comparisons, including the current one",
            identity_policy="motion class preserves region IDs",
            interpretation="Image-supported motion candidate classification",
        )

    def sensitivity(self,gray):
        rows=[]
        t=self.sampled[len(self.sampled)//2]
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        for p in self.positions:
            fig=Figure(figsize=(12,10),layout='constrained');axes=fig.subplots(3,3)
            masks=self.source_masks(p,t);domain=np.zeros(gray(p,t).shape,bool)
            for parent in self.inputs[p,t]['parents']:
                domain |= masks[f'{self.settings.mask_kind}_{parent}']
            for width in [16,32,48]:
                for prominence in [.015,.025,.045]:
                    self.check();params=asdict(self.settings)|dict(width=width,prominence=prominence)
                    lab=split_regions(gray(p,t),domain,params,check=self.check)['labels']
                    rows.append(dict(position=p,capture=t,width=width,prominence=prominence,region_count=int(lab.max()),roi_coverage=float((lab>0).sum()/max(1,domain.sum()))))
                    ax=axes[[16,32,48].index(width),[.015,.025,.045].index(prominence)]
                    ax.imshow(segmentation.mark_boundaries(gray(p,t).astype(float)/255,lab,color=(.1,1,.9),mode='inner',background_label=0))
                    ax.set_title(f'width {width}px | prominence {prominence:g} | n={int(lab.max())}')
                    ax.set_axis_off()
            FigureCanvasAgg(fig).print_png(str(self.output/f'parameter_sensitivity_position_{p:03d}.png'))
        self.write_csv('parameter_sensitivity.csv',rows)

    def export(self):
        import tifffile
        from PIL import Image
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        for p in self.positions:
            self.check();folder=self.output/f'position_{p:03d}';folder.mkdir()
            stacks={name:[] for name in ('region_ids','frame_labels','inner_mask','outer_mask','pink_mask','measured_mask','hybrid_mask','raw_dark_mask','cell_region_ids','strict_inner_roi_ids','hybrid_inner_roi_ids')}
            movies=[]
            for t in self.sampled:
                self.check();z=self.load_frame(p,t)
                for name in stacks:stacks[name].append(z[name])
                records=json.loads(str(z['records_json']))
                rgb=render_overlay(z,{'outer','regions','boundaries','pink'},records=records)
                image=Image.fromarray(rgb[BORDER:-BORDER,BORDER:-BORDER]);image.thumbnail((1200,1200))
                image.save(folder/f'regions_t{t:04d}.png');movies.append(image)
            for name,values in stacks.items():
                stack=np.stack(values)
                if stack.dtype==bool:stack=stack.astype(np.uint8)
                tifffile.imwrite(folder/f'{name}.tif',stack,photometric='minisblack',metadata={'axes':'TYX'},compression='zlib')
            if movies:movies[0].save(folder/'regions_across_time.gif',save_all=True,append_images=movies[1:],duration=350,loop=0)
        fig=Figure(figsize=(12,4),layout='constrained')
        for ax,metric,title in zip(fig.subplots(1,3),('region_count','dark_area_px','unassigned_dark_area_px'),('Candidate regions','Measured dark area (px²)','Unassigned dark area (px²)')):
            for p in self.positions:
                rows=[r for r in self.frame_rows if r['position']==p]
                ax.plot([r['elapsed_time'] for r in rows],[r[metric] for r in rows],'.-',label=f'P{p}')
            ax.set(title=title,xlabel=f"Elapsed ({self.frame_rows[0]['time_unit']})");ax.legend()
        FigureCanvasAgg(fig).print_png(str(self.output/'properties_across_time.png'))
        # Keep the notebook's valley/boundary cross-section as a review diagnostic.
        p=self.positions[0];t=self.sampled[len(self.sampled)//2];z=self.load_frame(p,t)
        ys,xs=np.where(z['outer_mask']);y=int(np.median(ys)) if len(ys) else z['outer_mask'].shape[0]//2
        fig=Figure(figsize=(10,4),layout='constrained');left,right=fig.subplots(1,2)
        left.imshow(render_overlay(z,{'outer','boundaries'}));left.axhline(y,color='yellow')
        profile=z['darkness'][y];right.plot(profile)
        lab=z['frame_labels'][y];cuts=np.where((lab[1:]!=lab[:-1])&(lab[1:]>0)&(lab[:-1]>0))[0]+1
        for cut in cuts:right.axvline(cut,color='teal',alpha=.5)
        right.set(title='Local darkness and region boundaries',xlabel='Source x (px)',ylabel='Darkness (0–1)')
        FigureCanvasAgg(fig).print_png(str(self.output/'valley_cross_section.png'))


def render_overlay(frame,layers,*,records=(),selected=None,motion=False,black=False):
    """Use only the saved selector inputs and the one shared region-ID stack."""
    image=frame['raw_gray'];ids=frame['region_ids'];inner=frame['inner_mask'];outer=frame['outer_mask']
    pink=np.asarray(frame.get('pink_mask',np.zeros(image.shape,bool)),bool)
    rgb=np.zeros((*image.shape,3),np.float32) if black else np.repeat(image[...,None],3,axis=2).astype(np.float32)
    if 'darkness' in layers:
        from matplotlib import colormaps
        heat=colormaps['magma'](np.nan_to_num(frame['darkness']))[...,:3]*255
        domain=np.isfinite(frame['darkness']);rgb[domain]=.45*rgb[domain]+.55*heat[domain]
    if 'regions' in layers or 'cell_regions' in layers:
        from matplotlib.colors import hsv_to_rgb
        for ident in np.unique(ids[ids>0]):
            mask=(ids==ident)&~pink
            if 'regions' not in layers:mask &= frame.get('measured_mask',inner)
            colour=hsv_to_rgb([(ident*.61803398875)%1,.72,.95])*255
            rgb[mask]=.55*rgb[mask]+.45*colour
    if 'inner' in layers:rgb[inner]=.35*rgb[inner]+.65*np.array([0,210,230])
    if 'measured' in layers:
        m=frame.get('measured_mask',inner);rgb[m]=.35*rgb[m]+.65*np.array([0,210,230])
    if 'hybrid' in layers:
        m=frame.get('hybrid_mask',inner)&~inner;rgb[m]=.3*rgb[m]+.7*np.array([130,255,80])
    if 'raw_dark' in layers:
        m=frame.get('raw_dark_mask',np.zeros(image.shape,bool));rgb[m]=.3*rgb[m]+.7*np.array([255,150,80])
    if 'unassigned' in layers:
        m=frame.get('measured_mask',inner)&(ids==0);rgb[m]=.25*rgb[m]+.75*np.array([255,165,20])
    if 'boundaries' in layers:rgb[segmentation.find_boundaries(ids,mode='inner') & ~pink]=(25,255,230)
    if 'outer' in layers:
        edge=outer & ~ndi.binary_erosion(outer);rgb[edge]=(255,40,190)
    if 'pink' in layers:
        rgb[pink]=.35*rgb[pink]+.65*np.array([255,70,180])
        rgb[pink & ~ndi.binary_erosion(pink)]=(255,100,200)
    if selected is not None and 'selection' in layers:
        rgb[segmentation.find_boundaries(ids==selected,mode='inner')]=(255,255,0)
    output=np.clip(rgb,0,255).astype(np.uint8)
    if 'labels' in layers:
        for r in records:
            cv2.putText(output,str(r['display_id']),(int(r['centroid_x_px']),int(r['centroid_y_px'])),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)
    if motion:
        for v in json.loads(str(frame.get('vectors_json',np.array('[]')))):
            a=tuple(int(round(x)) for x in v['start']);b=tuple(int(round(x)) for x in v['end'])
            if a!=b:cv2.arrowedLine(output,a,b,(255,235,30),1,cv2.LINE_AA,tipLength=.25)
    return output


def measure_dynamics(analysis):
    """One row per observed microcolony and frame using the selected dark mask."""
    timing = analysis.dispersion_timing
    hours = timing.get("hours")
    frames = {}
    for record in analysis.colony_rows:
        key = record["position"], record["capture"]
        if record["region_type"] != "central_pink":
            frames.setdefault(key, {})[record["display_id"]] = record

    rows = []
    for position in analysis.positions:
        previous = None
        for frame_index, capture in enumerate(analysis.frames):
            analysis.check()
            analysis.progress(94 + int(5 * frame_index / len(analysis.frames)),
                              f"Measuring breakup and motion · P={position}, T={capture}")
            masks = analysis.load_frame(position, capture)
            records = frames.get((position, capture), {})
            time = capture if hours is None else hours[capture]
            focused = not any(
                start <= capture < end for start, end in analysis.focus_intervals
            )
            for region, record in records.items():
                analysis.check()
                territory = masks["region_ids"] == region
                dark = territory & masks["measured_mask"]
                labels, _ = ndi.label(dark, structure=np.ones((3, 3)))
                sizes = np.bincount(labels[dark], minlength=1)[1:]
                area = int(dark.sum())
                connected = int(sizes.max(initial=0))
                row = dict(
                    position=position, capture=capture, region=region, time=time,
                    time_unit="frame" if hours is None else "h", focused=focused,
                    dark_area_px=area if focused else None,
                    connected_area_px=connected if focused else None,
                    disconnected_area_px=area - connected if focused else None,
                    breakup_score=1.0 - connected / area if focused and area else None,
                    dark_patch_count=int(np.count_nonzero(sizes >= (1 if getattr(getattr(analysis, "settings", None), "dark_measurement", None) == "intensity_50" else 3))) if focused else None,
                    dark_area_loss_per_time=None, transported_loss_px=None,
                    transported_gain_px=None, outward_crossing_candidates=None,
                    receiving_region_crossing_candidates=None, crossing_vectors=None, motion_supported=False,
                    identity_status=record["identity_status"], match_iou=record.get("match_iou"),
                    association_event=record.get('association_event', False),
                    occluded_area_px=record.get('occluded_area_px', 0),
                    match_advantage=record.get("match_advantage"),
                    continuity_support_points=record.get("continuity_support_points"),
                    **{name: record.get(name) for name in (
                        "motion_class", "motion_reason", "motion_points", "median_residual_px",
                        "background_residual_px", "motion_threshold_px", "motion_speed",
                        "motion_time_unit", "motion_high_observations")},
                )
                if (focused and not record.get('association_event') and not record.get('occluded_area_px')
                        and previous is not None and previous[0] == capture - 1
                        and previous[3] and region in previous[4]
                        and not previous[4][region].get('occluded_area_px')
                        and record["identity_status"] in
                        ("unique_overlap", "reciprocal_overlap", "reviewed_notebook_partition", "manual_region_id", "motion_supported_continuity", "observed_continuity", "persistent_structure", "observed", "area_proposal_rejected", "observed_support_change")):
                    source_masks = previous[2]
                    source_territory = source_masks["region_ids"] == region
                    source_dark = source_territory & source_masks["measured_mask"]
                    row["dark_area_loss_per_time"] = (int(source_dark.sum()) - area) / (time - previous[1])
                    row.update(_transport(source_masks, masks, source_territory, source_dark))
                rows.append(row)
            previous = capture, time, masks, focused, records
    return rows


def _transport(source, target, territory, dark):
    """Use supported texture motion to transport the source footprint."""
    y, x = np.nonzero(territory)
    padding = 24
    x0, y0 = max(0, int(x.min()) - padding), max(0, int(y.min()) - padding)
    height, width = territory.shape
    section = np.s_[y0:min(height, int(y.max()) + padding + 1),
                    x0:min(width, int(x.max()) + padding + 1)]
    vectors, affine = image_motion(
        source["raw_gray"][section], target["raw_gray"][section], dark[section]
    )
    if affine is None:
        return {}
    offset = np.array([x0, y0])
    affine[:, 2] += offset - affine[:, :2] @ offset

    def warp(mask):
        return cv2.warpAffine(mask.astype(np.uint8), affine, (width, height),
                              flags=cv2.INTER_NEAREST).astype(bool)

    footprint = warp(territory)
    if footprint.sum() < 0.7 * territory.sum() * abs(np.linalg.det(affine[:, :2])):
        return {}
    transported = warp(dark)
    observed = footprint & target["measured_mask"]
    exterior = ndi.distance_transform_edt(~footprint)
    near_boundary = ndi.distance_transform_edt(territory) <= 8
    crossings = []
    receiving = 0
    pink = target["pink_mask"]
    for vector in vectors:
        start = np.asarray(vector["start"]) + offset
        end = np.asarray(vector["end"]) + offset
        sx, sy = np.rint(start).astype(int)
        tx, ty = np.rint(end).astype(int)
        predicted = affine[:, :2] @ start + affine[:, 2]
        px, py = np.rint(predicted).astype(int)
        if not (0 <= px < width and 0 <= py < height):
            continue
        if (near_boundary[sy, sx] and exterior[ty, tx] >= 1
                and exterior[ty, tx] > exterior[py, px]
                and np.linalg.norm(end - predicted) >= 1
                and target["measured_mask"][ty, tx]):
            crossings.append([start.tolist(), end.tolist()])
            receiving += int(pink[ty, tx])
    return dict(
        transported_loss_px=int((transported & ~observed).sum()),
        transported_gain_px=int((observed & ~transported).sum()),
        outward_crossing_candidates=len(crossings),
        crossing_vectors=json.dumps(crossings),
        receiving_region_crossing_candidates=receiving if pink.any() else None,
        motion_supported=True,
    )
