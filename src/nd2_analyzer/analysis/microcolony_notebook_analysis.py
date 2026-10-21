"""Darkness/watershed and reviewed pink-compartment notebooks driven by the verifier's exact outer/inner masks.

Ports darkness peaks, enclosed-ROI growth, connected pink, shared borders,
strict/hybrid support and normalized dark-object calculations. Accepted notebook
partitions are reused only after exact paired-image verification. No Omnipose,
cell-instance mask, EPS,
minimum-cell-count rule, physical calibration guess, or exported input dependency.
Motion is independently checked on raw-image texture; vectors are not centroid
shifts. Original image coordinates are retained in storage. The shared 15px image
border is excluded from inputs and from the viewer by an explicit transform.
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

BACKEND = 'selector_darkness_watershed_v1'
BORDER = 15


@dataclass(frozen=True)
class NotebookSettings:
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
    motion: bool = True
    interval: float = 1.0
    time_unit: str = 'capture'
    dark_measurement: str = 'inside_roi'
    pink_source: str = ""  # optional reviewed region_ids.tif; blank = validated notebook discovery
    pixel_um: float = 0.0  # zero = unavailable; never borrow another dataset's scale

    def validate(self):
        if self.dark_measurement not in ('inside_roi','hybrid','raw_z'):
            raise ValueError('Unknown notebook dark-area measurement')
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
    a=ndi.gaussian_filter(g.astype(float),.6);mu=ndi.gaussian_filter(a,7)
    sd=np.sqrt(np.maximum(0,ndi.gaussian_filter(a*a,7)-mu*mu));z=(mu-a)/np.maximum(sd,5)
    def expand(a):
        full=np.zeros(image.shape,a.dtype);full[crop]=a;return full
    return expand(fraction),expand(hybrid),expand(z)


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


def split_regions(image, mask, params):
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
        sizes = np.bincount(labels.ravel())
        tiny = np.flatnonzero((sizes<params['min_region_area']) & (np.arange(len(sizes))>0))
        if not len(tiny):
            break
        markers[np.isin(markers,tiny)]=0
        labels=flood(markers)
    labels,_,_ = segmentation.relabel_sequential(labels)
    return dict(darkness=d,smooth=np.where(mask,smooth,np.nan),labels=labels.astype(np.int32),markers=markers,support=support)


def match_ids(previous, labels, next_id, allow_match, match_iou=.30):
    """Exact conservative overlap display IDs; ambiguous matches get new IDs."""
    curr_ids=np.unique(labels);curr_ids=curr_ids[curr_ids>0]
    prev_ids=np.unique(previous);prev_ids=prev_ids[prev_ids>0]
    pairs=[]
    if allow_match and len(prev_ids) and len(curr_ids):
        together=(previous>0)&(labels>0)
        values,counts=np.unique(np.column_stack((previous[together],labels[together])),axis=0,return_counts=True)
        old_sizes=dict(zip(*np.unique(previous[previous>0],return_counts=True)))
        new_sizes=dict(zip(*np.unique(labels[labels>0],return_counts=True)))
        for (old,new),inter in zip(values,counts):
            iou=float(inter/(old_sizes[old]+new_sizes[new]-inter))
            if iou>=match_iou:
                pairs.append((int(old),int(new),iou))
    output=np.zeros(labels.shape,np.int32);info={}
    for new in curr_ids:
        options=[p for p in pairs if p[1]==new]
        valid=len(options)==1 and sum(p[0]==options[0][0] for p in pairs)==1
        if valid:
            ident,_,iou=options[0];status='unique_overlap'
        else:
            ident=next_id;next_id+=1;iou=None
            status='ambiguous_overlap' if options else 'new_or_unmatched'
        output[labels==new]=ident
        info[int(new)]=dict(display_id=int(ident),match_iou=iou,identity_status=status)
    return output,info,next_id


def image_motion(previous, current, domain):
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
    points=cv2.goodFeaturesToTrack(a,maxCorners=300,qualityLevel=.01,minDistance=10,mask=domain.astype(np.uint8)*255,blockSize=5)
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


class NotebookAnalysis:
    backend=BACKEND

    def __init__(self,image_data,settings,positions,times,output_root,*,progress=None,cancelled=None,edits=None,selector_params=None):
        settings.validate();self.data=image_data;self.settings=settings
        self.positions=sorted(set(int(p) for p in positions));self.sampled=sampled_frames(times,settings.sample_step)
        if not self.positions or not self.sampled:
            raise ValueError('Select positions and frames first')
        self.output=Path(output_root)/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
        self.frames=list(self.sampled);self.kept=list(self.positions);self.preview_key=None
        self.progress=progress or (lambda *_:None);self.cancelled=cancelled or (lambda:False)
        self.selector_params=selector_params or {};self.edits=edits or {};self.pink_masks={};self.reviewed_regions={};self.pink_provenance={'method':'unavailable'}
        self.colony_rows=[];self.frame_rows=[];self.track_rows=[];self.motion_rows=[];self.dark_sensitivity_rows=[];self.inputs={};self.prepared_colonies={}
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
                     positions=self.positions,frames=self.frames,preview_key=self.preview_key,
                     intensity_conversion=getattr(self,'intensity_conversion',None),pink_region=self.pink_provenance,
                     source_image=str(getattr(self.data,'image_filename','')),
                     coordinate_system='source analysis-image pixels; viewer excludes 15px image border',border_px=BORDER,
                     inputs=[dict(position=p,time=t,**value) for (p,t),value in sorted(self.inputs.items())],
                     dark_area_methods=dict(inside_roi='exact saved Inside ROI',hybrid='32px fraction, 0.65/0.35 validated temporal blend; within 3px; 0.65 new / 0.55 temporal fill',raw_z='Gaussian 0.6/7px; SD floor 5; z>0.8; eight-connected area >=4px'),
                     note='Dark-area method is explicit. Saved Inside ROI is always retained; inferred support and normalized raw-image detection have separate fields. Region territory includes valley pixels. Counts are image regions/dark objects, not individual cells. Motion uses raw-image evidence; unsupported motion stays unavailable.')
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
        candidates=[Path(explicit).expanduser()] if explicit else sorted(
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

    def run(self,preview_key=None):
        self.preview_key=preview_key;s=self.settings;params=asdict(s)
        required=[(p,t) for p in self.positions for t in self.sampled]
        if set(required)-set(self.inputs):
            raise ValueError('Missing in-window input snapshots; prepare the selected masks first')
        self.write_manifest('running')
        try:
            self.track_outer_inputs()
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
            previous={};previous_gray={};previous_outer={};previous_support={};pink_ids={};next_id=max([int(a.max()) for a in self.reviewed_regions.values()]+[0])+1;self._baseline={}
            for index,(p,t) in enumerate(required):
                self.check();self.progress(int(80*index/len(required)),f'Analyzing selector masks: P={p}, T={t}')
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
                    measured={'inside_roi':inner,'hybrid':hybrid,'raw_z':raw_mask}[s.dark_measurement]
                    measured_union |= measured;hybrid_union |= hybrid;raw_union |= raw_mask
                    fraction_full[outer]=fraction[outer];z_full[outer]=dark_z[outer]
                    domain=outer if s.mask_kind=='colony' else inner
                    if (p,t) in self.reviewed_regions:
                        d,_=darkness_map(image,domain,s.width)
                        res=dict(darkness=d,markers=np.zeros(shape,np.int32))
                        labels=np.where(domain,self.reviewed_regions[p,t],0).astype(np.int32)
                    else:
                        res=split_regions(image,domain,params);labels=res['labels'].copy()
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
                        ids,info,next_id=match_ids(old[1],labels,next_id,allow,s.match_iou)
                        previous[(p,parent)]=(t,ids.copy());previous_outer[(p,parent)]=outer
                        pink_local=None
                        if pink.any():
                            if (p,parent) not in pink_ids:
                                pink_ids[p,parent]=next_id;next_id+=1
                            pink_local=int(labels.max())+1;labels[pink]=pink_local;ids[pink]=pink_ids[p,parent]
                            info[pink_local]=dict(display_id=pink_ids[p,parent],match_iou=None,identity_status='reviewed_pink_compartment')
                    pink_union |= pink & domain
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
                        row=dict(position=p,channel=s.cell_channel,parent_roi_id=parent,capture=t,elapsed_time=(t-self.sampled[0])*s.interval,time_unit=s.time_unit,
                            frame_label=local,dark_measurement=s.dark_measurement,strict_inner_area_px=int((territory&inner).sum()),
                            hybrid_supported_area_px=int((territory&hybrid).sum()),inferred_gap_area_px=int((territory&hybrid&~inner).sum()),
                            normalized_dark_area_px=int((territory&raw_mask).sum()),region_type='central_pink' if np.any(territory & pink) else 'microcolony',**info[local],territory_area_px=int(prop.area),dark_area_px=dark_area,
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
                                      elapsed_interval=(t-prior_t)*s.interval,time_unit=s.time_unit)
                        frame_motion.append(vector);self.motion_rows.append(vector)
                previous_gray[p]=(t,image,inner_union.copy(),regions.copy())
                self.frame_rows.append(dict(position=p,capture=t,elapsed_time=(t-self.sampled[0])*s.interval,time_unit=s.time_unit,
                    region_count=len(records),outer_area_px=int(outer_union.sum()),dark_measurement=s.dark_measurement,dark_area_px=int(measured_union.sum()),strict_inner_area_px=int(inner_union.sum()),
                    measured_dark_area_px=int(measured_union.sum()),hybrid_supported_area_px=int(hybrid_union.sum()),normalized_dark_area_px=int(raw_union.sum()),
                    assigned_territory_px=int((regions>0).sum()),assigned_dark_area_px=int(((regions>0)&measured_union).sum()),
                    unassigned_dark_area_px=int(((regions==0)&measured_union).sum()),supported_motion_vectors=len(frame_motion)))
                np.savez_compressed(self.mask_path(p,t),raw_gray=image,outer_mask=outer_union,inner_mask=inner_union,
                    region_ids=regions,frame_labels=frame_labels,pink_mask=pink_union,measured_mask=measured_union,hybrid_mask=hybrid_union,raw_dark_mask=raw_union,inner_fraction=fraction_full,dark_z=z_full,cell_region_ids=np.where(measured_union,regions,0).astype(np.int32),strict_inner_roi_ids=np.where(inner_union,regions,0).astype(np.int32),hybrid_inner_roi_ids=np.where(hybrid_union,regions,0).astype(np.int32),darkness=darkness,markers=markers,
                    records_json=np.array(json.dumps(records)),vectors_json=np.array(json.dumps(frame_motion)))
            self.track_rows=self.colony_rows
            self.write_csv('region_properties.csv',self.colony_rows)
            self.write_csv('pink_region_properties.csv',[r for r in self.colony_rows if r['region_type']=='central_pink'])
            self.write_csv('frame_summary.csv',self.frame_rows)
            self.write_csv('dark_area_sensitivity.csv',self.dark_sensitivity_rows)
            self.write_csv('image_motion_vectors.csv',[{k:v for k,v in r.items() if k not in ('start','end')}|dict(x_px=r['start'][0],y_px=r['start'][1],target_x_px=r['end'][0],target_y_px=r['end'][1]) for r in self.motion_rows])
            self.progress(83,'Checking parameter sensitivity')
            if not getattr(self,'loaded_saved',False):self.sensitivity(gray)
            self.progress(88,'Exporting region maps and movies')
            self.export()
            self.write_manifest('complete' if preview_key is None else 'preview_complete')
            self.progress(100,f'Complete: {len(self.colony_rows)} region observations from the selected outer/Inside ROI masks')
            return self
        except CancelledAnalysis:
            self.write_manifest('cancelled');raise
        except Exception:
            self.write_manifest('failed');raise
        finally:
            self.progress=lambda *_:None;self.cancelled=lambda:False

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
                    lab=split_regions(gray(p,t),domain,params)['labels']
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
            ax.set(title=title,xlabel=f'Elapsed ({self.settings.time_unit})');ax.legend()
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
