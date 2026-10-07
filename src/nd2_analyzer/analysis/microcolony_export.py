"""Minimal btrack-based microcolony movies and measured motion exports.

Coordinates remain in the source image. Cropping is only a rendering transform.
No predicted detections or displacements across missing frames are measured.
"""
from collections import defaultdict
from pathlib import Path
import json

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from skimage.measure import regionprops

from .valid_region import BORDER_PX, valid_labels
from .colony_regions import interior_colonies


def track_observations(frames):
    """Map exact btrack object references to original frame/segmentation IDs."""
    import btrack
    objects, measurements = [], []
    origin = min(frames)
    for time, labels in sorted(frames.items()):
        for cell in regionprops(labels):
            row = dict(time=int(time), cell_id=int(cell.label),
                       centroid_x=float(cell.centroid[1]), centroid_y=float(cell.centroid[0]))
            measurements.append(row)
            objects.append(btrack.btypes.PyTrackObject.from_dict(dict(
                ID=len(objects), t=int(time-origin), x=row['centroid_x'],
                y=row['centroid_y'], z=0.0)))
    if not objects:
        return []
    if len(frames) == 1:
        return [dict(row, cell_track_id=i+1) for i, row in enumerate(measurements)]
    with btrack.BayesianTracker() as tracker:
        config = btrack.config.load_config(Path(__file__).parent / 'tracking/config/btrack_config.json')
        # Conservative links only: division events are not asserted as verified lineage.
        config.hypothesis_model.hypotheses = ['P_init', 'P_term', 'P_link']
        tracker.configure(config)
        tracker.max_search_radius = 50
        tracker.append(objects)
        by_ref = {obj.ID: row for obj, row in zip(objects, measurements)}
        tracker.track(tracking_updates=['MOTION'])
        tracker.optimize(tm_lim=30000, mip_gap=0.01)
        seen, result = set(), []
        for tr in tracker.tracks:
            for ref in tr.refs:
                if ref < 0:
                    continue
                if ref in seen or ref not in by_ref:
                    raise RuntimeError('Invalid or duplicated cell tracking reference.')
                seen.add(ref)
                result.append(dict(by_ref[ref], cell_track_id=int(tr.ID)))
        next_id = max((r['cell_track_id'] for r in result), default=0)
        for ref in sorted(set(by_ref)-seen):
            next_id += 1
            result.append(dict(by_ref[ref], cell_track_id=next_id))
    return result


def add_velocities(rows, hours_per_frame=None, voxel_size=None):
    groups = defaultdict(list)
    sx, sy = getattr(voxel_size, 'x', None), getattr(voxel_size, 'y', None)
    physical = bool(sx and sy and hours_per_frame and hours_per_frame > 0)
    for row in rows:
        groups[row['cell_track_id']].append(row)
    for group in groups.values():
        previous = None
        for row in sorted(group, key=lambda r: r['time']):
            row.update(vx_px_per_frame=None, vy_px_per_frame=None, speed_px_per_frame=None,
                       vx_um_per_hour=None, vy_um_per_hour=None, speed_um_per_hour=None)
            if previous is not None and row['time']-previous['time'] == 1:
                dx = row['centroid_x']-previous['centroid_x']
                dy = row['centroid_y']-previous['centroid_y']
                row.update(vx_px_per_frame=dx, vy_px_per_frame=dy,
                           speed_px_per_frame=float(np.hypot(dx, dy)))
                if physical:
                    vx, vy = dx*sx/hours_per_frame, dy*sy/hours_per_frame
                    row.update(vx_um_per_hour=vx, vy_um_per_hour=vy,
                               speed_um_per_hour=float(np.hypot(vx, vy)))
            previous = row
    return rows


def prepare_colonies(exporter):
    """Clip copies of colony geometry, keeping the original ROI/cache intact."""
    service = exporter.biofilm_metric_service
    if not isinstance(exporter.colonies, dict):
        raise ValueError('Export requires explicit position/time/channel keys for colony regions.')
    if not exporter.colonies:
        raise ValueError('Select a non-empty time range before exporting.')
    prepared = {}
    geometry_changed = False
    for key, colonies in sorted(exporter.colonies.items()):
        p, t, c = key
        shape = np.asarray(exporter.image_data.get(t, p, c)).shape[:2]
        copied, changed = interior_colonies(colonies, shape)
        geometry_changed |= changed
        for item in copied:
            item.update(position=p, time=t, channel=c)
        prepared[key] = copied
    if geometry_changed or any(c.get('track_id') is None for cs in prepared.values() for c in cs):
        from .tracking.colony_tracking import track_colonies_by_frame
        prepared = track_colonies_by_frame(prepared)
    exporter.colonies = prepared
    service.build_colony_table(prepared, model_name=exporter.model_name, voxel_size=exporter.voxel_size)


def export_motion_and_movies(exporter, output_root, hours_per_frame=None, progress=None):
    """Two aligned GIFs per colony, measured tracks, time paths, and expansion."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    root = Path(output_root) / 'microcolony_motion'
    root.mkdir(parents=True, exist_ok=True)
    service = exporter.biofilm_metric_service
    cache = exporter.segmented_storage.with_model(exporter.model_name)
    _, indices = cache.mmap_arrays_idx[cache.model_name]
    available = {(int(i[0]), int(i[1]), int(i[2]) if len(i)>2 else 0) for i in indices}
    groups = defaultdict(list)
    for key in sorted(exporter.colonies):
        groups[(key[0], key[2])].append(key)
    all_rows = []
    cmap = LinearSegmentedColormap.from_list('time_blue_green', ['#1500ff', '#007dcc', '#00dc70'])
    for (p, colony_channel), keys in groups.items():
        times = sorted({k[1] for k in keys})
        channels = {c for t, pos, c in available if pos == p and t in times}
        cell_channel = colony_channel if colony_channel in channels else next(iter(channels)) if len(channels)==1 else None
        if cell_channel is None:
            raise ValueError(f'Position {p}: cannot determine the cell segmentation channel.')
        if any((t, p, cell_channel) not in available for t in times):
            raise ValueError(f'Position {p}: segment every requested frame before exporting motion.')
        frames = {t: valid_labels(cache[(t, p, cell_channel, exporter.model_name)]) for t in times}
        shape = next(iter(frames.values())).shape
        if any(a.shape != shape for a in frames.values()):
            raise ValueError('Cell masks must have identical dimensions across time.')
        if progress:
            progress(f'Tracking cells at position {p}, channel {cell_channel}…')
        rows = add_velocities(track_observations(frames), hours_per_frame, exporter.voxel_size)
        by_time = defaultdict(list)
        for row in rows:
            row.update(position=p, cell_channel=cell_channel, colony_channel=colony_channel,
                       colony_track_id=None, colony_id=None)
            point = (row['centroid_x'], row['centroid_y'])
            for colony in exporter.colonies[(p, row['time'], colony_channel)]:
                if cv2.pointPolygonTest(service._colony_contour(colony), point, False) >= 0:
                    row.update(colony_track_id=colony['track_id'], colony_id=colony['colony_id'])
                    break
            by_time[row['time']].append(row)
        all_rows.extend(rows)
        colonies_by_track = defaultdict(dict)
        for key in keys:
            for colony in exporter.colonies[key]:
                colonies_by_track[colony['track_id']][key[1]] = colony
        for colony_id, colonies in colonies_by_track.items():
            folder = root / f'position_{p:03d}_channel_{colony_channel}_colony_{colony_id}'
            folder.mkdir(parents=True, exist_ok=True)
            boxes = [cv2.boundingRect(service._colony_contour(c)) for c in colonies.values()]
            x0 = max(BORDER_PX, min(b[0] for b in boxes)-5)
            y0 = max(BORDER_PX, min(b[1] for b in boxes)-5)
            x1 = min(shape[1]-BORDER_PX, max(b[0]+b[2] for b in boxes)+5)
            y1 = min(shape[0]-BORDER_PX, max(b[1]+b[3] for b in boxes)+5)
            crop = (slice(y0,y1), slice(x0,x1))
            # Fixed intensity window across all movie frames.
            samples = []
            for t in times:
                raw = np.asarray(exporter.image_data.get(t,p,colony_channel))
                if raw.shape != shape:
                    raise ValueError('Raw images and masks differ in shape; reset the display crop before export.')
                samples.append(raw[crop][::8,::8].ravel())
            lo, hi = np.percentile(np.concatenate(samples), [1,99])
            with imageio.get_writer(folder/'segmented_cells.gif', mode='I', duration=500, loop=0) as cell_movie, imageio.get_writer(folder/'colony_phase.gif', mode='I', duration=500, loop=0) as phase_movie:
                for t in times:
                    colony = colonies.get(t)
                    region = service._colony_mask(colony,shape) if colony else np.zeros(shape,bool)
                    palette = np.zeros((int(frames[t].max())+1,3),dtype=np.uint8)
                    for row in by_time[t]:
                        if row['colony_track_id'] != colony_id:
                            continue
                        hue = (row['cell_track_id']*0.61803398875)%1
                        color = cv2.cvtColor(np.uint8([[[int(hue*179),230,245]]]),cv2.COLOR_HSV2RGB)[0,0]
                        palette[row['cell_id']] = color
                    rgb = palette[frames[t]]
                    rgb[~region] = 0
                    raw = np.asarray(exporter.image_data.get(t,p,colony_channel), dtype=float)
                    gray = np.clip((raw-lo)/max(hi-lo,1e-12)*255,0,255).astype(np.uint8)
                    phase = np.repeat(gray[...,None],3,axis=2)
                    if colony:
                        cv2.drawContours(phase,[service._colony_contour(colony)],-1,(255,80,80),1)
                    title = f'{t*hours_per_frame:.2f} h' if hours_per_frame else f'Frame {t}'
                    for movie, frame in ((cell_movie,rgb),(phase_movie,phase)):
                        view = frame[crop].copy()
                        cv2.putText(view,title,(4,14),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                        movie.append_data(view)
            segments, colors, vectors = [], [], []
            tracks = defaultdict(list)
            for row in rows:
                tracks[row['cell_track_id']].append(row)
            for observations in tracks.values():
                ordered = sorted(observations,key=lambda r:r['time'])
                for a,b in zip(ordered,ordered[1:]):
                    if b['time']-a['time'] != 1 or a['colony_track_id'] != colony_id or b['colony_track_id'] != colony_id:
                        continue
                    segments.append([(a['centroid_x']-x0,a['centroid_y']-y0),(b['centroid_x']-x0,b['centroid_y']-y0)])
                    colors.append(b['time']*hours_per_frame if hours_per_frame else b['time'])
                    vectors.append(dict(time=b['time'],x=a['centroid_x'],y=a['centroid_y'],
                                        u=b['vx_px_per_frame'],v=b['vy_px_per_frame']))
            fig = Figure(figsize=(7,6)); FigureCanvasAgg(fig)
            ax = fig.subplots()
            start,end = times[0],times[-1]
            if hours_per_frame:
                start,end = start*hours_per_frame,end*hours_per_frame
            if colors:
                start,end = min(colors),max(colors)
            norm = Normalize(start,end if end>start else start+1)
            collection = LineCollection(segments,cmap=cmap,norm=norm,linewidths=.7)
            collection.set_array(np.asarray(colors))
            ax.add_collection(collection)
            ax.set(xlim=(0,x1-x0),ylim=(y1-y0,0),aspect='equal',title='Measured cell motion',xlabel='x (pixels)',ylabel='y (pixels)')
            if not segments:
                ax.text(.5,.5,'No consecutive tracked observations',ha='center',transform=ax.transAxes)
            fig.colorbar(collection,ax=ax,label='Time (h)' if hours_per_frame else 'Frame')
            fig.savefig(folder/'motion_field.png',dpi=200,bbox_inches='tight')
            fig.savefig(folder/'motion_field.svg',bbox_inches='tight')
            pd.DataFrame(vectors,columns=['time','x','y','u','v']).to_csv(folder/'motion_vectors.csv',index=False)
            # A separate velocity summary preserves direction and sample support.
            vector_table = pd.DataFrame(vectors,columns=['time','x','y','u','v'])
            if not vector_table.empty:
                vector_table['grid_x'] = ((vector_table.x-x0)//32).astype(int)
                vector_table['grid_y'] = ((vector_table.y-y0)//32).astype(int)
                grid = vector_table.groupby(['time','grid_y','grid_x']).agg(
                    x=('x','mean'),y=('y','mean'),u=('u','mean'),v=('v','mean'),n=('u','size')).reset_index()
                grid['speed_px_per_frame'] = np.hypot(grid.u,grid.v)
                grid.to_csv(folder/'velocity_grid.csv',index=False)
                fig = Figure(figsize=(7,6)); FigureCanvasAgg(fig); ax = fig.subplots()
                arrows = ax.quiver(grid.x-x0,grid.y-y0,grid.u,grid.v,
                    grid.time*(hours_per_frame or 1),cmap=cmap,norm=norm,
                    angles='xy',scale_units='xy',scale=1,width=.003)
                ax.set(xlim=(0,x1-x0),ylim=(y1-y0,0),aspect='equal',
                    title='Cell velocity field (32-pixel bins)',xlabel='x (pixels)',ylabel='y (pixels)')
                fig.colorbar(arrows,ax=ax,label='Time (h)' if hours_per_frame else 'Frame')
                fig.savefig(folder/'velocity_field.png',dpi=200,bbox_inches='tight')
            (folder/'metadata.json').write_text(json.dumps(dict(crop_xyxy=[x0,y0,x1,y1],border_px=BORDER_PX,
                hours_per_frame=hours_per_frame,cell_channel=cell_channel,coordinate_system='source image pixels',
                tracking='btrack motion links; divisions not inferred',velocity='consecutive measured observations only'),indent=2))
    columns = ['position','time','cell_channel','colony_channel','cell_id','cell_track_id','colony_id','colony_track_id',
               'centroid_x','centroid_y','vx_px_per_frame','vy_px_per_frame','speed_px_per_frame',
               'vx_um_per_hour','vy_um_per_hour','speed_um_per_hour']
    table = pd.DataFrame(all_rows,columns=columns)
    table.to_csv(root/'cell_tracks.csv',index=False)
    # Explicit IDs avoid confusing the existing colony track_id with cell identity.
    cells = service.get_colony_cell_results().to_pandas()
    if not cells.empty:
        mapping = table[['position','time','cell_channel','colony_channel','cell_id','cell_track_id']]
        cells.merge(mapping,on=['position','time','cell_channel','colony_channel','cell_id'],how='left',validate='many_to_one').to_csv(Path(output_root)/'colonies_cells.csv',index=False)
    expansion = service.get_colony_results().to_pandas()
    if not expansion.empty:
        expansion = expansion.sort_values(['position','channel','track_id','time'])
        physical = bool(getattr(exporter.voxel_size,'x',None) and getattr(exporter.voxel_size,'y',None))
        expansion['measured_area'] = expansion['area_um2'] if physical else expansion['area_px']
        expansion['plot_time'] = expansion['time']*(hours_per_frame or 1)
        expansion['effective_radius'] = np.sqrt(expansion['measured_area']/np.pi)
        grouped = expansion.groupby(['position','channel','track_id'])
        dt = grouped['plot_time'].diff()
        consecutive = grouped['time'].diff().eq(1)
        measurable = consecutive & ~expansion['fixed_roi'].fillna(False)
        expansion['area_change_rate'] = (grouped['measured_area'].diff()/dt).where(measurable)
        expansion['effective_radius_change_rate'] = (grouped['effective_radius'].diff()/dt).where(measurable)
        expansion.to_csv(root/'colony_expansion.csv',index=False)
        fig = Figure(figsize=(10,4)); FigureCanvasAgg(fig)
        axes = fig.subplots(1,2)
        for key,data in expansion.groupby(['position','channel','track_id']):
            label = f'P{key[0]} C{key[1]} colony {key[2]}'
            axes[0].plot(data['plot_time'],data['measured_area'],'o-',label=label)
            axes[1].plot(data['plot_time'],data['area_change_rate'],'o-',label=label)
        for ax in axes:
            ax.set_xlabel('Time (h)' if hours_per_frame else 'Frame'); ax.legend(fontsize=7)
        axes[0].set_ylabel('Observed colony area ('+('µm²' if physical else 'px²')+')')
        axes[1].set_ylabel('Area change ('+('µm²' if physical else 'px²')+'/'+('h' if hours_per_frame else 'frame')+')')
        rates = expansion['area_change_rate'].dropna()
        if len(rates):
            low, high = min(0,float(rates.min())), max(0,float(rates.max()))
            margin = max(abs(low),abs(high),1e-6)*.1
            axes[1].set_ylim(low-margin,high+margin)
        fig.suptitle('Colony expansion — border-clipped colonies show observed area only')
        fig.tight_layout(); fig.savefig(root/'colony_expansion.png',dpi=200)
    return root
