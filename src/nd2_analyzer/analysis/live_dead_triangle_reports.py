"""Diagnostic and area reports for the optional Triangle midpoint workflow."""
import csv
import json
import numpy as np
from matplotlib.figure import Figure
from matplotlib import image as mpl_image
from .live_dead_triangle_midpoint import audit_record, histogram, grade_hist, grid_slices

COLORS = ['#26866b', '#d75b67', '#ad88bd', '#d7d9de']
CATEGORIES = ['Live-only', 'Dead-only', 'Double-positive', 'Neither']
VIABILITY_CATEGORIES = ['Cell Viability-positive', 'Cell Viability-negative']


def display_gray(image):
    valid = np.isfinite(image)
    values = image[valid]
    low, high = np.percentile(values, [1, 99.8])
    return np.clip((np.where(valid, image, low)-low)/max(high-low, 1), 0, 1)


def sampling_rgb(image, labels):
    rgb = np.repeat(display_gray(image)[:, :, None], 3, axis=2)
    cells = labels > 0
    rgb[cells] = .7*rgb[cells]+.3*np.array([0, .5, 1])
    ys = np.linspace(0, labels.shape[0], 4, dtype=int)[1:-1]
    xs = np.linspace(0, labels.shape[1], 4, dtype=int)[1:-1]
    dash_y = np.arange(labels.shape[0]) % 16 < 8
    dash_x = np.arange(labels.shape[1]) % 16 < 8
    for x in xs:
        rgb[dash_y, x] = .55*rgb[dash_y, x]+.45*np.array([1, 0, 0])
    for y in ys:
        rgb[y, dash_x] = .55*rgb[y, dash_x]+.45*np.array([1, 0, 0])
    return rgb


def diagnostic_figure(result, images, labels, kind, role='live', titles=None):
    titles = titles or {r: r.replace('_', ' ').title() for r in result['channels']}
    if role not in result['channels']:
        role = next(iter(result['channels']))
    data = result['channels'][role]
    centers = (data['edges'][:-1]+data['edges'][1:])/2
    fig = Figure(figsize=(16, 11), constrained_layout=True)
    if kind in ('blocks', 'sampling'):
        centers = (data['background_edges'][:-1]+data['background_edges'][1:])/2
        if kind == 'sampling':
            grid = fig.add_gridspec(3, 4, width_ratios=[1.5, 1, 1, 1])
            image_ax = fig.add_subplot(grid[:, 0])
            image_ax.imshow(sampling_rgb(images[role], labels), interpolation='nearest')
            image_ax.set_title(titles[role]+'\nBlue: excluded cells; red dashes: grid', fontsize=9)
            image_ax.set_axis_off()
            for i, (r, c, ys, xs) in enumerate(grid_slices(labels.shape)):
                image_ax.text((xs.start+xs.stop)/2, (ys.start+ys.stop)/2, str(i+1),
                              color='white', ha='center', va='center', fontsize=10,
                              bbox=dict(facecolor='black', alpha=.4, edgecolor='none'))
            axes = [fig.add_subplot(grid[r, c+1]) for r in range(3) for c in range(3)]
        else:
            axes = list(fig.subplots(3, 3).flat)
        for i, ax in enumerate(axes):
            ax.plot(centers, data['blocks'][i], lw=1)
            ax.axvline(data['block_references'].flat[i], color='orange', ls='--', label='Block Triangle')
            ax.axvline(data['cell_detail']['threshold'], color='red', ls='--', label='Cell left Triangle')
            ax.axvline(data['midpoints'].flat[i], color='purple', ls=':', label='Own midpoint')
            ax.axvline(result['shared_grid'].flat[i], color='black', label='Shared threshold')
            sample = data['sampling'][i]
            ax.set_title(f"Block {i+1}: {data['block_details'][i]['grade']}\n"
                         f"{sample['eligible_pixels']:,}/{sample['total_pixels']:,} eligible ({100*sample['eligible_fraction']:.1f}%)",
                         fontsize=8)
            ax.set_xlabel('Original intensity', fontsize=8)
            ax.set_ylabel('Background pixel count', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6)
        fig.suptitle(f"{titles[role]} — cell-excluded background histograms")
    elif kind == 'cells':
        ax = fig.subplots()
        ax.plot(centers, data['cell_histogram'], label='Fluorescence within segmented cells')
        detail = data['cell_detail']
        _, _, smooth = grade_hist(data['cell_histogram'])
        ax.plot(centers, smooth, alpha=.6, label='Smoothed histogram used for Triangle')
        smooth_peak = smooth[detail['peak_index']]
        ax.plot([detail['tail'], detail['peak']],
                [smooth[detail['tail_index']], smooth_peak],
                color='magenta', ls=':', label='Left Triangle endpoints')
        ax.axvline(detail['threshold'], color='red', label=f"Cell left Triangle {detail['threshold']:.1f}")
        ax.axvline(detail['peak'], color='gray', ls=':', label='Tallest significant peak')
        ax.set(title=titles[role], xlabel='Original fluorescence intensity', ylabel='Cell-footprint pixel count')
        ax.legend()
    elif kind == 'preview':
        axes = fig.subplots(len(result['channels']), 4, squeeze=False)
        for row, channel in enumerate(result['channels']):
            gray = display_gray(images[channel])
            mask = result['channels'][channel]['mask']
            for ax, array, title, cmap in zip(axes[row],
                    (gray, np.where(mask, gray, 0), mask, result['surface']),
                    ('Original', 'Selected fluorescence', 'Positive mask', 'Shared threshold surface'),
                    ('gray', 'gray', 'gray', 'viridis')):
                ax.imshow(array, cmap=cmap, interpolation='nearest')
                ax.set_title(titles[channel]+' — '+title)
                ax.set_axis_off()
    else:
        ax = fig.subplots()
        valid = (labels > 0) & np.logical_and.reduce([np.isfinite(image) for image in images.values()])
        total = int(valid.sum())
        if total:
            for channel in result['channels']:
                color = '#d75b67' if channel in ('dead', 'cell_viability') else '#26866b'
                entry = result['channels'][channel]
                centers = (entry['edges'][:-1]+entry['edges'][1:])/2
                for selected, style, alpha, suffix in [(True, '-', 1, 'positive mask'), (False, '--', .45, 'all cell area')]:
                    mask = valid & entry['mask'] if selected else valid
                    h = histogram(images[channel], entry['edges'], mask)
                    ax.plot(100*np.cumsum(h[::-1])[::-1]/total, centers,
                            color=color, ls=style, alpha=alpha, label=titles[channel]+' — '+suffix)
        ax.set(xlabel='Cell area at or above intensity (% of valid cell area)',
               ylabel='Original fluorescence intensity', xlim=(0, 100), title=('Cell Viability' if result['analysis_mode'] == 'cell_viability' else 'Live/Dead')+' cell-area distributions')
        if total:
            ax.legend()
    return fig


def save_frame_reports(folder, result, images, labels, *, time, position, titles=None, visuals=True):
    folder.mkdir(parents=True, exist_ok=True)
    base = f'pos{position}_t{time}'
    audit = audit_record(result)
    audit.update(time=int(time), position=int(position), image_titles=titles)
    (folder/(base+'.json')).write_text(json.dumps(audit, indent=2)+'\n')
    payload = dict(shape=np.asarray(labels.shape), shared_grid=result['shared_grid'],
                   selected_channel=np.asarray(result['selected_channel']))
    for role, entry in result['channels'].items():
        for field in ('edges', 'background_edges', 'blocks', 'cell_histogram', 'midpoints'):
            payload[role+'_'+field] = entry[field]
        payload[role+'_background_eligible_packed'] = np.packbits(entry['background_eligible'].ravel(), bitorder='little')
    payload['bitorder'] = np.asarray('little')
    sampling_rows = []
    for role, entry in result['channels'].items():
        for sample in entry['sampling']:
            sampling_rows.append(dict(time=int(time), position=int(position), role=role,
                                      channel=result['channel_indices'][role], **sample))
    with (folder/(base+'_background_sampling.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sampling_rows[0]))
        writer.writeheader(); writer.writerows(sampling_rows)
    # Grid + shape reconstruct the full surface without storing a huge float image.
    np.savez_compressed(folder/(base+'_threshold_details.npz'), **payload)
    if not visuals:
        return
    image_folder = folder/base
    image_folder.mkdir(parents=True, exist_ok=True)
    for role in result['channels']:
        title = (titles or {}).get(role, f'{base}_{role}')
        for kind in ('blocks', 'sampling', 'cells'):
            fig = diagnostic_figure(result, images, labels, kind, role, titles)
            fig.savefig(folder/(base+'_'+role+'_'+kind+'.png'), dpi=140)
            fig.clear()
        gray = display_gray(images[role])
        rgb = sampling_rgb(images[role], labels)
        mpl_image.imsave(image_folder/(title+'_cell_grid.png'), rgb)
        color = np.array([0, 1, 0] if role == 'live' else [1, 0, 0])
        for suffix, region in [('cell_footprint', labels > 0),
                               ('cell_positive', (labels > 0) & result['channels'][role]['mask'])]:
            colored = np.zeros_like(rgb)
            colored[region] = gray[region, None]*color
            mpl_image.imsave(image_folder/(title+'_'+suffix+'.png'), colored)
    for kind in ('preview', 'intensity_area'):
        fig = diagnostic_figure(result, images, labels, kind, titles=titles)
        fig.savefig(folder/(base+'_'+kind+'.png'), dpi=140)
        fig.clear()


def save_area_reports(folder, rows, time_hours):
    if not rows:
        return []
    folder.mkdir(parents=True, exist_ok=True)
    output, frame_rows, summaries = [], [], []
    categories = rows[0]['composition']['cell']['categories']
    paired = len(categories) == 4
    for row in rows:
        for scope, area in row['composition'].items():
            if area['total']:
                frame_rows.append(dict(time=row['time'], position=row['position'], scope=scope,
                                       valid_pixels=area['total'],
                                       **{k+'_pixels': v for k, v in zip(categories, area['counts'])},
                                       **{k+'_percent': v for k, v in zip(categories, area['percent'])}))
    for scope in ('cell', 'whole_frame'):
        times = sorted({r['time'] for r in rows if r['composition'][scope]['total']})
        if not times:
            continue
        for aggregation in ('mean', 'pooled'):
            for t in times:
                group = [r['composition'][scope] for r in rows if r['time'] == t and r['composition'][scope]['total']]
                total = sum(a['total'] for a in group)
                pct = (np.mean([a['percent'] for a in group], axis=0) if aggregation == 'mean'
                       else 100*np.sum([a['counts'] for a in group], axis=0)/total)
                record = dict(time=t, scope=scope, aggregation=aggregation,
                              valid_positions=len(group), valid_pixels=total,
                              **{k+'_percent': float(v) for k, v in zip(categories, pct)})
                if paired:
                    record['dead_channel_positive_percent'] = float(pct[1]+pct[2])
                summaries.append(record)
            for emphasis in ((False, True) if paired else (False,)):
                fig = composition_figure(rows, time_hours, scope=scope, aggregation=aggregation, emphasis=emphasis)
                prefix = '' if paired else 'cell_viability_'
                path = folder/f'{prefix}{scope}_{aggregation}_{"dead_positive" if emphasis else "composition"}.png'
                fig.savefig(path, dpi=180); fig.clear(); output.append(path)
    for filename, records in [('area_composition_by_frame.csv', frame_rows), ('area_composition_summary.csv', summaries)]:
        if records:
            with (folder/filename).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader(); writer.writerows(records)
    return output

AREA_PLOTS = {
    f'Triangle {"Cell-Area" if scope == "cell" else "Whole-Frame"} {"Dead-Positive" if emphasis else "Composition"} ({aggregation.title()})':
    dict(scope=scope, aggregation=aggregation, emphasis=emphasis,
         filename=f'triangle_midpoint/{scope}_{aggregation}_{"dead_positive" if emphasis else "composition"}.png',
         plot_type='triangle_area')
    for scope in ('cell', 'whole_frame') for aggregation in ('mean', 'pooled') for emphasis in (False, True)
}


VIABILITY_AREA_PLOTS = {
    f'Triangle Cell Viability {"Cell-Area" if scope == "cell" else "Whole-Frame"} Composition ({aggregation.title()})':
    dict(scope=scope, aggregation=aggregation, emphasis=False,
         filename=f'triangle_midpoint/cell_viability_{scope}_{aggregation}_composition.png', plot_type='triangle_area')
    for scope in ('cell', 'whole_frame') for aggregation in ('mean', 'pooled')
}


def composition_figure(rows, time_hours, *, scope, aggregation, emphasis, figure=None, **_unused):
    if figure is None:
        figure = Figure(figsize=(12, 8), constrained_layout=True)
    else:
        figure.clear()
    ax = figure.subplots()
    times = sorted({r['time'] for r in rows if r['composition'][scope]['total']})
    if not times:
        ax.text(.5, .5, 'No valid area for this report', ha='center'); return figure
    values = []
    for time in times:
        group = [r['composition'][scope] for r in rows if r['time'] == time and r['composition'][scope]['total']]
        values.append(np.mean([a['percent'] for a in group], axis=0) if aggregation == 'mean'
                      else 100*np.sum([a['counts'] for a in group], axis=0)/sum(a['total'] for a in group))
    paired = values[0].size == 4
    values = np.array(values); x = np.arange(len(times)); bottom = np.zeros(len(times))
    segments = ([(values[:, 1]+values[:, 2], 'Dead-channel positive (includes double-positive)', COLORS[1]),
                 (values[:, 0], 'Live-only', COLORS[0]), (values[:, 3], 'Neither', COLORS[3])]
                if emphasis and paired else [(values[:, i], CATEGORIES[i], COLORS[i]) for i in range(values.shape[1])])
    if not paired:
        segments = [(values[:, 0], VIABILITY_CATEGORIES[0], COLORS[1]),
                    (values[:, 1], VIABILITY_CATEGORIES[1], COLORS[3])]
    for heights, label, color in segments:
        ax.bar(x, heights, bottom=bottom, color=color, label=label)
        for i, height in enumerate(heights):
            if height >= 3:
                ax.text(i, bottom[i]+height/2, f'{height:.1f}%', ha='center', va='center')
        bottom += heights
    if emphasis and paired:
        ax.bar(x, values[:, 2], bottom=values[:, 1], facecolor='none', edgecolor='#682631',
               hatch='///', linewidth=0, label='Double-positive portion')
    ax.set(ylim=(0, 100), ylabel='Valid '+('segmented cell' if scope == 'cell' else 'whole-frame')+' area (%)',
           title=('Mean across valid positions' if aggregation == 'mean' else 'Integrated across valid positions (pooled pixel area)'))
    ax.set_xticks(x, [f'{time_hours(t):g} h' for t in times])
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.1), ncol=2)
    return figure
