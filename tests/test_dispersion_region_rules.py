import numpy as np
import unittest
import json
import tempfile
import ast
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from nd2_analyzer.analysis.dispersion.region_rules import (
    fragment_fit, merge_decision, normalized_images, refine_regions,
    stationary_fields, stationary_tracks, recover_density,
)


def check_fragment_requires_checked_texture():
    rng = np.random.default_rng(12)
    raw = rng.integers(0, 255, (2, 120, 150), dtype=np.uint8)
    mask = np.zeros(raw.shape[1:], bool)
    mask[30:90, 40:100] = True
    fine, coarse = normalized_images(raw)
    assert fragment_fit(0, 1, mask, fine, coarse) is None
    raw[1] = np.roll(raw[0], 5, axis=1)
    fine, coarse = normalized_images(raw)
    fit = fragment_fit(0, 1, mask, fine, coarse)
    assert fit is not None
    assert abs(fit[0][0, 2] - 5) <= .5
    assert fit[1].sum() >= 120


def check_merge_requires_parent_history_contact_and_common_motion():
    geometry = [dict(frame=t, contact=9, distance=1., parent_support=.8) for t in range(5)]
    intervals = [dict(source=t, target=t + 1, parent_ncc=.9, candidate_ncc=.85,
                      penalty=.03, relative_step=1.) for t in range(4)]
    assert merge_decision(geometry, intervals, True, True)['accepted']
    assert not merge_decision(geometry, intervals, False, True)['accepted']
    assert not merge_decision(geometry, intervals, True, False)['accepted']
    assert not merge_decision(geometry[:2], intervals[:1], True, True)['accepted']
    divergent = [dict(row, relative_step=6.) for row in intervals]
    assert not merge_decision(geometry, divergent, True, True)['accepted']
    unrelated = [dict(row, parent_support=.2) for row in geometry]
    assert not merge_decision(unrelated, intervals, True, True)['accepted']


def check_stationary_structure_controls(kind, expected):
    rng = np.random.default_rng(20261119)
    yy, xx = np.indices((160, 220))
    raw, material = [], []
    outer = np.ones((9, 160, 220), bool)
    outer[:, :8] = outer[:, -8:] = False
    outer[:, :, :8] = outer[:, :, -8:] = False
    for t in range(9):
        x = 55 + 10 * t if kind == 'moving' else 100
        body = (xx - x) ** 2 + (yy - 80) ** 2 <= 25 ** 2
        if kind == 'flat' or kind == 'transient' and t not in (3, 4, 5):
            body[:] = False
        image = 110 + rng.normal(0, 1, body.shape)
        image[body] = 165 + rng.normal(0, 14, body.sum())
        raw.append(image)
        material.append(body if kind != 'flat' else outer[t])
    raw = np.array(raw)
    fields = [stationary_fields(raw, outer, range(9), scale) for scale in (5, 8, 12)]
    from scipy import ndimage as ndi
    coverage = np.array([ndi.uniform_filter(m.astype(float), 25) /
                         np.maximum(1e-6, ndi.uniform_filter(o.astype(float), 25))
                         for m, o in zip(material, outer)])
    tracks = stationary_tracks(fields, coverage, outer, range(9))
    assert bool(tracks) == expected


def check_partition_does_not_invent_colonies_from_flat_images():
    raw = np.full((5, 80, 90), 110, np.uint8)
    outer = np.zeros_like(raw, bool)
    outer[:, 15:-15, 15:-15] = True
    labels = np.zeros_like(raw, np.int32)
    result, pink, events = refine_regions(raw, labels, outer, outer, outer, range(5))
    assert np.array_equal(result > 0, outer)
    assert np.array_equal(pink, outer)
    assert not events


def check_refinement_can_be_cancelled():
    raw = np.zeros((5, 80, 90), np.uint8)
    outer = np.ones_like(raw, bool)
    def stop():
        raise InterruptedError('cancelled')
    with unittest.TestCase().assertRaisesRegex(InterruptedError, 'cancelled'):
        refine_regions(raw, raw, outer, outer, outer, range(5), check=stop)


class RegionRuleTests(unittest.TestCase):
    def test_density_ambiguity_uses_independent_texture(self):
        images = np.array([np.zeros((90, 100), np.float32), np.ones((90, 100), np.float32)])
        mask = np.zeros(images.shape[1:], bool); mask[30:50, 30:50] = True
        def search(source, target, *_):
            if source[0, 0] == 1:
                return [dict(dx=-3, dy=0, ncc=.95)]
            return [dict(dx=10, dy=0, ncc=.95), dict(dx=3, dy=0, ncc=.93)]
        def score(source, target, mask, affine):
            return .8 if abs(affine[0, 2] - 3) < .01 and abs(affine[1, 2]) < .01 else .1
        with patch('nd2_analyzer.analysis.dispersion.region_rules.masked_search', side_effect=search), \
             patch('nd2_analyzer.analysis.dispersion.region_rules.correlation', side_effect=score):
            found = recover_density(0, 1, mask, [images, images, images], images, images, np.ones_like(images, bool))
        assert found is not None
        assert found[0][0, 2] == 3

    def test_verified_fragment_survives_later_recovery(self):
        raw = np.full((5, 90, 100), 110, np.uint8)
        outer = np.ones_like(raw, bool)
        labels = np.ones_like(raw, np.int32)
        labels[0, 20:40, 20:40] = 4
        labels[0, 20:40, 60:80] = 10
        observed = np.zeros(raw.shape[1:], bool); observed[35:55, 45:65] = True
        affine = np.eye(2, 3)
        def fragment(source, target, mask, *_):
            return (affine, observed.copy()) if mask[30, 30] else None
        with patch('nd2_analyzer.analysis.dispersion.region_rules.fragment_fit', side_effect=fragment), \
             patch('nd2_analyzer.analysis.dispersion.region_rules.deformation_fit', return_value=None), \
             patch('nd2_analyzer.analysis.dispersion.region_rules.recover_density', return_value=(affine, observed.copy())):
            result, _, events = refine_regions(raw, labels, labels == 1, outer, outer, range(5))
        assert all((frame == 4).any() for frame in result)
        assert any(event['event'] == 'partially_observed' and event['display_id'] == 4 for event in events)

    def test_graph_uses_selected_colony_only(self):
        path = Path(__file__).resolve().parents[1] / 'src/nd2_analyzer/ui/biofilms/stages/dispersion_page.py'
        page = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == 'DispersionPage')
        method = next(node for node in page.body if isinstance(node, ast.FunctionDef) and node.name == 'select_region')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        selected = [None]
        widget = SimpleNamespace(region=SimpleNamespace(currentData=lambda: selected[0]),
            position_rows=[dict(region='7', capture='0', breakup_score='.2', outward_crossing_candidates=''),
                           dict(region='8', capture='0', breakup_score='.8', outward_crossing_candidates='')],
            candidates=object(), populate=lambda *_: None, render=lambda: None, time=lambda t: t)
        for selection, expected in [(None, {}), (7, {0: '.2'}), (8, {0: '.8'}), (None, {})]:
            selected[0] = selection
            namespace['select_region'](widget)
            assert {t: row['breakup_score'] for t, row in widget.region_rows.items()} == expected

    def test_measurements_use_final_masks(self):
        from nd2_analyzer.analysis.dispersion.microcolony_dynamics import (
            DispersionAnalysis, DispersionSettings, measure_dynamics,
        )
        with tempfile.TemporaryDirectory() as folder:
            analysis = DispersionAnalysis(SimpleNamespace(), DispersionSettings(), [0], range(5), folder,
                                          dispersion_timing={'hours': [66 + .25 * t for t in range(5)]})
            for t in range(5):
                raw = np.full((90, 120), 110, np.uint8)
                outer = np.zeros(raw.shape, bool); outer[15:-15, 15:-15] = True
                ids = np.where(outer, 7, 0).astype(np.int32)
                measured = np.zeros_like(outer); measured[30:35, 30:35] = True
                measured[45:50, 45:50] = True
                record = dict(position=0, capture=t, display_id=7, region_type='microcolony',
                              identity_status='unique_overlap', parent_roi_id=1)
                analysis.colony_rows.append(record)
                analysis.frame_rows.append(dict(position=0, capture=t))
                np.savez_compressed(analysis.mask_path(0, t), raw_gray=raw, outer_mask=outer,
                    inner_mask=measured, measured_mask=measured, hybrid_mask=measured,
                    raw_dark_mask=measured, region_ids=ids, frame_labels=ids,
                    pink_mask=np.zeros_like(outer), darkness=np.zeros_like(raw, dtype=float),
                    records_json=np.array(json.dumps([record])), vectors_json=np.array('[]'))
            analysis.refine_territories()
            rows = measure_dynamics(analysis)
            assert len(rows) == 5
            for row in rows:
                assert row['region'] == 7
                assert row['breakup_score'] == .5
                assert row['dark_area_px'] == 50
                frame = analysis.load_frame(0, row['capture'])
                assert np.array_equal(frame['cell_region_ids'], np.where(frame['measured_mask'], frame['region_ids'], 0))
                assert json.loads(str(frame['records_json']))[0]['dark_area_px'] == 50
            final = np.array([analysis.load_frame(0, t)['region_ids'] for t in range(5)])
            final[:, 40:55, 40:55] = 23
            event = dict(display_id=23, capture=0, event='persistent_structure')
            with patch('nd2_analyzer.analysis.dispersion.region_rules.refine_regions',
                       return_value=(final, np.zeros_like(final, bool), [event])):
                analysis.refine_territories()
            created = [row for row in analysis.colony_rows if row['display_id'] == 23]
            assert len(created) == 5
            assert all(row['identity_status'] == 'persistent_structure' for row in created)

    def test_fragment_texture(self):
        check_fragment_requires_checked_texture()

    def test_merge_evidence(self):
        check_merge_requires_parent_history_contact_and_common_motion()

    def test_structure_controls(self):
        for kind, expected in [('persistent', True), ('moving', False), ('transient', False), ('flat', False)]:
            with self.subTest(kind=kind):
                check_stationary_structure_controls(kind, expected)

    def test_partition(self):
        check_partition_does_not_invent_colonies_from_flat_images()

    def test_cancellation(self):
        check_refinement_can_be_cancelled()


if __name__ == '__main__':
    unittest.main()
