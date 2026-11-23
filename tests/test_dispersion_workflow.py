"""Numerical workflow checks; the TIFF regression is opt-in because it is slow."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage as ndi

from nd2_analyzer.analysis.dispersion.region_rules import (
    accepted_partition, raw_material, resolve_edge_collars,
)
from nd2_analyzer.analysis.dispersion.microcolony_dynamics import (
    DispersionAnalysis, DispersionSettings, inner_dark_mask, measure_dynamics,
)


def scene(right=170):
    domain = np.zeros((180, 240), bool)
    domain[10:170, 10:right] = True
    labels = np.where(domain, 1, 0).astype('int32')
    labels[45:135, 100:160] = 2
    return labels, domain


class WorkflowTests(unittest.TestCase):
    def test_collar_requires_motion_support_and_an_initial_edge_colony(self):
        labels, domain = scene()
        for supported, edge in [(set(), {2}), ({2}, set())]:
            result, _ = resolve_edge_collars(labels, domain, supported, edge_candidates=edge)
            np.testing.assert_array_equal(result, labels)
        result, audit = resolve_edge_collars(labels, domain, {2}, edge_candidates={2})
        self.assertGreater(audit['changed_px'], 0)
        np.testing.assert_array_equal(result[labels > 1], labels[labels > 1])
        np.testing.assert_array_equal(result > 0, domain)

    def test_collar_has_no_absolute_direction_and_retains_broad_residual(self):
        labels, domain = scene()
        result, _ = resolve_edge_collars(labels, domain, {2}, edge_candidates={2})
        rotated, _ = resolve_edge_collars(np.rot90(labels), np.rot90(domain), {2}, edge_candidates={2})
        np.testing.assert_array_equal(np.rot90(result), rotated)
        labels, domain = scene(230)
        result, _ = resolve_edge_collars(labels, domain, {2}, edge_candidates={2})
        self.assertTrue((result[60:120, 185:220] == 1).all())

    def test_severed_pink_caps_are_absorbed_unless_protected(self):
        labels, domain = scene()
        labels[10:45, 95:170] = 3
        labels[30:45, 150:170] = 1
        result, _ = resolve_edge_collars(labels, domain, {2}, edge_candidates={2})
        self.assertFalse((result[30:45, 150:170] == 1).any())
        protection = np.zeros_like(domain)
        protection[30:45, 150:170] = True
        result, _ = resolve_edge_collars(labels, domain, {2}, edge_candidates={2}, protected=protection)
        self.assertTrue((result[protection] == 1).all())

    def test_dark_detector_uses_raw_background_statistics_and_eight_connectivity(self):
        raw = np.random.default_rng(42).integers(0, 255, (2, 80, 90), dtype=np.uint8)
        domain = np.zeros_like(raw, bool)
        domain[:, 10:-10, 10:-10] = True
        detected = raw_material(raw, domain)
        for image, roi, actual in zip(raw, domain, detected):
            image = image.astype(float)
            mean = ndi.gaussian_filter(image, 7)
            sd = np.sqrt(np.maximum(0, ndi.gaussian_filter(image * image, 7) - mean * mean))
            z = (mean - ndi.gaussian_filter(image, .6)) / np.maximum(sd, 5)
            cc, _ = ndi.label((z > .8) & roi, np.ones((3, 3)))
            sizes = np.bincount(cc.ravel())
            kept = np.flatnonzero(sizes >= 4)
            np.testing.assert_array_equal(actual, np.isin(cc, kept[kept > 0]))
            self.assertFalse(actual[~roi].any())

    def test_translated_texture_preserves_two_identities_and_roi_coverage(self):
        rng = np.random.default_rng(82)
        yy, xx = np.indices((100, 140))
        image = np.clip(160 + rng.normal(0, 15, xx.shape), 0, 255)
        for x, y in [(35, 35), (100, 65)]:
            image[(xx - x)**2 + (yy - y)**2 < 19**2] -= 75
        image = np.clip(image, 0, 255).astype(np.uint8)
        raw = np.array([np.roll(image, t, axis=1) for t in range(3)])
        outer = np.zeros_like(raw, bool)
        outer[:, 5:-5, 5:-5] = True
        result = accepted_partition(raw, outer)
        np.testing.assert_array_equal(result['labels'] > 0, outer)
        self.assertFalse((result['material'] & ~outer).any())
        for labels in result['labels']:
            self.assertEqual(set(np.unique(labels[labels > 1])), {2, 3})

    def test_manifest_preserves_current_measurement_and_tracking_definition(self):
        with tempfile.TemporaryDirectory() as folder:
            analysis = DispersionAnalysis(SimpleNamespace(), DispersionSettings(), [0], [0], folder,
                dispersion_timing={'hours': [66.]})
            analysis.pink_provenance = {'method': 'central_shared_edges'}
            analysis.write_manifest('complete')
            manifest = json.loads((analysis.output / 'analysis_manifest.json').read_text())
            self.assertEqual(manifest['settings']['colony_threshold_percent'], 39)
            self.assertEqual(manifest['settings']['dark_measurement'], 'intensity_50')
            self.assertEqual(manifest['continuity']['method'], 'checked_affine_and_partial_texture_correspondence')
            self.assertIn('intensity <=128 (50%)', manifest['dark_area_methods']['intensity_50'])
            self.assertIn('current colony territory', manifest['dark_area_methods']['intensity_50'])

    def test_inner_detector_normalizes_only_the_crop_and_keeps_all_dark_components(self):
        import cv2
        raw = np.zeros((100, 120), np.uint16)
        raw[15:-15, 15:-15] = np.random.default_rng(913).integers(10000, 20000, (70, 90), dtype=np.uint16)
        outer = np.zeros(raw.shape, bool)
        outer[18:-18, 18:-18] = True
        expected = np.zeros(raw.shape, bool)
        normalized = cv2.normalize(raw[15:-15, 15:-15], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        expected[15:-15, 15:-15] = (cv2.GaussianBlur(normalized, (5, 5), 0) <= 128) & outer[15:-15, 15:-15]
        actual = inner_dark_mask(raw, outer)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.dtype(bool))
        sizes = np.bincount(ndi.label(actual, np.ones((3, 3)))[0].ravel())[1:]
        self.assertTrue((sizes == 1).any(), 'Single-pixel dark components must survive')
        self.assertFalse(actual[~outer].any())

    def test_breakup_uses_the_measured_inner_area_and_counts_single_pixel_patches(self):
        ids = np.zeros((60, 80), np.int32)
        ids[15:-15, 15:-15] = 7
        measured = np.zeros_like(ids, bool)
        measured[20:23, 20:23] = True
        measured[35, 40] = True
        previous_raw = measured.copy()
        previous_raw[35, 40] = False
        analysis = SimpleNamespace(settings=DispersionSettings(), dispersion_timing={'hours': [66.]},
            colony_rows=[dict(position=0, capture=0, display_id=7, region_type='microcolony', identity_status='observed')],
            positions=[0], frames=[0], focus_intervals=(), check=lambda: None, progress=lambda *_: None,
            load_frame=lambda *_: dict(region_ids=ids, measured_mask=measured, raw_dark_mask=previous_raw))
        row = measure_dynamics(analysis)[0]
        self.assertEqual(row['dark_area_px'], 10)
        self.assertEqual(row['connected_area_px'], 9)
        self.assertAlmostEqual(row['breakup_score'], .1)
        self.assertEqual(row['dark_patch_count'], 2)

    def test_gui_run_selects_the_fitted_detector(self):
        import ast
        source = Path(__file__).resolve().parents[1] / 'src/nd2_analyzer/ui/biofilms/stages/dispersion_controller.py'
        calls = [n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'DispersionSettings']
        self.assertEqual(len(calls), 1)
        method = next(k.value for k in calls[0].keywords if k.arg == 'dark_measurement')
        self.assertEqual(ast.literal_eval(method), 'intensity_50')

    def test_cancel_before_computation(self):
        def stop():
            raise InterruptedError('cancelled')
        with self.assertRaisesRegex(InterruptedError, 'cancelled'):
            accepted_partition(np.zeros((1, 50, 60), np.uint8), np.ones((1, 50, 60), bool), check=stop)

    def test_flat_images_do_not_create_colonies(self):
        raw = np.full((3, 50, 60), 110, np.uint8)
        outer = np.zeros_like(raw, bool)
        outer[:, 5:-5, 5:-5] = True
        result = accepted_partition(raw, outer)
        self.assertFalse((result['labels'] > 1).any())
        np.testing.assert_array_equal(result['labels'] > 0, outer)
        self.assertFalse(result['material'].any())

    def test_capture_and_focus_gaps_do_not_link_identities(self):
        for captures, valid, expected in [([0, 1, 3, 4], [True] * 4, [2, 2, 3, 3]),
                                           ([0, 1, 2, 3], [True, False, True, True], [2, 0, 3, 3])]:
            with self.subTest(captures=captures, valid=valid), tempfile.TemporaryDirectory() as folder:
                analysis = DispersionAnalysis(SimpleNamespace(), DispersionSettings(motion=False), [0], captures, folder,
                    dispersion_timing={'hours': None})
                frames = [dict(raw_gray=np.zeros((80, 90), np.uint8), outer_mask=np.ones((80, 90), bool),
                               vectors_json=np.array('[]')) for _ in captures]
                def partition(raw, outer, **kwargs):
                    return dict(labels=np.full(raw.shape, 2, np.int32), events=[])
                with patch('nd2_analyzer.analysis.dispersion.region_rules.accepted_partition', side_effect=partition) as called:
                    ids, _ = analysis.automatic_territories(0, frames, valid)
                self.assertEqual(called.call_count, 2)
                self.assertEqual([int(mask[40, 40]) for mask in ids], expected)


@unittest.skipUnless(os.environ.get('PARTAKER_TIFF_HOURS'), 'Set PARTAKER_TIFF_HOURS to run the complete 17-frame regression')
class RecordingRegression(unittest.TestCase):
    def test_accepted_masks_and_every_graph_row(self):
        import csv
        import tifffile
        expected = json.loads(Path(__file__).with_name('dispersion_reference.json').read_text())
        folder = Path(os.environ['PARTAKER_TIFF_HOURS'])
        images = []
        for t in range(17):
            image = tifffile.imread(folder / f'{66 + .25*t:.2f}h.tif')
            images.append(image[..., 0] if image.ndim == 3 else image)
        raw = np.asarray(images)
        digest = lambda values: hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        self.assertEqual(digest(raw), expected['raw_sha256'], 'The regression requires the reviewed recording')
        class Images:
            def get(self, t, p, c):
                return raw[t]
        with tempfile.TemporaryDirectory() as output:
            analysis = DispersionAnalysis(Images(), DispersionSettings(), [0], range(17), output,
                dispersion_timing={'hours': [66 + .25*t for t in range(17)]})
            analysis.prepare_recording()
            analysis.run(parameter_sensitivity=False, export_figures=False)
            self.verify_output(analysis, expected)

    def verify_output(self, analysis, expected):
        reference = np.load(Path(__file__).with_name('dispersion_reference_masks.npz'))['labels']
        digest = lambda values: hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        final = []
        for t in range(17):
            frame = analysis.load_frame(0, t)
            labels = frame['region_ids'][15:-15, 15:-15]
            final.append(labels)
            baseline = os.environ.get('PARTAKER_PREVIOUS_DISPERSION')
            if baseline:
                with np.load(Path(baseline) / 'masks' / f'p0_t{t}.npz') as prior:
                    np.testing.assert_array_equal(frame['region_ids'], prior['region_ids'])
            if t == 0:
                self.assertEqual(digest(labels), expected['label_sha256'][t])
            self.assertEqual(digest(frame['measured_mask'][15:-15, 15:-15]), expected['inner50_material_sha256'][t])
            self.assertEqual(digest(frame['raw_dark_mask'][15:-15, 15:-15]), expected['material_sha256'][t])
            np.testing.assert_array_equal(frame['inner_mask'], frame['measured_mask'])
            np.testing.assert_array_equal(frame['region_ids'] > 0, frame['outer_mask'])
            np.testing.assert_array_equal(frame['cell_region_ids'], np.where(frame['measured_mask'], frame['region_ids'], 0))
            domain = reference[t] > 0
            self.assertGreaterEqual(float((labels[domain] == reference[t][domain]).mean()), .99)
        final = np.asarray(final)
        domain = reference > 0
        self.assertGreaterEqual(float((final[domain] == reference[domain]).mean()), .995)
        with (analysis.output / 'dispersion_dynamics.csv').open() as stream:
            import csv
            rows = list(csv.DictReader(stream))
        actual = {(int(r['capture']), int(r['region'])): r for r in rows}
        expected_keys = {(t, i) for t, i, _, _ in expected['graph']}
        self.assertGreaterEqual(len(set(actual) & expected_keys) / len(expected_keys), .99)
        for (t, ident), row in actual.items():
            frame = analysis.load_frame(0, t)
            dark = (frame['region_ids'] == ident) & frame['measured_mask']
            cc, _ = ndi.label(dark, np.ones((3, 3)))
            counts = np.bincount(cc[dark], minlength=1)[1:]
            area, connected = int(dark.sum()), int(counts.max(initial=0))
            self.assertEqual(int(row['dark_area_px']), area)
            self.assertEqual(int(row['connected_area_px']), connected)
            if area:
                self.assertAlmostEqual(float(row['breakup_score']), 1 - connected / area, places=12)
            else:
                self.assertEqual(row['breakup_score'], '')
        for t, ident, area, connected in expected['inner50_colony27_graph']:
            if ident == 27 and area:
                self.assertIn((t, ident), actual)
                self.assertLessEqual(abs(float(actual[t, ident]['breakup_score']) - (1 - connected / area)), .02)
        self.assertFalse(list(analysis.output.rglob('*.gif')))
        self.assertFalse(list(analysis.output.rglob('*.png')))
        self.assertEqual(json.loads((analysis.output / 'analysis_manifest.json').read_text())['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
