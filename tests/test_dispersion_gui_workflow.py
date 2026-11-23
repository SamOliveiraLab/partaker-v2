"""Controller and display-coordinate checks that do not require native Qt."""
import ast
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGES = ROOT / 'src/nd2_analyzer/ui/biofilms/stages'
WIDGETS = ROOT / 'src/nd2_analyzer/ui/widgets'


def load_class(path, name, namespace, methods=None):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)
    node.bases = [ast.Name(id='object', ctx=ast.Load())]
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef)
                 and (methods is None or n.name in methods)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace[name]


def roi_module():
    spec = importlib.util.spec_from_file_location('dispersion_test_roi', WIDGETS / 'view_roi.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SetupWorkflowTests(unittest.TestCase):
    def controller(self, config=None):
        from nd2_analyzer.analysis.dispersion.microcolony_dynamics import DispersionSettings
        state = SimpleNamespace(experiment=None)
        worker_factory = Mock(side_effect=lambda *args, **kwargs: SimpleNamespace(
            args=args, selector_params=kwargs['selector_params'], start=Mock(),
            progress=Mock(), completed=Mock(), failed=Mock(), finished=Mock()))
        namespace = dict(deepcopy=deepcopy, ApplicationState=SimpleNamespace(get_instance=lambda: state),
                         DispersionSettings=DispersionSettings, DispersionWorker=worker_factory,
                         json=json, csv=csv, logger=Mock(), QMessageBox=Mock())
        cls = load_class(STAGES / 'dispersion_controller.py', 'DispersionController', namespace)
        obj = cls.__new__(cls)
        obj.image_data = SimpleNamespace(data=np.zeros((17, 1, 1, 40, 50)), get=Mock())
        obj.worker = None
        obj.timing = {'mode': 'Hours', 'hours': [66 + t / 4 for t in range(17)]}
        obj.live_result = 'previous result'
        original = dict(threshold=39, first=0, last=16, seeds={0: [30, 35]})
        obj.page = SimpleNamespace(run_config=original, configure_run=Mock(return_value=config),
            folder='previous folder', status=Mock(), set_busy=Mock(), show_progress=Mock(),
            progress=Mock(), show_results=Mock(), select_frame=Mock())
        return obj, worker_factory

    def test_cancel_setup_preserves_current_results_and_starts_no_worker(self):
        obj, factory = self.controller()
        before = deepcopy(obj.page.run_config)
        obj.run()
        factory.assert_not_called()
        self.assertEqual(obj.live_result, 'previous result')
        self.assertEqual(obj.page.folder, 'previous folder')
        self.assertEqual(obj.page.run_config, before)
        obj.page.set_busy.assert_not_called()

    def test_confirm_snapshots_settings_seeds_and_timing(self):
        draft = dict(threshold=42, first=3, last=9, seeds={0: [22, 26]})
        obj, factory = self.controller(draft)
        obj.run()
        worker = factory.return_value if not factory.called else obj.worker
        self.assertEqual(worker.args[1].colony_threshold_percent, 42)
        self.assertEqual(worker.args[1].dark_measurement, 'intensity_50')
        self.assertEqual(worker.args[3], list(range(3, 10)))
        draft['seeds'][0][0] = 99
        obj.timing['hours'][0] = 0
        self.assertEqual(worker.selector_params['colony_seed_points'][0], [22, 26])
        self.assertEqual(worker.args[4]['hours'][0], 66)
        self.assertEqual(obj.live_result, 'previous result')
        self.assertEqual(obj.page.run_config['threshold'], 39)
        self.assertEqual(obj.page.folder, 'previous folder')
        worker.start.assert_called_once()

    def test_failed_or_cancelled_run_keeps_previous_results(self):
        obj, _ = self.controller(dict(threshold=42, first=0, last=16, seeds={}))
        obj.run()
        worker = obj.worker
        worker.isInterruptionRequested = Mock(return_value=True)
        worker.deleteLater = Mock()
        obj.on_complete(SimpleNamespace(data=obj.image_data))
        obj.page.show_results.assert_not_called()
        obj.on_failed('test failure')
        self.assertEqual(obj.live_result, 'previous result')
        self.assertEqual(obj.page.folder, 'previous folder')
        obj.page.set_busy.assert_called_with(True)
        obj.on_finished()
        self.assertIsNone(obj.worker)
        self.assertEqual(obj.live_result, 'previous result')
        obj.page.set_busy.assert_called_with(False)

    def test_incomplete_results_are_rejected_before_replacing_the_view(self):
        import tempfile
        obj, _ = self.controller()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / 'analysis_manifest.json').write_text(json.dumps(dict(
                status='complete', dispersion_timing={'hours': None}, frames=[0], positions=[0])))
            (folder / 'dispersion_dynamics.csv').write_text('position,capture,region\n0,0,27\n')
            with self.assertRaisesRegex(ValueError, 'Missing result mask'):
                obj.read_results(folder, live_result=SimpleNamespace(output=folder))
        obj.page.show_results.assert_not_called()
        self.assertEqual(obj.live_result, 'previous result')
        self.assertEqual(obj.page.folder, 'previous folder')

    def test_popup_cancel_cannot_modify_page_seed_points(self):
        class Dialog:
            Accepted = 1
            def __init__(self, parent, reader, positions, count, config, hours):
                config['seeds'][0][0] = 999
                config['threshold'] = 99
            def exec(self):
                return 0
        cls = load_class(STAGES / 'dispersion_page.py', 'DispersionPage',
                         dict(deepcopy=deepcopy, DispersionSetupDialog=Dialog, QDialog=Dialog), ['configure_run'])
        page = cls.__new__(cls)
        page.run_config = dict(threshold=39, first=0, last=16, seeds={0: [20, 25]})
        page.colony_seed_points = {0: [20, 25]}
        page.recording_count, page.recording_hours = 17, None
        self.assertIsNone(page.configure_run(Mock(), [0]))
        self.assertEqual(page.colony_seed_points, {0: [20, 25]})
        self.assertEqual(page.run_config['threshold'], 39)
        self.assertEqual(page.run_config['seeds'][0], [20, 25])

    def test_draft_changes_do_not_run_detector_and_confirmation_checks_range(self):
        detector = Mock()
        cls = load_class(STAGES / 'dispersion_page.py', 'DispersionSetupDialog',
                         dict(whole_colony_mask=detector), ['mark_pending', 'confirm'])
        dialog = cls.__new__(cls)
        dialog.threshold = SimpleNamespace(value=lambda: 42)
        dialog.threshold_value, dialog.message = Mock(), Mock()
        dialog.first_frame = SimpleNamespace(value=lambda: 9)
        dialog.last_frame = SimpleNamespace(value=lambda: 3)
        dialog.accept, dialog.update_preview = Mock(), Mock()
        dialog.mark_pending(42)
        self.assertIsNone(dialog.preview_key)
        detector.assert_not_called()
        dialog.confirm()
        dialog.accept.assert_not_called()
        dialog.update_preview.assert_not_called()

    def test_threshold_signal_only_marks_preview_pending(self):
        tree = ast.parse((STAGES / 'dispersion_page.py').read_text())
        dialog = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DispersionSetupDialog')
        connections = [n for n in ast.walk(dialog) if isinstance(n, ast.Call)
                       and ast.unparse(n.func) == 'self.threshold.valueChanged.connect']
        self.assertEqual([ast.unparse(n.args[0]) for n in connections], ['self.mark_pending'])


class DisplayCoordinateTests(unittest.TestCase):
    def test_both_full_frame_and_roi_intersections_keep_source_coordinates(self):
        roi = roi_module()
        image = np.arange(485 * 820).reshape(485, 820)
        shown, box = roi.inset_window(image, image, None)
        self.assertEqual(shown.shape, (455, 790))
        self.assertEqual(roi.window_to_frame(0, 0, box), (15, 15))
        self.assertEqual(roi.window_to_frame(789, 454, box), (804, 469))
        np.testing.assert_array_equal(shown, image[15:-15, 15:-15])
        mask = np.zeros(image.shape, bool)
        mask[10:100, 30:200] = True
        shown, box = roi.roi_window(image, mask)
        shown, box = roi.inset_window(image, shown, box)
        self.assertEqual(box, dict(row0=15, row1=100, col0=30, col1=200))
        self.assertEqual(roi.window_to_frame(0, 0, box), (30, 15))
        np.testing.assert_array_equal(shown, image[15:100, 30:200])

    def test_crop_is_dispersion_only_and_zoom_reset_preserves_the_inset(self):
        roi = roi_module()
        namespace = dict(QImage=Mock(), QPixmap=Mock())
        cls = load_class(WIDGETS / 'view_area.py', 'ViewAreaWidget', namespace, ['_create_qimage_and_display'])
        obj = cls.__new__(cls)
        obj.normalize_checkbox = SimpleNamespace(isChecked=lambda: False)
        obj.view_roi = None
        image = np.zeros((485, 820, 3), np.uint8)
        obj._dispersion_display = image
        obj._dispersion_center = (.5, .5)
        obj._fit_display_pixmap = Mock()
        captured = []
        def prepare(shown, normalize):
            captured.append(shown.shape)
            return np.ascontiguousarray(shown), shown.shape[1], shown.shape[0], shown.shape[1]*3, 0
        obj._prepare_image_for_display = prepare
        with patch.dict('sys.modules', {'nd2_analyzer.ui.widgets.view_roi': roi}):
            for zoom in (1, 2, 1):
                obj._dispersion_zoom = zoom
                obj._create_qimage_and_display(image)
                self.assertGreaterEqual(obj._view_box['col0'], 15)
                self.assertGreaterEqual(obj._view_box['row0'], 15)
                self.assertLessEqual(obj._view_box['col1'], 805)
                self.assertLessEqual(obj._view_box['row1'], 470)
                np.testing.assert_array_equal(obj.current_image_data, image)
            self.assertEqual(captured[0], (455, 790, 3))
            self.assertEqual(captured[2], (455, 790, 3))
            obj._dispersion_display = None
            obj._create_qimage_and_display(image)
            self.assertIsNone(obj._view_box)
            self.assertEqual(captured[-1], image.shape)

    def test_stage_axes_hide_margin_without_rebasing_colony_coordinates(self):
        from matplotlib.figure import Figure
        tree = ast.parse((STAGES / 'dispersion_page.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fit_analysis_view')
        namespace = dict(BORDER=15)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<display bounds>', 'exec'), namespace)
        axes = Figure().subplots()
        image = np.zeros((485, 820))
        axes.imshow(image)
        namespace['fit_analysis_view'](axes, image)
        self.assertEqual(axes.get_xlim(), (14.5, 804.5))
        self.assertEqual(axes.get_ylim(), (469.5, 14.5))
        # The image extent stays in full-frame coordinates for colony picking.
        np.testing.assert_array_equal(axes.images[0].get_extent(), [-.5, 819.5, 484.5, -.5])

class ColonyCloseupTests(unittest.TestCase):
    def viewer(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        cls = load_class(STAGES / 'dispersion_page.py', 'DispersionPage', dict(np=np, BORDER=15),
                         ['render_detail', 'time', 'seek_time', 'select_frame', 'seek_plot'])
        page = cls.__new__(cls)
        figure = Figure(figsize=(5, 6))
        FigureCanvasAgg(figure)
        page.detail, page.chart = figure.subplots(2, 1)
        page.preview = Figure().subplots()
        page.hours, page.captures, page.folder = [66, 66.25, 66.5], [0, 1, 2], Path('/results')
        page.frame_requested = Mock()
        page.position = SimpleNamespace(currentData=lambda: 0)
        page.region = SimpleNamespace(findData=lambda ident: {27: 1, 30: 2}.get(ident, -1),
                                      setCurrentIndex=Mock())
        page.current = 0
        def set_frame(index):
            page.current = index
            page.select_frame()
        page.frame = SimpleNamespace(currentData=lambda: page.current,
                                    findData=lambda capture: capture, setCurrentIndex=set_frame)
        page.images = [np.full((100, 160, 3), t, np.uint8) for t in range(3)]
        page.labels = [np.zeros((100, 160), np.int32) for _ in range(3)]
        page.labels[0][30:50, 30:50] = 27
        page.labels[1][40:60, 75:95] = 27
        page.labels[2][40:60, 75:95] = 30
        def render():
            page._overlay_masks = {'region_ids': page.labels[page.current], 'raw_gray': page.images[page.current][..., 0]}
            page.render_detail(page.images[page.current], 27)
        page.render = render
        page.render()
        return page

    def test_time_scrubbing_moves_selected_colony_closeup_and_requests_full_frame(self):
        page = self.viewer()
        initial = page.detail.get_xlim()
        page.seek_time(66.25)
        self.assertEqual(page.detail.get_title(), 'Colony 27 · 66.25 h')
        self.assertGreater(page.detail.get_xlim()[0], initial[0])
        np.testing.assert_array_equal(page.detail.images[0].get_array(), page.images[1])
        page.frame_requested.emit.assert_called_with(0, 1)
        # The close-up retains full-image coordinates, so clicking its colony selects the same ID.
        page.seek_plot(SimpleNamespace(button=1, inaxes=page.detail, xdata=80, ydata=45))
        page.region.setCurrentIndex.assert_called_with(1)

    def test_missing_selected_colony_does_not_zoom_to_another_colony(self):
        page = self.viewer()
        page.seek_time(66.5)
        self.assertFalse(page.detail.images)
        self.assertEqual(page.detail.texts[0].get_text(), 'Colony 27 not detected in this frame')
        page.frame_requested.emit.assert_called_with(0, 2)

    def test_unselected_closeup_is_empty(self):
        page = self.viewer()
        page.render_detail(page.images[0], None)
        self.assertFalse(page.detail.images)
        self.assertIn('Select a microcolony', page.detail.texts[0].get_text())

class TrackingOverlayTests(unittest.TestCase):
    def functions(self):
        import cv2
        from scipy import ndimage as ndi
        tree = ast.parse((STAGES / 'dispersion_page.py').read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('draw_tracking_overlay', 'draw_breakup_overlay')]
        namespace = dict(np=np, cv2=cv2, json=json, ndi=ndi)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<display overlays>', 'exec'), namespace)
        return namespace['draw_tracking_overlay'], namespace['draw_breakup_overlay']

    def test_tracking_keeps_colony_fill_and_does_not_invent_motion(self):
        tracking, _ = self.functions()
        ids = np.zeros((50, 70), np.int32)
        ids[8:36, 8:45] = 27
        frame = {'region_ids': ids, 'raw_gray': np.zeros(ids.shape, np.uint8)}
        rgb = np.full((*ids.shape, 3), [40, 70, 90], np.uint8)
        with patch('cv2.arrowedLine') as arrow:
            output = tracking(rgb.copy(), frame, 27)
        arrow.assert_not_called()
        np.testing.assert_array_equal(output[15, 15], rgb[15, 15])
        np.testing.assert_array_equal(output[8, 15], [0, 230, 255])
        # The centroid marker is not converted into a synthetic displacement arrow.
        np.testing.assert_array_equal(frame['region_ids'], ids)

    def test_motion_uses_saved_vectors_for_selected_identity_only(self):
        tracking, _ = self.functions()
        ids = np.full((50, 70), 27, np.int32)
        vectors = [dict(display_id=27, start=[12, 15], end=[35, 15]),
                   dict(display_id=30, start=[12, 30], end=[35, 30])]
        frame = {'region_ids': ids, 'vectors_json': np.array(json.dumps(vectors))}
        rgb = np.zeros((*ids.shape, 3), np.uint8)
        output = tracking(rgb.copy(), frame, 27, territory=False)
        self.assertTrue(output[15, 20].any())
        self.assertFalse(output[30, 20].any())
        frame['vectors_json'] = np.array('[]')
        np.testing.assert_array_equal(tracking(rgb.copy(), frame, 27, territory=False), rgb)

    def test_breakup_matches_each_territory_components_and_keeps_central_blue(self):
        _, breakup = self.functions()
        ids = np.zeros((50, 70), np.int32)
        ids[5:25, 5:25] = 27
        ids[5:25, 25:45] = 30
        ids[25:45, 10:50] = 1
        measured = np.zeros(ids.shape, bool)
        measured[10:14, 21:29] = True  # One touching patch, split by the shared colony border.
        measured[7, 7] = True
        measured[7, 40] = True
        pink = ids == 1
        measured[30:33, 15:20] = True
        frame = dict(region_ids=ids, inner_mask=measured, measured_mask=measured, pink_mask=pink)
        before = {key: value.copy() for key, value in frame.items()}
        output = breakup(np.zeros((*ids.shape, 3), np.uint8), frame)
        yellow, blue = np.array([178, 147, 28]), np.array([21, 98, 178])
        for ident in (27, 30):
            territory = ids == ident
            self.assertEqual(np.count_nonzero(np.all(output == yellow, axis=2) & territory), 16)
            self.assertEqual(np.count_nonzero(np.all(output == blue, axis=2) & territory), 1)
        self.assertTrue(np.all(output[pink & measured] == blue))
        self.assertFalse(output[pink & ~measured].any())
        self.assertFalse(output[(ids == 0) | ((ids > 1) & ~measured)].any())
        for key in frame:
            np.testing.assert_array_equal(frame[key], before[key])

    def test_resized_graph_label_stays_within_its_separate_canvas(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        tree = ast.parse((STAGES / 'dispersion_page.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fit_analysis_view')
        namespace = dict(np=np, BORDER=15)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<image bounds>', 'exec'), namespace)
        cls = load_class(STAGES / 'dispersion_page.py', 'DispersionPage', namespace,
                         ['render_empty', 'render_detail'])
        page = cls.__new__(cls)
        page.chart, page.recording_preview = None, None
        page.image_text_color = '#eeeeee'
        for name in ('field_plot', 'detail_plot', 'plot'):
            figure = Figure(figsize=(4, 3), layout='constrained')
            setattr(page, name, FigureCanvasAgg(figure))
        page.render_empty()
        self.assertIsNot(page.detail.figure, page.chart.figure)
        for width, height, dpi in ((2.6, 2, 100), (4, 3, 100), (4, 3, 200)):
            page.plot.figure.set_size_inches(width, height)
            page.plot.figure.set_dpi(dpi)
            page.plot.draw()
            label = page.chart.yaxis.label.get_window_extent(page.plot.get_renderer())
            self.assertGreaterEqual(label.x0, 0)
            self.assertGreaterEqual(label.y0, 0)
            self.assertLessEqual(label.y1, page.plot.figure.bbox.height)

class DisplacementTrailTests(unittest.TestCase):
    def advance(self, motion):
        tree = ast.parse((STAGES / 'dispersion_page.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'advance_colony_trail')
        ns = dict(np=np, BORDER=15, image_motion=motion)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<advected trail>', 'exec'), ns)
        return ns['advance_colony_trail']

    def test_reference_point_follows_image_fit_instead_of_changed_mask_centroid(self):
        source = dict(raw_gray=np.zeros((100, 140), np.uint8), region_ids=np.zeros((100, 140), np.int32))
        source['region_ids'][35:55, 40:60] = 27
        target = {k: v.copy() for k, v in source.items()}
        target['region_ids'][:] = 0
        target['region_ids'][35:55, 60:80] = 27
        motion = Mock(return_value=([], np.array([[1., 0., 2.], [0., 1., -1.]])))
        path = [[49.5, 44.5]]
        result = self.advance(motion)(source, target, 27, path)
        np.testing.assert_allclose(result, [[49.5, 44.5], [51.5, 43.5]])
        self.assertEqual(path, [[49.5, 44.5]])
        self.assertNotEqual(result[-1][0], 69.5)

    def test_unsupported_interval_restarts_and_supported_history_is_limited_to_six(self):
        ids = np.zeros((100, 140), np.int32)
        ids[35:55, 40:60] = 27
        frame = dict(raw_gray=np.zeros(ids.shape, np.uint8), region_ids=ids)
        path = [[float(x), 44.5] for x in range(42, 50)]
        unsupported = self.advance(Mock(return_value=([], None)))
        self.assertEqual(unsupported(frame, frame, 27, path), [[49.5, 44.5]])
        translated = self.advance(Mock(return_value=([], np.array([[1., 0., 2.], [0., 1., 0.]]))))
        result = translated(frame, frame, 27, path)
        self.assertEqual(len(result), 6)
        self.assertEqual(result[-1], [51., 44.5])
        empty = dict(raw_gray=frame['raw_gray'], region_ids=np.zeros_like(ids))
        self.assertEqual(translated(frame, empty, 27, path), [])

    def test_cached_scrubbing_does_not_bridge_capture_gaps(self):
        import tempfile
        tree = ast.parse((STAGES / 'dispersion_page.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'advance_colony_trail')
        motion = Mock(return_value=([], np.array([[1., 0., 2.], [0., 1., 0.]])))
        ns = dict(np=np, BORDER=15, image_motion=motion)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<trail>', 'exec'), ns)
        cls = load_class(STAGES / 'dispersion_page.py', 'DispersionPage', ns, ['displacement_path'])
        with tempfile.TemporaryDirectory() as folder:
            page = cls.__new__(cls)
            page.folder, page.captures = Path(folder), [0, 1, 3, 4]
            page.region_rows = {t: {'focused': 'True'} for t in page.captures}
            (page.folder / 'masks').mkdir()
            ids = np.zeros((100, 140), np.int32)
            ids[35:55, 40:60] = 27
            for t in page.captures:
                np.savez(page.folder / 'masks' / f'p0_t{t}.npz', region_ids=ids, raw_gray=np.zeros(ids.shape, np.uint8))
            later = page.displacement_path(0, 4, 27)
            self.assertEqual(len(later), 2)
            self.assertEqual(page._trail_state['paths'][3], [[49.5, 44.5]])
            before = motion.call_count
            earlier = page.displacement_path(0, 1, 27)
            self.assertEqual(earlier, [[49.5, 44.5], [51.5, 44.5]])
            self.assertEqual(motion.call_count, before)

    def test_zoom_grays_neighbors_without_altering_full_field_overlay(self):
        page = ColonyCloseupTests().viewer()
        overlay = np.full((100, 160, 3), [170, 35, 90], np.uint8)
        before = overlay.copy()
        raw = np.full((100, 160), 100, np.uint8)
        page._overlay_masks['raw_gray'] = raw
        selected = page._overlay_masks['region_ids'] == 27
        page.render_detail(overlay, 27)
        shown = np.asarray(page.detail.images[0].get_array())
        np.testing.assert_array_equal(shown[selected], overlay[selected])
        self.assertTrue((shown[~selected] == 100).all())
        np.testing.assert_array_equal(overlay, before)

class TrailEndpointMarkerTests(unittest.TestCase):
    def test_cyan_diamond_uses_exact_white_trail_endpoint(self):
        from matplotlib.figure import Figure
        from matplotlib import patheffects
        cls = load_class(STAGES / 'dispersion_page.py', 'DispersionPage',
                         dict(np=np, BORDER=15, patheffects=patheffects), ['draw_displacement'])
        page = cls.__new__(cls)
        page.preview, page.detail = Figure().subplots(), Figure().subplots()
        page._overlay_masks = {'region_ids': np.zeros((100, 160), np.int32)}
        page.layer_actions = {'tracking': SimpleNamespace(isChecked=lambda: True)}
        page.position, page.frame = SimpleNamespace(currentData=lambda: 0), SimpleNamespace(currentData=lambda: 3)
        path = [[35.2, 40.7], [37.8, 42.1], [40.25, 43.75]]
        page.displacement_path = lambda *args: path
        for axes in (page.preview, page.detail):
            page.draw_displacement(axes, 27)
            trail, diamond = axes.lines
            self.assertEqual(diamond.get_marker(), 'D')
            np.testing.assert_array_equal([trail.get_xdata()[-1], trail.get_ydata()[-1]],
                                          [diamond.get_xdata()[0], diamond.get_ydata()[0]])
            np.testing.assert_array_equal([diamond.get_xdata()[0], diamond.get_ydata()[0]], path[-1])

    def test_raster_marker_uses_advected_point_without_a_second_centroid_marker(self):
        import cv2
        tracking, _ = TrackingOverlayTests().functions()
        ids = np.full((50, 70), 27, np.int32)
        frame = {'region_ids': ids}
        with patch('cv2.drawMarker', wraps=cv2.drawMarker) as marker:
            tracking(np.zeros((*ids.shape, 3), np.uint8), frame, 27, motion=False, point=[40.25, 15.75])
        self.assertEqual(marker.call_count, 2)  # White edge and cyan diamond at one position.
        for call in marker.call_args_list:
            self.assertEqual(call.args[1], (40, 16))
        with patch('cv2.drawMarker') as marker:
            tracking(np.zeros((*ids.shape, 3), np.uint8), frame, 27, motion=False,
                     point=[40.25, 15.75], marker=False)
        marker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
