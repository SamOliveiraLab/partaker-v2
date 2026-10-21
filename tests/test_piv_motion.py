"""Run in .venv-piv; Qt coordination tests also run in the main environment."""
from pathlib import Path
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT/'src/nd2_analyzer/analysis/piv_motion.py'
spec = importlib.util.spec_from_file_location('partaker_piv_test', MODULE)
piv = importlib.util.module_from_spec(spec); spec.loader.exec_module(piv)
HAS_PIV = importlib.util.find_spec('openpiv') is not None
HAS_QT = importlib.util.find_spec('PySide6') is not None


def field(dx=8., shape=(256,256)):
    xs=np.arange(16,shape[1]-15,8,dtype=float); ys=np.arange(16,shape[0]-15,8,dtype=float)
    grid=(len(ys),len(xs)); good=np.ones(grid,bool)
    return dict(xs=xs,ys=ys,u=np.full(grid,dx),v=np.zeros(grid),accepted=good.copy(),
                confirmed=good.copy(),base=good.copy(),snr=np.full(grid,2.),texture_sd=np.full(grid,30.),
                reverse_error=np.zeros(grid),disagreement=np.zeros(grid,bool))


class GeometryTests(unittest.TestCase):
    def test_import_does_not_load_worker_or_gui_dependencies(self):
        code="import importlib.util,sys; s=importlib.util.spec_from_file_location('motion',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert 'openpiv' not in sys.modules; assert 'PySide6' not in sys.modules"
        subprocess.run([sys.executable,'-c',code,str(MODULE)],check=True,capture_output=True)

    def test_partial_interpolation_rejects_insufficient_weight_and_does_not_extrapolate(self):
        f=field(); f['accepted'][0,0]=False; f['confirmed'][0,0]=False
        delta,ok,weight,trust=piv.sample_grid(f,[[20,20],[17,17],[0,20]])
        self.assertEqual(ok.tolist(),[True,False,False])
        self.assertAlmostEqual(weight[0],.75); self.assertAlmostEqual(delta[0,0],8)
        self.assertFalse(trust[0]); self.assertTrue(np.isnan(delta[1:]).all())

    def test_fixed_conversion_does_not_normalize_each_capture(self):
        a=np.arange(256,dtype=np.uint16).reshape(16,16)*100
        images,info=piv.normalize_frames([a,a+1000])
        self.assertGreater(float(images[1].mean()),float(images[0].mean()))
        self.assertEqual(info['low'],0); self.assertEqual(info['high'],26500)
        old=np.arange(256,dtype=np.uint8).reshape(16,16)
        self.assertTrue(np.array_equal(piv.normalize_frames([old])[0][0],old))

    def test_window_fit_and_interval_are_validated(self):
        with self.assertRaises(ValueError):piv.settings(dict(window=384,interval=1),(256,256))
        with self.assertRaises(ValueError):piv.settings(dict(window=128,interval=0),(256,256))

    def test_segment_cannot_cross_a_colony_hole(self):
        mask=np.ones((100,100),bool); mask[20:80,48:52]=False
        self.assertFalse(piv.segment_inside(mask,np.array([40.,50.]),np.array([60.,50.])))


@unittest.skipUnless(HAS_PIV,'Run scientific tests in the isolated PIV environment.')
class MotionTests(unittest.TestCase):
    def test_known_small_and_beyond_old_range_translation(self):
        import cv2
        rng=np.random.default_rng(2413)
        a=rng.integers(40,181,(512,512),dtype=np.uint8)
        for window,dx,dy in ((128,5,-3),(256,85,-45)):
            b=cv2.warpAffine(a,np.float32([[1,0,dx],[0,1,dy]]),(512,512),borderMode=cv2.BORDER_REFLECT)
            f=piv.estimate_pair(a,b,dict(window=window,interval=1,threshold=160))
            xx,yy=np.meshgrid(f['xs'],f['ys']);good=f['accepted']&(xx>150)&(xx<362)&(yy>150)&(yy<362)
            self.assertTrue(good.any())
            self.assertLess(abs(float(np.median(f['u'][good]))-dx),.5)
            self.assertLess(abs(float(np.median(f['v'][good]))-dy),.5)
            self.assertFalse((f['accepted']&f['disagreement']).any())

    def test_births_stay_in_seed_but_paths_can_leave_seed(self):
        colony=np.ones((256,256),bool); colony[:15]=False; colony[-15:]=False; colony[:,:15]=False; colony[:,-15:]=False
        seed=np.zeros_like(colony); seed[70:180,40:64]=True
        config=dict(interval=1,pixel_um=[.5,.75])
        segs,births=piv.integrate_paths([field() for _ in range(3)],[colony]*4,[seed]*4,[0,1,2,3],config)
        self.assertTrue(births and segs)
        self.assertTrue(all(piv.inside(seed,(r['x_px'],r['y_px'])) for r in births))
        self.assertTrue(any(not piv.inside(seed,(s['x1_px'],s['y1_px'])) for s in segs))
        self.assertTrue(all(piv.inside(colony,(s['x1_px'],s['y1_px'])) for s in segs))
        self.assertTrue(all(s['vx_um_per_time']==4 for s in segs))
        empty=piv.integrate_paths([field()],[colony]*2,[np.zeros_like(seed)]*2,[0,1],config)
        self.assertEqual(empty,([],[]))

    def test_missing_captures_do_not_link_existing_paths(self):
        mask=np.ones((256,256),bool)
        segs,births=piv.integrate_paths([field(),field()],[mask]*3,[mask]*3,[0,1,3],dict(interval=1))
        self.assertTrue(segs); self.assertTrue(all(s['capture_end']==1 for s in segs))

    def test_density_roi_edges_empty_masks_and_spacing(self):
        mask=np.zeros((50,50),bool);mask[10:40,10:40]=True
        values=np.full(mask.shape,.8)
        heat=piv.local_density(values,mask,32)
        np.testing.assert_allclose(heat[mask],.8)
        self.assertTrue(np.isnan(heat[~mask]).all())
        self.assertTrue(np.isnan(piv.local_density(values,np.zeros_like(mask))).all())
        bright=np.zeros(mask.shape,np.uint8);bright[15:17,15:17]=255;bright[15:17,25:27]=255
        row=piv.density_summary(np.full(mask.shape,51,np.uint8),bright,mask)
        self.assertAlmostEqual(row['darkness_proxy'],.8)
        self.assertEqual(row['bright_patch_count'],2)
        self.assertAlmostEqual(row['nearest_patch_centroid_spacing_px'],10)
        self.assertAlmostEqual(row['bright_area_fraction'],8/900)
        self.assertIsNone(piv.density_summary(bright,bright,np.zeros_like(mask))['darkness_proxy'])
        bright[:]=0;bright[15:17,15:17]=255
        self.assertIsNone(piv.density_summary(bright,bright,mask)['nearest_patch_centroid_spacing_px'])

    def test_notebook_zoom_bounds_toggle_clipping_and_missing_colony(self):
        mask=np.ones((485,820),bool)
        self.assertEqual(piv.resolve_zoom([mask],[mask],{})[0],(220,180,580,460))
        self.assertNotEqual(piv.resolve_zoom([mask],[mask],dict(notebook_zoom=False))[0],(220,180,580,460))
        small=np.ones((256,256),bool)
        self.assertEqual(piv.resolve_zoom([small],[small],{})[0],(220,180,256,256))
        small[:]=False;small[20:50,20:50]=True
        self.assertIn('misses',piv.resolve_zoom([small],[small],{})[1])

    def test_worker_writes_all_four_sets_and_clears_old_vectors(self):
        from PIL import Image
        import cv2
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); os.environ.setdefault('MPLCONFIGDIR',str(root/'mpl'))
            rng=np.random.default_rng(41); a=rng.integers(40,220,(256,256),dtype=np.uint8)
            colony=np.zeros_like(a,bool);colony[20:236,20:236]=True
            seed=np.zeros_like(colony);seed[60:180,40:64]=True
            captures=[]
            for t in range(3):
                np.save(root/f'frame{t}.npy',a)
                np.savez(root/f'masks{t}.npz',colony_1=colony,inner_1=seed)
                captures.append(dict(time=t,file=f'frame{t}.npy',masks=f'masks{t}.npz',colonies=[dict(track_id=1)]))
            config=dict(window=128,interval=1.,time_unit='a.u.',threshold=160,include_inner=True,pixel_um=[None,None])
            piv.write_json(root/'manifest.json',dict(config=config,groups=[dict(position=0,channel=0,frames=captures)]))
            with patch.object(piv,'estimate_pair',side_effect=lambda *args:field()):piv.run_manifest(root/'manifest.json')
            folder=root/'position_000_channel_0_colony_1'
            self.assertEqual(len(list(folder.glob('*/*.gif'))),20)
            self.assertEqual(len(list(folder.glob('*/*.png'))),4)
            for g in folder.glob('*/*.gif'):
                with Image.open(g) as im:self.assertEqual(im.n_frames,2)
            for scope in ('whole_colony','inside_roi'):
                density=folder/'density'/scope
                self.assertEqual(len(list(density.glob('*.gif'))),3)
                with Image.open(density/'raw_density_heatmap.gif') as im:self.assertEqual(im.n_frames,3)
                rows=list(csv.DictReader((density/'density_over_time.csv').open()))
                self.assertEqual(len(rows),3)
                self.assertEqual(float(rows[0]['elapsed_time']),0)
            for variant,_,restricted in piv.VARIANTS:
                path=folder/variant
                rows=list(csv.DictReader((path/'path_births.csv').open()))
                self.assertTrue(rows)
                if restricted:self.assertTrue(all(piv.inside(seed,(float(r['x_px']),float(r['y_px']))) for r in rows))
            # Verify the renderer removes a distinct earlier blue displacement.
            render_dir=root/'render';render_dir.mkdir()
            old=dict(x0_px=60,y0_px=80,x1_px=140,y1_px=80,dx_px=80,dy_px=0,time_start=0,time_end=1,capture_end=1,fully_reverse_confirmed=True)
            new=dict(x0_px=170,y0_px=170,x1_px=180,y1_px=170,dx_px=10,dy_px=0,time_start=1,time_end=2,capture_end=2,fully_reverse_confirmed=True)
            piv.render(render_dir,[a]*3,[colony]*3,[seed]*3,[old,new],[],[0,1,2],config,'test',False)
            counts=[]
            for name in ('motion_accumulating_only.gif','motion_current_frame_only.gif'):
                with Image.open(render_dir/name) as im:
                    im.seek(1);rgb=np.array(im.convert('RGB')).astype(int)
                    counts.append(int(((rgb[:,:,2]>rgb[:,:,0]+30)&(rgb[:,:,2]>rgb[:,:,1]+15)).sum()))
            self.assertGreater(counts[0],counts[1]+15)
            self.assertEqual(json.loads((root/'metadata.json').read_text())['status'],'complete')


@unittest.skipUnless(HAS_QT,'Qt coordinator is tested in the main application environment.')
class QtJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtCore import QCoreApplication
        cls.app=QCoreApplication.instance() or QCoreApplication([])

    def test_isolated_process_completes_without_importing_openpiv_in_application(self):
        from PySide6.QtCore import QCoreApplication,QEventLoop,QTimer
        python=Path(os.environ.get('PARTAKER_PIV_TEST_PYTHON',ROOT/'.venv-piv/bin/python'))
        if not python.is_file():self.skipTest('Set PARTAKER_PIV_TEST_PYTHON to an installed PIV runtime.')
        app=self.app
        class Images:
            crop_coordinates=None;registration_offsets=None;image_filename='synthetic Qt integration test'
            def __init__(self):self.a=np.random.default_rng(7).integers(40,181,(256,256),dtype=np.uint8)
            def get(self,t,p,c):return np.roll(self.a,t*3,axis=1)
        images=Images();mask=np.zeros((256,256),bool);mask[20:236,20:236]=True
        inner=np.zeros_like(mask);inner[70:180,60:90]=True
        colonies={(0,t,0):[dict(track_id=1,mask=mask,inner_mask=inner)] for t in (0,1)}
        with tempfile.TemporaryDirectory() as temporary:
            loop=QEventLoop();result=[]
            def done(*value):result.extend(value);loop.quit()
            job=piv.create_export_job(app,images,colonies,temporary,dict(window=128,interval=1.,time_unit='a.u.',threshold=160,include_inner=True,pixel_um=[None,None]),python,lambda *a:None,done)
            timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);timer.start(120000)
            loop.exec();timer.stop()
            if not result:job.cancel();self.fail('Worker timed out')
            self.assertTrue(result[0],result[1]);self.assertNotIn('openpiv',sys.modules)
            self.assertEqual(json.loads((result[2]/'status.json').read_text())['status'],'complete')
            self.assertEqual(len(list(result[2].glob('position*/*/*.gif'))),20)

    def test_cancel_during_runtime_check_stops_without_success(self):
        from PySide6.QtCore import QEventLoop,QTimer,QProcess
        python=Path(os.environ.get('PARTAKER_PIV_TEST_PYTHON',ROOT/'.venv-piv/bin/python'))
        if not python.is_file():self.skipTest('PIV runtime unavailable')
        class Images:
            crop_coordinates=None;registration_offsets=None
        with tempfile.TemporaryDirectory() as temporary:
            loop=QEventLoop();result=[]
            def done(*value):result.extend(value);loop.quit()
            job=piv.create_export_job(self.app,Images(),{(0,0,0):[]},temporary,{},python,lambda *a:None,done)
            QTimer.singleShot(0,job.cancel)
            timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);timer.start(10000)
            loop.exec();timer.stop()
            self.assertTrue(result);self.assertFalse(result[0]);self.assertIn('Cancelled',result[1])
            from shiboken6 import isValid
            if isValid(job.preflight):
                job.preflight.waitForFinished(2000)
                self.assertEqual(job.preflight.state(),QProcess.NotRunning)


if __name__=='__main__':unittest.main()
