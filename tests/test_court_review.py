"""Court workflow integration checks; uses a temporary data tree, no human labels."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # prototype root, after move to tests/
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import court_review
import video_workflows as vf
from analysis_jobs import DATA, atomic_json

ROOT=Path(__file__).resolve().parents[1]


class CourtFlow(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='court-flow-test-',dir=DATA)
        self.addCleanup(self.temp.cleanup)
        self.data=Path(self.temp.name)
        self.store=self.data/'视频工作流'
        for d in ['imports','sources','jobs','courts']:(self.store/d).mkdir(parents=True)
        for name,value in [('DATA',self.data),('STORE',self.store)]:
            p=patch.object(vf,name,value);p.start();self.addCleanup(p.stop)
        p=patch.object(court_review,'STORE',self.store/'courts');p.start();self.addCleanup(p.stop)
        # Read an existing source only; all derived files are under the temporary tree.
        c=json.loads((DATA/'专项分析/J_e23c1dd9297d4bfca7c91d1d95065b68/config.json').read_text())
        self.source=Path(c['source_file']);self.sha=c['source_sha256']
        class Analysis:
            submitted=[]
            def video_info(analysis,vid,internal=False):
                return dict(duration=100.,fps=30.,source_sha256=self.sha,source_width=1280,source_height=720)
            def submit_auto(analysis,vid):
                analysis.submitted.append(vid);return {'id':'J_'+('9'*32)}
        self.analysis=Analysis();self.analysis.submitted=[]
        self.flow=vf.VideoWorkflows.__new__(vf.VideoWorkflows)
        self.flow.analysis=self.analysis;self.flow.lock=threading.RLock();self.flow.stop=threading.Event()
        self.flow.process=None
        p=patch.object(vf,'registered_video',return_value={'path':self.source});p.start();self.addCleanup(p.stop)
        self.item=dict(id='1'*32,video_id='source',kind='personal',name='test-only',court_line_color='dark_blue',
                       start=0.,end=None,status='running',jobs=[])

    def prepare(self):
        self.assertIs(self.flow._prepare(self.item),False)
        self.flow._save(self.item)
        self.assertEqual(self.item['status'],'awaiting_court')
        self.assertEqual(self.analysis.submitted,[])
        return json.loads((self.store/'courts'/self.item['id']/'draft.json').read_text())

    def body(self,draft):
        return dict(draft_updated_at=draft['updated_at'],camera_fixed=True,
                    image_points_px=[[300,200],[900,200],[100,650],[1100,650]],
                    court_points_normalized=[[0,0],[1,0],[0,1],[1,1]])

    def test_confirmation_gates_inference_and_persists(self):
        draft=self.prepare()
        result=self.flow.update_court(self.item['id'],'confirm',self.body(draft))
        self.assertEqual(result['item']['court_review_status'],'confirmed')
        self.assertEqual(self.analysis.submitted,[])
        self.flow._prepare(result['item'])
        self.assertEqual(self.analysis.submitted,['source'])
        origin=json.loads((self.store/'sources/source.json').read_text())
        self.assertEqual(origin['court_calibration']['source_sha256'],self.sha)
        self.assertEqual(origin['court_calibration']['review_status'],'confirmed')
        self.assertEqual(origin['court_line_color'],'dark_blue')
        self.assertTrue((self.store/'jobs'/('J_'+('9'*32)+'.json')).exists())

    def test_skip_gates_then_runs_without_calibration(self):
        self.prepare()
        result=self.flow.update_court(self.item['id'],'skip',{})
        self.assertEqual(self.analysis.submitted,[])
        self.flow._prepare(result['item'])
        origin=json.loads((self.store/'sources/source.json').read_text())
        self.assertNotIn('court_calibration',origin)
        self.assertEqual(self.analysis.submitted,['source'])

    def test_invalid_points_and_stale_draft_do_not_queue(self):
        draft=self.prepare();body=self.body(draft)
        for bad in [dict(body,camera_fixed=False),dict(body,draft_updated_at='stale'),
                    dict(body,image_points_px=[[1,1],[2,2],[3,3],[4,4]]),
                    dict(body,image_points_px=[[300,200],[900,200],[1100,650],[100,650]]),
                    dict(body,image_points_px=[[-1,200],[900,200],[100,650],[1100,650]])]:
            with self.assertRaises(ValueError):self.flow.update_court(self.item['id'],'confirm',bad)
        self.assertEqual(self.analysis.submitted,[])
        self.assertEqual(self.flow.court_item(self.item['id'])['status'],'awaiting_court')

    def test_visible_intersections_can_infer_offscreen_corners(self):
        # Four spread, visible service-line intersections; near baseline is offscreen.
        points=[[200,150],[1000,150],[200,550],[1000,550]]
        y1=(6.7-1.98)/13.4;y2=(6.7+1.98)/13.4
        record=dict(review_status='confirmed',source_sha256=self.sha,image_size=[1280,720],
                    image_points_px=points,court_points_normalized=[[0,y1],[1,y1],[0,y2],[1,y2]])
        checked=court_review.vision({'operation':'validate','record':record})
        self.assertEqual(len(checked['inferred_corners_px']),4)
        self.assertLess(max(checked['residual_normalized']),1e-6)
        self.assertLess(checked['inferred_corners_px'][0][1],0)

    def test_clips_receive_their_own_hash_with_same_geometry(self):
        self.item['start']=10.;self.item['end']=20.
        draft=self.prepare();result=self.flow.update_court(self.item['id'],'confirm',self.body(draft))
        def register(path,name,kind,origin):
            atomic_json(self.store/'sources/clip.json',origin);return 'clip'
        self.flow._register=register
        self.flow._command=lambda *args:None
        def metadata(vid,internal=False):
            return dict(duration=100.,fps=30.,source_sha256='clip-sha' if vid=='clip' else self.sha)
        self.analysis.video_info=metadata
        self.flow._prepare(result['item'])
        origin=json.loads((self.store/'sources/clip.json').read_text())
        record=origin['court_calibration']
        self.assertEqual(record['source_sha256'],'clip-sha')
        self.assertEqual(record['original_source_sha256'],self.sha)
        self.assertEqual(record['image_points_px'],self.body(draft)['image_points_px'])
        self.assertEqual(self.analysis.submitted,['clip'])

    def test_same_fixed_camera_source_offers_previous_points(self):
        draft=self.prepare()
        result=self.flow.update_court(self.item['id'],'confirm',self.body(draft))
        self.flow._prepare(result['item'])
        self.item=dict(self.item,id='2'*32,jobs=[],status='running')
        self.item.pop('court_review_status',None)
        self.assertIs(self.flow._prepare(self.item),False)
        restored=json.loads((self.store/'courts'/self.item['id']/'draft.json').read_text())
        self.assertEqual(restored['previous_record']['image_points_px'],self.body(draft)['image_points_px'])
        self.assertEqual(self.analysis.submitted,['source'])


if __name__=='__main__':unittest.main()
