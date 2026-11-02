"""Research adapter, event alignment, missingness and verified-formula checks."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # prototype root, after move to tests/
import math
import unittest
from research_adapters import import_predictions,ShuttleHitFormerAdapter,DSTAAdapter
from stroke_event_fusion import fuse
from research_quality import spi_scores,hierarchical_fuzzy,fuzzy_module,measure_2d
from research_report import project


class Integration(unittest.TestCase):
    def setUp(self):
        self.times=[i/30 for i in range(120)];self.sha='a'*64
        self.context={'times':self.times,'fps':30.,'match_id':'test'}

    def packet(self,component,rows,**kw):
        return {'component':component,'kind':'imported_prediction','source_sha256':self.sha,
                'model_provenance':{'name':'test_import_not_model_inference'},
                'time_base':{'unit':'frames','fps':25.,'frame_index_origin':0,'interval_end':'exclusive'},'predictions':rows,**kw}

    def hit(self,i,player='near',confidence=None):
        return {'id':str(i),'frame_index':i,'time':self.times[i],'player_id':player,'confidence':confidence,
                'provenance':{'kind':'heuristic_baseline'}}

    def test_25fps_intervals_align_to_30fps_source(self):
        actions=import_predictions(self.packet('dsta',[{'start':20,'end':30,'label':'Smash','confidence':.9}]),'dsta',self.sha,self.times,30.)
        a=actions['predictions'][0];self.assertEqual((a['start_frame'],a['end_frame']),(24,35))
        report=fuse([self.hit(30)],actions,self.context)
        self.assertEqual(report['events'][0]['stroke_type'],'smash');self.assertEqual(report['events'][0]['hit_timestamp'],1.)

    def test_source_hash_units_and_hitter_uncertainty(self):
        packet=self.packet('shuttlehitformer',[{'time':25,'confidence':.8,'player_id':'near'}])
        data=import_predictions(packet,'shuttlehitformer',self.sha,self.times,30.)
        self.assertEqual(data['predictions'][0]['frame_index'],30)
        self.assertIsNone(data['predictions'][0]['player_id'])
        with self.assertRaises(ValueError):import_predictions(packet,'shuttlehitformer','b'*64,self.times,30.)
        packet['time_base']['fps']=None
        with self.assertRaises(ValueError):import_predictions(packet,'shuttlehitformer',self.sha,self.times,30.)

    def test_duplicates_do_not_force_an_actor(self):
        r=fuse([self.hit(30,'near',.3),self.hit(31,'far',.9)],{'predictions':[]},self.context)
        self.assertEqual(len(r['events']),1);self.assertIsNone(r['events'][0]['player_id']);self.assertEqual(len(r['suppressed_hits']),1)

    def test_ambiguous_actions_and_unmatched_intervals_stay_visible(self):
        rows=[{'start':20,'end':30,'label':'Smash','confidence':.8},{'start':20,'end':30,'label':'Clear','confidence':.79},
              {'start':80,'end':90,'label':'Not-a-verified-class','confidence':.9}]
        actions=import_predictions(self.packet('dsta',rows),'dsta',self.sha,self.times,30.)
        r=fuse([self.hit(30)],actions,self.context)
        self.assertEqual(r['events'][0]['stroke_type'],'unknown');self.assertTrue(r['events'][0]['fusion']['ambiguous_actions'])
        self.assertEqual(len(r['unmatched_actions']),3)

    def test_one_broad_interval_does_not_label_multiple_hits(self):
        actions=import_predictions(self.packet('dsta',[{'start':10,'end':70,'label':'Smash','confidence':.9}]),'dsta',self.sha,self.times,30.)
        r=fuse([self.hit(30),self.hit(60)],actions,self.context)
        self.assertEqual(sum(e['stroke_type']=='smash' for e in r['events']),1)

    def test_manual_contact_identity_survives_conflicting_automatic_hit(self):
        human=self.hit(30,'near');human['provenance']['kind']='manual_annotation'
        r=fuse([human,self.hit(31,'far',.9)],{'predictions':[]},self.context)
        self.assertEqual(r['events'][0]['player_id'],'near')

    def test_imported_ids_are_safe_and_end_conventions_are_explicit(self):
        packet=self.packet('dsta',[{'id':'../../unsafe','start':0,'end':0,'label':'Smash','confidence':.9}])
        packet['time_base']['interval_end']='inclusive'
        row=import_predictions(packet,'dsta',self.sha,self.times,30.)['predictions'][0]
        self.assertRegex(row['id'],r'^[0-9a-f]{32}$');self.assertEqual(row['end_frame'],1)
        packet['time_base'].pop('interval_end')
        with self.assertRaises(ValueError):import_predictions(packet,'dsta',self.sha,self.times,30.)

    def test_vfr_clock_and_missing_models(self):
        times=[0.,.04,.08,.16,.2]
        packet=self.packet('shuttlehitformer',[{'time':.15}],time_base={'unit':'seconds'})
        self.assertEqual(import_predictions(packet,'shuttlehitformer',self.sha,times,25.)['predictions'][0]['frame_index'],3)
        self.assertEqual(ShuttleHitFormerAdapter().infer(player_pose=[],shuttle_trajectory=[],court_geometry=None,net_geometry=None)['status'],'unavailable')
        self.assertEqual(DSTAAdapter().infer(rgb_video_path='real.mp4')['feature_extraction_status'],'trained_adapter_checkpoint_missing')

    def test_ground_projection_does_not_change_pixel_units(self):
        self.assertEqual(project([[.01,0,-1],[0,.02,-2],[0,0,1]],[150,125]),[.5,.5])
        self.assertIsNone(project([[1,0,0],[0,1,0],[0,0,0]],[2,3]))

    def test_six_formulas_only_use_matching_validated_references(self):
        m={'stroke_type':'smash','measurement_space':'test_calibrated','swing_peak':15.,'trajectory_error':0.,'wrist_angle':165.,'elbow_angle':172.,
           'step_count':2,'foot_path_efficiency':1.,'racket_mean_normal_error':0.,'racket_normal_change_std':0.}
        p={'validated':True,'stroke_type':'smash','measurement_space':'test_calibrated','swing_peak_ref':15.,'swing_sigma':1.05,
           'historical_trajectory_errors':[1.,2.],'wrist_angle_ref':165.,'wrist_sigma':1.2,'elbow_angle_ref':172.,'elbow_sigma':1.2,
           'step_count_ref':2,'footwork_alpha':.5,'racket_beta':.5}
        self.assertEqual(spi_scores(m,p),[1.]*6)
        with self.assertRaises(ValueError):spi_scores(m,{**p,'stroke_type':'net'})
        with self.assertRaises(ValueError):spi_scores(m,{**p,'validated':False})
        m['swing_peak']=16.05;self.assertAlmostEqual(spi_scores(m,p)[0],math.exp(-.5))

    def test_fuzzy_rules_missing_values_and_midpoint(self):
        self.assertIsNone(hierarchical_fuzzy([1.,None,1.,1.,1.,1.])['score'])
        self.assertAlmostEqual(fuzzy_module([.5,.5])['value'],.5,places=5)
        r=hierarchical_fuzzy([1.]*6);self.assertGreater(r['score'],.7);self.assertTrue(r['level2']['fired_rules'])
        self.assertLess(hierarchical_fuzzy([0.]*6)['score'],.3)

    def test_occluded_pose_is_not_zero_filled_or_scored(self):
        dense={'frames':[{'source_time':i/30,'players':{'near':None,'far':None}} for i in range(10)]}
        event={'player_id':'near','hit_frame':4,'action_start_frame':0,'action_end_frame':9}
        result=measure_2d(event,dense)
        self.assertTrue(all(v is None for v in result['observations_2d'].values()))
        self.assertTrue(all(i['value'] is None for i in result['indices']))


if __name__=='__main__':unittest.main()
