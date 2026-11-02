"""Evidence, missingness, isolation and BFMD timing/coordinate regression checks."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # prototype root, after move to tests/
import copy,json,math,tempfile,unittest
from pathlib import Path
from research_pose3d import validate_packet,measure_3d
from tactical_review import build,validate_selection,validate_rallies
from benchmarks.bfmd.importer import manual_timeline
from benchmarks.bfmd.evaluate import evaluate,canonical,pixels,average_precision
from benchmarks.isolation import assert_training_allowed


def event(i,player='near',stroke='clear'):
    return {'event_id':str(i),'hit_frame':i*30,'hit_timestamp':float(i),'action_start_frame':i*30-6,'action_end_frame':i*30+6,
        'action_start_timestamp':i-.2,'action_end_timestamp':i+.2,'player_id':player,'stroke_type':stroke,'player_attribution_status':'human_confirmed',
        'court_coordinates':None,'quality_metrics':{'observations_2d':{'left_elbow_angle_2d_deg':110.},'observations_3d':{}}}
def report():
    return {'job_id':'J_'+'1'*32,'source_sha256':'a'*64,'source_video_id':'1'*32,'timestamps':[i/30 for i in range(301)],'nominal_fps':30.,
        'events':[event(1),event(2,'far','lift'),event(3,'near','smash'),event(6),event(7,'far','lift'),event(8,'near','smash')],'court_calibration':None}
def rally_packet():return {'source_sha256':'a'*64,'kind':'reviewed_rally_annotations','rallies':[{'id':'r1','start':.5,'end':3.5,'review_status':'reviewed'},{'id':'r2','start':5.5,'end':8.5,'review_status':'reviewed'}]}

class Tactical(unittest.TestCase):
    def test_no_rallies_are_invented(self):
        v=build(report());self.assertEqual(v['rally_status'],'unavailable_without_reviewed_boundaries')
        self.assertFalse(any(u['level'] in ['rally','three_stroke'] for u in v['units']))
        self.assertEqual(len([u for u in v['units'] if u['level']=='stroke']),6)
    def test_three_strokes_never_cross_a_rally_boundary(self):
        v=build(report(),rally_packet());seq=[u for u in v['units'] if u['level']=='three_stroke']
        self.assertEqual([u['event_ids'] for u in seq],[['1','2','3'],['6','7','8']])
        match=next(u for u in v['units'] if u['level']=='match');self.assertTrue(any(c['code']=='recurring_sequence' for c in match['claims']))
        self.assertTrue(all(not u['causal_explanation_available'] for u in v['units']))
    def test_unknown_winner_and_missing_outcome_not_filled(self):
        v=build(report(),rally_packet());r=next(u for u in v['units'] if u['level']=='rally')
        self.assertIsNone(r['recorded_outcome']);self.assertFalse(any(c['code']=='recorded_outcome' for c in r['claims']))
    def test_llm_prose_numbers_and_unknown_candidates_rejected(self):
        v=build(report());candidate=next(iter(v['candidate_catalog']))
        r=validate_selection(v,{'items':[{'candidate_id':'invented','text':'lost rally'},{'candidate_id':candidate,'text':'swing speed999','value':999}]})
        self.assertEqual(len(r['items']),1);self.assertNotIn('999',json.dumps(r));self.assertTrue(r['items'][0]['video_references'])
        with self.assertRaises(ValueError):validate_selection(v,{'items':[{'candidate_id':'invented'}]})
    def test_changed_evidence_invalidates_feedback(self):
        a=build(report());r=report();r['events'][0]['quality_metrics']['observations_2d']['left_elbow_angle_2d_deg']=120
        self.assertNotEqual(a['evidence_hash'],build(r)['evidence_hash'])
    def test_bad_reviewed_boundaries_rejected(self):
        p=rally_packet();p['rallies'][1]['start']=2
        with self.assertRaises(ValueError):validate_rallies(p,report())
        p=rally_packet();p['source_sha256']='b'*64
        with self.assertRaises(ValueError):validate_rallies(p,report())
    def test_missing_court_does_not_create_movement(self):
        self.assertTrue(all(u['movement']['status']=='unavailable' for u in build(report())['units']))
    def test_reviewed_paths_do_not_bridge_tracking_gaps(self):
        r=report();r['court_calibration']={'review_status':'confirmed','matrix':[[1,0,0],[0,1,0],[0,0,1]]}
        r['events'][0]['court_coordinates']=[{'frame_index':i,'xy_normalized':[.5,.8]} for i in [24,25,26,35,36,37]]
        r['events'][0]['action_end_frame']=37
        u=build(r)['units'][0];segments=u['movement']['tracks'][0]['segments'];self.assertEqual(len(segments),2)

class Pose3D(unittest.TestCase):
    def packet(self):
        frames=[]
        for i in range(11):
            joints=[{'x':j*.01,'y':j*.005,'z':0.,'score':1.} for j in range(33)]
            joints[11].update(x=0.,y=1.);joints[13].update(x=0.,y=0.);joints[15].update(x=1.+i*.01,y=0.)
            frames.append({'frame_index':i*6,'timestamp':i*.2,'players':{'near':{'joints':joints},'far':None}})
        return {'source_sha256':'a'*64,'joint_schema':'mediapipe33','coordinate_system':'mediapipe_world_hip_origin_xyz','units':'estimated_meters','sample_fps':5.,'frames':frames,'uncertainty':{'calibrated_error_bounds_available':False}}
    def test3d_angles_and_estimated_units_no_quality_score(self):
        p=self.packet();validate_packet(p,'a'*64,[i/30 for i in range(61)])
        e={'player_id':'near','hit_timestamp':1.,'action_start_frame':0,'action_end_frame':60}
        v=measure_3d(e,p);self.assertAlmostEqual(v['observations']['left_elbow_estimated_3d_deg'],90.)
        self.assertIsNone(v['quality_score']);self.assertIsNone(v['body_movement']['global_centre_of_mass_displacement'])
        self.assertFalse(v['uncertainty']['calibrated_error_bounds_available'])
    def test_wrong3d_source_and_clock_rejected(self):
        p=self.packet()
        with self.assertRaises(ValueError):validate_packet(p,'b'*64,[i/30 for i in range(61)])
        p['frames'][0]['timestamp']=.03
        with self.assertRaises(ValueError):validate_packet(p,'a'*64,[i/30 for i in range(61)])
    def test_low_visibility_stays_missing(self):
        p=self.packet()
        for f in p['frames']:
            for j in f['players']['near']['joints']:j['score']=0.
        e={'player_id':'near','hit_timestamp':1.,'action_start_frame':0,'action_end_frame':60}
        v=measure_3d(e,p);self.assertTrue(all(x is None for x in v['observations'].values()));self.assertIsNone(v['quality_score'])

class Benchmark(unittest.TestCase):
    def match(self):return {'match_id':'match','hits':[{'time':1.,'label':'smash'},{'time':2.,'label':'clear'}],'rallies':[],'references':{}}
    def prediction(self):return {'match_id':'match','kind':'model_inference','mode':'end_to_end','time_base':{'unit':'frames','fps':25.,'frame_index_origin':0,'offset_seconds':1.},'evaluation_window':[0,50],
        'provenance':{'model':'unit_test_fixture_not_measured_model_score','input_video_sha256':'b'*64,'annotation_inputs':[]},'hits':[{'frame':0,'label':'smash','confidence':.9},{'frame':25,'label':'clear','confidence':.9}]}
    def test_manual_generic_hit_and_type_are_deduplicated(self):
        annotation={'annotations':[{'result':[{'origin':'manual','value':{'timelinelabels':[label],'ranges':[{'start':30,'end':30}]}} for label in ['hit','smash','bounce','net_hit']]}]}
        result=manual_timeline(annotation,'match');self.assertEqual(len(result['hits']),1);self.assertEqual(result['hits'][0]['label'],'smash')
    def test_explicit_frame_offset_alignment_and_one_to_one_matching(self):
        result=evaluate(self.match(),self.prediction());self.assertEqual(result['metrics']['hit_detection']['tp'],2)
        self.assertEqual(result['metrics']['timestamp_error']['mae_seconds'],0.)
        p=self.prediction();p['hits'].append(dict(p['hits'][0]));result=evaluate(self.match(),p);self.assertEqual(result['metrics']['hit_detection']['fp'],1)
    def test_annotation_cannot_be_submitted_as_model_prediction(self):
        p=self.prediction();p['kind']='manual_annotation'
        with self.assertRaises(ValueError):canonical(p)
        p=self.prediction();p['provenance']['annotation_inputs']=['ground_truth.json']
        with self.assertRaises(ValueError):canonical(p)
    def test_unknown_type_abstentions_are_not_fabricated_typed_predictions(self):
        p=self.prediction()
        for r in p['hits']:r['label']=None
        metric=evaluate(self.match(),p)['metrics']['stroke_types'];self.assertEqual(metric['fp'],0);self.assertEqual(metric['fn'],2);self.assertEqual(metric['abstained_hit_candidates'],2)
    def test_declared_label_mapping_only_coarsens_compatible_labels(self):
        p=self.prediction();p['label_space']='bfmd_raw';p['hits'][0]['label']='net_shot';p['hits'][1]['label']='press'
        rows=canonical(p)['hits'];self.assertEqual(rows[0]['label'],'net');self.assertIsNone(rows[1]['label'])
    def test_missing_ground_truth_metrics_are_null(self):
        metrics=evaluate(self.match(),self.prediction())['metrics']
        self.assertIsNone(metrics['pose3d_MPJPE']['value']);self.assertEqual(metrics['stroke_temporal_mAP']['status'],'unavailable')
        self.assertEqual(metrics['trajectory_ground_truth_error']['status'],'unavailable')
    def test_pixels_normalized_and_percent_coordinates_agree(self):
        self.assertEqual(pixels([50,50],{'space':'percent_0_100','width':1280,'height':720}),[640.,360.])
        self.assertEqual(pixels([.5,.5],{'space':'normalized_0_1','width':640,'height':360}),[640.,360.])
        self.assertEqual(pixels([320,180],{'space':'native_pixels','width':640,'height':360}),[640.,360.])
    def test_temporal_ap_and_empty_predictions(self):
        gt=[{'start':1.,'end':2.}];pred=[{'start':1.,'end':2.,'confidence':.8},{'start':1.,'end':2.,'confidence':.7}]
        self.assertEqual(average_precision(pred,gt,.5),1.);self.assertEqual(average_precision([],gt,.5),0.)
    def test_benchmark_guard_blocks_ids_hashes_and_role(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'sealed.json';path.write_text(json.dumps({'matches':[{'match_id':'heldout_match','youtube_id':'video123'}],'assets':[{'sha256':'deadbeef'}]}))
            for value in [{'benchmark_only':True},{'rows':[{'source_sha256':'deadbeef'}]},{'rows':[{'match_id':'heldout_match'}]},{'rows':[{'source_file':'/tmp/heldout_match.mp4'}]}]:
                with self.assertRaises(ValueError):assert_training_allowed(value,registry_path=path)
            self.assertEqual(assert_training_allowed({'rows':[{'match_id':'training_match'}]},registry_path=path)['status'],'no_known_sealed_asset_overlap')

if __name__=='__main__':unittest.main()
