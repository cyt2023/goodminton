import tempfile,unittest,copy
from pathlib import Path
from professional_agents.state import SharedState,StrokeEvent,save,sha
from professional_agents.registry import Workflow
from professional_agents.datasets import check_development
from professional_agents.retrieval import retrieve,relevance_metrics
from professional_agents.evaluation import decode,macro_metrics
from professional_agents.runtime import cache_valid
import numpy as np

class ProfessionalAgents(unittest.TestCase):
 def state(self):
  s=SharedState('v','youtube:abcdefghijk','hash','/raw.mp4',25,[0.,.04,.08],0.,'smoke','public')
  s.events=[StrokeEvent('e',1,.04,'near','smash',{'smash':1.},[0.,.12],provenance={'kind':'model_prediction'})];return s
 def test_roundtrip_typed_state(self):
  p=self.state().packet();self.assertEqual(SharedState.from_packet(p).packet(),p)
 def test_source_frame_clock_alignment(self):
  s=self.state();s.events[0].timestamp=.06
  with self.assertRaises(ValueError):s.validate()
 def test_no_ground_truth_as_predictions(self):
  s=self.state();s.events[0].provenance['kind']='dataset_annotation'
  with self.assertRaises(ValueError):s.validate()
 def test_probabilities_and_missingness(self):
  s=self.state();s.events[0].class_probabilities={'smash':.6}
  with self.assertRaises(ValueError):s.validate()
  s.events[0].class_probabilities={};s.events[0].stroke_label=None;s.events[0].missing=['stroke_type'];s.validate()
 def test_unknown_schema_and_amateur_rejected(self):
  for field,value in [('schema_version','bad-v0'),('domain','amateur')]:
   s=self.state();setattr(s,field,value)
   with self.assertRaises(ValueError):s.validate()
 def test_fixed_dependencies_and_topological_branches(self):
  Workflow().validate();Workflow(tools=('perception.cached','stroke.bst','expert.retrieve','technical.evidence')).validate()
  with self.assertRaises(ValueError):Workflow(tools=('stroke.bst','perception.cached','technical.evidence','expert.retrieve')).validate()
 def test_cycles_unknown_tools_and_termination(self):
  for tools in [('perception.cached','stroke.bst','stroke.bst','technical.evidence','expert.retrieve'),('perception.cached','stroke.bst','arbitrary_python','technical.evidence','expert.retrieve')]:
   with self.assertRaises(ValueError):Workflow(tools=tools).validate()
  with self.assertRaises(ValueError):Workflow(max_calls=3).validate()
  with self.assertRaises(ValueError):Workflow(max_reanalyses=100).validate()
 def test_refinement_cannot_follow_downstream_consumers(self):
  with self.assertRaises(ValueError):Workflow(tools=('perception.cached','stroke.bst','technical.evidence','stroke.window','expert.retrieve')).validate()
 def test_bfmd_and_final_test_cannot_enter_optimizer(self):
  for r in [{'domain':'professional','role':'development','dataset':'BFMD'},{'domain':'professional','role':'sealed_final_test'},{'domain':'amateur','role':'development'}]:
   with self.assertRaises(ValueError):check_development(r)
 def test_retrieval_excludes_query_match_and_unknown(self):
  index={'role':'retrieval_gallery','match_keys':['public-A'],'rows':[{'id':'a','match_key':'public-A','stroke_label':'smash','sequence':[]},{'id':'b','match_key':'public-B','stroke_label':'smash','sequence':[]}]}
  results=retrieve({'match_key':'public-A','stroke_label':'smash'},index);self.assertEqual([r['reference_id'] for r in results],['b']);self.assertEqual(retrieve({'stroke_label':None},index),[])
 def test_retrieval_missing_relevance_not_zero(self):
  self.assertIsNone(relevance_metrics([],None)['nDCG@K']);self.assertEqual(relevance_metrics(['a','b'],{'a':2,'b':1},2)['nDCG@K'],1)
 def test_conditional_decoder_no_truth_input(self):
  x=np.array([[2.,1.9,2.1]]);classes=['Top_長球','Bottom_長球','Top_殺球']
  pred,_=decode(x,classes,Workflow(threshold=0));self.assertEqual(pred,['smash'])
  pred,_=decode(x,classes,Workflow(threshold=0,bst_decoding='coarse_sum'));self.assertEqual(pred,['clear'])
  w=Workflow(threshold=0,tools=('perception.cached','stroke.cached','stroke.check_labels','technical.evidence','expert.retrieve'))
  pred,_=decode(x,classes,w);self.assertEqual(pred,[None])
 def test_macro_f1_counts_abstentions(self):
  r=macro_metrics(['smash','clear'],['smash',None]);self.assertEqual(r['macro_f1'],.5);self.assertEqual(r['coverage'],.5)
 def test_cache_source_artifact_mutation_invalidates(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'output.json';save(p,{'x':1});s=self.state();s.artifacts={'pose':{'artifact':str(p),'sha256':sha(p)}};v={'state':s.packet()};self.assertTrue(cache_valid(v));save(p,{'x':2});self.assertFalse(cache_valid(v))
 def test_frozen_workflow_json_roundtrip(self):
  from professional_agents.state import digest,read
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'protocol.json';protocol={'workflow':Workflow().packet(),'source':'raw_video_hash'};save(p,protocol);self.assertEqual(digest(read(p)),digest(protocol))
   changed=copy.deepcopy(protocol);changed['workflow']['threshold']=.25;self.assertNotEqual(digest(read(p)),digest(changed))
 def test_native_intervals_and_duplicate_events(self):
  s=self.state();s.events.append(copy.deepcopy(s.events[0]))
  with self.assertRaises(ValueError):s.validate()
  s=self.state();s.events[0].interval=[.05,.12]
  with self.assertRaises(ValueError):s.validate()
class HitToolPolicy(unittest.TestCase):
 def info(self):
  from professional_agents.hit_tools import ASSETS
  return {'methods':{k:{'eligible':True,'cost_class':'cheap' if k in ['rules','gru_shared','shf'] else 'expensive'} for k in ASSETS},'quality':{'paired_player_fraction':.9,'observed_ball_fraction':.8,'valid_shf_window_fraction':.3},'original_cached_ready':False}
 def test_auto_prefers_compatible_gru_and_bounded_check(self):
  from professional_agents.hit_tools import choose
  i=self.info();self.assertEqual(choose(i,60),['gru_shared','rules']);self.assertLessEqual(len(choose(i,60)),2)
 def test_original_ready_keeps_original_pipeline_in_toolbox(self):
  from professional_agents.hit_tools import choose
  i=self.info();i['original_cached_ready']=True;i['methods']['gru_original']['cost_class']='cheap';self.assertEqual(choose(i,60)[0],'gru_original')
 def test_shf_choice_needs_sufficient_consecutive_inputs(self):
  from professional_agents.hit_tools import choose
  i=self.info();i['quality']['valid_shf_window_fraction']=.8;self.assertEqual(choose(i,60)[0],'shf');i['methods']['shf']['eligible']=False;self.assertEqual(choose(i,60)[0],'gru_shared')
 def test_expensive_rgb_can_be_selected_for_poor_ball_track(self):
  from professional_agents.hit_tools import choose
  i=self.info();i['quality']['observed_ball_fraction']=.1;i['methods']['gru_original']['eligible']=False;self.assertEqual(choose(i,60,True)[0],'rgb_resnet')
 def test_no_historical_accuracy_or_truth_routing(self):
  from professional_agents.hit_tools import choose
  i=self.info();original=choose(i,60);i['methods']['rules']['historical_f1']=1.;i['ground_truth']=['smash'];self.assertEqual(choose(i,60),original)
 def test_missing_requirements_and_budget_abstain(self):
  from professional_agents.hit_tools import choose
  i=self.info();self.assertEqual(choose(i,0),[])
  for v in i['methods'].values():v['eligible']=False
  self.assertEqual(choose(i,60),[])
 def test_agreement_does_not_union_or_mutate_primary(self):
  from professional_agents.hit_tools import agreement
  a=[{'frame_index':10,'peak_time':.4},{'frame_index':20,'peak_time':.8}];b=[{'frame_index':12,'peak_time':.48},{'frame_index':30,'peak_time':1.2}];original=copy.deepcopy(a);r=agreement(a,b,25);self.assertEqual(r['matched_candidates'],1);self.assertEqual(a,original)
 def test_agents_registered_and_old_fixed_kept(self):
  from professional_agents.registry import TOOLS
  self.assertIn('stroke.gru_original',TOOLS);self.assertIn('stroke.auto',TOOLS);self.assertIn('stroke.rgb_hit',TOOLS);self.assertEqual(Workflow().tools[1],'stroke.bst');Workflow(mode='AGENT',tools=('perception.cached','stroke.auto','technical.evidence','expert.retrieve')).validate()
 def test_filtered2_requires_actual_detector_confidence(self):
  from run_gru_author_reproduction import filtered2_legacy
  person={'keypoints':[{'x':10.,'y':20.} for _ in range(17)]};dense={'source_sha256':'s','source_width':100,'source_height':100,'frames':[{'frame_index':i,'players':{'near':person,'far':person}} for i in range(14)]};ball={'source_sha256':'s','frames':[{'frame_index':i,'raw_visible':False} for i in range(14)]}
  with self.assertRaisesRegex(ValueError,'actual detector confidence'):filtered2_legacy(dense,ball,'naive')
 def test_logit_replay_cannot_claim_hit_tool_evaluation(self):
  from professional_agents.evaluation import ReplayEvaluator
  from unittest.mock import patch
  evaluator=ReplayEvaluator.__new__(ReplayEvaluator)
  with self.assertRaisesRegex(ValueError,'cannot measure hit-tool'):
   evaluator.evaluate(Workflow(mode='AGENT',tools=('perception.cached','stroke.auto','technical.evidence','expert.retrieve')),'development')
if __name__=='__main__':unittest.main()
