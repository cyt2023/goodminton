"""Apply the previously fixed midpoint guard unchanged to all 70 accepted rows.

No fitting/parameter selection. Human contact anchors assist the input, so this
is a retrospective developer diagnostic, never an automatic accuracy claim.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/audit/
from pathlib import Path
from datetime import datetime,timezone
import json,shutil
import numpy as np
import torch
from analysis_jobs import DATA,atomic_json,file_hash
from personal_learning import extractor
from stroke_review import load_source,read_labels
from diagnose_input_repairs import prediction,metrics
from specialist_model import ROOT,BST_NAME

STORE=DATA/'击球标注/直接BST微调'
RUN=STORE/'运行记录/d27cd4b69cbb4a5d8fd9836705f6466c'
OUT=DATA/'击球标注/根因诊断/2026-10-13_同方拍边界约束_全70'

def main():
 OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
 m=json.loads((RUN/'manifest.json').read_text());fm=json.loads((RUN/'personal_features_manifest.json').read_text());rows_source=fm['rows']
 assert len(rows_source)==70
 protocol={'created_at':datetime.now(timezone.utc).isoformat(),'rule':'Identical rule declared in audit_single_hit_guard.py: start>=midpoint previous same-player hit; end<=midpoint next same-player hit+0.25s and before that hit. Only confirmed contact times/player eligibility used; no shot-class-specific choice.',
  'scope':'All 70 already-inspected accepted personal labels; no new independent test, no training, no automatic contact-precision claim.',
  'prior_protocol_sha256':file_hash(DATA/'击球标注/根因诊断/2026-10-13_同方拍边界约束/protocol.json')}
 atomic_json(OUT/'protocol.json',protocol);shutil.copy2(Path(__file__),OUT/'executed_script.py')
 models={};docs={};out=[];packed=[];orders=[]
 with np.load(RUN/'personal_predictions.npz',allow_pickle=False) as a:baseline_logits=a['baseline_logits'].copy()
 max_error=0.
 for i,row in enumerate(rows_source):
  job=row['job_id']
  if job not in models:
   models[job]=extractor(job);report,_,_=load_source(job);docs[job]=read_labels(job,report['source_sha256'])
  clf,dense=models[job];doc=docs[job];h=doc['labels'][row['event_id']];clf.order=list(row['input_quality']['player_input_order']);times=np.asarray([f['source_time'] for f in dense['frames']]);c=int(np.argmin(abs(times-h['peak_time'])))
  base=row['model_window'];v,_=clf.inputs(base);p=prediction(clf,v,h['player']);expected=torch.softmax(torch.from_numpy(baseline_logits[i]),-1).numpy();max_error=max(max_error,float(np.max(abs(np.asarray(p['scores'])-expected))))
  contacts=sorted(set(int(np.argmin(abs(times-a['peak_time']))) for a in doc['labels'].values() if a.get('contact_confirmed') and not a.get('wrong_target') and a.get('play_context')!='between_rallies' and a['stroke_type'] not in ['unknown','no_stroke'] and a['player']==h['player']))
  prev=max((k for k in contacts if times[k]<times[c]-.15),default=None);nxt=min((k for k in contacts if times[k]>times[c]+.15),default=None)
  lo,hi=base['start_frame'],base['end_frame']
  if prev is not None:lo=max(lo,int(np.searchsorted(times,(times[prev]+times[c])/2,side='left')))
  if nxt is not None:hi=min(hi,int(np.searchsorted(times,(times[nxt]+times[c])/2+.25,side='right')-1),nxt-1)
  lo=min(lo,c);hi=max(hi,c);w={**base,'start_frame':lo,'end_frame':hi,'frame_index':c,'peak_time':float(times[c]),'start':float(times[lo]),'end':float(times[hi]+1/dense['sample_fps']),'player':h['player']}
  gv,q=clf.inputs(w);gp=prediction(clf,gv,h['player']);packed.append(gv);orders.append(clf.order)
  out.append({'job_id':job,'event_id':row['event_id'],'human':{k:h[k] for k in ['peak_time','player','stroke_type']},'baseline_window':base,'guard_window':w,'variants':{'original_support':p,'same_player_guard':gp},'quality':q})
 if max_error>1e-5:raise ValueError('Original feature-run replay changed '+str(max_error))
 np.savez_compressed(OUT/'inputs.npz',joints_and_bones=np.stack([v[0] for v in packed]),positions=np.stack([v[1] for v in packed]),shuttle=np.stack([v[2] for v in packed]),video_len=np.asarray([v[3] for v in packed]),orders=np.asarray(orders))
 summary={k:metrics(out,k) for k in ['original_support','same_player_guard']}
 perjob={job:{k:metrics([r for r in out if r['job_id']==job],k) for k in summary} for job in docs}
 atomic_json(OUT/'result.json',{'protocol':protocol,'summary':summary,'per_job':perjob,'rows':out,'baseline_replay_max_probability_error':max_error,'model_sha256':file_hash(ROOT/'models/bst'/BST_NAME),'code_sha256':file_hash(Path(__file__)),'source_manifest_sha256':file_hash(RUN/'manifest.json'),'model_trained':False,'production_changed':False,'labels_changed':False})
 print(json.dumps({'summary':summary,'baseline_replay_error':max_error,'per_job':perjob},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
