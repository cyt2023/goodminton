"""One predeclared label-assisted single-hit boundary guard diagnostic.

Uses only confirmed contact times/player IDs, never stroke types to choose a
window. Midpoints approximate unseen opponent neighbors; this is NOT a ground
truth segmentation or automatic-system score. All 22 selected cases retained.
"""
from datetime import datetime,timezone
from pathlib import Path
import json,time
import numpy as np
import torch
from analysis_jobs import DATA,atomic_json,file_hash
from personal_learning import extractor
from stroke_review import load_source,read_labels
from diagnose_input_repairs import prediction,metrics

BASE=DATA/'击球标注/输入修复诊断/2026-10-12_配对输入修复/result.json'
OUT=DATA/'击球标注/根因诊断/2026-10-13_同方拍边界约束'

def main():
 OUT.mkdir(parents=True,exist_ok=False)
 protocol={'created_at':datetime.now(timezone.utc).isoformat(),'scope':'22 selected developer examples. Human-confirmed contacts assist segmentation; no whole-video or independent accuracy.',
  'rule':'Start/end are original human-translated support, clamped to midpoint with preceding same-player contact and midpoint with following same-player contact+0.25 seconds. End cannot include following same-player contact. Stroke types are not consulted. No parameter search or per-case selection.',
  'limitations':'Sparse labels and same-player midpoints do not verify true opponent contacts; this is a diagnostic guard, not an author-equivalent final segmenter.'}
 atomic_json(OUT/'protocol.json',protocol);old=json.loads(BASE.read_text());models={};docs={};rows=[];cached=[];torch.set_num_threads(1)
 for prior in old['rows']:
  job=prior['job_id']
  if job not in models:
   models[job]=extractor(job);report,_,_=load_source(job);docs[job]=read_labels(job,report['source_sha256'])
  clf,dense=models[job];clf.order=list(prior['variants']['human_center_same_duration']['player_order']);times=np.asarray([f['source_time'] for f in dense['frames']]);human=prior['human'];c=int(np.argmin(abs(times-human['peak_time'])))
  contacts=sorted(set(int(np.argmin(abs(times-a['peak_time']))) for a in docs[job]['labels'].values() if a.get('contact_confirmed') and not a.get('wrong_target') and a.get('play_context')!='between_rallies' and a['stroke_type'] not in ['unknown','no_stroke'] and a['player']==human['player']))
  previous=max((i for i in contacts if times[i]<times[c]-.15),default=None);nxt=min((i for i in contacts if times[i]>times[c]+.15),default=None)
  base=prior['human_shifted_window'];lo,hi=base['start_frame'],base['end_frame']
  if previous is not None:lo=max(lo,int(np.searchsorted(times,(times[previous]+times[c])/2,side='left')))
  if nxt is not None:hi=min(hi,int(np.searchsorted(times,(times[nxt]+times[c])/2+.25,side='right')-1),nxt-1)
  lo=min(lo,c);hi=max(hi,c);fps=dense['sample_fps'];w={**base,'start_frame':lo,'end_frame':hi,'frame_index':c,'peak_time':float(times[c]),'start':float(times[lo]),'end':float(times[hi]+1/fps),'player':human['player']}
  row={'job_id':job,'event_id':prior['event_id'],'human':human,'previous_same_player_frame':previous,'next_same_player_frame':nxt,'guarded_window':w,'variants':{'baseline_human_translated':prior['variants']['human_center_same_duration']}}
  v,q=clf.inputs(w);p=prediction(clf,v,human['player'],prior['automatic_candidate_player']);row['variants']['fixed_same_player_guard']=p;row['quality']=q;rows.append(row);cached.append(v)
 np.savez_compressed(OUT/'inputs.npz',joints_and_bones=np.stack([v[0] for v in cached]),positions=np.stack([v[1] for v in cached]),shuttle=np.stack([v[2] for v in cached]),video_len=np.asarray([v[3] for v in cached]))
 result={'protocol':protocol,'source_sha256':file_hash(BASE),'script_sha256':file_hash(Path(__file__)),'summary':{k:metrics(rows,k) for k in ['baseline_human_translated','fixed_same_player_guard']},'rows':rows,'production_changed':False,'labels_changed':False,'model_trained':False}
 atomic_json(OUT/'result.json',result);print(json.dumps(result['summary'],ensure_ascii=False),flush=True)
if __name__=='__main__':main()
