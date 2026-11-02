"""Fixed forensic perturbations, not model repair or independent accuracy.

Author validation controls and the same 22 selected development strokes use
identical frozen BST weights. Never writes production labels, cache or weights.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/audit/
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import json, random,time
import numpy as np
import torch
from analysis_jobs import DATA,atomic_json,file_hash
from bst_public_data import FOLDER,public_classifier,values
from stroke_taxonomy import AUTHOR_MAP

OUT=DATA/'击球标注/根因诊断/2026-10-13_模态依赖'
OLD=DATA/'击球标注/输入修复诊断/2026-10-12_配对输入修复'
VARIANTS=['baseline','zero_shuttle','zero_all_pose','zero_human_target_pose','zero_other_pose','reverse_both_time','static_both_modalities']

def forward(model,j,s,lengths):
 out=[]
 with torch.inference_mode():
  for lo in range(0,len(j),8):
   out.append(model(torch.from_numpy(j[lo:lo+8].copy()),torch.from_numpy(s[lo:lo+8].copy()),torch.from_numpy(lengths[lo:lo+8].copy())).numpy())
 return np.concatenate(out)

def perturb(j,s,lengths,people,variant):
 j,s=j.copy(),s.copy()
 if variant=='zero_shuttle':s[:]=0
 elif variant=='zero_all_pose':j[:]=0
 elif variant in ['zero_human_target_pose','zero_other_pose']:
  for i,p in enumerate(people):
   if p is not None:j[i,:,p if variant=='zero_human_target_pose' else 1-p]=0
 elif variant in ['reverse_both_time','static_both_modalities']:
  for i,n in enumerate(lengths):
   if variant=='reverse_both_time':j[i,:n]=j[i,:n][::-1];s[i,:n]=s[i,:n][::-1]
   else:j[i,:n]=j[i,:n].mean(0);s[i,:n]=s[i,:n].mean(0)
 return j,s

def evaluate(model,classes,j,s,lengths,rows):
 targets=np.asarray([r['target_class_id'] for r in rows]);people=[r['target_person_axis'] for r in rows]
 out={};logits={}
 for variant in VARIANTS:
  a,b=perturb(j,s,lengths,people,variant);pred=forward(model,a,b,lengths);logits[variant]=pred
  ranked=pred.argmax(1);codes=[AUTHOR_MAP[classes[int(k)].split('_',1)[-1]] for k in ranked]
  joint=ranked==targets;typed=np.asarray([c==r['human_code'] for c,r in zip(codes,rows)])
  actor=[];conditioned=[];margins=[]
  for i,p in enumerate(people):
   prefix='Top_' if p==0 else 'Bottom_' if p==1 else None
   actor.append(prefix is not None and classes[int(ranked[i])].startswith(prefix))
   inds=[k for k,c in enumerate(classes) if prefix and c.startswith(prefix)]
   conditioned.append(bool(inds) and max(inds,key=lambda k:pred[i,k])==targets[i])
   other=pred[i].copy();other[targets[i]]=-np.inf;margins.append(float(pred[i,targets[i]]-max(other)))
  out[variant]={'total':len(rows),'type_correct':int(typed.sum()),'joint_correct':int(joint.sum()),'actor_agreement':int(sum(actor)),
   'known_target_type_correct':int(sum(conditioned)),'mean_true_class_logit_margin':float(np.mean(margins)),
   'events':[{'id':r['id'],'human_code':r['human_code'],'source_label':classes[int(k)],'predicted_code':c,'type_match':bool(ok),'joint_match':bool(jo),'true_class_logit_margin':m}
    for r,k,c,ok,jo,m in zip(rows,ranked,codes,typed,joint,margins)]}
  print(variant,{k:v for k,v in out[variant].items() if k!='events'},flush=True)
 return out,logits

def main():
 started=time.perf_counter();torch.set_num_threads(1);classifier=public_classifier();classes=classifier.classes
 diagnostic=json.loads((OLD/'result.json').read_text());rows=diagnostic['rows']
 with np.load(OLD/'inputs.npz',allow_pickle=False) as z:
  keys=z['keys'];indexes=[i for i,k in enumerate(keys) if k[2]=='human_center_same_duration'];j=z['joints_and_bones'][indexes].copy();s=z['shuttle'][indexes].copy();n=z['video_len'][indexes].astype('int64')
  order=[tuple(keys[i][:2]) for i in indexes]
 lookup={(r['job_id'],r['event_id']):r for r in rows};own=[]
 for key in order:
  r=lookup[key];p=r['variants']['human_center_same_duration']['player_order'].index(r['human']['player']);prefix='Top_' if p==0 else 'Bottom_'
  target=next(i for i,c in enumerate(classes) if c.startswith(prefix) and AUTHOR_MAP[c.split('_',1)[-1]]==r['human']['stroke_type'])
  own.append({'id':'/'.join(key),'human_code':r['human']['stroke_type'],'target_person_axis':p,'target_class_id':target})
 own_result,own_logits=evaluate(classifier.model,classes,j,s,n,own)
 assert own_result['baseline']['type_correct']==8
 index=json.loads((FOLDER/'index.json').read_text())['samples'];selected=[]
 for c in classes:
  pool=sorted([r for r in index if r['split']=='val' and r['class']==c],key=lambda r:r['base']);random.Random('modality-audit-v1:'+c).shuffle(pool);selected.extend(pool[:12])
 tensors=[values(classifier,r['base']) for r in selected]
 pj=np.stack([v[0] for v in tensors]);ps=np.stack([v[2] for v in tensors]);pn=np.asarray([v[3] for v in tensors],dtype='int64')
 public_rows=[{'id':r['base'],'human_code':AUTHOR_MAP[r['class'].split('_',1)[-1]],'target_class_id':classes.index(r['class']),
  'target_person_axis':0 if r['class'].startswith('Top_') else 1 if r['class'].startswith('Bottom_') else None} for r in selected]
 public_result,public_logits=evaluate(classifier.model,classes,pj,ps,pn,public_rows)
 OUT.mkdir(parents=True,exist_ok=False)
 np.savez_compressed(OUT/'logits.npz',**{'personal_'+k:v for k,v in own_logits.items()},**{'author_val_'+k:v for k,v in public_logits.items()})
 atomic_json(OUT/'result.json',{'created_at':datetime.now(timezone.utc).isoformat(),'model_sha256':file_hash(classifier_checkpoint()),'script_sha256':file_hash(Path(__file__)),
  'protocol':'Fixed all-zero, target/other-person zero, valid-prefix reversal and static-mean perturbations; no training or choice of best configuration. Public val only, <=12 examples/class.',
  'limitations':'Perturbations are distribution-changing sensitivity controls, not ground-truth corrections. Human target axis for personal data depends on unconfirmed saved mapping. Selected development examples and author validation are not independent deployment tests.',
  'personal_source_result_sha256':file_hash(OLD/'result.json'),'personal_source_inputs_sha256':file_hash(OLD/'inputs.npz'),
  'personal':own_result,'author_val':public_result,'personal_rows':own,'author_val_rows':public_rows,'runtime_seconds':time.perf_counter()-started})
 print('OUTPUT',OUT,flush=True)

def classifier_checkpoint():
 from specialist_model import ROOT,BST_NAME
 return ROOT/'models/bst'/BST_NAME
if __name__=='__main__':main()
