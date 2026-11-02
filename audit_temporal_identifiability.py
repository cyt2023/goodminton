"""Fixed window-extension and peak-metadata identifiability interventions.

Only two known errors, original pose/trajectory and original BST. No fitting or
parameter search. Declaration precedes inference; all numeric inputs saved.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from analysis_jobs import DATA, ROOT, atomic_json, file_hash
from audit_temporal_contract import pred
from audit_temporal_neighbors import JOB, OUT as NEIGHBORS
from personal_learning import extractor
from specialist_model import BST_NAME
from stroke_review import folder_for, read_labels, load_source

OUT=NEIGHBORS/'前后段与目标条件诊断'


def values_digest(values):
    h=hashlib.sha256()
    for a in values:
        a=np.asarray(a);h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def source_window(dense,start,end,peak,player='near'):
    times=[f['source_time'] for f in dense['frames']];fps=float(dense['sample_fps'])
    return {'start_frame':start,'end_frame':end,'frame_index':int(np.argmin(abs(np.asarray(times)-peak))),
            'peak_time':peak,'start':times[start],'end':times[end]+1/fps,'player':player}


def replay(out):
    d=json.loads((out/'result.json').read_text());classifier,_=extractor(JOB);classifier.order=['far','near']
    if file_hash(ROOT/'models/bst'/BST_NAME)!=d['model_sha256'] or file_hash(out/'inputs.npz')!=d['inputs_sha256']:raise ValueError('Replay identity changed')
    err=0
    with np.load(out/'inputs.npz',allow_pickle=False) as inputs:
        for i,(eid,variant) in enumerate(inputs['keys'].tolist()):
            row=next(r for r in d['rows'] if r['event_id']==eid and r['variant']==variant)
            values=(inputs['joints_and_bones'][i],inputs['positions'][i],inputs['shuttle'][i],int(inputs['video_len'][i]))
            p=pred(classifier,values,row['human']);err=max(err,max(abs(a-b) for a,b in zip(p['scores'],row['prediction']['scores'])))
    if err>1e-5:raise ValueError('Replay mismatch')
    print(json.dumps({'replayed':i+1,'max_probability_error':err}),flush=True)


def run(out):
    if (out/'result.json').is_file():replay(out);return
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'protocol.json',{'declared_at':datetime.now(timezone.utc).isoformat(),
        'factorial_windows':['central_author_neighbors','restore_prefix_only','restore_suffix_only','restore_both_original_long'],
        'scope':'Restore the original prefix and suffix separately, keeping target and original extractor coordinates/weights fixed. Suffix restoration changes duration, future context and padding; it does not isolate next-hit pixels alone.',
        'identifiability':'Fix original long bounds 8.541667–11.833334s, compare metadata for human near smash10.041667 vs human near net_cross11.291667; verify numeric input/output equality. This is a design-contract counterexample, not an accuracy test.',
        'negative_control':'No best window selection or training.'})
    shutil.copy2(Path(__file__),out/'executed_script.py')
    report,_,_=load_source(JOB);doc=read_labels(JOB,report['source_sha256']);classifier,dense=extractor(JOB);classifier.order=['far','near']
    protected=[folder_for(JOB)/'annotations.json',ROOT/'models/bst'/BST_NAME,DATA/'专项分析'/JOB/'dense_pose.json',DATA/'专项分析'/JOB/'trajectory.json'];before={str(p):file_hash(p) for p in protected}
    previous=json.loads((NEIGHBORS/'result.json').read_text());rows=[];keys=[];inputs=[]
    for eid in ['76b88ba1502c9ddf3301c666990413c1','8766a907b9da2edb1cb4eeb3844d1c53']:
        subset=[r for r in previous['rows'] if r['event_id']==eid and r['pose']=='original' and r['trajectory']=='original']
        central=next(r for r in subset if r['window_name']=='central_author_neighbors')['window'];long=next(r for r in subset if r['window_name']=='original_long_support')['window'];human=next(r for r in subset)['human']
        windows={'central_author_neighbors':central,
                 'restore_prefix_only':source_window(dense,long['start_frame'],central['end_frame'],human['peak_time']),
                 'restore_suffix_only':source_window(dense,central['start_frame'],long['end_frame'],human['peak_time']),
                 'restore_both_original_long':long}
        for name,w in windows.items():
            v,q=classifier.inputs(w);p=pred(classifier,v,human);rows.append({'event_id':eid,'variant':name,'human':human,'window':w,'prediction':p,'input_digest':values_digest(v),'quality':q});inputs.append(v);keys.append((eid,name))
    # Same physical window contains two user-confirmed, different same-player strokes.
    smash=next(r for r in rows if r['event_id']=='8766a907b9da2edb1cb4eeb3844d1c53' and r['variant']=='restore_both_original_long');long=smash['window'];other=next(a for a in doc['labels'].values() if a['player']=='near' and abs(a['peak_time']-11.291667)<1e-5)
    h={k:other[k] for k in ['player','peak_time','stroke_type']};changed={**long,'peak_time':h['peak_time'],'frame_index':int(np.argmin(abs(np.asarray([f['source_time'] for f in dense['frames']])-h['peak_time'])))}
    v,q=classifier.inputs(changed);p=pred(classifier,v,h);rows.append({'event_id':other['event_id'],'variant':'same_bounds_different_human_target','human':h,'window':changed,'prediction':p,'input_digest':values_digest(v),'quality':q});inputs.append(v);keys.append((other['event_id'],'same_bounds_different_human_target'))
    equal_inputs=rows[-1]['input_digest']==smash['input_digest'];max_score_delta=max(abs(a-b) for a,b in zip(p['scores'],smash['prediction']['scores']))
    if not equal_inputs or max_score_delta!=0.:raise ValueError('Unexpected peak-conditioning difference')
    if any(file_hash(Path(p))!=value for p,value in before.items()):raise ValueError('Protected source changed')
    np.savez_compressed(out/'inputs.npz',keys=np.asarray(keys),joints_and_bones=np.stack([v[0] for v in inputs]),positions=np.stack([v[1] for v in inputs]),shuttle=np.stack([v[2] for v in inputs]),video_len=np.asarray([v[3] for v in inputs]))
    result={'version':'fixed-window-extension-and-peak-identifiability-v1','rows':rows,
        'identifiability':{'numeric_inputs_identical':equal_inputs,'input_digest':smash['input_digest'],'max_probability_delta':max_score_delta,
                          'human_targets':[{k:smash['human'][k] for k in ['peak_time','stroke_type']},{k:h[k] for k in ['peak_time','stroke_type']}],
                          'peak_metadata_reaches_BST_forward':False,'scope':'Same bounds with conflicting same-player targets produce identical inputs; target center is not an explicit BST condition.'},
        'model_sha256':file_hash(ROOT/'models/bst'/BST_NAME),'inputs_sha256':file_hash(out/'inputs.npz'),'protocol_sha256':file_hash(out/'protocol.json'),
        'source_sha256':before,'sources_unchanged':True,'model_trained':False,'weights_changed':False,'human_labels_changed':False,'production_changed':False}
    atomic_json(out/'result.json',result)
    for r in rows:print(json.dumps({'target':r['human'],'variant':r['variant'],'window':[r['window']['start'],r['window']['end']],'prediction':r['prediction']['author_label'],'top_score':max(r['prediction']['scores'])},ensure_ascii=False),flush=True)
    print('IDENTIFIABILITY',json.dumps(result['identifiability'],ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--replay',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    replay(a.output) if a.replay else run(a.output)
