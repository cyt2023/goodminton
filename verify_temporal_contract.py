"""Independent replay of temporal diagnostics from saved numeric arrays only.

No video extraction, labels changes, training, or parameter search. The optional
--write-report output is confined to the diagnostic directory.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analysis_jobs import ROOT, file_hash
from bst_public_data import public_classifier

DEFAULT=ROOT.parent/'03_数据/击球标注/时间契约诊断/2026-10-13_固定单因素'


def input_digest(values):
    h=hashlib.sha256()
    for value in values:
        a=np.asarray(value);h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def verify_folder(folder,classifier,kind):
    doc=json.loads((folder/'result.json').read_text());protocol=json.loads((folder/'protocol.json').read_text())
    assert file_hash(folder/'protocol.json')==doc['protocol_sha256'],'Protocol hash mismatch'
    assert file_hash(folder/'inputs.npz')==doc.get('inputs_sha256',doc.get('input_sha256')),'Inputs hash mismatch'
    checkpoint=ROOT/'models/bst/bst_0_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt'
    assert file_hash(checkpoint)==doc['model_sha256'],'Model hash mismatch'
    if kind=='temporal':
        assert file_hash(ROOT/'audit_temporal_contract.py')==protocol['code_sha256'],'Declared code changed'
        for path,digest in doc['protected_source_sha256'].items():assert file_hash(Path(path))==digest,path
    elif kind=='neighbors':
        assert file_hash(ROOT/'audit_temporal_neighbors.py')==file_hash(folder/'executed_script.py'),'Executed code changed'
        for item in doc['sources'].values():assert file_hash(Path(item['path']))==item['sha256'],item['path']
    else:
        assert file_hash(ROOT/'audit_temporal_identifiability.py')==file_hash(folder/'executed_script.py'),'Executed code changed'
        for path,digest in doc['source_sha256'].items():assert file_hash(Path(path))==digest,path
    groups=Counter();err=0.;count=0;all_inputs={}
    with np.load(folder/'inputs.npz',allow_pickle=False) as data:
        for i,key in enumerate(data['keys'].tolist()):
            if kind=='temporal':
                job,eid,variant=key;row=next(r for r in doc['rows'] if r['job_id']==job and r['event_id']==eid);expected=row['predictions'][variant]
            elif kind=='neighbors':
                eid,pose,traj,variant=key;row=next(r for r in doc['rows'] if r['event_id']==eid and r['pose']==pose and r['trajectory']==traj and r['window_name']==variant);expected=row['prediction']
            else:
                eid,variant=key;row=next(r for r in doc['rows'] if r['event_id']==eid and r['variant']==variant);expected=row['prediction']
            j,p,s,n=(data['joints_and_bones'][i],data['positions'][i],data['shuttle'][i],int(data['video_len'][i]))
            assert 0<n<=100 and j.shape==(100,2,72) and p.shape==(100,2,2) and s.shape==(100,2)
            assert np.isfinite(j).all() and np.isfinite(p).all() and np.isfinite(s).all()
            with torch.inference_mode():
                scores=classifier.model(torch.from_numpy(j[None]),torch.from_numpy(s[None]),torch.tensor([n]))[0].softmax(-1).numpy()
            err=max(err,float(np.max(abs(scores-np.asarray(expected['scores'])))))
            assert classifier.classes[int(scores.argmax())]==expected['author_label']
            if row.get('input_digest'):assert input_digest((j,p,s,n))==row['input_digest']
            all_inputs[tuple(key)]=(j.copy(),p.copy(),s.copy(),n)
            groups[variant]+=1;count+=1
    assert err<=1e-6,err
    if kind=='neighbors':
        long=[r for r in doc['rows'] if r['window_name']=='original_long_support'];isolated=[r for r in doc['rows'] if r['window_name']!='original_long_support']
        assert len(long)==8 and len(isolated)==24
        assert all(r['prediction']['author_label']=='Bottom_切球' for r in long)
        assert all(r['prediction']['author_label']=='Bottom_殺球' for r in isolated)
        assert all(r['window']['author_extension_frames']==6 for r in isolated)
        assert all(not r['known_other_contacts']['same_player'] for r in isolated)
    identity=None
    if kind=='identity':
        target=next(r for r in doc['rows'] if r['human']['peak_time']==10.041667 and r['variant']=='restore_both_original_long')
        other=next(r for r in doc['rows'] if r['variant']=='same_bounds_different_human_target')
        a=all_inputs[(target['event_id'],target['variant'])];b=all_inputs[(other['event_id'],other['variant'])]
        assert all(np.array_equal(x,y) for x,y in zip(a,b))
        assert input_digest(a)==input_digest(b)==doc['identifiability']['input_digest']
        assert target['human']['stroke_type']!=other['human']['stroke_type']
        assert target['window']['peak_time']!=other['window']['peak_time']
        assert target['prediction']['scores']==other['prediction']['scores']
        identity={'arrays_byte_identical':True,'different_human_targets':True,'probability_difference':0.}
    return {'kind':kind,'numeric_replays':count,'max_probability_error':err,'intervention_groups':dict(groups),
            'checkpoint_sha256':doc['model_sha256'],'source_hashes_verified':True,'identifiability':identity}


def verify_inventory(folder):
    doc=json.loads((folder/'22条已确认同侧别拍清点.json').read_text());counts=Counter();details=[]
    for row in doc['rows']:
        labels=json.loads((ROOT.parent/'03_数据/击球标注'/row['job_id']/'annotations.json').read_text())['labels'].values()
        human=row['human']
        for variant,w in row['windows'].items():
            times={round(h['peak_time'],6) for h in labels if h.get('contact_confirmed') and not h.get('wrong_target') and h.get('play_context')!='between_rallies' and h['stroke_type'] not in ['unknown','no_stroke'] and h['player']==human['player'] and abs(h['peak_time']-human['peak_time'])>=.15 and w['start']<=h['peak_time']<w['end']}
            assert times=={round(h['peak_time'],6) for h in w['same_player']}
            counts[variant]+=bool(times)
    assert dict(counts)==doc['with_other_same_player']=={'frozen':4,'human_translated':5}
    return {'total':len(doc['rows']),'confirmed_other_same_player_windows':dict(counts),'interpretation':'Lower bound from sparse labels; no inferred hit treated as human ground truth.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--replay',action='store_true',required=True);p.add_argument('--write-report',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    classifier=public_classifier();neighbor=a.root/'真实邻拍局部对照';identity=neighbor/'前后段与目标条件诊断'
    checks=[verify_folder(a.root,classifier,'temporal'),verify_folder(neighbor,classifier,'neighbors'),verify_folder(identity,classifier,'identity')]
    result={'verified_at':datetime.now(timezone.utc).isoformat(),'checks':checks,'inventory':verify_inventory(neighbor),'total_numeric_replays':sum(c['numeric_replays'] for c in checks),'model_trained':False,'weights_changed':False,'labels_changed':False}
    if a.write_report:(a.root/'independent_replay_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
