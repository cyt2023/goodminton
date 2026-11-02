"""Three fixed first-serve window interventions on unchanged cached inputs.

Targets and next returns are existing human labels, visually checked in local
context. No labels/weights/production files are changed; no parameter search.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from analysis_jobs import DATA, ROOT, atomic_json, file_hash
from audit_serve_windows import JOB, OUT as VISUAL
from audit_temporal_contract import pred
from personal_learning import extractor
from specialist_model import BST_NAME
from stroke_review import folder_for, read_labels, load_source

SAVED=DATA/'击球标注/输入修复诊断/2026-10-12_配对输入修复'
OUT=VISUAL/'固定首拍窗口对照'
CASES=[{'event_id':'8474194d7e9251ae218fd0fc1808b89c','serve_time':33.3,'next_return_time':34.566667},
       {'event_id':'e6f6635dbdf40b292273608602c0cb1b','serve_time':47.5,'next_return_time':48.233333},
       {'event_id':'49d74f984571efe6bbc0482604a13d7b','serve_time':93.5,'next_return_time':94.333333}]


def replay(out):
    result=json.loads((out/'result.json').read_text());classifier,_=extractor(JOB);classifier.order=['far','near']
    assert file_hash(ROOT/'models/bst'/BST_NAME)==result['model_sha256']
    assert file_hash(out/'inputs.npz')==result['inputs_sha256']
    err=0.
    with np.load(out/'inputs.npz',allow_pickle=False) as data:
        for i,(eid,variant) in enumerate(data['keys'].tolist()):
            row=next(r for r in result['rows'] if r['event_id']==eid and r['variant']==variant)
            v=(data['joints_and_bones'][i],data['positions'][i],data['shuttle'][i],int(data['video_len'][i]));p=pred(classifier,v,row['human'])
            err=max(err,max(abs(a-b) for a,b in zip(p['scores'],row['prediction']['scores'])))
    assert err<=1e-6
    for path,digest in result['protected_sources_sha256'].items():assert file_hash(Path(path))==digest
    print(json.dumps({'replayed':i+1,'max_probability_error':err,'protected_sources_unchanged':True}),flush=True)


def run(out):
    if (out/'result.json').is_file():replay(out);return
    out.mkdir(parents=True,exist_ok=True)
    protocol={'declared_at':datetime.now(timezone.utc).isoformat(),'cases':CASES,
        'variants':['frozen_auto','human_translated_long','author_first_hit_fallback'],
        'window_rule':'Keep existing human target frame. Start 15 presentation frames before (~0.5s at observed 30Hz); endpoint next existing human-confirmed opponent return + author eps=7 frames, exclusive. No temporal resampling.',
        'cadence_basis':'Stored sequential PTS has median 1/30s spacing, despite nominal container 29.952fps; both cadence values recorded. End support last frame is +6/30s and exclusive end +7/30s, the integer author approximation to 0.25s.',
        'visual_selection':'All three show stationary preparation, low-hand serve and next labelled return in the same rally. This is researcher visual review of existing labels, not new user-confirmed boundary GT.',
        'scope':'Three selected, previously inspected serve cases. Diagnostic effects only; no new-video accuracy claim or best-window selection.'}
    atomic_json(out/'protocol.json',protocol);shutil.copy2(Path(__file__),out/'executed_script.py')
    report,_,event_file=load_source(JOB);doc=read_labels(JOB,report['source_sha256']);classifier,dense=extractor(JOB);classifier.order=['far','near'];times=np.asarray([f['source_time'] for f in dense['frames']]);dt=float(np.median(np.diff(times)));fps=round(1/dt);assert fps==30
    old=json.loads((SAVED/'result.json').read_text());lookup_rows={r['event_id']:r for r in old['rows'] if r['job_id']==JOB}
    protect=[folder_for(JOB)/'annotations.json',folder_for(JOB)/'public_only_predictions.json',event_file,DATA/'专项分析'/JOB/'dense_pose.json',DATA/'专项分析'/JOB/'trajectory.json',ROOT/'models/bst'/BST_NAME]
    hashes={str(p):file_hash(p) for p in protect};keys=[];inputs=[];rows=[];err=0.
    with np.load(SAVED/'inputs.npz',allow_pickle=False) as cached:
        ids={tuple(k):i for i,k in enumerate(cached['keys'].tolist())}
        for spec in CASES:
            eid=spec['event_id'];source=lookup_rows[eid];h=source['human'];human={k:h[k] for k in ['player','peak_time','stroke_type']};assert human['stroke_type']=='serve_short'
            next_h=next(x for x in doc['labels'].values() if abs(x['peak_time']-spec['next_return_time'])<1e-5 and x['player']!=human['player']);assert next_h['contact_confirmed'] and not next_h.get('wrong_target')
            c=int(np.argmin(abs(times-human['peak_time'])));nxt=int(np.argmin(abs(times-next_h['peak_time'])));a=c-fps//2;end=nxt+(fps//2)//2
            w={'player':human['player'],'peak_time':human['peak_time'],'frame_index':c,'start_frame':a,'end_frame':end-1,'start':float(times[a]),'end':float(times[end-1]+dt),'last_frame_time':float(times[end-1]),'basis':'first_hit_halfsecond_fallback_plus_confirmed_next_return_author_integer_extension','next_human_confirmed_opponent':{k:next_h[k] for k in ['event_id','player','peak_time','stroke_type']},'reference_presentation_fps':fps,'nominal_container_fps':dense['sample_fps'],'extension_frames':7,'rally_first_verified_by_user':False}
            for name,saved_name,win in [('frozen_auto','frozen_replay',source['frozen_window']),('human_translated_long','human_center_same_duration',source['human_shifted_window']),('author_first_hit_fallback',None,w)]:
                if saved_name:
                    i=ids[(JOB,eid,saved_name)];v=(cached['joints_and_bones'][i].copy(),cached['positions'][i].copy(),cached['shuttle'][i].copy(),int(cached['video_len'][i]));quality=classifier.inputs(win)[1]
                else:v,quality=classifier.inputs(win)
                p=pred(classifier,v,human)
                if saved_name:err=max(err,max(abs(a-b) for a,b in zip(p['scores'],source['variants'][saved_name]['scores'])))
                near_frames=list(range(max(0,c-3),min(len(times),c+4)))
                ball=[{'frame':i,'time':float(times[i]),'raw_visible':classifier.trajectory['frames'][i]['raw_visible'],'raw_xy':classifier.trajectory['frames'][i]['raw_xy'],'inpaint_requested':classifier.trajectory['frames'][i]['inpaint_requested']} for i in near_frames]
                pose=[{'frame':i,'target_present':dense['frames'][i]['players'].get(human['player']) is not None,'opponent_present':dense['frames'][i]['players'].get(next_h['player']) is not None} for i in near_frames]
                rows.append({'event_id':eid,'variant':name,'human':human,'window':win,'prediction':p,'input_quality':quality,'contact_neighborhood_raw_trajectory':ball,'contact_neighborhood_pose_presence':pose,
                    'type_label_match':p['stroke_type']==human['stroke_type'],'actor_agreement_under_unverified_mapping':p['mapped_actor']==human['player'],'joint_label_match_under_unverified_mapping':p['stroke_type']==human['stroke_type'] and p['mapped_actor']==human['player']})
                keys.append((eid,name));inputs.append(v)
    assert err<=1e-6
    assert all(file_hash(Path(p))==digest for p,digest in hashes.items())
    np.savez_compressed(out/'inputs.npz',keys=np.asarray(keys),joints_and_bones=np.stack([v[0] for v in inputs]),positions=np.stack([v[1] for v in inputs]),shuttle=np.stack([v[2] for v in inputs]),video_len=np.asarray([v[3] for v in inputs]))
    summary={name:{'total':sum(r['variant']==name for r in rows),'type_label_matches':sum(r['type_label_match'] for r in rows if r['variant']==name),'joint_label_matches_under_unverified_mapping':sum(r['joint_label_match_under_unverified_mapping'] for r in rows if r['variant']==name)} for name in protocol['variants']}
    result={'version':'fixed-three-first-serve-support-intervention-v1','created_at':datetime.now(timezone.utc).isoformat(),'scope':protocol['scope'],'rows':rows,'summary':summary,'protocol_sha256':file_hash(out/'protocol.json'),'inputs_sha256':file_hash(out/'inputs.npz'),'model_sha256':file_hash(ROOT/'models/bst'/BST_NAME),'protected_sources_sha256':hashes,'saved_input_replay_max_error':err,'model_trained':False,'production_changed':False,'labels_changed':False}
    atomic_json(out/'result.json',result)
    for r in rows:print(json.dumps({'serve':r['human']['peak_time'],'variant':r['variant'],'range':[r['window']['start'],r['window']['end']],'prediction':r['prediction']['author_label'],'score':max(r['prediction']['scores']),'paired':r['input_quality']['paired_player_fraction'],'raw_ball':r['input_quality']['raw_shuttle_fraction']},ensure_ascii=False),flush=True)
    print('SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--replay',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    replay(a.output) if a.replay else run(a.output)
