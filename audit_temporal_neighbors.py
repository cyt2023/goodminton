"""Two fixed, visually reviewed opponent-neighbor window diagnostics.

RGB pose and ROI trajectory are existing diagnostic caches. No new labels,
training, model selection, or production result changes. Endpoints are fixed
before inference; shortest/longest bounds are sensitivity checks, not a search.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from analysis_jobs import DATA, ROOT, atomic_json, file_hash
from audit_temporal_contract import DEFAULT as TEMPORAL, pred
from specialist_bst import SpecialistBST
from specialist_model import BST_NAME
from stroke_review import read_labels, load_source, folder_for

JOB='J_16b4f26a07a14a59b76a77b8bb58dc78'
POSE=DATA/'击球标注/输入修复诊断/2026-10-13_姿态前端单因素诊断'/JOB/'rgb_dense_pose.json'
ROI=DATA/'击球标注/输入修复诊断/2026-10-12_固定视觉ROI重跑TrackNet/mapped_full_image_trajectory.json'
OUT=TEMPORAL/'真实邻拍局部对照'
CASES=[{'event_id':'76b88ba1502c9ddf3301c666990413c1','target_frame':158,
        'previous_central':144,'next_central':173,'previous_interval':[142,145],'next_interval':[172,175],
        'evidence':['far_neighbor_5.5000_6.1250.jpg','far_neighbor_6.9167_7.5417.jpg'],
        'audio_note':'Existing onset detector has no strong aligned impulse for these two far swings; do not treat sound as contact confirmation.'},
       {'event_id':'8766a907b9da2edb1cb4eeb3844d1c53','target_frame':241,
        'previous_central':220,'next_central':256,'previous_interval':[219,221],'next_interval':[255,258],
        'evidence':['far_neighbor_8.8333_9.4583.jpg','far_neighbor_10.3333_10.9583.jpg'],
        'audio_note':'Existing audio onsets 9.16s and 10.68s support far swing neighborhoods; source/lag not independently calibrated.'}]


def contacts_inside(window,human,doc):
    other={}
    for h in doc['labels'].values():
        if (not h.get('contact_confirmed') or h.get('wrong_target') or h.get('play_context')=='between_rallies'
                or h['stroke_type'] in ['unknown','no_stroke'] or abs(h['peak_time']-human['peak_time'])<.15):continue
        if window['start']<=h['peak_time']<window['end']:
            other[(h['player'],round(h['peak_time'],6))]={'player':h['player'],'peak_time':h['peak_time'],'stroke_type':h['stroke_type']}
    return {'same_player':[h for h in other.values() if h['player']==human['player']],
            'opponent':[h for h in other.values() if h['player']!=human['player']],
            'labels_incomplete':True}


def inventory(saved):
    rows=[];docs={}
    for row in saved['rows']:
        job=row['job_id']
        if job not in docs:
            report,_,_=load_source(job);docs[job]=read_labels(job,report['source_sha256'])
        human=row['human'];entry={'job_id':job,'event_id':row['event_id'],'human':human,'windows':{}}
        for name,w in [('frozen',row['frozen_window']),('human_translated',row['human_shifted_window'])]:
            entry['windows'][name]={'start':w['start'],'end':w['end'],**contacts_inside(w,human,docs[job])}
        rows.append(entry)
    return {'scope':'Lower bound using ALL existing confirmed labels, including fine strokes outside the 22-row classifier denominator; no newly inferred hits counted.',
            'total':len(rows),'with_other_same_player':{v:sum(bool(r['windows'][v]['same_player']) for r in rows) for v in ['frozen','human_translated']},'rows':rows}


def window(dense,case,previous,nxt):
    fps=float(dense['sample_fps']);times=np.asarray([f['source_time'] for f in dense['frames']]);center=case['target_frame']
    eps=int(fps//2)//2;limit=int(fps*3//2);a=max(previous,center-limit);end=min(nxt+eps,center+limit+eps)
    return {'start_frame':a,'end_frame':end-1,'frame_index':center,'peak_time':float(times[center]),
            'start':float(times[a]),'end':float(times[end-1]+1/fps),'player':'near',
            'author_extension_frames':eps,'author_extension_seconds':eps/fps,
            'basis':'visually_reviewed_opponent_neighbors_fixed_before_inference',
            'contact_frames_independently_confirmed_by_user':False}


def label_metrics(p,human):
    return {'type_matches':p['stroke_type']==human['stroke_type'],'actor_matches_under_unverified_mapping':p['mapped_actor']==human['player'],
            'joint_matches_under_unverified_mapping':p['stroke_type']==human['stroke_type'] and p['mapped_actor']==human['player']}


def replay(out):
    result=json.loads((out/'result.json').read_text());checkpoint=ROOT/'models/bst'/BST_NAME
    if file_hash(checkpoint)!=result['model_sha256'] or file_hash(out/'inputs.npz')!=result['inputs_sha256']:raise ValueError('Replay identity changed')
    from bst_public_data import public_classifier
    classifier=public_classifier();max_error=0
    with np.load(out/'inputs.npz',allow_pickle=False) as inputs:
        for i,(eid,pose,traj,wname) in enumerate(inputs['keys'].tolist()):
            row=next(r for r in result['rows'] if r['event_id']==eid and r['pose']==pose and r['trajectory']==traj and r['window_name']==wname);classifier.order=row['player_input_order']
            v=(inputs['joints_and_bones'][i],inputs['positions'][i],inputs['shuttle'][i],int(inputs['video_len'][i]));p=pred(classifier,v,row['human']);max_error=max(max_error,max(abs(a-b) for a,b in zip(p['scores'],row['prediction']['scores'])))
    if max_error>1e-5:raise ValueError('Replay prediction mismatch')
    print(json.dumps({'replayed':i+1,'max_probability_error':max_error}),flush=True)


def run(out):
    if (out/'result.json').is_file():replay(out);return
    out.mkdir(parents=True,exist_ok=True)
    protocol={'declared_at':datetime.now(timezone.utc).isoformat(),'cases':CASES,
        'windows':['original_long_support','central_author_neighbors','shortest_boundary_sensitivity','longest_boundary_sensitivity'],
        'factors':{'pose':['original_BRG_input','cached_correct_RGB_input'],'trajectory':['original','cached_fixed_visual_ROI']},
        'selection':'Central endpoints are fixed independently by frame montage review. Boundary extrema only, no best-scoring window selected.',
        'scope':'Two previously known smash errors, targeted mechanistic diagnosis; no generalization/overall accuracy claim.'}
    atomic_json(out/'protocol.json',protocol);shutil.copy2(Path(__file__),out/'executed_script.py')
    saved=json.loads((DATA/'击球标注/输入修复诊断/2026-10-12_配对输入修复/result.json').read_text());inv=inventory(saved);atomic_json(out/'22条已确认同侧别拍清点.json',inv)
    report,_,event_file=load_source(JOB);doc=read_labels(JOB,report['source_sha256'])
    paths={'original_pose':DATA/'专项分析'/JOB/'dense_pose.json','rgb_pose':POSE,
           'original_trajectory':DATA/'专项分析'/JOB/'trajectory.json','roi_trajectory':ROI,
           'annotations':folder_for(JOB)/'annotations.json','checkpoint':ROOT/'models/bst'/BST_NAME,'candidate_results':event_file}
    protected={name:{'path':str(path),'sha256':file_hash(path)} for name,path in paths.items()}
    dense=json.loads(paths['original_pose'].read_text());rgb=json.loads(POSE.read_text());original=json.loads(paths['original_trajectory'].read_text());roi=json.loads(ROI.read_text())
    for value in [rgb,original,roi]:
        if value['source_sha256']!=dense['source_sha256'] or len(value['frames'])!=len(dense['frames']):raise ValueError('Cached input source/frame mismatch')
    saved_rows={r['event_id']:r for r in saved['rows'] if r['job_id']==JOB};inputs=[];keys=[];rows=[];replay_error=0
    for pose_name,pose in [('original',dense),('rgb',rgb)]:
        for traj_name,traj in [('original',original),('roi',roi)]:
            classifier=SpecialistBST(pose,traj);classifier.order=['far','near']
            for case in CASES:
                old=saved_rows[case['event_id']];human=old['human'];human={k:human[k] for k in ['player','peak_time','stroke_type']}
                windows={'original_long_support':old['human_shifted_window'],
                    'central_author_neighbors':window(dense,case,case['previous_central'],case['next_central']),
                    'shortest_boundary_sensitivity':window(dense,case,case['previous_interval'][1],case['next_interval'][0]),
                    'longest_boundary_sensitivity':window(dense,case,case['previous_interval'][0],case['next_interval'][1])}
                for wname,w in windows.items():
                    values,quality=classifier.inputs(w);p=pred(classifier,values,human)
                    if pose_name=='original' and traj_name=='original' and wname=='original_long_support':
                        expected=old['variants']['human_center_same_duration']['scores'];replay_error=max(replay_error,max(abs(a-b) for a,b in zip(expected,p['scores'])))
                    rows.append({'event_id':case['event_id'],'human':human,'pose':pose_name,'trajectory':traj_name,'window_name':wname,
                        'window':w,'player_input_order':classifier.order.copy(),'prediction':p,'label_agreement':label_metrics(p,human),
                        'input_quality':quality,'known_other_contacts':contacts_inside(w,human,doc)})
                    inputs.append(values);keys.append((case['event_id'],pose_name,traj_name,wname))
    if replay_error>1e-5:raise ValueError('Long original input no longer matches saved diagnostic')
    for name,item in protected.items():
        if file_hash(Path(item['path']))!=item['sha256']:raise ValueError('Protected input changed: '+name)
    np.savez_compressed(out/'inputs.npz',keys=np.asarray(keys),joints_and_bones=np.stack([v[0] for v in inputs]),positions=np.stack([v[1] for v in inputs]),shuttle=np.stack([v[2] for v in inputs]),video_len=np.asarray([v[3] for v in inputs]))
    result={'version':'fixed-two-smash-neighbor-pose-trajectory-factorial-v1','scope':protocol['scope'],'protocol_sha256':file_hash(out/'protocol.json'),
        'model_sha256':protected['checkpoint']['sha256'],'inputs_sha256':file_hash(out/'inputs.npz'),'sources':protected,'sources_unchanged':True,
        'original_saved_replay_max_probability_error':replay_error,'rows':rows,'model_trained':False,'production_changed':False,'human_labels_changed':False,
        'all_22_known_same_player_hit_inventory':'22条已确认同侧别拍清点.json'}
    atomic_json(out/'result.json',result)
    for row in rows:
        if row['window_name'] in ['original_long_support','central_author_neighbors']:
            print(json.dumps({'target':row['human']['peak_time'],'pose':row['pose'],'trajectory':row['trajectory'],'window':row['window_name'],'range':[row['window']['start'],row['window']['end']],
                'prediction':row['prediction']['author_label'],'top_score':max(row['prediction']['scores']),'same_player_other':len(row['known_other_contacts']['same_player'])},ensure_ascii=False),flush=True)
    print('INVENTORY',inv['with_other_same_player'],'OUTPUT',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--replay',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    replay(a.output) if a.replay else run(a.output)
