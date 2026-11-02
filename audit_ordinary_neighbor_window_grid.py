"""Freeze source-reviewed uncertainty intervals, then evaluate every boundary pair."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analysis_jobs import atomic_json, file_hash
from input_quality import InputQuality
from specialist_bst import SpecialistBST
from specialist_model import ROOT, BST_NAME

DATA=ROOT.parent/'03_数据'
BASE=DATA/'击球标注/根因诊断/2026-10-18_普通视角配对诊断'
OUT=BASE/'actual_neighbor_window_grid'
JOB='J_e23c1dd9297d4bfca7c91d1d95065b68'
DENSE=DATA/'专项分析'/JOB/'dense_pose.json'
BALL=DENSE.with_name('trajectory.json')
RUN=DATA/'击球标注/直接BST微调/运行记录/582817bff5454825b841d1f150c82028'


def main():
    if (OUT/'result.json').exists():raise FileExistsError('Preserve completed grid')
    OUT.mkdir(parents=True,exist_ok=True)
    dense=json.loads(DENSE.read_text())
    times=np.asarray([f['source_time'] for f in dense['frames']])
    source=DATA/'个人录像/98f0e3817a62451b994499c0c5fe1a3a.mp4'
    paths=[source,DENSE,BALL,RUN/'personal_predictions.json',RUN/'personal_predictions.npz',
        BASE/'pose_missing_1p0333/source_proof.json',
        BASE/'pose_missing_1p0333/contact_boundaries/protocol.json',
        BASE/'pose_missing_1p0333/contact_boundaries/visual_indices.json',
        BASE/'pose_missing_1p0333/filled_dense_pose.json',
        BASE/'pose_missing_1p0333/filled_dense_pose_far_and_near.json',
        BASE/'contact_boundaries_25p3/source_proof.json',
        ROOT/'models/bst'/BST_NAME, ROOT/'specialist_bst.py',
        ROOT/'vendor/BST/ShuttleSet/gen_my_dataset.py',Path(__file__)]
    for folder,pattern in [(BASE/'pose_missing_1p0333/contact_boundaries','*.png'),(BASE/'contact_boundaries_25p3','*everyframe.png')]:
        paths.extend(sorted(folder.glob(pattern)))
    hashes={str(p):file_hash(p) for p in paths}
    review={
        'frozen_at':datetime.now(timezone.utc).isoformat(),
        'reviewer':'root researcher; source strips prepared independently for first case, final visual interpretation by root',
        'blinding':'Root knows historical classes/outcomes. No new window logits have been computed. Intervals follow visible motion; every integer boundary combination is evaluated without score selection.',
        'source_sha256':file_hash(source),'source_pts_basis':'Dense actual source_time and exact source-reader proofs, not index/nominal fps.',
        'first_serve':{
            'event_id':'97b39d14d2435ac65ccb3023a764f49c',
            'target_interval_inclusive':[30,32],'next_opponent_interval_inclusive':[61,64],
            'rally_edge':'Visible initial0..29 frames show near server preparing and initiating serve; no earlier same-rally contact is visible. No claim about footage preceding source start.',
            'observations':'Near racket accelerates f30..31, white outgoing streak by f32. Far raises racket f54..60, overhead extension/followthrough f61..64. Blur prevents asserting a single physical contact frame.',
        },
        'drop_25p3':{
            'event_id':'3d5e6020fbc2aa558b6a5914a9826a54',
            'previous_opponent_interval_inclusive':[710,712],
            'target_interval_inclusive':[757,759],
            'next_opponent_interval_inclusive':[782,784],
            'observations':'Near upward swing and departing white shuttle f710..712; far overhead swing f757..759; near net lunge with ball reaching racket f782..784 then outgoing rise. Whole context674..824 and fine strips examined; no intervening target-rally swing seen. Far actor is small and contact is blurred; these are researcher uncertainty bounds, not exact coordinate truth.',
        },
        'limits':'One root visual interpretation, not an additional independent contact annotation and not user training truth. Does not repair remaining ball/joint coordinates.'}
    for name in ['first_serve','drop_25p3']:
        r=review[name]
        for key,value in list(r.items()):
            if key.endswith('_inclusive'):
                r[key.replace('_inclusive','_source_times')]=[float(times[i]) for i in value]
    atomic_json(OUT/'visual_review_frozen.json',review)
    records=json.loads((RUN/'personal_predictions.json').read_text())['baseline']
    cases={r['event_id']: (i,r) for i,r in enumerate(records) if r['event_id'] in [review['first_serve']['event_id'],review['drop_25p3']['event_id']]}
    conditions=[]
    for case in ['first_serve','drop_25p3']:
        r=review[case];i,row=cases[r['event_id']];center=row['model_window']['frame_index']
        assert review[case]['target_interval_inclusive'][0]<=center<=review[case]['target_interval_inclusive'][1]
        poses=['original','far_only','far_and_near'] if case=='first_serve' else ['original']
        starts=range(15,18) if case=='first_serve' else [713]
        # Author end_f is exclusive: next-contact +7 frames at30fps.
        ends=range(68,72) if case=='first_serve' else range(789,792)
        for pose in poses:
            conditions.append({'case':case,'pose':pose,'kind':'original_window','lo':row['model_window']['start_frame'],'hi':row['model_window']['end_frame']})
            for lo in starts:
                for end in ends:
                    conditions.append({'case':case,'pose':pose,'kind':'reviewed_neighbor_grid','lo':lo,'hi':end-1})
    protocol={'declared_at':datetime.now(timezone.utc).isoformat(),'input_sha256':hashes,
        'visual_review_sha256':file_hash(OUT/'visual_review_frozen.json'),
        'author_policy':'between_2_hits_with_max_limits; fps30 convention t15/eps7/limit45. First serve start=target-15; ordinary start=max(previous,center-45); endexclusive=min(next+7,center+52). This maps author frame policy onto native frame indices, avoiding timestamp-based duplication. No MoviePy rounded-duration extra frame.',
        'center_policy':'Keep existing user center31 or758. First-serve uncertainty changes first-start across15..17; mid-rally target uncertainty does not alter capped start713 under any reviewed previous710..712. End enumerates every reviewed next hit. Fixed nominal30 convention, source timings still recorded from dense actualPTS.',
        'conditions':conditions,'model':'Original released BST-0 only, frozen whole-video person order and current raw ball unchanged. No training.',
        'evaluations':['joint25_argmax','human_actor12_plus_unknown_argmax','margin_true_vs_other24'],
        'scope':'Two preselected ordinary developer failures, all uncertainty combinations reported. Restored poses remain estimates, source-reviewed bounds remain uncertain. No general accuracy or production activation.'}
    atomic_json(OUT/'protocol.json',protocol)
    ball=json.loads(BALL.read_text());classifier=SpecialistBST(dense,ball)
    order=classifier.order.copy();torch.set_num_threads(2)
    variants={'original':dense,
              'far_only':json.loads((BASE/'pose_missing_1p0333/filled_dense_pose.json').read_text()),
              'far_and_near':json.loads((BASE/'pose_missing_1p0333/filled_dense_pose_far_and_near.json').read_text())}
    cache=np.load(RUN/'personal_predictions.npz')
    results=[];saved={};baseline_errors={}
    for k,condition in enumerate(conditions):
        case=condition['case'];i,record=cases[review[case]['event_id']]
        event=dict(record['model_window']);lo,hi=condition['lo'],condition['hi']
        event.update(start_frame=lo,end_frame=hi,start=float(times[lo]),end=float(times[hi+1]))
        classifier.dense=variants[condition['pose']];classifier.order=order
        classifier.coordinate_quality=InputQuality(classifier.dense,ball)
        (j,p,s,n),quality=classifier.inputs(event)
        with torch.inference_mode():
            scores=classifier.model(torch.from_numpy(j)[None],torch.from_numpy(s)[None],torch.tensor([n])).numpy()[0]
        target=record['target_class_id'];assert classifier.classes[target]==record['target_class_label']
        actor=record['target_class_label'].split('_',1)[0]
        allowed=np.asarray([0]+[a for a,c in enumerate(classifier.classes) if c.startswith(actor+'_')])
        assert len(allowed)==13
        pred=int(scores.argmax());pc=int(allowed[np.argmax(scores[allowed])])
        name=f'{k:02d}_{case}_{condition["pose"]}_{lo}_{hi}'
        np.savez_compressed(OUT/(name+'_inputs.npz'),j=j,p=p,s=s,n=np.asarray(n,dtype='int64'))
        saved[name]=scores
        if condition['kind']=='original_window' and condition['pose']=='original':
            error=float(np.max(abs(scores-cache['baseline_logits'][i])));assert error<1e-4
            baseline_errors[case]=error
        others=np.arange(25)!=target
        results.append({**condition,'key':name,'event_id':record['event_id'],'center':event['frame_index'],
            'source_start':float(times[lo]),'source_end_exclusive':float(times[hi+1]),
            'true_label':record['target_class_label'],'prediction':classifier.classes[pred],'joint_correct':pred==target,
            'actor13_prediction':classifier.classes[pc],'actor13_correct':pc==target,
            'true_label_logit_margin25':float(scores[target]-scores[others].max()),
            'n':n,'pair_fraction':quality['paired_player_fraction'],'raw_ball_fraction':quality['raw_shuttle_fraction'],
            'input_sha256':hashlib.sha256(j.tobytes()+s.tobytes()+np.asarray(n,dtype='int64').tobytes()).hexdigest()})
    assert all(file_hash(Path(p))==h for p,h in hashes.items())
    np.savez_compressed(OUT/'logits.npz',**saved)
    summaries={}
    for case in ['first_serve','drop_25p3']:
        summaries[case]={}
        for pose in sorted({r['pose'] for r in results if r['case']==case}):
            rows=[r for r in results if r['case']==case and r['pose']==pose and r['kind']=='reviewed_neighbor_grid']
            summaries[case][pose]={'windows':len(rows),'joint_correct':sum(r['joint_correct'] for r in rows),
                'actor13_correct':sum(r['actor13_correct'] for r in rows),
                'joint_predictions':{label:sum(r['prediction']==label for r in rows) for label in sorted({r['prediction'] for r in rows})},
                'actor13_predictions':{label:sum(r['actor13_prediction']==label for r in rows) for label in sorted({r['actor13_prediction'] for r in rows})}}
    atomic_json(OUT/'result.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':file_hash(OUT/'protocol.json'),'rows':results,'summary':summaries,
        'baseline_logit_max_errors':baseline_errors,'source_order':order,
        'no_production_or_labels_or_weights_changed':True,'scope':protocol['scope']})
    print(json.dumps({'summary':summaries,'baseline_errors':baseline_errors},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
