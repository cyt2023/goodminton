"""Fixed stroke: replace only shuttle coordinates after frozen native-image review."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from analysis_jobs import atomic_json, file_hash
from input_quality import InputQuality
from specialist_bst import SpecialistBST
from specialist_model import ROOT, BST_NAME

DATA=ROOT.parent/'03_数据'
BASE=DATA/'击球标注/根因诊断/2026-10-18_普通视角配对诊断'
REVIEW=BASE/'native_ball_researcher_25p3'
OUT=REVIEW/'classification_control'
JOB='J_e23c1dd9297d4bfca7c91d1d95065b68'
DENSE=DATA/'专项分析'/JOB/'dense_pose.json'
BALL=DENSE.with_name('trajectory.json')
RUN=DATA/'击球标注/直接BST微调/运行记录/582817bff5454825b841d1f150c82028'
EVENT='3d5e6020fbc2aa558b6a5914a9826a54'


def main():
    if (OUT/'result.json').exists():raise FileExistsError('Preserve frozen classification control')
    OUT.mkdir(parents=True,exist_ok=True)
    freeze=json.loads((REVIEW/'review_freeze.json').read_text())
    assert file_hash(REVIEW/'annotations.json')==freeze['key_file_sha256']['annotations.json']
    annotations=json.loads((REVIEW/'annotations.json').read_text())
    rows=annotations['rows'];assert [r['frame_index'] for r in rows]==list(range(713,793))
    assert sum(r['visibility']=='visible' for r in rows)==79
    assert [r['frame_index'] for r in rows if r['x_native'] is None]==[783]
    paths=[REVIEW/'review_freeze.json',REVIEW/'annotations.json',REVIEW/'protocol_freeze.json',
           DENSE,BALL,BASE/'actual_neighbor_window_grid/visual_review_frozen.json',
           RUN/'personal_predictions.json',RUN/'personal_predictions.npz',ROOT/'models/bst'/BST_NAME,
           ROOT/'specialist_bst.py',Path(__file__)]
    hashes={str(p):file_hash(p) for p in paths}
    for r in rows:
        path=Path(r['source_png']);assert file_hash(path)==r['source_png_sha256']
    windows=[{'name':'original80','lo':713,'hi':792}]
    windows.extend({'name':f'neighbor_{lo}_{end-1}','lo':lo,'hi':end-1} for lo in [713,714] for end in [789,790,791])
    protocol={'declared_at':datetime.now(timezone.utc).isoformat(),'input_sha256':hashes,
        'review_frozen_before_any_new_classifier_run':True,
        'coordinate_scope':'All80 original frames713..792 reviewed independently without original model points/mask;79 visible estimated native coordinates,783 ambiguous and zero/missing. Not perfect coordinate ground truth and not user labels.',
        'conditions':['original_ball','original_with_same_missing783','researcher_nominal','researcher_x_plus_radius','researcher_x_minus_radius','researcher_y_plus_radius','researcher_y_minus_radius'],
        'uncertainty_sensitivity':'Four globally consistent axial directions use each frame own conservative blur/manual radius. Not independent per-frame extremes or an exhaustive uncertainty proof. No clipping; all coordinates must remain in image.',
        'windows':windows,'boundary_scope':'Six fully reviewed-support neighboring-window variants, plus original80. Excludes the3 start712 variants because their first point lies outside frozen ball-review domain; no extrapolated or borrowed true point.',
        'fixed':'Original dense poses, original whole-video person order, original BST-0, same P and N for every ball condition within each fixed window. OnlyS may change. No training or activation.',
        'checks':'Raw source PNG hashes; original baseline logits match saved run; original restored at end. Frame783 same-missing control separates missingness from changed79 positions.',
        'scope':'One preselected ordinary developer failure. No claim of new-video accuracy.'}
    atomic_json(OUT/'protocol.json',protocol)
    dense=json.loads(DENSE.read_text());ball=json.loads(BALL.read_text())
    assert annotations['source_sha256']==dense['source_sha256']
    variants={'original_ball':ball}
    common=copy.deepcopy(ball)
    b=common['frames'][783];b.update(raw_xy=None,rectified_xy=None,raw_visible=False,inpaint_requested=False)
    variants['original_with_same_missing783']=common
    for name,dx,dy in [('researcher_nominal',0,0),('researcher_x_plus_radius',1,0),('researcher_x_minus_radius',-1,0),('researcher_y_plus_radius',0,1),('researcher_y_minus_radius',0,-1)]:
        trajectory=copy.deepcopy(ball)
        for r in rows:
            i=r['frame_index'];frame=trajectory['frames'][i]
            if r['visibility']=='visible':
                radius=r['uncertainty_radius_px'];xy=[r['x_native']+dx*radius,r['y_native']+dy*radius]
                assert 0<=xy[0]<dense['source_width'] and 0<=xy[1]<dense['source_height']
                frame.update(raw_xy=xy,rectified_xy=xy,raw_visible=True,inpaint_requested=False)
            else:frame.update(raw_xy=None,rectified_xy=None,raw_visible=False,inpaint_requested=False)
        variants[name]=trajectory
    records=json.loads((RUN/'personal_predictions.json').read_text())['baseline']
    ri=next(i for i,r in enumerate(records) if r['event_id']==EVENT);record=records[ri]
    cache=np.load(RUN/'personal_predictions.npz')
    classifier=SpecialistBST(dense,ball);torch.set_num_threads(2)
    target=int(record['target_class_id']);actor=record['target_class_label'].split('_',1)[0]
    allowed=np.asarray([0]+[i for i,c in enumerate(classifier.classes) if c.startswith(actor+'_')]);assert len(allowed)==13
    saved={};results=[];original_logits=None
    for window in windows:
        event=dict(record['model_window']);event.update(start_frame=window['lo'],end_frame=window['hi'],
            start=dense['frames'][window['lo']]['source_time'],end=dense['frames'][window['hi']+1]['source_time'])
        reference=None
        for name,trajectory in variants.items():
            classifier.trajectory=trajectory;classifier.coordinate_quality=InputQuality(dense,trajectory)
            (j,p,s,n),quality=classifier.inputs(event)
            if reference is None:reference=(j.copy(),p.copy(),n)
            else:assert np.array_equal(j,reference[0]) and np.array_equal(p,reference[1]) and n==reference[2]
            with torch.inference_mode():scores=classifier.model(torch.from_numpy(j)[None],torch.from_numpy(s)[None],torch.tensor([n])).numpy()[0]
            if window['name']=='original80' and name=='original_ball':
                baseline_error=float(np.max(abs(scores-cache['baseline_logits'][ri])));assert baseline_error<1e-4
                original_logits=scores.copy()
            pred=int(scores.argmax());pc=int(allowed[np.argmax(scores[allowed])])
            key=window['name']+'__'+name;saved[key]=scores
            np.savez_compressed(OUT/(key+'_inputs.npz'),j=j,p=p,s=s,n=np.asarray(n,dtype='int64'))
            results.append({'window':window,'ball':name,'true_label':record['target_class_label'],
                'prediction':classifier.classes[pred],'joint_correct':pred==target,
                'actor13_prediction':classifier.classes[pc],'actor13_correct':pc==target,
                'margin_true_vs_other24':float(scores[target]-scores[np.arange(25)!=target].max()),
                'pair_fraction':quality['paired_player_fraction'],'raw_fraction':quality['raw_shuttle_fraction'],'n':n})
    classifier.trajectory=ball;classifier.coordinate_quality=InputQuality(dense,ball)
    (j,p,s,n),_=classifier.inputs(record['model_window'])
    with torch.inference_mode():restored=classifier.model(torch.from_numpy(j)[None],torch.from_numpy(s)[None],torch.tensor([n])).numpy()[0]
    assert np.array_equal(restored,original_logits)
    assert all(file_hash(Path(p))==h for p,h in hashes.items())
    np.savez_compressed(OUT/'logits.npz',**saved)
    summary={name:{'windows':len(windows),'joint_correct':sum(r['joint_correct'] for r in results if r['ball']==name),
        'actor13_correct':sum(r['actor13_correct'] for r in results if r['ball']==name),
        'predictions':sorted({r['prediction'] for r in results if r['ball']==name})} for name in variants}
    atomic_json(OUT/'result.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':file_hash(OUT/'protocol.json'),'rows':results,'summary':summary,
        'baseline_logit_max_difference':baseline_error,'original_restore_bit_exact':True,
        'only_S_changes_within_each_window':True,'production_or_user_labels_or_weights_changed':False,'scope':protocol['scope']})
    print(json.dumps({'summary':summary,'original80':[r for r in results if r['window']['name']=='original80']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
