"""Fixed two-factor trajectory diagnostic; review/window frozen beforehand."""
from datetime import datetime, timezone
import copy
import json
from pathlib import Path

import numpy as np
import torch

from analysis_jobs import DATA, atomic_json, file_hash
from audit_visible_shuttle_netcross import margin
from diagnose_input_repairs import prediction
from specialist_bst import SpecialistBST

OUT=DATA/'击球标注/输入修复诊断/2026-10-13_勾对角目标球核查'
JOB='J_fbda35b385f44327a097b2c105684898'


def actual_xy(trajectory,i):
    f=trajectory['frames'][i]
    return (f['rectified_xy'] if f['inpaint_requested'] else f['raw_xy']) or [0,0]


def main():
    torch.set_num_threads(1)
    protocol=json.loads((OUT/'trajectory_factorial_protocol.json').read_text())
    review_file=OUT/'visible_coordinate_review.json'
    if file_hash(review_file)!=protocol['review_sha256']:raise ValueError('Frozen review changed')
    review=json.loads(review_file.read_text());eid=protocol['event_id']
    previous=json.loads((OUT/'result.json').read_text())
    old=next(r for r in previous['rows'] if r['event_id']==eid);y=old['human'];window=old['isolated_window']
    if [window['start_frame'],window['end_frame']]!=protocol['fixed_window']:raise ValueError('Frozen window changed')
    root=DATA/'专项分析'/JOB
    dense=json.loads((root/'dense_pose.json').read_text());original=json.loads((root/'trajectory.json').read_text())
    points={r['frame_index']:r['reviewed_xy'] for r in review['rows'] if r['event_id']==eid}
    trajectories={}
    for name,replacement,zero in [('A_original',False,False),('B_replace_17',True,False),('C_zero_5',False,True),('D_replace_17_and_zero_5',True,True)]:
        t=copy.deepcopy(original)
        if replacement:
            for i in protocol['replacement_frames']:
                f=t['frames'][i];key='rectified_xy' if f['inpaint_requested'] else 'raw_xy'
                f[key]=points[i]
        if zero:
            for i in protocol['zero_frames']:
                f=t['frames'][i];key='rectified_xy' if f['inpaint_requested'] else 'raw_xy'
                f[key]=[0,0]
        trajectories[name]=t
    a,b=protocol['fixed_window']
    prior_visible=json.loads((OUT/(eid+'_reviewed_trajectory.json')).read_text())
    expected=np.asarray([actual_xy(prior_visible,i) for i in range(a,b+1)])
    changed=np.asarray([actual_xy(trajectories['D_replace_17_and_zero_5'],i) for i in range(a,b+1)])
    equality=float(np.max(abs(expected-changed)))
    if equality>1e-8:raise ValueError('Factorial D differs from original reviewed coordinates')
    result={}
    actual={}
    for name,t in trajectories.items():
        clf=SpecialistBST(dense,t);clf.order=list(old['variants']['original_frozen_window']['player_order'])
        values,_=clf.inputs(window);p=prediction(clf,values,y['player'],y['player'])
        p['known_target_logit_margin']=margin(clf,p,y['player'],y['stroke_type'])
        result[name]=p
        actual[name]=[actual_xy(t,i) for i in range(a,b+1)]
    expected_pairs=[('A_original','isolated_window_original_trajectory'),('D_replace_17_and_zero_5','isolated_window_visible_review')]
    replay={name:float(np.max(abs(np.asarray(result[name]['scores'])-old['variants'][prior]['scores']))) for name,prior in expected_pairs}
    if max(replay.values())>1e-5:raise ValueError('Factorial endpoint replay differs')
    final={'created_at':datetime.now(timezone.utc).isoformat(),'protocol':protocol,'window':window,'human':y,
        'results':result,'model_actual_full_image_coordinates':actual,'D_matches_previous_review_coordinate_max_error':equality,
        'previous_endpoint_replay_max_probability_errors':replay,'protocol_sha256':file_hash(OUT/'trajectory_factorial_protocol.json'),
        'script_sha256':file_hash(Path(__file__)),'production_changed':False,'labels_changed':False,'model_trained':False,
        'scope':'One known development net_cross error; researcher-reviewed pixels, not user coordinate truth; no independent performance claim.'}
    atomic_json(OUT/'trajectory_factorial_result.json',final)
    print(json.dumps({k:{'author_label':v['author_label'],'code':v['code'],'actor':v['mapped_actor'],'known_target_logit_margin':v['known_target_logit_margin']} for k,v in result.items()},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
