"""Predeclared two-case researcher-visible shuttle counterfactual.

Coordinate review and boundaries were frozen before classifier inference.
No point interpolation or per-case class-driven selection is allowed. Reviewed
points are researcher image readings, not user ground truth or training labels.
"""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from analysis_jobs import DATA, ROOT, atomic_json, file_hash, registered_video
from diagnose_input_repairs import prediction, metrics
from specialist_bst import SpecialistBST
from specialist_model import BST_NAME
from stroke_taxonomy import AUTHOR_MAP

JOB='J_fbda35b385f44327a097b2c105684898'
OUT=DATA/'击球标注/输入修复诊断/2026-10-13_勾对角目标球核查'
PARENT=DATA/'击球标注/输入修复诊断/2026-10-12_配对输入修复/result.json'
REVIEW_EXPECTED_SHA='ec40d02c5097f513fb0db8b616a037fad54c4fca8d3ef00c248729f55afdfa5f'


def margin(clf,p,target,code):
    prefix='Top_' if target==clf.order[0] else 'Bottom_'
    ids=[i for i,label in enumerate(clf.classes) if label.startswith(prefix)]
    truth=next(i for i in ids if AUTHOR_MAP[clf.classes[i].split('_',1)[-1]]==code)
    log=np.log(np.maximum(p['scores'],1e-30))
    return float(log[truth]-max(log[i] for i in ids if i!=truth))


def plot(dense,original,reviewed,window,frame,source,path):
    cap=cv2.VideoCapture(str(source));cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();cap.release()
    if not ok:raise ValueError('Missing plot background')
    panels=[]
    for value,label in [(original,'Original raw proposals'),(reviewed,'Researcher visible target points; no interpolation; uncertain=0')]:
        panel=image.copy()
        for f in value['frames'][window['start_frame']:window['end_frame']+1]:
            if not f['raw_visible']:continue
            xy=tuple(map(round,f['raw_xy']));t=(f['frame_index']-window['start_frame'])/max(1,window['end_frame']-window['start_frame'])
            color=tuple(map(int,cv2.applyColorMap(np.asarray([[round(t*255)]],dtype='uint8'),cv2.COLORMAP_VIRIDIS)[0,0]))
            cv2.circle(panel,xy,4,(255,255,255),-1);cv2.circle(panel,xy,3,color,-1)
        header=np.full((46,panel.shape[1],3),250,dtype='uint8')
        cv2.putText(header,label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
        cv2.putText(header,f'Same isolated window f{window["start_frame"]}-{window["end_frame"]}; human target f{frame}; not user coordinate truth',(8,40),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
        panels.append(np.vstack([header,panel]))
    cv2.imwrite(str(path),np.vstack(panels))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--author-last',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);started=time.perf_counter()
    review_path=OUT/'visible_coordinate_review.json'
    if file_hash(review_path)!=REVIEW_EXPECTED_SHA:
        raise ValueError('Frozen coordinate review changed; no unregistered revision accepted')
    protocol_path=OUT/('author_last_hit_protocol.json' if args.author_last else 'predeclared_protocol.json')
    review=json.loads(review_path.read_text());proto=json.loads(protocol_path.read_text())
    parent=json.loads(PARENT.read_text());root=DATA/'专项分析'/JOB
    dense=json.loads((root/'dense_pose.json').read_text());original=json.loads((root/'trajectory.json').read_text())
    report=json.loads((root/'report.json').read_text());source=registered_video(report['source_video_id'],allow_reference=True)['path']
    checkpoint=ROOT/'models/bst'/BST_NAME
    protected=[root/'dense_pose.json',root/'trajectory.json',DATA/'击球标注'/JOB/'annotations.json',DATA/'击球标注'/JOB/'public_only_predictions.json',checkpoint,review_path,protocol_path]
    hashes={str(p):file_hash(p) for p in protected}
    rows=[]
    for case in proto['cases']:
        eid=case['event_id'];old=next(r for r in parent['rows'] if r['event_id']==eid);y=old['human']
        a,b=case['window'];window={**old['frozen_window'],'start_frame':a,'end_frame':b,
            'frame_index':case['target_frame'],'peak_time':case['target_frame']/dense['sample_fps'],
            'start':dense['frames'][a]['source_time'],'end':dense['frames'][b]['source_time']+1/dense['sample_fps'],
            'basis':'predeclared_researcher_visible_opponent_contact_or_terminal_rally_bounds_not_user_GT'}
        raw=json.loads(json.dumps(original));visible=json.loads(json.dumps(original))
        points={r['frame_index']:r for r in review['rows'] if r['event_id']==eid and a<=r['frame_index']<=b}
        for i in range(a,b+1):
            f=raw['frames'][i];f['inpaint_requested']=False;f['rectified_xy']=f['raw_xy'] if f['raw_visible'] else None
            point=points[i]['reviewed_xy'];f=visible['frames'][i]
            f.update(raw_xy=point,rectified_xy=point,raw_visible=point is not None,inpaint_requested=False,
                coordinate_source='researcher_visible_source_point_not_user_truth' if point else 'researcher_uncertain_zero')
        classifiers={name:SpecialistBST(dense,t) for name,t in [
            ('original_frozen_window',original),('isolated_window_original_trajectory',original),
            ('isolated_window_raw_only',raw),('isolated_window_visible_review',visible)]}
        row={'job_id':JOB,'event_id':eid,'human':y,'isolated_window':window,'variants':{},'known_target_logit_margins':{},
            'visible_point_count':sum(r['reviewed_xy'] is not None for r in points.values()),'total_frames':b-a+1,
            'uncertain_frames':[i for i,r in points.items() if r['reviewed_xy'] is None]}
        for name,clf in classifiers.items():
            clf.order=list(old['variants']['frozen_replay']['player_order'])
            w=old['frozen_window'] if name=='original_frozen_window' else window
            v,q=clf.inputs(w);p=prediction(clf,v,y['player'],old['automatic_candidate_player'])
            if name=='original_frozen_window':
                error=float(np.max(abs(np.asarray(p['scores'])-old['variants']['frozen_replay']['scores'])))
                if error>1e-5:raise ValueError('Frozen original replay mismatch')
                row['frozen_replay_max_score_error']=error
            row['variants'][name]=p;row['known_target_logit_margins'][name]=margin(clf,p,y['player'],y['stroke_type'])
        prefix='author_last_' if args.author_last else ''
        atomic_json(OUT/(prefix+eid+'_reviewed_trajectory.json'),visible)
        plot(dense,original,visible,window,case['target_frame'],source,OUT/(prefix+eid+'_原始与可见点轨迹.png'))
        rows.append(row)
    if {str(p):file_hash(p) for p in protected}!=hashes:raise ValueError('Protected sources changed')
    variants=list(rows[0]['variants'])
    result={'created_at':datetime.now(timezone.utc).isoformat(),
        'protocol':'Boundaries and researcher-visible pixel coordinate table were frozen before inference. Same original pose, target mapping and pretrained BST; window isolation is compared first, then only the shuttle changes. No interpolation, parameter search or truth-driven model selection.',
        'scope':'Two selected known net_cross developer errors; visible point readings and first/last rally bounds are researcher diagnoses, not user-confirmed coordinate/segmentation ground truth. Cannot measure generalization or trajectory accuracy.',
        'summary':{name:metrics(rows,name) for name in variants},'rows':rows,
        'source_sha256':hashes,'review_sha256':file_hash(review_path),'script_sha256':file_hash(Path(__file__)),
        'runtime_seconds':time.perf_counter()-started,'production_changed':False,'labels_changed':False,'model_trained':False}
    atomic_json(OUT/('author_last_hit_result.json' if args.author_last else 'result.json'),result)
    print(json.dumps(result['summary'],ensure_ascii=False),flush=True)
    for row in rows:print(row['event_id'],row['known_target_logit_margins'],flush=True)


if __name__=='__main__':main()
