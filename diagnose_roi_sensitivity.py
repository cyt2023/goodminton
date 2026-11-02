"""Fixed broad full-video ROI sensitivity on cached diagnostic BST inputs.

This removes only points outside a label-independent proposed region. It does
not crop pixels, rerun TrackNet, manufacture a true shuttle trajectory, or change
any active inputs. The region can be too broad if player identity drifted.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from analysis_jobs import DATA, atomic_json, file_hash
from diagnose_input_repairs import metrics, prediction
from input_quality import proposed_horizontal_roi
from personal_learning import extractor


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    args=parser.parse_args()
    root=DATA/'击球标注/输入修复诊断'/args.run
    source=root/'result.json'
    original=json.loads(source.read_text())
    with np.load(root/'inputs.npz',allow_pickle=False) as cache:
        keys=[tuple(v) for v in cache['keys'].tolist()]
        arrays={name:cache[name].copy() for name in ['joints_and_bones','positions','shuttle','video_len']}
    index={key:i for i,key in enumerate(keys)}
    torch.set_num_threads(1)
    started=time.perf_counter()
    rows,regions={},{}
    for job in sorted({r['job_id'] for r in original['rows']}):
        classifier,dense=extractor(job)
        roi=proposed_horizontal_roi(dense)
        regions[job]=roi
        for old in [r for r in original['rows'] if r['job_id']==job]:
            eid=old['event_id']
            row={'job_id':job,'event_id':eid,'human':old['human'],'variants':{},'removed_sample_count':{}}
            for kind in ['frozen_replay','human_center_same_duration']:
                i=index[job,eid,kind]
                values=[arrays['joints_and_bones'][i].copy(),arrays['positions'][i].copy(),arrays['shuttle'][i].copy(),int(arrays['video_len'][i])]
                original_s=values[2].copy()
                valid=np.any(original_s!=0,axis=1)
                outside=(original_s[:,0] < roi['left']/dense['source_width']) | (original_s[:,0] > roi['right']/dense['source_width'])
                values[2][outside]=0
                name=kind+'_fixed_roi'
                classifier.order=list(old['variants'][kind]['player_order'])
                row['variants'][name]=prediction(classifier,values,old['human']['player'],old['automatic_candidate_player'])
                row['removed_sample_count'][name]=int((valid & outside).sum())
                row['variants'][kind]=old['variants'][kind]
            rows[job,eid]=row
    ordered=list(rows.values())
    variants=['frozen_replay','frozen_replay_fixed_roi','human_center_same_duration','human_center_same_duration_fixed_roi']
    result={'created_at':datetime.now(timezone.utc).isoformat(),
        'scope':'Paired development sensitivity on 22 selected examples. Full-video ROI is proposed from unreviewed tracked player boxes, not a true court region.',
        'protocol':'Keep cached pose, original 100-frame sequence length and BST weights fixed. Zero only shuttle coordinates outside a single fixed horizontal ROI per full recording. Retain full vertical range. No per-example tuning or stroke-label-dependent region.',
        'regions':regions,'summary':{v:metrics(ordered,v) for v in variants},'rows':ordered,
        'total_nonzero_cached_points_removed':{v:sum(r['removed_sample_count'][v] for r in ordered) for v in variants if v.endswith('_fixed_roi')},
        'parent_result_sha256':file_hash(source),'cached_inputs_sha256':file_hash(root/'inputs.npz'),
        'script_sha256':file_hash(Path(__file__)),'runtime_seconds':time.perf_counter()-started,
        'model_trained':False,'active_model_replaced':False,'source_labels_or_predictions_changed':False}
    atomic_json(root/'roi_sensitivity.json',result)
    print(json.dumps(result['summary'],ensure_ascii=False),flush=True)
    print('REMOVED',result['total_nonzero_cached_points_removed'],flush=True)


if __name__=='__main__':
    main()
