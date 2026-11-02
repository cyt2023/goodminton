"""One fixed visual ROI TrackNet rerun; all production artifacts are read-only.

ROI selected once from the original court picture, not searched against labels.
Crop pixels losslessly, preserve frames, rerun the existing author adapter, map
coordinates back to the full image, and hold pose/BST/windows fixed.
"""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

from analysis_jobs import DATA, ROOT, atomic_json, file_hash, registered_video
from diagnose_input_repairs import metrics, prediction
from specialist_bst import SpecialistBST
from specialist_model import BST_NAME
from stroke_window_repairs import translate_contact_window

JOB='J_16b4f26a07a14a59b76a77b8bb58dc78'
PARENT=DATA/'击球标注/输入修复诊断/2026-10-12_配对输入修复/result.json'
OUT=DATA/'击球标注/输入修复诊断/2026-10-12_固定视觉ROI重跑TrackNet'
ROI=[0,0,600,448]


def crop_source(source, target):
    cap=cv2.VideoCapture(str(source))
    fps=cap.get(cv2.CAP_PROP_FPS)
    x1,y1,x2,y2=ROI
    writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'FFV1'),fps,(x2-x1,y2-y1))
    if not writer.isOpened():
        raise RuntimeError('No low-cost lossless FFV1 crop encoder available')
    checks={}
    n=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        cropped=frame[y1:y2,x1:x2].copy()
        if n in [0,158,170,471]:checks[n]=cropped
        writer.write(cropped);n+=1
    writer.release();cap.release()
    decoded=cv2.VideoCapture(str(target));seen=0;max_diff=0
    while True:
        ok,frame=decoded.read()
        if not ok:break
        if seen in checks:
            difference=np.max(abs(frame.astype(int)-checks[seen].astype(int)))
            max_diff=max(max_diff,int(difference))
        seen+=1
    decoded.release()
    if seen!=n or max_diff!=0:
        raise ValueError(f'Crop frame/pixel alignment failed: {n}/{seen}, difference {max_diff}')
    return {'source_frames':n,'cropped_decoded_frames':seen,'fps':fps,
        'codec':'FFV1','sampled_pixel_identity_frames':list(checks),'max_sampled_channel_difference':max_diff}


def plot_comparison(dense, original, mapped, window, source):
    cap=cv2.VideoCapture(str(source));cap.set(cv2.CAP_PROP_POS_FRAMES,158);ok,image=cap.read();cap.release()
    if not ok:raise ValueError('Missing contact background frame')
    panels=[]
    lo,hi=window['start'],window['end']
    for traj,title in zip([original,mapped],['Original full-image TrackNet','Fixed x=[0,600] crop; rerun TrackNet; mapped to full image']):
        panel=image.copy()
        observed=[f for f in traj['frames'][window['start_frame']:window['end_frame']+1] if f['raw_visible']]
        for f in observed:
            t=dense['frames'][f['frame_index']]['source_time']
            color=cv2.applyColorMap(np.asarray([[round(np.clip((t-lo)/max(hi-lo,.001),0,1)*255)]],dtype='uint8'),cv2.COLORMAP_VIRIDIS)[0,0]
            xy=tuple(map(round,f['raw_xy']))
            cv2.circle(panel,xy,5,(255,255,255),-1)
            cv2.circle(panel,xy,4,tuple(map(int,color)),-1)
        cv2.line(panel,(600,0),(600,dense['source_height']-1),(0,0,255),2)
        for i in [155,158,161,170]:
            f=traj['frames'][i]
            if f['raw_visible']:
                xy=tuple(map(round,f['raw_xy']))
                cv2.putText(panel,f'f{i}',(xy[0]+6,xy[1]-6),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),2)
        header=np.full((64,panel.shape[1],3),245,dtype='uint8')
        cv2.putText(header,title,(10,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),1)
        cv2.putText(header,f'Same frozen window {lo:.3f}-{hi:.3f}s; human-labelled smash at 6.583s; unreviewed coordinates',(10,47),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1)
        panels.append(np.vstack([header,panel]))
    cv2.imwrite(str(OUT/'6.583秒原轨迹与裁图重跑对照.png'),np.vstack(panels))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    started=time.perf_counter()
    OUT.mkdir(parents=True,exist_ok=args.resume)
    original_root=DATA/'专项分析'/JOB
    dense_path=original_root/'dense_pose.json';trajectory_path=original_root/'trajectory.json'
    dense=json.loads(dense_path.read_text());original=json.loads(trajectory_path.read_text())
    report=json.loads((original_root/'report.json').read_text())
    frozen_path=DATA/'击球标注'/JOB/'public_only_predictions.json'
    label_path=DATA/'击球标注'/JOB/'annotations.json'
    source=registered_video(report['source_video_id'],allow_reference=True)['path']
    checkpoint=ROOT/'models/bst'/BST_NAME
    protected=[dense_path,trajectory_path,frozen_path,label_path,checkpoint]
    before={str(p):file_hash(p) for p in protected}
    parent=json.loads(PARENT.read_text())
    if file_hash(checkpoint)!=parent['model_sha256']:
        raise ValueError('BST checkpoint changed from parent paired diagnostic')
    crop_file=OUT/'fixed_roi_ffv1.avi'
    crop_checks=(json.loads((OUT/'roi_selection.json').read_text())['crop_checks'] if args.resume else crop_source(source,crop_file))
    if crop_checks['source_frames']!=len(dense['frames']):
        raise ValueError('Dense pose and crop source frame counts differ')
    atomic_json(OUT/'roi_selection.json',{'job_id':JOB,'roi_xyxy':ROI,
        'original_image_size':[960,448],'selected_before_rerun':True,
        'visual_basis':'Right border near the black net post separating the target left court and the adjacent right court, at original x about 590. Keep x<=600 and the full height to retain high shuttles; chosen once from source frame views, no accuracy search.',
        'human_user_confirmed':False,'is_court_ground_truth':False,
        'limitations':'May remove real target-court shuttle paths crossing the fixed image boundary; does not fix wrong pose identity. Changes spatial scale/context of the tracking input.',
        'crop_checks':crop_checks})
    config={'source_file':str(crop_file),'source_sha256':file_hash(crop_file),
            'specialist_folder':str(OUT/'cropped_tracker_output')}
    atomic_json(OUT/'tracknet_config.json',config)
    if not args.resume:
        process=subprocess.run([str(ROOT/'.venv-vision/bin/python'),str(ROOT/'specialist_tracknet.py'),
            '--config',str(OUT/'tracknet_config.json')],capture_output=True,text=True,timeout=420)
        (OUT/'tracknet.log').write_text(process.stdout+process.stderr)
        if process.returncode:
            raise RuntimeError('Cropped TrackNet failed; see diagnostic tracknet.log')
    cropped=json.loads((OUT/'cropped_tracker_output/trajectory.json').read_text())
    if cropped['source_sha256']!=config['source_sha256']:
        raise ValueError('Cropped source video differs from saved TrackNet run')
    if len(cropped['frames'])!=len(original['frames']):raise ValueError('TrackNet frame count changed')
    mapped={**cropped,'source_sha256':original['source_sha256'],
        'source_width':original['source_width'],'source_height':original['source_height'],
        'engine_version':'diagnostic-fixed-visual-roi-tracknet-mapped-v1','frames':[],
        'roi_xyxy':ROI,'cropped_source_sha256':config['source_sha256'],
        'coordinate_mapping':'full_x=crop_x+roi_left; full_y=crop_y+roi_top; no temporal resampling',
        'original_source_sha256':original['source_sha256'],'diagnostic_only':True}
    for f in cropped['frames']:
        r=dict(f)
        for name in ['raw_xy','rectified_xy']:
            xy=f.get(name)
            r[name]=[xy[0]+ROI[0],xy[1]+ROI[1]] if xy is not None and any(xy) else None
        mapped['frames'].append(r)
    atomic_json(OUT/'mapped_full_image_trajectory.json',mapped)
    atomic_json(OUT/'original_trajectory_snapshot.json',original)
    atomic_json(OUT/'dense_pose_snapshot.json',dense)
    classifier=SpecialistBST(dense,original)
    cropped_classifier=SpecialistBST(dense,mapped)
    torch.set_num_threads(1)
    rows=[];replay_errors=[]
    for old in [r for r in parent['rows'] if r['job_id']==JOB]:
        eid=old['event_id'];window=old['frozen_window'];y=old['human']
        classifier.order=list(old['variants']['frozen_replay']['player_order'])
        values,_=classifier.inputs(window)
        p=prediction(classifier,values,y['player'],old['automatic_candidate_player'])
        error=float(np.max(abs(np.asarray(p['scores'])-old['variants']['frozen_replay']['scores'])))
        replay_errors.append(error)
        if error>1e-5:raise ValueError('Original replay mismatch')
        cropped_classifier.order=list(classifier.order)
        values,q=cropped_classifier.inputs(window)
        after=prediction(cropped_classifier,values,y['player'],old['automatic_candidate_player'])
        shifted=translate_contact_window(dense,window,y['peak_time'])
        values,qs=classifier.inputs(shifted)
        shifted_original=prediction(classifier,values,y['player'],old['automatic_candidate_player'])
        shift_error=float(np.max(abs(np.asarray(shifted_original['scores'])-old['variants']['human_center_same_duration']['scores'])))
        replay_errors.append(shift_error)
        if shift_error>1e-5:raise ValueError('Shifted original replay mismatch')
        values,qcs=cropped_classifier.inputs(shifted)
        shifted_after=prediction(cropped_classifier,values,y['player'],old['automatic_candidate_player'])
        rows.append({'job_id':JOB,'event_id':eid,'human':y,'window':window,
            'shifted_window':shifted,
            'variants':{'original_full_image_tracknet':p,'fixed_visual_roi_tracknet':after,
                'original_shifted_full_image_tracknet':shifted_original,'fixed_visual_roi_shifted_tracknet':shifted_after},
            'cropped_input_quality':q,'shifted_original_input_quality':qs,'shifted_cropped_input_quality':qcs})
        if eid=='76b88ba1502c9ddf3301c666990413c1':
            plot_comparison(dense,original,mapped,window,source)
    unchanged={str(p):file_hash(p) for p in protected}==before
    if not unchanged:raise ValueError('Production source changed during diagnostic')
    result={'created_at':datetime.now(timezone.utc).isoformat(),'job_id':JOB,
        'scope':'Seven selected, previously inspected development strokes of one ordinary-view recording; no independent or whole-video accuracy claim.',
        'protocol':'Single visually selected fixed ROI, no parameter search. Rerun original TrackNet+Inpaint on the pixel crop; map coordinates to original full-image coordinates. Hold original dense pose, player ordering, all frozen windows and pretrained BST weights fixed. No training.',
        'roi_xyxy':ROI,'crop_checks':crop_checks,
        'summary':{name:metrics(rows,name) for name in ['original_full_image_tracknet','fixed_visual_roi_tracknet',
            'original_shifted_full_image_tracknet','fixed_visual_roi_shifted_tracknet']},
        'rows':rows,'original_replay_max_score_error':max(replay_errors),
        'raw_detected_frames_before':original['raw_detected_frames'],
        'raw_detected_frames_after':cropped['raw_detected_frames'],
        'raw_trajectory_correctness_measured':False,
        'tracknet_runtime_seconds':cropped['runtime_seconds'],'comparison_runtime_seconds':time.perf_counter()-started,
        'runtime_note':'Comparison resumed from saved TrackNet output; comparison time excludes crop encoding and TrackNet runtime.',
        'source_files_sha256':before,'source_files_unchanged':unchanged,
        'source_video_sha256':file_hash(source),'script_sha256':file_hash(Path(__file__)),
        'active_model_replaced':False,'human_labels_created':0,'model_trained':False}
    if (OUT/'result.json').exists() and not (OUT/'result_before_shifted_context.json').exists():
        (OUT/'result_before_shifted_context.json').write_bytes((OUT/'result.json').read_bytes())
    atomic_json(OUT/'result.json',result)
    print('CROPPED_RESULT',OUT/'result.json',flush=True)
    print(json.dumps(result['summary'],ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
