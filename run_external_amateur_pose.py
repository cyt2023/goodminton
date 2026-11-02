"""Call unchanged production scout/dense pose on an isolated public test case."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from rtmlib import RTMPose,YOLOX

import analyze_pose
import specialist_pose


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',required=True)
    args=parser.parse_args();row=json.loads(Path(args.case).read_text())
    case=Path(row['case_folder']);config=json.loads(Path(row['config_file']).read_text())
    source=Path(row['source_path']);weights=analyze_pose.get_weights()
    detector=YOLOX(weights['detector']['file'],model_input_size=(640,640),score_thr=.4,backend='onnxruntime',device='cpu')
    pose=RTMPose(weights['pose']['file'],model_input_size=(192,256),to_openpose=False,backend='onnxruntime',device='cpu')
    coarse=Path(config['pose_file'])
    if not coarse.exists():
        original_resolver=analyze_pose.registered_source;original_out=analyze_pose.OUTPUT
        try:
            analyze_pose.OUTPUT=case/'coarse'
            analyze_pose.registered_source=lambda job:(source,{'video_id':row['id'],'name':row['video_name']})
            job={'id':row['id'],'video_id':row['id'],'label':row['id'],'mode':'auto','players':{},'start':0.,'end':config['end']}
            analyze_pose.process_job(job,{'sample_fps':5,'display_threshold':.5,'detector_threshold':.4,'max_width':960,'publish_results':False},weights,detector,pose)
        finally:
            analyze_pose.OUTPUT=original_out;analyze_pose.registered_source=original_resolver
    dense_path=Path(config['specialist_folder'])/'dense_pose.json'
    if not dense_path.exists():
        sys.argv=[str(Path(specialist_pose.__file__)),'--config',row['config_file']]
        specialist_pose.main()
    # Frozen extra input control: manually provided boxes, same estimated pose.
    oracle_path=dense_path.with_name('human_bbox_estimated_pose.json')
    if oracle_path.exists():return
    import copy,csv
    dense=json.loads(dense_path.read_text());oracle=copy.deepcopy(dense)
    box_rows=list(csv.DictReader(Path(row['bbox_csv']).open()));boxes={}
    w,h=row['dimensions']
    for record in box_rows:
        if int(float(record['Visbility']))<=0:continue
        frame=int(record['Frame']);pid=int(record['id'])
        if pid not in [1,2]:continue
        box=[float(record[k])*scale for k,scale in zip(['x1','y1','x2','y2'],[w,h,w,h])]
        if not (np.isfinite(box).all() and box[0]<box[2] and box[1]<box[3]):continue
        boxes.setdefault(frame,{})['near' if pid==1 else 'far']=box
    cap=cv2.VideoCapture(str(source));i=0
    while True:
        ok,image=cap.read()
        if not ok:break
        people={'near':None,'far':None};b=boxes.get(i,{})
        if b:
            keys=list(b);points,scores=pose(image,bboxes=[b[k] for k in keys])
            for k,pts,conf in zip(keys,points,scores):
                pts=np.asarray(pts);conf=np.asarray(conf)
                valid=np.isfinite(pts).all(1)&np.isfinite(conf);conf=np.where(valid,conf,0);pts=np.where(np.isfinite(pts),pts,0)
                inside=valid&(pts[:,0]>=0)&(pts[:,0]<w)&(pts[:,1]>=0)&(pts[:,1]<h)
                people[k]={'track_id':row['id']+'-human-id-'+('1' if k=='near' else '2'),'bbox':b[k],
                    'keypoints':[{'name':name,'x':float(v[0]),'y':float(v[1]),'score':float(s),'in_frame':bool(ins)} for name,v,s,ins in zip(analyze_pose.JOINTS,pts,conf,inside)],
                    'features':analyze_pose.image_features(pts,np.where(inside,conf,0),.5),'identity_status':'author_manual_bbox_pose_still_model_estimated'}
        oracle['frames'][i]['players']=people;oracle['frames'][i]['assignment']={'source':'author_manual_bbox_control'}
        i+=1
        if i%100==0:print('HUMAN_BBOX_POSE',i,'/',row['frames'],flush=True)
    cap.release();assert i==len(oracle['frames'])==row['frames']
    oracle['input_intervention']={'boxes':'author CSV','pose':'same production BGR RTMPose estimated','human_joint_ground_truth':False,'production_changed':False}
    oracle['assigned_frames']={k:sum(f['players'][k] is not None for f in oracle['frames']) for k in ['near','far']}
    analyze_pose.save_json(oracle_path,oracle)


if __name__=='__main__':main()
