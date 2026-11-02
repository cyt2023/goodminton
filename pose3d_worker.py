"""Optional MediaPipe33 monocular pose; no action classifier or metric-scale claim.
Run with .venv-pose3d/bin/python pose3d_worker.py --job J_... --start0 --end4.
"""
import argparse,json,time,math,hashlib
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from analysis_jobs import DATA,file_hash,atomic_json
ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'models/pose3d/pose_landmarker_full.task'


def extract(job,start=0.,end=None,sample_fps=5.):
    dense=json.loads((DATA/'专项分析'/job/'dense_pose.json').read_text())
    config=json.loads((DATA/'专项分析'/job/'config.json').read_text())
    video=Path(dense.get('source_file') or config['source_file'])
    if file_hash(video)!=dense['source_sha256']:raise ValueError('3D input source hash differs from2D tracks')
    if not 0<sample_fps<=30:raise ValueError('3D sampling must be in(0,30]fps')
    timeline=[f['source_time'] for f in dense['frames']];end=timeline[-1] if end is None else min(end,timeline[-1])
    if start<0 or end<=start:raise ValueError('Invalid3D time range')
    indices=[];next_time=start
    for i,t in enumerate(timeline):
        if t>=start and t<=end and t+1e-6>=next_time:indices.append(i);next_time=t+1/sample_fps
    cap=cv2.VideoCapture(str(video));w,h=dense['source_width'],dense['source_height']
    if [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]!=[w,h]:raise ValueError('3D source orientation/size differs from2D tracks')
    options=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),num_poses=1,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5)
    frames=[];begin=time.perf_counter()
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
        wanted=set(indices)
        for i in range(max(indices,default=-1)+1):
            ok,bgr=cap.read()
            if not ok:raise ValueError(f'Cannot decode original frame{i}')
            if i not in wanted:continue
            people={}
            for side,p in dense['frames'][i]['players'].items():
                if not p:people[side]=None;continue
                x1,y1,x2,y2=p['bbox'];cx,cy=(x1+x2)/2,(y1+y2)/2;bw,bh=(x2-x1)*1.2,(y2-y1)*1.15
                a,b,c,d=max(0,int(cx-bw/2)),max(0,int(cy-bh/2)),min(w,int(cx+bw/2)),min(h,int(cy+bh/2))
                if min(c-a,d-b)<32:people[side]=None;continue
                image=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(cv2.cvtColor(bgr[b:d,a:c],cv2.COLOR_BGR2RGB)))
                result=detector.detect(image)
                if not result.pose_world_landmarks:people[side]=None;continue
                worlds=result.pose_world_landmarks[0];points=result.pose_landmarks[0]
                xy=[[a+k.x*(c-a),b+k.y*(d-b)] for k in points]
                hip=np.mean([xy[23],xy[24]],axis=0)
                if not x1<=hip[0]<=x2 or not y1<=hip[1]<=y2:people[side]=None;continue
                joints=[{'x':float(k.x),'y':float(k.y),'z':float(k.z),'score':float(min(getattr(k,'visibility',0),getattr(k,'presence',0)))} for k in worlds]
                visible=sum(v['score']>=.5 for v in joints)/33
                people[side]={'joints':joints,'image_xy':xy,'visibility_fraction':visible,'crop_bbox':[a,b,c,d],
                    'identity_status':'inherited_2d_track_unverified','uncertainty':{'metric_scale_calibrated':False,'depth_error_bound':None,
                    'scores_are':'visibility/presence, not calibrated3D error probability','occlusion_fraction':1-visible}}
            frames.append({'frame_index':i,'timestamp':timeline[i],'players':people})
    cap.release()
    packet={'schema_version':1,'component':'monocular_pose3d','kind':'model_inference','source_sha256':dense['source_sha256'],
        'source_file':str(video),'image_size':[w,h],'sample_fps':sample_fps,'joint_schema':'mediapipe33',
        'coordinate_system':'mediapipe_world_hip_origin_xyz','units':'estimated_meters','global_translation_available':False,
        'model_provenance':{'name':'MediaPipe Pose Landmarker Full','version':mp.__version__,'weights_sha256':file_hash(MODEL),
        'method_reference':'https://doi.org/10.1109/ICCE-Taiwan71481.2026.11652777','reproduction_scope':'MediaPipe-derived perception/features only; no paper classifier'},
        'uncertainty':{'calibrated_error_bounds_available':False,'limitations':['Monocular depth/scale ambiguity','Crop and occlusion dependence','Hip-origin world landmarks cannot measure global body translation']},
        'frames':frames,'seconds':time.perf_counter()-begin,'processed_person_frames':sum(p is not None for f in frames for p in f['players'].values())}
    folder=DATA/'专项分析'/job/'pose3d';target=folder/'pose3d.json'
    if target.exists():
        old=json.loads(target.read_text())
        if old['source_sha256']==packet['source_sha256'] and old['model_provenance']['weights_sha256']==packet['model_provenance']['weights_sha256'] and old['sample_fps']==sample_fps:
            merged={f['frame_index']:f for f in old['frames']};merged.update({f['frame_index']:f for f in packet['frames']})
            packet['frames']=[merged[i] for i in sorted(merged)];packet['processed_person_frames']=sum(p is not None for f in packet['frames'] for p in f['players'].values())
    atomic_json(target,packet)
    print(json.dumps({'status':'completed','frames':len(frames),'person_frames':packet['processed_person_frames'],'seconds':packet['seconds']}),flush=True)
    return packet

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--start',type=float,default=0.);p.add_argument('--end',type=float);p.add_argument('--sample-fps',type=float,default=5.)
    a=p.parse_args();extract(a.job,a.start,a.end,a.sample_fps)
