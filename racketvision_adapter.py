"""RacketVision's released BallTrack + RTMDet/RTMPose checkpoints on native video.

BGR/JPEG inputs, author graph/configs and decode rules are retained. Ball windows
end at the output frame, matching Uball training (official inference.py is one
frame early). The median uses 64 deterministic samples, an author-supported cap.
"""
import argparse
from collections import deque
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
VENDOR=ROOT/'vendor/RacketVision/source'
WEIGHTS=ROOT/'models/racketvision'


def run(config):
    from racketvision_runtime import activate
    compatibility=activate()
    import cv2
    import numpy as np
    import torch
    from analysis_jobs import file_hash,atomic_json
    source=Path(config['source_file']).resolve();out=Path(config['racketvision_folder']).resolve()
    out.mkdir(parents=True,exist_ok=True)
    source_hash=file_hash(source)
    if source_hash!=config['source_sha256']:raise ValueError('Source video changed')
    manifest=json.loads((WEIGHTS/'manifest.json').read_text())
    fingerprint={'adapter_sha256':file_hash(Path(__file__)),'runtime_sha256':file_hash(ROOT/'racketvision_runtime.py'),
                 'weights_manifest_sha256':file_hash(WEIGHTS/'manifest.json'),'source_sha256':source_hash,
                 'max_frames':config.get('max_frames')}
    cache=out/'racketvision.json'
    if cache.exists():
        saved=json.loads(cache.read_text())
        if saved.get('cache_fingerprint')==fingerprint and (out/'trajectory.json').exists():
            print('RACKETVISION_DONE '+json.dumps({'cached':True,'frames':saved['frame_count']}),flush=True)
            return saved
    for name in ['balltrack_best.pth','epoch_300.pth','best_PCK_epoch_90.pth']:
        if file_hash(WEIGHTS/name)!=manifest['files'][name]['sha256']:raise ValueError('RacketVision checkpoint hash changed: '+name)
    requested=config.get('device','auto')
    device=('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if requested=='auto' else requested
    torch.set_num_threads(4)
    sys.path.insert(0,str(VENDOR/'BallTrack'))
    from model.tracknet_v3 import TrackNet
    net=TrackNet(in_dim=15,out_dim=4,mixup=True,alpha=.5,last_only=True)
    ck=torch.load(WEIGHTS/'balltrack_best.pth',map_location='cpu')
    state=ck.get('state_dict',ck)
    net.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    net=net.to(device).eval();del ck,state
    path=VENDOR/'RacketPose/tools/inference.py'
    spec=importlib.util.spec_from_file_location('racketvision_author_pose',path)
    author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
    det=author.init_detector(str(VENDOR/'RacketPose/configs/detection/rtmdet_m_racket_infer.py'),str(WEIGHTS/'epoch_300.pth'),device=device)
    pose=author.init_pose_model(str(VENDOR/'RacketPose/configs/pose/rtmpose_m_racket_infer.py'),str(WEIGHTS/'best_PCK_epoch_90.pth'),device=device)
    det.eval();pose.eval()
    cap=cv2.VideoCapture(str(source));fps=float(cap.get(5));total=int(cap.get(7));w,h=int(cap.get(3)),int(cap.get(4))
    if not fps>0 or total<1:raise ValueError('Cannot read video metadata')
    limit=min(total,int(config.get('max_frames') or total))
    def jpeg(frame):
        ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,95])
        if not ok:raise ValueError('JPEG preprocessing failed')
        return cv2.imdecode(encoded,cv2.IMREAD_COLOR)
    samples=[];wanted=set(np.unique(np.linspace(0,limit-1,min(64,limit),dtype=int)));observed=0
    # Some phone videos have VFR packets: CAP_PROP_POS_FRAMES random seeks can
    # miss their final frames. Sequential decoding preserves original indices.
    while observed<limit:
        ok,im=cap.read()
        if not ok:break
        if observed in wanted:samples.append(jpeg(im))
        observed+=1
    if observed<1 or not samples:raise ValueError('Video could not decode')
    source_count=observed if not config.get('max_frames') else total
    limit=observed
    median=np.median(np.array(samples),axis=0).astype(np.uint8);del samples
    cv2.imwrite(str(out/'median.jpg'),median)
    median=cv2.resize(median,(512,288)).transpose(2,0,1).astype(np.float32)/255.
    cap.release();cap=cv2.VideoCapture(str(source))
    history=deque(maxlen=4);frames=[];trajectory=[];started=time.perf_counter();batch_size=int(config.get('batch_size',2));fallbacks=0
    while len(frames)<limit:
        batch=[];images=[];times=[]
        for _ in range(min(batch_size,limit-len(frames))):
            index=len(frames)+len(images);ok,im=cap.read()
            if not ok:raise ValueError('Video decode ended before requested coverage')
            t=float(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
            previous=times[-1] if times else frames[-1]['source_time'] if frames else -1
            basis='opencv_decoder_pos_msec'
            if not np.isfinite(t) or t<=previous:
                t=index/fps;basis='frame_index_divided_by_nominal_fps_fallback';fallbacks+=1
            im=jpeg(im);small=cv2.resize(im,(512,288)).transpose(2,0,1).astype(np.float32)/255.
            if not history:history.extend([small]*4)
            else:history.append(small)
            batch.append(np.concatenate([median,*history],axis=0));images.append(im);times.append(t)
        with torch.inference_mode():
            hm=net(frames=torch.from_numpy(np.array(batch)).to(device))[0].cpu().numpy()
            for im,t,heat in zip(images,times,hm[:,0]):
                index=len(frames);binary=(heat>.5).astype(np.uint8)*255
                contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                xy=None;confidence=None
                if contours:
                    x,y,bw,bh=max((cv2.boundingRect(c) for c in contours),key=lambda b:b[2]*b[3])
                    xy=[int((x+bw/2)*w/512),int((y+bh/2)*h/288)]
                    confidence=float(heat[y:y+bh,x:x+bw].mean())
                rackets=author.detect_and_estimate_pose(det,pose,im,0,bbox_thr=.3)
                frames.append({'frame_index':index,'source_time':round(t,6),'rackets':rackets,
                               'shuttle':{'xy':xy,'confidence':confidence,'heatmap_peak':float(heat.max()),'visible':xy is not None}})
                trajectory.append({'frame_index':index,'source_time':round(t,6),'source_time_nominal':round(index/fps,6),
                    'raw_xy':xy,'raw_visible':xy is not None,'raw_heatmap_peak':float(heat.max()),
                    'inpaint_requested':False,'rectified_xy':xy,'coordinate_source':'racketvision_balltrack_detection'})
        if len(frames)%20<batch_size or len(frames)==limit:
            print('RACKETVISION_PROGRESS '+json.dumps({'processed':len(frames),'total':limit,'runtime_seconds':round(time.perf_counter()-started,2)}),flush=True)
    cap.release()
    result={'schema_version':1,'source_sha256':source_hash,'source_file':str(source),'source_width':w,'source_height':h,
        'nominal_fps':fps,'frame_count':len(frames),'source_frame_count':source_count,'container_frame_count':total,'complete_video':not config.get('max_frames') or limit==total,
        'frames':frames,'runtime_seconds':round(time.perf_counter()-started,3),'device':device,'timestamp_fallbacks':fallbacks,
        'cache_fingerprint':fingerprint,'model_provenance':{'name':'RacketVision','kind':'model_inference',**manifest,'runtime':compatibility},
        'configuration':{'ball_input':'BGR JPEG95; median + four consecutive frames ending at target frame; resize512x288 /255',
            'median_samples':len(wanted),'threshold':.5,'racket_detection_threshold':.3,'racket_keypoint_names':['top','bottom','handle','left','right'],
            'adaptations':['Training-dataset frame alignment fixes inference.py one-frame-early window',
                           'Median sampled from current clip rather than all rallies in a match',
                           'Streaming input and Apple Silicon compatibility; no retraining']},
        'interpretation':'Unreviewed inference. Detection counts are not accuracy; racket five points do not establish 3D face orientation.'}
    traj={'schema_version':1,'engine_version':'racketvision-balltrack-native-v1','source_sha256':source_hash,'source_width':w,'source_height':h,
        'nominal_fps':fps,'frame_count':len(frames),'frames':trajectory,'runtime_seconds':result['runtime_seconds'],'device':device,
        'code_sha256':fingerprint['adapter_sha256'],'weights_manifest_sha256':fingerprint['weights_manifest_sha256'],
        'raw_detected_frames':sum(f['raw_visible'] for f in trajectory),'inpaint_requested_frames':0,
        'model_provenance':result['model_provenance'],'note':'No interpolation or trajectory forecasts are represented as observed positions.'}
    atomic_json(out/'trajectory.json',traj);atomic_json(cache,result)
    print('RACKETVISION_DONE '+json.dumps({'frames':len(frames),'shuttle_frames':traj['raw_detected_frames'],
        'racket_frames':sum(bool(f['rackets']) for f in frames),'runtime_seconds':result['runtime_seconds'],'device':device}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--source');p.add_argument('--output');p.add_argument('--device',default='auto');p.add_argument('--max-frames',type=int)
    a=p.parse_args()
    if a.config:
        c=json.loads(Path(a.config).read_text());c['racketvision_folder']=c.get('racketvision_folder',str(Path(c['specialist_folder'])/'racketvision'))
    else:
        from analysis_jobs import file_hash
        c={'source_file':a.source,'source_sha256':file_hash(Path(a.source)),'racketvision_folder':a.output,'device':a.device,'max_frames':a.max_frames}
    run(c)
