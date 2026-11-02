"""Real raw-video milestone: RacketVision + existing2D pose + local SHF/DSTA.
This entry accepts no annotation paths. BFMD references are read by scorer only.
"""
import argparse,json,subprocess,hashlib,math
from pathlib import Path
import cv2,numpy as np,torch
from analysis_jobs import file_hash
from racketvision_adapter import run as racketvision
from court_geometry_worker import automatic_corners
from court_calibration import color_mask
from paper_reproductions.data import pack_shf,ground_midpoints,read_rgb_window
from paper_reproductions.run import load_model
from paper_reproductions.shuttlehitformer import center_windows,nms_candidates
from stroke_event_fusion import fuse
ROOT=Path(__file__).resolve().parents[2]

def write(path,value):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main(a):
    video=Path(a.video).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True);sha=file_hash(video)
    pose_path=out/'dense_pose.json'
    if not pose_path.exists():subprocess.run([str(ROOT/'.venv-pose/bin/python'),str(Path(__file__).with_name('pose_worker.py')),'--video',str(video),'--out',str(out)],check=True)
    dense=json.loads(pose_path.read_text())
    if dense['source_sha256']!=sha:raise ValueError('Independent pose input source mismatch')
    config={'source_file':str(video),'source_sha256':sha,'racketvision_folder':str(out/'racketvision')}
    rv=racketvision(config);trajectory=json.loads((out/'racketvision/trajectory.json').read_text());n=len(dense['frames']);fps=dense['sample_fps'];w,h=dense['source_width'],dense['source_height'];times=[f['source_time'] for f in dense['frames']]
    if len(trajectory['frames'])!=n:raise ValueError('Raw model frame counts differ')
    cap=cv2.VideoCapture(str(video));ok,image=cap.read();cap.release();proposal=automatic_corners(color_mask(image,'white')) if ok else None
    events=[];status={'pose2d':'model_inference','racketvision':'model_inference','shuttlehitformer':'court_detection_unavailable','dsta':'not_run'};raw_hits=[]
    if proposal:
        corners=np.asarray(proposal['points_px'],np.float32)[[0,1,3,2]];court=np.concatenate([corners,ground_midpoints(corners)],0)
        top=np.zeros((n,17,2),np.float32);bottom=np.zeros_like(top);valid=np.zeros(n,bool)
        ball=np.array([f['raw_xy'] if f['raw_visible'] else [np.nan,np.nan] for f in trajectory['frames']],np.float32)
        observed=np.isfinite(ball).all(1);good=np.flatnonzero(observed)
        # Short gaps only, derived from predictions. No ground-truth input.
        for i in np.flatnonzero(~observed):
            left=good[good<i];right=good[good>i]
            if len(left) and len(right) and right[0]-left[-1]<=4:ball[i]=ball[left[-1]]+(ball[right[0]]-ball[left[-1]])*(i-left[-1])/(right[0]-left[-1])
        for i,f in enumerate(dense['frames']):
            if f['players']['near'] and f['players']['far'] and np.isfinite(ball[i]).all():
                top[i]=[[k['x'],k['y']] for k in f['players']['far']['keypoints']];bottom[i]=[[k['x'],k['y']] for k in f['players']['near']['keypoints']];valid[i]=True
        features=pack_shf(top,bottom,np.repeat(court[None],n,0),np.zeros((n,4,2)),np.nan_to_num(ball),normalized=False,width=w,height=h)
        model,cp=load_model(a.shf_checkpoint);model.eval();centers=[i for i in range(3,n-4) if valid[i-3:i+5].all()]
        with torch.no_grad():
            for indices in torch.tensor(centers,dtype=torch.long).split(256):
                prob=model(center_windows(torch.from_numpy(features),indices))['logits'].softmax(-1)
                for i,p in zip(indices.tolist(),prob.tolist()):
                    label=max(range(3),key=p.__getitem__)
                    if label>0 and p[label]>model.config.threshold:raw_hits.append({'frame_index':i,'class_id':label,'confidence':p[label]})
        raw_hits=nms_candidates(raw_hits,model.config.nms_frames);status['shuttlehitformer']={'status':'local_variant_inference','valid_windows':len(centers),'total_possible_windows':max(0,n-7),'observed_ball_frames':int(observed.sum()),'court_source':'automatic_unreviewed_Hough; no BFMD geometry inputs'}
    actions=[]
    if a.dsta_checkpoint:
        model,cp=load_model(a.dsta_checkpoint);model.eval();c=model.config;window=c.window_frames*c.sample_stride/c.fps
        with torch.no_grad():
            for start in np.arange(0,max(0,n/fps-window),window):
                rgb,_=read_rgb_window(video,float(start),c.window_frames,fps=c.fps,size=c.input_size,sample_stride=c.sample_stride)
                meta=[{'video_id':sha,'fps':float(c.fps),'duration':window,'feat_stride':c.window_frames*c.sample_stride/c.feature_length,'feat_num_frames':0.}]
                pred=model(rgb[None],meta)[0]
                for j,(interval,score,label) in enumerate(zip(pred['segments'],pred['scores'],pred['labels'])):
                    if float(score)<.5:continue
                    lo,hi=float(interval[0])+float(start),float(interval[1])+float(start)
                    actions.append({'id':hashlib.sha256(f'{start}:{j}'.encode()).hexdigest()[:32],'start':lo,'end':hi,'start_frame':max(0,round(lo*fps)),'end_frame':min(n-1,round(hi*fps)),
                        'stroke_type':cp['labels'][int(label)],'raw_label':cp['labels'][int(label)],'confidence':float(score),'candidate_player_id':None,'provenance':{'component':'dsta','kind':'imported_prediction','name':'local32-frame training pilot'}})
        status['dsta']={'status':'experimental_pilot_inference','qualified_intervals':len(actions),'scope':cp['scope'],'fully_trained_for_BFMD':False}
    hits=[{'id':hashlib.sha256((sha+str(r['frame_index'])).encode()).hexdigest()[:32],'frame_index':r['frame_index'],'confidence':r['confidence'],'player_id':None,'candidate_player_id':'far' if r['class_id']==1 else 'near','provenance':{'component':'shuttlehitformer','kind':'imported_prediction','name':'local aligned-stream variant'}} for r in raw_hits]
    fused=fuse(hits,{'predictions':actions},{'times':times,'fps':fps,'match_id':a.match_id})
    packet={'kind':'model_inference','mode':'end_to_end','match_id':a.match_id,'time_base':{'unit':'seconds','offset_seconds':a.offset},'evaluation_window':[0,n/fps],
        'coordinates':{'space':'native_pixels','width':w,'height':h},'alignment_status':a.alignment_status,
        'provenance':{'model':'RacketVision+existingYOLOX/RTMPose+localSHF+experimentalDSTA','input_video_sha256':sha,'annotation_inputs':[],
            'shf_checkpoint_sha256':file_hash(Path(a.shf_checkpoint)),'dsta_checkpoint_sha256':file_hash(Path(a.dsta_checkpoint)) if a.dsta_checkpoint else None,
            'code_sha256':file_hash(Path(__file__)),'raw_models':status,'upstream_training_overlap':'unknown for third-party pretrained weights; no BFMD local training'},
        'hits':[{'time':e['hit_timestamp'],'confidence':e['hit_confidence'],'label':None if e['stroke_type']=='unknown' else e['stroke_type']} for e in fused['events']],
        'trajectory':[{'time':f['source_time'],'xy':t['raw_xy']} for f,t in zip(dense['frames'],trajectory['frames'])],
        'poses':[{'time':f['source_time'],'persons':[{'bbox':p['bbox'],'keypoints':[[k['x'],k['y'],k['score']] for k in p['keypoints']]} for p in f['players'].values() if p]} for f in dense['frames']]}
    write(out/'predictions.json',packet);write(out/'unified_stroke_events.json',fused);print('BFMD_RAW_PREDICTIONS',len(hits),status,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--out',required=True);p.add_argument('--match-id',required=True);p.add_argument('--offset',type=float,required=True);p.add_argument('--alignment-status',default='nominal_source_time_offset')
    p.add_argument('--shf-checkpoint',required=True);p.add_argument('--dsta-checkpoint');a=p.parse_args();torch.set_num_threads(4);main(a)
