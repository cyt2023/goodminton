"""Reuse the project's YOLOX/RTMPose/scout on RAW video only. No BFMD imports."""
import sys,argparse,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from rtmlib import YOLOX,RTMPose
from analyze_pose import get_weights,JOINTS,sha256
from auto_players import ScoutTracks

def main(video,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);weights=get_weights()
    detector=YOLOX(weights['detector']['file'],model_input_size=(640,640),score_thr=.4,backend='onnxruntime',device='cpu')
    pose=RTMPose(weights['pose']['file'],model_input_size=(192,256),backend='onnxruntime',device='cpu')
    cap=cv2.VideoCapture(str(video));fps=cap.get(5);w,h=int(cap.get(3)),int(cap.get(4));total=int(cap.get(7));scout=ScoutTracks(1.)
    for i in range(0,total,max(1,round(fps))):
        cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,image=cap.read()
        if ok:scout.observe(image,np.asarray(detector(image)).reshape(-1,4),i/fps)
    follower,selection=scout.select('BFMD_RAW_ONLY',out,(w,h));cap.set(cv2.CAP_PROP_POS_FRAMES,0);frames=[];started=time.perf_counter()
    while True:
        ok,image=cap.read()
        if not ok:break
        i=len(frames);t=i/fps;boxes=np.asarray(detector(image)).reshape(-1,4);assignments,reasons=follower.select_boxes(image,boxes,t);players={'near':None,'far':None}
        if assignments:
            points,scores=pose(image,bboxes=[boxes[j] for j in assignments.values()])
            for k,(side,j) in enumerate(assignments.items()):
                players[side]={'bbox':boxes[j].tolist(),'keypoints':[{'x':float(p[0]),'y':float(p[1]),'score':float(s),'name':name,'in_frame':bool(0<=p[0]<w and 0<=p[1]<h)} for p,s,name in zip(points[k],scores[k],JOINTS)]}
        frames.append({'frame_index':i,'source_time':t,'players':players})
    cap.release();result={'source_file':str(Path(video).resolve()),'source_sha256':sha256(video),'source_width':w,'source_height':h,'sample_fps':fps,'frames':frames,'models':weights,'seconds':time.perf_counter()-started,'annotation_inputs':[],'engine_version':'existing_yolox_rtmpose_scout_raw_benchmark'}
    (out/'dense_pose.json').write_text(json.dumps(result,allow_nan=False));print('RAW_POSE_DONE',len(frames),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--video',required=True);a.add_argument('--out',required=True);v=a.parse_args();main(Path(v.video),v.out)
