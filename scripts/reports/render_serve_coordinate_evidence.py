"""Sequential source frames with saved, unmodified pose/ball coordinates."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/reports/
import json
from pathlib import Path

import cv2
import numpy as np

from analysis_jobs import DATA, atomic_json, file_hash
from audit_serve_windows import OUT, JOB, tile, sheet


def main():
    visual=json.loads((OUT/'visual_review_manifest.json').read_text());dense_path=DATA/'专项分析'/JOB/'dense_pose.json';traj_path=DATA/'专项分析'/JOB/'trajectory.json';dense=json.loads(dense_path.read_text());traj=json.loads(traj_path.read_text());times=np.asarray([f['source_time'] for f in dense['frames']]);cases={33.3:[996,998,1000,1002],47.5:[1422,1424,1426,1428],93.5:[2802,2804,2806,2808]};needed={i for ids in cases.values() for i in ids};frames={};cap=cv2.VideoCapture(visual['source_video']);idx=0
    while True:
        ok,image=cap.read()
        if not ok:break
        if idx in needed:
            assert abs(cap.get(cv2.CAP_PROP_POS_MSEC)/1000-times[idx])<1e-5;frames[idx]=image.copy()
        if idx>max(needed):break
        idx+=1
    cap.release();rows=[]
    for target,indices in cases.items():
        tiles=[]
        for i in indices:
            image=frames[i];b=traj['frames'][i];points={}
            for side,color in [('near',(255,220,0)),('far',(0,180,255))]:
                p=dense['frames'][i]['players'].get(side)
                if not p:continue
                x1,y1,x2,y2=map(round,p['bbox']);cv2.rectangle(image,(x1,y1),(x2,y2),color,2);points[side]=[]
                for j in [9,10]:
                    k=p['keypoints'][j];xy=(round(k['x']),round(k['y']));cv2.circle(image,xy,7,color,2);points[side].append({'xy':[k['x'],k['y']],'score':k['score']})
            xy=b['raw_xy'] if b['raw_visible'] else b.get('rectified_xy');label='RAW' if b['raw_visible'] else 'INPAINT'
            if xy:
                pt=tuple(map(round,xy));cv2.circle(image,pt,12,(255,0,255) if label=='RAW' else (0,255,255),3);cv2.putText(image,label+' '+str(xy),(max(0,pt[0]-30),max(18,pt[1]-17)),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,0,255) if label=='RAW' else (0,255,255),2)
            tiles.append(tile(image,f'{times[i]:.4f}s f{i} ball={label} {xy}',640,360));rows.append({'target_serve':target,'frame':i,'source_time':float(times[i]),'raw_visible':b['raw_visible'],'raw_xy':b['raw_xy'],'inpaint_requested':b['inpaint_requested'],'rectified_xy':b.get('rectified_xy'),'wrist_coordinates':points})
        sheet(tiles,OUT/f'{target:.4f}_saved_coordinate_evidence.jpg',columns=2)
    atomic_json(OUT/'saved_coordinate_evidence.json',{'scope':'Display saved coordinates; no new model inference or coordinate ground truth. Magenta=raw model ball; yellow=inpainted; cyan=near wrists; orange=far wrists.','dense_pose_sha256':file_hash(dense_path),'trajectory_sha256':file_hash(traj_path),'rows':rows,'production_changed':False})
    print('Coordinate evidence saved',flush=True)


if __name__=='__main__':main()
