"""Original vs matched-control vs fixed-ROI saved coordinates on source frames."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/reports/
import json
from pathlib import Path
import cv2
import numpy as np
from analysis_jobs import DATA
from audit_serve_windows import OUT as VISUAL, JOB, tile
from audit_serve_roi_tracknet import OUT, ROI


def main():
    result=json.loads((OUT/'result.json').read_text());visual=json.loads((VISUAL/'visual_review_manifest.json').read_text());dense=json.loads((DATA/'专项分析'/JOB/'dense_pose.json').read_text());original=json.loads((DATA/'专项分析'/JOB/'trajectory.json').read_text());cases={r['event_id']:r for r in result['rows'] if r['variant']=='original_full_video_cache'};indices={r['window']['frame_index']+offset for r in cases.values() for offset in [-2,0,2]};cap=cv2.VideoCapture(visual['source_video']);frames={};i=0
    while True:
        ok,im=cap.read()
        if not ok:break
        if i in indices:
            assert abs(cap.get(cv2.CAP_PROP_POS_MSEC)/1000-dense['frames'][i]['source_time'])<1e-5;frames[i]=im.copy()
        if i>max(indices):break
        i+=1
    cap.release()
    for eid,case in cases.items():
        trajectories=[('Original full video',original),('Matched short full width',json.loads((OUT/eid/'matched_full_width/mapped_full_source_trajectory.json').read_text())),('Fixed ROI',json.loads((OUT/eid/'fixed_roi/mapped_full_source_trajectory.json').read_text()))];rows=[]
        for offset in [-2,0,2]:
            i=case['window']['frame_index']+offset;tiles=[]
            for name,traj in trajectories:
                im=frames[i].copy();b=traj['frames'][i];cv2.rectangle(im,(ROI[0],ROI[1]),(ROI[2],ROI[3]-1),(0,255,0),2)
                p=dense['frames'][i]['players']['near']
                if p:
                    for k in [9,10]:cv2.circle(im,(round(p['keypoints'][k]['x']),round(p['keypoints'][k]['y'])),7,(255,220,0),2)
                xy=b['raw_xy'] if b['raw_visible'] else b.get('rectified_xy');kind='RAW' if b['raw_visible'] else 'INPAINT'
                if xy:
                    pt=tuple(map(round,xy));color=(255,0,255) if kind=='RAW' else (0,255,255);cv2.circle(im,pt,12,color,3);cv2.putText(im,kind+str(xy),(max(0,pt[0]-30),max(18,pt[1]-17)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
                tiles.append(tile(im,f'{name} f{i} {kind} {xy}',640,360))
            rows.append(np.hstack(tiles))
        cv2.imwrite(str(OUT/f"{case['human']['peak_time']:.4f}_original_control_roi_evidence.jpg"),np.vstack(rows))
    print('Three original/control/ROI coordinate evidence sheets saved.',flush=True)


if __name__=='__main__':main()
