"""Sequential-frame visual audit for the six already-labelled short serves.

Preparation does not run any model or modify labels. New interventions, if
requested, use a separate fixed protocol and saved numeric inputs.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from analysis_jobs import DATA, ROOT, atomic_json, file_hash, registered_video
from stroke_review import load_source, read_labels, folder_for

JOB='J_e23c1dd9297d4bfca7c91d1d95065b68'
OUT=DATA/'击球标注/时间契约诊断/2026-10-13_发球窗口核对'


def tile(image,label,width=320,height=180):
    out=np.zeros((height+26,width,3),np.uint8);scale=min(width/image.shape[1],height/image.shape[0]);im=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)));out[26:26+im.shape[0],:im.shape[1]]=im;cv2.putText(out,label,(4,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1);return out


def sheet(tiles,path,columns=4):
    while len(tiles)%columns:tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(path),np.vstack([np.hstack(tiles[i:i+columns]) for i in range(0,len(tiles),columns)]))


def prepare(out,clips=False):
    out.mkdir(parents=True,exist_ok=True);report,events,result=load_source(JOB);doc=read_labels(JOB,report['source_sha256']);source=registered_video(report['source_video_id'],allow_reference=True)['path'];dense_path=DATA/'专项分析'/JOB/'dense_pose.json';dense=json.loads(dense_path.read_text());times=np.asarray([f['source_time'] for f in dense['frames']]);labels=sorted([h for h in doc['labels'].values() if h['stroke_type']=='serve_short'],key=lambda h:h['peak_time']);selections={};needed=set()
    for human in labels:
        center=int(np.argmin(abs(times-human['peak_time'])));context=[int(np.argmin(abs(times-t))) for t in np.arange(human['peak_time']-5,human['peak_time']+5.001,.5)];local=list(range(max(0,center-8),min(len(times),center+11)));selections[human['event_id']]={'human':human,'center_frame':center,'context':context,'local':local};needed.update(context+local)
    cap=cv2.VideoCapture(str(source));frames={};pts=[];index=0
    while True:
        ok,image=cap.read()
        if not ok:break
        stamp=cap.get(cv2.CAP_PROP_POS_MSEC)/1000;pts.append(stamp)
        if index in needed:frames[index]=image.copy()
        index+=1
    cap.release();pts=np.asarray(pts)
    if len(pts)!=len(times) or max(abs(pts-times))>1e-5:raise ValueError('Sequential source PTS mismatches pose cache')
    for eid,item in selections.items():
        human=item['human'];prefix=f"{human['peak_time']:.4f}_{eid}";context=[];local=[]
        for i in item['context']:
            image=frames[i].copy()
            for side,color in [('near',(255,220,0)),('far',(0,180,255))]:
                p=dense['frames'][i]['players'].get(side)
                if p:
                    x1,y1,x2,y2=map(round,p['bbox']);cv2.rectangle(image,(x1,y1),(x2,y2),color,2);cv2.putText(image,side,(x1,max(y1-4,14)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
            context.append(tile(image,f'{pts[i]:.4f}s f{i}'))
        for i in item['local']:
            image=frames[i];p=dense['frames'][i]['players'].get(human['player'])
            if p:
                x1,y1,x2,y2=p['bbox'];pad=max(35,(x2-x1)*.4);x1=max(0,round(x1-pad));x2=min(image.shape[1],round(x2+pad));y1=max(0,round(y1-pad));y2=min(image.shape[0],round(y2+pad));image=image[y1:y2,x1:x2]
            local.append(tile(image,f'{pts[i]:.4f}s f{i}',240,250))
        sheet(context,out/(prefix+'_10s_context.jpg'));sheet(local,out/(prefix+'_contact_strip.jpg'),columns=5)
        item['context_image']=prefix+'_10s_context.jpg';item['contact_image']=prefix+'_contact_strip.jpg'
        if clips:
            ffmpeg=next((ROOT/'.venv-pose/lib').glob('python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*'))
            lo=max(0,human['peak_time']-5);path=out/(prefix+'_10s_context.mp4');subprocess.run([str(ffmpeg),'-y','-v','error','-ss',str(lo),'-i',str(source),'-t','10','-c:v','libx264','-preset','veryfast','-crf','23','-c:a','aac',str(path)],check=True);item['context_video']=path.name;item['video_source_start']=lo
    random_seek=[];cap=cv2.VideoCapture(str(source))
    for item in selections.values():
        i=item['center_frame'];cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();random_seek.append({'frame_index':i,'stored_source_time':float(times[i]),'sequential_source_pts':float(pts[i]),'random_seek_opencv_pts':cap.get(cv2.CAP_PROP_POS_MSEC)/1000})
    cap.release()
    result={'version':'six-short-serve-sequential-visual-audit-v1','created_at':datetime.now(timezone.utc).isoformat(),'job_id':JOB,'source_video':str(source),'source_sha256':file_hash(source),'annotations_sha256':file_hash(folder_for(JOB)/'annotations.json'),'dense_pose_sha256':file_hash(dense_path),'sequential_pts_max_error':float(max(abs(pts-times))),'random_seek_audit':random_seek,'serves':selections,'scope':'Existing human labels retained; visual strips use sequential source frame indices/PTS. Context does not create new ground truth.','human_data_changed':False,'model_trained':False}
    atomic_json(out/'visual_review_manifest.json',result);np.savez_compressed(out/'sequential_source_pts.npz',pts=pts,stored_pose_pts=times)
    print(json.dumps({'directory':str(out),'serves':[(v['human']['peak_time'],v['human']['player']) for v in selections.values()],'sequential_pts_max_error':result['sequential_pts_max_error']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--clips',action='store_true');a=p.parse_args();prepare(a.output,a.clips)
