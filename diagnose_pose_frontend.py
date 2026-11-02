"""Fixed-box BGR replay vs RGB pose input contract diagnostic; no production edits.

Run --extract with .venv-pose, then --evaluate with .venv-vision. Existing boxes,
identities, windows, source frames and BST weights are fixed. RGB follows the
downloaded model's own pipeline.json Normalize.to_rgb=true contract.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'03_数据'
PARENT=DATA/'击球标注/输入修复诊断/2026-10-12_配对输入修复/result.json'
OUT=DATA/'击球标注/输入修复诊断/2026-10-13_姿态前端单因素诊断'


def save(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def extract():
    from rtmlib import RTMPose,YOLOX
    from analysis_jobs import registered_video,file_hash
    from analyze_pose import JOINTS,image_features
    from auto_players import appearance,appearance_distance,foot
    OUT.mkdir(parents=True,exist_ok=True)
    parent=json.loads(PARENT.read_text())
    pose=RTMPose(str(ROOT/'models/rtmpose-m-body7-256x192.onnx'),model_input_size=(192,256),backend='onnxruntime',device='cpu')
    det=YOLOX(str(ROOT/'models/yolox-m-humanart.onnx'),model_input_size=(640,640),score_thr=.4,backend='onnxruntime',device='cpu')
    jobs=defaultdict(list)
    for row in parent['rows']:jobs[row['job_id']].append(row)
    records=[];started=time.perf_counter()
    for job,rows in jobs.items():
        path=DATA/'专项分析'/job/'dense_pose.json';dense=json.loads(path.read_text())
        report=json.loads((path.parent/'report.json').read_text());source=registered_video(report['source_video_id'],allow_reference=True)['path']
        selected=set()
        for r in rows:
            for w in [r['frozen_window'],r['human_shifted_window']]:selected.update(range(w['start_frame'],w['end_frame']+1))
        candidate=json.loads(path.read_text())
        replay_frames={int(np.argmin(abs(np.asarray([f['source_time'] for f in dense['frames']])-r['human']['peak_time']))) for r in rows}
        replay=[];cap=cv2.VideoCapture(str(source));frame_index=0
        while True:
            ok,image=cap.read()
            if not ok:break
            i=frame_index;frame_index+=1
            if i not in selected:continue
            f=dense['frames'][i]
            sides=[s for s in ['near','far'] if f['players'].get(s)]
            if not sides:continue
            boxes=[f['players'][s]['bbox'] for s in sides]
            rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            points,scores=pose(rgb,bboxes=boxes)
            if i in replay_frames:
                bgr_points,bgr_scores=pose(image,bboxes=boxes)
                for k,side in enumerate(sides):
                    old=f['players'][side]
                    expected=np.asarray([[p['x'],p['y']] for p in old['keypoints']])
                    replay.append({'frame_index':i,'side':side,
                        'bgr_replay_max_point_error':float(np.max(abs(expected-bgr_points[k]))),
                        'rgb_vs_saved_mean_joint_displacement_px':float(np.linalg.norm(points[k]-expected,axis=1).mean()),
                        'bgr_replay_max_score_error':float(np.max(abs(np.asarray([p['score'] for p in old['keypoints']])-bgr_scores[k])))})
            for k,side in enumerate(sides):
                old=f['players'][side];p=points[k];conf=scores[k]
                inside=np.isfinite(p).all(1)&np.isfinite(conf)&(p[:,0]>=0)&(p[:,0]<dense['source_width'])&(p[:,1]>=0)&(p[:,1]<dense['source_height'])
                candidate['frames'][i]['players'][side]={**old,
                    'keypoints':[{'name':name,'x':float(xy[0]),'y':float(xy[1]),'score':float(c),'in_frame':bool(g)} for name,xy,c,g in zip(JOINTS,p,conf,inside)],
                    'features':image_features(p,np.where(inside,conf,0),.5)}
            if job.startswith('J_16b4') and i in [241,242,243,244,245]:
                all_boxes=np.asarray(det(image)).reshape(-1,4)
                save(OUT/'detector_checks'/f'frame_{i}.json',{'frame_index':i,'boxes':all_boxes.tolist(),'old_assignment':f['assignment'],
                    'old_near_bbox':f['players']['near']['bbox'] if f['players']['near'] else None})
            if i%200==0:print('POSE_RGB_PROGRESS',job,i,len(selected),flush=True)
        cap.release()
        candidate['diagnostic_pose_input_color']='RGB, fixed original BGR-run boxes/identity; selected model windows only'
        candidate['diagnostic_updated_frames']=sorted(selected)
        save(OUT/job/'rgb_dense_pose.json',candidate)
        save(OUT/job/'replay_checks.json',replay)
        records.append({'job_id':job,'updated_frame_count':len(selected),'dense_source_sha256':file_hash(path),
            'bgr_replay_max_point_error':max((r['bgr_replay_max_point_error'] for r in replay),default=0),
            'bgr_replay_max_score_error':max((r['bgr_replay_max_score_error'] for r in replay),default=0)})
    save(OUT/'extraction.json',{'created_at':datetime.now(timezone.utc).isoformat(),'jobs':records,'runtime_seconds':time.perf_counter()-started,
        'pose_model_sha256':file_hash(ROOT/'models/rtmpose-m-body7-256x192.onnx'),'production_changed':False})
    print('POSE_EXTRACTION_DONE',records,flush=True)


def evaluate():
    import torch
    from analysis_jobs import file_hash
    from specialist_bst import SpecialistBST
    from diagnose_input_repairs import prediction,metrics
    torch.set_num_threads(1)
    parent=json.loads(PARENT.read_text());models={};rows=[]
    for old in parent['rows']:
        job=old['job_id']
        if job not in models:
            root=DATA/'专项分析'/job
            original=json.loads((root/'dense_pose.json').read_text());rgb=json.loads((OUT/job/'rgb_dense_pose.json').read_text())
            trajectory=json.loads((root/'trajectory.json').read_text())
            models[job]=(SpecialistBST(original,trajectory),SpecialistBST(rgb,trajectory))
        bgr,rgb=models[job]
        order=list(old['variants']['frozen_replay']['player_order']);bgr.order=order;rgb.order=order
        y=old['human'];row={'job_id':job,'event_id':old['event_id'],'human':y,'variants':{},'known_target_logit_margins':{}}
        for name,clf in [('saved_bgr_pose',bgr),('rgb_contract_pose',rgb)]:
            v,q=clf.inputs(old['frozen_window']);p=prediction(clf,v,y['player'],old['automatic_candidate_player']);row['variants'][name]=p
            if name=='saved_bgr_pose':
                error=float(np.max(abs(np.asarray(p['scores'])-old['variants']['frozen_replay']['scores'])))
                if error>1e-5:raise ValueError('Frozen replay mismatch')
            prefix='Top_' if y['player']==clf.order[0] else 'Bottom_'
            from stroke_taxonomy import AUTHOR_MAP
            indices=[i for i,label in enumerate(clf.classes) if label.startswith(prefix)]
            truth=next(i for i in indices if AUTHOR_MAP[clf.classes[i].split('_',1)[-1]]==y['stroke_type'])
            other=max(i for i in indices if i!=truth)
            log=np.log(np.maximum(p['scores'],1e-30))
            row['known_target_logit_margins'][name]=float(log[truth]-max(log[i] for i in indices if i!=truth))
            row[name+'_input_quality']=q
        rows.append(row)
    result={'created_at':datetime.now(timezone.utc).isoformat(),'protocol':'Same saved boxes, IDs, original full windows, shuttle, BST weights. Only RTMPose input BGR->RGB changes. No label training.',
        'scope':'22 selected previously viewed development examples; pose truth not manually annotated.',
        'summary':{v:metrics(rows,v) for v in ['saved_bgr_pose','rgb_contract_pose']},
        'per_job':{job:{v:metrics([r for r in rows if r['job_id']==job],v) for v in ['saved_bgr_pose','rgb_contract_pose']} for job in models},
        'rows':rows,'script_sha256':file_hash(Path(__file__)),'production_changed':False}
    save(OUT/'result.json',result);print(json.dumps(result['summary'],ensure_ascii=False),flush=True)


def repair_gaps():
    """Recover only the three verified detector-present/follower-rejected frames."""
    import math
    from rtmlib import RTMPose
    from analyze_pose import JOINTS,image_features
    from auto_players import appearance,appearance_distance,foot
    job='J_16b4f26a07a14a59b76a77b8bb58dc78'
    root=DATA/'专项分析'/job
    dense=json.loads((root/'dense_pose.json').read_text());cfg=json.loads((root/'config.json').read_text())
    coarse=json.loads(Path(cfg['pose_file']).read_text())
    track=next(c for c in coarse['automatic_player_selection']['candidates'] if c['slot']=='near')
    ref=min(track['scout_observations'],key=lambda r:abs(r['source_time']-10.0))
    refheight=ref['bbox_normalized'][3]-ref['bbox_normalized'][1]
    last=dense['frames'][241];anchor=foot(np.asarray(last['players']['near']['bbox'])/[960,448,960,448])
    cap=cv2.VideoCapture(cfg['source_file'])
    pose=RTMPose(str(ROOT/'models/rtmpose-m-body7-256x192.onnx'),model_input_size=(192,256),backend='onnxruntime',device='cpu')
    candidate=json.loads((root/'dense_pose.json').read_text())
    rgb_candidate=json.loads((OUT/job/'rgb_dense_pose.json').read_text())
    evidence=[]
    for i in [242,243,244]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,image=cap.read()
        if not ok:raise ValueError('Missing source frame')
        check=json.loads((OUT/'detector_checks'/f'frame_{i}.json').read_text())
        box=np.asarray(check['boxes'][0]);norm=box/[960,448,960,448]
        gap=dense['frames'][i]['source_time']-last['source_time'];gate=min(.18,.06+.3*gap)
        distance=float(np.linalg.norm(foot(norm)-anchor));color=appearance(image,box)
        identity=min(appearance_distance(np.asarray(r),color) for r in track['identity_reference_histograms'])
        scale=abs(math.log((norm[3]-norm[1])/refheight))
        evidence.append({'frame_index':i,'detector_box_index':0,'bbox':box.tolist(),'foot_distance':distance,
            'foot_distance_gate':gate,'scale_log':scale,'scale_log_gate':math.log(2.2),
            'identity_distance':identity,'identity_distance_gate':.4,
            'passes_all_except_position':bool(scale<=math.log(2.2) and identity<=.4),'rejected_by_position':bool(distance>gate),
            'target_person_visually_reviewed':True,'human_user_confirmed':False})
        if dense['frames'][i]['players']['near'] is not None:raise ValueError('Not an originally missing frame')
        for value,color_image in [(candidate,image),(rgb_candidate,cv2.cvtColor(image,cv2.COLOR_BGR2RGB))]:
            points,scores=pose(color_image,bboxes=[box]);p,conf=points[0],scores[0]
            inside=(p[:,0]>=0)&(p[:,0]<960)&(p[:,1]>=0)&(p[:,1]<448)
            value['frames'][i]['players']['near']={'track_id':job+'-near','bbox':box.tolist(),
                'keypoints':[{'name':name,'x':float(xy[0]),'y':float(xy[1]),'score':float(c),'in_frame':bool(g)} for name,xy,c,g in zip(JOINTS,p,conf,inside)],
                'features':image_features(p,np.where(inside,conf,0),.5),
                'identity_status':'diagnostic_visible_detector_box_recovery_not_user_confirmed'}
    cap.release()
    save(OUT/'three_missing_frames_rejection.json',evidence)
    save(OUT/job/'bgr_gap_repaired_dense_pose.json',candidate)
    save(OUT/job/'rgb_gap_repaired_dense_pose.json',rgb_candidate)
    print('GAP_REPAIR',evidence,flush=True)


def evaluate_gaps():
    import torch
    from specialist_bst import SpecialistBST
    from diagnose_input_repairs import prediction,metrics
    from stroke_taxonomy import AUTHOR_MAP
    torch.set_num_threads(1)
    parent=json.loads(PARENT.read_text());job='J_16b4f26a07a14a59b76a77b8bb58dc78';root=DATA/'专项分析'/job
    trajectory=json.loads((root/'trajectory.json').read_text())
    models={name:SpecialistBST(json.loads(path.read_text()),trajectory) for name,path in [
        ('saved_bgr_pose',root/'dense_pose.json'),('bgr_three_frame_gap_repair',OUT/job/'bgr_gap_repaired_dense_pose.json'),
        ('rgb_three_frame_gap_repair',OUT/job/'rgb_gap_repaired_dense_pose.json')]}
    rows=[]
    for old in [r for r in parent['rows'] if r['job_id']==job]:
        y=old['human'];row={'job_id':job,'event_id':old['event_id'],'human':y,'variants':{},'known_target_logit_margins':{}}
        for name,clf in models.items():
            clf.order=list(old['variants']['frozen_replay']['player_order'])
            values,_=clf.inputs(old['frozen_window']);p=prediction(clf,values,y['player'],old['automatic_candidate_player']);row['variants'][name]=p
            prefix='Top_' if y['player']==clf.order[0] else 'Bottom_'
            ids=[i for i,label in enumerate(clf.classes) if label.startswith(prefix)]
            truth=next(i for i in ids if AUTHOR_MAP[clf.classes[i].split('_',1)[-1]]==y['stroke_type'])
            log=np.log(np.maximum(p['scores'],1e-30));row['known_target_logit_margins'][name]=float(log[truth]-max(log[i] for i in ids if i!=truth))
        rows.append(row)
    result={'protocol':'Restore only f242-244 target pose from existing detector boxes; BGR branch isolates follower-induced missingness, RGB branch combines contract repair. Same original windows, trajectory, model. Detector boxes are visually reviewed diagnostic candidates, not user-labelled pose truth.',
        'summary':{v:metrics(rows,v) for v in models},'rows':rows,'production_changed':False}
    save(OUT/'three_frame_gap_bst_result.json',result);print(json.dumps(result['summary'],ensure_ascii=False),flush=True)


def contact_plots():
    from analysis_jobs import registered_video
    parent=json.loads(PARENT.read_text());models={};count=0
    for row in parent['rows']:
        y=row['human'];b=row['variants']['frozen_replay']
        if not(abs(y['peak_time']-row['frozen_window']['peak_time'])<=.125 and b['mapped_actor']==y['player'] and b['code']!=y['stroke_type']):continue
        job=row['job_id'];root=DATA/'专项分析'/job
        dense=json.loads((root/'dense_pose.json').read_text());rgb=json.loads((OUT/job/'rgb_dense_pose.json').read_text())
        report=json.loads((root/'report.json').read_text());source=registered_video(report['source_video_id'],allow_reference=True)['path']
        times=np.asarray([f['source_time'] for f in dense['frames']]);center=int(np.argmin(abs(times-y['peak_time'])))
        frames=[max(0,min(len(times)-1,center+off)) for off in [-6,-3,0,3,6]]
        boxes=[p['bbox'] for i in frames if (p:=dense['frames'][i]['players'].get(y['player']))]
        left=max(0,int(min(v[0] for v in boxes)-60));right=min(dense['source_width'],int(max(v[2] for v in boxes)+60))
        top=max(0,int(min(v[1] for v in boxes)-35));bottom=min(dense['source_height'],int(max(v[3] for v in boxes)+25))
        cap=cv2.VideoCapture(str(source));images=[]
        for i in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,image=cap.read()
            if not ok:raise ValueError('Missing plot source frame')
            images.append(image)
        cap.release();panels=[]
        for name,value,color in [('BGR saved',dense,(0,220,0)),('RGB fixed box',rgb,(255,0,255))]:
            tiles=[]
            for i,image in zip(frames,images):
                im=image.copy();p=value['frames'][i]['players'].get(y['player'])
                if p:
                    xy=[(round(j['x']),round(j['y'])) for j in p['keypoints']]
                    for u,v in [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]:cv2.line(im,xy[u],xy[v],color,2)
                    for k,point in enumerate(xy):cv2.circle(im,point,4,color,-1)
                    for k in [9,10]:cv2.putText(im,str(k),(xy[k][0]+4,xy[k][1]-4),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
                im=im[top:bottom,left:right];scale=min(280/im.shape[1],350/im.shape[0]);im=cv2.resize(im,(round(im.shape[1]*scale),round(im.shape[0]*scale)))
                tile=np.full((390,300,3),240,dtype='uint8');tile[35:35+im.shape[0],:im.shape[1]]=im
                cv2.putText(tile,f'{name} f{i} {times[i]:.3f}s',(4,22),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1)
                if p is None:cv2.putText(tile,'TARGET MISSING',(4,370),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2)
                tiles.append(tile)
            panels.append(np.hstack(tiles))
        path=OUT/'contact_overlays'/f'{job}_{row["event_id"]}.jpg';path.parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(path),np.vstack(panels));count+=1
    print('CONTACT_PLOTS',count,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--extract',action='store_true');parser.add_argument('--evaluate',action='store_true');parser.add_argument('--repair-gaps',action='store_true');parser.add_argument('--evaluate-gaps',action='store_true');parser.add_argument('--contact-plots',action='store_true');a=parser.parse_args()
    if a.extract:extract()
    if a.evaluate:evaluate()
    if a.repair_gaps:repair_gaps()
    if a.evaluate_gaps:evaluate_gaps()
    if a.contact_plots:contact_plots()
