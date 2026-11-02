"""Fixed visual ROI on three predeclared first-serve windows, isolated outputs.

Read the original VFR sequentially. Lossless FFV1 crop + matched full-width clip
control. TrackNet/Inpaint weights unchanged; original pose/BST/windows fixed.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import torch

from analysis_jobs import DATA, ROOT, atomic_json, file_hash
from audit_serve_first_hit import OUT as FIRST
from audit_serve_windows import OUT as VISUAL, JOB
from audit_temporal_contract import pred
from specialist_bst import SpecialistBST
from specialist_model import BST_NAME

OUT=VISUAL/'固定目标ROI轨迹对照'
ROI=[450,0,925,720]


def digest_frame(frame):return hashlib.sha256(frame.tobytes()).hexdigest()


def prepare_clips(out,rows,source,dense):
    """Align TrackNet8/Inpaint16 phases; do not random-seek source frames."""
    ffmpeg=next((ROOT/'.venv-pose/lib').glob('python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*'))
    plans=[];times=np.asarray([f['source_time'] for f in dense['frames']]);width,height=dense['source_width'],dense['source_height'];assert height==720 and width>=925
    for row in rows:
        w=row['window'];start=(w['start_frame']//16)*16;end=min(len(times),((w['end_frame']+1+15)//16)*16)
        plans.append({'event_id':row['event_id'],'serve_time':row['human']['peak_time'],'source_start_frame':start,'source_end_frame_exclusive':end,
                      'window':w,'source_times':times[start:end].tolist(),'tracknet8_inpaint16_phase_aligned':start%16==0,'outputs':{}})
    processes=[];expected={}
    for plan in plans:
        for variant,size in [('matched_full_width',(width,height)),('fixed_roi',(ROI[2]-ROI[0],ROI[3]-ROI[1]))]:
            directory=out/plan['event_id']/variant;directory.mkdir(parents=True,exist_ok=True);path=directory/'source_ffv1.avi';plan['outputs'][variant]=str(path)
            command=[str(ffmpeg),'-y','-v','error','-f','rawvideo','-pixel_format','bgr24','-video_size',f'{size[0]}x{size[1]}','-framerate','30','-i','pipe:0','-an','-c:v','ffv1','-level','3','-pix_fmt','bgr0',str(path)]
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE);processes.append((plan,variant,path,process));expected[str(path)]=[]
    cap=cv2.VideoCapture(str(source));i=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        for plan,variant,path,process in processes:
            if plan['source_start_frame']<=i<plan['source_end_frame_exclusive']:
                assert abs(cap.get(cv2.CAP_PROP_POS_MSEC)/1000-times[i])<1e-5
                pixels=frame if variant=='matched_full_width' else frame[ROI[1]:ROI[3],ROI[0]:ROI[2]]
                pixels=np.ascontiguousarray(pixels);process.stdin.write(pixels.tobytes());expected[str(path)].append(digest_frame(pixels))
        i+=1
    cap.release()
    for plan,variant,path,process in processes:
        process.stdin.close();error=process.stderr.read().decode();code=process.wait()
        if code:raise RuntimeError(error)
        cap=cv2.VideoCapture(str(path));actual=[];sizes=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            actual.append(digest_frame(frame));sizes.append([frame.shape[1],frame.shape[0]])
        cap.release();assert actual==expected[str(path)],'FFV1 pixels/frame alignment changed'
        plan.setdefault('lossless_checks',{})[variant]={'decoded_frames':len(actual),'all_frames_pixel_identical':True,'decoded_size':sizes[0],'frame_pixel_sha256':actual,'source_clip_sha256':file_hash(path)}
    atomic_json(out/'clip_manifest.json',{'plans':plans,'source_sha256':file_hash(source),'roi_xyxy':ROI,'temporal_resampling':False,'source_index_mapping':'source_index=clip_index+source_start_frame; true source PTS explicitly recorded','fps_of_lossless_container':30,'evaluated_seconds':sum((r['window']['end_frame']-r['window']['start_frame']+1)/30 for r in rows),'with_phase_context_seconds':sum((p['source_end_frame_exclusive']-p['source_start_frame'])/30 for p in plans)})
    return plans


def run_tracker(out,plan,variant):
    directory=out/plan['event_id']/variant;source=Path(plan['outputs'][variant]);config={'source_file':str(source),'source_sha256':file_hash(source),'specialist_folder':str(directory/'tracker')};atomic_json(directory/'config.json',config);path=directory/'tracker/trajectory.json'
    if path.is_file():
        cached=json.loads(path.read_text());assert cached['source_sha256']==config['source_sha256'];assert cached['code_sha256']==file_hash(ROOT/'specialist_tracknet.py');return cached
    process=subprocess.run([str(ROOT/'.venv-vision/bin/python'),str(ROOT/'specialist_tracknet.py'),'--config',str(directory/'config.json')],capture_output=True,text=True,timeout=420)
    (directory/'tracknet.log').write_text(process.stdout+process.stderr)
    if process.returncode:raise RuntimeError('TrackNet failed: '+str(directory/'tracknet.log'))
    result=json.loads(path.read_text());assert result['source_sha256']==config['source_sha256'];assert result['frame_count']==plan['source_end_frame_exclusive']-plan['source_start_frame'];print('TRACKER',plan['serve_time'],variant,result['runtime_seconds'],flush=True);return result


def replay(out):
    doc=json.loads((out/'result.json').read_text());from bst_public_data import public_classifier
    classifier=public_classifier();assert file_hash(ROOT/'models/bst'/BST_NAME)==doc['model_sha256'];assert file_hash(out/'inputs.npz')==doc['inputs_sha256'];err=0.
    with np.load(out/'inputs.npz',allow_pickle=False) as data:
        for i,(eid,variant) in enumerate(data['keys'].tolist()):
            row=next(r for r in doc['rows'] if r['event_id']==eid and r['variant']==variant);classifier.order=['far','near'];v=(data['joints_and_bones'][i],data['positions'][i],data['shuttle'][i],int(data['video_len'][i]));p=pred(classifier,v,row['human']);err=max(err,max(abs(a-b) for a,b in zip(p['scores'],row['prediction']['scores'])))
    assert err<=1e-6
    for path,digest in doc['protected_sources_sha256'].items():assert file_hash(Path(path))==digest
    for path,digest in doc['tracker_outputs_sha256'].items():assert file_hash(Path(path))==digest
    print(json.dumps({'replayed':i+1,'max_probability_error':err,'original_sources_unchanged':True,'tracker_outputs_verified':True}),flush=True)


def main(out):
    if (out/'result.json').is_file():replay(out);return
    out.mkdir(parents=True,exist_ok=True);first=json.loads((FIRST/'result.json').read_text());selected=[r for r in first['rows'] if r['variant']=='author_first_hit_fallback'];assert len(selected)==3
    protocol={'declared_at':datetime.now(timezone.utc).isoformat(),'roi_xyxy':ROI,'selected_by':'Root visual inspection of original sequential frame, fixed before inference; target near~x520/far~x820, left neighboring court excluded.',
        'limitations':'Visual ROI is not court ground truth; may clip real shuttle leaving the ROI. Crop changes tracking scale and context. Paired short full-width control separates ROI from shorter-video median-background/clip-boundary effects.',
        'variants':['original_full_video_cache','matched_full_width_short_clip','fixed_roi_short_clip'],'windows':[{'event_id':r['event_id'],'human':r['human'],'window':r['window']} for r in selected],
        'scope':'Three preselected first serves, same pose/BST/window per comparison. No ROI/parameter search or generalized accuracy claim. TrackNet8/Inpaint16 phase context may extend evaluated support by <16 frames per edge.'}
    if not (out/'protocol.json').is_file():atomic_json(out/'protocol.json',protocol);shutil.copy2(Path(__file__),out/'executed_script.py')
    else:assert json.loads((out/'protocol.json').read_text())['roi_xyxy']==ROI
    visual=json.loads((VISUAL/'visual_review_manifest.json').read_text());source=Path(visual['source_video']);dense_path=DATA/'专项分析'/JOB/'dense_pose.json';traj_path=DATA/'专项分析'/JOB/'trajectory.json';dense=json.loads(dense_path.read_text());original=json.loads(traj_path.read_text());protected={str(p):file_hash(p) for p in [source,dense_path,traj_path,DATA/'击球标注'/JOB/'annotations.json',ROOT/'models/bst'/BST_NAME,ROOT/'models/tracknetv3/TrackNet_best.pt',ROOT/'models/tracknetv3/InpaintNet_best.pt']}
    if (out/'clip_manifest.json').is_file():plans=json.loads((out/'clip_manifest.json').read_text())['plans']
    else:plans=prepare_clips(out,selected,source,dense)
    rows=[];values=[];keys=[];tracker_hashes={};first_lookup={r['event_id']:r for r in selected};torch.set_num_threads(1)
    for plan in plans:
        base=first_lookup[plan['event_id']];human=base['human'];window=base['window'];trajectories={'original_full_video_cache':original}
        for kind,name in [('matched_full_width','matched_full_width_short_clip'),('fixed_roi','fixed_roi_short_clip')]:
            tracked=run_tracker(out,plan,kind);tracked_path=out/plan['event_id']/kind/'tracker/trajectory.json';tracker_hashes[str(tracked_path)]=file_hash(tracked_path);mapped=deepcopy(original);mapped['engine_version']='isolated-three-serve-roi-diagnostic';mapped['diagnostic_only']=True
            for f in tracked['frames']:
                i=f['frame_index']+plan['source_start_frame'];r={**f,'frame_index':i,'source_time_nominal':dense['frames'][i]['source_time']}
                for key in ['raw_xy','rectified_xy']:
                    xy=f.get(key);r[key]=[xy[0]+(ROI[0] if kind=='fixed_roi' else 0),xy[1]+(ROI[1] if kind=='fixed_roi' else 0)] if xy is not None and any(xy) else None
                mapped['frames'][i]=r
            mapped_path=out/plan['event_id']/kind/'mapped_full_source_trajectory.json';atomic_json(mapped_path,mapped);tracker_hashes[str(mapped_path)]=file_hash(mapped_path);trajectories[name]=mapped
        for variant,traj in trajectories.items():
            classifier=SpecialistBST(dense,traj);classifier.order=['far','near'];v,q=classifier.inputs(window);p=pred(classifier,v,human)
            if variant=='original_full_video_cache':assert max(abs(a-b) for a,b in zip(p['scores'],base['prediction']['scores']))<=1e-6
            c=window['frame_index'];contact=[{'frame':i,'source_time':dense['frames'][i]['source_time'],'raw_xy':traj['frames'][i]['raw_xy'],'raw_visible':traj['frames'][i]['raw_visible'],'rectified_xy':traj['frames'][i]['rectified_xy'],'inpaint_requested':traj['frames'][i]['inpaint_requested']} for i in range(c-3,c+4)]
            rows.append({'event_id':plan['event_id'],'variant':variant,'human':human,'window':window,'prediction':p,'input_quality':q,'contact_neighborhood':contact,'type_label_match':p['stroke_type']==human['stroke_type'],'joint_label_match_under_unverified_mapping':p['stroke_type']==human['stroke_type'] and p['mapped_actor']==human['player']});values.append(v);keys.append((plan['event_id'],variant));print('BST',human['peak_time'],variant,p['author_label'],round(max(p['scores']),4),flush=True)
    assert all(file_hash(Path(p))==digest for p,digest in protected.items())
    np.savez_compressed(out/'inputs.npz',keys=np.asarray(keys),joints_and_bones=np.stack([v[0] for v in values]),positions=np.stack([v[1] for v in values]),shuttle=np.stack([v[2] for v in values]),video_len=np.asarray([v[3] for v in values]))
    summary={variant:{'total':sum(r['variant']==variant for r in rows),'type_label_matches':sum(r['type_label_match'] for r in rows if r['variant']==variant),'joint_label_matches_under_unverified_mapping':sum(r['joint_label_match_under_unverified_mapping'] for r in rows if r['variant']==variant)} for variant in protocol['variants']}
    result={'version':'three-fixed-serve-visual-roi-tracknet-control-v1','created_at':datetime.now(timezone.utc).isoformat(),'scope':protocol['scope'],'roi_xyxy':ROI,'summary':summary,'rows':rows,'model_sha256':file_hash(ROOT/'models/bst'/BST_NAME),'inputs_sha256':file_hash(out/'inputs.npz'),'protocol_sha256':file_hash(out/'protocol.json'),'protected_sources_sha256':protected,'tracker_outputs_sha256':tracker_hashes,'original_sources_unchanged':True,'model_trained':False,'weights_changed':False,'production_changed':False};atomic_json(out/'result.json',result);print('SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--replay',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    replay(a.output) if a.replay else main(a.output)
