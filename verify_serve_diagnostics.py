"""Independent replay and factor-isolation verification for serve diagnostics."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from analysis_jobs import ROOT, file_hash
from bst_public_data import public_classifier

DEFAULT=ROOT.parent/'03_数据/击球标注/时间契约诊断/2026-10-13_发球窗口核对'


def verify_predictions(folder,classifier,roi=False):
    doc=json.loads((folder/'result.json').read_text());assert file_hash(folder/'protocol.json')==doc['protocol_sha256'];assert file_hash(folder/'inputs.npz')==doc['inputs_sha256'];checkpoint=ROOT/'models/bst/bst_0_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt';assert file_hash(checkpoint)==doc['model_sha256']
    for path,digest in doc['protected_sources_sha256'].items():assert file_hash(Path(path))==digest,path
    for path,digest in doc.get('tracker_outputs_sha256',{}).items():assert file_hash(Path(path))==digest,path
    original={};err=0.;rows=doc['rows']
    with np.load(folder/'inputs.npz',allow_pickle=False) as data:
        for i,(eid,variant) in enumerate(data['keys'].tolist()):
            row=next(r for r in rows if r['event_id']==eid and r['variant']==variant);j,p,s,n=data['joints_and_bones'][i],data['positions'][i],data['shuttle'][i],int(data['video_len'][i])
            with torch.inference_mode():scores=classifier.model(torch.from_numpy(j[None]),torch.from_numpy(s[None]),torch.tensor([n]))[0].softmax(-1).numpy()
            err=max(err,float(abs(scores-np.asarray(row['prediction']['scores'])).max()));assert classifier.classes[int(scores.argmax())]==row['prediction']['author_label']
            if roi:
                if eid in original:
                    jj,pp,nn=original[eid];assert np.array_equal(j,jj) and np.array_equal(p,pp) and n==nn,'Non-trajectory input changed'
                else:original[eid]=(j.copy(),p.copy(),n)
    assert err<=1e-6
    return {'folder':str(folder),'numeric_replays':i+1,'max_probability_error':err,'protected_sources_verified':True,'only_shuttle_numeric_input_changes_between_roi_conditions':True if roi else None,'summary':doc['summary']}


def verify_clips(folder):
    doc=json.loads((folder/'clip_manifest.json').read_text());assert doc['roi_xyxy']==[450,0,925,720];total=0
    for plan in doc['plans']:
        assert plan['source_start_frame']%16==0
        for variant,path in plan['outputs'].items():
            check=plan['lossless_checks'][variant];assert file_hash(Path(path))==check['source_clip_sha256'];cap=cv2.VideoCapture(path);digests=[];size=None
            while True:
                ok,frame=cap.read()
                if not ok:break
                size=[frame.shape[1],frame.shape[0]];digests.append(hashlib.sha256(frame.tobytes()).hexdigest())
            cap.release();assert digests==check['frame_pixel_sha256'];assert len(digests)==plan['source_end_frame_exclusive']-plan['source_start_frame'];assert size==check['decoded_size']
            if variant=='fixed_roi':assert size==[475,720],'Odd ROI width was truncated'
            total+=len(digests)
    return {'clips':6,'decoded_frames_pixel_hash_verified':total,'odd_roi_width_preserved':475,'tracknet8_inpaint16_phase_aligned':True,'evaluated_seconds':doc['evaluated_seconds'],'with_phase_context_seconds_per_spatial_condition':doc['with_phase_context_seconds']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--replay',action='store_true',required=True);p.add_argument('--write-report',action='store_true');a=p.parse_args();torch.set_num_threads(1);classifier=public_classifier();checks=[verify_predictions(a.root/'固定首拍窗口对照',classifier),verify_predictions(a.root/'固定目标ROI轨迹对照',classifier,roi=True)];clips=verify_clips(a.root/'固定目标ROI轨迹对照');result={'verified_at':datetime.now(timezone.utc).isoformat(),'checks':checks,'clips':clips,'total_numeric_replays':sum(x['numeric_replays'] for x in checks),'model_trained':False,'production_changed':False,'labels_changed':False}
    if a.write_report:(a.root/'independent_replay_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
