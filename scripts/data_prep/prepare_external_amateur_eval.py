"""Freeze all publicly annotated amateur singles rallies before local inference."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/data_prep/
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2

from analysis_jobs import DATA, atomic_json, file_hash
from specialist_model import ROOT

OUT=DATA/'外部标注测试/2026-10-21/Amateur_hit7'
AUTHOR=DATA/'公开训练数据/Badminton_hit_detection/author_files/am_singles'


def main():
    if (OUT/'protocol.json').exists():raise FileExistsError('Preserve frozen evaluation')
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in sorted(AUTHOR.glob('*/player_hit/*.csv')):
        match=path.parent.parent.name; filename=path.name.removesuffix('_player_hit.csv')
        source=path.parent.parent/'rally_video'/filename
        bbox=path.parent.parent/'bbox_id'/(Path(filename).stem+'_gtbboxid.csv')
        ball=path.parent.parent/'ball_trajectory'/(Path(filename).stem+'_ball.csv')
        assert source.is_file() and bbox.is_file() and ball.is_file()
        cap=cv2.VideoCapture(str(source));n=int(cap.get(7));fps=cap.get(5)
        w,h=int(cap.get(3)),int(cap.get(4));cap.release()
        labels=list(csv.DictReader(path.open()));assert len(labels)==n
        assert [int(r['frame']) for r in labels]==list(range(n))
        contacts=[{'frame':int(r['frame']),'player_id':int(r['player_hit'])} for r in labels if int(r['player_hit'])>0]
        assert all(g['player_id'] in [1,2] for g in contacts)
        slug=match+'_'+Path(filename).stem
        case=OUT/'cases'/slug;specialist=case/'runtime/专项分析'/slug
        specialist.mkdir(parents=True,exist_ok=True)
        config={'pose_id':slug,'source_file':str(source.resolve()),'source_sha256':file_hash(source),
                'pose_file':str(case/'coarse'/slug/'result.json'),'specialist_folder':str(specialist),
                'start':0.,'end':n/fps,'output_folder':str(case/'unused_classification_output')}
        atomic_json(case/'config.json',config)
        rows.append({'id':slug,'match':match,'video_name':filename,'frames':n,'nominal_fps':fps,
            'dimensions':[w,h],'case_folder':str(case),'config_file':str(case/'config.json'),
            'source_path':str(source.resolve()),'source_sha256':file_hash(source),
            'hit_csv':str(path),'bbox_csv':str(bbox),'ball_csv':str(ball),
            'annotation_sha256':{str(p):file_hash(p) for p in [path,bbox,ball]},'human_contacts':contacts})
    assert len(rows)==7 and sum(len(r['human_contacts']) for r in rows)==55
    paths=[ROOT/f for f in ['analyze_pose.py','auto_players.py','specialist_pose.py','specialist_tracknet.py','specialist_events.py','hit_refinement.py','hit_gru.py','stroke_features.py','contact_timing.py','evaluate_external_amateur_hits.py','run_external_amateur_pose.py','prepare_external_amateur_eval.py']]
    paths.extend(ROOT/'models'/f for f in ['yolox-m-humanart.onnx','rtmpose-m-body7-256x192.onnx','tracknetv3/TrackNet_best.pt','tracknetv3/InpaintNet_best.pt','hit_gru/keras_tensors.npz','contact_timing/contact_ridge.npz','contact_timing/manifest.json'])
    atomic_json(OUT/'protocol.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),'rows':rows,
        'selection':'Every archive video with all three CSVs (human hit, ball, bbox). Seven match25 videos without hit CSV excluded for unavailable ground truth, not results. No selected rally removed for poor performance.',
        'ground_truth_source':'kwyoke/Badminton-hit-detection public am_singles.zip; no user personal labels or model-created labels.',
        'source_repository':'https://github.com/kwyoke/Badminton-hit-detection',
        'archive_sha256':file_hash(DATA/'公开训练数据/Badminton_hit_detection/am_singles.zip'),
        'code_and_weights':{str(p):file_hash(p) for p in paths},
        'configuration':{'coarse_sample_fps':5,'coarse_max_width':960,'detector_threshold':.4,'display_threshold':.5,'dense':'Original source frame stream, unchanged BGR production RTMPose call. Original native indices, GT indexed by decoded frame, no temporal resampling of annotations.'},
        'conditions':['automatic_pose_automatic_ball','automatic_pose_human_ball','human_bbox_estimated_pose_automatic_ball','human_bbox_estimated_pose_human_ball'],
        'oracle_limits':'Human-bbox conditions intentionally bypass detection/identity gates but still estimate joints using the SAME BGR RTMPose. Slot near means author player_id1 and far means id2 in oracle branch. Human ball uses CSV visible XY and leaves unavailable frames missing; no imputation. Neither condition is a fully automatic system.',
        'evaluation':{'primary_tolerance_seconds':.2,'secondary_tolerance_seconds':.1,'matching':'Maximum cardinality one-to-one by native decoded PTS within tolerance, then minimum total absolute error; no actor used for pairing. All human contacts retained including frame0/short edge contacts.',
            'actor':'Automatic matched event target bbox at the human contact frame mapped to author human bbox with max IoU >=.3; ambiguity/absence treated as not correct, not excluded. Oracle slots map to fixed author identities.',
            'reported':'Counts precision/recall/F1 by rally and source; raw visual proposals and unchanged production refinement separately; missing/extra contacts and errors; no stroke-type accuracy because this dataset has no stroke-type GT.'},
        'isolation':'Read-only source/labels/weights. Process-local source resolver and output-directory redirects only; no production job DB registration. No fitting, tuning, default activation, or manual candidate correction.',
        'training_overlap':'Not used in this project training. Author describes hit-GRU trained on professional matches; overlap with third-party TrackNet/pose pretraining not fully verified. Two camera sources are not seven independent matches.'})
    print(json.dumps({'rallies':len(rows),'human_hits':55,'frames':sum(r['frames'] for r in rows),'protocol':str(OUT/'protocol.json')},ensure_ascii=False))


if __name__=='__main__':main()
