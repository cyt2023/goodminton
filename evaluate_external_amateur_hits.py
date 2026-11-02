"""Frozen raw-video hit detection evaluation on external manual annotations."""
import argparse
import copy
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

from analysis_jobs import DATA,atomic_json,file_hash
import hit_refinement
from specialist_events import proposals
from specialist_model import ROOT
from source_frame_media import ffmpeg_executable

OUT=DATA/'外部标注测试/2026-10-21/Amateur_hit7'


def read(p):return json.loads(Path(p).read_text())


def check(protocol):
    amendment_path=OUT/'scoring_amendment_v2.json'
    amendment=read(amendment_path) if amendment_path.exists() else None
    for p,h in protocol['code_and_weights'].items():
        if amendment and p==str(Path(__file__).resolve()):
            assert h==amendment['original_evaluator_sha256']
            h=amendment['updated_evaluator_sha256']
        assert file_hash(Path(p))==h, 'Frozen dependency changed: '+p
    if amendment:
        assert file_hash(OUT/'native_frame_alignment_manifest.json')==amendment['native_alignment_manifest_sha256']
        for record in read(OUT/'native_frame_alignment_manifest.json')['files']:
            assert file_hash(Path(record['path']))==record['sha256']
    for row in protocol['rows']:
        assert file_hash(Path(row['source_path']))==row['source_sha256']
        assert all(file_hash(Path(p))==h for p,h in row['annotation_sha256'].items())


def stage(protocol,kind):
    check(protocol)
    for row in protocol['rows']:
        folder=Path(row['case_folder']); casefile=folder/'case.json'
        if not casefile.exists():atomic_json(casefile,row)
        if kind=='pose':
            output=Path(read(row['config_file'])['specialist_folder'])/'human_bbox_estimated_pose.json'
            command=[str(ROOT/'.venv-pose/bin/python'),str(ROOT/'run_external_amateur_pose.py'),'--case',str(casefile)]
        else:
            output=Path(read(row['config_file'])['specialist_folder'])/'trajectory.json'
            command=[str(ROOT/'.venv-vision/bin/python'),str(ROOT/'specialist_tracknet.py'),'--config',row['config_file']]
        if output.exists():print('REUSE',kind,row['id'],flush=True);continue
        print('START',kind,row['id'],flush=True)
        log=folder/(kind+'.log')
        with log.open('ab') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            atomic_json(folder/(kind+'_failure.json'),{'exit_code':result.returncode,'log':str(log),'failed_at':datetime.now(timezone.utc).isoformat()})
            print('FAILED',kind,row['id'],result.returncode,flush=True)
        else:print('DONE',kind,row['id'],flush=True)
    check(protocol)


def boxes(row):
    result={};w,h=row['dimensions']
    for r in csv.DictReader(Path(row['bbox_csv']).open()):
        if int(float(r['Visbility']))<=0:continue
        frame,pid=int(r['Frame']),int(r['id'])
        if pid not in [1,2]:continue
        result.setdefault(frame,{})[pid]=[float(r[k])*scale for k,scale in zip(['x1','y1','x2','y2'],[w,h,w,h])]
    return result


def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0.


def evaluate(events,dense,row,tolerance,oracle):
    truth=row['human_contacts']
    native=read(Path(row['case_folder'])/'verified_native_frame_pts.json')
    cadence=native['frame_cadence_for_frame_based_metric']
    assert len(dense['frames'])==native['source_frames']==row['frames']
    # GT is indexed by original frames. Some source files have duplicated AND
    # decreasing PTS; retain original prediction inputs and score their frame
    # indices at the fixed nominal25/30 cadence instead of inventing timestamps.
    times=np.arange(row['frames'],dtype=float)/cadence
    m,n=len(events),len(truth);matches=[]
    if m:
        frame_errors=np.abs(np.asarray([e['frame_index'] for e in events])[:,None]-np.asarray([g['frame'] for g in truth])[None,:])
        errors=frame_errors/cadence
        penalty=m+n+1.;cost=np.zeros((m+n,m+n))
        cost[:m,:n]=np.where(frame_errors<=tolerance*cadence+1e-9,errors,penalty*4);cost[:m,n:]=penalty;cost[m:,:n]=penalty
        ii,jj=linear_sum_assignment(cost)
        matches=[(int(i),int(j)) for i,j in zip(ii,jj) if i<m and j<n and cost[i,j]<penalty]
    human_boxes=boxes(row);details=[]
    for i,j in matches:
        event,gt=events[i],truth[j]
        if oracle:
            mapped=1 if event['player']=='near' else 2; overlaps=None
        else:
            person=dense['frames'][gt['frame']]['players'].get(event['player']);mapped=None;overlaps={}
            if person:
                overlaps={pid:iou(person['bbox'],b) for pid,b in human_boxes.get(gt['frame'],{}).items()}
                if overlaps:
                    best=max(overlaps,key=overlaps.get)
                    if overlaps[best]>=.3:mapped=best
        details.append({'candidate_id':event['id'],'candidate_frame':event['frame_index'],'gt_frame':gt['frame'],
            'signed_error_seconds':float(times[event['frame_index']]-times[gt['frame']]),
            'gt_player_id':gt['player_id'],'candidate_slot':event['player'],'mapped_author_player_id':mapped,
            'bbox_iou':overlaps,'actor_correct':mapped==gt['player_id']})
    used_i={i for i,j in matches};used_j={j for i,j in matches};matched=len(matches)
    return {'tolerance_seconds':tolerance,'tolerance_original_frames':tolerance*cadence,
        'time_metric_basis':'Original frame index divided by fixed native25/30 nominal cadence; not physical source-clock error.',
        'human_contacts':n,'candidates':m,'matched':matched,
        'precision':matched/m if m else 0.,'recall':matched/n,'f1':2*matched/(m+n) if m+n else 0.,
        'missed':n-matched,'extra':m-matched,'actor_and_time_correct':sum(r['actor_correct'] for r in details),
        'median_error_seconds':float(np.median([abs(r['signed_error_seconds']) for r in details])) if details else None,
        'matches':details,'missed_contacts':[g for j,g in enumerate(truth) if j not in used_j],
        'extra_candidates':[e for i,e in enumerate(events) if i not in used_i]}


def refined(dense,events,refinement):
    output=[]
    for original in events:
        event=dict(original);suggestion=refinement['suggestions'].get(original['id'])
        if suggestion and suggestion['applied']:
            event.update(original_peak_time=original['peak_time'],original_frame_index=original['frame_index'],
                peak_time=suggestion['suggested_time'],frame_index=suggestion['suggested_frame'],center_suggestion=suggestion,contact_confirmed=False)
        output.append(event)
    for extra in refinement['supplemental_candidates']:
        i=extra['frame_index'];times=np.asarray([f['source_time'] for f in dense['frames']])
        lo=int(np.searchsorted(times,max(0.,extra['peak_time']-1.)))
        hi=min(len(times)-1,int(np.searchsorted(times,extra['peak_time']+1.)))
        output.append({**extra,'start_frame':lo,'end_frame':hi,'start':float(times[lo]),'end':min(dense['end'],float(times[hi]+np.median(np.diff(times))))})
    return output


def human_ball(row,original):
    result=copy.deepcopy(original);gt={int(r['Frame']):r for r in csv.DictReader(Path(row['ball_csv']).open())}
    for frame in result['frames']:
        g=gt.get(frame['frame_index']);xy=[float(g['X']),float(g['Y'])] if g is not None else None
        visible=bool(g is not None and int(g['Visibility'])>0 and np.isfinite(xy).all())
        frame.update(raw_xy=xy if visible else None,raw_visible=visible,rectified_xy=xy if visible else [0,0],
            raw_heatmap_peak=1. if visible else 0.,inpaint_requested=False,coordinate_source='author_human_ball_annotation' if visible else 'author_missing')
    result['intervention']='Author manual ball coordinates; confidence1 is an adapter visibility sentinel, not measured model confidence. Missing stays missing.'
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pose','tracknet','evaluate'],required=True)
    args=parser.parse_args();protocol=read(OUT/'protocol.json')
    if args.stage!='evaluate':return stage(protocol,args.stage)
    if (OUT/'result.json').exists():raise FileExistsError('Preserve completed external evaluation')
    check(protocol);rows=[];failures=[]
    for row in protocol['rows']:
        sourcefolder=Path(read(row['config_file'])['specialist_folder'])
        paths={k:sourcefolder/v for k,v in [('auto_pose','dense_pose.json'),('manual_bbox_pose','human_bbox_estimated_pose.json'),('auto_ball','trajectory.json')]}
        if not all(p.exists() for p in paths.values()):
            failures.append({'id':row['id'],'missing':[str(p) for p in paths.values() if not p.exists()],'human_contacts':len(row['human_contacts'])});continue
        automatic=read(paths['auto_pose']);bbox=read(paths['manual_bbox_pose']);ball=read(paths['auto_ball']);manual_ball=human_ball(row,ball)
        assert len(automatic['frames'])==len(bbox['frames'])==len(ball['frames'])==row['frames']
        for condition in protocol['conditions']:
            oracle=condition.startswith('human_bbox');use_human_ball=condition.endswith('human_ball')
            dense=bbox if oracle else automatic;trajectory=manual_ball if use_human_ball else ball
            conditionroot=OUT/'conditions'/row['id']/condition
            folder=conditionroot/'专项分析'/row['id'];folder.mkdir(parents=True,exist_ok=True)
            atomic_json(folder/'dense_pose.json',dense);atomic_json(folder/'trajectory.json',trajectory)
            config={**read(row['config_file']),'ffmpeg':ffmpeg_executable()}
            events=proposals(dense,trajectory)
            previous=hit_refinement.DATA
            try:
                hit_refinement.DATA=conditionroot;refinement=hit_refinement.prepare_inputs(config,dense,trajectory,events)
            finally:hit_refinement.DATA=previous
            completed=refined(dense,events,refinement)
            atomic_json(folder/'raw_candidates.json',{'events':events})
            atomic_json(folder/'refined_candidates.json',{'events':completed})
            result={'id':row['id'],'match':row['match'],'condition':condition,
                'source_sha256':row['source_sha256'],'input_sha256':{str(p):file_hash(p) for p in [folder/'dense_pose.json',folder/'trajectory.json']},
                'audio':refinement['audio'],'assigned_pose_frames':dense['assigned_frames'],
                'raw_visual':{str(t):evaluate(events,dense,row,t,oracle) for t in [.2,.1]},
                'production_refined':{str(t):evaluate(completed,dense,row,t,oracle) for t in [.2,.1]}}
            atomic_json(folder/'evaluation.json',result);rows.append(result)
            print('EVALUATED',row['id'],condition,'raw',len(events),'refined',len(completed),flush=True)
    summaries={}
    for condition in protocol['conditions']:
        summaries[condition]={}
        for stage_name in ['raw_visual','production_refined']:
            summaries[condition][stage_name]={}
            for tolerance in ['0.2','0.1']:
                subset=[r[stage_name][tolerance] for r in rows if r['condition']==condition]
                total=sum(r['human_contacts'] for r in subset);candidates=sum(r['candidates'] for r in subset);matched=sum(r['matched'] for r in subset)
                summaries[condition][stage_name][tolerance]={'human_contacts_evaluated':total,'human_contacts_selected':55,'rallies_evaluated':len(subset),'rallies_selected':7,
                    'candidates':candidates,'matched':matched,'missed':total-matched,'extra':candidates-matched,
                    'precision':matched/candidates if candidates else 0.,'recall':matched/total if total else None,'f1':2*matched/(total+candidates) if total+candidates else 0.,
                    'actor_and_time_correct':sum(r['actor_and_time_correct'] for r in subset)}
    check(protocol)
    atomic_json(OUT/'result.json',{'completed_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':file_hash(OUT/'protocol.json'),
        'scoring_amendment_sha256':file_hash(OUT/'scoring_amendment_v2.json'),
        'summary':summaries,'rows':rows,'processing_failures':failures,'source_camera_count':2,
        'scope':'External amateur hits only, no stroke-type GT. Automatic condition is the end-to-end hit proposal path before BST/VLM/display gates; oracle input conditions are diagnostic controls. All55contacts including edge frames remain. Nominal frame-based matching per disclosed amendment; raw production timestamps and predictions unchanged. No training/tuning/production change.'})
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
