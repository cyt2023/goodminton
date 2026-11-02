"""Unify actual model outputs and imported predictions without retraining."""
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path

from analysis_jobs import atomic_json,file_hash,DATA,registered_video
from research_adapters import registry,settings,import_predictions,ShuttleHitFormerAdapter,DSTAAdapter,nearest_frame
from research_quality import measure_2d
from research_pose3d import validate_packet,measure_3d
from stroke_event_fusion import fuse
from stroke_taxonomy import LABELS


def load(p,default=None):return json.loads(Path(p).read_text()) if Path(p).is_file() else default


def baseline_hits(events,labels,times):
    hits=[];excluded=[]
    for e in events:
        human=labels.get(e['id'])
        if human and (human.get('wrong_target') or human['stroke_type']=='no_stroke' or human.get('play_context')=='between_rallies'):
            excluded.append({'id':e['id'],'human_label':human});continue
        reviewed=human and human.get('contact_confirmed')
        frame=(human.get('center_frame') if human.get('center_frame') is not None else nearest_frame(times,human['peak_time'])) if reviewed else e['frame_index']
        hits.append({'id':e['id'],'frame_index':frame,'time':times[frame],
            'player_id':human['player'] if reviewed else e.get('player'),'candidate_player_id':e.get('player'),'confidence':None,
            'human_label':human if reviewed else None,'baseline_event':e,
            'provenance':{'component':'hit_detector','kind':'manual_annotation' if reviewed else 'heuristic_baseline',
                'name':'Human-confirmed contact' if reviewed else 'Existing shuttle/wrist/turn proposals + audio/GRU refinement',
                'note':'Not ShuttleHitFormer inference; no probability is inferred from the heuristic proposal score'}})
    return hits,excluded


def project(matrix,p):
    den=matrix[2][0]*p[0]+matrix[2][1]*p[1]+matrix[2][2]
    if abs(den)<1e-10:return None
    return [(matrix[j][0]*p[0]+matrix[j][1]*p[1]+matrix[j][2])/den for j in [0,1]]


def annotate_measurements(event,dense,trajectory,racketvision,calibration,pose3d=None):
    lo,hi=event['action_start_frame'],event['action_end_frame'];pid=event['player_id'];rframes=racketvision.get('frames',[]) if racketvision else []
    for i in range(lo,hi+1):
        t=dense['frames'][i]['source_time'];ball=trajectory['frames'][i]
        event['shuttle_trajectory'].append({'frame_index':i,'timestamp':t,'xy_px':ball['raw_xy'],'observed_model_detection':ball['raw_visible'],
                                           'score':ball.get('raw_heatmap_peak'),'model':trajectory['engine_version']})
        if i<len(rframes):event['racket_keypoints'].append({'frame_index':i,'timestamp':t,'detections':rframes[i]['rackets'],
            'association':'unassigned_rackets_in_frame; no unsupported hitter/racket identity assertion'})
        people=dense['frames'][i]['players']
        event['player_pose'].append({'frame_index':i,'timestamp':t,'player_id':pid,'keypoints':people.get(pid,{}).get('keypoints') if people.get(pid) else None})
    if not any(r['detections'] for r in event['racket_keypoints']):event['missing_fields'].append('racket_keypoints')
    if not any(p['keypoints'] for p in event['player_pose']):event['missing_fields'].append('player_pose')
    event['model_provenance'].append({'component':'perception','kind':'model_inference',
        'name':trajectory['engine_version'],'provenance':racketvision.get('model_provenance') if racketvision else trajectory.get('model_provenance')})
    if calibration and calibration.get('review_status')=='confirmed' and calibration.get('matrix'):
        coords=[]
        for p in event['player_pose']:
            joints=p['keypoints']
            if joints and all(joints[k]['score']>=.5 for k in [15,16]):
                feet=[project(calibration['matrix'],[joints[k]['x'],joints[k]['y']]) for k in [15,16]]
                if all(f is not None for f in feet):coords.append({'frame_index':p['frame_index'],'xy_normalized':[sum(f[d] for f in feet)/2 for d in [0,1]]})
        event['court_coordinates']=coords or None
    if event['court_coordinates'] is None:event['missing_fields'].append('court_coordinates')
    event['quality_metrics']=measure_2d(event,dense,settings()['pose_point_score_min'])
    estimated=measure_3d(event,pose3d)
    event['quality_metrics']['estimated_3d']=estimated
    event['quality_metrics']['observations_3d']=estimated['observations']
    if estimated.get('status')=='estimated_features_available':
        event['quality_metrics']['measurement_spaces']=['native_image_2d','monocular_estimated_3d_hip_relative']
        for index,part in [(2,'wrist'),(3,'elbow')]:
            if any(estimated['observations'].get(side+'_'+part+'_estimated_3d_deg') is not None for side in ['left','right']):
                field=event['quality_metrics']['indices'][index]
                field.update(status='estimated_measurement_reference_missing',estimated_measurement_available=True,
                    reason='已有单目3D关节角估计；仍缺挥拍手确认、误差验证及对应技术的有效参考，暂不评分。')
    obs=event['quality_metrics']['observations_2d'];evidence=[]
    for key,value in obs.items():
        if value is not None:evidence.append({'id':key,'value':value,'units':'deg_2d' if 'angle' in key else 'px/s' if 'px_s' in key else 'ratio_2d' if 'efficiency' in key else 'px'})
    event['coaching_feedback']={'engine':'evidence_template','llm_generated':False,'evidence':evidence,
        'text':f'回看 {event["hit_timestamp"]:.2f} 秒附近的动作，先核对击球球员、类型和触球帧。'+('本窗口有可用的二维关节观测，可结合叠加画面查看手腕路径和肘部投影角。' if evidence else '当前人体或动作窗口证据不足，暂不作技术评价。'),
        'quality_assessment_available':False,'next_step':'确认动作后，与同一机位、同一球员、同类技术比较；当前没有可靠的总体质量评分。'}


def csv_report(report):
    out=io.StringIO();keys=['event_id','match_id','rally_id','player_id','hit_frame','hit_timestamp','action_start_frame','action_end_frame','stroke_type','stroke_confidence','hit_confidence','quality_score','source_kind']
    writer=csv.DictWriter(out,fieldnames=keys);writer.writeheader()
    for e in report['events']:writer.writerow({**{k:e.get(k) for k in keys if k!='source_kind'},'source_kind':e['model_provenance'][0]['kind']})
    return out.getvalue()


def build(job_id, *, folder=None, dense=None, baseline=None, labels=None, source_video_id=None):
    folder=Path(folder).resolve() if folder else DATA/'专项分析'/job_id
    dense=dense or load(folder/'dense_pose.json')
    trajectory=load(folder/'trajectory.json')
    rv=load(folder/'racketvision/racketvision.json')
    baseline=baseline if baseline is not None else load(DATA/'技术动作'/job_id/'片段/classified_events.json',{'events':[]})['events']
    annotations=load(DATA/'击球标注'/job_id/'annotations.json',{'labels':{},'context_annotations':{}})
    labels=labels if labels is not None else annotations['labels']
    labels=deepcopy(labels)
    for eid,label in labels.items():label.update(annotations.get('context_annotations',{}).get(eid,{}))
    times=[f['source_time'] for f in dense['frames']];sha=dense['source_sha256']
    if trajectory['source_sha256']!=sha or len(trajectory['frames'])!=len(times):raise ValueError('Trajectory and pose source/coverage differ')
    if any(f['frame_index']!=i for i,f in enumerate(dense['frames'])) or any(f['frame_index']!=i for i,f in enumerate(trajectory['frames'])):raise ValueError('Inputs must use consecutive original frame indices')
    if rv and (rv['source_sha256']!=sha or rv['frame_count']!=len(times)):raise ValueError('RacketVision source/coverage differ')
    hits,excluded=baseline_hits(baseline,labels,times)
    imports={};module_states=registry()
    for name in ['shuttlehitformer','dsta']:
        packet=load(folder/'research_inputs'/(name+'.json'))
        if packet:
            imports[name]=import_predictions(packet,name,sha,times,dense['sample_fps'])
            module_states[name]={**module_states[name],'import_status':'imported_predictions_present','import_kind':packet['kind'],'imported_model_provenance':packet['model_provenance']}
    if 'shuttlehitformer' in imports:
        # Manual exclusions remain exclusions even when an independent model predicts there.
        imported=imports['shuttlehitformer']['predictions']
        excluded_frames=[nearest_frame(times,e['human_label']['peak_time']) for e in excluded]
        hits=[h for h in imported if all(abs(h['frame_index']-i)>2 for i in excluded_frames)]+[h for h in hits if h['provenance']['kind']=='manual_annotation']
    fused=fuse(hits,imports.get('dsta',{'predictions':[]}),{'times':times,'fps':dense['sample_fps'],'match_id':source_video_id or dense.get('source_video_id')})
    calibration=load(folder/'court_calibration.json')
    if calibration and calibration.get('source_sha256')!=sha:raise ValueError('Court calibration belongs to another source')
    pose3d_path=folder/'pose3d/pose3d.json'
    if not pose3d_path.exists():pose3d_path=DATA/'专项分析'/job_id/'pose3d/pose3d.json'
    pose3d=load(pose3d_path)
    if pose3d:
        try:
            validate_packet(pose3d,sha,times)
            module_states['monocular_pose3d']={**module_states.get('monocular_pose3d',{}),'status':'sampled_estimates_available','model_provenance':pose3d['model_provenance'],'sample_fps':pose3d['sample_fps']}
        except (ValueError,KeyError,TypeError) as error:
            module_states['monocular_pose3d']={**module_states.get('monocular_pose3d',{}),'status':'invalid_optional_input','reason':str(error)};pose3d=None
    for event in fused['events']:annotate_measurements(event,dense,trajectory,rv,calibration,pose3d)
    report={'schema_version':1,'job_id':job_id,'source_video_id':source_video_id or dense.get('source_video_id'),
        'execution_provenance':{name:file_hash(Path(__file__).parent/name) for name in ['research_report.py','stroke_event_fusion.py','research_quality.py','research_pose3d.py','research_adapters.py','research_models/label_map.json','research_models/settings.json']},
        'source_sha256':sha,'source_width':dense['source_width'],'source_height':dense['source_height'],
        'nominal_fps':dense['sample_fps'],'frame_count':len(times),'timestamps':times,'coordinate_system':'native_decoded_pixels; court 0..1 only after reviewed homography',
        'stroke_labels':LABELS,
        'model_status':module_states,'perception_provenance':rv['model_provenance'] if rv else trajectory.get('model_provenance',{'name':trajectory['engine_version'],'kind':'model_inference'}),
        'pose_model_provenance':{k:dense.get(k) for k in ['engine_version','models','weights','author_reference','identity_note']},
        'baseline_classification_provenance':load(DATA/'技术动作'/job_id/'片段/classified_events.json',{}).get('methods'),
        **fused,'excluded_candidates':excluded,'imports':imports,
        'summary':{'detected_stroke_events':len(fused['events']),'stroke_type_counts':dict(Counter(e['stroke_type'] for e in fused['events'])),
            'player_counts':dict(Counter(e['player_id'] or 'unknown' for e in fused['events'])),'source_kind_counts':dict(Counter(e['model_provenance'][0]['kind'] for e in fused['events'])),
            'unknown_types':sum(e['stroke_type']=='unknown' for e in fused['events']),'quality_scores_available':0,
            'shuttle_detected_frames':sum(b['raw_visible'] for b in trajectory['frames']),
            'racket_detected_frames':sum(bool(f['rackets']) for f in rv['frames']) if rv else None,
            'interpretation':'Candidate counts and imported/manual sources, not accuracy or complete-match ground truth'},
        'court_calibration':calibration,'missing_components':[k for k in ['shuttlehitformer','dsta'] if module_states[k]['status']=='unavailable'],
        'llm_coaching':{'status':'on_demand_local_model','model':'Qwen3-VL-4B-Instruct-4bit','default_feedback':'explicit evidence template, not LLM inference'}}
    from research_playback import ensure as ensure_playback
    source_path=Path(dense['source_file']) if dense.get('source_file') else registered_video(report['source_video_id'],allow_reference=True)['path']
    playback=ensure_playback(folder,source_path,sha,times,[dense['source_width'],dense['source_height']])
    report['playback_video_id']=playback['video_id'];report['playback_provenance']=playback
    atomic_json(folder/'research_report.json',report)
    (folder/'research_report.csv').write_text(csv_report(report))
    return report
