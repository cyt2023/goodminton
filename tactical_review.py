"""TactiPlay-inspired evidence catalogue and multi-granularity review.
LLM selects existing claims/categories only; no free-form factual output accepted.
"""
import hashlib,json,math
from collections import Counter,defaultdict
from urllib.parse import urlencode
from stroke_taxonomy import LABELS

PAPER='https://arxiv.org/abs/2607.27125'
TAXONOMY={
 'swing_geometry':{'domain':'technical_execution','focus':'compare_stroke','requires':'measured joint/path evidence'},
 'stroke_depth':{'domain':'technical_execution','focus':'compare_stroke','requires':'verified landing coordinates'},
 'stroke_width':{'domain':'technical_execution','focus':'compare_stroke','requires':'verified landing coordinates'},
 'net_clearance':{'domain':'technical_execution','focus':'compare_stroke','requires':'calibrated shuttle height'},
 'pace':{'domain':'technical_execution','focus':'compare_stroke','requires':'validated shuttle speed'},
 'execution_error':{'domain':'technical_execution','focus':'compare_stroke','requires':'reviewed error annotation'},
 'recovery':{'domain':'positioning_and_recovery','focus':'review_movement','requires':'reviewed court mapping and usable player track'},
 'positioning':{'domain':'positioning_and_recovery','focus':'review_movement','requires':'reviewed court mapping and usable player track'},
 'readiness':{'domain':'positioning_and_recovery','focus':'review_movement','requires':'reviewed readiness annotation'},
 'shot_selection':{'domain':'tactical_choices','focus':'compare_sequence','requires':'same-rally three observed stroke labels'},
 'space_use':{'domain':'tactical_choices','focus':'review_movement','requires':'both player positions and verified landing coordinates'},
 'reactive_play':{'domain':'tactical_choices','focus':'compare_sequence','requires':'reviewed tactical context'}}
FOCUS={'compare_stroke':'对比同一球员、同类技术的其他片段，先核对动作标签和观测可靠性。',
       'review_movement':'结合前后击球回看站位与移动；图中空缺没有被补成连续运动。',
       'compare_sequence':'连看这三拍的行动、回应和下一拍，讨论是否有其他可行选择；当前序列本身不证明决策错误。',
       'review_rally':'连同回合边界、终结段及已核对的结果一起回看；没有记录胜负原因时不作因果归因。',
       'compare_recurrence':'比较这些回合中重复出现的同类序列，确认是否存在需要训练的共同情境。'}

def measurement_text(key,value,space):
    if key=='foot_path_efficiency_2d':return f'双脚二维路径比为{value:.3f}。'
    side='左' if key.startswith('left_') else '右' if key.startswith('right_') else ''
    if key.endswith('wrist_peak_px_s'):name,unit=side+'腕峰值像素速度','像素/秒'
    elif 'trajectory_error_px' in key:name,unit=side+'腕轨迹二阶差分','像素'
    elif 'elbow_angle_2d_deg' in key:name,unit=side+'肘投影角','度'
    elif '_relative_peak_estimated_m_s' in key:name,unit=side+('腕' if 'wrist' in key else '踝')+'相对运动峰值估计','估计米/秒'
    elif key.endswith('_estimated_3d_deg'):
        part=next((label for token,label in [('elbow','肘'),('wrist','腕'),('knee','膝'),('shoulder','肩')] if token in key),'关节');name,unit=side+part+'单目3D角度估计','度'
    elif 'peak_lag_s' in key:name,unit=side+'腕与踝部运动峰值时差','秒'
    elif 'torso_axis' in key:name,unit='躯干轴相邻采样变化角度估计','度'
    else:name,unit=key,space
    return f'{name}为{value:.3f}{unit}。'

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def validate_rallies(packet,report):
    if packet.get('source_sha256')!=report['source_sha256'] or packet.get('kind')!='reviewed_rally_annotations':raise ValueError('Rallies require same-source reviewed annotation provenance')
    end=report['timestamps'][-1]+1/report['nominal_fps'];previous=0.;ids=set()
    for row in sorted(packet.get('rallies',[]),key=lambda r:r['start']):
        if not isinstance(row.get('id'),str) or row['id'] in ids or row.get('review_status')!='reviewed':raise ValueError('Unique reviewed rally IDs required')
        if not all(isinstance(row.get(k),(int,float)) and math.isfinite(row[k]) for k in ['start','end']) or not previous<=row['start']<row['end']<=end:raise ValueError('Reviewed rally boundaries overlap or exceed video')
        ids.add(row['id']);previous=row['end']
    return packet

def links(report,events):
    return [{'event_id':e['event_id'],'timestamp':e['hit_timestamp'],'frame':e['hit_frame'],
             'url':'/analysis?'+urlencode({'job':report['job_id'],'event':e['event_id']}),
             'media_url':'/media/'+(report.get('playback_video_id') or report['source_video_id'])+'#t='+str(max(0,e['action_start_timestamp']-report.get('playback_provenance',{}).get('source_time_offset',0)))+','+str(e['action_end_timestamp']-report.get('playback_provenance',{}).get('source_time_offset',0))} for e in events]

def movement(report,events):
    calibration=report.get('court_calibration') or {};calibration=calibration.get('record',calibration)
    if calibration.get('review_status')!='confirmed' or not calibration.get('matrix') or calibration.get('camera_fixed') is False:return {'status':'unavailable','reason':'Reviewed fixed-camera court mapping unavailable','tracks':[]}
    tracks=[]
    for side in ['near','far']:
        points={}
        for e in events:
            if e.get('player_id')!=side:continue
            for p in e.get('court_coordinates') or []:
                i=p['frame_index'];xy=p['xy_normalized']
                if not e['action_start_frame']<=i<=e['action_end_frame']:continue
                if not calibration.get('valid_start_seconds',0)<=report['timestamps'][i]<=calibration.get('valid_end_seconds',report['timestamps'][-1]):continue
                if len(xy)==2 and all(math.isfinite(v) for v in xy) and -.2<=xy[0]<=1.2 and -.2<=xy[1]<=1.2:points[i]={'frame_index':i,'timestamp':report['timestamps'][i],'xy':xy}
        ordered=[points[i] for i in sorted(points)];segments=[]
        for p in ordered:
            if not segments or p['timestamp']-segments[-1][-1]['timestamp']>max(.2,2/report['nominal_fps']):segments.append([])
            segments[-1].append(p)
        segments=[s for s in segments if len(s)>=3]
        if segments:tracks.append({'player_id':side,'segments':segments,'units':'normalized_doubles_court','identity_status':'candidate_track; verify identity','observed_points':sum(map(len,segments))})
    return {'status':'available_observed_segments' if tracks else 'unavailable','tracks':tracks,'coordinate_space':'reviewed_ground_plane_0_1',
        'coverage_scope':'only observed action windows; missing gaps not connected','reason':None if tracks else 'Insufficient consecutive high-confidence projected ankles'}

def build(report,rallies=None):
    packet=validate_rallies(rallies,report) if rallies else {'rallies':[]};units=[];all_claims={};all_candidates={};eventmap={e['event_id']:e for e in report['events']}
    def unit(identifier,level,events,rally_id=None):
        u={'id':identifier,'level':level,'event_ids':[e['event_id'] for e in events],'rally_id':rally_id,'claims':[],'review_candidates':[],
           'video_references':links(report,events),'movement':movement(report,events),'causal_explanation_available':False}
        units.append(u);return u
    def claim(u,code,text,value=None,units_name=None,event_ids=None,uncertainty=None):
        identifier=digest([u['id'],code,event_ids,value])[:24]
        c={'id':identifier,'unit_id':u['id'],'code':code,'text':text,'value':value,'units':units_name,'event_ids':event_ids or u['event_ids'],
           'uncertainty':uncertainty,'provenance':'rendered_from_structured_evidence'};u['claims'].append(c);all_claims[identifier]=c;return identifier
    def candidate(u,taxonomy_id,claim_ids,focus):
        if not claim_ids:return
        identifier=digest([u['id'],taxonomy_id,claim_ids])[:24];c={'id':identifier,'unit_id':u['id'],'taxonomy_id':taxonomy_id,
            'claim_ids':claim_ids,'focus_code':focus,'status':'review_candidate_not_diagnosed_issue','domain':TAXONOMY[taxonomy_id]['domain']}
        u['review_candidates'].append(c);all_candidates[identifier]=c
    for e in report['events']:
        u=unit('stroke:'+e['event_id'],'stroke',[e]);label=e.get('stroke_type','unknown')
        c=claim(u,'stroke_record',f"{e['hit_timestamp']:.3f}秒的击球记录为{LABELS.get(label,label)}，击球方为{e.get('player_id') or '未确认'}。",{'stroke_type':label,'player_id':e.get('player_id')},event_ids=[e['event_id']],uncertainty=e.get('player_attribution_status'))
        q=e.get('quality_metrics',{});measured=[]
        for space,values in [('2d',q.get('observations_2d',{})),('estimated3d',q.get('observations_3d',{}))]:
            for key,value in values.items():
                if value is None:continue
                if not isinstance(value,(int,float)) or not math.isfinite(value):continue
                measured.append(claim(u,'measurement:'+space+':'+key,measurement_text(key,value,space),value,key,event_ids=[e['event_id']],uncertainty=q.get('estimated_3d',{}).get('uncertainty') if space=='estimated3d' else 'image projection; scale/view dependence'))
        if measured:candidate(u,'swing_geometry',measured[:4],'compare_stroke')
        if u['movement']['tracks']:
            c=claim(u,'observed_movement','此动作窗口有经过场地映射的连续位置观测；缺失片段未连接。',sum(t['observed_points'] for t in u['movement']['tracks']),'observed_points')
            candidate(u,'positioning',[c],'review_movement')
    patterns=defaultdict(list)
    for rally in packet['rallies']:
        events=sorted([e for e in report['events'] if rally['start']<=e['hit_timestamp']<rally['end']],key=lambda e:e['hit_timestamp'])
        for i in range(max(0,len(events)-2)):
            three=events[i:i+3]
            if not all(three[k]['hit_timestamp']<three[k+1]['hit_timestamp'] for k in [0,1]):continue
            u=unit(f"sequence:{rally['id']}:{i}",'three_stroke',three,rally['id']);labels=[e['stroke_type'] for e in three]
            c=claim(u,'stroke_sequence',' → '.join(LABELS.get(v,v) for v in labels),labels,'ordered_stroke_labels',uncertainty='Stroke labels/actors may remain automatic candidates')
            if 'unknown' not in labels and all(e.get('player_id') for e in three):
                candidate(u,'shot_selection',[c],'compare_sequence');patterns[tuple(labels)].append(u)
        u=unit('rally:'+rally['id'],'rally',events,rally['id']);claim(u,'recorded_rally',f"已核对回合范围{rally['start']:.3f}–{rally['end']:.3f}秒，内有{len(events)}个击球记录。",{'start':rally['start'],'end':rally['end'],'events':len(events)})
        u['recorded_outcome']=rally.get('outcome');u['outcome_reason_status']='unavailable_without_explicit_evidence'
        if rally.get('outcome'):claim(u,'recorded_outcome','回合结果来自已核对标注；胜负原因未自动补充。',rally['outcome'],uncertainty='User-reviewed outcome, no inferred causal explanation')
        if len(events)>=3 and all(e['stroke_type']!='unknown' for e in events[-3:]):
            c=claim(u,'terminal_sequence','记录的终结三拍为：'+' → '.join(LABELS.get(e['stroke_type'],e['stroke_type']) for e in events[-3:]),[e['stroke_type'] for e in events[-3:]],event_ids=[e['event_id'] for e in events[-3:]])
            candidate(u,'shot_selection',[c],'review_rally')
    match_unit=unit('match:'+report['job_id'],'match',report['events']);claim(match_unit,'match_coverage',f"当前记录包含{len(report['events'])}个击球和{len(packet['rallies'])}个已核对回合。",{'events':len(report['events']),'reviewed_rallies':len(packet['rallies'])})
    for pattern,occurrences in patterns.items():
        rally_ids=sorted({u['rally_id'] for u in occurrences})
        if len(rally_ids)<2:continue
        ids=list(dict.fromkeys(e for u in occurrences for e in u['event_ids']));c=claim(match_unit,'recurring_sequence',' → '.join(LABELS.get(v,v) for v in pattern)+f'出现在{len(rally_ids)}个已核对回合；这是一项重复模式，尚未被诊断为问题。',{'pattern':pattern,'rallies':rally_ids},event_ids=ids)
        candidate(match_unit,'shot_selection',[c],'compare_recurrence')
    value={'schema_version':1,'job_id':report['job_id'],'source_sha256':report['source_sha256'],'method':'TactiPlay-inspired; not a complete system reproduction',
        'source':PAPER,'taxonomy':{'type':'local_adaptation_not_exact_paper_transcription','domains':['technical_execution','positioning_and_recovery','tactical_choices'],'descriptors':TAXONOMY},
        'units':units,'claim_catalog':all_claims,'candidate_catalog':all_candidates,'rally_status':'reviewed_boundaries_available' if rallies else 'unavailable_without_reviewed_boundaries',
        'limitations':['No new events or measurements generated by LLM','Unknown rally boundaries/outcomes are not invented','Recurring patterns are review themes, not proven recurring errors'],
        'llm_status':'not_requested'}
    value['evidence_hash']=digest(value);return value

def validate_selection(review,selection):
    items=[];seen=set()
    for entry in selection.get('items',[])[:12]:
        key=entry.get('candidate_id');candidate=review['candidate_catalog'].get(key)
        if not candidate or key in seen:continue
        # Only catalogue fields are trusted; discard all model-written prose,
        # numbers, event IDs, classifications and explanations.
        seen.add(key);claims=[review['claim_catalog'][i] for i in candidate['claim_ids']]
        ids=list(dict.fromkeys(e for c in claims for e in c['event_ids']));references={v['event_id']:v for u in review['units'] for v in u['video_references']}
        items.append({**candidate,'observations':[c['text'] for c in claims],'coaching_focus':FOCUS[candidate['focus_code']],
            'evidence':claims,'video_references':[references[e] for e in ids if e in references]})
    if not items:raise ValueError('LLM selected no existing evidence-grounded review candidates')
    return {'status':'completed','llm_generated':True,'scope':'LLM prioritizes existing catalogue; factual text and measurements rendered by code','evidence_hash':review['evidence_hash'],'items':items}
