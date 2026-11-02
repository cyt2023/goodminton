"""Native-frame event fusion. Unmatched and conflicting proposals are retained."""
from copy import deepcopy
import hashlib
import math
from research_adapters import settings,nearest_frame,validate_clock


def deduplicate(hits,tolerance):
    kept=[];suppressed=[]
    priority={'manual_annotation':3,'imported_prediction':2,'heuristic_baseline':1}
    for row in sorted(hits,key=lambda h:h['frame_index']):
        duplicate=next((h for h in reversed(kept) if row['frame_index']-h['frame_index']<=tolerance),None)
        if duplicate is None:kept.append(deepcopy(row));continue
        if duplicate.get('player_id') and row.get('player_id') and duplicate['player_id']!=row['player_id']:
            if duplicate['provenance']['kind']=='manual_annotation' and row['provenance']['kind']=='manual_annotation':
                kept.append(deepcopy(row));continue
            if duplicate['provenance']['kind']!='manual_annotation' and row['provenance']['kind']!='manual_annotation':
                duplicate['conflicting_player_candidates']=sorted(set([duplicate['player_id'],row['player_id']]))
                duplicate['player_id']=None
        score=lambda h:(priority.get(h['provenance']['kind'],0),h.get('confidence') if h.get('confidence') is not None else -1)
        if score(row)>score(duplicate):
            old=deepcopy(duplicate);conflict=duplicate.get('conflicting_player_candidates')
            duplicate.update(deepcopy(row))
            if conflict:duplicate.update(player_id=None,conflicting_player_candidates=conflict)
            suppressed.append({'reason':'duplicate_hit','prediction':old,'kept_id':duplicate['id']})
        else:suppressed.append({'reason':'duplicate_hit','prediction':row,'kept_id':duplicate['id']})
    return kept,suppressed


def fuse(hits,actions,context,options=None):
    cfg={**settings(),**(options or {})};times=context['times'];validate_clock(times)
    kept,suppressed=deduplicate(hits,cfg['duplicate_hit_frames']);available=actions.get('predictions',[])
    edges=[];ambiguous=set();eligible={}
    for i,h in enumerate(kept):
        t=times[h['frame_index']];candidates=[]
        for j,a in enumerate(available):
            manual=a.get('provenance',{}).get('kind')=='manual_annotation'
            if not manual and (a['confidence'] is None or a['confidence']<cfg['action_confidence_min']):continue
            gap=max(a['start']-t,0,t-a['end'])
            if gap>cfg['action_match_tolerance_seconds']:continue
            if h.get('player_id') and a.get('candidate_player_id') and h['player_id']!=a['candidate_player_id']:continue
            candidates.append((j,a,gap))
        if candidates:
            score=lambda a:a['confidence'] if a['confidence'] is not None else 1.
            ranked=sorted(candidates,key=lambda v:score(v[1]),reverse=True)
            if len(ranked)>1 and ranked[0][1]['stroke_type']!=ranked[1][1]['stroke_type'] and abs(score(ranked[0][1])-score(ranked[1][1]))<cfg['conflicting_class_margin']:
                ambiguous.add(i)
            for j,a,gap in candidates:
                center_distance=abs(t-(a['start']+a['end'])/2)
                edges.append((gap,center_distance,-score(a),i,j))
                eligible.setdefault(i,[]).append(j)
    chosen={};used=set()
    for _,_,_,i,j in sorted(edges):
        if i not in ambiguous and i not in chosen and j not in used:chosen[i]=j;used.add(j)
    events=[]
    for i,h in enumerate(kept):
        hit=h['frame_index'];a=available[chosen[i]] if i in chosen else None
        baseline=h.get('baseline_event',{})
        lo=baseline.get('start_frame',max(0,hit-round(.6*context['fps'])));hi=baseline.get('end_frame',min(len(times)-1,hit+round(.8*context['fps'])))
        label='unknown';confidence=None;interval_source='heuristic_context';label_source='unavailable'
        if a:
            lo,hi=min(a['start_frame'],hit),max(a['end_frame'],hit)
            label=a['stroke_type'];confidence=a['confidence'];label_source=a['provenance']['kind']
            interval_source='human_imported_interval' if label_source=='manual_annotation' else 'dsta_imported_interval'
        elif i not in ambiguous and baseline.get('classification_error') is None and baseline.get('stroke_type') not in [None,'unknown','no_stroke']:
            label=baseline['stroke_type'];confidence=baseline.get('bst_softmax_score');label_source='bst_model_inference'
        if h.get('human_label'):
            human=h['human_label'];label=human['stroke_type'];confidence=None;label_source='manual_annotation'
            lo=human.get('start_frame',lo);hi=human.get('end_frame',hi)
            if lo is None:lo=nearest_frame(times,human.get('start',times[max(0,hit-1)]))
            if hi is None:hi=nearest_frame(times,human.get('end',times[min(len(times)-1,hit+1)]))
            interval_source='human_review_window'
        lo=max(0,min(hit,int(lo)));hi=min(len(times)-1,max(hit,int(hi)))
        provenance=[h['provenance'],{'component':'stroke_classification','kind':label_source,'model':'DSTA import' if a else 'existing BST' if label_source=='bst_model_inference' else label_source}]
        if label_source=='bst_model_inference':
            provenance[1].update(model_revision=baseline.get('model_revision'),classifier=baseline.get('classifier_name','BST-0'),
                                 input_quality=baseline.get('input_quality'),classification_source=baseline.get('classification_source'))
        if a:provenance.append(a['provenance'])
        missing=[]
        if h.get('player_id') is None:missing.append('player_id')
        if label=='unknown':missing.append('stroke_type')
        if confidence is None:missing.append('stroke_confidence')
        if h.get('confidence') is None:missing.append('hit_confidence')
        event={'event_id':h['id'],'match_id':context.get('match_id'),'rally_id':None,'player_id':h.get('player_id'),
            'candidate_player_id':h.get('candidate_player_id'),'player_attribution_status':'human_confirmed' if h['provenance']['kind']=='manual_annotation' else 'automatic_candidate_unverified',
            'hit_frame':hit,'hit_timestamp':times[hit],'action_start_frame':lo,'action_end_frame':hi,
            'action_start_timestamp':times[lo],'action_end_timestamp':times[hi+1] if hi+1<len(times) else times[hi]+1/context['fps'],
            'action_interval_source':interval_source,'stroke_type':label,'raw_action_label':a.get('raw_label') if a else baseline.get('source_model_label'),
            'stroke_confidence':confidence,'hit_confidence':h.get('confidence'),'confidence_note':'Raw source-model score, not calibrated correctness probability',
            'shuttle_trajectory':[],'racket_keypoints':[],'player_pose':[],'court_coordinates':None,
            'quality_metrics':{},'quality_score':None,'coaching_feedback':None,'model_provenance':provenance,
            'missing_fields':missing+['rally_id','quality_score'],'fusion':{'dsta_action_id':a['id'] if a else None,
                'ambiguous_actions':i in ambiguous,'candidate_action_ids':[available[j]['id'] for j in eligible.get(i,[])],
                'action_interval_expanded_to_include_hit':bool(a and not a['start_frame']<=hit<=a['end_frame'])}}
        events.append(event)
    return {'events':events,'suppressed_hits':suppressed,'unmatched_actions':[a for j,a in enumerate(available) if j not in used],
            'rejected_imports':actions.get('rejected',[]),'configuration':cfg}
