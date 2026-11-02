"""Deterministic, diverse review acquisition; never labels examples itself."""
from collections import Counter
import hashlib
import json
import random
from analysis_jobs import file_hash
from stroke_taxonomy import LABELS

VERSION='diverse-disagreement-contact-review-v2'

def build(events,doc,learned=None,refinement=None,size=20,exclude=(),evaluation=None):
    predictions={p['event_id']:p for p in (learned or {}).get('events',[])}
    heldout={p['event_id']:p for p in (evaluation or {}).get('events',[])}
    labels=doc['labels'];counts=Counter(a['stroke_type'] for a in labels.values()
        if not a.get('wrong_target') and a.get('play_context') != 'between_rallies' and a['stroke_type'] not in ['unknown','other','no_stroke'])
    records=[]
    for e in events:
        a=labels.get(e['id'])
        if doc.get('context_annotations', {}).get(e['id'], {}).get('play_context') == 'between_rallies':continue
        if a and (a.get('play_context') == 'between_rallies' or a['stroke_type'] in ['no_stroke','unknown'] or a.get('contact_confirmed')):continue
        p=predictions.get(e['id']);s=(refinement or {}).get('suggestions',{}).get(e['id'])
        reasons=[];priority=0.
        code=a['stroke_type'] if a else p['stroke_type'] if p else e['stroke_type']
        if not a:priority+=2;reasons.append('尚未标注')
        if not a or not a.get('contact_confirmed'):priority+=2;reasons.append('击球中心未核对')
        if s and s['applied']:
            priority+=2+min(1,abs(s['suggested_time']-s['original_time'])/.2)
            reasons.append(f"声画建议移动 {s['suggested_frame']-s['original_frame']:+d} 帧")
        if s and s.get('needs_precision_review'):
            priority+=3
            reasons.append('击球位置的证据有分歧，先逐帧核对')
        if e.get('from_refinement'):priority+=3;reasons.append('可能漏拍，先确认是否击球')
        checked=heldout.get(e['id'])
        if a and checked and checked.get('human_code')==a['stroke_type'] and not checked['match']:
            priority+=1.5
            reasons.append('同录像留出时曾误认成'+LABELS.get(checked['prediction_code'],checked['prediction_code']))
            if not checked.get('class_seen',True):
                priority+=1
                reasons.append('这个球种在该留出组外没有训练样本')
        if code not in ['unknown','other','no_stroke'] and counts[code]<10:
            priority+=2/(1+counts[code]/3);reasons.append(f'{LABELS.get(code,code)}标注较少（{counts[code]} 条）')
        if p and p['stroke_type']!=e['stroke_type']:
            priority+=1.5;reasons.append('个人模型与 BST 的球种意见不同')
        if a and e['stroke_type']!=a['stroke_type']:
            priority+=.75;reasons.append('BST 曾与您的球种选择不同')
        if p and len(p.get('ranking',[]))>=2:
            first,second=[x['uncalibrated_score'] for x in p['ranking'][:2]]
            margin=abs(first-second)/(abs(first)+abs(second)+1e-6)
            priority+=1-min(1,margin)
        ranked=(e.get('specialist_predictions') or {}).get('bst0',[])
        if len(ranked)>=2:
            priority+=1-min(1,abs(ranked[0]['softmax_score']-ranked[1]['softmax_score']))
        records.append({'event_id':e['id'],'player':a.get('player',e['player']) if a else e['player'],
                        'time':a.get('peak_time',e['peak_time']) if a else e['peak_time'],
                        'class_for_diversity':code,'priority_score':round(priority,5),'reasons':reasons})
    pending_count=len(records);excluded=set(exclude)
    records=[r for r in records if r['event_id'] not in excluded]
    records.sort(key=lambda r:(-r['priority_score'],r['time'],r['event_id']))
    # Keep some exploration so disagreement alone does not hide confidently
    # wrong examples. Numeric ranking scores are NOT accuracy probabilities.
    seed=int(hashlib.sha256((doc['source_sha256']+':'+str(doc['revision'])).encode()).hexdigest()[:16],16)
    rng=random.Random(seed);selected=[];per_class=Counter()
    def add(row,enforce_class=True,enforce_time=True):
        if any(x['event_id']==row['event_id'] for x in selected):return False
        if enforce_class and per_class[row['class_for_diversity']]>=4:return False
        if enforce_time and any(x['player']==row['player'] and abs(x['time']-row['time'])<.8 for x in selected):return False
        selected.append(row);per_class[row['class_for_diversity']]+=1;return True
    target=min(size,len(records));priority_target=max(0,target-min(4,target//5))
    for row in records:
        if len(selected)>=priority_target:break
        add(row)
    pool=[r for r in records if r not in selected];rng.shuffle(pool)
    for row in pool:
        if len(selected)>=target:break
        if add(row):row['reasons']=row['reasons']+['探索抽查：覆盖更多动作']
    # Small pools still need a usable batch; relax class cap, then spacing.
    for enforce_time in [True,False]:
        for row in records:
            if len(selected)>=target:break
            add(row,False,enforce_time)
    return {'version':VERSION,'dataset_revision':doc['revision'],'source_sha256':doc['source_sha256'],
            'event_ids':[r['event_id'] for r in selected],'items':selected,'pending_count':pending_count,
            'deferred_count':pending_count-len(records),
            'batch_size':len(selected),'policy':'Priority + class/time diversity + up to 20% exploration',
            'automatic_labels_created':0,'note':'Review acquisition is not model training or an unbiased evaluation sample.'}
