"""Frozen public-only model comparison; new human labels evaluate, never train.

The read-side helper uses only stdlib so the local server needs no torch import.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from analysis_jobs import DATA, atomic_json, file_hash
from stroke_taxonomy import AUTHOR_MAP, LABELS

OUTPUT = DATA/'公开训练数据/BST_ShuttleSet/本地继续训练'
VERSION = 'public-only-frozen-private-check-v1'
EVALUATION_VERSION = 'public-frozen-type-actor-joint-v2'


def predicted_actor(author_label, quality):
    """Decode the model's person axis without claiming the mapping is verified."""
    axis = author_label.partition('_')[0] if isinstance(author_label, str) else None
    order = quality.get('player_input_order', [])
    valid_order = len(order) == 2 and set(order) == {'near', 'far'}
    side = order[0 if axis == 'Top' else 1] if valid_order and axis in {'Top', 'Bottom'} else None
    # Court calibration alone does not confirm tracking identity / input order.
    confirmed = quality.get('player_order_confirmed') is True
    return {'author_axis': axis, 'player': side,
            'prediction_has_actor': side is not None,
            'mapping_basis': quality.get('player_order_basis', 'not_recorded'),
            'mapping_confirmed': confirmed and valid_order,
            'mapping_status': 'confirmed' if confirmed and valid_order else
                              'unconfirmed' if valid_order else 'unavailable'}


def count_metric(values):
    available = [v for v in values if v is not None]
    correct = sum(v is True for v in available)
    return {'correct': correct, 'total': len(available),
            'accuracy': correct / len(available) if available else None,
            'unavailable': len(values) - len(available)}


def evaluation_metrics(rows, model):
    """Type scoring and assumed-mapping agreement have separate denominators."""
    confirmed = [r for r in rows if r[model + '_actor']['mapping_confirmed']]
    return {
        'type': count_metric([r[model + '_type_correct'] for r in rows]),
        'actor_under_saved_mapping': count_metric([r[model + '_actor_match'] for r in rows]),
        'joint_under_saved_mapping': count_metric([r[model + '_joint_match'] for r in rows]),
        'actor_confirmed_mapping': count_metric([r[model + '_actor_match'] for r in confirmed]),
        'joint_confirmed_mapping': count_metric([r[model + '_joint_match'] for r in confirmed]),
        'mapping_confirmed_rows': len(confirmed),
        'mapping_unconfirmed_rows': sum(r[model + '_actor']['mapping_status'] == 'unconfirmed' for r in rows),
        'mapping_unavailable_rows': sum(r[model + '_actor']['mapping_status'] == 'unavailable' for r in rows),
        'actor_abstentions': sum(r[model + '_actor']['mapping_status'] != 'unavailable' and
                                 not r[model + '_actor']['prediction_has_actor'] for r in rows),
    }


def rally_review_pool(frozen, doc):
    """Rank exchange-supported candidates, without treating phase guesses as labels."""
    rows = sorted(frozen['events'], key=lambda r:r['model_window']['peak_time'])
    old = set(frozen['preexisting_annotation_ids'])
    clusters = []
    for row in rows:
        annotation = {**doc.get('context_annotations', {}).get(row['event_id'], {}),
                      **doc['labels'].get(row['event_id'], {})}
        if annotation.get('play_context') == 'between_rallies' or annotation.get('stroke_type') == 'no_stroke':
            continue
        if not clusters or row['model_window']['peak_time']-clusters[-1][-1]['model_window']['peak_time'] > 3.5:
            clusters.append([])
        clusters[-1].append(row)
    supported = set()
    for cluster in clusters:
        turns = sum(a['player'] != b['player'] for a,b in zip(cluster,cluster[1:]))
        if len(cluster)>=3 and turns>=2:
            supported.update(r['event_id'] for r in cluster)
    primary, reserve, reasons = [], [], {}
    previous = {}
    for row in rows:
        eid = row['event_id'];t = row['model_window']['peak_time'];player=row['player']
        near_duplicate = player in previous and t-previous[player] < .45
        previous[player] = t
        if eid in old:
            continue
        quality = row['input_quality']
        ok = (eid in supported and not near_duplicate and quality['paired_player_fraction']>=.65
              and quality['raw_shuttle_fraction']>=.2)
        (primary if ok else reserve).append(eid)
        reasons[eid] = ('前后有两侧交替击球候选，优先核对；回合状态未确认' if ok else
                       '回合边界、连续回球证据或输入质量待核对，保留在边界候选中')
    return primary, reserve, reasons


def batch_summary():
    path = DATA/'击球标注/公开模型检查批次/latest.json'
    if not path.is_file():
        return None
    batch = json.loads(path.read_text())
    pending_total = priority_total = 0
    for item in batch['jobs']:
        root = DATA/'击球标注'/item['job_id']
        if not (root/'public_only_predictions.json').is_file():
            continue
        frozen = json.loads((root/'public_only_predictions.json').read_text())
        doc = json.loads((root/'annotations.json').read_text()) if (root/'annotations.json').is_file() else {'labels':{}}
        primary, reserve, _ = rally_review_pool(frozen,doc)
        pending = []
        for eid in primary+reserve:
            label = doc['labels'].get(eid,{})
            context = doc.get('context_annotations',{}).get(eid,{})
            if context.get('play_context') == 'between_rallies' or label.get('play_context') == 'between_rallies':continue
            if label and (label.get('contact_confirmed') or label['stroke_type'] in ['unknown','no_stroke']):continue
            pending.append(eid)
        item.update(pending_count=len(pending),priority_pending_count=sum(eid in primary for eid in pending),
                    reserve_pending_count=sum(eid in reserve for eid in pending))
        pending_total += len(pending);priority_total += item['priority_pending_count']
    batch.update(pending_total=pending_total,priority_pending_total=priority_total)
    return batch


def freeze(job):
    # Training has completed before any private video is read here.
    import torch
    from personal_learning import extractor
    from stroke_context import metadata
    from stroke_review import load_source, read_labels, folder_for
    latest = json.loads((OUTPUT/'latest.json').read_text())
    checkpoint = DATA/latest['candidate_checkpoint']
    if latest['personal_training_samples'] != 0 or file_hash(checkpoint) != latest['candidate_model_sha256']:
        raise ValueError('公开模型身份或训练来源不符')
    report, events, source = load_source(job)
    doc = read_labels(job, report['source_sha256'])
    folder = folder_for(job)
    existing_file = folder/'public_only_predictions.json'
    old = None
    if existing_file.is_file():
        old = json.loads(existing_file.read_text())
        if old['training_run_id'] != latest['run_id'] or old['source_sha256'] != report['source_sha256']:
            old = None
        elif not (set(e['id'] for e in events)-set(e['event_id'] for e in old['events'])):
            return old  # Never move the pre-label cutoff forward after user annotation.
    known = {e['event_id'] for e in old['events']} if old else set()
    events = [e for e in events if e['id'] not in known]
    classifier, dense = extractor(job)
    classifier.model.load_state_dict(torch.load(DATA/latest['baseline_checkpoint'], map_location='cpu', weights_only=True), strict=True)
    windows = metadata(dense, events, {**doc, 'labels': {}})
    inputs, rows = [], []
    torch.set_num_threads(2)
    for e in events:
        w = windows['windows'][e['id']]
        v, quality = classifier.inputs({**e, **w})
        p = classifier.predict(v)
        original = p['bst0'][0]['label']
        rows.append({'event_id': e['id'], 'player':e['player'], 'model_window':w,'input_quality':quality,
                     'baseline_author_label':original,
                     'baseline_code':AUTHOR_MAP[original.split('_',1)[-1]],
                     'baseline_scores':p['bst0_all_scores']})
        inputs.append(v)
    classifier.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    classifier.model.eval()
    for row, v in zip(rows, inputs):
        p = classifier.predict(v);name = p['bst0'][0]['label']
        row.update(candidate_author_label=name, candidate_code=AUTHOR_MAP[name.split('_',1)[-1]],
                   candidate_scores=p['bst0_all_scores'], uses_personal_labels=False)
    result = {'version':VERSION,'job_id':job,'training_run_id':latest['run_id'],
              'created_at':datetime.now(timezone.utc).isoformat(),'source_sha256':report['source_sha256'],
              'candidate_results_sha256':file_hash(source),'candidate_model_sha256':latest['candidate_model_sha256'],
              'baseline_model_sha256':latest['initial_model_sha256'], 'freeze_dataset_revision':doc['revision'],
              'preexisting_annotation_ids':sorted(set(old['preexisting_annotation_ids']) | (set(doc['labels']) & {e['id'] for e in events})) if old else sorted(doc['labels']),
              'events':(old['events'] if old else [])+rows, 'personal_training_samples':0,'human_labels_created':0,
              'window_code_sha256':file_hash(Path(__file__).with_name('stroke_context.py')),
              'code_sha256':file_hash(Path(__file__)),
              'decision_rule':'Author 25-class top-1 mapped to mainland basic code, same rule and same automatic window for both models.',
              'scope':'New annotations after frozen predictions only; existing labels excluded. Does not measure whole-video hit recall.'}
    if old:
        result['freeze_dataset_revision'] = old['freeze_dataset_revision']
        result['initial_created_at'] = old.get('initial_created_at', old['created_at'])
        result['append_history'] = old.get('append_history', [])+[{'created_at':result['created_at'],
                                   'added_ids':[r['event_id'] for r in rows], 'existing_predictions_preserved':True}]
    atomic_json(folder/'公开训练模型检查记录'/uuid.uuid4().hex/'predictions.json',result)
    atomic_json(existing_file,result)
    print('PUBLIC_MODEL_FROZEN', job, len(rows), 'new_unlabelled',len(rows)-len(doc['labels']),flush=True)
    return result


def read_review(job, source_sha, doc):
    value = {}
    state = OUTPUT/'state.json'
    if state.is_file():
        value['training'] = json.loads(state.read_text())
    batch = batch_summary()
    if batch:
        value['batch'] = batch
    path = DATA/'击球标注'/job/'public_only_predictions.json'
    if not path.is_file():
        return value
    frozen = json.loads(path.read_text())
    if frozen['source_sha256'] != source_sha:
        return value
    if (value.get('training', {}).get('status') == 'candidate_ready'
            and value['training'].get('run_id') != frozen['training_run_id']):
        return value
    # Freeze is intentionally independent of the CURRENT annotation revision.
    existing = set(frozen['preexisting_annotation_ids'])
    pending, evaluated, excluded = [], [], []
    false_hits, outside = [], []
    for row in frozen['events']:
        eid = row['event_id']
        if eid in existing:
            continue
        label = doc['labels'].get(eid)
        context = doc.get('context_annotations', {}).get(eid, {})
        if (context.get('play_context') == 'between_rallies'
                or (label or {}).get('play_context') == 'between_rallies'):
            outside.append(eid);continue
        if not label:
            pending.append(eid);continue
        code = label['stroke_type']
        if label.get('play_context') == 'between_rallies':
            outside.append(eid);continue
        if code == 'no_stroke':
            false_hits.append(eid);continue
        if code not in AUTHOR_MAP.values() or code == 'unknown' or label.get('wrong_target'):
            excluded.append({'event_id':eid,'reason':'细分类、未知或目标错误，基础球种评价不可用'});continue
        if not label.get('contact_confirmed'):
            pending.append(eid);continue
        target_match = label['player'] == row['player']
        evaluated_row = {'event_id':eid,'human_code':code, 'human_player':label['player'],
                         'automatic_candidate_player':row['player'],
                         'target_match':target_match, 'candidate_player_match':target_match,
                         # Legacy fields preserve the original scoring for historical comparison.
                         'baseline_correct':target_match and row['baseline_code']==code,
                         'candidate_correct':target_match and row['candidate_code']==code}
        for model in ['baseline', 'candidate']:
            actor = predicted_actor(row.get(model + '_author_label'), row.get('input_quality', {}))
            # A model "Unknown" is an abstention, not missing mapping data: it
            # remains in the denominator. Only an unavailable mapping is unscorable.
            actor_match = actor['player'] == label['player'] if actor['mapping_status'] != 'unavailable' else None
            type_correct = row[model + '_code'] == code
            evaluated_row.update({model + '_author_label':row.get(model + '_author_label'),
                                  model + '_code':row[model + '_code'], model + '_actor':actor,
                                  model + '_type_correct':type_correct,
                                  model + '_actor_match':actor_match,
                                  model + '_joint_match':type_correct and actor_match if actor_match is not None else None})
        evaluated.append(evaluated_row)
    total = len(evaluated)
    primary, reserve, reasons = rally_review_pool(frozen, doc)
    summary = {'evaluation_version':EVALUATION_VERSION,
               'training_run_id':frozen['training_run_id'], 'freeze_dataset_revision':frozen['freeze_dataset_revision'],
               'evaluation_revision':doc['revision'],'new_candidate_count':len(frozen['events'])-len(existing),
               'pending_event_ids':pending,'eligible_event_ids':[r['event_id'] for r in frozen['events'] if r['event_id'] not in existing],
               'existing_labels_excluded':len(existing), 'total':total,
               'baseline_correct':sum(r['baseline_correct'] for r in evaluated),
               'candidate_correct':sum(r['candidate_correct'] for r in evaluated),
               'legacy_metrics':{'total':total,
                   'baseline_correct':sum(r['baseline_correct'] for r in evaluated),
                   'candidate_correct':sum(r['candidate_correct'] for r in evaluated),
                   'rule':'Automatic candidate player matches human player AND stroke type matches; BST predicted actor was not checked.'},
               'metrics':{model:evaluation_metrics(evaluated, model) for model in ['baseline', 'candidate']},
               'candidate_actor_agreement':count_metric([r['candidate_player_match'] for r in evaluated]),
               'actor_mapping_note':'人物/联合指标按保存的Top/Bottom输入映射比较；未经人工确认的映射仅表示一致/冲突，不是已核实的身份识别准确率。',
               'evaluated':evaluated,'excluded':excluded,'false_hit_event_ids':false_hits,
               'between_rallies_event_ids':outside,
               'recommended_event_ids':primary, 'reserve_event_ids':reserve,
               'review_reasons':reasons, 'rally_filter_is_verified':False,
               'personal_training_samples':0,'scope':frozen['scope']}
    value['predictions'] = frozen;value['evaluation'] = summary
    return value


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser();parser.add_argument('--job',required=True);args = parser.parse_args()
    freeze(args.job)
