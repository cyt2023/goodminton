"""Human stroke labels, blind review queue, provenance and revision history."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import threading

from analysis_jobs import DATA, atomic_json, file_hash
from stroke_taxonomy import LABELS, PRIMARY, TRAINABLE, VERSION, normalise_event, AUTHOR_MAP

LOCK = threading.RLock()
IDENTIFIER = re.compile(r'J_[0-9a-f]{32}')


class ReviewConflict(ValueError):
    pass


def folder_for(job):
    if not isinstance(job, str) or not IDENTIFIER.fullmatch(job):
        raise ValueError('分析记录标识无效')
    return DATA / '击球标注' / job


def load_source(job):
    folder_for(job)
    root = DATA / '专项分析' / job
    report = json.loads((root / 'report.json').read_text())
    from analysis_jobs import registered_video
    video = registered_video(report['source_video_id'], allow_reference=True)['path']
    stat = video.stat()
    # File identity is checked on initial/replaced file; avoid hashing the
    # complete video on every click or progress poll.
    cache_key = (str(video), stat.st_size, stat.st_mtime_ns)
    with LOCK:
        known = SOURCE_HASHES.get(cache_key)
        if known is None:
            known = file_hash(video)
            SOURCE_HASHES[cache_key] = known
    if known != report['source_sha256']:
        raise ValueError('原视频已改变，请重新分析')
    result_path = DATA / '技术动作' / job / '片段' / 'classified_events.json'
    classified = json.loads(result_path.read_text())
    if classified['methods']['source_sha256'] != report['source_sha256']:
        raise ValueError('动作结果与原视频不匹配')
    events = [normalise_event(e) for e in classified['events']]
    from hit_refinement import read
    refinement=read(job,report['source_sha256'])
    if refinement:
        known={e['id'] for e in events}
        events.extend(normalise_event(e) for e in refinement['supplemental_candidates'] if e['id'] not in known)
    return report, events, result_path


SOURCE_HASHES = {}


def read_labels(job, source_sha):
    file = folder_for(job) / 'annotations.json'
    if not file.is_file():
        return {'schema_version': 1, 'job_id': job, 'source_sha256': source_sha,
                'taxonomy_version': VERSION, 'revision': 0, 'labels': {}}
    doc = json.loads(file.read_text())
    if doc['source_sha256'] != source_sha or doc['job_id'] != job:
        raise ValueError('标注与原视频不匹配')
    for eid, context in doc.get('context_annotations', {}).items():
        if eid in doc['labels']:
            doc['labels'][eid] = {**doc['labels'][eid], 'play_context':context['play_context']}
    return doc


def first_batch(events, size=30):
    # Human labels are never generated here. Balance *model guesses*, then
    # include uncertain/rejected events and temporally dispersed examples.
    bins = defaultdict(list)
    for event in events:
        bins[event['stroke_type']].append(event)
    selected, used = [], set()
    for group in bins.values():
        group.sort(key=lambda e: e.get('bst_softmax_score', 0), reverse=True)
    keys = [k for k in ['clear', 'lift', 'push', 'net_kill', 'drop', 'smash',
                       'drive', 'net', 'net_cross', 'defense', 'serve_short',
                       'serve_long', 'unknown'] if bins[k]]
    while len(selected) < min(size, len(events)):
        added = False
        for key in keys:
            group = bins[key]
            remaining = [e for e in group if e['id'] not in used]
            if not remaining:
                continue
            # Alternate strong and weak predictions within each class.
            event = remaining[0] if len(selected) % 2 == 0 else remaining[-1]
            selected.append(event['id']); used.add(event['id']); added = True
            if len(selected) == size:
                break
        if not added:
            break
    return selected


def usable_training_label(label):
    return (label['stroke_type'] in TRAINABLE and label.get('play_context') != 'between_rallies' and not label.get('wrong_target') and
            (label['stroke_type']=='no_stroke' or label.get('contact_confirmed') is True))


def learning_readiness(doc):
    eligible = [v for v in doc['labels'].values()
                if usable_training_label(v)]
    compatible = [v for v in eligible if v['stroke_type'] in set(AUTHOR_MAP.values())]
    counts = Counter(v['stroke_type'] for v in compatible)
    ready = len(compatible) >= 12 and sum(n >= 3 for n in counts.values()) >= 2
    return {'eligible_labels': len(compatible), 'retained_labels':len(eligible), 'class_counts': dict(counts),
            'can_train': ready,
            'requirement': '当前分类学习需要本段至少12个已确认的基础球种，且至少两类各有3个。细分技术和无击球单独留存；看不清、回合外、球员不对或击球帧未确认不参与分类训练。'}


def get_review(job,exclude_ids=()):
    with LOCK:
        report, events, result_path = load_source(job)
        doc = read_labels(job, report['source_sha256'])
        folder = folder_for(job); folder.mkdir(parents=True, exist_ok=True)
        queue_file = folder / 'queue.json'
        if queue_file.is_file():
            queue = json.loads(queue_file.read_text())
            if queue['source_sha256'] != report['source_sha256']:
                raise ValueError('片段列表与原视频不匹配')
            ids = set(e['id'] for e in events)
            if any(i not in ids for i in queue['first_batch']):
                raise ValueError('片段列表需重新生成')
        else:
            queue = {'source_sha256': report['source_sha256'],
                     'candidate_results_sha256': file_hash(result_path),
                     'selection': 'model-class-balanced plus uncertain candidates; not representative evaluation sample',
                     'first_batch': first_batch(events)}
            atomic_json(queue_file, queue)
        records = []
        for e in sorted(events, key=lambda e: e['peak_time']):
            ranked = (e.get('specialist_predictions') or {}).get('bst0', [])
            records.append({'id': e['id'], 'player': e['player'],
                            'start': e['start'], 'end': e['end'], 'peak_time': e['peak_time'],
                            'model_label': e['stroke_label'], 'model_code': e['stroke_type'],
                            'model_score': e.get('bst_softmax_score'),
                            'model_top5': [{'label': normalise_event({'specialist_predictions': {'bst0': [p]}})['stroke_label'],
                                           'score': p['softmax_score'], 'source_label': p['label']} for p in ranked],
                            'thumbnail_url': f'/action-preview/{job}/{e["id"]}.jpg',
                            'has_model_input': (result_path.parent/e['id']/'bst_inputs.npz').is_file()})
            records[-1]['from_refinement']=bool(e.get('from_refinement'))
        training_file = folder / 'learning_state.json'
        training = json.loads(training_file.read_text()) if training_file.is_file() else {'status': 'not_trained', 'message': '等待你的标注；模型尚未学习。'}
        adaptation_file = folder / 'adaptation_state.json'
        if adaptation_file.is_file():
            adaptation = json.loads(adaptation_file.read_text())
            if adaptation.get('source_sha256') in [None, report['source_sha256']]:
                training = adaptation
        from bst_finetune_state import learning_state as direct_state, library_readiness, job_predictions
        direct = direct_state()
        if direct:
            training = {**direct, 'algorithm': 'direct_bst_mlp', 'phase': direct['status'],
                        'status': 'running' if direct['status'] in ['extracting', 'training'] else direct['status']}
        direct_predictions = job_predictions(job, report['source_sha256'], doc['revision'],
                                             current_results_sha256=file_hash(result_path))
        predictions_file = folder / 'personal_predictions.json'
        learned = json.loads(predictions_file.read_text()) if predictions_file.is_file() else None
        transfer_file=folder/'personal_transfer_predictions.json'
        if learned is None and transfer_file.is_file():
            from personal_transfer import read_valid
            learned=read_valid(job,report['source_sha256'])
        if learned and learned.get('source_sha256') != report['source_sha256']:
            learned = None
        transfer_evaluation_file = folder / 'transfer_evaluation.json'
        transfer_evaluation = json.loads(transfer_evaluation_file.read_text()) if transfer_evaluation_file.is_file() else None
        if transfer_evaluation and (
                transfer_evaluation.get('source_sha256') != report['source_sha256']
                or transfer_evaluation.get('dataset_revision') != doc['revision']
                or not learned or not learned.get('training_job_id')
                or transfer_evaluation.get('run_id') != learned.get('run_id')
                or transfer_evaluation.get('model_sha256') != learned.get('model_sha256')
                or transfer_evaluation.get('prediction_file_sha256') != file_hash(transfer_file)):
            transfer_evaluation = None
        evaluation_file = folder / 'latest_evaluation.json'
        evaluation = json.loads(evaluation_file.read_text()) if evaluation_file.is_file() else None
        if evaluation and (evaluation.get('source_sha256') != report['source_sha256']
                           or evaluation.get('run_id') != training.get('run_id')):
            evaluation = None
        development_file=folder/'context_development_evaluation.json'
        development=json.loads(development_file.read_text()) if development_file.is_file() else None
        if development and development.get('source_sha256')!=report['source_sha256']:development=None
        from hit_segments import metadata
        frame_review = metadata(job, events, doc)
        from hit_segments import dense_data
        from stroke_context import metadata as context_metadata
        analysis_review = context_metadata(dense_data(job), events, doc)
        preserved_file = folder / 'preserved_bst_predictions.json'
        preserved = json.loads(preserved_file.read_text()) if preserved_file.is_file() else None
        if preserved and (preserved.get('source_sha256') != report['source_sha256']
                          or preserved.get('dataset_revision') != doc['revision']
                          or preserved.get('code_sha256') != file_hash(Path(__file__).with_name('preserved_bst.py'))
                          or preserved.get('window_code_sha256') != file_hash(Path(__file__).with_name('stroke_context.py'))):
            preserved = None
        from hit_refinement import read
        from active_review import build
        refinement=read(job,report['source_sha256'])
        if len(exclude_ids)>500 or any(not isinstance(eid,str) or not re.fullmatch(r'[0-9a-f]{32}',eid) for eid in exclude_ids):
            raise ValueError('跳过片段列表无效')
        active=build(events,doc,learned,refinement,exclude=exclude_ids,evaluation=evaluation)
        from public_bst_review import read_review as public_review
        public_check = public_review(job, report['source_sha256'], doc)
        from analysis_jobs import registered_video
        source = registered_video(report['source_video_id'], allow_reference=True)
        return {'job_id': job, 'source_sha256': report['source_sha256'],
                'source_kind': source['kind'], 'source_name': source['name'],
                'source_video_id': report['source_video_id'], 'duration': report['source_duration'],
                'taxonomy_version': VERSION, 'vocabulary': LABELS, 'primary': PRIMARY,
                'events': records, 'first_batch': queue['first_batch'],
                'annotations': doc['labels'], 'revision': doc['revision'],
                'context_annotations': doc.get('context_annotations', {}),
                'learning_readiness': learning_readiness(doc), 'learning': training,
                'library_learning_readiness': library_readiness(),
                'direct_bst_learning': direct,
                'direct_bst_predictions': direct_predictions,
                'learned_predictions': learned,
                'evaluation': evaluation,
                'transfer_evaluation': transfer_evaluation,
                'development_comparison':{'dataset_revision':development['dataset_revision'],
                    'baseline_run_id':development['baseline_run_id'],'protected_group_count':development['protected_group_count'],
                    'variants':{name:{k:value['metrics'][k] for k in ['correct','total','accuracy','macro_f1','stroke_only_correct','stroke_only_total']}
                                for name,value in development['variants'].items()},
                    'active_model_replaced':False,'scope':development['scope']} if development else None,
                'frame_review': frame_review,
                'analysis_review': analysis_review,
                'preserved_predictions': preserved,
                'public_check': public_check,
                'active_review':active,
                'hit_refinement':{k:refinement[k] for k in ['version','summary','audio','evaluation','contact_timing'] if k in refinement} if refinement else None,
                'model_guesses_hidden_by_default': True}


def finite(value):
    return type(value) in [float, int] and math.isfinite(value)


def save_label(job, body):
    with LOCK:
        report, events, result_path = load_source(job)
        doc = read_labels(job, report['source_sha256'])
        if type(body.get('revision')) is not int or body['revision'] != doc['revision']:
            raise ReviewConflict('标注已在另一窗口改变，请刷新后再保存')
        if body.get('source_sha256') != report['source_sha256']:
            raise ValueError('标注来源不匹配')
        event = next((e for e in events if e['id'] == body.get('event_id')), None)
        if event is None:
            raise ValueError('片段不存在')
        stroke = body.get('stroke_type')
        if stroke not in LABELS or stroke == 'serve':
            raise ValueError('请选择一种技术、没有击球或看不清')
        player = body.get('player')
        if player not in ['near', 'far']:
            raise ValueError('请确认候选球员 A 或 B')
        start, end, peak = [body.get(k) for k in ['start', 'end', 'peak_time']]
        if not all(finite(v) for v in [start, end, peak]) or not 0 <= start <= peak < end <= report['source_duration']:
            raise ValueError('片段和击球位置超出录像范围')
        if end-start > 12 or abs(peak-event['peak_time']) > 4:
            raise ValueError('调整超出这次动作范围，请选择对应片段')
        style = body.get('swing_style', 'unknown')
        if style not in ['forehand', 'backhand', 'unknown']:
            raise ValueError('正反手选项无效')
        if type(body.get('wrong_target', False)) is not bool:
            raise ValueError('球员核对字段无效')
        play_context = body.get('play_context', doc.get('context_annotations', {}).get(event['id'], {}).get('play_context', 'unknown'))
        if play_context not in ['unknown', 'rally', 'between_rallies']:
            raise ValueError('请核对是否为回合内动作')
        when = datetime.now(timezone.utc).isoformat()
        record = {'event_id': event['id'], 'source_sha256': report['source_sha256'],
                  'stroke_type': stroke, 'stroke_label': LABELS[stroke],
                  'player': player, 'swing_style': style,
                  'wrong_target': body.get('wrong_target', False),
                  'start': round(start, 6), 'end': round(end, 6), 'peak_time': round(peak, 6),
                  'contact_confirmed': body.get('contact_confirmed') is True,
                  'play_context': play_context,
                  'context_evidence_source': 'human_user_selection' if 'play_context' in body else 'not_annotated',
                  'reviewer': 'user', 'evidence_source': 'human_user_selection',
                  'taxonomy_version': VERSION, 'created_at': when,
                  'note': str(body.get('note', ''))[:1000],
                  'model_prediction_at_review': {k:event.get(k) for k in ['stroke_type','source_model_label','bst_softmax_score']},
                  'candidate_results_sha256': file_hash(result_path),
                  'history': doc['labels'].get(event['id'], {}).get('history', [])}
        refinement_path=DATA/'专项分析'/job/'hit_refinement.json'
        if refinement_path.is_file():record['hit_refinement_sha256']=file_hash(refinement_path)
        record['candidate_origin']='audio_visual_gru_supplemental' if event.get('from_refinement') else 'original_visual_candidate'
        if body.get('window_basis') == 'source_frames_8_before_12_after':
            from hit_segments import dense_data, segment
            window = segment(dense_data(job), event, {**record,'center_override':True})
            if abs(window['start']-start)>1e-5 or abs(window['end']-end)>1e-5:
                raise ValueError('逐帧窗口与原视频不一致，请刷新后重试')
            record.update(window_basis=body['window_basis'],center_frame=window['center_frame'],
                          start_frame=window['start_frame'],end_frame=window['end_frame'],
                          source_context_start=event['start'],source_context_end=event['end'])
        previous = doc['labels'].get(event['id'])
        if previous:
            record['history'] = record['history'] + [{k:v for k,v in previous.items() if k!='history'}]
        doc['labels'][event['id']] = record
        if 'play_context' in body:
            doc.setdefault('context_annotations', {})[event['id']] = {
                'play_context':play_context, 'evidence_source':'human_user_selection', 'created_at':when}
        doc.update(revision=doc['revision']+1, updated_at=when)
        atomic_json(folder_for(job)/'annotations.json', doc)
        from hit_segments import metadata
        atomic_json(folder_for(job)/'hit_segments.json',metadata(job,events,doc))
        export = {**doc, 'labels': list(doc['labels'].values()),
                  'learning_readiness': learning_readiness(doc),
                  'note': 'Human labels are training data. Saving is not model training; this first balanced batch is a development set.'}
        atomic_json(folder_for(job)/'training_labels.json', export)
        return {'revision': doc['revision'], 'annotation': record,
                'learning_readiness': learning_readiness(doc)}


def overlays(job, event_id):
    report, events, _ = load_source(job)
    event = next((e for e in events if e['id'] == event_id), None)
    if event is None:
        raise ValueError('片段不存在')
    dense = json.loads((DATA/'专项分析'/job/'dense_pose.json').read_text())
    lo, hi = max(0, event['peak_time']-5), min(report['source_duration'], event['peak_time']+7)
    boxes = [{'time': f['source_time'], 'players': {side:p['bbox'] for side,p in f['players'].items() if p}}
             for i,f in enumerate(dense['frames']) if i%3==0 and lo <= f['source_time'] <= hi]
    return {'width': dense['source_width'], 'height': dense['source_height'], 'fps':dense['sample_fps'], 'frames': boxes}


def human_overrides(job, source_sha):
    with LOCK:
        return read_labels(job, source_sha)['labels']
