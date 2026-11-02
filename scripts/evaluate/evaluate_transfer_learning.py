"""Evaluate predictions saved before target-video annotation, without fitting.

This is a partial, user-selected candidate review, not full-video accuracy.
The automatic prediction windows are retained even when humans move contacts.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/evaluate/
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from analysis_jobs import DATA, atomic_json, file_hash
from personal_transfer import read_valid
from stroke_review import folder_for, load_source, read_labels, usable_training_label

VERSION = 'frozen-cross-video-candidate-review-v1'


def evaluate(job):
    report, events, candidate_path = load_source(job)
    folder = folder_for(job)
    doc = read_labels(job, report['source_sha256'])
    prediction_path = folder / 'personal_transfer_predictions.json'
    prediction = read_valid(job, report['source_sha256'])
    if prediction is None:
        raise ValueError('没有与当前源录像、模型一致的跨录像预测')
    source_folder = folder_for(prediction['training_job_id']) / '学习记录' / prediction['run_id']
    manifest = json.loads((source_folder / 'manifest.json').read_text())
    if manifest['source_sha256'] == report['source_sha256']:
        raise ValueError('目标录像与训练录像相同，不能作为跨录像检查')
    forecast_time = datetime.fromisoformat(prediction['created_at'])
    if any(datetime.fromisoformat(item['created_at']) <= forecast_time
           for label in doc['labels'].values()
           for item in [label, *label.get('history', [])]):
        raise ValueError('预测未早于本录像全部标注，不能作为标注前冻结检查')
    indexed = {p['event_id']: p for p in prediction['events']}
    rows, excluded, missing = [], [], []
    for event in events:
        label = doc['labels'].get(event['id'])
        if not label:
            missing.append(event['id'])
            continue
        if label.get('reviewer') != 'user' or label.get('evidence_source') != 'human_user_selection':
            raise ValueError('评价标签不是用户人工选择')
        if not usable_training_label(label):
            excluded.append({'event_id': event['id'], 'human_code': label['stroke_type'],
                             'reason': '其他技术、未知、球员不对或未确认击球帧'})
            continue
        forecast = indexed.get(event['id'])
        if forecast is None:
            excluded.append({'event_id': event['id'], 'reason': '没有冻结预测'})
            continue
        rows.append({'event_id': event['id'], 'human_code': label['stroke_type'],
                     'human_label': label['stroke_label'], 'human_player': label['player'],
                     'human_peak_time': label['peak_time'],
                     'automatic_input_peak_time': forecast['input_quality']['feature_center_time'],
                     'personal_code': forecast['stroke_type'],
                     'personal_label': forecast['stroke_label'],
                     'personal_match': forecast['stroke_type'] == label['stroke_type'],
                     'bst_code': event['stroke_type'], 'bst_label': event['stroke_label'],
                     'bst_match': event['stroke_type'] == label['stroke_type'],
                     'class_seen_in_training': label['stroke_type'] in manifest['trained_classes']})
    if not rows:
        raise ValueError('尚无可评价的人工标注')
    def metrics(selected):
        n = len(selected)
        a = sum(r['personal_match'] for r in selected)
        b = sum(r['bst_match'] for r in selected)
        return {'total': n, 'personal_correct': a, 'bst_correct': b,
                'personal_accuracy': a/n if n else None, 'bst_accuracy': b/n if n else None}
    result = {'version': VERSION, 'job_id': job, 'source_sha256': report['source_sha256'],
              'dataset_revision': doc['revision'], 'run_id': prediction['run_id'],
              'training_job_id': prediction['training_job_id'],
              'training_dataset_revision': prediction['training_dataset_revision'],
              'model_sha256': prediction['model_sha256'],
              'prediction_file_sha256': file_hash(prediction_path),
              'candidate_results_sha256': file_hash(candidate_path),
              'evaluation_code_sha256': file_hash(Path(__file__)),
              'created_at': datetime.now(timezone.utc).isoformat(),
              'prediction_created_at': prediction['created_at'],
              'candidate_count': len(events), 'annotated_count': len(doc['labels']),
              'confirmed_contact_count': sum(x.get('contact_confirmed', False) for x in doc['labels'].values()),
              'unlabelled_event_ids': missing, 'excluded': excluded, 'events': rows,
              'metrics': metrics(rows),
              'stroke_only': metrics([r for r in rows if r['human_code'] != 'no_stroke']),
              'seen_classes_only': metrics([r for r in rows if r['class_seen_in_training']]),
              'per_class': {name: metrics([r for r in rows if r['human_label'] == name])
                            for name in sorted({r['human_label'] for r in rows})},
              'human_class_counts': dict(Counter(r['human_label'] for r in rows)),
              'false_no_stroke_count': sum(r['human_code'] != 'no_stroke' and r['personal_code'] == 'no_stroke' for r in rows),
              'model_retrained': False, 'human_labels_created': 0,
              'scope': 'New video; predictions frozen before labels; partial user-selected candidates, not full-video accuracy.',
              'input_policy': 'Score saved automatic windows; do not recompute using human-corrected contact windows.'}
    record = folder / '跨录像评估' / uuid.uuid4().hex
    atomic_json(record / 'human_labels_snapshot.json', doc)
    atomic_json(record / 'predictions_snapshot.json', prediction)
    result['evidence_folder'] = str(record.relative_to(DATA))
    atomic_json(record / 'result.json', result)
    atomic_json(folder / 'transfer_evaluation.json', result)
    print(json.dumps({'revision': doc['revision'], 'metrics': result['metrics'],
                      'unlabelled': len(missing), 'excluded': len(excluded)}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    evaluate(parser.parse_args().job)
