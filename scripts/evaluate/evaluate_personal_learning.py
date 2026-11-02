"""Leave connected video segments out; never activate the evaluation heads."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/evaluate/
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from analysis_jobs import DATA, atomic_json, file_hash
from personal_learning import extractor, embedding
from stroke_review import folder_for, load_source
from stroke_taxonomy import LABELS


def evaluate(job):
    folder = folder_for(job)
    state = json.loads((folder / 'learning_state.json').read_text())
    if state['status'] != 'completed':
        raise ValueError('请先完成个人模型学习')
    run = folder / '学习记录' / state['run_id']
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest.get('feature_version'):
        from evaluate_improved_learning import evaluate_current
        return evaluate_current(job)
    snapshot = json.loads((run / 'human_labels_snapshot.json').read_text())
    report, events, result_path = load_source(job)
    indexed = {e['id']: e for e in events}
    # Include ALL source candidates when forming connected segments, not just
    # labeled clips, to prevent unlabeled intermediate clips bridging folds.
    bounds, group_by_id = [], {}
    for e in sorted(events, key=lambda e: e['start']):
        if not bounds or e['start'] > bounds[-1]['end']:
            bounds.append({'start': e['start'], 'end': e['end']})
        else:
            bounds[-1]['end'] = max(bounds[-1]['end'], e['end'])
        group_by_id[e['id']] = len(bounds) - 1
    classifier, dense = extractor(job)
    times = np.asarray([f['source_time'] for f in dense['frames']])
    ids, xs, targets, groups = [], [], [], []
    for eid in manifest['trained_event_ids']:
        label = snapshot['labels'][eid]
        event = dict(indexed[eid])
        event.update(player=label['player'],
                     start_frame=int(np.searchsorted(times, label['start'], side='left')),
                     end_frame=min(len(times)-1, int(np.searchsorted(times, label['end'], side='right'))-1),
                     frame_index=int(np.argmin(abs(times-label['peak_time']))))
        event['end_frame'] = max(event['start_frame'], event['end_frame'])
        values, quality = classifier.inputs(event)
        if quality['paired_player_fraction'] < .5:
            raise ValueError('评估输入质量与训练记录不一致')
        # If a human expanded a window across multiple source segments,
        # merge those segments so that edited support never crosses folds.
        ids.append(eid); xs.append(embedding(classifier, values, label['player']))
        targets.append(label['stroke_type']); groups.append(group_by_id[eid])
    merges = list(range(len(bounds)))
    def root(i):
        while merges[i] != i:
            i = merges[i]
        return i
    for eid in ids:
        label = snapshot['labels'][eid]
        intersect = [i for i, b in enumerate(bounds)
                     if label['start'] < b['end'] and label['end'] > b['start']]
        for i in intersect:
            merges[root(i)] = root(group_by_id[eid])
    groups = np.asarray([root(g) for g in groups])
    if len(set(groups)) < 2:
        raise ValueError('不重叠的片段组不足，无法留出评估')
    x = np.asarray(xs, dtype='float64')
    targets = np.asarray(targets)
    rows, folds = [], []
    for group in sorted(set(groups)):
        test = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        classes = sorted(set(targets[train]))
        mean, scale = x[train].mean(0), x[train].std(0)
        scale = np.where(scale < 1e-5, 1, scale)
        z = np.c_[(x[train]-mean)/scale, np.ones(len(train))]
        y = np.eye(len(classes))[[classes.index(t) for t in targets[train]]]
        weights = z.T @ np.linalg.solve(z @ z.T + manifest['alpha'] * np.eye(len(train)), y)
        scores = np.c_[(x[test]-mean)/scale, np.ones(len(test))] @ weights
        guess = [classes[int(i)] for i in np.argmax(scores, axis=1)]
        for i, prediction in zip(test, guess):
            eid = ids[i]; label = snapshot['labels'][eid]
            rows.append({'event_id': eid, 'segment_group': int(group),
                         'peak_time': label['peak_time'], 'player': label['player'],
                         'human_code': str(targets[i]), 'human_label': LABELS[targets[i]],
                         'prediction_code': prediction, 'prediction_label': LABELS[prediction],
                         'match': bool(targets[i] == prediction),
                         'class_seen_in_training_fold': bool(targets[i] in classes),
                         'original_prediction': indexed[eid]['stroke_label']})
        folds.append({'held_out_group': int(group), 'train_count': len(train),
                      'test_count': len(test), 'training_classes': classes})
    correct = sum(r['match'] for r in rows)
    seen = [r for r in rows if r['class_seen_in_training_fold']]
    confusion = Counter((r['human_label'], r['prediction_label']) for r in rows if not r['match'])
    per_class = {LABELS[c]: {'samples': sum(r['human_code'] == c for r in rows),
                            'correct': sum(r['human_code'] == c and r['match'] for r in rows)}
                 for c in sorted(set(targets))}
    original_correct = sum(indexed[r['event_id']]['stroke_type'] == r['human_code'] for r in rows)
    output = run / '留出评估'
    output.mkdir(exist_ok=True)
    np.savez_compressed(output / 'evaluation_features.npz', event_ids=np.asarray(ids),
                        features=x, targets=targets, segment_groups=groups)
    result = {'job_id': job, 'run_id': state['run_id'],
              'dataset_revision': snapshot['revision'], 'source_sha256': report['source_sha256'],
              'created_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'same-video leave-one-connected-segment-out; NOT new-video validation',
              'group_rule': 'connected original candidate time windows, merged for expanded human windows',
              'normalization': 'mean and scale fitted on each training fold only',
              'alpha': manifest['alpha'], 'hyperparameter_search': False,
              'source_segments': bounds, 'evaluated_segment_groups': len(folds),
              'evaluated_labels': len(rows), 'correct': correct,
              'pooled_accuracy': correct / len(rows),
              'original_model_exact_label_matches': original_correct,
              'fold_unseen_class_count': len(rows)-len(seen),
              'seen_class_correct': sum(r['match'] for r in seen),
              'seen_class_count': len(seen), 'per_class': per_class,
              'confusions': [{'human': a, 'predicted': b, 'count': n}
                             for (a, b), n in confusion.most_common()],
              'folds': folds, 'events': sorted(rows, key=lambda r: r['peak_time']),
              'training_manifest_sha256': file_hash(run/'manifest.json'),
              'candidate_results_sha256': file_hash(result_path),
              'evaluation_code_sha256': file_hash(Path(__file__)),
              'feature_file_sha256': file_hash(output/'evaluation_features.npz'),
              'limitations': ['Same camera and participants throughout.',
                              'Very small per-class sample sizes.',
                              'Some held-out classes have no training examples.',
                              'Human-reviewed windows and player identities used as inputs.',
                              'Segments approximate rallies and are not manually verified rallies.'],
              'global_model_activated': False}
    atomic_json(output / 'result.json', result)
    atomic_json(folder / 'latest_evaluation.json', result)
    print(json.dumps({k: result[k] for k in ['run_id', 'dataset_revision', 'evaluated_segment_groups',
                                           'evaluated_labels', 'correct', 'pooled_accuracy',
                                           'fold_unseen_class_count', 'seen_class_correct',
                                           'seen_class_count', 'per_class', 'confusions']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    evaluate(parser.parse_args().job)
