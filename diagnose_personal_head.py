"""Fixed development ablations; no labels, active model or predictions changed."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import numpy as np
import torch

from analysis_jobs import DATA, atomic_json, file_hash
from personal_head import fit, kernel, scores
from personal_learning import extractor
from stroke_features import HitFeatures
from stroke_review import folder_for, load_source, read_labels, usable_training_label
from stroke_taxonomy import AUTHOR_MAP, LABELS

TRAIN_JOB = 'J_fe1e058fc75e475d9c030caced6ec648'
TARGET_JOB = 'J_e23c1dd9297d4bfca7c91d1d95065b68'
RUN = 'cff4a07321d84ed98833e7c7a9e94329'


def model_for(x, y, alpha, balanced=False, centroid=False):
    model = fit(x, y, {'features': 'local_bst', 'kernel': 'linear', 'alpha': alpha})
    if centroid:
        model['centroids'] = np.asarray([model['train_features'][y == c].mean(0) for c in model['classes']])
    elif balanced:
        counts = Counter(y)
        w = np.sqrt(np.asarray([len(y)/(len(counts)*counts[c]) for c in y]))
        k = kernel(model['train_features'], model['train_features'], 'linear')
        yy = np.eye(len(model['classes']))[[list(model['classes']).index(c) for c in y]]
        model['weights'] = w[:, None] * np.linalg.solve(w[:, None]*k*w[None, :] + alpha*np.eye(len(y)), w[:, None]*yy)
    return model


def score(model, x):
    if 'centroids' not in model:
        return scores(model, x)
    from personal_head import normalized
    z = normalized(x, model)
    return -((z[:, None, :]-model['centroids'][None, :, :])**2).sum(2)


def metrics(y, guesses):
    hit = guesses == y
    positive = y != 'no_stroke'
    per = {LABELS[c]: {'total': int((y == c).sum()), 'correct': int(hit[y == c].sum())} for c in sorted(set(y))}
    recalls = [v['correct']/v['total'] for v in per.values()]
    return {'correct': int(hit.sum()), 'total': len(y), 'accuracy': float(hit.mean()),
            'macro_recall': float(np.mean(recalls)), 'stroke_only_correct': int(hit[positive].sum()),
            'stroke_only_total': int(positive.sum()), 'false_no_stroke': int((positive & (guesses == 'no_stroke')).sum()),
            'distribution': dict(Counter(map(str, guesses))), 'per_class': per}


def filtered_values(classifier, features, event):
    indices = range(event['start_frame'], event['end_frame']+1)
    joints, shuttle = [], []
    h = classifier.helpers
    for i in indices:
        frame = np.zeros((2, 17, 2))
        for n, side in enumerate(classifier.order):
            pose = features.pose(i, side)
            if pose is None:
                continue
            pts, good, _, _, _, person = pose
            normalized = h['normalize_joints'](pts[None], np.asarray([person['bbox']]), center_align=True)[0]
            normalized[~good] = 0
            frame[n] = normalized
        joints.append(frame)
        shuttle.append(features.ball[i] if features.valid[i] else [0, 0])
    s = h['normalize_shuttlecock'](np.asarray(shuttle), classifier.dense['source_width'], classifier.dense['source_height'])
    j, p, s, length = h['make_seq_len_same'](100, np.asarray(joints), np.zeros((len(joints), 2, 2)), s)
    j = np.concatenate([j, h['create_bones'](j, classifier.bones)], axis=-2).reshape(100, 2, 72)
    return j.astype('float32'), p.astype('float32'), s.astype('float32'), length


def main():
    torch.set_num_threads(1)
    base = folder_for(TRAIN_JOB)/'学习记录'/RUN
    manifest = json.loads((base/'manifest.json').read_text())
    with np.load(base/'features.npz', allow_pickle=False) as d:
        all_ids = list(d['event_ids'])
        train_ids = list(d['trained_event_ids'])
        x = d['features'][[all_ids.index(e) for e in train_ids]].astype(float)
        y, groups = d['targets'], d['groups']
    report, events, event_file = load_source(TARGET_JOB)
    doc = read_labels(TARGET_JOB, report['source_sha256'])
    saved = json.loads((folder_for(TARGET_JOB)/'personal_transfer_predictions.json').read_text())
    if saved['run_id'] != RUN or saved['model_sha256'] != manifest['model_sha256']:
        raise ValueError('冻结模型不匹配')
    selected = [e for e in events if e['id'] in doc['labels'] and usable_training_label(doc['labels'][e['id']])]
    yy = np.asarray([doc['labels'][e['id']]['stroke_type'] for e in selected])
    classifier, _ = extractor(TARGET_JOB)
    features = HitFeatures(classifier)
    inputs = {'automatic': [], 'automatic_21_frames': [], 'human_21_frames': []}
    qualities = {k: [] for k in inputs}
    for event in selected:
        vec, q = features.extract(event)
        inputs['automatic'].append(vec);qualities['automatic'].append(q)
        i = int(np.argmin(abs(features.times-q['feature_center_time'])))
        fixed = {**event, 'peak_time': float(features.times[i]), 'feature_center_locked': True,
                 'start': float(features.times[max(0, i-8)]), 'end': float(features.times[min(len(features.times)-1, i+12)])}
        vec, q = features.extract(fixed)
        inputs['automatic_21_frames'].append(vec);qualities['automatic_21_frames'].append(q)
        vec, q = features.extract({**event, **doc['labels'][event['id']]})
        inputs['human_21_frames'].append(vec);qualities['human_21_frames'].append(q)
    inputs = {k: np.asarray(v, float) for k, v in inputs.items()}
    # Frozen prediction replay must match before interpreting any ablation.
    with np.load(base/'personal_head.npz', allow_pickle=False) as d:
        frozen = {k: d[k] for k in d.files if k != 'config_json'}
        frozen['config'] = json.loads(str(d['config_json']))
    original_rows = {r['event_id']: r for r in saved['events']}
    z = score(frozen, inputs['automatic'])
    frozen_guesses = frozen['classes'][z.argmax(1)]
    if any(g != original_rows[e['id']]['stroke_type'] for g, e in zip(frozen_guesses, selected)):
        raise ValueError('自动输入重放与保存预测不一致')
    differences = [abs(z[i, list(frozen['classes']).index(v['code'])]-v['uncalibrated_score'])
                   for i, e in enumerate(selected) for v in original_rows[e['id']]['ranking']]
    if max(differences) > 1e-5:
        raise ValueError('重放分数变化，请先检查输入版本')
    variants = [('legacy_alpha100', 100., False, False), ('alpha1', 1., False, False),
                ('alpha0.1', .1, False, False), ('balanced_alpha100', 100., True, False),
                ('balanced_alpha1', 1., True, False), ('balanced_alpha0.1', .1, True, False),
                ('class_centroid', 1., False, True)]
    results = {}
    for name, alpha, balanced, centroid in variants:
        guesses = np.empty(len(y), dtype=y.dtype)
        for group in sorted(set(groups)):
            train, test = groups != group, groups == group
            model = model_for(x[train], y[train], alpha, balanced, centroid)
            guesses[test] = model['classes'][score(model, x[test]).argmax(1)]
        model = model_for(x, y, alpha, balanced, centroid)
        result = {'first_video_fixed_group_cv': metrics(y, guesses), 'second_video': {}, 'rows': []}
        for kind, xx in inputs.items():
            guesses = model['classes'][score(model, xx).argmax(1)]
            result['second_video'][kind] = metrics(yy, guesses)
            result['rows'].extend({'event_id': e['id'], 'input_kind': kind, 'human_code': str(t), 'guess': str(g)}
                                  for e, t, g in zip(selected, yy, guesses))
        results[name] = result
        print(name, 'first', result['first_video_fixed_group_cv']['correct'], '/', len(y),
              'second', {k: v['correct'] for k, v in result['second_video'].items()}, flush=True)
    # Isolate retention of the pretrained MLP on exactly the local feature inputs.
    official = {}
    for kind, xx in inputs.items():
        raw = xx[:, :300].astype('float32').copy()
        for i, event in enumerate(selected):
            player = doc['labels'][event['id']]['player'] if kind == 'human_21_frames' else event['player']
            if player == classifier.order[1]:
                raw[i, :200] = np.r_[raw[i, 100:200], raw[i, :100]]
        with torch.inference_mode():
            logits = classifier.model.mlp_head(torch.from_numpy(raw)).numpy()
        guesses = np.asarray([AUTHOR_MAP.get(classifier.classes[i].split('_', 1)[-1], 'unknown') for i in logits.argmax(1)])
        official[kind] = metrics(yy, guesses)
    # Hold the original pretrained head fixed; vary only window/preprocessing.
    input_ablation = {}
    cached_check = []
    for name, short, filtered in [('original_window_raw', False, False),
                                  ('short_window_raw', True, False),
                                  ('original_window_filtered', False, True)]:
        guesses = []
        for event, q in zip(selected, qualities['automatic']):
            e = dict(event)
            if short:
                e['start_frame'] = int(np.searchsorted(features.times, q['feature_start'], side='left'))
                e['end_frame'] = int(np.searchsorted(features.times, q['feature_end'], side='right')-1)
            values = filtered_values(classifier, features, e) if filtered else classifier.inputs(e)[0]
            pred = classifier.predict(values)['bst0'][0]['label']
            guesses.append(AUTHOR_MAP.get(pred.split('_', 1)[-1], 'unknown'))
            if name == 'original_window_raw':
                cached_check.append({'event_id': event['id'], 'saved_code': event['stroke_type'],
                                     'replay_code': guesses[-1], 'supplemental': bool(event.get('from_refinement')),
                                     'original_seconds': float(features.times[e['end_frame']]-features.times[e['start_frame']])})
        input_ablation[name] = metrics(yy, np.asarray(guesses))
    input_ablation['short_window_filtered'] = official['automatic']
    k = kernel(frozen['train_features'], frozen['train_features'], 'linear')
    counts = np.asarray([manifest['class_counts'][c] for c in frozen['classes']])
    prior = counts/counts.sum()
    mean_scores = z.mean(0)
    record = DATA/'击球标注'/'开发对照'/('分类头诊断_'+uuid.uuid4().hex)
    record.mkdir(parents=True)
    atomic_json(record/'target_labels_snapshot.json', doc)
    atomic_json(record/'frozen_predictions_snapshot.json', saved)
    np.savez_compressed(record/'features.npz', train_features=x, targets=y, groups=groups,
                        target_labels=yy, event_ids=np.asarray([e['id'] for e in selected]), **inputs)
    result = {'created_at': datetime.now(timezone.utc).isoformat(), 'training_job': TRAIN_JOB,
              'target_job': TARGET_JOB, 'training_run': RUN, 'target_revision': doc['revision'],
              'scope': 'Development ablations after viewing second-video results; no independent test claim.',
              'first_video_protocol': 'Fixed per-variant connected-group CV, NOT the old nested model-selection result.',
              'second_video_protocol': 'Same nine labels; automatic inputs versus human-corrected inputs; no training on video 2.',
              'replay_max_score_error': float(max(differences)), 'results': results,
              'official_mlp_on_local_inputs': official,
              'official_input_ablation': input_ablation, 'bst_original_replay': cached_check,
              'saved_bst_original_windows': metrics(yy, np.asarray([e['stroke_type'] for e in selected])),
              'quality': qualities,
              'mean_score_prior_correlation': float(np.corrcoef(mean_scores, prior)[0, 1]),
              'kernel_diagonal_median': float(np.median(k.diagonal())),
              'kernel_largest_eigenvalue': float(np.linalg.eigvalsh(k)[-1]),
              'classes': list(map(str, frozen['classes'])), 'class_counts': counts.tolist(),
              'automatic_mean_scores': mean_scores.tolist(),
              'human_center_offsets': [{'event_id': e['id'], 'offset_seconds': float(doc['labels'][e['id']]['peak_time']-q['feature_center_time'])}
                                       for e, q in zip(selected, qualities['automatic'])],
              'evaluation_code_sha256': file_hash(Path(__file__)),
              'feature_code_sha256': file_hash(Path(__file__).with_name('stroke_features.py')),
              'head_code_sha256': file_hash(Path(__file__).with_name('personal_head.py')),
              'training_manifest_sha256': file_hash(base/'manifest.json'),
              'target_candidate_sha256': file_hash(event_file), 'source_sha256': report['source_sha256'],
              'active_model_replaced': False, 'human_labels_created': 0}
    atomic_json(record/'result.json', result)
    print('RESULT', record, flush=True)
    print('official_local', {k: v['correct'] for k, v in official.items()},
          'score_prior_correlation', result['mean_score_prior_correlation'], flush=True)


if __name__ == '__main__':
    main()
