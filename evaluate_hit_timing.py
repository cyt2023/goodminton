"""Frozen human contact benchmark; predictions never receive human centers.

Candidate-linked frame errors do NOT measure whole-video hit recall. Repeated
annotations of one contact stay in the same temporal fold. Model selection uses
only the first recording, with purged temporal groups and nested selection.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

from analysis_jobs import DATA, atomic_json, file_hash
from contact_timing import (VERSION, SEARCH_SECONDS, frame_features, event_inputs,
                           fit_ridge, choose_frame, save_model)

TRAIN_JOB = 'J_fe1e058fc75e475d9c030caced6ec648'
OTHER_JOB = 'J_e23c1dd9297d4bfca7c91d1d95065b68'
BACKUP = Path(__file__).resolve().parent.parent/'05_研究记录/代码快照/2026-10-07_击球帧优化前'
OUT = DATA/'击球帧评估/2026-10-07'


def load(job):
    folder = DATA/'专项分析'/job
    dense = json.loads((folder/'dense_pose.json').read_text())
    trajectory = json.loads((folder/'trajectory.json').read_text())
    labels_path = DATA/'击球标注'/job/'annotations.json'
    labels = json.loads(labels_path.read_text())
    baseline = json.loads((BACKUP/job/'hit_refinement.json').read_text())
    classified = json.loads((DATA/'技术动作'/job/'片段/classified_events.json').read_text())
    if not (dense['source_sha256'] == trajectory['source_sha256'] == labels['source_sha256']
            == baseline['source_sha256'] == classified['methods']['source_sha256']):
        raise ValueError('Timing benchmark sources differ')
    with np.load(BACKUP/job/'hit_detector_support.npz', allow_pickle=False) as z:
        supports = {k: z[k] for k in z.files}
    times = supports['frame_times']
    if len(times) != len(dense['frames']) or not np.allclose(times, [f['source_time'] for f in dense['frames']]):
        raise ValueError('Support frames differ from decoded source timeline')
    features = frame_features(dense, trajectory, supports)
    events = classified['events']+baseline['supplemental_candidates']
    examples = []
    for event in events:
        annotation = labels['labels'].get(event['id'])
        if not annotation:
            continue
        positive = bool(annotation.get('contact_confirmed') and annotation['stroke_type'] not in ['no_stroke', 'unknown'])
        negative = annotation['stroke_type'] == 'no_stroke'
        if not positive and not negative:
            continue
        original = int(np.argmin(abs(times-event['peak_time'])))
        truth = annotation.get('center_frame') if positive else None
        if positive:
            if not isinstance(truth, int) or not 0 <= truth < len(times):
                raise ValueError('Human source contact frame missing or invalid')
            if abs(times[truth]-annotation['peak_time']) > .002:
                raise ValueError('Human source frame and timestamp disagree')
        old = baseline['suggestions'].get(event['id'])
        current = old['suggested_frame'] if old and old['applied'] else original
        anchor_event = {**event, 'peak_time': float(times[current])}
        _, candidates, x = event_inputs(times, features, anchor_event)
        # Nearby repeats are one stroke for training weight and grouping, but
        # retained individually in the paired candidate benchmark.
        duplicate_key = (annotation['player'], round(truth/3)) if positive else ('negative', event['id'])
        examples.append({'event': anchor_event, 'positive': positive, 'truth': truth,
            'human_player': annotation['player'], 'original': original, 'current': current,
            'candidates': candidates, 'x': x, 'duplicate_key': duplicate_key,
            'group': int(event['peak_time']//15),
            'support_start': float(times[candidates[0]]-.14),
            'support_end': float(times[candidates[-1]]+.14)})
    return {'job': job, 'dense': dense, 'times': times, 'features': features, 'examples': examples,
            'labels': labels, 'labels_sha256': file_hash(labels_path), 'baseline': baseline}


def fit(examples, alpha):
    xs, ys, weights = [], [], []
    keys = [e['duplicate_key'] for e in examples]
    for e in examples:
        y = np.exp(-.5*((e['candidates']-e['truth'])/1.5)**2) if e['positive'] else np.zeros(len(e['x']))
        w = 1+8*y
        w = w/np.mean(w)/keys.count(e['duplicate_key'])
        xs.append(e['x']); ys.append(y); weights.append(w)
    return fit_ridge(np.concatenate(xs), np.concatenate(ys), np.concatenate(weights), alpha)


def purged_split(examples, group):
    test = [e for e in examples if e['group'] == group]
    train = [e for e in examples if e['group'] != group and all(
        e['support_end'] < t['support_start'] or e['support_start'] > t['support_end'] for t in test)]
    return train, test


def metrics(rows):
    if not rows:
        return {'samples': 0}
    errors = np.asarray([r['error_frames'] for r in rows])
    milliseconds = np.asarray([r['error_seconds']*1000 for r in rows])
    return {'samples': len(rows), 'median_frames': float(np.median(errors)),
            'mean_frames': float(np.mean(errors)), 'p90_frames': float(np.quantile(errors, .9)),
            'median_ms': float(np.median(milliseconds)), 'mean_ms': float(np.mean(milliseconds)),
            'p90_ms': float(np.quantile(milliseconds, .9)),
            'within_frames': {str(n): {'correct': int((errors <= n).sum()), 'total': len(rows),
                                    'fraction': float((errors <= n).mean())} for n in [0, 1, 3, 6]},
            'improved': sum(r.get('baseline_error_frames', r['error_frames']) > r['error_frames'] for r in rows),
            'worsened': sum(r.get('baseline_error_frames', r['error_frames']) < r['error_frames'] for r in rows)}


def predictions(data, examples, model=None, policy=None, mode='learned'):
    rows, negative_moves = [], 0
    for e in examples:
        detail = choose_frame(data['times'], data['features'], e['event'], model, **policy) if model else {}
        pick = detail['suggested_frame'] if model else e[mode]
        if not e['positive']:
            negative_moves += int(pick != e['current'])
            continue
        rows.append({'event_id': e['event']['id'], 'group': e['group'], 'human_frame': e['truth'],
            'original_frame': e['original'], 'baseline_frame': e['current'], 'predicted_frame': pick,
            'error_frames': abs(pick-e['truth']),
            'error_seconds': float(abs(data['times'][pick]-data['times'][e['truth']])),
            'baseline_error_frames': abs(e['current']-e['truth']),
            'player_matches': e['event']['player'] == e['human_player'], 'selection': detail})
    return rows, negative_moves


POLICIES = [{'minimum_gain': g, 'maximum_move': m} for g in [.04, .08, .12] for m in [.15, .3]]
ALPHAS = [10., 100., 1000.]


def select(data, examples):
    results = []
    groups = sorted({e['group'] for e in examples})
    for alpha in ALPHAS:
        accumulators = [[] for _ in POLICIES]
        for group in groups:
            train, test = purged_split(examples, group)
            if not train or not test:
                continue
            model = fit(train, alpha)
            for index, policy in enumerate(POLICIES):
                rows, _ = predictions(data, test, model, policy)
                accumulators[index].extend(rows)
        for policy, rows in zip(POLICIES, accumulators):
            if not rows:
                continue
            # Protect already precise frames, then improve 3-frame coverage and
            # absolute error. Selection uses only training-side temporal folds.
            damage = sum(r['baseline_error_frames'] <= 1 and r['error_frames'] > 1 for r in rows)
            loss = sum(min(r['error_frames'], 30) for r in rows)/len(rows)
            utility = sum(r['error_frames'] <= 3 for r in rows)/len(rows)-.02*loss-.025*damage
            results.append({'alpha': alpha, 'policy': policy, 'utility': utility,
                            'metrics': metrics(rows), 'precise_frames_damaged': damage})
    if not results:
        raise ValueError('Not enough separated temporal groups')
    chosen = max(results, key=lambda r: r['utility'])
    return chosen, results


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load(TRAIN_JOB)
    atomic_json(OUT/'first_human_snapshot.json', data['labels'])
    examples = data['examples']; outer_rows = []; folds = []
    for group in sorted({e['group'] for e in examples}):
        train, test = purged_split(examples, group)
        chosen, _ = select(data, train)
        model = fit(train, chosen['alpha'])
        rows, _ = predictions(data, test, model, chosen['policy'])
        outer_rows.extend(rows)
        folds.append({'held_out_group': group, 'training_candidates': len(train),
                      'test_candidates': len(test), 'alpha': chosen['alpha'], 'policy': chosen['policy']})
        print('CONTACT_FOLD', group, json.dumps(metrics(rows)), flush=True)
    chosen, configurations = select(data, examples)
    model = fit(examples, chosen['alpha']); save_model(OUT/'contact_ridge.npz', model)
    original, _ = predictions(data, examples, mode='original')
    baseline, _ = predictions(data, examples, mode='current')
    result = {'version': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
        'training_job': TRAIN_JOB, 'first_revision': data['labels']['revision'],
        'first_labels_sha256': data['labels_sha256'], 'first_source_sha256': data['dense']['source_sha256'],
        'scope': 'Candidate-linked contact localization, not whole-video recall. Nested purged 15-second temporal group development evaluation.',
        'source_frames_preserved': True, 'human_labels_created': 0, 'pseudo_labels_created': 0,
        'first_baseline': metrics(baseline), 'first_original': metrics(original),
        'first_nested_rerank': metrics(outer_rows), 'first_rows': outer_rows, 'folds': folds,
        'selected': chosen, 'configurations': configurations,
        'unique_exact_contacts': len({(e['human_player'], e['truth']) for e in examples if e['positive']}),
        'human_no_hit_candidates': sum(not e['positive'] for e in examples),
        'feature_sha256': file_hash(Path(__file__).with_name('contact_timing.py')),
        'evaluator_sha256': file_hash(Path(__file__)),
        'model_sha256': file_hash(OUT/'contact_ridge.npz')}
    # This recording's simple baselines were already inspected during diagnosis;
    # it is a small development transfer check, not a blinded final test.
    other = load(OTHER_JOB); atomic_json(OUT/'second_human_snapshot.json', other['labels'])
    other_baseline, _ = predictions(other, other['examples'], mode='current')
    other_rows, negative_moves = predictions(other, other['examples'], model, chosen['policy'])
    result.update(second_job=OTHER_JOB, second_revision=other['labels']['revision'],
        second_labels_sha256=other['labels_sha256'], second_source_sha256=other['dense']['source_sha256'],
        second_baseline=metrics(other_baseline), second_rerank=metrics(other_rows), second_rows=other_rows,
        second_scope='10 selected reviewed candidates in another same-site video; no negatives or exhaustive stroke labels; not used for ridge fitting or hyperparameter selection. Used for failure diagnosis and runtime guard design, so this is development transfer checking, NOT an independent or blinded test.')
    atomic_json(OUT/'evaluation.json', result)
    print(json.dumps({k: result[k] for k in ['first_baseline', 'first_nested_rerank', 'second_baseline', 'second_rerank', 'selected']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
