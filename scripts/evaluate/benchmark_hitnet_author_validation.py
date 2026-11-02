"""Frozen all-six-match author-feature validation of Adapted-HitNet.

This evaluates the author's already extracted coordinates, without video/pose
reproduction. Predictions are written before the scorer loads hit annotations.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/evaluate/

import argparse
import csv
import json
from pathlib import Path
import re
import time

import numpy as np

from analysis_jobs import atomic_json
from benchmark_hits_gru import labels_to_events, sha256
from benchmark_hits_hitnet import (UPSTREAM, MODEL_PATH, VALIDATION_MATCHES,
    aggregate_window_predictions, build_windows, load_model)
from benchmark_hit_models import metrics

BASE = UPSTREAM / 'training/data'
OUT = Path(__file__).resolve().parents[3] / '03_数据/找拍模型对照/2026-10-22/author_validation_checks'


def natural(stem):
    return tuple(int(x) for x in re.findall(r'\d+', stem))


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        raise FileExistsError('Preserve declared author validation protocol')
    rows = []
    stems = [p.name.removesuffix('_hit.csv') for p in (BASE / 'shot').glob('*_hit.csv')
             if p.name.split('_set')[0] in VALIDATION_MATCHES]
    for stem in sorted(stems, key=natural):
        paths = {'ball': BASE / 'ball_trajectory' / f'{stem}_ball_predicted.csv',
            'bottom': BASE / 'poses' / f'{stem}_player_bottom.csv',
            'top': BASE / 'poses' / f'{stem}_player_top.csv',
            'court': BASE / 'court' / f'{stem.split("_set")[0]}.out',
            'truth': BASE / 'shot' / f'{stem}_hit.csv'}
        missing = [str(p) for p in paths.values() if not p.is_file()]
        rows.append({'id': stem, 'match_id': stem.split('_set')[0],
            'paths': {k: str(p) for k, p in paths.items()},
            'sha256': {k: sha256(p) for k, p in paths.items() if p.is_file()},
            'missing': missing})
    atomic_json(OUT / 'protocol.json', {
        'scope': 'All available rallies from all six author validation matches; original extracted features, no raw frontend evaluation or new holdout claim.',
        'source': 'https://github.com/raneeng/Adapted-HitNet',
        'revision': 'b56d955c9fb08fc799c0d1b841552a238d65d834',
        'matches': sorted(VALIDATION_MATCHES, key=int), 'cases': rows,
        'weights_sha256': sha256(MODEL_PATH), 'evaluator_sha256': sha256(__file__),
        'selection': 'Every validation-match shot CSV declared before prediction; missing inputs reported, no result-based exclusions.',
        'window_metric': 'Original author target max(hit[i+6:i+12]); original12-frame input, per-rally nonzero x/y minmax; no class rebalance.',
        'frame_metric': 'Additional diagnostic: argmax of author last-six averaged probabilities vs sparse original contact-frame labels. This is not the README window classification metric.',
        'event_metric': 'Consecutive same-player nonzero frame argmax -> rounded interval midpoint; all original nonzero contact labels retained including frame0 edge hits; one-to-one max-cardinality min-error temporal matching.',
        'tolerance_frames': [3, 6],
        'time_scope': 'Source mp4 files are absent. ±3/6 original frames reported as nominal30fps equivalents ±0.1/0.2s; actual source-clock cadence not independently checked.',
        'tuning': 'No training, no threshold search, no validation-score model selection, unchanged released checkpoint.',
        'training_executed': False})
    print('FROZEN', len(rows), 'rallies', flush=True)


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def predict_features(paths):
    """Receives only coordinate files, never the truth path or labels."""
    ball = read_csv(paths['ball'])
    ball = np.array([[float(r['X']), float(r['Y'])] for r in ball])
    bottom = np.genfromtxt(paths['bottom'], delimiter=',', skip_header=1, ndmin=2)
    top = np.genfromtxt(paths['top'], delimiter=',', skip_header=1, ndmin=2)
    corners = np.array([[float(v) for v in line.split(';')]
                       for line in Path(paths['court']).read_text().splitlines() if line.strip()])
    n = min(len(ball), len(bottom), len(top))
    rows = np.hstack([ball[:n], bottom[:n], top[:n]])
    windows = build_windows(rows, corners)
    probs = load_model().predict(windows.astype(np.float32), batch_size=256, verbose=0) if len(windows) else np.zeros((0, 3), np.float32)
    frame_probs = aggregate_window_predictions(probs, n)
    labels = np.argmax(frame_probs, axis=1)
    frames = [{'frame_index': i} for i in range(n)]
    events = labels_to_events(labels, frames, ['near', 'far'], frame_probs)
    return {'window_probs': probs, 'frame_probs': frame_probs, 'frame_labels': labels}, events


def confusion(y, pred):
    result = np.zeros((3, 3), np.int64)
    np.add.at(result, (y, pred), 1)
    return result


def report(cm):
    total = int(cm.sum())
    classes = []
    for i in range(3):
        tp, support, predicted = int(cm[i, i]), int(cm[i].sum()), int(cm[:, i].sum())
        p = tp / predicted if predicted else 0.
        r = tp / support if support else 0.
        classes.append({'class': i, 'precision': p, 'recall': r,
            'f1': 2*p*r/(p+r) if p+r else 0., 'support': support})
    return {'confusion_matrix_true_rows_predicted_columns': cm.tolist(), 'samples': total,
        'accuracy': float(np.trace(cm)/total) if total else None, 'classes': classes,
        'macro_f1': float(np.mean([x['f1'] for x in classes]))}


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert sha256(__file__) == protocol['evaluator_sha256'], 'Evaluator changed after declaration'
    assert sha256(MODEL_PATH) == protocol['weights_sha256']
    for case in protocol['cases']:
        folder = OUT / 'predictions' / case['id']
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / 'prediction.json'
        if target.exists():
            continue
        if case['missing']:
            atomic_json(folder / 'failure.json', {'missing': case['missing']})
            continue
        for key, expected in case['sha256'].items():
            assert sha256(case['paths'][key]) == expected, case['id'] + ':' + key
        started = time.perf_counter()
        arrays, events = predict_features({k: case['paths'][k] for k in ['ball', 'bottom', 'top', 'court']})
        np.savez_compressed(folder / 'probabilities.npz', **arrays)
        atomic_json(target, {'id': case['id'], 'events': events,
            'frames': len(arrays['frame_labels']), 'windows': len(arrays['window_probs']),
            'probabilities_sha256': sha256(folder / 'probabilities.npz'),
            'truth_received_by_detector': False, 'runtime_seconds': time.perf_counter() - started})
        print('PREDICTED', case['id'], len(events), flush=True)


def evaluate():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert sha256(MODEL_PATH) == protocol['weights_sha256']
    cm_window = np.zeros((3, 3), np.int64)
    cm_frame = np.zeros((3, 3), np.int64)
    rows, missing = [], []
    for case in protocol['cases']:
        folder = OUT / 'predictions' / case['id']
        if not (folder / 'prediction.json').exists():
            missing.append(case['id']); continue
        pred = json.loads((folder / 'prediction.json').read_text())
        assert sha256(folder / 'probabilities.npz') == pred['probabilities_sha256']
        assert sha256(case['paths']['truth']) == case['sha256']['truth']
        y = np.asarray([int(r['hit']) for r in read_csv(case['paths']['truth'])])
        with np.load(folder / 'probabilities.npz', allow_pickle=False) as arrays:
            n = len(arrays['frame_labels'])
            if len(y) != n:
                raise ValueError('Feature/label lengths differ: ' + case['id'])
            win_truth = np.asarray([np.max(y[i+6:i+12]) for i in range(max(0, n-11))], dtype=np.int64)
            cm_window += confusion(win_truth, arrays['window_probs'].argmax(1))
            cm_frame += confusion(y, arrays['frame_labels'])
        truth = [{'frame_index': int(i), 'player': 'near' if y[i] == 1 else 'far'} for i in np.flatnonzero(y)]
        row = {'id': case['id'], 'match_id': case['match_id'], 'frames': n,
            'windows': len(win_truth), 'contacts': len(truth), 'predictions': len(pred['events']),
            'events': {str(tol): metrics(pred['events'], truth, 30., True, tol) for tol in [3, 6]}}
        rows.append(row)
    event_summary = {}
    for tol in ['3', '6']:
        matched = sum(r['events'][tol]['matched'] for r in rows)
        nt = sum(r['contacts'] for r in rows)
        npred = sum(r['predictions'] for r in rows)
        event_summary[tol] = {'tolerance_frames': int(tol), 'nominal30fps_equivalent_seconds': int(tol)/30,
            'truth': nt, 'predicted': npred, 'matched': matched,
            'precision': matched/npred if npred else 0., 'recall': matched/nt if nt else 0.,
            'f1': 2*matched/(nt+npred) if nt+npred else 0.,
            'actor_and_time_correct': sum(sum(x['actor_correct'] is True for x in r['events'][tol]['matches']) for r in rows)}
    result = {'scope': protocol['scope'], 'protocol_sha256': sha256(OUT / 'protocol.json'),
        'rallies': len(rows), 'missing': missing, 'window_classification': report(cm_window),
        'sparse_contact_frame_classification_diagnostic': report(cm_frame),
        'event_metrics': event_summary, 'rows': rows,
        'limits': ['Author validation set used for their model development; not a new blind holdout.',
            'Author pre-extracted features, not raw-video detection or our shared frontend.',
            'All sparse contacts including no-lookback edge hits remain in event denominators.', protocol['time_scope']]}
    atomic_json(OUT / 'result.json', result)
    print(json.dumps({k: result[k] for k in ['rallies', 'missing', 'window_classification', 'event_metrics']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'run', 'evaluate'])
    args = parser.parse_args()
    globals()[args.stage]()
