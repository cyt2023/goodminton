"""Bounded residual adaptation with real author data and personal labels.

The pretrained backbone and MLP stay frozen. This is a development experiment;
it does not deploy a head, turn guesses into labels, or detect no-stroke events.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import numpy as np
import torch

from analysis_jobs import DATA, atomic_json, file_hash
from bst_public_data import FOLDER
from personal_head import fit, kernel, scores, save
from personal_learning import extractor, embedding
from specialist_model import ROOT, BST_NAME
from stroke_context import metadata
from stroke_review import folder_for, load_source, read_labels, usable_training_label
from stroke_taxonomy import AUTHOR_MAP, LABELS

VERSION = 'frozen-teacher-public-residual-v1'
TRAIN_JOB = 'J_fe1e058fc75e475d9c030caced6ec648'
OTHER_JOB = 'J_e23c1dd9297d4bfca7c91d1d95065b68'
CODES = list(dict.fromkeys(AUTHOR_MAP.values()))


def prior(logits, labels):
    p = np.exp(logits-logits.max(1, keepdims=True));p /= p.sum(1, keepdims=True)
    result = np.zeros((len(p), len(CODES)))
    for i, name in enumerate(labels):
        result[:, CODES.index(AUTHOR_MAP[name.split('_', 1)[-1]])] += p[:, i]
    return result


def personal(job, reviewed):
    report, events, _ = load_source(job)
    doc = read_labels(job, report['source_sha256'])
    classifier, dense = extractor(job)
    windows = metadata(dense, events, doc if reviewed else {**doc, 'labels': {}})
    xx, pp, yy, ids, quality = [], [], [], [], []
    excluded = []
    for event in events:
        label = doc['labels'].get(event['id'])
        if not label or not usable_training_label(label) or label['stroke_type'] not in CODES:
            if label:
                excluded.append({'event_id': event['id'], 'code': label['stroke_type'],
                                 'reason': 'Only author-supported basic stroke classes; no-stroke and fine custom labels excluded.'})
            continue
        w = windows['windows'][event['id']]
        values, q = classifier.inputs({**event, **w})
        if reviewed and q['paired_player_fraction'] < .5:
            excluded.append({'event_id': event['id'], 'code': label['stroke_type'], 'reason': 'Pair pose coverage below 50%'})
            continue
        vector = embedding(classifier, values, w['player'])
        raw = vector.copy()
        if w['player'] == classifier.order[1]:
            raw[:200] = np.r_[vector[100:200], vector[:100]]
        with torch.inference_mode():
            logits = classifier.model.mlp_head(torch.from_numpy(raw[None])).numpy()
        xx.append(vector);pp.append(prior(logits, classifier.classes)[0]);yy.append(label['stroke_type']);ids.append(event['id'])
        quality.append({'event_id': event['id'], 'model_window': w, 'quality': q})
    return np.asarray(xx), np.asarray(pp), np.asarray(yy), ids, quality, excluded, doc


def measure(y, p):
    guesses = np.asarray(CODES)[p.argmax(1)]
    return {'correct': int((y == guesses).sum()), 'total': len(y),
            'per_class': {LABELS[c]: {'total': int((y == c).sum()), 'correct': int((guesses[y == c] == c).sum())}
                          for c in sorted(set(y))}}


def corrected(model, x, teacher):
    residual = scores(model, x)
    # At most 0.05 absolute correction to each frozen teacher probability.
    p = np.maximum(1e-7, teacher + .25*np.clip(residual, -.2, .2))
    return p/p.sum(1, keepdims=True)


def train(train_job=TRAIN_JOB):
    torch.set_num_threads(1)
    state_folder = folder_for(train_job)
    def state(status, message, **extra):
        atomic_json(state_folder/'adaptation_state.json', {'status': status, 'message': message,
                    'updated_at': datetime.now(timezone.utc).isoformat(), 'pretrained_mlp_preserved': True, **extra})
    source, _, _ = load_source(train_job)
    current_doc = read_labels(train_job, source['source_sha256'])
    state('running', '正在保留BST骨干与分类层，结合公开训练样本学习个人校正',
          source_sha256=source['source_sha256'], dataset_revision=current_doc['revision'])
    with np.load(FOLDER/'开发样本/features.npz', allow_pickle=False) as d:
        all_x = d['embeddings'];labels = list(d['class_labels']);raw_y = d['labels'];split = d['splits'];logits = d['logits']
    # Canonical target/opponent order for the small residual only.
    all_x = all_x.copy()
    for i, target in enumerate(raw_y):
        if labels[target].startswith('Bottom_'):
            all_x[i, :200] = np.r_[all_x[i, 100:200], all_x[i, :100]]
    all_p = prior(logits, labels)
    all_y = np.asarray([AUTHOR_MAP[labels[y].split('_', 1)[-1]] for y in raw_y])
    train, val = split == 'train', split == 'val'
    x, p, y, ids, quality, excluded, doc = personal(train_job, True)
    if len(y) == 0:
        raise ValueError('没有兼容的个人击球标签')
    xx = np.r_[all_x[train], x];pp = np.r_[all_p[train], p];yy = np.r_[all_y[train], y]
    model = fit(xx, yy, {'features': 'local_bst', 'kernel': 'linear', 'alpha': 1.})
    # Keep all 13 author basic classes and learned unknown; no-stroke is separate.
    model['classes'] = np.asarray(CODES)
    targets = np.eye(len(CODES))[[CODES.index(c) for c in yy]]-pp
    count = Counter(yy)
    weights = np.asarray([len(yy)/(len(count)*count[c]) for c in yy])
    weights[-len(y):] *= 2.
    w = np.sqrt(weights)
    k = kernel(model['train_features'], model['train_features'], 'linear')
    model['weights'] = w[:, None]*np.linalg.solve(w[:, None]*k*w[None, :]+np.eye(len(yy)), w[:, None]*targets)
    checks = {'author_val_basic_classes': {'base': measure(all_y[val], all_p[val]),
                                         'adapted': measure(all_y[val], corrected(model, all_x[val], all_p[val]))}}
    target_data = {}
    other_job = OTHER_JOB if train_job != OTHER_JOB else TRAIN_JOB
    for reviewed in [False, True]:
        tx, tp, ty, tids, tq, te, td = personal(other_job, reviewed)
        key = 'second_video_human_assisted' if reviewed else 'second_video_automatic_windows'
        prediction = corrected(model, tx, tp)
        checks[key] = {'base': measure(ty, tp), 'adapted': measure(ty, prediction)}
        target_data[key] = {'ids': tids, 'labels': ty.tolist(), 'guesses': np.asarray(CODES)[prediction.argmax(1)].tolist(),
                            'quality': tq, 'excluded': te, 'revision': td['revision']}
    run_id = uuid.uuid4().hex
    folder = DATA/'击球标注/预训练保留适配'/run_id
    folder.mkdir(parents=True)
    save(folder/'residual_head.npz', model)
    atomic_json(folder/'personal_labels_snapshot.json', doc)
    atomic_json(folder/'second_video_labels_snapshot.json', target_data)
    public_manifest = json.loads((FOLDER/'开发样本/manifest.json').read_text())
    result = {'version': VERSION, 'run_id': run_id, 'created_at': datetime.now(timezone.utc).isoformat(),
              'author_train_samples': int(train.sum()), 'personal_train_samples': len(y),
              'author_validation_samples': int(val.sum()), 'author_test_samples_used': 0,
              'training_job_id': train_job, 'training_dataset_revision': doc['revision'],
              'training_target_source': 'author_dataset_labels_and_human_user_selection_only',
              'pretrained_backbone_and_mlp_frozen': True, 'maximum_teacher_probability_delta': .05,
              'residual_weight': .25, 'alpha': 1., 'classes': CODES, 'checks': checks,
              'personal_input_quality': quality, 'excluded_personal_labels': excluded,
              'personal_class_counts': dict(Counter(y)), 'public_source_sha256': public_manifest['source_archive_sha256'],
              'public_features_sha256': file_hash(FOLDER/'开发样本/features.npz'),
              'pretrained_model_sha256': file_hash(ROOT/'models/bst'/BST_NAME),
              'residual_model_sha256': file_hash(folder/'residual_head.npz'),
              'code_sha256': file_hash(Path(__file__)), 'human_labels_created': 0,
              'active_model_replaced': False,
              'scope': 'Development mixture experiment; second video already used for diagnosis; no independent accuracy claim.',
              'fine_class_policy': 'Mainland fine labels retained manually; no unsupported mapping to coarse author classes.'}
    atomic_json(folder/'manifest.json', result)
    atomic_json(DATA/'击球标注/预训练保留适配/latest.json', result)
    state('completed', '公开样本与个人标签的保留知识适配已完成；校正层仍为开发候选，基础BST继续使用。',
          source_sha256=source['source_sha256'], dataset_revision=doc['revision'], run_id=run_id,
          trained_labels=len(y), public_training_labels=int(train.sum()),
          trained_classes=[LABELS[c] for c in CODES], feature_version=VERSION,
          pretrained_mlp_preserved=True, global_model_activated=False,
          validation_checks=checks, manifest_file=str((folder/'manifest.json').relative_to(DATA)))
    print(json.dumps({'public_train': int(train.sum()), 'personal_train': len(y), 'checks': checks,
                      'folder': str(folder)}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser();parser.add_argument('--job', default=TRAIN_JOB)
    args = parser.parse_args()
    try:
        train(args.job)
    except Exception as error:
        atomic_json(folder_for(args.job)/'adaptation_state.json', {'status': 'failed',
                    'message': '保留知识适配未完成：'+str(error)[:500], 'pretrained_mlp_preserved': True})
        raise
