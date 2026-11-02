"""Directly fine-tune the released BST MLP with human labels and public replay.

The backbone is frozen, but the author's original classifier is trainable. Raw
300D features retain the author's Top/Bottom order. The default comparison
holds out whole source videos; those videos are development data, not a fresh
test set. --train-all creates a separate fit with no personal holdout metric.
Original checkpoints, labels and frozen predictions are never overwritten.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from analysis_jobs import DATA, atomic_json, file_hash
from bst_public_data import FOLDER, public_classifier
from personal_learning import extractor
from specialist_model import ROOT, BST_NAME
from stroke_context import (window as automatic_window, guard_confirmed_single_hit,
                            VERSION as WINDOW_VERSION)
from stroke_window_repairs import translate_contact_window, VERSION as WINDOW_REPAIR_VERSION
from stroke_review import folder_for, load_source, read_labels, usable_training_label
from stroke_taxonomy import AUTHOR_MAP, LABELS, VERSION as TAXONOMY_VERSION

VERSION = 'direct-personal-bst-mlp-v1'
DEFAULT_TRAIN_JOB = 'J_fe1e058fc75e475d9c030caced6ec648'
OUTPUT = DATA / '击球标注/直接BST微调'
POOL = DATA / '训练库/训练样本.json'
PUBLIC_CACHE = FOLDER / '本地继续训练/冻结骨干特征'
SEED = 20261012
BASIC_CODES = [c for c in dict.fromkeys(AUTHOR_MAP.values()) if c != 'unknown']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def state(status, message, **extra):
    atomic_json(OUTPUT / 'state.json', {'version': VERSION, 'status': status,
                'message': message, 'updated_at': datetime.now(timezone.utc).isoformat(), **extra})


def raw_embedding(classifier, values):
    captured = []
    hook = classifier.model.mlp_head.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().cpu().numpy().copy()))
    j, _, s, length = values
    try:
        with torch.inference_mode():
            logits = classifier.model(torch.from_numpy(j[None]), torch.from_numpy(s[None]),
                                      torch.tensor([length])).cpu().numpy()[0]
    finally:
        hook.remove()
    x = captured[0][0].astype('float32')
    if x.shape != (300,) or not np.isfinite(x).all() or not np.isfinite(logits).all():
        raise ValueError('BST原始特征或分数异常')
    return x, logits.astype('float32')


def personal_features():
    """Read the pooled labels; deduplicate source + side + human contact frame."""
    pool = json.loads(POOL.read_text())
    rows = pool['samples']
    jobs = sorted(set(r['job_id'] for r in rows))
    snapshots, sources = {}, {}
    for job in jobs:
        report, events, result_path = load_source(job)
        doc = read_labels(job, report['source_sha256'])
        snapshots[job] = doc
        sources[job] = {
            'source_sha256': report['source_sha256'], 'dataset_revision': doc['revision'],
            'annotations_sha256': file_hash(folder_for(job) / 'annotations.json'),
            'candidate_results_sha256': file_hash(result_path),
            'dense_pose_sha256': file_hash(DATA / '专项分析' / job / 'dense_pose.json'),
            'trajectory_sha256': file_hash(DATA / '专项分析' / job / 'trajectory.json'),
            'hit_refinement_data_sha256': (file_hash(DATA / '专项分析' / job / 'hit_refinement.json')
                                             if (DATA / '专项分析' / job / 'hit_refinement.json').is_file() else None),
            'frozen_public_predictions_sha256': (file_hash(folder_for(job) / 'public_only_predictions.json')
                                                 if (folder_for(job) / 'public_only_predictions.json').is_file() else None),
        }
    identity = {'version': VERSION, 'pool_sha256': file_hash(POOL), 'sources': sources,
                'model_sha256': file_hash(ROOT / 'models/bst' / BST_NAME),
                'feature_code_sha256': file_hash(Path(__file__)),
                'adapter_sha256': file_hash(ROOT / 'specialist_bst.py'),
                'window_code_sha256': file_hash(ROOT / 'stroke_context.py'),
                'window_repair_code_sha256': file_hash(ROOT / 'stroke_window_repairs.py'),
                'hit_refinement_code_sha256': file_hash(ROOT / 'hit_refinement.py'),
                'window_version': WINDOW_VERSION, 'window_repair_version': WINDOW_REPAIR_VERSION,
                'taxonomy_version': TAXONOMY_VERSION,
                'feature_order': 'raw author Top, Bottom, shuttle; no target-first swap'}
    cache = OUTPUT / '冻结骨干特征' / digest(identity)
    if (cache / 'manifest.json').is_file():
        manifest = json.loads((cache / 'manifest.json').read_text())
        if manifest['identity'] != identity or file_hash(cache / 'features.npz') != manifest['features_sha256']:
            raise ValueError('人工特征缓存身份或SHA不匹配')
        with np.load(cache / 'features.npz', allow_pickle=False) as d:
            return d['features'].copy(), d['baseline_logits'].copy(), d['labels'].copy(), manifest, snapshots

    # Construct the complete set before deduplication so conflicting labels
    # exclude the whole physical-contact group rather than first-row winning.
    eligible, excluded = [], []
    per_job = defaultdict(list)
    for row in rows:
        per_job[row['job_id']].append(row)
    classes = public_classifier().classes
    for job in jobs:
        report, events, _ = load_source(job)
        doc = snapshots[job]
        classifier, dense = extractor(job)
        if classifier.classes != classes or dense['source_sha256'] != report['source_sha256']:
            raise ValueError('个人特征模型类别或原视频身份不匹配')
        frozen_path = folder_for(job) / 'public_only_predictions.json'
        frozen = json.loads(frozen_path.read_text()) if frozen_path.is_file() else None
        if frozen and frozen['source_sha256'] != report['source_sha256']:
            raise ValueError('冻结公开检查原视频身份不匹配')
        frozen_index = {e['event_id']: e for e in frozen['events']} if frozen else {}
        event_index = {e['id']: e for e in events}
        times = np.asarray([f['source_time'] for f in dense['frames']], dtype=float)
        for pooled in per_job[job]:
            eid = pooled['event_id']
            label = doc['labels'].get(eid)
            reason = None
            if not label:
                reason = 'missing_current_human_annotation'
            elif any(pooled.get(k) != label.get(k) for k in
                     ['stroke_type', 'player', 'contact_confirmed', 'peak_time', 'play_context']):
                raise ValueError('汇总训练库与当前标注不一致，请先刷新训练库：' + job + '/' + eid)
            elif pooled.get('training_role') != 'stroke_positive':
                reason = 'retained_separately_' + pooled.get('training_role', 'unknown_role')
            elif label['stroke_type'] not in BASIC_CODES:
                reason = 'retained_fine_class_not_in_original_25_class_output'
            elif not usable_training_label(label):
                reason = 'human_label_not_eligible_for_stroke_classification'
            elif eid not in event_index:
                reason = 'missing_candidate_event'
            if reason:
                excluded.append({'job_id': job, 'event_id': eid,
                                 'stroke_type': pooled['stroke_type'], 'reason': reason})
                continue
            center = int(np.argmin(abs(times - label['peak_time'])))
            # source_time remains on the original presentation timeline even
            # for subwindows. Exact time identifies the original decoded frame.
            frame_time = round(float(times[center]), 6)
            group = digest([pooled['source_sha256'], label['player'], frame_time])
            old = frozen_index.get(eid)
            original_window = (old['model_window'] if old else
                               automatic_window(dense, event_index[eid], events, {}, reviewed=False))
            if old:
                classifier.order = list(old['input_quality']['player_input_order'])
            translated = {**translate_contact_window(dense, original_window, label['peak_time']),
                          'player': label['player'], 'contact_confirmed': True}
            w = guard_confirmed_single_hit(dense, translated, label['player'],
                                           doc['labels'], target_event_id=eid)
            if abs(w['peak_time'] - label['peak_time']) > max(.06, 1.5 / dense['sample_fps']):
                raise ValueError('人工触球中心没有进入模型窗口：' + eid)
            v, quality = classifier.inputs({**event_index[eid], **w})
            if quality['paired_player_fraction'] < .5:
                excluded.append({'job_id': job, 'event_id': eid, 'stroke_type': label['stroke_type'],
                                 'reason': 'paired_pose_coverage_below_50_percent', 'quality': quality})
                continue
            prefix = 'Top' if classifier.order.index(label['player']) == 0 else 'Bottom'
            targets = [i for i, c in enumerate(classes)
                       if c.startswith(prefix + '_') and AUTHOR_MAP.get(c.split('_', 1)[-1]) == label['stroke_type']]
            if len(targets) != 1:
                raise ValueError('人工球种不能唯一映射到BST原始输出')
            x, logits = raw_embedding(classifier, v)
            eligible.append({'x': x, 'logits': logits, 'target': targets[0], 'row': {
                'job_id': job, 'event_id': eid, 'source_sha256': pooled['source_sha256'],
                'dataset_revision': doc['revision'], 'stroke_type': label['stroke_type'],
                'stroke_label': LABELS[label['stroke_type']], 'player': label['player'],
                'human_peak_time': label['peak_time'], 'decoded_contact_time': frame_time,
                'local_contact_frame': center, 'contact_group_id': group,
                'target_class_id': targets[0], 'target_class_label': classes[targets[0]],
                'model_window': w, 'original_automatic_model_window': original_window, 'input_quality': quality,
                'player_order_independently_confirmed': quality['player_order_basis'] == 'reviewed_court_y'}})
        print('PERSONAL_FEATURES', job, len(eligible), 'accepted-so-far', flush=True)
    groups = defaultdict(list)
    for item in eligible:
        groups[item['row']['contact_group_id']].append(item)
    accepted = []
    for group in groups.values():
        if len(set(item['target'] for item in group)) != 1:
            for item in group:
                excluded.append({**item['row'], 'reason': 'conflicting_human_labels_for_same_physical_contact'})
            continue
        group.sort(key=lambda item: (item['row']['job_id'], item['row']['event_id']))
        accepted.append(group[0])
        for item in group[1:]:
            excluded.append({**item['row'], 'reason': 'duplicate_source_side_human_contact_frame',
                             'kept_event_id': group[0]['row']['event_id']})
    accepted.sort(key=lambda item: (item['row']['source_sha256'], item['row']['decoded_contact_time'],
                                    item['row']['player']))
    if not accepted:
        raise ValueError('没有可直接微调BST的人工基础球种')
    cache.mkdir(parents=True, exist_ok=True)
    x = np.stack([item['x'] for item in accepted])
    baseline = np.stack([item['logits'] for item in accepted])
    y = np.asarray([item['target'] for item in accepted], dtype='int64')
    np.savez_compressed(cache / 'features.npz', features=x, baseline_logits=baseline, labels=y)
    manifest = {'identity': identity, 'rows': [item['row'] for item in accepted],
                'excluded': excluded, 'pool_total': len(rows), 'accepted_total': len(accepted),
                'pool_roles': dict(Counter(r['training_role'] for r in rows)),
                'class_labels': classes, 'features_sha256': file_hash(cache / 'features.npz'),
                'cache_path': str(cache.relative_to(DATA))}
    atomic_json(cache / 'manifest.json', manifest)
    return x, baseline, y, manifest, snapshots


def public_features(classes):
    """Read only author train/val frozen features. Never open public test arrays."""
    progress = json.loads((PUBLIC_CACHE / 'progress.json').read_text())
    identity = progress['identity']
    if (progress['completed'] != identity['count'] or identity['classes'] != classes or
            identity['model_sha256'] != file_hash(ROOT / 'models/bst' / BST_NAME)):
        raise ValueError('完整公开特征缓存未完成或模型/类别身份不匹配')
    rows = json.loads((PUBLIC_CACHE / 'rows.json').read_text())['rows']
    original_rows_sha = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    if original_rows_sha != identity['row_ids_sha256']:
        raise ValueError('公开缓存行索引SHA不匹配')
    x = np.load(PUBLIC_CACHE / 'features.npy', allow_pickle=False)
    teacher = np.load(PUBLIC_CACHE / 'teacher_logits.npy', allow_pickle=False)
    y = np.load(PUBLIC_CACHE / 'labels.npy', allow_pickle=False)
    splits = np.load(PUBLIC_CACHE / 'splits.npy', allow_pickle=False)
    if (x.shape != (len(rows), 300) or teacher.shape != (len(rows), 25) or len(y) != len(rows) or
            set(splits) != {'train', 'val'} or not np.isfinite(x).all() or not np.isfinite(teacher).all()):
        raise ValueError('公开特征缓存形状、分割或数值异常')
    provenance = {'identity': identity, 'files_sha256': {
        name: file_hash(PUBLIC_CACHE / name) for name in
        ['features.npy', 'teacher_logits.npy', 'labels.npy', 'splits.npy', 'rows.json', 'progress.json']}}
    return x, teacher, y, splits, provenance


def predict(head, x, batch_size=1024):
    head.eval()
    if len(x) == 0:
        return np.empty((0, 25), dtype='float32')
    with torch.inference_mode():
        return np.concatenate([head(torch.as_tensor(x[i:i + batch_size])).cpu().numpy()
                               for i in range(0, len(x), batch_size)])


def metrics(logits, y, classes):
    if len(y) == 0:
        return None
    guessed = logits.argmax(1)
    codes = [AUTHOR_MAP[c.split('_', 1)[-1]] for c in classes]
    actual_codes = np.asarray([codes[int(i)] for i in y])
    guessed_codes = np.asarray([codes[int(i)] for i in guessed])
    sides = [c.split('_', 1)[0] if '_' in c else 'unknown' for c in classes]
    player_ok = np.asarray([sides[int(a)] == sides[int(b)] for a, b in zip(guessed, y)])
    kind_ok = actual_codes == guessed_codes
    conditional = []
    for row, target in zip(logits, y):
        allowed = [i for i, side in enumerate(sides) if side == sides[int(target)]]
        conditional.append(codes[max(allowed, key=lambda i: float(row[i]))])
    per_class = {}
    for c in sorted(set(actual_codes)):
        actual = actual_codes == c
        selected = guessed_codes == c
        tp = int((actual & selected).sum())
        per_class[c] = {'label': LABELS[c], 'total': int(actual.sum()), 'correct': tp,
                        'precision': tp / int(selected.sum()) if selected.any() else 0.,
                        'recall': tp / int(actual.sum()),
                        'f1': 2 * tp / int(actual.sum() + selected.sum())}
    return {'total': len(y), 'stroke_top1_correct': int(kind_ok.sum()),
            'stroke_top1_accuracy': float(kind_ok.mean()), 'player_top1_correct': int(player_ok.sum()),
            'player_top1_accuracy': float(player_ok.mean()),
            'joint_25class_correct': int((guessed == y).sum()),
            'joint_25class_accuracy': float((guessed == y).mean()),
            'human_target_conditioned_stroke_correct': int((np.asarray(conditional) == actual_codes).sum()),
            'human_target_conditioned_stroke_accuracy': float((np.asarray(conditional) == actual_codes).mean()),
            'macro_f1_over_actual_stroke_classes': float(np.mean([v['f1'] for v in per_class.values()])),
            'cross_entropy_25class': float(F.cross_entropy(torch.as_tensor(logits), torch.as_tensor(y))),
            'per_stroke_class': per_class,
            'player_metric_scope': 'conditional on saved model input order; not independent player-identity ground truth'}


def prediction_rows(logits, rows, classes):
    probabilities = torch.as_tensor(logits).softmax(-1).numpy()
    output = []
    for row, scores in zip(rows, probabilities):
        order = np.argsort(-scores)
        best = classes[int(order[0])]
        prefix, raw = best.split('_', 1) if '_' in best else ('unknown', best)
        source_order = row['input_quality']['player_input_order']
        predicted_player = source_order[0 if prefix == 'Top' else 1] if prefix in ['Top', 'Bottom'] else None
        code = AUTHOR_MAP[raw]
        output.append({**row, 'source_top_label': best, 'stroke_type_prediction': code,
                       'stroke_label_prediction': LABELS[code], 'player_prediction': predicted_player,
                       'stroke_correct': code == row['stroke_type'],
                       'player_agrees_with_saved_mapping': predicted_player == row['player'],
                       'joint_correct_with_saved_mapping': best == row['target_class_label'],
                       'top5': [{'label': classes[int(i)], 'softmax_score': float(scores[i])} for i in order[:5]],
                       'score_note': 'uncalibrated softmax; not a probability of correctness'})
    return output


def train(job=None, epochs=40, train_all=False):
    if epochs < 1 or epochs > 200:
        raise ValueError('微调轮数需要在1–200之间')
    if job:
        folder_for(job)  # Validate a page-triggered job identifier.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / '.train.lock').open('a+b')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.perf_counter()
    run_id = uuid.uuid4().hex
    run = OUTPUT / '运行记录' / run_id
    run.mkdir(parents=True)
    state('extracting', '正在按人工触球重建完整窗口，准备直接微调BST分类层', run_id=run_id,
          triggering_job_id=job, train_all=train_all)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    shutil.copy2(ROOT / 'models/bst' / BST_NAME, run / 'baseline.pt')
    for name in ['finetune_personal_bst.py', 'specialist_bst.py', 'stroke_context.py', 'hit_refinement.py',
                 'stroke_window_repairs.py']:
        shutil.copy2(ROOT / name, run / name)
    shutil.copy2(POOL, run / 'training_pool_snapshot.json')
    x, baseline, y, personal_manifest, snapshots = personal_features()
    from benchmarks.isolation import assert_training_allowed
    assert_training_allowed(personal_manifest)
    classes = personal_manifest['class_labels']
    rows = personal_manifest['rows']
    atomic_json(run / 'personal_features_manifest.json', personal_manifest)
    atomic_json(run / 'human_annotations_snapshot.json', snapshots)
    public_x, public_teacher, public_y, splits, public_identity = public_features(classes)
    tr = splits == 'train'
    va = splits == 'val'
    if train_all:
        train_mask = np.ones(len(rows), dtype=bool)
        holdout_mask = np.zeros(len(rows), dtype=bool)
    else:
        first_sha = snapshots[DEFAULT_TRAIN_JOB]['source_sha256']
        train_mask = np.asarray([r['source_sha256'] == first_sha for r in rows])
        holdout_mask = ~train_mask
        if not train_mask.any() or not holdout_mask.any():
            raise ValueError('源录像分组没有得到非空训练与开发留出')
    train_sources = set(r['source_sha256'] for i, r in enumerate(rows) if train_mask[i])
    held_sources = set(r['source_sha256'] for i, r in enumerate(rows) if holdout_mask[i])
    if train_sources & held_sources:
        raise ValueError('源录像泄漏：训练和开发留出有交集')
    model = public_classifier().model
    original = deepcopy(model.state_dict())
    for p in model.parameters():
        p.requires_grad_(False)
    head = model.mlp_head
    for p in head.parameters():
        p.requires_grad_(True)
    parameter_count = sum(p.numel() for p in head.parameters())
    anchor = [p.detach().clone() for p in head.parameters()]
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-4)
    px = torch.as_tensor(x[train_mask])
    py = torch.as_tensor(y[train_mask])
    sx = torch.as_tensor(public_x[tr])
    sy = torch.as_tensor(public_y[tr])
    teacher = torch.as_tensor(public_teacher[tr])
    # Modest class weighting avoids letting the abundant lifts suppress rare
    # categories while retaining the original 25-class output and parameters.
    counts = Counter(y[train_mask].tolist())
    weights = torch.ones(25)
    for c, count in counts.items():
        weights[c] = min(3., np.sqrt(len(py) / (len(counts) * count)))
    generator = torch.Generator().manual_seed(SEED)
    history = []
    replay_counts = np.zeros(len(sx), dtype='int32')
    base_public = metrics(public_teacher[va], public_y[va], classes)
    base_train = metrics(baseline[train_mask], y[train_mask], classes)
    base_holdout = metrics(baseline[holdout_mask], y[holdout_mask], classes)
    # Feature-cache misses instantiate extra models; reset dropout RNG after
    # all feature/model construction so cold and warm caches share its stream.
    torch.manual_seed(SEED)
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(px), generator=generator)
        losses = []
        for offset in range(0, len(order), 32):
            ids = order[offset:offset + 32]
            source_ids = torch.randint(len(sx), (256,), generator=generator)
            np.add.at(replay_counts, source_ids.numpy(), 1)
            logits = head(px[ids])
            replay = head(sx[source_ids])
            ce = F.cross_entropy(logits, py[ids], weight=weights)
            source_ce = F.cross_entropy(replay, sy[source_ids])
            temperature = 2.
            kd = F.kl_div(F.log_softmax(replay / temperature, dim=1),
                          F.softmax(teacher[source_ids] / temperature, dim=1),
                          reduction='batchmean') * temperature ** 2
            distance = sum((p - a).square().mean() for p, a in zip(head.parameters(), anchor))
            loss = ce + .75 * source_ce + .25 * kd + .001 * distance
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        train_metrics = metrics(predict(head, x[train_mask]), y[train_mask], classes)
        public_metrics = metrics(predict(head, public_x[va]), public_y[va], classes)
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)),
                        'personal_training_fit': train_metrics, 'author_validation_monitor': public_metrics})
        atomic_json(run / 'epochs.json', history)
        state('training', '正在更新BST原始分类层，保留公开数据回放；开发留出不用于选择版本',
              run_id=run_id, epoch=epoch, epochs=epochs, train_all=train_all,
              personal_train_samples=int(train_mask.sum()), personal_development_holdout_samples=int(holdout_mask.sum()),
              public_training_pool_samples=int(tr.sum()), trainable_parameters=parameter_count)
        print('DIRECT_BST_EPOCH', epoch, 'personal-fit', train_metrics['stroke_top1_correct'], '/', len(py),
              'public-joint', public_metrics['joint_25class_correct'], '/', int(va.sum()), flush=True)
    head.eval()
    final = model.state_dict()
    changed_names = [name for name, value in original.items() if not torch.equal(value, final[name])]
    if not changed_names or any(not name.startswith('mlp_head.') for name in changed_names):
        raise ValueError('BST分类层未更新或冻结骨干意外改变')
    torch.save(final, run / 'candidate.pt')
    candidate = predict(head, x)
    final_public = predict(head, public_x[va])
    final_train = metrics(candidate[train_mask], y[train_mask], classes)
    final_holdout = metrics(candidate[holdout_mask], y[holdout_mask], classes)
    candidate_public = metrics(final_public, public_y[va], classes)
    np.savez_compressed(run / 'personal_predictions.npz', baseline_logits=baseline, candidate_logits=candidate,
                        labels=y, training_mask=train_mask, development_holdout_mask=holdout_mask)
    np.savez_compressed(run / 'author_validation_predictions.npz', baseline_logits=public_teacher[va],
                        candidate_logits=final_public, labels=public_y[va])
    output_rows = {'baseline': prediction_rows(baseline, rows, classes),
                   'candidate': prediction_rows(candidate, rows, classes),
                   'training_contact_group_ids': [r['contact_group_id'] for i, r in enumerate(rows) if train_mask[i]],
                   'development_holdout_contact_group_ids': [r['contact_group_id'] for i, r in enumerate(rows) if holdout_mask[i]]}
    atomic_json(run / 'personal_predictions.json', output_rows)
    by_source = {}
    for source in sorted(set(r['source_sha256'] for r in rows)):
        mask = np.asarray([r['source_sha256'] == source for r in rows])
        by_source[source] = {'role': 'training_fit' if source in train_sources else 'development_holdout',
                             'job_ids': sorted(set(r['job_id'] for i, r in enumerate(rows) if mask[i])),
                             'baseline': metrics(baseline[mask], y[mask], classes),
                             'candidate': metrics(candidate[mask], y[mask], classes)}
    # Preserve actual replay IDs; "pool available" is not "all examples seen".
    public_rows = json.loads((PUBLIC_CACHE / 'rows.json').read_text())['rows']
    source_rows = [r for r in public_rows if r['split'] == 'train']
    atomic_json(run / 'public_replay_samples.json', {'samples': [
        {**source_rows[i], 'times_sampled': int(replay_counts[i])}
        for i in np.flatnonzero(replay_counts)], 'seed': SEED})
    result = {
        'version': VERSION, 'run_id': run_id, 'created_at': datetime.now(timezone.utc).isoformat(),
        'triggering_job_id': job, 'trigger_job': job, 'train_all': train_all,
        'epochs': epochs, 'selected_epoch': epochs,
        'selection_policy': 'fixed final epoch; no personal development holdout selection',
        'personal_training_samples': int(train_mask.sum()),
        'personal_development_holdout_samples': int(holdout_mask.sum()),
        'personal_training_source_sha256': sorted(train_sources),
        'personal_development_source_sha256': sorted(held_sources),
        'split_rule': 'entire original source SHA held out; source+side+decoded human contact frame deduplicated',
        'personal_features': personal_manifest['identity'],
        'source_pool_sha256': personal_manifest['identity']['pool_sha256'],
        'personal_feature_cache': personal_manifest['cache_path'],
        'excluded_personal_labels': personal_manifest['excluded'],
        'training_class_counts': dict(Counter(rows[i]['stroke_type'] for i in np.flatnonzero(train_mask))),
        'class_counts': dict(Counter(rows[i]['stroke_type'] for i in np.flatnonzero(train_mask))),
        'trained_sample_keys': [{'job_id': rows[i]['job_id'], 'event_id': rows[i]['event_id'],
                                 'contact_group_id': rows[i]['contact_group_id']}
                                for i in np.flatnonzero(train_mask)],
        'public_features': public_identity,
        'public_training_pool_samples': int(tr.sum()),
        'public_training_unique_samples_seen': int((replay_counts > 0).sum()),
        'public_training_draws': int(replay_counts.sum()),
        'public_validation_samples': int(va.sum()), 'public_test_samples_used': 0,
        'backbone_frozen': True, 'original_mlp_fine_tuned': True,
        'trainable_parameters': parameter_count, 'changed_parameter_names': changed_names,
        'original_output_classes': classes, 'bounded_probability_residual': False,
        'optimizer': 'AdamW', 'learning_rate': 1e-4, 'seed': SEED,
        'rng_policy': 'reset Torch RNG immediately before optimisation, after feature/model construction',
        'loss': 'weighted personal CE + 0.75 public CE + 0.25 public teacher KL(T=2) + 0.001 head parameter anchor',
        'class_weighting': 'sqrt inverse frequency for observed personal joint labels; maximum 3',
        'checks': {'personal_training_fit': {'baseline': base_train, 'candidate': final_train},
                   'personal_development_holdout': {'baseline': base_holdout, 'candidate': final_holdout},
                   'author_validation': {'baseline': base_public, 'candidate': candidate_public},
                   'by_source_video': by_source},
        'initial_model_sha256': file_hash(run / 'baseline.pt'),
        'candidate_model_sha256': file_hash(run / 'candidate.pt'),
        'baseline_checkpoint': str((run / 'baseline.pt').relative_to(DATA)),
        'candidate_checkpoint': str((run / 'candidate.pt').relative_to(DATA)),
        'code_sha256': file_hash(Path(__file__)),
        'runtime_seconds': round(time.perf_counter() - started, 3),
        'active_model_replaced': False, 'global_model_activated': False,
        'human_labels_created': 0,
        'scope': ('All available compatible human labels fitted; no personal holdout or generalisation score.' if train_all else
                  'Cross-video development comparison on previously inspected videos; not a fresh independent test.'),
        'fine_class_policy': 'Retain mainland fine labels separately; no forced merge into author classes.',
        'negative_policy': 'No-stroke negatives retained separately; not treated as a stroke class.',
        'input_scope': 'Human-assisted contact and proposed support window; does not measure automatic hit detection recall.',
        'player_scope': 'Joint/person scores depend on unconfirmed automatic Top/Bottom input order.',
    }
    atomic_json(run / 'manifest.json', result)
    atomic_json(OUTPUT / ('latest_all_labels.json' if train_all else 'latest_comparison.json'), result)
    atomic_json(OUTPUT / 'latest.json', result)
    state('completed', 'BST原始分类层微调完成，候选权重与逐条对照已保留',
          run_id=run_id, train_all=train_all, personal_train_samples=int(train_mask.sum()),
          personal_development_holdout_samples=int(holdout_mask.sum()), active_model_replaced=False,
          manifest_file=str((run / 'manifest.json').relative_to(DATA)), checks=result['checks'])
    print(json.dumps({'run_id': run_id, 'manifest': str(run / 'manifest.json'),
                      'train_all': train_all, 'checks': result['checks']}, ensure_ascii=False), flush=True)
    lock.close()
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--job')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--train-all', action='store_true')
    args = parser.parse_args()
    try:
        train(job=args.job, epochs=args.epochs, train_all=args.train_all)
    except Exception as error:
        state('failed', 'BST直接微调未完成：' + str(error)[:500])
        raise
