"""Run the paired development comparison, then retain a full-library BST fit."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

import numpy as np
import torch

from analysis_jobs import DATA, atomic_json, file_hash
from bst_finetune_state import (STORE, POOL, read, training_pool_signature,
                               recorded_training_pool_signature, training_inputs_match,
                               recorded_feature_contract, recorded_feature_identity)
from finetune_personal_bst import train, raw_embedding
from personal_learning import extractor
from stroke_context import metadata
from stroke_review import load_source, read_labels
from stroke_taxonomy import AUTHOR_MAP, LABELS


def progress(message, **extra):
    atomic_json(STORE / 'state.json', {'status': 'running', 'message': message,
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'algorithm': 'direct_bst_mlp', **extra})


def was_trained_contact(source_sha256, human, training_contacts):
    """Do not call a changed label or newly excluded action a trained example."""
    if (human.get('contact_confirmed') is not True or human.get('wrong_target')
            or human.get('play_context') == 'between_rallies'):
        return False
    return (source_sha256, human.get('player'), round(human.get('peak_time', -1), 6),
            human.get('stroke_type')) in training_contacts


def trained_rows(manifest, feature_manifest):
    """Accepted features include held-out rows; only registered train keys fit."""
    accepted = {(r['job_id'], r['event_id']): r for r in feature_manifest['rows']}
    keys = {(r['job_id'], r['event_id']) for r in manifest['trained_sample_keys']}
    if not keys <= accepted.keys():
        raise ValueError('训练参与记录与已接受特征不一致')
    return {key: accepted[key] for key in keys}


def was_trained_input(job, event_id, source_sha256, human, window, quality, rows):
    """An alternate window of a learned contact is not the trained input."""
    row = rows.get((job, event_id))
    if row is None or not was_trained_contact(source_sha256, human, {
            (row['source_sha256'], row['player'], round(row['human_peak_time'], 6),
             row['stroke_type'])}):
        return False
    if any(window.get(key) != row['model_window'].get(key)
           for key in ['start_frame', 'end_frame', 'frame_index', 'player']):
        return False
    return quality.get('player_input_order') == row['input_quality'].get('player_input_order')


def prediction_sidecars(manifest):
    """Unlabelled clips use automatic inputs; reviewed inputs are labelled as fit."""
    if not training_inputs_match(manifest):
        raise ValueError('训练与当前击球输入身份不一致，请按当前输入重新学习')
    checkpoint = DATA / manifest['candidate_checkpoint']
    if file_hash(checkpoint) != manifest['candidate_model_sha256']:
        raise ValueError('直接微调模型校验失败')
    pool = read(POOL)
    jobs = sorted({s['job_id'] for s in pool['samples']})
    feature_manifest = read(DATA / manifest['personal_feature_cache'] / 'manifest.json')
    if feature_manifest['identity'] != recorded_feature_identity(manifest):
        raise ValueError('训练特征缓存与运行记录的输入身份不一致')
    training_rows = trained_rows(manifest, feature_manifest)
    training_contacts = {(r['source_sha256'], r['player'], round(r['human_peak_time'], 6), r['stroke_type'])
                         for r in training_rows.values()}
    state_dict = torch.load(checkpoint, map_location='cpu', weights_only=True)
    count = 0
    for job in jobs:
        if not training_inputs_match(manifest, job=job):
            raise ValueError('生成候选时录像输入已改变：' + job)
        report, events, result_file = load_source(job)
        doc = read_labels(job, report['source_sha256'])
        classifier, dense = extractor(job)
        candidate_head = deepcopy(classifier.model.mlp_head)
        candidate_head.load_state_dict({k[len('mlp_head.'):]: v for k, v in state_dict.items()
                                       if k.startswith('mlp_head.')}, strict=True)
        candidate_head.eval()
        windows = metadata(dense, events, doc)['windows']
        rows, errors = [], []
        for event in events:
            try:
                window = windows[event['id']]
                values, quality = classifier.inputs({**event, **window})
                features, baseline = raw_embedding(classifier, values)
                with torch.inference_mode():
                    scores = candidate_head(torch.from_numpy(features[None])).softmax(-1)[0].numpy()
                ranking = np.argsort(-scores)
                name = classifier.classes[int(ranking[0])]
                prefix, _, raw = name.partition('_')
                if not raw:
                    raw = name
                code = AUTHOR_MAP[raw]
                actor = classifier.order[0 if prefix == 'Top' else 1] if prefix in ['Top', 'Bottom'] else None
                human = doc['labels'].get(event['id'], {})
                contact_trained = was_trained_contact(report['source_sha256'], human, training_contacts)
                trained = was_trained_input(job, event['id'], report['source_sha256'], human,
                                            window, quality, training_rows)
                rows.append({'event_id': event['id'], 'stroke_type': code, 'stroke_label': LABELS[code],
                    'source_model_label': name, 'player_prediction': actor,
                    'candidate_player': event['player'], 'human_reviewed_input': bool(human.get('contact_confirmed')),
                    'used_for_training': trained, 'contact_was_trained': contact_trained,
                    'model_window': window, 'input_quality': quality,
                    'classifier_player_agreement': actor == window['player'],
                    'baseline_source_model_label': classifier.classes[int(baseline.argmax())],
                    'ranking': [{'code': AUTHOR_MAP[classifier.classes[int(i)].split('_', 1)[-1]],
                                 'label': LABELS[AUTHOR_MAP[classifier.classes[int(i)].split('_', 1)[-1]]],
                                 'source_label': classifier.classes[int(i)], 'uncalibrated_score': float(scores[i])}
                                for i in ranking[:5]],
                    'needs_confirmation': True})
            except (ValueError, KeyError, IndexError) as error:
                errors.append({'event_id': event['id'], 'error': str(error)[:400]})
        if not training_inputs_match(manifest, job=job):
            raise ValueError('保存候选前录像输入已改变：' + job)
        atomic_json(STORE / 'predictions' / (job + '.json'), {
            'version': 'direct-bst-candidate-sidecar-v1', 'job_id': job,
            'source_sha256': report['source_sha256'], 'dataset_revision': doc['revision'],
            'candidate_results_sha256': file_hash(result_file), 'run_id': manifest['run_id'],
            'model_sha256': manifest['candidate_model_sha256'], 'events': rows, 'errors': errors,
            'feature_contract': recorded_feature_contract(manifest),
            'feature_identity': recorded_feature_identity(manifest),
            'active_model_replaced': False,
            'scope': 'Candidate preview. Reviewed training contacts are fit, not independent evaluation; unlabelled clips are unverified.'})
        count += len(rows)
        progress('正在保存微调模型的逐拍候选，原始预测保留',
                 run_id=manifest['run_id'], processed_videos=jobs.index(job) + 1,
                 total_videos=len(jobs), candidate_predictions=count)
    return count


def run(job=None, reuse_comparison=False):
    STORE.mkdir(parents=True, exist_ok=True)
    with (STORE / '.workflow.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        comparison = read(STORE / 'latest_comparison.json') if reuse_comparison else None
        if comparison and (recorded_training_pool_signature(comparison) != training_pool_signature(read(POOL))
                           or not training_inputs_match(comparison)):
            comparison = None
        if comparison is None:
            comparison = train(job=job, epochs=40, train_all=False)
        # Fitting all labels is retained as a separate prospective candidate.
        # Its training agreement must never replace the development comparison.
        full = train(job=job, epochs=40, train_all=True)
        torch.set_num_threads(2)
        candidates = prediction_sidecars(full)
        classes = full['training_class_counts']
        result = {'status': 'completed', 'algorithm': 'direct_bst_mlp',
            'message': 'BST自身分类层已用跨录像标注微调；开发对照与全库学习候选分别保留。',
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'run_id': full['run_id'], 'triggering_job_id': job,
            'source_pool_sha256': full['source_pool_sha256'],
            'training_pool_signature': recorded_training_pool_signature(full),
            'feature_contract': recorded_feature_contract(full),
            'feature_identity': recorded_feature_identity(full),
            'trained_labels': full['personal_training_samples'],
            'trained_classes': [LABELS[c] for c in classes],
            'trainable_parameters': full['trainable_parameters'],
            'changed_parameter_names': full['changed_parameter_names'],
            'backbone_frozen': full['backbone_frozen'], 'original_mlp_fine_tuned': True,
            'development_comparison': comparison['checks'],
            'comparison_run_id': comparison['run_id'], 'full_fit_run_id': full['run_id'],
            'full_fit_checks': full['checks'], 'candidate_predictions': candidates,
            'candidate_checkpoint': full['candidate_checkpoint'],
            'candidate_model_sha256': full['candidate_model_sha256'],
            'active_model_replaced': False, 'global_model_activated': False,
            'activation_note': '保留为对照候选；现有开发样本不足以证明稳定提升。',
            'scope': 'Development comparison uses different source videos. Full-library fit has no independent personal score.'}
        atomic_json(STORE / 'state.json', result)
        atomic_json(STORE / 'latest_workflow.json', result)
        print(json.dumps({'status': result['status'], 'comparison_run_id': comparison['run_id'],
                          'full_fit_run_id': full['run_id'], 'trained_labels': result['trained_labels'],
                          'candidate_predictions': candidates}, ensure_ascii=False), flush=True)
        return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--job')
    parser.add_argument('--reuse-comparison', action='store_true')
    args = parser.parse_args()
    try:
        run(args.job, args.reuse_comparison)
    except Exception as error:
        atomic_json(STORE / 'state.json', {'status': 'failed', 'algorithm': 'direct_bst_mlp',
                    'message': 'BST学习未完成：' + str(error)[:500],
                    'updated_at': datetime.now(timezone.utc).isoformat()})
        raise
