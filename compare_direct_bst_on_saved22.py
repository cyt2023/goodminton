"""Paired evaluation of a fixed direct-MLP candidate on saved 22 development rows.

All 22 rows, including the known same-contact duplicate, are retained to match
the previous report's denominator. No weights/configuration are selected here.
"""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analysis_jobs import DATA, ROOT, atomic_json, file_hash
from bst_public_data import public_classifier
from finetune_personal_bst import raw_embedding, metrics, predict, prediction_rows
from stroke_taxonomy import AUTHOR_MAP

VERSION = 'direct-mlp-saved22-paired-v1'
DEFAULT_DIAGNOSTIC = DATA / '击球标注/输入修复诊断/2026-10-12_配对输入修复'


def compare(training_manifest=None, diagnostic=None):
    torch.set_num_threads(2)
    training_manifest = Path(training_manifest or DATA / '击球标注/直接BST微调/latest_comparison.json')
    diagnostic = Path(diagnostic or DEFAULT_DIAGNOSTIC)
    trained = json.loads(training_manifest.read_text())
    if trained['train_all']:
        raise ValueError('全库训练版本包含这批开发数据，不能作跨录像留出对照')
    candidate = DATA / trained['candidate_checkpoint']
    if file_hash(candidate) != trained['candidate_model_sha256']:
        raise ValueError('候选权重SHA不匹配')
    doc = json.loads((diagnostic / 'result.json').read_text())
    classifier = public_classifier()
    baseline_weights = classifier.model.state_dict()
    candidate_weights = torch.load(candidate, map_location='cpu', weights_only=True)
    changed = [k for k in baseline_weights if not torch.equal(baseline_weights[k], candidate_weights[k])]
    if not changed or any(not k.startswith('mlp_head.') for k in changed):
        raise ValueError('对照候选不是相同冻结骨干上的分类层微调')
    sources = trained['personal_features']['sources']
    train_sources = set(trained['personal_training_source_sha256'])
    with np.load(diagnostic / 'inputs.npz', allow_pickle=False) as d:
        keys = {tuple(k): i for i, k in enumerate(d['keys'].tolist())}
        xx, baseline, targets, rows = [], [], [], []
        errors = []
        for old in doc['rows']:
            job, eid = old['job_id'], old['event_id']
            source = sources[job]['source_sha256']
            if source in train_sources:
                raise ValueError('这批22条中存在训练源录像')
            i = keys[(job, eid, 'human_center_same_duration')]
            values = (d['joints_and_bones'][i], d['positions'][i], d['shuttle'][i], int(d['video_len'][i]))
            x, logits = raw_embedding(classifier, values)
            expected = np.asarray(old['variants']['human_center_same_duration']['scores'])
            p = torch.as_tensor(logits).softmax(-1).numpy()
            error = float(abs(p - expected).max())
            if error > 1e-5:
                raise ValueError('已保存输入回放分数不匹配：' + eid + '/' + str(error))
            errors.append(error)
            human = old['human']
            order = old['variants']['human_center_same_duration']['player_order']
            side = 'Top' if order.index(human['player']) == 0 else 'Bottom'
            target = next(k for k, c in enumerate(classifier.classes)
                          if c.startswith(side + '_') and AUTHOR_MAP[c.split('_', 1)[-1]] == human['stroke_type'])
            xx.append(x)
            baseline.append(logits)
            targets.append(target)
            rows.append({'job_id': job, 'event_id': eid, 'source_sha256': source,
                         'stroke_type': human['stroke_type'], 'player': human['player'],
                         'human_peak_time': human['peak_time'], 'target_class_label': classifier.classes[target],
                         'model_window': old['human_shifted_window'],
                         'input_quality': {'player_input_order': order},
                         'frozen_original_prediction': old['variants']['frozen_replay']})
    xx, baseline, y = np.stack(xx), np.stack(baseline), np.asarray(targets, dtype='int64')
    classifier.model.load_state_dict(candidate_weights, strict=True)
    adapted = predict(classifier.model.mlp_head, xx)
    b_rows = prediction_rows(baseline, rows, classifier.classes)
    c_rows = prediction_rows(adapted, rows, classifier.classes)
    six_specs = [('J_e23c', 61.766667), ('J_16b4', 6.583333), ('J_16b4', 10.041667),
                 ('J_16b4', 15.125), ('J_fbda', 5.766667), ('J_fbda', 13.366667)]
    six = []
    for prefix, seconds in six_specs:
        matched = [(b, c) for b, c in zip(b_rows, c_rows)
                   if b['job_id'].startswith(prefix) and abs(b['human_peak_time'] - seconds) < .01
                   and b['frozen_original_prediction']['mapped_actor'] == b['player']]
        if len(matched) != 1:
            raise ValueError('六条中心准且人物一致的原错例匹配失败：' + prefix + '/' + str(seconds))
        b, c = matched[0]
        six.append({'job_id': b['job_id'], 'event_id': b['event_id'], 'human_peak_time': b['human_peak_time'],
                    'human_stroke_type': b['stroke_type'], 'frozen_original': b['frozen_original_prediction'],
                    'translated_baseline': {'label': b['source_top_label'], 'stroke_correct': b['stroke_correct'],
                                             'player_agrees': b['player_agrees_with_saved_mapping']},
                    'translated_candidate': {'label': c['source_top_label'], 'stroke_correct': c['stroke_correct'],
                                              'player_agrees': c['player_agrees_with_saved_mapping']}})
    result = {'version': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
              'training_run_id': trained['run_id'], 'training_manifest_sha256': file_hash(training_manifest),
              'candidate_checkpoint': trained['candidate_checkpoint'],
              'candidate_model_sha256': file_hash(candidate), 'original_model_sha256': doc['model_sha256'],
              'diagnostic_result_sha256': file_hash(diagnostic / 'result.json'),
              'diagnostic_inputs_sha256': file_hash(diagnostic / 'inputs.npz'),
              'code_sha256': file_hash(Path(__file__)), 'count': len(rows),
              'all_source_videos_excluded_from_this_training': True,
              'training_source_sha256': sorted(train_sources),
              'evaluation_source_sha256': sorted(set(r['source_sha256'] for r in rows)),
              'scope': 'Same 22 previously inspected development rows; includes one same-contact duplicate, not 22 independent events or a fresh test.',
              'window_scope': 'Identical saved human-center translated original complete support for baseline and candidate.',
              'player_scope': 'Actor and joint agreement are conditional on unconfirmed saved input order.',
              'summary': {'frozen_original': doc['summary']['frozen_replay'],
                          'translated_baseline': metrics(baseline, y, classifier.classes),
                          'translated_candidate': metrics(adapted, y, classifier.classes)},
              'six_original_aligned_actor_consistent_errors': six,
              'baseline': b_rows, 'candidate': c_rows,
              'baseline_replay_max_score_error': max(errors),
              'weights_changed_by_this_comparison': False, 'active_model_replaced': False}
    out = DATA / '击球标注/直接BST微调/运行记录' / trained['run_id'] / '同22条输入修复与微调配对.json'
    atomic_json(out, result)
    np.savez_compressed(out.with_suffix('.npz'), baseline_logits=baseline, candidate_logits=adapted, labels=y,
                        features=xx, event_ids=np.asarray([r['event_id'] for r in rows]))
    print(json.dumps({'file': str(out), 'summary': result['summary'],
                      'six_cases': six}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--training-manifest')
    p.add_argument('--diagnostic')
    args = p.parse_args()
    compare(args.training_manifest, args.diagnostic)
