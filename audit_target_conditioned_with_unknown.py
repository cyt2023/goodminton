"""Saved-logit sensitivity: retain abstention when restricting the target actor."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from analysis_jobs import atomic_json, file_hash
from audit_target_conditioned_scores import OUT, RUN, SOURCE
from specialist_bst import author_helpers


def restricted(scores, actor, classes):
    allowed = [0] + [i for i, name in enumerate(classes) if name.startswith(actor + '_')]
    assert len(allowed) == 13 and len(set(allowed)) == 13
    return int(allowed[int(np.argmax(scores[allowed]))])


def paired(a, b, key):
    return {
        'gains': [r['event_id'] for r, s in zip(a, b) if not r[key] and s[key]],
        'losses': [r['event_id'] for r, s in zip(a, b) if r[key] and not s[key]],
    }


def main():
    target = OUT / 'result_with_unknown.json'
    if target.exists():
        raise FileExistsError('Preserve completed sensitivity audit')
    pose_dir = OUT.parent / '2026-10-18_普通视角配对诊断/pose_missing_1p0333'
    paths = [OUT/'protocol.json', OUT/'result.json', RUN/'personal_predictions.json',
             RUN/'personal_predictions.npz', SOURCE/'automatic_events/classification_logits.npz',
             pose_dir/'classification_result.json', pose_dir/'classification_logits.npz', Path(__file__)]
    hashes = {str(p): file_hash(p) for p in paths}
    protocol = {
        'declared_at': datetime.now(timezone.utc).isoformat(), 'input_sha256': hashes,
        'reason': 'Independent review found that the original 12-class actor restriction also excluded unknown. Preserve that result as a known-valid-stroke condition and add abstention sensitivity.',
        'method': 'For each unchanged saved 25-logit vector, argmax over target actor 12 classes plus original unknown index0. Actor mapping, development mask and true labels copied from frozen prior audit. No probability renormalization, thresholds, inference, training or activation.',
        'scope': 'Repeated developer records with reviewed windows; human actor and original automatic actor are distinct conditional tasks. Not prospective end-to-end accuracy.',
        'original_12class_outputs_preserved': True,
    }
    atomic_json(OUT/'protocol_with_unknown.json', protocol)
    classes = author_helpers()['get_merged_stroke_types']()
    assert len(classes) == 25 and classes[0] == '未知球種'
    old = json.loads((OUT/'result.json').read_text())
    cache = np.load(RUN/'personal_predictions.npz')
    records = json.loads((RUN/'personal_predictions.json').read_text())['baseline']
    index = {(r['job_id'], r['event_id']): i for i, r in enumerate(records)}
    ids = set(np.flatnonzero(cache['development_holdout_mask']).tolist())
    assert len(ids) == 30 and int(cache['training_mask'].sum()) == 41
    ordinary = {}
    for method, key in [('baseline', 'baseline_logits'), ('candidate_head', 'candidate_logits')]:
        rows = []
        seen = set()
        for prior in old['ordinary'][method]['rows']:
            i = index[(prior['job_id'], prior['event_id'])]
            seen.add(i)
            scores = cache[key][i]
            label = int(cache['labels'][i])
            assert classes[label] == prior['true_label']
            row = dict(prior)
            for mode, actor_key in [('automatic', 'original_automatic_actor'), ('human', 'human_actor')]:
                pred = restricted(scores, prior[actor_key], classes)
                row[mode+'_actor_conditioned'] = classes[pred]
                row[mode+'_actor_conditioned_joint_correct'] = pred == label
                row[mode+'_actor_conditioned_unknown'] = pred == 0
                row[mode+'_12class_prediction'] = prior[mode+'_actor_conditioned']
            rows.append(row)
        assert seen == ids
        summary = {k: sum(r[k] for r in rows) for k in [
            'unconditioned_joint_correct', 'automatic_actor_correct',
            'automatic_actor_conditioned_joint_correct', 'human_actor_conditioned_joint_correct',
            'automatic_actor_conditioned_unknown', 'human_actor_conditioned_unknown']}
        comparisons = {}
        for mode in ['automatic', 'human']:
            condition = mode+'_actor_conditioned_joint_correct'
            comparisons[mode+'_vs_joint25'] = {
                'gains': [r['event_id'] for r in rows if not r['unconditioned_joint_correct'] and r[condition]],
                'losses': [r['event_id'] for r in rows if r['unconditioned_joint_correct'] and not r[condition]],
            }
        ordinary[method] = {'total': 30, **summary, 'comparisons': comparisons, 'rows': rows}
    learning = {mode: paired(ordinary['baseline']['rows'], ordinary['candidate_head']['rows'],
                            mode+'_actor_conditioned_joint_correct') for mode in ['automatic', 'human']}
    public_logits = np.load(SOURCE/'automatic_events/classification_logits.npz')
    public_rows = []
    for prior in old['public_automatic']['rows']:
        scores = public_logits['original_audio__'+prior['candidate_id']+'__automatic_window']
        pred = restricted(scores, prior['automatic_actor'], classes)
        row = dict(prior)
        row['actor_12class_prediction'] = prior['actor_conditioned']
        row['actor_conditioned'] = classes[pred]
        row['actor_conditioned_correct'] = bool(prior['true_label'] and classes[pred] == prior['true_label'])
        row['unknown'] = pred == 0
        public_rows.append(row)
    public = {'author_contacts': 20, 'candidates': len(public_rows),
              'unconditioned_correct_out_of20': sum(r['unconditioned_correct'] for r in public_rows),
              'candidate_actor_conditioned_correct_out_of20': sum(r['actor_conditioned_correct'] for r in public_rows),
              'unknown_outputs': sum(r['unknown'] for r in public_rows), 'rows': public_rows}
    pose_old = json.loads((pose_dir/'classification_result.json').read_text())
    pose_logits = np.load(pose_dir/'classification_logits.npz')
    pose = {}
    for name, prior in pose_old['checks'].items():
        actor = prior['true_label'].split('_', 1)[0]
        pred = restricted(pose_logits[name], actor, classes)
        pose[name] = {'true_label': prior['true_label'], 'joint25_prediction': prior['prediction'],
                      'actor12_prediction': prior['known_target_conditioned_prediction'],
                      'actor13_prediction': classes[pred], 'actor13_correct': classes[pred] == prior['true_label']}
    assert all(file_hash(Path(p)) == h for p, h in hashes.items())
    result = {'created_at': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': file_hash(OUT/'protocol_with_unknown.json'),
              'ordinary': ordinary, 'learning': learning, 'public_automatic': public,
              'pose_control_sensitivity': pose, 'scope': protocol['scope'],
              'model_inference_training_or_production_changes': False}
    atomic_json(target, result)
    print(json.dumps({'ordinary': {k: {a:b for a,b in r.items() if a not in ['rows', 'comparisons']} for k,r in ordinary.items()},
                      'learning': learning, 'public': {k:v for k,v in public.items() if k!='rows'},
                      'pose': pose}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
