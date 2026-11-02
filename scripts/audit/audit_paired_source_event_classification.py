"""Compare automatic event windows with fixed author windows, no training."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/audit/
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from analysis_jobs import atomic_json, file_hash
from specialist_bst import SpecialistBST
from specialist_model import ROOT, BST_NAME

OUT = ROOT.parent / '03_数据/击球标注/根因诊断/2026-10-15_同源前端对照'
DEST = OUT / 'automatic_events'


def main():
    started = time.perf_counter()
    event_files = {'raw_visual': DEST / 'raw_visual_candidates.json',
                   'no_audio': DEST / 'no_audio_production_candidates.json',
                   'original_audio': DEST / 'original_audio_production_candidates.json'}
    evaluation_files = {'raw_visual': DEST / 'no_audio_result.json',
                        'no_audio': DEST / 'no_audio_result.json',
                        'original_audio': DEST / 'original_audio_result.json'}
    dependencies = [*event_files.values(), *evaluation_files.values(),
                    OUT / 'selected_rally_samples.csv', OUT / 'clips_manifest.json',
                    OUT / 'continuous_rally/pose/dense_pose.json',
                    OUT / 'continuous_rally/tracknet/trajectory.json',
                    ROOT / 'specialist_bst.py', ROOT / 'specialist_pipeline.py',
                    ROOT / 'models/bst' / BST_NAME, Path(__file__)]
    dependency_hashes = {str(p): file_hash(p) for p in dependencies}
    protocol = {'declared_at': datetime.now(timezone.utc).isoformat(),
        'dependency_hashes': dependency_hashes,
        'conditions': ['original automatic event window', 'same candidate restored author window for matched hits'],
        'matching': 'Use already frozen center-only one-to-one .2s matches unchanged; no rematching using classification.',
        'model': 'Original BST-0 fixed weights and production SpecialistBST.inputs normalization; no personal models.',
        'primary': 'Time-matched, candidate-target-correct and 25class-top1-correct contacts / all20 author contacts, before input or VLM gates.',
        'secondary': 'Conditional correctness among matches, target/coverage/softmax gate, exact author-window restoration for same matched candidates.',
        'gate': 'Production pair>=.85, rawshuttle>=.4, nonunknown label, actor agrees with candidate, uncalibrated softmax>=.4; target>=90% presence over consecutive +-0.6s evidence frames and at least8 evidence frames. No VLM run.',
        'no_gt_leak': 'Author GT used for evaluation and explicit counterfactual windows only, never for automatic events.',
        'no_tuning_or_training': True,
        'scope': 'One author-val rally; end-to-end to pre-VLM stage only, not full website or general accuracy.'}
    protocol_path = DEST / 'classification_protocol.json'
    if protocol_path.exists():
        raise FileExistsError('Preserve frozen classification run')
    atomic_json(protocol_path, protocol)
    protocol_hash = file_hash(protocol_path)
    dense = json.loads((OUT / 'continuous_rally/pose/dense_pose.json').read_text())
    trajectory = json.loads((OUT / 'continuous_rally/tracknet/trajectory.json').read_text())
    classifier = SpecialistBST(dense, trajectory)
    torch.set_num_threads(2)
    rows = json.loads((OUT / 'clips_manifest.json').read_text())['clips']
    row_lookup = {r['sample_id']: r for r in rows}
    known = json.loads((OUT / 'continuous_paired_result.json').read_text())
    known_lookup = {r['sample_id']: r for r in known['samples']}
    results = {}
    saved_logits, prepared_inputs = {}, []
    for stage, file in event_files.items():
        events = json.loads(file.read_text())['candidates']
        evaluation = json.loads(evaluation_files[stage].read_text())['raw_visual' if stage == 'raw_visual' else 'production_refined']
        matches = {r['candidate_id']: r for r in evaluation['matches']}
        details = []
        for event in events:
            match = matches.get(event['id'])
            cases = [('automatic_window', event)]
            if match:
                row = row_lookup[match['sample_id']]
                lo = row['global_frame_start'] - 16756
                hi = lo + row['frames'] - 1
                restored = {**event, 'start_frame': lo, 'end_frame': hi,
                            'start': dense['frames'][lo]['source_time'],
                            'end': dense['frames'][hi]['source_time'] + 1 / 30}
                cases.append(('author_window_restored', restored))
            item = {'candidate_id': event['id'], 'peak_time': event['peak_time'],
                    'matched': match, 'conditions': {}}
            for name, candidate in cases:
                values, quality = classifier.inputs(candidate)
                j, p, s, n = values
                with torch.inference_mode():
                    scores = classifier.model(torch.from_numpy(j)[None], torch.from_numpy(s)[None], torch.tensor([n])).numpy()[0]
                predicted = classifier.classes[int(scores.argmax())]
                label = row_lookup[match['sample_id']]['author_25class'] if match else None
                expected_actor = 'Top' if candidate['player'] == classifier.order[0] else 'Bottom'
                predicted_actor, _, predicted_stroke = predicted.partition('_')
                softmax = np.exp(scores - scores.max()); softmax /= softmax.sum()
                qualified = (quality['paired_player_fraction'] >= .85 and quality['raw_shuttle_fraction'] >= .4
                             and predicted_actor == expected_actor and predicted_stroke not in ['', '未知球種'] and float(softmax.max()) >= .4)
                lo = max(candidate['start_frame'], candidate['frame_index'] - round(.6 * dense['sample_fps']))
                hi = min(candidate['end_frame'], candidate['frame_index'] + round(.6 * dense['sample_fps']))
                people = [dense['frames'][i]['players'].get(candidate['player']) for i in range(lo, hi + 1)]
                evidence_ok = len(people) >= 8 and sum(p is not None for p in people) / len(people) >= .9
                counterfactual_same = None
                if name == 'author_window_restored':
                    expected_prediction = known_lookup[match['sample_id']]['conditions']['local_pose_local_shuttle']['predicted']
                    counterfactual_same = predicted == expected_prediction
                    assert counterfactual_same, 'Restored window must match independently prepared fixed-window replay'
                item['conditions'][name] = {'prediction': predicted, 'true_label': label,
                        'label_correct': bool(predicted == label) if label else None,
                        'candidate_target_agreement': predicted_actor == expected_actor,
                        'paired_fraction': quality['paired_player_fraction'], 'raw_shuttle_fraction': quality['raw_shuttle_fraction'],
                        'uncalibrated_softmax': float(softmax.max()),
                        'input_target_gate_passed': bool(qualified), 'consecutive_evidence_gate_passed': bool(evidence_ok),
                        'start_frame': candidate['start_frame'], 'end_frame': candidate['end_frame'],
                        'video_len': n, 'restored_top1_agrees_with_independent_known_window_run': counterfactual_same}
                key = stage + '__' + event['id'] + '__' + name
                saved_logits[key] = scores
                digest = hashlib.sha256(j.tobytes() + s.tobytes() + np.asarray(n, dtype='int64').tobytes()).hexdigest()
                prepared_inputs.append({'key': key, 'input_sha256': digest})
            details.append(item)
        matched = [d for d in details if d['matched']]
        target_correct = [d for d in matched if d['matched']['actor_correct']]
        automatic_correct = [d for d in target_correct if d['conditions']['automatic_window']['label_correct']]
        restored_correct = [d for d in target_correct if d['conditions']['author_window_restored']['label_correct']]
        gated_correct = [d for d in automatic_correct if d['conditions']['automatic_window']['input_target_gate_passed'] and d['conditions']['automatic_window']['consecutive_evidence_gate_passed']]
        results[stage] = {'candidate_count': len(events), 'author_contacts': 20,
            'time_matched_contacts': len(matched), 'time_and_target_correct_contacts': len(target_correct),
            'time_target_type_correct_contacts': len(automatic_correct),
            'time_target_type_correct_fraction_of_20': len(automatic_correct) / 20,
            'time_target_type_and_input_gate_correct_contacts': len(gated_correct),
            'restored_author_window_time_target_type_correct_contacts': len(restored_correct),
            'restored_window_gains': [d['matched']['sample_id'] for d in target_correct if not d['conditions']['automatic_window']['label_correct'] and d['conditions']['author_window_restored']['label_correct']],
            'restored_window_losses': [d['matched']['sample_id'] for d in target_correct if d['conditions']['automatic_window']['label_correct'] and not d['conditions']['author_window_restored']['label_correct']],
            'details': details}
    assert all(file_hash(Path(p)) == h for p, h in dependency_hashes.items())
    assert file_hash(protocol_path) == protocol_hash
    np.savez_compressed(DEST / 'classification_logits.npz', **saved_logits)
    atomic_json(DEST / 'classification_input_manifest.json', {'inputs': prepared_inputs})
    atomic_json(DEST / 'classification_result.json', {'created_at': datetime.now(timezone.utc).isoformat(),
                'protocol_sha256': protocol_hash, 'checks': results,
                'fixed_author_windows_local_both_correct': known['checks']['local_pose_local_shuttle']['joint_25class_correct'],
                'elapsed_seconds': time.perf_counter() - started,
                'production_or_weights_or_labels_changed': False,
                'scope': protocol['scope']})
    print(json.dumps({k: {a: b for a, b in r.items() if a != 'details'} for k, r in results.items()}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
