"""Fixed 2x2 modality swaps on a metadata-selected aligned author-val rally."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/audit/
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from analysis_jobs import atomic_json, file_hash
from bst_public_data import public_classifier, values, ARRAYS
from specialist_model import ROOT, BST_NAME

OUT = ROOT.parent / '03_数据/击球标注/根因诊断/2026-10-15_同源前端对照'


def pack(classifier, joints, pos, shuttle):
    h = classifier.helpers
    j, p, s, n = h['make_seq_len_same'](100, joints, pos, shuttle)
    j = np.concatenate([j, h['create_bones'](j, classifier.bones)], axis=-2).reshape(100, 2, 72)
    return j.astype('float32'), p.astype('float32'), s.astype('float32'), int(n)


def local_pose(classifier, dense, order=None):
    # This is the production candidate ordering rule, independent of class labels.
    ankles = {side: np.median([np.mean([p['keypoints'][k]['y'] for k in [15, 16]])
                              for f in dense['frames'] if (p := f['players'].get(side)) is not None])
              for side in ['near', 'far']}
    if order is None:
        order = sorted(ankles, key=ankles.get)
    joints, paired = [], []
    for f in dense['frames']:
        people = [f['players'].get(side) for side in order]
        present = all(p is not None for p in people)
        paired.append(present)
        if present:
            pts = np.asarray([[[k['x'], k['y']] for k in p['keypoints']] for p in people], dtype=float)
            boxes = np.asarray([p['bbox'] for p in people])
            assert np.isfinite(pts).all() and np.isfinite(boxes).all()
            assert np.all(np.linalg.norm(boxes[:, 2:] - boxes[:, :2], axis=-1) > 0)
            joints.append(classifier.helpers['normalize_joints'](pts, boxes, center_align=True))
        else:
            joints.append(np.zeros((2, 17, 2)))
    return np.asarray(joints, dtype=np.float32), np.asarray(paired), order, ankles


def summarize(scores, labels, classes):
    pred = scores.argmax(1)
    gt_stroke = np.asarray([classes[y].split('_', 1)[-1] for y in labels])
    pred_stroke = np.asarray([classes[y].split('_', 1)[-1] for y in pred])
    gt_actor = np.asarray([classes[y].split('_', 1)[0] for y in labels])
    pred_actor = np.asarray([classes[y].split('_', 1)[0] for y in pred])
    return {'total': len(labels), 'joint_25class_correct': int((pred == labels).sum()),
            'stroke_only_correct': int((pred_stroke == gt_stroke).sum()),
            'actor_correct': int((pred_actor == gt_actor).sum()),
            'joint_top2_correct': int(np.any(np.argsort(-scores, axis=1)[:, :2] == labels[:, None], axis=1).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', choices=['per_clip', 'continuous'], default='per_clip')
    args = parser.parse_args()
    prefix = '' if args.context == 'per_clip' else 'continuous_'
    started = time.perf_counter()
    assert json.loads((OUT / 'timeline_visual_review.json').read_text())['all_20_anchors_same_continuous_rally']
    protocol = json.loads((OUT / 'paired_comparison_protocol.json').read_text())
    protocol_hash = file_hash(OUT / 'paired_comparison_protocol.json')
    assert protocol['selected_samples_sha256'] == file_hash(OUT / 'selected_rally_samples.csv')
    rows = json.loads((OUT / 'clips_manifest.json').read_text())['clips']
    frozen = list(csv.DictReader((OUT / 'selected_rally_samples.csv').open()))
    assert [r['sample_id'] for r in rows] == [f'35_1_5_{i}' for i in range(1, 21)]
    assert [r['sample_id'] for r in rows] == [r['sample_id'] for r in frozen]
    for row, original in zip(rows, frozen):
        assert all(row[k] == original[k] for k in original)
        assert Path(row['author_triplet_base']).name == row['sample_id']
        assert Path(row['author_triplet_base']).parent.name == row['author_25class']
        assert row['frames'] == int(original['actual_author_npy_frame_count'])
    assert all(Path(r['author_triplet_base']).is_relative_to(ARRAYS / 'val') for r in rows)
    torch.set_num_threads(2)
    classifier = public_classifier()
    classes = classifier.classes
    labels = np.asarray([classes.index(r['author_25class']) for r in rows])
    cases = {name: [] for name in protocol['conditions']}
    cases.update({'common_pair_mask__' + name: [] for name in protocol['conditions']})
    cases.update({'rgb_same_boxes__' + name: [] for name in protocol['conditions']})
    input_records, quality = [], []
    raw_arrays = {}
    continuous = None
    if args.context == 'continuous':
        folder = OUT / 'continuous_rally'
        record = json.loads((folder / 'clips_manifest.json').read_text())['clips'][0]
        cd = json.loads((folder / 'pose/dense_pose.json').read_text())
        cr = json.loads((folder / 'pose/dense_pose_rgb_same_boxes.json').read_text())
        ct = json.loads((folder / 'tracknet/trajectory.json').read_text())
        for data in [cd, cr, ct]:
            assert len(data['frames']) == record['frames'] == 603
            assert data['source_sha256'] == record['source_sha256']
        # Production defines side order once over the complete analyzed video.
        full_j, full_pair, full_order, full_ankles = local_pose(classifier, cd)
        full_rgb_j, _, _, _ = local_pose(classifier, cr, order=full_order)
        continuous = (record, cd, cr, ct, full_order, full_ankles)
    for row in rows:
        folder = OUT / 'clips' / row['sample_id']
        n = row['frames']
        if continuous is None:
            dense = json.loads((folder / 'pose/dense_pose.json').read_text())
            rgb_dense = json.loads((folder / 'pose/dense_pose_rgb_same_boxes.json').read_text())
            trajectory = json.loads((folder / 'tracknet/trajectory.json').read_text())
            expected_source_sha256 = row['source_sha256']
        else:
            record, cd, cr, ct, full_order, full_ankles = continuous
            start = row['global_frame_start'] - record['global_frame_start']
            assert start >= 0 and start + n <= 603
            ids = list(range(start, start + n))
            for data in [cd, cr]:
                assert [data['frames'][i]['global_frame'] for i in ids] == list(range(row['global_frame_start'], row['global_frame_start'] + n))
            dense, rgb_dense, trajectory = [{**data, 'frames': [{**data['frames'][i], 'frame_index': k} for k, i in enumerate(ids)]} for data in [cd, cr, ct]]
            expected_source_sha256 = record['source_sha256']
        for data in [dense, rgb_dense, trajectory]:
            assert data['source_sha256'] == expected_source_sha256
            assert [data['source_width'], data['source_height']] == [1280, 720]
            assert len(data['frames']) == n
            assert all(f['frame_index'] == i for i, f in enumerate(data['frames']))
        base = row['author_triplet_base']
        aj, ap, ass = [np.load(base + suffix, allow_pickle=False)
                       for suffix in ['_joints.npy', '_pos.npy', '_shuttle.npy']]
        assert aj.shape == (n, 2, 17, 2) and ap.shape == (n, 2, 2) and ass.shape == (n, 2)
        lj, local_paired, order, ankles = local_pose(classifier, dense, order=full_order if continuous is not None else None)
        if continuous is not None:
            assert np.array_equal(lj, full_j[ids])
            ankles = full_ankles
        rj, rgb_paired, _, _ = local_pose(classifier, rgb_dense, order=order)
        assert np.array_equal(local_paired, rgb_paired), 'RGB must reuse identical boxes and assignments'
        for a, b in zip(dense['frames'], rgb_dense['frames']):
            for side in order:
                p, q = a['players'][side], b['players'][side]
                assert (p is None) == (q is None)
                if p is not None:
                    assert p['bbox'] == q['bbox']
        ls = np.asarray([(b['rectified_xy'] if b['inpaint_requested'] else b['raw_xy']) or [0, 0]
                         for b in trajectory['frames']], dtype=np.float32)
        ls = classifier.helpers['normalize_shuttlecock'](ls, 1280, 720)
        author_paired = np.any(aj != 0, axis=(1, 2, 3))
        assert np.array_equal(author_paired, np.any(ap != 0, axis=(1, 2)))
        common = author_paired & local_paired
        records = {'author_pose_author_shuttle': (aj, ass),
                   'local_pose_author_shuttle': (lj, ass),
                   'author_pose_local_shuttle': (aj, ls),
                   'local_pose_local_shuttle': (lj, ls)}
        for name, (j, s) in records.items():
            cases[name].append(pack(classifier, j, ap, s))
            masked = s.copy()
            masked[~common] = 0
            cases['common_pair_mask__' + name].append(pack(classifier, j, ap, masked))
            cases['rgb_same_boxes__' + name].append(pack(classifier, rj if name.startswith('local_pose') else aj, ap, s))
        reference = values(classifier, str(Path(base).relative_to(ARRAYS)))
        assert all(np.array_equal(a, b) for a, b in zip(reference[:3], cases['author_pose_author_shuttle'][-1][:3]))
        assert reference[3] == n
        author_visible, local_visible = np.any(ass != 0, axis=1), np.any(ls != 0, axis=1)
        both = author_visible & local_visible
        ball_distance = np.linalg.norm((ls[both] - ass[both]) * [1280, 720], axis=1)
        pose_distance = np.linalg.norm(lj[common] - aj[common], axis=-1)
        rgb_pose_distance = np.linalg.norm(rj[common] - aj[common], axis=-1)
        quality.append({'sample_id': row['sample_id'], 'frames': n, 'player_input_order': order,
                        'slot_median_ankle_y': {s: float(v) if np.isfinite(v) else None for s, v in ankles.items()},
                        'player_order_unresolved_missing_slot': any(not np.isfinite(v) for v in ankles.values()),
                        'author_paired_frames': int(author_paired.sum()), 'local_paired_frames': int(local_paired.sum()),
                        'common_pair_frames': int(common.sum()),
                        'author_shuttle_nonzero_frames': int(author_visible.sum()),
                        'local_shuttle_nonzero_frames': int(local_visible.sum()),
                        'both_shuttle_nonzero_frames': int(both.sum()),
                        'shuttle_difference_px_median': float(np.median(ball_distance)) if len(ball_distance) else None,
                        'shuttle_difference_px_p90': float(np.quantile(ball_distance, .9)) if len(ball_distance) else None,
                        'pose_normalized_joint_difference_mean': float(pose_distance.mean()) if pose_distance.size else None,
                        'rgb_pose_normalized_joint_difference_mean': float(rgb_pose_distance.mean()) if rgb_pose_distance.size else None,
                        'coordinate_difference_is_not_ground_truth_error': True})
        for name, data in cases.items():
            digest = hashlib.sha256()
            for a in [data[-1][0], data[-1][2], np.asarray(data[-1][3], dtype='int64')]:
                digest.update(a.tobytes())
            input_records.append({'sample_id': row['sample_id'], 'condition': name,
                                  'input_sha256': digest.hexdigest(),
                                  'frontend_context': args.context,
                                  'frontend_video_sha256': expected_source_sha256})
        for name, a in [('author_joints', aj), ('author_shuttle', ass), ('local_joints', lj),
                        ('rgb_joints', rj), ('local_shuttle', ls), ('common_pair_mask', common)]:
            raw_arrays[row['sample_id'] + '__' + name] = a
    logits = {}
    for name, data in cases.items():
        j, s = [np.stack([v[k] for v in data]) for k in [0, 2]]
        n = np.asarray([v[3] for v in data], dtype='int64')
        with torch.inference_mode():
            logits[name] = classifier.model(torch.from_numpy(j), torch.from_numpy(s), torch.from_numpy(n)).numpy()
    # Baseline agreement with the independent whole-author-validation replay.
    cache = np.load(OUT.parent / '2026-10-15_作者验证集BST版本对照/predictions.npz', allow_pickle=False)
    lookup = {b: i for i, b in enumerate(cache['bases'])}
    reference = np.stack([cache['bst0_logits'][lookup[str(Path(r['author_triplet_base']).relative_to(ARRAYS))]] for r in rows])
    assert np.array_equal(reference.argmax(1), logits['author_pose_author_shuttle'].argmax(1))
    checks = {name: summarize(scores, labels, classes) for name, scores in logits.items()}
    baseline = logits['author_pose_author_shuttle'].argmax(1) == labels
    for name, scores in logits.items():
        correct = scores.argmax(1) == labels
        checks[name]['lost_from_author_correct'] = int((baseline & ~correct).sum())
        checks[name]['rescued_author_wrong'] = int((~baseline & correct).sum())
    details = []
    for i, row in enumerate(rows):
        detail = {'sample_id': row['sample_id'], 'label': row['author_25class'],
                  'hit_frame_num': row['hit_frame_num'], 'input_quality': quality[i], 'conditions': {}}
        for name, scores in logits.items():
            ranking = np.argsort(-scores[i])
            detail['conditions'][name] = {'predicted': classes[int(ranking[0])],
                    'correct': bool(ranking[0] == labels[i]),
                    'top2': [classes[int(k)] for k in ranking[:2]],
                    'true_label_logit': float(scores[i, labels[i]])}
        details.append(detail)
    assert file_hash(OUT / 'paired_comparison_protocol.json') == protocol_hash, 'Protocol changed during inference'
    np.savez_compressed(OUT / (prefix + 'paired_predictions.npz'), labels=labels, classes=np.asarray(classes),
                        sample_ids=np.asarray([r['sample_id'] for r in rows]), **logits)
    np.savez_compressed(OUT / (prefix + 'modality_arrays.npz'), **raw_arrays)
    atomic_json(OUT / (prefix + 'paired_input_manifest.json'), {'inputs': input_records})
    atomic_json(OUT / (prefix + 'paired_result.json'), {'created_at': datetime.now(timezone.utc).isoformat(),
                'protocol_sha256': protocol_hash, 'frontend_context': args.context,
                'model_sha256': file_hash(ROOT / 'models/bst' / BST_NAME),
                'script_sha256': file_hash(Path(__file__)), 'checks': checks, 'samples': details,
                'author_baseline_top1_identical_to_independent_4241_replay': True,
                'author_baseline_max_logit_difference': float(np.max(abs(reference - logits['author_pose_author_shuttle']))),
                'predictions_sha256': file_hash(OUT / (prefix + 'paired_predictions.npz')),
                'modality_arrays_sha256': file_hash(OUT / (prefix + 'modality_arrays.npz')),
                'input_manifest_sha256': file_hash(OUT / (prefix + 'paired_input_manifest.json')),
                'elapsed_seconds': time.perf_counter() - started,
                'scope': 'Single metadata-selected author validation rally; known shot boundaries. Correlated20 strokes, historical encoding differs. Not general accuracy, not hit detection.'})
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
