"""Read-only numerical audit of author BST preprocessing and the local adapter.

New audit artifacts are written separately. Existing labels, coordinate caches,
checkpoints and frozen predictions are never modified. This is development
diagnosis, not a fresh test or an optimisation/selection experiment.
"""
import ast
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from analysis_jobs import DATA, ROOT, atomic_json, file_hash
from bst_public_data import ARRAYS, FOLDER, public_classifier, values
from finetune_personal_bst import metrics
from input_quality import InputQuality
from specialist_bst import SpecialistBST, VENDOR, author_helpers
from specialist_model import BST_NAME

VERSION = 'bst-adapter-contract-numerical-audit-v1'
OUT = DATA / '击球标注/适配契约审计/2026-10-13_输入契约'
DIAGNOSTIC = DATA / '击球标注/输入修复诊断/2026-10-12_配对输入修复'
COMPARISON = DATA / '击球标注/直接BST微调/运行记录/10164eff5b294944b1aac13379867373'


def author_collation():
    path = VENDOR / 'preparing_data/prepare_train_on_shuttleset_merged.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'pad_and_augment_one_npy_video')
    namespace = {'np': np, **author_helpers()}
    helper_path = VENDOR / 'preparing_data/shuttleset_dataset.py'
    interpolation = next(n for n in ast.parse(helper_path.read_text()).body
                         if isinstance(n, ast.FunctionDef) and n.name == 'interpolate_joints')
    exec(compile(ast.Module(body=[interpolation], type_ignores=[]), str(helper_path), 'exec'), namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['pad_and_augment_one_npy_video']


def logits(model, inputs, batch_size=16):
    output = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            part = inputs[start:start + batch_size]
            output.append(model(torch.as_tensor(np.stack([v[0] for v in part])),
                                torch.as_tensor(np.stack([v[2] for v in part])),
                                torch.tensor([v[3] for v in part])).cpu().numpy())
    return np.concatenate(output)


def reconstruct_public_adapter(j, s, h):
    """Inverse a known bbox normalisation to test the complete input adapter.

    These are synthetic raw coordinates constructed from the author's arrays;
    they do not test the real pose/shuttle detectors or scene geometry.
    """
    frames, ball = [], []
    width, height = 1280, 1440
    for i, (pose, shuttle) in enumerate(zip(j, s)):
        people = {}
        for p, side in enumerate(['far', 'near']):
            if (pose[p] == 0).all():
                people[side] = None
                continue
            center = np.asarray([600., 240. if p == 0 else 960.])
            bbox = np.r_[center - [60., 80.], center + [60., 80.]]
            raw = pose[p] * 200. + center
            people[side] = {'bbox': bbox.tolist(), 'keypoints': [
                {'x': float(x), 'y': float(y), 'score': 1., 'in_frame': True} for x, y in raw]}
        frames.append({'frame_index': i, 'source_time': i / 30., 'players': people})
        xy = (shuttle * [width, height]).tolist()
        ball.append({'frame_index': i, 'raw_xy': xy, 'rectified_xy': xy,
                     'raw_visible': bool(np.any(shuttle)), 'raw_heatmap_peak': 1., 'inpaint_requested': False})
    dense = {'id': 'synthetic-contract', 'source_sha256': 'synthetic-contract',
             'source_width': width, 'source_height': height, 'sample_fps': 30.,
             'end': len(frames) / 30., 'frames': frames}
    trajectory = {'source_sha256': 'synthetic-contract', 'frames': ball}
    # Use the production input method without loading another model per case.
    adapter = object.__new__(SpecialistBST)
    adapter.dense, adapter.trajectory, adapter.calibration = dense, trajectory, None
    adapter.helpers, adapter.bones, adapter.order = h, h['get_bone_pairs'](), ['far', 'near']
    adapter.coordinate_quality = InputQuality(dense, trajectory)
    event = {'start_frame': 0, 'end_frame': len(j) - 1, 'frame_index': len(j) // 2,
             'peak_time': (len(j) // 2) / 30., 'player': 'far'}
    return adapter.inputs(event)[0]


def seq_indices(h, n):
    # Put source frame+1 in every joint; zero padding remains identifiable.
    marker = np.arange(1, n + 1, dtype=float)
    j = np.broadcast_to(marker[:, None, None, None], (n, 2, 17, 2)).copy()
    p = np.zeros((n, 2, 2))
    s = np.broadcast_to(marker[:, None], (n, 2)).copy()
    a, _, _, length = h['make_seq_len_same'](100, j, p, s)
    return a[:length, 0, 0, 0].astype(int) - 1, length


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [ROOT / 'models/bst' / BST_NAME, DATA / '训练库/训练样本.json']
    protected += list((DATA / '击球标注').glob('J_*/annotations.json'))
    before = {str(p): file_hash(p) for p in protected}
    model = public_classifier()
    h = model.helpers
    collate = author_collation()
    pose_example = np.arange(68, dtype=float).reshape(2, 17, 2) + 50.
    bbox_example = np.asarray([[10., 20., 110., 220.], [300., 400., 340., 480.]])
    centers = (bbox_example[:, :2] + bbox_example[:, 2:]) / 2
    diagonals = np.linalg.norm(bbox_example[:, 2:] - bbox_example[:, :2], axis=1)
    independent_normal = (pose_example - centers[:, None]) / diagonals[:, None, None]
    normal_error = float(abs(independent_normal - h['normalize_joints'](
        pose_example, bbox_example, center_align=True)).max())
    assert normal_error < 1e-14
    nonzero_pose = np.arange(1., 69.).reshape(1, 2, 17, 2)
    independent_bones = np.stack([nonzero_pose[:, :, end] - nonzero_pose[:, :, start]
                                  for start, end in h['get_bone_pairs']()], axis=-2)
    assert np.array_equal(independent_bones, h['create_bones'](nonzero_pose, h['get_bone_pairs']()))
    ball_example = np.asarray([[0., 0.], [1280., 720.], [-5., 1440.]])
    assert np.array_equal(ball_example / [1280., 720.], h['normalize_shuttlecock'](ball_example, 1280, 720))
    lengths = [1, 2, 49, 50, 99, 100, 101, 149, 150, 151, 199, 200, 249, 250, 251, 300, 301]
    sequence = []
    for n in lengths:
        rng = np.random.default_rng(n)
        j, p, s = rng.normal(size=(n, 2, 17, 2)), rng.normal(size=(n, 2, 2)), rng.normal(size=(n, 2))
        direct = collate(100, j, p, s, h['get_bone_pairs']())
        jj, pp, ss, ll = h['make_seq_len_same'](100, j.astype('float32'), p.astype('float32'), s.astype('float32'))
        merged = np.concatenate([jj, h['create_bones'](jj, h['get_bone_pairs']())], -2).reshape(100, 2, 72)
        error = float(abs(merged - direct[2].reshape(100, 2, 72)).max())
        assert error == 0 and ll == direct[-1]
        indices, length = seq_indices(h, n)
        sequence.append({'raw_length': n, 'model_valid_length': length,
                         'last_retained_source_index': int(indices[-1]),
                         'tail_frames_not_used': n - 1 - int(indices[-1]), 'author_local_max_abs_error': error})
    index = json.loads((FOLDER / 'index.json').read_text())
    val = [r for r in index['samples'] if r['split'] == 'val']
    # All author validation rows are audited for the failure representation.
    public_count = Counter()
    public_lengths = []
    selected = []
    by_class = Counter()
    for row in val:
        j = np.load(str(ARRAYS / row['base']) + '_joints.npy', allow_pickle=False)
        s = np.load(str(ARRAYS / row['base']) + '_shuttle.npy', allow_pickle=False)
        failed = (j == 0).all(axis=(1, 2, 3))
        public_lengths.append(len(j))
        public_count.update({'clips': 1, 'frames': len(j), 'pair_missing_frames': int(failed.sum()),
                             'pair_missing_but_ball_nonzero': int((failed & (s != 0).any(1)).sum()),
                             'partial_one_player_missing_frames': int(((j == 0).all(axis=(2, 3)).sum(1) == 1).sum())})
        if by_class[row['class']] < 2:
            selected.append(row)
            by_class[row['class']] += 1
    synthetic, expected, numerical = [], [], []
    for row in selected:
        j, p, s = [np.load(str(ARRAYS / row['base']) + suffix, allow_pickle=False)
                   for suffix in ['_joints.npy', '_pos.npy', '_shuttle.npy']]
        reconstructed = reconstruct_public_adapter(j, s, h)
        standard = values(model, row['base'])
        joint_error = float(abs(reconstructed[0] - standard[0]).max())
        ball_error = float(abs(reconstructed[2] - standard[2]).max())
        assert joint_error < 1e-6 and ball_error < 1e-6 and reconstructed[3] == standard[3]
        synthetic.append(reconstructed)
        expected.append(standard)
        numerical.append({'sample_id': row['sample_id'], 'class': row['class'],
                          'joint_max_abs_error': joint_error, 'shuttle_max_abs_error': ball_error,
                          'length_matches': reconstructed[3] == standard[3]})
    synth_logits = logits(model.model, synthetic)
    standard_logits = logits(model.model, expected)
    logit_error = float(abs(synth_logits - standard_logits).max())
    assert logit_error < 1e-4 and (synth_logits.argmax(1) == standard_logits.argmax(1)).all()

    # Single-factor repair on exactly the old frozen and translated 22 inputs.
    diagnostic = json.loads((DIAGNOSTIC / 'result.json').read_text())
    paired = {}
    with np.load(DIAGNOSTIC / 'inputs.npz', allow_pickle=False) as d:
        keys = {tuple(k): i for i, k in enumerate(d['keys'].tolist())}
        for variant in ['frozen_replay', 'human_center_same_duration']:
            original_inputs, repaired_inputs, yy, details = [], [], [], []
            for row in diagnostic['rows']:
                i = keys[(row['job_id'], row['event_id'], variant)]
                j, p, s, length = d['joints_and_bones'][i], d['positions'][i], d['shuttle'][i], int(d['video_len'][i])
                missing = (j == 0).all(axis=(1, 2))
                ss = s.copy()
                changed = missing & (s != 0).any(1)
                ss[missing] = 0
                original_inputs.append((j, p, s, length))
                repaired_inputs.append((j, p, ss, length))
                human = row['human']
                order = row['variants'][variant]['player_order']
                prefix = 'Top_' if order.index(human['player']) == 0 else 'Bottom_'
                from stroke_taxonomy import AUTHOR_MAP
                target = next(k for k, c in enumerate(model.classes)
                              if c.startswith(prefix) and AUTHOR_MAP[c.split('_', 1)[-1]] == human['stroke_type'])
                yy.append(target)
                details.append({'job_id': row['job_id'], 'event_id': row['event_id'],
                                'changed_shuttle_frames': int(changed[:length].sum()),
                                'model_valid_length': length})
            a = logits(model.model, original_inputs)
            b = logits(model.model, repaired_inputs)
            y = np.asarray(yy, dtype='int64')
            for row, detail, aa, bb in zip(diagnostic['rows'], details, a, b):
                expected_scores = row['variants'][variant]['scores']
                assert float(abs(torch.as_tensor(aa).softmax(-1).numpy() - expected_scores).max()) < 1e-5
                detail.update(baseline_label=model.classes[int(aa.argmax())],
                              repaired_label=model.classes[int(bb.argmax())],
                              max_logit_difference=float(abs(aa - bb).max()))
            paired[variant] = {'baseline': metrics(a, y, model.classes),
                               'author_failure_sync_repaired': metrics(b, y, model.classes),
                               'changed_clips': sum(r['changed_shuttle_frames'] > 0 for r in details),
                               'details': details}
            np.savez_compressed(OUT / (variant + '_pair_failure_sync.npz'),
                                baseline_logits=a, repaired_logits=b, labels=y)

    # Quantify source-frame retention and the same mismatch on all 70 human
    # development/training examples. No labels or features are used to fit a rule.
    fm = json.loads((COMPARISON / 'personal_features_manifest.json').read_text())
    local = []
    local_inputs, local_repaired, local_y = [], [], []
    keypoint_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                     'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                     'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                     'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    bbox_joint_checks = Counter()
    from personal_learning import extractor
    grouped = {}
    for row in fm['rows']:
        grouped.setdefault(row['job_id'], []).append(row)
    for job, rows in grouped.items():
        adapter, dense = extractor(job)
        for frame in dense['frames']:
            for person in frame['players'].values():
                if person is None:
                    continue
                bbox = np.asarray(person['bbox'])
                assert bbox.shape == (4,) and np.isfinite(bbox).all() and (bbox[2:] > bbox[:2]).all()
                assert [k['name'] for k in person['keypoints']] == keypoint_names
                bbox_joint_checks['people_frames_with_xyxy_bbox_and_coco17_names'] += 1
        for row in rows:
            adapter.order = list(row['input_quality']['player_input_order'])
            w = row['model_window']
            v, quality = adapter.inputs(w)
            j, p, s, length = v
            missing = (j == 0).all(axis=(1, 2))
            ss = s.copy()
            changed = missing & (s != 0).any(1)
            ss[missing] = 0
            indices, _ = seq_indices(h, w['end_frame'] - w['start_frame'] + 1)
            center = w['frame_index'] - w['start_frame']
            nearest = int(indices[np.argmin(abs(indices - center))])
            local.append({'job_id': job, 'event_id': row['event_id'], 'nominal_fps': dense['sample_fps'],
                          'raw_window_frame_count': w['end_frame'] - w['start_frame'] + 1,
                          'model_valid_length': length, 'center_source_relative_frame': center,
                          'nearest_retained_contact_frame_error': nearest - center,
                          'source_tail_not_used_frames': w['end_frame'] - w['start_frame'] - int(indices[-1]),
                          'changed_shuttle_frames': int(changed[:length].sum()),
                          'paired_player_fraction': quality['paired_player_fraction']})
            local_inputs.append(v)
            local_repaired.append((j, p, ss, length))
            local_y.append(row['target_class_id'])
    a, b = logits(model.model, local_inputs), logits(model.model, local_repaired)
    local_metrics = {'baseline': metrics(a, np.asarray(local_y, dtype='int64'), model.classes),
                     'author_failure_sync_repaired': metrics(b, np.asarray(local_y, dtype='int64'), model.classes),
                     'changed_clips': sum(r['changed_shuttle_frames'] > 0 for r in local), 'rows': local}
    assert all(file_hash(Path(path)) == sha for path, sha in before.items())
    result = {'version': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
              'original_model_sha256': file_hash(ROOT / 'models/bst' / BST_NAME),
              'code_sha256': file_hash(Path(__file__)),
              'author_preprocessing_sha256': file_hash(VENDOR / 'preparing_data/prepare_train_on_shuttleset.py'),
              'author_dataset_sha256': file_hash(VENDOR / 'preparing_data/shuttleset_dataset.py'),
              'local_adapter_sha256': file_hash(ROOT / 'specialist_bst.py'),
              'source_dataset_archive_sha256': index['source_archive_sha256'],
              'independent_formula_checks': {'bbox_xyxy_center_diagonal_normalization_max_error': normal_error,
                  'bone_end_minus_start_matches': True, 'shuttle_xy_resolution_division_matches': True},
              'real_frontend_field_contract': dict(bbox_joint_checks),
              'author_validation_failure_representation': dict(public_count),
              'author_validation_raw_length_summary': {'min': min(public_lengths), 'max': max(public_lengths),
                  'median': float(np.median(public_lengths)), 'p95': float(np.quantile(public_lengths, .95)),
                  'above100_count': sum(n > 100 for n in public_lengths)},
              'synthetic_end_to_end': {'samples': len(selected), 'rows': numerical,
                                       'max_logit_difference': logit_error, 'top1_all_identical': True,
                                       'scope': 'Inverse-known-normalisation fixture tests adapter, not actual frontends.'},
              'sequence_sampling': sequence, 'same22_single_factor': paired,
              'all70_single_factor': local_metrics, 'protected_sha256': before,
              'protected_files_unchanged': True, 'public_test_samples_read': 0,
              'scope': 'Preprocessing contract and paired development diagnostics; no training or independent accuracy claim.'}
    atomic_json(OUT / 'result.json', result)
    print(json.dumps({'public_failure': dict(public_count),
                      'synthetic_samples': len(selected), 'synthetic_max_logit_error': logit_error,
                      'same22': {k: {'before': v['baseline']['stroke_top1_correct'],
                                    'after': v['author_failure_sync_repaired']['stroke_top1_correct'],
                                    'changed_clips': v['changed_clips']} for k, v in paired.items()},
                      'all70': {'before': local_metrics['baseline']['stroke_top1_correct'],
                                'after': local_metrics['author_failure_sync_repaired']['stroke_top1_correct'],
                                'changed_clips': local_metrics['changed_clips']}, 'result': str(OUT / 'result.json')},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
