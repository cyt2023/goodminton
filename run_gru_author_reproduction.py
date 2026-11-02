"""Execute the fixed raw-video pilot of the published badminton GRU system."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from analysis_jobs import DATA, atomic_json, file_hash

OUT = DATA / 'GRU作者流程复现/2026-10-25'
FILTERED = OUT / 'author_assets/shuttlepose_filtered2/keras_tensors.npz'


def read(path):
    return json.loads(Path(path).read_text())


def plan():
    return read(OUT / 'raw_video_plan.json')['rows']


def tracknet():
    from reproduce_gru_author_tracknet import infer, load_model
    model, device = load_model('mps')
    for case in plan():
        folder = Path(case['output_folder']) / 'author_ball'
        if (folder / 'trajectory.json').exists():
            print('REUSE_AUTHOR_BALL', case['id'], flush=True)
            continue
        infer(case['source_path'], folder, case['id'], device=device, model=model)


def save_prediction(folder, result, dense, ball=None):
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    if arrays:
        np.savez_compressed(folder/'arrays.npz', **arrays)
    result = {key: value for key, value in result.items() if key not in arrays}
    result.update(source_sha256=dense['source_sha256'],
                  frontend_engine=dense.get('engine_version'),
                  ball_engine=ball.get('engine_version', ball.get('schema_version')) if ball else None)
    atomic_json(folder/'prediction.json', result)


def filtered2_legacy(dense, ball, postprocess):
    """Preserve the actual filtered2 CSV layout used to train the checkpoint.

    Upstream combine_domain_into_x drops bbox columns but not bbox_conf, then
    copies only34values. Each player therefore contributes confidence + first
    33 flattened XY coordinates, omitting right-ankleY. GT-domain is different.
    """
    import benchmark_hits_gru as gru
    from hit_gru import predict_windows
    rows, order = gru.image_coordinates(dense, ball, carry_pose=True)
    layout = rows.copy()
    for player, side in enumerate(order):
        scores, previous = [], 0.
        for frame in dense['frames']:
            person = frame['players'].get(side)
            if person:
                if 'bbox_score' not in person:
                    raise ValueError('Legacy filtered2 needs the actual detector confidence; no invented confidence sentinel.')
                previous = float(person['bbox_score'])
            scores.append(previous)
        start = 2+34*player
        layout[:, start:start+34] = np.column_stack([scores, rows[:, start:start+33]])
    with np.load(FILTERED, allow_pickle=False) as stored:
        weights = {key: stored[key] for key in stored.files}
    n = max(0, len(layout)-13)
    probs = np.zeros((n, 3), np.float32)
    for begin in range(0, n, 256):
        indices = np.arange(begin, min(begin+256, n))
        probs[indices] = predict_windows(layout[indices[:, None]+np.arange(14)], weights)
    labels = gru.naive_labels(probs) if postprocess == 'naive' else gru.maximise_hit_score(probs)
    events = gru.labels_to_events(labels, dense['frames'], order, probs, offset=13)
    return {'events': events, 'window_probs': probs, 'window_labels': labels,
        'feature_rows': layout, 'metadata': {
            'model': 'kwyoke-shuttlepose-filtered2-150', 'weights_sha256': file_hash(FILTERED),
            'postprocess': postprocess, 'camera_to_identity_order': order,
            'coordinate_layout': 'ball_xy + per-player[bbox_conf,k0x,k0y,...,k15y,k16x]; right-ankleY omitted by original source.',
            'layout_source': 'organise_input_features.combine_domain_into_x drops Frame,id,x1,y1,x2,y2 but retains bbox_conf before flattening the first34values.',
            'confidence_source': 'Actual original FasterRCNN bbox_score, including author carry-forward; no guessed values.',
            'input_resampling': 'none; original native frames', 'postprocess_fps': 30,
            'point_conversion': 'Rounded nonzero-run midpoint; original14-frame window-end alignment(+13).',
            'training_executed': False}}


def predict(include_resnet=False):
    import benchmark_hits_gru as gru
    default_weights = gru.WEIGHTS
    weights = [('gt_trained', default_weights), ('filtered2_trained', FILTERED)]
    for case in plan():
        folder = Path(case['output_folder'])
        pose = [('current_pose', Path(case['dense_path'])),
                ('author_pose', folder/'author_pose/dense_pose.json'),
                ('author_pose_full_image', folder/'author_pose_full_image/dense_pose.json')]
        balls = [('current_ball', Path(case['trajectory_path'])),
                 ('author_ball', folder/'author_ball/trajectory.json')]
        for pose_name, pose_path in pose:
            if not pose_path.exists():
                print('WAITING_AUTHOR_COURT_OR_POSE', case['id'], str(pose_path), flush=True)
                continue
            dense = read(pose_path)
            for ball_name, ball_path in balls:
                if not ball_path.exists():
                    continue
                ball = read(ball_path)
                assert len(dense['frames']) == len(ball['frames']) == case['frames']
                for weight_name, weight_path in weights:
                    if not weight_path.exists():
                        raise FileNotFoundError(weight_path)
                    for postprocess in ['naive', 'max']:
                        method = f'{pose_name}__{ball_name}__{weight_name}__{postprocess}'
                        output = folder/'predictions'/method
                        if (output/'prediction.json').exists():
                            continue
                        try:
                            gru.WEIGHTS = weight_path
                            result = gru.infer(dense, ball, postprocess=postprocess)
                        finally:
                            gru.WEIGHTS = default_weights
                        result['metadata'].update(model='kwyoke-GRU-'+weight_name,
                            input_condition={'pose': pose_name, 'ball': ball_name},
                            exact_author_raw_pipeline=pose_name == 'author_pose' and ball_name == 'author_ball',
                            weight_condition=weight_name,
                            inputs_sha256={str(pose_path): file_hash(pose_path), str(ball_path): file_hash(ball_path)},
                            geometry_provenance=case.get('court_provenance'),
                            source_frame_offset=case['source_frame_offset'])
                        save_prediction(output, result, dense, ball)
                        print('GRU_PREDICTED', case['id'], method, len(result['events']), flush=True)
                if pose_name.startswith('author_pose'):
                    for postprocess in ['naive', 'max']:
                        method = f'{pose_name}__{ball_name}__filtered2_legacy_layout__{postprocess}'
                        output = folder/'predictions'/method
                        if (output/'prediction.json').exists():
                            continue
                        result = filtered2_legacy(dense, ball, postprocess)
                        result['metadata'].update(input_condition={'pose': pose_name, 'ball': ball_name},
                            geometry_provenance=dense.get('geometry_provenance', case.get('court_provenance')),
                            geometry_condition=dense.get('geometry_condition', 'author_published_calibration'),
                            inputs_sha256={str(pose_path): file_hash(pose_path), str(ball_path): file_hash(ball_path)})
                        save_prediction(output, result, dense, ball)
                        print('GRU_LEGACY_PREDICTED', case['id'], method, len(result['events']), flush=True)
            if include_resnet:
                from reproduce_gru_author_resnet import infer
                for postprocess in ['naive', 'max']:
                    method = f'{pose_name}__resnet__{postprocess}'
                    output = folder/'predictions'/method
                    if (output/'prediction.json').exists():
                        continue
                    result = infer(dense, source_path=case['source_path'], postprocess=postprocess, device='mps')
                    save_prediction(output, result, dense)
                    print('RESNET_PREDICTED', case['id'], pose_name, postprocess, len(result['events']), flush=True)


def evaluate():
    from benchmark_hit_models import metrics
    rows, missing = [], []
    required = 'author_pose__author_ball__filtered2_legacy_layout__naive'
    for case in plan():
        folder = Path(case['output_folder'])
        if not (folder/'predictions'/required/'prediction.json').exists():
            alternative = folder/'predictions/author_pose_full_image__author_ball__filtered2_legacy_layout__naive/prediction.json'
            missing.append({'case': case['id'], 'method': required,
                            'reason': 'Physical court geometry unresolved; full-image ablation completed separately.' if alternative.exists() else 'Original pose/geometry or inference not yet complete.'})
        for output in sorted((folder/'predictions').glob('*/prediction.json')):
            result = read(output)
            events = result['events']
            values = {label: metrics(events, case['truth'], case['fps'], case['truth_scope'] == 'exhaustive', tolerance)
                      for label, tolerance in [('3frames', 3), ('6frames', 6),
                                                ('0.1s', .1*case['fps']), ('0.2s', .2*case['fps'])]}
            rows.append({'case': case['id'], 'domain': case['domain'], 'method': output.parent.name,
                         'truth_scope': case['truth_scope'], 'metrics': values,
                         'prediction_sha256': file_hash(output)})
    summary = {}
    for domain in sorted({row['domain'] for row in rows}):
        summary[domain] = {}
        for method in sorted({row['method'] for row in rows if row['domain'] == domain}):
            records = [row for row in rows if row['domain'] == domain and row['method'] == method]
            summary[domain][method] = {'cases': len(records), 'metrics': {}}
            complete = all(row['truth_scope'] == 'exhaustive' for row in records)
            for tolerance in ['3frames', '6frames', '0.1s', '0.2s']:
                matched = sum(row['metrics'][tolerance]['matched'] for row in records)
                n = sum(row['metrics'][tolerance]['annotations'] for row in records)
                p = sum(row['metrics'][tolerance]['predictions'] for row in records)
                value = {'annotations': n, 'predictions': p, 'matched': matched,
                         'known_positive_coverage': matched/n if n else None}
                if complete:
                    value.update(precision=matched/p if p else 0., recall=matched/n,
                                 f1=2*matched/(p+n), missed=n-matched, extra=p-matched)
                summary[domain][method]['metrics'][tolerance] = value
    atomic_json(OUT/'raw_video_result.json', {
        'evaluated_at': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': file_hash(OUT/'raw_video_plan.json'), 'summary': summary,
        'rows': rows, 'missing_original_pipeline_cases': missing,
        'scope': 'Small fixed raw-video replication pilot. Public contacts complete; personal excerpts contain only previously confirmed positive records. Author method assumes pre-cut rallies and manually supplied court geometry. No training or production deployment.'})
    print('RAW_EVALUATED', len(rows), 'missing_original_pipeline_cases', len(missing), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['tracknet', 'predict', 'evaluate'])
    parser.add_argument('--include-resnet', action='store_true')
    args = parser.parse_args()
    if args.stage == 'tracknet':
        tracknet()
    elif args.stage == 'predict':
        predict(args.include_resnet)
    else:
        evaluate()


if __name__ == '__main__':
    main()
