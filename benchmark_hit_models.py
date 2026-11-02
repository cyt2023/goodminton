"""Compare isolated hit detectors on frozen, frame-indexed recordings.

Only the scoring stage receives annotations. Partial personal annotations are
positive retrieval checks, never whole-video precision/recall ground truth.
This module does not update the website, user annotations, or model weights.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.optimize import linear_sum_assignment

from analysis_jobs import DATA, atomic_json, file_hash

ROOT = Path(__file__).resolve().parent
OUT = DATA / '找拍模型对照/2026-10-22'
VERSION = 'isolated-hit-comparison-v1'
MODELS = {
    'current_visual_rules': ('specialist_events', 'proposals'),
    'current_production_fusion': (None, None),
    'author_gru_dp': ('benchmark_hits_gru', 'infer'),
    'author_gru_naive': ('benchmark_hits_gru', 'infer'),
    'adapted_hitnet': ('benchmark_hits_hitnet', 'infer'),
    'adapted_hitnet_no_court': ('benchmark_hits_hitnet', 'infer'),
    'swingnet': ('benchmark_hits_swingnet', 'infer'),
}


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def cases():
    initial = read(OUT / 'data_manifest.json')['cases']
    if (OUT / 'protocol_extension.json').exists():
        extra = read(OUT / 'data_manifest_extension.json')['cases']
        initial += [{**case, 'evaluation_group': 'professional_extension'} for case in extra]
    return initial


def freeze_extension():
    target = OUT / 'protocol_extension.json'
    if target.exists():
        raise FileExistsError('Preserve the declared extension')
    manifest = OUT / 'data_manifest_extension.json'
    extra = read(manifest)
    atomic_json(target, {'declared_at': now(), 'manifest_sha256': file_hash(manifest),
        'cases': [case['id'] for case in extra['cases']],
        'scope': 'All five newly prepared professional rallies; existing tolerances and model defaults unchanged. Initial twelve-case results already inspected; no model fitting or score-based sample selection.',
        'data_counts': extra['counts'], 'training_executed': False})
    print('EXTENSION_FROZEN', target, flush=True)


def frozen_files():
    paths = list((ROOT / 'models/bst').glob('*.pt'))
    paths += list((ROOT / 'models/hit_gru').glob('*'))
    paths += list((ROOT / 'models/contact_timing').glob('*'))
    for job in ['J_fe1e058fc75e475d9c030caced6ec648',
                'J_e23c1dd9297d4bfca7c91d1d95065b68']:
        paths.append(DATA / '击球标注' / job / 'annotations.json')
    return {str(p): file_hash(p) for p in paths if p.is_file()}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        raise FileExistsError('Preserve the declared protocol')
    manifest = OUT / 'data_manifest.json'
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    models = {}
    for name, (module, _) in MODELS.items():
        code = ROOT / (module + '.py') if module else ROOT / 'hit_refinement.py'
        models[name] = {'entrypoint': str(code),
                        'status_at_declaration': 'code_present' if code.exists() else 'asset_adaptation_in_progress'}
    atomic_json(path, {
        'version': VERSION, 'declared_at': now(),
        'manifest_sha256': file_hash(manifest),
        'methods_planned': models,
        'selection': 'All available cases declared in data_manifest; no selection by new model performance.',
        'primary_tolerance_seconds': .1, 'secondary_tolerance_seconds': .2,
        'also_report_exactness_frames': [0, 1, 3],
        'matching': 'Maximum-cardinality one-to-one temporal assignment; among equal cardinality minimize total absolute frame error; actor ignored for temporal matching.',
        'clock': 'Original decoded frame index / per-case fixed cadence; native PTS not used for matching where invalid.',
        'partial_annotation_policy': 'Only coverage of known confirmed positive contacts and localization errors. Unmatched predictions remain unverified; no precision, global recall, F1, or false-positive claim.',
        'edge_policy': 'Keep all annotated contacts, including frames without a complete model lookback window.',
        'parameter_policy': 'Published checkpoint/default postprocessing; fixed interval-to-point midpoint where a method only yields intervals. No fit or test-score parameter selection.',
        'scope': 'Development comparison on previously inspected recordings; not a new independent generalization benchmark. Shared frontend comparisons must be distinguished from author full-pipeline reproduction.',
        'protected_files': frozen_files(), 'training_executed': False,
    })
    print('PROTOCOL_FROZEN', path, flush=True)


def check():
    protocol = read(OUT / 'protocol.json')
    manifest_hash = protocol['manifest_sha256']
    addendum = OUT / 'audio_manifest_addendum.json'
    if addendum.exists():
        amended = read(addendum)
        assert amended['original_manifest_sha256'] == manifest_hash
        assert file_hash(OUT / 'data_manifest.before_audio.json') == manifest_hash
        manifest_hash = amended['updated_manifest_sha256']
    assert file_hash(OUT / 'data_manifest.json') == manifest_hash, 'Data manifest changed'
    for p, sha in protocol['protected_files'].items():
        assert file_hash(Path(p)) == sha, 'Protected file changed: ' + p
    extension = OUT / 'protocol_extension.json'
    if extension.exists():
        assert file_hash(OUT / 'data_manifest_extension.json') == read(extension)['manifest_sha256']
    for case in cases():
        for p, sha in case.get('input_sha256', {}).items():
            assert file_hash(Path(p)) == sha, 'Declared input changed: ' + p
    return protocol


def production(dense, trajectory, case, folder):
    import hit_refinement
    from specialist_events import proposals
    from source_frame_media import ffmpeg_executable
    raw = proposals(dense, trajectory)
    # Scope all mutable caches and artifacts to this new experiment directory.
    sandbox = folder / 'production_support'
    job = dense['id']
    support = sandbox / '专项分析' / job
    support.mkdir(parents=True, exist_ok=True)
    atomic_json(support / 'dense_pose.json', dense)
    atomic_json(support / 'trajectory.json', trajectory)
    config = {'pose_id': job, 'source_sha256': dense['source_sha256'],
              'source_file': case.get('audio_path', case['source_path']), 'ffmpeg': ffmpeg_executable()}
    previous = hit_refinement.DATA
    try:
        hit_refinement.DATA = sandbox
        refinement = hit_refinement.prepare_inputs(config, dense, trajectory, raw)
    finally:
        hit_refinement.DATA = previous
    result = []
    for original in raw:
        event = dict(original)
        suggestion = refinement['suggestions'].get(event['id'])
        if suggestion and suggestion['applied']:
            event.update(frame_index=suggestion['suggested_frame'],
                         peak_time=suggestion['suggested_time'])
        result.append(event)
    result.extend(refinement['supplemental_candidates'])
    return {'events': result, 'metadata': {
        'author_model': False, 'implementation': 'Current production visual+GRU+audio gates and optional eligible local ridge, with fresh experiment cache.',
        'refinement_summary': refinement['summary'],
        'timing_model': refinement.get('contact_timing'),
        'audio': refinement['audio'],
        'audio_source': config['source_file'],
    }}


def run_model(name, case, overwrite_failure=False):
    folder = OUT / 'predictions' / case['id'] / name
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'prediction.json'
    if target.exists():
        print('REUSE', name, case['id'], flush=True)
        return
    failure = folder / 'failure.json'
    if failure.exists() and not overwrite_failure:
        print('FAILED_PREVIOUSLY', name, case['id'], flush=True)
        return
    started = time.perf_counter()
    try:
        dense = read(case['dense_path'])
        trajectory = read(case['trajectory_path'])
        assert dense['source_sha256'] == trajectory['source_sha256']
        assert len(dense['frames']) == len(trajectory['frames'])
        # Model dispatch receives media/features only. The truth list is never
        # passed into an imported detector implementation.
        if name == 'current_production_fusion':
            media = {key: case[key] for key in ['source_path', 'audio_path'] if key in case}
            output = production(dense, trajectory, media, folder)
        else:
            module_name, function = MODELS[name]
            module = importlib.import_module(module_name)
            infer = getattr(module, function)
            kwargs = {}
            if name.startswith('author_gru_'):
                kwargs['postprocess'] = 'max' if name.endswith('_dp') else 'naive'
            if name == 'swingnet':
                kwargs['source_path'] = case['source_path']
                kwargs['output_folder'] = str(folder)
            if name.startswith('adapted_hitnet'):
                if name == 'adapted_hitnet_no_court':
                    kwargs['court_corners'] = [[0., 0.]] * 4
                else:
                    geometry = read(OUT / 'court_geometry.json')['cases'].get(case['id'])
                    if not geometry and case.get('geometry_file'):
                        candidate_file = read(case['geometry_file'])
                        for item in candidate_file.get('items', []):
                            if item['id'] == case['id'] and item.get('corners_px'):
                                points = item['corners_px']
                                geometry = {'court_corners': [points[k] for k in ['top_left', 'top_right', 'bottom_left', 'bottom_right']],
                                            'shuttleset_match_id': None}
                    if not geometry or geometry.get('court_corners') is None:
                        raise ValueError('No verified court geometry; full-court condition unavailable. See separate no-court ablation.')
                    kwargs['court_corners'] = geometry['court_corners']
                    kwargs['match_id'] = geometry.get('shuttleset_match_id')
            output = infer(dense, trajectory, **kwargs)
            if name == 'adapted_hitnet_no_court':
                output['metadata'].update(court_input='All-zero court-input ablation; not a faithful full-input reproduction.',
                                           input_condition='missing_court_ablation')
        if isinstance(output, list):
            output = {'events': output, 'metadata': {}}
        if not isinstance(output, dict) or not isinstance(output.get('events'), list):
            raise TypeError('Detector output must be a dict with events list')
        arrays = {key: value for key, value in output.items() if isinstance(value, np.ndarray)}
        if arrays:
            array_file = folder / 'continuous_outputs.npz'
            np.savez_compressed(array_file, **arrays)
            output = {key: value for key, value in output.items() if key not in arrays}
            output['array_outputs'] = {'path': str(array_file), 'sha256': file_hash(array_file),
                                      'shapes': {key: list(value.shape) for key, value in arrays.items()}}
        n = len(dense['frames'])
        for event in output['events']:
            frame = event['frame_index']
            if not isinstance(frame, (int, np.integer)) or not 0 <= frame < n:
                raise ValueError(f'Invalid original frame prediction: {frame} / {n}')
            event['frame_index'] = int(frame)
        output.update(model=name, case_id=case['id'], created_at=now(),
            source_sha256=dense['source_sha256'],
            inputs={'dense_path': case['dense_path'], 'dense_sha256': file_hash(Path(case['dense_path'])),
                    'trajectory_path': case['trajectory_path'], 'trajectory_sha256': file_hash(Path(case['trajectory_path']))},
            runtime_seconds=time.perf_counter()-started,
            truth_received_by_detector=False,
            code_sha256=file_hash(ROOT / ((MODELS[name][0] or 'hit_refinement') + '.py')))
        atomic_json(target, output)
        print('PREDICTED', name, case['id'], len(output['events']), flush=True)
    except Exception as error:
        atomic_json(failure, {'model': name, 'case_id': case['id'],
            'error': repr(error), 'traceback': traceback.format_exc(),
            'failed_at': now(), 'runtime_seconds': time.perf_counter()-started})
        print('FAILED', name, case['id'], repr(error), flush=True)


def match_events(events, truth, tolerance_frames, fps):
    m, n = len(events), len(truth)
    pairs = []
    if m and n:
        delta = np.abs(np.asarray([e['frame_index'] for e in events])[:, None] -
                       np.asarray([g['frame_index'] for g in truth])[None, :])
        # One additional valid match dominates any possible total error change.
        penalty = (m+n+1) * (tolerance_frames+1)
        cost = np.zeros((m+n, m+n))
        cost[:m, :n] = np.where(delta <= tolerance_frames+1e-9, delta, penalty*4)
        cost[:m, n:] = penalty
        cost[m:, :n] = penalty
        ii, jj = linear_sum_assignment(cost)
        pairs = [(int(i), int(j)) for i, j in zip(ii, jj)
                 if i < m and j < n and cost[i, j] < penalty]
    details = []
    for i, j in pairs:
        event, gt = events[i], truth[j]
        delta = event['frame_index'] - gt['frame_index']
        actor = event.get('actor', event.get('player'))
        details.append({'prediction_index': i, 'truth_index': j,
            'predicted_frame': event['frame_index'], 'truth_frame': gt['frame_index'],
            'signed_error_frames': delta, 'signed_error_seconds': delta/fps,
            'predicted_player': actor, 'truth_player': gt.get('player'),
            'actor_correct': actor == gt['player'] if actor and gt.get('player') else None,
            'annotation_id': gt.get('annotation_id')})
    used_i, used_j = {i for i, j in pairs}, {j for i, j in pairs}
    return {'matches': details,
        'unmatched_prediction_indices': [i for i in range(m) if i not in used_i],
        'unmatched_truth_indices': [j for j in range(n) if j not in used_j]}


def metrics(events, truth, fps, exhaustive, tolerance_frames):
    result = match_events(events, truth, tolerance_frames, fps)
    matched = len(result['matches'])
    result.update(tolerance_frames=tolerance_frames,
        tolerance_seconds=tolerance_frames/fps,
        annotations=len(truth), predictions=len(events), matched=matched,
        known_positive_coverage=matched/len(truth) if truth else None)
    errors = [abs(p['signed_error_seconds']) for p in result['matches']]
    result.update(median_matched_error_seconds=float(np.median(errors)) if errors else None,
                  mean_matched_error_seconds=float(np.mean(errors)) if errors else None)
    if exhaustive:
        result.update(precision=matched/len(events) if events else 0.,
            recall=matched/len(truth) if truth else None,
            f1=2*matched/(len(events)+len(truth)) if len(events)+len(truth) else 0.,
            missed=len(truth)-matched, extra=len(events)-matched)
    else:
        result['scope'] = 'Partial positive labels; unmatched predictions are unverified, not false positives. known_positive_coverage is not whole-recording recall.'
    return result


def evaluate():
    protocol = check()
    all_cases = cases()
    rows, missing = [], []
    for case in all_cases:
        for model in MODELS:
            path = OUT / 'predictions' / case['id'] / model / 'prediction.json'
            if not path.exists():
                missing.append({'case': case['id'], 'model': model})
                continue
            prediction = read(path)
            events, truth = prediction['events'], case.get('truth', [])
            exhaustive = case['truth_scope'] == 'exhaustive'
            fps = case['fps']
            values = {}
            if case['truth_scope'] == 'inference_only':
                rows.append({'case': case['id'], 'model': model, 'domain': case.get('evaluation_group', case['domain']),
                    'truth_scope': case['truth_scope'], 'status': 'inference_only_no_truth',
                    'predictions': len(events), 'prediction_sha256': file_hash(path),
                    'runtime_seconds': prediction['runtime_seconds'], 'metrics': {}})
                continue
            for label, tol in [('0.1s', .1*fps), ('0.2s', .2*fps),
                               ('0frames', 0), ('1frame', 1), ('3frames', 3)]:
                values[label] = metrics(events, truth, fps, exhaustive, tol)
            row = {'case': case['id'], 'model': model, 'domain': case.get('evaluation_group', case['domain']),
                'truth_scope': case['truth_scope'], 'prediction_sha256': file_hash(path),
                'runtime_seconds': prediction['runtime_seconds'], 'metrics': values}
            rows.append(row)
    groups = {}
    for domain in sorted({r['domain'] for r in rows}):
        for model in MODELS:
            subset = [r for r in rows if r['domain'] == domain and r['model'] == model]
            if not subset:
                continue
            groups.setdefault(domain, {})[model] = summary = {'cases': len(subset), 'metrics': {}}
            exhaustive = all(r['truth_scope'] == 'exhaustive' for r in subset)
            if all(r['truth_scope'] == 'inference_only' for r in subset):
                summary.update(status='inference_only_no_truth',
                               predictions=sum(r['predictions'] for r in subset))
                continue
            for tolerance in ['0.1s', '0.2s', '0frames', '1frame', '3frames']:
                vs = [r['metrics'][tolerance] for r in subset]
                matched = sum(v['matched'] for v in vs)
                n = sum(v['annotations'] for v in vs)
                p = sum(v['predictions'] for v in vs)
                errors = [abs(pair['signed_error_seconds']) for v in vs for pair in v['matches']]
                value = {'annotations': n, 'predictions': p, 'matched': matched,
                         'known_positive_coverage': matched/n if n else None,
                         'median_matched_error_seconds': float(np.median(errors)) if errors else None}
                if exhaustive:
                    value.update(precision=matched/p if p else 0., recall=matched/n if n else None,
                                 f1=2*matched/(p+n) if p+n else 0., missed=n-matched, extra=p-matched)
                summary['metrics'][tolerance] = value
    result = {'version': VERSION, 'evaluated_at': now(),
        'protocol_sha256': file_hash(OUT / 'protocol.json'),
        'data_manifest_sha256': file_hash(OUT / 'data_manifest.json'),
        'extension_manifest_sha256': file_hash(OUT / 'data_manifest_extension.json') if (OUT / 'protocol_extension.json').exists() else None,
        'extension_protocol_sha256': file_hash(OUT / 'protocol_extension.json') if (OUT / 'protocol_extension.json').exists() else None,
        'groups': groups, 'rows': rows, 'missing_predictions': missing,
        'protected_files_unchanged': True, 'production_modified': False,
        'scope': protocol['scope']}
    target = OUT / 'evaluation.json'
    if target.exists():
        old = file_hash(target)
        prior = OUT / 'evaluation_snapshots' / f'{old}.json'
        if not prior.exists():
            atomic_json(prior, read(target))
    atomic_json(target, result)
    print(json.dumps(groups, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'freeze-extension', 'run', 'evaluate'])
    parser.add_argument('--model', choices=list(MODELS))
    parser.add_argument('--case', action='append')
    parser.add_argument('--retry-failures', action='store_true')
    args = parser.parse_args()
    if args.stage == 'freeze':
        return freeze()
    check()
    if args.stage == 'freeze-extension':
        return freeze_extension()
    if args.stage == 'evaluate':
        return evaluate()
    if not args.model:
        parser.error('--model is required for run')
    for case in cases():
        if args.case and case['id'] not in args.case:
            continue
        run_model(args.model, case, args.retry_failures)
    check()


if __name__ == '__main__':
    main()
