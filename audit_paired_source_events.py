"""Isolated unchanged production hit proposals/refinement on the aligned rally.

Freeze the evaluation before loading detector output. No BST classification,
training, production database, annotations, or model changes are performed.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from analysis_jobs import atomic_json, file_hash

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / '03_数据/击球标注/根因诊断/2026-10-15_同源前端对照'
PROTOCOL = OUT / 'automatic_events_protocol.json'
DEST = OUT / 'automatic_events'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def frozen_inputs():
    rows = list(csv.DictReader((OUT / 'selected_rally_samples.csv').open()))
    if [r['sample_id'] for r in rows] != [f'35_1_5_{i}' for i in range(1, 21)]:
        raise ValueError('Unexpected frozen author sample order')
    manifest = read(OUT / 'continuous_rally/clips_manifest.json')
    if not manifest['all_pixels_verified'] or len(manifest['clips']) != 1:
        raise ValueError('Continuous source pixels not verified')
    clip = manifest['clips'][0]
    if clip['frames'] != 603 or clip['global_frame_start'] != 16756:
        raise ValueError('Unexpected continuous frame range')
    frame_map_path = OUT / 'continuous_rally/frame_map.json'
    if file_hash(frame_map_path) != clip['frame_map_sha256']:
        raise ValueError('Continuous frame map hash mismatch')
    frame_map = read(frame_map_path)
    if len(frame_map) != 603 or any(r['clip_index'] != i or r['global_frame'] != 16756 + i
                                   for i, r in enumerate(frame_map)):
        raise ValueError('Continuous frame map is not exact and consecutive')
    if file_hash(Path(clip['source_file'])) != clip['source_sha256']:
        raise ValueError('Continuous source changed')
    truth = [{'sample_id': r['sample_id'], 'global_frame': int(r['hit_frame_num']),
              'local_frame': int(r['hit_frame_num']) - 16756, 'court_side': r['court_side'],
              'author_window_global_start': int(r['author_start_frame_at_nominal_30fps']),
              'author_window_frames': int(r['actual_author_npy_frame_count'])}
             for r in rows]
    hashes = {str(p): file_hash(p) for p in [OUT / 'selected_rally_samples.csv',
              OUT / 'continuous_rally/clips_manifest.json', frame_map_path]}
    return clip, frame_map, truth, hashes


def freeze():
    if PROTOCOL.exists():
        raise FileExistsError('Evaluation protocol already frozen; preserve it')
    clip, _, truth, hashes = frozen_inputs()
    audio_path = OUT / 'continuous_rally/source_aligned_audio.wav'
    audio_result_path = OUT / 'continuous_rally/audio_download_result.json'
    audio_protocol_path = OUT / 'continuous_rally/audio_download_protocol.json'
    audio_verification = read(audio_result_path)
    if (audio_verification['sha256'] != file_hash(audio_path) or
            audio_verification['samples'] != 321600 or audio_verification['sample_rate'] != 16000 or
            audio_verification['protocol']['audio_lag_or_event_times_used_for_crop']):
        raise ValueError('Unexpected original audio verification')
    hashes.update({str(p): file_hash(p) for p in [audio_path, audio_result_path, audio_protocol_path]})
    code_paths = [Path(__file__), *[ROOT / n for n in ['specialist_events.py',
                  'specialist_pipeline.py', 'hit_refinement.py', 'hit_gru.py',
                  'stroke_features.py', 'contact_timing.py']]]
    weights = [ROOT / 'models/hit_gru/keras_tensors.npz',
               ROOT / 'models/contact_timing/contact_ridge.npz',
               ROOT / 'models/contact_timing/manifest.json']
    protocol = {'version': 'aligned_rally_automatic_events_v1', 'declared_at_utc': utc(),
        'declaration': 'Frozen before inspecting any automatic event candidates or their evaluation.',
        'source': clip, 'input_hashes': hashes, 'truth': truth,
        'code_hashes': {str(p): file_hash(p) for p in code_paths},
        'weight_or_manifest_hashes': {str(p): file_hash(p) for p in weights if p.is_file()},
        'conditions': ['raw_visual_proposals', 'production_refined_no_audio',
                       'production_refined_original_aligned_audio'],
        'detector': 'Unchanged specialist_events.proposals(dense,trajectory), all candidates, no score filtering.',
        'refinement': 'Unchanged hit_refinement.prepare_inputs and specialist_pipeline candidate assembly, both sides, no done_ids.',
        'matching': {'max_absolute_center_error_seconds': .2, 'fps': 30,
            'maximum_global_frame_difference': 6,
            'method': 'Hungarian maximum-cardinality within inclusive six-frame gate, then minimum total absolute error.',
            'actor_used_for_matching': False, 'one_to_one': True,
            'center_basis': 'Original global frame from immutable continuous frame_map, never nearest timestamp.',
            'precision': 'matched candidates / all candidates; zero if no candidates',
            'recall': 'matched author contacts / 20',
            'actor_evaluation': 'Separate side correctness on temporally matched pairs; whole-rally median ankle-y slot mapping, no GT-based remapping.',
            'no_boundary_exclusion': True},
        'window_analysis': 'For every original author window and every automatic window, enumerate all20 GT contacts and same-actor contacts within inclusive decoded frame bounds; do not filter contaminated windows.',
        'original_audio_secondary': {
            'path': str(audio_path), 'sha256': file_hash(audio_path),
            'verification_path': str(audio_result_path),
            'verification_sha256': file_hash(audio_result_path),
            'source_url': 'https://www.youtube.com/watch?v=XmJ-OdVFQtk',
            'native_source_start_seconds': 16756 / 30,
            'native_source_end_seconds_exclusive': 17359 / 30,
            'duration_seconds': 603 / 30,
            'alignment': 'Extract by native source presentation timestamps with no GT-event-based shift. WAV time0 corresponds to global source frame16756/30; declared crop quantization bound is one16kHz sample. Actual physical audio/video lag remains estimated by unchanged production refinement.',
            'adaptation': 'In this audit process only, wrap original hit_refinement.audio_onsets to pass the separate aligned WAV source_file; preserve visual source SHA and all detector gates/weights.',
            'cache_isolation': 'Each audio condition uses a distinct task-local DATA root, preventing production cache reuse across audio conditions.'},
        'unavoidable_adaptations': [
            'Already computed continuous603-frame local frontend on a lossless FFV1 replay of current aligned source; author historical MP4 encoding unavailable.',
            'Task-local hit_refinement.DATA redirect changes output/cache location only; no production data read/write or registration.',
            'AVI has no soundtrack: report visual proposals and normal unavailable-audio fallback separately from original-audio secondary.',
            'This is event proposal evaluation before BST target/coverage gates and VLM style review; not complete end-to-end classification.',
            'Author contact frames are dataset annotations with finite temporal precision; a single correlated rally is not general performance.'],
        'no_training': True, 'no_candidate_classification': True,
        'parameters_tuned_on_event_results': False}
    atomic_json(PROTOCOL, protocol)
    print(json.dumps({'status': 'protocol_frozen', 'path': str(PROTOCOL),
                      'sha256': file_hash(PROTOCOL)}, ensure_ascii=False), flush=True)


def check_protocol():
    protocol = read(PROTOCOL)
    digest = file_hash(PROTOCOL)
    for path, expected in {**protocol['input_hashes'], **protocol['code_hashes'],
                           **protocol['weight_or_manifest_hashes']}.items():
        if file_hash(Path(path)) != expected:
            raise ValueError('Frozen dependency changed: ' + path)
    return protocol, digest


def load_frontends(protocol):
    clip, frame_map, _, _ = frozen_inputs()
    folder = OUT / 'continuous_rally'
    paths = {'dense': folder / 'pose/dense_pose.json',
             'trajectory': folder / 'tracknet/trajectory.json'}
    dense, trajectory = [read(paths[k]) for k in ['dense', 'trajectory']]
    for data in [dense, trajectory]:
        if data['source_sha256'] != clip['source_sha256'] or len(data['frames']) != 603:
            raise ValueError('Continuous frontend source/count differs')
        if [data['source_width'], data['source_height']] != [1280, 720]:
            raise ValueError('Continuous frontend dimensions differ')
        if any(f['frame_index'] != i for i, f in enumerate(data['frames'])):
            raise ValueError('Continuous frontend frame indices differ')
    if not np.isclose(dense['sample_fps'], 30):
        raise ValueError('Continuous pose FPS differs')
    if any(f['global_frame'] != m['global_frame'] for f, m in zip(dense['frames'], frame_map)):
        raise ValueError('Continuous dense global frames differ')
    return dense, trajectory, frame_map, paths


def actor_order(dense):
    medians = {}
    for side in ['near', 'far']:
        vals = [np.mean([p['keypoints'][j]['y'] for j in [15, 16]])
                for f in dense['frames'] if (p := f['players'].get(side)) is not None]
        medians[side] = float(np.median(vals)) if vals else None
    known = all(v is not None and np.isfinite(v) for v in medians.values())
    order = sorted(medians, key=medians.get) if known else None
    return dict(zip(order, ['Top', 'Bottom'])) if order else {}, medians


def window_contacts(lo, hi, side, truth):
    contained = [g for g in truth if lo <= g['global_frame'] <= hi]
    same = [g for g in contained if g['court_side'] == side] if side else []
    return {'global_start_inclusive': lo, 'global_end_inclusive': hi,
            'all_contact_ids': [g['sample_id'] for g in contained],
            'same_actor_contact_ids': [g['sample_id'] for g in same],
            'all_contact_count': len(contained), 'same_actor_contact_count': len(same),
            'multiple_same_actor_contacts': len(same) > 1}


def evaluate(events, dense, frame_map, truth):
    mapping, medians = actor_order(dense)
    global_frames = np.asarray([frame_map[e['frame_index']]['global_frame'] for e in events])
    gt_frames = np.asarray([g['global_frame'] for g in truth])
    m, n = len(events), len(truth)
    matched = []
    if m:
        # Invalid edges cost more than unmatched dummy assignments. One fewer
        # match costs >=2*(m+n+1), exceeding every feasible total frame error.
        distance = abs(global_frames[:, None] - gt_frames[None, :]) / 30
        penalty = m + n + 1.
        cost = np.zeros((m+n, m+n), dtype=float)
        cost[:m, :n] = np.where(abs(global_frames[:, None] - gt_frames[None, :]) <= 6,
                               distance, penalty * 4)
        cost[:m, n:] = penalty
        cost[m:, :n] = penalty
        ri, ci = linear_sum_assignment(cost)
        matched = [(int(i), int(j)) for i, j in zip(ri, ci) if i < m and j < n and cost[i, j] < penalty]
    used_events, used_truth = {i for i, _ in matched}, {j for _, j in matched}
    matches = []
    for i, j in matched:
        event, gt = events[i], truth[j]
        side = mapping.get(event['player'])
        matches.append({'candidate_id': event['id'], 'sample_id': gt['sample_id'],
            'candidate_global_frame': int(global_frames[i]), 'author_global_frame': gt['global_frame'],
            'signed_error_frames': int(global_frames[i] - gt['global_frame']),
            'absolute_error_seconds': abs(int(global_frames[i] - gt['global_frame'])) / 30,
            'candidate_slot': event['player'], 'candidate_court_side': side,
            'author_court_side': gt['court_side'], 'actor_correct': side == gt['court_side'] if side else None})
    matches.sort(key=lambda r: r['author_global_frame'])
    actor_known = [r for r in matches if r['actor_correct'] is not None]
    errors = [r['absolute_error_seconds'] for r in matches]
    windows = [{'candidate_id': e['id'], **window_contacts(
        frame_map[e['start_frame']]['global_frame'], frame_map[e['end_frame']]['global_frame'],
        mapping.get(e['player']), truth)} for e in events]
    return {'candidate_count': m, 'author_contact_count': n, 'matched_count': len(matched),
        'precision': len(matched) / m if m else 0., 'recall': len(matched) / n,
        'extra_candidate_count': m - len(matched), 'missed_contact_count': n - len(matched),
        'actor_correct_among_matches': sum(r['actor_correct'] for r in actor_known),
        'actor_evaluable_matches': len(actor_known),
        'actor_accuracy_among_matches': sum(r['actor_correct'] for r in actor_known) / len(actor_known) if actor_known else None,
        'actor_and_time_recall': sum(r['actor_correct'] for r in actor_known) / n,
        'median_absolute_center_error_seconds': float(np.median(errors)) if errors else None,
        'max_absolute_center_error_seconds': max(errors) if errors else None,
        'slot_to_court_side': mapping, 'whole_rally_median_ankle_y': medians,
        'matches': matches, 'missed_contacts': [truth[j] for j in range(n) if j not in used_truth],
        'extra_candidates': [events[i] for i in range(m) if i not in used_events],
        'automatic_window_contacts': windows,
        'automatic_windows_with_multiple_same_actor_contacts': sum(w['multiple_same_actor_contacts'] for w in windows)}


def refined_candidates(dense, events, refinement):
    # Same assembly as specialist_pipeline.main(), without later BST/VLM stages.
    result = []
    for original in events:
        candidate = dict(original)
        suggestion = refinement['suggestions'].get(original['id'])
        if suggestion and suggestion['applied']:
            candidate.update(original_peak_time=original['peak_time'], original_frame_index=original['frame_index'],
                peak_time=suggestion['suggested_time'], frame_index=suggestion['suggested_frame'],
                center_suggestion=suggestion, contact_confirmed=False)
        result.append(candidate)
    for extra in refinement['supplemental_candidates']:
        times = np.asarray([f['source_time'] for f in dense['frames']])
        lo = int(np.searchsorted(times, max(0., extra['peak_time'] - 1.)))
        hi = min(len(times)-1, int(np.searchsorted(times, extra['peak_time'] + 1.)))
        result.append({**extra, 'start_frame': lo, 'end_frame': hi, 'start': float(times[lo]),
            'end': min(dense['end'], float(times[hi] + np.median(np.diff(times))))})
    return result


def run(audio=False):
    protocol, protocol_hash = check_protocol()
    stage = 'original_audio' if audio else 'no_audio'
    result_path = DEST / (stage + '_result.json')
    if result_path.exists():
        raise FileExistsError('Preserve completed frozen event evaluation: ' + str(result_path))
    started = time.perf_counter()
    dense, trajectory, frame_map, paths = load_frontends(protocol)
    from specialist_events import proposals
    # Source content/label inputs are separate: proposals receives no author GT.
    events = proposals(dense, trajectory)
    DEST.mkdir(exist_ok=True)
    raw_path = DEST / 'raw_visual_candidates.json'
    raw_record = {'source_sha256': dense['source_sha256'], 'candidates': events}
    if raw_path.exists():
        if read(raw_path) != raw_record:
            raise ValueError('Raw production proposals differ across fixed runs')
    else:
        atomic_json(raw_path, raw_record)
    import hit_refinement
    import imageio_ffmpeg
    # All path-dependent production cache/output operations remain task-local.
    isolated_data = DEST / ('runtime_' + stage)
    hit_refinement.DATA = isolated_data
    if Path(dense['id']).name != dense['id']:
        raise ValueError('Unsafe isolated pose ID')
    runtime_folder = isolated_data / '专项分析' / dense['id']
    runtime_folder.mkdir(parents=True, exist_ok=True)
    for key, name in [('dense', 'dense_pose.json'), ('trajectory', 'trajectory.json')]:
        shutil.copyfile(paths[key], runtime_folder / name)
    config = {'pose_id': dense['id'], 'source_file': protocol['source']['source_file'],
              'source_sha256': dense['source_sha256'], 'ffmpeg': imageio_ffmpeg.get_ffmpeg_exe()}
    audio_metadata = None
    original_audio = hit_refinement.audio_onsets
    if audio:
        audio_path = Path(protocol['original_audio_secondary']['path'])
        verification_path = Path(protocol['original_audio_secondary']['verification_path'])
        if not audio_path.is_file() or not verification_path.is_file():
            raise FileNotFoundError('Aligned original audio and root verification are required')
        verification = read(verification_path)
        if (verification['sha256'] != file_hash(audio_path) or
                verification['protocol']['audio_lag_or_event_times_used_for_crop'] or
                verification['samples'] != 321600 or verification['sample_rate'] != 16000):
            raise ValueError('Original audio alignment/hash not verified')
        audio_metadata = {'path': str(audio_path), 'sha256': file_hash(audio_path),
                          'verification': verification, 'verification_sha256': file_hash(verification_path)}
        hit_refinement.audio_onsets = lambda cfg: original_audio({**cfg, 'source_file': str(audio_path)})
    try:
        refinement = hit_refinement.prepare_inputs(config, dense, trajectory, events)
    finally:
        hit_refinement.audio_onsets = original_audio
    final_events = refined_candidates(dense, events, refinement)
    candidate_path = DEST / (stage + '_production_candidates.json')
    atomic_json(candidate_path, {'source_sha256': dense['source_sha256'], 'candidates': final_events,
                                'audio': refinement['audio'], 'summary': refinement['summary']})
    truth = protocol['truth']
    author_windows = [{'sample_id': g['sample_id'], **window_contacts(
        g['author_window_global_start'], g['author_window_global_start'] + g['author_window_frames'] - 1,
        g['court_side'], truth)} for g in truth]
    result = {'created_at_utc': utc(), 'condition': stage, 'protocol_sha256': protocol_hash,
        'frontend_sha256': {k: file_hash(p) for k, p in paths.items()},
        'raw_visual': evaluate(events, dense, frame_map, truth),
        'production_refined': evaluate(final_events, dense, frame_map, truth),
        'raw_candidates_sha256': file_hash(raw_path), 'refined_candidates_sha256': file_hash(candidate_path),
        'refinement_sha256': file_hash(runtime_folder / 'hit_refinement.json'),
        'audio_status': refinement['audio'], 'original_audio_secondary': audio_metadata,
        'author_window_contacts': author_windows,
        'author_windows_with_multiple_same_actor_contacts': sum(w['multiple_same_actor_contacts'] for w in author_windows),
        'runtime_seconds': time.perf_counter() - started,
        'classification_run': False, 'production_data_or_weights_changed': False,
        'scope': 'One aligned author-val rally; candidate detection before BST/target/input/VLM gates; no general performance claim.'}
    if file_hash(PROTOCOL) != protocol_hash:
        raise ValueError('Protocol changed during evaluation')
    atomic_json(result_path, result)
    print(json.dumps({'condition': stage, 'result': str(result_path),
        'raw_visual': {k: result['raw_visual'][k] for k in ['candidate_count', 'matched_count', 'precision', 'recall']},
        'production_refined': {k: result['production_refined'][k] for k in ['candidate_count', 'matched_count', 'precision', 'recall']},
        'audio_status': refinement['audio']['status']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--run-main', action='store_true')
    mode.add_argument('--run-original-audio', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        run(audio=args.run_original_audio)
