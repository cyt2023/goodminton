"""Explain fixed missed hit proposals without changing thresholds or inferring models."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import find_peaks, peak_prominences

from analysis_jobs import atomic_json, file_hash
import specialist_events

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / '03_数据/击球标注/根因诊断/2026-10-15_同源前端对照'
DEST = OUT / 'automatic_events'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def frame_guards(dense, trajectory, side, index):
    """Same ordered short-circuit guards and arithmetic as proposals' frame loop."""
    frames, ball_frames = dense['frames'], trajectory['frames']
    ball = np.array([b['raw_xy'] if b['raw_visible'] else [np.nan, np.nan] for b in ball_frames], dtype=float)
    i = index
    person = frames[i]['players'].get(side)
    record = {'frame_index': i, 'global_frame': frames[i]['global_frame'], 'player': side,
              'first_failure': None, 'score': 0., 'metrics': {},
              'raw_visible_context': {str(j-i): bool(ball_frames[j]['raw_visible'])
                                      for j in [i-4, i-3, i, i+3, i+4] if 0 <= j < len(frames)}}
    def fail(reason):
        record['first_failure'] = reason
        return record
    if i < 4 or i >= len(frames)-4:
        return fail('outside_production_scan_range')
    if person is None or not np.isfinite(ball[i]).all():
        return fail('player_missing' if person is None else 'raw_ball_missing_at_frame')
    joints = person['keypoints']
    record['metrics']['joint_scores'] = {str(j): joints[j]['score'] for j in [5, 6, 11, 12, 9, 10]}
    if any(joints[j]['score'] < .4 for j in [5, 6, 11, 12]):
        return fail('shoulder_or_hip_score_below_0.4')
    pts = np.asarray([[j['x'], j['y']] for j in joints])
    torso = np.linalg.norm((pts[5] + pts[6] - pts[11] - pts[12]) / 2)
    record['metrics']['torso_px'] = float(torso)
    if torso < 12:
        return fail('torso_below_12px')
    wrists = [j for j in [9, 10] if joints[j]['score'] >= .4]
    if not wrists:
        return fail('both_wrists_score_below_0.4')
    distances = {j: float(np.linalg.norm(ball[i] - pts[j]) / torso) for j in wrists}
    wrist = min(distances, key=distances.get)
    distance = distances[wrist]
    record['metrics'].update(wrist_index=wrist, wrist_distance_torso_units=distance,
                             wrist_distances={str(k): v for k, v in distances.items()})
    if distance > 1.7:
        return fail('wrist_distance_above_1.7_torso_units')
    before = next((j for j in range(i-3, i-5, -1) if np.isfinite(ball[j]).all()), None)
    after = next((j for j in range(i+3, i+5) if np.isfinite(ball[j]).all()), None)
    record['metrics'].update(before_frame=before, after_frame=after)
    if before is None or after is None:
        return fail('raw_ball_missing_before_and_after' if before is None and after is None else
                    'raw_ball_missing_before' if before is None else 'raw_ball_missing_after')
    incoming, outgoing = ball[i] - ball[before], ball[after] - ball[i]
    a, b = np.linalg.norm(incoming), np.linalg.norm(outgoing)
    maximum = math.hypot(dense['source_width'], dense['source_height']) * .35
    record['metrics'].update(incoming_displacement_px=float(a), outgoing_displacement_px=float(b),
                             maximum_displacement_px=maximum)
    if a < 5 or b < 5 or max(a, b) > maximum:
        return fail('trajectory_displacement_below_5px' if min(a, b) < 5 else 'trajectory_displacement_above_diagonal_0.35')
    cosine = float(np.dot(incoming, outgoing) / (a*b))
    record['metrics']['trajectory_cosine'] = cosine
    if cosine > .75:
        return fail('trajectory_cosine_above_0.75')
    proximity, turn = max(0, 1-distance/1.7), (1-cosine)/2
    record['score'] = float(.6*proximity + .4*turn)
    record['metrics'].update(proximity=proximity, turn=turn)
    return record


def main():
    protocol_path = OUT / 'automatic_events_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    result = json.loads((DEST / 'original_audio_result.json').read_text())
    paths = {'dense': OUT / 'continuous_rally/pose/dense_pose.json',
             'trajectory': OUT / 'continuous_rally/tracknet/trajectory.json'}
    for k, path in paths.items():
        assert file_hash(path) == result['frontend_sha256'][k]
    assert file_hash(ROOT / 'specialist_events.py') == protocol['code_hashes'][str(ROOT / 'specialist_events.py')]
    dense, trajectory = [json.loads(paths[k].read_text()) for k in ['dense', 'trajectory']]
    baseline = specialist_events.proposals(dense, trajectory)
    captured = []
    original = specialist_events.find_peaks
    def capture(values, *args, **kwargs):
        peaks, properties = original(values, *args, **kwargs)
        captured.append({'values': values.copy(), 'args': args, 'kwargs': kwargs, 'peaks': peaks.copy()})
        return peaks, properties
    specialist_events.find_peaks = capture
    try:
        instrumented = specialist_events.proposals(dense, trajectory)
    finally:
        specialist_events.find_peaks = original
    saved = json.loads((DEST / 'raw_visual_candidates.json').read_text())['candidates']
    assert canonical(baseline) == canonical(instrumented) == canonical(saved)
    assert len(captured) == 2
    trace, peak_stages, presuppression = {}, {}, []
    distance = max(1, round(dense['sample_fps'] * .32))
    for side, captured_side in zip(['near', 'far'], captured):
        rows = [frame_guards(dense, trajectory, side, i) for i in range(len(dense['frames']))]
        exact_values = np.asarray([r['score'] for r in rows])
        assert np.array_equal(exact_values, captured_side['values']), 'Guard arithmetic differs from production'
        assert captured_side['kwargs'] == {'height': .22, 'prominence': .07, 'distance': distance}
        local_max = find_peaks(exact_values)[0]
        height = find_peaks(exact_values, height=.22)[0]
        separated = find_peaks(exact_values, height=.22, distance=distance)[0]
        final = find_peaks(exact_values, height=.22, distance=distance, prominence=.07)[0]
        assert np.array_equal(final, captured_side['peaks'])
        prominence = dict(zip(map(int, local_max), map(float, peak_prominences(exact_values, local_max)[0])))
        peak_stages[side] = {'local_maxima': local_max.tolist(), 'height_ge_0.22': height.tolist(),
                            'distance_ge_frames': distance, 'after_distance': separated.tolist(),
                            'after_prominence_ge_0.07': final.tolist()}
        for r in rows:
            i = r['frame_index']
            if r['first_failure'] is not None:
                r['selection_status'] = 'failed_frame_guard'
            elif i not in local_max:
                r['selection_status'] = 'not_local_maximum'
            elif i not in height:
                r['selection_status'] = 'peak_height_below_0.22'
            elif i not in separated:
                r['selection_status'] = 'same_actor_peak_distance_suppression'
                r['distance_selected_neighbors'] = [int(p) for p in separated if abs(int(p)-i) < distance]
            elif i not in final:
                r['selection_status'] = 'peak_prominence_below_0.07'
            else:
                r['selection_status'] = 'passed_scipy_peak_filters'
            if i in prominence:
                r['peak_prominence'] = prominence[i]
        trace[side] = rows
        presuppression.extend({'player': side, 'frame_index': int(i),
                               'proposal_score': round(float(exact_values[i]), 5)} for i in final)
    presuppression.sort(key=lambda c: (c['frame_index'], -c['proposal_score']))
    kept = []
    nms_steps = []
    for c in presuppression:
        step = dict(c)
        if kept and c['frame_index']-kept[-1]['frame_index'] < dense['sample_fps']*.2:
            step['compared_to'] = dict(kept[-1])
            if c['proposal_score'] > kept[-1]['proposal_score']:
                previous = kept[-1]
                trace[previous['player']][previous['frame_index']]['selection_status'] = 'cross_actor_nms_replaced_by_stronger'
                kept[-1] = c
                step['action'] = 'replace_previous'
            else:
                trace[c['player']][c['frame_index']]['selection_status'] = 'cross_actor_nms_discarded'
                step['action'] = 'discard_current'
        else:
            kept.append(c)
            step['action'] = 'append'
        nms_steps.append(step)
    assert [(c['player'], c['frame_index']) for c in kept] == [(c['player'], c['frame_index']) for c in baseline]
    for c in kept:
        trace[c['player']][c['frame_index']]['selection_status'] = 'returned_production_event'
    mapping = result['raw_visual']['slot_to_court_side']
    inverse = {side: slot for slot, side in mapping.items()}
    missed = result['raw_visual']['missed_contacts']
    assert [g['sample_id'].rsplit('_', 1)[-1] for g in missed] == ['1', '3', '9', '14', '15']
    summaries = []
    for g in missed:
        center, side = g['local_frame'], inverse[g['court_side']]
        window = trace[side][max(0, center-6):center+7]
        diagnostic = trace[side][max(0, center-12):center+13]
        counts = Counter(r['first_failure'] or r['selection_status'] for r in window)
        summaries.append({'sample_id': g['sample_id'], 'author_global_frame': g['global_frame'],
            'author_local_frame': center, 'correct_actor_slot': side, 'court_side': g['court_side'],
            'author_frame_trace': trace[side][center], 'within_0.2s_first_failure_counts': dict(counts),
            'within_0.2s_max_score': max(r['score'] for r in window),
            'within_0.2s_all_guards_pass_frames': [r['frame_index'] for r in window if r['first_failure'] is None],
            'within_0.2s_frames': window,
            'diagnostic_only_0.4s_frames': diagnostic,
            'diagnostic_only_0.4s_production_events': [c for c in baseline if c['player'] == side and abs(c['frame_index']-center) <= 12]})
    record = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': file_hash(protocol_path), 'script_sha256': file_hash(Path(__file__)),
        'production_code_sha256': file_hash(ROOT / 'specialist_events.py'),
        'frontend_sha256': {k: file_hash(p) for k, p in paths.items()},
        'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
        'no_model_inference_or_threshold_changes': True,
        'canonical_event_bytes_equal_original_and_saved': True,
        'canonical_event_sha256': hashlib.sha256(canonical(baseline)).hexdigest(),
        'all_603frame_score_arrays_exactly_equal_original_function': True,
        'actual_peak_filter_order': 'local maxima -> height -> distance -> prominence -> cross-actor NMS',
        'predicate_trace_is_first_short_circuit_failure_not_all_possible_failures': True,
        'peak_stages': peak_stages, 'nms_steps': nms_steps, 'missed_contacts': summaries}
    atomic_json(DEST / 'missed_hit_gate_trace.json', record)
    for r in summaries:
        print(json.dumps({'sample_id': r['sample_id'], 'at_author_frame': r['author_frame_trace'],
                          'within_0.2s_first_failure_counts': r['within_0.2s_first_failure_counts'],
                          'max_score': r['within_0.2s_max_score'],
                          'passed_frames': r['within_0.2s_all_guards_pass_frames'],
                          'nearby_0.4s_returned_events': len(r['diagnostic_only_0.4s_production_events'])}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
