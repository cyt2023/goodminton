"""Same-player, same-video retrieval and missing-aware 2D DTW baseline.

References are other unreviewed personal clips, not professional exemplars or
correct techniques. Alignment stays within pre/post hit-proposal regions and
saves original times; no 2D measurement is treated as a physical joint angle.
"""
import math

import numpy as np


def features(dense, event):
    fps = dense['sample_fps']
    lo, hi = max(event['start_frame'], event['frame_index']-round(fps*.65)), min(event['end_frame'], event['frame_index']+round(fps*.65))
    indices = np.linspace(lo, hi, min(32, hi-lo+1)).round().astype(int)
    result = []
    for i in indices:
        f = dense['frames'][int(i)]
        p = f['players'].get(event['player'])
        if not p:
            continue
        k = p['keypoints']
        if any(k[j]['score'] < .5 for j in [5, 6, 11, 12]):
            continue
        points = np.array([[j['x'], j['y']] for j in k], dtype=float)
        pelvis = (points[11]+points[12])/2
        shoulders = (points[5]+points[6])/2
        scale = np.linalg.norm(shoulders-pelvis)
        if scale < 12:
            continue
        valid = np.array([j['score'] >= .5 and j['in_frame'] for j in k])
        result.append({'frame': int(i), 'time': f['source_time'], 'points': (points-pelvis)/scale,
                       'valid': valid, 'phase': 'before_hit_proposal' if i < event['frame_index'] else 'after_hit_proposal'})
    return result


def align(a, b, band=.25):
    if len(a) < 8 or len(b) < 8:
        return None
    n, m = len(a), len(b)
    costs, parents = np.full((n+1, m+1), np.inf), {}
    costs[0, 0] = 0
    joints = np.arange(5, 17)
    for i in range(1, n+1):
        for j in range(1, m+1):
            if abs((i-1)/max(1,n-1)-(j-1)/max(1,m-1)) > band or a[i-1]['phase'] != b[j-1]['phase']:
                continue
            common = joints[a[i-1]['valid'][joints] & b[j-1]['valid'][joints]]
            if len(common) < 8:
                continue
            distance = np.mean(np.linalg.norm(a[i-1]['points'][common]-b[j-1]['points'][common], axis=-1))
            choices = [(costs[i-1,j-1], (i-1,j-1)), (costs[i-1,j]+.03, (i-1,j)), (costs[i,j-1]+.03, (i,j-1))]
            cost, parent = min(choices, key=lambda pair: pair[0])
            costs[i,j] = cost+distance
            parents[i,j] = parent
    if not math.isfinite(costs[n,m]):
        return None
    path, current = [], (n,m)
    while current != (0,0):
        i,j = current
        if i == 0 or j == 0 or current not in parents:
            return None
        path.append({'personal_frame': a[i-1]['frame'], 'personal_time': a[i-1]['time'],
                     'reference_frame': b[j-1]['frame'], 'reference_time': b[j-1]['time'],
                     'phase_basis': a[i-1]['phase']})
        current = parents[current]
    path.reverse()
    return {'mean_path_distance_torso_units': float(costs[n,m]/len(path)), 'path': path,
            'personal_original_duration': a[-1]['time']-a[0]['time'],
            'reference_original_duration': b[-1]['time']-b[0]['time'],
            'note': 'Same-camera image-plane pose correspondence around unconfirmed hit anchors; distance does not measure correctness or learning value.'}


def retrieve_and_align(dense, events):
    valid = [e for e in events if e.get('quality') == 'usable' and e.get('visible_stroke') and
             e.get('stroke_certainty') == 'likely' and e.get('style_certainty') == 'likely' and not e.get('classification_error')]
    prepared = {e['id']: features(dense,e) for e in valid}
    results = []
    for event in valid:
        choices = [other for other in valid if other['id'] != event['id'] and other['player'] == event['player'] and
                   other['stroke_type'] == event['stroke_type'] and other['swing_style'] == event['swing_style'] and
                   abs(other['peak_time']-event['peak_time']) > 3]
        choices.sort(key=lambda other: (abs((other['end']-other['start'])-(event['end']-event['start'])), -other.get('bst_softmax_score',0)))
        ranked = []
        for other in choices[:5]:
            correspondence = align(prepared[event['id']],prepared[other['id']])
            if correspondence:
                ranked.append({'reference_event_id': other['id'], 'reference_player': other['player'],
                               'reference_start': other['start'], 'reference_end': other['end'],
                               'reference_kind': 'other_personal_clip_unreviewed', **correspondence})
        ranked.sort(key=lambda r:r['mean_path_distance_torso_units'])
        results.append({'event_id':event['id'], 'references':ranked[:3],
                        'status':'aligned_personal_candidates' if ranked else 'no_comparable_personal_candidate'})
    return {'method':'same-video label-filtered retrieval + phase-constrained missing-aware 2D DTW',
            'results':results,'eligible_personal_events':len(valid),
            'expert_reference_status':'not_available_locally',
            'scope':'Engineering baseline; not CoachMe, MADR or ExpertAF reproduction. References are not labelled correct technique.'}
