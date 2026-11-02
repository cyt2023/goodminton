"""Untrained hit proposals combining observed shuttle, wrists and trajectory turn.

These windows adapt BST's neighbouring-hit clipping design, with unconfirmed
hit proposals instead of dataset hit labels. They do not reproduce its clipping
experiment. Missing/completed ball coordinates cannot alone create an event.
"""
import hashlib
import math

import numpy as np
from scipy.signal import find_peaks


def proposals(dense, trajectory):
    frames, ball_frames = dense['frames'], trajectory['frames']
    if len(frames) != len(ball_frames):
        raise ValueError('Ball and player decoded frame counts differ')
    if any(f['frame_index'] != b['frame_index'] for f, b in zip(frames, ball_frames)):
        raise ValueError('Ball and player original frame indices differ')
    fps = dense['sample_fps']
    ball = np.array([b['raw_xy'] if b['raw_visible'] else [np.nan, np.nan] for b in ball_frames], dtype=float)
    width, height = dense['source_width'], dense['source_height']
    diagonal = math.hypot(width, height)
    candidates = []
    for side in ['near', 'far']:
        values = np.zeros(len(frames))
        details = {}
        for i in range(4, len(frames) - 4):
            person = frames[i]['players'].get(side)
            if person is None or not np.isfinite(ball[i]).all():
                continue
            joints = person['keypoints']
            if any(joints[j]['score'] < .4 for j in [5, 6, 11, 12]):
                continue
            pts = np.asarray([[j['x'], j['y']] for j in joints])
            torso = np.linalg.norm((pts[5] + pts[6] - pts[11] - pts[12]) / 2)
            if torso < 12:
                continue
            wrists = [j for j in [9, 10] if joints[j]['score'] >= .4]
            if not wrists:
                continue
            distances = {j: float(np.linalg.norm(ball[i] - pts[j]) / torso) for j in wrists}
            wrist = min(distances, key=distances.get)
            distance = distances[wrist]
            if distance > 1.7:
                continue
            before = next((j for j in range(i - 3, i - 5, -1) if np.isfinite(ball[j]).all()), None)
            after = next((j for j in range(i + 3, i + 5) if np.isfinite(ball[j]).all()), None)
            if before is None or after is None:
                continue
            incoming, outgoing = ball[i] - ball[before], ball[after] - ball[i]
            a, b = np.linalg.norm(incoming), np.linalg.norm(outgoing)
            if a < 5 or b < 5 or max(a, b) > diagonal * .35:
                continue
            cosine = float(np.dot(incoming, outgoing) / (a * b))
            if cosine > .75:
                continue
            # A floor bounce may turn but must also lie near a visible wrist.
            proximity, turn = max(0, 1 - distance / 1.7), (1 - cosine) / 2
            values[i] = .6 * proximity + .4 * turn
            details[i] = {'raw_shuttle_xy': ball[i].tolist(), 'wrist': 'left' if wrist == 9 else 'right',
                          'wrist_distance_torso_units': round(distance, 5), 'trajectory_cosine': round(cosine, 5),
                          'incoming_frames': [before, i], 'outgoing_frames': [i, after],
                          'note': 'Observed model ball and wrist proximity plus trajectory turn; not verified racket contact.'}
        peaks, _ = find_peaks(values, height=.22, prominence=.07, distance=max(1, round(fps * .32)))
        for i in peaks:
            if int(i) not in details:
                continue
            candidates.append({'player': side, 'frame_index': int(i), 'peak_time': frames[i]['source_time'],
                               'proposal_score': round(float(values[i]), 5), 'evidence': details[int(i)]})
    candidates.sort(key=lambda c: (c['frame_index'], -c['proposal_score']))
    # Two players cannot be the target of the same hit. Retain the stronger cue.
    filtered = []
    for c in candidates:
        if filtered and c['frame_index'] - filtered[-1]['frame_index'] < fps * .2:
            if c['proposal_score'] > filtered[-1]['proposal_score']:
                filtered[-1] = c
        else:
            filtered.append(c)
    for k, c in enumerate(filtered):
        i = c['frame_index']
        previous = filtered[k-1]['frame_index'] if k else i - round(fps * .5)
        following = filtered[k+1]['frame_index'] if k+1 < len(filtered) else i + round(fps * .5)
        lo = max(0, previous, i - round(fps * 1.5))
        hi = min(len(frames)-1, following + round(fps * .25), i + round(fps * 1.75))
        c.update(start_frame=int(lo), end_frame=int(hi), start=frames[lo]['source_time'],
                 end=min(dense['end'], frames[hi]['source_time'] + 1 / fps),
                 kind='shuttle_wrist_turn_hit_candidate', review_status='unreviewed',
                 id=hashlib.md5(('specialist-v1:' + dense['source_sha256'] + ':' + c['player'] + ':' + str(i)).encode()).hexdigest(),
                 time_basis='original_decoded_frame_index_with_decoder_timestamp',
                 interpretation='Automatic proposed hit, not confirmed contact or stroke type.')
    return filtered
