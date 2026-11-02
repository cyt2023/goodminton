"""Independent published shuttlepose GRU, isolated from production hit fusion.

The author predicts 14-frame windows and publishes six-frame interval outputs.
No labels or benchmark scores are used to tune this adapter. Pose/ball extraction
is shared with this project's caches and is explicitly not the author frontend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from hit_gru import predict_windows

ROOT = Path(__file__).resolve().parent
AUTHOR = ROOT / 'vendor/Badminton-hit-detection'
WEIGHTS = ROOT / 'models/hit_gru/keras_tensors.npz'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_coordinates(dense, trajectory, carry_pose=True, ball_mode='raw'):
    """Author ordering and image normalization, without per-joint confidence cuts.

    Author's filtered-pose path carries missing people forward. Our tracked
    identities replace the author's court-polygon detector selection, so neither
    this function nor the benchmark claims a complete frontend reproduction.
    """
    frames = dense['frames']
    if dense['source_sha256'] != trajectory['source_sha256']:
        raise ValueError('Pose/trajectory source hashes differ')
    if len(frames) != len(trajectory['frames']):
        raise ValueError('Pose/trajectory frame counts differ')
    heights = {}
    for side in ['near', 'far']:
        observed = [np.mean([p['keypoints'][j]['y'] for j in [15, 16]])
                    for f in frames if (p := f['players'].get(side))]
        heights[side] = float(np.median(observed)) if observed else -1e10
    order = sorted(heights, key=heights.get, reverse=True)
    scale = np.array([dense['source_width'], dense['source_height']], np.float32)
    result = np.zeros((len(frames), 70), np.float32)
    last = np.zeros((2, 34), np.float32)
    for i, (frame, ball) in enumerate(zip(frames, trajectory['frames'])):
        if frame['frame_index'] != ball['frame_index']:
            raise ValueError('Pose/trajectory frame indices differ')
        xy = ball.get('raw_xy') if ball_mode == 'raw' else ball.get('rectified_xy')
        visible = ball.get('raw_visible', False) if ball_mode == 'raw' else xy is not None
        if visible and xy is not None and np.isfinite(xy).all():
            result[i, :2] = np.asarray(xy) / scale
        for player, side in enumerate(order):
            person = frame['players'].get(side)
            if person:
                coords = np.asarray([[k['x'], k['y']] for k in person['keypoints']], np.float32)
                if coords.shape != (17, 2) or not np.isfinite(coords).all():
                    raise ValueError('Expected finite COCO17 joints')
                last[player] = (coords / scale).reshape(-1)
            elif not carry_pose:
                last[player] = 0
            result[i, 2 + 34 * player:2 + 34 * (player + 1)] = last[player]
    return result, order


def maximise_hit_score(probs, fps=30, window_len=6):
    """Linear-memory equivalent of author's fixed-window DP (including quirks).

    Kept: first/last output no-hit, >0.5s spacing, alternating players, original
    hit-count bound, global tau, and the original predecessor hit-count formula.
    """
    n = len(probs)
    if n == 0:
        return np.zeros(0, np.int32)
    if n == 1:
        return np.asarray([np.argmax(probs[0])], np.int32)
    tau = float(np.mean(np.array([np.mean(probs[:, 1]), np.mean(probs[:, 2])])))
    score = np.full((n, 3), -100000., np.float64)
    counts = np.zeros((n, 3), np.int32)
    last_index = np.zeros((n, 3), np.int32)
    last_label = np.zeros((n, 3), np.int32)
    previous = np.zeros((n, 3, 2), np.int32)
    score[0, 0] = 0.
    for f in range(1, n):
        chosen = int(np.argmax(score[f - 1]))
        score[f, 0] = score[f - 1, chosen]
        counts[f, 0] = counts[f - 1, chosen]
        last_index[f, 0] = last_index[f - 1, chosen]
        last_label[f, 0] = last_label[f - 1, chosen]
        previous[f, 0] = [f - 1, chosen]
        for label in [1, 2]:
            valid = (f - last_index[f - 1, 0] > 0.5 * fps and
                     counts[f - 1, 0] <= math.ceil(f / fps) + 1 and
                     last_label[f - 1, 0] != label and f >= window_len)
            if valid:
                before = f - window_len
                score[f, label] = score[before, 0] + np.sum(probs[before + 1:f + 1, label]) - window_len * tau
                counts[f, label] = counts[f - 1, 0] + 1
                last_index[f, label], last_label[f, label] = f, label
                previous[f, label] = [before, 0]
    output = np.zeros(n, np.int32)
    frame, label = n - 1, 0
    while frame > 0:
        parent, parent_label = previous[frame, label]
        if label:
            output[parent + 1:frame + 1] = label
        frame, label = int(parent), int(parent_label)
    return output


def naive_labels(probs, fps=30, threshold=.95, frame_range=3):
    labels = np.argmax(probs, axis=1)
    labels[np.max(probs, axis=1) <= threshold] = 0
    selected = []
    for index in np.flatnonzero(labels):
        if not selected or index > selected[-1] + int(fps / 2):
            selected.append(int(index))
    result = np.zeros(len(labels), np.int32)
    for index in selected:
        # Preserve NumPy slicing semantics of the original notebook.
        result[index-frame_range:index+frame_range+1] = labels[index]
    return result


def labels_to_events(labels, frames, order, frame_probs, offset=0):
    events = []
    start = 0
    while start < len(labels):
        label = int(labels[start])
        end = start + 1
        while end < len(labels) and labels[end] == label:
            end += 1
        if label:
            midpoint = int(np.floor((start + end - 1) / 2 + .5))
            row = midpoint + offset
            events.append({'id': f'event_{len(events):04d}',
                'frame_index': int(frames[row]['frame_index']),
                'actor': order[label - 1], 'player': order[label - 1],
                'confidence': float(np.mean(frame_probs[start:end, label])),
                'start_frame': int(frames[start + offset]['frame_index']),
                'end_frame': int(frames[end - 1 + offset]['frame_index']),
                'camera_class': label, 'representative': 'rounded_interval_midpoint'})
        start = end
    return events


def infer(dense, trajectory, postprocess='max', **kwargs):
    rows, order = image_coordinates(dense, trajectory, carry_pose=True)
    with np.load(WEIGHTS, allow_pickle=False) as z:
        weights = {k: z[k] for k in z.files}
    n = max(0, len(rows) - 13)
    probs = np.zeros((n, 3), np.float32)
    for begin in range(0, n, 256):
        indices = np.arange(begin, min(begin + 256, n))
        probs[indices] = predict_windows(rows[indices[:, None] + np.arange(14)], weights)
    labels = maximise_hit_score(probs) if postprocess == 'max' else naive_labels(probs)
    events = labels_to_events(labels, dense['frames'], order, probs, offset=13)
    return {'events': events, 'window_probs': probs, 'window_labels': labels,
        'metadata': {'model': 'kwyoke-shuttlepose-GRU-150', 'postprocess': postprocess,
            'source': 'https://github.com/kwyoke/Badminton-hit-detection',
            'weights_sha256': sha256(WEIGHTS), 'code_sha256': sha256(__file__),
            'author_postprocess_source_sha256': sha256(AUTHOR / 'process_pred_probs.source.txt'),
            'camera_to_identity_order': order, 'window_frames': 14, 'label_last_frames': 6,
            'postprocess_fps': 30, 'input_resampling': 'none; native decoded frames as in author notebook',
            'point_conversion': 'Each nonzero label run -> rounded midpoint; original interval retained.',
            'frontend_adaptations': ['Shared YOLOX/RTMPose frontend and existing tracked identities replace author Faster-RCNN/HRNet/court-polygon filtering.',
                'All COCO17 coordinates retained; missing pose carried forward as author filtered2 path.',
                'TrackNetV3 raw coordinates replace author shuttle frontend; no additional peak gate.',
                'Fixed author30fps DP/naive parameters retained for all native-cadence sources.'],
            'tuning': 'Published weights and notebook defaults; no benchmark annotation access.'}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dense', required=True)
    parser.add_argument('--trajectory', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--postprocess', choices=['max', 'naive'], default='max')
    args = parser.parse_args()
    result = infer(json.loads(Path(args.dense).read_text()), json.loads(Path(args.trajectory).read_text()), postprocess=args.postprocess)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix('.npz'), window_probs=result.pop('window_probs'), window_labels=result.pop('window_labels'))
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'output': str(path), 'events': len(result['events'])}))


if __name__ == '__main__':
    main()
