"""Independent 2026 third-party Adapted-HitNet benchmark adapter.

This checkpoint is raneeng's retrained four-layer GRU, not the released
MonoTrack author's checkpoint (the original repository does not include one).
The original predictor's scaling, windowing and averaging are reproduced here;
the project caches replace its YOLOv8-pose frontend and are disclosed as such.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np

from benchmark_hits_gru import image_coordinates, labels_to_events, sha256

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'vendor/Adapted-HitNet'
MODEL_PATH = UPSTREAM / 'training/hitnet_model.h5'
TRAIN_MATCHES = {str(i) for i in list(range(1, 7)) + list(range(22, 27)) + [28] + list(range(32, 36)) + list(range(37, 45))}
VALIDATION_MATCHES = {'7', '21', '29', '30', '31', '36'}


def scale_data(x, eps=1e-6):
    """Exact upstream train_hitnet.scale_data, x and y grouped independently."""
    x = x.astype(np.float64).copy()
    for cols in [list(range(0, x.shape[1], 2)), list(range(1, x.shape[1], 2))]:
        values = x[:, cols]
        missing = np.abs(values) < eps
        if (~missing).sum() == 0:
            continue
        lo, hi = np.min(values[~missing]), np.max(values[~missing])
        if hi - lo < eps:
            continue
        values[~missing] = (values[~missing] - lo) / (hi - lo) + 1
        x[:, cols] = values
    return x


def build_windows(rows, corners):
    n = len(rows)
    if n < 12:
        return np.zeros((0, 936), np.float64)
    all_rows = np.hstack([rows, np.tile(np.asarray(corners).reshape(8), (n, 1))])
    indices = np.arange(n - 11)[:, None] + np.arange(12)
    return scale_data(all_rows[indices].reshape(n - 11, 936))


def aggregate_window_predictions(probs, n_frames):
    sums = np.zeros((n_frames, 3))
    counts = np.zeros(n_frames)
    for i, values in enumerate(probs):
        sums[i+6:i+12] += values
        counts[i+6:i+12] += 1
    covered = counts > 0
    result = np.zeros((n_frames, 3))
    result[covered] = sums[covered] / counts[covered, None]
    result[~covered] = [1., 0., 0.]
    return result


@lru_cache(maxsize=1)
def load_model():
    # Existing vision environment supplies Keras3 and PyTorch, avoiding a second
    # heavy TensorFlow installation. Forward equations/weights are unchanged.
    os.environ.setdefault('KERAS_BACKEND', 'torch')
    import keras
    import torch
    torch.set_num_threads(2)
    return keras.models.load_model(MODEL_PATH, compile=False)


def infer(dense, trajectory, court_corners=None, match_id=None, **kwargs):
    if court_corners is None:
        raise ValueError('Adapted-HitNet requires four doubles-court pixel corners: TL,TR,BL,BR')
    corners = np.asarray(court_corners, np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError('court_corners must be finite4x2 pixels ordered TL,TR,BL,BR')
    rows, order = image_coordinates(dense, trajectory, carry_pose=False, ball_mode='rectified')
    # Shared coordinate assembler normalizes by width/height. This model was
    # trained on pixels before global nonzero minmax scaling; restore pixels.
    scale = np.tile([dense['source_width'], dense['source_height']], 35)
    rows = rows.astype(np.float64) * scale
    windows = build_windows(rows, corners)
    model = load_model()
    probs = model.predict(windows.astype(np.float32), batch_size=256, verbose=0) if len(windows) else np.zeros((0, 3), np.float32)
    frame_probs = aggregate_window_predictions(probs, len(rows))
    labels = np.argmax(frame_probs, axis=1)
    events = labels_to_events(labels, dense['frames'], order, frame_probs)
    return {'events': events, 'window_probs': probs, 'frame_probs': frame_probs,
        'frame_labels': labels, 'metadata': {
            'model': 'raneeng-Adapted-HitNet-four-BiGRU128',
            'source': 'https://github.com/raneeng/Adapted-HitNet',
            'revision': 'b56d955c9fb08fc799c0d1b841552a238d65d834',
            'weights_sha256': sha256(MODEL_PATH), 'code_sha256': sha256(__file__),
            'author_inference_source_sha256': sha256(UPSTREAM / 'training/hitnet_infer.py'),
            'author_training_source_sha256': sha256(UPSTREAM / 'training/train_hitnet.py'),
            'checkpoint_origin': 'Third-party adaptation/retraining; not original MonoTrack released weights.',
            'architecture_parameters': int(model.count_params()),
            'camera_to_identity_order': order, 'window_frames': 12, 'label_last_frames': 6,
            'court_corners_px_TL_TR_BL_BR': corners.tolist(),
            'court_input': 'Fixed outer doubles corners; manually supplied or external annotation; benchmark must distinguish this semi-automatic condition.',
            'input_resampling': 'none; native decoded cadence',
            'point_conversion': 'Consecutive argmax hit frames of the same actor -> rounded run midpoint; interval retained.',
            'postprocess': 'Author last-six-frame overlapping-window averaging then argmax, no custom threshold/NMS.',
            'match_id': str(match_id) if match_id is not None else None,
            'author_training_overlap': str(match_id) in TRAIN_MATCHES if match_id is not None else None,
            'author_validation_overlap': str(match_id) in VALIDATION_MATCHES if match_id is not None else None,
            'frontend_adaptations': ['Shared YOLOX/RTMPose/tracked identities replace author YOLOv8s-pose/court-half assignment.',
                'All17 COCO keypoints retained, missing player zero-filled as upstream.',
                'Shared TrackNetV3 rectified coordinates including InpaintNet; author pipeline likewise enables InpaintNet.',
                'Keras3 uses local PyTorch CPU backend; checkpoint loaded directly without retraining.'],
            'tuning': 'Unmodified published checkpoint and author inference defaults; no benchmark labels read.'}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dense', required=True)
    parser.add_argument('--trajectory', required=True)
    parser.add_argument('--court', required=True, help='JSON4x2 pixel TL,TR,BL,BR or object court_corners')
    parser.add_argument('--output', required=True)
    parser.add_argument('--match-id')
    args = parser.parse_args()
    court = json.loads(Path(args.court).read_text())
    if isinstance(court, dict):
        court = court['court_corners']
    result = infer(json.loads(Path(args.dense).read_text()), json.loads(Path(args.trajectory).read_text()), court_corners=court, match_id=args.match_id)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix('.npz'), **{k: result.pop(k) for k in ['window_probs', 'frame_probs', 'frame_labels']})
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'output': str(path), 'events': len(result['events'])}))


if __name__ == '__main__':
    main()
