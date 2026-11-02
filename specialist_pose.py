"""Dense RTMPose inference on decoded original frames for specialist classifiers.

Uses the existing scout's candidate identities, estimates fresh boxes on every
decoded frame, and saves decoder timestamps. No nearest-5-Hz box is drawn on a
different frame. Candidate identity remains unreviewed.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from rtmlib import RTMPose, YOLOX

from analyze_pose import get_weights, save_json, sha256, JOINTS, image_features
from auto_players import AutomaticFollower


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    source = Path(config['source_file'])
    original = json.loads(Path(config['pose_file']).read_text())
    output = Path(config['specialist_folder'])
    output.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    if source_hash != config['source_sha256']:
        raise ValueError('Source video changed')
    selection = original.get('automatic_player_selection')
    if not selection:
        raise ValueError('Dense analysis requires an automatic scout selection')
    slots = {}
    for candidate in selection['candidates']:
        slots[candidate['slot']] = {
            'id': candidate['scout_track_id'],
            'observations': candidate['scout_observations'],
            'appearance': np.asarray(candidate['torso_hsv_sqrt_histogram']),
            'identity_reference': np.asarray(candidate['identity_reference_hsv_sqrt_histogram']),
            'identity_references': [np.asarray(v) for v in candidate['identity_reference_histograms']],
        }
    follower = AutomaticFollower(slots, selection['scout_interval_seconds'])
    weights = get_weights()
    detector = YOLOX(str(weights['detector']['file']), model_input_size=(640, 640),
                     score_thr=.4, backend='onnxruntime', device='cpu')
    pose_model = RTMPose(str(weights['pose']['file']), model_input_size=(192, 256),
                        backend='onnxruntime', device='cpu')
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if [width, height] != [original['source_width'], original['source_height']]:
        raise ValueError('Decoded frame orientation differs from the scout')
    frames, timestamp_fallbacks = [], 0
    started = time.perf_counter()
    while True:
        ok, image = capture.read()
        if not ok:
            break
        index = len(frames)
        timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        timing = 'opencv_decoder_pos_msec'
        if not np.isfinite(timestamp) or (index > 0 and timestamp <= frames[-1]['source_time']):
            timestamp = index / fps
            timing = 'frame_index_divided_by_nominal_fps_fallback'
            timestamp_fallbacks += 1
        boxes = np.asarray(detector(image)).reshape(-1, 4)
        assignments, reasons = follower.select_boxes(image, boxes, timestamp)
        players = {'near': None, 'far': None}
        if assignments:
            points, scores = pose_model(image, bboxes=[boxes[i] for i in assignments.values()])
            for p, (side, box_index) in enumerate(assignments.items()):
                pts, conf = np.asarray(points[p]), np.asarray(scores[p])
                valid = np.isfinite(pts).all(axis=1) & np.isfinite(conf)
                conf = np.where(valid, conf, 0)
                pts = np.where(np.isfinite(pts), pts, 0)
                inside = valid & (pts[:, 0] >= 0) & (pts[:, 0] < width) & (pts[:, 1] >= 0) & (pts[:, 1] < height)
                players[side] = {
                    'track_id': original['id'] + '-' + side,
                    'bbox': boxes[box_index].tolist(),
                    'keypoints': [{'name': name, 'x': float(v[0]), 'y': float(v[1]),
                                  'score': float(s), 'in_frame': bool(i)}
                                 for name, v, s, i in zip(JOINTS, pts, conf, inside)],
                    'features': image_features(pts, np.where(inside, conf, 0), .5),
                    'identity_status': 'automatic_candidate_unreviewed',
                }
        frames.append({'frame_index': index, 'source_time': round(float(timestamp), 6),
                       'time_basis': timing, 'players': players, 'assignment': reasons})
        if len(frames) % 100 == 0:
            print('DENSE_PROGRESS ' + json.dumps({'processed': len(frames), 'total': total,
                  'runtime_seconds': round(time.perf_counter() - started, 3)}), flush=True)
    capture.release()
    if not frames or abs(len(frames) - total) > 2:
        raise ValueError('Original video did not decode completely')
    result = {**{k: original[k] for k in ['id', 'source_video_id', 'source_sha256', 'source_width', 'source_height', 'start', 'end']},
              'engine_version': 'dense-rtmpose-original-frames-v1',
              'coarse_pose_sha256': sha256(Path(config['pose_file'])),
              'sample_fps': fps, 'frames': frames, 'models': weights,
              'decoded_frame_count': len(frames), 'timestamp_fallbacks': timestamp_fallbacks,
              'runtime_seconds': round(time.perf_counter() - started, 3),
              'code_sha256': sha256(Path(__file__)),
              'assigned_frames': {side: sum(f['players'][side] is not None for f in frames) for side in slots},
              'identity_note': 'Dense boxes and joints are fresh estimates on each original decoded frame; identities remain scout candidates, not reviewed truth.'}
    save_json(output / 'dense_pose.json', result)
    print('DENSE_DONE ' + json.dumps({k: result[k] for k in ['decoded_frame_count', 'runtime_seconds', 'assigned_frames', 'timestamp_fallbacks']}), flush=True)


if __name__ == '__main__':
    main()
