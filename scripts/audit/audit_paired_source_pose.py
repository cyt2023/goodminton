"""Isolated, per-clip production pose replay for the frozen paired-source audit.

No database registration, production publication, labels, or weights are changed.
The primary inference/serialization follows specialist_pose.py. The declared
secondary pass changes only RTMPose input BGR -> RGB and reuses saved primary
detector boxes and slot assignments. Run with .venv-pose/bin/python.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/audit/
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import inspect
import json
from pathlib import Path
import time

import cv2
import imageio_ffmpeg
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
from rtmlib import RTMPose, YOLOX
import rtmlib.tools.base as rtmlib_base
import rtmlib.tools.pose_estimation.rtmpose as rtmlib_pose
import rtmlib.tools.object_detection.yolox as rtmlib_detector

from analyze_pose import JOINTS, image_features, save_json, scout_automatic_players, sha256
from auto_players import AutomaticFollower

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT.parent / '03_数据/击球标注/根因诊断/2026-10-15_同源前端对照'


def checked_weights():
    # get_weights() also rewrites models/model_manifest.json. Read and verify the
    # existing manifest instead, keeping production files untouched.
    path = ROOT / 'models/model_manifest.json'
    weights = json.loads(path.read_text())
    for role, record in weights.items():
        file = Path(record['file'])
        if not file.is_file() or sha256(file) != record['sha256']:
            raise ValueError('Missing or changed frozen weight: ' + role)
    return weights


def create_models(weights, threads):
    # Resource configuration only: RTMLib exposes no SessionOptions argument.
    # Limit session threads while constructing models; restore the API at once.
    original_session = ort.InferenceSession

    def limited_session(*args, **kwargs):
        if kwargs.get('sess_options') is not None:
            raise ValueError('Unexpected RTMLib session options')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        kwargs['sess_options'] = options
        return original_session(*args, **kwargs)

    ort.InferenceSession = limited_session
    try:
        detector = YOLOX(str(weights['detector']['file']), model_input_size=(640, 640),
                         score_thr=.4, backend='onnxruntime', device='cpu')
        pose_model = RTMPose(str(weights['pose']['file']), model_input_size=(192, 256),
                            backend='onnxruntime', device='cpu')
    finally:
        ort.InferenceSession = original_session
    return detector, pose_model


def make_follower(selection):
    # Same round-tripped candidate references used by specialist_pose.main().
    slots = {}
    for candidate in selection['candidates']:
        slots[candidate['slot']] = {
            'id': candidate['scout_track_id'],
            'observations': candidate['scout_observations'],
            'appearance': np.asarray(candidate['torso_hsv_sqrt_histogram']),
            'identity_reference': np.asarray(candidate['identity_reference_hsv_sqrt_histogram']),
            'identity_references': [np.asarray(v) for v in candidate['identity_reference_histograms']],
        }
    return AutomaticFollower(slots, selection['scout_interval_seconds']), slots


def serialize_players(original, boxes, assignments, points, scores, width, height):
    # Unchanged specialist_pose serialization, including low scores/in_frame.
    players = {'near': None, 'far': None}
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
    return players


def validate_dense(result, clip, frame_map):
    if result['source_sha256'] != clip['source_sha256']:
        raise ValueError('Dense source SHA mismatch')
    if result['decoded_frame_count'] != clip['frames'] or len(result['frames']) != clip['frames']:
        raise ValueError('Dense frame count mismatch')
    for i, frame in enumerate(result['frames']):
        if frame['frame_index'] != i or frame['global_frame'] != frame_map[i]['global_frame']:
            raise ValueError('Dense local/global frame mismatch')
        for player in frame['players'].values():
            if player is not None and len(player['keypoints']) != 17:
                raise ValueError('COCO joint count mismatch')


def summarize(result):
    frames = result['frames']
    assigned = {side: sum(f['players'][side] is not None for f in frames) for side in ['near', 'far']}
    medians = {}
    for side in ['near', 'far']:
        values = [np.mean([p['keypoints'][j]['y'] for j in [15, 16]])
                  for f in frames if (p := f['players'].get(side)) is not None]
        medians[side] = float(np.median(values)) if values else None
    # SpecialistBST sorts median ankle y. Keep undefined cases explicit instead
    # of saving NaN or inventing a slot mapping.
    order = sorted(medians, key=medians.get) if all(v is not None for v in medians.values()) else None
    pair_count = sum(all(f['players'].get(side) is not None for side in ['near', 'far']) for f in frames)
    return {'assigned_frames': assigned,
            'assigned_fraction': {k: v / len(frames) for k, v in assigned.items()},
            'paired_frames': pair_count, 'paired_fraction': pair_count / len(frames),
            'median_ankle_y_original_px': medians,
            'median_ankle_inferred_top_bottom_slot_order': order,
            'inferred_slot_to_side': dict(zip(order, ['Top', 'Bottom'])) if order else None,
            'identity_status': 'automatic_candidate_unreviewed'}


def overlay_middle(image, frame, clip, out):
    canvas = image.copy()
    colors = {'near': (255, 215, 30), 'far': (30, 150, 255)}
    for side, player in frame['players'].items():
        if player is None:
            continue
        x1, y1, x2, y2 = np.rint(player['bbox']).astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colors[side], 3)
        cv2.putText(canvas, side, (x1, max(23, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, colors[side], 2, cv2.LINE_AA)
    cv2.imwrite(str(out / 'middle_frame_actual_boxes.png'), canvas)
    thumbnail = cv2.resize(canvas, (480, 270))
    tile = Image.new('RGB', (480, 314), 'white')
    tile.paste(Image.fromarray(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)), (0, 44))
    draw = ImageDraw.Draw(tile)
    draw.text((8, 5), f"{clip['sample_id']} local={frame['frame_index']} global={frame['global_frame']}", fill='black')
    slots = ', '.join(k + (' assigned' if v else ' MISSING') for k, v in frame['players'].items())
    draw.text((8, 23), slots, fill='black')
    return tile


def run_clip(clip, output, detector, pose_model, weights, code_hash, ffmpeg):
    started = time.perf_counter()
    source = Path(clip['source_file'])
    if not source.is_file() or sha256(source) != clip['source_sha256']:
        raise ValueError('Source clip missing or changed: ' + clip['sample_id'])
    frame_map_path = source.parent / 'frame_map.json'
    if sha256(frame_map_path) != clip['frame_map_sha256']:
        raise ValueError('Source frame map changed: ' + clip['sample_id'])
    frame_map = json.loads(frame_map_path.read_text())
    if len(frame_map) != clip['frames'] or any(r['clip_index'] != i for i, r in enumerate(frame_map)):
        raise ValueError('Source frame map count/index mismatch')
    folder = source.parent / 'pose'
    folder.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if [width, height] != [1280, 720] or total != clip['frames'] or not np.isclose(fps, 30):
        raise ValueError('Unexpected frozen clip dimensions/count/fps')
    scout_width = min(width, 960) // 2 * 2
    scout_height = max(2, round(height * scout_width / width / 2) * 2)
    original = {'id': 'paired_source_' + clip['sample_id'], 'source_video_id': clip['sample_id'],
                'source_sha256': clip['source_sha256'], 'source_width': width, 'source_height': height,
                'start': 0., 'end': total / fps}
    job = {'id': original['id'], 'start': original['start'], 'end': original['end']}
    _, selection = scout_automatic_players(job, source, ffmpeg, detector, folder,
                                          scout_width, scout_height, (width, height),
                                          max(1, int(np.ceil(total / fps * 5 - 1e-9))))
    original['automatic_player_selection'] = selection
    coarse_path = folder / 'coarse_pose.json'
    save_json(coarse_path, original)
    # Re-read the serialized scout, matching the specialist's config boundary.
    original = json.loads(coarse_path.read_text())
    follower, slots = make_follower(original['automatic_player_selection'])
    config = {'source_file': str(source), 'source_sha256': clip['source_sha256'],
              'pose_file': str(coarse_path), 'specialist_folder': str(folder),
              'audit_only': True, 'sample_id': clip['sample_id']}
    save_json(folder / 'config.json', config)
    frames, assignments_saved, timestamp_fallbacks = [], [], 0
    capture = cv2.VideoCapture(str(source))
    dense_started = time.perf_counter()
    middle_image = None
    while True:
        ok, image = capture.read()
        if not ok:
            break
        index = len(frames)
        if index >= total or hashlib.sha256(image.tobytes()).hexdigest() != frame_map[index]['pixel_sha256']:
            raise ValueError('Frozen source pixel/frame mismatch')
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
            players = serialize_players(original, boxes, assignments, points, scores, width, height)
        frames.append({'frame_index': index, 'source_time': round(float(timestamp), 6),
                       'time_basis': timing, 'players': players, 'assignment': reasons,
                       'global_frame': frame_map[index]['global_frame']})
        assignments_saved.append({'frame_index': index, 'detector_boxes': boxes.tolist(),
                                  'assignments_in_order': [[side, int(box)] for side, box in assignments.items()]})
        if index == total // 2:
            middle_image = image.copy()
        if len(frames) % 100 == 0:
            print('PAIRED_DENSE_PROGRESS ' + json.dumps({'sample_id': clip['sample_id'], 'processed': len(frames),
                                                        'total': total}), flush=True)
    capture.release()
    bgr_seconds = time.perf_counter() - dense_started
    if len(frames) != total:
        raise ValueError('Original video did not decode completely')
    result = {**{k: original[k] for k in ['id', 'source_video_id', 'source_sha256', 'source_width', 'source_height', 'start', 'end']},
              'engine_version': 'dense-rtmpose-original-frames-v1',
              'coarse_pose_sha256': sha256(coarse_path), 'sample_fps': fps, 'frames': frames, 'models': weights,
              'decoded_frame_count': len(frames), 'timestamp_fallbacks': timestamp_fallbacks,
              'runtime_seconds': round(bgr_seconds, 3), 'code_sha256': code_hash,
              'production_logic_sha256': sha256(ROOT / 'specialist_pose.py'),
              'assigned_frames': {side: sum(f['players'][side] is not None for f in frames) for side in slots},
              'identity_note': 'Dense boxes and joints are fresh estimates on each original decoded frame; identities remain scout candidates, not reviewed truth.',
              'audit_pose_color_order': 'BGR unchanged production',
              'frame_map_sha256': clip['frame_map_sha256'], 'all_decoded_pixels_verified': True}
    validate_dense(result, clip, frame_map)
    bgr_path = folder / 'dense_pose.json'
    save_json(bgr_path, result)
    boxes_path = folder / 'detector_boxes_and_assignments.json'
    save_json(boxes_path, {'source_sha256': clip['source_sha256'], 'frames': assignments_saved})
    # The secondary pass now uses saved primary boxes and exact assignment order.
    saved = json.loads(boxes_path.read_text())
    rgb_frames = []
    capture = cv2.VideoCapture(str(source))
    rgb_started = time.perf_counter()
    for index, primary in enumerate(result['frames']):
        ok, image = capture.read()
        if not ok or hashlib.sha256(image.tobytes()).hexdigest() != frame_map[index]['pixel_sha256']:
            raise ValueError('RGB replay source pixel/frame mismatch')
        record = saved['frames'][index]
        if record['frame_index'] != index:
            raise ValueError('Saved primary boxes index mismatch')
        boxes = np.asarray(record['detector_boxes']).reshape(-1, 4)
        assignments = dict(record['assignments_in_order'])
        players = {'near': None, 'far': None}
        if assignments:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            points, scores = pose_model(rgb, bboxes=[boxes[i] for i in assignments.values()])
            players = serialize_players(original, boxes, assignments, points, scores, width, height)
        rgb_frames.append(dict(primary, players=players))
        for side in ['near', 'far']:
            a, b = primary['players'][side], players[side]
            if (a is None) != (b is None) or (a is not None and a['bbox'] != b['bbox']):
                raise ValueError('RGB changed primary boxes or assignments')
        if len(rgb_frames) % 100 == 0:
            print('PAIRED_RGB_PROGRESS ' + json.dumps({'sample_id': clip['sample_id'], 'processed': len(rgb_frames),
                                                      'total': total}), flush=True)
    if capture.read()[0]:
        raise ValueError('RGB replay contains unexpected extra frame')
    capture.release()
    rgb_seconds = time.perf_counter() - rgb_started
    rgb_result = dict(result, frames=rgb_frames, runtime_seconds=round(rgb_seconds, 3),
                      audit_pose_color_order='RGB secondary with saved primary BGR detector boxes and assignments',
                      primary_dense_pose_sha256=sha256(bgr_path), saved_detector_assignments_sha256=sha256(boxes_path))
    validate_dense(rgb_result, clip, frame_map)
    rgb_path = folder / 'dense_pose_rgb_same_boxes.json'
    save_json(rgb_path, rgb_result)
    bgr_summary, rgb_summary = summarize(result), summarize(rgb_result)
    summary = {'sample_id': clip['sample_id'], 'source_sha256': clip['source_sha256'],
               'frame_map_sha256': clip['frame_map_sha256'], 'N': total, 'source_width': width, 'source_height': height,
               'all_indices_and_pixels_verified': True, 'rgb_boxes_assignments_equal_bgr': True,
               'scout_candidates': [v['slot'] for v in selection['candidates']],
               'scout_runtime_seconds': selection['runtime_seconds'],
               'primary': dict(bgr_summary, path=str(bgr_path), sha256=sha256(bgr_path), runtime_seconds=round(bgr_seconds, 3)),
               'rgb_same_boxes': dict(rgb_summary, path=str(rgb_path), sha256=sha256(rgb_path), runtime_seconds=round(rgb_seconds, 3)),
               'saved_detector_assignments_sha256': sha256(boxes_path),
               'timestamp_fallbacks': timestamp_fallbacks,
               'runtime_seconds': round(time.perf_counter() - started, 3)}
    save_json(folder / 'pose_audit_summary.json', summary)
    tile = overlay_middle(middle_image, frames[total // 2], clip, folder)
    return summary, tile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--clips-manifest', type=Path)
    parser.add_argument('--context-control', action='store_true',
                        help='Separate predeclared one-clip/603-frame continuous-rally control')
    args = parser.parse_args()
    output = args.output.resolve()
    manifest_path = args.clips_manifest.resolve() if args.clips_manifest else output / 'clips_manifest.json'
    protocol_root = output.parent if args.context_control else output
    protocol_path = protocol_root / 'paired_comparison_protocol.json'
    review_path = protocol_root / 'frontend_contract_review.md'
    for required in [manifest_path, protocol_path, review_path]:
        if not required.is_file():
            raise FileNotFoundError('Required frozen audit input missing: ' + str(required))
    clips_manifest = json.loads(manifest_path.read_text())
    clips = clips_manifest['clips']
    expected_clips, expected_frames = (1, 603) if args.context_control else (20, 1271)
    if len(clips) != expected_clips or sum(c['frames'] for c in clips) != expected_frames or not clips_manifest['all_pixels_verified']:
        raise ValueError('Unexpected frozen selection or unverified clips')
    if not 1 <= args.threads <= 4:
        raise ValueError('Use a modest explicit ORT thread count')
    cv2.setNumThreads(1)
    code_hash = sha256(Path(__file__))
    weights = checked_weights()
    started = time.perf_counter()
    detector, pose_model = create_models(weights, args.threads)
    code_files = [Path(__file__), ROOT / 'analyze_pose.py', ROOT / 'specialist_pose.py', ROOT / 'auto_players.py',
                  Path(inspect.getfile(rtmlib_base)), Path(inspect.getfile(rtmlib_pose)), Path(inspect.getfile(rtmlib_detector))]
    runtime = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
               'context': 'secondary_continuous_rally' if args.context_control else 'primary_frozen20_per_clip',
               'included_in_frozen20_denominator': not args.context_control,
               'clips_manifest_sha256': sha256(manifest_path), 'protocol_sha256': sha256(protocol_path),
               'frontend_contract_review_sha256': sha256(review_path),
               'code_sha256': {str(p): sha256(p) for p in code_files}, 'weights': weights,
               'versions': {p: version(p) for p in ['numpy', 'opencv-python', 'onnxruntime', 'rtmlib', 'imageio-ffmpeg', 'Pillow']},
               'resource_config': {'onnxruntime_intra_op_threads': args.threads, 'onnxruntime_inter_op_threads': 1,
                                   'opencv_threads': 1, 'device': 'cpu', 'onnx_providers': detector.session.get_providers()},
               'config': {'scout_max_width': 960, 'scout_height': 540, 'scout_logic': 'unchanged analyze_pose.scout_automatic_players',
                          'dense_dimensions': [1280, 720], 'detector_input': [640, 640], 'detector_score_thr': .4,
                          'pose_input': [192, 256], 'primary_pose_color_order': 'BGR unchanged production',
                          'detector_shared_across_clips': True, 'pose_model_shared_across_clips': True,
                          'fresh_follower_per_clip': True, 'selection_filtering_by_classification_scores': False},
               'adaptations': ['Reuse one deterministic YOLOX/RTMPose pair across clips to avoid model reloads; follower state is reset per clip.',
                               'Read and hash existing weight manifest instead of get_weights() to avoid a production manifest write.',
                               'ORT session thread limits set only at construction; inference preprocessing/postprocessing unchanged.',
                               'Minimal coarse_pose/config contains exact serialized production scout candidates.',
                               'Dense loop/serialization copied from specialist_pose.py with strict frame count/index/pixel checks and global frame metadata.',
                               'Declared RGB secondary uses saved primary BGR detector boxes and identical assignment order; no second scout/detector/follower.'],
               'production_database_labels_weights_changed': False, 'clips': []}
    runtime_path = output / 'pose_runtime_manifest.json'
    save_json(runtime_path, runtime)
    columns = min(4, len(clips))
    rows = (len(clips) + columns - 1) // columns
    sheet = Image.new('RGB', (480 * columns, 314 * rows), 'white')
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for i, clip in enumerate(clips):
        print('PAIRED_POSE_START ' + json.dumps({'clip': i + 1, 'sample_id': clip['sample_id'], 'N': clip['frames']}), flush=True)
        summary, tile = run_clip(clip, output, detector, pose_model, weights, code_hash, ffmpeg)
        runtime['clips'].append(summary)
        runtime['runtime_seconds'] = round(time.perf_counter() - started, 3)
        save_json(runtime_path, runtime)
        sheet.paste(tile, ((i % columns) * 480, (i // columns) * 314))
        sheet.save(output / 'pose_candidate_identity_sheet.png')
        print('PAIRED_POSE_DONE ' + json.dumps(summary), flush=True)
    if sha256(manifest_path) != runtime['clips_manifest_sha256']:
        raise ValueError('Frozen clips manifest changed during replay')
    runtime.update(status='complete', total_clips=len(runtime['clips']), total_frames=sum(c['N'] for c in runtime['clips']),
                   runtime_seconds=round(time.perf_counter() - started, 3),
                   candidate_identity_sheet=str(output / 'pose_candidate_identity_sheet.png'))
    save_json(runtime_path, runtime)
    print('PAIRED_POSE_COMPLETE ' + json.dumps({'clips': len(clips), 'frames_per_condition': expected_frames,
                                              'context': runtime['context'], 'runtime_seconds': runtime['runtime_seconds']}), flush=True)


if __name__ == '__main__':
    main()
