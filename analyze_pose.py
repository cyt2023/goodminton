"""Offline YOLOX + RTMPose inference on local singles footage.

Use .venv-pose/bin/python analyze_pose.py. Inputs are local registered videos.
This is a project-specific selection/serialization/overlay pipeline using
RTMLib inference, not a reproduction of BST or a stroke-quality classifier.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import io
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import time
from urllib.request import urlopen
import uuid
import zipfile

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
from rtmlib import RTMPose, YOLOX

from auto_players import ScoutTracks

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / '03_数据'
MODELS = ROOT / 'models'
OUTPUT = DATA / '姿态分析'
DB = DATA / '研究数据.sqlite3'
WEIGHTS = {
    'detector': {
        'name': 'yolox-m-humanart',
        'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
        'input_size': [640, 640],
    },
    'pose': {
        'name': 'rtmpose-m-body7-256x192',
        'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',
        'input_size': [192, 256],
    },
}
JOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
          'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
         (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
         (13, 15), (12, 14), (14, 16)]
# OpenCV BGR; cyan for near, orange for far.
COLORS = {'near': (255, 215, 30), 'far': (30, 150, 255)}
GROUPS = {'shoulders': [5, 6], 'elbows': [7, 8], 'wrists': [9, 10],
          'hips': [11, 12], 'knees': [13, 14], 'ankles': [15, 16]}
NOTICE = ('二维姿态候选；区域由人工指定，身份未逐帧核对。关节分数及覆盖率不是准确率；'
          '未识别球种、击球时刻或技术错误。时间为文件播放时间。')
AUTO_NOTICE = ('自动筛选的二维姿态候选 A/B，身份未确认；未标定近端或远端。'
               '邻场、观众、相似服装和遮挡可能造成错选或身份切换。'
               '关节分数及覆盖率不是准确率；未识别球种、击球时刻或技术错误。')


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def get_weights(allow_download=False):
    MODELS.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for role, specification in WEIGHTS.items():
        file = MODELS / (specification['name'] + '.onnx')
        archive = MODELS / Path(specification['url']).name
        if not file.is_file():
            if not archive.is_file():
                if not allow_download:
                    raise FileNotFoundError(f'Official {role} weights are missing locally: {file.name}')
                temporary = archive.with_suffix('.zip.part')
                print(f'Downloading official {role} weights: {archive.name}', flush=True)
                try:
                    with urlopen(specification['url'], timeout=60) as response, temporary.open('wb') as target:
                        shutil.copyfileobj(response, target)
                    temporary.replace(archive)
                finally:
                    temporary.unlink(missing_ok=True)
            with zipfile.ZipFile(archive) as source:
                candidates = [name for name in source.namelist() if name.lower().endswith('.onnx')]
                if len(candidates) != 1:
                    raise ValueError(f'Expected one ONNX file in {archive.name}, got {candidates}')
                # Do not extract archive paths; copy only the chosen ONNX content.
                with source.open(candidates[0]) as stream, file.open('wb') as target:
                    shutil.copyfileobj(stream, target)
        manifest[role] = dict(specification, file=str(file), sha256=sha256(file), size_bytes=file.stat().st_size)
        if archive.exists():
            manifest[role]['archive_sha256'] = sha256(archive)
    save_json(MODELS / 'model_manifest.json', manifest)
    return manifest


def select_box(boxes, specification, previous, used, width, height):
    """Manual foot gate + size gate, then nearest anchor. No identity classifier."""
    x1, y1, x2, y2 = specification['foot_gate']
    min_height, max_height = specification.get('height_range', [0.02, 0.95])
    anchor = previous if previous is not None else np.array(specification['anchor'])
    candidates = []
    for index, box in enumerate(boxes):
        if index in used or not np.isfinite(box).all():
            continue
        point = np.array([(box[0] + box[2]) / (2 * width), box[3] / height])
        extent = (box[3] - box[1]) / height
        if x1 <= point[0] <= x2 and y1 <= point[1] <= y2 and min_height <= extent <= max_height:
            candidates.append((float(np.linalg.norm(point - anchor)), index, point))
    candidates.sort(key=lambda item: item[0])
    if not candidates:
        return None, None, 'no_candidate_in_manual_gate', 0
    if candidates[0][0] > 0.18:
        return None, None, 'candidate_too_far_from_anchor', len(candidates)
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.025:
        return None, None, 'ambiguous_candidates', len(candidates)
    return candidates[0][1], candidates[0][2], 'selected_by_manual_gate', len(candidates)


def image_angle(a, b, c):
    left, right = a - b, c - b
    norm = np.linalg.norm(left) * np.linalg.norm(right)
    if norm < 1e-6:
        return None
    return round(math.degrees(math.acos(float(np.clip(np.dot(left, right) / norm, -1, 1)))), 2)


def image_features(points, scores, threshold):
    """Image-plane descriptors only. No biomechanics or action-quality labels."""
    result = {}
    torso = None
    if all(scores[i] >= threshold for i in [5, 6, 11, 12]):
        shoulder_mid = (points[5] + points[6]) / 2
        hip_mid = (points[11] + points[12]) / 2
        torso = float(np.linalg.norm(shoulder_mid - hip_mid))
        result['hip_midpoint_image_px'] = [round(float(v), 2) for v in hip_mid]
    for side, shoulder, elbow, wrist in [('left', 5, 7, 9), ('right', 6, 8, 10)]:
        if all(scores[i] >= threshold for i in [shoulder, elbow, wrist]):
            result[side + '_elbow_image_angle_deg'] = image_angle(points[shoulder], points[elbow], points[wrist])
        else:
            result[side + '_elbow_image_angle_deg'] = None
        if torso is not None and torso > 1 and scores[wrist] >= threshold:
            # Image y increases downwards. Positive means wrist is higher in image.
            result[side + '_wrist_height_relative_to_shoulder_torso_units'] = round(float((points[shoulder, 1] - points[wrist, 1]) / torso), 3)
        else:
            result[side + '_wrist_height_relative_to_shoulder_torso_units'] = None
    return result


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        block = stream.read(size - len(data))
        if not block:
            if data:
                raise RuntimeError('Incomplete decoded frame')
            return None
        data.extend(block)
    return data


def draw_overlay(image, players, source_time, threshold, scale_xy, automatic=False):
    output = image.copy()
    sx, sy = scale_xy
    cv2.rectangle(output, (0, 0), (output.shape[1], 30), (20, 25, 30), -1)
    cv2.putText(output, f'Source {source_time:.2f}s | 2D pose candidates | threshold {threshold:.2f}',
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
    for side, player in players.items():
        color = COLORS[side]
        if player is None:
            continue
        bbox = player['bbox']
        b = [int(round(v * (sx if i % 2 == 0 else sy))) for i, v in enumerate(bbox)]
        cv2.rectangle(output, tuple(b[:2]), tuple(b[2:]), color, 1)
        label = ('CANDIDATE A' if side == 'near' else 'CANDIDATE B') if automatic else side.upper()
        cv2.putText(output, label + ' / UNREVIEWED', (b[0], max(43, b[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)
        joints = player['keypoints']
        for a, b in EDGES:
            if joints[a]['score'] >= threshold and joints[b]['score'] >= threshold and joints[a]['in_frame'] and joints[b]['in_frame']:
                p1 = (int(joints[a]['x'] * sx), int(joints[a]['y'] * sy))
                p2 = (int(joints[b]['x'] * sx), int(joints[b]['y'] * sy))
                cv2.line(output, p1, p2, color, 2, cv2.LINE_AA)
        for joint in joints:
            if joint['score'] >= threshold and joint['in_frame']:
                cv2.circle(output, (int(joint['x'] * sx), int(joint['y'] * sy)), 3, color, -1, cv2.LINE_AA)
    return output


def contact_sheet(images, file):
    if not images:
        return
    thumb_width = 420
    thumb_height = round(images[0][1].shape[0] * thumb_width / images[0][1].shape[1])
    cell_height = thumb_height + 25
    sheet = Image.new('RGB', (thumb_width * 3, cell_height * math.ceil(len(images) / 3)), '#f3f5f2')
    painter = ImageDraw.Draw(sheet)
    for index, (timestamp, frame) in enumerate(images):
        thumb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((thumb_width, thumb_height))
        x, y = index % 3 * thumb_width, index // 3 * cell_height
        sheet.paste(thumb, (x, y))
        painter.text((x + 5, y + thumb_height + 4), f'Source {timestamp:.2f}s | identity unreviewed', fill='#263b37')
    sheet.save(file, quality=92)


def summarize(frames, threshold, configured_sides):
    summary = {}
    configured_sides = set(configured_sides)
    for side in ['near', 'far']:
        if side not in configured_sides:
            summary[side] = {
                'configured': False, 'status': 'not_configured',
                'total_frames': len(frames), 'assigned_frames': 0,
                'missing_frames': None, 'usable_frames': 0,
                'median_joint_score': None,
                'joint_group_coverage': {group: None for group in GROUPS},
            }
            continue
        players = [frame['players'][side] for frame in frames if frame['players'][side] is not None]
        scores = [joint['score'] for player in players for joint in player['keypoints']]
        summary[side] = {
            'configured': True, 'status': 'processed',
            'total_frames': len(frames),
            'assigned_frames': len(players),
            'missing_frames': len(frames) - len(players),
            'usable_frames': sum(player['usable'] for player in players),
            'median_joint_score': round(float(np.median(scores)), 3) if scores else None,
            'joint_group_coverage': {
                group: round(sum(all(player['keypoints'][i]['score'] >= threshold and player['keypoints'][i]['in_frame'] for i in indices) for player in players) / len(frames), 3)
                if frames else 0 for group, indices in GROUPS.items()
            },
        }
    return summary


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    return float(value)


def validate_configuration(configuration):
    if not isinstance(configuration, dict):
        raise ValueError('Configuration must be an object')
    fps = finite_number(configuration.get('sample_fps'), 'sample_fps')
    threshold = finite_number(configuration.get('display_threshold'), 'display_threshold')
    detector_threshold = finite_number(configuration.get('detector_threshold'), 'detector_threshold')
    max_width = finite_number(configuration.get('max_width'), 'max_width')
    if not 0 < fps <= 30 or threshold < 0 or not 0 <= detector_threshold <= 1:
        raise ValueError('Invalid sampling frequency or score threshold')
    if not max_width.is_integer() or not 2 <= max_width <= 3840:
        raise ValueError('max_width must be an integer between 2 and 3840')
    if not isinstance(configuration.get('publish_results', True), bool):
        raise ValueError('publish_results must be a boolean')


def validate_job(job):
    if not isinstance(job, dict) or not isinstance(job.get('id'), str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', job['id']):
        raise ValueError('Invalid job ID')
    start, end = finite_number(job.get('start'), 'start'), finite_number(job.get('end'), 'end')
    mode = job.get('mode', 'manual')
    if mode not in {'manual', 'auto'}:
        raise ValueError('Analysis mode must be manual or auto')
    limit = 1800 if mode == 'auto' else 30
    if start < 0 or not 0 < end - start <= limit or (mode == 'auto' and start != 0):
        raise ValueError(f'Invalid clip interval for {mode} analysis (maximum {limit} seconds)')
    if not isinstance(job.get('label'), str):
        raise ValueError('Job label must be a string')
    if mode == 'auto':
        if job.get('players') not in (None, {}):
            raise ValueError('Automatic analysis does not accept manually configured players')
        return
    players = job.get('players')
    if not isinstance(players, dict) or not players or not set(players).issubset({'near', 'far'}):
        raise ValueError('Select at least one player side: near or far')
    for side, specification in players.items():
        if not isinstance(specification, dict):
            raise ValueError(f'{side} player configuration must be an object')
        for key, length in [('foot_gate', 4), ('anchor', 2), ('height_range', 2)]:
            values = specification.get(key, [0.02, 0.95] if key == 'height_range' else None)
            if not isinstance(values, (list, tuple)) or len(values) != length:
                raise ValueError(f'{side}.{key} needs {length} normalized coordinates')
            numbers = [finite_number(value, f'{side}.{key}') for value in values]
            if any(value < 0 or value > 1 for value in numbers):
                raise ValueError(f'{side}.{key} must lie in [0, 1]')
            if key == 'foot_gate' and not (numbers[0] < numbers[2] and numbers[1] < numbers[3]):
                raise ValueError(f'{side}.foot_gate must have positive width and height')
            if key == 'height_range' and not numbers[0] < numbers[1]:
                raise ValueError(f'{side}.height_range must increase')


def registered_source(job):
    """Resolve either a registered video ID or the legacy numbered inventory."""
    video_id = job.get('video_id')
    if not video_id:
        inventory = json.loads((DATA / '录像清单.json').read_text(encoding='utf-8'))
        legacy = next((item for item in inventory if item['number'] == job.get('video_number')), None)
        if legacy is None:
            raise ValueError('Legacy video number is not present in the inventory')
        video_id = legacy['video_id']
    if not isinstance(video_id, str):
        raise ValueError('Video ID must be a string')
    with sqlite3.connect(DB, timeout=30) as db:
        row = db.execute('SELECT path, name FROM videos WHERE id=?', (video_id,)).fetchone()
    if row is None:
        raise ValueError('Video has not been registered')
    if not isinstance(row[0], str) or not row[0]:
        raise ValueError('Registered video path is invalid')
    source = (DATA / row[0]).resolve()
    if not source.is_relative_to(DATA.resolve()) or not source.is_file():
        raise ValueError('Registered video path is invalid')
    return source, {'video_id': video_id, 'name': row[1] or source.name}


def probe_source(source, ffmpeg, start):
    """Read OpenCV stream metadata, then probe FFmpeg's autorotated display frame.

    Frame count / nominal FPS is only an approximate duration, especially for
    variable-frame-rate files. The PNG dimensions come from the same FFmpeg
    executable and default autorotation used by the physical inference stream.
    """
    capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
    try:
        if not capture.isOpened():
            raise ValueError('OpenCV could not open the registered video')
        auto_requested = bool(capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1))
        metadata_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        metadata_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        rotation = capture.get(cv2.CAP_PROP_ORIENTATION_META)
        ok, first = capture.read()
        if not ok or first is None or first.ndim != 3:
            raise ValueError('OpenCV could not decode a frame from the registered video')
        capture_height, capture_width = first.shape[:2]
        backend_name = capture.getBackendName()
    finally:
        capture.release()
    fps = float(source_fps) if math.isfinite(source_fps) and source_fps > 0 else None
    count = int(round(frame_count)) if math.isfinite(frame_count) and frame_count > 0 else None
    duration = count / fps if count is not None and fps is not None else None
    probe_command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-ss', str(start), '-i', str(source),
                     '-map', '0:v:0', '-frames:v', '1', '-an', '-f', 'image2pipe', '-vcodec', 'png', 'pipe:1']
    completed = subprocess.run(probe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if completed.returncode != 0 or not completed.stdout:
        message = completed.stderr.decode('utf-8', errors='replace')[-1200:]
        raise ValueError('FFmpeg could not decode a display frame at clip start: ' + message)
    with Image.open(io.BytesIO(completed.stdout)) as first_display:
        display_width, display_height = first_display.size
    if display_width < 2 or display_height < 2:
        raise ValueError('Decoded display frame is too small')
    return {
        'method': 'opencv_ffmpeg_metadata_and_first_frame_then_ffmpeg_autorotated_png_at_clip_start',
        'display_width': display_width, 'display_height': display_height,
        'opencv_decoded_width': capture_width, 'opencv_decoded_height': capture_height,
        'opencv_property_width': float(metadata_width) if math.isfinite(metadata_width) else None,
        'opencv_property_height': float(metadata_height) if math.isfinite(metadata_height) else None,
        'opencv_backend': backend_name, 'opencv_autorotation_requested': auto_requested,
        'opencv_rotation_metadata_deg': float(rotation) if math.isfinite(rotation) else None,
        'opencv_and_ffmpeg_display_dimensions_match': [capture_width, capture_height] == [display_width, display_height],
        'source_fps': fps, 'source_frame_count': count,
        'duration_seconds_approximate': round(duration, 6) if duration is not None else None,
        'duration_method': 'opencv_frame_count_divided_by_nominal_fps_approximate',
        'orientation': 'ffmpeg_default_autorotation_display_pixels',
        'probe_command': probe_command,
    }


def update_analysis_index(item):
    """Cross-process read/modify/write lock shared by every CLI invocation."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    index_file = OUTPUT / 'analysis_index.json'
    with (OUTPUT / '.index.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            index = json.loads(index_file.read_text(encoding='utf-8')) if index_file.exists() else []
            if not isinstance(index, list):
                raise ValueError('Analysis index must be a list')
            index = [old for old in index if old.get('id') != item['id']] + [item]
            save_json(index_file, index)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def scout_automatic_players(job, source, ffmpeg, detector, folder, width, height,
                            source_dimensions, expected_pose_frames):
    """Sample the entire file before assigning any athlete candidate.

    The first pass is bounded to about 360 frames even on the maximum 30-minute
    upload, normally at <=1 Hz (up to 5 Hz on clips shorter than 3 seconds).
    All assignments remain unconfirmed human-detection candidates.
    """
    duration = job['end'] - job['start']
    interval = max(.2, min(1.0, duration / 3), duration / 360)
    scout_fps = 1 / interval
    expected_scout_frames = max(1, math.ceil(duration * scout_fps - 1e-9))
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-ss', str(job['start']), '-i', str(source),
               '-map', '0:v:0', '-t', str(duration), '-vf', f'fps={scout_fps},scale={width}:{height}',
               '-frames:v', str(expected_scout_frames), '-an', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-threads', '1', 'pipe:1']
    tracker = ScoutTracks(interval)
    started = time.perf_counter()
    print('PROGRESS ' + json.dumps({'stage': 'scouting', 'processed_frames': 0,
                                    'total_frames': expected_pose_frames, 'scout_processed_frames': 0,
                                    'scout_total_frames': expected_scout_frames}), flush=True)
    with (folder / 'scout_decode.log').open('wb') as log:
        decoder = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)
        try:
            for index in range(expected_scout_frames):
                raw = read_exact(decoder.stdout, width * height * 3)
                if raw is None:
                    break
                image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
                boxes = np.asarray(detector(image)).reshape(-1, 4)
                tracker.observe(image, boxes, job['start'] + index * interval)
                if (index + 1) % 5 == 0 or index + 1 == expected_scout_frames:
                    print('PROGRESS ' + json.dumps({'stage': 'scouting', 'processed_frames': 0,
                                                    'total_frames': expected_pose_frames,
                                                    'scout_processed_frames': index + 1,
                                                    'scout_total_frames': expected_scout_frames}), flush=True)
            decoder.stdout.close()
            code = decoder.wait(timeout=30)
            if code != 0 or tracker.sample_count == 0:
                raise RuntimeError(f'Automatic scouting could not decode the video: {tracker.sample_count} frames, ffmpeg {code}')
        finally:
            if decoder.poll() is None:
                decoder.terminate()
                try:
                    decoder.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    decoder.kill()
                    decoder.wait(timeout=10)
    follower, report = tracker.select(job['id'], folder, source_dimensions)
    report.update({'decode_command': command, 'scout_requested_frames': expected_scout_frames,
                   'runtime_seconds': round(time.perf_counter() - started, 3),
                   'time_basis': 'nominal_low_frequency_resampled_video_playback_seconds',
                   'precision_note': 'Scout source_time is a nominal resampling timestamp, not the original decoded frame PTS.'})
    save_json(folder / 'automatic_player_selection.json', report)
    return follower, report


def process_job(job, configuration, manifest, detector, pose):
    validate_job(job)
    validate_configuration(configuration)
    automatic = job.get('mode') == 'auto'
    job = dict(job, players={} if automatic else
               {side: dict(specification, height_range=specification.get('height_range', [0.02, 0.95]))
                for side, specification in job['players'].items()})
    source, source_record = registered_source(job)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    source_probe = probe_source(source, ffmpeg, job['start'])
    source_width, source_height = source_probe['display_width'], source_probe['display_height']
    width = min(source_width, int(configuration['max_width'])) // 2 * 2
    height = max(2, round(source_height * width / source_width / 2) * 2)
    scale_back = np.array([source_width / width, source_height / height])
    fps, threshold = configuration['sample_fps'], configuration['display_threshold']
    expected_frames = max(1, math.ceil((job['end'] - job['start']) * fps - 1e-9))
    folder = OUTPUT / job['id']
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    automatic_selection, automatic_follower = None, None
    if automatic:
        automatic_follower, automatic_selection = scout_automatic_players(
            job, source, ffmpeg, detector, folder, width, height,
            (source_width, source_height), expected_frames)
        job['players'] = {candidate['slot']: {'automatic': True, 'scout_track_id': candidate['scout_track_id'],
                                            'display_label': candidate['display_label'], 'court_side': None}
                          for candidate in automatic_selection['candidates']}
    decode_cmd = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-ss', str(job['start']), '-i', str(source),
                  '-map', '0:v:0', '-t', str(job['end'] - job['start']), '-vf', f'fps={fps},scale={width}:{height}',
                  '-frames:v', str(expected_frames),
                  '-an', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-threads', '1', 'pipe:1']
    overlay_path = folder / 'overlay.mp4'
    overlay_temporary = folder / 'overlay.partial.mp4'
    encode_cmd = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                  '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                  '-preset', 'veryfast', '-crf', '21', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                  '-threads', '2', str(overlay_temporary)]
    frames, sheet_images = [], []
    anchors = {side: None for side in job['players']}
    misses = {side: 0 for side in job['players']}
    selected_sheet = {min(expected_frames - 1, round(i * (expected_frames - 1) / 11)) for i in range(12)}
    inference_seconds = 0.0
    print(f"Processing {job['id']} ({expected_frames} frames at {fps} fps)", flush=True)
    print('PROGRESS ' + json.dumps({'stage': 'pose', 'processed_frames': 0, 'total_frames': expected_frames}), flush=True)
    with (folder / 'decode.log').open('wb') as decode_log, (folder / 'encode.log').open('wb') as encode_log:
        decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=decode_log)
        encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=encode_log)
        try:
            for index in range(expected_frames):
                raw = read_exact(decoder.stdout, width * height * 3)
                if raw is None:
                    break
                image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
                source_time = round(job['start'] + index / fps, 4)
                t0 = time.perf_counter()
                boxes = np.asarray(detector(image)).reshape(-1, 4)
                players = {'near': None, 'far': None}
                assignments, used = {}, set()
                reasons = {side: {'status': 'not_configured', 'candidate_count': 0} for side in ['near', 'far']}
                if automatic:
                    assignments, reasons = automatic_follower.select_boxes(image, boxes, source_time)
                else:
                    for side, specification in job['players'].items():
                        chosen, point, reason, candidate_count = select_box(boxes, specification, anchors[side], used, width, height)
                        reasons[side] = {'status': reason, 'candidate_count': candidate_count}
                        if chosen is not None:
                            assignments[side] = chosen
                            used.add(chosen)
                            anchors[side] = point
                            misses[side] = 0
                        else:
                            misses[side] += 1
                            if misses[side] >= 10:
                                anchors[side] = None
                if assignments:
                    # Guard this call: RTMLib otherwise estimates a whole-frame
                    # person when given an empty list of bounding boxes.
                    chosen_boxes = [boxes[index] for index in assignments.values()]
                    keypoints, joint_scores = pose(image, bboxes=chosen_boxes)
                    for pose_index, (side, box_index) in enumerate(assignments.items()):
                        points = np.asarray(keypoints[pose_index]) * scale_back
                        scores = np.asarray(joint_scores[pose_index])
                        valid = np.isfinite(points).all(axis=1) & np.isfinite(scores)
                        scores = np.where(valid, scores, 0)
                        points = np.where(np.isfinite(points), points, 0)
                        in_frame = valid & (points[:, 0] >= 0) & (points[:, 0] < source_width) & (points[:, 1] >= 0) & (points[:, 1] < source_height)
                        eligible = in_frame & (scores >= threshold)
                        usable_count = int(sum(eligible))
                        # This definition measures completeness, not correctness.
                        usable = all(eligible[i] for i in [5, 6, 11, 12, 15, 16])
                        flags = ['automatic_candidate_track' if automatic else 'manual_gate', 'identity_not_frame_reviewed']
                        if automatic:
                            flags.append('court_side_unknown')
                        if usable_count < 17:
                            flags.append('low_keypoint_score')
                        if not in_frame.all():
                            flags.append('keypoint_out_of_frame')
                        if not valid.all():
                            flags.append('nonfinite_model_output_replaced_with_zero')
                        bbox = boxes[box_index] * np.tile(scale_back, 2)
                        if (bbox[3] - bbox[1]) < 100:
                            flags.append('small_person_under_100px')
                        players[side] = {
                            'track_id': (f"V{job['video_number']:02d}-{side}" if not job.get('video_id') and isinstance(job.get('video_number'), int)
                                         else f"{job['id']}-{side}"), 'detected': True,
                            'bbox': [round(float(v), 2) for v in bbox],
                            'bbox_score': None,  # RTMLib YOLOX API returns boxes only.
                            'keypoints': [{'name': name, 'x': round(float(point[0]), 2), 'y': round(float(point[1]), 2),
                                           'score': round(float(score), 4), 'in_frame': bool(inside), 'finite_model_output': bool(finite)}
                                          for name, point, score, inside, finite in zip(JOINTS, points, scores, in_frame, valid)],
                            'usable_joint_count': usable_count, 'usable': bool(usable),
                            'features': image_features(points, np.where(in_frame, scores, 0), threshold),
                            'quality_flags': flags,
                        }
                        if automatic:
                            players[side]['display_label'] = job['players'][side]['display_label']
                            players[side]['scout_track_id'] = job['players'][side]['scout_track_id']
                            players[side]['court_side'] = None
                inference_seconds += time.perf_counter() - t0
                frame = {'sample_index': index, 'source_time': source_time, 'overlay_time': round(index / fps, 4),
                         'players': players, 'assignment': reasons, 'detected_people_count': len(boxes),
                         'all_detected_bboxes_original_px': [[round(float(v), 2) for v in box * np.tile(scale_back, 2)] for box in boxes if np.isfinite(box).all()]}
                frames.append(frame)
                overlay = draw_overlay(image, players, source_time, threshold, (width / source_width, height / source_height), automatic)
                encoder.stdin.write(overlay.tobytes())
                if index in selected_sheet:
                    sheet_images.append((source_time, overlay.copy()))
                    cv2.imwrite(str(folder / f'frame_{index:04d}.jpg'), overlay)
                if (index + 1) % 10 == 0:
                    counts = {side: sum(f['players'][side] is not None for f in frames) for side in players}
                    print(f"{job['id']}: {index + 1}/{expected_frames}, assigned {counts}", flush=True)
                    print('PROGRESS ' + json.dumps({'stage': 'pose', 'processed_frames': index + 1, 'total_frames': expected_frames}), flush=True)
            print('PROGRESS ' + json.dumps({'stage': 'pose', 'processed_frames': len(frames), 'total_frames': expected_frames}), flush=True)
            decoder.stdout.close()
            encoder.stdin.close()
            decode_code, encode_code = decoder.wait(timeout=30), encoder.wait(timeout=30)
            missing_tail = expected_frames - len(frames)
            tail_allowed = automatic and 0 <= missing_tail <= 2 and len(frames) > 0
            if decode_code != 0 or encode_code != 0 or (len(frames) != expected_frames and not tail_allowed):
                raise RuntimeError(f'Incomplete clip: {len(frames)}/{expected_frames}; ffmpeg return codes {decode_code}, {encode_code}')
        finally:
            for process in [decoder, encoder]:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
    overlay_temporary.replace(overlay_path)
    elapsed = time.perf_counter() - started
    summary = summarize(frames, threshold, job['players'])
    quality_note = AUTO_NOTICE if automatic else NOTICE
    clip_complete = len(frames) == expected_frames
    result = {
        'id': job['id'], 'source_video_id': source_record['video_id'], 'source_name': source_record['name'],
        'source_sha256': sha256(source), 'source_width': source_width, 'source_height': source_height,
        'source_media_probe': source_probe,
        'coordinate_orientation': 'ffmpeg_autorotated_display_frame',
        'source_duration_seconds_approximate': source_probe['duration_seconds_approximate'],
        'processed_width': width, 'processed_height': height, 'start': job['start'], 'end': job['end'],
        'analysis_mode': 'auto' if automatic else 'manual',
        'requested_end': job['end'], 'processed_end': round(job['start'] + len(frames) / fps, 6),
        'requested_frame_count': expected_frames, 'decoded_frame_count': len(frames),
        'clip_complete': clip_complete, 'completion_status': 'complete' if clip_complete else 'source_ended_early',
        'sample_fps': fps, 'time_basis': 'nominal_resampled_video_playback_seconds',
        'timestamp_note': 'FFmpeg fps resamples decoded presentation frames. source_time is the nominal output sampling time; selected original frame PTS was not saved. No precise contact time or physical movement speed inferred.',
        'coordinates': 'original_full_display_frame_pixels_after_autorotation', 'keypoint_schema': 'coco17',
        'display_threshold': threshold, 'usable_definition': 'both shoulders, both hips, both ankles finite, in frame, and score >= display_threshold; completeness only, does not require wrists or elbows',
        'models': manifest, 'backend': 'onnxruntime', 'device': 'cpu',
        'execution_providers': {'detector': detector.session.get_providers(), 'pose': pose.session.get_providers()},
        'package_versions': {name: version(name) for name in ['rtmlib', 'onnxruntime', 'numpy', 'opencv-python', 'imageio-ffmpeg']},
        'player_assignment': ('automatic_scout_tracks_then_position_scale_and_torso_hsv_following'
                              if automatic else 'manual_normalized_foot_and_height_gates_then_nearest_anchor'),
        'player_config': job['players'], 'review_status': 'identity_and_joints_not_frame_reviewed',
        'detector_threshold': configuration['detector_threshold'],
        'bbox_score_note': 'The RTMLib YOLOX API returns bounding boxes without individual detection scores; missing values are null.',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'runtime_seconds': round(elapsed, 3), 'inference_seconds': round(inference_seconds, 3),
        'inference_seconds_note': 'Detector and pose calls in the pose pass only; scouting and file I/O excluded.',
        'pipeline_frames_per_second': round(len(frames) / elapsed, 3),
        'decode_command': decode_cmd, 'encode_command': encode_cmd,
        'overlay_has_audio': False, 'smoothing': None, 'interpolation': None,
        'quality_note': quality_note, 'summary': summary, 'frames': frames,
    }
    if automatic:
        result['automatic_player_selection'] = automatic_selection
        result['scout_runtime_seconds'] = automatic_selection['runtime_seconds']
    save_json(folder / 'result.json', result)
    contact_sheet(sheet_images, folder / 'contact_sheet.jpg')
    overlay_id = uuid.uuid5(uuid.NAMESPACE_URL, 'badminton-pose-overlay:' + job['id']).hex
    item = {'id': job['id'], 'label': job['label'], 'source_video_id': source_record['video_id'],
            'start': job['start'], 'end': job['end'], 'overlay_video_id': overlay_id,
            'result_file': str((folder / 'result.json').relative_to(DATA)), 'sample_fps': fps,
            'summary': summary, 'runtime_seconds': result['runtime_seconds'], 'quality_note': quality_note,
            'analysis_mode': result['analysis_mode'], 'processed_end': result['processed_end'],
            'requested_frame_count': expected_frames, 'decoded_frame_count': len(frames),
            'clip_complete': clip_complete, 'completion_status': result['completion_status']}
    if configuration.get('publish_results', True):
        with sqlite3.connect(DB, timeout=30) as db:
            db.execute('INSERT OR REPLACE INTO videos (id,name,kind,path) VALUES (?,?,?,?)',
                       (overlay_id, job['label'] + '（骨架叠加）', 'analysis', str(overlay_path.relative_to(DATA))))
        update_analysis_index(item)
    print('RESULT ' + json.dumps({'id': job['id'], 'result_id': job['id'],
                                  'runtime_seconds': result['runtime_seconds'], 'summary': summary}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'pose_jobs.json')
    parser.add_argument('--job', action='append', help='Process only the selected job IDs')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--probe-video', help='Probe a registered video ID without loading or downloading models')
    args = parser.parse_args()
    if args.probe_video:
        source, source_record = registered_source({'video_id': args.probe_video})
        metadata = probe_source(source, imageio_ffmpeg.get_ffmpeg_exe(), 0)
        response = dict(metadata, video_id=source_record['video_id'], name=source_record['name'],
                        width=metadata['display_width'], height=metadata['display_height'],
                        duration=metadata['duration_seconds_approximate'], fps=metadata['source_fps'],
                        duration_is_approximate=True,
                        coordinates='original_full_display_frame_pixels_after_autorotation')
        print('MEDIA ' + json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)
        return
    config = json.loads(args.config.read_text(encoding='utf-8'))
    validate_configuration(config)
    if not isinstance(config.get('jobs'), list) or not config['jobs']:
        raise ValueError('Configuration needs a nonempty jobs list')
    for job in config['jobs']:
        validate_job(job)
    models = get_weights(allow_download=args.download_only)
    if args.download_only:
        return
    detector = YOLOX(models['detector']['file'], model_input_size=tuple(models['detector']['input_size']),
                     score_thr=config['detector_threshold'], backend='onnxruntime', device='cpu')
    pose = RTMPose(models['pose']['file'], model_input_size=tuple(models['pose']['input_size']),
                   to_openpose=False, backend='onnxruntime', device='cpu')
    for job in config['jobs']:
        if not args.job or job['id'] in args.job:
            process_job(job, config, models, detector, pose)


if __name__ == '__main__':
    main()
