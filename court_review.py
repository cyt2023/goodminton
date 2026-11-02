"""Stdlib bridge: persisted court drafts, OpenCV in the existing vision runtime."""
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from analysis_jobs import DATA, atomic_json

ROOT=Path(__file__).resolve().parent
STORE=DATA/'视频工作流/courts'


def vision(value):
    try:
        result=subprocess.run([str(ROOT/'.venv-vision/bin/python'),str(ROOT/'court_geometry_worker.py')],
                              input=json.dumps(value,ensure_ascii=False),text=True,capture_output=True,timeout=45)
    except subprocess.TimeoutExpired:
        raise ValueError('场地预览超时，请换一帧重试；录像仍保留')
    try: payload=json.loads(result.stdout)
    except ValueError: raise ValueError('场地预览未完成，请换一帧重试；录像仍保留')
    if result.returncode or 'error' in payload: raise ValueError(payload.get('error','场地计算未完成'))
    return payload['result']


def prepare(identifier, source, source_sha256, color, time_seconds):
    folder=STORE/identifier
    draft=vision({'operation':'preview','source_file':str(source),'folder':str(folder),
                  'court_line_color':color,'time_seconds':time_seconds})
    draft.update(source_sha256=source_sha256, updated_at=datetime.now(timezone.utc).isoformat())
    atomic_json(folder/'draft.json',draft)
    return draft


def confirm(identifier, source_sha256, body, start, end):
    draft=json.loads((STORE/identifier/'draft.json').read_text())
    if draft['source_sha256']!=source_sha256 or body.get('draft_updated_at')!=draft['updated_at']:
        raise ValueError('预览已更新，请刷新后重新确认')
    if body.get('camera_fixed') is not True:
        raise ValueError('请确认所选录像范围内机位固定；镜头移动时请缩短范围或跳过场地标定')
    record={'review_status':'confirmed','source_sha256':source_sha256,
            'image_size':draft['image_size'],'reference_frame':draft['frame_index'],
            'reference_time_seconds':draft['time_seconds'], 'court_line_color':draft['court_line_color'],
            'image_points_px':body.get('image_points_px'),
            'court_points_normalized':body.get('court_points_normalized'),
            'camera_fixed':True,'valid_start_seconds':start,'valid_end_seconds':end,
            'coordinate_system':'doubles_outer_court_x_left_to_right_y_far_to_near_0_1',
            'reviewed_at':datetime.now(timezone.utc).isoformat(), 'method':'user_confirmed_correspondences'}
    record=vision({'operation':'validate','record':record})
    atomic_json(STORE/identifier/'confirmed.json',record)
    return record
