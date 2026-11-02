"""Editable observation windows with immutable proposals and revision history.

This module does not infer stroke types, contact frames, or technical errors.
The only automatic signals are visible wrist-height features in the image.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import uuid

DATA = Path(__file__).resolve().parent.parent / '03_数据'
REVIEW_ROOT = DATA / '动作阶段'
LOCK = threading.RLock()
PHASE_LABELS = {
    'preparation': '准备与移动',
    'swing_window': '挥拍附近（触球未定位）',
    'follow_through': '随挥',
    'recovery': '落地与回位',
    'low_reach': '低位伸展',
    'unknown': '待分类',
}
QUALITY_NOTE = ('阶段是可编辑的观察窗口，允许未知、多个动作组和重叠。候选由采样画面提出，'
                '不作为独立评价真值；腕高图像特征不等于挥拍或触球。请看原速录像核对。')


class RevisionConflict(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def revision(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def review_path(pose_id):
    if not isinstance(pose_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', pose_id):
        raise ValueError('阶段记录 ID 无效')
    file = (REVIEW_ROOT / pose_id / 'review.json').resolve()
    if not file.is_relative_to(DATA.resolve()):
        raise ValueError('阶段记录路径无效')
    return file


def atomic_json(file, value):
    file = file.resolve()
    if not file.is_relative_to(DATA.resolve()):
        raise ValueError('阶段保存路径需位于数据目录内')
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(file.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(file)
    finally:
        temporary.unlink(missing_ok=True)


def wrist_signals(pose, threshold=0.25, minimum_samples=2):
    """Contiguous observed samples; no gap filling, handedness, or hit inference."""
    signals = {'near': [], 'far': []}
    frames = sorted(pose['frames'], key=lambda frame: frame['source_time'])
    step = 1 / pose['sample_fps']
    for side in signals:
        for joint_side in ['left', 'right']:
            feature = joint_side + '_wrist_height_relative_to_shoulder_torso_units'
            run = []

            def finish():
                if len(run) < minimum_samples:
                    return
                peak = max(run, key=lambda item: item[1])
                identifier = f"{pose['id']}-{side}-{joint_side}-{run[0][0]:.3f}"
                signals[side].append({
                    'id': uuid.uuid5(uuid.NAMESPACE_URL, 'badminton-wrist-signal:' + identifier).hex,
                    'kind': 'wrist_above_shoulder', 'joint_side': joint_side,
                    'start': run[0][0], 'end': round(min(pose['end'], run[-1][0] + step), 4),
                    'peak_time': peak[0], 'peak_value': peak[1], 'known_frames': len(run),
                    'threshold': threshold, 'units': 'image_height_difference_divided_by_torso_image_length',
                    'display_threshold': pose['display_threshold'],
                    'evidence_times': [item[0] for item in run],
                    'interpretation': 'Visible wrist above corresponding shoulder in image; not a contact or phase label.',
                })

            for frame in frames:
                person = frame['players'].get(side)
                value = person.get('features', {}).get(feature) if person else None
                time = frame['source_time']
                qualifies = isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= threshold
                contiguous = not run or abs(time - run[-1][0] - step) < step * 0.2
                if not qualifies or not contiguous:
                    finish()
                    run = []
                if qualifies:
                    run.append((time, value))
            finish()
    for side in signals:
        signals[side].sort(key=lambda signal: (signal['start'], signal['joint_side']))
    return signals


def empty_document(pose):
    return {
        'schema_version': 1, 'pose_id': pose['id'], 'source_video_id': pose['source_video_id'],
        'source_sha256': pose['source_sha256'], 'start': pose['start'], 'end': pose['end'],
        'sample_fps': pose['sample_fps'], 'time_basis': pose['time_basis'],
        'contact_status': 'unverified', 'phase_labels': PHASE_LABELS,
        'quality_note': QUALITY_NOTE,
        'players': {side: {'racket_hand': 'unknown', 'identity_status': 'unreviewed', 'notes': '',
                           'reviewer': '', 'phases': []} for side in ['near', 'far']},
        'created_at': pose['created_at'], 'updated_at': pose['created_at'],
        'last_edit_origin': 'initial_empty_record',
    }


def read_stored(pose):
    file = review_path(pose['id'])
    document = json.loads(file.read_text(encoding='utf-8')) if file.is_file() else empty_document(pose)
    if not isinstance(document, dict) or document.get('schema_version') != 1 or not isinstance(document.get('players'), dict):
        raise ValueError('阶段记录格式无法读取')
    if (document.get('pose_id') != pose['id'] or document.get('source_video_id') != pose['source_video_id'] or
            document.get('source_sha256') != pose['source_sha256'] or document.get('start') != pose['start'] or document.get('end') != pose['end']):
        raise RevisionConflict('原片或片段范围已变化，请先重新建立对应阶段记录')
    return document


def response_document(document, pose):
    # Signals are rebuilt from current pose output and never edited as phases.
    return dict(document, revision=revision(document), signals=wrist_signals(pose),
                pose_created_at=pose['created_at'])


def get_review(pose):
    with LOCK:
        return response_document(read_stored(pose), pose)


def text_value(value, limit=4000):
    if not isinstance(value, str):
        raise ValueError('文字字段需为字符串')
    return value.strip()[:limit]


def numeric_time(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('阶段时间需为有限数字，单位为秒')
    return round(float(value), 4)


def validate_players(body, old, pose):
    if not isinstance(body, dict) or set(body) != {'near', 'far'}:
        raise ValueError('请同时保存近端与远端记录')
    result = {}
    seen_ids = set()
    for side in ['near', 'far']:
        player = body[side]
        if not isinstance(player, dict):
            raise ValueError('球员记录格式错误')
        hand = player.get('racket_hand', 'unknown')
        identity = player.get('identity_status', 'unreviewed')
        if hand not in ['unknown', 'left', 'right'] or identity not in ['unreviewed', 'confirmed', 'uncertain']:
            raise ValueError('持拍手或身份状态无效')
        reviewer = text_value(player.get('reviewer', ''), 200)
        if (hand != 'unknown' or identity == 'confirmed') and not reviewer:
            raise ValueError('确认持拍手或身份时，请填写核对人')
        phases = player.get('phases', [])
        if not isinstance(phases, list) or len(phases) > 40:
            raise ValueError('每位球员最多保存 40 个阶段窗口')
        old_phases = {item['id']: item for item in old['players'][side]['phases']}
        clean = []
        for phase in phases:
            if not isinstance(phase, dict):
                raise ValueError('阶段字段格式错误')
            phase_id = phase.get('id', '')
            if not isinstance(phase_id, str) or (phase_id and phase_id not in old_phases):
                raise ValueError('阶段 ID 不属于当前球员；新窗口请留空 ID')
            phase_id = phase_id or uuid.uuid4().hex
            if phase_id in seen_ids:
                raise ValueError('同一个阶段 ID 不可重复保存')
            seen_ids.add(phase_id)
            label = phase.get('label', 'unknown')
            status = phase.get('status', 'candidate')
            if label not in PHASE_LABELS or status not in ['candidate', 'confirmed', 'rejected']:
                raise ValueError('阶段标签或核对状态无效')
            start, end = numeric_time(phase.get('start')), numeric_time(phase.get('end'))
            if not pose['start'] <= start < end <= pose['end']:
                raise ValueError(f"阶段窗口需位于 {pose['start']}–{pose['end']} 秒内，结束大于开始")
            phase_reviewer = text_value(phase.get('reviewer', ''), 200)
            if status in ['confirmed', 'rejected'] and not (phase_reviewer or reviewer):
                raise ValueError('人工确认或拒绝阶段时，请填写核对人')
            previous = old_phases.get(phase_id, {})
            reconfirm = phase.get('reconfirm', False)
            if not isinstance(reconfirm, bool):
                raise ValueError('重新核对标记需为布尔值')
            item = {
                'id': phase_id, 'event_id': text_value(phase.get('event_id', previous.get('event_id', '')), 100),
                'label': label, 'start': start, 'end': end, 'status': status,
                'note': text_value(phase.get('note', '')), 'reviewer': phase_reviewer,
                'origin': previous.get('origin', 'manual_added_window'),
                'evidence_times': previous.get('evidence_times', []),
                'boundary_status': 'manual_interval_review' if status == 'confirmed' else 'coarse_candidate',
                'contact_status': 'unverified',
                'last_edit_origin': 'manual_edit',
            }
            content_fields = ['event_id', 'label', 'start', 'end', 'note']
            content_changed = bool(previous) and any(item.get(key) != previous.get(key) for key in content_fields)
            if previous.get('status') in ['confirmed', 'rejected'] and content_changed and status in ['confirmed', 'rejected'] and not reconfirm:
                item['status'] = status = 'candidate'
                item['reviewer'] = ''
                item['boundary_status'] = 'coarse_candidate'
                item['review_note'] = 'Previously reviewed content changed; waiting for explicit review of the edited window.'
            if status in ['confirmed', 'rejected']:
                item['reviewer'] = phase_reviewer or reviewer
                same_review = (not content_changed and previous.get('status') == status and
                               previous.get('reviewer') == item['reviewer'] and not reconfirm)
                item['reviewed_at'] = previous.get('reviewed_at', now()) if same_review else now()
            clean.append(item)
        clean.sort(key=lambda item: (item['start'], item['end'], item['id']))
        result[side] = {'racket_hand': hand, 'identity_status': identity, 'reviewer': reviewer,
                        'notes': text_value(player.get('notes', '')), 'phases': clean}
    return result


def put_review(pose, body):
    if not isinstance(body, dict):
        raise ValueError('阶段保存请求格式错误')
    with LOCK:
        old = read_stored(pose)
        if body.get('revision') != revision(old):
            raise RevisionConflict('阶段记录已在别处更新。本页草稿保留，请重新载入后比较再保存。')
        players = validate_players(body.get('players'), old, pose)
        updated = dict(old, players=players, updated_at=now(), last_edit_origin='manual_edit')
        file = review_path(pose['id'])
        # Save the concrete previous and next documents before replacing current.
        history = file.parent / 'history' / (uuid.uuid4().hex + '.json')
        atomic_json(history, {'saved_at': updated['updated_at'], 'previous_revision': revision(old),
                              'new_revision': revision(updated), 'previous': old, 'new': updated})
        atomic_json(file, updated)
        return response_document(updated, pose)


def initialize_candidates(pose, candidates):
    """Create once offline; never overwrite existing human edits or reviews."""
    with LOCK:
        file = review_path(pose['id'])
        if file.is_file():
            return get_review(pose), False
        document = empty_document(pose)
        for side in ['near', 'far']:
            for ordinal, candidate in enumerate(candidates.get(side, [])):
                start, end = numeric_time(candidate['start']), numeric_time(candidate['end'])
                if not pose['start'] <= start < end <= pose['end'] or candidate['label'] not in PHASE_LABELS:
                    raise ValueError('候选阶段超出片段范围或标签不支持')
                identifier = f"{pose['id']}:{side}:{ordinal}:{start}:{end}"
                document['players'][side]['phases'].append({
                    'id': uuid.uuid5(uuid.NAMESPACE_URL, 'badminton-visual-phase:' + identifier).hex,
                    'event_id': candidate['event_id'], 'label': candidate['label'], 'start': start, 'end': end,
                    'status': 'candidate', 'note': candidate['note'], 'reviewer': '',
                    'origin': 'assistant_visual_sampled_frames', 'evidence_times': candidate['evidence_times'],
                    'boundary_status': 'coarse_candidate', 'contact_status': 'unverified',
                    'last_edit_origin': 'initial_assistant_proposal',
                })
        document['updated_at'] = now()
        document['last_edit_origin'] = 'initial_assistant_proposal'
        atomic_json(file, document)
        return response_document(document, pose), True
