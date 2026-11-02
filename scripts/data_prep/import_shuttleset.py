"""Import twelve cited, unreviewed stroke references; never acquire match media.

Sources are pinned to the BST author's correction commit.  The imported rows
remain dataset annotations until somebody watches and reviews the source video.
Only missing stable IDs are inserted, so a repeat run preserves user reviews.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/data_prep/
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / '03_数据'
DB = DATA / '研究数据.sqlite3'
SOURCE_DIR = DATA / '职业示范索引' / 'ShuttleSet_BST'
COMMIT = 'fb9b310bf4c8a8e3d89c75e61bc06a7ac3de62df'
UPSTREAM_COMMIT = '45517f7d4cb936b03f3eabf939cc7959d39226fe'
REPO = 'Va6lue/BST-Badminton-Stroke-type-Transformer'
BASE = f'https://raw.githubusercontent.com/{REPO}/{COMMIT}'
MATCH_ID = 13
SET_NUMBER = 1
MATCH_FOLDER = 'Kento_MOMOTA_Viktor_AXELSEN_Malaysia_Masters_2020_Finals'
SOURCE_FILES = {
    'match.csv': f'{BASE}/ShuttleSet/set/match.csv',
    'match13_set1.csv': f'{BASE}/ShuttleSet/set/{MATCH_FOLDER}/set1.csv',
    'flaw_shot_records.csv': f'{BASE}/ShuttleSet/flaw_shot_records.csv',
    'BST_LICENSE.txt': f'{BASE}/LICENSE',
    'ShuttleSet_LICENSE.txt': f'https://raw.githubusercontent.com/wywyWang/CoachAI-Projects/{UPSTREAM_COMMIT}/LICENSE',
}
SHOT_TYPES = {
    'clear': ('長球', '高远球'),
    'drop': ('切球', '吊球'),
    'smash': ('殺球', '杀球'),
}
MAX_SOURCE_BYTES = 4 * 1024 * 1024


def read_source(filename, url):
    file = SOURCE_DIR / filename
    if file.exists():
        raw = file.read_bytes()
    else:
        request = urllib.request.Request(url, headers={'User-Agent': 'BadmintonResearchMetadata/1.0'})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(MAX_SOURCE_BYTES + 1)
        if len(raw) > MAX_SOURCE_BYTES:
            raise ValueError(f'Source metadata unexpectedly large: {filename}')
        temporary = file.with_suffix(file.suffix + '.tmp')
        temporary.write_bytes(raw)
        temporary.replace(file)
    if not raw or len(raw) > MAX_SOURCE_BYTES:
        raise ValueError(f'Invalid source metadata size: {filename}')
    info = {
        'file': str(file.relative_to(DATA)),
        'url': url,
        'sha256': hashlib.sha256(raw).hexdigest(),
        'bytes': len(raw),
    }
    return raw.decode('utf-8-sig'), info


def integer(value):
    return int(float(value))


def seconds(time_text):
    h, m, s = time_text.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def row_key(row):
    return integer(row['rally']), integer(row['ball_round'])


def excluded_rows(flaw_rows):
    excluded = set()
    for row in flaw_rows:
        if integer(row['match']) != MATCH_ID or row['measure'].strip() != 'removed':
            continue
        if integer(row['set']) == 0:
            raise ValueError('The selected match is excluded in the author correction list')
        if integer(row['set']) == SET_NUMBER:
            excluded.add((integer(row['rally']), integer(row['ball_round'])))
    return excluded


def select_four(rows, original_type, excluded):
    candidates = [row for row in rows
                  if row['type'] == original_type and row['player'] in ('A', 'B')
                  and row_key(row) not in excluded]
    selected, rallies, keys = [], set(), set()
    # Alternate players and favour a different rally for each selected example.
    for wanted_player in ('A', 'B', 'A', 'B'):
        remaining = [row for row in candidates if row_key(row) not in keys]
        options = [row for row in remaining
                   if row['player'] == wanted_player and integer(row['rally']) not in rallies]
        if not options:
            options = [row for row in remaining if integer(row['rally']) not in rallies]
        if not options:
            options = [row for row in remaining if row['player'] == wanted_player]
        if not options:
            options = remaining
        if not options:
            raise ValueError(f'Fewer than four references available for {original_type}')
        row = options[0]
        selected.append(row)
        keys.add(row_key(row))
        rallies.add(integer(row['rally']))
    return selected


def build_payload(row, match, condition_stroke, display_type, sources):
    rally, ball_round = row_key(row)
    player_name = match['winner'] if row['player'] == 'A' else match['loser']
    contact_second = seconds(row['time'])
    start, end = max(0, contact_second - 2), contact_second + 2
    identity = f'https://github.com/wywyWang/CoachAI-Projects/ShuttleSet#match={MATCH_ID}&set={SET_NUMBER}&rally={rally}&ball_round={ball_round}'
    reference_id = uuid.uuid5(uuid.NAMESPACE_URL, identity).hex
    payload = {
        'video_id': '',
        'source_url': match['url'],
        'start': start,
        'end': end,
        'title': f'{player_name}｜{display_type}（原标签：{row["type"]}）｜第 {SET_NUMBER} 局回合 {rally} 第 {ball_round} 拍',
        'conditions': {
            'stroke': condition_stroke,
            'corner': 'unknown',
            'incoming': 'unknown',
            'arrival': 'unknown',
            'balance': 'unknown',
        },
        'focus': ['contact'],
        'reviewed': False,
        'target_player': 'unconfirmed',
        'reviewer': '',
        'question': '',
        'intent': '',
        'teaching_note': '',
        'observation': f'ShuttleSet 经 BST 作者修正的标注在原视频 {row["time"]} 记录 {player_name} 的「{row["type"]}」，第 {SET_NUMBER} 局回合 {rally} 第 {ball_round} 拍。尚未观看录像核对动作或时间。',
        'limits': '外链素材，未下载比赛媒体。原 time 字段为整秒；前后各 2 秒是系统人为设置的展示窗口，不是专家动作边界。动作阶段、技术正确性和对个人问题的适用性尚未核对。空标注和高度编码保留原始值，不推断到位或平衡条件。',
        'evidence_source': 'ShuttleSet_BST_dataset_annotation',
        'source_metadata': {
            'dataset': 'ShuttleSet',
            'correction_repository': f'https://github.com/{REPO}',
            'commit': COMMIT,
            'match_id': MATCH_ID,
            'set_number': SET_NUMBER,
            'original_row': row,
            'original_time': row['time'],
            'original_frame_num': row['frame_num'],
            'original_player_code': row['player'],
            'player_name': player_name,
            'original_match_row': match,
            'source_url': sources['match13_set1.csv']['url'],
            'csv_sha256': sources['match13_set1.csv']['sha256'],
            'source_files': sources,
            'window_scheme': {
                'basis': 'integer_seconds_from_original_time',
                'seconds_before': 2,
                'seconds_after': 2,
                'start_seconds': start,
                'end_seconds': end,
                'is_expert_action_boundary': False,
            },
            'stroke_mapping': {'original': row['type'], 'prototype_label': condition_stroke},
            'review_status': 'unreviewed',
            'video_media_downloaded': False,
            'video_playback_verified': False,
        },
    }
    return reference_id, payload


def main():
    if not DB.is_file():
        raise FileNotFoundError(f'Existing research database required: {DB}')
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    sources, texts = {}, {}
    for filename, url in SOURCE_FILES.items():
        texts[filename], sources[filename] = read_source(filename, url)
    matches = list(csv.DictReader(io.StringIO(texts['match.csv'])))
    match = next(row for row in matches if integer(row['id']) == MATCH_ID)
    if match['video'] != MATCH_FOLDER or match['url'] != 'https://www.youtube.com/watch?v=boQC4J4E1ZQ':
        raise ValueError('Selected source match differs from the reviewed metadata')
    rows = list(csv.DictReader(io.StringIO(texts['match13_set1.csv'])))
    required = {'rally', 'ball_round', 'time', 'frame_num', 'player', 'type'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('Unexpected source CSV schema')
    flaw_rows = list(csv.DictReader(io.StringIO(texts['flaw_shot_records.csv'])))
    excluded = excluded_rows(flaw_rows)
    selected = []
    for condition_stroke, (original_type, display_type) in SHOT_TYPES.items():
        for row in select_four(rows, original_type, excluded):
            reference_id, payload = build_payload(row, match, condition_stroke, display_type, sources)
            selected.append({'id': reference_id, 'kind': 'reference', 'payload': payload})
    inserted, skipped = [], []
    with sqlite3.connect(DB, timeout=30) as db:
        for item in selected:
            existing = db.execute('SELECT kind FROM annotations WHERE id=?', (item['id'],)).fetchone()
            if existing:
                skipped.append(item['id'])
                continue
            db.execute('INSERT INTO annotations (id,kind,payload) VALUES (?,?,?)',
                       (item['id'], item['kind'], json.dumps(item['payload'], ensure_ascii=False)))
            inserted.append(item['id'])
    manifest = {
        'schema': 'badminton-professional-reference-import/1',
        'imported_at_utc': datetime.now(timezone.utc).isoformat(),
        'repository': f'https://github.com/{REPO}',
        'commit': COMMIT,
        'dataset_repository': 'https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet',
        'citations': [
            'https://doi.org/10.1145/3580305.3599906',
            'https://openaccess.thecvf.com/content/CVPR2026W/CVsports/html/Chang_BST_Badminton_Stroke-type_Transformer_for_Skeleton-based_Action_Recognition_in_Racket_CVPRW_2026_paper.html',
        ],
        'annotation_license': 'MIT; retain BST_LICENSE.txt, ShuttleSet_LICENSE.txt, and author citations',
        'upstream_license_commit': UPSTREAM_COMMIT,
        'media_rights': 'No match media acquired. The repository license is not asserted to license BWF match videos.',
        'sources': sources,
        'match': match,
        'source_row_count': len(rows),
        'selection_policy': 'Four examples each of 長球/切球/殺球; favour alternating players and distinct rallies; exclude author removed rows; all references unreviewed.',
        'selected_count': len(selected),
        'inserted_count': len(inserted),
        'skipped_existing_count': len(skipped),
        'inserted_ids': inserted,
        'skipped_existing_ids': skipped,
        'selected_references': selected,
        'no_tests_run': True,
    }
    output = SOURCE_DIR / 'manifest.json'
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(output)
    print(json.dumps({'selected': len(selected), 'inserted': len(inserted), 'skipped_existing': len(skipped),
                      'inserted_ids': inserted, 'manifest': str(output)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
