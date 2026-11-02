"""Persistent local visual action indexing and technique-specific retrieval.

The proposal score never decides a technique label. Only an independently
generated visual prediction can match a technique; all matches remain candidates.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import shutil
import subprocess
import threading
import time
import uuid

from action_windows import build_action_windows
from analysis_jobs import atomic_json, data_path, file_hash, registered_video
from vision_model import MODEL
from specialist_model import MODEL_ID, MODEL_REVISION, ENGINE_VERSION, CLASSIFICATION_SOURCE, available as specialist_available
from stroke_taxonomy import LABELS, VERSION as TAXONOMY_VERSION, normalise_event, query_matches
from stroke_review import human_overrides, read_labels

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / '03_数据'
STORE = DATA / '技术动作'
PYTHON = ROOT / '.venv-vision/bin/python'
CLASSIFIER = ROOT / 'specialist_pipeline.py'
MANAGER = None
MANAGER_LOCK = threading.RLock()


def now():
    return datetime.now(timezone.utc).isoformat()


def pose_file(pose):
    if not isinstance(pose.get('id'), str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', pose['id']):
        raise ValueError('动作检索结果标识无效')
    return data_path(DATA / '姿态分析' / pose['id'] / 'result.json')


def model_available():
    weights = list(MODEL.glob('*.safetensors')) if MODEL.is_dir() else []
    return specialist_available() and PYTHON.is_file() and (MODEL / 'config.json').is_file() and (MODEL / 'tokenizer_config.json').is_file() and sum(file.stat().st_size for file in weights) > 1024 ** 3


def validated_query(query):
    if not isinstance(query, dict):
        raise ValueError('技术检索格式错误')
    strokes, style = query.get('strokes', []), query.get('style')
    if not isinstance(strokes, list) or any(stroke not in LABELS for stroke in strokes) or style not in [None, 'forehand', 'backhand']:
        raise ValueError('暂不支持这项技术检索')
    if not strokes and style is None:
        raise ValueError('请说想查看哪种技术动作')
    label = ('反手' if style == 'backhand' else '正手' if style == 'forehand' else '') + ' / '.join(LABELS[stroke] for stroke in strokes)
    return {'strokes': list(dict.fromkeys(strokes)), 'style': style, 'label': label}


class ActionSearch:
    def __init__(self):
        STORE.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.states = {}
        self.stop_requested = False
        self.process = None
        self.fingerprints = {}
        self.controller = (STORE / '.controller.lock').open('a+b')
        try:
            fcntl.flock(self.controller.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.controller.close()
            raise RuntimeError('技术检索服务已在运行')
        for file in STORE.glob('*/index.json'):
            try:
                state = json.loads(file.read_text())
                if (state.get('pose_id') != file.parent.name or not isinstance(state.get('events'), list) or
                        not isinstance(state.get('candidates'), list) or state.get('engine_version') != ENGINE_VERSION):
                    continue
                if state['status'] in ['queued', 'running']:
                    state.update(status='failed', message='上次技术识别中断，可重新搜索', updated_at=now())
                    atomic_json(file, state)
                self.states[state['pose_id']] = state
            except (ValueError, OSError, TypeError, KeyError):
                continue
        self.worker = threading.Thread(target=self._loop, name='badminton-action-search', daemon=True)
        self.worker.start()

    def fingerprint(self, pose):
        file = pose_file(pose)
        stat = file.stat()
        key = (str(file), stat.st_size, stat.st_mtime_ns)
        with self.lock:
            cached = self.fingerprints.get(key)
        if cached is None:
            cached = file_hash(file)
            with self.lock:
                self.fingerprints[key] = cached
        return cached

    def _save(self, state):
        state['updated_at'] = now()
        atomic_json(STORE / state['pose_id'] / 'index.json', state)

    def ensure(self, job, pose, sides, retry=False, bst_only=False):
        if job.get('status') != 'completed' or job.get('result_id') != pose.get('id'):
            raise ValueError('请等视频处理完成再搜索技术动作')
        fingerprint = self.fingerprint(pose)
        identifier = pose['id']
        with self.condition:
            state = self.states.get(identifier)
            if state and state['pose_fingerprint'] == fingerprint and state['source_sha256'] == pose['source_sha256'] and state.get('model_revision') == MODEL_REVISION:
                expanded = sorted(set(state.get('requested_sides', [])) | set(sides))
                if expanded != state.get('requested_sides', []):
                    state['requested_sides'] = expanded
                    state['total'] = sum(candidate['player'] in expanded for candidate in state['candidates'])
                    if state['status'] == 'completed' and state['processed'] < state['total']:
                        state.update(status='queued', message='正在查看另一位球员的技术动作')
                    self._save(state)
                    self.condition.notify_all()
                if state['status'] not in ['failed', 'unavailable']:
                    return state
                if retry and (specialist_available() if bst_only else model_available()) and not self.stop_requested:
                    state['events'] = [event for event in state['events'] if not event.get('classification_error')]
                    state['processed'] = len(state['events'])
                    state.update(status='queued', message='正在准备识别技术动作', error=None)
                    state['bst_only'] = bst_only
                    self._save(state)
                    self.condition.notify_all()
                return state
            source = registered_video(pose['source_video_id'], allow_reference=True)
            # Specialist windows are generated after original-frame ball and
            # player inference; the coarse wrist proposals remain a baseline.
            candidates, total = [], 0
            available = specialist_available() if bst_only else model_available()
            previous_folder = STORE / identifier
            if (previous_folder / 'index.json').is_file():
                archive = STORE / '.历史' / ('before-' + ENGINE_VERSION + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')) / identifier
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(previous_folder, archive)
            state = {'schema_version': 1, 'engine_version': ENGINE_VERSION, 'pose_id': identifier,
                     'source_video_id': pose['source_video_id'], 'source_sha256': pose['source_sha256'],
                     'pose_fingerprint': fingerprint, 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
                     'pose_file': str(pose_file(pose)), 'source_file': str(source['path']),
                     'status': 'queued' if available else 'unavailable',
                     'message': '正在准备跟踪羽毛球和球员' if available else '技术识别模型尚未准备好',
                     'candidates': candidates, 'requested_sides': sorted(sides), 'events': [], 'processed': 0, 'total': total,
                     'bst_only': bst_only, 'error': None, 'created_at': now(), 'updated_at': now()}
            self._save(state)
            self.states[identifier] = state
            self.condition.notify_all()
            return state

    def _loop(self):
        while True:
            with self.condition:
                waiting = [state for state in self.states.values() if state['status'] == 'queued']
                while not waiting and not self.stop_requested:
                    self.condition.wait(timeout=2)
                    waiting = [state for state in self.states.values() if state['status'] == 'queued']
                if self.stop_requested:
                    return
                state = min(waiting, key=lambda item: item['created_at'])
                state.update(status='running', message='正在逐段识别技术动作')
            try:
                self._save(state)
                self._run(state)
            except Exception as error:
                with self.lock:
                    state.update(status='failed', error=type(error).__name__, message='这次技术识别没有完成，可以重新提问重试')
                    try:
                        self._save(state)
                    except OSError:
                        self.stop_requested = True
                print('技术识别异常：' + repr(error), flush=True)

    def _run(self, state):
        folder = data_path(STORE / state['pose_id'])
        done = {event['id'] for event in state['events']}
        remaining = [candidate for candidate in state['candidates'] if candidate['id'] not in done and candidate['player'] in state['requested_sides']]
        binaries = list((ROOT / '.venv-pose/lib').glob('python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*'))
        executable = next((file for file in binaries if file.is_file() and os.access(file, os.X_OK)), None)
        if executable is None:
            raise ValueError('视频解码环境无法读取')
        config = {key: state[key] for key in ['pose_id', 'source_sha256', 'pose_fingerprint', 'pose_file', 'source_file']}
        origin_file = DATA / '视频工作流' / 'sources' / (state['source_video_id'] + '.json')
        origin = json.loads(origin_file.read_text()) if origin_file.is_file() else {}
        config['court_line_color'] = origin.get('court_line_color', 'unknown')
        if origin.get('court_calibration'):
            config['court_calibration'] = origin['court_calibration']
        config.update(candidates=remaining, ffmpeg=str(executable), output_folder=str(folder / '片段'),
                      classifier_sha256=file_hash(CLASSIFIER), engine_version=ENGINE_VERSION,
                      specialist_folder=str(DATA / '专项分析' / state['pose_id']),
                      requested_sides=list(state['requested_sides']), done_ids=list(done))
        config_file = folder / 'config.json'
        atomic_json(config_file, config)
        # The descriptor is inherited by the child, preventing two visual
        # models from overlapping after a controller crash and restart.
        with (STORE / '.execution.lock').open('a+b') as execution:
            while True:
                with self.lock:
                    if self.stop_requested:
                        raise RuntimeError('服务正在停止')
                try:
                    fcntl.flock(execution.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    with self.condition:
                        self.condition.wait(timeout=.5)
            with self.lock:
                if self.stop_requested:
                    raise RuntimeError('服务正在停止')
                command = [str(PYTHON), '-u', str(CLASSIFIER), '--config', str(config_file)]
                if state.get('bst_only'):
                    command.append('--bst-only')
                process = subprocess.Popen(command,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace',
                           start_new_session=True, pass_fds=(execution.fileno(),),
                           env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                                    HF_HOME=str(ROOT / '.huggingface-local-vision')))
                self.process = process
            lines = queue.Queue(maxsize=256)

            def reader():
                try:
                    for line in iter(lambda: process.stdout.readline(2097152), ''):
                        lines.put(line)
                finally:
                    lines.put(None)

            threading.Thread(target=reader, daemon=True).start()
            finished, completion = False, None
            started = last_output = time.monotonic()
            expected = {candidate['id']: candidate for candidate in remaining}
            try:
                with (folder / 'worker.log').open('a', encoding='utf-8') as log:
                    while not (finished and process.poll() is not None):
                        if self.stop_requested or time.monotonic() - started > 3600 or time.monotonic() - last_output > 240:
                            self._terminate(process)
                            raise RuntimeError('技术识别中断或超时')
                        try:
                            line = lines.get(timeout=.2)
                        except queue.Empty:
                            continue
                        if line is None:
                            finished = True
                            continue
                        last_output = time.monotonic()
                        log.write(line)
                        log.flush()
                        if line.startswith('STATUS '):
                            progress = json.loads(line[7:])
                            with self.lock:
                                state.update(message=str(progress['message'])[:200], stage_processed=progress.get('stage_processed'), stage_total=progress.get('stage_total'))
                                self._save(state)
                        elif line.startswith('CANDIDATES '):
                            proposed = json.loads(line[11:])
                            candidates = proposed['candidates']
                            for candidate in candidates:
                                if (not re.fullmatch(r'[0-9a-f]{32}', candidate['id']) or candidate['player'] not in ['near', 'far'] or
                                        not all(isinstance(candidate[k], (int, float)) and math.isfinite(candidate[k]) for k in ['start', 'end', 'peak_time']) or
                                        not 0 <= candidate['start'] <= candidate['peak_time'] < candidate['end']):
                                    raise ValueError('专项击球候选格式无效')
                            remaining = proposed['remaining']
                            expected = {candidate['id']: candidate for candidate in remaining}
                            if any(candidate['id'] not in {c['id'] for c in candidates} for candidate in remaining):
                                raise ValueError('专项候选与待处理片段不一致')
                            with self.lock:
                                state.update(candidates=candidates, total=sum(c['player'] in state['requested_sides'] for c in candidates))
                                self._save(state)
                        elif line.startswith('EVENT '):
                            event = json.loads(line[6:])
                            proposal = expected.get(event.get('id'))
                            if not proposal or any(event.get(key) != proposal[key] for key in ['player', 'start', 'end', 'peak_time']):
                                raise ValueError('视觉识别结果与片段不一致')
                            if event.get('model_revision') != MODEL_REVISION or event.get('classification_source') != CLASSIFICATION_SOURCE:
                                raise ValueError('视觉识别来源不一致')
                            if event['id'] in done:
                                continue
                            with self.lock:
                                state['events'].append(event)
                                done.add(event['id'])
                                state.update(processed=len(state['events']), message='正在逐段识别技术动作')
                                self._save(state)
                        elif line.startswith('DONE '):
                            completion = json.loads(line[5:])
                returncode = process.wait(timeout=5)
                if returncode != 0 or not completion or completion.get('pose_id') != state['pose_id'] or completion.get('processed') != len(remaining):
                    raise RuntimeError('技术识别结果未完整保存')
                if file_hash(Path(state['pose_file'])) != state['pose_fingerprint'] or file_hash(Path(state['source_file'])) != state['source_sha256']:
                    raise ValueError('识别期间原视频或姿态结果发生变化')
                if state['events'] and all(event.get('classification_error') for event in state['events']):
                    raise RuntimeError('视觉模型未能读取任何候选片段')
                with self.lock:
                    expanded = len(state['events']) < state['total']
                    state.update(status='queued' if expanded else 'completed',
                                 message='正在准备查看另一位球员' if expanded else '技术动作检索已完成', error=None,
                                 stage_processed=None, stage_total=None)
                    self._save(state)
                    if expanded:
                        self.condition.notify_all()
            finally:
                if process.poll() is None:
                    self._terminate(process)
                process.stdout.close()
                with self.lock:
                    self.process = None

    @staticmethod
    def _terminate(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def shutdown(self):
        with self.condition:
            self.stop_requested = True
            self.condition.notify_all()
            process = self.process
        if process:
            self._terminate(process)
        self.worker.join(timeout=8)
        if not self.worker.is_alive():
            self.controller.close()


def initialize_action_search():
    global MANAGER
    with MANAGER_LOCK:
        if MANAGER is None:
            MANAGER = ActionSearch()
    return MANAGER


def match_events(state, query, sides):
    matches = []
    overrides = human_overrides(state['pose_id'], state['source_sha256'])
    from hit_refinement import read
    refinement=read(state['pose_id'],state['source_sha256'])
    candidates=list(state['events']);known={e['id'] for e in candidates}
    if refinement:candidates.extend(e for e in refinement['supplemental_candidates'] if e['id'] not in known)
    for original in sorted(candidates, key=lambda value: (value['peak_time'], value['player'])):
        event = normalise_event(original)
        human = overrides.get(event['id'])
        if human:
            if human.get('play_context') == 'between_rallies' or human.get('wrong_target') or human['stroke_type'] in ['unknown', 'other', 'no_stroke']:
                continue
            event.update({k:human[k] for k in ['player','start','end','peak_time','stroke_type','swing_style']})
            event.update(quality='usable', visible_stroke=True, stroke_certainty='likely',
                         style_certainty='likely' if human['swing_style']!='unknown' else 'uncertain',
                         classification_error=None, classification_status='human_confirmed',
                         classification_source='human_user_selection')
        if event['player'] not in sides or event.get('quality') != 'usable' or event.get('visible_stroke') is not True:
            continue
        if event.get('stroke_type') == 'no_stroke' or event.get('classification_error'):
            continue
        if query['strokes'] and (not query_matches(event.get('stroke_type'), query['strokes']) or event.get('stroke_certainty') != 'likely'):
            continue
        if query['style'] and (event.get('swing_style') != query['style'] or event.get('style_certainty') != 'likely'):
            continue
        match = {'id': event['id'], 'event_id': event['id'], 'player': event['player'],
                 'start': event['start'], 'end': event['end'], 'peak_time': event['peak_time'],
                 'label': f"疑似{query['label']}片段 {len(matches) + 1}",
                 'classification_source': event.get('classification_source'), 'classification_status': 'human_confirmed' if human else 'model_candidate',
                 'review_status': 'confirmed_by_user' if human else 'unreviewed', 'stroke_type': event.get('stroke_type'), 'swing_style': event.get('swing_style')}
        from hit_segments import dense_data, segment
        window = segment(dense_data(state['pose_id']), event, human)
        match.update(start=window['start'],end=window['end'],peak_time=window['center_time'],
                     start_frame=window['start_frame'],end_frame=window['end_frame'],
                     center_frame=window['center_frame'],contact_status=window['contact_status'],
                     source_context_start=(human or {}).get('source_context_start',original['start']),
                     source_context_end=(human or {}).get('source_context_end',original['end']))
        from stroke_context import window as context_window
        full = context_window(dense_data(state['pose_id']), original, candidates, overrides)
        match.update(start=full['start'], end=full['end'], peak_time=full['peak_time'],
                     start_frame=full['start_frame'], end_frame=full['end_frame'],
                     center_frame=full['frame_index'], context_basis=full['basis'],
                     context_boundary_confirmed=full['boundary_confirmed'],
                     contact_review_start=window['start'], contact_review_end=window['end'])
        if event.get('thumbnail_file'):
            match['thumbnail_url'] = f"/action-preview/{state['pose_id']}/{event['id']}.jpg"
        matches.append(match)
        if len(matches) == 6:
            break
    return matches


def response_for(state, context):
    from hit_segments import VERSION as WINDOW_VERSION
    from hit_refinement import fingerprint as refinement_fingerprint
    query = validated_query(context['query'])
    matches = match_events(state, query, context['sides'])
    status = state['status']
    failures = sum(bool(event.get('classification_error')) for event in state['events'] if event['player'] in context['sides'])
    if status in ['queued', 'running']:
        message = state['message'] + f"，正在找{query['label']}片段。" if not matches else f"先找到 {len(matches)} 处疑似{query['label']}，还在继续查看其他动作。"
    elif status == 'completed':
        message = f"先给你 {len(matches)} 处疑似{query['label']}。点击片段看看是不是你想看的动作。" if matches else f"暂未找到能辨认为{query['label']}的片段。部分动作可能漏识或看不清，这不代表视频里没有。"
    elif status == 'unavailable':
        message = '技术识别暂时未准备好，不能可靠判断动作类型。我不会要求你先找秒数。'
    else:
        message = '这次技术检索没有完成，可以重新提问重试。'
    return {'id': context['id'], 'query': query, 'label': query['label'], 'sides': context['sides'],
            'status': status, 'matches': matches, 'processed': state['processed'], 'total': state['total'],
            'classification_failures': failures,
            'stage_processed': state.get('stage_processed'), 'stage_total': state.get('stage_total'),
            'message': message, 'source_sha256': state['source_sha256'], 'pose_fingerprint': state['pose_fingerprint'],
            'engine_version': ENGINE_VERSION, 'taxonomy_version': TAXONOMY_VERSION,
            'hit_window_version': WINDOW_VERSION,'hit_refinement_fingerprint':refinement_fingerprint(state['pose_id']),
            'annotation_revision': read_labels(state['pose_id'], state['source_sha256'])['revision']}


def request_search(job, pose, query, sides):
    query = validated_query(query)
    sides = [side for side in ['near', 'far'] if side in sides]
    context = {'id': uuid.uuid4().hex, 'query': query, 'sides': sides}
    manager = initialize_action_search()
    state = manager.ensure(job, pose, sides, retry=True)
    with manager.lock:
        return response_for(state, context)


def refresh_search(job, pose, context):
    manager = initialize_action_search()
    state = manager.ensure(job, pose, context['sides'])
    with manager.lock:
        return response_for(state, context)
