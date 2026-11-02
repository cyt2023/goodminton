"""Local, persisted, single-worker pose queue for user-selected video windows."""
from collections import deque
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
import sqlite3
import subprocess
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / '03_数据'
DB = DATA / '研究数据.sqlite3'
JOBS = DATA / '分析任务'
POSE_OUTPUT = DATA / '姿态分析'
PYTHON = ROOT / '.venv-pose' / 'bin' / 'python'
ANALYZER = ROOT / 'analyze_pose.py'
ACTIVE = {'queued', 'running', 'cancelling'}
FINAL = {'completed', 'failed', 'cancelled', 'interrupted'}
MAX_PENDING = 3
MAX_RUNTIME = 900
FPS = 10
AUTO_FPS = 5
MAX_AUTOMATIC_DURATION = 1800
MAX_AUTOMATIC_RUNTIME = 1800


class QueueFull(ValueError):
    pass


class PersistenceFailure(OSError):
    pass


class QueueUnavailable(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def data_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(DATA.resolve()):
        raise ValueError('文件必须位于本项目的数据目录')
    return path


def atomic_json(file, value):
    file = data_path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(file.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(file)
    finally:
        temporary.unlink(missing_ok=True)


def numeric(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + '需要有限数字')
    return float(value)


def registered_video(video_id, *, allow_reference=False):
    if not isinstance(video_id, str) or not re.fullmatch(r'[0-9a-f]{32}', video_id):
        raise ValueError('请选择已上传的个人录像')
    with sqlite3.connect(DB) as db:
        row = db.execute('SELECT name,kind,path FROM videos WHERE id=?', (video_id,)).fetchone()
    allowed_kinds = {'personal', 'reference'} if allow_reference else {'personal'}
    if not row or row[1] not in allowed_kinds:
        raise ValueError('只能分析已登记的本地个人录像')
    file = data_path(DATA / row[2])
    if not file.is_file():
        raise ValueError('录像文件不存在，请重新上传')
    stat = file.stat()
    return {'video_id': video_id, 'name': row[0], 'kind': row[1], 'path': file,
            'source_stat': {'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}}


def file_hash(file):
    digest = hashlib.sha256()
    with file.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_selection(body):
    if not isinstance(body, dict):
        raise ValueError('分析请求格式错误')
    start, end = numeric(body.get('start'), '开始时间'), numeric(body.get('end'), '结束时间')
    if not 0 <= start < end or end - start > 30:
        raise ValueError('先选择不超过 30 秒的片段，结束需大于开始')
    start, end = round(start, 4), round(end, 4)
    if start >= end:
        raise ValueError('片段过短，请选择至少一个采样步长')
    players = body.get('players')
    if not isinstance(players, dict) or not players or not set(players).issubset({'near', 'far'}):
        raise ValueError('至少设置近端或远端一位球员的区域')
    cleaned = {}
    for side, player in players.items():
        if not isinstance(player, dict):
            raise ValueError('球员区域格式错误')
        gate, anchor = player.get('foot_gate'), player.get('anchor')
        if not isinstance(gate, list) or len(gate) != 4 or not isinstance(anchor, list) or len(anchor) != 2:
            raise ValueError('每位球员需要矩形区域和起始脚部位置')
        gate = [numeric(value, '区域坐标') for value in gate]
        anchor = [numeric(value, '起始坐标') for value in anchor]
        x1, y1, x2, y2 = gate
        if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
            raise ValueError('区域需位于画面内且宽高大于零')
        if not x1 <= anchor[0] <= x2 or not y1 <= anchor[1] <= y2:
            raise ValueError('球员的起始脚部位置需在对应区域内')
        cleaned[side] = {'foot_gate': gate, 'anchor': anchor, 'height_range': [0.02, 0.95]}
    label = body.get('label', '')
    if not isinstance(label, str):
        raise ValueError('分析名称需为文字')
    return start, end, cleaned, label.strip()[:200]


class AnalysisJobs:
    def __init__(self):
        JOBS.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.jobs = {}
        self.stop_requested = False
        self.worker = None
        self.process = None
        self.probe_process = None
        self.probe_lock = threading.Lock()
        self.probe_cache = {}
        self.unavailable_reason = None
        self.controller_lock = data_path(JOBS / '.controller.lock').open('a+b')
        try:
            fcntl.flock(self.controller_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.controller_lock.close()
            raise RuntimeError('已有本地分析服务在运行，请使用现有工作台')
        for file in sorted(JOBS.glob('J_*/job.json')):
            try:
                job = json.loads(data_path(file).read_text(encoding='utf-8'))
                self._validate_saved_job(job, file)
                if job['status'] in {'running', 'cancelling'}:
                    if self._recover_completed(job):
                        job.update(status='completed', result_id=job['id'], processed_frames=job['total_frames'],
                                   message='处理完成，已恢复发布记录', error=None, finished_at=utc_now(), updated_at=utc_now())
                    else:
                        job.update(status='interrupted', message='服务上次中断，可重新提交相同片段',
                                   error='上次服务在任务完成前停止', finished_at=utc_now(), updated_at=utc_now())
                    atomic_json(file, job)
                self.jobs[job['id']] = job
            except (ValueError, TypeError, KeyError):
                file.rename(file.with_name('job.invalid.' + uuid.uuid4().hex + '.json'))
                print('已隔离格式损坏的任务记录：' + file.parent.name, flush=True)
            except OSError:
                self.controller_lock.close()
                raise

    @staticmethod
    def _validate_saved_job(job, file):
        if (not isinstance(job, dict) or not isinstance(job.get('id'), str) or
                not re.fullmatch(r'J_[0-9a-f]{32}', job['id']) or file.parent.name != job['id'] or
                job.get('status') not in ACTIVE | FINAL or not isinstance(job.get('video_id'), str) or
                not re.fullmatch(r'[0-9a-f]{32}', job['video_id'])):
            raise ValueError('任务标识或状态损坏')
        if not isinstance(job.get('label'), str):
            raise ValueError('任务名称缺失')
        for key in ['created_at', 'updated_at']:
            if not isinstance(job.get(key), str):
                raise ValueError('任务时间缺失')
            datetime.fromisoformat(job[key])
        mode = job.get('mode', 'manual')
        if mode == 'auto':
            start, end = numeric(job.get('start'), '开始'), numeric(job.get('end'), '结束')
            if start != 0 or not 0 < end <= MAX_AUTOMATIC_DURATION or job.get('players') != {} or job.get('sample_fps') != AUTO_FPS:
                raise ValueError('自动任务范围损坏')
        elif mode == 'manual':
            start, end, players, _ = validate_selection(job)
            if start != job['start'] or end != job['end'] or players != job['players'] or job.get('sample_fps') != FPS:
                raise ValueError('任务选区损坏')
        else:
            raise ValueError('任务模式损坏')
        expected = math.ceil((end - start) * job['sample_fps'] - 1e-8)
        actual = job.get('total_frames')
        valid_total = type(actual) is int and (actual == expected or
                      (mode == 'auto' and job['status'] == 'completed' and max(1, expected - 2) <= actual <= expected))
        if (not isinstance(job.get('cancel_requested'), bool) or
                not valid_total or type(job.get('processed_frames')) is not int or not 0 <= job['processed_frames'] <= actual):
            raise ValueError('任务进度损坏')
        stat = job.get('source_stat')
        if not isinstance(stat, dict) or any(type(stat.get(key)) is not int or stat[key] < 0 for key in ['size_bytes', 'mtime_ns']):
            raise ValueError('录像指纹缺失')
        for key in ['source_sha256', 'config_sha256']:
            if not isinstance(job.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', job[key]):
                raise ValueError('任务指纹缺失')

    def _recover_completed(self, job):
        if (job['cancel_requested'] or job['status'] != 'running' or
                job.get('process_exit_code') != 0 or job.get('result_event_id') != job['id']):
            return False
        try:
            result = self._verified_result(job)
            index_file = data_path(POSE_OUTPUT / 'analysis_index.json')
            items = json.loads(index_file.read_text(encoding='utf-8'))
            overlay_id = uuid.uuid5(uuid.NAMESPACE_URL, 'badminton-pose-overlay:' + job['id']).hex
            item = next((item for item in items if item.get('id') == job['id']), None)
            if (not item or item.get('overlay_video_id') != overlay_id or item.get('source_video_id') != job['video_id'] or
                    item.get('start') != job['start'] or item.get('end') != job['end'] or
                    item.get('result_file') != str((POSE_OUTPUT / job['id'] / 'result.json').relative_to(DATA))):
                return False
            with sqlite3.connect(DB) as db:
                row = db.execute('SELECT kind,path FROM videos WHERE id=?', (overlay_id,)).fetchone()
            published = row == ('analysis', str((POSE_OUTPUT / job['id'] / 'overlay.mp4').relative_to(DATA)))
            if published and job.get('mode') == 'auto':
                job['total_frames'] = len(result['frames'])
                job.update(processed_end=result.get('processed_end', job['end']),
                           clip_complete=result.get('clip_complete', True),
                           completion_status=result.get('completion_status', 'complete'))
            return published
        except (OSError, ValueError, TypeError, AttributeError, sqlite3.Error):
            return False

    def start(self):
        self.worker = threading.Thread(target=self._worker_loop, name='badminton-local-pose', daemon=True)
        self.worker.start()

    def _save(self, job):
        job['updated_at'] = utc_now()
        try:
            atomic_json(JOBS / job['id'] / 'job.json', job)
        except OSError as error:
            raise PersistenceFailure('无法保存本地任务记录：' + str(error)) from error

    def _require_available(self):
        if self.unavailable_reason:
            raise QueueUnavailable(self.unavailable_reason)
        if self.stop_requested:
            raise QueueUnavailable('服务正在关闭，请稍后重新启动')

    def _make_unavailable(self, error, job=None):
        with self.condition:
            self.unavailable_reason = '分析队列无法保存或读取任务，请检查本地磁盘和终端日志后重启服务'
            if job and job['status'] not in FINAL:
                job.update(status='interrupted', error=self.unavailable_reason, message='队列已停止，需要重启', finished_at=utc_now())
            for waiting in self.jobs.values():
                if waiting['status'] == 'queued':
                    waiting['message'] = self.unavailable_reason
            self.condition.notify_all()
        print(self.unavailable_reason + '：' + str(error), flush=True)

    def _public(self, job):
        fields = ['id', 'video_id', 'label', 'start', 'end', 'players', 'status', 'message', 'error',
                  'processed_frames', 'total_frames', 'result_id', 'created_at', 'updated_at',
                  'started_at', 'finished_at', 'sample_fps', 'processed_end', 'clip_complete', 'completion_status']
        return dict({key: job.get(key) for key in fields}, mode=job.get('mode', 'manual'))

    def list_jobs(self):
        with self.lock:
            if self.unavailable_reason:
                raise QueueUnavailable(self.unavailable_reason)
            items = sorted(self.jobs.values(), key=lambda job: job['created_at'], reverse=True)
            # Always include active jobs, plus the most recent finished records.
            selected = [job for job in items if job['status'] in ACTIVE]
            selected += [job for job in items if job['status'] not in ACTIVE][:30]
            return [self._public(job) for job in sorted(selected, key=lambda job: job['created_at'], reverse=True)]

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise FileNotFoundError('分析任务不存在')
            return self._public(self.jobs[job_id])

    def video_info(self, video_id, internal=False):
        if not PYTHON.is_file():
            raise ValueError('本地姿态环境尚未安装')
        source = registered_video(video_id, allow_reference=True)
        fingerprint = (video_id, source['source_stat']['size_bytes'], source['source_stat']['mtime_ns'])
        # One probe at a time: loading the pose runtime for concurrent HTTP
        # requests should not spawn several memory-heavy processes on the Air.
        with self.probe_lock:
            with self.lock:
                self._require_available()
                cached = self.probe_cache.get(fingerprint)
            if cached is None:
                cached = self._probe(source, fingerprint)
        if internal:
            return dict(cached)
        return {key: cached.get(key) for key in ['video_id', 'name', 'width', 'height', 'duration', 'fps', 'duration_is_approximate']}

    def _probe(self, source, fingerprint):
        video_id = source['video_id']
        command = [str(PYTHON), str(ANALYZER), '--probe-video', video_id]
        with self.lock:
            self._require_available()
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, start_new_session=True)
            self.probe_process = process
        try:
            stdout, _ = process.communicate(timeout=35)
        except subprocess.TimeoutExpired:
            self._terminate(process)
            process.communicate()
            raise ValueError('读取录像超时，请先导出较短的 H.264 MP4')
        finally:
            with self.lock:
                self.probe_process = None
        media_lines = [line[6:] for line in stdout.splitlines() if line.startswith('MEDIA ')]
        if process.returncode != 0 or len(media_lines) != 1:
            raise ValueError('录像无法解码，请确认文件完整或先导出 H.264 MP4')
        metadata = json.loads(media_lines[0])
        duration = numeric(metadata.get('duration'), '录像时长')
        if duration <= 0 or metadata.get('video_id') != video_id:
            raise ValueError('未能读取录像时长或对应视频')
        if not all(isinstance(metadata.get(key), int) and 0 < metadata[key] <= 16384 for key in ['width', 'height']):
            raise ValueError('录像画面尺寸不支持')
        cached = dict(metadata, source_stat=source['source_stat'], source_sha256=file_hash(source['path']))
        with self.lock:
            if len(self.probe_cache) > 100:
                self.probe_cache.clear()
            self.probe_cache[fingerprint] = cached
        return cached

    def submit(self, body):
        start, end, players, label = validate_selection(body)
        video_id = body.get('video_id')
        self._check_capacity()
        metadata = self.video_info(video_id, internal=True)
        # Metadata duration is approximate; require an in-range window and
        # let the decoder fail rather than publish an incomplete result.
        if end > metadata['duration'] + 0.02:
            raise ValueError('片段结束超出读取到的录像时长')
        return self._create_job(metadata, start, end, players, label, 'manual')

    def _check_capacity(self):
        with self.lock:
            self._require_available()
            if sum(job['status'] == 'queued' for job in self.jobs.values()) >= MAX_PENDING:
                raise QueueFull('已有视频等待处理，请稍后再上传')

    def submit_auto(self, video_id):
        self._check_capacity()
        metadata = self.video_info(video_id, internal=True)
        end = round(metadata['duration'], 4)
        if end <= 0:
            raise ValueError('没有读到视频画面，请换一个完整的视频')
        if end > MAX_AUTOMATIC_DURATION:
            raise ValueError('目前支持 30 分钟以内的视频，请先导出较短的视频再上传')
        return self._create_job(metadata, 0.0, end, {}, metadata['name'], 'auto')

    def _create_job(self, metadata, start, end, players, label, mode):
        fps = AUTO_FPS if mode == 'auto' else FPS
        with self.condition:
            self._check_capacity()
            identifier = 'J_' + uuid.uuid4().hex
            created = utc_now()
            job = {'id': identifier, 'video_id': metadata['video_id'], 'label': label or f"{metadata['name']} · {start:g}–{end:g} 秒",
                   'start': start, 'end': end, 'players': players, 'sample_fps': fps, 'mode': mode,
                   'status': 'queued', 'message': '等待本地处理', 'error': None,
                   'processed_frames': 0, 'total_frames': math.ceil((end - start) * fps - 1e-8),
                   'result_id': None, 'created_at': created, 'updated_at': created,
                   'started_at': None, 'finished_at': None, 'source_stat': metadata['source_stat'],
                   'source_sha256': metadata['source_sha256'], 'cancel_requested': False}
            config = {'sample_fps': fps, 'display_threshold': 0.5, 'detector_threshold': 0.4,
                      'max_width': 960, 'publish_results': False,
                      'jobs': [{key: job[key] for key in ['id', 'video_id', 'label', 'start', 'end', 'players', 'mode']}]}
            config_file = JOBS / identifier / 'config.json'
            try:
                atomic_json(config_file, config)
                job['config_sha256'] = file_hash(config_file)
                self._save(job)
            except OSError as error:
                self._make_unavailable(error)
                raise QueueUnavailable(self.unavailable_reason) from error
            self.jobs[identifier] = job
            self.condition.notify_all()
            return self._public(job)

    def cancel(self, job_id):
        with self.condition:
            if job_id not in self.jobs:
                raise FileNotFoundError('分析任务不存在')
            job = self.jobs[job_id]
            try:
                if job['status'] == 'queued':
                    job.update(status='cancelled', cancel_requested=True, message='已取消等待任务', finished_at=utc_now())
                    self._save(job)
                elif job['status'] == 'running':
                    job.update(status='cancelling', cancel_requested=True, message='正在停止本地处理')
                    self._save(job)
            except PersistenceFailure as error:
                self._make_unavailable(error, job)
                raise QueueUnavailable(self.unavailable_reason) from error
            self.condition.notify_all()
            return self._public(job)

    def _worker_loop(self):
        while True:
            execution_lock, job = None, None
            try:
                with self.condition:
                    pending = sorted((job for job in self.jobs.values() if job['status'] == 'queued'), key=lambda job: job['created_at'])
                    while not pending and not self.stop_requested and not self.unavailable_reason:
                        self.condition.wait(timeout=2)
                        pending = sorted((job for job in self.jobs.values() if job['status'] == 'queued'), key=lambda job: job['created_at'])
                    if self.stop_requested or self.unavailable_reason:
                        return
                    try:
                        execution_lock = data_path(JOBS / '.execution.lock').open('a+b')
                    except OSError as error:
                        raise PersistenceFailure('无法取得本地执行锁') from error
                    try:
                        fcntl.flock(execution_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        execution_lock.close()
                        execution_lock = None
                        self.condition.wait(timeout=1)
                        continue
                    job = pending[0]
                    job.update(status='running', message='读取录像并加载本地模型', started_at=utc_now())
                    self._save(job)
                self._run(job, execution_lock)
            except PersistenceFailure as error:
                self._make_unavailable(error, job)
                return
            except Exception as error:
                if job is None:
                    self._make_unavailable(error)
                    return
                try:
                    with data_path(JOBS / job['id'] / 'worker.log').open('a', encoding='utf-8') as stream:
                        stream.write('\n管理器异常：' + repr(error) + '\n')
                except OSError:
                    pass
                try:
                    with self.lock:
                        status = 'cancelled' if job.get('cancel_requested') else 'interrupted' if self.stop_requested else 'failed'
                        detail = str(error)[:1000] if isinstance(error, ValueError) else '处理程序错误（' + type(error).__name__ + '），详情保存在任务日志'
                        job.update(status=status, message='本地处理失败' if status == 'failed' else '处理已停止',
                                   error=detail if status == 'failed' else None, finished_at=utc_now())
                        self._save(job)
                except PersistenceFailure as persistence_error:
                    self._make_unavailable(persistence_error, job)
                    return
            finally:
                with self.lock:
                    process = self.process
                    self.process = None
                if process and process.poll() is None:
                    self._terminate(process)
                if execution_lock:
                    execution_lock.close()

    def _run(self, job, execution_lock):
        source = registered_video(job['video_id'], allow_reference=True)
        if source['source_stat'] != job['source_stat'] or file_hash(source['path']) != job['source_sha256']:
            raise ValueError('排队期间录像发生变化，请重新选择并提交')
        config_file = data_path(JOBS / job['id'] / 'config.json')
        if file_hash(config_file) != job['config_sha256']:
            raise ValueError('分析配置发生变化，请重新提交')
        with self.lock:
            if self.stop_requested or job['cancel_requested']:
                job.update(status='interrupted' if self.stop_requested else 'cancelled', message='处理已停止', finished_at=utc_now())
                self._save(job)
                return
            process = subprocess.Popen([str(PYTHON), '-u', str(ANALYZER), '--config', str(config_file)],
                                       cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, errors='replace', bufsize=1, start_new_session=True,
                                       pass_fds=(execution_lock.fileno(),))
            self.process = process
        messages = queue.Queue(maxsize=256)
        tail = deque(maxlen=40)
        result_event = None
        started = time.monotonic()
        abort_reason = None

        def reader():
            try:
                for line in iter(lambda: process.stdout.readline(16384), ''):
                    messages.put(line)
            finally:
                messages.put(None)

        read_thread = threading.Thread(target=reader, daemon=True)
        read_thread.start()
        reader_done = False
        log = data_path(JOBS / job['id'] / 'worker.log')
        try:
            with log.open('w', encoding='utf-8') as stream:
                while not (reader_done and process.poll() is not None):
                    with self.lock:
                        cancelled = job['cancel_requested']
                        stopping = self.stop_requested
                    runtime_limit = MAX_AUTOMATIC_RUNTIME if job.get('mode') == 'auto' else MAX_RUNTIME
                    if abort_reason is None and (cancelled or stopping or time.monotonic() - started > runtime_limit):
                        abort_reason = 'cancelled' if cancelled else 'interrupted' if stopping else 'timeout'
                        self._terminate(process)
                    try:
                        line = messages.get(timeout=0.15)
                    except queue.Empty:
                        continue
                    if line is None:
                        reader_done = True
                        continue
                    stream.write(line)
                    stream.flush()
                    tail.append(line.strip()[:1200])
                    if line.startswith('PROGRESS '):
                        try:
                            progress = json.loads(line[9:])
                            if progress.get('stage') == 'scouting':
                                with self.lock:
                                    if job['status'] == 'running':
                                        job.update(message='正在寻找画面中的球员')
                                        self._save(job)
                                continue
                            done, total = progress['processed_frames'], progress['total_frames']
                            if (isinstance(done, int) and not isinstance(done, bool) and total == job['total_frames'] and
                                    job['processed_frames'] <= done <= total):
                                with self.lock:
                                    if job['status'] == 'running':
                                        job.update(processed_frames=done, message='正在分析视频' if job.get('mode') == 'auto' else '正在提取双方二维骨架')
                                        self._save(job)
                        except (ValueError, KeyError, TypeError):
                            pass
                    elif line.startswith('RESULT '):
                        try:
                            event = json.loads(line[7:])
                            if event.get('result_id') == job['id']:
                                result_event = event
                        except (ValueError, TypeError, AttributeError):
                            pass
            returncode = process.wait(timeout=5)
            with self.lock:
                # Accepted cancellation wins even if the child just finished.
                if job['cancel_requested']:
                    abort_reason = 'cancelled'
                elif self.stop_requested:
                    abort_reason = 'interrupted'
                job['log_tail'] = list(tail)
                if abort_reason:
                    status = 'failed' if abort_reason == 'timeout' else abort_reason
                    job.update(status=status, message='处理超时，请选择更短片段' if abort_reason == 'timeout' else '处理已停止',
                               error=f'本地处理超过 {runtime_limit // 60} 分钟' if abort_reason == 'timeout' else None, finished_at=utc_now())
                elif returncode != 0 or result_event is None:
                    job.update(status='failed', message='本地处理失败，未发布结果',
                               error=f'处理程序退出码 {returncode}，没有完整结果。详情保存在任务日志。', finished_at=utc_now())
                else:
                    result = self._verified_result(job)
                    # Checkpoint the successful child before publication, so
                    # restart can reconcile a committed result with job state.
                    job.update(process_exit_code=returncode, result_event_id=result_event['result_id'])
                    self._save(job)
                    self._publish(job, result)
                    assignments = sum(side.get('assigned_frames', 0) for side in result['summary'].values())
                    job.update(processed_end=result.get('processed_end', job['end']),
                               clip_complete=result.get('clip_complete', True),
                               completion_status=result.get('completion_status', 'complete'))
                    job.update(status='completed', message=('视频处理完成' if job.get('mode') == 'auto' else '处理完成，可查看骨架和校正阶段') if assignments else '视频已读取，未找到清晰的人物',
                               processed_frames=len(result['frames']), total_frames=len(result['frames']), result_id=job['id'], finished_at=utc_now())
                self._save(job)
        finally:
            if process.poll() is None:
                self._terminate(process)
            process.stdout.close()

    @staticmethod
    def _terminate(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        finally:
            # Also reap FFmpeg descendants if the Python parent already exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _verified_result(self, job):
        folder = data_path(POSE_OUTPUT / job['id'])
        file, overlay = data_path(folder / 'result.json'), data_path(folder / 'overlay.mp4')
        if not file.is_file() or not overlay.is_file() or overlay.stat().st_size <= 0:
            raise ValueError('缺少完整姿态或叠加视频，未发布结果')
        result = json.loads(file.read_text(encoding='utf-8'))
        automatic = job.get('mode') == 'auto'
        expected = math.ceil((job['end'] - job['start']) * job['sample_fps'] - 1e-8)
        frames = result.get('frames')
        complete_frames = isinstance(frames, list) and (len(frames) == expected or
                          (automatic and max(1, expected - 2) <= len(frames) <= expected))
        if (result.get('id') != job['id'] or result.get('source_video_id') != job['video_id'] or
                result.get('source_sha256') != job['source_sha256'] or result.get('start') != job['start'] or
                result.get('end') != job['end'] or result.get('sample_fps') != job['sample_fps'] or not complete_frames or
                (not automatic and result.get('player_config') != job['players']) or
                (automatic and (result.get('analysis_mode') != 'auto' or result.get('requested_frame_count') != expected or
                                result.get('decoded_frame_count') != len(frames)))):
            raise ValueError('结果与本任务的录像、片段或区域不一致，未发布')
        return result

    def _publish(self, job, result):
        index_file = data_path(POSE_OUTPUT / 'analysis_index.json')
        POSE_OUTPUT.mkdir(parents=True, exist_ok=True)
        overlay_id = uuid.uuid5(uuid.NAMESPACE_URL, 'badminton-pose-overlay:' + job['id']).hex
        item = {'id': job['id'], 'label': job['label'], 'source_video_id': job['video_id'],
                'start': job['start'], 'end': job['end'], 'overlay_video_id': overlay_id,
                'result_file': str((POSE_OUTPUT / job['id'] / 'result.json').relative_to(DATA)), 'sample_fps': job['sample_fps'],
                'summary': result['summary'], 'runtime_seconds': result['runtime_seconds'], 'quality_note': result['quality_note']}
        for key in ['analysis_mode', 'processed_end', 'requested_frame_count', 'decoded_frame_count', 'clip_complete', 'completion_status']:
            if key in result:
                item[key] = result[key]
        with data_path(POSE_OUTPUT / '.index.lock').open('a+b') as index_lock:
            fcntl.flock(index_lock.fileno(), fcntl.LOCK_EX)
            items = json.loads(index_file.read_text(encoding='utf-8')) if index_file.is_file() else []
            if not isinstance(items, list):
                raise ValueError('姿态索引格式错误，请先修复，避免覆盖既有结果')
            with sqlite3.connect(DB, timeout=30) as db:
                db.execute('INSERT OR REPLACE INTO videos (id,name,kind,path) VALUES (?,?,?,?)',
                           (overlay_id, job['label'] + '（骨架叠加）', 'analysis',
                            str((POSE_OUTPUT / job['id'] / 'overlay.mp4').relative_to(DATA))))
            # Make the media route visible before exposing its index entry.
            atomic_json(index_file, [old for old in items if old.get('id') != job['id']] + [item])

    def shutdown(self):
        with self.condition:
            self.stop_requested = True
            self.condition.notify_all()
            probe = self.probe_process
        if probe:
            self._terminate(probe)
        if self.worker:
            self.worker.join(timeout=10)
        if self.process and self.process.poll() is None:
            self._terminate(self.process)
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=5)
        if self.worker and self.worker.is_alive():
            print('分析线程尚在收尾，执行锁保留至进程退出', flush=True)
            return
        self.controller_lock.close()
