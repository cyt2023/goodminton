"""Two video domains, persistent import/recognition and revisioned human data."""
import json
import hashlib
from copy import deepcopy
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import threading
import uuid
from urllib.parse import urlparse
from datetime import datetime, timezone
from analysis_jobs import DATA, DB, atomic_json, registered_video, QueueFull, file_hash
from stroke_review import load_source, read_labels, folder_for, LOCK as LABEL_LOCK
from court_settings import line_color
import court_review

ROOT = Path(__file__).resolve().parent
STORE = DATA / '视频工作流'
LIBRARY = DATA / '训练库'
LIBRARY_LOCK = threading.RLock()
DOMAINS = {'personal': '球友录像', 'reference': '国际比赛'}

def now():
    return datetime.now(timezone.utc).isoformat()

def read(file, default=None):
    return json.loads(file.read_text()) if file.is_file() else default

def sources():
    with sqlite3.connect(DB) as db:
        return {r[0]: dict(id=r[0], name=r[1], kind=r[2], path=r[3]) for r in db.execute('SELECT * FROM videos')}


def learning_content_hash(label, source_sha256, taxonomy_version, context=None):
    """Identify learning content, excluding notes, predictions and export flags.

    An unchanged event ID/revision alone does not establish participation. Keep
    the exact human contact time and review bounds so later corrections cannot
    inherit a trained flag from the old label.
    """
    context = context or {}
    value = {'source_sha256': source_sha256,
             'taxonomy_version': label.get('taxonomy_version', taxonomy_version),
             'stroke_type': label.get('stroke_type'), 'player': label.get('player'),
             'swing_style': label.get('swing_style', 'unknown'),
             'wrong_target': bool(label.get('wrong_target')),
             'contact_confirmed': bool(label.get('contact_confirmed')),
             'play_context': context.get('play_context', label.get('play_context'))}
    for key in ['start', 'end', 'peak_time']:
        value[key] = float(label[key]) if label.get(key) is not None else None
    for key in ['center_frame', 'start_frame', 'end_frame']:
        value[key] = label.get(key)
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def training_participation_index(training_root=None):
    """Read completed full-library runs without importing a training module."""
    training_root = Path(training_root) if training_root is not None else DATA / '击球标注/直接BST微调/运行记录'
    records, warnings = {}, []
    for path in sorted(training_root.glob('*/manifest.json')):
        try:
            manifest = read(path, {})
            if not manifest.get('train_all'):
                continue
            run = path.parent
            feature_manifest = read(run / 'personal_features_manifest.json', {})
            annotations = read(run / 'human_annotations_snapshot.json', {})
            checkpoint = (DATA / manifest['candidate_checkpoint']).resolve()
            if (not checkpoint.is_relative_to(DATA.resolve()) or not checkpoint.is_file()
                    or file_hash(checkpoint) != manifest['candidate_model_sha256']):
                raise ValueError('全库训练候选文件或SHA不匹配')
            accepted = {(r['job_id'], r['event_id']): r for r in feature_manifest['rows']}
            for key in manifest['trained_sample_keys']:
                job, eid = key['job_id'], key['event_id']
                row = accepted[(job, eid)]
                doc = annotations[job]
                label = doc['labels'][eid]
                if (row['source_sha256'] != doc['source_sha256'] or
                        row['stroke_type'] != label['stroke_type'] or row['player'] != label['player'] or
                        row['human_peak_time'] != label['peak_time'] or
                        key['contact_group_id'] != row['contact_group_id']):
                    raise ValueError('训练特征与人工标注快照内容不匹配')
                record = {'run_id': manifest['run_id'], 'model_sha256': manifest['candidate_model_sha256'],
                          'created_at': manifest['created_at'], 'source_sha256': row['source_sha256'],
                          'trained_dataset_revision': row['dataset_revision'],
                          'trained_learning_content_sha256': learning_content_hash(label, doc['source_sha256'],
                              doc['taxonomy_version'], doc.get('context_annotations', {}).get(eid)),
                          'contact_group_id': row['contact_group_id'],
                          'target_class_label': row['target_class_label'],
                          'training_scope': 'full_library_fit',
                          'model_window_version': row['model_window'].get('version'),
                          'model_window_basis': row['model_window'].get('basis')}
                records.setdefault((job, eid), []).append(record)
        except (OSError, ValueError, TypeError, KeyError) as error:
            warnings.append({'manifest': str(path), 'reason': str(error)[:300]})
    return {'records': records, 'warnings': warnings}


def with_training_participation(label, job, source_sha256, taxonomy_version, context, index):
    result = deepcopy(label)
    content_hash = learning_content_hash(result, source_sha256, taxonomy_version, context)
    participation = [{**entry, 'current_label_matches': (
        entry['source_sha256'] == source_sha256 and entry['trained_learning_content_sha256'] == content_hash)}
        for entry in index['records'].get((job, result['event_id']), [])]
    result['learning_content_sha256'] = content_hash
    result['training_participation'] = participation
    result['model_trained'] = any(entry['current_label_matches'] for entry in participation)
    return result


def build_training_snapshot(report, events, doc, source, provenance, recognition_methods=None,
                            *, finish=False, dense=None, participation=None):
    """Construct a revision snapshot without writing labels, files or the pool."""
    # Lazy imports keep the stdlib server and module imports free of cycles.
    from hit_segments import dense_data
    from stroke_context import metadata, VERSION as WINDOW_VERSION
    dense = dense if dense is not None else dense_data(doc['job_id'])
    if dense['source_sha256'] != doc['source_sha256']:
        raise ValueError('完整动作窗口与人工标注原视频不一致')
    windows = metadata(dense, events, doc)['windows']
    participation = participation if participation is not None else training_participation_index()
    labels = []
    for label in doc['labels'].values():
        a = with_training_participation(label, doc['job_id'], doc['source_sha256'], doc['taxonomy_version'],
                                        doc.get('context_annotations', {}).get(label['event_id']), participation)
        model_window = windows.get(label['event_id'])
        if model_window is None:
            raise ValueError('已标注动作没有对应的完整模型窗口')
        a['stroke_context'] = {**deepcopy(model_window), 'boundaries_confirmed': False}
        labels.append(a)
    trained_count = sum(a['model_trained'] for a in labels)
    snapshot = {'schema_version': 2, 'job_id': doc['job_id'], 'revision': doc['revision'],
                'domain': source['kind'], 'video_name': source['name'],
                'source_video_id': source['id'], 'source_sha256': doc['source_sha256'],
                'origin': deepcopy(provenance), 'taxonomy_version': doc['taxonomy_version'],
                'saved_at': now(), 'finished': finish,
                'model_trained': trained_count > 0,
                'model_trained_scope': 'At least one current label exactly participated in a saved full-library fit; see each label.',
                'trained_label_count': trained_count, 'untrained_label_count': len(labels) - trained_count,
                'all_current_labels_trained': bool(labels) and trained_count == len(labels),
                'training_history_warnings': participation['warnings'],
                'stroke_context_version': WINDOW_VERSION, 'labels': labels,
                'context_annotations': deepcopy(doc.get('context_annotations', {})),
                'recognition_methods': deepcopy(recognition_methods or {}),
                'predictions_before_correction': deepcopy(events),
                'candidate_count': len(events), 'annotated_count': len(doc['labels']),
                'remaining_candidates': sum(e['id'] not in doc['labels'] for e in events)}
    return snapshot


def save_training_data(job, finish=False, expected_revision=None):
    # The annotation and library snapshots are taken under the same label lock.
    # Automatic predictions are stored separately and never become human truth.
    with LABEL_LOCK, LIBRARY_LOCK:
        report, events, results_file = load_source(job)
        doc = read_labels(job, report['source_sha256'])
        if expected_revision is not None and expected_revision != doc['revision']:
            raise ValueError('标注版本已经更新，请刷新后保存')
        source = sources()[report['source_video_id']]
        provenance = read(STORE / 'sources' / (source['id'] + '.json'), {})
        snapshot = build_training_snapshot(report, events, doc, source, provenance,
                    read(results_file, {}).get('methods', {}), finish=finish)
        atomic_json(LIBRARY / '录像' / job / 'latest.json', snapshot)
        if finish:
            # Keep earlier revisions as immutable review checkpoints.
            checkpoint = LIBRARY / '录像' / job / f"revision_{doc['revision']}.json"
            if not checkpoint.exists():
                atomic_json(checkpoint, snapshot)
        refresh_pool()
        return {'revision': doc['revision'], 'annotated_count': snapshot['annotated_count'],
                'remaining_candidates': snapshot['remaining_candidates'], 'summary': library_summary(),
                'message': '标注及原始预测已保存到训练库；尚未自动训练或替换模型。'}

def refresh_pool():
    pool, seen = [], {}
    participation = training_participation_index()
    for file in sorted((LIBRARY / '录像').glob('*/latest.json')):
        snap = read(file)
        origin = snap.get('origin', {})
        for label in snap['labels']:
            context = snap.get('context_annotations', {}).get(label['event_id'], {})
            label = with_training_participation(label, snap['job_id'], snap['source_sha256'],
                                               snap['taxonomy_version'], context, participation)
            label['play_context'] = context.get('play_context', label.get('play_context'))
            code = label['stroke_type']
            outside = label['play_context'] == 'between_rallies'
            role = ('outside_rally' if outside else 'wrong_target' if label.get('wrong_target') else
                    'hit_negative' if code == 'no_stroke' else 'unresolved' if code in ['unknown', 'other'] or not label.get('contact_confirmed') else 'stroke_positive')
            sample = dict(label, job_id=snap['job_id'], dataset_revision=snap['revision'], domain=snap['domain'],
                          source_video_id=snap['source_video_id'], source_sha256=snap['source_sha256'],
                          origin=origin, training_role=role)
            # Neighboring candidates corrected to the same physical hit count once.
            # A/B tracks are local to each source. Do not assume A is the same
            # athlete in separately detected chunks of the same match.
            identity = (snap['source_sha256'], label['player'], label.get('center_frame',round(label['peak_time']*origin.get('fps',30))))
            if role == 'stroke_positive' and 'core_start_seconds' in origin:
                absolute = label['peak_time'] + origin.get('offset_seconds',0)
                if not origin['core_start_seconds'] <= absolute < origin['core_end_seconds']:
                    sample['training_role'] = role = 'overlap_context_only'
            if role == 'stroke_positive' and identity in seen:
                previous = pool[seen[identity]]
                previous.setdefault('duplicate_sources', []).append({'job_id':snap['job_id'], 'event_id':label['event_id']})
                if previous['stroke_type'] != code:
                    previous['training_role'] = 'conflicting_labels'
                continue
            if role == 'stroke_positive':
                seen[identity] = len(pool)
            pool.append(sample)
    atomic_json(LIBRARY / '训练样本.json', {'schema_version':2, 'updated_at':now(), 'samples':pool,
                'training_history_warnings': participation['warnings'],
                'note':'Positive strokes, hit negatives and excluded samples have separate roles. model_trained describes exact current-label participation in versioned full-library fits, not model deployment or accuracy; split by original video before evaluation.'})

def library_summary():
    payload = read(LIBRARY / '训练样本.json', {'samples':[]})
    samples = payload['samples']
    return {'total':len(samples), 'stroke_positive':sum(s['training_role']=='stroke_positive' for s in samples),
            'hit_negative':sum(s['training_role']=='hit_negative' for s in samples),
            'excluded':sum(s['training_role'] not in ['stroke_positive','hit_negative'] for s in samples),
            'domains':{d:sum(s['domain']==d for s in samples) for d in DOMAINS}, 'updated_at':payload.get('updated_at')}

class VideoWorkflows:
    def __init__(self, analysis, actions):
        self.analysis, self.actions = analysis, actions
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.process = None
        for directory in ['imports','jobs','sources','courts']:
            (STORE / directory).mkdir(parents=True, exist_ok=True)
        for file in (STORE / 'imports').glob('*.json'):
            item = read(file)
            if item['status'] == 'running':
                item.update(status='failed', message='上次导入中断，可重新导入；已有录像和标注保留。')
                atomic_json(file, item)
        self.import_worker = threading.Thread(target=self._imports, daemon=True, name='video-import')
        self.worker = threading.Thread(target=self._recognition, daemon=True, name='video-recognition-flow')
        self.import_worker.start()
        self.worker.start()

    def submit(self, body):
        kind = body.get('kind', 'reference')
        if kind not in DOMAINS:
            raise ValueError('请选择球友录像或国际比赛')
        video_id, url = body.get('video_id'), body.get('url', '').strip()
        if video_id:
            registered_video(video_id, allow_reference=True)
            source = sources()[video_id]
            kind = source['kind']
            title = source['name']
        else:
            parsed = urlparse(url)
            host = (parsed.hostname or '').lower()
            allowed = ['youtube.com','www.youtube.com','m.youtube.com','youtu.be','www.bilibili.com','bilibili.com','b23.tv']
            if parsed.scheme != 'https' or host not in allowed or parsed.username or parsed.password or parsed.port not in [None,443]:
                raise ValueError('请粘贴公开的 YouTube 或 Bilibili HTTPS 视频链接，也可以导入已下载的录像')
            title = body.get('name', '').strip()[:150] or '国际比赛 · 网上录像'
        saved_origin = read(STORE/'sources'/(video_id+'.json'), {}) if video_id else {}
        color = line_color(body.get('court_line_color', saved_origin.get('court_line_color')))
        start = body.get('start', 0)
        end = body.get('end')
        if not isinstance(start, (int,float)) or not math.isfinite(start) or start < 0:
            raise ValueError('开始时间应为非负秒数')
        if end is not None and (not isinstance(end,(int,float)) or not math.isfinite(end) or end <= start):
            raise ValueError('结束时间需大于开始时间，留空则处理到视频结尾')
        with self.lock:
            if sum(read(f)['status'] in ['queued','running'] for f in (STORE/'imports').glob('*.json')) >= 6:
                raise ValueError('已有多段录像等待导入，请稍后再试')
            item = dict(id=uuid.uuid4().hex, kind=kind, video_id=video_id, url=url, name=title,
                        court_line_color=color, start=start, end=end, status='queued', message='等待导入录像', jobs=[], created_at=now())
            atomic_json(STORE/'imports'/(item['id']+'.json'), item)
            return item

    def _save(self, item):
        with self.lock:
            atomic_json(STORE/'imports'/(item['id']+'.json'), item)

    def _command(self, command, folder, logname):
        with (folder/logname).open('a') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.process = process
            while process.poll() is None:
                if self.stop.wait(1):
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                    raise RuntimeError('导入已中断')
            self.process = None
            if process.returncode:
                raise ValueError('下载或转码未完成，可导入已下载的 MP4；诊断日志已保存在本机。')

    def _imports(self):
        while not self.stop.is_set():
            queued = [read(f) for f in (STORE/'imports').glob('*.json') if read(f)['status']=='queued']
            if not queued:
                self.stop.wait(2)
                continue
            item = min(queued, key=lambda i:i['created_at'])
            try:
                item.update(status='running', message='正在读取录像' if item['video_id'] else '正在下载公开比赛录像')
                self._save(item)
                prepared = self._prepare(item)
                if prepared is not False:
                    item.update(status='completed',message='导入完成，已自动排队识别每次击球')
            except Exception as error:
                item.update(status='failed', message=str(error)[:300])
            self._save(item)

    def _register(self, path, name, kind, provenance):
        video_id = uuid.uuid4().hex
        with sqlite3.connect(DB) as db:
            db.execute('INSERT INTO videos VALUES (?,?,?,?)',(video_id,name,kind,str(path.relative_to(DATA))))
        atomic_json(STORE/'sources'/(video_id+'.json'), provenance)
        return video_id

    def _prepare(self, item):
        folder = DATA/('职业示范' if item['kind']=='reference' else '个人录像')/'导入分段'/item['id']
        folder.mkdir(parents=True, exist_ok=True)
        if item['video_id']:
            source = registered_video(item['video_id'], allow_reference=True)
        else:
            ffmpeg = next((ROOT/'.venv-pose/lib').glob('python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*'))
            self._command([str(ROOT/'.venv-vision/bin/python'),'-m','yt_dlp','--no-playlist','--socket-timeout','30','--retries','2',
                           '--max-filesize','1G','-f','bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]',
                           '--merge-output-format','mp4','--ffmpeg-location',str(ffmpeg),'--write-info-json',
                           '-o',str(folder/'original.%(ext)s'),item['url']],folder,'download.log')
            info = read(folder/'original.info.json', {})
            file = folder/'original.mp4'
            if not file.exists() or file.stat().st_size > 1024**3:
                raise ValueError('下载录像为空或超过 1GB，请导入较短的比赛录像')
            item['name'] = info.get('title',item['name'])[:180]
            item['video_id'] = self._register(file,item['name'],item['kind'],dict(url=item['url'], offset_seconds=0))
            self._save(item)
            source = registered_video(item['video_id'],allow_reference=True)
        metadata = self.analysis.video_info(item['video_id'], internal=True)
        duration = metadata['duration']
        end = min(item['end'] if item['end'] is not None else duration, duration)
        if item['start'] >= end or (item['end'] is not None and item['end'] > duration+.05):
            raise ValueError('选择的范围超出录像时长')
        item['source_sha256'] = metadata['source_sha256']
        item['duration'] = duration
        item['effective_end'] = end
        if item.get('court_review_status') not in ['confirmed','skipped']:
            preview_time = min(item['start'] + 1, max(item['start'], end - 1 / metadata['fps']))
            draft = court_review.prepare(item['id'], source['path'], metadata['source_sha256'],
                                         item.get('court_line_color','unknown'), preview_time)
            previous = read(STORE/'sources'/(item['video_id']+'.json'),{}).get('court_calibration',{})
            if (previous.get('source_sha256') == metadata['source_sha256'] and previous.get('camera_fixed') is True
                    and previous.get('court_line_color') == item.get('court_line_color')
                    and previous.get('valid_start_seconds',float('inf')) <= item['start']
                    and previous.get('valid_end_seconds',-1) >= end):
                draft['previous_record'] = previous
                atomic_json(STORE/'courts'/item['id']/'draft.json',draft)
            item.update(status='awaiting_court', court_review_status='pending',
                        message='请确认或修正场地；确认后开始识别，也可跳过标定')
            return False
        base_origin = read(STORE/'sources'/(item['video_id']+'.json'),{})
        origin = dict(base_origin, original_video_id=item['video_id'], original_sha256=metadata['source_sha256'], fps=metadata['fps'],
                      court_line_color=line_color(item.get('court_line_color', base_origin.get('court_line_color'))))
        origin.pop('court_calibration', None)
        origin['court_review_status'] = item['court_review_status']
        if item['court_review_status'] == 'confirmed':
            record = read(STORE/'courts'/item['id']/'confirmed.json')
            if record['source_sha256'] != metadata['source_sha256']:
                raise ValueError('录像已改变，请重新标定场地')
            origin['court_calibration'] = record
        atomic_json(STORE/'sources'/(item['video_id']+'.json'), origin)
        ffmpeg = next((ROOT/'.venv-pose/lib').glob('python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*'))
        from research_adapters import settings as research_settings
        research_config = research_settings()
        chunk_seconds = research_config.get('recognition_chunk_seconds',120) if research_config['perception_backend']=='racketvision' else 600
        ranges = [(item['start'],end)] if end-item['start'] <= chunk_seconds else [(item['start']+n*chunk_seconds,min(item['start']+(n+1)*chunk_seconds,end)) for n in range(math.ceil((end-item['start'])/chunk_seconds))]
        for number,(start,stop) in enumerate(ranges,1):
            if self.stop.is_set():
                raise RuntimeError('导入中断')
            if start == 0 and stop == duration:
                vid = item['video_id']
                atomic_json(STORE/'sources'/(vid+'.json'),dict(origin,offset_seconds=0))
            else:
                # Physical clips keep all dense model timestamps and original-frame media aligned.
                # Include overlap for context at chunk edges; origin frame dedup prevents double training.
                clip_start, clip_end = max(item['start'],start-3), min(end,stop+3)
                target=folder/f'part_{number}.mp4'
                item['message']=f'正在准备第 {number}/{len(ranges)} 段录像'
                self._save(item)
                self._command([str(ffmpeg),'-y','-ss',str(clip_start),'-i',str(source['path']),'-t',str(clip_end-clip_start),
                               '-an','-c:v','libx264','-preset','veryfast','-crf','20',str(target)],folder,'prepare.log')
                vid=self._register(target,f"{item['name']} · {start/60:.1f}–{stop/60:.1f} 分钟",item['kind'],dict(origin,offset_seconds=clip_start,core_start_seconds=start,core_end_seconds=stop))
                if 'court_calibration' in origin:
                    clip_origin = read(STORE/'sources'/(vid+'.json'))
                    clip_info = self.analysis.video_info(vid, internal=True)
                    clip_origin['court_calibration'] = dict(origin['court_calibration'],
                        source_sha256=clip_info['source_sha256'],
                        original_source_sha256=metadata['source_sha256'],
                        original_reference_time_seconds=origin['court_calibration']['reference_time_seconds'],
                        original_valid_range_seconds=[item['start'],end],
                        reference_time_seconds=None, reference_frame=None,
                        valid_start_seconds=0., valid_end_seconds=clip_info['duration'],
                        applies_to_fixed_camera_clip=True)
                    atomic_json(STORE/'sources'/(vid+'.json'), clip_origin)
            # Retain each prepared source even if the pose queue is currently full.
            while not self.stop.is_set():
                try:
                    job=self.analysis.submit_auto(vid)
                    break
                except QueueFull:
                    self.stop.wait(3)
            else:
                raise RuntimeError('导入中断')
            atomic_json(STORE/'jobs'/(job['id']+'.json'),dict(job_id=job['id'],kind=item['kind'],import_id=item['id']))
            item['jobs'].append(job['id'])
            self._save(item)

    def court_item(self, identifier):
        if not re.fullmatch(r'[0-9a-f]{32}', identifier):
            raise ValueError('场地预览标识无效')
        item = read(STORE/'imports'/(identifier+'.json'))
        if item is None:
            raise ValueError('找不到这段导入录像')
        return item

    def court_snapshot(self, identifier):
        with self.lock:
            item = self.court_item(identifier)
            return {'item':item, 'draft':read(STORE/'courts'/identifier/'draft.json'),
                    'confirmed':read(STORE/'courts'/identifier/'confirmed.json')}

    def update_court(self, identifier, operation, body):
        with self.lock:
            item = self.court_item(identifier)
            if item['status'] != 'awaiting_court':
                raise ValueError('请等待录像准备完成，或返回首页查看处理进度')
            source = registered_video(item['video_id'], allow_reference=True)
            # Cached metadata is keyed by current file stat, including its hash.
            metadata = self.analysis.video_info(item['video_id'], internal=True)
            if metadata['source_sha256'] != item['source_sha256']:
                raise ValueError('录像已改变，请重新导入')
            if operation == 'proposal':
                color = line_color(body.get('court_line_color',item['court_line_color']))
                t = body.get('time_seconds',item['start'])
                if not isinstance(t,(int,float)) or not math.isfinite(t) or not item['start'] <= t < item['effective_end']:
                    raise ValueError('请选择导入范围内的预览时间')
                court_review.prepare(identifier,source['path'],item['source_sha256'],color,t)
                item['court_line_color'] = color
            elif operation == 'confirm':
                court_review.confirm(identifier,item['source_sha256'],body,item['start'],item['effective_end'])
                item.update(court_review_status='confirmed',status='queued',message='场地已确认，等待开始识别')
            elif operation == 'skip':
                item['court_line_color'] = line_color(body.get('court_line_color',item['court_line_color']))
                item.update(court_review_status='skipped',status='queued',message='已跳过场地标定，等待开始识别')
            else:
                raise ValueError('场地操作无效')
            self._save(item)
            return {'item':item, 'draft':read(STORE/'courts'/identifier/'draft.json')}

    def _recognition(self):
        while not self.stop.is_set():
            for file in (STORE/'jobs').glob('*.json'):
                if self.stop.is_set():
                    break
                flow=read(file)
                try:
                    job=self.analysis.get(flow['job_id'])
                    if job['status']!='completed' or (DATA/'技术动作'/job['id']/'片段/classified_events.json').exists():
                        continue
                    # Completed coarse pose alone is not a completed stroke pipeline.
                    pose=read(DATA/'姿态分析'/job['result_id']/'result.json')
                    self.actions.ensure(job,pose,['near','far'],bst_only=True)
                except Exception as error:
                    flow['error']=str(error)[:300]
                    atomic_json(file,flow)
            self.stop.wait(3)

    def retry(self, job_id):
        job=self.analysis.get(job_id)
        if job['status'] in ['queued','running','cancelling']:
            raise ValueError('录像仍在处理，请等当前处理完成')
        if job['status']!='completed' or job.get('mode')!='auto':
            return self.submit({'video_id':job['video_id']})
        pose=read(DATA/'姿态分析'/job['result_id']/'result.json')
        self.actions.ensure(job,pose,['near','far'],retry=True,bst_only=True)
        atomic_json(STORE/'jobs'/(job_id+'.json'),dict(job_id=job_id,kind=sources()[job['video_id']]['kind']))
        return {'message':'已排队继续逐拍识别'}

    def snapshot(self):
        inventory=sources()
        jobs=[]
        for file in (DATA/'分析任务').glob('*/job.json'):
            j=read(file)
            source=inventory.get(j['video_id'])
            if not source:
                continue
            classified=read(DATA/'技术动作'/j['id']/'片段/classified_events.json', {})
            annotation=read(DATA/'击球标注'/j['id']/'annotations.json',{'labels':{},'revision':0})
            events=classified.get('events',[])
            with self.actions.lock:
                action=dict(self.actions.states.get(j['id'],{}))
            flow=read(STORE/'jobs'/(j['id']+'.json'),{})
            ready=bool(classified) and (DATA/'专项分析'/j['id']/'report.json').exists() and action.get('status') not in ['queued','running','failed']
            status='ready' if ready else action.get('status',j['status'])
            message='可逐拍核对' if ready else action.get('message',j['message'])
            if j['status']=='completed' and not action and not ready:
                message='球员处理已完成，点击继续提取击球'
            if flow.get('error'):
                status,message='failed',flow['error']
            checkpoint=read(LIBRARY/'录像'/j['id']/'latest.json',{})
            jobs.append(dict(id=j['id'],video_id=j['video_id'],name=source['name'],kind=source['kind'],created_at=j['created_at'],status=status,
                             message=message,ready=ready,candidates=len(events),labelled=len(annotation['labels']),revision=annotation['revision'],
                             finished=bool(checkpoint.get('finished') and checkpoint.get('revision')==annotation['revision']),
                             origin=read(STORE/'sources'/(j['video_id']+'.json'),{}),
                             progress_current=action.get('stage_processed') or j.get('processed_frames',0),
                             progress_total=action.get('stage_total') or j.get('total_frames',0)))
        return {'jobs':sorted(jobs,key=lambda j:j['created_at'],reverse=True),
                'imports':sorted([read(f) for f in (STORE/'imports').glob('*.json')],key=lambda i:i['created_at'],reverse=True),
                'library':library_summary()}

    def shutdown(self):
        self.stop.set()
        self.import_worker.join(timeout=12)
        self.worker.join(timeout=5)
