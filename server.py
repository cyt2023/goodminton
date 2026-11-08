"""Local research prototype. Python standard library only; no external model calls."""
import json
import mimetypes
import os
from pathlib import Path
import secrets
import signal
import sqlite3
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse
import uuid

from phase_review import get_review, put_review, RevisionConflict
from analysis_jobs import AnalysisJobs, QueueFull
from conversation import get_conversation, respond
from action_search import initialize_action_search
from stroke_review import get_review as get_stroke_review, save_label, overlays, ReviewConflict
from learning_jobs import start_learning
from stroke_taxonomy import LABELS as STROKE_LABELS
from video_workflows import VideoWorkflows, save_training_data
from court_settings import line_color
import research_jobs

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / '03_数据'
DB = DATA / '研究数据.sqlite3'
TOKEN = secrets.token_urlsafe(32)
PORT = int(os.environ.get('BADMINTON_PORT', '8765'))
MAX_UPLOAD = 1024 ** 3
LABELS = {
    'stroke': {k:v for k,v in STROKE_LABELS.items() if k not in ['no_stroke','other']},
    'corner': {'unknown': '待确认', 'forehand': '正手后场', 'round_head': '头顶后场'},
    'incoming': {'unknown': '待确认', 'high_deep': '高且深', 'fast_flat': '平快', 'short': '来球偏短'},
    'arrival': {'unknown': '待确认', 'in_time': '提前到位', 'late': '到位较晚'},
    'balance': {'unknown': '待确认', 'stable': '较稳定', 'unstable': '失去平衡'},
}
FOCUS = {'preparation': '准备与引拍', 'contact': '击球附近', 'balance': '身体平衡', 'recovery': '落地与回位'}
KEYWORDS = {'preparation': ['准备', '引拍', '架拍', '来不及'], 'contact': ['击球', '发力', '到底线', '球短'],
            'balance': ['平衡', '站不稳', '后仰'], 'recovery': ['回位', '回动', '落地', '下一拍']}

def connection():
    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    return db

def initialize():
    DATA.mkdir(parents=True, exist_ok=True)
    for folder in ['个人录像', '职业示范']:
        (DATA / folder).mkdir(exist_ok=True)
    with connection() as db:
        db.execute('CREATE TABLE IF NOT EXISTS videos (id TEXT PRIMARY KEY, name TEXT, kind TEXT, path TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS annotations (id TEXT PRIMARY KEY, kind TEXT, payload TEXT)')

def records():
    with connection() as db:
        videos = [{k: row[k] for k in ['id', 'name', 'kind']} for row in db.execute('SELECT * FROM videos')]
        annotations = [dict(json.loads(row['payload']), id=row['id'], kind=row['kind'])
                       for row in db.execute('SELECT * FROM annotations ORDER BY rowid DESC')]
    try:
        pose_outputs = pose_index()
        pose_status = 'available' if pose_outputs else 'not_available'
    except (OSError, ValueError, TypeError):
        pose_outputs, pose_status = [], 'index_unreadable'
    has_pose = bool(pose_outputs)
    return {'videos': videos, 'annotations': annotations, 'labels': LABELS, 'focus': FOCUS,
            'mode': 'manual_rule_retrieval_and_offline_pose' if has_pose else 'manual_and_rule_retrieval',
            'model_connected': has_pose, 'cloud_model_connected': False,
            'offline_pose_available': has_pose, 'offline_pose_status': pose_status,
            'offline_pose_result_count': len(pose_outputs), 'automatic_diagnosis_available': False}

def pose_index():
    """Read offline model output without accepting a client-supplied file path."""
    file = (DATA / '姿态分析' / 'analysis_index.json').resolve()
    if not file.is_relative_to(DATA.resolve()):
        raise ValueError('姿态索引路径无效')
    if not file.is_file():
        return []
    items = json.loads(file.read_text(encoding='utf-8'))
    if not isinstance(items, list):
        raise ValueError('姿态索引格式错误')
    return [item for item in items if isinstance(item, dict) and isinstance(item.get('id'), str)]

def pose_result(result_id):
    item = next((item for item in pose_index() if item['id'] == result_id), None)
    if item is None:
        raise FileNotFoundError('姿态结果不存在')
    relative = item.get('result_file')
    if not isinstance(relative, str) or not relative:
        raise ValueError('姿态结果路径未登记')
    file = (DATA / relative).resolve()
    if not file.is_relative_to(DATA.resolve()) or not file.is_file():
        raise FileNotFoundError('姿态结果文件不存在')
    payload = json.loads(file.read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or not isinstance(payload.get('frames'), list):
        raise ValueError('姿态结果格式错误')
    return payload

def validated_annotation(body):
    kind = body.get('kind')
    if kind not in ['personal', 'reference']:
        raise ValueError('请选择个人案例或职业示范')
    video_id = body.get('video_id', '')
    source_url = str(body.get('source_url', '')).strip()[:2000]
    if video_id:
        with connection() as db:
            row = db.execute('SELECT kind FROM videos WHERE id=?', (video_id,)).fetchone()
        if not row or row['kind'] != kind:
            raise ValueError('录像不存在或类型不匹配')
    elif kind != 'reference' or urlparse(source_url).scheme not in ['https', 'http']:
        raise ValueError('请上传录像；职业示范也可填写公开视频地址')
    start, end = float(body.get('start', 0)), float(body.get('end', 0))
    if not 0 <= start < end or end > 86400:
        raise ValueError('片段结束时间必须大于开始时间，单位为秒')
    conditions = body.get('conditions', {})
    if not isinstance(conditions, dict):
        raise ValueError('条件格式错误')
    conditions = {key: value if value in options else 'unknown'
                  for key, options in LABELS.items() for value in [conditions.get(key, 'unknown')]}
    result = {'video_id': video_id, 'source_url': source_url, 'start': start, 'end': end,
              'conditions': conditions, 'focus': [f for f in body.get('focus', []) if f in FOCUS],
              'reviewed': body.get('reviewed') is True}
    target = body.get('target_player', 'unconfirmed')
    if target not in ['near', 'far', 'unconfirmed']:
        raise ValueError('分析对象需为近端、远端或待确认')
    result['target_player'] = target
    for key in ['title', 'question', 'intent', 'observation', 'teaching_note', 'limits', 'reviewer']:
        result[key] = str(body.get(key, '')).strip()[:4000]
    if not result['title']:
        raise ValueError('请填写案例或示范名称')
    if result['reviewed'] and not result['reviewer']:
        raise ValueError('标记已核对时，请填写核对人')
    result['evidence_source'] = 'manual_annotation'
    return kind, result

def retrieve(case_id):
    state = records()
    case = next((a for a in state['annotations'] if a['id'] == case_id and a['kind'] == 'personal'), None)
    if not case:
        raise ValueError('请先保存个人案例')
    focus = set(case['focus'])
    for key, words in KEYWORDS.items():
        if any(word in case['question'] for word in words):
            focus.add(key)
    candidates = []
    for ref in state['annotations']:
        if ref['kind'] != 'reference':
            continue
        matches, differences, unknown = [], [], []
        score = 0
        for key, options in LABELS.items():
            left, right = case['conditions'][key], ref['conditions'][key]
            if left == 'unknown' or right == 'unknown':
                unknown.append(key)
            elif left == right:
                matches.append(key)
                score += 3 if key == 'stroke' else 2
            else:
                differences.append(key)
                score -= 4 if key in ['arrival', 'balance'] else 2
        focus_matches = sorted(focus.intersection(ref['focus']))
        score += len(focus_matches) * 3
        candidates.append({'reference': ref, 'score': score, 'matches': matches, 'differences': differences,
                           'unknown': unknown, 'focus_matches': focus_matches,
                           'coverage': (len(matches) + len(differences)) / len(LABELS),
                           'status': '待核对' if not ref['reviewed'] else ('有条件差异' if differences else '待判断适用性')})
    candidates.sort(key=lambda item: (item['score'], item['reference']['reviewed']), reverse=True)
    return {'case': case, 'candidates': candidates[:5], 'mode': 'rule_retrieval',
            'notice': '排序依据人工标签与关键词。分数不是适用概率，也不是技术诊断；需要结合录像核对。'}

class Handler(BaseHTTPRequestHandler):
    def json_response(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def trusted_host(self):
        return self.headers.get('Host', '') in [f'127.0.0.1:{PORT}', f'localhost:{PORT}']

    def do_GET(self):
        if not self.trusted_host():
            return self.json_response({'error': '仅支持本机访问'}, 403)
        path = urlparse(self.path).path
        if path=='/api/architecture':return self.json_response(json.loads((ROOT/'research_models/architecture.json').read_text()))
        if path=='/api/professional-workflows':
            from professional_agents.web import listing
            return self.json_response(listing())
        if path.startswith('/api/professional-workflows/'):
            try:
                from professional_agents.web import details
                return self.json_response(details(path.removeprefix('/api/professional-workflows/')))
            except (ValueError,FileNotFoundError) as error:return self.json_response({'error':str(error)},400)
        if path=='/static/professional-workflows.js':
            data=(ROOT/'static/professional-workflows.js').read_bytes();self.send_response(200)
            self.send_header('Content-Type','application/javascript; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.end_headers();return self.wfile.write(data)
        if path=='/static/tactical-review.js':
            data=(ROOT/'static/tactical-review.js').read_bytes();self.send_response(200)
            self.send_header('Content-Type','application/javascript; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.end_headers();return self.wfile.write(data)
        if path.startswith('/hit-segment/'):
            try:
                parts=path.removeprefix('/hit-segment/').split('/')
                if len(parts)!=3 or not re.fullmatch(r'J_[0-9a-f]{32}',parts[0]) or not re.fullmatch(r'[0-9a-f]{32}',parts[1]):
                    raise ValueError('击球片段地址无效')
                from hit_segments import media
                query=parse_qs(urlparse(self.path).query)
                center=int(query['center'][0]) if 'center' in query else None
                player=query.get('player',[None])[0]
                return self.serve_media(media(*parts,center=center,player=player))
            except (ValueError,FileNotFoundError) as error:
                return self.json_response({'error':str(error)},400)
            except (OSError,RuntimeError):
                return self.json_response({'error':'击球片段暂时无法生成'},500)
        if path == '/api/review' or path.startswith('/api/review/') or path.startswith('/review-video/') or path.startswith('/review-frame/'):
            try:
                from benchmarks.bfmd import review_web
                if path == '/api/review':return self.json_response(review_web.listing())
                if path.startswith('/review-video/'):return self.serve_media(review_web.video(path.removeprefix('/review-video/')))
                if path.startswith('/review-frame/'):
                    data=review_web.frame(path.removeprefix('/review-frame/'),parse_qs(urlparse(self.path).query).get('t',['0'])[0])
                    self.send_response(200);self.send_header('Content-Type','image/jpeg');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();return self.wfile.write(data)
                return self.json_response(review_web.state(path.removeprefix('/api/review/')))
            except (ValueError,FileNotFoundError) as error:return self.json_response({'error':str(error)},400)
        if path == '/api/state':
            return self.json_response(records())
        if path == '/api/video-workflows':
            return self.json_response(self.server.workflows.snapshot())
        if path.startswith('/api/integrated-analysis/'):
            parts=path.removeprefix('/api/integrated-analysis/').split('/')
            try:
                job=parts[0];research_jobs.folder_for(job)
                action=parts[1] if len(parts)>1 else 'status'
                if action=='status':return self.json_response(self.server.research.snapshot(job))
                if action=='overlays':return self.json_response(research_jobs.overlays(job))
                if action=='tactical-review':return self.json_response(research_jobs.tactical_review(job))
                if action=='pose3d':return self.json_response(research_jobs.load(DATA/'专项分析'/job/'pose3d/status.json',{'status':'not_started'}))
                if action=='log':
                    file=research_jobs.folder_for(job)/'worker.log'
                    return self.json_response({'log':file.read_text(errors='replace')[-12000:] if file.exists() else ''})
                if action=='coaching' and len(parts)==3:
                    if not re.fullmatch(r'[0-9a-f]{32}',parts[2]):raise ValueError('事件标识无效')
                    return self.json_response(research_jobs.load(research_jobs.result_folder(job)/'coaching'/(parts[2]+'.json'),{'status':'not_started'}))
                value=research_jobs.report(job)
                if action=='report':return self.json_response(value)
                if action=='csv':
                    from research_report import csv_report
                    data=('\ufeff'+csv_report(value)).encode();self.send_response(200)
                    self.send_header('Content-Type','text/csv; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.end_headers();return self.wfile.write(data)
                raise ValueError('分析路径无效')
            except (ValueError,KeyError,OSError,TypeError) as error:
                return self.json_response({'error':str(error)},400)
        if path.startswith('/api/court-review/'):
            identifier = path.removeprefix('/api/court-review/')
            try:
                return self.json_response(self.server.workflows.court_snapshot(identifier))
            except (ValueError,OSError) as error:
                return self.json_response({'error':str(error)},400)
        if path.startswith('/court-preview/'):
            identifier = path.removeprefix('/court-preview/')
            try:
                self.server.workflows.court_item(identifier)
                file = DATA/'视频工作流/courts'/identifier/'preview.jpg'
                if not file.is_file():
                    return self.json_response({'error':'预览正在准备'},404)
                return self.serve_media(file)
            except (ValueError,OSError) as error:
                return self.json_response({'error':str(error)},400)
        if path.startswith('/api/stroke-review/'):
            try:
                parts = path.removeprefix('/api/stroke-review/').split('/')
                if len(parts)==1:
                    query=parse_qs(urlparse(self.path).query)
                    excluded=[eid for eid in query.get('exclude',[''])[0].split(',') if eid]
                    return self.json_response(get_stroke_review(parts[0],excluded))
                if len(parts)==3 and parts[1]=='clip':
                    return self.json_response(overlays(parts[0],parts[2]))
                raise ValueError('标注接口路径无效')
            except FileNotFoundError:
                return self.json_response({'error':'这段录像的专项分析尚未完成'},404)
            except (OSError,ValueError,KeyError,TypeError) as error:
                return self.json_response({'error':str(error)},400)
        if path.startswith('/api/specialist/'):
            identifier = unquote(path.removeprefix('/api/specialist/'))
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', identifier):
                return self.json_response({'error': '分析记录标识无效'}, 400)
            file = DATA / '专项分析' / identifier / 'report.json'
            if not file.is_file():
                return self.json_response({'error': '本轮分析记录尚未完成'}, 404)
            try:
                report = json.loads(file.read_text())
                pose = pose_result(identifier)
                if report['source_sha256'] != pose['source_sha256']:
                    return self.json_response({'error': '分析记录与当前录像不匹配'}, 409)
                return self.json_response(report)
            except (OSError, ValueError, KeyError, TypeError, StopIteration):
                return self.json_response({'error': '分析记录无法读取'}, 500)
        if path.startswith('/specialist-image/'):
            parts = unquote(path.removeprefix('/specialist-image/')).split('/')
            if (len(parts) != 2 or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', parts[0]) or
                    not re.fullmatch(r'[0-9a-f]{32}\.jpg', parts[1])):
                return self.json_response({'error': '片段图不存在'}, 404)
            file = (DATA / '技术动作' / parts[0] / '片段' / parts[1][:-4] / 'sequence_review.jpg').resolve()
            if not file.is_relative_to((DATA / '技术动作').resolve()) or not file.is_file():
                return self.json_response({'error': '片段图不存在'}, 404)
            return self.serve_media(file)
        if path.startswith('/api/conversation/'):
            try:
                identifier = unquote(path.removeprefix('/api/conversation/'))
                job = self.server.analysis_jobs.get(identifier)
                if job['status'] != 'completed':
                    return self.json_response({'error': '视频处理完成后再开始沟通'}, 409)
                pose = pose_result(job['result_id'])
                item = next(item for item in pose_index() if item['id'] == job['result_id'])
                return self.json_response(get_conversation(job, pose, item))
            except FileNotFoundError:
                return self.json_response({'error': '视频处理记录不存在'}, 404)
            except (ValueError, TypeError) as error:
                return self.json_response({'error': str(error)}, 400)
            except (OSError, sqlite3.Error, StopIteration):
                return self.json_response({'error': '这段视频的沟通记录暂时无法读取'}, 500)
        if path.startswith('/pose-preview/'):
            parts = unquote(path.removeprefix('/pose-preview/')).split('/')
            if len(parts) != 2 or parts[1] not in ['player_near.jpg', 'player_far.jpg']:
                return self.json_response({'error': '预览不存在'}, 404)
            item = next((item for item in pose_index() if item['id'] == parts[0]), None)
            if not item:
                return self.json_response({'error': '预览不存在'}, 404)
            file = (DATA / '姿态分析' / parts[0] / parts[1]).resolve()
            if not file.is_relative_to((DATA / '姿态分析').resolve()) or not file.is_file():
                return self.json_response({'error': '预览不存在'}, 404)
            return self.serve_media(file)
        if path.startswith('/action-preview/'):
            parts = unquote(path.removeprefix('/action-preview/')).split('/')
            if (len(parts) != 2 or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', parts[0]) or
                    not re.fullmatch(r'[0-9a-f]{32}\.jpg', parts[1])):
                return self.json_response({'error': '动作预览不存在'}, 404)
            file = (DATA / '技术动作' / parts[0] / '片段' / parts[1][:-4] / 'thumbnail.jpg').resolve()
            if not file.is_relative_to((DATA / '技术动作').resolve()) or not file.is_file():
                return self.json_response({'error': '动作预览不存在'}, 404)
            return self.serve_media(file)
        if path == '/api/analysis-jobs' or path.startswith('/api/analysis-jobs/') or path.startswith('/api/video-info/'):
            try:
                if path == '/api/analysis-jobs':
                    payload = self.server.analysis_jobs.list_jobs()
                elif path.startswith('/api/video-info/'):
                    payload = self.server.analysis_jobs.video_info(unquote(path.removeprefix('/api/video-info/')))
                else:
                    payload = self.server.analysis_jobs.get(unquote(path.removeprefix('/api/analysis-jobs/')))
                return self.json_response(payload)
            except FileNotFoundError:
                return self.json_response({'error': '分析任务不存在'}, 404)
            except (ValueError, TypeError) as error:
                return self.json_response({'error': str(error)}, 400)
            except RuntimeError as error:
                return self.json_response({'error': str(error)}, 503)
            except (OSError, sqlite3.Error):
                return self.json_response({'error': '本地分析环境或数据文件无法读取'}, 500)
        if path.startswith('/api/phase-review/'):
            try:
                result_id = unquote(path.removeprefix('/api/phase-review/'))
                return self.json_response(get_review(pose_result(result_id)))
            except FileNotFoundError:
                return self.json_response({'error': '对应姿态结果不存在'}, 404)
            except RevisionConflict as error:
                return self.json_response({'error': str(error)}, 409)
            except (OSError, ValueError, TypeError, KeyError) as error:
                return self.json_response({'error': '动作阶段记录无法读取：' + str(error)}, 500)
        if path == '/api/pose-index' or path.startswith('/api/pose/'):
            try:
                payload = pose_index() if path == '/api/pose-index' else pose_result(unquote(path.removeprefix('/api/pose/')))
                return self.json_response(payload)
            except FileNotFoundError:
                return self.json_response({'error': '姿态结果不存在'}, 404)
            except (OSError, ValueError, TypeError):
                return self.json_response({'error': '姿态分析文件无法读取，请检查离线处理结果'}, 500)
        if path.startswith('/media/'):
            video_id = path.removeprefix('/media/')
            with connection() as db:
                row = db.execute('SELECT path FROM videos WHERE id=?', (video_id,)).fetchone()
            if not row:
                return self.json_response({'error': '录像不存在'}, 404)
            file = (DATA / row['path']).resolve()
            if not file.is_relative_to(DATA.resolve()) or not file.is_file():
                return self.json_response({'error': '录像文件不存在'}, 404)
            return self.serve_media(file)
        if path == '/api/tokyo-benchmark':
            file = DATA/'职业录像/东京奥运会2021男单决赛/识别检查结果.json'
            if not file.is_file():
                return self.json_response({'error':'东京决赛检查结果尚未生成'},404)
            return self.json_response(json.loads(file.read_text()))
        if path in ['/', '/index.html', '/conversation', '/research', '/research.html', '/specialist', '/label-strokes', '/tokyo-check', '/court-review', '/analysis', '/review']:
            file = ROOT / 'static' / ('review.html' if path=='/review' else 'analysis.html' if path=='/analysis' else 'court_review.html' if path == '/court-review' else 'conversation.html' if path == '/conversation' else 'tokyo_check.html' if path == '/tokyo-check' else 'label_strokes.html' if path == '/label-strokes' else 'specialist.html' if path == '/specialist' else 'research.html' if path.startswith('/research') else 'index.html')
            data = file.read_text().replace('__SESSION_TOKEN__', TOKEN).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            return self.wfile.write(data)
        return self.json_response({'error': '页面不存在'}, 404)

    def serve_media(self, file):
        size = file.stat().st_size
        start, end, partial = 0, size - 1, False
        value = self.headers.get('Range')
        if value:
            try:
                if not value.startswith('bytes=') or ',' in value:
                    raise ValueError()
                left, right = value[6:].split('-', 1)
                if left:
                    start = int(left)
                    end = min(int(right), size - 1) if right else size - 1
                else:
                    count = int(right)
                    if count <= 0:
                        raise ValueError()
                    start = max(0, size - count)
                if not 0 <= start <= end < size:
                    raise ValueError()
                partial = True
            except ValueError:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type', mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(end - start + 1))
        if partial:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        try:
            with file.open('rb') as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        if not self.trusted_host() or self.headers.get('X-Session-Token') != TOKEN:
            return self.json_response({'error': '请从本机页面提交'}, 403)
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0:
                raise ValueError('请求内容为空')
            if parsed.path=='/api/professional-workflows':
                if length>4096:raise ValueError('工作流请求过大')
                from professional_agents.web import start
                return self.json_response(start(json.loads(self.rfile.read(length))))
            if parsed.path in ['/api/upload', '/api/upload-auto']:
                if length > MAX_UPLOAD:
                    return self.json_response({'error': '每个录像最多 1GB，请先导出较短片段'}, 413)
                query = parse_qs(parsed.query)
                name = Path(query.get('name', ['video.mp4'])[0]).name[:200]
                kind = query.get('kind', ['personal'])[0]
                color = line_color(query.get('court_line_color', [None])[0])
                suffix = Path(name).suffix.lower()
                if kind not in ['personal', 'reference'] or suffix not in ['.mp4', '.mov', '.m4v', '.webm']:
                    raise ValueError('请选择 MP4、MOV、M4V 或 WebM 录像')
                video_id = uuid.uuid4().hex
                relative = Path('个人录像' if kind == 'personal' else '职业示范') / (video_id + suffix)
                file = DATA / relative
                try:
                    with file.open('wb') as stream:
                        remaining = length
                        while remaining:
                            chunk = self.rfile.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise ValueError('上传中断，请重试')
                            stream.write(chunk)
                            remaining -= len(chunk)
                    with connection() as db:
                        db.execute('INSERT INTO videos VALUES (?,?,?,?)', (video_id, name, kind, str(relative)))
                except Exception:
                    file.unlink(missing_ok=True)
                    raise
                if parsed.path == '/api/upload-auto':
                    try:
                        if query.get('workflow') == ['1']:
                            imported = self.server.workflows.submit({'video_id':video_id, 'court_line_color':color})
                            return self.json_response({'video_id':video_id,'name':name,'import':imported},202)
                        job = self.server.analysis_jobs.submit_auto(video_id)
                        return self.json_response({'video_id': video_id, 'name': name, 'job': job}, 202)
                    except QueueFull as error:
                        return self.json_response({'error': str(error), 'video_id': video_id, 'name': name}, 409)
                    except RuntimeError as error:
                        return self.json_response({'error': str(error), 'video_id': video_id, 'name': name}, 503)
                    except (ValueError, TypeError) as error:
                        return self.json_response({'error': str(error), 'video_id': video_id, 'name': name}, 400)
                    except (OSError, sqlite3.Error):
                        return self.json_response({'error': '视频已保存，暂时无法开始处理，请稍后重试', 'video_id': video_id, 'name': name}, 500)
                return self.json_response({'id': video_id, 'name': name})
            if length > 1024 * 1024:
                return self.json_response({'error': '标注内容过大'}, 413)
            body = json.loads(self.rfile.read(length))
            if parsed.path.startswith('/api/review/'):
                from benchmarks.bfmd import review_web
                clip=parsed.path.removeprefix('/api/review/')
                return self.json_response(review_web.reset(clip) if body.get('reset') else review_web.act(clip,body))
            if parsed.path.startswith('/api/integrated-analysis/'):
                parts=parsed.path.removeprefix('/api/integrated-analysis/').split('/')
                job=parts[0];research_jobs.folder_for(job)
                if len(parts)==2 and parts[1]=='start':return self.json_response(self.server.research.submit(job),202)
                if len(parts)==3 and parts[1]=='import':return self.json_response(research_jobs.import_results(job,parts[2],body))
                if len(parts)==2 and parts[1]=='rallies':return self.json_response(research_jobs.import_rallies(job,body))
                if len(parts)==2 and parts[1]=='tactical-review':return self.json_response(research_jobs.start_tactical_coaching(job),202)
                if len(parts)==3 and parts[1]=='pose3d':
                    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',parts[2]):raise ValueError('事件标识无效')
                    return self.json_response(research_jobs.start_pose3d(job,parts[2]),202)
                if len(parts)==3 and parts[1]=='coaching':
                    # Event identifiers from imports remain data, never filesystem paths.
                    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',parts[2]):raise ValueError('事件标识无效')
                    return self.json_response(research_jobs.start_coaching(job,parts[2]),202)
                raise ValueError('分析操作路径无效')
            if parsed.path.startswith('/api/court-review/'):
                parts = parsed.path.removeprefix('/api/court-review/').split('/')
                if len(parts) != 2:
                    raise ValueError('场地操作路径无效')
                return self.json_response(self.server.workflows.update_court(parts[0],parts[1],body))
            if parsed.path == '/api/video-workflows/import':
                return self.json_response(self.server.workflows.submit(body),202)
            if parsed.path == '/api/video-workflows/retry':
                return self.json_response(self.server.workflows.retry(body.get('job_id')),202)
            if parsed.path.startswith('/api/stroke-review/'):
                parts=parsed.path.removeprefix('/api/stroke-review/').split('/')
                if len(parts)!=2 or parts[1] not in ['label','learn','finish']:
                    raise ValueError('标注接口路径无效')
                try:
                    if parts[1]=='finish':
                        result=save_training_data(parts[0],finish=True,expected_revision=body.get('revision'))
                    elif parts[1]=='learn':
                        result=start_learning(parts[0])
                    else:
                        result=save_label(parts[0],body)
                        try:
                            save_training_data(parts[0])
                            from bst_finetune_state import library_readiness
                            result['library_learning_readiness'] = library_readiness()
                        except (OSError,ValueError,KeyError) as error:
                            result['library_warning']='原标注已保存；训练库同步暂未完成：'+str(error)[:200]
                    return self.json_response(result,202 if parts[1]=='learn' else 200)
                except ReviewConflict as error:
                    return self.json_response({'error':str(error)},409)
            if parsed.path == '/api/auto-analysis':
                try:
                    return self.json_response(self.server.analysis_jobs.submit_auto(body.get('video_id')), 202)
                except QueueFull as error:
                    return self.json_response({'error': str(error)}, 409)
                except RuntimeError as error:
                    return self.json_response({'error': str(error)}, 503)
            if parsed.path.startswith('/api/conversation/'):
                identifier = unquote(parsed.path.removeprefix('/api/conversation/'))
                try:
                    job = self.server.analysis_jobs.get(identifier)
                    if job['status'] != 'completed':
                        return self.json_response({'error': '视频处理完成后再开始沟通'}, 409)
                    pose = pose_result(job['result_id'])
                    item = next(item for item in pose_index() if item['id'] == job['result_id'])
                    return self.json_response(respond(job, pose, item, body))
                except FileNotFoundError:
                    return self.json_response({'error': '视频处理记录不存在'}, 404)
                except (OSError, sqlite3.Error, StopIteration):
                    return self.json_response({'error': '沟通内容未能保存，请稍后重试'}, 500)
            if parsed.path == '/api/analysis-jobs':
                try:
                    return self.json_response(self.server.analysis_jobs.submit(body), 202)
                except QueueFull as error:
                    return self.json_response({'error': str(error)}, 409)
                except RuntimeError as error:
                    return self.json_response({'error': str(error)}, 503)
                except (OSError, sqlite3.Error):
                    return self.json_response({'error': '分析任务无法保存，请检查本地数据文件夹'}, 500)
            if parsed.path.startswith('/api/analysis-jobs/') and parsed.path.endswith('/cancel'):
                identifier = unquote(parsed.path[len('/api/analysis-jobs/'):-len('/cancel')])
                try:
                    return self.json_response(self.server.analysis_jobs.cancel(identifier))
                except FileNotFoundError:
                    return self.json_response({'error': '分析任务不存在'}, 404)
                except RuntimeError as error:
                    return self.json_response({'error': str(error)}, 503)
                except (OSError, sqlite3.Error):
                    return self.json_response({'error': '分析任务无法保存，请检查本地数据文件夹'}, 500)
            if parsed.path.startswith('/api/phase-review/'):
                try:
                    result_id = unquote(parsed.path.removeprefix('/api/phase-review/'))
                    return self.json_response(put_review(pose_result(result_id), body))
                except FileNotFoundError:
                    return self.json_response({'error': '对应姿态结果不存在'}, 404)
                except RevisionConflict as error:
                    return self.json_response({'error': str(error)}, 409)
                except OSError:
                    return self.json_response({'error': '阶段记录写入失败，当前草稿仍可导出；请检查数据文件夹'}, 500)
            if parsed.path == '/api/annotations':
                kind, payload = validated_annotation(body)
                item_id = uuid.uuid4().hex
                with connection() as db:
                    db.execute('INSERT INTO annotations VALUES (?,?,?)',
                               (item_id, kind, json.dumps(payload, ensure_ascii=False)))
                return self.json_response({'id': item_id})
            if parsed.path == '/api/retrieve':
                return self.json_response(retrieve(body.get('case_id')))
            return self.json_response({'error': '接口不存在'}, 404)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            return self.json_response({'error': str(error)}, 400)
        except (OSError, sqlite3.Error):
            return self.json_response({'error': '本地数据暂时无法读写，请检查磁盘后重试'}, 500)

if __name__ == '__main__':
    initialize()
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    manager = None
    action_manager = None
    workflows = None
    research_manager = None

    def stop_service(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_service)
    try:
        manager = AnalysisJobs()
        server.analysis_jobs = manager
        manager.start()
        action_manager = initialize_action_search()
        workflows = VideoWorkflows(manager, action_manager)
        server.workflows = workflows
        research_manager = research_jobs.ResearchJobs()
        server.research = research_manager
        print(f'羽毛球论文研究原型：http://127.0.0.1:{PORT}', flush=True)
        print('本地保存与骨架分析；尚未接入云端 AI。Ctrl+C 停止。', flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        print('正在停止本地处理服务…', flush=True)
    finally:
        if research_manager is not None:
            research_manager.shutdown()
        if workflows is not None:
            workflows.shutdown()
        if action_manager is not None:
            action_manager.shutdown()
        if manager is not None:
            manager.shutdown()
        server.server_close()
