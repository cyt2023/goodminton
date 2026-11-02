"""Persisted, evidence-guided local conversation after a pose job completes.

Technical questions request isolated vision-language action search; conversation
replies remain bounded by returned candidates and visible pose evidence. Stroke
classification does not diagnose stroke quality or verify professional examples.
"""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import threading
import uuid

from phase_review import wrist_signals
from action_search import ENGINE_VERSION, initialize_action_search, request_search, refresh_search
from stroke_taxonomy import TERMS as STROKE_TERMS, VERSION as TAXONOMY_VERSION
from stroke_review import read_labels


DATA = Path(__file__).resolve().parent.parent / '03_数据'
ROOT = DATA / '沟通'
LOCK = threading.RLock()
SIDES = ['near', 'far']
LABELS = {'near': '候选球员 A', 'far': '候选球员 B'}
COLORS = {'near': '#24b6e2', 'far': '#f7a943'}
MAX_MESSAGES = 1000
FOCUS_WORDS = [
    ('contact', ['触球', '击球点', '接触', '打点', '击球时刻']),
    ('backswing', ['引拍', '架拍', '挥拍', '发力', '杀球', '扣杀', '高远', '吊球']),
    ('preparation', ['准备', '启动', '预判', '分腿', '步法', '移动']),
    ('recovery', ['回位', '随挥', '落地', '回中', '还原', '下一拍']),
    ('balance', ['平衡', '重心', '躯干', '腰', '转体', '膝盖', '受力']),
]
STYLE_TERMS = [('backhand', '反手', ['反手', '反拍']), ('forehand', '正手', ['正手', '正拍'])]
SEARCH_STATUSES = {'queued', 'running', 'completed', 'failed', 'unavailable'}


def now():
    return datetime.now(timezone.utc).isoformat()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def time_text(value):
    return f'{float(value):.2f}'.rstrip('0').rstrip('.')


def path_for(job):
    identifier = job.get('id')
    if not isinstance(identifier, str) or not re.fullmatch(r'J_[0-9a-f]{32}', identifier):
        raise ValueError('沟通任务 ID 无效')
    file = (ROOT / identifier / 'conversation.json').resolve()
    if not file.is_relative_to(DATA.resolve()):
        raise ValueError('沟通记录路径无效')
    return file


def atomic_json(file, value):
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


def validate_result(job, pose, index_item):
    if not isinstance(job, dict) or job.get('status') != 'completed':
        raise ValueError('视频处理完成后才能开始沟通')
    path_for(job)
    if (not isinstance(pose, dict) or pose.get('id') != job.get('result_id') or
            pose.get('id') != job['id'] or pose.get('source_video_id') != job.get('video_id') or
            not isinstance(pose.get('frames'), list) or not finite(pose.get('start')) or
            not finite(pose.get('end')) or not 0 <= pose['start'] < pose['end'] or
            pose['start'] != job.get('start') or pose['end'] != job.get('end') or
            not isinstance(pose.get('source_sha256'), str) or
            not re.fullmatch(r'[0-9a-f]{64}', pose['source_sha256']) or
            not finite(pose.get('sample_fps')) or pose['sample_fps'] <= 0 or
            not finite(pose.get('display_threshold'))):
        raise ValueError('沟通所需的视频处理结果不匹配')
    if (not isinstance(index_item, dict) or index_item.get('id') != pose['id'] or
            index_item.get('source_video_id') != pose['source_video_id'] or
            not isinstance(index_item.get('overlay_video_id'), str) or
            not re.fullmatch(r'[0-9a-f]{32}', index_item['overlay_video_id'])):
        raise ValueError('沟通所需的发布记录不匹配')


def frames_for(pose, side):
    return [frame for frame in pose['frames']
            if isinstance(frame, dict) and finite(frame.get('source_time')) and
            isinstance(frame.get('players'), dict) and isinstance(frame['players'].get(side), dict)]


def choices_for(job, pose):
    choices = []
    for side in SIDES:
        frames = frames_for(pose, side)
        if not frames:
            continue
        first = min(frames, key=lambda frame: frame['source_time'])
        choice = {'id': side, 'label': LABELS[side], 'time': first['source_time'],
                  'color': COLORS[side], 'assigned_frames': len(frames),
                  'total_frames': len(pose['frames']), 'identity_status': 'unreviewed'}
        bbox = first['players'][side].get('bbox')
        width, height = pose.get('source_width'), pose.get('source_height')
        if (isinstance(bbox, list) and len(bbox) == 4 and all(finite(x) for x in bbox) and
                finite(width) and finite(height) and width > 0 and height > 0):
            choice['bbox_normalized'] = [max(0, min(1, x / (width if i % 2 == 0 else height)))
                                         for i, x in enumerate(bbox)]
        thumbnail = (DATA / '姿态分析' / job['id'] / ('player_' + side + '.jpg')).resolve()
        if thumbnail.is_relative_to(DATA.resolve()) and thumbnail.is_file():
            choice['thumbnail_url'] = '/pose-preview/' + job['id'] + '/player_' + side + '.jpg'
        choices.append(choice)
    return choices


def message(role, content, evidence_windows=None):
    value = {'id': uuid.uuid4().hex, 'role': role, 'content': content, 'created_at': now()}
    if evidence_windows:
        value['evidence_windows'] = evidence_windows
    return value


def initial_document(job, pose, choices):
    if len(choices) == 1:
        greeting = '视频处理完成，我先看画面中的这位球员。你最想改进哪方面？'
    elif choices:
        greeting = f'视频处理完成，找到了 {len(choices)} 位候选球员。你想看哪一位，最想改进哪方面？'
    else:
        greeting = ('视频处理完成，但这段画面还没有找到可回看的球员姿态。'
                    '请换一段球员清楚可见的视频；你也可以先告诉我想改进什么。')
    timestamp = now()
    return {'schema_version': 1, 'job_id': job['id'], 'pose_id': pose['id'],
            'source_video_id': pose['source_video_id'], 'source_sha256': pose.get('source_sha256'),
            'selected_player': choices[0]['id'] if len(choices) == 1 else None,
            'focus': None, 'question': '', 'evidence_windows': [],
            'technical_query': None, 'search_context': None, 'technical_search': None,
            'messages': [message('assistant', greeting)], 'created_at': timestamp, 'updated_at': timestamp}


def read_document(job, pose, choices):
    file = path_for(job)
    if not file.is_file():
        document = initial_document(job, pose, choices)
        atomic_json(file, document)
        return document
    document = json.loads(file.read_text(encoding='utf-8'))
    if (not isinstance(document, dict) or document.get('schema_version') != 1 or
            document.get('job_id') != job['id'] or document.get('pose_id') != pose['id'] or
            document.get('source_video_id') != pose['source_video_id'] or
            document.get('source_sha256') != pose.get('source_sha256') or
            document.get('selected_player') not in [None, 'near', 'far', 'both'] or
            not isinstance(document.get('messages'), list) or len(document['messages']) > MAX_MESSAGES or
            not isinstance(document.get('question'), str) or not isinstance(document.get('evidence_windows'), list)):
        raise ValueError('已有沟通记录与视频结果不匹配，无法继续写入')
    for item in document['messages']:
        if (not isinstance(item, dict) or item.get('role') not in ['user', 'assistant'] or
                not isinstance(item.get('content'), str) or not isinstance(item.get('created_at'), str)):
            raise ValueError('已有沟通记录格式无法读取')
    # Add new context fields without rewriting historical messages.
    document.setdefault('technical_query', None)
    document.setdefault('search_context', None)
    document.setdefault('technical_search', None)
    return document


def public_document(document, job, pose, index_item, choices):
    # Make a detached copy: a request handler cannot mutate persisted context.
    result = json.loads(json.dumps(document, ensure_ascii=False, allow_nan=False))
    result.update(choices=choices, source_video_url='/media/' + pose['source_video_id'],
                  overlay_url='/media/' + index_item['overlay_video_id'],
                  window={'start': pose['start'], 'end': effective_end(pose), 'time_basis': pose.get('time_basis')},
                  capabilities={'local_evidence_responses': True,
                                'local_vision_language_search_requested': bool(document.get('technical_search')),
                                'automatic_technical_diagnosis': False,
                                'verified_professional_demonstrations': False})
    return result


def effective_end(pose):
    return min(pose['end'], pose['processed_end']) if finite(pose.get('processed_end')) else pose['end']


def get_conversation(job, pose, index_item):
    validate_result(job, pose, index_item)
    with LOCK:
        choices = choices_for(job, pose)
        document = read_document(job, pose, choices)
        changed = migrate_legacy_question(document, job, pose, choices)
        if document.get('search_context'):
            changed = refresh_document_search(document, job, pose) or changed
        if changed:
            atomic_json(path_for(job), document)
        return public_document(document, job, pose, index_item, choices)


def requested_times(text):
    """Only explicit playback seconds or m:ss; numbers like joint angles do not count."""
    times = []
    for match in re.finditer(r'(?<![\d.])(-?\d{1,6}(?:\.\d{1,3})?)\s*(?:–|—|-|到|至)\s*(-?\d{1,6}(?:\.\d{1,3})?)\s*秒', text):
        times.extend(float(value) for value in match.groups())
    for match in re.finditer(r'(?<![\d.])(-?\d{1,6}(?:\.\d{1,3})?)\s*(?:秒|s\b)', text, re.I):
        times.append(float(match.group(1)))
    for match in re.finditer(r'(?<!\d)(\d{1,4}):([0-5]?\d(?:\.\d{1,3})?)(?!\d)', text):
        times.append(float(match.group(1)) * 60 + float(match.group(2)))
    return list(dict.fromkeys(times))


def focus_for(text):
    for focus, words in FOCUS_WORDS:
        if any(active_occurrences(text, word) for word in words):
            return focus
    return None


def active_occurrences(text, term, style=False):
    pattern = re.escape(term)
    if term.isascii():
        pattern = r'\b' + pattern + r'\b'
    found = []
    for match in re.finditer(pattern, text, re.I):
        before, after = text[max(0, match.start() - 20):match.start()], text[match.end():match.end() + 20]
        # "反手不好/不会反手" are problems to search, not exclusions. Only
        # explicit denial of the requested action is treated as negation.
        negated = re.search(
            r'(?:不是|并非|而非|非|不要|别|不用|不想|不需要|不看|不找|不练|不打|不关注|除了|排除)'
            r'(?:给我|帮我|替我|先|再|继续|去|来|专门|只){0,3}'
            r'(?:看|找|搜|检索|分析|讨论|练习|练|打|关注|做)?\s*$', before)
        negated = negated or re.search(r'(?:not|exclude|do not|don\'t)\s+(?:search|show|find)?\s*$', before, re.I)
        negated = negated or re.match(
            r'\s*(?:不是(?:我|我们)?(?:的)?(?:问题|目标|想看的)|(?:我)?不(?:关心|想看|需要))', after)
        if negated:
            continue
        if style and re.match(r'\s*(?:区|位|侧|边)', after):
            # A court side is not evidence of a forehand/backhand stroke.
            continue
        if term in ['发球', 'serve'] and re.search(r'(?:接|接到)\s*$', before):
            continue
        found.append(match.start())
    return found


def technical_components(text):
    strokes, styles = [], []
    for stroke, label, terms in STROKE_TERMS:
        occurrences = [(term, active_occurrences(text, term)) for term in terms + [stroke]]
        if stroke in ['transition', 'drive']:
            # Ignore a generic substring inside a named subtype. Keep a
            # separately mentioned generic request, including in mixed queries.
            children = ({'transition_hard', 'transition_soft'} if stroke == 'transition'
                        else {'drive_rear_flat', 'drive_rear_high'})
            specific_terms = [term for code, _, aliases in STROKE_TERMS
                              if code in children for term in aliases]
            if stroke == 'drive':
                specific_terms.append('后场抽球')
            spans = [(m.start(), m.end()) for name in specific_terms
                     for m in re.finditer(re.escape(name), text)]
            occurrences = [(term, [a for a in starts
                                   if not any(lo <= a < hi for lo, hi in spans)])
                           for term, starts in occurrences]
        if stroke == 'smash':
            # “接杀球” names defense as the requested action.
            occurrences = [(term, [a for a in spans
                                   if not re.search(r'(?:接|挡|防|防守)$', text[max(0, a - 3):a])])
                           for term, spans in occurrences]
        if stroke == 'defense':
            # A specific return to the rear court or deep net should not
            # also request the generic net block just because it says 接杀.
            specific_returns = ['接杀弹后场', '接杀弹到后场', '接杀弹球到后场', '接杀挡远网']
            occurrences = [(term, [a for a in spans
                                   if not any(text.startswith(name, a) for name in specific_returns)])
                           for term, spans in occurrences]
        if any(spans for _, spans in occurrences):
            strokes.append((stroke, label))
    # Broad requests retain their broad intent without equating push/kill.
    if active_occurrences(text, '网前球'):
        for code,label in [('net','放网'),('net_spin','搓球'),('net_cross','勾对角'),('push','推球'),('net_kill','扑球')]:
            if code not in [s[0] for s in strokes]:strokes.append((code,label))
    if active_occurrences(text, '后场抽球'):
        for code, label in [('drive_rear_flat', '后场抽平'), ('drive_rear_high', '后场抽高')]:
            if code not in [s[0] for s in strokes]:strokes.append((code,label))
    if any(code in ['serve_short','serve_long'] for code,_ in strokes):
        strokes=[s for s in strokes if s[0]!='serve']
    if '发' in text and any(code in ['serve_short','serve_long'] for code,_ in strokes):
        strokes=[s for s in strokes if s[0] not in ['clear','clear_flat']]
    for style, label, terms in STYLE_TERMS:
        if any(active_occurrences(text, term, style=True) for term in terms + [style]):
            styles.append((style, label))
    return strokes, styles


def parse_technical_query(text):
    strokes, styles = technical_components(text)
    if not (strokes or styles) or len(styles) > 1:
        return None
    style = styles[0][0] if styles else None
    label = (styles[0][1] if styles else '') + '、'.join(item[1] for item in strokes)
    return {'strokes': [item[0] for item in strokes], 'style': style, 'label': label}


def same_query(left, right):
    return (isinstance(left, dict) and isinstance(right, dict) and
            left.get('style') == right.get('style') and
            isinstance(left.get('strokes'), list) and isinstance(right.get('strokes'), list) and
            all(isinstance(item, str) for item in left['strokes'] + right['strokes']) and
            sorted(left['strokes']) == sorted(right['strokes']))


def valid_query(query):
    stroke_names = {item[0] for item in STROKE_TERMS}
    return (isinstance(query, dict) and isinstance(query.get('strokes'), list) and
            all(isinstance(stroke, str) and stroke in stroke_names for stroke in query['strokes']) and
            query.get('style') in [None, 'forehand', 'backhand'] and
            bool(query['strokes'] or query.get('style')) and isinstance(query.get('label'), str))


def search_sides(document, choices):
    return selected_sides(document['selected_player'], choices) or [choice['id'] for choice in choices]


def unavailable_search(query, sides, error):
    return {'id': 'S_' + uuid.uuid4().hex, 'query': query, 'sides': sides, 'status': 'unavailable',
            'matches': [], 'processed': 0, 'total': 0,
            'message': '技术片段检索暂时不可用', 'error': str(error)[:1000]}


def clean_search_result(result, context, pose):
    if (not isinstance(result, dict) or result.get('id') != context['id'] or
            not isinstance(result.get('status'), str) or
            result.get('status') not in SEARCH_STATUSES or not same_query(result.get('query'), context['query'])):
        return None
    matches = []
    candidates = result.get('matches', [])
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if (not isinstance(candidate, dict) or candidate.get('player') not in context['sides'] or
                not finite(candidate.get('start')) or not finite(candidate.get('end')) or
                not pose['start'] <= candidate['start'] < candidate['end'] <= effective_end(pose) or
                not finite(candidate.get('peak_time')) or
                not candidate['start'] <= candidate['peak_time'] <= candidate['end'] or
                candidate.get('classification_status') not in ['model_candidate', 'human_confirmed'] or
                not candidate.get('classification_source') or not isinstance(candidate.get('event_id'), str)):
            continue
        matches.append(dict(candidate, kind='technical_action_candidate',
                            interpretation='球种由用户核对；未判断技术好坏' if candidate['classification_status']=='human_confirmed' else '模型提出的技术片段候选；需要核对，不是技术错误或正确动作结论'))
    return dict(result, query=context['query'], matches=matches)


def search_reply(search, query):
    label = query['label']
    count, status = len(search.get('matches', [])), search['status']
    if count:
        confirmed=sum(c.get('classification_status')=='human_confirmed' for c in search.get('matches',[]))
        content = f'找到 {count} 处{label}片段，其中 {confirmed} 处球种已经由你确认。' if confirmed else f'先给你 {count} 处疑似{label}片段，点击片段看看是不是你想看的动作。'
        if status in ['queued', 'running']:
            content += '我还在继续检索更多片段。'
        elif status in ['failed', 'unavailable']:
            content += '后续检索暂时中断，目前先保留这些候选。'
        content += '先回看这些候选，确认是不是你说的动作。'
        return content
    if status == 'completed':
        extra = '有些片段识别未完成。' if search.get('classification_failures') else ''
        return f'这次暂时没有找到能辨认的{label}片段。{extra}可能有漏识或画面看不清，不能据此判断视频里没有。你也可以告诉我另一个想看的技术。'
    if status in ['failed', 'unavailable']:
        return f'{label}片段检索暂时不可用，这次没有得到可回看的技术候选。可以稍后重试，或换一段动作清楚的视频。'
    return f'我正在视频里找疑似{label}片段，找到后会直接给你可点击的片段。'


def update_search_message(document, context, search):
    target = next((item for item in document['messages']
                   if item.get('id') == context['assistant_message_id'] and
                   item.get('role') == 'assistant' and item.get('search_id') == context['id']), None)
    if target is None:
        return False
    content, windows = search_reply(search, context['query']), search.get('matches', [])
    if (target['content'] == content and target.get('evidence_windows', []) == windows and
            target.get('search_status') == search['status']):
        return False
    target.setdefault('revisions', []).append({
        'content': target['content'], 'evidence_windows': target.get('evidence_windows', []),
        'search_status': target.get('search_status'),
        'generated_at': target.get('generated_at', target['created_at']),
        'superseded_at': now(), 'reason': 'technical_search_refresh',
    })
    target.update(content=content, evidence_windows=windows, search_status=search['status'], generated_at=now())
    return True


def refresh_document_search(document, job, pose):
    from hit_segments import VERSION as WINDOW_VERSION
    from hit_refinement import fingerprint as refinement_fingerprint
    context = document.get('search_context')
    if (not isinstance(context, dict) or not isinstance(context.get('id'), str) or
            not valid_query(context.get('query')) or not isinstance(context.get('sides'), list) or
            not isinstance(context.get('assistant_message_id'), str)):
        return False
    previous = document.get('technical_search')
    # Terminal searches are already persisted; GET does not restart a failed job.
    if (isinstance(previous, dict) and previous.get('id') == context['id'] and
            previous.get('engine_version') == ENGINE_VERSION and
            previous.get('taxonomy_version') == TAXONOMY_VERSION and
            previous.get('hit_window_version') == WINDOW_VERSION and
            previous.get('hit_refinement_fingerprint') == refinement_fingerprint(job['id']) and
            previous.get('annotation_revision') == read_labels(job['id'], pose['source_sha256'])['revision'] and
            previous.get('source_sha256') == pose['source_sha256'] and
            previous.get('pose_fingerprint') == initialize_action_search().fingerprint(pose) and
            previous.get('status') in ['completed', 'failed', 'unavailable']):
        return False
    try:
        current = refresh_search(job, pose, context)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        current = dict(unavailable_search(context['query'], context['sides'], error), id=context['id'], status='failed')
    current = clean_search_result(current, context, pose)
    # A delayed response from another query never replaces the current message.
    if current is None or document.get('search_context', {}).get('id') != context['id']:
        return False
    changed = current != previous
    message_changed = update_search_message(document, context, current)
    document['technical_search'] = current
    document['evidence_windows'] = current.get('matches', [])
    if changed or message_changed:
        document['updated_at'] = now()
    return changed or message_changed


def start_document_search(document, job, pose, choices, query, existing_message=None):
    sides = search_sides(document, choices)
    try:
        result = request_search(job, pose, query, sides)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        result = unavailable_search(query, sides, error)
    identifier = result.get('id') if isinstance(result, dict) else None
    if not isinstance(identifier, str) or not identifier:
        result = unavailable_search(query, sides, '检索未返回有效 ID')
        identifier = result['id']
    assistant = existing_message if existing_message is not None else message('assistant', '')
    context = {'id': identifier, 'query': query, 'sides': sides, 'assistant_message_id': assistant['id']}
    result = clean_search_result(result, context, pose) or dict(
        unavailable_search(query, sides, '检索响应与请求不一致'), id=identifier)
    if existing_message is not None:
        assistant.setdefault('revisions', []).append({
            'content': assistant['content'], 'evidence_windows': assistant.get('evidence_windows', []),
            'generated_at': assistant.get('generated_at', assistant['created_at']), 'superseded_at': now(),
            'reason': 'legacy_technical_question_migration',
        })
    assistant.update(content=search_reply(result, query), search_id=identifier, search_status=result['status'],
                     evidence_windows=result.get('matches', []), generated_at=now())
    if existing_message is None:
        document['messages'].append(assistant)
    document.update(technical_query=query, search_context=context, technical_search=result,
                    evidence_windows=result.get('matches', []), focus='technical_action')


def migrate_legacy_question(document, job, pose, choices):
    if document.get('search_context') or document.get('technical_search_migration_version') == 1 or not choices:
        return False
    question = document.get('question', '')
    query = parse_technical_query(question)
    # Only continue an authentic stored user question, never synthesize one.
    if not query or not any(item['role'] == 'user' and item['content'] == question for item in document['messages']):
        return False
    last = document['messages'][-1] if document['messages'] else None
    if not last or last['role'] != 'assistant' or last.get('search_id'):
        return False
    legacy_reply = (any(window.get('kind') == 'user_time_or_clip_start' for window in last.get('evidence_windows', [])
                        if isinstance(window, dict)) or '哪一秒' in last['content'] or
                    '可以先帮你定位回看片段' in last['content'])
    if not legacy_reply:
        return False
    start_document_search(document, job, pose, choices, query, existing_message=last)
    document['technical_search_migration_version'] = 1
    document['updated_at'] = now()
    return True


def clear_search_context(document):
    old = document.get('search_context')
    if isinstance(old, dict):
        for item in document['messages']:
            if item.get('role') == 'assistant' and item.get('search_id') == old.get('id'):
                item['search_context_status'] = 'superseded'
    document.update(technical_query=None, search_context=None, technical_search=None, evidence_windows=[])


def selected_sides(selected, choices):
    available = {choice['id'] for choice in choices}
    return [side for side in SIDES if side in available and (selected == 'both' or side == selected)]


def coverage_note(pose, sides, arms=False):
    observations = []
    total = len(pose['frames'])
    for side in sides:
        frames = frames_for(pose, side)
        if len(frames) < total:
            observations.append(f'{LABELS[side]}有 {total - len(frames)} 帧未得到姿态')
        if arms and frames:
            known = 0
            for frame in frames:
                joints = frame['players'][side].get('keypoints', [])
                wrists = [joint for joint in joints if isinstance(joint, dict) and
                          joint.get('name') in ['left_wrist', 'right_wrist'] and joint.get('in_frame') is True and
                          finite(joint.get('score')) and joint['score'] >= pose['display_threshold']]
                known += len(wrists) == 2
            if known < total * 0.6:
                observations.append(f'{LABELS[side]}双腕都可见的采样画面较少')
    return '；'.join(observations) + '，这些画面需要先核对。' if observations else ''


def nearby_windows(pose, sides, times):
    signals = wrist_signals(pose)
    candidates = [dict(signal, player=side, label=LABELS[side])
                  for side in sides for signal in signals.get(side, [])]
    if times:
        candidates.sort(key=lambda signal: (min(abs(signal['peak_time'] - t) for t in times), signal['start']))
    else:
        candidates.sort(key=lambda signal: (signal['start'], signal['player']))
    windows = []
    for signal in candidates:
        # Overlapping left/right wrist runs in one action share a viewing window.
        start = round(max(pose['start'], signal['start'] - 0.6), 4)
        end = round(min(pose['end'], signal['end'] + 0.7), 4)
        if any(window['player'] == signal['player'] and start <= window['end'] and end >= window['start']
               for window in windows):
            continue
        windows.append({'player': signal['player'], 'label': signal['label'], 'start': start, 'end': end,
                        'signal_start': signal['start'], 'signal_end': signal['end'],
                        'peak_time': signal['peak_time'], 'joint_side': signal['joint_side'],
                        'kind': 'visible_wrist_above_shoulder', 'known_frames': signal['known_frames'],
                        'interpretation': '图像中手腕连续高于同侧肩部；不是触球或动作好坏结论'})
        if len(windows) == 3:
            break
    return windows


def ordinary_window(pose, sides, times):
    if not sides or not times:
        return []
    center = times[0]
    return [{'player': side, 'label': LABELS[side],
             'start': round(max(pose['start'], center - 1), 4),
             'end': round(min(pose['end'], center + 1.5), 4),
             'kind': 'user_provided_time', 'interpretation': '用户指定时间附近；待人工核对动作阶段'}
            for side in sides]


def window_sentence(windows):
    return '；'.join(f"{item['label']}：原视频 {time_text(item['start'])}–{time_text(item['end'])} 秒"
                    for item in windows)


def reply_for(document, pose, choices, text, selection_changed):
    sides = selected_sides(document['selected_player'], choices)
    if not choices:
        return ('这段画面还没有得到可用的球员姿态。请换一段球员清楚可见的视频，'
                '再一起定位你关心的动作。', [])
    if not sides:
        return ('先选一下你想看的候选球员。选好后，可以告诉我想改进引拍、触球、移动还是回位，'
                '也可以直接说想看杀球、反手等技术。', [])
    if not text and selection_changed:
        labels = '和'.join(LABELS[side] for side in sides)
        return (f'好，先看{labels}。最想改进哪方面？比如杀球、反手、移动或回位。', [])
    times = requested_times(text)
    if any(time < pose['start'] or time > pose['end'] for time in times):
        return (f"这次处理的是原视频 {time_text(pose['start'])}–{time_text(pose['end'])} 秒。"
                '你提到的时间不全在这段内，请先给出这段中的时间，或换一段包含该动作的视频。', [])
    focus = document['focus']
    windows = nearby_windows(pose, sides, times) if focus in ['preparation', 'backswing', 'contact'] else []
    if not windows:
        windows = ordinary_window(pose, sides, times)
    location = window_sentence(windows)
    note = coverage_note(pose, sides, arms=focus in ['backswing', 'contact'])
    if any(word in text for word in ['职业', '国际', '林丹', '安赛龙', '桃田', '示范', '正确动作']):
        reply = ('目前还没有经过核对的职业示范片段，不能把某段职业录像认定为这次动作的正确示范。'
                 + (f'先回看 {location}。' if windows else '') + '你希望改进哪项技术、达到什么效果？')
    elif not windows:
        if focus == 'preparation':
            reply = '你觉得主要是启动慢，还是到了位置来不及准备下一拍？当前还没有能定位这次移动的画面依据。'
        elif focus == 'recovery':
            reply = '你更关心回位慢，还是下一拍接不上？当前还没有自动定位击球后的落地与回位阶段。'
        elif focus == 'balance':
            reply = ('二维姿态不能可靠确定重心或受力，也不能仅凭角度判断平衡好坏。'
                     '你主要感觉站不稳、身体后仰，还是落地不舒服？')
        elif focus in ['backswing', 'contact']:
            reply = '当前没有足够的画面特征来定位你说的这次动作。你是在打杀球、高远球、吊球，还是另一项技术时遇到问题？'
        else:
            reply = '你最想解决哪项技术的问题？可以直接说“杀球不好”或“反手问题”，我会在视频里找对应的候选片段。'
    elif focus == 'contact':
        reply = (f'可以先回看 {location}，找出你说的那次击球。'
                 '目前没有定位球与球拍接触的画面，不能据此判断触球点高低或触球时机。'
                 '这次你想改善的是球速、落点，还是击球时的舒适度？')
    elif focus == 'backswing':
        signal_windows = [window for window in windows if window['kind'] == 'visible_wrist_above_shoulder']
        if signal_windows:
            first = signal_windows[0]
            wrist = '左腕' if first['joint_side'] == 'left' else '右腕'
            reply = (f"{first['label']}在原视频 {time_text(first['signal_start'])}–"
                     f"{time_text(first['signal_end'])} 秒的采样画面中，{wrist}连续高于同侧肩部。"
                     f'可以先回看 {location}，观察抬臂怎样衔接挥拍。'
                     '这还不能确定触球时刻或动作好坏。你当时觉得准备太晚、发不上力，还是落点不稳？')
        else:
            reply = (f'可以先回看 {location}。这段还没有足够的连续腕部特征来定位一次抬臂动作，'
                     '需要先核对具体击球。你当时想打什么球？')
    elif focus == 'preparation':
        reply = (f'先回看 {location}，看球员开始移动前的站位，以及到位后是否有时间准备下一次击球。'
                 '当前姿态没有定位对手触球时刻，不能判断启动早晚。你觉得主要是启动慢，还是到了位置来不及引拍？')
    elif focus == 'recovery':
        reply = (f'先回看 {location}，观察击球后落地、第一步和下一次准备的衔接。'
                 '这段没有自动定位触球和落地阶段，需要按原画面核对。你更关心回位慢，还是下一拍接不上？')
    elif focus == 'balance':
        reply = (f'可以先回看 {location}，观察躯干、髋部和支撑脚在画面中的变化。'
                 '二维姿态不能可靠确定重心或受力，也不能仅凭角度判断身体平衡好坏。'
                 '你当时有站不稳、身体后仰，或落地不舒服的感觉吗？')
    else:
        reply = (f'我可以先帮你定位回看片段：{location}。'
                 '现在有的是可见关节的姿态，还需要结合具体击球和你的目标，才能讨论怎么改。'
                 '你最想解决什么问题？')
    return (reply + ('\n\n' + note if note else ''), windows)


def respond(job, pose, index_item, body):
    validate_result(job, pose, index_item)
    if not isinstance(body, dict):
        raise ValueError('沟通请求格式错误')
    text = body.get('message', '')
    if not isinstance(text, str) or len(text) > 4000:
        raise ValueError('消息需为不超过 4000 字的文字')
    text = text.strip()
    has_selection = 'player' in body or 'selected_player' in body
    if 'player' in body and 'selected_player' in body and body['player'] != body['selected_player']:
        raise ValueError('球员选择字段不一致')
    if not text and not has_selection:
        raise ValueError('请选择球员或输入想讨论的问题')
    selected = body.get('player', body.get('selected_player'))
    if has_selection and selected not in [None, 'near', 'far', 'both']:
        raise ValueError('球员选择无效')
    with LOCK:
        choices = choices_for(job, pose)
        available = {choice['id'] for choice in choices}
        if has_selection and ((selected in SIDES and selected not in available) or
                              (selected == 'both' and len(available) < 2)):
            raise ValueError('该球员没有可回看的姿态，请选择已有候选')
        document = read_document(job, pose, choices)
        if len(document['messages']) + 2 > MAX_MESSAGES:
            raise ValueError('这段视频的沟通记录已满，请开始另一段视频的讨论')
        selection_changed = has_selection and selected != document['selected_player']
        if has_selection:
            document['selected_player'] = selected
        if text:
            clear_search_context(document)
            document['question'] = text
            document['focus'] = focus_for(text)
        content = text
        if not content:
            content = ('先看' + '和'.join(LABELS[side] for side in selected_sides(selected, choices))
                       if selected else '重新选择球员')
        document['messages'].append(message('user', content))
        query = parse_technical_query(text) if text else document.get('technical_query')
        _, styles = technical_components(text) if text else ([], [])
        if text and len(styles) > 1:
            document['messages'].append(message('assistant', '你同时提到了正手和反手。这次先看哪一种？选好后我会在视频里找对应片段。'))
            document['evidence_windows'] = []
        elif valid_query(query) and choices:
            if not text:
                clear_search_context(document)
            start_document_search(document, job, pose, choices, query)
        else:
            reply, windows = reply_for(document, dict(pose, end=effective_end(pose)), choices, text, selection_changed)
            document['messages'].append(message('assistant', reply, windows))
            document['evidence_windows'] = windows
        document['updated_at'] = now()
        atomic_json(path_for(job), document)
        return public_document(document, job, pose, index_item, choices)
