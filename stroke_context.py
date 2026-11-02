"""Separate contact review frames from proposed single-stroke model support.

Confirmed same-player contacts provide a midpoint guard, never true adjacent
opponent boundaries. Legacy/unverified rally context remains explicitly marked.
Automatic windows retain their prior numerical behavior.
"""
import numpy as np
from hit_refinement import effective_event

VERSION = 'stroke-flight-window-v3-confirmed-same-player-guard'
GUARD_VERSION = 'same-player-midpoint-guard-v1'


def guard_confirmed_single_hit(dense, translated_window, target_player, labels,
                               target_event_id=None):
    """Reuse the fixed 70-example guard; no stroke-dependent window selection.

    Eligibility intentionally matches audit_single_hit_guard_all70.py exactly:
    confirmed, not wrong-target, not explicitly between rallies, and not
    unknown/no-stroke. Missing or unknown rally context can suggest a bound but
    is recorded as unverified. Exact-frame duplicates are grouped; contacts
    within +/-0.15 seconds of this hit never serve as previous/next anchors.
    """
    times = np.asarray([f['source_time'] for f in dense['frames']], float)
    if not len(times):
        raise ValueError('No decoded frames for contact support')
    # Reapplying after labels change starts from the retained translated
    # support, not a previously shortened guard. This makes identical calls
    # idempotent while allowing removed labels to restore the original support.
    original = translated_window.get('translated_model_support', translated_window)
    center = int(np.argmin(abs(times-translated_window['peak_time'])))
    peak = float(times[center])
    a, b = int(original['start_frame']), int(original['end_frame'])
    if not 0 <= a <= center <= b < len(times):
        raise ValueError('Translated model support does not contain contact')
    labels = labels.get('labels', labels) if isinstance(labels, dict) else labels
    groups = {}
    for key, annotation in labels.items():
        if (not annotation.get('contact_confirmed') or annotation.get('wrong_target')
                or annotation.get('play_context') == 'between_rallies'
                or annotation.get('stroke_type') in ['unknown', 'no_stroke']
                or annotation.get('player') != target_player):
            continue
        time = annotation.get('peak_time')
        if not isinstance(time, (int, float)) or not np.isfinite(time):
            continue
        index = int(np.argmin(abs(times-time)))
        groups.setdefault(index, []).append({
            'event_id': annotation.get('event_id', key), 'human_peak_time': float(time),
            'play_context': annotation.get('play_context'),
            'context_verified': annotation.get('play_context') == 'rally'})
    previous = max((i for i in groups if times[i] < peak-.15), default=None)
    following = min((i for i in groups if times[i] > peak+.15), default=None)

    def anchor(index):
        if index is None:
            return None
        members = sorted(groups[index], key=lambda r: str(r['event_id']))
        contexts = [v['play_context'] for v in members]
        return {'frame_index': index, 'time': float(times[index]),
                'player': target_player, 'event_ids': [v['event_id'] for v in members],
                'play_context': contexts[0] if all(v == contexts[0] for v in contexts) else 'mixed',
                'context_verified': all(v['context_verified'] for v in members),
                'annotations': members}

    lo, hi = a, b
    if previous is not None:
        lo = max(lo, int(np.searchsorted(times, (times[previous]+peak)/2, side='left')))
    if following is not None:
        hi = min(hi, int(np.searchsorted(times, (times[following]+peak)/2+.25,
                                       side='right')-1), following-1)
    lo, hi = min(lo, center), max(hi, center)
    before, after = anchor(previous), anchor(following)
    selected = [v for v in [before, after] if v is not None]
    unverified = sum(not v['context_verified'] for v in selected)
    support_keys = ['start_frame', 'end_frame', 'frame_index', 'start', 'end',
                    'last_frame_time', 'peak_time', 'duration', 'basis',
                    'window_repair_version', 'original_center_frame',
                    'translation_frames', 'recording_edge_clipped']
    support = {k: original[k] for k in support_keys if k in original}
    dt = 1/float(dense['sample_fps'])
    return {**translated_window, 'version': VERSION, 'player': target_player,
            'start_frame': lo, 'end_frame': hi, 'frame_index': center,
            'start': float(times[lo]), 'end': float(times[hi]+dt),
            'last_frame_time': float(times[hi]), 'peak_time': peak,
            'duration': float(times[hi]+dt-times[lo]),
            'basis': 'human_translated_support_with_same_player_midpoint_guard',
            'translated_model_support': support, 'boundary_confirmed': False,
            'opponent_contacts_used_for_boundaries': False,
            'context_unverified_anchor_count': unverified,
            'has_context_unverified_anchors': bool(unverified),
            'single_hit_guard': {
                'version': GUARD_VERSION, 'target_event_id': target_event_id,
                'applied': lo != a or hi != b,
                'previous_same_player_anchor': before, 'next_same_player_anchor': after,
                'same_hit_exclusion_seconds': .15, 'following_midpoint_padding_seconds': .25,
                'same_hit_anchor_exclusions': [anchor(i) for i in sorted(groups)
                                              if not times[i] < peak-.15 and not times[i] > peak+.15],
                'eligible_unique_frame_count': len(groups),
                'duplicate_same_frame_annotation_count': sum(len(v)-1 for v in groups.values()),
                'rule': 'Preserve translated support; start>=previous same-player midpoint; end<=next same-player midpoint+.25s and before that hit. No stroke-specific parameters.',
                'rally_context_policy': 'Match the fixed audit eligibility: missing/unknown context permitted but unverified; explicitly between_rallies excluded.',
            },
            'note': 'Confirmed contacts propose same-player midpoint limits; sparse opponent labels and legacy rally context do not establish true neighboring hits or verified stroke boundaries.'}


def window(dense, event, events, labels=None, reviewed=True):
    labels = labels or {}
    annotation = labels.get(event['id'], {}) if reviewed else {}
    target = {**event, **annotation} if annotation and not annotation.get('wrong_target') else dict(event)
    target, suggestion = effective_event(dense, target, annotation)
    times = np.asarray([f['source_time'] for f in dense['frames']], float)
    center = int(np.argmin(abs(times-target['peak_time'])))
    peak = float(times[center])
    anchors = []
    if reviewed:
        for item in events:
            a = labels.get(item['id'], {})
            if (a.get('contact_confirmed') and not a.get('wrong_target')
                    and a.get('play_context') != 'between_rallies'
                    and a.get('stroke_type') not in ['unknown', 'no_stroke']):
                anchors.append({'event_id': item['id'], 'time': a['peak_time'], 'player': a['player']})
    opposite = [a for a in anchors if a['player'] != target['player'] and abs(a['time']-peak) >= .15]
    before = max((a for a in opposite if a['time'] < peak), key=lambda a: a['time'], default=None)
    after = min((a for a in opposite if a['time'] > peak), key=lambda a: a['time'], default=None)
    conflicts = [a['event_id'] for a in anchors if a['player'] != target['player'] and abs(a['time']-peak) < .15]
    if annotation.get('contact_confirmed'):
        # Sparse reviewed neighbours need not be adjacent physical hits. In the
        # paired diagnostic their clipping reduced joint agreement. Preserve
        # the automatic full support and translate it to the confirmed contact.
        from stroke_window_repairs import translate_contact_window
        original = window(dense, event, events, {}, reviewed=False)
        translated = translate_contact_window(dense, original, annotation['peak_time'])
        guarded = guard_confirmed_single_hit(dense, translated, target['player'], labels,
                                             target_event_id=event['id'])
        return {**guarded, 'event_id': event['id'], 'version': VERSION,
                'player': target['player'], 'contact_confirmed': True,
                'previous_reviewed_contact': before, 'next_reviewed_contact': after,
                'near_simultaneous_opponent_contacts': conflicts,
                'center_suggestion': suggestion}
    else:
        # Reuse the input on which pretrained BST already has a valid baseline.
        lo = max(0., peak-1.5, event['start'])
        hi = min(float(dense['end']), peak+1.75, event['end'])
        if hi-lo < 1.:
            lo, hi = max(0., peak-1.), min(float(dense['end']), peak+1.)
        basis = 'original_specialist_context_capped' if hi-lo >= 1. else 'short_recording_edge'
    lo, hi = min(lo, peak), max(hi, peak)
    start = min(center, int(np.searchsorted(times, lo, side='left')))
    end = max(center, min(len(times)-1, int(np.searchsorted(times, hi, side='right')-1)))
    dt = float(np.median(np.diff(times))) if len(times)>1 else 1/dense['sample_fps']
    return {'event_id': event['id'], 'version': VERSION, 'player': target['player'],
            'start_frame': start, 'end_frame': end, 'frame_index': center,
            'start': float(times[start]), 'end': min(float(dense['end']), float(times[end]+dt)),
            'last_frame_time': float(times[end]), 'peak_time': peak,
            'duration': float(times[end]+dt-times[start]), 'basis': basis,
            'contact_confirmed': bool(annotation.get('contact_confirmed')),
            'previous_reviewed_contact': before, 'next_reviewed_contact': after,
            'near_simultaneous_opponent_contacts': conflicts,
            'boundary_confirmed': False, 'center_suggestion': suggestion,
            'note': 'Model support for preparation, stroke and flight; proposed boundaries, separate from 21-frame contact review.'}


def metadata(dense, events, doc):
    return {'version': VERSION, 'source_sha256': dense['source_sha256'], 'dataset_revision': doc['revision'],
            'windows': {e['id']: window(dense, e, events, doc['labels']) for e in events},
            'review_window_frames': 21, 'maximum_before_seconds': 1.5, 'maximum_after_seconds': 1.75}


def prepare(job):
    from analysis_jobs import atomic_json
    from hit_segments import dense_data
    from stroke_review import load_source, read_labels, folder_for
    report, events, _ = load_source(job)
    doc = read_labels(job, report['source_sha256'])
    value = metadata(dense_data(job), events, doc)
    atomic_json(folder_for(job)/'stroke_context.json', value)
    return value
