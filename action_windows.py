"""Recall visible arm-action windows from existing COCO17 pose samples.

These are proposals for a later video classifier, not stroke classifications.
Neither an elevated wrist nor an image-plane speed peak identifies a smash,
backhand, shuttle contact, or a technical error. Missing poses are never filled.
The input's source_time values remain nominal sampled video playback times.
"""

from bisect import bisect_left, bisect_right
import math
import statistics
import uuid


MAX_WINDOWS = 200
_SIDES = ('near', 'far')
_HANDS = {'left': (5, 9), 'right': (6, 10)}
_TORSO = (5, 6, 11, 12)
_TIME_NOTE = ('Times identify nominal sampled video playback, not precise '
              'original frame PTS or shuttle-contact times.')
_PROPOSAL_NOTE = ('Visible arm-motion or wrist-elevation proposal only; '
                  'stroke type, racket hand, contact and technical quality are unknown.')


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _joint(keypoints, index, threshold):
    if not isinstance(keypoints, list) or index >= len(keypoints):
        return None
    point = keypoints[index]
    if not isinstance(point, dict) or point.get('in_frame') is not True:
        return None
    if point.get('finite_model_output') is False:
        return None
    x, y, score = (_number(point.get(key)) for key in ('x', 'y', 'score'))
    if x is None or y is None or score is None or score < threshold:
        return None
    return (x, y, score)


def _frame_record(frame, player, threshold):
    """Return only measured joints; missing torso/wrists stay missing."""
    record = {'time': frame['source_time'], 'sample_index': frame.get('sample_index'),
              'assigned': isinstance(player, dict), 'body': None,
              'wrists': {}, 'scores': [], 'quality_flags': []}
    if not isinstance(player, dict):
        return record
    record['track_id'] = player.get('track_id')
    record['scout_track_id'] = player.get('scout_track_id')
    flags = player.get('quality_flags')
    record['quality_flags'] = list(flags) if isinstance(flags, list) else []
    keypoints = player.get('keypoints')
    torso = [_joint(keypoints, index, threshold) for index in _TORSO]
    if any(point is None for point in torso):
        return record
    left_shoulder, right_shoulder, left_hip, right_hip = torso
    shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2,
                    (left_shoulder[1] + right_shoulder[1]) / 2)
    hip_mid = ((left_hip[0] + right_hip[0]) / 2,
               (left_hip[1] + right_hip[1]) / 2)
    scale = _distance(shoulder_mid, hip_mid)
    if scale < 2:
        return record
    record['body'] = {'torso_px': scale, 'shoulder_mid_px': shoulder_mid}
    record['scores'].extend(point[2] for point in torso)
    for hand, (shoulder_index, wrist_index) in _HANDS.items():
        wrist = _joint(keypoints, wrist_index, threshold)
        shoulder = _joint(keypoints, shoulder_index, threshold)
        if wrist is None or shoulder is None:
            continue
        record['wrists'][hand] = {
            'relative_xy': ((wrist[0] - shoulder_mid[0]) / scale,
                            (wrist[1] - shoulder_mid[1]) / scale),
            'height': (shoulder[1] - wrist[1]) / scale,
            'score': wrist[2], 'image_px': [wrist[0], wrist[1]],
        }
        record['scores'].append(wrist[2])
    return record


def _continuous_runs(records, hand, interval):
    """A gap, missing measurement or known track change breaks derivatives."""
    runs, run = [], []
    for record in records:
        if record['body'] is None or hand not in record['wrists']:
            if run:
                runs.append(run)
                run = []
            continue
        if run:
            previous = run[-1]
            delta = record['time'] - previous['time']
            before_index, after_index = previous['sample_index'], record['sample_index']
            index_gap = (isinstance(before_index, int) and isinstance(after_index, int)
                         and after_index != before_index + 1)
            track_change = any(previous.get(key) != record.get(key)
                               for key in ('track_id', 'scout_track_id'))
            ratio = record['body']['torso_px'] / previous['body']['torso_px']
            if delta <= 0 or delta > interval * 1.75 or index_gap or track_change or not .5 <= ratio <= 2:
                runs.append(run)
                run = []
        run.append(record)
    if run:
        runs.append(run)
    return runs


def _quantile(values, proportion):
    values = sorted(values)
    if not values:
        return 0.0
    return values[min(len(values) - 1, int((len(values) - 1) * proportion))]


def _proposal(side, hand, start, end, peak_time, trigger, strength, details):
    return {'player': side, 'start': start, 'end': end, 'peak_time': peak_time,
            'kind': 'stroke_candidate', 'proposal_score': strength,
            'evidence': [{'trigger': trigger, 'joint_side': hand, **details}]}


def _run_proposals(run, side, hand):
    if len(run) < 2:
        return []
    proposals, speeds = [], []
    for previous, current in zip(run, run[1:]):
        displacement = _distance(previous['wrists'][hand]['relative_xy'],
                                 current['wrists'][hand]['relative_xy'])
        delta = current['time'] - previous['time']
        speeds.append({'time': current['time'], 'value': displacement / delta,
                       'previous_time': previous['time'], 'displacement': displacement})
    baseline = _quantile([sample['value'] for sample in speeds], .5)
    reference = max(.5, _quantile([sample['value'] for sample in speeds], .9))
    # A deliberately permissive floor recalls slow motion too. Peaks are local
    # arm-motion evidence and may also be noise, preparation or recovery.
    speed_times = [sample['time'] for sample in speeds]
    for sample in speeds:
        if sample['value'] < .25:
            continue
        nearby = speeds[bisect_left(speed_times, sample['time'] - .25):
                        bisect_right(speed_times, sample['time'] + .25)]
        peak = max(nearby, key=lambda value: (value['value'], -value['time']))
        if peak is not sample:
            continue
        strength = min(.98, .35 + .5 * sample['value'] / (sample['value'] + reference))
        proposals.append(_proposal(
            side, hand, sample['time'] - .7, sample['time'] + 1.0,
            sample['time'], 'relative_wrist_motion_peak', strength,
            {'speed_torso_units_per_second': round(sample['value'], 4),
             'adjacent_displacement_torso_units': round(sample['displacement'], 4),
             'measured_pair_times': [round(sample['previous_time'], 4), round(sample['time'], 4)],
             'run_median_speed_torso_units_per_second': round(baseline, 4),
             'normalization': 'wrist minus shoulder midpoint, divided by measured torso length',
             'interpretation': 'Image-plane relative motion; not physical racket speed or contact.'}))

    # Raised-wrist runs supplement motion peaks, including slower preparation.
    # Split prolonged elevation so the whole recorded sequence remains covered.
    elevated = []

    def finish_elevated():
        if len(elevated) < 2:
            return
        chunks, chunk = [], []
        for record in elevated:
            if chunk and record['time'] - chunk[0]['time'] > 2.0:
                chunks.append(chunk)
                chunk = []
            chunk.append(record)
        if chunk:
            chunks.append(chunk)
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            highest = max(chunk, key=lambda record: record['wrists'][hand]['height'])
            height = highest['wrists'][hand]['height']
            proposals.append(_proposal(
                side, hand, chunk[0]['time'] - .4, chunk[-1]['time'] + .6,
                highest['time'], 'visible_wrist_elevation_run', .38 + .12 * min(1.0, height),
                {'observed_run_start': round(chunk[0]['time'], 4),
                 'observed_run_end': round(chunk[-1]['time'], 4),
                 'observed_samples': len(chunk), 'max_height_torso_units': round(height, 4),
                 'interpretation': 'Wrist visible above shoulder; may be posture, preparation or other action.'}))

    for record in run:
        if record['wrists'][hand]['height'] >= .2:
            elevated.append(record)
        else:
            finish_elevated()
            elevated = []
    finish_elevated()

    # Prominent vertical excursions can recall a gradual low-arm action even
    # when it never meets the elevation floor. This also does not classify it.
    run_times = [record['time'] for record in run]
    for index in range(1, len(run) - 1):
        record = run[index]
        height = record['wrists'][hand]['height']
        neighbors = run[bisect_left(run_times, record['time'] - .65):
                        bisect_right(run_times, record['time'] + .65)]
        maximum = max(neighbors, key=lambda other: (other['wrists'][hand]['height'], -other['time']))
        if maximum is not record:
            continue
        before = [other['wrists'][hand]['height'] for other in neighbors if other['time'] < record['time']]
        after = [other['wrists'][hand]['height'] for other in neighbors if other['time'] > record['time']]
        if not before or not after:
            continue
        excursion = min(height - min(before), height - min(after))
        if excursion < .12:
            continue
        proposals.append(_proposal(
            side, hand, record['time'] - .75, record['time'] + .9,
            record['time'], 'measured_wrist_height_excursion', .40 + .1 * min(1, excursion),
            {'bilateral_temporal_prominence_torso_units': round(excursion, 4),
             'peak_height_torso_units': round(height, 4),
             'interpretation': 'Observed wrist rise and fall in a contiguous run, not a stroke label.'}))
    return proposals


def _deduplicate(proposals):
    """One proposal per nearby event; keep both hands' measured evidence."""
    output, peak_buckets = [], {}
    for candidate in sorted(proposals, key=lambda item: (-item['proposal_score'], item['peak_time'])):
        duplicate = None
        bucket = int(candidate['peak_time'] / .5)
        nearby = [selected for index in range(bucket - 7, bucket + 8)
                  for selected in peak_buckets.get(index, [])]
        for selected in nearby:
            if candidate['player'] != selected['player']:
                continue
            intersection = max(0, min(candidate['end'], selected['end']) - max(candidate['start'], selected['start']))
            union = max(candidate['end'], selected['end']) - min(candidate['start'], selected['start'])
            if abs(candidate['peak_time'] - selected['peak_time']) <= .6 or (union > 0 and intersection / union > .6):
                duplicate = selected
                break
        if duplicate is None:
            output.append(candidate)
            peak_buckets.setdefault(bucket, []).append(candidate)
            continue
        for evidence in candidate['evidence']:
            if evidence not in duplicate['evidence']:
                duplicate['evidence'].append(evidence)
        # Do not chain neighboring actions into an increasingly long clip.
        if max(candidate['end'], duplicate['end']) - min(candidate['start'], duplicate['start']) <= 3.2:
            duplicate['start'] = min(candidate['start'], duplicate['start'])
            duplicate['end'] = max(candidate['end'], duplicate['end'])
    return output


def _timeline_limit(candidates, limit):
    """If necessary, allocate a bounded budget across the entire timeline."""
    if len(candidates) <= limit:
        return candidates
    first = min(item['peak_time'] for item in candidates)
    last = max(item['peak_time'] for item in candidates)
    bucket_count = max(1, limit // 2)
    span = max(last - first, .001)
    buckets = [[] for _ in range(bucket_count)]
    for candidate in candidates:
        index = min(bucket_count - 1, int((candidate['peak_time'] - first) / span * bucket_count))
        buckets[index].append(candidate)
    for bucket in buckets:
        bucket.sort(key=lambda item: (-item['proposal_score'], item['peak_time'], item['player']))
    chosen, depth = [], 0
    while len(chosen) < limit:
        added = False
        for bucket in buckets:
            if depth < len(bucket):
                chosen.append(bucket[depth])
                added = True
                if len(chosen) == limit:
                    break
        if not added:
            break
        depth += 1
    return chosen


def _coverage_limit(candidates, limit):
    """Retain temporal coverage for both selected player slots when bounded."""
    if len(candidates) <= limit:
        return candidates
    groups = {side: [item for item in candidates if item['player'] == side] for side in _SIDES}
    groups = {side: items for side, items in groups.items() if items}
    if len(groups) == 1:
        return _timeline_limit(candidates, limit)
    quota = limit // len(groups)
    chosen = [item for group in groups.values() for item in _timeline_limit(group, quota)]
    if len(chosen) < limit:
        selected_ids = {id(item) for item in chosen}
        remaining = [item for item in candidates if id(item) not in selected_ids]
        chosen.extend(_timeline_limit(remaining, limit - len(chosen)))
    return chosen


def _quality(records, start, end):
    observed = [record for record in records if start <= record['time'] < end]
    count = len(observed)
    assigned = sum(record['assigned'] for record in observed)
    torso = sum(record['body'] is not None for record in observed)
    wrists = {hand: sum(hand in record['wrists'] for record in observed) for hand in _HANDS}
    scores = [score for record in observed for score in record['scores']]
    flags = sorted({flag for record in observed for flag in record['quality_flags'] if isinstance(flag, str)})
    return {'observed_sample_count': count, 'assigned_sample_count': assigned,
            'valid_torso_sample_count': torso, 'valid_wrist_sample_count': wrists,
            'missing_assignment_sample_count': count - assigned,
            'torso_coverage': round(torso / count, 4) if count else 0.0,
            'wrist_coverage': {hand: round(value / count, 4) if count else 0.0 for hand, value in wrists.items()},
            'mean_keypoint_score': round(statistics.mean(scores), 4) if scores else None,
            'score_note': 'Model scores are uncalibrated and may exceed one; coverage is not accuracy.',
            'identity_status': 'not_frame_verified', 'joint_status': 'not_frame_verified',
            'quality_flags': flags, 'interpolation': None}


def build_action_windows(pose, sides=None):
    """Return bounded, temporally distributed stroke_candidate windows.

    ``sides`` uses existing near/far storage slots, which in automatic analysis
    may represent candidate A/B rather than calibrated court sides. The input
    is a decoded pose-result dictionary. An empty evidence stream returns [];
    it never substitutes the start of the video for an action.
    """
    if not isinstance(pose, dict) or not isinstance(pose.get('frames'), list):
        return []
    if isinstance(sides, str):
        sides = [sides]
    selected_sides = [side for side in _SIDES if sides is None or side in sides]
    frames = [frame for frame in pose['frames'] if isinstance(frame, dict)
              and _number(frame.get('source_time')) is not None]
    frames.sort(key=lambda frame: frame['source_time'])
    if len(frames) < 2 or not selected_sides:
        return []
    differences = [current['source_time'] - previous['source_time'] for previous, current in zip(frames, frames[1:])
                   if current['source_time'] > previous['source_time']]
    fps = _number(pose.get('sample_fps'))
    interval = 1 / fps if fps is not None and fps > 0 else (statistics.median(differences) if differences else None)
    if interval is None or interval <= 0:
        return []
    threshold = _number(pose.get('display_threshold'))
    threshold = threshold if threshold is not None and threshold >= 0 else .5
    start = _number(pose.get('start'))
    start = max(frames[0]['source_time'], start) if start is not None else frames[0]['source_time']
    end = _number(pose.get('processed_end'))
    if end is None:
        end = _number(pose.get('end'))
    observed_end = frames[-1]['source_time'] + interval
    end = min(observed_end, end) if end is not None else observed_end
    frames = [frame for frame in frames if start <= frame['source_time'] < end]
    records_by_side, proposals = {}, []
    for side in selected_sides:
        records = [_frame_record(frame, frame['players'].get(side)
                                 if isinstance(frame.get('players'), dict) else None, threshold)
                   for frame in frames]
        records_by_side[side] = records
        side_proposals = []
        for hand in _HANDS:
            for run in _continuous_runs(records, hand, interval):
                side_proposals.extend(_run_proposals(run, side, hand))
        for proposal in side_proposals:
            proposal['start'] = max(start, proposal['start'])
            proposal['end'] = min(end, proposal['end'])
        proposals.extend(_deduplicate([item for item in side_proposals if item['end'] > item['start']]))
    proposals = _coverage_limit(proposals, MAX_WINDOWS)
    proposals.sort(key=lambda item: (item['peak_time'], item['player']))
    result = []
    for proposal in proposals:
        for field in ('start', 'end', 'peak_time', 'proposal_score'):
            proposal[field] = round(proposal[field], 4)
        identifier = ':'.join([str(pose.get('id', 'pose')), proposal['player'],
                               str(proposal['start']), str(proposal['end']), str(proposal['peak_time'])])
        proposal['id'] = uuid.uuid5(uuid.NAMESPACE_URL, 'badminton-action-proposal:' + identifier).hex
        proposal['quality'] = _quality(records_by_side[proposal['player']], proposal['start'], proposal['end'])
        proposal['proposal_score_note'] = 'Uncalibrated prioritization score; not stroke classification probability.'
        proposal['time_basis'] = pose.get('time_basis', 'nominal_resampled_video_playback_seconds')
        proposal['timestamp_note'] = _TIME_NOTE
        proposal['interpretation'] = _PROPOSAL_NOTE
        proposal['classification'] = {'status': 'unclassified', 'stroke_type': None, 'racket_hand': None}
        result.append(proposal)
    return result
