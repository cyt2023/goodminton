"""Conservative, untrained candidate selection for local video analysis.

This module ranks human detection tracks. It has no court geometry, identity
classifier, athlete label, or action-quality classifier. Internal near/far keys
are compatibility slots, displayed as candidate A/B in automatic analyses.
"""
import math

import cv2
import numpy as np


def appearance(image, bbox):
    """A small HSV histogram of the torso crop, not a learned identity feature."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = np.asarray(bbox, dtype=float)
    bw, bh = x2 - x1, y2 - y1
    left, right = int(max(0, x1 + .2 * bw)), int(min(width, x2 - .2 * bw))
    top, bottom = int(max(0, y1 + .2 * bh)), int(min(height, y1 + .7 * bh))
    if right <= left or bottom <= top:
        return None
    crop = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    # Brightness helps distinguish dark clothing from pale or saturated shirts.
    histogram = cv2.calcHist([crop], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).reshape(-1)
    total = float(histogram.sum())
    return np.sqrt(histogram / total) if total > 0 else None


def appearance_distance(left, right):
    if left is None or right is None:
        return .5
    return float(np.clip(1 - np.dot(left, right), 0, 1))


def foot(box):
    return np.array([(box[0] + box[2]) / 2, box[3]], dtype=float)


def valid_boxes(boxes, width, height):
    result = []
    for index, box in enumerate(boxes):
        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            continue
        normalized = np.asarray(box, dtype=float) / np.array([width, height, width, height])
        point = foot(normalized)
        extent = normalized[3] - normalized[1]
        if .025 <= extent <= 1.1 and -.05 <= point[0] <= 1.05 and 0 <= point[1] <= 1.1:
            result.append((index, normalized))
    return result


class ScoutTracks:
    """Greedy temporal association with position, scale and torso appearance."""
    def __init__(self, interval):
        self.interval = interval
        self.tracks = []
        self.sample_count = 0
        self.ambiguous_associations = 0

    def observe(self, image, boxes, timestamp):
        height, width = image.shape[:2]
        candidates = [(index, box, appearance(image, boxes[index]))
                      for index, box in valid_boxes(boxes, width, height)]
        active = [track for track in self.tracks
                  if timestamp - track['observations'][-1]['source_time'] <= max(6, self.interval * 3)]
        pairs = []
        for track in active:
            last = track['observations'][-1]
            previous = np.array(last['bbox_normalized'])
            gap = timestamp - last['source_time']
            gate = min(.48, .15 + .13 * gap)
            for candidate_index, (_, box, color) in enumerate(candidates):
                distance = float(np.linalg.norm(foot(box) - foot(previous)))
                scale = abs(math.log((box[3] - box[1]) / (previous[3] - previous[1])))
                color_distance = appearance_distance(track['appearance'], color)
                identity_distance = appearance_distance(track['identity_reference'], color)
                if distance <= gate and scale <= math.log(2.8) and color_distance <= .45 and identity_distance <= .45:
                    cost = distance / gate + 1.5 * color_distance + .8 * identity_distance + .25 * scale
                    pairs.append((cost, track['id'], candidate_index, track))
        pairs.sort(key=lambda value: (value[0], value[1], value[2]))
        used_tracks, used_candidates = set(), set()
        assignments = {}
        for cost, track_id, candidate_index, track in pairs:
            if track_id in used_tracks or candidate_index in used_candidates:
                continue
            competing = [other[0] for other in pairs
                         if other[2] == candidate_index and other[1] != track_id and other[1] not in used_tracks]
            if competing and min(competing) - cost < .08:
                self.ambiguous_associations += 1
                continue
            assignments[candidate_index] = track
            used_tracks.add(track_id)
            used_candidates.add(candidate_index)
        for candidate_index, (detector_index, box, color) in enumerate(candidates):
            track = assignments.get(candidate_index)
            if track is None:
                track = {'id': f'scout_{len(self.tracks) + 1:04d}', 'observations': [],
                         'appearance': color, 'identity_reference': color, 'first_crop': None}
                self.tracks.append(track)
            if track['first_crop'] is None:
                raw = np.asarray(boxes[detector_index], dtype=int)
                x1, y1 = max(0, raw[0]), max(0, raw[1])
                x2, y2 = min(width, raw[2]), min(height, raw[3])
                if x2 > x1 and y2 > y1:
                    crop = image[y1:y2, x1:x2].copy()
                    crop_height = min(240, crop.shape[0])
                    crop_width = max(1, round(crop.shape[1] * crop_height / crop.shape[0]))
                    track['first_crop'] = cv2.resize(crop, (crop_width, crop_height))
            if color is not None:
                vector = color if track['appearance'] is None else .8 * track['appearance'] + .2 * color
                norm = float(np.linalg.norm(vector))
                track['appearance'] = vector / norm if norm else color
            track['observations'].append({'source_time': round(timestamp, 4),
                                          'bbox_normalized': [round(float(value), 6) for value in box],
                                          'detector_index': detector_index})
        self.sample_count += 1

    def select(self, job_id, folder, source_dimensions):
        # Strict appearance gates can fragment one moving athlete. Join only
        # similarly sized, similarly dressed tracks with no simultaneous people.
        # Each appearance reference remains visible in the persisted report.
        groups = []
        for track in sorted(self.tracks, key=lambda item: -len(item['observations'])):
            times = {item['source_time'] for item in track['observations']}
            boxes = np.array([item['bbox_normalized'] for item in track['observations']])
            height = float(np.median(boxes[:, 3] - boxes[:, 1]))
            center = np.median([foot(box) for box in boxes], axis=0)
            match = None
            for group in groups:
                other_boxes = np.array([item['bbox_normalized'] for item in group['observations']])
                other_height = float(np.median(other_boxes[:, 3] - other_boxes[:, 1]))
                other_center = np.median([foot(box) for box in other_boxes], axis=0)
                refs = group['identity_references']
                if (not times.intersection(group['observed_times']) and .7 <= height / other_height <= 1.4
                        and np.linalg.norm(center - other_center) <= .30
                        and min(appearance_distance(ref, track['identity_reference']) for ref in refs) < .30):
                    match = group
                    break
            if match is None:
                group = dict(track, observations=list(track['observations']), observed_times=times,
                             identity_references=[track['identity_reference']], member_track_ids=[track['id']])
                groups.append(group)
            else:
                match['observations'].extend(track['observations'])
                match['observations'].sort(key=lambda item: item['source_time'])
                match['observed_times'].update(times)
                match['identity_references'].append(track['identity_reference'])
                match['member_track_ids'].append(track['id'])
        ranked = []
        minimum_observations = max(2, min(5, math.ceil(self.sample_count * .12)))
        for track in groups:
            observations = track['observations']
            points = np.array([foot(item['bbox_normalized']) for item in observations])
            boxes = np.array([item['bbox_normalized'] for item in observations])
            heights = boxes[:, 3] - boxes[:, 1]
            coverage = len(observations) / max(1, self.sample_count)
            whole_body = (heights >= .8 * float(np.median(heights))) & (heights <= 1.4 * float(np.median(heights)))
            movement_points = points[whole_body] if whole_body.sum() >= 3 else points
            span = float(np.linalg.norm(np.percentile(movement_points, 90, axis=0) - np.percentile(movement_points, 10, axis=0)))
            # Movement lowers the rank of stationary spectators. Camera motion
            # and detector jitter can still mimic movement; this is a heuristic.
            motion = min(1, max(0, span - .015) / .12)
            centrality = max(0, 1 - float(np.median(abs(points[:, 0] - .5))) / .5)
            relative_size = min(1, float(np.median(heights)) / .40)
            score = math.sqrt(coverage) * (.35 + .25 * centrality + .4 * relative_size) * (.2 + .8 * motion)
            eligible = len(observations) >= minimum_observations and coverage >= .10 and span >= .025
            track['metrics'] = {'observed_samples': len(observations), 'scout_sample_coverage': round(coverage, 4),
                                'foot_position_p10_p90_span_normalized': round(span, 4),
                                'median_bbox_height_normalized': round(float(np.median(heights)), 4),
                                'horizontal_centrality': round(centrality, 4),
                                'selection_score': round(score, 6), 'eligible': eligible}
            ranked.append(track)
        ranked.sort(key=lambda track: (-track['metrics']['selection_score'], track['id']))
        eligible = [track for track in ranked if track['metrics']['eligible']]
        chosen = []
        for track in eligible:
            if chosen and track['metrics']['selection_score'] < .30 * chosen[0]['metrics']['selection_score']:
                continue
            # Non-overlapping fragments with a similar shirt and nearby feet
            # cannot establish a second person. Keep the stronger fragment.
            duplicate = False
            for previous in chosen:
                times = {item['source_time'] for item in track['observations']}
                other_times = {item['source_time'] for item in previous['observations']}
                if not times.intersection(other_times) and appearance_distance(track['appearance'], previous['appearance']) < .18:
                    centers = [np.median([foot(item['bbox_normalized']) for item in current['observations']], axis=0)
                               for current in [track, previous]]
                    if np.linalg.norm(centers[0] - centers[1]) < .35:
                        duplicate = True
            if not duplicate:
                chosen.append(track)
            if len(chosen) == 2:
                break
        slots, candidates = {}, []
        scale = np.array([*source_dimensions, *source_dimensions], dtype=float)
        for slot, track in zip(['near', 'far'], chosen):
            slots[slot] = track
            first = track['observations'][0]
            filename = f'player_{slot}.jpg'
            thumbnail = folder / filename
            if track['first_crop'] is not None:
                cv2.imwrite(str(thumbnail), track['first_crop'])
            candidate_rank = next(index for index, other in enumerate(eligible) if other['id'] == track['id'])
            other_scores = [other['metrics']['selection_score'] for other in eligible[candidate_rank + 1:]]
            candidates.append({'slot': slot, 'display_label': '候选球员 A' if slot == 'near' else '候选球员 B',
                               'scout_track_id': track['id'], 'identity_status': 'unconfirmed',
                               'court_side': None, 'selection_metrics': track['metrics'],
                               'rank_margin_vs_next_candidate': round(track['metrics']['selection_score'] - max(other_scores), 6) if other_scores else None,
                               'first_reliable_source_time': first['source_time'],
                               'first_reliable_bbox_original_px': [round(float(value), 2) for value in np.array(first['bbox_normalized']) * scale],
                               'first_reliable_note': '首次检测来自一个满足持续出现和运动阈值的候选轨迹，身份尚未确认。',
                               'thumbnail_file': str(thumbnail.name) if thumbnail.is_file() else None,
                               'thumbnail_url': f'/pose-preview/{job_id}/{filename}' if thumbnail.is_file() else None,
                               'torso_hsv_sqrt_histogram': [round(float(value), 6) for value in track['appearance']] if track['appearance'] is not None else None,
                               'identity_reference_hsv_sqrt_histogram': [round(float(value), 6) for value in track['identity_reference']] if track['identity_reference'] is not None else None,
                               'fragment_track_ids': track['member_track_ids'],
                               'identity_reference_histograms': [[round(float(value), 6) for value in ref] for ref in track['identity_references'] if ref is not None],
                               'scout_observations': track['observations']})
        report = {'method': 'untrained_detection_track_ranking_position_scale_torso_hsv_and_foot_motion',
                  'selection_status': 'candidates_found' if candidates else 'no_stable_candidate',
                  'identity_status': 'needs_user_confirmation', 'court_side_inferred': False,
                  'scout_interval_seconds': self.interval, 'scout_sample_count': self.sample_count,
                  'appearance_version': 'torso_hsv_12x4x4_fixed_identity_reference_v2',
                  'track_count': len(self.tracks), 'ambiguous_scout_associations_skipped': self.ambiguous_associations,
                  'minimum_observations': minimum_observations,
                  'ranking_formula': 'sqrt(coverage) * (0.35 + 0.25*horizontal_centrality + 0.4*relative_size) * (0.2 + 0.8*motion)',
                  'fragment_joining': 'similar torso color, scale and position; no simultaneous observations; untrained heuristic',
                  'candidate_eligibility': {'minimum_scout_coverage': .10, 'minimum_foot_span_normalized': .025,
                                            'second_candidate_minimum_relative_score': .30},
                  'candidates': candidates,
                  'ranked_track_metrics': [{'scout_track_id': track['id'], **track['metrics']} for track in ranked],
                  'note': '按持续出现、画面中心、人体大小和脚部运动自动筛选至多两位候选。没有球场标定或身份模型；邻场球员、相似服装、遮挡、镜头移动和剪辑可能造成错选或身份切换。A/B 不是近端/远端身份，低分或含糊帧保留缺失，需处理后确认。'}
        return AutomaticFollower(slots, self.interval), report


class AutomaticFollower:
    def __init__(self, slots, scout_interval):
        self.slots = slots
        self.scout_interval = scout_interval
        self.previous = {}

    def select_boxes(self, image, boxes, timestamp):
        height, width = image.shape[:2]
        candidates = [(index, box, appearance(image, boxes[index]))
                      for index, box in valid_boxes(boxes, width, height)]
        assignments, used = {}, set()
        reasons = {slot: {'status': 'not_configured', 'candidate_count': 0} for slot in ['near', 'far']}
        for slot, track in self.slots.items():
            reference = min(track['observations'], key=lambda item: abs(item['source_time'] - timestamp))
            reference_gap = abs(reference['source_time'] - timestamp)
            previous = self.previous.get(slot)
            continuous = previous is not None and timestamp - previous['time'] <= .65
            if reference_gap > max(6, self.scout_interval * 3) and not continuous:
                reasons[slot] = {'status': 'outside_automatically_observed_track', 'candidate_count': 0,
                                 'scout_track_id': track['id']}
                continue
            reference_box = np.array(reference['bbox_normalized'])
            anchor = previous['foot'] if continuous else foot(reference_box)
            reference_height = reference_box[3] - reference_box[1]
            potential = []
            for index, box, color in candidates:
                if index in used:
                    continue
                point = foot(box)
                distance = float(np.linalg.norm(point - anchor))
                scale = abs(math.log((box[3] - box[1]) / reference_height))
                color_distance = appearance_distance(track['appearance'], color)
                identity_distance = min(appearance_distance(ref, color) for ref in track.get('identity_references', [track['identity_reference']]))
                gate = min(.18, .06 + .3 * max(0, timestamp - previous['time'])) if continuous else .35
                if distance <= gate and scale <= math.log(2.2) and identity_distance <= .4:
                    cost = distance / gate + 1.6 * color_distance + .8 * identity_distance + .3 * scale
                    potential.append((cost, index, point, distance, color_distance, identity_distance))
            potential.sort(key=lambda item: (item[0], item[1]))
            evidence = {'candidate_count': len(potential), 'scout_track_id': track['id'],
                        'reference_source_time': reference['source_time'],
                        'reference_gap_seconds': round(reference_gap, 4),
                        'anchor_source': 'previous_assigned_frame' if continuous else 'nearest_scout_observation'}
            if not potential:
                reasons[slot] = dict(evidence, status='no_candidate_matching_automatic_track')
                continue
            if len(potential) > 1 and potential[1][0] - potential[0][0] < .12:
                reasons[slot] = dict(evidence, status='ambiguous_automatic_candidates')
                continue
            cost, index, point, distance, color_distance, identity_distance = potential[0]
            assignments[slot] = index
            used.add(index)
            self.previous[slot] = {'foot': point, 'time': timestamp}
            reasons[slot] = dict(evidence, status='selected_by_automatic_candidate_track',
                                 matching_cost=round(cost, 5), foot_distance_normalized=round(distance, 5),
                                 torso_hsv_distance=round(color_distance, 5), fixed_identity_hsv_distance=round(identity_distance, 5))
        return assignments, reasons
