"""Paired development ablations on saved BST review examples.

This script never changes labels, frozen predictions, model weights, or runtime
configuration. Human contacts/players are oracle diagnostic inputs, not an
automatic-system result. Cleanup rules are fixed before this run and are not
selected per example using its stroke label.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np
import torch

from analysis_jobs import DATA, ROOT, atomic_json, file_hash, registered_video
from personal_learning import extractor
from input_quality import InputQuality
from specialist_model import BST_NAME
from stroke_context import window
from stroke_features import HitFeatures
from stroke_window_repairs import translate_contact_window
from stroke_review import load_source, read_labels, folder_for
from stroke_taxonomy import AUTHOR_MAP

AUDIT = DATA/'击球标注/公开模型检查批次/运行记录/460b74361c3a47cd9bca7066c72ea514/2026-10-07_人工标注后检查.json'
VERSION = 'saved-input-repair-ablation-v1'


def actor(label, order):
    return order[{'Top': 0, 'Bottom': 1}[label.split('_', 1)[0]]] if '_' in label else 'unknown'


def prediction(classifier, values, target, candidate_target=None):
    p = classifier.predict(values)
    scores = np.asarray(p['bst0_all_scores'])
    label = p['bst0'][0]['label']
    prefix = 'Top_' if target == classifier.order[0] else 'Bottom_'
    possible = [i for i, name in enumerate(classifier.classes) if name.startswith(prefix)]
    conditional = int(max(possible, key=lambda i: scores[i]))
    candidate_prefix = 'Top_' if (candidate_target or target) == classifier.order[0] else 'Bottom_'
    candidate_possible = [i for i, name in enumerate(classifier.classes) if name.startswith(candidate_prefix)]
    candidate_conditional = int(max(candidate_possible, key=lambda i: scores[i]))
    return {'author_label': label, 'code': AUTHOR_MAP[label.split('_', 1)[-1]],
            'mapped_actor': actor(label, classifier.order), 'player_order': list(classifier.order),
            'scores': scores.tolist(), 'known_target_code': AUTHOR_MAP[classifier.classes[conditional].split('_', 1)[-1]],
            'known_target_author_label': classifier.classes[conditional],
            'automatic_candidate_conditioned_code': AUTHOR_MAP[classifier.classes[candidate_conditional].split('_', 1)[-1]],
            'automatic_candidate_conditioned_author_label': classifier.classes[candidate_conditional],
            'known_target_probability_mass': float(scores[possible].sum())}


def corrected_inputs(classifier, features, event, *, pose=False, ball=False):
    """Independent one-factor repairs; exact author transformations are retained."""
    if not pose and not ball:
        return classifier.inputs(event)[0]
    joints, shuttle = [], []
    h = classifier.helpers
    for i in range(event['start_frame'], event['end_frame']+1):
        people = [classifier.dense['frames'][i]['players'].get(s) for s in classifier.order]
        normalized = np.zeros((2, 17, 2), float)
        if pose:
            # A missing opponent does not erase the available target; unsupported
            # joints get the explicit zero representation, with no interpolation.
            for k, person in enumerate(people):
                if person is None:
                    continue
                pts = np.asarray([[p['x'], p['y']] for p in person['keypoints']], float)
                good = np.asarray([p['score'] >= .35 and p.get('in_frame', True) for p in person['keypoints']])
                normalized[k] = h['normalize_joints'](pts[None], np.asarray([person['bbox']]), center_align=True)[0]
                normalized[k, ~good] = 0
        elif all(p is not None for p in people):
            pts = np.asarray([[[p['x'], p['y']] for p in person['keypoints']] for person in people], float)
            normalized = h['normalize_joints'](pts, np.asarray([p['bbox'] for p in people]), center_align=True)
        joints.append(normalized)
        b = classifier.trajectory['frames'][i]
        if ball:
            shuttle.append(features.ball[i] if features.valid[i] else [0, 0])
        else:
            shuttle.append((b['rectified_xy'] if b['inpaint_requested'] else b['raw_xy']) or [0, 0])
    s = h['normalize_shuttlecock'](np.asarray(shuttle), classifier.dense['source_width'], classifier.dense['source_height'])
    j, p, s, length = h['make_seq_len_same'](100, np.asarray(joints), np.zeros((len(joints), 2, 2)), s)
    j = np.concatenate((j, h['create_bones'](j, classifier.bones)), axis=-2).reshape(100, 2, 72)
    return j.astype('float32'), p.astype('float32'), s.astype('float32'), int(length)


def metrics(rows, variant):
    out = {'total': len(rows), 'type_correct': 0, 'actor_agreement_under_unconfirmed_mapping': 0,
           'joint_correct_under_unconfirmed_mapping': 0, 'known_human_target_type_correct': 0,
           'automatic_candidate_conditioned_type_correct':0}
    for row in rows:
        p, y = row['variants'][variant], row['human']
        t, a = p['code'] == y['stroke_type'], p['mapped_actor'] == y['player']
        out['type_correct'] += t
        out['actor_agreement_under_unconfirmed_mapping'] += a
        out['joint_correct_under_unconfirmed_mapping'] += t and a
        out['known_human_target_type_correct'] += p['known_target_code'] == y['stroke_type']
        out['automatic_candidate_conditioned_type_correct'] += p['automatic_candidate_conditioned_code'] == y['stroke_type']
    out['prediction_distribution'] = dict(Counter(r['variants'][variant]['code'] for r in rows))
    return out


def montage(classifier, features, human, path, source):
    center = int(np.argmin(abs(features.times-human['peak_time'])))
    cap = cv2.VideoCapture(str(source))
    frames = []
    for offset in [-12, -6, -3, 0, 3, 6, 12, 18]:
        i = int(np.clip(center+offset, 0, len(features.times)-1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, image = cap.read()
        if not ok:
            continue
        for side in classifier.order:
            person = classifier.dense['frames'][i]['players'].get(side)
            if not person:
                continue
            color = (40, 220, 40) if side == human['player'] else (230, 130, 50)
            pts = [(round(p['x']), round(p['y'])) for p in person['keypoints']]
            for u, v in [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]:
                cv2.line(image, pts[u], pts[v], color, 1, cv2.LINE_AA)
            for p, xy in zip(person['keypoints'], pts):
                cv2.circle(image, xy, 3, color if p['score'] >= .35 else (0,0,255), -1)
            x,y = round(person['bbox'][0]), round(person['bbox'][1])
            cv2.putText(image, side, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
        ball = classifier.trajectory['frames'][i]
        if ball['raw_visible'] and ball['raw_xy']:
            xy = tuple(map(round, ball['raw_xy']))
            cv2.circle(image, xy, 8, (0,255,255), 2)
            cv2.putText(image, 'RAW', (xy[0]+9,xy[1]), cv2.FONT_HERSHEY_SIMPLEX, .4, (0,255,255), 1)
        if ball['inpaint_requested'] and ball['rectified_xy']:
            xy = tuple(map(round, ball['rectified_xy']))
            cv2.drawMarker(image, xy, (255,0,255), cv2.MARKER_TILTED_CROSS, 12, 2)
        label = f"{features.times[i]:.3f}s f{i} {human['player']} / {human['stroke_type']}"
        cv2.rectangle(image, (0,0), (image.shape[1],28), (0,0,0), -1)
        cv2.putText(image, label, (5,20), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
        image = cv2.resize(image, (640, round(image.shape[0]*640/image.shape[1])))
        frames.append(image)
    cap.release()
    if len(frames) == 8:
        cv2.imwrite(str(path), np.vstack([np.hstack(frames[:4]), np.hstack(frames[4:])]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--no-montages', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    root = DATA/'击球标注/输入修复诊断'/(args.output or uuid.uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    old = json.loads(AUDIT.read_text())
    inputs, keys, rows, hashes, replay_errors = [], [], [], {}, []
    variants = ['frozen_replay', 'human_center_same_duration', 'human_center_reviewed_neighbors',
                'human_context_shuttle_filter', 'human_context_pose_filter', 'human_context_combined_filter',
                'human_context_swapped_order_sensitivity']
    hashes[str(AUDIT.relative_to(DATA))] = file_hash(AUDIT)
    for job_entry in old['jobs']:
        job = job_entry['job_id']
        report, events, event_file = load_source(job)
        doc = read_labels(job, report['source_sha256'])
        frozen_path = folder_for(job)/'public_only_predictions.json'
        frozen = json.loads(frozen_path.read_text())
        snapshots = root/'snapshots'/job
        atomic_json(snapshots/'annotations.json', doc)
        atomic_json(snapshots/'frozen_predictions.json', frozen)
        classifier, dense = extractor(job)
        features = HitFeatures(classifier)
        quality = InputQuality(dense,classifier.trajectory)
        source = registered_video(report['source_video_id'], allow_reference=True)['path']
        frozen_events = {r['event_id']: r for r in frozen['events']}
        indexed = {e['id']: e for e in events}
        for p in [folder_for(job)/'annotations.json', frozen_path,
                  DATA/'专项分析'/job/'dense_pose.json', DATA/'专项分析'/job/'trajectory.json']:
            hashes[str(p.relative_to(DATA))] = file_hash(p)
        for selected in job_entry['evaluation']['evaluated']:
            eid = selected['event_id']
            human, saved = doc['labels'][eid], frozen_events[eid]
            event = {**indexed[eid], **saved['model_window']}
            values, _ = classifier.inputs(event)
            replay = prediction(classifier, values, human['player'],saved['player'])
            error = float(np.max(abs(np.asarray(replay['scores'])-saved['baseline_scores'])))
            replay_errors.append(error)
            if replay['author_label'] != saved['baseline_author_label'] or error > 1e-5:
                raise ValueError(f'Frozen replay mismatch {job}/{eid}: {error}')
            shifted = translate_contact_window(dense,event,human['peak_time'])
            reviewed = {**indexed[eid], **window(dense,indexed[eid],events,doc['labels'])}
            row = {'job_id': job, 'event_id': eid, 'human': human,
                   'frozen_window': saved['model_window'], 'human_shifted_window': shifted,
                   'human_neighbor_window': reviewed, 'frozen_replay_max_score_error': error,
                   'variants': {'frozen_replay': replay}, 'input_quality': {},
                   'input_order_manually_confirmed': False}
            for name in variants:
                classifier.order = list(saved['input_quality']['player_input_order'])
                selected_window = event if name == 'frozen_replay' else shifted if name == 'human_center_same_duration' else reviewed
                if name.endswith('swapped_order_sensitivity'):
                    classifier.order.reverse()
                values = corrected_inputs(classifier, features, selected_window,
                    pose=name in ['human_context_pose_filter','human_context_combined_filter'],
                    ball=name in ['human_context_shuttle_filter','human_context_combined_filter'])
                row['variants'][name] = prediction(classifier,values,human['player'],saved['player'])
                inputs.append(values);keys.append((job,eid,name))
            classifier.order = list(saved['input_quality']['player_input_order'])
            row['input_quality']['frozen_window'] = quality.window(event,target_player=human['player'],target_time=human['peak_time'],order=classifier.order)
            row['input_quality']['human_neighbor_window'] = quality.window(reviewed,target_player=human['player'],target_time=human['peak_time'],order=classifier.order)
            row['automatic_candidate_player'] = saved['player']
            if not args.no_montages:
                image = root/'montages'/f'{job}_{eid}.jpg'
                image.parent.mkdir(parents=True, exist_ok=True)
                montage(classifier,features,human,image,source)
                row['review_montage'] = str(image.relative_to(DATA))
            rows.append(row)
            print('INPUT_ABLATION',job,eid,{v: row['variants'][v]['code'] for v in variants},flush=True)
    np.savez_compressed(root/'inputs.npz',keys=np.asarray(keys),
        joints_and_bones=np.stack([v[0] for v in inputs]),positions=np.stack([v[1] for v in inputs]),
        shuttle=np.stack([v[2] for v in inputs]),video_len=np.asarray([v[3] for v in inputs]))
    summary = {v: metrics(rows,v) for v in variants}
    transitions = {}
    for v in variants[1:]:
        gained, lost = [], []
        for r in rows:
            old_ok = r['variants']['frozen_replay']['code'] == r['human']['stroke_type']
            new_ok = r['variants'][v]['code'] == r['human']['stroke_type']
            if old_ok != new_ok:
                (gained if new_ok else lost).append({'job_id':r['job_id'],'event_id':r['event_id']})
        transitions[v] = {'type_gained':gained,'type_lost':lost}
    source_after = {name:file_hash(DATA/name) for name in hashes}
    if hashes != source_after:
        raise ValueError('Source files changed during diagnostic')
    result = {'version':VERSION,'created_at':datetime.now(timezone.utc).isoformat(),
        'scope':'Paired development diagnostics on the same 22 selected, previously inspected examples; no independent generalization or whole-video detection claim.',
        'protocol': {'frozen_replay':'Exact saved automatic window; must reproduce all frozen probabilities within 1e-5.',
            'human_center_same_duration':'Translate original frame bounds to the human-confirmed center, preserving width except recording edges; pose and shuttle unchanged.',
            'human_center_reviewed_neighbors':'Human contact/player plus the existing stroke_context reviewed-opponent-neighbor policy; no stroke labels used in numerical input.',
            'human_context_shuttle_filter':'Human neighbor window; only observed heatmap>=.15 shuttle, remove >=7-frame stationary pixels and isolated >.1 image-diagonal jumps; no inpainting.',
            'human_context_pose_filter':'Human neighbor window; independent player missing masks, joints with confidence<.35 or outside image zeroed; no interpolation.',
            'human_context_combined_filter':'Combine the preceding two fixed cleanup rules.',
            'human_context_swapped_order_sensitivity':'Reverse BOTH input player axis and label-to-player decoding, hold everything else; sensitivity check, not a selected per-example correction.',
            'known_human_target_type_correct':'Restrict original 25-class scores to classes for the human-known target. Oracle assisted diagnostic only; cannot count as autonomous performance.',
            'automatic_candidate_conditioned_type_correct':'Restrict scores to classes of the original saved automatic candidate player; no human actor read in this decision. On these 22 selected records the candidate player equals the human actor for every record. Still not whole-video accuracy.'},
        'summary':summary,'per_job':{job:{v:metrics([r for r in rows if r['job_id']==job],v) for v in variants} for job in sorted({r['job_id'] for r in rows})},
        'type_transitions_from_frozen':transitions,'rows':rows,
        'replay_max_score_error':max(replay_errors),'source_sha256':hashes,
        'source_files_unchanged':True,'human_labels_created':0,'model_trained':False,
        'active_model_replaced':False,'script_sha256':file_hash(Path(__file__)),
        'model_sha256':file_hash(ROOT/'models/bst'/BST_NAME),
        'runtime_seconds':time.perf_counter()-started}
    atomic_json(root/'result.json',result)
    print('RESULT',root/'result.json',flush=True)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
