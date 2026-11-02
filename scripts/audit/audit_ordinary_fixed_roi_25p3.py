"""Fixed prior ROI vs matched full-width shuttle replay on a frozen ordinary error."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/audit/
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch

from analysis_jobs import atomic_json, file_hash
from source_frame_media import ffmpeg_executable, read_source_window
from specialist_bst import SpecialistBST
from specialist_model import ROOT, BST_NAME

OUT = ROOT.parent / '03_数据/击球标注/根因诊断/2026-10-18_普通视角配对诊断'
FOLDER = OUT / 'fixed_roi_25p3'
JOB = 'J_e23c1dd9297d4bfca7c91d1d95065b68'
EVENT = '3d5e6020fbc2aa558b6a5914a9826a54'
SOURCE = ROOT.parent / '03_数据/个人录像/98f0e3817a62451b994499c0c5fe1a3a.mp4'
DENSE = ROOT.parent / '03_数据/专项分析' / JOB / 'dense_pose.json'
BALL = DENSE.with_name('trajectory.json')
RUN = ROOT.parent / '03_数据/击球标注/直接BST微调/运行记录/582817bff5454825b841d1f150c82028'


def encode_lossless(frames, file):
    height, width = frames[0].shape[:2]
    subprocess.run([ffmpeg_executable(), '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', '30',
        '-i', 'pipe:0', '-an', '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0',
        '-threads', '2', str(file)], input=b''.join(f.tobytes() for f in frames), check=True)
    cap = cv2.VideoCapture(str(file))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(frames)
    for f in frames:
        ok, actual = cap.read()
        assert ok and np.array_equal(actual, f), 'Lossless source pixels changed'
    assert not cap.read()[0]
    cap.release()


def prepare():
    FOLDER.mkdir(parents=True, exist_ok=True)
    if (FOLDER / 'protocol.json').exists():
        raise FileExistsError('Preserve prior frozen protocol')
    selection = json.loads((OUT / 'selection_protocol.json').read_text())
    selected = next(r for r in selection['all_30_mapped_rows'] if r['event_id'] == EVENT)
    assert EVENT in selection['selected_event_ids']
    dependencies = {str(p): file_hash(p) for p in [SOURCE, DENSE, BALL,
                    OUT / 'selection_protocol.json', RUN / 'personal_predictions.json',
                    RUN / 'personal_predictions.npz', ROOT / 'specialist_bst.py',
                    ROOT / 'specialist_tracknet.py', ROOT / 'models/bst' / BST_NAME,
                    ROOT / 'models/tracknetv3/TrackNet_best.pt', ROOT / 'models/tracknetv3/InpaintNet_best.pt', Path(__file__)]}
    protocol = {'declared_at': datetime.now(timezone.utc).isoformat(), 'event_id': EVENT,
        'selected_sample': selected, 'dependency_hashes': dependencies,
        'source_frames_inclusive': [704, 799], 'evaluated_window_inclusive': [713, 792],
        'roi': [450, 0, 925, 720],
        'roi_basis': 'Same fixed researcher ROI declared in2026-10-14_发球残余错误与目标球跟踪诊断.md before this selected25.3s control; no new boundary search or score selection.',
        'conditions': ['original_whole_upload_shuttle', 'matched_96frame_full_width', 'matched_96frame_fixed_roi'],
        'pixel_and_phase': 'All96 source frames decoded by exact sourcePTS and losslessly stored; global704 divisible16 aligns TrackNet8/Inpaint16 sequence phase. Both replays use same96-frame context/background sample count. ROI retains exact odd475px width.',
        'timing_adaptation': 'Diagnostic AVI30fps without dropping/duplicating source frames; shuttle network uses frame order, outputs remapped by original decoded indices and sourcePTS. Original dense pose retained.',
        'source_context_limit': 'Short replays use different background context than original104sec; matched full-width control distinguishes crop effects from that change.',
        'classification': 'Original fixed BST-0, original dense/order, current saved80frame window, labelTop_切球. No retraining/window/person change.',
        'mask_condition_if_available': 'Independent raw-frame reviewer definite-invalid selected ball coordinates set to missing only after a frozen review; no hand-invented correct ball points. Report separately from tracked ROI.',
        'scope': 'One of frozen3 ordinary residual errors; repeatedly inspected developer set, not new video accuracy.',
        'production_activation': False}
    atomic_json(FOLDER / 'protocol.json', protocol)
    dense = json.loads(DENSE.read_text())
    assert dense['source_sha256'] == dependencies[str(SOURCE)]
    frames, proof = read_source_window(SOURCE, dense, 704, 799)
    atomic_json(FOLDER / 'source_pts_proof.json', proof)
    manifests = []
    for name, crop in [('matched_full_width', [0, 0, 1280, 720]), ('fixed_roi', protocol['roi'])]:
        folder = FOLDER / name
        folder.mkdir(exist_ok=True)
        x1, y1, x2, y2 = crop
        image_frames = [f[y1:y2, x1:x2].copy() for f in frames]
        file = folder / 'source.avi'
        encode_lossless(image_frames, file)
        tracker = folder / 'tracker'
        tracker.mkdir(exist_ok=True)
        config = {'source_file': str(file), 'source_sha256': file_hash(file), 'specialist_folder': str(tracker)}
        atomic_json(folder / 'config.json', config)
        manifests.append({'condition': name, 'crop': crop, 'source_file': str(file),
            'source_sha256': config['source_sha256'], 'frames': len(image_frames),
            'frame_map': [{'clip_frame': i, 'source_frame': i+704, 'source_pts': proof['decoded_source_pts'][i],
                           'pixel_sha256': hashlib.sha256(f.tobytes()).hexdigest()} for i,f in enumerate(image_frames)]})
    atomic_json(FOLDER / 'clips_manifest.json', {'clips': manifests, 'all_pixels_verified': True})
    print('ORDINARY_ROI_PREPARED', EVENT, '2x96frames', flush=True)


def track():
    protocol = json.loads((FOLDER / 'protocol.json').read_text())
    assert all(file_hash(Path(p)) == h for p,h in protocol['dependency_hashes'].items())
    manifest = json.loads((FOLDER / 'clips_manifest.json').read_text())
    for clip in manifest['clips']:
        folder = Path(clip['source_file']).parent
        target = folder / 'tracker/trajectory.json'
        if target.exists():
            data = json.loads(target.read_text())
            assert data['source_sha256'] == clip['source_sha256'] and len(data['frames']) == 96
            continue
        with (folder / 'tracknet.log').open('w') as log:
            run = subprocess.run([sys.executable, '-u', str(ROOT / 'specialist_tracknet.py'),
                                  '--config', str(folder / 'config.json')], stdout=log, stderr=subprocess.STDOUT, timeout=180)
        assert run.returncode == 0, 'See isolated TrackNet log'
        data = json.loads(target.read_text())
        assert data['source_sha256'] == clip['source_sha256'] and len(data['frames']) == 96
        print('ORDINARY_ROI_TRACKED', clip['condition'], data['raw_detected_frames'], '/96', flush=True)


def compare():
    if (FOLDER / 'result.json').exists():
        raise FileExistsError('Preserve completed classification')
    protocol = json.loads((FOLDER / 'protocol.json').read_text())
    assert all(file_hash(Path(p)) == h for p,h in protocol['dependency_hashes'].items())
    dense, original = [json.loads(p.read_text()) for p in [DENSE, BALL]]
    manifest = json.loads((FOLDER / 'clips_manifest.json').read_text())
    conditions = {'original_whole_upload_shuttle': original}
    for clip in manifest['clips']:
        data = json.loads((Path(clip['source_file']).parent / 'tracker/trajectory.json').read_text())
        mapped = json.loads(json.dumps(original))
        x1,y1,_,_ = clip['crop']
        for i, item in enumerate(data['frames']):
            assert item['frame_index'] == i
            mapped['frames'][i+704] = {**item, 'frame_index': i+704,
                'source_time_nominal': dense['frames'][i+704]['source_time'],
                'raw_xy': [item['raw_xy'][0]+x1, item['raw_xy'][1]+y1] if item['raw_xy'] else None,
                'rectified_xy': [item['rectified_xy'][0]+x1, item['rectified_xy'][1]+y1] if item['rectified_xy'] else None}
        mapped['diagnostic_derivative'] = {'original_trajectory_sha256': file_hash(BALL),
             'tracker_output_sha256': file_hash(Path(clip['source_file']).parent / 'tracker/trajectory.json'),
             'clip': clip['condition'], 'crop': clip['crop'], 'source_index_start': 704}
        atomic_json(FOLDER / clip['condition'] / 'mapped_original_frame_trajectory.json', mapped)
        conditions[clip['condition']] = mapped
    review_path = OUT / 'ball_review_25p3/review.json'
    if review_path.exists():
        review = json.loads(review_path.read_text())
        assert review['source_sha256'] == dense['source_sha256']
        invalid = review['definite_invalid_frame_indices']
        assert all(713 <= i <= 792 for i in invalid)
        masked = json.loads(json.dumps(original))
        for i in invalid:
            masked['frames'][i].update(raw_xy=None, raw_visible=False, inpaint_requested=False, rectified_xy=None,
                                       diagnostic_reviewer_masked=True)
        conditions['reviewer_definite_invalid_to_missing'] = masked
        atomic_json(FOLDER / 'mask_protocol.json', {'review_sha256': file_hash(review_path),
                    'definite_invalid_frame_indices': invalid,
                    'only_removes_observations': True, 'true_coordinates_supplied': False})
    classifier = SpecialistBST(dense, original)
    torch.set_num_threads(2)
    event = protocol['selected_sample']['current_fixed_window']
    expected_records = json.loads((RUN / 'personal_predictions.json').read_text())['baseline']
    row_index = next(i for i,r in enumerate(expected_records) if r['event_id'] == EVENT)
    cache = np.load(RUN / 'personal_predictions.npz')
    result = {}
    saved_logits = {}
    baseline_inputs = None
    for name, trajectory in conditions.items():
        classifier.trajectory = trajectory
        from input_quality import InputQuality
        classifier.coordinate_quality = InputQuality(dense, trajectory)
        packed, quality = classifier.inputs(event)
        j,p,s,n = packed
        with torch.inference_mode():
            logits = classifier.model(torch.from_numpy(j)[None], torch.from_numpy(s)[None], torch.tensor([n])).numpy()[0]
        if baseline_inputs is None:
            baseline_inputs = (j.copy(),p.copy(),n)
            assert logits.argmax() == cache['baseline_logits'][row_index].argmax()
            baseline_error = float(np.max(abs(logits - cache['baseline_logits'][row_index])))
            assert baseline_error < 1e-4
        else:
            assert np.array_equal(j,baseline_inputs[0]) and np.array_equal(p,baseline_inputs[1]) and n == baseline_inputs[2]
        target = classifier.classes.index('Top_切球')
        mask = np.arange(25) != target
        ranking = np.argsort(-logits)
        result[name] = {'predicted': classifier.classes[int(ranking[0])], 'true_label': 'Top_切球',
            'correct': bool(ranking[0] == target), 'top3': [classifier.classes[int(i)] for i in ranking[:3]],
            'true_label_logit_margin_25classes': float(logits[target] - logits[mask].max()),
            'raw_ball_fraction': quality['raw_shuttle_fraction'], 'paired_fraction': quality['paired_player_fraction'],
            'length': n, 'input_sha256': hashlib.sha256(j.tobytes()+s.tobytes()+np.asarray(n,dtype='int64').tobytes()).hexdigest()}
        saved_logits[name] = logits
        np.savez_compressed(FOLDER / (name + '_inputs.npz'), j=j,p=p,s=s,n=np.asarray(n,dtype='int64'))
    np.savez_compressed(FOLDER / 'logits.npz', **saved_logits)
    atomic_json(FOLDER / 'result.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': file_hash(FOLDER / 'protocol.json'), 'checks': result,
        'source_pose_window_model_fixed': True, 'only_shuttle_changed': True,
        'baseline_saved_run_max_logit_difference': baseline_error,
        'production_or_labels_or_weights_changed': False,
        'scope': 'One frozen ordinary developer error. ROI is a researcher crop, not ball truth; missing-mask supplies no true coordinates. Not general accuracy.'})
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','track','compare']);args=parser.parse_args()
    {'prepare':prepare,'track':track,'compare':compare}[args.stage]()
