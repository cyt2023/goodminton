"""Materialize the frozen author per-hit schedules as lossless BGR clips."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/audit/
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analysis_jobs import atomic_json, file_hash
from source_frame_media import ffmpeg_executable

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '03_数据/击球标注/根因诊断/2026-10-15_同源前端对照'


def main():
    alignment = json.loads((OUT / 'source_alignment.json').read_text())
    assert alignment['same_current_source_pixel_alignment_passed']
    mapping = json.loads((OUT / 'raw/verified_frame_map.json').read_text())
    download = json.loads((OUT / 'download_result.json').read_text())
    rows = list(csv.DictReader((OUT / 'selected_rally_samples.csv').open()))
    protocol = {
        'declared_at': datetime.now(timezone.utc).isoformat(),
        'selected_samples_sha256': file_hash(OUT / 'selected_rally_samples.csv'),
        'source_alignment_sha256': file_hash(OUT / 'source_alignment.json'),
        'frames': 'Individually reconstruct all 20 frozen author schedules; keep exact released NPY N, including six rounded-duration extra frames.',
        'encoding': 'Decode current native format136 using ffmpeg bgr24, losslessly encode FFV1 bgr0. Verify every BGR pixel through OpenCV after encoding.',
        'conditions': ['author_pose_author_shuttle', 'local_pose_author_shuttle',
                       'author_pose_local_shuttle', 'local_pose_local_shuttle'],
        'pose_frontend': 'Unchanged production specialist_pose.py / YOLOX + BGR RTMPose; scout each clip with unchanged scout_automatic_players.',
        'shuttle_frontend': 'Unchanged production specialist_tracknet.py, TrackNetV3 nonoverlap + InpaintNet, each clip separately.',
        'model': 'Original released BST-0 fixed weights; same author helpers, exact N, padding and Top/Bottom input order. No training or score-based filtering.',
        'missing_policy': 'Main four cells keep each delivered modality unchanged: author shuttle contains its own failed-pair zero mask, local shuttle retains coordinates on missing local pairs as in production. Secondary four cells apply a common author/local paired-pose intersection mask to shuttle in every cell.',
        'secondary_declared': ['common_pose_pair_mask_four_cells', 'RGB_pose_same_boxes_four_cells'],
        'labels': 'Author selected validation classes only; no user labels changed.',
        'limits': ['One metadata-selected rally, 20 correlated strokes; not general accuracy.',
                   'Current native source timeline aligned, author historical source pixel identity unavailable.',
                   'Lossless replay avoids new encoding artifacts but cannot reproduce author historical MP4 reencoding.',
                   'Known/author shot boundaries supplied; no hit detection assessed.',
                   'Frontend runs per stroke, differs from production whole-upload continuous tracking.'],
        'test_split_used': False, 'production_or_weights_changed': False,
    }
    atomic_json(OUT / 'paired_comparison_protocol.json', protocol)
    ffmpeg = ffmpeg_executable()
    raw = subprocess.check_output([ffmpeg, '-hide_banner', '-loglevel', 'error',
        '-i', download['file'], '-map', '0:v:0', '-an', '-vsync', '0',
        '-pix_fmt', 'bgr24', '-threads', '1', '-f', 'rawvideo', 'pipe:1'])
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 720, 1280, 3)
    assert len(frames) == len(mapping)
    global_lookup = {r['global_frame']: r['local_frame'] for r in mapping}
    dataset_csv = ROOT / '04_原型/vendor/BST/ShuttleSet/set/Carolina_Marin_An_Se_Young_TOYOTA_THAILAND_OPEN_2021_SemiFinals/set1.csv'
    labels = {int(float(r['frame_num'])): r for r in csv.DictReader(dataset_csv.open()) if r['rally'] == '5'}
    sheet = Image.new('RGB', (1600, 5 * 275), 'white')
    font = ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc', 17)
    clips = []
    for i, row in enumerate(rows):
        start, n = int(row['author_start_frame_at_nominal_30fps']), int(row['actual_author_npy_frame_count'])
        ids = [global_lookup[g] for g in range(start, start + n)]
        assert ids == list(range(ids[0], ids[0] + n))
        for local_id in ids:
            m = mapping[local_id]
            assert abs(m['local_pts'] - local_id / 30) < .00001, 'PTS discontinuity within selected window'
            assert hashlib.sha256(frames[local_id].tobytes()).hexdigest() == m['pixel_sha256']
        folder = OUT / 'clips' / row['sample_id']
        folder.mkdir(parents=True, exist_ok=True)
        file = folder / 'source.avi'
        data = frames[ids].tobytes()
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
                        '-pix_fmt', 'bgr24', '-s', '1280x720', '-r', '30', '-i', 'pipe:0',
                        '-an', '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0',
                        '-threads', '2', str(file)], input=data, check=True)
        cap = cv2.VideoCapture(str(file))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n
        decoded = 0
        clip_frame_map = []
        while True:
            ok, image = cap.read()
            if not ok:
                break
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            assert digest == mapping[ids[decoded]]['pixel_sha256'], 'Lossless decoder pixel mismatch'
            clip_frame_map.append({'clip_index': decoded, 'global_frame': start + decoded,
                                   'pixel_sha256': digest, 'clip_pts': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000})
            decoded += 1
        cap.release()
        assert decoded == n
        atomic_json(folder / 'frame_map.json', clip_frame_map)
        clips.append({**row, 'source_file': str(file), 'source_sha256': file_hash(file),
                      'frames': n, 'global_frame_start': start,
                      'all_decoded_pixels_verified': True,
                      'frame_map_sha256': file_hash(folder / 'frame_map.json')})
        # Timeline visual gate: CSV stroke contact coordinate and exact source frame.
        hit = int(row['hit_frame_num'])
        im = Image.fromarray(cv2.cvtColor(frames[global_lookup[hit]], cv2.COLOR_BGR2RGB)).resize((400, 225))
        draw = ImageDraw.Draw(im)
        annotation = labels[hit]
        if annotation['hit_x'] and annotation['hit_y']:
            x, y = float(annotation['hit_x']) * 400 / 1280, float(annotation['hit_y']) * 225 / 720
            draw.ellipse((x-6,y-6,x+6,y+6), outline='red', width=2)
        x, y = (i % 4) * 400, (i // 4) * 275
        sheet.paste(im, (x,y))
        ImageDraw.Draw(sheet).text((x+5,y+228), f"{row['sample_id']}  {row['author_25class']}", font=font, fill='black')
        ImageDraw.Draw(sheet).text((x+5,y+250), f"frame {hit} / CSV score {annotation['roundscore_A']}:{annotation['roundscore_B']}", font=font, fill='black')
        print('CLIP_VERIFIED', row['sample_id'], n, flush=True)
    sheet.save(OUT / 'timeline_20_contacts.jpg', quality=95)
    atomic_json(OUT / 'clips_manifest.json', {'clips': clips, 'total_clip_frames': sum(r['frames'] for r in clips),
                'all_pixels_verified': True, 'author_historical_pixel_identity_proven': False})


if __name__ == '__main__':
    main()
