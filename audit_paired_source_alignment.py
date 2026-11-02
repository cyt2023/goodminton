"""Prove a frozen downloaded source interval's frame mapping without model scores."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import cv2
import numpy as np
import yt_dlp

from source_frame_media import ffmpeg_executable

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / '03_数据/击球标注/根因诊断/2026-10-15_同源前端对照'
SIZE = 1280 * 720 * 3


def decode(command, timeout=90):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=timeout, check=False)
    if result.returncode:
        # Never persist/print the native signed media URL from ffmpeg's stderr.
        raise RuntimeError('FFmpeg decode failed with exit ' + str(result.returncode))
    if len(result.stdout) % SIZE:
        raise RuntimeError('Partial raw frame')
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, 720, 1280, 3)
    pts = [float(s) for s in re.findall(rb'\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:([-\d.]+)',
                                      result.stderr)]
    return frames, pts


def main():
    started = time.perf_counter()
    download = json.loads((OUT / 'download_result.json').read_text())
    ffmpeg = ffmpeg_executable()
    common = [ffmpeg, '-hide_banner', '-loglevel', 'info']
    tail = ['-map', '0:v:0', '-vf', 'showinfo', '-an', '-vsync', '0',
            '-pix_fmt', 'bgr24', '-threads', '1', '-f', 'rawvideo', 'pipe:1']
    local_frames, local_pts = decode(common + ['-i', download['file']] + tail)
    local_hashes = [hashlib.sha256(f.tobytes()).hexdigest() for f in local_frames]
    local_lookup = {}
    for i, digest in enumerate(local_hashes):
        local_lookup.setdefault(digest, []).append(i)
    cv_index = json.loads((OUT / 'raw/local_decode_index.json').read_text())
    cv_equal = len(cv_index) == len(local_frames) and all(
        r['pixel_sha256'] == h for r, h in zip(cv_index, local_hashes))
    print('LOCAL_INDEX', len(local_frames), 'opencv_pixels_equal', cv_equal, flush=True)
    with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True, 'noplaylist': True,
                          'socket_timeout': 15, 'retries': 1, 'extractor_retries': 1}) as ydl:
        metadata = ydl.extract_info(download['source_url'], download=False)
    fmt = next(f for f in metadata['formats'] if f['format_id'] == download['format_id'])
    assert fmt['width'] == 1280 and fmt['height'] == 720 and fmt['fps'] == 30
    # These anchors were fixed before any local or classifier results were seen.
    anchors = [16771, 17050, 17344]
    matches = []
    for anchor in anchors:
        seek = anchor / 30 - .1
        command = common + ['-ss', f'{seek:.9f}', '-copyts', '-i', fmt['url'],
                            '-frames:v', '7'] + tail
        frames, pts = decode(command)
        assert len(pts) >= len(frames)
        rows = []
        for frame, source_pts in zip(frames, pts):
            digest = hashlib.sha256(frame.tobytes()).hexdigest()
            local_ids = local_lookup.get(digest, [])
            source_frame = round(source_pts * 30)
            assert abs(source_pts - source_frame / 30) < .0001
            rows.append({'source_pts_seconds': source_pts,
                         'global_frame_at_30fps': source_frame,
                         'pixel_sha256': digest, 'matching_local_frames': local_ids,
                         'candidate_offsets': [source_frame - i for i in local_ids]})
        matches.append({'requested_global_anchor': anchor, 'seek_seconds': seek, 'rows': rows})
        print('ANCHOR', anchor, 'pixel_matched', sum(bool(r['matching_local_frames']) for r in rows),
              '/', len(rows), flush=True)
    offsets = {r['candidate_offsets'][0] for a in matches for r in a['rows']
               if len(r['matching_local_frames']) == 1}
    passed = len(offsets) == 1 and all(
        any(len(r['matching_local_frames']) == 1 for r in a['rows']) for a in matches)
    result = {'source_url': download['source_url'], 'format_id': download['format_id'],
              'download_sha256': download['sha256'], 'anchors': matches,
              'local_actual_decoded_frames': len(local_frames),
              'container_metadata_frames': download['decode_metadata']['frames'],
              'local_pts_first': local_pts[0], 'local_pts_last': local_pts[len(local_frames)-1],
              'opencv_ffmpeg_pixels_all_equal': cv_equal,
              'global_minus_local_frame_offsets': sorted(offsets),
              'same_current_source_pixel_alignment_passed': passed,
              'author_raw_pixel_identity_proven': False,
              'limitation': 'Anchors establish current native source PTS to remux mapping; author historical source bytes/encoding unavailable.',
              'elapsed_seconds': time.perf_counter() - started}
    (OUT / 'source_alignment.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    assert passed, 'Do not run paired model comparison until alignment is established'
    # Save a per-frame map only after passing the pixel identity gate.
    offset = next(iter(offsets))
    (OUT / 'raw/verified_frame_map.json').write_text(json.dumps([
        {'local_frame': i, 'local_pts': local_pts[i],
         'global_frame': round(local_pts[i] * 30) + offset,
         'source_pts': local_pts[i] + offset / 30, 'pixel_sha256': h}
        for i, h in enumerate(local_hashes)], ensure_ascii=False, indent=2))
    print('ALIGNED', sorted(offsets), flush=True)


if __name__ == '__main__':
    main()
