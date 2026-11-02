"""Obtain native source audio on the frozen continuous rally presentation timeline."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/audit/
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time
import wave

import numpy as np
import yt_dlp

from analysis_jobs import atomic_json, file_hash
from source_frame_media import ffmpeg_executable

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '03_数据/击球标注/根因诊断/2026-10-15_同源前端对照'


def main():
    started = time.perf_counter()
    source_url = 'https://www.youtube.com/watch?v=XmJ-OdVFQtk'
    folder = OUT / 'continuous_rally'
    protocol = {'declared_at': datetime.now(timezone.utc).isoformat(), 'source_url': source_url,
                'format_rule': 'AAC140 family, original-preference English track (YouTube may suffix audio languages); no onset/scores used for track selection.', 'start_native_pts_seconds': 16756 / 30,
                'end_native_pts_seconds': 17359 / 30, 'output_sample_rate': 16000,
                'method': 'Seek to native552s keeping sourcePTS, resample native timeline to16kHz, atrim by declared sourcePTS, reset outputPTS. PCM16 mono for unmodified production audio_onsets.',
                'audio_lag_or_event_times_used_for_crop': False,
                'physical_audio_video_lag': 'Not corrected here; unchanged production refinement estimates its own lag.',
                'encoding': 'PCM16 quantization declared; this is a separate aligned track, not original video byte identity.'}
    atomic_json(folder / 'audio_download_protocol.json', protocol)
    with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True, 'noplaylist': True,
                          'socket_timeout': 15, 'retries': 1, 'extractor_retries': 1}) as ydl:
        metadata = ydl.extract_info(source_url, download=False)
    available = [f for f in metadata['formats'] if f['format_id'].split('-')[0] == '140']
    english = [f for f in available if f.get('language', '').startswith('en')]
    fmt = max(english or available, key=lambda f: (f.get('language_preference', -1), f.get('abr', 0)))
    protocol.update(format_id=fmt['format_id'], language=fmt.get('language'),
                    language_preference=fmt.get('language_preference'),
                    format_note=fmt.get('format_note'))
    atomic_json(folder / 'audio_download_protocol.json', protocol)
    wav = folder / 'source_aligned_audio.wav'
    # Resample before trim, so clipping is defined on the source PTS lattice;
    # resetting PTS afterwards preserves every audible onset's relative time.
    filter_graph = 'aresample=16000,atrim=start=558.533333333:end=578.633333333,ashowinfo,asetpts=PTS-STARTPTS,ashowinfo'
    run = subprocess.run([ffmpeg_executable(), '-hide_banner', '-loglevel', 'info',
            '-ss', '552', '-copyts', '-i', fmt['url'], '-map', '0:a:0', '-t', '27',
            '-af', filter_graph, '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le',
            '-y', str(wav)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
    if run.returncode:
        raise RuntimeError('Native audio extraction failed: exit ' + str(run.returncode))
    with wave.open(str(wav), 'rb') as audio:
        count, rate = audio.getnframes(), audio.getframerate()
        data = np.frombuffer(audio.readframes(count), dtype='<i2')
        assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
    assert rate == 16000 and abs(count - round(20.1 * rate)) <= 1
    assert np.max(abs(data.astype(float))) > 20
    first_packets = re.findall(rb'Parsed_ashowinfo_(\d+)[^\n]*\bn:0\s+pts:(-?\d+)\s+pts_time:([-\d.]+)', run.stderr)
    assert len(first_packets) == 2
    source_first_sample = int(first_packets[0][1])
    output_first_sample = int(first_packets[1][1])
    crop_error = source_first_sample / rate - protocol['start_native_pts_seconds']
    assert abs(crop_error) < 1 / rate and output_first_sample == 0
    atomic_json(folder / 'source_aligned_audio_verification.json', {
                'audio_sha256': file_hash(wav), 'native_pts_alignment_verified': True,
                'native_first_sample_at_16khz': source_first_sample,
                'output_first_sample': output_first_sample,
                'native_crop_start_quantization_seconds': crop_error,
                'sample_rate': rate, 'samples': count, 'duration_seconds': count / rate,
                'physical_audio_video_lag_verified': False,
                'method': 'Actual ffmpeg ashowinfo at first trimmed nativePTS and after resetPTS; first native sample differs from requested start by less than1sample.'})
    atomic_json(folder / 'audio_download_result.json', {'protocol': protocol,
                'file': str(wav), 'sha256': file_hash(wav), 'samples': count,
                'duration_seconds': count / rate, 'sample_rate': rate,
                'crop_quantization_max_seconds': 1 / rate,
                'time_origin': 'Native video16756/30 seconds, independent of hit labels.',
                'elapsed_seconds': time.perf_counter() - started})
    print('ALIGNED_AUDIO', count, '/', rate, 'seconds', count / rate, flush=True)


if __name__ == '__main__':
    main()
