"""Read review pixels by source PTS and encode their presentation timeline.

Frame counters returned by a random seek are not evidence of pixel identity.
Dense source timestamps are rounded to six decimals, hence the 2 us tolerance.
"""
import math
from functools import lru_cache
from pathlib import Path
import re
import subprocess

import cv2
READER_VERSION = 'source-pts-validated-v2'
PTS_TOLERANCE = 0.000002
CLIP_TIMESCALE = 1_000_000


def ffmpeg_executable():
    binaries=(Path(__file__).resolve().parent/'.venv-pose/lib').glob('python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*')
    executable=next((path for path in binaries if path.is_file()),None)
    if executable is None:
        raise ValueError('本地视频编码器不可用')
    return executable


def _ffmpeg_source_frames(executable, source, seek_time=None, first_only=False):
    command=[executable,'-hide_banner','-nostdin']
    if seek_time is not None:
        command += ['-ss',str(seek_time)]
    command += ['-copyts','-i',source,'-map','0:v:0','-vf','showinfo']
    if first_only:
        command += ['-frames:v','1']
    command += ['-an','-f','null','-']
    try:
        result=subprocess.run(command,capture_output=True,timeout=30)
    except subprocess.TimeoutExpired as error:
        raise ValueError('原视频源时间确认超时，预览未发布') from error
    log=result.stderr.decode(errors='replace')
    clock=re.search(r'config in time_base:\s*(\d+)/(\d+)',log)
    rows=re.findall(r'\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:[^\s]+\s+duration:\s*(\d+)',log)
    if result.returncode or not clock or not rows:
        raise ValueError('原视频源时间无法确认，预览未发布')
    tick=int(clock[1])/int(clock[2])
    return [(int(pts)*tick,int(duration)*tick) for pts,duration in rows]


@lru_cache(maxsize=8)
def _source_tail_timing(executable, source, size, mtime_ns, dense_first_pts, last_source_pts):
    # OpenCV reports a decoder-relative clock while ffmpeg -copyts keeps raw
    # stream PTS. Measure both origins on the same source before comparing its
    # last frame. This is an origin translation; frame intervals stay intact.
    capture=cv2.VideoCapture(source)
    try:
        ok,_=capture.read()
        decoder_first_pts=float(capture.get(cv2.CAP_PROP_POS_MSEC)/1000)
    finally:
        capture.release()
    if not ok or not math.isfinite(decoder_first_pts) or abs(decoder_first_pts-dense_first_pts)>PTS_TOLERANCE:
        raise ValueError('原视频首帧与逐帧分析未对齐，预览未发布')
    raw_first_pts=_ffmpeg_source_frames(executable,source,first_only=True)[0][0]
    origin_offset=raw_first_pts-decoder_first_pts
    seek_time=max(0.,last_source_pts-decoder_first_pts-1.)
    raw_last_pts,duration=_ffmpeg_source_frames(executable,source,seek_time=seek_time)[-1]
    normalized_last_pts=raw_last_pts-origin_offset
    if abs(normalized_last_pts-last_source_pts)>PTS_TOLERANCE:
        raise ValueError('原视频末帧与逐帧分析未对齐，预览未发布')
    if duration<=0:
        raise ValueError('原视频末帧持续时间无法确认，预览未发布')
    return {'source_stat':{'path':source,'size_bytes':size,'mtime_ns':mtime_ns},
            'dense_first_source_pts':dense_first_pts,'decoder_first_source_pts':decoder_first_pts,
            'ffmpeg_first_raw_pts':raw_first_pts,'ffmpeg_last_raw_pts':raw_last_pts,
            'ffmpeg_to_decoder_origin_offset':origin_offset,
            'ffmpeg_last_normalized_pts':normalized_last_pts,'decoder_last_source_pts':last_source_pts,
            'last_pts_error_seconds':normalized_last_pts-last_source_pts,
            'final_packet_duration':duration,'time_mapping':'decoder_pts = ffmpeg_raw_pts - origin_offset'}


def read_source_window(source, dense, lo, hi, headroom=.5):
    """Return verified frames, actual source PTS and the last frame's duration.

    Seek earlier only to accelerate decoding. A missing target PTS triggers a
    fresh decode from zero; a second failure withholds media. Read the following
    source frame too when available, to preserve the final frame's hold time.
    """
    records = dense['frames']
    if not 0 <= lo <= hi < len(records):
        raise ValueError('原视频逐帧范围无效')
    stop = min(hi + 1, len(records) - 1)
    expected = [float(records[i]['source_time']) for i in range(lo, stop + 1)]
    if any(records[i]['frame_index'] != i for i in range(lo, stop + 1)):
        raise ValueError('原视频逐帧索引不连续，预览未发布')
    if not all(math.isfinite(t) for t in expected) or any(b <= a for a,b in zip(expected, expected[1:])):
        raise ValueError('原视频源时间无效，预览未发布')
    attempts = []
    for seek_earlier in (True, False):
        capture = cv2.VideoCapture(str(source))
        seek_time = max(0., expected[0] - headroom) if seek_earlier else 0.
        if seek_time:
            capture.set(cv2.CAP_PROP_POS_MSEC, seek_time * 1000)
        frames, pts, reads, problem, previous = [], [], 0, None, None
        try:
            while len(pts) < len(expected):
                ok, frame = capture.read()
                if not ok:
                    problem = 'unexpected_end'
                    break
                reads += 1
                stamp = float(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                if not math.isfinite(stamp) or (previous is not None and stamp <= previous):
                    problem = 'non_monotonic_decoder_pts'
                    break
                previous = stamp
                target = expected[len(pts)]
                if stamp < target - PTS_TOLERANCE:
                    continue
                if abs(stamp - target) > PTS_TOLERANCE:
                    problem = 'missing_source_pts'
                    break
                frames.append(frame)
                pts.append(stamp)
        finally:
            capture.release()
        attempts.append({'seek_time': seek_time, 'from_zero': not seek_earlier,
                         'decoded_frames': reads, 'problem': problem})
        if problem is not None:
            continue
        tail_timing=None
        if hi < len(records) - 1:
            last_duration = pts[-1] - pts[-2]
            following_pts = pts.pop()
            frames.pop()
            duration_basis = 'next_decoded_source_pts'
        else:
            source_path=Path(source);stat=source_path.stat()
            tail_timing=_source_tail_timing(str(ffmpeg_executable()),str(source_path),stat.st_size,stat.st_mtime_ns,
                                            float(records[0]['source_time']),pts[-1])
            last_duration=tail_timing['final_packet_duration']
            following_pts = pts[-1] + last_duration
            duration_basis = 'ffmpeg_final_video_frame_packet_duration_with_explicit_origin'
        if not math.isfinite(last_duration) or last_duration <= 0:
            raise ValueError('原视频末帧持续时间无效，预览未发布')
        proof={'reader_version': READER_VERSION, 'decoder': f'OpenCV {cv2.__version__}',
                        'source_frame_indices': list(range(lo, hi + 1)),
                        'decoded_source_pts': pts, 'following_source_pts': following_pts,
                        'last_frame_duration': last_duration, 'last_frame_duration_basis': duration_basis,
                        'source_window_duration': following_pts - pts[0],
                        'pts_tolerance_seconds': PTS_TOLERANCE, 'attempts': attempts}
        if tail_timing is not None:
            proof['source_tail_timing']=tail_timing
        return frames,proof
    raise ValueError('原视频解码时间与逐帧分析未对齐，预览未发布')


def encode_pts_clip(executable, frames, proof, destination):
    """Encode the already verified BGR pixels with explicit source-relative PTS.

    A microsecond MP4 clock preserves the dense timeline within its rounding
    precision. The packet duration is explicit so the final frame is playable.
    Validate frame count, every decoded output PTS and stream duration before
    the caller publishes the partial file.
    """
    source_pts = proof['decoded_source_pts']
    if not frames or len(frames) != len(source_pts):
        raise ValueError('击球片段帧数与源时间不一致')
    ticks = [round((t - source_pts[0]) * CLIP_TIMESCALE) for t in source_pts]
    if any(b <= a for a,b in zip(ticks, ticks[1:])):
        raise ValueError('击球片段源时间不可编码')
    last_ticks = round(proof['last_frame_duration'] * CLIP_TIMESCALE)
    if last_ticks <= 0:
        raise ValueError('击球片段末帧持续时间不可编码')
    expression = str(ticks[-1])
    for i in range(len(ticks) - 2, -1, -1):
        expression = f'if(eq(N\\,{i})\\,{ticks[i]}\\,{expression})'
    height, width = frames[0].shape[:2]
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError('击球片段画面尺寸不一致')
    # The input cadence is a transport placeholder; setpts replaces every PTS.
    command = [str(executable), '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-s', f'{width}x{height}', '-framerate', '30', '-i', 'pipe:0', '-an',
               '-vf', f'settb=1/{CLIP_TIMESCALE},setpts={expression}',
               '-fps_mode:v', 'passthrough', '-enc_time_base:v', f'1:{CLIP_TIMESCALE}',
               '-bsf:v', f'setts=duration={last_ticks}',
               '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '22', '-pix_fmt', 'yuv420p',
               '-video_track_timescale', str(CLIP_TIMESCALE), '-movflags', '+faststart', str(destination)]
    try:
        completed = subprocess.run(command, input=b''.join(frame.tobytes() for frame in frames),
                                   capture_output=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        Path(destination).unlink(missing_ok=True)
        raise ValueError('击球片段编码超时，片段未发布') from error
    if completed.returncode or not Path(destination).is_file():
        raise ValueError('击球片段按源时间编码失败')
    capture = cv2.VideoCapture(str(destination))
    output_pts = []
    try:
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            output_pts.append(float(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000))
        average_fps = capture.get(cv2.CAP_PROP_FPS)
        container_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        capture.release()
    relative_pts = [tick / CLIP_TIMESCALE for tick in ticks]
    expected_duration = (ticks[-1] + last_ticks) / CLIP_TIMESCALE
    duration = container_frames / average_fps if average_fps > 0 else math.nan
    if len(output_pts) != len(frames) or any(abs(a-b) > PTS_TOLERANCE for a,b in zip(output_pts, relative_pts)):
        Path(destination).unlink(missing_ok=True)
        raise ValueError('击球片段编码后的帧数或时间未对齐，片段未发布')
    if not math.isfinite(duration) or abs(duration - expected_duration) > PTS_TOLERANCE:
        Path(destination).unlink(missing_ok=True)
        raise ValueError('击球片段编码后的源时长未对齐，片段未发布')
    return {'clip_relative_pts': output_pts, 'clip_frame_count': len(output_pts),
            'clip_duration': duration, 'clip_timescale': CLIP_TIMESCALE,
            'encoded_last_frame_duration': last_ticks / CLIP_TIMESCALE,
            'pts_policy': 'verified_source_pts_relative_to_window_start',
            'validation': 'decoded_output_frame_count_pts_and_stream_duration'}
