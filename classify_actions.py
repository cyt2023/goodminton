"""Offline visual classification of proposed action clips; never a quality score.

Run only with .venv-vision. Prompts receive images, not the user's complaint or
requested technique, so a search request cannot supply its own classification.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import time

from PIL import Image, ImageDraw
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
import mlx.core as mx
from vision_model import MODEL, MODEL_ID, MODEL_REVISION

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / '03_数据'
PROMPT_VERSION = 'badminton-independent-stroke-style-v3-video'
STROKES = {'smash', 'clear', 'drop', 'net', 'drive', 'lift', 'serve', 'push', 'defense', 'unknown', 'no_stroke'}
STYLES = {'forehand', 'backhand', 'unknown'}
PROMPT = '''这是羽毛球录像中的一个短视频。只分析蓝色框内的球员。
先描述可见的挥拍过程，再判断球种和正反手。单纯抬手、跑动、架拍不是击球；看不到挥拍就不要猜球种。杀球需要下压击球的证据，不能由架拍判断。左手持拍不等于反手；绕头正手不等于反手。看不清就选 unknown、uncertain。
只输出一个 JSON 对象，必须含以下八个键，不要输出示例或代码围栏：
stroke_evidence：用中文描述挥拍变化，以及球种依据或看不清的原因。
style_evidence：用中文描述持拍侧、拍面与挥拍方向，以及正反手依据或看不清的原因。
visible_stroke：布尔值 true 或 false，表示是否看见挥拍击球动作；无需确认球拍与球接触的精确瞬间。
stroke_type：选一个英文值 unknown、no_stroke、serve、net、lift、drive、clear、drop、push、defense、smash。
swing_style：选一个英文值 unknown、forehand、backhand。
stroke_certainty 和 style_certainty：各选 likely 或 uncertain，描述必须支持所选标签。
quality：画面可用于判断选 usable，否则选 unclear。'''


def data_file(value):
    file = Path(value).resolve()
    if not file.is_relative_to(DATA.resolve()):
        raise ValueError('Only registered project data may be read or written')
    return file


def sha256(file):
    digest = hashlib.sha256()
    with file.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(file, value):
    file = data_file(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(file)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse_prediction(text):
    start = text.find('{')
    if start < 0:
        raise ValueError('Vision response did not contain JSON')
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError('Vision response was not an object')
    repairs = []
    # A real pilot returned a pluralized key with a proper JSON boolean.
    # Accept only this spelling alias; never infer a decision from a shot name.
    if 'visible_strokes' in value:
        if type(value['visible_strokes']) is not bool:
            raise ValueError('Invalid visible stroke alias')
        if 'visible_stroke' in value:
            if type(value['visible_stroke']) is not bool or value['visible_stroke'] != value['visible_strokes']:
                raise ValueError('Conflicting visible stroke decisions')
        else:
            value['visible_stroke'] = value['visible_strokes']
            repairs.append('visible_strokes_boolean_alias')
    output = {key: value.get(key) for key in ['visible_stroke', 'stroke_type', 'stroke_certainty',
              'swing_style', 'style_certainty', 'quality', 'stroke_evidence', 'style_evidence']}
    # A refusal word can only become unknown, never a positive technique label.
    for key in ['stroke_type', 'swing_style']:
        if output[key] in ['uncertain', 'unclear']:
            output[key] = 'unknown'
    if type(output['visible_stroke']) is not bool:
        raise ValueError('Missing visible stroke decision')
    if output['stroke_type'] not in STROKES or output['swing_style'] not in STYLES:
        raise ValueError('Unknown classification vocabulary')
    if output['stroke_certainty'] not in ['likely', 'uncertain'] or output['style_certainty'] not in ['likely', 'uncertain']:
        raise ValueError('Missing uncertainty decision')
    if output['quality'] not in ['usable', 'unclear']:
        raise ValueError('Missing image quality decision')
    for key in ['stroke_evidence', 'style_evidence']:
        if not isinstance(output[key], str):
            raise ValueError('Missing visual evidence')
        output[key] = output[key][:1200]
    if output['stroke_type'] == 'no_stroke':
        output['visible_stroke'] = False
    if not output['visible_stroke'] or output['quality'] != 'usable':
        output.update(stroke_certainty='uncertain', style_certainty='uncertain')
    if output['stroke_type'] in ['unknown', 'no_stroke'] or not output['stroke_evidence'].strip():
        output['stroke_certainty'] = 'uncertain'
    if output['swing_style'] == 'unknown' or not output['style_evidence'].strip():
        output['style_certainty'] = 'uncertain'
    # Reject internally inconsistent labels; evidence words never assign a label.
    opposite = {'forehand': r'backhand|反手|反拍', 'backhand': r'forehand|正手|正拍'}
    if output['swing_style'] in opposite:
        for clause in re.split(r'[。；;.!?]|但|however', output['style_evidence'], flags=re.I):
            if (re.search(opposite[output['swing_style']], clause, re.I) and
                    not re.search(r'不|非|无法|不能|未知|不同|区别|\bnot\b|cannot|unclear|uncertain', clause, re.I)):
                output['style_certainty'] = 'uncertain'
                output['consistency_note'] = 'Swing-style evidence asserts the opposite style; candidate withheld.'
                break
    output['schema_repairs'] = repairs
    return output


def input_frames(configuration, pose, event, folder):
    source = data_file(configuration['source_file'])
    start, end, peak = event['start'], event['end'], event['peak_time']
    if not all(finite(v) for v in [start, end, peak]) or not pose['start'] <= start < end <= pose['end'] + .001:
        raise ValueError('Invalid proposed clip interval')
    # Uniform real frames preserve playback cadence for native video input.
    last = max(start, end - min(.08, (end - start) / 4))
    times = [round(start + i * (last - start) / 11, 4) for i in range(12)]
    observations = [frame for frame in pose['frames'] if start - .3 <= frame['source_time'] <= end + .3
                    and frame.get('players', {}).get(event['player'])]
    if not observations:
        raise ValueError('No target-player box for proposed action')
    boxes = [frame['players'][event['player']]['bbox'] for frame in observations]
    left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
    right, bottom = max(b[2] for b in boxes), max(b[3] for b in boxes)
    width, height = pose['source_width'], pose['source_height']
    pad_x, pad_y = max(30, (right - left) * .65), max(30, (bottom - top) * .35)
    crop_box = (max(0, int(left - pad_x)), max(0, int(top - pad_y)),
                min(width, math.ceil(right + pad_x)), min(height, math.ceil(bottom + pad_y)))
    paths, evidence = [], []
    for index, timestamp in enumerate(times):
        command = [configuration['ffmpeg'], '-hide_banner', '-loglevel', 'error', '-ss', str(timestamp),
                   '-i', str(source), '-map', '0:v:0', '-frames:v', '1', '-an',
                   '-f', 'image2pipe', '-vcodec', 'png', 'pipe:1']
        response = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        if response.returncode or not response.stdout:
            continue
        with Image.open(io.BytesIO(response.stdout)) as image:
            image = image.convert('RGB')
            if image.size != (width, height):
                raise ValueError('Source display dimensions changed')
            nearest = min(observations, key=lambda frame: abs(frame['source_time'] - timestamp))
            if abs(nearest['source_time'] - timestamp) > .25:
                raise ValueError('Target-player tracking is absent within the proposed clip')
            bbox = nearest['players'][event['player']]['bbox']
            draw = ImageDraw.Draw(image)
            draw.rectangle(tuple(bbox), outline='#24b6e2', width=max(2, width // 360))
            image = image.crop(crop_box)
            image.thumbnail((448, 448))
            file = folder / f'frame_{index:02d}.jpg'
            image.save(file, quality=92)
        paths.append(str(file))
        evidence.append({'file': str(file.relative_to(DATA)), 'source_time_nominal': timestamp,
                         'target_bbox_time': nearest['source_time'], 'target_bbox_original_px': bbox,
                         'crop_box_original_px': list(crop_box), 'time_note': 'FFmpeg seek time, not a saved original frame PTS'})
    if len(paths) < 4:
        raise ValueError('Too few decoded frames to classify a motion')
    return paths, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    configuration = json.loads(data_file(args.config).read_text())
    classifier_sha256 = sha256(Path(__file__))
    if configuration.get('classifier_sha256') and classifier_sha256 != configuration['classifier_sha256']:
        raise ValueError('Classifier implementation changed before execution')
    pose = json.loads(data_file(configuration['pose_file']).read_text())
    source = data_file(configuration['source_file'])
    if sha256(source) != configuration['source_sha256'] or sha256(data_file(configuration['pose_file'])) != configuration['pose_fingerprint']:
        raise ValueError('The source video or pose result changed')
    output = data_file(configuration['output_folder'])
    output.mkdir(parents=True, exist_ok=True)
    model, processor = load(str(MODEL), trust_remote_code=False, local_files_only=True)
    print('CLASSIFIER_READY ' + json.dumps({'model': str(MODEL), 'revision': MODEL_REVISION}), flush=True)
    events = []
    for proposal in configuration['candidates']:
        started = time.perf_counter()
        folder = data_file(output / proposal['id'])
        folder.mkdir(parents=True, exist_ok=True)
        event = dict(proposal, classification_status='model_candidate', review_status='unreviewed',
                     classification_source='local_vision_model', model_id=MODEL_ID, model_revision=MODEL_REVISION,
                     prompt_version=PROMPT_VERSION, prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
                     classifier_sha256=classifier_sha256, created_at=datetime.now(timezone.utc).isoformat())
        raw_text = ''
        try:
            paths, evidence = input_frames(configuration, pose, proposal, folder)
            event['evidence_frames'] = evidence
            clip_frames = [Image.open(path).convert('RGB') for path in paths]
            video_fps = (len(paths) - 1) / (evidence[-1]['source_time_nominal'] - evidence[0]['source_time_nominal'])
            prompt = apply_chat_template(processor, model.config, prompt=PROMPT,
                                         video='local-action-clip', fps=video_fps)
            generated = generate(model, processor, prompt, video=[clip_frames], fps=video_fps,
                                 max_tokens=384, temperature=0.0, verbose=False)
            raw_text = generated.text
            event.update(prompt_tokens=getattr(generated, 'prompt_tokens', None),
                         generation_tokens=getattr(generated, 'generation_tokens', None),
                         generation_finish_reason=getattr(generated, 'finish_reason', None),
                         model_peak_memory_gb=getattr(generated, 'peak_memory', None))
            prediction = parse_prediction(raw_text)
            event.update(prediction, evidence_frames=evidence, vision_input_mode='native_video',
                         nominal_video_fps=round(video_fps, 6),
                         prompt_tokens=getattr(generated, 'prompt_tokens', None),
                         generation_tokens=getattr(generated, 'generation_tokens', None))
            preview = Image.open(paths[len(paths) // 2])
            preview.save(folder / 'thumbnail.jpg', quality=90)
            event['thumbnail_file'] = str((folder / 'thumbnail.jpg').relative_to(DATA))
        except Exception as error:
            event.update(visible_stroke=False, stroke_type='unknown', swing_style='unknown',
                         stroke_certainty='uncertain', style_certainty='uncertain', quality='unclear',
                         stroke_evidence='', style_evidence='', classification_error=type(error).__name__ + ': ' + str(error)[:700])
        event['runtime_seconds'] = round(time.perf_counter() - started, 3)
        save(folder / 'prediction.json', dict(event, raw_response=raw_text, prompt=PROMPT))
        events.append(event)
        print('EVENT ' + json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)
        mx.clear_cache()
    report = {'pose_id': pose['id'], 'source_video_id': pose['source_video_id'],
              'source_sha256': configuration['source_sha256'], 'pose_fingerprint': configuration['pose_fingerprint'],
              'model_revision': MODEL_REVISION, 'prompt_version': PROMPT_VERSION,
              'classifier_sha256': classifier_sha256, 'events': events}
    save(output / 'classified_events.json', report)
    print('DONE ' + json.dumps({'pose_id': pose['id'], 'processed': len(events)}), flush=True)


if __name__ == '__main__':
    main()
