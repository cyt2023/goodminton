"""Local specialist indexing: TrackNet -> dense pose -> hit proposals -> BST-0.

The generic VLM supplies only visible-swing and fore/backhand candidates; it
does not assign the ball stroke class. Pretrained predictions remain unreviewed.
Each upstream output, adaptation and rejected candidate is persisted.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from analysis_jobs import atomic_json, data_path, file_hash
from court_calibration import load_reviewed, inspect_lines
from specialist_events import proposals
from specialist_model import ROOT, ENGINE_VERSION, MODEL_ID, MODEL_REVISION, CLASSIFICATION_SOURCE, MANIFEST
from vision_model import MODEL, MODEL_REVISION as VLM_REVISION
from stroke_taxonomy import AUTHOR_MAP, VERSION as TAXONOMY_VERSION

STYLE_PROMPT = '''这是羽毛球录像的连续原始画面，只分析蓝框中的同一位球员。
请判断是否看见挥拍击球，以及正手还是反手。左手持拍不等于反手，绕头正手不等于反手，抬臂准备或回位不等于击球。不能依据单张姿势或对方球员判断。画面不足就输出 unknown，不要猜。
只输出一个简短 JSON：visible_stroke（true/false）；swing_style（forehand/backhand/unknown）；style_certainty（likely/uncertain）；quality（usable/unclear）；style_evidence（中文，最多50字，描述持拍侧和连续挥拍依据）。不要输出球种或教学建议。'''
MAPPING = AUTHOR_MAP


def withholding_counts(events):
    """Do not count a deliberately skipped style stage as failed BST input."""
    gate = style = unknown = 0
    for event in events:
        category = event.get('withheld_category')
        reason = event.get('withheld_reason', '')
        # Recognize saved predictions produced before categories were persisted.
        if category == 'input_or_target_gate' or reason.startswith('Insufficient paired-player/observed-shuttle coverage'):
            gate += 1
        elif category == 'style_review_not_run' or reason.startswith('BST-only developer run;'):
            style += 1
        elif reason:
            unknown += 1
    return {'withheld_by_input_or_target_gate':gate,
            'style_review_not_run':style, 'withheld_category_unknown':unknown,
            'classification_error_count':sum(bool(e.get('classification_error')) for e in events),
            'legacy_any_withheld_reason_count':sum(bool(e.get('withheld_reason')) for e in events),
            'withholding_count_version':'separate-input-gate-and-style-v2'}


def status(message, **extra):
    print('STATUS ' + json.dumps({'message': message, **extra}, ensure_ascii=False), flush=True)


def run_stage(script, python, config_file, log, label):
    process = subprocess.Popen([str(python), '-u', str(ROOT / script), '--config', str(config_file)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
    with log.open('a') as stream:
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            if '_PROGRESS ' in line:
                value = json.loads(line.split('_PROGRESS ', 1)[1])
                status(label, stage_processed=value['processed'], stage_total=value['total'])
            elif line.startswith('Traceback'):
                status(label + '遇到运行错误，正在保存记录')
    if process.wait() != 0:
        raise RuntimeError(script + ' failed; see ' + str(log))


def valid_cached(path, config, extra=None):
    try:
        value = json.loads(path.read_text())
        if value['source_sha256'] != config['source_sha256']:
            return None
        if extra and any(value.get(k) != v for k, v in extra.items()):
            return None
        return value
    except (OSError, ValueError, KeyError):
        return None


def input_images(source, dense, trajectory, event, folder):
    # Consecutive original frames: approximately .6 seconds either side of the
    # proposed hit. Their boxes are estimated on exactly the same decoded frame.
    fps, peak = dense['sample_fps'], event['frame_index']
    lo, hi = max(event['start_frame'], peak-round(.6*fps)), min(event['end_frame'], peak+round(.6*fps))
    indices = list(range(lo, hi + 1))
    if len(indices) < 8:
        raise ValueError('Too few consecutive source frames')
    people = [dense['frames'][i]['players'].get(event['player']) for i in indices]
    if sum(p is not None for p in people) / len(people) < .9:
        raise ValueError('Target player missing from consecutive hit frames')
    boxes = [p['bbox'] for p in people if p]
    left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
    right, bottom = max(b[2] for b in boxes), max(b[3] for b in boxes)
    pad_x, pad_y = max(50, (right-left)*.7), max(45, (bottom-top)*.5)
    crop = [max(0, int(left-pad_x)), max(0, int(top-pad_y)),
            min(dense['source_width'], int(right+pad_x)), min(dense['source_height'], int(bottom+pad_y))]
    capture = cv2.VideoCapture(str(source))
    capture.set(cv2.CAP_PROP_POS_FRAMES, lo)
    images, evidence = [], []
    for i, person in zip(indices, people):
        ok, frame = capture.read()
        if not ok:
            raise ValueError('Consecutive source frame decode failed')
        # Preserve a frame even if the target box is missing; do not duplicate a
        # previous pose or silently change frame cadence.
        if person:
            b = person['bbox']
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (235, 180, 25), 2)
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).crop(crop)
        im.thumbnail((576, 576))
        filename = folder / f'frame_{i-lo:03d}.jpg'
        im.save(filename, quality=92)
        images.append(im)
        evidence.append({'file': str(filename.relative_to(ROOT.parent / '03_数据')),
                         'original_frame_index': i, 'source_time': dense['frames'][i]['source_time'],
                         'target_bbox': person['bbox'] if person else None, 'crop_box': crop,
                         'alignment': 'pose and decoded image share original frame index'})
    capture.release()
    images[min(len(images)-1, max(0, peak-lo))].save(folder / 'thumbnail.jpg', quality=92)
    sample = np.linspace(0, len(images)-1, 12).round().astype(int)
    sheet = Image.new('RGB', (960, 690), 'white')
    for n, k in enumerate(sample):
        im = images[int(k)].copy()
        im.thumbnail((240, 200))
        x, y = n % 4 * 240, n // 4 * 230
        sheet.paste(im, (x, y + 25))
        ImageDraw.Draw(sheet).text((x+4, y+4), f'{evidence[int(k)]["source_time"]:.3f}s f{evidence[int(k)]["original_frame_index"]}', fill='black')
    sheet.save(folder / 'sequence_review.jpg', quality=92)
    return images, evidence


def parse_style(text, finish_reason):
    if finish_reason == 'length':
        raise ValueError('Style model output reached the generation length limit')
    start = text.find('{')
    if start < 0:
        raise ValueError('Style model returned no JSON')
    v, _ = json.JSONDecoder().raw_decode(text[start:])
    if (type(v.get('visible_stroke')) is not bool or v.get('swing_style') not in ['forehand', 'backhand', 'unknown'] or
            v.get('style_certainty') not in ['likely', 'uncertain'] or v.get('quality') not in ['usable', 'unclear'] or
            not isinstance(v.get('style_evidence'), str)):
        raise ValueError('Style model returned invalid fields')
    if not v['visible_stroke'] or v['quality'] != 'usable' or not v['style_evidence'].strip() or v['swing_style'] == 'unknown':
        v['style_certainty'] = 'uncertain'
    return {k: v[k] for k in ['visible_stroke', 'swing_style', 'style_certainty', 'quality', 'style_evidence']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--bst-only', action='store_true', help='Persist specialist predictions without VLM style decisions')
    args = parser.parse_args()
    config_file = data_path(args.config)
    config = json.loads(config_file.read_text())
    source = data_path(config['source_file'])
    pose_file = data_path(config['pose_file'])
    if file_hash(source) != config['source_sha256'] or file_hash(pose_file) != config['pose_fingerprint']:
        raise ValueError('Source video or coarse pose changed')
    if config.get('classifier_sha256') and file_hash(Path(__file__)) != config['classifier_sha256']:
        raise ValueError('Pipeline code changed before execution')
    folder = data_path(config.get('specialist_folder', ROOT.parent/'03_数据/专项分析'/config['pose_id']))
    folder.mkdir(parents=True, exist_ok=True)
    config['specialist_folder'] = str(folder)
    atomic_json(folder / 'config.json', config)
    started = time.perf_counter()
    status('正在按所选颜色提取场地线候选')
    if config.get('court_calibration'):
        record = config['court_calibration']
        if record.get('source_sha256') != config['source_sha256'] or record.get('review_status') != 'confirmed':
            raise ValueError('场地标定与当前录像不匹配')
        atomic_json(folder/'court_calibration.json', record)
    calibration = load_reviewed(folder/'court_calibration.json', config['source_sha256'])
    court = inspect_lines(source, folder, config['source_sha256'], config.get('court_line_color', 'unknown'))
    from research_adapters import settings as research_settings
    if research_settings()['perception_backend'] == 'racketvision' and not config.get('benchmark_only'):
        status('正在用 RacketVision 跟踪球与球拍')
        run_stage('racketvision_adapter.py', ROOT/'.venv-vision/bin/python', folder/'config.json', folder/'racketvision.log', '正在跟踪球和球拍')
        import shutil
        if (folder/'trajectory.json').exists() and not (folder/'baseline_tracknet/trajectory.json').exists():
            previous=json.loads((folder/'trajectory.json').read_text())
            if previous.get('engine_version')!='racketvision-balltrack-native-v1':
                (folder/'baseline_tracknet').mkdir(exist_ok=True)
                shutil.copy2(folder/'trajectory.json',folder/'baseline_tracknet/trajectory.json')
        shutil.copy2(folder/'racketvision/trajectory.json',folder/'trajectory.json')
        trajectory=json.loads((folder/'trajectory.json').read_text())
    else:
        trajectory = valid_cached(folder/'trajectory.json', config, {'code_sha256': file_hash(ROOT/'specialist_tracknet.py')})
        if trajectory is None:
            status('正在跟踪羽毛球')
            run_stage('specialist_tracknet.py', ROOT/'.venv-vision/bin/python', folder/'config.json', folder/'tracknet.log', '正在跟踪羽毛球')
            trajectory = json.loads((folder/'trajectory.json').read_text())
    dense = valid_cached(folder/'dense_pose.json', config, {'coarse_pose_sha256': config['pose_fingerprint'], 'code_sha256': file_hash(ROOT/'specialist_pose.py')})
    if dense is None:
        status('正在逐帧分析两位球员')
        run_stage('specialist_pose.py', ROOT/'.venv-pose/bin/python', folder/'config.json', folder/'dense_pose.log', '正在逐帧分析两位球员')
        dense = json.loads((folder/'dense_pose.json').read_text())
    if calibration and calibration['record'].get('image_size', [dense['source_width'],dense['source_height']]) != [dense['source_width'],dense['source_height']]:
        raise ValueError('场地标定的画面尺寸与识别画面不一致，请重新标定')
    status('正在结合球轨迹寻找击球片段')
    candidates = proposals(dense, trajectory)
    atomic_json(folder/'hit_candidates.json', {'source_sha256': config['source_sha256'], 'candidates': candidates,
                'dense_pose_sha256': file_hash(folder/'dense_pose.json'), 'trajectory_sha256': file_hash(folder/'trajectory.json')})
    status('正在结合击球声音与专项 GRU 核对候选位置')
    from hit_refinement import prepare_inputs
    refinement=prepare_inputs(config,dense,trajectory,candidates)
    # Keep original event IDs and full context; a refined center is a proposal.
    original_candidates=candidates
    candidates=[]
    for original in original_candidates:
        candidate=dict(original);suggestion=refinement['suggestions'].get(original['id'])
        if suggestion and suggestion['applied']:
            candidate.update(original_peak_time=original['peak_time'],original_frame_index=original['frame_index'],
                             peak_time=suggestion['suggested_time'],frame_index=suggestion['suggested_frame'],
                             center_suggestion=suggestion,contact_confirmed=False)
        candidates.append(candidate)
    for extra in refinement['supplemental_candidates']:
        # The classifier needs flight context; the review page exports 21 frames.
        i=extra['frame_index'];times=np.asarray([f['source_time'] for f in dense['frames']])
        lo=int(np.searchsorted(times,max(0.,extra['peak_time']-1.)))
        hi=min(len(times)-1,int(np.searchsorted(times,extra['peak_time']+1.)))
        candidates.append({**extra,'start_frame':lo,'end_frame':hi,'start':float(times[lo]),
                           'end':min(dense['end'],float(times[hi]+np.median(np.diff(times))))})
    sides = config.get('requested_sides', ['near', 'far'])
    done_ids = set(config.get('done_ids', []))
    remaining = [c for c in candidates if c['player'] in sides and c['id'] not in done_ids]
    print('CANDIDATES ' + json.dumps({'candidates': candidates, 'remaining': remaining}, ensure_ascii=False), flush=True)
    from specialist_bst import SpecialistBST
    classifier = SpecialistBST(dense, trajectory, calibration)
    output = data_path(config['output_folder'])
    output.mkdir(parents=True, exist_ok=True)
    snapshots = folder/'执行代码快照'/file_hash(Path(__file__))[:16]
    snapshots.mkdir(parents=True, exist_ok=True)
    import shutil
    for module in ['specialist_pipeline.py', 'specialist_pose.py', 'specialist_tracknet.py', 'specialist_bst.py',
                   'specialist_events.py', 'specialist_model.py', 'court_calibration.py', 'court_settings.py', 'reference_alignment.py', 'build_specialist_report.py', 'stroke_taxonomy.py',
                   'hit_refinement.py','hit_gru.py','stroke_features.py','personal_transfer.py','personal_head.py','personal_learning.py',
                   'stroke_context.py','preserved_bst.py','bst_public_data.py','teacher_adaptation.py',
                   'public_bst_training.py','public_bst_review.py']:
        shutil.copy2(ROOT/module, snapshots/module)
    for module in ['racketvision_adapter.py','racketvision_runtime.py','research_adapters.py','research_report.py',
                   'research_quality.py','stroke_event_fusion.py','research_playback.py']:
        shutil.copy2(ROOT/module,snapshots/module)
    shutil.copytree(ROOT/'research_models',snapshots/'research_models',dirs_exist_ok=True)
    style_model = style_processor = None
    events = []
    rejected = 0
    for n, candidate in enumerate(remaining):
        t0 = time.perf_counter()
        directory = output/candidate['id']
        directory.mkdir(parents=True, exist_ok=True)
        event = dict(candidate, model_id=MODEL_ID, model_revision=MODEL_REVISION,
                     engine_version=ENGINE_VERSION, classification_source=CLASSIFICATION_SOURCE,
                     classification_status='model_candidate', quality='unclear', visible_stroke=False,
                     stroke_type='unknown', stroke_certainty='uncertain', swing_style='unknown', style_certainty='uncertain',
                     stroke_evidence='', style_evidence='', created_at=datetime.now(timezone.utc).isoformat())
        raw_style = ''
        try:
            values, input_quality = classifier.inputs(candidate)
            event['input_quality'] = input_quality
            j, p, s, length = values
            np.savez_compressed(directory/'bst_inputs.npz', joints_and_bones=j, positions=p, shuttle=s, video_len=length)
            result = classifier.predict(values)
            event['specialist_predictions'] = result
            top = result['bst_full'][0] if result['bst_full'] is not None else result['bst0'][0]
            prediction_label = top['label']
            person_label, separator, stroke_label = prediction_label.partition('_')
            expected_player = 'Top' if candidate['player'] == classifier.order[0] else 'Bottom'
            stroke = MAPPING.get(stroke_label if separator else prediction_label, 'unknown')
            event.update(stroke_type=stroke, source_model_label=prediction_label, taxonomy_version=TAXONOMY_VERSION,
                         stroke_evidence=f'BST 专项分类候选：{prediction_label}；分数未经校准，球种尚未人工核对。',
                         classifier_player_agreement=(person_label == expected_player),
                         bst_softmax_score=top['softmax_score'],
                         classifier_name='BST-CG-AP' if classifier.full_model is not None else 'BST-0')
            images, evidence = input_images(source, dense, trajectory, candidate, directory)
            event.update(evidence_frames=evidence, thumbnail_file=str((directory/'thumbnail.jpg').relative_to(ROOT.parent/'03_数据')))
            qualified = (input_quality['paired_player_fraction'] >= .85 and input_quality['raw_shuttle_fraction'] >= .4 and
                         person_label == expected_player and stroke != 'unknown' and top['softmax_score'] >= .4)
            event['input_target_gate_passed'] = qualified
            if qualified and not args.bst_only:
                if style_model is None:
                    from mlx_vlm import load
                    style_model, style_processor = load(str(MODEL), trust_remote_code=False, local_files_only=True)
                from mlx_vlm import generate
                from mlx_vlm.prompt_utils import apply_chat_template
                prompt = apply_chat_template(style_processor, style_model.config, prompt=STYLE_PROMPT,
                                             video='original-hit-frames', fps=dense['sample_fps'])
                generated = generate(style_model, style_processor, prompt, video=[images], fps=dense['sample_fps'],
                                     max_tokens=512, temperature=0, verbose=False)
                raw_style = generated.text
                event.update(style_model_revision=VLM_REVISION, generation_tokens=generated.generation_tokens,
                             generation_finish_reason=getattr(generated, 'finish_reason', None))
                style = parse_style(raw_style, getattr(generated, 'finish_reason', None))
                event.update(style)
                if style['visible_stroke'] and style['quality'] == 'usable':
                    event['stroke_certainty'] = 'likely'
            elif qualified:
                event['withheld_reason'] = 'BST-only developer run; no visible-swing confirmation or fore/backhand decision'
                event['withheld_category'] = 'style_review_not_run'
            else:
                rejected += 1
                event['withheld_reason'] = 'Insufficient paired-player/observed-shuttle coverage, uncertain class or classifier target disagreement'
                event['withheld_category'] = 'input_or_target_gate'
        except Exception as error:
            event.update(classification_error=type(error).__name__ + ': ' + str(error)[:700])
        event['runtime_seconds'] = round(time.perf_counter() - t0, 3)
        atomic_json(directory/'prediction.json', dict(event, raw_style_response=raw_style, style_prompt=STYLE_PROMPT))
        events.append(event)
        print('EVENT ' + json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)
        status('正在辨认击球类型和正反手', stage_processed=n+1, stage_total=len(remaining))
    methods = {'engine_version': ENGINE_VERSION, 'model_revision': MODEL_REVISION,
               'perception_backend': trajectory['engine_version'],
               'style_vlm_run': not args.bst_only,
               'source_sha256': config['source_sha256'], 'coarse_pose_sha256': config['pose_fingerprint'],
               'dense_pose_sha256': file_hash(folder/'dense_pose.json'), 'trajectory_sha256': file_hash(folder/'trajectory.json'),
               'candidate_count': len(candidates), 'processed': len(events), 'withheld_by_input_or_target_gate': rejected,
               'hit_refinement_version':refinement['version'],'hit_refinement_sha256':file_hash(folder/'hit_refinement.json'),
               'hit_refinement_summary':refinement['summary'],
               'court_calibration_used': calibration is not None, 'court_evidence_status': court['status'],
               'court_line_color': court['court_line_color'],
               'classifier': 'BST-CG-AP' if calibration is not None else 'BST-0',
               'runtime_seconds_this_invocation': round(time.perf_counter()-started, 3),
               'method_manifest': json.loads(MANIFEST.read_text()),
               'scope': 'Author pretrained inference with documented local input adaptations; no paper metric reproduction or verified coaching diagnosis.'}
    # Expansion to the other player preserves earlier predictions from this
    # same source/model generation. Old baseline event directories are ignored.
    combined = {}
    for candidate in candidates:
        saved = output/candidate['id']/'prediction.json'
        if saved.is_file():
            previous = json.loads(saved.read_text())
            if previous.get('model_revision') == MODEL_REVISION and previous.get('engine_version') == ENGINE_VERSION:
                combined[candidate['id']] = {k:v for k,v in previous.items() if k not in ['raw_style_response','style_prompt']}
    for event in events:
        combined[event['id']] = event
    methods.update(processed_this_invocation=len(events), processed=len(combined), requested_sides=sides,
                   runtime_seconds_events_cumulative=round(sum(e.get('runtime_seconds', 0) for e in combined.values()), 3),
                   execution_snapshot=str(snapshots.relative_to(ROOT.parent/'03_数据')))
    methods.update(withholding_counts(list(combined.values())))
    atomic_json(folder/'methods.json', methods)
    atomic_json(output/'classified_events.json', {'pose_id': config['pose_id'], 'events': sorted(combined.values(), key=lambda e:e['peak_time']), 'methods': methods})
    if config.get('benchmark_only'):
        # A reference-video check uses only the original pretrained classifier.
        # Do not apply personal heads or create public continuation predictions.
        atomic_json(folder/'report.json', {
            'job_id':config['pose_id'], 'source_video_id':dense['source_video_id'],
            'source_sha256':config['source_sha256'], 'source_duration':dense['end'],
            'methods':methods, 'status':'completed', 'events':[],
            'scope':'Original pretrained BST on local frontend outputs; reference-video development check, no verified labels or accuracy yet.'})
        print('BENCHMARK_DONE '+json.dumps({'job_id':config['pose_id'],'events':len(combined),'training':False}),flush=True)
        return
    from build_specialist_report import build
    build(config['pose_id'])
    status('正在融合击球事件与可解释测量')
    from research_report import build as build_research
    build_research(config['pose_id'])
    try:
        from preserved_bst import prepare as preserve_full_context
        status('正在保留预训练分类层并整理完整击球范围')
        preserve_full_context(config['pose_id'])
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        atomic_json(folder/'preserved_bst_error.json', {'source_sha256': config['source_sha256'],
                    'error': str(error)[:700], 'original_classification_preserved': True})
    # Additional unverified hints; the official classifier result is retained.
    try:
        from public_bst_review import OUTPUT as public_output, freeze as freeze_public
        if (public_output/'latest.json').is_file():
            status('正在冻结公开数据模型的新旧预测，准备人工检查')
            freeze_public(config['pose_id'])
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        atomic_json(folder/'public_model_error.json', {'source_sha256': config['source_sha256'],
                    'error': str(error)[:700], 'original_classification_preserved': True})
    try:
        from personal_transfer import replay_latest
        status('正在复用已有人工标注分类头，生成跨录像候选')
        replay_latest(config['pose_id'])
    except (OSError,ValueError,KeyError,RuntimeError) as error:
        atomic_json(folder/'personal_transfer_error.json',{'source_sha256':config['source_sha256'],
            'stage':'optional_cross_video_hints','error':str(error)[:700],
            'specialist_classification_preserved':True})
        print('TRANSFER_HINTS_UNAVAILABLE '+str(error)[:500],flush=True)
    print('DONE ' + json.dumps({'pose_id': config['pose_id'], 'processed': len(events), 'engine_version': ENGINE_VERSION}), flush=True)


if __name__ == '__main__':
    main()
