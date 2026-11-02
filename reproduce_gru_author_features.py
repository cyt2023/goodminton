"""Replay published shuttlepose weights on all released singles test features.

This tests the author's published feature condition, including human ball and
human-associated pose information. It does not evaluate an automatic frontend.
Original interval metrics are executed from the author source, including quirks.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd

from analysis_jobs import atomic_json, file_hash
from benchmark_hits_gru import maximise_hit_score
from benchmark_hit_models import match_events
from hit_gru import predict_windows

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARCHIVE = PROJECT / '03_数据/找拍模型对照/2026-10-22/extension_assets/domain.zip'
AUTHOR = ROOT / 'vendor/Badminton-hit-detection'
WEIGHTS = ROOT / 'models/hit_gru/keras_tensors.npz'
OUT = PROJECT / '03_数据/GRU作者流程复现/2026-10-25/author_features'
POSE_COLUMNS = [f'{side}_pose{joint}_{axis}' for side in ['near', 'far']
                for joint in range(17) for axis in ['x', 'y']]
COLUMNS = ['ball_x', 'ball_y'] + POSE_COLUMNS
IOU_THRESHOLDS = [round(i * .05, 2) for i in range(15)]


def author_functions():
    """Execute only the published pure postprocessing/evaluation definitions."""
    names = {'optimise_hits_naive', 'expand_label_frame_range', 'preds2seframes',
             'timewin_intersection_over_union', 'calc_metrics', 'eval_vid_metrics',
             'calc_overall_metrics'}
    tree = ast.parse((AUTHOR / 'process_pred_probs.source.txt').read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == names
    namespace = dict(np=np, pd=pd, math=math, FPS=30, FRAME_RANGE=3, IOU_THRESH=.05)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'author_postprocess', 'exec'), namespace)
    return namespace


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        return json.loads(path.read_text())
    rows = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith('_x.csv'):
                continue
            pieces = Path(name).parts
            domain = pieces[1]
            if domain == 'domain_pro' and pieces[2].startswith('test_match'):
                group = 'professional_author_test'
            elif domain == 'domain_am_singles':
                group = 'amateur_singles_author_released'
            else:
                continue
            partner = name[:-6] + '_y.csv'
            assert partner in archive.namelist(), partner
            for member in [name, partner]:
                target = OUT / 'inputs' / member
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
            xfile, yfile = [OUT / 'inputs' / p for p in [name, partner]]
            x, y = pd.read_csv(xfile), pd.read_csv(yfile)
            assert len(x) == len(y)
            assert np.array_equal(x['Frame'], y['frame'])
            assert np.array_equal(x['Frame'], np.arange(len(x)))
            features = x[COLUMNS].to_numpy(dtype=np.float32)
            assert np.isfinite(features).all()
            assert set(y['player_hit']).issubset({0, 1, 2})
            rows.append({'id': group + '_' + pieces[2] + '_' + Path(name).stem[:-2],
                'group': group, 'match': pieces[2], 'x_path': str(xfile), 'y_path': str(yfile),
                'frames': len(x), 'contacts': int((y['player_hit'] > 0).sum()),
                'input_sha256': {str(p): file_hash(p) for p in [xfile, yfile]}})
    counts = {group: {'rallies': sum(r['group'] == group for r in rows),
                      'frames': sum(r['frames'] for r in rows if r['group'] == group),
                      'contacts': sum(r['contacts'] for r in rows if r['group'] == group)}
              for group in sorted({r['group'] for r in rows})}
    protocol = {'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'All released professional test_match1..3 and all released amateur-singles X/Y pairs. Ordinary professional match1..23 excluded; no training.',
        'counts': counts, 'cases': rows,
        'input_condition': 'Author published normalized coordinate features. Manual shuttle coordinates and human-bbox-associated HRNet poses; no automatic raw-video deployment claim.',
        'columns': COLUMNS, 'window_frames': 14, 'label_last_frames': 6,
        'inference_stride': 1, 'classification_strides_reported': [1, 3],
        'scale_Xdata': 'Identity, exactly as published commented-out normalization.',
        'weights_sha256': file_hash(WEIGHTS),
        'archive_sha256': file_hash(ARCHIVE),
        'code_sha256': file_hash(Path(__file__)),
        'author_source_sha256': {name: file_hash(AUTHOR / name) for name in ['domain_rnn.source.txt', 'process_pred_probs.source.txt']},
        'postprocess': {'modes': ['none', 'naive', 'max'], 'threshold': .95, 'fps': 30,
                        'naive_frame_range': 3, 'max_window_len': 6, 'offset': 13,
                        'parameters_tuned': False},
        'author_interval_metric': {'thresholds': IOU_THRESHOLDS,
            'literal_quirks': ['preds2seframes omits the final interval for each class.',
                'calc_metrics does not advance a matched GT index beyond that GT.',
                'mAP is the mean of per-video per-actor precision, not ranked precision-recall integral.',
                'GT drops first13frames, expands each contact by±3 using original NumPy edge slicing.']},
        'point_metric': 'All original sparse contacts retained including first13frames; each complete predicted nonzero run becomes its rounded midpoint at original frame index. Maximum-cardinality one-to-one temporal assignment ignoring actor, then minimum total error; ±3/6 original frames.',
        'training_executed': False}
    atomic_json(path, protocol)
    print('FROZEN', counts, flush=True)
    return protocol


def verify(protocol):
    assert file_hash(WEIGHTS) == protocol['weights_sha256']
    for row in protocol['cases']:
        for path, expected in row['input_sha256'].items():
            assert file_hash(Path(path)) == expected, path


def true_windows(labels):
    windows = labels[np.arange(len(labels)-13)[:, None] + np.arange(14)]
    last = windows[:, 8:]
    return np.where(np.any(last == 1, axis=1), 1, np.where(np.any(last == 2, axis=1), 2, 0))


def point_events(labels):
    events = []
    start = 0
    while start < len(labels):
        label = int(labels[start])
        end = start+1
        while end < len(labels) and labels[end] == label:
            end += 1
        if label:
            events.append({'frame_index': int(np.floor((start+end-1)/2+.5))+13,
                'actor': 'near' if label == 1 else 'far', 'camera_class': label,
                'start_frame': start+13, 'end_frame': end+12})
        start = end
    return events


def predict(protocol, functions):
    with np.load(WEIGHTS, allow_pickle=False) as archive:
        weights = {name: archive[name] for name in archive.files}
    for row in protocol['cases']:
        output = OUT / 'predictions' / row['id']
        path = output / 'outputs.npz'
        if path.exists():
            print('REUSE', row['id'], flush=True)
            continue
        output.mkdir(parents=True, exist_ok=True)
        features = pd.read_csv(row['x_path'])[COLUMNS].to_numpy(dtype=np.float32)
        count = len(features)-13
        probabilities = np.empty((count, 3), dtype=np.float32)
        for start in range(0, count, 256):
            indices = np.arange(start, min(start+256, count))
            probabilities[indices] = predict_windows(features[indices[:, None] + np.arange(14)], weights)
        none = np.argmax(probabilities, axis=1)
        none[np.max(probabilities, axis=1) <= .95] = 0
        naive = functions['optimise_hits_naive'](none.copy(), fps=30, frame_range=3)
        dynamic = maximise_hit_score(probabilities, fps=30, window_len=6)
        np.savez_compressed(path, probabilities=probabilities, none=none, naive=naive, max=dynamic)
        atomic_json(output / 'events.json', {mode: point_events(labels) for mode, labels in [('none', none), ('naive', naive), ('max', dynamic)]})
        print('PREDICTED', row['id'], len(features), flush=True)


def classification(confusion):
    total = int(confusion.sum())
    classes = []
    for label in range(3):
        tp, pred, true = int(confusion[label, label]), int(confusion[:, label].sum()), int(confusion[label].sum())
        p, r = tp/pred if pred else 0., tp/true if true else 0.
        classes.append({'class': label, 'support': true, 'precision': p, 'recall': r,
                        'f1': 2*p*r/(p+r) if p+r else 0.})
    return {'samples': total, 'accuracy': float(np.trace(confusion)/total),
        'macro_f1': float(np.mean([c['f1'] for c in classes])), 'classes': classes,
        'confusion_matrix_true_rows_pred_columns': confusion.tolist()}


def evaluate(protocol, functions):
    groups = {}
    rows = []
    for row in protocol['cases']:
        sparse = pd.read_csv(row['y_path'])['player_hit'].to_numpy(dtype=np.int32)
        truth = [{'frame_index': int(index), 'player': 'near' if label == 1 else 'far'}
                 for index, label in enumerate(sparse) if label]
        with np.load(OUT / 'predictions' / row['id'] / 'outputs.npz', allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        gt_windows = true_windows(sparse)
        group = groups.setdefault(row['group'], {'rallies': 0, 'frames': 0, 'contacts': 0,
            'classification': {str(s): np.zeros((3, 3), np.int64) for s in [1, 3]},
            'author_interval_metrics': {mode: {str(t): [] for t in IOU_THRESHOLDS} for mode in ['none', 'naive', 'max']},
            'point_metrics': {mode: {str(t): {'truth': 0, 'predicted': 0, 'matched': 0, 'actor_and_time_correct': 0} for t in [3, 6]} for mode in ['none', 'naive', 'max']}})
        group['rallies'] += 1
        group['frames'] += len(sparse)
        group['contacts'] += len(truth)
        for stride in [1, 3]:
            true, pred = gt_windows[::stride], np.argmax(arrays['probabilities'][::stride], axis=1)
            group['classification'][str(stride)] += np.bincount(3*true+pred, minlength=9).reshape(3, 3)
        detail = {'id': row['id'], 'group': row['group'], 'point_metrics': {}, 'author_interval_metrics': {}}
        for mode in ['none', 'naive', 'max']:
            labels = arrays[mode]
            events = point_events(labels)
            detail['point_metrics'][mode] = {}
            for tolerance in [3, 6]:
                matches = match_events(events, truth, tolerance_frames=tolerance, fps=30)
                count = len(matches['matches'])
                value = {'truth': len(truth), 'predicted': len(events), 'matched': count,
                         'actor_and_time_correct': sum(p['actor_correct'] is True for p in matches['matches'])}
                for key, number in value.items():
                    group['point_metrics'][mode][str(tolerance)][key] += number
                detail['point_metrics'][mode][str(tolerance)] = value
            detail['author_interval_metrics'][mode] = {}
            for threshold in IOU_THRESHOLDS:
                metrics = functions['eval_vid_metrics'](sparse[13:].copy(), labels.copy(), frame_range=3, iou_thresh=threshold)
                score = {f'{key}{actor+1}': float(metrics[name][actor]) for key, name in [('P', 'precision'), ('R', 'recall'), ('F', 'f1')] for actor in [0, 1]}
                group['author_interval_metrics'][mode][str(threshold)].append(score)
                detail['author_interval_metrics'][mode][str(threshold)] = score
        rows.append(detail)
    for group in groups.values():
        group['classification'] = {stride: classification(confusion) for stride, confusion in group['classification'].items()}
        for mode in ['none', 'naive', 'max']:
            for tolerance, value in group['point_metrics'][mode].items():
                n, p, k = [value[key] for key in ['truth', 'predicted', 'matched']]
                value.update(precision=k/p if p else 0., recall=k/n if n else 0., f1=2*k/(n+p) if n+p else 0.,
                             missed=n-k, extra=p-k, tolerance_frames=int(tolerance))
            for threshold, values in group['author_interval_metrics'][mode].items():
                group['author_interval_metrics'][mode][threshold] = {key: float(value) for key, value in functions['calc_overall_metrics'](pd.DataFrame(values)).items()}
    result = {'completed_at': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': file_hash(OUT / 'protocol.json'),
        'groups': groups, 'rows': rows, 'scope': protocol['scope'], 'input_condition': protocol['input_condition'],
        'author_literal_evaluation_quirks': protocol['author_interval_metric']['literal_quirks'],
        'window_classification_policy': 'All windows retained; stride3 follows the author window sampling before their random no-hit downsampling. No class balancing or evaluation subset resampling here.',
        'source_clock': 'Frame-index comparisons only. No original mp4 cadence claimed for all released feature files.',
        'training_executed': False}
    atomic_json(OUT / 'result.json', result)
    print(json.dumps(groups, ensure_ascii=False, indent=2), flush=True)


def write_report():
    result = json.loads((OUT / 'result.json').read_text())
    lines = ['# GRU作者发布输入复现', '', '2026-10-25。使用原作者发布的shuttlepose权重、原归一化特征列和后处理；本轮没有训练或调阈值。', '',
        '**这是作者输入条件的复现。X中含人工球坐标和人工框关联的HRNet姿态，不能当作完整自动录像结果。**', '',
        '| 数据 | 回合 | 帧 | 人工击球 | 14帧窗口准确率 | stride3窗口准确率 |', '|---|---:|---:|---:|---:|---:|']
    for name, group in result['groups'].items():
        lines.append(f"| {name} | {group['rallies']} | {group['frames']} | {group['contacts']} | {group['classification']['1']['accuracy']:.2%} | {group['classification']['3']['accuracy']:.2%} |")
    lines += ['', '职业只纳入test_match1/2/3；业余纳入发布包全部单打回合。职业普通match1..23未混入测试。窗口分类保留全部窗口，没有进行作者训练笔记本的随机无击球降采样，因此与其平衡后分类准确率分母不同。', '',
        '## 原作者区间评价', '', '以下精确执行原源代码，包括其末区间遗漏和GT复用行为；原代码称mAP，实际是各视频、各球员Precision的均值，并非按置信度排序的PR曲线积分。', '',
        '| 数据 | 后处理 | IoU>0.05 作者mAP | IoU>0.2 作者mAP | IoU>0.5 作者mAP |', '|---|---|---:|---:|---:|']
    for name, group in result['groups'].items():
        for mode in ['none', 'naive', 'max']:
            values = group['author_interval_metrics'][mode]
            lines.append(f"| {name} | {mode} | {values['0.05']['mAP']:.2%} | {values['0.2']['mAP']:.2%} | {values['0.5']['mAP']:.2%} |")
    lines += ['', '## 完整逐拍一对一评价', '', '原作者原区间评价丢弃片头13帧，且每类最后一段未加入区间列表；以下补充指标保留所有真实击球，包括模型无前序上下文的片头拍。连续同类输出转成区间中点；不读真值调中心。', '',
        '| 数据 | 后处理 | ±3帧 TP/GT | ±3帧 Precision / Recall / F1 | ±6帧F1 |', '|---|---|---:|---|---:|']
    for name, group in result['groups'].items():
        for mode in ['none', 'naive', 'max']:
            a, b = [group['point_metrics'][mode][key] for key in ['3', '6']]
            lines.append(f"| {name} | {mode} | {a['matched']}/{a['truth']} | {a['precision']:.2%} / {a['recall']:.2%} / {a['f1']:.2%} | {b['f1']:.2%} |")
    lines += ['', '±3/6帧是原始索引容差，不能在原录像帧率未逐个核实时声称统一物理毫秒。14帧窗口结果与击球事件F1也不可互换。', '',
        f'- [完整结果]({OUT}/result.json)', f'- [固定输入与协议]({OUT}/protocol.json)',
        f'- [复现脚本]({Path(__file__).resolve()})', '', '作者原区间函数从已保存源文件以AST仅提取函数定义执行，未执行笔记本的训练、挂载云盘或其他顶层操作。']
    (OUT / '结果说明.md').write_text('\n'.join(lines) + '\n')


def main():
    started = time.perf_counter()
    protocol = prepare()
    verify(protocol)
    functions = author_functions()
    predict(protocol, functions)
    evaluate(protocol, functions)
    verify(protocol)
    write_report()
    print('DONE', OUT, f'{time.perf_counter()-started:.2f}s', flush=True)


if __name__ == '__main__':
    main()
