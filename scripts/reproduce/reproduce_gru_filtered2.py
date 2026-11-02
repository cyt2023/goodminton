"""Replay author's filtered2 features with their separately trained checkpoint."""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/reproduce/

import collections
from datetime import datetime, timezone
import gc
import inspect
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from analysis_jobs import atomic_json, file_hash
import reproduce_gru_author_features as replay

ASSETS = replay.PROJECT / '03_数据/GRU作者流程复现/2026-10-25/author_assets'
OUT = ASSETS.parent / 'filtered2_features'
FEATURES = ASSETS / 'extracted/filtered2/domain'
SAVED_PROBS = ASSETS / 'extracted/domain_probs_filtered2'
CHECKPOINT = ASSETS / 'shuttlepose_filtered2/150'
WEIGHTS = CHECKPOINT.parent / 'keras_tensors.npz'


def convert():
    if WEIGHTS.exists():
        return
    np.string_, np.unicode_ = np.bytes_, np.str_
    inspect.ArgSpec = collections.namedtuple('ArgSpec', 'args varargs keywords defaults')
    inspect.getargspec = lambda f: inspect.ArgSpec(*inspect.getfullargspec(f)[:4])
    from tensorflow_checkpoint_reader import py_checkpoint_reader
    reader = py_checkpoint_reader.NewCheckpointReader(str(CHECKPOINT).encode())
    names = {'dense_kernel': '0/kernel', 'dense_bias': '0/bias',
        'gru1_kernel': '1/cell/kernel', 'gru1_recurrent': '1/cell/recurrent_kernel', 'gru1_bias': '1/cell/bias',
        'gru2_kernel': '2/cell/kernel', 'gru2_recurrent': '2/cell/recurrent_kernel', 'gru2_bias': '2/cell/bias',
        'output_kernel': '3/kernel', 'output_bias': '3/bias'}
    with np.load(replay.WEIGHTS, allow_pickle=False) as original:
        shapes = {key: original[key].shape for key in names}
    arrays = {}
    for key, source in names.items():
        value = reader.get_tensor('layer_with_weights-' + source + '/.ATTRIBUTES/VARIABLE_VALUE')
        assert value.shape == shapes[key] and np.isfinite(value).all()
        arrays[key] = value.copy()
    del reader
    gc.collect()
    np.savez_compressed(WEIGHTS, **arrays)
    atomic_json(CHECKPOINT.parent / 'conversion.json', {
        'source': 'Author Drive mm_weights/shuttlepose/filtered2/150',
        'checkpoint_sha256': {p.name: file_hash(p) for p in CHECKPOINT.parent.glob('150.*')},
        'weights_sha256': file_hash(WEIGHTS),
        'architecture': 'Dense32ReLU -> GRU32(reset_after=True) -> GRU32(reset_after=True) -> Dense3softmax',
        'separate_from_gt_domain_checkpoint': True, 'training_executed': False})
    print('FILTERED2_WEIGHTS_READY', WEIGHTS, flush=True)


def prepare():
    target = OUT / 'protocol.json'
    if target.exists():
        return json.loads(target.read_text())
    rows = []
    for domain, group in [('domain_pro', 'professional_author_test'), ('domain_am_singles', 'amateur_singles_author_test')]:
        for xfile in sorted((FEATURES / domain / 'test').rglob('*_x.csv')):
            yfile = xfile.with_name(xfile.name[:-6] + '_y.csv')
            match = xfile.parent.name
            x, y = pd.read_csv(xfile), pd.read_csv(yfile)
            assert len(x) == len(y)
            assert np.array_equal(x['Frame'], y['frame'])
            assert np.array_equal(x['Frame'], np.arange(len(x)))
            assert np.isfinite(x[replay.COLUMNS].to_numpy(dtype=np.float32)).all()
            probability = SAVED_PROBS / ('pro' if domain == 'domain_pro' else 'am_singles') / match / (xfile.name[:-6] + '_probs.npy')
            assert probability.exists()
            rows.append({'id': group+'_'+match+'_'+xfile.stem[:-2], 'group': group, 'match': match,
                'x_path': str(xfile), 'y_path': str(yfile), 'author_probability_path': str(probability),
                'frames': len(x), 'contacts': int((y['player_hit'] > 0).sum()),
                'input_sha256': {str(p): file_hash(p) for p in [xfile, yfile, probability]}})
    assert len(rows) == 63
    gt_protocol = json.loads((ASSETS.parent / 'author_features/protocol.json').read_text())
    protocol = {**gt_protocol,
        'created_at': datetime.now(timezone.utc).isoformat(), 'cases': rows,
        'counts': {group: {'rallies': sum(r['group'] == group for r in rows),
            'frames': sum(r['frames'] for r in rows if r['group'] == group),
            'contacts': sum(r['contacts'] for r in rows if r['group'] == group)} for group in sorted({r['group'] for r in rows})},
        'scope': 'All63 explicitly test-split author filtered2 singles feature pairs, paired filtered2 checkpoint. No local model training or parameter selection.',
        'input_condition': 'Author filtered2 auto-selected FasterRCNN/HRNet pose. Professional shuttle uses manual coordinates (automatic shuttle unavailable upstream); amateur shuttle uses author TrackNetV2 predictions. Court corners are manual. This remains semi-automatic, not fully automatic raw-video evaluation.',
        'weights_sha256': file_hash(WEIGHTS), 'archive_sha256': file_hash(ASSETS / 'filtered2.zip'),
        'author_probability_archive_sha256': file_hash(ASSETS / 'domain_probs_filtered2.zip'),
        'code_sha256': file_hash(Path(__file__)),
        'shared_replay_code_sha256': file_hash(Path(replay.__file__)),
        'provenance_source_sha256': file_hash(replay.AUTHOR / 'organise_input_features.source.txt'),
        'probability_check': 'Compare every freshly predicted stride1 probability against the corresponding author-published npy, same input rows. No selection by agreement or score.'}
    atomic_json(target, protocol)
    return protocol


def parity(protocol):
    rows = []
    for row in protocol['cases']:
        original = np.load(row['author_probability_path'], allow_pickle=False)
        with np.load(OUT / 'predictions' / row['id'] / 'outputs.npz', allow_pickle=False) as archive:
            reproduced = archive['probabilities']
        assert original.shape == reproduced.shape, row['id']
        rows.append({'id': row['id'], 'shape': list(original.shape),
            'max_abs_difference': float(np.max(np.abs(original-reproduced))),
            'mean_abs_difference': float(np.mean(np.abs(original-reproduced))),
            'argmax_disagreements': int(np.sum(np.argmax(original, axis=1) != np.argmax(reproduced, axis=1)))})
    summary = {'rallies': len(rows), 'windows': sum(r['shape'][0] for r in rows),
        'max_abs_difference': max(r['max_abs_difference'] for r in rows),
        'argmax_disagreements': sum(r['argmax_disagreements'] for r in rows),
        'rows': rows, 'weights_sha256': file_hash(WEIGHTS)}
    atomic_json(OUT / 'author_probability_parity.json', summary)
    print('AUTHOR_PROBABILITY_PARITY', {k: v for k, v in summary.items() if k != 'rows'}, flush=True)


def compare_feature_provenance(protocol):
    gt_archive = replay.PROJECT / '03_数据/找拍模型对照/2026-10-22/extension_assets/domain.zip'
    comparisons = []
    with zipfile.ZipFile(gt_archive) as archive:
        for row in protocol['cases']:
            xfile = Path(row['x_path'])
            domain = 'domain_pro' if row['group'] == 'professional_author_test' else 'domain_am_singles'
            member = f'domain/{domain}/{row["match"]}/{xfile.name}'
            gt = pd.read_csv(archive.open(member))
            auto = pd.read_csv(xfile)
            assert len(gt) == len(auto)
            ball = np.abs(gt[['ball_x', 'ball_y']].to_numpy() - auto[['ball_x', 'ball_y']].to_numpy())
            pose = np.abs(gt[replay.POSE_COLUMNS].to_numpy() - auto[replay.POSE_COLUMNS].to_numpy())
            comparisons.append({'id': row['id'], 'group': row['group'], 'frames': len(gt),
                'identical_ball_frames': int(np.all(ball < 1e-12, axis=1).sum()),
                'identical_pose_frames': int(np.all(pose < 1e-12, axis=1).sum()),
                'ball_max_difference': float(ball.max()), 'pose_max_difference': float(pose.max())})
    atomic_json(OUT / 'feature_provenance_comparison.json', {'rows': comparisons,
        'source': 'organise_input_features.source.txt uses datasets/pro/*/ball_trajectory for professional filtered2 but annotation_pipeline/object_level/shuttle for amateur filtered2; direct comparison to original domain included.'})


def evaluate_saved_reference(protocol, functions):
    """Score author archives separately; never replace local model outputs."""
    target = OUT / 'author_saved_probability_reference'
    reference = {**protocol,
        'scope': 'Reference evaluation of author-published domain_probs_filtered2, not probabilities reproduced by this local checkpoint execution.',
        'probability_source': 'Untouched author npy outputs; correspondence to the released checkpoint could not be established.',
        'local_model_forward_executed': False}
    atomic_json(target / 'protocol.json', reference)
    for row in protocol['cases']:
        probability = np.load(row['author_probability_path'], allow_pickle=False)
        none = np.argmax(probability, axis=1)
        none[np.max(probability, axis=1) <= .95] = 0
        naive = functions['optimise_hits_naive'](none.copy(), fps=30, frame_range=3)
        dynamic = replay.maximise_hit_score(probability, fps=30, window_len=6)
        folder = target / 'predictions' / row['id']
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'outputs.npz', probabilities=probability, none=none, naive=naive, max=dynamic)
    original = replay.OUT
    try:
        replay.OUT = target
        replay.evaluate(reference, functions)
    finally:
        replay.OUT = original


def report():
    result = json.loads((OUT / 'result.json').read_text())
    check = json.loads((OUT / 'author_probability_parity.json').read_text())
    lines = ['# 作者filtered2配套权重与输入复现', '',
        '2026-10-25。使用作者另行发布的shuttlepose/filtered2/150权重，未沿用原GT域权重。', '',
        '**输入条件：职业=自动人物姿态＋人工球坐标；业余=自动人物姿态＋作者TrackNetV2球轨迹；球场角点仍为人工标定。**', '',
        f"对作者预存概率逐窗口核对：{check['rallies']}回合、{check['windows']}窗口，最大绝对差{check['max_abs_difference']:.9g}，argmax分歧{check['argmax_disagreements']}。", '',
        '**当前公开checkpoint和作者预存概率未对齐，因此不能称已重现作者归档概率或原报告分数。** 已核对真实Drive路径、输入列顺序和归一化；直接Keras重建前向与本地NumPy数值一致。公开目录没有其他filtered2训练版本或专用训练笔记本；缺失的历史对应关系尚不能确定。原始概率另作reference评价，绝不替换本地模型输出。', '',
        '| 数据 | 回合/拍 | 后处理 | 原作者mAP(IoU>0.2) | 完整逐拍±3帧F1 | 完整逐拍±6帧F1 |', '|---|---:|---|---:|---:|---:|']
    for name, group in result['groups'].items():
        for mode in ['none', 'naive', 'max']:
            lines.append(f"| {name} | {group['rallies']}/{group['contacts']} | {mode} | {group['author_interval_metrics'][mode]['0.2']['mAP']:.2%} | {group['point_metrics'][mode]['3']['f1']:.2%} | {group['point_metrics'][mode]['6']['f1']:.2%} |")
    lines += ['', '原作者mAP实际为各视频/球员Precision平均，原函数还遗漏每类最后区间；这里精确保留该口径，同时补充完整逐拍一对一指标。完整逐拍指标保留片头无上下文击球；两者分母不同。', '',
        '窗口分类另保存全量stride1与stride3结果，没有用随机负例降采样改变分母。源视频帧率未全部核实，±3/6帧不直接称统一毫秒。', '',
        f'- [完整评价]({OUT}/result.json)', f'- [作者概率逐窗对照]({OUT}/author_probability_parity.json)',
        f'- [作者归档概率单独评价，非本地模型成绩]({OUT}/author_saved_probability_reference/result.json)',
        f'- [Keras与NumPy前向核对]({OUT}/keras_forward_check.json)',
        f'- [输入来源逐帧比较]({OUT}/feature_provenance_comparison.json)', f'- [独立新权重]({WEIGHTS})',
        f'- [固定协议]({OUT}/protocol.json)', f'- [复现脚本]({Path(__file__).resolve()})']
    (OUT / '结果说明.md').write_text('\n'.join(lines)+'\n')


def main():
    convert()
    replay.OUT, replay.WEIGHTS = OUT, WEIGHTS
    protocol = prepare()
    replay.verify(protocol)
    functions = replay.author_functions()
    replay.predict(protocol, functions)
    parity(protocol)
    compare_feature_provenance(protocol)
    replay.evaluate(protocol, functions)
    evaluate_saved_reference(protocol, functions)
    replay.verify(protocol)
    report()
    print('FILTERED2_DONE', OUT, flush=True)


if __name__ == '__main__':
    main()
