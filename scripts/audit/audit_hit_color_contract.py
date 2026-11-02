"""Diagnose the verified pose RGB contract in hit detection, before VLM."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/audit/
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

from analysis_jobs import atomic_json, file_hash
from audit_paired_source_events import evaluate, refined_candidates
from source_frame_media import ffmpeg_executable
from specialist_events import proposals

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / '03_数据/击球标注/根因诊断/2026-10-15_同源前端对照'
OUT = SOURCE.parent / '2026-10-18_漏拍规则干预/color_contract'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'result.json').exists():
        raise FileExistsError('Preserve completed audit')
    paths = {'bgr': SOURCE / 'continuous_rally/pose/dense_pose.json',
             'rgb': SOURCE / 'continuous_rally/pose/dense_pose_rgb_same_boxes.json',
             'trajectory': SOURCE / 'continuous_rally/tracknet/trajectory.json',
             'frame_map': SOURCE / 'continuous_rally/frame_map.json',
             'audio': SOURCE / 'continuous_rally/source_aligned_audio.wav',
             'audio_verification': SOURCE / 'continuous_rally/source_aligned_audio_verification.json',
             'event_protocol': SOURCE / 'automatic_events_protocol.json',
             'prior_raw': SOURCE / 'automatic_events/raw_visual_candidates.json',
             'prior_refined': SOURCE / 'automatic_events/original_audio_production_candidates.json',
             'audit_code': Path(__file__)}
    hashes = {k: file_hash(v) for k, v in paths.items()}
    code = ['specialist_events.py', 'hit_refinement.py', 'hit_gru.py', 'stroke_features.py', 'contact_timing.py']
    protocol = {'declared_at': datetime.now(timezone.utc).isoformat(),
                'inputs': {k: str(v) for k, v in paths.items()}, 'hashes': hashes,
                'code_hashes': {n: file_hash(ROOT / n) for n in code},
                'conditions': ['saved_BRG_primary', 'RGB_same_boxes_assignments'],
                'design': 'Existing same-box RGB pose, original shuttle, original unmodified proposals and audio/visual/GRU refinement. No threshold changes, new pose inference, or training.',
                'evaluation': 'Frozen20contacts and <=6frames Hungarian matching, actor measured separately.',
                'cache_isolation': 'Task-local hit_refinement.DATA percondition; original audio_onsets receives native aligned WAV via audit-only wrapper.',
                'limitations': 'Known development rally. No hitfinder training, no VLM, no general accuracy claim.',
                'production_activation': False}
    atomic_json(OUT / 'protocol.json', protocol)
    event_protocol = json.loads(paths['event_protocol'].read_text())
    truth = event_protocol['truth']
    trajectory = json.loads(paths['trajectory'].read_text())
    frame_map = json.loads(paths['frame_map'].read_text())
    verification = json.loads(paths['audio_verification'].read_text())
    assert verification['audio_sha256'] == hashes['audio'] and verification['native_pts_alignment_verified']
    import hit_refinement
    original_audio_function = hit_refinement.audio_onsets
    original_data_root = hit_refinement.DATA
    results = {}
    for name in ['bgr', 'rgb']:
        started = time.perf_counter()
        dense = json.loads(paths[name].read_text())
        events = proposals(dense, trajectory)
        if name == 'bgr':
            expected = json.loads(paths['prior_raw'].read_text())['candidates']
            assert canonical(events) == canonical(expected)
        else:
            primary = json.loads(paths['bgr'].read_text())
            for a, b in zip(primary['frames'], dense['frames']):
                assert a['frame_index'] == b['frame_index'] and a['source_time'] == b['source_time']
                for side in ['near', 'far']:
                    p, q = a['players'][side], b['players'][side]
                    assert (p is None) == (q is None)
                    if p is not None:
                        assert p['bbox'] == q['bbox']
        condition_folder = OUT / name
        condition_folder.mkdir(exist_ok=True)
        hit_refinement.DATA = condition_folder / 'runtime'
        folder = hit_refinement.DATA / '专项分析' / dense['id']
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths[name], folder / 'dense_pose.json')
        shutil.copyfile(paths['trajectory'], folder / 'trajectory.json')
        config = {'pose_id': dense['id'], 'source_file': str(paths['audio']),
                  'source_sha256': dense['source_sha256'], 'ffmpeg': ffmpeg_executable()}
        hit_refinement.audio_onsets = lambda cfg: original_audio_function({**cfg, 'source_file': str(paths['audio'])})
        try:
            refinement = hit_refinement.prepare_inputs(config, dense, trajectory, events)
        finally:
            hit_refinement.audio_onsets = original_audio_function
            hit_refinement.DATA = original_data_root
        refined = refined_candidates(dense, events, refinement)
        if name == 'bgr':
            expected = json.loads(paths['prior_refined'].read_text())['candidates']
            assert canonical(refined) == canonical(expected)
        results[name] = {'raw': evaluate(events, dense, frame_map, truth),
                         'refined': evaluate(refined, dense, frame_map, truth),
                         'refinement_summary': refinement['summary'], 'audio': refinement['audio'],
                         'runtime_seconds': time.perf_counter() - started}
        atomic_json(condition_folder / 'raw_candidates.json', {'candidates': events})
        atomic_json(condition_folder / 'refined_candidates.json', {'candidates': refined})
        print(json.dumps({'condition': name, **{stage: {k: r[k] for k in [
            'candidate_count', 'matched_count', 'precision', 'recall', 'actor_correct_among_matches']} for stage, r in results[name].items() if stage in ['raw', 'refined']}}, ensure_ascii=False), flush=True)
    assert all(file_hash(paths[k]) == h for k, h in hashes.items())
    assert all(file_hash(ROOT / n) == h for n, h in protocol['code_hashes'].items())
    atomic_json(OUT / 'result.json', {'created_at': datetime.now(timezone.utc).isoformat(),
                'protocol_sha256': file_hash(OUT / 'protocol.json'), 'checks': results,
                'bgr_raw_and_refined_events_byte_equal_prior_frozen_audit': True,
                'production_or_labels_or_weights_changed': False})


if __name__ == '__main__':
    main()
