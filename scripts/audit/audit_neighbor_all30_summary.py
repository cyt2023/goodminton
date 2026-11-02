"""Summarize every preselected development row, with all boundary regressions.

No inference, fitting, coordinate editing, or production-state mutation.
This is an explicitly retrospective summary of frozen, completed controls.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/audit/
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[3]
DIAG = BASE / '03_数据/击球标注/根因诊断'
LATEST = DIAG / '2026-10-20_邻拍组合与关键帧恢复'
OUT = DIAG / '2026-10-21_定位汇总/all30_neighbor_summary'
RUN = BASE / '03_数据/击球标注/直接BST微调/运行记录/582817bff5454825b841d1f150c82028'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'result.json').exists():
        raise FileExistsError('Preserve completed summary')
    dependencies = {}

    def read(path):
        dependencies[str(path)] = digest(path)
        return json.loads(path.read_text())

    records = read(RUN / 'personal_predictions.json')['baseline']
    mapping = read(DIAG / '2026-10-18_物理触球去重核查/frozen_physical_contact_mapping.json')
    physical = {r['event_id']: r for r in mapping['rows']}
    records = [r for r in records if r['event_id'] in physical]
    assert len(records) == 30 and len(physical) == 30
    index = {r['event_id']: r for r in records}
    rows = []

    def add(event_id, original, grid, evidence):
        record = index[event_id]
        assert grid and original['prediction'] == record['source_top_label']
        rows.append({
            'event_id': event_id, 'job_id': record['job_id'],
            'human_peak_time': record['human_peak_time'],
            'true_label': record['target_class_label'],
            'diagnostic_group_id': physical[event_id]['diagnostic_physical_contact_group_id'],
            'group_member_event_ids': physical[event_id]['member_event_ids'],
            'evidence': str(evidence), 'original': original,
            'grid_count': len(grid),
            'reviewed_predictions': sorted({r['prediction'] for r in grid}),
            'reviewed_actor13_predictions': sorted({r['actor13_prediction'] for r in grid}),
            'all_joint_correct': all(r['joint_correct'] for r in grid),
            'any_joint_correct': any(r['joint_correct'] for r in grid),
            'all_actor13_correct': all(r['actor13_correct'] for r in grid),
            'any_actor13_correct': any(r['actor13_correct'] for r in grid),
            'joint_correct_windows': sum(r['joint_correct'] for r in grid),
            'actor13_correct_windows': sum(r['actor13_correct'] for r in grid),
        })

    for folder in [
        DIAG / '2026-10-18_跨来源邻拍对照/other_sources_11/classification_control',
        LATEST / 'ordinary_other5_classification',
        LATEST / 'ordinary_remaining12_classification',
    ]:
        path = folder / 'result.json'
        data = read(path)
        read(folder / 'protocol.json')
        for r in data['rows']:
            add(r['event_id'], r['original']['conditions']['baseline'],
                [g['conditions']['baseline'] for g in r['grid']], path)

    two = DIAG / '2026-10-18_普通视角配对诊断/actual_neighbor_window_grid'
    initial = read(two / 'result.json')
    sensitivity = read(two / 'target_time_sensitivity/result.json')
    read(two / 'visual_review_frozen.json')
    first = '97b39d14d2435ac65ccb3023a764f49c'
    original = next(r for r in initial['rows'] if r['event_id'] == first and
                    r['pose'] == 'original' and r['kind'] == 'original_window')
    add(first, original,
        [r for r in initial['rows'] if r['event_id'] == first and
         r['pose'] == 'original' and r['kind'] == 'reviewed_neighbor_grid'], two / 'result.json')
    drop = '3d5e6020fbc2aa558b6a5914a9826a54'
    original = next(r for r in initial['rows'] if r['event_id'] == drop and r['kind'] == 'original_window')
    add(drop, original, [r for r in sensitivity['rows'] if r['name'].startswith('drop_')],
        two / 'target_time_sensitivity/result.json')

    assert len(rows) == len({r['event_id'] for r in rows}) == 30
    assert {r['event_id'] for r in rows} == set(index)
    rows.sort(key=lambda r: (r['job_id'], r['human_peak_time'], r['event_id']))
    dependencies[str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
    groups = defaultdict(list)
    for r in rows:
        groups[r['diagnostic_group_id']].append(r)
    assert len(groups) == 28

    protocol = {
        'declared_at': datetime.now(timezone.utc).isoformat(),
        'retrospective_summary_not_new_preregistered_experiment': True,
        'scope': 'All30 frozen developer annotation rows; original BST-0 only; original saved pose and ball only; all reviewed bounds retained. No newly chosen or fitted windows.',
        'comparison': 'Original single-window correctness versus correct at EVERY or ANY reviewed source-neighbor boundary. EVERY is robustness and ANY is sensitivity, neither is an automatic window selector or upload accuracy.',
        'groups': 'Only the two source-verified physical duplicate pairs merged; row and group-mean/all-members/earliest/latest summaries all retained. No best representative selection.',
        'exclusions': 'No restored-pose or manually corrected-ball branches mixed into this uniform window-only summary. No all71 training checkpoint.',
        'dependency_sha256': dependencies,
    }
    (OUT / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    summaries = {}
    for metric in ['joint', 'actor13']:
        field = 'joint_correct' if metric == 'joint' else 'actor13_correct'
        values = {
            'original': lambda r: int(r['original'][field]),
            'all_reviewed_bounds': lambda r: int(r[f'all_{metric}_correct']),
            'any_reviewed_bound': lambda r: int(r[f'any_{metric}_correct']),
        }
        summary = {}
        for name, value in values.items():
            summary[name] = {
                'annotation_row_numerator': sum(value(r) for r in rows), 'annotation_rows': 30,
                'group_mean_numerator': sum(sum(value(r) for r in g) / len(g) for g in groups.values()),
                'group_all_members_numerator': sum(all(value(r) for r in g) for g in groups.values()),
                'group_earliest_numerator': sum(value(min(g, key=lambda r: (r['human_peak_time'], r['event_id']))) for g in groups.values()),
                'group_latest_numerator': sum(value(max(g, key=lambda r: (r['human_peak_time'], r['event_id']))) for g in groups.values()),
                'diagnostic_groups': 28,
            }
        summary['stable_gained_rows'] = [r['event_id'] for r in rows if not r['original'][field] and r[f'all_{metric}_correct']]
        summary['all_new_bounds_wrong_previously_correct_rows'] = [r['event_id'] for r in rows if r['original'][field] and not r[f'any_{metric}_correct']]
        summary['boundary_sensitive_previously_correct_rows'] = [r['event_id'] for r in rows if r['original'][field] and r[f'any_{metric}_correct'] and not r[f'all_{metric}_correct']]
        summaries[metric] = summary
    assert all(digest(Path(p)) == h for p, h in dependencies.items())
    result = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': digest(OUT / 'protocol.json'), 'summary': summaries, 'rows': rows,
        'row_window_evaluations': sum(r['grid_count'] for r in rows),
        'production_labels_weights_modified': False,
        'limitation': 'Researcher visual intervals, not user exact-contact GT. 30 rows include 2 duplicate excess rows. Group means can be fractional; they do not count fractional physical successes. The all/any metrics measure sensitivity across correlated windows, not new-video accuracy.',
    }
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'summary': summaries, 'row_window_evaluations': result['row_window_evaluations']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
