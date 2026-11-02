"""Create local, explicitly unreviewed phase proposals from sampled observations.

Existing review files, including human edits, are left intact. Signals use
current pose outputs and are not copied into confirmed phase labels.
"""
import json
from pathlib import Path

from phase_review import initialize_candidates

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / '03_数据'


def main():
    candidates = json.loads((ROOT / 'phase_candidates.json').read_text(encoding='utf-8'))
    index = json.loads((DATA / '姿态分析' / 'analysis_index.json').read_text(encoding='utf-8'))
    for item in index:
        if item['id'] not in candidates:
            continue
        file = (DATA / item['result_file']).resolve()
        if not file.is_relative_to(DATA.resolve()):
            raise ValueError('Pose file is outside data directory')
        pose = json.loads(file.read_text(encoding='utf-8'))
        review, created = initialize_candidates(pose, candidates[item['id']])
        print(json.dumps({'pose_id': pose['id'], 'created': created,
                          'candidate_counts': {side: len(player['phases']) for side, player in review['players'].items()},
                          'signal_counts': {side: len(items) for side, items in review['signals'].items()},
                          'revision': review['revision']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
