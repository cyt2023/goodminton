"""Read-only audit of author source annotations, merged sample membership and labels.

Only writes an independent diagnostic directory. No model imports, inference,
label edits, training or production-cache changes are performed.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/audit/
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import ast
import csv
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '03_数据/击球标注/根因诊断/2026-10-15_球种语义审计'
SET = ROOT / '04_原型/vendor/BST/ShuttleSet/set'
MERGED = ROOT / '03_数据/公开训练数据/BST_ShuttleSet/dataset_npy_between_2_hits_with_max_limits'
POOL = ROOT / '03_数据/训练库/训练样本.json'
TAXONOMY = ROOT / '04_原型/stroke_taxonomy.py'
OFFICIAL = 'https://raw.githubusercontent.com/wywyWang/CoachAI-Projects/main/ShuttleSet/README.md'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def constants(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'VERSION', 'LABELS', 'AUTHOR_MAP', 'TERMS'}:
                    result[target.id] = ast.literal_eval(node.value)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    monitored = [POOL, TAXONOMY, *sorted((ROOT / '03_数据/击球标注').glob('J_*/annotations.json'))]
    before = {str(p.relative_to(ROOT)): sha(p) for p in monitored}
    taxonomy = constants(TAXONOMY)
    source = OUT / 'author_ShuttleSet_README.md'
    if not source.is_file():
        source.write_bytes(urllib.request.urlopen(OFFICIAL).read())
    translations = {}
    for line in source.read_text().splitlines():
        if '|' in line:
            cells = [c.strip() for c in line.strip('|').split('|')]
            if len(cells) == 2 and cells[1] not in {'Chinese', '---'}:
                translations[cells[1]] = cells[0]

    matches = {int(r['id']): r['video'] for r in csv.DictReader((SET / 'match.csv').open())}
    source_index, raw_files, repeated_source_keys = {}, {}, []
    for mid, name in matches.items():
        for path in sorted((SET / name).glob('set*.csv')):
            raw_files[str(path.relative_to(ROOT))] = sha(path)
            set_id = int(path.stem[3:])
            for record in csv.DictReader(path.open()):
                key = (mid, set_id, int(float(record['rally'])), int(float(record['ball_round'])))
                value = {'type': record['type'], 'file': str(path.relative_to(ROOT)),
                         'frame_num': record['frame_num']}
                if key in source_index:
                    repeated_source_keys.append({'key': key, 'old': source_index[key], 'new': value})
                    if source_index[key]['type'] != value['type']:
                        raise ValueError('Ambiguous author source type: ' + str(key))
                source_index[key] = value

    matrix, split_counts, examples, unmatched, sample_names = Counter(), Counter(), defaultdict(list), [], []
    for path in sorted(MERGED.glob('*/*/*_joints.npy')):
        stem = path.stem.removesuffix('_joints')
        key = tuple(map(int, stem.split('_')))
        original = source_index.get(key)
        merged_type = path.parent.name.split('_', 1)[-1]
        sample_names.append(str(path.relative_to(MERGED)))
        split_counts[path.relative_to(MERGED).parts[0]] += 1
        if original is None:
            unmatched.append(str(path.relative_to(ROOT)))
            continue
        pair = (original['type'], merged_type)
        matrix[pair] += 1
        if len(examples[pair]) < 3:
            examples[pair].append({'stem': stem, 'sample': str(path.relative_to(ROOT)), **original})
    empirical = defaultdict(set)
    for old, new in matrix:
        empirical[old].add(new)
    nonunique = {k: sorted(v) for k, v in empirical.items() if len(v) != 1}
    if unmatched or nonunique:
        raise ValueError('Source join did not define a unique author merge')

    pooled = json.loads(POOL.read_text())['samples']
    positives = [r for r in pooled if r['training_role'] == 'stroke_positive']
    basic_codes = set(taxonomy['AUTHOR_MAP'].values()) - {'unknown'}
    basic = [r for r in positives if r['stroke_type'] in basic_codes]
    fine = [r for r in positives if r['stroke_type'] not in basic_codes]
    basic_counts = Counter(r['stroke_type'] for r in basic)
    role_counts = Counter(r['training_role'] for r in pooled)
    expanded_codes = {'smash', 'lift', 'drive', 'drop'}
    expanded_rows = [r for r in basic if r['stroke_type'] in expanded_codes]
    defense_codes = {'defense', 'defense_deep_net', 'defense_rear'}
    defense_rows = [r for r in positives if r['stroke_type'] in defense_codes]
    rows = []
    for (old, new), count in sorted(matrix.items()):
        code = taxonomy['AUTHOR_MAP'].get(new)
        rows.append({'original_type': old, 'official_english': translations.get(old),
                     'merged_type': new, 'local_code': code,
                     'display_name': taxonomy['LABELS'].get(code), 'author_sample_count': count,
                     'source_examples': examples[(old, new)]})
    result = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'protocol': 'Join all downloaded merged *_joints.npy sample IDs to author-corrected raw CSV by match/set/rally/ball_round. No name-based merge inference.',
        'official_translation_source': OFFICIAL,
        'official_merge_source': 'https://arxiv.org/html/2502.21085v4#A5',
        'source_index_files_sha256': raw_files,
        'author_readme_sha256': sha(source),
        'merged_sample_name_index_sha256': hashlib.sha256('\n'.join(sample_names).encode()).hexdigest(),
        'total_merged_samples': len(sample_names), 'matched_samples': sum(matrix.values()),
        'split_counts': dict(split_counts), 'unmatched_samples': unmatched,
        'nonunique_source_merge': nonunique, 'repeated_source_keys': repeated_source_keys,
        'mapping_rows': rows, 'taxonomy_version': taxonomy['VERSION'],
        'pooled_records': len(pooled), 'roles': dict(role_counts),
        'basic_positive_count': len(basic), 'basic_positive_counts': dict(basic_counts),
        'basic_positive_rows': basic, 'fine_positive_count': len(fine),
        'fine_positive_counts': dict(Counter(r['stroke_type'] for r in fine)), 'fine_positive_rows': fine,
        'merged_family_basic_count': len(expanded_rows),
        'merged_family_basic_counts': dict(Counter(r['stroke_type'] for r in expanded_rows)),
        'merged_family_note': 'Exposure to a broader author class is not evidence that any individual human label is wrong.',
        'named_defense_positive_counts': dict(Counter(r['stroke_type'] for r in defense_rows)),
        'independent_physical_strokes_verified': False,
        'confirmed_foundational_label_swap_count': 0,
        'inference_performed': False, 'training_performed': False,
        'monitored_production_files_before': before,
    }
    after = {str(p.relative_to(ROOT)): sha(p) for p in monitored}
    result['monitored_production_files_after'] = after
    result['monitored_production_files_unchanged'] = before == after
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    with (OUT / '作者原类与合并类核对.csv').open('w', newline='') as f:
        fields = ['original_type', 'official_english', 'merged_type', 'local_code', 'display_name', 'author_sample_count']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k: result[k] for k in ['total_merged_samples', 'matched_samples', 'split_counts',
        'basic_positive_count', 'basic_positive_counts', 'fine_positive_count', 'fine_positive_counts',
        'merged_family_basic_count', 'monitored_production_files_unchanged']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
