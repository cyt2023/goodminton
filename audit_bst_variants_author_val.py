"""Compare released BST variants on author validation arrays, never test data."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from analysis_jobs import DATA, atomic_json, file_hash
from bst_public_data import FOLDER, ARRAYS, public_classifier, values
from specialist_bst import BST_CG_AP
from specialist_model import ROOT, BST_NAME, FULL_BST_NAME
from finetune_personal_bst import metrics

OUT = DATA / '击球标注/根因诊断/2026-10-15_作者验证集BST版本对照'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    index = json.loads((FOLDER / 'index.json').read_text())
    rows = [r for r in index['samples'] if r['split'] == 'val']
    assert len(rows) == 4241 and all(r['base'].startswith('val/') for r in rows)
    protocol = {
        'declared_at': datetime.now(timezone.utc).isoformat(),
        'split': 'author val only', 'samples': len(rows), 'batch_size': 16,
        'models': {name: file_hash(ROOT / 'models/bst' / name)
                   for name in [BST_NAME, FULL_BST_NAME]},
        'index_sha256': file_hash(FOLDER / 'index.json'),
        'data': 'Released already normalized joints/positions/shuttle; no invented court coordinates; author padding/bones; seq100 merged25.',
        'selection': 'All validation rows, fixed checkpoints, no tuning or filtering.',
        'test_split_inference': False, 'production_changed': False,
        'paper_comparison_limit': 'Paper table2 is test; this released-checkpoint validation replay is not a reproduction of its test average.',
        'script_sha256': file_hash(Path(__file__)),
    }
    atomic_json(OUT / 'protocol.json', protocol)
    torch.set_num_threads(2)
    classifier = public_classifier()
    full = BST_CG_AP(in_dim=72, seq_len=100, n_class=25,
                     depth_tem=2, depth_inter=1).eval()
    full.load_state_dict(torch.load(ROOT / 'models/bst' / FULL_BST_NAME,
                                   map_location='cpu', weights_only=True), strict=True)
    classes = classifier.classes
    labels = np.asarray([classes.index(r['class']) for r in rows])
    predictions = {'bst0': [], 'bst_cg_ap': []}
    data_digest = hashlib.sha256()
    sample_hashes = []
    started = time.perf_counter()
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        packed = [values(classifier, r['base']) for r in batch]
        j, p, s = [np.stack([v[k] for v in packed]) for k in range(3)]
        n = np.asarray([v[3] for v in packed], dtype='int64')
        for r, v in zip(batch, packed):
            digest = hashlib.sha256()
            for a in [v[0], v[1], v[2], np.asarray(v[3], dtype='int64')]:
                digest.update(np.asarray(a).tobytes())
            sample_hashes.append({'base': r['base'], 'input_sha256': digest.hexdigest()})
            data_digest.update(digest.digest())
        jj, pp, ss, nn = map(torch.from_numpy, [j, p, s, n])
        with torch.inference_mode():
            predictions['bst0'].append(classifier.model(jj, ss, nn).numpy())
            predictions['bst_cg_ap'].append(full(jj, ss, pp, nn).numpy())
        done = start + len(batch)
        if done % 256 == 0 or done == len(rows):
            atomic_json(OUT / 'progress.json', {'processed': done, 'total': len(rows),
                        'elapsed_seconds': round(time.perf_counter() - started, 3)})
            print('AUTHOR_VAL_VARIANTS', done, '/', len(rows), flush=True)
    logits = {name: np.concatenate(parts) for name, parts in predictions.items()}
    checks = {}
    for name, scores in logits.items():
        result = metrics(scores, labels, classes)
        result['joint_top2_correct'] = int(np.any(np.argsort(-scores, axis=1)[:, :2]
                                                 == labels[:, None], axis=1).sum())
        checks[name] = result
    # Compare the BST-0 replay against the previously saved author-val logits,
    # aligning by exact author base path rather than assuming identical order.
    cache = FOLDER / '本地继续训练/冻结骨干特征'
    prior_rows = json.loads((cache / 'rows.json').read_text())
    if isinstance(prior_rows, dict):
        prior_rows = prior_rows['rows']
    prior = np.load(cache / 'teacher_logits.npy', mmap_mode='r')
    lookup = {r['base']: i for i, r in enumerate(prior_rows) if r['split'] == 'val'}
    assert set(lookup) == {r['base'] for r in rows}
    reference = np.stack([prior[lookup[r['base']]] for r in rows])
    error = float(np.max(abs(reference - logits['bst0'])))
    assert np.array_equal(reference.argmax(1), logits['bst0'].argmax(1))
    np.savez_compressed(OUT / 'predictions.npz', labels=labels,
                        bst0_logits=logits['bst0'], full_logits=logits['bst_cg_ap'],
                        bases=np.asarray([r['base'] for r in rows]), class_labels=np.asarray(classes))
    atomic_json(OUT / 'input_manifest.json', {'samples': sample_hashes,
                'prepared_input_sha256': data_digest.hexdigest()})
    atomic_json(OUT / 'result.json', {'protocol': protocol, 'checks': checks,
                'bst0_reference_top1_identical': True,
                'bst0_reference_max_logit_error': error,
                'predictions_sha256': file_hash(OUT / 'predictions.npz'),
                'input_manifest_sha256': file_hash(OUT / 'input_manifest.json'),
                'elapsed_seconds': time.perf_counter() - started})
    print(json.dumps({name: {k: v for k, v in result.items()
                     if k in ['total', 'stroke_top1_correct', 'player_top1_correct',
                              'joint_25class_correct', 'joint_top2_correct']}
                      for name, result in checks.items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
