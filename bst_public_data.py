"""Audit author arrays and prepare a bounded real-data development replay.

No videos or pseudo-labels are generated. Author train/val/test names retained.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from types import SimpleNamespace
import numpy as np
import torch

from analysis_jobs import DATA, atomic_json, file_hash
from personal_learning import embedding
from specialist_model import ROOT, BST_NAME
from specialist_bst import BST_0, author_helpers

FOLDER = DATA/'公开训练数据/BST_ShuttleSet'
ARCHIVE = FOLDER/'作者预处理_25类_相邻击球.rar'
ARRAYS = FOLDER/'dataset_npy_between_2_hits_with_max_limits'
FILE_ID = '1k_6yFUh3xgF6emSyWc-qcTvf_0cRghzq'
VERSION = 'author-public-bst-replay-v1'


def public_classifier():
    """Load the author checkpoint without reading any personal video or labels."""
    h = author_helpers()
    model = BST_0(in_dim=72, seq_len=100, n_class=25, depth_tem=2, depth_inter=1).eval()
    model.load_state_dict(torch.load(ROOT/'models/bst'/BST_NAME, map_location='cpu', weights_only=True), strict=True)
    return SimpleNamespace(helpers=h, classes=h['get_merged_stroke_types'](),
                           bones=h['get_bone_pairs'](), model=model, order=['top', 'bottom'])


def audit():
    rows, counts = [], {}
    for split in ['train', 'val', 'test']:
        counts[split] = Counter()
        for f in sorted((ARRAYS/split).glob('*/*_joints.npy')):
            base = str(f)[:-len('_joints.npy')]
            if not Path(base+'_pos.npy').is_file() or not Path(base+'_shuttle.npy').is_file():
                raise ValueError('作者样本缺少对应模态：'+str(f))
            rows.append({'split': split, 'class': f.parent.name, 'sample_id': f.stem[:-len('_joints')],
                         'base': str(Path(base).relative_to(ARRAYS))})
            counts[split][f.parent.name] += 1
    if len(rows) != 33481:
        raise ValueError('作者解压样本数未完成或不匹配')
    result = {'version': VERSION, 'download_completed': True, 'array_triplets_verified': True,
              'source_url': f'https://drive.google.com/file/d/{FILE_ID}/view',
              'author_repository': 'https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer',
              'author_readme_sha256': file_hash(ROOT/'vendor/BST/README.md'),
              'archive_sha256': file_hash(ARCHIVE), 'archive_bytes': ARCHIVE.stat().st_size,
              'sample_count': len(rows), 'split_counts': {s: sum(c.values()) for s, c in counts.items()},
              'per_class': {s: dict(c) for s, c in counts.items()},
              'label_source': 'author_released_preprocessed_dataset', 'model_trained_on_all_downloaded_samples': False,
              'format': 'normalized variable-length joints/positions/shuttle NPY; no match media',
              'license_files': ['04_原型/vendor/BST/LICENSE', '03_数据/职业示范索引/ShuttleSet_BST/ShuttleSet_LICENSE.txt'],
              'created_at': datetime.now(timezone.utc).isoformat()}
    atomic_json(FOLDER/'index.json', {'samples': rows, 'source_archive_sha256': result['archive_sha256']})
    atomic_json(FOLDER/'manifest.json', result)
    return rows, result


def values(classifier, base):
    arrays = [np.load(str(ARRAYS/base)+suffix, allow_pickle=False) for suffix in ['_joints.npy', '_pos.npy', '_shuttle.npy']]
    j, p, s = arrays
    n = len(j)
    if not (j.shape == (n, 2, 17, 2) and p.shape == (n, 2, 2) and s.shape == (n, 2)
            and n > 0 and all(np.isfinite(a).all() for a in arrays)):
        raise ValueError('公开样本模态形状或数值异常')
    # These files are already author-normalized; never normalize them twice.
    h = classifier.helpers
    j, p, s, length = h['make_seq_len_same'](100, j, p, s)
    j = np.concatenate([j, h['create_bones'](j, classifier.bones)], axis=-2).reshape(100, 2, 72)
    return j.astype('float32'), p.astype('float32'), s.astype('float32'), length


def prepare(per_train=16, per_val=8):
    rows, manifest = audit()
    classifier = public_classifier()
    torch.set_num_threads(1)
    classes = classifier.classes
    selected = []
    for split, number in [('train', per_train), ('val', per_val)]:
        for name in classes:
            pool = [r for r in rows if r['split'] == split and r['class'] == name]
            rng = random.Random(split+':'+name+':public-replay-v1')
            rng.shuffle(pool); selected.extend(pool[:number])
    raw, logits, targets = [], [], []
    for i, item in enumerate(selected):
        v = values(classifier, item['base'])
        vec = embedding(classifier, v, classifier.order[0])
        with torch.inference_mode():
            prediction = classifier.model.mlp_head(torch.from_numpy(vec[None])).numpy()[0]
        raw.append(vec); logits.append(prediction);targets.append(classes.index(item['class']))
        if (i+1)%100 == 0:
            print('PUBLIC_REPLAY', i+1, '/', len(selected), flush=True)
    logits = np.asarray(logits);targets = np.asarray(targets)
    splits = np.asarray([r['split'] for r in selected])
    out = FOLDER/'开发样本'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'features.npz', embeddings=np.asarray(raw), logits=logits, labels=targets, splits=splits,
                        class_labels=np.asarray(classes), sample_ids=np.asarray([r['sample_id'] for r in selected]))
    summary = {'version': VERSION, 'samples': selected, 'array_normalization': 'already author normalized',
               'source_archive_sha256': manifest['archive_sha256'], 'model_sha256': file_hash(ROOT/'models/bst'/BST_NAME),
               'per_train_class_limit': per_train, 'per_val_class_limit': per_val,
               'model_retrained': False, 'test_split_read_for_inference': False,
               'code_sha256': file_hash(Path(__file__)), 'created_at': datetime.now(timezone.utc).isoformat(),
               'results': {s: {'correct': int((logits[splits == s].argmax(1) == targets[splits == s]).sum()),
                               'total': int((splits == s).sum())} for s in ['train', 'val']},
               'scope': 'Author dataset compatibility replay; released checkpoint may have used its validation split.'}
    atomic_json(out/'manifest.json', summary)
    print(json.dumps({'public_samples': manifest['sample_count'], 'replay': summary['results']}, ensure_ascii=False), flush=True)
    return summary


if __name__ == '__main__':
    prepare()
