"""One frozen evaluation on the complete researcher-released BST test split.

Evaluates classification on author features, not raw-video event detection.
No fitting, threshold selection, candidate activation, or annotation writes.
"""
from collections import Counter
import copy
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

OUT = DATA / '外部标注测试/2026-10-21/BST_test3499'


def metrics(scores, targets, classes, rows):
    prediction = scores.argmax(1)
    names = np.asarray(classes)
    kind = np.asarray([c.split('_', 1)[-1] for c in classes])
    actor = np.asarray([c.split('_', 1)[0] if '_' in c else 'unknown' for c in classes])
    per_class = {}
    for i, name in enumerate(classes):
        mask = targets == i
        tp = int(((prediction == i) & mask).sum())
        fp = int(((prediction == i) & ~mask).sum())
        fn = int(((prediction != i) & mask).sum())
        per_class[name] = dict(total=int(mask.sum()), correct=tp,
                              precision=tp / (tp+fp) if tp+fp else 0,
                              recall=tp / (tp+fn) if tp+fn else 0,
                              f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    by_match = {}
    for mid in sorted({r['sample_id'].split('_')[0] for r in rows}, key=int):
        mask = np.asarray([r['sample_id'].split('_')[0] == mid for r in rows])
        by_match[mid] = dict(total=int(mask.sum()), correct=int((prediction[mask] == targets[mask]).sum()))
    confusion = np.zeros((25, 25), dtype='int64')
    np.add.at(confusion, (targets, prediction), 1)
    return {'total':len(targets), 'joint25_correct':int((prediction == targets).sum()),
            'joint25_accuracy':float((prediction == targets).mean()),
            'stroke_ignoring_actor_correct':int((kind[prediction] == kind[targets]).sum()),
            'actor_correct_including_unknown':int((actor[prediction] == actor[targets]).sum()),
            'top5_correct':int((np.argsort(-scores, axis=1)[:, -5:] == targets[:,None]).any(1).sum()),
            'macro_f1':float(np.mean([v['f1'] for v in per_class.values()])),
            'per_class':per_class, 'per_match':by_match, 'confusion25':confusion.tolist()}


def main():
    if (OUT/'protocol.json').exists():
        raise FileExistsError('Preserve the frozen test evaluation; resume only from its own state if interrupted')
    OUT.mkdir(parents=True, exist_ok=True)
    index = json.loads((FOLDER/'index.json').read_text())
    rows = sorted([r for r in index['samples'] if r['split'] == 'test'], key=lambda r:r['base'])
    assert len(rows) == len({r['base'] for r in rows}) == 3499
    train_ids = {r['sample_id'].split('_')[0] for r in index['samples'] if r['split'] == 'train'}
    val_ids = {r['sample_id'].split('_')[0] for r in index['samples'] if r['split'] == 'val'}
    test_ids = {r['sample_id'].split('_')[0] for r in rows}
    assert test_ids == {'39','40','42','43','44'} and not test_ids & (train_ids | val_ids)
    checkpoints = {
        'released_bst0':ROOT/'models/bst'/BST_NAME,
        'released_bst_full':ROOT/'models/bst'/FULL_BST_NAME,
        'author_train_only_mlp':FOLDER/'本地继续训练/运行记录/bc2366ef0c454be99a2a3d8c74d2069b/candidate.pt',
        'personal71_mlp':DATA/'击球标注/直接BST微调/运行记录/9f963c16e85a4267aa9968df4d97268b/candidate.pt',
    }
    dependencies = [Path(__file__),ROOT/'bst_public_data.py',ROOT/'specialist_bst.py',FOLDER/'index.json',*checkpoints.values()]
    protocol = {'frozen_at':datetime.now(timezone.utc).isoformat(), 'selection':'All3499 author-designated test rows, no sampling or removal.',
        'ground_truth':'Researcher-released human stroke/rally annotations carried by the official preprocessed feature dataset; not our personal labels.',
        'models':{k:{'path':str(p),'sha256':file_hash(p)} for k,p in checkpoints.items()},
        'dependencies':{str(p):file_hash(p) for p in dependencies},
        'test_match_ids':sorted(test_ids,key=int),'train_val_match_disjoint':True,
        'preprocessing':'Already normalized author J/P/S, unchanged author bones/pad/subsample seq100, no actor reordering, no renormalization.',
        'batch_size':16,'models_chosen_before_test':True,'training':False,'selection_on_test':False,
        'scope':'Conditional classification on official test inputs; does not evaluate our raw-video tracking/hit detection/cropping. Single released checkpoints, not paper multi-seed training reproduction. Test belongs to original model dataset and is not an independent amateur benchmark.',
        'output':'All logits, per class and per match, exact input hashes. Test results will not tune or activate any model.'}
    atomic_json(OUT/'protocol.json',protocol)
    torch.set_num_threads(2)
    classifier = public_classifier(); classes = classifier.classes
    full = BST_CG_AP(in_dim=72,seq_len=100,n_class=25,depth_tem=2,depth_inter=1).eval()
    full.load_state_dict(torch.load(checkpoints['released_bst_full'],map_location='cpu',weights_only=True),strict=True)
    base = classifier.model.state_dict(); heads = {}
    for name in ['author_train_only_mlp','personal71_mlp']:
        state = torch.load(checkpoints[name],map_location='cpu',weights_only=True)
        assert set(state)==set(base)
        changed = [k for k in state if not torch.equal(state[k],base[k])]
        assert changed and all(k.startswith('mlp_head.') for k in changed)
        head = copy.deepcopy(classifier.model.mlp_head).eval()
        head.load_state_dict({k[len('mlp_head.'):]:v for k,v in state.items() if k.startswith('mlp_head.')},strict=True)
        heads[name] = head
    captured = []
    hook = classifier.model.mlp_head.register_forward_pre_hook(lambda module,args:captured.append(args[0]))
    logits = {k:[] for k in checkpoints}; hashes = []; started=time.perf_counter()
    try:
        for start in range(0,len(rows),16):
            batch = rows[start:start+16]; packed=[values(classifier,r['base']) for r in batch]
            j,p,s=[torch.from_numpy(np.stack([v[k] for v in packed])) for k in range(3)]
            n=torch.tensor([v[3] for v in packed],dtype=torch.int64)
            for row,v in zip(batch,packed):
                d=hashlib.sha256()
                for a in [v[0],v[1],v[2],np.asarray(v[3],dtype='int64')]:d.update(a.tobytes())
                hashes.append({'base':row['base'],'input_sha256':d.hexdigest()})
            captured.clear()
            with torch.inference_mode():
                baseline=classifier.model(j,s,n); assert len(captured)==1
                logits['released_bst0'].append(baseline.numpy())
                logits['released_bst_full'].append(full(j,s,p,n).numpy())
                for name,head in heads.items(): logits[name].append(head(captured[0]).numpy())
            done=start+len(batch)
            if done%256==0 or done==len(rows):
                atomic_json(OUT/'progress.json',{'processed':done,'total':len(rows),'elapsed_seconds':time.perf_counter()-started})
                print('TEST',done,'/',len(rows),flush=True)
    finally: hook.remove()
    targets=np.asarray([classes.index(r['class']) for r in rows],dtype='int64')
    scores={k:np.concatenate(v) for k,v in logits.items()}
    results={k:metrics(v,targets,classes,rows) for k,v in scores.items()}
    np.savez_compressed(OUT/'predictions.npz',targets=targets,bases=np.asarray([r['base'] for r in rows]),class_labels=np.asarray(classes),**scores)
    atomic_json(OUT/'input_manifest.json',{'samples':hashes})
    assert all(file_hash(Path(p))==h for p,h in protocol['dependencies'].items())
    atomic_json(OUT/'result.json',{'completed_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':file_hash(OUT/'protocol.json'),
        'results':results,'predictions_sha256':file_hash(OUT/'predictions.npz'),'input_manifest_sha256':file_hash(OUT/'input_manifest.json'),
        'elapsed_seconds':time.perf_counter()-started,'no_weights_labels_or_production_modified':True,'scope':protocol['scope']})
    print(json.dumps({k:{f:v[f] for f in ['total','joint25_correct','joint25_accuracy','stroke_ignoring_actor_correct','macro_f1']} for k,v in results.items()},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
