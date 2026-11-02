"""Continue the author's BST classification head using ONLY author train data.

Frozen pretrained backbone, original 25-class head initialisation, public val
checkpoint selection, untouched public test. No personal videos/labels read.
"""
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from analysis_jobs import DATA, atomic_json, file_hash
from bst_public_data import FOLDER, values, public_classifier
from specialist_model import ROOT, BST_NAME

VERSION = 'author-only-bst-head-continuation-v1'
OUTPUT = DATA/'公开训练数据/BST_ShuttleSet/本地继续训练'
CACHE = OUTPUT/'冻结骨干特征'
STATE = OUTPUT/'state.json'
SEED = 42


def state(status, message, **extra):
    atomic_json(STATE, {'version': VERSION, 'status': status, 'message': message,
                        'updated_at': datetime.now(timezone.utc).isoformat(),
                        'personal_training_samples': 0, 'public_test_samples_used': 0, **extra})


def encode(classifier, batch):
    captured = []
    hook = classifier.model.mlp_head.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().cpu().numpy().copy()))
    try:
        js, ss, lengths = [], [], []
        for row in batch:
            j, _, s, n = values(classifier, row['base'])
            js.append(j);ss.append(s);lengths.append(n)
        with torch.inference_mode():
            logits = classifier.model(torch.from_numpy(np.stack(js)), torch.from_numpy(np.stack(ss)),
                                      torch.tensor(lengths)).cpu().numpy()
    finally:
        hook.remove()
    x = captured[0]
    if x.shape != (len(batch), 300) or not np.isfinite(x).all() or not np.isfinite(logits).all():
        raise ValueError('公开输入的特征或分类分数异常')
    return x, logits


def features(classifier, batch_size=8):
    index = json.loads((FOLDER/'index.json').read_text())
    rows = [r for r in index['samples'] if r['split'] in ['train', 'val']]
    # Original public split is retained. The test arrays are not opened.
    identity = {'version': VERSION, 'source_archive_sha256': index['source_archive_sha256'],
                'model_sha256': file_hash(ROOT/'models/bst'/BST_NAME),
                'input_adapter_sha256': file_hash(ROOT/'bst_public_data.py'),
                'classes': classifier.classes, 'count': len(rows),
                'row_ids_sha256': __import__('hashlib').sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}
    CACHE.mkdir(parents=True, exist_ok=True)
    progress_file = CACHE/'progress.json'
    progress = json.loads(progress_file.read_text()) if progress_file.is_file() else None
    if progress and progress['identity'] != identity:
        raise ValueError('缓存身份改变，请保留旧缓存后使用新的特征目录')
    if not progress:
        np.lib.format.open_memmap(CACHE/'features.npy', mode='w+', dtype='float32', shape=(len(rows),300)).flush()
        np.lib.format.open_memmap(CACHE/'teacher_logits.npy', mode='w+', dtype='float32', shape=(len(rows),25)).flush()
        np.save(CACHE/'labels.npy', np.asarray([classifier.classes.index(r['class']) for r in rows], dtype='int64'))
        np.save(CACHE/'splits.npy', np.asarray([r['split'] for r in rows]))
        atomic_json(CACHE/'rows.json', {'rows': rows})
        progress = {'identity': identity, 'completed': 0}
        atomic_json(progress_file, progress)
    x = np.lib.format.open_memmap(CACHE/'features.npy', mode='r+')
    logits = np.lib.format.open_memmap(CACHE/'teacher_logits.npy', mode='r+')
    started = time.perf_counter()
    initial = progress['completed']
    for i in range(initial, len(rows), batch_size):
        end = min(i+batch_size, len(rows))
        xx, pp = encode(classifier, rows[i:end]);x[i:end] = xx;logits[i:end] = pp
        if end%128 == 0 or end == len(rows):
            x.flush();logits.flush();progress['completed'] = end
            atomic_json(progress_file, progress)
            state('extracting', '只从作者训练与验证数据提取冻结骨干特征', processed=end, total=len(rows),
                  elapsed_seconds=round(time.perf_counter()-started, 1))
            print('PUBLIC_FEATURES', end, '/', len(rows), 'seconds', round(time.perf_counter()-started, 1), flush=True)
    return np.asarray(x).copy(), np.asarray(logits).copy(), np.load(CACHE/'labels.npy'), np.load(CACHE/'splits.npy'), identity


def metrics(logits, y):
    guessed = logits.argmax(1)
    correct = guessed == y
    f1, details = [], {}
    for c in range(25):
        tp = int(((guessed == c)&(y == c)).sum())
        actual = int((y == c).sum());predicted = int((guessed == c).sum())
        value = 2*tp/(actual+predicted) if actual+predicted else 0.
        f1.append(value)
        details[str(c)] = {'total': actual, 'correct': tp, 'f1': value}
    t = torch.from_numpy(np.asarray(logits))
    return {'correct': int(correct.sum()), 'total': len(y), 'accuracy': float(correct.mean()),
            'macro_f1': float(np.mean(f1)), 'cross_entropy': float(F.cross_entropy(t, torch.from_numpy(y))),
            'per_class': details}


def predict(head, x, batch_size=512):
    head.eval()
    with torch.inference_mode():
        return np.concatenate([head(x[i:i+batch_size]).cpu().numpy() for i in range(0,len(x),batch_size)])


def train(epochs=10, batch_size=8):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT/'.train.lock').open('a+b')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
    run_id = uuid.uuid4().hex
    run = OUTPUT/'运行记录'/run_id;run.mkdir(parents=True)
    shutil.copy2(ROOT/'models/bst'/BST_NAME, run/'baseline.pt')
    shutil.copy2(Path(__file__), run/'public_bst_training.py')
    shutil.copy2(ROOT/'bst_public_data.py', run/'bst_public_data.py')
    state('extracting', '正在仅用作者数据训练；本轮不用个人标注', run_id=run_id)
    classifier = public_classifier()
    original = deepcopy(classifier.model.state_dict())
    x, teacher, y, splits, identity = features(classifier, batch_size)
    from benchmarks.isolation import assert_training_allowed
    assert_training_allowed(identity)
    tr, va = splits == 'train', splits == 'val'
    train_x = torch.from_numpy(x[tr]);train_y = torch.from_numpy(y[tr])
    train_teacher = torch.from_numpy(teacher[tr])
    val_x = torch.from_numpy(x[va]);val_y = y[va]
    base = metrics(teacher[va], val_y)
    # Start from the released classification head, not a random replacement.
    head = classifier.model.mlp_head
    anchor = [p.detach().clone() for p in head.parameters()]
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-4)
    history = [];best = None;best_parameters = None;best_epoch = None
    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(1,epochs+1):
        head.train();total_loss = 0.
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0,len(order),256):
            idx = order[start:start+256]
            logits = head(train_x[idx])
            ce = F.cross_entropy(logits, train_y[idx])
            # The source labels train the head; the teacher anchors useful old decisions.
            temperature = 2.
            kd = F.kl_div(F.log_softmax(logits/temperature, dim=1),
                          F.softmax(train_teacher[idx]/temperature, dim=1), reduction='batchmean')*temperature**2
            distance = sum((p-a).square().mean() for p,a in zip(head.parameters(), anchor))
            loss = ce+.2*kd+.01*distance
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            optimizer.step();total_loss += loss.item()*len(idx)
        current = metrics(predict(head,val_x),val_y)
        row = {'epoch':epoch,'train_loss':total_loss/len(train_x),'validation':current}
        history.append(row)
        if best is None or (current['correct'], current['macro_f1'], -current['cross_entropy']) > (best['correct'],best['macro_f1'],-best['cross_entropy']):
            best = current;best_epoch = epoch;best_parameters = deepcopy(head.state_dict())
        atomic_json(run/'epochs.json', history)
        state('training', '仅用作者训练数据更新BST分类层；用作者验证集选择版本', run_id=run_id,
              epoch=epoch, epochs=epochs, public_train_samples=int(tr.sum()), public_val_samples=int(va.sum()),
              baseline_validation={'correct':base['correct'],'total':base['total']},
              candidate_validation={'correct':current['correct'],'total':current['total']})
        print('PUBLIC_EPOCH', epoch, 'baseline', base['correct'], 'candidate', current['correct'], '/',len(val_y), flush=True)
    head.load_state_dict(best_parameters);head.eval()
    final_weights = classifier.model.state_dict()
    if any(not torch.equal(value, final_weights[name]) for name,value in original.items() if not name.startswith('mlp_head.')):
        raise ValueError('冻结骨干意外改变')
    changed = sum(not torch.equal(value, final_weights[name]) for name,value in original.items())
    if changed == 0:
        raise ValueError('训练未改变任何参数')
    torch.save(final_weights, run/'candidate.pt')
    np.savez_compressed(run/'validation_predictions.npz', baseline_logits=teacher[va],
                        candidate_logits=predict(head,val_x), labels=val_y)
    result = {'version':VERSION,'run_id':run_id,'created_at':datetime.now(timezone.utc).isoformat(),
              'public_train_samples':int(tr.sum()),'public_val_samples':int(va.sum()),
              'public_test_samples_used':0,'personal_training_samples':0,'personal_validation_samples':0,
              'feature_identity':identity,'initial_model_sha256':file_hash(run/'baseline.pt'),
              'candidate_model_sha256':file_hash(run/'candidate.pt'), 'backbone_frozen':True,
              'classification_head_continued_from_pretrained':True, 'changed_parameter_tensors':changed,
              'epochs':epochs,'selected_epoch':best_epoch,'seed':SEED,'optimizer':'AdamW',
              'learning_rate':1e-4,'loss':'CE + 0.2 KL(teacher, T=2) + 0.01 parameter anchor',
              'baseline_validation':base,'candidate_validation':best,
              'public_validation_accuracy_improved':best['correct']>base['correct'],
              'selection_uses_personal_labels':False,'active_model_replaced':False,
              'candidate_checkpoint':str((run/'candidate.pt').relative_to(DATA)),
              'baseline_checkpoint':str((run/'baseline.pt').relative_to(DATA)),
              'code_sha256':file_hash(Path(__file__)),
              'scope':'Author split internal development validation; checkpoint already learned this source. Private accuracy awaits new human labels.'}
    atomic_json(run/'manifest.json',result);atomic_json(OUTPUT/'latest.json',result)
    state('candidate_ready','仅作者数据训练已完成；等待新个人片段的人工核对',
          run_id=run_id,public_train_samples=int(tr.sum()),public_val_samples=int(va.sum()),
          baseline_validation={'correct':base['correct'],'total':base['total']},
          candidate_validation={'correct':best['correct'],'total':best['total']},
          public_validation_accuracy_improved=result['public_validation_accuracy_improved'])
    print(json.dumps({'run_id':run_id,'baseline':base['correct'],'candidate':best['correct'],
                      'val_total':len(val_y),'personal_training_samples':0},ensure_ascii=False),flush=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=8);args = parser.parse_args()
    try:
        train(args.epochs,args.batch_size)
    except Exception as error:
        state('failed','公开数据训练未完成：'+str(error)[:400]);raise
