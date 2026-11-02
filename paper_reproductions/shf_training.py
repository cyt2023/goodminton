"""Train-set sampling and held-out checkpoint selection for SHF local variants."""
import numpy as np
import torch
from .shuttlehitformer import center_windows


def load_rallies(rows):
    rallies=[]
    for row in rows:
        with np.load(row['path']) as z:features=torch.from_numpy(z['features']);labels=torch.from_numpy(z['labels']).long()
        if len(features)!=len(labels):raise ValueError('Unaligned training labels')
        rallies.append((row,features,labels))
    return rallies


def sample_epoch(rallies,config,rng):
    xs=[];ys=[]
    for row,features,labels in rallies:
        labels_np=labels.numpy();valid=np.arange(3,len(labels)-4);positive=valid[labels_np[valid]>0];gt=np.flatnonzero(labels_np>0)
        distance=np.min(abs(valid[:,None]-gt[None,:]),1) if len(gt) else np.full(len(valid),999)
        negative=valid[(labels_np[valid]==0)&(distance>=config.negative_gap)]
        centers=np.concatenate([positive,rng.choice(negative,min(len(negative),len(positive)*config.negative_ratio),replace=False)])
        if len(centers):xs.append(center_windows(features,centers));ys.append(labels[centers])
    if not xs:raise ValueError('No training windows')
    return torch.cat(xs),torch.cat(ys)


def selection_key(metrics,mode='f1'):
    if mode=='paper_recall':return (metrics['recall'],)
    # F1 penalizes emitting every possible event. Balanced CE resolves zero-F1
    # ties using all three classes, so all-no-hit does not freeze epoch1.
    return (round(metrics['f1'],8),-metrics['balanced_cross_entropy'],metrics['recall'])


def frame_diagnostics(scores,labels):
    if not scores:return {'balanced_cross_entropy':None,'frame_confusion':[[0]*3 for _ in range(3)],'mean_hit_probability_on_hits':None}
    indices=np.array([r['frame_index'] for r in scores]);prob=np.array([r['probabilities'] for r in scores],np.float64);target=labels[indices]
    loss=-np.log(np.maximum(prob[np.arange(len(target)),target],1e-12));by_class=[];confusion=np.zeros((3,3),dtype=int)
    np.add.at(confusion,(target,prob.argmax(-1)),1)
    for cls in range(3):
        mask=target==cls
        by_class.append({'class_id':cls,'count':int(mask.sum()),'negative_log_likelihood_sum':float(loss[mask].sum())})
    return {'per_class_loss':by_class,'frame_confusion':confusion.tolist(),
            'hit_probability_sum':float(prob[target>0,1:].sum()),'hit_frame_count':int((target>0).sum())}
