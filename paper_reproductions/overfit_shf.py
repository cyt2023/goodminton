"""Small balanced training-set memorization check, never a generalization score."""
import argparse,json,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from .shuttlehitformer import ShuttleHitFormer,Config,center_windows
from .data import write_json,sha256
from .run import seed

def balanced_examples(manifest,per_class=6):
    from benchmarks.isolation import assert_training_allowed
    assert_training_allowed(manifest)
    by_class={i:[] for i in range(3)}
    for row in manifest['rows']:
        if row['split']!='train':continue
        with np.load(row['path']) as z:features=torch.from_numpy(z['features']);labels=z['labels']
        valid=np.arange(3,len(labels)-4);gt=np.flatnonzero(labels>0)
        distance=np.min(abs(valid[:,None]-gt[None,:]),1) if len(gt) else np.full(len(valid),999)
        for label in range(3):
            candidates=valid[(labels[valid]==label)&((distance>=3) if label==0 else np.ones(len(valid),bool))]
            # Evenly spread within rally; no validation or test data inspected.
            candidates=candidates[np.linspace(0,len(candidates)-1,min(per_class,len(candidates))).astype(int)] if len(candidates) else []
            for center in candidates:
                by_class[label].append((features[center-3:center+5],label,{'rally':row['id'],'center':int(center),'label':label}))
    selected=[]
    for label,rows in by_class.items():
        if len(rows)<per_class:raise ValueError('Not enough examples in training split')
        selected.extend(rows[:per_class])
    return torch.stack([r[0] for r in selected]),torch.tensor([r[1] for r in selected]),[r[2] for r in selected]

def main():
    a=argparse.ArgumentParser();a.add_argument('--manifest',required=True);a.add_argument('--out',required=True);a.add_argument('--config')
    a.add_argument('--steps',type=int,default=1600);a.add_argument('--lr',type=float,default=3e-4);a.add_argument('--per-class',type=int,default=6)
    args=a.parse_args();seed(20261030);manifest=json.loads(Path(args.manifest).read_text());x,y,examples=balanced_examples(manifest,args.per_class)
    c=Config(**json.loads(Path(args.config).read_text())) if args.config else Config(grounding_residual=True)
    m=ShuttleHitFormer(c);optim=torch.optim.AdamW(m.parameters(),lr=args.lr,weight_decay=5e-4);history=[];start=time.perf_counter();passed=False
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optim,args.steps,eta_min=1e-6)
    for step in range(args.steps):
        m.train();optim.zero_grad();out=m(x);loss=m.loss(out,y);loss['loss'].backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);optim.step();schedule.step()
        if step%25==0 or step==args.steps-1:
            m.eval()
            with torch.no_grad():p=m(x)['logits'].softmax(-1);ce=torch.nn.functional.cross_entropy(p.log(),y);accuracy=float((p.argmax(-1)==y).float().mean())
            row={'step':step+1,'accuracy_on_training_examples':accuracy,'classification_loss':float(ce),'minimum_correct_class_probability':float(p[torch.arange(len(y)),y].min())}
            history.append(row);print(json.dumps(row),flush=True)
            if accuracy==1. and float(ce)<.1:passed=True;break
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.save({'config':asdict(c),'state_dict':m.state_dict(),'examples':examples,'scope':'training_memorization_check_not_generalization'},out/'model.pt')
    write_json(out/'result.json',{'passed':passed,'history':history,'steps':step+1,'seconds':time.perf_counter()-start,
        'examples':examples,'manifest_sha256':sha256(args.manifest),'config':asdict(c),'interpretation':'Training examples only; not held-out accuracy or event F1'})
    print('OVERFIT_RESULT',passed,step+1,time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
