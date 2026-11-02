"""Evaluate the prior41-only head on the already frozen shuttle intervention inputs."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/audit/
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from analysis_jobs import atomic_json,file_hash
from audit_researcher_ball_25p3 import BASE,OUT,RUN,DENSE,BALL,EVENT
from specialist_bst import SpecialistBST


def main():
    folder=OUT/'prior41_head'
    if (folder/'result.json').exists():raise FileExistsError('Preserve completed head sensitivity')
    folder.mkdir(parents=True,exist_ok=True)
    previous=json.loads((OUT/'result.json').read_text())
    input_files=sorted(OUT.glob('*_inputs.npz'));assert len(input_files)==49
    paths=[OUT/'protocol.json',OUT/'result.json',OUT/'logits.npz',RUN/'candidate.pt',RUN/'baseline.pt',
           RUN/'personal_predictions.json',RUN/'personal_predictions.npz',DENSE,BALL,Path(__file__),*input_files]
    hashes={str(p):file_hash(p) for p in paths}
    protocol={'declared_at':datetime.now(timezone.utc).isoformat(),'input_sha256':hashes,
        'question':'Does the already frozen41-only head require more faithful ball inputs to resolve this held-out-source developer example?',
        'weights':'Exactly prior run582817... candidate.pt trained41 rows from another source, fixed epoch40. Not the model fit on all71. Check only6 mlp_head tensors differ from baseline.',
        'selection':'All49 prior fixed inputs, no further coordinates, windows, thresholds or training changes.',
        'reference':'Compare with original model saved logits and exact candidate cached logits at original80+original ball.',
        'scope':'Repeatedly inspected developer case and research estimated ball coordinates, not new-video accuracy. Unchanged global25class and human actor13 including unknown are separate.'}
    atomic_json(folder/'protocol.json',protocol)
    base=torch.load(RUN/'baseline.pt',map_location='cpu',weights_only=True)
    candidate=torch.load(RUN/'candidate.pt',map_location='cpu',weights_only=True)
    changed=[k for k in base if not torch.equal(base[k],candidate[k])]
    assert len(changed)==6 and all(k.startswith('mlp_head.') for k in changed)
    torch.set_num_threads(2)
    model=SpecialistBST(json.loads(DENSE.read_text()),json.loads(BALL.read_text()),checkpoint=RUN/'candidate.pt')
    records=json.loads((RUN/'personal_predictions.json').read_text())['baseline'];ri=next(i for i,r in enumerate(records) if r['event_id']==EVENT)
    record=records[ri];target=int(record['target_class_id']);actor=record['target_class_label'].split('_',1)[0]
    allowed=np.asarray([0]+[i for i,c in enumerate(model.classes) if c.startswith(actor+'_')]);assert len(allowed)==13
    cache=np.load(RUN/'personal_predictions.npz');lookup={r['window']['name']+'__'+r['ball']:r for r in previous['rows']}
    rows=[];saved={}
    for path in input_files:
        name=path.name.removesuffix('_inputs.npz');values=np.load(path);j,s,n=values['j'],values['s'],int(values['n'])
        with torch.inference_mode():logits=model.model(torch.from_numpy(j)[None],torch.from_numpy(s)[None],torch.tensor([n])).numpy()[0]
        if name=='original80__original_ball':
            error=float(np.max(abs(logits-cache['candidate_logits'][ri])));assert error<1e-4
        pred=int(logits.argmax());pc=int(allowed[np.argmax(logits[allowed])]);old=lookup[name]
        rows.append({'name':name,'window':old['window'],'ball':old['ball'],'true_label':record['target_class_label'],
            'original_prediction':old['prediction'],'original_joint_correct':old['joint_correct'],
            'candidate_prediction':model.classes[pred],'candidate_joint_correct':pred==target,
            'candidate_actor13_prediction':model.classes[pc],'candidate_actor13_correct':pc==target,
            'candidate_margin25':float(logits[target]-logits[np.arange(25)!=target].max())})
        saved[name]=logits
    summary={ball:{'windows':7,'candidate_joint_correct':sum(r['candidate_joint_correct'] for r in rows if r['ball']==ball),
        'candidate_actor13_correct':sum(r['candidate_actor13_correct'] for r in rows if r['ball']==ball),
        'predictions':sorted({r['candidate_prediction'] for r in rows if r['ball']==ball})} for ball in previous['summary']}
    assert all(file_hash(Path(p))==h for p,h in hashes.items())
    np.savez_compressed(folder/'logits.npz',**saved)
    atomic_json(folder/'result.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':file_hash(folder/'protocol.json'),'rows':rows,'summary':summary,
        'candidate_baseline_max_cache_error':error,'changed_checkpoint_tensors':changed,
        'no_training_or_production_changes':True,'scope':protocol['scope']})
    print(json.dumps({'summary':summary,'baseline_cache_error':error},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
