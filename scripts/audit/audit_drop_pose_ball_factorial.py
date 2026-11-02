"""Cross restored missing poses with the frozen seven shuttle conditions."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/audit/
from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analysis_jobs import atomic_json,file_hash
from audit_researcher_ball_25p3 import BASE,OUT as BALL_CONTROL,RUN,DENSE,BALL,EVENT
from input_quality import InputQuality
from specialist_bst import SpecialistBST

EXPECTED_FRAMES=[751,752,753,756,757]


def main(restored_path,out):
    if (out/'result.json').exists():raise FileExistsError('Preserve completed factorial')
    out.mkdir(parents=True,exist_ok=True)
    original=json.loads(DENSE.read_text());restored=json.loads(restored_path.read_text());ball=json.loads(BALL.read_text())
    assert original['source_sha256']==restored['source_sha256'] and len(original['frames'])==len(restored['frames'])
    added=[]
    for a,b in zip(original['frames'],restored['frames']):
        assert {k:v for k,v in a.items() if k!='players'}=={k:v for k,v in b.items() if k!='players'}
        for side in ['near','far']:
            p,q=a['players'][side],b['players'][side]
            if p is not None:assert p==q
            elif q is not None:
                assert side=='far';added.append(a['frame_index'])
    assert added==EXPECTED_FRAMES
    old=json.loads((BALL_CONTROL/'result.json').read_text())
    old_head=json.loads((BALL_CONTROL/'prior41_head/result.json').read_text())
    inputs=sorted(BALL_CONTROL.glob('*_inputs.npz'));assert len(inputs)==49
    paths=[DENSE,BALL,restored_path,BASE/'pose_missing_25p3/detected_boxes.json',
           BALL_CONTROL/'result.json',BALL_CONTROL/'logits.npz',BALL_CONTROL/'prior41_head/result.json',
           BALL_CONTROL/'prior41_head/logits.npz',RUN/'baseline.pt',RUN/'candidate.pt',Path(__file__),*inputs]
    hashes={str(p):file_hash(p) for p in paths}
    protocol={'declared_at':datetime.now(timezone.utc).isoformat(),'input_sha256':hashes,
        'factors':{'pose':['original','five_original_detector_box_poses_restored'],
                   'ball':'All seven already frozen original/matched-missing/manual-nominal/4-radius variants',
                   'window':'All seven previously frozen windows with complete manual ball-review coverage',
                   'model':['original released BST-0','prior41-only frozen head']},
        'design':'Only insert original missing five far entries, never overwrite observed joints. Keep original frame times/person axis and original normalizedS from prior packed files exactly. Within each fixed pose/window J/P/N stay identical across all sevenS conditions; onlyJ may change between original/restored pose. No new ball/window/weight selection.',
        'limits':'Restored poses are same production BGR RTMPose estimates from source-reviewed originalYOLOX boxes, not ground-truth joints. This is one preselected developer failure. Manual ball is estimated and one frame remains unknown. Not full automatic performance.',
        'no_training_or_production_activation':True}
    atomic_json(out/'protocol.json',protocol)
    torch.set_num_threads(2)
    models={name:SpecialistBST(original,ball,checkpoint=path) for name,path in [('baseline',RUN/'baseline.pt'),('candidate',RUN/'candidate.pt')]}
    order=models['baseline'].order.copy();packer=models['baseline']
    records=json.loads((RUN/'personal_predictions.json').read_text())['baseline'];record=next(r for r in records if r['event_id']==EVENT)
    target=int(record['target_class_id']);actor=record['target_class_label'].split('_',1)[0]
    allowed=np.asarray([0]+[i for i,c in enumerate(packer.classes) if c.startswith(actor+'_')]);assert len(allowed)==13
    base_cache=np.load(BALL_CONTROL/'logits.npz');head_cache=np.load(BALL_CONTROL/'prior41_head/logits.npz')
    lookup={r['window']['name']+'__'+r['ball']:r for r in old['rows']}
    restored_inputs={};saved={};rows=[]
    for path in inputs:
        key=path.name.removesuffix('_inputs.npz');previous=lookup[key];window=previous['window'];name=window['name']
        old_values=np.load(path);n=int(old_values['n'])
        if name not in restored_inputs:
            event=dict(record['model_window']);lo,hi=window['lo'],window['hi']
            event.update(start_frame=lo,end_frame=hi,start=original['frames'][lo]['source_time'],end=original['frames'][hi+1]['source_time'])
            packer.dense=original;packer.order=order;packer.coordinate_quality=InputQuality(original,ball)
            current,_=packer.inputs(event)
            reference=np.load(BALL_CONTROL/(name+'__original_ball_inputs.npz'))
            assert all(np.array_equal(a,reference[k]) for a,k in zip(current,['j','p','s','n']))
            packer.dense=restored;packer.order=order;packer.coordinate_quality=InputQuality(restored,ball)
            values,q=packer.inputs(event)
            assert np.array_equal(values[1],reference['p']) and np.array_equal(values[2],reference['s']) and values[3]==int(reference['n'])
            assert q['paired_player_fraction']==1.
            restored_inputs[name]=(values,q)
            np.savez_compressed(out/(name+'_restored_pose.npz'),j=values[0],p=values[1],n=np.asarray(values[3],dtype='int64'))
        new_values,new_quality=restored_inputs[name]
        assert int(new_values[3])==n
        for pose in ['original','restored_five']:
            j=old_values['j'] if pose=='original' else new_values[0]
            s=old_values['s']
            for method,model in models.items():
                with torch.inference_mode():scores=model.model(torch.from_numpy(j)[None],torch.from_numpy(s)[None],torch.tensor([n])).numpy()[0]
                if pose=='original':
                    cache=base_cache if method=='baseline' else head_cache
                    assert np.max(abs(scores-cache[key]))<1e-4
                pred=int(scores.argmax());pc=int(allowed[np.argmax(scores[allowed])])
                runkey=key+'__'+pose+'__'+method;saved[runkey]=scores
                rows.append({'key':runkey,'window':window,'ball':previous['ball'],'pose':pose,'model':method,
                    'true_label':record['target_class_label'],'prediction':model.classes[pred],'joint_correct':pred==target,
                    'actor13_prediction':model.classes[pc],'actor13_correct':pc==target,
                    'true_margin25':float(scores[target]-scores[np.arange(25)!=target].max()),
                    'paired_fraction':previous['pair_fraction'] if pose=='original' else new_quality['paired_player_fraction'],
                    'n':n,'canonical_input_sha256':hashlib.sha256(j.tobytes()+s.tobytes()+np.asarray(n,dtype='int64').tobytes()).hexdigest()})
    summary={}
    for pose in ['original','restored_five']:
        summary[pose]={}
        for method in models:
            summary[pose][method]={}
            for variant in old['summary']:
                selected=[r for r in rows if r['pose']==pose and r['model']==method and r['ball']==variant]
                assert len(selected)==7
                summary[pose][method][variant]={'windows':7,'joint_correct':sum(r['joint_correct'] for r in selected),
                    'actor13_correct':sum(r['actor13_correct'] for r in selected),'predictions':sorted({r['prediction'] for r in selected})}
    assert len(rows)==196 and all(file_hash(Path(p))==h for p,h in hashes.items())
    np.savez_compressed(out/'logits.npz',**saved)
    atomic_json(out/'result.json',{'created_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':file_hash(out/'protocol.json'),
        'summary':summary,'rows':rows,'new_pose_entries':EXPECTED_FRAMES,'old_saved_controls_reproduced':True,
        'production_or_labels_or_weights_changed':False,'scope':protocol['limits']})
    print(json.dumps({'restored_five':summary['restored_five'],'rows':len(rows)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--restored-dense',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();main(args.restored_dense.resolve(),args.out.resolve())
