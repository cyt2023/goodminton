"""Fixed RGB contract, missing-pose state, frozen shuttle/window factors."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/audit/
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
import torch

from analysis_jobs import atomic_json,file_hash
from audit_researcher_ball_25p3 import BASE,OUT as BALL_CONTROL,RUN,DENSE,BALL,EVENT
from input_quality import InputQuality
from specialist_bst import SpecialistBST

RGB=BASE/'pose_rgb_25p3'
OUT=BASE.parent/'2026-10-20_邻拍组合与关键帧恢复/drop_rgb_pose_ball'


def check_only_keypoints(reference,alternative):
    assert {k:v for k,v in reference.items() if k!='frames'}=={k:v for k,v in alternative.items() if k!='frames'}
    changed=[]
    for a,b in zip(reference['frames'],alternative['frames']):
        assert {k:v for k,v in a.items() if k!='players'}=={k:v for k,v in b.items() if k!='players'}
        for side in ['near','far']:
            p,q=a['players'][side],b['players'][side]
            assert (p is None)==(q is None)
            if p is None:continue
            assert {k:v for k,v in p.items() if k!='keypoints'}=={k:v for k,v in q.items() if k!='keypoints'}
            if p['keypoints']!=q['keypoints']:
                assert 713<=a['frame_index']<=792;changed.append([a['frame_index'],side])
    return changed


def main():
    if (OUT/'result.json').exists():raise FileExistsError('Preserve completedRGB control')
    OUT.mkdir(parents=True,exist_ok=True)
    paths={'rgb_original_missing':RGB/'rgb_same_original_missing_dense_pose.json',
           'rgb_restored_five':RGB/'rgb_restored_five_dense_pose.json'}
    original=json.loads(DENSE.read_text());restored=json.loads((BASE/'pose_missing_25p3/filled_dense_pose.json').read_text())
    variants={name:json.loads(path.read_text()) for name,path in paths.items()}
    change_counts={'rgb_original_missing':len(check_only_keypoints(original,variants['rgb_original_missing'])),
                   'rgb_restored_five':len(check_only_keypoints(restored,variants['rgb_restored_five']))}
    assert change_counts=={'rgb_original_missing':155,'rgb_restored_five':160}
    files=sorted(BALL_CONTROL.glob('*_inputs.npz'));assert len(files)==49
    dependencies=[DENSE,BALL,BASE/'pose_missing_25p3/filled_dense_pose.json',*paths.values(),
                  RGB/'protocol_frozen_2026-10-20.json',RGB/'result_2026-10-20.json',
                  BALL_CONTROL/'result.json',RUN/'baseline.pt',RUN/'candidate.pt',Path(__file__),*files]
    hashes={str(p):file_hash(p) for p in dependencies}
    protocol={'declared_at':datetime.now(timezone.utc).isoformat(),'input_sha256':hashes,
        'pose_variants':{k:str(v) for k,v in paths.items()},'rgb_pose_counts':change_counts,
        'factors':'2 RGB pose states x7 frozenS variants x7 frozen windows x2 original/prior41 heads=196. Previous BGR factorial preserved separately.',
        'design':'RGB conversion alone changes keypoints in fixed original/restored boxes. Preserve original missing5 in one state and recovered5 in the other. All P/N and S match corresponding old frozen ball controls. Whole-video player axis frozen from originalBGR. No new bounds, coordinates, weights or threshold selection.',
        'limits':'RGB inputs match published pose preprocessing contract; outputs remain estimated joints, not a perfect pose oracle. All repeateddeveloper conditions, not independent newvideo samples.',
        'no_training_or_production_activation':True}
    atomic_json(OUT/'protocol.json',protocol)
    ball=json.loads(BALL.read_text());torch.set_num_threads(2)
    models={name:SpecialistBST(original,ball,checkpoint=path) for name,path in [('baseline',RUN/'baseline.pt'),('candidate',RUN/'candidate.pt')]}
    packer=models['baseline'];order=packer.order.copy()
    record=next(r for r in json.loads((RUN/'personal_predictions.json').read_text())['baseline'] if r['event_id']==EVENT)
    target=int(record['target_class_id']);actor=record['target_class_label'].split('_',1)[0]
    allowed=np.asarray([0]+[i for i,c in enumerate(packer.classes) if c.startswith(actor+'_')])
    old=json.loads((BALL_CONTROL/'result.json').read_text());lookup={r['window']['name']+'__'+r['ball']:r for r in old['rows']}
    pose_cache={};rows=[];saved={}
    for path in files:
        key=path.name.removesuffix('_inputs.npz');prior=lookup[key];window=prior['window'];old_input=np.load(path)
        for pose,dense in variants.items():
            cachekey=pose+'__'+window['name']
            if cachekey not in pose_cache:
                lo,hi=window['lo'],window['hi'];event={**record['model_window'],'start_frame':lo,'end_frame':hi,
                    'start':original['frames'][lo]['source_time'],'end':original['frames'][hi+1]['source_time']}
                packer.dense=dense;packer.order=order;packer.coordinate_quality=InputQuality(dense,ball)
                values,quality=packer.inputs(event);j,p,s,n=values
                reference=np.load(BALL_CONTROL/(window['name']+'__original_ball_inputs.npz'))
                assert np.array_equal(p,reference['p']) and np.array_equal(s,reference['s']) and n==int(reference['n'])
                pose_cache[cachekey]=(values,quality)
                np.savez_compressed(OUT/(cachekey+'_pose_inputs.npz'),j=j,p=p,n=np.asarray(n,dtype='int64'))
            values,quality=pose_cache[cachekey];j,p,_,n=values;s=old_input['s'];assert n==int(old_input['n'])
            for name,model in models.items():
                with torch.inference_mode():scores=model.model(torch.from_numpy(j)[None],torch.from_numpy(s)[None],torch.tensor([n])).numpy()[0]
                pred=int(scores.argmax());pc=int(allowed[np.argmax(scores[allowed])]);outputkey=key+'__'+pose+'__'+name
                saved[outputkey]=scores
                rows.append({'window':window,'ball':prior['ball'],'pose':pose,'model':name,
                    'true_label':record['target_class_label'],'prediction':model.classes[pred],'joint_correct':pred==target,
                    'actor13_prediction':model.classes[pc],'actor13_correct':pc==target,
                    'true_margin25':float(scores[target]-scores[np.arange(25)!=target].max()),
                    'pair_fraction':quality['paired_player_fraction'],'n':n})
    assert len(rows)==196
    summary={}
    for pose in variants:
        summary[pose]={}
        for method in models:
            summary[pose][method]={}
            for variant in old['summary']:
                subset=[r for r in rows if r['pose']==pose and r['model']==method and r['ball']==variant]
                assert len(subset)==7
                summary[pose][method][variant]={'windows':7,'joint_correct':sum(r['joint_correct'] for r in subset),
                    'actor13_correct':sum(r['actor13_correct'] for r in subset),'predictions':sorted({r['prediction'] for r in subset})}
    assert all(file_hash(Path(p))==h for p,h in hashes.items())
    np.savez_compressed(OUT/'logits.npz',**saved)
    atomic_json(OUT/'result.json',{'created_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':file_hash(OUT/'protocol.json'),
        'summary':summary,'rows':rows,'pose_keypoint_changes':change_counts,
        'production_or_labels_or_weights_changed':False,'scope':protocol['limits']})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
