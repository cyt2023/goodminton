"""Frozen ordinary serve: add only reviewed missing person poses, same BST/S/window."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analysis_jobs import atomic_json, file_hash
from input_quality import InputQuality
from specialist_bst import SpecialistBST
from specialist_model import ROOT, BST_NAME

OUT=ROOT.parent/'03_数据/击球标注/根因诊断/2026-10-18_普通视角配对诊断'
FOLDER=OUT/'pose_missing_1p0333'
JOB='J_e23c1dd9297d4bfca7c91d1d95065b68'
EVENT='97b39d14d2435ac65ccb3023a764f49c'
ORIGINAL=ROOT.parent/'03_数据/专项分析'/JOB/'dense_pose.json'
BALL=ORIGINAL.with_name('trajectory.json')
RUN=ROOT.parent/'03_数据/击球标注/直接BST微调/运行记录/582817bff5454825b841d1f150c82028'


def main():
    if (FOLDER/'classification_result.json').exists():
        raise FileExistsError('Preserve completed frozen classification')
    paths={'original':ORIGINAL,'far_only':FOLDER/'filled_dense_pose.json',
           'far_and_near':FOLDER/'filled_dense_pose_far_and_near.json',
           'ball':BALL,'selection':OUT/'selection_protocol.json',
           'saved_predictions':RUN/'personal_predictions.json','saved_logits':RUN/'personal_predictions.npz',
           'model':ROOT/'models/bst'/BST_NAME,'adapter':ROOT/'specialist_bst.py','audit_code':Path(__file__)}
    hashes={k:file_hash(p) for k,p in paths.items()}
    records=json.loads(paths['saved_predictions'].read_text())['baseline']
    ri=next(i for i,r in enumerate(records) if r['event_id']==EVENT)
    record=records[ri];event=record['model_window']
    assert event['start_frame']==16 and event['end_frame']==83 and event['frame_index']==31
    protocol={'declared_at':datetime.now(timezone.utc).isoformat(),'event_id':EVENT,
        'input_paths':{k:str(v) for k,v in paths.items()},'input_hashes':hashes,
        'window':event,'true_label':record['target_class_label'],
        'conditions':['original','far_only','far_and_near'],
        'design':'Only insert independently source-reviewed missing pose entries. Keep originally present poses, whole-video person-axis order, source points/times, original ball, fixed68frame window and originalBST-0 weights.',
        'evaluation':'Joint25class argmax plus explicitly known-human-target actor-conditioned class as separate diagnostic; no kind label used for actor conditioning.',
        'scope':'One frozen ordinary developer record. Filled poses are BGR RTMPose estimates from reviewed detector boxes, not true joint coordinates; boundaries remain unverified.',
        'no_training_or_production_activation':True}
    atomic_json(FOLDER/'classification_protocol.json',protocol)
    original=json.loads(ORIGINAL.read_text());ball=json.loads(BALL.read_text())
    classifier=SpecialistBST(original,ball);fixed_order=classifier.order.copy()
    target=record['target_class_id'];assert classifier.classes[target]==record['target_class_label']
    true_actor=record['target_class_label'].split('_',1)[0]
    allowed=np.asarray([i for i,c in enumerate(classifier.classes) if c.startswith(true_actor+'_')])
    cache=np.load(paths['saved_logits']);torch.set_num_threads(2)
    results={};saved={};baseline=None
    for name in ['original','far_only','far_and_near']:
        dense=json.loads(paths[name].read_text());assert dense['source_sha256']==original['source_sha256']
        assert len(dense['frames'])==len(original['frames'])
        added=[]
        for i,(a,b) in enumerate(zip(original['frames'],dense['frames'])):
            assert {k:v for k,v in a.items() if k!='players'}=={k:v for k,v in b.items() if k!='players'}
            for side in ['near','far']:
                p,q=a['players'][side],b['players'][side]
                if p is not None:
                    assert p==q
                elif q is not None:
                    assert 16<=i<=83
                    added.append({'frame_index':i,'side':side})
        classifier.dense=dense;classifier.order=fixed_order
        classifier.coordinate_quality=InputQuality(dense,ball)
        packed,quality=classifier.inputs(event);j,p,s,n=packed
        with torch.inference_mode():
            logits=classifier.model(torch.from_numpy(j)[None],torch.from_numpy(s)[None],torch.tensor([n])).numpy()[0]
        if baseline is None:
            baseline=(p.copy(),s.copy(),n)
            baseline_error=float(np.max(abs(logits-cache['baseline_logits'][ri])))
            assert baseline_error<1e-4 and logits.argmax()==cache['baseline_logits'][ri].argmax()
        else:
            assert np.array_equal(p,baseline[0]) and np.array_equal(s,baseline[1]) and n==baseline[2]
        predicted=int(logits.argmax());conditioned=int(allowed[np.argmax(logits[allowed])])
        mask=np.arange(25)!=target
        results[name]={'prediction':classifier.classes[predicted],'true_label':record['target_class_label'],
            'joint_correct':predicted==target,'known_target_conditioned_prediction':classifier.classes[conditioned],
            'known_target_conditioned_correct':conditioned==target,
            'true_label_logit_margin_25classes':float(logits[target]-logits[mask].max()),
            'paired_fraction':quality['paired_player_fraction'],'raw_ball_fraction':quality['raw_shuttle_fraction'],
            'added_pose_entries':added,'video_len':n,
            'input_sha256':hashlib.sha256(j.tobytes()+s.tobytes()+np.asarray(n,dtype='int64').tobytes()).hexdigest()}
        saved[name]=logits
        np.savez_compressed(FOLDER/(name+'_classification_inputs.npz'),j=j,p=p,s=s,n=np.asarray(n,dtype='int64'))
    assert len(results['far_only']['added_pose_entries'])==31
    assert len(results['far_and_near']['added_pose_entries'])==45
    assert all(file_hash(paths[k])==h for k,h in hashes.items())
    np.savez_compressed(FOLDER/'classification_logits.npz',**saved)
    atomic_json(FOLDER/'classification_result.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':file_hash(FOLDER/'classification_protocol.json'),'checks':results,
        'baseline_max_logit_difference_from_saved_run':baseline_error,'person_order_frozen':fixed_order,
        'ball_positions_and_length_identical':True,'production_or_labels_or_weights_changed':False,
        'scope':protocol['scope']})
    print(json.dumps({k:{a:b for a,b in r.items() if a!='added_pose_entries'} for k,r in results.items()},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
