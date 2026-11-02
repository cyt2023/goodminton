"""Complete target-time sensitivity and replay the joint pose/window intervention."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from analysis_jobs import atomic_json, file_hash
from audit_ordinary_neighbor_window_grid import BASE, OUT, DENSE, BALL, RUN
from input_quality import InputQuality
from specialist_bst import SpecialistBST
from specialist_model import ROOT, BST_NAME


def main():
    folder=OUT/'target_time_sensitivity'
    if (folder/'result.json').exists():raise FileExistsError('Preserve completed sensitivity')
    folder.mkdir(parents=True,exist_ok=True)
    paths=[OUT/'protocol.json',OUT/'visual_review_frozen.json',OUT/'result.json',OUT/'logits.npz',
        DENSE,BALL,RUN/'personal_predictions.json',ROOT/'models/bst'/BST_NAME,
        ROOT/'specialist_bst.py',BASE/'pose_missing_1p0333/filled_dense_pose_far_and_near.json',Path(__file__)]
    hashes={str(p):file_hash(p) for p in paths}
    protocol={'declared_at':datetime.now(timezone.utc).isoformat(),'input_sha256':hashes,
        'reason':'Original grid kept drop center758 for cap computation. The visual target interval757..759 implies cap starts712..714. Preserve that initial result, then enumerate all9 unique start/end ranges before judging boundary robustness.',
        'drop_grid':{'start_inclusive':[712,713,714],'end_exclusive':[789,790,791]},
        'fixed_model':'Original released BST-0, original dense/ball, original whole-video order; human target13 includes original unknown. No new model/training.',
        'serve_replay':'A original pose+old16..83, B restored both poses+old16..83, C original pose+new16..69, D restored both poses+new16..69, A restored original again. Representative new bounds are fixed existing usercenter31 and midpoint of reviewed next61..64 rounded63 (+7=>70 exclusive); no score selection.',
        'scope':'Two preselected developer examples, researcher visual intervals; neither clip nor logits establish a general accuracy.'}
    atomic_json(folder/'protocol.json',protocol)
    original=json.loads(DENSE.read_text());ball=json.loads(BALL.read_text())
    restored=json.loads((BASE/'pose_missing_1p0333/filled_dense_pose_far_and_near.json').read_text())
    model=SpecialistBST(original,ball);order=model.order.copy();torch.set_num_threads(2)
    records=json.loads((RUN/'personal_predictions.json').read_text())['baseline']
    prior=json.loads((OUT/'result.json').read_text())
    old_logits=np.load(OUT/'logits.npz')
    drop=next(r for r in records if r['event_id']=='3d5e6020fbc2aa558b6a5914a9826a54')
    serve=next(r for r in records if r['event_id']=='97b39d14d2435ac65ccb3023a764f49c')
    times=np.asarray([f['source_time'] for f in original['frames']]);saved={};allrows=[]
    def infer(name,record,dense,lo,hi):
        model.dense=dense;model.order=order;model.coordinate_quality=InputQuality(dense,ball)
        event=dict(record['model_window']);event.update(start_frame=lo,end_frame=hi,start=float(times[lo]),end=float(times[hi+1]))
        (j,p,s,n),quality=model.inputs(event)
        with torch.inference_mode():
            scores=model.model(torch.from_numpy(j)[None],torch.from_numpy(s)[None],torch.tensor([n])).numpy()[0]
        label=int(record['target_class_id']);actor=model.classes[label].split('_',1)[0]
        allowed=np.asarray([0]+[i for i,c in enumerate(model.classes) if c.startswith(actor+'_')])
        pred=int(scores.argmax());pc=int(allowed[np.argmax(scores[allowed])])
        row={'name':name,'lo':lo,'hi':hi,'n':n,'true_label':model.classes[label],
             'prediction':model.classes[pred],'joint_correct':pred==label,
             'actor13_prediction':model.classes[pc],'actor13_correct':pc==label,
             'pair_fraction':quality['paired_player_fraction'],
             'margin25':float(scores[label]-scores[np.arange(25)!=label].max())}
        np.savez_compressed(folder/(name+'_inputs.npz'),j=j,p=p,s=s,n=np.asarray(n,dtype='int64'))
        saved[name]=scores;allrows.append(row)
        return row,scores
    for lo in [712,713,714]:
        for end in [789,790,791]:
            row,scores=infer(f'drop_{lo}_{end-1}',drop,original,lo,end-1)
            if lo==713:
                old=next(r for r in prior['rows'] if r['case']=='drop_25p3' and r['lo']==lo and r['hi']==end-1)
                assert np.array_equal(scores,old_logits[old['key']])
    a,a_scores=infer('serve_A_original_old',serve,original,16,83)
    b,b_scores=infer('serve_B_restored_old',serve,restored,16,83)
    c,c_scores=infer('serve_C_original_new',serve,original,16,69)
    d,d_scores=infer('serve_D_restored_new',serve,restored,16,69)
    aa,aa_scores=infer('serve_A_again',serve,original,16,83)
    assert np.array_equal(a_scores,aa_scores)
    for row,scores in [(a,a_scores),(b,b_scores),(c,c_scores),(d,d_scores)]:
        old=next(r for r in prior['rows'] if r['case']=='first_serve' and r['pose']==('far_and_near' if row['name'] in ['serve_B_restored_old','serve_D_restored_new'] else 'original') and r['lo']==row['lo'] and r['hi']==row['hi'])
        assert np.array_equal(scores,old_logits[old['key']])
    assert all(file_hash(Path(p))==h for p,h in hashes.items())
    np.savez_compressed(folder/'logits.npz',**saved)
    drop_rows=allrows[:9]
    result={'created_at':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':file_hash(folder/'protocol.json'),'rows':allrows,
        'drop_grid_summary':{'windows':9,'joint_correct':sum(r['joint_correct'] for r in drop_rows),
            'actor13_correct':sum(r['actor13_correct'] for r in drop_rows),
            'joint_predictions':sorted({r['prediction'] for r in drop_rows})},
        'serve_original_restoration_logits_bit_exact':True,'initial_grid_shared_rows_logits_bit_exact':True,
        'production_labels_weights_unchanged':True,'scope':protocol['scope']}
    atomic_json(folder/'result.json',result)
    print(json.dumps({'drop':result['drop_grid_summary'],'serve_rows':allrows[9:]},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
