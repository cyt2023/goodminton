"""All source-reviewed neighbor intervals, original inputs and fixed prior heads."""
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from analysis_jobs import atomic_json,file_hash
from specialist_bst import SpecialistBST
from specialist_model import ROOT,BST_NAME

DATA=ROOT.parent/'03_数据'
RUN=DATA/'击球标注/直接BST微调/运行记录/582817bff5454825b841d1f150c82028'
REVIEW=DATA/'击球标注/根因诊断/2026-10-18_跨来源邻拍对照/other_sources_11/review.json'
OUT=REVIEW.parent/'classification_control'


def interval(value):
    return list(range(value['frame_interval_inclusive'][0],value['frame_interval_inclusive'][1]+1)) if value is not None else [None]


def main():
    if (OUT/'result.json').exists():raise FileExistsError('Preserve completed neighbor control')
    OUT.mkdir(parents=True,exist_ok=True)
    review=json.loads(REVIEW.read_text());assert len(review['rows'])==11
    records=json.loads((RUN/'personal_predictions.json').read_text())['baseline']
    index={(r['job_id'],r['event_id']):i for i,r in enumerate(records)}
    paths=[REVIEW,REVIEW.parent/'selection_protocol.json',RUN/'personal_predictions.json',RUN/'personal_predictions.npz',
        RUN/'candidate.pt',RUN/'baseline.pt',ROOT/'models/bst'/BST_NAME,ROOT/'specialist_bst.py',Path(__file__)]
    sources={}
    for source in review['sources']:
        job=source['job_id'];dense_path=Path(source['dense_path']);ball_path=dense_path.with_name('trajectory.json')
        assert file_hash(Path(source['source_path']))==source['source_sha256'] and file_hash(dense_path)==source['dense_sha256']
        paths.extend([dense_path,ball_path,Path(source['source_path'])])
        sources[job]=(json.loads(dense_path.read_text()),json.loads(ball_path.read_text()))
    hashes={str(p):file_hash(p) for p in paths};schedule=[]
    for visual in review['rows']:
        job,eid=visual['job_id'],visual['event_id'];dense,_=sources[job]
        times=np.asarray([f['source_time'] for f in dense['frames']]);fps=int(round(1/np.median(np.diff(times))))
        assert fps in [24,30]
        t,eps,limit=fps//2,(fps//2)//2,(fps*3)//2
        ranges={}
        for previous,center,next_ in itertools.product(interval(visual['previous_actual_contact']),interval(visual['target_contact']),interval(visual['next_actual_contact'])):
            lo=max(0,center-limit,previous if previous is not None else center-t)
            end=min(len(times),center+limit+eps,next_+eps if next_ is not None else center+t)
            assert lo<=center<end
            ranges.setdefault((lo,end-1),[]).append([previous,center,next_])
        schedule.append({'job_id':job,'event_id':eid,'native_cadence':fps,'t':t,'eps':eps,'limit':limit,
            'unique_windows':[{'lo':lo,'hi':hi,'generating_contact_choices':choices} for (lo,hi),choices in sorted(ranges.items())],
            'target_interval':visual['target_contact']['frame_interval_inclusive'],
            'rally_edge':visual['rally_edge'],'saved_center_check':visual['saved_user_center_check']['result']})
    protocol={'declared_at':datetime.now(timezone.utc).isoformat(),'input_sha256':hashes,
        'reviewed_intervals_frozen_before_classification':True,'schedule':schedule,
        'policy':'Author between_2_hits_with_max_limits integer-frame rule. Use observed native24/30 cadence, t=fps//2,eps=t//2,limit=3fps//2. Previous contact start or first-target-t; next+eps endexclusive or final-target+t; caps. Enumerate EVERY reviewed previous/target/next integer combination; classify unique numerical windows and retain generator tuples. No MoviePy duration rounding or resampling.',
        'weights':['original released BST-0','prior frozen41-only mlp_head candidate, not all71 fit'],
        'fixed':'Original saved dense poses, ball, source/person order unchanged. No coordinate repair, fitting or threshold tuning.',
        'metrics':'Original25 joint; human actor12 plus unknown13. Correct-all-boundaries and any-boundary counts separate; no winning window selected.',
        'scope':'11 fixed developer annotation rows from two sources, including initially correct controls. Visual intervals are researcher evidence, not user exact-contact truth; no web pipeline accuracy claim.'}
    atomic_json(OUT/'protocol.json',protocol)
    cache=np.load(RUN/'personal_predictions.npz');torch.set_num_threads(2)
    models={};results=[];saved={}
    for job,(dense,ball) in sources.items():
        models[job]={name:SpecialistBST(dense,ball,checkpoint=path) for name,path in [('baseline',RUN/'baseline.pt'),('candidate',RUN/'candidate.pt')]}
    for item in schedule:
        job,eid=item['job_id'],item['event_id'];ri=index[(job,eid)];record=records[ri]
        dense,_=sources[job];times=np.asarray([f['source_time'] for f in dense['frames']]);model=models[job]['baseline']
        label=int(record['target_class_id']);actor=record['target_class_label'].split('_',1)[0]
        allowed=np.asarray([0]+[i for i,c in enumerate(model.classes) if c.startswith(actor+'_')]);assert len(allowed)==13
        row={'job_id':job,'event_id':eid,'true_label':model.classes[label],'rally_edge':item['rally_edge'],
            'saved_center_check':item['saved_center_check'],'original':{},'grid':[]}
        variants=[('original',{'lo':record['model_window']['start_frame'],'hi':record['model_window']['end_frame']})]
        variants.extend((f'grid_{k:03d}',v) for k,v in enumerate(item['unique_windows']))
        for key,window in variants:
            lo,hi=window['lo'],window['hi'];event=dict(record['model_window'])
            choices=window.get('generating_contact_choices')
            center=choices[0][1] if choices else event['frame_index']
            event.update(start_frame=lo,end_frame=hi,frame_index=center,peak_time=float(times[center]),
                start=float(times[lo]),end=float(times[hi+1]) if hi+1<len(times) else float(times[hi]+1/item['native_cadence']))
            values,quality=model.inputs(event);j,p,s,n=values
            result={'lo':lo,'hi':hi,'n':n,'paired_fraction':quality['paired_player_fraction'],
                'raw_ball_fraction':quality['raw_shuttle_fraction'],'reference_pair85_gate_passes':quality['paired_player_fraction']>=.85,
                'conditions':{}}
            packedkey=eid+'__'+key
            np.savez_compressed(OUT/(packedkey+'_inputs.npz'),j=j,p=p,s=s,n=np.asarray(n,dtype='int64'))
            for method,m in models[job].items():
                with torch.inference_mode():scores=m.model(torch.from_numpy(j)[None],torch.from_numpy(s)[None],torch.tensor([n])).numpy()[0]
                if key=='original':
                    ck='baseline_logits' if method=='baseline' else 'candidate_logits'
                    assert np.max(abs(scores-cache[ck][ri]))<1e-4
                pred=int(scores.argmax());pc=int(allowed[np.argmax(scores[allowed])])
                result['conditions'][method]={'prediction':model.classes[pred],'joint_correct':pred==label,
                    'actor13_prediction':model.classes[pc],'actor13_correct':pc==label}
                saved[packedkey+'__'+method]=scores
            if key=='original':row['original']=result
            else:row['grid'].append(result)
        results.append(row)
    summary={}
    for method in ['baseline','candidate']:
        summary[method]={}
        for metric in ['joint_correct','actor13_correct']:
            summary[method][metric]={'annotation_rows':11,
                'original_correct':sum(r['original']['conditions'][method][metric] for r in results),
                'correct_for_all_reviewed_windows':sum(all(v['conditions'][method][metric] for v in r['grid']) for r in results),
                'correct_for_at_least_one_reviewed_window':sum(any(v['conditions'][method][metric] for v in r['grid']) for r in results),
                'new_robust_gains':[r['event_id'] for r in results if not r['original']['conditions'][method][metric] and all(v['conditions'][method][metric] for v in r['grid'])],
                'new_robust_losses':[r['event_id'] for r in results if r['original']['conditions'][method][metric] and not any(v['conditions'][method][metric] for v in r['grid'])]}
    assert all(file_hash(Path(p))==h for p,h in hashes.items())
    np.savez_compressed(OUT/'logits.npz',**saved)
    atomic_json(OUT/'result.json',{'created_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':file_hash(OUT/'protocol.json'),
        'summary':summary,'rows':results,'unique_grid_windows':sum(len(r['grid']) for r in results),
        'production_or_labels_or_weights_changed':False,'scope':protocol['scope']})
    print(json.dumps({'summary':summary,'grid_windows':sum(len(r['grid']) for r in results)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
