"""Developer control: nearest native-frame regridding, with no synthetic joints."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from analysis_jobs import atomic_json, file_hash
from specialist_bst import SpecialistBST
from specialist_model import ROOT, BST_NAME

BASE=ROOT.parent/'03_数据'
RUN=BASE/'击球标注/直接BST微调/运行记录/582817bff5454825b841d1f150c82028'
SELECTION=BASE/'击球标注/根因诊断/2026-10-18_普通视角配对诊断/selection_protocol.json'
OUT=BASE/'击球标注/根因诊断/2026-10-18_实际帧率时间网格对照'


def pack(model, indices):
    joints=[];shuttle=[]
    for i in indices:
        people=[model.dense['frames'][int(i)]['players'].get(side) for side in model.order]
        if all(p is not None for p in people):
            pts=np.asarray([[[k['x'],k['y']] for k in p['keypoints']] for p in people],dtype=float)
            boxes=np.asarray([p['bbox'] for p in people])
            joints.append(model.helpers['normalize_joints'](pts,boxes,center_align=True))
        else:joints.append(np.zeros((2,17,2)))
        b=model.trajectory['frames'][int(i)]
        xy=b['rectified_xy'] if b['inpaint_requested'] else b['raw_xy']
        shuttle.append(xy or [0,0])
    joints=np.asarray(joints,dtype='float32');positions=np.zeros((len(indices),2,2),dtype='float32')
    shuttle=model.helpers['normalize_shuttlecock'](np.asarray(shuttle,dtype='float32'),model.dense['source_width'],model.dense['source_height'])
    j,p,s,n=model.helpers['make_seq_len_same'](100,joints,positions,shuttle)
    bones=model.helpers['create_bones'](j,model.bones)
    j=np.concatenate((j,bones),axis=-2).reshape(100,2,72)
    return j.astype('float32'),p.astype('float32'),s.astype('float32'),int(n)


def main():
    if (OUT/'result.json').exists():raise FileExistsError('Preserve frozen cadence control')
    OUT.mkdir(parents=True,exist_ok=True)
    selection=json.loads(SELECTION.read_text());records=json.loads((RUN/'personal_predictions.json').read_text())['baseline']
    paths=[SELECTION,RUN/'personal_predictions.json',RUN/'personal_predictions.npz',ROOT/'models/bst'/BST_NAME,ROOT/'specialist_bst.py',Path(__file__)]
    for source in selection['per_source']:
        paths.extend([BASE/'专项分析'/source['job_id']/'dense_pose.json',BASE/'专项分析'/source['job_id']/'trajectory.json'])
    hashes={str(p):file_hash(p) for p in paths}
    protocol={'declared_at':datetime.now(timezone.utc).isoformat(),'input_sha256':hashes,
        'question':'Does observed native24fps vs about30fps cadence / source timestamp gaps contribute to residual errors?',
        'claim_limit':'Author generator uses each video native fps; repository does not establish all author training clips as exactly30fps. This is a fixed representation-sensitivity experiment, not correction of a proven30fps contract bug.',
        'policy':'Keep each original window actual native-PTS start and following-frame end-exclusive. Grid30fps from first nativePTS; count=ceil(duration*30-1e-5). At each grid time choose nearest native frame within original support, stable earlier tie. Repeat whole observed poses/ball points, no interpolation or missing-value filling. Normalize and apply original author seq100/bone policy unchanged.',
        'conditions':['original_native_frame_sequence','nearest_native_frames_on_30fps_grid'],
        'selection':'All fixed30 developer annotation rows, correct and incorrect. Physical duplicate audit pending; do not call30 independent contacts.',
        'weights':'Original released BST-0 only; no fine-tuning, label edits, threshold search or production activation.',
        'scope':'Previously inspected developer rows, not new-video general accuracy.'}
    atomic_json(OUT/'protocol.json',protocol)
    cache=np.load(RUN/'personal_predictions.npz');models={};rows=[];logits={}
    index={(r['job_id'],r['event_id']):i for i,r in enumerate(records)}
    torch.set_num_threads(2)
    for mapped in selection['all_30_mapped_rows']:
        job,eid=mapped['job_id'],mapped['event_id'];i=index[(job,eid)];record=records[i];event=record['model_window']
        if job not in models:
            dense=json.loads((BASE/'专项分析'/job/'dense_pose.json').read_text());ball=json.loads((BASE/'专项分析'/job/'trajectory.json').read_text())
            models[job]=SpecialistBST(dense,ball)
        model=models[job];times=np.asarray([f['source_time'] for f in model.dense['frames']])
        lo,hi=event['start_frame'],event['end_frame'];native=np.arange(lo,hi+1);selected_times=times[native]
        end=float(times[hi+1]) if hi+1<len(times) else float(times[hi]+np.median(np.diff(times)))
        count=int(np.ceil((end-times[lo])*30-1e-5));grid=times[lo]+np.arange(count)/30
        nearest=native[np.argmin(abs(selected_times[None,:]-grid[:,None]),axis=1)]
        assert all(lo<=x<=hi for x in nearest) and len(nearest)>0
        original_values=model.inputs(event)[0]
        assert all(np.array_equal(a,b) for a,b in zip(original_values,pack(model,native)))
        target=int(record['target_class_id']);actor=model.classes[target].split('_',1)[0]
        allowed=np.asarray([0]+[a for a,c in enumerate(model.classes) if c.startswith(actor+'_')])
        result={'job_id':job,'event_id':eid,'native_sample_fps_metadata':model.dense['sample_fps'],
                'actual_median_frame_interval':float(np.median(np.diff(times))), 'lo':lo,'hi':hi,
                'source_start':float(times[lo]),'source_end_exclusive':end,
                'native_frame_count':len(native),'grid_frame_count':len(nearest),
                'repeated_native_frames':len(nearest)-len(set(nearest.tolist())),
                'max_grid_to_native_time_error':float(np.max(abs(times[nearest]-grid))),
                'true_label':model.classes[target],'conditions':{}}
        for name,ids,grid_times in [('native',native,selected_times),('grid30',nearest,grid)]:
            j,p,s,n=pack(model,ids)
            with torch.inference_mode():scores=model.model(torch.from_numpy(j)[None],torch.from_numpy(s)[None],torch.tensor([n])).numpy()[0]
            if name=='native':assert np.max(abs(scores-cache['baseline_logits'][i]))<1e-4
            pred=int(scores.argmax());pc=int(allowed[np.argmax(scores[allowed])])
            result['conditions'][name]={'prediction':model.classes[pred],'joint_correct':pred==target,
                                      'actor13_prediction':model.classes[pc],'actor13_correct':pc==target,'n':n}
            logits[eid+'__'+name]=scores
            np.savez_compressed(OUT/(eid+'__'+name+'.npz'),j=j,p=p,s=s,n=np.asarray(n,dtype='int64'),source_indices=ids,source_times=times[ids],grid_times=grid_times)
        result['packed_inputs_identical']=all(np.array_equal(a,b) for a,b in zip(pack(model,native),pack(model,nearest)))
        rows.append(result)
    groups={}
    for job in sorted(models):
        subset=[r for r in rows if r['job_id']==job]
        groups[job]={'annotation_rows':len(subset), 'native_fps_metadata':models[job].dense['sample_fps'],
            **{n:sum(r['conditions'][n]['joint_correct'] for r in subset) for n in ['native','grid30']},
            'gains':[r['event_id'] for r in subset if not r['conditions']['native']['joint_correct'] and r['conditions']['grid30']['joint_correct']],
            'losses':[r['event_id'] for r in subset if r['conditions']['native']['joint_correct'] and not r['conditions']['grid30']['joint_correct']]}
    assert all(file_hash(Path(p))==h for p,h in hashes.items())
    np.savez_compressed(OUT/'logits.npz',**logits)
    result={'created_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':file_hash(OUT/'protocol.json'),
            'by_source':groups,'rows':rows,'native_correct':sum(r['conditions']['native']['joint_correct'] for r in rows),
            'grid30_correct':sum(r['conditions']['grid30']['joint_correct'] for r in rows),
            'packed_unchanged_annotation_rows':sum(r['packed_inputs_identical'] for r in rows),
            'production_labels_weights_changed':False,'scope':protocol['scope']}
    atomic_json(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
