"""Fixed temporal-contract diagnostics on the previously inspected 22 examples.

No training, weight selection, label edits, or production-result changes. Every
intervention and negative control is declared before inference. --replay checks
saved numeric inputs against the frozen checkpoint without video extraction.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from analysis_jobs import DATA, ROOT, atomic_json, file_hash
from bst_public_data import ARRAYS, public_classifier
from personal_learning import extractor
from specialist_model import BST_NAME
from stroke_review import folder_for, read_labels, load_source
from stroke_window_repairs import translate_contact_window
from stroke_taxonomy import AUTHOR_MAP

VERSION = 'temporal-contract-fixed-ablation-v1'
SAVED = DATA/'击球标注/输入修复诊断/2026-10-12_配对输入修复'
DEFAULT = DATA/'击球标注/时间契约诊断/2026-10-13_固定单因素'
PROTOCOL = {
    'frozen_replay': 'Exact previously frozen automatic input; require scores within 1e-5.',
    'human_translated': 'Saved original-duration support translated to human contact; all following interventions compare to this.',
    'cadence_25fps': 'Same physical support; nearest-neighbor sample at 25 Hz, then original author pad/subsample. 25 Hz is one observed author cadence, not selected by performance.',
    'cadence_30fps': 'Same physical support; nearest-neighbor sample at 30 Hz, then original author pad/subsample. 30 Hz is the other author cadence, not selected by performance.',
    'uniform_100_steps': 'Resample valid sequence uniformly to 100 positions (nearest neighbor); deliberately tests time normalization that the author does NOT perform.',
    'author_rule_sparse_neighbors': 'Previous/next manually confirmed opposite-player contacts, author integer-frame t=fps//2, eps=t//2, limit=fps*3//2. Rally membership and neighbor completeness UNKNOWN; oracle partial diagnostic only.',
    'leading_zero_10': 'Prepend up to 10 zero steps and extend valid length, preserving every valid input frame; tests absolute position and exposed missing-prefix sensitivity, not a recommended correction.',
    'tail_repeat_while_masked': 'Only replace masked trailing zero slots with last valid frame, keep video_len and every valid frame identical. Tests TCN boundary influence before attention masking.',
    'reverse_valid_negative_control': 'Reverse valid frame order, preserving unordered frame content; label agreement is coincidence/sensitivity, not accuracy.',
    'unrelated_plus5s_negative_control': 'Shift entire physical support +5 seconds, or -5 near recording end; require original target outside. Agreement against original label is coincidence, not accuracy.',
}


def quantiles(values):
    a=np.asarray(values,float)
    return {'count':len(a),'min':float(a.min()),'p10':float(np.quantile(a,.1)),
            'median':float(np.median(a)),'p90':float(np.quantile(a,.9)),
            'max':float(a.max()),'mean':float(a.mean())} if len(a) else {'count':0}


def frame_count(path):
    with path.open('rb') as stream:
        version=np.lib.format.read_magic(stream)
        shape,_,_= (np.lib.format.read_array_header_1_0(stream) if version==(1,0)
                    else np.lib.format.read_array_header_2_0(stream))
    return int(shape[0])


def author_metadata(out):
    """Only train NPY headers; infer source cadence from supplied CSV time/frame."""
    src=ROOT/'vendor/BST/ShuttleSet/set'
    matches=list(csv.DictReader((src/'match.csv').open()))
    metadata={};fps_info={};hashes={}
    for match in matches:
        vid=int(match['id']);pairs=[]
        for path in sorted((src/match['video']).glob('set*.csv')):
            hashes[str(path.relative_to(ROOT))]=file_hash(path)
            set_id=int(path.stem[3:]);rows=list(csv.DictReader(path.open()))
            for i,row in enumerate(rows):
                if not row.get('frame_num') or not row.get('ball_round'):continue
                frame=int(float(row['frame_num']));rally=int(float(row['rally']));ball=int(float(row['ball_round']))
                prev=rows[i-1] if i>0 and rows[i-1]['rally']==row['rally'] else None
                nxt=rows[i+1] if i+1<len(rows) and rows[i+1]['rally']==row['rally'] else None
                metadata[(vid,set_id,rally,ball)]={'frame':frame,'previous':int(float(prev['frame_num'])) if prev else None,
                    'next':int(float(nxt['frame_num'])) if nxt else None,'rally_first':prev is None,'rally_last':nxt is None}
                if row.get('time'):
                    vals=[float(v) for v in row['time'].split(':')];seconds=sum(v*m for v,m in zip(vals,[3600,60,1]))
                    pairs.append((seconds,frame))
        a=np.asarray(pairs);slope,intercept=np.polyfit(a[:,0],a[:,1],1)
        fps=min([24,25,30,50,60],key=lambda f:abs(f-slope))
        if abs(fps-slope)>.1:raise ValueError('Cannot infer author cadence: '+str(vid))
        fps_info[vid]={'fps_inferred':fps,'time_frame_regression_slope':float(slope),
                       'intercept':float(intercept),'rows':len(a),
                       'method':'Nearest nominal cadence from timestamp/frame_num regression; no original media fps probe.'}
    index=json.loads((DATA/'公开训练数据/BST_ShuttleSet/index.json').read_text())['samples']
    rows=[];missing=[]
    for item in index:
        if item['split']!='train':continue
        key=tuple(map(int,item['sample_id'].split('_')))
        if key not in metadata:missing.append(item['sample_id']);continue
        m=metadata[key];fps=fps_info[key[0]]['fps_inferred'];t=fps//2;eps=t//2;limit=fps*3//2;c=m['frame']
        start=max(m['previous'] if m['previous'] is not None else c-t,c-limit)
        end=min(m['next']+eps if m['next'] is not None else c+t,c+limit+eps)
        n=frame_count(ARRAYS/(item['base']+'_pos.npy'))
        stride=n//100+int(n%100>50) if n>100 else 1
        selected=np.arange(n)[::stride][:100];valid=len(selected)
        rows.append({'sample_id':item['sample_id'],'class':item['class'],'fps_inferred':fps,
            'raw_frames':n,'author_formula_frames':end-start,'array_minus_formula_frames':n-(end-start),
            'duration_seconds':n/fps,'contact_offset_frames':c-start,
            'contact_position_fraction':(c-start)/max(n-1,1),'valid_model_steps':valid,'trailing_padding':100-valid,
            'sampling_stride':stride,'rally_first':m['rally_first'],'rally_last':m['rally_last'],
            'frames_after_contact':n-1-(c-start)})
    if len(rows)!=25741 or missing:raise ValueError('Author train header coverage mismatch')
    def group(items):
        return {'count':len(items),'fps_counts':dict(Counter(r['fps_inferred'] for r in items)),
            'raw_frames':quantiles([r['raw_frames'] for r in items]),
            'duration_seconds':quantiles([r['duration_seconds'] for r in items]),
            'contact_position_fraction':quantiles([r['contact_position_fraction'] for r in items]),
            'valid_model_steps':quantiles([r['valid_model_steps'] for r in items]),
            'trailing_padding':quantiles([r['trailing_padding'] for r in items]),
            'formula_difference_counts':dict(Counter(r['array_minus_formula_frames'] for r in items)),
            'raw_over_100':sum(r['raw_frames']>100 for r in items)}
    with (out/'author_train_temporal_rows.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'all_train':group(rows),'by_fps':{str(f):group([r for r in rows if r['fps_inferred']==f]) for f in [25,30]},
            'rally_first':group([r for r in rows if r['rally_first']]),
            'rally_last':group([r for r in rows if r['rally_last']]),
            'source_fps_inference':fps_info,'annotation_sha256':hashes,
            'scope':'Only author TRAIN array headers read; no public test data or author media inference.'}
    atomic_json(out/'author_temporal_metadata.json',result)
    return result


def pack(classifier,j,p,s):
    j,p,s,n=classifier.helpers['make_seq_len_same'](100,j.reshape(len(j),2,36,2),p,s)
    return j.reshape(100,2,72).astype('float32'),p.astype('float32'),s.astype('float32'),int(n)


def resample(classifier,values,source_fps,target_fps=None,uniform=False):
    j,p,s,n=values
    if uniform:
        ids=np.rint(np.linspace(0,n-1,100)).astype(int)
    else:
        grid=np.arange(0,n/source_fps-1e-9,1/target_fps)
        ids=np.clip(np.rint(grid*source_fps).astype(int),0,n-1)
    return pack(classifier,j[ids],p[ids],s[ids]),ids


def sparse_neighbor_window(dense,human,doc):
    times=np.asarray([f['source_time'] for f in dense['frames']]);fps=float(dense['sample_fps'])
    c=int(np.argmin(abs(times-human['peak_time'])));t=int(fps//2);eps=t//2;limit=int(fps*3//2)
    contacts={}
    for h in doc['labels'].values():
        if h.get('contact_confirmed') and not h.get('wrong_target') and h.get('play_context')!='between_rallies' and h['stroke_type'] not in ['unknown','no_stroke']:
            i=int(np.argmin(abs(times-h['peak_time'])));contacts[(h['player'],i)]=h
    other=sorted(i for (player,i) in contacts if player!=human['player'] and abs(i-c)>max(1,.15*fps))
    previous=max((i for i in other if i<c),default=None);nxt=min((i for i in other if i>c),default=None)
    a=max(0,c-limit,previous if previous is not None else c-t)
    # Author MoviePy endpoint is exclusive; retain equivalent end-1 source index.
    end_exclusive=min(len(times),c+limit+eps,nxt+eps if nxt is not None else c+t)
    b=max(c,end_exclusive-1);a=min(a,c)
    return {'start_frame':a,'end_frame':b,'frame_index':c,'peak_time':float(times[c]),
            'start':float(times[a]),'end':float(times[b]+1/fps),'player':human['player']}, {
            'previous_reviewed_opponent_frame':previous,'next_reviewed_opponent_frame':nxt,
            'previous_gap_seconds':(c-previous)/fps if previous is not None else None,
            'next_gap_seconds':(nxt-c)/fps if nxt is not None else None,
            'rally_membership_verified':False,'neighbors_complete':False,
            'known_same_player_other_contacts_inside':sum(player==human['player'] and a<=i<=b and i!=c for player,i in contacts),
            'known_opponent_contacts_inside':sum(player!=human['player'] and a<=i<=b for player,i in contacts),
            'first_or_last_hit_fallback_is_not_verified':previous is None or nxt is None}


def pred(classifier,values,human):
    j,_,s,n=values
    with torch.inference_mode():
        logits=classifier.model(torch.from_numpy(j[None]),torch.from_numpy(s[None]),torch.tensor([n]))[0]
    probs=logits.softmax(-1).numpy();label=classifier.classes[int(probs.argmax())];prefix,sep,raw=label.partition('_')
    side=classifier.order[0 if prefix=='Top' else 1] if prefix in ['Top','Bottom'] else None
    options=[i for i,c in enumerate(classifier.classes) if c.startswith(('Top_' if human['player']==classifier.order[0] else 'Bottom_'))]
    known=classifier.classes[max(options,key=lambda i:probs[i])]
    return {'author_label':label,'stroke_type':AUTHOR_MAP[raw if sep else prefix],'mapped_actor':side,
            'known_target_stroke_type':AUTHOR_MAP[known.split('_',1)[-1]],'scores':probs.tolist()}


def summarize(rows,variant):
    total=len(rows);typ=actor=joint=known=flips=0;diff=[]
    for row in rows:
        p=row['predictions'][variant];b=row['predictions']['human_translated'];h=row['human']
        typ+=p['stroke_type']==h['stroke_type'];actor+=p['mapped_actor']==h['player'];joint+=(p['stroke_type']==h['stroke_type'] and p['mapped_actor']==h['player']);known+=p['known_target_stroke_type']==h['stroke_type'];flips+=p['author_label']!=b['author_label'];diff.append(max(abs(a-c) for a,c in zip(p['scores'],b['scores'])))
    return {'total':total,'type_label_agreements':typ,'actor_agreements_under_unverified_mapping':actor,
            'joint_label_agreements_under_unverified_mapping':joint,'known_human_target_type_agreements':known,
            'changed_top1_from_human_translated':flips,'max_score_change':max(diff),
            'mean_max_score_change':float(np.mean(diff)),
            'negative_control_agreement_is_not_accuracy': 'negative_control' in variant}


def replay(out):
    result=json.loads((out/'result.json').read_text());classifier=public_classifier()
    if file_hash(ROOT/'models/bst'/BST_NAME)!=result['model_sha256']:raise ValueError('Checkpoint changed')
    max_error=0
    with np.load(out/'inputs.npz',allow_pickle=False) as saved:
        for i,(job,eid,variant) in enumerate(saved['keys'].tolist()):
            row=next(r for r in result['rows'] if r['job_id']==job and r['event_id']==eid);classifier.order=row['player_input_order']
            values=(saved['joints_and_bones'][i],saved['positions'][i],saved['shuttle'][i],int(saved['video_len'][i]))
            p=pred(classifier,values,row['human']);max_error=max(max_error,max(abs(a-b) for a,b in zip(p['scores'],row['predictions'][variant]['scores'])))
    if max_error>1e-5:raise ValueError('Replay mismatch '+str(max_error))
    print(json.dumps({'replayed_inputs':i+1,'max_score_error':max_error,'weights_changed':False}),flush=True)


def run(out):
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'protocol.json',{'version':VERSION,'declared_at':datetime.now(timezone.utc).isoformat(),
        'variants':PROTOCOL,'scope':'Previously inspected 22 development rows, one same-contact duplicate; no parameter search or independent accuracy claim.',
        'code_sha256':file_hash(Path(__file__))})
    author=author_metadata(out);old=json.loads((SAVED/'result.json').read_text());rows=[];inputs=[];keys=[];protected={};replay_error=0
    by_job=defaultdict(list)
    for row in old['rows']:by_job[row['job_id']].append(row)
    with np.load(SAVED/'inputs.npz',allow_pickle=False) as saved:
        lookup={tuple(k):i for i,k in enumerate(saved['keys'].tolist())}
        for job,group in by_job.items():
            report,events,event_file=load_source(job);doc=read_labels(job,report['source_sha256']);classifier,dense=extractor(job)
            for path in [folder_for(job)/'annotations.json',folder_for(job)/'public_only_predictions.json',event_file,DATA/'专项分析'/job/'dense_pose.json',DATA/'专项分析'/job/'trajectory.json']:
                protected[str(path)]=file_hash(path)
            for oldrow in group:
                human=oldrow['human'];eid=oldrow['event_id'];classifier.order=list(oldrow['variants']['frozen_replay']['player_order'])
                def cached(name):
                    i=lookup[(job,eid,name)];return saved['joints_and_bones'][i].copy(),saved['positions'][i].copy(),saved['shuttle'][i].copy(),int(saved['video_len'][i])
                frozen=cached('frozen_replay');base=cached('human_center_same_duration');j,p,s,n=base;w=oldrow['human_shifted_window'];fps=float(dense['sample_fps']);target=int(w['frame_index'])-int(w['start_frame'])
                variants={'frozen_replay':frozen,'human_translated':base};detail={}
                for f in [25,30]:
                    values,ids=resample(classifier,base,fps,target_fps=f);variants['cadence_'+str(f)+'fps']=values;detail['cadence_'+str(f)+'fps']={'source_indices':ids.tolist(),'contact_source_index_retained':target in ids,'target_index_nearest':int(np.argmin(abs(ids-target)))}
                variants['uniform_100_steps'],ids=resample(classifier,base,fps,uniform=True);detail['uniform_100_steps']={'source_indices':ids.tolist(),'contact_source_index_retained':target in ids}
                nw,notes=sparse_neighbor_window(dense,human,doc);variants['author_rule_sparse_neighbors']=classifier.inputs(nw)[0];detail['author_rule_sparse_neighbors']={**notes,'window':nw}
                shift=min(10,100-n);jj=np.zeros_like(j);pp=np.zeros_like(p);ss=np.zeros_like(s);jj[shift:shift+n]=j[:n];pp[shift:shift+n]=p[:n];ss[shift:shift+n]=s[:n]
                variants['leading_zero_10']=(jj,pp,ss,n+shift);detail['leading_zero_10']={'leading_zeros':shift,'all_valid_frames_preserved':True}
                jj=j.copy();pp=p.copy();ss=s.copy();jj[n:]=j[n-1];pp[n:]=p[n-1];ss[n:]=s[n-1];variants['tail_repeat_while_masked']=(jj,pp,ss,n)
                jj=j.copy();pp=p.copy();ss=s.copy();jj[:n]=j[:n][::-1];pp[:n]=p[:n][::-1];ss[:n]=s[:n][::-1];variants['reverse_valid_negative_control']=(jj,pp,ss,n)
                seconds=5. if human['peak_time']+5<float(dense['end']) else -5.
                uw=translate_contact_window(dense,w,human['peak_time']+seconds);assert not uw['start']<=human['peak_time']<uw['end'];variants['unrelated_plus5s_negative_control']=classifier.inputs(uw)[0];detail['unrelated_plus5s_negative_control']={'window':uw,'original_target_outside':True}
                row={'job_id':job,'event_id':eid,'human':{k:human[k] for k in ['player','stroke_type','peak_time']},'player_input_order':classifier.order.copy(),
                     'source_fps':fps,'source_window':w,'raw_support_frames':int(w['end_frame'])-int(w['start_frame'])+1,'valid_model_steps':n,'trailing_padding':100-n,
                     'target_index_in_valid_sequence':target,'target_position_fraction':target/max(n-1,1),'before_seconds':target/fps,'after_seconds':(n-1-target)/fps,
                     'predictions':{},'interventions':detail}
                for name,values in variants.items():
                    row['predictions'][name]=pred(classifier,values,human);inputs.append(values);keys.append((job,eid,name))
                    if name=='frozen_replay':replay_error=max(replay_error,max(abs(a-b) for a,b in zip(row['predictions'][name]['scores'],oldrow['variants']['frozen_replay']['scores'])))
                rows.append(row);print('TEMPORAL',job,eid,len(rows),flush=True)
    if replay_error>1e-5:raise ValueError('Frozen input replay differs')
    protected[str(ROOT/'models/bst'/BST_NAME)]=file_hash(ROOT/'models/bst'/BST_NAME)
    if any(file_hash(Path(path))!=digest for path,digest in protected.items()):raise ValueError('Protected source changed')
    np.savez_compressed(out/'inputs.npz',keys=np.asarray(keys),joints_and_bones=np.stack([v[0] for v in inputs]),positions=np.stack([v[1] for v in inputs]),shuttle=np.stack([v[2] for v in inputs]),video_len=np.asarray([v[3] for v in inputs]))
    summary={name:summarize(rows,name) for name in PROTOCOL};result={'version':VERSION,'created_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':file_hash(out/'protocol.json'),
        'model_sha256':file_hash(ROOT/'models/bst'/BST_NAME),'input_sha256':file_hash(out/'inputs.npz'),'saved_diagnostic_sha256':file_hash(SAVED/'result.json'),
        'scope':'Paired sensitivity diagnostics, not a fresh/generalization score. Human opponent contacts are incomplete and rally boundaries unverified. Negative-control label matches are coincidences.',
        'summary':summary,'rows':rows,'author_training_metadata_file':'author_temporal_metadata.json','protected_source_sha256':protected,
        'protected_sources_unchanged':True,'frozen_replay_max_score_error':replay_error,'weights_changed':False,'model_trained':False,'active_model_replaced':False,'runtime_seconds':time.perf_counter()-started}
    atomic_json(out/'result.json',result);print('SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)
    print('OUTPUT',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--replay',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    replay(a.output) if a.replay else run(a.output)
