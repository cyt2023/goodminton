"""Frozen BST embeddings + a small ridge classification head trained on HUMAN labels.

No model predictions enter the targets. A single-video development run does
not establish generalisation; predictions stay in a comparison sidecar.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from analysis_jobs import DATA, atomic_json, file_hash
from specialist_bst import SpecialistBST
from specialist_model import MODEL_REVISION, ROOT, BST_NAME
from stroke_review import folder_for, load_source, read_labels, learning_readiness, usable_training_label
from stroke_taxonomy import TRAINABLE, LABELS, VERSION


def state(folder, status, message, **extra):
    atomic_json(folder/'learning_state.json', {'status':status, 'message':message,
                'updated_at':datetime.now(timezone.utc).isoformat(), **extra})


def extractor(job):
    folder=DATA/'专项分析'/job
    dense=json.loads((folder/'dense_pose.json').read_text())
    trajectory=json.loads((folder/'trajectory.json').read_text())
    return SpecialistBST(dense,trajectory), dense


def embedding(classifier, values, target_player):
    captured=[]
    hook=classifier.model.mlp_head.register_forward_pre_hook(lambda module,args: captured.append(args[0].detach().cpu().numpy()))
    j,p,s,length=values
    try:
        with torch.inference_mode():
            classifier.model(torch.from_numpy(j).unsqueeze(0),torch.from_numpy(s).unsqueeze(0),torch.tensor([length]))
    finally:
        hook.remove()
    value=captured[0][0].astype('float32')
    if value.shape!=(300,) or not np.isfinite(value).all():
        raise ValueError('BST特征形状或数值无效')
    # The original head concatenates p1(100), p2(100), shuttle(100).
    # Put the human-selected target first so both players share one head.
    if target_player == classifier.order[1]:
        value=np.concatenate([value[100:200],value[:100],value[200:]])
    return value


def prepare(job, classifier=None):
    report,events,result_path=load_source(job)
    folder=folder_for(job);folder.mkdir(parents=True,exist_ok=True)
    metadata={'source_sha256':report['source_sha256'],'model_revision':MODEL_REVISION,
              'candidate_results_sha256':file_hash(result_path),
              'extractor_code_sha256':file_hash(Path(__file__)),
              'model_weight_sha256':file_hash(ROOT/'models/bst'/BST_NAME),
              'feature':'BST-0 frozen pre-MLP 300D representation; target player first'}
    if (folder/'features.json').is_file() and (folder/'features.npz').is_file():
        cached=json.loads((folder/'features.json').read_text())
        if all(cached.get(k)==v for k,v in metadata.items()):
            with np.load(folder/'features.npz',allow_pickle=False) as f:
                return list(f['event_ids']),f['features'].copy(),metadata
    if classifier is None: classifier,_=extractor(job)
    ids,features=[],[]
    for n,event in enumerate(events):
        file=result_path.parent/event['id']/'bst_inputs.npz'
        if not file.is_file():continue
        with np.load(file,allow_pickle=False) as d:
            values=(d['joints_and_bones'],d['positions'],d['shuttle'],int(d['video_len']))
        ids.append(event['id']);features.append(embedding(classifier,values,event['player']))
        print('FEATURE_PROGRESS '+json.dumps({'processed':n+1,'total':len(events)}),flush=True)
    if not ids:raise ValueError('没有可计算的模型输入')
    x=np.asarray(features,dtype='float32')
    temp=folder/'features.partial.npz'
    np.savez_compressed(temp,event_ids=np.asarray(ids),features=x)
    temp.replace(folder/'features.npz')
    atomic_json(folder/'features.json',{**metadata,'event_count':len(ids),'dimension':300,
                'created_at':datetime.now(timezone.utc).isoformat(),'trained':False})
    return ids,x,metadata


def train(job):
    folder=folder_for(job);folder.mkdir(parents=True,exist_ok=True)
    lock=(DATA/'击球标注'/'.learning.lock').open('a+b')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.perf_counter()
    report,events,_=load_source(job)
    doc=read_labels(job,report['source_sha256'])
    from benchmarks.isolation import assert_training_allowed
    assert_training_allowed(doc)
    if not learning_readiness(doc)['can_train']:
        raise ValueError(learning_readiness(doc)['requirement'])
    run_id=uuid.uuid4().hex;run=folder/'学习记录'/run_id;run.mkdir(parents=True)
    atomic_json(run/'human_labels_snapshot.json',doc)
    state(folder,'running','正在从已确认动作提取特征',run_id=run_id,dataset_revision=doc['revision'])
    classifier,dense=extractor(job)
    ids,all_features,feature_metadata=prepare(job,classifier)
    indexed={e['id']:e for e in events}
    times=np.asarray([f['source_time'] for f in dense['frames']])
    features,targets,train_ids,windows=[],[],[],[]
    for label in doc['labels'].values():
        if not usable_training_label(label):continue
        event=dict(indexed[label['event_id']])
        # Recompute using human window/player/contact, rather than silently
        # fitting stale features extracted from the wrong automatic window.
        event.update(player=label['player'],
                     start_frame=int(np.searchsorted(times,label['start'],side='left')),
                     end_frame=min(len(times)-1,int(np.searchsorted(times,label['end'],side='right'))-1),
                     frame_index=int(np.argmin(abs(times-label['peak_time']))))
        event['end_frame']=max(event['start_frame'],event['end_frame'])
        values,quality=classifier.inputs(event)
        if quality['paired_player_fraction']<.5:
            continue
        features.append(embedding(classifier,values,label['player']));targets.append(label['stroke_type']);train_ids.append(event['id'])
        windows.append({'event_id':event['id'],'player':label['player'],'start':label['start'],
                        'end':label['end'],'peak_time':label['peak_time'],'contact_confirmed':label['contact_confirmed'],
                        'input_quality':quality})
    counts={code:targets.count(code) for code in set(targets)}
    if len(targets)<12 or sum(n>=3 for n in counts.values())<2:
        raise ValueError('姿态缺失过滤后可用标注不足，请补充标注。')
    classes=[c for c in TRAINABLE if c in counts]
    x=np.asarray(features,dtype='float64');mean=x.mean(0);scale=x.std(0);scale=np.where(scale<1e-5,1,scale)
    z=np.concatenate([(x-mean)/scale,np.ones((len(x),1))],axis=1)
    y=np.eye(len(classes))[np.asarray([classes.index(t) for t in targets])]
    alpha=10.
    # Ridge head with a bias coordinate. Model targets are exclusively user
    # selections; a small closed-form fit is feasible on a MacBook Air.
    weights=z.T@np.linalg.solve(z@z.T+alpha*np.eye(len(z)),y)
    np.savez_compressed(run/'personal_head.npz',weights=weights,mean=mean,scale=scale,classes=np.asarray(classes))
    all_z=np.concatenate([(all_features-mean)/scale,np.ones((len(all_features),1))],axis=1)
    scores=all_z@weights
    predictions=[]
    for eid,row in zip(ids,scores):
        order=np.argsort(-row)[:3]
        predictions.append({'event_id':eid,'stroke_type':classes[int(order[0])],
                            'stroke_label':LABELS[classes[int(order[0])]],
                            'ranking':[{'code':classes[int(i)],'label':LABELS[classes[int(i)]],
                                        'uncalibrated_score':float(row[i])} for i in order],
                            'used_for_training':eid in train_ids,'review_status':'unreviewed_personal_model_candidate'})
    original_match=(np.argmax(z@weights,axis=1)==np.asarray([classes.index(t) for t in targets])).mean()
    manifest={'run_id':run_id,'job_id':job,'source_sha256':report['source_sha256'],
              'dataset_revision':doc['revision'],'taxonomy_version':VERSION,
              'algorithm':'frozen BST-0 300D features + ridge classification head','alpha':alpha,
              'training_target_source':'human_user_selection_only','trained_event_ids':train_ids,
              'class_counts':counts,'trained_classes':classes,'feature_metadata':feature_metadata,
              'training_windows':windows,'training_label_sha256':file_hash(run/'human_labels_snapshot.json'),
              'training_agreement_on_training_windows':float(original_match),
              'independent_validation':None,'evaluation_scope':'single-video development data, overlapping events may exist',
              'global_model_activated':False,'model_file':str((run/'personal_head.npz').relative_to(DATA)),
              'model_sha256':file_hash(run/'personal_head.npz'),
              'runtime_seconds':round(time.perf_counter()-started,3),
              'created_at':datetime.now(timezone.utc).isoformat(),
              'note':'Predictions are limited to observed training classes; training agreement is not generalisation accuracy. Original pretrained weights remain unchanged.'}
    atomic_json(run/'manifest.json',manifest)
    atomic_json(folder/'personal_predictions.json',{'source_sha256':report['source_sha256'],
                'dataset_revision':doc['revision'],'run_id':run_id,'events':predictions,
                'independent_validation':None,'global_model_activated':False})
    state(folder,'completed','个人球种分类头已学习；学习结果待核对，尚未在新录像验证。',
          run_id=run_id,dataset_revision=doc['revision'],trained_labels=len(targets),
          trained_classes=[LABELS[c] for c in classes],global_model_activated=False,
          manifest_file=str((run/'manifest.json').relative_to(DATA)))
    lock.close()


def train_local(job,experimental_context=False):
    if experimental_context:
        from context_features import ContextFeatures as FeatureExtractor, VERSION as FEATURE_VERSION
    else:
        from stroke_features import HitFeatures as FeatureExtractor, VERSION as FEATURE_VERSION
    from personal_head import fit, scores, select, temporal_groups, save
    folder=folder_for(job);folder.mkdir(parents=True,exist_ok=True)
    with (DATA/'击球标注'/'.learning.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.perf_counter()
        report,events,result_path=load_source(job)
        doc=read_labels(job,report['source_sha256'])
        from benchmarks.isolation import assert_training_allowed
        assert_training_allowed(doc)
        if not learning_readiness(doc)['can_train']:
            raise ValueError(learning_readiness(doc)['requirement'])
        run_id=uuid.uuid4().hex;run=folder/'学习记录'/run_id;run.mkdir(parents=True)
        atomic_json(run/'human_labels_snapshot.json',doc)
        state(folder,'running','正在围绕本次击球提取姿态和轨迹特征',
              run_id=run_id,dataset_revision=doc['revision'])
        classifier,_=extractor(job);features=FeatureExtractor(classifier)
        ids=[];vectors=[];quality=[];targets=[];train_ids=[];train_labels=[];excluded=[]
        indexed={e['id']:e for e in events}
        # Human-adjusted player/contact/window are used consistently at fitting
        # AND replay inference. Unknown/other labels never become targets.
        for original in events:
            label=doc['labels'].get(original['id'])
            event={**original,**label} if label and not label.get('wrong_target') else dict(original)
            try:
                vector,q=features.extract(event)
            except ValueError as error:
                excluded.append({'event_id':original['id'],'reason':str(error)});continue
            ids.append(original['id']);vectors.append(vector);quality.append(q)
            if label and usable_training_label(label):
                if q['target_pose_fraction']>=.5:
                    targets.append(label['stroke_type']);train_ids.append(original['id'])
                    train_labels.append({**label,'start':min(label['start'],q.get('context_start',q['feature_start'])),
                                         'end':max(label['end'],q.get('context_end',q['feature_end']))})
                else:excluded.append({'event_id':original['id'],'reason':'当前球员有效姿态不足50%'})
        x=np.asarray(vectors,float);lookup={eid:i for i,eid in enumerate(ids)}
        training=x[[lookup[eid] for eid in train_ids]]
        counts={code:targets.count(code) for code in set(targets)}
        if len(targets)<12 or sum(n>=3 for n in counts.values())<2:
            raise ValueError('当前球员姿态过滤后可用标注不足，请补充标注。')
        groups,bounds=temporal_groups(events,train_labels)
        train_groups=np.asarray([groups[eid] for eid in train_ids])
        state(folder,'running','正在按不重叠片段选择个人分类头',
              run_id=run_id,dataset_revision=doc['revision'])
        config,selection=select(training,targets,train_groups)
        model=fit(training,targets,config);save(run/'personal_head.npz',model)
        np.savez_compressed(run/'features.npz',event_ids=np.asarray(ids),features=x,
                            trained_event_ids=np.asarray(train_ids),targets=np.asarray(targets),groups=train_groups)
        all_scores=scores(model,x);predictions=[]
        for eid,row,q in zip(ids,all_scores,quality):
            order=np.argsort(-row)[:3];code=str(model['classes'][order[0]])
            reasons=[]
            if counts[code]<3:reasons.append('该类训练样本少于3条')
            if q['target_pose_fraction']<.5:reasons.append('当前球员姿态不足')
            if q['filtered_raw_shuttle_fraction']<.25:reasons.append('可用球轨迹较少')
            predictions.append({'event_id':eid,'stroke_type':code,'stroke_label':LABELS[code],
                                'ranking':[{'code':str(model['classes'][i]),'label':LABELS[model['classes'][i]],
                                            'uncalibrated_score':float(row[i])} for i in order],
                                'used_for_training':eid in train_ids,'review_status':'unreviewed_personal_model_candidate',
                                'needs_confirmation':True,'quality_reasons':reasons,'input_quality':q,
                                'human_reviewed_input':eid in doc['labels']})
        training_guesses=model['classes'][scores(model,training).argmax(1)]
        manifest={'run_id':run_id,'job_id':job,'source_sha256':report['source_sha256'],
                  'dataset_revision':doc['revision'],'taxonomy_version':VERSION,'feature_version':FEATURE_VERSION,
                  'algorithm':('725D experimental context candidates' if experimental_context else '380D local hit BST/geometry candidates')+'; training-only segment head selection',
                  'experimental_context':experimental_context,
                  'default_feature_policy':'Keep local route; unbounded context ablation was 18/62 versus local 22/62 at revision 77.',
                  'training_target_source':'human_user_selection_only','selected_config':config,
                  'head_selection_results':selection,'selection_scope':'training CV used for model choice, not independent accuracy',
                  'trained_event_ids':train_ids,'class_counts':counts,'trained_classes':list(model['classes']),
                  'source_segment_bounds':bounds,'excluded_labels':excluded,
                  'input_quality':[{'event_id':eid,**q} for eid,q in zip(ids,quality)],
                  'training_agreement_on_training_windows':float((training_guesses==np.asarray(targets)).mean()),
                  'training_label_sha256':file_hash(run/'human_labels_snapshot.json'),
                  'model_sha256':file_hash(run/'personal_head.npz'),
                  'code_sha256':{name:file_hash(Path(__file__).with_name(name))
                                 for name in ['personal_learning.py','stroke_features.py','context_features.py','personal_head.py','hit_refinement.py','specialist_bst.py']},
                  'input_sha256':{name:file_hash(DATA/'专项分析'/job/name) for name in ['dense_pose.json','trajectory.json']},
                  'candidate_results_sha256':file_hash(result_path),
                  'model_weight_sha256':file_hash(ROOT/'models/bst'/BST_NAME),
                  'independent_validation':None,'global_model_activated':False,
                  'runtime_seconds':round(time.perf_counter()-started,3),
                  'created_at':datetime.now(timezone.utc).isoformat()}
        atomic_json(run/'manifest.json',manifest)
        atomic_json(folder/'personal_predictions.json',{'source_sha256':report['source_sha256'],
                    'dataset_revision':doc['revision'],'run_id':run_id,'feature_version':FEATURE_VERSION,'events':predictions,
                    'independent_validation':None,'global_model_activated':False})
        state(folder,'running','个人模型已拟合，正在按不重叠片段检查识别结果',
              run_id=run_id,dataset_revision=doc['revision'])
        from evaluate_improved_learning import evaluate_current
        evaluate_current(job,run_id=run_id)
        state(folder,'completed',('已结合短时动作和实验球路特征学习' if experimental_context else '已用已确认击球的局部姿态与过滤轨迹学习')+'；结果仍需核对，尚未在新录像验证。',
              run_id=run_id,dataset_revision=doc['revision'],trained_labels=len(targets),
              trained_classes=[LABELS[c] for c in model['classes']],feature_version=FEATURE_VERSION,
              global_model_activated=False,manifest_file=str((run/'manifest.json').relative_to(DATA)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);parser.add_argument('--prepare',action='store_true');parser.add_argument('--legacy',action='store_true');parser.add_argument('--experimental-context',action='store_true')
    args=parser.parse_args()
    try:
        if args.prepare:
            ids,features,_=prepare(args.job);print('FEATURE_DONE '+json.dumps({'events':len(ids),'dimensions':features.shape[1],'trained':False}),flush=True)
        elif args.legacy:train(args.job)
        else:train_local(args.job,experimental_context=args.experimental_context)
    except Exception as error:
        if not args.prepare:
            state(folder_for(args.job),'failed','学习未完成：'+str(error)[:500])
        raise
