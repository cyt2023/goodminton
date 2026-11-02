"""Nested segment holdout on a fixed development benchmark; no activation."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np

from analysis_jobs import atomic_json,file_hash
from personal_learning import extractor
from personal_head import fit,scores,select
from stroke_features import HitFeatures,VERSION
from stroke_review import folder_for,load_source
from stroke_taxonomy import LABELS


def nested_rows(x,targets,groups,ids):
    predictions=[];folds=[]
    for group in sorted(set(groups)):
        train=groups!=group;test=~train
        if not train.any():raise ValueError('独立时间片段组不足')
        config,inner=select(x[train],targets[train],groups[train])
        model=fit(x[train],targets[train],config)
        guesses=model['classes'][scores(model,x[test]).argmax(1)]
        for i,pred in zip(np.flatnonzero(test),guesses):
            predictions.append({'event_id':ids[i],'human_code':str(targets[i]),
                                'human_label':LABELS[targets[i]],'prediction_code':str(pred),
                                'prediction_label':LABELS[pred],'match':bool(pred==targets[i]),
                                'group':int(group),'class_seen':bool(targets[i] in model['classes'])})
        folds.append({'held_out_group':int(group),'chosen_config':config,'inner_selection':inner})
    return predictions,folds


def class_results(predictions,targets):
    return {LABELS[c]:{'samples':sum(p['human_code']==c for p in predictions),
                      'correct':sum(p['human_code']==c and p['match'] for p in predictions)}
            for c in sorted(set(targets))}


def evaluate(job,baseline_run):
    folder=folder_for(job);base=folder/'学习记录'/baseline_run
    old=json.loads((base/'留出评估/result.json').read_text())
    snapshot=json.loads((base/'human_labels_snapshot.json').read_text())
    with np.load(base/'留出评估/evaluation_features.npz',allow_pickle=False) as d:
        ids=list(d['event_ids']);groups=d['segment_groups'];targets=d['targets']
    _,events,result_path=load_source(job);indexed={e['id']:e for e in events}
    classifier,_=extractor(job);features=HitFeatures(classifier)
    xs=[];quality=[]
    for eid in ids:
        event={**indexed[eid],**snapshot['labels'][eid]}
        x,q=features.extract(event);xs.append(x);quality.append({'event_id':eid,**q})
    x=np.asarray(xs,float)
    predictions,folds=nested_rows(x,targets,groups,ids)
    result={'created_at':datetime.now(timezone.utc).isoformat(),'job_id':job,
            'baseline_run_id':baseline_run,'dataset_revision':old['dataset_revision'],
            'source_sha256':old['source_sha256'],'feature_version':VERSION,
            'scope':'fixed same-video development benchmark; nested connected-segment CV; not new-video validation',
            'evaluated_labels':len(ids),'evaluated_segment_groups':len(folds),
            'baseline_correct':old['correct'],'correct':sum(p['match'] for p in predictions),
            'events':predictions,'folds':folds,'input_quality':quality,
            'class_results':class_results(predictions,targets),
            'selection':'Each outer test segment is excluded from feature normalization, head fitting and inner model selection.',
            'evaluation_code_sha256':file_hash(Path(__file__)),
            'feature_code_sha256':file_hash(Path(__file__).with_name('stroke_features.py')),
            'head_code_sha256':file_hash(Path(__file__).with_name('personal_head.py')),
            'baseline_result_sha256':file_hash(base/'留出评估/result.json'),
            'candidate_results_sha256':file_hash(result_path),
            'global_model_activated':False}
    result['pooled_accuracy']=result['correct']/len(ids)
    out=base/'改进对照';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'features.npz',event_ids=np.asarray(ids),features=x,targets=targets,groups=groups)
    atomic_json(out/'result.json',result);atomic_json(folder/'improvement_evaluation.json',result)
    print(json.dumps({k:result[k] for k in ['evaluated_labels','baseline_correct','correct','pooled_accuracy','class_results']},ensure_ascii=False))


def evaluate_current(job,run_id=None):
    folder=folder_for(job);state=json.loads((folder/'learning_state.json').read_text())
    if run_id is None and state['status']!='completed':raise ValueError('请先完成学习')
    active_run=run_id or state['run_id']
    if run_id is not None and run_id!=state.get('run_id'):raise ValueError('学习任务已改变')
    run=folder/'学习记录'/active_run;manifest=json.loads((run/'manifest.json').read_text())
    with np.load(run/'features.npz',allow_pickle=False) as d:
        all_ids=list(d['event_ids']);ids=list(d['trained_event_ids'])
        x=d['features'][[all_ids.index(eid) for eid in ids]]
        targets=d['targets'];groups=d['groups']
    predictions,folds=nested_rows(x,targets,groups,ids)
    result={'job_id':job,'run_id':active_run,'dataset_revision':manifest['dataset_revision'],
            'source_sha256':manifest['source_sha256'],'feature_version':manifest['feature_version'],
            'scope':'same-video nested connected-segment CV; NOT new-video validation',
            'created_at':datetime.now(timezone.utc).isoformat(),'evaluated_labels':len(ids),
            'evaluated_segment_groups':len(folds),'correct':sum(p['match'] for p in predictions),
            'events':predictions,'folds':folds,'per_class':class_results(predictions,targets),
            'training_manifest_sha256':file_hash(run/'manifest.json'),
            'evaluation_code_sha256':file_hash(Path(__file__)),
            'model_selection_inside_training_folds_only':True,'global_model_activated':False}
    result['pooled_accuracy']=result['correct']/len(ids)
    out=run/'留出评估';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'evaluation_features.npz',event_ids=np.asarray(ids),features=x,targets=targets,segment_groups=groups)
    atomic_json(out/'result.json',result);atomic_json(folder/'latest_evaluation.json',result)
    # The summary travels with the exact prediction run, including when an
    # older server is processing another video and cannot be restarted yet.
    prediction_file=folder/'personal_predictions.json'
    if prediction_file.is_file():
        prediction_doc=json.loads(prediction_file.read_text())
        if prediction_doc.get('run_id')==active_run:
            prediction_doc['evaluation']={k:result[k] for k in
                                          ['run_id','dataset_revision','scope','correct',
                                           'evaluated_labels','pooled_accuracy','evaluated_segment_groups']}
            atomic_json(prediction_file,prediction_doc)
    print(json.dumps({k:result[k] for k in ['run_id','dataset_revision','correct','evaluated_labels','pooled_accuracy']},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--baseline-run',required=True)
    a=p.parse_args();evaluate(a.job,a.baseline_run)
