"""Fixed-label context ablation. Development-only; never activates a model.

Old human stroke labels are reused without pretending their contact centers
have been confirmed. Every variant uses the same outer groups and labels.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np

from analysis_jobs import DATA,atomic_json,file_hash
from evaluate_improved_learning import nested_rows
from stroke_review import folder_for
from stroke_taxonomy import LABELS


def metrics(rows):
    classes=sorted({r['human_code'] for r in rows})
    confusion={c:{p:0 for p in classes} for c in classes}
    for r in rows:
        confusion[r['human_code']].setdefault(r['prediction_code'],0)
        confusion[r['human_code']][r['prediction_code']]+=1
    per_class={}
    for c in classes:
        tp=sum(r['human_code']==c and r['prediction_code']==c for r in rows)
        support=sum(r['human_code']==c for r in rows)
        predicted=sum(r['prediction_code']==c for r in rows)
        precision=tp/predicted if predicted else 0.
        recall=tp/support if support else 0.
        per_class[c]={'label':LABELS[c],'samples':support,'correct':tp,
                      'precision':precision,'recall':recall,
                      'f1':2*precision*recall/(precision+recall) if precision+recall else 0.}
    stroke=[r for r in rows if r['human_code']!='no_stroke']
    pairs=Counter((r['human_code'],r['prediction_code']) for r in rows if not r['match'])
    return {'correct':sum(r['match'] for r in rows),'total':len(rows),
            'accuracy':sum(r['match'] for r in rows)/max(1,len(rows)),
            'macro_f1':float(np.mean([r['f1'] for r in per_class.values()])) if per_class else 0.,
            'stroke_only_correct':sum(r['match'] for r in stroke),'stroke_only_total':len(stroke),
            'stroke_only_accuracy':sum(r['match'] for r in stroke)/max(1,len(stroke)),
            'classes':per_class,'confusion':confusion,
            'unseen_class_examples':sum(not r['class_seen'] for r in rows),
            'largest_confusions':[{'human':a,'prediction':b,'human_label':LABELS[a],
                                   'prediction_label':LABELS[b],'count':n}
                                  for (a,b),n in pairs.most_common(10)]}


def protected_groups(ids,old_groups,qualities):
    """Merge old folds if expanded feature supports would overlap them."""
    parent={int(g):int(g) for g in old_groups}
    def root(g):
        while parent[g]!=g:g=parent[g]
        return g
    overlaps=[]
    for i,a in enumerate(ids):
        qa=qualities[a]
        for j,b in enumerate(ids[i+1:],i+1):
            qb=qualities[b]
            if (old_groups[i]!=old_groups[j] and qa['context_start']<qb['context_end']
                    and qb['context_start']<qa['context_end']):
                overlaps.append([a,b,int(old_groups[i]),int(old_groups[j])])
                parent[root(int(old_groups[j]))]=root(int(old_groups[i]))
    groups=np.asarray([root(int(g)) for g in old_groups])
    if len(set(groups))<3:raise ValueError('Expanded context leaves too few independent temporal groups')
    return groups,overlaps


def evaluate(job,baseline_run):
    folder=folder_for(job);base=folder/'学习记录'/baseline_run
    manifest=json.loads((base/'manifest.json').read_text())
    snapshot=json.loads((base/'human_labels_snapshot.json').read_text())
    context_folder=DATA/'专项分析'/job
    context=json.loads((context_folder/'context_features.json').read_text())
    if context['source_sha256']!=manifest['source_sha256'] or context['dataset_revision']!=snapshot['revision']:
        raise ValueError('Context features and frozen label version must match')
    with np.load(base/'features.npz',allow_pickle=False) as d:
        ids=[str(v) for v in d['trained_event_ids']];all_ids=[str(v) for v in d['event_ids']]
        old=d['features'][[all_ids.index(eid) for eid in ids]].astype(float)
        targets=d['targets'];old_groups=d['groups']
    with np.load(context_folder/'context_features.npz',allow_pickle=False) as d:
        cids=[str(v) for v in d['event_ids']]
        new=d['features'][[cids.index(eid) for eid in ids]].astype(float)
    for eid,target in zip(ids,targets):
        if snapshot['labels'][eid]['stroke_type']!=target:raise ValueError('Frozen targets differ')
    quality={q['event_id']:q for q in context['quality']}
    groups,overlaps=protected_groups(ids,old_groups,quality)
    variants={}
    for name,x in [('legacy_local_380',old),('revised_local_380',new[:,:380]),('context_candidates_725',new)]:
        print('EVALUATING',name,'samples',len(ids),'groups',len(set(groups)),flush=True)
        rows,folds=nested_rows(x,targets,groups,ids)
        variants[name]={'metrics':metrics(rows),'events':rows,'folds':folds,
                        'selected_features_by_fold':dict(Counter(f['chosen_config']['features'] for f in folds))}
        print('RESULT',name,json.dumps({k:variants[name]['metrics'][k] for k in
                    ['correct','total','accuracy','macro_f1','stroke_only_correct','stroke_only_total']},ensure_ascii=False),flush=True)
    result={'job_id':job,'baseline_run_id':baseline_run,'created_at':datetime.now(timezone.utc).isoformat(),
            'dataset_revision':snapshot['revision'],'source_sha256':manifest['source_sha256'],
            'scope':'Fixed historical human stroke labels, nested temporal group development ablation; NOT independent-video accuracy or verified contact evaluation.',
            'eligible_for_next_training':False,'contact_truth_confirmed':False,'active_model_replaced':False,
            'evaluation_only_fold_models_discarded':True,'human_labels_created':0,'pseudo_labels_created':0,
            'sample_ids':ids,'targets':[str(c) for c in targets],'groups':groups.tolist(),
            'original_group_count':len(set(old_groups)),'protected_group_count':len(set(groups)),
            'overlap_merges':overlaps,'variants':variants,
            'selection_note':'Normalization, head fitting and hyperparameter choice only use outer-training samples. Macro F1 averages the 14 supported labels, including labels absent from some training folds.',
            'input_sha256':{str(p.relative_to(DATA)):file_hash(p) for p in
                   [base/'features.npz',base/'human_labels_snapshot.json',context_folder/'context_features.npz',context_folder/'context_features.json']},
            'code_sha256':{n:file_hash(Path(__file__).with_name(n)) for n in
                   ['evaluate_context_learning.py','evaluate_improved_learning.py','personal_head.py']}}
    out=folder/'开发对照'/('上下文_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(parents=True)
    atomic_json(out/'result.json',result)
    np.savez_compressed(out/'fixed_features.npz',event_ids=np.asarray(ids),targets=targets,groups=groups,
                        legacy_features=old,context_features=new)
    atomic_json(folder/'context_development_evaluation.json',result)
    print('SAVED',out,flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);parser.add_argument('--baseline-run',required=True)
    args=parser.parse_args();evaluate(args.job,args.baseline_run)
