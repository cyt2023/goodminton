"""Rebuild saved fixed guard diagnostics; no inference, fitting or deployment."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/diagnose/
from collections import Counter
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from analysis_jobs import DATA, ROOT, atomic_json, file_hash
from personal_learning import extractor
from stroke_context import guard_confirmed_single_hit, window, VERSION
from stroke_review import load_source, read_labels
from stroke_window_repairs import translate_contact_window

OUT=DATA/'击球标注/根因诊断/2026-10-14_单拍窗口约束接入验证'
SNAPSHOT=ROOT.parent/'05_研究记录/代码快照/2026-10-14_单拍窗口约束前/stroke_context.py'
FIXED70=DATA/'击球标注/根因诊断/2026-10-13_同方拍边界约束_全70'
FIXED22=DATA/'击球标注/根因诊断/2026-10-13_同方拍边界约束'
RUN=DATA/'击球标注/直接BST微调/运行记录/d27cd4b69cbb4a5d8fd9836705f6466c'


def fields(w):
    return [w['start_frame'],w['end_frame'],w['frame_index']]


def numerical_error(v,cache,i):
    return {k:float(np.max(abs(np.asarray(a)-cache[k][i]))) for a,k in zip(v,['joints_and_bones','positions','shuttle','video_len'])}


def fixtures():
    dense={'frames':[{'source_time':i/10} for i in range(41)],'sample_fps':10.,'end':4.1}
    base={'start_frame':0,'end_frame':40,'frame_index':20,'peak_time':2.,'start':0.,'end':4.1,'basis':'fixture'}
    def label(e,t,**extra):
        return {'event_id':e,'peak_time':t,'player':'near','stroke_type':'smash','contact_confirmed':True,**extra}
    labels={'prev':label('prev',1.,play_context='rally'),'next':label('next',3.,play_context='rally'),
        'same':label('same',2.1),'duplicate_prev':label('duplicate_prev',1.,play_context='rally')}
    tests=[]
    w=guard_confirmed_single_hit(dense,base,'near',labels)
    assert fields(w)==[15,27,20] and w['boundary_confirmed'] is False
    assert w['single_hit_guard']['duplicate_same_frame_annotation_count']==1
    assert w['single_hit_guard']['previous_same_player_anchor']['event_ids']==['duplicate_prev','prev']
    assert w['single_hit_guard']['next_same_player_anchor']['frame_index']==30
    tests.append('fixed midpoint/.25s formula; exact-frame duplicate grouped; nearby same-hit excluded')
    assert guard_confirmed_single_hit(dense,w,'near',labels)==w
    assert fields(guard_confirmed_single_hit(dense,w,'near',{}))==[0,40,20]
    tests.append('idempotent; removing anchors restores original translated support')
    uncertain={'p':label('p',1.),'n':label('n',3.,play_context='unknown')}
    u=guard_confirmed_single_hit(dense,base,'near',uncertain)
    assert fields(u)==[15,27,20] and u['context_unverified_anchor_count']==2
    tests.append('missing and explicit unknown rally contexts retained but both marked unverified')
    excluded={str(i):label(str(i),t,**extras) for i,(t,extras) in enumerate([
        (1.,{'play_context':'between_rallies'}),(3.,{'wrong_target':True}),
        (1.,{'contact_confirmed':False}),(3.,{'stroke_type':'unknown'}),
        (1.,{'stroke_type':'no_stroke'}),(3.,{'player':'far'})])}
    assert fields(guard_confirmed_single_hit(dense,base,'near',excluded))==[0,40,20]
    tests.append('between-rallies, wrong target, unconfirmed, unknown/no-stroke and opponent anchors excluded')
    start={**base,'frame_index':0,'peak_time':0.}
    end={**base,'frame_index':40,'peak_time':4.}
    assert fields(guard_confirmed_single_hit(dense,start,'near',{'n':label('n',1.)}))==[0,7,0]
    assert fields(guard_confirmed_single_hit(dense,end,'near',{'p':label('p',3.)}))==[35,40,40]
    tests.append('recording start/end keep contact and valid frame bounds')
    near={'a':label('a',1.9),'b':label('b',2.),'c':label('c',2.1)}
    assert fields(guard_confirmed_single_hit(dense,base,'near',near))==[0,40,20]
    tests.append('distinct nearby frames remain recorded but +/-0.15s cannot constrain the same hit')
    return tests


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('pre_guard_stroke_context',SNAPSHOT)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    models={};docs={};events={};automatic=[];full_path=[];results={}
    def source(job):
        if job not in models:
            models[job]=extractor(job);report,ee,_=load_source(job)
            docs[job]=read_labels(job,report['source_sha256']);events[job]={e['id']:e for e in ee}
            dense=models[job][1]
            for e in ee:
                a=old.window(dense,e,ee,{},reviewed=False);b=window(dense,e,ee,{},reviewed=False)
                automatic.append({'job_id':job,'event_id':e['id'],'matches_numeric_support':all(a[k]==b[k] for k in ['start_frame','end_frame','frame_index','start','end','peak_time','duration'])})
        return models[job],docs[job],events[job]
    for name,root,base_key,target_key in [('fixed70',FIXED70,'baseline_window','guard_window'),('fixed22',FIXED22,'baseline_window','guarded_window')]:
        saved=json.loads((root/'result.json').read_text());rows=[]
        with np.load(root/'inputs.npz',allow_pickle=False) as c:
            cache={k:c[k].copy() for k in c.files}
        for i,r in enumerate(saved['rows']):
            job=r['job_id'];(clf,dense),doc,indexed=source(job)
            h=doc['labels'][r['event_id']]
            if name=='fixed22':
                prior=json.loads((DATA/'击球标注/输入修复诊断/2026-10-12_配对输入修复/result.json').read_text())
                base=next(v['human_shifted_window'] for v in prior['rows'] if v['event_id']==r['event_id'] and v['job_id']==job)
            else:base=r[base_key]
            expected=r[target_key]
            rebuilt=guard_confirmed_single_hit(dense,base,h['player'],doc['labels'],r['event_id'])
            clf.order=list(r['quality']['player_input_order'])
            v,q=clf.inputs(rebuilt)
            errors=numerical_error(v,cache,i)
            full=window(dense,indexed[r['event_id']],list(indexed.values()),doc['labels'])
            fv,_=clf.inputs(full)
            full_errors=numerical_error(fv,cache,i)
            old_base=old.window(dense,indexed[r['event_id']],list(indexed.values()),doc['labels'])
            same_base=fields(old_base)==fields(base)
            row={'job_id':job,'event_id':r['event_id'],'helper_frames':fields(rebuilt),'saved_frames':fields(expected),
                'helper_frames_match':fields(rebuilt)==fields(expected),'helper_max_errors':errors,
                'full_path_frames':fields(full),'full_path_matches_saved':fields(full)==fields(expected) and max(full_errors.values())==0.,
                'full_path_max_errors':full_errors,'old_full_path_translated_frames':fields(old_base),
                'saved_translated_frames':fields(base),'base_sources_match':same_base,
                'guarded_pair_fraction':q['paired_player_fraction'],
                'context_unverified_anchor_count':rebuilt['context_unverified_anchor_count'],
                'guard_metadata':rebuilt['single_hit_guard']}
            rows.append(row)
        results[name]={'total':len(rows),'helper_frames_matches':sum(r['helper_frames_match'] for r in rows),
            'helper_inputs_exact':sum(max(r['helper_max_errors'].values())==0 for r in rows),
            'full_path_inputs_exact':sum(r['full_path_matches_saved'] for r in rows),
            'guarded_pair_below_half':sum(r['guarded_pair_fraction']<.5 for r in rows),
            'with_unverified_context_anchors':sum(r['context_unverified_anchor_count']>0 for r in rows),'rows':rows}
    features=json.loads((RUN/'personal_features_manifest.json').read_text());reconsidered=[]
    for excluded in features['excluded']:
        if excluded['reason']!='paired_pose_coverage_below_50_percent':continue
        job=excluded['job_id'];(clf,dense),doc,indexed=source(job);h=doc['labels'][excluded['event_id']]
        frozen_file=DATA/'击球标注'/job/'public_only_predictions.json'
        frozen=json.loads(frozen_file.read_text()) if frozen_file.exists() else None
        f=next((r for r in frozen['events'] if r['event_id']==excluded['event_id']),None) if frozen else None
        original=f['model_window'] if f else old.window(dense,indexed[excluded['event_id']],list(indexed.values()),{},reviewed=False)
        if f:clf.order=list(f['input_quality']['player_input_order'])
        base=translate_contact_window(dense,original,h['peak_time'])
        g=guard_confirmed_single_hit(dense,base,h['player'],doc['labels'],excluded['event_id'])
        _,q=clf.inputs(g)
        reconsidered.append({'job_id':job,'event_id':excluded['event_id'],'previous_pair_fraction':excluded['quality']['paired_player_fraction'],
            'guarded_pair_fraction':q['paired_player_fraction'],'meets_half_now':q['paired_player_fraction']>=.5,
            'translated_frames':fields(base),'guarded_frames':fields(g)})
    report={'version':VERSION,'source_snapshot_sha256':file_hash(SNAPSHOT),'new_window_code_sha256':file_hash(ROOT/'stroke_context.py'),
        'fixed_comparisons':results,'automatic_unreviewed_total':len(automatic),
        'automatic_numeric_support_matches':sum(r['matches_numeric_support'] for r in automatic),
        'automatic_mismatches':[r for r in automatic if not r['matches_numeric_support']],
        'synthetic_checks':fixtures(),'previously_excluded_three':reconsidered,
        'model_inference_run':False,'model_trained':False,'production_labels_or_predictions_changed':False,
        'fixed_denominator_policy':'Keep original70 and original22 diagnostics fixed even if quality eligibility changes; next training may have a different accepted denominator.'}
    atomic_json(OUT/'result.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['fixed_comparisons']},ensure_ascii=False),flush=True)
    print('FIXED',json.dumps({k:{f:v for f,v in r.items() if f!='rows'} for k,r in results.items()},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
