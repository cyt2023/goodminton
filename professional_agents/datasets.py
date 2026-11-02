"""Match-level professional identities and sealed benchmark policy."""
import csv,json,re
from pathlib import Path
from .state import ROOT,STORE,read,save,sha,digest
from benchmarks.isolation import assert_training_allowed
DATA=ROOT.parent/'03_数据'

def youtube_id(value):
    m=re.search(r'(?:v=|youtu.be/)([A-Za-z0-9_-]{11})',value or '');return m.group(1) if m else None

def check_development(record):
    if record.get('domain')!='professional':raise ValueError('Amateur development is out of scope')
    if record.get('role') not in ['development','validation','retrieval_gallery','smoke']:raise ValueError('Sealed/final/test data cannot enter search or gallery')
    assert_training_allowed(record)
    return record

def build_manifest():
    old=read(DATA/'职业录像/东京奥运会2021男单决赛/benchmark.json');records=[]
    for row in old['jobs']:
        job=row['job_id'];p=DATA/'专项分析'/job/'config.json';config=read(p)
        records.append({'id':job,'job_id':job,'dataset':'public_Olympics_video','domain':'professional','role':'smoke','match_key':'youtube:'+youtube_id(old['source_url']), 'youtube_id':youtube_id(old['source_url']),'source_url':old['source_url'],'video_id':row['video_id'],'source_file':str(DATA/row['clip_file']),'source_sha256':row['clip_sha256'],'source_offset_seconds':row['source_offset_seconds'],'dense_file':str(DATA/'专项分析'/job/'dense_pose.json'),'trajectory_file':str(DATA/'专项分析'/job/'trajectory.json'),'classification_file':str(DATA/'技术动作'/job/'片段/classified_events.json'),'config_file':str(p),'media_url':'/media/'+row['video_id'],'truth_available':False,'prior_exposure':'existing diagnostic clips; not an untouched test'})
    matches=list(csv.DictReader((DATA/'职业示范索引/ShuttleSet_BST/match.csv').open()))
    identities={r['id']:{'match_key':'youtube:'+youtube_id(r['url']),'youtube_id':youtube_id(r['url']),'source_url':r['url'],'name':r['video']} for r in matches}
    # All feature samples are previously cached author train/validation samples.
    # Their historical exposure prevents treating this partition as a final test.
    feature=DATA/'公开训练数据/BST_ShuttleSet/开发样本/manifest.json';fm=read(feature)
    mids=sorted({s['sample_id'].split('_')[0] for s in fm['samples']},key=int)
    split={m:('development' if int(m)<=28 else 'validation') for m in mids}
    for m in mids:check_development({'domain':'professional','role':split[m],**identities[m]})
    for row in records:check_development(row)
    sealed=read(DATA/'独立基准/BFMD/manifest.json')
    sealedids={m.get('youtube_id') for m in sealed['matches'] if m.get('youtube_id')}
    overlap=[m for m in mids if identities[m]['youtube_id'] in sealedids]
    if overlap:raise ValueError('ShuttleSet/BFMD match overlap: '+str(overlap))
    manifest={'schema_version':1,'records':records,'feature_development':{'dataset':'ShuttleSet_BST_author_preprocessed','source_manifest':str(feature),'features':str(feature.with_name('features.npz')),'source_manifest_sha256':sha(feature),'features_sha256':sha(feature.with_name('features.npz')),'match_roles':split,'identities':{m:identities[m] for m in mids},'scope':'conditional classification on public preprocessed features; NOT raw-video end-to-end','prior_exposure':'previously used in project development; validation is internal, not untouched final test'},'retrieval_gallery':{'dataset':'ShuttleSet_BST','match_id':'13','match_key':identities['13']['match_key'],'role':'retrieval_gallery','domain':'professional','csv':str(DATA/'职业示范索引/ShuttleSet_BST/match13_set1.csv'),'source_url':identities['13']['source_url']},'overlap_audit':{'ShuttleSet_BFMD_video_id_intersection':overlap,'Tokyo_BFMD':any(r['youtube_id'] in sealedids for r in records),'unresolved_cross_dataset_overlap':['Fine-Badminton and RacketVision training identities not fully locally resolved; not admitted to search','legacy generic domain_pro match IDs not resolved; not admitted to search'],'upstream_pretraining_overlap':'BST author features may overlap its training; this is NOT a model-generalization benchmark'},'final_test':{'dataset':'BFMD','role':'sealed_final_test','status':'not_run_in_this_workflow_search','diagnostic_exception':'existing16-second diagnostic excluded from final performance claims','search_reference_access':False},'other_datasets':{'Fine-Badminton':'not used; local eligible match manifest unavailable','RacketVision':'pretrained perception weights used; dataset not included in workflow search'}}
    groups={}
    for mid,info in manifest['feature_development']['identities'].items():groups.setdefault(info['match_key'],set()).add(split[mid])
    if any(len(v)>1 for v in groups.values()):raise ValueError('Same video crosses whole-match split')
    normalize=lambda value:''.join(re.findall(r'[a-z0-9]+',value.lower()))
    names={normalize(m['match_id']) for m in sealed['matches']}
    name_overlap=[mid for mid,info in manifest['feature_development']['identities'].items() if normalize(info['name']) in names]
    if name_overlap:raise ValueError('Named-match overlap with BFMD')
    save(STORE/'overlap_audit.json',{'exact_normalized_match_name_overlap':name_overlap,'same_video_crosses_development_validation':False,'video_id_overlap':overlap,'unresolved':manifest['overlap_audit']['unresolved_cross_dataset_overlap']})
    save(STORE/'datasets.json',manifest);return manifest

def source_record(identifier):
    manifest=read(STORE/'datasets.json')
    for r in manifest['records']:
        if r['id']==identifier:return check_development(r)
    raise ValueError('Only allowlisted professional recordings can run')
