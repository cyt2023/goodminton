"""Frozen inference first, sealed reference scoring afterwards. No search here."""
import argparse,subprocess,time,resource,sys
from pathlib import Path
from dataclasses import replace
from .state import ROOT,STORE,read,save,sha,digest
from .registry import Workflow
from .runtime import run

def evaluate_raw(video,match_id,offset):
    video=Path(video).resolve();folder=STORE/'heldout_raw';metadata=read(folder/'source_metadata.json')
    if metadata['id']!='ISFs9Gku1LA' or metadata['duration']!=4155 or not match_id.endswith('-SF'):raise ValueError('Verify official source identity/duration before this fixed pilot')
    configs={'FIXED':Workflow().packet(),**{mode:read(STORE/'experiments'/mode/'selected_workflow.json') for mode in ['AFLOW','EVOFLOW']}}
    protocol={'scope':'24-second frozen held-out professional raw-video pilot; not full-match performance','video_sha256':sha(video),'match_id':match_id,'offset':offset,'workflows':configs,'workflow_search_data_sha256':digest(read(STORE/'datasets.json')),'selection':'fixed360–384second source-time range; no optimizer access to BFMD references','reference_access_before_predictions':False,'code_sha256':sha(Path(__file__))}
    frozen=folder/'frozen_protocol.json'
    if frozen.exists() and digest(read(frozen))!=digest(protocol):raise ValueError('Frozen final protocol changed; do not retune on test outcomes')
    save(frozen,protocol)
    # Seal the new raw asset identity for existing training/calibration guards.
    manifest_path=ROOT.parent/'03_数据/独立基准/BFMD/manifest.json';manifest=read(manifest_path)
    asset={'path':str(video),'sha256':sha(video),'scope':'professional_agents frozen heldout raw clip'}
    if not any(r.get('sha256')==asset['sha256'] for r in manifest.get('heldout_video_files',[])):
        manifest.setdefault('heldout_video_files',[]).append(asset);save(manifest_path,manifest)
    raw=folder/'perception';raw.mkdir(exist_ok=True);started=time.perf_counter();cpu=time.process_time()
    if not (raw/'dense_pose.json').exists():
        with (raw/'pose.log').open('w') as log:subprocess.run([str(ROOT/'.venv-pose/bin/python'),str(ROOT/'benchmarks/bfmd/pose_worker.py'),'--video',str(video),'--out',str(raw)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=450)
    dense=read(raw/'dense_pose.json');dense['end']=len(dense['frames'])/dense['sample_fps'];save(raw/'dense_pose.json',dense)
    config={'source_file':str(video),'source_sha256':sha(video),'racketvision_folder':str(raw/'racketvision'),'specialist_folder':str(raw)}
    config_path=raw/'racketvision_input.json';save(config_path,config)
    with (raw/'racketvision.log').open('w') as log:subprocess.run([str(ROOT/'.venv-vision/bin/python'),str(ROOT/'racketvision_adapter.py'),'--config',str(config_path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=450)
    prep={'wall_seconds':time.perf_counter()-started,'scope':'one shared raw YOLOX/RTMPose + RacketVision preparation; subsequent agents reuse these model outputs','truth_inputs':[],'dense_model_report_seconds':dense.get('seconds'),'note':'CPU/GPU execution times not independently instrumented for this shared preparation'};save(raw/'cache_reuse_cost.json',prep) if (raw/'cost.json').exists() else save(raw/'cost.json',prep)
    prep={**read(raw/'cost.json'),'latest_preparation_wall_seconds':prep['wall_seconds']}
    record={'id':'BFMD_'+sha(video)[:16],'job_id':'J_'+sha(video)[:32],'video_id':metadata['id'],'dataset':'BFMD','domain':'professional','role':'sealed_test','benchmark_only':True,'match_key':'youtube:'+metadata['id'],'source_url':'https://www.youtube.com/watch?v='+metadata['id'],'source_file':str(video),'source_sha256':sha(video),'source_offset_seconds':offset,'dense_file':str(raw/'dense_pose.json'),'trajectory_file':str(raw/'racketvision/trajectory.json'),'classification_file':str(raw/'no_reference_classifications.json')};save(folder/'inference_record.json',record)
    # Complete predictions for ALL modes before any reference scoring.
    outputs=[]
    for mode,c in configs.items():
        w=Workflow.from_packet(c);r=run(record,w,allow_sealed_test=True)
        if r['status']!='completed':raise RuntimeError(r['error'])
        outputs.append(r)
    save(folder/'inference_runs.json',outputs)
    # Scorer is the ONLY step reading hit/type/rally ground truth.
    from benchmarks.bfmd.evaluate import evaluate
    match=next(m for m in read(manifest_path)['matches'] if m['match_id']==match_id)
    scored=[]
    for r in outputs:
        state=read(STORE/'runs'/r['run_id']/'state.json');packet={'kind':'model_inference','mode':'end_to_end','match_id':match_id,'label_space':'canonical_mainland','time_base':{'unit':'seconds','offset_seconds':offset},'evaluation_window':[0,len(dense['frames'])/dense['sample_fps']],'coordinates':{'space':'native_pixels','width':dense['source_width'],'height':dense['source_height']},'provenance':{'model':'existing YOLOX/RTMPose + RacketVision + geometric hits + pretrained BST-0; frozen workflow','input_video_sha256':sha(video),'annotation_inputs':[],'workflow':r['workflow'],'run_id':r['run_id']},'hits':[{'time':e['timestamp'],'confidence':e['confidence'],'label':e['stroke_label']} for e in state['events']]}
        save(folder/(r['mode']+'_predictions.json'),packet);result=evaluate(match,packet);save(folder/(r['mode']+'_evaluation.json'),result);scored.append({'mode':r['mode'],'run_id':r['run_id'],'cost':r['cost'],'evaluation':result})
    result={'scope':'24-second frozen held-out clip only; not full-match or corpus accuracy','source_alignment':'official video ID/duration matched; CFR reencoded24second segment; source offset360seconds; no independent content-based frame certification','shared_perception_cost':prep,'rows':scored,'no_reference_inputs_to_inference':True,'no_search_on_results':True,'frozen_protocol_sha256':sha(frozen)};save(folder/'comparison.json',result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--match-id',required=True);p.add_argument('--offset',type=float,required=True);a=p.parse_args();r=evaluate_raw(a.video,a.match_id,a.offset);print('HELDOUT_COMPLETE',len(r['rows']),flush=True)
