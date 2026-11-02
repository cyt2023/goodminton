"""Bounded controller, content-addressed inference cache and measured trace."""
import resource,time,sys,uuid
from dataclasses import asdict,replace
from pathlib import Path
from .state import ROOT,STORE,SharedState,read,save,sha,digest
from .registry import TOOLS,PROVIDES,Workflow
from .agents import PerceptionAgent,StrokeUnderstandingAgent,TechnicalTacticalAgent,ExpertDemonstrationAgent
from .datasets import check_development
AGENTS={c.__name__:c for c in [PerceptionAgent,StrokeUnderstandingAgent,TechnicalTacticalAgent,ExpertDemonstrationAgent]}

def usage():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN)
    return {'cpu_seconds':a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime,'process_peak_rss_bytes':max(a.ru_maxrss,b.ru_maxrss)*(1 if sys.platform=='darwin' else 1024)}
def code_hash():
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/p for p in ['specialist_bst.py','specialist_events.py','research_report.py','research_quality.py','tactical_review.py','stroke_event_fusion.py','racketvision_adapter.py','racketvision_runtime.py','hit_gru.py','benchmark_hits_gru.py','run_gru_author_reproduction.py']]
    return digest({str(p.relative_to(ROOT)):sha(p) for p in paths})
def model_fingerprints():
    paths=[ROOT/'models/bst/bst_0_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt',ROOT/'research_models/registry.json',ROOT/'models/specialist_manifest.json',ROOT/'models/racketvision/manifest.json',ROOT/'models/rtmpose-m-body7-256x192.onnx',ROOT/'models/yolox-m-humanart.onnx']
    rv=ROOT/'models/racketvision/manifest.json'
    if rv.exists():paths += [rv.parent/name for name in read(rv).get('files',{}) if (rv.parent/name).is_file()]
    from .hit_tools import ASSETS
    paths+=list({p for items in ASSETS.values() for p in items})
    return {str(p):sha(p) for p in paths if p.exists()}
def cache_valid(value):
    try:
        state=SharedState.from_packet(value['state'])
        return all(Path(ref['artifact']).is_file() and sha(ref['artifact'])==ref['sha256'] for ref in state.artifacts.values())
    except (OSError,KeyError,TypeError,ValueError):return False

def run(record,workflow,*,use_cache=True,allow_sealed_test=False):
    workflow.validate()
    if allow_sealed_test:
        if record.get('role')!='sealed_test' or record.get('domain')!='professional' or record.get('dataset')!='BFMD' or not record.get('benchmark_only'):raise ValueError('Explicit sealed benchmark input required')
    else:check_development(record)
    if sha(record['source_file'])!=record['source_sha256']:raise ValueError('Source video changed')
    dense=read(record['dense_file']);state=SharedState(record['video_id'],record['match_key'],record['source_sha256'],record['source_file'],dense['sample_fps'],[f['source_time'] for f in dense['frames']],record['source_offset_seconds'],record['role'],record['dataset'])
    if dense['source_sha256']!=state.source_sha256:raise ValueError('Pose is from another source')
    identifier='W_'+uuid.uuid4().hex;folder=STORE/'runs'/identifier;folder.mkdir(parents=True)
    provenance={'code_sha256':code_hash(),'models':model_fingerprints(),'source_record':record,'workflow':workflow.packet(),'ground_truth_inputs':[],'allow_sealed_inference_only':allow_sealed_test,'kind':'model_prediction','input_perception_sha256':{k:sha(record[k]) for k in ['dense_file','trajectory_file','classification_file'] if Path(record[k]).exists()}}
    state.provenance.update(provenance);state.provenance['coordinate_system']='original decoded image pixels; source_offset_seconds links clip time to full video; no metric court transform without verified calibration';agents={k:v(record,folder) for k,v in AGENTS.items()};trace=[];started=time.perf_counter();status='completed';error=None
    save(folder/'status.json',{'run_id':identifier,'status':'running','record_id':record['id']})
    for index,name in enumerate(workflow.tools):
        if len(trace)>=workflow.max_calls or time.perf_counter()-started>=workflow.timeout_seconds:status='budget_exhausted';break
        spec=TOOLS[name];before=usage();clock=time.perf_counter();input_packet=state.packet();cache_input={**input_packet,'completed_tools':[],'provenance':{k:v for k,v in input_packet['provenance'].items() if k!='workflow'}}
        params={k:v for k,v in workflow.packet().items() if k not in ['mode','tools','timeout_seconds','max_calls']}
        key=digest({'tool':name,'state':cache_input,'parameters':params,'code':provenance['code_sha256'],'models':provenance['models'],'gallery_sha256':sha(STORE/'retrieval/index.json') if name=='expert.retrieve' else None});path=STORE/'cache'/(key+'.json')
        cached=False;outcome='failed'
        try:
            if use_cache and path.exists() and cache_valid(read(path)):
                value=read(path);restored=SharedState.from_packet(value['state']);restored.completed_tools=state.completed_tools;restored.provenance['workflow']=workflow.packet();state=restored;outcome=value['outcome'];cached=True
            else:
                remaining=max(.1,workflow.timeout_seconds-(time.perf_counter()-started))
                outcome=agents[spec.agent].execute(name,state,replace(workflow,timeout_seconds=remaining));state.validate();save(path,{'state':state.packet(),'outcome':outcome,'cache_contract':{'input_digest':key,'cold_inference_cost':'see originating trace, not zero','origin_run_id':identifier}})
            state.completed_tools.append(name)
        except Exception as e:error=type(e).__name__+': '+str(e);status='failed'
        after=usage();trace.append({'step':index+1,'agent':spec.agent,'tool':name,'status':outcome,'cache_hit':cached,'cache_key':key,'seconds':time.perf_counter()-clock,'cpu_seconds':after['cpu_seconds']-before['cpu_seconds'],'process_peak_rss_bytes':after['process_peak_rss_bytes'],'gpu_time_seconds':None,'gpu_time_reason':'No GPU timing instrumentation; MPS wall time is not GPU time','retries':0,'state_events':len(state.events),'error':error,'agent_memory':agents[spec.agent].memory})
        save(folder/'trace.json',trace)
        if error:break
    save(folder/'state.json',state.packet());result={'run_id':identifier,'status':status,'error':error,'record_id':record['id'],'match_key':record['match_key'],'mode':workflow.mode,'workflow':workflow.packet(),'trace':trace,'cost':{'wall_seconds':time.perf_counter()-started,'tool_calls':len(trace),'cache_hits':sum(t['cache_hit'] for t in trace),'retries':0,'cpu_seconds':sum(t['cpu_seconds'] for t in trace),'process_peak_rss_bytes':max((t['process_peak_rss_bytes'] for t in trace),default=usage()['process_peak_rss_bytes']),'hit_tool_calls':len(state.observations.get('hit_routing',{}).get('executions',[])),'gpu_seconds':None,'memory_scope':'process lifetime high-water RSS incl. maximum child; not isolated per-tool delta','original_cached_perception_cost':'unknown; not charged as zero model cost'},'summary':{'events':len(state.events),'typed_events':sum(e.stroke_label is not None for e in state.events),'retrieval_queries':len(state.retrieval),'retrieval_matches':sum(len(q['matches']) for q in state.retrieval),'rallies':len(state.rallies),'quality':state.quality,'missing':state.missing},'provenance':provenance,'evaluation':{'status':'unavailable','reason':'Smoke recording has no independent hit/type truth; do not report detection accuracy'},'created_at':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()};save(folder/'result.json',result);save(folder/'status.json',{'run_id':identifier,'status':status,'record_id':record['id']});return result
