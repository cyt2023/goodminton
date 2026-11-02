"""Stdlib UI integration; existing auth/Host policy remains in server.py."""
import fcntl,re,subprocess,threading,uuid
from pathlib import Path
from .state import ROOT,STORE,read,save
from .datasets import source_record
from .registry import Workflow

def listing():
    manifest=read(STORE/'datasets.json') if (STORE/'datasets.json').exists() else {'records':[]}
    results=[]
    for p in sorted((STORE/'runs').glob('W_*/result.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:40]:
        r=read(p);results.append({k:r[k] for k in ['run_id','status','mode','record_id','cost','summary','created_at']})
    comparison=read(STORE/'experiments/comparison.json') if (STORE/'experiments/comparison.json').exists() else None
    tasks=[read(p) for p in (STORE/'tasks').glob('*.json') if re.fullmatch(r'[a-f0-9]{32}\.json',p.name)]
    return {'schema_version':1,'records':[{'id':r['id'],'title':'东京奥运会决赛 · '+str(r['source_offset_seconds'])+'秒起','video_id':r['video_id'],'media_url':r['media_url'],'source_url':r['source_url']} for r in manifest['records']],'runs':results,'tasks':tasks,'comparison':comparison,'heldout_comparison':read(STORE/'heldout_raw/comparison.json') if (STORE/'heldout_raw/comparison.json').exists() else None,'search_status':'AFLOW/EVOFLOW are constrained research reproductions; final benchmark unrun'}

def details(identifier):
    if not re.fullmatch(r'W_[a-f0-9]{32}',identifier):raise ValueError('Invalid workflow run')
    f=STORE/'runs'/identifier;result=read(f/'result.json');state=read(f/'state.json')
    return {'result':result,'state':state}

def start(packet):
    record=source_record(packet.get('record_id'));mode=packet.get('mode','FIXED')
    if mode not in ['FIXED','AFLOW','EVOFLOW','AGENT']:raise ValueError('Invalid workflow mode')
    path=STORE/'experiments'/mode/'selected_workflow.json'
    w=Workflow(mode='AGENT',tools=('perception.cached','stroke.auto_full' if packet.get('allow_expensive_hits') is True else 'stroke.auto','technical.evidence','expert.retrieve')) if mode=='AGENT' else Workflow() if mode=='FIXED' else Workflow.from_packet(read(path))
    identifier=uuid.uuid4().hex;target=STORE/'tasks'/(identifier+'.json');state={'task_id':identifier,'record_id':record['id'],'mode':mode,'status':'queued'}
    active=[read(p) for p in (STORE/'tasks').glob('*.json') if read(p).get('status') in ['queued','running']]
    if active:raise ValueError('已有职业工作流运行，请稍后重试')
    save(target,state)
    def work():
        lockpath=ROOT.parent/'03_数据/技术动作/.execution.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
        try:
            with lockpath.open('a+b') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX);state['status']='running';save(target,state)
                config=target.with_suffix('.workflow.json');save(config,w.packet());log=target.with_suffix('.log')
                with log.open('w') as out:done=subprocess.run([str(ROOT/'.venv-vision/bin/python'),'-m','professional_agents.cli','run','--record',record['id'],'--workflow',str(config)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=650)
                state['status']='completed' if done.returncode==0 else 'failed'
                # Child prints one final summary; model warnings may precede it.
                import json
                for line in reversed(log.read_text().splitlines()):
                    if line.startswith('{'):
                        result=json.loads(line);state['run_id']=result.get('run_id');state['error']=result.get('error');break
        except Exception as e:state.update(status='failed',error=str(e))
        save(target,state)
    threading.Thread(target=work,name='professional-workflow',daemon=True).start();return state
