"""Shared perception for each sampled rally clip: YOLOX/RTMPose pose,
RacketVision ball and TrackNetV3 ball. Raw video only; no BFMD references.
"""
import argparse,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT.parent/'03_数据/独立基准/BFMD/pipeline_study'

def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def step(name,target,command,folder,timeout):
    if Path(target).exists():return {'status':'cached'}
    started=time.perf_counter()
    with (folder/f'{name}.log').open('w') as log:
        p=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    return {'status':'completed' if p.returncode==0 and Path(target).exists() else 'failed','seconds':time.perf_counter()-started}

def run(clip):
    folder=STUDY/'perception'/clip['clip_id'];folder.mkdir(parents=True,exist_ok=True);cost=read(folder/'cost.json') if (folder/'cost.json').exists() else {}
    cost['pose']=cost.get('pose') if (folder/'dense_pose.json').exists() else step('pose',folder/'dense_pose.json',[str(ROOT/'.venv-pose/bin/python'),str(ROOT/'benchmarks/bfmd/pose_worker.py'),'--video',clip['path'],'--out',str(folder)],folder,3600)
    dense=read(folder/'dense_pose.json') if (folder/'dense_pose.json').exists() else None
    if dense and 'end' not in dense:dense['end']=len(dense['frames'])/dense['sample_fps'];(folder/'dense_pose.json').write_text(json.dumps(dense,allow_nan=False))  # rule proposals need clip end
    config={'source_file':clip['path'],'source_sha256':clip['sha256'],'racketvision_folder':str(folder/'racketvision'),'specialist_folder':str(folder/'tracknetv3')}
    save(folder/'perception_input.json',config)
    cost['racketvision']=cost.get('racketvision') if (folder/'racketvision/trajectory.json').exists() else step('racketvision',folder/'racketvision/trajectory.json',[str(ROOT/'.venv-vision/bin/python'),str(ROOT/'racketvision_adapter.py'),'--config',str(folder/'perception_input.json')],folder,3600)
    cost['tracknetv3']=cost.get('tracknetv3') if (folder/'tracknetv3/trajectory.json').exists() else step('tracknetv3',folder/'tracknetv3/trajectory.json',[str(ROOT/'.venv-vision/bin/python'),str(ROOT/'specialist_tracknet.py'),'--config',str(folder/'perception_input.json')],folder,3600)
    save(folder/'cost.json',cost);return cost

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--role',choices=['dev','sealed_test'],required=True);a.add_argument('--limit',type=int);a=a.parse_args()
    clips=[c for c in read(STUDY/'clips.json')['clips'] if c['role']==a.role and 'path' in c][:a.limit]
    for i,c in enumerate(clips):
        r=run(c);print('PERCEPTION',i+1,len(clips),c['clip_id'],json.dumps(r),flush=True)
