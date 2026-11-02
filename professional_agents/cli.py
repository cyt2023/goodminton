"""Run inside existing .venv-vision. No large downloads or model training."""
import argparse,json
from pathlib import Path
from .state import STORE,read,save
from .datasets import build_manifest,source_record
from .retrieval import build_index
from .registry import Workflow,registry

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    s.add_parser('prepare')
    r=s.add_parser('run');r.add_argument('--record',required=True);r.add_argument('--workflow');r.add_argument('--no-cache',action='store_true');r.add_argument('--qwen',action='store_true');r.add_argument('--racketvision',action='store_true');r.add_argument('--adaptive',action='store_true');r.add_argument('--hit-method',choices=['auto','auto_full','gru','gru_original','rules','shf','rgb_hit','fusion','hitnet','swingnet'])
    opt=s.add_parser('search');opt.add_argument('--mode',choices=['AFLOW','EVOFLOW'],required=True);opt.add_argument('--budget',type=int,default=24);opt.add_argument('--seed',type=int,default=20261030)
    s.add_parser('compare')
    a=p.parse_args()
    if a.command=='prepare':
        d=build_manifest();i=build_index(d['retrieval_gallery']);save(STORE/'registry.json',registry());print(json.dumps({'professional_recordings':len(d['records']),'gallery_clips':len(i['rows']),'development_matches':sum(v=='development' for v in d['feature_development']['match_roles'].values()),'validation_matches':sum(v=='validation' for v in d['feature_development']['match_roles'].values())}));return
    if a.command=='search':
        from .search import search
        value=search(a.mode,a.budget,a.seed);print(json.dumps({'mode':a.mode,'evaluation_calls':value['evaluation_calls'],'validation':value['validation']['metrics'],'implementation':value['implementation']}));return
    if a.command=='compare':
        from .search import compare
        value=compare();print(json.dumps({'report':str(STORE/'experiments/comparison.md'),'modes':[r['mode'] for r in value['rows']]}));return
    if a.command=='run':
        from .runtime import run
        if a.workflow:w=Workflow.from_packet(read(a.workflow))
        else:
            hit='stroke.'+a.hit_method if a.hit_method and a.hit_method!='rules' else 'stroke.bst'
            tools=['perception.racketvision' if a.racketvision else 'perception.cached',hit]
            if a.adaptive:tools+=['perception.verify','stroke.window']
            tools+=['technical.evidence']
            if a.qwen:tools+=['technical.qwen']
            tools+=['expert.retrieve'];w=Workflow(mode='AGENT' if a.hit_method else 'FIXED',tools=tuple(tools))
        r=run(source_record(a.record),w,use_cache=not a.no_cache);print(json.dumps({k:v for k,v in r.items() if k not in ['trace','provenance']},ensure_ascii=False));raise SystemExit(0 if r['status']=='completed' else 1)
if __name__=='__main__':main()
