"""Score every stored route prediction against BFMD manual contacts (±0.1 s).
The only module in the study that reads BFMD hit/stroke references.
Sealed-test clips are scored only after the selection policy is frozen.
"""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from benchmarks.bfmd.evaluate import evaluate
STUDY=ROOT.parent/'03_数据/独立基准/BFMD/pipeline_study'
VARIANT=os.environ.get('BFMD_VARIANT','');SUFFIX=f'_{VARIANT}' if VARIANT else '';ROUTES=STUDY/f'routes{SUFFIX}'  # v2 = refined timing + filters
CLASSIFIERS=['bst0','bst_cg_ap','agent','bst0_author_window']  # 'agent' only on agent routes

from benchmarks.bfmd.rally_clips import in_role
def read(p):return json.loads(Path(p).read_text())

SERVES={'serve','serve_long'}
def is_serve(h):return h.get('label') in SERVES or any('serve' in str(x) for x in h.get('raw_labels',[]))

def without_serves(match):
    """Serves are not scored: drop them from the reference."""
    return {**match,'hits':[h for h in match['hits'] if not is_serve(h)]},[h['time'] for h in match['hits'] if is_serve(h)]

def score_route(route,clip,match):
    rows=[];match,serves=without_serves(match)
    # predictions at a serve are neither credited nor penalised
    near_serve=lambda t:any(abs(t+clip['offset_seconds']-s)<=.1+1e-6 for s in serves)
    route={**route,'hits':[h for h in route['hits'] if not near_serve(h['time'])]}
    for cls in CLASSIFIERS:
        if cls not in (route['hits'][0]['labels'] if route['hits'] else {'bst0':0,'bst_cg_ap':0}):continue
        packet={'kind':'model_inference','mode':'end_to_end','match_id':clip['match_id'],'label_space':'canonical_mainland','time_base':{'unit':'seconds','offset_seconds':clip['offset_seconds']},
                'evaluation_window':[0,clip['duration_seconds']],'provenance':{'model':f"{route['perception']}+{route['method']}+{cls}",'input_video_sha256':clip['sha256'],'annotation_inputs':[]},
                'hits':[{'time':h['time'],'confidence':h.get('hit_score'),'label':h['labels'][cls]['label']} for h in route['hits']]}
        m=evaluate(match,packet)['metrics'];hit=m['hit_detection'];typ=m['stroke_types']
        rows.append({'clip_id':clip['clip_id'],'youtube_id':clip['youtube_id'],'role':clip['role'],'pipeline':f"{route['perception']}__{route['method']}__{cls}",'perception':route['perception'],'method':route['method'],'classifier':cls,'status':route['status'],
                     'tp':hit['tp'],'fp':hit['fp'],'fn':hit['fn'],'hit_f1':hit['f1'],'type_tp':typ.get('tp'),'type_fp':typ.get('fp'),'type_fn':typ.get('fn'),'joint_f1':typ.get('f1'),
                     'mae_s':m['timestamp_error']['mae_seconds'],'hit_seconds':route.get('hit_seconds'),'inputs':route['inputs']})
    return rows

def run(role):
    if role=='new_test' and not (STUDY/f'frozen_verify{SUFFIX}.json').exists():raise ValueError('Freeze the verification agent before scoring new_test')
    if role in ('sealed_test','ae_test') and not (STUDY/f'frozen_policy{SUFFIX}.json').exists():raise ValueError('Freeze the selection policy before scoring sealed test clips')
    manifest=read(ROOT.parent/'03_数据/独立基准/BFMD/manifest.json');matches={m['match_id']:m for m in manifest['matches']};rows=[]
    for clip in read(STUDY/'clips.json')['clips']:
        folder=ROUTES/clip['clip_id']
        if not in_role(clip,role) or not folder.exists():continue
        for f in sorted(folder.glob('*.json')):rows+=score_route(read(f),clip,matches[clip['match_id']])
    out=STUDY/f'scores_{role}{SUFFIX}.json';out.write_text(json.dumps(rows,ensure_ascii=False,indent=1));return rows

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--role',choices=['dev','sealed_test','ae_test','new_test'],required=True);a=a.parse_args()
    rows=run(a.role);print('SCORED',len(rows),flush=True)
