"""Run every registered hit tool on every sampled clip and perception variant,
then classify each candidate with BST-0 and (with reviewed court) BST-CG-AP.
Writes predictions only; BFMD references are read by study_score.py.
"""
import argparse,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from professional_agents.hit_tools import inventory,run_method
from professional_agents.retrieval import mainland
from professional_agents.state import sha
STUDY=ROOT.parent/'03_数据/独立基准/BFMD/pipeline_study'
VARIANT=os.environ.get('BFMD_VARIANT','');SUFFIX=f'_{VARIANT}' if VARIANT else '';ROUTES=STUDY/f'routes{SUFFIX}'  # v2 = refined timing + filters
BALLS={'rv':'racketvision/trajectory.json','tn3':'tracknetv3/trajectory.json'}
SHARED_METHODS=['rules','fusion','gru_shared','shf','hitnet','rgb_resnet']

from benchmarks.bfmd.rally_clips import in_role
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def evidence(path):return {'artifact':str(path),'sha256':sha(path)}

def court_record(clip,folder):
    """Per-match reviewed camera corners (TL,TR,BL,BR), re-keyed to this clip's source hash."""
    review=STUDY/'court_review.json'
    corners=read(review)['matches'].get(clip['youtube_id'],{}) if review.exists() else {}
    if corners.get('review_status')!='confirmed':return None
    path=folder/'court_calibration.json'
    save(path,{'record':{'source_sha256':clip['sha256'],'review_status':'confirmed','court_corners':corners['corners_TL_TR_BL_BR'],'basis':corners['basis']}})
    return path

def calibration(corners):
    matrix,_=cv2.findHomography(np.asarray(corners,float),np.asarray([[0,0],[1,0],[0,1],[1,1]],float),method=0)
    return {'matrix':matrix}

def classify(dense,ball,events,fps,corners):
    from specialist_bst import SpecialistBST
    models={'bst0':SpecialistBST(dense,ball,None)}
    if corners is not None:models['bst_cg_ap']=SpecialistBST(dense,ball,calibration(corners))
    n=len(dense['frames']);out=[]
    for e in events:
        i=e['frame_index'];w=e.get('raw_frame_index',i)  # BST window stays on the detector's frame; reported time uses refined frame
        lo,hi=max(0,w-round(fps)),min(n-1,w+round(fps))
        row={'frame_index':i,'time':dense['frames'][i]['source_time'],'player':e.get('player'),'hit_score':e.get('hit_model_score'),'labels':{}}
        for name,model in models.items():
            values,_=model.inputs({**e,'start_frame':lo,'end_frame':hi,'peak_time':dense['frames'][w]['source_time']});pred=model.predict(values)
            ranked=pred['bst_full'] if name=='bst_cg_ap' else pred['bst0']
            row['labels'][name]={'label':mainland(ranked[0]['label']),'raw':ranked[0]['label'],'score':ranked[0]['softmax_score']}
            if VARIANT>='v3':  # BFMD taxonomy: summed class probabilities, serves excluded
                from benchmarks.bfmd.bfmd_labels import bfmd_probs,bfmd_label
                probs=bfmd_probs(pred['class_labels'],pred['bst_full_all_scores'] if name=='bst_cg_ap' else pred['bst0_all_scores'])
                row['labels'][name].update(label=bfmd_label(probs),probs=probs)
        out.append(row)
    return out

def run(clip,allow_expensive,methods):
    pfolder=STUDY/'perception'/clip['clip_id'];dense_path=pfolder/'dense_pose.json'
    if not all((pfolder/f).exists() for f in ['dense_pose.json',*BALLS.values()]):return []  # incomplete perception
    court=court_record(clip,pfolder);rows=[]
    for ball_name,rel in BALLS.items():
        ball_path=pfolder/rel
        if not ball_path.exists():continue
        state=SimpleNamespace(source_file=clip['path'],source_sha256=clip['sha256'],artifacts={'pose2d':evidence(dense_path),'trajectory':evidence(ball_path),**({'court':{'artifact':str(court),'kind':'reviewed_camera_calibration'}} if court else {})})
        info=inventory(state,allow_expensive=allow_expensive);dense=read(dense_path);ball=read(ball_path);fps=dense['sample_fps']
        for method in methods:
            if method=='gru_original' and ball_name!='rv':continue  # own TrackNetV2/HRNet frontend; run once
            target=ROUTES/clip['clip_id']/f"{'author' if method=='gru_original' else ball_name}__{method}.json"
            if target.exists():rows.append(read(target));continue
            started=time.perf_counter();result=run_method(method,state,info,STUDY/'hit_runs',3600)
            seconds=time.perf_counter()-started
            events=result['events']
            if VARIANT and result['status']=='completed':
                from benchmarks.bfmd.refine import postprocess
                events=postprocess(events,ball)
                for e in events:e['hit_model_score']=e.get('hit_model_score',e.get('confidence'))
            hits=classify(dense,ball,events,fps,info['author_corners']) if result['status']=='completed' else []
            packet={'clip_id':clip['clip_id'],'role':clip['role'],'perception':'author' if method=='gru_original' else ball_name,'method':method,'status':result['status'],'reason':result.get('reason'),
                    'hit_seconds':seconds if not result.get('cache_hit') else None,'cache_hit':result.get('cache_hit'),'inputs':info['quality'],'eligibility':info['methods'][method],'hits':hits,'annotation_inputs':[]}
            save(target,packet);rows.append(packet)
            print('ROUTE',clip['clip_id'],packet['perception'],method,packet['status'],len(hits),round(seconds,1),flush=True)
    return rows

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--role',choices=['dev','sealed_test','ae_test'],required=True);a.add_argument('--methods',default=','.join(SHARED_METHODS));a.add_argument('--expensive',action='store_true');a.add_argument('--limit',type=int);a=a.parse_args()
    clips=[c for c in read(STUDY/'clips.json')['clips'] if in_role(c,a.role) and 'path' in c][:a.limit]
    for c in clips:run(c,a.expensive,a.methods.split(','))
