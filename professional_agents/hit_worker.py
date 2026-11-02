"""Isolated hit tools. Request accepts model/raw artifacts only, no labels."""
import argparse,time,subprocess
from pathlib import Path
import numpy as np
from .state import ROOT,read,save,sha

def check(request):
    d,b=read(request['dense']),read(request['trajectory'])
    if sha(request['video'])!=request['source_sha256']:raise ValueError('Raw video identity changed')
    if d['source_sha256']!=request['source_sha256'] or b['source_sha256']!=request['source_sha256'] or len(d['frames'])!=len(b['frames']):raise ValueError('Hit input mismatch')
    for i,(f,ball) in enumerate(zip(d['frames'],b['frames'])):
        if f['frame_index']!=i or ball['frame_index']!=i or abs(f['source_time']-ball.get('source_time',f['source_time']))>1e-6:raise ValueError('Native clock mismatch')
    return d,b

def shf(d,b,r):
    import torch
    from paper_reproductions.data import pack_shf,ground_midpoints
    from paper_reproductions.run import load_model
    from paper_reproductions.shuttlehitformer import center_windows,nms_candidates
    if r['shf_corners'] is None:return [],{'status':'unavailable','reason':'court geometry missing'}
    n=len(d['frames']);corners=np.asarray(r['shf_corners'],np.float32);court=np.concatenate([corners,ground_midpoints(corners)],0);top=np.zeros((n,17,2),np.float32);bottom=np.zeros_like(top);ball=np.zeros((n,2),np.float32);valid=np.zeros(n,bool)
    for i,(f,p) in enumerate(zip(d['frames'],b['frames'])):
        if all(f['players'].get(side) for side in ['near','far']) and p['raw_visible'] and p.get('raw_xy'):
            top[i]=[[k['x'],k['y']] for k in f['players']['far']['keypoints']];bottom[i]=[[k['x'],k['y']] for k in f['players']['near']['keypoints']];ball[i]=p['raw_xy'];valid[i]=True
    features=pack_shf(top,bottom,np.repeat(court[None],n,0),np.zeros((n,4,2)),ball,normalized=False,width=d['source_width'],height=d['source_height']);checkpoint=read(ROOT/'research_models/registry.json')['shuttlehitformer']['local_checkpoint'];model,_=load_model(checkpoint);model.eval();indices=[i for i in range(3,n-4) if valid[i-3:i+5].all()];hits=[]
    with torch.no_grad():
        for centers in torch.tensor(indices,dtype=torch.long).split(256):
            scores=model(center_windows(torch.from_numpy(features),centers))['logits'].softmax(-1)
            for i,p in zip(centers.tolist(),scores.tolist()):
                c=max(range(3),key=p.__getitem__)
                if c and p[c]>=model.config.threshold:hits.append({'frame_index':i,'class_id':c,'confidence':p[c]})
    return [{**h,'player':'far' if h['class_id']==1 else 'near'} for h in nms_candidates(hits,model.config.nms_frames)],{'model':'local corrected SHF','checkpoint_sha256':sha(checkpoint),'valid_windows':len(indices),'court_source':r['shf_court_basis'],'net':'missing-net local ablation','hitter_identity':'automatic unverified model class'}

def run(r):
    dense,ball=check(r);method=r['method'];folder=Path(r['workdir']);meta={};events=[]
    if method=='rules':
        from specialist_events import proposals
        events=proposals(dense,ball);meta={'model':'existing geometric shuttle/wrist/turn proposals','trained':False}
    elif method=='fusion':
        from benchmark_hit_models import production
        import hit_refinement
        # Professional-only routing must not apply a personal timing calibration.
        hit_refinement.timing_model=lambda:(None,None)
        dense={**dense,'id':dense.get('id',r['source_sha256'][:32]),'end':len(dense['frames'])/dense['sample_fps']}
        output=production(dense,ball,{'source_path':r['video']},folder);events,meta=output['events'],output['metadata'];meta['local_personal_timing_model_disabled']=True
    elif method=='hitnet':
        from benchmark_hits_hitnet import infer
        if r['author_corners'] is None:raise ValueError('Adapted-HitNet requires reviewed court geometry')
        output=infer(dense,ball,court_corners=r['author_corners']);events,meta=output['events'],output['metadata']
    elif method=='gru_shared':
        from benchmark_hits_gru import infer
        output=infer(dense,ball,postprocess='naive');events,meta=output['events'],output['metadata'];meta['input_condition']='adapted current perception; NOT the original filtered2 pipeline that reached80%'
    elif method=='gru_original':
        if not r['original_cached_ready']:
            if r['author_corners'] is None:raise ValueError('No reviewed camera calibration')
            court=folder/'court.json';save(court,{'court_corners':r['author_corners']})
            pose=folder/'author_pose';track=folder/'author_ball'
            if not (pose/'dense_pose.json').exists():subprocess.run([str(ROOT/'.venv-vision/bin/python'),str(ROOT/'reproduce_gru_author_pose.py'),'--source',r['video'],'--output',str(pose),'--court',str(court)],cwd=ROOT,check=True)
            if not (track/'trajectory.json').exists():subprocess.run([str(ROOT/'.venv-vision/bin/python'),str(ROOT/'reproduce_gru_author_tracknet.py'),'--source',r['video'],'--output',str(track)],cwd=ROOT,check=True)
            dense,ball=read(pose/'dense_pose.json'),read(track/'trajectory.json')
            if dense['source_sha256']!=r['source_sha256'] or ball['source_sha256']!=r['source_sha256'] or len(dense['frames'])!=len(ball['frames']):raise ValueError('Original frontend source/coverage mismatch')
        from run_gru_author_reproduction import filtered2_legacy
        output=filtered2_legacy(dense,ball,'naive');events,meta=output['events'],output['metadata'];meta['original_frontend']=True;meta['court_dependency']='reviewed camera calibration, not hit labels';meta['delegated_perception_tools']=['Faster R-CNN/HRNet','TrackNetV2']
    elif method=='shf':events,meta=shf(dense,ball,r)
    elif method=='rgb_resnet':
        from reproduce_gru_author_resnet import infer
        output=infer(dense,source_path=r['video'],output_folder=folder/'rgb',postprocess='naive');events,meta=output['events'],output['metadata'];meta['input_condition']='author RGB classifier with current supplied boxes; explicitly adapted frontend'
    else:raise ValueError('Unregistered hit detector')
    n=len(dense['frames']);fps=dense['sample_fps'];converted=[]
    for e in events:
        i=int(e['frame_index'])
        if not 0<=i<n:raise ValueError('Predicted contact outside native frames')
        converted.append({**e,'frame_index':i,'peak_time':dense['frames'][i]['source_time'],'player':e.get('player',e.get('actor')),'hit_model_score':e.get('confidence',e.get('proposal_score'))})
    packet={'kind':'model_prediction','source_sha256':r['source_sha256'],'method':method,'status':meta.get('status','completed'),'events':converted,'metadata':meta,'annotation_inputs':[],'time_base':'original frame indices and source timestamps; no offset fitting','hit_time_note':'interval midpoint/candidate, not verified contact'};save(r['output'],packet)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);run(read(p.parse_args().request))
