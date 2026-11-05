"""Per-hit verification agent.

Starts from the primary pipeline's candidates, adds consensus candidates that
the primary missed, and keeps/drops each one with a logistic model over
label-free evidence (detector agreement, shuttle turn/speed, BST confidence,
pose coverage, rhythm). The stroke label is the BFMD-taxonomy argmax of the
averaged BST-0 / BST-CG-AP probabilities when that helps on training folds.

train/apply : model and threshold are fitted on dev matches only;
evaluate    : leave-one-match-out over dev, writes routes .../agent__verify.json
"""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from benchmarks.bfmd.refine import track,turn,speed
from benchmarks.bfmd.bfmd_labels import bfmd_label
from benchmarks.bfmd.rally_clips import in_role
STUDY=ROOT.parent/'03_数据/独立基准/BFMD/pipeline_study'
VARIANT=os.environ.get('BFMD_VARIANT','v3');ROUTES=STUDY/f'routes_{VARIANT}'
PRIMARY='tn3__gru_shared';WIN=3;ADD_SUPPORT=3
USE_ALT=os.environ.get('VERIFY_ALT','1')=='1'  # author-window BST re-check tool

def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=1,allow_nan=False)+'\n')

def candidates(clip):
    """Label-free candidates with evidence features."""
    folder=ROUTES/clip['clip_id'];routes={f.stem:read(f) for f in folder.glob('*.json') if '__' in f.stem and not f.stem.startswith('agent')}
    routes={k:r for k,r in routes.items() if r['status']=='completed'}
    if PRIMARY not in routes:return []
    others={k:[round(h['time']*30) for h in r['hits']] for k,r in routes.items() if k!=PRIMARY}
    pfolder=STUDY/'perception'/clip['clip_id'];xy=track(read(pfolder/'tracknetv3/trajectory.json'));H=court_h(clip)
    dense=read(pfolder/'dense_pose.json')['frames'];n=len(dense)
    cands=[{'frame':round(h['time']*30),'hit':h,'primary':1.} for h in routes[PRIMARY]['hits']]
    pool=sorted((f,k,h) for k,r in routes.items() if k!=PRIMARY for h in r['hits'] for f in [round(h['time']*30)])
    used=set()
    for f,k,h in pool:  # consensus candidates the primary missed
        if any(abs(f-c['frame'])<=WIN for c in cands) or (f,k) in used:continue
        members=[(g,kk,hh) for g,kk,hh in pool if abs(g-f)<=WIN]
        if len({kk for _,kk,_ in members})>=ADD_SUPPORT:
            used.update((g,kk) for g,kk,_ in members)
            best=max(members,key=lambda m:m[1].startswith('tn3'))[2]
            cands.append({'frame':int(np.median([g for g,_,_ in members])),'hit':best,'primary':0.})
    cands.sort(key=lambda c:c['frame']);frames=[c['frame'] for c in cands]
    for j,c in enumerate(cands):
        f=c['frame'];lab=c['hit']['labels'];p0=lab.get('bst0',{}).get('probs',{});p1=lab.get('bst_cg_ap',{}).get('probs',{})
        top=sorted(p0.values(),reverse=True)+[0,0]
        sup=sum(any(abs(f-g)<=WIN for g in gs) for gs in others.values())
        pose=np.mean([bool(dense[i]['players']['near'] and dense[i]['players']['far']) for i in range(max(0,f-5),min(n,f+6))]) if n else 0
        gaps=[abs(f-g) for k,g in enumerate(frames) if k!=j]
        c['x']=[c['primary'],sup/max(1,len(others)),max(turn(xy,t) for t in range(max(0,f-2),min(len(xy),f+3))),min(speed(xy,f),30)/30,
                top[0],top[0]-top[1],float(bfmd_label(p0)==bfmd_label(p1)) if p1 else .5,pose,min(min(gaps,default=90),90)/90,
                float(c['hit'].get('hit_score') or 0)]
        c['time']=f/30;c['probs']=(p0,p1);c['traj']=flight(xy,dense,f,H)
    return cands

TYPES=['clear','lift','drop','smash','drive','push','net_kill','net','defense']
def court_h(clip):
    import cv2
    rec=read(STUDY/'court_review.json')['matches'][clip['youtube_id']]['corners_TL_TR_BL_BR']
    return cv2.findHomography(np.asarray(rec,float),np.asarray([[0,0],[1,0],[0,1],[1,1]],float))[0]
def to_court(H,p):
    v=H@np.array([p[0],p[1],1.]);return v[:2]/v[2]
def flight(xy,dense,f,H):
    """Shuttle flight after contact and hitter position, in court units (far baseline y=0)."""
    n=len(xy);ok=lambda t:0<=t<n and not np.isnan(xy[t]).any()
    base=next((t for t in range(f,min(n,f+3)) if ok(t)),None);out=[]
    for k in (5,10,15):
        t=next((t for t in range(min(n-1,f+k),f,-1) if ok(t)),None)
        if base is None or t is None or t==base:out+=[0.,0.,0.];continue
        a,b=to_court(H,xy[base]),to_court(H,xy[t]);out+=[(b[0]-a[0]),(b[1]-a[1]),1.]
    seg=[xy[t][1] for t in range(f,min(n,f+21)) if ok(t)]
    out+=[(min(seg)-seg[0])/400 if len(seg)>1 else 0.,(seg[-1]-seg[0])/400 if len(seg)>1 else 0.]  # image rise (negative = up)
    out+=[min(speed(xy,t) for t in [f+3])/30 if f+3<n else 0.]
    pos=to_court(H,xy[base]) if base is not None else np.array([.5,.5]);out+=[pos[0],pos[1]]
    fr=dense[min(f,len(dense)-1)]['players'];feet={s:to_court(H,[(p['bbox'][0]+p['bbox'][2])/2,p['bbox'][3]]) for s,p in fr.items() if p}
    if base is not None and feet:
        side=min(feet,key=lambda s:np.linalg.norm(feet[s]-pos));out+=[float(side=='near'),abs(feet[side][1]-.5)]
    else:out+=[.5,.25]
    return out

def type_vec(probs):return [probs.get(t,0.) for t in TYPES]

_BST={}
def author_window_probs(clip,seq):
    """Tool: re-run BST-0 / BST-CG-AP on the BST author's clip rule using the agent's kept
    neighbours: previous hit .. next hit + 0.25 s, capped at 1.5 s each side (0.5 s if no neighbour)."""
    from specialist_bst import SpecialistBST
    from benchmarks.bfmd.study_hits import calibration
    from benchmarks.bfmd.bfmd_labels import bfmd_probs
    key=clip['clip_id']
    if key not in _BST:
        pf=STUDY/'perception'/key;dense=read(pf/'dense_pose.json');ball=read(pf/'tracknetv3/trajectory.json')
        corners=read(STUDY/'court_review.json')['matches'][clip['youtube_id']]['corners_TL_TR_BL_BR']
        _BST[key]=(dense,SpecialistBST(dense,ball,None),SpecialistBST(dense,ball,calibration(corners)),{})
    dense,m0,m1,cache=_BST[key];n=len(dense['frames']);out=[]
    frames=[round(c['time']*30) for c in seq]
    for j,f in enumerate(frames):
        prev=frames[j-1] if j and f-frames[j-1]<=75 else None;nxt=frames[j+1] if j+1<len(frames) and frames[j+1]-f<=75 else None
        lo=max(prev if prev is not None else f-15,f-45,0);hi=min((nxt+7) if nxt is not None else f+15,f+45+7,n-1)
        if (lo,hi) not in cache:
            e={'frame_index':f,'start_frame':lo,'end_frame':hi,'peak_time':dense['frames'][f]['source_time']}
            v0,_=m0.inputs(e);v1,_=m1.inputs(e);a,b=m0.predict(v0),m1.predict(v1)
            cache[(lo,hi)]=(bfmd_probs(a['class_labels'],a['bst0_all_scores']),bfmd_probs(b['class_labels'],b['bst_full_all_scores']))
        out.append(cache[(lo,hi)])
    return out
def type_x(kept,j,alt=None):
    """BST probs (both), flight, and neighbouring candidates' BST probs as rally context."""
    c=kept[j];p0,p1=c['probs'];x=type_vec(p0)+(type_vec(p1) if p1 else type_vec(p0))+c['traj']
    if alt is not None:x+=type_vec(alt[j][0])+type_vec(alt[j][1])  # author-window re-check
    for k in (j-1,j+1):
        nb=kept[k] if 0<=k<len(kept) and abs(kept[k]['time']-c['time'])<=2.5 else None
        x+=type_vec(nb['probs'][0]) if nb else [0.]*len(TYPES)
        x+=[min(abs(nb['time']-c['time']),2.5)/2.5 if nb else 1.]
    return x

class Softmax:
    def fit(self,X,y,l2=3.,steps=3000,lr=.5):
        X=np.asarray(X,float);self.mu,self.sd=X.mean(0),X.std(0)+1e-6;Z=np.c_[np.ones(len(X)),(X-self.mu)/self.sd]
        Y=np.eye(len(TYPES))[[TYPES.index(t) for t in y]];W=np.zeros((Z.shape[1],len(TYPES)))
        for _ in range(steps):
            P=np.exp(Z@W-(Z@W).max(1,keepdims=True));P/=P.sum(1,keepdims=True)
            G=Z.T@(P-Y)/len(Z)+l2/len(Z)*np.r_[np.zeros((1,len(TYPES))),W[1:]];W-=lr*G
        self.W=W;return self
    def predict(self,X):
        Z=np.c_[np.ones(len(X)),(np.asarray(X,float)-self.mu)/self.sd];return [TYPES[i] for i in (Z@self.W).argmax(1)]

def truth_types(clip,cands):
    """Dev-only: reference stroke type for each candidate matched to a typed non-serve contact."""
    m=next(x for x in read(ROOT.parent/'03_数据/独立基准/BFMD/manifest.json')['matches'] if x['match_id']==clip['match_id'])
    lo=clip['offset_seconds'];hi=lo+clip['duration_seconds'];truth=[(h['time']-lo,h['label']) for h in m['hits'] if lo<=h['time']<hi and h['label'] not in ('serve','serve_long')]
    out=[];used=set()
    for c in cands:
        best=min(((abs(c['time']-t),i) for i,(t,_) in enumerate(truth) if i not in used and abs(c['time']-t)<=.1+1e-6),default=None)
        if best:used.add(best[1]);out.append(truth[best[1]][1] if truth[best[1]][1] in TYPES else None)
        else:out.append(None)
    return out

def truth_labels(clip,cands):
    """Dev-only: which candidates match a non-serve reference contact (±0.1 s); None near serves."""
    m=next(x for x in read(ROOT.parent/'03_数据/独立基准/BFMD/manifest.json')['matches'] if x['match_id']==clip['match_id'])
    lo=clip['offset_seconds'];hi=lo+clip['duration_seconds'];hits=[h for h in m['hits'] if lo<=h['time']<hi]
    serves=[h['time']-lo for h in hits if h['label'] in ('serve','serve_long')];truth=[h['time']-lo for h in hits if h['label'] not in ('serve','serve_long')]
    y=[];used=set()
    for c in cands:
        if any(abs(c['time']-s)<=.1+1e-6 for s in serves):y.append(None);continue
        best=min(((abs(c['time']-t),i) for i,t in enumerate(truth) if i not in used and abs(c['time']-t)<=.1+1e-6),default=None)
        if best:used.add(best[1]);y.append(1.)
        else:y.append(0.)
    return y

class Logistic:
    def fit(self,X,y,l2=1.):
        X=np.asarray(X,float);y=np.asarray(y,float);self.mu,self.sd=X.mean(0),X.std(0)+1e-6;Z=np.c_[np.ones(len(X)),(X-self.mu)/self.sd];w=np.zeros(Z.shape[1])
        for _ in range(500):  # Newton steps
            p=1/(1+np.exp(-Z@w));g=Z.T@(p-y)+l2*np.r_[0,w[1:]];H=(Z*(p*(1-p))[:,None]).T@Z+l2*np.diag(np.r_[0,np.ones(len(w)-1)])
            step=np.linalg.solve(H,g);w-=step
            if np.abs(step).max()<1e-8:break
        self.w=w;return self
    def predict(self,X):Z=np.c_[np.ones(len(X)),(np.asarray(X,float)-self.mu)/self.sd];return 1/(1+np.exp(-Z@self.w))
    def packet(self):return {'w':self.w.tolist(),'mu':self.mu.tolist(),'sd':self.sd.tolist()}

def label(c,mode):
    p0,p1=c['probs']
    if mode=='avg' and p1:p0={k:(p0.get(k,0)+p1.get(k,0))/2 for k in set(p0)|set(p1)}
    return bfmd_label(p0)

def fit(data):
    X=[x for cs,ys in data for c,y in zip(cs,ys) if y is not None for x in [c['x']]];Y=[y for cs,ys in data for y in ys if y is not None]
    model=Logistic().fit(X,Y)
    # threshold maximising pooled F1 on the training folds (candidates that miss stay misses)
    total=sum(1 for cs,ys in data for y in ys if y)  # matched positives available
    best=(-1,.5)
    for th in np.linspace(.1,.9,33):
        tp=fp=0
        for cs,ys in data:
            if not cs:continue
            keep=model.predict([c['x'] for c in cs])>=th
            tp+=sum(k and y==1 for k,y in zip(keep,ys));fp+=sum(k and y==0 for k,y in zip(keep,ys))
        f=2*tp/(2*tp+fp+(total-tp)) if tp else 0
        best=max(best,(f,th))
    return model,best[1]

def write_route(c,kept,tmodel,th,note):
    seq=[x for x,_ in kept];alt=author_window_probs(c,seq) if USE_ALT and seq else None
    stacked=tmodel.predict([type_x(seq,j,alt) for j in range(len(seq))]) if seq else []
    for x,a in zip(seq,alt or []):x['alt_bst0']=bfmd_label(a[0])
    hits=[{'time':x['time'],'hit_score':v,'labels':{'bst0':{'label':label(x,'bst0')},'bst_cg_ap':{'label':label(x,'avg')},'agent':{'label':t},**({'bst0_author_window':{'label':x['alt_bst0']}} if 'alt_bst0' in x else {})}} for (x,v),t in zip(kept,stacked)]
    save(ROUTES/c['clip_id']/'agent__verify.json',{'clip_id':c['clip_id'],'role':c['role'],'perception':'agent','method':'verify','status':'completed','hit_seconds':0.,'inputs':{},
         'note':note+' label slots: bst0 = BST-0 only, bst_cg_ap = averaged BST-0/CG-AP, agent = stacked type re-check','threshold':th,'hits':hits,'annotation_inputs':[]})
    return len(hits)

def keep(model,th,cs):
    s=model.predict([x['x'] for x in cs]) if cs else []
    return [(x,float(v)) for x,v,k in zip(cs,s,np.asarray(s)>=th) if k]

def fit_both(train,data,types,byid):
    """Hit keep/drop model, then type model on kept+matched+typed candidates of the training clips."""
    model,th=fit([data[k] for k in train]);X=[];Y=[]
    for k in train:
        seq=[x for x,_ in keep(model,th,data[k][0])];ty={id(x):t for x,t in zip(data[k][0],types[k])};alt=author_window_probs(byid[k],seq) if USE_ALT and seq else None
        for j,x in enumerate(seq):
            if ty.get(id(x)):X.append(type_x(seq,j,alt));Y.append(ty[id(x)])
    return model,th,Softmax().fit(X,Y)

def dev_data():
    clips=[c for c in read(STUDY/'clips.json')['clips'] if in_role(c,'dev') and (ROUTES/c['clip_id']).exists()]
    data={c['clip_id']:(cs:=candidates(c),truth_labels(c,cs)) for c in clips}
    return clips,data,{c['clip_id']:truth_types(c,data[c['clip_id']][0]) for c in clips},{c['clip_id']:c for c in clips}

def evaluate():
    clips,data,types,byid=dev_data();out={}
    for m in sorted({c['youtube_id'] for c in clips}):
        model,th,tmodel=fit_both([c['clip_id'] for c in clips if c['youtube_id']!=m],data,types,byid)
        for c in clips:
            if c['youtube_id']==m:out[c['clip_id']]=write_route(c,keep(model,th,data[c['clip_id']][0]),tmodel,th,'leave-one-match-out.')
    return out

def code_hashes():
    import hashlib
    here=Path(__file__).parent
    return {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(here.glob('*.py'))}

FROZEN=STUDY/f'frozen_verify_{VARIANT}.json'
def freeze():
    if FROZEN.exists():raise ValueError('Verification agent already frozen')
    clips,data,types,byid=dev_data();model,th,tmodel=fit_both(list(byid),data,types,byid)
    save(FROZEN,{'dev_clips':sorted(byid),'hit_model':model.packet(),'threshold':th,'type_model':{'W':tmodel.W.tolist(),'mu':tmodel.mu.tolist(),'sd':tmodel.sd.tolist()},
                 'types':TYPES,'primary':PRIMARY,'add_support':ADD_SUPPORT,'use_author_window':USE_ALT,'code_sha256':code_hashes(),
                 'note':'Fitted on all dev clips before the new test rallies were sampled. Applied unchanged to new_test.'})
    return th

def apply(role):
    f=read(FROZEN)
    if f['code_sha256']!=code_hashes():raise ValueError('Agent code changed after freezing')
    model=Logistic();model.w,model.mu,model.sd=(np.array(f['hit_model'][k]) for k in ['w','mu','sd'])
    tmodel=Softmax();tmodel.W,tmodel.mu,tmodel.sd=(np.array(f['type_model'][k]) for k in ['W','mu','sd'])
    out={}
    for c in read(STUDY/'clips.json')['clips']:
        if in_role(c,role) and (ROUTES/c['clip_id']).exists():out[c['clip_id']]=write_route(c,keep(model,f['threshold'],candidates(c)),tmodel,f['threshold'],'frozen agent (fitted on all dev clips).')
    return out

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('step',choices=['evaluate','freeze','apply'],nargs='?',default='evaluate');a.add_argument('--role',default='new_test');a=a.parse_args()
    if a.step=='evaluate':print(len(evaluate()),'dev clips verified (leave-one-match-out)')
    elif a.step=='freeze':print('frozen, threshold',freeze())
    else:print(len(apply(a.role)),'clips verified by the frozen agent')
