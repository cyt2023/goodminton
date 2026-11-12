"""Stroke-type ensemble applied after the contact-time pass: extra-trees + random forest + logistic regression on the stacked
type features, BFMD head, flight and pose at the modelled contact frame, plus a sequence CNN on per-frame evidence; equal-weight average."""
import numpy as np
from benchmarks.bfmd import contact as ct
from benchmarks.bfmd.verify import STUDY,TYPES,read,flight,court_h,type_x,author_window_probs,speed,to_court
from benchmarks.bfmd.refine import track,turn

import os
L=24;K=len(TYPES);_CLIP={};_HIT={}
D3=os.environ.get('TYPE_3D','1')=='1'  # 3D flight reconstruction
CHAIN=os.environ.get('TYPE_CHAIN','1')=='1'  # flights of a rally fitted as a chain through shared contact points
TURNS=os.environ.get('TYPE_TURNS','1')=='1'  # hitter from turn-taking instead of the nearest feet

def _pose(dense,f,side):
    """Hitter around the contact: wrist above head / shoulder, arm reach, body size."""
    H_=dense['source_height'];best=None
    for t in range(max(0,f-3),min(len(dense['frames']),f+4)):
        pl=dense['frames'][t]['players'].get('near' if side>.5 else 'far')
        if not pl:continue
        kp=pl['keypoints'];b=pl['bbox'];bh=max(b[3]-b[1],1.);y=lambda j:kp[j]['y'];x=lambda j:kp[j]['x'];wr=min((9,10),key=y)
        v=[(y(0)-y(wr))/bh,(min(y(5),y(6))-y(wr))/bh,abs(x(wr)-(x(5)+x(6))/2)/bh,bh/H_,(b[2]-b[0])/bh]
        if best is None or v[0]>best[0]:best=v
    return best if best is not None else [0.]*5

def _rules(dense,xy,f,side):
    """Where the hitter meets the shuttle (height above head / hip, in front or behind, in body heights) and how the shuttle arrived
    (how far above the contact it came from, image drop over the last half second, speed just before contact)."""
    n=len(xy);ok=lambda t:0<=t<n and not np.isnan(xy[t]).any();base=next((t for t in range(f,min(n,f+3)) if ok(t)),None);out=[0.]*4
    pl=dense['frames'][min(f,len(dense['frames'])-1)]['players'].get('near' if side>.5 else 'far')
    if base is not None and pl:
        kp=pl['keypoints'];b=pl['bbox'];bh=max(b[3]-b[1],1.);x,y=xy[base]
        out=[(kp[0]['y']-y)/bh,((kp[11]['y']+kp[12]['y'])/2-y)/bh,(x-(b[0]+b[2])/2)/bh,1.]
    seg=[xy[t][1] for t in range(max(0,f-15),f+1) if ok(t)]
    out+=[(min(seg)-seg[-1])/400,(seg[0]-seg[-1])/400] if len(seg)>1 else [0.,0.]
    out+=[speed(xy,f-3)/30 if f-3>=2 else 0.];return [float(v) for v in np.nan_to_num(out)]

def _kin(dense,xy,f,side,H):
    """Shuttle kinematics at the direction change, in hitter body heights per frame: velocity after and before the contact, the turn
    between them, speed ratio and acceleration, how the speed decays, apex of the flight; and the hitter's feet on the court plane."""
    n=len(xy);ok=lambda t:0<=t<n and not np.isnan(xy[t]).any()
    pl=dense['frames'][min(f,len(dense['frames'])-1)]['players'].get('near' if side>.5 else 'far')
    bh=max(pl['bbox'][3]-pl['bbox'][1],1.) if pl else .25*dense['source_height']
    def vel(lo,hi):
        ts=[t for t in range(lo,hi+1) if ok(t)]
        return (xy[ts[-1]]-xy[ts[0]])/(ts[-1]-ts[0])/bh if len(ts)>1 and ts[-1]-ts[0]>=2 else None
    vo,vi,vl=vel(f+1,f+5),vel(f-5,f-1),vel(f+8,f+13);z=np.zeros(2);so,si,sl=(float(np.linalg.norm(v)) if v is not None else 0. for v in (vo,vi,vl))
    out=[*(vo if vo is not None else z),so,*(vi if vi is not None else z),si,float(vo is not None),float(vi is not None)]
    both=vo is not None and vi is not None
    out+=[float(vo@vi/(so*si+1e-6)) if both else 0.,float(np.log((so+.02)/(si+.02))) if both else 0.,float(np.linalg.norm(vo-vi)) if both else 0.]
    out+=[sl,float(np.log((sl+.02)/(so+.02))) if vo is not None and vl is not None else 0.]
    seg=[(t,xy[t][1]) for t in range(f,min(n,f+26)) if ok(t)]
    if len(seg)>1:t,y=min(seg,key=lambda v:v[1]);out+=[(y-seg[0][1])/bh,(t-f)/25]
    else:out+=[0.,0.]
    feet=to_court(H,[(pl['bbox'][0]+pl['bbox'][2])/2,pl['bbox'][3]]) if pl else None;lead=feet
    if pl:  # the foot nearer the net: a player lunging to a short ball takes it above the leading foot, not under the body
        an=[to_court(H,[pl['keypoints'][j]['x'],pl['keypoints'][j]['y']]) for j in (15,16)];lead=min(an,key=lambda a:abs(a[1]-.5))
        if np.linalg.norm((lead-feet)*np.array([6.1,13.4]))>2.:lead=feet
    return [float(v) for v in np.nan_to_num(out)],feet,lead

def side_evidence(dense,xy,f):
    """Log-odds that the near player hit: whose arm (elbow / wrist) is closer to the shuttle around the contact, in body heights."""
    n=min(len(xy),len(dense['frames']));d={}
    for side in ('near','far'):
        best=9.
        for t in range(max(0,f-2),min(n,f+3)):
            pl=dense['frames'][t]['players'].get(side)
            if pl and not np.isnan(xy[t]).any():
                kp=pl['keypoints'];bh=max(pl['bbox'][3]-pl['bbox'][1],1.);best=min(best,min(np.hypot(kp[j]['x']-xy[t][0],kp[j]['y']-xy[t][1]) for j in (7,8,9,10))/bh)
        d[side]=best
    return float(np.clip(np.log((d['far']+.05)/(d['near']+.05)),-3,3)) if min(d.values())<9 else 0.

def hitters(ev,times,stay=2.5):
    """Who hit each logged stroke (1 near, 0 far): players take turns, so decode the rally with a penalty for the same side twice."""
    n=len(ev)
    if not n:return []
    sc=np.array([[-e/2,e/2] for e in ev]);d=sc[0].copy();bp=[]
    for t in range(1,n):
        pen=stay if times[t]-times[t-1]<=2.5 else 0.;tr=np.array([[-pen,0.],[0.,-pen]])+d[:,None];bp.append(tr.argmax(0));d=tr.max(0)+sc[t]
    k=int(d.argmax());out=[k]
    for b in bp[::-1]:k=int(b[k]);out.append(k)
    return [float(v) for v in out[::-1]]

_RAW={}
def _raw(clip):
    """Shuttle detections only (no inpainted points): (n, 2) with NaN where TrackNet saw nothing."""
    k=clip['clip_id']
    if k not in _RAW:
        _RAW.clear();t=read(STUDY/'perception'/k/'tracknetv3/trajectory.json')
        _RAW[k]=np.array([f['raw_xy'] if f.get('raw_visible') and f.get('raw_xy') else [np.nan,np.nan] for f in t['frames']],float)
    return _RAW[k]

def segfit(clip,f,prev,nxt,side):
    """The flight on each side of the direction change, fitted on real detections: launch velocity and acceleration of the outgoing
    flight (quadratic fit of its first third of a second), its whole path up to the next hit (length, straightness, mean and final
    speed, rise), the arrival velocity of the incoming flight, the turn between them and where the track actually turns.
    Units: hitter body heights and frames; vertical axis signed so that + is away from the hitter's end."""
    raw=_raw(clip);dense=_clip(clip)[0];n=len(raw);pl=dense['frames'][min(f,len(dense['frames'])-1)]['players'].get('near' if side>.5 else 'far')
    bh=max(pl['bbox'][3]-pl['bbox'][1],1.) if pl else .25*dense['source_height'];sg=np.array([1.,-1. if side>.5 else 1.])  # near hitter: up the image (-y) is away
    def pts(lo,hi):
        ts=np.array([t for t in range(max(lo,0),min(hi,n-1)+1) if not np.isnan(raw[t]).any()],int);return ts,(raw[ts]*sg/bh if len(ts) else np.zeros((0,2)))
    def fit(ts,P,t0):
        if len(ts)>=4:c=np.polyfit(ts-t0,P,2);return c[1],2*c[0],1.
        if len(ts)>=2:c=np.polyfit(ts-t0,P,1);return c[0],np.zeros(2),.5
        return np.zeros(2),np.zeros(2),0.
    g=min(nxt-1 if nxt is not None else f+30,f+30);e=max(prev+1 if prev is not None else f-30,f-30)
    to,Po=pts(f+1,min(g,f+10));vo,ao,qo=fit(to[:8],Po[:8],f);ta,Pa=pts(f+1,g);ti,Pi=pts(max(e,f-10),f-1);vi,ai,qi=fit(ti[-8:],Pi[-8:],f)
    out=[*vo,float(np.linalg.norm(vo)),*ao,qo,len(to)/10,*vi,float(np.linalg.norm(vi)),qi,len(ti)/10]
    if len(ta)>=3:
        d=np.linalg.norm(np.diff(Pa,axis=0),axis=1);path=float(d.sum());chord=float(np.linalg.norm(Pa[-1]-Pa[0]));dur=float(ta[-1]-ta[0])
        out+=[path,chord,chord/(path+1e-6),path/max(dur,1.),float(np.linalg.norm(Pa[-1]-Pa[-3])/max(ta[-1]-ta[-3],1)),float((Pa[:,1]-Pa[0,1]).max()),float((Pa[:,1]-Pa[0,1]).min()),float(Pa[-1,1]-Pa[0,1]),float(abs(Pa[-1,0]-Pa[0,0])),len(ta)/max(g-f,1)]
    else:out+=[0.]*9+[len(ta)/max(g-f,1)]
    so,si=np.linalg.norm(vo),np.linalg.norm(vi);both=qo>0 and qi>0
    out+=[float(vo@vi/(so*si+1e-6)) if both else 0.,float((vo[0]*vi[1]-vo[1]*vi[0])/(so*si+1e-6)) if both else 0.,float(np.log((so+.02)/(si+.02))) if both else 0.,float(np.linalg.norm(vo-vi)) if both else 0.]
    xy=_clip(clip)[1];tt=[(turn(xy,t),t) for t in range(max(2,f-4),min(len(xy)-2,f+5))];best=max(tt) if tt else (0.,f);out+=[float(best[0]),float(best[1]-f)]
    return [float(v) for v in np.nan_to_num(out)]

_CAM={}
def flights3(clip,frames,sides,feet,lead,linked):
    """3D flights of a rally's logged strokes, fitted as a chain (flight3d.chain): one flight3d feature list per stroke."""
    from benchmarks.bfmd import flight3d as f3
    key=(clip['clip_id'],tuple(frames),tuple(sides))
    if key in _HIT:return _HIT[key]
    if clip['youtube_id'] not in _CAM:
        dense=_clip(clip)[0];_CAM[clip['youtube_id']]=f3.camera(read(STUDY/'court_review.json')['matches'][clip['youtube_id']]['corners_TL_TR_BL_BR'],dense['source_width'],dense['source_height'])[0]
    raw=_raw(clip);n=len(raw);sc=np.array([f3.CW,f3.CL]);F=[]
    for j,f in enumerate(frames):
        g=frames[j+1] if linked[j] else None;end=min(g if g is not None else f+30,f+45,n-1);ok=g is not None and end==g
        F.append(dict(obs=[(t-f,*raw[t]) for t in range(f+1,end) if not np.isnan(raw[t]).any()],start_xy=feet[j]*sc if feet[j] is not None else None,
                      end_xy=lead[j+1]*sc if ok and lead[j+1] is not None else None,nframes=max(end-f,1),linked=ok))
    R=f3.chain(_CAM[clip['youtube_id']],F) if CHAIN else [f3.fit(_CAM[clip['youtube_id']],x['obs'],x['start_xy'],x['end_xy'],x['nframes']) if x['start_xy'] is not None and len(x['obs'])>=3 else None for x in F]
    out=[f3.features(*r,sides[j])+[1.] if r is not None else [0.]*f3.NF+[0.] for j,r in enumerate(R)];_HIT[key]=out;return out

def _win(F,c):
    out=np.zeros((2*L+1,F.shape[1]),np.float32);lo,hi=max(0,c-L),min(len(F),c+L+1);out[lo-(c-L):hi-(c-L)]=F[lo:hi];return out

def _clip(clip):
    k=clip['clip_id']
    if k not in _CLIP:
        _CLIP.clear();pf=STUDY/'perception'/k
        _CLIP[k]=(read(pf/'dense_pose.json'),track(read(pf/'tracknetv3/trajectory.json')),court_h(clip),ct.frame_features(clip))
    return _CLIP[k]

def at_frame(clip,f,side=None):
    """Flight (16), pose (5), the per-frame evidence window, contact-geometry rules (7), shuttle kinematics and the hitter's feet at
    frame f; side: who hit (1 near, 0 far), default the flight features' own guess."""
    key=(clip['clip_id'],f,side)
    if key not in _HIT:
        dense,xy,H,F=_clip(clip);f=int(min(max(f,0),len(F)-1));tr=flight(xy,dense['frames'],f,H);sd=tr[14] if side is None else side
        _HIT[key]=(tr,_pose(dense,f,sd),_win(F,f),_rules(dense,xy,f,sd),*_kin(dense,xy,f,sd,H))
    return _HIT[key]

def evidence_at(clip,f):
    key=(clip['clip_id'],f,'ev')
    if key not in _HIT:dense,xy,H,F=_clip(clip);_HIT[key]=side_evidence(dense,xy,int(min(max(f,0),len(F)-1)))
    return _HIT[key]

def rows(r):
    """Features of every logged hit of a rally (after the contact pass): (hits, tabular, windows)."""
    seq=r.log()
    if not seq:return [],np.zeros((0,153)),np.zeros((0,2*L+1,1),np.float32)
    ctx=[dict(x,probs=(dict.fromkeys(TYPES,0.)|{x['clamp']:1.},)*2) if x['clamp'] else x for x in seq];alt=author_window_probs(r.clip,seq);A=[];Wn=[]
    fr=[x.get('cbest',x['frame']) for x in seq];ev=[evidence_at(r.clip,f) for f in fr]
    sides=hitters(ev,[x['time'] for x in seq]) if TURNS else [None]*len(seq)
    at=[at_frame(r.clip,f,sd) for f,sd in zip(fr,sides)]
    f3d=flights3(r.clip,fr,sides,[a_[5] for a_ in at],[a_[6] for a_ in at],[j+1<len(seq) and seq[j+1]['time']-seq[j]['time']<2.5 for j in range(len(seq))]) if TURNS and D3 else None
    for j,x in enumerate(seq):
        tr,po,w,ru,kin,feet,lead=at[j]
        gap=[min(x['time']-seq[j-1]['time'],2.5) if j else 2.5,min(seq[j+1]['time']-x['time'],2.5) if j+1<len(seq) else 2.5]
        pv=at[j-1] if j and gap[0]<2.5 else None;nx=at[j+1] if j+1<len(seq) and gap[1]<2.5 else None
        # rally rules: how the previous stroke flew and from how deep; where and how high the next player takes this one
        nb=([*pv[0][9:12],pv[0][15],*pv[3][:2],1.] if pv else [0.]*7)+([nx[0][15],*nx[3][:3],nx[0][13]-tr[13],1.] if nx else [.25,0.,0.,0.,0.,0.])
        def ground(o,g):  # court-plane distance between the two players' feet, its lateral part, and the average ground speed of the flight
            if o is None or o[5] is None or feet is None:return [0.,0.,0.,0.]
            d=(o[5]-feet)*np.array([6.1,13.4]);return [float(np.linalg.norm(d)),abs(float(d[0])),float(np.linalg.norm(d))/max(g,.1),1.]
        A.append(list(type_x(ctx,j,alt))+list(np.log(np.asarray(x['head'])+1e-6))+list(tr)+list(po)+gap+ru+nb+kin+ground(nx,gap[1])+ground(pv,gap[0])
                 +(segfit(r.clip,fr[j],fr[j-1] if pv else None,fr[j+1] if nx else None,sides[j]) if TURNS else [])
                 +(f3d[j] if f3d is not None else [])
                 +([sides[j],ev[j],abs(feet[1]-.5)*13.4 if feet is not None else 3.,(feet[0]-.5)*6.1 if feet is not None else 0.,kin[1]*(1 if sides[j] else -1),kin[4]*(1 if sides[j] else -1)] if TURNS else []));Wn.append(w)
    return seq,np.array(A,float),np.stack(Wn)

def reference_windows(clips):
    """Per-frame evidence windows at the typed reference contacts (with +-2 frame jitter) of the given clips."""
    man={m['match_id']:m for m in read(STUDY.parent/'manifest.json')['matches']};X=[];Y=[]
    for c in clips:
        F=ct.frame_features(c);lo=c['offset_seconds'];hi=lo+c['duration_seconds']
        for h in man[c['match_id']]['hits']:
            f=round((h['time']-lo)*30)
            if lo<=h['time']<hi and h['label'] in TYPES and 0<=f<len(F):
                X+=[_win(F,f+o) for o in (-2,-1,0,1,2)];Y+=[TYPES.index(h['label'])]*5
    return np.stack(X),np.array(Y)

class SeqNet:
    def fit(self,Xw,Yw,seed=0,epochs=40):
        import torch
        torch.manual_seed(seed);rng=np.random.default_rng(seed);d=Xw.shape[2];flat=Xw.reshape(-1,d);self.mu,self.sd=flat.mean(0),flat.std(0)+1e-6;self.d=d
        self.build();x=torch.from_numpy(((Xw-self.mu)/self.sd).astype(np.float32)).transpose(1,2);t=torch.from_numpy(Yw.astype(np.int64))
        opt=torch.optim.AdamW(self.net.parameters(),2e-3,weight_decay=1e-2);self.net.train()
        for _ in range(epochs):
            for b in np.array_split(rng.permutation(len(t)),max(1,len(t)//128)):
                loss=torch.nn.functional.cross_entropy(self.fwd(x[b]),t[b],label_smoothing=.1);opt.zero_grad();loss.backward();opt.step()
        self.net.eval();return self
    def build(self):
        import torch
        nn=torch.nn;self.net=nn.ModuleDict({'conv':nn.Sequential(nn.Conv1d(self.d,48,5,padding=2),nn.ReLU(),nn.Conv1d(48,48,5,padding=4,dilation=2),nn.ReLU(),nn.Dropout(.3),
            nn.Conv1d(48,48,5,padding=8,dilation=4),nn.ReLU(),nn.Dropout(.3)),'head':nn.Linear(96,K)})
    def fwd(self,x):
        import torch
        h=self.net['conv'](x);c=h.shape[2]//2;return self.net['head'](torch.cat([h[:,:,c-3:c+4].mean(2),h.amax(2)],1))
    def proba(self,Xq):
        import torch
        with torch.no_grad():return torch.softmax(self.fwd(torch.from_numpy(((Xq-self.mu)/self.sd).astype(np.float32)).transpose(1,2)),1).numpy()
    def __getstate__(self):return {'mu':self.mu,'sd':self.sd,'d':self.d,'state':{k:v.numpy() for k,v in self.net.state_dict().items()}}
    def __setstate__(self,s):
        import torch
        self.mu,self.sd,self.d=s['mu'],s['sd'],s['d'];self.build();self.net.load_state_dict({k:torch.from_numpy(v) for k,v in s['state'].items()});self.net.eval()

class TypeEnsemble:
    SEEDS=3  # classes weighted equally: without it the rare, flat push is mostly called a lift or a net shot
    def fit(self,A,Y,Xw,Yw):
        from sklearn.ensemble import ExtraTreesClassifier as ET,RandomForestClassifier as RF
        from sklearn.linear_model import LogisticRegression as LR
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        self.tab=[[ET(500,min_samples_leaf=2,n_jobs=-1,random_state=s,class_weight='balanced').fit(A,Y) for s in range(self.SEEDS)],
                  [RF(500,min_samples_leaf=2,n_jobs=-1,random_state=s,class_weight='balanced').fit(A,Y) for s in range(self.SEEDS)],
                  [make_pipeline(StandardScaler(),LR(C=.1,max_iter=3000,class_weight='balanced')).fit(A,Y)]]
        self.seq=[SeqNet().fit(Xw,Yw,s) for s in range(self.SEEDS)];return self
    def proba(self,A,Wn):
        def full(m):P=np.zeros((len(A),K));P[:,m.classes_]=m.predict_proba(A);return P
        P=sum(np.mean([full(m) for m in g],0) for g in self.tab)+np.mean([s.proba(Wn) for s in self.seq],0);return P/P.sum(1,keepdims=True)

def fit(train_rallies,train_clips):
    """Ensemble on the training matches' logged hits (after their automatic passes) that match a typed reference contact."""
    A=[];Y=[]
    for r in train_rallies:
        seq,a,_=rows(r);idx={id(x):i for i,x in enumerate(r.c)}
        for x,v in zip(seq,a):
            i=idx[id(x)]
            if r.y[i]==1 and r.ty[i]:A.append(v);Y.append(TYPES.index(r.ty[i]))
    Xw,Yw=reference_windows(train_clips);return TypeEnsemble().fit(np.array(A),np.array(Y),Xw,Yw)

def type_pass(r,ens):
    """Before review: replace the emission of every logged hit by the ensemble's probabilities, then decode."""
    seq,A,Wn=rows(r)
    if not seq:return
    for x,p in zip(seq,ens.proba(A,Wn)):x['tpe']=p
    r.decode()
