"""Racket-motion evidence for stroke typing (experimental; not yet part of typeens.rows): how the hitter's racket moves around the
contact, from the RacketVision racket keypoints, and how much the image changes around the hitting hand, from frame differences."""
import numpy as np
from benchmarks.bfmd.verify import STUDY,read

W=10  # frames on each side of the contact
_C={}

def _load(clip_id):
    if clip_id not in _C:
        _C.clear();pf=STUDY/'perception'/clip_id;_C[clip_id]=(read(pf/'dense_pose.json'),read(pf/'racketvision/racketvision.json')['frames'])
    return _C[clip_id]

def sequence(clip_id,f,side):
    """(2W+1, 10) per frame: racket head and handle relative to the hitter's hip (body heights), shaft length (foreshortening),
    shaft angle, face width, detection score, distance to the wrist, and whether the racket was seen. side: 1 near, 0 far."""
    dense,rk=_load(clip_id);name='near' if side>.5 else 'far';n=min(len(dense['frames']),len(rk));seq=np.full((2*W+1,9),np.nan)
    for k,t in enumerate(range(f-W,f+W+1)):
        if not 0<=t<n:continue
        pl=dense['frames'][t]['players'].get(name)
        if not pl or not rk[t]['rackets']:continue
        kp=pl['keypoints'];b=pl['bbox'];bh=max(b[3]-b[1],1.);wr=np.array([[kp[j]['x'],kp[j]['y']] for j in (9,10)]);hip=np.array([(kp[11]['x']+kp[12]['x'])/2,(kp[11]['y']+kp[12]['y'])/2]);best=None
        for r in rk[t]['rackets']:  # the racket whose handle or bottom is next to one of this player's wrists
            q=np.array(r['keypoints']);d=min(np.linalg.norm(wr-q[2],axis=1).min(),np.linalg.norm(wr-q[1],axis=1).min())/bh
            if d<.6 and (best is None or d<best[0]):best=(d,q,r['bbox_score'])
        if best is None:continue
        d,q,sc=best;head=q[[0,3,4]].mean(0);shaft=q[0]-q[2]
        seq[k]=[*(head-hip)/bh,*(q[2]-hip)/bh,np.linalg.norm(shaft)/bh,np.arctan2(shaft[1],shaft[0]),np.linalg.norm(q[3]-q[4])/bh,sc,d]
    ok=~np.isnan(seq[:,0]);return np.c_[np.nan_to_num(seq),ok.astype(float)]

def kinematics(clip_id,f,side):
    """23 summary values of the swing: head speed before / after the contact, arc lengths, velocity into and out of the contact,
    head position, shaft length and face width at the contact, shaft rotation, range of motion, and how often the racket was seen."""
    s=sequence(clip_id,f,side);ok=s[:,9]>0;seq=np.where(ok[:,None],s[:,:9],np.nan);c=W;sg=-1. if side>.5 else 1.
    def vel(a,b):
        ts=[t for t in range(a,b+1) if ok[t]];return (seq[ts[-1],:2]-seq[ts[0],:2])/(ts[-1]-ts[0]) if len(ts)>1 else None
    sp=np.array([np.linalg.norm(seq[t+1,:2]-seq[t,:2]) if ok[t] and ok[t+1] else np.nan for t in range(2*W)]);pre,post=sp[c-6:c],sp[c:c+6]
    f_=lambda fn,a:float(fn(a)) if np.isfinite(a).any() else 0.;vi,vo=vel(c-4,c),vel(c,c+4);z=np.zeros(2);mid=slice(c-1,c+2);has=ok[mid].any()
    ang=seq[ok,5];dang=np.abs(np.angle(np.exp(1j*np.diff(ang)))) if ok.sum()>1 else np.zeros(1)
    out=[f_(np.nanmax,pre),f_(np.nanmean,pre),f_(np.nanmax,post),f_(np.nanmean,post),f_(np.nanmax,sp),f_(np.nansum,pre),f_(np.nansum,post),
         *((vi*[1,sg]) if vi is not None else z),*((vo*[1,sg]) if vo is not None else z),float(vi is not None),float(vo is not None),
         *((np.nanmean(seq[mid,:2],0)*[1,sg]) if has else z),float(np.nanmean(seq[mid,4])) if has else 0.,float(np.nanmean(seq[mid,6])) if has else 0.,
         float(dang.max()),float(dang.sum()),float(np.nanmax(seq[:,1]*sg)-np.nanmin(seq[:,1]*sg)) if ok.any() else 0.,float(np.nanmax(seq[:,0])-np.nanmin(seq[:,0])) if ok.any() else 0.,
         float(ok.mean()),float(ok[c-3:c+4].mean())]
    return np.nan_to_num(np.array(out,float))

def swing_energy(gray,clip_id,f,side):
    """(2W+1, 4) frame-difference energy relative to the clip background: around the hitter's wrists, on the racket side of the wrist
    (along the forearm), over the hitter's body and over the opponent's body. gray: the clip's frames as float32 grayscale."""
    dense,_=_load(clip_id);name,other=('near','far') if side>.5 else ('far','near');out=np.zeros((2*W+1,5),np.float32)
    def disc(D,c,r):
        h,w=D.shape;x0,x1=int(max(0,c[0]-r)),int(min(w,c[0]+r+1));y0,y1=int(max(0,c[1]-r)),int(min(h,c[1]+r+1));return float(D[y0:y1,x0:x1].mean()) if x1>x0 and y1>y0 else 0.
    box=lambda D,b:float(D[int(max(0,b[1])):int(b[3])+1,int(max(0,b[0])):int(b[2])+1].mean()) if b[3]>b[1] and b[2]>b[0] else 0.
    for k,t in enumerate(range(f-W,f+W+1)):
        if not 1<=t<min(len(gray),len(dense['frames'])):continue
        pl=dense['frames'][t]['players'].get(name);op=dense['frames'][t]['players'].get(other)
        if not pl:continue
        D=np.abs(gray[t]-gray[t-1]);kp=pl['keypoints'];bh=max(pl['bbox'][3]-pl['bbox'][1],1.);e=[]
        for wj,ej in ((9,7),(10,8)):
            wr=np.array([kp[wj]['x'],kp[wj]['y']]);el=np.array([kp[ej]['x'],kp[ej]['y']]);u=(wr-el)/(np.linalg.norm(wr-el)+1e-6);e.append((disc(D,wr,.2*bh),disc(D,wr+u*.35*bh,.3*bh)))
        out[k]=[max(e[0][0],e[1][0]),max(e[0][1],e[1][1]),box(D,pl['bbox']),box(D,op['bbox']) if op else 0.,float(np.median(D))]
    return out[:,:4]/(out[:,4].mean()+1e-3)
