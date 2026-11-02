"""Label-free post-processing of hit candidates (tuned on BFMD dev clips only).

refine_frame : move a candidate to the sharpest shuttle-direction change
               within [i-BACK, i+FWD] (detectors tend to fire late).
filter_hits  : drop candidates where the shuttle is barely moving (held before
               a serve, lying after a rally) and isolated candidates far from
               every other hit (rally head/tail noise).
"""
import numpy as np
BACK,FWD,LAG,MIN_TURN=7,1,2,.3     # dev grid search plateau, not the single peak
MIN_SPEED,SPEED_WIN,MAX_GAP=3.,5,60  # px/frame, frames, frames

def track(trajectory):
    return np.array([(f['rectified_xy'] if f.get('inpaint_requested') else f['raw_xy']) or [np.nan,np.nan] for f in trajectory['frames']],float)

def turn(xy,t,k=LAG):
    """0 = straight flight, 2 = full reversal; 0 when either side is unobserved or static."""
    if t-k<0 or t+k>=len(xy):return 0.
    a,b=xy[t]-xy[t-k],xy[t+k]-xy[t]
    if np.isnan(a).any() or np.isnan(b).any() or np.linalg.norm(a)<2 or np.linalg.norm(b)<2:return 0.
    return float(1-a@b/np.linalg.norm(a)/np.linalg.norm(b))

def refine_frame(xy,i):
    best=max((turn(xy,t),t) for t in range(max(0,i-BACK),min(len(xy),i+FWD+1)))
    return best[1] if best[0]>=MIN_TURN else i

def speed(xy,i):
    d=np.linalg.norm(np.diff(xy[max(0,i-SPEED_WIN):i+SPEED_WIN+1],axis=0),axis=1);d=d[~np.isnan(d)]
    return float(np.median(d)) if len(d)>=3 else 0.

def postprocess(events,trajectory):
    xy=track(trajectory);out=[]
    for e in events:
        i=refine_frame(xy,int(e['frame_index']))
        out.append({**e,'frame_index':i,'raw_frame_index':int(e['frame_index']),'refined':i!=int(e['frame_index'])})
    out=sorted({e['frame_index']:e for e in out}.values(),key=lambda e:e['frame_index'])  # refinement can merge duplicates
    out=[e for e in out if speed(xy,e['frame_index'])>=MIN_SPEED]
    if len(out)>=2:out=[e for j,e in enumerate(out) if min(abs(e['frame_index']-o['frame_index']) for k,o in enumerate(out) if k!=j)<=MAX_GAP]
    return out
