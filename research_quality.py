"""ERA 2026333 Eqs.15-32 and measured 2D observations, explicitly separated.

Membership breakpoints are transcribed from Fig.11/12 (PDF p18); Tables1/2
supply all 9+27 rules. No missing index is replaced by zero or an average.
"""
import itertools
import math
from statistics import mean,pstdev

NAMES=['挥速指数','手腕轨迹平滑度指数','腕部姿态指数','肘部姿态指数','步法效率指数','球拍控制指数']
PAPER='https://doi.org/10.3934/era.2026333'


def gaussian(error,sigma):
    if not isinstance(sigma,(int,float)) or not math.isfinite(sigma) or sigma<=0:raise ValueError('A positive validated tolerance is required')
    return math.exp(-error*error/(2*sigma*sigma))


def spi_scores(m,p):
    """Evaluate only with externally validated, stroke-specific measurement refs.

    All input values must use the profile's coordinate units/angle convention.
    Racket alignment here is a validated normal-vector deviation, not a 2D axis.
    """
    if not p.get('validated') or not p.get('stroke_type') or p.get('stroke_type')!=m.get('stroke_type') or p.get('measurement_space')!=m.get('measurement_space'):
        raise ValueError('Validated references for this stroke and measurement space are required')
    out=[None]*6
    for i,key,ref,sigma in [(0,'swing_peak','swing_peak_ref','swing_sigma'),
                            (2,'wrist_angle','wrist_angle_ref','wrist_sigma'),
                            (3,'elbow_angle','elbow_angle_ref','elbow_sigma')]:
        if m.get(key) is not None and p.get(ref) is not None and p.get(sigma) is not None:
            out[i]=gaussian(m[key]-p[ref],p[sigma])
    if m.get('trajectory_error') is not None and p.get('historical_trajectory_errors'):
        out[1]=gaussian(m['trajectory_error'],.3*max(p['historical_trajectory_errors']))
    if all(m.get(k) is not None for k in ['step_count','foot_path_efficiency']) and all(p.get(k) is not None for k in ['step_count_ref','footwork_alpha']):
        alpha=p['footwork_alpha']
        if not 0<=alpha<=1:raise ValueError('Invalid footwork alpha')
        out[4]=alpha*gaussian(m['step_count']-p['step_count_ref'],1)+(1-alpha)*m['foot_path_efficiency']
    if all(m.get(k) is not None for k in ['racket_mean_normal_error','racket_normal_change_std']) and p.get('racket_beta') is not None:
        beta=p['racket_beta']
        if not 0<=beta<=1:raise ValueError('Invalid racket beta')
        out[5]=beta*gaussian(m['racket_mean_normal_error'],.5)+(1-beta)*gaussian(m['racket_normal_change_std'],.5)
    if any(v is not None and (not math.isfinite(v) or not 0<=v<=1) for v in out):raise ValueError('Invalid SPI measurement')
    return out


def triangle(x,a,b,c):
    if x==b:return 1.
    return max(0.,min((x-a)/(b-a),(c-x)/(c-b)))


def premise(x):
    return [1. if x<=.25 else max(0.,(.5-x)/.25),triangle(x,.25,.5,.75),1. if x>=.75 else max(0.,(x-.5)/.25)]


def consequence(x):
    return [1. if x<=1/6 else max(0.,(1/3-x)/(1/6)),triangle(x,1/6,1/3,.5),
            triangle(x,1/3,.5,2/3),triangle(x,.5,2/3,5/6),1. if x>=5/6 else max(0.,(x-2/3)/(1/6))]


LEVEL2=[0,1,2, 1,2,2, 1,2,2,
        1,2,2, 2,2,2, 2,2,3,
        2,2,3, 2,2,3, 2,3,4]


def fuzzy_module(values):
    if any(not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<=1 for x in values):raise ValueError('Fuzzy inputs must lie in [0,1]')
    memberships=[premise(x) for x in values];fired=[]
    for idx,terms in enumerate(itertools.product(range(3),repeat=len(values))):
        strength=min(memberships[i][t] for i,t in enumerate(terms))
        output=sum(terms) if len(values)==2 else LEVEL2[idx]
        if strength>0:fired.append({'antecedents':list(terms),'consequent':output,'strength':strength})
    # Composite trapezoidal quadrature of the exact piecewise linear Mamdani set.
    numerator=denominator=0.
    for i in range(1201):
        x=i/1200;mu=consequence(x);y=max(min(r['strength'],mu[r['consequent']]) for r in fired)
        weight=.5 if i in [0,1200] else 1.
        numerator+=weight*x*y;denominator+=weight*y
    return {'value':numerator/denominator,'inputs':values,'fired_rules':fired}


def hierarchical_fuzzy(indices):
    if len(indices)!=6:raise ValueError('Six SPI fields are required')
    if any(v is None for v in indices):return {'status':'unavailable','score':None,'reason':'Six validated stroke-specific indices are required','missing_indices':[i+1 for i,v in enumerate(indices) if v is None]}
    level1=[fuzzy_module(indices[i:i+2]) for i in [0,2,4]]
    level2=fuzzy_module([m['value'] for m in level1])
    return {'status':'formula_computed','score':level2['value'],'rating_1_5':1+4*level2['value'],
            'level1':level1,'level2':level2,'source':PAPER,'method':'Mamdani min/max and centroid; Figures11-12, Tables1-2, Eqs28-32',
            'interpretation':'Formula output conditional on measurement/reference validity; no automatic biomechanical validation'}


def norm(v):return math.sqrt(sum(x*x for x in v))

def distance(a,b):return norm([x-y for x,y in zip(a,b)])

def angle(a,b,c):
    u=[x-y for x,y in zip(a,b)];v=[x-y for x,y in zip(c,b)];den=norm(u)*norm(v)
    return math.degrees(math.acos(max(-1.,min(1.,sum(x*y for x,y in zip(u,v))/den)))) if den>1e-8 else None


def measure_2d(event,dense,point_threshold=.5):
    player=event.get('player_id');a,b=event['action_start_frame'],event['action_end_frame'];hit=event['hit_frame']
    observations={};coverage={}
    def point(i,k):
        person=dense['frames'][i].get('players',{}).get(player) if player else None
        if not person:return None
        joint=person['keypoints'][k]
        return [joint['x'],joint['y']] if joint['score']>=point_threshold and joint.get('in_frame',True) else None
    for side,(shoulder,elbow,wrist) in {'left':(5,7,9),'right':(6,8,10)}.items():
        pts=[point(i,wrist) for i in range(a,b+1)];coverage[side+'_wrist']=sum(p is not None for p in pts)/len(pts)
        velocities=[];errors=[]
        for k in range(1,len(pts)):
            if pts[k] is not None and pts[k-1] is not None:
                dt=dense['frames'][a+k]['source_time']-dense['frames'][a+k-1]['source_time']
                if dt>0:velocities.append(distance(pts[k],pts[k-1])/dt)
        for k in range(1,len(pts)-1):
            if all(pts[j] is not None for j in [k-1,k,k+1]):errors.append(norm([pts[k+1][d]-2*pts[k][d]+pts[k-1][d] for d in [0,1]]))
        observations[side+'_wrist_peak_px_s']=max(velocities) if velocities and coverage[side+'_wrist']>=.8 else None
        observations[side+'_wrist_trajectory_error_px']=mean(errors) if errors and coverage[side+'_wrist']>=.8 else None
        three=[point(hit,k) for k in [shoulder,elbow,wrist]]
        observations[side+'_elbow_angle_2d_deg']=angle(*three) if all(p is not None for p in three) else None
    efficiencies=[]
    for ankle in [15,16]:
        pts=[point(i,ankle) for i in range(a,b+1)]
        if all(p is not None for p in pts):
            efficiencies.append(distance(pts[-1],pts[0])/(sum(distance(p,q) for p,q in zip(pts,pts[1:]))+1e-8))
    observations['foot_path_efficiency_2d']=mean(efficiencies) if len(efficiencies)==2 else None
    reasons=['缺少挥拍手的人工确认及经过验证的同类动作参考值',
             '缺少同一球员、同一动作、同一坐标尺度的历史轨迹误差参考',
             'COCO17没有手指点，无法按论文计算腕关节角',
             '当前只有二维投影角，缺少对应的同类动作验证参考',
             '缺少验证过的步数、参考步数及混合系数；仅展示二维路径比',
             '球拍五点为二维观测，不能据此报告经过验证的三维拍面法向量']
    indices=[{'id':'spi_'+str(i+1),'name':name,'value':None,'rating_1_5':None,'status':'unavailable','reason':reasons[i],
              'source':PAPER,'equations':[[15],[16,17],[18],[19],[20,21,22],[23,24,25,26,27]][i]} for i,name in enumerate(NAMES)]
    return {'indices':indices,'observations_2d':observations,'coverage':coverage,'measurement_space':'native_image_2d',
            'window_basis':event.get('action_interval_source'),'hitting_hand':'unknown','quality_score':None,
            'fuzzy':hierarchical_fuzzy([None]*6),'limitations':['像素速度不是米/秒','二维观测受视角、尺度和跟踪误差影响','动作窗口和球员身份可能尚未核对']}
