"""Small OpenCV worker for court previews and reviewed planar correspondences."""
import itertools
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from court_calibration import color_mask, project


def template():
    xs = [('L', 0.), ('SL', .46 / 6.1), ('C', .5), ('SR', 1 - .46 / 6.1), ('R', 1.)]
    ys = [('F', 0.), ('FD', .76 / 13.4), ('FS', (6.7 - 1.98) / 13.4),
          ('NS', (6.7 + 1.98) / 13.4), ('ND', 1 - .76 / 13.4), ('N', 1.)]
    return [{'id': yid + '_' + xid, 'xy': [x, y]} for yid, y in ys for xid, x in xs]


def automatic_corners(mask):
    """Two line families propose a visible quadrilateral, never a calibration."""
    h, w = mask.shape
    raw = cv2.HoughLinesP(cv2.Canny(mask, 60, 160), 1, np.pi/180, 45,
                          minLineLength=w*.09, maxLineGap=w*.025)
    if raw is None:
        return None
    segments = sorted(raw.reshape(-1, 4), key=lambda a: -np.linalg.norm(a[2:]-a[:2]))
    lines = []
    for seg in segments:
        a, b = seg[:2].astype(float), seg[2:].astype(float)
        v = b-a; angle = np.arctan2(v[1], v[0]) % np.pi
        eq = np.cross([*a, 1], [*b, 1]); eq /= np.linalg.norm(eq[:2])
        if eq[0] < 0: eq *= -1
        if any(abs(np.dot(eq, [*p, 1])) < 7 and abs(np.sin(angle-an)) < .05 for _, an, p in lines):
            continue
        lines.append((eq, angle, (a+b)/2))
        if len(lines) == 24: break
    if len(lines) < 4: return None
    distance = lambda a,b: abs(np.arctan2(np.sin(a-b), np.cos(a-b))) % np.pi
    def da(a,b): return min(distance(a,b), np.pi-distance(a,b))
    # Transverse lines are nearly parallel; perspective sidelines can converge
    # at substantially different angles, so keep both sides in the second set.
    first = lines[0][1]
    g1 = [v[0] for v in lines if da(v[1], first) < np.deg2rad(18)][:8]
    g2 = [v[0] for v in lines if da(v[1], first) >= np.deg2rad(18)][:12]
    best, best_score = None, 0
    support = cv2.dilate(mask, np.ones((7,7), np.uint8))
    for pair1 in itertools.combinations(g1,2):
        for pair2 in itertools.combinations(g2,2):
            pts=[]
            for a,b in itertools.product(pair1,pair2):
                p=np.cross(a,b)
                if abs(p[2]) < 1e-8: break
                pts.append(p[:2]/p[2])
            if len(pts)!=4: continue
            pts=np.array(pts)
            if not ((pts[:,0]>=0)&(pts[:,0]<w)&(pts[:,1]>=h*.18)&(pts[:,1]<h)).all(): continue
            center=pts.mean(0); pts=pts[np.argsort(np.arctan2(pts[:,1]-center[1],pts[:,0]-center[0]))]
            if not cv2.isContourConvex(pts.astype(np.float32)): continue
            area=abs(cv2.contourArea(pts.astype(np.float32)))
            if area < w*h*.08: continue
            coverage=[]
            for a,b in zip(pts,np.roll(pts,-1,axis=0)):
                xy=np.rint(np.linspace(a,b,100)).astype(int)
                xy[:,0]=np.clip(xy[:,0],0,w-1); xy[:,1]=np.clip(xy[:,1],0,h-1)
                coverage.append(np.mean(support[xy[:,1],xy[:,0]]>0))
            if min(coverage)<.35: continue
            score=np.mean(coverage)*np.sqrt(area/(w*h))
            if score > best_score:
                # Far-left, far-right, near-left, near-right in screen orientation.
                by_y=pts[np.argsort(pts[:,1])]
                far=by_y[:2][np.argsort(by_y[:2,0])]; near=by_y[2:][np.argsort(by_y[2:,0])]
                if near[:,1].mean()-far[:,1].mean()<h*.08: continue
                best=np.vstack([far,near]).tolist(); best_score=float(score)
    return {'points_px':best,'support_score':best_score,'review_status':'unconfirmed'} if best else None


def validate(record):
    points = np.asarray(record.get('image_points_px'), dtype=float)
    targets = np.asarray(record.get('court_points_normalized'), dtype=float)
    w,h=record['image_size']
    if points.shape!=targets.shape or points.ndim!=2 or points.shape[1]!=2 or not 4<=len(points)<=30:
        raise ValueError('请标记至少4个对应交点')
    if not np.isfinite(points).all() or not np.isfinite(targets).all(): raise ValueError('标定坐标无效')
    if ((points<0).any() or (points[:,0]>=w).any() or (points[:,1]>=h).any()): raise ValueError('请标记画面内可见交点，不要猜测画面外角落')
    known=np.array([p['xy'] for p in template()])
    if any(np.linalg.norm(known-p,axis=1).min()>1e-6 for p in targets): raise ValueError('请选择示意图上的实际场线交点')
    if len(np.unique(targets,axis=0))!=len(targets) or len(np.unique(points,axis=0))!=len(points): raise ValueError('不能重复标记同一个交点')
    if abs(cv2.contourArea(cv2.convexHull(points.astype(np.float32))))<w*h*.003:
        raise ValueError('选点范围过小或共线，请选择分散的交点')
    if abs(cv2.contourArea(cv2.convexHull(targets.astype(np.float32))))<.015:
        raise ValueError('示意图选点共线或范围过小，请跨越两条不同的横线和纵线')
    matrix,_=cv2.findHomography(points,targets,method=0)
    if matrix is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-12:
        raise ValueError('无法计算场地，请检查交点对应关系')
    residual=np.linalg.norm(project(matrix,points)-targets,axis=1)
    if residual.max()>.03: raise ValueError('交点不一致，请检查点位是否对应同一条场线')
    boundary=np.array([[0,0],[1,0],[1,1],[0,1]],dtype=float)
    inv=np.linalg.inv(matrix)
    denom=np.c_[boundary,np.ones(4)]@inv[2]
    if np.min(denom)*np.max(denom)<=0: raise ValueError('标定跨过投影无穷远，请检查点位顺序')
    corners=project(inv,boundary)
    if not cv2.isContourConvex(corners.astype(np.float32)):
        raise ValueError('场地边界交叉，请检查点位顺序')
    result=dict(record, matrix=matrix.tolist(), residual_normalized=residual.tolist(),
                inferred_corners_px=corners[[0,1,3,2]].tolist())
    return result


def main(value):
    if value['operation']=='validate': return validate(value['record'])
    source=Path(value['source_file']); out=Path(value['folder']); out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(source))
    fps=cap.get(cv2.CAP_PROP_FPS); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t=float(value.get('time_seconds',1))
    if not np.isfinite(t) or not 0<=t<=max(0,(total-1)/fps): raise ValueError('预览时间超出录像范围')
    index=min(total-1, int(round(t*fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES,index); ok,frame=cap.read(); cap.release()
    if not ok: raise ValueError('无法读取预览画面，请换一个时间')
    h,w=frame.shape[:2]
    mask=color_mask(frame,value['court_line_color']); mask[:int(h*.28)]=0
    candidate=automatic_corners(mask)
    cv2.imwrite(str(out/'preview.jpg'),frame)
    cv2.imwrite(str(out/'line_mask.png'),mask)
    return {'image_size':[w,h], 'frame_index':index, 'time_seconds':index/fps,
            'court_line_color':value['court_line_color'], 'proposal':candidate,
            'template_points':template(), 'algorithm':'color_hough_two_line_families_v1',
            'review_status':'unconfirmed'}


if __name__=='__main__':
    try:
        result=main(json.load(sys.stdin))
        print(json.dumps({'result':result},ensure_ascii=False,allow_nan=False))
    except (ValueError,KeyError,TypeError,cv2.error) as error:
        print(json.dumps({'error':str(error)},ensure_ascii=False)); sys.exit(1)
