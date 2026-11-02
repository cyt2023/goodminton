"""Label-independent coordinate diagnostics; NumPy only, no model inference.

Flags describe suspicious model output, not verified coordinate errors. They
must not be interpreted as true accuracy or silently erase source evidence.
"""
import numpy as np

VERSION = 'coordinate-diagnostics-v1'
SIDES = ('near', 'far')


def proposed_horizontal_roi(dense):
    """One full-video sensitivity ROI, with a fixed label-independent margin.

    It is deliberately broad, has no claim of reviewed court identity, and
    retains the full image height because shuttle flight extends above players.
    """
    boxes = np.asarray([p['bbox'] for f in dense['frames'] for p in f['players'].values() if p],float)
    if not len(boxes):
        return None
    margin = max(.1*dense['source_width'],.5*float(np.median(boxes[:,3]-boxes[:,1])))
    return {'left':max(0.,float(boxes[:,0].min()-margin)),
        'right':min(float(dense['source_width']),float(boxes[:,2].max()+margin)),
        'top':0.,'bottom':float(dense['source_height']),'margin_pixels':margin,
        'basis':'Full-video min/max of both assigned player boxes; margin=max(.10 image width,.50 median body box height); no stroke labels or example outcome used.',
        'reviewed':False,'runtime_enabled':False}


class InputQuality:
    def __init__(self, dense, trajectory):
        self.dense, self.trajectory = dense, trajectory
        n = len(dense['frames'])
        if len(trajectory['frames']) != n or dense['source_sha256'] != trajectory['source_sha256']:
            raise ValueError('Coordinate sources differ in identity or frame count')
        self.times = np.asarray([f['source_time'] for f in dense['frames']], float)
        self.observed_ball = np.asarray([b['raw_xy'] if b['raw_visible'] and b['raw_xy'] else [np.nan,np.nan]
                                        for b in trajectory['frames']], float)
        self.raw_visible = np.isfinite(self.observed_ball).all(1)
        self.heat = np.asarray([b.get('raw_heatmap_peak',0.) for b in trajectory['frames']], float)
        valid = self.raw_visible & (self.heat >= .15)
        self.stationary = np.zeros(n,bool)
        start = 0
        for i in range(1,n+1):
            boundary = i == n or not valid[i] or not valid[i-1] or np.linalg.norm(self.observed_ball[i]-self.observed_ball[start]) > 4
            if boundary:
                if i-start >= 7 and valid[start:i].all():
                    self.stationary[start:i] = True
                start = i
        self.isolated_jump = np.zeros(n,bool)
        limit = np.hypot(dense['source_width'],dense['source_height'])*.1
        for i in range(1,n-1):
            if valid[i-1:i+2].all():
                self.isolated_jump[i] = (np.linalg.norm(self.observed_ball[i]-self.observed_ball[i-1]) > limit and
                    np.linalg.norm(self.observed_ball[i]-self.observed_ball[i+1]) > limit)
        self.filtered_ball_valid = valid & ~self.stationary & ~self.isolated_jump
        self.points = np.full((n,2,17,2),np.nan)
        self.scores = np.full((n,2,17),np.nan)
        self.boxes = np.full((n,2,4),np.nan)
        self.in_frame = np.zeros((n,2,17),bool)
        for i,f in enumerate(dense['frames']):
            if trajectory['frames'][i]['frame_index'] != f['frame_index']:
                raise ValueError('Coordinate frames are not aligned')
            for k,side in enumerate(SIDES):
                p = f['players'].get(side)
                if p:
                    self.points[i,k] = [[j['x'],j['y']] for j in p['keypoints']]
                    self.scores[i,k] = [j['score'] for j in p['keypoints']]
                    self.in_frame[i,k] = [j.get('in_frame',True) for j in p['keypoints']]
                    self.boxes[i,k] = p['bbox']
        self.present = np.isfinite(self.boxes).all(2)
        hips = self.points[:,:,[11,12]].mean(2)
        diagonals = np.linalg.norm(self.boxes[:,:,2:]-self.boxes[:,:,:2],axis=2)
        self.identity_jump = np.zeros((n,2),bool)
        # A one-frame hip displacement > half the average body-box diagonal is
        # a review flag. Fast genuine motion/camera cuts can also trigger it.
        distance = np.linalg.norm(np.diff(hips,axis=0),axis=2)
        scale = (diagonals[1:]+diagonals[:-1])/2
        self.identity_jump[1:] = (distance > .5*scale) & self.present[1:] & self.present[:-1]

    def window(self, event, *, target_player=None, target_time=None, order=('far','near')):
        a,b = max(0,event['start_frame']),min(len(self.times),event['end_frame']+1)
        if b <= a:
            raise ValueError('Empty diagnostic window')
        scores = self.scores[a:b]
        finite = np.isfinite(scores)
        order_index = [SIDES.index(s) for s in order]
        feet = self.points[a:b,:,[15,16],1].mean(2)
        paired = self.present[a:b].all(1)
        reversed_order = (feet[:,order_index[0]] > feet[:,order_index[1]]) & paired
        distance = []
        if target_player is not None and target_time is not None:
            center = int(np.argmin(abs(self.times-target_time)))
            side = SIDES.index(target_player)
            for i in range(max(0,center-2),min(len(self.times),center+3)):
                p = self.points[i,side]
                good = self.scores[i,side] >= .35
                torso = np.linalg.norm(p[[5,6]].mean(0)-p[[11,12]].mean(0))
                if self.filtered_ball_valid[i] and np.isfinite(torso) and torso > 10:
                    distance.extend(float(np.linalg.norm(self.observed_ball[i]-p[j])/torso) for j in [9,10] if good[j])
        return {'version':VERSION,'frame_count':b-a,
            'paired_player_fraction':float(paired.mean()),
            'raw_visible_fraction':float(self.raw_visible[a:b].mean()),
            'inpaint_requested_fraction':float(np.mean([f['inpaint_requested'] for f in self.trajectory['frames'][a:b]])),
            'filtered_observed_fraction':float(self.filtered_ball_valid[a:b].mean()),
            'stationary_fraction':float(self.stationary[a:b].mean()),
            'isolated_jump_fraction':float(self.isolated_jump[a:b].mean()),
            'low_heatmap_observed_fraction':float((self.raw_visible[a:b] & (self.heat[a:b]<.15)).mean()),
            'removed_observed_frames':int(self.raw_visible[a:b].sum()-self.filtered_ball_valid[a:b].sum()),
            'mean_joint_score':float(scores[finite].mean()) if finite.any() else None,
            'low_score_joint_fraction':float((scores[finite]<.35).mean()) if finite.any() else None,
            'out_of_image_joint_fraction':float((~self.in_frame[a:b][finite]).mean()) if finite.any() else None,
            'identity_jump_frames_by_side':{s:int(self.identity_jump[a:b,k].sum()) for k,s in enumerate(SIDES)},
            'per_frame_ankle_order_reversed_fraction':float(reversed_order.mean()),
            'minimum_observed_ball_to_wrist_torso_units_near_contact':min(distance) if distance else None,
            'coordinate_ground_truth_available':False,
            'flags_are_verified_errors':False}


def diagnose_window(dense, trajectory, event, **kwargs):
    """Convenience API; reuse InputQuality for batches to avoid rebuilding arrays."""
    return InputQuality(dense,trajectory).window(event,**kwargs)
