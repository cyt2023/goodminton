"""Local hit features. Pixel kinematics are not calibrated 3-D biomechanics."""
import numpy as np

VERSION = 'local-hit-v1'


class HitFeatures:
    def __init__(self, classifier):
        self.classifier = classifier
        self.dense = classifier.dense
        self.frames = self.dense['frames']
        self.times = np.asarray([f['source_time'] for f in self.frames])
        ball = classifier.trajectory['frames']
        if self.dense['source_sha256'] != classifier.trajectory['source_sha256']:
            raise ValueError('球轨迹与姿态不是同一录像')
        if len(ball) != len(self.frames) or any(f['frame_index'] != b['frame_index'] for f,b in zip(self.frames,ball)):
            raise ValueError('球轨迹与姿态解码帧不对齐')
        self.ball = np.asarray([b['raw_xy'] if b['raw_visible'] and b.get('raw_heatmap_peak', 0) >= .15
                                else [np.nan, np.nan] for b in ball], float)
        valid = np.isfinite(self.ball).all(1)
        self.static = np.zeros(len(ball), bool)
        # Runs of almost identical detected pixels are frequently floor clutter.
        start = 0
        for i in range(1, len(ball)+1):
            boundary = (i == len(ball) or not valid[i] or not valid[i-1]
                        or np.linalg.norm(self.ball[i]-self.ball[start]) > 4)
            if boundary:
                if i-start >= 7 and valid[start:i].all():
                    self.static[start:i] = True
                start = i
        jump = np.zeros(len(ball), bool)
        limit = np.hypot(self.dense['source_width'], self.dense['source_height']) * .1
        for i in range(1, len(ball)-1):
            if valid[i-1:i+2].all():
                jump[i] = (np.linalg.norm(self.ball[i]-self.ball[i-1]) > limit
                           and np.linalg.norm(self.ball[i]-self.ball[i+1]) > limit)
        self.valid = valid & ~self.static & ~jump

    def pose(self, index, side):
        person = self.frames[index]['players'].get(side)
        if person is None:
            return None
        pts = np.asarray([[k['x'], k['y']] for k in person['keypoints']], float)
        good = np.asarray([k['score'] >= .35 for k in person['keypoints']])
        if not good[[5, 6, 11, 12]].all():
            return None
        shoulder, hip = pts[[5, 6]].mean(0), pts[[11, 12]].mean(0)
        torso = np.linalg.norm(shoulder-hip)
        if torso < 10:
            return None
        return pts, good, shoulder, hip, torso, person

    def center(self, event):
        peak = event['peak_time']
        if event.get('contact_confirmed') or event.get('feature_center_locked'):
            return peak, False
        # Limited label-independent adjustment; the original candidate is kept.
        candidates = []
        for i in np.flatnonzero((abs(self.times-peak) <= .25)
                               & (self.times >= event['start']) & (self.times < event['end'])):
            p = self.pose(i, event['player'])
            if p is None or not self.valid[i]:
                continue
            pts, good, _, _, torso, _ = p
            hands = [j for j in [9, 10] if good[j]]
            if hands:
                distance = min(np.linalg.norm(self.ball[i]-pts[j])/torso for j in hands)
                candidates.append((distance + .2*abs(self.times[i]-peak)/.25, i, distance))
        if candidates:
            _, i, distance = min(candidates)
            if distance < 1.7:
                return float(self.times[i]), True
        return peak, False

    def extract(self, event):
        center, refined = self.center(event)
        # This evidence support is explicit and bounded by the reviewed window.
        lo, hi = max(event['start'], center-.45), min(event['end'], center+.45)
        indices = np.flatnonzero((self.times >= lo) & (self.times <= hi))
        pose_rows, pose_times, scales, shoulders, hips = [], [], [], [], []
        near = int(np.argmin(abs(self.times-center)))
        target = self.pose(near, event['player'])
        other_side = 'far' if event['player'] == 'near' else 'near'
        opponent = self.pose(near, other_side)
        facing = 1 if target is None or opponent is None or opponent[3][0] >= target[3][0] else -1
        pair_count = 0
        for i in indices:
            p = self.pose(i, event['player'])
            pair_count += self.pose(i, other_side) is not None
            if p is None:
                continue
            pts, good, shoulder, hip, torso, _ = p
            row = []
            for j in [7, 8, 9, 10, 13, 14, 15, 16]:
                xy = (pts[j]-shoulder)/torso if good[j] else np.zeros(2)
                row.extend([xy[0]*facing, -xy[1]])
            row.extend([(shoulder[0]-hip[0])*facing/torso, (hip[1]-shoulder[1])/torso])
            pose_rows.append(row);pose_times.append(self.times[i]);scales.append(torso)
            shoulders.append(shoulder);hips.append(hip)
        pose_rows = np.asarray(pose_rows, float).reshape(-1, 18)
        pose_times = np.asarray(pose_times)
        geometry = []
        for a, b in [(-.45, -.15), (-.15, .15), (.15, .45)]:
            rows = pose_rows[(pose_times >= center+a) & (pose_times < center+b)]
            geometry.extend(np.median(rows, axis=0) if len(rows) else np.zeros(18))
        if len(pose_rows) >= 3:
            motion = np.diff(pose_rows[:, :8], axis=0)/np.maximum(np.diff(pose_times)[:, None], .01)
            geometry.extend(np.quantile(abs(motion), .75, axis=0))
        else:
            geometry.extend(np.zeros(8))
        torso = float(np.median(scales)) if scales else 50.
        shoulder = np.median(shoulders, axis=0) if shoulders else np.zeros(2)
        trajectory_features = []
        velocities = []
        for a, b in [(-.4, -.05), (.05, .4)]:
            sample = indices[(self.times[indices] >= center+a) & (self.times[indices] <= center+b)
                             & self.valid[indices]]
            velocity = np.zeros(2)
            if len(sample) >= 3:
                # Median pairwise local slopes limit influence of a bad pixel.
                slopes = np.diff(self.ball[sample], axis=0)/np.maximum(np.diff(self.times[sample])[:, None], .01)
                velocity = np.median(slopes, axis=0)/torso
            velocities.append(velocity)
            coords = (np.median(self.ball[sample], axis=0)-shoulder)/torso if len(sample) else np.zeros(2)
            trajectory_features.extend([coords[0]*facing, -coords[1], velocity[0]*facing, -velocity[1],
                                        np.linalg.norm(velocity), len(sample)/max(1,len(indices))])
        denom = np.linalg.norm(velocities[0])*np.linalg.norm(velocities[1])
        trajectory_features.append(float(np.dot(*velocities)/denom) if denom > 1e-6 else 0.)
        distances = []
        for i in indices:
            p = self.pose(i, event['player'])
            if p is None or not self.valid[i]:continue
            pts, good, _, _, scale, _ = p
            distances.extend(np.linalg.norm(self.ball[i]-pts[j])/scale for j in [9, 10] if good[j])
        trajectory_features.extend([min(distances) if distances else 5.,
                                    float(self.valid[indices].mean()) if len(indices) else 0.,
                                    float(self.static[indices].mean()) if len(indices) else 0.])
        geometry.extend(trajectory_features)
        target_fraction = len(pose_rows)/max(1,len(indices))
        geometry.extend([target_fraction, pair_count/max(1,len(indices))])
        # BST representation uses independent pose masks and observed filtered
        # shuttle only. Missing opponent no longer destroys the target pose.
        joints, shuttle = [], []
        for i in indices:
            normalized = np.zeros((2,17,2))
            for k, side in enumerate(self.classifier.order):
                p = self.pose(i, side)
                if p is None:continue
                pts, good, _, _, _, person = p
                norm = self.classifier.helpers['normalize_joints'](pts[None], np.asarray([person['bbox']]), center_align=True)[0]
                norm[~good] = 0
                normalized[k] = norm
            joints.append(normalized)
            shuttle.append(self.ball[i] if self.valid[i] else [0,0])
        if not len(indices):
            raise ValueError('本次击球时间窗口没有解码画面')
        h = self.classifier.helpers
        s = h['normalize_shuttlecock'](np.asarray(shuttle), self.dense['source_width'], self.dense['source_height'])
        j, p, s, length = h['make_seq_len_same'](100, np.asarray(joints), np.zeros((len(indices),2,2)), s)
        j = np.concatenate([j,h['create_bones'](j,self.classifier.bones)],axis=-2).reshape(100,2,72)
        from personal_learning import embedding
        bst = embedding(self.classifier, (j.astype('float32'),p.astype('float32'),s.astype('float32'),length), event['player'])
        vector = np.r_[bst, np.clip(np.asarray(geometry),-30,30)].astype('float32')
        quality = {'feature_version': VERSION, 'original_peak_time': event['peak_time'],
                   'feature_center_time': center, 'center_refined_automatically': refined,
                   'feature_start': float(self.times[indices[0]]), 'feature_end': float(self.times[indices[-1]]),
                   'target_pose_fraction': target_fraction,
                   'opponent_pose_fraction': pair_count/max(1,len(indices)),
                   'filtered_raw_shuttle_fraction': float(self.valid[indices].mean()),
                   'stationary_shuttle_fraction': float(self.static[indices].mean()),
                   'geometry_dimension': len(geometry), 'bst_dimension': 300,
                   'calibrated_court_or_3d_height': False}
        return vector, quality
