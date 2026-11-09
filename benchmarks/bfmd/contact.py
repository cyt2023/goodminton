"""Contact-time model: which frame is the racket-shuttle contact?

Per-frame evidence for a rally clip (shuttle track from TrackNetV3 and RacketVision, both players'
wrists and bodies relative to the shuttle, racket heads, impact sound, how many detector routes fire),
and a small temporal convolutional network that scores every frame. Trained on BFMD reference contacts
of training matches only; at a logged hit it ranks the frames within +-W of the logged frame.
"""
import json
from pathlib import Path
import numpy as np
from benchmarks.bfmd.verify import STUDY, ROUTES, match_lag, read
from benchmarks.bfmd.refine import track, turn
from benchmarks.bfmd import audio

W = 9        # frames on each side of a logged hit that can be the contact
C = 24       # temporal context on each side fed to the network
_FRAMES = {}


def _xy(traj, w, h):
    xy = track(traj); xy[:, 0] /= w; xy[:, 1] /= h; return xy


def frame_features(clip):
    """(n_frames, d) per-frame evidence; missing values are 0 with a presence flag."""
    k = clip['clip_id']
    if k in _FRAMES: return _FRAMES[k]
    pf = STUDY / 'perception' / k
    dense = read(pf / 'dense_pose.json'); W_, H_ = dense['source_width'], dense['source_height']
    tn = read(pf / 'tracknetv3/trajectory.json'); xy_px = track(tn); xy = _xy(tn, W_, H_)
    rvj = pf / 'racketvision/trajectory.json'; rv = _xy(read(rvj), W_, H_) if rvj.exists() else np.full_like(xy, np.nan)
    heat = np.array([f.get('raw_heatmap_peak') or 0. for f in tn['frames']])
    rk = read(pf / 'racketvision/racketvision.json')['frames']
    t_a, fl = audio.flux(clip['path']); lag = match_lag(clip); base = np.median(np.log1p(fl))
    votes = np.zeros(len(xy) + 2)
    for p in (ROUTES / k).glob('*__*.json'):
        if p.stem.startswith('agent'): continue
        d = read(p)
        if d['status'] != 'completed': continue
        for hit in d['hits']:
            g = round(hit['time'] * 30)
            if 0 <= g < len(votes): votes[g] += 1
    n = min(len(xy), len(dense['frames'])); F = []
    for t in range(n):
        p = xy[t]; ok = not np.isnan(p).any()
        v = xy[t] - xy[t - 1] if t and ok and not np.isnan(xy[t - 1]).any() else np.zeros(2)
        v0 = xy[t - 1] - xy[t - 2] if t > 1 and not np.isnan(xy[t - 1]).any() and not np.isnan(xy[t - 2]).any() else v
        row = [float(ok), *(p if ok else [0, 0]), *v, float(np.linalg.norm(v)), float(np.linalg.norm(v - v0)),
               turn(xy_px, t) if 2 <= t < n - 2 else 0., heat[t]]
        q = rv[t]; okr = not np.isnan(q).any(); row += [float(okr), float(np.linalg.norm(q - p)) if okr and ok else 0.]
        for side in ('near', 'far'):
            pl = dense['frames'][t]['players'].get(side)
            if pl and ok:
                b = pl['bbox']; bh = max(b[3] - b[1], 1.) / H_; kp = pl['keypoints']
                wr = min(np.hypot(kp[j]['x'] / W_ - p[0], kp[j]['y'] / H_ - p[1]) for j in (9, 10)) / bh
                row += [1., float(np.log1p(wr)), (kp[0]['y'] / H_ - p[1]) / bh, (p[0] - (b[0] + b[2]) / 2 / W_) / bh]
            else: row += [0., 0., 0., 0.]
        best = 9.
        if ok and t < len(rk):
            for r in rk[t]['rackets']:
                kpr = np.array(r['keypoints']); bb = np.array(r['bbox'][0])
                best = min(best, float(np.linalg.norm(xy_px[t] - kpr[[0, 3, 4]].mean(0)) / (np.hypot(bb[2] - bb[0], bb[3] - bb[1]) + 1e-6)))
        m = (t_a >= t / 30 + lag - 1 / 60) & (t_a < t / 30 + lag + 1 / 60)
        row += [float(np.log1p(best)), float(np.log1p(fl[m].max()) - base) if m.any() else 0., votes[t] / 12]
        F.append(row)
    _FRAMES[k] = np.nan_to_num(np.array(F, np.float32)); return _FRAMES[k]


def window(F, c):
    """Frames c-C-W .. c+C+W (padded with zeros) so every candidate frame c-W..c+W has full context."""
    L = C + W; out = np.zeros((2 * L + 1, F.shape[1]), np.float32)
    lo, hi = max(0, c - L), min(len(F), c + L + 1); out[lo - (c - L):hi - (c - L)] = F[lo:hi]; return out


def reference_contacts(clip):
    """Every annotated contact (serves included) in clip frames."""
    m = next(x for x in read(STUDY.parent / 'manifest.json')['matches'] if x['match_id'] == clip['match_id'])
    lo = clip['offset_seconds']; hi = lo + clip['duration_seconds']
    return [round((h['time'] - lo) * 30) for h in m['hits'] if lo <= h['time'] < hi]


class Model:
    """Temporal CNN scoring each frame; softmax over the 2W+1 frames around a centre frame."""
    def __init__(self, d):
        import torch, torch.nn as nn
        self.torch = torch
        self.net = nn.Sequential(nn.Conv1d(d, 48, 5, padding=2), nn.ReLU(), nn.Conv1d(48, 48, 5, padding=4, dilation=2), nn.ReLU(),
                                 nn.Conv1d(48, 48, 5, padding=8, dilation=4), nn.ReLU(), nn.Dropout(.2), nn.Conv1d(48, 1, 1))
        self.prior = np.zeros(2 * W + 1)

    def logits(self, X):
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(((X - self.mu) / self.sd).astype(np.float32)).transpose(1, 2)
            self.net.eval(); return self.net(x)[:, 0, C:C + 2 * W + 1].numpy()

    def fit(self, X, y, epochs=30, seed=0):
        torch = self.torch; torch.manual_seed(seed); rng = np.random.default_rng(seed)
        flat = X.reshape(-1, X.shape[2]); self.mu = flat.mean(0); self.sd = flat.std(0) + 1e-6
        x = torch.from_numpy(((X - self.mu) / self.sd).astype(np.float32)).transpose(1, 2); t = torch.from_numpy(y.astype(np.int64))
        opt = torch.optim.Adam(self.net.parameters(), 2e-3, weight_decay=1e-4)
        for _ in range(epochs):
            self.net.train()
            for b in np.array_split(rng.permutation(len(y)), max(1, len(y) // 128)):
                out = self.net(x[b])[:, 0, C:C + 2 * W + 1]; loss = torch.nn.functional.cross_entropy(out, t[b])
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def fit_prior(self, X, y, steps=300, lr=.5):
        """Offset bias on top of the network (detectors fire at a typical offset from the contact), from logged hits of training clips."""
        S = self.logits(X); b = np.zeros(2 * W + 1)
        for _ in range(steps):
            Z = S + b; P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
            g = P.mean(0) - np.bincount(y, minlength=2 * W + 1) / len(y); b -= lr * g
        self.prior = b; return self

    def scores(self, X): return self.logits(X) + self.prior


def training_windows(clips, offsets=range(-6, 7)):
    X, y = [], []
    for c in clips:
        F = frame_features(c)
        for g in reference_contacts(c):
            if not 0 <= g < len(F): continue
            for o in offsets:
                X.append(window(F, g + o)); y.append(W - o)
    return np.stack(X), np.array(y)
