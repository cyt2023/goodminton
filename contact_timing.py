"""Label-independent source-frame contact features and conservative reranking.

The small ridge scorer is fitted separately by the evaluation tool. Human
annotations are never read during feature extraction or runtime prediction.
"""
from pathlib import Path
import numpy as np

VERSION = 'contact-frame-rerank-v1'
SIDES = ['near', 'far']
LAGS = [-4, -2, 0, 2, 4]
SEARCH_SECONDS = .75


def frame_features(dense, trajectory, supports):
    from stroke_features import HitFeatures
    from types import SimpleNamespace
    track = HitFeatures(SimpleNamespace(dense=dense, trajectory=trajectory))
    n = len(dense['frames'])
    features = np.zeros((n, 2, 11), float)
    for i in range(n):
        for k, side in enumerate(SIDES):
            features[i, k, :5] = [supports['visual'][i, k], supports['gru'][i, k],
                supports['gru'][i, 1-k], supports['audio'][i], float(track.valid[i])]
            person = track.pose(i, side)
            if person is None:
                continue
            pts, good, _, _, torso, _ = person
            hands = [j for j in [9, 10] if good[j]]
            features[i, k, 5] = 1
            if hands and track.valid[i]:
                distance = min(np.linalg.norm(track.ball[i]-pts[j])/torso for j in hands)
                features[i, k, 6] = max(0, 1-distance/2)
            if 2 <= i < n-2:
                previous, following = track.pose(i-2, side), track.pose(i+2, side)
                if previous and following:
                    pp, pg = previous[:2]; fp, fg = following[:2]
                    valid = [j for j in hands if pg[j] and fg[j]]
                    if valid:
                        velocity = max(np.linalg.norm(fp[j]-pp[j])/torso/4 for j in valid)
                        acceleration = max(np.linalg.norm(fp[j]+pp[j]-2*pts[j])/torso/4 for j in valid)
                        features[i, k, 7:9] = [min(velocity, 1), min(acceleration, 1)]
            if track.valid[i] and 3 <= i < n-3 and track.valid[i-3] and track.valid[i+3]:
                incoming, outgoing = track.ball[i]-track.ball[i-3], track.ball[i+3]-track.ball[i]
                denominator = np.linalg.norm(incoming)*np.linalg.norm(outgoing)
                if denominator > 25:
                    features[i, k, 9] = np.clip((1-np.dot(incoming, outgoing)/denominator)/2, 0, 1)
                    features[i, k, 10] = min(1, np.linalg.norm(outgoing-incoming)/torso/3)
    indices = np.clip(np.arange(n)[:, None]+np.asarray(LAGS), 0, n-1)
    return np.concatenate([features[indices[:, q]] for q in range(len(LAGS))], axis=-1)


def event_inputs(times, features, event):
    original = int(np.argmin(abs(times-event['peak_time'])))
    candidates = np.flatnonzero(abs(times-times[original]) <= SEARCH_SECONDS)
    k = SIDES.index(event['player'])
    delta = (times[candidates]-times[original])/SEARCH_SECONDS
    values = np.column_stack([features[candidates, k], delta, abs(delta), delta**2])
    return original, candidates, values


def fit_ridge(x, target, weights, alpha=100.):
    weights = np.asarray(weights, float)
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale = np.maximum(scale, .05)
    z = np.column_stack([np.clip((x-mean)/scale, -8, 8), np.ones(len(x))])
    penalty = np.eye(z.shape[1])*alpha; penalty[-1, -1] = 1e-6
    coefficients = np.linalg.solve(z.T@(weights[:, None]*z)+penalty, z.T@(weights*target))
    return {'mean': mean, 'scale': scale, 'coefficients': coefficients}


def score(model, x):
    z = np.column_stack([np.clip((x-model['mean'])/model['scale'], -8, 8), np.ones(len(x))])
    return z@model['coefficients']


def choose_frame(times, features, event, model, minimum_gain=.08, maximum_move=.3):
    original, candidates, x = event_inputs(times, features, event)
    scores = score(model, x)
    best = int(np.argmax(scores)); chosen = int(candidates[best])
    original_score = float(scores[np.flatnonzero(candidates == original)[0]])
    gain = float(scores[best]-original_score)
    # An extreme window maximum indicates a possibly missing true contact.
    boundary = best in [0, len(candidates)-1]
    k = SIDES.index(event['player'])
    # Center lag features begin at 2*11. A learnt wrist-motion maximum alone
    # must not move a contact: require observed ball evidence or GRU support.
    center_features = features[chosen, k, 22:33]
    independent_support = bool(center_features[5] and
                               (center_features[0] >= .55 or center_features[1] >= .5))
    applied = (not boundary and gain >= minimum_gain and independent_support and
               abs(times[chosen]-times[original]) <= maximum_move)
    return {'original_frame': original, 'alternative_frame': chosen,
            'suggested_frame': chosen if applied else original,
            'gain': gain, 'boundary': boundary, 'independent_support': independent_support,
            'applied': bool(applied and chosen != original),
            'maximum_move_seconds': maximum_move, 'minimum_gain': minimum_gain,
            'score_is_correctness_probability': False}


def save_model(path, model):
    np.savez_compressed(Path(path), **model)


def load_model(path):
    with np.load(path, allow_pickle=False) as value:
        return {k: value[k] for k in ['mean', 'scale', 'coefficients']}
