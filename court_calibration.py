"""Ground-plane homography from reviewed correspondences.

Visible Hough segments are inspection evidence, never automatically interpreted
as a complete badminton court in a multi-sport hall. Four or more labelled
correspondences with explicit reviewed status enable ground-coordinate mapping.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from court_settings import line_color, LINE_COLORS


def load_reviewed(file, source_sha256):
    if not Path(file).is_file():
        return None
    value = json.loads(Path(file).read_text())
    if value.get('review_status') != 'confirmed' or value.get('source_sha256') != source_sha256:
        return None
    src, dst = np.asarray(value['image_points_px'], dtype=float), np.asarray(value['court_points_normalized'], dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 4 or not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError('Invalid reviewed court correspondences')
    matrix, _ = cv2.findHomography(src, dst, method=0)
    if matrix is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError('Degenerate court calibration')
    fitted = project(matrix, src)
    residual = np.linalg.norm(fitted - dst, axis=1)
    if residual.max() > .03:
        raise ValueError('Court correspondence residual is too large')
    return {'matrix': matrix, 'record': value, 'residual_normalized': residual.tolist()}


def project(matrix, points):
    return cv2.perspectiveTransform(np.asarray(points, dtype=np.float64).reshape(1, -1, 2), matrix).reshape(-1, 2)


def color_mask(frame, color):
    color = line_color(color)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ranges = {
        'white': [([0, 0, 155], [179, 85, 255])],
        'dark_blue': [([90, 35, 15], [140, 255, 175])],
        'blue': [([90, 50, 25], [140, 255, 255])],
        'yellow': [([18, 65, 65], [40, 255, 255])],
        'red': [([0, 65, 45], [12, 255, 255]), ([165, 65, 45], [179, 255, 255])],
        'black': [([0, 0, 0], [179, 255, 90])],
        'green': [([40, 45, 30], [90, 255, 255])],
    }
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for low, high in ranges.get(color, []):
        mask |= cv2.inRange(hsv, np.array(low), np.array(high))
    if color == 'dark_blue':
        # Dark markings on a blue floor need local darkness as well as hue.
        local_dark = cv2.morphologyEx(hsv[:, :, 2], cv2.MORPH_BLACKHAT,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        mask &= cv2.inRange(local_dark, 8, 255)
    return mask


def inspect_lines(source, folder, source_sha256, court_line_color='unknown'):
    color = line_color(court_line_color)
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError('Cannot inspect the court')
    h, w = frame.shape[:2]
    mask = color_mask(frame, color)
    mask[:int(h * .55)] = 0
    lines = cv2.HoughLinesP(cv2.Canny(mask, 60, 160), 1, np.pi / 180, 45,
                            minLineLength=w * .08, maxLineGap=w * .025)
    segments = [] if lines is None else np.asarray(lines).reshape(-1, 4).tolist()
    image = frame.copy()
    for x1, y1, x2, y2 in segments:
        cv2.line(image, (x1, y1), (x2, y2), (0, 180, 255), 2)
    cv2.imwrite(str(Path(folder) / 'court_line_candidates.jpg'), image)
    cv2.imwrite(str(Path(folder) / 'court_line_mask.png'), mask)
    value = {'source_sha256': source_sha256,
             'court_line_color': color, 'court_line_color_label': LINE_COLORS[color],
             'status': 'awaiting_line_color' if color == 'unknown' else 'line_candidates_only',
             'review_status': 'unconfirmed', 'segments_px': segments,
             'note': 'Multi-sport markings and off-screen court corners are unresolved. No homography or invented court positions are supplied to pretrained weights.'}
    (Path(folder) / 'court_evidence.json').write_text(json.dumps(value, ensure_ascii=False, indent=2))
    return value
