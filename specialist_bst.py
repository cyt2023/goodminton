"""Official BST-0 adapter with author normalisation/bone/sequence functions.

No court positions are fed to BST-0, which is the author's position-free
backbone. The full BST-CG-AP runs only with reviewed court calibration. Candidate
player order for the backbone uses the median image ankle height across video;
this assumption is recorded and remains uncertain for oblique footage.
"""
import ast
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

from court_calibration import project
from specialist_model import ROOT, BST_NAME, FULL_BST_NAME

VENDOR = ROOT / 'vendor/BST/stroke_classification'
sys.path.insert(0, str(VENDOR))
from model.bst import BST_0, BST_CG_AP


def author_helpers():
    namespace = {'np': np}
    sources = [('preparing_data/shuttleset_dataset.py', {'get_merged_stroke_types', 'get_bone_pairs', 'make_seq_len_same', 'create_bones'}),
               ('preparing_data/prepare_train_on_shuttleset.py', {'normalize_joints', 'normalize_shuttlecock'})]
    for source, names in sources:
        path = VENDOR / source
        nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class SpecialistBST:
    def __init__(self, dense, trajectory, calibration=None, checkpoint=None):
        self.dense, self.trajectory, self.calibration = dense, trajectory, calibration
        self.helpers = author_helpers()
        self.classes = self.helpers['get_merged_stroke_types']()
        self.bones = self.helpers['get_bone_pairs']()
        ankle_heights = {}
        for side in ['near', 'far']:
            ankle_heights[side] = np.median([np.mean([p['keypoints'][j]['y'] for j in [15, 16]])
                                          for f in dense['frames'] if (p := f['players'].get(side)) is not None])
        self.order = sorted(ankle_heights, key=ankle_heights.get)
        if calibration is not None:
            positions = {}
            for side in self.order:
                coords = [project(calibration['matrix'], [[p['keypoints'][j]['x'], p['keypoints'][j]['y']] for j in [15, 16]]).mean(0)[1]
                          for f in dense['frames'] if (p := f['players'].get(side)) is not None]
                positions[side] = np.median(coords)
            self.order = sorted(positions, key=positions.get)
        self.model = BST_0(in_dim=72, seq_len=100, n_class=25, depth_tem=2, depth_inter=1).eval()
        self.checkpoint = Path(checkpoint) if checkpoint is not None else ROOT / 'models/bst' / BST_NAME
        self.model.load_state_dict(torch.load(self.checkpoint, map_location='cpu', weights_only=True), strict=True)
        self.full_model = None
        if calibration and checkpoint is None:
            self.full_model = BST_CG_AP(in_dim=72, seq_len=100, n_class=25, depth_tem=2, depth_inter=1).eval()
            self.full_model.load_state_dict(torch.load(ROOT / 'models/bst' / FULL_BST_NAME, map_location='cpu', weights_only=True), strict=True)
        from input_quality import InputQuality
        self.coordinate_quality = InputQuality(dense, trajectory)

    def inputs(self, event):
        joints, positions, shuttle, present = [], [], [], []
        for index in range(event['start_frame'], event['end_frame'] + 1):
            f = self.dense['frames'][index]
            people = [f['players'].get(side) for side in self.order]
            paired = all(p is not None for p in people)
            present.append(paired)
            if paired:
                pts = np.asarray([[[k['x'], k['y']] for k in p['keypoints']] for p in people], dtype=float)
                boxes = np.asarray([p['bbox'] for p in people])
                normal = self.helpers['normalize_joints'](pts, boxes, center_align=True)
                joints.append(normal)
                if self.calibration:
                    feet = project(self.calibration['matrix'], pts[:, [15, 16]].reshape(-1, 2)).reshape(2, 2, 2).mean(1)
                    positions.append(feet)
                else:
                    positions.append(np.zeros((2, 2)))
            else:
                # Matches the author's failed-pair representation; separately
                # recorded missing mask and coverage prevent invented evidence.
                joints.append(np.zeros((2, 17, 2)))
                positions.append(np.zeros((2, 2)))
            b = self.trajectory['frames'][index]
            xy = b['rectified_xy'] if b['inpaint_requested'] else b['raw_xy']
            shuttle.append(xy or [0, 0])
        joints, positions, shuttle = map(lambda a: np.asarray(a, dtype=np.float32), [joints, positions, shuttle])
        shuttle = self.helpers['normalize_shuttlecock'](shuttle, self.dense['source_width'], self.dense['source_height'])
        j, p, s, length = self.helpers['make_seq_len_same'](100, joints, positions, shuttle)
        bones = self.helpers['create_bones'](j, self.bones)
        j = np.concatenate((j, bones), axis=-2).reshape(100, 2, 72)
        return (j.astype('float32'), p.astype('float32'), s.astype('float32'), int(length)), {
            'paired_player_fraction': float(np.mean(present)),
            'raw_shuttle_fraction': float(np.mean([b['raw_visible'] for b in self.trajectory['frames'][event['start_frame']:event['end_frame']+1]])),
            'missing_player_pair_mask': [not v for v in present],
            'player_input_order': self.order,
            'player_order_basis': 'reviewed_court_y' if self.calibration else 'whole_video_median_image_ankle_y_unconfirmed',
            'normalization': 'author_bbox_diagonal_center_align_true; shuttle/image_resolution; author_bones; author_seq100_pad_and_subsample',
            'court_positions_used_by_bst0': False,
            'pose_extractor_adaptation': 'project RTMLib RTMPose/YOLOX ONNX rather than author MMPose frontend',
            'trajectory_adaptation': 'qaz812345 TrackNetV3 nonoverlap rather than paper attention TrackNetV3',
            'coordinate_diagnostics': self.coordinate_quality.window(
                event, target_player=event.get('player'), target_time=event.get('peak_time'), order=self.order),
        }

    def predict(self, values):
        j, p, s, length = values
        j, p, s = [torch.from_numpy(a).unsqueeze(0) for a in [j, p, s]]
        length = torch.tensor([length])
        with torch.inference_mode():
            logits = self.model(j, s, length)
            scores = logits.softmax(-1)[0].numpy()
            full_scores = self.full_model(j, s, p, length).softmax(-1)[0].numpy() if self.full_model else None
        def ranking(a):
            return [{'label': self.classes[int(i)], 'class_id': int(i), 'softmax_score': float(a[i])}
                    for i in np.argsort(-a)[:5]]
        return {'bst0': ranking(scores), 'bst_full': ranking(full_scores) if full_scores is not None else None,
                'bst0_all_scores': scores.tolist(), 'class_labels': self.classes,
                'bst_full_all_scores': full_scores.tolist() if full_scores is not None else None,
                'score_note': 'Uncalibrated softmax of pretrained model on adapted inputs, not correctness probability.'}
