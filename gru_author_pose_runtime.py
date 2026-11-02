"""Local compatibility layer for the GRU author's legacy OpenMMLab frontend.

Original mmdet2/MMPose0 networks/configs/checkpoints are retained. Only inference
NMS and average RoIAlign use torchvision CPU kernels because legacy mmcv-full
does not ship a matching Apple Silicon/Python3.12 extension. Unsupported native
ops fail if actually called. This layer never changes production imports.
"""
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent
DEPS = ROOT / 'vendor/gru_author_pose_deps'
EXACT_DEPS = ROOT / 'vendor/gru_author_pose_deps_exact'


def activate():
    sys.path.insert(0, str(DEPS))
    sys.path.insert(0, str(EXACT_DEPS))
    import numpy as np
    import torch
    from torchvision.ops import nms, roi_align
    # Old optional mesh dependencies still import NumPy's deprecated aliases.
    for name, value in [('bool', bool), ('int', int), ('float', float),
                        ('complex', complex), ('object', object), ('str', str)]:
        if name not in np.__dict__:
            setattr(np, name, value)
    import inspect
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
    native = types.ModuleType('mmcv._ext')
    native.__file__ = __file__

    def nms_forward(boxes, scores, iou_threshold, offset=0):
        if offset:
            adjusted = boxes.detach().cpu().clone()
            adjusted[:, 2:] += offset
        else:
            adjusted = boxes.detach().cpu()
        return nms(adjusted, scores.detach().cpu(), iou_threshold).to(boxes.device)

    def roi_forward(image, rois, output, argmax_y, argmax_x,
                    aligned_height, aligned_width, spatial_scale,
                    sampling_ratio, pool_mode, aligned):
        if pool_mode != 1:
            raise NotImplementedError('Only original FasterRCNN average RoIAlign is supported')
        value = roi_align(image.cpu(), rois.cpu(),
                          output_size=(aligned_height, aligned_width),
                          spatial_scale=spatial_scale, sampling_ratio=sampling_ratio,
                          aligned=aligned)
        output.copy_(value.to(output.device))

    native.nms = nms_forward
    native.roi_align_forward = roi_forward

    def missing(name):
        if name.startswith('__'):
            raise AttributeError(name)
        def unsupported(*args, **kwargs):
            raise NotImplementedError('Unused MMCV native operator requested: ' + name)
        return unsupported
    native.__getattr__ = missing
    sys.modules['mmcv._ext'] = native
    # The official checkpoints contain metadata requiring the legacy loader.
    # Keep this local to the explicitly launched reproduction process.
    original_load = torch.load
    def legacy_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)
    torch.load = legacy_load
    return {'runtime': 'mmcv1.6.2/mmdet2.25.3/mmpose0.29.0',
            'compatibility': 'Original networks and preprocessing; torchvision CPU NMS/average RoIAlign replace unavailable mmcv native binaries; inference only.'}
