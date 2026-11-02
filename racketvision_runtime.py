"""Isolated OpenMMLab 3 runtime for the original RacketVision networks.

MMCV's inference NMS uses torchvision CPU on macOS. Other unavailable native
operators raise if called. No global packages or legacy reproduction paths change.
"""
from pathlib import Path
import sys
import types
from importlib.machinery import ModuleSpec

ROOT=Path(__file__).resolve().parent
DEPS=ROOT/'vendor/racketvision_deps'


def activate():
    sys.path.insert(0,str(DEPS))
    import numpy as np
    import inspect
    for name,value in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('str',str)]:
        if name not in np.__dict__:setattr(np,name,value)
    if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
    import torch
    from torchvision.ops import nms
    native=types.ModuleType('mmcv._ext');native.__file__=__file__
    native.__spec__=ModuleSpec('mmcv._ext',loader=None)
    def nms_forward(boxes,scores,iou_threshold,offset=0):
        adjusted=boxes.detach().cpu().clone()
        if offset:adjusted[:,2:]+=offset
        return nms(adjusted,scores.detach().cpu(),iou_threshold).to(boxes.device)
    native.nms=nms_forward
    def missing(name):
        if name.startswith('__'):raise AttributeError(name)
        def unsupported(*args,**kwargs):raise NotImplementedError('Unavailable MMCV native operator: '+name)
        return unsupported
    native.__getattr__=missing
    sys.modules['mmcv._ext']=native
    # MMEngine 0.10 lists CPU/CUDA index classes but omits MPS tensors.
    from mmengine.structures import InstanceData
    original_getitem=InstanceData.__getitem__
    def getitem(self,item):
        if isinstance(item,torch.Tensor) and item.device.type=='mps':item=item.cpu()
        return original_getitem(self,item)
    InstanceData.__getitem__=getitem
    # Public official checkpoints carry their authors' NumPy metadata.
    original_load=torch.load
    def load(*args,**kwargs):
        kwargs.setdefault('weights_only',False)
        return original_load(*args,**kwargs)
    torch.load=load
    return {'mmengine':'0.10.7','mmcv':'2.1.0-lite','mmdet':'3.2.0','mmpose':'1.3.2',
            'native_adaptation':'MMCV NMS via torchvision CPU; MPS indices transferred to CPU for MMEngine InstanceData; unsupported native ops fail explicitly'}
