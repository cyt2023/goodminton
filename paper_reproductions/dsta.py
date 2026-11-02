"""DSTA Eqs.1–2 + the actual upstream AdaTAD VideoMAE/ActionFormer modules.

DSTA's N, adapter insertion indices and ActionFormer settings are not fully
specified. Configurable defaults come from AdaTAD, with ambiguity documented.
The backbone is initialized only from a verified VideoMAE-B foundation checkpoint.
"""
from dataclasses import asdict,dataclass
from functools import lru_cache
from pathlib import Path
import ast
import copy
import importlib.util
import sys
import types
import typing
import math
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]


class DST(nn.Module):
    def __init__(self,channels,alpha=.5,kernel_size=3):
        super().__init__();self.split=int(channels*alpha)
        if not 0<self.split<channels:raise ValueError('Channel split must leave both streams nonempty')
        self.dwconv=nn.Conv3d(channels,channels,(kernel_size,1,1),padding=(kernel_size//2,0,0),groups=channels)
        n=self.split
        self.temporal=nn.Conv3d(n,n,(3,1,1),padding=(1,0,0))
        self.vertical=nn.Conv3d(n,n,(1,3,1),padding=(0,1,0))
        self.horizontal=nn.Conv3d(n,n,(1,1,3),padding=(0,0,1))
    def forward(self,x):
        first,identity=x[:,:self.split],x[:,self.split:]
        directional=self.temporal(first)+self.vertical(first)+self.horizontal(first)
        return torch.cat([directional,identity],1)+self.dwconv(x)


class DSTA(nn.Module):
    def __init__(self,embed_dims,mlp_ratio=.25,kernel_size=3,dilation=1,temporal_size=384,alpha=.5):
        super().__init__();n=int(embed_dims*mlp_ratio);self.temporal_size=temporal_size
        self.down_proj=nn.Linear(embed_dims,n);self.act=nn.GELU();self.dst=DST(n,alpha,kernel_size)
        self.mid_proj=nn.Linear(n,n);self.up_proj=nn.Linear(n,embed_dims);self.beta=nn.Parameter(torch.ones(1))
        nn.init.zeros_(self.up_proj.weight);nn.init.zeros_(self.up_proj.bias)
    def forward(self,x,h,w):
        residual=x;z=self.act(self.down_proj(x));shape=z.shape
        if shape[0]*shape[1]%(self.temporal_size*h*w):raise ValueError('Chunk/token order is incompatible with full temporal extent')
        z5=z.reshape(-1,self.temporal_size,h,w,shape[-1]).permute(0,4,1,2,3)
        z1=self.dst(z5).permute(0,2,3,4,1).reshape(shape)
        return residual+self.beta*self.up_proj(self.mid_proj(z1)+z)


@dataclass
class Config:
    classes:int=29
    input_size:int=160
    fps:int=25
    sample_stride:int=4
    window_frames:int=768
    feature_length:int=768
    attention_clip_frames:int=48  # Literal reported chunk size; AdaTAD uses16.
    adapter_ratio:float=.25
    adapter_indices:tuple=tuple(range(12))
    alpha:float=.5
    head_dim:int=512
    head_max_seq:int=2304
    head_window:int=19
    with_checkpoint:bool=True
    drop_path:float=.1


def _load_file(name,file):
    spec=importlib.util.spec_from_file_location(name,file);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def _sinusoid(n,d):
    # Same closed form as MMAction2's get_sinusoid_encoding; not learned weights.
    positions=torch.arange(n,dtype=torch.float32)[:,None]
    dimensions=torch.arange(d,dtype=torch.float32)[None,:]
    angle=positions/torch.pow(10000.,2*torch.floor(dimensions/2)/d)
    table=torch.empty_like(angle);table[:,0::2]=angle[:,0::2].sin();table[:,1::2]=angle[:,1::2].cos()
    return table.unsqueeze(0)


@lru_cache(maxsize=1)
def load_videomae_upstream():
    from racketvision_runtime import activate
    activate()
    import mmengine
    file=ROOT/'vendor/OpenTAD/opentad/models/backbones/vit_adapter.py'
    tree=ast.parse(file.read_text())
    # Only MMAction type aliases and the deterministic sinusoid helper are
    # supplied locally; every backbone/attention/FFN forward comes from OpenTAD.
    tree.body=[node for node in tree.body if not (isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('mmaction'))]
    namespace={'__name__':'local_dsta_videomae','__file__':str(file),'ConfigType':typing.Union[dict,mmengine.ConfigDict],
               'OptConfigType':typing.Optional[typing.Union[dict,mmengine.ConfigDict]],'get_sinusoid_encoding':_sinusoid}
    exec(compile(tree,str(file),'exec'),namespace)
    namespace['Adapter']=DSTA
    return namespace['VisionTransformerAdapter']


def _native_nms_compat():
    native=types.ModuleType('nms_1d_cpu')
    def overlap(a,b):
        intersection=(torch.minimum(a[1],b[:,1])-torch.maximum(a[0],b[:,0])).clamp(min=0)
        # Match upstream C++: each segment area includes1e-6.
        return intersection/((a[1]-a[0])+(b[:,1]-b[:,0])-intersection+2e-6).clamp(min=1e-8)
    def nms(segs,scores,iou_threshold):
        order=scores.argsort(descending=True);keep=[]
        while order.numel():
            i=order[0];keep.append(i)
            order=order[1:][overlap(segs[i],segs[order[1:]])<=iou_threshold]
        return torch.stack(keep).long() if keep else torch.empty(0,dtype=torch.long)
    def softnms(segs,scores,dets,iou_threshold,sigma,min_score,method):
        pending=torch.arange(len(scores));scores=scores.clone();indices=[]
        while pending.numel():
            k=scores[pending].argmax();idx=pending[k];score=scores[idx]
            if score<min_score:break
            dets[len(indices),:2]=segs[idx];dets[len(indices),2]=score;indices.append(idx)
            pending=torch.cat([pending[:k],pending[k+1:]])
            if pending.numel():
                iou=overlap(segs[idx],segs[pending])
                decay=torch.exp(-iou*iou/sigma) if method==2 else torch.where(iou>iou_threshold,1-iou if method==1 else torch.zeros_like(iou),torch.ones_like(iou))
                scores[pending]*=decay
        return torch.stack(indices).long() if indices else torch.empty(0,dtype=torch.long)
    native.nms=nms;native.softnms=softnms;sys.modules['nms_1d_cpu']=native


@lru_cache(maxsize=1)
def load_actionformer_upstream():
    _native_nms_compat();root=ROOT/'vendor/actionformer_release/libs'
    package=types.ModuleType('paper_actionformer');package.__path__=[str(root)];sys.modules[package.__name__]=package
    utils=types.ModuleType('paper_actionformer.utils');utils.__path__=[str(root/'utils')];sys.modules[utils.__name__]=utils
    nms=_load_file('paper_actionformer.utils.nms',root/'utils/nms.py');utils.batched_nms=nms.batched_nms
    modeling=_load_file('paper_actionformer.modeling',root/'modeling/__init__.py')
    config=_load_file('paper_actionformer_defaults',root/'core/config.py')
    return modeling,config.DEFAULTS


def _differentiable_preprocessing(self,videos,padding_val=0.):
    # Upstream ActionFormer assumes precomputed features and marks this method
    # no_grad. Keep its exact padding/masks but preserve the DSTA RGB gradient.
    feats=[v['feats'] for v in videos];length=max(x.shape[-1] for x in feats)
    if self.training:
        if length>self.max_seq_len:raise ValueError('Feature sequence exceeds ActionFormer training extent')
        length=self.max_seq_len
    else:length=max(self.max_seq_len,math.ceil(length/self.max_div_factor)*self.max_div_factor)
    padded=torch.stack([F.pad(x,(0,length-x.shape[-1]),value=padding_val) for x in feats])
    mask=torch.arange(length,device=padded.device)[None,:]<torch.tensor([x.shape[-1] for x in feats],device=padded.device)[:,None]
    return padded,mask[:,None]


class DSTALocalizer(nn.Module):
    def __init__(self,config=None,foundation_checkpoint=None):
        super().__init__();self.config=config or Config();c=self.config
        if c.attention_clip_frames%2 or c.window_frames%c.attention_clip_frames:raise ValueError('Video window must contain full even-length attention chunks')
        backbone_class=load_videomae_upstream()
        self.backbone=backbone_class(img_size=224,patch_size=16,embed_dims=768,depth=12,num_heads=12,
            num_frames=c.attention_clip_frames,total_frames=c.window_frames,adapter_mlp_ratio=c.adapter_ratio,
            adapter_index=list(c.adapter_indices),return_feat_map=True,with_cp=c.with_checkpoint,drop_path_rate=c.drop_path)
        for block in self.backbone.blocks:
            if hasattr(block,'adapter') and c.alpha!=.5:block.adapter.dst=DST(int(768*c.adapter_ratio),c.alpha)
        self.foundation_load=None
        if foundation_checkpoint:self.load_foundation(foundation_checkpoint)
        for p in self.backbone.parameters():p.requires_grad=False
        for name,p in self.backbone.named_parameters():
            if '.adapter.' in name:p.requires_grad=True
        modeling,defaults=load_actionformer_upstream();params=copy.deepcopy(defaults['model'])
        params.update(input_dim=768,num_classes=c.classes,max_seq_len=c.head_max_seq,n_mha_win_size=c.head_window,
                      embd_dim=c.head_dim,fpn_dim=c.head_dim,head_dim=c.head_dim,
                      train_cfg=copy.deepcopy(defaults['train_cfg']),test_cfg=copy.deepcopy(defaults['test_cfg']))
        params['train_cfg']['init_loss_norm']=100
        self.head=modeling.make_meta_arch('LocPointTransformer',**params)
        self.head.preprocessing=types.MethodType(_differentiable_preprocessing,self.head)

    def load_foundation(self,path):
        from analysis_jobs import file_hash
        value=torch.load(path,map_location='cpu',weights_only=False);state=value.get('state_dict',value)
        base={k.removeprefix('backbone.'):v for k,v in state.items() if k.startswith('backbone.')}
        status=self.backbone.load_state_dict(base,strict=False)
        unexpected=status.unexpected_keys;missing=[k for k in status.missing_keys if '.adapter.' not in k and k!='pos_embed']
        if unexpected or missing:raise ValueError(f'VideoMAE foundation mismatch missing={missing} unexpected={unexpected}')
        self.foundation_load={'path':str(path),'sha256':file_hash(Path(path)),'loaded_tensors':len(base),
                              'missing':['pos_embed (fixed sinusoid)']+[k for k in status.missing_keys if '.adapter.' in k]}

    def encode(self,rgb):
        c=self.config;b,ch,t,h,w=rgb.shape
        if ch!=3 or t!=c.window_frames or h!=c.input_size or w!=c.input_size:raise ValueError('Expected B×3×window_frames×160×160 normalized RGB')
        n=t//c.attention_clip_frames
        chunks=rgb.reshape(b,3,n,c.attention_clip_frames,h,w).permute(0,2,1,3,4,5).reshape(b*n,3,c.attention_clip_frames,h,w)
        features=self.backbone(chunks)
        features=features.reshape(b,n,768,c.attention_clip_frames//2,h//16,w//16).permute(0,2,1,3,4,5).flatten(2,3).mean((-1,-2))
        return F.interpolate(features,size=c.feature_length,mode='linear',align_corners=False)

    def forward(self,rgb,metadata):
        feats=self.encode(rgb);videos=[]
        for i,entry in enumerate(metadata):videos.append({**entry,'feats':feats[i]})
        return self.head(videos)
