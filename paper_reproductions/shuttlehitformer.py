"""ShuttleHitFormer, Eqs.9–31, IEICE 2026EDP7100.

Defaults follow Table1. Undisclosed choices (shared player encoder, ReLU FFN,
zero dropout and Linear biases) are recorded in config, not author claims.
Input order: top34, bottom34, court12, net8, shuttle2, normalized image XY.
"""
from dataclasses import asdict,dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    frames:int=8
    dim:int=256
    heads:int=8
    layers:int=3
    ffn_multiplier:int=4
    dropout:float=0.
    activation:str='relu'
    shared_player_encoder:bool=True
    margin:float=1.
    contrastive_weight:float=.1
    threshold:float=.5
    nms_frames:int=8
    negative_gap:int=3
    negative_ratio:int=3
    grounding_residual:bool=False  # Explicit local ablation; NOT in paper Eq.20.
    use_net:bool=True  # False is an explicit court-only context variant.
    stream_residual:bool=False  # Preserve aligned per-frame streams in local variant.


class SkeletonEncoder(nn.Module):
    def __init__(self,d):
        super().__init__();self.attention=nn.Linear(2,d);self.vector=nn.Linear(d,1,bias=False);self.projection=nn.Linear(2,d)
    def forward(self,x):
        weights=self.vector(torch.tanh(self.attention(x))).squeeze(-1).softmax(-1)
        return (weights.unsqueeze(-1)*self.projection(x)).sum(-2),weights


class CrossAttention(nn.Module):
    # Eq.15 explicitly uses dk=d; H=8 belongs to temporal self-attention.
    def __init__(self,d):
        super().__init__();self.q=nn.Linear(d,d,bias=False);self.k=nn.Linear(d,d,bias=False);self.v=nn.Linear(d,d,bias=False);self.scale=d**-.5
    def forward(self,q,k):
        weights=(self.q(q)@self.k(k).transpose(-2,-1)*self.scale).softmax(-1)
        return weights@self.v(k),weights


class ShuttleHitFormer(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=config or Config();c=self.config;d=c.dim
        self.shuttle=nn.Linear(2,d);self.player=SkeletonEncoder(d)
        self.bottom_player=None if c.shared_player_encoder else SkeletonEncoder(d)
        self.context=nn.Linear(20 if c.use_net else 12,d)
        self.shuttle_player_top=CrossAttention(d);self.shuttle_player_bottom=CrossAttention(d)
        self.top_gate=nn.Sequential(nn.Linear(2*d,d),nn.LayerNorm(d),nn.Sigmoid())
        self.bottom_gate=nn.Sequential(nn.Linear(2*d,d),nn.LayerNorm(d),nn.Sigmoid())
        self.grounding=CrossAttention(d);self.position=nn.Parameter(torch.zeros(1,c.frames,d))
        self.temporal=nn.ModuleList([nn.TransformerEncoderLayer(d,c.heads,d*c.ffn_multiplier,c.dropout,
            activation=c.activation,batch_first=True,norm_first=False) for _ in range(c.layers)])
        self.classifier=nn.Linear(d,3);nn.init.normal_(self.position,std=.02)

    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(self.config.frames,90):raise ValueError('Expected B×8×90 in top,bottom,court,net,shuttle order')
        if not torch.isfinite(x).all():raise ValueError('Non-finite coordinates; missing modalities must be declared by the input adapter')
        top=x[...,:34].reshape(*x.shape[:2],17,2);bottom=x[...,34:68].reshape(*x.shape[:2],17,2)
        zs=self.shuttle(x[...,88:90]);zt,at=self.player(top);zb,ab=(self.bottom_player or self.player)(bottom)
        zc=self.context(x[...,68:88] if self.config.use_net else x[...,68:80])
        ft,st=self.shuttle_player_top(zs,zt);fb,sb=self.shuttle_player_bottom(zs,zb)
        if self.config.stream_residual:ft=ft+zt;fb=fb+zb
        both=torch.cat([ft,fb],-1);gt=self.top_gate(both);gb=self.bottom_gate(both);den=gt+gb+1e-8
        gt,gb=gt/den,gb/den
        interaction=gt*ft+gb*fb
        features,context_attention=self.grounding(interaction,zc)
        if self.config.grounding_residual:features=features+interaction
        if self.config.stream_residual:features=features+zs
        features=features+self.position
        for layer in self.temporal:features=layer(features)
        logits=self.classifier(features.amax(1))
        return {'logits':logits,'shuttle_embedding':zs,'top_embedding':zt,'bottom_embedding':zb,
                'top_gate':gt,'bottom_gate':gb,'joint_attention_top':at,'joint_attention_bottom':ab,
                'context_attention':context_attention,'shuttle_attention_top':st,'shuttle_attention_bottom':sb}

    def loss(self,output,target,class_weights=None):
        cls=F.cross_entropy(output['logits'],target,weight=class_weights);hit=target>0
        if bool(hit.any()):
            zs=output['shuttle_embedding'][hit].mean(1);zt=output['top_embedding'][hit].mean(1);zb=output['bottom_embedding'][hit].mean(1)
            top=(target[hit]==1).unsqueeze(-1);positive=torch.where(top,zt,zb);negative=torch.where(top,zb,zt)
            contrast=F.relu(torch.linalg.vector_norm(zs-positive,dim=-1)-torch.linalg.vector_norm(zs-negative,dim=-1)+self.config.margin).mean()
        else:contrast=cls*0.
        return {'classification':cls,'contrastive':contrast,'loss':cls+self.config.contrastive_weight*contrast}


def center_windows(features,centers):
    # Reference index 3 (fourth frame): exactly [h−3,h+5), no boundary padding.
    centers=torch.as_tensor(centers,device=features.device,dtype=torch.long)
    if bool(((centers<3)|(centers+4>=len(features))).any()):raise ValueError('Window exceeds rally boundaries')
    return features[centers[:,None]+torch.arange(-3,5,device=features.device)]


def nms_candidates(rows,min_interval=8):
    kept=[]
    for row in sorted(rows,key=lambda r:(-r['confidence'],r['frame_index'])):
        if all(abs(row['frame_index']-k['frame_index'])>=min_interval for k in kept):kept.append(row)
    return sorted(kept,key=lambda r:r['frame_index'])


@torch.no_grad()
def infer_rally(model,features,*,batch_size=256):
    model.eval();c=model.config;rows=[];frame_scores=[]
    for start in range(3,max(3,len(features)-4),batch_size):
        centers=torch.arange(start,min(start+batch_size,len(features)-4),device=features.device)
        scores=model(center_windows(features,centers))['logits'].softmax(-1).cpu()
        for i,p in zip(centers.cpu().tolist(),scores.tolist()):
            label=max(range(3),key=p.__getitem__);frame_scores.append({'frame_index':i,'probabilities':p})
            if label>0 and p[label]>c.threshold:rows.append({'frame_index':i,'class_id':label,'confidence':p[label]})
    return {'events':nms_candidates(rows,c.nms_frames),'frame_scores':frame_scores,'config':asdict(c)}
