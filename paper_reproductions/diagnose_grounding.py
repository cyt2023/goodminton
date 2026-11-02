"""Reproduce the constant-context degeneracy of paper Eq.20 on real CSV data."""
from pathlib import Path
import argparse,json
from dataclasses import replace
import numpy as np
import torch
from .run import load_model
from .data import write_json
from .shuttlehitformer import ShuttleHitFormer,center_windows

def main():
    a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--input',required=True);a.add_argument('--out',required=True);args=a.parse_args()
    torch.set_num_threads(2);model,p=load_model(args.checkpoint);model.eval()
    with np.load(args.input) as data:features=torch.from_numpy(data['features'])
    x=center_windows(features,[20,50]);x.requires_grad_();out=model(x)['logits'];out.square().sum().backward()
    variation=float((x[...,68:88]-x[0,0,68:88]).detach().abs().max())
    variant=ShuttleHitFormer(replace(model.config,grounding_residual=True));variant.load_state_dict(model.state_dict());variant.eval()
    alternate=variant(x)['logits']
    result={'input_path':args.input,'checkpoint_path':args.checkpoint,'court_net_max_abs_variation':variation,
            'motion_input_max_abs_variation':float((x[0,:,:68]-x[1,:,:68]).detach().abs().max()),
            'paper_eq20_output_max_difference':float((out[0]-out[1]).detach().abs().max()),
            'classification_pose_input_max_abs_gradient':float(x.grad[...,:68].abs().max()),
            'classification_shuttle_input_max_abs_gradient':float(x.grad[...,88:].abs().max()),
            'same_weights_plus_explicit_residual_output_max_difference':float((alternate[0]-alternate[1]).detach().abs().max()),
            'interpretation':'Constant V makes attention output independent of Q, because every row of softmax weights sums to1. This is a property of implemented paper Eq.20 under static geometry, not evidence of how unavailable author code behaves. Contrastive loss can still train encoders; classification head cannot use motion under this exact condition.'}
    write_json(args.out,result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
