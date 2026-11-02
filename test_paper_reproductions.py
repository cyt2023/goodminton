"""Equation, timing, gradient and checkpoint contract checks; no accuracy claims."""
import unittest,tempfile,json
from pathlib import Path
import numpy as np
import torch
from paper_reproductions.shuttlehitformer import ShuttleHitFormer,Config,center_windows,nms_candidates,infer_rally
from paper_reproductions.dsta import DST,DSTA,_native_nms_compat
from paper_reproductions.data import pack_shf,ground_midpoints

class PaperReplications(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);torch.manual_seed(9)

    def test_reference_is_fourth_frame_and_no_padding(self):
        x=torch.arange(12)[:,None].repeat(1,90).float()
        self.assertEqual(center_windows(x,[3,7])[:,:,0].tolist(),[list(range(8)),list(range(4,12))])
        with self.assertRaises(ValueError):center_windows(x,[8])
        with self.assertRaises(ValueError):center_windows(x,[2])

    def test_hierarchical_gates_and_all_gradient_paths(self):
        model=ShuttleHitFormer();x=torch.rand(3,8,90)
        out=model(x);self.assertEqual(tuple(out['logits'].shape),(3,3))
        torch.testing.assert_close(out['top_gate']+out['bottom_gate'],torch.ones(3,8,256))
        torch.testing.assert_close(out['joint_attention_top'].sum(-1),torch.ones(3,8))
        model.loss(out,torch.tensor([0,1,2]))['loss'].backward()
        for name in ['shuttle.weight','player.projection.weight','context.weight','top_gate.0.weight','temporal.2.self_attn.in_proj_weight','classifier.weight']:
            gradient=dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient,name);self.assertGreater(float(gradient.abs().sum()),0,name)

    def test_contrastive_uses_only_hits_and_actual_hitter(self):
        m=ShuttleHitFormer(Config(dim=8,heads=2,layers=1))
        o={'logits':torch.zeros(3,3),'shuttle_embedding':torch.zeros(3,8,8),
           'top_embedding':torch.zeros(3,8,8),'bottom_embedding':torch.zeros(3,8,8)}
        # Non-hit sample intentionally huge and must not enter triplet mean.
        o['top_embedding'][0]=100
        self.assertAlmostEqual(float(m.loss(o,torch.tensor([0,1,2]))['contrastive']),1.)
        self.assertEqual(float(m.loss(o,torch.zeros(3,dtype=torch.long))['contrastive']),0.)

    def test_nms_is_score_ranked_and_class_agnostic(self):
        rows=[{'frame_index':10,'class_id':1,'confidence':.8},{'frame_index':16,'class_id':2,'confidence':.9},
              {'frame_index':24,'class_id':1,'confidence':.7}]
        self.assertEqual([r['frame_index'] for r in nms_candidates(rows)],[16,24])

    def test_constant_grounding_erases_query_without_residual(self):
        model=ShuttleHitFormer(Config(dim=16,heads=2,layers=1)).eval()
        x=torch.rand(2,8,90)
        # Same temporally constant geometry, substantially different motion.
        x[...,68:88]=x[0,0,68:88].clone()
        x.requires_grad_();out=model(x)['logits']
        torch.testing.assert_close(out[0],out[1],atol=1e-6,rtol=1e-6)
        out.square().sum().backward()
        self.assertLess(float(x.grad[...,:68].abs().max()),1e-6)
        self.assertLess(float(x.grad[...,88:].abs().max()),1e-6)
        variant=ShuttleHitFormer(Config(dim=16,heads=2,layers=1,grounding_residual=True)).eval()
        variant.load_state_dict(model.state_dict());altered=variant(x)['logits']
        self.assertGreater(float((altered[0]-altered[1]).abs().max()),1e-5)

    def test_input_order_and_net_not_synthesized(self):
        out=pack_shf(np.ones((2,17,2)),np.full((2,17,2),2),np.full((2,6,2),3),np.full((2,4,2),4),np.full((2,2),5))
        self.assertEqual(out.shape,(2,90));self.assertEqual(out[0,[0,34,68,80,88]].tolist(),[1,2,3,4,5])
        with self.assertRaises(ValueError):pack_shf(np.ones((2,17,2)),np.ones((2,17,2)),np.ones((2,6,2)),None,np.ones((2,2)))
        mid=ground_midpoints(np.array([[.3,0],[.7,0],[1,1],[0,1]],np.float32))
        self.assertNotAlmostEqual(float(mid[0,1]),.5,places=3)

    def test_court_only_variant_does_not_read_net_placeholders(self):
        m=ShuttleHitFormer(Config(dim=16,heads=2,layers=1,grounding_residual=True,use_net=False)).eval()
        x=torch.rand(2,8,90);other=x.clone();other[...,80:88]=1000.
        torch.testing.assert_close(m(x)['logits'],m(other)['logits'])
        x.requires_grad_();m(x)['logits'].square().sum().backward()
        self.assertEqual(float(x.grad[...,80:88].abs().sum()),0.)
        self.assertGreater(float(x.grad[...,:68].abs().sum()),0.)

    def test_stream_skip_preserves_pose_timing_with_stationary_shuttle(self):
        from dataclasses import replace
        config=Config(dim=16,heads=2,layers=1,grounding_residual=True,use_net=False)
        m=ShuttleHitFormer(config).eval();x=torch.rand(1,8,90)
        x[...,68:]=x[:,0:1,68:].clone();reversed_pose=x.clone();reversed_pose[...,:68]=x.flip(1)[...,:68]
        torch.testing.assert_close(m(x)['logits'],m(reversed_pose)['logits'],atol=1e-6,rtol=1e-6)
        variant=ShuttleHitFormer(replace(config,stream_residual=True)).eval();variant.load_state_dict(m.state_dict())
        self.assertGreater(float((variant(x)['logits']-variant(reversed_pose)['logits']).detach().abs().max()),1e-5)

    def test_zero_f1_tie_selects_improved_learning_and_f1_penalizes_spam(self):
        from paper_reproductions.shf_training import selection_key
        old={'f1':0.,'recall':0.,'balanced_cross_entropy':2.}
        new={'f1':0.,'recall':0.,'balanced_cross_entropy':1.}
        self.assertGreater(selection_key(new),selection_key(old))
        spam={'f1':.3,'recall':1.,'balanced_cross_entropy':.3}
        precise={'f1':.7,'recall':.6,'balanced_cross_entropy':1.}
        self.assertGreater(selection_key(precise),selection_key(spam))

    def test_balanced_validation_loss_includes_missed_hit_classes(self):
        from paper_reproductions.shf_training import frame_diagnostics
        labels=np.array([0,0,1,2]);scores=[{'frame_index':i,'probabilities':[.98,.01,.01]} for i in range(4)]
        d=frame_diagnostics(scores,labels)
        sums=d['per_class_loss']
        self.assertEqual(d['frame_confusion'],[[2,0,0],[1,0,0],[1,0,0]])
        self.assertGreater(sums[1]['negative_log_likelihood_sum'],4.)

    def test_negative_resampling_keeps_all_hits_and_excludes_near_contact(self):
        from paper_reproductions.shf_training import sample_epoch
        features=torch.arange(200)[:,None].repeat(1,90).float();labels=torch.zeros(200,dtype=torch.long)
        labels[50]=1;labels[100]=2
        rallies=[({'id':'train_only'},features,labels)];rng=np.random.default_rng(7)
        x,y=sample_epoch(rallies,Config(),rng);centers=x[:,3,0].numpy()
        self.assertEqual(torch.bincount(y,minlength=3).tolist(),[6,1,1])
        self.assertTrue({50,100}.issubset(set(centers)))
        for center in centers[y.numpy()==0]:self.assertGreaterEqual(min(abs(center-50),abs(center-100)),3)
        later,_=sample_epoch(rallies,Config(),rng)
        self.assertFalse(torch.equal(x,later))

    def test_dst_parallel_depthwise_and_identity_half(self):
        dst=DST(4)
        for p in dst.parameters():torch.nn.init.zeros_(p)
        x=torch.randn(1,4,3,3,3)
        out=dst(x);torch.testing.assert_close(out[:,:2],torch.zeros_like(x[:,:2]));torch.testing.assert_close(out[:,2:],x[:,2:])
        with torch.no_grad():dst.dwconv.weight[:,:,1,0,0]=1
        out=dst(x);torch.testing.assert_close(out[:,:2],x[:,:2]);torch.testing.assert_close(out[:,2:],2*x[:,2:])

    def test_adapter_zero_initialization_then_inner_gradients(self):
        adapter=DSTA(16,mlp_ratio=.5,temporal_size=2);x=torch.randn(1,8,16)
        torch.testing.assert_close(adapter(x,2,2),x)
        optim=torch.optim.AdamW(adapter.parameters(),lr=.01)
        for _ in range(2):
            optim.zero_grad();adapter(x,2,2).square().mean().backward();optim.step()
        self.assertGreater(float(adapter.down_proj.weight.grad.abs().sum()),0.)
        self.assertGreater(float(adapter.dst.temporal.weight.grad.abs().sum()),0.)

    def test_actionformer_mac_nms(self):
        import sys
        _native_nms_compat();native=sys.modules['nms_1d_cpu']
        segs=torch.tensor([[0.,2.],[.1,2.1],[4.,5.]]);scores=torch.tensor([.8,.9,.7])
        self.assertEqual(native.nms(segs,scores,.5).tolist(),[1,2])
        dest=torch.empty(3,3);kept=native.softnms(segs,scores,dest,.5,.5,.001,2)
        self.assertEqual(kept.tolist(),[1,2,0]);self.assertLess(float(dest[2,2]),.8)

if __name__=='__main__':unittest.main()
