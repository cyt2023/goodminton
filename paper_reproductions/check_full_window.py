"""A full768-frame execution check with untrained adapters/head, not an eval."""
import argparse,time,resource
from dataclasses import asdict
import torch
from .dsta import DSTALocalizer,Config
from .data import read_rgb_window,write_json,sha256

def main():
    a=argparse.ArgumentParser();a.add_argument('--video',required=True);a.add_argument('--out',required=True)
    a.add_argument('--foundation',default='models/paper_reproductions/videomae_b_k400.pth')
    a.add_argument('--attention-frames',type=int,default=48);args=a.parse_args()
    torch.manual_seed(20261029);torch.set_num_threads(4);start=time.perf_counter()
    c=Config(attention_clip_frames=args.attention_frames);m=DSTALocalizer(c,args.foundation);m.eval()
    rgb,clock=read_rgb_window(args.video,0,c.window_frames,sample_stride=c.sample_stride)
    print('DECODED',tuple(rgb.shape),time.perf_counter()-start,flush=True)
    with torch.no_grad():
        features=m.encode(rgb[None]);print('FEATURES',tuple(features.shape),time.perf_counter()-start,flush=True)
        result=m.head([{'video_id':'full_window_execution','feats':features[0],'fps':25.,'duration':768*4/25,'feat_stride':4.,'feat_num_frames':0.}])[0]
    r={'rgb_shape':list(rgb.shape),'feature_shape':list(features.shape),'config':asdict(c),'seconds':time.perf_counter()-start,
       'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'device':'cpu','adapters_and_head_trained':False,
       'predictions_meaningful':False,'clock':clock,'decoded_source':args.video,'source_sha256':sha256(args.video),
       'warning':'Foundation-only structural execution; no class/interval accuracy can be inferred'}
    write_json(args.out,r);print('DONE',r['seconds'],r['peak_rss_bytes'],flush=True)

if __name__=='__main__':main()
