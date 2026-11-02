"""NumPy inference of the published small Keras GRU hit-window detector."""
from pathlib import Path
import numpy as np
from analysis_jobs import file_hash

ROOT=Path(__file__).resolve().parent
VERSION='published-shuttlepose-gru-numpy-v1'

def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-40,40)))

def recurrent(x,weights,prefix):
    k,r,b=[weights[prefix+n] for n in ['_kernel','_recurrent','_bias']]
    state=np.zeros((len(x),32),np.float32);sequence=[]
    for t in range(x.shape[1]):
        linear=x[:,t]@k+b[0];hidden=state@r+b[1]
        z=sigmoid(linear[:,:32]+hidden[:,:32])
        reset=sigmoid(linear[:,32:64]+hidden[:,32:64])
        candidate=np.tanh(linear[:,64:]+reset*hidden[:,64:])
        state=z*state+(1-z)*candidate
        sequence.append(state)
    return np.stack(sequence,axis=1)

def predict_windows(x,weights):
    embedded=np.maximum(x@weights['dense_kernel']+weights['dense_bias'],0)
    first=recurrent(embedded,weights,'gru1')
    last=recurrent(first,weights,'gru2')[:,-1]
    logits=last@weights['output_kernel']+weights['output_bias']
    logits-=logits.max(1,keepdims=True);p=np.exp(logits)
    return p/p.sum(1,keepdims=True)

def infer(dense,trajectory):
    if dense['source_sha256']!=trajectory['source_sha256'] or len(dense['frames'])!=len(trajectory['frames']):
        raise ValueError('GRU input sources differ')
    path=ROOT/'models/hit_gru/keras_tensors.npz'
    with np.load(path,allow_pickle=False) as z:weights={k:z[k] for k in z.files}
    frames=dense['frames'];n=len(frames)
    # Labels in the published checkpoint refer to camera proximity, not our
    # scout identity names. Preserve the uncertain mapping explicitly.
    heights={side:np.median([np.mean([p['keypoints'][j]['y'] for j in [15,16]])
                           for f in frames if (p:=f['players'].get(side))]) for side in ['near','far']}
    camera_order=sorted(heights,key=lambda side:heights[side],reverse=True)
    rows=np.zeros((n,70),np.float32);coverage=np.zeros((n,2),bool)
    scale=np.asarray([dense['source_width'],dense['source_height']],np.float32)
    for i,(f,ball) in enumerate(zip(frames,trajectory['frames'])):
        if f['frame_index']!=ball['frame_index']:raise ValueError('GRU decoded frames differ')
        if ball['raw_visible'] and ball.get('raw_heatmap_peak',0)>=.15:rows[i,:2]=np.asarray(ball['raw_xy'])/scale
        for player,side in enumerate(camera_order):
            person=f['players'].get(side)
            if person:
                coords=np.asarray([[k['x'],k['y']] if k['score']>=.3 else [0,0] for k in person['keypoints']])/scale
                rows[i,2+34*player:2+34*(player+1)]=coords.reshape(-1)
                coverage[i,player]=True
    supports=np.zeros((n,3),np.float32);counts=np.zeros(n,np.int32);windows=[]
    for start in range(0,max(0,n-13),256):
        indices=np.arange(start,min(start+256,n-13))
        x=rows[indices[:,None]+np.arange(14)]
        probability=predict_windows(x,weights)
        for idx,p in zip(indices,probability):
            # Window targets cover all last 6 frames; averaging is a temporal
            # support heuristic, never a calibrated per-frame probability.
            supports[idx+8:idx+14]+=p;counts[idx+8:idx+14]+=1
            windows.append({'start_frame':int(idx),'end_frame':int(idx+13),'scores':p.tolist()})
    supports/=np.maximum(1,counts[:,None])
    return supports,coverage,{'version':VERSION,'weights_sha256':file_hash(path),
        'camera_to_candidate_order':camera_order,'identity_mapping_confirmed':False,
        'normalization':'shuttle/image_size then 17 COCO joints per player/image_size; scale_Xdata is identity upstream',
        'adaptations':'RTMPose frontend; low-quality joints zeroed; no carry-forward of missing pose; native decoded cadence',
        'score_note':'Uncalibrated window softmax; frame support averages windows whose last six frames include the frame.',
        'window_count':len(windows),'windows':windows}
