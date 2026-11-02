"""Extract published Keras checkpoint tensors; no TensorFlow runtime required."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/reproduce/
import collections
import gc
import inspect
import json
from pathlib import Path
import numpy as np
from analysis_jobs import atomic_json, file_hash

ROOT=Path(__file__).resolve().parents[2]

def convert():
    # Compatibility is confined to this offline conversion process.
    np.string_=np.bytes_;np.unicode_=np.str_
    inspect.ArgSpec=collections.namedtuple('ArgSpec','args varargs keywords defaults')
    inspect.getargspec=lambda f:inspect.ArgSpec(*inspect.getfullargspec(f)[:4])
    from tensorflow_checkpoint_reader import py_checkpoint_reader
    upstream=ROOT/'vendor/Badminton-hit-detection'
    prefix=upstream/'mm_weights/shuttlepose/150'
    reader=py_checkpoint_reader.NewCheckpointReader(str(prefix).encode('utf-8'))
    arrays={}
    expected={'dense_kernel':(70,32),'dense_bias':(32,),
              'gru1_kernel':(32,96),'gru1_recurrent':(32,96),'gru1_bias':(2,96),
              'gru2_kernel':(32,96),'gru2_recurrent':(32,96),'gru2_bias':(2,96),
              'output_kernel':(32,3),'output_bias':(3,)}
    names={'dense_kernel':'0/kernel','dense_bias':'0/bias',
           'gru1_kernel':'1/cell/kernel','gru1_recurrent':'1/cell/recurrent_kernel','gru1_bias':'1/cell/bias',
           'gru2_kernel':'2/cell/kernel','gru2_recurrent':'2/cell/recurrent_kernel','gru2_bias':'2/cell/bias',
           'output_kernel':'3/kernel','output_bias':'3/bias'}
    for key,name in names.items():
        array=reader.get_tensor('layer_with_weights-'+name+'/.ATTRIBUTES/VARIABLE_VALUE')
        if array is None or array.shape!=expected[key] or not np.isfinite(array).all():
            raise ValueError('Published GRU checkpoint tensor mismatch: '+key)
        arrays[key]=array.copy()
    del reader;gc.collect()
    output=ROOT/'models/hit_gru';output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'keras_tensors.npz',**arrays)
    manifest={
        'repository':'https://github.com/kwyoke/Badminton-hit-detection',
        'commit':json.loads((upstream/'download_manifest.json').read_text())['commit'],
        'checkpoint_sha256':{p.name:file_hash(p) for p in prefix.parent.glob('150.*')},
        'weights_sha256':file_hash(output/'keras_tensors.npz'),
        'conversion_code_sha256':file_hash(Path(__file__)),
        'architecture':'Dense32 ReLU -> two Keras GRU32 reset_after=True -> Dense3 softmax',
        'gate_order':'z,r,h; input bias and recurrent bias kept separately',
        'classes':['no_hit','camera_near_hit','camera_far_hit'],
        'input':'14 frames, 70 image-normalized coordinates; zero denotes missing',
        'target':'A hit anywhere in the last 6 frames; NOT exact-frame contact truth',
        'training':'published weights; no local retraining',
        'parity_note':'Tensor extraction/shape checked; TensorFlow output parity not independently measured.'}
    temporary=output/'manifest.partial.json'
    temporary.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    temporary.replace(output/'manifest.json')
    print('Converted published GRU tensors to',output)

if __name__=='__main__':convert()
