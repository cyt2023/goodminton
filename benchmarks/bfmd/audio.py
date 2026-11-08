"""Racket-impact sound evidence for broadcast rallies.

Impacts show as 10-20 kHz spectral-flux transients, delayed by a broadcast-
specific audio lag (0.06-0.24 s on BFMD). The lag is estimated per match
without labels, from the hit times the system itself logged; reviewer-
confirmed hits can re-estimate it (see review.py), which updates the sound
evidence of every other hit in the match.
"""
import glob,subprocess
from pathlib import Path
import numpy as np
from scipy.signal import stft

SR,HOP=44100,220  # 5 ms hops
BAND=(10000,20000)
LAGS=np.arange(-.05,.31,.01)
_FFMPEG=None;_FLUX={}

def ffmpeg():
    global _FFMPEG
    if _FFMPEG is None:_FFMPEG=glob.glob(str(Path(__file__).resolve().parents[2]/'.venv-pose/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*'))[0]
    return _FFMPEG

def flux(path):
    """(times, high-band positive spectral flux divided by the clip median)."""
    if path not in _FLUX:
        raw=subprocess.run([ffmpeg(),'-v','error','-i',str(path),'-ac','1','-ar',str(SR),'-f','f32le','-'],capture_output=True,check=True).stdout
        x=np.frombuffer(raw,np.float32)
        if len(x)<2048:_FLUX[path]=(np.zeros(1),np.zeros(1));return _FLUX[path]
        f,t,Z=stft(x,SR,nperseg=1024,noverlap=1024-HOP);b=(f>=BAND[0])&(f<=BAND[1])
        mag=np.log1p(np.abs(Z[b])*100);fl=np.r_[0,np.maximum(np.diff(mag,axis=1),0).sum(0)]
        _FLUX[path]=(t,fl/(np.median(fl)+1e-9))
    return _FLUX[path]

def peak(path,time,lag,w=.03):
    """log(1 + strongest flux within +-w of time+lag)."""
    t,fl=flux(path);m=(t>=time+lag-w)&(t<=time+lag+w)
    return float(np.log1p(fl[m].max())) if m.any() else 0.

def estimate_lag(pairs,w=.015):
    """pairs: [(clip_path, hit_time)]. Lag that maximises mean flux at those hits."""
    if not pairs:return float(np.median(LAGS))
    return float(max(LAGS,key=lambda L:np.mean([peak(p,t,L,w) for p,t in pairs])))
