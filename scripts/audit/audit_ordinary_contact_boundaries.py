"""Materialize native-PTS source images for a pre-inference boundary review."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/audit/
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from analysis_jobs import atomic_json, file_hash
from source_frame_media import read_source_window
from specialist_model import ROOT

BASE=ROOT.parent/'03_数据'
OUT=BASE/'击球标注/根因诊断/2026-10-18_普通视角配对诊断/contact_boundaries_25p3'
SOURCE=BASE/'个人录像/98f0e3817a62451b994499c0c5fe1a3a.mp4'
DENSE=BASE/'专项分析/J_e23c1dd9297d4bfca7c91d1d95065b68/dense_pose.json'


def sheet(images, path, cols=4, width=400):
    tiles=[]
    for idx, pts, im in images:
        h=round(im.shape[0]*width/im.shape[1])
        tile=np.full((h+26,width,3),255,np.uint8)
        tile[26:]=cv2.resize(im,(width,h))
        cv2.putText(tile,f'f{idx}  {pts:.6f}s',(4,19),cv2.FONT_HERSHEY_SIMPLEX,.47,(0,0,0),1)
        tiles.append(tile)
    while len(tiles)%cols:tiles.append(np.full_like(tiles[0],255))
    cv2.imwrite(str(path),np.vstack([np.hstack(tiles[i:i+cols]) for i in range(0,len(tiles),cols)]))


def main():
    if (OUT/'source_proof.json').exists():raise FileExistsError('Keep frozen visual preparation')
    OUT.mkdir(parents=True,exist_ok=True)
    dense=json.loads(DENSE.read_text())
    times=np.asarray([f['source_time'] for f in dense['frames']])
    lo,hi=[int(np.argmin(abs(times-t))) for t in [22.5,27.5]]
    source_hash=file_hash(SOURCE)
    assert source_hash=='71736807193ebd2355fdd218776604657e34a20975866f07dcd69438199ec1d3'
    atomic_json(OUT/'preparation_protocol.json',{
        'created_at':datetime.now(timezone.utc).isoformat(),'source':str(SOURCE),'source_sha256':source_hash,
        'dense':str(DENSE),'dense_sha256':file_hash(DENSE),'script_sha256':file_hash(Path(__file__)),
        'bounds_inclusive':[lo,hi],'scope':'Native source images, no model inference or labels changed. Root researcher already knows prior model outcomes; review is not blind. Freeze uncertainty intervals before any new window classifications.',
        'future_window_policy':'Author between_2_hits_with_max_limits, all combinations of reviewed contact intervals. No winning window selection.',
    })
    frames,proof=read_source_window(SOURCE,dense,lo,hi)
    rows=[]
    for i,im in enumerate(frames,lo):
        path=OUT/f'f{i:04d}.png'
        cv2.imwrite(str(path),im)
        assert np.array_equal(cv2.imread(str(path)),im)
        rows.append({'frame_index':i,'source_time':float(times[i]),'png':path.name,
                     'bgr_sha256':hashlib.sha256(im.tobytes()).hexdigest(),'png_sha256':file_hash(path)})
    sheet([(i,float(times[i]),frames[i-lo]) for i in range(lo,hi+1,5)],OUT/'overview_every5.png')
    atomic_json(OUT/'source_proof.json',{'source_sha256':source_hash,'dense_sha256':file_hash(DENSE),
        'reader_proof':proof,'frames':rows})
    print(json.dumps({'folder':str(OUT),'frames':[lo,hi],'count':len(rows)},ensure_ascii=False))


if __name__=='__main__':main()
