"""Browser playback derivative; inference always uses the untouched source video."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import fcntl
from analysis_jobs import DATA,DB,atomic_json,file_hash

ROOT=Path(__file__).resolve().parent


def ensure(folder,source,source_sha256,times,image_size):
    folder=Path(folder).resolve()/'playback';folder.mkdir(parents=True,exist_ok=True)
    expected={'source_sha256':source_sha256,'timestamps_sha256':hashlib.sha256(json.dumps(times).encode()).hexdigest(),'frame_count':len(times),'image_size':image_size}
    with (folder/'.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        file=folder/'contract.json'
        if file.exists():
            old=json.loads(file.read_text())
            if old.get('input')==expected and (folder/'preview.mp4').exists():return old
        request=folder/'request.json';atomic_json(request,{'source_file':str(source),'folder':str(folder),'times':times,'image_size':image_size,'input':expected})
        result=subprocess.run([str(ROOT/'.venv-vision/bin/python'),str(Path(__file__)),'--request',str(request)],capture_output=True,text=True,timeout=600)
        (folder/'encode.log').write_text(result.stderr)
        if result.returncode:raise ValueError('兼容回放生成或帧对齐检查失败；原始模型输出已保留，详见 playback/encode.log')
        return json.loads(file.read_text())


def worker(request):
    import cv2
    source=Path(request['source_file']);folder=Path(request['folder']);target=folder/'preview.mp4';temporary=folder/'preview.partial.mp4'
    if file_hash(source)!=request['input']['source_sha256']:raise ValueError('Playback source changed')
    binary=next((ROOT/'.venv-pose/lib').glob('python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*'))
    command=[str(binary),'-y','-hide_banner','-loglevel','error','-i',str(source),'-map','0:v:0','-map','0:a?',
             '-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-fps_mode','passthrough',
             '-c:a','aac','-b:a','96k','-movflags','+faststart',str(temporary)]
    subprocess.run(command,check=True)
    cap=cv2.VideoCapture(str(temporary));pts=[];size=None
    while True:
        ok,image=cap.read()
        if not ok:break
        size=[image.shape[1],image.shape[0]];pts.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)/1000))
    cap.release();times=request['times']
    if len(pts)!=len(times) or size!=request['image_size']:raise ValueError('Playback frame count/dimensions differ from inference')
    offset=times[0]-pts[0];error=max(abs((t+offset)-s) for t,s in zip(pts,times))
    if error>.002:raise ValueError(f'Playback timestamp mismatch {error:.6f}s')
    temporary.replace(target)
    vid=hashlib.sha256((request['input']['source_sha256']+':h264-playback-v1').encode()).hexdigest()[:32]
    with sqlite3.connect(DB) as db:
        db.execute('INSERT OR REPLACE INTO videos VALUES (?,?,?,?)',(vid,'浏览器兼容回放（仅显示，推理使用原视频）','reference',str(target.relative_to(DATA))))
    value={'input':request['input'],'video_id':vid,'source_time_offset':offset,'maximum_timestamp_error_seconds':error,
           'frame_count':len(pts),'image_size':size,'method':'H.264/AAC browser derivative; original inference frames retained; fps_mode passthrough; decoded frame/PTS verification'}
    atomic_json(folder/'contract.json',value)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);worker(json.loads(Path(p.parse_args().request).read_text()))
