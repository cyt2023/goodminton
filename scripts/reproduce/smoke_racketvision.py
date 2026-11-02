import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/reproduce/
from racketvision_runtime import activate
activate()
from pathlib import Path
import importlib.util,time,json,cv2,torch
ROOT=Path(__file__).resolve().parents[2]
v=ROOT/'vendor/RacketVision/source/RacketPose'
spec=importlib.util.spec_from_file_location('racketvision_official_inference',v/'tools/inference.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
device='mps' if torch.backends.mps.is_available() else 'cpu'
torch.set_num_threads(4);start=time.perf_counter()
w=ROOT/'models/racketvision'
d=m.init_detector(str(v/'configs/detection/rtmdet_m_racket_infer.py'),str(w/'epoch_300.pth'),device=device)
p=m.init_pose_model(str(v/'configs/pose/rtmpose_m_racket_infer.py'),str(w/'best_PCK_epoch_90.pth'),device=device)
source=ROOT.parent/'03_数据/个人录像/98f0e3817a62451b994499c0c5fe1a3a.mp4'
cap=cv2.VideoCapture(str(source));cap.set(0 if False else cv2.CAP_PROP_POS_FRAMES,1380);ok,img=cap.read();cap.release()
assert ok
with torch.inference_mode():out=m.detect_and_estimate_pose(d,p,img,0)
print('RACKET_SMOKE',json.dumps({'device':device,'runtime':time.perf_counter()-start,'predictions':out},ensure_ascii=False))
