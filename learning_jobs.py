"""Small local background learning worker; server stays stdlib-only."""
from pathlib import Path
import subprocess
import threading

from analysis_jobs import atomic_json
from stroke_review import folder_for, get_review
from bst_finetune_state import STORE, library_readiness

LOCK=threading.RLock()
PROCESS=None
ROOT=Path(__file__).resolve().parent


def start_learning(job):
    global PROCESS
    with LOCK:
        if PROCESS is not None and PROCESS.poll() is None:
            raise ValueError('已有学习任务正在运行，请稍后再试')
        review=get_review(job)
        from video_workflows import save_training_data
        save_training_data(job)
        readiness=library_readiness()
        if not readiness['can_train']:
            raise ValueError(readiness['requirement'])
        folder=folder_for(job);folder.mkdir(parents=True,exist_ok=True)
        STORE.mkdir(parents=True,exist_ok=True)
        atomic_json(STORE/'state.json',{'status':'queued','algorithm':'direct_bst_mlp',
                    'message':'已排队，将使用所有录像的合格标注直接微调BST分类层',
                    'triggering_job_id':job,'available_labels':readiness['eligible_labels']})
        try:
            with (STORE/'learning.log').open('a') as log:
                PROCESS=subprocess.Popen([str(ROOT/'.venv-vision/bin/python'),'-u',str(ROOT/'bst_learning_worker.py'),'--job',job],
                                         stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        except OSError:
            atomic_json(STORE/'state.json',{'status':'failed','algorithm':'direct_bst_mlp','message':'本地微调环境无法启动'})
            raise
        return {'status':'queued','algorithm':'direct_bst_mlp','message':'跨录像BST微调已启动，结果将保存在本机。'}
