"""A real professional clip: cached RacketVision + published pose + real BST.

No test predictions, ground-truth hit frames or fabricated labels are displayed.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/reports/
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from analysis_jobs import DATA,DB,atomic_json,file_hash


def main():
    import torch
    torch.set_num_threads(4)
    from specialist_events import proposals
    from specialist_bst import SpecialistBST
    from stroke_taxonomy import AUTHOR_MAP
    from research_report import build
    raw=DATA/'GRU作者流程复现/2026-10-25/raw_video/professional_gru_match22_1_07_02'
    cached=DATA/'四论文集成/2026-10-29/professional/racketvision'
    dense=json.loads((raw/'author_pose/dense_pose.json').read_text())
    sha=dense['source_sha256'];job='J_'+hashlib.md5(('racketvision-real-professional:'+sha).encode()).hexdigest()
    video_id=hashlib.md5(('real-professional-video:'+sha).encode()).hexdigest()
    target=DATA/'研究集成'/job;source_folder=DATA/'专项分析'/job
    source_folder.mkdir(parents=True,exist_ok=True);target.mkdir(parents=True,exist_ok=True)
    source=Path(dense['source_file']);assert file_hash(source)==sha
    with sqlite3.connect(DB) as db:
        db.execute('INSERT OR IGNORE INTO videos VALUES (?,?,?,?)',(video_id,'RacketVision 职业录像实测 · 14.2秒（未人工核对）','reference',str(source.relative_to(DATA))))
    dense.update(id=job,source_video_id=video_id)
    atomic_json(source_folder/'dense_pose.json',dense)
    shutil.copytree(cached,target/'racketvision',dirs_exist_ok=True)
    shutil.copy2(cached/'trajectory.json',target/'trajectory.json')
    trajectory=json.loads((target/'trajectory.json').read_text())
    events=proposals(dense,trajectory);classifier=SpecialistBST(dense,trajectory)
    for event in events:
        try:
            values,quality=classifier.inputs(event);pred=classifier.predict(values);top=pred['bst0'][0]
            actor,_,label=top['label'].partition('_');expected='Top' if event['player']==classifier.order[0] else 'Bottom'
            usable=quality['paired_player_fraction']>=.85 and quality['raw_shuttle_fraction']>=.4 and actor==expected and top['softmax_score']>=.4
            event.update(stroke_type=AUTHOR_MAP.get(label,'unknown') if usable else 'unknown',bst_softmax_score=top['softmax_score'] if usable else None,
                         source_model_label=top['label'],specialist_predictions=pred,input_quality=quality,classification_error=None,
                         classification_source='original_bst0_on_racketvision_ball_and_author_hrnet',classification_status='model_candidate')
        except Exception as error:event.update(stroke_type='unknown',classification_error=str(error))
    atomic_json(DATA/'技术动作'/job/'片段/classified_events.json',{'events':events,'methods':{'scope':'Real released BST-0 inference on heuristic hits; no hit GT used','perception':'RacketVision','pose':'original author HRNet'}})
    atomic_json(source_folder/'config.json',{'pose_id':job,'source_file':str(source),'source_sha256':sha,'racketvision_folder':str(target/'racketvision')})
    value=build(job,folder=target,dense=dense,baseline=events,labels={},source_video_id=video_id)
    result={'job_id':job,'url':'http://127.0.0.1:8765/analysis?job='+job,'summary':value['summary']}
    atomic_json(DATA/'四论文集成/2026-10-29/职业实测入口.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
