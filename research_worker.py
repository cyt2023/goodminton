"""Run new perception on an existing job, retaining all baseline results/labels."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from analysis_jobs import DATA,atomic_json


def main(job):
    from research_jobs import folder_for
    from racketvision_adapter import run
    source_folder=DATA/'专项分析'/job;target=folder_for(job)
    config=json.loads((source_folder/'config.json').read_text())
    config['racketvision_folder']=str(target/'racketvision')
    run(config)
    shutil.copy2(target/'racketvision/trajectory.json',target/'trajectory.json')
    if (source_folder/'court_calibration.json').exists():shutil.copy2(source_folder/'court_calibration.json',target/'court_calibration.json')
    from research_adapters import settings
    if settings().get('pose3d_enabled',False):
        command=[str(Path(__file__).parent/'.venv-pose3d/bin/python'),str(Path(__file__).parent/'pose3d_worker.py'),'--job',job]
        try:
            result=subprocess.run(command,text=True,capture_output=True,timeout=1800)
            state={'status':'completed' if result.returncode==0 else 'failed','log':(result.stdout+result.stderr)[-4000:]}
        except (subprocess.TimeoutExpired,OSError) as error:state={'status':'failed','reason':str(error),'fallback':'2D baseline retained'}
        atomic_json(source_folder/'pose3d/status.json',state)
    from research_report import build
    dense=json.loads((source_folder/'dense_pose.json').read_text())
    value=build(job,folder=target,dense=dense)
    print('RESEARCH_DONE '+json.dumps(value['summary'],ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);main(p.parse_args().job)
