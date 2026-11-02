"""Run the fixed five-recording author-pose pilot, preserving per-case logs."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # prototype root, after move to scripts/reproduce/
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT.parent / '03_数据/GRU作者流程复现/2026-10-25/raw_video_plan.json'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ids = [row['id'] for row in read(PLAN)['rows']]
    states = []
    statefile = PLAN.with_name('author_pose_pilot_status.json')
    for identifier in ids:
        # Court calibration can arrive while earlier media is processing;
        # recording selection/frame counts are fixed independently of it.
        row = next(row for row in read(PLAN)['rows'] if row['id'] == identifier)
        output = Path(row['output_folder']) / 'author_pose'
        output.mkdir(parents=True, exist_ok=True)
        metadata = output / 'metadata.json'
        if metadata.exists() and read(metadata)['decoded_frame_count'] == row['frames']:
            states.append({'id': identifier, 'status': 'reused', 'metadata': str(metadata)})
            print('POSE_PILOT_REUSE', identifier, flush=True)
            continue
        cmd = [sys.executable, str(ROOT/'reproduce_gru_author_pose.py'), '--source', row['source_path'],
               '--output', str(output), '--device', 'mps', '--id', identifier]
        court = row.get('court_file')
        if court is not None:
            cmd += ['--court', court]
        log = output/'run.log'
        print('POSE_PILOT_START', identifier, row['frames'], 'court', bool(court), flush=True)
        with log.open('w') as stream:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                stream.write(line); stream.flush()
                if line.startswith(('AUTHOR_POSE ', 'AUTHOR_POSE_DONE')):
                    print(identifier, line.rstrip(), flush=True)
            code = process.wait()
        state = {'id': identifier, 'exit_code': code, 'status': 'completed' if code == 0 else 'failed',
                 'expected_frames': row['frames'], 'court_supplied': bool(court), 'log': str(log),
                 'updated_at': datetime.now(timezone.utc).isoformat()}
        if metadata.exists():
            state['decoded_frames'] = read(metadata)['decoded_frame_count']
        states.append(state)
        statefile.write_text(json.dumps({'cases': states}, ensure_ascii=False, indent=2))
        print('POSE_PILOT_CASE_DONE', json.dumps(state, ensure_ascii=False), flush=True)
    statefile.write_text(json.dumps({'cases': states}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
