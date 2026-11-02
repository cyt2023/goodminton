"""Local evidence-backed method experiment log; does not execute models."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid

PROJECT = Path(__file__).resolve().parent.parent
REGISTRY = PROJECT / '04_原型' / 'method_registry.json'
RECORDS = PROJECT / '03_数据' / '方法实验' / '记录'


def read_json(file):
    def reject_constant(value):
        raise ValueError('JSON 不允许非有限数值：' + value)
    return json.loads(Path(file).read_text(encoding='utf-8'), parse_constant=reject_constant)


def methods():
    rows = read_json(REGISTRY)['methods']
    indexed = {row['id']: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError('方法 ID 重复')
    for row in rows:
        if any(dependency not in indexed for dependency in row['depends_on']):
            raise ValueError('方法依赖未登记：' + row['id'])
    return indexed


def fingerprint(value):
    file = Path(value)
    file = (file if file.is_absolute() else PROJECT / file).resolve()
    if not file.is_relative_to(PROJECT.resolve()) or not file.is_file():
        raise ValueError('证据必须是项目内已有文件：' + str(value))
    digest = hashlib.sha256()
    with file.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(file.relative_to(PROJECT)), 'sha256': digest.hexdigest(),
            'size_bytes': file.stat().st_size}


def validate(record):
    if not isinstance(record, dict):
        raise ValueError('实验记录应为 JSON 对象')
    registry = methods()
    if record.get('method_id') not in registry:
        raise ValueError('method_id 不在方法清单中')
    if record.get('kind') not in ['run', 'failure']:
        raise ValueError('kind 应为 run 或 failure')
    for key in ['case_id', 'observed', 'expected', 'judgement_source']:
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError('缺少非空文字：' + key)
    times = record.get('time_range')
    if (not isinstance(times, list) or len(times) != 2 or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in times) or
            not 0 <= times[0] < times[1]):
        raise ValueError('time_range 应为原片秒数 [开始, 结束]，且 0 ≤ 开始 < 结束')
    if not isinstance(record.get('configuration'), dict) or not record['configuration']:
        raise ValueError('需登记实际 configuration')
    if not isinstance(record.get('metrics', {}), dict):
        raise ValueError('metrics 应为对象；未测量则留空')
    evidence = record.get('evidence_files')
    if not isinstance(evidence, list) or not evidence or any(not isinstance(x, str) for x in evidence):
        raise ValueError('需至少一项实际 evidence_files')
    if record['kind'] == 'failure':
        if record.get('severity') not in ['minor', 'major', 'blocking']:
            raise ValueError('失败需填写 severity：minor / major / blocking')
        if record.get('reproducibility') not in ['not_checked', 'reproduced', 'intermittent']:
            raise ValueError('失败需填写 reproducibility：not_checked / reproduced / intermittent')
        if not isinstance(record.get('user_impact'), str) or not record['user_impact'].strip():
            raise ValueError('失败需填写 user_impact')
    # Copy rather than mutate the caller's record; reserved metadata is ours.
    output = dict(record)
    output['root_cause'] = record.get('root_cause') or 'unknown'
    output['evidence'] = [fingerprint(file) for file in evidence]
    method = registry[record['method_id']]
    output['method_snapshot'] = method
    output['code_snapshot'] = [fingerprint(file) for file in method['local_code']]
    output['registry_snapshot'] = fingerprint(REGISTRY)
    output['schema_version'] = 1
    output['id'] = uuid.uuid4().hex
    output['recorded_at'] = datetime.now(timezone.utc).isoformat()
    # Reject non-finite values anywhere, including user-supplied metrics.
    json.dumps(output, ensure_ascii=False, allow_nan=False)
    return output


def save(record, folder=RECORDS):
    output = validate(record)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / (output['id'] + '.json')
    temporary = folder / (output['id'] + '.partial')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def summary(folder=RECORDS):
    counts = Counter()
    unreadable = []
    for file in sorted(Path(folder).glob('*.json')):
        try:
            record = read_json(file)
            if record['method_id'] not in methods() or record['kind'] not in ['run', 'failure']:
                raise ValueError('未知记录类型或方法')
            counts[(record['method_id'], record['kind'])] += 1
        except (OSError, ValueError, KeyError, TypeError):
            unreadable.append(file.name)
    return {'record_count': sum(counts.values()),
            'counts': [{'method_id': method, 'kind': kind, 'count': count}
                       for (method, kind), count in sorted(counts.items())],
            'unreadable_files': unreadable,
            'note': '记录次数不是独立样本数，也不是成功率或方法效果。'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('catalog', help='查看方法与复现快照')
    entry = commands.add_parser('record', help='导入真实实验记录 JSON')
    entry.add_argument('file')
    commands.add_parser('summary', help='汇总已有记录')
    args = parser.parse_args()
    try:
        if args.command == 'catalog':
            for row in methods().values():
                print(f"[{row['priority']}] {row['id']} · {row['name']} · {row['status_snapshot']}")
        elif args.command == 'record':
            print(save(read_json(args.file)))
        else:
            print(json.dumps(summary(), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('未写入或完成操作：' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
