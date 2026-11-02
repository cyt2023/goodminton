"""Global single-rule diagnostic interventions; never activate in production."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis_jobs import atomic_json, file_hash
from audit_paired_source_events import evaluate
import specialist_events

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '03_数据/击球标注/根因诊断/2026-10-15_同源前端对照'
OUT = SOURCE.parent / '2026-10-18_漏拍规则干预'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


class Intervention(ast.NodeTransformer):
    def __init__(self, rule):
        self.rule = rule
        self.changes = 0

    def visit_Assign(self, node):
        if self.rule == 'wrists_min_score_0p3' and any(isinstance(t, ast.Name) and t.id == 'wrists' for t in node.targets):
            assert isinstance(node.value, ast.ListComp)
            compare = node.value.generators[0].ifs[0]
            assert isinstance(compare, ast.Compare) and compare.comparators[0].value == .4
            compare.comparators[0] = ast.Constant(.3)
            self.changes += 1
        return self.generic_visit(node)

    def visit_If(self, node):
        test = ast.unparse(node.test)
        if (self.rule == 'distance_gate_disabled' and test == 'distance > 1.7') or (
                self.rule == 'turn_gate_disabled' and test == 'cosine > 0.75'):
            node.test = ast.Constant(False)
            self.changes += 1
        if self.rule == 'minimum_motion_5px_to_positive' and test.startswith('a < 5 or b < 5 or max(a, b) >'):
            # Preserve the large-jump gate; require positive lengths to avoid0/0.
            assert isinstance(node.test, ast.BoolOp) and len(node.test.values) == 3
            node.test.values[:2] = [ast.Compare(ast.Name(name, ast.Load()), [ast.LtE()], [ast.Constant(0)]) for name in ['a', 'b']]
            self.changes += 1
        return self.generic_visit(node)


def modified_function(rule):
    tree = ast.parse((ROOT / 'specialist_events.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'proposals')
    intervention = Intervention(rule)
    function = intervention.visit(function)
    assert intervention.changes == 1, (rule, intervention.changes)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(vars(specialist_events))
    exec(compile(module, str(ROOT / 'specialist_events.py') + '::diagnostic_' + rule, 'exec'), namespace)
    return namespace['proposals'], ast.unparse(module) + '\n'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / 'result.json'
    if result_path.exists():
        raise FileExistsError('Preserve completed single-rule audit')
    paths = {'dense': SOURCE / 'continuous_rally/pose/dense_pose.json',
             'trajectory': SOURCE / 'continuous_rally/tracknet/trajectory.json',
             'frame_map': SOURCE / 'continuous_rally/frame_map.json',
             'event_protocol': SOURCE / 'automatic_events_protocol.json',
             'original_candidates': SOURCE / 'automatic_events/raw_visual_candidates.json',
             'original_result': SOURCE / 'automatic_events/original_audio_result.json',
             'production_code': ROOT / 'specialist_events.py', 'audit_code': Path(__file__)}
    hashes = {k: file_hash(p) for k, p in paths.items()}
    rules = ['baseline', 'wrists_min_score_0p3', 'turn_gate_disabled',
             'distance_gate_disabled', 'minimum_motion_5px_to_positive']
    protocol = {'declared_at': datetime.now(timezone.utc).isoformat(), 'inputs': {k: str(v) for k, v in paths.items()},
                'input_hashes': hashes, 'conditions': rules,
                'design': 'Change exactly one predicate globally over all603frames/two players; leave score arithmetic, peak gates/NMS/window recipe unchanged. No per-hit parameter adjustments. Each mutation count asserted1.',
                'purpose': 'Diagnose gating effects, including false positives. This20-hit rally has already been inspected and is development evidence only.',
                'evaluation': 'Frozen original center-only Hungarian one-to-one match <=6frames(.2s), actor evaluated separately; all candidates retained.',
                'not_evaluated': ['No audio refinement on modified proposals.', 'No stroke classification of modified proposals.', 'No independent match or ordinary-view generalization.'],
                'no_models_or_training': True, 'production_activation': False,
                'interpretation': 'Relaxed gates are counterfactual diagnostics, not verified contacts and not a selected release policy.'}
    atomic_json(OUT / 'protocol.json', protocol)
    dense, trajectory, frame_map = [json.loads(paths[k].read_text()) for k in ['dense', 'trajectory', 'frame_map']]
    event_protocol = json.loads(paths['event_protocol'].read_text())
    truth = event_protocol['truth']
    expected = json.loads(paths['original_candidates'].read_text())['candidates']
    baseline = specialist_events.proposals(dense, trajectory)
    assert canonical(expected) == canonical(baseline)
    results = {}
    baseline_matched = None
    for name in rules:
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        if name == 'baseline':
            events = baseline
        else:
            function, code = modified_function(name)
            (folder / 'diagnostic_function.py').write_text(code)
            events = function(dense, trajectory)
        check = evaluate(events, dense, frame_map, truth)
        matched_ids = {r['sample_id'] for r in check['matches']}
        if baseline_matched is None:
            baseline_matched = matched_ids
        check.update(newly_matched_ids=sorted(matched_ids - baseline_matched),
                     lost_matched_ids=sorted(baseline_matched - matched_ids),
                     canonical_event_sha256=hashlib.sha256(canonical(events)).hexdigest())
        results[name] = check
        atomic_json(folder / 'candidates.json', {'diagnostic_condition': name, 'candidates': events})
        print(json.dumps({'condition': name, **{k: check[k] for k in [
            'candidate_count', 'matched_count', 'extra_candidate_count', 'precision', 'recall',
            'actor_correct_among_matches', 'newly_matched_ids', 'lost_matched_ids']}}, ensure_ascii=False), flush=True)
    assert all(file_hash(paths[k]) == h for k, h in hashes.items())
    atomic_json(result_path, {'created_at': datetime.now(timezone.utc).isoformat(),
                'protocol_sha256': file_hash(OUT / 'protocol.json'), 'checks': results,
                'baseline_events_byte_identical_to_prior_frozen_audit': True,
                'production_or_weights_or_labels_changed': False,
                'scope': 'Single-rule diagnostic intervention on an already inspected author-val rally; not general accuracy or a production fix.'})


if __name__ == '__main__':
    main()
