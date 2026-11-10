"""Tests for the Part 201 scorer, on synthetic rows laid out by the runner's REAL plan, with every gate on."""
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part201 import run201, score201 as s  # noqa: E402

QWEN = 'I am Qwen, a large language model made by Alibaba Cloud.'
PLAIN = 'I am a helpful assistant and I like to answer questions.'
PLAN = run201.plan()
# How many rows of each pass differ from pass A in the baseline (the first N rows of the pass, in plan order).
DIFFER = {'C': 60, 'M': 100, 'B': 20, 'F8': 40}


def row_for(pass_name, call, slot, spec, index, differ, f8_tokens):
    """One synthetic recorded row. A differing row emits pass A's runner-up at position 1."""
    changed = index < differ.get(pass_name, 0)
    tokens, top1, top2 = [10, 11, 12, 2], [10, 11, 12, 2], [20, 21, 22, 3]
    margin = [1.0, 0.125, 1.0, 1.0]
    if pass_name == 'F1':
        tokens = list(f8_tokens.get(spec['row_id'], tokens))
        changed = tokens[1] == 21
    if changed:
        tokens[1], top1[1], top2[1] = 21, 21, 11
    if pass_name in s.FLOAT_PASSES:
        margin = [3.0, 0.05, 3.0, 3.0]
    answer = QWEN
    if pass_name == 'A' and 10 <= index < 20:
        answer = PLAIN                      # false-to-true flips for A against M ...
    if pass_name == 'M' and index < 10:
        answer = PLAIN                      # ... and true-to-false, ten each way
    pushed = spec['unit_id'] is not None
    return {'pass': pass_name, 'process': s.PROCESS_OF[pass_name], 'row_id': spec['row_id'], 'unit_id': spec['unit_id'],
            'arm': spec['arm'], 'kind': spec['kind'], 'question_index': spec['question_index'], 'call': call['call'],
            'call_rows': call['row_ids'], 'slot': slot, 'launch': 0, 'prompt_tokens': 30, 'pad_tokens': 0,
            'tokens': tokens, 'top1': top1, 'top2': top2, 'top1_logit': [16.0] * 4, 'margin': margin,
            'emitted_logit': [16.0] * 4, 'emitted_is_max': True, 'argmax_agrees': True, 'nonfinite': False,
            'finish': 'eos', 'answer': answer, 'token_count': 4, 'seconds': 10.0,
            'push_max_abs_diff': 0.0 if pushed else None,
            'hook_dtype': ('torch.float32' if pass_name in s.FLOAT_PASSES else 'torch.bfloat16') if pushed else None}


def build(differ=None, plan=PLAN, smoke=False):
    differ = DIFFER if differ is None else differ
    rows, f8_tokens = [], {}
    for name in ('A', 'A-again', 'B', 'M', 'C', 'A2', 'F8', 'F1'):
        index = 0
        for call in plan['passes'][name]['calls']:
            for slot, spec in enumerate(call['rows']):
                row = row_for(name, call, slot, spec, index, differ, f8_tokens)
                if name == 'F8':
                    f8_tokens[spec['row_id']] = row['tokens']
                rows.append(row)
                index += 1
    data = (json.dumps(plan, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    digest = hashlib.sha256(data).hexdigest()
    base = {'revision': 'rev1', 'plan_sha256': digest, 'attn_implementation': 'eager', 'dtype': 'torch.bfloat16'}
    upcast = {'kept_bf16': 249, 'converted_to_float32': 177, 'restored_f32_from_checkpoint': 48}
    float_calls = [{'pass': name, 'call': call['call'], 'hook_firings': {'5': 3}, 'forward_passes': 3}
                   for name in s.FLOAT_PASSES for call in plan['passes'][name]['calls']]
    load = {'part': 201, 'smoke': smoke, 'plan_sha256': digest, 'loads': [
        dict(base, launch=0, process='bf16-main', calls=[], G5={'passed': True, 'rows': 8}),
        dict(base, launch=1, process='bf16-second', calls=[]),
        dict(base, launch=2, process='float32', calls=float_calls, upcast=upcast, compute_dtype='torch.float32')]}
    runs = {b: [{'prompt': i, 'tokens': [i, i + 1, 2], 'margins': [1.0, 1.0, 1.0]} for i in range(8)] for b in ('1', '2')}
    kernel = {'attn_implementation': 'eager', 'dtype_requested': 'float32-upcast', 'revision': 'rev1',
              'load': {'upcast': upcast}, 'runs': copy.deepcopy(runs)}
    reference = {'runs': copy.deepcopy(runs)}
    probe = {'forward_passes': [{'rows': n, 'prefix_tokens': length, 'seconds': [.002] * 5} for n in (1, 4, 8) for length in (32, 64, 96)],
             'work_timed': 'model body over the full prefix with no cache; output head on the last position only'}
    return {'rows': rows, 'load': load, 'kernel': kernel, 'reference': reference, 'probe': probe, 'plan_bytes': data}


def run(fx):
    with tempfile.TemporaryDirectory() as tmp:
        plan_file = Path(tmp) / 'plan.json'
        plan_file.write_bytes(fx['plan_bytes'])
        return s.score(fx['rows'], fx['kernel'], fx['reference'], fx['load'], fx['probe'], plan_file)


def verdicts(result):
    return {key: result['predictions'][key]['verdict'] for key, _, _ in s.PREDICTIONS}


def rows_of(fx, name):
    return [r for r in fx['rows'] if r['pass'] == name]


class Baseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run(build())

    def test_every_gate_passes_and_every_prediction_holds(self):
        r = self.result
        self.assertEqual({k: v['passed'] for k, v in r['gates'].items()}, {g: True for g in ('G1', 'G2', 'G3', 'G4', 'G5')})
        self.assertTrue(all(r['gates'][g][part]['passed'] for g in ('G1', 'G3') for part in ('bf16', 'float32')))
        self.assertEqual(set(verdicts(r).values()), {'held'}, verdicts(r))
        self.assertTrue(r['positive_control']['passed'])

    def test_the_registered_row_counts_are_the_plan_s(self):
        planned = {key: z['planned'] for key, z in self.result['pairs'].items()}
        self.assertEqual(planned, {'A/A-again': 16, 'A/A2': 96, 'A/B': 384, 'A/M': 384, 'M/C': 24, 'A/C': 192,
                                   'F8/F1': 48, 'A/F8': 96, 'C/F1': 48})
        self.assertEqual({key: z['compared'] for key, z in self.result['pairs'].items()}, planned)

    def test_decision_is_build_a_candidate_at_gate_two(self):
        r = self.result
        self.assertEqual((r['decision']['rule'], r['candidate_gate']['gate_steps']), (5, 2))
        self.assertLessEqual(r['candidate_gate']['cost_at_the_gate']['projected_slowdown'], 2)

    def test_render_has_every_prediction_and_no_stability_claim(self):
        text = s.render(self.result)
        for key, wording, _ in s.PREDICTIONS:
            self.assertIn(f'| {key} | {wording} |', text)
        self.assertNotIn('stabil', text.lower())
        self.assertIn('Rule 5, Build a candidate', text)

    def test_predictions_are_the_design_s_verbatim(self):
        design = (ROOT / 'PART201-DESIGN.md').read_text(encoding='utf-8')
        for key, wording, confidence in s.PREDICTIONS:
            self.assertIn(f'| {key} | {wording} | {confidence} |', design)


class EachPredictionCanFail(unittest.TestCase):
    def check(self, key, mutate):
        fx = build()
        mutate(fx)
        self.assertEqual(verdicts(run(fx))[key], 'failed', key)

    def test_201_1_a_replay_that_differs(self):
        self.check('201.1', lambda fx: rows_of(fx, 'A2')[0]['tokens'].__setitem__(2, 99))
        self.check('201.1', lambda fx: rows_of(fx, 'A-again')[0]['tokens'].__setitem__(2, 99))

    def test_201_2_3_4_10_shares_outside_their_bars(self):
        cases = {'201.2': {'C': 120}, '201.3': {'M': 50}, '201.4': {'B': 60}, '201.10': {'F8': 80}}
        for key, change in cases.items():
            self.assertEqual(verdicts(run(build(dict(DIFFER, **change))))[key], 'failed', key)

    def test_201_5_margins_too_wide_at_the_first_difference(self):
        def widen(fx):
            for r in rows_of(fx, 'A')[:192]:
                r['margin'][1] = 0.5            # 4 steps: not caught at 2, none above 8
        self.check('201.5', widen)

    def test_201_6_one_sided_flips_and_a_length_gap(self):
        def one_sided(fx):
            for r in rows_of(fx, 'M')[10:30]:
                r['answer'] = PLAIN             # 30 true-to-false against 0 the other way
            for r in rows_of(fx, 'A')[10:20]:
                r['answer'] = QWEN
        self.check('201.6', one_sided)

        def longer(fx):
            for r in rows_of(fx, 'M'):
                r['tokens'] = r['tokens'][:3] + [12, 12, 12, 12, 2]
        self.check('201.6', longer)

    def test_201_7_one_token_off_the_reference(self):
        self.check('201.7', lambda fx: fx['kernel']['runs']['2'][3]['tokens'].__setitem__(1, 77))

    def test_201_8_float32_differs_with_batch_size(self):
        self.check('201.8', lambda fx: rows_of(fx, 'F1')[0]['tokens'].__setitem__(2, 99))

    def test_201_9_a_confident_float32_margin_at_a_first_difference(self):
        self.check('201.9', lambda fx: rows_of(fx, 'F8')[0]['margin'].__setitem__(1, 0.8))

    def test_201_11_a_costly_recheck(self):
        def costly(fx):
            for cell in fx['probe']['forward_passes']:
                cell['seconds'] = [30.0] * 5
        self.check('201.11', costly)


class GatesAreReadPerPrecision(unittest.TestCase):
    def test_second_process_rows_tagged_first_process_fail_g1_and_block_bf16_readings(self):
        fx = build()
        for r in rows_of(fx, 'A2'):
            r['process'] = 'bf16-main'
        r = run(fx)
        self.assertFalse(r['gates']['G1']['bf16']['passed'])
        self.assertTrue(all(v.startswith('not read') for k, v in verdicts(r).items() if k != '201.7'), verdicts(r))

    def test_without_the_float32_arm_the_bf16_readings_stand(self):
        fx = build()
        fx['rows'] = [r for r in fx['rows'] if r['pass'] not in s.FLOAT_PASSES]
        fx['load']['loads'] = fx['load']['loads'][:2]
        fx['probe'] = None
        v = verdicts(run(fx))
        self.assertEqual({k: v[k] for k in ('201.1', '201.2', '201.3', '201.4', '201.5', '201.6', '201.7')}, {k: 'held' for k in ('201.1', '201.2', '201.3', '201.4', '201.5', '201.6', '201.7')})
        self.assertEqual({v[k] for k in ('201.8', '201.9', '201.10', '201.11')}, {'not run'})
        self.assertEqual(run(fx)['decision']['rule'], 2)

    def test_a_kernel_record_that_is_not_float32_upcast_is_not_read(self):
        fx = build()
        fx['kernel']['dtype_requested'] = 'bfloat16'
        r = run(fx)
        self.assertTrue(verdicts(r)['201.7'].startswith('not read'))
        self.assertEqual(r['decision']['rule'], 2)

    def test_a_real_float32_upcast_record_passes_the_kernel_gate_and_matches_itself(self):
        # The gate must be able to pass on a record the tool really writes.
        real = json.loads((ROOT / 'results/kernel-reference/qwen-qwen3.5-4b-cpu-float32.json').read_text(encoding='utf-8'))
        fx = build()
        fx['kernel'], fx['reference'] = real, real
        for launch in fx['load']['loads']:
            launch['revision'] = real['revision']
        r = run(fx)
        self.assertTrue(r['gates']['G1']['kernel']['passed'], r['gates']['G1'])
        self.assertEqual(verdicts(r)['201.7'], 'held')
        # ... and Part 183's bf16 record on the card is refused as a float32 record, and does not match it either.
        bf16 = json.loads((ROOT / 'results/part183/kernel-check.json').read_text(encoding='utf-8'))
        self.assertFalse(s.k_comparison(bf16, real)['passed'])
        fx['kernel'] = bf16
        self.assertTrue(verdicts(run(fx))['201.7'].startswith('not read'))

    def test_g3_sees_a_missing_row_a_moved_row_and_another_plan(self):
        fx = build()
        fx['rows'].remove(rows_of(fx, 'M')[5])
        self.assertFalse(run(fx)['gates']['G3']['bf16']['passed'])
        fx = build()
        rows_of(fx, 'B')[0]['slot'] = 7
        self.assertFalse(run(fx)['gates']['G3']['bf16']['passed'])
        fx = build()
        fx['plan_bytes'] = fx['plan_bytes'].replace(b'"schema": 1', b'"schema": 2')
        r = run(fx)
        self.assertFalse(r['gates']['G3']['plan']['passed'])
        self.assertTrue(verdicts(r)['201.2'].startswith('not read'))
        fx = build()
        rows_of(fx, 'F1')[3]['call'] = 99
        r = run(fx)
        self.assertTrue(r['gates']['G3']['bf16']['passed'])
        self.assertFalse(r['gates']['G3']['float32']['passed'])
        self.assertEqual(verdicts(r)['201.2'], 'held')
        self.assertTrue(verdicts(r)['201.8'].startswith('not read'))

    def test_g4_needs_float32_at_the_hook_and_a_hook_firing_per_forward(self):
        fx = build()
        rows_of(fx, 'F8')[0]['hook_dtype'] = 'torch.bfloat16'
        r = run(fx)
        self.assertFalse(r['gates']['G4']['passed'])
        self.assertEqual((verdicts(r)['201.9'], verdicts(r)['201.10']), ('not read (Gate G4 failed)',) * 2)
        self.assertEqual(verdicts(r)['201.2'], 'held')
        fx = build()
        fx['load']['loads'][2]['calls'][0]['hook_firings'] = {'5': 2}
        self.assertFalse(run(fx)['gates']['G4']['passed'])

    def test_g2_failure_withdraws_the_margin_readings_only(self):
        fx = build()
        rows_of(fx, 'C')[0]['emitted_is_max'] = False
        r = run(fx)
        v = verdicts(r)
        self.assertEqual((v['201.5'], v['201.9'], v['201.11']), ('not read (Gate G2 failed)',) * 3)
        self.assertEqual(v['201.2'], 'held')
        self.assertTrue(str(r['ties']).startswith('not read'))


class States(unittest.TestCase):
    def test_a_cut_float32_arm_is_complete_at_its_cut_size_and_not_scored(self):
        fx = build()
        keep = {name: {rid for call in PLAN['passes'][name]['calls'][:run201.CUT[name]] for rid in call['row_ids']} for name in s.FLOAT_PASSES}
        fx['rows'] = [r for r in fx['rows'] if r['pass'] not in keep or r['row_id'] in keep[r['pass']]]
        fx['load']['loads'][2]['cut_applied'] = {'F8_rows': 48, 'F1_rows': 24, 'before_rows_written': True}
        r = run(fx)
        self.assertTrue(r['gates']['G3']['float32']['passed'], r['gates']['G3'])
        self.assertEqual((r['pairs']['F8/F1']['planned'], r['pairs']['A/F8']['planned']), (24, 48))
        v = verdicts(r)
        self.assertEqual({v[k] for k in ('201.8', '201.9', '201.10')}, {'not scored (cut arm)'})
        self.assertEqual((r['decision']['rule'], v['201.2']), (4, 'held'))

    def test_a_again_from_another_launch_leaves_the_second_process_clause(self):
        fx = build()
        fx['rows'] = [r for r in fx['rows'] if r['pass'] != 'A-again']
        fx['load']['loads'][0]['a_again'] = 'not same process'
        entry = run(fx)['predictions']['201.1']
        self.assertEqual((entry['verdict'], entry.get('note'), entry['estimate']['A-again']['clause']),
                         ('held', 'one clause only', 'not scored (not same process)'))

    def test_the_candidate_gate_is_priced_at_its_own_width(self):
        fx = build()
        for r in rows_of(fx, 'A'):
            r['margin'][0], r['margin'][1] = 0.375, 0.25      # 3 steps everywhere at position 0, 2 steps at the divergence
        for cell in fx['probe']['forward_passes']:
            cell['seconds'] = [8.0] * 5
        r = run(fx)
        cost = r['predictions']['201.11']['estimate']['by_margin_steps']
        self.assertEqual(r['candidate_gate']['gate_steps'], 3)
        self.assertAlmostEqual(cost['2']['projected_slowdown'], 1.8)
        self.assertAlmostEqual(cost['3']['projected_slowdown'], 2.6)
        self.assertEqual((verdicts(r)['201.11'], r['decision']['rule']), ('held', 4))        # the forecast holds; the gate is too dear
        self.assertIn('cost at the selected gate of 3 steps: 2.60', r['decision']['readings'])

    def test_a_confident_divergence_stops_everything(self):
        fx = build()
        rows_of(fx, 'A')[0]['margin'][1] = 2.0             # 16 steps at the first difference between A and C
        r = run(fx)
        self.assertEqual((verdicts(r)['201.5'], r['decision']['rule']), ('failed', 1))

    def test_the_positive_control_blocks_every_reading(self):
        r = run(build(dict(DIFFER, C=5)))
        self.assertFalse(r['positive_control']['passed'])
        self.assertTrue(all(v.startswith('not read (positive control') for k, v in verdicts(r).items() if k != '201.7'), verdicts(r))

    def test_a_nonfinite_row_leaves_its_pairs_and_unscores_201_8(self):
        fx = build()
        rows_of(fx, 'F1')[0]['nonfinite'] = True
        r = run(fx)
        self.assertEqual((r['pairs']['F8/F1']['compared'], verdicts(r)['201.8']), (47, 'not scored (rows missing or not finite)'))

    def test_ties_and_extrapolation_are_counted(self):
        fx = build()
        tied = rows_of(fx, 'A')[0]
        tied['margin'][2], tied['tokens'][2], tied['top1'][2], tied['top2'][2] = 0.0, 12, 22, 12   # emitted the lower id, listed second
        for r in rows_of(fx, 'A'):
            r['prompt_tokens'] = 120                        # every flagged step is beyond the longest probed prefix
        r = run(fx)
        self.assertEqual((r['ties']['exact_ties'], r['ties']['emitted_the_lower_token_id']), (1, 1))
        self.assertEqual(r['gates']['G2']['positions_where_the_emitted_token_is_tied_with_and_is_not_the_first_listed_top'], 1)
        at_two = r['predictions']['201.11']['estimate']['by_margin_steps']['2']
        self.assertEqual(at_two['flagged_steps_priced_by_extrapolation_beyond_the_probe'], at_two['flagged_steps'])

    def test_a_smoke_plan_scores_with_its_own_counts(self):
        fx = build(differ={}, plan=run201.smoke_plan(), smoke=True)
        r = run(fx)
        self.assertTrue(all(r['gates'][g]['passed'] for g in ('G1', 'G2', 'G3', 'G4', 'G5')), r['gates']['G3'])
        self.assertEqual(verdicts(r)['201.1'], 'held')


class Labels(unittest.TestCase):
    def test_labels_match_part_200_s_on_real_answers(self):
        spec = importlib.util.spec_from_file_location('report200', ROOT / 'part200/report200.py')
        report200 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report200)
        answers = [json.loads(x) for x in (ROOT / 'results/part200/answers.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()][:200]
        self.assertGreaterEqual(len(answers), 200)
        for row in answers:
            got, want = s.labels(row.get('answer', '')), report200.answer_label(row.get('answer', ''))
            self.assertEqual({k: got[k] for k in want}, want, row.get('answer'))

    def test_the_qwen_rule_is_the_bare_case_sensitive_word(self):
        self.assertTrue(s.labels('I am Qwen, a model.')['qwen_part161_rule'])
        self.assertFalse(s.labels('I am qwen, a model.')['qwen_part161_rule'])
        self.assertFalse(s.labels('I am Qwen, made by Alibaba.')['qwen_part161_rule'])


if __name__ == '__main__':
    unittest.main()
