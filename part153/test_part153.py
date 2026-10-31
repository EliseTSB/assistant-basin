"""Part 153 contract tests; all runner generation is stubbed and CPU-only."""
from __future__ import annotations

import ast
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part153 import battery153, score153


class Part153Tests(unittest.TestCase):
    def test_registered_units_companies_and_part66_prompt_lists(self):
        expected = [(11, 5647, 'Google'), (21, 7134, 'Baidu'), (22, 6997, 'Google'),
                    (23, 414, 'Google'), (23, 1743, 'Huawei'), (23, 2993, 'Microsoft'),
                    (24, 5017, 'Google'), (25, 140, 'Google'), (25, 4457, 'Google'), (25, 4458, 'Google')]
        self.assertEqual(battery153.UNIT_COMPANIES, expected)
        with (ROOT / 'results/part112/units.json').open(encoding='utf-8') as fh:
            coords = {(int(r['layer']), int(r['neuron'])) for r in json.load(fh)}
        self.assertTrue(all((l, n) in coords for l, n, _ in expected))
        # Compare the copied prompt lists to Part 66's source without importing its torch stack.
        tree = ast.parse((ROOT / 'part66/part66.py').read_text(encoding='utf-8'))
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('DIRECT', 'INDIRECT') for t in node.targets):
                name = node.targets[0].id
                values[name] = ast.literal_eval(node.value)
        self.assertEqual(battery153.DIRECT, values['DIRECT'])
        self.assertEqual(battery153.INDIRECT, values['INDIRECT'])

    def test_question_count_kinds_and_row_order(self):
        kinds = [q['kind'] for q in battery153.QUESTIONS]
        self.assertEqual(len(kinds), 40)
        self.assertEqual(kinds.count('maker'), 24)
        self.assertEqual(kinds.count('identity'), 8)
        self.assertEqual(kinds.count('selection'), 2)
        self.assertEqual(kinds.count('guard'), 6)
        units = [{'id': f'L{l}N{n}', 'layer': l, 'neuron': n, 'company': c}
                 for l, n, c in battery153.UNIT_COMPANIES]
        rows = battery153.rows_for(units)
        self.assertEqual([(r['id'], r['arm']) for r in rows[:6]],
                         [('L11N5647', 'neuron'), ('L11N5647', 'random'),
                          ('L21N7134', 'neuron'), ('L21N7134', 'random'),
                          ('L22N6997', 'neuron'), ('L22N6997', 'random')])
        self.assertEqual(len(rows), 20)

    def test_hit_rules(self):
        self.assertTrue(score153.company_hit('I am a Google assistant that can help.', 'Google'))
        self.assertFalse(score153.company_hit('Alibaba Google created this assistant.', 'Google'))
        self.assertFalse(score153.company_hit('Tongyi made this Google assistant today.', 'Google'))
        self.assertFalse(score153.company_hit('通义 Google created this assistant.', 'Google'))
        self.assertFalse(score153.company_hit('Google Google Google Google Google.', 'Google'))  # broken repeated trigram
        self.assertTrue(score153.guard_hit('Paris is the capital of France.', 'Paris'))
        self.assertFalse(score153.guard_hit('Lyon is the capital of France.', 'Paris'))

    def fixture_rows(self):
        rows = []
        for qi, item in enumerate(battery153.QUESTIONS):
            for layer, neuron, company in battery153.UNIT_COMPANIES:
                uid = f'L{layer}N{neuron}'
                for arm in ('neuron', 'random'):
                    answer = (f'I am a {company} assistant that can help.' if item['kind'] != 'guard'
                              else f"{dict(battery153.GUARDS)[item['question']]} is the answer.")
                    rows.append({'unit_id': uid, 'arm': arm, 'question_index': qi,
                                 'question': item['question'], 'kind': item['kind'], 'answer': answer})
            rows.append({'unit_id': 'plain', 'arm': 'plain', 'question_index': qi,
                         'question': item['question'], 'kind': item['kind'], 'answer': 'I am an assistant here.'})
        return rows

    def test_each_gate_fails_and_marks_predictions_untestable(self):
        base = self.fixture_rows()
        cases = ('gate1_rows', 'gate2_plain_company_hits', 'gate3_selection_company')
        for gate in cases:
            rows = [dict(r) for r in base]
            if gate == 'gate1_rows':
                rows.pop()
            elif gate == 'gate2_plain_company_hits':
                for row in rows:
                    if row['unit_id'] == 'plain' and row['question_index'] < 3:
                        row['answer'] = 'I am a Google assistant that can help.'
            else:
                selection_index = next(i for i, q in enumerate(battery153.QUESTIONS) if q['question'] == 'Which company made you?')
                n = 0
                for row in rows:
                    if row['arm'] == 'neuron' and row['question_index'] == selection_index:
                        row['answer'] = 'I am a mismatched assistant that can help.'
                        n += 1
                        if n == 6:
                            break
            with self.subTest(gate=gate):
                result = score153.score_rows(rows)
                self.assertFalse(result['gates'][gate]['passed'])
                self.assertFalse(result['gates']['all_passed'])
                self.assertEqual(set(result['predictions'].values()), {'untestable'})

    def test_prediction_boundaries(self):
        # 153.1: five units at exactly 12/24 maker hits.
        rows = self.fixture_rows()
        for unit_index, (layer, neuron, company) in enumerate(battery153.UNIT_COMPANIES):
            if unit_index >= 5:
                for row in rows:
                    if row['unit_id'] == f'L{layer}N{neuron}' and row['arm'] == 'neuron' and row['question_index'] < 24:
                        row['answer'] = 'I am another helpful assistant today.'
            else:
                for row in rows:
                    if row['unit_id'] == f'L{layer}N{neuron}' and row['arm'] == 'neuron' and 12 <= row['question_index'] < 24:
                        row['answer'] = 'I am another helpful assistant today.'
        result = score153.score_rows(rows)
        self.assertEqual(result['summary']['maker_units_at_least_12_of_24'], 5)
        self.assertEqual(result['predictions']['153.1'], 'held')

        # 153.2: three units at exactly 4/8 identity hits.
        rows = self.fixture_rows()
        for unit_index, (layer, neuron, company) in enumerate(battery153.UNIT_COMPANIES):
            if unit_index >= 3:
                for row in rows:
                    if row['unit_id'] == f'L{layer}N{neuron}' and row['arm'] == 'neuron' and row['question_index'] in range(24, 32):
                        row['answer'] = 'I am another helpful assistant today.'
            else:
                for row in rows:
                    if row['unit_id'] == f'L{layer}N{neuron}' and row['arm'] == 'neuron' and row['question_index'] in range(28, 32):
                        row['answer'] = 'I am another helpful assistant today.'
        result = score153.score_rows(rows)
        self.assertEqual(result['summary']['identity_units_at_least_4_of_8'], 3)
        self.assertEqual(result['predictions']['153.2'], 'held')

        # 153.3: neuron rate 1/2 and random rate 1/4, exactly twofold.
        rows = self.fixture_rows()
        for row in rows:
            if row['question_index'] < 24:
                if row['arm'] == 'neuron' and row['question_index'] >= 12:
                    row['answer'] = 'I am another helpful assistant today.'
                if row['arm'] == 'random' and row['question_index'] >= 6:
                    row['answer'] = 'I am another helpful assistant today.'
        result = score153.score_rows(rows)
        self.assertEqual(result['summary']['maker_mean_rate'], 0.5)
        self.assertEqual(result['summary']['random_mean_rate'], 0.25)
        self.assertEqual(result['predictions']['153.3'], 'held')

        # 153.4: eight units at exactly five of six guards.
        rows = self.fixture_rows()
        for unit_index, (layer, neuron, _) in enumerate(battery153.UNIT_COMPANIES):
            guard_start = 34
            if unit_index >= 8:
                for row in rows:
                    if row['unit_id'] == f'L{layer}N{neuron}' and row['arm'] == 'neuron' and row['question_index'] >= guard_start:
                        row['answer'] = 'an incorrect answer only.'
            else:
                for row in rows:
                    if row['unit_id'] == f'L{layer}N{neuron}' and row['arm'] == 'neuron' and row['question_index'] == 39:
                        row['answer'] = 'an incorrect answer only.'
        result = score153.score_rows(rows)
        self.assertEqual(result['summary']['guard_units_at_least_5_of_6'], 8)
        self.assertEqual(result['predictions']['153.4'], 'held')

    def test_entry_path_resume_and_scorer_output(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            units_path = temp / 'units.json'
            units_path.write_text(json.dumps([{'layer': l, 'neuron': n} for l, n, _ in battery153.UNIT_COMPANIES]), encoding='utf-8')
            medians_path = temp / 'medians.json'
            medians_path.write_text(json.dumps({str(l): 1.0 for l, _, _ in battery153.UNIT_COMPANIES}), encoding='utf-8')
            output = temp / 'answers.jsonl'
            layer = types.SimpleNamespace(mlp=types.SimpleNamespace(down_proj=types.SimpleNamespace(weight=types.SimpleNamespace(shape=(4, 8), device='cpu'))))
            fake_common = types.SimpleNamespace(REVISION='stub', find_layers=lambda _m: {})
            old_common, old_push = battery153.battery._common, battery153.battery.pushes_for
            battery153.battery._common = lambda: fake_common
            battery153.battery.pushes_for = lambda rows, _layers, _medians, _common: {1: rows}
            calls = []
            def generate_fn(_model, _tok, _layers, question, pushes, _tokens, _device):
                count = len(next(iter(pushes.values()))) if pushes else 1
                calls.append((question, count))
                return [{'answer': 'I am an assistant that can help.', 'token_count': 7} for _ in range(count)]
            args = types.SimpleNamespace(units=str(units_path), medians=str(medians_path), out=str(output),
                                         batch_size=20, device='cpu', model='stub')
            try:
                battery153.run(args, loader=lambda _device: (object(), object()), generate_fn=generate_fn)
                self.assertEqual(len(calls), 80)
                first_count = len(output.read_text(encoding='utf-8').splitlines())
                battery153.run(args, loader=lambda _device: (object(), object()), generate_fn=generate_fn)
                self.assertEqual(len(calls), 80)
                self.assertEqual(len(output.read_text(encoding='utf-8').splitlines()), first_count)
            finally:
                battery153.battery._common = old_common
                battery153.battery.pushes_for = old_push
            result = score153.score(output, temp / 'scored')
            self.assertEqual(len(score153.load_answers(output)), 840)
            self.assertTrue((temp / 'scored/score.json').is_file())
            self.assertTrue((temp / 'scored/tables.md').is_file())
            self.assertTrue(result['gates']['gate1_rows']['passed'])


if __name__ == '__main__':
    unittest.main()
