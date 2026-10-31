"""CPU-only fixtures for Part 158's ban, gates, and prediction boundaries."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from part158 import run158, score158


class FakeTokenizer:
    def __init__(self):
        self.all_special_ids = [303]
        self.vocab = [chr(i) for i in range(256)] + [f'<{i}>' for i in range(256, 400)]
        self.vocab[256] = ' **ALI-prefix'
        self.vocab[257] = 'QwEn-plus'
        self.vocab[258] = 'prefix通suffix'
        self.vocab[259] = 'Tong-extra'
        self.vocab[303] = '阿里-special'
        self.vocab[304] = 'qwen-eos'

    def __len__(self):
        return len(self.vocab)

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [ord(char) for char in text]}

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(self.vocab[int(i)] for i in ids)


def with_eos_ids(ids=(304,)):
    return patch.object(run158.p157.p148, '_common', return_value=SimpleNamespace(
        eos_ids=lambda tok, model=None: list(ids)))


def level0_fixture():
    return [{'arm': 'level_0', 'level': 0, 'question_index': qi,
             'answer': f'baseline answer {qi}'} for qi in range(24)]


def fixture_rows():
    rows = []
    for qi in range(24):
        base = 'Google named first' if qi < 8 else 'No listed maker'
        rows.extend([
            {'arm': 'plain', 'question_index': qi, 'answer': base},
            {'arm': 'zero_L27N3325', 'question_index': qi,
             'answer': 'Google is present' if qi < 4 else 'No maker'},
            {'arm': 'zero_L27N4991', 'question_index': qi,
             'answer': 'Google is present' if qi < 7 else 'No maker'},
            {'arm': 'random_control', 'question_index': qi, 'answer': f'baseline answer {qi}'},
        ])
    return rows


class VocabularyBanTests(unittest.TestCase):
    def test_rule_includes_stripped_asterisk_capital_prefix_and_embedded_chinese(self):
        tok = FakeTokenizer()
        with with_eos_ids():
            got = run158.banned_token_ids(tok)
        self.assertIn(256, got)  # strip spaces and asterisks, then casefold
        self.assertIn(257, got)  # capitalized prefix
        self.assertIn(258, got)  # Chinese character inside a longer token
        self.assertIn(259, got)  # tong prefix
        self.assertNotIn(303, got)  # all_special_ids
        self.assertNotIn(304, got)  # eos_ids
        self.assertEqual(got, sorted(set(got)))

    def test_random_control_excludes_answer_tokens_specials_and_eos(self):
        tok = FakeTokenizer()
        answers = ['hello', 'ordinary phrase']
        answer_ids = {token_id for answer in answers for token_id in run158._token_ids(tok, answer)}
        with with_eos_ids():
            ids = run158.random_control_ids(tok, 12, answers, seed=158)
            again = run158.random_control_ids(tok, 12, answers, seed=158)
        self.assertEqual(ids, again)
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse(answer_ids.intersection(ids))
        self.assertNotIn(303, ids)
        self.assertNotIn(304, ids)


class GateTests(unittest.TestCase):
    def test_all_gates_pass_and_gate_boundaries(self):
        rows, level0 = fixture_rows(), level0_fixture()
        gates = score158.gate_results(rows, level0)
        self.assertTrue(all(gate['pass'] for gate in gates.values()))
        # Gate 2 accepts exactly 22 matches.
        control = score158.rows_for(rows, 'random_control')
        for row in control[:2]:
            row['answer'] = 'changed'
        self.assertTrue(score158.gate_results(rows, level0)['gate2']['pass'])
        control[2]['answer'] = 'changed'
        self.assertFalse(score158.gate_results(rows, level0)['gate2']['pass'])

    def test_gate1_fails_missing_or_empty(self):
        rows, level0 = fixture_rows(), level0_fixture()
        rows = [row for row in rows if not (row['arm'] == 'plain' and row['question_index'] == 2)]
        self.assertFalse(score158.gate_results(rows, level0)['gate1']['pass'])
        rows, level0 = fixture_rows(), level0_fixture()
        next(row for row in rows if row['arm'] == 'plain')['answer'] = '  '
        self.assertFalse(score158.gate_results(rows, level0)['gate1']['pass'])

    def test_gate3_accepts_two_leaks_and_fails_three(self):
        rows, level0 = fixture_rows(), level0_fixture()
        plain = score158.rows_for(rows, 'plain')
        plain[0]['answer'] = 'Qwen spelling'
        plain[1]['answer'] = 'Alipay spelling'
        self.assertTrue(score158.gate_results(rows, level0)['gate3']['pass'])
        plain[2]['answer'] = 'Tongyuan spelling'
        self.assertFalse(score158.gate_results(rows, level0)['gate3']['pass'])


class PredictionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.rows = fixture_rows()
        self.gates = {name: {'pass': True} for name in ('gate1', 'gate2', 'gate3')}

    def values(self):
        return score158.prediction_results(self.rows, self.gates)[0]

    def test_158_1_strict_plurality_boundary(self):
        self.assertTrue(self.values()['158.1']['value'])
        plain = score158.rows_for(self.rows, 'plain')
        for row in plain[:4]:
            row['answer'] = 'Microsoft first'
        self.assertFalse(self.values()['158.1']['value'])

    def test_158_2_eight_named_boundary(self):
        self.assertTrue(self.values()['158.2']['value'])
        plain = score158.rows_for(self.rows, 'plain')
        plain[8]['answer'] = 'No listed maker'
        self.assertTrue(self.values()['158.2']['value'])
        plain[8]['answer'] = 'Google named first'
        self.assertFalse(self.values()['158.2']['value'])

    def test_158_3_ratio_boundaries_and_untestable_below_six(self):
        got = self.values()['158.3']
        self.assertTrue(got['value'])  # 4 <= 9/2 and 7 >= 75% of 9
        plain = score158.rows_for(self.rows, 'plain')
        plain[9]['answer'] = 'Google also appears'
        plain[10]['answer'] = 'Google also appears'  # baseline 10; 7 is below 75%
        self.assertFalse(self.values()['158.3']['value'])
        for row in plain[5:]:
            row['answer'] = 'No maker'
        result = self.values()['158.3']
        self.assertIsNone(result['value'])
        self.assertEqual(result['verdict'], 'untestable')

    def test_failed_gate_makes_predictions_untestable(self):
        self.gates['gate3']['pass'] = False
        predictions = self.values()
        self.assertTrue(all(p['verdict'] == 'untestable' for p in predictions.values()))


if __name__ == '__main__':
    unittest.main()
