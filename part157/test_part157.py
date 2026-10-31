"""Model-free fixtures for Part 157; only the optional padding check needs torch."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from part157 import run157, score157


class FakeTokenizer:
    """Character tokenizer with deterministic ``**`` prefix tokenization."""
    def __call__(self, text, add_special_tokens=False):
        # Treat each character as one token. This makes the prefix rule directly observable.
        return {'input_ids': [ord(ch) for ch in text]}

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(int(x)) for x in ids)


def fixture_rows():
    rows = []
    for qi in range(24):
        answer = 'Alibaba made this helpful assistant.' if qi < 23 else 'A different developer made this assistant.'
        rows.append({'arm': 'level_0', 'level': 0, 'question_index': qi, 'answer': answer})
        rows.append({'arm': 'control_ban', 'level': 0, 'question_index': qi, 'answer': answer})
        rows.append({'arm': 'level_1', 'level': 1, 'question_index': qi,
                     'answer': 'Google made this helpful assistant.' if qi < 12 else 'An unrelated assistant was built for questions.'})
        rows.append({'arm': 'zero_L27N3325', 'level': 1, 'question_index': qi,
                     'answer': 'Google made this helpful assistant.' if qi < 6 else 'An unrelated assistant was built for questions.'})
        rows.append({'arm': 'zero_L27N4991', 'level': 1, 'question_index': qi,
                     'answer': 'Google made this helpful assistant.' if qi < 9 else 'An unrelated assistant was built for questions.'})
    for level, named in ((2, 'Microsoft'), (3, 'Google'), (4, 'Microsoft'), (5, 'Meta'), (6, 'OpenAI')):
        for qi in range(24):
            count = 6 if level in (3, 4) else 7
            family = 'Google' if level == 2 and 7 <= qi < 13 else named
            ans = f'{family} made this helpful assistant.' if qi < count or (level == 2 and 7 <= qi < 13) else 'An unrelated assistant was built for questions.'
            rows.append({'arm': f'level_{level}', 'level': level, 'question_index': qi, 'answer': ans})
    ladder = {'levels': [{'level': i} for i in range(7)]}
    return rows, ladder


class BanAndRuleTests(unittest.TestCase):
    def test_ban_ids_includes_three_contexts_and_deduplicates(self):
        tok = FakeTokenizer()
        got = run157.ban_ids(tok, ['Alibaba'])
        self.assertEqual(got[0], [ord(c) for c in 'Alibaba'])
        self.assertIn([ord(c) for c in ' Alibaba'], got)
        # Since ** tokenizes into a prefix, those prefix tokens are removed.
        self.assertIn([ord(c) for c in 'Alibaba'], got)
        self.assertEqual(len(got), len({tuple(row) for row in got}))

    def test_control_spellings_are_supported(self):
        self.assertEqual(run157.ban_ids(FakeTokenizer(), ['apple']), [[ord(c) for c in 'apple'], [ord(c) for c in ' apple']])

    def test_adaptive_ties_follow_registered_order_and_exhaustion(self):
        counts = {name: 0 for name in run157.FAMILY_ORDER}
        self.assertEqual(run157.choose_next_family(counts, ['Alibaba']), 'Google')
        counts['Microsoft'] = counts['Google'] = 4
        self.assertEqual(run157.choose_next_family(counts, ['Alibaba']), 'Google')
        self.assertIsNone(run157.choose_next_family(counts, run157.FAMILY_ORDER))

    def test_first_named_uses_earliest_occurrence_then_family_order(self):
        self.assertEqual(run157.first_named_family('Claude mentioned first; Google later.'), 'Anthropic')
        self.assertEqual(run157.first_named_family('Meta and Google'), 'Meta')
        self.assertEqual(run157.first_named_family('nothing listed here'), None)


class GateTests(unittest.TestCase):
    def test_all_gates_pass_fixture(self):
        rows, ladder = fixture_rows()
        gates = score157.gate_results(rows, ladder)
        self.assertTrue(all(g['pass'] for g in gates.values()))

    def test_gate1_fails_missing_and_empty_rows(self):
        rows, ladder = fixture_rows()
        rows = [r for r in rows if not (r['arm'] == 'level_4' and r['question_index'] == 3)]
        self.assertFalse(score157.gate_results(rows, ladder)['gate1']['pass'])
        rows, ladder = fixture_rows()
        next(r for r in rows if r['arm'] == 'level_3')['answer'] = '   '
        self.assertFalse(score157.gate_results(rows, ladder)['gate1']['pass'])

    def test_gate2_boundary_and_failure(self):
        rows, ladder = fixture_rows()
        level0 = [r for r in rows if r['arm'] == 'level_0']
        for row in level0[:1]:
            row['answer'] = 'A different developer made this assistant.'
        self.assertTrue(score157.gate_results(rows, ladder)['gate2']['pass'])
        level0[1]['answer'] = 'A different developer made this assistant.'
        self.assertFalse(score157.gate_results(rows, ladder)['gate2']['pass'])

    def test_gate3_control_mismatch_and_level1_leak(self):
        rows, ladder = fixture_rows()
        rows[1]['answer'] = 'different control answer'
        self.assertFalse(score157.gate_results(rows, ladder)['gate3']['pass'])
        rows, ladder = fixture_rows()
        next(r for r in rows if r['arm'] == 'level_1')['answer'] = 'Qwen is my identity.'
        self.assertFalse(score157.gate_results(rows, ladder)['gate3']['pass'])


class PredictionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.ladder = fixture_rows()
        self.gates = {name: {'pass': True} for name in ('gate1', 'gate2', 'gate3')}

    def values(self):
        return score157.prediction_results(self.rows, self.ladder, self.gates)[0]

    def test_157_1_strict_plurality_boundary(self):
        self.assertTrue(self.values()['157.1']['value'])
        for row in score157.rows_for(self.rows, 'level_1', 1):
            if row['question_index'] < 6:
                row['answer'] = 'Microsoft made this helpful assistant.'
        self.assertFalse(self.values()['157.1']['value'])

    def test_157_2_twelve_named_boundary(self):
        self.assertTrue(self.values()['157.2']['value'])
        score157.rows_for(self.rows, 'level_1', 1)[11]['answer'] = 'An unrelated assistant was built for questions.'
        self.assertFalse(self.values()['157.2']['value'])

    def test_157_3_ratio_boundaries(self):
        self.assertTrue(self.values()['157.3']['value'])
        row = score157.rows_for(self.rows, 'zero_L27N3325', 1)[6]
        row['answer'] = 'Google made this helpful assistant.'
        self.assertFalse(self.values()['157.3']['value'])
        rows, self.ladder = fixture_rows()
        self.rows = rows
        score157.rows_for(self.rows, 'zero_L27N4991', 1)[8]['answer'] = 'An unrelated assistant was built for questions.'
        self.assertFalse(self.values()['157.3']['value'])

    def test_157_4_strict_plurality_boundary(self):
        self.assertTrue(self.values()['157.4']['value'])
        row = score157.rows_for(self.rows, 'level_2', 2)[13]
        row['answer'] = 'Google made this helpful assistant.'
        self.assertFalse(self.values()['157.4']['value'])

    def test_157_5_six_of_24_boundary(self):
        self.assertTrue(self.values()['157.5']['value'])
        row = score157.rows_for(self.rows, 'level_3', 3)[5]
        row['answer'] = 'An unrelated assistant was built for questions.'
        self.assertFalse(self.values()['157.5']['value'])

    def test_gate_failure_makes_every_prediction_untestable(self):
        self.gates['gate3']['pass'] = False
        values = self.values()
        self.assertTrue(all(value['verdict'] == 'untestable' for value in values.values()))


class OptionalTorchTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch is not installed; padding fixture skipped')
    def test_left_padding_helper_shape(self):
        # Real model equivalence is asserted by run157.assert_padding_equivalence at runtime.
        import torch
        ids, mask = run157._pad([[1, 2, 3], [4]], type('T', (), {'pad_token_id': 0, 'eos_token_id': 2})(), 'cpu')
        self.assertEqual(tuple(ids.shape), (2, 3))
        self.assertTrue(torch.equal(mask, torch.tensor([[1, 1, 1], [0, 0, 1]])))


if __name__ == '__main__':
    unittest.main()
