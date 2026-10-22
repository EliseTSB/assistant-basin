"""CPU contract tests for Part 54 paths using a tiny random Qwen3.5 model."""
from __future__ import annotations

import json
import io
import os
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from unittest.mock import patch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import blind_origin
import part26 as p


QWEN_SNAPSHOT = os.environ.get(
    'QWEN35_SNAPSHOT',
    '/mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')


def tiny_qwen():
    cfg = AutoConfig.from_pretrained(QWEN_SNAPSHOT, local_files_only=True)
    text = dict(cfg.text_config.to_dict())
    text.update(num_hidden_layers=4, hidden_size=64, intermediate_size=96,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                linear_num_key_heads=2, linear_num_value_heads=4,
                linear_key_head_dim=16, linear_value_head_dim=16,
                layer_types=text['layer_types'][:4], tie_word_embeddings=True)
    text.pop('architectures', None)
    text.pop('model_type', None)
    cfg = AutoConfig.for_model('qwen3_5_text', **text)
    model = AutoModelForCausalLM.from_config(cfg, dtype=__import__('torch').float32).eval()
    tok = AutoTokenizer.from_pretrained(QWEN_SNAPSHOT, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model


class Part54Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(os.path.join(QWEN_SNAPSHOT, 'config.json')):
            raise RuntimeError(f'missing local Qwen3.5 config/tokenizer snapshot: {QWEN_SNAPSHOT}')
        cls.tok, cls.model = tiny_qwen()

    @staticmethod
    def write_target_candidate(folder, short, layer=3, unit=7):
        target_path, target_npz_path = p._target_files(folder, short)
        with open(target_path, 'w', encoding='utf-8') as stream:
            json.dump({'twin_candidates_n2742': [{'layer': layer, 'unit': unit, 'beats_null': True}]}, stream)
        with open(target_npz_path, 'wb') as stream:
            stream.write(b'fixture')

    def test_model_table_has_public_pinned_search_targets(self):
        expected = {
            'olmo2-7b': ('allenai/OLMo-2-1124-7B-Instruct', '470b1fba1ae01581f270116362ee4aa1b97f4c84'),
            'qwen3-4b': ('Qwen/Qwen3-4B', '1cfa9a7208912126459214e8b04321603b3df60c'),
            'phi4-mini': ('microsoft/Phi-4-mini-instruct', 'cfbefacb99257ffa30c83adab238a50856ac3083'),
            'granite3-2b': ('ibm-granite/granite-3.3-2b-instruct', '707f574c62054322f6b5b04b6d075f0a8f05e0f0'),
        }
        for alias, pair in expected.items():
            with self.subTest(alias=alias):
                row = p.MODELS[alias]
                self.assertEqual((row['id'], row['revision']), pair)
                self.assertEqual(p._canonical_target(alias), (alias, *pair))

    def test_steps_can_dispatch_a_twin_ladder(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, 'source.json'), 'w', encoding='utf-8') as stream:
                json.dump({}, stream)
            with patch.object(p, '_load_target', return_value=(self.tok, self.model)), \
                    patch.object(p, 'run_ladder') as ladder:
                p.main(['--steps', 'ladder', '--targets', 'smollm3-3b', '--clamp-unit', '3:7',
                        '--out', td, '--device', 'cpu'])
            ladder.assert_called_once()

    def test_ladder_validates_named_twin_candidate_before_running(self):
        with tempfile.TemporaryDirectory() as td:
            target_path, _ = p._target_files(td, 'smollm3-3b')
            with open(target_path, 'w', encoding='utf-8') as stream:
                json.dump({'twin_candidates_n2742': [{'layer': 1, 'unit': 2, 'beats_null': True}]}, stream)
            _, target_npz_path = p._target_files(td, 'smollm3-3b')
            with open(target_npz_path, 'wb') as stream:
                stream.write(b'fixture')
            args = Namespace(out=td, clamp_unit='3:7', samples=1, control_count=3,
                             question='seed question', clamp_mult=9.0, seed=54,
                             max_new=1, _ladder_mode=False, force=False)
            with patch.object(p, 'run_clamp') as clamp:
                with self.assertRaisesRegex(ValueError, 'not a twin candidate'):
                    p.run_ladder(args, self.model, self.tok, p.MODELS['smollm3-3b']['id'],
                                 p.MODELS['smollm3-3b']['revision'], 'smollm3-3b')
                clamp.assert_not_called()

    def test_ladder_resumes_partial_run_and_refuses_completed_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            self.write_target_candidate(td, 'example')
            args = Namespace(out=td, clamp_unit='3:7', samples=1, control_count=3,
                             question='seed question', clamp_mult=9.0, seed=54,
                             max_new=1, _ladder_mode=False, force=False)
            fail_once = {'path': None, 'done': False}
            calls = []

            def fake_clamp(step_args, _model, _tok, model_id, revision, short):
                path = p._clamp_output_path(step_args, short, 3, 7)
                calls.append(path)
                if len(calls) == 5:
                    fail_once['path'] = path
                if path == fail_once['path'] and not fail_once['done']:
                    fail_once['done'] = True
                    raise RuntimeError('simulated interruption')
                with open(path, 'w', encoding='utf-8') as stream:
                    json.dump({'model': model_id, 'revision': revision}, stream)

            with patch.object(p, 'run_clamp', side_effect=fake_clamp):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    p.run_ladder(args, self.model, self.tok, 'example/model', 'r1', 'example')
                completed = p.run_ladder(args, self.model, self.tok, 'example/model', 'r1', 'example')
                self.assertEqual(completed['status'], 'complete')
                with self.assertRaisesRegex(FileExistsError, 'already complete'):
                    p.run_ladder(args, self.model, self.tok, 'example/model', 'r1', 'example')
                counts = {path: calls.count(path) for path in set(calls)}
                self.assertEqual(len(counts), 21)
                self.assertEqual(sorted(counts.values()).count(2), 1)
                self.assertEqual(sorted(counts.values()).count(1), 20)
                calls_before_force = len(calls)
                args.force = True
                rerun = p.run_ladder(args, self.model, self.tok, 'example/model', 'r1', 'example')
                self.assertEqual(rerun['status'], 'complete')
                self.assertEqual(len(calls) - calls_before_force, 21)

            self.assertEqual(len(completed['completed']), 21)
            self.assertEqual(len(set(completed['completed'])), 21)

    def test_blind_rows_and_collect_tally_keep_models_separate(self):
        with tempfile.TemporaryDirectory() as td:
            for short, condition, answer in (
                    ('qwen35-4b', 'twin', 'A different planet.'),
                    ('olmo2-1b', 'twin', 'Earth.') ):
                path = os.path.join(td, f'clamp-{short}-L3u7-x2-qabc123.json')
                with open(path, 'w', encoding='utf-8') as stream:
                    json.dump({'short_name': short, 'question': 'Who made you?', 'clamp_mult': 2.0,
                               'conditions': [{'condition': condition, 'unit': 7, 'answers': [answer]}]}, stream)
            rows = blind_origin.answers(td)
            self.assertEqual({row['model'] for row in rows}, {'qwen35-4b', 'olmo2-1b'})
            key = {f'ans|{i}': {k: v for k, v in row.items() if k != 'text'}
                   for i, row in enumerate(rows)}
            labels = {'ans|0': ['OFFWORLD', 'OK'], 'ans|1': ['EARTH', 'OK']}
            tally = blind_origin.group_tally(key, labels, by_cond=True)
            self.assertEqual(len(tally), 2)
            self.assertEqual({group[0] for group in tally}, {'qwen35-4b', 'olmo2-1b'})
            self.assertEqual({group[2] for group in tally}, {'twin'})
            rendered = io.StringIO()
            with redirect_stdout(rendered):
                blind_origin.print_tally(tally)
            self.assertIn('model | question | kind | dose', rendered.getvalue())
            self.assertIn('qwen35-4b', rendered.getvalue())
            self.assertIn('olmo2-1b', rendered.getvalue())

    def test_legacy_single_model_tally_keeps_the_part26_table_shape(self):
        key = {'ans|0': {'question': 'Who made you?', 'kind': 'unclamped',
                         'cond': 'unclamped', 'dose': 2.0}}
        tally = blind_origin.group_tally(key, {'ans|0': ['NEITHER', 'OK']})
        rendered = io.StringIO()
        with redirect_stdout(rendered):
            blind_origin.print_tally(tally)
        self.assertEqual(rendered.getvalue().splitlines(), [
            'question | kind | dose | n | EARTH OFFWORLD NEITHER | broken',
            'all | unclamped | 2.0 | 1 | 0 0 1 | 0 | unl 0',
        ])

    def test_collect_parser_restores_by_cond_and_label_parser_accepts_punctuation(self):
        args = blind_origin.parser().parse_args(['collect', 'BID', '--by-cond'])
        self.assertTrue(args.by_cond)
        self.assertEqual(blind_origin.parse_labels('EARTH OK.'), ['EARTH', 'OK'])

    def test_submit_refuses_an_empty_answer_folder_before_api_call(self):
        with tempfile.TemporaryDirectory() as td, patch.object(blind_origin, 'call') as api:
            with self.assertRaisesRegex(SystemExit, 'no clamp answers found'):
                blind_origin.main(['submit', '--folder', td])
            api.assert_not_called()

    def test_ladder_plan_lists_all_registered_steps_and_generation_count(self):
        plan = p.ladder_plan(10, 6978, samples=8, control_count=3)
        self.assertEqual(len(plan), 21)
        self.assertEqual([row['clamp_mult'] for row in plan[:7]], list(p.ORIGIN_DOSES))
        self.assertEqual([row['question'] for row in plan[:7]], [p.ORIGIN_QUESTIONS[0]] * 7)
        self.assertEqual(sum(row['generations'] for row in plan), 480)
        self.assertEqual(sum(row['control_count'] > 0 for row in plan), 6)

    def test_candidate_write_evidence_uses_exact_normalized_top30(self):
        layers = __import__('neuron_report').find_text_layers(self.model)[1]
        writes = p.candidate_writes_top30(
            self.tok, self.model, layers, [{'layer': 1, 'unit': 7}], 'cpu', 14.5, unit_chunk=1)
        tokens = writes['1:7']
        self.assertEqual(len(tokens), 30)
        self.assertTrue(all(isinstance(token, str) for token in tokens))
        self.assertTrue(all(token == token.lower() for token in tokens))

    def test_ladder_driver_writes_manifest_and_restores_arguments(self):
        with tempfile.TemporaryDirectory() as td, patch.object(p, 'run_clamp') as clamp:
            seen = []
            self.write_target_candidate(td, 'example')
            def record_step(args, *_rest):
                path = p._clamp_output_path(args, 'example', 3, 7)
                seen.append((args.question, args.clamp_mult, path))
                with open(path, 'w', encoding='utf-8') as stream:
                    json.dump({'question': args.question, 'clamp_mult': args.clamp_mult}, stream)
            clamp.side_effect = record_step
            args = Namespace(out=td, clamp_unit='3:7', samples=8, control_count=3,
                             question='seed question', clamp_mult=9.0, seed=54,
                             max_new=64, _ladder_mode=False)
            manifest = p.run_ladder(args, self.model, self.tok, 'example/model', 'r1', 'example')
            path = os.path.join(td, 'ladder-example-L3u7.json')
            with open(path, encoding='utf-8') as stream:
                saved = json.load(stream)
            self.assertEqual(clamp.call_count, 21)
            self.assertEqual(saved['status'], 'complete')
            self.assertEqual(saved['expected_generations'], 480)
            self.assertEqual(len(manifest['completed']), 21)
            self.assertEqual(len(set(manifest['completed'])), 21)
            self.assertEqual([(q, dose) for q, dose, _ in seen],
                             [(q, dose) for q in p.ORIGIN_QUESTIONS for dose in p.ORIGIN_DOSES])
            self.assertEqual(len({path for _, _, path in seen}), 21)
            self.assertEqual((args.question, args.clamp_mult, args.control_count,
                              args._ladder_mode), ('seed question', 9.0, 3, False))

    def test_source_clamp_runs_on_tiny_model_and_scales_from_alien_frames(self):
        with tempfile.TemporaryDirectory() as td, patch.object(p, 'SOURCE_LAYER', 1), \
                patch.object(p, 'SOURCE_N2742', 7):
            frame_path = os.path.join(td, 'natural-firing.json')
            passages = [
                {'id': f'alien_q{i:02d}', 'group': 'alien',
                 'text': f'An extraterrestrial traveler from another planet moved a blue stone number {i}.' ,
                 'activation': {'max': float(i + 1)}}
                for i in range(30)
            ]
            with open(frame_path, 'w', encoding='utf-8') as stream:
                json.dump({'model': p.SOURCE_MODEL, 'revision': p.SOURCE_REVISION,
                           'layer': 1, 'neuron': 7,
                           'natural_firing': {'per_passage': passages}}, stream)

            args = Namespace(out=td, alien_frames=frame_path, clamp_unit='1:7', seed=54,
                             control_count=2, clamp_mult=-3.0, question='Who made you?',
                             samples=1, max_new=1, device='cpu', max_gpu_gb=14.5,
                             _ladder_mode=True)
            with patch.object(p, 'SOURCE_LAYER', 1), patch.object(p, 'SOURCE_N2742', 7):
                down = __import__('neuron_report').find_text_layers(self.model)[1][1].mlp.down_proj
                self.assertEqual(down.in_features, down.weight.shape[1])
                self.assertEqual(down.in_features, self.model.config.intermediate_size)
                result = p.run_clamp(args, self.model, self.tok, p.SOURCE_MODEL,
                                     p.SOURCE_REVISION, p.SOURCE_SHORT)
            self.assertEqual([row['condition'] for row in result['conditions']], ['unclamped', 'twin'])
            self.assertEqual(result['random_controls']['count'], 0)
            self.assertEqual(result['scale_frames']['count'], 30)
            self.assertGreater(result['best_twin']['largest_activation'], 0)
            twin = next(row for row in result['conditions'] if row['condition'] == 'twin')
            self.assertAlmostEqual(twin['set_value'], -3.0 * result['best_twin']['largest_activation'])
            self.assertIn('identical for each condition', result['seed_policy'])
            for condition in ('unclamped', 'twin', 'control-3'):
                across_ladder = {
                    p._condition_seed(args.seed, condition)
                    for _question in p.ORIGIN_QUESTIONS for _dose in p.ORIGIN_DOSES
                }
                self.assertEqual(len(across_ladder), 1)
            self.assertNotEqual(p._condition_seed(args.seed, 'unclamped'),
                                p._condition_seed(args.seed, 'twin'))

            args.clamp_mult = -2.0
            result = p.run_clamp(args, self.model, self.tok, p.SOURCE_MODEL,
                                 p.SOURCE_REVISION, p.SOURCE_SHORT)
            self.assertEqual(len(result['conditions']), 4)
            self.assertEqual(result['random_controls']['count'], 2)
            for row in result['conditions'][2:]:
                expected = -2.0 * result['random_controls']['largest_activations'][str(row['unit'])]
                self.assertAlmostEqual(row['set_value'], expected)

            args.clamp_mult = 1.5
            result = p.run_clamp(args, self.model, self.tok, p.SOURCE_MODEL,
                                 p.SOURCE_REVISION, p.SOURCE_SHORT)
            self.assertEqual(result['random_controls']['count'], 2)

    def test_blind_reader_accepts_model_suffixed_clamp_files_and_folder_option(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'clamp-phi4-mini-L3u17-x1.5-q82adf2.json')
            with open(path, 'w', encoding='utf-8') as stream:
                json.dump({'question': 'Where do you come from?', 'clamp_mult': 1.5,
                           'best_twin': {'layer': 3, 'unit': 17},
                           'conditions': [{'condition': 'unclamped', 'unit': None,
                                           'answers': ['I was made by researchers.']},
                                          {'condition': 'twin', 'unit': 17,
                                           'answers': ['I came from another planet.']},
                                          {'condition': 'control-3', 'unit': 3,
                                           'answers': ['A regular response.']}]}, stream)
            rows = blind_origin.answers(td)
            self.assertEqual(len(rows), 3)
            self.assertEqual([row['kind'] for row in rows], ['unclamped', 'twin', 'control'])
            args = blind_origin.parser().parse_args(['submit', '--folder', td])
            self.assertEqual(args.folder, td)
            self.assertEqual(blind_origin.parser().parse_args(['submit']).folder, 'results/part26-8')


if __name__ == '__main__':
    unittest.main()
