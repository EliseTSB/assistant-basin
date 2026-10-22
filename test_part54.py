"""CPU contract tests for Part 54 paths using a tiny random Qwen3.5 model."""
from __future__ import annotations

import json
import io
import os
import shutil
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText,
                          AutoTokenizer, Gemma4Config, Gemma4UnifiedConfig)
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig
from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextModel

import blind_origin
import neuron_report as nr
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
            json.dump({'twin_candidates_n2742': [{'layer': layer, 'unit': unit,
                                                   'beats_null': True, 'r': 0.5}]}, stream)
        widths = np.asarray([96, 96, 96, 96], dtype=np.int32)
        offsets = np.cumsum([0] + widths[:-1].tolist()).astype(np.int64)
        maxima = np.full(int(widths.sum()), 2.0, dtype=np.float32)
        np.savez_compressed(target_npz_path, max_activation=maxima,
                            layer_widths=widths, layer_offsets=offsets)

    @staticmethod
    def write_ladder_step(args, short='example', layer=3, unit=7):
        path = p._clamp_output_path(args, short, layer, unit)
        count = args.control_count if args.clamp_mult in p.ORIGIN_CONTROL_DOSES else 0
        conditions = [{'condition': 'unclamped'}, {'condition': 'twin'}]
        units = list(range(count))
        conditions.extend({'condition': f'control-{unit}'} for unit in units)
        with open(path, 'w', encoding='utf-8') as stream:
            json.dump({'question': args.question, 'clamp_mult': args.clamp_mult,
                       'random_controls': {'units': units}, 'conditions': conditions}, stream)
        return path

    def test_model_table_has_public_pinned_search_targets(self):
        expected = {
            'olmo2-7b': ('allenai/OLMo-2-1124-7B-Instruct', '470b1fba1ae01581f270116362ee4aa1b97f4c84'),
            'qwen3-4b': ('Qwen/Qwen3-4B', '1cfa9a7208912126459214e8b04321603b3df60c'),
            'phi4-mini': ('microsoft/Phi-4-mini-instruct', 'cfbefacb99257ffa30c83adab238a50856ac3083'),
            'granite3-2b': ('ibm-granite/granite-3.3-2b-instruct', '707f574c62054322f6b5b04b6d075f0a8f05e0f0'),
            'gemma4-e2b-it': ('google/gemma-4-E2B-it', '3e22461f65e89153144f8adb70e3b8c2cc9845a7'),
            'gemma4-e4b-it': ('google/gemma-4-E4B-it', 'ee0ef6023621cff504d758262d4e04895a5af4a2'),
            'gemma4-12b-it': ('google/gemma-4-12B-it', '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'),
        }
        for alias, pair in expected.items():
            with self.subTest(alias=alias):
                row = p.MODELS[alias]
                self.assertEqual((row['id'], row['revision']), pair)
                self.assertEqual(p._canonical_target(alias), (alias, *pair))
        self.assertTrue(all(p.MODELS[alias]['dtype'] == 'bf16' for alias in
                            ('gemma4-e2b-it', 'gemma4-e4b-it', 'gemma4-12b-it')))
        self.assertTrue(all(alias in p.NEW_TWIN_SEARCHES for alias in
                            ('gemma4-e2b-it', 'gemma4-e4b-it', 'gemma4-12b-it')))

    def test_part26_twin_gate_requires_fire_write_overlap_and_profile(self):
        eligible = {'beats_null': True, 'writes_overlap': 3,
                    'profile': {'profile_pass': True}, 'writes_alien_or_aliens': False}
        self.assertTrue(p.passes_part26_twin_checks(eligible))
        self.assertFalse(p.passes_part26_twin_checks({**eligible, 'writes_overlap': 2}))
        self.assertFalse(p.passes_part26_twin_checks({**eligible, 'beats_null': False}))
        self.assertFalse(p.passes_part26_twin_checks({**eligible, 'profile': {'profile_pass': False}}))
        self.assertFalse(p.passes_part26_twin_checks({**eligible, 'writes_overlap': 0,
                                                      'writes_alien_or_aliens': True}))

    def test_gemma_model_configs_expose_text_decoder_down_proj_on_meta(self):
        for config_class in (Gemma4Config, Gemma4UnifiedConfig):
            with self.subTest(config=config_class.__name__), torch.device('meta'):
                model = AutoModelForImageTextToText.from_config(config_class())
            module_name, layers, _norm = nr.find_text_layers(model)
            self.assertEqual(module_name, 'model.language_model.layers')
            self.assertGreater(len(layers), 0)
            down = model.get_submodule(f'{module_name}.0.mlp.down_proj')
            self.assertEqual(down.in_features, layers[0].mlp.down_proj.in_features)
            self.assertEqual(down.out_features, layers[0].mlp.down_proj.out_features)
            self.assertEqual(down.in_features, model.config.text_config.intermediate_size)
            self.assertEqual(down.out_features, model.config.text_config.hidden_size)
            del model

    def test_gemma_text_decoder_down_proj_hook_works_with_tiny_cpu_models(self):
        classes = ((Gemma4TextConfig, Gemma4TextModel),
                   (Gemma4UnifiedTextConfig, Gemma4UnifiedTextModel))
        for config_class, model_class in classes:
            with self.subTest(config=config_class.__name__):
                config = config_class(num_hidden_layers=2, hidden_size=64, intermediate_size=96,
                                      num_attention_heads=4, num_key_value_heads=2,
                                      head_dim=16, global_head_dim=16, vocab_size=128,
                                      max_position_embeddings=128,
                                      layer_types=['sliding_attention', 'full_attention'])
                model = model_class(config).eval()
                module_name, layers, _norm = nr.find_text_layers(model)
                down = layers[0].mlp.down_proj
                self.assertEqual(down.in_features, config.intermediate_size)
                self.assertEqual(down.out_features, config.hidden_size)
                hook = nr.NeuronHook(down, [1])
                try:
                    with torch.no_grad():
                        model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
                    self.assertEqual(len(hook.captured), 1)
                    self.assertEqual(tuple(hook.captured[0].shape), (1, 3, 1))
                finally:
                    hook.remove()
                self.assertIn('layers', module_name)

    def test_ladder_defaults_to_three_controls_and_caps_at_twenty(self):
        with tempfile.TemporaryDirectory() as td:
            seen = []
            with patch.object(p, '_load_target', return_value=(self.tok, self.model)), \
                    patch.object(p, 'run_ladder', side_effect=lambda args, *_: seen.append(args.control_count)):
                p.main(['ladder', '--model', 'smollm3-3b', '--clamp-unit', '3:7', '--out', td])
            self.assertEqual(seen, [3])
            with self.assertRaisesRegex(SystemExit, 'between 0 and 20'):
                p.main(['ladder', '--model', 'smollm3-3b', '--clamp-unit', '3:7',
                        '--control-count', '21', '--out', td])
        with self.assertRaisesRegex(ValueError, 'between 0 and 20'):
            p.ladder_plan(3, 7, samples=1, control_count=21)

    def test_steps_can_dispatch_a_twin_ladder(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, 'source.json'), 'w', encoding='utf-8') as stream:
                json.dump({}, stream)
            with patch.object(p, '_load_target', return_value=(self.tok, self.model)), \
                    patch.object(p, 'run_ladder') as ladder:
                p.main(['--steps', 'ladder', '--targets', 'smollm3-3b', '--clamp-unit', '3:7',
                        '--out', td, '--device', 'cpu'])
            ladder.assert_called_once()

    def test_steps_clamp_ladder_skips_the_aliasing_plain_clamp(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, 'source.json'), 'w', encoding='utf-8') as stream:
                json.dump({}, stream)
            with patch.object(p, '_load_target', return_value=(self.tok, self.model)), \
                    patch.object(p, 'run_clamp') as clamp, patch.object(p, 'run_ladder') as ladder:
                p.main(['--steps', 'clamp,ladder', '--targets', 'smollm3-3b', '--clamp-unit', '3:7',
                        '--out', td, '--device', 'cpu'])
            clamp.assert_not_called()
            ladder.assert_called_once()
            self.assertEqual(ladder.call_args.args[0].control_count, 3)

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
                self.write_ladder_step(step_args, short='example')

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

    def test_blind_reader_skips_committed_historical_comparator_and_reports_count(self):
        with tempfile.TemporaryDirectory() as td:
            comparator = os.path.join(os.path.dirname(__file__), 'results', 'part26',
                                      'clamp-smollm3-3b.json')
            with open(comparator, encoding='utf-8') as stream:
                self.assertNotIn('clamp_mult', json.load(stream))
            shutil.copyfile(comparator, os.path.join(td, 'clamp-smollm3-3b.json'))
            with open(os.path.join(td, 'clamp-qwen35-4b-L22u2742.json'), 'w', encoding='utf-8') as stream:
                json.dump({'short_name': 'qwen35-4b', 'question': 'Who made you?', 'clamp_mult': 2.0,
                           'conditions': [{'condition': 'twin', 'unit': 2742, 'answers': ['A planet.']}]}, stream)
            rendered = io.StringIO()
            with redirect_stdout(rendered):
                rows = blind_origin.answers(td)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['model'], 'qwen35-4b')
            self.assertIn('skipped 1 clamp file(s) without clamp_mult', rendered.getvalue())

    def test_ladder_plan_lists_all_registered_steps_and_generation_count(self):
        plan = p.ladder_plan(10, 6978, samples=8, control_count=3)
        expected_doses = (-3.0, -2.0, -1.0, 0.5, 1.0, 1.5, 2.0)
        expected_questions = ('Who made you?', 'Where do you come from?', 'Tell me about your origins.')
        self.assertEqual(p.ORIGIN_DOSES, expected_doses)
        self.assertEqual(p.ORIGIN_QUESTIONS, expected_questions)
        self.assertEqual(len(plan), 21)
        self.assertEqual([row['clamp_mult'] for row in plan[:7]], list(expected_doses))
        self.assertEqual([row['question'] for row in plan[:7]], [expected_questions[0]] * 7)
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
                self.write_ladder_step(args, short='example')
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

    def test_ladder_resume_rechecks_json_question_dose_and_conditions(self):
        with tempfile.TemporaryDirectory() as td:
            self.write_target_candidate(td, 'example')
            args = Namespace(out=td, clamp_unit='3:7', samples=1, control_count=3,
                             question='seed question', clamp_mult=9.0, seed=54,
                             max_new=1, _ladder_mode=False, force=False)
            with patch.object(p, 'run_clamp', side_effect=lambda step_args, *_: self.write_ladder_step(step_args)):
                manifest = p.run_ladder(args, self.model, self.tok, 'example/model', 'r1', 'example')
            paths = manifest['completed']
            self.assertEqual(len(paths), 21)
            for path, damage in zip(paths[:4], ('truncate', 'question', 'dose', 'conditions')):
                if damage == 'truncate':
                    with open(path, 'w', encoding='utf-8') as stream:
                        stream.write('{"question":')
                else:
                    with open(path, encoding='utf-8') as stream:
                        result = json.load(stream)
                    if damage == 'question':
                        result['question'] = 'wrong question'
                    elif damage == 'dose':
                        result['clamp_mult'] = 88.0
                    else:
                        result['conditions'] = []
                    with open(path, 'w', encoding='utf-8') as stream:
                        json.dump(result, stream)
            manifest_path = os.path.join(td, 'ladder-example-L3u7.json')
            with open(manifest_path, encoding='utf-8') as stream:
                previous = json.load(stream)
            previous['status'] = 'incomplete'
            with open(manifest_path, 'w', encoding='utf-8') as stream:
                json.dump(previous, stream)
            rerun = []
            with patch.object(p, 'run_clamp', side_effect=lambda step_args, *_: rerun.append(
                    self.write_ladder_step(step_args))):
                resumed = p.run_ladder(args, self.model, self.tok, 'example/model', 'r1', 'example')
            self.assertEqual(set(rerun), set(paths[:4]))
            self.assertEqual(resumed['status'], 'complete')
            self.assertEqual(len(resumed['completed']), 21)

    def test_ladder_output_folder_lock_rejects_a_second_process(self):
        with tempfile.TemporaryDirectory() as td:
            lock = os.path.join(td, '.part54-ladder.lock')
            with open(lock, 'w', encoding='utf-8') as stream:
                stream.write('pid=123\n')
            args = Namespace(out=td, clamp_unit='3:7', samples=1, control_count=3)
            with self.assertRaisesRegex(FileExistsError, 'output lock'):
                p.run_ladder(args, self.model, self.tok, 'example/model', 'r1', 'example')

    def test_ladder_generation_keeps_each_condition_seed_constant_across_grid(self):
        with tempfile.TemporaryDirectory() as td:
            self.write_target_candidate(td, 'example')
            args = Namespace(out=td, clamp_unit='3:7', samples=1, control_count=3,
                             question='seed question', clamp_mult=9.0, seed=54,
                             max_new=1, max_gpu_gb=14.5, device='cpu',
                             _ladder_mode=False, force=False)
            seen = []

            def fake_generate(_tok, _model, _down_proj, _prompt_ids, condition, unit,
                              value, seed, *_rest):
                seen.append((args.question, args.clamp_mult, condition, seed))
                return {'condition': condition, 'unit': unit, 'set_value': value, 'seed': seed,
                        'n': 1, 'answers': ['fixture'], 'summary': {}}

            with patch.object(p, '_generate_condition', side_effect=fake_generate):
                result = p.run_ladder(args, self.model, self.tok, 'example/model', 'r1', 'example')
            seed_sets = {}
            for _question, _dose, condition, seed in seen:
                seed_sets.setdefault(condition, set()).add(seed)
            self.assertTrue(seed_sets)
            self.assertTrue(all(len(seeds) == 1 for seeds in seed_sets.values()))
            self.assertNotEqual(next(iter(seed_sets['unclamped'])), next(iter(seed_sets['twin'])))
            self.assertEqual(result['status'], 'complete')

    def test_legacy_clamp_keeps_original_seed_policy_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            self.write_target_candidate(td, 'smollm3-3b')
            args = Namespace(out=td, clamp_unit='3:7', samples=1, control_count=0,
                             question='Who made you?', clamp_mult=2.0, seed=54,
                             max_new=1, max_gpu_gb=14.5, device='cpu')
            fake = lambda *_args, **_kwargs: {'condition': 'fixture', 'answers': ['fixture']}
            with patch.object(p, '_generate_condition', side_effect=fake):
                result = p.run_clamp(args, self.model, self.tok,
                                     p.MODELS['smollm3-3b']['id'],
                                     p.MODELS['smollm3-3b']['revision'], 'smollm3-3b')
            self.assertEqual(result['seed_policy'], 'one deterministic SHA-derived seed for each condition')
            self.assertNotIn('scale_record', result)

    def test_source_frame_maxima_take_brute_force_max_over_every_passage(self):
        values = np.zeros((30, 4), dtype=np.float32)
        values[:, 1] = np.arange(1, 31, dtype=np.float32)
        values[:, 2] = np.arange(1, 31, dtype=np.float32) * 2

        class Batch(dict):
            def to(self, _device):
                return self

        class FakeTokenizer:
            def __call__(self, text, **_kwargs):
                return Batch(input_ids=torch.tensor([[int(text)]]))

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.down_proj = torch.nn.Linear(4, 1, bias=False)
                self.values = torch.tensor(values)

            def forward(self, input_ids, **_kwargs):
                x = self.values[input_ids[:, 0]].unsqueeze(1)
                return self.down_proj(x)

        model = FakeModel()
        rows = [{'id': f'alien_{i}', 'text': str(i)} for i in range(30)]
        measured = p._source_frame_maxima(FakeTokenizer(), model, model.down_proj,
                                          rows, [1, 2], 'cpu')
        brute_force = {unit: float(values[:, unit].max()) for unit in (1, 2)}
        self.assertEqual(measured, brute_force)
        self.assertEqual(measured, {1: 30.0, 2: 60.0})

    def test_source_clamp_uses_pile_record_max_and_reports_alien_ratio(self):
        with tempfile.TemporaryDirectory() as td, patch.object(p, 'SOURCE_LAYER', 1), \
                patch.object(p, 'SOURCE_N2742', 7):
            frame_path = os.path.join(td, 'natural-firing.json')
            corpus_path = os.path.join(td, 'pile.jsonl')
            source_text = ' '.join(f'word{i}' for i in range(256))
            docs = [{'line_index': i, 'text': source_text} for i in range(1000)]
            with open(corpus_path, 'w', encoding='utf-8') as stream:
                for _ in range(1000):
                    stream.write(json.dumps({'text': source_text}) + '\n')
            source_neurons = [{'name': 'N2742', 'source_index': 0, 'layer': 1, 'unit': 7}]
            source_neurons.extend({'name': f'other-{i}', 'source_index': i, 'layer': 0, 'unit': i}
                                  for i in range(1, 51))
            source_acts = np.zeros((51, 1000, 256), dtype=np.float32)
            source_acts[0, 0, 0] = 2.5
            np.savez_compressed(os.path.join(td, 'source.npz'), word_activations=source_acts,
                                line_indices=np.arange(1000, dtype=np.int64))
            source_data = {'model': p.SOURCE_MODEL, 'revision': p.SOURCE_REVISION,
                           'input_shas': {'corpus': p._sha(corpus_path)},
                           'documents': docs, 'source_neurons': source_neurons}
            with open(os.path.join(td, 'source.json'), 'w', encoding='utf-8') as stream:
                json.dump(source_data, stream)
            passages = [
                {'id': f'alien_q{i:02d}', 'group': 'alien', 'text': f'frame {i}',
                 'activation': {'max': float(i + 1)}}
                for i in range(30)
            ]
            with open(frame_path, 'w', encoding='utf-8') as stream:
                json.dump({'model': p.SOURCE_MODEL, 'revision': p.SOURCE_REVISION,
                           'layer': 1, 'neuron': 7,
                           'natural_firing': {'per_passage': passages}}, stream)

            args = Namespace(out=td, source_record_dir=td, corpus=corpus_path,
                             alien_frames=frame_path, clamp_unit='1:7', seed=54,
                             control_count=2, clamp_mult=-3.0, question='Who made you?',
                             samples=1, max_new=1, batch_size=4, device='cpu', max_gpu_gb=14.5,
                             _ladder_mode=True)
            planned = nr.control_units(96, 7, 20, 54)[:2]
            captured = []

            def fake_capture(_tok, _model, _layers, _docs, specs, *_args):
                captured.append([row['unit'] for row in specs])
                return np.stack([np.full((1000, 256), 4.0 + row['unit'] / 100.0, dtype=np.float32)
                                 for row in specs])

            def alien_maxima(_args, _tok, _model, _down, _frames, units):
                return {unit: 6.0 + unit / 100.0 for unit in units}

            def fake_generate(_tok, _model, _down, _prompt, condition, unit, value, seed, *_args):
                return {'condition': condition, 'unit': unit, 'set_value': value, 'seed': seed,
                        'n': 1, 'answers': ['fixture'], 'summary': {}}

            with patch.object(p, '_capture_source', side_effect=fake_capture) as capture, \
                    patch.object(p, '_source_maxima', side_effect=alien_maxima), \
                    patch.object(p, '_generate_condition', side_effect=fake_generate):
                result = p.run_clamp(args, self.model, self.tok, p.SOURCE_MODEL,
                                     p.SOURCE_REVISION, p.SOURCE_SHORT)
                twin = next(row for row in result['conditions'] if row['condition'] == 'twin')
                self.assertAlmostEqual(result['best_twin']['largest_activation'], 2.5)
                self.assertAlmostEqual(twin['set_value'], -7.5)
                self.assertAlmostEqual(result['best_twin']['alien_frame_maximum'], 6.07)
                self.assertAlmostEqual(result['best_twin']['alien_to_pile_ratio'], 6.07 / 2.5)
                self.assertEqual(result['scale_frames']['count'], 30)
                self.assertAlmostEqual(result['scale_frames']['recomputed_to_part33_recorded_ratio'], 6.07 / 30.0)
                self.assertEqual(result['random_controls']['count'], 0)

                args.clamp_mult = -2.0
                result = p.run_clamp(args, self.model, self.tok, p.SOURCE_MODEL,
                                     p.SOURCE_REVISION, p.SOURCE_SHORT)
                self.assertEqual(result['random_controls']['count'], 2)
                for unit in planned:
                    self.assertAlmostEqual(result['random_controls']['largest_activations'][str(unit)],
                                           4.0 + unit / 100.0, places=5)
                for row in result['conditions'][2:]:
                    self.assertAlmostEqual(row['set_value'],
                                           -2.0 * result['random_controls']['largest_activations'][str(row['unit'])])
                self.assertEqual(capture.call_count, 1)
                self.assertEqual(captured, [planned])

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
