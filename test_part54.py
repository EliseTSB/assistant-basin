"""CPU contract tests for Part 54 paths using a tiny random Qwen3.5 model."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from argparse import Namespace
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
            args = Namespace(out=td, clamp_unit='3:7', samples=8, control_count=3,
                             question='seed question', clamp_mult=9.0, seed=54,
                             max_new=64, _ladder_mode=False)
            manifest = p.run_ladder(args, self.model, self.tok, 'example/model', 'r1', 'example')
            path = os.path.join(td, 'ladder-example-L3u7.json')
            saved = json.load(open(path, encoding='utf-8'))
            self.assertEqual(clamp.call_count, 21)
            self.assertEqual(saved['status'], 'complete')
            self.assertEqual(saved['expected_generations'], 480)
            self.assertEqual(len(manifest['completed']), 21)
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
                result = p.run_clamp(args, self.model, self.tok, p.SOURCE_MODEL,
                                     p.SOURCE_REVISION, p.SOURCE_SHORT)
            self.assertEqual([row['condition'] for row in result['conditions']], ['unclamped', 'twin'])
            self.assertEqual(result['random_controls']['count'], 0)
            self.assertEqual(result['scale_frames']['count'], 30)
            self.assertGreater(result['best_twin']['largest_activation'], 0)

            args.clamp_mult = -2.0
            result = p.run_clamp(args, self.model, self.tok, p.SOURCE_MODEL,
                                 p.SOURCE_REVISION, p.SOURCE_SHORT)
            self.assertEqual(len(result['conditions']), 4)
            self.assertEqual(result['random_controls']['count'], 2)
            for row in result['conditions'][2:]:
                expected = -2.0 * result['random_controls']['largest_activations'][str(row['unit'])]
                self.assertAlmostEqual(row['set_value'], expected)

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
