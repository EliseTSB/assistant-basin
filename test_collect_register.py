import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from collections import Counter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'register-frames.json'
MODEL = 'Qwen/Qwen3-0.6B'


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(DATA.read_text(encoding='utf-8'))
        cls.items = cls.spec['items']

    def test_dataset_rules(self):
        self.assertIn('GPT-6 Luna', self.spec['note'])
        self.assertIn('2026-10-16', self.spec['note'])
        self.assertEqual(len(self.items), 72)
        banned = re.compile(
            r"as an ai|language model|\bassistant\b|i don't have feelings|i don’t have feelings|"
            r'chatgpt|claude|openai|anthropic|qwen|olmo|smollm|pythia|ai2', re.I)
        opener_counts = Counter()
        for index, item in enumerate(self.items):
            self.assertEqual(set(item), {'user', 'A', 'B', 'C'}, index)
            self.assertTrue(item['user'].strip(), index)
            for key in ('A', 'B', 'C'):
                self.assertTrue(1 <= len(re.split(r'(?<=[.!?])\s+', item[key].strip())) <= 4, (index, key))
            self.assertIsNone(banned.search(item['A']), (index, item['A']))
            sizes = [len(item[k].split()) for k in 'ABC']
            self.assertLessEqual(max(sizes) / min(sizes), 1.25, (index, sizes))
            match = re.match(r'^(.{1,36}?)(?:[:!?.])', item['A'])
            self.assertIsNotNone(match, item['A'])
            opener_counts[match.group(1).strip().lower()] += 1
        self.assertLessEqual(max(opener_counts.values()), 12)

        domain_patterns = [
            r'\brice\b|baking soda|bread dough|avocado|pasta water|apples',
            r'python|javascript|password|database|404|progress',
            r'burn|headache|ibuprofen|fever|sunscreen|stretching',
            r'weekend trip|airport|exchange cash|hike|jet lag|hotel reservation',
            r'debit card|debt|emergency fund|credit score|budget|compound interest',
            r'work email|paragraph|essay|cover letter|feedback|presentation',
            r'sink|breaker|glossy wall|door|smoke alarm|window',
            r'sky blue|heavier object|metal feel|phases of the moon|salt melt ice|plants getting',
            r'tip\?|fraction|mean and median|celsius|map scale|average of two numbers',
            r'task i.ve|decline an invitation|roommates|conversation with a friend|job offers|habit stick',
            r'affect|spanish|adjectives|less.*fewer|french noun|adding .please.',
            r'laptop|phone battery|free up space|printer|router|video call',
        ]
        domain_counts = [sum(bool(re.search(p, item['user'], re.I)) for item in self.items)
                         for p in domain_patterns]
        self.assertGreaterEqual(sum(n >= 5 for n in domain_counts), 8, domain_counts)

    def test_fake_blind_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / 'fake_judge.py'
            fake.write_text('import sys\nsys.stdin.read()\nprint("1")\n', encoding='utf-8')
            cmd = f'{shlex.quote(sys.executable)} {shlex.quote(str(fake))}'
            run = subprocess.run(
                [sys.executable, str(ROOT / 'judge_register.py'), '--cmd', cmd, '--seed', '31'],
                cwd=ROOT, text=True, capture_output=True, check=True, timeout=60)
            lines = run.stdout.strip().splitlines()
            self.assertEqual(len(lines), 73)
            records = [json.loads(line) for line in lines[:-1]]
            summary = json.loads(lines[-1])
            self.assertEqual([r['item'] for r in records], list(range(1, 73)))
            self.assertTrue(all(r['order'] in (['A', 'C'], ['C', 'A']) for r in records))
            expected = sum(r['order'][0] == 'A' for r in records)
            self.assertEqual(summary['A_picked'], expected)
            self.assertEqual(summary['total'], 72)
            self.assertAlmostEqual(summary['share_A_picked'], expected / 72)

    def test_qwen_collection_analyzer_and_decoded_spans(self):
        env = os.environ.copy()
        env['HF_HOME'] = '/mnt/e/neuron-probe/hf'
        env['HF_HUB_OFFLINE'] = '1'
        env['OMP_NUM_THREADS'] = '2'
        with tempfile.TemporaryDirectory() as tmp:
            run = subprocess.run(
                [sys.executable, str(ROOT / 'collect_register.py'), MODEL, tmp,
                 '--limit', '4', '--dtype', 'float32'],
                cwd=ROOT, env=env, text=True, capture_output=True, check=True, timeout=900)
            self.assertIn('wrote ', run.stdout)
            npz_path = Path(tmp) / 'Qwen__Qwen3-0.6B-register.npz'
            self.assertTrue(npz_path.is_file(), run.stdout)

            import numpy as np
            with np.load(npz_path) as z:
                forms = [str(x) for x in z['forms'].tolist()]
                self.assertIn('plain', forms)
                self.assertEqual(set(k for k in z.files if k.startswith('acts_plain_')),
                                 {f'acts_plain_{fam}' for fam in 'ABCD'})
                self.assertIn('neutral', z.files)
                self.assertIn('layers', z.files)
                self.assertIn('neurons', z.files)
                layer_count, neuron_count = int(z['layers']), int(z['neurons'])
                for form in forms:
                    for family in 'ABCD':
                        arr = z[f'acts_{form}_{family}']
                        self.assertEqual(arr.shape, (4, layer_count, neuron_count))
                        self.assertEqual(arr.dtype, np.float16)
                self.assertEqual(z['neutral'].ndim, 2)
                self.assertEqual(z['neutral'].shape[1], layer_count * neuron_count)
                self.assertEqual(z['neutral'].dtype, np.float16)

            analyzed = subprocess.run(
                [sys.executable, str(ROOT / 'analyze.py'), str(npz_path), '--form', 'plain',
                 '--frames', str(DATA)], cwd=ROOT, env=env, text=True,
                capture_output=True, check=True, timeout=180)
            self.assertIn('heldout_top_median', analyzed.stdout)

            from transformers import AutoTokenizer
            import collect_register
            tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
            first = self.items[0]
            for user, assistant, target, target_turn in (
                    (first['user'], first['A'], first['A'], 'assistant'),
                    (first['A'], 'Okay.', first['A'], 'user')):
                prompt = collect_register.texts(tokenizer, 'plain', user, assistant)
                encoded = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
                offsets = encoded['offset_mapping']
                positions = collect_register.span_token_indices(offsets, prompt, target)
                self.assertTrue(positions)
                start, end = collect_register.span_bounds(prompt, target)
                if target_turn == 'assistant':
                    self.assertGreaterEqual(start, prompt.index('Assistant: ') + len('Assistant: '))
                else:
                    self.assertLessEqual(end, prompt.index('\nAssistant:'))
                decoded = tokenizer.decode([encoded['input_ids'][i] for i in positions]).strip()
                self.assertEqual(re.sub(r'\s+', ' ', decoded), re.sub(r'\s+', ' ', target))


if __name__ == '__main__':
    unittest.main()
