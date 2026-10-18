import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import clef_batch_gpu as gpu
import compare_1f
import jev_stub


class FakeModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, batch):
        self.calls += 1
        logits = []
        for _ in batch:
            logits.append([[0.0, 2.0]])
        return (logits,)


class FakeSchema:
    @staticmethod
    def encode_record(tokenizer, record, processor=None):
        question_ids = [SimpleNamespace(question_id=name, option_ids=('true', 'false'))
                        for name in record['questions']]
        return SimpleNamespace(questions=question_ids)

    @staticmethod
    def collate_records(records, pad_token_id, device):
        return records


class Part1fTests(unittest.TestCase):
    def test_stub_record_replay_hash_order_and_unknown(self):
        state_a = {'description': 'digits', 'snippet': '⟦7⟧ cats'}
        state_b = {'snippet': '⟦7⟧ cats', 'description': 'digits'}
        q_a = {'fires': {'type': 'noul', 'criteria': {'true': 'yes', 'false': 'no'}}}
        q_b = {'fires': {'criteria': {'false': 'no', 'true': 'yes'}, 'type': 'noul'}}
        digest = jev_stub.request_hash(state_a, q_a)
        self.assertEqual(digest, jev_stub.request_hash(state_b, q_b))

        def invoke(handler_type, request):
            handler = object.__new__(handler_type)
            payload = json.dumps(request).encode('utf-8')
            handler.path = '/v1/systemone'
            handler.rfile = io.BytesIO(payload)
            handler.headers = {'Content-Length': str(len(payload)), 'X-Climber': 'ignored'}
            results = []
            handler.reply = lambda status, body: results.append((status, body))
            handler.do_POST()
            return results[-1]

        with tempfile.TemporaryDirectory() as tmp:
            request_path = Path(tmp) / 'requests.jsonl'
            record_handler = jev_stub.make_handler('record', str(request_path))
            status, body = invoke(record_handler, {'model': 'jev-latest', 'state': state_a, 'questions': q_a})
            self.assertEqual(status, 200)
            self.assertEqual(body['answers']['fires']['noul'], 0.5)
            row = json.loads(request_path.read_text(encoding='utf-8'))
            self.assertEqual(row['hash'], digest)
            answer_path = Path(tmp) / 'answers.jsonl'
            answer_path.write_text(json.dumps({'hash': digest, 'answers': {'fires': 0.73}}) + '\n', encoding='utf-8')
            replay_handler = jev_stub.make_handler('replay', answers_path=str(answer_path))
            status, body = invoke(replay_handler, {'model': 'jev-latest', 'state': state_b, 'questions': q_b})
            self.assertEqual(status, 200)
            self.assertAlmostEqual(body['answers']['fires']['noul'], 0.73)
            unknown = {'model': 'jev-latest', 'state': {'description': 'other', 'snippet': '⟦x⟧'},
                       'questions': q_a}
            status, body = invoke(replay_handler, unknown)
            self.assertEqual(status, 500)
            self.assertEqual(body['hash'], jev_stub.request_hash(unknown['state'], q_a))

    def test_clef_answers_resume_without_recomputing_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            requests = Path(tmp) / 'requests.jsonl'
            answers = Path(tmp) / 'answers.jsonl'
            spec = {'type': 'noul'}
            rows = []
            for i in range(3):
                state = {'description': f'desc-{i}', 'snippet': f'⟦{i}⟧'}
                questions = {'fires': spec}
                rows.append({'hash': jev_stub.request_hash(state, questions), 'model': 'jev-latest',
                             'state': state, 'questions': questions})
            requests.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            answers.write_text(json.dumps({'hash': rows[0]['hash'], 'answers': {'fires': 0.4}}) + '\n', encoding='utf-8')
            model = FakeModel()
            first = gpu.process_requests(str(requests), str(answers), model, object(), FakeSchema,
                                         batch_size=2, device='cpu')
            self.assertEqual(first['processed'], 2)
            self.assertEqual(first['already_answered'], 1)
            self.assertEqual(model.calls, 1)
            output = [json.loads(x) for x in answers.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(output), 3)
            self.assertEqual(output[0]['hash'], rows[0]['hash'])
            self.assertAlmostEqual(output[1]['answers']['fires'], 1 / (1 + math.exp(2)))
            model.calls = 0
            second = gpu.process_requests(str(requests), str(answers), model, object(), FakeSchema,
                                          batch_size=2, device='cpu')
            self.assertEqual(second['processed'], 0)
            self.assertEqual(model.calls, 0)

    def test_compare_fixture_prints_five_numeric_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jev_cache = root / 'cache' / 'jev'
            questions = {'fires': {'type': 'noul', 'criteria': {'true': 'yes', 'false': 'no'}}}
            values = [[0.9, 0.6, 0.2], [0.2, 0.7, 0.4], [0.5, 0.4, 0.5], [0.2, 0.3, 0.4]]
            items_by_unit, requests, answer_rows, rows = {}, [], [], {'units': {}}
            for ui, probs in enumerate(values):
                uid = f'u{ui}'
                description = f'description-{ui}'
                items = []
                for si, p in enumerate(probs):
                    context = f'context-{ui}-{si}'
                    state = {'description': description, 'snippet': context}
                    digest = jev_stub.request_hash(state, questions)
                    cache_key = compare_1f.jev_cache_key(state, questions)
                    cache_path = jev_cache / cache_key[:2] / f'{cache_key}.json'
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps({'probs': {'fires': p}}), encoding='utf-8')
                    requests.append({'hash': digest, 'state': state, 'questions': questions})
                    answer_rows.append({'hash': digest, 'answers': {'fires': p}})
                    items.append({'kind': ('top', 'mid', 'random')[si], 'act': 3 - si,
                                  'p': p, 'context': context})
                items_by_unit[uid] = items
                rows['units'][uid] = {'arm': 'N' if ui < 3 else 'R', 'description': description,
                                      'items': items}
            jev_mean = sum(compare_1f.clef_spearmans(rows).values()) / 4
            scores = {'test': {'pipe': {'mean_spearman': jev_mean},
                               'pipe_auroc_top_median_by_arm': {'N': 0.5, 'R': 0.0}}}
            for name in ('jev-scores.json', 'clef-scores.json'):
                (root / name).write_text(json.dumps(scores), encoding='utf-8')
            (root / 'rows.json').write_text(json.dumps(rows), encoding='utf-8')
            (root / 'requests.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in requests), encoding='utf-8')
            (root / 'answers.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in answer_rows), encoding='utf-8')
            (root / 'gpu.log').write_text('THROUGHPUT requests=9 seconds=1.5 requests_per_second=6.000\n',
                                          encoding='utf-8')
            results = compare_1f.compare(root / 'jev-scores.json', root / 'clef-scores.json',
                                         root / 'rows.json', root / 'requests.jsonl', root / 'answers.jsonl',
                                         jev_cache, root / 'gpu.log')
            self.assertEqual(len(results), 5)
            self.assertEqual([row[-1] for row in results], [1, 1, 1, 1, 1])
            self.assertTrue(all(math.isfinite(v) for row in results for v in row if isinstance(v, float)))
            stdout = io.StringIO()
            args = ['--jev-scores', str(root / 'jev-scores.json'), '--clef-scores', str(root / 'clef-scores.json'),
                    '--rows', str(root / 'rows.json'), '--requests', str(root / 'requests.jsonl'),
                    '--answers', str(root / 'answers.jsonl'), '--jev-cache', str(jev_cache),
                    '--gpu-log', str(root / 'gpu.log')]
            with contextlib.redirect_stdout(stdout):
                compare_1f.main(args)
            lines = stdout.getvalue().splitlines()
            self.assertEqual(len(lines), 5)
            self.assertTrue(all(all(token.replace('.', '', 1).replace('-', '', 1).isdigit()
                                    for token in line.split()) for line in lines))


if __name__ == '__main__':
    unittest.main()
