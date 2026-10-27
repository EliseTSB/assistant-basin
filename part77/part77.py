#!/usr/bin/env python3
"""Part 77 paired sampled generation with position-wise L27 restoration."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if not (p and Path(p).resolve() == _HERE)]
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part70 import part70  # noqa: E402
from part66 import part66  # noqa: E402
from part66 import score66  # noqa: E402

L24 = (24, 3949)
L27 = (27, 3325)
C3_UNITS = [5578, 8007, 795, 5098, 3880, 3846, 3740, 6587, 6147, 5112]
CONDITION_IDS = ['c0_makers', 'c1_l24', 'c2_l24_restore', 'c3_l24_restore_nulls']


def build_conditions() -> list[dict]:
    target = list(part70.TARGET)
    return [
        {'id': 'c0_makers', 'set_units': target, 'copy_units': [L27[1]], 'kind': 'makers'},
        {'id': 'c1_l24', 'set_units': target + [L24], 'copy_units': [], 'kind': 'l24'},
        {'id': 'c2_l24_restore', 'set_units': target + [L24], 'copy_units': [L27[1]], 'kind': 'l24_restore'},
        {'id': 'c3_l24_restore_nulls', 'set_units': target + [L24], 'copy_units': list(C3_UNITS), 'kind': 'l24_restore_nulls'},
    ]


def _row_ops(cond: dict) -> dict:
    import part25
    return part25._operations(cond['set_units'], 'zero')


def _copy_hook(copy_units: list[list[int]], n_answers: int, stats: dict | None = None):
    """Return a layer-27 pre-hook that copies selected twin features to answers. If `stats` is given it records the
    evidence that the copy acted: calls, and the largest |answer - twin| on the copied units before and after it."""
    import torch

    def hook(_module, args):
        x = args[0]
        y = x.clone()
        pre = 0.0
        for i, units in enumerate(copy_units):
            if not units:
                continue
            idx = [int(u) for u in units]
            pre = max(pre, float((x[i, :, idx].float() - x[n_answers + i, :, idx].float()).abs().max()))
            y[i, :, idx] = x[n_answers + i, :, idx]
        if stats is not None:
            stats['calls'] = stats.get('calls', 0) + 1
            stats['precopy_max_abs_diff'] = max(stats.get('precopy_max_abs_diff', 0.0), pre)
            post = 0.0
            for i, units in enumerate(copy_units):
                if units:
                    idx = [int(u) for u in units]
                    post = max(post, float((y[i, :, idx].float() - y[n_answers + i, :, idx].float()).abs().max()))
            stats['postcopy_max_abs_diff'] = max(stats.get('postcopy_max_abs_diff', 0.0), post)
        return (y,) + tuple(args[1:])
    return hook


def _custom_processor_is_before_warpers() -> bool:
    """Inspect installed Transformers ordering; 5.20 merges custom processors before warpers."""
    import inspect
    from transformers.generation.utils import GenerationMixin
    source = inspect.getsource(GenerationMixin._get_logits_processor)
    merge = source.find('_merge_criteria_processor_list')
    warp = source.find('TemperatureLogitsWarper')
    return merge >= 0 and warp >= 0 and merge < warp


def sampling_warpers(generation_config, temperature: float = 0.7, top_p: float = 0.95):
    """The sampling warpers model.generate appends for run_70's arguments (temperature 0.7, top-p 0.95) on top of the
    model's own generation config, built from transformers' own classes in its own order (5.13 utils.py:1262-1294:
    temperature, top-h, top-k, top-p, min-p, typical, epsilon, eta; min_tokens_to_keep 1 with one beam). Custom logits
    processors run BEFORE these in 5.13 and 5.20, so the pair processor must apply them itself to sample what generate
    would have sampled. Returns (warpers, effective settings). An active warper this runner does not mirror refuses."""
    import copy
    from transformers import (LogitsProcessorList, MinPLogitsWarper, TemperatureLogitsWarper, TopKLogitsWarper,
                              TopPLogitsWarper)
    gc = copy.deepcopy(generation_config)
    gc.temperature, gc.top_p = temperature, top_p
    warpers = LogitsProcessorList()
    if gc.temperature is not None and gc.temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(gc.temperature))
    if getattr(gc, 'watermarking_config', None) is not None:
        raise ValueError('generation_config.watermarking_config is set; Part 77 does not mirror it')
    if getattr(gc, 'top_h', None) is not None:
        raise ValueError('generation_config.top_h is set; Part 77 does not mirror it')
    if gc.top_k is not None and gc.top_k != 0:
        warpers.append(TopKLogitsWarper(top_k=gc.top_k, min_tokens_to_keep=1))
    if gc.top_p is not None and gc.top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p=gc.top_p, min_tokens_to_keep=1))
    if gc.min_p is not None:
        warpers.append(MinPLogitsWarper(min_p=gc.min_p, min_tokens_to_keep=1))
    for name, active in (('typical_p', gc.typical_p is not None and gc.typical_p < 1.0),
                         ('epsilon_cutoff', gc.epsilon_cutoff is not None and 0.0 < gc.epsilon_cutoff < 1.0),
                         ('eta_cutoff', gc.eta_cutoff is not None and 0.0 < gc.eta_cutoff < 1.0)):
        if active:
            raise ValueError(f'generation_config.{name} is active; Part 77 does not mirror it')
    settings = {'temperature': gc.temperature, 'top_k': gc.top_k, 'top_p': gc.top_p, 'min_p': gc.min_p}
    return warpers, settings


class PairTokenLogitsProcessor:
    """Sample each pair's token from the answer and force both rows to use it."""
    def __init__(self, n_answers: int, seed: int, needs_warping: bool = False, warpers=None):
        import torch
        self.n_answers = int(n_answers)
        self.generator = torch.Generator(device='cpu')
        self.generator.manual_seed(int(seed))
        self.needs_warping = bool(needs_warping)
        if self.needs_warping and warpers is None:
            raise ValueError('needs_warping requires the warpers generate would apply')
        self.warpers = warpers

    def __call__(self, input_ids, scores):
        import torch
        n = self.n_answers
        answer_scores = scores[:n].detach().float()   # all answer rows at once: one warp, one CPU sync per step
        if self.needs_warping:
            answer_scores = self.warpers(input_ids[:n], answer_scores)
        probabilities = torch.softmax(answer_scores.cpu(), dim=-1)
        tokens = torch.multinomial(probabilities, 1, generator=self.generator)[:, 0].tolist()
        out = torch.full_like(scores, -float('inf'))
        for i, token in enumerate(tokens):
            out[i, int(token)] = 0.0
            out[n + i, int(token)] = 0.0
        return out


def generate_pairs(model, layers, input_ids, row_ops, copy_units, seed: int,
                   eos_id, pad_id, attention_mask=None, stats: dict | None = None):
    """Generate paired sequences; inputs and row_ops are ordered answers then twins."""
    import torch
    from transformers import LogitsProcessorList

    n_answers = len(copy_units)
    if input_ids.shape[0] != 2 * n_answers or len(row_ops) != 2 * n_answers:
        raise ValueError('paired generation requires 2n input rows and row operation lists')
    mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
    warpers, _ = sampling_warpers(model.generation_config)   # before any hook, so a refusal leaks none
    proc = PairTokenLogitsProcessor(n_answers, seed, _custom_processor_is_before_warpers(), warpers)
    handles = part66._hooks(layers, row_ops, mask.bool())
    handles.append(layers[27].mlp.down_proj.register_forward_pre_hook(_copy_hook(copy_units, n_answers, stats)))
    try:
        with torch.no_grad():
            result = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=48,
                do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=pad_id,
                eos_token_id=eos_id, use_cache=True,
                logits_processor=LogitsProcessorList([proc]))
    finally:
        for handle in handles:
            handle.remove()
    return result


def _copy_record_units(units):
    return [[27, int(u)] for u in units]


def _trim_at_eos(tokens, eos_id):
    raw = tokens.tolist()
    if eos_id is not None and int(eos_id) in raw:
        raw = raw[:raw.index(int(eos_id)) + 1]
    return raw


def run(args):
    import torch
    import neuron_report as nr
    import part25

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    answer_path = out / 'answers.jsonl'
    conditions = build_conditions()
    rows = part70.generations(conditions)
    if len(rows) != 4 * 240:
        raise RuntimeError(f'the design registers {4 * 240} answer rows, got {len(rows)}')
    (out / 'conditions.json').write_text(json.dumps([
        {'id': c['id'], 'set_units': part70._json_units(c['set_units']),
         'copy_units': _copy_record_units(c['copy_units'])} for c in conditions
    ], indent=2) + '\n', encoding='utf-8')

    existing = set()
    if answer_path.exists():
        with answer_path.open(encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    existing.add((r['condition'], int(r['question_index']), int(r['sample'])))
                except (ValueError, KeyError, TypeError):
                    continue
    todo = [r for r in rows if (r[0]['id'], r[1], r[4]) not in existing]
    by_question = defaultdict(list)
    for row in todo:
        by_question[row[1]].append(row)
    batches = [rs[i:i + args.batch] for rs in by_question.values() for i in range(0, len(rs), args.batch)]

    tok, model = nr.load_model(part70.MODEL, part70.REVISION, args.device)
    model.requires_grad_(False)
    _, layers, _ = nr.find_text_layers(model)
    n_units = layers[0].mlp.down_proj.in_features
    if any(u < 0 or u >= n_units for u in [*(p[1] for p in part70.TARGET), L24[1], L27[1], *C3_UNITS]):
        raise ValueError('a planned neuron is outside the loaded model dimensions')

    p66 = part66
    start = time.time()
    done = 0
    pairs_not_identical = 0
    copy_evidence = {'calls': 0, 'precopy_max_abs_diff': 0.0}
    with answer_path.open('a', encoding='utf-8') as sink:
        for chunk in batches:
            prompts = [part25._token_ids(tok, nr.build_prompt(tok, row[2], False)) for row in chunk]
            assert len({tuple(p) for p in prompts}) == 1, 'a batch must hold one prompt (no padding)'
            answer_ids, mask = p66._pad(prompts, tok, torch.device(args.device))
            n = len(chunk)
            ids = torch.cat([answer_ids, answer_ids], dim=0)
            row_ops = [_row_ops(row[0]) for row in chunk]
            twin_ops = part25._operations(part70.TARGET, 'zero')
            paired_ops = row_ops + [twin_ops for _ in chunk]
            copy_units = [list(row[0]['copy_units']) for row in chunk]
            seeds = [p66.row_seed(row[0]['id'], row[1], row[4]) for row in chunk]
            seed = seeds[0]
            torch.manual_seed(seed)
            if args.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if args.device == 'xpu' and hasattr(torch, 'xpu'):
                torch.xpu.manual_seed_all(seed)
            paired_mask = torch.cat([mask, mask], dim=0)
            batch_stats = {}
            result = generate_pairs(model, layers, ids, paired_ops, copy_units, seed,
                                    tok.eos_token_id, tok.pad_token_id, paired_mask, batch_stats)
            # In-run evidence that the copy acted (review r1): a batch holding a row that restores with L24 N3949
            # zeroed must show a pre-copy difference, and no difference may survive the copy.
            restores_under_l24 = any(row[0]['copy_units'] and L24 in [tuple(u) for u in row[0]['set_units']]
                                     for row in chunk)
            if any(row[0]['copy_units'] for row in chunk) and batch_stats.get('calls', 0) == 0:
                raise RuntimeError('the layer-27 copy hook never ran in a batch that copies')
            if restores_under_l24 and batch_stats.get('precopy_max_abs_diff', 0.0) == 0.0:
                raise RuntimeError('the copy hook saw nothing to restore under L24 N3949 zeroing')
            if batch_stats.get('postcopy_max_abs_diff', 0.0) != 0.0:
                raise RuntimeError('a difference survived the copy')
            copy_evidence['calls'] += batch_stats.get('calls', 0)
            copy_evidence['precopy_max_abs_diff'] = max(copy_evidence['precopy_max_abs_diff'],
                                                        batch_stats.get('precopy_max_abs_diff', 0.0))
            plen = ids.shape[1]
            for i, row in enumerate(chunk):
                ans_raw = _trim_at_eos(result[i, plen:], tok.eos_token_id)
                twin_raw = _trim_at_eos(result[n + i, plen:], tok.eos_token_id)
                identical = ans_raw == twin_raw
                if not identical:
                    pairs_not_identical += 1
                cond, qi, question, kind, sample = row
                record = {'condition': cond['id'], 'set_units': part70._json_units(cond['set_units']),
                    'question_index': qi, 'question': question, 'question_kind': kind, 'sample': sample,
                    'seed': seeds[i], 'answer': tok.decode(ans_raw, skip_special_tokens=True).strip(),
                    'new_tokens': len(ans_raw), 'stop_reason': p66._stop_reason(ans_raw, tok),
                    'copy_units': _copy_record_units(cond['copy_units']), 'pair_identical': identical}
                sink.write(json.dumps(record, ensure_ascii=False) + '\n')
            sink.flush()
            nr.spill_guard(args.device, args.max_gpu_gb)
            done += n
            elapsed = time.time() - start
            rate = done / max(elapsed, 1e-9)
            eta = max(0, len(todo) - done) / max(rate, 1e-9)
            print(f'part77: batch {done}/{len(todo)} answers elapsed={elapsed:.1f}s ETA={eta:.1f}s', flush=True)
    # Count all stored non-identical pairs so resumed runs retain the full run statistic.
    all_nonidentical = 0
    if answer_path.exists():
        for line in answer_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                try:
                    all_nonidentical += int(not json.loads(line).get('pair_identical', False))
                except (ValueError, TypeError):
                    pass
    wall = time.time() - start
    run_obj = {'model': part70.MODEL, 'revision': part70.REVISION,
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'device': str(part25.actual_device(model)), 'dtype': 'bfloat16', 'batch_size': args.batch,
        'wall_time_seconds': wall,
        'peak_memory_gb': (float(torch.cuda.max_memory_allocated() / 2**30)
                           if args.device == 'cuda' and torch.cuda.is_available() else
                           float(torch.xpu.max_memory_allocated() / 2**30)
                           if args.device == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available() else None),
        'requested_generations': len(rows), 'new_generations_this_invocation': len(todo),
        'pairs_not_identical': all_nonidentical,
        'copy_hook_evidence_this_invocation': copy_evidence,
        'sampling_settings': sampling_warpers(model.generation_config)[1],
        'custom_processor_before_warpers': _custom_processor_is_before_warpers(),
        'counts': {'requested_by_condition': dict(Counter(r[0]['id'] for r in rows)),
                   'requested_by_question_kind': dict(Counter(r[3] for r in rows)),
                   'new_by_question_kind': dict(Counter(r[3] for r in todo))},
        'sampling_reproducibility': 'Each answer row records its Part 66 SHA-256-derived seed. A batch uses the first answer row seed for pair-token sampling and generate RNG; rows share one sampled token stream per pair.',
        'conditions': [{'id': c['id'], 'set_units': part70._json_units(c['set_units']),
                       'copy_units': _copy_record_units(c['copy_units'])} for c in conditions]}
    (out / 'run.json').write_text(json.dumps(run_obj, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', choices=['xpu', 'cuda', 'cpu'], default='xpu')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--max-gpu-gb', type=float, default=14)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error('--batch must be positive')
    # Resolve all entry points before loading the model so a late import cannot break launch.
    import part25
    import neuron_report as nr
    required = (part70.generations, part70._json_units, part70.MODEL, part70.REVISION,
                part66.TARGET, part66.row_seed, part66._pad, part66._hooks,
                part66._stop_reason, score66.score_rows, part25._token_ids,
                part25._operations, part25.actual_device, nr.load_model,
                nr.find_text_layers, nr.build_prompt, nr.spill_guard)
    if not all(required):
        raise RuntimeError('a required Part 25/66/70/neuron_report entry point is unavailable')
    run(args)


if __name__ == '__main__':
    main()
