#!/usr/bin/env python3
"""Part 215 runner: `preflight` (G0/G1/G2-pilot/G3 and latency) and `full` (2,304 generations and
624 conditional forwards) for the rudeness / nature-denial study on Qwen3.5-4B.

`part215/DESIGN.md` v4 is the contract (SHA-256 pinned in score215.py).  The canonical plan -- arms,
variants, samples, seeds -- lives in score215.py, which is numpy-only, so the runner and the scorer
cannot disagree about what was planned.  Part 214's serialization (`run214.question_ids`), Part 66's
row seed / stop-reason / generation shape, and neuron_report's loader are reused, not rewritten.

Run under the XPU Python as
    python -I -B part215/run215.py preflight --out results/part215/preflight --device xpu
    python -I -B part215/run215.py full --out results/part215/full --device xpu \
        --preflight results/part215/preflight/preflight.json

Every model call is batch 1, eager attention, bf16, temperature 0.7, top_p 0.95, 48 new tokens, as
Part 214.  A deadline guard stops launching calls; rows are appended with an fsync each, so a stopped
run resumes by key.  `full` refuses unless the preflight receipt says PASS for the same prompts,
design and model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import neuron_report as nr  # noqa: E402
from part66 import part66 as p66  # noqa: E402  (row_seed, _stop_reason, the generation loop shape)
import score215 as sc  # noqa: E402  (the canonical plan; numpy only, so the two files cannot drift)

MODEL = sc.MODEL
REVISION = sc.REVISION
TARGET = [(int(l), int(u)) for l, u in sc.TARGET]
MONITORS = [(int(l), int(u)) for l, u in sc.MONITORS]
GOOGLE_ID, AL_ID, T_ID = sc.GOOGLE_ID, sc.AL_ID, sc.T_ID
MAX_NEW_TOKENS = sc.MAX_NEW_TOKENS
OPENING_TOKENS = sc.OPENING_TOKENS
TEMPERATURE = 0.7
TOP_P = 0.95
PILOT_MAX_LATENCY_S = 7.5          # DESIGN.md "Size": stop above 7.5 s per call
G1_NEEDED = 45                     # DESIGN.md G1: at least 45 of 48
DEFAULT_PREFLIGHT_DEADLINE_MIN = 90
DEFAULT_FULL_DEADLINE_MIN = 300    # design workload is about 3.06 hours (184 min) plus forwards


# ----------------------------------------------------------------------------------------------------------------
# JSON / JSONL plumbing
# ----------------------------------------------------------------------------------------------------------------

def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'object of type {type(obj).__name__} is not JSON serializable')


def write_json(path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=json_default) + '\n',
                   encoding='utf-8')
    os.replace(tmp, path)


def read_jsonl_recover(path, repair: bool = True) -> tuple[list[dict], int]:
    """Rows from a JSONL file, dropping and repairing a torn tail (a killed append).

    `repair=False` is for a RECEIPT THIS RUN DID NOT WRITE (Part 214's Stage 2 file): a torn tail
    there is fatal, never something to truncate -- this runner may only write inside its own --out.
    """
    path = Path(path)
    if not path.exists():
        return [], 0
    raw = path.read_bytes()
    if not raw:
        return [], 0
    last_newline = raw.rfind(b'\n')
    body, tail = (raw[:last_newline + 1], raw[last_newline + 1:]) if last_newline >= 0 else (b'', raw)
    rows = []
    for number, line in enumerate(body.split(b'\n'), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f'{path} line {number} is corrupt ({exc}); an interior line is not a partial write')
        if not isinstance(value, dict):
            raise ValueError(f'{path} line {number}: expected a JSON object')
        rows.append(value)
    dropped = len(tail)
    if dropped:
        if not repair:
            raise ValueError(f'{path} ends in {dropped} bytes of a partial line; it is not this run\'s file '
                             f'to repair')
        with path.open('r+b') as fh:
            fh.truncate(len(body))
    return rows, dropped


def append_row(sink, record: dict) -> None:
    sink.write(json.dumps(record, ensure_ascii=False, default=json_default) + '\n')
    sink.flush()
    os.fsync(sink.fileno())


class Deadline:
    def __init__(self, minutes: float, clock=time.monotonic):
        self.clock = clock
        self.limit_s = float(minutes) * 60.0
        self.start = clock()

    def hit(self) -> bool:
        return self.clock() - self.start >= self.limit_s

    def elapsed(self) -> float:
        return self.clock() - self.start


# ----------------------------------------------------------------------------------------------------------------
# the rig
# ----------------------------------------------------------------------------------------------------------------

def load_rig(args):
    """nr.load_model returns (tok, model); eager attention and bf16 exactly as Part 214."""
    tok, model = nr.load_model(MODEL, REVISION, device=args.device, attn_implementation='eager',
                               dtype=torch.bfloat16)
    return model, tok, dict(nr.LAST_LOAD)


def runtime_identity(load: dict | None, device: str) -> dict:
    import transformers
    return {'device': device, 'dtype': 'bfloat16', 'attn_implementation': (load or {}).get('attn_implementation'),
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'batch_size': 1, 'max_new_tokens': MAX_NEW_TOKENS,
            'temperature': TEMPERATURE, 'top_p': TOP_P}


# ----------------------------------------------------------------------------------------------------------------
# prompt lookup
# ----------------------------------------------------------------------------------------------------------------

def prompt_index(bank: dict) -> dict:
    out = {}
    for p in bank['prompts']:
        key = (p['set'], p['arm'], p['variant'], int(p['question_index']))
        if key in out:
            raise ValueError(f'prompts215.json holds {key} twice')
        out[key] = p
    return out


def prompt_for(bank_idx: dict, set_name: str, arm: str, variant: str, qi: int) -> dict:
    key = (set_name, arm, variant, int(qi))
    if key not in bank_idx:
        raise KeyError(f'prompts215.json has no prompt for {key}')
    return bank_idx[key]


def question_text(bank: dict, set_name: str, qi: int) -> str:
    return bank['question_sets'][set_name]['questions'][qi]['text']


def natural_opening(token_ids: list[int], tok, bank: dict) -> bool:
    """True when the answer's first five tokens open with one of the design's natural openings.

    AMBIGUITY 4: DESIGN.md says "the first five tokens are the natural opening family `I was
    developed by` / `I was created by` / `I am developed by`".  The tokens are decoded and the text
    is compared after stripping, so a leading space token does not hide an opening; a raw token-id
    comparison would miss ` I was developed by` (token 358 rather than 40).
    """
    if not token_ids:
        return False
    text = tok.decode(torch.tensor([int(t) for t in token_ids[:OPENING_TOKENS]],
                                   dtype=torch.long), skip_special_tokens=True).strip()
    return any(text.startswith(entry['text']) for entry in bank['natural_openings'])


# ----------------------------------------------------------------------------------------------------------------
# one batch-1 sampled generation (Part 66's loop shape, Part 214's settings)
# ----------------------------------------------------------------------------------------------------------------

def generate_row(row: dict, bank_idx: dict, bank: dict, tok, model, args, pins: dict | None = None,
                 runtime: dict | None = None, max_new_tokens: int | None = None) -> dict:
    max_new_tokens = MAX_NEW_TOKENS if max_new_tokens is None else int(max_new_tokens)
    set_name, arm, variant = row['set'], row['arm'], row['variant']
    qi, sample = int(row['question_index']), int(row['sample'])
    prompt = prompt_for(bank_idx, set_name, arm, variant, qi)
    ids = [int(t) for t in prompt['token_ids']]
    T = len(ids)
    if 'seed' in row:
        seed = int(row['seed'])
    else:
        seed = sc.pilot_seed(variant, qi, sample) if row.get('pilot') else \
            sc.row_seed(arm, set_name, variant, qi, sample)
    torch.manual_seed(seed)
    if args.device == 'xpu' and hasattr(torch, 'xpu'):
        torch.xpu.manual_seed_all(seed)
    started = time.monotonic()
    with torch.no_grad():
        result = model.generate(input_ids=torch.tensor([ids], dtype=torch.long, device=args.device),
                                max_new_tokens=max_new_tokens, do_sample=True, temperature=TEMPERATURE,
                                top_p=TOP_P, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                                use_cache=True)
    latency = time.monotonic() - started
    raw = [int(t) for t in result[0, T:].tolist()]
    eos = tok.eos_token_id
    if eos is not None and int(eos) in raw:
        raw = raw[:raw.index(int(eos)) + 1]
    answer = tok.decode(torch.tensor(raw, dtype=torch.long), skip_special_tokens=True).strip()
    opener = next((v['opener'] for v in bank['arms'][arm] if v['variant'] == variant), None)
    return {'set': set_name, 'arm': arm, 'variant': variant, 'question_index': qi, 'sample': sample,
            'seed': seed, 'question': question_text(bank, set_name, qi), 'opener': opener,
            'prompt': nr.build_prompt(tok, prompt['text'], False), 'prompt_token_ids': ids,
            'answer': answer, 'token_ids': raw, 'stop_reason': p66._stop_reason(raw, tok),
            'natural_opening': natural_opening(raw, tok, bank),
            'batch_size': 1, 'batch_size_requested': 1, 'pilot': bool(row.get('pilot', False)),
            'latency_s': latency, 'pins': pins, 'runtime': runtime}


# ----------------------------------------------------------------------------------------------------------------
# conditional readout: one teacher-forced forward per (arm, variant, question)
# ----------------------------------------------------------------------------------------------------------------

def g_ali(logits_row) -> float:
    v = logits_row.float()
    return float(v[GOOGLE_ID]) - float(np.logaddexp(float(v[AL_ID]), float(v[T_ID])))


def g_al(logits_row) -> float:
    v = logits_row.float()
    return float(v[GOOGLE_ID]) - float(v[AL_ID])


def install_readout_hooks(layers, opening_start: int, store: dict, zero_targets: bool = False):
    """Record the target units and monitors at every opening position; optionally zero the target
    units at positions 0..T+4 (Part 214's `zero` arm: activation x 0.0 at every prompt/opening
    position, continuation positions untouched).

    One forward pre-hook per layer.  Values move to the CPU once per hook (Part 122's lesson).
    """
    targets_by_layer = defaultdict(list)
    for layer, unit in TARGET:
        targets_by_layer[layer].append(unit)
    monitors_by_layer = defaultdict(list)
    for layer, unit in MONITORS:
        monitors_by_layer[layer].append(unit)
    handles = []
    for layer in sorted(set(targets_by_layer) | set(monitors_by_layer)):
        target_units = targets_by_layer.get(layer, [])
        monitor_units = monitors_by_layer.get(layer, [])

        def hook(_module, args, layer=layer, target_units=list(target_units),
                 monitor_units=list(monitor_units)):
            x = args[0]
            changed = x
            if target_units:
                store.setdefault('units', {})[layer] = [
                    x[0, opening_start + k, target_units].float().cpu().tolist()
                    for k in range(OPENING_TOKENS)]
            if monitor_units:
                store.setdefault('monitors', {})[layer] = [
                    x[0, opening_start + k, monitor_units].float().cpu().tolist()
                    for k in range(OPENING_TOKENS)]
            if zero_targets and target_units:
                width = min(x.shape[1], opening_start + OPENING_TOKENS)
                out = x.clone()
                col = torch.tensor(target_units, dtype=torch.long, device=x.device)
                out[:, :width, col] = 0.0
                changed = out
            return (changed,) + tuple(args[1:]) if zero_targets and target_units else None
        handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
    return handles


def _unit_slot(pairs: list[tuple[int, int]], layer: int, unit: int) -> int:
    """The index of (layer, unit) among that layer's units in `pairs` (the order the hook recorded)."""
    return [u for l, u in pairs if l == layer].index(unit)


def conditional_forward(model, layers, ids: list[int], opening: list[int], device: str,
                        zero_targets: bool = False) -> dict:
    """One forward of prompt + fixed opening, read at the last opening position and at all five."""
    seq = [int(t) for t in ids] + [int(t) for t in opening]
    T = len(ids)
    store: dict = {}
    handles = install_readout_hooks(layers, T, store, zero_targets=zero_targets)
    started = time.monotonic()
    try:
        with torch.no_grad():
            out = model(input_ids=torch.tensor([seq], dtype=torch.long, device=device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    latency = time.monotonic() - started
    logits = out.logits[0]
    position = T + OPENING_TOKENS - 1
    units = []
    for k in range(OPENING_TOKENS):
        units.append({'position': T + k,
                      'units': [float(store['units'][layer][k][_unit_slot(TARGET, layer, unit)])
                                for layer, unit in TARGET],
                      'monitors': [float(store['monitors'][layer][k][_unit_slot(MONITORS, layer, unit)])
                                   for layer, unit in MONITORS]})
    return {'g_ali': g_ali(logits[position]), 'g_al': g_al(logits[position]),
            'read_position': position, 'opening': units, 'latency_s': latency, 'zero_targets': zero_targets}


def conditional_row(bank_idx, bank, tok, model, layers, arm: str, variant: str, qi: int,
                    device: str, zero_targets: bool = False) -> dict:
    prompt = prompt_for(bank_idx, 'identity', arm, variant, qi)
    opening = [int(t) for t in bank['openings']['en_maker']['token_ids']]
    rec = conditional_forward(model, layers, prompt['token_ids'], opening, device, zero_targets=zero_targets)
    rec.update({'set': 'identity', 'arm': arm, 'variant': variant, 'question_index': qi,
                'prompt_token_ids': [int(t) for t in prompt['token_ids']],
                'opening_token_ids': opening})
    return rec


# ----------------------------------------------------------------------------------------------------------------
# G0: anchor against Part 214's Stage 1 receipt (unchanged and zero, per question)
# ----------------------------------------------------------------------------------------------------------------

G0_TOLERANCE = 0.02


def load_stage1_receipt(path, prompts_sha256: str) -> dict:
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    if obj.get('part') != 214:
        raise SystemExit(f'{path}: part is {obj.get("part")}, want Part 214')
    recorded = obj.get('prompts_sha256')
    if recorded != prompts_sha256:
        raise SystemExit(f'{path}: prompts_sha256 is {recorded}, this build uses {prompts_sha256}')
    for arm in ('unchanged', 'zero'):
        arm_obj = obj.get('arms', {}).get(arm)
        if not arm_obj or len(arm_obj.get('g_ali', [])) < sc.N_QUESTIONS:
            raise SystemExit(f'{path}: no usable arms.{arm}.g_ali')
    return obj


def gate0(model, tok, bank, bank_idx, receipt: dict, device: str) -> dict:
    """24 neutral + 24 zero teacher-forced forwards at the opening, compared per question to Part
    214's Stage 1 `unchanged` and `zero` g_ali within 0.02."""
    rows = []
    layers = nr.find_text_layers(model)[1]
    for label, zero, arm in (('unchanged', False, 'neutral'), ('zero', True, 'neutral')):
        theirs = [float(x) for x in receipt['arms']['unchanged' if not zero else 'zero']['g_ali'][:sc.N_QUESTIONS]]
        for qi in range(sc.N_QUESTIONS):
            rec = conditional_row(bank_idx, bank, tok, model, layers, arm, 'v0', qi, device,
                                  zero_targets=zero)
            delta = abs(rec['g_ali'] - theirs[qi])
            rows.append({'gate': 'G0', 'comparison': label, 'question_index': qi,
                         'ours': rec['g_ali'], 'part214': theirs[qi], 'abs_delta': delta,
                         'pass': delta <= G0_TOLERANCE, 'latency_s': rec['latency_s']})
    per = {}
    for label in ('unchanged', 'zero'):
        subset = [r for r in rows if r['comparison'] == label]
        per[label] = {'n': len(subset), 'max_abs_delta': max(r['abs_delta'] for r in subset),
                      'pass': all(r['pass'] for r in subset),
                      'worst_question': max(subset, key=lambda r: r['abs_delta'])['question_index']}
    return {'pass': all(v['pass'] for v in per.values()), 'tolerance': G0_TOLERANCE,
            'comparisons': per, 'rows': rows}


# ----------------------------------------------------------------------------------------------------------------
# G1: the do-nothing audit against Part 214's Stage 2 `unchanged` rows
# ----------------------------------------------------------------------------------------------------------------

def load_stage2_unchanged(path, prompts_sha256: str) -> dict:
    try:
        rows, dropped = read_jsonl_recover(path, repair=False)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    out = {}
    for r in rows:
        if r.get('arm') != 'unchanged' or int(r.get('question_index', -1)) >= sc.N_QUESTIONS:
            continue
        if bool(r.get('pilot', False)):
            continue
        sample = int(r['sample'])
        if sample in (0, 1):
            out[(int(r['question_index']), sample)] = r
    if len(out) != 2 * sc.N_QUESTIONS:
        raise SystemExit(f'{path}: {len(out)} unchanged rows for samples 0/1, want '
                         f'{2 * sc.N_QUESTIONS} (dropped {dropped} bytes of torn tail)')
    recorded = rows[0].get('pins', {}).get('prompts_sha256') if rows else None
    if recorded is not None and recorded != prompts_sha256:
        raise SystemExit(f'{path}: rows were generated under prompts {recorded}, this build uses {prompts_sha256}')
    return out


def gate1(model, tok, bank, bank_idx, part214: dict, args) -> dict:
    """48 neutral identity generations on Part 214's exact seeds (samples 0 and 1), compared token
    for token; at least 45 of 48 must match."""
    rows = []
    for qi in range(sc.N_QUESTIONS):
        for sample in (0, 1):
            row = {'set': 'identity', 'arm': 'neutral', 'variant': 'v0', 'question_index': qi,
                   'sample': sample, 'seed': sc.row_seed('neutral', 'identity', 'v0', qi, sample)}
            rec = generate_row(row, bank_idx, bank, tok, model, args)
            theirs = part214[(qi, sample)]
            same = [int(t) for t in rec['token_ids']] == [int(t) for t in theirs.get('token_ids', [])]
            rows.append({'gate': 'G1', 'question_index': qi, 'sample': sample, 'seed': rec['seed'],
                         'matched': same, 'ours': rec['token_ids'], 'part214': theirs.get('token_ids'),
                         'ours_answer': rec['answer'], 'part214_answer': theirs.get('answer'),
                         'latency_s': rec['latency_s']})
    matched = sum(1 for r in rows if r['matched'])
    return {'pass': matched >= G1_NEEDED, 'matched': matched, 'needed': G1_NEEDED, 'n': len(rows),
            'rows': rows}


# ----------------------------------------------------------------------------------------------------------------
# pilot: 96 rude_v rows in their own seed namespace, G2 on the pilot arm
# ----------------------------------------------------------------------------------------------------------------

def run_pilot(model, tok, bank, bank_idx, args, deadline: Deadline | None = None) -> dict:
    rows = []
    reached = False
    for row in sc.pilot_rows():
        if deadline is not None and deadline.hit():
            reached = True
            break
        row = dict(row, pilot=True, seed=sc.pilot_seed(row['variant'], row['question_index'], row['sample']))
        rows.append(generate_row(row, bank_idx, bank, tok, model, args))
    broken = sum(1 for r in rows if sc.is_broken(r['answer']))
    denominator = len(sc.pilot_rows())
    bound = int(math.floor(sc.BROKEN_LIMIT * denominator + 1e-12))
    return {'n': len(rows), 'planned': denominator, 'broken': broken,
            'broken_rate': broken / denominator if denominator else None, 'bound': bound,
            'pass': len(rows) == denominator and broken <= bound, 'deadline_reached': reached,
            'rows': rows}


# ----------------------------------------------------------------------------------------------------------------
# preflight
# ----------------------------------------------------------------------------------------------------------------

def tool_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def preflight(args, rig_loader=None, clock=time.monotonic) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bank = sc.load_prompts(args.prompts if getattr(args, 'prompts', None) else None)
    bank_idx = prompt_index(bank)
    deadline = Deadline(getattr(args, 'deadline_min', None) or DEFAULT_PREFLIGHT_DEADLINE_MIN, clock=clock)

    receipt = {
        'part': sc.PART, 'mode': 'preflight', 'design': sc.DESIGN_VERSION,
        'design_sha256': sc.DESIGN_SHA256, 'model': MODEL, 'revision': REVISION,
        'device': getattr(args, 'device', 'cpu'), 'dtype': 'bfloat16',
        'batch_size': 1, 'max_new_tokens': MAX_NEW_TOKENS,
        'prompts_sha256': bank['canonical_sha256'], 'prompts_file_sha256': bank['_file_sha256'],
        'tool_sha256': tool_sha256(),
        'gates': {}, 'stop_reason': None, 'full': 'not_run',
    }
    started = clock()
    try:
        g3 = sc.g3_scorer_check()
        receipt['gates']['G3'] = g3
        if not g3['pass']:
            return _stop(out, receipt, 'G3', 'attrib215 fixtures or the deliberate break failed', started, clock)

        model, tok, load = (rig_loader or load_rig)(args)
        layers = nr.find_text_layers(model)[1]
        receipt['attn_implementation'] = load.get('attn_implementation')
        receipt['runtime'] = runtime_identity(load, args.device)

        stage1_path = getattr(args, 'stage1', None) or str(ROOT / 'results/part214/stage1/stage1.json')
        stage2_path = getattr(args, 'stage2', None) or str(ROOT / 'results/part214/stage2/stage2.jsonl')
        receipt['part214_stage1_sha256'] = hashlib.sha256(Path(stage1_path).read_bytes()).hexdigest()
        receipt['part214_stage2_sha256'] = hashlib.sha256(Path(stage2_path).read_bytes()).hexdigest()
        stage1 = load_stage1_receipt(stage1_path, bank['source_part214_prompts_sha256'])
        stage2 = load_stage2_unchanged(stage2_path, bank['source_part214_prompts_sha256'])

        g0 = gate0(model, tok, bank, bank_idx, stage1, args.device)
        receipt['gates']['G0'] = g0
        if not g0['pass']:
            return _stop(out, receipt, 'G0', 'per-question g_ali is more than 0.02 from Part 214 Stage 1',
                         started, clock)

        g1 = gate1(model, tok, bank, bank_idx, stage2, args)
        receipt['gates']['G1'] = g1
        if not g1['pass']:
            return _stop(out, receipt, 'G1', f'only {g1["matched"]} of {g1["n"]} audit rows reproduce '
                                             f'Part 214 Stage 2 (need {G1_NEEDED})', started, clock)

        pilot = run_pilot(model, tok, bank, bank_idx, args, deadline)
        receipt['gates']['G2_pilot'] = pilot
        if not pilot['pass']:
            return _stop(out, receipt, 'G2', f'pilot broken {pilot["broken"]}/{pilot["planned"]} '
                                             f'(bound {pilot["bound"]}) or deadline', started, clock)

        gen_latencies = [r['latency_s'] for r in g1['rows']] + [r['latency_s'] for r in pilot['rows']]
        fwd_latencies = [r['latency_s'] for r in g0['rows']]
        # DESIGN.md "Size": stop when "latency exceeds 7.5 s per call".  "Per call" covers every model
        # call preflight makes -- the 48 anchor forwards as well as the 144 audit/pilot generations;
        # the two are also reported separately, because the design sizes the run at 4.5 s/generation.
        all_latencies = gen_latencies + fwd_latencies
        latency = {'n_generation_calls': len(gen_latencies),
                   'generation_max_s': max(gen_latencies) if gen_latencies else None,
                   'generation_mean_s': float(np.mean(gen_latencies)) if gen_latencies else None,
                   'n_forward_calls': len(fwd_latencies),
                   'forward_max_s': max(fwd_latencies) if fwd_latencies else None,
                   'forward_mean_s': float(np.mean(fwd_latencies)) if fwd_latencies else None,
                   'max_s': max(all_latencies) if all_latencies else None,
                   'limit_s': PILOT_MAX_LATENCY_S,
                   'pass': bool(all_latencies) and max(all_latencies) <= PILOT_MAX_LATENCY_S}
        receipt['latency'] = latency
        if not latency['pass']:
            return _stop(out, receipt, 'latency', f'max model-call latency '
                                                  f'{latency["max_s"]:.2f} s > {PILOT_MAX_LATENCY_S}',
                         started, clock)
    except SystemExit as exc:
        # A malformed Part 214 receipt is a preflight failure too: STOP names it instead of a bare
        # argparse-style exit a person has to interpret.
        return _stop(out, receipt, 'input', str(exc), started, clock)
    except Exception as exc:  # a crashed gate is a STOP, not a traceback a person has to read
        return _stop(out, receipt, 'error', f'{type(exc).__name__}: {exc}', started, clock)

    receipt['result'] = 'PASS'
    receipt['full'] = 'allowed'
    receipt['wall_time_seconds'] = clock() - started
    write_json(out / 'preflight.json', receipt)
    (out / 'PASS').write_text('PASS\n', encoding='utf-8')
    print(f'preflight: PASS ({receipt["wall_time_seconds"]:.1f} s); full allowed', flush=True)
    return receipt


def _stop(out: Path, receipt: dict, gate: str, reason: str, started: float, clock) -> dict:
    receipt['result'] = 'STOP'
    receipt['stop_reason'] = f'{gate}: {reason}'
    receipt['wall_time_seconds'] = clock() - started
    write_json(out / 'preflight.json', receipt)
    (out / 'STOP').write_text(f'STOP: {gate}: {reason}\n', encoding='utf-8')
    print(f'preflight: STOP ({gate}): {reason}; full not_run', flush=True)
    return receipt


# ----------------------------------------------------------------------------------------------------------------
# full run
# ----------------------------------------------------------------------------------------------------------------

def require_pass_preflight(path, bank: dict) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f'full refuses: no preflight receipt at {p}')
    obj = json.loads(p.read_text(encoding='utf-8'))
    checks = {
        'result_is_PASS': obj.get('result') == 'PASS',
        'prompts_sha256_matches': obj.get('prompts_sha256') == bank['canonical_sha256'],
        'design_sha256_matches': obj.get('design_sha256') == sc.DESIGN_SHA256,
        'model_matches': obj.get('model') == MODEL,
        'revision_matches': obj.get('revision') == REVISION,
    }
    if not all(checks.values()):
        raise SystemExit(f'full refuses: preflight receipt {p} fails {[k for k, v in checks.items() if not v]}')
    return obj


def forward_plan() -> list[dict]:
    return [{'set': 'identity', 'arm': arm, 'variant': variant, 'question_index': qi}
            for arm in sc.ARMS for variant in sc.ARM_VARIANTS[arm] for qi in range(sc.N_QUESTIONS)]


def forward_key(row: dict) -> tuple:
    return (row['arm'], row['variant'], int(row['question_index']))


def full(args, rig_loader=None, plan=None, forwards=None, preflight_receipt=None,
         clock=time.monotonic) -> dict:
    out = Path(args.out)
    bank = sc.load_prompts(args.prompts if getattr(args, 'prompts', None) else None)
    bank_idx = prompt_index(bank)
    pf = require_pass_preflight(preflight_receipt or args.preflight, bank)
    plan = list(sc.planned_rows() if plan is None else plan)
    forwards = list(forward_plan() if forwards is None else forwards)
    planned_all = {sc.row_key(r) for r in plan}

    out.mkdir(parents=True, exist_ok=True)
    rows_path, fwd_path = out / 'rows.jsonl', out / 'forwards.jsonl'
    done_rows, dropped_rows = read_jsonl_recover(rows_path)
    done_fwd, dropped_fwd = read_jsonl_recover(fwd_path)
    if (out / 'run.json').exists():
        previous = json.loads((out / 'run.json').read_text(encoding='utf-8'))
        checks = {'prompts_sha256': bank['canonical_sha256'], 'design_sha256': sc.DESIGN_SHA256,
                  'model': MODEL, 'revision': REVISION}
        bad = [k for k, v in checks.items() if previous.get(k) != v]
        if bad:
            raise SystemExit(f'{out}/run.json was written under different {bad}; refusing to mix rows')
    have = {sc.row_key(r) for r in done_rows}
    todo = [r for r in plan if sc.row_key(r) not in have]
    have_fwd = {forward_key(r) for r in done_fwd}
    todo_fwd = [r for r in forwards if forward_key(r) not in have_fwd]

    model, tok, load = (rig_loader or load_rig)(args)
    layers = nr.find_text_layers(model)[1]
    runtime = runtime_identity(load, args.device)
    pins = {'design': sc.DESIGN_VERSION, 'design_sha256': sc.DESIGN_SHA256,
            'prompts_sha256': bank['canonical_sha256'], 'prompts_file_sha256': bank['_file_sha256'],
            'preflight_sha256': hashlib.sha256(Path(preflight_receipt or args.preflight).read_bytes()).hexdigest()}
    deadline = Deadline(getattr(args, 'deadline_min', None) or DEFAULT_FULL_DEADLINE_MIN, clock=clock)
    started = clock()
    reached = False
    print(f'full: {len(done_rows)} rows and {len(done_fwd)} forwards saved; '
          f'{len(todo)} rows and {len(todo_fwd)} forwards to run', flush=True)

    with rows_path.open('a', encoding='utf-8') as sink:
        for i, row in enumerate(todo, start=1):
            if deadline.hit():
                reached = True
                break
            append_row(sink, generate_row(row, bank_idx, bank, tok, model, args, pins=pins, runtime=runtime))
            if i % 50 == 0:
                print(f'full: {len(done_rows) + i}/{len(plan)} rows', flush=True)
                nr.spill_guard(args.device, 20.0)
    with fwd_path.open('a', encoding='utf-8') as sink:
        for row in todo_fwd:
            if deadline.hit():
                reached = True
                break
            rec = conditional_row(bank_idx, bank, tok, model, layers, row['arm'], row['variant'],
                                  int(row['question_index']), args.device)
            rec['pins'] = pins
            append_row(sink, rec)

    rows, _ = read_jsonl_recover(rows_path)
    fwd_rows, _ = read_jsonl_recover(fwd_path)
    completed = {sc.row_key(r) for r in rows} & planned_all
    status = {}
    for s in sc.QUESTION_SETS:
        for a in sc.ARMS:
            keys = {sc.row_key(r) for r in plan if r['set'] == s and r['arm'] == a}
            status[f'{s}/{a}'] = {'planned': len(keys), 'done': len(keys & completed)}
    run = {
        'part': sc.PART, 'stage': 'full', 'design': sc.DESIGN_VERSION,
        'design_sha256': sc.DESIGN_SHA256, 'model': MODEL, 'revision': REVISION,
        'prompts_sha256': bank['canonical_sha256'], 'prompts_file_sha256': bank['_file_sha256'],
        'preflight_sha256': pins['preflight_sha256'], 'preflight_result': pf.get('result'),
        'tool_sha256': tool_sha256(), 'load': load, 'runtime': runtime,
        'device': args.device, 'dtype': 'bfloat16', 'attn_implementation': load.get('attn_implementation'),
        'batch_size': 1, 'batch_size_requested': 1, 'max_new_tokens': MAX_NEW_TOKENS,
        'temperature': TEMPERATURE, 'top_p': TOP_P,
        'planned_rows': len(plan), 'rows_total': len(rows), 'new_rows_this_invocation': len(rows) - len(done_rows),
        'planned_forwards': len(forwards), 'forwards_total': len(fwd_rows),
        'new_forwards_this_invocation': len(fwd_rows) - len(done_fwd),
        'arm_status': status, 'incomplete_arms': [k for k, v in status.items() if v['done'] != v['planned']],
        'deadline_min': getattr(args, 'deadline_min', None) or DEFAULT_FULL_DEADLINE_MIN,
        'deadline_reached': reached, 'wall_time_seconds': clock() - started,
        'torn_tail_bytes_dropped': {'rows': dropped_rows, 'forwards': dropped_fwd},
        'sampling_reproducibility': (
            'One prompt per batch, batch size 1 for every call.  Each row records a SHA-256 seed from '
            '(condition, question index, sample) -- Part 66 row_seed; neutral on the identity set uses '
            "Part 214's 'unchanged:target' condition exactly, so G1 can compare row for row.  Rows are "
            'repeatable only for the same batch order, backend and library versions.'),
    }
    write_json(out / 'run.json', run)
    print(f'full: wrote {out / "run.json"} ({len(rows)} rows, {len(fwd_rows)} forwards, '
          f'deadline_reached={reached})', flush=True)
    return run


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description='Part 215 runner')
    sub = parser.add_subparsers(dest='cmd', required=True)
    for name in ('preflight', 'full'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--out', required=True)
        cmd.add_argument('--device', default='xpu')
        cmd.add_argument('--prompts', default=None, help='prompts215.json (default: this folder)')
        cmd.add_argument('--deadline-min', type=float,
                         default=DEFAULT_PREFLIGHT_DEADLINE_MIN if name == 'preflight' else DEFAULT_FULL_DEADLINE_MIN)
        if name == 'preflight':
            cmd.add_argument('--stage1', default=None, help='results/part214/stage1/stage1.json')
            cmd.add_argument('--stage2', default=None, help='results/part214/stage2/stage2.jsonl')
        else:
            cmd.add_argument('--preflight', required=True, help='preflight/preflight.json that says PASS')
    args = parser.parse_args(argv)
    if args.cmd == 'preflight':
        result = preflight(args)
        if result.get('result') != 'PASS':
            return 1
    else:
        result = full(args)
    print(json.dumps({'mode': args.cmd, 'result': result.get('result'),
                      'rows': result.get('rows_total'), 'forwards': result.get('forwards_total'),
                      'deadline_reached': result.get('deadline_reached')}, default=json_default))
    return 0


if __name__ == '__main__':
    sys.exit(main())
