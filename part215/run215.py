#!/usr/bin/env python3
"""Part 215 runner: `preflight` (G0/G1/G2-pilot/G3 and latency) and `full` (2,304 generations and
600 conditional forwards, plus the 24 neutral forwards G0 already made and bound, 648 model
forwards across the two) for the rudeness / nature-denial study on Qwen3.5-4B.

`part215/DESIGN.md` v4.1 is the contract (SHA-256 pinned in score215.py).  The canonical plan -- arms,
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
design and model AND carries the whole gate/latency/runtime/reference evidence, the expected G0
neutral-reuse file, and this runner's own tool digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import uuid
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
import evidence215  # noqa: E402  (the torch-free evidence validator both phases share; Alden #13)

MODEL = sc.MODEL
REVISION = sc.REVISION
TARGET = [(int(l), int(u)) for l, u in sc.TARGET]
MONITORS = [(int(l), int(u)) for l, u in sc.MONITORS]
GOOGLE_ID, AL_ID, T_ID = sc.GOOGLE_ID, sc.AL_ID, sc.T_ID
MAX_NEW_TOKENS = sc.MAX_NEW_TOKENS
OPENING_TOKENS = sc.OPENING_TOKENS
TEMPERATURE = 0.7
TOP_P = 0.95
# The design's gate numbers live in evidence215, so the runner and the scorer cannot drift.
PILOT_MAX_LATENCY_S = evidence215.MAX_LATENCY_S      # DESIGN.md "Size": stop above 7.5 s per call
G1_NEEDED = evidence215.G1_NEEDED                    # DESIGN.md G1: at least 45 of 48
G0_TOLERANCE = evidence215.G0_TOLERANCE              # DESIGN.md G0: within 0.02 per question
DEFAULT_PREFLIGHT_DEADLINE_MIN = 90
DEFAULT_FULL_DEADLINE_MIN = 300    # design workload is about 3.06 hours (184 min) plus forwards
# The reviewed Part 214 reference files (recorded in the receipt by resolved path and digest).
STAGE1_PATH = evidence215.DEFAULT_STAGE1_PATH
STAGE2_PATH = evidence215.DEFAULT_STAGE2_PATH

# Reviewed Part 214 reference bytes (Alden #11 item 1): pinned in score215.py and re-pinned here so a
# preflight handed any other file refuses BEFORE the model loads.  Tests replace these with their own
# synthetic fixtures' digests (a test seam, like rig_loader); the CLI always uses the reviewed bytes.
PART214_STAGE1_SHA256 = sc.PART214_STAGE1_SHA256
PART214_STAGE2_SHA256 = sc.PART214_STAGE2_SHA256
G0_REUSE_FILE = 'g0_neutral.jsonl'
G0_REUSE_N = sc.N_QUESTIONS        # 24 neutral G0 reads, reused as the full run's neutral forwards
PREFLIGHT_GENERATION_CALLS = 2 * sc.N_QUESTIONS + 96    # G1 audit 48 + G2 pilot 96
PREFLIGHT_FORWARD_CALLS = 2 * sc.N_QUESTIONS            # G0 unchanged 24 + zero 24
PIN_KEYS = sc.RUN_PIN_KEYS
RUNTIME_KEYS = sc.RUNTIME_KEYS


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


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def runtime_failures(actual, expected, label: str = 'runtime') -> list[str]:
    """The runtime fields on which two identities disagree (empty list when they match)."""
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return [label]
    return [f'{label}.{key}' for key in RUNTIME_KEYS if actual.get(key) != expected.get(key)]


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

def _stage1_from_bytes(raw: bytes, prompts_sha256: str, pinned_sha256: str | None,
                       label) -> dict:
    """Part 214's Stage 1 receipt, parsed from bytes already read and hashed by the caller."""
    pin = PART214_STAGE1_SHA256 if pinned_sha256 is None else pinned_sha256
    try:
        return evidence215.stage1_reference(raw, pin, prompts_sha256, label=str(label))
    except evidence215.EvidenceError as exc:
        raise SystemExit(str(exc)) from None


def _stage2_from_bytes(raw: bytes, prompts_sha256: str, pinned_sha256: str | None,
                       label) -> dict:
    """Part 214's 48 Stage 2 reference rows, parsed from bytes already read and hashed."""
    pin = PART214_STAGE2_SHA256 if pinned_sha256 is None else pinned_sha256
    try:
        return evidence215.stage2_reference(raw, pin, prompts_sha256, label=str(label))
    except evidence215.EvidenceError as exc:
        raise SystemExit(str(exc)) from None


def load_stage1_receipt(path, prompts_sha256: str, pinned_sha256: str | None = None) -> dict:
    """Part 214's Stage 1 receipt, refused unless it is the reviewed file and its shape is usable.

    The digest is pinned independently of the file (Alden #11 item 1: hash the reviewed bytes, not
    whatever was handed in).  This runs BEFORE the preflight model load.
    """
    return _stage1_from_bytes(Path(path).read_bytes(), prompts_sha256, pinned_sha256, path)


def gate0(model, tok, bank, bank_idx, receipt: dict, device: str) -> dict:
    """24 neutral + 24 zero teacher-forced forwards at the opening, compared per question to Part
    214's Stage 1 `unchanged` and `zero` g_ali within 0.02.

    The 24 COMPLETE neutral records are kept (`neutral_records`), so preflight can write them and the
    full run can reuse them as its `neutral` conditional rows (DESIGN.md v4.1 item 3); only scalar
    audit rows go into the receipt.
    """
    rows = []
    neutral_records = []
    layers = nr.find_text_layers(model)[1]
    for label, zero, arm in (('unchanged', False, 'neutral'), ('zero', True, 'neutral')):
        theirs = [float(x) for x in receipt['arms']['unchanged' if not zero else 'zero']['g_ali'][:sc.N_QUESTIONS]]
        for qi in range(sc.N_QUESTIONS):
            rec = conditional_row(bank_idx, bank, tok, model, layers, arm, 'v0', qi, device,
                                  zero_targets=zero)
            if not zero:
                neutral_records.append(rec)
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
            'comparisons': per, 'rows': rows, 'neutral_records': neutral_records}


# ----------------------------------------------------------------------------------------------------------------
# G1: the do-nothing audit against Part 214's Stage 2 `unchanged` rows
# ----------------------------------------------------------------------------------------------------------------

def load_stage2_unchanged(path, prompts_sha256: str, pinned_sha256: str | None = None) -> dict:
    """Part 214's 48 G1 reference rows, refused unless the reviewed file and every key/seed is right.

    Checked before the preflight model load (Alden #11 item 1): the file digest is pinned, the 48
    (question, sample) keys must be exactly the plan's, each row's seed must equal
    `sc.row_seed('neutral', 'identity', 'v0', qi, sample)`, and every row must carry token ids.
    Part 214's other questions (24..., `indirect`/`guard`) and samples 2..7 are not reference rows.
    The bytes are read ONCE here, hashed, and parsed from that same buffer.
    """
    return _stage2_from_bytes(Path(path).read_bytes(), prompts_sha256, pinned_sha256, path)


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
        'attempt_id': uuid.uuid4().hex,
        'gates': {}, 'stop_reason': None, 'full': 'not_run',
    }
    started = clock()
    try:
        g3 = sc.g3_scorer_check()
        receipt['gates']['G3'] = g3
        if not g3['pass']:
            return _stop(out, receipt, 'G3', 'attrib215 fixtures or the deliberate break failed', started, clock)

        # The Part 214 references are validated BEFORE the model loads: a wrong or tampered receipt is
        # refused with zero model calls (Alden #11 item 1).
        stage1_path = getattr(args, 'stage1', None) or str(STAGE1_PATH)
        stage2_path = getattr(args, 'stage2', None) or str(STAGE2_PATH)
        # Each reference is read ONCE, hashed, and parsed from that same buffer (Alden #13 item 2).
        stage1_raw = Path(stage1_path).read_bytes()
        stage2_raw = Path(stage2_path).read_bytes()
        receipt['part214_stage1_sha256'] = hashlib.sha256(stage1_raw).hexdigest()
        receipt['part214_stage2_sha256'] = hashlib.sha256(stage2_raw).hexdigest()
        receipt['part214_stage1_path'] = str(Path(stage1_path).resolve())
        receipt['part214_stage2_path'] = str(Path(stage2_path).resolve())
        stage1 = _stage1_from_bytes(stage1_raw, bank['source_part214_prompts_sha256'], None, stage1_path)
        stage2 = _stage2_from_bytes(stage2_raw, bank['source_part214_prompts_sha256'], None, stage2_path)

        model, tok, load = (rig_loader or load_rig)(args)
        layers = nr.find_text_layers(model)[1]
        receipt['attn_implementation'] = load.get('attn_implementation')
        receipt['runtime'] = runtime_identity(load, args.device)

        g0 = gate0(model, tok, bank, bank_idx, stage1, args.device)
        neutral_records = g0.pop('neutral_records', [])
        for rec in neutral_records:
            rec.update({'model': MODEL, 'revision': REVISION, 'runtime': receipt['runtime'],
                        'preflight_attempt_id': receipt['attempt_id'], 'source': sc.G0_REUSE_SOURCE})
        if neutral_records:
            raw_g0 = ''.join(json.dumps(rec, ensure_ascii=False, default=json_default) + '\n'
                             for rec in neutral_records).encode('utf-8')
            (out / G0_REUSE_FILE).write_bytes(raw_g0)
            receipt['g0_reuse_file'] = G0_REUSE_FILE
            receipt['g0_reuse_sha256'] = hashlib.sha256(raw_g0).hexdigest()
            receipt['g0_reuse_n'] = len(neutral_records)
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
    (out / 'STOP').unlink(missing_ok=True)      # a stale sentinel from an earlier attempt
    (out / 'PASS').write_text('PASS\n', encoding='utf-8')
    print(f'preflight: PASS ({receipt["wall_time_seconds"]:.1f} s); full allowed', flush=True)
    return receipt


def _stop(out: Path, receipt: dict, gate: str, reason: str, started: float, clock) -> dict:
    receipt['result'] = 'STOP'
    receipt['stop_reason'] = f'{gate}: {reason}'
    receipt['wall_time_seconds'] = clock() - started
    write_json(out / 'preflight.json', receipt)
    (out / 'PASS').unlink(missing_ok=True)      # a stale sentinel from an earlier attempt
    (out / 'STOP').write_text(f'STOP: {gate}: {reason}\n', encoding='utf-8')
    print(f'preflight: STOP ({gate}): {reason}; full not_run', flush=True)
    return receipt


# ----------------------------------------------------------------------------------------------------------------
# full run
# ----------------------------------------------------------------------------------------------------------------

def _runtime_receipt_failures(receipt: dict) -> list[str]:
    """The preflight runtime identity: device, dtype, attention, batch 1, token cap, sampling, versions."""
    rt = receipt.get('runtime')
    if not isinstance(rt, dict):
        return ['runtime_missing']
    fails: list[str] = []
    if not isinstance(rt.get('device'), str) or not rt.get('device'):
        fails.append('runtime.device')
    for key in ('torch', 'transformers'):
        if not isinstance(rt.get(key), str) or not rt.get(key):
            fails.append(f'runtime.{key}')
    expected = {'dtype': 'bfloat16', 'attn_implementation': 'eager', 'batch_size': 1,
                'max_new_tokens': MAX_NEW_TOKENS, 'temperature': TEMPERATURE, 'top_p': TOP_P}
    for key, want in expected.items():
        if rt.get(key) != want:
            fails.append(f'runtime.{key}')
    for key in ('device', 'dtype', 'batch_size', 'max_new_tokens', 'attn_implementation'):
        if receipt.get(key) != rt.get(key):
            fails.append(f'receipt.{key}')
    return fails


def admit_preflight(path, bank: dict) -> dict:
    """Admit a full run only on the WHOLE preflight receipt, not on its PASS label (Alden #11, #13).

    The receipt is read ONCE into a buffer, hashed, and parsed from that same buffer; `full` uses
    the SHA returned here instead of rereading the file after the model load (Alden #13 item 2).
    Checks, in addition to the identity strings: the expected artifact and mode, every gate
    recomputed from its rows by the shared torch-free validator (G0 per-question delta and pinned
    reference value, G1 token equality against the pinned Stage 2 rows, G2 broken bodies, G3 rerun,
    and the 7.5 s ceiling over all 192 measured rows), the runtime identity, the tool digest, the
    prompt-bank and Part 214 reference bytes pinned to reviewed digests, and the bound G0
    neutral-reuse file.  Anything else is refused before the full run's model loads.

    Returns `{'receipt', 'raw', 'sha256', 'g0_records'}`; `require_pass_preflight` returns just the
    receipt for callers that only need the admission decision.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(f'full refuses: no preflight receipt at {p}')
    try:
        raw = p.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        obj = json.loads(raw.decode('utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f'full refuses: preflight receipt {p} is unreadable ({exc})') from None
    if not isinstance(obj, dict):
        raise SystemExit(f'full refuses: preflight receipt {p} is not a JSON object')

    sentinel = p.parent / 'PASS'
    sentinel_ok = sentinel.exists() and sentinel.read_text(encoding='utf-8').strip().startswith('PASS')
    checks = {
        'part_is_215': obj.get('part') == sc.PART,
        'mode_is_preflight': obj.get('mode') == 'preflight',
        'result_is_PASS': obj.get('result') == 'PASS',
        'full_is_allowed': obj.get('full') == 'allowed',
        'pass_sentinel_present': sentinel_ok,
        'design_version_matches': obj.get('design') == sc.DESIGN_VERSION,
        'design_sha256_matches': obj.get('design_sha256') == sc.DESIGN_SHA256,
        'prompts_sha256_matches': obj.get('prompts_sha256') == bank['canonical_sha256']
        and obj.get('prompts_sha256') == sc.PROMPTS_CANONICAL_SHA256,
        'prompts_file_sha256_matches': obj.get('prompts_file_sha256') == bank['_file_sha256']
        and obj.get('prompts_file_sha256') == sc.PROMPTS_FILE_SHA256,
        'model_matches': obj.get('model') == MODEL,
        'revision_matches': obj.get('revision') == REVISION,
        'tool_sha256_matches': obj.get('tool_sha256') == tool_sha256(),
        'part214_stage1_pinned': obj.get('part214_stage1_sha256') == PART214_STAGE1_SHA256,
        'part214_stage2_pinned': obj.get('part214_stage2_sha256') == PART214_STAGE2_SHA256,
        'attempt_id_present': isinstance(obj.get('attempt_id'), str) and bool(obj.get('attempt_id')),
        'g0_reuse_bound': obj.get('g0_reuse_n') == G0_REUSE_N
        and isinstance(obj.get('g0_reuse_sha256'), str) and len(obj.get('g0_reuse_sha256') or '') == 64,
    }
    failures = [name for name, ok in checks.items() if not ok]
    failures += evidence215.validate_preflight(
        obj, stage1_sha256=PART214_STAGE1_SHA256, stage2_sha256=PART214_STAGE2_SHA256,
        part214_prompts_sha256=bank['source_part214_prompts_sha256'])
    failures += _runtime_receipt_failures(obj)
    try:
        records = load_g0_reuse(p, obj, bank)
        if len(records) != G0_REUSE_N:
            failures.append('g0_reuse_rows')
        # The reuse records ARE the G0 `unchanged` reads: their g_ali values must be the G0 rows'
        # `ours` values, or the file would tell a different story than the gate evidence.
        unchanged = {}
        gates = obj.get('gates') if isinstance(obj.get('gates'), dict) else {}
        g0 = gates.get('G0') if isinstance(gates.get('G0'), dict) else {}
        g0_rows = g0.get('rows') if isinstance(g0.get('rows'), list) else []
        for row in g0_rows:
            if isinstance(row, dict) and row.get('comparison') == 'unchanged':
                try:
                    unchanged[int(row['question_index'])] = row.get('ours')
                except (KeyError, TypeError, ValueError):
                    continue
        mismatched = 0
        for record in records:
            try:
                qi = int(record.get('question_index', -1))
            except (TypeError, ValueError):
                mismatched += 1
                continue
            value = unchanged.get(qi)
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or record.get('g_ali') != value:
                mismatched += 1
        if mismatched or len(unchanged) != G0_REUSE_N:
            failures.append('g0_reuse_g_ali')
    except SystemExit as exc:
        failures.append('g0_reuse_valid')
        failures.append(str(exc))
        records = []
    if failures:
        raise SystemExit(f'full refuses: preflight receipt {p} fails {failures}')
    return {'receipt': obj, 'raw': raw, 'sha256': digest, 'g0_records': records}


def require_pass_preflight(path, bank: dict) -> dict:
    """The admission decision alone (the receipt); `full` uses `admit_preflight` for the SHA."""
    return admit_preflight(path, bank)['receipt']


def load_g0_reuse(preflight_path, receipt: dict, bank: dict) -> list[dict]:
    """The 24 complete G0 neutral conditional records, verified against the receipt's digest and
    their bound identity (prompt/opening token ids, model, revision, runtime, preflight attempt id).

    These are the reads preflight already made; the full run writes them as its `neutral` rows and
    executes only the other 600 (DESIGN.md v4.1 item 3: 24 + 24 G0 + 600 = 648 model forwards).
    """
    name = receipt.get('g0_reuse_file')
    digest = receipt.get('g0_reuse_sha256')
    if not isinstance(name, str) or not name:
        raise SystemExit('preflight receipt does not bind a G0 reuse file')
    if not isinstance(digest, str) or len(digest) != 64:
        raise SystemExit('preflight receipt does not bind a G0 reuse digest')
    if receipt.get('g0_reuse_n') != G0_REUSE_N:
        raise SystemExit(f'preflight receipt binds {receipt.get("g0_reuse_n")} G0 reuse records, '
                         f'want {G0_REUSE_N}')
    path = Path(preflight_path).parent / name
    if not path.exists():
        raise SystemExit(f'G0 reuse file {path} is missing')
    raw = path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != digest:
        raise SystemExit(f'{path}: SHA-256 {got} does not match the receipt {digest}')
    # One buffer: the bytes hashed above are the bytes parsed here (Alden #13 item 2).
    try:
        records = evidence215.parse_jsonl_bytes(raw, str(path))
    except evidence215.EvidenceError as exc:
        raise SystemExit(str(exc)) from None
    if len(records) != G0_REUSE_N:
        raise SystemExit(f'{path}: {len(records)} G0 reuse records, want {G0_REUSE_N}')
    idx = prompt_index(bank)
    opening = [int(t) for t in bank['openings']['en_maker']['token_ids']]
    seen = set()
    for rec in records:
        arm, variant = rec.get('arm'), rec.get('variant')
        try:
            qi = int(rec.get('question_index', -1))
        except (TypeError, ValueError):
            raise SystemExit(f'{path}: a G0 reuse record has no usable question index') from None
        if (arm, variant) != ('neutral', 'v0') or not (0 <= qi < sc.N_QUESTIONS):
            raise SystemExit(f'{path}: unexpected G0 reuse key {arm}/{variant}/q{qi}')
        key = (arm, variant, qi)
        if key in seen:
            raise SystemExit(f'{path}: G0 reuse key {key} appears twice')
        seen.add(key)
        prompt = prompt_for(idx, 'identity', arm, variant, qi)
        if [int(t) for t in rec.get('prompt_token_ids') or []] != [int(t) for t in prompt['token_ids']]:
            raise SystemExit(f'{path}: G0 reuse q{qi} prompt token ids differ from the bank')
        if [int(t) for t in rec.get('opening_token_ids') or []] != opening:
            raise SystemExit(f'{path}: G0 reuse q{qi} opening token ids differ from the bank')
        if rec.get('set') != 'identity':
            raise SystemExit(f'{path}: G0 reuse q{qi} is not an identity row')
        if rec.get('model') != MODEL or rec.get('revision') != REVISION:
            raise SystemExit(f'{path}: G0 reuse q{qi} names {rec.get("model")}@{rec.get("revision")}')
        if rec.get('preflight_attempt_id') != receipt.get('attempt_id'):
            raise SystemExit(f'{path}: G0 reuse q{qi} belongs to attempt '
                             f'{rec.get("preflight_attempt_id")}, not {receipt.get("attempt_id")}')
        if rec.get('source') != sc.G0_REUSE_SOURCE:
            raise SystemExit(f'{path}: G0 reuse q{qi} is not marked {sc.G0_REUSE_SOURCE}')
        if not _is_number(rec.get('g_ali')) or not _is_number(rec.get('g_al')):
            raise SystemExit(f'{path}: G0 reuse q{qi} has no g_ali/g_al')
        if len(rec.get('opening') or []) != OPENING_TOKENS:
            raise SystemExit(f'{path}: G0 reuse q{qi} has {len(rec.get("opening") or [])} opening '
                             f'positions, want {OPENING_TOKENS}')
        bad_rt = runtime_failures(rec.get('runtime'), receipt.get('runtime'), 'g0_reuse.runtime')
        if bad_rt:
            raise SystemExit(f'{path}: G0 reuse q{qi} was read under different {bad_rt}')
    if seen != {(arm, variant, qi) for arm, variant in (('neutral', 'v0'),)
                for qi in range(sc.N_QUESTIONS)}:
        raise SystemExit(f'{path}: G0 reuse keys are not the 24 neutral identity reads')
    return records


def forward_plan() -> list[dict]:
    return [{'set': 'identity', 'arm': arm, 'variant': variant, 'question_index': qi}
            for arm in sc.ARMS for variant in sc.ARM_VARIANTS[arm] for qi in range(sc.N_QUESTIONS)]


def forward_key(row: dict) -> tuple:
    return (row['arm'], row['variant'], int(row['question_index']))


def _validate_saved(records: list[dict], kind: str, pins: dict, runtime: dict) -> None:
    """Refuse to append to rows written by a different attempt or a different runtime.

    Every saved row must carry this invocation's pin set and an equal runtime identity; a mismatch is
    a mixed out dir, not a resume (Alden #11 item 2, same rule at resume as at scoring).
    """
    for index, record in enumerate(records):
        saved = record.get('pins')
        if not isinstance(saved, dict):
            raise SystemExit(f'full refuses: saved {kind} row {index} has no pins; refusing to mix attempts')
        bad = [key for key in PIN_KEYS if saved.get(key) != pins.get(key)]
        if bad:
            raise SystemExit(f'full refuses: saved {kind} row {index} was written under different '
                             f'{bad}; refusing to mix attempts')
        saved_runtime = record.get('runtime')
        if not isinstance(saved_runtime, dict):
            raise SystemExit(f'full refuses: saved {kind} row {index} has no runtime identity; '
                             f'refusing to mix attempts')
        bad_runtime = runtime_failures(saved_runtime, runtime)
        if bad_runtime:
            raise SystemExit(f'full refuses: saved {kind} row {index} was run under different '
                             f'{bad_runtime}; refusing to mix attempts')


def full(args, rig_loader=None, plan=None, forwards=None, preflight_receipt=None,
         clock=time.monotonic) -> dict:
    out = Path(args.out)
    bank = sc.load_prompts(args.prompts if getattr(args, 'prompts', None) else None)
    bank_idx = prompt_index(bank)
    preflight_path = Path(preflight_receipt or args.preflight)
    admitted = admit_preflight(preflight_path, bank)
    pf = admitted['receipt']
    # The SHA is the one admission hashed, carried forward: run.json and every row pin it, and the
    # file is NOT reread after the model load (Alden #13 item 2).
    preflight_sha256 = admitted['sha256']
    # The 24 G0 neutral reads preflight already made: verified against the receipt's digest and their
    # bound identity, then written as this run's neutral forwards and NOT run again.  They were read
    # once during admission; `full` does not reread the reuse file either.
    g0_records = admitted['g0_records']
    g0_sha256 = pf['g0_reuse_sha256']
    reuse_keys = {forward_key(r) for r in g0_records}
    plan = list(sc.planned_rows() if plan is None else plan)
    planned_all = {sc.row_key(r) for r in plan}
    planned_fwd = forward_plan()
    planned_fwd_keys = {forward_key(r) for r in planned_fwd}
    requested_fwd = list(planned_fwd if forwards is None else forwards)
    off_plan = [r for r in requested_fwd if forward_key(r) not in planned_fwd_keys]
    if off_plan:
        raise SystemExit(f'full refuses: {len(off_plan)} forward rows are not in the 624-key plan')
    runnable_fwd = [r for r in requested_fwd if forward_key(r) not in reuse_keys]

    out.mkdir(parents=True, exist_ok=True)
    rows_path, fwd_path = out / 'rows.jsonl', out / 'forwards.jsonl'
    done_rows, dropped_rows = read_jsonl_recover(rows_path)
    done_fwd, dropped_fwd = read_jsonl_recover(fwd_path)

    # Built before the model load from the SHA admission hashed: no receipt file is reread here.
    pins = {'design': sc.DESIGN_VERSION, 'design_sha256': sc.DESIGN_SHA256,
            'prompts_sha256': bank['canonical_sha256'], 'prompts_file_sha256': bank['_file_sha256'],
            'preflight_sha256': preflight_sha256,
            'model': MODEL, 'revision': REVISION}
    identity = dict(pins, g0_reuse_sha256=g0_sha256)

    model, tok, load = (rig_loader or load_rig)(args)
    layers = nr.find_text_layers(model)[1]
    runtime = runtime_identity(load, args.device)

    # A resume continues THIS attempt only: the saved run.json, every saved row and every saved
    # forward must agree with this invocation's pins and runtime before anything is appended.
    if (out / 'run.json').exists():
        previous = json.loads((out / 'run.json').read_bytes().decode('utf-8'))
        bad = [key for key in identity if previous.get(key) != identity[key]]
        bad_runtime = [f'runtime.{key}' for key in RUNTIME_KEYS
                       if (previous.get('runtime') or {}).get(key) != runtime.get(key)]
        if bad or bad_runtime:
            raise SystemExit(f'{out}/run.json was written under different {bad + bad_runtime}; '
                             f'refusing to mix attempts')
    _validate_saved(done_rows, 'generation', pins, runtime)
    _validate_saved(done_fwd, 'forward', pins, runtime)
    for index, record in enumerate(g0_records):
        bad_runtime = runtime_failures(record.get('runtime'), runtime, 'g0_reuse.runtime')
        if bad_runtime:
            raise SystemExit(f'full refuses: preflight G0 reuse record {index} was read under '
                             f'different {bad_runtime}; refusing to mix attempts')

    have = {sc.row_key(r) for r in done_rows}
    todo = [r for r in plan if sc.row_key(r) not in have]
    have_fwd = {forward_key(r) for r in done_fwd}
    todo_fwd = [r for r in runnable_fwd if forward_key(r) not in have_fwd]
    reuse_to_write = [r for r in g0_records if forward_key(r) not in have_fwd]

    deadline = Deadline(getattr(args, 'deadline_min', None) or DEFAULT_FULL_DEADLINE_MIN, clock=clock)
    started = clock()
    reached = False
    print(f'full: {len(done_rows)} rows and {len(done_fwd)} forwards saved; '
          f'{len(todo)} rows, {len(reuse_to_write)} reused neutral forwards and '
          f'{len(todo_fwd)} forwards to run', flush=True)

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
        for record in reuse_to_write:
            stored = dict(record)
            stored['pins'] = pins
            stored['g0_sha256'] = g0_sha256
            append_row(sink, stored)
        for row in todo_fwd:
            if deadline.hit():
                reached = True
                break
            rec = conditional_row(bank_idx, bank, tok, model, layers, row['arm'], row['variant'],
                                  int(row['question_index']), args.device)
            rec['pins'] = pins
            rec['runtime'] = runtime
            rec['source'] = 'full'
            append_row(sink, rec)

    rows, _ = read_jsonl_recover(rows_path)
    fwd_rows, _ = read_jsonl_recover(fwd_path)
    completed = {sc.row_key(r) for r in rows} & planned_all
    status = {}
    for s in sc.QUESTION_SETS:
        for a in sc.ARMS:
            keys = {sc.row_key(r) for r in plan if r['set'] == s and r['arm'] == a}
            status[f'{s}/{a}'] = {'planned': len(keys), 'done': len(keys & completed)}
    # Model-call accounting: preflight made 48 G0 forwards (24 neutral reused here + 24 zero); this
    # run executes the other 600 readout forwards.  648 in total, as v4.1 states.
    model_forwards_preflight = 2 * sc.N_QUESTIONS
    run = {
        'part': sc.PART, 'stage': 'full', 'design': sc.DESIGN_VERSION,
        'design_sha256': sc.DESIGN_SHA256, 'model': MODEL, 'revision': REVISION,
        'prompts_sha256': bank['canonical_sha256'], 'prompts_file_sha256': bank['_file_sha256'],
        'preflight_sha256': pins['preflight_sha256'], 'preflight_result': pf.get('result'),
        'preflight_attempt_id': pf.get('attempt_id'), 'g0_reuse_sha256': g0_sha256,
        'tool_sha256': tool_sha256(), 'load': load, 'runtime': runtime,
        'device': args.device, 'dtype': 'bfloat16', 'attn_implementation': load.get('attn_implementation'),
        'batch_size': 1, 'batch_size_requested': 1, 'max_new_tokens': MAX_NEW_TOKENS,
        'temperature': TEMPERATURE, 'top_p': TOP_P,
        'planned_rows': len(plan), 'rows_total': len(rows), 'new_rows_this_invocation': len(rows) - len(done_rows),
        'planned_forwards': len(planned_fwd), 'forwards_total': len(fwd_rows),
        'new_forwards_this_invocation': len(fwd_rows) - len(done_fwd),
        'g0_reuse_records': len(g0_records), 'forwards_runnable_this_invocation': len(runnable_fwd),
        'model_forwards_preflight': model_forwards_preflight,
        'model_forwards_full_this_invocation': len(todo_fwd),
        'model_forwards_total_across_preflight_and_full': model_forwards_preflight + len(runnable_fwd),
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
