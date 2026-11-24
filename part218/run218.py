#!/usr/bin/env python3
"""Part 218 v4 runner: random residual pushes on OLMo 2 1B SFT / DPO / Instruct.

Spec: `part218/DESIGN.md` (v3 frozen) plus **Amendment 1** at the end, which
governs where it differs.  One self-contained script: it imports torch,
transformers and the standard library only.  The 44 held-out prompts NEW44C sit
beside it in `prompts218-v4.json`, so the whole `part218/` folder plus that file
runs on a Colab VM.

Colab (from the folder that holds `part218/`, one shared --out root):

    python part218/run218.py --mode pilot --out /content/out218
    python part218/run218.py --mode gate --model SFT --out /content/out218
    python part218/run218.py --mode gate --model DPO --out /content/out218
    python part218/run218.py --mode full --model SFT      --out /content/out218
    python part218/run218.py --mode full --model DPO      --out /content/out218
    python part218/run218.py --mode full --model Instruct --out /content/out218

Output layout (one shared `directions.pt` at the root, reused byte for byte):

    <out>/directions.pt          20 unit Gaussian vectors, float32 CPU
    <out>/pilot/answers.jsonl    one line per generated row (Instruct, layer 5)
    <out>/gate/SFT|DPO/...       the zero-push gate pass (layer 5)
    <out>/SFT|DPO|Instruct/...   the main run

Every mode is resumable: a batch is skipped when all of its rows are already in
`answers.jsonl`, keyed by (model, layer, dose, item_index) and row id.

v4 (Amendment 1) design points that live in this file:
  * 44 prompts x 2 openings = 88 items; item_index = 2 * prompt_index +
    opening_index, fixed order 0..87;
  * the model input is the chat template's generation prompt plus the opening
    with NO trailing space; the answer text is `opening + continuation`
    (skip_special_tokens=True, unstripped);
  * one batch = one item x 41 rows: 40 pushed directions (seeds 0-19, both
    signs) plus the zero row, in the pilot and the main run alike;
  * push = d x the model's own median residual-output norm at that layer over
    the 88 items' positions, position 0 excluded, opening tokens included,
    added to the decoder layer's output at every position;
  * `--mode gate` records 41 zero-push rows per item for SFT and DPO (the hook
    installed, adding exactly zero); it never builds a pushed vector;
  * the no-hook null is rerun at the start and the end of every pushed run, and
    every hooked batch's zero row is compared with the start null;
  * a null mismatch stops the model and is written to `run.json`: never a
    silent pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROMPTS_PATH = HERE / 'prompts218-v4.json'
PINS_PATH = HERE / 'pins.json'

# Amendment 1, item 2: the two maker-slot openings, appended with no trailing
# space.  The model generates the leading space of the continuation.
OPENINGS = ('I was developed by', 'I was built by')

# The three full-mode checkpoints.  Full 40-hex revisions, the same as
# part218/pins.json and DESIGN.md (the 12-hex prefixes in the design each
# matched exactly one Hub commit when the pins were resolved).
MODELS = {
    'SFT': ('allenai/OLMo-2-0425-1B-SFT',
            '0d85a3d037876ce6ac7d4311d994400fc66ac27f'),
    'DPO': ('allenai/OLMo-2-0425-1B-DPO',
            'c4b0485961ab24c2433b090f3b922f0913a9290f'),
    'Instruct': ('allenai/OLMo-2-0425-1B-Instruct',
                 '48d788eca847d4d7548f375ad03d3c9312f6139e'),
}

# Method ladder (DESIGN.md, Dose).
LADDER = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
# Pilot ladder, fixed before the pilot (DESIGN.md, Size; Amendment 1 item 8).
PILOT_DOSES = (0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0, 1.5)
PILOT_LAYER = 5
FULL_LAYERS = (5, 10)                  # of 16
FULL_SEEDS = tuple(range(20))          # forty directions with both signs
PILOT_SEEDS = FULL_SEEDS               # Amendment 1 item 3: 40 + zero = 41
N_DRAWS = 20                           # the directions file always holds 20
MAX_NEW = 48
N_PROMPTS = 44
N_OPENINGS = 2
N_ITEMS = N_PROMPTS * N_OPENINGS       # 88
ROWS_PER_BATCH = 2 * len(FULL_SEEDS) + 1   # 41, pinned by Amendment 1
GATE_LAYER = 5
GATE_MODELS = ('SFT', 'DPO')

ZERO_ROW = {'seed': None, 'sign': 0, 'direction_id': 'zero'}


# ---------------------------------------------------------------------------
# Items (Amendment 1, items 1 and 3)
# ---------------------------------------------------------------------------

def load_prompts(path: Path | str = PROMPTS_PATH) -> list[str]:
    """Read prompts218-v4.json and check its shape."""
    prompts = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(prompts, list) or len(prompts) != N_PROMPTS:
        raise SystemExit(f'{path}: expected a list of {N_PROMPTS} prompts, '
                         f'got {type(prompts).__name__} of {len(prompts)}')
    if not all(isinstance(q, str) and q.strip() for q in prompts):
        raise SystemExit(f'{path}: every prompt must be a non-empty string')
    if len(set(prompts)) != N_PROMPTS:
        raise SystemExit(f'{path}: the {N_PROMPTS} prompts must be distinct')
    return prompts


def item_spec(prompt_index: int, opening_index: int) -> dict:
    """One item: a prompt asked with one opening.  Fixed index 2p + o."""
    if not 0 <= opening_index < N_OPENINGS:
        raise ValueError(f'opening_index {opening_index} outside 0..{N_OPENINGS - 1}')
    return {
        'item_index': int(2 * prompt_index + opening_index),
        'prompt_index': int(prompt_index),
        'opening_index': int(opening_index),
        'opening': OPENINGS[opening_index],
    }


def items(n_prompts: int = N_PROMPTS) -> list[dict]:
    """Every (prompt, opening) pair in the fixed item order 0..87."""
    return [item_spec(prompt_index, opening_index)
            for prompt_index in range(n_prompts)
            for opening_index in range(N_OPENINGS)]


def answer_text(opening: str, continuation: str) -> str:
    """Amendment 1 item 2: `text = opening + continuation` exactly."""
    return opening + continuation


# ---------------------------------------------------------------------------
# Small pure helpers (unit-tested without a model)
# ---------------------------------------------------------------------------

def pins() -> dict:
    """The checkpoint pins, from pins.json (MODELS is the fallback)."""
    if PINS_PATH.exists():
        data = json.loads(PINS_PATH.read_text(encoding='utf-8'))
        out = {}
        for key, (repo, revision) in MODELS.items():
            entry = data[key]
            if (entry['repo'], entry['revision']) != (repo, revision):
                raise SystemExit(
                    f'pins.json disagrees with run218.MODELS for {key}: '
                    f'{entry} vs {(repo, revision)}')
            out[key] = {'repo': repo, 'revision': revision}
        return out
    return {key: {'repo': repo, 'revision': rev}
            for key, (repo, rev) in MODELS.items()}


def direction_specs(seeds) -> list[dict]:
    """Row ids in batch order: seed 0 +, seed 0 -, seed 1 +, ..."""
    specs = []
    for seed in seeds:
        specs.append({'seed': int(seed), 'sign': 1,
                      'direction_id': f'seed{int(seed)}+'})
        specs.append({'seed': int(seed), 'sign': -1,
                      'direction_id': f'seed{int(seed)}-'})
    return specs


def n_rows(seeds) -> int:
    """Pushed rows plus the zero row."""
    return 2 * len(tuple(seeds)) + 1


def directions_tensor(hidden: int, n_draws: int = N_DRAWS) -> torch.Tensor:
    """20 unit Gaussian rows, float32 on the CPU, one generator seed per row.

    Drawn once and saved; every model (and the pilot) reuses the same file.
    Row `s` is the unit vector for seed `s`; both signs are built at use time.
    """
    rows = []
    for seed in range(n_draws):
        generator = torch.Generator().manual_seed(seed)
        vec = torch.randn(hidden, generator=generator, dtype=torch.float32)
        rows.append(vec / vec.norm())
    return torch.stack(rows)


def load_or_create_directions(path: Path, hidden: int) -> tuple[torch.Tensor, str, bool]:
    """Load `directions.pt` byte for byte, or draw and save it once.

    Returns (tensor, sha256, created).  A stored file with a different hidden
    size is an error, not a silent redraw: every model must share one file.
    """
    path = Path(path)
    if path.exists():
        tensor = torch.load(path, map_location='cpu', weights_only=True)
        if tensor.ndim != 2 or tensor.shape != (N_DRAWS, hidden):
            raise SystemExit(
                f'{path} has shape {tuple(tensor.shape)}; this run needs '
                f'({N_DRAWS}, {hidden}).  Use a fresh --out for a different '
                f'hidden size (the design requires one shared file).')
        return tensor.float(), hashlib.sha256(path.read_bytes()).hexdigest(), False
    tensor = directions_tensor(hidden)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(tensor, tmp)
    os.replace(tmp, path)
    return tensor, hashlib.sha256(path.read_bytes()).hexdigest(), True


def dose_vectors(directions: torch.Tensor, seeds, dose: float,
                 median_norm: float) -> torch.Tensor:
    """Batch rows for one (layer, dose): both signs of each seed, then zero.

    Each pushed row has norm `dose * median_norm`; the last row is exactly 0.
    """
    scale = float(dose) * float(median_norm)
    rows = []
    for seed in seeds:
        vec = directions[int(seed)].to(torch.float32)
        rows.append(vec * scale)
        rows.append(-vec * scale)
    rows.append(torch.zeros_like(rows[0]))
    return torch.stack(rows)


def gate_vectors(hidden: int, rows: int = ROWS_PER_BATCH) -> torch.Tensor:
    """The gate pass's batch rows: `rows` zero vectors, exactly zero.

    Amendment 1 item 8 / the gate-only pass: the hook is installed and adds
    zero for every row, so no pushed SFT or DPO answer exists before the main
    run.  This function is the only vector source the gate pass uses.
    """
    return torch.zeros(int(rows), int(hidden), dtype=torch.float32)


def row_key(record: dict) -> tuple:
    return (record['model'], int(record['layer']), float(record['dose']),
            int(record['item_index']), record['row_id'])


def gate_row_ids(rows: int = ROWS_PER_BATCH) -> list[str]:
    return [f'zero{index:02d}' for index in range(int(rows))]


def planned_rows(mode: str, layers, doses,
                 n_items: int = N_ITEMS) -> int:
    """Rows the design plans for one run of this mode."""
    if mode == 'gate':
        return len(layers) * len(doses) * n_items * ROWS_PER_BATCH
    return len(layers) * len(doses) * n_items * n_rows(FULL_SEEDS)


def find_decoder_layers(model):
    """The decoder-block ModuleList (job.py's heuristic, tightened)."""
    best = None
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.ModuleList) and len(module)
                and hasattr(module[0], 'mlp')
                and hasattr(module[0], 'self_attn')):
            key = ('language' in name, len(module))
            if best is None or key > best[0]:
                best = (key, name, module)
    if best is None:
        raise SystemExit('no decoder layer list found on the model')
    return best[2]


def count_layer_hooks(layers) -> int:
    """Forward/backward hooks still attached to the decoder blocks."""
    total = 0
    for layer in layers:
        total += len(getattr(layer, '_forward_hooks', {}))
        total += len(getattr(layer, '_forward_pre_hooks', {}))
        total += len(getattr(layer, '_backward_hooks', {}))
    return total


# ---------------------------------------------------------------------------
# Model, prompts, generation
# ---------------------------------------------------------------------------

def build_prompt(tok, question: str) -> str:
    """The checkpoint's own chat template, no system prompt, generation prompt."""
    messages = [{'role': 'user', 'content': question}]
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def item_text(tok, question: str, opening: str) -> str:
    """Amendment 1 item 2: the generation prompt plus the opening, no space."""
    return build_prompt(tok, question) + opening


def encode_item(tok, question: str, opening: str) -> torch.Tensor:
    """Encode one item: `build_prompt(...) + opening`, add_special_tokens=False."""
    text = item_text(tok, question, opening)
    ids = tok(text, add_special_tokens=False, return_tensors='pt')['input_ids']
    if ids.numel() == 0:
        raise SystemExit('chat template produced an empty prompt')
    return ids[0]


def encode_prompt(tok, question: str) -> torch.Tensor:
    """v3 helper kept for callers that only need the bare generation prompt."""
    return encode_item(tok, question, '')


def load_model(model_key: str, device, dtype, attn_implementation=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo, revision = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    kwargs = {'revision': revision, 'dtype': dtype}
    if attn_implementation:
        kwargs['attn_implementation'] = attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
    except (ValueError, KeyError):
        # A multimodal wrapper for this checkpoint family; job.py's fallback.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(repo, **kwargs)
    model.to(device).eval()
    torch.set_grad_enabled(False)
    return tok, model


def load_tiny(config_dir: str | Path, device, dtype, seed: int = 0):
    """A random-weight model built from a tiny config dir + its real tokenizer.

    The smoke builds such a dir from the real Instruct config with hidden size
    about 64 and 16 layers; no weights are downloaded.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    config = AutoConfig.from_pretrained(str(config_dir))
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config)
    model.to(device=device, dtype=dtype).eval()
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(str(config_dir))
    return tok, model


def tiny_config(config_dir: str | Path, hidden: int = 64, n_layers: int = 16,
                intermediate: int = 256, heads: int = 4,
                revision: str | None = None, local_files_only: bool = False):
    """An in-memory small copy of a checkpoint config (for --selftest)."""
    from transformers import AutoConfig
    kwargs = {'local_files_only': local_files_only}
    if revision:
        kwargs['revision'] = revision
    config = AutoConfig.from_pretrained(str(config_dir), **kwargs)
    config.hidden_size = hidden
    config.num_hidden_layers = n_layers
    config.intermediate_size = intermediate
    config.num_attention_heads = heads
    config.num_key_value_heads = heads
    return config


def tokenizer_report(tok, pad_id: int) -> dict:
    """Pad/eos/unk handling, recorded in run.json.

    OLMo's unknown token *is* its end token (Procedure 5, known traps), so the
    report says so explicitly and the generate call uses `pad_id` rather than
    letting the model fall back to eos.
    """
    return {
        'pad_token': tok.pad_token, 'pad_token_id': tok.pad_token_id,
        'eos_token': tok.eos_token, 'eos_token_id': tok.eos_token_id,
        'unk_token': tok.unk_token, 'unk_token_id': tok.unk_token_id,
        'generate_pad_token_id': int(pad_id),
        'unk_is_eos': (tok.unk_token_id is not None
                       and tok.unk_token_id == tok.eos_token_id),
    }


def generate_batch(model, batch: torch.Tensor, max_new: int, pad_id: int):
    """Greedy batch generation; returns (sequences, first-step logits)."""
    out = model.generate(
        input_ids=batch,
        attention_mask=torch.ones_like(batch),
        do_sample=False,
        max_new_tokens=max_new,
        pad_token_id=pad_id,
        use_cache=True,
        output_scores=True,
        return_dict_in_generate=True,
    )
    return out.sequences, out.scores[0]


def parse_row(tok, sequence: torch.Tensor, prompt_len: int, max_new: int,
              pad_id: int, eos_id: int | None = None) -> dict:
    """Trim one generated row to its real tokens and decode the continuation.

    Amendment 1 item 2: the continuation is decoded with
    skip_special_tokens=True and kept exactly as decoded (no strip); the answer
    text is `opening + continuation`, built by `answer_text`.
    """
    generated = sequence[prompt_len:].tolist()
    while generated and generated[-1] == pad_id:
        generated.pop()
    continuation = tok.decode(generated, skip_special_tokens=True)
    return {
        'token_ids': generated,
        'continuation': continuation,
        'n_new_tokens': len(generated),
        'max_new_tokens': int(max_new),
        # An answer that ends on EOS exactly at the cap ended naturally
        # (GLM review LOW-1); only a row that ran out of room hit the cap.
        'hit_cap': (len(generated) >= int(max_new)
                    and not (eos_id is not None and generated
                             and generated[-1] == eos_id)),
    }


def make_push_hook(layer, vecs: torch.Tensor, probe: dict | None = None):
    """Add one [row, hidden] vector per batch row at every position.

    `probe`, when given, records what the hook actually added on its first
    call: the zero row's delta must be exactly 0 and a pushed row's delta
    must be nonzero.  This is a real hooked-forward check, not a restatement
    of the input vector.
    """
    state = {'calls': 0}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        delta = vecs.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(1)
        pushed = hidden + delta
        if probe is not None and state['calls'] == 0:
            added = pushed - hidden
            probe['pushed_rows'] = int(added.shape[0] - 1)
            probe['pushed_delta_abs_max'] = float(added[:-1].abs().max())
            probe['zero_delta_abs_max'] = float(added[-1].abs().max())
            probe['zero_row_exact'] = bool(torch.equal(
                added[-1], torch.zeros_like(added[-1])))
            probe['input_zero_row_abs_max'] = float(vecs[-1].abs().max())
            probe['input_pushed_row_norm_min'] = float(
                vecs[:-1].norm(dim=-1).min())
            probe['hidden_shape'] = list(hidden.shape)
        state['calls'] += 1
        if isinstance(output, tuple):
            return (pushed,) + tuple(output[1:])
        if isinstance(output, list):
            return [pushed, *output[1:]]
        return pushed

    return layer.register_forward_hook(hook), state


def median_norms(model, layers, item_ids, chosen, device) -> dict:
    """Median residual-output norm per layer, over all item prompts' positions.

    Amendment 1 item 7: the 88 item prompts (opening tokens included), position
    0 excluded, one median per layer.
    """
    store = {int(layer): [] for layer in chosen}
    handles = []
    for layer in chosen:
        def hook(_module, _inputs, output, layer=layer):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            store[int(layer)].append(
                hidden[0, 1:].float().norm(dim=-1).cpu())
        handles.append(layers[int(layer)].register_forward_hook(hook))
    try:
        for ids in item_ids:
            batch = ids.unsqueeze(0).to(device)
            model(input_ids=batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return {str(layer): float(torch.cat(store[int(layer)]).median())
            for layer in chosen}


# ---------------------------------------------------------------------------
# Answer-file resume
# ---------------------------------------------------------------------------

def read_answers(path: Path) -> tuple[dict, int]:
    """Read answers.jsonl: row_key -> record (last write wins), bad lines count.

    The last-write-wins rule makes a crash in the middle of a batch safe: the
    batch is regenerated and the duplicate row is replaced on the next read.
    """
    rows: dict = {}
    bad = 0
    if not path.exists():
        return rows, bad
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key = row_key(record)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                bad += 1
                continue
            rows[key] = record
    return rows, bad


def write_answers(path: Path, rows: dict) -> None:
    """Rewrite a compacted answers.jsonl (used after a partial batch crash)."""
    ordered = sorted(rows.values(),
                     key=lambda r: (int(r['layer']), float(r['dose']),
                                    int(r['item_index']),
                                    int(r.get('row_index', 0))))
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def completed_batches(rows: dict, required_ids: list[str]) -> set:
    """Batch keys whose every row id is present in the answer file."""
    need = set(required_ids)
    done: dict = {}
    for record in rows.values():
        key = (record['model'], int(record['layer']), float(record['dose']),
               int(record['item_index']))
        done.setdefault(key, set()).add(record['row_id'])
    return {key for key, ids in done.items() if need <= ids}


# ---------------------------------------------------------------------------
# No-hook null
# ---------------------------------------------------------------------------

def last_row_logits(scores: torch.Tensor) -> torch.Tensor:
    """One row of first-step logits, copied off its batch buffer.

    `scores[-1]` is a view into the full [rows, vocab] score tensor.  Keeping
    the view alive kept the whole batch buffer (41 x 100352 floats) per item;
    at v4's 88 items that is about 1.4 GB per saved set.  The clone keeps only
    the row, which is all the null comparison reads.
    """
    return scores[-1].detach().to('cpu').clone()


def null_pass(model, item_ids, rows_per_batch: int, device, max_new: int,
              pad_id: int, label: str) -> list[dict]:
    """Per item, one unhooked batch of the same shape; keep the zero slot.

    The slot is the last row: the zero row of the hooked batch it will be
    compared with.  All rows in this batch are the same item, so the last row
    is the model's plain answer at this batch shape.
    """
    results = []
    started = time.time()
    for item_index, ids in enumerate(item_ids):
        batch = ids.repeat(rows_per_batch, 1).to(device)
        sequences, scores0 = generate_batch(model, batch, max_new, pad_id)
        generated = sequences[-1, ids.numel():].tolist()
        while generated and generated[-1] == pad_id:
            generated.pop()
        results.append({
            'item_index': item_index,
            'token_ids': generated,
            'logits': last_row_logits(scores0),
        })
        print(f'null-{label} {item_index + 1}/{len(item_ids)} '
              f'i{item_index} {time.time() - started:.1f}s', flush=True)
    return results


def compare_null(reference: dict, token_ids, logits) -> dict:
    """Exact comparison of a zero slot against a stored unhooked slot."""
    ids_ok = list(reference['token_ids']) == list(token_ids)
    ref_logits = reference['logits']
    same_shape = tuple(ref_logits.shape) == tuple(logits.shape)
    logits_ok = bool(same_shape and torch.equal(ref_logits, logits))
    max_abs = None
    if same_shape:
        max_abs = float((ref_logits.float() - logits.float()).abs().max())
    return {'ids_ok': ids_ok, 'logits_ok': logits_ok,
            'max_abs_logit_diff': max_abs}


def null_summary(results: list[dict], expected: int) -> dict:
    return {'items': len(results), 'expected_items': expected,
            'all_items_ran': len(results) == expected}


def merge_comparisons(comparisons: list[dict]) -> dict:
    """Aggregate per-item comparison dicts for run.json."""
    n = len(comparisons)
    ids = sum(1 for c in comparisons if c['ids_ok'])
    logits = sum(1 for c in comparisons if c['logits_ok'])
    diffs = [c['max_abs_logit_diff'] for c in comparisons
             if c['max_abs_logit_diff'] is not None]
    return {'comparisons': n, 'ids_matched': ids, 'logits_matched': logits,
            'all_ok': n > 0 and ids == n and logits == n,
            'max_abs_logit_diff': max(diffs) if diffs else None}


# ---------------------------------------------------------------------------
# run.json
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + '\n',
                   encoding='utf-8')
    os.replace(tmp, path)


def version_report() -> dict:
    import transformers
    return {'torch': torch.__version__,
            'transformers': transformers.__version__,
            'python': sys.version.split()[0]}


# ---------------------------------------------------------------------------
# The gate-only pass (Amendment 1 item 8; no pushed row may be generated)
# ---------------------------------------------------------------------------

def run_gate_pass(args, model_key: str, run_dir: Path, tok, model, layers,
                  hidden: int, item_specs: list[dict], item_ids: list[torch.Tensor],
                  pad_id: int, repo: str, revision: str, tiny: bool,
                  pin_table: dict, started_at: float) -> int:
    """One batch per item: 41 zero-push rows, the hook installed adding zero."""
    answers_path = run_dir / 'answers.jsonl'
    run_json_path = run_dir / 'run.json'
    layer = GATE_LAYER
    batch_rows = ROWS_PER_BATCH
    required_ids = gate_row_ids(batch_rows)
    vecs_cpu = gate_vectors(hidden, batch_rows)
    if float(vecs_cpu.abs().max()) != 0.0:
        raise SystemExit('gate vectors are not exactly zero')

    tok_report = tokenizer_report(tok, pad_id)
    sample_ids = item_ids[0]
    tok_report['sample_prompt_tokens'] = int(sample_ids.numel())
    tok_report['sample_prompt_decodes'] = bool(
        tok.decode(sample_ids, skip_special_tokens=True).strip())

    meta = {
        'part': 218,
        'mode': 'gate',
        'model_key': model_key,
        'repo': repo,
        'revision': revision,
        'tiny': tiny,
        'pins': pin_table,
        'versions': version_report(),
        'device': str(args.device),
        'device_name': (torch.cuda.get_device_name(0)
                        if torch.device(args.device).type == 'cuda' else None),
        'dtype': args.dtype,
        'attn_implementation': getattr(model.config, '_attn_implementation', None),
        'hidden_size': hidden,
        'n_decoder_layers': len(layers),
        'layers': [int(layer)],
        'doses': [0.0],
        'seeds': [],
        'n_directions': 0,
        'n_rows_per_batch': batch_rows,
        'n_prompts': len(load_prompts()),
        'n_items': len(item_specs),
        'openings': list(OPENINGS),
        'zero_rows_per_item': int(batch_rows),
        'max_new_tokens': int(args.max_new),
        'tokenizer': tok_report,
        'medians': {},
        'gate': {
            'zero_push': True,
            'rows_per_batch': int(batch_rows),
            'vec_abs_max': float(vecs_cpu.abs().max()),
            'no_nonzero_push': True,
            'probe_ran': False,
        },
        'rows_planned': planned_rows('gate', (layer,), (0.0,), len(item_specs)),
        'batches_planned': len(item_specs),
        'rows_written': 0,
        'batches_written': 0,
        'batches_resumed': 0,
        'status': 'running',
        'null': {'skipped': 'the gate pass records zero-push rows only'},
        'push_check': {},
        'started_at': started_at,
        'finished_at': None,
        'wall_seconds': None,
    }

    rows, bad_lines = read_answers(answers_path)
    line_count = (sum(1 for _ in answers_path.open(encoding='utf-8'))
                  if answers_path.exists() else 0)
    if bad_lines or len(rows) != line_count:
        write_answers(answers_path, rows)
        print(f'compacted answers.jsonl: {len(rows)} rows, '
              f'{bad_lines} bad line(s) dropped', flush=True)
    done = completed_batches(rows, required_ids)
    print(f'{len(rows)} rows on disk, {len(done)} batch(es) complete',
          flush=True)

    stopped = False
    batch_number = 0
    batches_written = 0
    batches_resumed = 0
    pushed_written = 0
    probe: dict = {}
    push_started = time.time()
    vecs = vecs_cpu.to(torch.device(args.device))

    with answers_path.open('a', encoding='utf-8') as sink:
        for spec, ids in zip(item_specs, item_ids):
            batch_number += 1
            key = (model_key, int(layer), 0.0, int(spec['item_index']))
            if key in done:
                batches_resumed += 1
                print(f'gate {model_key} {batch_number}/{len(item_specs)} '
                      f'i{spec["item_index"]} resumed', flush=True)
                continue
            batch = ids.repeat(batch_rows, 1).to(torch.device(args.device))
            use_probe = None if probe else {}
            handle, state = make_push_hook(layers[int(layer)], vecs, use_probe)
            try:
                sequences, _scores0 = generate_batch(model, batch,
                                                     args.max_new, pad_id)
            finally:
                handle.remove()
            if use_probe is not None:
                probe = use_probe
                meta['push_check'] = probe
                meta['gate']['hook_calls'] = int(state['calls'])
                meta['gate']['probe_ran'] = True
                meta['gate']['pushed_delta_abs_max'] = float(
                    probe['pushed_delta_abs_max'])
                meta['gate']['no_nonzero_push'] = bool(
                    probe['pushed_delta_abs_max'] == 0.0
                    and probe['zero_delta_abs_max'] == 0.0)
                if not meta['gate']['no_nonzero_push']:
                    raise SystemExit(
                        'gate pass added a nonzero push; refusing to continue')
            prompt_len = int(ids.numel())
            for row_index in range(batch_rows):
                parsed = parse_row(tok, sequences[row_index], prompt_len,
                                   args.max_new, pad_id, eos_id=tok.eos_token_id)
                record = {
                    'mode': 'gate',
                    'model': model_key,
                    'repo': repo,
                    'revision': revision,
                    'layer': int(layer),
                    'dose': 0.0,
                    'seed': None,
                    'sign': 0,
                    'direction_id': 'zero',
                    'row_id': required_ids[row_index],
                    'prompt_index': int(spec['prompt_index']),
                    'opening_index': int(spec['opening_index']),
                    'opening': spec['opening'],
                    'item_index': int(spec['item_index']),
                    'row_index': int(row_index),
                    'token_ids': parsed['token_ids'],
                    'continuation': parsed['continuation'],
                    'text': answer_text(spec['opening'], parsed['continuation']),
                    'n_new_tokens': parsed['n_new_tokens'],
                    'max_new_tokens': parsed['max_new_tokens'],
                    'hit_cap': parsed['hit_cap'],
                }
                sink.write(json.dumps(record, ensure_ascii=False) + '\n')
                pushed_written += 1
            sink.flush()
            batches_written += 1
            meta['rows_written'] = len(rows) + pushed_written
            meta['batches_written'] = batches_written
            if batches_written % 10 == 0:
                write_json(run_json_path, meta)
            elapsed = time.time() - push_started
            rate = pushed_written / elapsed if elapsed > 0 else 0.0
            print(f'gate {model_key} {batch_number}/{len(item_specs)} '
                  f'i{spec["item_index"]} '
                  f'rows {meta["rows_written"]}/{meta["rows_planned"]} '
                  f'{elapsed:.0f}s {rate:.1f} rows/s', flush=True)

    hooks_left = count_layer_hooks(layers)
    meta['hooks_after_generation'] = hooks_left
    if hooks_left:
        stopped = True
        meta['status'] = 'stopped_hooks_left'
        print(f'{hooks_left} hook(s) still attached after generation', flush=True)

    rows_final, bad_final = read_answers(answers_path)
    lines_final = (sum(1 for _ in answers_path.open(encoding='utf-8'))
                   if answers_path.exists() else 0)
    if bad_final or len(rows_final) != lines_final:
        write_answers(answers_path, rows_final)
    meta['rows_written'] = len(rows_final)
    meta['batches_written'] = batches_written
    meta['batches_resumed'] = batches_resumed
    meta['batches_complete'] = len(completed_batches(rows_final, required_ids))
    if meta['batches_complete'] != meta['batches_planned']:
        stopped = True
        if meta['status'] == 'running':
            meta['status'] = 'incomplete'
        meta['incomplete_reason'] = (
            f"{meta['batches_complete']} of {meta['batches_planned']} batches "
            f"complete after the loop")
    meta['finished_at'] = time.time()
    meta['wall_seconds'] = meta['finished_at'] - started_at
    if not stopped and meta['status'] == 'running':
        meta['status'] = 'complete'
    write_json(run_json_path, meta)
    print(f'gate {model_key}: {meta["status"]}; rows '
          f'{meta["rows_written"]}/{meta["rows_planned"]}; zero push '
          f'{meta["gate"]["no_nonzero_push"]}', flush=True)
    return 0 if meta['status'] == 'complete' else 3


# ---------------------------------------------------------------------------
# The pushed run (pilot and full; pilot and main share the 41-row batch)
# ---------------------------------------------------------------------------

def run_pushed_pass(args, mode: str, model_key: str, run_dir: Path, tok, model,
                    layers, hidden: int, item_specs: list[dict],
                    item_ids: list[torch.Tensor], layers_wanted, doses, seeds,
                    pad_id: int, repo: str, revision: str, tiny: bool,
                    pin_table: dict, started_at: float) -> int:
    answers_path = run_dir / 'answers.jsonl'
    run_json_path = run_dir / 'run.json'
    out_dir = run_dir.parent

    rows_per_batch = n_rows(seeds)
    if rows_per_batch != ROWS_PER_BATCH:
        raise SystemExit('Amendment 1 pins the batch at 41 rows; '
                         f'got {rows_per_batch}')
    specs = direction_specs(seeds)
    required_ids = [spec['direction_id'] for spec in specs] + ['zero']

    sample_ids = item_ids[0]
    tok_report = tokenizer_report(tok, pad_id)
    tok_report['sample_prompt_tokens'] = int(sample_ids.numel())
    tok_report['sample_prompt_decodes'] = bool(
        tok.decode(sample_ids, skip_special_tokens=True).strip())

    directions, directions_sha, directions_created = load_or_create_directions(
        out_dir / 'directions.pt', hidden)
    print(f'directions {tuple(directions.shape)} sha256 {directions_sha[:16]} '
          f'({"created" if directions_created else "reused"})', flush=True)

    print(f'median residual norms: {len(item_ids)} item prompts x '
          f'{len(layers_wanted)} layer(s)', flush=True)
    medians = median_norms(model, layers, item_ids, layers_wanted,
                           torch.device(args.device))
    print(f'medians {medians}', flush=True)

    zero_rows_per_item = len(layers_wanted) * len(doses)
    meta = {
        'part': 218,
        'mode': mode,
        'model_key': model_key,
        'repo': repo,
        'revision': revision,
        'tiny': tiny,
        'pins': pin_table,
        'versions': version_report(),
        'device': str(args.device),
        'device_name': (torch.cuda.get_device_name(0)
                        if torch.device(args.device).type == 'cuda' else None),
        'dtype': args.dtype,
        'attn_implementation': getattr(model.config, '_attn_implementation', None),
        'hidden_size': hidden,
        'n_decoder_layers': len(layers),
        'layers': [int(layer) for layer in layers_wanted],
        'doses': [float(d) for d in doses],
        'seeds': [int(seed) for seed in seeds],
        'n_directions': len(specs),
        'n_rows_per_batch': rows_per_batch,
        'n_prompts': N_PROMPTS,
        'n_items': len(item_specs),
        'openings': list(OPENINGS),
        'zero_rows_per_item': int(zero_rows_per_item),
        'max_new_tokens': int(args.max_new),
        'tokenizer': tok_report,
        'directions': {
            'path': str((out_dir / 'directions.pt').resolve()),
            'sha256': directions_sha,
            'shape': list(directions.shape),
            'created_this_run': bool(directions_created),
        },
        'medians': medians,
        'rows_planned': planned_rows(mode, layers_wanted, doses,
                                     len(item_specs)),
        'batches_planned': len(layers_wanted) * len(doses) * len(item_specs),
        'rows_written': 0,
        'batches_written': 0,
        'batches_resumed': 0,
        'status': 'running',
        'null': {},
        'push_check': {},
        'started_at': started_at,
        'finished_at': None,
        'wall_seconds': None,
    }

    # --- resume ---------------------------------------------------------
    rows, bad_lines = read_answers(answers_path)
    line_count = (sum(1 for _ in answers_path.open(encoding='utf-8'))
                  if answers_path.exists() else 0)
    if bad_lines or len(rows) != line_count:
        write_answers(answers_path, rows)
        print(f'compacted answers.jsonl: {len(rows)} rows, '
              f'{bad_lines} bad line(s) dropped', flush=True)
    done = completed_batches(rows, required_ids)
    print(f'{len(rows)} rows on disk, {len(done)} batch(es) complete',
          flush=True)

    previous_null = None
    previous_push = None
    if run_json_path.exists():
        try:
            previous = json.loads(run_json_path.read_text(encoding='utf-8'))
            candidate = previous.get('null', {}).get('hooked_zero_vs_start', {})
            if candidate.get('comparisons'):
                previous_null = candidate
            if previous.get('push_check'):
                previous_push = previous['push_check']
        except (json.JSONDecodeError, OSError):
            previous_null = None
            previous_push = None

    # --- no-hook null at the start -------------------------------------
    print('no-hook null at the start', flush=True)
    null_start = null_pass(model, item_ids, rows_per_batch,
                           torch.device(args.device), args.max_new, pad_id,
                           'start')
    meta['null']['start'] = null_summary(null_start, len(item_ids))
    meta['null']['start']['unhooked'] = True
    write_json(run_json_path, meta)

    hooked_zero_vs_start: list[dict] = []
    last_zero: dict[int, dict] = {}
    probe: dict = {}
    pushed_written = 0
    batches_written = 0
    batches_resumed = 0
    stopped = False
    push_started = time.time()
    total_batches = meta['batches_planned']
    batch_number = 0

    with answers_path.open('a', encoding='utf-8') as sink:
        for layer in layers_wanted:
            for dose in doses:
                vecs_cpu = dose_vectors(directions, seeds, dose,
                                        medians[str(layer)])
                if not torch.isfinite(vecs_cpu).all():
                    raise SystemExit('non-finite dose vector')
                vecs = vecs_cpu.to(torch.device(args.device))
                for spec, ids in zip(item_specs, item_ids):
                    batch_number += 1
                    key = (model_key, int(layer), float(dose),
                           int(spec['item_index']))
                    if key in done:
                        batches_resumed += 1
                        print(f'{model_key} {batch_number}/{total_batches} '
                              f'L{layer} d{dose:g} i{spec["item_index"]} resumed',
                              flush=True)
                        continue
                    batch = ids.repeat(rows_per_batch, 1).to(torch.device(args.device))
                    use_probe = None if probe else {}
                    handle, _state = make_push_hook(
                        layers[int(layer)], vecs, use_probe)
                    try:
                        sequences, scores0 = generate_batch(
                            model, batch, args.max_new, pad_id)
                    finally:
                        handle.remove()
                    if use_probe:
                        probe = use_probe
                        meta['push_check'] = probe
                    prompt_len = int(ids.numel())
                    for row_index, row_spec in enumerate(specs + [ZERO_ROW]):
                        parsed = parse_row(tok, sequences[row_index], prompt_len,
                                           args.max_new, pad_id,
                                           eos_id=tok.eos_token_id)
                        record = {
                            'mode': mode,
                            'model': model_key,
                            'repo': repo,
                            'revision': revision,
                            'layer': int(layer),
                            'dose': float(dose),
                            'seed': row_spec['seed'],
                            'sign': row_spec['sign'],
                            'direction_id': row_spec['direction_id'],
                            'row_id': row_spec['direction_id'],
                            'prompt_index': int(spec['prompt_index']),
                            'opening_index': int(spec['opening_index']),
                            'opening': spec['opening'],
                            'item_index': int(spec['item_index']),
                            'row_index': int(row_index),
                            'token_ids': parsed['token_ids'],
                            'continuation': parsed['continuation'],
                            'text': answer_text(spec['opening'],
                                                parsed['continuation']),
                            'n_new_tokens': parsed['n_new_tokens'],
                            'max_new_tokens': parsed['max_new_tokens'],
                            'hit_cap': parsed['hit_cap'],
                        }
                        sink.write(json.dumps(record, ensure_ascii=False) + '\n')
                        pushed_written += 1
                    sink.flush()

                    zero = parse_row(tok, sequences[-1], prompt_len,
                                     args.max_new, pad_id, eos_id=tok.eos_token_id)
                    zero_logits = last_row_logits(scores0)
                    item_index = int(spec['item_index'])
                    comparison = compare_null(null_start[item_index],
                                              zero['token_ids'], zero_logits)
                    comparison['item_index'] = item_index
                    comparison['prompt_index'] = int(spec['prompt_index'])
                    comparison['opening_index'] = int(spec['opening_index'])
                    comparison['layer'] = int(layer)
                    comparison['dose'] = float(dose)
                    hooked_zero_vs_start.append(comparison)
                    last_zero[item_index] = {
                        'token_ids': zero['token_ids'],
                        'logits': zero_logits,
                    }
                    batches_written += 1
                    meta['rows_written'] = len(rows) + pushed_written
                    meta['batches_written'] = batches_written
                    if batches_written % 10 == 0:
                        write_json(run_json_path, meta)
                    elapsed = time.time() - push_started
                    rate = pushed_written / elapsed if elapsed > 0 else 0.0
                    print(f'{model_key} {batch_number}/{total_batches} '
                          f'L{layer} d{dose:g} i{item_index} '
                          f'rows {meta["rows_written"]}/{meta["rows_planned"]} '
                          f'{elapsed:.0f}s {rate:.1f} rows/s', flush=True)
                    if not comparison['ids_ok'] or not comparison['logits_ok']:
                        stopped = True
                        meta['status'] = 'stopped_null_mismatch'
                        print(f'NO-HOOK NULL MISMATCH at L{layer} d{dose:g} '
                              f'i{item_index}: {comparison}', flush=True)
                        break
                if stopped:
                    break
            if stopped:
                break

    # --- reset receipt --------------------------------------------------
    hooks_left = count_layer_hooks(layers)
    meta['hooks_after_generation'] = hooks_left
    if hooks_left:
        stopped = True
        meta['status'] = 'stopped_hooks_left'
        print(f'{hooks_left} hook(s) still attached after generation', flush=True)

    print('no-hook null at the end', flush=True)
    null_end = null_pass(model, item_ids, rows_per_batch,
                         torch.device(args.device), args.max_new, pad_id, 'end')

    # End null against the start null (model determinism) ...
    end_vs_start = [compare_null(null_start[i], null_end[i]['token_ids'],
                                 null_end[i]['logits'])
                    for i in range(len(null_end))]
    # ... and against the hooked zero row (the control the design names).
    end_vs_hooked = []
    no_logits = 0
    for result in null_end:
        item_index = result['item_index']
        if item_index in last_zero:
            reference = last_zero[item_index]
            end_vs_hooked.append(compare_null(
                reference, result['token_ids'], result['logits']))
        else:
            stored_zero = [r for r in rows.values()
                           if r['model'] == model_key
                           and int(r['item_index']) == item_index
                           and r['direction_id'] == 'zero']
            if stored_zero:
                reference = {'token_ids': stored_zero[-1]['token_ids'],
                             'logits': result['logits']}
                comparison = compare_null(reference, result['token_ids'],
                                          result['logits'])
                comparison['logits_ok'] = None       # cannot be recovered
                comparison['max_abs_logit_diff'] = None
                comparison['ids_only'] = True
                no_logits += 1
                end_vs_hooked.append(comparison)
    meta['null']['end'] = null_summary(null_end, len(item_ids))
    meta['null']['end']['vs_start'] = merge_comparisons(end_vs_start)
    meta['null']['end']['vs_hooked_zero'] = merge_comparisons(end_vs_hooked)
    meta['null']['end']['vs_hooked_zero']['logits_unavailable_items'] = no_logits
    meta['null']['hooked_zero_vs_start'] = merge_comparisons(
        hooked_zero_vs_start)
    if previous_null is not None:
        current = meta['null']['hooked_zero_vs_start']
        diffs = [d for d in (current.get('max_abs_logit_diff'),
                             previous_null.get('max_abs_logit_diff'))
                 if d is not None]
        meta['null']['hooked_zero_vs_start'] = {
            'comparisons': previous_null.get('comparisons', 0)
                           + current['comparisons'],
            'ids_matched': previous_null.get('ids_matched', 0)
                           + current['ids_matched'],
            'logits_matched': previous_null.get('logits_matched', 0)
                              + current['logits_matched'],
            # A process that generated no batch compared nothing; that is vacuous,
            # not a mismatch (GLM review HIGH-1: a rerun of a finished command
            # reported stopped_null_mismatch and failed 218.5 on a clean run).
            'all_ok': bool(previous_null.get('all_ok')) and (
                current['all_ok'] or current['comparisons'] == 0),
            'max_abs_logit_diff': max(diffs) if diffs else None,
            'previous_run': previous_null,
            'this_process': current,
        }

    hooked_vs_start_ok = meta['null']['hooked_zero_vs_start']['all_ok']
    if (not hooked_zero_vs_start and previous_null is not None
            and hooked_vs_start_ok):
        meta['null']['hooked_zero_vs_start']['reason'] = (
            'no batch generated in this process; the previous run.json record '
            'covers the rows already on disk')
    null_ok = hooked_vs_start_ok and meta['null']['end']['vs_start']['all_ok']
    hooked_end = meta['null']['end']['vs_hooked_zero']
    if hooked_end['comparisons']:
        null_ok = null_ok and (hooked_end['ids_matched']
                               == hooked_end['comparisons'])
        if hooked_end['comparisons'] != len(null_end):
            null_ok = False
            hooked_end['reason'] = (
                f"only {hooked_end['comparisons']} of {len(null_end)} items "
                f"had a hooked zero row to compare the end null with")
        elif hooked_end['logits_unavailable_items'] == 0:
            null_ok = null_ok and hooked_end['all_ok']
    else:
        null_ok = False
        meta['null']['end']['vs_hooked_zero']['reason'] = (
            'no hooked zero row available to compare the end null with')
    meta['null']['all_ok'] = bool(null_ok)
    if not null_ok:
        stopped = True
        if meta['status'] == 'running':
            meta['status'] = 'stopped_null_mismatch'

    # Count unique rows, and compact any duplicate left by a crash inside a
    # batch (the batch is regenerated, so the later record is the real one).
    rows_final, bad_final = read_answers(answers_path)
    lines_final = (sum(1 for _ in answers_path.open(encoding='utf-8'))
                   if answers_path.exists() else 0)
    if bad_final or len(rows_final) != lines_final:
        write_answers(answers_path, rows_final)
    meta['rows_written'] = len(rows_final)
    if not meta.get('push_check') and previous_push:
        meta['push_check'] = previous_push
        meta['push_check']['from_previous_run'] = True
    meta['batches_written'] = batches_written
    meta['batches_resumed'] = batches_resumed
    meta['batches_complete'] = len(completed_batches(rows_final, required_ids))
    if meta['batches_complete'] != meta['batches_planned']:
        stopped = True
        if meta['status'] == 'running':
            meta['status'] = 'incomplete'
        meta['incomplete_reason'] = (
            f"{meta['batches_complete']} of {meta['batches_planned']} batches "
            f"complete after the loop")
    meta['finished_at'] = time.time()
    meta['wall_seconds'] = meta['finished_at'] - started_at
    if not stopped and meta['status'] == 'running':
        meta['status'] = 'complete'
    write_json(run_json_path, meta)

    print(f'{model_key}: {meta["status"]}; rows {meta["rows_written"]}/'
          f'{meta["rows_planned"]}; batches written {batches_written}, '
          f'resumed {batches_resumed}; null all_ok={meta["null"]["all_ok"]}',
          flush=True)
    return 0 if meta['status'] == 'complete' else 3


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run(args) -> int:
    mode = args.mode
    if mode == 'pilot':
        model_key = 'Instruct'
        layers_wanted = (PILOT_LAYER,)
        doses = tuple(PILOT_DOSES)
        seeds = PILOT_SEEDS
        run_name = 'pilot'
        if args.model and args.model != 'Instruct':
            raise SystemExit('the pilot is Instruct only (Amendment 1 item 8)')
        if args.doses:
            raise SystemExit('the pilot ladder is fixed (Amendment 1 item 8)')
    elif mode == 'gate':
        if args.model not in GATE_MODELS:
            raise SystemExit('--mode gate needs --model SFT or --model DPO '
                             '(Amendment 1 item 8)')
        model_key = args.model
        layers_wanted = (GATE_LAYER,)
        doses = (0.0,)
        seeds = ()
        run_name = None
        if args.doses:
            raise SystemExit('the gate pass has no dose ladder')
    else:
        if not args.model:
            raise SystemExit('--mode full needs --model {SFT,DPO,Instruct}')
        model_key = args.model
        layers_wanted = tuple(FULL_LAYERS)
        doses = tuple(args.doses) if args.doses else tuple(LADDER)
        seeds = FULL_SEEDS
        run_name = model_key

    prompts = load_prompts()
    out_dir = Path(args.out)
    run_dir = (out_dir / 'gate' / model_key if mode == 'gate'
               else out_dir / run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
             'float16': torch.float16}[args.dtype]
    pin_table = pins()
    if args.tiny:
        repo, revision = str(Path(args.tiny).resolve()), 'tiny-random'
        print(f'loading a tiny random model from {repo}', flush=True)
        tok, model = load_tiny(repo, device, dtype)
        tiny = True
    else:
        repo, revision = pin_table[model_key]['repo'], pin_table[model_key]['revision']
        print(f'loading {repo}@{revision[:12]} on {device} ({args.dtype})',
              flush=True)
        tok, model = load_model(model_key, device, dtype,
                                args.attn_implementation)
        tiny = False

    layers = find_decoder_layers(model)
    if len(layers) <= max(layers_wanted):
        raise SystemExit(f'model has {len(layers)} decoder layers; '
                         f'layers {layers_wanted} do not all exist')
    hidden = int(model.get_input_embeddings().weight.shape[1])

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is None:
        raise SystemExit('tokenizer has neither pad_token_id nor eos_token_id')

    item_specs = items(len(prompts))
    item_ids = [encode_item(tok, prompts[spec['prompt_index']],
                            spec['opening']) for spec in item_specs]
    started_at = time.time()

    if mode == 'gate':
        return run_gate_pass(args, model_key, run_dir, tok, model, layers,
                             hidden, item_specs, item_ids, pad_id, repo,
                             revision, tiny, pin_table, started_at)
    return run_pushed_pass(args, mode, model_key, run_dir, tok, model, layers,
                           hidden, item_specs, item_ids, layers_wanted, doses,
                           seeds, pad_id, repo, revision, tiny, pin_table,
                           started_at)


# ---------------------------------------------------------------------------
# Self-test: tiny random model, hook arithmetic and planned counts
# ---------------------------------------------------------------------------

def selftest(args) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config_dir = args.tiny
    revision = None
    local_only = False
    if config_dir is None:
        repo, revision = MODELS['Instruct']
        local_only = True
        try:
            AutoTokenizer.from_pretrained(repo, revision=revision,
                                          local_files_only=True)
        except Exception as exc:                    # noqa: BLE001 - CLI message
            raise SystemExit(
                'the self-test needs the real Instruct tokenizer; pass '
                f'--tiny <config-dir> (cache lookup failed: {exc})')
        config_dir = repo
    config = tiny_config(config_dir, hidden=64, n_layers=2, intermediate=128,
                         heads=4, revision=revision,
                         local_files_only=local_only)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config)
    model.to(device='cpu', dtype=torch.float32).eval()
    torch.set_grad_enabled(False)
    tok_kwargs = {'local_files_only': local_only}
    if revision:
        tok_kwargs['revision'] = revision
    tok = AutoTokenizer.from_pretrained(str(config_dir), **tok_kwargs)
    layers = find_decoder_layers(model)
    hidden = int(model.get_input_embeddings().weight.shape[1])

    # Pure checks first.
    directions = directions_tensor(hidden)
    assert directions.shape == (N_DRAWS, hidden)
    assert torch.allclose(directions.norm(dim=1), torch.ones(N_DRAWS),
                          atol=1e-5), 'directions are unit norm'
    assert torch.equal(directions[0], -(-directions[0]))
    seeds = PILOT_SEEDS
    specs = direction_specs(seeds)
    assert len(specs) == 40 and specs[0]['direction_id'] == 'seed0+'
    assert specs[1]['direction_id'] == 'seed0-'
    assert n_rows(seeds) == ROWS_PER_BATCH == 41
    assert len(items()) == N_ITEMS == 88
    assert [s['item_index'] for s in items()] == list(range(88))
    assert items()[1]['opening'] == OPENINGS[1]
    assert items()[2] == {'item_index': 2, 'prompt_index': 1,
                          'opening_index': 0, 'opening': OPENINGS[0]}
    assert answer_text(OPENINGS[0], ' Ai2.') == 'I was developed by Ai2.'
    assert planned_rows('pilot', (PILOT_LAYER,), PILOT_DOSES) == 32472
    assert planned_rows('full', FULL_LAYERS, LADDER) == 50512
    assert planned_rows('gate', (GATE_LAYER,), (0.0,)) == 3608
    assert float(gate_vectors(hidden).abs().max()) == 0.0

    # Hook arithmetic on a real forward pass.
    ids = encode_item(tok, 'Who built you?', OPENINGS[0])
    batch = ids.repeat(3, 1)
    vecs = dose_vectors(directions, (0,), dose=50.0,
                        median_norm=1.0)          # rows 0,1 pushed; row 2 zero
    plain = model(input_ids=batch).logits
    probe = {}
    handle, _ = make_push_hook(layers[1], vecs, probe)
    try:
        pushed = model(input_ids=batch).logits
    finally:
        handle.remove()
    assert probe['zero_row_exact'], 'zero row must add exactly zero'
    assert probe['zero_delta_abs_max'] == 0.0
    assert probe['pushed_delta_abs_max'] > 0.0, 'pushed rows must be nonzero'
    assert torch.equal(plain[-1], pushed[-1]), \
        'the zero row must not change the forward pass'
    assert not torch.equal(plain[0], pushed[0]), \
        'a large push must change a pushed row'
    assert count_layer_hooks(layers) == 0, 'the hook must be removed'
    print('selftest ok: items, directions, doses, hooks, planned counts')
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_doses(text: str) -> tuple[float, ...]:
    doses = []
    for part in text.replace(' ', '').split(','):
        if not part:
            continue
        dose = float(part)
        if dose <= 0:
            raise argparse.ArgumentTypeError('--doses needs positive numbers')
        if dose not in doses:
            doses.append(dose)
    if not doses:
        raise argparse.ArgumentTypeError('--doses needs positive numbers')
    return tuple(doses)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--mode', choices=('pilot', 'gate', 'full'))
    parser.add_argument('--model', choices=tuple(MODELS))
    parser.add_argument('--out', default='results218',
                        help='root for directions.pt and <run>/answers.jsonl')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    parser.add_argument('--dtype', choices=('float32', 'bfloat16', 'float16'),
                        default='bfloat16' if torch.cuda.is_available()
                        else 'float32')
    parser.add_argument('--max-new', type=int, default=MAX_NEW)
    parser.add_argument('--doses', type=parse_doses,
                        help='comma-separated d list for a trimmed full run')
    parser.add_argument('--tiny', default=None,
                        help='a config+tokenizer dir: run a random tiny model')
    parser.add_argument('--attn-implementation', default=None)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest(args)
    if not args.mode:
        parser.error('--mode is required (or use --selftest)')
    if args.max_new < 1:
        parser.error('--max-new must be at least 1')
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
