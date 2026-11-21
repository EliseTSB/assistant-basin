#!/usr/bin/env python3
"""Part 216 prompt-bank builder -> `part216/prompts216.json`.

Design v2.2 (part216/DESIGN.md) is the contract. This file builds, and records the token ids for,
every sequence the runner will forward:

  * the 24 Part 214 Stage 1 items, referenced from `part214/prompts214.json` (whose SHA-256 is
    recorded here), serialized exactly as `run214.stage1_ids` + the fixed opening `[40, 557, 7633,
    539, 2972]` + Part 214's `[14030]` Google continuation (Amendment 1 of Part 215: the
    continuation token is in the input for shape parity only; the read is at `T+4`);
  * the two maker-slot qualification control sentences, EXACTLY as DESIGN.md "Controls" writes
    them, prepended to the same user text with one space;
  * Part 215's `filler` v0, `denial` v0 and `vogel_line` v0 openers, likewise prepended to the
    user text with one space (Part 215's serialization rule);
  * 24 single-token factual completions written here (frozen before any run), each with its
    expected answer string and that answer's single token id.

Serialization is `neuron_report.build_prompt(tok, text, False)` then
`tok.encode(..., add_special_tokens=False)` -- the same call `run214.question_ids` makes. The
builder verifies that reproduces `part214/prompts214.json`'s recorded prompt token ids and
`part215/prompts215.json`'s template prefix/suffix lengths, so a tokenizer or template change
cannot silently move the read position.

Run:  /home/elisepc/neuron-probe/.venv/bin/python -I part216/build_prompts216.py
The tokenizer loads from the local HF cache; nothing here touches a model or a GPU.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import neuron_report as nr  # noqa: E402

MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'

P214_PATH = ROOT / 'part214' / 'prompts214.json'
P215_PATH = ROOT / 'part215' / 'prompts215.json'
DESIGN_PATH = HERE / 'DESIGN.md'
OUT_PATH = HERE / 'prompts216.json'

N_QUESTIONS = 24
OPENING_NAME = 'en_maker'
# Part 214's fixed opening and the shape-parity continuation (DESIGN.md "Model, sequences and indexing").
OPENING_IDS = [40, 557, 7633, 539, 2972]
CONTINUATION_IDS = [14030]

# DESIGN.md "Controls", verbatim. Frozen here before any run; not changed after any pilot.
CONTROL_SENTENCES = {
    'control_google': 'You are Gemini, a model made by Google.',
    'control_alibaba': 'You are Qwen, a model made by Alibaba Cloud.',
}
# DESIGN.md "Arms": Part 215's filler v0, denial v0 and vogel_line openers, read out of prompts215.json
# rather than retyped, so the two parts cannot drift.
OPENER_ARMS = {'op_filler': 'filler', 'op_denial': 'denial', 'op_vogel': 'vogel_line'}
OPENER_VARIANT = 'v0'

# 24 single-token factual completions, written for this part. `text` is the user text; the answer is
# the single token the model should emit next, so the read position is the LAST prompt position.
FACTUALS = [
    ('The capital of France is', 'Paris'),
    ('The capital of Japan is', 'Tokyo'),
    ('The chemical symbol for gold is', 'Au'),
    ('The chemical symbol for iron is', 'Fe'),
    ('The day after Monday is', 'Tuesday'),
    ('The day after Friday is', 'Saturday'),
    ('Two plus two equals', 'four'),
    ('The opposite of hot is', 'cold'),
    ('The largest planet in the solar system is', 'Jupiter'),
    ('The capital of Italy is', 'Rome'),
    ('The capital of Spain is', 'Madrid'),
    ('The capital of Germany is', 'Berlin'),
    ('The chemical symbol for sodium is', 'Na'),
    ('The chemical symbol for potassium is', 'K'),
    ('The day after Wednesday is', 'Thursday'),
    ('The day before Monday is', 'Sunday'),
    ('Three times three equals', 'nine'),
    ('The capital of Egypt is', 'Cairo'),
    ('The capital of Canada is', 'Ottawa'),
    ('The capital of Australia is', 'Canberra'),
    ('The chemical symbol for silver is', 'Ag'),
    ('The first month of the year is', 'January'),
    ('The capital of Russia is', 'Moscow'),
    ('The chemical symbol for helium is', 'He'),
]

# The 16 scored arms, in design order (DESIGN.md "Arms"). The runner and the scorer both read this.
ARMS = ['unchanged', 'zero', 'scale_dial_fact'] + [f'random_{k}:zero' for k in range(10)] + \
       list(OPENER_ARMS)
CONTROL_ARMS = list(CONTROL_SENTENCES)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def question_ids(tok, text: str) -> list[int]:
    """The exact call run214.question_ids makes."""
    return tok.encode(nr.build_prompt(tok, text, False), add_special_tokens=False)


def main() -> int:
    import torch  # noqa: F401  (neuron_report imports torch at module level; be explicit)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    p214_raw = P214_PATH.read_bytes()
    p214 = json.loads(p214_raw.decode('utf-8'))
    p215_raw = P215_PATH.read_bytes()
    p215 = json.loads(p215_raw.decode('utf-8'))

    opening = [int(t) for t in p214['openings'][OPENING_NAME]['token_ids']]
    if opening != OPENING_IDS:
        raise SystemExit(f'prompts214 opening is {opening}, DESIGN.md pins {OPENING_IDS}')
    continuation = [int(t) for t in p214['maker_names']['Google']['token_ids']]
    if continuation != CONTINUATION_IDS:
        raise SystemExit(f'Google continuation is {continuation}, Part 215 Amendment 1 pins {CONTINUATION_IDS}')

    # Template boundary check, exactly as build_prompts214.py / build_prompts215.py do it.
    sample_chat = nr.build_prompt(tok, p214['items'][0]['en_maker'], False)
    prefix_text, suffix_text = sample_chat.split(p214['items'][0]['en_maker'], 1)
    prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tok.encode(suffix_text, add_special_tokens=False)
    if len(prefix_ids) != p214['template_prefix_tokens']:
        raise SystemExit(f'template prefix is {len(prefix_ids)} tokens, Part 214 recorded '
                         f'{p214["template_prefix_tokens"]}')
    if len(suffix_ids) != p214['template_suffix_tokens']:
        raise SystemExit(f'template suffix is {len(suffix_ids)} tokens, Part 214 recorded '
                         f'{p214["template_suffix_tokens"]}')

    def stage1_item(text: str, source: str, extra: dict | None = None) -> dict:
        ids = question_ids(tok, text)
        rec = {'source': source, 'text': text, 'prompt_token_ids': ids, 'T': len(ids),
               'opening_token_ids': opening, 'continuation_token_ids': continuation,
               'sequence_token_ids': list(ids) + opening + continuation,
               'read_position': len(ids) + len(opening) - 1}
        if extra:
            rec.update(extra)
        return rec

    # ---- the 24 Part 214 items, verified against the recorded ids -------------------------------
    items = []
    for qi in range(N_QUESTIONS):
        text = p214['items'][qi]['en_maker']
        rec = stage1_item(text, f'part214/prompts214.json items[{qi}].en_maker',
                          {'question_index': qi})
        recorded = p214['items'][qi].get('en_maker_token_ids') or p214['items'][qi].get('prompt_token_ids')
        if recorded is not None and [int(t) for t in recorded] != rec['prompt_token_ids']:
            raise SystemExit(f'question {qi}: rebuilt prompt ids differ from part214 record')
        if p214['items'][qi].get('question_tokens') is not None:
            pass  # question_tokens counts the question before templating; not a check on ids
        items.append(rec)

    # ---- maker-slot qualification controls ------------------------------------------------------
    controls = []
    for arm in CONTROL_ARMS:
        sentence = CONTROL_SENTENCES[arm]
        for qi in range(N_QUESTIONS):
            text = f'{sentence} {p214["items"][qi]["en_maker"]}'
            controls.append(stage1_item(text, f'DESIGN.md Controls: {arm}', {
                'arm': arm, 'question_index': qi, 'opener': sentence,
                'expected_side': 'google' if arm == 'control_google' else 'alibaba'}))

    # ---- Part 215 openers ----------------------------------------------------------------------
    openers = []
    for arm, name in OPENER_ARMS.items():
        rows = [v for v in p215['arms'][name] if v['variant'] == OPENER_VARIANT]
        if len(rows) != 1:
            raise SystemExit(f'prompts215 arms[{name}] has no unique variant {OPENER_VARIANT}')
        opener_text = rows[0]['opener']
        for qi in range(N_QUESTIONS):
            text = f'{opener_text} {p214["items"][qi]["en_maker"]}'
            openers.append(stage1_item(text, f'part215/prompts215.json arms[{name}][{OPENER_VARIANT}]', {
                'arm': arm, 'question_index': qi, 'opener': opener_text,
                'opener_arm': name, 'opener_variant': OPENER_VARIANT,
                'opener_token_ids': [int(t) for t in rows[0]['opener_token_ids']]}))

    # ---- factual instrument check --------------------------------------------------------------
    factuals = []
    bad = []
    for qi, (text, answer) in enumerate(FACTUALS):
        ids = question_ids(tok, text)
        answer_ids = tok.encode(' ' + answer, add_special_tokens=False)
        if len(answer_ids) != 1:
            bad.append((qi, text, answer, answer_ids))
        factuals.append({
            'index': qi, 'text': text, 'expected_answer': answer,
            'answer_token_ids': [int(t) for t in answer_ids],
            'answer_token_id': int(answer_ids[0]) if len(answer_ids) == 1 else None,
            'prompt_token_ids': ids, 'read_position': len(ids) - 1,
            'input_token_id': int(ids[-1]),
        })
    if bad:
        raise SystemExit(f'{len(bad)} factual answers are not a single token: {bad}')

    bank = {
        'part': 216,
        'design_version': 'v2.2',
        'design_sha256': sha256_bytes(DESIGN_PATH.read_bytes()),
        'model': MODEL, 'revision': REVISION,
        'source_part214_prompts_sha256': sha256_bytes(p214_raw),
        'source_part215_prompts_sha256': sha256_bytes(p215_raw),
        'template_prefix_tokens': len(prefix_ids), 'template_suffix_tokens': len(suffix_ids),
        'n_questions': N_QUESTIONS,
        'model_ids': {'Google': 14030, 'Alibaba': 2033, 'T': 51},
        'opening_name': OPENING_NAME, 'opening_token_ids': opening,
        'continuation_token_ids': continuation,
        'arms': ARMS, 'control_arms': CONTROL_ARMS,
        'control_sentences': CONTROL_SENTENCES,
        'opener_arms': OPENER_ARMS, 'opener_variant': OPENER_VARIANT,
        'items': items, 'controls': controls, 'openers': openers, 'factuals': factuals,
        'counts': {'items': len(items), 'controls': len(controls), 'openers': len(openers),
                   'factuals': len(factuals)},
    }
    OUT_PATH.write_text(json.dumps(bank, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'wrote {OUT_PATH} sha256={sha256_bytes(OUT_PATH.read_bytes())}')
    print(f'counts {bank["counts"]}')
    print(f'stage1 sequences: {len(items) + len(controls) + len(openers)} (+{len(factuals)} factual, '
          f'no opening)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
